# src/auralis/models/xttsv2/components/vllm_mm_gpt.py
"""XTTSv2 GPT model for vLLM V1 Engine.

This module provides the V1-compliant implementation of the XTTS GPT decoder
that runs inside vLLM's AsyncLLM. All legacy V0 hacks have been removed:

- No ``is_logits_only_mode`` branching (hidden states are extracted via V1's
  ``extract_hidden_states`` speculative method + KV connector, see XTTSv2.py).
- No ``PositionalEmbeddingsCorrecter``, no manual conditioning scatter —
  placeholder-based multimodal merging is handled by vLLM V1.
- No ``Sampler``/``sample()`` override — V1 runs sampling outside the model.
"""
from __future__ import annotations

import functools
import random
from collections.abc import Iterable, Mapping, Sequence
from typing import ClassVar, Literal, Optional, Union

import torch
import torch.nn as nn
from transformers import GPT2Config

from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.gpt2 import GPT2Block
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import (
    make_empty_intermediate_tensors_factory,
    make_layers,
)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalInputs,
    NestedTensors,
)
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Token ID used as a placeholder for audio-conditioning positions in the
#: prompt. vLLM V1 scatters the tensor returned by :meth:`XttsGPT.embed_multimodal`
#: into these positions via :meth:`XttsGPT.get_input_embeddings`. Value ``1``
#: preserves the convention used by the legacy V0 pipeline (fake text-token id).
AUDIO_PLACEHOLDER_TOKEN_ID: int = 1

#: Placeholder string emitted by :meth:`XttsGPT.get_placeholder_str`. Auralis
#: builds prompts from token ids directly, but V1 requires the method to exist.
AUDIO_PLACEHOLDER_STR: str = "<|auralis_audio|>"


# ---------------------------------------------------------------------------
# Positional embeddings (kept for re-use by XTTSv2Engine itself)
# ---------------------------------------------------------------------------


class LearnedPositionEmbeddings(nn.Module):
    """Learned positional embeddings used by both the XTTS engine (for text
    conditioning) and the internal GPT stack (for audio generation)."""

    def __init__(
        self,
        seq_len: int,
        model_dim: int,
        init: float = 0.02,
        relative: bool = False,
        supports_pp: bool = False,
    ) -> None:
        super().__init__()
        self.emb = (
            VocabParallelEmbedding(seq_len, model_dim)
            if supports_pp
            else nn.Embedding(seq_len, model_dim)
        )
        self.emb.weight.data.normal_(mean=0.0, std=init)
        self.relative = relative
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sl = x.shape[1]
        if self.relative:
            start = random.randint(sl, self.seq_len) - sl
            indices = torch.arange(start, start + sl, device=x.device)
        else:
            indices = torch.arange(0, sl, device=x.device)
        assert (indices < self.seq_len).all() and (indices >= 0).all(), (
            f"position indices out of range: min={indices.min().item()}, "
            f"max={indices.max().item()}, limit={self.seq_len}"
        )
        return self.emb(indices)

    def get_fixed_embedding(
        self, ind: torch.Tensor, dev: torch.device
    ) -> torch.Tensor:
        assert (ind < self.seq_len).all(), (
            f"max index {ind.max().item()} exceeds {self.seq_len - 1}"
        )
        assert (ind >= 0).all(), f"negative index: {ind.min().item()}"
        if ind.shape[0] > 1:
            return self.emb(ind)
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


# ---------------------------------------------------------------------------
# Multimodal processor — passes pre-computed embeddings through to the model
# ---------------------------------------------------------------------------


class XttsProcessingInfo(BaseProcessingInfo):
    """Processing-info provider for XTTS multimodal data.

    XTTS "multimodal data" is not raw audio but a pre-merged tensor of shape
    ``[audio_cond_len + text_emb_len, hidden_size]`` computed upstream
    (``XTTSv2Engine._merge_conditioning``). The processing pipeline therefore
    becomes a thin pass-through.
    """

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": 1}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        hf_config = self.get_hf_config()
        max_text_tokens = int(getattr(hf_config, "max_text_tokens", 402))
        # 32 = perceiver-resampler output length; +2 text BOS/EOS; +1 slack.
        return {"audio": 32 + max_text_tokens + 3}


class XttsDummyInputsBuilder(BaseDummyInputsBuilder[XttsProcessingInfo]):
    """Generates dummy data for memory profiling at engine start-up."""

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        audio_count = mm_counts.get("audio", 0)
        hf_config = self.info.get_hf_config()
        hidden_size = int(hf_config.hidden_size)
        dtype = self.info.ctx.model_config.dtype

        max_audio_len = self.info.get_mm_max_tokens_per_item(
            seq_len, mm_counts
        )["audio"]
        placeholder_tokens = [AUDIO_PLACEHOLDER_TOKEN_ID] * max_audio_len
        start_token = int(hf_config.start_audio_token)
        prompt_token_ids = (placeholder_tokens + [start_token]) * max(
            audio_count, 1
        )
        prompt_token_ids = prompt_token_ids[:seq_len]

        dummy_embeds = [
            torch.zeros((max_audio_len, hidden_size), dtype=dtype)
            for _ in range(audio_count)
        ]

        return ProcessorInputs(
            prompt_text="",
            mm_data={"audio": {"embeds": dummy_embeds}},
            hf_processor_mm_kwargs={},
        )


class XttsMultiModalProcessor(
    BaseMultiModalProcessor[XttsProcessingInfo]
):
    """Pass-through multimodal processor for XTTS pre-computed embeddings.

    Contract with the caller (``XTTSv2Engine.get_generation_context``):
      * ``prompt_token_ids`` starts with ``N`` placeholder ids (value
        ``AUDIO_PLACEHOLDER_TOKEN_ID``) where ``N == embeds.shape[0]``.
      * ``multi_modal_data["audio"]["embeds"]`` is a ``torch.Tensor`` of shape
        ``[N, hidden_size]`` (pre-merged audio conditioning + text embedding).
    """

    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser()

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> MultiModalInputs:
        audio_payload = mm_data.get("audio", {}) or {}
        embeds = audio_payload.get("embeds") if isinstance(audio_payload, Mapping) else None
        if embeds is None:
            embeds_list: list[torch.Tensor] = []
        elif isinstance(embeds, torch.Tensor):
            embeds_list = [embeds]
        else:
            embeds_list = list(embeds)

        return MultiModalInputs(
            type="multimodal",
            prompt=prompt,
            prompt_token_ids=None,
            mm_kwargs=MultiModalKwargs({"audio_embeds": embeds_list}),
            mm_hashes=None,
            mm_placeholders={},
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: Mapping[str, object],
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {"audio_embeds": MultiModalFieldConfig.batched("audio")}

    def _get_prompt_updates(
        self,
        mm_items,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptReplacement]:
        # Placeholder positions are already present in prompt_token_ids as
        # contiguous runs of AUDIO_PLACEHOLDER_TOKEN_ID. No replacement needed.
        return ()


# ---------------------------------------------------------------------------
# Internal GPT stack (audio-token decoder)
# ---------------------------------------------------------------------------


class GPT2Model(nn.Module):
    """Core GPT transformer used by XttsGPT.

    Simplified V1-compatible variant of the legacy stack:
      * No manual ``_insert_conditioning_into_hidden_states`` — V1 scatters
        multimodal embeddings into ``inputs_embeds`` via
        :meth:`XttsGPT.get_input_embeddings`.
      * Forward receives already-merged ``inputs_embeds`` when multimodal
        data is present; otherwise computes text-only embeddings.
      * Attention state is managed by vLLM V1 (thread-local metadata); the
        forward signature does not expose ``kv_caches`` / ``attn_metadata``.
    """

    def __init__(
        self,
        config: GPT2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        assert not config.add_cross_attention
        assert not config.scale_attn_by_inverse_layer_idx
        assert not config.reorder_and_upcast_attn

        self.config = config
        self.embed_dim = config.hidden_size

        self.wte = VocabParallelEmbedding(
            config.num_audio_tokens, self.embed_dim
        )
        self.wpe = (
            LearnedPositionEmbeddings(
                config.max_audio_tokens + 3, config.decoder_input_dim
            )
            if config.max_audio_tokens != -1
            else functools.partial(
                config.null_position_embeddings, dim=config.decoder_input_dim
            )
        )

        self.start_layer, self.end_layer, self.h = make_layers(
            config.num_hidden_layers,
            lambda prefix: GPT2Block(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.h",
        )
        self.ln_f = nn.LayerNorm(
            self.embed_dim, eps=config.layer_norm_epsilon
        )

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states"], config.hidden_size
            )
        )

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        token_embeds = self.wte(input_ids)
        pos_embeds = self.wpe.get_fixed_embedding(
            torch.arange(input_ids.shape[-1], device=input_ids.device),
            input_ids.device,
        )
        return (token_embeds + pos_embeds).view(-1, self.embed_dim)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                assert input_ids is not None
                hidden_states = self.embed_tokens(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for i in range(self.start_layer, self.end_layer):
            hidden_states = self.h[i](hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Public model class
# ---------------------------------------------------------------------------


@MULTIMODAL_REGISTRY.register_processor(
    XttsMultiModalProcessor,
    info=XttsProcessingInfo,
    dummy_inputs=XttsDummyInputsBuilder,
)
class XttsGPT(nn.Module, SupportsMultiModal, SupportsPP):
    """V1-compliant XTTS GPT decoder.

    Flow per request:
      1. Caller (``XTTSv2Engine.get_generation_context``) prepares:
         - ``prompt_token_ids`` = ``N`` placeholders + ``start_audio_token``.
         - ``multi_modal_data["audio"]["embeds"]`` = pre-merged conditioning
           tensor of shape ``[N, hidden]``.
      2. vLLM V1 calls :meth:`embed_multimodal` to obtain the embedding
         tensor and :meth:`get_input_embeddings` to scatter it over
         placeholder positions.
      3. :class:`GPT2Model` runs the transformer; V1's registered KV
         connector captures per-layer hidden states for HiFi-GAN decoding.
      4. :meth:`compute_logits` projects the last hidden state through
         ``mel_head`` for autoregressive mel-token sampling.
    """

    supports_multimodal: ClassVar[Literal[True]] = True
    supports_multimodal_raw_input_only: ClassVar[bool] = False
    requires_raw_input_tokens: ClassVar[bool] = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.gpt_config: GPT2Config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        # Native attribute name preserved: checkpoints ship weights under
        # the "gpt.*" prefix and load directly without any remapping.
        self.gpt = GPT2Model(
            config=self.gpt_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}gpt",
        )

        self.final_norm = nn.LayerNorm(
            self.gpt_config.hidden_size,
            bias=True,
            eps=self.gpt_config.layer_norm_epsilon,
        )

        self.mel_head = ParallelLMHead(
            self.gpt_config.num_audio_tokens,
            self.gpt_config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}mel_head",
        )

        logit_scale: float = getattr(self.gpt_config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.gpt_config.num_audio_tokens,
            self.gpt_config.num_audio_tokens,
            logit_scale,
        )

        self.make_empty_intermediate_tensors = (
            self.gpt.make_empty_intermediate_tensors
        )

    # -- SupportsMultiModal protocol ---------------------------------------

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        if modality == "audio":
            return AUDIO_PLACEHOLDER_STR
        return None

    def embed_multimodal(self, **kwargs: object) -> NestedTensors:
        """Return the pre-computed conditioning embeddings as-is.

        Expected keyword: ``audio_embeds`` — list of tensors, one per audio
        item. Each tensor has shape ``[N_i, hidden_size]`` and is already
        cast to the engine dtype by ``XTTSv2Engine._merge_conditioning``.
        """
        audio_embeds = kwargs.get("audio_embeds")
        if audio_embeds is None:
            raise ValueError(
                "XttsGPT.embed_multimodal: missing 'audio_embeds' kwarg"
            )
        if isinstance(audio_embeds, torch.Tensor):
            return [audio_embeds]
        return list(audio_embeds)

    def get_language_model(self) -> nn.Module:
        return self.gpt

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.gpt.embed_tokens(input_ids)
        if multimodal_embeddings is None:
            return inputs_embeds

        placeholder_mask = input_ids == AUDIO_PLACEHOLDER_TOKEN_ID
        if not placeholder_mask.any():
            return inputs_embeds

        if isinstance(multimodal_embeddings, torch.Tensor):
            flat_mm = multimodal_embeddings.reshape(-1, inputs_embeds.shape[-1])
        else:
            flat_mm = torch.cat(
                [t.reshape(-1, inputs_embeds.shape[-1])
                 for t in multimodal_embeddings],
                dim=0,
            )

        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[placeholder_mask.view(-1)] = flat_mm.to(
            dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        return inputs_embeds

    # -- Core model forward ------------------------------------------------

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        return self.gpt(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        hidden_states = self.final_norm(hidden_states)
        logits = self.logits_processor(
            self.mel_head, hidden_states, sampling_metadata, self.mel_head.bias
        )
        return logits

    # -- Weight loading ----------------------------------------------------

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        """Native weight loader — no key remapping."""
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded: set[str] = set()

        for name, loaded_weight in weights:
            if name not in params_dict:
                continue

            # GPT-2 conv1d weights are transposed relative to nn.Linear.
            if any(tag in name for tag in ("c_attn", "c_proj", "c_fc")):
                if name.endswith(".weight"):
                    loaded_weight = loaded_weight.t()

            param = params_dict[name]
            weight_loader = getattr(
                param, "weight_loader", default_weight_loader
            )
            weight_loader(param, loaded_weight)
            loaded.add(name)

        missing = set(params_dict.keys()) - loaded
        if missing:
            raise RuntimeError(
                "XttsGPT.load_weights: missing weights for parameters: "
                f"{sorted(missing)}"
            )
        return loaded
