# Auralis — План миграции на vLLM V1 Engine

**Дата:** 2026-04-22
**Скоуп:** полный переход с vLLM `0.6.4.post1` (V0) на актуальный V1 Engine (vLLM `0.14.x`+).
**Статус:** спецификация. Код **не** меняется до прямого разрешения.
**Основание:** `audit_report.md` (Фаза 1) + Фаза 2 (ресерч context7).

## Принципы

1. **Никаких легаси-хаков.** Удаляем `ExtendedSamplingParams`, `HiddenStatesCollector`, `SyncCollectorWrapper`, ключ `is_logits_only_mode`, перехват `forward()`, per-request `logits_processors` callable.
2. **Разделение обязанностей.** Модель = `nn.Module + SupportsMultiModal`. Процессинг мультимодальных входов = `BaseMultiModalProcessor`. Извлечение hidden states = штатный V1 механизм `extract_hidden_states` + KV-connector.
3. **Одна entry-точка для vLLM.** Вся конфигурация движка — в `XTTSv2Engine.init_vllm_engine()`. Никаких прямых импортов из `vllm.model_executor.*` вне `vllm_mm_gpt.py`.
4. **Обратной совместимости по TTSRequest / TTSOutput нет изменений.** Публичный API `TTS.generate_speech*` остаётся идентичным.

---

## 1. Архитектурный рефакторинг `XttsGPT`

### 1.1. Целевая структура `src/auralis/models/xttsv2/components/vllm_mm_gpt.py`

```
vllm_mm_gpt.py
├── Импорты (V1-совместимые, см. ниже)
├── XttsProcessingInfo(BaseProcessingInfo)          # лимиты, подсчёт placeholder-токенов
├── XttsDummyInputsBuilder(BaseDummyInputsBuilder)   # dummy-данные для memory profiling
├── XttsMultiModalProcessor(BaseMultiModalProcessor) # raw audio cond → MultiModalKwargs
├── LearnedPositionEmbeddings(nn.Module)             # без изменений
└── @MULTIMODAL_REGISTRY.register_processor(...)
    class XttsGPT(nn.Module, SupportsMultiModal, SupportsPP):
        # class-level флаги
        supports_multimodal: ClassVar[Literal[True]] = True
        # __init__, load_weights, get_placeholder_str, embed_multimodal,
        # get_language_model, get_input_embeddings, forward
```

### 1.2. Новые импорты (замена текущих 17-и)

**Удалить (`vllm_mm_gpt.py` строки 15–36, из `audit_report.md` §1):**
- `AttentionMetadata`, `SamplingMetadata`, `SamplerOutput`, `Sampler` (V1 их вычищает из модельного forward'а).
- `INPUT_REGISTRY`, `InputContext`, `DecoderOnlyInputs`, `token_inputs`, `DummyData` (удалены в V1).
- `MultiModalInputs` (переименован в `MultiModalKwargs`).
- `SequenceData`, `VLLM_TOKEN_ID_ARRAY_TYPE` (V1 не использует).
- `MultiModalConfig` (перенесён внутрь `VllmConfig`).
- `consecutive_placeholder_ranges` (нативный путь через processor).

**Добавить:**
```
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.gpt2 import GPT2Block
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import (
    make_layers, make_empty_intermediate_tensors_factory,
)
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
from vllm.multimodal.inputs import MultiModalDataDict, PlaceholderRange
from vllm.multimodal.processing import (
    BaseMultiModalProcessor, BaseProcessingInfo, PromptReplacement,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils import is_list_of
```

### 1.3. Новый `XttsGPT` (эскиз сигнатур, без реализации)

```python
@MULTIMODAL_REGISTRY.register_processor(
    XttsMultiModalProcessor,
    info=XttsProcessingInfo,
    dummy_inputs=XttsDummyInputsBuilder,
)
class XttsGPT(nn.Module, SupportsMultiModal, SupportsPP):
    supports_multimodal: ClassVar[Literal[True]] = True
    supports_multimodal_raw_input_only: ClassVar[bool] = False
    requires_raw_input_tokens: ClassVar[bool] = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        # вернуть плейсхолдер для "audio" (например, спец-токен id → строка)
        ...

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """Замена старой логики get_multimodal_embeddings / куска forward().
        Принимает готовые conditioning-embeddings, возможно энкодирует, возвращает
        тензор в порядке появления audio-элементов в промпте."""
        ...

    def get_language_model(self) -> nn.Module:
        return self.transformer  # внутренний GPT-стек

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
    ) -> torch.Tensor:
        """Склеивает text embeddings и multimodal embeddings стандартным путём V1."""
        ...

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors: ...

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
```

### 1.4. Processor / Info / DummyInputs

**`XttsProcessingInfo(BaseProcessingInfo)`** — возвращает:
- `get_supported_mm_limits()` → `{"audio": 1}` (как текущий `limit_mm_per_prompt={"audio": 1}`).
- `get_mm_max_tokens_per_item(seq_len, mm_counts)` → `{"audio": <audio_cond + text_emb len>}`.
- `get_hf_config()` → XTTSGPTConfig.

**`XttsDummyInputsBuilder(BaseDummyInputsBuilder)`** — возвращает:
- `get_dummy_processor_inputs(seq_len, mm_counts)` → dummy `MultiModalDataDict` с рандомным `torch.zeros(T, hidden_size)` для профайлинга памяти.

**`XttsMultiModalProcessor(BaseMultiModalProcessor[XttsProcessingInfo])`** — отвечает за:
- `_call_hf_processor(prompt, mm_data, mm_kwargs)` → (processed tokens, `MultiModalKwargs` с ключом `audio_embeds`).
- `_get_prompt_updates(...)` → список `PromptReplacement` с плейсхолдерами типа `<|audio|>` → N повторений.
- `_get_mm_fields_config(...)` → описание, какие поля из kwargs относятся к `audio`.

### 1.5. Миграция логики старого `forward()` и `get_multimodal_embeddings`

| Старое поведение | Новое место |
|---|---|
| Энкодирование `embeds` внутри `forward()` через ветки `is_logits_only_mode` | Разделяется: merging conditioning → `embed_multimodal(**kwargs)`. Logits-only проход больше не нужен (см. §2). |
| `get_multimodal_embeddings(...)` | Переименовать и адаптировать под сигнатуру `embed_multimodal(self, **kwargs) -> MultiModalEmbeddings`. Входы — из `MultiModalKwargs`, возврат — тензор `[total_audio_tokens, hidden_size]` в порядке появления. |
| Ручная вставка conditioning в `forward()` по позиции `sequence_length` | Заменить на штатный механизм `get_input_embeddings(input_ids, multimodal_embeddings=...)` + плейсхолдер-токены (см. `PromptReplacement` в processor'е). V1 сам делает scatter по позициям плейсхолдеров. |
| `Sampler`, `SamplerOutput`, `SamplingMetadata` | **Удалить из модели полностью.** В V1 sampler выведен из forward'а в `EngineCore`. Модель возвращает hidden states / logits. |
| `compute_logits(...)` через `LogitsProcessor` (класс vLLM) | Оставить: `self.logits_processor = LogitsProcessor(vocab_size)` + метод `compute_logits(hidden_states, sampling_metadata)` — контракт V1 сохраняется. |

### 1.6. План полного удаления хака `is_logits_only_mode`

1. Удалить ключ `is_logits_only_mode` из ВСЕХ `multi_modal_data["audio"]` словарей в `XTTSv2.py` (строки `645–651`, `741–749`).
2. Удалить все ветки `if ... is_logits_only_mode` в `XttsGPT.forward()` (после её переписки).
3. `XTTSv2Engine.get_model_logits()` (`XTTSv2.py:617–687`) — **удалить метод целиком**. Hidden states теперь приходят напрямую из основного `.generate()` через KV-connector (см. §2).
4. Вызов `get_model_logits(...)` в `process_tokens_to_speech` (`XTTSv2.py:789`) — удалить. Логика заменяется чтением hidden states из `output.kv_transfer_params`.

---

## 2. Миграция Hidden States (избавление от хайджека)

### 2.1. Что удаляем

| Файл | Что | Обоснование |
|---|---|---|
| `components/vllm/hijack.py` | **Удалить файл целиком.** Классы `ExtendedSamplingParams`, `LogitsRepetitionPenalizer` больше не используются. | `SamplingParams` — msgspec Struct, наследование через `kw_only=True` несовместимо с V1 (новые обязательные поля, поля внутреннего V1 состояния). Per-request `logits_processors` callable **удалены в V1** ("Removed Features"). |
| `components/vllm/hidden_state_collector.py` | **Удалить файл целиком.** Классы `HiddenStatesCollector`, `SyncCollectorWrapper`. | Хайджек `forward()` через поле в `SamplingParams` + sync callback в V1 невозможен. |
| `components/vllm/__init__.py` | Удалить. | Папка `components/vllm/` исчезает. |
| `XTTSv2.py:33-34` | Удалить импорты `HiddenStatesCollector`, `ExtendedSamplingParams`, `LogitsRepetitionPenalizer`. | |
| `XTTSv2.py:617-687` | Удалить метод `get_model_logits` целиком. | |

### 2.2. Что добавляем

Вместо `get_model_logits` — **один** вызов `llm_engine.generate(...)` в `get_generation_context`, настроенный на выдачу hidden states через `extract_hidden_states`. Результат `RequestOutput` теперь несёт:
- `.outputs[0].token_ids` — сгенерированные mel-токены (как сейчас).
- `.kv_transfer_params["hidden_states_path"]` — путь к safetensors c per-token, per-layer hidden states для всего промпта + сгенерированных токенов.

**Конфигурация движка** (только новые поля, в дополнение к существующим `AsyncEngineArgs`):
```python
speculative_config = {
    "method": "extract_hidden_states",
    "num_speculative_tokens": 1,
    "draft_model_config": {
        "hf_config": {
            "eagle_aux_hidden_state_layer_ids": [<LAST_LAYER_ID>],  # для XTTS — последний слой GPT
        }
    },
}
kv_transfer_config = {
    "kv_connector": "ExampleHiddenStatesConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
        "shared_storage_path": "/tmp/auralis/hidden_states",  # tmpfs / SSD
    },
}
```

### 2.3. Псевдокод нового потока (уровень «имплементируй без вопросов»)

**`XTTSv2Engine.init_vllm_engine(concurrency)`** — новый конфиг:
```python
from vllm import AsyncLLMEngine, AsyncEngineArgs  # alias → v1.engine.async_llm.AsyncLLM

engine_args = AsyncEngineArgs(
    model=self.gpt_model,
    tensor_parallel_size=self.tp,
    pipeline_parallel_size=self.pp,
    dtype="auto",
    max_model_len=<как сейчас>,
    gpu_memory_utilization=mem_utils,
    trust_remote_code=True,
    enforce_eager=True,
    limit_mm_per_prompt={"audio": 1},
    max_num_seqs=max_seq_num,
    disable_log_stats=True,
    max_num_batched_tokens=<как сейчас>,
    # NEW:
    speculative_config={
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {"hf_config": {
            "eagle_aux_hidden_state_layer_ids": [self.gpt_config.num_hidden_layers - 1],
        }},
    },
    kv_transfer_config={
        "kv_connector": "ExampleHiddenStatesConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
            "shared_storage_path": str(self.hidden_states_dir),
        },
    },
)
self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
```

**`XTTSv2Engine.get_generation_context(request, ...)`** — один генеративный проход, без logits-only подрауна:
```python
# вход: tokens_list, gpt_embed_inputs, speaker_embeddings — как раньше
generators, request_ids = [], []
for seq_index, sequence in enumerate(tokens_list):
    # НОВЫЙ SamplingParams (без legacy-полей и без logits_processors=[callable])
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty,   # нативный параметр вместо ручного
        max_tokens=self.gpt_config.gpt_max_audio_tokens,
        ignore_eos=True,
        stop_token_ids=[self.mel_eos_token_id],
        detokenize=False,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )

    engine_inputs: TokensPrompt = {"prompt_token_ids": sequence}
    if gpt_embed_inputs is not None:
        engine_inputs["multi_modal_data"] = {
            "audio": {
                "embeds": gpt_embed_inputs[seq_index],
                "sequence_length": len(sequence),
                # is_logits_only_mode — УДАЛЁН
            }
        }

    request_id = f"{request.request_id}_{seq_index}"
    gen = self.llm_engine.generate(
        prompt=engine_inputs,
        sampling_params=sampling_params,
        request_id=request_id,
    )
    generators.append(gen)
    request_ids.append(request_id)

return generators, request_ids, speaker_embeddings, gpt_embed_inputs
```

**`XTTSv2Engine.process_tokens_to_speech(...)`** — без второго раунда генерации:
```python
from safetensors import safe_open

async for output in generator:
    if not output.finished:
        continue

    # 1. забираем токены (как и раньше)
    token_ids = list(output.outputs[0].token_ids)

    # 2. НОВЫЙ путь: hidden states приходят через KV-connector
    hs_path = output.kv_transfer_params["hidden_states_path"]
    with safe_open(hs_path, framework="pt", device=str(self.device)) as f:
        hidden_states = f.get_tensor("hidden_states")
        # shape: [prompt_len + generated_len, num_layers_selected, hidden_size]

    # 3. slice по тому же принципу, что и сейчас:
    #    audio_start = len(audio_conditioning)  (известно из multi_modal_data)
    audio_start = multimodal_data.shape[0]
    hs = hidden_states[audio_start:-5, -1, :]   # -1: единственный сохранённый (последний) слой
    hs = self.final_norm(hs.unsqueeze(0).to(self.device).to(self.dtype))

    # 4. опциональная очистка файла (если нужно не захламлять tmpfs)
    #    os.unlink(hs_path) — если shared_storage_path не является ephemeral

    # 5. декодирование в waveform — БЕЗ ИЗМЕНЕНИЙ
    async with self.decoder_semaphore, self.cuda_memory_manager():
        wav = (await asyncio.to_thread(
            self.hifigan_decoder, hs, g=speaker_embeddings
        )).cpu().detach().numpy().squeeze()

    yield TTSOutput(
        array=wav,
        start_time=request.start_time,
        token_length=len(token_ids),
    )
```

### 2.4. Replacement для `LogitsRepetitionPenalizer`

- Штатный `SamplingParams.repetition_penalty: float` покрывает 95% кейсов (в V1 всё ещё поддерживается).
- Если требуется **кастомная** логика (что в Auralis не требуется — формула один-в-один со стандартной), новый путь — **class-based** `LogitsProcessor` с `update_state(...)`, регистрируемый через `LogitsProcessorRegistry`. В этом плане — **не реализуем**, используем нативный `repetition_penalty=request.repetition_penalty`.

### 2.5. Удаление `ExtendedSamplingParams.request_id` / `hidden_state_collector`

Оба кастомных поля — наследие хайджека. После миграции:
- `request_id` передаётся напрямую в `.generate(request_id=...)` (как сейчас).
- `hidden_state_collector` не нужен (hidden states — через KV-connector).
- Используется только базовый `vllm.SamplingParams` — никаких подклассов.

---

## 3. Пофайловые изменения (на основе `audit_report.md`)

### 3.1. `src/auralis/models/xttsv2/XTTSv2.py`

**Удалить:**
- Строка `17`: `RequestOutput` из импорта (останется, но см. ниже — `RequestOutput` мы читаем напрямую, импорт оставляем; удалить нечего кроме неиспользуемых).
- Строки `33-34`: импорты из `components/vllm/hidden_state_collector` и `components/vllm/hijack`.
- Строка `72`: поле `self.max_gb_for_vllm_model` (переименовать в `self.max_gb_for_llm_model` — косметика; опционально).
- Строка `85`: `self.request_counter = Counter()` — не используется нигде, удалить. Импорт `vllm.utils.Counter` тоже удалить (строка `20`).
- Строка `346`: доступ к `self.llm_engine.engine.model_config.dtype` → заменить на `self.llm_engine.vllm_config.model_config.dtype` (путь V1).
- Строки `617-687`: **метод `get_model_logits` целиком**.
- Строки `785-799`: блок вызова `get_model_logits` внутри `process_tokens_to_speech` — заменить чтением safetensors (см. §2.3).
- Строка `819`: `self.llm_engine.shutdown_background_loop()` → `await self.llm_engine.shutdown()` (метод `AsyncLLM` V1).

**Добавить:**
- Импорт `from safetensors import safe_open` (в `process_tokens_to_speech`).
- Импорт `from vllm import SamplingParams` (вместо подкласса).
- Импорт `from pathlib import Path` — если ещё нет.
- В `__init__`: `self.hidden_states_dir = Path("/tmp/auralis/hidden_states"); self.hidden_states_dir.mkdir(parents=True, exist_ok=True)`.
- В `init_vllm_engine`: параметры `speculative_config` + `kv_transfer_config` (см. §2.3).
- В `get_generation_context`: замена `ExtendedSamplingParams(...)` на `SamplingParams(...)` с нативным `repetition_penalty` и без `logits_processors`/`hidden_state_collector`/`request_id` (см. псевдокод §2.3).
- В `process_tokens_to_speech`: чтение hidden states через `safe_open(output.kv_transfer_params["hidden_states_path"])`.
- В `shutdown`: `asyncio.to_thread(shutil.rmtree, self.hidden_states_dir, ignore_errors=True)` для очистки.

### 3.2. `src/auralis/models/xttsv2/components/vllm/hijack.py`

**Удалить файл целиком.** Ни один его символ не используется после §2.1–§2.5.

### 3.3. `src/auralis/models/xttsv2/components/vllm/hidden_state_collector.py`

**Удалить файл целиком.**

### 3.4. `src/auralis/models/xttsv2/components/vllm/__init__.py`

**Удалить** (папка `components/vllm/` исчезает).

### 3.5. `src/auralis/models/xttsv2/components/vllm_mm_gpt.py`

**Переработка полностью (см. §1).** Содержит:
- Замена импортов (§1.2).
- Новые классы `XttsProcessingInfo`, `XttsDummyInputsBuilder`, `XttsMultiModalProcessor` (§1.4).
- Новый `XttsGPT(nn.Module, SupportsMultiModal, SupportsPP)` с методами `embed_multimodal`, `get_language_model`, `get_input_embeddings`, `get_placeholder_str`, `forward`, `load_weights`, `compute_logits` (§1.3, §1.5).
- Декоратор `@MULTIMODAL_REGISTRY.register_processor(XttsMultiModalProcessor, info=XttsProcessingInfo, dummy_inputs=XttsDummyInputsBuilder)`.
- **Удалить:** `Sampler`, `SamplerOutput`, `SamplingMetadata`, `AttentionMetadata`, `SequenceData`, все функции-регистраторы `INPUT_REGISTRY.register_*`, старый ручной scatter conditioning'а в `forward`.

### 3.6. `src/auralis/models/xttsv2/__init__.py`

**Без изменений по структуре**, но проверить:
- Строка `5`: `from vllm import ModelRegistry` — ok.
- Строка `8`: `ModelRegistry.register_model("XttsGPT", XttsGPT)` — ok. При желании завернуть в entry-point-плагин (опционально, не блокер миграции).

### 3.7. `src/auralis/core/tts.py`

**Без функциональных изменений.** Только:
- Проверить, что `set_vllm_logging_level` (используется на строке `34`) ещё работает с логгерами `vllm.*` в V1. В V1 имена логгеров изменились (`vllm.v1.engine.*` и т.п.) — наш `name.startswith('vllm')` в `common/logging/logger.py` продолжит работать.

### 3.8. `src/auralis/common/logging/logger.py`

**Без изменений.** Проверить, что overrider подхватывает новые логгеры V1 (префикс `vllm` остаётся).

### 3.9. `src/auralis/common/definitions/requests.py`

**Без изменений.** `TTSRequest.repetition_penalty` теперь идёт напрямую в нативный `SamplingParams.repetition_penalty`.

### 3.10. `requirements.txt` и `setup.py`

**`requirements.txt`:**
```diff
- vllm==0.6.4.post1
+ vllm>=0.14.0
+ safetensors>=0.4.5
  transformers          # снять пин транзитивно от vllm
  torchaudio            # см. §4 для совместимых версий
  tokenizers
```

**`setup.py` (`install_requires`):**
- Заменить `"vllm==0.6.4.post1"` на `"vllm>=0.14.0"`.
- Добавить `"safetensors>=0.4.5"` (уже присутствует дважды — убрать дубликат, обновить пин).
- Удалить упоминание устаревших версий в комментариях, если есть.
- Версия пакета: `0.2.8.post2` → `0.3.0` (major-ish bump, ломающие изменения в внутренностях модели).

### 3.11. `Dockerfile`

Проверить базовый образ CUDA: vLLM 0.14 собран под CUDA 12.9 колёса; минимальная поддерживаемая `cu118`/`cu128` через доп. index. Обновить `FROM` на `nvidia/cuda:12.9.0-devel-ubuntu22.04` (или аналогичный). **Не в скоупе миграции кода** — отдельная задача.

---

## 4. Colab Setup Block

### 4.1. Обоснование версий

Из документации vLLM (`docs.vllm.ai/en/latest/getting_started/installation/gpu`):
- Официально рекомендуемый путь — `pip install vllm --extra-index-url https://download.pytorch.org/whl/cu129` (CUDA 12.9 wheels).
- vLLM `0.14.0rc2` собран против **PyTorch 2.8.x** (из `api/vllm/compilation/wrapper`: «Context manager for dispatching to compiled code (torch<2.8)» — значит 2.8 уже baseline; use_inductor_graph_partition требует `torch>=2.9.0.dev` — опционально).
- Соответственно для `torch==2.8.0` совместимы: `torchvision==0.23.0`, `torchaudio==2.8.0` (официальная PyTorch release matrix).
- Colab: по умолчанию CUDA 12.x. Используем `cu129` index (backward compatible с 12.1+ драйвером в Colab).

### 4.2. Блок для `notebooks/colab_quickstart.ipynb` (ячейка установки)

```bash
# === Auralis + vLLM V1 Engine (Colab setup) ===
# vLLM 0.14 (V1 Engine), PyTorch 2.8, CUDA 12.9 wheels.

# 1. Снять старые вариации torch, если Colab уже что-то установил.
!pip uninstall -y torch torchvision torchaudio vllm || true

# 2. Установить PyTorch 2.8.0 + matching torchvision/torchaudio для CUDA 12.9.
!pip install --index-url https://download.pytorch.org/whl/cu129 \
    torch==2.8.0 \
    torchvision==0.23.0 \
    torchaudio==2.8.0

# 3. Установить vLLM (0.14.x, V1 Engine по умолчанию).
!pip install "vllm>=0.14.0" --extra-index-url https://download.pytorch.org/whl/cu129

# 4. Поставить Auralis и его TTS-зависимости (кроме уже установленного torch*).
!pip install "safetensors>=0.4.5"
!pip install --no-deps "auralis @ git+https://github.com/astramind-ai/Auralis.git@<v1-migration-branch>"
!pip install aiofiles langid beautifulsoup4 cachetools colorama cutlet EbookLib einops \
    ffmpeg fsspec hangul_romanize huggingface_hub ipython librosa networkx num2words \
    opencc packaging pyloudnorm pypinyin sounddevice soundfile "spacy==3.7.5" setuptools \
    tokenizers transformers nvidia-ml-py numpy

# 5. Проверка.
import torch, vllm
print("torch:", torch.__version__, "CUDA:", torch.version.cuda, "device:", torch.cuda.get_device_name(0))
print("vllm:", vllm.__version__)

# 6. (Опционально) Форсировать V1 Engine, если релиз даёт такую переменную.
import os
os.environ.setdefault("VLLM_USE_V1", "1")  # no-op в 0.14, но декларативно.
```

### 4.3. Примечания

- Если Colab обновит драйвер CUDA до 13.x — сменить `cu129` на `cu130` в обоих URL; прочие пины совместимы.
- `--no-deps` на установке Auralis — чтобы не переставить поверх `torch` транзитивно-старую версию.
- Блок `pip uninstall -y torch ...` обязателен: Colab предустанавливает `torch` несовместимой ветки, и без него `vllm` может подтянуть микс версий.

---

## 5. План валидации миграции (checkpoints)

1. **Smoke-тест импортов.** `python -c "from auralis import TTS; from auralis.models.xttsv2 import XTTSv2Engine"` — не должно быть ImportError.
2. **Регистрация модели.** `ModelRegistry.resolve_model_cls("XttsGPT")` возвращает новый класс.
3. **Processor контракт.** `MULTIMODAL_REGISTRY.get_processing_info(model_config)` возвращает `XttsProcessingInfo`.
4. **Инициализация движка.** `TTS().from_pretrained("AstraMindAI/xttsv2")` — проходит без падений с V1 Engine.
5. **Один проход генерации.** Запросить короткий текст → получить waveform. Сравнить на слух + по длительности с baseline (V0, 0.6.4).
6. **Hidden states.** Файл `<shared_storage_path>/*.safetensors` существует, корректно парсится `safe_open`, форма `[prompt+gen, 1, hidden_size]`.
7. **Concurrency.** `max_concurrency=8`, 8 параллельных запросов — нет дедлоков, нет утечки VRAM (сравнить с baseline).
8. **Shutdown.** `await tts.shutdown()` не бросает — `AsyncLLM.shutdown()` корректно закрывает фоновый loop.
9. **Очистка `hidden_states_dir`.** После серии запросов нет нарастания файлов на диске.

---

## 6. Риски и невыполненные требования

| Риск | Митигация |
|---|---|
| `ExampleHiddenStatesConnector` — «example», не гарантирован как stable API. | Зафиксировать `vllm>=0.14.0,<0.15.0` в requirements; в долгосроке — написать свой `KVConnector` с той же семантикой. |
| Формат `eagle_aux_hidden_state_layer_ids` предполагает совместимую архитектуру (EAGLE speculative path). Для кастомного XttsGPT может потребоваться реализовать `ExtractHiddenStatesModel`-совместимый draft. | Проверить на smoke-тесте (§5.6). Альтернатива: собственный `KVConnector` с извлечением hidden states из последнего слоя любым путём. |
| `repetition_penalty=5.0` по умолчанию в `TTSRequest` — агрессивный; в текущей реализации применяется ручным `LogitsRepetitionPenalizer` с идентичной формулой, но семантика нативного vLLM `repetition_penalty` может отличаться (scope: prompt+generated vs только generated). | Smoke-тест с фиксированным seed + сравнение на слух. Если расходится — пишем class-based `LogitsProcessor` (V1-совместимый). |
| Удаление `is_logits_only_mode` предполагает, что hidden states последнего слоя для **уже-сгенерированных** токенов сохраняются. KV-connector пишет hidden states по мере прохождения токенов через `forward` — это выполняется штатно. | Smoke-тест §5.6: проверить, что `hidden_states.shape[0] == prompt_len + len(token_ids)`. |
| `max_gb_for_vllm_model` рассчитан эмпирически под V0. V1 имеет другой overhead на request tracking / KV-connector. | Перекалибровать `get_memory_usage_curve` после первого успешного прогона (отдельная задача, не блокирует миграцию). |

---

## 7. Что НЕ меняем в этой миграции

- Базовую модель XTTS v2 (веса, config, tokenizer).
- Публичный API `TTS`, `TTSRequest`, `TTSOutput`.
- `TwoPhaseScheduler` и общую архитектуру `core/tts.py`.
- HiFi-GAN декодер, `ConditioningEncoder`, `PerceiverResampler`.
- Формат аудио-выхода (sample_rate, bit_depth).
- Логирование (`common/logging/logger.py`).

---

**Конец спецификации.** Ждём подтверждения на переход к Фазе 3 (имплементация).
