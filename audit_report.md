# Auralis — vLLM Integration Audit Report

**Дата:** 2026-04-22
**Скоуп:** только чтение, базовая модель (XTTSv2) не менялась.
**Цель:** проследить полный путь данных `text → vLLM → audio` и зафиксировать все точки сопряжения с vLLM.

---

## 0. Краткое резюме

Auralis — асинхронный TTS-движок поверх XTTS v2, где **авторегрессионный GPT-декодер аудио-токенов исполняется внутри vLLM** (`AsyncLLMEngine`). Интеграция глубокая, не ограничивается вызовом `.generate()`:

- **Кастомная модель** `XttsGPT` регистрируется в `vllm.ModelRegistry` и использует внутренности `vllm.model_executor.*` (Sampler, SamplingMetadata, AttentionMetadata, VllmConfig, ParallelLMHead, GPT2Block и т.д.).
- **Наследование `SamplingParams`** (msgspec Struct с `kw_only=True`) → добавление полей `hidden_state_collector`, `request_id`.
- **Multimodal-ввод** через кастомный ключ `"audio"` с полями `embeds` / `is_logits_only_mode` / `sequence_length` (не стандартный vLLM-контракт).
- **Два режима вызова `.generate()`**: (1) `max_tokens=1` для сбора hidden states через callback, (2) полная AR-генерация mel-токенов.
- Используется закрытый API: `llm_engine.engine.model_config.dtype`, `shutdown_background_loop()`, `RequestOutputKind.FINAL_ONLY`.

**Любое обновление vLLM > 0.6.4.post1 с высокой вероятностью ломает проект** — см. раздел 1.

---

## 1. Карта вызовов vLLM

| Файл | Строки | Тип вызова | Аргументы (дословно) | Риск при обновлении |
|---|---|---|---|---|
| `src/auralis/models/xttsv2/__init__.py` | 5 | `import` (public) | `from vllm import ModelRegistry` | **Низкий**. Публичный API, стабилен. |
| `src/auralis/models/xttsv2/__init__.py` | 8 | Регистрация модели | `ModelRegistry.register_model("XttsGPT", XttsGPT)` | **Средний**. Сигнатура стабильна, но требования к регистрируемому классу (SupportsMultiModal, SupportsPP) мигрируют. |
| `src/auralis/models/xttsv2/XTTSv2.py` | 17 | `import` (public) | `from vllm import AsyncLLMEngine, AsyncEngineArgs, TokensPrompt, RequestOutput` | **Средний**. `TokensPrompt` перенесён в `vllm.inputs` в новых релизах; `AsyncLLMEngine` в v0.8+ помечен как legacy (миграция на `LLM`/`AsyncLLM` V1 Engine). |
| `src/auralis/models/xttsv2/XTTSv2.py` | 18 | `import` (public) | `from vllm.multimodal import MultiModalDataDict` | **Высокий**. Multimodal API в vLLM активно рефакторится (v0.7+: `MultiModalKwargs`, `PlaceholderRange`, новые процессоры). |
| `src/auralis/models/xttsv2/XTTSv2.py` | 19 | `import` (internal) | `from vllm.sampling_params import RequestOutputKind` | **Средний**. Подмодуль `sampling_params`, но enum относительно стабилен. |
| `src/auralis/models/xttsv2/XTTSv2.py` | 20 | `import` (internal) | `from vllm.utils import Counter` | **Низкий**. Utility, вряд ли удалят. |
| `src/auralis/models/xttsv2/XTTSv2.py` | 212–230 | Конфиг движка | `AsyncEngineArgs(model=self.gpt_model, tensor_parallel_size=self.tp, pipeline_parallel_size=self.pp, dtype="auto", max_model_len=..., gpu_memory_utilization=mem_utils, trust_remote_code=True, enforce_eager=True, limit_mm_per_prompt={"audio": 1}, max_num_seqs=max_seq_num, disable_log_stats=True, max_num_batched_tokens=...)` | **Высокий**. Набор/семантика полей `AsyncEngineArgs` менялся в 0.7 / 0.8 / 0.9 (появились `kv_transfer_config`, переезды `swap_space`, обязательное `--engine v0/v1`). |
| `src/auralis/models/xttsv2/XTTSv2.py` | 232 | Инициализация движка | `AsyncLLMEngine.from_engine_args(engine_args)` | **Высокий**. В V1 Engine путь инициализации изменён; `from_engine_args` остался как shim. |
| `src/auralis/models/xttsv2/XTTSv2.py` | 346 | Использование внутреннего поля | `self.llm_engine.engine.model_config.dtype` | **Очень высокий**. Лазание во внутренности (`.engine.model_config`) — в V1 этот атрибут может отсутствовать / переехать. |
| `src/auralis/models/xttsv2/XTTSv2.py` | 648 | Конструктор промпта | `TokensPrompt(prompt_token_ids=token_ids)` | **Средний**. TypedDict-совместимый, но с 0.8 добавлены обязательные/опциональные поля. |
| `src/auralis/models/xttsv2/XTTSv2.py` | 658–664 | Кастомный SamplingParams | `ExtendedSamplingParams(detokenize=False, request_id=request_id, max_tokens=1, hidden_state_collector=bound_collector, output_kind=RequestOutputKind.FINAL_ONLY)` | **Очень высокий**. Наследование msgspec-Struct + приватные поля; `SamplingParams` в 0.7+ получил новые обязательные поля, `output_kind` семантика меняется в V1 (streaming API перестроен). |
| `src/auralis/models/xttsv2/XTTSv2.py` | 667–671 | Генерация (logits-only) | `self.llm_engine.generate(prompt=engine_inputs, sampling_params=sampling_params, request_id=request_id)` | **Высокий**. Сигнатура `AsyncLLMEngine.generate` в V1 Engine меняется (например, `prompt` заменяется на `PromptType`, `lora_request`/`trace_headers` стали keyword-only). |
| `src/auralis/models/xttsv2/XTTSv2.py` | 727–739 | Кастомный SamplingParams (AR) | `ExtendedSamplingParams(temperature=request.temperature, top_p=request.top_p, detokenize=False, request_id=uuid.uuid4(), top_k=request.top_k, logits_processors=[LogitsRepetitionPenalizer(request.repetition_penalty)], repetition_penalty=1.0, max_tokens=self.gpt_config.gpt_max_audio_tokens, ignore_eos=True, stop_token_ids=[self.mel_eos_token_id], output_kind=RequestOutputKind.FINAL_ONLY)` | **Очень высокий**. `logits_processors` в V1 Engine имеют новый контракт (класс-based `LogitsProcessor` с `update_state`), callable-сигнатура устаревает. |
| `src/auralis/models/xttsv2/XTTSv2.py` | 741–749 | `TokensPrompt` + multimodal | `engine_inputs["multi_modal_data"] = {"audio": {"embeds": ..., "is_logits_only_mode": False, "sequence_length": ...}}` | **Очень высокий**. Кастомная модальность `"audio"` с нестандартной структурой — в новых vLLM требуется регистрация через `MULTIMODAL_REGISTRY.register_processor`. Ключ `is_logits_only_mode` — проектный хак. |
| `src/auralis/models/xttsv2/XTTSv2.py` | 752–756 | Генерация (AR) | `self.llm_engine.generate(prompt=engine_inputs, sampling_params=sampling_params, request_id=request_id)` | **Высокий**. Как выше. |
| `src/auralis/models/xttsv2/XTTSv2.py` | 765, 773 | Тип-аннотация | `AsyncGenerator[RequestOutput, None]` | **Низкий**. Структура `RequestOutput` стабильна в публичной части (`outputs[0].token_ids`, `finished`, `request_id`). |
| `src/auralis/models/xttsv2/XTTSv2.py` | 819 | Shutdown | `self.llm_engine.shutdown_background_loop()` | **Средний**. Метод существует у `AsyncLLMEngine` V0; в V1 (`AsyncLLM`) отсутствует — используется `shutdown()`. |
| `src/auralis/models/xttsv2/components/vllm/hijack.py` | 4 | `import` (public) | `from vllm import SamplingParams` | **Высокий**. Наследование от msgspec-Struct сильно связано с внутренней структурой класса. |
| `src/auralis/models/xttsv2/components/vllm/hijack.py` | 9 | Наследование | `class ExtendedSamplingParams(SamplingParams, kw_only=True)` | **Очень высокий**. `kw_only=True` — msgspec-атрибут; добавление обязательных полей в `SamplingParams` ломает субклассирование. |
| `src/auralis/models/xttsv2/components/vllm_mm_gpt.py` | 15 | `import` (internal) | `from vllm.attention import AttentionMetadata` | **Очень высокий**. `AttentionMetadata` переработан (v0.7+: Backend-specific metadata, FlashAttention/XFormers split). |
| `src/auralis/models/xttsv2/components/vllm_mm_gpt.py` | 16 | `import` (internal) | `from vllm.config import CacheConfig, MultiModalConfig, VllmConfig` | **Очень высокий**. `MultiModalConfig` перенесён / переименован; `VllmConfig` расширяется каждый релиз. |
| `src/auralis/models/xttsv2/components/vllm_mm_gpt.py` | 17 | `import` (internal) | `from vllm.distributed import get_pp_group` | **Средний**. |
| `src/auralis/models/xttsv2/components/vllm_mm_gpt.py` | 18 | `import` (internal) | `from vllm.inputs import InputContext, INPUT_REGISTRY, DecoderOnlyInputs, token_inputs, DummyData` | **Очень высокий**. `INPUT_REGISTRY` / `InputContext` / `DummyData` реструктурированы в v0.7–0.8 (новые Input Processors и Processor Plugins). |
| `src/auralis/models/xttsv2/components/vllm_mm_gpt.py` | 19 | `import` (internal) | `from vllm.model_executor.layers.logits_processor import LogitsProcessor` | **Высокий**. |
| `src/auralis/models/xttsv2/components/vllm_mm_gpt.py` | 20 | `import` (internal) | `from vllm.model_executor.layers.quantization import QuantizationConfig` | **Средний**. |
| `src/auralis/models/xttsv2/components/vllm_mm_gpt.py` | 21 | `import` (internal) | `from vllm.model_executor.layers.sampler import Sampler, SamplerOutput` | **Очень высокий**. `Sampler` полностью переписан в V1 Engine; `SamplerOutput` структура сменилась. |
| `src/auralis/models/xttsv2/components/vllm_mm_gpt.py` | 22 | `import` (internal) | `from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead` | **Средний**. Сигнатуры конструкторов стабильнее, но не гарантируются. |
| `src/auralis/models/xttsv2/components/vllm_mm_gpt.py` | 23 | `import` (internal) | `from vllm.model_executor.model_loader.weight_utils import default_weight_loader` | **Средний**. |
| `src/auralis/models/xttsv2/components/vllm_mm_gpt.py` | 24 | `import` (internal) | `from vllm.model_executor.models.gpt2 import GPT2Block` | **Высокий**. Внутренние `GPT2Block`-имплементации меняют конструкторы при апгрейде attention backend. |
| `src/auralis/models/xttsv2/components/vllm_mm_gpt.py` | 25 | `import` (internal) | `from vllm.model_executor.models.utils import make_layers, make_empty_intermediate_tensors_factory` | **Высокий**. |
| `src/auralis/models/xttsv2/components/vllm_mm_gpt.py` | 26 | `import` (internal) | `from vllm.model_executor.sampling_metadata import SamplingMetadata` | **Очень высокий**. В V1 `SamplingMetadata` заменена `InputBatch`/`SamplerMetadata`. |
| `src/auralis/models/xttsv2/components/vllm_mm_gpt.py` | 27 | `import` (internal) | `from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalInputs, MultiModalKwargs` | **Очень высокий**. `MultiModalInputs` удалён / переименован в v0.7+. |
| `src/auralis/models/xttsv2/components/vllm_mm_gpt.py` | 28 | `import` (internal) | `from vllm.multimodal.inputs import PlaceholderRange` | **Высокий**. |
| `src/auralis/models/xttsv2/components/vllm_mm_gpt.py` | 29 | `import` (internal) | `from vllm.multimodal.utils import consecutive_placeholder_ranges` | **Высокий**. |
| `src/auralis/models/xttsv2/components/vllm_mm_gpt.py` | 30 | `import` (internal) | `from vllm.sequence import IntermediateTensors, SequenceData, VLLM_TOKEN_ID_ARRAY_TYPE` | **Очень высокий**. `SequenceData` в V1 Engine не используется (новый `InputBatch`/`Request`). |
| `src/auralis/models/xttsv2/components/vllm_mm_gpt.py` | 31 | `import` (internal) | `from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP` | **Высокий**. Интерфейсы расширяются (добавляются новые обязательные методы типа `get_multimodal_embeddings`). |
| `src/auralis/models/xttsv2/components/vllm_mm_gpt.py` | 36 | `import` (internal) | `from vllm.utils import is_list_of` | **Низкий**. |
| `src/auralis/common/logging/logger.py` | — | Перехват логгеров | `logging.getLogger(name)` для `name.startswith('vllm')` | **Низкий**. Не API-контракт, но зависит от имён логгеров vLLM. |

**Итог по вызовам:**
- `AsyncLLMEngine.generate(...)` — **2 точки** (`XTTSv2.py:667`, `XTTSv2.py:752`).
- `AsyncLLMEngine.from_engine_args(...)` — **1 точка** (`XTTSv2.py:232`).
- `AsyncLLMEngine.shutdown_background_loop()` — **1 точка** (`XTTSv2.py:819`).
- `.add_request(` — **0 вызовов** в коде проекта.
- `.abort(` — **0 вызовов** в коде проекта (no cancellation path!).

---

## 2. Контракты пайплайна

### 2.1. Полный путь данных

```
TTSRequest(text, speaker_files, lang, sampling params, ...)
    │
    ▼
TTS.generate_speech / generate_speech_async                 (core/tts.py)
    │
    ▼
TwoPhaseScheduler.run(first_phase_fn, second_phase_fn)
    │
    ├── Phase 1: TTS._prepare_generation_context
    │       │
    │       ▼
    │   XTTSv2Engine.prepare_inputs_async
    │       ├── XTTSTokenizerFast → List[int] (text tokens, BOS/EOS вставлены)
    │       ├── text_embedding + text_pos_embedding → List[Tensor]
    │       ├── get_audio_conditioning → (gpt_cond_latent, speaker_embeddings)
    │       └── _merge_conditioning → cond_latents (concat audio+text, cast в dtype vLLM-модели)
    │
    │   XTTSv2Engine.get_generation_context
    │       для каждой подпоследовательности:
    │         ExtendedSamplingParams(...)            [см. 2.3]
    │         TokensPrompt(prompt_token_ids=seq)
    │         engine_inputs["multi_modal_data"] = {"audio": {...}}  [см. 2.2]
    │         token_generator = llm_engine.generate(prompt, sampling_params, request_id)
    │       возвращает: (generators, request_ids, speaker_embeddings, gpt_embed_inputs)
    │
    ▼
Phase 2: TTS._second_phase_fn → XTTSv2Engine.process_tokens_to_speech
    async for output (RequestOutput) in generator:
        if output.finished:
            hidden_states = await get_model_logits(          ← ВТОРОЙ вызов .generate()
                token_ids=output.outputs[0].token_ids,
                conditioning={"audio": {..., "is_logits_only_mode": True}},
                request_id=output.request_id
            )
            # get_model_logits:
            #   ExtendedSamplingParams(max_tokens=1, hidden_state_collector=...)
            #   llm_engine.generate(...)
            #   callback через forward() внутри XttsGPT собирает hidden states
            #   → HiddenStatesCollector.get_hidden_states(request_id)
            wav = hifigan_decoder(hidden_states, g=speaker_embeddings)
            yield TTSOutput(array=wav, start_time, token_length)
    │
    ▼
Combine / stream → Union[AsyncGenerator[TTSOutput], TTSOutput]
```

### 2.2. Форматы данных на каждом этапе

| Этап | Структура | Источник |
|---|---|---|
| **Вход** | `TTSRequest` (dataclass): `text: str`, `speaker_files: str\|bytes\|List`, `language`, `temperature=0.75`, `top_p=0.85`, `top_k=50`, `repetition_penalty=5.0`, `stream`, `request_id`, `max_ref_length=60`, `gpt_cond_len=30`, `gpt_cond_chunk_len=4` | `common/definitions/requests.py` |
| **После токенизации** | `List[List[int]]` (split по предложениям), каждый с BOS/EOS (`tokenizer.bos_token_id`, `tokenizer.eos_token_id`) | `XTTSv2.prepare_text_tokens_async` |
| **Text embeddings** | `List[Tensor]` формы `[1, seq_len, hidden_size]` (= `text_embedding + text_pos_embedding`) | `XTTSv2.prepare_text_tokens_async` |
| **Audio conditioning** | `gpt_cond_latent: Tensor [1, 1024, T]`, `speaker_embeddings: Tensor` | `XTTSv2.get_conditioning_latents` |
| **Merged cond_latents** | `List[Tensor]` формы `[audio_cond + text_emb, hidden_size]`, кастованы в `llm_engine.engine.model_config.dtype` | `XTTSv2._merge_conditioning` |
| **vLLM prompt** | `TokensPrompt(prompt_token_ids=List[int])` + `multi_modal_data={"audio": {"embeds": Tensor, "is_logits_only_mode": bool, "sequence_length": int}}` | `XTTSv2.get_generation_context` / `get_model_logits` |
| **vLLM выход (AR)** | `RequestOutput` (`.finished: bool`, `.request_id: str`, `.outputs: List[CompletionOutput]`, `outputs[0].token_ids: List[int]`) — только `FINAL_ONLY` | vLLM |
| **vLLM выход (logits-only)** | hidden states через callback `HiddenStatesCollector.sync_collect(hidden_states, request_id)`, hijacked в `forward()` кастомной модели `XttsGPT` | `hidden_state_collector.py` + `vllm_mm_gpt.py` |
| **После hidden-state slice** | `Tensor[1, T_audio, hidden_size]` = `final_norm(hidden_states[audio_start:-5])` | `XTTSv2.get_model_logits:687` |
| **После HiFi-GAN** | `np.ndarray` waveform (sample_rate=24000 по умолч.) | `HifiDecoder` |
| **Финальный выход** | `TTSOutput(array: np.ndarray\|bytes, sample_rate=24000, bit_depth=32, bit_rate=192, channel=1, start_time, token_length)` | `common/definitions/output.py` |

### 2.3. Структура "промптов" (vLLM не получает текст — только токены + эмбеддинги)

**AR-генерация аудио-токенов (`XTTSv2.py:741–749`):**
```python
engine_inputs = TokensPrompt(prompt_token_ids=sequence)  # sequence = [1,1,1,...] fake text ids
engine_inputs["multi_modal_data"] = {
    "audio": {
        "embeds": gpt_embed_inputs[seq_index],   # Tensor [audio_cond + text_emb]
        "is_logits_only_mode": False,
        "sequence_length": len(sequence)
    }
}
```

**Logits-only проход (`XTTSv2.py:645–651`):**
```python
token_ids = [mel_bos_token_id] + list(output_token_ids) + [mel_eos_token_id]*4  # ровно 5 eos хвостом
engine_inputs = TokensPrompt(prompt_token_ids=token_ids)
conditioning['audio']['sequence_length'] = len(token_ids)
engine_inputs["multi_modal_data"] = {
    "audio": {
        "embeds": multimodal_data,
        "is_logits_only_mode": True,
        "sequence_length": len(token_ids)
    }
}
```

**Примечание:** формат `{"audio": {"embeds": ..., "is_logits_only_mode": ..., "sequence_length": ...}}` — **не стандартный vLLM Multimodal contract**. Кастомный ключ `is_logits_only_mode` переключает поведение `forward()` в `XttsGPT` (см. `vllm_mm_gpt.py`). Стандартный vLLM ожидает сырые тензоры/`PIL.Image`, а процессоры описывает через `InputProcessor` / `MULTIMODAL_REGISTRY`.

### 2.4. Текущие значения `SamplingParams` (фактические, не defaults)

**Для сбора hidden states (`XTTSv2.py:658–664`):**
```python
ExtendedSamplingParams(
    detokenize=False,
    request_id=request_id,                # кастомное поле (not in vLLM)
    max_tokens=1,
    hidden_state_collector=bound_collector,   # кастомное поле (not in vLLM)
    output_kind=RequestOutputKind.FINAL_ONLY
)
```

**Для AR-генерации mel-токенов (`XTTSv2.py:727–739`):**
```python
ExtendedSamplingParams(
    temperature=request.temperature,          # default 0.75
    top_p=request.top_p,                      # default 0.85
    detokenize=False,
    request_id=uuid.uuid4(),                  # кастомное поле
    top_k=request.top_k,                      # default 50
    logits_processors=[LogitsRepetitionPenalizer(request.repetition_penalty)],  # callable(prompt_ids, token_ids, logits) → logits
    repetition_penalty=1.0,                   # обнулён, т.к. применяется вручную (!)
    max_tokens=gpt_config.gpt_max_audio_tokens,
    ignore_eos=True,                          # игнорируется текстовый EOS
    stop_token_ids=[mel_eos_token_id],        # mel-specific EOS
    output_kind=RequestOutputKind.FINAL_ONLY  # стриминг токенов vLLM→Auralis отключён
)
```

`length_penalty=1.0` и `do_sample=True` из `TTSRequest` **в vLLM не пробрасываются** (vLLM не поддерживает length_penalty для AR; `do_sample` имплицитно через temperature>0).

### 2.5. Структура `RequestOutput` и путь результата

Используемые поля `RequestOutput`:
- `output.finished: bool` — триггер для запуска hifigan.
- `output.request_id: str` — для корреляции с `HiddenStatesCollector`.
- `output.outputs: List[CompletionOutput]` — всегда берётся `[0]`.
- `output.outputs[0].token_ids: List[int]` — список сгенерированных mel-токенов.

Поток:
1. `async for output in generator` в `process_tokens_to_speech` (`XTTSv2.py:785`).
2. Из-за `RequestOutputKind.FINAL_ONLY` приходит **ровно один финальный output** на запрос.
3. Вызов `get_model_logits(output.outputs[0].token_ids, ...)` → второй раунд `.generate()` с `max_tokens=1` для сбора hidden states (через hijacked `forward()` + callback).
4. `final_norm(hidden_states[audio_start:-5])` → `hifigan_decoder(hidden_states, g=speaker_embeddings)` в thread-pool.
5. Waveform → `TTSOutput(array=wav, start_time=request.start_time, token_length=len(token_ids))`.
6. `TwoPhaseScheduler` агрегирует чанки → `TTS.combine_outputs` (если `stream=False`) или стримит по одному.

---

## 3. Точки импорта (grep)

Полный листинг (только файлы проекта, без `combined_python_files.txt`):

```
src/auralis/__init__.py
  4: from .common.logging.logger import setup_logger, set_vllm_logging_level

src/auralis/models/xttsv2/__init__.py
  2: from .components.vllm_mm_gpt import XttsGPT
  5: from vllm import ModelRegistry
  8: ModelRegistry.register_model("XttsGPT", XttsGPT)

src/auralis/models/xttsv2/XTTSv2.py
 17: from vllm import AsyncLLMEngine, AsyncEngineArgs, TokensPrompt, RequestOutput
 18: from vllm.multimodal import MultiModalDataDict
 19: from vllm.sampling_params import RequestOutputKind
 20: from vllm.utils import Counter
 33: from .components.vllm.hidden_state_collector import HiddenStatesCollector
 34: from .components.vllm.hijack import ExtendedSamplingParams, LogitsRepetitionPenalizer
232: self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
667: generator = self.llm_engine.generate(prompt=engine_inputs, sampling_params=sampling_params, request_id=request_id)
752: token_generator = self.llm_engine.generate(prompt=engine_inputs, sampling_params=sampling_params, request_id=request_id)
819: self.llm_engine.shutdown_background_loop()

src/auralis/models/xttsv2/components/vllm/hijack.py
  4: from vllm import SamplingParams
  9: class ExtendedSamplingParams(SamplingParams, kw_only=True)

src/auralis/models/xttsv2/components/vllm_mm_gpt.py
 15: from vllm.attention import AttentionMetadata
 16: from vllm.config import CacheConfig, MultiModalConfig, VllmConfig
 17: from vllm.distributed import get_pp_group
 18: from vllm.inputs import InputContext, INPUT_REGISTRY, DecoderOnlyInputs, token_inputs, DummyData
 19: from vllm.model_executor.layers.logits_processor import LogitsProcessor
 20: from vllm.model_executor.layers.quantization import QuantizationConfig
 21: from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
 22: from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
 23: from vllm.model_executor.model_loader.weight_utils import default_weight_loader
 24: from vllm.model_executor.models.gpt2 import GPT2Block
 25: from vllm.model_executor.models.utils import make_layers, make_empty_intermediate_tensors_factory
 26: from vllm.model_executor.sampling_metadata import SamplingMetadata
 27: from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalInputs, MultiModalKwargs
 28: from vllm.multimodal.inputs import PlaceholderRange
 29: from vllm.multimodal.utils import consecutive_placeholder_ranges
 30: from vllm.sequence import IntermediateTensors, SequenceData, VLLM_TOKEN_ID_ARRAY_TYPE
 31: from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
 36: from vllm.utils import is_list_of
```

**Вызовы публичного API (grep `.generate( / .add_request( / .abort(`):**
- `llm_engine.generate(...)` — **2** вхождения (`XTTSv2.py:667`, `XTTSv2.py:752`).
- `llm_engine.add_request(...)` — **0**.
- `llm_engine.abort(...)` — **0**. ⚠️ отсутствует код отмены зависших запросов.

**Упоминания "vllm" в строках/комментариях/идентификаторах:**
- `self.max_gb_for_vllm_model` (`XTTSv2.py:72, 169, 209`)
- `self.init_vllm_engine` (`XTTSv2.py:145, 198`)
- `self.llm_engine` — атрибут класса, используется повсеместно.
- Логгер: `set_vllm_logging_level`, `VLLMLogOverrider` — перехватывают все логгеры с префиксом `vllm` (`logger.py`).

---

## 4. Текущие версии зависимостей

### 4.1. `requirements.txt`

| Пакет | Версия | Примечание |
|---|---|---|
| **vllm** | `==0.6.4.post1` | Жёсткий пин. Это **vLLM V0 Engine** — старый путь. Текущая стабильная 0.11.x использует V1 Engine с несовместимым `Sampler`/`SamplingMetadata`/multimodal API. |
| transformers | (без версии) | Резолвится транзитивно; в 0.6.4.post1 vLLM требует `transformers>=4.45.2,<4.47`. |
| tokenizers | (без версии) | Транзитивно через `transformers`. |
| torch / torchaudio | (без версии) | vLLM 0.6.4.post1 собран под `torch==2.5.1` и CUDA 12.1. |
| safetensors | (без версии) | Дублируется (строки 22 и 27). |
| spacy | `==3.7.5` | Единственный строго закреплённый, кроме vLLM. |
| numpy | (без версии) | |
| nvidia-ml-py | (без версии) | Нужна для расчёта `get_memory_percentage`. |

**`accelerate`** в `requirements.txt` / `setup.py` **отсутствует**. Проект не использует HF Accelerate напрямую.

### 4.2. `pyproject.toml`

**Файл отсутствует** в корне репозитория (и в любых подкаталогах). Метаданные хранятся только в `setup.py`.

### 4.3. `setup.py` (`install_requires`)

Дублирует `requirements.txt`. Ключевое:
```
vllm==0.6.4.post1
transformers            # без пина
torchaudio              # без пина
tokenizers              # без пина
safetensors             # указан дважды
spacy==3.7.5
numpy
nvidia-ml-py
```
- **Версия пакета Auralis:** `0.2.8.post2` (`setup.py:19`).
- **Python:** `>=3.10`.
- **Платформа:** жёстко Linux (`setup.py:check_platform()` кидает `RuntimeError` на Windows/macOS).

### 4.4. Совместимость vLLM 0.6.4.post1 (справочно)

| Требование vLLM 0.6.4.post1 | Значение |
|---|---|
| Python | 3.9–3.12 |
| torch | 2.5.1 |
| CUDA | 12.1 (колёса) / 11.8 (отдельный wheel) |
| transformers | ≥4.45.2, <4.47 |
| xformers | 0.0.28.post3 |
| V1 Engine | **нет** (V1 появился с 0.7+) |

### 4.5. Риск-карта обновления vLLM

| Целевая версия | Вероятная оценка трудозатрат | Главные блокеры |
|---|---|---|
| 0.6.5 → 0.6.6 | Низкая | минорные правки |
| 0.7.x | Средняя | `MultiModalInputs` переименован; `InputProcessor` API |
| 0.8.x | Высокая | V1 Engine по умолчанию; `Sampler` переписан; `SequenceData` удалён из V1 пути; `SamplingParams.logits_processors` — новый класс-based контракт |
| 0.9.x–0.11.x | Очень высокая | полная миграция `AsyncLLMEngine → AsyncLLM`; `AttentionMetadata` backend-specific; `MULTIMODAL_REGISTRY.register_processor` обязателен; `SamplingMetadata` заменён |

**Минимум, что придётся переписать при апгрейде на vLLM ≥ 0.8:**
1. `vllm_mm_gpt.py` целиком (кастомная модель) — новый интерфейс `SupportsMultiModal.get_multimodal_embeddings`, новый `Sampler`, новый путь `forward()`.
2. `hijack.py` — наследование `SamplingParams` через msgspec сломается; `logits_processors` → класс `LogitsProcessor` c `update_state`.
3. `XTTSv2.py` — конфигурация `AsyncEngineArgs`, сигнатура `.generate()`, `shutdown_background_loop()`, доступ к `.engine.model_config.dtype`.
4. `HiddenStatesCollector` — hijack `forward()` привязан к внутреннему потоку выполнения vLLM V0 (pre-execute hook через SamplingParams); в V1 нужен новый механизм (hidden-states output API).

---

## Стоп-файлы / артефакты

- Нет `.abort()` — зависшие запросы не отменяются, запрос остаётся в очереди vLLM до `stop_token_ids` / `max_tokens`.
- `ExtendedSamplingParams.request_id` дублирует аргумент `request_id` у `.generate()` — исторический артефакт.
- `repetition_penalty=1.0` при наличии кастомного `LogitsRepetitionPenalizer` — корректно, но неочевидно.
- `shutdown_background_loop()` вызывается из `XTTSv2.shutdown()`, но `TTS.shutdown()` вызывает `tts_engine.shutdown()` → OK.
- Комбо `enforce_eager=True` + кастомная модель + `limit_mm_per_prompt={"audio":1}` — осознанное ограничение (CUDA graphs отключены, PP/TP работают, но без speculative decoding).

---

**Конец отчёта.** Исходники не изменены.
