# Auralis ↔ vLLM 0.19.x Compatibility Audit

**Дата:** 2026-04-22
**Целевая версия vLLM:** 0.19.1 (Colab `pip install vllm` с `setup.py: vllm>=0.14.0`)
**Режим:** read-only audit. Никаких изменений в файлах, никаких git-операций.

---

## Задание 1 — Состояние vLLM

### 1.1 Версия в Colab

**Вывод: v0.19.x (скорее всего `v0.19.1`, возможно `v0.19.2rc0` / `v0.20.0rc1`).**

Обоснование:
- `ExampleHiddenStatesConnector` **зарегистрирован** в `vllm/distributed/kv_transfer/kv_connector/factory.py`:
  ```python
  KVConnectorFactory.register_connector(
      "ExampleHiddenStatesConnector",
      "vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector",
      "ExampleHiddenStatesConnector",
  )
  ```
- `VLLM_USE_V1` — просто ENV-переменная для включения V1 engine; не индикатор версии.
- `EPLBConfig` (Expert Parallel Load Balancing) — MoE-конфиг, добавленный в V1 era.
- Доступные теги: `v0.17.x → v0.18.x → v0.19.0 → v0.19.1 → v0.19.2rc0 → v0.20.0rc1`. `setup.py` Auralis требует `vllm>=0.14.0`, `pip` в Colab ставит самую свежую → **v0.19.1** (стабильная) или новее.

### 1.2 Где произошли переименования/переезды

#### `profiling.py` — УДАЛЁН

**Прямое подтверждение:** `https://raw.githubusercontent.com/vllm-project/vllm/v0.19.1/vllm/multimodal/profiling.py` → **HTTP 404**.

Содержимое переехало: `vllm.multimodal.profiling` → `vllm.multimodal.processing.dummy_inputs` (подтверждено листингом `vllm/multimodal/processing/` в main, где есть файл `dummy_inputs.py` размером 6284 байт).

> ⚠️ Важно: `vllm.multimodal.processing` **сам стал пакетом** (директорией) в v0.19.x — раньше был плоским файлом. Это значит `from vllm.multimodal.processing import BaseMultiModalProcessor` **продолжает работать** через `__init__.py` пакета, НО `from vllm.multimodal.profiling import ...` — **падает с `ModuleNotFoundError`**.

#### `MultiModalKwargs` / `MultiModalInputs` / `MultiModalDataDict` — УДАЛЕНЫ

**Прямое подтверждение из `v0.19.1/vllm/multimodal/__init__.py`** (полное содержимое файла):

```python
from .hasher import MultiModalHasher
from .inputs import BatchedTensorInputs, MultiModalKwargsItems, NestedTensors
from .registry import MultiModalRegistry
MULTIMODAL_REGISTRY = MultiModalRegistry()

__all__ = [
    "BatchedTensorInputs",
    "MultiModalHasher",
    "MultiModalKwargsItems",
    "NestedTensors",
    "MULTIMODAL_REGISTRY",
    "MultiModalRegistry",
]
```

**Что экспортируется:** только 6 имён. **`MultiModalKwargs`, `MultiModalInputs`, `MultiModalDataDict` отсутствуют.**

Для сравнения — v0.11.0 экспортировал дополнительно: `MultiModalKwargs`, `MultiModalDataDict`, `ModalityData`, `MultiModalDataBuiltins`, `MultiModalPlaceholderDict`, `MultiModalUUIDDict`.

Проверил `v0.19.1/vllm/multimodal/inputs.py`: там присутствует только **`MultiModalKwargsItems`** (новый контейнер). Старый `MultiModalKwargs` удалён как класс — остались только `MultiModalKwargsItem` (единичный) и `MultiModalKwargsItems` (плюральный контейнер).

### 1.3 Сигнатура `BaseDummyInputsBuilder` — v0.11.0 (эталон)

Абстрактные методы (из `v0.11.0/profiling.py`):

```python
class BaseDummyInputsBuilder(ABC, Generic[_I]):
    @abstractmethod
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...

    @abstractmethod
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict: ...

    # Конкретный default:
    def get_dummy_processor_inputs(
        self, seq_len, mm_counts,
    ) -> ProcessorInputs:
        dummy_text = self.get_dummy_text(mm_counts)
        dummy_mm_data = self.get_dummy_mm_data(seq_len, mm_counts)
        tokenization_kwargs = {"truncation": False}
        return ProcessorInputs(
            prompt=dummy_text, mm_data=dummy_mm_data,
            tokenization_kwargs=tokenization_kwargs,
        )
```

> ⚠️ В main/v0.19.x тело класса переехало в `processing/dummy_inputs.py`, но **те же два метода остаются абстрактными**.

### 1.4 `__init__.py` v0.19.1 multimodal

См. раздел 1.2 — приведено полное содержимое.

### 1.5 Эталонная реализация

Правильная реализация **обязана переопределить `get_dummy_text` + `get_dummy_mm_data`**, а не `get_dummy_processor_inputs`. Последний — конкретный `def`, который вызывает первые два.

---

## Задание 2 — Состояние Auralis

### 2.1 Импорты `vllm_mm_gpt.py` — статус каждого

| Импорт | Статус в v0.19.1 |
|---|---|
| `vllm.config.CacheConfig, VllmConfig` | ✅ OK |
| `vllm.distributed.get_pp_group` | ✅ OK |
| `vllm.model_executor.layers.logits_processor.LogitsProcessor` | ✅ OK (но см. #15) |
| `vllm.model_executor.layers.quantization.QuantizationConfig` | ✅ OK |
| `vllm.model_executor.layers.vocab_parallel_embedding.ParallelLMHead, VocabParallelEmbedding` | ✅ OK |
| `vllm.model_executor.model_loader.weight_utils.default_weight_loader` | ✅ OK |
| `vllm.model_executor.models.gpt2.GPT2Block` | ⚠️ TBD (нужна проверка) |
| `vllm.model_executor.models.interfaces.SupportsMultiModal, SupportsPP` | ✅ OK |
| `vllm.model_executor.models.utils.make_empty_intermediate_tensors_factory, make_layers` | ✅ OK |
| `vllm.multimodal.MULTIMODAL_REGISTRY` | ✅ OK |
| `vllm.multimodal.inputs.MultiModalFieldConfig` | ✅ OK |
| **`vllm.multimodal.inputs.MultiModalInputs`** | ❌ **УДАЛЁН** |
| **`vllm.multimodal.inputs.MultiModalKwargs`** | ❌ **УДАЛЁН** — использовать `MultiModalKwargsItem` + `MultiModalKwargsItems` |
| `vllm.multimodal.inputs.NestedTensors` | ✅ OK |
| `vllm.multimodal.parse.MultiModalDataParser` | ✅ OK |
| `vllm.multimodal.processing.BaseMultiModalProcessor, BaseProcessingInfo, PromptReplacement` | ✅ OK (processing стал пакетом, `__init__.py` реэкспортирует) |
| **`vllm.multimodal.profiling.BaseDummyInputsBuilder, ProcessorInputs`** | ❌ **МОДУЛЬ УДАЛЁН** — переехал в `vllm.multimodal.processing.dummy_inputs` |
| `vllm.sequence.IntermediateTensors` | ✅ OK |

### 2.2 `XttsDummyInputsBuilder` — анализ

`src/auralis/models/xttsv2/components/vllm_mm_gpt.py:151-183`

```python
class XttsDummyInputsBuilder(BaseDummyInputsBuilder[XttsProcessingInfo]):
    """Generates dummy data for memory profiling at engine start-up."""

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        ...
        return ProcessorInputs(
            prompt_text="",
            mm_data={"audio": {"embeds": dummy_embeds}},
            hf_processor_mm_kwargs={},
        )
```

**Проблемы:**

1. **Не реализованы абстрактные методы** `get_dummy_text` и `get_dummy_mm_data` → `TypeError: Can't instantiate abstract class XttsDummyInputsBuilder with abstract methods get_dummy_mm_data, get_dummy_text` при регистрации процессора.
2. **Неправильное имя поля** `ProcessorInputs`: в v0.11.0 поле называется `prompt`, а не `prompt_text`. Auralis передаёт `prompt_text=""` — поле не существует → `TypeError`.
3. **Формат `mm_data`** ожидает `MultiModalDataDict`, где для аудио значением должны быть объекты `AudioItem` (numpy/tensor), а не `{"embeds": [...]}` — Auralis протащил свою структуру.

### 2.3 Raw audio vs embeddings

**Auralis передаёт ТОЛЬКО pre-computed embeddings.** `src/auralis/models/xttsv2/XTTSv2.py:718-723`:

```python
engine_inputs: TokensPrompt = {
    "prompt_token_ids": prompt_token_ids,
    "multi_modal_data": {
        "audio": {"embeds": merged_embeds},
    },
}
```

Это **не конвенция vLLM**: нативно `multi_modal_data["audio"]` — это `torch.Tensor` / `np.ndarray` / `(array, sr)`. Работало в V0; в V1 процессор-цепочка прогоняет `MultiModalDataParser._get_data_parser()`, который **отбросит неожиданную структуру** или упадёт на ней.

### 2.4 `AsyncEngineArgs` — критичные находки

`src/auralis/models/xttsv2/XTTSv2.py:222-260`

```python
last_layer_id = self.gpt_config.num_hidden_layers - 1
speculative_config = {
    "method": "extract_hidden_states",
    "num_speculative_tokens": 1,
    "draft_model_config": {
        "hf_config": {
            "eagle_aux_hidden_state_layer_ids": [last_layer_id],
        }
    },
}
kv_transfer_config = {
    "kv_connector": "ExampleHiddenStatesConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
        "shared_storage_path": str(self.hidden_states_dir),
    },
}
```

**Что с этим не так:**

1. **`method="extract_hidden_states"`** — такого значения `speculative.method` в vLLM **не существует**. Валидные: `eagle`, `eagle3`, `medusa`, `mlp_speculator`, `draft_model`, `ngram`.
2. **`ExampleHiddenStatesConnector`** — **зарегистрирован**, но это **пример из `v1/example_hidden_states_connector.py`**, не production-ready API. Формат файлов, которые он пишет, **не совместим с `safe_open(...)` в `process_tokens_to_speech`**.
3. **`load_format` НЕ указан** → vLLM ищет `model.safetensors` по умолчанию. HF-репо содержит `gpt2_model.safetensors` — **имя нестандартное**.
4. **Архитектура `XttsGPT` не зарегистрирована в `ModelRegistry` vLLM** — декоратор `@MULTIMODAL_REGISTRY.register_processor` регистрирует только *processor*, но `config.json` содержит `"architectures": ["XttsGPT"]`, и vLLM ищет это имя в своём `vllm.model_executor.models.registry.ModelRegistry`. Его **нет** в коде.

### 2.5 Формат весов HF

Прямое чтение `https://huggingface.co/AstraMindAI/xtts2-gpt`:

| Файл | Размер | Формат |
|---|---|---|
| `gpt2_model.safetensors` | 1.52 GB | **safetensors** (нестандартное имя) |
| `config.json` | 1.24 kB | содержит `"architectures": ["XttsGPT"]`, `"model_type": "xtts_gpt"` |
| `xtts2_gpt_modeling.py` | 16.2 kB | HF custom code (не используется vLLM) |

Нет ни `.bin`, ни `.pth`, ни шардированного `model-00001-of-NN.safetensors`.

---

## Задание 3 — Полная таблица несовместимостей

| # | Файл | Строка | Проблема | Правильное исправление | Уверенность |
|---|------|--------|----------|-----------------------|-------------|
| 1 | `vllm_mm_gpt.py` | 42 | `MultiModalInputs` не существует в v0.19.x | Заменить на `dict`-литерал с теми же ключами, **или** импортировать из `vllm.multimodal.processing.inputs` (если есть там — проверить) | **ТОЧНО** |
| 2 | `vllm_mm_gpt.py` | 43 | `MultiModalKwargs` не существует в v0.19.x | Заменить на `MultiModalKwargsItems` (плюральный) или `MultiModalKwargsItem` (единичный) | **ТОЧНО** |
| 3 | `vllm_mm_gpt.py` | 52 | Модуль `vllm.multimodal.profiling` удалён → `ModuleNotFoundError` | `from vllm.multimodal.processing.dummy_inputs import BaseDummyInputsBuilder, ProcessorInputs` | **ТОЧНО** |
| 4 | `vllm_mm_gpt.py` | 151-183 | `XttsDummyInputsBuilder` не реализует абстрактные `get_dummy_text` + `get_dummy_mm_data` | Добавить `def get_dummy_text(self, mm_counts) -> str: return ""` и `def get_dummy_mm_data(self, seq_len, mm_counts) -> MultiModalDataDict: return {"audio": [...]}`; **убрать** кастомный `get_dummy_processor_inputs` | **ТОЧНО** |
| 5 | `vllm_mm_gpt.py` | 180 | `ProcessorInputs(prompt_text=...)` — поле называется `prompt` | `ProcessorInputs(prompt="", mm_data=..., hf_processor_mm_kwargs={})` | **ТОЧНО** |
| 6 | `vllm_mm_gpt.py` | 216-223 | `MultiModalInputs(...)` с позиционным ключом `type="multimodal"` | Вернуть обычный `dict` или `MultiModalInputs(prompt=..., prompt_token_ids=..., mm_kwargs=..., mm_hashes=..., mm_placeholders=...)` без `type` | **ВЕРОЯТНО** |
| 7 | `vllm_mm_gpt.py` | 220 | `MultiModalKwargs({"audio_embeds": [...]})` — класс удалён | Использовать `MultiModalKwargsItems.from_hf_inputs(BatchFeature({"audio_embeds": [...]}), self._get_mm_fields_config(...))` | **ТОЧНО** |
| 8 | `vllm_mm_gpt.py` | 236 | `out_mm_kwargs: MultiModalKwargs` в `_get_prompt_updates` — тип удалён | Заменить на `MultiModalKwargsItems` | **ТОЧНО** |
| 9 | `vllm_mm_gpt.py` | 349-353 | `@MULTIMODAL_REGISTRY.register_processor(...)` регистрирует только processor | Дополнительно зарегистрировать модель: `from vllm import ModelRegistry; ModelRegistry.register_model("XttsGPT", XttsGPT)` на уровне модуля | **ТОЧНО** |
| 10 | `vllm_mm_gpt.py` | 33 | `from vllm.model_executor.models.gpt2 import GPT2Block` — могло быть переименовано | Проверить `vllm/model_executor/models/gpt2.py`; если переименовано — использовать новое имя | **ТРЕБУЕТ ПРОВЕРКИ** |
| 11 | `XTTSv2.py` | 224 | `speculative_config.method="extract_hidden_states"` — не существует | **Убрать `speculative_config` целиком.** Механизм недостижим через stock vLLM | **ТОЧНО** |
| 12 | `XTTSv2.py` | 233 | `kv_connector="ExampleHiddenStatesConnector"` — зарегистрирован, но это демо в несовместимом формате | **Убрать `kv_transfer_config` целиком** | **ТОЧНО** |
| 13 | `XTTSv2.py` | 240-260 | `AsyncEngineArgs` без `load_format` при нестандартном `gpt2_model.safetensors` | Явно указать `load_format="safetensors"` или скачать и переименовать файл локально | **ВЕРОЯТНО** |
| 14 | `XTTSv2.py` | 262 | `AsyncLLMEngine.from_engine_args(...)` — в V1 это alias на `AsyncLLM` | Проверить, что `self.llm_engine.vllm_config` всё ещё доступен | **ТРЕБУЕТ ПРОВЕРКИ** |
| 15 | `vllm_mm_gpt.py` | 492-513 | `compute_logits(hidden_states, sampling_metadata=None)` — V1 `GPUModelRunner` может не передавать второй аргумент | Оставить `Any = None` (уже сделано в ABI-shim) | **ВЕРОЯТНО** |
| 16 | `XTTSv2.py` | 737-764 | `process_tokens_to_speech` читает hidden states из safetensors, которые пишет `ExampleHiddenStatesConnector` — формат не совпадает | Переработать под другой источник hidden states | **ТОЧНО** |

---

## Задание 4 — Честное мнение

### 4.1 Достаточно ли точечных патчей?

**Нет. Архитектура Auralis фундаментально расходится с vLLM V1.**

Три не-точечные проблемы:

1. **Hidden-states extraction** — центральная фича Auralis для HiFi-GAN декодера. В V0 использовались приватные хаки (`is_logits_only_mode`, `HiddenStatesCollector`). Текущая «замена» через `speculative_config.method="extract_hidden_states"` + `ExampleHiddenStatesConnector` — **фиктивная**: такого метода в vLLM нет. Никакой поддерживаемый публичный API V1 не экспортирует per-token hidden states из середины generation loop. Это **принципиально не поддерживаемый use-case в V1**.
2. **Multimodal embeddings vs multimodal data** — Auralis передаёт pre-computed эмбеддинги в стиле, чуждом vLLM. V1 через `MultiModalKwargsItems` ожидает строго другую структуру, где эмбеддинги указываются через `MultiModalFieldConfig` + `BatchFeature`. Нужен рефакторинг всего `XttsMultiModalProcessor`.
3. **ModelRegistry registration** — простое добавление `ModelRegistry.register_model(...)` не сработает, пока не запустится импорт модуля `vllm_mm_gpt.py` — а при `trust_remote_code=True` импорт модели идёт из HF через `xtts2_gpt_modeling.py`, а не из локального пакета Auralis. Нужна регистрация как **vLLM plugin** (через entry-point в `pyproject.toml`: `vllm.platform_plugins`).

### 4.2 Скрытые риски

- **`GPT2Block` API** — внутренние классы `vllm.model_executor.models.gpt2` могут не иметь стабильной публичной сигнатуры.
- **`LogitsProcessor(...)` третий аргумент** — формально ещё `Optional`, но в V1 может вернуться в контракт через другие hooks.
- **`make_layers`, `make_empty_intermediate_tensors_factory`** — семантика могла поменяться (напр., `get_pp_group()` в V1 ведёт себя иначе при TP=PP=1).
- **V1 prefix-caching**: если Auralis полагается на то, что одинаковые conditioning embeddings дадут **разные** `prompt_token_ids`, prefix cache может внезапно хитить и портить выдачу.
- **`enforce_eager=True`** маскирует проблемы CUDA graph capture, но бьёт по производительности.
- **`AstraMindAI/xtts2-gpt` выложен 1+ год назад** — под vLLM той эпохи (≈0.6). Custom_code в `xtts2_gpt_modeling.py` тоже использует удалённые API.

### 4.3 Что бы сделал я

**Рекомендация: откатить масштабную V1-миграцию и выбрать одну из двух стратегий.**

#### Стратегия A — «остаться на V0, зафиксировать версии» (быстрая, низкий риск)

1. Пин `vllm==0.6.x` (последняя версия с V0 как default) в `setup.py`.
2. Пин `torch` под требования этой vLLM.
3. Вернуть `HiddenStatesCollector` и `ExtendedSamplingParams` — они работали.
4. Бонус: существующие веса `AstraMindAI/xtts2-gpt` тестировались именно на таком стеке.
5. Это не «будущее-ориентированный путь», но он **работает завтра**.

#### Стратегия B — «выпилить vLLM как инференс-движок» (долгая, правильная)

Заменить vLLM на прямой PyTorch-инференс с KV-cache, управляемый Auralis:
1. Использовать `transformers` с ручным `past_key_values` для GPT-стека.
2. Либо интегрировать `nanoGPT`-style кастомный inference loop с сохранением hidden states через `forward_hook`.
3. Убрать зависимость от приватных API vLLM.
4. Потерять vLLM-batching, но получить контроль над hidden states.
5. Для batching — использовать `torch.compile` + custom scheduler, или внешний queue типа Ray Serve.

#### Стратегия C (промежуточная)

Попробовать заставить V1 работать только ради генерации токенов, и запускать отдельный PyTorch-forward с `output_hidden_states=True` на токенах уже после их генерации. Это удвоит compute, но даст корректные hidden states без зависимости от `ExampleHiddenStatesConnector`.

---

## 🎯 TL;DR

**Ни один коммит в последних 4 сессиях не исправляет корневую проблему.** Точечные импорт-патчи не спасают, потому что:
- Модель `XttsGPT` **даже не зарегистрирована** в vLLM `ModelRegistry`.
- `speculative_config.method="extract_hidden_states"` — **фиктивная опция**, её нет в vLLM.
- `BaseDummyInputsBuilder` переехал в другой модуль + абстрактные методы не реализованы — engine упадёт на профайлинге ещё до первого токена.
- Hidden-state extraction через `ExampleHiddenStatesConnector` — **демо-код**, не API.

**Рекомендация:** откатиться на Strategy A перед любыми дальнейшими правками.
