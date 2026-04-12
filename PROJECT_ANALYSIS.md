# Auralis: анализ директории

## Кратко

`Auralis` - Python-пакет для text-to-speech и voice cloning на базе XTTSv2 и vLLM. Проект предоставляет Python API, OpenAI-compatible FastAPI сервер, примеры, документацию и набор интеграционных тестов.

Основной сценарий из README:

```python
from auralis import TTS, TTSRequest

tts = TTS().from_pretrained("AstraMindAI/xttsv2", gpt_model="AstraMindAI/xtts2-gpt")

request = TTSRequest(
    text="Hello Earth! This is Auralis speaking.",
    speaker_files=["reference.wav"],
)

output = tts.generate_speech(request)
output.save("hello.wav")
```

## Структура проекта

```text
Auralis-main/
├── src/auralis/              # Основной исходный код пакета
│   ├── core/                 # Публичный фасад TTS
│   ├── common/               # DTO, output, logging, scheduler, metrics, utilities
│   ├── models/               # Базовые классы и XTTSv2 implementation
│   └── entrypoints/          # CLI/FastAPI entrypoints
├── tests/                    # Unit и integration tests
├── examples/                 # Примеры использования API, OpenAI server, Gradio
├── docs/                     # MkDocs-документация
├── Dockerfile                # Контейнер для запуска OpenAI-compatible server
├── requirements.txt          # Список зависимостей
├── setup.py                  # Packaging и console script
└── README.md                 # Основное описание проекта
```

## Главные модули

### Публичный API

Файл: `src/auralis/__init__.py`

Экспортирует:

- `TTS`
- `TTSRequest`
- `TTSOutput`
- `AudioPreprocessingConfig`
- `setup_logger`
- `set_vllm_logging_level`

### TTS facade

Файл: `src/auralis/core/tts.py`

Класс `TTS` отвечает за:

- загрузку модели через `from_pretrained`;
- sync и async генерацию речи;
- streaming output;
- разбиение длинного текста на несколько запросов;
- передачу задач в двухфазный scheduler.

Ключевые методы:

- `from_pretrained(model_name_or_path, **kwargs)`
- `generate_speech_async(request)`
- `generate_speech(request)`
- `split_requests(request, max_length=100000)`
- `shutdown()`

### Модель XTTSv2

Файл: `src/auralis/models/xttsv2/XTTSv2.py`

Класс `XTTSv2Engine` реализует основной TTS engine:

- использует `vllm.AsyncLLMEngine` для генерации audio tokens;
- содержит conditioning encoder;
- содержит HiFi-GAN decoder;
- поддерживает speaker embeddings и GPT-like decoder conditioning;
- оценивает использование GPU memory в зависимости от concurrency.

### Scheduler

Файл: `src/auralis/common/scheduling/two_phase_scheduler.py`

`TwoPhaseScheduler` запускает генерацию в две фазы:

1. Первая фаза готовит контекст генерации.
2. Вторая фаза параллельно обрабатывает генераторы и собирает output.

Это важная часть проекта, потому что Auralis заявляет быстрый async/multi-request режим.

### Request/output модели

Файл: `src/auralis/common/definitions/requests.py`

`TTSRequest` хранит:

- текст;
- reference audio / speaker files;
- язык;
- параметры voice conditioning;
- параметры sampling;
- настройки speech enhancement;
- флаг streaming.

Файл: `src/auralis/common/definitions/output.py`

`TTSOutput` хранит audio array и умеет:

- сохранять аудио;
- конвертировать в bytes;
- менять скорость;
- делать resample;
- объединять несколько outputs;
- проигрывать/превьюить аудио.

### OpenAI-compatible server

Файл: `src/auralis/entrypoints/oai_server.py`

Сервер предоставляет:

- `POST /v1/audio/speech`
- `POST /v1/chat/completions`

Console script из `setup.py`:

```bash
auralis.openai --host 127.0.0.1 --port 8000 --model AstraMindAI/xttsv2 --gpt_model AstraMindAI/xtts2-gpt
```

## Зависимости и запуск

Проект требует Python `>=3.10`.

Важная деталь: `setup.py` содержит проверку платформы и падает, если система не Linux. Это связано с `vllm`, который в данном проекте закреплен как:

```text
vllm==0.6.4.post1
```

На Windows запускать напрямую, скорее всего, не получится. Реалистичные варианты:

- WSL2 с Linux;
- Linux-машина;
- Docker/Linux контейнер с доступом к NVIDIA GPU;
- удаленный Linux/GPU сервер.

## Colab и CUDA-совместимость

По симптомам из Colab проблема выглядит не как поломка Auralis-логики, а как конфликт бинарных CUDA-сборок в окружении.

Наблюдение:

- `torchaudio` ищет `libcudart.so.13`;
- `torch` и бинарники `vllm` собраны вокруг `libcudart.so.12`;
- `vllm` и `xformers` дополнительно могут не находить `libtorch*.so`, если не настроен `LD_LIBRARY_PATH`.

Итог: в окружении смешаны пакеты, собранные под разные CUDA ABI. Если исправить только один пакет, импорт может упасть на следующем бинарном модуле.

Для линии `torch==2.5.1` / `torchaudio==2.5.1` официальная PyTorch previous versions matrix указывает согласованные wheel-наборы для:

- `cu118`;
- `cu121`;
- `cu124`;
- `cpu`.

В этой матрице нет `cu130` для `torch==2.5.1` / `torchaudio==2.5.1`.

Источник: https://pytorch.org/get-started/previous-versions/

Практический вывод для Colab:

1. Не смешивать `torch`, `torchaudio`, `torchvision`, `xformers` и `vllm` из разных CUDA-линий.
2. Выбрать одну CUDA-линию, вероятнее всего `cu124` для старой связки PyTorch 2.5.1.
3. После переустановки проверять не только `import torch`, но и:

```python
import torch
import torchaudio
import vllm

print(torch.__version__)
print(torch.version.cuda)
print(torchaudio.__version__)
```

4. Если падает `libtorch*.so`, проверить путь к torch libraries и добавить его в `LD_LIBRARY_PATH`.
5. Если Colab работает на Python 3.12 и конкретная старая связка не имеет подходящих Linux wheels, лучше откатиться на окружение с Python 3.10/3.11 или запускать через собственный контейнер/VM.

## Потенциальные проблемы

### 1. Не все серверные зависимости указаны

Код импортирует:

- `fastapi`
- `uvicorn`
- `aiohttp`
- `pydantic`
- `openai`

Но они не указаны явно в `requirements.txt` и `setup.py`. Поэтому OpenAI-compatible server может не стартовать после обычной установки без ручной доустановки этих пакетов.

### 2. Возможная проблема с регистрацией модели

`TTS.from_pretrained()` берет модель из `MODEL_REGISTRY`, но регистрация `"xtts"` происходит как side effect при импорте `auralis.models.xttsv2`.

Публичный `src/auralis/__init__.py` не импортирует `auralis.models`, поэтому сценарий из README может получить пустой registry, если модельный модуль не был импортирован раньше.

### 3. Документация местами устарела

В docs встречается API вида:

```python
tts.generate("Hello, world!")
```

Но в реальном классе `TTS` есть методы:

- `generate_speech(...)`
- `generate_speech_async(...)`

Метода `generate(...)` в найденном коде нет.

### 4. Возможный баг в Pydantic-модели

В `src/auralis/common/definitions/openai.py` поле `speed` объявлено с запятой в конце:

```python
speed: float = Field(default=1.0, description="List of base64-encoded audio files"),
```

Из-за запятой это может стать tuple, а не обычным Pydantic field. Это риск для endpoint `/v1/audio/speech`.

### 5. Возможная проблема streaming scheduler

В `TwoPhaseScheduler._yield_ordered_outputs()` индекс `current_index` увеличивается после одного элемента из buffer. Если один generator возвращает несколько чанков, scheduler может перейти к следующему generator раньше, чем полностью отдаст текущий buffer.

Это стоит проверить отдельным unit-тестом на multi-chunk generator.

### 6. Unit-тесты почти отсутствуют

`tests/unit/test_tts.py` содержит только `# TODO`. Основные проверки лежат в integration tests и требуют модели/GPU.

## Проверка

Была выполнена синтаксическая проверка:

```bash
python -m compileall -q src tests
```

Результат: синтаксических ошибок не найдено.

Полные тесты не запускались, потому что проект требует тяжелых ML-зависимостей, Linux/vLLM и, по смыслу, GPU и загрузку моделей с Hugging Face.

## Что делать дальше

Рекомендуемый порядок, если нужно довести проект до запуска:

1. Запускать в Linux/WSL2/Docker окружении с Python 3.10.
2. Доустановить недостающие серверные зависимости: `fastapi`, `uvicorn`, `aiohttp`, `pydantic`, `openai`.
3. Проверить регистрацию модели при `from auralis import TTS`.
4. Исправить поле `speed` в OpenAI request model.
5. Обновить docs, где используется несуществующий `tts.generate(...)`.
6. Добавить unit-тесты на `TTSRequest`, `TTSOutput`, model registry и `TwoPhaseScheduler`.
