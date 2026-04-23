# Streaming-запись аудио — аналитический отчёт

**Дата:** 2026-04-23
**Scope:** только анализ. Код не менялся.
**Цель:** определить, можно ли писать аудио на диск по чанкам (streaming) без полной буферизации в RAM.

---

## Краткий вердикт

**Streaming уже работает на уровне `generate_speech(..., stream=True)`.** Чанки `TTSOutput` yield'ятся наружу пользователю в арифметическом порядке, `combine_outputs()` при `stream=True` **не вызывается**. Для «записи WAV по чанкам на Google Drive» достаточно добавить **тонкий helper**, итерирующийся по streaming-генератору и пишущий через `soundfile.SoundFile` в `mode='w'`. **Модифицировать core не нужно.**

Рекомендуемый вариант — **D** (использовать существующий `stream=True`) + опциональный helper.

---

## 1. Текущее состояние API

### stream=True в `TTSRequest`

Поле объявлено в `src/auralis/common/definitions/requests.py:184`:
```python
stream: bool = False
```

### Как ветвится `generate_speech()`

`@c:\Users\kister\Documents\GitHub\Auralis\src\auralis\core\tts.py:414-466` (текущий код после коммита `841d341`):

- **stream=True** (строки 430-459): возвращает `Generator[TTSOutput, None, None]`. Внутри `streaming_wrapper()` итерируется по `split_requests(request)` **последовательно**, для каждого `sub_request` прогоняет `scheduler.run(...)` через `loop.run_until_complete(anext(...))` и **yield'ит чанк наружу** сразу по приходу. `combine_outputs` **не вызывается**.
- **stream=False** (строки 460-466): возвращает **один** `TTSOutput`. Внутри — `loop.run_until_complete(self._process_multiple_requests(requests))`, который через `asyncio.gather` собирает ВСЕ чанки в память, затем `TTSOutput.combine_outputs(complete_audio)` делает `np.concatenate`.

### `generate_speech_async()`

`@c:\Users\kister\Documents\GitHub\Auralis\src\auralis\core\tts.py:210-247`. Возвращает `AsyncGenerator[TTSOutput]` или один `TTSOutput`. Внутренний `process_chunks()` делает `async for chunk in self.scheduler.run(inputs=request, ...)`. **Важно:** здесь `inputs=request` — оригинальный request **без** `split_requests()`. При `stream=True` сразу yield'ит чанки, при `stream=False` — собирает и yield'ит один `combine_outputs(chunks)`.

### Является ли путь уже async-итерируемым

**Да.**
- `generate_speech_async(request)` с `stream=True` → `AsyncGenerator[TTSOutput]` (честный async).
- `generate_speech(request)` с `stream=True` → синхронный `Generator[TTSOutput]` (обёртка над async через `loop.run_until_complete(anext(...))`).

### Ответы на вопросы #1

| # | Вопрос | Ответ |
|---|---|---|
| 1a | Как ветвится код при stream? | sync: sequential split_requests → streaming_wrapper yield; async: scheduler.run yield async. Оба branches **не собирают буфер** |
| 1b | generate_speech возвращает generator или List? | При stream=True — `Generator[TTSOutput]`. При stream=False — единственный `TTSOutput` (не List) |
| 1c | Есть ли async-итерируемый путь? | **Да** — `generate_speech_async(stream=True)` |

---

## 2. `combine_outputs()` — точная реализация

`@c:\Users\kister\Documents\GitHub\Auralis\src\auralis\common\definitions\output.py:105-122`:

```python
@staticmethod
def combine_outputs(outputs: List['TTSOutput']) -> 'TTSOutput':
    combined_audio = np.concatenate([out.array for out in outputs])
    return TTSOutput(
        array=combined_audio,
        sample_rate=outputs[0].sample_rate
    )
```

- **Только `np.concatenate`** — никаких ресемплов, кроссфейдов, нормализации.
- **`sample_rate` берётся от первого чанка.**
- Вызывается **только при `stream=False`** (`tts.py:227` в async-пути, `tts.py:331` в sync-пути).
- При `stream=True` **не вызывается** — наружу уходят сырые `TTSOutput` по одному.

---

## 3. Существующий streaming API — grep-результаты

```
tts.py:9    from typing import AsyncGenerator, Callable, Optional, Dict, Union, Generator, List
tts.py:196  async def generate_speech_async(self, request: TTSRequest) -> Union[AsyncGenerator[TTSOutput, None], TTSOutput]
tts.py:174  async for chunk in self.tts_engine.process_tokens_to_speech(...)        # engine level
tts.py:178              yield chunk
tts.py:194              yield chunk                                                  # _second_phase_fn
tts.py:213  async for chunk in self.scheduler.run(...)
tts.py:220              yield chunk                                                  # async path, stream=True
tts.py:227  yield TTSOutput.combine_outputs(chunks)                                 # async path, stream=False
tts.py:272  async for chunk in self.scheduler.run(...)                              # _process_multiple_requests
tts.py:443  async for chunk in self.scheduler.run(...)                              # streaming_wrapper
tts.py:444                              yield chunk
tts.py:455                          yield chunk                                      # sync streaming
```

**Вывод:** есть 3 уровня `yield`:
1. **Engine level** — `XTTSv2.process_tokens_to_speech` yield'ит `TTSOutput` после `output.finished`.
2. **Scheduler level** — `scheduler.run` async-итерирует второй фазой.
3. **Top API level** — `generate_speech[_async]` пробрасывает yield наружу при `stream=True`.

---

## 4. `soundfile` — проверка

```
python -c "import soundfile; print(soundfile.__version__)"
→ soundfile 0.13.1  ✅
```

### Smoke-тест streaming WAV в BytesIO (seekable)

```python
import soundfile as sf, numpy as np, io
buf = io.BytesIO()
with sf.SoundFile(buf, mode='w', samplerate=24000, channels=1, format='WAV') as f:
    f.write(np.zeros(1000, dtype=np.float32))
    f.write(np.zeros(1000, dtype=np.float32))
# Результат: 4044 bytes, roundtrip → 2000 samples @ 24000 Hz ✅
```

### Smoke-тест real file

```python
with sf.SoundFile('test.wav', mode='w', samplerate=24000, channels=1, format='WAV') as f:
    for _ in range(5): f.write(np.zeros(1000, dtype=np.float32))
# Roundtrip: 5000 samples, 24000 Hz ✅
```

### Smoke-тест FLAC (streaming-safe формат)

```python
with sf.SoundFile('test.flac', mode='w', samplerate=24000, channels=1, format='FLAC') as f:
    for _ in range(5): f.write(np.zeros(1000, dtype=np.float32))
# 110 bytes (pure silence сжат), 5000 samples ✅
```

**Все три сценария работают.** `soundfile` корректно обновляет WAV-header на закрытии (seek back + rewrite RIFF chunk size).

---

## 5. `TTSOutput.save()` — как работает сейчас

`@c:\Users\kister\Documents\GitHub\Auralis\src\auralis\common\definitions\output.py:200-233`:

```python
def save(self, filename, sample_rate=None, format=None):
    wav_tensor = self.to_tensor()
    if wav_tensor.dim() == 1: wav_tensor = wav_tensor.unsqueeze(0)
    if sample_rate and sample_rate != self.sample_rate:
        wav_tensor = torchaudio.functional.resample(...)
    torchaudio.save(filename, wav_tensor, sample_rate, format=format, ...)
```

- **Принимает путь напрямую** — да (`str | Path`).
- **Можно вызвать на одном чанке** — технически да, но запишет только его (не append).
- **Append-режим** — **нет**. Каждый вызов `save()` создаёт новый файл через `torchaudio.save`, который всегда пишет целиком.
- Резолвит формат по расширению через torchaudio.

**Вывод:** `save()` НЕ подходит для streaming-append. Нужен отдельный helper через `soundfile.SoundFile(mode='w')` контекст.

---

## 6. Asyncio-контекст

### `quickstart.ipynb` — как вызывается `generate_speech`

`@c:\Users\kister\Documents\GitHub\Auralis\notebooks\colab_quickstart.ipynb`:

```python
# Cell 3:
output = tts.generate_speech(request)   # ← sync-вызов
output.save("out.wav")

# Cell 4 (shutdown):
import asyncio
asyncio.get_event_loop().run_until_complete(tts.shutdown())
```

- **Синхронный вызов** `tts.generate_speech(request)` — Jupyter автоматически управляет event loop, но метод сам создаёт/находит loop через `_ensure_event_loop()` (`tts.py:57-65`).
- `shutdown()` — async, вызывается через `run_until_complete` явно.
- **Колаб-ячейки по умолчанию не asyncio-контекст.** Для streaming-варианта пользователь просто итерирует `for chunk in tts.generate_speech(request_stream):` синхронно.

---

## Найденные камни

### А. Sample rate — общий для всех чанков?

**Да, 24000 Hz гарантированно.**

- `TTSOutput.sample_rate: int = 24000` — дефолт в `@dataclass` (`output.py:20`).
- `XTTSv2.process_tokens_to_speech` (`XTTSv2.py:809`) yield'ит `TTSOutput(array=wav, start_time=..., token_length=...)` — **sample_rate не передаётся** → берётся дефолт 24000.
- HiFi-GAN decoder сконфигурирован через `hifi_config.output_sample_rate` (`XTTSv2.py:129`), который в XTTSv2 конфиге = 24000.
- Значения 22050 в `XTTSv2.py:359-399` относятся к **reference audio loading** (speaker embeddings), не к выходному вокодеру.

**Обход:** не нужен. Все чанки @ 24000 Hz. Но в коде streaming-helper'а лучше брать `sr` из **первого чанка** (не хардкодить), чтобы не сломаться при будущем добавлении моделей.

### Б. Seekable vs pipe

**Обход найден.** `soundfile` требует seek только для WAV-header-fixup на close. На Colab файлы пишутся на `/content/` (local SSD) или на `/content/drive/MyDrive/` (Google Drive-mount) — оба **seekable** POSIX-ом. `BytesIO` тоже seekable (верифицировано).

**Реальный риск:** только если писать напрямую в `sys.stdout.buffer` (non-seekable pipe) — для этого WAV не подойдёт, нужен FLAC. Но этот кейс не планируется.

### В. Concurrency и порядок чанков

- **stream=True sync path** (`tts.py:430-459`): итерирует `split_requests` **последовательно** (`for sub_request in requests:`) → чанки прилетают строго в порядке текста. Нет concurrency. **Lock не нужен.**
- **stream=True async path** (`tts.py:210-227`): вызывает `scheduler.run(inputs=request, ...)` на целом request (без split). Scheduler внутри может запустить несколько `audio_token_generators` параллельно, НО все чанки async-иterируются через **один** async-for → они сериализуются на уровне генератора. **Lock не нужен.**
- **stream=False**: `_process_multiple_requests` → `asyncio.gather(*tasks)` → возвращает список **в порядке создания tasks** (Python-гарантия, не as-completed). `complete_audio = [chunk for chunks in all_chunks for chunk in chunks]` — flatten сохраняет порядок sub_request'ов. **Order-safe.**

**Вывод:** concurrency не создаёт race на писателе, если писать из одного потребителя streaming-генератора. Lock не нужен.

### Г. Частичный файл при краше

**Риск: WAV-header не обновится, если процесс умирает между `.write()` и `close()`.**

- Внутри `with sf.SoundFile(...) as f:` при нормальном exit header пишется корректно.
- При exception Python вызывает `__exit__` → soundfile seeks + rewrites header → OK.
- **При SIGKILL / crash Python'а / потере Colab-runtime** header останется с инициальным размером (`0xFFFFFFFF` или 0). Некоторые плееры восстановят длину из фактического размера файла, некоторые нет. **Audacity, ffmpeg восстанавливают** (есть утилиты типа `ffmpeg -f wav -i broken.wav -c copy fixed.wav`).

**Обходы (по убыванию complexity):**
1. **FLAC вместо WAV** — self-delimited фреймы, любой обрыв даёт воспроизводимый частичный файл (на проигрываемую длину). Overhead encode ~5-10% CPU, но на GPU-bottlenecked пайплайне незаметно.
2. **Flush каждые N чанков + re-seek header** — сложно, soundfile не экспонирует API для этого.
3. **Писать raw float32 в `.f32` + sidecar `.json` с sr/channels** — потом конвертить. Максимально crash-safe, но нужна конвертация.

**Рекомендация:** выбор формата оставить пользователю через параметр. Дефолт — **WAV** (совместимость). Для длинных генераций (> 10 минут, риск дисконнекта Colab) — **FLAC**.

---

## Оценка вариантов

| Вариант | Что | Плюсы | Риски / Минусы |
|---|---|---|---|
| **A** | `soundfile.SoundFile('out.wav', 'w')` + итерация `stream=True` в helper'е | Простая реализация; seekable файл → header fix on close OK; формат WAV совместимый | WAV-header при crash'е (частичный риск, обходится ffmpeg recovery); зависим от `soundfile` (уже в `pip install` блоке quickstart) |
| **B** | То же, но format=FLAC | Streaming-safe при любом crash'е; меньше размер (~50% от WAV) | Encode overhead (~5-10% CPU); FLAC декодеры есть везде, но меньше совместимы с нативными плеерами Windows |
| **C** | Patch `combine_outputs` → писать в `tempfile`/`mmap` вместо `np.concatenate` | Zero RAM-copy для больших генераций | Ломает API (`combine_outputs` возвращает `TTSOutput` с `array: np.ndarray`, а не с file-handle); трогает core-логику; низкая ценность при существующем `stream=True` |
| **D** | Использовать готовый `generate_speech(stream=True)` + helper `save_stream(path)` | **Zero изменений в core generation**; leverage существующего API; reuse чанков из `_second_phase_fn`; совместимо с rich.Progress из прошлой сессии | Нужно добавить helper в public API (низкий риск); пользователь должен знать про `stream=True` |

---

## Рекомендуемый вариант

### **Вариант D** — тонкий helper поверх существующего `stream=True`.

**Обоснование:**
1. `stream=True` путь уже протестирован и протёк через scheduler + progress-bar (текущая сессия `841d341`).
2. Core generation logic не трогается (user explicitly запрещал в прошлых заданиях).
3. `combine_outputs` остаётся как есть — используется в не-streaming пути без изменений.
4. `soundfile` уже в `requirements.txt` / `quickstart.ipynb:56` (`... sounddevice soundfile "spacy==3.7.5" ...`).
5. Helper можно сделать **методом на TTS** или **standalone функцией** в `tts.py` / в новом `auralis/io/streaming.py`. Публичный API вырастает на одну функцию.

### Реализация (псевдокод, НЕ применять)

```python
# src/auralis/core/tts.py (или отдельный модуль)

def save_stream(
    self,
    request: TTSRequest,
    filename: Union[str, Path],
    format: Optional[str] = None,
    progress: bool = True,
) -> dict:
    """Generate speech and stream-write to a file chunk-by-chunk.

    Does NOT buffer the full audio in RAM. Safe for arbitrarily long
    texts (books, podcasts).

    Args:
        request: TTSRequest. If request.stream is False, it is set to True
            transparently for the duration of this call.
        filename: Target path. Format inferred from extension ('.wav', '.flac').
        format: Optional explicit format override (soundfile codes).
        progress: Show a rich.Progress bar (default True).

    Returns:
        dict with keys 'path', 'sample_rate', 'n_samples', 'duration_sec',
        'wall_sec', 'rtf'.

    Raises:
        RuntimeError: if generation produces zero chunks.
    """
    import soundfile as sf  # already a dependency; lazy-import for cold start

    # Force stream=True without mutating user's object permanently
    streaming_request = request.copy() if hasattr(request, 'copy') else request
    streaming_request.stream = True

    fmt = format or Path(filename).suffix.lstrip('.').upper() or 'WAV'
    # Map common extensions to soundfile format codes
    fmt_map = {'WAV': 'WAV', 'FLAC': 'FLAC', 'OGG': 'OGG'}
    fmt = fmt_map.get(fmt, fmt)

    start_wall = time.time()
    sr: Optional[int] = None
    total_samples = 0
    sf_file: Optional[sf.SoundFile] = None

    description = self._make_progress_description(request)
    ctx = (
        self._progress_context(len(self.split_requests(streaming_request)), description)
        if progress
        else _nullcontext(_noop)
    )

    try:
        with ctx as advance:
            for chunk in self.generate_speech(streaming_request):
                # Lazy-open file on first chunk so sr is known from actual output
                if sf_file is None:
                    sr = chunk.sample_rate
                    sf_file = sf.SoundFile(
                        str(filename),
                        mode='w',
                        samplerate=sr,
                        channels=chunk.array.ndim if chunk.array.ndim > 1 else 1,
                        format=fmt,
                        subtype='PCM_16' if fmt == 'WAV' else None,
                    )
                # soundfile expects float32 in [-1, 1] or int16 PCM
                arr = chunk.array
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                sf_file.write(arr)
                total_samples += arr.shape[0]
                advance(chunk)  # feed the progress bar
    finally:
        if sf_file is not None:
            sf_file.close()  # finalizes WAV header / FLAC stream

    if total_samples == 0:
        raise RuntimeError('generate_speech yielded zero chunks')

    wall = time.time() - start_wall
    duration = total_samples / sr
    return {
        'path': str(filename),
        'sample_rate': sr,
        'n_samples': total_samples,
        'duration_sec': duration,
        'wall_sec': wall,
        'rtf': wall / duration if duration > 0 else 0.0,
    }
```

**Мелочи реализации:**
- Lazy-open file на первом чанке, чтобы взять реальный `sample_rate` от модели, а не хардкодить.
- `advance(chunk)` переиспользует progress из прошлой сессии. Но **conflict**: текущий `_progress_context` сам печатает summary на exit. Для `save_stream` этот summary будет напечатан **до** получения финального dict со статистикой — надо либо:
  - добавить флаг `_progress_context(..., print_summary=False)` и печатать summary из `save_stream`;
  - либо пусть печатается как есть (дублирование в stdout — не критично).
- `request.copy()` — убедиться, что у `TTSRequest` есть метод `copy` (в tts.py:267 он используется). Проверка показывает `copy := request.copy()` — да, есть.

---

## Остающиеся риски

1. **Формат WAV при crash'е:** частичный файл может не проиграться строгими плеерами. **Обход:** документировать рекомендацию FLAC для длинных генераций, либо добавить пост-recovery в `auralis.io.fix_wav_header()`.
2. **`request.copy()`** — если пользователь передаст ссылку на Pydantic v1/v2 или на какой-то mutable state, copy может быть shallow → `stream=True` утечёт в оригинал. **Обход:** использовать `dataclasses.replace` / explicit mutation только локально + `finally: request.stream = original_stream`.
3. **Progress-bar conflict с `_progress_context`** — см. мелочь выше. **Обход:** параметризовать `print_summary`.
4. **Sample rate hardcoded default:** если в будущем появится модель с `sr != 24000` (Kokoro / OpenVoice), hardcode в helper сломается. **Обход:** lazy-open файла на первом чанке (уже заложено в псевдокод).
5. **Первый чанк долгий → файл создаётся поздно.** Пока идёт Phase 1 (GPT cond + vLLM add_request), file handle не открыт → SIGKILL до первого чанка = нет файла вовсе. **Приемлемо** (у WAV до первого чанка и нечего было бы писать).
6. **`scheduler.run` перепоглощает все sub-requests внутри `stream=True` через `asyncio` loop.** Если юзер сам уже внутри async-контекста (FastAPI / jupyter-async), sync вариант упадёт на `loop.run_until_complete`. **Обход:** `save_stream_async` parallel версия — это отдельный helper.

---

## Итоговая матрица решения

| Критерий | A (WAV) | B (FLAC) | C (mmap patch) | **D (helper + stream)** |
|---|---|---|---|---|
| Изменений в core | 0 | 0 | **много** | **0** |
| Crash-safe | ~ | ✅ | ✅ | зависит от format |
| RAM usage | O(1) | O(1) | O(1) | **O(1)** |
| Время реализации | 1ч | 1ч | 1 день | **1ч** |
| Overhead CPU | ~ | +5-10% | ~ | **~ (от format)** |
| Public API ломается | 0 | 0 | **да** | **0** (добавление метода) |

**Вариант D** побеждает по всем осям, даёт пользователю выбор формата через расширение файла.

---

**Анализ завершён. Жду подтверждения на реализацию варианта D.**
