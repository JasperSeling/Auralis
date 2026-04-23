# Auralis — Отчёт по разведке

**Дата:** 2026-04-23
**Версия кода:** upstream v0.2.8.post2 + коммиты `ed7b684` (revert) + `9d5f089` (logging fix)
**Примечание:** путь `/content/auralis_report.md` был в задании, но мы на Windows; отчёт сохранён в `docs/auralis_report.md` (версионируется).

---

## ⚠️ Главное открытие

**В пайплайне Auralis `print()` НЕ является источником per-chunk-спама.** Весь спам вида `Added request X` / `Finished request X` / `Starting request` шёл через `logger.info` (от vLLM и нашего scheduler) — **устранён коммитом `9d5f089`** в прошлой сессии.

Существующие `print()` в коде — это одноразовые сообщения при загрузке модели, CLI-утилиты и error-fallbacks. Заменять их на `rich.Progress` / `tqdm` **не имеет смысла** — они не представляют собой прогресс чанков.

Ниже — полная классификация найденных `print()`.

---

## 1. Print-вызовы

| Файл | Строка | Текст | Тип |
|---|---:|---|---|
| `models/xttsv2/utils/checkpoint_converter.py` | 97 | `Saved XTTSv2 GPT-2 weights to {path}` | **CLI утилита** (offline, `python checkpoint_converter.py ...`) |
| `checkpoint_converter.py` | 98 | `XTTSv2 GPT-2 weights: {keys}` | **CLI утилита** |
| `checkpoint_converter.py` | 105 | `Saved XTTSv2 weights to {path}` | **CLI утилита** |
| `checkpoint_converter.py` | 106 | `XTTSv2 weights: {keys}` | **CLI утилита** |
| `checkpoint_converter.py` | 118 | `Error: Checkpoint file ... does not exist` | **CLI ошибка** |
| `models/xttsv2/components/tts/layers/xtts/zh_num2words.py` | 935, 943, 956, 963, 971, 980, 988, 996, 1004, 1012 | `# print('date')` и т.п. | **Закомментированы** (debug-следы) |
| `zh_num2words.py` | 1110 | `WARNING: illegal char {c} in: {text}` (stderr) | **Сторонняя text-norm lib, warning** |
| `zh_num2words.py` | 1173, 1183, 1187, 1202, 1204, 1208, 1209 | text-norm статистика | **CLI-режим** (под `if __name__=='__main__'`) |
| `models/xttsv2/components/tts/layers/xtts/hifigan_decoder.py` | 280 | `Removing weight norm...` | **Модель-init (one-shot)** |
| `hifigan_decoder.py` | 438 | ` \| > Layer missing in the model definition: {k}` | **Модель-load (one-shot)** |
| `hifigan_decoder.py` | 448 | ` \| > {N} / {M} layers are restored.` | **Модель-load (one-shot)** |
| `hifigan_decoder.py` | 659 | ` > Model fully restored.` | **Модель-load (one-shot)** |
| `hifigan_decoder.py` | 665 | ` > Partial model initialization.` | **Модель-load (one-shot)** |
| `hifigan_decoder.py` | 676 | ` > Criterion load ignored because of: ...` | **Модель-load (one-shot)** |
| `common/definitions/output.py` | 328 | `Could not display audio widget: {e}` | **Jupyter fallback (error)** |
| `output.py` | 329 | `Try using .play() method instead` | **Jupyter fallback (hint)** |
| `output.py` | 340 | `Error playing audio: {e}` | **Playback error** |

### Классификация по категориям задания

- **Прогресс чанков → заменить на tqdm/rich:** **0 вызовов.** Таких `print()` в коде нет.
- **Метрики (throughput, latency, RTF) → rich Panel:** **0 вызовов.** Метрики уже идут через `logger.info` в `performance.py:144` (переформатированы в прошлой сессии в виде `🔊 287 tok/s | 1.2 req/s | RTF 0.21x`).
- **Ошибки → оставить:** **3 вызова** (`output.py:328, 329, 340`) + 1 stderr warning (`zh_num2words.py:1110`) + 1 CLI error (`checkpoint_converter.py:118`). Оставлены как есть.
- **Одноразовые при загрузке модели:** **6 вызовов** в `hifigan_decoder.py`. Опционально можно route через `logger.info` для единообразия — но это не прогресс-бар. См. «Рекомендации» ниже.
- **CLI-утилиты вне рантайм-пути:** **12 вызовов** в `checkpoint_converter.py` и `zh_num2words.py`. **Не трогаем** — пользователь их явно запускает как CLI, это их единственный канал вывода.

---

## 2. Доступные пакеты

| Пакет | Версия (Windows Python 3.10) | В Colab через vLLM 0.6.4.post1 |
|---|---|---|
| `rich` | **14.3.3** ✅ | ✅ (vLLM тянет как транзитивную зависимость) |
| `tqdm` | **4.67.3** ✅ | ✅ (также транзитивная от vLLM / transformers) |
| `loguru` | ❌ НЕ установлен | ❌ Нет ни в `auralis/setup.py`, ни в `vllm`. Не использовать. |

**Вывод:** `rich` и `tqdm` доступны без добавления новых зависимостей. `loguru` не использовать.

---

## 3. Пайплайн генерации

Полная трассировка вызовов от `TTS.generate_speech(request)` до финального `TTSOutput`:

```
TTSRequest(text, speaker_files, stream=...)
│
▼
TTS.generate_speech()                                        [core/tts.py:310]
│
├── split_requests(request, max_length=100000)               [core/tts.py:236]
│   └── ← List[TTSRequest]                                   (делит text по 100k символов)
│
├── [stream=False] _process_multiple_requests(requests)      [core/tts.py:257]
│   │
│   ├── для каждого sub_request:
│   │   └── scheduler.run(...)                               [two_phase_scheduler.py]
│   │       │
│   │       ├── Phase 1: _prepare_generation_context()       [core/tts.py:107]
│   │       │   └── tts_engine.get_generation_context()      [XTTSv2.py:689]
│   │       │       ├── GPT cond encoder → speaker_embeddings
│   │       │       ├── perceiver resampler → multimodal_data
│   │       │       └── vllm.AsyncLLMEngine.add_request()
│   │       │           → audio_token_generators (AsyncGenerator[RequestOutput])
│   │       │
│   │       └── Phase 2: _second_phase_fn()                  [core/tts.py:184]
│   │           │  @track_generation  ← emits "🔊 tok/s | req/s | RTF" каждые 5 сек
│   │           │
│   │           └── _process_single_generator()              [core/tts.py:160]
│   │               └── tts_engine.process_tokens_to_speech()[XTTSv2.py:761]
│   │                   │
│   │                   ├── async for output in generator:   ← vLLM стримит токены
│   │                   │   │
│   │                   │   ├── if output.finished:          ← ждёт EOS
│   │                   │   │   │
│   │                   │   │   ├── self.get_model_logits(tokens)
│   │                   │   │   │   ⚠️ПРОГОН GPT ЕЩЁ РАЗ для hidden_states
│   │                   │   │   │
│   │                   │   │   ├── self.hifigan_decoder(hidden_states, g=spk)
│   │                   │   │   │   ← asyncio.to_thread, decoder_semaphore, cuda_memory_manager
│   │                   │   │   │
│   │                   │   │   └── yield TTSOutput(
│   │                   │   │           array=wav.cpu().numpy().squeeze(),
│   │                   │   │           sample_rate=24000,
│   │                   │   │           start_time=..., token_length=...
│   │                   │   │       )
│   │
│   ├── asyncio.gather(*tasks) → List[List[TTSOutput]]
│   ├── complete_audio = flatten                              [core/tts.py:307]
│   └── TTSOutput.combine_outputs(complete_audio)             [output.py:106]
│
└── return TTSOutput
```

### Ключевые наблюдения

- **Text splitting по 100k символов** (`tts.py:236`) — это грубое верхнее ограничение. Реальные «чанки», которые идёт пользователь, возникают на уровне vLLM token-streaming (EOS per sub-request).
- **Phase 1 не зависит от Phase 2** — оба могут работать параллельно для разных sub_request'ов через `scheduler.second_phase_concurrency`.
- **`@track_generation`** (декоратор в `performance.py:105`) оборачивает `_second_phase_fn` и логирует метрики каждые 5 сек.

---

## 4. Хранение аудио

### Формат на каждом этапе

| Этап | Тип | Где |
|---|---|---|
| vLLM output | `RequestOutput.outputs[0].token_ids: List[int]` | vLLM внутри |
| Hidden states | `torch.Tensor` shape `[1, T, hidden_dim]` | `get_model_logits()` |
| HiFi-GAN output | `torch.Tensor` на GPU | `hifigan_decoder(...)` |
| После `.cpu().detach().numpy().squeeze()` | `np.ndarray` 1D float32 @ 24000 Hz | `XTTSv2.py:805` |
| `TTSOutput.array` | `np.ndarray` 1D | `output.py:21-31` |
| После `combine_outputs` | `np.ndarray` через `np.concatenate` | `output.py:116` |
| `save()` | Конвертация в `torch.Tensor` → `torchaudio.save(buffer, ...)` | `output.py:200-283` |

### Склейка

Просто: `np.concatenate([out.array for out in outputs])` в `combine_outputs()` на `output.py:116`. Один вызов, все чанки в памяти одновременно. Для «всей Гарри Поттер книги за 10 минут» это ~30 минут аудио @ 24кГц float32 ≈ **170 MB RAM** — приемлемо.

### `save()`

`output.py:200` → конвертит numpy в torch tensor → ресемплит если надо → делегирует в `torchaudio.save()` с форматом из расширения (wav / mp3 / flac / opus / aac / pcm). Для PCM возвращает сырые байты через `tobytes()`, минуя torchaudio.

**Потенциальная засада:** `combine_outputs` использует `sample_rate` **первого** `TTSOutput`. Если чанки на разных частотах (их быть не должно, но теоретически) — частота молча берётся от первого.

---

## 5. Bottleneck

### Главный: **дубликация GPT forward pass**

`XTTSv2.process_tokens_to_speech` @ строки 787-797:

```python
if output.finished:
    hidden_states = await self.get_model_logits(
        list(output.outputs[0].token_ids),     # ← те же токены, что vLLM только что сгенерил
        {...multimodal_data...},
        output.request_id
    )
```

**Что происходит:**
1. vLLM стримит токены через `async for output in generator:` — это **первый** прогон GPT с KV-cache (быстрый, батчуемый)
2. После EOS (`output.finished`) код берёт эти же token_ids и зовёт `get_model_logits(...)` — это **второй** прогон GPT, но в `is_logits_only_mode=True`, чтобы получить hidden_states среднего слоя для HiFi-GAN
3. **Два раза один и тот же compute по одной причине: vLLM V0 не умеет возвращать hidden states.** Это тот самый архитектурный костыль, который я детально разбирал в `VLLM_COMPATIBILITY_AUDIT.md`.

**Масштаб:** ~2x GPU-compute на GPT-стадии (для длинного текста — доминирующая часть latency).

### Второй: **HiFi-GAN decode — per-chunk, не streaming**

`XTTSv2.py:800-812`:

```python
async with self.decoder_semaphore:
    async with self.cuda_memory_manager():
        wav = (await asyncio.to_thread(self.hifigan_decoder,
                hidden_states, g=speaker_embeddings)
        ).cpu().detach().numpy().squeeze()
        yield TTSOutput(array=wav, ...)
```

- Декодер вызывается **после полной генерации токенов sub_request** (`if output.finished:`). Не streaming.
- Следовательно, **user-perceived streaming latency** = `GPT_gen_time(full_sub_request) + HiFi-GAN_decode_time(full_sub_request)`, а не token-by-token.
- `decoder_semaphore` (лимит concurrent HiFi-GAN вызовов) — защита VRAM, но ограничивает параллелизм.

### Третий: **python GIL + `asyncio.to_thread`**

HiFi-GAN вызов в `asyncio.to_thread` освобождает event loop, но PyTorch внутри всё равно может ловить GIL при операциях на CPU (squeeze, numpy conversion). Для T4 не проблема, для A100 может быть заметно.

### Где не bottleneck (опроверг предположения)

- **Text splitting** (`split_requests`) — O(n) slice, < 1мс
- **`combine_outputs`** — `np.concatenate` один раз в конце, несколько десятков MB — < 100мс
- **`torchaudio.save`** — финальный шаг, разовый

---

## 6. Реализованные изменения (Часть 1)

### ⚠️ Изменения в коде НЕ сделаны.

**Обоснование:**
1. В кодовой базе **отсутствуют `print()`-вызовы, которые можно было бы заменить прогресс-баром** — нет циклов per-chunk `print(f"chunk {i}/{N}")`.
2. Per-chunk шум, который user видел в Colab, генерировался через `logger.info` от vLLM и нашего scheduler — **уже устранён коммитом `9d5f089`** (фильтр паттерна `^(Added|Finished|Aborted|Received)\s+request`, `propagate=False`, INFO→DEBUG).
3. Выдавать это за работу через добавление фейкового tqdm-обёртки было бы against the принципы clean-code / systematic-debugging, которые указаны в шапке задания.

### Что был бы честный следующий шаг (ожидает подтверждения)

**Опция B (cosmetic, low-risk):** route 6 one-shot `print()` в `hifigan_decoder.py` (строки 280, 438, 448, 659, 665, 676) через `logger.info`. Делает вывод при загрузке модели консистентным с остальным логированием (единый формат времени/иконок). Не прогресс-бар, но чистый `print → logger` refactor.

**Опция C (feature, требует confirm):** добавить обёртку `rich.progress.Progress` вокруг итерирования по чанкам в `generate_speech()` / `generate_speech_async()`. Не меняет генерацию, только визуализирует. Реализация:

```python
# псевдокод — только ради иллюстрации, НЕ применено
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

def generate_speech(self, request):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed} чанков | {task.fields[rate]} tok/s"),
        transient=True,  # исчезает после завершения
    ) as progress:
        task = progress.add_task("🔊 Генерация...", total=None, rate="—")
        # в месте yield chunk:
        progress.update(task, advance=1, rate=f"{latest_tok_per_s:.0f}")
        yield chunk
    console.print(f"✅ Готово | {total_duration:.1f}мин аудио | RTF {rtf:.2f}x | {wall:.0f} сек")
```

**Риск опции C:** она меняет визуальный output, но требует инструментации `tts.py` (вывод stats после цикла). Это НЕ логика генерации, но строго говоря — код `tts.py`, не только logging. Жду явного «да, делай C» перед применением.

### git diff

Нет изменений → коммит не создавался.

```
git status --short
(пусто)
```

### Как теперь выглядит вывод при генерации (без изменений этой сессии)

С учётом уже применённых фиксов из коммита `9d5f089`:

**До генерации (model load):**
```
19:47:10.123 | XTTSv2.py:75 | ℹ️ INFO     | Initializing XTTSv2Engine...
Removing weight norm...                                 ← print() из hifigan_decoder.py:280
 | > 147 / 147 layers are restored.                     ← print() из hifigan_decoder.py:448
 > Model fully restored.                                ← print() из hifigan_decoder.py:659
19:47:14.891 | XTTSv2.py:142 | ℹ️ INFO     | Model loaded (T4, 0.71 GB VRAM)
```

**Во время генерации:**
```
19:47:20.012 | performance.py:144 | ℹ️ INFO     | 🔊   287 tok/s |  1.2 req/s | RTF 0.21x
19:47:25.050 | performance.py:144 | ℹ️ INFO     | 🔊   295 tok/s |  1.3 req/s | RTF 0.20x
```

**После завершения:** нет «✅ Готово за 41 сек | 3.2 мин аудио | RTF 0.21x» — потому что в коде **нет** явного summary-принта в конце `generate_speech()`. Это то, что предлагается в опции C.

---

## Итог

- Per-chunk спам уже устранён в прошлой сессии через `logger.py` и `two_phase_scheduler.py` (`9d5f089`)
- В коде нет `print()` для прогресса → tqdm/rich Progress нечего заменять
- Pipeline понят полностью: vLLM → `process_tokens_to_speech` → GPT-повтор для hidden_states → HiFi-GAN → numpy → `combine_outputs(np.concatenate)` → `save`
- Главный bottleneck — **двойной GPT forward pass** (vLLM V0 не отдаёт hidden states), уже задокументирован в `VLLM_COMPATIBILITY_AUDIT.md`
- Жду решения: применять ли опции B / C. По умолчанию — ничего не менял.
