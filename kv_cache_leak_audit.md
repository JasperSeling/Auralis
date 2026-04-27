# KV Cache Leak Audit — XTTSv2.py

**Date:** 2026-04-27
**Codebase state:** `main` after `git revert 9e924d5` → working commit `39770e3`. This audit therefore reflects the *post-revert* code: lazy `_make_sentence_generator` is in place again, the structured `try/finally` cleanup that `9e924d5` added is **gone**, and `process_tokens_to_speech` only contains a comment where `abort()` used to be.
**Method:** static read of `src/auralis/models/xttsv2/XTTSv2.py` (898 lines) + `src/auralis/models/xttsv2/components/vllm_mm_gpt.py` (851 lines). No generation was run.

---

## Проблема

- **Симптом (наблюдаемый в Colab после save_stream):** +1.87 GiB GPU `allocated`, 51 живых тензоров после возврата управления.
- **Гипотеза:** `~51` тензоров ≈ один полный набор KV-cache slabs (XTTS GPT-2 имеет ~12 слоёв × ~4 тензора на слой = ~48 на запрос). То есть утекает **KV-кэш одного-двух vLLM-запросов** (audio-token request и/или logits-only request) на каждый sub-request, потому что vLLM не получает `abort()` и держит `SequenceGroup` живым до момента следующего add_request, который вытеснит старый.
- **Цена:** сама ошибка делится на 4–5 независимых leak points (см. таблицу ниже). Все они уже были раньше частично закрыты коммитами `eba22ae`, `f697e8f`, `9e924d5`; revert `9e924d5` снова открыл #1 и #2.

---

## 1. `get_model_logits` — `XTTSv2.py:645–716`

### Точный код

```python
async def get_model_logits(
        self,
        token_ids: List[int],
        conditioning: MultiModalDataDict,
        request_id: str,
) -> torch.Tensor:
    ...
    request_id = f"{request_id}_logits"          # 669

    token_ids = ([self.mel_bos_token_id] + list(token_ids) + [self.mel_eos_token_id] * 4)  # 673

    engine_inputs = TokensPrompt(prompt_token_ids=token_ids)                     # 676
    conditioning['audio']['sequence_length'] = len(token_ids)                    # 677
    engine_inputs["multi_modal_data"] = conditioning                             # 679

    bound_collector = self.hidden_states_collector.bind_to_request(request_id)   # 684

    sampling_params = ExtendedSamplingParams(                                    # 687
        detokenize=False,
        request_id=request_id,
        max_tokens=1,
        hidden_state_collector=bound_collector,
        output_kind=RequestOutputKind.FINAL_ONLY
    )

    generator = self.llm_engine.generate(                                        # 696
        prompt=engine_inputs,
        sampling_params=sampling_params,
        request_id=request_id
    )

    async for output in generator:                                               # 702
        if output.finished:
            pass

    hidden_states = await self.hidden_states_collector.get_hidden_states(request_id)  # 707

    if hidden_states is None:
        raise RuntimeError(...)

    start_of_audio_hs = conditioning["audio"]["embeds"].shape[0]                 # 714
    return self.final_norm(hidden_states[start_of_audio_hs:-5, ...]
                          .unsqueeze(0).to(self.device).to(self.dtype))          # 716
```

### Анализ

| Вопрос | Ответ |
|---|---|
| Как формируется `request_id` для logits-запроса | `f"{request_id}_logits"` (`:669`) — суффикс отделяет от audio-token запроса. |
| Где создаётся `ExtendedSamplingParams` | `:687–693`. `hidden_state_collector=bound_collector` подаётся как ссылка; vLLM удерживает `sampling_params` всё время жизни `RequestOutput`. |
| Где `llm_engine.generate()` | `:696–700`. **Eager** — регистрирует request в vLLM scheduler в момент вызова. |
| Вызывается ли `llm_engine.abort()` | **НЕТ.** Ни в happy path, ни в except, ни в finally. После `return` (`:716`) vLLM-side `SequenceGroup` для `request_id_logits` остаётся в `engine.scheduler` до тех пор, пока его не вытолкнет следующий запрос. |
| Какие GPU-тензоры держат локали после return | `hidden_states` (Tensor на GPU, ~`[T, hidden_dim]`, freed после `final_norm` slice — но slice держит ссылку на исходный tensor через storage). `engine_inputs.multi_modal_data["audio"]["embeds"]` (audio-conditioning Tensor, тот же на который ссылается caller). `bound_collector`, `sampling_params`, `generator`, `output` — Python objects, удерживают ссылки на multimodal data, audio embeds и output token_ids. |
| Есть ли `del` / `empty_cache` | **НЕТ.** Ни одного `del`, ни одного `torch.cuda.empty_cache()`. |

### Лики

- **vLLM request `{request_id}_logits` остаётся зарегистрированным** — KV-cache slabs (~50 тензоров) не возвращаются в pool до next add_request.
- `bound_collector` живёт до выхода из coroutine frame; через него `SyncCollectorWrapper` держит ссылку на `HiddenStatesCollector.outputs[request_id]` (даже если `_cleanup_request` отработал внутри `get_hidden_states`).
- `sampling_params.hidden_state_collector` остаётся не-`None` — vLLM `RequestOutput` history может ссылаться на этот объект.

---

## 2. `process_tokens_to_speech` — `XTTSv2.py:826–887`

### Точный код

```python
@torch.inference_mode()
async def process_tokens_to_speech(
        self,
        generator: AsyncGenerator[RequestOutput, None],
        speaker_embeddings: Optional[torch.Tensor] = None,
        multimodal_data: Optional[torch.Tensor] = None,
        request: TTSRequest = None,
) -> AsyncGenerator[TTSOutput, None]:
    ...
    async for output in generator:                                               # 849

        if output.finished:
            hidden_states = await self.get_model_logits(                         # 853
                list(output.outputs[0].token_ids),
                {
                    "audio": {
                        'embeds': multimodal_data,
                        "is_logits_only_mode": True,
                        "sequence_length": False
                    },
                },
                output.request_id
            )

            async with self.decoder_semaphore:                                   # 866
                async with self.cuda_memory_manager():                           # 867
                    wav = (await asyncio.to_thread(self.hifigan_decoder,
                            hidden_states,
                            g=speaker_embeddings
                        )).cpu().detach().numpy().squeeze()                      # 868–871

                    yield TTSOutput(array= wav,                                  # 875
                                    start_time = request.start_time,
                                    token_length = len(output.outputs[0].token_ids)
                                    )

            # Note: llm_engine.abort(output.request_id) used to live here        # 880
            # ... (only a comment)
```

### Анализ

| Вопрос | Ответ |
|---|---|
| Где вызывается `llm_engine.abort()` | **Нигде.** Только комментарий-памятник на `:880–886`. |
| До или после `yield TTSOutput` | N/A — вызова нет. |
| Что в памяти во время `yield` (`:875`) | `output: RequestOutput` (включая `output.outputs[0].token_ids` — список ~600 int), `hidden_states: torch.Tensor` на GPU (`[1, T, hidden_dim]`, ~MB-scale), `wav: np.ndarray` на CPU (большой — ~MB), `multimodal_data: torch.Tensor` на GPU (audio conditioning embed). Все они удерживаются coroutine frame'ом до возобновления после `yield`. |
| Где находится `yield` относительно ctx-managers | **Внутри** `decoder_semaphore` **и** `cuda_memory_manager()`. То есть `cuda_memory_manager.finally` (с `synchronize` + `empty_cache`) **не выполняется до тех пор, пока consumer не дёрнет следующий `__anext__`**. На long-tail задержку downstream (запись в файл, переключение sub-request) `empty_cache` откладывается. |
| Есть ли `try/finally` | **НЕТ.** Никакого finally нет — ни вокруг `get_model_logits`, ни вокруг HiFi-GAN, ни вокруг yield. |

### Лики

- **vLLM audio-token request `output.request_id` не аббортится** — после `output.finished` его `SequenceGroup` остаётся в vLLM scheduler. Lazy `_make_sentence_generator` body выходит после `async for` — но vLLM в `0.6.4` не гарантирует освобождение KV-cache синхронно при выходе из user-side generator; нужен явный `abort()`.
- **`empty_cache` отложен** на время yield. Пока consumer обрабатывает chunk, ~MB-scale `hidden_states` и audio cond. embed остаются в активной памяти + cache не освобождён.
- **Никакого `del`** для `hidden_states`/`wav`/`output` после yield — последние локали освободятся только когда внешний `async for` сделает следующую итерацию (что повторно перезапишет переменные).

---

## 3. `cuda_memory_manager` — `XTTSv2.py:498–509`

### Точный код

```python
@asynccontextmanager
async def cuda_memory_manager(self):
    """Context manager for CUDA memory management.
    
    Ensures proper allocation and deallocation of CUDA memory during processing.
    """
    try:
        yield
    finally:
        torch.cuda.synchronize()       # 507
        await asyncio.sleep(0.1)       # 508 — 100 ms ожидания (origin: workaround неизвестен)
        torch.cuda.empty_cache()       # 509
```

### Анализ

- **Когда вызывается `empty_cache`:** только в `finally`, т.е. **на выходе** из ctx-manager.
- **Есть ли `yield` внутри блока, использующего этот менеджер:** **ДА.** В `process_tokens_to_speech:867–878` `yield TTSOutput(...)` находится внутри `async with self.cuda_memory_manager():`. Пока consumer не возобновит coroutine, `finally` не отрабатывает.
- **`asyncio.sleep(0.1)`** — добавляет 100 мс на каждый chunk (×900 sub-requests = +90 сек на длинном тексте). Происхождение неизвестно; возможно workaround на race PyTorch CUDA caching allocator.

### Лики

- Нет утечки сама по себе, но **усиливает леки #1, #2**: `empty_cache` откладывается ровно на время yield. Если consumer медленный или сам suspend'ится — кэш растёт.

---

## 4. `PositionalEmbeddingsCorrecter` — `vllm_mm_gpt.py:61–163`

### Точный код (релевантные части)

```python
class PositionalEmbeddingsCorrecter:
    def __init__(self):
        self.request_tracker_dict: Dict[str, TokenPositionAndPrefillTuple] = ...   # 69
        self.token_to_request: Dict[str, str] = {}                                 # 71

    def init_request_id_prefill(self, request_id, prefill_len, nex_token):         # 73
        self.request_tracker_dict[request_id] = TokenPositionAndPrefillTuple(...)
        self.token_to_request[f"{nex_token}_{prefill_len}"] = request_id

    def associate_new_tokens(self, request_id, next_token_id):                     # 136
        pos_id = self._get_pos_id_and_update(request_id)
        self._invalidate_previous_mapping(request_id)
        self.token_to_request[f"{next_token_id}_{pos_id}"] = request_id

    def clear_request(self, request_id: str):                                      # 154
        if request_id in self.request_tracker_dict:
            self._invalidate_previous_mapping(request_id)
            del self.request_tracker_dict[request_id]
```

Единственный callsite `clear_request` — `vllm_mm_gpt.py:680`, в перехваченной `compute_logits`:

```python
sampling_params = seq.sampling_params
if (hasattr(sampling_params, 'hidden_state_collector')
        and sampling_params.hidden_state_collector is not None):              # 678–679
    self.positional_embeddings_correcter.clear_request(sampling_params.request_id)  # 680
    sampling_params.hidden_state_collector(...)                               # 682
```

### Анализ

| Вопрос | Ответ |
|---|---|
| Есть ли метод `clear_request` | **ДА** (`:154–163`). |
| Где он вызывается | Только один callsite: `vllm_mm_gpt.py:680`, **внутри `if hidden_state_collector is not None`** — то есть ТОЛЬКО для logits-only запросов (где `bound_collector` подключён). |
| Какие данные накапливаются | `request_tracker_dict[request_id] -> TokenPositionAndPrefillTuple` (один на request) и `token_to_request[f"{tok}_{pos}"] -> request_id` (одна запись **на каждый сгенерированный аудио-токен**). |

### Лики

- **`clear_request` НЕ вызывается для audio-token запросов.** `_make_sentence_generator` (`XTTSv2.py:719–768`) строит `sampling_params` **без** `hidden_state_collector` — у audio-token путей `hidden_state_collector is None`, поэтому `if` на `:678` пропускает `clear_request`.
- На каждое предложение мы делаем `init_request_id_prefill` + `associate_new_tokens` × ~600 раз внутри vLLM compute_logits hijack, но **ничего не очищаем**. На 900-sentence job: `request_tracker_dict` = 900 записей, `token_to_request` ≈ 540 000 строк-ключей. Это **CPU/RAM leak** (не GPU), но накапливается монотонно между sub-requests внутри одного `save_stream()` и **между разными `save_stream()` вызовами на одном engine**.
- `_invalidate_previous_mapping` (`:112`) делается per-token внутри `associate_new_tokens` — но удаляет только записи **этого** request_id, не предыдущих. Завершённый request полностью никогда не чистится.

---

## Точки утечки — сводка

| # | Место | Файл:строка | Что утекает | Цена за 154-мин job |
|---|---|---|---|---|
| 1 | `get_model_logits` — нет `abort` | `XTTSv2.py:716` | KV-cache `{rid}_logits` request, `SequenceGroup`, `multi_modal_data` ref, `sampling_params.hidden_state_collector` ref | ~50 тензоров KV / последний sub-request |
| 2 | `process_tokens_to_speech` — нет `abort` | `XTTSv2.py:880` (только комментарий) | KV-cache audio-token request | ~50 тензоров KV / последний sub-request |
| 3 | `yield` внутри `cuda_memory_manager` | `XTTSv2.py:867–878` | `empty_cache` отложен; `hidden_states` + `multimodal_data` + `wav` живут в frame во время yield | усиливает #1, #2 |
| 4 | `PositionalEmbeddingsCorrecter` — `clear_request` не вызывается для audio-token | `vllm_mm_gpt.py:678` (guard на `hidden_state_collector is not None`) | `request_tracker_dict` и `token_to_request` (CPU dict, не GPU) | ~540 000 stale string keys на книгу, монотонно копится между save_stream |
| 5 | `get_model_logits` — нет `del` локалей перед return | `XTTSv2.py:716` | `engine_inputs`, `bound_collector`, `sampling_params`, `generator`, `output` живут до GC frame'а | косвенно держит #1 |

**Симптом «+1.87 GiB allocated, 51 тензор»** наиболее точно объясняется суммой #1 + #2: один не-аббортнутый logits-only request + один не-аббортнутый audio-token request от *последнего* sub-request `save_stream()`, чьи KV-cache slabs vLLM не успел вытеснить, потому что новых add_request больше не было.

---

## Предлагаемый фикс (НЕ применять в этой сессии — только описать)

Все пять точек закрываются одной структурной правкой `XTTSv2.py`, которую раньше делал `9e924d5` (и которая была откачена). Минимальный апстрим-набор:

### A. `get_model_logits` — try/finally + abort + del + empty_cache (`:645–716`)

- Добавить локали `generator = output = sampling_params = bound_collector = hidden_states = None` до `try`.
- Обернуть тело (от `bind_to_request` до `return`) в `try` / `finally`.
- В `finally`:
  - `await self.llm_engine.abort(request_id)` (best-effort, в `try/except` → `logger.debug`);
  - `if sampling_params is not None: sampling_params.hidden_state_collector = None` — снимает удержание collector через `RequestOutput.history`;
  - `del bound_collector, sampling_params, generator, output, engine_inputs, conditioning, hidden_states`;
  - `if torch.cuda.is_available(): torch.cuda.empty_cache()`.
- В happy path сначала вычислить `result = self.final_norm(...)` в локаль, потом `return result` после `finally` — чтобы `hidden_states` можно было `del`'нуть до возврата.

Эффект: закрывает #1 и #5.

### B. `process_tokens_to_speech` — abort + yield outside ctx + del (`:826–887`)

- Сначала захватить `output_request_id = output.request_id` и `token_ids = list(output.outputs[0].token_ids)` сразу после `if output.finished`.
- `try` вокруг `get_model_logits` + HiFi-GAN; **построить `tts_output = TTSOutput(...)` внутри `cuda_memory_manager`**, но НЕ yield внутри.
- `finally`:
  - `del hidden_states`;
  - `await self.llm_engine.abort(output_request_id)` (best-effort);
  - `del output, token_ids`;
  - `if torch.cuda.is_available(): torch.cuda.empty_cache()`.
- `yield tts_output` **после** выхода из обоих ctx-managers (`decoder_semaphore`, `cuda_memory_manager`) и после `finally`.

Эффект: закрывает #2 и #3.

### C. `PositionalEmbeddingsCorrecter` — расширить guard (`vllm_mm_gpt.py:678–680`)

- Снять зависимость от `hidden_state_collector is not None`. Звать `clear_request(sampling_params.request_id)` для **любого** sequence group, у которого есть `request_id` и `seq.is_finished()`.
- Альтернативно: вызвать `clear_request` явно из `XTTSv2._make_sentence_generator` после выхода из `async for output in self.llm_engine.generate(...)` (через ссылку `self.llm_engine.engine.model_executor.driver_worker.model_runner.model.positional_embeddings_correcter` — но это глубокая private-кишка vLLM, см. `VLLM_COMPATIBILITY_AUDIT.md`).

Эффект: закрывает #4. Низкий приоритет (CPU dict ≠ GPU память; не объясняет 1.87 GiB).

### D. `cuda_memory_manager` — рассмотреть удаление `asyncio.sleep(0.1)` (`:508`)

- Происхождение не задокументировано. На 900 sub-requests это +90 сек wall-clock впустую.
- Удалять отдельным коммитом после A+B, под измерениями.

### Регрессия

- Существующий `tests/unit/test_hidden_states_collector_leak.py` покрывает только `HiddenStatesCollector` — не ловит #1/#2/#3/#4. Нужен новый hermetic test с моком `llm_engine.abort` и проверкой что:
  1. в `get_model_logits` `abort` вызывается ровно один раз с `f"{rid}_logits"` на любом exit path;
  2. в `process_tokens_to_speech` `abort` вызывается с `output.request_id` до `yield`;
  3. `PositionalEmbeddingsCorrecter.request_tracker_dict` пуст после dispose.

### Что НЕ нужно делать

- НЕ возвращать структурную правку через cherry-pick `9e924d5` — там были смешаны два разных изменения (revert lazy pipeline + добавление cleanup). После revert код вернулся к **lazy** path, который желательно сохранить (eager loop был хуже по RAM на 900 sub-requests). Cleanup-патч надо переделать поверх текущего `_make_sentence_generator`, без отката лени.

---

## Контекст коммитов

```
39770e3 Revert "fix: abort logits-only vLLM requests and move yield outside cuda_memory_manager"
ec796d7 docs(claude,rules): sync with 9e924d5 + audit reports from 8932200       ← устарел после revert
8932200 docs: add memory leak audit reports
9e924d5 fix: abort logits-only vLLM requests and move yield outside cuda_memory_manager   ← откачен
bfdda62 fix: lazy per-sentence vLLM generator; revert abort() band-aid             ← снова в силе
e531a16 docs: add .windsurf/rules/project.md and CLAUDE.md
f697e8f fix: free vLLM request metadata after completion; shutdown default executor
eba22ae fix: HiddenStatesCollector thread leak — reuse single instance per engine
```

После revert текущее состояние — это `bfdda62` + `e531a16` + `8932200` (минус код-измения `9e924d5`). `CLAUDE.md` (от `ec796d7`) описывает несуществующий cleanup pattern и требует обновления, если revert постоянный.
