# CLAUDE.md — project brief for AI coding assistants

Auralis is an async TTS engine wrapping XTTSv2 on top of vLLM, optimised for high-concurrency streaming workloads. This file gives any AI pair-programmer the project-specific context needed to make non-trivial changes without re-deriving it from scratch.

---

## Quick facts

| | |
|---|---|
| **Upstream** | `astramind-ai/auralis` |
| **This fork** | `JasperSeling/Auralis` (default branch `main`) |
| **Runtime** | Linux only (`setup.py:check_platform()` rejects Windows/macOS). Python >=3.10. CUDA GPU required. |
| **Version pinned** | `vllm==0.6.4.post1` — V0 engine, **not V1**. |
| **Layout** | `src/auralis/` (package), `tests/unit/`, `notebooks/colab_quickstart.ipynb`, `docs/` (mkdocs). |
| **Install** | `pip install -e .` on Linux. On Windows you can edit and run the hermetic unit tests only. |
| **Test command** | `python -m pytest tests/unit -v` |
| **Entry point** | `auralis.openai` — OpenAI-compatible REST server (`src/auralis/entrypoints/oai_server.py`). |

---

## Minimal usage

```python
from auralis import TTS, TTSRequest

tts = TTS().from_pretrained(
    "AstraMindAI/xttsv2",
    gpt_model="AstraMindAI/xtts2-gpt",
    gpu_memory_utilization=0.85,   # see "Known bugs" #1
)
request = TTSRequest(text="...", speaker_files=["speaker.wav"], language="en")
stats = tts.save_stream(request, "output.flac")
```

---

## Architecture

Two-phase scheduler in `src/auralis/common/scheduling/two_phase_scheduler.py`:

1. **`TTS._prepare_generation_context`** (`src/auralis/core/tts.py`) — invokes `XTTSv2.get_generation_context`, which tokenises the text, builds conditioning, and **eagerly calls `llm_engine.generate()` for every sentence**, collecting async generators into a list.
2. **`TTS._second_phase_fn` → `XTTSv2.process_tokens_to_speech`** — iterates each generator; for every finished sentence it runs a **second** `llm_engine.generate(max_tokens=1)` via `get_model_logits` solely to collect hidden states (legacy V0 hack, see Known bugs #6), then decodes on a thread via `asyncio.to_thread(self.hifigan_decoder, ...)`.

Streaming writers `save_stream` / `save_stream_async` in `src/auralis/core/tts.py` flush audio chunks straight to disk — RAM footprint is O(1) regardless of duration.

Sub-request splitting: `TTS.split_requests(request, max_length=100_000)` splits input text by **character count**, not by sentence. Sentence-level splitting then happens inside `tokenizer.batch_encode_with_split` per sub-request.

---

## Known bugs and gotchas

Each item below corresponds to a real leak or correctness issue already paid for in this repo's history. Do not undo these workarounds without replacing them with something better.

### 1. `XTTSv2.get_memory_usage_curve` is numerically broken

The empirical polynomial (`src/auralis/models/xttsv2/XTTSv2.py`) returns GB of model weights, but `BaseAsyncTTSEngine.get_memory_percentage` then passes that to vLLM as a *fraction of total VRAM*. Result: `gpu_memory_utilization ≈ 0.09` on T4 16GB, `≈ 0.02` on A100 80GB — KV-cache is starved and long generations trigger `Sequence group preempted by RECOMPUTE mode`.

**Workaround**: callers pass `gpu_memory_utilization=0.85` to `TTS().from_pretrained(...)`. The override path in `XTTSv2.__init__` skips the polynomial entirely.

A proper fix would require recalibrating the polynomial (or deleting it and using a sensible default like `0.85`). That is not done here.

### 2. `llm_engine.generate()` is eager

It submits the request to vLLM's scheduler at call time; it is not a lazy iterator. Building `generators = [llm_engine.generate(...) for ... in tokens_list]` puts **every** sentence into vLLM's waiting queue simultaneously, each pinning its `SequenceGroup`, multi-modal audio embeddings, and our `ExtendedSamplingParams`.

**Rule**: `process_tokens_to_speech` calls `await self.llm_engine.abort(output.request_id)` immediately after yielding a finished sentence's audio. Any new code path that consumes vLLM generators must do the same or pay the RAM cost (~1 MB per sentence on 900-sentence jobs).

### 3. `HiddenStatesCollector` is shared per engine

Legacy V0 hidden-states collection uses `ExtendedSamplingParams.hidden_state_collector` (a `SyncCollectorWrapper` pointing at an instance of `HiddenStatesCollector`). vLLM retains `sampling_params` for the lifetime of finished `RequestOutput`s, which pins the collector.

**Rule**: there is exactly **one** `HiddenStatesCollector` per `XTTSv2` instance (constructed in `__init__`). `bind_to_request` installs per-request state within the shared collector, keyed by `request_id`. Do not reintroduce per-call instantiation — it caused ~1 GB RSS growth on 900-sentence jobs before this was fixed.

### 4. Collector cleanup on every exit path

`HiddenStatesCollector.get_hidden_states` must call `_cleanup_request` on:

- the happy path (returning concatenated hidden states)
- timeout (`collection_complete.wait(timeout)` returned False)
- no-outputs `ValueError`
- the outer `except` handler

`tests/unit/test_hidden_states_collector_leak.py` guards all four paths plus a shared-instance reuse test. Keep it passing.

### 5. `asyncio.to_thread` grows a default executor

HiFi-GAN decode (per finished sentence) and GPT conditioning (per sub-request) run via `asyncio.to_thread`, which lazily populates the event loop's default `ThreadPoolExecutor` up to `min(32, os.cpu_count()+4)` workers. Those threads persist until `loop.shutdown_default_executor()` is called.

**Rule**: `TTS.shutdown` awaits `self.loop.shutdown_default_executor()`. **Never** also call `self.loop.close()` there — in Colab/Jupyter `self.loop` is the host notebook loop captured via `asyncio.get_running_loop()` in `_ensure_event_loop`; closing it kills the notebook.

### 6. Legacy V0 hidden-states path is load-bearing

`XTTSv2.get_model_logits` runs a second vLLM `generate(max_tokens=1)` per sentence solely to collect hidden states via `ExtendedSamplingParams.hidden_state_collector`. A migration to a V1-style safetensors-on-disk handoff was drafted in `docs/VLLM_COMPATIBILITY_AUDIT.md` but never completed. Treat the double-pass pattern as load-bearing until explicitly refactored.

### 7. Progress UX contract

The rich progress setup in `tts.py` uses `MofNCompleteColumn` with **`total=None`** (indeterminate pulse bar). Passing `len(self.split_requests(request))` as total produced misleading `604/2` displays because `split_requests` counts 100k-char input slices, not audio chunks (XTTSv2 yields one audio chunk per sentence, hundreds per sub-request).

**Rule**: keep `total=None`; use `TimeElapsedColumn` (not `TimeRemainingColumn`, which requires a known total).

### 8. `TTS.shutdown` is async

Do not introduce `self.loop.run_until_complete(...)` inside it — the coroutine is already scheduled on `self.loop`.

---

## Editing playbook

- **Before changing `XTTSv2.process_tokens_to_speech`**: re-read §2 and §3. Any new yield path must abort the finished request.
- **Before changing `hidden_state_collector.py`**: run `python -m pytest tests/unit/test_hidden_states_collector_leak.py -v`. Every early-return path in `get_hidden_states` must call `_cleanup_request`.
- **Before adding new `asyncio.to_thread` sites**: confirm they run on `self.loop` so `TTS.shutdown` cleans them up.
- **Before touching the progress bar**: do not put a numeric denominator on it unless you also count audio chunks, not input slices.
- **Tests that must stay hermetic** (runnable without CUDA/vLLM/torio): load the module under test via `importlib.util.spec_from_file_location` rather than importing through `auralis.*`, following the pattern in `tests/unit/test_hidden_states_collector_leak.py`.

---

## Outstanding work (low-confidence backlog)

- **Lazy generator pipeline.** The eager-generate + per-sentence-abort pattern is a workaround. A real fix would iterate sentences one at a time (or in bounded batches of `max_concurrency`) rather than queueing hundreds of requests in vLLM up front. Requires changing the `_prepare_generation_context` → `TwoPhaseScheduler` → `_second_phase_fn` contract.
- **Polynomial replacement.** Either delete `get_memory_usage_curve` and default `gpu_memory_utilization` to 0.85, or recalibrate with real measurements across GPUs. The current polynomial is actively misleading.
- **V1 engine migration.** `docs/VLLM_COMPATIBILITY_AUDIT.md` enumerates the blocking issues. Until that lands, we are stuck on `vllm==0.6.4.post1`.

---

## Commit conventions

- Subject: `fix: <what>` / `feat: <what>` / `chore: <what>` (lowercase, present tense).
- Body: root cause → chosen fix → test/verification. Name the specific symptom you observed (e.g. `+1127 MiB RSS growth on 154-min jobs`) so the commit is self-explanatory a year later.
- Regression tests live under `tests/unit/`. Prefer them over integration tests for leak fixes — unit tests can stay platform-agnostic and CI-friendly.
