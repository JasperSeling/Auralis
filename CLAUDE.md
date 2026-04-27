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

## Audit reports (read before non-trivial changes)

- `docs/VLLM_COMPATIBILITY_AUDIT.md` — V0→V1 migration blockers, why we are pinned to `vllm==0.6.4.post1`.
- `docs/vllm_engine_lifecycle_audit.md` — where the engine is created, why GPU memory grows between `save_stream()` calls, and the `from_pretrained()` double-engine risk (§9 below).
- `docs/generation_parameters_audit.md` — sampling-params, multimodal-data and KV-cache retention paths.
- `hidden_states_leak_audit.md` (repo root, not `docs/`) — the long-form companion to §3–§4 below; deeper traces, raw greps, and the reasoning behind the structured cleanup pattern in §2.
- `docs/auralis_report.md` — print/logger inventory and pipeline trace.
- `docs/streaming_audio_analysis.md` — yield levels (engine / scheduler / API) and sample-rate handling.

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

1. **`TTS._prepare_generation_context`** (`src/auralis/core/tts.py`) — invokes `XTTSv2.get_generation_context`, which tokenises the text, builds conditioning, and **eagerly calls `llm_engine.generate()` for every sentence**, collecting async generators into a list. (A lazy variant was attempted in `bfdda62` and reverted in `9e924d5`; eager submission paired with the structured cleanup pattern in §2 is the production code path.)
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

### 2. `llm_engine.generate()` is eager — cleanup must be structured

`AsyncLLMEngine.generate()` registers the request with vLLM's scheduler at call time; it is not a lazy iterator. The current loop in `XTTSv2.get_generation_context` submits every sentence in a sub-request up front, each pinning its `SequenceGroup`, multi-modal audio embeddings, and our `ExtendedSamplingParams`. A lazy `_make_sentence_generator` variant was tried in `bfdda62` and reverted in `9e924d5`; the production answer is **eager submission + disciplined per-request cleanup**.

**Rule** (current shape of `process_tokens_to_speech` and `get_model_logits` after `9e924d5`):

- Wrap the body that owns vLLM-side state in `try` / `finally`.
- In the `finally`: `await self.llm_engine.abort(request_id)`, set `sampling_params.hidden_state_collector = None` (for the logits-only path), `del` every local that holds GPU tensors / generators / outputs, and call `torch.cuda.empty_cache()`.
- **Yield the `TTSOutput` *outside* `decoder_semaphore` and `cuda_memory_manager`** — build it inside the CUDA context, but `yield tts_output` only after the contexts exit and the `finally` block has aborted and freed locals. Yielding while still inside `cuda_memory_manager` lets the caller (and downstream `parallel_inputs` references) keep the per-sentence GPU state alive across `await` points.

Any new code path that calls `llm_engine.generate()` must follow the same shape, or it will leak ~1 MB of vLLM metadata per sentence (~900 MB on 154-minute jobs).

### 3. `HiddenStatesCollector` is shared per engine

Legacy V0 hidden-states collection uses `ExtendedSamplingParams.hidden_state_collector` (a `SyncCollectorWrapper` pointing at an instance of `HiddenStatesCollector`). vLLM retains `sampling_params` for the lifetime of finished `RequestOutput`s, which pins the collector.

**Rule**: there is exactly **one** `HiddenStatesCollector` per `XTTSv2` instance (constructed in `__init__`). `bind_to_request` installs per-request state within the shared collector, keyed by `request_id`. Do not reintroduce per-call instantiation — it caused ~1 GB RSS growth on 900-sentence jobs before this was fixed.

### 4. Collector cleanup on every exit path

`HiddenStatesCollector.get_hidden_states` must call `_cleanup_request` on:

- the happy path (returning concatenated hidden states)
- timeout (`collection_complete.wait(timeout)` returned False)
- no-outputs `ValueError`
- the outer `except` handler

In the collector itself, hidden states are stored as `hidden_states.detach().clone()` (not just `.clone()`). The `.detach()` is load-bearing: vLLM hands us tensors that are still attached to autograd graph nodes, and a bare `.clone()` keeps the entire computation graph alive across the request lifetime. See the diff in `9e924d5`.

`tests/unit/test_hidden_states_collector_leak.py` guards all four exit paths plus a shared-instance reuse test. Keep it passing.

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

### 9. `TTS.from_pretrained()` can leak the previous vLLM engine

Documented in `docs/vllm_engine_lifecycle_audit.md`. Calling `TTS.from_pretrained()` twice on the same `TTS` instance overwrites `self.tts_engine` without shutting down the previous engine — the old `AsyncLLMEngine` (and its KV-cache reservation) stays alive on the GPU. Only matters if user code reloads models on the same instance, but it is a real footgun and the audit recommends a `if self.tts_engine is not None: self.loop.run_until_complete(self.tts_engine.shutdown())` guard. Patch is suggested in the audit but **not yet applied** — see Outstanding work.

---

## Editing playbook

- **Before changing `XTTSv2.process_tokens_to_speech` or `get_model_logits`**: re-read §2. Cleanup must be in `finally`; `yield` must be outside `decoder_semaphore` / `cuda_memory_manager`; `abort` + `del` + `empty_cache` must run before yielding.
- **Before changing `hidden_state_collector.py`**: run `python -m pytest tests/unit/test_hidden_states_collector_leak.py -v`. Every early-return path in `get_hidden_states` must call `_cleanup_request`. Stored tensors must be `detach().clone()`, not bare `.clone()`.
- **Before adding new `asyncio.to_thread` sites**: confirm they run on `self.loop` so `TTS.shutdown` cleans them up.
- **Before touching the progress bar**: do not put a numeric denominator on it unless you also count audio chunks, not input slices.
- **Tests that must stay hermetic** (runnable without CUDA/vLLM/torio): load the module under test via `importlib.util.spec_from_file_location` rather than importing through `auralis.*`, following the pattern in `tests/unit/test_hidden_states_collector_leak.py`.

---

## Outstanding work (low-confidence backlog)

- **Engine shutdown in `TTS.from_pretrained`.** The patch in `docs/vllm_engine_lifecycle_audit.md` (§6) is small and concrete: shut down `self.tts_engine` if non-`None` before reloading. Not yet applied.
- **Lazy generator pipeline (revisit).** Attempted in `bfdda62`, reverted in `9e924d5` in favour of eager submission + structured cleanup. If `gpt_embed_inputs` floor (~135 MiB on 900-sentence jobs) becomes the dominant remaining growth, revisiting laziness — or better, streaming `gpt_embed_inputs` itself — is the next lever.
- **Polynomial replacement.** Either delete `get_memory_usage_curve` and default `gpu_memory_utilization` to 0.85, or recalibrate with real measurements across GPUs. The current polynomial is actively misleading.
- **V1 engine migration.** `docs/VLLM_COMPATIBILITY_AUDIT.md` enumerates the blocking issues. Until that lands, we are stuck on `vllm==0.6.4.post1`.

---

## Commit conventions

- Subject: `fix: <what>` / `feat: <what>` / `chore: <what>` (lowercase, present tense).
- Body: root cause → chosen fix → test/verification. Name the specific symptom you observed (e.g. `+1127 MiB RSS growth on 154-min jobs`) so the commit is self-explanatory a year later.
- Regression tests live under `tests/unit/`. Prefer them over integration tests for leak fixes — unit tests can stay platform-agnostic and CI-friendly.
