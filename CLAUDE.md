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
- `kv_cache_leak_audit.md` (repo root) — the static analysis behind the `9b48910` fix: 5 leak points (audio-token + logits-only `abort` missing, `yield` inside `cuda_memory_manager`, `PositionalEmbeddingsCorrecter.clear_request` only firing for logits-only, missing `del` of locals); each with file:line and the symptom math (1.87 GiB / ~51 tensors per `save_stream`).
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

1. **`TTS._prepare_generation_context`** (`src/auralis/core/tts.py`) — invokes `XTTSv2.get_generation_context`, which tokenises the text, builds conditioning, and appends one **unstarted** `XTTSv2._make_sentence_generator(...)` per sentence into the `generators` list. The body of each lazy generator (`ExtendedSamplingParams` construction, `TokensPrompt`, the actual `llm_engine.generate()` call) executes only on the first `__anext__` — i.e. after the `TwoPhaseScheduler` acquires `second_phase_sem`. So at any moment only ~`second_phase_concurrency` sentences are registered with vLLM, not all `len(tokens_list)`.
2. **`TTS._second_phase_fn` → `XTTSv2.process_tokens_to_speech`** — iterates each generator; for every finished sentence it runs a **second** `llm_engine.generate(max_tokens=1)` via `get_model_logits` solely to collect hidden states (legacy V0 hack, see Known bugs #6), then decodes on a thread via `asyncio.to_thread(self.hifigan_decoder, ...)`. Both `get_model_logits` and `process_tokens_to_speech` wrap the vLLM-owning body in `try`/`finally` and explicitly `abort` + `del` + `empty_cache` on exit — see §2.

*History note*: the lazy generator landed in `bfdda62`, was reverted by `9e924d5` (which also added structured cleanup tied to eager submission), then `9e924d5` itself was reverted in `39770e3` so the lazy pipeline came back. `9b48910` re-applied the cleanup pattern on top of the lazy pipeline — they compose. Read both commits if you need the why.

Streaming writers `save_stream` / `save_stream_async` in `src/auralis/core/tts.py` flush audio chunks straight to disk — RAM footprint is O(1) regardless of duration.

Sub-request splitting: `TTS.split_requests(request, max_length=100_000)` splits input text by **character count**, not by sentence. Sentence-level splitting then happens inside `tokenizer.batch_encode_with_split` per sub-request.

---

## Known bugs and gotchas

Each item below corresponds to a real leak or correctness issue already paid for in this repo's history. Do not undo these workarounds without replacing them with something better.

### 1. `XTTSv2.get_memory_usage_curve` is numerically broken

The empirical polynomial (`src/auralis/models/xttsv2/XTTSv2.py`) returns GB of model weights, but `BaseAsyncTTSEngine.get_memory_percentage` then passes that to vLLM as a *fraction of total VRAM*. Result: `gpu_memory_utilization ≈ 0.09` on T4 16GB, `≈ 0.02` on A100 80GB — KV-cache is starved and long generations trigger `Sequence group preempted by RECOMPUTE mode`.

**Workaround**: callers pass `gpu_memory_utilization=0.85` to `TTS().from_pretrained(...)`. The override path in `XTTSv2.__init__` skips the polynomial entirely.

A proper fix would require recalibrating the polynomial (or deleting it and using a sensible default like `0.85`). That is not done here.

### 2. `llm_engine.generate()` registers eagerly — cleanup must be structured

Even though `XTTSv2._make_sentence_generator` defers the actual call until the first `__anext__`, once that body runs `AsyncLLMEngine.generate()` registers the request with vLLM's scheduler immediately and pins a `SequenceGroup`, multi-modal audio embeddings, and our `ExtendedSamplingParams` until the request is explicitly aborted or evicted. A non-aborted finished request keeps its KV-cache slabs (~50 tensors / ~1.87 GiB per request on the 154-min book benchmark) alive until the next `add_request` evicts it — which on the *last* sub-request of a `save_stream()` never happens. See `kv_cache_leak_audit.md`.

**Rule** (current shape of `process_tokens_to_speech` and `get_model_logits` after `9b48910`):

- Pre-declare every local the `finally` will touch (e.g. `generator = output = sampling_params = bound_collector = hidden_states = None`) before `try` so the `finally` is always safe to run.
- Wrap the body that owns vLLM-side state in `try` / `finally`.
- In the `finally`: `await self.llm_engine.abort(request_id)` (best-effort, in `try`/`except` → `logger.debug`); set `sampling_params.hidden_state_collector = None` to break the `SyncCollectorWrapper` cycle vLLM keeps via finished `RequestOutput` history; reach into `self.llm_engine.engine.model_executor.driver_worker.model_runner.model.positional_embeddings_correcter` and call `clear_request(rid)` + `clear_request(f"{rid}_logits")` (best-effort, vLLM private path); `del` every local that pins GPU tensors / generators / outputs / dicts; call `torch.cuda.empty_cache()` last.
- **Yield the `TTSOutput` *outside* `decoder_semaphore` and `cuda_memory_manager`** — build it inside the CUDA context, but `yield tts_output` only after both contexts exit and the `finally` block has aborted, cleared and freed locals. Yielding while still inside `cuda_memory_manager` defers `empty_cache` for the entire time the consumer holds the generator, defeating the manager.

Any new code path that calls `llm_engine.generate()` must follow this shape.

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

### 9. `TTS.from_pretrained()` shuts down the previous vLLM engine

Applied. `TTS.from_pretrained` now checks `self.tts_engine is not None and hasattr(...,'shutdown')` and runs the previous engine's async `shutdown()` on `self.loop` before overwriting `self.tts_engine`. The check sits *after* the new config has been parsed successfully, so a bad `model_name_or_path` does not destroy the currently working engine. Failures from the prior shutdown are logged at WARNING and do not abort the reload. See `docs/vllm_engine_lifecycle_audit.md §6` for the original audit.

### 10. `cuda_memory_manager` adds 100 ms per chunk

`XTTSv2.cuda_memory_manager` (`src/auralis/models/xttsv2/XTTSv2.py:498-509`) does `torch.cuda.synchronize()` → `await asyncio.sleep(0.1)` → `torch.cuda.empty_cache()` in its `finally`. Origin of the 100 ms is undocumented (likely a workaround for a race in PyTorch's CUDA caching allocator). On a 900-sentence book that is ~90 s of wall-clock spent sleeping. A future commit should remove the `sleep` under measurement and confirm RTF improves without re-introducing the original race.

### 11. Conditioning encoder must not build autograd graphs

The conditioning encoder path (`get_style_emb`, `get_gpt_cond_latents`, `_get_speaker_embedding`) must run inside `torch.no_grad()`; outputs must be `.detach()`-ed before passing to vLLM. Resolved in commit `7d6299490b53aa30ff18c37a68cabe5c9ff8e2f3`.

---

## Editing playbook

- **Before changing `XTTSv2.process_tokens_to_speech` or `get_model_logits`**: re-read §2. Cleanup must be in `finally`; locals must be pre-declared; `yield` must be outside `decoder_semaphore` / `cuda_memory_manager`; `abort` + `clear_request` + `del` + `empty_cache` must run before yielding.
- **Before changing `XTTSv2._make_sentence_generator`**: it is intentionally laziness-only and contains no cleanup — cleanup happens in `process_tokens_to_speech.finally`. If you add work here, do not introduce per-sentence eager allocations.
- **Before changing `vllm_mm_gpt.py:678` (compute_logits hijack guard)**: the guard `if hidden_state_collector is not None` is load-bearing. `compute_logits` runs on every decode step; unconditionally calling `clear_request` here will empty `request_tracker_dict` mid-generation and break `get_by_next_token` on token #2. The audio-token path's `clear_request` is done from `process_tokens_to_speech.finally` via the private vLLM path (§2), not from inside the hijack.
- **Before changing `hidden_state_collector.py`**: run `python -m pytest tests/unit/test_hidden_states_collector_leak.py -v`. Every early-return path in `get_hidden_states` must call `_cleanup_request`. Stored tensors must be `detach().clone()`, not bare `.clone()`.
- **Before adding new `asyncio.to_thread` sites**: confirm they run on `self.loop` so `TTS.shutdown` cleans them up.
- **Before touching the progress bar**: do not put a numeric denominator on it unless you also count audio chunks, not input slices.
- **Tests that must stay hermetic** (runnable without CUDA/vLLM/torio): load the module under test via `importlib.util.spec_from_file_location` rather than importing through `auralis.*`, following the pattern in `tests/unit/test_hidden_states_collector_leak.py`.

---

## Outstanding work (low-confidence backlog)

- **Streaming `gpt_embed_inputs`.** With §2 cleanup applied (`9b48910`), KV-cache and per-request metadata are no longer the dominant growth. The remaining floor is the per-sentence `gpt_embed_inputs` list built eagerly in `XTTSv2._merge_conditioning` (~135 MiB on 900-sentence jobs). Streaming this through `prepare_inputs_async` rather than materialising it once per sub-request is the next lever, but it is invasive (changes the `_make_sentence_generator` signature).
- **Drop `cuda_memory_manager`'s `asyncio.sleep(0.1)`.** See §10. ~90 s wall-clock saved per 900-sentence job if the sleep is not protecting a real race; verify under measurement before removing.
- **Polynomial replacement.** Either delete `get_memory_usage_curve` and default `gpu_memory_utilization` to 0.85, or recalibrate with real measurements across GPUs. The current polynomial is actively misleading.
- **V1 engine migration.** `docs/VLLM_COMPATIBILITY_AUDIT.md` enumerates the blocking issues. Until that lands, we are stuck on `vllm==0.6.4.post1`.

---

## Commit conventions

- Subject: `fix: <what>` / `feat: <what>` / `chore: <what>` (lowercase, present tense).
- Body: root cause → chosen fix → test/verification. Name the specific symptom you observed (e.g. `+1127 MiB RSS growth on 154-min jobs`) so the commit is self-explanatory a year later.
- Regression tests live under `tests/unit/`. Prefer them over integration tests for leak fixes — unit tests can stay platform-agnostic and CI-friendly.
