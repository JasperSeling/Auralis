---
trigger: always_on
---

# Auralis — Cascade working rules

Auralis is a fork of `astramind-ai/auralis`: a vLLM-backed async TTS engine built around XTTSv2. These rules distil the non-obvious constraints and sharp edges discovered while debugging this codebase. **Read before modifying anything in `src/auralis/models/xttsv2/` or `src/auralis/core/tts.py`.**

## Platform

- **Runs only on Linux** — `setup.py` raises on Windows/macOS (`check_platform()`). All generation work must happen on a Linux host (Colab/Docker) with a CUDA GPU. A Windows dev box can edit code and run the unit tests (which load individual modules without touching vLLM/torio), but cannot exercise `TTS.from_pretrained` end-to-end.
- Python `>=3.10`. vLLM is **pinned** to `vllm==0.6.4.post1` (V0 engine — do not assume V1 behaviour from docs).

## Architecture at a glance

- Public API: `from auralis import TTS, TTSRequest` (`@src/auralis/__init__.py`).
- Two-phase pipeline (`@src/auralis/common/scheduling/two_phase_scheduler.py`):
  1. **`_prepare_generation_context`** — tokenises, builds speaker / GPT conditioning, and appends one **unstarted** `XTTSv2._make_sentence_generator(...)` per sentence into the `generators` list. The body of each lazy generator (`ExtendedSamplingParams`, `TokensPrompt`, the actual `llm_engine.generate()` call) only runs on the first `__anext__` — i.e. after `TwoPhaseScheduler` acquires `second_phase_sem`. So at any moment only ~`second_phase_concurrency` sentences are registered with vLLM, not all `len(tokens_list)`.
  2. **`_second_phase_fn` → `process_tokens_to_speech`** — iterates each generator, and for each finished sentence runs a **second** `llm_engine.generate(max_tokens=1)` via `get_model_logits` solely to collect hidden states, then `hifigan_decoder` on a thread via `asyncio.to_thread`. Both `get_model_logits` and `process_tokens_to_speech` wrap the vLLM-owning body in `try`/`finally` and explicitly `abort` + `clear_request` + `del` + `empty_cache` on exit (since `9b48910`).
- `TTS.save_stream` / `save_stream_async` are O(1)-RAM disk writers — audio is not accumulated in Python.
- Sub-request splitter: `TTS.split_requests(request, max_length=100_000)` — **splits by character count, not sentence boundaries**. Sentence-level splitting happens inside `tokenizer.batch_encode_with_split`, per sub-request.

## Known bugs / gotchas — keep these in mind

1. **`XTTSv2.get_memory_usage_curve` is numerically broken.** The empirical polynomial (`@src/auralis/models/xttsv2/XTTSv2.py:167-176`) returns GB of model weights, but `BaseAsyncTTSEngine.get_memory_percentage` then feeds that number to vLLM as a *fraction* of VRAM. Result: `gpu_memory_utilization ≈ 0.09` on T4 16GB, `≈ 0.02` on A100 80GB — KV-cache is starved and long generations trigger `Sequence group preempted by RECOMPUTE mode`. **Workaround**: callers pass `gpu_memory_utilization=0.85` explicitly to `TTS().from_pretrained(...)`; the override path in `XTTSv2.__init__` skips the polynomial.
2. **`llm_engine.generate()` registers eagerly — cleanup must be structured.** Even though `XTTSv2._make_sentence_generator` defers the call until first `__anext__`, once it runs vLLM pins a `SequenceGroup`, audio embeds and `ExtendedSamplingParams` until the request is explicitly aborted or evicted. A non-aborted finished request keeps its KV-cache slabs (~50 tensors / ~1.87 GiB on the 154-min benchmark) alive until the next `add_request` evicts it — which on the *last* sub-request never happens. The production rule (`9b48910`, on top of the lazy pipeline `bfdda62`): pre-declare every local the `finally` touches; wrap the vLLM-owning body in `try`/`finally`; in `finally` call `await self.llm_engine.abort(request_id)` (best-effort), set `sampling_params.hidden_state_collector = None` (logits-only path), reach into `self.llm_engine.engine.model_executor.driver_worker.model_runner.model.positional_embeddings_correcter` and call `clear_request(rid)` + `clear_request(f"{rid}_logits")` (best-effort, vLLM private path), `del` every local that holds GPU tensors / generators / outputs, then `torch.cuda.empty_cache()`; and **`yield` the `TTSOutput` only after exiting `decoder_semaphore` and `cuda_memory_manager` and after the `finally` block has run**. Yielding inside the CUDA context defers `empty_cache` for as long as the consumer holds the generator. See `@kv_cache_leak_audit.md`.
3. **`HiddenStatesCollector` is shared per engine, not per request.** Constructing one per sentence leaks because vLLM retains references to our `SyncCollectorWrapper` through `sampling_params`. `XTTSv2` builds exactly one in `__init__` and reuses it; `bind_to_request` installs isolated per-request state inside the shared instance. Do not reintroduce per-call instantiation.
4. **Per-request cleanup on every exit path.** `HiddenStatesCollector.get_hidden_states` must call `_cleanup_request` on the happy path *and* on timeout, on no-outputs `ValueError`, and in the outer `except`. Stored tensors must be `hidden_states.detach().clone()`, not bare `.clone()` — vLLM hands us tensors still attached to autograd, and a missing `.detach()` keeps the entire computation graph alive across the request lifetime (see `9e924d5`). The regression test in `@tests/unit/test_hidden_states_collector_leak.py` guards the cleanup paths.
5. **`asyncio.to_thread` grows a default executor.** HiFi-GAN decode and GPT conditioning run on the loop's default `ThreadPoolExecutor`. Worker threads persist until `loop.shutdown_default_executor()` — called from `TTS.shutdown`. **Never** call `self.loop.close()` there: in Colab/Jupyter `self.loop` is the host notebook loop captured via `asyncio.get_running_loop()` in `_ensure_event_loop`.
6. **The legacy V0 hidden-states collection path is still live** (`XTTSv2.get_model_logits` + `ExtendedSamplingParams.hidden_state_collector`). A full migration to V1-style safetensors-on-disk was drafted but never completed. Treat this double-pass pattern as load-bearing until explicitly refactored.
7. **`TTS.shutdown` is `async`.** Use `await` inside it; never call `self.loop.run_until_complete` from a coroutine already scheduled on `self.loop`.
8. **`TTS.from_pretrained()` shuts down the previous engine.** Applied in `@src/auralis/core/tts.py`. The check fires only after the new config has been parsed successfully, so a bad `model_name_or_path` does not destroy the currently working engine. Failures from the prior shutdown are logged at WARNING and do not abort the reload. Do not move the guard before the config-parse block.
9. **`vllm_mm_gpt.py:678` `compute_logits` hijack guard is load-bearing.** The guard `if hidden_state_collector is not None` gates `clear_request(...)` so it only fires for the logits-only path (which has `max_tokens=1` and finishes after one `compute_logits`). `compute_logits` runs on every decode step for audio-token requests; unconditionally calling `clear_request` from inside the hijack would empty `request_tracker_dict` mid-generation and break `get_by_next_token` on token #2 (audio-token generation fails with `ValueError: No valid mappings found`). For audio-token requests, `clear_request` is invoked from `process_tokens_to_speech.finally` via the vLLM private path — do **not** move it back into the hijack.

## Progress UX contract

- The rich progress column set uses `MofNCompleteColumn` with `total=None` (indeterminate pulse). **Never** pass `len(self.split_requests(request))` as `total` — that value counts 100k-char input slices, not audio chunks, which produces misleading `604/2` displays.
- `TimeElapsedColumn` is correct here; `TimeRemainingColumn` needs a known total.

## Running

- Install (Linux only): `pip install -e .`
- Unit tests (platform-agnostic): `python -m pytest tests/unit -v`
  - The hidden-states regression test loads the collector module directly via `importlib.util.spec_from_file_location` to avoid pulling in `torio`/vLLM transitive imports — keep that pattern for any pure-state leak tests.
- Entry point for an OpenAI-compatible server: `auralis.openai` (console script in `setup.py`).

## Audit reports (consult before non-trivial work)

- `@docs/VLLM_COMPATIBILITY_AUDIT.md` — V0→V1 migration blockers.
- `@docs/vllm_engine_lifecycle_audit.md` — engine creation, GPU memory growth, the `from_pretrained` leak.
- `@docs/generation_parameters_audit.md` — sampling-params / multimodal / KV-cache retention.
- `@hidden_states_leak_audit.md` (repo root) — long-form trace behind §3–§4.
- `@kv_cache_leak_audit.md` (repo root) — the static analysis behind `9b48910`: 5 leak points around `abort` / `yield` / `cuda_memory_manager` / `PositionalEmbeddingsCorrecter`.
- `@docs/auralis_report.md`, `@docs/streaming_audio_analysis.md` — pipeline trace and yield levels.

## Editing checklist

- Before changing `process_tokens_to_speech` or `get_model_logits`, re-read §2: cleanup in `finally`, locals pre-declared, `yield` outside `decoder_semaphore` / `cuda_memory_manager`, `abort` + `clear_request` + `del` + `empty_cache` before yielding.
- Before changing `_make_sentence_generator`: it is laziness-only, contains no cleanup. Do not add per-sentence eager allocations here.
- Before changing `@src/auralis/models/xttsv2/components/vllm_mm_gpt.py:678` (the `compute_logits` hijack guard), re-read §9 — the guard is load-bearing.
- Before changing anything in `hidden_state_collector.py`, run the leak regression test; every early return in `get_hidden_states` must call `_cleanup_request`; stored tensors must be `detach().clone()`.
- Before adding new `asyncio.to_thread` call sites, confirm the resulting threads get released via `TTS.shutdown`'s `shutdown_default_executor()`.
- Prefer minimal, upstream fixes; do not suppress symptoms in callers (see `.windsurf` global bug-fixing rule).

## Git hygiene

- Default branch `main`, origin `github.com/JasperSeling/Auralis.git` (fork of `astramind-ai/auralis`).
- Commit messages follow `fix: <subject>` / `feat: <subject>` with a body that names the root cause, the chosen fix, and any regression test added.
- On Windows, expect `LF will be replaced by CRLF` warnings — benign.
