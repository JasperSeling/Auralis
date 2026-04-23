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
  1. **`_prepare_generation_context`** — tokenises, builds speaker / GPT conditioning, and eagerly calls `llm_engine.generate()` for every sentence in a sub-request, collecting async generators into `generators` + `requests_id` lists.
  2. **`_second_phase_fn` → `process_tokens_to_speech`** — iterates each generator, and for each finished sentence runs a **second** `llm_engine.generate(max_tokens=1)` via `get_model_logits` solely to collect hidden states, then `hifigan_decoder` on a thread via `asyncio.to_thread`.
- `TTS.save_stream` / `save_stream_async` are O(1)-RAM disk writers — audio is not accumulated in Python.
- Sub-request splitter: `TTS.split_requests(request, max_length=100_000)` — **splits by character count, not sentence boundaries**. Sentence-level splitting happens inside `tokenizer.batch_encode_with_split`, per sub-request.

## Known bugs / gotchas — keep these in mind

1. **`XTTSv2.get_memory_usage_curve` is numerically broken.** The empirical polynomial (`@src/auralis/models/xttsv2/XTTSv2.py:167-176`) returns GB of model weights, but `BaseAsyncTTSEngine.get_memory_percentage` then feeds that number to vLLM as a *fraction* of VRAM. Result: `gpu_memory_utilization ≈ 0.09` on T4 16GB, `≈ 0.02` on A100 80GB — KV-cache is starved and long generations trigger `Sequence group preempted by RECOMPUTE mode`. **Workaround**: callers pass `gpu_memory_utilization=0.85` explicitly to `TTS().from_pretrained(...)`; the override path in `XTTSv2.__init__` skips the polynomial.
2. **`llm_engine.generate()` is eager.** It submits the request to vLLM's scheduler the moment it is called; it is not a lazy iterator. Building `generators = [...]` in a loop puts hundreds of requests into the vLLM waiting queue simultaneously. Each pinned `SequenceGroup` carries its multi-modal audio embeds and our `ExtendedSamplingParams`. **Always `await self.llm_engine.abort(output.request_id)` after yielding a finished request** to release that metadata promptly.
3. **`HiddenStatesCollector` is shared per engine, not per request.** Constructing one per sentence leaks because vLLM retains references to our `SyncCollectorWrapper` through `sampling_params`. `XTTSv2` builds exactly one in `__init__` and reuses it; `bind_to_request` installs isolated per-request state inside the shared instance. Do not reintroduce per-call instantiation.
4. **Per-request cleanup on every exit path.** `HiddenStatesCollector.get_hidden_states` must call `_cleanup_request` on the happy path *and* on timeout, on no-outputs `ValueError`, and in the outer `except`. The regression test in `@tests/unit/test_hidden_states_collector_leak.py` guards this.
5. **`asyncio.to_thread` grows a default executor.** HiFi-GAN decode and GPT conditioning run on the loop's default `ThreadPoolExecutor`. Worker threads persist until `loop.shutdown_default_executor()` — called from `TTS.shutdown`. **Never** call `self.loop.close()` there: in Colab/Jupyter `self.loop` is the host notebook loop captured via `asyncio.get_running_loop()` in `_ensure_event_loop`.
6. **The legacy V0 hidden-states collection path is still live** (`XTTSv2.get_model_logits` + `ExtendedSamplingParams.hidden_state_collector`). A full migration to V1-style safetensors-on-disk was drafted but never completed. Treat this double-pass pattern as load-bearing until explicitly refactored.
7. **`TTS.shutdown` is `async`.** Use `await` inside it; never call `self.loop.run_until_complete` from a coroutine already scheduled on `self.loop`.

## Progress UX contract

- The rich progress column set uses `MofNCompleteColumn` with `total=None` (indeterminate pulse). **Never** pass `len(self.split_requests(request))` as `total` — that value counts 100k-char input slices, not audio chunks, which produces misleading `604/2` displays.
- `TimeElapsedColumn` is correct here; `TimeRemainingColumn` needs a known total.

## Running

- Install (Linux only): `pip install -e .`
- Unit tests (platform-agnostic): `python -m pytest tests/unit -v`
  - The hidden-states regression test loads the collector module directly via `importlib.util.spec_from_file_location` to avoid pulling in `torio`/vLLM transitive imports — keep that pattern for any pure-state leak tests.
- Entry point for an OpenAI-compatible server: `auralis.openai` (console script in `setup.py`).

## Editing checklist

- Before changing anything in `XTTSv2.py` around sentence generation, re-read the eager-generate + abort pattern above.
- Before changing anything in `hidden_state_collector.py`, run the leak regression test; every early return in `get_hidden_states` must call `_cleanup_request`.
- Before adding new `asyncio.to_thread` call sites, confirm the resulting threads get released via `TTS.shutdown`'s `shutdown_default_executor()`.
- Prefer minimal, upstream fixes; do not suppress symptoms in callers (see `.windsurf` global bug-fixing rule).

## Git hygiene

- Default branch `main`, origin `github.com/JasperSeling/Auralis.git` (fork of `astramind-ai/auralis`).
- Commit messages follow `fix: <subject>` / `feat: <subject>` with a body that names the root cause, the chosen fix, and any regression test added.
- On Windows, expect `LF will be replaced by CRLF` warnings — benign.
