# vLLM engine lifecycle audit

Static audit date: 2026-04-24

Scope:

- `src/auralis/models/xttsv2/XTTSv2.py`
- `src/auralis/core/tts.py`
- `src/auralis/common/scheduling/two_phase_scheduler.py`
- grep across `src` and `tests` for:
  - `gpu_memory_utilization`
  - `AsyncLLMEngine`
  - `LLMEngine`
  - `EngineArgs`
  - `SamplingParams`
  - `ExtendedSamplingParams`

No TTS generation was run.

## Executive summary

The current code does not recreate the vLLM engine for every `TTSRequest` in the normal `tts.save_stream(request, path)` path.

The vLLM engine is created once when the XTTSv2 model instance is constructed during `TTS.from_pretrained()`.

Observed GPU memory growth between sequential `save_stream()` calls is therefore more likely caused by one of these:

- vLLM KV-cache allocation/high-water behavior;
- vLLM request metadata not being released promptly;
- retained `RequestOutput`, `SamplingParams`, multimodal conditioning, or hidden-state collector references;
- PyTorch/CUDA allocator caching;
- repeated calls to `TTS.from_pretrained()` on the same `TTS` object without shutting down the previous engine.

There is one real lifecycle risk: `TTS.from_pretrained()` overwrites `self.tts_engine` without shutting down an existing engine. If user code calls `from_pretrained()` repeatedly, old vLLM engines can remain alive and keep GPU memory.

## Where vLLM engine is created

### Import

File: `src/auralis/models/xttsv2/XTTSv2.py`

```python
from vllm import AsyncLLMEngine, AsyncEngineArgs, TokensPrompt, RequestOutput
```

This only imports vLLM types at module import time.

### `XTTSv2Engine.__init__()`

File: `src/auralis/models/xttsv2/XTTSv2.py`

Relevant lines:

```python
self.get_memory_usage_curve()

# Initialize VLLM engine at the end, settings its concurrency
self.init_vllm_engine(self.max_concurrency)
```

This is the only internal call to `init_vllm_engine()`.

`rg` result:

```text
src\auralis\models\xttsv2\XTTSv2.py:151:        self.init_vllm_engine(self.max_concurrency)
src\auralis\models\xttsv2\XTTSv2.py:213:    def init_vllm_engine(self, concurrency):
```

### `XTTSv2Engine.init_vllm_engine()`

File: `src/auralis/models/xttsv2/XTTSv2.py`

Relevant code:

```python
def init_vllm_engine(self, concurrency):
    max_seq_num = concurrency
    if self.gpu_memory_utilization is not None:
        mem_utils = self.gpu_memory_utilization
        self.logger.info(
            f"Using user-supplied gpu_memory_utilization={mem_utils:.2f}"
        )
    else:
        mem_utils = self.get_memory_percentage(self.max_gb_for_vllm_model * 1024 ** 3)
        if not mem_utils:
            raise RuntimeError(
                "Could not estimate gpu_memory_utilization for vLLM init. "
                "Pass gpu_memory_utilization=<float> to TTS.from_pretrained "
                "to override (recommended on memory-constrained GPUs)."
            )
    engine_args = AsyncEngineArgs(
        model=self.gpt_model,
        tensor_parallel_size=self.tp,
        pipeline_parallel_size=self.pp,
        dtype="auto",
        max_model_len=self.gpt_config.max_text_tokens +
                      self.gpt_config.max_audio_tokens +
                      32 + 5 + 3,
        gpu_memory_utilization=mem_utils,
        trust_remote_code=True,
        enforce_eager=True,
        limit_mm_per_prompt={"audio": 1},
        max_num_seqs=max_seq_num,
        disable_log_stats=True,
        max_num_batched_tokens=(self.gpt_config.max_text_tokens +
                                self.gpt_config.max_audio_tokens +
                                32 + 5 + 3) * max_seq_num,
    )
    self.logger.info(f"Initializing VLLM engine with args: {engine_args}")
    self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
```

This is the exact creation site of the vLLM engine:

```python
self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
```

### Creation path from public API

File: `src/auralis/core/tts.py`

`TTS.from_pretrained()`:

```python
async def _load_model():
    return MODEL_REGISTRY[config['model_type']].from_pretrained(model_name_or_path, **kwargs)

self.tts_engine = self.loop.run_until_complete(_load_model())
```

File: `src/auralis/models/xttsv2/XTTSv2.py`

`XTTSv2Engine.from_pretrained()`:

```python
model = cls(
    hifi_config=hifi_config,
    gpt_config=gpt_config,
    tensor_parallel_size=tensor_parallel_size,
    pipeline_parallel_size=pipeline_parallel_size,
    **kwargs
)
```

`cls(...)` calls `XTTSv2Engine.__init__()`, which calls `init_vllm_engine()`.

## Does vLLM engine get created once or per request?

For the normal path:

```python
tts = TTS(...).from_pretrained(...)
tts.save_stream(request_1, path_1)
tts.save_stream(request_2, path_2)
```

the engine is created once during `from_pretrained()`.

`save_stream()` does not call:

- `from_pretrained()`;
- `XTTSv2Engine.from_pretrained()`;
- `XTTSv2Engine.__init__()`;
- `init_vllm_engine()`;
- `AsyncLLMEngine.from_engine_args()`.

`save_stream()` only:

1. temporarily sets `request.stream = True`;
2. calls `generate_speech(request, _show_progress=False)`;
3. iterates chunks;
4. writes each chunk to a streaming file writer;
5. restores the original `request.stream`.

Relevant code from `src/auralis/core/tts.py`:

```python
original_stream = request.stream
request.stream = True

writer = self._StreamingFileWriter(filename, fmt)

try:
    if progress:
        with self._progress_context(
            None, description, print_summary=False
        ) as advance:
            for chunk in self.generate_speech(request, _show_progress=False):
                writer.write(chunk)
                advance(chunk)
    else:
        for chunk in self.generate_speech(request, _show_progress=False):
            writer.write(chunk)
finally:
    request.stream = original_stream
    writer.close()
```

## Is there a condition that triggers engine recreation?

No request-level condition was found.

No code path changes `EngineArgs` or calls `init_vllm_engine()` during `TTSRequest` processing.

The only found trigger for creating a new engine is constructing a new `XTTSv2Engine`, normally through `TTS.from_pretrained()`.

Potential lifecycle bug:

- `TTS.from_pretrained()` can be called repeatedly on the same `TTS` instance.
- It overwrites `self.tts_engine`.
- It does not call shutdown on the previous engine before overwriting it.

That repeated-loader scenario can look like vLLM engine recreation and GPU memory growth.

## grep findings with context

### `gpu_memory_utilization`

File: `src/auralis/models/xttsv2/XTTSv2.py`

Context:

```python
self.max_concurrency = kwargs.pop('max_concurrency', 10)
# User-supplied GPU memory utilization override. When None (default),
# init_vllm_engine falls back to the empirical polynomial derived from
# max_concurrency via get_memory_usage_curve(). Exposed so callers can
# force a higher utilization (e.g. 0.95) to reduce KV-cache preemption
# on memory-constrained GPUs like T4, where the default can leave too
# little headroom for the KV-cache of parallel sequences.
self.gpu_memory_utilization: Optional[float] = kwargs.pop('gpu_memory_utilization', None)
```

When it runs:

- once in `XTTSv2Engine.__init__()`;
- value comes from `TTS.from_pretrained(..., gpu_memory_utilization=<float>)`.

Context in `init_vllm_engine()`:

```python
if self.gpu_memory_utilization is not None:
    mem_utils = self.gpu_memory_utilization
    self.logger.info(
        f"Using user-supplied gpu_memory_utilization={mem_utils:.2f}"
    )
else:
    mem_utils = self.get_memory_percentage(self.max_gb_for_vllm_model * 1024 ** 3)
```

When it runs:

- once when vLLM engine is initialized;
- not per `TTSRequest`.

Context in `AsyncEngineArgs`:

```python
engine_args = AsyncEngineArgs(
    ...
    gpu_memory_utilization=mem_utils,
    ...
)
```

When it runs:

- once during `AsyncEngineArgs` construction;
- value becomes part of fixed engine config.

### `AsyncLLMEngine`

File: `src/auralis/models/xttsv2/XTTSv2.py`

Context:

```python
from vllm import AsyncLLMEngine, AsyncEngineArgs, TokensPrompt, RequestOutput
```

When it runs:

- module import time.

Creation context:

```python
self.logger.info(f"Initializing VLLM engine with args: {engine_args}")
self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
```

When it runs:

- once per `XTTSv2Engine` instance;
- normally once per `TTS.from_pretrained()`.

### `LLMEngine`

No direct `LLMEngine` creation or import was found in `src` or `tests`.

### `EngineArgs`

File: `src/auralis/models/xttsv2/XTTSv2.py`

Only `AsyncEngineArgs` is used directly.

Context:

```python
engine_args = AsyncEngineArgs(
    model=self.gpt_model,
    tensor_parallel_size=self.tp,
    pipeline_parallel_size=self.pp,
    dtype="auto",
    max_model_len=self.gpt_config.max_text_tokens +
                  self.gpt_config.max_audio_tokens +
                  32 + 5 + 3,
    gpu_memory_utilization=mem_utils,
    trust_remote_code=True,
    enforce_eager=True,
    limit_mm_per_prompt={"audio": 1},
    max_num_seqs=max_seq_num,
    disable_log_stats=True,
    max_num_batched_tokens=(self.gpt_config.max_text_tokens +
                            self.gpt_config.max_audio_tokens +
                            32 + 5 + 3) * max_seq_num,
)
```

When it runs:

- once during `init_vllm_engine()`.

### `SamplingParams` / `ExtendedSamplingParams`

File: `src/auralis/models/xttsv2/components/vllm/hijack.py`

Context:

```python
from vllm import SamplingParams

class ExtendedSamplingParams(SamplingParams, kw_only=True):
    hidden_state_collector: Optional[HiddenStatesCollector] = None
    request_id: Optional[str] = None
```

When it runs:

- class definition at import time;
- instances are created per vLLM generation request.

File: `src/auralis/models/xttsv2/XTTSv2.py`

Logits-only hidden-state pass:

```python
sampling_params = ExtendedSamplingParams(
    detokenize=False,
    request_id=request_id,
    max_tokens=1,
    hidden_state_collector=bound_collector,
    output_kind=RequestOutputKind.FINAL_ONLY
)
```

When it runs:

- inside `get_model_logits()`;
- after an audio-token sequence has finished;
- per sentence/chunk;
- does not recreate the engine.

Main audio-token generation:

```python
sampling_params = ExtendedSamplingParams(
    temperature=request.temperature,
    top_p=request.top_p,
    detokenize=False,
    request_id=uuid.uuid4(),
    top_k=request.top_k,
    logits_processors=[LogitsRepetitionPenalizer(request.repetition_penalty)],
    repetition_penalty=1.0,
    max_tokens=self.gpt_config.gpt_max_audio_tokens,
    ignore_eos=True,
    stop_token_ids=[self.mel_eos_token_id],
    output_kind=RequestOutputKind.FINAL_ONLY
)
```

When it runs:

- inside `get_generation_context()`;
- once for each text split/sentence sequence;
- does not recreate the engine.

## Request processing path

### `save_stream()`

File: `src/auralis/core/tts.py`

Runs per user call:

```python
def save_stream(self, request: TTSRequest, filename, ...):
    original_stream = request.stream
    request.stream = True
    ...
    for chunk in self.generate_speech(request, _show_progress=False):
        writer.write(chunk)
    ...
    request.stream = original_stream
```

No engine creation.

### `generate_speech()`

File: `src/auralis/core/tts.py`

Runs per user call:

```python
requests = self.split_requests(request)
...
async for chunk in self.scheduler.run(
        inputs=sub_request,
        request_id=sub_request.request_id,
        first_phase_fn=self._prepare_generation_context,
        second_phase_fn=self._second_phase_fn
):
    yield chunk
```

No engine creation.

### `TwoPhaseScheduler`

File: `src/auralis/common/scheduling/two_phase_scheduler.py`

Scheduler startup:

```python
if not self.is_running:
    await self.start()
```

`start()`:

```python
if self.is_running:
    return

self.request_queue = asyncio.Queue()
self.second_phase_sem = asyncio.Semaphore(self.second_phase_concurrency)
self.is_running = True
self.queue_processor_tasks = [
    asyncio.create_task(self._process_queue())
    for _ in range(self.second_phase_concurrency)
]
```

When it runs:

- first scheduler use starts worker tasks;
- subsequent requests reuse the scheduler;
- no vLLM engine creation.

Request lifecycle:

```python
request = QueuedRequest(
    id=request_id,
    input=inputs,
    first_fn=first_phase_fn,
    second_fn=second_phase_fn
)

await self.request_queue.put(request)
```

When it runs:

- per `scheduler.run()`;
- creates scheduler request metadata, not vLLM engine.

## Suspected memory growth mechanisms

### Not engine recreation

These request-varying values do not change `EngineArgs`:

- `request.temperature`
- `request.top_p`
- `request.top_k`
- `request.repetition_penalty`
- `request.max_ref_length`
- `request.gpt_cond_len`
- `request.gpt_cond_chunk_len`
- generated `request_id`
- text split count

They affect conditioning or `SamplingParams`, not engine construction.

### `max_tokens`

Main generation:

```python
max_tokens=self.gpt_config.gpt_max_audio_tokens
```

Logits-only pass:

```python
max_tokens=1
```

These are per-request `SamplingParams`. They do not recreate vLLM engine or KV-cache config.

They can affect how much KV cache a request uses while active.

### `max_model_len`

Engine config:

```python
max_model_len=self.gpt_config.max_text_tokens +
              self.gpt_config.max_audio_tokens +
              32 + 5 + 3
```

This is fixed at engine creation.

### `gpt_cond_len`

`gpt_cond_len` affects reference conditioning:

```python
gpt_cond_latents = await asyncio.to_thread(
    self.get_gpt_cond_latents,
    full_audio,
    load_sr,
    length=gpt_cond_len,
    chunk_length=gpt_cond_chunk_len
)
```

Then `_merge_conditioning()` concatenates audio conditioning and text embedding:

```python
torch.cat([audio_conditioning, text_embedding], dim=1)
```

However, with the perceiver resampler path, conditioning is reduced to a fixed latent count. It changes conditioning computation but does not recreate the engine.

### `abort()` cleanup

File: `src/auralis/models/xttsv2/XTTSv2.py`

```python
try:
    await self.llm_engine.abort(output.request_id)
except Exception as e:
    self.logger.debug(
        f"llm_engine.abort({output.request_id}) failed: {e}"
    )
```

When it runs:

- after `process_tokens_to_speech()` finishes a generator output;
- intended to free vLLM-internal metadata earlier.

Risk:

- if `abort()` is a no-op for already-finished requests in the installed vLLM version;
- if `RequestOutput`, generator, or scheduler buffers retain references;
- if vLLM holds finished sequence metadata until engine cleanup;
- GPU memory can grow without any engine recreation.

## Minimal fix for repeated `from_pretrained()`

This patch protects against a real lifecycle leak when `from_pretrained()` is called multiple times on the same `TTS` instance.

File: `src/auralis/core/tts.py`

```diff
diff --git a/src/auralis/core/tts.py b/src/auralis/core/tts.py
--- a/src/auralis/core/tts.py
+++ b/src/auralis/core/tts.py
@@
     def from_pretrained(self, model_name_or_path: str, **kwargs):
@@
         from auralis.models.registry import MODEL_REGISTRY
 
+        # Avoid leaking an existing vLLM engine if callers reload a model on
+        # the same TTS instance. The normal path creates one engine and reuses
+        # it across TTSRequest objects.
+        self._ensure_event_loop()
+        if self.tts_engine is not None:
+            self.loop.run_until_complete(self.tts_engine.shutdown())
+            self.tts_engine = None
+
-        # Ensure an event loop exists for potential async operations within from_pretrained
-        self._ensure_event_loop()
-
         try:
             with open(os.path.join(model_name_or_path, 'config.json'), 'r') as f:
                 config = json.load(f)
```

This does not fix memory growth between `save_stream()` calls if `from_pretrained()` is not repeated.

## Optional instrumentation patch

To prove whether engine recreation happens, add a monotonic init counter/log.

File: `src/auralis/models/xttsv2/XTTSv2.py`

```diff
diff --git a/src/auralis/models/xttsv2/XTTSv2.py b/src/auralis/models/xttsv2/XTTSv2.py
--- a/src/auralis/models/xttsv2/XTTSv2.py
+++ b/src/auralis/models/xttsv2/XTTSv2.py
@@
 class XTTSv2Engine(BaseAsyncTTSEngine):
+    _vllm_engine_init_count = 0
@@
     def init_vllm_engine(self, concurrency):
@@
+        type(self)._vllm_engine_init_count += 1
+        self.logger.warning(
+            "Initializing vLLM engine #%s for XTTSv2Engine id=%s",
+            type(self)._vllm_engine_init_count,
+            id(self),
+        )
         self.logger.info(f"Initializing VLLM engine with args: {engine_args}")
         self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
```

Expected result:

- one log line for one `TTS.from_pretrained()`;
- no new log line for later `save_stream()` calls.

## Recommended next audit target

If memory grows across sequential requests without repeated `from_pretrained()`, focus on:

- `process_tokens_to_speech()` generator lifetime;
- `get_model_logits()` logits-only `generate()` call;
- `ExtendedSamplingParams.hidden_state_collector`;
- `HiddenStatesCollector` state cleanup;
- scheduler `sequence_buffers`;
- whether `RequestOutput` objects keep multimodal data alive after yielding;
- vLLM behavior of `abort()` on finished requests;
- explicit post-request cleanup such as draining engine state or forcing `torch.cuda.empty_cache()` after all generators finish.

## Final diagnosis

For normal usage, this is not expected engine recreation per `TTSRequest`.

It is likely one of:

- expected vLLM/CUDA allocator memory reservation behavior;
- a request lifecycle leak around vLLM outputs or multimodal metadata;
- a lifecycle bug only if `TTS.from_pretrained()` is called repeatedly without shutdown.

The minimal safe code fix is to shut down an existing engine before replacing it in `TTS.from_pretrained()`. Further fixes for per-request memory growth require auditing request/output retention rather than engine initialization.
