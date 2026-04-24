# HiddenStatesCollector leak audit

Static audit date: 2026-04-24

Scope:

- `src/auralis/models/xttsv2/components/vllm/hidden_state_collector.py`
- `src/auralis/models/xttsv2/components/vllm/hijack.py`
- `src/auralis/models/xttsv2/XTTSv2.py`
  - `get_model_logits()`
  - `get_generation_context()`
  - `process_tokens_to_speech()`

No TTS generation was run.

## Summary

The current `HiddenStatesCollector` no longer has the old obvious dict/thread leak: successful, timeout, error, and shutdown paths all remove per-request state.

However, the surrounding XTTSv2/vLLM lifecycle still has two high-risk retention points that can explain GPU memory growth after each `save_stream()`:

1. `get_model_logits()` creates a second logits-only vLLM request per sentence/chunk, but does not explicitly abort that logits request after consuming it.
2. `process_tokens_to_speech()` yields `TTSOutput` while still inside `cuda_memory_manager()`, so `torch.cuda.empty_cache()` is delayed until the caller asks for the next chunk. On the final chunk of a request, that cleanup may be delayed until generator finalization.

The highest-value minimal patch is:

- explicitly abort the logits-only request in `get_model_logits()`;
- release local references in `get_model_logits()`;
- move `yield TTSOutput(...)` outside `cuda_memory_manager()`;
- abort/free the original vLLM request before yielding the CPU audio chunk;
- explicitly delete `hidden_states`/`output`/temporary references and call `torch.cuda.empty_cache()` after each sentence.

## 1. Где создаётся (файл + строка)

### Engine-wide collector

File: `src/auralis/models/xttsv2/XTTSv2.py`

Lines 157-165:

```python
# Single HiddenStatesCollector shared across every get_model_logits call.
# Previously a fresh instance was constructed per sentence (per-request);
# because vLLM retains our SyncCollectorWrapper via sampling_params for
# the lifetime of the finished RequestOutput, each of those collectors
# stayed pinned in memory — contributing ~1GB RSS growth over a 900-
# sentence job. The class was already designed to be multi-request
# thread-safe (per-request dicts keyed by request_id), so one instance
# is the intended usage. See components/vllm/hidden_state_collector.py.
self.hidden_states_collector = HiddenStatesCollector()
```

This creates one shared `HiddenStatesCollector` per `XTTSv2Engine` instance. That is good. It is not created per `TTSRequest`.

### Per-logits-request wrapper

File: `src/auralis/models/xttsv2/XTTSv2.py`

Lines 681-692:

```python
# Reuse the engine-wide collector. bind_to_request installs fresh
# per-request state within the shared instance so concurrent calls
# remain isolated.
bound_collector = self.hidden_states_collector.bind_to_request(request_id)

# Set up sampling parameters with the bound collector
sampling_params = ExtendedSamplingParams(
    detokenize=False,
    request_id=request_id,
    max_tokens=1,
    hidden_state_collector=bound_collector,
    output_kind=RequestOutputKind.FINAL_ONLY
)
```

`bind_to_request()` does not create a new collector. It creates per-request dict entries and returns a `SyncCollectorWrapper`.

### `SyncCollectorWrapper`

File: `src/auralis/models/xttsv2/components/vllm/hidden_state_collector.py`

Lines 17-26:

```python
def __init__(self, collector_fn: Callable[[torch.Tensor, str], None], request_id: str):
    self.collector_fn = collector_fn
    self.request_id = request_id
```

This wrapper retains:

- `collector_fn`, currently a lambda closing over the engine-wide collector;
- `request_id`.

It does not directly retain tensors.

### `ExtendedSamplingParams`

File: `src/auralis/models/xttsv2/components/vllm/hijack.py`

Lines 9-22:

```python
class ExtendedSamplingParams(SamplingParams, kw_only=True):
    hidden_state_collector: Optional[HiddenStatesCollector] = None
    request_id: Optional[str] = None
```

Every logits-only vLLM request stores the `SyncCollectorWrapper` in `sampling_params.hidden_state_collector`.

### vLLM call site that invokes the collector

File: `src/auralis/models/xttsv2/components/vllm_mm_gpt.py`

Lines 671-682:

```python
hidden_states = self.final_norm(hidden_states)

last_collected_idx = 0
for seq in sampling_metadata.seq_groups:
    sampling_params = seq.sampling_params
    if (hasattr(sampling_params, 'hidden_state_collector')
            and sampling_params.hidden_state_collector is not None):
        self.positional_embeddings_correcter.clear_request(sampling_params.request_id)
        sampling_params.hidden_state_collector(
            hidden_states[last_collected_idx:last_collected_idx+seq.seq_len],
            sampling_params.request_id
        )
```

This passes a GPU hidden-state slice into the collector callback.

## 2. Где должен освобождаться (файл + строка)

### Collector per-request state cleanup

File: `src/auralis/models/xttsv2/components/vllm/hidden_state_collector.py`

Lines 182-199:

```python
def _cleanup_request(self, request_id: str):
    with self.global_lock:
        self.outputs.pop(request_id, None)
        self.collection_ready.pop(request_id, None)
        self.collection_complete.pop(request_id, None)
        self.locks.pop(request_id, None)
        self.states_count.pop(request_id, None)
        self.expected_states.pop(request_id, None)
        self.notifications.pop(request_id, None)
        self.logger.debug(f"Cleaned up request {request_id}")
```

This is the main collector cleanup. It removes references to hidden-state tensor clones stored in `self.outputs[request_id]`.

### Successful retrieval path

File: `src/auralis/models/xttsv2/components/vllm/hidden_state_collector.py`

Lines 167-170:

```python
result = torch.cat(outputs, dim=0)
self._cleanup_request(request_id)
return result
```

This frees the list of cloned hidden states from `self.outputs`.

Important nuance:

- `torch.cat(outputs, dim=0)` creates a new tensor, usually on GPU because `outputs` are GPU tensors.
- `_cleanup_request()` drops the individual cloned tensors.
- the returned `result` tensor remains alive in `get_model_logits()`.

### Timeout path

File: `src/auralis/models/xttsv2/components/vllm/hidden_state_collector.py`

Lines 147-154:

```python
completed = self.collection_complete[request_id].wait(timeout)
if not completed:
    self._cleanup_request(request_id)
    return None
```

Timeouts are cleaned.

### No-output error path

File: `src/auralis/models/xttsv2/components/vllm/hidden_state_collector.py`

Lines 156-165:

```python
outputs = self.outputs.get(request_id, [])
if not outputs:
    self.logger.critical(...)
    self._cleanup_request(request_id)
    raise ValueError(...)
```

No-output errors are cleaned.

### General exception path

File: `src/auralis/models/xttsv2/components/vllm/hidden_state_collector.py`

Lines 176-180:

```python
except Exception as e:
    self.logger.error(f"Error retrieving hidden states: {e}")
    self._cleanup_request(request_id)
    return None
```

Unexpected errors are cleaned.

### Engine shutdown

File: `src/auralis/models/xttsv2/components/vllm/hidden_state_collector.py`

Lines 201-221:

```python
def shutdown(self) -> None:
    with self.global_lock:
        for request_id in list(self.outputs.keys()):
            self.outputs.pop(request_id, None)
            self.collection_ready.pop(request_id, None)
            self.collection_complete.pop(request_id, None)
            self.locks.pop(request_id, None)
            self.states_count.pop(request_id, None)
            self.expected_states.pop(request_id, None)
            self.notifications.pop(request_id, None)
        self.logger.debug("HiddenStatesCollector shutdown: released all per-request state")
```

File: `src/auralis/models/xttsv2/XTTSv2.py`

Lines 861-868:

```python
async def shutdown(self):
    try:
        self.hidden_states_collector.shutdown()
    except Exception as e:
        self.logger.warning(f"HiddenStatesCollector shutdown failed: {e}")
    self.llm_engine.shutdown_background_loop()
```

This only runs when the TTS/engine is shut down, not after each `save_stream()`.

### CUDA cache cleanup

File: `src/auralis/models/xttsv2/XTTSv2.py`

Lines 498-509:

```python
@asynccontextmanager
async def cuda_memory_manager(self):
    try:
        yield
    finally:
        torch.cuda.synchronize()
        await asyncio.sleep(0.1)
        torch.cuda.empty_cache()
```

This is intended to run after the decoder block, but because the current code yields inside the context manager, cleanup is delayed.

## 3. Где НЕ освобождается — точка утечки

### Leak point A: logits-only vLLM request is not aborted

File: `src/auralis/models/xttsv2/XTTSv2.py`

Lines 695-707:

```python
generator = self.llm_engine.generate(
    prompt=engine_inputs,
    sampling_params=sampling_params,
    request_id=request_id
)

async for output in generator:  # consume the generator
    if output.finished:
        pass

# Get the collected hidden states
hidden_states = await self.hidden_states_collector.get_hidden_states(request_id)
```

Issue:

- this creates a separate vLLM request id: `f"{output.request_id}_logits"`;
- it consumes the generator;
- it never calls `await self.llm_engine.abort(request_id)` for the logits-only request;
- its `ExtendedSamplingParams` still contains `hidden_state_collector=bound_collector`;
- vLLM may retain the finished `SequenceGroup`, `sampling_params`, prompt, `multi_modal_data`, and callback wrapper longer than expected.

Current code aborts only the original audio-token request later:

File: `src/auralis/models/xttsv2/XTTSv2.py`

Lines 852-853:

```python
await self.llm_engine.abort(output.request_id)
```

That is not the logits-only request. The logits-only request id is modified at line 669:

```python
request_id = f"{request_id}_logits"
```

### Leak point B: GPU tensor clone is kept until `get_hidden_states()` cleanup

File: `src/auralis/models/xttsv2/components/vllm/hidden_state_collector.py`

Lines 111-114:

```python
with self.locks[request_id]:
    if hidden_states is not None:
        self.outputs[request_id].append(hidden_states.clone())
        self.states_count[request_id] += 1
```

Issue:

- `hidden_states` is a GPU tensor slice from vLLM;
- `hidden_states.clone()` creates another GPU tensor;
- this clone is retained in `self.outputs[request_id]` until `get_hidden_states()` completes.

This is short-lived if `get_hidden_states()` succeeds. It is not the main persistent leak because `_cleanup_request()` runs after `torch.cat()`.

Still, the clone should be detached:

```python
hidden_states.detach().clone()
```

This avoids retaining graph/autograd metadata if inference mode is accidentally not active in a future path.

### Leak point C: returned hidden-state tensor remains live through decoder

File: `src/auralis/models/xttsv2/XTTSv2.py`

Line 716:

```python
return self.final_norm(hidden_states[start_of_audio_hs:-5, ...].unsqueeze(0).to(self.device).to(self.dtype))
```

Issue:

- `hidden_states` from `get_hidden_states()` is a GPU tensor;
- `self.final_norm(...)` returns another GPU tensor;
- the intermediate `hidden_states` is not explicitly deleted before return.

Python will normally drop locals when the coroutine returns, but explicit cleanup is safer in a hot per-sentence path.

### Leak point D: `yield` inside `cuda_memory_manager()` delays `empty_cache()`

File: `src/auralis/models/xttsv2/XTTSv2.py`

Lines 831-843:

```python
async with self.decoder_semaphore:
    async with self.cuda_memory_manager():
        wav = (await asyncio.to_thread(self.hifigan_decoder,
                hidden_states,
                g=speaker_embeddings
            )).cpu().detach().numpy().squeeze()

        # yield the audio output
        yield TTSOutput(array= wav,
                        start_time = request.start_time,
                        token_length = len(output.outputs[0].token_ids)
                        )
```

Issue:

- the async generator suspends at `yield`;
- while suspended, it is still inside `async with self.cuda_memory_manager()`;
- `cuda_memory_manager.__aexit__()` has not run;
- therefore `torch.cuda.synchronize()` and `torch.cuda.empty_cache()` do not run until the next `anext()` resumes the generator;
- on the final chunk of a request, cleanup can be delayed until generator finalization.

This is the most concrete per-chunk cleanup bug in the inspected code.

### Leak point E: original vLLM request is aborted after the delayed yield

File: `src/auralis/models/xttsv2/XTTSv2.py`

Lines 840-853:

```python
yield TTSOutput(...)

try:
    await self.llm_engine.abort(output.request_id)
except Exception as e:
    ...
```

Issue:

- `abort(output.request_id)` happens only after the caller requests the next item;
- while the caller writes the chunk, `output`, token ids, vLLM request metadata, `hidden_states`, `multimodal_data`, and decoder temporaries may remain live;
- on final chunk, this can be delayed until generator close/finalization.

The abort should happen before yielding the CPU-only `TTSOutput`.

## 4. GPU тензоры — список всех retained объектов

### `hidden_states` from vLLM model forward

File: `src/auralis/models/xttsv2/components/vllm_mm_gpt.py`

Lines 648-661:

```python
hidden_states = self.gpt(...)
return hidden_states
```

This is the model hidden state tensor inside vLLM.

### Normalized `hidden_states` in `compute_logits()`

File: `src/auralis/models/xttsv2/components/vllm_mm_gpt.py`

Line 671:

```python
hidden_states = self.final_norm(hidden_states)
```

This is the tensor passed to the hidden-state collector.

### Slice passed to collector

File: `src/auralis/models/xttsv2/components/vllm_mm_gpt.py`

Line 682:

```python
sampling_params.hidden_state_collector(
    hidden_states[last_collected_idx:last_collected_idx+seq.seq_len],
    sampling_params.request_id
)
```

This slice references GPU storage from the normalized hidden states.

### Collector clone

File: `src/auralis/models/xttsv2/components/vllm/hidden_state_collector.py`

Line 113:

```python
self.outputs[request_id].append(hidden_states.clone())
```

This is a cloned GPU tensor retained in:

```python
self.outputs[request_id]
```

It is removed by `_cleanup_request()`.

### Concatenated collector result

File: `src/auralis/models/xttsv2/components/vllm/hidden_state_collector.py`

Line 168:

```python
result = torch.cat(outputs, dim=0)
```

This is a new GPU tensor returned to `XTTSv2Engine.get_model_logits()`.

### `hidden_states` local in `get_model_logits()`

File: `src/auralis/models/xttsv2/XTTSv2.py`

Line 707:

```python
hidden_states = await self.hidden_states_collector.get_hidden_states(request_id)
```

This retains the concatenated collector result.

### Final-normalized decoder input

File: `src/auralis/models/xttsv2/XTTSv2.py`

Line 716:

```python
return self.final_norm(hidden_states[start_of_audio_hs:-5, ...].unsqueeze(0).to(self.device).to(self.dtype))
```

This returned GPU tensor is assigned in `process_tokens_to_speech()`:

File: `src/auralis/models/xttsv2/XTTSv2.py`

Lines 818-828:

```python
hidden_states = await self.get_model_logits(...)
```

### Decoder output before CPU conversion

File: `src/auralis/models/xttsv2/XTTSv2.py`

Lines 833-836:

```python
wav = (await asyncio.to_thread(self.hifigan_decoder,
        hidden_states,
        g=speaker_embeddings
    )).cpu().detach().numpy().squeeze()
```

The tensor returned by `self.hifigan_decoder(...)` is GPU until `.cpu()` completes.

### `gpt_embed_inputs`

File: `src/auralis/models/xttsv2/XTTSv2.py`

Lines 737-745 and 789:

```python
tokens_list, gpt_embed_inputs, speaker_embeddings = await self.prepare_inputs_async(...)
...
return generators, requests_id, speaker_embeddings, gpt_embed_inputs
```

This conditioning tensor/list is intentionally retained for second phase decoding. It can be large, but it is part of the generation context rather than the `HiddenStatesCollector`.

### `multimodal_data`

File: `src/auralis/models/xttsv2/XTTSv2.py`

Lines 794-797 and 822:

```python
multimodal_data: Optional[torch.Tensor] = None
...
'embeds': multimodal_data
```

This is passed into the logits-only request. It may be retained by vLLM request metadata until the logits request is cleaned.

## Answers to the audit questions

### 1. Где создаётся HiddenStatesCollector?

Created once in `XTTSv2Engine.__init__()`:

- `src/auralis/models/xttsv2/XTTSv2.py:165`

```python
self.hidden_states_collector = HiddenStatesCollector()
```

Per-request wrappers are created in:

- `src/auralis/models/xttsv2/XTTSv2.py:684`
- `src/auralis/models/xttsv2/components/vllm/hidden_state_collector.py:223`

### 2. Где он освобождается?

The collector object itself is released only with the engine. Its per-request state is freed by:

- `_cleanup_request()` in `hidden_state_collector.py:182`;
- success path in `hidden_state_collector.py:169`;
- timeout path in `hidden_state_collector.py:153`;
- no-output path in `hidden_state_collector.py:163`;
- exception path in `hidden_state_collector.py:179`;
- engine shutdown in `XTTSv2.py:865`.

There is no per-sentence `del bound_collector` or `sampling_params.hidden_state_collector = None`.

### 3. Удерживает ли он GPU тензоры после завершения генерации?

The collector dicts should not retain tensors after successful `get_hidden_states()`, because `_cleanup_request()` removes `self.outputs[request_id]`.

But the surrounding vLLM request may retain `sampling_params.hidden_state_collector`, and the logits-only vLLM request is not explicitly aborted. That means the wrapper and request metadata can survive longer than intended.

The collector itself briefly retains GPU tensor clones here:

- `hidden_state_collector.py:113`

```python
self.outputs[request_id].append(hidden_states.clone())
```

### 4. Освобождается ли bound_collector после каждого предложения?

Not explicitly.

`bound_collector` is a local variable in `get_model_logits()`, but it is also stored into:

```python
sampling_params.hidden_state_collector=bound_collector
```

If vLLM retains `sampling_params` for a finished logits-only request, the wrapper remains retained too.

There is no:

```python
sampling_params.hidden_state_collector = None
del bound_collector
```

and there is no logits-request:

```python
await self.llm_engine.abort(request_id)
```

### 5. Вызывается ли torch.cuda.empty_cache() после генерации?

It is called in `cuda_memory_manager()`:

- `src/auralis/models/xttsv2/XTTSv2.py:509`

```python
torch.cuda.empty_cache()
```

But current `process_tokens_to_speech()` yields inside that context manager, so cleanup is delayed until the next generator resume.

This means it is not guaranteed to run promptly after each generated sentence/chunk.

## 5. Патч — минимальные изменения с diff

This patch focuses on prompt cleanup after each logits-only request and avoids suspending the async generator while GPU cleanup is pending.

```diff
diff --git a/src/auralis/models/xttsv2/components/vllm/hidden_state_collector.py b/src/auralis/models/xttsv2/components/vllm/hidden_state_collector.py
--- a/src/auralis/models/xttsv2/components/vllm/hidden_state_collector.py
+++ b/src/auralis/models/xttsv2/components/vllm/hidden_state_collector.py
@@
             with self.locks[request_id]:
                 if hidden_states is not None:
-                    self.outputs[request_id].append(hidden_states.clone())
+                    self.outputs[request_id].append(hidden_states.detach().clone())
                     self.states_count[request_id] += 1
                     self.logger.debug(f"Collected state {self.states_count[request_id]} for request {request_id}")
```

```diff
diff --git a/src/auralis/models/xttsv2/XTTSv2.py b/src/auralis/models/xttsv2/XTTSv2.py
--- a/src/auralis/models/xttsv2/XTTSv2.py
+++ b/src/auralis/models/xttsv2/XTTSv2.py
@@
     async def get_model_logits(
             self,
             token_ids: List[int],
             conditioning: MultiModalDataDict,
             request_id: str,
     ) -> torch.Tensor:
@@
-        request_id = f"{request_id}_logits"
+        request_id = f"{request_id}_logits"
+        generator = None
+        output = None
+        sampling_params = None
+        bound_collector = None
@@
-        bound_collector = self.hidden_states_collector.bind_to_request(request_id)
-
-        # Set up sampling parameters with the bound collector
-        sampling_params = ExtendedSamplingParams(
-            detokenize=False,
-            request_id=request_id,
-            max_tokens=1,
-            hidden_state_collector=bound_collector,
-            output_kind=RequestOutputKind.FINAL_ONLY
-        )
-
-        # Generate with unique request ID
-        generator = self.llm_engine.generate(
-            prompt=engine_inputs,
-            sampling_params=sampling_params,
-            request_id=request_id
-        )
-
-        async for output in generator:  # consume the generator
-            if output.finished:
-                pass
-
-        # Get the collected hidden states
-        hidden_states = await self.hidden_states_collector.get_hidden_states(request_id)
-
-        if hidden_states is None:
-            raise RuntimeError(
-                f"No hidden states collected for request {request_id}. "
-                f"This should never happen! Please report this issue on GitHub."
-            )
-        start_of_audio_hs = conditioning["audio"]["embeds"].shape[0] # type: ignore
-        # Successfully got hidden states
-        return self.final_norm(hidden_states[start_of_audio_hs:-5, ...].unsqueeze(0).to(self.device).to(self.dtype))
+        try:
+            bound_collector = self.hidden_states_collector.bind_to_request(request_id)
+
+            # Set up sampling parameters with the bound collector
+            sampling_params = ExtendedSamplingParams(
+                detokenize=False,
+                request_id=request_id,
+                max_tokens=1,
+                hidden_state_collector=bound_collector,
+                output_kind=RequestOutputKind.FINAL_ONLY
+            )
+
+            # Generate with unique request ID
+            generator = self.llm_engine.generate(
+                prompt=engine_inputs,
+                sampling_params=sampling_params,
+                request_id=request_id
+            )
+
+            async for output in generator:  # consume the generator
+                if output.finished:
+                    pass
+
+            # Get the collected hidden states
+            hidden_states = await self.hidden_states_collector.get_hidden_states(request_id)
+
+            if hidden_states is None:
+                raise RuntimeError(
+                    f"No hidden states collected for request {request_id}. "
+                    f"This should never happen! Please report this issue on GitHub."
+                )
+            start_of_audio_hs = conditioning["audio"]["embeds"].shape[0] # type: ignore
+            result = self.final_norm(
+                hidden_states[start_of_audio_hs:-5, ...]
+                .unsqueeze(0)
+                .to(self.device)
+                .to(self.dtype)
+            )
+            del hidden_states
+            return result
+        finally:
+            # The logits-only pass is a separate vLLM request. Abort it just
+            # like the main audio-token request so vLLM can release its
+            # SequenceGroup, SamplingParams, multimodal data and callback refs.
+            try:
+                await self.llm_engine.abort(request_id)
+            except Exception as e:
+                self.logger.debug(f"llm_engine.abort({request_id}) failed: {e}")
+            if sampling_params is not None:
+                sampling_params.hidden_state_collector = None
+            del bound_collector, sampling_params, generator, output, engine_inputs, conditioning
+            if torch.cuda.is_available():
+                torch.cuda.empty_cache()
```

```diff
diff --git a/src/auralis/models/xttsv2/XTTSv2.py b/src/auralis/models/xttsv2/XTTSv2.py
--- a/src/auralis/models/xttsv2/XTTSv2.py
+++ b/src/auralis/models/xttsv2/XTTSv2.py
@@
         async for output in generator:
 
             if output.finished:
+                output_request_id = output.request_id
+                token_ids = list(output.outputs[0].token_ids)
+                token_length = len(token_ids)
+                tts_output = None
                 # get the hidden states
                 hidden_states = await self.get_model_logits(
-                    list(output.outputs[0].token_ids),
+                    token_ids,
                     {
                         "audio": {
                             'embeds': multimodal_data,  # Use multimodal data for conditioning
                             "is_logits_only_mode": True,
                             "sequence_length": False # to be inserted later
                         },
                     },
-                    output.request_id
+                    output_request_id
                 )
 
-
-                async with self.decoder_semaphore:
-                    async with self.cuda_memory_manager():
-                        wav = (await asyncio.to_thread(self.hifigan_decoder,
-                                hidden_states,
-                                g=speaker_embeddings
-                            )).cpu().detach().numpy().squeeze()
-                         # noqa
-
-                        # yield the audio output
-                        yield TTSOutput(array= wav,
-                                        start_time = request.start_time,
-                                        token_length = len(output.outputs[0].token_ids)
-                                        )
-
-                # Free vLLM-internal metadata for the finished request now
-                # instead of waiting for the caller to release the generator.
-                # Every pending request keeps its SequenceGroup, multi-modal
-                # data (audio embeds), token_ids history and our sampling
-                # params alive — ~1 MB per sentence. On a 900-sentence job
-                # that is ~900 MB of avoidable RSS growth. abort() is a
-                # no-op for already-cleaned requests in vLLM 0.6.x.
-                try:
-                    await self.llm_engine.abort(output.request_id)
-                except Exception as e:  # best-effort — do not break streaming
-                    self.logger.debug(
-                        f"llm_engine.abort({output.request_id}) failed: {e}"
-                    )
+                try:
+                    async with self.decoder_semaphore:
+                        async with self.cuda_memory_manager():
+                            wav = (await asyncio.to_thread(
+                                self.hifigan_decoder,
+                                hidden_states,
+                                g=speaker_embeddings
+                            )).cpu().detach().numpy().squeeze()
+                            tts_output = TTSOutput(
+                                array=wav,
+                                start_time=request.start_time,
+                                token_length=token_length,
+                            )
+                            del wav
+                finally:
+                    del hidden_states
+                    # Free vLLM-internal metadata for the finished request
+                    # before yielding the CPU audio chunk to the caller.
+                    try:
+                        await self.llm_engine.abort(output_request_id)
+                    except Exception as e:  # best-effort — do not break streaming
+                        self.logger.debug(
+                            f"llm_engine.abort({output_request_id}) failed: {e}"
+                        )
+                    del output, token_ids
+                    if torch.cuda.is_available():
+                        torch.cuda.empty_cache()
+
+                if tts_output is not None:
+                    yield tts_output
```

## Why this patch is minimal

It does not change generation semantics:

- same prompts;
- same sampling params;
- same hidden-state extraction;
- same HiFi-GAN decode;
- same `TTSOutput` content.

It only changes cleanup timing:

- logits-only request is aborted;
- collector callback reference is removed from `sampling_params`;
- local references are dropped earlier;
- `empty_cache()` is no longer delayed by yielding inside `cuda_memory_manager()`;
- original request metadata is aborted before yielding the CPU chunk.

## Verification plan

Do not run full audiobook generation for this verification.

Suggested quick tests:

1. Run existing collector unit tests:

```powershell
pytest tests/unit/test_hidden_states_collector_leak.py
```

2. Add a focused test/mock around `get_model_logits()` to assert `llm_engine.abort(logits_request_id)` is called.

3. Add an integration smoke test with a very short text, then record:

```python
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
```

before and after two sequential `save_stream()` calls.

Expected after patch:

- `memory_allocated()` should return closer to baseline after each request;
- `memory_reserved()` may stay high because CUDA allocator/vLLM can reserve memory;
- no monotonic +2 GiB `allocated` growth across requests.

## Final diagnosis

This is not primarily a leak in `HiddenStatesCollector._cleanup_request()`: that cleanup exists and is covered by tests.

The concrete leak/retention points are around it:

- logits-only vLLM requests created in `get_model_logits()` are not explicitly aborted;
- `ExtendedSamplingParams.hidden_state_collector` can remain referenced by vLLM request metadata;
- `process_tokens_to_speech()` delays `cuda_memory_manager` cleanup by yielding inside the context manager;
- the original vLLM request is aborted after the delayed yield instead of before yielding the CPU chunk.

The minimal patch above should be applied before deeper vLLM scheduler/cache changes.
