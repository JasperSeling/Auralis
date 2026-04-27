# Conditioning Encoder Grad Leak Audit

Дата анализа: 2026-04-27

Область анализа:
- `src/auralis/models/xttsv2/XTTSv2.py`
- `src/auralis/models/xttsv2/components/vllm_mm_gpt.py`
- `src/auralis/models/xttsv2/config/xttsv2_config.py`
- `src/auralis/models/xttsv2/config/xttsv2_gpt_config.py`
- `src/auralis/core/tts.py`

Примечание по путям: указанные в задаче `src/auralis/models/xttsv2/components/vllm_mm_model.py` и `src/auralis/models/xttsv2/config/xttsconfig.py` в текущем дереве не найдены. Фактические соответствующие файлы: `components/vllm_mm_gpt.py`, `config/xttsv2_config.py`, `config/xttsv2_gpt_config.py`.

## Найденные места

### 1. Speaker encoder запускается без локального no_grad/inference_mode

ФАЙЛ: `src/auralis/models/xttsv2/XTTSv2.py`

МЕТОД: `_get_speaker_embedding`, строки 340-356

ПРОБЛЕМА: вызов `self.hifigan_decoder.speaker_encoder.forward(...)` находится внутри `async def`, но сам метод не обернут в `torch.no_grad()` или `torch.inference_mode()`, внутри тела нет `with torch.no_grad()` / `with torch.inference_mode()`, на выходе нет `.detach()`. Это прямой forward-путь speaker encoder.

КОД:

```python
async def _get_speaker_embedding(self, audio, sr):
    audio_16k = torchaudio.functional.resample(audio, sr, 16000)
    async with self.decoder_semaphore:
        return (
            self.hifigan_decoder.speaker_encoder.forward(audio_16k.to(self.device), l2_norm=True)
            .unsqueeze(-1)
            .to(self.device)
        )
```

РИСК: КРИТИЧЕСКИЙ

### 2. GPT conditioning mel chunks + style encoder без no_grad/inference_mode и без detach

ФАЙЛ: `src/auralis/models/xttsv2/XTTSv2.py`

МЕТОД: `get_gpt_cond_latents`, строки 377-419

ПРОБЛЕМА: `wav_to_mel_cloning(...)` вычисляет mel-спектрограмму, затем `self.get_style_emb(...)` запускает conditioning encoder. Метод не обернут в `torch.no_grad()` или `torch.inference_mode()`, внутри нет контекста, `style_emb` добавляется в список без `.detach()`, затем `torch.stack(style_embs).mean(...)` сохраняет граф.

КОД:

```python
mel_chunk = wav_to_mel_cloning(
    audio_chunk,
    mel_norms=self.mel_stats.cpu(),
    n_fft=2048,
    hop_length=256,
)
style_emb = self.get_style_emb(mel_chunk.to(self.device), None)
style_embs.append(style_emb)
```

РИСК: КРИТИЧЕСКИЙ

### 3. GPT conditioning full-audio mel + style encoder без no_grad/inference_mode и без detach

ФАЙЛ: `src/auralis/models/xttsv2/XTTSv2.py`

МЕТОД: `get_gpt_cond_latents`, строки 421-435

ПРОБЛЕМА: альтернативная ветка `use_perceiver_resampler == False` также вычисляет mel через `wav_to_mel_cloning(...)` и передает его в `self.get_style_emb(...)` без `torch.no_grad()` / `torch.inference_mode()` и без `.detach()` на `cond_latent`.

КОД:

```python
mel = wav_to_mel_cloning(
    audio,
    mel_norms=self.mel_stats.cpu(),
    n_fft=4096,
)
cond_latent = self.get_style_emb(mel.to(self.device))
return cond_latent.transpose(1, 2)
```

РИСК: КРИТИЧЕСКИЙ

### 4. get_style_emb вызывает conditioning_encoder и conditioning_perceiver с autograd

ФАЙЛ: `src/auralis/models/xttsv2/XTTSv2.py`

МЕТОД: `get_style_emb`, строки 511-529

ПРОБЛЕМА: прямой forward-путь GPT style encoder: `self.conditioning_encoder(cond_input)` и `self.conditioning_perceiver(...)`. Метод не имеет `@torch.no_grad()` / `@torch.inference_mode()`, внутри нет контекста, результат `conds` возвращается без `.detach()`. Это место соответствует наблюдаемым `ConvolutionBackward0`, `NativeGroupNormBackward0`, `SoftmaxBackward0`, `ViewBackward0`, `AddBackward0`, `MulBackward0`, `UnsafeViewBackward0`.

КОД:

```python
if cond_input.ndim == 4:
    cond_input = cond_input.squeeze(1)
conds = self.conditioning_encoder(cond_input)

if hasattr(self, 'conditioning_perceiver'):
    conds = self.conditioning_perceiver(
        conds.permute(0, 2, 1)
    ).transpose(1, 2)
```

РИСК: КРИТИЧЕСКИЙ

### 5. get_conditioning_latents удерживает speaker/GPT conditioning outputs без detach

ФАЙЛ: `src/auralis/models/xttsv2/XTTSv2.py`

МЕТОД: `get_conditioning_latents`, строки 437-497

ПРОБЛЕМА: метод собирает outputs из `_get_speaker_embedding(...)` и `get_gpt_cond_latents(...)`, затем возвращает `gpt_cond_latents` и усредненный `speaker_embedding` без `.detach()`. Метод `async def`, локального `torch.no_grad()` / `torch.inference_mode()` внутри тела нет.

КОД:

```python
speaker_embedding = await self._get_speaker_embedding(audio, load_sr)
speaker_embeddings.append(speaker_embedding)

full_audio = torch.cat(audios, dim=-1)
gpt_cond_latents = await asyncio.to_thread(self.get_gpt_cond_latents,
    full_audio, load_sr, length=gpt_cond_len, chunk_length=gpt_cond_chunk_len
)
```

РИСК: КРИТИЧЕСКИЙ

### 6. get_audio_conditioning только проксирует conditioning без inference context

ФАЙЛ: `src/auralis/models/xttsv2/XTTSv2.py`

МЕТОД: `get_audio_conditioning`, строки 607-642

ПРОБЛЕМА: публичный async-вход для аудио conditioning вызывает `get_conditioning_latents(...)` без локального `torch.no_grad()` / `torch.inference_mode()`. На возвращаемом `result` нет `.detach()`. Так как это `async def`, отсутствие контекста внутри метода важно для call sites, которые вызывают его напрямую.

КОД:

```python
async with self.encoder_semaphore:
    result = await self.get_conditioning_latents(
        audio_reference,
        max_ref_length,
        gpt_cond_len,
        gpt_cond_chunk_len,
    )
    return result
```

РИСК: КРИТИЧЕСКИЙ

### 7. prepare_for_streaming_generation напрямую вызывает get_audio_conditioning

ФАЙЛ: `src/auralis/core/tts.py`

МЕТОД: `prepare_for_streaming_generation`, строки 122-136

ПРОБЛЕМА: метод вызывает `self.tts_engine.get_audio_conditioning(request.speaker_files)` без `torch.no_grad()` / `torch.inference_mode()` и затем замыкает returned tensors в `partial(...)`. Если tensors пришли с `grad_fn`, partial удерживает их до использования.

КОД:

```python
if conditioning_config.speaker_embeddings or conditioning_config.gpt_like_decoder_conditioning:
    gpt_cond_latent, speaker_embeddings = await self.tts_engine.get_audio_conditioning(request.speaker_files)
    return partial(self.tts_engine.get_generation_context,
                   gpt_cond_latent=gpt_cond_latent,
                   speaker_embeddings=speaker_embeddings)
```

РИСК: КРИТИЧЕСКИЙ

### 8. get_generation_context помечен @torch.inference_mode(), но async-тело не содержит локального контекста

ФАЙЛ: `src/auralis/models/xttsv2/XTTSv2.py`

МЕТОД: `get_generation_context`, строки 809-863

ПРОБЛЕМА: метод имеет `@torch.inference_mode()`, но это `async def`; в теле нет явного `with torch.inference_mode()`. Внутри вызывается `prepare_inputs_async(...)`, который вызывает `get_audio_conditioning(...)`. Проверяемое условие "если метод async - применяется ли контекст внутри async def?" не выполнено: локального async-body context нет. Возвращаемые `speaker_embeddings` и `gpt_embed_inputs` не detach-ятся.

КОД:

```python
@torch.inference_mode()
async def get_generation_context(self, request: TTSRequest, ...):
    if gpt_cond_latent is None or speaker_embeddings is None:
        tokens_list, gpt_embed_inputs, speaker_embeddings = await self.prepare_inputs_async(
            request.text,
            request.language,
            request.speaker_files,
```

РИСК: ВЫСОКИЙ

### 9. prepare_inputs_async протаскивает conditioning outputs в merge без detach

ФАЙЛ: `src/auralis/models/xttsv2/XTTSv2.py`

МЕТОД: `prepare_inputs_async`, строки 575-605

ПРОБЛЕМА: метод получает `gpt_cond_latent, speaker_embeddings` из `get_audio_conditioning(...)`, затем передает `gpt_cond_latent` в `_merge_conditioning(...)`. Локального `torch.no_grad()` / `torch.inference_mode()` нет, `.detach()` перед merge и return нет.

КОД:

```python
gpt_cond_latent, speaker_embeddings = await self.get_audio_conditioning(
    speaker_file,
    max_ref_length,
    gpt_cond_len,
    gpt_cond_chunk_len
)

cond_latents = await self._merge_conditioning(text_embeddings, gpt_cond_latent)
```

РИСК: ВЫСОКИЙ

### 10. _merge_conditioning возвращает cat(text + audio) без detach

ФАЙЛ: `src/auralis/models/xttsv2/XTTSv2.py`

МЕТОД: `_merge_conditioning`, строки 360-375

ПРОБЛЕМА: `torch.cat([audio_conditioning, text_embedding], ...)` объединяет аудио conditioning с text embedding. Если `audio_conditioning` имеет `grad_fn`, итоговый `cond_latents` тоже удерживает граф. Локального `no_grad` / `inference_mode` и `.detach()` нет.

КОД:

```python
cond_latents = []
for text_embedding in text_conditioning:
    cond_latents.append((torch.cat([audio_conditioning, text_embedding], dim=1).squeeze(0)
                         .to(self.llm_engine.engine.model_config.dtype)))
return cond_latents
```

РИСК: ВЫСОКИЙ

### 11. vLLM multimodal mapper принимает embeds без detach

ФАЙЛ: `src/auralis/models/xttsv2/components/vllm_mm_gpt.py`

МЕТОД: `input_mapper_for_xtts`, строки 274-294

ПРОБЛЕМА: mapper извлекает `dat["embeds"]` и передает их как `cond_latents` в vLLM без `.detach()`. Если `embeds` пришли из conditioning encoder с `grad_fn`, mapper не разрывает граф.

КОД:

```python
embeds = [dat["embeds"] for dat in data]
is_logits_only_mode = [dat.get("is_logits_only_mode", False) for dat in data]
sequence_length = [dat.get("sequence_length", -1) for dat in data]
return MultiModalKwargs(
    {
        "cond_latents": embeds,
```

РИСК: ВЫСОКИЙ

### 12. XttsGPT.forward передает cond_latents в GPT без detach

ФАЙЛ: `src/auralis/models/xttsv2/components/vllm_mm_gpt.py`

МЕТОД: `XttsGPT.forward`, строки 614-659

ПРОБЛЕМА: `cond_latents` преобразуются в list и передаются как `input_embeds` в `self.gpt(...)`. Локального `torch.no_grad()` / `torch.inference_mode()` и `.detach()` нет.

КОД:

```python
if isinstance(cond_latents, torch.Tensor):
    if len(cond_latents.shape) > 4:
        is_profiling_run = True
    else:
        cond_latents = list(cond_latents)

hidden_states = self.gpt(
    input_embeds=cond_latents,
)
```

РИСК: ВЫСОКИЙ

### 13. GPT2Model inserts conditioning tensors into hidden_states without detach

ФАЙЛ: `src/auralis/models/xttsv2/components/vllm_mm_gpt.py`

МЕТОД: `_insert_conditioning_into_hidden_states`, строки 768-784

ПРОБЛЕМА: `conditioning_input.squeeze(0)` вставляется в `hidden_states` через `torch.cat(...)` без `.detach()`. Если conditioning tensor имеет `grad_fn`, итоговые `hidden_states` получают связь с графом conditioning encoder.

КОД:

```python
for idx, (inserion_idx, conditioning_input) in enumerate(zip(insertion_ids, conditioning_inputs)):
        hidden_states = torch.cat([
        hidden_states[:inserion_idx],
        conditioning_input.squeeze(0),
        (start_of_generation_embed if ~is_logit_only[idx] else empty_tensor),
        hidden_states[inserion_idx:]], dim=0
    )
```

РИСК: ВЫСОКИЙ

### 14. core streaming path приводит save_stream к conditioning path

ФАЙЛ: `src/auralis/core/tts.py`

МЕТОД: `save_stream` -> `generate_speech` -> scheduler first phase, строки 606-681 и 455-500

ПРОБЛЕМА: `save_stream()` сам не вызывает mel/encoder напрямую, но запускает `generate_speech(...)`, который в streaming path вызывает scheduler с `first_phase_fn=self._prepare_generation_context`. Этот first phase ведет к `tts_engine.get_generation_context(...)` / `get_audio_conditioning(...)`. В `save_stream` / `generate_speech` нет outer `torch.no_grad()` / `torch.inference_mode()`.

КОД:

```python
for chunk in self.generate_speech(request, _show_progress=False):
    writer.write(chunk)

async for chunk in self.scheduler.run(
        inputs=sub_request,
        request_id=sub_request.request_id,
        first_phase_fn=self._prepare_generation_context,
```

РИСК: ВЫСОКИЙ

### 15. save_stream_async приводит к тому же conditioning path

ФАЙЛ: `src/auralis/core/tts.py`

МЕТОД: `save_stream_async` -> `generate_speech_async`, строки 693-754 и 227-249

ПРОБЛЕМА: async-stream path вызывает `generate_speech_async(...)`, который запускает scheduler first phase `_prepare_generation_context`. Внутри `save_stream_async` и `generate_speech_async` нет локального `torch.no_grad()` / `torch.inference_mode()`.

КОД:

```python
gen = await self.generate_speech_async(request)
async for chunk in gen:
    writer.write(chunk)

async for chunk in self.scheduler.run(
        first_phase_fn=self._prepare_generation_context,
        second_phase_fn=self._second_phase_fn
):
```

РИСК: ВЫСОКИЙ

### 16. PositionalEmbeddingsCorrecter хранит request state в dict, но не tensors с grad_fn

ФАЙЛ: `src/auralis/models/xttsv2/components/vllm_mm_gpt.py`

МЕТОД: `PositionalEmbeddingsCorrecter`, строки 61-163

ПРОБЛЕМА: class хранит `TokenPositionAndPrefillTuple` и строковые ключи в `request_tracker_dict` / `token_to_request`. Прямого хранения CUDA tensors с `grad_fn` в этих dict не найдено. Аргумент `nex_token: torch.Tensor` используется только в f-string ключе. Есть `clear_request(...)`, но полного сброса всего dict между запросами нет; очистка зависит от вызовов `clear_request(request_id)`.

КОД:

```python
self.request_tracker_dict: Dict[str, TokenPositionAndPrefillTuple] = defaultdict(lambda: TokenPositionAndPrefillTuple())
self.token_to_request: Dict[str, str] = {}

def init_request_id_prefill(self, request_id: str, prefill_len: PrefillLength, nex_token: torch.Tensor):
    self.request_tracker_dict[request_id] = TokenPositionAndPrefillTuple(prefill_len, prefill_len)
    self.token_to_request[f"{nex_token}_{prefill_len}"] = request_id
```

РИСК: СРЕДНИЙ

### 17. PositionalEmbeddingsCorrecter очищается частично по request_id

ФАЙЛ: `src/auralis/models/xttsv2/components/vllm_mm_gpt.py`

МЕТОД: `compute_logits` / `sample`, строки 664-710

ПРОБЛЕМА: `compute_logits(...)` вызывает `clear_request(...)` только для logits-only requests с `hidden_state_collector`. `sample(...)` создает/обновляет записи для всех requests с `request_id`. Полная очистка всего внутреннего dict между запросами в этом файле не найдена.

КОД:

```python
if (hasattr(sampling_params, 'hidden_state_collector')
        and sampling_params.hidden_state_collector is not None):
    self.positional_embeddings_correcter.clear_request(sampling_params.request_id)

if not self.positional_embeddings_correcter.get_by_request_id(seq_groups.sampling_params.request_id):
    self.positional_embeddings_correcter.init_request_id_prefill(...)
else:
    self.positional_embeddings_correcter.associate_new_tokens(...)
```

РИСК: СРЕДНИЙ

### 18. process_tokens_to_speech дополнительно чистит PositionalEmbeddingsCorrecter

ФАЙЛ: `src/auralis/models/xttsv2/XTTSv2.py`

МЕТОД: `process_tokens_to_speech`, строки 865-982

ПРОБЛЕМА: метод имеет `@torch.inference_mode()`, но это `async def`, внутри нет локального context. По PositionalEmbeddingsCorrecter здесь есть best-effort очистка `pe.clear_request(output_request_id)` и `pe.clear_request(f"{output_request_id}_logits")`; это снижает риск накопления dict, но не является полным глобальным clear между запросами.

КОД:

```python
pe = (
    self.llm_engine.engine
    .model_executor.driver_worker
    .model_runner.model
    .positional_embeddings_correcter
)
pe.clear_request(output_request_id)
pe.clear_request(f"{output_request_id}_logits")
```

РИСК: СРЕДНИЙ

### 19. Config files содержат только параметры audio/mel, без tensor cache

ФАЙЛ: `src/auralis/models/xttsv2/config/xttsv2_config.py`

МЕТОД: `GPTAudioConfig`, `XTTSAudioConfig`, `XTTSGPTConfig`, `XTTSConfig`

ПРОБЛЕМА: найдены параметры mel/audio (`mel_channels`, `sample_rate`, `hop_length`, `n_fft`, `mel_norms_file`, `kv_cache`), но вычисления mel, forward-пути conditioning encoder, tensor cache, `.detach()`, `torch.no_grad()` / `torch.inference_mode()` отсутствуют. Декораторов `@lru_cache` / `@functools.cache` не найдено.

КОД:

```python
@dataclass
class XTTSAudioConfig:
    sample_rate: int = 22050
    output_sample_rate: int = 24000
    mel_channels: int = 80
    hop_length: int = 256
```

РИСК: СРЕДНИЙ

### 20. GPT config file содержит только параметры audio/mel, без tensor cache

ФАЙЛ: `src/auralis/models/xttsv2/config/xttsv2_gpt_config.py`

МЕТОД: `GPTAudioConfig`, `XTTSAudioConfig`, `XTTSGPTConfig`

ПРОБЛЕМА: найдены параметры mel/audio и `kv_cache`, но вычисления mel, forward-пути conditioning encoder, tensor cache, `.detach()`, `torch.no_grad()` / `torch.inference_mode()` отсутствуют. Декораторов `@lru_cache` / `@functools.cache` не найдено.

КОД:

```python
@dataclass
class GPTAudioConfig:
    mel_channels: int = 80
    sample_rate: int = 22050
    output_sample_rate: int = 24000
```

РИСК: СРЕДНИЙ

## Проверка cache-декораторов

В целевых файлах не найдено:
- `@lru_cache`
- `@functools.lru_cache`
- `@functools.cache`
- `@cache`

Следовательно, в проверенных файлах не найдено мест, где `lru_cache` / `functools.cache` кешируют tensors с `requires_grad=True`. Отдельно: `functools` импортируется в `XTTSv2.py` и `vllm_mm_gpt.py`, но используется для `functools.partial(...)`, а не для tensor cache.

## Итоговый список по риску

КРИТИЧЕСКИЙ:
1. `XTTSv2.py:_get_speaker_embedding` - speaker encoder forward без no_grad/inference_mode/detach.
2. `XTTSv2.py:get_gpt_cond_latents` - chunked `wav_to_mel_cloning` + `get_style_emb` без no_grad/inference_mode/detach.
3. `XTTSv2.py:get_gpt_cond_latents` - full-audio `wav_to_mel_cloning` + `get_style_emb` без no_grad/inference_mode/detach.
4. `XTTSv2.py:get_style_emb` - `conditioning_encoder` и `conditioning_perceiver` возвращают tensors без detach.
5. `XTTSv2.py:get_conditioning_latents` - собирает и возвращает conditioning outputs без detach.
6. `XTTSv2.py:get_audio_conditioning` - async-вход в conditioning path без локального inference/no_grad context.
7. `core/tts.py:prepare_for_streaming_generation` - напрямую вызывает `get_audio_conditioning` и замыкает tensors в `partial`.

ВЫСОКИЙ:
1. `XTTSv2.py:get_generation_context` - `@torch.inference_mode()` на async method, но без локального context внутри async body.
2. `XTTSv2.py:prepare_inputs_async` - протаскивает conditioning outputs в merge без detach.
3. `XTTSv2.py:_merge_conditioning` - `torch.cat` сохраняет связь с graph при grad-bearing audio conditioning.
4. `vllm_mm_gpt.py:input_mapper_for_xtts` - передает `embeds` как `cond_latents` без detach.
5. `vllm_mm_gpt.py:XttsGPT.forward` - передает `cond_latents` в GPT без detach.
6. `vllm_mm_gpt.py:GPT2Model._insert_conditioning_into_hidden_states` - вставляет conditioning tensors в hidden_states через `torch.cat` без detach.
7. `core/tts.py:save_stream/generate_speech` - streaming save path приводит к conditioning path без outer inference/no_grad.
8. `core/tts.py:save_stream_async/generate_speech_async` - async streaming save path приводит к conditioning path без outer inference/no_grad.

СРЕДНИЙ:
1. `vllm_mm_gpt.py:PositionalEmbeddingsCorrecter` - tensors с `grad_fn` не хранит, но dict не имеет полного глобального clear между запросами.
2. `vllm_mm_gpt.py:compute_logits/sample` - очистка PositionalEmbeddingsCorrecter частичная и request-scoped.
3. `XTTSv2.py:process_tokens_to_speech` - есть best-effort request cleanup PositionalEmbeddingsCorrecter, но async inference context не локальный.
4. `xttsv2_config.py` - только audio/mel параметры, tensor cache не найден.
5. `xttsv2_gpt_config.py` - только audio/mel параметры, tensor cache не найден.
