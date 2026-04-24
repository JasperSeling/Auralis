# Auralis generation parameters audit

Static audit date: 2026-04-24

Scope:

- `src/auralis/common/definitions/requests.py`
- `src/auralis/common/definitions/enhancer.py`
- `src/auralis/common/definitions/output.py`
- `src/auralis/models/xttsv2/XTTSv2.py`
- usage search across `src` and `tests`

No speech generation was run.

## Executive summary

The main production path for long audiobook generation is:

```python
request = TTSRequest(
    text=text,
    language="ru",
    speaker_files=["/content/reference.wav"],
)
stats = tts.save_stream(request, OUTPUT_PATH)
```

`save_stream()` temporarily forces `request.stream = True`, iterates generated chunks, and writes them directly to disk. This is the correct low-memory path for long texts.

Important findings:

- `AudioPreprocessingConfig` is defined in `src/auralis/common/definitions/enhancer.py`, then imported by `requests.py`.
- `TTSRequest.enhance_speech=True` is the top-level switch that enables preprocessing of `speaker_files`.
- `AudioPreprocessingConfig.enhance_speech=True` is one specific preprocessing step inside `EnhancedAudioProcessor.process()`.
- `sound_norm_refs` and `load_sample_rate` exist on `TTSRequest`, and XTTSv2 has matching lower-level parameters, but the request values are not passed into XTTSv2 in the normal generation path.
- `length_penalty` and `do_sample` are declared and copied, but not used by XTTSv2 generation.
- The generation parameters that currently matter most for Russian audiobook stability are `language`, reference audio quality, `max_ref_length`, `gpt_cond_len`, `gpt_cond_chunk_len`, `temperature`, `top_p`, `top_k`, and `repetition_penalty`.

## TTSRequest parameters

Definition: `src/auralis/common/definitions/requests.py`

| Parameter | Type | Default | Where used | Status |
|---|---|---:|---|---|
| `text` | `Union[AsyncGenerator[str, None], str, List[str]]` | required | language autodetect, tokenization, long-text splitting, progress label | Works |
| `speaker_files` | `Union[Union[str, List[str]], Union[bytes, List[bytes]]]` | required | reference audio conditioning, optional preprocessing | Works |
| `context_partial_function` | `Optional[Callable]` | `None` | streaming chat context reuse in `TTS._prepare_generation_context()` | Works, internal/specialized |
| `start_time` | `Optional[float]` | `None` | set before generation, copied into `TTSOutput` | Works, metadata |
| `enhance_speech` | `bool` | `False` | enables reference preprocessing in `TTSRequest.__post_init__()` | Works, but only when `speaker_files` is a list |
| `audio_config` | `AudioPreprocessingConfig` | `AudioPreprocessingConfig()` | controls `EnhancedAudioProcessor` | Works only when top-level `enhance_speech=True` triggers preprocessing |
| `language` | `SupportedLanguages` | `"auto"` | autodetect/validation/tokenization | Works; use `"ru"` for Russian audiobooks |
| `request_id` | `str` | `uuid.uuid4().hex` | scheduler and vLLM request ids | Works, internal |
| `load_sample_rate` | `int` | `22050` | copied, documented | Effectively unused in normal XTTSv2 path |
| `sound_norm_refs` | `bool` | `False` | copied, documented | Effectively unused in normal XTTSv2 path |
| `max_ref_length` | `int` | `60` | passed to XTTSv2 conditioning | Works |
| `gpt_cond_len` | `int` | `30` | passed to XTTSv2 conditioning | Works |
| `gpt_cond_chunk_len` | `int` | `4` | passed to XTTSv2 conditioning | Works |
| `stream` | `bool` | `False` | controls streaming vs combined output; `save_stream()` mutates temporarily | Works |
| `temperature` | `float` | `0.75` | passed to vLLM sampling params | Works |
| `top_p` | `float` | `0.85` | passed to vLLM sampling params | Works |
| `top_k` | `int` | `50` | passed to vLLM sampling params | Works |
| `repetition_penalty` | `float` | `5.0` | passed to custom `LogitsRepetitionPenalizer` | Works |
| `length_penalty` | `float` | `1.0` | declared, copied, exposed in OpenAI schemas | Unused by XTTSv2 |
| `do_sample` | `bool` | `True` | declared, copied, exposed in OpenAI schemas | Unused by XTTSv2 |

### Notes on request lifecycle

`TTSRequest.__post_init__()`:

- autodetects language if `language == "auto"` and `len(text) > 0`;
- validates language;
- creates `self.processor = EnhancedAudioProcessor(self.audio_config)`;
- if `speaker_files` is a list and `enhance_speech=True`, replaces every reference file with a preprocessed temporary file.

Important edge cases:

- If `speaker_files` is a single string, top-level `enhance_speech=True` does not preprocess it because the code checks `isinstance(self.speaker_files, list)`.
- If `text` is an async generator or list, `len(self.text)` may not be valid or semantically correct. For the audiobook path with plain `str`, it is fine.
- `preprocess_audio()` takes `audio_config` as an argument, but internally loads with `self.audio_config.sample_rate`, not the passed `audio_config.sample_rate`.

## AudioPreprocessingConfig parameters

Definition: `src/auralis/common/definitions/enhancer.py`

| Parameter | Type | Default | Used in | Effect | Status |
|---|---|---:|---|---|---|
| `sample_rate` | `int` | `22050` | librosa load during preprocessing, mel/VAD, LUFS meter, clarity shaping | Reference preprocessing sample rate | Works |
| `normalize` | `bool` | `True` | `EnhancedAudioProcessor.process()` | enables LUFS normalization | Works |
| `trim_silence` | `bool` | `True` | `EnhancedAudioProcessor.process()` | enables VAD masking | Works |
| `remove_noise` | `bool` | `True` | `EnhancedAudioProcessor.process()` | enables spectral gating | Works |
| `enhance_speech` | `bool` | `True` | `EnhancedAudioProcessor.process()` | enables spectral clarity boost | Works |
| `vad_threshold` | `float` | `0.02` | `vad_split()` | VAD mask threshold | Works |
| `vad_frame_length` | `int` | `1024 * 4` | `vad_split()` | VAD frame length | Works |
| `noise_reduce_margin` | `float` | `1.0` | `spectral_gating()` | spectral gate aggressiveness | Works |
| `noise_reduce_frames` | `int` | `25` | `spectral_gating()` | low-energy frames used for noise profile | Works |
| `enhance_amount` | `float` | `1.0` | `enhance_clarity()` | strength of clarity boost around 2 kHz | Works |
| `target_lufs` | `float` | `-18.0` | `normalize_loudness()` | target loudness | Works |

### How preprocessing actually works

`EnhancedAudioProcessor.process()` runs the following steps in order:

1. `trim_silence`: calls `vad_split()`.
2. `remove_noise`: calls `spectral_gating()`.
3. `enhance_speech`: calls `enhance_clarity()`.
4. `normalize`: calls `normalize_loudness()`.

This preprocessing applies to reference audio, not to the final generated waveform.

### normalize=True/False

`normalize=True` computes integrated loudness with `pyloudnorm.Meter(sample_rate)`, applies gain toward `target_lufs`, then returns `np.tanh(audio_normalized)`.

Expected effect:

- more consistent reference loudness;
- may help if the reference is too quiet or too loud;
- can slightly alter timbre because `tanh` is soft clipping.

### trim_silence=True/False

`trim_silence=True` calls `vad_split()`. Despite the name, it does not remove samples like a classic trim. It computes a VAD-like mask from energy and mel spectral features, interpolates the mask to audio length, and multiplies the waveform by the mask.

Expected effect:

- suppresses low-activity/silent regions;
- can help if the reference has noisy silence;
- can damage natural pauses or quiet speech if thresholding is too aggressive.

### enhance_speech=True/False and enhance_amount

`audio_config.enhance_speech=True` calls `enhance_clarity()`. It performs spectral shaping with a boost centered around roughly 2 kHz. `enhance_amount` scales this boost.

Expected effect:

- can improve perceived clarity of a muffled reference;
- can make a clean studio reference harsher or less natural if too strong.

Suggested range for Russian audiobook references:

- clean studio reference: disable preprocessing or use `enhance_amount=0.3`;
- mildly noisy/muffled reference: `enhance_amount=0.4..0.7`;
- avoid very high values unless testing confirms better voice similarity.

## TTSOutput parameters and utilities

Definition: `src/auralis/common/definitions/output.py`

| Parameter | Type | Default | Effect |
|---|---|---:|---|
| `array` | `Union[np.ndarray, bytes]` | required | audio samples |
| `sample_rate` | `int` | `24000` | output audio sample rate metadata |
| `bit_depth` | `int` | `32` | used by `save()` |
| `bit_rate` | `int` | `192` | kbps for compressed formats in `to_bytes()` |
| `compression` | `int` | `10` | compression level, capped for FLAC |
| `channel` | `int` | `1` | passed as `channels_first` in `save()` |
| `start_time` | `Optional[float]` | `None` | metadata |
| `end_time` | `Optional[float]` | `None` | metadata, not broadly used |
| `token_length` | `Optional[int]` | `None` | number of generated audio tokens for the chunk |

Useful methods:

- `combine_outputs(outputs)`: concatenates arrays and uses the first output sample rate.
- `to_bytes(format="wav", sample_width=2)`: serializes to wav/flac/mp3/opus/aac/pcm.
- `save(filename, sample_rate=None, format=None)`: saves full output.
- `resample(new_sample_rate)`: returns resampled `TTSOutput`.
- `change_speed(speed_factor)`: phase-vocoder speed change.
- `get_info()`: returns sample count, sample rate, duration.

`save_stream()` does not use `TTSOutput.save()`. It writes each chunk directly through `soundfile.SoundFile`.

## XTTSv2 parameter flow

### Reference conditioning

XTTSv2 has lower-level parameters in `get_conditioning_latents()` and `get_audio_conditioning()`:

- `max_ref_length`
- `gpt_cond_len`
- `gpt_cond_chunk_len`
- `librosa_trim_db`
- `sound_norm_refs`
- `load_sr`

In normal `TTSRequest` generation, only these are passed from request:

- `max_ref_length`
- `gpt_cond_len`
- `gpt_cond_chunk_len`

The request values below are not passed:

- `sound_norm_refs`
- `load_sample_rate`

This means:

```python
TTSRequest(sound_norm_refs=True, load_sample_rate=44100)
```

does not currently change XTTSv2 reference loading/normalization in the standard `save_stream()` path.

### Generation sampling

`XTTSv2Engine.get_generation_context()` passes these request values into `ExtendedSamplingParams`:

- `temperature=request.temperature`
- `top_p=request.top_p`
- `top_k=request.top_k`
- `logits_processors=[LogitsRepetitionPenalizer(request.repetition_penalty)]`

It also sets:

- `repetition_penalty=1.0`, because repetition penalty is handled manually;
- `max_tokens=self.gpt_config.gpt_max_audio_tokens`;
- `ignore_eos=True`;
- `stop_token_ids=[self.mel_eos_token_id]`.

`length_penalty` and `do_sample` are not used here.

## Special parameter checks

### `TTSRequest.enhance_speech`

Works, but only as a top-level switch for preprocessing reference audio.

Example:

```python
request = TTSRequest(
    text="Проверка голоса.",
    language="ru",
    speaker_files=["/content/reference.wav"],
    enhance_speech=True,
)
```

Expected effect:

- `request.speaker_files` becomes a list of temporary processed audio paths;
- final generation uses those processed reference files.

### `AudioPreprocessingConfig.normalize`

Works if top-level `enhance_speech=True`.

Example:

```python
request = TTSRequest(
    text="Проверка.",
    language="ru",
    speaker_files=["/content/reference.wav"],
    enhance_speech=True,
    audio_config=AudioPreprocessingConfig(normalize=True),
)
```

Expected effect:

- reference loudness is normalized toward `target_lufs`;
- useful for quiet/loud reference recordings.

### `AudioPreprocessingConfig.trim_silence`

Works if top-level `enhance_speech=True`.

Example:

```python
request = TTSRequest(
    text="Проверка.",
    language="ru",
    speaker_files=["/content/reference.wav"],
    enhance_speech=True,
    audio_config=AudioPreprocessingConfig(trim_silence=True),
)
```

Expected effect:

- quiet/no-speech regions in reference are suppressed;
- not a true sample-removing trim.

### `AudioPreprocessingConfig.enhance_speech` and `enhance_amount`

Works if top-level `enhance_speech=True`.

Example:

```python
request = TTSRequest(
    text="Проверка.",
    language="ru",
    speaker_files=["/content/reference.wav"],
    enhance_speech=True,
    audio_config=AudioPreprocessingConfig(
        enhance_speech=True,
        enhance_amount=0.5,
    ),
)
```

Expected effect:

- reference clarity is boosted around speech-relevant frequencies;
- good for dull references, risky for already clean references.

### `sound_norm_refs`

XTTSv2 implements this internally:

```python
if sound_norm_refs:
    audio = (audio / torch.abs(audio).max()) * 0.75
```

But the request field is not passed into `get_audio_conditioning()` from `get_generation_context()`.

Example that currently does not affect normal generation:

```python
request = TTSRequest(
    text=text,
    language="ru",
    speaker_files=["/content/reference.wav"],
    sound_norm_refs=True,
)
```

Status: effectively unused from `TTSRequest`.

### `load_sample_rate`

XTTSv2 supports lower-level `load_sr`, default `22050`, but `request.load_sample_rate` is not passed.

Example that currently does not affect normal generation:

```python
request = TTSRequest(
    text=text,
    language="ru",
    speaker_files=["/content/reference.wav"],
    load_sample_rate=44100,
)
```

Status: effectively unused from `TTSRequest`.

### `max_ref_length`

Works. XTTSv2 loads each reference and truncates it:

```python
audio = audio[:, : load_sr * max_ref_length]
```

Example:

```python
request = TTSRequest(
    text=text,
    language="ru",
    speaker_files=["/content/reference.wav"],
    max_ref_length=60,
)
```

Expected effect:

- controls how many seconds of each reference file can contribute;
- longer can improve voice/style capture if the reference is clean;
- too long can include unwanted noise, room tone, mistakes, or style drift.

### `gpt_cond_len`

Works. Passed into `get_gpt_cond_latents(..., length=gpt_cond_len)`.

Example:

```python
request = TTSRequest(
    text=text,
    language="ru",
    speaker_files=["/content/reference.wav"],
    gpt_cond_len=30,
)
```

Expected effect:

- longer conditioning can improve style stability;
- may increase conditioning cost.

### `gpt_cond_chunk_len`

Works. Passed into `get_gpt_cond_latents(..., chunk_length=gpt_cond_chunk_len)`.

Example:

```python
request = TTSRequest(
    text=text,
    language="ru",
    speaker_files=["/content/reference.wav"],
    gpt_cond_chunk_len=4,
)
```

Expected effect:

- controls chunk size for conditioning latent extraction;
- smaller chunks can average across reference variation;
- must be interpreted together with `gpt_cond_len`.

### `temperature`

Works. Passed to vLLM sampling.

Example:

```python
request = TTSRequest(
    text=text,
    language="ru",
    speaker_files=["/content/reference.wav"],
    temperature=0.55,
)
```

Expected effect:

- lower values improve stability and reduce random prosody changes;
- too low may make speech flatter or increase repetitive patterns.

### `top_p`

Works. Passed to vLLM sampling.

Example:

```python
request = TTSRequest(
    text=text,
    language="ru",
    speaker_files=["/content/reference.wav"],
    top_p=0.75,
)
```

Expected effect:

- lower values reduce sampling diversity;
- useful for long-form consistency.

### `top_k`

Works. Passed to vLLM sampling.

Example:

```python
request = TTSRequest(
    text=text,
    language="ru",
    speaker_files=["/content/reference.wav"],
    top_k=35,
)
```

Expected effect:

- restricts token candidates;
- lower values can improve stability but may reduce expressiveness.

### `repetition_penalty`

Works through custom `LogitsRepetitionPenalizer`.

Example:

```python
request = TTSRequest(
    text=text,
    language="ru",
    speaker_files=["/content/reference.wav"],
    repetition_penalty=5.5,
)
```

Expected effect:

- reduces repeated audio-token patterns;
- too high may cause unnatural pronunciation or abrupt prosody.

### `length_penalty`

Declared but unused by XTTSv2 generation.

Example with no current effect:

```python
request = TTSRequest(
    text=text,
    language="ru",
    speaker_files=["/content/reference.wav"],
    length_penalty=0.8,
)
```

Status: placeholder/dead parameter for XTTSv2.

### `do_sample`

Declared but unused by XTTSv2 generation.

Example with no current effect:

```python
request = TTSRequest(
    text=text,
    language="ru",
    speaker_files=["/content/reference.wav"],
    do_sample=False,
)
```

Status: placeholder/dead parameter for XTTSv2.

## Mini-tests for working parameters

These are configuration smoke tests and expected-effect tests. They should not run full audiobook generation.

### Reference preprocessing smoke test

```python
from auralis import TTSRequest, AudioPreprocessingConfig

request = TTSRequest(
    text="Короткая проверка.",
    language="ru",
    speaker_files=["/content/reference.wav"],
    enhance_speech=True,
    audio_config=AudioPreprocessingConfig(
        normalize=True,
        trim_silence=True,
        remove_noise=True,
        enhance_speech=True,
        enhance_amount=0.5,
        target_lufs=-18.0,
    ),
)

assert isinstance(request.speaker_files, list)
assert request.speaker_files[0] != "/content/reference.wav"
```

Expected effect:

- reference file is preprocessed immediately during request construction;
- generated audio may inherit a more stable/clean voice reference.

### Disable preprocessing for clean studio reference

```python
request = TTSRequest(
    text="Короткая проверка.",
    language="ru",
    speaker_files=["/content/reference.wav"],
    enhance_speech=False,
)

assert request.speaker_files == ["/content/reference.wav"]
```

Expected effect:

- no timbre alteration from preprocessing;
- best first attempt for high-quality studio reference audio.

### Conditioning parameters

```python
request = TTSRequest(
    text=text,
    language="ru",
    speaker_files=["/content/reference.wav"],
    max_ref_length=60,
    gpt_cond_len=30,
    gpt_cond_chunk_len=4,
)
```

Expected effect:

- longer and richer reference conditioning;
- likely relevant to voice consistency in long Russian audiobook generation.

### Stable generation sampling

```python
request = TTSRequest(
    text=text,
    language="ru",
    speaker_files=["/content/reference.wav"],
    temperature=0.55,
    top_p=0.75,
    top_k=35,
    repetition_penalty=5.5,
)
```

Expected effect:

- less randomness;
- fewer long-form prosody jumps;
- potentially less expressive than defaults.

### Parameters that currently do not affect standard XTTSv2 generation

```python
request = TTSRequest(
    text=text,
    language="ru",
    speaker_files=["/content/reference.wav"],
    sound_norm_refs=True,
    load_sample_rate=44100,
    length_penalty=0.8,
    do_sample=False,
)
```

Expected effect:

- no effect in current normal `save_stream()` path.

## Recommended TTSRequest for long Russian audiobooks

For maximum stability, explicit Russian language, O(1) memory writing through `save_stream()`, and moderately conservative sampling:

```python
from auralis import TTSRequest, AudioPreprocessingConfig

request = TTSRequest(
    text=text,
    language="ru",
    speaker_files=["/content/reference.wav"],

    # Enable only if reference.wav is noisy, quiet, or has problematic silence.
    enhance_speech=True,
    audio_config=AudioPreprocessingConfig(
        sample_rate=22050,
        normalize=True,
        trim_silence=True,
        remove_noise=True,
        enhance_speech=True,
        enhance_amount=0.5,
        target_lufs=-18.0,
    ),

    # Voice conditioning
    max_ref_length=60,
    gpt_cond_len=30,
    gpt_cond_chunk_len=4,

    # Stable audiobook generation
    temperature=0.55,
    top_p=0.75,
    top_k=35,
    repetition_penalty=5.5,

    # save_stream() temporarily forces stream=True internally.
    stream=False,
)

stats = tts.save_stream(request, OUTPUT_PATH)
```

For a clean studio-quality reference, first try:

```python
request = TTSRequest(
    text=text,
    language="ru",
    speaker_files=["/content/reference.wav"],
    enhance_speech=False,
    max_ref_length=60,
    gpt_cond_len=30,
    gpt_cond_chunk_len=4,
    temperature=0.55,
    top_p=0.75,
    top_k=35,
    repetition_penalty=5.5,
)

stats = tts.save_stream(request, OUTPUT_PATH)
```

## Suggested code follow-up

If these request fields are intended to work, `XTTSv2Engine.get_generation_context()` should pass them into `prepare_inputs_async()` / `get_audio_conditioning()`:

- `request.sound_norm_refs`
- `request.load_sample_rate`

Also either wire or remove/mark unsupported:

- `length_penalty`
- `do_sample`

Current behavior is not dangerous, but it is surprising: users can set these fields and receive no effect in the standard XTTSv2 generation path.
