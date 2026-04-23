import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import AsyncGenerator, Callable, Optional, Dict, Union, Generator, List

from huggingface_hub import hf_hub_download
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from auralis.common.logging.logger import setup_logger, set_vllm_logging_level
from auralis.common.definitions.output import TTSOutput
from auralis.common.definitions.requests import TTSRequest
from auralis.common.metrics.performance import metrics, track_generation
from auralis.common.scheduling.two_phase_scheduler import TwoPhaseScheduler
from auralis.models.base import BaseAsyncTTSEngine, AudioOutputGenerator

# Shared rich console for progress display and summaries.
# Module-level so it reuses the same Terminal handle across calls.
_console = Console()

class TTS:
    """A high-performance text-to-speech engine optimized for inference speed.

    This class provides an interface for both synchronous and asynchronous speech generation,
    with support for streaming output and parallel processing of multiple requests.
    """

    def __init__(self, scheduler_max_concurrency: int = 10, vllm_logging_level=logging.DEBUG):
        """Initialize the TTS engine.

        Args:
            scheduler_max_concurrency (int): Maximum number of concurrent requests to process.
            vllm_logging_level: Logging level for the VLLM backend.
        """
        set_vllm_logging_level(vllm_logging_level)

        self.scheduler: Optional[TwoPhaseScheduler] = TwoPhaseScheduler(scheduler_max_concurrency)
        self.tts_engine: Optional[BaseAsyncTTSEngine] = None
        self.concurrency = scheduler_max_concurrency
        self.max_vllm_memory: Optional[int] = None
        self.logger = setup_logger(__file__)
        self.loop = None

    def _ensure_event_loop(self):
        """Ensures that an event loop exists and is set."""

        if not self.loop:
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)

    def from_pretrained(self, model_name_or_path: str, **kwargs):
        """Load a pretrained model from local path or Hugging Face Hub.
           **THIS METHOD IS SYNCHRONOUS**

        Args:
            model_name_or_path (str): Local path or Hugging Face model identifier.
            **kwargs: Additional arguments passed to the model's from_pretrained method.

        Returns:
            TTS: The TTS instance with loaded model.

        Raises:
            ValueError: If the model cannot be loaded from the specified path.
        """
        from auralis.models.registry import MODEL_REGISTRY

        # Ensure an event loop exists for potential async operations within from_pretrained
        self._ensure_event_loop()

        try:
            with open(os.path.join(model_name_or_path, 'config.json'), 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            try:
                config_path = hf_hub_download(repo_id=model_name_or_path, filename='config.json')
                with open(config_path, 'r') as f:
                    config = json.load(f)
            except Exception as e:
                raise ValueError(f"Could not load model from {model_name_or_path} neither locally or online: {e}")

        # Run potential async operations within from_pretrained in the event loop
        async def _load_model():
            return MODEL_REGISTRY[config['model_type']].from_pretrained(model_name_or_path, **kwargs)

        self.tts_engine = self.loop.run_until_complete(_load_model()) # to start form the correct loop

        return self

    async def prepare_for_streaming_generation(self, request: TTSRequest):
        """Prepare conditioning for streaming generation.

        Args:
            request (TTSRequest): The TTS request containing speaker files.

        Returns:
            Partial function with prepared conditioning for generation.
        """
        conditioning_config = self.tts_engine.conditioning_config
        if conditioning_config.speaker_embeddings or conditioning_config.gpt_like_decoder_conditioning:
            gpt_cond_latent, speaker_embeddings = await self.tts_engine.get_audio_conditioning(request.speaker_files)
            return partial(self.tts_engine.get_generation_context,
                           gpt_cond_latent=gpt_cond_latent,
                           speaker_embeddings=speaker_embeddings)

    async def _prepare_generation_context(self, input_request: TTSRequest):
        """Prepare the generation context for the first phase of speech synthesis.

        Args:
            input_request (TTSRequest): The TTS request to process.

        Returns:
            dict: Dictionary containing parallel inputs and the original request.
        """
        conditioning_config = self.tts_engine.conditioning_config
        input_request.start_time = time.time()
        if input_request.context_partial_function:
            (audio_token_generators, requests_ids,
             speaker_embeddings,
             gpt_like_decoder_conditioning) = \
                await input_request.context_partial_function(input_request)
        else:
            audio_token_generators, speaker_embeddings, gpt_like_decoder_conditioning = None, None, None

            if conditioning_config.speaker_embeddings and conditioning_config.gpt_like_decoder_conditioning:
                (audio_token_generators, requests_ids,
                 speaker_embeddings,
                 gpt_like_decoder_conditioning) = await self.tts_engine.get_generation_context(input_request)
            elif conditioning_config.speaker_embeddings:
                (audio_token_generators, requests_ids,
                 speaker_embeddings) = await self.tts_engine.get_generation_context(input_request)
            elif conditioning_config.gpt_like_decoder_conditioning:
                (audio_token_generators, requests_ids,
                 gpt_like_decoder_conditioning) = await self.tts_engine.get_generation_context(input_request)
            else:
                audio_token_generators, requests_ids = await self.tts_engine.get_generation_context(input_request)

        parallel_inputs = [
            {
                'generator': gen,
                'speaker_embedding': speaker_embeddings[i] if
                speaker_embeddings is not None and isinstance(speaker_embeddings, list) else
                speaker_embeddings if speaker_embeddings is not None else
                None,
                'multimodal_data': gpt_like_decoder_conditioning[i] if
                gpt_like_decoder_conditioning is not None and isinstance(gpt_like_decoder_conditioning, list) else
                gpt_like_decoder_conditioning if gpt_like_decoder_conditioning is not None else
                None,
                'request': input_request,
            }
            for i, gen in enumerate(audio_token_generators)
        ]

        return {
            'parallel_inputs': parallel_inputs,
            'request': input_request
        }

    async def _process_single_generator(self, gen_input: Dict) -> AudioOutputGenerator:
        """Process a single generator to produce speech output.

        Args:
            gen_input (Dict): Dictionary containing generator and conditioning information.

        Returns:
            AudioOutputGenerator: Generator yielding audio chunks.

        Raises:
            Exception: If any error occurs during processing.
        """
        try:
            async for chunk in self.tts_engine.process_tokens_to_speech(  # type: ignore
                    generator=gen_input['generator'],
                    speaker_embeddings=gen_input['speaker_embedding'],
                    multimodal_data=gen_input['multimodal_data'],
                    request=gen_input['request'],
            ):
                yield chunk
        except Exception as e:
            raise e

    @track_generation
    async def _second_phase_fn(self, gen_input: Dict) -> AudioOutputGenerator:
        """Second phase of speech generation: Convert tokens to speech.

        Args:
            gen_input (Dict): Dictionary containing generator and conditioning information.

        Returns:
            AudioOutputGenerator: Generator yielding audio chunks.
        """
        async for chunk in self._process_single_generator(gen_input):
            yield chunk

    async def generate_speech_async(self, request: TTSRequest) -> Union[AsyncGenerator[TTSOutput, None], TTSOutput]:
        """Generate speech asynchronously from text.

        Args:
            request (TTSRequest): The TTS request to process.

        Returns:
            Union[AsyncGenerator[TTSOutput, None], TTSOutput]: Audio output, either streamed or complete.

        Raises:
            RuntimeError: If instance was not created for async generation.
        """
        self._ensure_event_loop()

        async def process_chunks():
            chunks = []
            try:
                async for chunk in self.scheduler.run(
                        inputs=request,
                        request_id=request.request_id,
                        first_phase_fn=self._prepare_generation_context,
                        second_phase_fn=self._second_phase_fn
                ):
                    if request.stream:
                        yield chunk
                    chunks.append(chunk)
            except Exception as e:
                self.logger.error(f"Error during speech generation: {e}")
                raise

            if not request.stream:
                yield TTSOutput.combine_outputs(chunks)

        if request.stream:
            return process_chunks()
        else:
            async for result in process_chunks():
                return result

    @staticmethod
    def split_requests(request: TTSRequest, max_length: int = 100000) -> List[TTSRequest]:
        """Split a long text request into multiple smaller requests.

        Args:
            request (TTSRequest): The original TTS request.
            max_length (int): Maximum length of text per request.

        Returns:
            List[TTSRequest]: List of split requests.
        """
        if len(request.text) <= max_length:
            return [request]

        text_chunks = [request.text[i:i + max_length]
                       for i in range(0, len(request.text), max_length)]

        return [
            (copy := request.copy(), setattr(copy, 'text', chunk), setattr(copy, 'request_id', uuid.uuid4().hex))[0]
            for chunk in text_chunks
        ]

    async def _process_multiple_requests(
        self,
        requests: List[TTSRequest],
        results: Optional[List] = None,
        on_chunk: Optional[Callable[[TTSOutput], None]] = None,
    ) -> Optional[TTSOutput]:
        """Process multiple TTS requests in parallel.

        Args:
            requests (List[TTSRequest]): List of requests to process.
            results (Optional[List]): Optional list to store results for streaming.
            on_chunk (Optional[Callable]): Optional callback invoked after each
                chunk is produced. Used to drive progress display. Does not
                affect generation semantics.

        Returns:
            Optional[TTSOutput]: Combined audio output if not streaming, None otherwise.
        """
        output_queues = [asyncio.Queue() for _ in requests] if results is not None else None

        async def process_subrequest(idx, sub_request, queue: Optional[asyncio.Queue] = None):
            chunks = []
            async for chunk in self.scheduler.run(
                    inputs=sub_request,
                    request_id=sub_request.request_id,
                    first_phase_fn=self._prepare_generation_context,
                    second_phase_fn=self._second_phase_fn
            ):
                chunks.append(chunk)
                if queue is not None:
                    await queue.put(chunk)
                if on_chunk is not None:
                    on_chunk(chunk)

            if queue is not None:
                await queue.put(None)
            return chunks

        tasks = [
            asyncio.create_task(
                process_subrequest(
                    idx,
                    sub_request,
                    output_queues[idx] if output_queues else None
                )
            )
            for idx, sub_request in enumerate(requests)
        ]

        if results is not None:
            for idx, queue in enumerate(output_queues):
                while True:
                    chunk = await queue.get()
                    if chunk is None:
                        break
                    results[idx].append(chunk)
            return None
        else:
            all_chunks = await asyncio.gather(*tasks)
            complete_audio = [chunk for chunks in all_chunks for chunk in chunks]
            return TTSOutput.combine_outputs(complete_audio)

    @staticmethod
    @contextmanager
    def _progress_context(
        total_subrequests: Optional[int],
        description: str,
        print_summary: bool = True,
        enabled: bool = True,
    ):
        """Yield a rich Progress advance-callback; optionally print summary on exit.

        The callback signature is ``advance(chunk: TTSOutput) -> None``. Each
        call advances the bar by one and updates the live ``tok/s`` field from
        the global metrics tracker.

        When ``print_summary`` is True (default, used by ``generate_speech``),
        a one-line summary is printed on exit. When False (used by
        ``save_stream`` which prints its own file-oriented summary), the caller
        is responsible for any post-generation output.

        When ``enabled`` is False, the helper yields a no-op ``advance`` and
        does not create a ``Progress`` / ``Live`` at all. This is required when
        a caller already owns an active Live on ``_console`` (for example
        ``save_stream`` invokes ``generate_speech`` internally) — rich forbids
        two simultaneous Live displays on the same console and would raise
        ``rich.errors.LiveError``.

        This helper does not affect generation semantics — it only wires
        display callbacks.

        When ``total_subrequests`` is None (recommended), rich renders an
        **indeterminate** pulse bar and the MofN column shows only the
        completed count. Callers should prefer None over a crude estimate:
        previously we passed ``len(split_requests(request))`` which counts
        100k-char chunks of input text, not audio chunks (XTTSv2 yields one
        audio chunk per sentence — typically hundreds per sub-request). This
        produced misleading ``604/2`` displays.

        Args:
            total_subrequests (Optional[int]): Total chunk count if known,
                otherwise None for an indeterminate bar.
            description (str): Short label shown next to the spinner.
            print_summary (bool): Whether to print the default summary line
                after the Progress bar closes. Defaults to True.
            enabled (bool): When False, skip the Progress display entirely
                and yield a no-op advance. Defaults to True.

        Yields:
            Callable[[TTSOutput], None]: The ``advance`` callback.
        """
        if not enabled:
            # Early-return path: no Progress, no Live, no summary. Used when
            # an outer caller already owns the rich.Live on _console.
            yield lambda _chunk: None
            return

        start_time = time.time()
        chunks_seen: List[TTSOutput] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]🔊 {task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("чанков | [yellow]{task.fields[rate]} tok/s"),
            TimeElapsedColumn(),
            transient=True,
            console=_console,
        ) as progress:
            task = progress.add_task(description, total=total_subrequests, rate="—")

            def advance(chunk: TTSOutput) -> None:
                chunks_seen.append(chunk)
                tps = metrics.tokens_per_second
                rate = f"{tps:>4.0f}" if tps > 0 else "  —"
                progress.update(task, advance=1, rate=rate)

            yield advance

        # Compute and print summary after the Progress context has torn down.
        if print_summary and chunks_seen:
            elapsed = time.time() - start_time
            sr = chunks_seen[0].sample_rate or 0
            total_samples = sum(c.array.shape[0] for c in chunks_seen)
            audio_sec = total_samples / sr if sr else 0.0
            rtf = elapsed / audio_sec if audio_sec > 0 else 0.0
            _console.print(
                f"[bold green]✅ Готово за {elapsed:.0f} сек[/] | "
                f"[cyan]{audio_sec / 60:.1f} мин аудио[/] | "
                f"[magenta]RTF {rtf:.2f}x[/]"
            )

    @staticmethod
    def _make_progress_description(request: TTSRequest) -> str:
        """Build a short human-readable description from the request text.

        Takes up to 40 chars of the text (whitespace-normalised) and appends
        an ellipsis if truncated. Falls back to the request id when text is
        a list or generator rather than a plain string.
        """
        text = request.text
        if not isinstance(text, str):
            return f"req {request.request_id[:8]}"
        flat = " ".join(text.split())
        return flat[:40] + ("…" if len(flat) > 40 else "")

    def generate_speech(
        self,
        request: TTSRequest,
        _show_progress: bool = True,
    ) -> Union[Generator[TTSOutput, None, None], TTSOutput]:
        """Generate speech synchronously from text.

        Args:
            request (TTSRequest): The TTS request to process.
            _show_progress (bool): Internal flag. When False, suppresses the
                rich.Progress display managed by ``_progress_context``. Used by
                ``save_stream`` so its outer Progress bar is the only active
                Live on the console — rich forbids nested Live displays.
                Leading underscore signals "not part of the stable public API".

        Returns:
            Union[Generator[TTSOutput, None, None], TTSOutput]: Audio output, either streamed or complete.

        Raises:
            RuntimeError: If instance was created for async generation.
        """
        self._ensure_event_loop()
        requests = self.split_requests(request)
        description = self._make_progress_description(request)

        if request.stream:
            # Streaming case
            def streaming_wrapper():
                with self._progress_context(
                    None, description, enabled=_show_progress
                ) as advance:
                    for sub_request in requests:
                        # For streaming, execute the async gen
                        async def process_stream():
                            try:
                                async for chunk in self.scheduler.run(
                                        inputs=sub_request,
                                        request_id=sub_request.request_id,
                                        first_phase_fn=self._prepare_generation_context,
                                        second_phase_fn=self._second_phase_fn
                                ):
                                    yield chunk
                            except Exception as e:
                                self.logger.error(f"Error during streaming: {e}")
                                raise

                        # Execute the async gen
                        generator = process_stream()
                        try:
                            while True:
                                chunk = self.loop.run_until_complete(anext(generator))
                                advance(chunk)
                                yield chunk
                        except StopAsyncIteration:
                            pass

            return streaming_wrapper()
        else:
            # Non streaming
            with self._progress_context(
                None, description, enabled=_show_progress
            ) as advance:
                result = self.loop.run_until_complete(
                    self._process_multiple_requests(requests, on_chunk=advance)
                )
            return result

    class _StreamingFileWriter:
        """Chunk-at-a-time audio writer backed by ``soundfile.SoundFile``.

        Opens the output file lazily on the first chunk so the sample rate is
        taken from the actual model output (not hardcoded). Tracks total
        samples and wall-time to compute RTF/duration in ``stats()``.

        This is intentionally a small helper class used by
        ``save_stream`` / ``save_stream_async`` so the sync and async variants
        don't duplicate the write-and-account logic.
        """

        def __init__(self, filename: str, fmt: str):
            import soundfile as sf  # lazy: only cost for users who stream
            self._sf_module = sf
            self.filename = filename
            self.fmt = fmt
            self.sf_file = None
            self.sr: Optional[int] = None
            self.total_samples: int = 0
            self.start_wall: float = time.time()

        def write(self, chunk: TTSOutput) -> None:
            if self.sf_file is None:
                self.sr = int(chunk.sample_rate)
                channels = 1 if chunk.array.ndim == 1 else chunk.array.shape[0]
                self.sf_file = self._sf_module.SoundFile(
                    self.filename,
                    mode='w',
                    samplerate=self.sr,
                    channels=channels,
                    format=self.fmt,
                )
            arr = chunk.array
            # soundfile accepts float32/float64/int16/int32. TTSOutput.array is
            # typically float32 already (XTTSv2.py:805), but guard for safety.
            if hasattr(arr, 'dtype') and arr.dtype.kind == 'f' and arr.dtype.itemsize != 4:
                import numpy as np  # lazy; already at module scope transitively
                arr = arr.astype(np.float32)
            self.sf_file.write(arr)
            self.total_samples += arr.shape[0]

        def close(self) -> None:
            if self.sf_file is not None:
                self.sf_file.close()
                self.sf_file = None

        def stats(self) -> dict:
            wall = time.time() - self.start_wall
            duration = self.total_samples / self.sr if self.sr else 0.0
            rtf = wall / duration if duration > 0 else 0.0
            return {
                'path': self.filename,
                'sample_rate': self.sr,
                'n_samples': self.total_samples,
                'duration_sec': duration,
                'wall_sec': wall,
                'rtf': rtf,
            }

    @staticmethod
    def _resolve_format(filename: str, fmt: Optional[str]) -> str:
        """Pick a soundfile format code from an explicit override or file extension.

        Defaults to ``'WAV'`` when no extension is present. Common mappings
        (``.wav``, ``.flac``, ``.ogg``) are passed through uppercased, which
        matches ``soundfile``'s ``format=`` API.
        """
        from pathlib import Path as _Path
        if fmt:
            return fmt.upper()
        suffix = _Path(filename).suffix.lstrip('.').upper()
        return suffix or 'WAV'

    def _print_stream_summary(self, stats: dict) -> None:
        """Print the post-stream summary line via the shared rich console."""
        duration_min = stats['duration_sec'] / 60.0
        _console.print(
            f"[bold green]✅ {stats['path']}[/] | "
            f"[cyan]{duration_min:.1f} мин[/] | "
            f"[magenta]RTF {stats['rtf']:.2f}x[/] | "
            f"{stats['wall_sec']:.0f} сек"
        )

    def save_stream(
        self,
        request: TTSRequest,
        filename: Union[str, Path],
        format: Optional[str] = None,
        progress: bool = True,
    ) -> dict:
        """Generate speech and stream-write the audio to a file chunk-by-chunk.

        This is the **O(1) RAM** path: chunks are yielded by the existing
        ``generate_speech(stream=True)`` pipeline and written to disk as they
        arrive. The full audio is **never** held in memory — safe for
        arbitrarily long generations (audiobooks, podcasts).

        The file is opened lazily on the first chunk, with ``sample_rate``
        taken from the actual model output. The output format is inferred
        from the ``filename`` extension (``.wav``, ``.flac``, ``.ogg``) or
        explicitly overridden via ``format=``.

        !!! tip "Long generations"
            For generations longer than ~10 minutes, prefer ``.flac`` —
            it is streaming-safe and recovers gracefully from an interrupted
            Colab runtime. WAV requires a header rewrite on close; if the
            process is killed mid-generation the header size will be wrong
            (though most players will still play the file up to its
            actual end).

        Args:
            request (TTSRequest): The TTS request. Its ``stream`` flag is
                temporarily forced to True for the duration of this call
                and restored in a ``finally`` block regardless of outcome.
            filename (str | Path): Target output path. Format inferred from
                extension when ``format`` is not provided.
            format (Optional[str]): Optional explicit ``soundfile`` format
                code (``'WAV'``, ``'FLAC'``, ``'OGG'``, ...). Overrides
                the extension-based guess.
            progress (bool): Display a rich.Progress bar during generation.
                Defaults to True. Set False for non-TTY environments.

        Returns:
            dict: Statistics with keys ``'path'``, ``'sample_rate'``,
                ``'n_samples'``, ``'duration_sec'``, ``'wall_sec'``, ``'rtf'``.

        Raises:
            RuntimeError: If the generator yielded zero chunks.
        """
        filename = str(filename)
        fmt = self._resolve_format(filename, format)
        description = self._make_progress_description(request)

        # KAMEN 1 — мутируем stream локально, restore в finally.
        # Не используем request.copy() потому что Pydantic copy может быть
        # shallow и ссылочные поля (speaker_files) будут shared.
        original_stream = request.stream
        request.stream = True

        writer = self._StreamingFileWriter(filename, fmt)

        try:
            if progress:
                # total=None — indeterminate bar: actual chunk count is only
                # known after generation (one chunk per sentence, typically
                # hundreds). Using len(split_requests) would display misleading
                # "604/2" counters.
                with self._progress_context(
                    None, description, print_summary=False
                ) as advance:
                    # _show_progress=False prevents generate_speech from
                    # opening a nested rich.Live on the same _console — rich
                    # disallows nested Live displays (LiveError otherwise).
                    for chunk in self.generate_speech(request, _show_progress=False):
                        writer.write(chunk)
                        advance(chunk)
            else:
                for chunk in self.generate_speech(request, _show_progress=False):
                    writer.write(chunk)
        finally:
            request.stream = original_stream
            writer.close()

        if writer.total_samples == 0:
            raise RuntimeError('save_stream: generate_speech yielded zero chunks')

        stats = writer.stats()
        self._print_stream_summary(stats)
        return stats

    async def save_stream_async(
        self,
        request: TTSRequest,
        filename: Union[str, Path],
        format: Optional[str] = None,
        progress: bool = True,
    ) -> dict:
        """Async variant of ``save_stream`` for callers inside an event loop.

        Use this from FastAPI handlers, jupyter-async cells, or any ``async
        def``. Internally iterates ``generate_speech_async(stream=True)`` and
        writes each chunk to the target file without buffering the full audio
        in RAM.

        See ``save_stream`` for parameter and return semantics. The return
        value and printed summary are identical; only the call-site concurrency
        model differs.

        Args:
            request (TTSRequest): The TTS request. ``stream`` is forced True
                locally and restored in ``finally``.
            filename (str | Path): Output path.
            format (Optional[str]): Explicit ``soundfile`` format override.
            progress (bool): Show a rich.Progress bar during generation.

        Returns:
            dict: Same schema as ``save_stream``.

        Raises:
            RuntimeError: If the generator yielded zero chunks.
        """
        filename = str(filename)
        fmt = self._resolve_format(filename, format)
        description = self._make_progress_description(request)

        original_stream = request.stream
        request.stream = True

        writer = self._StreamingFileWriter(filename, fmt)

        try:
            gen = await self.generate_speech_async(request)
            if progress:
                with self._progress_context(
                    None, description, print_summary=False
                ) as advance:
                    async for chunk in gen:
                        writer.write(chunk)
                        advance(chunk)
            else:
                async for chunk in gen:
                    writer.write(chunk)
        finally:
            request.stream = original_stream
            writer.close()

        if writer.total_samples == 0:
            raise RuntimeError('save_stream_async: generate_speech_async yielded zero chunks')

        stats = writer.stats()
        self._print_stream_summary(stats)
        return stats

    async def shutdown(self):
        """Shuts down the TTS engine and scheduler."""
        if self.scheduler:
            await self.scheduler.shutdown()
        if self.tts_engine and hasattr(self.tts_engine, 'shutdown'):
            await self.tts_engine.shutdown()
        # Release worker threads spawned by asyncio.to_thread calls
        # (HiFi-GAN decode, GPT conditioning). Without this the default
        # ThreadPoolExecutor keeps ~6-8 worker threads alive until process
        # exit — observed as a persistent +6 thread delta after long jobs.
        # We do NOT close the loop: on Colab/Jupyter self.loop is the host
        # notebook loop (grabbed via get_running_loop in _ensure_event_loop)
        # and must survive TTS shutdown.
        if self.loop and not self.loop.is_closed():
            try:
                await self.loop.shutdown_default_executor()
            except Exception as e:
                self.logger.warning(f"shutdown_default_executor failed: {e}")