__all__ = [
    "TTS",
    "TTSRequest",
    "TTSOutput",
    "AudioPreprocessingConfig",
    "setup_logger",
    "set_vllm_logging_level",
]


def __getattr__(name):
    if name == "TTS":
        from .core.tts import TTS
        return TTS
    if name == "TTSRequest":
        from .common.definitions.requests import TTSRequest
        return TTSRequest
    if name == "TTSOutput":
        from .common.definitions.output import TTSOutput
        return TTSOutput
    if name == "AudioPreprocessingConfig":
        from .common.definitions.enhancer import AudioPreprocessingConfig
        return AudioPreprocessingConfig
    if name in {"setup_logger", "set_vllm_logging_level"}:
        from .common.logging.logger import setup_logger, set_vllm_logging_level

        return {
            "setup_logger": setup_logger,
            "set_vllm_logging_level": set_vllm_logging_level,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
