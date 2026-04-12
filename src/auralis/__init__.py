import os

os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from auralis.core.tts import TTS
from auralis.common.definitions.requests import TTSRequest
from auralis.common.definitions.output import TTSOutput
from auralis.common.definitions.enhancer import AudioPreprocessingConfig
from auralis.common.logging.logger import setup_logger, set_vllm_logging_level

__all__ = [
    "TTS",
    "TTSRequest",
    "TTSOutput",
    "AudioPreprocessingConfig",
    "setup_logger",
    "set_vllm_logging_level",
]
