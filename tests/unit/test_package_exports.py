from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_package_init_uses_explicit_top_level_imports():
    source = (ROOT / "src" / "auralis" / "__init__.py").read_text(encoding="utf-8")

    assert source.startswith(
        'import os\n'
        '\n'
        'os.environ.setdefault("VLLM_USE_V1", "0")\n'
        'os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")\n'
    )
    assert "from auralis.core.tts import TTS" in source
    assert "from auralis.common.definitions.requests import TTSRequest" in source
    assert "from auralis.common.definitions.output import TTSOutput" in source
    assert "from auralis.common.definitions.enhancer import AudioPreprocessingConfig" in source
    assert (
        "from auralis.common.logging.logger import setup_logger, "
        "set_vllm_logging_level" in source
    )
    assert '    "TTS",' in source
    assert '    "TTSRequest",' in source
    assert '    "TTSOutput",' in source
    assert '    "AudioPreprocessingConfig",' in source
    assert '    "setup_logger",' in source
    assert '    "set_vllm_logging_level",' in source
    assert "__getattr__" not in source
