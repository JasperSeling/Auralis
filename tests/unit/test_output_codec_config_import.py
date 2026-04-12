from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_codec_config_import_falls_back_to_torchaudio():
    source = (ROOT / "src" / "auralis" / "common" / "definitions" / "output.py").read_text(
        encoding="utf-8"
    )

    assert "try:\n    from torio.io import CodecConfig" in source
    assert "except ImportError:\n    from torchaudio.io import CodecConfig" in source
