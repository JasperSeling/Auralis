from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_output_does_not_depend_on_codec_config():
    source = (ROOT / "src" / "auralis" / "common" / "definitions" / "output.py").read_text(
        encoding="utf-8"
    )

    assert "CodecConfig" not in source
