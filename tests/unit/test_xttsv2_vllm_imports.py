from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_xttsv2_imports_multimodal_data_dict_from_inputs_module():
    source = (ROOT / "src" / "auralis" / "models" / "xttsv2" / "XTTSv2.py").read_text(
        encoding="utf-8"
    )

    assert "from vllm.multimodal.inputs import MultiModalDataDict" in source
    assert "from vllm.multimodal import MultiModalDataDict" not in source
