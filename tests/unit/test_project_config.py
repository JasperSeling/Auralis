from pathlib import Path

import tomli


ROOT = Path(__file__).resolve().parents[2]


def test_pyproject_uses_python312_colab_dependency_stack():
    pyproject = tomli.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]

    assert pyproject["project"]["requires-python"] == ">=3.10,<3.13"
    assert "vllm==0.19.0" in dependencies
    assert "torch==2.10.0" in dependencies
    assert "torchaudio==2.10.0" in dependencies
    assert "transformers>=4.56.0,<5" in dependencies
    assert "thinc>=8.3.0" in dependencies
    assert "spacy>=3.7.5" in dependencies
    assert "numpy>=1.26.0" in dependencies
    assert "ipython>=7.34.0" in dependencies
    assert "ipython>=8.0.0" not in dependencies
    assert "numpy>=1.26.0,<2.0" not in dependencies


def test_openai_server_console_script_uses_hyphenated_name():
    pyproject = tomli.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"] == {
        "auralis-openai": "auralis.entrypoints.oai_server:main",
    }


def test_tts_lazy_imports_model_package_before_registry_lookup():
    source = (ROOT / "src" / "auralis" / "core" / "tts.py").read_text(encoding="utf-8")

    assert "import auralis.models" in source
    assert source.index("import auralis.models") < source.index("from auralis.models.registry import MODEL_REGISTRY")


def test_xtts_vllm_compatibility_source_hooks_are_present():
    xtts_source = (ROOT / "src" / "auralis" / "models" / "xttsv2" / "XTTSv2.py").read_text(encoding="utf-8")
    mm_source = (
        ROOT / "src" / "auralis" / "models" / "xttsv2" / "components" / "vllm_mm_gpt.py"
    ).read_text(encoding="utf-8")

    assert 'device="cuda"' in xtts_source
    assert "MultiModalInputs" not in mm_source
    assert "INPUT_REGISTRY" not in mm_source
    assert "register_input_mapper" not in mm_source
    assert "register_max_multimodal_tokens" not in mm_source
    assert "@MULTIMODAL_REGISTRY.register_processor" in mm_source
    assert "XttsMultiModalProcessor" in mm_source
    assert "XttsProcessingInfo" in mm_source
    assert "XttsDummyInputsBuilder" in mm_source
