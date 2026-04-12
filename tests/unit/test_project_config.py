import json
from pathlib import Path

import tomli


ROOT = Path(__file__).resolve().parents[2]


def test_pyproject_uses_numpy2_compatible_runtime_stack():
    pyproject = tomli.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]

    assert pyproject["project"]["requires-python"] == ">=3.10,<3.13"
    assert "vllm==0.19.0" in dependencies
    assert "torch==2.10.0" in dependencies
    assert "torchaudio==2.10.0" in dependencies
    assert "transformers>=4.56.0,<5" in dependencies
    assert "thinc>=8.3.12,<8.4.0" in dependencies
    assert "spacy>=3.8.0,<3.9.0" in dependencies
    assert "numpy>=2.0.0,<2.2.0" in dependencies
    assert all(not dependency.startswith("ipython") for dependency in dependencies)
    assert "thinc>=8.3.0" not in dependencies
    assert "spacy==3.7.5" not in dependencies
    assert "spacy>=3.8.0" not in dependencies
    assert "ipython>=7.34.0" not in dependencies
    assert "ipython>=8.0.0" not in dependencies
    assert "numpy>=1.26.0" not in dependencies
    assert "numpy>=1.26.0,<2.0" not in dependencies


def test_openai_server_console_script_uses_hyphenated_name():
    pyproject = tomli.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"] == {
        "auralis-openai": "auralis.entrypoints.oai_server:main",
    }


def test_pyproject_registers_vllm_general_plugin():
    pyproject = tomli.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["entry-points"]["vllm.general_plugins"] == {
        "auralis_xtts": "auralis.models.xttsv2.components.vllm_mm_gpt:register_vllm_models",
    }


def test_tts_lazy_imports_model_package_before_registry_lookup():
    source = (ROOT / "src" / "auralis" / "core" / "tts.py").read_text(encoding="utf-8")

    assert "import auralis.models" in source
    assert source.index("import auralis.models") < source.index(
        "from auralis.models.registry import MODEL_REGISTRY"
    )


def test_xtts_vllm_compatibility_source_hooks_are_present():
    xtts_source = (
        ROOT / "src" / "auralis" / "models" / "xttsv2" / "XTTSv2.py"
    ).read_text(encoding="utf-8")
    mm_source = (
        ROOT / "src" / "auralis" / "models" / "xttsv2" / "components" / "vllm_mm_gpt.py"
    ).read_text(encoding="utf-8")

    assert xtts_source.startswith(
        'import os\nos.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")\n'
    )
    assert 'device="cuda"' not in xtts_source
    assert 'load_format="pt"' in xtts_source
    assert 'load_format="auto"' not in xtts_source
    assert 'plugin = "auralis.models.xttsv2.components.vllm_mm_gpt"' in xtts_source
    assert 'plugins = os.environ.get("VLLM_PLUGINS", "")' in xtts_source
    assert 'os.environ["VLLM_PLUGINS"] = (plugins + "," + plugin).strip(",")' in xtts_source
    assert "MultiModalInputs" not in mm_source
    assert "INPUT_REGISTRY" not in mm_source
    assert "register_input_mapper" not in mm_source
    assert "register_max_multimodal_tokens" not in mm_source
    assert "@MULTIMODAL_REGISTRY.register_processor" in mm_source
    assert "def register_vllm_models():" in mm_source
    assert '"""Called by vLLM general plugin system in every worker process."""' in mm_source
    assert "from vllm.model_executor.models import ModelRegistry" in mm_source
    assert 'if "XttsGPT" not in ModelRegistry.get_supported_archs():' in mm_source
    assert 'ModelRegistry.register_model("XttsGPT", XttsGPT)' in mm_source
    assert "XttsMultiModalProcessor" in mm_source
    assert "XttsProcessingInfo" in mm_source
    assert "XttsDummyInputsBuilder" in mm_source


def test_colab_quickstart_sets_vllm_env_before_install_cell():
    notebook = json.loads(
        (ROOT / "notebooks" / "colab_quickstart.ipynb").read_text(encoding="utf-8")
    )

    assert notebook["cells"][0]["source"] == [
        "import os\n",
        'os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"\n',
    ]
    assert notebook["cells"][1]["source"] == [
        "!pip install ipython==7.34.0 -q\n",
        "!apt-get install -q libportaudio2\n",
        (
            "!pip install torchaudio==2.10.0 "
            "--index-url https://download.pytorch.org/whl/cu128 -q\n"
        ),
        (
            '!pip install "thinc>=8.3.0" "spacy==3.7.5" '
            "--force-reinstall --no-cache-dir -q\n"
        ),
        "!pip install vllm==0.19.0 -q\n",
        (
            "!pip install git+https://github.com/JasperSeling/Auralis.git "
            "--no-cache-dir -q\n"
        ),
    ]
