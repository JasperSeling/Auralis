from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_xtts_uses_vllm_019_import_paths():
    xtts_source = (ROOT / "src" / "auralis" / "models" / "xttsv2" / "XTTSv2.py").read_text(
        encoding="utf-8"
    )
    mm_source = (
        ROOT / "src" / "auralis" / "models" / "xttsv2" / "components" / "vllm_mm_gpt.py"
    ).read_text(encoding="utf-8")

    assert "from vllm.inputs import MultiModalDataDict" in xtts_source
    assert "from vllm.utils.counter import Counter" in xtts_source
    assert "from vllm.multimodal.inputs import MultiModalDataDict" not in xtts_source
    assert "from vllm.utils import Counter" not in xtts_source

    assert "from vllm.v1.attention.backend import AttentionMetadata" in mm_source
    assert "from vllm.distributed.parallel_state import get_pp_group" in mm_source
    assert "from vllm.v1.sample.sampler import Sampler" in mm_source
    assert "from vllm.v1.outputs import SamplerOutput" in mm_source
    assert "from vllm.v1.sample.metadata import SamplingMetadata" in mm_source
    assert "from vllm.attention import AttentionMetadata" not in mm_source
    assert "from vllm.distributed import get_pp_group" not in mm_source
    assert "from vllm.model_executor.layers.sampler import Sampler, SamplerOutput" not in mm_source
    assert "from vllm.model_executor.sampling_metadata import SamplingMetadata" not in mm_source
