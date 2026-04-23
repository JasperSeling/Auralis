"""Regression tests for HiddenStatesCollector RAM / thread leak.

Ensures that when a single collector instance is reused across many requests
(the engine-wide pattern introduced after the per-sentence leak bug), no
per-request state or threads accumulate after each ``get_hidden_states``
completes — including on the timeout and no-data error paths.
"""

from __future__ import annotations

import asyncio
import importlib.util
import pathlib
import threading

import torch

# Load the collector module directly from its source file to keep the unit
# test hermetic — importing via ``auralis.models.xttsv2...`` would pull in the
# whole package __init__ chain (including torio/vLLM) which is overkill for a
# state-cleanup regression test and is platform-sensitive.
_COLLECTOR_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src"
    / "auralis"
    / "models"
    / "xttsv2"
    / "components"
    / "vllm"
    / "hidden_state_collector.py"
)
_spec = importlib.util.spec_from_file_location(
    "_hidden_state_collector_under_test", _COLLECTOR_PATH
)
# The module imports ``auralis.common.logging.logger`` for ``setup_logger``;
# provide a minimal stub under that dotted name so the direct load succeeds
# without triggering the full auralis package init.
import sys
import logging
import types as _types


def _install_logger_stub() -> None:
    if "auralis.common.logging.logger" in sys.modules:
        return
    for name in ("auralis", "auralis.common", "auralis.common.logging"):
        sys.modules.setdefault(name, _types.ModuleType(name))
    logger_mod = _types.ModuleType("auralis.common.logging.logger")
    logger_mod.setup_logger = lambda _name: logging.getLogger("test-collector")
    sys.modules["auralis.common.logging.logger"] = logger_mod


_install_logger_stub()
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
HiddenStatesCollector = _module.HiddenStatesCollector


def _state_footprint(collector: HiddenStatesCollector) -> dict:
    """Snapshot of all per-request dict sizes. All should stay 0 between jobs."""
    return {
        "outputs": len(collector.outputs),
        "collection_ready": len(collector.collection_ready),
        "collection_complete": len(collector.collection_complete),
        "locks": len(collector.locks),
        "states_count": len(collector.states_count),
        "expected_states": len(collector.expected_states),
        "notifications": len(collector.notifications),
    }


def test_success_path_no_state_growth():
    """Repeated bind -> sync_collect -> get_hidden_states leaves zero state."""
    collector = HiddenStatesCollector()
    assert all(v == 0 for v in _state_footprint(collector).values())

    async def _run():
        for i in range(20):
            rid = f"req-{i}"
            wrapper = collector.bind_to_request(rid)
            # Feed one hidden-state tensor (matches expected_states default of 1).
            wrapper(torch.ones(4, 8), rid)
            result = await collector.get_hidden_states(rid, timeout=1.0)
            assert result is not None
            assert result.shape == (4, 8)

    asyncio.run(_run())

    assert _state_footprint(collector) == {
        "outputs": 0,
        "collection_ready": 0,
        "collection_complete": 0,
        "locks": 0,
        "states_count": 0,
        "expected_states": 0,
        "notifications": 0,
    }, "success path must not retain per-request state"


def test_timeout_path_cleans_up_state():
    """get_hidden_states timing out must still evict the request's state.

    Previously the timeout branch in get_hidden_states returned None without
    calling _cleanup_request, leaving the dicts populated forever.
    """
    collector = HiddenStatesCollector()

    async def _run():
        for i in range(5):
            rid = f"timeout-{i}"
            collector.bind_to_request(rid)
            # Never call sync_collect -> collection_complete.wait() will time out.
            result = await collector.get_hidden_states(rid, timeout=0.05)
            assert result is None

    asyncio.run(_run())

    assert _state_footprint(collector) == {
        "outputs": 0,
        "collection_ready": 0,
        "collection_complete": 0,
        "locks": 0,
        "states_count": 0,
        "expected_states": 0,
        "notifications": 0,
    }, "timeout path must evict per-request state"


def test_shutdown_releases_all_state():
    """shutdown() must drop per-request state even for in-flight requests."""
    collector = HiddenStatesCollector()
    for i in range(10):
        collector.bind_to_request(f"inflight-{i}")

    assert _state_footprint(collector)["outputs"] == 10

    collector.shutdown()

    assert _state_footprint(collector) == {
        "outputs": 0,
        "collection_ready": 0,
        "collection_complete": 0,
        "locks": 0,
        "states_count": 0,
        "expected_states": 0,
        "notifications": 0,
    }

    # Idempotent: a second shutdown must not raise.
    collector.shutdown()


def test_reuse_does_not_spawn_threads():
    """Reusing one collector across N requests must not spawn N threads.

    Before the fix, a new ThreadPoolExecutor(max_workers=4) was constructed
    inside __init__ and a fresh collector was built per sentence — after ~900
    sentences, threads and executor objects accumulated. A shared collector
    plus the removal of the unused executor field must keep the thread count
    flat across many requests.
    """
    baseline = threading.active_count()
    collector = HiddenStatesCollector()
    # The collector itself should not spawn background threads on construction.
    assert threading.active_count() == baseline, (
        f"constructing HiddenStatesCollector spawned threads: "
        f"{threading.active_count() - baseline}"
    )

    async def _run():
        for i in range(50):
            rid = f"reuse-{i}"
            wrapper = collector.bind_to_request(rid)
            wrapper(torch.ones(2, 3), rid)
            await collector.get_hidden_states(rid, timeout=1.0)
        # Allow any transient asyncio worker threads to settle.
        await asyncio.sleep(0.05)

    asyncio.run(_run())
    delta = threading.active_count() - baseline
    assert delta <= 1, (
        f"thread count drifted by {delta} after 50 reuse cycles — "
        f"expected stable thread count across requests"
    )


def test_executor_attribute_removed():
    """The unused ThreadPoolExecutor field must be gone from __init__.

    Keeping it pinned 4 potential worker threads per collector instance for
    the lifetime of the engine; in the per-sentence collector pattern it was
    also leaking on every request. The field is dead code and should remain
    removed.
    """
    collector = HiddenStatesCollector()
    assert not hasattr(collector, "executor"), (
        "HiddenStatesCollector.executor must stay removed — it was dead code "
        "that pinned ThreadPoolExecutor resources unnecessarily"
    )
