#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Unit tests for test_gemm_search_space.py's sampler and skip path.

Both behaviours regressed silently once before and neither is observable from
the runner's own output, so they are pinned here rather than left to the GPU
lane: these run on any box, with no GPU, hipcc or built kernels.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_TESTS = Path(__file__).parent
sys.path.insert(0, str(_TESTS.parent / "python"))


def _load_runner():
    """Import the runner by path.

    It is a script whose name begins with ``test_``, so a plain ``import`` would
    have pytest collect it as a second test module.
    """
    spec = importlib.util.spec_from_file_location(
        "_gemm_search_space_runner", _TESTS / "test_gemm_search_space.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = _load_runner()

# The real 16 strata the standard variant enumerates.
_DTYPES = ["bf16", "bf8", "fp16", "fp8"]
_LAYOUTS = ["ccr", "crr", "rcr", "rrr"]


def _strata(per_stratum=10):
    return {
        f"{d}/{l}": [f"{d}/{l}/cfg{i}" for i in range(per_stratum)]
        for d in _DTYPES
        for l in _LAYOUTS
    }


def _dtypes_of(selected):
    return {s.split("/")[0] for s in selected}


def test_budget_below_stratum_count_varies_which_strata_are_picked():
    """budget < n_strata must let the seed move which *strata* are visited.

    The remainder used to be handed out in sorted stratum order, so with
    ``budget=4`` over 16 strata every share was 0, the whole budget was
    remainder, and it always landed on bf16/{ccr,crr,rcr,rrr}. The ctest
    registration was therefore permanently bf16-only and the daily rotating
    seed bought nothing.

    This asserts on the stratum keys, not on the selected configs: the old code
    did vary the configs it drew from within those four fixed bf16 strata, so a
    whole-sample comparison passes against the bug and pins nothing.
    """
    strata = _strata()
    visited = {
        frozenset(s.rsplit("/", 1)[0] for s in runner._sample(strata, budget=4, seed=s))
        for s in range(20)
    }
    assert len(visited) > 1, "every seed drew from the identical four strata"


def test_budget_below_stratum_count_reaches_every_dtype_across_seeds():
    """Across seeds the union of picks must reach every dtype, not just bf16."""
    seen = set()
    for seed in range(50):
        seen |= _dtypes_of(runner._sample(_strata(), budget=4, seed=seed))
    assert seen == set(_DTYPES)


def test_sample_is_reproducible_from_seed():
    strata = _strata()
    assert runner._sample(strata, budget=7, seed=42) == runner._sample(
        strata, budget=7, seed=42
    )


@pytest.mark.parametrize("budget", [1, 4, 15, 16, 17, 64, 160])
def test_sample_never_exceeds_budget(budget):
    selected = runner._sample(_strata(), budget=budget, seed=7)
    assert len(selected) == budget
    assert len(set(selected)) == len(selected), "sampled the same config twice"


def test_sample_handles_strata_smaller_than_their_share():
    """A short stratum releases its slack instead of over-drawing.

    ``random.sample`` raises if asked for more items than exist, so the clamp
    and the top-up draw have to agree.
    """
    strata = {"bf16/rcr": ["a", "b"], "fp16/rcr": [f"c{i}" for i in range(50)]}
    selected = runner._sample(strata, budget=20, seed=3)
    assert len(selected) == 20
    assert len(set(selected)) == 20


def test_sample_caps_at_total_when_budget_exceeds_space():
    strata = {"bf16/rcr": ["a", "b"], "fp16/rcr": ["c"]}
    assert sorted(runner._sample(strata, budget=99, seed=1)) == ["a", "b", "c"]


def test_sample_empty_inputs():
    assert runner._sample({}, budget=10, seed=1) == []
    assert runner._sample({"bf16/rcr": []}, budget=10, seed=1) == []
    assert runner._sample(_strata(), budget=0, seed=1) == []


def test_no_gpu_exits_77(monkeypatch, capsys):
    """A CPU-only runner must report skipped, not walk into hipcc.

    ``detect_gpu_arch`` defaults to ``fallback="gfx942"`` and so never returns
    falsy; calling it bare made this skip path dead code and a GPU-less box
    compiled kernels for an arch it does not have.

    The stub returns the fallback it is handed rather than a fixed "", which is
    what the real function does when rocminfo finds nothing. That is what makes
    this test discriminating: the runner only skips if it asks for ``fallback=""``.
    Stubbing a bare "" would pass against the old bare ``detect_gpu_arch()`` call
    too, and pin nothing.
    """
    monkeypatch.setattr(runner, "detect_gpu_arch", lambda fallback="gfx942": fallback)
    monkeypatch.setattr(sys, "argv", ["test_gemm_search_space.py", "--budget", "1"])
    assert runner.main() == runner._SKIP
    assert "SKIP" in capsys.readouterr().out


def test_unsupported_arch_exits_77(capsys):
    """An unsupported GPU skips too, rather than failing every hipcc step."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(sys, "argv", ["x", "--arch", "gfx90a", "--budget", "1"])
    try:
        assert runner.main() == runner._SKIP
    finally:
        monkeypatch.undo()
    assert "SKIP" in capsys.readouterr().out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
