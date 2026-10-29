#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Tests for select_kernels.py and its runbook (HEURISTIC_BENCHMARKING.md).

Covers: the driver-to-feature-engine field mapping, top-K selection with
deterministic ties, the model/run compatibility check, an end-to-end run on an
in-tree model, the runbook's scoring snippet, and that every option the
runbook's commands use exists.
"""

import contextlib
import io
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

HEURISTICS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HEURISTICS_DIR))

import select_kernels as sk  # noqa: E402

RUNBOOK = HEURISTICS_DIR / "HEURISTIC_BENCHMARKING.md"
DRIVER = (
    HEURISTICS_DIR.parents[1]
    / "tile_engine"
    / "ops"
    / "gemm"
    / "gemm_full_benchmark.py"
)
FP16_MODEL = HEURISTICS_DIR / "models" / "gemm_universal_fp16_gfx950"


def _cfg(tile=(128, 128, 64), waves=(2, 2, 1), vec=(0, 0, 0), pipeline="compv3"):
    return {
        "tile_m": tile[0], "tile_n": tile[1], "tile_k": tile[2],
        "wave_m": waves[0], "wave_n": waves[1], "wave_k": waves[2],
        "warp_tile_m": 32, "warp_tile_n": 32, "warp_tile_k": 16,
        "pipeline": pipeline, "scheduler": "intrawave", "epilogue": "cshuffle",
        "pad_m": True, "pad_n": True, "pad_k": True, "persistent": False,
        "vector_size_a": vec[0], "vector_size_b": vec[1], "vector_size_c": vec[2],
    }  # fmt: skip


def _candidates(dtype="fp16", arch="gfx950"):
    kernels = {
        "ka": _cfg(),
        "kb": _cfg(tile=(256, 128, 64)),
        "kc": _cfg(waves=(4, 1, 1)),
        "kd": _cfg(vec=(2, 2, 8)),
    }
    pairs = {"1024x1024x512": ["ka", "kb", "kc"], "512x256x250": ["kd", "ka"]}
    return {"arch": arch, "dtype": dtype, "layout": "rcr", "variant": "standard",
            "kernels": kernels, "pairs": pairs}  # fmt: skip


class _Stub:
    """Predictor stand-in: a fixed score per kernel, recording each call."""

    def __init__(self, scores):
        self.scores, self.calls = scores, []

    def rank_kernels(self, problem, kernels):
        self.calls.append((problem, kernels))
        return [(k["kernel_name"], self.scores[k["kernel_name"]]) for k in kernels]


def test_feature_kernel_maps_driver_names():
    feat = sk.feature_kernel("k", _cfg(waves=(4, 1, 1), vec=(2, 4, 8)))
    assert (feat["warp_m"], feat["warp_n"], feat["warp_k"]) == (4, 1, 1)
    assert (feat["warp_tile_m"], feat["warp_tile_k"]) == (32, 16)
    assert (feat["vec_a"], feat["vec_b"], feat["vec_c"]) == (2, 4, 8)
    assert feat["kernel_name"] == "k" and "wave_m" not in feat


def test_select_keeps_top_k_and_breaks_ties_on_name():
    stub = _Stub({"ka": 5.0, "kb": 7.0, "kc": 5.0, "kd": 1.0})
    picks, rows = sk.select(_candidates(), stub, top_k=2)
    assert picks == {"1024x1024x512": ["kb", "ka"], "512x256x250": ["ka", "kd"]}
    assert [(r["problem"], r["rank"], r["kernel"]) for r in rows][:2] == [
        ("1024x1024x512", 1, "kb"),
        ("1024x1024x512", 2, "ka"),
    ]
    assert rows[0]["candidates"] == 3
    problem = stub.calls[0][0]
    assert (problem["m"], problem["n"], problem["k"]) == (1024, 1024, 512)
    assert (problem["dtype"], problem["layout"], problem["arch"]) == (
        "fp16",
        "rcr",
        "gfx950",
    )


@pytest.mark.parametrize(
    "spec, candidates, ok",
    [
        ({"dtype": "fp16", "arch": "gfx950"}, _candidates(), True),
        ({"dtype": "fp16", "arch": "gfx950"}, _candidates(arch="gfx950:xnack-"), True),
        ({"dtype": "bf16", "arch": "gfx950"}, _candidates(), False),
        ({"dtype": "fp16", "arch": "gfx942"}, _candidates(), False),
        ({}, {**_candidates(), "variant": "multi_abd"}, False),
    ],
)
def test_model_must_fit_the_run(spec, candidates, ok):
    if ok:
        sk.check_model_fits(spec, candidates)
    else:
        with pytest.raises(ValueError):
            sk.check_model_fits(spec, candidates)


@pytest.mark.skipif(not FP16_MODEL.exists(), reason="in-tree fp16 model not present")
def test_end_to_end_with_an_in_tree_model(tmp_path):
    cand = tmp_path / "candidates.json"
    cand.write_text(json.dumps(_candidates()))
    out_pairs, out_ranked = tmp_path / "picks.json", tmp_path / "picks.csv"
    argv = [
        "--candidates", str(cand), "--model_dir", str(FP16_MODEL), "--top_k", "2",
        "--out_pairs", str(out_pairs), "--out_ranked", str(out_ranked),
    ]  # fmt: skip
    assert sk.main(argv) == 0
    picks = json.loads(out_pairs.read_text())
    assert set(picks) == set(_candidates()["pairs"])
    assert all(len(v) == 2 for v in picks.values())
    ranked = pd.read_csv(out_ranked)
    assert list(ranked.columns) == sk.RANKED_FIELDS and len(ranked) == 4
    # A second run writes the same picks.
    first = out_pairs.read_text()
    assert sk.main(argv) == 0 and out_pairs.read_text() == first


def _code_blocks(lang):
    return re.findall(rf"```{lang}\n(.*?)```", RUNBOOK.read_text(), re.S)


def test_runbook_scoring_snippet(tmp_path):
    (snippet,) = _code_blocks("python")
    rep = ["kernel,M,N,K,rep,latency_ms,ok"]
    sweep = rep + [f"x,64,64,64,{r},1.0,1" for r in range(3)]
    sweep += [f"y,64,64,64,{r},{v},1" for r, v in enumerate((2.0, 2.0, 9.0))]
    (tmp_path / "sweep.repeats.csv").write_text("\n".join(sweep) + "\n")
    measured = rep + [f"y,64,64,64,{r},2.0,1" for r in range(3)]
    measured += [f"x,64,64,64,{r},1.0,1" for r in range(3)]
    (tmp_path / "measured.repeats.csv").write_text("\n".join(measured) + "\n")
    picks = ["problem,rank,kernel,predicted_tflops,candidates"]
    picks += ["64x64x64,1,y,9.0,2", "64x64x64,2,x,8.0,2"]
    (tmp_path / "picks.csv").write_text("\n".join(picks) + "\n")
    out, cwd = io.StringIO(), os.getcwd()
    try:
        os.chdir(tmp_path)
        with contextlib.redirect_stdout(out):
            exec(snippet, {})
    finally:
        os.chdir(cwd)
    lines = out.getvalue().splitlines()
    assert lines[0].startswith("K=1: mean 0.500")
    assert lines[1].startswith("K=3: mean 1.000")


def _runbook_options():
    """{script name: set of --options} used by the runbook's shell commands."""
    used = {}
    for block in _code_blocks("bash"):
        for cmd in block.replace("\\\n", " ").splitlines():
            script = re.search(r"(\w+\.py)", cmd)
            if script:
                used.setdefault(script.group(1), set()).update(
                    re.findall(r"(--[a-z][a-z0-9_-]*)", cmd)
                )
    return used


@pytest.mark.parametrize(
    "script", ["select_kernels.py", "train.py", "convert_csv_to_parquet.py"]
)
def test_runbook_options_exist(script):
    used = _runbook_options()[script]
    source = (HEURISTICS_DIR / script).read_text()
    assert used and all(f'"{opt}"' in source for opt in used), used


def test_runbook_driver_options_exist():
    source = DRIVER.read_text()
    if '"--list-pairs"' not in source:
        pytest.skip("the driver's measurement options are not in this tree yet")
    used = _runbook_options()["gemm_full_benchmark.py"]
    missing = sorted(o for o in used if f'"{o}"' not in source)
    assert not missing, missing
