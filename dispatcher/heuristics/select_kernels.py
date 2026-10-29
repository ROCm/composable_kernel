#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Pick kernels for tile_engine/ops/gemm/gemm_full_benchmark.py with a trained model.

    python3 select_kernels.py --candidates candidates.json \\
        --model_dir models/gemm_universal_vec_bf16_gfx950 --top_k 5 \\
        --out_pairs picks.json --out_ranked picks.csv

``--candidates`` is the driver's ``--list-pairs`` output: the (problem, kernel)
pairs a run would measure, each kernel's configuration, and the run's arch,
dtype and layout. Each problem's candidates are ranked by
``Predictor.rank_kernels``; the top K go to ``--out_pairs``, a ``--pairs`` file
for the driver, and to ``--out_ranked``, a CSV with one row per pick.

The candidates come from the driver's own pairing (vector-width and tile rules
applied), so every pick is a pair the driver can launch. Ties in the predicted
value break on the kernel name, so the output depends only on the inputs.
See HEURISTIC_BENCHMARKING.md for the workflow around it.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "codegen"))

from codegen_common import normalize_gfx_arch  # noqa: E402

RANKED_FIELDS = ["problem", "rank", "kernel", "predicted_tflops", "candidates"]


def feature_kernel(name: str, cfg: dict) -> dict:
    """A driver kernel configuration in the feature engine's field names.

    The driver (GemmKernelConfig) calls the warps per block ``wave_m/n/k`` and
    the per-warp tile ``warp_tile_m/n/k``; the feature engine calls them
    ``warp_m/n/k`` and ``warp_tile_m/n/k`` (see dispatcher_integration.py).
    ``vector_size_a/b/c`` become ``vec_a/b/c``; 0 means the kernel's native
    width in both.
    """
    return {
        "kernel_name": name,
        "tile_m": cfg["tile_m"],
        "tile_n": cfg["tile_n"],
        "tile_k": cfg["tile_k"],
        "warp_m": cfg["wave_m"],
        "warp_n": cfg["wave_n"],
        "warp_k": cfg["wave_k"],
        "warp_tile_m": cfg["warp_tile_m"],
        "warp_tile_n": cfg["warp_tile_n"],
        "warp_tile_k": cfg["warp_tile_k"],
        "pipeline": cfg["pipeline"],
        "scheduler": cfg["scheduler"],
        "epilogue": cfg["epilogue"],
        "pad_m": bool(cfg["pad_m"]),
        "pad_n": bool(cfg["pad_n"]),
        "pad_k": bool(cfg["pad_k"]),
        "persistent": bool(cfg["persistent"]),
        "vec_a": int(cfg.get("vector_size_a", 0)),
        "vec_b": int(cfg.get("vector_size_b", 0)),
        "vec_c": int(cfg.get("vector_size_c", 0)),
    }


def check_model_fits(spec: dict, candidates: dict) -> None:
    """Raise if the model was trained for another dtype or arch than the run."""
    for key, norm in (("dtype", str), ("arch", normalize_gfx_arch)):
        trained, run = spec.get(key), candidates.get(key)
        if trained and run and norm(str(trained)) != norm(str(run)):
            raise ValueError(
                f"the model was trained for {key} {trained}, the candidates are "
                f"for {key} {run}"
            )
    if candidates.get("variant", "standard") != "standard":
        raise ValueError(
            f"only gemm_universal candidates can be ranked, got variant "
            f"{candidates['variant']!r}"
        )


def select(candidates: dict, predictor, top_k: int):
    """Rank every problem's candidates; return ({problem: [top_k names]}, rows)."""
    kernels = candidates["kernels"]
    base = {
        "dtype": candidates["dtype"],
        "layout": candidates["layout"],
        "arch": candidates["arch"],
        "split_k": 1,
    }
    picks, rows = {}, []
    for key, names in candidates["pairs"].items():
        m, n, k = (int(d) for d in key.split("x"))
        ranked = predictor.rank_kernels(
            {**base, "m": m, "n": n, "k": k},
            [feature_kernel(name, kernels[name]) for name in names],
        )
        ranked = sorted(ranked, key=lambda t: (-t[1], t[0]))[:top_k]
        picks[key] = [name for name, _ in ranked]
        rows.extend(
            {
                "problem": key,
                "rank": rank,
                "kernel": name,
                "predicted_tflops": tflops,
                "candidates": len(names),
            }
            for rank, (name, tflops) in enumerate(ranked, 1)
        )
    return picks, rows


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Pick the top-K kernels per problem for gemm_full_benchmark.py"
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="gemm_full_benchmark.py --list-pairs output",
    )
    parser.add_argument(
        "--model_dir", required=True, help="Model directory written by train.py"
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="Kernels per problem (default 5)"
    )
    parser.add_argument(
        "--out_pairs",
        required=True,
        help="Picks as a gemm_full_benchmark.py --pairs file",
    )
    parser.add_argument(
        "--out_ranked",
        default=None,
        help="CSV with one row per pick: problem, rank, kernel, predicted_tflops",
    )
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top_k must be >= 1")

    from predict import Predictor

    candidates = json.loads(Path(args.candidates).read_text())
    predictor = Predictor(args.model_dir)
    check_model_fits(predictor.spec, candidates)
    picks, rows = select(candidates, predictor, args.top_k)

    Path(args.out_pairs).write_text(json.dumps(picks, indent=1) + "\n")
    if args.out_ranked:
        with open(args.out_ranked, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RANKED_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    n_kernels = len({name for names in picks.values() for name in names})
    print(
        f"{len(picks)} problems, {len(rows)} picks, {n_kernels} distinct kernels "
        f"(from {len(candidates['kernels'])} candidates) -> {args.out_pairs}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
