#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
GPU correctness test for the preshuffle GEMM dispatcher bridge.

Builds the complete five-pipeline gfx1250 configuration for fp16 and bf16/rcr,
plus padded compute TDM configurations to check rejection of partial N/K tiles,
and the existing V2 configuration on gfx9. Exercises all supported epilogues and
persistent modes, short and hot loops, odd and even K tails, multiple output
tiles, and changing B values against an fp32 NumPy reference. Returns 77 when
the GPU or dispatcher build is unavailable.

The packed-B pipelines (preshufflev2, preshuffle_tdm) read the B (weight)
operand in a packed layout; that shuffle is done HOST-SIDE inside the ctypes .so
(ck_tile::shuffle_b_v0, selected by the kernel's Preshuffle trait; the gfx1250
compute pipelines read ordinary B and skip it), so the caller still hands
the runner logical row-major A (M x K) and logical B (K x N) — identical to the
plain-GEMM path. The result must therefore match the ordinary C = A @ B.

Run:
  python3 test_preshuffle_gpu_correctness.py
  python3 test_preshuffle_gpu_correctness.py -v
  python3 test_preshuffle_gpu_correctness.py --gfx gfx942
"""

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codegen"))

from gemm_utils import (  # noqa: E402
    GemmKernelConfig,
    GemmProblem,
    GpuGemmRunner,
    setup_multiple_gemm_dispatchers,
    _resolve_arch,
    _fp32_to_bf16_u16,
    _bf16_u16_to_fp32,
    expand_sweep,
)
from codegen_common import PRESHUFFLE_GFX1250_PIPELINES  # noqa: E402

log = logging.getLogger(__name__)

# Rounded fp16/bf16 inputs, fp32 accumulation, and dtype-appropriate output.
TOLERANCE = 1e-2

PASS = "PASS"
FAIL = "FAIL"

# ctest SKIP_RETURN_CODE: main() returns this when the box cannot run the test
# at all, so the lane reports Skipped rather than a vacuous Passed or a Failed.
SKIP_EXIT = 77

# Mirrors gemm_utils._SUPPORTED_ARCHES, which is what actually builds the
# kernel. Spelled out here because --gfx bypasses autodetection: without this
# check a typo'd or unsupported arch is passed straight through to
# --offload-arch, and the test reports a build FAIL instead of a clean skip.
# Keep in sync with gemm_utils; the two must not drift.
_SUPPORTED_ARCHS = ("gfx90a", "gfx942", "gfx950", "gfx1250")


def _has_gpu() -> bool:
    try:
        _resolve_arch(None)
        return True
    except Exception:
        return False


def _max_rel_err(C_gpu: np.ndarray, C_ref: np.ndarray) -> float:
    """Max absolute error normalized by the largest reference magnitude.

    A GEMM output has elements that partially cancel toward zero, so a naive
    per-element relative error is dominated by those near-zero entries and does
    NOT measure whether the kernel computed the right matrix. Normalizing the
    worst absolute error by the global reference scale (max |ref|) is the honest
    correctness bar: a mis-shuffled B operand (the whole risk of the preshuffle
    path) blows far past 1e-2 (GPU-verified ~1.25 for the wrong permute), while
    correct fp16 math lands at ~1e-3-1e-4 here.
    """
    g = C_gpu.astype(np.float32)
    r = C_ref.astype(np.float32)
    ref_scale = max(float(np.abs(r).max()), 1e-6)
    return float(np.max(np.abs(g - r)) / ref_scale)


def _make_preshuffle_config(gfx_arch: str, dtype: str) -> GemmKernelConfig:
    """A small rcr preshuffle kernel for the requested 16-bit dtype.

    The base kernel uses the ``preshufflev2`` pipeline (the only preshuffle
    pipeline on every arch; gfx1250 adds ``preshuffle_tdm``, see
    _preshuffle_configs) with a 16x16x32 warp-tile (see
    gemm_preshuffle/configs/default_ci_config.json). 128x128x64 tile / 2x2x1 waves is divisibility-valid:
    128/(2*16)=4, 64/(1*32)=2. variant='preshuffle' appends the _preshuffle name
    token; permute_n stays False (the only bridged shuffle, per BRIDGE_PERMUTE_N).
    rcr (col-major B) is required for the host-side shuffle_b_v0 byte-identity
    contract inside the .so.
    """
    return GemmKernelConfig(
        dtype_a=dtype, dtype_b=dtype, dtype_c=dtype, dtype_acc="fp32",
        layout_a="row", layout_b="col", layout_c="row",
        tile_m=128, tile_n=128, tile_k=64,
        wave_m=2, wave_n=2, wave_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=32,
        pipeline="preshufflev2", scheduler="default", epilogue="cshuffle",
        pad_m=False, pad_n=False, pad_k=False, persistent=False,
        variant="preshuffle", permute_n=False,
        gfx_arch=gfx_arch,
    )


def _preshuffle_configs(gfx_arch: str, dtype: str) -> list[GemmKernelConfig]:
    """preshufflev2 in every epilogue/persistent mode, plus on gfx1250 the whole
    default_config_gfx1250.json sweep (all five preshuffle pipelines)."""
    base = _make_preshuffle_config(gfx_arch, dtype)
    configs = [
        replace(base, epilogue=epilogue, persistent=persistent)
        for epilogue in ("default", "cshuffle")
        for persistent in (False, True)
    ]
    if gfx_arch.split(":")[0] == "gfx1250":
        config_path = (Path(__file__).resolve().parents[2] / "tile_engine/ops/gemm/"
                       "gemm_preshuffle/configs/default_config_gfx1250.json")
        configs += expand_sweep(str(config_path), arch=gfx_arch, dtype=dtype, variant="preshuffle")
    return configs


def _run_preshuffle(gfx_arch: str, dtype: str) -> tuple[str, str]:
    configs = _preshuffle_configs(gfx_arch, dtype)
    if gfx_arch.split(":")[0] == "gfx1250" and not set(PRESHUFFLE_GFX1250_PIPELINES) <= {
        cfg.pipeline for cfg in configs
    }:
        return FAIL, f"incomplete gfx1250 pipeline sweep: {len(configs)} configs"
    paths = setup_multiple_gemm_dispatchers(configs, max_workers=4, verbose=False)
    if len(paths) != len(configs) or any(path is None for path in paths):
        return FAIL, f"preshuffle/{dtype}: kernel build failed"

    # Cover both short-loop tails, both hot-loop tails, and multiple output tiles.
    shapes = [(128, 128, k) for k in (64, 128, 192, 256, 320)]
    shapes += [(256, 384, 256), (128, 128, 256)]
    rng = np.random.default_rng(23)
    worst = 0.0
    count = 0
    rejected = 0
    for cfg, path in zip(configs, paths):
        runner = GpuGemmRunner(path, arch=gfx_arch)
        if runner.kernel_name != cfg.name:
            return FAIL, f"registered {runner.kernel_name!r}, expected {cfg.name!r}"
        for M, N, K in shapes:
            if K % cfg.tile_k:  # unpadded kernels need K to be a tile_k multiple
                continue
            # Generate fresh B even for a repeated shape, detecting stale packing.
            A = rng.uniform(-1.0, 1.0, (M, K)).astype(np.float32)
            B = rng.uniform(-1.0, 1.0, (K, N)).astype(np.float32)
            if dtype == "bf16":
                A = _bf16_u16_to_fp32(_fp32_to_bf16_u16(A))
                B = _bf16_u16_to_fp32(_fp32_to_bf16_u16(B))
            else:
                A = A.astype(np.float16).astype(np.float32)
                B = B.astype(np.float16).astype(np.float32)
            result = runner.run(A, B, GemmProblem(M=M, N=N, K=K))
            label = f"{cfg.name}, MNK={M}/{N}/{K}"
            if result.status != 0:
                return FAIL, f"{label}: status={result.status}"
            if result.output.shape != (M, N) or not np.isfinite(result.output).all():
                return FAIL, f"{label}: invalid output shape or NaN/Inf"
            error = _max_rel_err(result.output, A @ B)
            if error > TOLERANCE:
                return FAIL, f"{label}: max_rel_err={error:.4e} > {TOLERANCE:.1e}"
            if result.time_ms <= 0.0:
                return FAIL, f"{label}: nonpositive time {result.time_ms}"
            worst = max(worst, error)
            count += 1
        if cfg.pipeline in ("comp_tdm", "comp_tdm_v2"):
            for M, N, K in ((128, 160, 128), (128, 128, 96)):
                result = runner.run(
                    np.zeros((M, K), dtype=np.float32),
                    np.zeros((K, N), dtype=np.float32),
                    GemmProblem(M=M, N=N, K=K),
                )
                if result.status != -2:  # No supported kernel, before launch.
                    return FAIL, f"{cfg.name}: accepted a partial N/K tile"
                rejected += 1
        runner.lib.cleanup()
    return PASS, (f"preshuffle/{dtype}: {count} cases, {rejected} rejected shapes, "
                  f"max_rel_err={worst:.4e}")


def test_preshuffle_gpu(gpu_arch, dispatcher_static_lib) -> None:
    """Build the prerequisite and verify the result whenever a GPU is available."""
    for dtype in ("fp16", "bf16"):
        status, detail = _run_preshuffle(gpu_arch, dtype)
        assert status == PASS, detail


def main() -> int:
    parser = argparse.ArgumentParser(description="Preshuffle GEMM GPU correctness test")
    parser.add_argument("--gfx", default=None, help="GPU arch (default: auto-detect)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not _has_gpu():
        print("SKIP: no supported GPU detected (rocminfo); preshuffle GPU test skipped")
        return SKIP_EXIT

    gfx = args.gfx or _resolve_arch(None)
    if gfx not in _SUPPORTED_ARCHS:
        print(f"SKIP: preshuffle GEMM needs one of "
              f"{'/'.join(_SUPPORTED_ARCHS)}; got {gfx}")
        return SKIP_EXIT
    log.info("Running preshuffle GEMM GPU correctness on %s", gfx)

    # Build the prerequisite the pytest entry point gets from the
    # dispatcher_static_lib fixture; ctest runs this script directly.
    from dispatcher_build import ensure_dispatcher_static_lib

    try:
        ensure_dispatcher_static_lib()
    except RuntimeError as exc:
        print(f"FAIL: could not build the dispatcher static lib: {exc}")
        return 1

    results = []
    for dtype in ("fp16", "bf16"):
        try:
            status, detail = _run_preshuffle(gfx, dtype)
        except FileNotFoundError as exc:
            print(f"SKIP: dispatcher not built ({exc})")
            return SKIP_EXIT
        except Exception as exc:
            status, detail = FAIL, f"preshuffle/{dtype}: exception: {exc}"
        results.append(status)
        print(f"[{status}] {detail}", flush=True)
    return 0 if all(status == PASS for status in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
