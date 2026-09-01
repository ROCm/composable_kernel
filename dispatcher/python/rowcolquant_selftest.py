#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
RowColQuant bridge self-test / default-config runner.

Two modes:

  --build-only   codegen + hipcc compile of the default fp8/bf8 configs, no GPU
                 required. Verifies the whole toolchain up to a loadable .so.

  (default)      build + run on a GPU: generates random A/B and per-row/per-col
                 scales, runs each kernel, and -- when ml_dtypes is available --
                 performs a GENUINE numeric check: A/B are encoded to real
                 fp8/bf8 bytes fed to the kernel, and the numpy reference is
                 rounded through the SAME fp8/bf8 quantization so the comparison
                 is apples-to-apples. If ml_dtypes is unavailable the run is
                 SMOKE-ONLY (kernel launches, timing reported, NO correctness
                 claim) -- genuine numeric verify is then left to the tester.

Reference math (RowColQuant):
    C[m, n] = sum_k ( A[m, k] * AQ[m] ) * ( B[k, n] * BQ[n] )
            = AQ[m] * BQ[n] * sum_k A[m, k] * B[k, n]

where A[m, k] and B[k, n] are the fp8/bf8-rounded values (quantize->dequantize),
matching exactly what the kernel dequantizes on device.

Usage:
    python3 rowcolquant_selftest.py --build-only
    python3 rowcolquant_selftest.py --arch gfx950 --m 256 --n 256 --k 512
"""

import argparse
import logging
import sys
from pathlib import Path

import gemm_rowcolquant_utils as u

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("rowcolquant_selftest")


def _reference(A, B, AQ, BQ):
    import numpy as np

    acc = A.astype(np.float32) @ B.astype(np.float32)          # [M, N]
    acc = acc * AQ.astype(np.float32)[:, None]                 # per-row scale
    acc = acc * BQ.astype(np.float32)[None, :]                 # per-col scale
    return acc


def _run_one(so_path, variant_key, M, N, K, verify, gfx_arch=None):
    import numpy as np

    runner = u.RowColQuantGpuGemmRunner(so_path)
    name = runner.kernel_name

    # Genuine numeric verify requires ml_dtypes to encode real fp8/bf8 bytes and
    # to round the reference identically. Without it we can only smoke-test.
    can_verify = verify and u.fp8_encoding_available()
    if verify and not can_verify:
        log.warning(
            "  ml_dtypes unavailable: running SMOKE-ONLY (no correctness "
            "claim). Install ml_dtypes for a genuine fp8/bf8 numeric check."
        )

    rng = np.random.default_rng(0)
    A_f = rng.uniform(-2.0, 2.0, size=(M, K)).astype(np.float32)
    B_f = rng.uniform(-2.0, 2.0, size=(K, N)).astype(np.float32)
    AQ = rng.uniform(0.5, 1.5, size=(M,)).astype(np.float32)
    BQ = rng.uniform(0.5, 1.5, size=(N,)).astype(np.float32)

    if can_verify:
        # Feed the kernel REAL 1-byte-per-element fp8/bf8 bytes (the ctypes lib
        # reads A/B as const fp8_t*/bf8_t*). Encoding to uint8 preserves shape
        # and the exact bit pattern the device dequantizes. The fp8 flavour
        # (OCP on gfx950, FNUZ on gfx942) MUST match the kernel's arch or the
        # reference silently NaNs.
        A = u.encode_fp8_bytes(A_f, variant_key, gfx_arch)
        B = u.encode_fp8_bytes(B_f, variant_key, gfx_arch)
        # Round the reference inputs through the SAME quantization.
        A_ref = u.quantize_dequantize_fp8(A_f, variant_key, gfx_arch)
        B_ref = u.quantize_dequantize_fp8(B_f, variant_key, gfx_arch)
    else:
        # SMOKE-ONLY: ml_dtypes is unavailable, so we cannot build a valid fp8
        # encoding. Feed correctly-sized 1-byte buffers (random bytes) so the
        # kernel launches and run()'s dtype guard is satisfied; the device reads
        # these as fp8_t/bf8_t, so we make NO numeric claim in this path.
        A = rng.integers(0, 256, size=(M, K), dtype=np.uint8)
        B = rng.integers(0, 256, size=(K, N), dtype=np.uint8)

    result = runner.run(A, B, AQ, BQ, u.RowColQuantGemmProblem(M=M, N=N, K=K))
    log.info("kernel %s ran in %.4f ms", name, result.time_ms)

    if can_verify:
        ref = _reference(A_ref, B_ref, AQ, BQ)
        got = result.C.astype(np.float32)
        denom = np.maximum(np.abs(ref), 1.0)
        rel = np.abs(got - ref) / denom
        max_rel = float(rel.max())
        log.info("  max relative error vs fp8/bf8-rounded reference: %.4g", max_rel)
        # fp8 e4m3 has ~2 decimal digits of precision; accumulate over K terms.
        tol = 0.15
        if max_rel > tol:
            log.error("  VERIFY FAILED: max_rel %.4g > tol %.4g", max_rel, tol)
            return False
        log.info("  VERIFY PASSED (max_rel %.4g <= tol %.4g)", max_rel, tol)
    return True


def main() -> int:
    p = argparse.ArgumentParser(description="RowColQuant bridge self-test")
    p.add_argument("--build-only", action="store_true",
                   help="Only codegen + compile; do not launch on a GPU")
    p.add_argument("--arch", default=None, help="Target GFX arch (default: autodetect)")
    p.add_argument("--m", type=int, default=256)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--k", type=int, default=512)
    p.add_argument("--no-verify", action="store_true",
                   help="Skip numpy reference verification")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    # Resolve the target arch ONCE so the configs, the tile selection
    # (warp_tile_k), and the fp8 encoding flavour (OCP vs FNUZ) all agree. If
    # --arch was not given, autodetect the local GPU (raises if none), because
    # both the tile shape and the fp8 encoding are arch-specific: building the
    # gfx950 16x16x128 tile on gfx942 outputs all-zeros, and OCP-vs-FNUZ
    # mismatch NaNs the reference.
    arch = args.arch or u._detect_gpu_arch()
    log.info("Target arch: %s", arch)

    configs = [u.default_fp8_config(arch), u.default_bf8_config(arch)]
    log.info("Configs: %s", [c.name for c in configs])

    so_paths = u.setup_multiple_rowcolquant_dispatchers(
        configs,
        output_dir=args.output_dir,
        gfx_arch=arch,
    )

    ok = True
    for cfg, so in zip(configs, so_paths):
        if so is None:
            log.error("BUILD FAILED: %s", cfg.name)
            ok = False
        else:
            log.info("BUILT: %s -> %s", cfg.name, so)

    if not ok:
        return 1
    if args.build_only:
        log.info("build-only: all %d kernels built", len(configs))
        return 0

    for cfg, so in zip(configs, so_paths):
        try:
            if not _run_one(so, cfg.variant_key, args.m, args.n, args.k,
                            verify=not args.no_verify, gfx_arch=arch):
                ok = False
        except Exception as e:
            log.error("RUN FAILED for %s: %s", so, e)
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
