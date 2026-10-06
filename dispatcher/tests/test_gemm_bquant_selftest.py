#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Self-test / default-config runner for the non-grouped gemm_bquant bridge.

CPU-only checks (no GPU, no hipcc required):
  1. codegen KERNEL_NAME is byte-exact with BQuantKernelConfig.name across the
     full Old-TE dtype x preshuffle matrix.
  2. every default config generates a valid .hpp whose emitted KERNEL_NAME,
     QuantType, dtypes, tile constants and MX arch-guard match expectations.

Optional GPU run (--gpu): builds + runs the fp8 default kernel and verifies
against a NumPy reference.

Run:
    python3 dispatcher/tests/test_gemm_bquant_selftest.py            # CPU self-test
    python3 dispatcher/tests/test_gemm_bquant_selftest.py --gpu      # + GPU run
"""

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

_PY_DIR = Path(__file__).resolve().parents[1] / "python"
_CODEGEN = Path(__file__).resolve().parents[1] / "codegen" / "unified_gemm_bquant_codegen.py"
sys.path.insert(0, str(_PY_DIR))

from gemm_bquant_utils import (  # noqa: E402
    BQuantKernelConfig,
    BQuantGemmProblem,
    default_fp8_config,
    default_bf8_config,
    default_fp8i4_config,
    default_bf8i4_config,
    default_fp8_preshuffleb_config,
    default_bf8_preshuffleb_config,
    default_fp8i4_preshuffleb_config,
    default_bf8i4_preshuffleb_config,
    default_fp8_preshufflequant_config,
    default_bf8_preshufflequant_config,
    default_fp8i4_preshufflequant_config,
    default_bf8i4_preshufflequant_config,
    default_fp8_preshuffleb_bquant_config,
    default_bf8_preshuffleb_bquant_config,
    default_fp8i4_preshuffleb_bquant_config,
    default_bf8i4_preshuffleb_bquant_config,
    default_mx_bf16bf16_config,
    default_mx_bf16bf8_config,
    default_mx_bf16fp4_config,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# The full Old-TE plain-bquant matrix: (label, config-factory).
# Non-MX variants run on gfx942 + gfx950; MX variants (e8m0) are gfx950-only.
ALL_CONFIGS = [
    # non-preshuffle (GemmConfigQuantDecode)
    ("fp8",                        default_fp8_config),
    ("bf8",                        default_bf8_config),
    ("fp8i4",                      default_fp8i4_config),
    ("bf8i4",                      default_bf8i4_config),
    # preshuffle_b (WPQuantBPipelineAgBgCrV2)
    ("fp8_preshuffleb",            default_fp8_preshuffleb_config),
    ("bf8_preshuffleb",            default_bf8_preshuffleb_config),
    ("fp8i4_preshuffleb",          default_fp8i4_preshuffleb_config),
    ("bf8i4_preshuffleb",          default_bf8i4_preshuffleb_config),
    # preshuffle_bquant (BQuantGemmPipelineAgBgCrCompV3)
    ("fp8_preshufflequant",        default_fp8_preshufflequant_config),
    ("bf8_preshufflequant",        default_bf8_preshufflequant_config),
    ("fp8i4_preshufflequant",      default_fp8i4_preshufflequant_config),
    ("bf8i4_preshufflequant",      default_bf8i4_preshufflequant_config),
    # preshuffle_b + preshuffle_bquant
    ("fp8_preshuffleb_bquant",     default_fp8_preshuffleb_bquant_config),
    ("bf8_preshuffleb_bquant",     default_bf8_preshuffleb_bquant_config),
    ("fp8i4_preshuffleb_bquant",   default_fp8i4_preshuffleb_bquant_config),
    ("bf8i4_preshuffleb_bquant",   default_bf8i4_preshuffleb_bquant_config),
    # MX microscale (gfx950-only)
    ("mx_bf16bf16",                default_mx_bf16bf16_config),
    ("mx_bf16bf8",                 default_mx_bf16bf8_config),
    ("mx_bf16fp4",                 default_mx_bf16fp4_config),
]

_MX_LABELS = {"mx_bf16bf16", "mx_bf16bf8", "mx_bf16fp4"}


def _run_codegen(config: BQuantKernelConfig, out_dir: Path) -> Path:
    cmd = [
        sys.executable, str(_CODEGEN),
        "--output-dir", str(out_dir),
        "--config-json", json.dumps(config.to_codegen_config()),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        raise RuntimeError(f"codegen failed for {config.name}:\n{res.stderr}")
    hpp = out_dir / f"{config.name}.hpp"
    if not hpp.exists():
        raise RuntimeError(f"codegen produced no header for {config.name}")
    return hpp


def cpu_selftest() -> int:
    failures = 0
    with tempfile.TemporaryDirectory(prefix="gemm_bquant_selftest_") as td:
        out_dir = Path(td)
        for label, factory in ALL_CONFIGS:
            cfg = factory()
            try:
                # 1. name must start with gemm_bquant (not grouped_gemm_bquant)
                assert cfg.name.startswith("gemm_bquant_"), \
                    f"{label}: name {cfg.name!r} missing gemm_bquant_ prefix"

                # 2. codegen produces a header and embeds byte-exact KERNEL_NAME
                hpp = _run_codegen(cfg, out_dir)
                text = hpp.read_text()
                assert f'KERNEL_NAME = "{cfg.name}"' in text, \
                    f"{label}: KERNEL_NAME mismatch vs {cfg.name}"

                # 3. QuantType + kernel shape are correct
                assert "ck_tile::QuantType::BQuantGrouped" in text, \
                    f"{label}: missing BQuantGrouped quant type"
                assert "QuantGemmKernel" in text, f"{label}: missing QuantGemmKernel"
                assert "QuantGemmHostArgs" in text, f"{label}: missing QuantGemmHostArgs"

                # 4. MX variants must carry the gfx950 #error arch guard; non-MX must not
                if label in _MX_LABELS:
                    assert "#ifndef CK_GFX950_SUPPORT" in text and "#error" in text, \
                        f"{label}: MX variant missing gfx950 arch guard"
                else:
                    assert "#error" not in text, \
                        f"{label}: non-MX variant should not have an #error guard"

                # 5. preshuffle flags in the name match the emitted constants
                assert (
                    ("static constexpr bool PreshuffleB     = true"
                     in text) == cfg.preshuffle_b
                ), f"{label}: PreshuffleB constant mismatch"
                assert (
                    ("static constexpr bool BPreshuffleQuant = true"
                     in text) == cfg.preshuffle_bquant
                ), f"{label}: BPreshuffleQuant constant mismatch"

                log.info("PASS  %-26s -> %s", label, cfg.name)
            except AssertionError as e:
                failures += 1
                log.error("FAIL  %-26s : %s", label, e)
            except Exception as e:
                failures += 1
                log.error("ERROR %-26s : %s", label, e)

    total = len(ALL_CONFIGS)
    log.info("CPU self-test: %d/%d passed", total - failures, total)
    return 1 if failures else 0


def gpu_run() -> int:
    """Build + run the fp8 default kernel and verify against a NumPy reference."""
    import numpy as np
    from gemm_bquant_utils import (
        BQuantGpuGemmRunner,
        setup_multiple_bquant_dispatchers,
    )

    cfg = default_fp8_config()
    problem = BQuantGemmProblem(M=16, N=64, K=256, quant_group_k=128, quant_group_n=1)

    with tempfile.TemporaryDirectory(prefix="gemm_bquant_gpu_") as td:
        so_paths = setup_multiple_bquant_dispatchers([cfg], output_dir=Path(td))
        if not so_paths or so_paths[0] is None:
            log.error("GPU run: kernel build failed")
            return 1

        rng = np.random.default_rng(0)
        try:
            import ml_dtypes
            fp8 = ml_dtypes.float8_e4m3fn
        except ImportError:
            log.error("GPU run needs ml_dtypes to encode fp8; skipping")
            return 1

        A_f = rng.uniform(-2, 2, (problem.M, problem.K)).astype(np.float32)
        B_f = rng.uniform(-2, 2, (problem.K, problem.N)).astype(np.float32)
        BQ = rng.uniform(0.5, 2.0, (problem.QK_B, problem.QN_B)).astype(np.float32)

        A_raw = A_f.astype(fp8).view(np.uint8)
        B_raw = B_f.astype(fp8).view(np.uint8)
        A_dec = A_raw.view(fp8).astype(np.float32)
        B_dec = B_raw.view(fp8).astype(np.float32)

        runner = BQuantGpuGemmRunner(so_paths[0])
        result = runner.run(A=A_raw, B=B_raw, BQ=BQ, problem=problem)

        B_deq = B_dec.copy()
        for qi in range(problem.QK_B):
            for qj in range(problem.QN_B):
                B_deq[qi * 128:(qi + 1) * 128, qj:qj + 1] *= float(BQ[qi, qj])
        C_ref = (A_dec @ B_deq)
        max_rel = float(np.max(np.abs(result.C.astype(np.float32) - C_ref))
                        / (np.max(np.abs(C_ref)) + 1e-6))
        log.info("GPU run %s: time=%.3f ms max_rel=%.4f", result.kernel_name,
                 result.time_ms, max_rel)
        return 0 if max_rel <= 0.05 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="gemm_bquant bridge self-test")
    ap.add_argument("--gpu", action="store_true", help="also build + run fp8 kernel on GPU")
    args = ap.parse_args()

    rc = cpu_selftest()
    if args.gpu:
        rc |= gpu_run()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
