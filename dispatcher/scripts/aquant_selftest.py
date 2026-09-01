#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
AQuant (A-only quantized) GEMM bridge self-test / default-config runner.

No GPU required for the default (--codegen-only) mode:
  1. Verify AQuantKernelConfig.name (utils) == codegen KERNEL_NAME (byte-exact).
  2. Generate every default-config kernel header and assert the emitted
     `constexpr const char* KERNEL_NAME` matches the config .name.
  3. Print the dtype x layout coverage matrix.

With --build (GPU/hipcc host required) it additionally compiles each default
config into a .so via the Python bridge and reports pass/fail per kernel.

Usage:
  python3 aquant_selftest.py                 # codegen + name-parity only
  python3 aquant_selftest.py --build         # also hipcc-compile each .so
  python3 aquant_selftest.py --arch gfx942   # target arch for --build
  python3 aquant_selftest.py --preshufflequant
"""

import argparse
import re
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "python"))
sys.path.insert(0, str(_HERE.parent / "codegen"))

from gemm_aquant_utils import (  # noqa: E402
    AQuantKernelConfig,
    default_fp8_config,
    default_bf8_config,
    default_fp8i4_config,
    default_bf8i4_config,
    default_fp8_preshufflequant_config,
    default_bf8_preshufflequant_config,
    default_fp8i4_preshufflequant_config,
    default_bf8i4_preshufflequant_config,
    _generate_aquant_kernel,
    setup_multiple_aquant_dispatchers,
    _LAYOUTS_DECODE,
    _LAYOUTS_PRESHUFFLEQUANT,
)

_VARIANTS = ["fp8", "bf8", "fp8i4", "bf8i4"]


def _decode_configs(arch: str):
    fns = {
        "fp8": default_fp8_config,
        "bf8": default_bf8_config,
        "fp8i4": default_fp8i4_config,
        "bf8i4": default_bf8i4_config,
    }
    return [fns[v](layout=lay, gfx_arch=arch)
            for v in _VARIANTS for lay in _LAYOUTS_DECODE]


def _preshufflequant_configs(arch: str):
    fns = {
        "fp8": default_fp8_preshufflequant_config,
        "bf8": default_bf8_preshufflequant_config,
        "fp8i4": default_fp8i4_preshufflequant_config,
        "bf8i4": default_bf8i4_preshufflequant_config,
    }
    return [fns[v](layout=lay, gfx_arch=arch)
            for v in _VARIANTS for lay in _LAYOUTS_PRESHUFFLEQUANT]


_KERNEL_NAME_RE = re.compile(r'constexpr const char\* KERNEL_NAME = "([^"]+)"')


def _check_name_parity(configs, out_dir: Path) -> int:
    failures = 0
    for cfg in configs:
        hpp = _generate_aquant_kernel(cfg, out_dir)
        if hpp is None:
            print(f"  [FAIL] codegen produced no header for {cfg.name}")
            failures += 1
            continue
        text = hpp.read_text()
        m = _KERNEL_NAME_RE.search(text)
        emitted = m.group(1) if m else "<none>"
        if emitted != cfg.name:
            print(f"  [FAIL] name mismatch:\n"
                  f"         utils : {cfg.name}\n"
                  f"         header: {emitted}")
            failures += 1
        else:
            print(f"  [ok]   {cfg.name}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description="AQuant GEMM bridge self-test")
    ap.add_argument("--build", action="store_true",
                    help="Also hipcc-compile each default config into a .so")
    ap.add_argument("--arch", default="gfx950", help="Target GFX arch (default gfx950)")
    ap.add_argument("--preshufflequant", action="store_true",
                    help="Test the preshufflequant configs instead of decode")
    args = ap.parse_args()

    configs = (_preshufflequant_configs(args.arch) if args.preshufflequant
               else _decode_configs(args.arch))

    mode = "preshufflequant" if args.preshufflequant else "decode"
    print(f"=== AQuant {mode} self-test (arch={args.arch}) ===")
    print(f"dtype x layout coverage: {len(configs)} kernels")
    print()

    with tempfile.TemporaryDirectory(prefix="aquant_selftest_") as td:
        out_dir = Path(td)
        print("--- name parity (codegen KERNEL_NAME vs utils .name) ---")
        failures = _check_name_parity(configs, out_dir)

        print()
        if failures:
            print(f"NAME PARITY FAILED: {failures} mismatch(es)")
            return 1
        print(f"NAME PARITY OK: {len(configs)} kernels byte-exact")

        if args.build:
            print()
            print("--- hipcc build ---")
            so_paths = setup_multiple_aquant_dispatchers(
                configs, output_dir=out_dir, gfx_arch=args.arch,
            )
            built = sum(1 for p in so_paths if p is not None)
            for cfg, p in zip(configs, so_paths):
                status = "ok" if p is not None else "FAIL"
                print(f"  [{status}] {cfg.name}")
            print()
            print(f"BUILD: {built}/{len(configs)} kernels compiled")
            if built != len(configs):
                return 2

    print()
    print("SELF-TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
