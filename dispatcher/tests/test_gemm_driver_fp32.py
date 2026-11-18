#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
fp32 through the tile_engine GEMM sweep driver (gemm_full_benchmark.py):
fp32 is a --dtype choice with its own CI config and a tight verify tolerance,
an arch without fp32 warp tiles is rejected by name, a sweep that expands to 0
configs is an error, and GpuGemmRunner hands fp32 kernels fp32 host buffers
(raising on an unknown dtype instead of falling back to fp16).

No GPU needed. Run: python3 -m pytest -q tests/test_gemm_driver_fp32.py
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

DISPATCHER_DIR = Path(__file__).resolve().parent.parent
GEMM_TE_DIR = DISPATCHER_DIR.parent / "tile_engine" / "ops" / "gemm"
sys.path.insert(0, str(DISPATCHER_DIR / "python"))
sys.path.insert(0, str(GEMM_TE_DIR))
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))

import codegen_common as cc  # noqa: E402
from arch_filter import ArchFilter, OperatorType  # noqa: E402
from arch_filter import KernelConfig as FilterConfig  # noqa: E402
import gemm_full_benchmark as drv  # noqa: E402
from ctypes_utils import (  # noqa: E402
    KernelConfig,
    listed_warp_tiles,
    validate_kernel_config,
)
from gemm_utils import GemmProblem, GpuGemmRunner, expand_sweep  # noqa: E402

CFG_DIR = GEMM_TE_DIR / "configs"
FP32_CI = CFG_DIR / "default_ci_config_fp32.json"
GFX1250_FP32_CI = CFG_DIR / "default_ci_config_gfx1250_fp32.json"
DEFAULT_CI = CFG_DIR / "default_ci_config.json"


# ---------------------------------------------------------------- driver data


def test_fp32_is_a_dtype_choice_but_not_for_preshuffle():
    assert "fp32" in drv.SUPPORTED_DTYPES
    assert "fp32" not in drv.VARIANT_SUPPORTED_DTYPES["gemm_preshuffle"]


@pytest.mark.parametrize("variant", ["gemm_multi_d", "gemm_multi_abd"])
def test_fp16_only_variants_reject_fp32(monkeypatch, capsys, variant):
    rc, out, built = _main(monkeypatch, capsys, "--variant", variant, "--dtype", "fp32")
    assert rc == 1 and not built
    assert f"variant {variant} supports dtypes ('fp16',)" in out


@pytest.mark.parametrize(
    "arch, dtype, expected",
    [
        ("gfx950", "fp32", FP32_CI),
        ("gfx950", "fp16", DEFAULT_CI),
        ("gfx950", "bf8", DEFAULT_CI),
        # Most specific first: <arch>_<dtype>, then <arch>, then <dtype>.
        ("gfx1250", "fp32", GFX1250_FP32_CI),
        ("gfx1250", "fp16", CFG_DIR / "default_ci_config_gfx1250.json"),
        ("gfx1250", "fp8", CFG_DIR / "default_ci_config_gfx1250_fp8.json"),
        # bf8 has no file of its own and shares fp8's warp tiles.
        ("gfx1250", "bf8", CFG_DIR / "default_ci_config_gfx1250_fp8.json"),
    ],
)
def test_ci_config_selected_by_arch_and_dtype(arch, dtype, expected):
    args = SimpleNamespace(configs=[], variant="gemm_universal", dtype=dtype)
    assert drv.resolve_configs(args, arch) == [str(expected)]


@pytest.mark.parametrize("dtype", drv.SUPPORTED_DTYPES)
def test_gfx1250_default_ci_config_expands_for_every_dtype(dtype):
    args = SimpleNamespace(configs=[], variant="gemm_universal", dtype=dtype)
    (path,) = drv.resolve_configs(args, "gfx1250")
    assert expand_sweep(path, "gfx1250", dtype=dtype, layout="rcr")


def test_explicit_configs_win_over_dtype_ci_config():
    args = SimpleNamespace(configs=["x.json"], variant="gemm_universal", dtype="fp32")
    assert drv.resolve_configs(args, "gfx1250") == ["x.json"]


def test_fp32_verify_tol_catches_an_fp16_downcast():
    tol = drv.VERIFY_TOL["fp32"]
    # Tighter than one fp16 ulp at 1.0, so fp16-precision output fails.
    assert 0 < tol < float(np.finfo(np.float16).eps)
    assert drv.VERIFY_TOL.get("fp16", drv.DEFAULT_VERIFY_TOL) == 2e-2


# --------------------------------------------------------- arch warp tiles


def test_listed_warp_tiles_matches_validator():
    tiles = listed_warp_tiles("gfx950", "fp32")[2]
    assert [16, 16, 4] in tiles and [32, 32, 8] in tiles
    assert listed_warp_tiles("gfx1250", "fp32")[2] == [[16, 16, 4]]
    # Feature suffixes are stripped, as in the codegen.
    assert listed_warp_tiles("gfx1250:xnack-", "fp32")[2] == [[16, 16, 4]]
    assert listed_warp_tiles("gfx1201", "fp32")[2] == []
    # Accumulator defaults to fp32 (int32 for int8), as the codegen keys it.
    assert listed_warp_tiles("gfx950", "fp16")[0] == "fp16_fp16_fp32"
    assert listed_warp_tiles("gfx950", "int8")[0] == "int8_int8_int32"
    # Same key/table the validator reports when nothing is listed.
    cfg = KernelConfig(
        dtype_a="fp32", dtype_b="fp32", dtype_c="fp32", gfx_arch="gfx1201"
    )
    errs = validate_kernel_config(cfg).errors
    assert any("fp32_fp32_fp32 on gfx1201 in warp_tile_combos" in e for e in errs)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1250"])
def test_fp32_ci_config_expands_to_exactly_the_listed_warp_tiles(arch):
    cfgs = expand_sweep(str(FP32_CI), arch, dtype="fp32", layout="rcr")
    got = {(c.warp_tile_m, c.warp_tile_n, c.warp_tile_k) for c in cfgs}
    assert got == {tuple(t) for t in listed_warp_tiles(arch, "fp32")[2]}


def test_gfx1250_fp32_ci_config_adds_a_tdm_kernel_and_keeps_the_gates():
    cfgs = expand_sweep(str(GFX1250_FP32_CI), "gfx1250", dtype="fp32", layout="rcr")
    assert {(c.warp_tile_m, c.warp_tile_n, c.warp_tile_k) for c in cfgs} == {
        (16, 16, 4)
    }
    tdm = [c for c in cfgs if c.pipeline == "comp_tdm_v2"]
    # comp_tdm_v2 survives only as tdm epilogue + intrawave + non-persistent.
    assert [(c.epilogue, c.scheduler, c.persistent) for c in tdm] == [
        ("tdm", "intrawave", False)
    ]
    assert all(c.epilogue != "tdm" for c in cfgs if c.pipeline != "comp_tdm_v2")
    assert all(c.wave_m * c.wave_n * c.wave_k == 4 for c in cfgs)


# ------------------------------------------------------ gfx1250 fp32 gates


@pytest.mark.parametrize("layout", ["rcr", "rrr", "crr", "ccr"])
@pytest.mark.parametrize("dtype", ["fp32", "fp16"])
def test_gfx1250_fp32_tdm_is_rcr_only(layout, dtype):
    reason = cc.gfx1250_pipeline_reject_reason(
        "gfx1250",
        "comp_tdm_v2",
        "tdm",
        "intrawave",
        num_waves=4,
        warp_tile_k=4,
        dtype_a=dtype,
        dtype_b=dtype,
        layout=layout,
    )
    expect_reject = dtype == "fp32" and layout != "rcr"
    assert (reason == cc.GFX1250_TDM_FP32_LAYOUT_REJECT_REASON) == expect_reject
    assert expect_reject or reason == ""


@pytest.mark.parametrize(
    "arch, dtype, tile_m, tile_n, num_waves, rejected",
    [
        ("gfx1250", "fp32", 256, 256, 4, True),  # 512 acc/lane: spills
        ("gfx1250", "fp32", 256, 128, 4, False),
        ("gfx1250", "fp32", 128, 256, 4, False),
        ("gfx1250", "fp32", 256, 256, 8, False),
        ("gfx1250", "fp16", 256, 256, 4, False),
        ("gfx950", "fp32", 256, 256, 4, False),
    ],
)
def test_gfx1250_fp32_rejects_only_the_spilling_tile(
    arch, dtype, tile_m, tile_n, num_waves, rejected
):
    reason = cc.gfx1250_fp32_tile_reject_reason(arch, dtype, tile_m, tile_n, num_waves)
    assert bool(reason) == rejected


@pytest.mark.parametrize(
    "arch, dtype, tile_n, operator, rejected",
    [
        ("gfx1250", "fp32", 256, OperatorType.GEMM, True),
        ("gfx1250", "fp32", 256, OperatorType.GEMM_STREAMK, True),
        ("gfx1250", "fp32", 128, OperatorType.GEMM, False),
        ("gfx1250", "fp32", 256, OperatorType.CONV_FWD, False),
        ("gfx1250", "fp16", 256, OperatorType.GEMM, False),
        ("gfx950", "fp32", 256, OperatorType.GEMM, False),
    ],
)
def test_arch_filter_applies_the_gfx1250_fp32_tile_gate(
    arch, dtype, tile_n, operator, rejected
):
    # Same rule as the codegen and expand_sweep, for GEMM operators only.
    config = FilterConfig(
        datatype_a=dtype,
        datatype_b=dtype,
        datatype_c=dtype,
        tile_m=256,
        tile_n=tile_n,
        tile_k=32,
        warp_m=2,
        warp_n=2,
        warp_k=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=4,
        pipeline="compv3",
        operator=operator,
    )
    errors = ArchFilter(arch).validate_kernel(config).errors
    assert (cc.GFX1250_FP32_ACC_REJECT_REASON in errors) == rejected


def test_gfx1250_fp32_sweep_applies_both_gates(tmp_path):
    cfg = json.loads(GFX1250_FP32_CI.read_text())
    for dim in ("tile_m", "tile_n"):
        cfg["tile_config"][dim] = {"min": 128, "max": 256, "step": 128}
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg))

    def sweep(layout):
        return expand_sweep(str(path), "gfx1250", dtype="fp32", layout=layout)

    rcr, rrr = sweep("rcr"), sweep("rrr")
    tiles = {(c.tile_m, c.tile_n) for c in rcr + rrr}
    assert tiles == {(128, 128), (128, 256), (256, 128)}
    assert any(c.pipeline == "comp_tdm_v2" for c in rcr)
    assert rrr and not any(c.pipeline == "comp_tdm_v2" for c in rrr)


# ------------------------------------------------------------ driver main()


def _main(monkeypatch, capsys, *argv, arch="gfx950"):
    monkeypatch.setattr(sys, "argv", ["gemm_full_benchmark.py", *argv])
    monkeypatch.setattr(drv, "resolve_devices", lambda spec: ["0"])
    monkeypatch.setattr(drv, "_resolve_arch", lambda a: arch)
    built = []
    monkeypatch.setattr(
        drv,
        "setup_multiple_gemm_dispatchers",
        lambda cfgs, **kw: built.append(cfgs) or [None] * len(cfgs),
    )
    rc = drv.main()
    return rc, capsys.readouterr().out, built


def test_arch_without_fp32_warp_tiles_is_rejected_by_name(monkeypatch, capsys):
    rc, out, built = _main(monkeypatch, capsys, "--dtype", "fp32", arch="gfx1201")
    assert rc == 1 and not built
    assert "dtype fp32 is not supported on gfx1201" in out


def test_zero_expanded_configs_is_an_error(monkeypatch, capsys):
    # The fp16 CI config only has 32x32x16 warp tiles, which fp32 does not list.
    rc, out, built = _main(monkeypatch, capsys, "--dtype", "fp32", str(DEFAULT_CI))
    assert rc == 1 and not built
    assert "0 configs expanded for dtype fp32 on gfx950" in out


def test_zero_configs_after_the_gfx1250_gates_names_the_gates(
    monkeypatch, capsys, tmp_path
):
    # 16x16x4 is listed for fp32; only the TDM layout gate rejects these.
    cfg = json.loads(GFX1250_FP32_CI.read_text())
    cfg["trait_config"]["pipeline"]["values"] = ["comp_tdm_v2"]
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg))
    argv = ("--dtype", "fp32", "--layout", "rrr", str(path))
    rc, out, built = _main(monkeypatch, capsys, *argv, arch="gfx1250")
    assert rc == 1 and not built
    assert "0 configs expanded for dtype fp32 on gfx1250" in out
    assert "tile-size gates" in out


def test_fp32_default_run_reaches_the_build(monkeypatch, capsys):
    rc, out, built = _main(monkeypatch, capsys, "--dtype", "fp32")
    assert len(built) == 1 and len(built[0]) > 0
    assert all(c.dtype_a == "fp32" for c in built[0])
    assert str(FP32_CI) in out
    assert rc == 1  # the stubbed build returns no .so


def test_gfx1250_fp32_default_run_uses_the_arch_config(monkeypatch, capsys):
    _, out, built = _main(monkeypatch, capsys, "--dtype", "fp32", arch="gfx1250")
    assert str(GFX1250_FP32_CI) in out
    assert any(c.pipeline == "comp_tdm_v2" for c in built[0])


# ---------------------------------------------------------- GpuGemmRunner


class _FakeLib:
    def __init__(self):
        self.bufs = None

    def run(self, A_h, B_h, C_h, M, N, K):
        self.bufs = (A_h, B_h, C_h)
        C_h[...] = 1
        return 0, 1.0


def _runner(kernel_name):
    r = GpuGemmRunner.__new__(GpuGemmRunner)
    r.lib, r._kernel_name, r._use_ocp = _FakeLib(), kernel_name, None
    return r


@pytest.mark.parametrize(
    "dtype, np_in, np_out",
    [("fp32", np.float32, np.float32), ("fp16", np.float16, np.float16)],
)
def test_runner_host_buffers_match_kernel_dtype(dtype, np_in, np_out):
    r = _runner(f"gemm_{dtype}_rcr_compv3_cshuffle_intrawave")
    A = np.full((4, 8), 1.0 + 2.0**-20, dtype=np.float32)
    B = np.ones((8, 4), dtype=np.float32)
    res = r.run(A, B, GemmProblem(M=4, N=4, K=8))
    A_h, B_h, C_h = r.lib.bufs
    assert (A_h.dtype, B_h.dtype, C_h.dtype) == (np_in, np_in, np_out)
    if dtype == "fp32":
        # fp32 values below fp16 resolution must reach the kernel unrounded.
        np.testing.assert_array_equal(A_h, A)
    assert res.output.shape == (4, 4)


def test_runner_rejects_unknown_dtype():
    r = _runner("gemm_xf32_rcr_compv3_cshuffle_intrawave")
    with pytest.raises(ValueError, match="unsupported A/B dtype 'xf32'"):
        r.run(
            np.ones((4, 8), np.float32),
            np.ones((8, 4), np.float32),
            GemmProblem(M=4, N=4, K=8),
        )


def test_runner_rejects_int32_inputs_but_sizes_int8_c_as_int32():
    # int32 is an output-only dtype: it must not pass the A/B allow-list.
    with pytest.raises(ValueError, match="unsupported A/B dtype 'int32'"):
        _runner("gemm_int32_rcr_compv3_cshuffle_intrawave").run(
            np.ones((4, 8), np.float32),
            np.ones((8, 4), np.float32),
            GemmProblem(M=4, N=4, K=8),
        )
    r = _runner("gemm_int8_rcr_compv3_cshuffle_intrawave")
    r.run(np.ones((4, 8), np.float32), np.ones((8, 4), np.float32), GemmProblem(M=4, N=4, K=8))
    A_h, B_h, C_h = r.lib.bufs
    assert (A_h.dtype, B_h.dtype, C_h.dtype) == (np.int8, np.int8, np.int32)
