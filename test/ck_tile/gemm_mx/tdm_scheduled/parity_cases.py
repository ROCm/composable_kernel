#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The comp_tdm_scheduled validation cases both rule sets must agree on.

tile_engine's mx_tdm_scheduled.py (validate_instance, MxTdmScheduledSpec.validate) and the C++
policy (Policy::Validate, the pipeline's GetUnsupportedReason) implement the same rules. Each row
here is checked by both: dispatcher/tests/test_mx_gemm_tdm_scheduled.py runs the Python spec, and
test_mx_gemm_tdm_scheduled_host.cpp includes the rows as parity_cases.inc, which CMake generates
in the build tree with this script (--output).
"""

import argparse
import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CK = _HERE.parents[3]


def load_spec_module():
    """tile_engine's mx_tdm_scheduled module (shared with gemm_validation_utils' loader)."""
    name = "mx_tdm_scheduled"
    if name not in sys.modules:
        path = _CK / "tile_engine" / "ops" / "gemm" / "mx_gemm" / "mx_tdm_scheduled.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _tile(tile_m, tile_n, tile_k, warp_m, warp_n, atom_m):
    return {
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "warp_m": warp_m,
        "warp_n": warp_n,
        "warp_k": 1,
        "warp_tile_m": atom_m,
        "warp_tile_n": 16,
        "warp_tile_k": 128,
    }


T256X256X512 = _tile(256, 256, 512, 2, 2, 32)
T128X16X512 = _tile(128, 16, 512, 4, 1, 16)
T128X128X256 = _tile(128, 128, 256, 2, 2, 16)
NO_CLUSTER = {"cluster_dim": (1, 1)}


def _row(
    name, tile, reason, a="fp4", b=None, out="fp8e4m3", tuning=None, mnk=(0, 0, 0)
):
    return {
        "name": name,
        "tile": tile,
        "a": a,
        "b": b or a,
        "out": out,
        "tuning": tuning or {},
        "mnk": mnk,
        "reason": reason,
    }


#: M, N, K of 0: an instance row (Policy::Validate / validate_instance). Otherwise a problem row
#: on a valid instance (GetUnsupportedReason / MxTdmScheduledSpec.validate).
PARITY_CASES = [
    _row("t256x256x512", T256X256X512, "ok"),
    _row(
        "t128x16x512",
        T128X16X512,
        "ok",
        out="fp16",
        tuning={
            "cluster_dim": (1, 4),
            "gl2_prefetch_distance": 0,
            "prefetch_lds_read": False,
            "split_b_lds_read": False,
        },
    ),
    _row("out_f32", T256X256X512, "out_dtype", out="fp32"),
    _row("tile_m_48", _tile(48, 256, 512, 2, 2, 32), "tile_mn"),
    _row("in_f16", T256X256X512, "dtype", a="fp16"),
    _row("fp4_x_fp8", T256X256X512, "dtype_width", b="fp8"),
    _row("fp8_tile_k_512", T256X256X512, "tile_k", a="fp8"),
    _row("waves_2x1", _tile(256, 256, 512, 2, 1, 32), "wave_group"),
    _row("fp8_atom_32", _tile(128, 128, 256, 2, 2, 32), "mma", a="fp8"),
    _row("tile_16x16", _tile(16, 16, 128, 2, 2, 32), "tile_atoms"),
    _row(
        "split_b_no_prefetch",
        T256X256X512,
        "split_b_read",
        tuning={"prefetch_lds_read": False},
    ),
    _row("gl2_1", T256X256X512, "gl2_distance", tuning={"gl2_prefetch_distance": 1}),
    _row("cluster_1x0", T256X256X512, "cluster_dim", tuning={"cluster_dim": (1, 0)}),
    _row("cluster_8x4", T256X256X512, "cluster_size", tuning={"cluster_dim": (8, 4)}),
    _row(
        "interleave_4x1_m32",
        _tile(128, 16, 128, 4, 1, 32),
        "interleave_repeat",
        tuning={"split_b_lds_read": False},
    ),
    _row(
        "interleave_b_split",
        _tile(16, 256, 1024, 1, 4, 16),
        "interleave_baseline_split",
    ),
    _row("lds_64x256x1024", _tile(64, 256, 1024, 2, 2, 32), "lds_capacity"),
    _row("t256x256x512_1024", T256X256X512, "ok", mnk=(1024, 1024, 1024)),
    _row("m_0", T256X256X512, "shape_positive", tuning=NO_CLUSTER, mnk=(0, 256, 512)),
    _row(
        "k_1040", T256X256X512, "k_scale_block", tuning=NO_CLUSTER, mnk=(256, 256, 1040)
    ),
    _row(
        "m_4m_k_64k",
        T256X256X512,
        "scale_span",
        tuning=NO_CLUSTER,
        mnk=(1 << 22, 256, 1 << 16),
    ),
    _row("k_256", T256X256X512, "k_short", tuning=NO_CLUSTER, mnk=(256, 256, 256)),
    _row(
        "k_near_i32_max",
        _tile(32, 32, 256, 2, 2, 16),
        "k_long",
        a="fp8",
        tuning={"cluster_dim": (1, 1), "split_b_lds_read": False},
        mnk=(32, 32, 2147482368),
    ),
    _row(
        "k_tail_160",
        T128X128X256,
        "k_tail_full",
        a="fp8",
        tuning={"split_b_lds_read": False},
        mnk=(1024, 1024, 2048 + 160),
    ),
    _row("grid_3x3_cluster_4x4", T256X256X512, "cluster_grid", mnk=(768, 768, 1024)),
]

#: Reasons no row reaches, with why.
HOST_UNREACHABLE = {
    # The pads are derived from the tile, and every valid tile derives encodable ones.
    "lds_pad": "derived pads always encode",
    # At most 512 accumulators: 256x256 with a 2x2 wave group, 32x16 atoms (4 x 8 x 16).
    "accumulators": "valid tiles hold at most 512 accumulators per lane",
    # The aligned interleaved layouts of every valid tile fit in half the LDS.
    "interleave_capacity": "no valid tile's aligned layout exceeds the LDS",
}

#: validate_instance reasons outside REASONS and the C++ enum: TailSteps is a GemmTraits value.
PYTHON_ONLY = {"tail_steps"}

_CPP_DTYPES = {
    "fp4": "ck_tile::pk_fp4_t",
    "fp8": "ck_tile::fp8_t",
    "bf8": "ck_tile::bf8_t",
    "fp16": "ck_tile::half_t",
    "fp8e4m3": "ck_tile::fp8_t",
    "bf16": "ck_tile::bf16_t",
    "fp32": "float",
}


def row_spec(row):
    """The MxTdmScheduledSpec of a row."""
    sched = load_spec_module()
    m, n, k = row["mnk"]
    return sched.spec_from_tile(
        row["tile"], row["tuning"], row["a"], row["b"], row["out"], m, n, k
    )


def python_reason(row):
    """The Python spec's reason for a row."""
    sched = load_spec_module()
    spec = row_spec(row)
    if row["mnk"] == (0, 0, 0):
        return sched.validate_instance(spec, row["tuning"])
    return spec.validate(check_shape=True)


def _cpp_row(row):
    sched = load_spec_module()
    t, tuning = row["tile"], row["tuning"]
    shape = (
        f"ck_tile::ClusterTileGemmShape<{sched.cpp_cluster_sequence(tuning)}, "
        f"ck_tile::sequence<{t['tile_m']}, {t['tile_n']}, {t['tile_k']}>, "
        f"ck_tile::sequence<{t['warp_m']}, {t['warp_n']}, 1>, "
        f"ck_tile::sequence<{t['warp_tile_m']}, 16, 128>>"
    )
    problem = (
        f"ck_tile::MxGemmTdmScheduledProblem<{_CPP_DTYPES[row['a']]}, "
        f"{_CPP_DTYPES[row['b']]}, {_CPP_DTYPES[row['out']]}, {shape}, "
        f"{sched.cpp_kernel_traits(row_spec(row), tuning)}>"
    )
    policy = sched.cpp_policy_type(tuning)
    if row["mnk"] == (0, 0, 0):
        check = f'check_instance<P, Policy>("{row["name"]}", "{row["reason"]}");'
    else:
        m, n, k = row["mnk"]
        check = f'check_problem<P, Policy>("{row["name"]}", {m}, {n}, {k}, "{row["reason"]}");'
    return [
        "{",
        f"    using P = {problem};",
        f"    using Policy = {policy};",
        f"    {check}",
        "}",
    ]


def render_inc():
    """parity_cases.inc: the rows as C++ blocks, included in a TEST body."""
    lines = [
        "// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.",
        "// SPDX-License-Identifier: MIT",
        "",
        "// Generated by parity_cases.py from PARITY_CASES at build time.",
    ]
    for row in PARITY_CASES:
        lines += _cpp_row(row)
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, type=Path, help="parity_cases.inc to write"
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_inc())


if __name__ == "__main__":
    main()
