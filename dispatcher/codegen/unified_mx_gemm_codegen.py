# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
MX GEMM code generator (TileEngine -> Dispatcher bridge).

Emits ONE self-contained .hpp per concrete kernel config for the dispatcher's
ctypes path. To guarantee byte-identical kernels with Old-TE (the parity
contract), this generator does NOT re-implement the C++ header assembly -- it
directly reuses the Old-TE builder:

    tile_engine/ops/gemm/mx_gemm/mx_gemm_instance_builder.py
        -> MxGemmKernelBuilder._generate_kernel_instance(tile_config, trait_combo)

The generated instance defines, at GLOBAL scope, a ``SelectedKernel`` struct with
a static ``launch(const MxGemmHostArgs&, const ck_tile::stream_config&)``, the
``KERNEL_NAME`` string, and the ADataType/BDataType/CDataType/AccDataType/
ScaleType/MxGemmHostArgs/ALayout/BLayout/CLayout aliases the ctypes lib expects.

Each header is compiled per-kernel via force-include:

    hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE mx_gemm_ctypes_lib.cpp

mx_gemm is microscaling GEMM (fp4/fp8 A*B, per-32-K e8m0 block scales), gfx950
only. The single valid trait combo is comp_async + cshuffle + intrawave, with a
fixed 16x16x128 warp tile.
"""

import argparse
import contextlib
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Iterator, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Robust import of the Old-TE builder (independent of CWD).
# dispatcher/codegen/  ->  ../../tile_engine/ops/gemm/mx_gemm
# =============================================================================

_THIS_DIR = Path(__file__).resolve().parent
_GEMM_DIR = (_THIS_DIR / ".." / ".." / "tile_engine" / "ops" / "gemm").resolve()
_MX_GEMM_DIR = _GEMM_DIR / "mx_gemm"

for _p in (str(_GEMM_DIR), str(_MX_GEMM_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load_mx_builder():
    """Import MxGemmKernelBuilder from the Old-TE ops tree."""
    import importlib.util

    mx_path = _MX_GEMM_DIR / "mx_gemm_instance_builder.py"
    spec = importlib.util.spec_from_file_location(
        "mx_gemm_instance_builder", str(mx_path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MxGemmKernelBuilder


# =============================================================================
# Config validation (restrict to the single valid mx_gemm combo)
# =============================================================================

KERNEL_NAME_PREFIX = "mx_gemm"

VALID_DATATYPES = {"fp4", "fp8"}
VALID_LAYOUT = "rcr"
VALID_PIPELINE = "comp_async"
VALID_EPILOGUE = "cshuffle"
VALID_SCHEDULER = "intrawave"
FIXED_WARP_TILE = (16, 16, 128)

_REQUIRED_TILE_KEYS = (
    "tile_m",
    "tile_n",
    "tile_k",
    "warp_m",
    "warp_n",
    "warp_k",
    "warp_tile_m",
    "warp_tile_n",
    "warp_tile_k",
)


def _validate(cfg: dict) -> None:
    datatype = cfg.get("datatype")
    if datatype not in VALID_DATATYPES:
        raise ValueError(
            f"datatype must be one of {sorted(VALID_DATATYPES)}, got {datatype!r}"
        )

    layout = cfg.get("layout")
    if layout != VALID_LAYOUT:
        raise ValueError(f"layout must be {VALID_LAYOUT!r}, got {layout!r}")

    pipeline = cfg.get("pipeline", VALID_PIPELINE)
    if pipeline != VALID_PIPELINE:
        raise ValueError(
            f"pipeline must be {VALID_PIPELINE!r} for mx_gemm, got {pipeline!r}"
        )

    epilogue = cfg.get("epilogue", VALID_EPILOGUE)
    if epilogue != VALID_EPILOGUE:
        raise ValueError(
            f"epilogue must be {VALID_EPILOGUE!r} for mx_gemm, got {epilogue!r}"
        )

    scheduler = cfg.get("scheduler", VALID_SCHEDULER)
    if scheduler != VALID_SCHEDULER:
        raise ValueError(
            f"scheduler must be {VALID_SCHEDULER!r} for mx_gemm, got {scheduler!r}"
        )

    tc = cfg.get("tile_config")
    if not isinstance(tc, dict):
        raise ValueError("config must contain a 'tile_config' object")
    missing = [k for k in _REQUIRED_TILE_KEYS if k not in tc]
    if missing:
        raise ValueError(f"tile_config missing keys: {missing}")

    warp_tile = (tc["warp_tile_m"], tc["warp_tile_n"], tc["warp_tile_k"])
    if tuple(warp_tile) != FIXED_WARP_TILE:
        raise ValueError(
            f"mx_gemm warp tile is fixed at {FIXED_WARP_TILE}, got {tuple(warp_tile)}"
        )


def _tile_config_from_cfg(cfg: dict) -> dict:
    tc = cfg["tile_config"]
    return {k: int(tc[k]) for k in _REQUIRED_TILE_KEYS}


def _trait_combo_from_cfg(cfg: dict) -> Tuple:
    """7-tuple: (pipeline, epilogue, scheduler, pad_m, pad_n, pad_k, persistent)."""
    return (
        cfg.get("pipeline", VALID_PIPELINE),
        cfg.get("epilogue", VALID_EPILOGUE),
        cfg.get("scheduler", VALID_SCHEDULER),
        bool(cfg.get("pad_m", False)),
        bool(cfg.get("pad_n", False)),
        bool(cfg.get("pad_k", False)),
        bool(cfg.get("persistent", False)),
    )


# =============================================================================
# Builder driver
# =============================================================================

_AUTOGEN_BANNER = """\
// Auto-generated MX GEMM kernel instance -- DO NOT EDIT.
// Regenerate via dispatcher/codegen/unified_mx_gemm_codegen.py
// This header is force-included into the ctypes translation unit:
//   hipcc -include <this.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE mx_gemm_ctypes_lib.cpp
// It defines at global scope: SelectedKernel, KERNEL_NAME, ADataType/BDataType/
// CDataType/AccDataType/ScaleType, MxGemmHostArgs, and ALayout/BLayout/CLayout.
"""


@contextlib.contextmanager
def _make_builder(cfg: dict) -> Iterator["object"]:
    """Yield an MxGemmKernelBuilder bound to a self-cleaning temp working dir.

    The Old-TE __init__ loads config_json only when it is an existing FILE path,
    and _generate_kernel_instance reads config.get("k_block_per_cu"). So we write
    a minimal config file (carrying k_block_per_cu) and hand its path to the
    builder. The builder's own side-effect .hpp lands in the working dir (ignored).

    This is a context manager (not a plain factory) because it owns a temp dir:
    kernel_name()/generate_kernel() are shelled frequently (e.g. --list-name), so
    a leaked mkdtemp() per call would pile up mx_gemm_codegen_* dirs under /tmp.
    The builder is only used inside the `with` block, so cleanup on exit is safe.
    """
    MxGemmKernelBuilder = _load_mx_builder()

    with tempfile.TemporaryDirectory(prefix="mx_gemm_codegen_") as work_dir_str:
        work_dir = Path(work_dir_str)
        tmp_cfg = {"k_block_per_cu": int(cfg.get("k_block_per_cu", 1))}
        cfg_path = work_dir / "mx_gemm_codegen_config.json"
        cfg_path.write_text(json.dumps(tmp_cfg))

        gpu_target = cfg.get("gpu_target")
        if not gpu_target:
            raise ValueError(
                "mx_gemm codegen requires an explicit 'gpu_target' in the config; "
                "do not default to a specific GPU architecture."
            )
        yield MxGemmKernelBuilder(
            KERNEL_NAME_PREFIX,
            work_dir,
            gpu_target,
            cfg["datatype"],
            cfg["layout"],
            config_json=str(cfg_path),
        )


def _fix_includes(code: str) -> str:
    """Drop the Old-TE builder's stale ``ck_tile/ops/gemm_mx.hpp`` umbrella include.

    That umbrella does not exist on develop; the mx kernel + comp_async pipeline
    are already pulled in by ``ck_tile/ops/gemm.hpp`` (also emitted in the same
    header), and the host-side preshuffle helper by ``ck_tile/host.hpp`` in the
    ctypes TU. Removing the stale line keeps the generated header self-contained.
    """
    out_lines = []
    for line in code.splitlines(keepends=True):
        if '#include "ck_tile/ops/gemm_mx.hpp"' in line:
            continue
        out_lines.append(line)
    return "".join(out_lines)


def _generate(cfg: dict) -> Tuple[str, str]:
    """Return (kernel_name, instance_code) for the given concrete config."""
    _validate(cfg)
    tile_config = _tile_config_from_cfg(cfg)
    trait_combo = _trait_combo_from_cfg(cfg)
    with _make_builder(cfg) as builder:
        name, code = builder._generate_kernel_instance(tile_config, trait_combo)
    return name, _fix_includes(code)


def kernel_name(cfg: dict) -> str:
    """Compute the kernel name without generating/writing code."""
    _validate(cfg)
    tile_config = _tile_config_from_cfg(cfg)
    trait_combo = _trait_combo_from_cfg(cfg)
    with _make_builder(cfg) as builder:
        return builder._format_kernel_name(trait_combo, tile_config)


def generate_kernel(output_dir: Path, cfg: dict) -> Optional[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    name, code = _generate(cfg)
    out = output_dir / f"{name}.hpp"
    out.write_text(_AUTOGEN_BANNER + code)
    log.info("wrote %s", out.name)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="MX GEMM kernel header generator")
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--config", type=Path, help="path to a config JSON file")
    ap.add_argument("--config-json", type=str, help="inline config JSON string")
    ap.add_argument(
        "--list-name",
        action="store_true",
        help="print the kernel name only; write no file",
    )
    args = ap.parse_args()

    if args.config_json:
        cfg = json.loads(args.config_json)
    elif args.config:
        cfg = json.loads(Path(args.config).read_text())
    else:
        ap.error("one of --config-json or --config is required")

    if args.list_name:
        print(kernel_name(cfg))
        return 0

    if not args.output_dir:
        ap.error("--output-dir is required unless --list-name is given")

    return 0 if generate_kernel(args.output_dir, cfg) else 1


if __name__ == "__main__":
    raise SystemExit(main())
