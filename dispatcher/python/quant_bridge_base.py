#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared scaffolding for the block-scale quant GEMM dispatcher bridges.

The five ``gemm_<op>_utils.py`` bridges (aquant / abquant / bquant /
rowcolquant / tensor_quant) were ~85% mechanically duplicated.  This module
folds the genuinely-shared, mechanical parts into one place so they cannot
drift, while every op-specific / correctness-load-bearing part stays in its own
file.  It builds on :mod:`dispatcher_common` (path + tool helpers) rather than
re-deriving them.

What lives here (verified byte-identical across the copies it replaces):

  * :func:`find_ck_include_dir` -- the ``_get_ck_include_dir`` include-probe.
  * :func:`install_dispatcher_lib_api` -- the ctypes ``_setup`` scaffold for the
    ``dispatcher_initialize`` / ``get_kernel_name`` / ``get_kernel_count`` /
    ``cleanup`` symbols, plus the per-op ``dispatcher_run_<op>_gemm`` argtypes
    driven from an ARGSPEC the caller passes in.
  * :func:`generate_kernel` -- the codegen subprocess (``_generate_<op>_kernel``).
  * :func:`build_dispatchers` -- the dedupe-by-name + ThreadPoolExecutor +
    "fill duplicates" orchestration (``setup_multiple_<op>_dispatchers``),
    parameterized by a per-op ``compile_fn`` (each op keeps its own hipcc flag /
    arch-define / static-lib / timeout choices, which genuinely diverge).

What deliberately does NOT live here (kept per-op -- see the report / each file):

  * ``_detect_gpu_arch`` -- five distinct implementations (different supported-arch
    sets, messages, and validation; tensor_quant's even differs subtly).
  * the ``_compile_<op>_kernel`` flag/define/timeout/static-lib bodies.
  * the fp8/bf8 encode helpers -- rowcolquant and bquant DISAGREE on the gfx950
    fp8 ml_dtype (``float8_e4m3`` vs ``float8_e4m3fn``), so merging them would be
    a numerical behavior change.
  * ``_warp_tile_k_for`` / ``default_*_config`` / ``KernelConfig`` / ``run()``.

No GPU / hipcc is required to import or exercise the codegen path here.
"""

import concurrent.futures
import ctypes
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)


# ============================================================================
# Include-directory probe (canonical _get_ck_include_dir)
# ============================================================================


def find_ck_include_dir() -> Optional[Path]:
    """Locate the CK include directory relative to this file (or None).

    Byte-for-byte the ``_get_ck_include_dir`` every bridge carried: walk up from
    ``dispatcher/python/`` and return the first ancestor ``include/`` that
    contains ``ck_tile/``.
    """
    here = Path(__file__).resolve().parent
    for parent in [here.parent.parent, here.parent.parent.parent]:
        candidate = parent / "include"
        if (candidate / "ck_tile").is_dir():
            return candidate
    return None


# ============================================================================
# ctypes DispatcherLib scaffold
# ============================================================================


def install_dispatcher_lib_api(
    lib: "ctypes.CDLL",
    run_symbol: str,
    run_argtypes: Sequence["object"],
) -> None:
    """Wire the restype/argtypes on a loaded quant-bridge ``.so``.

    Every bridge's ``DispatcherLib._setup`` declared the identical
    ``dispatcher_initialize`` / ``dispatcher_get_kernel_name`` /
    ``dispatcher_get_kernel_count`` / ``dispatcher_cleanup`` signatures; only the
    ``dispatcher_run_<op>_gemm`` argtypes differed per op.  Pass that op's
    ``run_symbol`` name and ``run_argtypes`` list (its ARGSPEC) and this installs
    all of them exactly as the copies did.
    """
    lib.dispatcher_initialize.restype = ctypes.c_int
    lib.dispatcher_initialize.argtypes = []

    run_fn = getattr(lib, run_symbol)
    run_fn.restype = ctypes.c_int
    run_fn.argtypes = list(run_argtypes)

    lib.dispatcher_get_kernel_name.restype = ctypes.c_char_p
    lib.dispatcher_get_kernel_name.argtypes = []

    lib.dispatcher_get_kernel_count.restype = ctypes.c_int
    lib.dispatcher_get_kernel_count.argtypes = []

    lib.dispatcher_cleanup.restype = None
    lib.dispatcher_cleanup.argtypes = []


class DispatcherLibBase:
    """Common ctypes wrapper for a compiled quant-bridge ``.so``.

    Subclasses declare two class attributes and (optionally) their own ``run``:

      * ``_NOT_FOUND_LABEL`` -- op label used in the FileNotFoundError message
        (e.g. ``"AQuant"``), preserving each bridge's original wording.
      * ``_RUN_SYMBOL`` / ``_RUN_ARGTYPES`` -- the ``dispatcher_run_<op>_gemm``
        symbol name and its ctypes argtypes list (the per-op ARGSPEC).

    ``__init__`` / ``get_kernel_name`` / ``get_kernel_count`` / ``cleanup`` /
    ``__del__`` were byte-identical across the five bridges (modulo the op label),
    so they live here once.
    """

    _NOT_FOUND_LABEL: str = "quant"
    _RUN_SYMBOL: str = ""
    _RUN_ARGTYPES: Sequence["object"] = ()

    def __init__(self, so_path: Path):
        self.so_path = Path(so_path)
        if not self.so_path.exists():
            raise FileNotFoundError(
                f"{self._NOT_FOUND_LABEL} .so not found: {self.so_path}"
            )
        self._lib = ctypes.CDLL(str(self.so_path))
        self._setup()
        rc = self._lib.dispatcher_initialize()
        if rc != 0:
            raise RuntimeError(f"dispatcher_initialize() returned {rc}")

    def _setup(self):
        install_dispatcher_lib_api(
            self._lib, self._RUN_SYMBOL, self._RUN_ARGTYPES
        )

    def get_kernel_name(self) -> str:
        raw = self._lib.dispatcher_get_kernel_name()
        return raw.decode("utf-8") if raw else ""

    def get_kernel_count(self) -> int:
        return self._lib.dispatcher_get_kernel_count()

    def cleanup(self):
        self._lib.dispatcher_cleanup()

    def __del__(self):
        try:
            self._lib.dispatcher_cleanup()
        except Exception:
            pass


# ============================================================================
# Codegen subprocess (canonical _generate_<op>_kernel)
# ============================================================================


def generate_kernel(
    config,
    output_dir: Path,
    codegen_script: Path,
    timeout: int = 120,
) -> Optional[Path]:
    """Run a unified ``*_codegen.py`` for one config; return the ``.hpp`` or None.

    Identical to every bridge's ``_generate_<op>_kernel``: serialize
    ``config.to_codegen_config()`` to JSON, invoke the op's codegen script, and
    return ``<output_dir>/<config.name>.hpp`` if it materialized.
    """
    config_dict = config.to_codegen_config()
    config_json = json.dumps(config_dict)

    cmd = [
        sys.executable,
        str(codegen_script),
        "--output-dir", str(output_dir),
        "--config-json", config_json,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            log.error("Codegen failed for %s:\n%s", config.name, result.stderr)
            return None
    except subprocess.TimeoutExpired:
        log.error("Codegen timed out for %s", config.name)
        return None

    hpp = output_dir / f"{config.name}.hpp"
    if not hpp.exists():
        log.error("Codegen succeeded but %s not found", hpp)
        return None

    return hpp


# ============================================================================
# Build orchestration (canonical setup_multiple_<op>_dispatchers)
# ============================================================================


def build_dispatchers(
    configs: List,
    arch: str,
    tmp_prefix: str,
    log_label: str,
    generate_fn: Callable[[object, Path], Optional[Path]],
    compile_fn: Callable[[Path, Path, str], bool],
    output_dir: Optional[Path] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
) -> List[Optional[Path]]:
    """codegen -> compile -> ``.so`` for each config, deduped by name, in parallel.

    This is the ``setup_multiple_<op>_dispatchers`` body every bridge copied
    verbatim -- dedupe-by-name, ThreadPoolExecutor fan-out, ``[cached]`` skip,
    and the "fill duplicates" pass -- lifted out once.  The op-specific pieces
    are injected:

      * ``arch``           already resolved by the caller (each op has its own
                           ``_detect_gpu_arch`` / ``_validate_arch`` policy and
                           any early guards, e.g. bquant's MX-arch check).
      * ``tmp_prefix``     ``tempfile.mkdtemp`` prefix (e.g. ``"aquant_dispatcher_"``).
      * ``log_label``      human label in the log lines (e.g. ``"AQuant"``).
      * ``generate_fn``    ``lambda cfg, hdr_dir -> Optional[Path]`` (the op's
                           ``_generate_<op>_kernel``).
      * ``compile_fn``     ``lambda hpp, so, arch -> bool`` (the op's
                           ``_compile_<op>_kernel``; keeps its flags/defines).

    Returns a list parallel to ``configs`` (Path or None per entry).  No GPU is
    required to call this.
    """
    base_dir = output_dir or Path(tempfile.mkdtemp(prefix=tmp_prefix))
    base_dir.mkdir(parents=True, exist_ok=True)

    headers_dir = base_dir / "generated_kernels"
    so_dir = base_dir / "libs"
    headers_dir.mkdir(exist_ok=True)
    so_dir.mkdir(exist_ok=True)

    log.info(
        "Building %d %s kernel(s) for %s into %s",
        len(configs), log_label, arch, base_dir,
    )

    # Deduplicate by name so we don't build the same kernel twice.
    seen: Dict[str, int] = {}          # name -> index of first occurrence
    deduped: List[Tuple[int, object]] = []
    for i, cfg in enumerate(configs):
        if cfg.name not in seen:
            seen[cfg.name] = i
            deduped.append((i, cfg))

    results: List[Optional[Path]] = [None] * len(configs)

    def _build_one(idx: int, cfg) -> Tuple[int, Optional[Path]]:
        hpp = generate_fn(cfg, headers_dir)
        if hpp is None:
            return idx, None

        so = so_dir / f"lib{cfg.name}_{arch}.so"
        if so.exists():
            log.info("  [cached] %s", so.name)
            return idx, so

        ok = compile_fn(hpp, so, arch)
        return idx, so if ok else None

    if parallel and len(deduped) > 1:
        workers = max_workers or min(len(deduped), os.cpu_count() or 4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_build_one, idx, cfg): (idx, cfg) for idx, cfg in deduped}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    idx, so_path = fut.result()
                    results[idx] = so_path
                    if so_path:
                        log.info("  built %s", so_path.name)
                    else:
                        _, cfg = futures[fut]
                        log.error("  FAILED %s", cfg.name)
                except Exception as e:
                    _, cfg = futures[fut]
                    log.error("  EXCEPTION for %s: %s", cfg.name, e)
    else:
        for idx, cfg in deduped:
            _, so_path = _build_one(idx, cfg)
            results[idx] = so_path

    # Fill in duplicates.
    for i, cfg in enumerate(configs):
        if results[i] is None:
            first_idx = seen.get(cfg.name)
            if first_idx is not None and first_idx != i:
                results[i] = results[first_idx]

    built = sum(1 for r in results if r is not None)
    log.info("Built %d / %d %s kernels", built, len(configs), log_label)
    return results
