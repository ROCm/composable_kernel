# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GPU SMI wrappers (amd-smi / rocm-smi).

Prefers amd-smi and falls back to rocm-smi if amd-smi is missing or fails. All
call sites should use these helpers instead of invoking amd-smi or rocm-smi
directly.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Optional

_ROCM_SMI = "rocm-smi"
_AMD_SMI = "amd-smi"


def _smi_order() -> tuple[str, str]:
    """Return (primary, fallback) SMI tools to try, in order.

    Defaults to amd-smi first, rocm-smi as fallback. Set CK_SMI_TOOL=rocm-smi
    to force rocm-smi first (mainly for testing).
    """
    if os.environ.get("CK_SMI_TOOL", "").strip().lower() == "rocm-smi":
        return (_ROCM_SMI, _AMD_SMI)
    return (_AMD_SMI, _ROCM_SMI)


def parse_rocm_showid(text: str) -> list[str]:
    ids = re.findall(r"GPU\[(\d+)\]", text)
    return [str(i) for i in sorted(set(int(x) for x in ids))]


def parse_amd_list(text: str) -> list[str]:
    ids = re.findall(r"^GPU:\s*(\d+)", text, re.MULTILINE)
    return [str(i) for i in sorted(set(int(x) for x in ids))]


def parse_rocm_productname(text: str) -> Optional[str]:
    match = re.search(r"Card Series:\s+(.+)", text)
    return match.group(1).strip() if match else None


def parse_amd_static_market_name(text: str) -> Optional[str]:
    match = re.search(r"MARKET_NAME:\s+(.+)", text)
    return match.group(1).strip() if match else None


def parse_rocm_gfx(text: str) -> Optional[str]:
    match = re.search(r"GFX Version:\s+(\S+)", text)
    return match.group(1).strip() if match else None


def parse_amd_gfx(text: str) -> Optional[str]:
    match = re.search(r"TARGET_GRAPHICS_VERSION:\s+(\S+)", text)
    return match.group(1).strip() if match else None


def parse_rocm_driver_version(text: str) -> Optional[str]:
    match = re.search(r"Driver version:\s+(\S+)", text)
    return match.group(1).strip() if match else None


def parse_amd_driver_version(text: str) -> Optional[str]:
    match = re.search(r"amdgpu version:\s+(\S+)", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"VERSION:\s+(\S+)", text)
    return match.group(1).strip() if match else None


def normalize_gpu_fields(
    rocm_showid: str = "",
    amd_list: str = "",
    rocm_product: str = "",
    amd_static: str = "",
    rocm_driver: str = "",
    amd_version: str = "",
) -> dict[str, object]:
    """Extract comparable fields from raw rocm-smi / amd-smi output."""
    return {
        "gpu_ids_rocm": parse_rocm_showid(rocm_showid),
        "gpu_ids_amd": parse_amd_list(amd_list),
        "product_rocm": parse_rocm_productname(rocm_product),
        "product_amd": parse_amd_static_market_name(amd_static),
        "gfx_rocm": parse_rocm_gfx(rocm_product),
        "gfx_amd": parse_amd_gfx(amd_static),
        "driver_rocm": parse_rocm_driver_version(rocm_driver),
        "driver_amd": parse_amd_driver_version(amd_version),
    }


def fetch_live_normalized_fields() -> dict[str, object]:
    """Run rocm-smi and amd-smi on the live system; return normalized fields."""
    return normalize_gpu_fields(
        rocm_showid=_run_cmd(["rocm-smi", "--showid"]),
        amd_list=_run_cmd(["amd-smi", "list"]),
        rocm_product=_run_cmd(["rocm-smi", "--showproductname"]),
        amd_static=_run_cmd(["amd-smi", "static"]),
        rocm_driver=_run_cmd(["rocm-smi", "--showdriverversion"]),
        amd_version=_run_cmd(["amd-smi", "version"]),
    )


def detect_gpu_arch(fallback: Optional[str] = None) -> Optional[str]:
    """Return the GPU gfx arch string (e.g. "gfx950"), preferring amd-smi.

    Tries amd-smi ``static`` (TARGET_GRAPHICS_VERSION) first, then rocm-smi
    ``--showproductname`` (GFX Version), matching the amd-smi-first policy used
    throughout this module. Returns ``fallback`` if neither yields an arch.
    """
    for tool in _smi_order():
        if shutil.which(tool) is None:
            continue
        try:
            if tool == _AMD_SMI:
                arch = parse_amd_gfx(_run_cmd([_AMD_SMI, "static"]))
            else:
                arch = parse_rocm_gfx(_run_cmd([_ROCM_SMI, "--showproductname"]))
        except Exception:  # noqa: BLE001 - tool present but query failed; try next
            arch = None
        if arch:
            return arch
    return fallback


def smi_equivalence_pairs(
    fields: dict[str, object],
) -> list[tuple[str, object, object]]:
    """Return (name, rocm_value, amd_value) tuples for cross-tool comparison."""
    return [
        ("gpu_ids", fields["gpu_ids_rocm"], fields["gpu_ids_amd"]),
        ("product", fields["product_rocm"], fields["product_amd"]),
        ("gfx", fields["gfx_rocm"], fields["gfx_amd"]),
        ("driver", fields["driver_rocm"], fields["driver_amd"]),
    ]


def _run_cmd(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)


def _gpu_ids_from_env() -> Optional[list[str]]:
    env = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    )
    if not env:
        return None
    ids = [d.strip() for d in env.split(",") if d.strip() != ""]
    return ids if ids else None


def _gpu_ids_rocm_smi() -> list[str]:
    out = _run_cmd([_ROCM_SMI, "--showid"])
    ids = parse_rocm_showid(out)
    if ids:
        return ids
    raise RuntimeError("rocm-smi --showid returned no GPU ids")


def _gpu_ids_amd_smi() -> list[str]:
    out = _run_cmd([_AMD_SMI, "list"])
    ids = parse_amd_list(out)
    if ids:
        return ids
    raise RuntimeError("amd-smi list returned no GPU ids")


def detect_gpu_ids() -> list[str]:
    """Return visible GPU id strings (best-effort).

    Returns an empty list when no GPUs can be detected (no visibility env var
    set and neither amd-smi nor rocm-smi is available or succeeds).
    """
    env_ids = _gpu_ids_from_env()
    if env_ids is not None:
        return env_ids

    fetchers = {
        _ROCM_SMI: _gpu_ids_rocm_smi,
        _AMD_SMI: _gpu_ids_amd_smi,
    }
    for tool in _smi_order():
        if shutil.which(tool) is None:
            continue
        try:
            return fetchers[tool]()
        except Exception:
            continue
    return []


def count_gpus() -> int:
    return len(detect_gpu_ids())


def _run_smi_primary_fallback(rocm_cmd: list[str], amd_cmd: list[str]) -> str:
    cmds = {_ROCM_SMI: rocm_cmd, _AMD_SMI: amd_cmd}
    for tool in _smi_order():
        if shutil.which(tool) is None:
            continue
        try:
            return _run_cmd(cmds[tool])
        except Exception:
            continue
    raise RuntimeError("no SMI tool available")


def show_gpu_info(head: Optional[int] = 10) -> str:
    out = _run_smi_primary_fallback(
        [_ROCM_SMI, "--showproductname"],
        [_AMD_SMI, "static"],
    )
    if head is not None and head > 0:
        lines = out.splitlines()
        return "\n".join(lines[:head])
    return out


def check_gpu_available() -> bool:
    if shutil.which(_ROCM_SMI) is None and shutil.which(_AMD_SMI) is None:
        return False
    try:
        ids = detect_gpu_ids()
        return len(ids) > 0
    except Exception:
        return False


def show_version() -> str:
    version_cmds = {
        _AMD_SMI: [_AMD_SMI, "version"],
        _ROCM_SMI: [_ROCM_SMI, "--showdriverversion"],
    }
    for tool in _smi_order():
        if shutil.which(tool) is None:
            continue
        try:
            return _run_cmd(version_cmds[tool]).strip()
        except Exception:
            continue
    raise RuntimeError("no SMI tool available for version query")
