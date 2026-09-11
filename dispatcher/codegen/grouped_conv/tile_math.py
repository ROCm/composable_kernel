#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Mathematical functions for deriving valid tile/warp/vector configurations.

Key source files this is derived from:
  - block_universal_gemm_as_bs_cr.hpp   (tile divisibility by warps)
  - gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp  (vec/LDS formulas)
  - conv_algorithm_limits.hpp           (VMEM/LDS vector size validity)
  - warp_gemm_dispatcher.hpp            (XDL warp tile shapes per dtype)
  - arch_specs_generated.py             (arch-specific wave/warp-tile combos)
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Path setup — allow running from any directory
# ---------------------------------------------------------------------------
_CODEGEN_DIR = Path(__file__).resolve().parent.parent
if str(_CODEGEN_DIR) not in sys.path:
    sys.path.insert(0, str(_CODEGEN_DIR))

from arch_specs_generated import (
    WARP_SUPPORTED_COMBINATIONS,       # [wave_m, wave_n, wave_k] per arch
    WARP_TILE_SUPPORTED_COMBINATIONS,  # [warp_m, warp_n, warp_k] per arch+dtype
    ELEMENT_SIZE_MAP,                  # bytes per element per dtype string
)

# Warp size on AMD GPUs
WARP_SIZE = 64

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pos_divisors(n: int) -> List[int]:
    """Return all positive divisors of n in ascending order."""
    if n <= 0:
        return []
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    return sorted(divs)


def _lds_valid(vec: int, sizeof_dtype: float) -> bool:
    """LDS vector load/store must be a power-of-2 multiple of 8 bits, up to 256 bits.

    Some bwd_data configs use larger global-load vectors (e.g. fp32×8=256 bits)
    where the global load is split across DWORD pairs rather than going through LDS.
    We therefore accept up to 256 bits and require the width to be a power of 2 in bytes.
    """
    bits = vec * sizeof_dtype * 8
    # Must be positive, a power of 2 in bit-width, and at most 256 bits
    if bits <= 0 or bits > 256:
        return False
    # Check power of 2
    b = int(bits)
    return b > 0 and (b & (b - 1)) == 0


def _pipeline_wave_valid(
    wave_m: int, wave_n: int, wave_k: int,
    warp_tile_m: int, warp_tile_n: int, warp_tile_k: int,
    pipeline: Optional[str],
) -> bool:
    """Return True if this wave/warp combo is valid for the given pipeline.

    Pipeline-specific constraints derived from static asserts in:
      - gemm_pipeline_ag_bg_cr_comp_async_eight_waves_policy.hpp
        (NWarps==2, WarpTile::at(I1)==16 for basic_async_v1 eight-wave)
      - TDM pipeline (BlockSize == warp_size * 4, WarpTile M=N=32)
    """
    if pipeline is None:
        return True

    p = pipeline.lower()

    if p == "basic_async_v1":
        # Eight-wave async: NWarps must be 2, warp_tile_n must be 16
        return wave_n == 2 and warp_tile_n == 16

    if p in ("tdm", "tdmv2"):
        # TDM requires exactly 4 waves and 32x32 warp tile
        return (wave_m * wave_n * wave_k == 4
                and warp_tile_m == 32 and warp_tile_n == 32)

    # All other pipelines (compv1..v6, mem, comp_async, basic_v1, etc.): no constraint
    return True


def _deduplicate(pairs: List[Tuple[Tuple, Tuple]]) -> List[Tuple[Tuple, Tuple]]:
    """Remove duplicate (wave, warp_tile) pairs while preserving order."""
    return list(dict.fromkeys(pairs))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_valid_wave_warp_pairs(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    dtype_key: str,
    arch: str = "gfx942",
    pipeline: Optional[str] = None,
) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """Return all valid ((wave_m, wave_n, wave_k), (warp_tile_m, warp_tile_n, warp_tile_k)) pairs.

    Derived from the static assert in block_universal_gemm_as_bs_cr.hpp:
        MIterPerWarp * MWarp * WarpGemm::kM == MPerBlock
        NIterPerWarp * NWarp * WarpGemm::kN == NPerBlock

    which means:  tile_m == wave_m * warp_tile_m * iter_m  (iter_m >= 1)
                  tile_n == wave_n * warp_tile_n * iter_n  (iter_n >= 1)

    Args:
        tile_m, tile_n, tile_k: block tile dimensions
        dtype_key: e.g. "bf16_bf16_fp32", "fp32_fp32_fp32"
        arch: GPU architecture string, default "gfx942"
        pipeline: optional pipeline name to apply pipeline-specific constraints

    Returns:
        List of ((wave_m, wave_n, wave_k), (warp_tile_m, warp_tile_n, warp_tile_k)) tuples.
        Each pair is structurally valid for the given arch and pipeline.
    """
    supported_wave_combos: Set[Tuple[int, int, int]] = {
        tuple(c) for c in WARP_SUPPORTED_COMBINATIONS.get(arch, [])
    }
    warp_tile_shapes: List[List[int]] = (
        WARP_TILE_SUPPORTED_COMBINATIONS
        .get(arch, {})
        .get(dtype_key, [])
    )

    results: List[Tuple[Tuple, Tuple]] = []

    for wt in warp_tile_shapes:
        warp_m, warp_n, warp_k = wt[0], wt[1], wt[2]

        # Tile must be divisible by the warp tile in M and N
        if tile_m % warp_m != 0 or tile_n % warp_n != 0:
            continue

        # Enumerate all integer (iter_m, iter_n) >= 1 such that the block is tiled exactly
        for iter_m in _pos_divisors(tile_m // warp_m):
            wave_m = tile_m // (warp_m * iter_m)
            for iter_n in _pos_divisors(tile_n // warp_n):
                wave_n = tile_n // (warp_n * iter_n)

                # Normal case: wave_k = 1
                if (wave_m, wave_n, 1) in supported_wave_combos:
                    if _pipeline_wave_valid(wave_m, wave_n, 1, warp_m, warp_n, warp_k, pipeline):
                        results.append(((wave_m, wave_n, 1), (warp_m, warp_n, warp_k)))

                # Special case: wave_k = 2
                # Only a small number of tiles use this (e.g. (128,32,32) with warp=(32,32,8)).
                # Supported on gfx942/gfx950 via the [2,1,2] wave combo.
                if (wave_m, wave_n, 2) in supported_wave_combos:
                    if _pipeline_wave_valid(wave_m, wave_n, 2, warp_m, warp_n, warp_k, pipeline):
                        results.append(((wave_m, wave_n, 2), (warp_m, warp_n, warp_k)))

    return _deduplicate(results)


def get_valid_vec_sizes(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    wave_m: int,
    wave_n: int,
    wave_k: int,
    warp_tile_m: int,
    warp_tile_n: int,
    warp_tile_k: int,
    dtype_key: str,
    pipeline: Optional[str] = None,
) -> List[Tuple[int, int, int]]:
    """Return all valid (vec_a, vec_b, vec_c) triples for a fully-specified config.

    The thread-pixel budget formula:
        block_size = WARP_SIZE * wave_m * wave_n * wave_k
        pixels_a   = tile_m * tile_k / block_size   (elements per thread, A tile)
        pixels_b   = tile_n * tile_k / block_size   (elements per thread, B tile)

    Valid vec_a/vec_b must be compatible with their respective pixel budget and
    satisfy VMEM/LDS hardware constraints.  Compatibility means either the
    vector divides the per-thread pixel budget (v divides pixels) OR the vector
    is an exact multiple of it (pixels divides v) — in the latter case a single
    wide load simply decomposes into v/pixels sub-loads, which is valid on
    hardware (observed for asymmetric small-pixel tiles).  vec_c is constrained
    by the XDL output shuffle: tile_n must be divisible by vec_c.

    Args:
        tile_m, tile_n, tile_k: block tile dimensions
        wave_m, wave_n, wave_k: wave counts
        warp_tile_m, warp_tile_n, warp_tile_k: XDL warp tile dimensions
        dtype_key: e.g. "bf16_bf16_fp32"
        pipeline: optional, currently unused

    Returns:
        Sorted list of (vec_a, vec_b, vec_c) tuples.
    """
    dtype_a = dtype_key.split("_")[0]
    dtype_b = dtype_key.split("_")[1]
    sizeof_a = float(ELEMENT_SIZE_MAP.get(dtype_a, 2))  # bytes per A element

    # vec_b / vec_c reuse sizeof_a, so this is only valid when A and B share an
    # element type (the C output element type is assumed to match the input).
    # The third field is the accumulator (fp32/int32), so it is intentionally
    # not compared here.
    if dtype_a != dtype_b:
        raise ValueError(
            f"get_valid_vec_sizes assumes A and B share an element type, got {dtype_key}"
        )

    block_size = WARP_SIZE * wave_m * wave_n * wave_k

    if block_size == 0 or tile_m * tile_k % block_size != 0 or tile_n * tile_k % block_size != 0:
        return []

    pixels_a = (tile_m * tile_k) // block_size
    pixels_b = (tile_n * tile_k) // block_size

    # Maximum vector width per element type.
    # Standard VMEM load limit is 16 bytes (128 bits), which gives:
    #   fp32 (4 bytes) -> 4 elements;  bf16/fp16 (2 bytes) -> 8;  fp8 (1 byte) -> 16
    # However, some bwd_data configurations use vec_a=8 for fp32 (32-byte loads via
    # 2×16-byte split), which compiles and runs on hardware.  To avoid false negatives
    # the cap is relaxed to 16 bytes × 2 = the hardware dword-per-lane pair limit.
    # The LDS validity check below enforces the finer-grained hardware constraint.
    max_vec_ab = max(1, int(32 // sizeof_a))   # 2× standard VMEM width

    # Output vec_c: relaxed to the same 32-byte (2× VMEM) ceiling as vec_a/vec_b.
    # The finer-grained _lds_valid check below still enforces the <=256-bit
    # hardware ceiling, so e.g. bf16 vec_c stays <= 16.  This admits fp32 vec_c=8
    # observed in the profiler configs.
    max_vec_c = max(1, int(32 // sizeof_a))

    # A vector width is compatible with the per-thread pixel budget if it either
    # divides the budget or is an exact multiple of it (the wide load decomposes
    # into v/pixels sub-loads).
    valid_a = [
        v for v in [1, 2, 4, 8, 16]
        if v <= max_vec_ab
        and (pixels_a % v == 0 or v % pixels_a == 0)
        and _lds_valid(v, sizeof_a)
    ]

    valid_b = [
        v for v in [1, 2, 4, 8, 16]
        if v <= max_vec_ab
        and (pixels_b % v == 0 or v % pixels_b == 0)
        and _lds_valid(v, sizeof_a)
    ]

    # vec_c constraint: XDL accumulator is laid out in N-major tiles of size warp_tile_n.
    # The output shuffle requires tile_n divisible by (wave_n * warp_tile_n * vec_c).
    # vec_c constraint: the C accumulator is stored contiguously along N per thread.
    # The output shuffle in the XDL block gemm only requires tile_n to be divisible
    # by vec_c (not by wave_n * warp_tile_n * vec_c as the input tiles).
    # thread_cluster_dims[3] = 1 because each thread writes one N-element per shuffle step;
    # the n_xdl_per_wave repeats are handled by the outer loop, not the vector width.
    valid_c = [
        v for v in [1, 2, 4, 8, 16]
        if v <= max_vec_c
        and tile_n % v == 0
        and _lds_valid(v, sizeof_a)
    ]

    return sorted({(va, vb, vc) for va in valid_a for vb in valid_b for vc in valid_c})


def get_vec_sizes_for_wave_warp(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    warp_tile_k: int,
    dtype_key: str,
    arch: str = "gfx942",
    pipeline: Optional[str] = None,
) -> List[Tuple[int, int, int]]:
    """Return union of valid (vec_a, vec_b, vec_c) across all wave/warp pairs with given warp_tile_k.

    Convenience wrapper matching the _TILE_WTILK_TO_VECS key signature:
        key = (tile_m, tile_n, tile_k, warp_tile_k)

    This takes the union over all valid wave/warp pairs whose warp_tile_k matches,
    so the result is a superset of what any single wave/warp pair would produce.

    Args:
        tile_m, tile_n, tile_k: block tile dimensions
        warp_tile_k: XDL warp tile K dimension (selects dtype variant, e.g. 8 or 16 for bf16)
        dtype_key: e.g. "bf16_bf16_fp32"
        arch: GPU architecture string
        pipeline: optional pipeline constraint

    Returns:
        Sorted list of (vec_a, vec_b, vec_c) tuples (union across matching wave/warp pairs).
    """
    results: Set[Tuple[int, int, int]] = set()

    for (wave_m, wave_n, wave_k), (wt_m, wt_n, wt_k) in get_valid_wave_warp_pairs(
        tile_m, tile_n, tile_k, dtype_key, arch=arch, pipeline=pipeline
    ):
        if wt_k == warp_tile_k:
            vecs = get_valid_vec_sizes(
                tile_m, tile_n, tile_k,
                wave_m, wave_n, wave_k,
                wt_m, wt_n, wt_k,
                dtype_key, pipeline=pipeline,
            )
            results.update(vecs)

    return sorted(results)


# ---------------------------------------------------------------------------
# Dtype key inference helpers (for test infrastructure)
# ---------------------------------------------------------------------------

def dtype_keys_for_warp_tile_k(warp_tile_k: int) -> List[str]:
    """Infer plausible dtype_keys from warp_tile_k.

    Used by tests to map _TILE_WTILK_TO_VECS keys (which encode warp_tile_k
    but not dtype explicitly) back to dtype_key strings.

    warp_tile_k mapping (from warp_gemm_dispatcher.hpp):
      fp32_fp32_fp32  : warp_tile_k ∈ {4, 8, 16}
      bf16_bf16_fp32  : warp_tile_k ∈ {8, 16, 32}  (gfx942: {8,16}; gfx950: {8,16,32})
      fp16_fp16_fp32  : warp_tile_k ∈ {8, 16, 32}
      fp8_fp8_fp32    : warp_tile_k ∈ {16, 32, 64, 128}
    """
    candidates = []
    if warp_tile_k in {4, 8, 16}:
        candidates.append("fp32_fp32_fp32")
    if warp_tile_k in {8, 16, 32}:
        candidates.append("bf16_bf16_fp32")
        candidates.append("fp16_fp16_fp32")
    if warp_tile_k in {16, 32, 64, 128}:
        candidates.append("fp8_fp8_fp32")
    return candidates


# ---------------------------------------------------------------------------
# Depthwise Convolution Configuration
# ---------------------------------------------------------------------------


def _ceil_div(a: int, b: int) -> int:
    """Ceiling integer division."""
    return (a + b - 1) // b


def _ceil_to_multiple(val: int, multiple: int) -> int:
    """Round up val to the nearest multiple."""
    if multiple <= 0:
        return val
    return _ceil_div(val, multiple) * multiple


@dataclass(frozen=True)
class DepthwiseConfig:
    """Configuration for a depthwise convolution kernel tile."""

    tile_h: int
    tile_w: int
    filt: int
    str_h: int
    str_w: int
    pad_h: int
    pad_w: int
    nbatch: int
    sub_h: int
    sub_w: int
    in_vec: int
    out_vec: int


def is_valid_depthwise_config(cfg: DepthwiseConfig, dtype_size: int = 2) -> bool:
    """Check all depthwise pipeline constraints for a config.

    Implements the 19 constraints from the depthwise pipeline's static_asserts
    and IsDepthwiseArgumentSupported checks.  Constraints 1-4, 10-12, 18-19 are
    auto-satisfied by construction (fixed BlockSize=64, Dilation=1, square odd
    filter, same-padding, and the LDS stride formula).

    Args:
        cfg: Depthwise configuration to validate.
        dtype_size: Bytes per element (2 for fp16/bf16, 4 for fp32).

    Returns:
        True if all constraints are satisfied.
    """
    # Constraint 5: filter must be odd
    if cfg.filt < 1 or cfg.filt % 2 != 1:
        return False

    # Constraint 6: in_vec and out_vec must be positive powers of 2
    for v in (cfg.in_vec, cfg.out_vec):
        if v <= 0 or (v & (v - 1)) != 0:
            return False

    # Constraint 7: SubTileH <= TileOutH, SubTileW <= TileOutW
    if cfg.sub_h <= 0 or cfg.sub_w <= 0:
        return False
    if cfg.sub_h > cfg.tile_h or cfg.sub_w > cfg.tile_w:
        return False

    # Constraint 9: StrideW == 1 || StrideW % 2 == 0
    if cfg.str_w != 1 and cfg.str_w % 2 != 0:
        return False

    # Constraint 15: PadW > 0
    if cfg.pad_w <= 0:
        return False

    # Derived values
    tile_in_w = cfg.tile_w * cfg.str_w
    lds_tile_h = cfg.tile_h * cfg.str_h + 2 * cfg.pad_h
    lds_tile_w = tile_in_w + 2 * cfg.pad_w

    h_repeats = _ceil_div(cfg.tile_h, cfg.sub_h)
    w_repeats = _ceil_div(cfg.tile_w, cfg.sub_w)
    total_subtiles = h_repeats * w_repeats

    # Constraint 8: TotalSubTiles <= 64 (BlockSize)
    if total_subtiles > 64:
        return False

    tile_per_wave = 64 // total_subtiles
    if tile_per_wave == 0:
        return False

    # Constraint 13: NBatch % TilePerWave == 0
    if cfg.nbatch <= 0 or cfg.nbatch % tile_per_wave != 0:
        return False

    in_vec_internal = min(cfg.in_vec, 4)

    # Constraint 14: SubTileW * StrideW % InVecInternal == 0
    if (cfg.sub_w * cfg.str_w) % in_vec_internal != 0:
        return False

    # LDS stride construction (constraints 10-12 are auto-satisfied)
    lds_stride_base = _ceil_to_multiple(lds_tile_w, cfg.in_vec)
    lds_stride_min = lds_tile_w + cfg.pad_w
    lds_stride = max(lds_stride_base,
                     _ceil_to_multiple(lds_stride_min, in_vec_internal))

    # Constraint 16: ceil(LdsTileW / InVec) <= 64
    if _ceil_div(lds_tile_w, cfg.in_vec) > 64:
        return False

    # Constraint 17: SmemSize <= 65536 bytes
    lds_tile_size = lds_tile_h * lds_stride
    smem_size = lds_tile_size * tile_per_wave * dtype_size
    if smem_size > 65536:
        return False

    return True


def get_valid_depthwise_configs(
    tile_sizes: List[Tuple[int, int]],
    filter_sizes: List[int],
    strides: List[Tuple[int, int]],
    block_size: int = 64,
    dtype_size: int = 2,
) -> List[DepthwiseConfig]:
    """Generate all valid depthwise configs from parameter space.

    For each combination of tile, filter, and stride, enumerates valid
    sub-tile, batch, and vector configurations.  Padding is derived from
    filter size as standard "same" padding: pad = (filt - 1) // 2.

    Args:
        tile_sizes: List of (tile_h, tile_w) output tile dimensions.
        filter_sizes: List of square filter sizes (must be odd).
        strides: List of (stride_h, stride_w) pairs.
        block_size: Thread block size (default 64).
        dtype_size: Bytes per element (default 2 for fp16/bf16).

    Returns:
        List of valid DepthwiseConfig objects (deduplicated).
    """
    configs: List[DepthwiseConfig] = []
    seen: Set[DepthwiseConfig] = set()
    vec_values = [1, 2, 4, 8]

    for tile_h, tile_w in tile_sizes:
        # Sub-tile candidates: powers of 2 + divisors of tile dimension
        sub_h_set: Set[int] = set()
        sub_w_set: Set[int] = set()
        v = 1
        while v <= tile_h:
            sub_h_set.add(v)
            v *= 2
        for d in _pos_divisors(tile_h):
            sub_h_set.add(d)
        v = 1
        while v <= tile_w:
            sub_w_set.add(v)
            v *= 2
        for d in _pos_divisors(tile_w):
            sub_w_set.add(d)

        sub_h_list = sorted(sub_h_set)
        sub_w_list = sorted(sub_w_set)

        for filt in filter_sizes:
            if filt % 2 != 1:
                continue
            pad = (filt - 1) // 2

            for str_h, str_w in strides:
                for sub_h in sub_h_list:
                    for sub_w in sub_w_list:
                        # Early prune on total sub-tiles
                        h_reps = _ceil_div(tile_h, sub_h)
                        w_reps = _ceil_div(tile_w, sub_w)
                        total_st = h_reps * w_reps
                        if total_st > block_size:
                            continue

                        tile_per_wave = block_size // total_st
                        if tile_per_wave == 0:
                            continue

                        # Enumerate nbatch (powers of 2, divisible by tile_per_wave)
                        nb = 1
                        while nb <= 128:
                            if nb % tile_per_wave == 0:
                                for in_vec in vec_values:
                                    for out_vec in vec_values:
                                        cfg = DepthwiseConfig(
                                            tile_h=tile_h, tile_w=tile_w,
                                            filt=filt,
                                            str_h=str_h, str_w=str_w,
                                            pad_h=pad, pad_w=pad,
                                            nbatch=nb,
                                            sub_h=sub_h, sub_w=sub_w,
                                            in_vec=in_vec, out_vec=out_vec,
                                        )
                                        if cfg not in seen and \
                                                is_valid_depthwise_config(cfg, dtype_size):
                                            seen.add(cfg)
                                            configs.append(cfg)
                            nb *= 2

    return configs
