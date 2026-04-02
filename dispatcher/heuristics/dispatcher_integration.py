#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Dispatcher integration for ML-based kernel selection.

Bridges the trained LightGBM Predictor with the CK Tile dispatcher's
kernel selection flow. Provides heuristic functions compatible with
both the Python pre-selection pattern (08_heuristics.py style) and
the C++ HeuristicFunction signature.

Name mapping between feature engine and dispatcher KernelConfig:
    Feature engine          Dispatcher KernelConfig
    ---------------------   ----------------------
    warp_m (warps/block)    wave_m
    warp_n                  wave_n
    warp_k                  wave_k
    warp_tile_m             warp_m
    warp_tile_n             warp_n
    warp_tile_k             warp_k

Usage:
    from dispatcher_integration import create_ml_heuristic

    heuristic = create_ml_heuristic("models/gemm_universal_fp8_gfx950")
    best_spec = heuristic(M=1024, N=1024, K=1024, kernel_pool=KERNEL_POOL)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


from data_pipeline import parse_kernel_name
from predict import Predictor


LAYOUT_TO_DISPATCHER = {
    "rcr": ("row", "col", "row"),
    "rrr": ("row", "row", "row"),
    "crr": ("col", "row", "row"),
    "ccr": ("col", "col", "row"),
}

DTYPE_TO_C_DTYPE = {
    "fp8": "fp16",
    "fp16": "fp16",
    "bf16": "bf16",
    "fp32": "fp32",
}


@dataclass
class MLKernelSpec:
    """Kernel spec returned by the ML heuristic, compatible with the dispatcher
    example pattern. Carries both the feature-engine-space config and the
    dispatcher-space KernelConfig fields."""

    kernel_name: str
    predicted_tflops: float

    tile_m: int
    tile_n: int
    tile_k: int
    wave_m: int
    wave_n: int
    wave_k: int
    warp_m: int
    warp_n: int
    warp_k: int
    pipeline: str
    scheduler: str
    epilogue: str
    pad_m: bool
    pad_n: bool
    pad_k: bool
    persistent: bool


def kernel_config_to_feature_dict(kernel_name: str) -> dict:
    """Parse a tile-engine kernel name into a feature-engine-compatible dict.

    Returns a dict with fields matching what GemmUniversalFeatureEngine.extract()
    expects for the kernel parameter: tile_m/n/k, warp_m/n/k (warps per block),
    warp_tile_m/n/k, pipeline, scheduler, epilogue, pad_m/n/k, persistent.
    """
    parsed = parse_kernel_name(kernel_name)
    if not parsed:
        return {}
    parsed["kernel_name"] = kernel_name
    return parsed


def feature_dict_to_dispatcher_config(
    feat: dict, dtype: str = "fp8", arch: str = "gfx950"
) -> dict:
    """Convert a feature-engine kernel dict to dispatcher KernelConfig fields.

    Handles the naming inversion:
        feature engine warp_m   -> KernelConfig wave_m  (warps per block)
        feature engine warp_tile_m -> KernelConfig warp_m (elements per warp)
    """
    layout = feat.get("layout", "rcr")
    la, lb, lc = LAYOUT_TO_DISPATCHER.get(layout, ("row", "col", "row"))
    c_dtype = DTYPE_TO_C_DTYPE.get(dtype, dtype)

    return {
        "dtype_a": dtype,
        "dtype_b": dtype,
        "dtype_c": c_dtype,
        "dtype_acc": "fp32",
        "layout_a": la,
        "layout_b": lb,
        "layout_c": lc,
        "tile_m": feat.get("tile_m", 128),
        "tile_n": feat.get("tile_n", 128),
        "tile_k": feat.get("tile_k", 64),
        "wave_m": feat.get("warp_m", 2),
        "wave_n": feat.get("warp_n", 2),
        "wave_k": feat.get("warp_k", 1),
        "warp_m": feat.get("warp_tile_m", 32),
        "warp_n": feat.get("warp_tile_n", 32),
        "warp_k": feat.get("warp_tile_k", 16),
        "pipeline": feat.get("pipeline", "compv3"),
        "scheduler": feat.get("scheduler", "intrawave"),
        "epilogue": feat.get("epilogue", "cshuffle"),
        "pad_m": feat.get("pad_m", True),
        "pad_n": feat.get("pad_n", True),
        "pad_k": feat.get("pad_k", True),
        "gfx_arch": arch,
    }


def feature_dict_to_ml_spec(feat: dict, predicted_tflops: float = 0.0) -> MLKernelSpec:
    """Convert a feature-engine kernel dict + prediction to an MLKernelSpec."""
    return MLKernelSpec(
        kernel_name=feat.get("kernel_name", "unknown"),
        predicted_tflops=predicted_tflops,
        tile_m=feat.get("tile_m", 128),
        tile_n=feat.get("tile_n", 128),
        tile_k=feat.get("tile_k", 64),
        wave_m=feat.get("warp_m", 2),
        wave_n=feat.get("warp_n", 2),
        wave_k=feat.get("warp_k", 1),
        warp_m=feat.get("warp_tile_m", 32),
        warp_n=feat.get("warp_tile_n", 32),
        warp_k=feat.get("warp_tile_k", 16),
        pipeline=feat.get("pipeline", "compv3"),
        scheduler=feat.get("scheduler", "intrawave"),
        epilogue=feat.get("epilogue", "cshuffle"),
        pad_m=feat.get("pad_m", False),
        pad_n=feat.get("pad_n", False),
        pad_k=feat.get("pad_k", False),
        persistent=feat.get("persistent", False),
    )


def load_kernel_pool_from_binaries(bin_dir: str | Path) -> list[dict]:
    """Discover benchmark executables and parse their names into feature dicts.

    Each executable name encodes the full kernel config. This creates the
    candidate pool for the ML heuristic without needing a registry JSON export.
    """
    bin_dir = Path(bin_dir)
    configs = []
    for exe in sorted(bin_dir.glob("benchmark_gemm_universal_*")):
        name = exe.stem.replace("benchmark_", "")
        feat = kernel_config_to_feature_dict(name)
        if feat and "tile_m" in feat:
            configs.append(feat)
    return configs


def create_ml_heuristic(
    model_dir: str | Path,
    dtype: str = "fp8",
    arch: str = "gfx950",
    layout: str = "rcr",
    kernel_pool: Optional[list[dict]] = None,
    bin_dir: Optional[str | Path] = None,
):
    """Create an ML heuristic function for kernel selection.

    Returns a callable with signature:
        (M: int, N: int, K: int) -> MLKernelSpec

    The returned function scores all candidate kernels using the trained
    LightGBM regressor and returns the best one as an MLKernelSpec.

    Parameters
    ----------
    model_dir : str or Path
        Path to trained model directory (must contain model_tflops.lgbm or
        model_tflops_log_big.lgbm and feature_spec.json).
    dtype : str
        Data type for the problem (fp8, fp16, bf16).
    arch : str
        GPU architecture (gfx942, gfx950).
    layout : str
        Matrix layout (rcr, rrr, crr, ccr).
    kernel_pool : list of dict, optional
        Pre-parsed kernel configs. If None, loads from bin_dir.
    bin_dir : str or Path, optional
        Directory with benchmark executables. Used to build kernel_pool if
        kernel_pool is not provided. Defaults to /workspace/ck_tile/bin.
    """
    model_dir = Path(model_dir)
    predictor = Predictor(model_dir)

    if kernel_pool is None:
        if bin_dir is None:
            bin_dir = Path("/workspace/ck_tile/bin")
        kernel_pool = load_kernel_pool_from_binaries(bin_dir)

    if not kernel_pool:
        raise ValueError(
            "No kernel configs found. Check bin_dir or provide kernel_pool."
        )

    def heuristic(M: int, N: int, K: int) -> MLKernelSpec:
        problem = {
            "m": M,
            "n": N,
            "k": K,
            "dtype": dtype,
            "layout": layout,
            "split_k": 1,
        }

        ranked = predictor.rank_kernels(problem, kernel_pool)

        if not ranked:
            feat = kernel_pool[0]
            return feature_dict_to_ml_spec(feat, 0.0)

        best_name, best_tflops = ranked[0]
        best_feat = next(
            (kp for kp in kernel_pool if kp.get("kernel_name") == best_name),
            kernel_pool[0],
        )
        return feature_dict_to_ml_spec(best_feat, best_tflops)

    return heuristic


def create_ranked_heuristic(
    model_dir: str | Path,
    dtype: str = "fp8",
    arch: str = "gfx950",
    layout: str = "rcr",
    kernel_pool: Optional[list[dict]] = None,
    bin_dir: Optional[str | Path] = None,
    top_k: int = 5,
):
    """Create an ML heuristic that returns the top-K ranked kernel specs.

    Returns a callable with signature:
        (M: int, N: int, K: int) -> list[MLKernelSpec]

    Useful when you want fallback options if the top-1 kernel fails to build.
    """
    model_dir = Path(model_dir)
    predictor = Predictor(model_dir)

    if kernel_pool is None:
        if bin_dir is None:
            bin_dir = Path("/workspace/ck_tile/bin")
        kernel_pool = load_kernel_pool_from_binaries(bin_dir)

    name_to_feat = {kp.get("kernel_name", ""): kp for kp in kernel_pool}

    def heuristic(M: int, N: int, K: int) -> list[MLKernelSpec]:
        problem = {
            "m": M,
            "n": N,
            "k": K,
            "dtype": dtype,
            "layout": layout,
            "split_k": 1,
        }

        ranked = predictor.rank_kernels(problem, kernel_pool)
        results = []
        for name, tflops in ranked[:top_k]:
            feat = name_to_feat.get(name, kernel_pool[0])
            results.append(feature_dict_to_ml_spec(feat, tflops))
        return results

    return heuristic


def ml_spec_to_dispatcher_config(
    spec: MLKernelSpec, dtype: str = "fp8", arch: str = "gfx950"
) -> dict:
    """Convert an MLKernelSpec to a dict compatible with ctypes_utils.KernelConfig."""
    layout_a, layout_b, layout_c = "row", "col", "row"
    c_dtype = DTYPE_TO_C_DTYPE.get(dtype, dtype)

    return {
        "dtype_a": dtype,
        "dtype_b": dtype,
        "dtype_c": c_dtype,
        "dtype_acc": "fp32",
        "layout_a": layout_a,
        "layout_b": layout_b,
        "layout_c": layout_c,
        "tile_m": spec.tile_m,
        "tile_n": spec.tile_n,
        "tile_k": spec.tile_k,
        "wave_m": spec.wave_m,
        "wave_n": spec.wave_n,
        "wave_k": spec.wave_k,
        "warp_m": spec.warp_m,
        "warp_n": spec.warp_n,
        "warp_k": spec.warp_k,
        "pipeline": spec.pipeline,
        "scheduler": spec.scheduler,
        "epilogue": spec.epilogue,
        "pad_m": spec.pad_m,
        "pad_n": spec.pad_n,
        "pad_k": spec.pad_k,
        "gfx_arch": arch,
    }
