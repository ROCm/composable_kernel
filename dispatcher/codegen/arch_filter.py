#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Architecture-Specific Kernel Filtering for CK Tile Dispatcher

Unified filtering mechanism for validating kernel configurations against
GPU architecture capabilities. Uses arch_specs.json as single source of truth.

Key Features:
- GPU architecture-specific warp tile and warp configuration validation
- Data type compatibility checking
- Trait combination validation (pipeline, epilogue, scheduler)
- LDS capacity validation
- Single source of truth (arch_specs.json)

Usage:
    from arch_filter import ArchFilter, get_supported_archs

    # Create filter for specific architecture
    filter = ArchFilter("gfx942")

    # Validate a kernel configuration
    is_valid = filter.is_kernel_valid(
        datatype_a="fp16", datatype_b="fp16", datatype_c="fp16",
        tile_m=256, tile_n=256, tile_k=64,
        warp_m=2, warp_n=2, warp_k=1,
        warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
        pipeline="compv4", epilogue="cshuffle", scheduler="intrawave"
    )

    # Get detailed validation results
    result = filter.validate_kernel_detailed(...)
    print(result.valid, result.errors)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OperatorType(Enum):
    """Supported operator types for kernel validation"""

    GEMM = "gemm"
    GEMM_PRESHUFFLE = "gemm_preshuffle"
    GEMM_MULTI_D = "gemm_multi_d"
    CONV_FWD = "conv_fwd"
    CONV_BWD_DATA = "conv_bwd_data"
    CONV_BWD_WEIGHT = "conv_bwd_weight"
    CONV3D_FWD = "conv3d_fwd"
    CONV3D_BWD_DATA = "conv3d_bwd_data"
    CONV3D_BWD_WEIGHT = "conv3d_bwd_weight"


# Operator-specific tile constraints
# Different operators may have different minimum tile sizes or alignment requirements
OPERATOR_TILE_CONSTRAINTS = {
    OperatorType.GEMM: {
        "min_tile_m": 16,
        "min_tile_n": 16,
        "min_tile_k": 8,
        "tile_m_alignment": 16,
        "tile_n_alignment": 16,
        "tile_k_alignment": 8,
    },
    OperatorType.GEMM_PRESHUFFLE: {
        "min_tile_m": 64,
        "min_tile_n": 64,
        "min_tile_k": 32,
        "tile_m_alignment": 32,
        "tile_n_alignment": 32,
        "tile_k_alignment": 16,
    },
    OperatorType.GEMM_MULTI_D: {
        "min_tile_m": 16,
        "min_tile_n": 16,
        "min_tile_k": 8,
        "tile_m_alignment": 16,
        "tile_n_alignment": 16,
        "tile_k_alignment": 8,
    },
    OperatorType.CONV_FWD: {
        "min_tile_m": 1,  # N dimension can be 1
        "min_tile_n": 16,  # K (output channels) should be reasonable
        "min_tile_k": 16,  # C (input channels) should be reasonable
        "tile_m_alignment": 1,
        "tile_n_alignment": 16,
        "tile_k_alignment": 16,
    },
    OperatorType.CONV_BWD_DATA: {
        "min_tile_m": 1,
        "min_tile_n": 16,  # C (input channels)
        "min_tile_k": 16,  # K (output channels)
        "tile_m_alignment": 1,
        "tile_n_alignment": 16,
        "tile_k_alignment": 16,
    },
    OperatorType.CONV_BWD_WEIGHT: {
        "min_tile_m": 16,  # K (output channels)
        "min_tile_n": 16,  # C (input channels)
        "min_tile_k": 1,  # Spatial reduction dimension
        "tile_m_alignment": 16,
        "tile_n_alignment": 16,
        "tile_k_alignment": 1,
    },
}

# Add 3D convolution constraints (same as 2D for now)
OPERATOR_TILE_CONSTRAINTS[OperatorType.CONV3D_FWD] = OPERATOR_TILE_CONSTRAINTS[
    OperatorType.CONV_FWD
]
OPERATOR_TILE_CONSTRAINTS[OperatorType.CONV3D_BWD_DATA] = OPERATOR_TILE_CONSTRAINTS[
    OperatorType.CONV_BWD_DATA
]
OPERATOR_TILE_CONSTRAINTS[OperatorType.CONV3D_BWD_WEIGHT] = OPERATOR_TILE_CONSTRAINTS[
    OperatorType.CONV_BWD_WEIGHT
]

# =============================================================================
# Import from Generated Module (Single Source of Truth)
# =============================================================================

# Try to import from the generated module (created from arch_specs.json)
try:
    from arch_specs_generated import (
        ARCH_FAMILY_MAP,
        ELEMENT_SIZE_MAP,
        WARP_SUPPORTED_COMBINATIONS,
        WARP_TILE_SUPPORTED_COMBINATIONS,
        PRESHUFFLE_WARP_TILE_SUPPORTED_COMBINATIONS,
        PRESHUFFLE_PIPELINES,
        LDS_CAPACITY_LIMITS,
        TRAIT_UNSUPPORTED_COMBINATIONS,
        DTYPE_COMBINATIONS,
    )

    _USING_GENERATED = True
except ImportError:
    # Fallback to hardcoded values if generated module not available
    logger.warning(
        "arch_specs_generated.py not found, using fallback values. "
        "Run 'python generate_arch_specs.py' to generate."
    )
    _USING_GENERATED = False

    # Fallback data (minimal subset for basic operation)
    ARCH_FAMILY_MAP = {
        "gfx90a": "cdna2",
        "gfx942": "cdna3",
        "gfx950": "cdna4",
        "gfx1201": "rdna4",
    }

    ELEMENT_SIZE_MAP = {
        "fp16": 2,
        "bf16": 2,
        "fp32": 4,
        "fp64": 8,
        "fp8": 1,
        "bf8": 1,
        "int8": 1,
        "int4": 0.5,
        "int32": 4,
    }

    WARP_SUPPORTED_COMBINATIONS = {
        "gfx90a": [[1, 4, 1], [2, 2, 1], [4, 1, 1]],
        "gfx942": [[1, 4, 1], [2, 2, 1], [4, 1, 1]],
        "gfx950": [[1, 4, 1], [2, 2, 1], [4, 1, 1]],
        "gfx1201": [[2, 4, 1], [1, 8, 1], [8, 1, 1], [4, 2, 1]],
    }

    WARP_TILE_SUPPORTED_COMBINATIONS = {
        "gfx942": {
            # Key format: A_B_Acc (e.g., fp16_fp16_fp32 = A/B are fp16, accumulator is fp32)
            # These match tile_engine's GEMM_WARP_TILE_SUPPORTED_COMBINATIONS
            "fp16_fp16_fp32": [
                [32, 32, 8],
                [16, 16, 16],
                [32, 32, 16],
                [16, 16, 32],
                [4, 64, 16],
                [64, 4, 16],
            ],
            "bf16_bf16_fp32": [
                [32, 32, 8],
                [16, 16, 16],
                [32, 32, 16],
                [16, 16, 32],
                [4, 64, 16],
                [64, 4, 16],
            ],
            "fp8_fp8_fp32": [[32, 32, 16], [32, 32, 32], [16, 16, 32], [16, 16, 64]],
            "bf8_bf8_fp32": [[32, 32, 16], [32, 32, 32], [16, 16, 64], [16, 16, 32]],
            "int8_int8_int32": [[16, 16, 32], [32, 32, 16]],
        },
    }

    # Preshuffle-specific warp tile combinations (no [4, 64, 16])
    PRESHUFFLE_WARP_TILE_SUPPORTED_COMBINATIONS = {
        "gfx942": {
            "fp16_fp16_fp32": [
                [32, 32, 8],
                [16, 16, 16],
                [32, 32, 16],
                [16, 16, 32],
                [64, 4, 16],
            ],
        },
    }

    PRESHUFFLE_PIPELINES = ["preshufflev2"]

    LDS_CAPACITY_LIMITS = {"compv4": 32768, "preshufflev2": 32768, "default": 65536}

    TRAIT_UNSUPPORTED_COMBINATIONS = {
        ("compv3", "cshuffle", "interwave"),
        ("compv3", "default", "interwave"),
        ("compv4", "cshuffle", "interwave"),
        ("compv4", "default", "interwave"),
    }

    DTYPE_COMBINATIONS = {
        "fp32_fp32": {"acc": "fp32", "notes": "Full precision"},
        "fp16_fp16": {"acc": "fp32", "notes": "Standard half precision"},
        "bf16_bf16": {"acc": "fp32", "notes": "Brain float 16"},
        "fp8_fp8": {"acc": "fp32", "notes": "FP8 E4M3"},
        "fp8_bf8": {"acc": "fp32", "notes": "Mixed FP8/BF8"},
        "bf8_fp8": {"acc": "fp32", "notes": "Mixed BF8/FP8"},
        "bf8_bf8": {"acc": "fp32", "notes": "BF8 E5M2"},
        "int8_int8": {"acc": "int32", "notes": "Integer GEMM"},
        "pk_fp4_pk_fp4": {"acc": "fp32", "notes": "Packed 4-bit float"},
    }


# =============================================================================
# GPU Family Enum (for backwards compatibility)
# =============================================================================


class GpuFamily(Enum):
    """GPU architecture families"""

    CDNA2 = "cdna2"
    CDNA3 = "cdna3"
    CDNA4 = "cdna4"
    RDNA4 = "rdna4"


# =============================================================================
# Dtype Validation Helpers
# =============================================================================


def is_dtype_combo_valid(dtype_a: str, dtype_b: str) -> bool:
    """Check if a dtype combination is valid for GEMM."""
    key = f"{dtype_a.lower()}_{dtype_b.lower()}"
    return key in DTYPE_COMBINATIONS


def get_dtype_acc(dtype_a: str, dtype_b: str) -> str:
    """Get the accumulator type for a dtype combination."""
    key = f"{dtype_a.lower()}_{dtype_b.lower()}"
    info = DTYPE_COMBINATIONS.get(key, {"acc": "fp32"})
    return info["acc"]


def get_valid_dtype_combos() -> List[str]:
    """Get list of all valid dtype combinations."""
    return list(DTYPE_COMBINATIONS.keys())


# =============================================================================
# Validation Result Types
# =============================================================================


@dataclass
class ValidationResult:
    """Result of kernel configuration validation"""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)


@dataclass
class KernelConfig:
    """Kernel configuration for validation"""

    # Data types
    datatype_a: str
    datatype_b: str
    datatype_c: str

    # Tile dimensions
    tile_m: int
    tile_n: int
    tile_k: int

    # Warp configuration
    warp_m: int
    warp_n: int
    warp_k: int

    # Warp tile dimensions
    warp_tile_m: int
    warp_tile_n: int
    warp_tile_k: int

    # Traits
    pipeline: str = "compv4"
    epilogue: str = "cshuffle"
    scheduler: str = "intrawave"

    # Layout (for whole-workgroup cover validation)
    layout: str = "rcr"

    # Operator type (affects validation rules)
    operator: OperatorType = OperatorType.GEMM

    @property
    def dtype_key(self) -> str:
        """Generate data type combination key for warp tile lookup.

        Uses accumulator dtype (not output C type) to match the format
        used in WARP_TILE_SUPPORTED_COMBINATIONS dictionaries which are
        keyed as {datatype_a}_{datatype_b}_{accumulator_dtype}.
        """
        acc_dtype = get_dtype_acc(self.datatype_a, self.datatype_b)
        return f"{self.datatype_a}_{self.datatype_b}_{acc_dtype}"


# =============================================================================
# Architecture Filter Class
# =============================================================================


class ArchFilter:
    """
    Architecture-specific kernel configuration filter.

    Validates kernel configurations against GPU architecture capabilities
    to ensure only compatible kernels are registered.

    Example:
        filter = ArchFilter("gfx942")

        # Quick validation
        if filter.is_kernel_valid(config):
            registry.register_kernel(kernel)

        # Detailed validation with error messages
        result = filter.validate_kernel(config)
        if not result.valid:
            for error in result.errors:
                print(f"Validation failed: {error}")
    """

    def __init__(self, gpu_arch: str, strict_mode: bool = True):
        """
        Initialize architecture filter.

        Args:
            gpu_arch: GPU architecture string (e.g., "gfx942", "gfx90a")
            strict_mode: If True, unknown configurations are rejected.
                        If False, unknown configurations pass with warnings.
        """
        self.gpu_arch = gpu_arch.lower()
        self.strict_mode = strict_mode
        self.family = ARCH_FAMILY_MAP.get(self.gpu_arch)

        if self.family is None and strict_mode:
            raise ValueError(
                f"Unknown GPU architecture: {gpu_arch}. "
                f"Supported: {list(ARCH_FAMILY_MAP.keys())}"
            )

    def validate_kernel(self, config: KernelConfig) -> ValidationResult:
        """
        Validate a kernel configuration against architecture constraints.

        Validation is performed based on the operator type, as different
        operators (GEMM, Conv FWD, Conv BWD) have different constraints.

        Args:
            config: Kernel configuration to validate

        Returns:
            ValidationResult with valid flag and error/warning messages
        """
        result = ValidationResult(valid=True)

        # Operator-specific tile constraint validation
        self._validate_operator_constraints(config, result)
        if not result.valid and self.strict_mode:
            return result

        # Basic sanity checks
        self._validate_dimensions(config, result)
        if not result.valid and self.strict_mode:
            return result

        # Warp configuration validation
        self._validate_warp_config(config, result)

        # Warp tile combination validation
        self._validate_warp_tile_combo(config, result)

        # Trait combination validation
        self._validate_trait_combo(config, result)

        # LDS capacity validation
        self._validate_lds_capacity(config, result)

        # Dimension alignment validation
        self._validate_dimension_alignment(config, result)

        return result

    def _validate_operator_constraints(
        self, config: KernelConfig, result: ValidationResult
    ):
        """Validate operator-specific tile constraints"""
        constraints = OPERATOR_TILE_CONSTRAINTS.get(config.operator)

        if constraints is None:
            # Unknown operator - add warning but don't fail
            result.add_warning(
                f"Unknown operator type: {config.operator}. "
                f"Skipping operator-specific validation."
            )
            return

        # Validate minimum tile sizes
        min_tile_m = constraints.get("min_tile_m", 1)
        min_tile_n = constraints.get("min_tile_n", 1)
        min_tile_k = constraints.get("min_tile_k", 1)

        if config.tile_m < min_tile_m:
            result.add_error(
                f"Operator {config.operator.value}: tile_m ({config.tile_m}) "
                f"< minimum ({min_tile_m})"
            )
        if config.tile_n < min_tile_n:
            result.add_error(
                f"Operator {config.operator.value}: tile_n ({config.tile_n}) "
                f"< minimum ({min_tile_n})"
            )
        if config.tile_k < min_tile_k:
            result.add_error(
                f"Operator {config.operator.value}: tile_k ({config.tile_k}) "
                f"< minimum ({min_tile_k})"
            )

        # Validate tile alignment
        tile_m_align = constraints.get("tile_m_alignment", 1)
        tile_n_align = constraints.get("tile_n_alignment", 1)
        tile_k_align = constraints.get("tile_k_alignment", 1)

        if tile_m_align > 1 and config.tile_m % tile_m_align != 0:
            result.add_error(
                f"Operator {config.operator.value}: tile_m ({config.tile_m}) "
                f"must be aligned to {tile_m_align}"
            )
        if tile_n_align > 1 and config.tile_n % tile_n_align != 0:
            result.add_error(
                f"Operator {config.operator.value}: tile_n ({config.tile_n}) "
                f"must be aligned to {tile_n_align}"
            )
        if tile_k_align > 1 and config.tile_k % tile_k_align != 0:
            result.add_error(
                f"Operator {config.operator.value}: tile_k ({config.tile_k}) "
                f"must be aligned to {tile_k_align}"
            )

    def is_kernel_valid(
        self,
        datatype_a: str = "fp16",
        datatype_b: str = "fp16",
        datatype_c: str = "fp16",
        tile_m: int = 256,
        tile_n: int = 256,
        tile_k: int = 64,
        warp_m: int = 2,
        warp_n: int = 2,
        warp_k: int = 1,
        warp_tile_m: int = 32,
        warp_tile_n: int = 32,
        warp_tile_k: int = 16,
        pipeline: str = "compv4",
        epilogue: str = "cshuffle",
        scheduler: str = "intrawave",
        layout: str = "rcr",
        operator: Optional[OperatorType] = None,
    ) -> bool:
        """
        Quick validation check for a kernel configuration.

        Args:
            datatype_a, datatype_b, datatype_c: Data types for A, B, C matrices
            tile_m, tile_n, tile_k: Block tile dimensions
            warp_m, warp_n, warp_k: Warp/wave configuration
            warp_tile_m, warp_tile_n, warp_tile_k: Warp tile dimensions
            pipeline, epilogue, scheduler: Kernel traits
            layout: Matrix layout (e.g., "rcr")
            operator: Operator type (GEMM, CONV_FWD, CONV_BWD_DATA, etc.)
                     Affects validation rules for tile constraints.
                     Defaults to GEMM if not specified.

        Returns:
            True if configuration is valid for this architecture
        """
        config = KernelConfig(
            datatype_a=datatype_a.lower(),
            datatype_b=datatype_b.lower(),
            datatype_c=datatype_c.lower(),
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            warp_m=warp_m,
            warp_n=warp_n,
            warp_k=warp_k,
            warp_tile_m=warp_tile_m,
            warp_tile_n=warp_tile_n,
            warp_tile_k=warp_tile_k,
            pipeline=pipeline.lower(),
            epilogue=epilogue.lower(),
            scheduler=scheduler.lower(),
            layout=layout.lower(),
            operator=operator if operator is not None else OperatorType.GEMM,
        )
        return self.validate_kernel(config).valid

    def _validate_dimensions(self, config: KernelConfig, result: ValidationResult):
        """Validate basic dimension constraints"""
        if config.tile_m <= 0 or config.tile_n <= 0 or config.tile_k <= 0:
            result.add_error(
                f"Tile dimensions must be positive: "
                f"{config.tile_m}x{config.tile_n}x{config.tile_k}"
            )

        if config.warp_m <= 0 or config.warp_n <= 0 or config.warp_k <= 0:
            result.add_error(
                f"Warp dimensions must be positive: "
                f"{config.warp_m}x{config.warp_n}x{config.warp_k}"
            )

        if (
            config.warp_tile_m <= 0
            or config.warp_tile_n <= 0
            or config.warp_tile_k <= 0
        ):
            result.add_error(
                f"Warp tile dimensions must be positive: "
                f"{config.warp_tile_m}x{config.warp_tile_n}x{config.warp_tile_k}"
            )

        # Check warp tiles fit within block tiles
        if config.warp_m * config.warp_tile_m > config.tile_m:
            result.add_error(
                f"warp_m * warp_tile_m ({config.warp_m}*{config.warp_tile_m}="
                f"{config.warp_m * config.warp_tile_m}) > tile_m ({config.tile_m})"
            )
        if config.warp_n * config.warp_tile_n > config.tile_n:
            result.add_error(
                f"warp_n * warp_tile_n ({config.warp_n}*{config.warp_tile_n}="
                f"{config.warp_n * config.warp_tile_n}) > tile_n ({config.tile_n})"
            )
        if config.warp_k * config.warp_tile_k > config.tile_k:
            result.add_error(
                f"warp_k * warp_tile_k ({config.warp_k}*{config.warp_tile_k}="
                f"{config.warp_k * config.warp_tile_k}) > tile_k ({config.tile_k})"
            )

    def _validate_warp_config(self, config: KernelConfig, result: ValidationResult):
        """Validate warp configuration against architecture"""
        allowed = WARP_SUPPORTED_COMBINATIONS.get(self.gpu_arch, [])
        current = [config.warp_m, config.warp_n, config.warp_k]

        if not allowed:
            msg = f"No warp configurations defined for {self.gpu_arch}"
            if self.strict_mode:
                result.add_error(msg)
            else:
                result.add_warning(msg)
            return

        if current not in allowed:
            result.add_error(
                f"Invalid warp configuration {current} for {self.gpu_arch}. "
                f"Allowed: {allowed}"
            )

    def _validate_warp_tile_combo(self, config: KernelConfig, result: ValidationResult):
        """Validate warp tile combination against architecture and data types"""
        # Use preshuffle-specific warp tiles for preshuffle operator
        if config.operator == OperatorType.GEMM_PRESHUFFLE:
            gpu_combos = PRESHUFFLE_WARP_TILE_SUPPORTED_COMBINATIONS.get(
                self.gpu_arch, {}
            )
            combo_source = "preshuffle"
        else:
            gpu_combos = WARP_TILE_SUPPORTED_COMBINATIONS.get(self.gpu_arch, {})
            combo_source = "standard"

        if not gpu_combos:
            msg = (
                f"No {combo_source} warp tile combinations defined for {self.gpu_arch}"
            )
            if self.strict_mode:
                result.add_error(msg)
            else:
                result.add_warning(msg)
            return

        dtype_combos = gpu_combos.get(config.dtype_key, [])
        if not dtype_combos:
            # Data type combo not explicitly listed - may still be valid
            result.add_warning(
                f"No {combo_source} warp tile combinations defined for {config.dtype_key} on {self.gpu_arch}"
            )
            return

        current = [config.warp_tile_m, config.warp_tile_n, config.warp_tile_k]
        if current not in dtype_combos:
            result.add_error(
                f"Invalid warp tile {current} for {config.dtype_key} on {self.gpu_arch} ({combo_source}). "
                f"Allowed: {dtype_combos}"
            )

    def _validate_trait_combo(self, config: KernelConfig, result: ValidationResult):
        """Validate trait (pipeline, epilogue, scheduler) combination"""
        # Preshuffle requires specific pipelines
        if config.operator == OperatorType.GEMM_PRESHUFFLE:
            if config.pipeline not in PRESHUFFLE_PIPELINES:
                result.add_error(
                    f"Preshuffle GEMM requires pipeline in {PRESHUFFLE_PIPELINES}, "
                    f"got {config.pipeline}"
                )

        # Conv backward operations only support compv3/mem pipelines
        # (compv4/compv5 have template issues: transpose_tile2d for bwd_weight,
        #  get_length for bwd_data in ck_tile kernels)
        conv_bwd_operators = {
            OperatorType.CONV_BWD_DATA,
            OperatorType.CONV_BWD_WEIGHT,
            OperatorType.CONV3D_BWD_DATA,
            OperatorType.CONV3D_BWD_WEIGHT,
        }
        conv_bwd_supported_pipelines = {"compv3", "mem"}
        if config.operator in conv_bwd_operators:
            if config.pipeline not in conv_bwd_supported_pipelines:
                result.add_error(
                    f"Conv backward operations require pipeline in "
                    f"{conv_bwd_supported_pipelines}, got {config.pipeline}. "
                    f"(compv4/compv5 have ck_tile template compatibility issues)"
                )

        combo = (config.pipeline, config.epilogue, config.scheduler)
        if combo in TRAIT_UNSUPPORTED_COMBINATIONS:
            result.add_error(
                f"Unsupported trait combination: pipeline={config.pipeline}, "
                f"epilogue={config.epilogue}, scheduler={config.scheduler}"
            )

    def _validate_lds_capacity(self, config: KernelConfig, result: ValidationResult):
        """Validate LDS (Local Data Share) memory capacity"""
        elem_size_a = ELEMENT_SIZE_MAP.get(config.datatype_a, 2)
        elem_size_b = ELEMENT_SIZE_MAP.get(config.datatype_b, 2)

        matrix_a_size = config.tile_m * config.tile_k * elem_size_a
        matrix_b_size = config.tile_n * config.tile_k * elem_size_b
        total_lds = matrix_a_size + matrix_b_size

        max_lds = LDS_CAPACITY_LIMITS.get(
            config.pipeline, LDS_CAPACITY_LIMITS["default"]
        )

        if total_lds > max_lds:
            result.add_error(
                f"LDS capacity exceeded: {total_lds} bytes > {max_lds} bytes limit. "
                f"Matrix A: {config.tile_m}x{config.tile_k}x{elem_size_a}={matrix_a_size}B, "
                f"Matrix B: {config.tile_n}x{config.tile_k}x{elem_size_b}={matrix_b_size}B"
            )

    def _validate_dimension_alignment(
        self, config: KernelConfig, result: ValidationResult
    ):
        """Validate tile dimensions are aligned with warp dimensions"""
        if config.tile_m % (config.warp_m * config.warp_tile_m) != 0:
            result.add_error(
                f"tile_m ({config.tile_m}) must be divisible by "
                f"warp_m*warp_tile_m ({config.warp_m}*{config.warp_tile_m}="
                f"{config.warp_m * config.warp_tile_m})"
            )

        if config.tile_n % (config.warp_n * config.warp_tile_n) != 0:
            result.add_error(
                f"tile_n ({config.tile_n}) must be divisible by "
                f"warp_n*warp_tile_n ({config.warp_n}*{config.warp_tile_n}="
                f"{config.warp_n * config.warp_tile_n})"
            )

        if config.tile_k % (config.warp_k * config.warp_tile_k) != 0:
            result.add_error(
                f"tile_k ({config.tile_k}) must be divisible by "
                f"warp_k*warp_tile_k ({config.warp_k}*{config.warp_tile_k}="
                f"{config.warp_k * config.warp_tile_k})"
            )

    def get_supported_warp_configs(self) -> List[List[int]]:
        """Get list of supported warp configurations for this architecture"""
        return WARP_SUPPORTED_COMBINATIONS.get(self.gpu_arch, [])

    def get_supported_warp_tiles(self, dtype_key: str) -> List[List[int]]:
        """Get list of supported warp tile configurations for given data types"""
        gpu_combos = WARP_TILE_SUPPORTED_COMBINATIONS.get(self.gpu_arch, {})
        return gpu_combos.get(dtype_key, [])

    def get_supported_datatypes(self) -> List[str]:
        """Get list of data type combinations supported on this architecture"""
        gpu_combos = WARP_TILE_SUPPORTED_COMBINATIONS.get(self.gpu_arch, {})
        return list(gpu_combos.keys())


# =============================================================================
# Registry Filter Integration
# =============================================================================


class RegistryFilter:
    """
    Filter wrapper for integrating with dispatcher Registry.

    Provides a callable interface that can be used with Registry.filter()
    or during kernel registration.

    Example:
        # Create filter for gfx942
        filter = RegistryFilter("gfx942")

        # Use with registry
        registry = Registry()
        registry.set_kernel_filter(filter)  # Auto-filter on registration

        # Or filter existing kernels
        valid_kernels = registry.filter(filter.accepts_kernel)
    """

    def __init__(self, gpu_arch: str, strict_mode: bool = False):
        """
        Initialize registry filter.

        Args:
            gpu_arch: Target GPU architecture
            strict_mode: If True, reject unknown configurations
        """
        self.arch_filter = ArchFilter(gpu_arch, strict_mode=strict_mode)
        self.gpu_arch = gpu_arch
        self._rejected_count = 0
        self._accepted_count = 0

    def accepts_kernel(self, kernel_config: Dict[str, Any]) -> bool:
        """
        Check if a kernel configuration should be accepted into the registry.

        Args:
            kernel_config: Dictionary with kernel configuration values

        Returns:
            True if kernel is valid for target architecture
        """
        try:
            is_valid = self.arch_filter.is_kernel_valid(
                datatype_a=kernel_config.get("dtype_a", "fp16"),
                datatype_b=kernel_config.get("dtype_b", "fp16"),
                datatype_c=kernel_config.get("dtype_c", "fp16"),
                tile_m=kernel_config.get("tile_m", 256),
                tile_n=kernel_config.get("tile_n", 256),
                tile_k=kernel_config.get("tile_k", 64),
                warp_m=kernel_config.get("warp_m", 2),
                warp_n=kernel_config.get("warp_n", 2),
                warp_k=kernel_config.get("warp_k", 1),
                warp_tile_m=kernel_config.get("warp_tile_m", 32),
                warp_tile_n=kernel_config.get("warp_tile_n", 32),
                warp_tile_k=kernel_config.get("warp_tile_k", 16),
                pipeline=kernel_config.get("pipeline", "compv4"),
                epilogue=kernel_config.get("epilogue", "cshuffle"),
                scheduler=kernel_config.get("scheduler", "intrawave"),
                layout=kernel_config.get("layout", "rcr"),
            )

            if is_valid:
                self._accepted_count += 1
            else:
                self._rejected_count += 1

            return is_valid

        except Exception as e:
            logger.warning(f"Error validating kernel config: {e}")
            self._rejected_count += 1
            return False

    def get_stats(self) -> Dict[str, int]:
        """Get filtering statistics"""
        return {
            "accepted": self._accepted_count,
            "rejected": self._rejected_count,
            "total": self._accepted_count + self._rejected_count,
        }

    def reset_stats(self):
        """Reset filtering statistics"""
        self._accepted_count = 0
        self._rejected_count = 0

    def __call__(self, kernel_config: Dict[str, Any]) -> bool:
        """Callable interface for use with filter functions"""
        return self.accepts_kernel(kernel_config)


# =============================================================================
# Convenience Functions
# =============================================================================


def get_supported_archs() -> List[str]:
    """Get list of all supported GPU architectures"""
    return list(ARCH_FAMILY_MAP.keys())


def get_arch_family(gpu_arch: str) -> Optional[str]:
    """Get the GPU family for an architecture"""
    family = ARCH_FAMILY_MAP.get(gpu_arch.lower())
    return family if family else None  # ARCH_FAMILY_MAP contains strings, not Enums


def create_filter_for_current_gpu() -> Optional[ArchFilter]:
    """
    Create a filter for the current GPU (auto-detect).

    Returns:
        ArchFilter for detected GPU, or None if detection fails
    """
    try:
        import subprocess

        result = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=5)

        for line in result.stdout.split("\n"):
            if "gfx" in line.lower():
                for arch in ARCH_FAMILY_MAP.keys():
                    if arch in line.lower():
                        return ArchFilter(arch)

        return None
    except Exception:
        return None


def filter_kernel_list(
    kernels: List[Dict[str, Any]], gpu_arch: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filter a list of kernel configurations for a specific architecture.

    Args:
        kernels: List of kernel configuration dictionaries
        gpu_arch: Target GPU architecture

    Returns:
        Tuple of (valid_kernels, rejected_kernels)
    """
    reg_filter = RegistryFilter(gpu_arch)
    valid = []
    rejected = []

    for kernel in kernels:
        if reg_filter.accepts_kernel(kernel):
            valid.append(kernel)
        else:
            rejected.append(kernel)

    return valid, rejected


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    # Test the filter
    print("Testing ArchFilter for gfx942...\n")

    filter_942 = ArchFilter("gfx942")

    # Test valid configuration
    print("Test 1: Valid FP16 GEMM kernel")
    result = filter_942.validate_kernel(
        KernelConfig(
            datatype_a="fp16",
            datatype_b="fp16",
            datatype_c="fp16",
            tile_m=256,
            tile_n=256,
            tile_k=64,
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            pipeline="compv4",
            epilogue="cshuffle",
            scheduler="intrawave",
        )
    )
    print(f"  Valid: {result.valid}")
    if result.errors:
        print(f"  Errors: {result.errors}")
    print()

    # Test invalid warp configuration
    print("Test 2: Invalid warp configuration")
    result = filter_942.validate_kernel(
        KernelConfig(
            datatype_a="fp16",
            datatype_b="fp16",
            datatype_c="fp16",
            tile_m=256,
            tile_n=256,
            tile_k=64,
            warp_m=3,
            warp_n=3,
            warp_k=1,  # Invalid!
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
        )
    )
    print(f"  Valid: {result.valid}")
    if result.errors:
        print(f"  Errors: {result.errors}")
    print()

    # Test LDS overflow
    print("Test 3: LDS capacity overflow")
    result = filter_942.validate_kernel(
        KernelConfig(
            datatype_a="fp16",
            datatype_b="fp16",
            datatype_c="fp16",
            tile_m=512,
            tile_n=512,
            tile_k=256,  # Too large!
            warp_m=2,
            warp_n=2,
            warp_k=1,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            pipeline="compv4",
        )
    )
    print(f"  Valid: {result.valid}")
    if result.errors:
        print(f"  Errors: {result.errors}")
    print()

    # Test quick validation
    print("Test 4: Quick validation (is_kernel_valid)")
    is_valid = filter_942.is_kernel_valid(
        tile_m=128,
        tile_n=128,
        tile_k=32,
        warp_m=2,
        warp_n=2,
        warp_k=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
    )
    print(f"  Valid: {is_valid}")
    print()

    # Show supported configurations
    print("Supported warp configurations for gfx942:")
    for cfg in filter_942.get_supported_warp_configs():
        print(f"  {cfg}")
    print()

    print("Supported data types for gfx942:")
    for dtype in filter_942.get_supported_datatypes():
        print(f"  {dtype}")
