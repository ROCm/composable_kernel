#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
CK Tile Convolution Dispatcher Utilities

Common utilities for convolution kernel specification using the
Signature/Algorithm/Arch pattern from experimental/builder/reflect.

Structure:
  - Signature: WHAT operation (types, layouts, direction, element ops)
  - Algorithm: HOW it's computed (tiles, warps, pipeline, scheduler, padding)
  - Arch:      WHERE it runs (target GPU architecture)

Usage:
    from conv_utils import (
        ConvSignature, ConvAlgorithm, ArchInfo,
        ConvKernelConfig, ConvKernelSet, ConvProblem
    )

    # Define signature (WHAT)
    sig = ConvSignature()
    sig.dtype("fp16")
    sig.layout = "nhwc"
    sig.direction = "forward"

    # Define algorithm (HOW)
    algo = ConvAlgorithm()
    algo.tile(1, 128, 128)
    algo.wave(2, 2, 1)
    algo.warp(32, 32, 16)
    algo.pipeline = "compv4"

    # Define arch (WHERE)
    arch = ArchInfo(name="gfx942")

    # Combine into config
    config = ConvKernelConfig(signature=sig, algorithm=algo, arch=arch)
"""

import ctypes
import subprocess
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


# =============================================================================
# PATH CONFIGURATION
# =============================================================================


def get_dispatcher_root() -> Path:
    """Get the dispatcher root directory"""
    # This file is in dispatcher/examples/conv/python/
    return Path(__file__).parent.parent.parent.parent


def get_ck_root() -> Path:
    """Get the CK root directory"""
    return get_dispatcher_root().parent


def get_build_dir() -> Path:
    """Get the build directory"""
    return get_dispatcher_root() / "build"


def get_generated_kernels_dir() -> Path:
    """Get the generated kernels directory"""
    return get_build_dir() / "generated_kernels"


def get_codegen_dir() -> Path:
    """Get the codegen scripts directory"""
    return get_dispatcher_root() / "codegen"


# =============================================================================
# ENUMS (matching conv_config.hpp)
# =============================================================================


class DataType(Enum):
    """Data types for convolution"""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    I8 = "i8"
    U8 = "u8"


class ConvDirection(Enum):
    """Convolution operation direction"""

    FORWARD = "forward"
    BACKWARD_DATA = "bwd_data"
    BACKWARD_WEIGHT = "bwd_weight"


class ConvLayout(Enum):
    """Memory layout for convolution tensors"""

    NHWC = "nhwc"
    NHWGC = "nhwgc"  # Grouped
    NCHW = "nchw"
    NGCHW = "ngchw"  # Grouped


class PipelineVersion(Enum):
    """Pipeline versions"""

    V3 = "compv3"
    V4 = "compv4"
    V5 = "compv5"
    MEMORY = "mem"


class PipelineScheduler(Enum):
    """Pipeline schedulers"""

    DEFAULT = "default"
    INTRAWAVE = "intrawave"
    INTERWAVE = "interwave"


class ElementwiseOp(Enum):
    """Elementwise operations"""

    PASS_THROUGH = "passthrough"
    BIAS = "bias"
    BIAS_CLAMP = "bias_clamp"
    SCALE = "scale"
    BILINEAR = "bilinear"


class ConvSpecialization(Enum):
    """Convolution specializations"""

    DEFAULT = "default"
    FILTER_1X1_PAD0 = "filter_1x1_pad0"
    FILTER_1X1_STRIDE1_PAD0 = "filter_1x1_stride1_pad0"
    FILTER_3X3 = "filter_3x3"


class GemmPadding(Enum):
    """GEMM padding modes"""

    DEFAULT = "default"
    M_PADDING = "m_padding"
    N_PADDING = "n_padding"
    K_PADDING = "k_padding"
    MN_PADDING = "mn_padding"
    MK_PADDING = "mk_padding"
    NK_PADDING = "nk_padding"
    MNK_PADDING = "mnk_padding"


# =============================================================================
# SIGNATURE: WHAT operation (types, layouts, direction)
# =============================================================================


@dataclass
class ConvSignature:
    """
    Convolution Signature - describes WHAT operation to perform.

    This groups all the "what" parameters:
      - Data types (input, weight, output, accumulator)
      - Memory layout (nhwc, nchw)
      - Operation direction (forward, backward data, backward weight)
      - Spatial dimensions (1D, 2D, 3D)
      - Grouping
      - Elementwise operations

    Attributes:
        dtype_in:       Input data type (fp16, fp32, bf16, etc.)
        dtype_wei:      Weight data type
        dtype_out:      Output data type
        dtype_acc:      Accumulator data type
        layout:         Memory layout (nhwc, nchw, nhwgc)
        direction:      Convolution direction (forward, bwd_data, bwd_weight)
        num_dims:       Spatial dimensions (1, 2, or 3)
        groups:         Number of groups for grouped convolution
        in_element_op:  Input elementwise operation
        wei_element_op: Weight elementwise operation
        out_element_op: Output elementwise operation
        specialization: Convolution specialization (default, 1x1, 3x3)
    """

    dtype_in: str = "fp16"
    dtype_wei: str = "fp16"
    dtype_out: str = "fp16"
    dtype_acc: str = "fp32"
    layout: str = "nhwc"
    direction: str = "forward"
    num_dims: int = 2
    groups: int = 1
    in_element_op: str = "passthrough"
    wei_element_op: str = "passthrough"
    out_element_op: str = "passthrough"
    specialization: str = "default"

    def dtype(
        self,
        in_type: str,
        wei_type: str = None,
        out_type: str = None,
        acc_type: str = "fp32",
    ):
        """Set all data types at once"""
        self.dtype_in = in_type
        self.dtype_wei = wei_type or in_type
        self.dtype_out = out_type or in_type
        self.dtype_acc = acc_type
        return self

    def copy(self):
        """Create a deep copy"""
        return ConvSignature(
            dtype_in=self.dtype_in,
            dtype_wei=self.dtype_wei,
            dtype_out=self.dtype_out,
            dtype_acc=self.dtype_acc,
            layout=self.layout,
            direction=self.direction,
            num_dims=self.num_dims,
            groups=self.groups,
            in_element_op=self.in_element_op,
            wei_element_op=self.wei_element_op,
            out_element_op=self.out_element_op,
            specialization=self.specialization,
        )

    def direction_short(self) -> str:
        """Get short direction string"""
        if self.direction == "forward":
            return "fwd"
        elif self.direction == "bwd_data":
            return "bwdd"
        elif self.direction == "bwd_weight":
            return "bwdw"
        return self.direction

    def __repr__(self):
        return (
            f"Signature(dtype={self.dtype_in}, layout={self.layout}, "
            f"dir={self.direction}, dims={self.num_dims}D)"
        )


# =============================================================================
# ALGORITHM: HOW it's computed (tiles, warps, pipeline, scheduler)
# =============================================================================


@dataclass
class ConvAlgorithm:
    """
    Convolution Algorithm - describes HOW the operation is computed.

    This groups all the "how" parameters:
      - Block tile dimensions
      - Warp distribution and tile sizes
      - Pipeline version and scheduler
      - Epilogue configuration
      - Padding mode

    Attributes:
        tile_n:      Block tile N dimension (batch)
        tile_k:      Block tile K dimension (output channels)
        tile_c:      Block tile C dimension (input channels)
        tile_ho:     Output tile height
        tile_wo:     Output tile width
        wave_m:      Number of warps along M dimension
        wave_n:      Number of warps along N dimension
        wave_k:      Number of warps along K dimension
        warp_m:      Warp tile M size (MPerXDL)
        warp_n:      Warp tile N size (NPerXDL)
        warp_k:      Warp tile K size
        pipeline:    Pipeline version (compv3, compv4, compv5, mem)
        scheduler:   Scheduler type (intrawave, interwave)
        epilogue:    Epilogue type (cshuffle)
        padding:     GEMM padding mode
        block_size:  Thread block size
        double_buffer: Use double buffering for LDS
    """

    tile_n: int = 1
    tile_k: int = 128
    tile_c: int = 128
    tile_ho: int = 1
    tile_wo: int = 16
    wave_m: int = 2
    wave_n: int = 2
    wave_k: int = 1
    warp_m: int = 32
    warp_n: int = 32
    warp_k: int = 16
    pipeline: str = "compv4"
    scheduler: str = "intrawave"
    epilogue: str = "cshuffle"
    padding: str = "mnk_padding"
    block_size: int = 256
    double_buffer: bool = False

    def tile(self, n: int, k: int, c: int):
        """Set block tile dimensions (N, K, C)"""
        self.tile_n = n
        self.tile_k = k
        self.tile_c = c
        return self

    def tile_output(self, ho: int, wo: int):
        """Set output spatial tile dimensions"""
        self.tile_ho = ho
        self.tile_wo = wo
        return self

    def wave(self, m: int, n: int, k: int = 1):
        """Set warp distribution across M, N, K"""
        self.wave_m = m
        self.wave_n = n
        self.wave_k = k
        return self

    def warp(self, m: int, n: int, k: int = 16):
        """Set warp tile sizes"""
        self.warp_m = m
        self.warp_n = n
        self.warp_k = k
        return self

    def copy(self):
        """Create a deep copy"""
        return ConvAlgorithm(
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            tile_c=self.tile_c,
            tile_ho=self.tile_ho,
            tile_wo=self.tile_wo,
            wave_m=self.wave_m,
            wave_n=self.wave_n,
            wave_k=self.wave_k,
            warp_m=self.warp_m,
            warp_n=self.warp_n,
            warp_k=self.warp_k,
            pipeline=self.pipeline,
            scheduler=self.scheduler,
            epilogue=self.epilogue,
            padding=self.padding,
            block_size=self.block_size,
            double_buffer=self.double_buffer,
        )

    def __repr__(self):
        return (
            f"Algorithm(tile={self.tile_k}x{self.tile_c}, "
            f"wave={self.wave_m}x{self.wave_n}, pipeline={self.pipeline})"
        )


# =============================================================================
# ARCH: WHERE it runs (target GPU)
# =============================================================================


@dataclass
class ArchInfo:
    """
    Architecture Info - describes WHERE the kernel runs.

    Attributes:
        name:             GPU architecture name (gfx942, gfx1100, etc.)
        max_waves_per_cu: Maximum waves per compute unit
        lds_size_kb:      LDS size in KB
        sgpr_count:       Number of SGPRs
        vgpr_count:       Number of VGPRs
    """

    name: str = "gfx942"
    max_waves_per_cu: int = 8
    lds_size_kb: int = 64
    sgpr_count: int = 108
    vgpr_count: int = 512

    def supports_mfma_fp16(self) -> bool:
        """Check if architecture supports FP16 MFMA"""
        return "gfx9" in self.name

    def supports_wmma(self) -> bool:
        """Check if architecture supports WMMA"""
        return "gfx11" in self.name

    def is_mi300(self) -> bool:
        """Check if MI300 series"""
        return self.name in ("gfx940", "gfx941", "gfx942")

    def is_mi200(self) -> bool:
        """Check if MI200 series"""
        return self.name in ("gfx90a",)

    def __repr__(self):
        return f"Arch({self.name})"


# =============================================================================
# COMPLETE KERNEL CONFIG (Signature + Algorithm + Arch)
# =============================================================================


@dataclass
class ConvKernelConfig:
    """
    Complete convolution kernel configuration.
    Combines Signature + Algorithm + Arch into a single config.
    """

    signature: ConvSignature = field(default_factory=ConvSignature)
    algorithm: ConvAlgorithm = field(default_factory=ConvAlgorithm)
    arch: ArchInfo = field(default_factory=ArchInfo)

    def name(self) -> str:
        """Generate unique kernel name"""
        sig = self.signature
        algo = self.algorithm
        return (
            f"conv_{sig.direction_short()}_{sig.dtype_in}_"
            f"{sig.num_dims}d_{algo.pipeline}_{algo.tile_k}x{algo.tile_c}"
        )

    def brief(self) -> str:
        """One-line summary"""
        sig = self.signature
        return f"{sig.num_dims}D {sig.direction} convolution ({sig.dtype_in})"

    def detailed(self) -> str:
        """Detailed hierarchical description"""
        sig = self.signature
        algo = self.algorithm
        arch = self.arch

        lines = [
            f"{sig.num_dims}D {sig.direction} Convolution Kernel",
            "",
            "  Signature (WHAT):",
            f"    Data Type:     {sig.dtype_in} -> {sig.dtype_out} (acc: {sig.dtype_acc})",
            f"    Layout:        {sig.layout}",
            f"    Direction:     {sig.direction}",
            f"    Spatial Dims:  {sig.num_dims}D",
            f"    Groups:        {sig.groups}",
            f"    Specialization: {sig.specialization}",
            "",
            "  Algorithm (HOW):",
            f"    Block Tile:    N={algo.tile_n}, K={algo.tile_k}, C={algo.tile_c}",
            f"    Output Tile:   Ho={algo.tile_ho}, Wo={algo.tile_wo}",
            f"    Wave Config:   {algo.wave_m}x{algo.wave_n}x{algo.wave_k}",
            f"    Warp Tile:     {algo.warp_m}x{algo.warp_n}x{algo.warp_k}",
            f"    Pipeline:      {algo.pipeline}",
            f"    Scheduler:     {algo.scheduler}",
            f"    Epilogue:      {algo.epilogue}",
            f"    Padding:       {algo.padding}",
            f"    Block Size:    {algo.block_size}",
            "",
            "  Arch (WHERE):",
            f"    Target:        {arch.name}",
            f"    MFMA FP16:     {arch.supports_mfma_fp16()}",
            f"    WMMA:          {arch.supports_wmma()}",
        ]
        return "\n".join(lines)

    def copy(self):
        """Create a deep copy"""
        return ConvKernelConfig(
            signature=self.signature.copy(),
            algorithm=self.algorithm.copy(),
            arch=ArchInfo(
                name=self.arch.name,
                max_waves_per_cu=self.arch.max_waves_per_cu,
                lds_size_kb=self.arch.lds_size_kb,
            ),
        )


# =============================================================================
# KERNEL SET (Collection of configs)
# =============================================================================


class ConvKernelSet:
    """
    Collection of convolution kernel configurations.

    Provides both simple and full APIs for adding kernels.
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.configs: List[ConvKernelConfig] = []

    def add_simple(
        self,
        dtype: str,
        layout: str,
        direction: str,
        tile_k: int,
        tile_c: int,
        arch: str = "gfx942",
    ):
        """
        Simple add with basic parameters.

        Args:
            dtype:     Data type (fp16, fp32, bf16)
            layout:    Memory layout (nhwc, nchw)
            direction: Operation direction (forward, bwd_data, bwd_weight)
            tile_k:    K tile size
            tile_c:    C tile size
            arch:      Target architecture
        """
        sig = ConvSignature()
        sig.dtype(dtype)
        sig.layout = layout
        sig.direction = direction

        algo = ConvAlgorithm()
        algo.tile_k = tile_k
        algo.tile_c = tile_c

        self.configs.append(
            ConvKernelConfig(signature=sig, algorithm=algo, arch=ArchInfo(name=arch))
        )
        return self

    def add(
        self, signature: ConvSignature, algorithm: ConvAlgorithm, arch: ArchInfo = None
    ):
        """
        Add with full Signature + Algorithm + Arch.

        Args:
            signature: ConvSignature instance
            algorithm: ConvAlgorithm instance
            arch:      ArchInfo instance (defaults to gfx942)
        """
        self.configs.append(
            ConvKernelConfig(
                signature=signature.copy(),
                algorithm=algorithm.copy(),
                arch=arch or ArchInfo(),
            )
        )
        return self

    def merge(self, other: "ConvKernelSet"):
        """Merge another kernel set into this one"""
        self.configs.extend(other.configs)
        return self

    def __len__(self):
        return len(self.configs)

    def __iter__(self):
        return iter(self.configs)

    def print(self, detailed: bool = False):
        """Print all configurations"""
        print(f"ConvKernelSet '{self.name}' ({len(self.configs)} configs):")
        for cfg in self.configs:
            if detailed:
                print(cfg.detailed())
                print()
            else:
                print(f"  - {cfg.name()}")


# =============================================================================
# CONV PROBLEM (Runtime problem specification)
# =============================================================================


@dataclass
class ConvProblem:
    """
    Convolution problem specification for runtime.

    Describes the actual sizes of a convolution to be computed.
    """

    # Batch and channels
    N: int = 1  # Batch size
    C: int = 64  # Input channels
    K: int = 128  # Output channels
    G: int = 1  # Groups

    # Spatial dimensions (2D default)
    Hi: int = 28  # Input height
    Wi: int = 28  # Input width
    Di: int = 1  # Input depth (for 3D)

    # Filter dimensions
    Y: int = 3  # Filter height
    X: int = 3  # Filter width
    Z: int = 1  # Filter depth (for 3D)

    # Stride
    stride_h: int = 1
    stride_w: int = 1
    stride_d: int = 1

    # Padding
    pad_h: int = 0
    pad_w: int = 0
    pad_d: int = 0

    # Dilation
    dilation_h: int = 1
    dilation_w: int = 1
    dilation_d: int = 1

    # Operation
    direction: str = "forward"

    @property
    def Ho(self) -> int:
        """Output height"""
        eff_y = (self.Y - 1) * self.dilation_h + 1
        return (self.Hi + 2 * self.pad_h - eff_y) // self.stride_h + 1

    @property
    def Wo(self) -> int:
        """Output width"""
        eff_x = (self.X - 1) * self.dilation_w + 1
        return (self.Wi + 2 * self.pad_w - eff_x) // self.stride_w + 1

    @property
    def Do(self) -> int:
        """Output depth (for 3D)"""
        eff_z = (self.Z - 1) * self.dilation_d + 1
        return (self.Di + 2 * self.pad_d - eff_z) // self.stride_d + 1

    @property
    def flops(self) -> float:
        """Total FLOPs for forward convolution"""
        c_per_group = self.C // self.G
        return 2.0 * self.N * self.K * self.Ho * self.Wo * c_per_group * self.Y * self.X

    @property
    def flops_3d(self) -> float:
        """Total FLOPs for 3D forward convolution"""
        c_per_group = self.C // self.G
        return (
            2.0
            * self.N
            * self.K
            * self.Do
            * self.Ho
            * self.Wo
            * c_per_group
            * self.Z
            * self.Y
            * self.X
        )

    def is_pointwise(self) -> bool:
        """Check if 1x1 convolution"""
        return self.Y == 1 and self.X == 1 and self.Z == 1

    def is_depthwise(self) -> bool:
        """Check if depthwise convolution"""
        return self.G == self.C == self.K

    def is_3d(self) -> bool:
        """Check if 3D convolution"""
        return self.Di > 1 or self.Z > 1

    def input_size(self) -> Tuple[int, ...]:
        """Get input tensor size (N, C, D, H, W) or (N, C, H, W)"""
        if self.is_3d():
            return (self.N, self.C, self.Di, self.Hi, self.Wi)
        return (self.N, self.C, self.Hi, self.Wi)

    def output_size(self) -> Tuple[int, ...]:
        """Get output tensor size"""
        if self.is_3d():
            return (self.N, self.K, self.Do, self.Ho, self.Wo)
        return (self.N, self.K, self.Ho, self.Wo)

    def filter_size(self) -> Tuple[int, ...]:
        """Get filter tensor size"""
        c_per_group = self.C // self.G
        if self.is_3d():
            return (self.K, c_per_group, self.Z, self.Y, self.X)
        return (self.K, c_per_group, self.Y, self.X)

    def __repr__(self):
        if self.is_3d():
            return (
                f"ConvProblem(N={self.N}, C={self.C}, K={self.K}, "
                f"Di={self.Di}, Hi={self.Hi}, Wi={self.Wi}, "
                f"Z={self.Z}, Y={self.Y}, X={self.X})"
            )
        return (
            f"ConvProblem(N={self.N}, C={self.C}, K={self.K}, "
            f"Hi={self.Hi}, Wi={self.Wi}, Y={self.Y}, X={self.X})"
        )


# =============================================================================
# CODEGEN RUNNER
# =============================================================================


class ConvCodegenRunner:
    """
    Runner for convolution kernel code generation.

    Generates kernels using unified_conv_codegen.py.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.codegen_script = get_codegen_dir() / "unified_conv_codegen.py"
        self.output_dir = get_generated_kernels_dir()

    def generate(self, config: ConvKernelConfig) -> Optional[Path]:
        """Generate a single kernel from config"""
        sig = config.signature
        algo = config.algorithm
        arch = config.arch

        cmd = [
            "python3",
            str(self.codegen_script),
            "--dtype",
            sig.dtype_in,
            "--layout",
            sig.layout,
            "--conv-type",
            sig.direction,
            "--spatial-dims",
            str(sig.num_dims),
            "--tile-k",
            str(algo.tile_k),
            "--tile-c",
            str(algo.tile_c),
            "--wave-m",
            str(algo.wave_m),
            "--wave-n",
            str(algo.wave_n),
            "--pipeline",
            algo.pipeline,
            "--scheduler",
            algo.scheduler,
            "--arch",
            arch.name,
            "--output-dir",
            str(self.output_dir),
        ]

        if self.verbose:
            print(f"  Generating: {config.name()}")

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Find generated file
            pattern = f"conv_{sig.direction_short()}_{sig.dtype_in}_*.hpp"
            files = list(self.output_dir.glob(pattern))
            return files[0] if files else None

        except subprocess.CalledProcessError as e:
            if self.verbose:
                print(f"  Error: {e.stderr}")
            return None

    def generate_set(
        self, kernel_set: ConvKernelSet, parallel: bool = True
    ) -> List[Path]:
        """Generate all kernels in a set"""
        generated = []

        if parallel and len(kernel_set) > 1:
            max_workers = min(len(kernel_set), multiprocessing.cpu_count())
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.generate, cfg): cfg for cfg in kernel_set
                }
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        generated.append(result)
        else:
            for cfg in kernel_set:
                result = self.generate(cfg)
                if result:
                    generated.append(result)

        return generated


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================


class ConvValidator:
    """Validation utilities for convolution results"""

    def __init__(self, rtol: float = 1e-3, atol: float = 1e-3):
        self.rtol = rtol
        self.atol = atol

    def check(self, result: np.ndarray, reference: np.ndarray) -> Dict[str, Any]:
        """Compare result against reference"""
        if result.shape != reference.shape:
            return {
                "passed": False,
                "error": f"Shape mismatch: {result.shape} vs {reference.shape}",
            }

        abs_diff = np.abs(result - reference)
        max_abs_diff = np.max(abs_diff)

        ref_norm = np.linalg.norm(reference.flatten())
        rel_diff = max_abs_diff / (ref_norm + 1e-10)

        passed = np.allclose(result, reference, rtol=self.rtol, atol=self.atol)

        return {
            "passed": passed,
            "max_abs_diff": float(max_abs_diff),
            "rel_diff": float(rel_diff),
            "rtol": self.rtol,
            "atol": self.atol,
        }

    def reference_conv2d_forward(
        self,
        input: np.ndarray,
        weight: np.ndarray,
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
    ) -> np.ndarray:
        """CPU reference for 2D forward convolution (NHWC layout)"""
        N, Hi, Wi, C = input.shape
        K, Y, X, _ = weight.shape

        pad_h, pad_w = padding
        stride_h, stride_w = stride

        # Pad input
        if pad_h > 0 or pad_w > 0:
            input = np.pad(input, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)))

        Ho = (Hi + 2 * pad_h - Y) // stride_h + 1
        Wo = (Wi + 2 * pad_w - X) // stride_w + 1

        output = np.zeros((N, Ho, Wo, K), dtype=input.dtype)

        for n in range(N):
            for ho in range(Ho):
                for wo in range(Wo):
                    for k in range(K):
                        for y in range(Y):
                            for x in range(X):
                                for c in range(C):
                                    hi = ho * stride_h + y
                                    wi = wo * stride_w + x
                                    output[n, ho, wo, k] += (
                                        input[n, hi, wi, c] * weight[k, y, x, c]
                                    )

        return output


# =============================================================================
# C STRUCTURE FOR CTYPES
# =============================================================================


class ConvProblemC(ctypes.Structure):
    """C structure matching ConvProblemC in conv_ctypes_lib.cpp"""

    _fields_ = [
        ("N", ctypes.c_int),
        ("G", ctypes.c_int),
        ("C", ctypes.c_int),
        ("K", ctypes.c_int),
        ("input_d", ctypes.c_int),
        ("input_h", ctypes.c_int),
        ("input_w", ctypes.c_int),
        ("filter_z", ctypes.c_int),
        ("filter_y", ctypes.c_int),
        ("filter_x", ctypes.c_int),
        ("stride_d", ctypes.c_int),
        ("stride_h", ctypes.c_int),
        ("stride_w", ctypes.c_int),
        ("pad_d", ctypes.c_int),
        ("pad_h", ctypes.c_int),
        ("pad_w", ctypes.c_int),
        ("dilation_d", ctypes.c_int),
        ("dilation_h", ctypes.c_int),
        ("dilation_w", ctypes.c_int),
        ("direction", ctypes.c_int),  # 0=forward, 1=bwd_data, 2=bwd_weight
    ]

    @classmethod
    def from_problem(cls, p: "ConvProblem") -> "ConvProblemC":
        """Create C struct from Python ConvProblem"""
        c = cls()
        c.N = p.N
        c.G = p.G
        c.C = p.C
        c.K = p.K
        c.input_d = p.Di
        c.input_h = p.Hi
        c.input_w = p.Wi
        c.filter_z = p.Z
        c.filter_y = p.Y
        c.filter_x = p.X
        c.stride_d = p.stride_d
        c.stride_h = p.stride_h
        c.stride_w = p.stride_w
        c.pad_d = p.pad_d
        c.pad_h = p.pad_h
        c.pad_w = p.pad_w
        c.dilation_d = p.dilation_d
        c.dilation_h = p.dilation_h
        c.dilation_w = p.dilation_w
        direction_map = {"forward": 0, "bwd_data": 1, "bwd_weight": 2}
        c.direction = direction_map.get(p.direction, 0)
        return c


# =============================================================================
# LIBRARY LOADING (for compiled kernels)
# =============================================================================


class ConvDispatcherLib:
    """
    Wrapper for the convolution dispatcher dynamic library.

    Provides Python interface to the C API in conv_ctypes_lib.cpp.

    Usage:
        lib = ConvDispatcherLib.find()
        lib.initialize()

        # Run convolution
        result = lib.run_conv(input, weight, output, problem)
    """

    SEARCH_PATHS = [
        "build/bindings/libdispatcher_conv_lib.so",
        "build/examples/libdispatcher_conv_lib.so",
        "build/lib/libdispatcher_conv.so",
        "bindings/ctypes/libdispatcher_conv_lib.so",
    ]

    def __init__(self, lib: ctypes.CDLL, path: Path):
        self._lib = lib
        self._path = path
        self._setup_functions()

    def _setup_functions(self):
        """Setup ctypes function signatures"""
        # Initialize
        self._lib.conv_dispatcher_init.argtypes = []
        self._lib.conv_dispatcher_init.restype = ctypes.c_int

        # Cleanup
        self._lib.conv_dispatcher_cleanup.argtypes = []
        self._lib.conv_dispatcher_cleanup.restype = ctypes.c_int

        # Get kernel count
        self._lib.conv_dispatcher_get_kernel_count.argtypes = []
        self._lib.conv_dispatcher_get_kernel_count.restype = ctypes.c_int

        # Version
        self._lib.conv_dispatcher_version.argtypes = []
        self._lib.conv_dispatcher_version.restype = ctypes.c_char_p

        # Has kernels
        self._lib.conv_dispatcher_has_kernels.argtypes = []
        self._lib.conv_dispatcher_has_kernels.restype = ctypes.c_int

        # Run convolution (actual GPU execution)
        self._lib.conv_dispatcher_run.argtypes = [
            ctypes.c_void_p,  # input_ptr
            ctypes.c_void_p,  # weight_ptr
            ctypes.c_void_p,  # output_ptr
            ctypes.POINTER(ConvProblemC),  # problem
            ctypes.c_void_p,  # stream
        ]
        self._lib.conv_dispatcher_run.restype = ctypes.c_float

    @property
    def path(self) -> Path:
        return self._path

    def initialize(self) -> bool:
        """Initialize the dispatcher"""
        return self._lib.conv_dispatcher_init() == 0

    def cleanup(self):
        """Cleanup dispatcher resources"""
        self._lib.conv_dispatcher_cleanup()

    def get_kernel_count(self) -> int:
        """Get number of registered kernels"""
        return self._lib.conv_dispatcher_get_kernel_count()

    def get_version(self) -> str:
        """Get library version"""
        version = self._lib.conv_dispatcher_version()
        return version.decode("utf-8") if version else "unknown"

    def has_kernels(self) -> bool:
        """Check if library was compiled with kernels"""
        return self._lib.conv_dispatcher_has_kernels() == 1

    def run(
        self,
        input_ptr: int,
        weight_ptr: int,
        output_ptr: int,
        problem: "ConvProblem",
        stream: int = 0,
    ) -> float:
        """
        Run convolution on GPU.

        Args:
            input_ptr:  Device pointer to input data
            weight_ptr: Device pointer to weight data
            output_ptr: Device pointer to output data
            problem:    ConvProblem describing the convolution
            stream:     HIP stream (0 for default)

        Returns:
            Elapsed time in milliseconds, or -1.0 on error
        """
        prob_c = ConvProblemC.from_problem(problem)
        return self._lib.conv_dispatcher_run(
            ctypes.c_void_p(input_ptr),
            ctypes.c_void_p(weight_ptr),
            ctypes.c_void_p(output_ptr),
            ctypes.byref(prob_c),
            ctypes.c_void_p(stream),
        )

    @classmethod
    def load(cls, path: str) -> "ConvDispatcherLib":
        """Load library from explicit path"""
        lib = ctypes.CDLL(path)
        return cls(lib, Path(path))

    @classmethod
    def find(cls) -> Optional["ConvDispatcherLib"]:
        """Find and load the library from common locations"""
        dispatcher_root = get_dispatcher_root()

        for rel_path in cls.SEARCH_PATHS:
            full_path = dispatcher_root / rel_path
            if full_path.exists():
                try:
                    return cls.load(str(full_path))
                except OSError:
                    continue

        return None

    @classmethod
    def auto(cls, recompile: bool = False) -> Optional["ConvDispatcherLib"]:
        """Auto-find the library and initialize it"""
        lib = cls.find()
        if lib is not None:
            lib.initialize()
            return lib
        return None


# =============================================================================
# REGISTRY AND DISPATCHER (Explicit API)
# =============================================================================


class ConvRegistry:
    """
    Convolution kernel registry - stores and manages kernel instances.

    This provides an explicit registry API that mirrors the C++ ConvRegistry class.

    Usage:
        registry = ConvRegistry()
        registry.register_kernel(kernel_config)
        dispatcher = ConvDispatcher(registry)
    """

    def __init__(self, lib: Optional[ConvDispatcherLib] = None, name: str = "default"):
        self._lib = lib
        self._name = name
        self._kernels: List[ConvKernelConfig] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def kernel_count(self) -> int:
        if self._lib:
            return self._lib.get_kernel_count()
        return len(self._kernels)

    def register_kernel(self, config: ConvKernelConfig) -> bool:
        """Register a kernel configuration."""
        self._kernels.append(config)
        return True

    def get_kernels(self) -> List[ConvKernelConfig]:
        """Get all registered kernel configs."""
        return self._kernels.copy()

    def clear(self):
        """Clear all kernels."""
        self._kernels.clear()

    def bind_library(self, lib: ConvDispatcherLib):
        """Bind to a loaded dispatcher library."""
        self._lib = lib

    def __repr__(self) -> str:
        return f"ConvRegistry(name='{self._name}', kernels={self.kernel_count})"


class ConvDispatcher:
    """
    Convolution kernel dispatcher - selects and runs kernels for problems.

    This provides an explicit dispatcher API that mirrors the C++ ConvDispatcher class.

    Usage:
        registry = ConvRegistry()
        registry.register_kernel(config)

        dispatcher = ConvDispatcher(registry)
        result = dispatcher.run(input, weight, problem)
    """

    def __init__(self, registry: ConvRegistry, lib: Optional[ConvDispatcherLib] = None):
        self._registry = registry
        self._lib = lib or registry._lib

    @property
    def registry(self) -> ConvRegistry:
        return self._registry

    def select_kernel(self, problem: ConvProblem) -> Optional[str]:
        """Select best kernel for problem."""
        # Fallback: return first matching kernel
        for config in self._registry.get_kernels():
            return config.name()
        return None

    def is_supported(self, problem: ConvProblem) -> bool:
        """Check if problem size is supported."""
        return len(self._registry.get_kernels()) > 0

    def __repr__(self) -> str:
        return f"ConvDispatcher(registry={self._registry.name}, kernels={self._registry.kernel_count})"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_conv2d_fwd_config(
    dtype: str = "fp16", tile_k: int = 128, tile_c: int = 128, arch: str = "gfx942"
) -> ConvKernelConfig:
    """Create a 2D forward convolution config"""
    sig = ConvSignature()
    sig.dtype(dtype)
    sig.layout = "nhwc"
    sig.direction = "forward"
    sig.num_dims = 2

    algo = ConvAlgorithm()
    algo.tile(1, tile_k, tile_c)
    algo.wave(2, 2, 1)
    algo.warp(32, 32, 16)
    algo.pipeline = "compv4"

    return ConvKernelConfig(signature=sig, algorithm=algo, arch=ArchInfo(name=arch))


def create_conv3d_fwd_config(
    dtype: str = "fp16", tile_k: int = 64, tile_c: int = 64, arch: str = "gfx942"
) -> ConvKernelConfig:
    """Create a 3D forward convolution config"""
    sig = ConvSignature()
    sig.dtype(dtype)
    sig.layout = "ndhwc"
    sig.direction = "forward"
    sig.num_dims = 3

    algo = ConvAlgorithm()
    algo.tile(1, tile_k, tile_c)
    algo.wave(2, 2, 1)
    algo.warp(16, 16, 32)
    algo.pipeline = "compv3"

    return ConvKernelConfig(signature=sig, algorithm=algo, arch=ArchInfo(name=arch))


def create_conv2d_bwd_data_config(
    dtype: str = "fp16", tile_k: int = 128, tile_c: int = 128, arch: str = "gfx942"
) -> ConvKernelConfig:
    """Create a 2D backward data convolution config"""
    sig = ConvSignature()
    sig.dtype(dtype)
    sig.layout = "nhwc"
    sig.direction = "bwd_data"
    sig.num_dims = 2

    algo = ConvAlgorithm()
    algo.tile(1, tile_k, tile_c)
    algo.wave(2, 2, 1)
    algo.warp(32, 32, 16)
    algo.pipeline = "compv4"

    return ConvKernelConfig(signature=sig, algorithm=algo, arch=ArchInfo(name=arch))


def create_conv2d_bwd_weight_config(
    dtype: str = "fp16", tile_k: int = 128, tile_c: int = 128, arch: str = "gfx942"
) -> ConvKernelConfig:
    """Create a 2D backward weight convolution config"""
    sig = ConvSignature()
    sig.dtype(dtype)
    sig.layout = "nhwc"
    sig.direction = "bwd_weight"
    sig.num_dims = 2

    algo = ConvAlgorithm()
    algo.tile(1, tile_k, tile_c)
    algo.wave(2, 2, 1)
    algo.warp(32, 32, 16)
    algo.pipeline = "compv4"

    return ConvKernelConfig(signature=sig, algorithm=algo, arch=ArchInfo(name=arch))


# =============================================================================
# GPU EXECUTION HELPER
# =============================================================================


class GpuConvRunner:
    """
    Simple helper for running convolution on GPU.

    Handles library loading, HIP memory management, and kernel execution.

    Usage:
        runner = GpuConvRunner()
        if runner.is_available():
            result = runner.run(input_np, weight_np, problem)
            print(f"Time: {result['time_ms']:.4f} ms")
            print(f"TFLOPS: {result['tflops']:.2f}")
    """

    def __init__(self):
        self._lib = None
        self._hip = None
        self._initialized = False
        self._init()

    def _init(self):
        """Initialize library and HIP"""
        try:
            self._lib = ConvDispatcherLib.find()
            if self._lib is None:
                return

            self._hip = ctypes.CDLL("libamdhip64.so")
            self._hip.hipMalloc.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_size_t,
            ]
            self._hip.hipMalloc.restype = ctypes.c_int
            self._hip.hipFree.argtypes = [ctypes.c_void_p]
            self._hip.hipFree.restype = ctypes.c_int
            self._hip.hipMemcpy.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            self._hip.hipMemcpy.restype = ctypes.c_int
            self._hip.hipDeviceSynchronize.argtypes = []
            self._hip.hipDeviceSynchronize.restype = ctypes.c_int

            self._lib.initialize()
            self._initialized = True
        except Exception:
            self._initialized = False

    def is_available(self) -> bool:
        """Check if GPU execution is available"""
        return self._initialized and self._lib is not None

    @property
    def library_path(self) -> Optional[str]:
        """Get library path"""
        return str(self._lib.path) if self._lib else None

    def run(
        self,
        input_np: np.ndarray,
        weight_np: np.ndarray,
        problem: ConvProblem,
        output_np: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Run convolution on GPU.

        Args:
            input_np: Input tensor (NHWGC layout)
            weight_np: Weight tensor (GKYXC layout)
            problem: ConvProblem specification
            output_np: Optional output buffer (for copy-back)

        Returns:
            Dict with 'time_ms', 'tflops', 'success', and optionally 'output'
        """
        if not self.is_available():
            return {"success": False, "error": "GPU not available"}

        try:
            # Calculate sizes
            input_size = input_np.nbytes
            weight_size = weight_np.nbytes

            # Output size depends on direction
            # Forward: output is (N, Ho, Wo, G, K)
            # Bwd_data: output is grad_input (N, Hi, Wi, G, C)
            # Bwd_weight: output is grad_weight (G, K, Y, X, C)
            direction = getattr(problem, "direction", "forward")

            if direction == "bwd_data":
                # Output is grad_input: (N, Hi, Wi, G, C)
                if hasattr(problem, "Di") and problem.Di > 0:
                    output_elements = (
                        problem.N
                        * problem.Di
                        * problem.Hi
                        * problem.Wi
                        * problem.G
                        * problem.C
                    )
                else:
                    output_elements = (
                        problem.N * problem.Hi * problem.Wi * problem.G * problem.C
                    )
            elif direction == "bwd_weight":
                # Output is grad_weight: (G, K, Y, X, C)
                if hasattr(problem, "Z") and problem.Z > 0:
                    output_elements = (
                        problem.G
                        * problem.K
                        * problem.Z
                        * problem.Y
                        * problem.X
                        * problem.C
                    )
                else:
                    output_elements = (
                        problem.G * problem.K * problem.Y * problem.X * problem.C
                    )
            else:
                # Forward: output is (N, Ho, Wo, G, K)
                if hasattr(problem, "Do") and problem.Do > 0:
                    output_elements = (
                        problem.N
                        * problem.Do
                        * problem.Ho
                        * problem.Wo
                        * problem.G
                        * problem.K
                    )
                else:
                    output_elements = (
                        problem.N * problem.Ho * problem.Wo * problem.G * problem.K
                    )

            output_size = output_elements * 2  # fp16

            # Allocate GPU memory
            input_dev = ctypes.c_void_p()
            weight_dev = ctypes.c_void_p()
            output_dev = ctypes.c_void_p()

            self._hip.hipMalloc(ctypes.byref(input_dev), input_size)
            self._hip.hipMalloc(ctypes.byref(weight_dev), weight_size)
            self._hip.hipMalloc(ctypes.byref(output_dev), output_size)

            # Copy to device
            self._hip.hipMemcpy(input_dev, input_np.ctypes.data, input_size, 1)  # H2D
            self._hip.hipMemcpy(weight_dev, weight_np.ctypes.data, weight_size, 1)

            # Run kernel
            time_ms = self._lib.run(
                input_dev.value, weight_dev.value, output_dev.value, problem
            )
            self._hip.hipDeviceSynchronize()

            # Copy back if needed
            result = {
                "success": time_ms > 0,
                "time_ms": time_ms if time_ms > 0 else 0,
                "tflops": problem.flops / (time_ms * 1e9) if time_ms > 0 else 0,
            }

            if output_np is not None and time_ms > 0:
                self._hip.hipMemcpy(
                    output_np.ctypes.data, output_dev, output_np.nbytes, 2
                )  # D2H
                result["output"] = output_np

            # Free GPU memory
            self._hip.hipFree(input_dev)
            self._hip.hipFree(weight_dev)
            self._hip.hipFree(output_dev)

            return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    def cleanup(self):
        """Cleanup resources"""
        if self._lib:
            try:
                self._lib.cleanup()
            except Exception:
                pass


def run_conv_on_gpu(
    input_np: np.ndarray, weight_np: np.ndarray, problem: ConvProblem
) -> Optional[Dict[str, Any]]:
    """
    Convenience function to run convolution on GPU.

    Returns result dict or None if GPU not available.
    """
    runner = GpuConvRunner()
    if not runner.is_available():
        return None
    result = runner.run(input_np, weight_np, problem)
    runner.cleanup()
    return result if result.get("success") else None


# =============================================================================
# TEST DATA GENERATION HELPERS
# =============================================================================


def generate_conv_test_data(
    problem: ConvProblem, dtype: str = "fp16", seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random test input and weight data for convolution.

    Args:
        problem: ConvProblem specification
        dtype: Data type ("fp16" or "fp32")
        seed: Optional random seed for reproducibility

    Returns:
        (input_np, weight_np) tuple with correctly shaped arrays
    """
    if seed is not None:
        np.random.seed(seed)

    np_dtype = np.float16 if dtype == "fp16" else np.float32

    # Determine if 2D or 3D (Di > 1 means actual 3D, Di=1 is 2D)
    is_3d = hasattr(problem, "Di") and problem.Di > 1

    if is_3d:
        # 3D: NDHWGC layout for input, GKZYXC layout for weight
        input_shape = (
            problem.N,
            problem.Di,
            problem.Hi,
            problem.Wi,
            problem.G,
            problem.C // problem.G,
        )
        weight_shape = (
            problem.G,
            problem.K // problem.G,
            problem.Z,
            problem.Y,
            problem.X,
            problem.C // problem.G,
        )
    else:
        # 2D: NHWGC layout for input, GKYXC layout for weight
        input_shape = (
            problem.N,
            problem.Hi,
            problem.Wi,
            problem.G,
            problem.C // problem.G,
        )
        weight_shape = (
            problem.G,
            problem.K // problem.G,
            problem.Y,
            problem.X,
            problem.C // problem.G,
        )

    input_np = np.random.uniform(-0.5, 0.5, input_shape).astype(np_dtype)
    weight_np = np.random.uniform(-0.5, 0.5, weight_shape).astype(np_dtype)

    return input_np, weight_np


def print_problem_info(problem: ConvProblem, title: str = "Problem"):
    """Print convolution problem information in a formatted way."""
    is_3d = hasattr(problem, "Di") and problem.Di > 1

    print(f"{title}:")
    print(f"  Batch:    N={problem.N}, G={problem.G}")
    print(f"  Channels: C={problem.C}, K={problem.K}")

    if is_3d:
        print(f"  Input:    Di={problem.Di}, Hi={problem.Hi}, Wi={problem.Wi}")
        print(f"  Filter:   Z={problem.Z}, Y={problem.Y}, X={problem.X}")
        print(f"  Output:   Do={problem.Do}, Ho={problem.Ho}, Wo={problem.Wo}")
        print(f"  FLOPs:    {problem.flops_3d:.2e}")
    else:
        print(f"  Input:    Hi={problem.Hi}, Wi={problem.Wi}")
        print(f"  Filter:   Y={problem.Y}, X={problem.X}")
        print(f"  Output:   Ho={problem.Ho}, Wo={problem.Wo}")
        print(f"  FLOPs:    {problem.flops:.2e}")


def print_gpu_result(result: Dict[str, Any], prefix: str = "  "):
    """Print GPU execution result in a formatted way."""
    if result.get("success"):
        print(f"{prefix}*** GPU EXECUTION SUCCESSFUL ***")
        print(f"{prefix}Time:   {result['time_ms']:.4f} ms")
        print(f"{prefix}TFLOPS: {result['tflops']:.2f}")
    else:
        error = result.get("error", "unknown error")
        print(f"{prefix}GPU execution failed: {error}")


# =============================================================================
# COMPLETE CONV EXECUTION HELPER
# =============================================================================


def run_conv_example(
    problem: ConvProblem,
    dtype: str = "fp16",
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Complete helper to run a convolution example end-to-end.

    Args:
        problem: ConvProblem specification
        dtype: Data type ("fp16" or "fp32")
        seed: Optional random seed
        verbose: Print progress information

    Returns:
        Dict with 'input', 'weight', 'result', 'success' keys
    """
    if verbose:
        print_problem_info(problem)
        print()

    # Generate test data
    input_np, weight_np = generate_conv_test_data(problem, dtype, seed)

    if verbose:
        print("Test Data:")
        print(f"  Input:  {input_np.shape} ({input_np.dtype})")
        print(f"  Weight: {weight_np.shape} ({weight_np.dtype})")
        print()

    # Run on GPU
    runner = GpuConvRunner()

    output = {
        "input": input_np,
        "weight": weight_np,
        "success": False,
        "result": None,
    }

    if runner.is_available():
        if verbose:
            print("GPU Execution:")
            print(f"  Library: {runner.library_path}")

        result = runner.run(input_np, weight_np, problem)
        output["result"] = result
        output["success"] = result.get("success", False)

        if verbose:
            print_gpu_result(result)

        runner.cleanup()
    else:
        if verbose:
            print("GPU library not available")

    return output


# =============================================================================
# BACKWARD WEIGHT LIBRARY (separate to avoid template conflicts)
# =============================================================================


class ConvBwdwProblemC(ctypes.Structure):
    """C structure for backward weight problem"""

    _fields_ = [
        ("N", ctypes.c_int),
        ("G", ctypes.c_int),
        ("C", ctypes.c_int),
        ("K", ctypes.c_int),
        ("input_d", ctypes.c_int),
        ("input_h", ctypes.c_int),
        ("input_w", ctypes.c_int),
        ("filter_z", ctypes.c_int),
        ("filter_y", ctypes.c_int),
        ("filter_x", ctypes.c_int),
        ("stride_d", ctypes.c_int),
        ("stride_h", ctypes.c_int),
        ("stride_w", ctypes.c_int),
        ("pad_d", ctypes.c_int),
        ("pad_h", ctypes.c_int),
        ("pad_w", ctypes.c_int),
        ("dilation_d", ctypes.c_int),
        ("dilation_h", ctypes.c_int),
        ("dilation_w", ctypes.c_int),
    ]

    @classmethod
    def from_problem(cls, p: "ConvProblem") -> "ConvBwdwProblemC":
        """Create C struct from Python ConvProblem"""
        c = cls()
        c.N = p.N
        c.G = p.G
        c.C = p.C
        c.K = p.K
        c.input_d = p.Di
        c.input_h = p.Hi
        c.input_w = p.Wi
        c.filter_z = p.Z
        c.filter_y = p.Y
        c.filter_x = p.X
        c.stride_d = p.stride_d
        c.stride_h = p.stride_h
        c.stride_w = p.stride_w
        c.pad_d = p.pad_d
        c.pad_h = p.pad_h
        c.pad_w = p.pad_w
        c.dilation_d = p.dilation_d
        c.dilation_h = p.dilation_h
        c.dilation_w = p.dilation_w
        return c


class ConvBwdWeightLib:
    """
    Wrapper for the backward weight convolution library.

    This is a SEPARATE library from the main conv library to avoid
    CK Tile template conflicts.

    Usage:
        lib = ConvBwdWeightLib.find()
        lib.initialize()
        time_ms = lib.run(input_ptr, grad_output_ptr, grad_weight_ptr, problem)
    """

    SEARCH_PATHS = [
        "build/examples/libdispatcher_conv_bwdw_lib.so",
        "build/bindings/libdispatcher_conv_bwdw_lib.so",
        "examples/build/libdispatcher_conv_bwdw_lib.so",
    ]

    def __init__(self, lib: ctypes.CDLL, path: Path):
        self._lib = lib
        self._path = path
        self._setup_functions()

    def _setup_functions(self):
        """Setup ctypes function signatures"""
        self._lib.conv_bwdw_init.argtypes = []
        self._lib.conv_bwdw_init.restype = ctypes.c_int

        self._lib.conv_bwdw_cleanup.argtypes = []
        self._lib.conv_bwdw_cleanup.restype = None

        self._lib.conv_bwdw_version.argtypes = []
        self._lib.conv_bwdw_version.restype = ctypes.c_char_p

        self._lib.conv_bwdw_has_kernels.argtypes = []
        self._lib.conv_bwdw_has_kernels.restype = ctypes.c_int

        self._lib.conv_bwdw_get_kernel_count.argtypes = []
        self._lib.conv_bwdw_get_kernel_count.restype = ctypes.c_int

        self._lib.conv_bwdw_run.argtypes = [
            ctypes.c_void_p,  # input_ptr
            ctypes.c_void_p,  # grad_output_ptr
            ctypes.c_void_p,  # grad_weight_ptr
            ctypes.POINTER(ConvBwdwProblemC),  # problem
            ctypes.c_void_p,  # stream
        ]
        self._lib.conv_bwdw_run.restype = ctypes.c_float

    @property
    def path(self) -> Path:
        return self._path

    def initialize(self) -> bool:
        """Initialize the backward weight dispatcher"""
        return self._lib.conv_bwdw_init() == 1

    def cleanup(self):
        """Cleanup resources"""
        self._lib.conv_bwdw_cleanup()

    def has_kernels(self) -> bool:
        """Check if backward weight kernels are available"""
        return self._lib.conv_bwdw_has_kernels() == 1

    def get_kernel_count(self) -> int:
        """Get number of registered kernels"""
        return self._lib.conv_bwdw_get_kernel_count()

    def run(
        self,
        input_ptr: int,
        grad_output_ptr: int,
        grad_weight_ptr: int,
        problem: "ConvProblem",
        stream: int = 0,
    ) -> float:
        """
        Run backward weight convolution on GPU.

        Args:
            input_ptr:       Device pointer to input data
            grad_output_ptr: Device pointer to gradient output (dY)
            grad_weight_ptr: Device pointer to gradient weight (dW) - OUTPUT
            problem:         ConvProblem describing the convolution
            stream:          HIP stream (0 for default)

        Returns:
            Elapsed time in milliseconds, or -1.0 on error
        """
        prob_c = ConvBwdwProblemC.from_problem(problem)
        return self._lib.conv_bwdw_run(
            ctypes.c_void_p(input_ptr),
            ctypes.c_void_p(grad_output_ptr),
            ctypes.c_void_p(grad_weight_ptr),
            ctypes.byref(prob_c),
            ctypes.c_void_p(stream),
        )

    @classmethod
    def find(cls) -> Optional["ConvBwdWeightLib"]:
        """Find and load the backward weight library"""
        script_dir = Path(__file__).parent
        dispatcher_dir = script_dir.parent.parent.parent

        search_paths = [dispatcher_dir / p for p in cls.SEARCH_PATHS] + [
            script_dir.parent.parent.parent
            / "build"
            / "examples"
            / "libdispatcher_conv_bwdw_lib.so",
        ]

        for path in search_paths:
            if path.exists():
                try:
                    lib = ctypes.CDLL(str(path))
                    return cls(lib, path)
                except OSError:
                    continue

        return None


class GpuConvBwdWeightRunner:
    """
    Runs backward weight convolution on GPU.

    Handles HIP memory allocation and the separate backward weight library.

    Usage:
        runner = GpuConvBwdWeightRunner()
        if runner.is_available():
            result = runner.run(input_np, grad_output_np, problem, grad_weight_np)
            print(f"Time: {result['time_ms']:.4f} ms")
    """

    def __init__(self):
        self._lib = None
        self._hip = None
        self._initialized = False
        self._init()

    def _init(self):
        """Initialize library and HIP"""
        try:
            self._lib = ConvBwdWeightLib.find()
            if self._lib is None:
                return

            self._lib.initialize()

            # Load HIP runtime
            try:
                self._hip = ctypes.CDLL("libamdhip64.so")
                self._hip.hipMalloc.argtypes = [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_size_t,
                ]
                self._hip.hipMalloc.restype = ctypes.c_int
                self._hip.hipFree.argtypes = [ctypes.c_void_p]
                self._hip.hipFree.restype = ctypes.c_int
                self._hip.hipMemcpy.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_int,
                ]
                self._hip.hipMemcpy.restype = ctypes.c_int
                self._hip.hipDeviceSynchronize.argtypes = []
                self._hip.hipDeviceSynchronize.restype = ctypes.c_int
            except OSError:
                self._hip = None
                return

            self._initialized = True
        except Exception:
            pass

    def is_available(self) -> bool:
        """Check if GPU backward weight is available"""
        return self._initialized and self._lib is not None and self._hip is not None

    @property
    def library_path(self) -> Optional[str]:
        """Get library path"""
        return str(self._lib.path) if self._lib else None

    def run(
        self,
        input_np: np.ndarray,
        grad_output_np: np.ndarray,
        problem: ConvProblem,
        grad_weight_np: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Run backward weight convolution on GPU.

        Args:
            input_np:       Input tensor (NHWGC layout)
            grad_output_np: Gradient output tensor (NHWGK layout)
            problem:        ConvProblem specification (with direction='bwd_weight')
            grad_weight_np: Optional output buffer for gradient weight (GKYXC layout)

        Returns:
            Dict with 'time_ms', 'tflops', 'success', and optionally 'output'
        """
        if not self.is_available():
            return {"success": False, "error": "GPU backward weight not available"}

        try:
            # Calculate sizes
            input_size = input_np.nbytes
            grad_output_size = grad_output_np.nbytes

            # Grad weight output: (G, K, Y, X, C)
            grad_weight_elements = (
                problem.G * problem.K * problem.Y * problem.X * problem.C
            )
            grad_weight_size = grad_weight_elements * 2  # fp16

            # Allocate GPU memory
            input_dev = ctypes.c_void_p()
            grad_output_dev = ctypes.c_void_p()
            grad_weight_dev = ctypes.c_void_p()

            self._hip.hipMalloc(ctypes.byref(input_dev), input_size)
            self._hip.hipMalloc(ctypes.byref(grad_output_dev), grad_output_size)
            self._hip.hipMalloc(ctypes.byref(grad_weight_dev), grad_weight_size)

            # Copy input data to device
            self._hip.hipMemcpy(input_dev, input_np.ctypes.data, input_size, 1)  # H2D
            self._hip.hipMemcpy(
                grad_output_dev, grad_output_np.ctypes.data, grad_output_size, 1
            )

            # Run kernel
            time_ms = self._lib.run(
                input_dev.value, grad_output_dev.value, grad_weight_dev.value, problem
            )
            self._hip.hipDeviceSynchronize()

            result = {
                "success": time_ms > 0,
                "time_ms": time_ms if time_ms > 0 else 0,
                "tflops": problem.flops / (time_ms * 1e9) if time_ms > 0 else 0,
            }

            # Copy back if needed
            if grad_weight_np is not None and time_ms > 0:
                self._hip.hipMemcpy(
                    grad_weight_np.ctypes.data,
                    grad_weight_dev,
                    grad_weight_np.nbytes,
                    2,
                )  # D2H
                result["output"] = grad_weight_np

            # Free GPU memory
            self._hip.hipFree(input_dev)
            self._hip.hipFree(grad_output_dev)
            self._hip.hipFree(grad_weight_dev)

            return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    def cleanup(self):
        """Cleanup resources"""
        if self._lib:
            try:
                self._lib.cleanup()
            except Exception:
                pass
