"""
Core Python interface for CK Tile Dispatcher

Provides high-level Python API wrapping C++ dispatcher.
"""

import numpy as np
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
from enum import Enum

# Try to import C++ extension
try:
    from . import _ck_dispatcher_cpp as cpp
    HAS_CPP = True
except ImportError:
    HAS_CPP = False
    import warnings
    warnings.warn("C++ extension not available. Using Python fallback.")


# ============================================================================
# Enums
# ============================================================================

class DataType(Enum):
    """Data types supported by dispatcher"""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    BF8 = "bf8"
    INT8 = "int8"
    INT32 = "int32"
    
    @classmethod
    def from_numpy(cls, dtype):
        """Convert from numpy dtype"""
        mapping = {
            np.float32: cls.FP32,
            np.float16: cls.FP16,
            np.int8: cls.INT8,
            np.int32: cls.INT32,
        }
        return mapping.get(dtype, cls.FP32)
    
    def to_numpy(self):
        """Convert to numpy dtype"""
        mapping = {
            self.FP32: np.float32,
            self.FP16: np.float16,
            self.INT8: np.int8,
            self.INT32: np.int32,
        }
        return mapping.get(self, np.float32)


class LayoutTag(Enum):
    """Memory layout tags"""
    ROW_MAJOR = "row"
    COL_MAJOR = "col"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Problem:
    """GEMM problem specification"""
    M: int
    N: int
    K: int
    
    # Pointers (can be numpy arrays or device pointers)
    A: Optional[Union[np.ndarray, int]] = None
    B: Optional[Union[np.ndarray, int]] = None
    C: Optional[Union[np.ndarray, int]] = None
    
    # Data types
    dtype_a: DataType = DataType.FP16
    dtype_b: DataType = DataType.FP16
    dtype_c: DataType = DataType.FP16
    
    # Layouts
    layout_a: LayoutTag = LayoutTag.ROW_MAJOR
    layout_b: LayoutTag = LayoutTag.COL_MAJOR
    layout_c: LayoutTag = LayoutTag.ROW_MAJOR
    
    # Optional parameters
    batch_size: int = 1
    alpha: float = 1.0
    beta: float = 0.0
    
    def validate(self) -> Tuple[bool, str]:
        """Validate problem specification"""
        if self.M <= 0 or self.N <= 0 or self.K <= 0:
            return False, "Dimensions must be positive"
        
        if self.batch_size <= 0:
            return False, "Batch size must be positive"
        
        return True, "Valid"
    
    def __repr__(self):
        return f"Problem(M={self.M}, N={self.N}, K={self.K}, batch={self.batch_size})"


@dataclass
class KernelKey:
    """Kernel configuration key"""
    dtype_a: DataType
    dtype_b: DataType
    dtype_c: DataType
    layout_a: LayoutTag
    layout_b: LayoutTag
    layout_c: LayoutTag
    tile_m: int
    tile_n: int
    tile_k: int
    
    def __repr__(self):
        return (f"KernelKey({self.dtype_a.value}, "
                f"tile={self.tile_m}x{self.tile_n}x{self.tile_k})")


@dataclass
class DispatchResult:
    """Result of kernel dispatch"""
    success: bool
    kernel_name: str
    execution_time_ms: float = 0.0
    gflops: float = 0.0
    error_message: str = ""
    
    def __repr__(self):
        if self.success:
            return f"DispatchResult(✓ {self.kernel_name}, {self.gflops:.2f} GFLOPS)"
        else:
            return f"DispatchResult(✗ {self.error_message})"


# ============================================================================
# Dispatcher Class
# ============================================================================

class Dispatcher:
    """
    Main dispatcher class
    
    Example:
        >>> dispatcher = Dispatcher()
        >>> dispatcher.register_kernels("fp16_rcr_essential")
        >>> result = dispatcher.gemm(A, B)
    """
    
    def __init__(self, gpu_arch: str = "gfx942"):
        """
        Initialize dispatcher
        
        Args:
            gpu_arch: Target GPU architecture (default: gfx942)
        """
        self.gpu_arch = gpu_arch
        self.registered_kernels = []
        
        if HAS_CPP:
            self._cpp_dispatcher = cpp.Dispatcher(gpu_arch)
        else:
            self._cpp_dispatcher = None
    
    def register_kernels(self, kernel_set: str = "fp16_rcr_essential"):
        """
        Register a set of kernels
        
        Args:
            kernel_set: Name of kernel set to register
                Options: fp16_rcr_essential, fp16_rcr_compute, etc.
        """
        if HAS_CPP:
            self._cpp_dispatcher.register_kernels(kernel_set)
        
        self.registered_kernels.append(kernel_set)
        print(f"✓ Registered kernel set: {kernel_set}")
    
    def dispatch(self, problem: Problem) -> DispatchResult:
        """
        Dispatch a GEMM problem
        
        Args:
            problem: Problem specification
        
        Returns:
            DispatchResult with execution info
        """
        # Validate problem
        valid, msg = problem.validate()
        if not valid:
            return DispatchResult(
                success=False,
                kernel_name="",
                error_message=msg
            )
        
        if HAS_CPP:
            # Use C++ dispatcher
            result = self._cpp_dispatcher.dispatch(problem)
            return result
        else:
            # Fallback: use reference implementation
            return self._dispatch_reference(problem)
    
    def gemm(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: Optional[np.ndarray] = None,
        alpha: float = 1.0,
        beta: float = 0.0,
        transpose_a: bool = False,
        transpose_b: bool = False
    ) -> np.ndarray:
        """
        High-level GEMM interface
        
        Computes: C = alpha * op(A) @ op(B) + beta * C
        
        Args:
            A: Input matrix A (M x K or K x M if transposed)
            B: Input matrix B (K x N or N x K if transposed)
            C: Output matrix C (M x N), allocated if None
            alpha: Scalar multiplier for A @ B
            beta: Scalar multiplier for C
            transpose_a: Whether to transpose A
            transpose_b: Whether to transpose B
        
        Returns:
            Output matrix C
        """
        # Determine dimensions
        if transpose_a:
            M, K = A.shape[1], A.shape[0]
        else:
            M, K = A.shape[0], A.shape[1]
        
        if transpose_b:
            K2, N = B.shape[1], B.shape[0]
        else:
            K2, N = B.shape[0], B.shape[1]
        
        if K != K2:
            raise ValueError(f"Dimension mismatch: A has K={K}, B has K={K2}")
        
        # Allocate output if needed
        if C is None:
            C = np.zeros((M, N), dtype=A.dtype)
        
        # Create problem
        problem = Problem(
            M=M, N=N, K=K,
            A=A, B=B, C=C,
            dtype_a=DataType.from_numpy(A.dtype),
            dtype_b=DataType.from_numpy(B.dtype),
            dtype_c=DataType.from_numpy(C.dtype),
            layout_a=LayoutTag.COL_MAJOR if transpose_a else LayoutTag.ROW_MAJOR,
            layout_b=LayoutTag.COL_MAJOR if transpose_b else LayoutTag.ROW_MAJOR,
            layout_c=LayoutTag.ROW_MAJOR,
            alpha=alpha,
            beta=beta
        )
        
        # Dispatch
        result = self.dispatch(problem)
        
        if not result.success:
            raise RuntimeError(f"Dispatch failed: {result.error_message}")
        
        return C
    
    def _dispatch_reference(self, problem: Problem) -> DispatchResult:
        """Reference implementation (NumPy)"""
        import time
        
        # Convert to numpy arrays if needed
        A = problem.A if isinstance(problem.A, np.ndarray) else None
        B = problem.B if isinstance(problem.B, np.ndarray) else None
        C = problem.C if isinstance(problem.C, np.ndarray) else None
        
        if A is None or B is None or C is None:
            return DispatchResult(
                success=False,
                kernel_name="reference",
                error_message="NumPy arrays required for reference implementation"
            )
        
        # Time execution
        start = time.perf_counter()
        
        # Compute GEMM
        result = problem.alpha * (A @ B)
        if problem.beta != 0.0:
            result += problem.beta * C
        
        # Copy result
        np.copyto(C, result)
        
        end = time.perf_counter()
        time_ms = (end - start) * 1000
        
        # Calculate GFLOPS
        flops = 2.0 * problem.M * problem.N * problem.K * problem.batch_size
        gflops = flops / (time_ms * 1e6)
        
        return DispatchResult(
            success=True,
            kernel_name="numpy_reference",
            execution_time_ms=time_ms,
            gflops=gflops
        )
    
    def get_registered_kernels(self) -> List[str]:
        """Get list of registered kernel sets"""
        return self.registered_kernels.copy()
    
    def clear_cache(self):
        """Clear kernel cache"""
        if HAS_CPP:
            self._cpp_dispatcher.clear_cache()
    
    def __repr__(self):
        return f"Dispatcher(arch={self.gpu_arch}, kernels={len(self.registered_kernels)})"


# ============================================================================
# Convenience Functions
# ============================================================================

def gemm(
    A: np.ndarray,
    B: np.ndarray,
    C: Optional[np.ndarray] = None,
    **kwargs
) -> np.ndarray:
    """
    Convenience function for GEMM
    
    Example:
        >>> import ck_tile_dispatcher as ckd
        >>> C = ckd.gemm(A, B)
    """
    # Create dispatcher (cached)
    if not hasattr(gemm, '_dispatcher'):
        gemm._dispatcher = Dispatcher()
        gemm._dispatcher.register_kernels("fp16_rcr_essential")
    
    return gemm._dispatcher.gemm(A, B, C, **kwargs)


def batched_gemm(
    A: np.ndarray,
    B: np.ndarray,
    C: Optional[np.ndarray] = None,
    **kwargs
) -> np.ndarray:
    """
    Batched GEMM
    
    Args:
        A: Input tensor (batch_size, M, K)
        B: Input tensor (batch_size, K, N)
        C: Output tensor (batch_size, M, N)
    
    Returns:
        Output tensor C
    """
    if A.ndim != 3 or B.ndim != 3:
        raise ValueError("Batched GEMM requires 3D tensors")
    
    batch_size = A.shape[0]
    if B.shape[0] != batch_size:
        raise ValueError("Batch size mismatch")
    
    # Allocate output
    if C is None:
        C = np.zeros((batch_size, A.shape[1], B.shape[2]), dtype=A.dtype)
    
    # Dispatch each batch
    dispatcher = Dispatcher()
    dispatcher.register_kernels("fp16_rcr_essential")
    
    for i in range(batch_size):
        C[i] = dispatcher.gemm(A[i], B[i], C[i], **kwargs)
    
    return C

