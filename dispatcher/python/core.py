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
    """
    Data types supported by dispatcher.
    Matches C++ DataType enum for full compatibility.
    """

    FP16 = "fp16"  # ck_tile::half_t
    BF16 = "bf16"  # ck_tile::bf16_t
    FP32 = "fp32"  # float
    FP64 = "fp64"  # double
    FP8 = "fp8"  # ck_tile::fp8_t (E4M3)
    BF8 = "bf8"  # ck_tile::bf8_t (E5M2)
    INT8 = "int8"  # ck_tile::int8_t
    INT4 = "int4"  # ck_tile::pk_int4_t (packed)
    INT32 = "int32"  # ck_tile::int32_t

    # Aliases for compatibility
    FP8_E4M3 = "fp8"
    FP8_E5M2 = "bf8"

    @classmethod
    def from_numpy(cls, dtype):
        """Convert from numpy dtype"""
        # Handle numpy dtype objects and type
        if hasattr(dtype, "type"):
            dtype = dtype.type
        elif hasattr(dtype, "name"):
            dtype = getattr(np, dtype.name, dtype)

        mapping = {
            np.float64: cls.FP64,
            np.float32: cls.FP32,
            np.float16: cls.FP16,
            np.int8: cls.INT8,
            np.int32: cls.INT32,
            np.int64: cls.INT32,  # Map int64 to int32
        }
        return mapping.get(dtype, cls.FP32)

    @classmethod
    def from_string(cls, s: str) -> "DataType":
        """Convert from string"""
        s = s.lower()
        mapping = {
            "fp16": cls.FP16,
            "half": cls.FP16,
            "bf16": cls.BF16,
            "bfloat16": cls.BF16,
            "fp32": cls.FP32,
            "float": cls.FP32,
            "float32": cls.FP32,
            "fp64": cls.FP64,
            "double": cls.FP64,
            "float64": cls.FP64,
            "fp8": cls.FP8,
            "fp8_e4m3": cls.FP8,
            "bf8": cls.BF8,
            "fp8_e5m2": cls.BF8,
            "int8": cls.INT8,
            "int4": cls.INT4,
            "int32": cls.INT32,
        }
        return mapping.get(s, cls.FP32)

    def to_numpy(self):
        """Convert to numpy dtype"""
        mapping = {
            DataType.FP64: np.float64,
            DataType.FP32: np.float32,
            DataType.FP16: np.float16,
            DataType.INT8: np.int8,
            DataType.INT32: np.int32,
        }
        return mapping.get(self, np.float32)

    @property
    def element_size(self) -> float:
        """Size in bytes per element"""
        sizes = {
            DataType.FP16: 2,
            DataType.BF16: 2,
            DataType.FP32: 4,
            DataType.FP64: 8,
            DataType.FP8: 1,
            DataType.BF8: 1,
            DataType.INT8: 1,
            DataType.INT4: 0.5,
            DataType.INT32: 4,
        }
        return sizes.get(self, 2)


class LayoutTag(Enum):
    """Memory layout tags"""

    ROW_MAJOR = "row"
    COL_MAJOR = "col"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class Problem:
    """
    GEMM problem specification with automatic MNK inference.

    Create a Problem in several ways:

    1. From numpy arrays (recommended):
        problem = Problem.from_arrays(A, B)  # C is optional
        problem = Problem.from_arrays(A, B, C)  # With C validation

    2. From dimensions only:
        problem = Problem.from_ab(512, 256, 256, 1024)  # A: 512x256, B: 256x1024
        problem = Problem.from_dimensions(512, 256, 256, 1024, 512, 1024)  # With C

    3. Direct MNK (legacy):
        problem = Problem(M=512, N=1024, K=256)
    """

    M: int = 0
    N: int = 0
    K: int = 0

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

    # Transpose flags
    transpose_a: bool = False
    transpose_b: bool = False

    @classmethod
    def from_arrays(
        cls,
        A: np.ndarray,
        B: np.ndarray,
        C: Optional[np.ndarray] = None,
        transpose_a: bool = False,
        transpose_b: bool = False,
        alpha: float = 1.0,
        beta: float = 0.0,
    ) -> "Problem":
        """
        Create Problem from numpy arrays with automatic MNK inference.

        For GEMM: C[M,N] = A[M,K] × B[K,N]

        Args:
            A: Input matrix A (M×K or K×M if transposed)
            B: Input matrix B (K×N or N×K if transposed)
            C: Output matrix C (M×N) - optional, used for validation
            transpose_a: Whether A is transposed
            transpose_b: Whether B is transposed
            alpha: Scalar for A×B
            beta: Scalar for C

        Returns:
            Problem with inferred dimensions

        Raises:
            ValueError: If dimensions are inconsistent

        Example:
            >>> A = np.random.randn(512, 256).astype(np.float16)
            >>> B = np.random.randn(256, 1024).astype(np.float16)
            >>> problem = Problem.from_arrays(A, B)
            >>> # Infers: M=512, N=1024, K=256
        """
        # Infer dimensions from A
        if transpose_a:
            K_from_A, M = A.shape[-2], A.shape[-1]
        else:
            M, K_from_A = A.shape[-2], A.shape[-1]

        # Infer dimensions from B
        if transpose_b:
            N, K_from_B = B.shape[-2], B.shape[-1]
        else:
            K_from_B, N = B.shape[-2], B.shape[-1]

        # Validate K dimension
        if K_from_A != K_from_B:
            raise ValueError(
                f"K dimension mismatch: A has K={K_from_A}, B has K={K_from_B}"
            )
        K = K_from_A

        # Validate C if provided
        if C is not None:
            M_from_C, N_from_C = C.shape[-2], C.shape[-1]
            if M_from_C != M:
                raise ValueError(
                    f"M dimension mismatch: A implies M={M}, C has M={M_from_C}"
                )
            if N_from_C != N:
                raise ValueError(
                    f"N dimension mismatch: B implies N={N}, C has N={N_from_C}"
                )

        # Determine batch size
        batch_size = 1
        if A.ndim == 3:
            batch_size = A.shape[0]
            if B.ndim == 3 and B.shape[0] != batch_size:
                raise ValueError(
                    f"Batch size mismatch: A has batch={batch_size}, B has batch={B.shape[0]}"
                )

        return cls(
            M=int(M),
            N=int(N),
            K=int(K),
            A=A,
            B=B,
            C=C,
            dtype_a=DataType.from_numpy(A.dtype),
            dtype_b=DataType.from_numpy(B.dtype),
            dtype_c=DataType.from_numpy(C.dtype)
            if C is not None
            else DataType.from_numpy(A.dtype),
            layout_a=LayoutTag.COL_MAJOR if transpose_a else LayoutTag.ROW_MAJOR,
            layout_b=LayoutTag.COL_MAJOR if transpose_b else LayoutTag.ROW_MAJOR,
            layout_c=LayoutTag.ROW_MAJOR,
            batch_size=batch_size,
            alpha=alpha,
            beta=beta,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
        )

    @classmethod
    def from_ab(
        cls,
        a_rows: int,
        a_cols: int,
        b_rows: int,
        b_cols: int,
        transpose_a: bool = False,
        transpose_b: bool = False,
    ) -> "Problem":
        """
        Create Problem from A and B dimensions only.

        Args:
            a_rows, a_cols: Dimensions of matrix A
            b_rows, b_cols: Dimensions of matrix B
            transpose_a: Whether A is transposed
            transpose_b: Whether B is transposed

        Returns:
            Problem with inferred dimensions

        Raises:
            ValueError: If K dimensions don't match

        Example:
            >>> problem = Problem.from_ab(512, 256, 256, 1024)
            >>> # Infers: M=512, N=1024, K=256
        """
        # Infer M, K from A
        if transpose_a:
            K_from_A, M = a_rows, a_cols
        else:
            M, K_from_A = a_rows, a_cols

        # Infer K, N from B
        if transpose_b:
            N, K_from_B = b_rows, b_cols
        else:
            K_from_B, N = b_rows, b_cols

        # Validate K
        if K_from_A != K_from_B:
            raise ValueError(
                f"K dimension mismatch: A.{'rows' if transpose_a else 'cols'}={K_from_A}, "
                f"B.{'cols' if transpose_b else 'rows'}={K_from_B}"
            )

        return cls(
            M=M, N=N, K=K_from_A, transpose_a=transpose_a, transpose_b=transpose_b
        )

    @classmethod
    def from_dimensions(
        cls,
        a_rows: int,
        a_cols: int,
        b_rows: int,
        b_cols: int,
        c_rows: int,
        c_cols: int,
        transpose_a: bool = False,
        transpose_b: bool = False,
    ) -> "Problem":
        """
        Create Problem from A, B, and C dimensions with full validation.

        Args:
            a_rows, a_cols: Dimensions of matrix A
            b_rows, b_cols: Dimensions of matrix B
            c_rows, c_cols: Dimensions of matrix C (for validation)
            transpose_a: Whether A is transposed
            transpose_b: Whether B is transposed

        Returns:
            Problem with inferred and validated dimensions

        Raises:
            ValueError: If any dimensions are inconsistent
        """
        # Get problem from A and B
        problem = cls.from_ab(a_rows, a_cols, b_rows, b_cols, transpose_a, transpose_b)

        # Validate C dimensions
        if c_rows != problem.M:
            raise ValueError(
                f"M dimension mismatch: inferred M={problem.M}, C has rows={c_rows}"
            )
        if c_cols != problem.N:
            raise ValueError(
                f"N dimension mismatch: inferred N={problem.N}, C has cols={c_cols}"
            )

        return problem

    def validate(self) -> Tuple[bool, str]:
        """Validate problem specification"""
        if self.M <= 0 or self.N <= 0 or self.K <= 0:
            return False, "Dimensions must be positive"

        if self.batch_size <= 0:
            return False, "Batch size must be positive"

        # Validate tensor sizes if arrays are provided
        if isinstance(self.A, np.ndarray):
            expected_a = self.M * self.K if not self.transpose_a else self.K * self.M
            if self.A.size != expected_a * self.batch_size:
                return (
                    False,
                    f"A tensor size mismatch: got {self.A.size}, expected {expected_a * self.batch_size}",
                )

        if isinstance(self.B, np.ndarray):
            expected_b = self.K * self.N if not self.transpose_b else self.N * self.K
            if self.B.size != expected_b * self.batch_size:
                return (
                    False,
                    f"B tensor size mismatch: got {self.B.size}, expected {expected_b * self.batch_size}",
                )

        if isinstance(self.C, np.ndarray):
            expected_c = self.M * self.N
            if self.C.size != expected_c * self.batch_size:
                return (
                    False,
                    f"C tensor size mismatch: got {self.C.size}, expected {expected_c * self.batch_size}",
                )

        return True, "Valid"

    def validate_or_raise(self):
        """Validate and raise ValueError if invalid"""
        valid, msg = self.validate()
        if not valid:
            raise ValueError(msg)

    @property
    def flops(self) -> int:
        """Total floating point operations"""
        return 2 * self.M * self.N * self.K * self.batch_size

    def __repr__(self):
        trans_str = ""
        if self.transpose_a:
            trans_str += "A^T"
        if self.transpose_b:
            trans_str += "B^T" if not trans_str else ",B^T"
        if trans_str:
            trans_str = f", trans=[{trans_str}]"
        return f"Problem(M={self.M}, N={self.N}, K={self.K}, batch={self.batch_size}{trans_str})"


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
        return (
            f"KernelKey({self.dtype_a.value}, "
            f"tile={self.tile_m}x{self.tile_n}x{self.tile_k})"
        )


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
            return DispatchResult(success=False, kernel_name="", error_message=msg)

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
        transpose_b: bool = False,
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
            M=M,
            N=N,
            K=K,
            A=A,
            B=B,
            C=C,
            dtype_a=DataType.from_numpy(A.dtype),
            dtype_b=DataType.from_numpy(B.dtype),
            dtype_c=DataType.from_numpy(C.dtype),
            layout_a=LayoutTag.COL_MAJOR if transpose_a else LayoutTag.ROW_MAJOR,
            layout_b=LayoutTag.COL_MAJOR if transpose_b else LayoutTag.ROW_MAJOR,
            layout_c=LayoutTag.ROW_MAJOR,
            alpha=alpha,
            beta=beta,
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
                error_message="NumPy arrays required for reference implementation",
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
            gflops=gflops,
        )

    def get_registered_kernels(self) -> List[str]:
        """Get list of registered kernel sets"""
        return self.registered_kernels.copy()

    def clear_cache(self):
        """Clear kernel cache"""
        if HAS_CPP:
            self._cpp_dispatcher.clear_cache()

    def __repr__(self):
        return (
            f"Dispatcher(arch={self.gpu_arch}, kernels={len(self.registered_kernels)})"
        )


# ============================================================================
# Convenience Functions
# ============================================================================


def gemm(
    A: np.ndarray, B: np.ndarray, C: Optional[np.ndarray] = None, **kwargs
) -> np.ndarray:
    """
    Convenience function for GEMM

    Example:
        >>> import ck_tile_dispatcher as ckd
        >>> C = ckd.gemm(A, B)
    """
    # Create dispatcher (cached)
    if not hasattr(gemm, "_dispatcher"):
        gemm._dispatcher = Dispatcher()
        gemm._dispatcher.register_kernels("fp16_rcr_essential")

    return gemm._dispatcher.gemm(A, B, C, **kwargs)


def batched_gemm(
    A: np.ndarray, B: np.ndarray, C: Optional[np.ndarray] = None, **kwargs
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
