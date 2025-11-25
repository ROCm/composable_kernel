"""
PyTorch Integration for CK Tile Dispatcher

Provides PyTorch custom operators and autograd functions.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .core import Dispatcher, Problem, DataType, LayoutTag


# Check if CUDA/ROCm is available
HAS_CUDA = torch.cuda.is_available()


# ============================================================================
# PyTorch Autograd Function
# ============================================================================


class CKTileGEMM(torch.autograd.Function):
    """
    CK Tile GEMM as PyTorch autograd function

    Supports automatic differentiation.
    """

    # Class-level dispatcher (shared across all instances)
    _dispatcher = None

    @classmethod
    def _get_dispatcher(cls):
        """Get or create dispatcher"""
        if cls._dispatcher is None:
            cls._dispatcher = Dispatcher()
            cls._dispatcher.register_kernels("fp16_rcr_essential")
        return cls._dispatcher

    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        B: torch.Tensor,
        transpose_a: bool = False,
        transpose_b: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass: C = A @ B

        Args:
            ctx: Context for backward pass
            A: Input tensor (M x K)
            B: Input tensor (K x N)
            transpose_a: Transpose A
            transpose_b: Transpose B

        Returns:
            Output tensor C (M x N)
        """
        # Save for backward
        ctx.save_for_backward(A, B)
        ctx.transpose_a = transpose_a
        ctx.transpose_b = transpose_b

        # Determine dimensions
        if transpose_a:
            M, K = A.shape[1], A.shape[0]
        else:
            M, K = A.shape

        if transpose_b:
            K2, N = B.shape[1], B.shape[0]
        else:
            K2, N = B.shape

        assert K == K2, f"Dimension mismatch: {K} != {K2}"

        # Allocate output
        C = torch.empty(M, N, dtype=A.dtype, device=A.device)

        if HAS_CUDA and A.is_cuda:
            # Use CK Tile dispatcher
            dispatcher = CKTileGEMM._get_dispatcher()

            # Create problem
            problem = Problem(
                M=M,
                N=N,
                K=K,
                A=A.data_ptr(),
                B=B.data_ptr(),
                C=C.data_ptr(),
                dtype_a=DataType.from_numpy(A.cpu().numpy().dtype),
                dtype_b=DataType.from_numpy(B.cpu().numpy().dtype),
                dtype_c=DataType.from_numpy(C.cpu().numpy().dtype),
                layout_a=LayoutTag.COL_MAJOR if transpose_a else LayoutTag.ROW_MAJOR,
                layout_b=LayoutTag.COL_MAJOR if transpose_b else LayoutTag.ROW_MAJOR,
                layout_c=LayoutTag.ROW_MAJOR,
            )

            # Dispatch
            result = dispatcher.dispatch(problem)

            if not result.success:
                # Fallback to PyTorch
                if transpose_a:
                    A = A.t()
                if transpose_b:
                    B = B.t()
                C = torch.matmul(A, B)
        else:
            # CPU fallback
            if transpose_a:
                A = A.t()
            if transpose_b:
                B = B.t()
            C = torch.matmul(A, B)

        return C

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Backward pass

        Given: dL/dC
        Compute: dL/dA, dL/dB

        Forward: C = A @ B
        Backward:
            dL/dA = dL/dC @ B^T
            dL/dB = A^T @ dL/dC
        """
        A, B = ctx.saved_tensors
        transpose_a = ctx.transpose_a
        transpose_b = ctx.transpose_b

        grad_A = grad_B = None

        if ctx.needs_input_grad[0]:
            # dL/dA = dL/dC @ B^T
            if transpose_b:
                grad_A = CKTileGEMM.apply(grad_output, B, False, False)
            else:
                grad_A = CKTileGEMM.apply(grad_output, B, False, True)

            if transpose_a:
                grad_A = grad_A.t()

        if ctx.needs_input_grad[1]:
            # dL/dB = A^T @ dL/dC
            if transpose_a:
                grad_B = CKTileGEMM.apply(A, grad_output, False, False)
            else:
                grad_B = CKTileGEMM.apply(A, grad_output, True, False)

            if transpose_b:
                grad_B = grad_B.t()

        return grad_A, grad_B, None, None


# ============================================================================
# High-Level Functions
# ============================================================================


def ck_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    transpose_a: bool = False,
    transpose_b: bool = False,
) -> torch.Tensor:
    """
    CK Tile GEMM for PyTorch

    Example:
        >>> import torch
        >>> from ck_tile_dispatcher import ck_gemm
        >>>
        >>> A = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
        >>> B = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
        >>> C = ck_gemm(A, B)

    Args:
        A: Input tensor
        B: Input tensor
        transpose_a: Transpose A
        transpose_b: Transpose B

    Returns:
        Output tensor C = A @ B
    """
    return CKTileGEMM.apply(A, B, transpose_a, transpose_b)


def ck_linear(
    input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Linear layer using CK Tile

    Example:
        >>> output = ck_linear(input, weight, bias)

    Args:
        input: Input tensor (*, in_features)
        weight: Weight tensor (out_features, in_features)
        bias: Optional bias tensor (out_features)

    Returns:
        Output tensor (*, out_features)
    """
    output = ck_gemm(input, weight, transpose_b=True)

    if bias is not None:
        output = output + bias

    return output


# ============================================================================
# PyTorch Module
# ============================================================================


class CKLinear(nn.Module):
    """
    Linear layer using CK Tile dispatcher

    Drop-in replacement for torch.nn.Linear

    Example:
        >>> import torch.nn as nn
        >>> from ck_tile_dispatcher import CKLinear
        >>>
        >>> # Replace nn.Linear with CKLinear
        >>> layer = CKLinear(1024, 2048)
        >>> output = layer(input)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        """
        Initialize linear layer

        Args:
            in_features: Size of input features
            out_features: Size of output features
            bias: If True, adds learnable bias
            device: Device to place parameters
            dtype: Data type of parameters
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, **factory_kwargs)
        )

        # Initialize bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            input: Input tensor (*, in_features)

        Returns:
            Output tensor (*, out_features)
        """
        return ck_linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class CKMLP(nn.Module):
    """
    Multi-layer perceptron using CK Tile

    Example:
        >>> mlp = CKMLP([1024, 2048, 4096, 2048])
        >>> output = mlp(input)
    """

    def __init__(
        self,
        layer_sizes: list,
        activation: str = "relu",
        dropout: float = 0.0,
        bias: bool = True,
    ):
        """
        Initialize MLP

        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
            activation: Activation function ('relu', 'gelu', 'silu')
            dropout: Dropout probability
            bias: Use bias in linear layers
        """
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            self.layers.append(CKLinear(layer_sizes[i], layer_sizes[i + 1], bias=bias))

        # Activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Apply activation (except last layer)
            if i < len(self.layers) - 1:
                x = self.activation(x)
                if self.dropout is not None:
                    x = self.dropout(x)

        return x


# ============================================================================
# Model Conversion
# ============================================================================


def convert_linear_to_ck(model: nn.Module, inplace: bool = True) -> nn.Module:
    """
    Convert all nn.Linear layers to CKLinear

    Example:
        >>> model = nn.Sequential(
        ...     nn.Linear(1024, 2048),
        ...     nn.ReLU(),
        ...     nn.Linear(2048, 1024)
        ... )
        >>> model = convert_linear_to_ck(model)

    Args:
        model: PyTorch model
        inplace: Modify model in-place

    Returns:
        Converted model
    """
    if not inplace:
        import copy

        model = copy.deepcopy(model)

    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Create CKLinear with same parameters
            ck_linear = CKLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )

            # Copy weights
            ck_linear.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                ck_linear.bias.data.copy_(module.bias.data)

            # Replace module
            setattr(model, name, ck_linear)
        else:
            # Recursively convert child modules
            convert_linear_to_ck(module, inplace=True)

    return model


# ============================================================================
# Registration
# ============================================================================


def register_ck_ops():
    """
    Register CK Tile operators with PyTorch

    Call this once at the beginning of your script.
    """
    # Register custom ops (if using TorchScript)
    try:
        torch.ops.load_library("libck_tile_dispatcher.so")
        print("✓ Registered CK Tile operators")
    except Exception as e:
        print(f"⚠ Could not register CK Tile operators: {e}")
        print("  Falling back to Python implementation")


# ============================================================================
# Benchmarking
# ============================================================================


def benchmark_vs_pytorch(
    M: int = 1024,
    N: int = 1024,
    K: int = 1024,
    num_warmup: int = 10,
    num_iterations: int = 100,
    dtype=torch.float16,
) -> dict:
    """
    Benchmark CK Tile vs PyTorch

    Example:
        >>> results = benchmark_vs_pytorch(2048, 2048, 2048)
        >>> print(f"CK Tile: {results['ck_tile_gflops']:.2f} GFLOPS")
        >>> print(f"PyTorch: {results['pytorch_gflops']:.2f} GFLOPS")
        >>> print(f"Speedup: {results['speedup']:.2f}x")

    Returns:
        Dictionary with benchmark results
    """
    import time

    if not HAS_CUDA:
        print("CUDA not available, skipping benchmark")
        return {}

    device = torch.device("cuda")

    # Create tensors
    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(K, N, device=device, dtype=dtype)

    # Warmup
    for _ in range(num_warmup):
        _ = ck_gemm(A, B)
        _ = torch.matmul(A, B)

    torch.cuda.synchronize()

    # Benchmark CK Tile
    start = time.perf_counter()
    for _ in range(num_iterations):
        C_ck = ck_gemm(A, B)
    torch.cuda.synchronize()
    ck_time = (time.perf_counter() - start) / num_iterations

    # Benchmark PyTorch
    start = time.perf_counter()
    for _ in range(num_iterations):
        C_pt = torch.matmul(A, B)
    torch.cuda.synchronize()
    pt_time = (time.perf_counter() - start) / num_iterations

    # Calculate GFLOPS
    flops = 2.0 * M * N * K
    ck_gflops = flops / (ck_time * 1e9)
    pt_gflops = flops / (pt_time * 1e9)

    # Check correctness
    max_diff = torch.max(torch.abs(C_ck - C_pt)).item()

    return {
        "ck_tile_time_ms": ck_time * 1000,
        "pytorch_time_ms": pt_time * 1000,
        "ck_tile_gflops": ck_gflops,
        "pytorch_gflops": pt_gflops,
        "speedup": pt_time / ck_time,
        "max_diff": max_diff,
        "problem_size": (M, N, K),
    }
