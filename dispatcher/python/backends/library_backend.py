"""
CK Library backend implementation

Wraps pre-compiled CK library kernels from DeviceOperationInstanceFactory.
"""

import time
from typing import List, Dict, Optional
import numpy as np

from .base import KernelInstance, BackendBase, BackendType


class LibraryKernelInstance(KernelInstance):
    """
    Kernel instance for CK Library pre-compiled kernels
    
    Wraps kernels from library/src/tensor_operation_instance/
    """
    
    def __init__(self, kernel_key, kernel_name: str, device_op=None):
        """
        Initialize library kernel instance
        
        Args:
            kernel_key: KernelKey object
            kernel_name: Kernel name
            device_op: Optional C++ device operation object (from bindings)
        """
        self._key = kernel_key
        self._name = kernel_name
        self._device_op = device_op
    
    def get_key(self):
        """Get kernel key"""
        return self._key
    
    def supports(self, problem) -> bool:
        """
        Check if kernel supports the problem
        
        For library kernels, delegate to IsSupportedArgument if available.
        """
        if self._device_op is not None:
            try:
                # Call C++ IsSupportedArgument
                return self._device_op.is_supported(problem)
            except:
                pass
        
        # Fallback: basic checks
        # Library kernels typically support any size
        return problem.M > 0 and problem.N > 0 and problem.K > 0
    
    def get_name(self) -> str:
        """Get kernel name"""
        return self._name
    
    def run(self, a, b, c, problem, stream=None) -> float:
        """
        Execute kernel
        
        Args:
            a: Input tensor A
            b: Input tensor B
            c: Output tensor C
            problem: Problem specification
            stream: Optional GPU stream
        
        Returns:
            Execution time in milliseconds
        """
        # If C++ device operation is available, use it
        if self._device_op is not None:
            return self._run_cpp_kernel(a, b, c, problem, stream)
        
        # Otherwise, use reference implementation
        return self._run_reference(a, b, c, problem)
    
    def _run_cpp_kernel(self, a, b, c, problem, stream) -> float:
        """Run using C++ library kernel (via bindings)"""
        try:
            # Get data pointers
            a_ptr = self._get_data_ptr(a)
            b_ptr = self._get_data_ptr(b)
            c_ptr = self._get_data_ptr(c)
            
            # Create argument object
            # This would call the library's MakeArgument
            # Simplified for now
            
            # Get invoker and run
            time_ms = self._device_op.run(a_ptr, b_ptr, c_ptr, problem, stream)
            return time_ms
        except Exception as e:
            # Fallback to reference
            print(f"Warning: C++ library kernel failed ({e}), using reference")
            return self._run_reference(a, b, c, problem)
    
    def _run_reference(self, a, b, c, problem) -> float:
        """Run using NumPy reference implementation"""
        start = time.perf_counter()
        
        # Convert to numpy
        a_np = self._to_numpy(a)
        b_np = self._to_numpy(b)
        
        # Compute
        result = np.matmul(a_np, b_np)
        
        # Copy to output
        if isinstance(c, np.ndarray):
            np.copyto(c, result)
        else:
            # Try to copy back to device tensor
            try:
                import torch
                if isinstance(c, torch.Tensor):
                    c.copy_(torch.from_numpy(result))
            except:
                pass
        
        elapsed = (time.perf_counter() - start) * 1000
        return elapsed
    
    def get_backend_type(self) -> BackendType:
        """Get backend type"""
        return BackendType.LIBRARY
    
    def get_metadata(self) -> dict:
        """Get kernel metadata"""
        meta = super().get_metadata()
        meta.update({
            'source': 'ck_library',
        })
        return meta


class LibraryBackend(BackendBase):
    """
    Backend for CK Library pre-compiled kernels
    
    Discovers and creates kernel instances from DeviceOperationInstanceFactory.
    """
    
    def __init__(self):
        """Initialize library backend"""
        self._cpp_backend = None
        self._load_cpp_backend()
    
    def _load_cpp_backend(self):
        """Try to load C++ backend"""
        try:
            from .. import _ck_dispatcher_cpp
            if hasattr(_ck_dispatcher_cpp, 'LibraryBackend'):
                self._cpp_backend = _ck_dispatcher_cpp.LibraryBackend()
        except ImportError:
            pass
    
    def discover_kernels(self, search_path: str = None) -> List[KernelInstance]:
        """
        Discover CK Library kernels
        
        Args:
            search_path: Optional path (not used for library kernels)
        
        Returns:
            List of LibraryKernelInstance objects
        """
        if self._cpp_backend is not None:
            try:
                # Use C++ backend to enumerate library kernels
                return self._cpp_backend.discover_kernels()
            except Exception as e:
                print(f"Warning: C++ library discovery failed: {e}")
        
        # Fallback: return empty list
        # Library kernels require C++ integration
        return []
    
    def create_kernel_instance(self, kernel_config: dict) -> LibraryKernelInstance:
        """
        Create kernel instance from configuration
        
        Args:
            kernel_config: Kernel configuration dictionary
        
        Returns:
            LibraryKernelInstance
        """
        # Extract configuration
        kernel_name = kernel_config.get('name', 'unknown')
        
        # Create kernel key from config
        # This would parse the library kernel's template parameters
        # Simplified for now
        from ..core import KernelKey, Signature, Algorithm, TileShape, WaveShape, WarpTileShape
        from ..core import DataType, LayoutTag, Pipeline, Epilogue, Scheduler
        
        # Default kernel key
        kernel_key = KernelKey(
            signature=Signature(
                dtype_a=DataType.FP16,
                dtype_b=DataType.FP16,
                dtype_c=DataType.FP16,
                dtype_acc=DataType.FP32,
                layout_a=LayoutTag.ROW_MAJOR,
                layout_b=LayoutTag.COL_MAJOR,
                layout_c=LayoutTag.ROW_MAJOR,
                transpose_a=False,
                transpose_b=False,
                grouped=False,
                split_k=1,
                elementwise_op="PassThrough",
                num_d_tensors=0,
                structured_sparsity=False,
            ),
            algorithm=Algorithm(
                tile_shape=TileShape(m=256, n=256, k=32),
                wave_shape=WaveShape(m=2, n=2, k=1),
                warp_tile_shape=WarpTileShape(m=32, n=32, k=16),
                pipeline=Pipeline.COMP_V4,
                scheduler=Scheduler.INTRAWAVE,
                epilogue=Epilogue.CSHUFFLE,
                block_size=256,
                double_buffer=True,
                persistent=False,
                preshuffle=False,
                transpose_c=False,
                num_wave_groups=1,
            ),
            gfx_arch=942,
        )
        
        # Get C++ device operation if available
        device_op = kernel_config.get('device_op')
        
        return LibraryKernelInstance(kernel_key, kernel_name, device_op)
    
    def get_backend_type(self) -> BackendType:
        """Get backend type"""
        return BackendType.LIBRARY
    
    def enumerate_operations(self) -> List[str]:
        """
        Enumerate available operation types
        
        Returns:
            List of operation type names (e.g., "gemm", "conv2d_fwd", etc.)
        """
        if self._cpp_backend is not None:
            try:
                return self._cpp_backend.enumerate_operations()
            except:
                pass
        
        # Default operations
        return [
            "gemm",
            "gemm_add",
            "gemm_softmax_gemm",
            "conv2d_fwd",
            "conv2d_bwd_data",
            "conv2d_bwd_weight",
        ]
    
    def get_factory_instances(self, operation: str) -> List[dict]:
        """
        Get factory instances for an operation
        
        Args:
            operation: Operation type (e.g., "gemm")
        
        Returns:
            List of kernel configuration dictionaries
        """
        if self._cpp_backend is not None:
            try:
                return self._cpp_backend.get_factory_instances(operation)
            except:
                pass
        
        return []

