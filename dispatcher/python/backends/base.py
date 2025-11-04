"""
Base classes for backend implementations
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Any
import numpy as np


class BackendType(Enum):
    """Backend type enumeration"""
    TILE = "tile"
    LIBRARY = "library"
    JIT = "jit"
    UNKNOWN = "unknown"


class KernelInstance(ABC):
    """
    Abstract base class for kernel instances
    
    All backend implementations must inherit from this class.
    """
    
    @abstractmethod
    def get_key(self):
        """
        Get kernel key
        
        Returns:
            KernelKey object
        """
        pass
    
    @abstractmethod
    def supports(self, problem) -> bool:
        """
        Check if kernel supports the given problem
        
        Args:
            problem: Problem specification
        
        Returns:
            True if kernel supports the problem
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Get kernel name
        
        Returns:
            Human-readable kernel name
        """
        pass
    
    @abstractmethod
    def run(self, a, b, c, problem, stream=None) -> float:
        """
        Execute kernel
        
        Args:
            a: Input tensor A (numpy array or device pointer)
            b: Input tensor B (numpy array or device pointer)
            c: Output tensor C (numpy array or device pointer)
            problem: Problem specification
            stream: Optional GPU stream
        
        Returns:
            Execution time in milliseconds
        """
        pass
    
    def validate(self, a, b, c, problem, rtol=1e-3, atol=1e-5) -> bool:
        """
        Validate kernel output
        
        Args:
            a: Input tensor A
            b: Input tensor B
            c: Output tensor C
            problem: Problem specification
            rtol: Relative tolerance
            atol: Absolute tolerance
        
        Returns:
            True if validation passes
        """
        # Default implementation: compute reference and compare
        try:
            # Convert to numpy if needed
            a_np = self._to_numpy(a)
            b_np = self._to_numpy(b)
            c_np = self._to_numpy(c)
            
            # Compute reference
            c_ref = np.matmul(a_np, b_np)
            
            # Compare
            return np.allclose(c_np, c_ref, rtol=rtol, atol=atol)
        except Exception:
            return False
    
    def get_backend_type(self) -> BackendType:
        """Get backend type"""
        return BackendType.UNKNOWN
    
    def get_metadata(self) -> dict:
        """
        Get kernel metadata
        
        Returns:
            Dictionary with kernel metadata
        """
        return {
            'name': self.get_name(),
            'backend': self.get_backend_type().value,
            'key': self.get_key().to_identifier() if hasattr(self.get_key(), 'to_identifier') else str(self.get_key()),
        }
    
    @staticmethod
    def _to_numpy(tensor) -> np.ndarray:
        """Convert tensor to numpy array"""
        if isinstance(tensor, np.ndarray):
            return tensor
        
        # Try PyTorch
        try:
            import torch
            if isinstance(tensor, torch.Tensor):
                return tensor.cpu().numpy()
        except ImportError:
            pass
        
        # Try CuPy
        try:
            import cupy as cp
            if isinstance(tensor, cp.ndarray):
                return cp.asnumpy(tensor)
        except ImportError:
            pass
        
        # Assume it's already array-like
        return np.asarray(tensor)
    
    @staticmethod
    def _get_data_ptr(tensor) -> int:
        """Get device pointer from tensor"""
        # Try PyTorch
        try:
            import torch
            if isinstance(tensor, torch.Tensor):
                return tensor.data_ptr()
        except ImportError:
            pass
        
        # Try CuPy
        try:
            import cupy as cp
            if isinstance(tensor, cp.ndarray):
                return tensor.data.ptr
        except ImportError:
            pass
        
        # Try numpy (for CPU)
        if isinstance(tensor, np.ndarray):
            return tensor.ctypes.data
        
        raise TypeError(f"Cannot get data pointer from {type(tensor)}")
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.get_name()})"


class BackendBase(ABC):
    """
    Abstract base class for backend implementations
    
    Backends are responsible for:
    - Discovering available kernels
    - Creating kernel instances
    - Managing backend-specific resources
    """
    
    @abstractmethod
    def discover_kernels(self, search_path: str) -> list:
        """
        Discover available kernels
        
        Args:
            search_path: Path to search for kernels
        
        Returns:
            List of kernel instances
        """
        pass
    
    @abstractmethod
    def create_kernel_instance(self, kernel_config: dict) -> KernelInstance:
        """
        Create kernel instance from configuration
        
        Args:
            kernel_config: Kernel configuration dictionary
        
        Returns:
            KernelInstance
        """
        pass
    
    @abstractmethod
    def get_backend_type(self) -> BackendType:
        """Get backend type"""
        pass
    
    def initialize(self):
        """Initialize backend (optional)"""
        pass
    
    def cleanup(self):
        """Cleanup backend resources (optional)"""
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(type={self.get_backend_type().value})"

