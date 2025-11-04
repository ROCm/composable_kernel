"""
Kernel Registry for CK Tile Dispatcher

Provides central registration and lookup of kernel instances with conflict resolution.
"""

from typing import Dict, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass
import threading


class Priority(Enum):
    """Registration priority for conflict resolution"""
    LOW = 0
    NORMAL = 1
    HIGH = 2


@dataclass
class RegistryEntry:
    """Entry in the kernel registry"""
    kernel_instance: 'KernelInstance'
    priority: Priority
    backend_type: str  # "tile", "library", "jit"
    registration_order: int


class Registry:
    """
    Central kernel registry with conflict resolution
    
    Features:
    - Thread-safe registration and lookup
    - Priority-based conflict resolution
    - Backend type tracking
    - Kernel enumeration and filtering
    
    Example:
        >>> registry = Registry()
        >>> registry.register(kernel, priority=Priority.HIGH)
        >>> kernel = registry.lookup(kernel_key)
    """
    
    def __init__(self):
        """Initialize registry"""
        self._registry: Dict[str, RegistryEntry] = {}
        self._lock = threading.RLock()
        self._registration_counter = 0
    
    def register(self, kernel_instance, priority: Priority = Priority.NORMAL,
                 backend_type: str = "unknown"):
        """
        Register a kernel instance
        
        Args:
            kernel_instance: Kernel instance to register
            priority: Registration priority for conflict resolution
            backend_type: Backend type ("tile", "library", "jit")
        
        Conflict Resolution:
            - Higher priority wins
            - Same priority: CK Tile > Library > JIT
            - Same priority and backend: earlier registration wins
        """
        with self._lock:
            key_id = kernel_instance.get_key().to_identifier()
            
            # Check for conflicts
            if key_id in self._registry:
                existing = self._registry[key_id]
                
                # Priority comparison
                if priority.value < existing.priority.value:
                    # Lower priority, skip
                    return
                elif priority.value > existing.priority.value:
                    # Higher priority, replace
                    pass
                else:
                    # Same priority, use backend preference
                    backend_order = {"tile": 2, "library": 1, "jit": 0}
                    new_order = backend_order.get(backend_type, -1)
                    existing_order = backend_order.get(existing.backend_type, -1)
                    
                    if new_order <= existing_order:
                        # Keep existing
                        return
            
            # Register kernel
            entry = RegistryEntry(
                kernel_instance=kernel_instance,
                priority=priority,
                backend_type=backend_type,
                registration_order=self._registration_counter
            )
            self._registry[key_id] = entry
            self._registration_counter += 1
    
    def lookup(self, key_id: str) -> Optional['KernelInstance']:
        """
        Lookup kernel by key identifier
        
        Args:
            key_id: Kernel key identifier
        
        Returns:
            Kernel instance or None if not found
        """
        with self._lock:
            entry = self._registry.get(key_id)
            return entry.kernel_instance if entry else None
    
    def lookup_by_key(self, kernel_key) -> Optional['KernelInstance']:
        """
        Lookup kernel by KernelKey object
        
        Args:
            kernel_key: KernelKey object
        
        Returns:
            Kernel instance or None if not found
        """
        key_id = kernel_key.to_identifier()
        return self.lookup(key_id)
    
    def enumerate_all(self) -> List['KernelInstance']:
        """
        Enumerate all registered kernels
        
        Returns:
            List of all kernel instances
        """
        with self._lock:
            return [entry.kernel_instance for entry in self._registry.values()]
    
    def filter(self, predicate: Callable[['KernelInstance'], bool]) -> List['KernelInstance']:
        """
        Filter kernels by predicate
        
        Args:
            predicate: Function that takes a kernel instance and returns bool
        
        Returns:
            List of kernel instances matching predicate
        
        Example:
            >>> # Find all FP16 kernels
            >>> fp16_kernels = registry.filter(
            ...     lambda k: k.get_key().signature.dtype_a == DataType.FP16
            ... )
        """
        with self._lock:
            return [
                entry.kernel_instance
                for entry in self._registry.values()
                if predicate(entry.kernel_instance)
            ]
    
    def filter_by_problem(self, problem) -> List['KernelInstance']:
        """
        Filter kernels that support a given problem
        
        Args:
            problem: Problem specification
        
        Returns:
            List of kernel instances that support the problem
        """
        return self.filter(lambda k: k.supports(problem))
    
    def size(self) -> int:
        """Get number of registered kernels"""
        with self._lock:
            return len(self._registry)
    
    def clear(self):
        """Clear all registered kernels"""
        with self._lock:
            self._registry.clear()
            self._registration_counter = 0
    
    def get_stats(self) -> Dict:
        """
        Get registry statistics
        
        Returns:
            Dictionary with statistics
        """
        with self._lock:
            backend_counts = {}
            priority_counts = {p: 0 for p in Priority}
            
            for entry in self._registry.values():
                # Count by backend
                backend_counts[entry.backend_type] = \
                    backend_counts.get(entry.backend_type, 0) + 1
                
                # Count by priority
                priority_counts[entry.priority] += 1
            
            return {
                'total_kernels': len(self._registry),
                'by_backend': backend_counts,
                'by_priority': {p.name: count for p, count in priority_counts.items()},
            }
    
    def print_stats(self):
        """Print registry statistics"""
        stats = self.get_stats()
        
        print("=" * 60)
        print("Registry Statistics")
        print("=" * 60)
        print(f"Total kernels: {stats['total_kernels']}")
        
        print("\nBy backend:")
        for backend, count in stats['by_backend'].items():
            print(f"  {backend:20s}: {count}")
        
        print("\nBy priority:")
        for priority, count in stats['by_priority'].items():
            print(f"  {priority:20s}: {count}")
        
        print("=" * 60)
    
    def __len__(self):
        """Get number of registered kernels"""
        return self.size()
    
    def __contains__(self, key_id: str):
        """Check if kernel is registered"""
        with self._lock:
            return key_id in self._registry
    
    def __repr__(self):
        return f"Registry(size={self.size()})"


# Singleton registry instance
_global_registry: Optional[Registry] = None


def get_global_registry() -> Registry:
    """Get global registry instance"""
    global _global_registry
    if _global_registry is None:
        _global_registry = Registry()
    return _global_registry


def reset_global_registry():
    """Reset global registry"""
    global _global_registry
    _global_registry = Registry()

