"""
Kernel cache management for CK Tile Dispatcher

Provides intelligent caching of kernel instances and dispatch decisions.
"""

import time
import pickle
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = 0.0
    size_bytes: int = 0
    
    def touch(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_access = time.time()


class LRUCache:
    """
    LRU (Least Recently Used) cache
    
    Features:
    - Size-based eviction
    - Access statistics
    - Persistence support
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize LRU cache
        
        Args:
            max_size: Maximum number of entries
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            entry = self.cache[key]
            entry.touch()
            self.cache.move_to_end(key)  # Mark as recently used
            self.hits += 1
            return entry.value
        else:
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any):
        """Put value in cache"""
        if key in self.cache:
            # Update existing entry
            entry = self.cache[key]
            entry.value = value
            entry.touch()
            self.cache.move_to_end(key)
        else:
            # Add new entry
            if len(self.cache) >= self.max_size:
                # Evict least recently used
                self.cache.popitem(last=False)
            
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                last_access=time.time()
            )
            self.cache[key] = entry
    
    def remove(self, key: str):
        """Remove entry from cache"""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self):
        """Clear all entries"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def size(self) -> int:
        """Get number of entries"""
        return len(self.cache)
    
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hit_rate(),
            'total_accesses': self.hits + self.misses,
        }
    
    def print_stats(self):
        """Print cache statistics"""
        stats = self.get_stats()
        print("=" * 60)
        print("Cache Statistics")
        print("=" * 60)
        print(f"Size: {stats['size']}/{stats['max_size']}")
        print(f"Hits: {stats['hits']}")
        print(f"Misses: {stats['misses']}")
        print(f"Hit rate: {stats['hit_rate']:.2%}")
        print("=" * 60)


class KernelCache:
    """
    Cache for kernel instances and dispatch decisions
    
    Features:
    - Problem-based caching
    - Persistent storage
    - Statistics tracking
    """
    
    def __init__(self, cache_dir: Optional[str] = None, max_size: int = 1000):
        """
        Initialize kernel cache
        
        Args:
            cache_dir: Directory for persistent cache
            max_size: Maximum number of cached entries
        """
        self.cache = LRUCache(max_size=max_size)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _make_key(self, problem_size: Tuple[int, int, int], 
                  dtype: str, layout: str) -> str:
        """Create cache key from problem specification"""
        M, N, K = problem_size
        key_str = f"{M}x{N}x{K}_{dtype}_{layout}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_kernel(self, problem_size: Tuple[int, int, int], 
                   dtype: str, layout: str) -> Optional[str]:
        """Get cached kernel name"""
        key = self._make_key(problem_size, dtype, layout)
        return self.cache.get(key)
    
    def put_kernel(self, problem_size: Tuple[int, int, int], 
                   dtype: str, layout: str, kernel_name: str):
        """Cache kernel name"""
        key = self._make_key(problem_size, dtype, layout)
        self.cache.put(key, kernel_name)
    
    def save(self, filepath: Optional[str] = None):
        """Save cache to disk"""
        if filepath is None:
            if self.cache_dir is None:
                raise ValueError("No cache directory specified")
            filepath = self.cache_dir / "kernel_cache.pkl"
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.cache.cache, f)
    
    def load(self, filepath: Optional[str] = None):
        """Load cache from disk"""
        if filepath is None:
            if self.cache_dir is None:
                raise ValueError("No cache directory specified")
            filepath = self.cache_dir / "kernel_cache.pkl"
        
        if Path(filepath).exists():
            with open(filepath, 'rb') as f:
                self.cache.cache = pickle.load(f)
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()
    
    def print_stats(self):
        """Print cache statistics"""
        self.cache.print_stats()


class PerformanceCache:
    """
    Cache for performance measurements
    
    Stores historical performance data to improve kernel selection.
    """
    
    def __init__(self, max_entries: int = 10000):
        """
        Initialize performance cache
        
        Args:
            max_entries: Maximum number of performance entries
        """
        self.cache = LRUCache(max_size=max_entries)
    
    def _make_key(self, kernel_name: str, problem_size: Tuple[int, int, int]) -> str:
        """Create cache key"""
        M, N, K = problem_size
        key_str = f"{kernel_name}_{M}x{N}x{K}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_performance(self, kernel_name: str, 
                       problem_size: Tuple[int, int, int]) -> Optional[float]:
        """Get cached performance (GFLOPS)"""
        key = self._make_key(kernel_name, problem_size)
        return self.cache.get(key)
    
    def put_performance(self, kernel_name: str, 
                       problem_size: Tuple[int, int, int], 
                       gflops: float):
        """Cache performance measurement"""
        key = self._make_key(kernel_name, problem_size)
        self.cache.put(key, gflops)
    
    def get_best_kernel(self, kernels: list, 
                       problem_size: Tuple[int, int, int]) -> Optional[str]:
        """Get best kernel based on cached performance"""
        best_kernel = None
        best_gflops = 0.0
        
        for kernel in kernels:
            gflops = self.get_performance(kernel, problem_size)
            if gflops and gflops > best_gflops:
                best_gflops = gflops
                best_kernel = kernel
        
        return best_kernel
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()


# Global cache instances
_kernel_cache: Optional[KernelCache] = None
_perf_cache: Optional[PerformanceCache] = None


def get_kernel_cache() -> KernelCache:
    """Get global kernel cache"""
    global _kernel_cache
    if _kernel_cache is None:
        from .config import get_config
        config = get_config()
        _kernel_cache = KernelCache(
            cache_dir=config.cache_dir,
            max_size=config.cache_size
        )
    return _kernel_cache


def get_perf_cache() -> PerformanceCache:
    """Get global performance cache"""
    global _perf_cache
    if _perf_cache is None:
        _perf_cache = PerformanceCache()
    return _perf_cache


def clear_all_caches():
    """Clear all caches"""
    if _kernel_cache:
        _kernel_cache.clear()
    if _perf_cache:
        _perf_cache.clear()


def print_cache_stats():
    """Print statistics for all caches"""
    print("\n" + "=" * 70)
    print("Cache Statistics Summary")
    print("=" * 70)
    
    if _kernel_cache:
        print("\nKernel Cache:")
        _kernel_cache.print_stats()
    
    if _perf_cache:
        print("\nPerformance Cache:")
        stats = _perf_cache.get_stats()
        print(f"  Entries: {stats['size']}/{stats['max_entries']}")
        print(f"  Hit rate: {stats['hit_rate']:.2%}")
    
    print("=" * 70)

