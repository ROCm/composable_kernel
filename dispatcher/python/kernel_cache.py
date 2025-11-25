#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Kernel Cache - Persistent compiled kernel caching with automatic invalidation

Features:
- Caches compiled kernel binaries (.so/.hsaco) to avoid recompilation
- Automatically invalidates cache when CK Tile source code changes
- Uses content hashing for robust change detection
- Thread-safe access
- Configurable cache location

Cache Invalidation:
- Hashes CK Tile include directory contents
- Hashes kernel source files
- Stores compiler version and flags
- Any change triggers recompilation

Usage:
    from kernel_cache import KernelCache
    
    cache = KernelCache()
    
    # Check if kernel is cached
    if binary := cache.lookup(kernel_key):
        # Use cached binary
        load_binary(binary)
    else:
        # Compile and cache
        binary = compile_kernel(kernel_key)
        cache.store(kernel_key, binary)
"""

import hashlib
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Hash Utilities
# =============================================================================

def hash_file(path: Path) -> str:
    """Hash a file's contents using SHA256."""
    if not path.exists():
        return ""
    
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def hash_directory(
    directory: Path,
    extensions: List[str] = None,
    exclude_dirs: List[str] = None
) -> str:
    """
    Hash a directory recursively.
    
    Args:
        directory: Directory to hash
        extensions: File extensions to include (default: .hpp, .h, .cpp, .py)
        exclude_dirs: Directory names to exclude (default: __pycache__, .git, build)
    
    Returns:
        Combined SHA256 hash of all matching files
    """
    if extensions is None:
        extensions = ['.hpp', '.h', '.cpp', '.py', '.cuh', '.hip']
    if exclude_dirs is None:
        exclude_dirs = ['__pycache__', '.git', 'build', '.cache', 'node_modules']
    
    if not directory.exists():
        return ""
    
    hasher = hashlib.sha256()
    
    # Sort for deterministic ordering
    for root, dirs, files in sorted(os.walk(directory)):
        # Filter out excluded directories
        dirs[:] = [d for d in sorted(dirs) if d not in exclude_dirs]
        
        for filename in sorted(files):
            if not any(filename.endswith(ext) for ext in extensions):
                continue
            
            filepath = Path(root) / filename
            
            # Hash the relative path and content
            rel_path = filepath.relative_to(directory)
            hasher.update(str(rel_path).encode())
            hasher.update(hash_file(filepath).encode())
    
    return hasher.hexdigest()


def hash_string(s: str) -> str:
    """Hash a string using SHA256."""
    return hashlib.sha256(s.encode()).hexdigest()


# =============================================================================
# Cache Metadata
# =============================================================================

@dataclass
class CacheMetadata:
    """Metadata for a cached kernel entry."""
    kernel_identifier: str
    gpu_arch: str
    source_hash: str          # Hash of CK Tile sources
    kernel_hash: str          # Hash of kernel config
    compiler_version: str = ""
    compile_flags: str = ""
    python_version: str = ""
    created_timestamp: float = 0.0
    last_accessed: float = 0.0
    binary_size: int = 0
    compile_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheMetadata":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    invalidations: int = 0
    total_cached: int = 0
    total_size_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def __repr__(self):
        return (f"CacheStats(hits={self.hits}, misses={self.misses}, "
                f"hit_rate={self.hit_rate:.1%}, cached={self.total_cached})")


# =============================================================================
# Kernel Cache
# =============================================================================

class KernelCache:
    """
    Persistent kernel cache with automatic invalidation.
    
    Caches compiled kernel binaries and automatically invalidates
    when source code changes.
    
    Example:
        cache = KernelCache()
        
        # Check cache
        if binary := cache.lookup("gemm_fp16_256x256x64"):
            use_cached(binary)
        else:
            binary = compile(...)
            cache.store("gemm_fp16_256x256x64", binary)
        
        # View stats
        print(cache.stats)
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        ck_tile_root: Optional[Path] = None,
        enabled: bool = True,
        max_entries: int = 1000,
        max_size_mb: int = 2048
    ):
        """
        Initialize kernel cache.
        
        Args:
            cache_dir: Cache directory (default: ~/.cache/ck_tile_dispatcher)
            ck_tile_root: Path to CK Tile include directory for hash computation
            enabled: Whether caching is enabled
            max_entries: Maximum number of cached entries
            max_size_mb: Maximum cache size in MB
        """
        self.cache_dir = cache_dir or self._get_default_cache_dir()
        self.ck_tile_root = ck_tile_root
        self.enabled = enabled
        self.max_entries = max_entries
        self.max_size_mb = max_size_mb
        
        self._lock = threading.RLock()
        self._cache_index: Dict[str, CacheMetadata] = {}
        self._stats = CacheStats()
        self._source_hash = ""
        
        # Create cache directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "binaries").mkdir(exist_ok=True)
        (self.cache_dir / "metadata").mkdir(exist_ok=True)
        
        # Compute source hash
        if self.ck_tile_root and self.ck_tile_root.exists():
            self._source_hash = hash_directory(self.ck_tile_root)
        
        # Load existing cache
        self._load_cache_index()
    
    @staticmethod
    def _get_default_cache_dir() -> Path:
        """Get default cache directory."""
        # Check environment variable first
        if cache_dir := os.environ.get("CK_TILE_CACHE_DIR"):
            return Path(cache_dir)
        
        # Use XDG cache directory
        if xdg_cache := os.environ.get("XDG_CACHE_HOME"):
            return Path(xdg_cache) / "ck_tile_dispatcher"
        
        # Fall back to ~/.cache
        return Path.home() / ".cache" / "ck_tile_dispatcher"
    
    def lookup(
        self,
        kernel_id: str,
        gpu_arch: str = ""
    ) -> Optional[bytes]:
        """
        Look up a cached kernel binary.
        
        Args:
            kernel_id: Kernel identifier
            gpu_arch: GPU architecture (optional additional key)
        
        Returns:
            Binary data if found and valid, None otherwise
        """
        if not self.enabled:
            return None
        
        with self._lock:
            key = self._make_key(kernel_id, gpu_arch)
            meta = self._cache_index.get(key)
            
            if meta is None:
                self._stats.misses += 1
                return None
            
            # Check if source hash still matches
            if self._source_hash and meta.source_hash != self._source_hash:
                logger.info(f"Cache invalidated (source changed): {kernel_id}")
                self._stats.invalidations += 1
                self._stats.misses += 1
                self._invalidate_entry(key)
                return None
            
            # Load binary
            binary_path = self._get_binary_path(key)
            if not binary_path.exists():
                self._stats.misses += 1
                return None
            
            try:
                binary = binary_path.read_bytes()
                
                # Update access time
                meta.last_accessed = time.time()
                self._stats.hits += 1
                
                logger.debug(f"Cache hit: {kernel_id}")
                return binary
                
            except Exception as e:
                logger.warning(f"Failed to load cached binary: {e}")
                self._stats.misses += 1
                return None
    
    def store(
        self,
        kernel_id: str,
        binary: bytes,
        gpu_arch: str = "",
        compiler_version: str = "",
        compile_flags: str = "",
        compile_time_ms: float = 0.0
    ) -> bool:
        """
        Store a compiled kernel binary in cache.
        
        Args:
            kernel_id: Kernel identifier
            binary: Compiled binary data
            gpu_arch: GPU architecture
            compiler_version: Compiler version string
            compile_flags: Compilation flags used
            compile_time_ms: Time taken to compile (for stats)
        
        Returns:
            True if stored successfully
        """
        if not self.enabled or not binary:
            return False
        
        with self._lock:
            key = self._make_key(kernel_id, gpu_arch)
            
            # Write binary
            binary_path = self._get_binary_path(key)
            try:
                binary_path.write_bytes(binary)
            except Exception as e:
                logger.error(f"Failed to write cache binary: {e}")
                return False
            
            # Create metadata
            import sys
            meta = CacheMetadata(
                kernel_identifier=kernel_id,
                gpu_arch=gpu_arch,
                source_hash=self._source_hash,
                kernel_hash=hash_string(kernel_id),
                compiler_version=compiler_version,
                compile_flags=compile_flags,
                python_version=sys.version,
                created_timestamp=time.time(),
                last_accessed=time.time(),
                binary_size=len(binary),
                compile_time_ms=compile_time_ms
            )
            
            # Write metadata
            meta_path = self._get_metadata_path(key)
            try:
                meta_path.write_text(json.dumps(meta.to_dict(), indent=2))
            except Exception as e:
                logger.warning(f"Failed to write metadata: {e}")
            
            # Update index
            self._cache_index[key] = meta
            self._stats.total_cached += 1
            self._stats.total_size_bytes += len(binary)
            
            # Save index
            self._save_cache_index()
            
            # Evict old entries if needed
            self._maybe_evict()
            
            logger.debug(f"Cached kernel: {kernel_id} ({len(binary)} bytes)")
            return True
    
    def invalidate(self, kernel_id: str, gpu_arch: str = ""):
        """Invalidate a specific cache entry."""
        with self._lock:
            key = self._make_key(kernel_id, gpu_arch)
            self._invalidate_entry(key)
    
    def invalidate_all(self):
        """Invalidate all cached entries."""
        with self._lock:
            for key in list(self._cache_index.keys()):
                self._invalidate_entry(key)
            
            self._cache_index.clear()
            self._stats.total_cached = 0
            self._stats.total_size_bytes = 0
            self._save_cache_index()
            
            logger.info("Cache invalidated")
    
    def refresh_source_hash(self):
        """
        Refresh the source hash.
        Call this when CK Tile source code may have changed.
        """
        if self.ck_tile_root and self.ck_tile_root.exists():
            new_hash = hash_directory(self.ck_tile_root)
            if new_hash != self._source_hash:
                logger.info(f"Source hash changed: {self._source_hash[:8]}... -> {new_hash[:8]}...")
                self._source_hash = new_hash
    
    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats
    
    @property
    def source_hash(self) -> str:
        """Get current source hash."""
        return self._source_hash
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        with self._lock:
            return {
                "cache_dir": str(self.cache_dir),
                "ck_tile_root": str(self.ck_tile_root) if self.ck_tile_root else None,
                "source_hash": self._source_hash[:16] + "..." if self._source_hash else None,
                "enabled": self.enabled,
                "entries": len(self._cache_index),
                "total_size_mb": self._stats.total_size_bytes / (1024 * 1024),
                "stats": {
                    "hits": self._stats.hits,
                    "misses": self._stats.misses,
                    "hit_rate": f"{self._stats.hit_rate:.1%}",
                    "invalidations": self._stats.invalidations,
                }
            }
    
    def _make_key(self, kernel_id: str, gpu_arch: str) -> str:
        """Create cache key from kernel ID and architecture."""
        if gpu_arch:
            return f"{gpu_arch}_{kernel_id}"
        return kernel_id
    
    def _get_binary_path(self, key: str) -> Path:
        """Get path to binary file."""
        # Sanitize key for filename
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self.cache_dir / "binaries" / f"{safe_key}.so"
    
    def _get_metadata_path(self, key: str) -> Path:
        """Get path to metadata file."""
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self.cache_dir / "metadata" / f"{safe_key}.json"
    
    def _get_index_path(self) -> Path:
        """Get path to cache index file."""
        return self.cache_dir / "cache_index.json"
    
    def _invalidate_entry(self, key: str):
        """Invalidate a single cache entry."""
        try:
            self._get_binary_path(key).unlink(missing_ok=True)
            self._get_metadata_path(key).unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to remove cache entry: {e}")
        
        if key in self._cache_index:
            self._stats.total_size_bytes -= self._cache_index[key].binary_size
            del self._cache_index[key]
            self._stats.total_cached = len(self._cache_index)
    
    def _load_cache_index(self):
        """Load cache index from disk."""
        index_path = self._get_index_path()
        if not index_path.exists():
            return
        
        try:
            data = json.loads(index_path.read_text())
            for key, meta_dict in data.get("entries", {}).items():
                meta = CacheMetadata.from_dict(meta_dict)
                
                # Verify binary exists
                if self._get_binary_path(key).exists():
                    self._cache_index[key] = meta
                    self._stats.total_size_bytes += meta.binary_size
            
            self._stats.total_cached = len(self._cache_index)
            logger.debug(f"Loaded {len(self._cache_index)} cached entries")
            
        except Exception as e:
            logger.warning(f"Failed to load cache index: {e}")
    
    def _save_cache_index(self):
        """Save cache index to disk."""
        try:
            data = {
                "version": "1.0",
                "source_hash": self._source_hash,
                "entries": {key: meta.to_dict() for key, meta in self._cache_index.items()}
            }
            self._get_index_path().write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")
    
    def _maybe_evict(self):
        """Evict old entries if cache is too large."""
        if (len(self._cache_index) <= self.max_entries and 
            self._stats.total_size_bytes <= self.max_size_mb * 1024 * 1024):
            return
        
        # Sort by last accessed time (oldest first)
        entries = sorted(
            self._cache_index.items(),
            key=lambda x: x[1].last_accessed
        )
        
        # Evict oldest entries
        while ((len(self._cache_index) > self.max_entries or
                self._stats.total_size_bytes > self.max_size_mb * 1024 * 1024) and
               entries):
            key, meta = entries.pop(0)
            self._invalidate_entry(key)
            logger.debug(f"Evicted cache entry: {key}")
        
        self._save_cache_index()


# =============================================================================
# Global Instance
# =============================================================================

_global_cache: Optional[KernelCache] = None
_global_cache_lock = threading.Lock()


def get_global_cache(
    ck_tile_root: Optional[Path] = None,
    **kwargs
) -> KernelCache:
    """
    Get or create the global kernel cache instance.
    
    Args:
        ck_tile_root: Path to CK Tile include directory
        **kwargs: Additional arguments passed to KernelCache
    
    Returns:
        Global KernelCache instance
    """
    global _global_cache
    
    with _global_cache_lock:
        if _global_cache is None:
            _global_cache = KernelCache(ck_tile_root=ck_tile_root, **kwargs)
        return _global_cache


def clear_global_cache():
    """Clear and reset the global cache."""
    global _global_cache
    
    with _global_cache_lock:
        if _global_cache is not None:
            _global_cache.invalidate_all()
        _global_cache = None


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface for cache management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CK Tile Kernel Cache Manager")
    parser.add_argument("command", choices=["info", "clear", "stats", "list"],
                       help="Command to execute")
    parser.add_argument("--cache-dir", type=Path, help="Cache directory")
    
    args = parser.parse_args()
    
    cache = KernelCache(cache_dir=args.cache_dir)
    
    if args.command == "info":
        info = cache.get_cache_info()
        print(json.dumps(info, indent=2))
    
    elif args.command == "clear":
        cache.invalidate_all()
        print("Cache cleared")
    
    elif args.command == "stats":
        print(cache.stats)
    
    elif args.command == "list":
        for key, meta in cache._cache_index.items():
            print(f"{key}: {meta.binary_size} bytes, "
                  f"accessed {time.strftime('%Y-%m-%d %H:%M', time.localtime(meta.last_accessed))}")


if __name__ == "__main__":
    main()

