"""
Configuration management for CK Tile Dispatcher

Provides centralized configuration with environment variable support.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class DispatcherConfig:
    """Global dispatcher configuration"""

    # GPU Architecture
    gpu_arch: str = "gfx942"

    # Kernel Selection
    default_kernel_set: str = "fp16_rcr_essential"
    selection_strategy: str = "heuristic"  # "first_fit" or "heuristic"

    # Performance
    enable_kernel_cache: bool = True
    cache_size: int = 1000
    enable_profiling: bool = False

    # Validation
    enable_validation: bool = False
    validation_rtol: float = 1e-3
    validation_atol: float = 1e-5

    # Logging
    log_level: str = "WARNING"  # DEBUG, INFO, WARNING, ERROR
    log_dispatch: bool = False
    log_performance: bool = False

    # Paths
    cache_dir: Optional[str] = None
    kernel_dir: Optional[str] = None

    # Advanced
    num_warmup_iterations: int = 10
    num_benchmark_iterations: int = 100
    prefer_persistent_kernels: bool = False
    max_smem_budget: int = 65536

    def __post_init__(self):
        """Load from environment variables"""
        self._load_from_env()

        # Set default paths
        if self.cache_dir is None:
            self.cache_dir = str(Path.home() / ".cache" / "ck_tile_dispatcher")
        if self.kernel_dir is None:
            self.kernel_dir = str(Path(__file__).parent.parent / "kernels")

    def _load_from_env(self):
        """Load configuration from environment variables"""
        env_mapping = {
            "CK_GPU_ARCH": "gpu_arch",
            "CK_DEFAULT_KERNEL_SET": "default_kernel_set",
            "CK_SELECTION_STRATEGY": "selection_strategy",
            "CK_ENABLE_CACHE": ("enable_kernel_cache", lambda x: x.lower() == "true"),
            "CK_CACHE_SIZE": ("cache_size", int),
            "CK_ENABLE_PROFILING": ("enable_profiling", lambda x: x.lower() == "true"),
            "CK_ENABLE_VALIDATION": (
                "enable_validation",
                lambda x: x.lower() == "true",
            ),
            "CK_LOG_LEVEL": "log_level",
            "CK_LOG_DISPATCH": ("log_dispatch", lambda x: x.lower() == "true"),
            "CK_CACHE_DIR": "cache_dir",
            "CK_KERNEL_DIR": "kernel_dir",
        }

        for env_var, config_attr in env_mapping.items():
            if env_var in os.environ:
                value = os.environ[env_var]

                if isinstance(config_attr, tuple):
                    attr_name, converter = config_attr
                    setattr(self, attr_name, converter(value))
                else:
                    setattr(self, config_attr, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def save(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "DispatcherConfig":
        """Load configuration from JSON file"""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(**data)

    def __repr__(self):
        return f"DispatcherConfig(arch={self.gpu_arch}, kernel_set={self.default_kernel_set})"


# Global configuration instance
_global_config: Optional[DispatcherConfig] = None


def get_config() -> DispatcherConfig:
    """Get global configuration instance"""
    global _global_config
    if _global_config is None:
        _global_config = DispatcherConfig()
    return _global_config


def set_config(config: DispatcherConfig):
    """Set global configuration instance"""
    global _global_config
    _global_config = config


def reset_config():
    """Reset configuration to defaults"""
    global _global_config
    _global_config = DispatcherConfig()


def configure(**kwargs):
    """
    Configure dispatcher globally

    Example:
        >>> import ck_tile_dispatcher as ckd
        >>> ckd.configure(
        ...     gpu_arch="gfx90a",
        ...     default_kernel_set="fp16_rcr_compute",
        ...     enable_profiling=True
        ... )
    """
    config = get_config()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration option: {key}")


# Context manager for temporary configuration
class config_context:
    """
    Temporary configuration context

    Example:
        >>> with ckd.config_context(enable_profiling=True):
        ...     C = dispatcher.gemm(A, B)
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.old_config = None

    def __enter__(self):
        self.old_config = get_config().to_dict()
        configure(**self.kwargs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.old_config:
            set_config(DispatcherConfig(**self.old_config))
        return False


# Preset configurations
PRESETS = {
    "performance": DispatcherConfig(
        default_kernel_set="fp16_rcr_compute",
        selection_strategy="heuristic",
        enable_kernel_cache=True,
        cache_size=2000,
        prefer_persistent_kernels=True,
    ),
    "memory": DispatcherConfig(
        default_kernel_set="fp16_rcr_memory",
        selection_strategy="heuristic",
        enable_kernel_cache=True,
        prefer_persistent_kernels=False,
    ),
    "debug": DispatcherConfig(
        default_kernel_set="fp16_rcr_essential",
        enable_validation=True,
        enable_profiling=True,
        log_level="DEBUG",
        log_dispatch=True,
        log_performance=True,
    ),
    "production": DispatcherConfig(
        default_kernel_set="fp16_rcr_compute",
        selection_strategy="heuristic",
        enable_kernel_cache=True,
        cache_size=5000,
        enable_validation=False,
        log_level="WARNING",
    ),
}


def use_preset(preset_name: str):
    """
    Use a preset configuration

    Available presets:
        - "performance": Optimized for performance
        - "memory": Optimized for memory usage
        - "debug": Debugging and validation
        - "production": Production deployment

    Example:
        >>> import ck_tile_dispatcher as ckd
        >>> ckd.use_preset("performance")
    """
    if preset_name not in PRESETS:
        raise ValueError(
            f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}"
        )

    set_config(PRESETS[preset_name])
    print(f"✓ Using preset: {preset_name}")


def print_config():
    """Print current configuration"""
    config = get_config()
    print("=" * 60)
    print("CK Tile Dispatcher Configuration")
    print("=" * 60)
    for key, value in config.to_dict().items():
        print(f"  {key:30s}: {value}")
    print("=" * 60)
