"""
CK Tile Dispatcher - Python Interface

High-level Python bindings for the CK Tile GEMM dispatcher.

Example:
    >>> import ck_tile_dispatcher as ckd
    >>>
    >>> # Simple API - everything automated
    >>> from ck_tile_dispatcher import SimpleGemmAPI
    >>> gemm = SimpleGemmAPI()
    >>> gemm.ensure_kernels_ready()
    >>> result = gemm.execute(M=1024, N=1024, K=1024)
    >>>
    >>> # Or use one-liner
    >>> from ck_tile_dispatcher import quick_gemm
    >>> result = quick_gemm(M=2048, N=2048, K=2048)
"""

__version__ = "1.0.0"
__author__ = "AMD CK Tile Team"

# Public API - all these are intentionally re-exported
__all__ = [
    # High-level API
    "Dispatcher",
    "SimpleGemmAPI",
    "generate_kernels",
    "quick_gemm",
    "list_available_presets",
    # Core types
    "LegacyDispatcher",
    "Problem",
    "KernelKey",
    "DataType",
    "LayoutTag",
    "DispatchResult",
    # Utilities
    "get_available_kernels",
    "benchmark_kernel",
    "profile_dispatch",
    # JSON export
    "export_registry_json",
    "print_registry_summary",
    "get_registry_statistics",
    "list_kernel_identifiers",
    "filter_kernels_by_property",
    "enable_auto_export",
    "disable_auto_export",
    "is_auto_export_enabled",
    # PyTorch integration (optional)
    "CKTileGEMM",
    "ck_gemm",
    "register_ck_ops",
    "HAS_TORCH",
]

# Import high-level API (primary interface)
from .dispatcher_api import (
    Dispatcher,
    SimpleGemmAPI,
    generate_kernels,
    quick_gemm,
    list_available_presets,
)

# Import legacy core functionality
from .core import (
    Dispatcher as LegacyDispatcher,  # Keep for backward compatibility
    Problem,
    KernelKey,
    DataType,
    LayoutTag,
    DispatchResult,
)

# Import utilities
from .utils import (
    get_available_kernels,
    benchmark_kernel,
    profile_dispatch,
)

# Import PyTorch integration (if available)
try:
    from .torch_integration import (
        CKTileGEMM,
        ck_gemm,
        register_ck_ops,
    )

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Import profiler
from .profiler import Profiler, ProfileReport

# Import configuration
from .config import (
    get_config,
    set_config,
    reset_config,
    configure,
    config_context,
    use_preset,
    print_config,
    DispatcherConfig,
)

# Import logging
from .logging_utils import (
    set_log_level,
    enable_file_logging,
    disable_logging,
    get_perf_logger,
    get_dispatch_logger,
    log_system_info,
)

# Import cache
from .cache import (
    get_kernel_cache,
    get_perf_cache,
    clear_all_caches,
    print_cache_stats,
)

# Import registry
from .registry import (
    Registry,
    Priority,
    get_global_registry,
    reset_global_registry,
)

# Import JSON export
from .json_export import (
    export_registry_json,
    print_registry_summary,
    get_registry_statistics,
    list_kernel_identifiers,
    filter_kernels_by_property,
    enable_auto_export,
    disable_auto_export,
    is_auto_export_enabled,
)

# Import selection
from .selection import (
    SelectionEngine,
    SelectionStrategy,
    SelectionResult,
    size_based_heuristic,
    datatype_aware_heuristic,
    ml_based_heuristic,
)

# Import backends
from .backends import (
    KernelInstance,
    BackendType,
    TileKernelInstance,
    TileBackend,
    LibraryKernelInstance,
    LibraryBackend,
)

__all__ = [
    # High-Level API (New)
    "Dispatcher",  # Main dispatcher class
    "SimpleGemmAPI",
    "generate_kernels",
    "quick_gemm",
    "list_available_presets",
    # Core
    "Problem",
    "KernelKey",
    "DataType",
    "LayoutTag",
    "DispatchResult",
    # Utils
    "get_available_kernels",
    "benchmark_kernel",
    "profile_dispatch",
    # Profiler
    "Profiler",
    "ProfileReport",
    # Configuration
    "get_config",
    "set_config",
    "reset_config",
    "configure",
    "config_context",
    "use_preset",
    "print_config",
    "DispatcherConfig",
    # Logging
    "set_log_level",
    "enable_file_logging",
    "disable_logging",
    "get_perf_logger",
    "get_dispatch_logger",
    "log_system_info",
    # Cache
    "get_kernel_cache",
    "get_perf_cache",
    "clear_all_caches",
    "print_cache_stats",
    # Registry
    "Registry",
    "Priority",
    "get_global_registry",
    "reset_global_registry",
    # Selection
    "SelectionEngine",
    "SelectionStrategy",
    "SelectionResult",
    "size_based_heuristic",
    "datatype_aware_heuristic",
    "ml_based_heuristic",
    # Backends
    "KernelInstance",
    "BackendType",
    "TileKernelInstance",
    "TileBackend",
    "LibraryKernelInstance",
    "LibraryBackend",
    # PyTorch (if available)
    "CKTileGEMM" if HAS_TORCH else None,
    "ck_gemm" if HAS_TORCH else None,
    "register_ck_ops" if HAS_TORCH else None,
    # Metadata
    "__version__",
]

# Remove None values from __all__
__all__ = [x for x in __all__ if x is not None]


def info():
    """Print dispatcher information"""
    print(f"CK Tile Dispatcher v{__version__}")
    print(f"PyTorch support: {'Yes' if HAS_TORCH else 'No'}")

    # Try to get C++ extension info
    try:
        from . import _ck_dispatcher_cpp  # noqa: F401

        print("C++ extension: Loaded")
        print(f"Available kernels: {len(get_available_kernels())}")
    except ImportError:
        print("C++ extension: Not loaded")
