"""
Advanced features examples for CK Tile Dispatcher

Demonstrates configuration, logging, caching, and performance optimization.
"""

import numpy as np
import ck_tile_dispatcher as ckd


def example_1_configuration():
    """Example 1: Configuration Management"""
    print("=" * 80)
    print("Example 1: Configuration Management")
    print("=" * 80)
    
    # Print default configuration
    print("\nDefault configuration:")
    ckd.print_config()
    
    # Configure globally
    ckd.configure(
        gpu_arch="gfx90a",
        default_kernel_set="fp16_rcr_compute",
        enable_profiling=True
    )
    
    print("\nAfter configuration:")
    config = ckd.get_config()
    print(f"  GPU arch: {config.gpu_arch}")
    print(f"  Kernel set: {config.default_kernel_set}")
    print(f"  Profiling: {config.enable_profiling}")
    
    # Reset to defaults
    ckd.reset_config()
    print("\n✓ Configuration reset")
    print()


def example_2_presets():
    """Example 2: Using Configuration Presets"""
    print("=" * 80)
    print("Example 2: Configuration Presets")
    print("=" * 80)
    
    presets = ["performance", "memory", "debug", "production"]
    
    for preset in presets:
        ckd.use_preset(preset)
        config = ckd.get_config()
        print(f"\n{preset.upper()} preset:")
        print(f"  Kernel set: {config.default_kernel_set}")
        print(f"  Strategy: {config.selection_strategy}")
        print(f"  Cache: {config.enable_kernel_cache}")
        print(f"  Validation: {config.enable_validation}")
    
    print()


def example_3_config_context():
    """Example 3: Temporary Configuration Context"""
    print("=" * 80)
    print("Example 3: Configuration Context")
    print("=" * 80)
    
    # Set default
    ckd.use_preset("production")
    print(f"Default: {ckd.get_config().default_kernel_set}")
    
    # Temporary override
    with ckd.config_context(
        default_kernel_set="fp16_rcr_memory",
        enable_profiling=True
    ):
        print(f"Inside context: {ckd.get_config().default_kernel_set}")
        print(f"Profiling: {ckd.get_config().enable_profiling}")
    
    # Back to default
    print(f"After context: {ckd.get_config().default_kernel_set}")
    print()


def example_4_logging():
    """Example 4: Logging Configuration"""
    print("=" * 80)
    print("Example 4: Logging")
    print("=" * 80)
    
    # Set log level
    ckd.set_log_level("INFO")
    print("✓ Log level set to INFO")
    
    # Log system info
    ckd.log_system_info()
    
    # Enable file logging
    # ckd.enable_file_logging("dispatcher.log")
    # print("✓ File logging enabled")
    
    # Disable logging
    ckd.disable_logging()
    print("✓ Logging disabled")
    print()


def example_5_performance_logging():
    """Example 5: Performance Logging"""
    print("=" * 80)
    print("Example 5: Performance Logging")
    print("=" * 80)
    
    # Get performance logger
    perf_logger = ckd.get_perf_logger()
    
    # Create dispatcher
    dispatcher = ckd.Dispatcher()
    dispatcher.register_kernels("fp16_rcr_essential")
    
    # Run some operations
    for size in [256, 512, 1024]:
        A = np.random.randn(size, size).astype(np.float16)
        B = np.random.randn(size, size).astype(np.float16)
        
        import time
        start = time.perf_counter()
        C = dispatcher.gemm(A, B)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Log performance
        perf_logger.log_execution(
            f"gemm_{size}x{size}",
            elapsed_ms,
            size=size
        )
    
    # Print summary
    perf_logger.print_summary()
    
    # Reset
    perf_logger.reset()
    print()


def example_6_dispatch_logging():
    """Example 6: Dispatch Logging"""
    print("=" * 80)
    print("Example 6: Dispatch Logging")
    print("=" * 80)
    
    # Get dispatch logger
    dispatch_logger = ckd.get_dispatch_logger()
    
    # Simulate dispatches
    for i in range(10):
        size = np.random.choice([256, 512, 1024, 2048])
        kernel = f"kernel_{np.random.choice(['A', 'B', 'C'])}"
        
        dispatch_logger.log_dispatch(
            problem_size=(size, size, size),
            kernel_name=kernel,
            selection_time_ms=np.random.uniform(0.1, 1.0)
        )
    
    # Print summary
    dispatch_logger.print_summary()
    
    # Reset
    dispatch_logger.reset()
    print()


def example_7_kernel_cache():
    """Example 7: Kernel Caching"""
    print("=" * 80)
    print("Example 7: Kernel Caching")
    print("=" * 80)
    
    # Get kernel cache
    kernel_cache = ckd.get_kernel_cache()
    
    # Cache some kernels
    kernel_cache.put_kernel((1024, 1024, 1024), "fp16", "rcr", "kernel_A")
    kernel_cache.put_kernel((2048, 2048, 2048), "fp16", "rcr", "kernel_B")
    kernel_cache.put_kernel((4096, 4096, 4096), "fp16", "rcr", "kernel_C")
    
    # Retrieve from cache
    kernel = kernel_cache.get_kernel((1024, 1024, 1024), "fp16", "rcr")
    print(f"✓ Retrieved kernel: {kernel}")
    
    # Print stats
    kernel_cache.print_stats()
    
    # Clear cache
    kernel_cache.clear()
    print("✓ Cache cleared")
    print()


def example_8_performance_cache():
    """Example 8: Performance Caching"""
    print("=" * 80)
    print("Example 8: Performance Caching")
    print("=" * 80)
    
    # Get performance cache
    perf_cache = ckd.get_perf_cache()
    
    # Cache performance data
    kernels = ["kernel_A", "kernel_B", "kernel_C"]
    problem_size = (1024, 1024, 1024)
    
    for kernel in kernels:
        gflops = np.random.uniform(100, 200)
        perf_cache.put_performance(kernel, problem_size, gflops)
        print(f"Cached {kernel}: {gflops:.2f} GFLOPS")
    
    # Get best kernel
    best = perf_cache.get_best_kernel(kernels, problem_size)
    print(f"\n✓ Best kernel: {best}")
    
    # Print stats
    stats = perf_cache.get_stats()
    print(f"\nCache stats:")
    print(f"  Size: {stats['size']}")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    print()


def example_9_cache_stats():
    """Example 9: Cache Statistics"""
    print("=" * 80)
    print("Example 9: Cache Statistics")
    print("=" * 80)
    
    # Print all cache stats
    ckd.print_cache_stats()
    
    # Clear all caches
    ckd.clear_all_caches()
    print("\n✓ All caches cleared")
    print()


def example_10_integrated_workflow():
    """Example 10: Integrated Workflow"""
    print("=" * 80)
    print("Example 10: Integrated Workflow")
    print("=" * 80)
    
    # Use performance preset
    ckd.use_preset("performance")
    
    # Enable logging
    ckd.set_log_level("INFO")
    
    # Create dispatcher
    dispatcher = ckd.Dispatcher()
    dispatcher.register_kernels("fp16_rcr_compute")
    
    # Run with profiling
    profiler = ckd.Profiler()
    
    with profiler:
        # Multiple GEMMs
        for size in [512, 1024, 2048]:
            A = np.random.randn(size, size).astype(np.float16)
            B = np.random.randn(size, size).astype(np.float16)
            C = dispatcher.gemm(A, B)
            print(f"  ✓ GEMM {size}x{size} complete")
    
    # Print profiling results
    print("\nProfiling results:")
    profiler.print_summary()
    
    # Print cache stats
    print("\nCache statistics:")
    ckd.print_cache_stats()
    
    # Print performance log
    print("\nPerformance log:")
    ckd.get_perf_logger().print_summary()
    
    print("\n✓ Integrated workflow complete")
    print()


def example_11_environment_variables():
    """Example 11: Environment Variables"""
    print("=" * 80)
    print("Example 11: Environment Variables")
    print("=" * 80)
    
    print("You can configure the dispatcher using environment variables:")
    print()
    print("  export CK_GPU_ARCH=gfx90a")
    print("  export CK_DEFAULT_KERNEL_SET=fp16_rcr_compute")
    print("  export CK_ENABLE_CACHE=true")
    print("  export CK_ENABLE_PROFILING=true")
    print("  export CK_LOG_LEVEL=INFO")
    print()
    print("These will be automatically loaded on import.")
    print()


def example_12_save_load_config():
    """Example 12: Save/Load Configuration"""
    print("=" * 80)
    print("Example 12: Save/Load Configuration")
    print("=" * 80)
    
    # Configure
    ckd.configure(
        gpu_arch="gfx90a",
        default_kernel_set="fp16_rcr_compute",
        enable_profiling=True
    )
    
    # Save configuration
    config = ckd.get_config()
    config.save("my_config.json")
    print("✓ Configuration saved to my_config.json")
    
    # Load configuration
    loaded_config = ckd.DispatcherConfig.load("my_config.json")
    ckd.set_config(loaded_config)
    print("✓ Configuration loaded from my_config.json")
    
    # Verify
    print(f"\nLoaded config:")
    print(f"  GPU arch: {loaded_config.gpu_arch}")
    print(f"  Kernel set: {loaded_config.default_kernel_set}")
    print(f"  Profiling: {loaded_config.enable_profiling}")
    
    # Cleanup
    import os
    if os.path.exists("my_config.json"):
        os.remove("my_config.json")
        print("\n✓ Cleanup complete")
    print()


def main():
    """Run all examples"""
    examples = [
        example_1_configuration,
        example_2_presets,
        example_3_config_context,
        example_4_logging,
        example_5_performance_logging,
        example_6_dispatch_logging,
        example_7_kernel_cache,
        example_8_performance_cache,
        example_9_cache_stats,
        example_10_integrated_workflow,
        example_11_environment_variables,
        example_12_save_load_config,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"✗ Example failed: {e}")
            import traceback
            traceback.print_exc()
            print()


if __name__ == "__main__":
    main()

