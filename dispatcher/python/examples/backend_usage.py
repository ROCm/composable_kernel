"""
Backend usage examples for CK Tile Dispatcher

Demonstrates how to use different backend implementations.
"""

import numpy as np
import ck_tile_dispatcher as ckd
from ck_tile_dispatcher.backends import (
    TileBackend,
    LibraryBackend,
    BackendType,
)


def example_1_tile_backend_discovery():
    """Example 1: Discover CK Tile Kernels"""
    print("=" * 80)
    print("Example 1: Tile Backend Discovery")
    print("=" * 80)
    
    # Create tile backend
    backend = TileBackend()
    
    # Discover kernels from codegen output
    # (Assumes tile_engine has generated kernels)
    codegen_dir = "build/tile_engine/generated"
    
    print(f"Discovering kernels in: {codegen_dir}")
    kernels = backend.discover_kernels(codegen_dir)
    
    print(f"✓ Found {len(kernels)} kernels")
    
    # Show first few kernels
    for i, kernel in enumerate(kernels[:5]):
        print(f"\n  Kernel {i+1}:")
        print(f"    Name: {kernel.get_name()}")
        print(f"    Backend: {kernel.get_backend_type().value}")
        meta = kernel.get_metadata()
        if 'tile_shape' in meta:
            print(f"    Tile: {meta['tile_shape']}")
    
    print()


def example_2_library_backend_discovery():
    """Example 2: Discover CK Library Kernels"""
    print("=" * 80)
    print("Example 2: Library Backend Discovery")
    print("=" * 80)
    
    # Create library backend
    backend = LibraryBackend()
    
    # Enumerate available operations
    operations = backend.enumerate_operations()
    print(f"Available operations: {operations}")
    
    # Discover kernels
    print("\nDiscovering library kernels...")
    kernels = backend.discover_kernels()
    
    print(f"✓ Found {len(kernels)} library kernels")
    
    # Show first few
    for i, kernel in enumerate(kernels[:5]):
        print(f"\n  Kernel {i+1}:")
        print(f"    Name: {kernel.get_name()}")
        print(f"    Backend: {kernel.get_backend_type().value}")
    
    print()


def example_3_register_tile_kernels():
    """Example 3: Register Tile Kernels with Dispatcher"""
    print("=" * 80)
    print("Example 3: Register Tile Kernels")
    print("=" * 80)
    
    # Create registry
    registry = ckd.Registry()
    
    # Create tile backend
    backend = TileBackend()
    
    # Discover and register kernels
    codegen_dir = "build/tile_engine/generated"
    kernels = backend.discover_kernels(codegen_dir)
    
    for kernel in kernels:
        registry.register(
            kernel,
            priority=ckd.Priority.HIGH,  # Tile kernels get high priority
            backend_type="tile"
        )
    
    print(f"✓ Registered {len(kernels)} tile kernels")
    registry.print_stats()
    print()


def example_4_register_library_kernels():
    """Example 4: Register Library Kernels with Dispatcher"""
    print("=" * 80)
    print("Example 4: Register Library Kernels")
    print("=" * 80)
    
    # Create registry
    registry = ckd.Registry()
    
    # Create library backend
    backend = LibraryBackend()
    
    # Discover and register kernels
    kernels = backend.discover_kernels()
    
    for kernel in kernels:
        registry.register(
            kernel,
            priority=ckd.Priority.NORMAL,  # Library kernels get normal priority
            backend_type="library"
        )
    
    print(f"✓ Registered {len(kernels)} library kernels")
    registry.print_stats()
    print()


def example_5_mixed_backend_registration():
    """Example 5: Register Kernels from Multiple Backends"""
    print("=" * 80)
    print("Example 5: Mixed Backend Registration")
    print("=" * 80)
    
    # Create registry
    registry = ckd.Registry()
    
    # Register tile kernels (high priority)
    tile_backend = TileBackend()
    tile_kernels = tile_backend.discover_kernels("build/tile_engine/generated")
    
    for kernel in tile_kernels:
        registry.register(kernel, priority=ckd.Priority.HIGH, backend_type="tile")
    
    print(f"✓ Registered {len(tile_kernels)} tile kernels (HIGH priority)")
    
    # Register library kernels (normal priority)
    lib_backend = LibraryBackend()
    lib_kernels = lib_backend.discover_kernels()
    
    for kernel in lib_kernels:
        registry.register(kernel, priority=ckd.Priority.NORMAL, backend_type="library")
    
    print(f"✓ Registered {len(lib_kernels)} library kernels (NORMAL priority)")
    
    # Show statistics
    print("\nRegistry statistics:")
    registry.print_stats()
    
    # Demonstrate conflict resolution
    print("\nConflict resolution:")
    print("  - Tile kernels have HIGH priority")
    print("  - Library kernels have NORMAL priority")
    print("  - When both exist for same config, Tile kernel is selected")
    print()


def example_6_backend_type_filtering():
    """Example 6: Filter Kernels by Backend Type"""
    print("=" * 80)
    print("Example 6: Filter by Backend Type")
    print("=" * 80)
    
    # Create registry with mixed backends
    registry = ckd.Registry()
    
    # Register from both backends
    tile_backend = TileBackend()
    lib_backend = LibraryBackend()
    
    tile_kernels = tile_backend.discover_kernels("build/tile_engine/generated")
    lib_kernels = lib_backend.discover_kernels()
    
    for k in tile_kernels:
        registry.register(k, backend_type="tile")
    for k in lib_kernels:
        registry.register(k, backend_type="library")
    
    # Filter by backend type
    print("Filtering kernels by backend type:")
    
    tile_only = registry.filter(
        lambda k: k.get_backend_type() == BackendType.TILE
    )
    print(f"  Tile kernels: {len(tile_only)}")
    
    lib_only = registry.filter(
        lambda k: k.get_backend_type() == BackendType.LIBRARY
    )
    print(f"  Library kernels: {len(lib_only)}")
    
    print()


def example_7_kernel_execution():
    """Example 7: Execute Kernel from Backend"""
    print("=" * 80)
    print("Example 7: Kernel Execution")
    print("=" * 80)
    
    # Create test problem
    M, N, K = 256, 256, 256
    A = np.random.randn(M, K).astype(np.float16)
    B = np.random.randn(K, N).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)
    
    # Create problem specification
    problem = ckd.Problem(M=M, N=N, K=K)
    
    # Get a tile kernel
    backend = TileBackend()
    kernels = backend.discover_kernels("build/tile_engine/generated")
    
    if kernels:
        kernel = kernels[0]
        
        print(f"Executing kernel: {kernel.get_name()}")
        print(f"Backend type: {kernel.get_backend_type().value}")
        
        # Check if kernel supports problem
        if kernel.supports(problem):
            # Execute
            time_ms = kernel.run(A, B, C, problem)
            
            print(f"✓ Execution time: {time_ms:.3f} ms")
            
            # Validate
            is_correct = kernel.validate(A, B, C, problem)
            print(f"✓ Validation: {'PASS' if is_correct else 'FAIL'}")
        else:
            print("✗ Kernel does not support this problem")
    else:
        print("No kernels found")
    
    print()


def example_8_backend_metadata():
    """Example 8: Inspect Backend Metadata"""
    print("=" * 80)
    print("Example 8: Backend Metadata")
    print("=" * 80)
    
    # Create backends
    tile_backend = TileBackend()
    lib_backend = LibraryBackend()
    
    print("Tile Backend:")
    print(f"  Type: {tile_backend.get_backend_type().value}")
    print(f"  {tile_backend}")
    
    print("\nLibrary Backend:")
    print(f"  Type: {lib_backend.get_backend_type().value}")
    print(f"  {lib_backend}")
    print(f"  Operations: {lib_backend.enumerate_operations()}")
    
    print()


def example_9_custom_backend():
    """Example 9: Custom Backend Implementation"""
    print("=" * 80)
    print("Example 9: Custom Backend (Concept)")
    print("=" * 80)
    
    print("To create a custom backend:")
    print("  1. Inherit from BackendBase")
    print("  2. Implement discover_kernels()")
    print("  3. Implement create_kernel_instance()")
    print("  4. Implement get_backend_type()")
    print()
    print("Example:")
    print("""
    class MyCustomBackend(BackendBase):
        def discover_kernels(self, search_path):
            # Discover kernels from custom source
            return [...]
        
        def create_kernel_instance(self, config):
            # Create kernel instance
            return MyKernelInstance(...)
        
        def get_backend_type(self):
            return BackendType.UNKNOWN
    """)
    print()


def main():
    """Run all examples"""
    examples = [
        example_1_tile_backend_discovery,
        example_2_library_backend_discovery,
        example_3_register_tile_kernels,
        example_4_register_library_kernels,
        example_5_mixed_backend_registration,
        example_6_backend_type_filtering,
        example_7_kernel_execution,
        example_8_backend_metadata,
        example_9_custom_backend,
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

