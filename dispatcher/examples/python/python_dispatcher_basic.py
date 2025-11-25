#!/usr/bin/env python3
"""
Basic Python Dispatcher Example - Using C++ Extension

Demonstrates:
1. Importing C++ dispatcher bindings
2. Creating Problem and KernelKey objects
3. Using Registry to query kernels
4. Using Dispatcher to select kernels

This example focuses on the dispatcher API without GPU execution.
"""

import sys
from pathlib import Path

# Add Python module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

try:
    import _dispatcher_native as cpp

    print("OK C++ extension loaded successfully\n")
except ImportError as e:
    print("[FAIL] Failed to load C++ extension")
    print(f"   Error: {e}")
    print("\n   Build with: -DBUILD_DISPATCHER_PYTHON=ON")
    print("   Run with: PYTHONPATH=../python python3 this_script.py\n")
    sys.exit(1)


def demo_problem_api():
    """Demo: Problem class"""
    print("=" * 70)
    print("Demo 1: Problem API")
    print("=" * 70 + "\n")

    # Create problems
    p1 = cpp.Problem()
    print(f"Empty problem: {p1}")
    print(f"  Valid: {p1.is_valid()}")
    print()

    p2 = cpp.Problem(1024, 1024, 1024)
    print(f"Problem 1024³: {p2}")
    print(f"  M={p2.M}, N={p2.N}, K={p2.K}")
    print(f"  Valid: {p2.is_valid()}")
    print(f"  Ops: {p2.num_ops():,}")
    print()

    # Modify problem
    p2.k_batch = 2
    p2.smem_budget = 65536
    print("Modified problem:")
    print(f"  k_batch: {p2.k_batch}")
    print(f"  smem_budget: {p2.smem_budget}")
    print()


def demo_kernel_key_api():
    """Demo: KernelKey construction"""
    print("=" * 70)
    print("Demo 2: KernelKey API")
    print("=" * 70 + "\n")

    # Create kernel key
    key = cpp.KernelKey()

    # Set signature
    key.signature.dtype_a = cpp.DataType.FP16
    key.signature.dtype_b = cpp.DataType.FP16
    key.signature.dtype_c = cpp.DataType.FP16
    key.signature.dtype_acc = cpp.DataType.FP32
    key.signature.layout_a = cpp.LayoutTag.RowMajor
    key.signature.layout_b = cpp.LayoutTag.ColMajor
    key.signature.layout_c = cpp.LayoutTag.RowMajor
    key.signature.elementwise_op = "PassThrough"
    key.signature.split_k = 1

    # Set algorithm
    key.algorithm.tile_shape.m = 128
    key.algorithm.tile_shape.n = 128
    key.algorithm.tile_shape.k = 32
    key.algorithm.wave_shape.m = 2
    key.algorithm.wave_shape.n = 2
    key.algorithm.wave_shape.k = 1
    key.algorithm.pipeline = cpp.Pipeline.CompV4
    key.algorithm.scheduler = cpp.Scheduler.Intrawave
    key.algorithm.epilogue = cpp.Epilogue.CShuffle
    key.algorithm.block_size = 256

    key.gfx_arch = "gfx942"

    print(f"Created KernelKey: {key}")
    print(f"  Identifier: {key.encode_identifier()}")
    print()

    # Create another key and compare
    key2 = cpp.KernelKey()
    key2.signature.dtype_a = cpp.DataType.FP16
    key2.gfx_arch = "gfx942"

    print("Key equality:")
    print(f"  key == key: {key == key}")
    print(f"  key == key2: {key == key2}")
    print()


def demo_registry_api():
    """Demo: Registry operations"""
    print("=" * 70)
    print("Demo 3: Registry API")
    print("=" * 70 + "\n")

    registry = cpp.Registry.instance()
    print(f"Registry: {registry}")
    print(f"  Current size: {len(registry)}")
    print()

    # In a real scenario, kernels would be registered from C++ side
    # This demo just shows the API
    print("Registry operations available:")
    print("  - registry.size() - Get number of registered kernels")
    print("  - registry.get_all() - Get all kernels")
    print("  - registry.lookup(name) - Find kernel by name")
    print("  - registry.filter(problem) - Find kernels for problem")
    print("  - registry.clear() - Clear all registrations")
    print()

    # Note: We can't register mock kernels from Python easily
    # since KernelInstance is abstract and needs C++ implementation
    print("Note: Kernel registration typically done from C++ side")
    print()


def demo_dispatcher_api():
    """Demo: Dispatcher usage"""
    print("=" * 70)
    print("Demo 4: Dispatcher API")
    print("=" * 70 + "\n")

    # Create dispatcher
    dispatcher = cpp.Dispatcher()
    print(f"Dispatcher: {dispatcher}")
    print()

    # Set strategy
    print("Selection strategies:")
    print(f"  - FirstFit: {cpp.SelectionStrategy.FirstFit}")
    print(f"  - Heuristic: {cpp.SelectionStrategy.Heuristic}")
    print()

    dispatcher.set_strategy(cpp.SelectionStrategy.FirstFit)
    print("OK Set strategy to FirstFit")
    print()

    # Define a heuristic function
    def my_heuristic(problem):
        """Example heuristic: prefer large tiles for large problems"""
        if problem.M >= 1000 and problem.N >= 1000:
            return ["256x256x32_4x4x1_32x32x16_nopers"]
        else:
            return ["128x128x32_2x2x1_32x32x16_nopers"]

    dispatcher.set_heuristic(my_heuristic)
    print("OK Set custom heuristic")
    print()

    # Try selection (will fail without registered kernels)
    problem = cpp.Problem(1024, 1024, 1024)
    kernel = dispatcher.select_kernel(problem)

    if kernel is None:
        print("No kernel selected (registry is empty)")
        print("  In real usage, kernels would be registered from C++")
    else:
        print(f"Selected kernel: {kernel.get_name()}")
    print()


def demo_enums():
    """Demo: Available enums"""
    print("=" * 70)
    print("Demo 5: Available Enums")
    print("=" * 70 + "\n")

    print("DataTypes:")
    for dtype in [
        cpp.DataType.FP16,
        cpp.DataType.BF16,
        cpp.DataType.FP32,
        cpp.DataType.FP8,
        cpp.DataType.INT8,
    ]:
        print(f"  - {dtype}")
    print()

    print("Layouts:")
    for layout in [cpp.LayoutTag.RowMajor, cpp.LayoutTag.ColMajor]:
        print(f"  - {layout}")
    print()

    print("Pipelines:")
    for pipe in [cpp.Pipeline.Mem, cpp.Pipeline.CompV3, cpp.Pipeline.CompV4]:
        print(f"  - {pipe}")
    print()

    print("Schedulers:")
    for sched in [cpp.Scheduler.Auto, cpp.Scheduler.Intrawave, cpp.Scheduler.Interwave]:
        print(f"  - {sched}")
    print()

    print("Priorities:")
    for prio in [cpp.Priority.Low, cpp.Priority.Normal, cpp.Priority.High]:
        print(f"  - {prio}")
    print()


def main():
    print("\n" + "=" * 70)
    print("CK Tile Dispatcher - Python C++ Extension Demo")
    print("=" * 70 + "\n")

    print(f"Module version: {cpp.__version__}")
    print(f"Module location: {cpp.__file__}")
    print()

    demo_problem_api()
    demo_kernel_key_api()
    demo_registry_api()
    demo_dispatcher_api()
    demo_enums()

    print("=" * 70)
    print("All Demos Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  OK C++ extension provides low-level dispatcher access")
    print("  OK Problem, KernelKey, Registry, Dispatcher all available")
    print("  OK Can set heuristics from Python")
    print("  OK Kernel registration happens from C++ side")
    print("  OK Use dispatcher_api.py for high-level functionality")
    print()


if __name__ == "__main__":
    main()
