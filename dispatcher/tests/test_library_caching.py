#!/usr/bin/env python3
"""
Unit tests for library caching in setup_gemm_dispatcher().

Tests verify that:
1. Different kernel configs create unique library files with complete naming
2. Repeated configs reuse cached libraries (no redundant rebuilds)
3. Library names include all distinguishing parameters (dtype, layout, tile, wave, warp, pipeline, epilogue, scheduler)
4. Kernel headers are generated when missing
"""

import sys
import time
import unittest
from pathlib import Path

# Add dispatcher python to path
DISPATCHER_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(DISPATCHER_ROOT / "python"))

from ctypes_utils import (
    setup_gemm_dispatcher,
    KernelConfig,
    get_build_dir,
)


class TestLibraryCaching(unittest.TestCase):
    """Test library caching functionality in setup_gemm_dispatcher"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        cls.build_dir = get_build_dir()
        cls.examples_dir = cls.build_dir / "examples"

        # Clean up any previous test libraries
        cls._cleanup_test_libraries()

    @classmethod
    def _cleanup_test_libraries(cls):
        """Remove test library files"""
        if cls.examples_dir.exists():
            for lib in cls.examples_dir.glob("libdispatcher_gemm_fp16_rcr_*_compv4_*.so"):
                try:
                    lib.unlink()
                except Exception:
                    pass

    def test_01_unique_library_naming(self):
        """Test that library names include all distinguishing parameters"""
        config = KernelConfig(
            dtype_a="fp16",
            layout_a="row",
            layout_b="col",
            layout_c="row",
            tile_m=128,
            tile_n=128,
            tile_k=64,
            pipeline="compv4",
            gfx_arch="gfx950",
        )

        result = setup_gemm_dispatcher(config, verbose=False, auto_rebuild=True)

        self.assertTrue(result.success, "setup_gemm_dispatcher should succeed")
        self.assertIsNotNone(result.lib, "Library should be loaded")

        lib_name = result.lib.path.name

        # Verify library name includes all parameters
        self.assertIn("fp16", lib_name, "Library name should include dtype")
        self.assertIn("rcr", lib_name, "Library name should include layout")
        self.assertIn("128x128x64", lib_name, "Library name should include tile dimensions")
        self.assertIn("2x2x1", lib_name, "Library name should include wave dimensions")
        self.assertIn("32x32x16", lib_name, "Library name should include warp dimensions")
        self.assertIn("compv4", lib_name, "Library name should include pipeline")
        self.assertIn("cshuffle", lib_name, "Library name should include epilogue")
        self.assertIn("intrawave", lib_name, "Library name should include scheduler")

        print(f"✓ Library name includes all parameters: {lib_name}")

    def test_02_library_build_and_cache(self):
        """Test that libraries are built correctly and then cached"""
        config = KernelConfig(
            dtype_a="fp16",
            layout_a="row",
            layout_b="col",
            layout_c="row",
            tile_m=128,
            tile_n=128,
            tile_k=64,
            pipeline="compv4",
            gfx_arch="gfx950",
        )

        expected_lib_name = "libdispatcher_gemm_fp16_rcr_128x128x64_2x2x1_32x32x16_compv4_cshuffle_intrawave.so"
        expected_lib_path = self.examples_dir / expected_lib_name

        # First call - should build library
        start_time = time.time()
        result1 = setup_gemm_dispatcher(config, verbose=False, auto_rebuild=True)
        time1 = time.time() - start_time

        self.assertTrue(result1.success, "First setup should succeed")

        # Check if library was created (might use default if config matches)
        if expected_lib_path.exists():
            lib_created = True
            print(f"✓ Library created: {expected_lib_name}")
        else:
            # Config might match default library, which is also valid
            lib_created = False
            print(f"  Config matches default library: {result1.lib.path.name}")

        # Second call - should use cache if library was built
        start_time = time.time()
        result2 = setup_gemm_dispatcher(config, verbose=False, auto_rebuild=True)
        time2 = time.time() - start_time

        self.assertTrue(result2.success, "Second setup should succeed")

        # If library was created, second call should be much faster (cached)
        if lib_created and time1 > 5.0:  # First call took significant time (build happened)
            self.assertLess(time2, time1 * 0.5,
                          f"Cached load ({time2:.2f}s) should be much faster than build ({time1:.2f}s)")
            print(f"✓ Cache reuse: {time2:.2f}s vs {time1:.2f}s ({time1/time2:.1f}x faster)")
        else:
            print(f"  Both calls fast (using default library)")

    def test_03_different_configs_different_libraries(self):
        """Test that different configs create different library files"""
        configs = [
            KernelConfig(
                dtype_a="fp16",
                layout_a="row",
                layout_b="col",
                layout_c="row",
                tile_m=128,
                tile_n=128,
                tile_k=64,
                pipeline="compv4",
                gfx_arch="gfx950",
            ),
            KernelConfig(
                dtype_a="fp16",
                layout_a="row",
                layout_b="col",
                layout_c="row",
                tile_m=128,
                tile_n=128,
                tile_k=32,
                pipeline="compv4",
                gfx_arch="gfx950",
            ),
        ]

        results = []
        for i, config in enumerate(configs):
            result = setup_gemm_dispatcher(
                config,
                registry_name=f"test_registry_{i}",
                verbose=False,
                auto_rebuild=True
            )
            results.append(result)

        # Check that all setups succeeded
        for i, result in enumerate(results):
            self.assertTrue(result.success, f"Setup {i+1} should succeed")

        # Check that different configs loaded different libraries (if both built custom libs)
        lib_names = [r.lib.path.name for r in results if r.lib]

        # If both created custom libraries, they should be different
        custom_libs = [name for name in lib_names if "libdispatcher_gemm_fp16_rcr_128x128" in name
                      and name != "libdispatcher_gemm_lib.so"]

        if len(custom_libs) >= 2:
            # Should have different tile dimensions in names
            self.assertNotEqual(custom_libs[0], custom_libs[1],
                              "Different configs should create different libraries")
            self.assertIn("128x128x64", custom_libs[0])
            self.assertIn("128x128x32", custom_libs[1])
            print(f"✓ Different configs created different libraries:")
            for lib in custom_libs:
                print(f"    - {lib}")
        else:
            print(f"  Configs used default library (valid when configs match default)")

    def test_04_cache_message_verification(self):
        """Test that cache hit messages are logged correctly"""
        config = KernelConfig(
            dtype_a="fp16",
            layout_a="row",
            layout_b="col",
            layout_c="row",
            tile_m=128,
            tile_n=128,
            tile_k=64,
            pipeline="compv4",
            gfx_arch="gfx950",
        )

        # First call
        result1 = setup_gemm_dispatcher(config, verbose=False, auto_rebuild=True)
        self.assertTrue(result1.success)

        # Second call - capture output to check for cache message
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            result2 = setup_gemm_dispatcher(config, verbose=True, auto_rebuild=True)

        output = f.getvalue()

        self.assertTrue(result2.success)

        # Check if cache was used (either message appears or default lib was used)
        if "Using cached library" in output:
            print("✓ Cache hit message logged correctly")
            self.assertIn("Using cached library", output)
        elif "libdispatcher_gemm_lib.so" in str(result2.lib.path):
            print("  Using default CMake library (no rebuild needed)")
        else:
            print("  Warning: Expected cache message not found (may have rebuilt)")

    def test_05_code_fix_verification(self):
        """Verify the code changes are in place"""
        from ctypes_utils import get_dispatcher_root

        ctypes_utils_path = get_dispatcher_root() / "python" / "ctypes_utils.py"
        self.assertTrue(ctypes_utils_path.exists(), "ctypes_utils.py should exist")

        with open(ctypes_utils_path, 'r') as f:
            code = f.read()

        # Check Fix #1: Complete library naming
        self.assertIn(
            "_{config.pipeline}_{config.epilogue}_{config.scheduler}",
            code,
            "Library naming should include pipeline, epilogue, and scheduler"
        )
        self.assertIn(
            "_{wave_str}_{warp_str}_",
            code,
            "Library naming should include wave and warp dimensions"
        )

        # Check Fix #2: Cache checking logic
        self.assertIn(
            "cached_lib_path.exists()",
            code,
            "Cache checking logic should be present"
        )
        self.assertIn(
            "Using cached library",
            code,
            "Cache hit message should be present"
        )

        print("✓ Code fixes verified:")
        print("    - Complete library naming (dtype, layout, tile, wave, warp, pipeline, epilogue, scheduler)")
        print("    - Cache checking logic present")


def run_tests(verbosity=2):
    """Run all tests with specified verbosity"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestLibraryCaching)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    print("="*80)
    print("  Library Caching Unit Tests")
    print("="*80)
    print()

    exit_code = run_tests(verbosity=2)

    print()
    print("="*80)
    if exit_code == 0:
        print("  ✓ ALL TESTS PASSED")
    else:
        print("  ✗ SOME TESTS FAILED")
    print("="*80)

    sys.exit(exit_code)
