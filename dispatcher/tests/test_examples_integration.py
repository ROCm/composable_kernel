#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Integration tests that verify examples work correctly.

These tests mimic the examples to ensure they continue working.
Run with: pytest test_examples_integration.py -v
"""

import unittest
import subprocess
import sys
import os
from pathlib import Path

# Get paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_ROOT = SCRIPT_DIR.parent
EXAMPLES_DIR = DISPATCHER_ROOT / "examples"
BUILD_DIR = DISPATCHER_ROOT / "build"
PYTHON_DIR = DISPATCHER_ROOT / "python"

# Add python utilities to path
sys.path.insert(0, str(PYTHON_DIR))


def run_python_example(
    example_path: Path, timeout: int = 120, extra_args: list = None
) -> subprocess.CompletedProcess:
    """Run a Python example and capture output."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PYTHON_DIR)

    cmd = [sys.executable, str(example_path)]
    if extra_args:
        cmd.extend(extra_args)

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=example_path.parent,
        env=env,
    )


def run_cpp_example(
    example_name: str, timeout: int = 60
) -> subprocess.CompletedProcess:
    """Run a C++ example and capture output."""
    example_path = BUILD_DIR / "examples" / example_name

    if not example_path.exists():
        return None

    return subprocess.run(
        [str(example_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class TestGemmPythonExamples(unittest.TestCase):
    """Test GEMM Python examples."""

    @classmethod
    def setUpClass(cls):
        """Check if examples directory exists."""
        cls.gemm_examples_dir = EXAMPLES_DIR / "gemm" / "python"
        if not cls.gemm_examples_dir.exists():
            raise unittest.SkipTest("GEMM Python examples not found")

    def test_01_basic_gemm(self):
        """Test basic GEMM example."""
        example = self.gemm_examples_dir / "01_basic_gemm.py"
        if not example.exists():
            self.skipTest(f"{example.name} not found")

        result = run_python_example(example)

        self.assertEqual(result.returncode, 0, f"Example failed:\n{result.stderr}")
        self.assertIn("TFLOPS", result.stdout, "Should report TFLOPS")

    def test_02_batch_gemm(self):
        """Test batch GEMM example."""
        example = self.gemm_examples_dir / "02_batch_gemm.py"
        if not example.exists():
            self.skipTest(f"{example.name} not found")

        result = run_python_example(example)

        self.assertEqual(result.returncode, 0, f"Example failed:\n{result.stderr}")

    def test_03_benchmark(self):
        """Test benchmark example."""
        example = self.gemm_examples_dir / "03_benchmark.py"
        if not example.exists():
            self.skipTest(f"{example.name} not found")

        result = run_python_example(example)

        self.assertEqual(result.returncode, 0, f"Example failed:\n{result.stderr}")

    def test_04_validation(self):
        """Test validation example."""
        example = self.gemm_examples_dir / "04_validation.py"
        if not example.exists():
            self.skipTest(f"{example.name} not found")

        result = run_python_example(example)

        self.assertEqual(result.returncode, 0, f"Example failed:\n{result.stderr}")
        self.assertIn("PASS", result.stdout.upper(), "Validation should pass")


class TestConvPythonExamples(unittest.TestCase):
    """Test grouped conv Python examples."""

    @classmethod
    def setUpClass(cls):
        """Check if examples directory exists."""
        cls.conv_examples_dir = EXAMPLES_DIR / "grouped_conv" / "python"
        if not cls.conv_examples_dir.exists():
            raise unittest.SkipTest("Grouped conv Python examples not found")

    def test_01_basic_grouped_conv(self):
        """Test basic grouped conv example."""
        example = self.conv_examples_dir / "01_basic_grouped_conv.py"
        if not example.exists():
            self.skipTest(f"{example.name} not found")
        result = run_python_example(example)
        self.assertEqual(result.returncode, 0, f"Example failed:\n{result.stderr}")
        self.assertIn("PASS", result.stdout.upper())

    def test_02_forward(self):
        """Test forward conv example (2D + 3D)."""
        example = self.conv_examples_dir / "02_forward.py"
        if not example.exists():
            self.skipTest(f"{example.name} not found")
        result = run_python_example(example)
        self.assertEqual(result.returncode, 0, f"Example failed:\n{result.stderr}")
        self.assertIn("PASS", result.stdout.upper())

    def test_03_bwd_data(self):
        """Test backward data example."""
        example = self.conv_examples_dir / "03_bwd_data.py"
        if not example.exists():
            self.skipTest(f"{example.name} not found")
        result = run_python_example(example)
        self.assertEqual(result.returncode, 0, f"Example failed:\n{result.stderr}")
        self.assertIn("PASS", result.stdout.upper())

    def test_04_bwd_weight(self):
        """Test backward weight example."""
        example = self.conv_examples_dir / "04_bwd_weight.py"
        if not example.exists():
            self.skipTest(f"{example.name} not found")
        result = run_python_example(example)
        self.assertEqual(result.returncode, 0, f"Example failed:\n{result.stderr}")
        self.assertIn("PASS", result.stdout.upper())

    def test_05_benchmark(self):
        """Test benchmark example."""
        example = self.conv_examples_dir / "05_benchmark.py"
        if not example.exists():
            self.skipTest(f"{example.name} not found")
        result = run_python_example(
            example, extra_args=["--warmup", "1", "--repeat", "1"]
        )
        self.assertEqual(result.returncode, 0, f"Example failed:\n{result.stderr}")
        self.assertIn("PASS", result.stdout.upper())

    def test_06_registry_json(self):
        """Test registry + heuristic + JSON example."""
        example = self.conv_examples_dir / "06_registry_json.py"
        if not example.exists():
            self.skipTest(f"{example.name} not found")
        result = run_python_example(example)
        self.assertEqual(result.returncode, 0, f"Example failed:\n{result.stderr}")
        self.assertIn("PASS", result.stdout.upper())


class TestGemmCppExamples(unittest.TestCase):
    """Test GEMM C++ examples."""

    @classmethod
    def setUpClass(cls):
        """Check if build directory exists."""
        cls.examples_dir = BUILD_DIR / "examples"
        if not cls.examples_dir.exists():
            raise unittest.SkipTest("C++ examples not built")

    def test_gemm_01_basic(self):
        """Test basic GEMM C++ example."""
        result = run_cpp_example("gemm_01_basic")
        if result is None:
            self.skipTest("gemm_01_basic not built")

        self.assertEqual(result.returncode, 0, f"Example failed:\n{result.stderr}")
        self.assertIn("TFLOPS", result.stdout, "Should report TFLOPS")

    def test_gemm_02_multi_size(self):
        """Test multi-size GEMM C++ example."""
        result = run_cpp_example("gemm_02_multi_size")
        if result is None:
            self.skipTest("gemm_02_multi_size not built")

        self.assertEqual(result.returncode, 0, f"Example failed:\n{result.stderr}")

    def test_gemm_03_benchmark_validation(self):
        """Test benchmark+validation GEMM C++ example."""
        result = run_cpp_example("gemm_03_benchmark_validation")
        if result is None:
            self.skipTest("gemm_03_benchmark_validation not built")

        self.assertEqual(result.returncode, 0, f"Example failed:\n{result.stderr}")
        self.assertIn("PASS", result.stdout.upper(), "Validation should pass")


class TestConvCppExamples(unittest.TestCase):
    """Test grouped conv C++ examples."""

    @classmethod
    def setUpClass(cls):
        """Check if build directory exists."""
        cls.examples_dir = BUILD_DIR / "examples"
        if not cls.examples_dir.exists():
            raise unittest.SkipTest("C++ examples not built")

    def test_grouped_conv_01_basic(self):
        """Test basic grouped conv C++ example."""
        result = run_cpp_example("grouped_conv_01_basic")
        if result is None:
            self.skipTest("grouped_conv_01_basic not built")
        self.assertEqual(result.returncode, 0, f"Example failed:\n{result.stderr}")
        self.assertIn("PASS", result.stdout.upper())

    def test_grouped_conv_02_all_dirs(self):
        """Test all directions grouped conv C++ example."""
        result = run_cpp_example("grouped_conv_02_all_dirs")
        if result is None:
            self.skipTest("grouped_conv_02_all_dirs not built")
        self.assertEqual(result.returncode, 0, f"Example failed:\n{result.stderr}")
        self.assertIn("PASS", result.stdout.upper())

    def test_grouped_conv_03_bench_val(self):
        """Test benchmark+validation grouped conv C++ example."""
        result = run_cpp_example("grouped_conv_03_bench_val")
        if result is None:
            self.skipTest("grouped_conv_03_bench_val not built")
        self.assertEqual(result.returncode, 0, f"Example failed:\n{result.stderr}")
        self.assertIn("PASS", result.stdout.upper())


class TestUtilityImports(unittest.TestCase):
    """Test that utility modules can be imported."""

    def test_import_ctypes_utils(self):
        """Test importing ctypes_utils."""
        try:
            from ctypes_utils import KernelConfig, setup_gemm_dispatcher  # noqa: F401

            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import ctypes_utils: {e}")

    def test_import_grouped_conv_utils(self):
        """Test importing grouped_conv_utils."""
        try:
            from grouped_conv_utils import (  # noqa: F401
                GroupedConvValidationResult,
                validate_grouped_conv_config,
                GroupedConvDataType,
            )

            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import grouped_conv_utils: {e}")

    def test_kernel_config_creation(self):
        """Test creating a KernelConfig."""
        from ctypes_utils import KernelConfig

        config = KernelConfig(
            dtype_a="fp16",
            dtype_b="fp16",
            dtype_c="fp16",
            dtype_acc="fp32",
            layout_a="row",
            layout_b="col",
            layout_c="row",
        )

        self.assertEqual(config.dtype_a, "fp16")
        self.assertEqual(config.layout_a, "row")

    def test_grouped_conv_default_config(self):
        """Test creating a grouped conv default config."""
        from grouped_conv_utils import get_grouped_conv_default_config

        config = get_grouped_conv_default_config(
            variant="forward",
            ndim_spatial=2,
            arch="gfx942",
        )

        d = config.to_dict() if hasattr(config, "to_dict") else config
        self.assertEqual(d["variant"], "forward")
        self.assertEqual(d["arch"], "gfx942")


class TestAutoCorrection(unittest.TestCase):
    """Test auto-correction functionality."""

    def test_gemm_auto_correct(self):
        """Test GEMM auto-correction."""
        from ctypes_utils import KernelConfig, auto_correct_kernel_config

        # Create a config with invalid wave config
        config = KernelConfig(
            dtype_a="fp16",
            dtype_b="fp16",
            dtype_c="fp16",
            dtype_acc="fp32",
            layout_a="row",
            layout_b="col",
            layout_c="row",
            wave_m=99,  # Invalid
            wave_n=99,  # Invalid
            wave_k=99,  # Invalid
        )

        corrected, was_modified, corrections = auto_correct_kernel_config(config)

        self.assertTrue(was_modified, "Config should be modified")
        self.assertGreater(len(corrections), 0, "Should have corrections")

    def test_grouped_conv_auto_correct(self):
        """Test Grouped Conv auto-correction."""
        from grouped_conv_utils import (
            auto_correct_grouped_conv_config,
            get_grouped_conv_default_config,
        )

        config = get_grouped_conv_default_config()
        d = config.to_dict() if hasattr(config, "to_dict") else config
        d["tile_config"]["warp_m"] = [99]
        d["tile_config"]["warp_n"] = [99]

        corrected, result = auto_correct_grouped_conv_config(d)

        self.assertIsInstance(corrected, dict)
        self.assertIn("tile_config", corrected)


if __name__ == "__main__":
    unittest.main()
