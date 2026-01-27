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
    example_path: Path, timeout: int = 120
) -> subprocess.CompletedProcess:
    """Run a Python example and capture output."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PYTHON_DIR)

    return subprocess.run(
        [sys.executable, str(example_path)],
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
        # Should pass validation
        self.assertIn("PASS", result.stdout.upper(), "Validation should pass")


class TestConvPythonExamples(unittest.TestCase):
    """Test Conv Python examples."""

    @classmethod
    def setUpClass(cls):
        """Check if examples directory exists."""
        cls.conv_examples_dir = EXAMPLES_DIR / "conv" / "python"
        if not cls.conv_examples_dir.exists():
            raise unittest.SkipTest("Conv Python examples not found")

    def test_01_basic_conv(self):
        """Test basic conv example."""
        example = self.conv_examples_dir / "01_basic_conv.py"
        if not example.exists():
            self.skipTest(f"{example.name} not found")

        result = run_python_example(example)

        self.assertEqual(result.returncode, 0, f"Example failed:\n{result.stderr}")
        self.assertIn("TFLOPS", result.stdout, "Should report TFLOPS")

    def test_02_conv2d_fwd(self):
        """Test 2D forward conv example."""
        example = self.conv_examples_dir / "02_conv2d_fwd.py"
        if not example.exists():
            self.skipTest(f"{example.name} not found")

        result = run_python_example(example)

        self.assertEqual(result.returncode, 0, f"Example failed:\n{result.stderr}")

    def test_03_conv3d_fwd(self):
        """Test 3D forward conv example."""
        example = self.conv_examples_dir / "03_conv3d_fwd.py"
        if not example.exists():
            self.skipTest(f"{example.name} not found")

        result = run_python_example(example)

        self.assertEqual(result.returncode, 0, f"Example failed:\n{result.stderr}")

    def test_07_validation(self):
        """Test validation example."""
        example = self.conv_examples_dir / "07_validation.py"
        if not example.exists():
            self.skipTest(f"{example.name} not found")

        result = run_python_example(example)

        self.assertEqual(result.returncode, 0, f"Example failed:\n{result.stderr}")
        self.assertIn("PASS", result.stdout.upper(), "Validation should pass")


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

    def test_gemm_04_validation(self):
        """Test validation GEMM C++ example."""
        result = run_cpp_example("gemm_04_validation")
        if result is None:
            self.skipTest("gemm_04_validation not built")

        self.assertEqual(result.returncode, 0, f"Example failed:\n{result.stderr}")
        self.assertIn("PASS", result.stdout.upper(), "Validation should pass")


class TestConvCppExamples(unittest.TestCase):
    """Test Conv C++ examples."""

    @classmethod
    def setUpClass(cls):
        """Check if build directory exists."""
        cls.examples_dir = BUILD_DIR / "examples"
        if not cls.examples_dir.exists():
            raise unittest.SkipTest("C++ examples not built")

    def test_conv_01_forward(self):
        """Test forward conv C++ example."""
        result = run_cpp_example("conv_01_forward")
        if result is None:
            self.skipTest("conv_01_forward not built")

        self.assertEqual(result.returncode, 0, f"Example failed:\n{result.stderr}")
        self.assertIn("TFLOPS", result.stdout, "Should report TFLOPS")

    def test_conv_02_validation(self):
        """Test validation conv C++ example."""
        result = run_cpp_example("conv_02_validation")
        if result is None:
            self.skipTest("conv_02_validation not built")

        self.assertEqual(result.returncode, 0, f"Example failed:\n{result.stderr}")
        self.assertIn("PASS", result.stdout.upper(), "Validation should pass")


class TestUtilityImports(unittest.TestCase):
    """Test that utility modules can be imported."""

    def test_import_ctypes_utils(self):
        """Test importing ctypes_utils."""
        try:
            from ctypes_utils import KernelConfig, setup_gemm_dispatcher  # noqa: F401

            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import ctypes_utils: {e}")

    def test_import_conv_utils(self):
        """Test importing conv_utils."""
        try:
            from conv_utils import ConvSignature, ConvAlgorithm, ConvProblem  # noqa: F401

            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import conv_utils: {e}")

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

    def test_conv_signature_creation(self):
        """Test creating a ConvSignature."""
        from conv_utils import ConvSignature

        sig = ConvSignature(
            dtype_in="fp16",
            dtype_wei="fp16",
            dtype_out="fp16",
            dtype_acc="fp32",
            layout="nhwgc",
            direction="forward",
            num_dims=2,
        )

        self.assertEqual(sig.dtype_in, "fp16")
        self.assertEqual(sig.direction, "forward")


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

    def test_conv_auto_correct(self):
        """Test Conv auto-correction."""
        from conv_utils import auto_correct_conv_config

        # Call with invalid wave config parameters
        corrected, was_modified, corrections = auto_correct_conv_config(
            wave_m=99,  # Invalid
            wave_n=99,  # Invalid
            wave_k=99,  # Invalid
            dtype="fp16",
            arch="gfx942",
        )

        self.assertTrue(was_modified, "Config should be modified")
        self.assertGreater(len(corrections), 0, "Should have corrections")


if __name__ == "__main__":
    unittest.main()
