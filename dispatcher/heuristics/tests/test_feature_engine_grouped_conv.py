#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Unit tests for feature_engine_grouped_conv.py - Grouped Convolution Feature Engineering.

Tests the feature extraction logic for ML-based kernel selection.
Run: python3 -m pytest heuristics/tests/test_feature_engine_grouped_conv.py -v
"""

import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directories to path
SCRIPT_DIR = Path(__file__).parent.resolve()
HEURISTICS_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(HEURISTICS_DIR))

from feature_engine_grouped_conv import GroupedConvFeatureEngine  # noqa: E402


class TestGroupedConvFeatureEngine(unittest.TestCase):
    """Test suite for GroupedConvFeatureEngine."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = GroupedConvFeatureEngine()

    def test_feature_names_count(self):
        """Test that feature names list has correct length.

        After the suffix-aware kernel-feature expansion the engine emits 97
        features (was 83): the 3 wave/dsb/si flags plus the 3 added pipeline
        one-hots (basic_v1, compv6, mem) extend the kernel-features block by
        6 entries, plus 8 more interaction/spatial features added previously.
        """
        names = self.engine.get_feature_names()
        self.assertEqual(len(names), 97, f"Expected 97 features, got {len(names)}")

    def test_categorical_features(self):
        """Test categorical features identification."""
        categorical = self.engine.get_categorical_features()
        self.assertIn("pipeline", categorical)
        self.assertEqual(len(categorical), 1)

    def test_extract_basic_forward_conv(self):
        """Test feature extraction for basic forward convolution."""
        problem = {
            "N": 1,
            "C": 64,
            "K": 128,
            "G": 1,
            "Hi": 32,
            "Wi": 32,
            "Y": 3,
            "X": 3,
            "stride_h": 1,
            "stride_w": 1,
            "pad_h": 1,
            "pad_w": 1,
            "dtype": "bf16",
        }

        kernel = {
            "block_size": 16,
            "gemm_m_per_block": 64,
            "gemm_n_per_block": 64,
            "pipeline": "compv3",
        }

        features = self.engine.extract(problem, kernel)

        # Should return numpy array with 97 features (post suffix-aware update)
        self.assertEqual(features.shape, (97,))
        self.assertFalse(np.any(np.isnan(features)), "Features should not contain NaN")
        self.assertFalse(np.any(np.isinf(features)), "Features should not contain Inf")

    def test_extract_with_dilation(self):
        """Test that dilation is correctly incorporated into Ho/Wo calculation."""
        # Without dilation
        problem_no_dilation = {
            "N": 1,
            "C": 64,
            "K": 64,
            "G": 1,
            "Hi": 32,
            "Wi": 32,
            "Y": 3,
            "X": 3,
            "stride_h": 1,
            "stride_w": 1,
            "pad_h": 1,
            "pad_w": 1,
            "dilation_h": 1,
            "dilation_w": 1,
        }

        # With dilation=2
        problem_with_dilation = {
            **problem_no_dilation,
            "dilation_h": 2,
            "dilation_w": 2,
        }

        kernel = {
            "block_size": 16,
            "gemm_m_per_block": 64,
            "gemm_n_per_block": 64,
            "pipeline": "compv3",
        }

        features_no_dil = self.engine.extract(problem_no_dilation, kernel)
        features_with_dil = self.engine.extract(problem_with_dilation, kernel)

        # Ho and Wo should be different (indices 12 and 13)
        # Without dilation: Ho = (32 + 2*1 - 3) // 1 + 1 = 32
        # With dilation=2: eff_y = (3-1)*2 + 1 = 5, Ho = (32 + 2*1 - 5) // 1 + 1 = 30
        Ho_no_dil = features_no_dil[12]
        Ho_with_dil = features_with_dil[12]

        self.assertEqual(Ho_no_dil, 32, "Ho without dilation should be 32")
        self.assertEqual(Ho_with_dil, 30, "Ho with dilation=2 should be 30")

    def test_extract_batch_basic(self):
        """Test batch extraction with DataFrame input."""
        df = pd.DataFrame(
            {
                "N": [1, 2],
                "C": [64, 128],
                "K": [128, 256],
                "G": [1, 2],
                "Hi": [32, 56],
                "Wi": [32, 56],
                "Y": [3, 3],
                "X": [3, 3],
                "stride_h": [1, 1],
                "stride_w": [1, 1],
                "pad_h": [1, 1],
                "pad_w": [1, 1],
                "block_size": [16, 16],
                "gemm_m_per_block": [64, 64],
                "gemm_n_per_block": [64, 64],
                "pipeline": ["compv3", "compv4"],
                "dtype": ["bf16", "bf16"],
            }
        )

        features = self.engine.extract_batch(df)

        # Should return (2, 97) array (post suffix-aware update)
        self.assertEqual(features.shape, (2, 97))
        self.assertFalse(np.any(np.isnan(features)), "Features should not contain NaN")

    def test_extract_batch_with_dilation(self):
        """Test batch extraction handles dilation properly."""
        df = pd.DataFrame(
            {
                "N": [1, 1],
                "C": [64, 64],
                "K": [64, 64],
                "G": [1, 1],
                "Hi": [32, 32],
                "Wi": [32, 32],
                "Y": [3, 3],
                "X": [3, 3],
                "stride_h": [1, 1],
                "stride_w": [1, 1],
                "pad_h": [1, 1],
                "pad_w": [1, 1],
                "dilation_h": [1, 2],  # Different dilations
                "dilation_w": [1, 2],
                "block_size": [16, 16],
                "gemm_m_per_block": [64, 64],
                "gemm_n_per_block": [64, 64],
                "pipeline": ["compv3", "compv3"],
                "dtype": ["bf16", "bf16"],
            }
        )

        features = self.engine.extract_batch(df)

        # Check Ho values (index 12)
        self.assertEqual(features[0, 12], 32, "First row Ho (no dilation) should be 32")
        self.assertEqual(features[1, 12], 30, "Second row Ho (dilation=2) should be 30")

    def test_extract_batch_without_dilation_column(self):
        """Test batch extraction defaults to dilation=1 when column absent."""
        df = pd.DataFrame(
            {
                "N": [1],
                "C": [64],
                "K": [128],
                "G": [1],
                "Hi": [32],
                "Wi": [32],
                "Y": [3],
                "X": [3],
                "stride_h": [1],
                "stride_w": [1],
                "pad_h": [1],
                "pad_w": [1],
                # No dilation_h, dilation_w columns
                "block_size": [16],
                "gemm_m_per_block": [64],
                "gemm_n_per_block": [64],
                "pipeline": ["compv3"],
                "dtype": ["bf16"],
            }
        )

        # Should not raise error, should default to dilation=1
        features = self.engine.extract_batch(df)
        self.assertEqual(features.shape, (1, 97))

        # Ho should be computed with dilation=1
        # Ho = (32 + 2*1 - 3) // 1 + 1 = 32
        self.assertEqual(features[0, 12], 32)

    def test_extract_batch_mixed_dtype(self):
        """Test batch extraction with mixed dtypes (vectorized bpe)."""
        df = pd.DataFrame(
            {
                "N": [1, 1, 1],
                "C": [64, 64, 64],
                "K": [128, 128, 128],
                "G": [1, 1, 1],
                "Hi": [32, 32, 32],
                "Wi": [32, 32, 32],
                "Y": [3, 3, 3],
                "X": [3, 3, 3],
                "stride_h": [1, 1, 1],
                "stride_w": [1, 1, 1],
                "pad_h": [1, 1, 1],
                "pad_w": [1, 1, 1],
                "dtype": ["bf16", "fp16", "fp32"],  # Mixed dtypes
                "block_size": [256, 256, 256],
                "gemm_m_per_block": [64, 64, 64],
                "gemm_n_per_block": [64, 64, 64],
                "pipeline": ["compv3", "compv3", "compv3"],
            }
        )

        features = self.engine.extract_batch(df)
        self.assertEqual(features.shape, (3, 97))

        # Verify arithmetic_intensity differs for different dtypes
        feature_names = self.engine.get_feature_names()
        ai_idx = feature_names.index("arithmetic_intensity")

        ai_bf16 = features[0, ai_idx]
        ai_fp16 = features[1, ai_idx]
        ai_fp32 = features[2, ai_idx]

        # bf16 and fp16 have same bpe=2, fp32 has bpe=4
        self.assertAlmostEqual(
            ai_bf16, ai_fp16, places=2, msg="bf16 and fp16 should have same AI"
        )
        self.assertAlmostEqual(
            ai_fp32,
            ai_bf16 / 2,
            places=2,
            msg="fp32 AI should be half of bf16 (2x bpe)",
        )

    def test_depthwise_convolution_features(self):
        """Test depthwise convolution feature flags."""
        # Depthwise: G == C == K
        problem_depthwise = {
            "N": 1,
            "C": 64,
            "K": 64,
            "G": 64,  # Depthwise
            "Hi": 32,
            "Wi": 32,
            "Y": 3,
            "X": 3,
            "stride_h": 1,
            "stride_w": 1,
            "pad_h": 1,
            "pad_w": 1,
        }

        kernel = {
            "block_size": 16,
            "gemm_m_per_block": 64,
            "gemm_n_per_block": 64,
            "pipeline": "compv3",
        }

        features = self.engine.extract(problem_depthwise, kernel)

        # Find is_depthwise feature (it's one of the Tier-1 group-specific features)
        # Based on get_feature_names(), is_depthwise should be around index 45-50
        # Let's just verify it exists and is 1.0
        feature_names = self.engine.get_feature_names()
        is_depthwise_idx = feature_names.index("is_depthwise")
        self.assertEqual(
            features[is_depthwise_idx],
            1.0,
            "is_depthwise should be 1.0 for depthwise conv",
        )

    def test_1x1_and_3x3_flags(self):
        """Test 1x1 and 3x3 convolution flags."""
        kernel = {
            "block_size": 16,
            "gemm_m_per_block": 64,
            "gemm_n_per_block": 64,
            "pipeline": "compv3",
        }

        # 1x1 convolution
        problem_1x1 = {
            "N": 1,
            "C": 64,
            "K": 128,
            "G": 1,
            "Hi": 32,
            "Wi": 32,
            "Y": 1,
            "X": 1,
            "stride_h": 1,
            "stride_w": 1,
            "pad_h": 0,
            "pad_w": 0,
        }

        # 3x3 convolution
        problem_3x3 = {
            **problem_1x1,
            "Y": 3,
            "X": 3,
            "pad_h": 1,
            "pad_w": 1,
        }

        features_1x1 = self.engine.extract(problem_1x1, kernel)
        features_3x3 = self.engine.extract(problem_3x3, kernel)

        feature_names = self.engine.get_feature_names()
        is_1x1_idx = feature_names.index("is_1x1_conv")
        is_3x3_idx = feature_names.index("is_3x3_conv")

        # 1x1 conv should have is_1x1_conv=1, is_3x3_conv=0
        self.assertEqual(features_1x1[is_1x1_idx], 1.0)
        self.assertEqual(features_1x1[is_3x3_idx], 0.0)

        # 3x3 conv should have is_1x1_conv=0, is_3x3_conv=1
        self.assertEqual(features_3x3[is_1x1_idx], 0.0)
        self.assertEqual(features_3x3[is_3x3_idx], 1.0)

    def test_pipeline_features(self):
        """Test pipeline categorical encoding."""
        problem = {
            "N": 1,
            "C": 64,
            "K": 128,
            "G": 1,
            "Hi": 32,
            "Wi": 32,
            "Y": 3,
            "X": 3,
            "stride_h": 1,
            "stride_w": 1,
            "pad_h": 1,
            "pad_w": 1,
        }

        kernel_v3 = {
            "block_size": 16,
            "gemm_m_per_block": 64,
            "gemm_n_per_block": 64,
            "pipeline": "compv3",
        }

        kernel_v5 = {
            **kernel_v3,
            "pipeline": "compv5",
        }

        features_v3 = self.engine.extract(problem, kernel_v3)
        features_v5 = self.engine.extract(problem, kernel_v5)

        feature_names = self.engine.get_feature_names()
        pipeline_idx = feature_names.index("pipeline")
        is_compv3_idx = feature_names.index("is_compv3")
        is_compv5_idx = feature_names.index("is_compv5")

        # CompV3 should have different pipeline encoding than CompV5
        self.assertNotEqual(features_v3[pipeline_idx], features_v5[pipeline_idx])

        # Boolean flags
        self.assertEqual(features_v3[is_compv3_idx], 1.0)
        self.assertEqual(features_v3[is_compv5_idx], 0.0)

        self.assertEqual(features_v5[is_compv3_idx], 0.0)
        self.assertEqual(features_v5[is_compv5_idx], 1.0)


class TestDilationFormula(unittest.TestCase):
    """Test dilation formula matches GroupedConvProblem.Ho/Wo."""

    def test_dilation_formula_2d(self):
        """Verify dilation formula: Ho = (Hi + 2*pad_h - eff_y) // stride_h + 1."""
        engine = GroupedConvFeatureEngine()

        test_cases = [
            # (Hi, Y, pad_h, stride_h, dilation_h, expected_Ho)
            (32, 3, 1, 1, 1, 32),  # Standard 3x3, no dilation
            (32, 3, 1, 1, 2, 30),  # 3x3 with dilation=2
            (56, 3, 1, 2, 1, 28),  # 3x3 with stride=2
            (56, 3, 1, 2, 2, 27),  # 3x3 with stride=2, dilation=2
            (32, 1, 0, 1, 1, 32),  # 1x1 conv
            (491, 1, 0, 1, 1, 491),  # Edge case: 1×491 spatial
        ]

        for Hi, Y, pad_h, stride_h, dilation_h, expected_Ho in test_cases:
            problem = {
                "N": 1,
                "C": 64,
                "K": 64,
                "G": 1,
                "Hi": Hi,
                "Wi": Hi,  # Same as Hi for simplicity
                "Y": Y,
                "X": Y,
                "stride_h": stride_h,
                "stride_w": stride_h,
                "pad_h": pad_h,
                "pad_w": pad_h,
                "dilation_h": dilation_h,
                "dilation_w": dilation_h,
            }

            kernel = {
                "block_size": 16,
                "gemm_m_per_block": 64,
                "gemm_n_per_block": 64,
                "pipeline": "compv3",
            }

            features = engine.extract(problem, kernel)
            feature_names = engine.get_feature_names()
            Ho_idx = feature_names.index("Ho")
            Ho_computed = features[Ho_idx]

            # Compute expected using formula: eff_y = (Y-1)*dilation_h + 1
            eff_y = (Y - 1) * dilation_h + 1
            Ho_expected = (Hi + 2 * pad_h - eff_y) // stride_h + 1

            self.assertEqual(
                Ho_computed,
                Ho_expected,
                f"Ho mismatch for Hi={Hi}, Y={Y}, pad={pad_h}, stride={stride_h}, "
                f"dilation={dilation_h}: got {Ho_computed}, expected {Ho_expected}",
            )


if __name__ == "__main__":
    unittest.main()
