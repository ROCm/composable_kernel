#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import unittest
from unittest.mock import mock_open, patch
from generate_test_files import suffix_to_file_tag, parse_types_header, output_path

# ------------------------------------------------------------ #
# Unit tests for helper functions in generate_test_files.py
# ------------------------------------------------------------ #


class TestSuffixToFileTag(unittest.TestCase):
    def test_fp16_token(self):
        suffix = "Fp16"
        expected_tag = "fp16"
        self.assertEqual(suffix_to_file_tag(suffix), expected_tag)

    def test_bf16_token(self):
        suffix = "Bf16"
        expected_tag = "bf16"
        self.assertEqual(suffix_to_file_tag(suffix), expected_tag)

    def test_fp8_token(self):
        suffix = "Fp8"
        expected_tag = "fp8"
        self.assertEqual(suffix_to_file_tag(suffix), expected_tag)

    def test_bf8_token(self):
        suffix = "Bf8"
        expected_tag = "bf8"
        self.assertEqual(suffix_to_file_tag(suffix), expected_tag)

    def test_nonpersistent_token(self):
        suffix = "NonPersistent"
        expected_tag = "nonpersistent"
        self.assertEqual(suffix_to_file_tag(suffix), expected_tag)

    def test_persistent_token(self):
        suffix = "Persistent"
        expected_tag = "persistent"
        self.assertEqual(suffix_to_file_tag(suffix), expected_tag)

    def test_atomic_token(self):
        suffix = "Atomic"
        expected_tag = "atomic"
        self.assertEqual(suffix_to_file_tag(suffix), expected_tag)

    def test_linear_token(self):
        suffix = "Linear"
        expected_tag = "linear"
        self.assertEqual(suffix_to_file_tag(suffix), expected_tag)

    def test_tree_token(self):
        suffix = "Tree"
        expected_tag = "tree"
        self.assertEqual(suffix_to_file_tag(suffix), expected_tag)

    def test_compv3_token(self):
        suffix = "CompV3"
        expected_tag = "compv3"
        self.assertEqual(suffix_to_file_tag(suffix), expected_tag)

    def test_pipelines_token(self):
        suffix = "Pipelines"
        expected_tag = "pipelines"
        self.assertEqual(suffix_to_file_tag(suffix), expected_tag)

    def test_regression_token(self):
        suffix = "Regression"
        expected_tag = "regression"
        self.assertEqual(suffix_to_file_tag(suffix), expected_tag)

    def test_unknown_token(self):
        suffix = "unknown"
        with self.assertRaises(ValueError):
            suffix_to_file_tag(suffix)

    def test_multiple_valid_tokens(self):
        suffix = "Fp16PersistentAtomicCompV3"
        expected_tag = "fp16_persistent_atomic_compv3"
        self.assertEqual(suffix_to_file_tag(suffix), expected_tag)

    def test_multiple_tokens_with_unknown(self):
        suffix = "Fp16PersistentUnknownCompV3"
        with self.assertRaises(ValueError):
            suffix_to_file_tag(suffix)


class TestParseTypesHeader(unittest.TestCase):
    def validate_entries(self, entries, expected_entries):
        self.assertEqual(len(entries), len(expected_entries))
        for idx in range(len(entries)):
            self.assertDictEqual(entries[idx], expected_entries[idx])

    def test_empty_entry(self):
        """Test that an empty file returns no entries."""
        mock_content = ""
        with patch("builtins.open", mock_open(read_data=mock_content)):
            entries = parse_types_header("fake_path.hpp", "atomic_smoke")
            self.assertEqual(len(entries), 0)

    def test_pipelines_smoke(self):
        """Test pipelines_smoke target: matches suffix == 'Pipelines'.
        Includes: Pipelines
        Excludes: Fp8NonPersistentTreeCompV3
        """
        mock_content = (
            "using KernelTypesStreamKPipelines = ...\n"
            "using KernelTypesStreamKFp8NonPersistentTreeCompV3 = ...\n"
        )
        with patch("builtins.open", mock_open(read_data=mock_content)):
            entries = parse_types_header("fake_path.hpp", "pipelines_smoke")
            expected = [
                {
                    "type_alias": "KernelTypesStreamKPipelines",
                    "class_name": "TestCkTileStreamKPipelines",
                    "file_tag": "pipelines",
                }
            ]
            self.validate_entries(entries, expected)

    def test_extended(self):
        """Test extended target: matches 'Atomic' in suffix OR suffix == 'Pipelines'.
        Includes: Fp16PersistentAtomic, Pipelines
        Excludes: Bf16Linear
        """
        mock_content = (
            "using KernelTypesStreamKFp16PersistentAtomic = ...\n"
            "using KernelTypesStreamKPipelines = ...\n"
            "using KernelTypesStreamKBf16Linear = ...\n"
        )
        with patch("builtins.open", mock_open(read_data=mock_content)):
            entries = parse_types_header("fake_path.hpp", "extended")
            expected = [
                {
                    "type_alias": "KernelTypesStreamKFp16PersistentAtomic",
                    "class_name": "TestCkTileStreamKFp16PersistentAtomic",
                    "file_tag": "fp16_persistent_atomic",
                },
                {
                    "type_alias": "KernelTypesStreamKPipelines",
                    "class_name": "TestCkTileStreamKPipelines",
                    "file_tag": "pipelines",
                },
            ]
            self.validate_entries(entries, expected)

    def test_atomic_smoke(self):
        """Test atomic_smoke target: matches 'Atomic' in suffix AND suffix != 'Pipelines'.
        Includes: Fp16PersistentAtomic
        Excludes: Bf16Linear, Pipelines
        """
        mock_content = (
            "using KernelTypesStreamKFp16PersistentAtomic = ...\n"
            "using KernelTypesStreamKBf16Linear = ...\n"
            "using KernelTypesStreamKPipelines = ...\n"
        )
        with patch("builtins.open", mock_open(read_data=mock_content)):
            entries = parse_types_header("fake_path.hpp", "atomic_smoke")
            expected = [
                {
                    "type_alias": "KernelTypesStreamKFp16PersistentAtomic",
                    "class_name": "TestCkTileStreamKFp16PersistentAtomic",
                    "file_tag": "fp16_persistent_atomic",
                }
            ]
            self.validate_entries(entries, expected)

    def test_linear_smoke(self):
        """Test linear_smoke target: matches 'Linear' in suffix AND suffix != 'Pipelines'.
        Includes: Fp8NonPersistentLinear
        Excludes: Bf16PersistentAtomic, Pipelines
        """
        mock_content = (
            "using KernelTypesStreamKFp8NonPersistentLinear = ...\n"
            "using KernelTypesStreamKBf16PersistentAtomic = ...\n"
            "using KernelTypesStreamKPipelines = ...\n"
        )
        with patch("builtins.open", mock_open(read_data=mock_content)):
            entries = parse_types_header("fake_path.hpp", "linear_smoke")
            expected = [
                {
                    "type_alias": "KernelTypesStreamKFp8NonPersistentLinear",
                    "class_name": "TestCkTileStreamKFp8NonPersistentLinear",
                    "file_tag": "fp8_nonpersistent_linear",
                }
            ]
            self.validate_entries(entries, expected)

    def test_tree_smoke(self):
        """Test tree_smoke target: matches 'Tree' in suffix AND suffix != 'Pipelines'.
        Includes: Bf8PersistentTreeCompV3
        Excludes: Fp16Linear, Pipelines
        """
        mock_content = (
            "using KernelTypesStreamKBf8PersistentTreeCompV3 = ...\n"
            "using KernelTypesStreamKFp16Linear = ...\n"
            "using KernelTypesStreamKPipelines = ...\n"
        )
        with patch("builtins.open", mock_open(read_data=mock_content)):
            entries = parse_types_header("fake_path.hpp", "tree_smoke")
            expected = [
                {
                    "type_alias": "KernelTypesStreamKBf8PersistentTreeCompV3",
                    "class_name": "TestCkTileStreamKBf8PersistentTreeCompV3",
                    "file_tag": "bf8_persistent_tree_compv3",
                }
            ]
            self.validate_entries(entries, expected)

    def test_regression(self):
        """Test regression target: matches suffix == 'Regression'.
        Includes: Regression
        """
        mock_content = (
            "using KernelTypesStreamKRegression = ...\n"
            "using KernelTypesStreamKFp16Linear = ...\n"
            "using KernelTypesStreamKPipelines = ...\n"
        )
        with patch("builtins.open", mock_open(read_data=mock_content)):
            entries = parse_types_header("fake_path.hpp", "regression")
            expected = [
                {
                    "type_alias": "KernelTypesStreamKRegression",
                    "class_name": "TestCkTileStreamKRegression",
                    "file_tag": "regression",
                }
            ]
            self.validate_entries(entries, expected)


class TestOutputPath(unittest.TestCase):
    def test_output_path(self):
        """Test that output_path generates the correct file path."""
        entry = {"file_tag": "fp16_persistent_atomic"}
        output_dir = "/some/output/dir"
        expected = "/some/output/dir/test_gemm_streamk_fp16_persistent_atomic.cpp"
        self.assertEqual(output_path(output_dir, entry), expected)


if __name__ == "__main__":
    unittest.main()
