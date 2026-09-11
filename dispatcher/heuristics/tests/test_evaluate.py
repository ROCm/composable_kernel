#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Tests for evaluate.py.

Covers: shape family classification, K-depth regime classification,
and basic evaluation metric checks.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import classify_shape_family, classify_k_regime


class TestClassifyShapeFamily:
    def test_tiny_m(self):
        assert classify_shape_family(1, 4096, 4096) == "tiny_m"
        assert classify_shape_family(16, 1536, 7168) == "tiny_m"

    def test_small_m(self):
        assert classify_shape_family(32, 1536, 7168) == "small_m"
        assert classify_shape_family(128, 4096, 4096) == "small_m"

    def test_medium_m(self):
        assert classify_shape_family(256, 1024, 1024) == "medium_m"
        assert classify_shape_family(2048, 2048, 2048) == "medium_m"

    def test_large_m(self):
        assert classify_shape_family(4096, 4096, 4096) == "large_m"
        assert classify_shape_family(20480, 7168, 256) == "large_m"


class TestClassifyKRegime:
    def test_shallow(self):
        assert classify_k_regime(256) == "shallow_k"
        assert classify_k_regime(32) == "shallow_k"

    def test_medium(self):
        assert classify_k_regime(1024) == "medium_k"
        assert classify_k_regime(2048) == "medium_k"

    def test_deep(self):
        assert classify_k_regime(4096) == "deep_k"
        assert classify_k_regime(7168) == "deep_k"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
