#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from fmha_utils import _validate_batch_prefill_input_dtypes  # noqa: E402


class TestBatchPrefillDtypeValidation(unittest.TestCase):
    def test_mixed_bf16_q_fp8_kv_gqa_decode_reports_unsupported(self):
        batch = 4
        q_len = 1
        ctx_len = 1024
        num_q_heads = 96
        num_kv_heads = 8
        head_dim = 128

        q = np.zeros((batch, num_q_heads, q_len, head_dim), dtype=np.uint16)
        k = np.zeros((batch, num_kv_heads, ctx_len, head_dim), dtype=np.uint8)
        v = np.zeros((batch, num_kv_heads, ctx_len, head_dim), dtype=np.uint8)

        with self.assertRaisesRegex(
            ValueError,
            "does not yet support mixed Q/KV dtypes",
        ):
            _validate_batch_prefill_input_dtypes("batch_prefill", "bf16", q, k, v)

    def test_all_fp8_bf16_output_path_remains_allowed(self):
        q = np.zeros((1, 96, 1, 128), dtype=np.uint8)
        k = np.zeros((1, 8, 128, 128), dtype=np.uint8)
        v = np.zeros((1, 8, 128, 128), dtype=np.uint8)

        _validate_batch_prefill_input_dtypes("batch_prefill", "fp8bf16", q, k, v)

    def test_non_batch_prefill_paths_are_unchanged(self):
        q = np.zeros((1, 96, 1, 128), dtype=np.uint16)
        k = np.zeros((1, 8, 128, 128), dtype=np.uint8)
        v = np.zeros((1, 8, 128, 128), dtype=np.uint8)

        _validate_batch_prefill_input_dtypes("fwd", "bf16", q, k, v)


if __name__ == "__main__":
    unittest.main()
