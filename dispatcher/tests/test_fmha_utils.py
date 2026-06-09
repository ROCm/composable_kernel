#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from fmha_dtype_contract import FmhaDTypeContractKind  # noqa: E402
from fmha_utils import (  # noqa: E402
    _validate_batch_prefill_input_dtypes,
    get_batch_prefill_dtype_contract,
)


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
            "mixed activation/FP8-KV dtype contract",
        ):
            _validate_batch_prefill_input_dtypes("batch_prefill", "bf16", q, k, v)

    def test_mixed_fp16_q_fp8_kv_reports_unsupported(self):
        q = np.zeros((4, 96, 1, 128), dtype=np.float16)
        k = np.zeros((4, 8, 1024, 128), dtype=np.uint8)
        v = np.zeros((4, 8, 1024, 128), dtype=np.uint8)

        contract = get_batch_prefill_dtype_contract("fp16", q, k, v)

        self.assertEqual(contract.kind, FmhaDTypeContractKind.MIXED_Q_FP8_KV)
        with self.assertRaisesRegex(ValueError, "AITER paged_attention_ragged"):
            _validate_batch_prefill_input_dtypes("batch_prefill", "fp16", q, k, v)

    def test_all_fp8_bf16_output_path_remains_allowed(self):
        q = np.zeros((1, 96, 1, 128), dtype=np.uint8)
        k = np.zeros((1, 8, 128, 128), dtype=np.uint8)
        v = np.zeros((1, 8, 128, 128), dtype=np.uint8)

        contract = get_batch_prefill_dtype_contract("fp8bf16", q, k, v)

        self.assertEqual(contract.kind, FmhaDTypeContractKind.ALL_FP8_WITH_BF16_OUTPUT)
        _validate_batch_prefill_input_dtypes("batch_prefill", "fp8bf16", q, k, v)

    def test_all_bf16_batch_prefill_path_remains_allowed(self):
        q = np.zeros((1, 96, 1, 128), dtype=np.uint16)
        k = np.zeros((1, 8, 128, 128), dtype=np.uint16)
        v = np.zeros((1, 8, 128, 128), dtype=np.uint16)

        contract = get_batch_prefill_dtype_contract("bf16", q, k, v)

        self.assertEqual(contract.kind, FmhaDTypeContractKind.HOMOGENEOUS)
        _validate_batch_prefill_input_dtypes("batch_prefill", "bf16", q, k, v)

    def test_non_batch_prefill_paths_are_unchanged(self):
        q = np.zeros((1, 96, 1, 128), dtype=np.uint16)
        k = np.zeros((1, 8, 128, 128), dtype=np.uint8)
        v = np.zeros((1, 8, 128, 128), dtype=np.uint8)

        _validate_batch_prefill_input_dtypes("fwd", "bf16", q, k, v)


if __name__ == "__main__":
    unittest.main()
