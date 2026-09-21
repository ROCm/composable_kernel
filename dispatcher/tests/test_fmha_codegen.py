#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codegen"))

from fmha.validation import profile_allows  # noqa: E402
from fmha.validation import validate_config  # noqa: E402

CODEGEN = ROOT / "codegen" / "fmha" / "codegen.py"


def sample_config(**overrides):
    config = {
        "arch": "gfx942",
        "signature": {
            "family": "fwd",
            "data_type": "fp16",
            "mode": "batch",
            "vlayout": "r",
            "hdim_q": 128,
            "hdim_v": 128,
            "mask": "no",
            "bias": "no",
            "lse": False,
            "dropout": False,
            "qscale": "no",
            "rope": "none",
            "logits": False,
            "paged_kv": False,
            "fp8_static_quant": False,
            "skip_min_seqlen_q": False,
            "sink": False,
            "dbias": False,
            "store_randval": False,
            "deterministic": False,
            "kv_memory_layout": "vectorized",
            "kv_lookup_table": "sglang",
            "page_size": 1,
        },
        "algorithm": {
            "pipeline": "qr_async",
            "tile": [128, 128, 32, 128, 32, 128],
            "wave": [2, 2, 1, 2, 2, 1, 1, 1, 1],
            "warp": [32, 32, 16, 32, 32, 16, 16, 16, 16],
            "padding": [True, True, True, True],
            "use_trload": False,
            "hdim_q_alignment": 128,
            "hdim_v_alignment": 128,
            "block_per_cu": 1,
            "num_wave_groups": 1,
            "max_splits_log2": 0,
            "max_seq_len_q": 0,
            "selection_rank": 0,
            "constraint_tag": "",
        },
    }

    for section, values in overrides.items():
        if isinstance(values, dict):
            config[section].update(values)
        else:
            config[section] = values
    return config


class TestFmhaCodegen(unittest.TestCase):
    def test_forward_codegen_emits_kernel_and_wrapper(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                sys.executable,
                str(CODEGEN),
                "--output-dir",
                tmpdir,
                "--gpu-target",
                "gfx942",
                "--config-json",
                json.dumps(sample_config()),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=str(ROOT / "codegen")
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

            generated = list(Path(tmpdir).glob("fmha_*.hpp"))
            wrappers = list(
                (Path(tmpdir) / "dispatcher_wrappers").glob(
                    "dispatcher_wrapper_fmha_*.hpp"
                )
            )
            self.assertEqual(len(generated), 1)
            self.assertEqual(len(wrappers), 1)

    def test_profile_filter_rejects_pytorch_unsupported_config(self):
        config = sample_config(signature={"bias": "alibi"})
        self.assertFalse(profile_allows(config, profile="pytorch"))
        self.assertTrue(profile_allows(config, profile="flash_fwd"))

    def test_batch_prefill_validation_requires_row_major(self):
        config = sample_config(
            signature={
                "family": "batch_prefill",
                "mode": "group",
                "paged_kv": True,
                "vlayout": "c",
                "page_size": 16,
            },
            algorithm={"pipeline": "qr_async"},
        )
        result = validate_config(config)
        self.assertFalse(result.valid)
        self.assertTrue(any("row-major" in error for error in result.errors))

    def test_qr_async_hdim_128_requires_bn0_128(self):
        config = sample_config(
            algorithm={
                "pipeline": "qr_async",
                "tile": [128, 64, 32, 128, 16, 128],
            }
        )
        result = validate_config(config)
        # Constraint-based tile rules allow various bn0 values for h128
        self.assertTrue(result.valid)

    def test_splitkv_combine_requires_bn1_32(self):
        config = sample_config(
            signature={"family": "fwd_splitkv_combine", "lse": True},
            algorithm={
                "pipeline": "qr",
                "tile": [64, 128, 32, 128, 32, 128],
                "max_splits_log2": 6,
            },
        )
        result = validate_config(config)
        self.assertFalse(result.valid)
        self.assertTrue(any("bn1" in error for error in result.errors))

    def test_batch_prefill_requires_group_mode(self):
        config = sample_config(
            signature={
                "family": "batch_prefill",
                "mode": "batch",
                "paged_kv": True,
                "page_size": 16,
            },
            algorithm={"pipeline": "qr_async"},
        )
        result = validate_config(config)
        self.assertFalse(result.valid)
        self.assertTrue(any("group mode" in error for error in result.errors))

    def test_receipt_aliases_match_profiles(self):
        flash = sample_config(signature={"bias": "alibi"})
        pytorch = sample_config(signature={"bias": "bias"})
        aiter = sample_config()

        self.assertTrue(profile_allows(flash, receipt=2))
        self.assertTrue(profile_allows(pytorch, receipt=4))
        self.assertTrue(profile_allows(aiter, receipt=100))


if __name__ == "__main__":
    unittest.main()
