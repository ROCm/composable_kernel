#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "codegen"))

from fmha.validation import validate_config, load_arch_specs

SPECS = load_arch_specs()


def _base_config(
    family="fwd",
    dtype="fp16",
    arch="gfx950",
    pipeline="qr_async",
    hdim_q=128,
    hdim_v=128,
    **sig_overrides,
):
    sig = {
        "family": family,
        "data_type": dtype,
        "mode": "batch",
        "vlayout": "r",
        "hdim_q": hdim_q,
        "hdim_v": hdim_v,
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
    }
    sig.update(sig_overrides)
    alg = {
        "pipeline": pipeline,
        "tile": [128, 128, 32, 128, 32, 128],
        "wave": [4, 1, 1, 4, 1, 1, 1, 1, 1],
        "warp": [32, 32, 16, 32, 32, 16, 16, 16, 16],
        "padding": [True, True, True, True],
        "block_per_cu": 1,
        "num_wave_groups": 1,
        "max_splits_log2": 0,
        "max_seq_len_q": 0,
    }
    return {"signature": sig, "algorithm": alg, "arch": arch}


class TestValidateConfig(unittest.TestCase):
    def test_valid_basic_config(self):
        r = validate_config(_base_config(), SPECS)
        self.assertTrue(r.valid, r.errors)

    def test_unsupported_arch(self):
        r = validate_config(_base_config(arch="gfx000"), SPECS)
        self.assertFalse(r.valid)
        self.assertTrue(any("architecture" in e for e in r.errors))

    def test_v3_hdim128_valid(self):
        r = validate_config(_base_config(pipeline="v3", hdim_q=128, hdim_v=128), SPECS)
        self.assertTrue(r.valid, r.errors)

    def test_hdim_not_multiple_of_8(self):
        r = validate_config(_base_config(hdim_q=65, hdim_v=128), SPECS)
        self.assertFalse(r.valid)
        self.assertTrue(any("multiples of 8" in e for e in r.errors))

    def test_bias_plus_logits_soft_cap(self):
        r = validate_config(_base_config(bias="bias", logits=True), SPECS)
        self.assertFalse(r.valid)
        self.assertTrue(any("logits_soft_cap" in e for e in r.errors))

    def test_hdim_192_128_with_bias(self):
        r = validate_config(_base_config(hdim_q=192, hdim_v=128, bias="bias"), SPECS)
        has_issue = any("(192,128)" in e for e in r.errors) or any(
            "(192,128)" in w for w in r.warnings
        )
        self.assertTrue(has_issue)

    def test_hdim_192_128_with_dropout(self):
        r = validate_config(_base_config(hdim_q=192, hdim_v=128, dropout=True), SPECS)
        has_issue = any("(192,128)" in e for e in r.errors) or any(
            "(192,128)" in w for w in r.warnings
        )
        self.assertTrue(has_issue)

    def test_appendkv_must_use_appendkv_pipeline(self):
        cfg = _base_config(family="fwd_appendkv", pipeline="qr_async")
        r = validate_config(cfg, SPECS)
        self.assertFalse(r.valid)
        self.assertTrue(any("appendkv pipeline" in e for e in r.errors))

    def test_pagedkv_requires_qr_pagedkv_pipeline(self):
        cfg = _base_config(family="fwd_pagedkv", pipeline="qr_async", paged_kv=True)
        r = validate_config(cfg, SPECS)
        self.assertFalse(r.valid)
        self.assertTrue(any("qr_pagedkv" in e for e in r.errors))

    def test_batch_prefill_requires_group_mode(self):
        cfg = _base_config(
            family="batch_prefill",
            pipeline="qr_async",
            mode="batch",
            paged_kv=True,
            page_size=64,
        )
        cfg["signature"]["mode"] = "batch"
        r = validate_config(cfg, SPECS)
        self.assertFalse(r.valid)
        self.assertTrue(any("group mode" in e for e in r.errors))

    def test_batch_prefill_valid_group(self):
        cfg = _base_config(
            family="batch_prefill", pipeline="qr_async", paged_kv=True, page_size=64
        )
        cfg["signature"]["mode"] = "group"
        r = validate_config(cfg, SPECS)
        self.assertTrue(r.valid, r.errors)

    def test_splitkv_combine_bn1_must_be_32(self):
        cfg = _base_config(family="fwd_splitkv_combine", pipeline="qr")
        cfg["algorithm"]["tile"][3] = 64
        r = validate_config(cfg, SPECS)
        self.assertFalse(r.valid)
        self.assertTrue(any("bn1" in e for e in r.errors))

    def test_bwd_dot_do_o_bm0_128_accepted(self):
        cfg = _base_config(family="bwd_dot_do_o", pipeline="qr")
        cfg["algorithm"]["tile"][0] = 128
        r = validate_config(cfg, SPECS)
        # bwd_dot_do_o with bm0=128 is now valid (relaxed from strict bm0=64)
        self.assertTrue(r.valid, r.errors)

    def test_mask_types_all_valid(self):
        for mask in ["no", "top_left", "bottom_right", "generic"]:
            r = validate_config(_base_config(mask=mask), SPECS)
            self.assertTrue(r.valid, f"mask={mask}: {r.errors}")


class TestMaskDistinction(unittest.TestCase):
    """Verify that top_left and bottom_right are distinct after fix."""

    def test_mask_canonical_distinguishes(self):
        from fmha.symbol_map import canonical_mask, MASK_TO_INT

        self.assertEqual(canonical_mask("top_left"), "top_left")
        self.assertEqual(canonical_mask("bottom_right"), "bottom_right")
        self.assertNotEqual(MASK_TO_INT["top_left"], MASK_TO_INT["bottom_right"])


if __name__ == "__main__":
    unittest.main()
