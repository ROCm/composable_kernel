# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Live tests for smi_utils — compare rocm-smi vs amd-smi command output.

Requires rocm-smi and amd-smi on PATH (GPU host).

    python3 -m unittest tile_engine.ops.common.test_smi_utils -v
"""

from __future__ import annotations

import os
import shutil
import sys
import unittest
import unittest.mock
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from smi_utils import (  # noqa: E402
    check_gpu_available,
    count_gpus,
    detect_gpu_ids,
    fetch_live_normalized_fields,
    show_gpu_info,
    show_version,
    smi_equivalence_pairs,
)

_GPU_HOST = shutil.which("rocm-smi") is not None and shutil.which("amd-smi") is not None


@unittest.skipUnless(_GPU_HOST, "requires rocm-smi and amd-smi on PATH")
class TestSmiCommandEquivalence(unittest.TestCase):
    """Live rocm-smi vs amd-smi: normalized fields must agree."""

    @classmethod
    def setUpClass(cls):
        cls.fields = fetch_live_normalized_fields()

    def test_gpu_ids_match(self):
        self.assertEqual(self.fields["gpu_ids_rocm"], self.fields["gpu_ids_amd"])

    def test_product_match(self):
        self.assertEqual(self.fields["product_rocm"], self.fields["product_amd"])

    def test_gfx_match(self):
        self.assertEqual(self.fields["gfx_rocm"], self.fields["gfx_amd"])

    def test_driver_match(self):
        self.assertEqual(self.fields["driver_rocm"], self.fields["driver_amd"])

    def test_all_pairs_match(self):
        mismatches = [
            (name, rocm_val, amd_val)
            for name, rocm_val, amd_val in smi_equivalence_pairs(self.fields)
            if rocm_val != amd_val
        ]
        self.assertEqual(mismatches, [])


@unittest.skipUnless(_GPU_HOST, "requires rocm-smi and amd-smi on PATH")
class TestLiveWrappers(unittest.TestCase):
    """Wrappers must agree with live command output (no hardcoded device values)."""

    @classmethod
    def setUpClass(cls):
        cls.fields = fetch_live_normalized_fields()

    def test_detect_gpu_ids_matches_smi_output(self):
        ids = detect_gpu_ids()
        self.assertEqual(ids, self.fields["gpu_ids_rocm"])
        self.assertEqual(ids, self.fields["gpu_ids_amd"])

    def test_count_gpus_matches_smi_output(self):
        self.assertEqual(count_gpus(), len(self.fields["gpu_ids_rocm"]))

    def test_check_gpu_available(self):
        self.assertTrue(check_gpu_available())

    def test_show_gpu_info_non_empty(self):
        self.assertTrue(show_gpu_info(head=25).strip())

    def test_show_version_non_empty(self):
        self.assertTrue(show_version().strip())

    def test_rocm_smi_override_matches_live_rocm_ids(self):
        with unittest.mock.patch.dict(os.environ, {"CK_SMI_TOOL": "rocm-smi"}):
            self.assertEqual(detect_gpu_ids(), self.fields["gpu_ids_rocm"])


if __name__ == "__main__":
    unittest.main()
