#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only unit tests for the stream-K surface of python/gemm_utils.py.

The stream-K bridge adds a ``reduction_strategy`` to GemmKernelConfig and a
dedicated ctypes source. These tests lock in the two pieces of pure host-side
logic that must stay byte-exact with the codegen and the build:

  * ``GemmKernelConfig.name`` -- the suffix rules mirror
    unified_gemm_codegen.py::KernelNaming.generate. Atomic keeps the bare
    ``_streamk`` (original parity name); linear/tree are disambiguated as
    ``_streamk_<strategy>``. If this drifts, the runtime registry lookup key
    misses the kernel baked into the generated header.
  * ``_ctypes_source_name`` -- stream-K launches StreamKHostArgs directly
    (registry-bypass) so it needs its own bridge .cpp; every other variant
    shares gemm_ctypes_lib.cpp.

No GPU is touched. Run: python3 -m pytest tests/test_streamk_gemm_utils.py -v
"""

import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "python"))

from gemm_utils import (  # noqa: E402
    GemmKernelConfig,
    _ctypes_source_name,
    _dtype_from_kernel_name,
    _layout_from_kernel_name,
)


class TestStreamKNaming(unittest.TestCase):
    """Stream-K variant naming and reduction-strategy plumbing."""

    def _cfg(self, variant="stream_k", reduction_strategy="atomic"):
        return GemmKernelConfig(variant=variant, reduction_strategy=reduction_strategy)

    def test_atomic_keeps_bare_streamk_suffix(self):
        name = self._cfg(reduction_strategy="atomic").name
        self.assertTrue(name.endswith("_streamk"))
        self.assertNotIn("_streamk_atomic", name)

    def test_linear_and_tree_are_disambiguated(self):
        self.assertTrue(
            self._cfg(reduction_strategy="linear").name.endswith("_streamk_linear")
        )
        self.assertTrue(
            self._cfg(reduction_strategy="tree").name.endswith("_streamk_tree")
        )

    def test_standard_has_no_streamk_suffix(self):
        self.assertNotIn("streamk", self._cfg(variant="standard").name)

    def test_streamk_name_still_roundtrips_dtype_and_layout(self):
        # The variant suffix must not disturb the dtype/layout tokens the runner
        # parses back out of the compiled .so name.
        for red in ("atomic", "linear", "tree"):
            cfg = GemmKernelConfig(
                dtype_a="bf16",
                dtype_b="bf16",
                dtype_c="bf16",
                layout_a="col",
                layout_b="col",
                layout_c="row",
                variant="stream_k",
                reduction_strategy=red,
            )
            self.assertEqual(_dtype_from_kernel_name(cfg.name), "bf16")
            self.assertEqual(_layout_from_kernel_name(cfg.name), "ccr")

    def test_codegen_json_pins_reduction_only_for_streamk(self):
        sk = self._cfg(reduction_strategy="tree").to_codegen_json()
        self.assertEqual(sk["streamk_config"], {"reduction_strategy": ["tree"]})
        # Non-stream-K configs must not emit a streamk_config block.
        self.assertNotIn(
            "streamk_config",
            GemmKernelConfig(variant="standard").to_codegen_json(),
        )


class TestCtypesSourceRouting(unittest.TestCase):
    """Each variant routes to the ctypes bridge .cpp matching its launch ABI."""

    def test_streamk_gets_dedicated_source(self):
        cfg = GemmKernelConfig(variant="stream_k")
        self.assertEqual(_ctypes_source_name(cfg), "streamk_gemm_ctypes_lib.cpp")

    def test_standard_uses_default_source(self):
        self.assertEqual(
            _ctypes_source_name(GemmKernelConfig(variant="standard")),
            "gemm_ctypes_lib.cpp",
        )
        self.assertEqual(
            _ctypes_source_name(GemmKernelConfig(variant="preshuffle")),
            "gemm_ctypes_lib.cpp",
        )


if __name__ == "__main__":
    unittest.main()
