# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only negative tests: the grouped quant dispatcher codegens (aquant,
abquant, bquant, rowcolquant, tensorquant) raise for async/TDM pipelines and
the tdm epilogue instead of warning and skipping them.
"""

import dataclasses
import importlib
import os
import sys
import unittest

_CK_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
)
sys.path.insert(0, os.path.join(_CK_ROOT, "dispatcher", "codegen"))

_OPS = ("aquant", "abquant", "bquant", "rowcolquant", "tensorquant")
_BAD_PIPELINES = ("comp_async", "comp_tdm", "comp_tdm_v2")


def _mod(op):
    return importlib.import_module(f"unified_grouped_gemm_{op}_codegen")


class TestGroupedQuantCodegenRejects(unittest.TestCase):
    def test_helper(self):
        for op in _OPS:
            m = _mod(op)
            for p in _BAD_PIPELINES:
                with self.assertRaisesRegex(ValueError, f"'{p}' pipeline", msg=op):
                    m.reject_async_tdm_traits(p, "cshuffle")
            with self.assertRaisesRegex(ValueError, "'tdm' epilogue", msg=op):
                m.reject_async_tdm_traits("compv3", "tdm")
            for p in ("compv3", "mem"):
                m.reject_async_tdm_traits(p, "cshuffle")

    def test_build_specs_rejects(self):
        for op in _OPS:
            m = _mod(op)
            for p in _BAD_PIPELINES:
                cfg = dict(m._default_config(), pipeline=p)
                with self.assertRaisesRegex(ValueError, p, msg=(op, p)):
                    m._build_specs(cfg)
            cfg = dict(m._default_config(), epilogue="tdm")
            with self.assertRaisesRegex(ValueError, "tdm", msg=op):
                m._build_specs(cfg)

    def test_spec_ctor_rejects(self):
        for op in _OPS:
            m = _mod(op)
            specs = m._build_specs(m._default_config())
            self.assertTrue(specs, msg=op)
            for p in _BAD_PIPELINES:
                with self.assertRaises(ValueError, msg=(op, p)):
                    dataclasses.replace(specs[0], pipeline=p)
            with self.assertRaises(ValueError, msg=op):
                dataclasses.replace(specs[0], epilogue="tdm")

    def test_legacy_default_unchanged(self):
        for op in _OPS:
            m = _mod(op)
            cfg = m._default_config()
            specs = m._build_specs(cfg)
            self.assertTrue(specs, msg=op)
            for s in specs:
                self.assertEqual(s.pipeline, cfg.get("pipeline", "compv3"), msg=op)
                self.assertEqual(s.epilogue, cfg.get("epilogue", "cshuffle"), msg=op)


if __name__ == "__main__":
    unittest.main()
