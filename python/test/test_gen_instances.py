# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import logging
import unittest

from ck4inductor.universal_gemm.gen_instances import (
    gen_ops_library as gen_gemm_ops_library,
)
from ck4inductor.universal_gemm.gen_instances import (
    gen_ops_library_wmma as gen_gemm_ops_library_wmma,
)
from ck4inductor.universal_gemm.gen_instances import (
    gen_ops_preselected as gen_gemm_ops_preselected,
)
from ck4inductor.grouped_conv_fwd.gen_instances import (
    gen_conv_ops_library as gen_conv_ops_library,
)
from ck4inductor.grouped_conv_fwd.gen_instances import (
    gen_conv_ops_library_wmma as gen_conv_ops_library_wmma,
)
from ck4inductor.batched_universal_gemm.gen_instances import (
    gen_ops_library as gen_batched_gemm_ops_library,
)
from ck4inductor.batched_universal_gemm.gen_instances import (
    gen_ops_library_wmma as gen_batched_gemm_ops_library_wmma,
)
from ck4inductor.ck_tile_universal_gemm.gen_instances import (
    ops as gen_ck_tile_gemm_ops_library,
)
from ck4inductor import check_headers, include_roots

log = logging.getLogger(__name__)


class TestGenInstances(unittest.TestCase):
    def test_gen_gemm_instances(self):
        instances = gen_gemm_ops_library()

        log.debug("%d gemm instances from library" % len(instances))
        self.assertTrue(instances)

    def test_preselected_gemm_instances(self):
        instances = gen_gemm_ops_preselected()

        log.debug("%d preselected gemm instances" % len(instances))
        self.assertTrue(instances)

    def test_gen_conv_instances(self):
        instances = gen_conv_ops_library()

        log.debug("%d gemm instances from library" % len(instances))
        self.assertTrue(instances)

    def test_gen_conv_wmma_instances(self):
        # gfx1250 WMMA grouped-conv enumerator. Purpose-built 16x16-warp kernels,
        # f16/bf16 only -- CK ships no f32 WMMA conv instance.
        instances = gen_conv_ops_library_wmma()

        log.debug("%d wmma conv instances from library" % len(instances))
        self.assertTrue(instances)
        # Parse completeness. parse_instances skips any line it cannot turn into
        # an op, so a CK template-param reorder would silently drop most of them
        # while every per-op assertion below still passed on the survivors.
        #
        # Each parsed line expands by 8: the WMMA instances hardcode their
        # scheduler (unlike the XDL ones, which leave a BlkGemmPipeSched
        # placeholder and expand by 16), so only the 4 conv specs x 2 layouts
        # multiply out.
        self.assertEqual(
            len(instances) % 8,
            0,
            "WMMA conv instance count is not a multiple of the substitution "
            "factor; parse_instances failed to parse some lines.",
        )
        self.assertGreaterEqual(
            len(instances) // 8,
            70,
            "Far fewer WMMA conv instances parsed than the 76 CK ships; "
            "parse_instances is skipping lines without reporting it "
            "(did the CK template parameters change?).",
        )
        for op in instances:
            self.assertTrue(op.is_wmma)
            self.assertEqual((op.m_per_xdl, op.n_per_xdl), (16, 16))
            self.assertIn(op.a_element_dtype, ("F16", "BF16"))
            self.assertIn(op.b_element_dtype, ("F16", "BF16"))
            # PyTorch's conv lowering passes no D tensor; an instance carrying one
            # would render a template argument the kernel wrapper never supplies.
            self.assertEqual(op.ds_layout, ())
            self.assertEqual(op.ds_element_dtype, ())
            # `is_wmma` is Python-side metadata. If it reaches dict_items it is
            # emitted as a C++ template argument and every instance fails to build.
            self.assertNotIn("is_wmma", dict(op.dict_items()))

    def test_conv_xdl_instances_are_not_wmma(self):
        # The two enumerators must never share an op: a WMMA-tagged op reaching the
        # XDL pool would be rendered through the wrong device op, and the aliases
        # would collide.
        xdl = gen_conv_ops_library()
        self.assertTrue(xdl)
        self.assertFalse(any(op.is_wmma for op in xdl))
        self.assertTrue(all("_xdl_" in op.name() for op in xdl))
        self.assertTrue(
            all("_wmma_" in op.name() for op in gen_conv_ops_library_wmma())
        )

    def test_gen_batched_gemm_instances(self):
        instances = gen_batched_gemm_ops_library()

        log.debug("%d gemm instances from library" % len(instances))
        self.assertTrue(instances)

    def test_gen_ck_tile_universal_gemm_instances(self):
        instances = gen_ck_tile_gemm_ops_library()

        log.debug("%d ck-tile gemm instances from library" % len(instances))
        self.assertTrue(instances)

    def test_gen_gemm_wmma_instances(self):
        # gfx1250 fat-tile WMMA enumerator. All shipped WMMA universal-gemm
        # instances are 16x16 warp, fp16/bf16.
        instances = gen_gemm_ops_library_wmma()

        log.debug("%d wmma gemm instances from library" % len(instances))
        self.assertTrue(instances)
        for op in instances:
            self.assertTrue(op.is_wmma)
            self.assertEqual((op.m_per_xdl, op.n_per_xdl), (16, 16))
            self.assertIn(op.a_element_dtype, ("F16", "BF16"))
            self.assertIn(op.b_element_dtype, ("F16", "BF16"))
            self.assertIn(op.c_element_dtype, ("F16", "BF16"))

    def test_gen_batched_gemm_wmma_instances(self):
        # gfx1250 batched WMMA enumerator. Same 16x16-warp fp16/bf16 shape as the
        # non-batched case, but parsed from a different folder and from a device op
        # with no Ds template parameters.
        instances = gen_batched_gemm_ops_library_wmma()

        log.debug("%d wmma batched gemm instances from library" % len(instances))
        self.assertTrue(instances)
        for op in instances:
            self.assertTrue(op.is_wmma)
            self.assertEqual((op.m_per_xdl, op.n_per_xdl), (16, 16))
            self.assertIn(op.a_element_dtype, ("F16", "BF16"))
            self.assertIn(op.b_element_dtype, ("F16", "BF16"))
            self.assertIn(op.c_element_dtype, ("F16", "BF16"))
            # Ds slots are inserted, not parsed: a wrong ds_mode shifts every
            # subsequent field by two and would leave these non-empty.
            self.assertEqual(op.ds_layouts, ())
            self.assertEqual(op.ds_element_dtypes, ())
            # The instance sources spell the scheduler bare; it must be qualified
            # or the rendered kernel will not compile.
            self.assertTrue(
                str(op.block_gemm_pipeline_scheduler).startswith(
                    "BlockGemmPipelineScheduler::"
                )
            )
            self.assertNotIn("is_wmma", dict(op.dict_items()))

        # v1 is the pipeline the XDL batched instances lack on gfx1250; losing it
        # would silently reintroduce the all-+inf autotune failure this enumerator
        # exists to fix.
        self.assertTrue(
            any(
                op.block_gemm_pipeline_version == "BlockGemmPipelineVersion::v1"
                for op in instances
            )
        )

    def test_batched_gemm_xdl_instances_are_not_wmma(self):
        # The WMMA enumerator must not perturb the XDL one: same instances, and the
        # emitted C++ alias must stay distinct so the two cannot collide.
        instances = gen_batched_gemm_ops_library()

        self.assertTrue(instances)
        for op in instances:
            self.assertFalse(op.is_wmma)
        self.assertIn("_xdl_", instances[0].name())


class TestCheckHeaders(unittest.TestCase):
    """check_headers() header-resolution diagnostic.

    Asserts the structure/invariants rather than a fixed resolved verdict, since
    resolution depends on the runtime layout ($ROCM_HOME / wheel vs source tree).
    The concrete two-environment resolution check is done in the wheel round-trip
    verification, not here (must pass with no GPU and no hipcc)."""

    def test_include_roots_shape(self):
        roots = include_roots()
        self.assertEqual(len(roots), 3)
        # Order must mirror _rocm_include_paths: CK include, CK library include,
        # then ROCm include last.
        self.assertTrue(roots[0].endswith("include"))
        self.assertTrue(roots[1].endswith(("library/include", "library\\include")))

    def test_check_headers_structure(self):
        # try_compile=False keeps this compiler-free and deterministic in CI.
        result = check_headers(try_compile=False)
        for key in ("ck_dir", "rocm_home", "include_roots", "headers", "ok"):
            self.assertIn(key, result)
        self.assertIsInstance(result["ok"], bool)
        self.assertEqual(result["include_roots"], include_roots())

        # Every diagnostic header is reported with the documented per-header shape.
        for hdr in ("ck/ck.hpp", "ck/config.h", "ck_tile/core.hpp"):
            self.assertIn(hdr, result["headers"])
            entry = result["headers"][hdr]
            self.assertIsInstance(entry["resolved"], bool)
            self.assertIn("found_in", entry)
            self.assertIn("compiled", entry)
            # try_compile=False => compile probe skipped.
            self.assertIsNone(entry["compiled"])
            # resolved <=> found_in is a concrete root (invariant, env-independent).
            self.assertEqual(entry["resolved"], entry["found_in"] is not None)

    def test_check_headers_custom_subset(self):
        # Callers may query a subset (the PyTorch gate does this per backend).
        result = check_headers(headers=("ck_tile/core.hpp",), try_compile=False)
        self.assertEqual(set(result["headers"].keys()), {"ck_tile/core.hpp"})
