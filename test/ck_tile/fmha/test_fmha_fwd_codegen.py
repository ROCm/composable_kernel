# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""FMHA codegen and real rendered-dispatch regression tests (CPU only)."""

import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from collections import Counter
from fnmatch import fnmatch
from pathlib import Path
from dataclasses import fields, replace
import sys

_CK_ROOT = Path(__file__).resolve().parents[3]
_FMHA_EXAMPLE_DIR = _CK_ROOT / "example" / "ck_tile" / "01_fmha"
sys.path.insert(0, str(_FMHA_EXAMPLE_DIR))

from codegen.ops.fmha_fwd import (  # noqa: E402
    CppConstraint,
    FMHA_FWD_API_FOOTER,
    FMHA_FWD_API_HEADER,
    FmhaFwdApiPool,
    KernelComponentFactoryGfx125,
    KernelContext,
    ProblemContext,
    get_fwd_blobs,
    write_fwd_api,
)

_SUPPORTED_FEATURE_FILTER = (
    "fmha_fwd_d*_bf16_*_nlogits*_nbias*_nmask*_nlse*_ndropout*_nskip*_nqscale*"
)
_D192_PIPELINE = "qr_tdm_sched"
_HOST_CXX = shutil.which("c++")
_ROCM_ROOT = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
_HIP_HOST_CXX = _ROCM_ROOT / "llvm" / "bin" / "clang++"
_RENDERED_CALL = re.compile(
    r"using trait_ = (fmha_fwd_traits_<[^;\n]+>);\s*"
    r"return fmha_fwd_<trait_, ([^>]+)>\(s, a\);"
)
_D192_ESM2_FILENAME = re.compile(r"^fmha_fwd_d192_bf16_.*_qr_tdm_sched_.*_gfx125\.cpp$")
_D192_BATCH_NMASK_FILENAME = (
    "fmha_fwd_d192_bf16_batch_b128x64x32x128x32x192_"
    "r4x1x1_r4x1x1_w16x16x32_w16x16x32_o2_qr_tdm_sched_"
    "vr_pddv_nlogits_nbias_nmask_nlse_ndropout_nskip_nqscale_"
    "ntrload_nsink_gfx125.cpp"
)
_D192_BATCH_MASK_FILENAME = _D192_BATCH_NMASK_FILENAME.replace("_nmask_", "_mask_")
_D192_GROUP_NMASK_FILENAME = _D192_BATCH_NMASK_FILENAME.replace("_batch_", "_group_")
_D192_BATCH_N128_FILENAME = _D192_BATCH_NMASK_FILENAME.replace(
    "b128x64x32", "b128x128x32"
).replace("_o2_", "_o1_")
_D192_SELECTOR = (
    "a.hdim_q == 192 && a.hdim_v == 128 && a.sink_ptr == nullptr && "
    "a.max_seqlen_q >= 128"
)


def _generate(receipt, optdim_list=None):
    return get_fwd_blobs(
        targets=["gfx1250"],
        kernel_filter=_SUPPORTED_FEATURE_FILTER,
        receipt=receipt,
        optdim_list=[192, 256] if optdim_list is None else optdim_list,
        mask_impl="simplified",
    )


def _is_family_kernel(kernel):
    return kernel.F_pipeline.tag == _D192_PIPELINE


# These stubs provide types and unique launch IDs only. All selection predicates,
# branch ordering, and v3/v2 fallback come from the renderer.
_HOST_HIP_STUB = """#pragma once
#include <cstdlib>
constexpr int hipSuccess = 0;
struct hipDeviceProp_t { unsigned multiProcessorCount = 256; };
inline int hipGetDevice(int* device) { *device = 0; return hipSuccess; }
inline int hipGetDeviceProperties(hipDeviceProp_t* props, int) {
    const char* value = std::getenv("FMHA_TEST_NUM_CUS");
    if(value != nullptr) props->multiProcessorCount = std::atoi(value);
    return hipSuccess;
}
"""

_HOST_FMHA_STUB = """#pragma once
#include <cstdlib>
#include <string>
namespace ck_tile {
using index_t = int;
struct stream_config {};
struct gfx125_t {};
inline std::string get_device_name() {
    const char* value = std::getenv("FMHA_TEST_DEVICE");
    return value == nullptr ? "gfx1250" : value;
}
template <bool> struct SimplifiedGenericAttentionMask {};
enum class BlockFmhaPipelineEnum { QRKSVS, QRKSVS_TDM, QRKSVS_TDM_SCHED };
enum class BlockAttentionBiasEnum { NO_BIAS, ELEMENTWISE_BIAS, ALIBI };
enum class BlockAttentionQuantScaleEnum { NO_SCALE, PERTENSOR, BLOCKSCALE, KV_BLOCKSCALE, MX };
}
enum class mask_enum { no_mask, mask_top_left, mask_bottom_right, window_generic };
enum class bias_enum { no_bias, elementwise_bias, alibi };
enum class quant_scale_enum { no_scale, pertensor, blockscale, kv_blockscale, mx };
struct FmhaFwdBf16 {};
struct FmhaFwdFp16 {};
struct FmhaMasks {
    struct NoMask {};
    struct GenericMask {};
    struct CausalMask {};
};
inline constexpr ck_tile::index_t fmha_fwd_largest_n_tile_size = 128;
template <int, typename, bool, int, int, int, int, int, int, bool, auto, bool,
          typename, auto, bool, bool, auto, bool, bool, bool, bool, bool, bool, bool, int = -1>
struct fmha_fwd_traits_ {};
struct fmha_fwd_traits {
    int hdim_q = 192, hdim_v = 128;
    std::string data_type = "bf16";
    bool is_group_mode = false, is_v_rowmajor = true, has_logits_soft_cap = false;
    mask_enum mask_type = mask_enum::no_mask;
    bias_enum bias_type = bias_enum::no_bias;
    bool has_lse = false, has_dropout = false;
    quant_scale_enum qscale_type = quant_scale_enum::no_scale;
    bool skip_min_seqlen_q = false, has_sink = false;
};
struct fmha_fwd_args {
    int hdim_q = 192, hdim_v = 128, seqlen_q = 128, max_seqlen_q = 128, seqlen_k = 128;
    int stride_q = 192, nhead_stride_q = 24576;
    int window_size_left = -1, window_size_right = -1;
    int batch = 2, nhead_q = 8, nhead_k = 2;
    const void* sink_ptr = nullptr;
    const void* seqlen_k_ptr = nullptr;
    const void* cu_seqlen_k_ptr = nullptr;
};
template <typename Trait, typename Arch> struct dispatch_id;
template <typename Trait, typename Arch>
float fmha_fwd_(const ck_tile::stream_config&, const fmha_fwd_args&) {
    return dispatch_id<Trait, Arch>::value;
}
"""

_HOST_MAIN = """
#include <iostream>
int main() {
    fmha_fwd_traits t;
    fmha_fwd_args a;
    int mask, bias, scale, arg_q, arg_v, max_q, has_cu_sk, has_sk_ptr, has_sink_ptr;
    if(!(std::cin >> t.hdim_q >> t.hdim_v >> a.seqlen_q >> a.seqlen_k
         >> t.is_group_mode >> t.is_v_rowmajor >> t.has_logits_soft_cap >> mask >> bias
         >> t.has_lse >> t.has_dropout >> scale >> t.skip_min_seqlen_q >> t.has_sink
         >> t.data_type >> arg_q >> arg_v >> max_q >> a.batch >> a.nhead_q
         >> a.nhead_k >> a.window_size_left >> a.window_size_right
         >> has_cu_sk >> has_sk_ptr >> has_sink_ptr)) return 2;
    t.mask_type = static_cast<mask_enum>(mask);
    t.bias_type = static_cast<bias_enum>(bias);
    t.qscale_type = static_cast<quant_scale_enum>(scale);
    a.hdim_q = arg_q < 0 ? t.hdim_q : arg_q;
    a.hdim_v = arg_v < 0 ? t.hdim_v : arg_v;
    a.max_seqlen_q = max_q < 0 ? a.seqlen_q : max_q;
    a.cu_seqlen_k_ptr = has_cu_sk ? &a : nullptr;
    a.seqlen_k_ptr = has_sk_ptr ? &a : nullptr;
    a.sink_ptr = has_sink_ptr ? &a : nullptr;
    std::cout << fmha_fwd(t, a, {}) << '\\n';
}
"""


class _CompiledDispatcher:
    def __init__(self, test, pool, kernels, filter_fn=None, mutate=None):
        directory = tempfile.TemporaryDirectory(prefix="fmha-dispatch-host-")
        test.addCleanup(directory.cleanup)
        root = Path(directory.name)
        if filter_fn is None:
            write_fwd_api(pool, root)
            api = (root / "fmha_fwd_api.cpp").read_text()
        else:
            api = (
                FMHA_FWD_API_HEADER
                + pool.render("fmha_fwd_v2", filter_fn=filter_fn)
                + pool.render("fmha_fwd_v3", filter_fn=lambda trait: False)
                + FMHA_FWD_API_FOOTER
            )
        signatures = list(dict.fromkeys(_RENDERED_CALL.findall(api)))
        ids = {signature: index + 1 for index, signature in enumerate(signatures)}
        self.kernel_ids = []
        for kernel in kernels:
            single = FmhaFwdApiPool()
            single.register_traits(kernel.api_trait())
            (signature,) = _RENDERED_CALL.findall(single.render("unused"))
            if signature in ids:
                self.kernel_ids.append((kernel, ids[signature]))
        test.assertEqual(set(ids.values()), {value for _, value in self.kernel_ids})
        specializations = "\n".join(
            f"template <> struct dispatch_id<{trait}, {arch}> {{ "
            f"static constexpr int value = {value}; }};"
            for (trait, arch), value in ids.items()
        )
        (root / "hip").mkdir()
        (root / "hip" / "hip_runtime.h").write_text(_HOST_HIP_STUB)
        (root / "fmha_fwd.hpp").write_text(_HOST_FMHA_STUB + specializations)
        (root / "fmha_fwd_api.cpp").write_text(
            (api if mutate is None else mutate(api)) + _HOST_MAIN
        )
        self.binary = root / "dispatcher"
        compile_result = subprocess.run(
            [
                _HOST_CXX,
                "-std=c++17",
                "-O0",
                "-I",
                str(root),
                str(root / "fmha_fwd_api.cpp"),
                "-o",
                str(self.binary),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        test.assertEqual(compile_result.returncode, 0, compile_result.stderr)

    def expected_id(self, **fields):
        def value(kernel, name):
            if name in ("mode", "hdim", "dtype"):
                return getattr(kernel, "F_" + name)
            if name in ("bm0", "bn0", "occupancy"):
                return getattr(kernel.F_tile, "F_" + name)
            if name == "pipeline":
                return kernel.F_pipeline.tag
            return getattr(kernel.F_pipeline, "F_" + name)

        matches = {
            index
            for kernel, index in self.kernel_ids
            if all(value(kernel, key) == expected for key, expected in fields.items())
        }
        if len(matches) != 1:
            raise AssertionError(
                f"expected one exact rendered trait for {fields}: {matches}"
            )
        return matches.pop()

    def run(self, device="gfx1250", num_cus=256, **changes):
        case = dict(
            hdim_q=192,
            hdim_v=128,
            seqlen_q=128,
            seqlen_k=128,
            group=0,
            row=1,
            logits=0,
            mask=0,
            bias=0,
            lse=0,
            dropout=0,
            qscale=0,
            skip=0,
            sink=0,
            dtype="bf16",
            arg_q=-1,
            arg_v=-1,
            max_seqlen_q=-1,
            batch=2,
            nhead_q=8,
            nhead_k=2,
            window_size_left=-1,
            window_size_right=-1,
            has_cu_sk=0,
            has_sk_ptr=0,
            has_sink_ptr=0,
        )
        if changes.keys() - case.keys():
            raise ValueError(
                f"unknown dispatcher fields: {changes.keys() - case.keys()}"
            )
        case.update(changes)
        environment = dict(
            os.environ,
            HIP_VISIBLE_DEVICES="",
            ROCR_VISIBLE_DEVICES="",
            FMHA_TEST_DEVICE=device,
            FMHA_TEST_NUM_CUS=str(num_cus),
        )
        result = subprocess.run(
            [str(self.binary)],
            input=" ".join(map(str, case.values())) + "\n",
            env=environment,
            text=True,
            capture_output=True,
            timeout=10,
            check=True,
        )
        return [int(value) for value in result.stdout.split()]


class TestGfx125D192Codegen(unittest.TestCase):
    @unittest.skipUnless(shutil.which("cmake"), "cmake is required")
    def test_cmake_esm2_scope_and_default_wwm(self):
        source_flags = (_FMHA_EXAMPLE_DIR / "cmake") / "fmha_fwd_source_flags.cmake"
        d256_source = (
            "fmha_fwd_d256_bf16_batch_b64x64x32x256x32x256_"
            "r4x1x1_r4x1x1_w16x16x32_w16x16x32_qr_"
            "vr_npad_nlogits_nbias_nmask_nlse_ndropout_nskip_nqscale_"
            "ntrload_nsink_gfx125.cpp"
        )
        sources = (
            _D192_BATCH_NMASK_FILENAME,
            _D192_BATCH_N128_FILENAME,
            _D192_BATCH_N128_FILENAME.replace("_bf16_", "_fp16_"),
            _D192_BATCH_NMASK_FILENAME.replace("_nlse_", "_lse_"),
            _D192_BATCH_MASK_FILENAME,
            _D192_GROUP_NMASK_FILENAME,
            d256_source,
        )

        def configure():
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for source in sources:
                    (root / source).touch()
                source_list = "\n  ".join(
                    f'"${{CMAKE_CURRENT_LIST_DIR}}/{x}"' for x in sources
                )
                (root / "CMakeLists.txt").write_text(
                    f"""cmake_minimum_required(VERSION 3.20)
project(fmha_source_flags LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(\"{source_flags}\")
set(sources
  {source_list}
)
add_library(fmha_source_flags OBJECT ${{sources}})
foreach(source IN LISTS sources)
  get_filename_component(source_name \"${{source}}\" NAME)
  ck_tile_fmha_fwd_get_source_compile_options(
    \"${{source_name}}\" source_compile_options)
  if(source_compile_options)
    set_property(SOURCE \"${{source}}\" APPEND PROPERTY COMPILE_OPTIONS
      ${{source_compile_options}})
  endif()
endforeach()
"""
                )
                subprocess.run(
                    ["cmake", "-S", str(root), "-B", str(root / "build")],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                commands = json.loads(
                    (root / "build" / "compile_commands.json").read_text()
                )
                return {
                    Path(entry["file"]).name: entry["command"] for entry in commands
                }

        commands = configure()
        for source in sources:
            self.assertNotIn("-wwm-regalloc=fast", commands[source])
            self.assertEqual(
                "-amdgpu-expert-scheduling-mode" in commands[source],
                source != d256_source,
            )

    def test_receipts_emit_d192_sched_and_fallback_tiles(self):
        for receipt in (100, 200, 600):
            with self.subTest(receipt=receipt):
                _, kernels = _generate(receipt)
                candidates = [
                    kernel
                    for kernel in kernels
                    if kernel.F_pipeline.tag == _D192_PIPELINE
                ]

                self.assertTrue(candidates)
                for kernel in candidates:
                    self.assertEqual(kernel.F_arch.name, "gfx125")
                    self.assertEqual(kernel.F_dtype, "bf16")
                    self.assertEqual(kernel.F_hdim, 192)
                    self.assertIn(
                        (kernel.F_tile.F_bn0, kernel.F_tile.F_occupancy),
                        ((64, 2), (128, 1)),
                    )
                    if kernel.F_tile.F_bn0 == 128:
                        self.assertEqual(
                            kernel.F_pipeline.F_spad,
                            "t" if kernel.F_mode == "group" else "f",
                        )
                    self.assertEqual(kernel.F_tile.F_bn1, 128)
                    self.assertEqual(kernel.F_tile.F_bk0max, 192)
                    self.assertEqual(kernel.F_pipeline.F_vlayout, "row")
                    self.assertEqual(kernel.F_pipeline.F_logits, "f")
                    self.assertEqual(kernel.F_pipeline.F_bias, "no")
                    self.assertEqual(kernel.F_pipeline.F_dropout, "f")
                    self.assertEqual(kernel.F_pipeline.F_qscale, "no")
                    self.assertEqual(kernel.F_pipeline.F_skip, "f")
                    self.assertEqual(kernel.F_pipeline.F_sink, "f")
                    self.assertRegex(kernel.filename, _D192_ESM2_FILENAME)

                for kernel in kernels:
                    if kernel not in candidates:
                        self.assertNotRegex(kernel.filename, _D192_ESM2_FILENAME)

                self.assertTrue(
                    any(
                        kernel.F_hdim == 192
                        and kernel.F_tile.F_bn1 == 128
                        and kernel.F_pipeline.tag == "qr_tdm"
                        for kernel in kernels
                    )
                )
                self.assertEqual(
                    sum(kernel.F_tile.F_bn0 == 128 for kernel in candidates),
                    2 if receipt == 600 else 1,
                )

    def test_d128_retains_short_query_fallback_tiles(self):
        for receipt in (100, 200, 600):
            with self.subTest(receipt=receipt):
                _, kernels = _generate(receipt, [128])
                self.assertTrue(any(_is_family_kernel(k) for k in kernels))
                fallback = [k for k in kernels if not _is_family_kernel(k)]
                self.assertTrue(fallback)
                self.assertTrue(all(k.F_hdim == 128 for k in fallback))
                self.assertTrue(any(k.F_tile.F_bm0 < 128 for k in fallback))
                self.assertTrue(all(k.F_tile.F_bn0 == 64 for k in fallback))
                self.assertTrue(any(k.F_pipeline.tag == "qr_tdm" for k in fallback))
                self.assertEqual(len(kernels), len({k.filename for k in kernels}))
                for kernel in fallback:
                    self.assertNotIn("QRKSVS_TDM_SCHED", kernel.render())

    def test_generated_api_places_exact_candidate_before_legacy_entries(self):
        for receipt in (100, 200, 600):
            with self.subTest(receipt=receipt):
                api_pool, kernels = _generate(receipt)
                with tempfile.TemporaryDirectory() as output_dir:
                    output_path = Path(output_dir)
                    write_fwd_api(api_pool, output_path)
                    api = (output_path / "fmha_fwd_api.cpp").read_text()

                self.assertIn("QRKSVS_TDM_SCHED", api)
                self.assertIn(_D192_SELECTOR, api)
                if receipt != 200:
                    self.assertIn(
                        "a.seqlen_k >= 512LL * (t.is_group_mode ? a.batch : 1)",
                        api,
                    )
                    self.assertIn("a.seqlen_k_ptr == nullptr", api)
                    self.assertIn("get_num_blocks(128) <= num_cus", api)
                self.assertNotIn(f"!({_D192_SELECTOR})", api)
                self.assertLess(api.index("t.hdim_q <= 192"), api.index(_D192_SELECTOR))
                self.assertLess(api.index(_D192_SELECTOR), api.index("t.hdim_q <= 256"))
                self.assertLess(
                    api.index("QRKSVS_TDM_SCHED", api.index("t.hdim_q <= 192")),
                    api.index(
                        "BlockFmhaPipelineEnum::QRKSVS_TDM,",
                        api.index("t.hdim_q <= 192"),
                    ),
                )
                self.assertNotIn("is_gfx125_tdm_v128_enabled", api)

                candidate = next(
                    kernel
                    for kernel in kernels
                    if kernel.F_pipeline.tag == _D192_PIPELINE
                )
                self.assertEqual((candidate.F_hdim, candidate.F_tile.F_bn1), (192, 128))
                source = candidate.render()
                self.assertIn("BlockFmhaPipelineQRKSVSTdmSched", source)
                self.assertIn(
                    "FmhaTdmSchedPolicyFor<fmha_pipeline_problem>, false", source
                )
                self.assertIn("QRKSVS_TDM_SCHED", source)
                self.assertIn(_D192_PIPELINE, candidate.name)

    def test_filtered_unsupported_feature_keeps_ungated_generic_fallback(self):
        filters = {
            "logits": "fmha_fwd_d*_bf16_*_logits*_nbias*_nmask*_nlse*_ndropout*_nskip*_nqscale*",
            "bias": "fmha_fwd_d*_bf16_*_nlogits*_bias*_nmask*_nlse*_ndropout*_nskip*_nqscale*",
            "dropout": "fmha_fwd_d*_bf16_*_nlogits*_nbias*_nmask*_nlse*_dropout*_nskip*_nqscale*",
            "skip": "fmha_fwd_d*_bf16_*_nlogits*_nbias*_nmask*_nlse*_ndropout*_skip*_nqscale*",
            "sink": "fmha_fwd_d*_bf16_*_nlogits*_nbias*_nmask*_nlse*_ndropout*_nskip*_nqscale*_sink*",
        }
        for feature, kernel_filter in filters.items():
            with self.subTest(feature=feature):
                api_pool, kernels = get_fwd_blobs(
                    targets=["gfx1250"],
                    kernel_filter=kernel_filter,
                    receipt=100,
                    optdim_list=[192, 256],
                    mask_impl="simplified",
                )
                with tempfile.TemporaryDirectory() as output_dir:
                    output_path = Path(output_dir)
                    write_fwd_api(api_pool, output_path)
                    api = (output_path / "fmha_fwd_api.cpp").read_text()

                self.assertTrue(kernels)
                self.assertFalse(
                    any(kernel.F_pipeline.tag == _D192_PIPELINE for kernel in kernels)
                )
                self.assertNotIn("QRKSVS_TDM_SCHED", api)
                self.assertNotIn(_D192_SELECTOR, api)

        _, qscale_kernels = get_fwd_blobs(
            targets=["gfx1250"],
            kernel_filter="fmha_fwd_d*_bf16_*_pertensor*",
            receipt=100,
            optdim_list=[192, 256],
            mask_impl="simplified",
        )
        self.assertFalse(qscale_kernels)

    def test_filters_select_only_matching_sources(self):
        for receipt in (100, 200, 600):
            for kernel_filter in ("*d192_bf16*", f"*{_D192_PIPELINE}*"):
                with self.subTest(receipt=receipt, kernel_filter=kernel_filter):
                    api_pool, kernels = get_fwd_blobs(
                        targets=["gfx1250"],
                        kernel_filter=kernel_filter,
                        receipt=receipt,
                        optdim_list=[192, 256],
                        mask_impl="simplified",
                    )
                    candidate_count = sum(
                        kernel.F_pipeline.tag == _D192_PIPELINE for kernel in kernels
                    )
                    self.assertGreater(candidate_count, 0)
                    self.assertTrue(
                        all(fnmatch(kernel.name, kernel_filter) for kernel in kernels)
                    )
                    self.assertTrue(all(kernel.F_hdim == 192 for kernel in kernels))
                    if kernel_filter == f"*{_D192_PIPELINE}*":
                        self.assertEqual(candidate_count, len(kernels))

                    with tempfile.TemporaryDirectory() as output_dir:
                        output_path = Path(output_dir)
                        write_fwd_api(api_pool, output_path)
                        api = (output_path / "fmha_fwd_api.cpp").read_text()
                    self.assertIn(_D192_SELECTOR, api)
                    self.assertNotIn(f"!({_D192_SELECTOR})", api)

    def test_only_gfx125_targets_emit_the_scheduled_pipeline(self):
        for target in ("gfx942", "gfx950"):
            with self.subTest(target=target):
                _, kernels = get_fwd_blobs(
                    targets=[target],
                    kernel_filter=f"*{_D192_PIPELINE}*",
                    receipt=100,
                    optdim_list=[192],
                    mask_impl="simplified",
                )
                self.assertFalse(kernels)

        for target in ("gfx1250", "gfx1251", "gfx1252"):
            with self.subTest(target=target):
                _, kernels = get_fwd_blobs(
                    targets=[target],
                    kernel_filter=f"*{_D192_PIPELINE}*",
                    receipt=100,
                    optdim_list=[192],
                    mask_impl="simplified",
                )
                self.assertTrue(
                    any(kernel.F_pipeline.tag == _D192_PIPELINE for kernel in kernels)
                )

    def test_n128_emits_batch_and_group_without_generic_windows(self):
        mask_options = {
            "simplified": ("s_no", "s_mask"),
            "generic": ("no", "causal"),
        }
        for dtype in ("fp16", "bf16"):
            for mask_impl, masks in mask_options.items():
                with self.subTest(dtype=dtype, mask_impl=mask_impl):
                    self.check_n128_masks(dtype, mask_impl, masks)

    def check_n128_masks(self, dtype, mask_impl, masks):
        _, kernels = get_fwd_blobs(
            targets=["gfx1250"],
            kernel_filter=f"*{dtype}*{_D192_PIPELINE}*",
            receipt=600,
            optdim_list=[192],
            mask_impl=mask_impl,
        )
        n128 = [
            kernel
            for kernel in kernels
            if kernel.F_pipeline.tag == _D192_PIPELINE and kernel.F_tile.F_bn0 == 128
        ]
        actual = Counter(
            (kernel.F_mode, kernel.F_pipeline.F_mask, kernel.F_pipeline.F_lse)
            for kernel in n128
        )
        expected = Counter(
            {
                (mode, mask, lse): 1
                for mode in ("batch", "group")
                for mask in masks
                for lse in ("f", "t")
            }
        )
        self.assertEqual(actual, expected)
        for kernel in n128:
            required_pad = "t" if kernel.F_mode == "group" else "f"
            self.assertEqual(kernel.F_pipeline.F_spad, required_pad)
            self.assertEqual(kernel.F_pipeline.F_skpad, required_pad)


@unittest.skipUnless(_HOST_CXX, "a C++17 host compiler is required")
class TestCompiledGfx125D192Dispatch(unittest.TestCase):
    """Check generated CPU dispatch; kernel validity is tested separately."""

    def compile(
        self,
        receipt=600,
        optdims=None,
        kernel_filter=_SUPPORTED_FEATURE_FILTER,
        filter_fn=None,
        mutate=None,
        mask_impl="simplified",
    ):
        pool, kernels = get_fwd_blobs(
            targets=["gfx1250"],
            kernel_filter=kernel_filter,
            receipt=receipt,
            optdim_list=[128, 192, 256] if optdims is None else optdims,
            mask_impl=mask_impl,
        )
        return _CompiledDispatcher(self, pool, kernels, filter_fn, mutate), kernels

    @staticmethod
    def candidate_id(dispatch, mode, mask="s_no", lse="f", bn0=64, dtype="bf16"):
        return dispatch.expected_id(
            mode=mode,
            dtype=dtype,
            hdim=192,
            pipeline=_D192_PIPELINE,
            mask=mask,
            lse=lse,
            bn0=bn0,
            sink="f",
        )

    @staticmethod
    def fallback_id(dispatch, mode, mask="s_no", lse="f", sink="f", bm0=128):
        return dispatch.expected_id(
            mode=mode,
            hdim=192,
            pipeline="qr_tdm",
            bm0=bm0,
            mask=mask,
            lse=lse,
            spad="t" if mode == "group" else "f",
            dpad="t",
            sink=sink,
        )

    @staticmethod
    def generic_qr_id(dispatch, mode, mask="s_no", lse="f", padded="t", sink="f"):
        return dispatch.expected_id(
            mode=mode,
            hdim=256,
            pipeline="qr",
            mask=mask,
            lse=lse,
            spad=padded,
            dpad=padded,
            sink=sink,
        )

    def test_real_dispatch_priority_exact_shape_and_short_q(self):
        dispatch, _ = self.compile()
        for mode, group in (("batch", 0), ("group", 1)):
            candidate = self.candidate_id(dispatch, mode)
            fallback = self.fallback_id(dispatch, mode)
            self.assertEqual(dispatch.run(group=group), [candidate])
            for length in (128, 129, 257):
                self.assertEqual(
                    dispatch.run(group=group, seqlen_q=length), [candidate]
                )
            for length in (0, 1, 64, 127):
                self.assertEqual(
                    dispatch.run(group=group, seqlen_q=length),
                    [self.fallback_id(dispatch, mode, bm0=64)],
                )
            self.assertEqual(
                dispatch.run(group=group, hdim_q=160, hdim_v=128), [fallback]
            )
            for q, v in ((191, 128), (192, 127)):
                with self.subTest(mode=mode, shape=(q, v)):
                    self.assertEqual(
                        dispatch.run(group=group, hdim_q=q, hdim_v=v),
                        [
                            dispatch.expected_id(
                                mode=mode,
                                hdim=192,
                                pipeline="qr",
                                bm0=128,
                                mask="s_no",
                                lse="f",
                                spad="t",
                                dpad="t",
                                sink="f",
                            )
                        ],
                    )
            self.assertEqual(
                dispatch.run(group=group, hdim_q=193, hdim_v=128),
                [self.generic_qr_id(dispatch, mode)],
            )
            self.assertEqual(
                dispatch.run(group=group, hdim_q=192, hdim_v=192),
                [
                    dispatch.expected_id(
                        mode=mode,
                        hdim=256,
                        pipeline="qr_tdm",
                        bm0=64,
                        mask="s_no",
                        lse="f",
                        spad="t" if group else "f",
                        dpad="t",
                        sink="f",
                    )
                ],
            )
            generic256 = dispatch.expected_id(
                mode=mode,
                hdim=256,
                pipeline="qr_tdm",
                bm0=64,
                mask="s_no",
                lse="f",
                spad="t" if group else "f",
                dpad="f",
                sink="f",
            )
            self.assertEqual(
                dispatch.run(group=group, hdim_q=256, hdim_v=256), [generic256]
            )

    def test_n128_long_k_one_wave_selector_and_n64_fallback(self):
        dispatch, _ = self.compile(
            kernel_filter=(
                "fmha_fwd_d*_bf16_*_nlogits*_nbias*_*lse*_ndropout*_nskip*_nqscale*"
            )
        )
        n64 = self.candidate_id(dispatch, "batch")
        n128 = self.candidate_id(dispatch, "batch", bn0=128)

        for sk in (128, 192, 256, 511):
            self.assertEqual(dispatch.run(seqlen_k=sk), [n64])
        for sk in (512, 513, 2048):
            self.assertEqual(dispatch.run(seqlen_k=sk), [n128])
        self.assertEqual(dispatch.run(seqlen_k=512, has_cu_sk=1), [n64])

        grid256 = dict(batch=8, nhead_q=8, seqlen_q=512, seqlen_k=512)
        self.assertEqual(dispatch.run(**grid256), [n128])
        self.assertEqual(dispatch.run(num_cus=128, **grid256), [n64])
        self.assertEqual(dispatch.run(**{**grid256, "batch": 16}), [n64])
        self.assertEqual(dispatch.run(max_seqlen_q=1024, **grid256), [n64])
        for device in ("gfx1251", "gfx1252"):
            self.assertEqual(dispatch.run(device=device, **grid256), [n128])
            self.assertEqual(dispatch.run(device=device, seqlen_k=511), [n64])

        for mask in (1, 2):
            causal = dict(
                mask=mask,
                seqlen_q=512,
                seqlen_k=512,
                window_size_left=-1,
                window_size_right=0,
            )
            self.assertEqual(
                dispatch.run(**causal),
                [self.candidate_id(dispatch, "batch", mask="s_mask", bn0=128)],
            )
            for disqualifier in (
                {"window_size_left": 128},
                {"window_size_right": 64},
                {"seqlen_q": 256},
            ):
                self.assertEqual(
                    dispatch.run(**{**causal, **disqualifier}),
                    [self.candidate_id(dispatch, "batch", mask="s_mask")],
                )
        self.assertEqual(
            dispatch.run(mask=3, seqlen_k=512),
            [self.candidate_id(dispatch, "batch", mask="s_mask")],
        )
        self.assertEqual(
            dispatch.run(group=1, seqlen_k=512),
            [self.candidate_id(dispatch, "group")],
        )
        self.assertEqual(
            dispatch.run(group=1, seqlen_k=1024),
            [self.candidate_id(dispatch, "group", bn0=128)],
        )
        self.assertEqual(
            dispatch.run(group=1, seqlen_k=1024, has_cu_sk=1),
            [self.candidate_id(dispatch, "group")],
        )
        self.assertEqual(
            dispatch.run(group=1, seqlen_k=1024, has_sk_ptr=1),
            [self.candidate_id(dispatch, "group")],
        )
        self.assertEqual(
            dispatch.run(group=1, batch=4, seqlen_k=2047),
            [self.candidate_id(dispatch, "group")],
        )
        self.assertEqual(
            dispatch.run(group=1, batch=4, seqlen_k=2048),
            [self.candidate_id(dispatch, "group", bn0=128)],
        )
        self.assertEqual(
            dispatch.run(lse=1, seqlen_k=512),
            [self.candidate_id(dispatch, "batch", lse="t", bn0=128)],
        )

    def test_fp16_scheduled_dispatch_for_d128_and_d192(self):
        dispatch, kernels = self.compile(kernel_filter="*qr_tdm_sched*")
        self.assertEqual(
            {(k.F_dtype, k.F_hdim, k.F_tile.F_bn0) for k in kernels},
            {
                (dtype, hdim, bn0)
                for dtype in ("fp16", "bf16")
                for hdim, bn0 in ((128, 128), (192, 64), (192, 128))
            },
        )

        for mode, group in (("batch", 0), ("group", 1)):
            for hdim, short_bn0 in ((128, 128), (192, 64)):
                with self.subTest(mode=mode, hdim=hdim, length="short"):
                    expected = dispatch.expected_id(
                        dtype="fp16",
                        mode=mode,
                        hdim=hdim,
                        bn0=short_bn0,
                        pipeline="qr_tdm_sched",
                        mask="s_no",
                        lse="f",
                        sink="f",
                    )
                    self.assertEqual(
                        dispatch.run(
                            dtype="fp16", group=group, hdim_q=hdim, seqlen_k=128
                        ),
                        [expected],
                    )

            long_k = 1024 if group else 512
            n128 = self.candidate_id(dispatch, mode, bn0=128, dtype="fp16")
            self.assertEqual(
                dispatch.run(
                    dtype="fp16",
                    group=group,
                    hdim_q=192,
                    seqlen_q=long_k,
                    seqlen_k=long_k,
                ),
                [n128],
            )

    def test_n64_only_filtered_inventory_still_dispatches_long_k(self):
        dispatch, _ = self.compile(
            filter_fn=lambda trait: (
                trait.pipeline_tag != _D192_PIPELINE or trait.bn0 == 64
            )
        )
        self.assertEqual(
            dispatch.run(seqlen_k=512), [self.candidate_id(dispatch, "batch")]
        )

    def test_n128_generic_no_mask_and_causal_dispatch(self):
        dispatch, _ = self.compile(kernel_filter="*qr_tdm_sched*", mask_impl="generic")
        for mode, group in (("batch", 0), ("group", 1)):
            with self.subTest(mode=mode):
                length = 1024 if group else 512
                self.assertEqual(
                    dispatch.run(group=group, seqlen_q=length, seqlen_k=length),
                    [self.candidate_id(dispatch, mode, mask="no", bn0=128)],
                )
                causal = dict(
                    group=group,
                    mask=1,
                    seqlen_q=length,
                    seqlen_k=length,
                    window_size_left=-1,
                    window_size_right=0,
                )
                n128 = self.candidate_id(dispatch, mode, mask="causal", bn0=128)
                self.assertEqual(dispatch.run(**causal), [n128])
                self.assertEqual(dispatch.run(**{**causal, "mask": 2}), [n128])
                generic = self.candidate_id(dispatch, mode, mask="generic")
                self.assertEqual(
                    dispatch.run(**{**causal, "mask": 3, "window_size_left": 64}),
                    [generic],
                )

    def test_family_only_and_empty_filtered_dispatch(self):
        family, _ = self.compile(
            filter_fn=lambda trait: (
                trait.pipeline_tag == _D192_PIPELINE and str(trait.hdim) == "192"
            )
        )
        candidate = self.candidate_id(family, "batch")
        self.assertEqual(family.run(), [candidate])
        for arguments in ({"seqlen_q": 127}, {"hdim_q": 128}):
            self.assertEqual(family.run(**arguments), [-1])
        empty, _ = self.compile(filter_fn=lambda trait: False)
        self.assertEqual(empty.run(), [-1])

    def test_mismatched_trait_argument_dimensions_never_select_family(self):
        dispatch, _ = self.compile()
        for mode, group in (("batch", 0), ("group", 1)):
            candidate = self.candidate_id(dispatch, mode)
            for dimensions in (
                dict(hdim_q=128, arg_q=192),
                dict(hdim_q=256, arg_q=192),
                dict(hdim_v=256, arg_v=128),
                dict(arg_q=128),
                dict(arg_v=256),
            ):
                with self.subTest(mode=mode, dimensions=dimensions):
                    self.assertNotEqual(
                        dispatch.run(group=group, **dimensions), [candidate]
                    )

    def test_real_dispatch_d128_priority_and_sequence_boundary(self):
        dispatch, kernels = self.compile()
        self.assertTrue(
            [
                kernel
                for kernel in kernels
                if kernel.F_hdim == 128 and _is_family_kernel(kernel)
            ]
        )
        for mode, group in (("batch", 0), ("group", 1)):
            for length, bm0 in ((1, 64), (127, 64), (128, 64), (2047, 64), (2048, 128)):
                expected = dispatch.expected_id(
                    mode=mode,
                    hdim=128,
                    pipeline="qr_tdm",
                    bm0=bm0,
                    spad="t" if group else "f",
                    mask="s_no",
                    lse="f",
                    sink="f",
                    dpad="f",
                )
                selected = expected
                if length >= 128:
                    selected = dispatch.expected_id(
                        mode=mode,
                        hdim=128,
                        pipeline="qr_tdm_sched",
                        mask="s_no",
                        lse="f",
                        sink="f",
                    )
                self.assertEqual(
                    dispatch.run(
                        group=group,
                        hdim_q=128,
                        hdim_v=128,
                        seqlen_q=length,
                    ),
                    [selected],
                )

    def test_real_dispatch_mask_lse_and_unsupported_features(self):
        kernel_filter = (
            "fmha_fwd_d*_bf16_*_nlogits*_nbias*_*lse*_ndropout*_nskip*_nqscale*"
        )
        dispatch, _ = self.compile(kernel_filter=kernel_filter)
        for mode, group in (("batch", 0), ("group", 1)):
            for mask in (0, 1, 2, 3):
                for lse in (0, 1):
                    fields = dict(
                        mask="s_no" if mask == 0 else "s_mask", lse="t" if lse else "f"
                    )
                    candidate = self.candidate_id(dispatch, mode, **fields)
                    self.assertEqual(
                        dispatch.run(group=group, mask=mask, lse=lse), [candidate]
                    )
            sink = self.fallback_id(dispatch, mode, sink="t")
            self.assertEqual(dispatch.run(group=group, sink=1), [sink])
        for changes in (
            dict(row=0),
            dict(logits=1),
            dict(bias=1),
            dict(dropout=1),
            dict(qscale=1),
            dict(skip=1),
            dict(dtype="fp16"),
            dict(hdim_q=257),
            dict(hdim_v=257),
            dict(device="gfx942"),
        ):
            with self.subTest(unsupported=changes):
                self.assertEqual(dispatch.run(**changes), [-1])

        for device in ("gfx1251", "gfx1252"):
            for mode, group in (("batch", 0), ("group", 1)):
                with self.subTest(device=device, mode=mode):
                    self.assertEqual(
                        dispatch.run(device=device, group=group),
                        [self.candidate_id(dispatch, mode)],
                    )

    def test_real_dispatch_missing_candidate_does_not_gate_generic(self):
        for render_filter in (True, False):
            with self.subTest(render_filter=render_filter):
                dispatch, _ = self.compile(
                    kernel_filter=(
                        _SUPPORTED_FEATURE_FILTER
                        if render_filter
                        else _SUPPORTED_FEATURE_FILTER.replace("d*_", "d256_", 1)
                    ),
                    filter_fn=(
                        (lambda trait: trait.pipeline_tag != _D192_PIPELINE)
                        if render_filter
                        else None
                    ),
                )
                for mode, group in (("batch", 0), ("group", 1)):
                    fallback = (
                        self.fallback_id(dispatch, mode)
                        if render_filter
                        else dispatch.expected_id(
                            mode=mode,
                            hdim=256,
                            pipeline="qr_tdm",
                            bm0=64,
                            mask="s_no",
                            lse="f",
                            spad="t" if group else "f",
                            dpad="t",
                            sink="f",
                        )
                    )
                    self.assertEqual(dispatch.run(group=group), [fallback])

    def test_real_dispatch_candidate_only_filter_is_strict(self):
        for receipt in (0, 100, 200, 600):
            with self.subTest(receipt=receipt):
                dispatch, kernels = self.compile(
                    receipt=receipt, kernel_filter=f"*{_D192_PIPELINE}*"
                )
                self.assertTrue(kernels)
                self.assertTrue(all(_is_family_kernel(kernel) for kernel in kernels))
                self.assertEqual(len({kernel.name for kernel in kernels}), len(kernels))
                for mode in sorted({kernel.F_mode for kernel in kernels}):
                    group = int(mode == "group")
                    self.assertEqual(
                        dispatch.run(group=group), [self.candidate_id(dispatch, mode)]
                    )
                    self.assertEqual(dispatch.run(group=group, seqlen_q=127), [-1])
                    self.assertEqual(dispatch.run(group=group, hdim_q=160), [-1])
                    self.assertEqual(dispatch.run(group=group, logits=1), [-1])

    def test_real_dispatch_receipt_and_optdim_baseline(self):
        cmake = (_FMHA_EXAMPLE_DIR / "CMakeLists.txt").read_text()
        gfx125_dims = re.search(
            r"if\(FMHA_FWD_HAS_GFX125\)\s+set\(FMHA_FWD_OPTDIM\s+([0-9,]+)\)",
            cmake,
        )
        self.assertIsNotNone(gfx125_dims)
        common_dims = list(map(int, gfx125_dims[1].split(",")))
        self.assertEqual(common_dims, [32, 64, 96, 128, 160, 192, 256])
        for receipt in (0, 100, 200, 600):
            for optdims in ([128], [192], [256], [-1], common_dims):
                with self.subTest(receipt=receipt, optdims=optdims):
                    dispatch, kernels = self.compile(receipt=receipt, optdims=optdims)
                    modes = sorted({kernel.F_mode for kernel in kernels})
                    family = [kernel for kernel in kernels if _is_family_kernel(kernel)]
                    expected_family_dims = (
                        {64, 128, 192}
                        if optdims == [-1]
                        else {64, 128, 192}.intersection(optdims)
                    )
                    self.assertEqual(
                        {kernel.F_hdim for kernel in family}, expected_family_dims
                    )
                    for mode in modes:
                        group = int(mode == "group")
                        if optdims == [128]:
                            self.assertEqual(dispatch.run(group=group), [-1])
                        elif optdims == [256]:
                            fallback = dispatch.expected_id(
                                mode=mode,
                                hdim=256,
                                pipeline="qr_tdm",
                                bm0=64,
                                mask="s_no",
                                lse="f",
                                spad="t" if group else "f",
                                dpad="t",
                                sink="f",
                            )
                            self.assertEqual(dispatch.run(group=group), [fallback])
                        else:
                            self.assertEqual(
                                dispatch.run(group=group),
                                [self.candidate_id(dispatch, mode)],
                            )
                    if receipt in (100, 200):
                        self.assertEqual(dispatch.run(group=int(receipt == 100)), [-1])

    def test_compiled_negative_control_detects_incorrect_short_q_gate(self):
        def mutate(api):
            self.assertIn("a.max_seqlen_q >= 128", api)
            return api.replace("a.max_seqlen_q >= 128", "a.max_seqlen_q >= 0")

        dispatch, _ = self.compile(mutate=mutate)
        actual = dispatch.run(seqlen_q=127)
        self.assertNotEqual(actual, [self.fallback_id(dispatch, "batch")])
        self.assertEqual(actual, [self.candidate_id(dispatch, "batch")])


_SUPPORTED_DISPATCH_FILTER = (
    "fmha_fwd_d*_bf16_*_nlogits*_nbias*_*lse*_ndropout*_nskip*_nqscale*_ntrload*_nsink*"
)


class TestGfx125V128Codegen(unittest.TestCase):
    @staticmethod
    def dimension_cases():
        cmake = (_FMHA_EXAMPLE_DIR / "CMakeLists.txt").read_text()
        default_dims = re.search(
            r"if\(FMHA_FWD_HAS_GFX125\)\s+set\(FMHA_FWD_OPTDIM\s+([0-9,]+)\)",
            cmake,
        )
        assert default_dims is not None
        return [
            [128],
            [192],
            [256],
            [128, 192],
            [-1],
            [int(value) for value in default_dims.group(1).split(",")],
        ]

    def test_sched_respects_requested_dimensions(self):
        for dimensions, expected in (
            ([-1], {64, 128, 192}),
            ([128], {128}),
            ([192], {192}),
            ([256], set()),
            ([128, 192], {128, 192}),
            ([32, 64, 80, 128, 256], {64, 128}),
            ([], set()),
            ([64], {64}),
        ):
            with self.subTest(dimensions=dimensions):
                _, kernels = get_fwd_blobs(
                    ["gfx1250"], "*_qr_tdm_sched_*", 100, dimensions, "simplified"
                )
                actual = {kernel.F_hdim for kernel in kernels}
                self.assertEqual(actual, expected)

    def test_dimension_filter_preserves_full_inventory(self):
        for targets in (["gfx1250"], ["gfx942", "gfx950", "gfx1250"]):
            for receipt in (0, 100, 200, 600):
                _, full = get_fwd_blobs(targets, None, receipt, [-1], "simplified")
                self.assertEqual(len(full), len({k.filename for k in full}))
                for dimensions in self.dimension_cases():
                    with self.subTest(
                        targets=targets, receipt=receipt, dims=dimensions
                    ):
                        _, kernels = get_fwd_blobs(
                            targets, None, receipt, dimensions, "simplified"
                        )
                        expected = [
                            k.filename
                            for k in full
                            if dimensions == [-1] or k.F_hdim in dimensions
                        ]
                        self.assertEqual([k.filename for k in kernels], expected)

    def test_sched_filter_emits_only_matching_kernels(self):
        for dtype in ("fp16", "bf16"):
            for receipt, count in ((0, 8), (100, 4), (200, 4), (600, 8)):
                for dimensions in self.dimension_cases():
                    with self.subTest(dtype=dtype, receipt=receipt, dims=dimensions):
                        self.check_sched_filter(dtype, receipt, count, dimensions)

    def check_sched_filter(self, dtype, receipt, count, dimensions):
        _, kernels = get_fwd_blobs(
            ["gfx1250"], f"*{dtype}*_qr_tdm_sched_*", receipt, dimensions, "simplified"
        )
        selected_dims = (
            {64, 128, 192}
            if dimensions == [-1]
            else {64, 128, 192}.intersection(dimensions)
        )
        self.assertEqual({k.F_hdim for k in kernels}, selected_dims)
        self.assertEqual(
            len(kernels),
            sum(
                (
                    {0: 12, 100: 8, 200: 4, 600: 12}[receipt]
                    if dim == 64
                    else count * (2 if dim == 192 else 1)
                )
                for dim in selected_dims
            ),
        )
        self.assertTrue(
            all(
                kernel.F_pipeline.tag == "qr_tdm_sched"
                and fnmatch(kernel.name, f"*{dtype}*_qr_tdm_sched_*")
                for kernel in kernels
            )
        )

    def test_dimension_generic_and_unsupported_filters_are_deterministic(self):
        for targets in (["gfx1250"], ["gfx942", "gfx950", "gfx1250"]):
            for dimensions in self.dimension_cases():
                for pattern in (
                    "*bf16*_qr_vr_*",
                    "*not_a_kernel*",
                ):
                    with self.subTest(
                        targets=targets, dims=dimensions, pattern=pattern
                    ):
                        pool, kernels = get_fwd_blobs(
                            targets, pattern, 600, dimensions, "simplified"
                        )
                        again_pool, again = get_fwd_blobs(
                            targets, pattern, 600, dimensions, "simplified"
                        )
                        self.assertFalse(
                            any(k.F_pipeline.tag == "qr_tdm_sched" for k in kernels)
                        )
                        self.assertEqual(
                            [k.filename for k in kernels], [k.filename for k in again]
                        )
                        self.assertEqual(pool.render("test"), again_pool.render("test"))
                        if pattern != "*bf16*_qr_vr_*":
                            self.assertFalse(kernels)

    def test_d64_geometry_and_padding(self):
        for dtype in ("fp16", "bf16"):
            pool, kernels = get_fwd_blobs(
                ["gfx1250"], f"*d64_{dtype}*_qr_tdm_sched_*", 600, [64], "simplified"
            )
            self.assertEqual(len(kernels), 12)
            for kernel in kernels:
                self.assertEqual(kernel.F_tile.F_bm0, 128)
                self.assertEqual(kernel.F_tile.F_bn0, 64)
                self.assertEqual(kernel.F_tile.F_bn1, 64)
                self.assertEqual(kernel.F_tile.F_occupancy, 3)
                self.assertEqual(kernel.F_pipeline.F_dpad, "f")
                self.assertEqual(kernel.F_pipeline.F_dvpad, "f")
                trait = kernel.api_trait()
                if kernel.F_mode == "batch" and trait.skpad == "f":
                    self.assertIn("a.seqlen_k % 64 == 0", trait.skcheck)
                    self.assertIn("a.seqlen_k != 0", trait.skcheck)
                    self.assertIn("a.cu_seqlen_k_ptr == nullptr", trait.skcheck)
                else:
                    self.assertEqual(trait.spad, "t")
                    self.assertEqual(trait.skpad, "t")
            self.assertIn("a.hdim_v == 64", pool.render("test"))

    def test_fp16_bf16_d128_sched_emission(self):
        for dtype in ("fp16", "bf16"):
            for receipt, count in ((0, 8), (100, 4), (200, 4), (600, 8)):
                with self.subTest(dtype=dtype, receipt=receipt):
                    self.check_d128_emission(dtype, receipt, count)

    def check_d128_emission(self, dtype, receipt, count):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"], f"*d128_{dtype}*_qr_tdm_sched_*", receipt, [128], "simplified"
        )
        family = [k for k in kernels if k.F_pipeline.tag == "qr_tdm_sched"]
        self.assertEqual(len(family), count)
        self.assertEqual(len(family), len(kernels))
        self.assertEqual(len({k.filename for k in kernels}), len(kernels))
        for kernel in family:
            self.assertEqual(kernel.F_hdim, 128)
            self.assertEqual(kernel.F_tile.F_bn0, 128)
            self.assertEqual(kernel.F_tile.F_bn1, 128)
            self.assertEqual(kernel.F_tile.F_bk0max, 128)
            self.assertEqual(kernel.F_tile.F_occupancy, 1)
            self.assertIn("BlockFmhaPipelineQRKSVSTdmSched", kernel.render())
            self.assertIn("QRKSVS_TDM_SCHED", kernel.render())
        api = pool.render("test")
        self.assertIn("a.hdim_q == 128", api)
        self.assertIn("a.max_seqlen_q >= 128", api)

    def test_family_padding_matches_validated_specializations(self):
        kernels = []
        for dtype in ("fp16", "bf16"):
            _, selected = get_fwd_blobs(
                ["gfx1250"], f"*{dtype}*", 600, [128, 192], "simplified"
            )
            kernels.extend(selected)
        family = [k for k in kernels if k.F_pipeline.tag == "qr_tdm_sched"]
        self.assertEqual(len(family), 48)
        rule = KernelComponentFactoryGfx125.get_rules()[-1]
        for kernel in family:
            with self.subTest(shape=kernel.F_hdim, name=kernel.filename):
                expected = (
                    "t" if kernel.F_hdim == 128 or kernel.F_mode == "group" else "f"
                )
                self.assertEqual(kernel.F_pipeline.F_spad, expected)
                self.assertEqual(kernel.F_pipeline.F_skpad, expected)
                problem = ProblemContext(
                    dtype=kernel.F_dtype,
                    mode=kernel.F_mode,
                    hdim=kernel.F_hdim,
                    hdim_v=128,
                )
                for qpad in ("f", "t"):
                    for kpad in ("f", "t"):
                        context = KernelContext(
                            tile=kernel.F_tile,
                            pipeline=replace(
                                kernel.F_pipeline, F_spad=qpad, F_skpad=kpad
                            ),
                            mask_impl="simplified",
                        )
                        self.assertEqual(
                            rule(problem, context),
                            qpad == expected and kpad == expected,
                        )

    def test_new_tag_is_not_emitted_for_other_shapes_or_architectures(self):
        _, kernels = get_fwd_blobs(
            ["gfx942", "gfx950"], "*_qr_tdm_sched_*", 600, [-1], "simplified"
        )
        self.assertFalse(kernels)
        _, kernels = get_fwd_blobs(
            ["gfx1250"], "*fp32*_qr_tdm_sched_*", 600, [-1], "simplified"
        )
        self.assertFalse(kernels)
        for target in ("gfx1250", "gfx1251", "gfx1252"):
            _, kernels = get_fwd_blobs(
                [target], "*_qr_tdm_sched_*", 600, [-1], "simplified"
            )
            self.assertTrue(
                any(kernel.F_pipeline.tag == "qr_tdm_sched" for kernel in kernels)
            )

    def test_family_and_legacy_entries_use_the_same_arch_check(self):
        _, kernels = get_fwd_blobs(
            ["gfx1250"], "*bf16*_qr_tdm*", 100, [128, 192], "simplified"
        )
        family = [k for k in kernels if k.F_pipeline.tag == "qr_tdm_sched"]
        legacy = [k for k in kernels if k.F_pipeline.tag != "qr_tdm_sched"]
        self.assertEqual(
            {k.F_pipeline.tag for k in family},
            {"qr_tdm_sched"},
        )
        self.assertTrue(legacy)
        for kernel in family + legacy:
            with self.subTest(kernel=kernel.name):
                source = kernel.render()
                self.assertIn("defined(__gfx125__)", source)
                self.assertNotIn("ck_tile::get_device_name()", source)

    def test_family_tile_and_pipeline_compatibility_is_symmetric(self):
        _, kernels = get_fwd_blobs(["gfx1250"], "*d128_bf16*", 100, [128], "simplified")
        candidate = next(k for k in kernels if k.F_pipeline.tag == "qr_tdm_sched")
        legacy = next(k for k in kernels if k.F_pipeline.tag == "qr_tdm")
        rule = KernelComponentFactoryGfx125.get_rules()[-1]
        problem = ProblemContext(dtype="bf16", mode="batch", hdim=128, hdim_v=128)
        context = KernelContext(
            tile=candidate.F_tile, pipeline=candidate.F_pipeline, mask_impl="simplified"
        )
        self.assertTrue(rule(problem, context))
        self.assertTrue(rule(replace(problem, dtype="fp16"), context))
        self.assertFalse(rule(problem, replace(context, tile=legacy.F_tile)))
        self.assertFalse(rule(problem, replace(context, pipeline=legacy.F_pipeline)))
        for changes in (dict(dtype="fp32"), dict(hdim=192), dict(hdim_v=192)):
            self.assertFalse(rule(replace(problem, **changes), context))

    def test_sched_tile_geometry_identity_and_inventory_isolation(self):
        factory = KernelComponentFactoryGfx125
        for dtype in ("fp16", "bf16"):
            inventory = factory.get_hdim_tile_size_dict(dtype)
            for hdim in (128, 192):
                tiles = [
                    tile
                    for tile in inventory[(hdim, 128)]
                    if factory.is_tile_supported_by_qr_tdm_sched(tile, hdim)
                ]
                self.assertEqual(len(tiles), 1 if hdim == 128 else 2)
                for tile in tiles:
                    # Runtime selection does not change tile compatibility.
                    self.assertTrue(
                        factory.is_tile_supported_by_qr_tdm_sched(
                            replace(tile, F_constraint=CppConstraint("false")), hdim
                        )
                    )
                    # Every geometry/occupancy change must reject this pairing.
                    for descriptor in fields(tile):
                        value = getattr(tile, descriptor.name)
                        if isinstance(value, int):
                            with self.subTest(
                                dtype=dtype, tile=tile.name, field=descriptor.name
                            ):
                                self.assertFalse(
                                    factory.is_tile_supported_by_qr_tdm_sched(
                                        replace(tile, **{descriptor.name: value + 1}),
                                        hdim,
                                    )
                                )
                # Callers can tune their inventory without changing later calls.
                tile = tiles[0]
                original_name = tile.name
                tile.F_bm0 += 1
                tile.F_constraint.bool_expr = "false"
                fresh = factory.get_hdim_tile_size_dict(dtype)[(hdim, 128)]
                restored = next(t for t in fresh if t.name == original_name)
                self.assertNotEqual(str(restored.F_constraint), "false")

    @unittest.skipUnless(
        _HIP_HOST_CXX.is_file() and (_ROCM_ROOT / "include").is_dir(),
        "ROCm clang and HIP headers are required for generated TU host syntax",
    )
    def test_sched_generated_occupancy_contract(self):
        _, kernels = get_fwd_blobs(["gfx1250"], "*", 200, [-1], "simplified")
        candidates = {}
        diagnostic = "qr_tdm_sched codegen occupancy must match policy tuning"
        for kernel in kernels:
            if kernel.F_pipeline.tag == "qr_tdm_sched":
                key = (
                    kernel.F_dtype,
                    kernel.F_hdim,
                    kernel.F_tile.F_bn0,
                    kernel.F_tile.F_bn1,
                )
                candidates.setdefault(key, kernel)
            else:
                self.assertNotIn(diagnostic, kernel.render())
        self.assertEqual(
            set(candidates),
            {
                (dtype, head, n, v)
                for dtype in ("bf16", "fp16")
                for head, n, v in (
                    (64, 64, 64),
                    (128, 128, 128),
                    (192, 64, 128),
                    (192, 128, 128),
                )
            },
        )
        with tempfile.TemporaryDirectory(
            prefix="fmha-occupancy-contract-"
        ) as directory:
            root = Path(directory)
            for key, kernel in candidates.items():
                for changed in (False, True):
                    with self.subTest(config=key, changed_occupancy=changed):
                        rendered = (
                            replace(
                                kernel,
                                F_tile=replace(
                                    kernel.F_tile,
                                    F_occupancy=kernel.F_tile.F_occupancy + 1,
                                ),
                            )
                            if changed
                            else kernel
                        ).render()
                        self.assertIn(diagnostic, rendered)
                        source = root / "generated.cpp"
                        source.write_text(rendered)
                        result = subprocess.run(
                            [
                                str(_HIP_HOST_CXX),
                                "-x",
                                "hip",
                                "--offload-host-only",
                                "-nogpuinc",
                                "-nogpulib",
                                "-include",
                                "__clang_hip_runtime_wrapper.h",
                                "-std=c++20",
                                "-DUSE_NEW_UNIFIED_FRAMEWORK=0",
                                "-DCK_TILE_FMHA_FWD_FAST_EXP2=1",
                                "-I",
                                str(_CK_ROOT / "include"),
                                "-I",
                                str(_CK_ROOT),
                                "-I",
                                str(_CK_ROOT / "library" / "include"),
                                "-I",
                                str(_FMHA_EXAMPLE_DIR),
                                "-I",
                                str(_ROCM_ROOT / "include"),
                                "-fsyntax-only",
                                str(source),
                            ],
                            capture_output=True,
                            text=True,
                            timeout=60,
                        )
                        if changed:
                            self.assertNotEqual(result.returncode, 0)
                            self.assertIn(diagnostic, result.stderr)
                        else:
                            self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("cmake"), "cmake is required")
    def test_cmake_family_esm2_exact_scope(self):
        helper = (_FMHA_EXAMPLE_DIR / "cmake") / "fmha_fwd_source_flags.cmake"
        d128 = "fmha_fwd_d128_bf16_batch_b128_qr_tdm_sched_vr_nmask_nlse_gfx125.cpp"
        d192 = "fmha_fwd_d192_bf16_batch_b128_qr_tdm_sched_vr_nmask_nlse_gfx125.cpp"
        sources = [
            d128,
            d128.replace("batch", "group"),
            d192,
            d192.replace("nmask", "mask"),
            d128.replace("qr_tdm_sched", "qr_tdm"),
            d128.replace("bf16", "fp16"),
            d128.replace("d128", "d256"),
            d128.replace("gfx125", "gfx950"),
            "prefix_" + d128,
            d128 + ".backup",
            "fmha_fwd_api.cpp",
            "mha_fwd.cu",
            # All supported scheduled head dimensions use expert scheduling mode.
            d128.replace("d128", "d64"),
            d128.replace("d128", "d64").replace("bf16", "fp16"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "flags.cmake"
            script.write_text(
                f'include("{helper}")\n'
                + "\n".join(
                    f'ck_tile_fmha_fwd_get_source_compile_options("{name}" flags)\n'
                    f'message("{i}:${{flags}}")'
                    for i, name in enumerate(sources)
                )
            )
            process = subprocess.run(
                ["cmake", "-P", str(script)],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            lines = process.stderr.strip().splitlines()
            self.assertEqual(len(lines), len(sources))
            for i, line in enumerate(lines):
                self.assertTrue(line.startswith(f"{i}:"))
                self.assertEqual(
                    "-amdgpu-expert-scheduling-mode" in line,
                    i in (0, 1, 2, 3, 5, 12, 13),
                )
                self.assertNotIn("-wwm-regalloc=fast", line)


@unittest.skipUnless(_HOST_CXX, "a C++17 host compiler is required")
class TestCompiledGfx125V128Dispatch(unittest.TestCase):
    def test_d64_sequence_padding_and_short_q_fallback(self):
        for dtype in ("fp16", "bf16"):
            pool, kernels = get_fwd_blobs(
                ["gfx1250"],
                f"fmha_fwd_d64_{dtype}_*_nlogits*_nbias*_*lse*_ndropout*_nskip*_nqscale*_ntrload*_nsink*",
                600,
                [64],
                "simplified",
            )
            dispatch = _CompiledDispatcher(self, pool, kernels)
            for sq, sk, has_cu_sk, group in (
                (127, 128, 0, 0),
                (128, 128, 0, 0),
                (129, 129, 0, 0),
                (255, 255, 0, 0),
                (128, 0, 0, 0),
                (128, 128, 1, 0),
                (255, 255, 0, 1),
            ):
                candidate = dispatch.expected_id(
                    dtype=dtype,
                    hdim=64,
                    mode="group" if group else "batch",
                    pipeline="qr_tdm" if sq < 128 else "qr_tdm_sched",
                    bm0=64 if sq < 128 else 128,
                    mask="s_no",
                    lse="t",
                    dpad="f",
                    dvpad="f",
                    sink="f",
                    spad="t" if group or has_cu_sk or sk == 0 or sk % 64 else "f",
                )
                self.assertEqual(
                    dispatch.run(
                        dtype=dtype,
                        hdim_q=64,
                        hdim_v=64,
                        seqlen_q=sq,
                        seqlen_k=sk,
                        has_cu_sk=has_cu_sk,
                        group=group,
                        lse=1,
                    ),
                    [candidate],
                )

    def test_batch_sched_only_dispatch(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"], "*d128_bf16*_qr_tdm_sched_*", 600, [128], "simplified"
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)
        for mask, lse in ((0, 0), (0, 1), (1, 0), (1, 1)):
            features = dict(
                mode="batch",
                hdim=128,
                mask="s_mask" if mask else "s_no",
                lse="t" if lse else "f",
                sink="f",
            )
            candidate = dispatch.expected_id(pipeline="qr_tdm_sched", **features)
            args = dict(hdim_q=128, group=0, mask=mask, lse=lse)
            self.assertEqual(dispatch.run(**args), [candidate])

    def test_group_max_q_is_independent_of_member_length_and_head_ratio(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"],
            _SUPPORTED_DISPATCH_FILTER,
            600,
            [128, 192, 256],
            "simplified",
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)
        for dim in (128, 192):
            for mask, lse in ((0, 0), (0, 1), (1, 0), (1, 1)):
                features = dict(
                    mode="group",
                    mask="s_mask" if mask else "s_no",
                    lse="t" if lse else "f",
                    sink="f",
                )
                candidate = dispatch.expected_id(
                    hdim=dim,
                    pipeline="qr_tdm_sched",
                    bn0=64 if dim == 192 else 128,
                    **features,
                )
                for heads, kv_heads in ((1, 1), (8, 8), (8, 2), (8, 1)):
                    for member_q, max_q in (
                        (0, 128),
                        (1, 128),
                        (127, 128),
                        (128, 128),
                        (0, 127),
                        (127, 127),
                    ):
                        fallback = dispatch.expected_id(
                            hdim=dim,
                            bm0=64 if max_q < 128 else 128,
                            pipeline="qr_tdm",
                            dpad="f" if dim == 128 else "t",
                            spad="t",
                            **features,
                        )
                        args = dict(
                            hdim_q=dim,
                            group=1,
                            mask=mask,
                            lse=lse,
                            seqlen_q=member_q,
                            max_seqlen_q=max_q,
                            nhead_q=heads,
                            nhead_k=kv_heads,
                        )
                        expected = candidate if max_q >= 128 else fallback
                        self.assertEqual(dispatch.run(**args), [expected])

    def test_default_dispatch_and_rejected_features(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"],
            _SUPPORTED_DISPATCH_FILTER,
            600,
            [128, 192, 256],
            "simplified",
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)
        family_ids = {
            value
            for kernel, value in dispatch.kernel_ids
            if kernel.F_pipeline.tag == "qr_tdm_sched"
        }
        fallback_ids = {
            value
            for kernel, value in dispatch.kernel_ids
            if kernel.F_pipeline.tag == "qr_tdm"
        }
        for dim in (128, 192):
            for mode, group in (("batch", 0), ("group", 1)):
                for mask in (0, 1, 2, 3):
                    for lse in (0, 1):
                        features = dict(
                            mode=mode,
                            mask="s_mask" if mask else "s_no",
                            lse="t" if lse else "f",
                            sink="f",
                        )
                        candidate = dispatch.expected_id(
                            hdim=dim,
                            pipeline="qr_tdm_sched",
                            bn0=64 if dim == 192 else 128,
                            **features,
                        )
                        args = dict(hdim_q=dim, group=group, mask=mask, lse=lse)
                        self.assertEqual(dispatch.run(**args), [candidate])
                        sink_fallback = dispatch.run(**args, has_sink_ptr=1)[0]
                        self.assertGreater(sink_fallback, 0)
                        self.assertIn(sink_fallback, fallback_ids)
                        for changes in (
                            dict(seqlen_q=127),
                            dict(arg_q=160),
                            dict(arg_v=192),
                            dict(hdim_q=160),
                            dict(hdim_v=192),
                            dict(hdim_v=127),
                            dict(device="gfx942"),
                        ):
                            rejected_args = dict(args, **changes)
                            self.assertNotIn(
                                dispatch.run(**rejected_args)[0], family_ids
                            )
                        for device in ("gfx1251", "gfx1252"):
                            self.assertEqual(
                                dispatch.run(**args, device=device), [candidate]
                            )
                        for rejected in (
                            dict(logits=1),
                            dict(bias=1),
                            dict(dropout=1),
                            dict(row=0),
                            dict(qscale=1),
                            dict(skip=1),
                            dict(sink=1),
                            dict(dtype="fp16"),
                        ):
                            self.assertNotIn(
                                dispatch.run(**args, **rejected)[0], family_ids
                            )

    def test_d128_filtered_priority_and_fallback(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"],
            _SUPPORTED_DISPATCH_FILTER,
            600,
            [128],
            "simplified",
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)
        for mode, group in (("batch", 0), ("group", 1)):
            for mask in (0, 1, 2, 3):
                for lse in (0, 1):
                    features = dict(
                        mode=mode,
                        hdim=128,
                        mask="s_mask" if mask else "s_no",
                        lse="t" if lse else "f",
                        sink="f",
                    )
                    candidate = dispatch.expected_id(
                        pipeline="qr_tdm_sched", **features
                    )
                    for size in (1, 127, 128, 129, 2047, 2048, 32768):
                        fallback = dispatch.expected_id(
                            pipeline="qr_tdm",
                            spad="t" if group else "f",
                            bm0=64 if size < 128 else 128,
                            dpad="f",
                            **features,
                        )
                        args = dict(
                            group=group,
                            hdim_q=128,
                            hdim_v=128,
                            seqlen_q=size,
                            mask=mask,
                            lse=lse,
                        )
                        self.assertEqual(
                            dispatch.run(**args),
                            [candidate if size >= 128 else fallback],
                        )
            candidate = dispatch.expected_id(
                mode=mode,
                hdim=128,
                pipeline="qr_tdm_sched",
                mask="s_no",
                lse="f",
                sink="f",
            )
            for changes in (
                dict(arg_q=192),
                dict(arg_v=192),
                dict(hdim_q=192),
                dict(hdim_v=192),
                dict(seqlen_q=127),
            ):
                args = dict(group=group, hdim_q=128, hdim_v=128)
                args.update(changes)
                self.assertNotEqual(dispatch.run(**args), [candidate])
            for changes in (
                dict(logits=1),
                dict(bias=1),
                dict(qscale=1),
                dict(dropout=1),
                dict(skip=1),
                dict(sink=1),
                dict(row=0),
                dict(dtype="fp16"),
            ):
                self.assertEqual(dispatch.run(group=group, hdim_q=128, **changes), [-1])

    def test_missing_d128_candidate_does_not_disable_generic(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"],
            _SUPPORTED_DISPATCH_FILTER,
            600,
            [128],
            "simplified",
        )
        dispatch = _CompiledDispatcher(
            self,
            pool,
            kernels,
            filter_fn=lambda trait: trait.pipeline_tag != "qr_tdm_sched",
        )
        for mode, group in (("batch", 0), ("group", 1)):
            fallback = dispatch.expected_id(
                mode=mode,
                hdim=128,
                bm0=128,
                pipeline="qr_tdm",
                spad="t" if group else "f",
                mask="s_no",
                lse="f",
                sink="f",
                dpad="f",
            )
            self.assertEqual(dispatch.run(group=group, hdim_q=128), [fallback])


if __name__ == "__main__":
    unittest.main()
