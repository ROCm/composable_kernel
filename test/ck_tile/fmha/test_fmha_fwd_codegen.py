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
    FmhaFwdTileSize,
    KernelComponentFactoryGfx125,
    KernelContext,
    ProblemContext,
    _validate_gfx125_tile_dispatch_order,
    get_fwd_blobs,
    write_fwd_api,
)

_SUPPORTED_FEATURE_FILTER = (
    "fmha_fwd_d*_bf16_*_nlogits*_nbias*_nmask*_nlse*_ndropout*_nskip*_nqscale*"
)
_SCHED_PIPELINE = "qr_tdm_sched"
# qr_tdm_pingpong serves every d192 query; qr_tdm_sched emits d128 only.
_D192_PIPELINE = "qr_tdm_pingpong"
_HOST_CXX = shutil.which("c++")
_ROCM_ROOT = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
_HIP_HOST_CXX = _ROCM_ROOT / "llvm" / "bin" / "clang++"
_RENDERED_CALL = re.compile(
    r"using trait_ = (fmha_fwd_traits_<[^;\n]+>);\s*"
    r"return fmha_fwd_<trait_, ([^>]+)>\(s, a\);"
)
_D192_ESM2_FILENAME = re.compile(
    r"^fmha_fwd_d192_bf16_.*_qr_tdm_pingpong_.*_gfx125\.cpp$"
)
_D192_BATCH_NMASK_FILENAME = (
    "fmha_fwd_d192_bf16_batch_b128x64x32x128x32x192_"
    "r4x1x1_r4x1x1_w16x16x32_w16x16x32_qr_tdm_pingpong_"
    "vr_npad_nlogits_nbias_nmask_nlse_ndropout_nskip_nqscale_"
    "ntrload_sp64_vbs128_nsink_gfx125.cpp"
)
_D192_BATCH_MASK_FILENAME = _D192_BATCH_NMASK_FILENAME.replace("_nmask_", "_mask_")
_D192_GROUP_NMASK_FILENAME = _D192_BATCH_NMASK_FILENAME.replace(
    "_batch_", "_group_"
).replace("_npad_", "_pssk_")
# The 16-row tile runs neither part of the LDS read schedule, so its name has none.
_D192_BATCH_M16_FILENAME = (
    _D192_BATCH_NMASK_FILENAME.replace("b128x64x32", "b16x64x32")
    .replace("r4x1x1_r4x1x1", "r1x4x1_r1x4x1")
    .replace("_sp64_vbs128", "")
)
# The d192 qr_tdm_sched selector (exact head dims) that no generated API carries now.
_D192_SCHED_SELECTOR = "a.hdim_q == 192 && a.hdim_v == 128"
# The features qr_tdm_sched and qr_tdm_pingpong took before they also took the logits
# soft cap, biases, dropout, sink tokens and (pingpong) col-major V.
_BASE_FEATURES = dict(
    logits="f", bias="no", dropout="f", sink="f", vlayout="row", qscale="no", skip="f"
)


def _generate(receipt, optdim_list=None):
    return get_fwd_blobs(
        targets=["gfx1250"],
        kernel_filter=_SUPPORTED_FEATURE_FILTER,
        receipt=receipt,
        optdim_list=[192, 256] if optdim_list is None else optdim_list,
        mask_impl="simplified",
    )


def _is_sched_kernel(kernel):
    return kernel.F_pipeline.tag == _SCHED_PIPELINE


def _has_base_features(kernel):
    return all(
        getattr(kernel.F_pipeline, "F_" + name) == value
        for name, value in _BASE_FEATURES.items()
    )


# These stubs provide types and unique launch IDs only. All selection predicates,
# branch ordering, and v3/v2 fallback come from the renderer.
def _softmax(kernels):
    """The softmax kernels: GDPA's (a non-softmax activation) share their other traits."""
    return [k for k in kernels if k.F_pipeline.F_activation == "softmax"]


_HOST_HIP_STUB = """#pragma once
#include <cstdlib>
constexpr int hipSuccess = 0;
struct hipDeviceProp_t { unsigned multiProcessorCount = 256; };
inline int hipGetDevice(int* device) { *device = 0; return hipSuccess; }
inline int hipGetDeviceCount(int* count) { *count = 1; return hipSuccess; }
inline int hipGetDeviceProperties(hipDeviceProp_t* props, int) {
    const char* value = std::getenv("FMHA_TEST_NUM_CUS");
    if(value != nullptr) props->multiProcessorCount = std::atoi(value);
    return hipSuccess;
}
"""

_HOST_FMHA_STUB = """#pragma once
#include <cstdint>
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
enum class BlockFmhaPipelineEnum { QRKSVS, QRKSVS_TDM, QRKSVS_TDM_SCHED, QRKSVS_TDM_PINGPONG,
                                  QRKSVS_TDM_KVSHARED };
enum class BlockAttentionBiasEnum { NO_BIAS, ELEMENTWISE_BIAS, ALIBI };
enum class BlockAttentionQuantScaleEnum { NO_SCALE, PERTENSOR, BLOCKSCALE, KV_BLOCKSCALE, MX };
enum class BlockAttentionActivationEnum { SOFTMAX, FAST_GELU };
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
          typename, auto, bool, bool, auto, bool, bool, bool, bool, bool, bool, bool, int = -1,
          ck_tile::BlockAttentionActivationEnum = ck_tile::BlockAttentionActivationEnum::SOFTMAX>
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
    ck_tile::BlockAttentionActivationEnum activation =
        ck_tile::BlockAttentionActivationEnum::SOFTMAX;
};
struct fmha_fwd_args {
    int hdim_q = 192, hdim_v = 128, seqlen_q = 128, max_seqlen_q = 128, seqlen_k = 128;
    int stride_q = 192, nhead_stride_q = 24576;
    int window_size_left = -1, window_size_right = -1;
    int batch = 2, nhead_q = 8, nhead_k = 2;
    const void* sink_ptr = nullptr;
    const void* seqlen_k_ptr = nullptr;
    const void* cu_seqlen_k_ptr = nullptr;
    const void* k_ptr = nullptr;
    const void* v_ptr = nullptr;
    int stride_k = 256, stride_v = 256, nhead_stride_k = 128, nhead_stride_v = 128;
    int batch_stride_k = 32768, batch_stride_v = 32768;
    bool broadcast_q = false;
    bool lse_log2 = false;
    const void* o_ptr = nullptr;
    int stride_o = 1024, nhead_stride_o = 128, batch_stride_o = 262144;
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
    int mask, bias, scale, arg_q, arg_v, max_q, has_cu_sk, has_sk_ptr, has_sink_ptr, o_offset,
        kv_shared, activation, broadcast_q, lse_log2;
    if(!(std::cin >> t.hdim_q >> t.hdim_v >> a.seqlen_q >> a.seqlen_k
         >> t.is_group_mode >> t.is_v_rowmajor >> t.has_logits_soft_cap >> mask >> bias
         >> t.has_lse >> t.has_dropout >> scale >> t.skip_min_seqlen_q >> t.has_sink
         >> t.data_type >> arg_q >> arg_v >> max_q >> a.batch >> a.nhead_q
         >> a.nhead_k >> a.window_size_left >> a.window_size_right
         >> has_cu_sk >> has_sk_ptr >> has_sink_ptr >> o_offset >> a.stride_o
         >> a.nhead_stride_o >> a.batch_stride_o >> kv_shared >> activation
         >> broadcast_q >> lse_log2)) return 2;
    a.broadcast_q = broadcast_q;
    a.lse_log2 = lse_log2;
    t.activation = static_cast<ck_tile::BlockAttentionActivationEnum>(activation);
    t.mask_type = static_cast<mask_enum>(mask);
    t.bias_type = static_cast<bias_enum>(bias);
    t.qscale_type = static_cast<quant_scale_enum>(scale);
    a.hdim_q = arg_q < 0 ? t.hdim_q : arg_q;
    a.hdim_v = arg_v < 0 ? t.hdim_v : arg_v;
    a.max_seqlen_q = max_q < 0 ? a.seqlen_q : max_q;
    a.cu_seqlen_k_ptr = has_cu_sk ? &a : nullptr;
    a.seqlen_k_ptr = has_sk_ptr ? &a : nullptr;
    a.sink_ptr = has_sink_ptr ? &a : nullptr;
    // K and V one buffer, or two
    a.k_ptr = &a;
    a.v_ptr = kv_shared ? a.k_ptr : &t;
    a.o_ptr = reinterpret_cast<const void*>(std::uintptr_t{4096} + o_offset);
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
                # As in the CK build: a dispatch condition that compares ck_tile::index_t
                # with the unsigned num_cus fails there.
                "-Wsign-compare",
                "-Werror",
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

        # Unless a field says otherwise, expect the base-feature variant.
        fields = {**_BASE_FEATURES, **fields}
        fields.setdefault(
            "activation", "softmax"
        )  # GDPA instances share the other traits
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
            o_offset=0,
            stride_o=1024,
            nhead_stride_o=128,
            batch_stride_o=262144,
            kv_shared=0,
            activation=0,  # BlockAttentionActivationEnum: 0 softmax, 1 fast_gelu
            broadcast_q=0,
            lse_log2=0,
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
        # qr_tdm_pingpong instances storing the LSE use expert scheduling too; plain qr
        # (d256_source) keeps the default.
        d192_lse_source = _D192_BATCH_NMASK_FILENAME.replace("_nlse_", "_lse_")
        sources = (
            _D192_BATCH_NMASK_FILENAME,
            _D192_BATCH_M16_FILENAME,
            _D192_BATCH_NMASK_FILENAME.replace("_bf16_", "_fp16_"),
            d192_lse_source,
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

    def test_receipts_emit_d192_pingpong_and_fallback_tiles(self):
        for receipt in (100, 200, 600):
            with self.subTest(receipt=receipt):
                _, kernels = _generate(receipt)
                # qr_tdm_sched emits no d192 instances any more.
                self.assertFalse(any(_is_sched_kernel(k) for k in kernels))
                family = [
                    kernel
                    for kernel in kernels
                    if kernel.F_pipeline.tag == _D192_PIPELINE and kernel.F_hdim == 192
                ]
                candidates = [k for k in family if _has_base_features(k)]

                self.assertTrue(candidates)
                # The kernel filter leaves the sink tokens as the only extra feature:
                # every base instance has a sink twin.
                self.assertEqual(
                    Counter(k.F_pipeline.F_sink for k in family)["t"], len(candidates)
                )
                self.assertEqual(len(family), 2 * len(candidates))
                for kernel in family:
                    self.assertEqual(kernel.F_arch.name, "gfx125")
                    self.assertEqual(kernel.F_dtype, "bf16")
                    self.assertIn(kernel.F_tile.F_bm0, (16, 64, 128))
                    self.assertEqual(kernel.F_tile.F_bn0, 64)
                    self.assertEqual(kernel.F_tile.F_bn1, 128)
                    self.assertEqual(kernel.F_tile.F_bk0max, 192)
                    if kernel.F_mode == "group":
                        self.assertEqual(kernel.F_pipeline.F_spad, "t")
                    self.assertEqual(kernel.F_pipeline.F_vlayout, "row")
                    self.assertEqual(kernel.F_pipeline.F_logits, "f")
                    self.assertEqual(kernel.F_pipeline.F_bias, "no")
                    self.assertEqual(kernel.F_pipeline.F_dropout, "f")
                    self.assertEqual(kernel.F_pipeline.F_qscale, "no")
                    self.assertEqual(kernel.F_pipeline.F_skip, "f")
                    self.assertRegex(kernel.filename, _D192_ESM2_FILENAME)

                for kernel in kernels:
                    if kernel not in family:
                        self.assertNotRegex(kernel.filename, _D192_ESM2_FILENAME)

                # qr_tdm_pingpong takes every d192 query, so no qr_tdm instance is
                # generated for it; the generic qr fallback stays.
                self.assertFalse(
                    any(
                        kernel.F_hdim == 192 and kernel.F_pipeline.tag == "qr_tdm"
                        for kernel in kernels
                    )
                )
                self.assertTrue(
                    any(
                        kernel.F_hdim == 192
                        and kernel.F_tile.F_bn1 == 128
                        and kernel.F_pipeline.tag == "qr"
                        for kernel in kernels
                    )
                )
                self.assertEqual(
                    {k.F_mode for k in candidates},
                    {100: {"batch"}, 200: {"group"}, 600: {"batch", "group"}}[receipt],
                )
                # Each mode has its 16-, 64- and 128-row tile.
                self.assertEqual(
                    Counter(k.F_tile.F_bm0 for k in candidates),
                    Counter({bm0: len(candidates) // 3 for bm0 in (16, 64, 128)}),
                )

    def test_tdm_unpadded_v_checks_the_v_row(self):
        # (192, 128) stages a 192-column Q/K row but 128-column V rows: the unpadded-V
        # kernels must take exactly hdim_v = 128, not a multiple of the Q/K row.
        _, kernels = _generate(100)
        tdm = [
            kernel
            for kernel in kernels
            if kernel.F_pipeline.tag in ("qr_tdm", "qr_tdm_pingpong")
            and kernel.F_hdim == 192
            and kernel.F_pipeline.F_dvpad == "f"
        ]
        self.assertTrue(tdm)
        for kernel in tdm:
            with self.subTest(kernel=kernel.filename):
                self.assertEqual(kernel.F_tile.F_bn1, 128)
                self.assertEqual(kernel.api_trait().dvcheck, "a.hdim_v == 128")

    def test_d128_retains_short_query_fallback_tiles(self):
        for receipt in (100, 200, 600):
            with self.subTest(receipt=receipt):
                _, kernels = _generate(receipt, [128])
                self.assertTrue(any(_is_sched_kernel(k) for k in kernels))
                fallback = [k for k in kernels if not _is_sched_kernel(k)]
                self.assertTrue(fallback)
                self.assertTrue(all(k.F_hdim == 128 for k in fallback))
                self.assertTrue(any(k.F_tile.F_bm0 < 128 for k in fallback))
                # qr_tdm_kvshared's one-warp tile takes 32 keys
                self.assertTrue(
                    all(
                        k.F_tile.F_bn0 == 64
                        for k in fallback
                        if k.F_pipeline.tag != "qr_tdm_kvshared"
                    )
                )
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

                # Neither the d192 qr_tdm_sched selector nor its N128 long-K one is
                # rendered any more.
                self.assertNotIn("QRKSVS_TDM_SCHED", api)
                self.assertNotIn(_D192_SCHED_SELECTOR, api)
                self.assertNotIn(
                    "a.seqlen_k >= 512LL * (t.is_group_mode ? a.batch : 1)", api
                )
                self.assertNotIn("get_num_blocks(128) <= num_cus", api)
                self.assertIn("QRKSVS_TDM_PINGPONG", api)
                d192 = api[api.index("t.hdim_q <= 192") : api.index("t.hdim_q <= 256")]
                # d192 has no qr_tdm entries; qr_tdm_pingpong's come before qr's.
                self.assertNotIn("BlockFmhaPipelineEnum::QRKSVS_TDM,", d192)
                self.assertLess(
                    d192.index("QRKSVS_TDM_PINGPONG"),
                    d192.index("BlockFmhaPipelineEnum::QRKSVS,"),
                )
                self.assertNotIn("is_gfx125_tdm_v128_enabled", api)

                candidate = next(
                    kernel
                    for kernel in kernels
                    if kernel.F_pipeline.tag == _D192_PIPELINE
                    and kernel.F_hdim == 192
                    and kernel.F_tile.F_bm0 == 128
                    and _has_base_features(kernel)
                )
                self.assertEqual((candidate.F_hdim, candidate.F_tile.F_bn1), (192, 128))
                source = candidate.render()
                self.assertIn("BlockFmhaPipelineQRKSVSTdmPingPong", source)
                # The 128-row tile runs both parts of d192's LDS read schedule.
                self.assertIn("QrTdmPingPongLdsReadSchedule<64, 128>", source)
                self.assertIn("QRKSVS_TDM_PINGPONG", source)
                self.assertIn(_D192_PIPELINE, candidate.name)

    def test_filtered_feature_keeps_ungated_generic_fallback(self):
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
                self.assertFalse(any(_is_sched_kernel(k) for k in kernels))
                family = [
                    k
                    for k in kernels
                    if k.F_pipeline.tag == _D192_PIPELINE and k.F_hdim == 192
                ]
                # d192 qr_tdm_pingpong takes the soft cap and sink tokens; d192 has
                # no bias or dropout instances at all, and only qr serves skip.
                self.assertEqual(bool(family), feature in ("logits", "sink"))
                for kernel in family:
                    self.assertEqual(getattr(kernel.F_pipeline, "F_" + feature), "t")
                # Other instances stay reachable behind it.
                self.assertTrue(
                    any(k.F_pipeline.tag != _D192_PIPELINE for k in kernels)
                )
                self.assertNotIn("QRKSVS_TDM_SCHED", api)
                self.assertNotIn(_D192_SCHED_SELECTOR, api)

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
            for kernel_filter in ("*d192_bf16*", f"*d192_bf16*_{_D192_PIPELINE}_*"):
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
                    self.assertFalse(any(_is_sched_kernel(k) for k in kernels))
                    if kernel_filter != "*d192_bf16*":
                        self.assertEqual(candidate_count, len(kernels))

                    with tempfile.TemporaryDirectory() as output_dir:
                        output_path = Path(output_dir)
                        write_fwd_api(api_pool, output_path)
                        api = (output_path / "fmha_fwd_api.cpp").read_text()
                    self.assertIn("QRKSVS_TDM_PINGPONG", api)
                    self.assertNotIn(_D192_SCHED_SELECTOR, api)

            with self.subTest(receipt=receipt, kernel_filter="sched"):
                _, kernels = get_fwd_blobs(
                    targets=["gfx1250"],
                    kernel_filter=f"*{_SCHED_PIPELINE}*",
                    receipt=receipt,
                    optdim_list=[192, 256],
                    mask_impl="simplified",
                )
                self.assertFalse(kernels)

    def test_only_gfx125_targets_emit_the_scheduled_pipeline(self):
        for target in ("gfx942", "gfx950"):
            with self.subTest(target=target):
                _, kernels = get_fwd_blobs(
                    targets=[target],
                    kernel_filter=f"*{_SCHED_PIPELINE}*",
                    receipt=100,
                    optdim_list=[128, 192],
                    mask_impl="simplified",
                )
                self.assertFalse(kernels)

        for target in ("gfx1250", "gfx1251", "gfx1252"):
            with self.subTest(target=target):
                _, kernels = get_fwd_blobs(
                    targets=[target],
                    kernel_filter=f"*{_SCHED_PIPELINE}*",
                    receipt=100,
                    optdim_list=[128, 192],
                    mask_impl="simplified",
                )
                self.assertTrue(kernels)
                # Only d128 keeps qr_tdm_sched instances.
                self.assertEqual({k.F_hdim for k in kernels}, {128})
                self.assertTrue(all(_is_sched_kernel(k) for k in kernels))

    def test_d192_m128_emits_batch_and_group_for_every_mask(self):
        mask_options = {
            "simplified": {
                (mask, lse) for mask in ("s_no", "s_mask", "causal") for lse in ("f", "t")
            },
            "generic": {
                (mask, lse)
                for mask in ("no", "causal", "generic")
                for lse in ("f", "t")
            },
        }
        for dtype in ("fp16", "bf16"):
            for mask_impl, masks in mask_options.items():
                with self.subTest(dtype=dtype, mask_impl=mask_impl):
                    self.check_m128_masks(dtype, mask_impl, masks)

    def check_m128_masks(self, dtype, mask_impl, masks):
        _, kernels = get_fwd_blobs(
            targets=["gfx1250"],
            kernel_filter=f"*d192_{dtype}*",
            receipt=600,
            optdim_list=[192],
            mask_impl=mask_impl,
        )
        self.assertFalse(any(_is_sched_kernel(k) for k in kernels))
        m128 = [
            kernel
            for kernel in kernels
            if kernel.F_pipeline.tag == _D192_PIPELINE
            and kernel.F_tile.F_bm0 == 128
            and _has_base_features(kernel)
        ]
        self.assertEqual(
            {
                (kernel.F_mode, kernel.F_pipeline.F_mask, kernel.F_pipeline.F_lse)
                for kernel in m128
            },
            {(mode, mask, lse) for mode in ("batch", "group") for mask, lse in masks},
        )
        for kernel in m128:
            if kernel.F_mode == "group":
                self.assertEqual(kernel.F_pipeline.F_spad, "t")
                self.assertEqual(kernel.F_pipeline.F_skpad, "t")


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
    def candidate_id(
        dispatch, mode, mask="s_no", lse="f", bm0=128, dtype="bf16", sink="f", dpad="f"
    ):
        # qr_tdm_pingpong serves every d192 query.
        return dispatch.expected_id(
            mode=mode,
            dtype=dtype,
            hdim=192,
            pipeline="qr_tdm_pingpong",
            bm0=bm0,
            mask=mask,
            lse=lse,
            spad="t" if mode == "group" else "f",
            dpad=dpad,
            sink=sink,
        )

    @staticmethod
    def fallback_id(dispatch, mode, mask="s_no", lse="f", sink="f"):
        # Without qr_tdm_pingpong, d192 is left to the generic qr fallback (qr_tdm is
        # not generated for it); its padded instance takes the (192, 128) heads.
        return dispatch.expected_id(
            mode=mode,
            hdim=192,
            pipeline="qr",
            bm0=128,
            mask=mask,
            lse=lse,
            spad="t",
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

    @staticmethod
    def d256_id(dispatch, mode, dpad):
        return dispatch.expected_id(
            mode=mode,
            hdim=256,
            pipeline="qr_tdm_pingpong",
            bm0=64,
            mask="s_no",
            lse="f",
            spad="t" if mode == "group" else "f",
            dpad=dpad,
            sink="f",
        )

    def test_real_dispatch_priority_exact_shape_and_short_q(self):
        dispatch, _ = self.compile()
        for mode, group in (("batch", 0), ("group", 1)):
            candidate = self.candidate_id(dispatch, mode)
            self.assertEqual(dispatch.run(group=group), [candidate])
            for length in (128, 129, 257, 2048):
                self.assertEqual(
                    dispatch.run(group=group, seqlen_q=length), [candidate]
                )
            # Shorter queries take qr_tdm_pingpong's 16- and 64-row tiles.
            for length, bm0 in ((0, 16), (1, 16), (16, 16), (17, 64), (127, 64)):
                self.assertEqual(
                    dispatch.run(group=group, seqlen_q=length),
                    [self.candidate_id(dispatch, mode, bm0=bm0)],
                )
            self.assertEqual(
                dispatch.run(group=group, hdim_q=160, hdim_v=128),
                [self.candidate_id(dispatch, mode, dpad="t")],
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
                [self.d256_id(dispatch, mode, dpad="t")],
            )
            self.assertEqual(
                dispatch.run(group=group, hdim_q=256, hdim_v=256),
                [self.d256_id(dispatch, mode, dpad="f")],
            )

    def test_long_k_and_masks_stay_on_pingpong(self):
        # qr_tdm_sched emits no d192 instances (no N128 long-K selector, no N64
        # fallback): qr_tdm_pingpong's 128-row tile takes every d192 query from 128
        # rows on, whatever the key length, grid or mask.
        dispatch, kernels = self.compile(
            kernel_filter=(
                "fmha_fwd_d*_bf16_*_nlogits*_nbias*_*lse*_ndropout*_nskip*_nqscale*"
            )
        )
        self.assertFalse(any(_is_sched_kernel(k) and k.F_hdim == 192 for k in kernels))
        candidate = self.candidate_id(dispatch, "batch")

        for sk in (128, 192, 256, 511, 512, 513, 2048):
            self.assertEqual(dispatch.run(seqlen_k=sk), [candidate])
        self.assertEqual(dispatch.run(seqlen_k=512, has_cu_sk=1), [candidate])

        grid256 = dict(batch=8, nhead_q=8, seqlen_q=512, seqlen_k=512)
        self.assertEqual(dispatch.run(**grid256), [candidate])
        self.assertEqual(dispatch.run(num_cus=128, **grid256), [candidate])
        self.assertEqual(dispatch.run(**{**grid256, "batch": 16}), [candidate])
        self.assertEqual(dispatch.run(max_seqlen_q=1024, **grid256), [candidate])
        for device in ("gfx1251", "gfx1252"):
            self.assertEqual(dispatch.run(device=device, **grid256), [candidate])
            self.assertEqual(dispatch.run(device=device, seqlen_k=511), [candidate])

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
                [self.candidate_id(dispatch, "batch", mask="causal")],
            )
            self.assertEqual(
                dispatch.run(**{**causal, "seqlen_q": 256}),
                [self.candidate_id(dispatch, "batch", mask="causal")],
            )
            for window in ({"window_size_left": 128}, {"window_size_right": 64}):
                self.assertEqual(
                    dispatch.run(**{**causal, **window}),
                    [self.candidate_id(dispatch, "batch", mask="s_mask")],
                )
        self.assertEqual(
            dispatch.run(mask=3, seqlen_k=512),
            [self.candidate_id(dispatch, "batch", mask="s_mask")],
        )
        group = self.candidate_id(dispatch, "group")
        for changes in (
            dict(seqlen_k=512),
            dict(seqlen_k=1024),
            dict(seqlen_k=1024, has_cu_sk=1),
            dict(seqlen_k=1024, has_sk_ptr=1),
            dict(batch=4, seqlen_k=2047),
            dict(batch=4, seqlen_k=2048),
        ):
            self.assertEqual(dispatch.run(group=1, **changes), [group])
        self.assertEqual(
            dispatch.run(lse=1, seqlen_k=512),
            [self.candidate_id(dispatch, "batch", lse="t")],
        )

    def test_fp16_scheduled_dispatch_is_d128_only(self):
        dispatch, kernels = self.compile(kernel_filter="*qr_tdm_sched*")
        self.assertEqual(
            {(k.F_dtype, k.F_hdim, k.F_tile.F_bn0) for k in kernels},
            {
                (dtype, hdim, bn0)
                for dtype in ("fp16", "bf16")
                for hdim, bn0 in ((128, 128),)
            },
        )

        for mode, group in (("batch", 0), ("group", 1)):
            with self.subTest(mode=mode, hdim=128):
                expected = dispatch.expected_id(
                    dtype="fp16",
                    mode=mode,
                    hdim=128,
                    bn0=128,
                    pipeline="qr_tdm_sched",
                    mask="s_no",
                    lse="f",
                    sink="f",
                )
                self.assertEqual(
                    dispatch.run(dtype="fp16", group=group, hdim_q=128, seqlen_k=128),
                    [expected],
                )
                # From 256 x num_cus query rows over all (batch, head) pairs on, d128
                # belongs to qr_tdm_pingpong, which this inventory leaves out.
                self.assertEqual(
                    dispatch.run(dtype="fp16", group=group, hdim_q=128, seqlen_q=4096),
                    [-1],
                )

            # d192 always belongs to qr_tdm_pingpong, and qr_tdm_sched emits no d192
            # instances: nothing serves it here, short or long K.
            long_k = 1024 if group else 512
            for length in (128, long_k):
                with self.subTest(mode=mode, hdim=192, length=length):
                    self.assertEqual(
                        dispatch.run(
                            dtype="fp16",
                            group=group,
                            hdim_q=192,
                            seqlen_q=length,
                            seqlen_k=length,
                        ),
                        [-1],
                    )

    def test_sched_filtered_inventory_still_dispatches_long_k(self):
        # Leaving out the (d128-only) scheduled instances changes nothing for d192.
        dispatch, _ = self.compile(
            filter_fn=lambda trait: trait.pipeline_tag != _SCHED_PIPELINE
        )
        candidate = self.candidate_id(dispatch, "batch")
        self.assertEqual(dispatch.run(seqlen_k=512), [candidate])
        self.assertEqual(dispatch.run(seqlen_q=512, seqlen_k=2048), [candidate])

    def test_n128_generic_no_mask_and_causal_dispatch(self):
        dispatch, _ = self.compile(
            kernel_filter=(
                "fmha_fwd_d*_bf16_*_nlogits*_nbias*_*lse*_ndropout*_nskip*_nqscale*"
            ),
            mask_impl="generic",
        )
        sched, _ = self.compile(kernel_filter="*qr_tdm_sched*", mask_impl="generic")
        for mode, group in (("batch", 0), ("group", 1)):
            with self.subTest(mode=mode):
                length = 1024 if group else 512
                dense = dict(group=group, seqlen_q=length, seqlen_k=length)
                causal = dict(dense, mask=1, window_size_left=-1, window_size_right=0)
                window = dict(causal, mask=3, window_size_left=64)
                for arguments, mask in (
                    (dense, "no"),
                    (causal, "causal"),
                    (dict(causal, mask=2), "causal"),
                    (window, "generic"),
                ):
                    self.assertEqual(
                        dispatch.run(**arguments),
                        [self.candidate_id(dispatch, mode, mask=mask)],
                    )
                    # The scheduled instances (d128 only) serve none of them.
                    self.assertEqual(sched.run(**arguments), [-1])

    def test_family_only_and_empty_filtered_dispatch(self):
        family, _ = self.compile(
            filter_fn=lambda trait: (
                trait.pipeline_tag == "qr_tdm_pingpong" and str(trait.hdim) == "192"
            )
        )
        self.assertEqual(family.run(), [self.candidate_id(family, "batch")])
        self.assertEqual(
            family.run(seqlen_q=127), [self.candidate_id(family, "batch", bm0=64)]
        )
        # Narrower heads take its padded instance; only what that cannot pad is left.
        self.assertEqual(
            family.run(hdim_q=128), [self.candidate_id(family, "batch", dpad="t")]
        )
        for arguments in ({"hdim_q": 191}, {"hdim_q": 193}):
            self.assertEqual(family.run(**arguments), [-1])
        # The scheduled family (d128 only) serves no d192 query.
        sched, _ = self.compile(
            filter_fn=lambda trait: trait.pipeline_tag == _SCHED_PIPELINE
        )
        for arguments in ({}, {"seqlen_k": 512}, {"seqlen_q": 2048, "seqlen_k": 2048}):
            self.assertEqual(sched.run(**arguments), [-1])
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
                dict(arg_v=192),
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
                if kernel.F_hdim == 128 and _is_sched_kernel(kernel)
            ]
        )
        for mode, group in (("batch", 0), ("group", 1)):
            sched = dispatch.expected_id(
                mode=mode,
                hdim=128,
                pipeline="qr_tdm_sched",
                mask="s_no",
                lse="f",
                sink="f",
            )
            # qr_tdm_sched takes d128 from 128 rows on while 2 x 8 (batch, head)
            # pairs hold fewer than 256 x num_cus query rows; qr_tdm_pingpong the rest.
            for length, bm0 in (
                (1, 16),
                (16, 16),
                (17, 64),
                (127, 64),
                (128, None),
                (2047, None),
                (2048, None),
                (4095, None),
                (4096, 128),
                (32768, 128),
            ):
                selected = sched
                if bm0 is not None:
                    selected = dispatch.expected_id(
                        mode=mode,
                        hdim=128,
                        pipeline="qr_tdm_pingpong",
                        bm0=bm0,
                        spad="t" if group else "f",
                        mask="s_no",
                        lse="f",
                        sink="f",
                        dpad="f",
                    )
                self.assertEqual(
                    dispatch.run(
                        group=group,
                        hdim_q=128,
                        hdim_v=128,
                        seqlen_q=length,
                    ),
                    [selected],
                    length,
                )

    def test_real_dispatch_d128_o_layout_fallback(self):
        dispatch, _ = self.compile()
        for mode, group in (("batch", 0), ("group", 1)):
            sched = dispatch.expected_id(
                mode=mode,
                hdim=128,
                pipeline="qr_tdm_sched",
                mask="s_no",
                lse="f",
                sink="f",
            )
            fallback = dispatch.expected_id(
                mode=mode,
                hdim=128,
                pipeline="qr_tdm",
                bm0=128,
                spad="t" if group else "f",
                mask="s_no",
                lse="f",
                sink="f",
                dpad="f",
            )
            rejected = [
                dict(o_offset=2),
                dict(stride_o=1028),
                dict(nhead_stride_o=132),
                dict(stride_o=-1024),
                dict(stride_o=16909320),
            ]
            accepted = [dict(stride_o=16909312)]
            (rejected if group == 0 else accepted).append(dict(batch_stride_o=262148))
            for changes in rejected + accepted:
                selected = fallback if changes in rejected else sched
                self.assertEqual(
                    dispatch.run(
                        group=group, hdim_q=128, hdim_v=128, seqlen_q=2048, **changes
                    ),
                    [selected],
                    changes,
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
            sink = self.candidate_id(dispatch, mode, sink="t")
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
                        (lambda trait: trait.pipeline_tag != "qr_tdm_pingpong")
                        if render_filter
                        else None
                    ),
                )
                for mode, group in (("batch", 0), ("group", 1)):
                    # Without qr_tdm_pingpong (and with no d192 qr_tdm or
                    # qr_tdm_sched instances) the generic qr fallback serves.
                    fallback = (
                        self.fallback_id(dispatch, mode)
                        if render_filter
                        else self.d256_id(dispatch, mode, dpad="t")
                    )
                    self.assertEqual(dispatch.run(group=group), [fallback])

    def test_real_dispatch_candidate_only_filter_is_strict(self):
        for receipt in (0, 100, 200, 600):
            with self.subTest(receipt=receipt):
                dispatch, kernels = self.compile(
                    receipt=receipt, kernel_filter="*d192_bf16*_qr_tdm_pingpong_*"
                )
                self.assertTrue(kernels)
                self.assertTrue(
                    all(
                        kernel.F_pipeline.tag == "qr_tdm_pingpong"
                        and kernel.F_hdim == 192
                        for kernel in kernels
                    )
                )
                self.assertEqual(len({kernel.name for kernel in kernels}), len(kernels))
                modes = sorted({kernel.F_mode for kernel in kernels})
                for mode in modes:
                    group = int(mode == "group")
                    self.assertEqual(
                        dispatch.run(group=group), [self.candidate_id(dispatch, mode)]
                    )
                    self.assertEqual(
                        dispatch.run(group=group, seqlen_q=127),
                        [self.candidate_id(dispatch, mode, bm0=64)],
                    )
                    self.assertEqual(dispatch.run(group=group, hdim_q=191), [-1])
                    self.assertEqual(dispatch.run(group=group, hdim_q=193), [-1])
                    self.assertEqual(dispatch.run(group=group, skip=1), [-1])

                # The scheduled family (d128 only) alone serves no d192 query.
                sched, sched_kernels = self.compile(
                    receipt=receipt, kernel_filter=f"*{_SCHED_PIPELINE}*"
                )
                self.assertTrue(sched_kernels)
                self.assertTrue(all(_is_sched_kernel(k) for k in sched_kernels))
                self.assertEqual({k.F_hdim for k in sched_kernels}, {128})
                for mode in modes:
                    group = int(mode == "group")
                    self.assertEqual(sched.run(group=group), [-1])
                    self.assertEqual(
                        sched.run(group=group, seqlen_q=2048, seqlen_k=2048), [-1]
                    )

    def test_real_dispatch_receipt_and_optdim_baseline(self):
        cmake = (_FMHA_EXAMPLE_DIR / "CMakeLists.txt").read_text()
        gfx125_dims = re.search(
            r"if\(FMHA_FWD_HAS_GFX125\)\s+set\(FMHA_FWD_OPTDIM\s+([0-9,]+)\)",
            cmake,
        )
        self.assertIsNotNone(gfx125_dims)
        common_dims = list(map(int, gfx125_dims[1].split(",")))
        self.assertEqual(common_dims, [32, 64, 96, 128, 160, 192, 256, 512])
        for receipt in (0, 100, 200, 600):
            for optdims in ([128], [192], [256], [-1], common_dims):
                with self.subTest(receipt=receipt, optdims=optdims):
                    dispatch, kernels = self.compile(receipt=receipt, optdims=optdims)
                    modes = sorted({kernel.F_mode for kernel in kernels})
                    # softmax's: GDPA keeps a d64 qr_tdm_sched too (receipt 0)
                    family = [
                        kernel
                        for kernel in _softmax(kernels)
                        if _is_sched_kernel(kernel)
                    ]
                    expected_family_dims = (
                        {128} if optdims == [-1] else {128}.intersection(optdims)
                    )
                    self.assertEqual(
                        {kernel.F_hdim for kernel in family}, expected_family_dims
                    )
                    for mode in modes:
                        group = int(mode == "group")
                        if optdims == [128]:
                            self.assertEqual(dispatch.run(group=group), [-1])
                        elif optdims == [256]:
                            self.assertEqual(
                                dispatch.run(group=group),
                                [self.d256_id(dispatch, mode, dpad="t")],
                            )
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

        # Below 128 rows the 64-row qr_tdm_pingpong tile takes the query; the broken
        # gate shuts it out and the unconditional 128-row tile behind it takes over.
        dispatch, _ = self.compile(mutate=mutate)
        actual = dispatch.run(seqlen_q=127)
        self.assertNotEqual(actual, [self.candidate_id(dispatch, "batch", bm0=64)])
        self.assertEqual(actual, [self.candidate_id(dispatch, "batch", bm0=128)])


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
        # Only d128 keeps qr_tdm_sched instances: qr_tdm_pingpong serves d64 and d192.
        for dimensions, expected in (
            ([-1], {128}),
            ([128], {128}),
            ([192], set()),
            ([256], set()),
            ([128, 192], {128}),
            ([32, 64, 80, 128, 256], {128}),
            ([], set()),
            ([64], set()),
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
        kernels = _softmax(kernels)
        # Only d128 keeps qr_tdm_sched instances.
        selected_dims = {128} if dimensions == [-1] else {128}.intersection(dimensions)
        self.assertEqual({k.F_hdim for k in kernels}, selected_dims)
        self.assertEqual(
            sum(_has_base_features(k) for k in kernels), count * len(selected_dims)
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
                ["gfx1250"], f"*d64_{dtype}*", 600, [64], "simplified"
            )
            # qr_tdm_pingpong serves every d64 query: neither qr_tdm_sched nor qr_tdm
            # emits d64 instances, and only the generic qr fallback stays.
            self.assertEqual(
                {k.F_pipeline.tag for k in kernels}, {"qr_tdm_pingpong", "qr"}
            )
            pingpong = [
                k
                for k in kernels
                if k.F_pipeline.tag == "qr_tdm_pingpong" and _has_base_features(k)
            ]
            self.assertEqual({k.F_tile.F_bm0 for k in pingpong}, {16, 64, 128})
            self.assertEqual({k.F_mode for k in pingpong}, {"batch", "group"})
            for kernel in pingpong:
                self.assertEqual(kernel.F_tile.F_bn0, 64)
                self.assertEqual(kernel.F_tile.F_bn1, 64)
                self.assertEqual(kernel.F_tile.F_bk0max, 64)
                trait = kernel.api_trait()
                if kernel.F_mode == "group":
                    self.assertEqual(trait.spad, "t")
                    self.assertEqual(trait.skpad, "t")
                elif trait.skpad == "f":
                    # The unpadded batch instances take any K length.
                    self.assertEqual(trait.skcheck, "true")
                if trait.dpad == "f":
                    self.assertEqual(trait.dcheck, "a.hdim_q % 64 == 0")
                if trait.dvpad == "f":
                    self.assertEqual(trait.dvcheck, "a.hdim_v == 64")
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
        kernels = _softmax(kernels)
        family = [k for k in kernels if k.F_pipeline.tag == "qr_tdm_sched"]
        self.assertEqual(sum(_has_base_features(k) for k in family), count)
        self.assertEqual(len(family), len(kernels))
        self.assertEqual(len({k.filename for k in kernels}), len(kernels))
        self.assertEqual({k.F_pipeline.F_sink for k in family}, {"f", "t"})
        for kernel in family:
            # Only the virtual-sink (sink) instances accept a sink pointer.
            self.assertEqual(
                "a.sink_ptr == nullptr" in str(kernel.F_pipeline.F_constraint),
                kernel.F_pipeline.F_sink == "f",
            )
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
        family = [k for k in _softmax(kernels) if k.F_pipeline.tag == "qr_tdm_sched"]
        # Only d128 keeps qr_tdm_sched instances.
        self.assertEqual({k.F_hdim for k in family}, {128})
        self.assertEqual(sum(_has_base_features(k) for k in family), 16)
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

    def test_tile_dispatch_order_validator(self):
        def tile(bm0, constraint=None, owner=None):
            return FmhaFwdTileSize(
                bm0, 64, 32, 64, 32, 64, 4, 1, 1, 4, 1, 1, 16, 16, 32, 16, 16, 32, -1,
                CppConstraint(constraint), owner,
            )  # fmt: skip

        pp = "qr_tdm_pingpong"
        valid = {
            # an owner's unconditional last tile, then another owner's tiles
            (64, 64): [tile(64, "a.max_seqlen_q < 128", pp), tile(128, "true", pp),
                       tile(128)],
            (128, 128): [tile(64, "a.max_seqlen_q < 128"), tile(128, None, pp),
                         tile(128)],
        }  # fmt: skip
        self.assertIs(_validate_gfx125_tile_dispatch_order(valid), valid)
        for tiles in (
            [tile(64), tile(128)],
            [tile(64, "true"), tile(128, "a.max_seqlen_q >= 128")],
            [tile(64, None, pp), tile(128, None, pp), tile(128)],
        ):
            with self.subTest(tiles=[(t.F_bm0, str(t.F_constraint)) for t in tiles]):
                with self.assertRaises(AssertionError):
                    _validate_gfx125_tile_dispatch_order({(64, 64): tiles})
        # The generated inventory passes it.
        for dtype in ("fp16", "bf16"):
            KernelComponentFactoryGfx125.get_hdim_tile_size_dict(dtype)

    def test_dropout_skips_tiles_with_qk_warps_on_n(self):
        _, kernels = get_fwd_blobs(
            ["gfx1250"], "*_dropout*", 600, [64, 512], "simplified"
        )
        self.assertTrue(kernels)
        # BlockDropout needs the QK gemm warps on M (rn0 == 1): the short-query tiles,
        # with their warps on N, take no dropout.
        self.assertTrue(all(k.F_tile.F_rn0 == 1 for k in kernels))
        self.assertTrue(
            any(
                k.F_pipeline.tag == "qr_tdm_pingpong"
                for k in kernels
                if k.F_hdim == 512
            )
        )

    def test_sched_tile_geometry_identity_and_inventory_isolation(self):
        factory = KernelComponentFactoryGfx125
        for dtype in ("fp16", "bf16"):
            inventory = factory.get_hdim_tile_size_dict(dtype)
            # softmax's qr_tdm_sched is generated for d128 only, and GDPA's also for
            # d64 (_QR_TDM_SCHED_GDPA_TILES), so no other head dim lists a tile of it.
            self.assertEqual(set(factory._QR_TDM_SCHED_GDPA_TILES), {(64, 64)})
            for (hdim, hdim_v), tiles in inventory.items():
                if (hdim, hdim_v) not in ((128, 128), (64, 64)):
                    self.assertFalse(
                        any(
                            factory.is_tile_supported_by_qr_tdm_sched(tile, hdim)
                            for tile in tiles
                        )
                    )
            for hdim in (64, 128):
                tiles = [
                    tile
                    for tile in inventory[(hdim, hdim)]
                    if factory.is_tile_supported_by_qr_tdm_sched(tile, hdim)
                ]
                self.assertEqual(len(tiles), 1)
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
                fresh = factory.get_hdim_tile_size_dict(dtype)[(hdim, hdim)]
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
                for head, n, v in ((128, 128, 128),)
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
        # qr_tdm_pingpong serves d64 and d192; only d128 keeps qr_tdm_sched.
        d192 = "fmha_fwd_d192_bf16_batch_b128_qr_tdm_pingpong_vr_nmask_nlse_gfx125.cpp"
        d64 = d192.replace("d192", "d64")
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
            # Every head dimension the tuned pipelines serve uses expert scheduling.
            d64,
            d64.replace("bf16", "fp16"),
            # every qr_tdm_pingpong instance, with or without LSE and at any head
            # dimension, while plain qr_tdm (4) and qr_tdm_sched outside d64/d128/d192 (6)
            # keep the default
            d128.replace("qr_tdm_sched", "qr_tdm_pingpong"),
            d128.replace("qr_tdm_sched", "qr_tdm_pingpong").replace("_nlse_", "_lse_"),
            d192.replace("d192", "d256"),
            d192.replace("d192", "d32").replace("_nlse_", "_lse_"),
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
                    i in (0, 1, 2, 3, 5, 12, 13, 14, 15, 16, 17),
                )
                self.assertNotIn("-wwm-regalloc=fast", line)


@unittest.skipUnless(_HOST_CXX, "a C++17 host compiler is required")
class TestCompiledGfx125V128Dispatch(unittest.TestCase):
    def test_d64_sequence_padding_and_pingpong_tiles(self):
        for dtype in ("fp16", "bf16"):
            pool, kernels = get_fwd_blobs(
                ["gfx1250"],
                f"fmha_fwd_d64_{dtype}_*_nlogits*_nbias*_*lse*_ndropout*_nskip*_nqscale*_ntrload*_nsink*",
                600,
                [64],
                "simplified",
            )
            # qr_tdm_sched emits no d64 instances.
            self.assertFalse(any(_is_sched_kernel(k) for k in kernels))
            dispatch = _CompiledDispatcher(self, pool, kernels)
            # Without qr_tdm_pingpong (d64 has no qr_tdm instances), the generic qr
            # fallback serves.
            without_pingpong = _CompiledDispatcher(
                self,
                pool,
                kernels,
                filter_fn=lambda trait: trait.pipeline_tag != "qr_tdm_pingpong",
            )
            qr_ids = {
                value
                for kernel, value in without_pingpong.kernel_ids
                if kernel.F_pipeline.tag == "qr"
            }
            # qr_tdm_pingpong takes every d64 query; its unpadded batch instances take
            # any K length, and its 128-row tile needs 1024 keys or 512 dense rows.
            for sq, sk, has_cu_sk, group, bm0 in (
                (16, 128, 0, 0, 16),
                (127, 128, 0, 0, 64),
                (128, 128, 0, 0, 64),
                (129, 129, 0, 0, 64),
                (255, 255, 0, 0, 64),
                (128, 0, 0, 0, 64),
                (128, 128, 1, 0, 64),
                (255, 255, 0, 1, 64),
                (128, 1024, 0, 0, 128),
                (600, 600, 0, 0, 128),
            ):
                candidate = dispatch.expected_id(
                    dtype=dtype,
                    hdim=64,
                    mode="group" if group else "batch",
                    pipeline="qr_tdm_pingpong",
                    bm0=bm0,
                    mask="s_no",
                    lse="t",
                    dpad="f",
                    dvpad="f",
                    sink="f",
                    spad="t" if group else "f",
                )
                args = dict(
                    dtype=dtype,
                    hdim_q=64,
                    hdim_v=64,
                    seqlen_q=sq,
                    seqlen_k=sk,
                    has_cu_sk=has_cu_sk,
                    group=group,
                    lse=1,
                )
                self.assertEqual(dispatch.run(**args), [candidate])
                self.assertIn(without_pingpong.run(**args)[0], qr_ids)

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
            # A grid of at least 256 x num_cus query rows belongs to qr_tdm_pingpong.
            self.assertEqual(dispatch.run(**args, seqlen_q=4096), [-1])

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
                pingpong = dict(
                    hdim=dim, pipeline="qr_tdm_pingpong", spad="t", dpad="f", **features
                )
                # d128 from 128 rows on a small grid: qr_tdm_sched; d192: always
                # qr_tdm_pingpong.
                candidate = (
                    dispatch.expected_id(hdim=dim, pipeline="qr_tdm_sched", **features)
                    if dim == 128
                    else dispatch.expected_id(bm0=128, **pingpong)
                )
                fallback = dispatch.expected_id(bm0=64, **pingpong)
                for heads, kv_heads in ((1, 1), (8, 8), (8, 2), (8, 1)):
                    for member_q, max_q in (
                        (0, 128),
                        (1, 128),
                        (127, 128),
                        (128, 128),
                        (0, 127),
                        (127, 127),
                    ):
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
        for dim in (128, 192):
            # d128 on a small grid: qr_tdm_sched; d192: qr_tdm_pingpong.
            family = "qr_tdm_sched" if dim == 128 else "qr_tdm_pingpong"
            family_ids = {
                value
                for kernel, value in dispatch.kernel_ids
                if kernel.F_pipeline.tag == family and kernel.F_hdim == dim
            }
            for mode, group in (("batch", 0), ("group", 1)):
                for mask in (0, 1, 2, 3):
                    for lse in (0, 1):
                        features = dict(
                            mode=mode,
                            mask="s_mask" if mask else "s_no",
                            lse="t" if lse else "f",
                            sink="f",
                        )
                        if dim == 128:
                            candidate = dispatch.expected_id(
                                hdim=dim, pipeline=family, bn0=128, **features
                            )
                        else:
                            candidate = dispatch.expected_id(
                                hdim=dim,
                                pipeline=family,
                                bm0=128,
                                spad="t" if group else "f",
                                dpad="f",
                                **features,
                            )
                        args = dict(hdim_q=dim, group=group, mask=mask, lse=lse)
                        self.assertEqual(dispatch.run(**args), [candidate])
                        # A sink pointer without sink tokens turns the sink-free
                        # qr_tdm_sched instances (no virtual sink) away to qr_tdm;
                        # qr_tdm_pingpong keeps the query.
                        sink_ptr = candidate
                        if dim == 128:
                            sink_ptr = dispatch.expected_id(
                                hdim=dim,
                                pipeline="qr_tdm",
                                bm0=128,
                                spad="t" if group else "f",
                                dpad="f",
                                **features,
                            )
                        self.assertEqual(
                            dispatch.run(**args, has_sink_ptr=1), [sink_ptr]
                        )
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
                            self.assertNotEqual(
                                dispatch.run(**rejected_args), [candidate]
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
                    # qr_tdm_sched from 128 rows while 2 x 8 (batch, head) pairs hold
                    # fewer than 256 x num_cus query rows, qr_tdm_pingpong otherwise;
                    # its 128-row tile needs 512 dense rows (or 1024 keys).
                    for size in (1, 16, 17, 127, 128, 129, 2047, 2048, 4095, 4096):
                        if 128 <= size < 4096:
                            expected = candidate
                        else:
                            bm0 = 16 if size <= 16 else 64
                            if size >= 512 and mask == 0:
                                bm0 = 128
                            expected = dispatch.expected_id(
                                pipeline="qr_tdm_pingpong",
                                spad="t" if group else "f",
                                bm0=bm0,
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
                        self.assertEqual(dispatch.run(**args), [expected], size)
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

    def test_pingpong_tiles(self):
        # qr_tdm_pingpong runs on its own fp16/bf16 tiles only, which no other
        # pipeline takes; it alone uses the LDS-shuffle epilogue.
        dims = [32, 128, 192, 256, 512]
        _, kernels = get_fwd_blobs(["gfx1250"], "*", 600, dims, "simplified")
        pingpong = [k for k in kernels if k.F_pipeline.tag == "qr_tdm_pingpong"]
        self.assertTrue(pingpong)
        for kernel in kernels:
            owner = kernel.F_tile.F_pipeline_tag
            self.assertEqual(
                owner == "qr_tdm_pingpong",
                kernel.F_pipeline.tag == "qr_tdm_pingpong",
                kernel.name,
            )
            self.assertEqual(
                "LdsShuffle2DEpilogue" in kernel.render(),
                owner in ("qr_tdm_pingpong", "qr_tdm_kvshared"),
                kernel.name,
            )
        # d256/d512 have no 128-row tile, d32 no 16-row one. Pingpong stages Q/K at
        # their own row (d32 does not pad to 64), and the parts of the LDS read
        # schedule that take effect on a tile show in its instance name: slice
        # prefetch from 64 rows on, V before the softmax on the 128-row tiles here.
        row_tiles = {
            32: {64, 128},
            128: {16, 64, 128},
            192: {16, 64, 128},
            256: {16, 64},
            512: {16, 64},
        }
        schedule = {
            (32, 64): "_sp64",
            (32, 128): "_sp64_vbs128",
            (192, 64): "_sp64",
            (192, 128): "_sp64_vbs128",
        }
        self.assertEqual(
            {(k.F_hdim, k.F_tile.F_bm0) for k in pingpong},
            {(hdim, bm0) for hdim, tiles in row_tiles.items() for bm0 in tiles},
        )
        for kernel in pingpong:
            with self.subTest(kernel=kernel.name):
                self.assertIn(kernel.F_dtype, ("fp16", "bf16"))
                self.assertEqual(kernel.F_tile.F_bk0max, kernel.F_hdim)
                self.assertEqual(
                    kernel.F_tile.F_bn1, 128 if kernel.F_hdim == 192 else kernel.F_hdim
                )
                suffix = schedule.get((kernel.F_hdim, kernel.F_tile.F_bm0), "")
                self.assertTrue(
                    kernel.name.endswith("_ntrload" + suffix + "_nsink")
                    or kernel.name.endswith("_ntrload" + suffix + "_sink")
                )
                # The rendered schedule is the one the name gives.
                sp = re.search(r"_sp(\d+)", suffix)
                vbs = re.search(r"_vbs(\d+)", suffix)
                self.assertIn(
                    "QrTdmPingPongLdsReadSchedule<"
                    f"{sp[1] if sp else 0}, {vbs[1] if vbs else 0}>",
                    kernel.render(),
                )
                # The short-query tile does not keep dropout's P layout.
                if kernel.F_tile.F_bm0 == 16:
                    self.assertEqual(kernel.F_pipeline.F_dropout, "f")
                trait = kernel.api_trait()
                if trait.dvpad == "f":
                    self.assertEqual(
                        trait.dvcheck, f"a.hdim_v == {kernel.F_tile.F_bn1}"
                    )
                self.assertIn("BlockFmhaPipelineQRKSVSTdmPingPong", kernel.render())
        self.assertTrue(
            any(k.F_pipeline.F_dropout == "t" for k in pingpong if k.F_tile.F_bm0 > 16)
        )

    def test_d128_pingpong_dispatch(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"],
            "fmha_fwd_d128_bf16_*_nlogits*_ndropout*_nskip*_nqscale*_ntrload*",
            600,
            [128],
            "simplified",
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)

        def kernel(pipeline, bm0, mask="s_no", skpad="f", bias="no", sink="f"):
            return dispatch.expected_id(
                hdim=128,
                mode="batch",
                pipeline=pipeline,
                bm0=bm0,
                mask=mask,
                skpad=skpad,
                dpad="f",
                bias=bias,
                lse="f",
                sink=sink,
            )

        def sched(mask="s_no", sink="f"):
            return dispatch.expected_id(
                hdim=128,
                mode="batch",
                pipeline="qr_tdm_sched",
                mask=mask,
                lse="f",
                sink=sink,
            )

        def run(**changes):
            return dispatch.run(hdim_q=128, hdim_v=128, **changes)

        square = dict(seqlen_q=532, seqlen_k=532)
        causal = dict(mask=1, window_size_left=-1, window_size_right=0)
        # 1280 x 2 (batch, head) pairs: 10 per num_cus (256).
        big = dict(batch=1280, nhead_q=2, nhead_k=2)

        # qr_tdm_sched from 128 rows while the (batch, head) pairs hold fewer than
        # 256 x num_cus query rows; qr_tdm_pingpong everywhere else, at any query
        # and key length.
        self.assertEqual(run(**square), [sched()])
        # The 128-row tile from 512 dense rows or 1024 keys, else the 64-row one.
        self.assertEqual(run(**square, **big), [kernel("qr_tdm_pingpong", 128)])
        self.assertEqual(
            run(seqlen_q=640, seqlen_k=640, **big), [kernel("qr_tdm_pingpong", 128)]
        )
        self.assertEqual(
            run(**square, **causal, **big),
            [kernel("qr_tdm_pingpong", 64, mask="causal")],
        )
        self.assertEqual(
            run(seqlen_q=532, seqlen_k=1024, **causal, **big),
            [kernel("qr_tdm_pingpong", 128, mask="causal")],
        )
        # The grid threshold: 16 x 16 heads of 256 rows hold 256 x 256 rows.
        grid = dict(batch=16, nhead_q=16, nhead_k=16)
        self.assertEqual(
            run(seqlen_q=256, seqlen_k=256, **grid), [kernel("qr_tdm_pingpong", 64)]
        )
        self.assertEqual(run(seqlen_q=255, seqlen_k=255, **grid), [sched()])
        self.assertEqual(
            run(seqlen_q=255, seqlen_k=255, num_cus=128, **grid),
            [kernel("qr_tdm_pingpong", 64)],
        )
        # Below 128 rows qr_tdm_pingpong on any grid: up to 16 rows the 16-row tile.
        for length, bm0 in ((8, 16), (16, 16), (17, 64), (100, 64), (127, 64)):
            self.assertEqual(
                run(seqlen_q=length, seqlen_k=512), [kernel("qr_tdm_pingpong", bm0)]
            )
            self.assertEqual(
                run(seqlen_q=length, seqlen_k=532, **big),
                [kernel("qr_tdm_pingpong", bm0)],
            )
        self.assertEqual(
            run(seqlen_q=32, seqlen_k=200, batch=1792, nhead_q=1, nhead_k=1),
            [kernel("qr_tdm_pingpong", 64)],
        )
        # The unpadded instance serves any seqlen_k, except with an elementwise bias,
        # which still needs a tile multiple.
        self.assertEqual(
            run(**square, **big, bias=1),
            [kernel("qr_tdm_pingpong", 128, skpad="t", bias="bias")],
        )
        self.assertEqual(
            run(seqlen_q=532, seqlen_k=512, bias=1, **big),
            [kernel("qr_tdm_pingpong", 128, bias="bias")],
        )
        self.assertEqual(
            run(**square, **big, bias=2),
            [kernel("qr_tdm_pingpong", 128, bias="alibi")],
        )
        # Both pipelines take sink tokens, and the sink instances also take a sink
        # pointer.
        self.assertEqual(run(**square, sink=1), [sched(sink="t")])
        self.assertEqual(run(**square, sink=1, has_sink_ptr=1), [sched(sink="t")])
        self.assertEqual(run(**square, **causal, sink=1), [sched("s_mask", "t")])
        self.assertEqual(
            run(**square, **big, sink=1), [kernel("qr_tdm_pingpong", 128, sink="t")]
        )
        # A sink pointer alone turns the sink-free qr_tdm_sched instances (no virtual
        # sink) away to qr_tdm; on a big grid qr_tdm_pingpong keeps the query.
        self.assertEqual(
            run(**square, has_sink_ptr=1), [kernel("qr_tdm", 128, skpad="t")]
        )
        self.assertEqual(
            run(**square, **causal, has_sink_ptr=1),
            [kernel("qr_tdm", 128, mask="causal", skpad="t")],
        )
        self.assertEqual(
            run(**square, **big, has_sink_ptr=1), [kernel("qr_tdm_pingpong", 128)]
        )

    def test_d128_kvshared_tiles(self):
        # qr_tdm_kvshared runs on its own d128/d256 fp16/bf16 16-row tiles only.
        _, kernels = get_fwd_blobs(["gfx1250"], "*", 600, [128, 192, 256], "simplified")
        kvshared = [k for k in kernels if k.F_pipeline.tag == "qr_tdm_kvshared"]
        self.assertTrue(kvshared)
        for kernel in kernels:
            self.assertEqual(
                kernel.F_tile.F_pipeline_tag == "qr_tdm_kvshared",
                kernel.F_pipeline.tag == "qr_tdm_kvshared",
                kernel.name,
            )
        for kernel in kvshared:
            self.assertIn(kernel.F_dtype, ("fp16", "bf16"))
            self.assertIn(kernel.F_hdim, (128, 256))
            self.assertEqual(kernel.F_tile.F_bm0, 16)
            self.assertIn("BlockFmhaPipelineQRKSVSTdmKVShared", kernel.render())

    def test_d128_kvshared_dispatch(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"],
            "fmha_fwd_d128_bf16_*_nlogits*_ndropout*_nskip*_nqscale*_ntrload*",
            600,
            [128],
            "simplified",
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)

        def kernel(pipeline, bm0, bn0=64, mask="s_no"):
            return dispatch.expected_id(
                hdim=128,
                mode="batch",
                pipeline=pipeline,
                bm0=bm0,
                bn0=bn0,
                mask=mask,
                skpad="f",
                dpad="f",
                bias="no",
                lse="f",
                sink="f",
            )

        def run(**changes):
            return dispatch.run(hdim_q=128, hdim_v=128, **changes)

        big = dict(batch=1280, nhead_q=2, nhead_k=2)
        # K and V one tensor: up to 16 rows on any grid, one warp from 10 workgroups per CU
        below = dict(batch=1279, nhead_q=2, nhead_k=2)
        for length in (1, 16):
            for grid, bn0 in (
                (dict(batch=1, nhead_q=1, nhead_k=1), 64),
                (below, 64),
                (big, 32),
            ):
                self.assertEqual(
                    run(seqlen_q=length, seqlen_k=532, kv_shared=1, **grid),
                    [kernel("qr_tdm_kvshared", 16, bn0)],
                )
        # two tensors, or more than 16 rows: qr_tdm_pingpong, as without kvshared
        for grid in (big, {}):
            self.assertEqual(
                run(seqlen_q=8, seqlen_k=532, **grid), [kernel("qr_tdm_pingpong", 16)]
            )
            self.assertEqual(
                run(seqlen_q=17, seqlen_k=532, kv_shared=1, **grid),
                [kernel("qr_tdm_pingpong", 64)],
            )


class TestGfx125GdpaCodegen(unittest.TestCase):
    """GDPA, P = act(scale * Q K^T) with no softmax, on qr_tdm, qr_tdm_pingpong and qr_tdm_sched."""

    def gdpa_kernels(self, dtype, receipt, dims):
        _, kernels = get_fwd_blobs(
            ["gfx1250"], f"*{dtype}*", receipt, dims, "simplified"
        )
        return [k for k in kernels if k.F_pipeline.F_activation == "fast_gelu"]

    def test_instances(self):
        for dtype in ("fp16", "bf16"):
            for receipt in (0, 5):
                with self.subTest(dtype=dtype, receipt=receipt):
                    kernels = self.gdpa_kernels(
                        dtype, receipt, [32, 64, 96, 128, 160, 192, 256, 512]
                    )
                    pipelines = {(k.F_hdim, k.F_pipeline.tag) for k in kernels}
                    # every qr_tdm_pingpong head dim (the Python API's receipt: Q and V of
                    # one head dim, up to 256); qr_tdm wherever it has tiles (not d512),
                    # qr_tdm_sched at d64 and d128
                    dims = (32, 64, 96, 128, 160, 256)
                    if receipt == 0:
                        dims += (192, 512)
                    self.assertEqual(
                        pipelines,
                        {(d, "qr_tdm") for d in dims if d != 512}
                        | {(d, "qr_tdm_pingpong") for d in dims}
                        | {(d, "qr_tdm_sched") for d in (64, 128)},
                    )
                    for kernel in kernels:
                        pipeline = kernel.F_pipeline
                        self.assertEqual(pipeline.F_lse, "f")
                        self.assertEqual(pipeline.F_bias, "no")
                        self.assertEqual(pipeline.F_dropout, "f")
                        self.assertEqual(pipeline.F_sink, "f")
                        self.assertEqual(pipeline.F_logits, "f")
                        self.assertEqual(pipeline.F_mask, "s_no")
                        self.assertIn("_fast_gelu", kernel.name)
                        self.assertIn(
                            "BlockAttentionActivationEnum::FAST_GELU", kernel.render()
                        )
                        if kernel.F_mode == "group":
                            self.assertEqual(pipeline.F_spad, "t")
                        if (kernel.F_hdim, pipeline.tag) == (128, "qr_tdm_sched"):
                            # the validated d128 sched specialization pads Q and K
                            self.assertEqual(pipeline.F_spad, "t")
                    names = [k.name for k in kernels]
                    self.assertEqual(len(names), len(set(names)))

    def test_only_gdpa_receipts_emit_it(self):
        # the integrations behind the other receipts never set traits.activation
        for receipt in (3, 4, 100, 200, 600):
            with self.subTest(receipt=receipt):
                self.assertEqual(self.gdpa_kernels("bf16", receipt, [64, 128, 256]), [])

    def test_gdpa_instances_twin_softmax_ones(self):
        # Each GDPA instance is a softmax instance's twin: the same name and trait but for the
        # activation, so GDPA adds instances and changes none of softmax's.
        _, kernels = get_fwd_blobs(["gfx1250"], "*bf16*", 0, [128], "simplified")
        softmax = {k.name for k in kernels if k.F_pipeline.F_activation == "softmax"}
        gdpa = [k for k in kernels if k.F_pipeline.F_activation != "softmax"]
        self.assertTrue(gdpa)
        for kernel in gdpa:
            with self.subTest(kernel=kernel.name):
                self.assertIn("fast_gelu", kernel.api_trait().name)
                self.assertIn(kernel.name.replace("_fast_gelu", ""), softmax)


class TestCompiledGfx125GdpaDispatch(unittest.TestCase):
    def dispatcher(self, mode_glob="*"):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"],
            f"fmha_fwd_d128_bf16_{mode_glob}*_nlogits*_nbias*_ndropout*_nskip*_nqscale*_ntrload*_nsink*",
            0,
            [128],
            "simplified",
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)

        def ids(pipeline, activation, bm0=None, mode=None):
            return {
                index
                for k, index in dispatch.kernel_ids
                if k.F_pipeline.tag == pipeline
                and k.F_pipeline.F_activation == activation
                and (bm0 is None or k.F_tile.F_bm0 == bm0)
                and (mode is None or k.F_mode == mode)
            }

        def run(**changes):
            return dispatch.run(hdim_q=128, hdim_v=128, **changes)[0]

        return run, ids

    def test_batch_routes_by_keys(self):
        run, ids = self.dispatcher()
        gelu = dict(activation=1)
        # few keys (at most 512 per batch): qr_tdm_pingpong on any grid (the softmax
        # range does not hold GDPA back), on the tile qr_tdm_pingpong's row rules pick:
        # 16 rows up to 16, 128 rows from 512 dense rows, 64 rows otherwise
        for grid in ({}, dict(batch=1280, nhead_q=2, nhead_k=2)):
            self.assertIn(
                run(seqlen_q=532, seqlen_k=256, **gelu, **grid),
                ids("qr_tdm_pingpong", "fast_gelu", 128),
            )
            self.assertIn(
                run(seqlen_q=532, seqlen_k=512, **gelu, **grid),
                ids("qr_tdm_pingpong", "fast_gelu", 128),
            )
            self.assertIn(
                run(seqlen_q=200, seqlen_k=512, **gelu, **grid),
                ids("qr_tdm_pingpong", "fast_gelu", 64),
            )
        self.assertIn(
            run(seqlen_q=8, seqlen_k=256, **gelu),
            ids("qr_tdm_pingpong", "fast_gelu", 16),
        )
        # many keys: qr_tdm_sched from 128 rows, qr_tdm below (as for softmax)
        self.assertIn(
            run(seqlen_q=532, seqlen_k=513, **gelu), ids("qr_tdm_sched", "fast_gelu")
        )
        self.assertIn(
            run(seqlen_q=4096, seqlen_k=8192, **gelu), ids("qr_tdm_sched", "fast_gelu")
        )
        self.assertIn(
            run(seqlen_q=100, seqlen_k=1024, **gelu), ids("qr_tdm", "fast_gelu")
        )

    def test_group_routes_by_average_keys(self):
        run, ids = self.dispatcher("group")
        group = dict(
            group=1, activation=1, batch=8, seqlen_q=8 * 5000, max_seqlen_q=5000
        )
        # group mode: seqlen_k is the total over the batch
        self.assertIn(
            run(seqlen_k=8 * 256, **group), ids("qr_tdm_pingpong", "fast_gelu", 128)
        )
        self.assertIn(run(seqlen_k=8 * 4096, **group), ids("qr_tdm_sched", "fast_gelu"))

    def test_gdpa_and_softmax_never_cross(self):
        run, ids = self.dispatcher()
        gdpa = set().union(
            *(
                ids(p, "fast_gelu")
                for p in ("qr_tdm", "qr_tdm_pingpong", "qr_tdm_sched")
            )
        )
        big = dict(batch=1280, nhead_q=2, nhead_k=2)
        for shape in (
            dict(seqlen_q=532, seqlen_k=256),
            dict(seqlen_q=532, seqlen_k=4096),
        ):
            self.assertIn(run(activation=1, **shape, **big), gdpa)
            self.assertNotIn(run(activation=0, **shape, **big), gdpa)
            self.assertNotIn(run(activation=0, lse=1, **shape, **big), gdpa)

    def test_gdpa_rejects_softmax_only_features(self):
        run, _ = self.dispatcher()
        for feature in (
            dict(lse=1),
            dict(mask=1, window_size_left=-1, window_size_right=0),
        ):
            self.assertLess(run(activation=1, seqlen_q=532, seqlen_k=256, **feature), 0)


class TestCompiledGfx125ForwardOptionDispatch(unittest.TestCase):
    """broadcast_q and lse_log2 reach only the kernels that serve them."""

    SHAPE = dict(hdim_q=128, hdim_v=128, seqlen_q=512, seqlen_k=1024)

    def dispatcher(self, name_glob):
        pool, kernels = get_fwd_blobs(["gfx1250"], name_glob, 0, [128], "simplified")
        return _CompiledDispatcher(self, pool, kernels)

    def test_lse_log2_only_on_kernels_that_store_it(self):
        # skip included: only qr serves it
        dispatch = self.dispatcher(
            "fmha_fwd_d128_bf16_batch*_nlogits*_nbias*_nmask*_nqscale*_ntrload*_nsink*"
        )
        qr_tdm_family = {
            index
            for kernel, index in dispatch.kernel_ids
            if kernel.F_pipeline.tag.startswith("qr_tdm")
        }
        (selected,) = dispatch.run(lse=1, lse_log2=1, **self.SHAPE)
        self.assertIn(selected, qr_tdm_family)
        # dropout runs on the qr_tdm family too, which stores the base-2 LSE itself
        (selected,) = dispatch.run(lse=1, dropout=1, lse_log2=1, **self.SHAPE)
        self.assertIn(selected, qr_tdm_family)
        # skip runs on qr only, which stores the natural log: no kernel takes lse_log2 there
        # (the bindings rescale the LSE instead)
        self.assertGreaterEqual(dispatch.run(lse=1, skip=1, **self.SHAPE)[0], 0)
        self.assertLess(dispatch.run(lse=1, skip=1, lse_log2=1, **self.SHAPE)[0], 0)
        # without an LSE to store, lse_log2 changes nothing
        self.assertEqual(
            dispatch.run(lse_log2=1, **self.SHAPE), dispatch.run(**self.SHAPE)
        )

    def test_broadcast_q_only_in_group_mode(self):
        dispatch = self.dispatcher(
            "fmha_fwd_d128_bf16_*_nlogits*_nbias*_nmask*_nlse*_ndropout*_nskip*_nqscale*_ntrload*_nsink*"
        )
        group = dispatch.run(group=1, **self.SHAPE)
        self.assertGreaterEqual(group[0], 0)
        self.assertEqual(dispatch.run(group=1, broadcast_q=1, **self.SHAPE), group)
        self.assertGreaterEqual(dispatch.run(**self.SHAPE)[0], 0)
        self.assertLess(dispatch.run(broadcast_q=1, **self.SHAPE)[0], 0)


if __name__ == "__main__":
    unittest.main()
