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
from dataclasses import replace
import sys

_CK_ROOT = Path(__file__).resolve().parents[3]
_FMHA_EXAMPLE_DIR = _CK_ROOT / "example" / "ck_tile" / "01_fmha"
sys.path.insert(0, str(_FMHA_EXAMPLE_DIR))

from codegen.ops.fmha_fwd_splitkv import get_fwd_splitkv_blobs  # noqa: E402
from codegen.ops.fmha_fwd import (  # noqa: E402
    CppConstraint,
    FMHA_FWD_API_FOOTER,
    FMHA_FWD_API_HEADER,
    FmhaFwdApiPool,
    FmhaFwdTileSize,
    GFX125_GDPA_TILE_OWNER,
    GFX125_QR_TDM_PINGPONG_HDIMS,
    GFX125_QR_TDM_PINGPONG_MX_64_KEY_TRIPS,
    GFX125_QR_TDM_SCHED_SMALL_GRID,
    KernelComponentFactoryGfx125,
    KernelContext,
    ProblemContext,
    _gfx125_qr_tdm_pingpong_tiles,
    _validate_gfx125_tile_dispatch_order,
    get_fwd_blobs,
    write_fwd_api,
)


def _short_q_bn0(keys, hdim_pair=(128, 128), dtype="bf16"):
    """bn0 (keys a trip) of the qr_tdm_pingpong 16-row tile dispatch takes for `keys` keys a
    sequence: the longest trip they fill, else the shortest."""
    trips = sorted(
        t.F_bn0
        for t in _gfx125_qr_tdm_pingpong_tiles(
            hdim_pair, fp8=dtype in ("fp8", "fp8bf16"), mx=dtype.startswith("mx")
        )
        if t.F_bm0 == 16
    )
    return max([b for b in trips[1:] if keys >= b] or trips[:1])


_SUPPORTED_FEATURE_FILTER = (
    "fmha_fwd_d*_bf16_*_nlogits*_nbias*_nmask*_nlse*_ndropout*_nskip*_nqscale*"
)
_SCHED_PIPELINE = "qr_tdm_sched"
# The scheduled pipelines: d128 only, at their input types, for small grids.
# gfx125's scheduled pipelines: qr_tdm_fp8_sched at fp8 (d128 small grids) and MX (prefill at
# d64/d128/d256); fp16/bf16 run qr_tdm_pingpong only
_SCHED_DTYPES = {
    "qr_tdm_fp8_sched": ("fp8", "fp8bf16", "mxfp8", "mxfp8bf16"),
}
_SCHED_HDIMS = {
    "fp8": {128},
    "fp8bf16": {128},
    "mxfp8": {64, 128, 256},
    "mxfp8bf16": {64, 128, 256},
}
_GFX125_DTYPES = ("fp16", "bf16", "fp8", "fp8bf16", "mxfp8", "mxfp8bf16")
_D192_PIPELINE = "qr_tdm_pingpong"
_HOST_CXX = shutil.which("c++")
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
    logits="f",
    bias="no",
    dropout="f",
    sink="f",
    vlayout="row",
    qscale="no",
    skip="f",
    packgqa="f",
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
                                  QRKSVS_TDM_KVSHARED, QRKSVS_TDM_SCHED_FP8 };
enum class BlockAttentionBiasEnum { NO_BIAS, ELEMENTWISE_BIAS, ALIBI };
enum class BlockAttentionQuantScaleEnum { NO_SCALE, PERTENSOR, BLOCKSCALE, KV_BLOCKSCALE, MX, PERHEAD };
enum class BlockAttentionActivationEnum { SOFTMAX, FAST_GELU };
}
enum class mask_enum { no_mask, mask_top_left, mask_bottom_right, window_generic };
enum class bias_enum { no_bias, elementwise_bias, alibi };
enum class quant_scale_enum { no_scale, pertensor, blockscale, kv_blockscale, mx, perhead };
struct FmhaFwdBf16 {};
struct FmhaFwdFp8 {};
struct FmhaFwdFp16 {};
struct FmhaFwdFp8Bf16 {};
struct FmhaFwdMxFp8 {};
struct FmhaFwdMxFp8Bf16 {};
struct FmhaMasks {
    struct NoMask {};
    struct GenericMask {};
    struct CausalMask {};
};
inline constexpr ck_tile::index_t fmha_fwd_largest_n_tile_size = 128;
template <int, typename, bool, int, int, int, int, int, int, bool, auto, bool,
          typename, auto, bool, bool, auto, bool, bool, bool, bool, bool, bool, bool, int = -1,
          ck_tile::BlockAttentionActivationEnum = ck_tile::BlockAttentionActivationEnum::SOFTMAX,
          bool = false>
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
    int max_seqlen_k = 0;
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
    int block_scale_size_q = 0, block_scale_size_kv = 0;
    // MX E8M0 descales (qr_tdm_fp8_sched's alignment checks): aligned by default
    const void* q_descale_ptr = nullptr;
    const void* k_descale_ptr = nullptr;
    int stride_q_descale = 4, stride_k_descale = 4;
    int nhead_stride_q_descale = 512, nhead_stride_k_descale = 512;
    int batch_stride_q_descale = 4096, batch_stride_k_descale = 4096;
};
@PACK_GQA_COSTS_PARALLELISM@
// fmha_fwd.hpp's check, on the fields modelled here (no cu_seqlen_q, foldable LSE/descales)
inline bool fmha_fwd_can_merge_head_groups(const fmha_fwd_args& a, int max_rows, bool, bool,
                                           bool allow_one_group = false) {
    if(a.nhead_k <= 0 || a.nhead_q < a.nhead_k || a.nhead_q % a.nhead_k != 0) return false;
    if(!allow_one_group && a.nhead_q == a.nhead_k) return false;
    if(a.cu_seqlen_k_ptr != nullptr) return false;
    const int s = a.seqlen_q;
    if(s != a.max_seqlen_q || s * (a.nhead_q / a.nhead_k) > max_rows) return false;
    return s == 1 || (a.nhead_stride_q == s * a.stride_q && a.nhead_stride_o == s * a.stride_o);
}
template <typename Trait, typename Arch> struct dispatch_id;
template <typename Trait, typename Arch>
float fmha_fwd_(const ck_tile::stream_config&, const fmha_fwd_args&) {
    return dispatch_id<Trait, Arch>::value;
}
"""

# The real dispatch rule, taken from fmha_fwd.hpp (it uses only ck_tile::index_t).
_HOST_FMHA_STUB = _HOST_FMHA_STUB.replace(
    "@PACK_GQA_COSTS_PARALLELISM@",
    re.search(
        r"^inline bool fmha_fwd_pack_gqa_costs_parallelism\(.*?^\}\n",
        (_FMHA_EXAMPLE_DIR / "fmha_fwd.hpp").read_text(),
        re.M | re.S,
    ).group(0),
)

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
         >> broadcast_q >> lse_log2 >> a.block_scale_size_q
         >> a.block_scale_size_kv)) return 2;
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


def _tags_without_kvshared(kernels):
    """The pipeline tags of kernels but qr_tdm_kvshared's (K and V one tensor), which may only
    run fp16/bf16 d128/d256 on its own tiles."""
    tags = set()
    for k in kernels:
        if k.F_pipeline.tag == "qr_tdm_kvshared":
            assert k.F_dtype in ("fp16", "bf16") and k.F_hdim in (128, 256), k.name
            assert k.F_tile.F_pipeline_tag == "qr_tdm_kvshared", k.name
        else:
            tags.add(k.F_pipeline.tag)
    return tags


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
            block_scale_size_q=0,
            block_scale_size_kv=0,
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
        fp8_sched_source = (
            "fmha_fwd_d128_fp8bf16_batch_b128x128x128x128x128x128_"
            "r4x1x1_r4x1x1_w16x16x128_w16x16x128_o2_qr_tdm_fp8_sched_"
            "vr_pssk_nlogits_nbias_nmask_nlse_ndropout_nskip_pertensor_"
            "ntrload_npackgqa_nsink_gfx125.cpp"
        )
        fp8_tdm_source = fp8_sched_source.replace(
            "_qr_tdm_fp8_sched_", "_qr_tdm_"
        ).replace("_npackgqa_", "_")
        mx_sched_source = fp8_sched_source.replace("_fp8bf16_", "_mxfp8_").replace(
            "_pertensor_", "_mx_"
        )
        sources = (
            _D192_BATCH_NMASK_FILENAME,
            _D192_BATCH_M16_FILENAME,
            _D192_BATCH_NMASK_FILENAME.replace("_bf16_", "_fp16_"),
            d192_lse_source,
            _D192_BATCH_MASK_FILENAME,
            _D192_GROUP_NMASK_FILENAME,
            d256_source,
            fp8_sched_source,
            mx_sched_source,
            fp8_tdm_source,
        )

        def configure(hip_version=None):
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
{f"set(CMAKE_HIP_COMPILER_VERSION {hip_version})" if hip_version else ""}
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

        for hip_version in (None, "24.0.0", "23.0.0"):
            commands = configure(hip_version)
            # Clang before 24 builds MX fp8_sched instances without the expert mode.
            no_expert = (d256_source, fp8_tdm_source) + (
                (mx_sched_source,) if hip_version == "23.0.0" else ()
            )
            for source in sources:
                with self.subTest(hip_version=hip_version, source=source):
                    self.assertNotIn("-wwm-regalloc=fast", commands[source])
                    self.assertEqual(
                        "-amdgpu-expert-scheduling-mode" in commands[source],
                        source not in no_expert,
                    )

    @unittest.skipUnless(shutil.which("cmake"), "cmake is required")
    def test_cmake_fp8_sched_rule_scope(self):
        # The qr_tdm_fp8_sched-only flags must reach none of the qr_tdm_pingpong fp8
        # sources codegen emits for gfx125, and follow their rules on the qr_tdm_fp8_sched
        # ones. Codegen emits qr_tdm_fp8_sched at d128 only (no MX), so the rules for the
        # other head dims and MX are checked on codegen's pingpong names with the pipeline
        # tag swapped: a naming change in the rest of the name cannot silently disable one.
        helper = (_FMHA_EXAMPLE_DIR / "cmake") / "fmha_fwd_source_flags.cmake"
        _, kernels = get_fwd_blobs(
            ["gfx1250"], "*fp8*", 0, [64, 128, 192, 256], "simplified"
        )
        self.assertEqual(
            {k.F_pipeline.tag for k in kernels}, {"qr_tdm_pingpong", "qr_tdm_fp8_sched"}
        )
        sources = [
            (
                k,
                k.filename,
                "sched" if k.F_pipeline.tag == "qr_tdm_fp8_sched" else "pingpong",
            )
            for k in kernels
        ] + [
            (
                k,
                k.filename.replace("_qr_tdm_pingpong_", "_qr_tdm_fp8_sched_"),
                "renamed",
            )
            for k in kernels
            if k.F_pipeline.tag == "qr_tdm_pingpong"
        ]

        rules = {
            "-amdgpu-codegenprepare-force-break-large-phis": lambda k: (
                k.F_pipeline.F_trload == "f"
                and (
                    k.F_pipeline.F_qscale == "blockscale"
                    or (
                        k.F_hdim == 64
                        and k.F_pipeline.F_qscale in ("no", "pertensor", "perhead")
                    )
                )
            ),
            "-structurizecfg-skip-uniform-regions": lambda k: (
                k.F_hdim in (128, 256) and k.F_dtype.startswith("mxfp8")
            ),
            "-regalloc-enable-priority-advisor=dummy": lambda k: (
                k.F_hdim == 256 and k.F_dtype.startswith("fp8")
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "flags.cmake"
            script.write_text(
                f'include("{helper}")\n'
                + "\n".join(
                    f'ck_tile_fmha_fwd_get_source_compile_options("{name}" flags)\n'
                    f'message("{i}:${{flags}}")'
                    for i, (_, name, _) in enumerate(sources)
                )
            )
            process = subprocess.run(
                ["cmake", "-P", str(script)],
                capture_output=True,
                text=True,
                check=True,
                timeout=240,
            )
        lines = process.stderr.strip().splitlines()
        self.assertEqual(len(lines), len(sources))
        for flag, expected in rules.items():
            seen = Counter()
            for i, ((k, name, kind), line) in enumerate(zip(sources, lines)):
                self.assertTrue(line.startswith(f"{i}:"))
                want = kind != "pingpong" and expected(k)
                seen[(kind, want)] += 1
                with self.subTest(flag=flag, source=name):
                    self.assertEqual(flag in line.split(";"), want)
            # Each rule has positive and negative qr_tdm_fp8_sched names, and no
            # qr_tdm_pingpong source takes it.
            for want in (True, False):
                with self.subTest(flag=flag, positive=want):
                    self.assertGreater(
                        seen[("sched", want)] + seen[("renamed", want)], 0
                    )
            self.assertGreater(seen[("sched", False)], 0)
            self.assertEqual(seen[("pingpong", True)], 0)
            self.assertGreater(seen[("pingpong", False)], 0)

    @unittest.skipUnless(shutil.which("cmake"), "cmake is required")
    def test_cmake_source_flags_follow_gfx125_family_targets(self):
        helper = (_FMHA_EXAMPLE_DIR / "cmake") / "fmha_fwd_source_flags.cmake"
        source = (
            "fmha_fwd_d256_fp8bf16_batch_b128x128x128x256x128x256_"
            "r4x1x1_r4x1x1_w16x16x128_w16x16x128_o1_qr_tdm_fp8_sched_"
            "vr_pssk_nlogits_nbias_nmask_nlse_ndropout_nskip_pertensor_"
            "ntrload_npackgqa_nsink_gfx125.cpp"
        )
        for targets, applied in (
            ("gfx1250-strict", True),
            ("gfx1250", True),
            ("gfx1251", True),
            ("gfx950", False),
            ("gfx1201", False),
        ):
            with self.subTest(targets=targets), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                (root / "CMakeLists.txt").write_text(
                    f"""cmake_minimum_required(VERSION 3.20)
project(fmha_source_flags LANGUAGES NONE)
include(\"{helper}\")
set(source \"${{CMAKE_CURRENT_LIST_DIR}}/{source}\")
ck_tile_fmha_fwd_apply_source_compile_options(\"{targets}\" \"${{source}}\")
get_source_file_property(options \"${{source}}\" COMPILE_OPTIONS)
message(\"options:${{options}}\")
"""
                )
                process = subprocess.run(
                    ["cmake", "-S", str(root), "-B", str(root / "build")],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                options = re.search(r"^options:(.*)$", process.stderr, re.M)
                self.assertIsNotNone(options)
                for flag in (
                    "-amdgpu-expert-scheduling-mode",
                    "-regalloc-enable-priority-advisor=dummy",
                ):
                    self.assertEqual(flag in options[1].split(";"), applied)

    def test_receipts_emit_d192_pingpong_tiles_only(self):
        for receipt in (100, 200, 600):
            with self.subTest(receipt=receipt):
                _, kernels = _generate(receipt)
                # qr_tdm_sched emits no d192 instances any more.
                self.assertFalse(any(_is_sched_kernel(k) for k in kernels))
                # the unpacked instances (packed-GQA decode: test_pingpong_packgqa_decode)
                family = [
                    kernel
                    for kernel in kernels
                    if kernel.F_pipeline.tag == _D192_PIPELINE
                    and kernel.F_hdim == 192
                    and kernel.F_pipeline.F_packgqa == "f"
                ]
                candidates = [k for k in family if _has_base_features(k)]

                self.assertTrue(candidates)
                # The kernel filter leaves the sink tokens as the only extra feature: every
                # base instance on the decode and general tiles has a sink twin, the 64-row
                # tile's none (qr_tdm_pingpong_tile_takes).
                with_sink = [k for k in candidates if k.F_tile.F_bm0 != 64]
                self.assertTrue(with_sink)
                self.assertEqual(
                    Counter(k.F_pipeline.F_sink for k in family)["t"], len(with_sink)
                )
                self.assertEqual(len(family), len(candidates) + len(with_sink))
                for kernel in family:
                    self.assertEqual(kernel.F_arch.name, "gfx125")
                    self.assertEqual(kernel.F_dtype, "bf16")
                    self.assertIn(
                        (kernel.F_tile.F_bm0, kernel.F_tile.F_bn0),
                        {(16, 128), (64, 64), (128, 64)},
                    )
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
                    if kernel not in family and kernel.F_pipeline.F_packgqa == "f":
                        self.assertNotRegex(kernel.filename, _D192_ESM2_FILENAME)

                # qr_tdm_pingpong is all gfx125 generates: no qr_tdm or generic qr
                # fallback is left behind it.
                self.assertEqual(_tags_without_kvshared(kernels), {"qr_tdm_pingpong"})
                self.assertEqual(
                    {k.F_mode for k in candidates},
                    {100: {"batch"}, 200: {"group"}, 600: {"batch", "group"}}[receipt],
                )
                # Each mode has its 16-, 64- and 128-row tile; the 64-row one has no
                # head-dim-padded instances (the general 128-row and the decode tile take
                # those: qr_tdm_pingpong_tile_takes).
                by_tile = Counter(k.F_tile.F_bm0 for k in candidates)
                self.assertEqual(set(by_tile), {16, 64, 128})
                self.assertEqual(by_tile[16], by_tile[128])
                self.assertGreater(by_tile[128], by_tile[64])
                self.assertFalse(
                    [
                        k
                        for k in candidates
                        if k.F_tile.F_bm0 == 64 and k.F_pipeline.F_dpad == "t"
                    ]
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

    def test_d128_emits_only_pingpong_tiles(self):
        # fp16/bf16 d128: qr_tdm_pingpong's tiles take every query, the short-query ones (128
        # or 256 keys a trip), the 64-row one, the 128-row 128-key-trip one (long rows without
        # a left window) and the 128-row one; no qr_tdm_sched (small grids included) and no
        # qr_tdm instance.
        tiles = {(t.F_bm0, t.F_bn0) for t in _gfx125_qr_tdm_pingpong_tiles((128, 128))}
        self.assertEqual(tiles, {(16, 128), (16, 256), (64, 64), (128, 128), (128, 64)})
        for receipt in (100, 200, 600):
            with self.subTest(receipt=receipt):
                _, kernels = _generate(receipt, [128])
                self.assertTrue(kernels)
                # besides qr_tdm_kvshared (K and V one tensor) on its own tiles
                self.assertEqual(
                    {k.F_pipeline.tag for k in kernels},
                    {"qr_tdm_pingpong", "qr_tdm_kvshared"},
                )
                self.assertTrue(all(k.F_hdim == 128 for k in kernels))
                pingpong = [k for k in kernels if k.F_pipeline.tag == "qr_tdm_pingpong"]
                self.assertEqual(
                    {(k.F_tile.F_bm0, k.F_tile.F_bn0) for k in pingpong}, tiles
                )
                kvshared_tiles = {
                    t.name
                    for t in KernelComponentFactoryGfx125._kvshared_tiles()[(128, 128)]
                }
                self.assertTrue(
                    all(
                        k.F_tile.name in kvshared_tiles
                        for k in kernels
                        if k.F_pipeline.tag == "qr_tdm_kvshared"
                    )
                )
                self.assertEqual(len(kernels), len({k.filename for k in kernels}))
                for kernel in kernels:
                    source = kernel.render()
                    self.assertNotIn("QRKSVS_TDM_SCHED", source)
                    self.assertIn(
                        "QRKSVS_TDM_KVSHARED"
                        if kernel.F_pipeline.tag == "qr_tdm_kvshared"
                        else "QRKSVS_TDM_PINGPONG",
                        source,
                    )

    def test_generated_api_renders_only_pingpong_entries(self):
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
                # qr_tdm_pingpong's entries are all there is: no generic qr or qr_tdm
                # entry follows them.
                self.assertNotIn("BlockFmhaPipelineEnum::QRKSVS,", api)
                self.assertNotIn("BlockFmhaPipelineEnum::QRKSVS_TDM,", api)
                d192 = api[api.index("t.hdim_q <= 192") : api.index("t.hdim_q <= 256")]
                self.assertIn("QRKSVS_TDM_PINGPONG", d192)
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

    def test_filtered_feature_emits_pingpong_only(self):
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
                    receipt=200 if feature == "skip" else 100,
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
                # d192 qr_tdm_pingpong takes the soft cap, sink tokens, an elementwise
                # bias and (group mode) skip_min_seqlen_q; d192 has no dropout instances
                # at all, d256's pingpong takes it.
                self.assertEqual(
                    bool(family), feature in ("logits", "sink", "bias", "skip")
                )
                for kernel in family:
                    self.assertEqual(
                        getattr(kernel.F_pipeline, "F_" + feature),
                        "bias" if feature == "bias" else "t",
                    )
                    if feature == "skip":
                        self.assertEqual(kernel.F_mode, "group")
                if feature == "dropout":
                    self.assertTrue(all(k.F_hdim == 256 for k in kernels))
                # No other pipeline is generated next to it (but qr_tdm_kvshared, K and V one
                # tensor).
                self.assertEqual(_tags_without_kvshared(kernels), {"qr_tdm_pingpong"})
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

    def test_scheduled_pipelines_are_gfx125_only(self):
        # qr_tdm_fp8_sched: gfx125 family targets only, at fp8 (d128) and MX (d64/d128/d256),
        # no packed GQA; no qr_tdm_sched at all; the other targets have no pingpong.
        for target in ("gfx1250", "gfx1251", "gfx1252"):
            with self.subTest(target=target, tag="qr_tdm_sched"):
                _, kernels = get_fwd_blobs(
                    targets=[target],
                    kernel_filter="*_qr_tdm_sched_*",
                    receipt=0,
                    optdim_list=[-1],
                    mask_impl="simplified",
                )
                self.assertFalse(kernels)
        for target in ("gfx942", "gfx950", "gfx1250", "gfx1251", "gfx1252"):
            for tag, dtypes in _SCHED_DTYPES.items():
                with self.subTest(target=target, tag=tag):
                    _, kernels = get_fwd_blobs(
                        targets=[target],
                        kernel_filter=f"*_{tag}_*",
                        receipt=0,
                        optdim_list=[-1],
                        mask_impl="simplified",
                    )
                    if not target.startswith("gfx125"):
                        self.assertFalse(kernels)
                        continue
                    self.assertTrue(kernels)
                    self.assertEqual({k.F_pipeline.tag for k in kernels}, {tag})
                    self.assertEqual({k.F_dtype for k in kernels}, set(dtypes))
                    for dtype in dtypes:
                        self.assertEqual(
                            {
                                (k.F_hdim, k.F_tile.F_bn1)
                                for k in kernels
                                if k.F_dtype == dtype
                            },
                            {(h, h) for h in _SCHED_HDIMS[dtype]},
                        )
                    self.assertEqual({k.F_pipeline.F_packgqa for k in kernels}, {"f"})
                    self.assertEqual({k.F_pipeline.F_skip for k in kernels}, {"f"})
                    self.assertEqual({k.F_pipeline.F_vlayout for k in kernels}, {"row"})
        for target in ("gfx942", "gfx950"):
            with self.subTest(target=target):
                _, kernels = get_fwd_blobs(
                    targets=[target],
                    kernel_filter="*_qr_tdm_pingpong_*",
                    receipt=100,
                    optdim_list=[128, 192],
                    mask_impl="simplified",
                )
                self.assertFalse(kernels)
        for target in ("gfx1250", "gfx1251", "gfx1252"):
            with self.subTest(target=target):
                _, kernels = get_fwd_blobs(
                    targets=[target],
                    kernel_filter="*_bf16_*",
                    receipt=100,
                    optdim_list=[128, 192],
                    mask_impl="simplified",
                )
                self.assertTrue(kernels)
                self.assertEqual({k.F_hdim for k in kernels}, {128, 192})
                self.assertEqual(
                    {k.F_pipeline.tag for k in kernels if k.F_hdim == 192},
                    {"qr_tdm_pingpong"},
                )
                self.assertEqual(
                    _tags_without_kvshared(k for k in kernels if k.F_hdim == 128),
                    {"qr_tdm_pingpong"},
                )

    def test_d192_m128_emits_batch_and_group_for_every_mask(self):
        mask_options = {
            "simplified": {
                (mask, lse)
                for mask in ("s_no", "s_mask", "causal")
                for lse in ("f", "t")
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
        dispatch,
        mode,
        mask="s_no",
        lse="f",
        bm0=128,
        dtype="bf16",
        sink="f",
        dpad="f",
        packgqa="f",
    ):
        # qr_tdm_pingpong serves every d192 query.
        return dispatch.expected_id(
            packgqa=packgqa,
            mode=mode,
            dtype=dtype,
            hdim=192,
            pipeline="qr_tdm_pingpong",
            bm0=bm0,
            mask=mask,
            lse=lse,
            # (a padded head dim takes the fully padded instance: qr_tdm_pingpong_tile_takes)
            spad="t" if mode == "group" or dpad == "t" else "f",
            dpad=dpad,
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
            spad="t" if mode == "group" or dpad == "t" else "f",
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
                # one query per head packs the 4 query heads of a KV head
                packgqa = "t" if length == 1 and not group else "f"
                self.assertEqual(
                    dispatch.run(group=group, seqlen_q=length),
                    [self.candidate_id(dispatch, mode, bm0=bm0, packgqa=packgqa)],
                )
            self.assertEqual(
                dispatch.run(group=group, hdim_q=160, hdim_v=128),
                [self.candidate_id(dispatch, mode, dpad="t")],
            )
            # Head dims off the 8-element vector take its padded instance too (O is
            # stored an element at a time); past 192 the d256 bucket's padded one.
            for q, v in ((191, 128), (192, 127)):
                with self.subTest(mode=mode, shape=(q, v)):
                    self.assertEqual(
                        dispatch.run(group=group, hdim_q=q, hdim_v=v),
                        [self.candidate_id(dispatch, mode, dpad="t")],
                    )
            self.assertEqual(
                dispatch.run(group=group, hdim_q=193, hdim_v=128),
                [self.d256_id(dispatch, mode, dpad="t")],
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
        # No scheduled pipeline is generated: qr_tdm_pingpong's 128-row tile takes
        # every d192 query from 128 rows on, whatever the key length, grid or mask.
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

    def test_n128_generic_no_mask_and_causal_dispatch(self):
        dispatch, _ = self.compile(
            kernel_filter=(
                "fmha_fwd_d*_bf16_*_nlogits*_nbias*_*lse*_ndropout*_nskip*_nqscale*"
            ),
            mask_impl="generic",
        )
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
        # Narrower heads, of whole vectors or not, take its padded instance; a wider
        # one is left to the (filtered out) d256 bucket.
        for arguments in ({"hdim_q": 128}, {"hdim_q": 191}):
            self.assertEqual(
                family.run(**arguments), [self.candidate_id(family, "batch", dpad="t")]
            )
        self.assertEqual(family.run(hdim_q=193), [-1])
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
        # (qr_tdm_kvshared takes K and V as one tensor only)
        self.assertEqual(
            {k.F_pipeline.tag for k in kernels if k.F_hdim == 128},
            {"qr_tdm_pingpong", "qr_tdm_kvshared"},
        )
        for mode, group in (("batch", 0), ("group", 1)):
            # qr_tdm_pingpong takes every d128 query, small grids included: up to 16 rows
            # its 16-row tile, below 128 its 64-row one, else (no left window) its 128-row
            # 128-key-trip one at any key length.
            for length, seqlen_k, bm0, bn0 in (
                (1, 128, 16, None),
                (16, 128, 16, None),
                (17, 128, 64, 64),
                (127, 128, 64, 64),
                (128, 128, 128, 128),
                (2047, 128, 128, 128),
                (4095, 4096, 128, 128),
                (4096, 128, 128, 128),
                (4096, 8192, 128, 128),
                (32768, 128, 128, 128),
            ):
                extra = {"bn0": bn0}
                if bm0 == 16:
                    # group mode compares the total keys against the batch's
                    extra["bn0"] = _short_q_bn0(seqlen_k // 2 if group else seqlen_k)
                # one query per head: the 4 query heads of a KV head pack into a tile
                if length == 1 and not group:
                    extra["packgqa"] = "t"
                selected = dispatch.expected_id(
                    mode=mode,
                    hdim=128,
                    pipeline="qr_tdm_pingpong",
                    bm0=bm0,
                    **extra,
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
                        seqlen_k=seqlen_k,
                    ),
                    [selected],
                    (mode, length, seqlen_k),
                )
        # 32 query heads per KV head overflow the 16-row tile: packed GQA moves to the
        # 64-row tile once the unpacked heads fill the grid twice over (batch x nhead_q >=
        # 2 x num_cus), else the heads stay unpacked on the 16-row tile.
        mqa = dict(hdim_q=128, hdim_v=128, seqlen_q=1, nhead_q=64, nhead_k=2)
        features = dict(mode="batch", hdim=128, pipeline="qr_tdm_pingpong", mask="s_no")
        features.update(lse="f", sink="f", dpad="f", spad="f")
        self.assertEqual(
            dispatch.run(batch=8, **mqa),
            [dispatch.expected_id(bm0=64, packgqa="t", **features)],
        )
        self.assertEqual(
            dispatch.run(batch=7, **mqa),
            [dispatch.expected_id(bm0=16, bn0=_short_q_bn0(128), **features)],
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
                if changes == dict(skip=1):
                    self.assertEqual(dispatch.run(**changes), dispatch.run())
                else:
                    self.assertEqual(dispatch.run(**changes), [-1])

        for device in ("gfx1251", "gfx1252"):
            for mode, group in (("batch", 0), ("group", 1)):
                with self.subTest(device=device, mode=mode):
                    self.assertEqual(
                        dispatch.run(device=device, group=group),
                        [self.candidate_id(dispatch, mode)],
                    )

    def test_real_dispatch_missing_candidate_does_not_gate_generic(self):
        # Without d192 instances (filtered at render time or at generation) a d192 query
        # falls through to the next bucket that takes it, d256's padded instance.
        for render_filter in (True, False):
            with self.subTest(render_filter=render_filter):
                dispatch, _ = self.compile(
                    kernel_filter=(
                        _SUPPORTED_FEATURE_FILTER
                        if render_filter
                        else _SUPPORTED_FEATURE_FILTER.replace("d*_", "d256_", 1)
                    ),
                    filter_fn=(
                        (lambda trait: str(trait.hdim) != "192")
                        if render_filter
                        else None
                    ),
                )
                for mode, group in (("batch", 0), ("group", 1)):
                    self.assertEqual(
                        dispatch.run(group=group),
                        [self.d256_id(dispatch, mode, dpad="t")],
                    )

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
                    self.assertEqual(
                        dispatch.run(group=group, hdim_q=191),
                        [self.candidate_id(dispatch, mode, dpad="t")],
                    )
                    self.assertEqual(dispatch.run(group=group, hdim_q=193), [-1])
                    # skip takes effect in group mode only, on pingpong's skip
                    # instances; batch mode reuses the instances without it
                    self.assertEqual(
                        dispatch.run(group=group, skip=1),
                        [
                            dispatch.expected_id(
                                mode=mode,
                                hdim=192,
                                pipeline="qr_tdm_pingpong",
                                bm0=128,
                                mask="s_no",
                                lse="f",
                                spad="t",
                                dpad="f",
                                sink="f",
                                skip="t",
                            )
                        ]
                        if group
                        else dispatch.run(group=group),
                    )
                # fp16/bf16 have no scheduled pipeline left to filter on
                _, sched_kernels = get_fwd_blobs(
                    ["gfx1250"],
                    f"*bf16*{_SCHED_PIPELINE}*",
                    receipt,
                    [-1],
                    "simplified",
                )
                self.assertFalse(sched_kernels)

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
                    # qr_tdm_pingpong only (fp16/bf16 have no scheduled pipeline), at
                    # every requested head dim it serves
                    self.assertEqual(
                        _tags_without_kvshared(kernels), {"qr_tdm_pingpong"}
                    )
                    all_dims = {h for h, _ in GFX125_QR_TDM_PINGPONG_HDIMS}
                    self.assertEqual(
                        {kernel.F_hdim for kernel in kernels},
                        all_dims if optdims == [-1] else all_dims.intersection(optdims),
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

    def test_dimension_generic_and_unsupported_filters_are_deterministic(self):
        for targets in (["gfx1250"], ["gfx942", "gfx950", "gfx1250"]):
            for dimensions in self.dimension_cases():
                for pattern in (
                    "*_bf16_*_qr_vr_*",
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
                        if targets == ["gfx1250"] or pattern == "*not_a_kernel*":
                            self.assertFalse(kernels)

    def test_d64_geometry_and_padding(self):
        for dtype in ("fp16", "bf16"):
            pool, kernels = get_fwd_blobs(
                ["gfx1250"], f"*d64_{dtype}*", 600, [64], "simplified"
            )
            # qr_tdm_pingpong serves every d64 query, skip (group mode) and head dims
            # off the vector included; no qr_tdm, qr_tdm_sched or generic qr is
            # generated.
            self.assertEqual({k.F_pipeline.tag for k in kernels}, {"qr_tdm_pingpong"})
            self.assertEqual(
                {k.F_mode for k in kernels if k.F_pipeline.F_skip == "t"}, {"group"}
            )
            pingpong = [
                k
                for k in kernels
                if k.F_pipeline.tag == "qr_tdm_pingpong" and _has_base_features(k)
            ]
            self.assertEqual({k.F_tile.F_bm0 for k in pingpong}, {16, 64, 128})
            self.assertEqual({k.F_mode for k in pingpong}, {"batch", "group"})
            for kernel in pingpong:
                # the short-query tile's trips of 128 or 512 keys (test_pingpong_decode_tiles)
                self.assertIn(
                    kernel.F_tile.F_bn0,
                    (128, 512) if kernel.F_tile.F_bm0 == 16 else (64,),
                )
                self.assertEqual(kernel.F_tile.F_bn1, 64)
                self.assertEqual(kernel.F_tile.F_bk0max, 64)
                trait = kernel.api_trait()
                if kernel.F_mode == "group":
                    self.assertEqual(trait.spad, "t")
                    self.assertEqual(trait.skpad, "t")
                elif trait.skpad == "f":
                    # The unpadded batch instances take any K length.
                    self.assertEqual(trait.skcheck, "true")
                # the padded fp16/bf16 instances take any head dim
                self.assertEqual(
                    trait.dcheck,
                    "a.hdim_q % 64 == 0" if trait.dpad == "f" else "true",
                )
                self.assertEqual(
                    trait.dvcheck, "a.hdim_v == 64" if trait.dvpad == "f" else "true"
                )
            self.assertIn("a.hdim_v == 64", pool.render("test"))

    def test_inventory_is_pingpong_and_fp8_sched_for_every_dtype(self):
        # gfx125 generates qr_tdm_pingpong at every input type and every head-dim pair it
        # serves, and besides it only qr_tdm_fp8_sched (fp8 d128, MX d64/d128/d256; none at
        # 16 bits): no qr_tdm or generic qr.
        factory = KernelComponentFactoryGfx125
        self.assertEqual(set(factory.supported_dtypes()), set(_GFX125_DTYPES))
        qscales = {
            "fp16": {"no"},
            "bf16": {"no"},
            "fp8": {"no", "pertensor", "perhead", "blockscale"},
            "fp8bf16": {"no", "pertensor", "perhead", "blockscale"},
            "mxfp8": {"mx"},
            "mxfp8bf16": {"mx"},
        }
        for dtype in _GFX125_DTYPES:
            sched_tags = {tag for tag, dts in _SCHED_DTYPES.items() if dtype in dts}
            sched_hdims = {(h, h) for h in _SCHED_HDIMS.get(dtype, ())}
            with self.subTest(dtype=dtype):
                tiles = factory.get_hdim_tile_size_dict(dtype)
                self.assertEqual(set(tiles), set(GFX125_QR_TDM_PINGPONG_HDIMS))
                for hdim_pair, hdim_tiles in tiles.items():
                    # fp16/bf16: qr_tdm_kvshared's tiles first at d128/d256, GDPA's own tile
                    # last at d128
                    kvshared = [
                        t for t in hdim_tiles if t.F_pipeline_tag == "qr_tdm_kvshared"
                    ]
                    gdpa = [
                        t
                        for t in hdim_tiles
                        if t.F_pipeline_tag == GFX125_GDPA_TILE_OWNER
                    ]
                    fp16_bf16 = dtype in ("fp16", "bf16")
                    self.assertEqual(
                        bool(kvshared),
                        fp16_bf16 and hdim_pair in ((128, 128), (256, 256)),
                    )
                    self.assertEqual(hdim_tiles[: len(kvshared)], kvshared)
                    self.assertEqual(bool(gdpa), fp16_bf16 and hdim_pair == (128, 128))
                    hdim_tiles = [
                        t for t in hdim_tiles if t not in kvshared and t not in gdpa
                    ]
                    owned = [
                        t for t in hdim_tiles if t.F_pipeline_tag == "qr_tdm_pingpong"
                    ]
                    others = [t for t in hdim_tiles if t not in owned]
                    # pingpong's tiles first, then the scheduled ones
                    self.assertEqual(hdim_tiles[: len(owned)], owned)
                    self.assertEqual(bool(others), hdim_pair in sched_hdims)
                    for t in others:
                        self.assertIsNone(t.F_pipeline_tag)
                        self.assertTrue(factory.is_sched_tile(dtype, t, hdim_pair[0]))
                    # pingpong's last tile takes every query left (no range clause)
                    self.assertEqual(str(owned[-1].F_constraint), "true")
                    pipelines = factory.get_pipelines(
                        dtype, *hdim_pair, 0, "simplified"
                    )
                    self.assertTrue(pipelines)
                    self.assertEqual(
                        {p.tag for p in pipelines} - {"qr_tdm_kvshared"},
                        {"qr_tdm_pingpong"}
                        | (sched_tags if hdim_pair in sched_hdims else set()),
                    )
                    self.assertEqual({p.F_qscale for p in pipelines}, qscales[dtype])
                    # skip_min_seqlen_q variants (group mode): pingpong's only
                    self.assertEqual(
                        {p.F_skip for p in pipelines if p.tag == "qr_tdm_pingpong"},
                        {"f", "t"},
                    )
                    self.assertEqual(
                        {
                            p.F_skip
                            for p in pipelines
                            if p.tag not in ("qr_tdm_pingpong", "qr_tdm_kvshared")
                        },
                        {"f"} if others else set(),
                    )
                _, kernels = get_fwd_blobs(
                    ["gfx1250"], f"*_{dtype}_*", 0, [-1], "simplified"
                )
                # ("_fp8_" also matches fp8bf16's "_qr_tdm_fp8_sched_" names)
                kernels = [k for k in kernels if k.F_dtype == dtype]
                self.assertTrue(kernels)
                for hdim_pair in GFX125_QR_TDM_PINGPONG_HDIMS:
                    self.assertEqual(
                        _tags_without_kvshared(
                            k
                            for k in kernels
                            if (k.F_hdim, k.F_tile.F_bn1) == hdim_pair
                        ),
                        {"qr_tdm_pingpong"}
                        | (sched_tags if hdim_pair in sched_hdims else set()),
                    )
                self.assertEqual(
                    {(k.F_hdim, k.F_tile.F_bn1) for k in kernels},
                    set(GFX125_QR_TDM_PINGPONG_HDIMS),
                )
                self.assertEqual(
                    {k.F_pipeline.F_qscale for k in kernels}, qscales[dtype]
                )
                self.assertEqual(
                    {k.F_mode for k in kernels if k.F_pipeline.F_skip == "t"}, {"group"}
                )
                # fp8 sched: d128 small grids; MX sched: its prefill (more than 16 rows)
                for k in kernels:
                    if k.F_pipeline.tag in sched_tags:
                        constraint = str(k.api_trait().constraint)
                        if dtype.startswith("mx"):
                            self.assertIn("a.max_seqlen_q > 16", constraint)
                            self.assertNotIn(GFX125_QR_TDM_SCHED_SMALL_GRID, constraint)
                        else:
                            self.assertIn(GFX125_QR_TDM_SCHED_SMALL_GRID, constraint)

    def test_mxfp8bf16_mirrors_mxfp8(self):
        def key(k):
            return (
                k.F_mode,
                k.F_tile.name,
                k.F_pipeline.tag,
                k.F_pipeline.F_qscale,
                k.F_pipeline.F_mask,
                k.F_pipeline.F_lse,
                k.F_pipeline.F_sink,
                k.F_pipeline.F_bias,
            )

        traits = {}
        for dtype in ("mxfp8", "mxfp8bf16"):
            pool, kernels = get_fwd_blobs(
                ["gfx1250"], f"*d128_{dtype}_*", 0, [128], "simplified"
            )
            self.assertTrue(all(k.F_dtype == dtype for k in kernels))
            traits[dtype] = Counter(key(k) for k in kernels)
            if dtype == "mxfp8bf16":
                self.assertIn('t.data_type.compare("mxfp8bf16")', pool.render("test"))
                self.assertIn("FmhaFwdMxFp8Bf16", kernels[0].render())
        # Only the O type differs: same pipelines, tiles and feature variants.
        self.assertTrue(traits["mxfp8bf16"])
        self.assertEqual(traits["mxfp8bf16"], traits["mxfp8"])

    def test_mx_on_pingpong_at_fp8_head_dims(self):
        """MX takes qr_tdm_pingpong at its every head dim: PV on 128-deep warp gemms (the
        scaled WMMA's K; the 16-row tile's trip is one per key warp), 64-deep ones zero
        padded in registers on the prefill tiles of GFX125_QR_TDM_PINGPONG_MX_64_KEY_TRIPS,
        whose trips take 64 keys; QK on the widest WMMA K
        dividing the Q/K row (64-deep slices zero padded in registers), rows padded to
        whole 64-deep slices, or 128-deep ones where that takes fewer scaled WMMAs (d160,
        d192: 2 for 3); prefill at d64/d128/d256 also has qr_tdm_fp8_sched."""
        for dtype in ("mxfp8", "mxfp8bf16"):
            for hdim, hdim_v in GFX125_QR_TDM_PINGPONG_HDIMS:
                with self.subTest(dtype=dtype, hdim=hdim):
                    _, kernels = get_fwd_blobs(
                        ["gfx1250"], f"*d{hdim}_{dtype}_*", 0, [hdim], "simplified"
                    )
                    kernels = [k for k in kernels if k.F_hdim == hdim]
                    self.assertEqual(
                        {k.F_pipeline.tag for k in kernels},
                        {"qr_tdm_pingpong"}
                        | (
                            {"qr_tdm_fp8_sched"}
                            if hdim in _SCHED_HDIMS[dtype]
                            else set()
                        ),
                    )
                    pingpong = [
                        k for k in kernels if k.F_pipeline.tag == "qr_tdm_pingpong"
                    ]
                    variants = {
                        (
                            k.F_mode,
                            k.F_pipeline.F_mask,
                            k.F_pipeline.F_lse,
                            k.F_pipeline.F_sink,
                        )
                        for k in pingpong
                        if k.F_pipeline.F_packgqa == "f"
                    }
                    # packed-GQA decode: mask (none, causal, window) x lse x sink
                    self.assertEqual(
                        len(
                            {
                                (
                                    k.F_pipeline.F_mask,
                                    k.F_pipeline.F_lse,
                                    k.F_pipeline.F_sink,
                                )
                                for k in pingpong
                                if k.F_pipeline.F_packgqa == "t"
                            }
                        ),
                        12,
                    )
                    # the causal band variants have no sink (the generic mask serves it)
                    self.assertEqual(len(variants), 2 * 3 * 2 * 2 - 2 * 2)
                    k_row = -(-hdim // 64) * 64
                    if -(-hdim // 128) < k_row // 64:
                        k_row = -(-hdim // 128) * 128
                    qk_k = 128 if k_row % 128 == 0 else 64
                    for k in pingpong:
                        t = k.F_tile
                        pv_k = 128
                        if t.F_bm0 > 16 and (hdim, hdim_v) in (
                            GFX125_QR_TDM_PINGPONG_MX_64_KEY_TRIPS
                        ):
                            pv_k = 64
                        self.assertEqual(k.F_pipeline.F_qscale, "mx")
                        self.assertEqual(
                            (t.F_bk0, t.F_wk0, t.F_wk1), (qk_k, qk_k, pv_k)
                        )
                        self.assertEqual(t.F_bk1, t.F_bn0 if t.F_bm0 == 16 else pv_k)
                        self.assertEqual(t.F_bk1, pv_k * t.F_rk1)
                        self.assertEqual(t.F_bk0max, k_row)
                        self.assertEqual(t.F_bn0 % pv_k, 0)
            # the narrower head dims' branches first, or a wider one shadows them
            pool, _ = get_fwd_blobs(
                ["gfx1250"],
                f"*_{dtype}_*",
                0,
                [h for h, _ in GFX125_QR_TDM_PINGPONG_HDIMS],
                "simplified",
            )
            api = pool.render("test")
            at = [
                api.index(f"t.hdim_q <= {h} && t.hdim_v <= {hv}")
                for h, hv in GFX125_QR_TDM_PINGPONG_HDIMS
            ]
            self.assertEqual(at, sorted(at))

    def test_d192_bias_exception_is_tdm_pingpong(self):
        # (192, 128) emits an elementwise bias for qr_tdm_pingpong at either width; no
        # other bias, no dropout.
        for dtype in ("fp16", "bf16", "fp8", "fp8bf16"):
            with self.subTest(dtype=dtype):
                _, kernels = get_fwd_blobs(
                    ["gfx1250"], f"*d192_{dtype}_*", 0, [192], "simplified"
                )
                biased = [k for k in kernels if k.F_pipeline.F_bias != "no"]
                self.assertTrue(
                    any(k.F_pipeline.tag == "qr_tdm_pingpong" for k in biased)
                )
                for k in biased:
                    self.assertEqual(k.F_pipeline.tag, "qr_tdm_pingpong")
                    self.assertEqual(k.F_pipeline.F_bias, "bias")
                self.assertFalse(any(k.F_pipeline.F_dropout == "t" for k in kernels))

    def test_generated_entries_use_the_compile_time_arch_check(self):
        _, kernels = get_fwd_blobs(
            ["gfx1250"], "*bf16*_qr_tdm*", 100, [128, 192], "simplified"
        )
        self.assertTrue(kernels)
        # (bf16 and fp8bf16)
        self.assertEqual(
            {k.F_pipeline.tag for k in kernels},
            {"qr_tdm_pingpong", "qr_tdm_kvshared", "qr_tdm_fp8_sched"},
        )
        for kernel in kernels:
            with self.subTest(kernel=kernel.name):
                source = kernel.render()
                self.assertIn("defined(__gfx125__)", source)
                self.assertNotIn("ck_tile::get_device_name()", source)

    def test_pingpong_tile_and_pipeline_ownership_is_symmetric(self):
        _, kernels = get_fwd_blobs(["gfx1250"], "*d128_bf16*", 100, [128], "simplified")
        candidate = next(
            k
            for k in kernels
            if k.F_tile.F_bm0 == 128
            and k.F_pipeline.F_packgqa == "f"
            and k.F_pipeline.F_dpad == "f"
        )
        rules = {r.__name__: r for r in KernelComponentFactoryGfx125.get_rules()}
        rule = rules["check_tile_owner"]
        problem = ProblemContext(dtype="bf16", mode="batch", hdim=128, hdim_v=128)
        context = KernelContext(
            tile=candidate.F_tile, pipeline=candidate.F_pipeline, mask_impl="simplified"
        )
        unowned_tile = replace(candidate.F_tile, F_pipeline_tag=None)
        other_pipeline = replace(candidate.F_pipeline, tag="qr_tdm")
        self.assertTrue(rule(problem, context))
        self.assertTrue(rule(replace(problem, dtype="fp16"), context))
        # pingpong runs on its own tiles only, and its tiles take no other pipeline
        self.assertFalse(rule(problem, replace(context, tile=unowned_tile)))
        self.assertFalse(rule(problem, replace(context, pipeline=other_pipeline)))
        self.assertTrue(
            rule(problem, replace(context, tile=unowned_tile, pipeline=other_pipeline))
        )
        # Q/K rows staged past the head dim are read by the padded instances only
        narrow = replace(problem, hdim=96)
        self.assertFalse(rule(narrow, context))
        self.assertTrue(
            rule(
                narrow,
                replace(context, pipeline=replace(candidate.F_pipeline, F_dpad="t")),
            )
        )
        # packed GQA: batch mode, tiles up to 64 rows
        packed = replace(context, pipeline=replace(candidate.F_pipeline, F_packgqa="t"))
        self.assertFalse(rule(problem, packed))

        # The scheduled pipeline and its tile pair with each other only (fp8 d128).
        sched_rule = rules["check_sched_tiles"]
        _, fp8_kernels = get_fwd_blobs(
            ["gfx1250"], "*d128_fp8bf16*", 100, [128], "simplified"
        )
        sched = next(k for k in fp8_kernels if k.F_pipeline.tag == "qr_tdm_fp8_sched")
        problem = replace(problem, dtype="fp8bf16")
        sched_context = KernelContext(
            tile=sched.F_tile, pipeline=sched.F_pipeline, mask_impl="simplified"
        )
        self.assertTrue(sched_rule(problem, sched_context))
        self.assertTrue(rule(problem, sched_context))
        self.assertTrue(sched_rule(replace(problem, dtype="fp8"), sched_context))
        self.assertFalse(
            sched_rule(problem, replace(sched_context, tile=candidate.F_tile))
        )
        self.assertFalse(sched_rule(problem, replace(context, tile=sched.F_tile)))
        # no scheduled tiles at 16 bits or at d192
        for changes in (dict(dtype="bf16"), dict(hdim=192)):
            self.assertFalse(sched_rule(replace(problem, **changes), sched_context))
        for pipeline_changes in (
            dict(F_vlayout="col"),
            dict(F_packgqa="t"),
        ):
            self.assertFalse(
                sched_rule(
                    problem,
                    replace(
                        sched_context,
                        pipeline=replace(sched.F_pipeline, **pipeline_changes),
                    ),
                )
            )
        self.assertFalse(
            sched_rule(
                problem,
                replace(
                    sched_context,
                    tile=replace(
                        sched.F_tile, F_occupancy=sched.F_tile.F_occupancy + 1
                    ),
                ),
            )
        )

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

    def test_tile_inventory_isolation(self):
        # Callers can tune their inventory (pingpong's tiles and the scheduled ones)
        # without changing later calls.
        factory = KernelComponentFactoryGfx125
        for dtype in _GFX125_DTYPES:
            with self.subTest(dtype=dtype):
                fp16_bf16 = dtype in ("fp16", "bf16")
                # fp16/bf16: qr_tdm_kvshared's tiles first, GDPA's own tiles behind pingpong's
                # (K and V resident, one workgroup a WGP, then the streaming q-loop's)
                expected = (
                    [t.name for t in factory._kvshared_tiles()[(128, 128)] if fp16_bf16]
                    + [
                        t.name
                        for t in _gfx125_qr_tdm_pingpong_tiles(
                            (128, 128),
                            fp8=dtype in ("fp8", "fp8bf16"),
                            mx=dtype.startswith("mx"),
                        )
                    ]
                    + (
                        [
                            "b128x32x32x128x32x128_r4x1x1_r4x1x1_w16x16x32_w16x16x32_o1",
                            "b128x32x32x128x32x128_r4x1x1_r4x1x1_w16x16x32_w16x16x32",
                        ]
                        if fp16_bf16
                        else []
                    )
                    + [t.name for t in factory._sched_tiles(dtype).get((128, 128), ())]
                )
                tiles = factory.get_hdim_tile_size_dict(dtype)[(128, 128)]
                self.assertEqual([t.name for t in tiles], expected)
                constraints = [str(t.F_constraint) for t in tiles]
                for tile in tiles:
                    tile.F_bm0 += 1
                    tile.F_constraint.bool_expr = "false"
                fresh = factory.get_hdim_tile_size_dict(dtype)[(128, 128)]
                self.assertEqual([t.name for t in fresh], expected)
                self.assertEqual([str(t.F_constraint) for t in fresh], constraints)

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
            # every qr_tdm_pingpong instance, with or without LSE, at any head dimension
            # and input width, while plain qr_tdm (4) and qr_tdm_sched outside
            # d64/d128/d192 (6) keep the default
            d128.replace("qr_tdm_sched", "qr_tdm_pingpong"),
            d128.replace("qr_tdm_sched", "qr_tdm_pingpong").replace("_nlse_", "_lse_"),
            d192.replace("d192", "d256"),
            d192.replace("d192", "d32").replace("_nlse_", "_lse_"),
            d128.replace("qr_tdm_sched", "qr_tdm_pingpong").replace("bf16", "fp8bf16"),
            d192.replace("d192", "d512").replace("bf16", "fp8"),
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
                    i in (0, 1, 2, 3, 5, 12, 13, 14, 15, 16, 17, 18, 19),
                )
                self.assertNotIn("-wwm-regalloc=fast", line)


@unittest.skipUnless(_HOST_CXX, "a C++17 host compiler is required")
class TestCompiledGfx125V128Dispatch(unittest.TestCase):
    def test_fp8_mx_selects_mx_data_type(self):
        # The entry point hands FP8 inputs with MX scales to the MX instances.
        stub = """#include <iostream>
#include <string>
namespace ck_tile { struct stream_config {}; }
enum class quant_scale_enum { no_scale, pertensor, blockscale, kv_blockscale, mx, perhead };
struct fmha_fwd_traits { std::string data_type; quant_scale_enum qscale_type; };
struct fmha_fwd_args { int max_seqlen_q = 1; };
float fmha_fwd_v3(fmha_fwd_traits, fmha_fwd_args, const ck_tile::stream_config&) { return -1; }
float fmha_fwd_v2(fmha_fwd_traits t, fmha_fwd_args, const ck_tile::stream_config&) {
    std::cout << t.data_type;
    return 0;
}
"""
        main = """
int main() {
    fmha_fwd_traits t;
    int scale;
    if(!(std::cin >> t.data_type >> scale)) return 2;
    t.qscale_type = static_cast<quant_scale_enum>(scale);
    return fmha_fwd(t, {}, {}) < 0;
}
"""
        with tempfile.TemporaryDirectory(prefix="fmha-mx-alias-") as directory:
            source = Path(directory) / "api.cpp"
            binary = Path(directory) / "api"
            source.write_text(stub + FMHA_FWD_API_FOOTER + main)
            subprocess.run(
                [_HOST_CXX, "-std=c++17", str(source), "-o", str(binary)],
                check=True,
                timeout=60,
            )
            for dtype, scale, expected in (
                ("fp8", 4, "mxfp8"),
                ("fp8bf16", 4, "mxfp8bf16"),
                ("mxfp8", 4, "mxfp8"),
                ("mxfp4", 4, "mxfp4"),
                ("fp8", 1, "fp8"),
                ("fp8bf16", 5, "fp8bf16"),
                ("bf16", 4, "bf16"),
            ):
                with self.subTest(dtype=dtype, scale=scale):
                    result = subprocess.run(
                        [str(binary)],
                        input=f"{dtype} {scale}\n",
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=True,
                    )
                    self.assertEqual(result.stdout, expected)

    def test_d64_sequence_padding_and_pingpong_tiles(self):
        for dtype in ("fp16", "bf16"):
            pool, kernels = get_fwd_blobs(
                ["gfx1250"],
                f"fmha_fwd_d64_{dtype}_*_nlogits*_nbias*_*lse*_ndropout*_nskip*_nqscale*_ntrload*_nsink*",
                600,
                [64],
                "simplified",
            )
            self.assertEqual({k.F_pipeline.tag for k in kernels}, {"qr_tdm_pingpong"})
            dispatch = _CompiledDispatcher(self, pool, kernels)
            # qr_tdm_pingpong takes every d64 query; its unpadded batch instances take
            # any K length. Its 128-row tile takes 128 query rows on, at any key length,
            # but under a sliding window of fewer than 512 keys (left + right + 1).
            for sq, sk, has_cu_sk, group, mask, wl, wr, bm0 in (
                (16, 128, 0, 0, 0, -1, -1, 16),
                (127, 128, 0, 0, 0, -1, -1, 64),
                (128, 128, 0, 0, 0, -1, -1, 128),
                (129, 129, 0, 0, 0, -1, -1, 128),
                (255, 255, 0, 0, 0, -1, -1, 128),
                (128, 0, 0, 0, 0, -1, -1, 128),
                (128, 128, 1, 0, 0, -1, -1, 128),
                (255, 255, 0, 1, 0, -1, -1, 128),
                (127, 1024, 0, 0, 0, -1, -1, 64),
                (600, 600, 0, 0, 0, -1, -1, 128),
                (127, 600, 0, 0, 1, -1, -1, 64),
                (128, 128, 0, 0, 1, -1, -1, 128),
                (600, 600, 0, 0, 1, -1, -1, 128),
                (600, 600, 1, 0, 1, -1, -1, 128),
                (600, 600, 0, 1, 1, -1, -1, 128),
                (600, 600, 0, 0, 3, 100, 0, 64),
                (600, 600, 0, 0, 3, 200, 310, 64),
                (600, 600, 0, 0, 3, 200, 311, 128),
                (600, 600, 0, 0, 3, 511, 0, 128),
                (600, 600, 0, 0, 3, 100, -1, 128),
                (600, 600, 0, 1, 3, 100, 0, 64),
            ):
                candidate = dispatch.expected_id(
                    dtype=dtype,
                    hdim=64,
                    mode="group" if group else "batch",
                    pipeline="qr_tdm_pingpong",
                    bm0=bm0,
                    mask="s_mask" if mask else "s_no",
                    lse="t",
                    dpad="f",
                    dvpad="f",
                    sink="f",
                    spad="t" if group else "f",
                    # 128 keys are too few for the 16-row tile's long trips
                    **({"bn0": _short_q_bn0(sk, (64, 64))} if bm0 == 16 else {}),
                )
                args = dict(
                    dtype=dtype,
                    hdim_q=64,
                    hdim_v=64,
                    seqlen_q=sq,
                    seqlen_k=sk,
                    has_cu_sk=has_cu_sk,
                    group=group,
                    mask=mask,
                    window_size_left=wl,
                    window_size_right=wr,
                    lse=1,
                )
                self.assertEqual(dispatch.run(**args), [candidate], args)

    def test_fp8_padded_head_dims_reach_pingpong(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"],
            "fmha_fwd_d*_fp8_[bg]*_nlogits_nbias_nmask_nlse_ndropout_nskip_*_nsink*",
            0,
            [64, 128],
            "simplified",
        )
        dispatch = _CompiledDispatcher(
            self, pool, kernels, filter_fn=lambda trait: True
        )
        pertensor, blockscale = 1, 2
        # d64 fp8 takes qr_tdm_pingpong's tiles at any query length, as fp16/bf16 do: 16
        # rows (decode, 256-key trips under 512 keys), 64 rows below 128, the 128-row tile
        # from 128 dense rows on; the smaller head dims of whole 8-element vectors on its
        # general tile's fully padded instances; block scales of any q block and whole
        # 32-key k/v blocks on its general and decode tiles (qr_tdm_pingpong_tile_takes).
        for (
            hdim_q,
            hdim_v,
            qscale,
            block_scale_size,
            padded,
            seqlen_q,
            tile,
        ) in (
            (32, 32, pertensor, 0, True, 128, dict(bm0=128)),
            (48, 48, pertensor, 0, True, 128, dict(bm0=128)),
            (64, 32, pertensor, 0, True, 128, dict(bm0=128)),
            (64, 64, pertensor, 0, False, 128, dict(bm0=128)),
            (64, 64, pertensor, 0, False, 127, dict(bm0=64)),
            (64, 64, pertensor, 0, False, 64, dict(bm0=64)),
            (64, 64, pertensor, 0, False, 1, dict(bm0=16, bn0=256)),
            (64, 64, blockscale, 64, False, 128, dict(bm0=128)),
            (64, 64, blockscale, 128, False, 128, dict(bm0=128)),
            (64, 64, blockscale, 96, False, 64, dict(bm0=128)),
        ):
            with self.subTest(
                hdim_q=hdim_q,
                hdim_v=hdim_v,
                qscale=qscale,
                block=block_scale_size,
                seqlen_q=seqlen_q,
            ):
                features = dict(
                    dtype="fp8",
                    hdim=64,
                    mode="batch",
                    pipeline="qr_tdm_pingpong",
                    qscale="pertensor" if qscale == pertensor else "blockscale",
                    dpad="t" if padded else "f",
                    dvpad="t" if padded else "f",
                    spad="t" if padded else "f",
                )
                features.update(tile)
                self.assertEqual(
                    dispatch.run(
                        dtype="fp8",
                        hdim_q=hdim_q,
                        hdim_v=hdim_v,
                        seqlen_q=seqlen_q,
                        max_seqlen_q=seqlen_q,
                        qscale=qscale,
                        block_scale_size_q=block_scale_size,
                        block_scale_size_kv=block_scale_size,
                        nhead_k=8,
                    ),
                    [dispatch.expected_id(**features)],
                )

    def test_fp8_pingpong_takes_the_fp16_bf16_features(self):
        # qr_tdm_pingpong's fp8 instances take bias, sink tokens and a runtime sink logit as
        # the fp16/bf16 ones do, and block scales with k/v blocks of whole 32-key chunks;
        # nothing serves other block sizes
        pool, kernels = get_fwd_blobs(
            ["gfx1250"],
            "fmha_fwd_d*_fp8_batch_*_nlogits_*_nmask_nlse_ndropout_nskip_*",
            0,
            [64, 128],
            "simplified",
        )
        dispatch = _CompiledDispatcher(
            self, pool, kernels, filter_fn=lambda trait: True
        )
        tag = {index: kernel.F_pipeline.tag for kernel, index in dispatch.kernel_ids}
        self.assertEqual(set(tag.values()), {"qr_tdm_pingpong", "qr_tdm_fp8_sched"})
        pertensor, blockscale = 1, 2
        for hdim in (64, 128):
            for qscale, block, bias, sink, has_sink_ptr, pingpong in (
                (pertensor, 0, 0, 0, 0, True),
                (pertensor, 0, 1, 0, 0, True),
                (pertensor, 0, 0, 1, 1, True),
                (pertensor, 0, 0, 0, 1, True),
                (blockscale, 256, 0, 0, 0, True),
                (blockscale, 128, 0, 0, 0, True),
                (blockscale, 32, 0, 0, 0, True),
                (blockscale, 96, 0, 0, 0, True),
                (blockscale, 48, 0, 0, 0, False),
                (blockscale, 16, 0, 0, 0, False),
            ):
                with self.subTest(
                    hdim=hdim,
                    qscale=qscale,
                    block=block,
                    bias=bias,
                    sink=sink,
                    sink_ptr=has_sink_ptr,
                ):
                    got = dispatch.run(
                        dtype="fp8",
                        hdim_q=hdim,
                        hdim_v=hdim,
                        seqlen_q=1024,
                        max_seqlen_q=1024,
                        seqlen_k=1024,
                        batch=64,  # past d128's small grids (qr_tdm_fp8_sched's)
                        qscale=qscale,
                        block_scale_size_q=block,
                        block_scale_size_kv=block,
                        bias=bias,
                        sink=sink,
                        has_sink_ptr=has_sink_ptr,
                        nhead_k=8,
                    )
                    self.assertEqual(len(got), 1)
                    self.assertEqual(
                        got[0] != -1 and tag[got[0]] == "qr_tdm_pingpong", pingpong
                    )
                    if not pingpong:
                        self.assertEqual(got, [-1])

    def test_causal_takes_no_window(self):
        # Top-left/bottom-right masks also describe finite windows (`t:l,r`). With the
        # generic mask_impl the causal mask (GenericAttentionMask<true, false>) bounds keys
        # on the right only, so a left window dispatches to the generic (local) mask; with
        # the simplified one, pingpong's causal (band-mask) instances take no window.
        api = FmhaFwdApiPool()
        _, kernels = get_fwd_blobs(
            ["gfx1250"], "*d128_bf16_*pingpong*", 0, [128], "generic"
        )
        for k in kernels:
            api.register_traits(k.api_trait())
        rendered = api.render("x")
        self.assertIn(
            "(t.mask_type == mask_enum::mask_top_left || t.mask_type == "
            "mask_enum::mask_bottom_right) && a.window_size_left < 0",
            rendered,
        )
        self.assertIn(
            "(t.mask_type == mask_enum::window_generic) || ((t.mask_type == "
            "mask_enum::mask_top_left || t.mask_type == mask_enum::mask_bottom_right) && "
            "a.window_size_left >= 0)",
            rendered,
        )
        for dtype in ("bf16", "fp8bf16"):
            with self.subTest(dtype=dtype):
                _, kernels = get_fwd_blobs(
                    ["gfx1250"], f"*d128_{dtype}_*pingpong*", 0, [128], "simplified"
                )
                causal = [
                    k
                    for k in kernels
                    if k.F_pipeline.F_mask == "causal" and k.F_pipeline.F_packgqa == "f"
                ]
                self.assertTrue(causal)
                for k in causal:
                    self.assertIn(
                        "a.window_size_left < 0 && a.window_size_right == 0",
                        str(k.api_trait().constraint),
                    )

    def test_pingpong_packgqa_decode(self):
        # GQA decode folds a KV head's query heads into one pingpong tile, for both widths:
        # no mask, or one query per head under a bottom-right causal mask or any window
        # (the mask's one query row serves every merged row); merged rows take
        # their heads' q descales and sink logits but no bias; batch only, so no sequence
        # padding. The
        # packed variants of every tile dispatch ahead of the unpacked pingpong ones.
        for dtype, qscales in (
            ("bf16", {"no"}),
            ("fp8bf16", {"no", "pertensor", "perhead", "blockscale"}),
        ):
            with self.subTest(dtype=dtype):
                pool, kernels = get_fwd_blobs(
                    ["gfx1250"], f"*d128_{dtype}_*pingpong*", 0, [128], "simplified"
                )
                packed = [k for k in kernels if k.F_pipeline.F_packgqa == "t"]
                # the tiles up to 64 rows (a 128-row tile's merged rows never fit it)
                self.assertEqual(
                    {k.F_tile.name for k in packed},
                    {k.F_tile.name for k in kernels if k.F_tile.F_bm0 <= 64},
                )
                for k in packed:
                    p = k.F_pipeline
                    self.assertEqual(k.F_mode, "batch")
                    self.assertEqual((p.F_bias, p.F_dropout), ("no", "f"))
                    self.assertIn(p.F_qscale, qscales)
                    self.assertIn(p.F_mask, ("s_no", "causal", "s_mask"))
                    self.assertEqual((p.F_spad, p.F_skpad), ("f", "f"))
                    constraint = str(k.api_trait().constraint)
                    self.assertIn(
                        f"fmha_fwd_can_merge_head_groups(a, {k.F_tile.F_bm0},",
                        constraint,
                    )
                    if p.F_mask == "causal":
                        self.assertIn("mask_enum::mask_bottom_right", constraint)
                    if p.F_mask != "s_no":
                        self.assertIn("a.seqlen_q == 1", constraint)
                    self.assertIn("ck_tile::TileFmhaMergeHeadsTraits<", k.render())
                # dispatch tries the rendered traits in order
                order = [
                    trait.endswith(", true>")  # the packed-GQA trait flag
                    for trait in re.findall(
                        r"using trait_ = (.*);", pool.render("fmha_fwd_v2")
                    )
                    if "QRKSVS_TDM_PINGPONG" in trait
                ]
                self.assertIn(True, order)
                self.assertLess(
                    max(i for i, packed in enumerate(order) if packed),
                    min(i for i, packed in enumerate(order) if not packed),
                )

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
                # From 128 rows on pingpong's 128-row tile (d128: its 128-key trips).
                candidate = dispatch.expected_id(
                    bm0=128, **({"bn0": 128} if dim == 128 else {}), **pingpong
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
            # d128 and d192 alike: qr_tdm_pingpong (no fp16/bf16 scheduled pipeline)
            family = "qr_tdm_pingpong"
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
                        # pingpong's 128-row tile, at any mask (no sliding window; d128 its
                        # 128-key trips), on small and big grids
                        pingpong = dispatch.expected_id(
                            hdim=dim,
                            pipeline="qr_tdm_pingpong",
                            bm0=128,
                            **({"bn0": 128} if dim == 128 else {}),
                            spad="t" if group else "f",
                            dpad="f",
                            **features,
                        )
                        candidate = pingpong
                        args = dict(hdim_q=dim, group=group, mask=mask, lse=lse)
                        self.assertEqual(dispatch.run(**args), [candidate])
                        # pingpong keeps a sink pointer without sink tokens
                        self.assertEqual(
                            dispatch.run(**args, has_sink_ptr=1), [pingpong]
                        )
                        # a big grid: pingpong
                        self.assertEqual(dispatch.run(**args, batch=256), [pingpong])
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
                            if rejected == dict(skip=1) and not group:
                                self.assertEqual(
                                    dispatch.run(**args, **rejected),
                                    dispatch.run(**args),
                                )
                            else:
                                self.assertNotIn(
                                    dispatch.run(**args, **rejected)[0], family_ids
                                )

    def test_d128_filtered_tile_priority(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"],
            _SUPPORTED_DISPATCH_FILTER,
            600,
            [128],
            "simplified",
        )
        # (qr_tdm_kvshared takes K and V as one tensor only)
        self.assertEqual(
            {k.F_pipeline.tag for k in kernels}, {"qr_tdm_pingpong", "qr_tdm_kvshared"}
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
                    # qr_tdm_pingpong: up to 16 rows its 16-row tile, below 128 its 64-row
                    # one, else its 128-row 128-key-trip one (masked too: no left window)
                    for size in (
                        1,
                        16,
                        17,
                        127,
                        128,
                        129,
                        2047,
                        2048,
                        4095,
                        4096,
                        8192,
                    ):
                        bm0 = 16 if size <= 16 else 64 if size < 128 else 128
                        extra = (
                            {"bn0": _short_q_bn0(64 if group else 128)}
                            if bm0 == 16
                            else {"bn0": 128}
                            if bm0 == 128
                            else {}
                        )
                        # one query per head packs the 4 query heads of a KV head
                        # (any mask: the windowed packed-GQA decode takes one query a head)
                        if size == 1 and not group:
                            extra["packgqa"] = "t"
                        expected = dispatch.expected_id(
                            pipeline="qr_tdm_pingpong",
                            spad="t" if group else "f",
                            bm0=bm0,
                            dpad="f",
                            **extra,
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
                pipeline="qr_tdm_pingpong",
                bm0=128,
                bn0=128,
                spad="t" if group else "f",
                dpad="f",
                mask="s_no",
                lse="f",
                sink="f",
            )
            self.assertEqual(
                dispatch.run(group=group, hdim_q=128, hdim_v=128), [candidate]
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
                if changes == dict(skip=1) and not group:
                    self.assertEqual(
                        dispatch.run(group=group, hdim_q=128, **changes),
                        dispatch.run(group=group, hdim_q=128),
                    )
                else:
                    self.assertEqual(
                        dispatch.run(group=group, hdim_q=128, **changes), [-1]
                    )

    def test_pingpong_tiles(self):
        # qr_tdm_pingpong runs on its own tiles only, which no other pipeline (d128's
        # scheduled ones included) takes; it alone uses the LDS-shuffle epilogue.
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
        # 16-bit d256/d512 (Q/K rows of 512 bytes or more) have no 128-row tile.
        # Pingpong stages Q/K at their own row rounded up to whole K slices (d32 does
        # not pad to 64 at 16 bits), and the parts of the LDS read schedule that take
        # effect on a tile show in its instance name: slice prefetch from 64 rows on, V
        # before the softmax on the 128-row tiles here.
        row_tiles = {
            32: {16, 64, 128},
            128: {16, 64, 128},
            192: {16, 64, 128},
            256: {16, 64},
            512: {16, 64},
        }
        schedule = {
            (32, 64): "_sp64",
            (32, 128): "_sp64_vbs128",
            (128, 64): "_sp64",
            (128, 128): "_sp64_vbs128",
            (192, 64): "_sp64",
            (192, 128): "_sp64_vbs128",
        }
        self.assertEqual(
            {(k.F_hdim, k.F_tile.F_bm0) for k in pingpong},
            {(hdim, bm0) for hdim, tiles in row_tiles.items() for bm0 in tiles},
        )
        for kernel in pingpong:
            with self.subTest(kernel=kernel.name):
                self.assertIn(kernel.F_dtype, ("fp16", "bf16", "fp8", "fp8bf16"))
                # 32 bytes of K per WMMA lane: 16x16x32 at 16 bits, 16x16x64 at fp8; the
                # fp8 short-query tiles take up to 16x16x128
                fp8 = kernel.F_dtype in ("fp8", "fp8bf16")
                t = kernel.F_tile
                decode = t.F_bm0 == 16
                wk = 64 if fp8 else 32
                k_row = -(-kernel.F_hdim // wk) * wk
                if fp8 and decode:
                    # 128-deep where the slices divide the row (d192 keeps 64)
                    wk = 128 if k_row % 128 == 0 else 64
                self.assertEqual(t.F_wk0, wk)
                # the decode tile's trip is one PV slice over its key warps
                # (test_pingpong_decode_tiles), on the widest WMMA K a key warp's keys
                # fill; the others' PV slices are as deep as their QK's
                if decode:
                    self.assertEqual(t.F_bk1, t.F_bn0)
                    self.assertEqual(
                        t.F_wk1, min(128 if fp8 else 32, t.F_bn0 // t.F_rk1)
                    )
                else:
                    self.assertEqual(t.F_wk1, wk)
                self.assertEqual(t.F_bk0max, k_row)
                self.assertEqual(
                    kernel.F_tile.F_bn1, 128 if kernel.F_hdim == 192 else kernel.F_hdim
                )
                suffix = schedule.get((kernel.F_hdim, kernel.F_tile.F_bm0), "")
                packgqa = "_packgqa" if kernel.F_pipeline.F_packgqa == "t" else ""
                self.assertTrue(
                    kernel.name.endswith("_ntrload" + suffix + packgqa + "_nsink")
                    or kernel.name.endswith("_ntrload" + suffix + packgqa + "_sink")
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

    def test_pingpong_decode_tiles(self):
        # The 16-row (decode) tiles per head dim and input type: (keys a trip, warps on the
        # trip's keys), the long trip first. Four warps split the keys while a warp's O and
        # V operand fit 192 VGPRs and a trip's K and V 128 KB of LDS on the shortest trip
        # (a PV warp gemm per key warp: 32 keys at 16 bits, 64 at fp8, 128 for MX), else
        # two; d512 splits none (its warps on gemm_0's N exchange S). Long trips: up to
        # 512 keys and 64 KB of K within the same budgets.
        expected = {
            "bf16": {
                32: ((512, 4), (128, 4)),
                64: ((512, 4), (128, 4)),
                96: ((256, 4), (128, 4)),
                128: ((256, 4), (128, 4)),
                160: ((128, 4),),
                192: ((128, 4),),
                256: ((128, 2), (64, 2)),
                512: ((32, 1),),
            },
            "fp8": {
                32: ((512, 4), (256, 4)),
                64: ((512, 4), (256, 4)),
                96: ((512, 4), (256, 4)),
                128: ((512, 4), (256, 4)),
                160: ((256, 4),),
                192: ((256, 4),),
                256: ((256, 2), (128, 2)),
                512: ((64, 1),),
            },
            "mx": {
                32: ((512, 4),),
                64: ((512, 4),),
                96: ((512, 4),),
                128: ((512, 4),),
                160: ((256, 2),),
                192: ((256, 2),),
                256: ((256, 2),),
                512: ((128, 1),),
            },
        }
        for kind, dims in expected.items():
            for hdim_pair in GFX125_QR_TDM_PINGPONG_HDIMS:
                with self.subTest(kind=kind, hdim=hdim_pair):
                    tiles = _gfx125_qr_tdm_pingpong_tiles(
                        hdim_pair, fp8=kind == "fp8", mx=kind == "mx"
                    )
                    decode = [t for t in tiles if t.F_bm0 == 16]
                    self.assertEqual(
                        tuple((t.F_bn0, t.F_rk1) for t in decode), dims[hdim_pair[0]]
                    )
                    # dispatched ahead of the longer query tiles, the long trip ahead of
                    # the shortest one, which takes every short query
                    self.assertEqual(list(tiles[: len(decode)]), decode)
                    for t in decode[:-1]:
                        self.assertIn("a.seqlen_k >=", str(t.F_constraint))
                    self.assertNotIn("a.seqlen_k >=", str(decode[-1].F_constraint))
                    for t in decode:
                        # kw warps on the keys (gemm_0's N, gemm_1's K) times 4 / kw on
                        # V's columns (gemm_1's N); one trip a PV slice
                        # (kw = 1: gemm_0's warps on N, one per 16 keys up to four, and
                        # as many on gemm_1's N)
                        kw = t.F_rk1
                        nw = 4 // kw if kw > 1 else min(4, t.F_bn0 // 16)
                        self.assertEqual(
                            (t.F_rm0, t.F_rn0, t.F_rk0), (1, kw if kw > 1 else nw, 1)
                        )
                        self.assertEqual((t.F_rm1, t.F_rn1), (1, nw))
                        self.assertEqual(t.F_bk1, t.F_bn0)

    def test_d128_pingpong_dispatch(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"],
            "fmha_fwd_d128_bf16_*_nlogits*_ndropout*_nskip*_nqscale*_ntrload*",
            600,
            [128],
            "simplified",
        )
        # (qr_tdm_kvshared takes K and V as one tensor only)
        self.assertEqual(
            {k.F_pipeline.tag for k in kernels}, {"qr_tdm_pingpong", "qr_tdm_kvshared"}
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)

        def kernel(bm0, mask="s_no", skpad="f", bias="no", sink="f", bn0=None):
            # the calls below with a 16-row tile have >= 512 keys (its long trips); the
            # 128-row tile takes 128-key trips without a left window, else 64-key ones
            if bn0 is None:
                bn0 = _short_q_bn0(512) if bm0 == 16 else 128 if bm0 == 128 else None
            return dispatch.expected_id(
                hdim=128,
                mode="batch",
                pipeline="qr_tdm_pingpong",
                bm0=bm0,
                **({"bn0": bn0} if bn0 is not None else {}),
                mask=mask,
                skpad=skpad,
                dpad="f",
                bias=bias,
                lse="f",
                sink=sink,
            )

        def run(**changes):
            return dispatch.run(hdim_q=128, hdim_v=128, **changes)

        square = dict(seqlen_q=532, seqlen_k=532)
        causal = dict(mask=1, window_size_left=-1, window_size_right=0)
        # 1280 x 2 (batch, head) pairs: 10 per num_cus (256).
        big = dict(batch=1280, nhead_q=2, nhead_k=2)

        # qr_tdm_pingpong on any grid: its 128-row tile from 128 rows on at any key length
        # and mask, with 128-key trips but under a left window (64-key trips; a narrow one
        # takes the 64-row tile).
        for grid in ({}, big):
            self.assertEqual(run(**square, **grid), [kernel(128)])
            self.assertEqual(
                run(**square, **causal, **grid), [kernel(128, mask="causal")]
            )
        self.assertEqual(run(seqlen_q=640, seqlen_k=640, **big), [kernel(128)])
        window = dict(mask=3, window_size_left=255, window_size_right=0)
        self.assertEqual(run(**square, **window, **big), [kernel(64, mask="s_mask")])
        self.assertEqual(
            run(**square, **dict(window, window_size_left=511), **big),
            [kernel(128, mask="s_mask", bn0=64)],
        )
        # no grid threshold: 16 x 16 heads of 255 or 256 rows alike
        grid = dict(batch=16, nhead_q=16, nhead_k=16)
        self.assertEqual(run(seqlen_q=256, seqlen_k=256, **grid), [kernel(128)])
        self.assertEqual(run(seqlen_q=255, seqlen_k=255, **grid), [kernel(128)])
        # Below 128 rows: up to 16 rows the 16-row tile, else the 64-row one.
        for length, bm0 in ((8, 16), (16, 16), (17, 64), (100, 64), (127, 64)):
            self.assertEqual(run(seqlen_q=length, seqlen_k=512), [kernel(bm0)])
            self.assertEqual(run(seqlen_q=length, seqlen_k=532, **big), [kernel(bm0)])
        self.assertEqual(
            run(seqlen_q=32, seqlen_k=200, batch=1792, nhead_q=1, nhead_k=1),
            [kernel(64)],
        )
        # The unpadded instance serves any seqlen_k, except with an elementwise bias,
        # which still needs a tile multiple. Bias, ALiBi and sink tokens run on the
        # general 128-row tile (64-key trips: qr_tdm_pingpong_tile_takes).
        self.assertEqual(
            run(**square, **big, bias=1), [kernel(128, skpad="t", bias="bias", bn0=64)]
        )
        self.assertEqual(
            run(seqlen_q=532, seqlen_k=512, bias=1, **big),
            [kernel(128, bias="bias", bn0=64)],
        )
        self.assertEqual(
            run(**square, **big, bias=2), [kernel(128, bias="alibi", bn0=64)]
        )
        # sink tokens (their instances also take a sink pointer), and a sink pointer alone
        for grid in ({}, big):
            self.assertEqual(
                run(**square, sink=1, **grid), [kernel(128, sink="t", bn0=64)]
            )
            self.assertEqual(
                run(**square, sink=1, has_sink_ptr=1, **grid),
                [kernel(128, sink="t", bn0=64)],
            )
            self.assertEqual(run(**square, has_sink_ptr=1, **grid), [kernel(128)])
            self.assertEqual(
                run(**square, **causal, has_sink_ptr=1, **grid),
                [kernel(128, mask="causal")],
            )

    def check_d128_grid_dispatch(self, small):
        # d128 at 1024 query rows: fp8 on a small grid (fewer than 256 x num_cus query
        # rows over the (batch, head) pairs) and MX prefill on any grid run
        # qr_tdm_fp8_sched; everything else pingpong's first 128-row tile (fp16/bf16: its
        # 128-key trips).
        grid = dict(batch=1, nhead_q=8, nhead_k=8) if small else dict(batch=64)
        for dtype, qscale, qscale_name, sched_tag, sched_here in (
            ("bf16", 0, "no", None, False),
            ("fp16", 0, "no", None, False),
            ("fp8", 1, "pertensor", "qr_tdm_fp8_sched", small),
            ("fp8bf16", 1, "pertensor", "qr_tdm_fp8_sched", small),
            ("mxfp8", 4, "mx", "qr_tdm_fp8_sched", True),
        ):
            with self.subTest(dtype=dtype, small=small):
                pool, kernels = get_fwd_blobs(
                    ["gfx1250"],
                    f"fmha_fwd_d128_{dtype}_batch_*_nlogits_nbias_nmask_nlse_ndropout_"
                    "nskip_*_nsink*",
                    0,
                    [128],
                    "simplified",
                )
                self.assertEqual(
                    _tags_without_kvshared(kernels),
                    {"qr_tdm_pingpong"} | ({sched_tag} if sched_tag else set()),
                )
                dispatch = _CompiledDispatcher(
                    self, pool, kernels, filter_fn=lambda trait: True
                )
                m128 = next(
                    t
                    for t in _gfx125_qr_tdm_pingpong_tiles(
                        (128, 128), fp8=dtype.startswith("fp8"), mx=qscale_name == "mx"
                    )
                    if t.F_bm0 == 128
                )
                features = dict(dtype=dtype, hdim=128, mode="batch", qscale=qscale_name)
                if sched_here:
                    expected = dispatch.expected_id(pipeline=sched_tag, **features)
                else:
                    expected = dispatch.expected_id(
                        pipeline="qr_tdm_pingpong",
                        bm0=128,
                        bn0=m128.F_bn0,
                        dpad="f",
                        spad="f",
                        **features,
                    )
                for seqlen_k in (1024, 2048):
                    for num_cus in (256, 128):
                        self.assertEqual(
                            dispatch.run(
                                dtype=dtype,
                                hdim_q=128,
                                hdim_v=128,
                                seqlen_q=1024,
                                seqlen_k=seqlen_k,
                                qscale=qscale,
                                num_cus=num_cus,
                                **grid,
                            ),
                            [expected],
                            (seqlen_k, num_cus),
                        )

    def test_d128_small_grid_dispatches_fp8_and_mx_to_sched(self):
        self.check_d128_grid_dispatch(small=True)

    def test_d128_big_grid_dispatches_to_pingpong_but_mx(self):
        self.check_d128_grid_dispatch(small=False)

    def test_group_mode_skip_dispatches_to_pingpong_skip_instances(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"],
            "fmha_fwd_d*_bf16_*_nlogits_nbias_nmask_nlse_ndropout_*_nqscale_*_nsink*",
            600,
            [64, 128, 192],
            "simplified",
        )
        skip = [k for k in kernels if k.F_pipeline.F_skip == "t"]
        self.assertTrue(skip)
        # skip_min_seqlen_q instances are group mode only, on every pingpong tile (the
        # scheduled pipeline has none)
        self.assertEqual({k.F_mode for k in skip}, {"group"})
        self.assertEqual({k.F_pipeline.tag for k in skip}, {"qr_tdm_pingpong"})
        self.assertEqual(
            {(k.F_hdim, k.F_tile.name) for k in skip},
            {
                (k.F_hdim, k.F_tile.name)
                for k in kernels
                if k.F_mode == "group"
                and k.F_pipeline.F_skip == "f"
                and k.F_pipeline.tag == "qr_tdm_pingpong"
            },
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)
        for hdim, hdim_v in ((64, 64), (128, 128), (192, 128)):
            for seqlen_q, bm0 in ((16, 16), (100, 64), (512, 128)):
                with self.subTest(hdim=hdim, seqlen_q=seqlen_q):
                    args = dict(
                        hdim_q=hdim, hdim_v=hdim_v, seqlen_q=seqlen_q, seqlen_k=512
                    )
                    features = dict(
                        hdim=hdim, pipeline="qr_tdm_pingpong", bm0=bm0, dpad="f"
                    )
                    if bm0 == 16:
                        features["bn0"] = _short_q_bn0(256, (hdim, hdim_v))
                    elif hdim == 128:
                        # 64-row tile: 64-key trips; 128-row: 128-key trips
                        features["bn0"] = 128 if bm0 == 128 else 64
                    self.assertEqual(
                        dispatch.run(group=1, skip=1, **args),
                        [
                            dispatch.expected_id(
                                mode="group", spad="t", skip="t", **features
                            )
                        ],
                    )
                    # without skip, the same tiles' instances without it
                    no_skip = dispatch.expected_id(mode="group", spad="t", **features)
                    self.assertEqual(dispatch.run(group=1, **args), [no_skip])
                    # batch mode reuses the instances without skip
                    self.assertEqual(
                        dispatch.run(group=0, skip=1, **args),
                        dispatch.run(group=0, **args),
                    )

    def test_padded_head_dims_off_the_vector(self):
        # Padded fp16/bf16 pingpong instances take any head dim (O is stored element by
        # element); fp8 ones still need whole 8-element rows.
        pool, kernels = get_fwd_blobs(
            ["gfx1250"],
            "fmha_fwd_d*_[bf]*_batch_*_nlogits_nbias_nmask_nlse_ndropout_nskip_*_nsink*",
            0,
            [96, 128],
            "simplified",
        )
        kernels = [k for k in kernels if k.F_dtype in ("bf16", "fp8")]
        self.assertEqual({k.F_dtype for k in kernels}, {"bf16", "fp8"})
        dispatch = _CompiledDispatcher(
            self, pool, kernels, filter_fn=lambda trait: trait.dtype in ("bf16", "fp8")
        )
        for dtype, qscale, hdim_q, hdim_v, padded in (
            ("bf16", 0, 100, 100, True),
            ("bf16", 0, 100, 128, True),
            ("bf16", 0, 128, 100, True),
            ("bf16", 0, 128, 128, False),
            ("fp8", 1, 104, 104, True),
            ("fp8", 1, 128, 128, False),
            ("fp8", 1, 100, 100, None),
            ("fp8", 1, 128, 100, None),
        ):
            with self.subTest(dtype=dtype, hdim=(hdim_q, hdim_v)):
                got = dispatch.run(
                    dtype=dtype,
                    hdim_q=hdim_q,
                    hdim_v=hdim_v,
                    qscale=qscale,
                    nhead_k=8,
                    batch=64,  # past d128's small grids (fp8's scheduled pipeline)
                )
                if padded is None:
                    self.assertEqual(got, [-1])
                    continue
                self.assertEqual(
                    got,
                    [
                        dispatch.expected_id(
                            dtype=dtype,
                            hdim=128,
                            mode="batch",
                            pipeline="qr_tdm_pingpong",
                            bm0=128,
                            # fp16/bf16 d128: 128-key trips; a padded head dim takes the
                            # general tile's fully padded instance
                            # (qr_tdm_pingpong_tile_takes)
                            bn0=128 if dtype == "bf16" and not padded else 64,
                            qscale="pertensor" if qscale else "no",
                            spad="t" if padded else "f",
                            dpad="t" if padded else "f",
                            dvpad="t" if padded else "f",
                        )
                    ],
                )
        for k in kernels:
            if k.F_pipeline.tag != "qr_tdm_pingpong":
                continue
            trait = k.api_trait()
            if trait.dpad == "t":
                self.assertEqual(
                    trait.dcheck, "true" if k.F_dtype == "bf16" else "a.hdim_q % 8 == 0"
                )
            if trait.dvpad == "t":
                self.assertEqual(
                    trait.dvcheck,
                    "true" if k.F_dtype == "bf16" else "a.hdim_v % 8 == 0",
                )

    def test_fp8_block_scale_sizes(self):
        # fp8 pingpong block scales: any q block, k/v blocks of whole 32-key chunks, on
        # its general and decode tiles (no per-tile clause; qr_tdm_pingpong_tile_takes).
        # On d128's small grids qr_tdm_fp8_sched takes q blocks of whole 128-row tiles
        # first; pingpong the ones it refuses.
        pool, kernels = get_fwd_blobs(
            ["gfx1250"],
            "fmha_fwd_d128_fp8_batch_*_nlogits_nbias_nmask_nlse_ndropout_nskip_"
            "blockscale_*_nsink*",
            0,
            [128],
            "simplified",
        )
        self.assertEqual(
            {k.F_pipeline.tag for k in kernels}, {"qr_tdm_pingpong", "qr_tdm_fp8_sched"}
        )
        for k in kernels:
            constraint = str(k.api_trait().constraint)
            self.assertIn("a.block_scale_size_kv % 32 == 0", constraint)
            self.assertEqual(constraint.count("a.block_scale_size_kv %"), 1)
            self.assertEqual(
                "a.block_scale_size_q % 128 == 0" in constraint,
                k.F_pipeline.tag == "qr_tdm_fp8_sched",
            )
        # (render only the dispatch: the host stub has no block-scale align table)
        dispatch = _CompiledDispatcher(
            self, pool, kernels, filter_fn=lambda trait: True
        )
        big, small = 64, 2
        for q_block, kv_block, seqlen_q, batch, expected in (
            (128, 96, 1024, big, 128),
            (48, 96, 1024, big, 128),
            (16, 32, 100, big, 128),
            (128, 96, 16, big, 16),
            (128, 48, 1024, big, None),
            (128, 48, 100, big, None),
            (128, 16, 1024, big, None),
            (0, 96, 1024, big, None),
            (128, 96, 1024, small, "sched"),
            (256, 32, 1024, small, "sched"),
            (48, 96, 1024, small, 128),
            (128, 48, 1024, small, None),
        ):
            with self.subTest(
                q_block=q_block, kv_block=kv_block, seqlen_q=seqlen_q, batch=batch
            ):
                got = dispatch.run(
                    dtype="fp8",
                    hdim_q=128,
                    hdim_v=128,
                    seqlen_q=seqlen_q,
                    seqlen_k=1024,
                    qscale=2,
                    block_scale_size_q=q_block,
                    block_scale_size_kv=kv_block,
                    batch=batch,
                    nhead_k=8,
                )
                features = dict(
                    dtype="fp8", hdim=128, mode="batch", qscale="blockscale"
                )
                if expected is None:
                    self.assertEqual(got, [-1])
                elif expected == "sched":
                    self.assertEqual(
                        got,
                        [dispatch.expected_id(pipeline="qr_tdm_fp8_sched", **features)],
                    )
                else:
                    self.assertEqual(
                        got,
                        [
                            dispatch.expected_id(
                                pipeline="qr_tdm_pingpong",
                                bm0=expected,
                                bn0=_short_q_bn0(1024, dtype="fp8")
                                if expected == 16
                                else 64,
                                spad="f",
                                dpad="f",
                                **features,
                            )
                        ],
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
        # two tensors, or more than 16 rows: qr_tdm_pingpong, as without kvshared (532 keys
        # take d128's long-trip decode tile, 256 keys a trip)
        for grid in (big, {}):
            self.assertEqual(
                run(seqlen_q=8, seqlen_k=532, **grid),
                [kernel("qr_tdm_pingpong", 16, 256)],
            )
            self.assertEqual(
                run(seqlen_q=17, seqlen_k=532, kv_shared=1, **grid),
                [kernel("qr_tdm_pingpong", 64)],
            )


class SplitKVQrTdmTiles(unittest.TestCase):
    """The split-kv side of the gfx125 qr_tdm tile contract."""

    def _tdm_kernels(self):
        kernels = get_fwd_splitkv_blobs(
            targets=["gfx1250"],
            kernel_filter="",
            receipt=600,
            mask_impl="simplified",
            optdim_list=[-1],
        )
        return [
            k
            for k in kernels
            if getattr(getattr(k, "F_pipeline", None), "tag", None) == "qr_tdm"
        ]

    def test_every_qr_tdm_tile_keeps_bn0_64(self):
        # kReloadKProgressively is gated on kN0 == 64 and turns itself off
        # silently on any other tile, so pin it here in codegen.
        tdm = self._tdm_kernels()
        self.assertTrue(tdm, "no qr_tdm split-kv kernels were generated")
        for kernel in tdm:
            self.assertEqual(kernel.F_tile.F_bn0, 64, kernel.filename)

    def test_qr_tdm_covers_every_gfx125_split_kv_head_dim(self):
        self.assertEqual(
            sorted({kernel.F_hdim for kernel in self._tdm_kernels()}),
            [32, 64, 128, 256],
        )

    def test_pingpong_only_buckets_fall_back_on_the_next_qr_tdm_bucket(self):
        # the split-KV buckets 96/160/192 hold qr_tdm_pingpong's instances, then dispatch the
        # next qr_tdm bucket up's padded instances (no copies compiled for them)
        from codegen.ops.fmha_fwd_splitkv import write_blobs

        with tempfile.TemporaryDirectory() as out:
            write_blobs(["gfx1250"], Path(out), "@*bf16_batch*", 0, [-1], "simplified")
            api = (Path(out) / "fmha_fwd_splitkv_api.cpp").read_text()
            files = {p.name for p in Path(out).glob("fmha_fwd_splitkv_d*.cpp")}
        # each dtype's head-dim buckets: "if(t.hdim_q <= h && t.hdim_v <= h) {"
        cases = re.split(r"if\(t\.hdim_q <= ", api)
        bucket = {}
        for case in cases[1:]:
            h = int(re.match(r"\d+", case).group(0))
            bucket[h] = bucket.get(h, "") + case
        for h, up in ((96, 128), (160, 256), (192, 256)):
            with self.subTest(bucket=h):
                traits = re.findall(
                    r"fmha_fwd_splitkv_traits_<(\d+), FmhaFwdBf16, [^;]*?BlockFmhaPipelineEnum::(\w+),",
                    bucket[h],
                )
                tags = [tag for _, tag in traits]
                self.assertIn("QRKSVS_TDM_PINGPONG", tags)
                self.assertIn("QRKSVS_TDM", tags)
                # pingpong's own instances first, then the aliases of bucket `up`
                last_pp = max(
                    i for i, t in enumerate(tags) if t == "QRKSVS_TDM_PINGPONG"
                )
                first_tdm = min(i for i, t in enumerate(tags) if t == "QRKSVS_TDM")
                self.assertLess(last_pp, first_tdm)
                self.assertEqual({int(d) for d, t in traits if t == "QRKSVS_TDM"}, {up})
                # d192's pingpong tile is 192/128: d192/192 takes d256's padded pingpong
                # instances, ahead of qr_tdm's
                pp_dims = {int(d) for d, t in traits if t == "QRKSVS_TDM_PINGPONG"}
                self.assertEqual(pp_dims, {h, up} if h == 192 else {h})
                self.assertFalse(
                    any(
                        f.startswith(f"fmha_fwd_splitkv_d{h}_") and "qr_tdm_v" in f
                        for f in files
                    )
                )

    def test_qr_tdm_owns_logits_soft_cap(self):
        # The soft cap entries that dispatch on an exact head dim must be
        # qr_tdm's. The head-dim check at the end keeps the set non-empty, so
        # the loop cannot pass vacuously.
        kernels = get_fwd_splitkv_blobs(
            targets=["gfx1250"],
            kernel_filter="",
            receipt=600,
            mask_impl="simplified",
            optdim_list=[-1],
        )
        exact = [
            k
            for k in kernels
            if k.F_pipeline.F_logits == "t" and k.F_pipeline.F_dpad == "f"
        ]
        for kernel in exact:
            self.assertEqual(kernel.F_pipeline.tag, "qr_tdm", kernel.filename)
        # d32 is absent by construction: its bucket carries bk0max 64, so the
        # unpadded head-dim predicate reads 64 and can never hold under
        # hdim_q <= 32. get_fwd_splitkv_blobs drops those as unreachable.
        self.assertEqual(sorted({kernel.F_hdim for kernel in exact}), [64, 128, 256])


class SplitKVDecodeDispatch(unittest.TestCase):
    """Which split-kv instances decode a seqlen_q 1 GQA problem, and how."""

    @staticmethod
    def _kernels(target, receipt):
        return get_fwd_splitkv_blobs(
            targets=[target],
            kernel_filter="",
            receipt=receipt,
            mask_impl="simplified",
            optdim_list=[-1],
        )

    def test_only_tdm_pipelines_merge_heads_with_a_sink(self):
        # A merged row reads the sink logit of its own q head only in the TDM
        # pipelines; the others read one per kv head. qr_tdm_pingpong's split-KV (the
        # forward kernel) merges in batch mode without a bias, in its own kernel body.
        for target in ("gfx1250", "gfx942"):
            sink = [
                k
                for k in self._kernels(target, 0)
                if k.F_pipeline.F_sink == "t" and k.F_pipeline.F_bias == "no"
            ]
            self.assertTrue(sink, target)
            for kernel in sink:
                if kernel.F_pipeline.tag == "qr_tdm_pingpong":
                    self.assertIn(
                        "constexpr bool kCanMergeHeads_",
                        kernel.template,
                        kernel.filename,
                    )
                    continue
                self.assertEqual(
                    "(!true || true)" in kernel.template,
                    kernel.F_pipeline.tag == "qr_tdm",
                    kernel.filename,
                )

    def test_pingpong_takes_up_to_16_rows_ahead_of_qr_tdm(self):
        # qr_tdm_pingpong's split-KV (the forward kernel): its 16-row decode tile and its
        # 64-row one; the plain call on whole head dims in batch mode, the features (soft cap,
        # bias / ALiBi, group mode) on padded ones, those with sinks as a runtime value and the
        # masking kernel for any mask; never paged KV
        kernels = self._kernels("gfx1250", 0)
        pingpong = [k for k in kernels if k.F_pipeline.tag == "qr_tdm_pingpong"]
        self.assertTrue(pingpong)
        for kernel in pingpong:
            p = kernel.F_pipeline
            self.assertIn(kernel.F_tile.F_bm0, (16, 64), kernel.filename)
            self.assertEqual(p.F_pagedkv, "f", kernel.filename)
            padded = (p.F_dpad, p.F_dvpad) == ("t", "t")
            if p.F_logits == "t" or p.F_bias != "no" or kernel.F_mode == "group":
                self.assertTrue(padded, kernel.filename)
            if padded:
                # the masking kernel takes no mask too; sinks alone keep a no-mask one
                if p.F_mask == "s_no":
                    self.assertEqual(
                        (p.F_logits, p.F_bias, p.F_sink),
                        ("f", "no", "t"),
                        kernel.filename,
                    )
                else:
                    self.assertEqual(p.F_mask, "s_mask", kernel.filename)
            else:
                self.assertEqual(p.F_sink, "f", kernel.filename)
        for mode in ("batch", "group"):
            for bm0 in (16, 64):
                self.assertTrue(
                    [k for k in pingpong if (k.F_mode, k.F_tile.F_bm0) == (mode, bm0)]
                )
        self.assertTrue(
            [
                k
                for k in pingpong
                if (k.F_pipeline.F_logits, k.F_pipeline.F_bias) == ("t", "bias")
            ]
        )
        # Dispatch takes the first hit in emission order: rows 17 and up fall
        # through to the b64 qr_tdm entries behind the 16-row ones.
        for dtype in ("fp16", "bf16"):
            d128 = [
                (i, k.F_pipeline.tag, k.F_tile.F_bm0)
                for i, k in enumerate(kernels)
                if (k.F_dtype, k.F_hdim, k.F_mode) == (dtype, 128, "batch")
            ]
            last_pingpong = max(i for i, tag, _ in d128 if tag == "qr_tdm_pingpong")
            first_b64 = min(i for i, tag, bm0 in d128 if tag == "qr_tdm" and bm0 == 64)
            self.assertLess(last_pingpong, first_b64, dtype)


class TestGfx125GdpaCodegen(unittest.TestCase):
    """GDPA, P = act(scale * Q K^T) with no softmax: meta-tcx's d128 calls, on
    BlockFmhaPipelineQRKSVSTdmGdpaQLoop (qr_tdm_pingpong's 128-row tile)."""

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
                    self.assertTrue(kernels)
                    self.assertEqual(
                        {(k.F_hdim, k.F_pipeline.tag, k.F_tile.F_bm0) for k in kernels},
                        {(128, "qr_tdm_pingpong", 128)},
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
                        rendered = kernel.render()
                        self.assertIn(
                            "BlockAttentionActivationEnum::FAST_GELU", rendered
                        )
                        self.assertIn("BlockFmhaPipelineQRKSVSTdmGdpaQLoop", rendered)
                        if kernel.F_mode == "group":
                            self.assertEqual(pipeline.F_spad, "t")
                    names = [k.name for k in kernels]
                    self.assertEqual(len(names), len(set(names)))

    def test_only_gdpa_receipts_emit_it(self):
        # the integrations behind the other receipts never set traits.activation
        for receipt in (3, 4, 100, 200, 600):
            with self.subTest(receipt=receipt):
                self.assertEqual(self.gdpa_kernels("bf16", receipt, [64, 128, 256]), [])

    def test_gdpa_instances_add_to_softmax_ones(self):
        # GDPA adds instances and changes none of softmax's: its kernels run on GDPA's own tile
        # (GFX125_GDPA_TILE_OWNER) alone, which no softmax kernel takes, and carry the
        # activation in their names and traits.
        _, kernels = get_fwd_blobs(["gfx1250"], "*bf16*", 0, [128], "simplified")
        softmax = [k for k in kernels if k.F_pipeline.F_activation == "softmax"]
        gdpa = [k for k in kernels if k.F_pipeline.F_activation != "softmax"]
        self.assertTrue(gdpa)
        self.assertFalse(
            [
                k.name
                for k in softmax
                if k.F_tile.F_pipeline_tag == GFX125_GDPA_TILE_OWNER
            ]
        )
        softmax_names = {k.name for k in softmax}
        for kernel in gdpa:
            with self.subTest(kernel=kernel.name):
                self.assertEqual(kernel.F_tile.F_pipeline_tag, GFX125_GDPA_TILE_OWNER)
                self.assertIn("fast_gelu", kernel.name)
                self.assertIn("fast_gelu", kernel.api_trait().name)
                self.assertNotIn(kernel.name, softmax_names)

    def test_resident_instances(self):
        # Each GDPA instance comes twice: with K and V resident on the one-workgroup-a-WGP tile
        # (teams of the tile's waves: its translation unit defines CK_TILE_WARP_TEAM_SIZE ahead
        # of every include), dispatched on the batches' longest keys and the grid, and the
        # streaming q-loop for the rest.
        _, kernels = get_fwd_blobs(["gfx1250"], "*bf16*", 0, [128], "simplified")
        gdpa = [k for k in kernels if k.F_pipeline.F_activation != "softmax"]
        resident = [k for k in gdpa if k.F_tile.F_occupancy == 1]
        stream = [k for k in gdpa if k.F_tile.F_occupancy == -1]
        self.assertTrue(resident)
        self.assertEqual(len(resident) + len(stream), len(gdpa))
        self.assertEqual(
            sorted(k.name.replace("_o1_", "_") for k in resident),
            sorted(k.name for k in stream),
        )
        for kernel in resident:
            with self.subTest(kernel=kernel.name):
                body = kernel.render()
                self.assertTrue(body.startswith("#define CK_TILE_WARP_TEAM_SIZE 4\n"))
                self.assertIn("ck_tile::BlockFmhaPipelineQRKSVSTdmGdpaResidentPolicy", body)
                self.assertIn("a.max_seqlen_k <= 384", str(kernel.F_tile.F_constraint))
        for kernel in stream:
            with self.subTest(kernel=kernel.name):
                body = kernel.render()
                self.assertNotIn("CK_TILE_WARP_TEAM_SIZE", body)
                self.assertIn("ck_tile::BlockFmhaPipelineQRKSVSTdmGdpaQLoopPolicy", body)


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
        gdpa = {
            index
            for k, index in dispatch.kernel_ids
            if k.F_pipeline.F_activation == "fast_gelu"
            and k.F_pipeline.tag == "qr_tdm_pingpong"
            and k.F_tile.F_bm0 == 128
        }

        def run(**changes):
            return dispatch.run(**{"hdim_q": 128, "hdim_v": 128, **changes})[0]

        return run, gdpa

    def test_every_call_on_the_q_loop_tile(self):
        # any queries and keys, any grid: the 128-row tile's GDPA pipeline
        run, gdpa = self.dispatcher()
        self.assertTrue(gdpa)
        for grid in ({}, dict(batch=1280, nhead_q=2, nhead_k=2)):
            for seqlen_q, seqlen_k in (
                (8, 256),
                (200, 512),
                (532, 256),
                (532, 513),
                (4096, 8192),
                (100, 1024),
            ):
                with self.subTest(grid=grid, seqlen_q=seqlen_q, seqlen_k=seqlen_k):
                    self.assertIn(
                        run(seqlen_q=seqlen_q, seqlen_k=seqlen_k, activation=1, **grid),
                        gdpa,
                    )

    def test_group_mode(self):
        run, gdpa = self.dispatcher("group")
        group = dict(
            group=1, activation=1, batch=8, seqlen_q=8 * 5000, max_seqlen_q=5000
        )
        for keys in (256, 4096):
            self.assertIn(run(seqlen_k=8 * keys, **group), gdpa)

    def test_gdpa_and_softmax_never_cross(self):
        run, gdpa = self.dispatcher()
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

    def test_gdpa_head_dim_and_o_layout(self):
        # exactly d128 (no head-dim padding: 112, which the d128 bucket also takes, has no
        # GDPA kernel), and an O the q-loop's 16-byte async O store addresses
        run, gdpa = self.dispatcher()
        shape = dict(seqlen_q=532, seqlen_k=256, activation=1)
        self.assertIn(run(**shape), gdpa)
        self.assertLess(run(hdim_q=112, hdim_v=112, **shape), 0)
        for layout in (
            dict(o_offset=8),
            dict(stride_o=1028),
            dict(nhead_stride_o=132),
            dict(stride_o=16909320),
        ):
            with self.subTest(**layout):
                self.assertLess(run(**layout, **shape), 0)

    def test_gdpa_other_head_dims_have_no_kernel(self):
        pool, kernels = get_fwd_blobs(
            ["gfx1250"], "fmha_fwd_d64_bf16_*", 0, [64], "simplified"
        )
        dispatch = _CompiledDispatcher(self, pool, kernels)
        self.assertLess(
            dispatch.run(
                hdim_q=64, hdim_v=64, seqlen_q=532, seqlen_k=256, activation=1
            )[0],
            0,
        )


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
        # skip runs on qr_tdm_pingpong (its skip_min_seqlen_q variants), which stores the
        # base-2 LSE itself too
        self.assertGreaterEqual(dispatch.run(lse=1, skip=1, **self.SHAPE)[0], 0)
        (selected,) = dispatch.run(lse=1, skip=1, lse_log2=1, **self.SHAPE)
        self.assertIn(selected, qr_tdm_family)
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
