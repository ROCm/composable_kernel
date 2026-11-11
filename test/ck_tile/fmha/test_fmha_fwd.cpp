// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "example/ck_tile/01_fmha/fmha_fwd.hpp"
#include "example/ck_tile/01_fmha/fmha_fwd_runner.hpp"

#include "gtest/gtest.h"

#ifndef DataTypeConfig
#define DataTypeConfig FmhaFwdFp16 // or FmhaFwdBf16 / FmhaFwdFp8Bf16 / FmhaFwdFp32
#endif

using ::testing::Bool;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;

template <typename T>
struct TestConfigs
{
    static constexpr auto HDimValues = std::array{
        std::tuple{32, -1},
        std::tuple{64, -1},
        std::tuple{96, 128},
        std::tuple{128, -1},
        std::tuple{192, 128},
        std::tuple{192, -1},
        std::tuple{256, -1},
    };
    static constexpr auto SplitKVHDimValues = std::array{
        std::tuple{32, -1},
        std::tuple{64, -1},
        std::tuple{96, -1},
        std::tuple{128, -1},
        std::tuple{256, -1},
    };
    static constexpr auto AppendKVHDimValues = std::array{
        std::tuple{32, -1}, std::tuple{64, -1}, std::tuple{128, -1}, std::tuple{256, -1}};
    static constexpr auto ModeValues        = std::array{mode_enum::batch, mode_enum::group};
    static constexpr auto IsVRowmajorValues = std::array{true};
    static constexpr auto qscale_str        = "n";
    static constexpr auto QScaleValues      = std::array{"n"};
    static constexpr bool def_lse           = true;
    static constexpr bool def_is_v_rowmajor = true;
    static constexpr auto init_method       = "uf";
    static int adjust_seqlen(int seqlen) { return seqlen; }
    static int adjust_hdim(int hdim)
    {
        return hdim < 0 ? hdim : ck_tile::integer_least_multiple(hdim, 8);
    }
};

template <>
struct TestConfigs<FmhaFwdFp8Bf16>
{
    static constexpr auto HDimValues =
        std::array{std::tuple{64, -1}, std::tuple{128, -1}, std::tuple{256, -1}};
    static constexpr auto SplitKVHDimValues  = std::array{std::tuple{64, -1}, std::tuple{128, -1}};
    static constexpr auto AppendKVHDimValues = std::array{std::tuple{64, -1}, std::tuple{128, -1}};
    static constexpr auto ModeValues         = std::array{mode_enum::batch, mode_enum::group};
    static constexpr auto IsVRowmajorValues  = std::array{true};
    static constexpr auto qscale_str         = "pt";
    static constexpr auto QScaleValues       = std::array{"n", "pt", "ph", "bs"};
    static constexpr bool def_lse            = false;
    static constexpr bool def_is_v_rowmajor  = true;
    static constexpr auto init_method        = "3";
    // When there are no fp8 instances with padding, pad seqlen to avoid skipping most of the tests:
    // return ck_tile::integer_least_multiple(seqlen, 128);
    static int adjust_seqlen(int seqlen) { return seqlen; }
    static int adjust_hdim(int hdim)
    {
        return hdim < 0 ? hdim : ck_tile::integer_least_multiple(hdim, 16);
    }
};

// d=128 is the only head dim the whole fp8 family shares; fp8 has no d=64 tile. No
// splitkv/appendkv instances are generated, and all four quantization scales exist here.
struct Fp8FamilyTestConfigs
{
    static constexpr auto HDimValues         = std::array{std::tuple{128, -1}};
    static constexpr auto SplitKVHDimValues  = std::array<std::tuple<int, int>, 0>{};
    static constexpr auto AppendKVHDimValues = std::array<std::tuple<int, int>, 0>{};
    static constexpr auto ModeValues         = std::array{mode_enum::batch, mode_enum::group};
    static constexpr auto IsVRowmajorValues  = std::array{true};
    static constexpr auto qscale_str         = "pt";
    static constexpr auto QScaleValues       = std::array{"n", "pt", "ph", "bs"};
    static constexpr bool def_lse            = false;
    static constexpr bool def_is_v_rowmajor  = true;
    static constexpr auto init_method        = "3";
    static int adjust_seqlen(int seqlen) { return seqlen; }
    static int adjust_hdim(int hdim)
    {
        return hdim < 0 ? hdim : ck_tile::integer_least_multiple(hdim, 16);
    }
};

template <>
struct TestConfigs<FmhaFwdFp8> : Fp8FamilyTestConfigs
{
};

template <>
struct TestConfigs<FmhaFwdMxFp8>
{
    static constexpr auto HDimValues         = std::array{std::tuple{128, -1}, std::tuple{256, -1}};
    static constexpr auto SplitKVHDimValues  = std::array<std::tuple<int, int>, 0>{};
    static constexpr auto AppendKVHDimValues = std::array<std::tuple<int, int>, 0>{};
    static constexpr auto ModeValues         = std::array{mode_enum::batch, mode_enum::group};
    static constexpr auto IsVRowmajorValues  = std::array{false};
    static constexpr auto qscale_str         = "mx";
    static constexpr auto QScaleValues       = std::array{"mx"};
    static constexpr bool def_lse            = true;
    static constexpr bool def_is_v_rowmajor  = false;
    static constexpr auto init_method        = "3";
    static int adjust_seqlen(int seqlen) { return seqlen; }
    static int adjust_hdim(int hdim)
    {
        return hdim < 0 ? hdim : ck_tile::integer_least_multiple(hdim, 16);
    }
};

// MXFP8 with a BF16 O exists only on gfx125x qr_tdm_fp8_sched; d64/d256 run in the Hdim suites.
template <>
struct TestConfigs<FmhaFwdMxFp8Bf16> : TestConfigs<FmhaFwdMxFp8>
{
    static constexpr auto HDimValues        = std::array{std::tuple{128, -1}};
    static constexpr auto IsVRowmajorValues = std::array{true};
    static constexpr bool def_is_v_rowmajor = true;
};

template <>
struct TestConfigs<FmhaFwdMxFp4>
{
    static constexpr auto HDimValues         = std::array{std::tuple{128, -1}, std::tuple{256, -1}};
    static constexpr auto SplitKVHDimValues  = std::array<std::tuple<int, int>, 0>{};
    static constexpr auto AppendKVHDimValues = std::array<std::tuple<int, int>, 0>{};
    static constexpr auto ModeValues         = std::array{mode_enum::batch, mode_enum::group};
    static constexpr auto IsVRowmajorValues  = std::array{false};
    static constexpr auto qscale_str         = "mx";
    static constexpr auto QScaleValues       = std::array{"mx"};
    static constexpr bool def_lse            = true;
    static constexpr bool def_is_v_rowmajor  = false;
    static constexpr auto init_method        = "3";
    static int adjust_seqlen(int seqlen)
    {
        return seqlen < 0 ? seqlen : ck_tile::integer_least_multiple(seqlen, 2);
    }
    static int adjust_hdim(int hdim)
    {
        return hdim < 0 ? hdim : ck_tile::integer_least_multiple(hdim, 32);
    }
};

template <>
struct TestConfigs<FmhaFwdFp32>
{
    static constexpr auto HDimValues = std::array{
        std::tuple{32, -1},
        std::tuple{48, -1},
        std::tuple{64, -1},
        std::tuple{96, 128},
        std::tuple{128, -1},
        std::tuple{192, -1},
        std::tuple{256, -1},
    };
    static constexpr auto SplitKVHDimValues  = std::array<std::tuple<int, int>, 0>{};
    static constexpr auto AppendKVHDimValues = std::array<std::tuple<int, int>, 0>{};
    static constexpr auto ModeValues         = std::array{mode_enum::batch, mode_enum::group};
    static constexpr auto IsVRowmajorValues  = std::array{true};
    static constexpr auto qscale_str         = "n";
    static constexpr auto QScaleValues       = std::array{"n"};
    static constexpr bool def_lse            = true;
    static constexpr bool def_is_v_rowmajor  = true;
    static constexpr auto init_method        = "uf";
    static int adjust_seqlen(int seqlen) { return seqlen; }
    static int adjust_hdim(int hdim) { return hdim; }
};

static auto HDimValues         = ValuesIn(TestConfigs<DataTypeConfig>::HDimValues);
static auto SplitKVHDimValues  = ValuesIn(TestConfigs<DataTypeConfig>::SplitKVHDimValues);
static auto AppendKVHDimValues = ValuesIn(TestConfigs<DataTypeConfig>::AppendKVHDimValues);
static auto ModeValues         = ValuesIn(TestConfigs<DataTypeConfig>::ModeValues);
static auto IsVRowmajorValues  = ValuesIn(TestConfigs<DataTypeConfig>::IsVRowmajorValues);
#ifdef CK_TILE_TEST_FMHA_QSCALE_SWEEP
static auto QScaleValues = ValuesIn(TestConfigs<DataTypeConfig>::QScaleValues);
#endif
constexpr static auto qscale_str = TestConfigs<DataTypeConfig>::qscale_str;
constexpr bool def_lse           = TestConfigs<DataTypeConfig>::def_lse;
constexpr bool def_is_v_rowmajor = TestConfigs<DataTypeConfig>::def_is_v_rowmajor;
constexpr auto init_method       = TestConfigs<DataTypeConfig>::init_method;
int adjust_seqlen(int seqlen) { return TestConfigs<DataTypeConfig>::adjust_seqlen(seqlen); }
int adjust_hdim(int hdim) { return TestConfigs<DataTypeConfig>::adjust_hdim(hdim); }

// Random seed used for initializing input tensors. 0 for non-deterministic seed
CK_TILE_DECLARE_ENV_VAR(CK_TILE_TEST_SEED, uint64_t, 123456)

// Whether to run long tests (from smoke_test_fwd.sh)
CK_TILE_DECLARE_ENV_VAR_BOOL(CK_TILE_FMHA_LONG_TESTS)

#define CHECK_RESULT(result)                                      \
    do                                                            \
    {                                                             \
        if(result == fwd_result::no_instance)                     \
            GTEST_SKIP() << "No instance for current parameters"; \
        ASSERT_EQ(result, fwd_result::success);                   \
    } while(0)

const ck_tile::stream_config stream_config{
    nullptr, // stream_id_
    false,   // time_kernel_
    1,       // log_level_
    0,       // cold_niters_
    1,       // nrepeat_
    true,    // is_gpu_timer_
    false,   // flush_cache_
    1,       // rotating_count_
};

#define COMMON_ARGS_INIT(init)                                                               \
    init, static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))), 1, 0, 1, \
        stream_config

#define COMMON_ARGS COMMON_ARGS_INIT(init_method)

auto EnableTestIf(bool condition)
{
    return ValuesIn(condition ? std::vector<bool>{true} : std::vector<bool>{});
}

class AllLong : public TestWithParam<
                    std::tuple<bool,
                               std::tuple<int, int>,
                               bool,
                               bool,
                               mode_enum,
                               bool,
                               std::string,
                               float,
                               std::tuple<int, int, int, int, int, int, int, int, std::string>>>
{
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AllLong);

// Test cases from example/ck_tile/01_fmha/script/smoke_test_fwd.sh

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    AllLong,
    Combine(EnableTestIf(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_FMHA_LONG_TESTS))),
            HDimValues,
            Bool(),
            IsVRowmajorValues,
            ModeValues,
            Bool(),
            Values("n", "e", "a"),
            Values(0.0f, 0.2f),
            Values(std::tuple{2, 2, 1, 16, -1, 55, 256, -1, "0"},
                   std::tuple{1, 3, -1, -1, -1, 100, 51, -1, "0"},
                   std::tuple{2, 1, -1, 16, -1, 99, 256, -1, "1"},
                   std::tuple{1, 2, 1, -1, -1, 1024, 256, -1, "2"},
                   std::tuple{2, 1, -1, -1, 24, 3, 99, -1, "2"},
                   std::tuple{3, 2, 1, -1, -1, 200, 520, -1, "t:128,30"},
                   std::tuple{2, 1, -1, -1, -1, 99, 32, -1, "b:4,35"},
                   std::tuple{1, 2, 1, -1, -1, 33, 0, -1, "2"},
                   std::tuple{1, 2, 1, -1, -1, 1, 10, 32, "2"},
                   // d=128 on gfx1250: seqlen<2048 selects the qr_tdm decode (b64) tile,
                   // seqlen>=2048 the prefill (b128) tile
                   std::tuple{1, 2, 1, 128, 128, 512, 512, -1, "0"},
                   std::tuple{1, 2, 1, 128, 128, 2048, 2048, -1, "0"})));

TEST_P(AllLong, DataTypeConfig)
{
    auto [_, hdims, perm, is_v_rowmajor, mode, lse, bias_str, p_drop, dims_mask] = GetParam();
    auto [hdim_q, hdim_v]                                                        = hdims;
    auto [batch, nhead, nhead_k, hdim_q_, hdim_v_, seqlen_q, seqlen_k, seqlen_kpad, mask_str] =
        dims_mask;

    hdim_q = hdim_q_ == -1 ? hdim_q : hdim_q_;
    hdim_v = hdim_v_ == -1 ? hdim_v : hdim_v_;

    auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {adjust_seqlen(seqlen_q)},
                                               {adjust_seqlen(seqlen_k)},
                                               hdim_q,
                                               hdim_v,
                                               0,             // seqlen_knew
                                               {-1},          // seqlen_qpads
                                               {seqlen_kpad}, // seqlen_kpads
                                               {},            // q_eff_lens_per_batch
                                               {},            // kv_eff_lens_per_batch
                                               0,             // rotary_dim
                                               perm,          // i_perm
                                               perm,          // o_perm
                                               0,             // scale_s
                                               0,             // logits_soft_cap
                                               is_v_rowmajor, // is_v_rowmajor
                                               lse,           // lse
                                               0,             // page_block_size
                                               false,         // use_cache_batch_idx
                                               bias_str,      // bias_str
                                               p_drop,        // p_drop
                                               123,           // drop_seed
                                               1024,          // drop_offset
                                               false,         // drop_prefs
                                               mask_str,      // mask_str
                                               qscale_str,
                                               true, // is_rotary_interleaved
                                               1,    // num_splits
                                               COMMON_ARGS);
    CHECK_RESULT(result);
}

TEST(TestCkTileFmhaFwd, QrTdmLdsArenaDecode)
{
    if constexpr(ck_tile::is_any_of<DataTypeConfig, FmhaFwdFp16, FmhaFwdBf16>::value)
    {
        if(!ck_tile::is_gfx125_supported())
            GTEST_SKIP() << "qr_tdm LDS arena is only supported on gfx1250";

        std::string selected_kernel;
        auto decode = fmha_fwd_run<DataTypeConfig>(mode_enum::batch,
                                                   1,
                                                   4,
                                                   2,
                                                   {127},
                                                   {509},
                                                   128,
                                                   128,
                                                   0,
                                                   {-1},
                                                   {-1},
                                                   {},
                                                   {},
                                                   0,
                                                   true,
                                                   true,
                                                   0,
                                                   0,
                                                   true,
                                                   false,
                                                   0,
                                                   false,
                                                   "n",
                                                   0.0f,
                                                   0,
                                                   0,
                                                   false,
                                                   "0",
                                                   qscale_str,
                                                   true,
                                                   1,
                                                   COMMON_ARGS,
                                                   std::nullopt,
                                                   &selected_kernel);
        ASSERT_EQ(decode, fwd_result::success);
        EXPECT_NE(selected_kernel.find("_qr_tdm_"), std::string::npos);
    }
}

TEST(TestCkTileFmhaFwd, QrTdmLdsArenaPrefill)
{
    if constexpr(ck_tile::is_any_of<DataTypeConfig, FmhaFwdFp16, FmhaFwdBf16>::value)
    {
        if(!ck_tile::is_gfx125_supported())
            GTEST_SKIP() << "qr_tdm LDS arena is only supported on gfx1250";

        std::string selected_kernel;
        auto result = fmha_fwd_run<DataTypeConfig>(mode_enum::batch,
                                                   1,
                                                   4,
                                                   2,
                                                   {2049},
                                                   {2177},
                                                   128,
                                                   128,
                                                   0,
                                                   {-1},
                                                   {-1},
                                                   {},
                                                   {},
                                                   0,
                                                   true,
                                                   true,
                                                   0,
                                                   0,
                                                   true,
                                                   true,
                                                   0,
                                                   false,
                                                   "a:1",
                                                   0.0f,
                                                   0,
                                                   0,
                                                   false,
                                                   "1",
                                                   qscale_str,
                                                   true,
                                                   1,
                                                   COMMON_ARGS,
                                                   std::nullopt,
                                                   &selected_kernel);
        ASSERT_EQ(result, fwd_result::success);
        EXPECT_NE(selected_kernel.find("_qr_tdm_"), std::string::npos);
    }
}

// Batch-mode XCD remap of the block index (gfx1250): 3 batches, seqlen_q 300, so
// 3 (bm0=128) or 5 (bm0=64) m-tiles per (batch, head) on grids that are not a
// multiple of the XCD count, where the remap is not the identity. Each case names
// the GetTileIndex branch it reaches. Square causal bhsd grids take the
// batch-local causal order instead, hence seqlen_k 400 for the bhsd causal case.
// Masked bshd grids remap only with at least 8 (batch, head) rows per m-tile: 3 x
// 16 heads covers both m-tile counts, 3 x 5 neither.
class BatchXCDRemap
    : public TestWithParam<
          std::tuple<bool, int, int, std::vector<int>, std::vector<int>, const char*>>
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    BatchXCDRemap,
    // bhsd/bshd, nhead, seqlen_k, q_eff_lens, kv_eff_lens, mask
    Values(
        // bhsd remap
        std::tuple{true, 5, 300, std::vector<int>{}, std::vector<int>{}, "0"},
        std::tuple{true, 5, 400, std::vector<int>{}, std::vector<int>{}, "2"},
        // bshd remap
        std::tuple{false, 5, 300, std::vector<int>{}, std::vector<int>{}, "0"},
        std::tuple{false, 16, 300, std::vector<int>{}, std::vector<int>{}, "2"},
        // bshd masked, too few heads per m-tile: round-robin
        std::tuple{false, 5, 300, std::vector<int>{}, std::vector<int>{}, "2"},
        // per-batch lengths: round-robin, early exit past each batch's seqlen_q
        std::tuple{true, 5, 300, std::vector<int>{300, 64, 1}, std::vector<int>{300, 100, 7}, "0"},
        std::tuple{
            false, 16, 300, std::vector<int>{300, 64, 1}, std::vector<int>{300, 100, 7}, "2"}));

TEST_P(BatchXCDRemap, DataTypeConfig)
{
    auto [perm, nhead, seqlen_k, q_eff_lens, kv_eff_lens, mask_str] = GetParam();

    auto result = fmha_fwd_run<DataTypeConfig>(mode_enum::batch,
                                               3, // batch
                                               nhead,
                                               -1, // nhead_k
                                               {adjust_seqlen(300)},
                                               {adjust_seqlen(seqlen_k)},
                                               adjust_hdim(128),
                                               adjust_hdim(128),
                                               0,    // seqlen_knew
                                               {-1}, // seqlen_qpads
                                               {-1}, // seqlen_kpads
                                               q_eff_lens,
                                               kv_eff_lens,
                                               0,    // rotary_dim
                                               perm, // i_perm
                                               perm, // o_perm
                                               0,    // scale_s
                                               0,    // logits_soft_cap
                                               def_is_v_rowmajor,
                                               def_lse,
                                               0,     // page_block_size
                                               false, // use_cache_batch_idx
                                               "n",   // bias_str
                                               0.0f,  // p_drop
                                               0,     // drop_seed
                                               0,     // drop_offset
                                               false, // drop_prefs
                                               mask_str,
                                               qscale_str,
                                               true, // is_rotary_interleaved
                                               1,    // num_splits
                                               COMMON_ARGS);
    CHECK_RESULT(result);
}

// d128 runs qr_tdm_pingpong below 640 rows with at least 2 (batch, head) pairs per
// multiprocessor (GFX125_QR_TDM_PINGPONG_RANGE in fmha_fwd.py), so each case sizes its
// grid to reach it on this device: the batch in batch mode; in group mode, where this
// runner passes the packed total length as max_seqlen_q (kept below 640), the head
// count for a given number of sequences (12 by default). The unpadded instance serves
// any seqlen_k except with an elementwise bias; causal masks take its causal kernel
// ("_mc_", without LSE, bias or sink). ALiBi without a mask makes every K tile raise the
// row maxima of the steep heads past the rebase threshold, so those cases take the
// finite-to-finite rescale of o_acc and l on every trip, with and without LSE.
std::pair<int, int> PingPongGrid(mode_enum mode, int nhead, int group_batch = 12)
{
    int device = 0;
    hipDeviceProp_t props{};
    EXPECT_EQ(hipGetDevice(&device), hipSuccess);
    EXPECT_EQ(hipGetDeviceProperties(&props, device), hipSuccess);
    const int pairs = 2 * props.multiProcessorCount;
    if(mode == mode_enum::batch)
        return {ck_tile::integer_divide_ceil(pairs, nhead), nhead};
    return {group_batch, ck_tile::integer_divide_ceil(pairs, group_batch)};
}

// max_seqlen_q <= 16 takes qr_tdm_pingpong's 16-row tile, which spends its warps on N
// instead of M and so sends S through LDS on its way from gemm_0 to gemm_1. A
// distribution mismatch there permutes P's columns within a row: the row sums, and so
// the LSE, stay right while O is wrong, so O is what these cases check. seqlen_k covers
// a tile multiple and a padded tail; the masks cover the causal and windowed paths. The
// grid is sized as for QrTdmPingPong; group mode packs 2 sequences of 8 rows.
class QrTdmShortQuery : public TestWithParam<std::tuple<mode_enum, int, int, const char*>>
{
};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         QrTdmShortQuery,
                         // mode, seqlen_q, seqlen_k, mask
                         Values(std::tuple{mode_enum::batch, 1, 509, "0"},
                                std::tuple{mode_enum::batch, 8, 64, "0"},
                                std::tuple{mode_enum::batch, 16, 509, "1"},
                                std::tuple{mode_enum::batch, 8, 509, "2"},
                                std::tuple{mode_enum::batch, 16, 64, "t:50,64"},
                                std::tuple{mode_enum::group, 8, 509, "2"}));

TEST_P(QrTdmShortQuery, DataTypeConfig)
{
    if constexpr(ck_tile::is_any_of<DataTypeConfig, FmhaFwdFp16, FmhaFwdBf16>::value)
    {
        if(!ck_tile::is_gfx125_supported())
            GTEST_SKIP() << "qr_tdm_pingpong is only supported on gfx1250";

        auto [mode, seqlen_q, seqlen_k, mask_str] = GetParam();
        const auto [batch, nhead]                 = PingPongGrid(mode, 4, 2);

        std::string selected_kernel;
        // MHA: pack-GQA would fold the heads into seqlen_q and move the tile choice.
        auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                                   batch,
                                                   nhead,
                                                   -1,
                                                   {seqlen_q},
                                                   {seqlen_k},
                                                   128,
                                                   128,
                                                   0,
                                                   {-1},
                                                   {-1},
                                                   {},
                                                   {},
                                                   0,
                                                   true,
                                                   true,
                                                   0,
                                                   0,
                                                   true,
                                                   true,
                                                   0,
                                                   false,
                                                   "n",
                                                   0.0f,
                                                   0,
                                                   0,
                                                   false,
                                                   mask_str,
                                                   qscale_str,
                                                   true,
                                                   1,
                                                   COMMON_ARGS,
                                                   std::nullopt,
                                                   &selected_kernel);
        ASSERT_EQ(result, fwd_result::success);
        EXPECT_NE(selected_kernel.find("_qr_tdm_pingpong_"), std::string::npos) << selected_kernel;
        EXPECT_NE(selected_kernel.find("_b16x64"), std::string::npos) << selected_kernel;
    }
}

class QrTdmPingPong
    : public TestWithParam<
          std::tuple<mode_enum, int, int, int, const char*, const char*, bool, const char*, bool>>
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    QrTdmPingPong,
    // mode, nhead (group mode: from the grid), seqlen_q, seqlen_k, mask, bias, lse, name part,
    // BHSD (i_perm/o_perm; BSHD otherwise)
    Values(std::tuple{mode_enum::batch, 4, 64, 64, "0", "n", true, "_npad_", false},
           std::tuple{mode_enum::batch, 4, 64, 100, "0", "n", true, "_npad_", false},
           std::tuple{mode_enum::batch, 4, 64, 100, "0", "e", true, "_pssk_", false},
           std::tuple{mode_enum::batch, 4, 64, 128, "0", "e", true, "_npad_", false},
           std::tuple{mode_enum::batch, 8, 64, 192, "0", "a", true, "_alibi_", false},
           std::tuple{mode_enum::batch, 8, 64, 192, "0", "a", false, "_alibi_", false},
           std::tuple{mode_enum::batch, 4, 232, 232, "t:50,64", "n", true, "_mask_", false},
           std::tuple{mode_enum::batch, 8, 100, 100, "2", "n", false, "_mc_", false},
           std::tuple{mode_enum::group, 4, 50, 100, "2", "n", true, "", false},
           std::tuple{mode_enum::batch, 4, 64, 64, "0", "n", true, "_npad_", true},
           std::tuple{mode_enum::batch, 8, 64, 192, "0", "a", true, "_alibi_", true},
           std::tuple{mode_enum::group, 4, 50, 100, "2", "n", true, "", true}));

TEST_P(QrTdmPingPong, DataTypeConfig)
{
    if constexpr(ck_tile::is_any_of<DataTypeConfig, FmhaFwdFp16, FmhaFwdBf16>::value)
    {
        if(!ck_tile::is_gfx125_supported())
            GTEST_SKIP() << "qr_tdm_pingpong is only supported on gfx1250";

        auto [mode, nhead_hint, seqlen_q, seqlen_k, mask_str, bias_str, lse, tag, perm] =
            GetParam();
        const auto [batch, nhead] = PingPongGrid(mode, nhead_hint);

        std::string selected_kernel;
        auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                                   batch,
                                                   nhead,
                                                   -1,
                                                   {seqlen_q},
                                                   {seqlen_k},
                                                   128,
                                                   128,
                                                   0,
                                                   {-1},
                                                   {-1},
                                                   {},
                                                   {},
                                                   0,
                                                   perm,
                                                   perm,
                                                   0,
                                                   0,
                                                   true,
                                                   lse,
                                                   0,
                                                   false,
                                                   bias_str,
                                                   0.0f,
                                                   0,
                                                   0,
                                                   false,
                                                   mask_str,
                                                   qscale_str,
                                                   true,
                                                   1,
                                                   COMMON_ARGS,
                                                   std::nullopt,
                                                   &selected_kernel);
        ASSERT_EQ(result, fwd_result::success);
        EXPECT_NE(selected_kernel.find("_b64x"), std::string::npos) << selected_kernel;
        EXPECT_NE(selected_kernel.find("_qr_tdm_pingpong_"), std::string::npos) << selected_kernel;
        EXPECT_NE(selected_kernel.find(tag), std::string::npos) << selected_kernel;
    }
}

// qr_tdm_pingpong with a sink, and below d128: the dpad/dvpad instances, whose O goes
// through the LDS-shuffle epilogue with a padded N. Same grids as QrTdmPingPong.
class QrTdmPingPongEdges
    : public TestWithParam<std::tuple<int, int, const char*, const char*, int, const char*>>
{
};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         QrTdmPingPongEdges,
                         // hdim_q, hdim_v, mask, bias, init_sink, name part
                         Values(std::tuple{128, 128, "0", "n", 1, "_sink"},
                                std::tuple{128, 128, "t:50,64", "n", 1, "_sink"},
                                std::tuple{120, 120, "2", "n", 0, "_pddv_"},
                                std::tuple{128, 120, "0", "e", 0, "_psskddv_"}));

TEST_P(QrTdmPingPongEdges, DataTypeConfig)
{
    if constexpr(ck_tile::is_any_of<DataTypeConfig, FmhaFwdFp16, FmhaFwdBf16>::value)
    {
        if(!ck_tile::is_gfx125_supported())
            GTEST_SKIP() << "qr_tdm_pingpong is only supported on gfx1250";

        auto [hdim_q, hdim_v, mask_str, bias_str, init_sink, tag] = GetParam();
        const auto [batch, nhead] = PingPongGrid(mode_enum::batch, 4);

        std::string selected_kernel;
        auto result = fmha_fwd_run<DataTypeConfig>(
            mode_enum::batch,
            batch,
            nhead,
            -1,
            {100},
            {300},
            hdim_q,
            hdim_v,
            0,
            {-1},
            {-1},
            {},
            {},
            0,
            false,
            false,
            0,
            0,
            true,
            true, // lse
            0,
            false,
            bias_str,
            0.0f,
            0,
            0,
            false,
            mask_str,
            qscale_str,
            true,
            1,
            init_method,
            static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
            1, // do_validation
            init_sink,
            1, // pack_gqa
            stream_config,
            std::nullopt,
            &selected_kernel);
        ASSERT_EQ(result, fwd_result::success);
        EXPECT_NE(selected_kernel.find("_qr_tdm_pingpong_"), std::string::npos) << selected_kernel;
        EXPECT_NE(selected_kernel.find(tag), std::string::npos) << selected_kernel;
    }
}

// gfx125x-only: qr_tdm correctness at the head dims this PR added but that the
// generic General sweep does not cover -- (160,160) and (96,96). The generic
// suite already exercises 32/64/128/192-128 through qr_tdm but never asserts
// the pipeline, and never requests 160 or the true 96/96 tile. Non-multiple
// seqlens select the seqlen-padded instances.
//
// In batch mode these cases also lock the bm0 crossover: seqlen_q below the
// measured N=128 (see GFX125_QR_TDM_BM0_CROSSOVER_MAX_SEQLEN_Q in fmha_fwd.py)
// must select the bm0=64 tile, and at/above N=128 the bm0=128 tile. seqlen_q=128
// covers the boundary; 99/127 cover below it. Group mode keys dispatch on the
// packed total q length, not this seqlen, so the bm0 assertion is batch-only.
class QrTdmNewHeadDim
    : public TestWithParam<std::tuple<mode_enum, std::tuple<int, int, int, int, const char*>>>
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    QrTdmNewHeadDim,
    Combine(Values(mode_enum::batch, mode_enum::group),
            Values(std::tuple{160, 160, 127, 509, "0"}, // 160 dense, seqlen-padded, bm0=64
                   std::tuple{160, 160, 128, 256, "0"}, // 160 at N=128: bm0=128
                   std::tuple{160, 160, 99, 256, "1"},  // 160 causal, bm0=64
                   std::tuple{96, 96, 127, 509, "0"},   // 96/96 dense, bm0=64
                   std::tuple{96, 96, 128, 256, "0"},   // 96/96 at N=128: bm0=128
                   std::tuple{96, 96, 99, 256, "1"}))); // 96/96 causal, bm0=64

TEST_P(QrTdmNewHeadDim, DataTypeConfig)
{
    if constexpr(ck_tile::is_any_of<DataTypeConfig, FmhaFwdFp16, FmhaFwdBf16>::value)
    {
        if(!ck_tile::is_gfx125_supported())
            GTEST_SKIP() << "qr_tdm is only supported on gfx1250";

        auto [mode, dims]                                   = GetParam();
        auto [hdim_q, hdim_v, seqlen_q, seqlen_k, mask_str] = dims;

        std::string selected_kernel;
        auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                                   2,
                                                   2,
                                                   2,
                                                   {adjust_seqlen(seqlen_q)},
                                                   {adjust_seqlen(seqlen_k)},
                                                   adjust_hdim(hdim_q),
                                                   adjust_hdim(hdim_v),
                                                   0,
                                                   {-1},
                                                   {-1},
                                                   {},
                                                   {},
                                                   0,
                                                   true,
                                                   true,
                                                   0,
                                                   0,
                                                   true,
                                                   false,
                                                   0,
                                                   false,
                                                   "n",
                                                   0.0f,
                                                   0,
                                                   0,
                                                   false,
                                                   mask_str,
                                                   qscale_str,
                                                   true,
                                                   1,
                                                   COMMON_ARGS,
                                                   std::nullopt,
                                                   &selected_kernel);
        ASSERT_EQ(result, fwd_result::success);
        EXPECT_NE(selected_kernel.find("_qr_tdm_"), std::string::npos);

        // bm0 crossover at max_seqlen_q == 128 (measured; see
        // GFX125_QR_TDM_BM0_CROSSOVER_MAX_SEQLEN_Q in fmha_fwd.py). The tile is
        // embedded in the kernel name as "_b<bm0>x<bn0>x...". Only assert in batch
        // mode: there a.max_seqlen_q == seqlen_q, so the input directly drives tile
        // selection. In group mode a.max_seqlen_q is the packed total q length across
        // the batch (sum of per-sequence lengths), not the per-sequence seqlen, so a
        // single input seqlen does not pin dispatch to one side of the crossover.
        if(mode == mode_enum::batch)
        {
            if(seqlen_q >= 128)
                EXPECT_NE(selected_kernel.find("_b128x"), std::string::npos) << selected_kernel;
            else
                EXPECT_NE(selected_kernel.find("_b64x"), std::string::npos) << selected_kernel;
        }
    }
}

// gfx125x-only: qr_tdm at the large head dims whose LDS arena only fits a single
// unguarded bm0=64 tile with no bm0=128 fallback -- (256,256) and (512,512).
// General reaches 256 through HDimValues but never 512. Keeping 512 out of
// HDimValues is a deliberate cost choice rather than a hard constraint: AllLong
// crosses that list with dropout and bias tuples, qr_tdm emits no dropout
// instance and the qr fallback stops at head dim 256 (the hdim <= 256 gate in
// fmha_fwd.py), so those rows would only ever CHECK_RESULT-skip -- paying full
// d512 codegen and build time to produce skips.
//
// Beyond the numerics these cases pin the codegen-level dispatch properties:
// bm0 stays 64 at every seqlen_q (the crossover QrTdmNewHeadDim checks does not
// apply -- there is no second tile to cross to), V stays single-buffered
// (_nkvlp_), progressive K reload stays on (_plk_), and the mask variant matches
// what the trait combination is supposed to select. The LDS layout behind those
// flags is already pinned at compile time by test_qr_tdm_lds_padding.cpp and
// test_qr_tdm_progressive_k_lds.cpp; what is new here is that the instance the
// runtime dispatcher actually hands out still has them. At 512 a regression to
// double-buffered V would not even link (the policy static_assert fires first),
// so there _nkvlp_ is belt-and-braces; at 256 it is load-bearing, since a double
// buffer still fits 128 KiB and would silently cost the bank-conflict padding.
class QrTdmLargeHeadDim
    : public TestWithParam<
          std::tuple<mode_enum, std::tuple<int, int, int, const char*, const char*, bool, bool>>>
{
};

// hdim, seqlen_q, seqlen_k, mask, bias, lse, perm. Non-multiple seqlens select the
// seqlen-padded instances; seqlen_k > seqlen_q and seqlen_q > seqlen_k are both
// represented, as are top-left, bottom-right and a live sliding window.
//
// lse is a parameter rather than def_lse because the simplified-mask causal
// specialisation (_mc, see get_pipelines in fmha_fwd.py) is only emitted for
// bias == "no" && lse == "f" && sink == "f". Pinning lse to def_lse (true for
// fp16/bf16) would route every causal row to the generic _mask instance and
// leave _mc with no coverage at these head dims at all.
INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    QrTdmLargeHeadDim,
    Combine(Values(mode_enum::batch, mode_enum::group),
            Values(std::tuple{512, 127, 509, "0", "n", true, false}, // padded seqlens, bshd
                   std::tuple{512, 128, 256, "0", "n", false, true}, // tile-multiple seqlen_q
                   std::tuple{512, 99, 256, "1", "n", false, true},  // causal top-left -> _mc
                   std::tuple{512, 256, 256, "2", "n", false, true}, // causal bottom-r -> _mc
                   std::tuple{512, 99, 256, "1", "n", true, true},   // lse -> generic _mask
                   std::tuple{512, 200, 520, "t:128,30", "n", false, true}, // window -> _mask
                   std::tuple{512, 127, 509, "0", "e", true, true},         // elementwise bias
                   std::tuple{512, 99, 256, "1", "a", false, true}, // alibi -> generic _mask
                   std::tuple{256, 127, 509, "0", "n", true, false},
                   std::tuple{256, 128, 256, "0", "n", false, true},
                   std::tuple{256, 99, 256, "1", "n", false, true},
                   std::tuple{256, 256, 256, "2", "n", false, true},
                   std::tuple{256, 99, 256, "1", "n", true, true},
                   std::tuple{256, 200, 520, "t:128,30", "n", false, true},
                   std::tuple{256, 127, 509, "0", "e", true, true},
                   std::tuple{256, 99, 256, "1", "a", false, true})));

TEST_P(QrTdmLargeHeadDim, DataTypeConfig)
{
    if constexpr(ck_tile::is_any_of<DataTypeConfig, FmhaFwdFp16, FmhaFwdBf16>::value)
    {
        if(!ck_tile::is_gfx125_supported())
            GTEST_SKIP() << "qr_tdm is only supported on gfx1250";

        auto [mode, dims]                                              = GetParam();
        auto [hdim, seqlen_q, seqlen_k, mask_str, bias_str, lse, perm] = dims;

        const int adjusted_hdim = adjust_hdim(hdim);

        std::string selected_kernel;
        auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                                   2, // batch
                                                   2, // nhead
                                                   2, // nhead_k
                                                   {adjust_seqlen(seqlen_q)},
                                                   {adjust_seqlen(seqlen_k)},
                                                   adjusted_hdim,     // hdim_q
                                                   adjusted_hdim,     // hdim_v
                                                   0,                 // seqlen_knew
                                                   {-1},              // seqlen_qpads
                                                   {-1},              // seqlen_kpads
                                                   {},                // q_eff_lens_per_batch
                                                   {},                // kv_eff_lens_per_batch
                                                   0,                 // rotary_dim
                                                   perm,              // i_perm
                                                   perm,              // o_perm
                                                   0,                 // scale_s
                                                   0,                 // logits_soft_cap
                                                   def_is_v_rowmajor, // is_v_rowmajor
                                                   lse,               // lse
                                                   0,                 // page_block_size
                                                   false,             // use_cache_batch_idx
                                                   bias_str,          // bias_str
                                                   0.0f,              // p_drop
                                                   0,                 // drop_seed
                                                   0,                 // drop_offset
                                                   false,             // drop_prefs
                                                   mask_str,          // mask_str
                                                   qscale_str,
                                                   true, // is_rotary_interleaved
                                                   1,    // num_splits
                                                   COMMON_ARGS,
                                                   std::nullopt,
                                                   &selected_kernel);
        ASSERT_EQ(result, fwd_result::success)
            << "no instance for d" << adjusted_hdim << "; check that --optdim in "
            << "example/ck_tile/01_fmha/CMakeLists.txt still lists this head dim";

        EXPECT_NE(selected_kernel.find("_qr_tdm_"), std::string::npos) << selected_kernel;
        EXPECT_NE(selected_kernel.find("_d" + std::to_string(adjusted_hdim) + "_"),
                  std::string::npos)
            << selected_kernel;
        // Single tile: bm0 is 64 regardless of seqlen_q and of batch vs group,
        // unlike the head dims QrTdmNewHeadDim covers.
        EXPECT_NE(selected_kernel.find("_b64x"), std::string::npos) << selected_kernel;
        EXPECT_NE(selected_kernel.find("_nkvlp_"), std::string::npos) << selected_kernel;
        EXPECT_NE(selected_kernel.find("_plk_"), std::string::npos) << selected_kernel;

        // Mask variant: no mask stays _nmask_; causal takes the _mc specialisation
        // only when nothing else forces the generic path (see the INSTANTIATE
        // comment); windows and causal+bias/+lse fall back to generic _mask.
        const bool is_causal      = mask_str == std::string("1") || mask_str == std::string("2");
        const char* expected_mask = mask_str == std::string("0") ? "_nmask_"
                                    : (is_causal && bias_str == std::string("n") && !lse)
                                        ? "_mc_"
                                        : "_mask_";
        EXPECT_NE(selected_kernel.find(expected_mask), std::string::npos)
            << "expected " << expected_mask << " in " << selected_kernel;
        EXPECT_NE(selected_kernel.find(lse ? "_lse_" : "_nlse_"), std::string::npos)
            << selected_kernel;
    }
    else
    {
        GTEST_SKIP() << "qr_tdm large head dims (256/512) are fp16/bf16 only";
    }
}

class General
    : public TestWithParam<std::tuple<std::tuple<int, int>,
                                      bool,
                                      bool,
                                      mode_enum,
                                      std::tuple<int, int, int, int, int, int, std::string>>>
{
};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         General,
                         Combine(HDimValues,
                                 Bool(),
                                 IsVRowmajorValues,
                                 ModeValues,
                                 Values(std::tuple{2, 2, 1, 55, 256, -1, "0"},
                                        std::tuple{1, 3, -1, 100, 51, -1, "0"},
                                        std::tuple{2, 1, -1, 99, 256, -1, "1"},
                                        std::tuple{1, 2, 1, 1024, 256, -1, "2"},
                                        std::tuple{2, 1, -1, 3, 99, -1, "2"},
                                        std::tuple{1, 2, 1, 33, 0, -1, "2"},
                                        std::tuple{1, 2, 1, 1, 10, 32, "2"})));

TEST_P(General, DataTypeConfig)
{
    auto [hdims, perm, is_v_rowmajor, mode, dims_mask]                      = GetParam();
    auto [hdim_q, hdim_v]                                                   = hdims;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, seqlen_kpad, mask_str] = dims_mask;

    auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {adjust_seqlen(seqlen_q)},
                                               {adjust_seqlen(seqlen_k)},
                                               hdim_q,
                                               hdim_v,
                                               0,             // seqlen_knew
                                               {-1},          // seqlen_qpads
                                               {seqlen_kpad}, // seqlen_kpads
                                               {},            // q_eff_lens_per_batch
                                               {},            // kv_eff_lens_per_batch
                                               0,             // rotary_dim
                                               perm,          // i_perm
                                               perm,          // o_perm
                                               0,             // scale_s
                                               0,             // logits_soft_cap
                                               is_v_rowmajor, // is_v_rowmajor
                                               def_lse,       // lse
                                               0,             // page_block_size
                                               false,         // use_cache_batch_idx
                                               "n",           // bias_str
                                               0.0f,          // p_drop
                                               0,             // drop_seed
                                               0,             // drop_offset
                                               false,         // drop_prefs
                                               mask_str,      // mask_str
                                               qscale_str,
                                               true, // is_rotary_interleaved
                                               1,    // num_splits
                                               COMMON_ARGS);
    CHECK_RESULT(result);
}

// Every other suite pins the scale to TestConfigs<T>::qscale_str, leaving the rest untested.
// Only gfx125x has more than one, so the sweep is compiled in there alone.
#ifdef CK_TILE_TEST_FMHA_QSCALE_SWEEP
enum class sink_kind
{
    none,
    gptoss,
    streamllm
};

// Full causal (left=-1) makes y == y_total, so x_start is 0 and the sink phase collapses.
// This row checks that the has_sink instantiation matches the sinkless answer; the live
// sink phase is carried by the local-window tuples below instead.
constexpr auto kStreamLlmMask = "b:-1,0,2";

constexpr bool kIsMxFp8Config =
    ck_tile::is_any_of<DataTypeConfig, FmhaFwdMxFp8, FmhaFwdMxFp8Bf16>::value;
constexpr bool kIsFp8Config = ck_tile::is_any_of<DataTypeConfig, FmhaFwdFp8, FmhaFwdFp8Bf16>::value;

// On gfx125x the FP8 suites below must reach qr_tdm_fp8_sched, not fall back to another pipeline.
bool ExpectsFp8Sched(bool is_v_rowmajor)
{
    // MX has no other gfx125x pipeline; FP8 prefers it at every seqlen_q.
    return ck_tile::is_gfx125_supported() && (kIsFp8Config || (kIsMxFp8Config && is_v_rowmajor));
}

void ExpectFp8Sched(const std::string& kernel_name, bool is_v_rowmajor)
{
    if(ExpectsFp8Sched(is_v_rowmajor))
        EXPECT_NE(kernel_name.find("qr_tdm_fp8_sched"), std::string::npos) << kernel_name;
}

// Unlike CHECK_RESULT, a missing instance fails wherever qr_tdm_fp8_sched is expected.
#define CHECK_FP8_SCHED_RESULT(result, is_v_rowmajor)                           \
    do                                                                          \
    {                                                                           \
        if(result == fwd_result::no_instance && ExpectsFp8Sched(is_v_rowmajor)) \
            FAIL() << "No qr_tdm_fp8_sched instance for current parameters";    \
        CHECK_RESULT(result);                                                   \
    } while(0)

// Skips what qr_tdm_fp8_sched leaves out by design: MX bias and gfx125x MX on col-major V.
#define SKIP_IF_MX_UNSUPPORTED(is_v_rowmajor, bias_str)                          \
    do                                                                           \
    {                                                                            \
        if(kIsMxFp8Config && std::string_view(bias_str) != "n")                  \
            GTEST_SKIP() << "MX has no elementwise bias";                        \
        if(kIsMxFp8Config && !(is_v_rowmajor) && ck_tile::is_gfx125_supported()) \
            GTEST_SKIP() << "gfx125x MX runs on row-major V only";               \
    } while(0)

class QuantScale
    : public TestWithParam<std::tuple<mode_enum,
                                      const char*,
                                      sink_kind,
                                      std::tuple<int, int, int, int, int, std::string>>>
{
};

// hdim 128 is where perhead and blockscale exist. The non-multiple seqlens select the
// seqlen-padded instances; the last tuple is a tile multiple so the unpadded pack-GQA path
// is covered too. This sweep pins bias to "n"; fp8 bias + sink is carried by SinkWindowMask.
INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    QuantScale,
    Combine(ModeValues,
            QScaleValues,
            Values(sink_kind::none, sink_kind::gptoss, sink_kind::streamllm),
            Values(std::tuple{2, 2, 1, 55, 256, "0"},   // GQA, seqlen_q << seqlen_k
                   std::tuple{1, 3, -1, 100, 51, "0"},  // plain MHA, seqlen_q > seqlen_k
                   std::tuple{2, 1, -1, 99, 256, "1"},  // causal
                   std::tuple{1, 2, 1, 1024, 256, "2"}, // GQA, causal bottom-right
                   std::tuple{1, 4, 2, 256, 256, "0"},  // Pack-GQA: ratio 2, no mask, s%128==0
                   // The two local-window tuples below are the only non-empty sink phases
                   // here; causal masks collapse it, and only then does the descale index
                   // leave k_origin. sink=128 == kN0 is the prologue jump.
                   std::tuple{1, 2, 1, 1024, 1024, "t:128,30,128"},
                   // sink=512 gives num_sink_loop 4, reaching the mainloop jump that
                   // sink <= kN0 never does; GQA ratio 2 crosses it with the descale stride.
                   std::tuple{2, 4, 2, 1024, 1024, "t:128,30,512"})));

// init=3 fills Q/K/V up to the fp8 maximum, which only stands for a real tensor when a
// descale maps that maximum back to qkv_max. Without a descale the values stay at the fp8
// scale, the logits reach ~3e5 in the exp2 domain and the softmax collapses to an argmax.
// There half an fp32 ulp of the row max is worth 1% of the exponential, and the fp8 P
// operand snaps that factor out of the numerator while the row sum keeps it, so the whole
// row lands 1% off. Anchor the no-scale inputs on qkv_max, as the three descales do.
const char* qscale_init_method(std::string_view qscale)
{
    return qscale == "n" ? "0" : init_method;
}

TEST_P(QuantScale, DataTypeConfig)
{
    auto [mode, qscale, sink, dims_mask]                       = GetParam();
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str] = dims_mask;

    const std::string mask = sink == sink_kind::streamllm ? kStreamLlmMask : mask_str;
    const int init_sink    = sink == sink_kind::gptoss ? 1 : 0;

    auto result = fmha_fwd_run<DataTypeConfig>(
        mode,
        batch,
        nhead,
        nhead_k,
        {adjust_seqlen(seqlen_q)},
        {adjust_seqlen(seqlen_k)},
        adjust_hdim(128),
        adjust_hdim(128),
        0,    // seqlen_knew
        {-1}, // seqlen_qpads
        {-1}, // seqlen_kpads
        {},   // q_eff_lens_per_batch
        {},   // kv_eff_lens_per_batch
        0,    // rotary_dim
        true, // i_perm
        true, // o_perm
        0,    // scale_s
        0,    // logits_soft_cap
        def_is_v_rowmajor,
        def_lse,
        0,     // page_block_size
        false, // use_cache_batch_idx
        "n",   // bias_str
        0.0f,  // p_drop
        0,     // drop_seed
        0,     // drop_offset
        false, // drop_prefs
        mask,
        qscale,
        true, // is_rotary_interleaved
        1,    // num_splits
        qscale_init_method(qscale),
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1,         // do_validation
        init_sink, // init_sink_value
        1,         // pack_gqa
        stream_config);
    CHECK_RESULT(result);
}

// Elementwise bias across the quantization scales, with the sink logit and sink tokens.
class QuantScaleBias : public QuantScale
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    QuantScaleBias,
    Combine(ModeValues,
            QScaleValues,
            // streamllm swaps in kStreamLlmMask; the t:128,30 rows carry the live sink tokens.
            Values(sink_kind::none, sink_kind::gptoss, sink_kind::streamllm),
            Values(std::tuple{2, 2, 1, 55, 256, "0"},   // GQA, seqlen_q << seqlen_k
                   std::tuple{1, 3, -1, 300, 129, "0"}, // query/key tails
                   std::tuple{2, 1, -1, 256, 256, "1"}, // causal
                   std::tuple{1, 4, 2, 256, 256, "0"},  // GQA ratio 2 (bias turns Pack-GQA off)
                   std::tuple{1, 2, 1, 1024, 1024, "t:128,30,128"},
                   // num_sink_loop 4: the bias window jumps from the sink tokens mid-loop
                   std::tuple{2, 4, 2, 1024, 1024, "t:128,30,512"})));

TEST_P(QuantScaleBias, DataTypeConfig)
{
    auto [mode, qscale, sink, dims_mask]                       = GetParam();
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str] = dims_mask;

    const std::string mask = sink == sink_kind::streamllm ? kStreamLlmMask : mask_str;
    const int init_sink    = sink == sink_kind::gptoss ? 1 : 0;
    SKIP_IF_MX_UNSUPPORTED(def_is_v_rowmajor, "e");

    std::string kernel_name;
    auto result = fmha_fwd_run<DataTypeConfig>(
        mode,
        batch,
        nhead,
        nhead_k,
        {adjust_seqlen(seqlen_q)},
        {adjust_seqlen(seqlen_k)},
        adjust_hdim(128),
        adjust_hdim(128),
        0,    // seqlen_knew
        {-1}, // seqlen_qpads
        {-1}, // seqlen_kpads
        {},   // q_eff_lens_per_batch
        {},   // kv_eff_lens_per_batch
        0,    // rotary_dim
        true, // i_perm
        true, // o_perm
        0,    // scale_s
        0,    // logits_soft_cap
        def_is_v_rowmajor,
        def_lse,
        0,     // page_block_size
        false, // use_cache_batch_idx
        "e",   // bias_str
        0.0f,  // p_drop
        0,     // drop_seed
        0,     // drop_offset
        false, // drop_prefs
        mask,
        qscale,
        true, // is_rotary_interleaved
        1,    // num_splits
        qscale_init_method(qscale),
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1,         // do_validation
        init_sink, // init_sink_value
        1,         // pack_gqa
        stream_config,
        std::nullopt,
        &kernel_name);
    CHECK_FP8_SCHED_RESULT(result, def_is_v_rowmajor);
    ExpectFp8Sched(kernel_name, def_is_v_rowmajor);
}

// LSE across every quantization scale; MXFP8 adds row-major V, the layout gfx125x MX runs on.
class QuantScaleLse
    : public TestWithParam<
          std::tuple<mode_enum, const char*, bool, std::tuple<int, int, int, int, int>>>
{
};

static auto LseVRowmajorValues =
    ValuesIn(std::is_same_v<DataTypeConfig, FmhaFwdMxFp8> ? std::vector<bool>{false, true}
                                                          : std::vector<bool>{def_is_v_rowmajor});

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         QuantScaleLse,
                         Combine(ModeValues,
                                 QScaleValues,
                                 LseVRowmajorValues,
                                 Values(std::tuple{2, 2, 1, 55, 256},       // GQA, short query
                                        std::tuple{1, 3, -1, 100, 51},      // seqlen_q > seqlen_k
                                        std::tuple{1, 4, 2, 256, 256},      // tile multiple
                                        std::tuple{2, 2, -1, 300, 1000}))); // query/key tails

TEST_P(QuantScaleLse, DataTypeConfig)
{
    auto [mode, qscale, is_v_rowmajor, dims]         = GetParam();
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k] = dims;
    SKIP_IF_MX_UNSUPPORTED(is_v_rowmajor, "n");

    std::string kernel_name;
    auto result = fmha_fwd_run<DataTypeConfig>(
        mode,
        batch,
        nhead,
        nhead_k,
        {adjust_seqlen(seqlen_q)},
        {adjust_seqlen(seqlen_k)},
        adjust_hdim(128),
        adjust_hdim(128),
        0,    // seqlen_knew
        {-1}, // seqlen_qpads
        {-1}, // seqlen_kpads
        {},   // q_eff_lens_per_batch
        {},   // kv_eff_lens_per_batch
        0,    // rotary_dim
        true, // i_perm
        true, // o_perm
        0,    // scale_s
        0,    // logits_soft_cap
        is_v_rowmajor,
        true,  // lse
        0,     // page_block_size
        false, // use_cache_batch_idx
        "n",   // bias_str
        0.0f,  // p_drop
        0,     // drop_seed
        0,     // drop_offset
        false, // drop_prefs
        "0",   // mask_str
        qscale,
        true, // is_rotary_interleaved
        1,    // num_splits
        qscale_init_method(qscale),
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1, // do_validation
        0, // init_sink_value
        1, // pack_gqa
        stream_config,
        std::nullopt,
        &kernel_name);
    CHECK_FP8_SCHED_RESULT(result, is_v_rowmajor);
    ExpectFp8Sched(kernel_name, is_v_rowmajor);
}

// Causal and sliding-window masks across every quantization scale, with and without LSE.
class QuantScaleMask
    : public TestWithParam<std::tuple<mode_enum,
                                      const char*,
                                      bool,
                                      bool,
                                      std::tuple<int, int, int, int, int, std::string>>>
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    QuantScaleMask,
    Combine(ModeValues,
            QScaleValues,
            Values(false, true), // lse
            LseVRowmajorValues,
            Values(std::tuple{1, 2, 1, 300, 1000, "t"},       // causal top-left, key tail
                   std::tuple{2, 2, -1, 1000, 300, "b"},      // causal bottom-right, empty rows
                   std::tuple{1, 2, 1, 256, 256, "t:128,30"}, // sliding window, tile multiple
                   std::tuple{1, 3, -1, 500, 700, "b:128,30"},
                   std::tuple{2, 2, 1, 129, 161, "b:-1,64"})));

TEST_P(QuantScaleMask, DataTypeConfig)
{
    auto [mode, qscale, lse, is_v_rowmajor, dims_mask]         = GetParam();
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str] = dims_mask;
    SKIP_IF_MX_UNSUPPORTED(is_v_rowmajor, "n");

    std::string kernel_name;
    auto result = fmha_fwd_run<DataTypeConfig>(
        mode,
        batch,
        nhead,
        nhead_k,
        {adjust_seqlen(seqlen_q)},
        {adjust_seqlen(seqlen_k)},
        adjust_hdim(128),
        adjust_hdim(128),
        0,    // seqlen_knew
        {-1}, // seqlen_qpads
        {-1}, // seqlen_kpads
        {},   // q_eff_lens_per_batch
        {},   // kv_eff_lens_per_batch
        0,    // rotary_dim
        true, // i_perm
        true, // o_perm
        0,    // scale_s
        0,    // logits_soft_cap
        is_v_rowmajor,
        lse,
        0,     // page_block_size
        false, // use_cache_batch_idx
        "n",   // bias_str
        0.0f,  // p_drop
        0,     // drop_seed
        0,     // drop_offset
        false, // drop_prefs
        mask_str,
        qscale,
        true, // is_rotary_interleaved
        1,    // num_splits
        qscale_init_method(qscale),
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1, // do_validation
        0, // init_sink_value
        1, // pack_gqa
        stream_config,
        std::nullopt,
        &kernel_name);
    CHECK_FP8_SCHED_RESULT(result, is_v_rowmajor);
    ExpectFp8Sched(kernel_name, is_v_rowmajor);
}

// Variable-length groups with physical Q/K padding; the runner drops K padding for MX types.
class QuantScaleGroupPad
    : public TestWithParam<
          std::tuple<std::tuple<int, int>, const char*, bool, std::string, std::tuple<int, int>>>
{
};

// Every head dim of the scheduled FP8 pipeline; other data types keep head dim 128.
static const auto kGroupPadHdims =
    kIsMxFp8Config || kIsFp8Config
        ? std::vector<std::tuple<int, int>>{{64, 64}, {128, 128}, {192, 128}, {256, 256}}
        : std::vector<std::tuple<int, int>>{{128, 128}};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         QuantScaleGroupPad,
                         Combine(ValuesIn(kGroupPadHdims),
                                 QScaleValues,
                                 Values(false, true), // lse
                                 Values("0", "t", "b:128,30"),
                                 Values(std::tuple{2, 1}, std::tuple{4, 2})));

TEST_P(QuantScaleGroupPad, DataTypeConfig)
{
    auto [hdims, qscale, lse, mask_str, heads] = GetParam();
    auto [hdim, hdim_v]                        = hdims;
    auto [nhead, nhead_k]                      = heads;

    std::string kernel_name;
    auto result = fmha_fwd_run<DataTypeConfig>(
        mode_enum::group,
        3, // batch
        nhead,
        nhead_k,
        {100, 300, 129},
        {161, 600, 129},
        adjust_hdim(hdim),
        adjust_hdim(hdim_v),
        0,               // seqlen_knew
        {128, 384, 256}, // seqlen_qpads
        {256, 640, 256}, // seqlen_kpads
        {},              // q_eff_lens_per_batch
        {},              // kv_eff_lens_per_batch
        0,               // rotary_dim
        true,            // i_perm
        true,            // o_perm
        0,               // scale_s
        0,               // logits_soft_cap
        true,            // is_v_rowmajor
        lse,
        0,     // page_block_size
        false, // use_cache_batch_idx
        "n",   // bias_str
        0.0f,  // p_drop
        0,     // drop_seed
        0,     // drop_offset
        false, // drop_prefs
        mask_str,
        qscale,
        true, // is_rotary_interleaved
        1,    // num_splits
        qscale_init_method(qscale),
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1, // do_validation
        0, // init_sink_value
        1, // pack_gqa
        stream_config,
        std::nullopt,
        &kernel_name);
    CHECK_FP8_SCHED_RESULT(result, true);
    ExpectFp8Sched(kernel_name, true);
}

// Per-head sink logit (init_sink) and mask sink tokens across every quantization scale and LSE.
class QuantScaleSink
    : public TestWithParam<std::tuple<mode_enum,
                                      const char*,
                                      bool,
                                      bool,
                                      std::tuple<int, int, int, int, int, std::string, int>>>
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    QuantScaleSink,
    Combine(ModeValues,
            QScaleValues,
            Values(false, true), // lse
            LseVRowmajorValues,
            Values(std::tuple{2, 2, 1, 300, 1000, "0", 1},  // sink logit, GQA, key tail
                   std::tuple{1, 3, -1, 1000, 256, "0", 1}, // sink logit, seqlen_q > seqlen_k
                   std::tuple{1, 4, 2, 256, 256, "0", 1},   // sink logit, tile multiple
                   std::tuple{2, 2, -1, 1000, 300, "b", 1}, // sink logit, empty rows
                   // gpt-oss: sliding window plus sink logit, GQA ratio 4
                   std::tuple{2, 4, 1, 1024, 1024, "t:127,0", 1},
                   std::tuple{1, 2, -1, 700, 1300, "b:128,0", 1},
                   // StreamingLLM sink tokens: prologue jump, num_sink_loop 4, key tail
                   std::tuple{1, 2, 1, 1024, 1024, "t:128,30,128", 0},
                   std::tuple{2, 4, 2, 1024, 1024, "t:128,30,512", 1},
                   std::tuple{1, 2, -1, 1000, 1300, "b:128,30,200", 1})));

TEST_P(QuantScaleSink, DataTypeConfig)
{
    auto [mode, qscale, lse, is_v_rowmajor, dims_mask]                    = GetParam();
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str, init_sink] = dims_mask;
    SKIP_IF_MX_UNSUPPORTED(is_v_rowmajor, "n");

    std::string kernel_name;
    auto result = fmha_fwd_run<DataTypeConfig>(
        mode,
        batch,
        nhead,
        nhead_k,
        {adjust_seqlen(seqlen_q)},
        {adjust_seqlen(seqlen_k)},
        adjust_hdim(128),
        adjust_hdim(128),
        0,    // seqlen_knew
        {-1}, // seqlen_qpads
        {-1}, // seqlen_kpads
        {},   // q_eff_lens_per_batch
        {},   // kv_eff_lens_per_batch
        0,    // rotary_dim
        true, // i_perm
        true, // o_perm
        0,    // scale_s
        0,    // logits_soft_cap
        is_v_rowmajor,
        lse,
        0,     // page_block_size
        false, // use_cache_batch_idx
        "n",   // bias_str
        0.0f,  // p_drop
        0,     // drop_seed
        0,     // drop_offset
        false, // drop_prefs
        mask_str,
        qscale,
        true, // is_rotary_interleaved
        1,    // num_splits
        qscale_init_method(qscale),
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1,         // do_validation
        init_sink, // init_sink_value
        1,         // pack_gqa
        stream_config,
        std::nullopt,
        &kernel_name);
    CHECK_FP8_SCHED_RESULT(result, is_v_rowmajor);
    ExpectFp8Sched(kernel_name, is_v_rowmajor);
}

// Head dim 64 across every quantization scale: masks, sink logit and tokens, bias, LSE.
using QuantScaleHdimParam =
    std::tuple<mode_enum,
               const char*,
               bool,
               std::tuple<int, int, int, int, int, std::string, std::string, int>>;

class QuantScaleHdim64 : public TestWithParam<QuantScaleHdimParam>
{
};

void RunQuantScaleHdim(int hdim, const QuantScaleHdimParam& param, int hdim_v = -1)
{
    auto [mode, qscale, lse, dims]                                                  = param;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str, bias_str, init_sink] = dims;
    // gfx125x MXFP8 runs on row-major V.
    const bool is_v_rowmajor = kIsMxFp8Config || def_is_v_rowmajor;
    SKIP_IF_MX_UNSUPPORTED(is_v_rowmajor, bias_str);

    std::string kernel_name;
    auto result = fmha_fwd_run<DataTypeConfig>(
        mode,
        batch,
        nhead,
        nhead_k,
        {adjust_seqlen(seqlen_q)},
        {adjust_seqlen(seqlen_k)},
        adjust_hdim(hdim),
        adjust_hdim(hdim_v < 0 ? hdim : hdim_v),
        0,    // seqlen_knew
        {-1}, // seqlen_qpads
        {-1}, // seqlen_kpads
        {},   // q_eff_lens_per_batch
        {},   // kv_eff_lens_per_batch
        0,    // rotary_dim
        true, // i_perm
        true, // o_perm
        0,    // scale_s
        0,    // logits_soft_cap
        is_v_rowmajor,
        lse,
        0,     // page_block_size
        false, // use_cache_batch_idx
        bias_str,
        0.0f,  // p_drop
        0,     // drop_seed
        0,     // drop_offset
        false, // drop_prefs
        mask_str,
        qscale,
        true, // is_rotary_interleaved
        1,    // num_splits
        qscale_init_method(qscale),
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1,         // do_validation
        init_sink, // init_sink_value
        1,         // pack_gqa
        stream_config,
        std::nullopt,
        &kernel_name);
    CHECK_FP8_SCHED_RESULT(result, is_v_rowmajor);
    ExpectFp8Sched(kernel_name, is_v_rowmajor);
}

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    QuantScaleHdim64,
    Combine(ModeValues,
            QScaleValues,
            Values(false, true),                                            // lse
            Values(std::tuple{2, 2, 1, 55, 256, "0", "n", 0},               // GQA, short query
                   std::tuple{1, 3, -1, 300, 129, "0", "n", 0},             // query/key tails
                   std::tuple{2, 1, -1, 256, 256, "1", "n", 0},             // causal
                   std::tuple{1, 2, 1, 256, 700, "t:128,30", "n", 0},       // sliding window
                   std::tuple{2, 4, 1, 1024, 1024, "t:127,0", "n", 1},      // gpt-oss
                   std::tuple{2, 4, 2, 1024, 1024, "t:128,30,512", "n", 1}, // sink tokens
                   std::tuple{1, 2, -1, 1000, 300, "b", "n", 1}, // sink logit, empty rows
                   std::tuple{2, 2, 1, 300, 1000, "0", "e", 0},  // elementwise bias
                   std::tuple{1, 2, -1, 256, 700, "t", "e", 1},  // bias, mask and sink logit
                   std::tuple{1, 2, 1, 300, 1000, "0", "n", 1},  // sink logit without a mask
                   std::tuple{1, 3, -1, 300, 129, "0", "e", 1},  // bias and sink logit, no mask
                   std::tuple{1, 3, -1, 500, 700, "b:128,30", "e", 0}, // bias, window past key 0
                   std::tuple{1, 2, 1, 256, 1000, "b:128,30,200", "e", 1}, // bias, sink tokens
                   std::tuple{1, 2, -1, 300, 512, "0", "n", 0})));         // MHA, key tile multiple

TEST_P(QuantScaleHdim64, DataTypeConfig) { RunQuantScaleHdim(64, GetParam()); }

// Head dim 256 (one workgroup per CU on gfx125x) with the same rows as head dim 64.
class QuantScaleHdim256 : public TestWithParam<QuantScaleHdimParam>
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    QuantScaleHdim256,
    Combine(ModeValues,
            QScaleValues,
            Values(false, true),                                            // lse
            Values(std::tuple{2, 2, 1, 55, 256, "0", "n", 0},               // GQA, short query
                   std::tuple{1, 3, -1, 300, 129, "0", "n", 0},             // query/key tails
                   std::tuple{2, 1, -1, 256, 256, "1", "n", 0},             // causal
                   std::tuple{1, 2, 1, 256, 700, "t:128,30", "n", 0},       // sliding window
                   std::tuple{2, 4, 1, 1024, 1024, "t:127,0", "n", 1},      // gpt-oss
                   std::tuple{2, 4, 2, 1024, 1024, "t:128,30,512", "n", 1}, // sink tokens
                   std::tuple{1, 2, -1, 1000, 300, "b", "n", 1}, // sink logit, empty rows
                   std::tuple{2, 2, 1, 300, 1000, "0", "e", 0},  // elementwise bias
                   std::tuple{1, 2, -1, 256, 700, "t", "e", 1},  // bias, mask and sink logit
                   std::tuple{1, 2, 1, 300, 1000, "0", "n", 1},  // sink logit without a mask
                   std::tuple{1, 3, -1, 300, 129, "0", "e", 1},  // bias and sink logit, no mask
                   std::tuple{1, 3, -1, 500, 700, "b:128,30", "e", 0}, // bias, window past key 0
                   std::tuple{1, 2, 1, 256, 1000, "b:128,30,200", "e", 1}, // bias, sink tokens
                   std::tuple{1, 2, -1, 300, 512, "0", "n", 0})));         // MHA, key tile multiple

TEST_P(QuantScaleHdim256, DataTypeConfig) { RunQuantScaleHdim(256, GetParam()); }

// Decode-length queries in batch mode on every head dim, GQA and MHA.
class QuantScaleShortQuery
    : public TestWithParam<
          std::tuple<int,
                     const char*,
                     bool,
                     std::tuple<int, int, int, int, int, std::string, std::string, int>>>
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    QuantScaleShortQuery,
    Combine(Values(64, 128, 256),
            QScaleValues,
            Values(false, true),                                           // lse
            Values(std::tuple{2, 8, 2, 1, 1024, "0", "n", 0},              // GQA 4, Pack-GQA n/pt
                   std::tuple{3, 4, -1, 1, 500, "0", "n", 0},              // MHA, key tail
                   std::tuple{2, 8, 2, 4, 700, "b", "n", 0},               // causal bottom-right
                   std::tuple{1, 4, 1, 16, 2048, "b", "n", 0},             // causal, GQA ratio 4
                   std::tuple{2, 8, 2, 16, 1000, "b:127,0", "n", 1},       // gpt-oss
                   std::tuple{1, 8, 2, 4, 2048, "b:1023,0,4", "n", 0},     // sink tokens
                   std::tuple{2, 4, 1, 1, 300, "0", "n", 1},               // sink logit
                   std::tuple{1, 2, -1, 55, 256, "b", "e", 0},             // elementwise bias
                   std::tuple{2, 4, 2, 127, 300, "b:1023,0,4", "n", 1}))); // 127 = tile - 1

TEST_P(QuantScaleShortQuery, DataTypeConfig)
{
    auto [hdim, qscale, lse, dims] = GetParam();
    RunQuantScaleHdim(hdim, {mode_enum::batch, qscale, lse, dims});
}

// Decode without Pack-GQA in both layouts; asserts which decode kernel each shape reaches.
using MergedDecodeDims = std::tuple<int, int, int, int, int, std::string, int, bool>;

class QuantScaleMergedDecode
    : public TestWithParam<std::tuple<int, const char*, bool, MergedDecodeDims>>
{
};

// Packed GQA on the 32-row tile up to 32 rows, on the 128-row tile up to 128 with GQA, else plain.
std::string ExpectedDecodeKernel(int nhead, int nhead_k, int seqlen_q, bool perm)
{
    const int ratio = nhead_k < 0 ? 1 : nhead / nhead_k;
    const int rows  = seqlen_q * ratio;
    // b*s*h*d query heads only fold into rows for a single query.
    if(seqlen_q > 1 && !perm)
        return "sched";
    if(rows <= 32)
        return "packgqa_32row";
    if(rows <= 128 && ratio > 1)
        return "packgqa";
    return "sched";
}

std::string DecodeKernelOf(const std::string& kernel_name)
{
    if(kernel_name.find("_qr_tdm_fp8_sched_") == std::string::npos)
        return "other";
    if(kernel_name.find("_packgqa_") == std::string::npos)
        return "sched";
    return kernel_name.find("_b32x") != std::string::npos ? "packgqa_32row" : "packgqa";
}

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    QuantScaleMergedDecode,
    Combine(Values(64, 128, 256),
            QScaleValues,
            Values(false, true),                                            // lse
            Values(MergedDecodeDims{2, 8, 2, 1, 1024, "0", 0, true},        // GQA 4, no mask
                   MergedDecodeDims{2, 8, 2, 1, 1024, "0", 0, false},       // same, b*s*h*d
                   MergedDecodeDims{1, 16, 2, 8, 700, "b", 0, true},        // 64 rows, causal
                   MergedDecodeDims{1, 16, 2, 8, 700, "b", 0, false},       // b*s*h*d, s > 1
                   MergedDecodeDims{2, 32, 2, 1, 513, "b:127,0", 1, false}, // gpt-oss, b*s*h*d
                   MergedDecodeDims{1, 32, 4, 8, 300, "0", 1, true},        // sink logit, 64 rows
                   MergedDecodeDims{2, 8, -1, 2, 300, "b", 0, true},        // MHA, 2 rows
                   MergedDecodeDims{1, 8, 1, 20, 300, "b", 0, true},        // 160 rows
                   MergedDecodeDims{2, 8, 2, 1, 0, "0", 1, true},  // empty KV, sink logit, 4 rows
                   MergedDecodeDims{2, 8, 2, 1, 0, "0", 0, true},  // empty KV, 4 rows
                   MergedDecodeDims{1, 16, 2, 8, 0, "0", 1, true}, // empty KV, sink logit, 64 rows
                   MergedDecodeDims{1, 16, 2, 8, 0, "0", 0, true}, // empty KV, 64 rows
                   MergedDecodeDims{1, 16, 2, 8, 5, "b", 1, true}, // fully masked rows, 64 rows
                   MergedDecodeDims{2, 8, 2, 4, 2, "b", 0, true},  // fully masked rows, 16 rows
                   MergedDecodeDims{1, 16, 2, 8, 8, "1", 0, true}, // square causal, 64 rows
                   MergedDecodeDims{2, 8, 2, 4, 4, "1", 0, true},  // square causal, 16 rows
                   MergedDecodeDims{1, 32, 2, 8, 700, "0", 0, true}, // full 128-row tile, key tail
                   MergedDecodeDims{1, 32, 4, 8, 2048, "b:128,30,512", 0, true}))); // sink tokens

void RunMergedDecode(
    int hdim, int hdim_v, const char* qscale, bool lse, const MergedDecodeDims& dims)
{
    if(!(kIsMxFp8Config || kIsFp8Config) || !ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "decode kernels are gfx125x FP8 only";
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str, init_sink, perm] = dims;
    const bool is_v_rowmajor = kIsMxFp8Config || def_is_v_rowmajor;

    std::string kernel_name;
    auto result = fmha_fwd_run<DataTypeConfig>(
        mode_enum::batch,
        batch,
        nhead,
        nhead_k,
        {adjust_seqlen(seqlen_q)},
        {adjust_seqlen(seqlen_k)},
        adjust_hdim(hdim),
        adjust_hdim(hdim_v),
        0,    // seqlen_knew
        {-1}, // seqlen_qpads
        {-1}, // seqlen_kpads
        {},   // q_eff_lens_per_batch
        {},   // kv_eff_lens_per_batch
        0,    // rotary_dim
        perm, // i_perm
        perm, // o_perm
        0,    // scale_s
        0,    // logits_soft_cap
        is_v_rowmajor,
        lse,
        0,     // page_block_size
        false, // use_cache_batch_idx
        "n",   // bias
        0.0f,  // p_drop
        0,     // drop_seed
        0,     // drop_offset
        false, // drop_prefs
        mask_str,
        qscale,
        true, // is_rotary_interleaved
        1,    // num_splits
        qscale_init_method(qscale),
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1,         // do_validation
        init_sink, // init_sink_value
        0,         // pack_gqa
        stream_config,
        std::nullopt,
        &kernel_name);
    CHECK_FP8_SCHED_RESULT(result, is_v_rowmajor);
    EXPECT_EQ(DecodeKernelOf(kernel_name), ExpectedDecodeKernel(nhead, nhead_k, seqlen_q, perm))
        << kernel_name;
}

TEST_P(QuantScaleMergedDecode, DataTypeConfig)
{
    auto [hdim, qscale, lse, dims] = GetParam();
    RunMergedDecode(hdim, hdim, qscale, lse, dims);
}

// Head dim 192 with a 128-wide V (D128 PV, K64 QK) on the rows of head dims 64 and 256.
class QuantScaleHdim192 : public TestWithParam<QuantScaleHdimParam>
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    QuantScaleHdim192,
    Combine(ModeValues,
            QScaleValues,
            Values(false, true),                                            // lse
            Values(std::tuple{2, 2, 1, 55, 256, "0", "n", 0},               // GQA, short query
                   std::tuple{1, 3, -1, 300, 129, "0", "n", 0},             // query/key tails
                   std::tuple{2, 1, -1, 256, 256, "1", "n", 0},             // causal
                   std::tuple{1, 2, 1, 256, 700, "t:128,30", "n", 0},       // sliding window
                   std::tuple{2, 4, 1, 1024, 1024, "t:127,0", "n", 1},      // gpt-oss
                   std::tuple{2, 4, 2, 1024, 1024, "t:128,30,512", "n", 1}, // sink tokens
                   std::tuple{1, 2, -1, 1000, 300, "b", "n", 1}, // sink logit, empty rows
                   std::tuple{2, 2, 1, 300, 1000, "0", "e", 0},  // elementwise bias
                   std::tuple{1, 2, -1, 256, 700, "t", "e", 1},  // bias, mask and sink logit
                   std::tuple{1, 2, 1, 300, 1000, "0", "n", 1},  // sink logit without a mask
                   std::tuple{1, 3, -1, 300, 129, "0", "e", 1},  // bias and sink logit, no mask
                   std::tuple{1, 3, -1, 500, 700, "b:128,30", "e", 0}, // bias, window past key 0
                   std::tuple{1, 2, 1, 256, 1000, "b:128,30,200", "e", 1}, // bias, sink tokens
                   std::tuple{1, 2, -1, 300, 512, "0", "n", 0})));         // MHA, key tile multiple

TEST_P(QuantScaleHdim192, DataTypeConfig) { RunQuantScaleHdim(192, GetParam(), 128); }

// Decode-length queries on head dim 192 / V 128, as QuantScaleShortQuery.
class QuantScaleShortQueryHdim192 : public TestWithParam<QuantScaleHdimParam>
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    QuantScaleShortQueryHdim192,
    Combine(Values(mode_enum::batch),
            QScaleValues,
            Values(false, true),                                           // lse
            Values(std::tuple{2, 8, 2, 1, 1024, "0", "n", 0},              // GQA 4, Pack-GQA n/pt
                   std::tuple{3, 4, -1, 1, 500, "0", "n", 0},              // MHA, key tail
                   std::tuple{2, 8, 2, 4, 700, "b", "n", 0},               // causal bottom-right
                   std::tuple{1, 4, 1, 16, 2048, "b", "n", 0},             // causal, GQA ratio 4
                   std::tuple{2, 8, 2, 16, 1000, "b:127,0", "n", 1},       // gpt-oss
                   std::tuple{1, 8, 2, 4, 2048, "b:1023,0,4", "n", 0},     // sink tokens
                   std::tuple{2, 4, 1, 1, 300, "0", "n", 1},               // sink logit
                   std::tuple{1, 2, -1, 55, 256, "b", "e", 0},             // elementwise bias
                   std::tuple{2, 4, 2, 127, 300, "b:1023,0,4", "n", 1}))); // 127 = tile - 1

TEST_P(QuantScaleShortQueryHdim192, DataTypeConfig) { RunQuantScaleHdim(192, GetParam(), 128); }

// Decode kernel selection on head dim 192 / V 128, as QuantScaleMergedDecode.
class QuantScaleMergedDecodeHdim192
    : public TestWithParam<std::tuple<const char*, bool, MergedDecodeDims>>
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    QuantScaleMergedDecodeHdim192,
    Combine(QScaleValues,
            Values(false, true),                                            // lse
            Values(MergedDecodeDims{2, 8, 2, 1, 1024, "0", 0, true},        // GQA 4, no mask
                   MergedDecodeDims{2, 8, 2, 1, 1024, "0", 0, false},       // same, b*s*h*d
                   MergedDecodeDims{1, 16, 2, 8, 700, "b", 0, true},        // 64 rows, causal
                   MergedDecodeDims{1, 16, 2, 8, 700, "b", 0, false},       // b*s*h*d, s > 1
                   MergedDecodeDims{2, 32, 2, 1, 513, "b:127,0", 1, false}, // gpt-oss, b*s*h*d
                   MergedDecodeDims{1, 32, 4, 8, 300, "0", 1, true},        // sink logit, 64 rows
                   MergedDecodeDims{2, 8, -1, 2, 300, "b", 0, true},        // MHA, 2 rows
                   MergedDecodeDims{1, 8, 1, 20, 300, "b", 0, true},        // 160 rows
                   MergedDecodeDims{2, 64, 8, 4, 1000, "0", 0, true},       // GQA 8, 32 rows
                   MergedDecodeDims{1, 32, 4, 8, 700, "b:127,0", 1, true},  // masked sink, 64 rows
                   MergedDecodeDims{1, 32, 2, 1, 2048, "b:1023,0,4", 0, true}, // 16 groups
                   MergedDecodeDims{2, 8, 2, 1, 0, "0", 1, true},  // empty KV, sink logit, 4 rows
                   MergedDecodeDims{2, 8, 2, 1, 0, "0", 0, true},  // empty KV, 4 rows
                   MergedDecodeDims{1, 16, 2, 8, 0, "0", 1, true}, // empty KV, sink logit, 64 rows
                   MergedDecodeDims{1, 16, 2, 8, 0, "0", 0, true}, // empty KV, 64 rows
                   MergedDecodeDims{1, 16, 2, 8, 5, "b", 1, true}, // fully masked rows, 64 rows
                   MergedDecodeDims{2, 8, 2, 4, 2, "b", 0, true},  // fully masked rows, 16 rows
                   MergedDecodeDims{1, 16, 2, 8, 8, "1", 0, true}, // square causal, 64 rows
                   MergedDecodeDims{2, 8, 2, 4, 4, "1", 0, true},  // square causal, 16 rows
                   MergedDecodeDims{1, 32, 2, 8, 700, "0", 0, true}, // full 128-row tile, key tail
                   MergedDecodeDims{1, 32, 4, 8, 2048, "b:128,30,512", 0, true}))); // sink tokens

TEST_P(QuantScaleMergedDecodeHdim192, DataTypeConfig)
{
    auto [qscale, lse, dims] = GetParam();
    RunMergedDecode(192, 128, qscale, lse, dims);
}

// Elementwise bias with LSE, including the early exit of empty rows next to a sink logit.
class QuantScaleBiasLse : public QuantScale
{
};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         QuantScaleBiasLse,
                         Combine(ModeValues,
                                 QScaleValues,
                                 Values(sink_kind::none, sink_kind::gptoss),
                                 Values(std::tuple{1, 3, -1, 300, 129, "0"},  // tails
                                        std::tuple{2, 2, -1, 1000, 300, "b"}, // early exit
                                        std::tuple{2, 4, 2, 1024, 1024, "t:128,30,512"})));

TEST_P(QuantScaleBiasLse, DataTypeConfig)
{
    auto [mode, qscale, sink, dims_mask]                       = GetParam();
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str] = dims_mask;
    SKIP_IF_MX_UNSUPPORTED(def_is_v_rowmajor, "e");

    std::string kernel_name;
    auto result = fmha_fwd_run<DataTypeConfig>(
        mode,
        batch,
        nhead,
        nhead_k,
        {adjust_seqlen(seqlen_q)},
        {adjust_seqlen(seqlen_k)},
        adjust_hdim(128),
        adjust_hdim(128),
        0,    // seqlen_knew
        {-1}, // seqlen_qpads
        {-1}, // seqlen_kpads
        {},   // q_eff_lens_per_batch
        {},   // kv_eff_lens_per_batch
        0,    // rotary_dim
        true, // i_perm
        true, // o_perm
        0,    // scale_s
        0,    // logits_soft_cap
        def_is_v_rowmajor,
        true,  // lse
        0,     // page_block_size
        false, // use_cache_batch_idx
        "e",   // bias_str
        0.0f,  // p_drop
        0,     // drop_seed
        0,     // drop_offset
        false, // drop_prefs
        mask_str,
        qscale,
        true, // is_rotary_interleaved
        1,    // num_splits
        qscale_init_method(qscale),
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1,                                 // do_validation
        sink == sink_kind::gptoss ? 1 : 0, // init_sink_value
        1,                                 // pack_gqa
        stream_config,
        std::nullopt,
        &kernel_name);
    CHECK_FP8_SCHED_RESULT(result, def_is_v_rowmajor);
    ExpectFp8Sched(kernel_name, def_is_v_rowmajor);
}

// MX batch instances without LSE, mask or sink; every other MX sweep adds one of them.
class QuantScaleMxBase
    : public TestWithParam<std::tuple<bool, bool, std::tuple<int, int, int, int, int>>>
{
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(QuantScaleMxBase);
INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    QuantScaleMxBase,
    Combine(EnableTestIf(ck_tile::is_any_of<DataTypeConfig, FmhaFwdMxFp8, FmhaFwdMxFp8Bf16>::value),
            LseVRowmajorValues,
            Values(std::tuple{1, 2, -1, 128, 128},   // one tile
                   std::tuple{2, 4, 2, 256, 256},    // GQA, tile multiple
                   std::tuple{2, 3, 1, 300, 1000},   // query/key tails
                   std::tuple{1, 2, -1, 1000, 161},  // seqlen_q > seqlen_k
                   std::tuple{2, 2, -1, 17, 333}))); // short query

TEST_P(QuantScaleMxBase, DataTypeConfig)
{
    auto [_, is_v_rowmajor, dims]                    = GetParam();
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k] = dims;
    SKIP_IF_MX_UNSUPPORTED(is_v_rowmajor, "n");

    std::string kernel_name;
    auto result = fmha_fwd_run<DataTypeConfig>(
        mode_enum::batch,
        batch,
        nhead,
        nhead_k,
        {adjust_seqlen(seqlen_q)},
        {adjust_seqlen(seqlen_k)},
        adjust_hdim(128),
        adjust_hdim(128),
        0,    // seqlen_knew
        {-1}, // seqlen_qpads
        {-1}, // seqlen_kpads
        {},   // q_eff_lens_per_batch
        {},   // kv_eff_lens_per_batch
        0,    // rotary_dim
        true, // i_perm
        true, // o_perm
        0,    // scale_s
        0,    // logits_soft_cap
        is_v_rowmajor,
        false, // lse
        0,     // page_block_size
        false, // use_cache_batch_idx
        "n",   // bias_str
        0.0f,  // p_drop
        0,     // drop_seed
        0,     // drop_offset
        false, // drop_prefs
        "0",   // mask_str
        "mx",
        true, // is_rotary_interleaved
        1,    // num_splits
        "3",  // init_method
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1, // do_validation
        0, // init_sink_value
        1, // pack_gqa
        stream_config,
        std::nullopt,
        &kernel_name);
    CHECK_FP8_SCHED_RESULT(result, is_v_rowmajor);
    ExpectFp8Sched(kernel_name, is_v_rowmajor);
}

// Batch mode with per-batch query/key lengths: workgroups past a batch's seqlen_q exit early.
using EarlyExitDims =
    std::tuple<int, int, int, int, int, std::vector<int>, std::vector<int>, std::string, bool>;

class QuantScaleBatchEarlyExit
    : public TestWithParam<std::tuple<std::tuple<int, int>, bool, EarlyExitDims>>
{
};

// Every head dim of the scheduled FP8 pipeline, as (hdim_q, hdim_v).
static const auto kEarlyExitHdims =
    kIsMxFp8Config || kIsFp8Config
        ? std::vector<std::tuple<int, int>>{{64, 64}, {128, 128}, {192, 128}, {256, 256}}
        : std::vector<std::tuple<int, int>>{};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(QuantScaleBatchEarlyExit);
INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    QuantScaleBatchEarlyExit,
    Combine(ValuesIn(kEarlyExitHdims),
            Values(false, true), // lse
            // batch, nhead, nhead_k, seqlen_q, seqlen_k, q_eff_lens, kv_eff_lens, mask, bhsd
            Values(EarlyExitDims{3, 4, -1, 300, 400, {300, 64, 1}, {400, 100, 7}, "0", true},
                   EarlyExitDims{3, 8, 2, 300, 300, {129, 0, 257}, {300, 50, 300}, "b", false},
                   EarlyExitDims{2, 4, 1, 600, 512, {513, 17}, {}, "t:128,30", true},
                   // Merge-eligible decode shapes stay unmerged once cu_seqlen_q or _k is set.
                   EarlyExitDims{2, 8, 2, 1, 1024, {1, 1}, {}, "0", true},
                   EarlyExitDims{2, 16, 2, 8, 700, {}, {700, 9}, "b", true})));

TEST_P(QuantScaleBatchEarlyExit, DataTypeConfig)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "qr_tdm_fp8_sched is gfx125x only";
    auto [hdims, lse, dims]                                                     = GetParam();
    auto [hdim, hdim_v]                                                         = hdims;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, q_eff, kv_eff, mask, perm] = dims;
    const char* qscale = kIsMxFp8Config ? "mx" : "bs";

    std::string kernel_name;
    auto result = fmha_fwd_run<DataTypeConfig>(
        mode_enum::batch,
        batch,
        nhead,
        nhead_k,
        {seqlen_q},
        {seqlen_k},
        hdim,
        hdim_v,
        0,    // seqlen_knew
        {-1}, // seqlen_qpads
        {-1}, // seqlen_kpads
        q_eff,
        kv_eff,
        0,    // rotary_dim
        perm, // i_perm
        perm, // o_perm
        0,    // scale_s
        0,    // logits_soft_cap
        true, // is_v_rowmajor
        lse,
        0,     // page_block_size
        false, // use_cache_batch_idx
        "n",   // bias_str
        0.0f,  // p_drop
        0,     // drop_seed
        0,     // drop_offset
        false, // drop_prefs
        mask,
        qscale,
        true, // is_rotary_interleaved
        1,    // num_splits
        "3",  // init_method
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1, // do_validation
        0, // init_sink_value
        1, // pack_gqa
        stream_config,
        std::nullopt,
        &kernel_name);
    CHECK_FP8_SCHED_RESULT(result, true);
    EXPECT_EQ(DecodeKernelOf(kernel_name), "sched") << kernel_name;
}

// Direct fmha_fwd() calls with arguments the runner never produces.
constexpr const char* DirectDataType()
{
    if constexpr(std::is_same_v<DataTypeConfig, FmhaFwdFp8>)
        return "fp8";
    else if constexpr(std::is_same_v<DataTypeConfig, FmhaFwdMxFp8>)
        return "mxfp8";
    else if constexpr(std::is_same_v<DataTypeConfig, FmhaFwdMxFp8Bf16>)
        return "mxfp8bf16";
    else
        return "fp8bf16";
}

struct DirectCase
{
    int batch               = 2;
    int nhead               = 2;
    int nhead_k             = 1;
    int seqlen_q            = 200;
    int seqlen_k            = 700;
    int hdim_q              = 128;
    int hdim_v              = 128;
    std::string mask        = "0";
    quant_scale_enum qscale = quant_scale_enum::blockscale;
    int bss_q               = 128;
    int bss_kv              = 32; // keys per K/V descale array element
    int bss_kv_value        = 32; // keys per distinct K/V descale value
    int max_seqlen_q        = -1;
    int lse_head_pad        = 0;
    std::vector<float> sink; // one logit per query head; empty for none
};

struct DirectResult
{
    float rc = -1;
    std::string kernel;
    std::vector<float> o;
    std::vector<float> lse;
};

// Batch mode, b*h*s*d layout, row-major V, LSE stored; inputs depend only on the shape.
DirectResult RunDirect(const DirectCase& c)
{
    using QDataType = typename FmhaFwdTypeConfig<DataTypeConfig>::QDataType;
    using ODataType = typename FmhaFwdTypeConfig<DataTypeConfig>::ODataType;
    const int b = c.batch, h = c.nhead, hk = c.nhead_k, sq = c.seqlen_q, sk = c.seqlen_k;
    const int dq = c.hdim_q, dv = c.hdim_v, lse_stride = sq + c.lse_head_pad;
    const bool bs   = c.qscale == quant_scale_enum::blockscale;
    const int nbq   = bs ? ck_tile::integer_divide_ceil(sq, c.bss_q > 0 ? c.bss_q : 128) : 1;
    const int nbk   = bs ? ck_tile::integer_divide_ceil(sk, c.bss_kv > 0 ? c.bss_kv : 32) : 1;
    const int nbval = bs ? ck_tile::integer_divide_ceil(sk, c.bss_kv_value) : 1;

    std::mt19937 gen(123);
    std::uniform_real_distribution<float> value(-2.f, 2.f), scale(0.5f, 1.5f);
    const auto fill = [&](int n) {
        std::vector<QDataType> t(n);
        for(auto& x : t)
            x = ck_tile::type_convert<QDataType>(value(gen));
        return t;
    };
    const auto q = fill(b * h * sq * dq), k = fill(b * hk * sk * dq), v = fill(b * hk * sk * dv);
    std::vector<float> q_ds(b * h * nbq), k_val(b * hk * nbval), v_val(b * hk * nbval);
    for(auto& x : q_ds)
        x = scale(gen);
    for(auto& x : k_val)
        x = scale(gen);
    // The V descale must be a power of two.
    for(auto& x : v_val)
        x = std::ldexp(1.f, static_cast<int>(gen() % 3) - 1);
    std::vector<float> k_ds(b * hk * nbk), v_ds(b * hk * nbk);
    for(int bh = 0; bh < b * hk; ++bh)
        for(int i = 0; i < nbk; ++i)
        {
            const int j        = bs ? i * c.bss_kv / c.bss_kv_value : 0;
            k_ds[bh * nbk + i] = k_val[bh * nbval + j];
            v_ds[bh * nbk + i] = v_val[bh * nbval + j];
        }

    const auto upload = [](const auto& host) {
        auto dev = std::make_unique<ck_tile::DeviceMem>(
            std::max<std::size_t>(1, host.size() * sizeof(host[0])));
        dev->ToDevice(host.data());
        return dev;
    };
    const auto q_buf = upload(q), k_buf = upload(k), v_buf = upload(v);
    const auto qd_buf = upload(q_ds), kd_buf = upload(k_ds), vd_buf = upload(v_ds);
    const auto sink_buf = upload(c.sink.empty() ? std::vector<float>{0.f} : c.sink);
    ck_tile::DeviceMem o_buf(b * h * sq * dv * sizeof(ODataType));
    ck_tile::DeviceMem lse_buf(b * h * lse_stride * sizeof(float));
    o_buf.SetZero();
    lse_buf.SetZero();

    const mask_info mask = mask_info::decode(c.mask, sq, sk);
    DirectResult r;
    fmha_fwd_args a{};
    a.selected_kernel_name   = &r.kernel;
    a.q_ptr                  = q_buf->GetDeviceBuffer();
    a.k_ptr                  = k_buf->GetDeviceBuffer();
    a.v_ptr                  = v_buf->GetDeviceBuffer();
    a.q_descale_ptr          = qd_buf->GetDeviceBuffer();
    a.k_descale_ptr          = kd_buf->GetDeviceBuffer();
    a.v_descale_ptr          = vd_buf->GetDeviceBuffer();
    a.o_ptr                  = o_buf.GetDeviceBuffer();
    a.lse_ptr                = lse_buf.GetDeviceBuffer();
    a.sink_ptr               = c.sink.empty() ? nullptr : sink_buf->GetDeviceBuffer();
    a.seqlen_q               = sq;
    a.seqlen_k               = sk;
    a.batch                  = b;
    a.max_seqlen_q           = c.max_seqlen_q < 0 ? sq : c.max_seqlen_q;
    a.hdim_q                 = dq;
    a.hdim_v                 = dv;
    a.nhead_q                = h;
    a.nhead_k                = hk;
    a.num_head_q_total       = h;
    a.scale_s                = 1.f / std::sqrt(static_cast<float>(dq));
    a.stride_q               = dq;
    a.stride_k               = dq;
    a.stride_v               = dv;
    a.stride_o               = dv;
    a.nhead_stride_q         = sq * dq;
    a.nhead_stride_k         = sk * dq;
    a.nhead_stride_v         = sk * dv;
    a.nhead_stride_o         = sq * dv;
    a.nhead_stride_lse       = lse_stride;
    a.nhead_stride_q_descale = nbq;
    a.nhead_stride_k_descale = nbk;
    a.nhead_stride_v_descale = nbk;
    a.batch_stride_q         = h * sq * dq;
    a.batch_stride_k         = hk * sk * dq;
    a.batch_stride_v         = hk * sk * dv;
    a.batch_stride_o         = h * sq * dv;
    a.batch_stride_lse       = h * lse_stride;
    a.batch_stride_q_descale = h * nbq;
    a.batch_stride_k_descale = hk * nbk;
    a.batch_stride_v_descale = hk * nbk;
    a.window_size_left       = mask.left;
    a.window_size_right      = mask.right;
    a.sink_size              = mask.sink;
    a.mask_type              = static_cast<ck_tile::index_t>(mask.type);
    a.block_scale_size_q     = c.bss_q;
    a.block_scale_size_kv    = c.bss_kv;

    const fmha_fwd_traits traits{dq,
                                 dv,
                                 DirectDataType(),
                                 false, // batch mode
                                 true,  // row-major V
                                 false, // no logits soft cap
                                 mask.type,
                                 bias_enum::no_bias,
                                 true,  // LSE
                                 false, // no dropout
                                 c.qscale,
                                 false, // no skip_min_seqlen_q
                                 mask.sink > 0 || !c.sink.empty()};
    r.rc = fmha_fwd(traits, a, ck_tile::stream_config{nullptr, false});
    if(r.rc < 0)
        return r;
    HIP_CHECK_ERROR(hipDeviceSynchronize());

    std::vector<ODataType> o(b * h * sq * dv);
    o_buf.FromDevice(o.data());
    r.o.resize(o.size());
    std::transform(o.begin(), o.end(), r.o.begin(), [](ODataType x) {
        return ck_tile::type_convert<float>(x);
    });
    r.lse.resize(b * h * lse_stride);
    lse_buf.FromDevice(r.lse.data());
    return r;
}

bool SameBits(const std::vector<float>& x, const std::vector<float>& y)
{
    return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) == 0;
}

static const auto kDirectHdims =
    kIsFp8Config ? kEarlyExitHdims : std::vector<std::tuple<int, int>>{};

// K/V descale blocks of 64-256 keys must match the same values spelled out per 32 keys.
class QuantScaleBlockScaleKv
    : public TestWithParam<std::tuple<std::tuple<int, int>, int, std::string>>
{
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(QuantScaleBlockScaleKv);
INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         QuantScaleBlockScaleKv,
                         Combine(ValuesIn(kDirectHdims),
                                 Values(64, 128, 256), // block_scale_size_kv
                                 // window starts mid-block for 256; sink tokens jump to it
                                 Values("0", "b:128,30", "b:128,30,200")));

TEST_P(QuantScaleBlockScaleKv, DataTypeConfig)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "qr_tdm_fp8_sched is gfx125x only";
    auto [hdims, bss_kv, mask] = GetParam();
    DirectCase c;
    std::tie(c.hdim_q, c.hdim_v) = hdims;
    c.mask                       = mask;
    c.bss_kv = c.bss_kv_value = bss_kv;
    const auto wide           = RunDirect(c);
    c.bss_kv                  = 32;
    const auto narrow         = RunDirect(c);
    ASSERT_GE(wide.rc, 0) << wide.kernel;
    ASSERT_GE(narrow.rc, 0) << narrow.kernel;
    ExpectFp8Sched(wide.kernel, true);
    EXPECT_EQ(wide.kernel, narrow.kernel);
    EXPECT_TRUE(std::any_of(wide.o.begin(), wide.o.end(), [](float x) { return x != 0.f; }));
    EXPECT_TRUE(SameBits(wide.o, narrow.o));
    EXPECT_TRUE(SameBits(wide.lse, narrow.lse));
}

// qr_tdm_fp8_sched rejects scale blocks that split its M tile or a 32-key group; zero ones fail.
class QuantScaleBlockScaleReject : public TestWithParam<std::tuple<bool, std::tuple<int, int>>>
{
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(QuantScaleBlockScaleReject);
INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         QuantScaleBlockScaleReject,
                         Combine(EnableTestIf(kIsFp8Config),
                                 Values(std::tuple{128, 0},
                                        std::tuple{128, 16},
                                        std::tuple{128, 48},
                                        std::tuple{64, 128},
                                        std::tuple{0, 128})));

TEST_P(QuantScaleBlockScaleReject, DataTypeConfig)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "qr_tdm_fp8_sched is gfx125x only";
    DirectCase c;
    std::tie(c.bss_q, c.bss_kv) = std::get<1>(GetParam());
    const auto r                = RunDirect(c);
    // qr_tdm, when built, takes the M64 Q scale blocks that qr_tdm_fp8_sched leaves.
    if(c.bss_q > 0 && c.bss_kv > 0 && r.rc >= 0)
        EXPECT_EQ(DecodeKernelOf(r.kernel), "other") << r.kernel;
    else
        EXPECT_LT(r.rc, 0) << r.kernel;
}

// A larger max_seqlen_q or a padded LSE head stride keeps decode heads unmerged, same results.
class QuantScaleMergeStride
    : public TestWithParam<std::tuple<std::tuple<int, int>, std::tuple<int, int, int, int>>>
{
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(QuantScaleMergeStride);
INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         QuantScaleMergeStride,
                         Combine(ValuesIn(kDirectHdims),
                                 Values(std::tuple{2, 8, 2, 2},     // 8 rows, one-wavefront
                                        std::tuple{1, 16, 2, 8}))); // 64 rows, merged-GQA

TEST_P(QuantScaleMergeStride, DataTypeConfig)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "qr_tdm_fp8_sched is gfx125x only";
    auto [hdims, dims] = GetParam();
    DirectCase c;
    std::tie(c.hdim_q, c.hdim_v)                      = hdims;
    std::tie(c.batch, c.nhead, c.nhead_k, c.seqlen_q) = dims;
    c.seqlen_k                                        = 300;
    c.qscale                                          = quant_scale_enum::pertensor;
    const auto merged                                 = RunDirect(c);
    ASSERT_GE(merged.rc, 0) << merged.kernel;
    EXPECT_EQ(DecodeKernelOf(merged.kernel),
              ExpectedDecodeKernel(c.nhead, c.nhead_k, c.seqlen_q, true));
    for(int pad : {0, 1})
    {
        DirectCase u     = c;
        u.max_seqlen_q   = pad ? -1 : c.seqlen_q + 1;
        u.lse_head_pad   = pad;
        const auto plain = RunDirect(u);
        ASSERT_GE(plain.rc, 0) << plain.kernel;
        EXPECT_EQ(DecodeKernelOf(plain.kernel), "sched") << plain.kernel;
        EXPECT_TRUE(SameBits(plain.o, merged.o)) << plain.kernel;
        for(int bh = 0; bh < c.batch * c.nhead; ++bh)
            for(int i = 0; i < c.seqlen_q; ++i)
                EXPECT_EQ(plain.lse[bh * (c.seqlen_q + pad) + i], merged.lse[bh * c.seqlen_q + i]);
    }
}

// An infinite sink logit takes the whole row (O = 0, LSE = +inf); a NaN one poisons it.
class QuantScaleSinkNonFinite
    : public TestWithParam<std::tuple<std::tuple<int, int>, std::tuple<int, int, int, int>, bool>>
{
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(QuantScaleSinkNonFinite);
INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         QuantScaleSinkNonFinite,
                         Combine(ValuesIn(kDirectHdims),
                                 Values(std::tuple{2, 2, 1, 200}, // tiled
                                        std::tuple{2, 8, 2, 1},   // one-wavefront
                                        std::tuple{1, 16, 2, 8}), // merged-GQA
                                 Bool()));                        // NaN

TEST_P(QuantScaleSinkNonFinite, DataTypeConfig)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "qr_tdm_fp8_sched is gfx125x only";
    auto [hdims, dims, nan] = GetParam();
    DirectCase c;
    std::tie(c.hdim_q, c.hdim_v)                      = hdims;
    std::tie(c.batch, c.nhead, c.nhead_k, c.seqlen_q) = dims;
    c.seqlen_k                                        = 300;
    c.qscale                                          = quant_scale_enum::pertensor;
    const float bad =
        nan ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
    // Odd query heads keep a finite logit.
    for(int i = 0; i < c.nhead; ++i)
        c.sink.push_back(i % 2 ? 0.5f : bad);
    const auto r = RunDirect(c);
    ASSERT_GE(r.rc, 0) << r.kernel;
    ExpectFp8Sched(r.kernel, true);
    const int dv = c.hdim_v, sq = c.seqlen_q;
    for(int bh = 0; bh < c.batch * c.nhead; ++bh)
    {
        const bool finite = (bh % c.nhead) % 2;
        for(int i = 0; i < sq; ++i)
        {
            const float lse = r.lse[bh * sq + i];
            EXPECT_TRUE(finite ? std::isfinite(lse) : nan ? std::isnan(lse) : lse == bad) << lse;
            for(int j = 0; j < dv; ++j)
            {
                const float o = r.o[(bh * sq + i) * dv + j];
                EXPECT_TRUE(finite ? std::isfinite(o) : nan ? std::isnan(o) : o == 0.f) << o;
            }
        }
    }
}

// MX dispatch on zero data: unfoldable Q scale strides and group mode without V scale starts.
float RunMxProbe(bool group, int q_scale_head_pad, bool v_scale_seqstart, std::string& kernel)
{
    const int h = 8, hk = 2, sq = 2, sk = 256, d = 128, ds = d / 32, sv = sk / 32;
    constexpr std::size_t kSlot = 1 << 20;
    ck_tile::DeviceMem arena(8 * kSlot);
    arena.SetZero();
    const auto slot = [&](int i) {
        return static_cast<char*>(arena.GetDeviceBuffer()) + i * kSlot;
    };
    const std::vector<int32_t> seqstarts{0, sq, 0, sk, 0, sv};
    HIP_CHECK_ERROR(hipMemcpy(
        slot(7), seqstarts.data(), seqstarts.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    const auto* starts = reinterpret_cast<const int32_t*>(slot(7));

    fmha_fwd_args a{};
    a.selected_kernel_name   = &kernel;
    a.q_ptr                  = slot(0);
    a.k_ptr                  = slot(1);
    a.v_ptr                  = slot(2);
    a.q_descale_ptr          = slot(3);
    a.k_descale_ptr          = slot(4);
    a.v_descale_ptr          = slot(5);
    a.o_ptr                  = slot(6);
    a.lse_ptr                = slot(6) + kSlot / 2;
    a.seqstart_q_ptr         = group ? starts : nullptr;
    a.seqstart_k_ptr         = group ? starts + 2 : nullptr;
    a.seqstart_v_scale_ptr   = group && v_scale_seqstart ? starts + 4 : nullptr;
    a.seqlen_q               = sq;
    a.seqlen_k               = sk;
    a.batch                  = 1;
    a.max_seqlen_q           = sq;
    a.hdim_q                 = d;
    a.hdim_v                 = d;
    a.nhead_q                = h;
    a.nhead_k                = hk;
    a.num_head_q_total       = h;
    a.scale_s                = 1.f / std::sqrt(static_cast<float>(d));
    a.stride_q               = d;
    a.stride_k               = d;
    a.stride_v               = d;
    a.stride_o               = d;
    a.nhead_stride_q         = sq * d;
    a.nhead_stride_k         = sk * d;
    a.nhead_stride_v         = sk * d;
    a.nhead_stride_o         = sq * d;
    a.nhead_stride_lse       = sq;
    a.batch_stride_q         = h * sq * d;
    a.batch_stride_k         = hk * sk * d;
    a.batch_stride_v         = hk * sk * d;
    a.batch_stride_o         = h * sq * d;
    a.batch_stride_lse       = h * sq;
    a.stride_q_descale       = ds;
    a.stride_k_descale       = ds;
    a.stride_v_descale       = sv;
    a.nhead_stride_q_descale = sq * ds + q_scale_head_pad;
    a.nhead_stride_k_descale = sk * ds;
    a.nhead_stride_v_descale = d * sv;
    a.batch_stride_q_descale = h * a.nhead_stride_q_descale;
    a.batch_stride_k_descale = hk * sk * ds;
    a.batch_stride_v_descale = hk * d * sv;
    a.window_size_left       = -1;
    a.window_size_right      = -1;

    const fmha_fwd_traits traits{d,
                                 d,
                                 DirectDataType(),
                                 group,
                                 true,  // row-major V
                                 false, // no logits soft cap
                                 mask_enum::no_mask,
                                 bias_enum::no_bias,
                                 true,  // LSE
                                 false, // no dropout
                                 quant_scale_enum::mx};
    const float rc = fmha_fwd(traits, a, ck_tile::stream_config{nullptr, false});
    if(rc >= 0)
        HIP_CHECK_ERROR(hipDeviceSynchronize());
    return rc;
}

TEST(TestCkTileFmhaFwd, QuantScaleMxDispatch)
{
    if(!kIsMxFp8Config || !ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "gfx125x MX only";
    std::string kernel;
    EXPECT_GE(RunMxProbe(false, 0, true, kernel), 0) << kernel;
    EXPECT_EQ(DecodeKernelOf(kernel), "packgqa_32row") << kernel;
    EXPECT_GE(RunMxProbe(false, 4, true, kernel), 0) << kernel;
    EXPECT_EQ(DecodeKernelOf(kernel), "sched") << kernel;
    EXPECT_GE(RunMxProbe(true, 0, true, kernel), 0) << kernel;
    EXPECT_EQ(DecodeKernelOf(kernel), "sched") << kernel;
    kernel.clear();
    EXPECT_LT(RunMxProbe(true, 0, false, kernel), 0) << kernel;
    EXPECT_EQ(DecodeKernelOf(kernel), "sched") << kernel;
}

// MX scales on random FP8 data, dispatched under the given data type string.
DirectResult RunMxAlias(const std::string& data_type, bool group, int sq)
{
    using ODataType = typename FmhaFwdTypeConfig<DataTypeConfig>::ODataType;
    const int h = 8, hk = 2, sk = 256, d = 128, ds = d / 32, sv = sk / 32;
    std::mt19937 gen(321);
    std::uniform_real_distribution<float> value(-2.f, 2.f);
    std::vector<ck_tile::fp8_t> q(h * sq * d), k(hk * sk * d), v(hk * sk * d);
    for(auto* t : {&q, &k, &v})
        for(auto& x : *t)
            x = ck_tile::type_convert<ck_tile::fp8_t>(value(gen));
    // E8M0 exponents of 0.5, 1 and 2.
    std::vector<uint8_t> qs(h * sq * ds), ks(hk * sk * ds), vs(hk * d * sv);
    for(auto* t : {&qs, &ks, &vs})
        for(auto& x : *t)
            x = static_cast<uint8_t>(126 + gen() % 3);
    const std::vector<int32_t> seqstarts{0, sq, 0, sk, 0, sv};
    const auto upload = [](const auto& host) {
        auto dev = std::make_unique<ck_tile::DeviceMem>(host.size() * sizeof(host[0]));
        dev->ToDevice(host.data());
        return dev;
    };
    const auto q_buf = upload(q), k_buf = upload(k), v_buf = upload(v);
    const auto qs_buf = upload(qs), ks_buf = upload(ks), vs_buf = upload(vs);
    const auto starts_buf = upload(seqstarts);
    const auto* starts    = static_cast<const int32_t*>(starts_buf->GetDeviceBuffer());
    ck_tile::DeviceMem o_buf(h * sq * d * sizeof(ODataType)), lse_buf(h * sq * sizeof(float));
    o_buf.SetZero();

    DirectResult r;
    fmha_fwd_args a{};
    a.selected_kernel_name   = &r.kernel;
    a.q_ptr                  = q_buf->GetDeviceBuffer();
    a.k_ptr                  = k_buf->GetDeviceBuffer();
    a.v_ptr                  = v_buf->GetDeviceBuffer();
    a.q_descale_ptr          = qs_buf->GetDeviceBuffer();
    a.k_descale_ptr          = ks_buf->GetDeviceBuffer();
    a.v_descale_ptr          = vs_buf->GetDeviceBuffer();
    a.o_ptr                  = o_buf.GetDeviceBuffer();
    a.lse_ptr                = lse_buf.GetDeviceBuffer();
    a.seqstart_q_ptr         = group ? starts : nullptr;
    a.seqstart_k_ptr         = group ? starts + 2 : nullptr;
    a.seqstart_v_scale_ptr   = group ? starts + 4 : nullptr;
    a.seqlen_q               = sq;
    a.seqlen_k               = sk;
    a.batch                  = 1;
    a.max_seqlen_q           = sq;
    a.hdim_q                 = d;
    a.hdim_v                 = d;
    a.nhead_q                = h;
    a.nhead_k                = hk;
    a.num_head_q_total       = h;
    a.scale_s                = 1.f / std::sqrt(static_cast<float>(d));
    a.stride_q               = d;
    a.stride_k               = d;
    a.stride_v               = d;
    a.stride_o               = d;
    a.nhead_stride_q         = sq * d;
    a.nhead_stride_k         = sk * d;
    a.nhead_stride_v         = sk * d;
    a.nhead_stride_o         = sq * d;
    a.nhead_stride_lse       = sq;
    a.batch_stride_q         = h * sq * d;
    a.batch_stride_k         = hk * sk * d;
    a.batch_stride_v         = hk * sk * d;
    a.batch_stride_o         = h * sq * d;
    a.batch_stride_lse       = h * sq;
    a.stride_q_descale       = ds;
    a.stride_k_descale       = ds;
    a.stride_v_descale       = sv;
    a.nhead_stride_q_descale = sq * ds;
    a.nhead_stride_k_descale = sk * ds;
    a.nhead_stride_v_descale = d * sv;
    a.batch_stride_q_descale = h * sq * ds;
    a.batch_stride_k_descale = hk * sk * ds;
    a.batch_stride_v_descale = hk * d * sv;
    a.window_size_left       = -1;
    a.window_size_right      = -1;

    const fmha_fwd_traits traits{d,
                                 d,
                                 data_type,
                                 group,
                                 true,  // row-major V
                                 false, // no logits soft cap
                                 mask_enum::no_mask,
                                 bias_enum::no_bias,
                                 true,  // LSE
                                 false, // no dropout
                                 quant_scale_enum::mx};
    r.rc = fmha_fwd(traits, a, ck_tile::stream_config{nullptr, false});
    if(r.rc < 0)
        return r;
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    std::vector<ODataType> o(h * sq * d);
    o_buf.FromDevice(o.data());
    r.o.resize(o.size());
    std::transform(o.begin(), o.end(), r.o.begin(), [](ODataType x) {
        return ck_tile::type_convert<float>(x);
    });
    r.lse.resize(h * sq);
    lse_buf.FromDevice(r.lse.data());
    return r;
}

// fp8/fp8bf16 with MX scales run the mxfp8/mxfp8bf16 kernel and give its exact output.
TEST(TestCkTileFmhaFwd, QuantScaleMxFromFp8)
{
    if(!kIsFp8Config || !ck_tile::is_gfx125_supported())
        GTEST_SKIP() << "gfx125x FP8 only";
    const std::string dtype = DirectDataType();
    for(auto [group, sq] : {std::pair{false, 1}, std::pair{false, 200}, std::pair{true, 200}})
    {
        const auto alias = RunMxAlias(dtype, group, sq);
        const auto mx    = RunMxAlias("mx" + dtype, group, sq);
        ASSERT_GE(alias.rc, 0) << alias.kernel;
        ASSERT_GE(mx.rc, 0) << mx.kernel;
        EXPECT_EQ(alias.kernel, mx.kernel);
        EXPECT_NE(alias.kernel.find("_mx" + dtype + "_"), std::string::npos) << alias.kernel;
        EXPECT_NE(alias.kernel.find("qr_tdm_fp8_sched"), std::string::npos) << alias.kernel;
        EXPECT_TRUE(std::any_of(alias.o.begin(), alias.o.end(), [](float x) { return x != 0.f; }));
        EXPECT_TRUE(SameBits(alias.o, mx.o));
        EXPECT_TRUE(SameBits(alias.lse, mx.lse));
    }
}
#endif

// ---------------------------------------------------------------
// Negative tests: padding not supported with appendkv/splitkv/pagedkv
// ---------------------------------------------------------------

#if CK_TILE_FMHA_FWD_APPENDKV_API
TEST(TestCkTileFmhaFwd, AppendKvWithBatchEffLensShouldFail)
{
    // batch mode effective lengths simulate padding
    auto result = fmha_fwd_run<DataTypeConfig>(
        mode_enum::batch,
        2,          // batch
        4,          // nhead
        -1,         // nhead_k
        {128},      // seqlen_qs
        {128},      // seqlen_ks
        64,         // hdim_q
        64,         // hdim_v
        32,         // seqlen_knew -> triggers appendkv
        {},         // seqlen_qpads
        {},         // seqlen_kpads
        {100, 120}, // q_eff_lens_per_batch
        {90, 110},  // kv_eff_lens_per_batch
        0,          // rotary_dim
        true,       // i_perm
        true,       // o_perm
        0,          // scale_s
        0,          // logits_soft_cap
        def_is_v_rowmajor,
        def_lse,
        0,     // page_block_size
        false, // use_cache_batch_idx
        "n",   // bias
        0.0f,  // p_drop
        0,     // drop_seed
        0,     // drop_offset
        false, // drop_prefs
        "0",   // mask
        qscale_str,
        true, // is_rotary_interleaved
        1,    // num_splits
        init_method,
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        0,
        1, // init_sink
        0, // pack_gqa
        stream_config);
    ASSERT_EQ(result, fwd_result::invalid_args);
}
#endif

#if CK_TILE_FMHA_FWD_SPLITKV_API
TEST(TestCkTileFmhaFwd, SplitKvWithGroupPaddingShouldFail)
{
    // group mode physical padding
    auto result = fmha_fwd_run<DataTypeConfig>(
        mode_enum::group,
        2,          // batch
        4,          // nhead
        -1,         // nhead_k
        {96, 120},  // seqlen_qs logical
        {96, 120},  // seqlen_ks logical
        64,         // hdim_q
        64,         // hdim_v
        0,          // seqlen_knew
        {128, 128}, // seqlen_qpads
        {128, 128}, // seqlen_kpads
        {},         // q_eff
        {},         // kv_eff
        0,          // rotary_dim
        true,       // i_perm
        true,       // o_perm
        0,          // scale_s
        0,          // logits_soft_cap
        def_is_v_rowmajor,
        def_lse,
        0,     // page_block_size
        false, // use_cache_batch_idx
        "n",   // bias
        0.0f,
        0,
        0,
        false,
        "0",
        qscale_str,
        true,
        2, // num_splits (>1 triggers splitkv)
        init_method,
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        0,
        1, // init_sink
        0, // pack_gqa
        stream_config);
    ASSERT_EQ(result, fwd_result::invalid_args);
}
#endif

#if CK_TILE_FMHA_FWD_SPLITKV_API
// The split-KV sink seed has to undo the scale_s the kernel divides out of the
// sink value, and how much it has to put back depends on whether the score path
// already folded scale_s in. Soft cap does fold it in, via LogitsTransform, so
// it needs the same seed as bias and alibi rather than the plain-path seed.
// Getting that wrong only shows up when a sink and a soft cap are used at once
// on a split, which no other case here does.
class SplitKvSinkSoftCap
    : public TestWithParam<std::tuple<std::tuple<int, int>, int, float, std::string>>
{
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(SplitKvSinkSoftCap);

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         SplitKvSinkSoftCap,
                         Combine(SplitKVHDimValues,
                                 Values(1, 3),      // num_splits
                                 Values(0.f, 30.f), // logits_soft_cap
                                 Values("0", "b")));

TEST_P(SplitKvSinkSoftCap, DataTypeConfig)
{
    auto [hdims, num_splits, soft_cap, mask_str] = GetParam();
    auto [hdim_q, hdim_v]                        = hdims;

    auto result = fmha_fwd_run<DataTypeConfig>(
        mode_enum::batch,
        2,  // batch
        2,  // nhead
        -1, // nhead_k
        {256},
        {1024},
        hdim_q,
        hdim_v,
        0,     // seqlen_knew
        {-1},  // seqlen_qpads
        {-1},  // seqlen_kpads
        {},    // q_eff
        {},    // kv_eff
        0,     // rotary_dim
        true,  // i_perm
        false, // o_perm
        0,     // scale_s
        soft_cap,
        def_is_v_rowmajor,
        def_lse,
        0,     // page_block_size
        false, // use_cache_batch_idx
        "n",   // bias
        0.0f,  // p_drop
        0,     // drop_seed
        0,     // drop_offset
        false, // drop_prefs
        mask_str,
        qscale_str,
        true, // is_rotary_interleaved
        num_splits,
        init_method,
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1, // do_validation, without which this checks nothing
        1, // init_sink
        0, // pack_gqa
        stream_config);
    CHECK_RESULT(result);
}

// Two paths want to contribute the sink logit to the merged denominator: split
// 0 publishes it on its early exit, and a split with work seeds it when the
// split one back is empty. One empty leading split makes both fire, so the
// window masks below matter more than the split count.
class SplitKvSinkDoubleCount
    : public TestWithParam<std::tuple<std::tuple<int, int>, int, std::string>>
{
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(SplitKvSinkDoubleCount);

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         SplitKvSinkDoubleCount,
                         Combine(SplitKVHDimValues,
                                 Values(2, 4),                 // num_splits
                                 Values("b:64,0", "xt:128"))); // empty leading split

TEST_P(SplitKvSinkDoubleCount, DataTypeConfig)
{
    auto [hdims, num_splits, mask_str] = GetParam();
    auto [hdim_q, hdim_v]              = hdims;

    auto result = fmha_fwd_run<DataTypeConfig>(
        mode_enum::batch,
        1,  // batch
        4,  // nhead
        -1, // nhead_k
        {1024},
        {1024},
        hdim_q,
        hdim_v,
        0,    // seqlen_knew
        {-1}, // seqlen_qpads
        {-1}, // seqlen_kpads
        {},   // q_eff
        {},   // kv_eff
        0,    // rotary_dim
        true, // i_perm
        true, // o_perm
        0,    // scale_s
        0,    // logits_soft_cap
        def_is_v_rowmajor,
        true,  // lse, the double count moves it
        0,     // page_block_size
        false, // use_cache_batch_idx
        "n",   // bias, alibi has no split-KV instance under the test filter
        0.0f,  // p_drop
        0,     // drop_seed
        0,     // drop_offset
        false, // drop_prefs
        mask_str,
        qscale_str,
        true, // is_rotary_interleaved
        num_splits,
        init_method,
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1, // do_validation, without which this checks nothing
        1, // init_sink
        0, // pack_gqa
        stream_config);
    CHECK_RESULT(result);
}

// Both split-KV pipelines seed m at an i_split > 0 lookback; the nwarp one had
// the wrong expression there. It is selected only when seqlen_q fits a single
// 16-row tile, so nothing else in this file reaches it. Codegen emits it for
// RDNA only: on gfx9 these parameters dispatch qr_vr instead, so a pass there
// is not nwarp coverage.
class SplitKvSinkNwarpSeed
    : public TestWithParam<std::tuple<std::tuple<int, int>, std::string, float>>
{
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(SplitKvSinkNwarpSeed);

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         SplitKvSinkNwarpSeed,
                         Combine(SplitKVHDimValues,
                                 Values("xb:128", "b:64,0"),
                                 Values(0.f, 30.f))); // plain and soft-cap seed

TEST_P(SplitKvSinkNwarpSeed, DataTypeConfig)
{
    auto [hdims, mask_str, soft_cap] = GetParam();
    auto [hdim_q, hdim_v]            = hdims;

    auto result = fmha_fwd_run<DataTypeConfig>(
        mode_enum::batch,
        1,  // batch
        4,  // nhead
        -1, // nhead_k
        {16},
        {1024},
        hdim_q,
        hdim_v,
        0,    // seqlen_knew
        {-1}, // seqlen_qpads
        {-1}, // seqlen_kpads
        {},   // q_eff
        {},   // kv_eff
        0,    // rotary_dim
        true, // i_perm
        true, // o_perm
        0,    // scale_s
        soft_cap,
        def_is_v_rowmajor,
        true,  // lse
        0,     // page_block_size
        false, // use_cache_batch_idx
        "n",   // bias, alibi has no split-KV instance under the test filter
        0.0f,  // p_drop
        0,     // drop_seed
        0,     // drop_offset
        false, // drop_prefs
        mask_str,
        qscale_str,
        true, // is_rotary_interleaved
        4,
        init_method,
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1, // do_validation, without which this checks nothing
        1, // init_sink
        0, // pack_gqa
        stream_config);
    CHECK_RESULT(result);
}

// Query rows past seqlen_k have no ordinary KV work, so every split takes the
// early exit. The sink still belongs in their softmax denominator, and only
// split 0 is left to publish it, so a wrong ownership gate leaves the final
// LSE at -inf. This passes on develop: it guards against reintroducing that
// misreading, and is not evidence for the double count above.
class SplitKvSinkDroppedSink
    : public TestWithParam<std::tuple<std::tuple<int, int>, int, std::string>>
{
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(SplitKvSinkDroppedSink);

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         SplitKvSinkDroppedSink,
                         Combine(SplitKVHDimValues,
                                 Values(2, 4),                   // split-KV only
                                 Values("t:0,0,1", "t:0,0,0"))); // mask sink prefix

TEST_P(SplitKvSinkDroppedSink, DataTypeConfig)
{
    auto [hdims, num_splits, mask_str] = GetParam();
    auto [hdim_q, hdim_v]              = hdims;

    auto result = fmha_fwd_run<DataTypeConfig>(
        mode_enum::batch,
        1,  // batch
        4,  // nhead
        -1, // nhead_k
        {320},
        {64},
        hdim_q,
        hdim_v,
        0,    // seqlen_knew
        {-1}, // seqlen_qpads
        {-1}, // seqlen_kpads
        {},   // q_eff
        {},   // kv_eff
        0,    // rotary_dim
        true, // i_perm
        true, // o_perm
        0,    // scale_s
        0,    // logits_soft_cap
        def_is_v_rowmajor,
        true,  // lse, the whole point of this case
        0,     // page_block_size
        false, // use_cache_batch_idx
        "n",   // bias, alibi has no split-KV instance under the test filter
        0.0f,  // p_drop
        0,     // drop_seed
        0,     // drop_offset
        false, // drop_prefs
        mask_str,
        qscale_str,
        true, // is_rotary_interleaved
        num_splits,
        init_method,
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1, // do_validation, without which this checks nothing
        1, // init_sink
        0, // pack_gqa
        stream_config);
    CHECK_RESULT(result);
}
#endif // CK_TILE_FMHA_FWD_SPLITKV_API

#if CK_TILE_FMHA_FWD_PAGEDKV_API
TEST(TestCkTileFmhaFwd, PagedKvWithGroupPaddingShouldFail)
{
    auto result = fmha_fwd_run<DataTypeConfig>(
        mode_enum::group,
        2,
        4,
        -1,
        {80, 100},
        {80, 100},
        64,
        64,
        0,         // seqlen_knew
        {96, 128}, // seqlen_qpads
        {96, 128}, // seqlen_kpads
        {},
        {},
        0,
        true,
        true,
        0,
        0,
        def_is_v_rowmajor,
        def_lse,
        128, // page_block_size triggers pagedkv
        false,
        "n",
        0.0f,
        0,
        0,
        false,
        "0",
        qscale_str,
        true,
        1,
        init_method,
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        0,
        1, // init_sink
        0, // pack_gqa
        stream_config);
    ASSERT_EQ(result, fwd_result::invalid_args);
}
#endif

class HDimPadding
    : public TestWithParam<std::tuple<std::tuple<int, int>,
                                      bool,
                                      bool,
                                      mode_enum,
                                      std::tuple<int, int, int, int, int, int, std::string>>>
{
};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         HDimPadding,
                         Combine(Values(std::tuple{24, 48},
                                        std::tuple{120, 160},
                                        std::tuple{256, 108},
                                        std::tuple{40, 64}),
                                 Bool(),
                                 IsVRowmajorValues,
                                 ModeValues,
                                 Values(std::tuple{1, 4, 2, 480, -1, -1, "0"},
                                        std::tuple{2, 2, -1, 300, 400, 512, "t:64,64"},
                                        std::tuple{1, 4, 1, 512, 201, 256, "1"},
                                        std::tuple{1, 2, -1, 900, 256, -1, "0"},
                                        std::tuple{2, 1, -1, 256, 256, -1, "1"})));

TEST_P(HDimPadding, DataTypeConfig)
{
    auto [hdims, perm, is_v_rowmajor, mode, dims_mask]                      = GetParam();
    auto [hdim_q, hdim_v]                                                   = hdims;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, seqlen_kpad, mask_str] = dims_mask;

    auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {adjust_seqlen(seqlen_q)},
                                               {adjust_seqlen(seqlen_k)},
                                               adjust_hdim(hdim_q),
                                               adjust_hdim(hdim_v),
                                               0,             // seqlen_knew
                                               {-1},          // seqlen_qpads
                                               {seqlen_kpad}, // seqlen_kpads
                                               {},            // q_eff_lens_per_batch
                                               {},            // kv_eff_lens_per_batch
                                               0,             // rotary_dim
                                               perm,          // i_perm
                                               perm,          // o_perm
                                               0,             // scale_s
                                               0,             // logits_soft_cap
                                               is_v_rowmajor, // is_v_rowmajor
                                               def_lse,       // lse
                                               0,             // page_block_size
                                               false,         // use_cache_batch_idx
                                               "n",           // bias_str
                                               0.0f,          // p_drop
                                               0,             // drop_seed
                                               0,             // drop_offset
                                               false,         // drop_prefs
                                               mask_str,      // mask_str
                                               qscale_str,
                                               true, // is_rotary_interleaved
                                               1,    // num_splits
                                               COMMON_ARGS);
    CHECK_RESULT(result);
}

class ElementwiseBias
    : public TestWithParam<std::tuple<std::tuple<int, int>,
                                      bool,
                                      mode_enum,
                                      std::string,
                                      std::tuple<int, int, int, int, int, std::string>>>
{
};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         ElementwiseBias,
                         Combine(HDimValues,
                                 Bool(), // layout of bias is controlled by i_perm
                                 ModeValues,
                                 Values("e:0", "e:1", "e:2"),
                                 Values(std::tuple{1, 4, 2, 1024, 100, "0"},
                                        std::tuple{3, 2, -1, 128, 256, "2"},
                                        std::tuple{2, 2, -1, 130, 499, "t:50,64"})));

TEST_P(ElementwiseBias, DataTypeConfig)
{
    auto [hdims, i_perm, mode, bias_str, dims_mask]            = GetParam();
    auto [hdim_q, hdim_v]                                      = hdims;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str] = dims_mask;

    auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {adjust_seqlen(seqlen_q)},
                                               {adjust_seqlen(seqlen_k)},
                                               hdim_q,
                                               hdim_v,
                                               0,                 // seqlen_knew
                                               {-1},              // seqlen_qpads
                                               {-1},              // seqlen_kpads
                                               {},                // q_eff_lens_per_batch
                                               {},                // kv_eff_lens_per_batch
                                               0,                 // rotary_dim
                                               i_perm,            // i_perm
                                               false,             // o_perm
                                               0,                 // scale_s
                                               0,                 // logits_soft_cap
                                               def_is_v_rowmajor, // is_v_rowmajor
                                               def_lse,           // lse
                                               0,                 // page_block_size
                                               false,             // use_cache_batch_idx
                                               bias_str,          // bias_str
                                               0.0f,              // p_drop
                                               0,                 // drop_seed
                                               0,                 // drop_offset
                                               false,             // drop_prefs
                                               mask_str,          // mask_str
                                               qscale_str,
                                               true, // is_rotary_interleaved
                                               1,    // num_splits
                                               COMMON_ARGS);
    CHECK_RESULT(result);
}

class Alibi : public TestWithParam<std::tuple<std::tuple<int, int>,
                                              mode_enum,
                                              std::string,
                                              std::tuple<int, int, int, int, int>,
                                              std::string>>
{
};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         Alibi,
                         Combine(HDimValues,
                                 ModeValues,
                                 Values("a:0", "a:1"),
                                 Values(std::tuple{1, 3, 3, 1024, 1000},
                                        std::tuple{3, 5, 5, 128, 256},
                                        std::tuple{2, 8, 2, 300, 355}),
                                 Values("0", "t", "b", "t:50,64", "b:32,40")));

TEST_P(Alibi, DataTypeConfig)
{
    auto [hdims, mode, bias_str, dims, mask_str]     = GetParam();
    auto [hdim_q, hdim_v]                            = hdims;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k] = dims;

    auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {adjust_seqlen(seqlen_q)},
                                               {adjust_seqlen(seqlen_k)},
                                               hdim_q,
                                               hdim_v,
                                               0,                 // seqlen_knew
                                               {-1},              // seqlen_qpads
                                               {-1},              // seqlen_kpads
                                               {},                // q_eff_lens_per_batch
                                               {},                // kv_eff_lens_per_batch
                                               0,                 // rotary_dim
                                               true,              // i_perm
                                               true,              // o_perm
                                               0,                 // scale_s
                                               0,                 // logits_soft_cap
                                               def_is_v_rowmajor, // is_v_rowmajor
                                               def_lse,           // lse
                                               0,                 // page_block_size
                                               false,             // use_cache_batch_idx
                                               bias_str,          // bias_str
                                               0.0f,              // p_drop
                                               0,                 // drop_seed
                                               0,                 // drop_offset
                                               false,             // drop_prefs
                                               mask_str,          // mask_str
                                               qscale_str,
                                               true, // is_rotary_interleaved
                                               1,    // num_splits
                                               COMMON_ARGS);
    CHECK_RESULT(result);
}

class Dropout : public TestWithParam<std::tuple<std::tuple<int, int>,
                                                mode_enum,
                                                float,
                                                std::tuple<uint64_t, uint64_t, bool>,
                                                std::tuple<int, int, int, int, int, std::string>>>
{
};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         Dropout,
                         Combine(HDimValues,
                                 ModeValues,
                                 Values(0.123f, 0.5f),
                                 Values(std::tuple{10, 123, false},
                                        std::tuple{34534564645, 7876878876864, true}),
                                 Values(std::tuple{2, 4, 2, 280, 512, "0"},
                                        std::tuple{3, 2, 2, 256, 128, "1"},
                                        std::tuple{4, 3, 1, 100, 768, "2"})));

TEST_P(Dropout, DataTypeConfig)
{
    auto [hdims, mode, p_drop, drop_seed_offset_prefs, dims_mask] = GetParam();
    auto [hdim_q, hdim_v]                                         = hdims;
    auto [drop_seed, drop_offset, drop_prefs]                     = drop_seed_offset_prefs;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str]    = dims_mask;

#if CK_TILE_WORKAROUND_ROCM_7_12_FP16_DROPOUT_MISCOMPILE
    if constexpr(std::is_same_v<DataTypeConfig, FmhaFwdFp16>)
    {
        if(hdim_q > 128 && mode == mode_enum::batch)
            GTEST_SKIP() << "Skipped: fp16 dropout d256 batch - compiler bug (ROCm >= 7.12)";
    }
#endif

    auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {adjust_seqlen(seqlen_q)},
                                               {adjust_seqlen(seqlen_k)},
                                               hdim_q,
                                               hdim_v,
                                               0,                 // seqlen_knew
                                               {-1},              // seqlen_qpads
                                               {-1},              // seqlen_kpads
                                               {},                // q_eff_lens_per_batch
                                               {},                // kv_eff_lens_per_batch
                                               0,                 // rotary_dim
                                               false,             // i_perm
                                               false,             // o_perm
                                               0,                 // scale_s
                                               0,                 // logits_soft_cap
                                               def_is_v_rowmajor, // is_v_rowmajor
                                               def_lse,           // lse
                                               0,                 // page_block_size
                                               false,             // use_cache_batch_idx
                                               "n",               // bias_str
                                               p_drop,            // p_drop
                                               drop_seed,         // drop_seed
                                               drop_offset,       // drop_offset
                                               drop_prefs,        // drop_prefs
                                               mask_str,          // mask_str
                                               qscale_str,
                                               true, // is_rotary_interleaved
                                               1,    // num_splits
                                               COMMON_ARGS);
    CHECK_RESULT(result);
}

#if CK_TILE_FMHA_FWD_PAGEDKV_API

class PagedKV : public TestWithParam<std::tuple<std::tuple<int, int>,
                                                bool,
                                                bool,
                                                mode_enum,
                                                int,
                                                std::tuple<int, int, int, int, int, std::string>>>
{
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(PagedKV);

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         PagedKV,
                         Combine(SplitKVHDimValues,
                                 Bool(),            // layouts of k and v are controlled by i_perm
                                 IsVRowmajorValues, // layout of v is controlled by is_v_rowmajor
                                 ModeValues,
                                 Values(128, 256),
                                 Values(std::tuple{2, 3, 1, 200, 1024, "0"},
                                        std::tuple{3, 2, -1, 128, 768, "2"},
                                        std::tuple{2, 2, -1, 230, 899, "t:50,64"})));

TEST_P(PagedKV, DataTypeConfig)
{
    auto [hdims, i_perm, is_v_rowmajor, mode, page_block_size, dims_mask] = GetParam();
    auto [hdim_q, hdim_v]                                                 = hdims;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str]            = dims_mask;

    auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {adjust_seqlen(seqlen_q)},
                                               {adjust_seqlen(seqlen_k)},
                                               hdim_q,
                                               hdim_v,
                                               0,               // seqlen_knew
                                               {-1},            // seqlen_qpads
                                               {-1},            // seqlen_kpads
                                               {},              // q_eff_lens_per_batch
                                               {},              // kv_eff_lens_per_batch
                                               0,               // rotary_dim
                                               i_perm,          // i_perm
                                               false,           // o_perm
                                               0,               // scale_s
                                               0,               // logits_soft_cap
                                               is_v_rowmajor,   // is_v_rowmajor
                                               false,           // lse
                                               page_block_size, // page_block_size
                                               false,           // use_cache_batch_idx
                                               "n",             // bias_str
                                               0.0f,            // p_drop
                                               0,               // drop_seed
                                               0,               // drop_offset
                                               false,           // drop_prefs
                                               mask_str,        // mask_str
                                               qscale_str,
                                               true, // is_rotary_interleaved
                                               1,    // num_splits
                                               COMMON_ARGS);
    CHECK_RESULT(result);
}

#endif // CK_TILE_FMHA_FWD_PAGEDKV_API

#if CK_TILE_FMHA_FWD_SPLITKV_API

class SplitKV : public TestWithParam<std::tuple<std::tuple<int, int>,
                                                bool,
                                                bool,
                                                std::tuple<mode_enum, bool>,
                                                int,
                                                std::tuple<int, int, int, int, int, std::string>>>
{
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(SplitKV);

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         SplitKV,
                         Combine(SplitKVHDimValues,
                                 Bool(),            // layouts of k and v are controlled by i_perm
                                 IsVRowmajorValues, // layout of v is controlled by is_v_rowmajor
                                 Values(std::tuple{mode_enum::batch, false},
                                        std::tuple{mode_enum::batch, true},
                                        std::tuple{mode_enum::group, false}),
                                 Values(3, 4),
                                 Values(std::tuple{4, 3, 1, 200, 1024, "0"},
                                        std::tuple{2, 2, -1, 512, 2000, "0"},
                                        std::tuple{2, 8, 2, 1, 1024, "0"},
                                        std::tuple{3, 2, -1, 230, 899, "t:128,128"})));

TEST_P(SplitKV, DataTypeConfig)
{
    auto [hdims, i_perm, is_v_rowmajor, mode_use_cache_batch_idx, num_splits, dims_mask] =
        GetParam();
    auto [hdim_q, hdim_v]                                      = hdims;
    auto [mode, use_cache_batch_idx]                           = mode_use_cache_batch_idx;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str] = dims_mask;

    auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {adjust_seqlen(seqlen_q)},
                                               {adjust_seqlen(seqlen_k)},
                                               hdim_q,
                                               hdim_v,
                                               0,                   // seqlen_knew
                                               {-1},                // seqlen_qpads
                                               {-1},                // seqlen_kpads
                                               {},                  // q_eff_lens_per_batch
                                               {},                  // kv_eff_lens_per_batch
                                               0,                   // rotary_dim
                                               i_perm,              // i_perm
                                               false,               // o_perm
                                               0,                   // scale_s
                                               0,                   // logits_soft_cap
                                               is_v_rowmajor,       // is_v_rowmajor
                                               def_lse,             // lse
                                               0,                   // page_block_size
                                               use_cache_batch_idx, // use_cache_batch_idx
                                               "n",                 // bias_str
                                               0.0f,                // p_drop
                                               0,                   // drop_seed
                                               0,                   // drop_offset
                                               false,               // drop_prefs
                                               mask_str,            // mask_str
                                               qscale_str,
                                               true,       // is_rotary_interleaved
                                               num_splits, // num_splits
                                               COMMON_ARGS);
    CHECK_RESULT(result);
}

#endif // CK_TILE_FMHA_FWD_SPLITKV_API

#if CK_TILE_FMHA_FWD_APPENDKV_API

class AppendKV : public TestWithParam<std::tuple<std::tuple<int, int>,
                                                 bool,
                                                 bool,
                                                 std::tuple<int, bool>,
                                                 int,
                                                 std::tuple<int, int, int, int, int, std::string>>>
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaFwd,
    AppendKV,
    Combine(AppendKVHDimValues,
            Bool(),            // layouts of k and v are controlled by i_perm
            IsVRowmajorValues, // layout of v is controlled by is_v_rowmajor
            ValuesIn({std::tuple{0, true}, std::tuple{0, false}, std::tuple{128, false}}),
            Values(1, 64, -1),
            Values(std::tuple{3, 3, -1, 60, 129, "t:32,32"},
                   std::tuple{3, 2, 2, 256, 256, "0"},
                   std::tuple{2, 3, 1, 264, 265, "1"},
                   std::tuple{4, 4, 2, 71, 64, "1"})));

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AppendKV);

TEST_P(AppendKV, DataTypeConfig)
{
    auto [hdims,
          i_perm,
          is_v_rowmajor,
          page_block_size_use_cache_batch_idx,
          seqlen_knew,
          dims_mask]                            = GetParam();
    auto [hdim_q, hdim_v]                       = hdims;
    auto [page_block_size, use_cache_batch_idx] = page_block_size_use_cache_batch_idx;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str] = dims_mask;

    seqlen_knew = seqlen_knew == -1 ? seqlen_k : seqlen_knew;

    auto result = fmha_fwd_run<DataTypeConfig>(mode_enum::batch,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {adjust_seqlen(seqlen_q)},
                                               {adjust_seqlen(seqlen_k)},
                                               hdim_q,
                                               hdim_v,
                                               seqlen_knew,         // seqlen_knew
                                               {-1},                // seqlen_qpads
                                               {-1},                // seqlen_kpads
                                               {},                  // q_eff_lens_per_batch
                                               {},                  // kv_eff_lens_per_batch
                                               0,                   // rotary_dim
                                               i_perm,              // i_perm
                                               true,                // o_perm
                                               0,                   // scale_s
                                               0,                   // logits_soft_cap
                                               is_v_rowmajor,       // is_v_rowmajor
                                               def_lse,             // lse
                                               page_block_size,     // page_block_size
                                               use_cache_batch_idx, // use_cache_batch_idx
                                               "n",                 // bias_str
                                               0.0f,                // p_drop
                                               0,                   // drop_seed
                                               0,                   // drop_offset
                                               false,               // drop_prefs
                                               mask_str,            // mask_str
                                               qscale_str,
                                               false, // is_rotary_interleaved
                                               1,     // num_splits
                                               COMMON_ARGS);
    CHECK_RESULT(result);
}

class AppendKVRoPE
    : public TestWithParam<std::tuple<bool,
                                      std::tuple<int, int>,
                                      bool,
                                      bool,
                                      std::tuple<int, bool>,
                                      int,
                                      std::tuple<int, int, int, int, int, std::string>>>
{
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AppendKVRoPE);

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         AppendKVRoPE,
                         Combine(EnableTestIf(!std::is_same_v<DataTypeConfig, FmhaFwdFp8Bf16>),
                                 AppendKVHDimValues,
                                 Bool(),            // layouts of k and v are controlled by i_perm
                                 IsVRowmajorValues, // layout of v is controlled by is_v_rowmajor
                                 Values(std::tuple{0, false},
                                        std::tuple{16, true},
                                        std::tuple{32, false},
                                        std::tuple{-1, true}),
                                 Values(16, 50, -1),
                                 Values(std::tuple{2, 3, -1, 60, 129, "t:32,32"},
                                        std::tuple{1, 2, 1, 128, 55, "0"},
                                        std::tuple{3, 4, 2, 72, 128, "1"})));

TEST_P(AppendKVRoPE, DataTypeConfig)
{
    auto [_, hdims, i_perm, is_v_rowmajor, rotary, seqlen_knew, dims_mask] = GetParam();
    auto [hdim_q, hdim_v]                                                  = hdims;
    auto [rotary_dim, is_rotary_interleaved]                               = rotary;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str]             = dims_mask;

    rotary_dim  = rotary_dim == -1 ? hdim_q : rotary_dim;
    seqlen_knew = seqlen_knew == -1 ? seqlen_k : seqlen_knew;

    auto result = fmha_fwd_run<DataTypeConfig>(mode_enum::batch,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {adjust_seqlen(seqlen_q)},
                                               {adjust_seqlen(seqlen_k)},
                                               hdim_q,
                                               hdim_v,
                                               seqlen_knew,   // seqlen_knew
                                               {-1},          // seqlen_qpads
                                               {-1},          // seqlen_kpads
                                               {},            // q_eff_lens_per_batch
                                               {},            // kv_eff_lens_per_batch
                                               rotary_dim,    // rotary_dim
                                               i_perm,        // i_perm
                                               true,          // o_perm
                                               0,             // scale_s
                                               0,             // logits_soft_cap
                                               is_v_rowmajor, // is_v_rowmajor
                                               true,          // lse
                                               0,             // page_block_size
                                               false,         // use_cache_batch_idx
                                               "n",           // bias_str
                                               0.0f,          // p_drop
                                               0,             // drop_seed
                                               0,             // drop_offset
                                               false,         // drop_prefs
                                               mask_str,      // mask_str
                                               qscale_str,
                                               is_rotary_interleaved, // is_rotary_interleaved
                                               1,                     // num_splits
                                               COMMON_ARGS);
    CHECK_RESULT(result);
}

#endif // CK_TILE_FMHA_FWD_APPENDKV_API

// ---------------------------------------------------------------
// Parameterized padding tests (batch & group) using Combine+Values
// ---------------------------------------------------------------

using PaddingParam = std::tuple<mode_enum,        // mode
                                int,              // batch
                                int,              // nhead
                                int,              // nhead_k
                                std::vector<int>, // seqlen_qs (logical)
                                std::vector<int>, // seqlen_ks (logical)
                                std::vector<int>, // seqlen_qpads (physical padded lengths)
                                std::vector<int>, // seqlen_kpads (physical padded lengths)
                                std::vector<int>, // q_eff_lens
                                std::vector<int>, // kv_eff_lens
                                bool,             // i_perm
                                bool,             // o_perm
                                std::string>;     // mask_str

class PaddingCases : public TestWithParam<PaddingParam>
{
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(PaddingCases);

// Build padding test params programmatically to enforce constraints
static std::vector<PaddingParam> BuildPaddingParams()
{
    std::vector<PaddingParam> params;

    if constexpr(ck_tile::is_any_of<DataTypeConfig,
                                    FmhaFwdFp8Bf16,
                                    FmhaFwdMxFp8,
                                    FmhaFwdMxFp8Bf16,
                                    FmhaFwdMxFp4>::value)
    {
        return params;
    }

    // mask variants to cover
    const std::vector<std::string> mask_variants{"0", "t:50,64", "b:32,40"};
    const std::vector<std::string> mask_variants_reduced{"0", "t:50,64"}; // used for trimmed sets

    // Representative ratio pairs (q_ratio, k_ratio) to avoid explosion
    const std::vector<std::pair<double, double>> ratio_pairs_full{
        {1.0, 1.0}, // both full
        {1.0, 0.5}, // q full, k half
        {0.5, 1.0}, // q half, k full
    };
    const std::vector<std::pair<double, double>> ratio_pairs_reduced{{1.0, 1.0}, {0.5, 1.0}};

    // candidate physical seqlens for batch mode (single value) & for group mode (per batch)
    const std::vector<int> physical_lengths_full{64, 128, 256};
    const std::vector<int> physical_lengths_reduced{64};

    // batch sizes to sample
    const std::vector<int> batch_sizes{1, 4};
    // --------------------------------------------------------------------
    // Head configuration space (cover MHA, GQA, MQA)
    //  - Standard MHA: nhead_k == -1 (treated internally as nhead)
    //  - GQA: nhead_k > 0 and nhead % nhead_k == 0, nhead_k < nhead
    //  - MQA: nhead_k == 1
    // We choose (9, -1), (9, 3), (9, 1) so that divisibility holds. Full
    // combinatorics only applied to the first (standard) configuration to
    // avoid test explosion.
    // --------------------------------------------------------------------
    struct HeadCfg
    {
        int nhead;
        int nhead_k; // -1 for standard; else must divide nhead
        bool full;   // whether to use full coverage sets
    };
    const std::vector<HeadCfg> head_cfgs = {
        {9, -1, true}, // MHA full
        {9, 3, false}, // GQA reduced (nhead/nhead_k=3)
        {9, 1, false}  // MQA reduced
    };

    // Helper to clamp and ensure >=1
    auto logical_len = [](int physical, double ratio) {
        int v = static_cast<int>(std::round(physical * ratio));
        v     = std::max(1, std::min(v, physical));
        return v;
    };
    // Iterate over head configurations
    for(const auto& hc : head_cfgs)
    {
        const auto& ratio_pairs        = hc.full ? ratio_pairs_full : ratio_pairs_reduced;
        const auto& phys_lengths_batch = hc.full ? physical_lengths_full : physical_lengths_reduced;
        const auto& phys_lengths_group_q = phys_lengths_batch; // reuse
        const auto& phys_lengths_group_k = phys_lengths_batch; // reuse
        const auto& masks                = hc.full ? mask_variants : mask_variants_reduced;

        // -----------------
        // Batch mode params (effective lengths only)
        // -----------------
        for(int b : batch_sizes)
        {
            for(int phys_qkv : phys_lengths_batch)
            {
                for(const auto& rkpair : ratio_pairs)
                {
                    double rq = rkpair.first;
                    double rk = rkpair.second;
                    std::vector<int> q_eff(b), kv_eff(b);
                    int log_q = logical_len(phys_qkv, rq);
                    int log_k = logical_len(phys_qkv, rk);
                    for(int i = 0; i < b; ++i)
                    {
                        q_eff[i]  = log_q;
                        kv_eff[i] = log_k;
                    }
                    for(const auto& mask : masks)
                    {
                        params.emplace_back(PaddingParam{mode_enum::batch,
                                                         b,
                                                         hc.nhead,
                                                         hc.nhead_k,
                                                         {phys_qkv}, // seqlen_qs
                                                         {phys_qkv}, // seqlen_ks
                                                         {},         // seqlen_qpads
                                                         {},         // seqlen_kpads
                                                         q_eff,
                                                         kv_eff,
                                                         true,
                                                         true,
                                                         mask});
                    }
                }
                // Single-token logical length case (both q & k = 1)
                for(const auto& mask : masks)
                {
                    std::vector<int> q_eff(b, 1), kv_eff(b, 1);
                    params.emplace_back(PaddingParam{mode_enum::batch,
                                                     b,
                                                     hc.nhead,
                                                     hc.nhead_k,
                                                     {phys_qkv},
                                                     {phys_qkv},
                                                     {},
                                                     {},
                                                     q_eff,
                                                     kv_eff,
                                                     true,
                                                     true,
                                                     mask});
                }
            }
        }

        // -----------------
        // Group mode params (physical padding + logical variants)
        // -----------------
        for(int b : batch_sizes)
        {
            for(int phys_q : phys_lengths_group_q)
            {
                for(int phys_k : phys_lengths_group_k)
                {
                    for(const auto& rkpair : ratio_pairs)
                    {
                        double rq = rkpair.first;
                        double rk = rkpair.second;
                        std::vector<int> seqlen_qs(b), seqlen_ks(b), seqlen_qpads(b),
                            seqlen_kpads(b);
                        for(int i = 0; i < b; ++i)
                        {
                            seqlen_qpads[i] = phys_q;
                            seqlen_kpads[i] = phys_k;
                            seqlen_qs[i]    = logical_len(phys_q, rq);
                            seqlen_ks[i]    = logical_len(phys_k, rk);
                        }
                        std::array<std::pair<std::vector<int>, std::vector<int>>, 3> pad_variants{
                            std::pair{seqlen_qpads, seqlen_kpads}, // both
                            std::pair{seqlen_qpads, seqlen_ks},    // only q padding
                            std::pair{seqlen_qs, seqlen_kpads}     // only kv padding
                        };
                        for(const auto& mask : masks)
                        {
                            for(const auto& pv : pad_variants)
                            {
                                params.emplace_back(PaddingParam{mode_enum::group,
                                                                 b,
                                                                 hc.nhead,
                                                                 hc.nhead_k,
                                                                 seqlen_qs,
                                                                 seqlen_ks,
                                                                 pv.first,
                                                                 pv.second,
                                                                 {},
                                                                 {},
                                                                 true,
                                                                 true,
                                                                 mask});
                            }
                        }
                    }
                    // Single-token logical length case
                    for(const auto& mask : masks)
                    {
                        std::vector<int> seqlen_qs(b, 1), seqlen_ks(b, 1);
                        std::vector<int> seqlen_qpads(b, phys_q), seqlen_kpads(b, phys_k);
                        // both padding variant only (others degenerate)
                        params.emplace_back(PaddingParam{mode_enum::group,
                                                         b,
                                                         hc.nhead,
                                                         hc.nhead_k,
                                                         seqlen_qs,
                                                         seqlen_ks,
                                                         seqlen_qpads,
                                                         seqlen_kpads,
                                                         {},
                                                         {},
                                                         true,
                                                         true,
                                                         mask});
                    }
                }
            }
        }
    }

    return params;
}

static const std::vector<PaddingParam> kPaddingParams = BuildPaddingParams();

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd, PaddingCases, ValuesIn(kPaddingParams));

TEST_P(PaddingCases, DataTypeConfig)
{
    auto [mode,
          batch,
          nhead,
          nhead_k,
          seqlen_qs,
          seqlen_ks,
          seqlen_qpads,
          seqlen_kpads,
          q_eff_lens,
          kv_eff_lens,
          i_perm,
          o_perm,
          mask_str] = GetParam();

    // For batch mode we wrap single logical lengths with adjust_seqlen.
    std::vector<int> adj_qs =
        (mode == mode_enum::batch) ? std::vector<int>{adjust_seqlen(seqlen_qs.at(0))} : seqlen_qs;
    std::vector<int> adj_ks =
        (mode == mode_enum::batch) ? std::vector<int>{adjust_seqlen(seqlen_ks.at(0))} : seqlen_ks;

    const int hdim_q      = 64;
    const int hdim_v      = 64;
    const int seqlen_knew = 0;

    auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               adj_qs,
                                               adj_ks,
                                               hdim_q,
                                               hdim_v,
                                               seqlen_knew,  // seqlen_knew
                                               seqlen_qpads, // seqlen_qpads
                                               seqlen_kpads, // seqlen_kpads
                                               q_eff_lens,   // q_eff_lens_per_batch
                                               kv_eff_lens,  // kv_eff_lens_per_batch
                                               0,            // rotary_dim
                                               i_perm,       // i_perm
                                               o_perm,       // o_perm
                                               0,            // scale_s
                                               0,            // logits_soft_cap
                                               def_is_v_rowmajor,
                                               def_lse,  // lse
                                               0,        // page_block_size
                                               false,    // use_cache_batch_idx
                                               "n",      // bias_str
                                               0.0f,     // p_drop
                                               0,        // drop_seed
                                               0,        // drop_offset
                                               false,    // drop_prefs
                                               mask_str, // mask_str
                                               qscale_str,
                                               true, // is_rotary_interleaved
                                               1,    // num_splits
                                               COMMON_ARGS);
    CHECK_RESULT(result);
}

// ============================================================================
// Sink combined with a sliding-window mask: tile-window offsets
// ----------------------------------------------------------------------------
// With a sink enabled, the qr_ks_vs pipelines first loop over a prefix of
// always-attended KV columns and then jump the K/V, bias and randval tile
// windows forward to seqlen_k_start. That jump must happen only when a sink
// phase actually ran, and only on its last iteration: the windows already start
// at kv_load_start, which equals seqlen_k_start when there is no sink prefix,
// so jumping unconditionally offsets them a second time.
//
// Reaching the case where the guard matters needs a window mask, so that
// seqlen_k_start > 0 for later query blocks, plus one of:
//   - sink_size == 0 while a sink pointer is set (init_sink != 0), giving
//     num_sink_loop == 0 with the windows already at seqlen_k_start;
//   - sink_size spanning more than one KV tile, giving num_sink_loop >= 2 so
//     that the first sink iteration is not the last one.
//
// Neither shape was covered before: a causal mask keeps seqlen_k_start == 0, so
// the extra offset is 0, and sink_size <= kN0 makes num_sink_loop == 1, where a
// guard on the first iteration coincides with a guard on the last.
//
// bias and hdim select the pipeline: elementwise bias or hdim 256 pick the
// non-async qr_ks_vs, where the K/V windows are affected too, while hdim 128
// with plain bias picks the async one; dropout exercises the randval window.
// ============================================================================

// hdim, bias_str, p_drop, seqlen_q/k, mask_str, init_sink
using SinkWindowParam = std::tuple<int, std::string, float, int, std::string, int>;

static const std::vector<SinkWindowParam> kSinkWindowParams = {
    // num_sink_loop == 0: sink pointer set, no sink columns, window mask
    {256, "n", 0.0f, 1024, "t:128,30", 1},
    {128, "e", 0.0f, 1024, "t:128,30", 1},
    // On gfx1250 both bias rows reach qr_tdm, where m is already log2-domain and the sink
    // pre-seed must carry scale_s. alibi has no fp8 instance, so only fp16/bf16 run it.
    {128, "a", 0.0f, 1024, "t:128,30", 1},
    // dropout exercises the randval window, which each pipeline guards separately. hdim 128
    // with plain bias routes to async, hdim 256 keeps it on the non-async one.
    {128, "n", 0.2f, 1024, "t:128,30", 1},
    {256, "n", 0.2f, 1024, "t:128,30", 1},
    {256, "n", 0.0f, 1024, "b:128,30", 1},
    {256, "n", 0.0f, 1024, "t:2,0", 1},
    // num_sink_loop >= 2: sink_size wider than one KV tile
    {256, "n", 0.0f, 2048, "t:128,30,256", 0},
    {128, "e", 0.0f, 2048, "t:128,30,256", 0},
    {256, "n", 0.0f, 2048, "b:128,30,256", 0},
    // hdim 128 with plain bias, so bf16/fp16 reach the async pipeline. It carries
    // its own copy of the K/V jump, which is wrong for any sink_size, hence the
    // num_sink_loop == 1 shape here as well.
    // async
    {128, "n", 0.0f, 2048, "t:128,33,4", 0},
    // sync
    {256, "n", 0.0f, 2048, "b:128,33,4", 0}};

class SinkWindowMask : public TestWithParam<std::tuple<mode_enum, SinkWindowParam>>
{
};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         SinkWindowMask,
                         Combine(ModeValues, ValuesIn(kSinkWindowParams)));

TEST_P(SinkWindowMask, DataTypeConfig)
{
    auto [mode, sink_param]                                    = GetParam();
    auto [hdim, bias_str, p_drop, seqlen, mask_str, init_sink] = sink_param;

    auto result = fmha_fwd_run<DataTypeConfig>(
        mode,
        2,  // batch
        2,  // nhead
        -1, // nhead_k
        {adjust_seqlen(seqlen)},
        {adjust_seqlen(seqlen)},
        adjust_hdim(hdim),
        adjust_hdim(hdim),
        0,    // seqlen_knew
        {-1}, // seqlen_qpads
        {-1}, // seqlen_kpads
        {},   // q_eff_lens_per_batch
        {},   // kv_eff_lens_per_batch
        0,    // rotary_dim
        true, // i_perm
        true, // o_perm
        0,    // scale_s
        0,    // logits_soft_cap
        def_is_v_rowmajor,
        def_lse,  // lse
        0,        // page_block_size
        false,    // use_cache_batch_idx
        bias_str, // bias_str
        p_drop,   // p_drop
        0,        // drop_seed
        0,        // drop_offset
        false,    // drop_prefs
        mask_str, // mask_str
        qscale_str,
        true, // is_rotary_interleaved
        1,    // num_splits
        init_method,
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1,         // do_validation
        init_sink, // init_sink_value
        1,         // pack_gqa
        stream_config);
    CHECK_RESULT(result);
}

// The tiny-scale_s rows below need the sink kept out of the fp8 P quantization frame,
// which only the gfx125x qr_tdm pipeline does; every other target dispatches qr_async_vr
// and still carries the systematic OUT gain. So the suite is compiled in there alone.
#ifdef CK_TILE_TEST_FMHA_GPTOSS_SINK
// ============================================================================
// gptoss sink: one learnable logit per Q head, with no sink columns in the mask.
// It only turns kHasSink on and never adds sink tiles (qr_ks_vs.hpp:791), so
// num_sink_loop is always 0 and none of the window-jump guards SinkWindowMask
// exists for are reached. What runs instead is:
//
//   - the m pre-seed, which branches on bias type: no bias takes
//     sink_v * log2e, elementwise bias and alibi take sink_v * scale_s * log2e
//     (qr_ks_vs.hpp:326, _async.hpp:325, _tdm.hpp:357 and :1050);
//   - the LSE pre-seed, set_tile(lse, sink_v * scale_s), on the same four sites;
//   - the per-head lookup sink_ptr[i_nhead] in the kernel.
//
// lse follows def_lse, so on the fp8 family only OUT is checked. That is enough
// for the pre-seed rows, where a mis-seeded m collapses P and moves OUT, but not
// for the head-lookup rows, which move LSE alone; those are carried by fp16/bf16.
// ============================================================================

enum class gqa_kind
{
    mha,        // nhead_k == nhead
    gqa,        // ratio 4, packing left off
    gqa_packed, // ratio 4, packing requested
};

// hdim, bias_str, seqlen_q, seqlen_k, gqa, scale_s
using GptossSinkParam = std::tuple<int, std::string, int, int, gqa_kind, float>;

static const std::vector<GptossSinkParam> kGptossSinkParams = {
    // One row per branch of the sink's scale_s split, at a seqlen that keeps qr_tdm in
    // run_decode.
    {128, "n", 1024, 1024, gqa_kind::mha, 0.f},
    {128, "e", 1024, 1024, gqa_kind::mha, 0.f},
    {128, "a", 1024, 1024, gqa_kind::mha, 0.f},
    // seqlen >= 2048 drops the kM0=64 tile, so kM0=128 makes PrefillCase true
    // (fmha_fwd_kernel.hpp:2524) and qr_tdm runs run_prefill, which carries its own
    // copy of the sink handling. Every init_sink=1 row in kSinkWindowParams is seqlen
    // 1024, so that copy has never run with a finite sink.
    {128, "n", 2048, 2048, gqa_kind::mha, 0.f},
    {128, "e", 2048, 2048, gqa_kind::mha, 0.f},
    // hdim 256 has no qr_tdm instance, so this keeps qr/qr_async covered.
    {256, "n", 1024, 1024, gqa_kind::mha, 0.f},
    // The kernel reads sink_ptr[i_nhead]. Pack-GQA folds the Q heads of a group into
    // seqlen and sets nhead_q = nhead_k, so i_nhead becomes a K head while the sink is
    // one value per logical Q head. scale_s is pinned small deliberately: at the
    // default 1/sqrt(d) the other columns spread out, the sink's share of the row mass
    // falls under the 1e-4 LSE check, and the mismatch is invisible.
    //
    // The same small scale_s makes these two the fp8 regression for the sink owning the P
    // quantization frame. Let m sit on the sink and every column carries the same
    // exp(s - sink) != 1 rather than an exactly representable 1.0; gemm_1 multiplies those
    // rounded weights while l sums the unrounded ones, so the shared factor does not divide
    // out and lands on OUT as a per-head gain.
    {128, "n", 1024, 1024, gqa_kind::gqa, 3e-6f},
    {128, "n", 1024, 1024, gqa_kind::gqa_packed, 3e-6f},
};

class GptossSink : public TestWithParam<std::tuple<mode_enum, GptossSinkParam>>
{
};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         GptossSink,
                         Combine(ModeValues, ValuesIn(kGptossSinkParams)));

TEST_P(GptossSink, DataTypeConfig)
{
    auto [mode, sink_param]                                 = GetParam();
    auto [hdim, bias_str, seqlen_q, seqlen_k, gqa, scale_s] = sink_param;

    const int nhead_k  = (gqa == gqa_kind::mha ? -1 : 1);
    const int pack_gqa = (gqa == gqa_kind::gqa_packed ? 1 : 0);

    auto result = fmha_fwd_run<DataTypeConfig>(
        mode,
        2, // batch
        4, // nhead
        nhead_k,
        {adjust_seqlen(seqlen_q)},
        {adjust_seqlen(seqlen_k)},
        adjust_hdim(hdim),
        adjust_hdim(hdim),
        0,    // seqlen_knew
        {-1}, // seqlen_qpads
        {-1}, // seqlen_kpads
        {},   // q_eff_lens_per_batch
        {},   // kv_eff_lens_per_batch
        0,    // rotary_dim
        true, // i_perm
        true, // o_perm
        scale_s,
        0, // logits_soft_cap
        def_is_v_rowmajor,
        def_lse,
        0,        // page_block_size
        false,    // use_cache_batch_idx
        bias_str, // bias_str
        0.0f,     // p_drop
        0,        // drop_seed
        0,        // drop_offset
        false,    // drop_prefs
        "0",      // mask_str
        qscale_str,
        true, // is_rotary_interleaved
        1,    // num_splits
        init_method,
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1,        // do_validation
        1,        // init_sink_value
        pack_gqa, // pack_gqa
        stream_config);
    CHECK_RESULT(result);
}
#endif

// ============================================================================
// Host-only unit tests for fmha_batch_prefill_select_kv_load_mode() (in
// fmha_fwd.hpp). Guards ROCm/aiter#3824: when page_block_size < kN0 the paged-KV
// gather uses one SRD whose signed int32 voffset spans the whole K (or V) pool,
// so base[31:0] + pool_bytes silently wraps once it crosses INT32_MAX (~2GB) -
// even for sub-2GB pools placed high in the VA space.
// ============================================================================

namespace {

using ck_tile::BlockAttentionKVCacheLoadModeEnum;

constexpr auto kBuffer = BlockAttentionKVCacheLoadModeEnum::BUFFER_LOAD;
constexpr auto kGlobal = BlockAttentionKVCacheLoadModeEnum::GLOBAL_LOAD_LDS;

// The selector only reads the low 32 bits of the base VA; it never dereferences.
const void* as_ptr(std::uint64_t va) { return reinterpret_cast<const void*>(va); }

// Same low-32 base and single stride for both K and V.
BlockAttentionKVCacheLoadModeEnum select(ck_tile::index_t page_block_size,
                                         ck_tile::index_t kN0,
                                         ck_tile::index_t num_total_pages,
                                         ck_tile::index_t batch_stride,
                                         ck_tile::index_t element_bytes,
                                         std::uint64_t base_va)
{
    return fmha_batch_prefill_select_kv_load_mode(page_block_size,
                                                  kN0,
                                                  num_total_pages,
                                                  batch_stride,
                                                  batch_stride,
                                                  element_bytes,
                                                  as_ptr(base_va),
                                                  as_ptr(base_va));
}

} // namespace

// Fast path: page >= kN0 rebases the SRD per page, so BUFFER_LOAD is always safe
// regardless of base VA or pool size.
TEST(FmhaBatchPrefillKvLoadMode, PageGeKN0_AlwaysBufferLoad)
{
    const std::uint64_t high_base = 0xF000'0000ULL; // ~3.75GB low32
    EXPECT_EQ(select(256, 128, 1'000'000, 256, 2, high_base), kBuffer);
    EXPECT_EQ(select(128, 128, 1'000'000, 256, 2, high_base), kBuffer);
}

// Core boundary: base + pool == INT32_MAX stays BUFFER_LOAD (strictly-greater
// comparison); one byte more flips to GLOBAL_LOAD_LDS.
TEST(FmhaBatchPrefillKvLoadMode, ExactInt32MaxBoundary)
{
    // sum == INT32_MAX -> BUFFER_LOAD.
    EXPECT_EQ(select(32, 128, 1, INT32_MAX, 1, 0x0ULL), kBuffer);
    // sum == INT32_MAX + 1 -> GLOBAL_LOAD_LDS.
    EXPECT_EQ(select(32, 128, 1, INT32_MAX, 1, 0x1ULL), kGlobal);
}

// The #3824 regression: a sub-2GB pool that is safe at a low base faults when
// the allocator places it high in the VA space (the (2GB,4GB] danger band a
// naive 0xFFFFFFFF/4GB check would miss).
TEST(FmhaBatchPrefillKvLoadMode, SubTwoGiBPool_HighBaseTipsOverflow)
{
    const ck_tile::index_t pages = 500'000, stride = 256, ebytes = 2;
    const auto pool = static_cast<std::uint64_t>(pages) * stride * ebytes; // ~244MB < 2GB
    EXPECT_EQ(select(32, 128, pages, stride, ebytes, 0x0010'0000ULL), kBuffer);
    const std::uint64_t high_base = 0x8000'0000ULL - pool / 2; // sum lands above 2GB
    EXPECT_EQ(select(32, 128, pages, stride, ebytes, high_base), kGlobal);
}

// K and V are allocated independently: either one overflowing forces
// GLOBAL_LOAD_LDS; neither overflowing stays BUFFER_LOAD.
TEST(FmhaBatchPrefillKvLoadMode, PerTensorIndependence)
{
    const ck_tile::index_t pages = 500'000, ebytes = 2;
    const std::uint64_t high = 0x8000'0000ULL - 0x0800'0000ULL; // near 2GB
    const std::uint64_t low  = 0x0010'0000ULL;

    // Only K overflows.
    EXPECT_EQ(fmha_batch_prefill_select_kv_load_mode(
                  32, 128, pages, 512, 1, ebytes, as_ptr(high), as_ptr(low)),
              kGlobal);
    // Only V overflows.
    EXPECT_EQ(fmha_batch_prefill_select_kv_load_mode(
                  32, 128, pages, 1, 512, ebytes, as_ptr(low), as_ptr(high)),
              kGlobal);
    // Neither overflows.
    EXPECT_EQ(fmha_batch_prefill_select_kv_load_mode(
                  32, 128, pages, 1, 1, ebytes, as_ptr(low), as_ptr(low)),
              kBuffer);
}

// Scattered 1D paged KV (page_size=1 LINEAR + SGLANG_PAGE_TABLE_1D): the page
// table holds arbitrary physical-page indices into the whole KV pool, so
// num_total_pages does not bound the per-page voffset and the signed int32 SRD
// offset can wrap independently of base[31:0] + pool_bytes. Single-token pages
// (page_block_size == 1) must always take GLOBAL_LOAD_LDS, even when the
// address-overflow check alone would keep BUFFER_LOAD.
TEST(FmhaBatchPrefillKvLoadMode, SinglePageAlwaysGlobalLoad)
{
    // Low base, tiny in-bounds pool: the #9214 overflow check alone would pick
    // BUFFER_LOAD, but a single-token page must be routed to GLOBAL_LOAD_LDS.
    EXPECT_EQ(select(1, 128, 1'000, 64, 2, 0x0010'0000ULL), kGlobal);
    // A >1 sub-tile page with the same in-bounds address keeps the BUFFER_LOAD
    // fast path: the guard is scoped to page_block_size == 1 and does not regress
    // the cases #9214 already handles.
    EXPECT_EQ(select(32, 128, 1'000, 64, 2, 0x0010'0000ULL), kBuffer);
}
