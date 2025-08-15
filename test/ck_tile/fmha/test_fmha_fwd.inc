// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "example/ck_tile/01_fmha/fmha_fwd.hpp"
#include "example/ck_tile/01_fmha/fmha_fwd_runner.hpp"

#include "gtest/gtest.h"

#include <tuple>

using ::testing::Bool;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;

CK_TILE_DECLARE_ENV_VAR_BOOL(CK_TILE_FMHA_RUN_LONG_TESTS)

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

const auto AllHDims = Values(32, 64, 128, 256);

class AllLong : public TestWithParam<
                    std::tuple<int,
                               bool,
                               bool,
                               mode_enum,
                               bool,
                               std::string,
                               float,
                               std::tuple<int, int, int, int, int, int, int, int, std::string>>>
{
    protected:
    void SetUp() override
    {
        if(!ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_FMHA_RUN_LONG_TESTS)))
        {
            GTEST_SKIP() << "CK_TILE_FMHA_RUN_LONG_TESTS env variable is not set";
        }
    }
};

// Test cases from example/ck_tile/01_fmha/script/smoke_test_fwd.sh

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaFwd,
                         AllLong,
                         Combine(AllHDims,
                                 Bool(),
                                 Bool(),
                                 Values(mode_enum::batch, mode_enum::group),
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
                                        std::tuple{1, 2, 1, -1, -1, 1, 10, 32, "2"})));

TEST_P(AllLong, Test)
{
    auto [hdim, perm, is_v_rowmajor, mode, lse, bias_str, p_drop, dims_mask] = GetParam();
    auto [batch, nhead, nhead_k, hdim_q, hdim_v, seqlen_q, seqlen_k, seqlen_kpad, mask_str] =
        dims_mask;

    hdim_q = hdim_q == -1 ? hdim : hdim_q;
    hdim_v = hdim_v == -1 ? hdim : hdim_v;

    auto bias = bias_info::decode(bias_str);

    constexpr bool squant = std::is_same_v<DataTypeConfig, FmhaFwdFp8>;

    auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {seqlen_q},            // seqlen_qs
                                               {seqlen_k},            // seqlen_ks
                                               hdim_q,                // hdim_q
                                               hdim_v,                // hdim_v
                                               0,                     // seqlen_knew
                                               {seqlen_kpad},         // seqlen_kpads
                                               0,                     // rotary_dim
                                               perm,                  // i_perm
                                               perm,                  // o_perm
                                               0,                     // scale_s
                                               0,                     // logits_soft_cap
                                               is_v_rowmajor,         // is_v_rowmajor
                                               lse,                   // lse
                                               0,                     // page_block_size
                                               false,                 // use_cache_batch_idx
                                               bias,                  // bias
                                               p_drop,                // p_drop
                                               123,                   // drop_seed
                                               1024,                  // drop_offset
                                               false,                 // drop_prefs
                                               mask_str,              // mask_str
                                               16,                    // range_q
                                               16,                    // range_k
                                               16,                    // range_v
                                               1,                     // range_p
                                               16,                    // range_o
                                               squant,                // squant
                                               true,                  // is_rotary_interleaved
                                               1,                     // num_splits
                                               squant ? "ufq" : "uf", // init_method
                                               5675,                  // seed
                                               1,                     // do_validation
                                               stream_config);        // stream_config)
    CHECK_RESULT(result);
}
