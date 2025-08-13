// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <tuple>

using ::testing::Bool;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;

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

const std::vector all_hdims{32, 64, 128, 256};

class TestCkTileFmhaBwd
    : public TestWithParam<std::tuple<bool,
                                      int,
                                      mode_enum,
                                      std::string,
                                      float,
                                      std::tuple<int, int, int, int, int, std::string>>>
{
};

// Test cases from example/ck_tile/01_fmha/script/smoke_test_bwd.sh

INSTANTIATE_TEST_SUITE_P(,
                         TestCkTileFmhaBwd,
                         Combine(Bool(),
                                 ValuesIn(all_hdims),
                                 Values(mode_enum::batch, mode_enum::group),
                                 Values("n", "a"),
                                 Values(0.0f, 0.2f),
                                 Values(std::tuple{1, 4, 2, 259, -1, std::string("0")},
                                        std::tuple{2, 2, -1, 516, 253, std::string("0")},
                                        std::tuple{1, 4, 1, 500, 251, std::string("1")},
                                        std::tuple{1, 2, -1, 900, 258, std::string("2")},
                                        std::tuple{2, 1, -1, 987, 219, std::string("t:128,30")},
                                        std::tuple{2, 3, 1, 244, 499, std::string("b:4,35")})));

TEST_P(TestCkTileFmhaBwd, All)
{
    auto [perm, hdim, mode, bias_str, p_drop, c]               = GetParam();
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str] = c;

    auto bias = bias_info::decode(bias_str);
    auto mask = mask_info::decode(mask_str, seqlen_q, seqlen_k);

    auto result = fmha_bwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               seqlen_q,
                                               seqlen_k,
                                               hdim,           // hdim_q
                                               hdim,           // hdim_v
                                               perm,           // i_perm
                                               perm,           // o_perm
                                               0,              // scale
                                               bias,           // bias
                                               false,          // use_dbias
                                               p_drop,         // p_drop
                                               123,            // drop_seed
                                               1024,           // drop_offset
                                               true,           // drop_prefs
                                               mask,           // mask
                                               false,          // deterministic
                                               "uf",           // init_method
                                               67889,          // seed
                                               1,              // do_validation
                                               stream_config); // stream_config
    ASSERT_TRUE(result);
}
