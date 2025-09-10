// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

using ::testing::Bool;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;

// Random seed used for initializing input tensors. 0 for non-deterministic seed
CK_TILE_DECLARE_ENV_VAR(CK_TILE_TEST_SEED, uint64_t, 123456)

// Whether to run long tests (from smoke_test_fwd.sh)
CK_TILE_DECLARE_ENV_VAR_BOOL(CK_TILE_FMHA_LONG_TESTS)

#define CHECK_RESULT(result)                                      \
    do                                                            \
    {                                                             \
        if(result == bwd_result::no_instance)                     \
            GTEST_SKIP() << "No instance for current parameters"; \
        ASSERT_EQ(result, bwd_result::success);                   \
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

#define COMMON_ARGS                                                                           \
    init_method, static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))), 1, \
        stream_config

auto EnableTestIf(bool condition)
{
    return ValuesIn(condition ? std::vector<bool>{true} : std::vector<bool>{});
}

class AllLong : public TestWithParam<std::tuple<bool,
                                                std::tuple<int, int>,
                                                bool,
                                                mode_enum,
                                                std::string,
                                                float,
                                                std::tuple<int, int, int, int, int, std::string>>>
{
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AllLong);

// Test cases from example/ck_tile/01_fmha/script/smoke_test_bwd.sh

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaBwd,
    AllLong,
    Combine(EnableTestIf(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_FMHA_LONG_TESTS))),
            HDimValues,
            Bool(),
            ModeValues,
            Values("n", "a"),
            Values(0.0f, 0.2f),
            Values(std::tuple{1, 4, 2, 259, -1, "0"},
                   std::tuple{2, 2, -1, 516, 253, "0"},
                   std::tuple{1, 4, 1, 500, 251, "1"},
                   std::tuple{1, 2, -1, 900, 258, "2"},
                   std::tuple{2, 1, -1, 987, 219, "t:128,30"},
                   std::tuple{2, 3, 1, 244, 499, "b:4,35"})));

TEST_P(AllLong, Test)
{
    auto [_, hdims, perm, mode, bias_str, p_drop, dims_mask]   = GetParam();
    auto [hdim_q, hdim_v]                                      = hdims;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str] = dims_mask;

    auto result = fmha_bwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {seqlen_q},
                                               {seqlen_k},
                                               hdim_q,
                                               hdim_v,
                                               perm,     // i_perm
                                               perm,     // o_perm
                                               0,        // scale
                                               bias_str, // bias_str
                                               false,    // use_dbias
                                               p_drop,   // p_drop
                                               123,      // drop_seed
                                               1024,     // drop_offset
                                               true,     // drop_prefs
                                               mask_str, // mask_str
                                               false,    // deterministic
                                               COMMON_ARGS);
    CHECK_RESULT(result);
}

class HDimPadding
    : public TestWithParam<std::tuple<std::tuple<int, int>,
                                      bool,
                                      mode_enum,
                                      std::tuple<int, int, int, int, int, std::string>>>
{
};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaBwd,
                         HDimPadding,
                         Combine(Values(std::tuple{24, 48},
                                        std::tuple{120, 160},
                                        std::tuple{256, 108},
                                        std::tuple{40, 64}),
                                 Bool(),
                                 ModeValues,
                                 Values(std::tuple{1, 4, 2, 480, -1, "0"},
                                        std::tuple{2, 2, -1, 300, 400, "t:64,64"},
                                        std::tuple{1, 4, 1, 512, 201, "1"},
                                        std::tuple{1, 2, -1, 900, 256, "0"},
                                        std::tuple{2, 1, -1, 256, 256, "1"})));

TEST_P(HDimPadding, Test)
{
    auto [hdims, perm, mode, dims_mask]                        = GetParam();
    auto [hdim_q, hdim_v]                                      = hdims;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str] = dims_mask;

    auto result = fmha_bwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {seqlen_q},
                                               {seqlen_k},
                                               hdim_q,
                                               hdim_v,
                                               perm,     // i_perm
                                               perm,     // o_perm
                                               0,        // scale
                                               "n",      // bias_str
                                               false,    // use_dbias
                                               0.0f,     // p_drop
                                               0,        // drop_seed
                                               0,        // drop_offset
                                               false,    // drop_prefs
                                               mask_str, // mask_str
                                               false,    // deterministic
                                               COMMON_ARGS);
    CHECK_RESULT(result);
}

class ElementwiseBias
    : public TestWithParam<std::tuple<std::tuple<int, int>,
                                      bool,
                                      mode_enum,
                                      std::string,
                                      bool,
                                      std::tuple<int, int, int, int, int, std::string>>>
{
};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaBwd,
                         ElementwiseBias,
                         Combine(HDimValues,
                                 Bool(), // layouts of bias and dbias are controlled by i_perm
                                 ModeValues,
                                 Values("e:0", "e:1", "e:2"),
                                 Bool(),
                                 Values(std::tuple{1, 4, 2, 1024, 100, "0"},
                                        std::tuple{3, 2, -1, 128, 256, "2"},
                                        std::tuple{2, 2, -1, 130, 499, "t:50,64"})));

TEST_P(ElementwiseBias, Test)
{
    auto [hdims, i_perm, mode, bias_str, use_dbias, dims_mask] = GetParam();
    auto [hdim_q, hdim_v]                                      = hdims;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str] = dims_mask;

    auto result = fmha_bwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {seqlen_q},
                                               {seqlen_k},
                                               hdim_q,
                                               hdim_v,
                                               i_perm,    // i_perm
                                               false,     // o_perm
                                               0,         // scale
                                               bias_str,  // bias_str
                                               use_dbias, // use_dbias
                                               0.0f,      // p_drop
                                               123,       // drop_seed
                                               1024,      // drop_offset
                                               true,      // drop_prefs
                                               mask_str,  // mask_str
                                               false,     // deterministic
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

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaBwd,
                         Alibi,
                         Combine(HDimValues,
                                 ModeValues,
                                 Values("a:0", "a:1"),
                                 Values(std::tuple{1, 3, 3, 1024, 1000},
                                        std::tuple{3, 5, 5, 128, 256},
                                        std::tuple{2, 8, 4, 130, 320}),
                                 Values("0", "t", "b", "t:50,64", "b:32,40")));

TEST_P(Alibi, Test)
{
    auto [hdims, mode, bias_str, dims, mask_str]     = GetParam();
    auto [hdim_q, hdim_v]                            = hdims;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k] = dims;

    auto result = fmha_bwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {seqlen_q},
                                               {seqlen_k},
                                               hdim_q,
                                               hdim_v,
                                               true,     // i_perm
                                               true,     // o_perm
                                               0,        // scale
                                               bias_str, // bias_str
                                               false,    // use_dbias
                                               0.0f,     // p_drop
                                               0,        // drop_seed
                                               0,        // drop_offset
                                               false,    // drop_prefs
                                               mask_str, // mask_str
                                               false,    // deterministic
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

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaBwd,
                         Dropout,
                         Combine(HDimValues,
                                 ModeValues,
                                 Values(0.123f, 0.5f),
                                 Values(std::tuple{10, 123, false},
                                        std::tuple{34534564645, 7876878876864, true}),
                                 Values(std::tuple{2, 6, 2, 180, 512, "0"},
                                        std::tuple{3, 2, 2, 256, 128, "1"},
                                        std::tuple{4, 2, 1, 100, 768, "2"})));

TEST_P(Dropout, Test)
{
    auto [hdims, mode, p_drop, drop_seed_offset_prefs, dims_mask] = GetParam();
    auto [hdim_q, hdim_v]                                         = hdims;
    auto [drop_seed, drop_offset, drop_prefs]                     = drop_seed_offset_prefs;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str]    = dims_mask;

    auto result = fmha_bwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {seqlen_q},
                                               {seqlen_k},
                                               hdim_q,
                                               hdim_v,
                                               true,        // i_perm
                                               true,        // o_perm
                                               0.1f,        // scale
                                               "n",         // bias_str
                                               false,       // use_dbias
                                               p_drop,      // p_drop
                                               drop_seed,   // drop_seed
                                               drop_offset, // drop_offset
                                               drop_prefs,  // drop_prefs
                                               mask_str,    // mask_str
                                               false,       // deterministic
                                               COMMON_ARGS);
    CHECK_RESULT(result);
}

class Deterministic
    : public TestWithParam<std::tuple<std::tuple<int, int>,
                                      bool,
                                      mode_enum,
                                      std::tuple<int, int, int, int, int, std::string>>>
{
};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaBwd,
                         Deterministic,
                         Combine(HDimValues,
                                 Bool(),
                                 ModeValues,
                                 Values(std::tuple{2, 6, 2, 180, 512, "0"},
                                        std::tuple{3, 3, 1, 256, 128, "1"},
                                        std::tuple{4, 2, 2, 768, 100, "2"})));

TEST_P(Deterministic, Test)
{
    auto [hdims, i_perm, mode, dims_mask]                      = GetParam();
    auto [hdim_q, hdim_v]                                      = hdims;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str] = dims_mask;

    auto result = fmha_bwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {seqlen_q},
                                               {seqlen_k},
                                               hdim_q,
                                               hdim_v,
                                               i_perm,   // i_perm
                                               true,     // o_perm
                                               0,        // scale
                                               "n",      // bias_str
                                               false,    // use_dbias
                                               0.0f,     // p_drop
                                               0,        // drop_seed
                                               0,        // drop_offset
                                               false,    // drop_prefs
                                               mask_str, // mask_str
                                               true,     // deterministic
                                               COMMON_ARGS);
    CHECK_RESULT(result);
}
