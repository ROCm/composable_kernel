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

// range_q, range_k, range_v, range_p, range_o, squant
#define QUANT_ARGS 1, 1, 1, 1, 1, squant

#define COMMON_ARGS                                                                           \
    init_method, static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))), 1, \
        stream_config

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
                   std::tuple{1, 2, 1, -1, -1, 1, 10, 32, "2"})));

TEST_P(AllLong, Test)
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
                                               {seqlen_kpad}, // seqlen_kpads
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
                                               QUANT_ARGS,
                                               true, // is_rotary_interleaved
                                               1,    // num_splits
                                               COMMON_ARGS);
    CHECK_RESULT(result);
}

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

TEST_P(HDimPadding, Test)
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
                                               {seqlen_kpad}, // seqlen_kpads
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
                                               QUANT_ARGS,
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

TEST_P(ElementwiseBias, Test)
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
                                               {-1},              // seqlen_kpads
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
                                               QUANT_ARGS,
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

TEST_P(Alibi, Test)
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
                                               {-1},              // seqlen_kpads
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
                                               QUANT_ARGS,
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

TEST_P(Dropout, Test)
{
    auto [hdims, mode, p_drop, drop_seed_offset_prefs, dims_mask] = GetParam();
    auto [hdim_q, hdim_v]                                         = hdims;
    auto [drop_seed, drop_offset, drop_prefs]                     = drop_seed_offset_prefs;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str]    = dims_mask;

    auto result = fmha_fwd_run<DataTypeConfig>(mode,
                                               batch,
                                               nhead,
                                               nhead_k,
                                               {adjust_seqlen(seqlen_q)},
                                               {adjust_seqlen(seqlen_k)},
                                               hdim_q,
                                               hdim_v,
                                               0,                 // seqlen_knew
                                               {-1},              // seqlen_kpads
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
                                               QUANT_ARGS,
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

TEST_P(PagedKV, Test)
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
                                               {-1},            // seqlen_kpads
                                               0,               // rotary_dim
                                               i_perm,          // i_perm
                                               false,           // o_perm
                                               0,               // scale_s
                                               0,               // logits_soft_cap
                                               is_v_rowmajor,   // is_v_rowmajor
                                               def_lse,         // lse
                                               page_block_size, // page_block_size
                                               false,           // use_cache_batch_idx
                                               "n",             // bias_str
                                               0.0f,            // p_drop
                                               0,               // drop_seed
                                               0,               // drop_offset
                                               false,           // drop_prefs
                                               mask_str,        // mask_str
                                               QUANT_ARGS,
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
                                        std::tuple{3, 2, -1, 230, 899, "t:128,128"})));

TEST_P(SplitKV, Test)
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
                                               {-1},                // seqlen_kpads
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
                                               QUANT_ARGS,
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

TEST_P(AppendKV, Test)
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
                                               {-1},                // seqlen_kpads
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
                                               QUANT_ARGS,
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
                         Combine(EnableTestIf(!std::is_same_v<DataTypeConfig, FmhaFwdFp8>),
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

TEST_P(AppendKVRoPE, Test)
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
                                               {-1},          // seqlen_kpads
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
                                               QUANT_ARGS,
                                               is_rotary_interleaved, // is_rotary_interleaved
                                               1,                     // num_splits
                                               COMMON_ARGS);
    CHECK_RESULT(result);
}

#endif // CK_TILE_FMHA_FWD_APPENDKV_API
