// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "example/ck_tile/01_fmha/fmha_bwd.hpp"
#include "example/ck_tile/01_fmha/fmha_bwd_runner.hpp"

#include "gtest/gtest.h"

#ifndef DataTypeConfig
#define DataTypeConfig FmhaBwdFp16 // or FmhaBwdBf16 / FmhaBwdFp32
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
        std::tuple{32, -1}, std::tuple{64, -1}, std::tuple{128, -1}, std::tuple{256, -1}};
};
template <>
struct TestConfigs<FmhaBwdFp32>
{
    static constexpr auto HDimValues =
        std::array{std::tuple{32, -1}, std::tuple{64, -1}, std::tuple{128, -1}};
};
static auto HDimValues     = ValuesIn(TestConfigs<DataTypeConfig>::HDimValues);
const auto ModeValues      = ValuesIn(std::vector<mode_enum>{mode_enum::batch, mode_enum::group});
constexpr auto init_method = "uf";

// Random seed used for initializing input tensors. 0 for non-deterministic seed
CK_TILE_DECLARE_ENV_VAR(CK_TILE_TEST_SEED, uint64_t, 123456)

// Whether to run long tests (from smoke_test_fwd.sh)
CK_TILE_DECLARE_ENV_VAR_BOOL(CK_TILE_FMHA_LONG_TESTS)

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

// batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str
using FmhaBwdDimsMaskParam = std::tuple<int, int, int, int, int, std::string>;
using FmhaBwdTestParam     = std::tuple<      //
    mode_enum,                            // mode
    std::tuple<int, int>,                 // hdim_q, hdim_v
    std::tuple<bool, bool>,               // io_perm
    std::string,                          // bias_str
    bool,                                 // use_dbias
    float,                                // p_drop
    std::tuple<uint64_t, uint64_t, bool>, // drop_seed, drop_offset, drop_prefs
    FmhaBwdDimsMaskParam,
    bool // deterministic
    >;
void fmha_bwd_test(const FmhaBwdTestParam& param)
{
    auto [mode, hdims, perm, bias_str, use_dbias, p_drop, drop_misc, dims_mask, det] = param;
    auto [hdim_q, hdim_v]                                                            = hdims;
    auto [i_perm, o_perm]                                                            = perm;
    auto [drop_seed, drop_offset, drop_prefs]                                        = drop_misc;
    auto [batch, nhead, nhead_k, seqlen_q, seqlen_k, mask_str]                       = dims_mask;

    auto result = fmha_bwd_run<DataTypeConfig>(
        mode,
        batch,
        nhead,
        nhead_k,
        {seqlen_q},
        {seqlen_k},
        hdim_q,
        hdim_v,
        i_perm,
        o_perm,
        0, // scale
        bias_str,
        use_dbias,
        p_drop,
        drop_seed,
        drop_offset,
        drop_prefs,
        mask_str,
        det, // deterministic
        init_method,
        static_cast<uint32_t>(ck_tile::EnvValue(CK_TILE_ENV(CK_TILE_TEST_SEED))),
        1,
        stream_config);

    if(result == bwd_result::no_instance)
        GTEST_SKIP() << "No instance for current parameters";
    ASSERT_EQ(result, bwd_result::success);
}

// Test cases from example/ck_tile/01_fmha/script/smoke_test_bwd.sh
class AllLong : public TestWithParam<FmhaBwdTestParam>
{
};
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AllLong);
INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaBwd,
                         AllLong,
                         Combine(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_FMHA_LONG_TESTS))
                                     ? ModeValues
                                     : ValuesIn(std::vector<mode_enum>{}),
                                 HDimValues,
                                 Values(std::tuple{true, true}, std::tuple{false, false}), // perm
                                 Values("n", "a"),
                                 Values(false),                       // use_dbias
                                 Values(0.0f, 0.2f),                  // p_drop
                                 Values(std::tuple{123, 1024, true}), // seed/offset/prefs
                                 Values(std::tuple{1, 4, 2, 259, -1, "0"},
                                        std::tuple{2, 2, -1, 516, 253, "0"},
                                        std::tuple{1, 4, 1, 500, 251, "1"},
                                        std::tuple{1, 2, -1, 900, 258, "2"},
                                        std::tuple{2, 1, -1, 987, 219, "t:128,30"},
                                        std::tuple{2, 3, 1, 244, 499, "b:4,35"}),
                                 Values(false) // deterministic
                                 ));
TEST_P(AllLong, DataTypeConfig) { fmha_bwd_test(GetParam()); }

class HDimPadding : public TestWithParam<FmhaBwdTestParam>
{
};
INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaBwd,
                         HDimPadding,
                         Combine(ModeValues,
                                 Values(std::tuple{24, 48},
                                        std::tuple{48, 48},
                                        std::tuple{72, 72},
                                        std::tuple{96, 96},
                                        std::tuple{120, 160},
                                        std::tuple{256, 108},
                                        std::tuple{40, 64}),
                                 Values(std::tuple{true, true}, std::tuple{false, false}), // perm
                                 Values("n"),                     // bias_str
                                 Values(false),                   // use_dbias
                                 Values(0.0f),                    // p_drop
                                 Values(std::tuple{0, 0, false}), // seed/offset/prefs
                                 Values(std::tuple{1, 4, 2, 480, -1, "0"},
                                        std::tuple{2, 2, -1, 300, 400, "t:64,64"},
                                        std::tuple{1, 4, 1, 512, 201, "1"},
                                        std::tuple{1, 2, -1, 900, 256, "0"},
                                        std::tuple{2, 1, -1, 256, 256, "1"}),
                                 Values(false) // deterministic
                                 ));
TEST_P(HDimPadding, DataTypeConfig) { fmha_bwd_test(GetParam()); }

class ElementwiseBias : public TestWithParam<FmhaBwdTestParam>
{
};
INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaBwd,
                         ElementwiseBias,
                         Combine(ModeValues,
                                 HDimValues,
                                 // layouts of bias and dbias are controlled by i_perm
                                 Values(std::tuple{true, false}, std::tuple{false, false}),
                                 Values("e:0", "e:1", "e:2"),
                                 Bool(),                          // use_dbias
                                 Values(0.0f),                    // p_drop
                                 Values(std::tuple{0, 0, false}), // seed/offset/prefs
                                 Values(std::tuple{1, 4, 2, 1024, 100, "0"},
                                        std::tuple{3, 2, -1, 128, 256, "2"},
                                        std::tuple{2, 2, -1, 130, 499, "t:50,64"}),
                                 Values(false) // deterministic
                                 ));
TEST_P(ElementwiseBias, DataTypeConfig) { fmha_bwd_test(GetParam()); }
class Alibi : public TestWithParam<FmhaBwdTestParam>
{
};

INSTANTIATE_TEST_SUITE_P(
    TestCkTileFmhaBwd,
    Alibi,
    Combine(ModeValues,
            HDimValues,
            Values(std::tuple{true, true}), // perm
            Values("a:0", "a:1"),
            Values(false),                   // use_dbias
            Values(0.0f),                    // p_drop
            Values(std::tuple{0, 0, false}), // seed/offset/prefs
            ValuesIn([]() {
                const std::array dims{
                    std::tuple{1, 3, 3, 1024, 1000},
                    std::tuple{3, 5, 5, 128, 256},
                    std::tuple{2, 8, 4, 130, 320},
                };
                const std::array mask_strs{"0", "t", "b", "t:50,64", "b:32,40"};
                std::vector<FmhaBwdDimsMaskParam> dims_masks;
                std::for_each(dims.begin(), dims.end(), [&](const auto& d) {
                    const auto& [b, h, hk, sq, sk] = d;
                    std::for_each(mask_strs.begin(), mask_strs.end(), [&](const auto& m) {
                        dims_masks.push_back(std::tuple{b, h, hk, sq, sk, m});
                    });
                });
                return dims_masks;
            }()),
            Values(false) // deterministic
            ));
TEST_P(Alibi, DataTypeConfig) { fmha_bwd_test(GetParam()); }

class Dropout : public TestWithParam<FmhaBwdTestParam>
{
};
INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaBwd,
                         Dropout,
                         Combine(ModeValues,
                                 HDimValues,
                                 Values(std::tuple{true, true}),    // perm
                                 Values("n"),                       // bias_str
                                 Values(false),                     // use_dbias
                                 Values(0.123f, 0.5f),              // p_drop
                                 Values(std::tuple{10, 123, false}, // seed/offset/prefs
                                        std::tuple{34534564645, 7876878876864, true}),
                                 Values(std::tuple{2, 6, 2, 180, 512, "0"},
                                        std::tuple{3, 2, 2, 256, 128, "1"},
                                        std::tuple{4, 2, 1, 100, 768, "2"}),
                                 Values(false) // deterministic
                                 ));

TEST_P(Dropout, DataTypeConfig) { fmha_bwd_test(GetParam()); }

class Deterministic : public TestWithParam<FmhaBwdTestParam>
{
};

INSTANTIATE_TEST_SUITE_P(TestCkTileFmhaBwd,
                         Deterministic,
                         Combine(ModeValues,
                                 HDimValues,
                                 Values(std::tuple{false, true}, std::tuple{true, true}), // perm
                                 Values("n"),                     // bias_str
                                 Values(false),                   // use_dbias
                                 Values(0.0f),                    // p_drop
                                 Values(std::tuple{0, 0, false}), // seed/offset/prefs
                                 Values(std::tuple{2, 6, 2, 180, 512, "0"},
                                        std::tuple{3, 3, 1, 256, 128, "1"},
                                        std::tuple{4, 2, 2, 768, 100, "2"}),
                                 Values(true) // deterministic
                                 ));
TEST_P(Deterministic, DataTypeConfig) { fmha_bwd_test(GetParam()); }
