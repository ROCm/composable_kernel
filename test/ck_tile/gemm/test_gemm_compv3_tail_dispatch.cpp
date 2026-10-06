// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp"
#include "gtest/gtest.h"

#include <array>
#include <utility>

namespace {

using ck_tile::TailNumber;
using Scenario = std::pair<bool, TailNumber>;

// Only the scheduling inputs are needed; no device allocation or kernel launch
// is involved in this test. The static assertions compile in both HIP passes.
template <ck_tile::index_t NumWarps_>
struct ScheduleProblem
{
    struct Traits
    {
        static constexpr bool UsePersistentKernel = false;
    };

    struct BlockGemmShape
    {
        static constexpr ck_tile::index_t NumWarps = NumWarps_;
    };
};

constexpr std::array<Scenario, 8> standard_scenarios = {
    Scenario{false, TailNumber::Odd},
    Scenario{false, TailNumber::Even},
    Scenario{true, TailNumber::Odd},
    Scenario{true, TailNumber::Odd},
    Scenario{true, TailNumber::Odd},
    Scenario{true, TailNumber::Odd},
    Scenario{true, TailNumber::Odd},
    Scenario{true, TailNumber::Odd},
};

[[maybe_unused]] constexpr std::array<Scenario, 8> mfma_eight_warp_scenarios = {
    Scenario{false, TailNumber::One},
    Scenario{false, TailNumber::Even},
    Scenario{false, TailNumber::Odd},
    Scenario{true, TailNumber::Even},
    Scenario{true, TailNumber::Odd},
    Scenario{true, TailNumber::Even},
    Scenario{true, TailNumber::Odd},
    Scenario{true, TailNumber::Even},
};

template <ck_tile::index_t NumWarps>
constexpr auto expected_scenarios()
{
#if CK_TILE_USE_WMMA
    return standard_scenarios;
#else
    if constexpr(NumWarps == 8)
        return mfma_eight_warp_scenarios;
    else
        return standard_scenarios;
#endif
}

template <ck_tile::index_t NumWarps>
constexpr bool check_schedule()
{
    using Schedule          = ck_tile::BaseGemmPipelineAgBgCrCompV3<ScheduleProblem<NumWarps>>;
    constexpr auto expected = expected_scenarios<NumWarps>();
    for(ck_tile::index_t loops = 1; loops <= 8; ++loops)
    {
        if(Schedule::BlockHasHotloop(loops) != expected[loops - 1].first ||
           Schedule::GetBlockLoopTailNum(loops) != expected[loops - 1].second)
            return false;
    }
    return true;
}

static_assert(check_schedule<2>(), "Two-warp CompV3 must use the standard schedule");
static_assert(check_schedule<4>(), "Four-warp CompV3 must use the standard schedule");
static_assert(check_schedule<8>(), "Host and device must use the selected MFMA/WMMA schedule");

struct ReturnScenario
{
    template <typename HotLoop, typename Tail>
    CK_TILE_HOST_DEVICE Scenario operator()(HotLoop, Tail) const
    {
#if CK_TILE_USE_WMMA
        // These assertions check the full set instantiated by TailHandler,
        // including branches not selected by a particular runtime loop count.
        static_assert(Tail::value != TailNumber::One,
                      "WMMA must not instantiate the MFMA one-loop specialization");
        static_assert(!HotLoop::value || Tail::value == TailNumber::Odd,
                      "WMMA must not instantiate a hot/Even specialization");
#endif
        return {HotLoop::value, Tail::value};
    }
};

template <ck_tile::index_t NumWarps>
void check_host_dispatch()
{
    using Schedule          = ck_tile::BaseGemmPipelineAgBgCrCompV3<ScheduleProblem<NumWarps>>;
    constexpr auto expected = expected_scenarios<NumWarps>();
    for(ck_tile::index_t loops = 1; loops <= 8; ++loops)
    {
        const auto actual = Schedule::TailHandler(ReturnScenario{},
                                                  Schedule::BlockHasHotloop(loops),
                                                  Schedule::GetBlockLoopTailNum(loops));
        EXPECT_EQ(actual, expected[loops - 1]) << "K tile count: " << loops;
    }
}

TEST(TestCkTileGemmCompV3TailDispatch, TwoWarps) { check_host_dispatch<2>(); }
TEST(TestCkTileGemmCompV3TailDispatch, FourWarps) { check_host_dispatch<4>(); }
TEST(TestCkTileGemmCompV3TailDispatch, EightWarps) { check_host_dispatch<8>(); }

} // namespace
