// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Shared setup of the comp_tdm_v3 gtests: GemmPipelineAgBgCrCompTDMV3 on gfx1250, fp8/bf8 A and B
// -> bf16, A row-major, B column-major, C row-major, warp tile 32x32x128.
//
// Instances: 16 waves (256x256x256, warps 4x4x1) and 4 waves (128x128x256, warps 2x2x1), each with
// the default policy (one split barrier per K tile) and with the two joint barriers per K tile
// (kUseSplitBarrier = false); plus, at 16 waves, the split barrier with fixed TDM issuers
// (kAlternateIssuers = false), with the TENSORcnt wait on the issuing waves only
// (kTensorWaitAllWaves = false) and with the refill issued right after the arrive
// (kRefillPoint = AfterArrive). All of these are fp8 x fp8; bf8 x bf8 runs at 16 and 4 waves and
// fp8 x bf8 and bf8 x fp8 at 16 waves, with the default policy.
//
// Split-K (k_batch != 1) must be rejected by IsSupportedArgument.
//
// Shapes: K = 256, 512 and 768 (one, two and three K tiles: both tail paths, with and without the
// hot loop) and 512x512x1024 in test_gemm_pipeline_comp_tdm_v3_wmma.cpp (SMOKE_TEST); K = 65536 (a
// long K loop with the real K stride) in test_gemm_pipeline_comp_tdm_v3_longk_wmma.cpp for the fp8
// instances and 16-wave bf8 x bf8 (REGRESSION_TEST: each long-K case runs for 7-40 s under the
// gfx1250 emulator).
//
// Inputs:
// - Int: integers in [-5, 5], exact in fp8 and bf8. Every product and every partial sum is an
// integer
//   below 2^24, so the fp32 accumulation is exact whatever the order, and the result must equal
//   the exact sum rounded to bf16, bit for bit.
// - Random: uniform reals in [-2, 2] rounded to the operand's type. Checked against a double
// reference with a
//   tolerance for bf16 rounding and fp32 accumulation.

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

// Policy variants. Outside the anonymous namespace: the host pass never reads the policy flags, and
// an internal-linkage constant would trip -Wunused-const-variable.
namespace comp_tdm_v3_test {
// One split barrier per K tile, refill at the top of the K tile (the default; set here so the
// variants below do not depend on it).
struct SplitBarrierPolicy : ck_tile::GemmPipelineAgBgCrCompTDMV3Policy
{
    static constexpr bool kUseSplitBarrier = true;
};
// Two joint barriers per K tile.
struct JointBarrierPolicy : ck_tile::GemmPipelineAgBgCrCompTDMV3Policy
{
    static constexpr bool kUseSplitBarrier = false;
};
// Split barrier; waves 0 (A) and 1 (B) fill both LDS stages.
struct SplitFixedIssuerPolicy : SplitBarrierPolicy
{
    static constexpr bool kAlternateIssuers = false;
};
// Split barrier; only the waves that issued the published fill wait for TENSORcnt 0. The resource
// test skips this kernel (TensorWaitIssuersOnly in its name): it spills on some compilers.
struct TensorWaitIssuersOnlyPolicy : SplitBarrierPolicy
{
    static constexpr bool kTensorWaitAllWaves = false;
};
// Split barrier; the issuers of the stage just read refill it right after their arrive. The
// resource test skips this kernel (RefillAfterArrive in its name): it spills on some compilers.
struct RefillAfterArrivePolicy : SplitBarrierPolicy
{
    static constexpr RefillPoint kRefillPoint = RefillPoint::AfterArrive;
};
} // namespace comp_tdm_v3_test

namespace {

using AccDataType = float;
using CDataType   = ck_tile::bf16_t;
using ALayout     = ck_tile::tensor_layout::gemm::RowMajor;
using BLayout     = ck_tile::tensor_layout::gemm::ColumnMajor;
using CLayout     = ck_tile::tensor_layout::gemm::RowMajor;

template <ck_tile::index_t TileM_,
          ck_tile::index_t TileN_,
          ck_tile::index_t WarpsM_,
          ck_tile::index_t WarpsN_,
          typename Policy_,
          typename ADataType_ = ck_tile::fp8_t,
          typename BDataType_ = ck_tile::fp8_t>
struct V3Config
{
    using ADataType                          = ADataType_;
    using BDataType                          = BDataType_;
    static constexpr ck_tile::index_t TileM  = TileM_;
    static constexpr ck_tile::index_t TileN  = TileN_;
    static constexpr ck_tile::index_t TileK  = 256;
    static constexpr ck_tile::index_t WarpsM = WarpsM_;
    static constexpr ck_tile::index_t WarpsN = WarpsN_;
    using Policy                             = Policy_;
};

template <typename Config>
struct V3KernelOf
{
    using ADataType = typename Config::ADataType;
    using BDataType = typename Config::BDataType;
    using TileShape =
        ck_tile::TileGemmShape<ck_tile::sequence<Config::TileM, Config::TileN, Config::TileK>,
                               ck_tile::sequence<Config::WarpsM, Config::WarpsN, 1>,
                               ck_tile::sequence<32, 32, 128>>;
    using Partitioner = ck_tile::GemmTile1DPartitioner<TileShape>;
    using Traits      = ck_tile::TileGemmUniversalTraits<false,
                                                         false,
                                                         false,
                                                         true, // DoubleSmemBuffer
                                                         ALayout,
                                                         BLayout,
                                                         CLayout>;
    using Problem =
        ck_tile::UniversalGemmPipelineProblem<ADataType,
                                              BDataType,
                                              AccDataType,
                                              TileShape,
                                              Traits,
                                              ck_tile::GemmPipelineScheduler::Intrawave>;
    using Pipeline        = ck_tile::GemmPipelineAgBgCrCompTDMV3<Problem, typename Config::Policy>;
    using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<ADataType,
                                                             BDataType,
                                                             ck_tile::tuple<>,
                                                             AccDataType,
                                                             CDataType,
                                                             ck_tile::tuple<>,
                                                             CLayout,
                                                             ck_tile::element_wise::PassThrough,
                                                             Config::TileM,
                                                             Config::TileN,
                                                             Config::WarpsM,
                                                             Config::WarpsN,
                                                             32,
                                                             32,
                                                             128,
                                                             false, // isCTransposed
                                                             1,     // kNumWaveGroups
                                                             false, // FixedVectorSize
                                                             1,     // VectorSizeC
                                                             1,     // BlockedXDLN_PerWarp
                                                             true>; // DoubleSmemBuffer
    using Epilogue        = ck_tile::CShuffleEpilogue<EpilogueProblem>;
    using type            = ck_tile::GemmKernel<Partitioner, Pipeline, Epilogue>;
};
template <typename Config>
using V3Kernel = typename V3KernelOf<Config>::type;

template <typename Config>
float run_v3_gemm(const ck_tile::GemmHostArgs& args)
{
    using Kernel = V3Kernel<Config>;

    auto kargs = Kernel::MakeKernelArgs(args);
    EXPECT_TRUE(Kernel::IsSupportedArgument(kargs));
    const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
    const dim3 blocks = Kernel::BlockSize();
    return ck_tile::launch_kernel(ck_tile::stream_config{nullptr, false},
                                  ck_tile::make_kernel<1>(Kernel{}, grids, blocks, 0, kargs));
}

uint16_t bf16_rne(float f)
{
    const uint32_t u = ck_tile::bit_cast<uint32_t>(f);
    return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

enum class Init
{
    Int,
    Random
};

template <typename Config>
class TestCkTileGemmPipelineCompTDMV3 : public ::testing::Test
{
    protected:
    void Run(ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K, Init init)
    {
        using ADataType = typename Config::ADataType;
        using BDataType = typename Config::BDataType;
        // A is M x K row-major, B is N x K (K x N column-major), both packed.
        std::mt19937 gen(0xC0FFEE + M + N + K + static_cast<int>(init));
        std::vector<ADataType> a(static_cast<size_t>(M) * K);
        std::vector<BDataType> b(static_cast<size_t>(N) * K);
        std::vector<float> af(a.size()), bf(b.size()); // the fp8/bf8 values, decoded
        auto fill = [&](auto& v, std::vector<float>& vf) {
            std::uniform_int_distribution<int> di(-5, 5);
            std::uniform_real_distribution<float> dr(-2.f, 2.f);
            for(size_t i = 0; i < v.size(); ++i)
            {
                const float x = init == Init::Int ? static_cast<float>(di(gen)) : dr(gen);
                v[i]  = ck_tile::type_convert<typename std::decay_t<decltype(v)>::value_type>(x);
                vf[i] = ck_tile::type_convert<float>(v[i]);
            }
        };
        fill(a, af);
        fill(b, bf);

        ck_tile::DeviceMem a_dev(a.size() * sizeof(ADataType));
        ck_tile::DeviceMem b_dev(b.size() * sizeof(BDataType));
        ck_tile::DeviceMem c_dev(static_cast<size_t>(M) * N * sizeof(CDataType));
        a_dev.ToDevice(a.data());
        b_dev.ToDevice(b.data());
        c_dev.SetZero();

        ck_tile::GemmHostArgs args(a_dev.GetDeviceBuffer(),
                                   b_dev.GetDeviceBuffer(),
                                   c_dev.GetDeviceBuffer(),
                                   1, // k_batch
                                   M,
                                   N,
                                   K,
                                   K,  // stride A
                                   K,  // stride B
                                   N); // stride C
        run_v3_gemm<Config>(args);

        std::vector<CDataType> c(static_cast<size_t>(M) * N);
        c_dev.FromDevice(c.data());

        size_t bad = 0;
        for(ck_tile::index_t m = 0; m < M; ++m)
        {
            for(ck_tile::index_t n = 0; n < N; ++n)
            {
                const float* pa = &af[static_cast<size_t>(m) * K];
                const float* pb = &bf[static_cast<size_t>(n) * K];
                double sum = 0, sum_abs = 0;
                for(ck_tile::index_t k = 0; k < K; ++k)
                {
                    const double p = static_cast<double>(pa[k]) * pb[k];
                    sum += p;
                    sum_abs += std::abs(p);
                }
                const CDataType got = c[static_cast<size_t>(m) * N + n];
                bool ok;
                if(init == Init::Int)
                {
                    // exact integer sum, rounded once to bf16 (round to nearest even)
                    ok = ck_tile::bit_cast<uint16_t>(got) == bf16_rne(static_cast<float>(sum));
                }
                else
                {
                    // bf16 rounding of the result plus fp32 rounding along the K sum
                    const double tol = std::abs(sum) * 0x1p-7 + sum_abs * 0x1p-16;
                    ok               = std::abs(ck_tile::type_convert<float>(got) - sum) <= tol;
                }
                if(!ok && bad++ < 5)
                {
                    ADD_FAILURE() << "M=" << M << " N=" << N << " K=" << K << " m=" << m
                                  << " n=" << n << " ref=" << sum
                                  << " got=" << ck_tile::type_convert<float>(got);
                }
            }
        }
        EXPECT_EQ(bad, 0u) << "mismatches at M=" << M << " N=" << N << " K=" << K;
    }

    // Host check only, no launch: k_batch = 1 is accepted and split-K is rejected. The CShuffle
    // epilogue has a split-K store path, so this rejection is the TDM check, not the epilogue's.
    void CheckSplitKRejected()
    {
        using Kernel = V3Kernel<Config>;
        for(ck_tile::index_t k_batch : {1, 2, 4})
        {
            const ck_tile::GemmHostArgs args(
                nullptr, nullptr, nullptr, k_batch, 512, 512, 1024, 1024, 1024, 512);
            EXPECT_EQ(Kernel::IsSupportedArgument(Kernel::MakeKernelArgs(args)), k_batch == 1)
                << "k_batch=" << k_batch;
        }
    }
};

using DefaultPolicy = ck_tile::GemmPipelineAgBgCrCompTDMV3Policy;

using V3Configs =
    ::testing::Types<V3Config<256, 256, 4, 4, ck_tile::GemmPipelineAgBgCrCompTDMV3Policy>,
                     V3Config<128, 128, 2, 2, ck_tile::GemmPipelineAgBgCrCompTDMV3Policy>,
                     V3Config<256, 256, 4, 4, comp_tdm_v3_test::JointBarrierPolicy>,
                     V3Config<128, 128, 2, 2, comp_tdm_v3_test::JointBarrierPolicy>,
                     V3Config<256, 256, 4, 4, comp_tdm_v3_test::SplitFixedIssuerPolicy>,
                     V3Config<256, 256, 4, 4, comp_tdm_v3_test::TensorWaitIssuersOnlyPolicy>,
                     V3Config<256, 256, 4, 4, comp_tdm_v3_test::RefillAfterArrivePolicy>,
                     V3Config<256, 256, 4, 4, DefaultPolicy, ck_tile::bf8_t, ck_tile::bf8_t>,
                     V3Config<128, 128, 2, 2, DefaultPolicy, ck_tile::bf8_t, ck_tile::bf8_t>,
                     V3Config<256, 256, 4, 4, DefaultPolicy, ck_tile::fp8_t, ck_tile::bf8_t>,
                     V3Config<256, 256, 4, 4, DefaultPolicy, ck_tile::bf8_t, ck_tile::fp8_t>>;

// The long-K set: the fp8 instances and 16-wave bf8 x bf8.
using V3LongKConfigs =
    ::testing::Types<V3Config<256, 256, 4, 4, DefaultPolicy>,
                     V3Config<128, 128, 2, 2, DefaultPolicy>,
                     V3Config<256, 256, 4, 4, comp_tdm_v3_test::JointBarrierPolicy>,
                     V3Config<128, 128, 2, 2, comp_tdm_v3_test::JointBarrierPolicy>,
                     V3Config<256, 256, 4, 4, comp_tdm_v3_test::SplitFixedIssuerPolicy>,
                     V3Config<256, 256, 4, 4, comp_tdm_v3_test::TensorWaitIssuersOnlyPolicy>,
                     V3Config<256, 256, 4, 4, comp_tdm_v3_test::RefillAfterArrivePolicy>,
                     V3Config<256, 256, 4, 4, DefaultPolicy, ck_tile::bf8_t, ck_tile::bf8_t>>;

} // namespace
