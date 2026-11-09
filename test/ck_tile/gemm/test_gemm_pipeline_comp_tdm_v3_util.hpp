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
// bf16 x bf16 -> bf16 instances (V3Bf16Configs, V3Bf16LargeConfigs) use the warp tile 16x16x32 and
// K tiles of 64 and 128, in every A/B layout (RCR, RRR, CRR, CCR); their shapes are one, two and
// three K tiles and 512x512x1024 in test_gemm_pipeline_comp_tdm_v3_bf16_wmma.cpp (SMOKE_TEST) and
// 4096x4096x4096 in test_gemm_pipeline_comp_tdm_v3_bf16_large_wmma.cpp (REGRESSION_TEST).
//
// Split-K (k_batch != 1): the CShuffle epilogue adds each K split into C atomically, so it is
// accepted when the epilogue has a split-K store path (a C vector size of at least 2 for bf16) and
// k_batch does not exceed the warp-tile K steps; the TdmEpilogue overwrites C and is rejected.
//
// Shapes: K = 256, 512 and 768 (one, two and three K tiles: both tail paths, with and without the
// hot loop) and 512x512x1024 in test_gemm_pipeline_comp_tdm_v3_wmma.cpp (SMOKE_TEST); K = 65536 (a
// long K loop with the real K stride) in test_gemm_pipeline_comp_tdm_v3_longk_wmma.cpp for the fp8
// instances and 16-wave bf8 x bf8 (REGRESSION_TEST: each long-K case runs for 7-40 s under the
// gfx1250 emulator).
//
// Inputs:
// - Int: integers in [-5, 5], exact in fp8, bf8 and bf16. Every product and every partial sum is an
// integer
//   below 2^24, so the fp32 accumulation is exact whatever the order, and the result must equal
//   the exact sum rounded to bf16, bit for bit.
// - Random: uniform reals in [-2, 2] rounded to the operand's type. Checked against a double
// reference with a
//   tolerance for bf16 rounding and fp32 accumulation.
// - Sparse: each element nonzero with probability sqrt(64 / K), then one of -2, -1, 1 and 2, so
//   sum|a*b| stays below 256 at almost every output. There every partial and running sum is an
//   integer that bf16 holds exactly, so the split-K result (each split's bf16 partial added into C
//   atomically, in any order) must still be exact; the rare output above 256 uses the split-K
//   tolerance.
// With k_batch > 1, Int and Random use |got - ref| <= |ref| * 2^-7 + k_batch * sum|a*b| * 2^-7:
// every split rounds its partial sum to bf16 and every atomic add rounds the running sum.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <thread>
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
using RowMajor    = ck_tile::tensor_layout::gemm::RowMajor;
using ColumnMajor = ck_tile::tensor_layout::gemm::ColumnMajor;
using CLayout     = RowMajor;

// The defaults after the data types are the fp8 setup: K tile 256, warp tile 32x32x128, A
// row-major, B column-major. The 16-bit instances set them (warp tile 16x16x32, the only bf16/fp16
// WMMA warp tile on gfx1250, and a K tile of 64 or 128 so two LDS stages fit).
template <ck_tile::index_t TileM_,
          ck_tile::index_t TileN_,
          ck_tile::index_t WarpsM_,
          ck_tile::index_t WarpsN_,
          typename Policy_,
          typename ADataType_         = ck_tile::fp8_t,
          typename BDataType_         = ck_tile::fp8_t,
          ck_tile::index_t TileK_     = 256,
          ck_tile::index_t WarpTileM_ = 32,
          ck_tile::index_t WarpTileN_ = 32,
          ck_tile::index_t WarpTileK_ = 128,
          typename ALayout_           = RowMajor,
          typename BLayout_           = ColumnMajor,
          bool PadMN_                 = false,
          ck_tile::index_t VectorC_   = 0,
          bool TdmEpilogue_           = false>
struct V3Config
{
    using ADataType                             = ADataType_;
    using BDataType                             = BDataType_;
    using ALayout                               = ALayout_;
    using BLayout                               = BLayout_;
    static constexpr ck_tile::index_t TileM     = TileM_;
    static constexpr ck_tile::index_t TileN     = TileN_;
    static constexpr ck_tile::index_t TileK     = TileK_;
    static constexpr ck_tile::index_t WarpsM    = WarpsM_;
    static constexpr ck_tile::index_t WarpsN    = WarpsN_;
    static constexpr ck_tile::index_t WarpTileM = WarpTileM_;
    static constexpr ck_tile::index_t WarpTileN = WarpTileN_;
    static constexpr ck_tile::index_t WarpTileK = WarpTileK_;
    // kPadM/kPadN of the pipeline traits: TDM ignores them for A/B (it clips in hardware), but
    // the CShuffle epilogue needs kPadN (kPadM) to store C when N (M) is not a multiple of the
    // tile.
    static constexpr bool PadMN = PadMN_;
    // C store vector size; 0 lets the epilogue choose it (8 bf16 elements).
    static constexpr ck_tile::index_t VectorC = VectorC_;
    // TdmEpilogue instead of CShuffleEpilogue: C is staged in LDS and written with one TDM
    // store, which clips to the real M x N in hardware.
    static constexpr bool TdmEpilogue = TdmEpilogue_;
    using Policy                      = Policy_;
};

// bf16 instance: warp tile 16x16x32 and the given K tile and A/B layouts.
template <ck_tile::index_t TileM,
          ck_tile::index_t TileN,
          ck_tile::index_t TileK,
          ck_tile::index_t WarpsM,
          ck_tile::index_t WarpsN,
          typename ALayout,
          typename BLayout,
          bool PadMN               = false,
          ck_tile::index_t VectorC = 0,
          bool TdmEpilogue         = false,
          typename Policy          = ck_tile::GemmPipelineAgBgCrCompTDMV3Policy>
using V3Bf16Config = V3Config<TileM,
                              TileN,
                              WarpsM,
                              WarpsN,
                              Policy,
                              ck_tile::bf16_t,
                              ck_tile::bf16_t,
                              TileK,
                              16,
                              16,
                              32,
                              ALayout,
                              BLayout,
                              PadMN,
                              VectorC,
                              TdmEpilogue>;

template <typename Config>
struct V3KernelOf
{
    using ADataType = typename Config::ADataType;
    using BDataType = typename Config::BDataType;
    using TileShape = ck_tile::TileGemmShape<
        ck_tile::sequence<Config::TileM, Config::TileN, Config::TileK>,
        ck_tile::sequence<Config::WarpsM, Config::WarpsN, 1>,
        ck_tile::sequence<Config::WarpTileM, Config::WarpTileN, Config::WarpTileK>>;
    using Partitioner = ck_tile::GemmTile1DPartitioner<TileShape>;
    using Traits      = ck_tile::TileGemmUniversalTraits<Config::PadMN,
                                                         Config::PadMN,
                                                         false,
                                                         true, // DoubleSmemBuffer
                                                         typename Config::ALayout,
                                                         typename Config::BLayout,
                                                         CLayout>;
    using Problem =
        ck_tile::UniversalGemmPipelineProblem<ADataType,
                                              BDataType,
                                              AccDataType,
                                              TileShape,
                                              Traits,
                                              ck_tile::GemmPipelineScheduler::Intrawave>;
    using Pipeline = ck_tile::GemmPipelineAgBgCrCompTDMV3<Problem, typename Config::Policy>;
    using EpilogueProblem =
        ck_tile::CShuffleEpilogueProblem<ADataType,
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
                                         Config::WarpTileM,
                                         Config::WarpTileN,
                                         Config::WarpTileK,
                                         false, // isCTransposed
                                         1,     // kNumWaveGroups
                                         (Config::VectorC != 0),
                                         (Config::VectorC != 0 ? Config::VectorC : 1),
                                         1,     // BlockedXDLN_PerWarp
                                         true>; // DoubleSmemBuffer
    using Epilogue = std::conditional_t<Config::TdmEpilogue,
                                        ck_tile::TdmEpilogue<EpilogueProblem>,
                                        ck_tile::CShuffleEpilogue<EpilogueProblem>>;
    using type     = ck_tile::GemmKernel<Partitioner, Pipeline, Epilogue>;
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
    Random,
    Sparse
};

template <typename Config>
class TestCkTileGemmPipelineCompTDMV3 : public ::testing::Test
{
    protected:
    void Run(ck_tile::index_t M,
             ck_tile::index_t N,
             ck_tile::index_t K,
             Init init,
             ck_tile::index_t k_batch = 1)
    {
        using ADataType            = typename Config::ADataType;
        using BDataType            = typename Config::BDataType;
        constexpr bool a_row_major = std::is_same_v<typename Config::ALayout, RowMajor>;
        constexpr bool b_col_major = std::is_same_v<typename Config::BLayout, ColumnMajor>;
        // af is A as M x K and bf is B as N x K, both row-major, whatever the device layouts: the
        // values are drawn in that order, then packed into a and b in A's and B's layouts.
        std::mt19937 gen(0xC0FFEE + M + N + K + static_cast<int>(init) + 7 * k_batch);
        const double density = std::min(1.0, std::sqrt(64.0 / static_cast<double>(K)));
        std::vector<ADataType> a(static_cast<size_t>(M) * K);
        std::vector<BDataType> b(static_cast<size_t>(N) * K);
        std::vector<float> af(a.size()), bf(b.size()); // the A/B values, decoded
        auto fill = [&](auto& v, std::vector<float>& vf, ck_tile::index_t rows, bool packed_rows) {
            std::uniform_int_distribution<int> di(-5, 5);
            std::uniform_real_distribution<float> dr(-2.f, 2.f);
            std::uniform_real_distribution<double> du(0.0, 1.0);
            std::uniform_int_distribution<int> ds(0, 3);
            static constexpr float kSparse[4] = {-2.f, -1.f, 1.f, 2.f};
            using T                           = typename std::decay_t<decltype(v)>::value_type;
            for(size_t i = 0; i < v.size(); ++i)
            {
                const float x = init == Init::Int ? static_cast<float>(di(gen))
                                : init == Init::Sparse
                                    ? (du(gen) < density ? kSparse[ds(gen)] : 0.f)
                                    : dr(gen);
                const T e     = ck_tile::type_convert<T>(x);
                vf[i]         = ck_tile::type_convert<float>(e);
                // row r = i / K, k = i % K; the transposed layout stores it at k * rows + r
                const size_t r = i / static_cast<size_t>(K), k = i % static_cast<size_t>(K);
                v[packed_rows ? i : k * static_cast<size_t>(rows) + r] = e;
            }
        };
        fill(a, af, M, a_row_major);
        fill(b, bf, N, b_col_major);

        ck_tile::DeviceMem a_dev(a.size() * sizeof(ADataType));
        ck_tile::DeviceMem b_dev(b.size() * sizeof(BDataType));
        ck_tile::DeviceMem c_dev(static_cast<size_t>(M) * N * sizeof(CDataType));
        a_dev.ToDevice(a.data());
        b_dev.ToDevice(b.data());
        c_dev.SetZero();

        ck_tile::GemmHostArgs args(a_dev.GetDeviceBuffer(),
                                   b_dev.GetDeviceBuffer(),
                                   c_dev.GetDeviceBuffer(),
                                   k_batch,
                                   M,
                                   N,
                                   K,
                                   a_row_major ? K : M, // stride A
                                   b_col_major ? K : N, // stride B
                                   N);                  // stride C
        run_v3_gemm<Config>(args);

        std::vector<CDataType> c(static_cast<size_t>(M) * N);
        c_dev.FromDevice(c.data());

        // The reference splits rows across threads, so a 4096^3 check takes seconds.
        std::vector<size_t> bad_rows(static_cast<size_t>(M), 0);
        std::vector<std::string> first_bad(static_cast<size_t>(M));
        auto check_rows = [&](ck_tile::index_t m0, ck_tile::index_t m1) {
            for(ck_tile::index_t m = m0; m < m1; ++m)
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
                    if(init == Init::Sparse && sum_abs <= 256.0)
                    {
                        // every partial and running sum is an integer exact in bf16
                        ok = ck_tile::bit_cast<uint16_t>(got) == bf16_rne(static_cast<float>(sum));
                    }
                    else if(k_batch > 1 || init == Init::Sparse)
                    {
                        const double tol = std::abs(sum) * 0x1p-7 + k_batch * sum_abs * 0x1p-7;
                        ok               = std::abs(ck_tile::type_convert<float>(got) - sum) <= tol;
                    }
                    else if(init == Init::Int)
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
                    if(!ok && bad_rows[static_cast<size_t>(m)]++ == 0)
                    {
                        first_bad[static_cast<size_t>(m)] =
                            "m=" + std::to_string(m) + " n=" + std::to_string(n) +
                            " ref=" + std::to_string(sum) +
                            " got=" + std::to_string(ck_tile::type_convert<float>(got));
                    }
                }
            }
        };
        const ck_tile::index_t workers = static_cast<ck_tile::index_t>(
            std::max(1u, std::min(std::thread::hardware_concurrency(), 64u)));
        const ck_tile::index_t rows_per = (M + workers - 1) / workers;
        std::vector<std::thread> pool;
        for(ck_tile::index_t m0 = 0; m0 < M; m0 += rows_per)
            pool.emplace_back(check_rows, m0, std::min(M, m0 + rows_per));
        for(auto& t : pool)
            t.join();

        size_t bad = 0, shown = 0;
        for(ck_tile::index_t m = 0; m < M; ++m)
        {
            bad += bad_rows[static_cast<size_t>(m)];
            if(bad_rows[static_cast<size_t>(m)] != 0 && shown++ < 5)
            {
                ADD_FAILURE() << "M=" << M << " N=" << N << " K=" << K << " "
                              << first_bad[static_cast<size_t>(m)];
            }
        }
        EXPECT_EQ(bad, 0u) << "mismatches at M=" << M << " N=" << N << " K=" << K
                           << " k_batch=" << k_batch;
    }

    // Host check only, no launch: whether the kernel accepts M x N x K, which must match the C
    // store rule (C row-major): an unaligned N needs kPadN, and N must be a multiple of the C
    // vector size. A/B never limit the shape: TDM clips them in hardware.
    // With k_batch > 1 it must also match the split-K rule (SplitKSupported).
    bool ExpectSupported(ck_tile::index_t M,
                         ck_tile::index_t N,
                         ck_tile::index_t K,
                         ck_tile::index_t k_batch = 1)
    {
        using Kernel         = V3Kernel<Config>;
        const bool tile_ok   = N % Config::TileN == 0 || Config::PadMN;
        const bool vector_ok = N % Kernel::EpiloguePipeline::GetVectorSizeC() == 0;
        // A TDM epilogue clips C in hardware, so it accepts every shape.
        const bool expected = (Config::TdmEpilogue || (tile_ok && vector_ok)) &&
                              (k_batch == 1 || SplitKSupported(K, k_batch));
        const ck_tile::GemmHostArgs args(nullptr, nullptr, nullptr, k_batch, M, N, K, K, K, N);
        const bool supported = Kernel::IsSupportedArgument(Kernel::MakeKernelArgs(args));
        EXPECT_EQ(supported, expected)
            << "M=" << M << " N=" << N << " K=" << K << " k_batch=" << k_batch;
        return supported;
    }

    // Split-K needs an epilogue with a split-K store path -- not the TdmEpilogue, which overwrites
    // C, and for bf16 C a vector size of at least 2 -- and no more splits than warp-tile K steps.
    static bool SplitKSupported(ck_tile::index_t K, ck_tile::index_t k_batch)
    {
        using Kernel = V3Kernel<Config>;
        constexpr ck_tile::index_t WarpK =
            Kernel::GemmPipeline::BlockGemmShape::WarpTile::at(ck_tile::number<2>{});
        return !Config::TdmEpilogue && Kernel::UniversalGemmKernel::EpilogueSupportsSplitK &&
               ck_tile::integer_divide_ceil(K, WarpK) >= k_batch;
    }

    // Host check only, no launch: k_batch = 1 is accepted, and split-K is accepted exactly when
    // SplitKSupported says so, including a k_batch beyond the K steps, which must be rejected.
    void CheckSplitKSupport()
    {
        for(ck_tile::index_t k_batch : {1, 2, 4, 64})
            ExpectSupported(512, 512, 1024, k_batch);
    }

    // Split-K runs: K of five K tiles plus a partial one, so the splits end inside K tiles and
    // the last split is shorter; k_batch 2, 3 and 4, on sparse (exact) and random inputs.
    void RunSplitK()
    {
        const ck_tile::index_t K = 5 * Config::TileK + 32;
        int ran                  = 0;
        for(ck_tile::index_t k_batch : {2, 3, 4})
        {
            if(!ExpectSupported(256, 256, K, k_batch))
                continue;
            Run(256, 256, K, Init::Sparse, k_batch);
            Run(256, 256, K, Init::Random, k_batch);
            ++ran;
        }
        // The configurations under test all have a split-K store path.
        EXPECT_EQ(ran > 0, SplitKSupported(K, 2));
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

// bf16 x bf16 -> bf16, default policy, every A/B layout: 16 waves at 256x256 (K tiles 64 and 128)
// and 4 waves at 128x128 (K tile 64) for each layout, plus the other 128-K tiles in RCR. CCR stops
// at the 64-K tile: at 256x256x128 it spills.
using RM            = RowMajor;
using CM            = ColumnMajor;
using V3Bf16Configs = ::testing::Types<V3Bf16Config<256, 256, 64, 4, 4, RM, CM>,
                                       V3Bf16Config<256, 256, 128, 4, 4, RM, CM>,
                                       V3Bf16Config<256, 128, 128, 4, 4, RM, CM>,
                                       V3Bf16Config<128, 256, 128, 4, 4, RM, CM>,
                                       V3Bf16Config<128, 128, 64, 2, 2, RM, CM>,
                                       V3Bf16Config<128, 128, 128, 2, 2, RM, CM>,
                                       V3Bf16Config<256, 256, 64, 4, 4, RM, RM>,
                                       V3Bf16Config<256, 256, 128, 4, 4, RM, RM>,
                                       V3Bf16Config<128, 128, 64, 2, 2, RM, RM>,
                                       V3Bf16Config<256, 256, 64, 4, 4, CM, RM>,
                                       V3Bf16Config<256, 256, 128, 4, 4, CM, RM>,
                                       V3Bf16Config<128, 128, 64, 2, 2, CM, RM>,
                                       V3Bf16Config<256, 256, 64, 4, 4, CM, CM>,
                                       V3Bf16Config<128, 128, 64, 2, 2, CM, CM>>;

// Padded (kPadM = kPadN = true) for unaligned M and N: every layout at 256x256x64 and 128x128x64
// with the default C vector (8), and RCR 256x256x64 with C vectors 2 and 1 for N that 8 does not
// divide, plus CCR 128x128x64 with vector 1.
using V3Bf16PaddedConfigs = ::testing::Types<V3Bf16Config<256, 256, 64, 4, 4, RM, CM, true>,
                                             V3Bf16Config<128, 128, 64, 2, 2, RM, CM, true>,
                                             V3Bf16Config<256, 256, 64, 4, 4, RM, RM, true>,
                                             V3Bf16Config<128, 128, 64, 2, 2, RM, RM, true>,
                                             V3Bf16Config<256, 256, 64, 4, 4, CM, RM, true>,
                                             V3Bf16Config<128, 128, 64, 2, 2, CM, RM, true>,
                                             V3Bf16Config<256, 256, 64, 4, 4, CM, CM, true>,
                                             V3Bf16Config<128, 128, 64, 2, 2, CM, CM, true>,
                                             V3Bf16Config<256, 256, 64, 4, 4, RM, CM, true, 2>,
                                             V3Bf16Config<256, 256, 64, 4, 4, RM, CM, true, 1>,
                                             V3Bf16Config<128, 128, 64, 2, 2, CM, CM, true, 1>>;

// TdmEpilogue, unpadded, every layout at 256x256x64 and 128x128x64, and RCR 256x256x128.
using V3Bf16TdmEpilogueConfigs =
    ::testing::Types<V3Bf16Config<256, 256, 64, 4, 4, RM, CM, false, 0, true>,
                     V3Bf16Config<256, 256, 128, 4, 4, RM, CM, false, 0, true>,
                     V3Bf16Config<128, 128, 64, 2, 2, RM, CM, false, 0, true>,
                     V3Bf16Config<256, 256, 64, 4, 4, RM, RM, false, 0, true>,
                     V3Bf16Config<128, 128, 64, 2, 2, RM, RM, false, 0, true>,
                     V3Bf16Config<256, 256, 64, 4, 4, CM, RM, false, 0, true>,
                     V3Bf16Config<128, 128, 64, 2, 2, CM, RM, false, 0, true>,
                     V3Bf16Config<256, 256, 64, 4, 4, CM, CM, false, 0, true>,
                     V3Bf16Config<128, 128, 64, 2, 2, CM, CM, false, 0, true>>;

// Early refill (RefillPoint::AfterArrive) on M, N and K that are not multiples of the tile: the
// refill path has its own K-tile bookkeeping, so it is checked on a partial last K tile too.
// bf16 with TdmEpilogue, so every shape runs.
using AfterArrive = comp_tdm_v3_test::RefillAfterArrivePolicy;
using V3Bf16AfterArriveConfigs =
    ::testing::Types<V3Bf16Config<256, 256, 64, 4, 4, RM, CM, false, 0, true, AfterArrive>,
                     V3Bf16Config<128, 128, 64, 2, 2, RM, CM, false, 0, true, AfterArrive>,
                     V3Bf16Config<256, 256, 64, 4, 4, CM, RM, false, 0, true, AfterArrive>>;

// fp8 x fp8 -> bf16 (RCR, warp tile 32x32x128, K tile 256) with the default policy and with
// early refill, unpadded CShuffle as the fp8 instances ship: M and K need no padding, an unaligned
// N must be rejected. Plus early refill with TdmEpilogue, where an unaligned N runs too.
using V3Fp8UnalignedConfigs = ::testing::Types<V3Config<256, 256, 4, 4, DefaultPolicy>,
                                               V3Config<128, 128, 2, 2, DefaultPolicy>,
                                               V3Config<256, 256, 4, 4, AfterArrive>,
                                               V3Config<128, 128, 2, 2, AfterArrive>,
                                               V3Config<256,
                                                        256,
                                                        4,
                                                        4,
                                                        AfterArrive,
                                                        ck_tile::fp8_t,
                                                        ck_tile::fp8_t,
                                                        256,
                                                        32,
                                                        32,
                                                        128,
                                                        RM,
                                                        CM,
                                                        false,
                                                        0,
                                                        true>>;

// The 4096^3 set: the three RCR tiles that matter for large shapes, and 256x256x64 in each other
// layout.
using V3Bf16LargeConfigs = ::testing::Types<V3Bf16Config<256, 256, 64, 4, 4, RM, CM>,
                                            V3Bf16Config<256, 256, 128, 4, 4, RM, CM>,
                                            V3Bf16Config<128, 128, 128, 2, 2, RM, CM>,
                                            V3Bf16Config<256, 256, 64, 4, 4, RM, RM>,
                                            V3Bf16Config<256, 256, 64, 4, 4, CM, RM>,
                                            V3Bf16Config<256, 256, 64, 4, 4, CM, CM>>;

} // namespace
