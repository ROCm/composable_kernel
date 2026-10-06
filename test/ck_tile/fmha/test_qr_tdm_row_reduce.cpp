// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Verify the actual FMHA row distribution, paired-half exchange, and distribution fallback.
#include "example/ck_tile/01_fmha/fmha_fwd.hpp"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <limits>
#include <vector>
#include "gtest/gtest.h"

namespace {

template <int M,
          bool Mask,
          typename T,
          ck_tile::BlockAttentionBiasEnum Bias = ck_tile::BlockAttentionBiasEnum::NO_BIAS,
          bool StoreLSE                        = false,
          ck_tile::BlockAttentionQuantScaleEnum QScale =
              ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
          bool HasSink            = false,
          bool ProgressiveDsLoadK = true>
using Problem = ck_tile::BlockFmhaPipelineProblem<
    T,
    T,
    T,
    float,
    float,
    T,
    uint8_t,
    float,
    T,
    float,
    T,
    ck_tile::TileFmhaShape<ck_tile::sequence<M, 64, 32, 128, 32, 128>,
                           ck_tile::sequence<4, 1, 1>,
                           ck_tile::sequence<16, 16, 32>,
                           ck_tile::sequence<4, 1, 1>,
                           ck_tile::sequence<16, 16, 32>,
                           true>,
    false,
    ck_tile::ComposedAttention<0>,
    ck_tile::SimplifiedGenericAttentionMask<Mask>,
    false,
    ck_tile::TileFmhaTraits<false,
                            false,
                            false,
                            false,
                            false,
                            Bias,
                            false,
                            StoreLSE,
                            false,
                            QScale,
                            -1,
                            false,
                            HasSink>,
    true,
    ProgressiveDsLoadK>;

template <typename P>
using TestGemm = ck_tile::remove_cvref_t<
    decltype(ck_tile::BlockFmhaPipelineQRKSVSTdmDefaultPolicy::GetQKBlockGemm<P>())>;

template <typename P>
using TestScores = decltype(TestGemm<P>{}.MakeCBlockTile());

// This legal pipeline type crosses the former FP16/M64, progressive-load,
// bias, sink and LSE guards. The selector itself depends only on tensor structure.
using FormerFallbackProblem = Problem<64,
                                      true,
                                      ck_tile::half_t,
                                      ck_tile::BlockAttentionBiasEnum::ELEMENTWISE_BIAS,
                                      true,
                                      ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE,
                                      true,
                                      false>;
using FormerFallbackPipe    = ck_tile::BlockFmhaPipelineQRKSVSTdm<FormerFallbackProblem>;
static_assert(
    FormerFallbackPipe::template GetRowMaxPartialCount<TestScores<FormerFallbackProblem>>() == 2);

template <int M, bool Mask, typename T>
__global__ void reduce_rows(float* out)
{
    using P     = Problem<M, Mask, T>;
    using Pipe  = ck_tile::BlockFmhaPipelineQRKSVSTdm<P>;
    auto gemm   = ck_tile::BlockFmhaPipelineQRKSVSTdmDefaultPolicy::GetQKBlockGemm<P>();
    auto scores = gemm.MakeCBlockTile();
    // Independent partial reduction is a QR-TDM local-row operation, not a
    // BF16/M128-only optimization. These instantiations include M64 and FP16.
    static_assert(Pipe::template GetRowMaxPartialCount<decltype(scores)>() == (Mask ? 2 : 4));
    // Distinct M128 row slots catch accidental cross-slot mixing.
    ck_tile::static_for<0, decltype(scores)::get_thread_buffer_size(), 1>{}([&](auto i) {
        scores.get_thread_buffer()(i) = float(threadIdx.x % 32 + 1 + 100 * (i / 32));
    });
    auto add    = [](float a, float b) { return a + b; };
    auto max    = [](float a, float b) { return ck_tile::max(a, b); };
    auto sums   = ck_tile::block_tile_reduce<float>(scores, ck_tile::sequence<1>{}, add, 0.f);
    auto maxima = Pipe::ReduceRowMaxLocal(scores);
    Pipe::ReduceRowSync(sums, add);
    Pipe::ReduceRowSync(maxima, max);
    ck_tile::static_for<0, decltype(sums)::get_thread_buffer_size(), 1>{}([&](auto i) {
        out[threadIdx.x * 4 + 2 * i]     = sums.get_thread_buffer()[i];
        out[threadIdx.x * 4 + 2 * i + 1] = maxima.get_thread_buffer()[i];
    });
}

template <int M, bool Mask, typename T>
bool check()
{
    float* device = nullptr;
    if(hipMalloc(&device, 512 * sizeof(float)) != hipSuccess)
        return false;
    hipLaunchKernelGGL((reduce_rows<M, Mask, T>), dim3(1), dim3(128), 0, 0, device);
    std::vector<float> got(512);
    const auto launch = hipGetLastError();
    const auto copy =
        hipMemcpy(got.data(), device, got.size() * sizeof(float), hipMemcpyDeviceToHost);
    const auto release = hipFree(device);
    if(launch != hipSuccess || copy != hipSuccess || release != hipSuccess)
        return false;
    for(int tid = 0; tid < 128; ++tid)
        for(int row = 0; row < M / 64; ++row)
        {
            const int row_lane   = tid % 16;
            const float want_sum = 32.f * (2 * row_lane + 18 + 200 * row);
            const float want_max = float(row_lane + 17 + 100 * row);
            if(got[tid * 4 + 2 * row] != want_sum || got[tid * 4 + 2 * row + 1] != want_max)
            {
                std::printf("FAIL M%d mask%d tid%d row%d: sum %g/%g max %g/%g\n",
                            M,
                            Mask,
                            tid,
                            row,
                            got[tid * 4 + 2 * row],
                            want_sum,
                            got[tid * 4 + 2 * row + 1],
                            want_max);
                return false;
            }
        }
    std::printf("PASS row reduction M%d mask%d element_bytes%zu\n", M, Mask, sizeof(T));
    return true;
}

__global__ void reduce_four_lanes(float* out)
{
    using namespace ck_tile;
    using Pipe = BlockFmhaPipelineQRKSVSTdm<Problem<64, false, bf16_t>>;
    using E    = tile_distribution_encoding<sequence<4>,
                                            tuple<sequence<4, 8, 2>>,
                                            tuple<sequence<1>, sequence<0, 1>>,
                                            tuple<sequence<0>, sequence<0, 1>>,
                                            sequence<1>,
                                            sequence<2>>;
    static_assert(E::rs_lengths_[0] == 4);
    static_assert(E::detail::ps_over_rs_derivative_[1][0] == 8);
    auto sum = make_static_distributed_tensor<float>(make_static_tile_distribution(E{}));
    auto max = sum;
    static_for<0, 2, 1>{}([&](auto i) {
        sum.get_thread_buffer()(i) =
            float(1000 * (threadIdx.x / 32) + 100 * i.value + threadIdx.x % 32 + 1);
        max.get_thread_buffer()(i) = sum.get_thread_buffer()[i];
    });
    Pipe::ReduceRowSync(sum, [](float a, float b) { return a + b; });
    Pipe::ReduceRowSync(max, [](float a, float b) { return ck_tile::max(a, b); });
    static_for<0, 2, 1>{}([&](auto i) {
        out[4 * threadIdx.x + 2 * i.value]     = sum.get_thread_buffer()[i];
        out[4 * threadIdx.x + 2 * i.value + 1] = max.get_thread_buffer()[i];
    });
}

__global__ void reduce_short_local_row(float* out)
{
    using namespace ck_tile;
    using Pipe  = BlockFmhaPipelineQRKSVSTdm<Problem<64, false, bf16_t>>;
    using E     = tile_distribution_encoding<sequence<>,
                                             tuple<sequence<1>, sequence<1, 3>>,
                                             tuple<sequence<2>>,
                                             tuple<sequence<0>>,
                                             sequence<1, 2>,
                                             sequence<0, 1>>;
    auto scores = make_static_distributed_tensor<float>(make_static_tile_distribution(E{}));
    static_assert(Pipe::template GetRowMaxPartialCount<decltype(scores)>() == 0);
    scores.get_thread_buffer()(number<0>{}) = -4.f;
    scores.get_thread_buffer()(number<1>{}) = -2.f;
    scores.get_thread_buffer()(number<2>{}) = -3.f;
    const auto max                          = [](float a, float b) { return ck_tile::max(a, b); };
    const auto got                          = Pipe::ReduceRowMaxLocal(scores);
    const auto reference =
        block_tile_reduce<float>(scores, sequence<1>{}, max, -numeric<float>::infinity());
    out[0] = got.get_thread_buffer()[0];
    out[1] = reference.get_thread_buffer()[0];
}

bool check_short_local_row_fallback()
{
    float* device = nullptr;
    if(hipMalloc(&device, 2 * sizeof(float)) != hipSuccess)
        return false;
    hipLaunchKernelGGL(reduce_short_local_row, dim3(1), dim3(1), 0, 0, device);
    float got[2]       = {};
    const auto launch  = hipGetLastError();
    const auto copy    = hipMemcpy(got, device, sizeof(got), hipMemcpyDeviceToHost);
    const auto release = hipFree(device);
    return launch == hipSuccess && copy == hipSuccess && release == hipSuccess && got[0] == -2.f &&
           got[1] == -2.f;
}

// A dropped local column/group or a mixed M128 row slot must lose a unique
// maximum in one of these blocks. Exercise every column in either lane half.
template <int M, bool Mask, typename T>
__global__ void reduce_column_peaks(float* out, int mode)
{
    using P         = Problem<M, Mask, T>;
    using Pipe      = ck_tile::BlockFmhaPipelineQRKSVSTdm<P>;
    auto gemm       = ck_tile::BlockFmhaPipelineQRKSVSTdmDefaultPolicy::GetQKBlockGemm<P>();
    auto scores     = gemm.MakeCBlockTile();
    const float inf = ck_tile::numeric<float>::infinity();
    ck_tile::static_for<0, decltype(scores)::get_thread_buffer_size(), 1>{}([&](auto i) {
        constexpr int row = i / 32;
        const bool peak   = i % 32 == blockIdx.x % 32 && (threadIdx.x % 32) / 16 == blockIdx.x / 32;
        const float background        = mode >= 3 ? __builtin_nanf("") : -1000.f - i - threadIdx.x;
        const float peak_value        = mode == 2 ? inf : -3.f - 100 * row - 2 * (threadIdx.x % 16);
        scores.get_thread_buffer()(i) = mode == 1   ? -inf
                                        : mode == 4 ? background
                                        : peak      ? peak_value
                                                    : background;
    });
    auto max    = [](float a, float b) { return ck_tile::max(a, b); };
    auto maxima = Pipe::ReduceRowMaxLocal(scores);
    Pipe::ReduceRowSync(maxima, max);
    ck_tile::static_for<0, decltype(maxima)::get_thread_buffer_size(), 1>{}([&](auto i) {
        out[(blockIdx.x * 128 + threadIdx.x) * 2 + i] = maxima.get_thread_buffer()[i];
    });
}

template <int M, bool Mask, typename T>
bool check_column_peaks(int mode)
{
    std::vector<float> got(64 * 128 * 2);
    float* device = nullptr;
    if(hipMalloc(&device, got.size() * sizeof(float)) != hipSuccess)
        return false;
    hipLaunchKernelGGL((reduce_column_peaks<M, Mask, T>), dim3(64), dim3(128), 0, 0, device, mode);
    const auto launch = hipGetLastError();
    const auto copy =
        hipMemcpy(got.data(), device, got.size() * sizeof(float), hipMemcpyDeviceToHost);
    const auto release = hipFree(device);
    if(launch != hipSuccess || copy != hipSuccess || release != hipSuccess)
        return false;
    for(int block = 0; block < 64; ++block)
        for(int tid = 0; tid < 128; ++tid)
            for(int row = 0; row < M / 64; ++row)
            {
                const float want = mode == 1 || mode == 4 ? -std::numeric_limits<float>::infinity()
                                   : mode == 2            ? std::numeric_limits<float>::infinity()
                                                          : -3.f - 100 * row - 2 * (tid % 16);
                const float actual = got[(block * 128 + tid) * 2 + row];
                if(actual != want)
                {
                    std::printf("FAIL column peak M%d mask%d mode%d block%d tid%d row%d: %g/%g\n",
                                M,
                                Mask,
                                mode,
                                block,
                                tid,
                                row,
                                actual,
                                want);
                    return false;
                }
            }
    return true;
}

bool check_distribution_fallback()
{
    float* device = nullptr;
    if(hipMalloc(&device, 512 * sizeof(float)) != hipSuccess)
        return false;
    hipLaunchKernelGGL(reduce_four_lanes, dim3(1), dim3(128), 0, 0, device);
    std::vector<float> got(512);
    const auto launch = hipGetLastError();
    const auto copy =
        hipMemcpy(got.data(), device, got.size() * sizeof(float), hipMemcpyDeviceToHost);
    const auto release = hipFree(device);
    if(launch != hipSuccess || copy != hipSuccess || release != hipSuccess)
        return false;
    for(int tid = 0; tid < 128; ++tid)
        for(int slot = 0; slot < 2; ++slot)
        {
            const int row    = tid % 8;
            const int offset = 1000 * (tid / 32) + 100 * slot;
            if(got[4 * tid + 2 * slot] != float(4 * offset + 4 * row + 52) ||
               got[4 * tid + 2 * slot + 1] != float(offset + row + 25))
            {
                std::printf("FAIL distribution fallback tid%d slot%d\n", tid, slot);
                return false;
            }
        }
    std::puts("PASS four-lane distribution fallback");
    return true;
}

TEST(QrTdmRowReduction, DenseBf16UsesCorrectRowPartners)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    EXPECT_TRUE((check<64, false, ck_tile::bf16_t>()));
    EXPECT_TRUE((check<128, false, ck_tile::bf16_t>()));
}

TEST(QrTdmRowReduction, MaskedAndOtherInputTypesUseCorrectRowPartners)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    EXPECT_TRUE((check<64, true, ck_tile::bf16_t>()));
    EXPECT_TRUE((check<128, true, ck_tile::bf16_t>()));
    EXPECT_TRUE((check<64, false, ck_tile::half_t>()));
    EXPECT_TRUE((check<128, false, ck_tile::half_t>()));
    EXPECT_TRUE((check<64, true, ck_tile::half_t>()));
    EXPECT_TRUE((check<128, true, ck_tile::half_t>()));
}

TEST(QrTdmRowReduction, FourLaneSyncUsesGenericReduction)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    EXPECT_TRUE(check_distribution_fallback());
}

TEST(QrTdmRowReduction, ShortLocalRowUsesGenericReduction)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    EXPECT_TRUE(check_short_local_row_fallback());
}

TEST(QrTdmRowReduction, EveryLocalColumnCanSupplyNegativeMaximum)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    EXPECT_TRUE((check_column_peaks<128, false, ck_tile::bf16_t>(0)));
    EXPECT_TRUE((check_column_peaks<128, true, ck_tile::bf16_t>(0)));
    EXPECT_TRUE((check_column_peaks<64, false, ck_tile::bf16_t>(0)));
    EXPECT_TRUE((check_column_peaks<128, false, ck_tile::half_t>(0)));
    EXPECT_TRUE((check_column_peaks<64, true, ck_tile::half_t>(0)));
}

TEST(QrTdmRowReduction, FullyMaskedRowsRemainNegativeInfinity)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    EXPECT_TRUE((check_column_peaks<128, true, ck_tile::bf16_t>(1)));
    EXPECT_TRUE((check_column_peaks<128, false, ck_tile::bf16_t>(1)));
}

TEST(QrTdmRowReduction, EveryLocalColumnCanSupplyPositiveInfinity)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    EXPECT_TRUE((check_column_peaks<128, false, ck_tile::bf16_t>(2)));
}

TEST(QrTdmRowReduction, NaNsRetainMaxNumAndNegativeInfinityIdentity)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    EXPECT_TRUE((check_column_peaks<128, false, ck_tile::bf16_t>(3)));
    EXPECT_TRUE((check_column_peaks<128, false, ck_tile::bf16_t>(4)));
    EXPECT_TRUE((check_column_peaks<128, true, ck_tile::bf16_t>(3)));
    EXPECT_TRUE((check_column_peaks<128, true, ck_tile::bf16_t>(4)));
    EXPECT_TRUE((check_column_peaks<64, false, ck_tile::bf16_t>(3)));
    EXPECT_TRUE((check_column_peaks<64, false, ck_tile::bf16_t>(4)));
    EXPECT_TRUE((check_column_peaks<128, false, ck_tile::half_t>(3)));
    EXPECT_TRUE((check_column_peaks<128, false, ck_tile::half_t>(4)));
    EXPECT_TRUE((check_column_peaks<64, true, ck_tile::half_t>(3)));
    EXPECT_TRUE((check_column_peaks<64, true, ck_tile::half_t>(4)));
}

} // namespace
