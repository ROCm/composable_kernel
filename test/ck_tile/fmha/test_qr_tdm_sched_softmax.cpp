// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "fmha_test_common.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_sched_policy.hpp"
#include "ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp"
#include "ck_tile/ops/gemm/warp/warp_wmma_gemm.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace {

struct MaxOp
{
    CK_TILE_HOST_DEVICE float operator()(float x, float y) const { return ck_tile::max(x, y); }
};

constexpr ck_tile::index_t kThreads      = 128;
constexpr ck_tile::index_t kTwoTileCount = 2;

template <typename Data, ck_tile::index_t HeadDim, ck_tile::index_t N, ck_tile::index_t ValueDim>
struct SoftmaxConfig
{
    using Geometry = ck_tile::FmhaTdmSchedNativeGeometry<Data, HeadDim, N, ValueDim>;
    using Policy   = ck_tile::BlockFmhaPipelineQRKSVSTdmSchedPolicy<
          Geometry,
          ck_tile::FmhaTdmSchedDerivedTuning<Geometry>>;
    using Mapping         = typename Policy::ScoreMapping;
    using OutputFragments = typename Policy::OutputFragments;
    using OutputMapping   = typename OutputFragments::Mapping;
    using BlockShape      = ck_tile::TileFmhaShape<
             ck_tile::sequence<Geometry::kM, Geometry::kN, 32, Geometry::kDv, 32, Geometry::kHeadDimQK>,
             ck_tile::sequence<4, 1, 1>,
             ck_tile::sequence<16, 16, 32>,
             ck_tile::sequence<4, 1, 1>,
             ck_tile::sequence<16, 16, 32>,
             true>;
    using WarpGemm   = std::conditional_t<std::is_same_v<Data, ck_tile::bf16_t>,
                                          ck_tile::WarpGemmWmma_f32_16x16x32_bf16_bf16<true>,
                                          ck_tile::WarpGemmWmma_f32_16x16x32_f16_f16<true>>;
    using GemmPolicy = ck_tile::BlockGemmARegBRegCRegV2CustomPolicy<Data,
                                                                    Data,
                                                                    float,
                                                                    ck_tile::sequence<4, 1, 1>,
                                                                    WarpGemm,
                                                                    ck_tile::GemmLoopOrder::MNK>;

    template <ck_tile::index_t Columns>
    using GemmProblem = ck_tile::BlockGemmProblem<
        Data,
        Data,
        float,
        kThreads,
        ck_tile::TileGemmShape<ck_tile::sequence<Geometry::kM, Columns, 32>,
                               ck_tile::sequence<4, 1, 1>,
                               ck_tile::sequence<16, 16, 32>>>;
    template <ck_tile::index_t Columns>
    using BlockGemm  = ck_tile::BlockGemmARegBRegCRegV2<GemmProblem<Columns>, GemmPolicy>;
    using ScoreTile  = decltype(BlockGemm<Geometry::kN>::MakeCBlockTile());
    using OutputTile = decltype(BlockGemm<Geometry::kDv>::MakeCBlockTile());
    using RowTile    = decltype(ck_tile::block_tile_reduce<float>(
        ScoreTile{}, ck_tile::sequence<1>{}, MaxOp{}, -ck_tile::numeric<float>::infinity()));

    template <bool Masking>
    struct Problem
    {
        static constexpr bool kIsGroupMode = false;
        struct FmhaMask
        {
            static constexpr bool IsMasking = Masking;
        };
        using QDataType                              = Data;
        using KDataType                              = Data;
        using VDataType                              = Data;
        using PDataType                              = Data;
        using SaccDataType                           = float;
        using OaccDataType                           = float;
        using BlockFmhaShape                         = BlockShape;
        static constexpr ck_tile::index_t kBlockSize = kThreads;
        static constexpr auto BiasEnum               = ck_tile::BlockAttentionBiasEnum::NO_BIAS;
        static constexpr bool kHasLogitsSoftCap      = false;
    };

    static constexpr ck_tile::index_t kOutputValues = OutputMapping::kThreadBufferSize;
    static constexpr ck_tile::index_t kStateOffset  = Mapping::kThreadBufferSize + kOutputValues;
    static constexpr ck_tile::index_t kInputStride  = kStateOffset + 5;
    static constexpr ck_tile::index_t kOutputStride = 4 + kStateOffset;
    static constexpr ck_tile::index_t kTwoTileStateOffset =
        kTwoTileCount * Mapping::kThreadBufferSize + kOutputValues;
    static constexpr ck_tile::index_t kTwoTileInputStride       = kTwoTileStateOffset + 5;
    static constexpr ck_tile::index_t kTwoTileScoreOutputOffset = 6;
    static constexpr ck_tile::index_t kTwoTileOutputAccumulatorOffset =
        kTwoTileScoreOutputOffset + kTwoTileCount * Mapping::kThreadBufferSize;
    static constexpr ck_tile::index_t kTwoTileOutputStride =
        kTwoTileOutputAccumulatorOffset + kOutputValues;

    static_assert(ScoreTile::get_thread_buffer_size() == Mapping::kThreadBufferSize);
    static_assert(OutputTile::get_thread_buffer_size() == kOutputValues);
    static_assert(RowTile::get_thread_buffer_size() == 2);
    static_assert(Mapping::kNumSu == Geometry::kQkStages);
    static_assert(OutputFragments::ValidateMapping());
    static_assert(Policy::template IsSupportedProblem<Problem<false>>() &&
                  Policy::template IsSupportedProblem<Problem<true>>());
};

using Bf16D192N128 = SoftmaxConfig<ck_tile::bf16_t, 192, 128, 128>;
using Fp16D192N128 = SoftmaxConfig<ck_tile::half_t, 192, 128, 128>;
using Bf16D192N64  = SoftmaxConfig<ck_tile::bf16_t, 192, 64, 128>;
using Fp16D192N64  = SoftmaxConfig<ck_tile::half_t, 192, 64, 128>;
using Bf16D64N64   = SoftmaxConfig<ck_tile::bf16_t, 64, 64, 64>;
using Fp16D64N64   = SoftmaxConfig<ck_tile::half_t, 64, 64, 64>;

template <typename Config = Bf16D192N128>
CK_TILE_DEVICE auto LoadOutputFragments(const float* input)
{
    using OutputFragments = typename Config::OutputFragments;
    return OutputFragments::Make([&](auto ordinal) {
        ck_tile::fp32x8_t fragment{};
        ck_tile::static_for<0, OutputFragments::kElementsPerFragment, 1>{}([&](auto element) {
            constexpr auto offset =
                OutputFragments::template GetThreadBufferOffset<decltype(ordinal)::value,
                                                                decltype(element)::value>();
            fragment[decltype(element)::value] = input[offset];
        });
        return fragment;
    });
}

template <bool Masking, typename Config = Bf16D192N128>
__global__ void RunSplitSoftmax(const float* input, float* output)
{
    using Mapping                = typename Config::Mapping;
    using ScoreTile              = typename Config::ScoreTile;
    using RowTile                = typename Config::RowTile;
    using Policy                 = typename Config::Policy;
    using OutputFragments        = typename Config::OutputFragments;
    constexpr auto kInputStride  = Config::kInputStride;
    constexpr auto kOutputStride = Config::kOutputStride;
    constexpr auto kStateOffset  = Config::kStateOffset;
    const auto* lane_input       = input + threadIdx.x * kInputStride;
    auto* lane_output            = output + threadIdx.x * kOutputStride;
    auto score                   = ScoreTile{};
    auto row_max                 = RowTile{};
    auto row_sum                 = RowTile{};
    auto output_fragments = LoadOutputFragments<Config>(lane_input + Mapping::kThreadBufferSize);

    for(ck_tile::index_t i = 0; i < Mapping::kThreadBufferSize; ++i)
    {
        score.get_thread_buffer()[i] = lane_input[i];
    }
    row_max.get_thread_buffer()[0] = lane_input[kStateOffset + 0];
    row_max.get_thread_buffer()[1] = lane_input[kStateOffset + 1];
    row_sum.get_thread_buffer()[0] = lane_input[kStateOffset + 2];
    row_sum.get_thread_buffer()[1] = lane_input[kStateOffset + 3];

    Policy::template RunSplitSoftmaxFragments<typename Config::template Problem<Masking>>(
        score, row_max, row_sum, output_fragments, lane_input[kStateOffset + 4]);
    const auto output_acc =
        OutputFragments::template Reconstruct<typename Config::OutputTile>(output_fragments);

    lane_output[0] = row_max.get_thread_buffer()[0];
    lane_output[1] = row_max.get_thread_buffer()[1];
    lane_output[2] = row_sum.get_thread_buffer()[0];
    lane_output[3] = row_sum.get_thread_buffer()[1];
    for(ck_tile::index_t i = 0; i < Mapping::kThreadBufferSize; ++i)
    {
        lane_output[4 + i] = score.get_thread_buffer()[i];
    }
    for(ck_tile::index_t i = 0; i < Config::kOutputValues; ++i)
    {
        lane_output[4 + Mapping::kThreadBufferSize + i] = output_acc.get_thread_buffer()[i];
    }
}

template <bool Masking, typename Config = Bf16D192N128>
__global__ void RunSplitSoftmaxTwoTiles(const float* input, float* output)
{
    using Mapping                                  = typename Config::Mapping;
    using ScoreTile                                = typename Config::ScoreTile;
    using RowTile                                  = typename Config::RowTile;
    using Policy                                   = typename Config::Policy;
    using OutputFragments                          = typename Config::OutputFragments;
    constexpr auto kTwoTileInputStride             = Config::kTwoTileInputStride;
    constexpr auto kTwoTileOutputStride            = Config::kTwoTileOutputStride;
    constexpr auto kTwoTileStateOffset             = Config::kTwoTileStateOffset;
    constexpr auto kTwoTileScoreOutputOffset       = Config::kTwoTileScoreOutputOffset;
    constexpr auto kTwoTileOutputAccumulatorOffset = Config::kTwoTileOutputAccumulatorOffset;
    const auto* lane_input                         = input + threadIdx.x * kTwoTileInputStride;
    auto* lane_output                              = output + threadIdx.x * kTwoTileOutputStride;
    auto row_max                                   = RowTile{};
    auto row_sum                                   = RowTile{};
    auto output_fragments =
        LoadOutputFragments<Config>(lane_input + kTwoTileCount * Mapping::kThreadBufferSize);
    row_max.get_thread_buffer()[0] = lane_input[kTwoTileStateOffset + 0];
    row_max.get_thread_buffer()[1] = lane_input[kTwoTileStateOffset + 1];
    row_sum.get_thread_buffer()[0] = lane_input[kTwoTileStateOffset + 2];
    row_sum.get_thread_buffer()[1] = lane_input[kTwoTileStateOffset + 3];

    ck_tile::static_for<0, kTwoTileCount, 1>{}([&](auto tile) {
        auto score = ScoreTile{};
        for(ck_tile::index_t i = 0; i < Mapping::kThreadBufferSize; ++i)
        {
            score.get_thread_buffer()[i] = lane_input[tile * Mapping::kThreadBufferSize + i];
        }

        Policy::template RunSplitSoftmaxFragments<typename Config::template Problem<Masking>>(
            score, row_max, row_sum, output_fragments, lane_input[kTwoTileStateOffset + 4]);

        for(ck_tile::index_t i = 0; i < Mapping::kThreadBufferSize; ++i)
        {
            lane_output[kTwoTileScoreOutputOffset + tile * Mapping::kThreadBufferSize + i] =
                score.get_thread_buffer()[i];
        }
    });

    lane_output[0] = row_max.get_thread_buffer()[0];
    lane_output[1] = row_max.get_thread_buffer()[1];
    lane_output[2] = row_sum.get_thread_buffer()[0];
    lane_output[3] = row_sum.get_thread_buffer()[1];
    lane_output[4] =
        row_max.get_thread_buffer()[0] * lane_input[kTwoTileStateOffset + 4] / ck_tile::log2e_v<> +
        ck_tile::log(row_sum.get_thread_buffer()[0]);
    lane_output[5] =
        row_max.get_thread_buffer()[1] * lane_input[kTwoTileStateOffset + 4] / ck_tile::log2e_v<> +
        ck_tile::log(row_sum.get_thread_buffer()[1]);

    const auto output_acc =
        OutputFragments::template Reconstruct<typename Config::OutputTile>(output_fragments);
    for(ck_tile::index_t i = 0; i < Config::kOutputValues; ++i)
    {
        lane_output[kTwoTileOutputAccumulatorOffset + i] = output_acc.get_thread_buffer()[i];
    }
}

void CheckHip(hipError_t status, const char* operation)
{
    if(status != hipSuccess)
    {
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
    }
}

float ScoreValue(ck_tile::index_t lane, ck_tile::index_t msb, ck_tile::index_t scalar)
{
    return -4.0f + 0.01f * static_cast<float>(lane & 15) + 0.02f * static_cast<float>(lane >> 4) +
           0.1f * static_cast<float>(msb) + 0.001f * static_cast<float>(scalar);
}

bool Near(float actual, float expected)
{
    if(std::isinf(expected))
    {
        return std::isinf(actual) && std::signbit(actual) == std::signbit(expected);
    }
    return std::abs(actual - expected) <= 2.0e-4f * std::max(1.0f, std::abs(expected));
}

template <bool Masking, typename Config = Bf16D192N128>
bool RunCase()
{
    using Mapping                = typename Config::Mapping;
    using OutputMapping          = typename Config::OutputMapping;
    constexpr auto kInputStride  = Config::kInputStride;
    constexpr auto kOutputStride = Config::kOutputStride;
    constexpr auto kStateOffset  = Config::kStateOffset;
    std::vector<float> input(kThreads * kInputStride);
    std::vector<float> output(kThreads * kOutputStride);
    constexpr float scale = 0.125f;

    for(ck_tile::index_t thread = 0; thread < kThreads; ++thread)
    {
        auto* lane_input        = input.data() + thread * kInputStride;
        const auto lane         = thread % 32;
        const float initial_max = Masking ? -std::numeric_limits<float>::infinity()
                                          : -5.0f + 0.01f * static_cast<float>(lane & 15);

        for(ck_tile::index_t offset = 0; offset < Mapping::kThreadBufferSize; ++offset)
        {
            const auto coordinate = Mapping::DecodeThreadBufferOffset(offset);
            lane_input[offset] =
                Masking
                    ? -std::numeric_limits<float>::infinity()
                    : ScoreValue(lane, coordinate.msb, coordinate.pair * 2 + coordinate.element);
        }
        for(ck_tile::index_t offset = 0; offset < Config::kOutputValues; ++offset)
        {
            lane_input[Mapping::kThreadBufferSize + offset] =
                1.0f + 0.001f * static_cast<float>(offset);
        }
        lane_input[kStateOffset + 0] = initial_max;
        lane_input[kStateOffset + 1] = initial_max + (Masking ? 0.0f : 0.25f);
        lane_input[kStateOffset + 2] = Masking ? 0.0f : 0.75f;
        lane_input[kStateOffset + 3] = Masking ? 0.0f : 1.25f;
        lane_input[kStateOffset + 4] = scale;
    }

    float* device_input  = nullptr;
    float* device_output = nullptr;
    CheckHip(hipMalloc(&device_input, input.size() * sizeof(float)), "hipMalloc input");
    CheckHip(hipMalloc(&device_output, output.size() * sizeof(float)), "hipMalloc output");
    CheckHip(
        hipMemcpy(device_input, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice),
        "hipMemcpy input");
    hipLaunchKernelGGL((RunSplitSoftmax<Masking, Config>),
                       dim3(1),
                       dim3(kThreads),
                       0,
                       0,
                       device_input,
                       device_output);
    CheckHip(hipGetLastError(), "softmax probe launch");
    CheckHip(hipDeviceSynchronize(), "softmax probe synchronize");
    CheckHip(
        hipMemcpy(
            output.data(), device_output, output.size() * sizeof(float), hipMemcpyDeviceToHost),
        "hipMemcpy output");
    CheckHip(hipFree(device_input), "hipFree input");
    CheckHip(hipFree(device_output), "hipFree output");

    bool valid = true;
    for(ck_tile::index_t thread = 0; thread < kThreads; ++thread)
    {
        const auto lane      = thread % 32;
        const auto wave      = thread / 32;
        const auto peer      = wave * 32 + (lane ^ 16);
        const auto* lane_in  = input.data() + thread * kInputStride;
        const auto* peer_in  = input.data() + peer * kInputStride;
        const auto* lane_out = output.data() + thread * kOutputStride;

        for(ck_tile::index_t row = 0; row < 2; ++row)
        {
            const float old_max = lane_in[kStateOffset + row];
            const float old_sum = lane_in[kStateOffset + 2 + row];
            float new_max       = old_max;
            for(ck_tile::index_t msb = row * 2; msb < row * 2 + 2; ++msb)
            {
                for(ck_tile::index_t pair = 0; pair < Mapping::kPairsPerMsb; ++pair)
                {
                    for(ck_tile::index_t element = 0; element < 2; ++element)
                    {
                        const auto offset = Mapping::GetThreadBufferOffset(msb, pair, element);
                        new_max           = std::max(new_max, lane_in[offset]);
                        new_max           = std::max(new_max, peer_in[offset]);
                    }
                }
            }

            const float exponent_max = Masking && std::isinf(new_max) ? 0.0f : new_max;
            const float delta        = std::fma(-exponent_max, scale, old_max * scale);
            float tile_sum           = 0.0f;
            for(ck_tile::index_t msb = row * 2; msb < row * 2 + 2; ++msb)
            {
                for(ck_tile::index_t pair = 0; pair < Mapping::kPairsPerMsb; ++pair)
                {
                    for(ck_tile::index_t element = 0; element < 2; ++element)
                    {
                        const auto offset = Mapping::GetThreadBufferOffset(msb, pair, element);
                        const float expected =
                            std::exp2(std::fma(lane_in[offset], scale, -exponent_max * scale));
                        const float actual = lane_out[4 + offset];
                        valid &= Near(actual, expected);
                        tile_sum +=
                            expected +
                            std::exp2(std::fma(peer_in[offset], scale, -exponent_max * scale));
                    }
                }
            }

            valid &= Near(lane_out[row], new_max);
            valid &= Near(lane_out[2 + row], std::fma(std::exp2(delta), old_sum, tile_sum));

            for(ck_tile::index_t msb = row * 2; msb < row * 2 + 2; ++msb)
            {
                for(ck_tile::index_t pair = 0; pair < OutputMapping::kPairsPerMsb; ++pair)
                {
                    for(ck_tile::index_t element = 0; element < 2; ++element)
                    {
                        const auto offset =
                            OutputMapping::GetThreadBufferOffset(msb, pair, element);
                        const float initial_output = lane_in[Mapping::kThreadBufferSize + offset];
                        valid &= Near(lane_out[4 + Mapping::kThreadBufferSize + offset],
                                      initial_output * std::exp2(delta));
                    }
                }
            }
        }
    }

    std::cout << (Masking ? "masked_all_inf" : "lane_distinct") << ": " << (valid ? "pass" : "fail")
              << '\n';
    return valid;
}

float TwoTileScoreValue(ck_tile::index_t tile,
                        ck_tile::index_t lane,
                        ck_tile::index_t msb,
                        ck_tile::index_t scalar)
{
    return ScoreValue(lane, msb, scalar) + 0.2f * static_cast<float>(tile);
}

template <bool Masking, typename Config = Bf16D192N128>
bool RunTwoTileCase()
{
    using Mapping                                  = typename Config::Mapping;
    using OutputMapping                            = typename Config::OutputMapping;
    constexpr auto kTwoTileInputStride             = Config::kTwoTileInputStride;
    constexpr auto kTwoTileOutputStride            = Config::kTwoTileOutputStride;
    constexpr auto kTwoTileStateOffset             = Config::kTwoTileStateOffset;
    constexpr auto kTwoTileScoreOutputOffset       = Config::kTwoTileScoreOutputOffset;
    constexpr auto kTwoTileOutputAccumulatorOffset = Config::kTwoTileOutputAccumulatorOffset;
    std::vector<float> input(kThreads * kTwoTileInputStride);
    std::vector<float> output(kThreads * kTwoTileOutputStride);
    constexpr float scale = 0.125f;

    for(ck_tile::index_t thread = 0; thread < kThreads; ++thread)
    {
        auto* lane_input        = input.data() + thread * kTwoTileInputStride;
        const auto lane         = thread % 32;
        const float initial_max = Masking ? -std::numeric_limits<float>::infinity()
                                          : -5.0f + 0.01f * static_cast<float>(lane & 15);

        for(ck_tile::index_t tile = 0; tile < kTwoTileCount; ++tile)
        {
            for(ck_tile::index_t offset = 0; offset < Mapping::kThreadBufferSize; ++offset)
            {
                const auto coordinate = Mapping::DecodeThreadBufferOffset(offset);
                lane_input[tile * Mapping::kThreadBufferSize + offset] =
                    Masking
                        ? -std::numeric_limits<float>::infinity()
                        : TwoTileScoreValue(
                              tile, lane, coordinate.msb, coordinate.pair * 2 + coordinate.element);
            }
        }
        for(ck_tile::index_t offset = 0; offset < Config::kOutputValues; ++offset)
        {
            lane_input[kTwoTileCount * Mapping::kThreadBufferSize + offset] =
                1.0f + 0.001f * static_cast<float>(offset);
        }
        lane_input[kTwoTileStateOffset + 0] = initial_max;
        lane_input[kTwoTileStateOffset + 1] = initial_max + (Masking ? 0.0f : 0.25f);
        lane_input[kTwoTileStateOffset + 2] = Masking ? 0.0f : 0.75f;
        lane_input[kTwoTileStateOffset + 3] = Masking ? 0.0f : 1.25f;
        lane_input[kTwoTileStateOffset + 4] = scale;
    }

    float* device_input  = nullptr;
    float* device_output = nullptr;
    CheckHip(hipMalloc(&device_input, input.size() * sizeof(float)), "hipMalloc input");
    CheckHip(hipMalloc(&device_output, output.size() * sizeof(float)), "hipMalloc output");
    CheckHip(
        hipMemcpy(device_input, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice),
        "hipMemcpy input");
    hipLaunchKernelGGL((RunSplitSoftmaxTwoTiles<Masking, Config>),
                       dim3(1),
                       dim3(kThreads),
                       0,
                       0,
                       device_input,
                       device_output);
    CheckHip(hipGetLastError(), "two-tile softmax probe launch");
    CheckHip(hipDeviceSynchronize(), "two-tile softmax probe synchronize");
    CheckHip(
        hipMemcpy(
            output.data(), device_output, output.size() * sizeof(float), hipMemcpyDeviceToHost),
        "hipMemcpy output");
    CheckHip(hipFree(device_input), "hipFree input");
    CheckHip(hipFree(device_output), "hipFree output");

    bool valid = true;
    for(ck_tile::index_t thread = 0; thread < kThreads; ++thread)
    {
        const auto lane       = thread % 32;
        const auto wave       = thread / 32;
        const auto peer       = wave * 32 + (lane ^ 16);
        const auto* lane_in   = input.data() + thread * kTwoTileInputStride;
        const auto* peer_in   = input.data() + peer * kTwoTileInputStride;
        const auto* lane_out  = output.data() + thread * kTwoTileOutputStride;
        float expected_max[2] = {lane_in[kTwoTileStateOffset + 0],
                                 lane_in[kTwoTileStateOffset + 1]};
        float expected_sum[2] = {lane_in[kTwoTileStateOffset + 2],
                                 lane_in[kTwoTileStateOffset + 3]};
        std::vector<float> expected_output(Config::kOutputValues);

        for(ck_tile::index_t offset = 0; offset < Config::kOutputValues; ++offset)
        {
            expected_output[offset] = lane_in[kTwoTileCount * Mapping::kThreadBufferSize + offset];
        }

        for(ck_tile::index_t tile = 0; tile < kTwoTileCount; ++tile)
        {
            for(ck_tile::index_t row = 0; row < 2; ++row)
            {
                const float old_max = expected_max[row];
                float new_max       = old_max;
                for(ck_tile::index_t msb = row * 2; msb < row * 2 + 2; ++msb)
                {
                    for(ck_tile::index_t pair = 0; pair < Mapping::kPairsPerMsb; ++pair)
                    {
                        for(ck_tile::index_t element = 0; element < 2; ++element)
                        {
                            const auto offset = Mapping::GetThreadBufferOffset(msb, pair, element);
                            const auto input_offset = tile * Mapping::kThreadBufferSize + offset;
                            new_max                 = std::max(new_max, lane_in[input_offset]);
                            new_max                 = std::max(new_max, peer_in[input_offset]);
                        }
                    }
                }

                const float exponent_max = Masking && std::isinf(new_max) ? 0.0f : new_max;
                const float delta        = std::fma(-exponent_max, scale, old_max * scale);
                float tile_sum           = 0.0f;
                for(ck_tile::index_t msb = row * 2; msb < row * 2 + 2; ++msb)
                {
                    for(ck_tile::index_t pair = 0; pair < Mapping::kPairsPerMsb; ++pair)
                    {
                        for(ck_tile::index_t element = 0; element < 2; ++element)
                        {
                            const auto offset = Mapping::GetThreadBufferOffset(msb, pair, element);
                            const auto input_offset = tile * Mapping::kThreadBufferSize + offset;
                            const float expected    = std::exp2(
                                std::fma(lane_in[input_offset], scale, -exponent_max * scale));
                            valid &=
                                Near(lane_out[kTwoTileScoreOutputOffset + input_offset], expected);
                            tile_sum += expected + std::exp2(std::fma(peer_in[input_offset],
                                                                      scale,
                                                                      -exponent_max * scale));
                        }
                    }
                }

                expected_sum[row] = std::fma(std::exp2(delta), expected_sum[row], tile_sum);
                expected_max[row] = new_max;
                for(ck_tile::index_t msb = row * 2; msb < row * 2 + 2; ++msb)
                {
                    for(ck_tile::index_t pair = 0; pair < OutputMapping::kPairsPerMsb; ++pair)
                    {
                        for(ck_tile::index_t element = 0; element < 2; ++element)
                        {
                            const auto offset =
                                OutputMapping::GetThreadBufferOffset(msb, pair, element);
                            expected_output[offset] *= std::exp2(delta);
                        }
                    }
                }
            }
        }

        for(ck_tile::index_t row = 0; row < 2; ++row)
        {
            valid &= Near(lane_out[row], expected_max[row]);
            valid &= Near(lane_out[2 + row], expected_sum[row]);
            const float expected_lse =
                expected_max[row] * scale / ck_tile::log2e_v<> + std::log(expected_sum[row]);
            valid &= Near(lane_out[4 + row], expected_lse);
        }
        for(ck_tile::index_t offset = 0; offset < Config::kOutputValues; ++offset)
        {
            valid &=
                Near(lane_out[kTwoTileOutputAccumulatorOffset + offset], expected_output[offset]);
        }
    }

    const auto* case_name = Masking ? "two_tile_masked_all_inf" : "two_tile_online";
    std::cout << case_name << ": " << (valid ? "pass" : "fail") << '\n';
    return valid;
}

} // namespace

using QrTdmSchedSoftmax = ck_tile::test::Gfx125FmhaTest;
TEST_F(QrTdmSchedSoftmax, DenseOnlineState)
{
    EXPECT_TRUE(RunCase<false>());
    EXPECT_TRUE(RunTwoTileCase<false>());
}
TEST_F(QrTdmSchedSoftmax, FullyMaskedOnlineState)
{
    EXPECT_TRUE(RunCase<true>());
    EXPECT_TRUE(RunTwoTileCase<true>());
}

template <typename Config>
class QrTdmSchedSoftmaxGeometry : public ck_tile::test::Gfx125FmhaTest
{
};

using AdditionalSoftmaxConfigs =
    ::testing::Types<Bf16D192N64, Fp16D192N64, Fp16D192N128, Bf16D64N64, Fp16D64N64>;
TYPED_TEST_SUITE(QrTdmSchedSoftmaxGeometry, AdditionalSoftmaxConfigs);

TYPED_TEST(QrTdmSchedSoftmaxGeometry, DenseOnlineState)
{
    EXPECT_TRUE((RunCase<false, TypeParam>()));
    EXPECT_TRUE((RunTwoTileCase<false, TypeParam>()));
}

TYPED_TEST(QrTdmSchedSoftmaxGeometry, FullyMaskedOnlineState)
{
    EXPECT_TRUE((RunCase<true, TypeParam>()));
    EXPECT_TRUE((RunTwoTileCase<true, TypeParam>()));
}
