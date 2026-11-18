// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "fmha_test_common.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_output.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_sched_policy.hpp"
#include "ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp"
#include "ck_tile/ops/gemm/warp/warp_wmma_gemm.hpp"

namespace {

using Policy = ck_tile::BlockFmhaPipelineQRKSVSTdmSchedPolicy<
    ck_tile::FmhaTdmSchedBf16D192N128Geometry,
    ck_tile::FmhaTdmSchedDerivedTuning<ck_tile::FmhaTdmSchedBf16D192N128Geometry>>;
using Mapping         = ck_tile::FmhaTdmSchedScoreFragmentMapping;
using Pair            = Mapping::Pair;
using OutputFragments = ck_tile::FmhaTdmSchedOutputFragments;

using SuGemmProblem =
    ck_tile::BlockGemmProblem<ck_tile::bf16_t,
                              ck_tile::bf16_t,
                              float,
                              128,
                              ck_tile::TileGemmShape<ck_tile::sequence<128, 32, 192>,
                                                     ck_tile::sequence<4, 1, 1>,
                                                     ck_tile::sequence<16, 16, 32>>>;
using SuWarpGemm   = ck_tile::WarpGemmWmma_f32_16x16x32_bf16_bf16<true>;
using SuGemmPolicy = ck_tile::BlockGemmARegBRegCRegV2CustomPolicy<ck_tile::bf16_t,
                                                                  ck_tile::bf16_t,
                                                                  float,
                                                                  ck_tile::sequence<4, 1, 1>,
                                                                  SuWarpGemm,
                                                                  ck_tile::GemmLoopOrder::MNK>;
using SuBlockGemm  = ck_tile::BlockGemmARegBRegCRegV2<SuGemmProblem, SuGemmPolicy>;
using OutputGemmProblem =
    ck_tile::BlockGemmProblem<ck_tile::bf16_t,
                              ck_tile::bf16_t,
                              float,
                              128,
                              ck_tile::TileGemmShape<ck_tile::sequence<128, 128, 32>,
                                                     ck_tile::sequence<4, 1, 1>,
                                                     ck_tile::sequence<16, 16, 32>>>;
using OutputBlockGemm = ck_tile::BlockGemmARegBRegCRegV2<OutputGemmProblem, SuGemmPolicy>;
using PvWarpGemm      = ck_tile::WarpGemmWmma_f32_16x16x32_bf16_bf16<true>;
using PvGemmPolicy    = ck_tile::BlockGemmARegBRegCRegV2CustomPolicy<ck_tile::bf16_t,
                                                                     ck_tile::bf16_t,
                                                                     float,
                                                                     ck_tile::sequence<4, 1, 1>,
                                                                     PvWarpGemm,
                                                                     ck_tile::GemmLoopOrder::KMN>;
using PvBlockGemm     = ck_tile::BlockGemmARegBRegCRegV2<OutputGemmProblem, PvGemmPolicy>;

constexpr ck_tile::index_t kNumOutputFragments  = OutputFragments::kNumFragments;
constexpr ck_tile::index_t kOutputFragmentSize  = OutputFragments::kElementsPerFragment;
constexpr ck_tile::index_t kOutputValuesPerLane = kNumOutputFragments * kOutputFragmentSize;

CK_TILE_DEVICE Pair AnchoredPackedMultiply(Pair input, Pair scale, float qk_dependency);
CK_TILE_DEVICE float AnchorWmmaAfter(float value, float output_dependency);
CK_TILE_DEVICE ck_tile::fp32x8_t AnchorFragmentForWmma(ck_tile::fp32x8_t fragment,
                                                       float output_dependency);

template <typename PBlockTensor, typename VBlockTensor>
CK_TILE_DEVICE void InitializePvInputs(PBlockTensor& p, VBlockTensor& v);

void CheckHip(hipError_t status, const char* operation);

template <ck_tile::index_t Ordinal>
CK_TILE_HOST_DEVICE constexpr float InitialOutputValue(ck_tile::index_t thread,
                                                       ck_tile::index_t element)
{
    return 0.125f + 0.002f * static_cast<float>(thread) + 0.01f * static_cast<float>(Ordinal) +
           0.0001f * static_cast<float>(element);
}

template <ck_tile::index_t Ordinal, typename OutputTensor>
CK_TILE_DEVICE void InitializeOutputFragment(OutputTensor& output)
{
    constexpr ck_tile::index_t d_msb = Ordinal / 4;
    constexpr ck_tile::index_t n     = Ordinal % 4;
    ck_tile::static_for<0, Mapping::kPairsPerFragment, 1>{}([&](auto pair) {
        constexpr ck_tile::index_t pair_index = n * Mapping::kPairsPerFragment + pair;
        constexpr ck_tile::index_t element    = 2 * decltype(pair)::value;
        Mapping::StorePair<d_msb, pair_index>(
            output,
            Pair{InitialOutputValue<Ordinal>(threadIdx.x, element),
                 InitialOutputValue<Ordinal>(threadIdx.x, element + 1)});
    });
}

template <ck_tile::index_t Ordinal,
          typename OutputTensor,
          typename PBlockTensor,
          typename VBlockTensor>
CK_TILE_DEVICE void
RunPvMonolithicWmma(OutputTensor& output, const PBlockTensor& p, const VBlockTensor& v)
{
    using WarpGemm    = typename PvBlockGemm::WarpGemm;
    using AWarpDstr   = typename WarpGemm::AWarpDstr;
    using BWarpDstr   = typename WarpGemm::BWarpDstr;
    using CWarpDstr   = typename WarpGemm::CWarpDstr;
    using AWarpTensor = typename WarpGemm::AWarpTensor;
    using BWarpTensor = typename WarpGemm::BWarpTensor;
    using CWarpTensor = typename WarpGemm::CWarpTensor;

    static_assert(Ordinal >= 0 && Ordinal < kNumOutputFragments);
    constexpr ck_tile::index_t d_msb       = Ordinal / OutputFragments::kNumN;
    constexpr ck_tile::index_t n           = Ordinal % OutputFragments::kNumN;
    constexpr ck_tile::index_t m_iter      = d_msb / 2;
    constexpr ck_tile::index_t v_msb       = d_msb % 2;
    constexpr ck_tile::index_t full_n_iter = n * 2 + v_msb;

    constexpr auto a_warp_y_lengths =
        ck_tile::to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    constexpr auto b_warp_y_lengths =
        ck_tile::to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    constexpr auto c_warp_y_lengths =
        ck_tile::to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    constexpr auto a_warp_y_index_zeros = ck_tile::uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
    constexpr auto b_warp_y_index_zeros = ck_tile::uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
    constexpr auto c_warp_y_index_zeros = ck_tile::uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

    AWarpTensor a_warp_tensor;
    a_warp_tensor.get_thread_buffer() = p.get_y_sliced_thread_data(
        ck_tile::merge_sequences(ck_tile::sequence<0, m_iter>{}, a_warp_y_index_zeros),
        ck_tile::merge_sequences(ck_tile::sequence<1, 1>{}, a_warp_y_lengths));

    BWarpTensor b_warp_tensor;
    b_warp_tensor.get_thread_buffer() = v.get_y_sliced_thread_data(
        ck_tile::merge_sequences(ck_tile::sequence<0, full_n_iter>{}, b_warp_y_index_zeros),
        ck_tile::merge_sequences(ck_tile::sequence<1, 1>{}, b_warp_y_lengths));

    CWarpTensor c_warp_tensor;
    c_warp_tensor.get_thread_buffer() = output.get_y_sliced_thread_data(
        ck_tile::merge_sequences(ck_tile::sequence<m_iter, full_n_iter>{}, c_warp_y_index_zeros),
        ck_tile::merge_sequences(ck_tile::sequence<1, 1>{}, c_warp_y_lengths));

    WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
    output.set_y_sliced_thread_data(
        ck_tile::merge_sequences(ck_tile::sequence<m_iter, full_n_iter>{}, c_warp_y_index_zeros),
        ck_tile::merge_sequences(ck_tile::sequence<1, 1>{}, c_warp_y_lengths),
        c_warp_tensor.get_thread_buffer());
}

template <ck_tile::index_t Ordinal, typename PBlockTensor, typename VBlockTensor>
CK_TILE_DEVICE ck_tile::fp32x8_t
RunPvFragmentWmma(ck_tile::fp32x8_t accumulator, const PBlockTensor& p, const VBlockTensor& v)
{
    using WarpGemm    = typename PvBlockGemm::WarpGemm;
    using AWarpDstr   = typename WarpGemm::AWarpDstr;
    using BWarpDstr   = typename WarpGemm::BWarpDstr;
    using AWarpTensor = typename WarpGemm::AWarpTensor;
    using BWarpTensor = typename WarpGemm::BWarpTensor;
    using CWarpTensor = typename WarpGemm::CWarpTensor;

    constexpr ck_tile::index_t d_msb       = Ordinal / 4;
    constexpr ck_tile::index_t n           = Ordinal % 4;
    constexpr ck_tile::index_t m_iter      = d_msb / 2;
    constexpr ck_tile::index_t v_msb       = d_msb % 2;
    constexpr ck_tile::index_t full_n_iter = n * 2 + v_msb;
    constexpr auto a_warp_y_lengths =
        ck_tile::to_sequence(AWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    constexpr auto b_warp_y_lengths =
        ck_tile::to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    constexpr auto a_warp_y_index_zeros = ck_tile::uniform_sequence_gen_t<AWarpDstr::NDimY, 0>{};
    constexpr auto b_warp_y_index_zeros = ck_tile::uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};

    AWarpTensor a_warp_tensor;
    a_warp_tensor.get_thread_buffer() = p.get_y_sliced_thread_data(
        ck_tile::merge_sequences(ck_tile::sequence<0, m_iter>{}, a_warp_y_index_zeros),
        ck_tile::merge_sequences(ck_tile::sequence<1, 1>{}, a_warp_y_lengths));
    BWarpTensor b_warp_tensor;
    b_warp_tensor.get_thread_buffer() = v.get_y_sliced_thread_data(
        ck_tile::merge_sequences(ck_tile::sequence<0, full_n_iter>{}, b_warp_y_index_zeros),
        ck_tile::merge_sequences(ck_tile::sequence<1, 1>{}, b_warp_y_lengths));
    CWarpTensor c_warp_tensor;
    c_warp_tensor.get_thread_buffer().template set_as<ck_tile::fp32x8_t>(ck_tile::number<0>{},
                                                                         accumulator);
    WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);
    return c_warp_tensor.get_thread_buffer().template get_as<ck_tile::fp32x8_t>(
        ck_tile::number<0>{});
}

CK_TILE_DEVICE ck_tile::fp32x8_t
RescaleFragmentScheduled(ck_tile::fp32x8_t input, float scale, float wmma_dependency)
{
    Pair scale_pair{scale, scale};
    ck_tile::fp32x8_t result;
    ck_tile::static_for<0, 4, 1>{}([&](auto pair) {
        constexpr ck_tile::index_t element = 2 * decltype(pair)::value;
        const Pair input_pair{input[element], input[element + 1]};
        const Pair output_pair = AnchoredPackedMultiply(input_pair, scale_pair, wmma_dependency);
        result[element]        = output_pair[0];
        result[element + 1]    = output_pair[1];
    });
    return result;
}

__global__ __launch_bounds__(128, 1) void D192PvMonolithicReference(float* output)
{
    constexpr auto p_distribution =
        ck_tile::make_static_tile_distribution(PvBlockGemm::MakeABlockDistributionEncode());
    constexpr auto v_distribution =
        ck_tile::make_static_tile_distribution(PvBlockGemm::MakeBBlockDistributionEncode());
    auto p = ck_tile::make_static_distributed_tensor<ck_tile::bf16_t>(p_distribution);
    auto v = ck_tile::make_static_distributed_tensor<ck_tile::bf16_t>(v_distribution);
    auto o = PvBlockGemm::MakeCBlockTile();
    InitializePvInputs(p, v);
    ck_tile::static_for<0, kNumOutputFragments, 1>{}(
        [&](auto ordinal) { InitializeOutputFragment<decltype(ordinal)::value>(o); });

    ck_tile::static_for<0, 8, 1>{}(
        [&](auto ordinal) { RunPvMonolithicWmma<decltype(ordinal)::value>(o, p, v); });
    // Preserve the WMMA/rescale boundary used by the scheduled kernel.
    __builtin_amdgcn_sched_barrier(0);
    ck_tile::FmhaTdmSchedSplitSoftmax::RescaleOutputTile<0, 0>(o, 0.5f);
    ck_tile::static_for<8, kNumOutputFragments, 1>{}(
        [&](auto ordinal) { RunPvMonolithicWmma<decltype(ordinal)::value>(o, p, v); });

    for(ck_tile::index_t i = 0; i < kOutputValuesPerLane; ++i)
    {
        output[threadIdx.x * kOutputValuesPerLane + i] = o.get_thread_buffer()[i];
    }
}

__global__ __launch_bounds__(128, 1) void D192PvFragmentRepresentation(float* output)
{
    constexpr auto p_distribution =
        ck_tile::make_static_tile_distribution(PvBlockGemm::MakeABlockDistributionEncode());
    constexpr auto v_distribution =
        ck_tile::make_static_tile_distribution(PvBlockGemm::MakeBBlockDistributionEncode());
    auto p = ck_tile::make_static_distributed_tensor<ck_tile::bf16_t>(p_distribution);
    auto v = ck_tile::make_static_distributed_tensor<ck_tile::bf16_t>(v_distribution);
    InitializePvInputs(p, v);

    auto fragments = ck_tile::generate_tuple(
        [&](auto ordinal) {
            ck_tile::fp32x8_t fragment;
            ck_tile::static_for<0, kOutputFragmentSize, 1>{}([&](auto element) {
                fragment[decltype(element)::value] = InitialOutputValue<decltype(ordinal)::value>(
                    threadIdx.x, decltype(element)::value);
            });
            return fragment;
        },
        ck_tile::number<kNumOutputFragments>{});

    ck_tile::static_for<0, 8, 1>{}([&](auto ordinal) {
        fragments.at(ordinal) =
            RunPvFragmentWmma<decltype(ordinal)::value, decltype(p), decltype(v)>(
                fragments.at(ordinal), p, v);
    });
    // Keep the representation comparison on the same instruction boundary.
    __builtin_amdgcn_sched_barrier(0);
    const float wmma_dependency = fragments.at(ck_tile::number<7>{})[0];
    fragments.at(ck_tile::number<0>{}) =
        RescaleFragmentScheduled(fragments.at(ck_tile::number<0>{}), 0.5f, wmma_dependency);
    const float output_dependency = fragments.at(ck_tile::number<0>{})[0];
    fragments.at(ck_tile::number<8>{}) =
        AnchorFragmentForWmma(fragments.at(ck_tile::number<8>{}), output_dependency);
    ck_tile::static_for<8, kNumOutputFragments, 1>{}([&](auto ordinal) {
        fragments.at(ordinal) =
            RunPvFragmentWmma<decltype(ordinal)::value, decltype(p), decltype(v)>(
                fragments.at(ordinal), p, v);
    });

    ck_tile::static_ford<ck_tile::sequence<kNumOutputFragments, kOutputFragmentSize>>{}(
        [&](auto indices) {
            constexpr ck_tile::index_t ordinal = indices[ck_tile::number<0>{}];
            constexpr ck_tile::index_t element = indices[ck_tile::number<1>{}];
            constexpr ck_tile::index_t offset =
                OutputFragments::GetThreadBufferOffset<ordinal, element>();
            output[threadIdx.x * kOutputValuesPerLane + offset] =
                fragments.at(ck_tile::number<ordinal>{})[element];
        });
}

__global__ __launch_bounds__(128, 1) void D192ZeroFragmentReconstruction(float* output)
{
    auto fragments = OutputFragments::MakeZero();
    auto o = OutputFragments::Reconstruct<decltype(PvBlockGemm::MakeCBlockTile())>(fragments);

    for(ck_tile::index_t i = 0; i < kOutputValuesPerLane; ++i)
    {
        output[threadIdx.x * kOutputValuesPerLane + i] = o.get_thread_buffer()[i];
    }
}

bool RunZeroFragmentReconstructionCase()
{
    constexpr ck_tile::index_t kThreads = 128;
    std::vector<float> output(kThreads * kOutputValuesPerLane);
    float* device_output = nullptr;
    CheckHip(hipMalloc(&device_output, output.size() * sizeof(float)),
             "hipMalloc zero reconstruction");
    hipLaunchKernelGGL(
        D192ZeroFragmentReconstruction, dim3(1), dim3(kThreads), 0, 0, device_output);
    CheckHip(hipGetLastError(), "zero reconstruction launch");
    CheckHip(hipDeviceSynchronize(), "zero reconstruction synchronize");
    CheckHip(
        hipMemcpy(
            output.data(), device_output, output.size() * sizeof(float), hipMemcpyDeviceToHost),
        "hipMemcpy zero reconstruction");
    CheckHip(hipFree(device_output), "hipFree zero reconstruction");

    const bool valid =
        std::all_of(output.begin(), output.end(), [](float value) { return value == 0.0f; });
    std::cout << "zero_fragment_reconstruction: " << (valid ? "pass" : "fail") << '\n';
    return valid;
}

bool RunPvFragmentRepresentationCase()
{
    constexpr ck_tile::index_t kThreads = 128;
    std::vector<float> reference(kThreads * kOutputValuesPerLane);
    std::vector<float> fragments(kThreads * kOutputValuesPerLane);
    float* device_reference = nullptr;
    float* device_fragments = nullptr;
    CheckHip(hipMalloc(&device_reference, reference.size() * sizeof(float)),
             "hipMalloc PV reference");
    CheckHip(hipMalloc(&device_fragments, fragments.size() * sizeof(float)),
             "hipMalloc PV fragments");
    hipLaunchKernelGGL(D192PvMonolithicReference, dim3(1), dim3(kThreads), 0, 0, device_reference);
    CheckHip(hipGetLastError(), "PV monolithic reference launch");
    hipLaunchKernelGGL(
        D192PvFragmentRepresentation, dim3(1), dim3(kThreads), 0, 0, device_fragments);
    CheckHip(hipGetLastError(), "PV fragment representation launch");
    CheckHip(hipDeviceSynchronize(), "PV fragment representation synchronize");
    CheckHip(hipMemcpy(reference.data(),
                       device_reference,
                       reference.size() * sizeof(float),
                       hipMemcpyDeviceToHost),
             "hipMemcpy PV reference");
    CheckHip(hipMemcpy(fragments.data(),
                       device_fragments,
                       fragments.size() * sizeof(float),
                       hipMemcpyDeviceToHost),
             "hipMemcpy PV fragments");
    CheckHip(hipFree(device_reference), "hipFree PV reference");
    CheckHip(hipFree(device_fragments), "hipFree PV fragments");

    bool valid                 = true;
    float max_error            = 0.0f;
    std::size_t first_mismatch = reference.size();
    for(std::size_t i = 0; i < reference.size(); ++i)
    {
        const float error = std::abs(reference[i] - fragments[i]);
        max_error         = std::max(max_error, error);
        const bool near   = error <= 1.0e-5f * std::max(1.0f, std::abs(reference[i]));
        if(!near && first_mismatch == reference.size())
        {
            first_mismatch = i;
        }
        valid &= near;
    }
    std::cout << "pv_fragment_representation: " << (valid ? "pass" : "fail")
              << " max_error=" << max_error;
    if(!valid)
    {
        std::cout << " first=" << first_mismatch << " reference=" << reference[first_mismatch]
                  << " fragment=" << fragments[first_mismatch];
    }
    std::cout << '\n';
    return valid;
}

CK_TILE_DEVICE ck_tile::fp32x8_t AnchorFragmentForWmma(ck_tile::fp32x8_t fragment,
                                                       float output_dependency)
{
    fragment[0] = AnchorWmmaAfter(fragment[0], output_dependency);
    return fragment;
}

template <typename PBlockTensor, typename VBlockTensor>
CK_TILE_DEVICE void InitializePvInputs(PBlockTensor& p, VBlockTensor& v)
{
    for(ck_tile::index_t i = 0; i < p.get_thread_buffer_size(); ++i)
    {
        p.get_thread_buffer()[i] = ck_tile::type_convert<ck_tile::bf16_t>(
            0.01f * static_cast<float>((threadIdx.x + i) % 17 + 1));
    }
    for(ck_tile::index_t i = 0; i < v.get_thread_buffer_size(); ++i)
    {
        v.get_thread_buffer()[i] = ck_tile::type_convert<ck_tile::bf16_t>(
            0.015f * static_cast<float>((threadIdx.x + 3 * i) % 19 + 1));
    }
}

CK_TILE_DEVICE Pair AnchoredPackedMultiply(Pair input, Pair scale, float qk_dependency)
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
    auto dependency = ck_tile::bit_cast<std::uint32_t>(qk_dependency);
    auto scale_bits = ck_tile::bit_cast<std::uint32_t>(scale[0]);
    std::uint32_t anchored_scale_bits;
    asm volatile("v_and_or_b32 %[dst], %[dependency], 0, %[scale]"
                 : [dst] "=v"(anchored_scale_bits)
                 : [dependency] "v"(dependency), [scale] "v"(scale_bits));
    const float anchored_scale = ck_tile::bit_cast<float>(anchored_scale_bits);
    const Pair anchored_pair{anchored_scale, anchored_scale};
    Pair result;
    asm volatile("v_pk_mul_f32 %[result], %[input], %[scale]"
                 : [result] "=v"(result)
                 : [input] "v"(input), [scale] "v"(anchored_pair));
    return result;
#else
    ck_tile::ignore = qk_dependency;
    return input * scale;
#endif
}

CK_TILE_DEVICE float AnchorWmmaAfter(float value, float output_dependency)
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
    auto dependency = ck_tile::bit_cast<std::uint32_t>(output_dependency);
    auto value_bits = ck_tile::bit_cast<std::uint32_t>(value);
    std::uint32_t anchored_value_bits;
    asm volatile("v_and_or_b32 %[dst], %[dependency], 0, %[value]"
                 : [dst] "=v"(anchored_value_bits)
                 : [dependency] "v"(dependency), [value] "v"(value_bits));
    return ck_tile::bit_cast<float>(anchored_value_bits);
#else
    ck_tile::ignore = output_dependency;
    return value;
#endif
}

__global__ __launch_bounds__(128, 1) void D192RescaleScheduleProbe(const float* input,
                                                                   float* output)
{
    constexpr auto a_distribution =
        ck_tile::make_static_tile_distribution(SuBlockGemm::MakeABlockDistributionEncode());
    constexpr auto b_distribution =
        ck_tile::make_static_tile_distribution(SuBlockGemm::MakeBBlockDistributionEncode());
    auto a = ck_tile::make_static_distributed_tensor<ck_tile::bf16_t>(a_distribution);
    auto b = ck_tile::make_static_distributed_tensor<ck_tile::bf16_t>(b_distribution);
    auto c = SuBlockGemm::MakeCBlockTile();
    auto o = OutputBlockGemm::MakeCBlockTile();

    for(ck_tile::index_t i = 0; i < a.get_thread_buffer_size(); ++i)
    {
        a.get_thread_buffer()[i] =
            ck_tile::type_convert<ck_tile::bf16_t>(0.01f * static_cast<float>(threadIdx.x + i + 1));
    }
    for(ck_tile::index_t i = 0; i < b.get_thread_buffer_size(); ++i)
    {
        b.get_thread_buffer()[i] =
            ck_tile::type_convert<ck_tile::bf16_t>(0.02f * static_cast<float>(threadIdx.x + i + 1));
    }
    ck_tile::clear_tile(c);
    ck_tile::clear_tile(o);
    ck_tile::static_for<0, Mapping::kPairsPerFragment, 1>{}([&](auto pair) {
        Mapping::StorePair<0, decltype(pair)::value>(
            o,
            Pair{input[threadIdx.x * 8 + 2 * decltype(pair)::value],
                 input[threadIdx.x * 8 + 2 * decltype(pair)::value + 1]});
    });

    Policy::RunQkSuWmma<0, SuBlockGemm>(c, a, b);
    const float qk_dependency = c.get_thread_buffer()[0];
    const Pair scale{0.5f, 0.5f};
    ck_tile::static_for<0, Mapping::kPairsPerFragment, 1>{}([&](auto pair) {
        constexpr ck_tile::index_t pair_index = decltype(pair)::value;
        Mapping::StorePair<0, pair_index>(
            o, AnchoredPackedMultiply(Mapping::LoadPair<0, pair_index>(o), scale, qk_dependency));
    });

    using WarpGemm    = typename SuBlockGemm::WarpGemm;
    using CWarpDstr   = typename WarpGemm::CWarpDstr;
    using CWarpTensor = typename WarpGemm::CWarpTensor;
    constexpr auto c_warp_y_lengths =
        ck_tile::to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    constexpr auto c_warp_y_index_zeros = ck_tile::uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};
    CWarpTensor c_warp_tensor;
    c_warp_tensor.get_thread_buffer() = c.get_y_sliced_thread_data(
        ck_tile::merge_sequences(ck_tile::sequence<1, 0>{}, c_warp_y_index_zeros),
        ck_tile::merge_sequences(ck_tile::sequence<1, 1>{}, c_warp_y_lengths));
    const auto output_dependency = Mapping::LoadPair<0, 0>(o);
    c_warp_tensor.get_thread_buffer()[0] =
        AnchorWmmaAfter(c_warp_tensor.get_thread_buffer()[0], output_dependency[0]);
    c.set_y_sliced_thread_data(
        ck_tile::merge_sequences(ck_tile::sequence<1, 0>{}, c_warp_y_index_zeros),
        ck_tile::merge_sequences(ck_tile::sequence<1, 1>{}, c_warp_y_lengths),
        c_warp_tensor.get_thread_buffer());
    Policy::RunQkSuWmma<1, SuBlockGemm>(c, a, b);

    ck_tile::static_for<0, Mapping::kPairsPerFragment, 1>{}([&](auto pair) {
        const auto value = Mapping::LoadPair<0, decltype(pair)::value>(o);
        output[threadIdx.x * 9 + 2 * decltype(pair)::value]     = value[0];
        output[threadIdx.x * 9 + 2 * decltype(pair)::value + 1] = value[1];
    });
    float qk_checksum = 0.0f;
    for(ck_tile::index_t i = 0; i < c.get_thread_buffer_size(); ++i)
    {
        qk_checksum += c.get_thread_buffer()[i];
    }
    output[threadIdx.x * 9 + 8] = qk_checksum;
}

void CheckHip(hipError_t status, const char* operation)
{
    if(status != hipSuccess)
    {
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
    }
}

} // namespace

using QrTdmSchedFragments = ck_tile::test::Gfx125FmhaTest;
TEST_F(QrTdmSchedFragments, PvFragmentRepresentation)
{
    EXPECT_TRUE(RunPvFragmentRepresentationCase());
}
TEST_F(QrTdmSchedFragments, ZeroFragmentReconstruction)
{
    EXPECT_TRUE(RunZeroFragmentReconstructionCase());
}
TEST_F(QrTdmSchedFragments, RescaleBetweenWmma)
{
    constexpr ck_tile::index_t threads = 128;
    std::vector<float> input(threads * 8), output(threads * 9);
    for(std::size_t i = 0; i < input.size(); ++i)
        input[i] = 0.25f + 0.001f * static_cast<float>(i);
    ck_tile::DeviceMem device_input(input.size() * sizeof(float));
    ck_tile::DeviceMem device_output(output.size() * sizeof(float));
    device_input.ToDevice(input.data());
    hipLaunchKernelGGL(D192RescaleScheduleProbe,
                       dim3(1),
                       dim3(threads),
                       0,
                       0,
                       static_cast<const float*>(device_input.GetDeviceBuffer()),
                       static_cast<float*>(device_output.GetDeviceBuffer()));
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    device_output.FromDevice(output.data());
    for(ck_tile::index_t thread = 0; thread < threads; ++thread)
    {
        for(ck_tile::index_t i = 0; i < 8; ++i)
            EXPECT_NEAR(output[thread * 9 + i], 0.5f * input[thread * 8 + i], 1.0e-6f);
        EXPECT_TRUE(std::isfinite(output[thread * 9 + 8]));
    }
}
