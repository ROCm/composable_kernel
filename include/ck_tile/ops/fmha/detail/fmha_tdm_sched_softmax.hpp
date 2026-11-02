// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <index_t NumSu>
struct FmhaTdmSchedScoreFragmentMappingFor
{
    static_assert(NumSu == 2 || NumSu == 4);
    using Pair = ext_vector_t<float, 2>;

    struct Coordinate
    {
        index_t msb;
        index_t su;
        index_t pair;
        index_t element;
    };

    static constexpr index_t kNumMIter         = 2;
    static constexpr index_t kNumNIter         = NumSu * 2;
    static constexpr index_t kNumMsb           = 4;
    static constexpr index_t kNumSu            = NumSu;
    static constexpr index_t kPairsPerFragment = 4;
    static constexpr index_t kElementsPerPair  = 2;
    static constexpr index_t kElementsPerWmma  = kPairsPerFragment * kElementsPerPair;
    static constexpr index_t kPairsPerMsb      = kNumSu * kPairsPerFragment;
    static constexpr index_t kElementsPerMsb   = kPairsPerMsb * kElementsPerPair;
    static constexpr index_t kThreadBufferSize = kNumMsb * kElementsPerMsb;
    static constexpr index_t kNIterPerSu       = 2;

    static_assert(kNumMIter * kNumNIter * kElementsPerWmma == kThreadBufferSize);
    static_assert(kNumSu * kNIterPerSu == kNumNIter);

    CK_TILE_HOST_DEVICE static constexpr index_t GetMIter(index_t msb) { return msb / 2; }

    CK_TILE_HOST_DEVICE static constexpr index_t GetNIterInSu(index_t msb) { return msb % 2; }

    CK_TILE_HOST_DEVICE static constexpr index_t GetFullNIter(index_t msb, index_t su)
    {
        return su * kNIterPerSu + GetNIterInSu(msb);
    }

    CK_TILE_HOST_DEVICE static constexpr index_t
    GetThreadBufferOffset(index_t msb, index_t pair, index_t element)
    {
        const index_t su               = pair / kPairsPerFragment;
        const index_t pair_in_fragment = pair % kPairsPerFragment;
        const index_t fragment         = GetMIter(msb) * kNumNIter + GetFullNIter(msb, su);
        return fragment * kElementsPerWmma + pair_in_fragment * kElementsPerPair + element;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetPairBufferOffset(index_t msb, index_t pair)
    {
        return GetThreadBufferOffset(msb, pair, 0) / kElementsPerPair;
    }

    template <index_t Msb, index_t PairIndex, typename ScoreTensor>
    CK_TILE_HOST_DEVICE static constexpr Pair LoadPair(const ScoreTensor& score)
    {
        static_assert(Msb >= 0 && Msb < kNumMsb);
        static_assert(PairIndex >= 0 && PairIndex < kPairsPerMsb);
        static_assert(std::is_same_v<remove_cvref_t<typename ScoreTensor::DataType>, float>);
        static_assert(ScoreTensor::get_thread_buffer_size() == kThreadBufferSize);

        constexpr index_t pair_offset = GetPairBufferOffset(Msb, PairIndex);
        return score.get_thread_buffer().template get_as<Pair>(number<pair_offset>{});
    }

    template <index_t Msb, index_t PairIndex, typename ScoreTensor>
    CK_TILE_HOST_DEVICE static constexpr void StorePair(ScoreTensor& score, const Pair& value)
    {
        static_assert(Msb >= 0 && Msb < kNumMsb);
        static_assert(PairIndex >= 0 && PairIndex < kPairsPerMsb);
        static_assert(std::is_same_v<remove_cvref_t<typename ScoreTensor::DataType>, float>);
        static_assert(ScoreTensor::get_thread_buffer_size() == kThreadBufferSize);

        constexpr index_t pair_offset = GetPairBufferOffset(Msb, PairIndex);
        score.get_thread_buffer().template set_as<Pair>(number<pair_offset>{}, value);
    }

    CK_TILE_HOST_DEVICE static constexpr Coordinate DecodeThreadBufferOffset(index_t offset)
    {
        const index_t fragment            = offset / kElementsPerWmma;
        const index_t element_in_fragment = offset % kElementsPerWmma;
        const index_t m_iter              = fragment / kNumNIter;
        const index_t full_n_iter         = fragment % kNumNIter;
        const index_t su                  = full_n_iter / kNIterPerSu;
        const index_t n_iter_in_su        = full_n_iter % kNIterPerSu;
        const index_t pair_in_fragment    = element_in_fragment / kElementsPerPair;

        return Coordinate{m_iter * 2 + n_iter_in_su,
                          su,
                          su * kPairsPerFragment + pair_in_fragment,
                          element_in_fragment % kElementsPerPair};
    }

    CK_TILE_HOST_DEVICE static constexpr bool IsValidCoordinate(const Coordinate& coordinate)
    {
        return coordinate.msb >= 0 && coordinate.msb < kNumMsb && coordinate.su >= 0 &&
               coordinate.su < kNumSu && coordinate.pair >= 0 && coordinate.pair < kPairsPerMsb &&
               coordinate.element >= 0 && coordinate.element < kElementsPerPair &&
               coordinate.su == coordinate.pair / kPairsPerFragment;
    }

    CK_TILE_HOST_DEVICE static constexpr bool ValidateBijection()
    {
        bool seen[kThreadBufferSize] = {};

        for(index_t msb = 0; msb < kNumMsb; ++msb)
        {
            for(index_t pair = 0; pair < kPairsPerMsb; ++pair)
            {
                for(index_t element = 0; element < kElementsPerPair; ++element)
                {
                    const index_t offset      = GetThreadBufferOffset(msb, pair, element);
                    const index_t pair_offset = GetPairBufferOffset(msb, pair);
                    if(offset < 0 || offset >= kThreadBufferSize || seen[offset])
                    {
                        return false;
                    }

                    if(offset / kElementsPerPair != pair_offset)
                    {
                        return false;
                    }

                    seen[offset]                = true;
                    const Coordinate coordinate = DecodeThreadBufferOffset(offset);
                    if(!IsValidCoordinate(coordinate) || coordinate.msb != msb ||
                       coordinate.pair != pair || coordinate.element != element)
                    {
                        return false;
                    }
                }
            }
        }

        for(bool visited : seen)
        {
            if(!visited)
            {
                return false;
            }
        }

        return true;
    }
};

using FmhaTdmSchedScoreFragmentMapping = FmhaTdmSchedScoreFragmentMappingFor<4>;
static_assert(FmhaTdmSchedScoreFragmentMappingFor<2>::ValidateBijection());
static_assert(FmhaTdmSchedScoreFragmentMapping::ValidateBijection());

template <index_t NumSu>
struct FmhaTdmSchedSplitSoftmaxFor
{
    using Mapping = FmhaTdmSchedScoreFragmentMappingFor<NumSu>;
    using Pair    = typename Mapping::Pair;

    struct Part0State
    {
        float local_max;
        float old_max_log2e;
    };

    struct Part1State
    {
        float row_max;
        float delta;
    };

    template <bool UseCompilerMax = false>
    CK_TILE_DEVICE static float Max3(float x, float y, float z)
    {
        if constexpr(UseCompilerMax)
        {
            return max(max(x, y), z);
        }
        else
        {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
            float result;
            asm volatile("v_max3_num_f32 %0, %1, %2, %3" : "=v"(result) : "v"(x), "v"(y), "v"(z));
            return result;
#else
            return max(max(x, y), z);
#endif
        }
    }

    CK_TILE_DEVICE static float PermuteLaneX16(float value)
    {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
        const auto bits = bit_cast<std::uint32_t>(value);
        return bit_cast<float>(
            __builtin_amdgcn_permlanex16(bits, bits, 0x76543210, 0xfedcba98, false, false));
#else
        return value;
#endif
    }

    template <index_t Msb, typename ScoreTensor, bool UseCompilerMax = false>
    CK_TILE_DEVICE static Part0State
    RunPart0(const ScoreTensor& score, float old_max, float log2e_scale)
    {
        array<float, Mapping::kNumSu> group_max{};

        static_for<0, Mapping::kNumSu, 1>{}([&](auto su) {
            constexpr index_t pair_base = su * Mapping::kPairsPerFragment;
            const auto pair0            = Mapping::template LoadPair<Msb, pair_base + 0>(score);
            const auto pair1            = Mapping::template LoadPair<Msb, pair_base + 1>(score);
            const auto pair2            = Mapping::template LoadPair<Msb, pair_base + 2>(score);
            const auto pair3            = Mapping::template LoadPair<Msb, pair_base + 3>(score);

            float value   = Max3<UseCompilerMax>(pair0[0], pair0[1], pair1[0]);
            value         = Max3<UseCompilerMax>(pair1[1], pair2[0], value);
            value         = Max3<UseCompilerMax>(pair2[1], pair3[0], value);
            group_max(su) = Max3<UseCompilerMax>(pair3[1], value, pair0[0]);
        });

        float lane_max;
        if constexpr(Mapping::kNumSu == 4)
        {
            lane_max = Max3<UseCompilerMax>(
                group_max[number<0>{}], group_max[number<1>{}], group_max[number<2>{}]);
            lane_max =
                Max3<UseCompilerMax>(lane_max, group_max[number<3>{}], group_max[number<1>{}]);
        }
        else
        {
            lane_max = max(group_max[number<0>{}], group_max[number<1>{}]);
        }
        lane_max = Max3<UseCompilerMax>(lane_max, PermuteLaneX16(lane_max), old_max);

        return Part0State{lane_max, old_max * log2e_scale};
    }

    template <bool ValidateMax = false>
    CK_TILE_DEVICE static Part1State
    RunPart1(float lhs_local_max, float rhs_local_max, float old_max_log2e, float log2e_scale)
    {
        const float row_max = max(lhs_local_max, rhs_local_max);
        const float exponent_max =
            ValidateMax && row_max <= -numeric<float>::infinity() ? 0.0f : row_max;
        return Part1State{row_max, __builtin_fmaf(-exponent_max, log2e_scale, old_max_log2e)};
    }

    CK_TILE_DEVICE static Pair Broadcast(float value) { return Pair{value, value}; }

    CK_TILE_DEVICE static float Exp2(float value)
    {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
        return __builtin_amdgcn_exp2f(value);
#else
        return exp2(value);
#endif
    }

    CK_TILE_DEVICE static Pair MakeExpPair(Pair score, float row_max, float log2e_scale)
    {
        const Pair delta = __builtin_elementwise_fma(
            score, Broadcast(log2e_scale), -Broadcast(row_max * log2e_scale));
        return Pair{Exp2(delta[0]), Exp2(delta[1])};
    }

    CK_TILE_DEVICE static Pair AddPair(Pair x, Pair y) { return x + y; }

    CK_TILE_DEVICE static Pair MulPair(Pair lhs, Pair rhs) { return lhs * rhs; }

    CK_TILE_DEVICE static float HorizontalAdd(Pair value) { return value[0] + value[1]; }

    template <index_t Msb, typename ScoreTensor>
    CK_TILE_DEVICE static float
    RunPart2LocalSum(ScoreTensor& score, float exponent_max, float log2e_scale)
    {
        static_assert(Msb >= 0 && Msb < Mapping::kNumMsb);
        array<Pair, Mapping::kPairsPerMsb> exp_pairs{};

        static_for<0, Mapping::kPairsPerMsb, 1>{}([&](auto pair) {
            exp_pairs(pair) = MakeExpPair(
                Mapping::template LoadPair<Msb, pair>(score), exponent_max, log2e_scale);
            Mapping::template StorePair<Msb, pair>(score, exp_pairs[pair]);
        });

        array<Pair, Mapping::kPairsPerMsb / 2> sum_l0{};
        static_for<0, Mapping::kPairsPerMsb / 2, 1>{}(
            [&](auto i) { sum_l0(i) = AddPair(exp_pairs[2 * i], exp_pairs[2 * i + 1]); });

        array<Pair, Mapping::kPairsPerMsb / 4> sum_l1{};
        static_for<0, Mapping::kPairsPerMsb / 4, 1>{}(
            [&](auto i) { sum_l1(i) = AddPair(sum_l0[2 * i], sum_l0[2 * i + 1]); });

        if constexpr(Mapping::kNumSu == 4)
        {
            array<Pair, Mapping::kPairsPerMsb / 8> sum_l2{};
            static_for<0, Mapping::kPairsPerMsb / 8, 1>{}(
                [&](auto i) { sum_l2(i) = AddPair(sum_l1[2 * i], sum_l1[2 * i + 1]); });
            return HorizontalAdd(AddPair(sum_l2[number<0>{}], sum_l2[number<1>{}]));
        }
        else
        {
            return HorizontalAdd(AddPair(sum_l1[number<0>{}], sum_l1[number<1>{}]));
        }
    }

    CK_TILE_DEVICE static float MergeRowSum(float lhs, float rhs)
    {
        const float lane_sum = lhs + rhs;
        return lane_sum + PermuteLaneX16(lane_sum);
    }

    CK_TILE_DEVICE static float UpdateRowSum(float old_row_sum, float local_row_sum, float delta)
    {
        return __builtin_fmaf(Exp2(delta), old_row_sum, local_row_sum);
    }

    template <index_t Msb, typename OutputTensor>
    CK_TILE_DEVICE static void RescaleOutput(OutputTensor& output, float scale)
    {
        static_assert(Msb >= 0 && Msb < Mapping::kNumMsb);
        static_for<0, Mapping::kNumSu, 1>{}([&](auto output_tile) {
            RescaleOutputTile<Msb, decltype(output_tile)::value>(output, scale);
        });
    }

    template <index_t Msb, index_t OutputTile, typename OutputTensor>
    CK_TILE_DEVICE static void RescaleOutputTile(OutputTensor& output, float scale)
    {
        static_assert(Msb >= 0 && Msb < Mapping::kNumMsb);
        static_assert(OutputTile >= 0 && OutputTile < Mapping::kNumSu);
        constexpr index_t pair_begin = OutputTile * Mapping::kPairsPerFragment;
        const Pair scale_pair        = Broadcast(scale);
        static_for<0, Mapping::kPairsPerFragment, 1>{}([&](auto pair) {
            constexpr index_t pair_index = pair_begin + decltype(pair)::value;
            Mapping::template StorePair<Msb, pair_index>(
                output, MulPair(Mapping::template LoadPair<Msb, pair_index>(output), scale_pair));
        });
    }
};

using FmhaTdmSchedSplitSoftmax = FmhaTdmSchedSplitSoftmaxFor<4>;

} // namespace ck_tile
