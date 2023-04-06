// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/math.hpp"
#include "ck/utility/amd_gemm_dpp.hpp"

namespace ck {

enum struct DppGemmInstr
{
    dpp_f32_8x8x8_f16 = 0,
    dpp_i32_8x8x8_i8
};

template <DppGemmInstr Instr, index_t WaveSize, typename = void>
struct dpp_gemm_type
{
};

template <index_t WaveSize>
struct dpp_gemm_type<DppGemmInstr::dpp_f32_8x8x8_f16,
                     WaveSize,
                     typename std::enable_if_t<WaveSize == 64>>
{
    // * DPP GEMM setup
    static constexpr index_t waves_per_wg    = 4;
    static constexpr index_t m_per_dpp       = 8;
    static constexpr index_t n_per_dpp       = 8;
    static constexpr index_t k_per_dpp       = 8;
    static constexpr index_t dpp_per_wave    = 8;
    static constexpr index_t src_a_data_size = 2;
    static constexpr index_t src_b_data_size = 2;
    static constexpr index_t acc_data_size   = 4;
    static constexpr index_t wave_size       = WaveSize;
    // * Thread mapping inside wave, num_thread_per_subgroups always alone N direction
    static constexpr index_t num_thread_per_dpp = n_per_dpp;

    template <index_t MPerWave, index_t NPerWave, index_t KPerWave, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_dpp_f32_8x8x8_f16<MPerWave, NPerWave, KPerWave>::Run(a, b, reg_c);
    }
};

template <index_t WaveSize>
struct dpp_gemm_type<DppGemmInstr::dpp_i32_8x8x8_i8,
                 WaveSize,
                 typename std::enable_if_t<WaveSize == 64>>
{
    // * DPP GEMM setup
    static constexpr index_t waves_per_wg             = 4;
    static constexpr index_t m_per_dpp                = 8;
    static constexpr index_t n_per_dpp                = 8;
    static constexpr index_t k_per_dpp                = 8;
    static constexpr index_t dpp_per_wave             = 8;
    static constexpr index_t src_a_data_size          = 1;
    static constexpr index_t src_b_data_size          = 1;
    static constexpr index_t acc_data_size            = 4;
    static constexpr index_t wave_size                = WaveSize;
    // * Thread mapping inside wave, num_thread_per_dpp always alone N direction
    static constexpr index_t num_thread_per_dpp = n_per_dpp;

    template <index_t MPerWave,
              index_t NPerWave,
              index_t KPerWave,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_dpp_i32_8x8x8_i8<MPerWave, NPerWave, KPerWave>::Run(
            a, b, reg_c);
    }
};

template <typename src_type_a,
          typename src_type_b,
          typename dst_type,
          index_t MPerWave,
          index_t NPerWave,
          index_t KPerWave>
struct DppGemmSelector
{
    template <typename src_type_a_,
              typename src_type_b_,
              typename dst_type_,
              index_t MPerWave_,
              index_t NPerWave_,
              index_t KPerWave_>
    static constexpr auto GetDppGemm();

    template <>
    static constexpr auto GetDppGemm<half_t, half_t, float, 8, 8, 8>()
    {
        return DppGemmInstr::dpp_f32_8x8x8_f16;
    }

    template <>
    static constexpr auto GetDppGemm<int8_t, int8_t, int, 8, 8, 8>()
    {
        return DppGemmInstr::dpp_i32_8x8x8_i8;
    }

    // get_warp_size do not return the correct wavesize, hardcode to 32 as workaround
    static constexpr auto selected_dpp_gemm =
        dpp_gemm_type<GetDppGemm<src_type_a, src_type_b, dst_type, MPerWave, NPerWave, KPerWave>(), Number<64>{}>{};

    __host__ __device__ constexpr DppGemmSelector()
    {
        static_assert(selected_dpp_gemm.m_per_wave == 8, "Something went wrong, M per wave should be equal to 8");

        static_assert(selected_dpp_gemm.n_per_wave == 8, "Something went wrong, N per wave should be equal to 8");

        static_assert(selected_dpp_gemm.k_per_wave == 8, "Something went wrong, K per wave should be equal to 8");
    }
};

template <typename src_type_a,
          typename src_type_b,
          typename dst_type,
          index_t MPerWave,
          index_t NPerWave,
          index_t KPerWave,
          bool TransposeC = false>
struct DppGemm
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    // using CIndex   = MultiIndex<2>;
    // using CIndex4D = MultiIndex<4>;

    __host__ __device__ constexpr DppGemm()
    {
        static_assert(MPerWave == 8 && NPerWave == 8 && KPerWave == 8,
                      "Only MPerWave == 8, NPerWave == 8 and KPerWave == 8 are supported");
    }

    // DPP output supporting C = A * B
    // MPerDpp x KPerDpp @ KPerDpp x NPerDpp -> MPerDpp x NPerDpp
    template <typename CDesc_MBlockxRepeat_NBlockxRepeat_MWave_NWave_MPerDpp_NPerDpp>
    __host__ __device__ static constexpr auto
    CDesc_MBlockxRepeat_NBlockxRepeat_MWave_NWave_MSubGroup_NThreadPerSubgroup(
        const CDesc_MBlockxRepeat_NBlockxRepeat_MWave_NWave_MPerDpp_NPerDpp&
            c_desc_mblockxrepeat_nblockxRepeat_mwave_nwave_mperdpp_nperdpp)
    {
        const auto MBlockxRepeat =
            c_desc_mblockxrepeat_nblockxRepeat_mwave_nwave_mperdpp_nperdpp.GetLength(I0);
        const auto NBlockxRepeat =
            c_desc_mblockxrepeat_nblockxRepeat_mwave_nwave_mperdpp_nperdpp.GetLength(I1);
        const auto MWave =
            c_desc_mblockxrepeat_nblockxRepeat_mwave_nwave_mperdpp_nperdpp.GetLength(I2);
        const auto NWave =
            c_desc_mblockxrepeat_nblockxRepeat_mwave_nwave_mperdpp_nperdpp.GetLength(I3);

        return transform_tensor_descriptor(
            c_desc_mblockxrepeat_nblockxRepeat_mwave_nwave_mperdpp_nperdpp,
            make_tuple(
                make_pass_through_transform(MBlockxRepeat),
                make_pass_through_transform(NBlockxRepeat),
                make_pass_through_transform(MWave),
                make_pass_through_transform(NWave),
                make_pass_through_transform(Number<dpp_gemm_instr.dpp_per_wave>{}),
                make_pass_through_transform(Number<dpp_gemm_instr.num_thread_per_dpp>{})),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5>{}),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5>{}));
    }

    __device__ static constexpr index_t GetWaveSize() { return dpp_gemm_instr.wave_size; }

    template <class FloatA, class FloatB, class FloatC>
    __device__ void Run(const FloatA& p_a_wave, const FloatB& p_b_wave, FloatC& p_c_thread) const
    {
        static_assert(
            (is_same<src_type_a, half_t>::value && is_same<src_type_b, half_t>::value &&
             is_same<dst_type, float>::value) ||
            (is_same<src_type_a, int8_t>::value && is_same<src_type_b, int8_t>::value &&
            is_same<dst_type, int32_t>::value),
            "base type couple must be (half, float) "
            "or (int8, int32)!");
        if constexpr(!TransposeC)
        {
            dpp_gemm_instr.template run<MPerWave, NPerWave, KPerWave>(p_a_wave, p_b_wave, p_c_thread);
        }
        else 
        {
            dpp_gemm_instr.template run<MPerWave, NPerWave, KPerWave>(p_b_wave, p_a_wave, p_c_thread);
        }

    }

    __device__ static auto GetLaneId() { return get_thread_local_1d_id() % dpp_gemm_instr.wave_size; }

    __device__ static auto GetSubGroupId()
    {
        return (GetLaneId() / dpp_gemm_instr.num_thread_per_dpp) % dpp_gemm_instr.dpp_per_wave;
    }

    __device__ static auto GetLaneIdUnderSubGroup()
    {
        return GetLaneId() % dpp_gemm_instr.num_thread_per_dpp;
    }

    static constexpr auto dpp_gemm =
        DppGemmSelector<src_type_a, src_type_b, dst_type, MPerWave, NPerWave, KPerWave>{};
    static constexpr auto dpp_gemm_instr = dpp_gemm.selected_dpp_gemm;
};

} // namespace ck
