// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/amd_gemm_dpp.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/utility/math.hpp"

namespace ck {

enum struct DppInstr
{
    dpp8_16x16x2 = 0,
    dpp8_8x32x2,
    dpp8_32x8x2
};

/**
 * Structure representing DPP GEMM executed by a single wavefront.
 *
 * Each structure instantiation must contain the following fields:
 * - wave_size - number of threads that execute a single DPP GEMM operation, usually equal to the
 *               number of threads in a wavefront;
 * - lanegroup_size - number of threads (lanes) that share data using DPP instruction modifier,
 *                    it's 8 in case of DPP8;
 * - m_per_wave - size along M dimension of matrix C that is processed in a single DPP GEMM
 *                operation;
 * - n_per_wave - size along N dimension of matrix C that is processed in a single DPP GEMM
 *                operation;
 * - m_per_lanegroup - size along M dimension that is processed by a single lanegroup;
 * - n_per_lanegroup - size along N dimension that is processed by a single lanegroup;
 * - m_per_thread - size along M dimension of the tile calculated by a single thread;
 * - n_per_thread - size along N dimension of the tile calculated by a single thread;
 * - k_per_dpp - size along K dimension that is reduced in a single DPP GEMM operation;
 * - share_a - indicates whether we share matrix A or matrix B between lanes using DPP modifiers.
 *
 * Not all the combinarions are supported now, for current restrictions see the static asserts
 * in the DppSelector's contructor.
 */
template <DppInstr instr>
struct dpp_type;

template <>
struct dpp_type<DppInstr::dpp8_32x8x2>
{
    static constexpr index_t wave_size       = 32;
    static constexpr index_t lanegroup_size  = 8;
    static constexpr index_t m_per_wave      = 32;
    static constexpr index_t n_per_wave      = 8;
    static constexpr index_t m_per_lanegroup = 8;
    static constexpr index_t n_per_lanegroup = 8;
    static constexpr index_t m_per_thread    = 8;
    static constexpr index_t n_per_thread    = 1;
    static constexpr index_t k_per_dpp       = 2;
    static constexpr bool share_a            = true;

    template <index_t MPerDpp, index_t NPerDpp, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        dpp8::RunGemm<m_per_lanegroup, n_per_lanegroup, k_per_dpp, FloatA, FloatB, FloatC, share_a>(
            a, b, reg_c);
    }
};

template <>
struct dpp_type<DppInstr::dpp8_8x32x2>
{
    static constexpr index_t wave_size       = 32;
    static constexpr index_t lanegroup_size  = 8;
    static constexpr index_t m_per_wave      = 8;
    static constexpr index_t n_per_wave      = 32;
    static constexpr index_t m_per_lanegroup = 8;
    static constexpr index_t n_per_lanegroup = 8;
    static constexpr index_t m_per_thread    = 8;
    static constexpr index_t n_per_thread    = 1;
    static constexpr index_t k_per_dpp       = 2;
    static constexpr bool share_a            = true;

    template <index_t MPerDpp, index_t NPerDpp, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        dpp8::RunGemm<m_per_lanegroup, n_per_lanegroup, k_per_dpp, FloatA, FloatB, FloatC, share_a>(
            a, b, reg_c);
    }
};

template <>
struct dpp_type<DppInstr::dpp8_16x16x2>
{
    static constexpr index_t wave_size       = 32;
    static constexpr index_t lanegroup_size  = 8;
    static constexpr index_t m_per_wave      = 16;
    static constexpr index_t n_per_wave      = 16;
    static constexpr index_t m_per_lanegroup = 8;
    static constexpr index_t n_per_lanegroup = 8;
    static constexpr index_t m_per_thread    = 8;
    static constexpr index_t n_per_thread    = 1;
    static constexpr index_t k_per_dpp       = 2;
    static constexpr bool share_a            = true;

    template <index_t MPerDpp, index_t NPerDpp, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        dpp8::RunGemm<m_per_lanegroup, n_per_lanegroup, k_per_dpp, FloatA, FloatB, FloatC, share_a>(
            a, b, reg_c);
    }
};

template <typename base_type, index_t MPerDpp, index_t NPerDpp>
struct DppSelector
{
    template <typename base_type_, index_t MPerDpp_, index_t NPerDpp_>
    static constexpr auto GetDpp();

    template <>
    static constexpr auto GetDpp<half_t, 8, 32>()
    {
        return DppInstr::dpp8_8x32x2;
    }

    template <>
    static constexpr auto GetDpp<half_t, 16, 16>()
    {
        return DppInstr::dpp8_16x16x2;
    }

    template <>
    static constexpr auto GetDpp<half_t, 32, 8>()
    {
        return DppInstr::dpp8_32x8x2;
    }

    static constexpr auto selected_dpp = dpp_type<GetDpp<base_type, MPerDpp, NPerDpp>()>{};

    __host__ __device__ constexpr DppSelector()
    {
        static_assert(selected_dpp.m_per_wave % selected_dpp.m_per_lanegroup == 0);
        static_assert(selected_dpp.n_per_wave % selected_dpp.n_per_lanegroup == 0);

        static_assert(selected_dpp.k_per_dpp % 2 == 0);

        static_assert(selected_dpp.wave_size % selected_dpp.lanegroup_size == 0);
        constexpr index_t num_dpp_per_wave = selected_dpp.wave_size / selected_dpp.lanegroup_size;
        constexpr index_t num_wave_c_elems = selected_dpp.m_per_wave * selected_dpp.n_per_wave;
        constexpr index_t num_dpp_c_elems =
            selected_dpp.m_per_lanegroup * selected_dpp.n_per_lanegroup;
        static_assert(num_wave_c_elems % num_dpp_c_elems == 0);
        static_assert(num_dpp_per_wave == num_wave_c_elems / num_dpp_c_elems);

        if constexpr(selected_dpp.share_a)
        {
            static_assert(selected_dpp.m_per_lanegroup == selected_dpp.m_per_thread);
            static_assert(selected_dpp.n_per_lanegroup % selected_dpp.n_per_thread == 0);
            static_assert(selected_dpp.n_per_lanegroup / selected_dpp.n_per_thread ==
                          selected_dpp.lanegroup_size);
        }
        else
        {
            static_assert(selected_dpp.m_per_lanegroup % selected_dpp.n_per_thread == 0);
            static_assert(selected_dpp.m_per_lanegroup / selected_dpp.n_per_thread ==
                          selected_dpp.lanegroup_size);
            static_assert(selected_dpp.n_per_lanegroup == selected_dpp.n_per_thread);
        }

        // Below checks come from the restrictions of the current implementation, could be removed
        // in the future when the implementation is more generalized.
        static_assert(selected_dpp.share_a);
        static_assert(selected_dpp.n_per_thread == 1);
        static_assert(selected_dpp.m_per_thread == selected_dpp.lanegroup_size);
        static_assert(selected_dpp.m_per_lanegroup == selected_dpp.m_per_thread);
        static_assert(selected_dpp.n_per_lanegroup ==
                      selected_dpp.n_per_thread * selected_dpp.lanegroup_size);
    }

    static constexpr index_t GetK1PerDpp() { return selected_dpp.k_per_dpp; }
};

template <typename base_type, index_t MPerDpp, index_t NPerDpp, index_t KPack>
struct DppGemm
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    using CIndex   = MultiIndex<2>;
    using CIndex4D = MultiIndex<4>;

    __host__ __device__ constexpr DppGemm()
    {
        static_assert(MPerDpp == 8 || MPerDpp == 16 || MPerDpp == 32,
                      "MPerDpp must be either 8, 16 or 32.");
        static_assert(NPerDpp == 8 || NPerDpp == 16 || NPerDpp == 32,
                      "NPerDpp must be either 8, 16 or 32.");

        static_assert(KPack % dpp_instr.k_per_dpp == 0, "KPack must be divisible by k_per_dpp.");
    }

    __device__ static constexpr index_t GetRegSizePerDpp()
    {
        return MPerDpp * NPerDpp / dpp_instr.wave_size;
    }

    template <class FloatA, class FloatB, class FloatC>
    __device__ void Run(const FloatA& p_a_wave, const FloatB& p_b_wave, FloatC& p_c_thread) const
    {
        static_assert(is_same<base_type, double>::value || is_same<base_type, float>::value ||
                          is_same<base_type, half_t>::value || is_same<base_type, bhalf_t>::value ||
                          is_same<base_type, int8_t>::value || is_same<base_type, f8_t>::value,
                      "base base_type must be double, float, half, bfloat16, and int8_t!");

        static_for<0, KPack / dpp_instr.k_per_dpp, 1>{}([&](auto k) {
            dpp_instr.template run<MPerDpp, NPerDpp>(p_a_wave[k], p_b_wave[k], p_c_thread);
        });
    }

    __device__ static auto GetLaneIdInWave()
    {
        return get_thread_local_1d_id() % dpp_instr.wave_size;
    }

    __device__ static auto GetWaveId() { return get_thread_local_1d_id() / dpp_instr.wave_size; }

    __device__ static auto GetLaneIdInLaneGroup()
    {
        return get_thread_local_1d_id() % dpp_instr.lanegroup_size;
    }

    __device__ static auto GetLaneGroupIdInWave()
    {
        return GetLaneIdInWave() / dpp_instr.lanegroup_size;
    }

    __device__ static auto GetDppOpIdx()
    {
        const auto lanegroupId = GetLaneGroupIdInWave();

        constexpr auto lanegroup_idx_1d_to_dpp_idx_2d_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(
                make_merge_transform(make_tuple(dpp_instr.m_per_wave / dpp_instr.m_per_lanegroup,
                                                dpp_instr.n_per_wave / dpp_instr.n_per_lanegroup))),
            make_tuple(Sequence<0, 1>{}),
            make_tuple(Sequence<0>{}));

        const auto dpp_idx = lanegroup_idx_1d_to_dpp_idx_2d_adaptor.CalculateBottomIndex(
            make_multi_index(lanegroupId));

        const auto m_dpp_idx = dpp_idx[I0];
        const auto n_dpp_idx = dpp_idx[I1];

        return make_tuple(m_dpp_idx, n_dpp_idx);
    }

    __host__ __device__ static auto CalculateAThreadOriginDataIndex_K_M()
    {
        const auto laneId   = get_thread_local_1d_id();
        const auto wave_row = laneId / dpp_instr.n_per_wave;
        auto m_idx          = dpp_instr.m_per_thread * wave_row + GetLaneIdInLaneGroup();
        return make_tuple(0, m_idx % dpp_instr.m_per_wave);
    }

    __host__ __device__ static auto CalculateBThreadOriginDataIndex_K_N()
    {
        const auto laneId = get_thread_local_1d_id();
        return make_tuple(0, laneId % dpp_instr.n_per_wave);
    }

    __device__ static CIndex GetBeginOfThreadBlk()
    {
        const auto dpp_op_idx = GetDppOpIdx();

        const auto m_dpp_op_idx = dpp_op_idx[I0];
        const auto n_dpp_op_idx = dpp_op_idx[I1];

        index_t n_offset = n_dpp_op_idx * dpp_instr.n_per_lanegroup + GetLaneIdInLaneGroup();
        index_t m_offset = m_dpp_op_idx * dpp_instr.m_per_lanegroup;

        return CIndex{m_offset, n_offset};
    }

    static constexpr auto dpp = DppSelector<base_type, MPerDpp, NPerDpp>{};

    static constexpr auto dpp_instr = dpp.selected_dpp;

    static constexpr auto K0PerDpp = 1;
    static constexpr auto K1PerDpp = dpp.GetK1PerDpp();

    __host__ __device__ static constexpr auto GetCMNThreadBlkLengths()
    {
        return make_tuple(Number<dpp_instr.m_per_thread>{}, Number<dpp_instr.n_per_thread>{});
    }
};

} // namespace ck
