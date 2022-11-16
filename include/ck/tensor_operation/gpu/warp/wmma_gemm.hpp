// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/math.hpp"
#include "ck/utility/amd_wmma.hpp"

namespace ck {

enum struct WmmaInstr
{
    wmma_f32_16x16x16_f16_w32 = 0,
    wmma_f32_16x16x16_bf16_w32 = 0,
    wmma_f16_16x16x16_f16_w32 = 0,
    wmma_bf16_16x16x16_bf16_w32 = 0,
    wmma_i32_16x16x16_iu8_w32 = 0,
    wmma_i32_16x16x16_iu4_w32 = 0
};

template <WmmaInstr instr>
struct wmma_type;

template <>
struct wmma_type<WmmaInstr::wmma_f32_16x16x16_f16_w32>
{
    static constexpr index_t m_per_wave            = 16;
    static constexpr index_t n_per_wave            = 16;
    static constexpr index_t k_per_wave            = 16;
    static constexpr index_t wave_size             = 32;
    static constexpr index_t lane_size             = 16;
    static constexpr index_t src_data_size         = 2;
    static constexpr index_t acc_data_size         = 4;
    static constexpr index_t num_srcregs_per_wave  = 8;
    static constexpr index_t num_accregs_per_wave  = 8;

    template <index_t MPerWmma, index_t NPerWmma, class FloatA, class FloatB, class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_wmma_f32_16x16x16_f16_w32<MPerWmma, NPerWmma>::Run(a, b, reg_c);
    }
};

template <typename src_type, typename dst_type, index_t MPerWmma, index_t NPerWmma>
struct WmmaSelector
{
    template <typename src_type, typename dst_type, index_t MPerWmma_, index_t NPerWmma_>
    static constexpr auto GetWmma();

    template <>
    static constexpr auto GetWmma<half_t, float, 16, 16>()
    {
        return WmmaInstr::wmma_f32_16x16x16_f16_w32;
    }

    template <>
    static constexpr auto GetWmma<bhalf_t, float, 16, 16>()
    {
        return WmmaInstr::wmma_f32_16x16x16_bf16_w32;
    }

    template <>
    static constexpr auto GetWmma<half_t, half_t, 16, 16>()
    {
        return WmmaInstr::wmma_f16_16x16x16_f16_w32;
    }

    template <>
    static constexpr auto GetWmma<bhalf_t, bhalf_t, 16, 16>()
    {
        return WmmaInstr::wmma_bf16_16x16x16_bf16_w32;
    }

    template <>
    static constexpr auto GetWmma<int8_t, float, 16, 16>()
    {
        return WmmaInstr::wmma_i32_16x16x16_iu8_w32;
    }
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    template <>
    static constexpr auto GetWmma<int4_t, float, 16, 16>()
    {
        return WmmaInstr::wmma_i32_16x16x16_iu4_w32;
    }
#endif

    static constexpr auto selected_wmma = wmma_type<GetWmma<src_type, dst_type, MPerWmma, NPerWmma>()>{};

    __host__ __device__ constexpr WmmaSelector()
    {
        static_assert(selected_wmma.m_per_wave == selected_wmma.n_per_wave,
                      "WRONG! WMMA_M must equal to WMMA_N");
        
        static_assert(selected_wmma.m_per_wave == selected_wmma.k_per_wave,
                      "WRONG! WMMA_M must equal to WMMA_K");
        
        static_assert(selected_wmma.k_per_wave == 16,
                      "WRONG! WMMA_M must equal to WMMA_N");

        static_assert(selected_wmma.wave_size * selected_wmma.num_accregs_per_wave * selected_wmma.acc_data_size==
                      selected_wmma.m_per_wave * selected_wmma.n_per_wave * 4,
                      "WRONG! Number of Accumulator Register");

        static_assert(selected_wmma.lane_size * selected_wmma.num_srcregs_per_wave * selected_wmma.src_data_size==
                      selected_wmma.m_per_wave * selected_wmma.k_per_wave * 4,
                      "WRONG! Number of Source Register");
    }
};

template <typename src_type,
          typename dst_type,
          index_t MPerWmma,
          index_t NPerWmma,
          index_t KPack,
          bool TransposeC = false>
struct WmmaGemm
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    using CIndex   = MultiIndex<2>;
    using CIndex4D = MultiIndex<4>;

    __device__ static constexpr index_t GetNumBlks() { return wmma_instr.num_output_blks; }

    __device__ static constexpr index_t GetNumXdlops()
    {
        return MPerWmma * NPerWmma /
               (wmma_instr.m_per_blk * wmma_instr.n_per_blk * wmma_instr.num_output_blks);
    }

    __host__ __device__ constexpr WmmaGemm()
    {
        static_assert(NPerWmma == 16 && MPerWmma == 16 ,
                      "Only support GemmNPerWmma == 16 and GemmMPerWmma == 16 for wmma");

        static_assert(KPack % wmma_instr.k_per_wave == 0, "KPack cannot be divided by k_per_wave");
    }

    // XDL output supporting C = A * B
    // M2_N2 -> M2_M3_M4_N2
    template <typename CDesc_M0_N0_M1_N1_M2_N2>
    __host__ __device__ static constexpr auto
    MakeCDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(const CDesc_M0_N0_M1_N1_M2_N2& c_desc_m0_n0_m1_n1_m2_n2)
    {
        const auto M0 = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I0);
        const auto N0 = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I1);
        const auto M1 = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I2);
        const auto N1 = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I3);

        return transform_tensor_descriptor(
            c_desc_m0_n0_m1_n1_m2_n2,
            make_tuple(make_pass_through_transform(M0),
                       make_pass_through_transform(N0),
                       make_pass_through_transform(M1),
                       make_pass_through_transform(N1),
                       make_unmerge_transform(make_tuple(Number<wmma_instr.num_groups_per_blk>{},
                                                         Number<wmma_instr.num_input_blks>{},
                                                         Number<wmma_instr.group_size>{})),
                       make_pass_through_transform(Number<wmma_instr.num_threads_per_blk>{})),
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
                       Sequence<4, 5, 6>{},
                       Sequence<7>{}));
    }

    // transposed XDL output supporting C' = B' * A'
    // M2_N2 -> M2_N2_N3_N4
    template <typename CDesc_M0_N0_M1_N1_M2_N2>
    __host__ __device__ static constexpr auto
    MakeCDescriptor_M0_N0_M1_N1_M2_N2_N3_N4(const CDesc_M0_N0_M1_N1_M2_N2& c_desc_m0_n0_m1_n1_m2_n2)
    {
        const auto M0 = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I0);
        const auto N0 = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I1);
        const auto M1 = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I2);
        const auto N1 = c_desc_m0_n0_m1_n1_m2_n2.GetLength(I3);

        return transform_tensor_descriptor(
            c_desc_m0_n0_m1_n1_m2_n2,
            make_tuple(make_pass_through_transform(M0),
                       make_pass_through_transform(N0),
                       make_pass_through_transform(M1),
                       make_pass_through_transform(N1),
                       make_pass_through_transform(Number<wmma_instr.num_threads_per_blk>{}),
                       make_unmerge_transform(make_tuple(Number<wmma_instr.num_groups_per_blk>{},
                                                         Number<wmma_instr.num_input_blks>{},
                                                         Number<wmma_instr.group_size>{}))),
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
                       Sequence<5, 6, 7>{}));
    }

    template <typename CDesc_G_M0_N0_M1_N1_M2_N2>
    __host__ __device__ static constexpr auto MakeCDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2(
        const CDesc_G_M0_N0_M1_N1_M2_N2& c_desc_g_m0_n0_m1_n1_m2_n2)
    {
        const auto G  = c_desc_g_m0_n0_m1_n1_m2_n2.GetLength(I0);
        const auto M0 = c_desc_g_m0_n0_m1_n1_m2_n2.GetLength(I1);
        const auto N0 = c_desc_g_m0_n0_m1_n1_m2_n2.GetLength(I2);
        const auto M1 = c_desc_g_m0_n0_m1_n1_m2_n2.GetLength(I3);
        const auto N1 = c_desc_g_m0_n0_m1_n1_m2_n2.GetLength(I4);

        return transform_tensor_descriptor(
            c_desc_g_m0_n0_m1_n1_m2_n2,
            make_tuple(make_pass_through_transform(G),
                       make_pass_through_transform(M0),
                       make_pass_through_transform(N0),
                       make_pass_through_transform(M1),
                       make_pass_through_transform(N1),
                       make_unmerge_transform(make_tuple(wmma_instr.num_groups_per_blk,
                                                         wmma_instr.num_input_blks,
                                                         wmma_instr.group_size)),
                       make_pass_through_transform(wmma_instr.num_threads_per_blk)),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5>{},
                       Sequence<6>{}),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5, 6, 7>{},
                       Sequence<8>{}));
    }

    __device__ static constexpr index_t GetRegSizePerXdlops()
    {
        return MPerWmma * NPerWmma / wmma_instr.wave_size;
    }

    __device__ static constexpr index_t GetWaveSize() { return wmma_instr.wave_size; }

    template <class FloatA, class FloatB, class FloatC>
    __device__ void Run(const FloatA& p_a_wave, const FloatB& p_b_wave, FloatC& p_c_thread) const
    {
        static_assert((is_same<src_type, half_t>::value && is_same<dst_type, float>::value) || 
                      (is_same<src_type, bhalf_t>::value && is_same<dst_type, float>::value) ||
                      (is_same<src_type, half_t>::value && is_same<dst_type, half_t>::value) || 
                      (is_same<src_type, bhalf_t>::value && is_same<dst_type, bhalf_t>::value) || 
                      (is_same<src_type, int8_t>::value && is_same<dst_type, int32_t>::value) 
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
                      || (is_same<src_type, int4_t>::value && is_same<dst_type, int32_t>::value) 
#endif
                      ,
                      "base type couple must be (half, float), (bhalf, float), (half, half), 
                                                (bhalf, bhalf), (int8, int32) or (int4, int32)!");

        static_for<0, KPack / wmma_instr.k_per_wave, 1>{}([&](auto k) {
            if constexpr(!TransposeC)
            {
                wmma_instr.template run<MPerWmma, NPerWmma>(
                    p_a_wave[k], p_b_wave[k], p_c_thread);
            }
            else
            {
                wmma_instr.template run<MPerWmma, NPerWmma>(
                    p_b_wave[k], p_a_wave[k], p_c_thread);
            }
        });
    }

    __device__ static auto GetLaneId() { return get_thread_local_1d_id() % wmma_instr.wave_size; }

    __device__ static auto GetBlkIdx()
    {
        const auto laneId = GetLaneId();

        constexpr auto threadidx_to_blk_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(
                make_tuple(1, wmma_instr.num_input_blks, wmma_instr.num_threads_per_blk))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));

        const auto blk_idx =
            threadidx_to_blk_idx_adaptor.CalculateBottomIndex(make_multi_index(laneId));

        const auto blk_id = blk_idx[I1];
        const auto blk_td = blk_idx[I2];

        return make_tuple(blk_id, blk_td);
    }

    __host__ __device__ static auto CalculateAThreadOriginDataIndex()
    {
        const auto laneId  = GetLaneId();
        const auto blk_idx = GetBlkIdx();

        const auto blk_id = blk_idx[I0];
        const auto blk_td = blk_idx[I1];

        if constexpr(wmma_instr.is_k_reduction)
        {
            return make_tuple(blk_id, blk_td);
        }
        else
        {
            return make_tuple(0, laneId);
        }
    }

    __host__ __device__ static auto CalculateBThreadOriginDataIndex()
    {
        const auto laneId  = GetLaneId();
        const auto blk_idx = GetBlkIdx();

        const auto blk_id = blk_idx[I0];
        const auto blk_td = blk_idx[I1];

        if constexpr(wmma_instr.is_k_reduction)
        {
            return make_tuple(blk_id, blk_td);
        }
        else
        {
            return make_tuple(0, laneId);
        }
    }

    __device__ static CIndex GetBeginOfThreadBlk(index_t xdlops_i, index_t blk_i)
    {
        const auto blk_idx = GetBlkIdx();

        const auto blk_id = blk_idx[I0];
        const auto blk_td = blk_idx[I1];

        index_t n_offset = blk_i * wmma_instr.n_per_blk + blk_td;
        index_t m_offset = xdlops_i * wmma_instr.m_per_blk + blk_id * wmma_instr.group_size;

        return TransposeC ? CIndex{n_offset, m_offset} : CIndex{m_offset, n_offset};
    }

    __device__ static CIndex4D GetBeginOfThreadBlk4D(index_t /* xdlops_i */, index_t /* blk_i */)
    {
        const auto blk_idx = GetBlkIdx();

        const auto blk_id = blk_idx[I0];
        const auto blk_td = blk_idx[I1];

        return TransposeC ? CIndex4D{blk_td, I0, blk_id, I0} : CIndex4D{I0, blk_id, I0, blk_td};
    }

    static constexpr auto mfma = MfmaSelector<base_type, MPerWmma, NPerWmma>{};

    static constexpr auto wmma_instr = mfma.selected_mfma;

    static constexpr auto KPerXdlops  = mfma.GetKPerXdlops();
    static constexpr auto K1PerXdlops = mfma.GetK1PerXdlops();
    static constexpr auto K0PerXdlops = KPerXdlops / K1PerXdlops;

    __host__ __device__ static constexpr auto GetCM0M1M2NThreadBlkLengths()
    {
        return make_tuple(
            Number<wmma_instr.num_groups_per_blk>{}, I1, Number<wmma_instr.group_size>{}, I1);
    }
};

} // namespace ck
