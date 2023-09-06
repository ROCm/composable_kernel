// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/philox_rand.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_dropout.hpp"

namespace ck {

template <typename ZDataType,
          typename GemmDataType,
          typename FloatGemmAcc,
          typename KGridDesc_N_K,
          typename ZGridDesc_M_N,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t Gemm1NPerBlock,
          index_t AK1Value,
          index_t BK1Value,
          index_t MPerXdl,
          index_t NPerXdl,
          index_t MXdlPerWave,
          index_t NXdlPerWave>
struct GridwiseBatchedDropout
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};

    static constexpr auto WaveSize = 64;

    // K1 should be Number<...>
    // Gemm0
    static constexpr auto AK0 = Number<KPerBlock / AK1Value>{};
    static constexpr auto BK0 = Number<KPerBlock / BK1Value>{};
    static constexpr auto AK1 = Number<AK1Value>{};
    static constexpr auto BK1 = Number<BK1Value>{};

    static constexpr auto Gemm0MWaves = MPerBlock / (MPerXdl * MXdlPerWave);
    static constexpr auto Gemm0NWaves = NPerBlock / (NPerXdl * NXdlPerWave);

    static constexpr auto mfma = MfmaSelector<GemmDataType, MPerXdl, NPerXdl>::selected_mfma;
    static constexpr auto DropoutNThread = mfma.num_input_blks; // 2
    // get_random_16x8() generates 16 random numbers each time
    static constexpr auto DropoutTile = Number<DropoutNThread * 16>{}; // 32

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    // C desc for source in blockwise copy
    __host__ __device__ static constexpr auto
    MakeCGridDescriptor_M0_N0_M1_N1_M2_N2_M3_M4_M5_N3(const ZGridDesc_M_N& z_grid_desc_m_n)
    {
        const auto M = z_grid_desc_m_n.GetLength(I0);
        const auto N = z_grid_desc_m_n.GetLength(I1);

        constexpr auto M3 = mfma.num_groups_per_blk;
        constexpr auto M4 = mfma.num_input_blks;
        constexpr auto M5 = mfma.group_size;

        return transform_tensor_descriptor(
            z_grid_desc_m_n,
            make_tuple(make_unmerge_transform(
                           make_tuple(M / MPerBlock, MXdlPerWave, Gemm0MWaves, M3, M4, M5)),
                       make_unmerge_transform(
                           make_tuple(N / NPerBlock, NXdlPerWave, Gemm0NWaves, NPerXdl))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2, 4, 6, 7, 8>{}, Sequence<1, 3, 5, 9>{}));
    }

    __host__ __device__ static constexpr auto GetPaddedSize(const index_t size)
    {
        return math::integer_divide_ceil(size, DropoutTile) * DropoutTile;
    }

    __device__ static auto GetGemm0WaveIdx()
    {
        const index_t thread_id = get_thread_local_1d_id();

        constexpr auto threadid_to_wave_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(Gemm0MWaves, Gemm0NWaves, WaveSize))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));

        return threadid_to_wave_idx_adaptor.CalculateBottomIndex(make_multi_index(thread_id));
    }

    __device__ static auto GetGemm0WaveMNIdx(const index_t thread_id)
    {
        constexpr auto wave_threadid_to_mn_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(WaveSize / MPerXdl, MPerXdl))),
            make_tuple(Sequence<0, 1>{}),
            make_tuple(Sequence<0>{}));

        return wave_threadid_to_mn_idx_adaptor.CalculateBottomIndex(make_multi_index(thread_id));
    }

    __host__ __device__ static constexpr auto GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1()
    {
        // A matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(make_tuple(AK0, Number<MPerBlock>{}, AK1),
                                            make_tuple(Number<MPerBlock + I1>{} * AK1, AK1, I1));
    }

    __host__ __device__ static constexpr auto GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1()
    {
        // B matrix in LDS memory, dst of blockwise copy
        return make_naive_tensor_descriptor(make_tuple(BK0, Number<NPerBlock>{}, BK1),
                                            make_tuple(Number<NPerBlock + I1>{} * BK1, BK1, I1));
    }

    __host__ __device__ static constexpr bool CheckValidity()
    {
        static_assert((MPerBlock % (MPerXdl * MXdlPerWave) == 0) &&
                          (NPerBlock % (NXdlPerWave * NPerXdl)) == 0,
                      "Invalid tuning param!");

        // TODO: also check validity of all components (blockwise-copy, threadwise-copy, etc)
        return true;
    }

    // return block_id to C matrix tile idx (m0, n0) mapping
    __host__ __device__ static constexpr auto
    MakeDefaultBlock2CTileMap(const KGridDesc_N_K& k_grid_desc_n_k)
    {
        return BlockToCTileMap_M00_N0_M01Adapt<NPerBlock, Gemm1NPerBlock, KGridDesc_N_K>(
            k_grid_desc_n_k);
    }

    using DefaultBlock2CTileMap =
        remove_cvref_t<decltype(MakeDefaultBlock2CTileMap(KGridDesc_N_K{}))>;

    using ZGridDescriptor_M0_N0_M1_N1_M2_N2_M3_M4_M5_N3 = remove_cvref_t<decltype(
        MakeCGridDescriptor_M0_N0_M1_N1_M2_N2_M3_M4_M5_N3(ZGridDesc_M_N{}))>;

    // S Gemm
    struct Gemm0
    {
        // A matrix in LDS memory, dst of blockwise copy
        static constexpr auto a_block_desc_ak0_m_ak1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1();

        // B matrix in LDS memory, dst of blockwise copy
        static constexpr auto b_block_desc_bk0_n_bk1 =
            GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1();

        template <typename ABlockDesc_AK0_M_AK1>
        __host__ __device__ static constexpr auto
        MakeGemm3AMmaTileDescriptor_M0_M1_M2_K(const ABlockDesc_AK0_M_AK1&)
        {
            constexpr index_t MWaves = MPerBlock / (MXdlPerWave * MPerXdl);

            return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<MXdlPerWave, MWaves, MPerXdl>(
                ABlockDesc_AK0_M_AK1{});
        }

        template <typename BBlockDesc_BK0_N_BK1>
        __host__ __device__ static constexpr auto
        MakeGemm3BMmaTileDescriptor_N0_N1_N2_K(const BBlockDesc_BK0_N_BK1&)
        {
            constexpr index_t NWaves = NPerBlock / (NXdlPerWave * NPerXdl);

            return MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K<NXdlPerWave, NWaves, NPerXdl>(
                BBlockDesc_BK0_N_BK1{});
        }

        static constexpr index_t KPack = math::max(math::lcm(AK1, BK1), mfma.k_per_blk);

        // Blockwise gemm with transposed XDL output
        using BlockwiseGemm = BlockwiseGemmXdlops_v2<
            BlockSize,
            GemmDataType,
            FloatGemmAcc,
            decltype(a_block_desc_ak0_m_ak1),
            decltype(b_block_desc_bk0_n_bk1),
            decltype(MakeGemm3AMmaTileDescriptor_M0_M1_M2_K(a_block_desc_ak0_m_ak1)),
            decltype(MakeGemm3BMmaTileDescriptor_N0_N1_N2_K(b_block_desc_bk0_n_bk1)),
            MPerBlock,
            NPerBlock,
            KPerBlock,
            MPerXdl,
            NPerXdl,
            MXdlPerWave,
            NXdlPerWave,
            KPack>;
    };

    template <typename Block2CTileMap>
    __device__ static void Run(ZDataType* __restrict__ p_z_grid,
                               const ZGridDescriptor_M0_N0_M1_N1_M2_N2_M3_M4_M5_N3&
                                   z_grid_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3,
                               const Block2CTileMap& block_2_ctile_map,
                               ck::philox& ph,
                               const index_t num_gemm0_m_block_outer_loop,
                               const index_t z_random_matrix_offset,
                               const index_t raw_n_padded)
    {
        // divide block work by [N, K]
        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        // HACK: this force n_block_data_idx_on_grid into SGPR
        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_work_idx[I0] * NPerBlock);

        // S: blockwise gemm
        auto s_blockwise_gemm = typename Gemm0::BlockwiseGemm{};

        auto s_slash_p_thread_buf = s_blockwise_gemm.GetCThreadBuffer();

        // get acc0 2D thread cluster & 2D thread slice
        constexpr auto thread_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
            s_blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();
        constexpr auto m0 = thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I0);
        constexpr auto n0 = thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I1);
        constexpr auto m1 = thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I2);
        constexpr auto n1 = thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I3);
        constexpr auto m2 = thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I4);
        constexpr auto m3 = thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I5);
        constexpr auto m4 = thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I6);
        constexpr auto n2 = thread_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I7);

        // only used for BlockwiseDropout
        constexpr auto thread_slice_desc_m_n =
            make_naive_tensor_descriptor_packed(make_tuple(m0 * m1 * m2 * m3 * m4, n0 * n1 * n2));

        // only used for providing ApplyDropoutAttnBwdSaveZ
        auto blockwise_dropout = BlockwiseDropout<FloatGemmAcc, decltype(thread_slice_desc_m_n)>{
            static_cast<unsigned short>(0.8f * 255.f), static_cast<FloatGemmAcc>(1.0f / 0.8f)};

        //
        // z vgpr copy to global
        //
        // z matrix threadwise desc
        constexpr auto z_thread_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3 =
            make_naive_tensor_descriptor_packed(make_tuple(I1,   // MBlockId
                                                           I1,   // NBlockID
                                                           m0,   // MRepeat
                                                           n0,   // NRepeat
                                                           m1,   // MWaveId
                                                           n1,   // NWaveId
                                                           m2,   // MGroupNum
                                                           m3,   // MInputNum
                                                           m4,   // RegisterNum
                                                           n2)); // NPerXdl

        StaticBuffer<AddressSpaceEnum::Vgpr,
                     uint8_t,
                     z_thread_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3.GetElementSpaceSize(),
                     true>
            z_tensor_buffer;
        z_tensor_buffer.Clear();

        auto z_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_z_grid, z_grid_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3.GetElementSpaceSize());

        const auto wave_id     = GetGemm0WaveIdx();
        const auto wave_m_n_id = GetGemm0WaveMNIdx(wave_id[I2]); // I2: 0~63

        auto z_thread_copy_vgpr_to_global = ThreadwiseTensorSliceTransfer_v1r3<
            uint8_t,
            ZDataType,
            decltype(z_thread_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3),
            decltype(z_grid_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3),
            tensor_operation::element_wise::PassThrough,
            Sequence<I1, // MBlockId
                     I1, // NBlockID
                     m0, // MRepeat
                     n0, // NRepeat
                     m1, // MWaveId
                     n1, // NWaveId
                     m2, // MPerXdl
                     m3, // NGroupNum
                     m4, // NInputNum
                     n2>,
            Sequence<0, 1, 2, 3, 4, 5, 6, 7, 8, 9>,
            9, // DstVectorDim
            1, // DstScalarPerVector
            InMemoryDataOperationEnum::Set,
            1, // DstScalarStrideInVector
            true>{z_grid_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3,
                  make_multi_index(num_gemm0_m_block_outer_loop - 1, // MBlockId
                                   block_work_idx[I0],               // NBlockId
                                   0,                                // MRepeat
                                   0,                                // NRepeat
                                   wave_id[I0],                      // MWaveId
                                   wave_id[I1],                      // NWaveId
                                   0,                                // MPerXdl
                                   wave_m_n_id[I0],                  //
                                   0,                                //
                                   wave_m_n_id[I1]),                 // NPerXdl
                  tensor_operation::element_wise::PassThrough{}};

        // 8d thread_desc in thread scope
        constexpr auto c_thread_lengths =
            s_blockwise_gemm.GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2().GetLengths();

        // 8d block_desc in block scope
        constexpr auto c_block_lengths =
            s_blockwise_gemm.GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2().GetLengths();

        constexpr auto M0 = c_block_lengths[I0];
        constexpr auto N0 = c_block_lengths[I1];
        constexpr auto M1 = c_block_lengths[I2];
        constexpr auto N1 = c_block_lengths[I3];
        constexpr auto M2 = c_block_lengths[I4];
        constexpr auto M3 = c_block_lengths[I5];
        constexpr auto M4 = c_block_lengths[I6];
        constexpr auto N2 = c_block_lengths[I7];

        // works like multi-dimension static_for (static_ford), but provides both the linear
        // index as well as n-d index
        using Acc0TileIterator =
            SpaceFillingCurve<decltype(c_thread_lengths),
                              typename arithmetic_sequence_gen<0, c_thread_lengths.Size(), 1>::type,
                              typename uniform_sequence_gen<c_thread_lengths.Size(), 1>::type,
                              false>; // SnakeCurved

        constexpr auto block_idx_to_m_n_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_unmerge_transform(make_tuple(M0, M1, M2, M3, M4)),
                       make_unmerge_transform(make_tuple(N0, N1, N2))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2, 4, 5, 6>{}, Sequence<1, 3, 7>{}));

        auto acc0_thread_origin = s_blockwise_gemm.CalculateCThreadOriginDataIndex8D(
            Number<0>{}, Number<0>{}, Number<0>{}, Number<0>{});

        auto acc0_thread_idx = Acc0TileIterator::GetIndex(I0) + acc0_thread_origin;
        auto m_local         = block_idx_to_m_n_adaptor.CalculateBottomIndex(acc0_thread_idx)[I0];
        auto n_local         = block_idx_to_m_n_adaptor.CalculateBottomIndex(acc0_thread_idx)[I1];

        // gemm0 M loop
        index_t gemm0_m_block_outer_index = num_gemm0_m_block_outer_loop - 1;

        do
        {
            auto m_block_data_idx_on_grid =
                __builtin_amdgcn_readfirstlane(gemm0_m_block_outer_index * MPerBlock);

            // save z to global
            auto m_global = m_local + m_block_data_idx_on_grid;
            auto n_global = n_local + n_block_data_idx_on_grid;

            auto global_tile_id = z_random_matrix_offset +
                                  (m_global / DropoutTile) * DropoutTile * raw_n_padded +
                                  (n_global / DropoutTile) * DropoutTile;

            auto global_elem_id =
                global_tile_id + (wave_m_n_id[I0] * M4) + (n_global % DropoutTile) * raw_n_padded;

            blockwise_dropout.template ApplyDropoutAttnBwdSaveZ<decltype(s_slash_p_thread_buf),
                                                                decltype(z_tensor_buffer),
                                                                decltype(DropoutTile),
                                                                true>(
                s_slash_p_thread_buf, ph, global_elem_id, z_tensor_buffer, raw_n_padded);

            z_thread_copy_vgpr_to_global.Run(z_thread_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3,
                                             make_tuple(I0, I0, I0, I0, I0, I0, I0, I0, I0, I0),
                                             z_tensor_buffer,
                                             z_grid_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3,
                                             z_grid_buf);

            // move slice window
            z_thread_copy_vgpr_to_global.MoveDstSliceWindow(
                z_grid_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3,
                make_multi_index(-1, 0, 0, 0, 0, 0, 0, 0, 0, 0));
        } while(0 < gemm0_m_block_outer_index--); // end j loop
    };
};

} // namespace ck
