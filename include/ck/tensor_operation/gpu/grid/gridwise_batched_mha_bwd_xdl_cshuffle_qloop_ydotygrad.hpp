// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/philox_rand.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v6r1.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_softmax.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_dropout.hpp"

namespace ck {

template <typename InputDataType,
          typename FloatD,
          typename YGridDesc_M_N,
          typename DGridDesc_M,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t NPadded>
struct GridwiseBatchedMultiheadAttentionBackward_YDotYGrad
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static constexpr auto WaveSize = 64;
    static_assert(BlockSize == MPerBlock, "BlockSize must be same with MPerBlock");

    // block_id to matrix tile idx (m0, n0) mapping are controlled by {M01, N01}
    template <typename Block2CTileMap>
    __host__ __device__ static constexpr bool CheckValidity(const YGridDesc_M_N& y_grid_desc_m_n,
                                                            const Block2CTileMap& block_2_ctile_map)
    {
        if(!block_2_ctile_map.CheckValidity(y_grid_desc_m_n))
        {
            return false;
        }
        const auto M = y_grid_desc_m_n.GetLength(I0);
        const auto N = y_grid_desc_m_n.GetLength(I1);

        if(N < NPerBlock)
        {
            return false;
        }

        if(M < MPerBlock)
        {
            return false;
        }
        if(M % MPerBlock != 0)
        {
            return false;
        }
        return true;
    }

    __host__ __device__ static constexpr auto
    MakeYGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(const YGridDesc_M_N& y_grid_desc_m_n)
    {
        const auto M = y_grid_desc_m_n.GetLength(I0);
        const auto N = y_grid_desc_m_n.GetLength(I1);

        const auto MBlock = M / MPerBlock;
        const auto NBlock = N / NPerBlock;

        const auto y_grid_desc_mblock_mperblock_nblock_nperblock = transform_tensor_descriptor(
            y_grid_desc_m_n,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{})),
                       make_unmerge_transform(make_tuple(NBlock, Number<NPerBlock>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));

        return y_grid_desc_mblock_mperblock_nblock_nperblock;
    }

    __host__ __device__ static constexpr auto
    MakeDGridDescriptor_MBlock_MPerBlock(const DGridDesc_M& d_grid_desc_m)
    {
        const index_t M      = d_grid_desc_m.GetLength(I0);
        const index_t MBlock = M / MPerBlock;

        const auto d_grid_desc_mblock_mperblock = transform_tensor_descriptor(
            d_grid_desc_m,
            make_tuple(make_unmerge_transform(make_tuple(MBlock, Number<MPerBlock>{}))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1>{}));

        return d_grid_desc_mblock_mperblock;
    }

    // return block_id to Y matrix tile idx (m0, n0) mapping
    __host__ __device__ static constexpr auto
    MakeDefaultBlock2CTileMap(const YGridDesc_M_N& y_grid_desc_m_n)
    {
        // should rewrite BlockToCTileMap_M00_N0_M01Adapt
        return BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPadded, YGridDesc_M_N>(y_grid_desc_m_n);
    }

    using YGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock = remove_cvref_t<decltype(
        MakeYGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(YGridDesc_M_N{}))>;

    using DefaultBlock2CTileMap =
        remove_cvref_t<decltype(MakeDefaultBlock2CTileMap(YGridDesc_M_N{}))>;

    template <index_t BlockSize_, index_t BlockSliceLength_M_, index_t BlockSliceLength_O_>
    struct YDotYGrad_M_N_
    {
        static_assert(BlockSize_ == BlockSliceLength_M_);
        static constexpr auto ThreadSliceLength_M   = Number<1>{};
        static constexpr index_t SrcScalarPerVector = 16 / sizeof(InputDataType);
        static constexpr auto ThreadClusterLength_O = Number<1>{};
        static constexpr auto ThreadClusterLength_M = Number<BlockSize_ / ThreadClusterLength_O>{};
        static constexpr auto ThreadSliceLength_O   = Number<BlockSliceLength_O_>{};

        static_assert(ThreadClusterLength_O * ThreadSliceLength_O == BlockSliceLength_O_, "");
        static_assert(ThreadClusterLength_M * ThreadSliceLength_M == BlockSliceLength_M_, "");

        using SrcBufType = StaticBuffer<AddressSpaceEnum::Vgpr,
                                        FloatD,
                                        ThreadSliceLength_M * ThreadSliceLength_O,
                                        true>;

        using DstBufType = StaticBuffer<AddressSpaceEnum::Vgpr, FloatD, ThreadSliceLength_M, true>;
    };
    using YDotYGrad_M_N = YDotYGrad_M_N_<BlockSize, MPerBlock, NPerBlock>;

    template <typename Block2CTileMap>
    __device__ static void Run(const InputDataType* __restrict__ p_y_grid,
                               const InputDataType* __restrict__ p_ygrad_grid,
                               FloatD* __restrict__ p_d_grid,
                               const YGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock&
                                   y_grid_desc_mblock_mperblock_nblock_nperblock,
                               const DGridDesc_M& d_grid_desc_m,
                               const Block2CTileMap& block_2_ctile_map,
                               const float p_drop)
    {
        const FloatD p_dropout = type_convert<FloatD>(1.0f - p_drop);
        const tensor_operation::element_wise::Scale scale_p_dropout(p_dropout);

        const auto y_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_y_grid, y_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());
        const auto ygrad_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_ygrad_grid, y_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

        auto d_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_d_grid, d_grid_desc_m.GetElementSpaceSize());

        // divide block work by [M, O]
        const auto block_work_idx =
            block_2_ctile_map.CalculateBottomIndex(make_multi_index(get_block_1d_id()));

        if(!block_2_ctile_map.ValidCTileIndex(
               block_work_idx,
               make_tuple(y_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I0),
                          y_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2))))
        {
            return;
        }

        const index_t block_work_idx_m = block_work_idx[I0];

        constexpr auto d_thread_desc_mblock_m1 =
            make_naive_tensor_descriptor_packed(make_tuple(I1, I1));

        constexpr auto y_thread_desc_m0_m1_n0_n1 = make_naive_tensor_descriptor_packed(make_tuple(
            I1, YDotYGrad_M_N::ThreadSliceLength_M, I1, YDotYGrad_M_N::ThreadSliceLength_O));

        constexpr auto y_thread_cluster_desc =
            make_cluster_descriptor(Sequence<I1,
                                             YDotYGrad_M_N::ThreadClusterLength_M,
                                             I1,
                                             YDotYGrad_M_N::ThreadClusterLength_O>{},
                                    Sequence<0, 1, 2, 3>{});
        const auto y_thread_cluster_idx =
            y_thread_cluster_desc.CalculateBottomIndex(make_multi_index(get_thread_local_1d_id()));

        const auto y_thread_data_on_block_idx =
            y_thread_cluster_idx * y_thread_desc_m0_m1_n0_n1.GetLengths();

        const auto y_thread_data_on_grid_idx =
            make_multi_index(
                block_work_idx_m, I0, I0 /* all WGs start from o_block_idx = 0 */, I0) +
            y_thread_data_on_block_idx;

        // performs double duty for both y and ygrad
        auto yygrad_threadwise_copy = ThreadwiseTensorSliceTransfer_v2<
            InputDataType,
            FloatD,
            YGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
            decltype(y_thread_desc_m0_m1_n0_n1),
            decltype(y_thread_desc_m0_m1_n0_n1.GetLengths()),
            Sequence<0, 1, 2, 3>,
            3,                                 // SrcVectorDim
            YDotYGrad_M_N::SrcScalarPerVector, // SrcScalarPerVector
            1,                                 // SrcScalarStrideInVector
            true /* ResetCoordAfterRun */,
            false /* InvalidElementAsNaN */>(y_grid_desc_mblock_mperblock_nblock_nperblock,
                                             y_thread_data_on_grid_idx);

        auto y_thread_buf                 = typename YDotYGrad_M_N::SrcBufType{};
        auto ygrad_thread_buf             = typename YDotYGrad_M_N::SrcBufType{};
        auto y_dot_ygrad_thread_accum_buf = typename YDotYGrad_M_N::DstBufType{};

        // clear accum buffers
        y_dot_ygrad_thread_accum_buf.Clear();

        index_t oblock_idx = 0;
        do
        {
            yygrad_threadwise_copy.Run(y_grid_desc_mblock_mperblock_nblock_nperblock,
                                       y_grid_buf,
                                       y_thread_desc_m0_m1_n0_n1,
                                       make_tuple(I0, I0, I0, I0),
                                       y_thread_buf);
            yygrad_threadwise_copy.Run(y_grid_desc_mblock_mperblock_nblock_nperblock,
                                       ygrad_grid_buf,
                                       y_thread_desc_m0_m1_n0_n1,
                                       make_tuple(I0, I0, I0, I0),
                                       ygrad_thread_buf);

            static_for<0, YDotYGrad_M_N::ThreadSliceLength_M, 1>{}([&](auto iM) {
                static_for<0, YDotYGrad_M_N::ThreadSliceLength_O, 1>{}([&](auto iO) {
                    constexpr auto offset =
                        y_thread_desc_m0_m1_n0_n1.CalculateOffset(make_multi_index(I0, iM, I0, iO));
                    y_dot_ygrad_thread_accum_buf(iM) +=
                        y_thread_buf[Number<offset>{}] * ygrad_thread_buf[Number<offset>{}];
                });
            });

            yygrad_threadwise_copy.MoveSrcSliceWindow(y_grid_desc_mblock_mperblock_nblock_nperblock,
                                                      make_multi_index(0, 0, 1, 0));

            oblock_idx++;
        } while(oblock_idx < y_grid_desc_mblock_mperblock_nblock_nperblock.GetLength(I2));

        auto d_grid_desc_mblock_mperblock = MakeDGridDescriptor_MBlock_MPerBlock(d_grid_desc_m);

        auto d_thread_copy_vgpr_to_global =
            ThreadwiseTensorSliceTransfer_v1r3<FloatD,
                                               FloatD,
                                               decltype(d_thread_desc_mblock_m1),
                                               decltype(d_grid_desc_mblock_mperblock),
                                               ck::tensor_operation::element_wise::Scale,
                                               Sequence<1, 1>,
                                               Sequence<0, 1>,
                                               1,
                                               1,
                                               InMemoryDataOperationEnum::Set,
                                               1,
                                               false>{
                d_grid_desc_mblock_mperblock,
                make_multi_index(block_work_idx_m,          // mblock
                                 get_thread_local_1d_id()), // mperblock
                scale_p_dropout};

        // copy from VGPR to Global
        d_thread_copy_vgpr_to_global.Run(d_thread_desc_mblock_m1,
                                         make_tuple(I0, I0),
                                         y_dot_ygrad_thread_accum_buf,
                                         d_grid_desc_mblock_mperblock,
                                         d_grid_buf);
    }
};

} // namespace ck
