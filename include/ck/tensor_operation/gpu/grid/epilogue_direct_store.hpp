// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"

namespace ck {

template <typename DsDataType,
          typename EDataType,
          typename AccDataType,
          index_t MRepeat,
          index_t NRepeat,
          typename CDEElementwiseOperation,
          typename BlockwiseGemmPipe>
struct EpilogueDirectStore
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};

    __device__ static constexpr bool IsLDSNeeded() { return false; }

    template <InMemoryDataOperationEnum EGlobalMemoryDataOperation,
              typename CThreadBuf,
              typename DsGridPointer,
              typename DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
              typename EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>
    __device__ static void Run(CThreadBuf& c_thread_buf,
                               DsGridPointer,
                               EDataType* p_e_grid,
                               void*,
                               const DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock&,
                               const EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock&
                                   e_grid_desc_mblock_mperblock_nblock_nperblock,
                               CDEElementwiseOperation& cde_element_op,
                               const index_t& block_m_id,
                               const index_t& block_n_id)
    {
        auto e_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_e_grid, e_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

        // C mapping in single thread.
        constexpr auto c_thread_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs =
            BlockwiseGemmPipe::
                GetCThreadDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs();

        // C mapping in single block
        constexpr auto c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp =
            BlockwiseGemmPipe::
                GetCBlockDescriptor_MRepeat_MWave_MSubGroup_NRepeat_NWave_NThreadPerSubGroup_MAccVgprs();

        constexpr auto MWave =
            c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp
                .GetLength(I1);
        constexpr auto MSubGroup =
            c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp
                .GetLength(I2);
        constexpr auto NWave =
            c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp
                .GetLength(I4);
        constexpr auto NThreadPerSubGroup =
            c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp
                .GetLength(I5);
        constexpr auto MAccVgprs =
            c_block_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs_tmp
                .GetLength(I6);

        // origin
        const auto c_thread_mtx_on_block =
            BlockwiseGemmPipe::CalculateCThreadOriginDataIndex(I0, I0);

        const auto m_thread_data_on_grid_to_mrepeat_mwave_msubgroup_maccvgprs_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(MRepeat, MWave, MSubGroup, MAccVgprs))),
                make_tuple(Sequence<0, 1, 2, 3>{}),
                make_tuple(Sequence<0>{}));

        const auto m_thread_data_on_grid_idx =
            m_thread_data_on_grid_to_mrepeat_mwave_msubgroup_maccvgprs_adaptor.CalculateBottomIndex(
                make_multi_index(c_thread_mtx_on_block[I0]));

        const auto n_thread_data_on_grid_to_nrepeat_nwave_nthreadpersubgroup_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(NRepeat, NWave, NThreadPerSubGroup))),
                make_tuple(Sequence<0, 1, 2>{}),
                make_tuple(Sequence<0>{}));

        const auto n_thread_data_on_grid_idx =
            n_thread_data_on_grid_to_nrepeat_nwave_nthreadpersubgroup_adaptor.CalculateBottomIndex(
                make_multi_index(c_thread_mtx_on_block[I1]));

        // E grid descriptor
        const auto c_grid_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs =
            transform_tensor_descriptor(
                e_grid_desc_mblock_mperblock_nblock_nperblock,
                make_tuple(make_freeze_transform(block_m_id),
                           make_unmerge_transform(make_tuple(Number<MRepeat>{},
                                                             Number<MWave>{},
                                                             Number<MSubGroup>{},
                                                             Number<MAccVgprs>{})),
                           make_freeze_transform(block_n_id),
                           make_unmerge_transform(make_tuple(
                               Number<NWave>{}, Number<NThreadPerSubGroup>{}, Number<NRepeat>{}))),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(
                    Sequence<>{}, Sequence<0, 1, 2, 6>{}, Sequence<>{}, Sequence<4, 5, 3>{}));

        auto c_thread_copy = ThreadwiseTensorSliceTransfer_v1r3<
            AccDataType,
            EDataType,
            decltype(c_thread_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs),
            decltype(c_grid_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs),
            CDEElementwiseOperation,
            Sequence<MRepeat, I1, I1, NRepeat, I1, I1, MAccVgprs>,
            Sequence<0, 1, 2, 3, 4, 5, 6>,
            3,
            NRepeat, // VectorSize
            EGlobalMemoryDataOperation,
            1,
            false>{c_grid_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs,
                   make_multi_index(m_thread_data_on_grid_idx[I0],
                                    m_thread_data_on_grid_idx[I1],
                                    m_thread_data_on_grid_idx[I2],
                                    n_thread_data_on_grid_idx[I0],
                                    n_thread_data_on_grid_idx[I1],
                                    n_thread_data_on_grid_idx[I2],
                                    m_thread_data_on_grid_idx[I3]),
                   cde_element_op};

        c_thread_copy.Run(
            c_thread_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs,
            make_tuple(I0, I0, I0, I0, I0, I0, I0),
            c_thread_buf,
            c_grid_desc_mrepeat_mwave_msubgroup_nrepeat_nwave_nthreadpersubgroup_maccvgprs,
            e_grid_buf);
    }
};

} // namespace ck
