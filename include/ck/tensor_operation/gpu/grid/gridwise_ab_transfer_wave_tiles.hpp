// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/amd_address_space.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_global.hpp"

namespace ck {

template <typename ABLayout,
          typename ABMajorLayout,
          index_t BlockSize,
          index_t MNPerBlock,
          index_t KPerBlock,
          index_t MNPerWmma,
          index_t KPack,
          index_t ABK1Value,
          index_t WaveSize>
struct ABTransferWaveTiles
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;

    static constexpr index_t NumberOfWaves = BlockSize / WaveSize;
    static constexpr index_t MNMajorWaves_ = std::min(MNPerBlock / MNPerWmma, NumberOfWaves);
    static constexpr index_t KMajorWaves_  = std::min(KPerBlock / KPack, NumberOfWaves);

    static constexpr bool ABDoTranspose = !is_same_v<ABLayout, ABMajorLayout>;

    static constexpr index_t MNWaves_ =
        ABDoTranspose ? NumberOfWaves / KMajorWaves_ : MNMajorWaves_;
    static constexpr index_t KWaves_ = ABDoTranspose ? KMajorWaves_ : NumberOfWaves / MNMajorWaves_;
    static constexpr index_t KRepeat_  = KPerBlock / (KWaves_ * KPack);
    static constexpr index_t MNRepeat_ = MNPerBlock / (MNWaves_ * MNPerWmma);

    template <bool PadMN, bool PadK, typename GridDescriptorBase>
    __host__ __device__ static auto MakeGridDescriptor(GridDescriptorBase& base_desc,
                                                       index_t sizeMN,
                                                       index_t,
                                                       index_t sizeK,
                                                       index_t,
                                                       index_t,
                                                       index_t)
    {
        // Notes: padding is currently not supported
        const auto ab_grid_desc_mnblock_kblock = transform_tensor_descriptor(
            base_desc,
            make_tuple(
                make_unmerge_transform(
                    make_tuple(sizeMN / Number<MNPerWmma>{}, Number<MNPerWmma>{})),
                make_unmerge_transform(make_tuple(sizeK / Number<KPack>{}, Number<KPack>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1, 3>{}));

        if constexpr(is_same_v<ABMajorLayout, ABLayout>)
        {
            const auto ab_grid_desc_mnrepeat_krepeat_kgroup_mnwmma_abk1 =
                transform_tensor_descriptor(
                    ab_grid_desc_mnblock_kblock,
                    make_tuple(make_pass_through_transform(sizeMN / Number<MNPerWmma>{}),
                               make_pass_through_transform(sizeK / Number<KPack>{}),
                               make_unmerge_transform(make_tuple(Number<MNPerWmma / ABK1Value>{},
                                                                 Number<ABK1Value>{})),
                               make_unmerge_transform(
                                   make_tuple(Number<KPack / ABK1Value>{}, Number<ABK1Value>{}))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 4>{}, Sequence<3, 5>{}));

            return transform_tensor_descriptor(
                ab_grid_desc_mnrepeat_krepeat_kgroup_mnwmma_abk1,
                make_tuple(make_pass_through_transform(sizeMN / Number<MNPerWmma>{}),
                           make_pass_through_transform(sizeK / Number<KPack>{}),
                           make_merge_transform(make_tuple(Number<MNPerWmma / ABK1Value>{},
                                                           Number<KPack / ABK1Value>{})),
                           make_pass_through_transform(Number<ABK1Value>{}),
                           make_freeze_transform(I0)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}, Sequence<5>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<>{}));
        }
        else
        {
            const auto ab_grid_desc_mnrepeat_krepeat_kgroup_mnwmma_abk1 =
                transform_tensor_descriptor(
                    ab_grid_desc_mnblock_kblock,
                    make_tuple(make_pass_through_transform(sizeMN / Number<MNPerWmma>{}),
                               make_pass_through_transform(sizeK / Number<KPack>{}),
                               make_unmerge_transform(
                                   make_tuple(Number<KPack / ABK1Value>{}, Number<ABK1Value>{})),
                               make_unmerge_transform(make_tuple(Number<MNPerWmma / ABK1Value>{},
                                                                 Number<ABK1Value>{}))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<3>{}, Sequence<2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 4>{}, Sequence<3, 5>{}));

            return transform_tensor_descriptor(
                ab_grid_desc_mnrepeat_krepeat_kgroup_mnwmma_abk1,
                make_tuple(make_pass_through_transform(sizeMN / Number<MNPerWmma>{}),
                           make_pass_through_transform(sizeK / Number<KPack>{}),
                           make_merge_transform(make_tuple(Number<MNPerWmma / ABK1Value>{},
                                                           Number<KPack / ABK1Value>{})),
                           make_pass_through_transform(Number<ABK1Value>{}),
                           make_freeze_transform(I0)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}, Sequence<5>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<>{}));
        }
    }

    // __host__ __device__ static auto
    // MakeBGridDescriptor(index_t sizeK, index_t, index_t sizeN, index_t, index_t StrideB, index_t)
    // {
    //     auto grid_descriptor = MakeBGridDescriptor_N_K(sizeN, sizeK, StrideB);

    //     const auto a_grid_desc_mblock_kblock = transform_tensor_descriptor(
    //         grid_descriptor,
    //         make_tuple(
    //             make_unmerge_transform(make_tuple(sizeN / Number<NPerWmma>{},
    //             Number<NPerWmma>{})), make_unmerge_transform(make_tuple(sizeK / Number<KPack>{},
    //             Number<KPack>{}))),
    //         make_tuple(Sequence<0>{}, Sequence<1>{}),
    //         make_tuple(Sequence<0, 2>{}, Sequence<1, 3>{}));

    //     if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, BLayout>)
    //     {
    //         const auto a_grid_desc_mrepeat_krepeat_kgroup_mwmma_bk1 =
    //         transform_tensor_descriptor(
    //             a_grid_desc_mblock_kblock,
    //             make_tuple(make_pass_through_transform(sizeN / Number<NPerWmma>{}),
    //                        make_pass_through_transform(sizeK / Number<KPack>{}),
    //                        make_unmerge_transform(
    //                            make_tuple(Number<NPerWmma / BK1Value>{}, Number<BK1Value>{})),
    //                        make_unmerge_transform(
    //                            make_tuple(Number<KPack / BK1Value>{}, Number<BK1Value>{}))),
    //             make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
    //             make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 4>{}, Sequence<3, 5>{}));

    //         return transform_tensor_descriptor(
    //             a_grid_desc_mrepeat_krepeat_kgroup_mwmma_bk1,
    //             make_tuple(make_pass_through_transform(sizeN / Number<NPerWmma>{}),
    //                        make_pass_through_transform(sizeK / Number<KPack>{}),
    //                        make_merge_transform(make_tuple(Number<NPerWmma / BK1Value>{},
    //                                                        Number<KPack / BK1Value>{})),
    //                        make_pass_through_transform(Number<BK1Value>{}),
    //                        make_freeze_transform(I0)),
    //             make_tuple(
    //                 Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{},
    //                 Sequence<5>{}),
    //             make_tuple(
    //                 Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<>{}));
    //     }
    //     else if constexpr(is_same_v<tensor_layout::gemm::RowMajor, BLayout>)
    //     {
    //         const auto a_grid_desc_mrepeat_krepeat_kgroup_mwmma_bk1 =
    //         transform_tensor_descriptor(
    //             a_grid_desc_mblock_kblock,
    //             make_tuple(make_pass_through_transform(sizeN / Number<NPerWmma>{}),
    //                        make_pass_through_transform(sizeK / Number<KPack>{}),
    //                        make_unmerge_transform(
    //                            make_tuple(Number<KPack / BK1Value>{}, Number<BK1Value>{})),
    //                        make_unmerge_transform(
    //                            make_tuple(Number<NPerWmma / BK1Value>{}, Number<BK1Value>{}))),
    //             make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<3>{}, Sequence<2>{}),
    //             make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 4>{}, Sequence<3, 5>{}));

    //         return transform_tensor_descriptor(
    //             a_grid_desc_mrepeat_krepeat_kgroup_mwmma_bk1,
    //             make_tuple(make_pass_through_transform(sizeN / Number<NPerWmma>{}),
    //                        make_pass_through_transform(sizeK / Number<KPack>{}),
    //                        make_merge_transform(make_tuple(Number<NPerWmma / BK1Value>{},
    //                                                        Number<KPack / BK1Value>{})),
    //                        make_pass_through_transform(Number<BK1Value>{}),
    //                        make_freeze_transform(I0)),
    //             make_tuple(
    //                 Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{},
    //                 Sequence<5>{}),
    //             make_tuple(
    //                 Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<>{}));
    //     }
    // }

    __device__ static constexpr auto GetBlockDescriptor()
    {
        const auto a_grid_desc_mraw_kraw = [&]() {
            return make_naive_tensor_descriptor(
                make_tuple(Number<MNRepeat_ * MNWaves_>{},
                           Number<KRepeat_ * KWaves_>{},
                           Number<MNPerWmma * KPack / ABK1Value / ABK1Value>{},
                           Number<ABK1Value>{},
                           Number<ABK1Value>{}),
                make_tuple(Number<KPack * MNPerWmma * KWaves_ * KRepeat_>{},
                           Number<KPack * MNPerWmma>{},
                           Number<ABK1Value * ABK1Value>{},
                           Number<ABK1Value>{},
                           I1));
        }();

        return transform_tensor_descriptor(
            a_grid_desc_mraw_kraw,
            make_tuple(
                make_pass_through_transform(Number<MNRepeat_ * MNWaves_>{}),
                make_pass_through_transform(Number<KRepeat_ * KWaves_>{}),
                make_pass_through_transform(Number<MNPerWmma * KPack / ABK1Value / ABK1Value>{}),
                make_pass_through_transform(Number<ABK1Value>{}),
                make_freeze_transform(I0)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<>{}));
    }

    // __device__ static constexpr auto GetBBlockDescriptor()
    // {
    //     const auto a_grid_desc_mraw_kraw = [&]() {
    //         return make_naive_tensor_descriptor(
    //             make_tuple(Number<BNRepeat_ * BNWaves_>{},
    //                        Number<BKRepeat_ * BKWaves_>{},
    //                        Number<NPerWmma * KPack / BK1Value / BK1Value>{},
    //                        Number<BK1Value>{},
    //                        Number<BK1Value>{}),
    //             make_tuple(Number<KPack * NPerWmma * BKWaves_ * BKRepeat_>{},
    //                        Number<KPack * NPerWmma>{},
    //                        Number<BK1Value * BK1Value>{},
    //                        Number<BK1Value>{},
    //                        I1));
    //     }();

    //     return transform_tensor_descriptor(
    //         a_grid_desc_mraw_kraw,
    //         make_tuple(make_pass_through_transform(Number<BNRepeat_ * BNWaves_>{}),
    //                    make_pass_through_transform(Number<BKRepeat_ * BKWaves_>{}),
    //                    make_pass_through_transform(
    //                        Number<NPerWmma * KPack / BK1Value / BK1Value>{}),
    //                    make_pass_through_transform(Number<BK1Value>{}),
    //                    make_freeze_transform(I0)),
    //         make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{},
    //         Sequence<4>{}), make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{},
    //         Sequence<3>{}, Sequence<>{}));
    // }

    template <typename GridDescriptor,
              typename BlockDescriptor,
              typename ABsDataType,
              typename ABElementwiseOperation,
              index_t GlobalBufferNum>
    __device__ static auto GetBlockTransfer(GridDescriptor& grid_descriptor,
                                            BlockDescriptor& block_descriptor,
                                            ABElementwiseOperation& ab_element_op,
                                            const index_t block_mn_id)
    {
        // Note: GlobalBufferNum is currently not used but it will be needed
        // once we add other pipelines. It is currently needed only for
        // consistency with the thread tiles approach

        using ABDataType  = remove_cvref_t<tuple_element_t<0, ABsDataType>>;
        index_t tid       = ThisThreadBlock::GetThreadId();
        index_t wave_id   = tid / WaveSize;
        index_t wave_idK  = wave_id % KWaves_;
        index_t wave_idMN = wave_id / KWaves_;

        index_t lane               = __lane_id();
        index_t lane_local_id_grid = lane % ABK1Value;

        index_t lane_group_grid =
            is_same_v<ABLayout, ABMajorLayout> ? lane / 16 + (lane / 8 % 2) * 2 : lane / ABK1Value;

        index_t lane_group_block    = lane / ABK1Value;
        index_t lane_local_id_block = lane % ABK1Value;

        return ThreadGroupTransferGlobal<decltype(grid_descriptor[I0]),
                                         BlockDescriptor,
                                         ABDataType,
                                         ABDataType,
                                         ABElementwiseOperation,
                                         Sequence<MNRepeat_, KRepeat_, I1, I1>,
                                         Sequence<MNWaves_, KWaves_, I1, I1>,
                                         Sequence<I0, I1, I2, I3>,
                                         ABK1Value,
                                         !is_same_v<ABLayout, ABMajorLayout>>(
            grid_descriptor[I0],
            block_descriptor,
            make_multi_index(block_mn_id * (MNRepeat_ * MNWaves_) + wave_idMN,
                             wave_idK,
                             lane_group_grid,
                             lane_local_id_grid),
            make_multi_index(wave_idMN, wave_idK, lane_group_block, lane_local_id_block),
            ab_element_op);
    }

    // template<
    // typename GridDescriptor,
    // typename BlockDescriptor,
    // typename BsDataType,
    // typename BElementwiseOperation,
    // index_t GlobalBufferNum>
    // __device__ static auto GetBBlockTransfer(GridDescriptor& grid_descriptor,
    //                                          BlockDescriptor& block_descriptor,
    //                                          BElementwiseOperation& b_element_op,
    //                                          const index_t block_n_id,
    //                                          const index_t)
    // {
    //     // Note: GlobalBufferNum is currently not used but it will be needed
    //     // once we add other pipelines. It is currently needed only for
    //     // consistency with the thread tiles approach

    //     using BDataType = remove_cvref_t<tuple_element_t<0, BsDataType>>;
    //     index_t tid      = ThisThreadBlock::GetThreadId();
    //     index_t wave_id  = tid / WaveSize;
    //     index_t wave_idK = wave_id % BKWaves_;
    //     index_t wave_idN = wave_id / BKWaves_;

    //     index_t lane               = __lane_id();
    //     index_t lane_local_id_grid = lane % BK1Value;

    //     index_t lane_group_grid = is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>
    //                                   ? lane / 16 + (lane / 8 % 2) * 2
    //                                   : lane / BK1Value;

    //     index_t lane_group_block    = lane / BK1Value;
    //     index_t lane_local_id_block = lane % BK1Value;

    //     return ThreadGroupTransferGlobal<
    //         decltype(grid_descriptor[I0]),
    //         BlockDescriptor,
    //         BDataType,
    //         BDataType,
    //         BElementwiseOperation,
    //         Sequence<Number<BNRepeat_>{}, Number<BKRepeat_>{}, I1, I1>,
    //         Sequence<Number<BNWaves_>{}, Number<BKWaves_>{}, I1, I1>,
    //         Sequence<I0, I1, I2, I3>,
    //         BK1Value,
    //         is_same_v<BLayout, tensor_layout::gemm::RowMajor>>(
    //         grid_descriptor[I0],
    //         block_descriptor,
    //         make_multi_index(block_n_id * (BNRepeat_ * BNWaves_) + wave_idN,
    //                          wave_idK,
    //                          lane_group_grid,
    //                          lane_local_id_grid),
    //         make_multi_index(wave_idN, wave_idK, lane_group_block, lane_local_id_block),
    //         b_element_op);
    // }

    template <index_t MNRepeat, index_t MNWaves>
    __host__ __device__ static constexpr auto MakeWmmaTileDescriptor()
    {
        // K0_MN_K1 -> K0_MNRepeat_MNWaves_KRow_MNPerWmma_K1
#ifdef __gfx12__
        constexpr auto KRow = I2;
#else
        constexpr auto KRow = I1;
#endif

        return make_naive_tensor_descriptor(make_tuple(Number<KPerBlock / KPack>{},
                                                       Number<MNRepeat>{},
                                                       Number<MNWaves>{},
                                                       Number<KRow>{},
                                                       Number<MNPerWmma>{},
                                                       Number<ABK1Value>{}),
                                            make_tuple(Number<KPack * MNPerWmma>{},
                                                       Number<KPerBlock * MNPerWmma * MNWaves>{},
                                                       Number<KPerBlock * MNPerWmma>{},
                                                       Number<MNPerWmma * ABK1Value>{},
                                                       Number<ABK1Value>{},
                                                       I1));
    }

    // template <index_t MNRepeat, index_t MNWaves, index_t MNPerWmma>
    // __host__ __device__ static constexpr auto MakeAWmmaTileDescriptor()
    // {
    //     return MakeWmmaTileDescriptor<MNRepeat, MNWaves, MNPerWmma, ABK1Value>();
    // }

    // template <index_t MNRepeat, index_t MNWaves, index_t MNPerWmma>
    // __host__ __device__ static constexpr auto MakeBWmmaTileDescriptor()
    // {
    //     return MakeWmmaTileDescriptor<MNRepeat, MNWaves, MNPerWmma, BK1Value>();
    // }

    __device__ static constexpr auto GetBlockStep()
    {
        return make_multi_index(I0, KWaves_ * KRepeat_, I0, I0);
    }

    // __device__ static constexpr auto GetBBlockStep()
    // {
    //     return make_multi_index(I0, BKWaves_ * BKRepeat_, I0, I0);
    // }

    template <typename GridDescriptor>
    __device__ static constexpr index_t GetKDimension(const GridDescriptor& grid_desc)
    {
        return grid_desc.GetLength(I1) * KPack;
    }
};

} // namespace ck
