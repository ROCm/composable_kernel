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

    // Tiles distribution for global memory loading
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
        static_assert(!PadMN && !PadK, "padding is currently not supported");

        // Divide the base descriptor MN_K into tiles
        const auto ab_grid_desc_mntiles_ktiles = transform_tensor_descriptor(
            base_desc,
            make_tuple(
                make_unmerge_transform(
                    make_tuple(sizeMN / Number<MNPerWmma>{}, Number<MNPerWmma>{})),
                make_unmerge_transform(make_tuple(sizeK / Number<KPack>{}, Number<KPack>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1, 3>{}));

        // The distinction is needed to get the same global indices for both layouts
        if constexpr(is_same_v<ABMajorLayout, ABLayout>)
        {
            // Divide each tile in 8x8 subtile
            // MNTiles - KTiles - LaneGroup - LaneLocal - VectorSize
            // LaneGroup = 0-3
            // LaneLocal = 0-7
            // VectorSize must be 8
            const auto ab_grid_desc_mntiles_ktiles_lanegroup_lanelocal_abk1 =
                transform_tensor_descriptor(
                    ab_grid_desc_mntiles_ktiles,
                    make_tuple(make_pass_through_transform(sizeMN / Number<MNPerWmma>{}),
                               make_pass_through_transform(sizeK / Number<KPack>{}),
                               make_unmerge_transform(make_tuple(Number<MNPerWmma / ABK1Value>{},
                                                                 Number<ABK1Value>{})),
                               make_unmerge_transform(
                                   make_tuple(Number<KPack / ABK1Value>{}, Number<ABK1Value>{}))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 4>{}, Sequence<3, 5>{}));

            // Freeze VectorSize to first element of the loading chunk (for convenience)
            return transform_tensor_descriptor(
                ab_grid_desc_mntiles_ktiles_lanegroup_lanelocal_abk1,
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
            // Divide each 16x16 tile into 4 8x8 subtiles
            // MNTiles - KTiles - LaneGroup - LaneLocal - VectorSize
            // LaneGroup = 0-3
            // LaneLocal = 0-7
            // VectorSize must be 8
            const auto ab_grid_desc_mntiles_ktiles_lanegroup_lanelocal_abk1 =
                transform_tensor_descriptor(
                    ab_grid_desc_mntiles_ktiles,
                    make_tuple(make_pass_through_transform(sizeMN / Number<MNPerWmma>{}),
                               make_pass_through_transform(sizeK / Number<KPack>{}),
                               make_unmerge_transform(
                                   make_tuple(Number<KPack / ABK1Value>{}, Number<ABK1Value>{})),
                               make_unmerge_transform(make_tuple(Number<MNPerWmma / ABK1Value>{},
                                                                 Number<ABK1Value>{}))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<3>{}, Sequence<2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 4>{}, Sequence<3, 5>{}));

            // Freeze VectorSize to first element of the loading chunk (for convenience)
            return transform_tensor_descriptor(
                ab_grid_desc_mntiles_ktiles_lanegroup_lanelocal_abk1,
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

    __device__ static constexpr auto GetBlockDescriptor()
    {
        // LDS memory layouts:
        // lanes within tiles stored contiguously in chunks of 8 elements
        // tiles are then stored first in K dimension
        // MNTiles - KTiles - LaneGroup - LaneLocal - VectorSize
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

        // Freeze VectorSize to first element of the chunk (for convenience)
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

    __device__ static auto GetWaveIdx()
    {
        const index_t thread_id = ThisThreadBlock::GetThreadId();

        constexpr auto threadid_to_wave_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(MNWaves_, KWaves_, WaveSize))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));

        return threadid_to_wave_idx_adaptor.CalculateBottomIndex(make_multi_index(thread_id));
    }

    __device__ static auto GetBlockLaneIdx()
    {
        const index_t lane_id = __lane_id();
        ;

        constexpr index_t NumberSubTiles = (MNPerWmma / ABK1Value) * (KPack / ABK1Value);
        constexpr auto laneid_to_block_lane_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(NumberSubTiles, ABK1Value))),
            make_tuple(Sequence<0, 1>{}),
            make_tuple(Sequence<0>{}));

        return laneid_to_block_lane_idx_adaptor.CalculateBottomIndex(make_multi_index(lane_id));
    }

    __device__ static auto GetGridLaneIdx()
    {
        const index_t lane_id = __lane_id();

        constexpr index_t SubTilesK                    = KPack / ABK1Value;
        constexpr index_t SubTilesMN                   = MNPerWmma / ABK1Value;
        constexpr auto laneid_to_grid_lane_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(SubTilesK, SubTilesMN, ABK1Value))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));

        const auto indices =
            laneid_to_grid_lane_idx_adaptor.CalculateBottomIndex(make_multi_index(lane_id));

        if constexpr(is_same_v<ABMajorLayout, ABLayout>)
        {
            return make_multi_index(indices[I1] * SubTilesMN + indices[I0], indices[I2]);
        }
        else
        {
            return make_multi_index(indices[I0] * SubTilesMN + indices[I1], indices[I2]);
        }
    }

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
        static_assert(GlobalBufferNum == 1, "single global buffer is only supported");
        constexpr index_t NumABTensor = ABsDataType::Size();
        static_assert(NumABTensor == 1, "multiAB currently not supported");

        using ABDataType = remove_cvref_t<tuple_element_t<0, ABsDataType>>;

        const auto wave_idx = GetWaveIdx();
        index_t wave_idK    = wave_idx[I1];
        index_t wave_idMN   = wave_idx[I0];

        const auto grid_lane_id    = GetGridLaneIdx();
        index_t lane_group_grid    = grid_lane_id[I0];
        index_t lane_local_id_grid = grid_lane_id[I1];

        const auto block_lane_id    = GetBlockLaneIdx();
        index_t lane_group_block    = block_lane_id[I0];
        index_t lane_local_id_block = block_lane_id[I1];

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

    template <index_t MNRepeat, index_t MNWaves>
    __host__ __device__ static constexpr auto MakeWmmaTileDescriptor()
    {
        // This is a block descriptor used to read LDS memory into register
        // It's defined in a way consistent with the existing implementation to
        // avoid changes in the pipelines
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

    __device__ static constexpr auto GetBlockStep()
    {
        // Grid descriptor step (MoveSrcSliceWindow)
        return make_multi_index(I0, KWaves_ * KRepeat_, I0, I0);
    }

    template <typename GridDescriptor>
    __device__ static constexpr index_t GetKDimension(const GridDescriptor& grid_desc)
    {
        // K dimension size. This should always be called with the A matrix grid descriptor
        // because it doesn't work for B matrix when packed int4 is used
        return grid_desc.GetLength(I1) * KPack;
    }
};

} // namespace ck
