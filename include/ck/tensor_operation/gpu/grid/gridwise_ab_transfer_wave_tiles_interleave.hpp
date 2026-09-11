// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/amd_address_space.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_global.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_ab_transfer_wave_tiles.hpp"
#include "ck/utility/math.hpp"

namespace ck {

template <typename ABLayout,
          typename ABMajorLayout,
          typename LDSTypeAB,
          index_t BlockSize,
          index_t MNPerBlock,
          index_t KPerBlock,
          index_t MNPerWmma,
          index_t KPack,
          index_t ABK1Value,
          index_t WaveSize,
          index_t MNWaves_Gemm>
struct ABTransferWaveTilesInterleave : ABTransferWaveTiles<ABLayout,
                                                           ABMajorLayout,
                                                           LDSTypeAB,
                                                           BlockSize,
                                                           MNPerBlock,
                                                           KPerBlock,
                                                           MNPerWmma,
                                                           KPack,
                                                           ABK1Value,
                                                           WaveSize>
{
    using Base = ABTransferWaveTiles<ABLayout,
                                     ABMajorLayout,
                                     LDSTypeAB,
                                     BlockSize,
                                     MNPerBlock,
                                     KPerBlock,
                                     MNPerWmma,
                                     KPack,
                                     ABK1Value,
                                     WaveSize>;

    using Base::ABDoTranspose;
    using Base::I0;
    using Base::I1;
    using Base::I2;
    using Base::I3;
    using Base::MNKRow;

    using Base::GetBlockLaneIdx;
    using Base::GetBlockStep;
    using Base::GetGridLaneIdx;
    using Base::GetWaveIdx;
    using Base::PadGridDescriptor;
    using typename Base::ThisThreadBlock;

    static constexpr auto I4 = Number<4>{};

    static_assert(!ABDoTranspose, "wave tile interleaved transfer does not support transpose yet");

    using Base::KRepeat_;
    using Base::KWaves_;
    using Base::MNRepeat_;

    static constexpr index_t MNWaves_Grid  = MNWaves_Gemm;
    static constexpr index_t KWaves_Grid   = (BlockSize / WaveSize) / MNWaves_Gemm;
    static constexpr index_t KRepeat_Grid  = KPerBlock / (KWaves_Grid * KPack);
    static constexpr index_t MNRepeat_Grid = MNPerBlock / (MNWaves_Grid * MNPerWmma);

    template <bool PadMN, bool PadK, typename GridDescriptorBase>
    __host__ __device__ static auto MakeGridDescriptor(GridDescriptorBase& base_desc,
                                                       index_t sizeMN,
                                                       index_t MNPad,
                                                       index_t sizeK,
                                                       index_t KPad,
                                                       index_t,
                                                       index_t)
    {
        const auto base_desc_padded = Base::template PadGridDescriptor<PadMN, PadK>(
            base_desc, sizeMN, MNPad, sizeK, KPad, 0, 0);

        const index_t MN_grid = !PadMN ? sizeMN : MNPad;
        const index_t K_grid  = !PadK ? sizeK : KPad;

        // Divide the base descriptor MN_K into tiles
        const auto ab_grid_desc_mntiles_ktiles = transform_tensor_descriptor(
            base_desc_padded,
            make_tuple(make_unmerge_transform(make_tuple(
                           math::integer_divide_ceil(MN_grid, Number<MNPerWmma * MNRepeat_Grid>{}),
                           Number<MNPerWmma * MNRepeat_Grid>{})),
                       make_unmerge_transform(make_tuple(
                           math::integer_divide_ceil(K_grid, Number<KPack>{}), Number<KPack>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1, 3>{}));

        // The distinction is needed to get the same global indices for both layouts
        // Divide each tile in 2 16x8 subtile
        // MNTiles - KTiles - MNKRow - LaneLocal - VectorSize
        // MNKRow    = 0-1
        // LaneLocal = 0-15
        // VectorSize must be 8
        if constexpr(!ABDoTranspose)
        {
            const auto ab_grid_desc_mntiles_ktiles_mnrepeat = transform_tensor_descriptor(
                ab_grid_desc_mntiles_ktiles,
                make_tuple(
                    make_pass_through_transform(
                        math::integer_divide_ceil(MN_grid, Number<MNPerWmma * MNRepeat_Grid>{})),
                    make_pass_through_transform(math::integer_divide_ceil(K_grid, Number<KPack>{})),
                    make_unmerge_transform(
                        make_tuple(Number<MNPerWmma>{}, Number<MNRepeat_Grid>{})),
                    make_pass_through_transform(Number<KPack>{})),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<3, 2>{}, Sequence<4>{}));

            const auto ab_grid_desc_mntiles_ktiles_lanegroup_lanelocal_abk1 =
                transform_tensor_descriptor(
                    ab_grid_desc_mntiles_ktiles_mnrepeat,
                    make_tuple(make_pass_through_transform(math::integer_divide_ceil(
                                   MN_grid, Number<MNPerWmma * MNRepeat_Grid>{})),
                               make_pass_through_transform(
                                   math::integer_divide_ceil(K_grid, Number<KPack>{})),
                               make_pass_through_transform(Number<MNRepeat_Grid>{}),
                               make_pass_through_transform(Number<MNPerWmma>{}),
                               make_unmerge_transform(
                                   make_tuple(Number<MNKRow>{}, Number<KPack / MNKRow>{}))),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                    make_tuple(Sequence<0>{},
                               Sequence<1>{},
                               Sequence<2>{},
                               Sequence<3>{},
                               Sequence<4, 5>{}));

            // Freeze VectorSize to first element of the loading chunk (for convenience)
            // Swap MNPerWmma and MNKRow for consistency with transpose descriptor
            return transform_tensor_descriptor(
                ab_grid_desc_mntiles_ktiles_lanegroup_lanelocal_abk1,
                make_tuple(
                    make_pass_through_transform(
                        math::integer_divide_ceil(MN_grid, Number<MNPerWmma * MNRepeat_Grid>{})),
                    make_pass_through_transform(math::integer_divide_ceil(K_grid, Number<KPack>{})),
                    make_pass_through_transform(Number<MNRepeat_Grid>{}),
                    make_pass_through_transform(Number<MNPerWmma>{}),
                    make_pass_through_transform(Number<MNKRow>{}),
                    make_freeze_transform(I0)),
                make_tuple(Sequence<0>{},
                           Sequence<1>{},
                           Sequence<2>{},
                           Sequence<4>{},
                           Sequence<3>{},
                           Sequence<5>{}),
                make_tuple(Sequence<0>{},
                           Sequence<1>{},
                           Sequence<2>{},
                           Sequence<3>{},
                           Sequence<4>{},
                           Sequence<>{}));
        }
    }

    __device__ static constexpr auto GetBlockDescriptor()
    {
        // LDS memory layouts:
        // lanes within tiles stored contiguously in chunks of 8 elements
        // tiles are then stored first in K dimension
        // MNTiles - KTiles - MNKRow - LaneLocal - VectorSize
        const auto a_grid_desc_mraw_kraw = [&]() {
            return make_naive_tensor_descriptor(
                make_tuple(Number<MNWaves_Grid>{},
                           Number<KRepeat_Grid * KWaves_Grid>{},
                           Number<MNRepeat_Grid>{},
                           Number<MNKRow>{},
                           Number<MNPerWmma>{},
                           Number<ABK1Value>{}),
                make_tuple(Number<KPack * MNPerWmma * KWaves_Grid * KRepeat_Grid>{},
                           Number<KPack * MNPerWmma>{},
                           Number<KPack * MNPerWmma * KWaves_Grid * KRepeat_Grid * MNWaves_Grid>{},
                           Number<ABK1Value * MNPerWmma>{},
                           Number<ABK1Value>{},
                           I1));
        }();

        // Freeze VectorSize to first element of the chunk (for convenience)
        return transform_tensor_descriptor(
            a_grid_desc_mraw_kraw,
            make_tuple(make_pass_through_transform(Number<MNWaves_Grid>{}),
                       make_pass_through_transform(Number<KRepeat_Grid * KWaves_Grid>{}),
                       make_pass_through_transform(Number<MNRepeat_Grid>{}),
                       make_pass_through_transform(Number<MNKRow>{}),
                       make_pass_through_transform(Number<MNPerWmma>{}),
                       make_freeze_transform(I0)),
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
                       Sequence<>{}));
    }

    template <typename GridDescriptor,
              typename BlockDescriptor,
              typename ABsDataType,
              typename ABElementwiseOperation,
              index_t GlobalBufferNum>
    __device__ static auto GetBlockTransfer(GridDescriptor& grid_descriptor,
                                            BlockDescriptor& block_descriptor,
                                            ABElementwiseOperation& ab_element_op,
                                            const index_t block_mn_id,
                                            const index_t)
    {
        constexpr index_t NumABTensor = ABsDataType::Size();

        const auto wave_idx = GetWaveIdx();
        index_t wave_idK    = wave_idx[I1];
        index_t wave_idMN   = wave_idx[I0];

        const auto block_lane_id    = GetBlockLaneIdx();
        index_t lane_group_block    = block_lane_id[I0];
        index_t lane_local_id_block = block_lane_id[I1];

        constexpr index_t MNRepeatRatio = MNRepeat_Grid / MNRepeat_;

        const auto idx_as_block_begin = generate_tuple(
            [&](auto iTensor) {
                using ABDataType           = remove_cvref_t<tuple_element_t<iTensor, ABsDataType>>;
                const auto grid_lane_id    = Base::template GetGridLaneIdx<ABDataType>();
                index_t lane_group_grid    = grid_lane_id[I0];
                index_t lane_local_id_grid = grid_lane_id[I1];
                return make_multi_index(block_mn_id * MNWaves_Grid + wave_idMN / MNRepeatRatio,
                                        wave_idK * KRepeat_Grid,
                                        (wave_idMN % MNRepeatRatio) * MNRepeat_,
                                        lane_group_grid,
                                        lane_local_id_grid);
            },
            Number<NumABTensor>{});

        return ThreadGroupTransferGlobal<GridDescriptor,
                                         BlockDescriptor,
                                         ABsDataType,
                                         LDSTypeAB,
                                         ABElementwiseOperation,
                                         Sequence<I1, KRepeat_, MNRepeat_, I1, I1>,
                                         Sequence<I1, KWaves_, I1, I1, I1>,
                                         Sequence<I0, I1, I2, I3, I4>,
                                         ABK1Value,
                                         ABDoTranspose,
                                         GlobalBufferNum>(
            grid_descriptor,
            block_descriptor,
            idx_as_block_begin,
            make_multi_index(wave_idMN / MNRepeatRatio,
                             wave_idK * KRepeat_,
                             (wave_idMN % MNRepeatRatio) * MNRepeat_,
                             lane_group_block,
                             lane_local_id_block),
            ab_element_op);
    }

    __device__ static constexpr auto GetBlockStep()
    {
        // Grid descriptor step (MoveSrcSliceWindow)
        return make_multi_index(I0, KWaves_ * KRepeat_, I0, I0, I0);
    }
};

} // namespace ck
