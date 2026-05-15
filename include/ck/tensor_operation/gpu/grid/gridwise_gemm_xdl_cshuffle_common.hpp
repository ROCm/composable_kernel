// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(__HIPCC_RTC__) || !defined(CK_CODE_GEN_RTC)
#include <iostream>
#include <ostream>
#endif

#include "ck/utility/env.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v6r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v6r1r2.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7r2.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7r3.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7r3_scatter.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

enum Activation
{
    gelu_and_mul       = 0,
    silu_and_mul       = 1,
    swiglustep_and_mul = 2
};

template <typename ALayout,
          typename BLayout,
          typename ELayout,
          typename ADataType, // ALDSType
          typename BDataType, // BLDSType
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1Value,
          index_t BK1Value,
          index_t MPerXdl,
          index_t NPerXdl,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          index_t BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          typename CDEShuffleBlockTransferScalarPerVectors,
          typename ComputeTypeA,
          typename ComputeTypeB,
          bool ForceNaiveLdsLayout,
          bool DirectLoad = false,
          bool IsMxGemm   = false>
struct GridwiseGemm_xdl_cshuffle_base
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};
    static constexpr auto I8 = Number<8>{};
    static constexpr auto I9 = Number<9>{};

    // K1 should be Number<...>
    static constexpr auto AKPerBlock = KPerBlock;
    static constexpr auto BKPerBlock = []() {
        if constexpr(IsMxGemm)
        {
            // KPerBlock is based on packed data type in MxGemm
            return KPerBlock * packed_size_v<ADataType> / packed_size_v<BDataType>;
        }
        else
        {
            return KPerBlock;
        }
    }();
    static constexpr auto AK0Number = Number<AKPerBlock / AK1Value>{};
    static constexpr auto BK0Number = Number<BKPerBlock / BK1Value>{};
    static constexpr auto AK1Number = Number<AK1Value>{};
    static constexpr auto BK1Number = Number<BK1Value>{};

    static constexpr auto MaxBlockSize = BlockSize;

    static constexpr auto CShuffleBlockTransferScalarPerVector_NPerBlock =
        CDEShuffleBlockTransferScalarPerVectors{}[I0];

    using ThisThreadBlock               = ThisThreadBlock<BlockSize>;
    static constexpr index_t NumDTensor = DsDataType::Size();

    static constexpr index_t APackedSize = []() {
        if constexpr(IsMxGemm)
        {
            // KPerBlock is based on packed data type in MxGemm
            return 1;
        }
        else if constexpr(is_same_v<remove_cvref_t<ADataType>, pk_i4_t>)
        {
            return 2;
        }
        else
        {
            return packed_size_v<ADataType>;
        }
    }();

    static constexpr index_t BPackedSize = []() {
        if constexpr(IsMxGemm)
        {
            // KPerBlock is based on packed data type in MxGemm
            return 1;
        }
        else if constexpr(is_same_v<remove_cvref_t<BDataType>, pk_i4_t>)
        {
            return 2;
        }
        else
        {
            return packed_size_v<BDataType>;
        }
    }();
    template <typename DeviceArch>
    __device__ __host__ static constexpr auto
    GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1(DeviceArch)
    {
        constexpr index_t MWave           = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWave           = NPerBlock / (NXdlPerWave * NPerXdl);
        constexpr index_t WaveSize        = BlockSize / (MWave * NWave);
        constexpr index_t KPerBlockInByte = AKPerBlock * sizeof(ADataType) / APackedSize;

        // A matrix in LDS memory, dst of blockwise copy
        if constexpr(DirectLoad &&
                     (is_same_v<DeviceArch, gfx950_t> || is_same_v<DeviceArch, gfx9_t>))
        {
            if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                // FIXME: our support to non-K contiguous layout is limited, only work in some
                // specific setting
                return make_naive_tensor_descriptor_packed(
                    make_tuple(AK0Number, Number<MPerBlock>{}, AK1Number));
            }
            else
            {
                return make_naive_tensor_descriptor(
                    make_tuple(AK0Number, Number<MPerBlock>{}, AK1Number),
                    make_tuple(AK1Number, Number<AKPerBlock>{}, I1));
            }
        }
        else if constexpr(ABlockLdsExtraM || ForceNaiveLdsLayout)
        {
            // bank conflict when writting the data into LDS, but don't worry, we have whole entire
            // loop to hide it in v4. it may give you some benefit from less valu in compute address
            return make_naive_tensor_descriptor(
                make_tuple(AK0Number, Number<MPerBlock>{}, AK1Number),
                make_tuple(Number<MPerBlock + ABlockLdsExtraM>{} * AK1Number, AK1Number, I1));
        }
        // xor tensor transformation request more unnecessary vgpr usage, would cause register spill
        // in some cases.
        else if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
        {
            constexpr index_t LdsSize       = 32 * 4 / KPerBlockInByte;
            constexpr auto MLdsLayer        = LdsSize < 1 ? 1 : LdsSize;
            constexpr auto a_lds_block_desc = make_naive_tensor_descriptor(
                make_tuple(
                    AK0Number * Number<MLdsLayer>{}, Number<MPerBlock / MLdsLayer>{}, AK1Number),
                make_tuple(AK1Number, Number<AKPerBlock * MLdsLayer>{}, I1));

            constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
                a_lds_block_desc,
                make_tuple(make_xor_with_modulo_transform(make_tuple(
                               Number<MPerBlock / MLdsLayer>{}, Number<AK0Number * MLdsLayer>{})),
                           make_pass_through_transform(AK1Number)),
                make_tuple(Sequence<1, 0>{}, Sequence<2>{}),
                make_tuple(Sequence<1, 0>{}, Sequence<2>{}));

            constexpr auto a_lds_block_desc_ak0_mldslayer_m_ak1 = transform_tensor_descriptor(
                a_lds_block_desc_permuted,
                make_tuple(make_unmerge_transform(make_tuple(AK0Number, Number<MLdsLayer>{})),
                           make_pass_through_transform(Number<MPerBlock / MLdsLayer>{}),
                           make_pass_through_transform(AK1Number)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}, Sequence<3>{}));

            constexpr auto a_lds_block_desc_ak0_m_ak1 = transform_tensor_descriptor(
                a_lds_block_desc_ak0_mldslayer_m_ak1,
                make_tuple(make_pass_through_transform(AK0Number),
                           make_merge_transform_v3_division_mod(
                               make_tuple(Number<MPerBlock / MLdsLayer>{}, Number<MLdsLayer>{})),
                           make_pass_through_transform(AK1Number)),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            return a_lds_block_desc_ak0_m_ak1;
        }
        else // ColumnMajor A
        {
            // kfold and mpair dimension is not always required.
            // more dimension in merge_transform increase the difficulty of generating immarg offset
            // for compiler.
            constexpr auto M0 = ABlockTransferThreadClusterLengths_AK0_M_AK1{}.At(I1);
            constexpr auto M1 = MPerBlock / M0;

            constexpr auto KThreadWrite     = ABlockTransferThreadClusterLengths_AK0_M_AK1{}.At(I0);
            constexpr auto K0PerThreadWrite = AK0Number / KThreadWrite;
            constexpr auto KThreadRead      = WaveSize / MPerXdl;
            constexpr auto K0PerThreadRead  = AK0Number / KThreadRead;

            constexpr auto kfold = (AK1Number * M0 * sizeof(ADataType) > 128)
                                       ? 1
                                       : 128 / (AK1Number * M0 * sizeof(ADataType));
            constexpr auto KThreadReadPerm =
                (kfold * K0PerThreadWrite / K0PerThreadRead) > 1
                    ? KThreadRead / (kfold * K0PerThreadWrite / K0PerThreadRead)
                    : KThreadRead;

            // 1<=mpair<=n0
            constexpr auto mpair = (AK1Number * MPerXdl * sizeof(ADataType) > 128)
                                       ? 1
                                       : ((128 / (AK1Number * MPerXdl * sizeof(ADataType))) > M0
                                              ? M0
                                              : 128 / (AK1Number * MPerXdl * sizeof(ADataType)));

            constexpr auto a_lds_block_desc = make_naive_tensor_descriptor_packed(
                make_tuple(Number<KThreadWrite / kfold / KThreadReadPerm>{},
                           Number<K0PerThreadWrite>{},
                           Number<KThreadReadPerm * M1>{},
                           Number<kfold * M0 / mpair>{},
                           Number<mpair>{},
                           AK1Number));

            constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
                a_lds_block_desc,
                make_tuple(
                    make_pass_through_transform(Number<KThreadWrite / kfold / KThreadReadPerm>{}),
                    make_pass_through_transform(Number<K0PerThreadWrite>{}),
                    make_xor_with_modulo_transform(
                        make_tuple(Number<KThreadReadPerm * M1>{}, Number<kfold * M0 / mpair>{})),
                    make_pass_through_transform(Number<mpair>{}),
                    make_pass_through_transform(AK1Number)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}, Sequence<5>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}, Sequence<5>{}));

            constexpr auto a_lds_block_desc_unmerged = transform_tensor_descriptor(
                a_lds_block_desc_permuted,
                make_tuple(
                    make_pass_through_transform(Number<KThreadWrite / kfold / KThreadReadPerm>{}),
                    make_pass_through_transform(Number<K0PerThreadWrite>{}),
                    make_unmerge_transform(make_tuple(Number<KThreadReadPerm>{}, Number<M1>{})),
                    make_unmerge_transform(make_tuple(Number<kfold>{}, Number<M0 / mpair>{})),
                    make_pass_through_transform(Number<mpair>{}),
                    make_pass_through_transform(AK1Number)),
                make_tuple(Sequence<0>{},
                           Sequence<1>{},
                           Sequence<2>{},
                           Sequence<3>{},
                           Sequence<4>{},
                           Sequence<5>{}),
                make_tuple(Sequence<1>{},
                           Sequence<2>{},
                           Sequence<0, 3>{},
                           Sequence<4, 5>{},
                           Sequence<6>{},
                           Sequence<7>{}));

            constexpr auto a_lds_block_desc_ak0_m_ak1 = transform_tensor_descriptor(
                a_lds_block_desc_unmerged,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(Number<KThreadReadPerm>{},
                                          Number<KThreadWrite / kfold / KThreadReadPerm>{},
                                          Number<kfold>{},
                                          Number<K0PerThreadWrite>{})),
                           make_merge_transform_v3_division_mod(
                               make_tuple(Number<M0 / mpair>{}, Number<mpair>{}, Number<M1>{})),
                           make_pass_through_transform(AK1Number)),
                make_tuple(Sequence<0, 1, 4, 2>{}, Sequence<5, 6, 3>{}, Sequence<7>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            return a_lds_block_desc_ak0_m_ak1;
        }
    }

    template <>
    __device__ __host__ constexpr auto
    GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1<gfx125_t>(gfx125_t)
    {
        constexpr index_t KPerBlockInByte = AKPerBlock * sizeof(ADataType) / APackedSize;
        constexpr index_t LdsSize         = get_n_lds_banks(gfx125_t{}) * 4 / KPerBlockInByte;
        constexpr bool EnableLdsLayer     = ABlockTransferThreadClusterLengths_AK0_M_AK1{}[0] *
                                            ABlockTransferThreadClusterLengths_AK0_M_AK1{}[1] *
                                            ABlockTransferThreadClusterLengths_AK0_M_AK1{}[2] ==
                                        BlockSize;
        constexpr index_t MLdsLayer = (EnableLdsLayer == false) || (LdsSize < 1) ? 1 : LdsSize;
        constexpr index_t MPerThread =
            MPerBlock / ABlockTransferThreadClusterLengths_AK0_M_AK1{}[1];
        constexpr index_t MPerThreadLayer = [&]() {
            if constexpr(DirectLoad || MPerThread == 1)
            {
                return 1;
            }
            // Disable MPerThreadLayer if it is non-power two.
            else if constexpr(math::next_power_of_two<MPerThread>() != MPerThread)
            {
                return 1;
            }
            else
            {
                return (MPerThread >= 16) ? 4 : MPerThread;
            }
        }();

        static_assert(MLdsLayer == 1 || MPerBlock % (MLdsLayer * MPerThreadLayer) == 0);
        // A matrix in LDS memory, dst of blockwise copy
        if constexpr(ABlockLdsExtraM || ForceNaiveLdsLayout || DirectLoad)
        {
            // 16 is the byte size of ds_load_b128 and ds_write_b128.
            constexpr auto PaddingSize = 16 / sizeof(ADataType);
            if constexpr(MLdsLayer == 1)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(AK0Number, Number<MPerBlock>{}, AK1Number),
                    make_tuple(AK1Number, Number<AKPerBlock + PaddingSize>{}, I1));
            }
            else
            {
                constexpr auto a_lds_block_desc_ak0_m_unmerge_ak1 = make_naive_tensor_descriptor(
                    make_tuple(AK0Number,
                               Number<MPerBlock / MLdsLayer / MPerThreadLayer>{},
                               Number<MPerThreadLayer>{},
                               Number<MLdsLayer>{},
                               AK1Number),
                    make_tuple(AK1Number,
                               Number<(AKPerBlock * MLdsLayer + PaddingSize) * MPerThreadLayer>{},
                               Number<AKPerBlock * MLdsLayer + PaddingSize>{},
                               Number<AKPerBlock>{},
                               I1));

                return transform_tensor_descriptor(
                    a_lds_block_desc_ak0_m_unmerge_ak1,
                    make_tuple(make_pass_through_transform(AK0Number),
                               make_merge_transform_v3_division_mod(
                                   make_tuple(Number<MPerBlock / MLdsLayer / MPerThreadLayer>{},
                                              Number<MLdsLayer>{},
                                              Number<MPerThreadLayer>{})),
                               make_pass_through_transform(AK1Number)),
                    make_tuple(Sequence<0>{}, Sequence<1, 3, 2>{}, Sequence<4>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
            }
        }
        // xor tensor transformation request more unnecessary vgpr usage, would cause register spill
        // in some cases.
        else if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
        {
            constexpr auto a_lds_block_desc = make_naive_tensor_descriptor(
                make_tuple(
                    AK0Number * Number<MLdsLayer>{}, Number<MPerBlock / MLdsLayer>{}, AK1Number),
                make_tuple(AK1Number, Number<AKPerBlock * MLdsLayer>{}, I1));

            constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
                a_lds_block_desc,
                make_tuple(make_xor_with_modulo_transform(make_tuple(
                               Number<MPerBlock / MLdsLayer>{}, Number<AK0Number * MLdsLayer>{})),
                           make_pass_through_transform(AK1Number)),
                make_tuple(Sequence<1, 0>{}, Sequence<2>{}),
                make_tuple(Sequence<1, 0>{}, Sequence<2>{}));
            if constexpr(MLdsLayer == 1)
            {
                return a_lds_block_desc_permuted;
            }
            else
            {

                constexpr auto a_lds_block_desc_ak0_mldslayer_m_ak1 = transform_tensor_descriptor(
                    a_lds_block_desc_permuted,
                    make_tuple(make_unmerge_transform(make_tuple(Number<MLdsLayer>{}, AK0Number)),
                               make_unmerge_transform(
                                   make_tuple(Number<MPerBlock / MLdsLayer / MPerThreadLayer>{},
                                              Number<MPerThreadLayer>{})),
                               make_pass_through_transform(AK1Number)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<2, 0>{}, Sequence<1, 3>{}, Sequence<4>{}));

                constexpr auto a_lds_block_desc_ak0_m_ak1 = transform_tensor_descriptor(
                    a_lds_block_desc_ak0_mldslayer_m_ak1,
                    make_tuple(make_pass_through_transform(AK0Number),
                               make_merge_transform_v3_division_mod(
                                   make_tuple(Number<MPerBlock / MLdsLayer / MPerThreadLayer>{},
                                              Number<MLdsLayer>{},
                                              Number<MPerThreadLayer>{})),
                               make_pass_through_transform(AK1Number)),
                    make_tuple(Sequence<0>{}, Sequence<1, 2, 3>{}, Sequence<4>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

                return a_lds_block_desc_ak0_m_ak1;
            }
        }
        else // ColumnMajor A
        {
            constexpr index_t MWave    = MPerBlock / (MXdlPerWave * MPerXdl);
            constexpr index_t NWave    = NPerBlock / (NXdlPerWave * NPerXdl);
            constexpr index_t WaveSize = BlockSize / (MWave * NWave);

            constexpr auto LdsBankSize = get_n_lds_banks(gfx125_t{}) * 4;
            constexpr auto M0          = ABlockTransferThreadClusterLengths_AK0_M_AK1{}.At(I1);
            constexpr auto M1          = MPerBlock / M0;

            constexpr auto KThreadWrite     = ABlockTransferThreadClusterLengths_AK0_M_AK1{}.At(I0);
            constexpr auto K0PerThreadWrite = AK0Number / KThreadWrite;
            constexpr auto KThreadRead      = WaveSize / MPerXdl;
            constexpr auto K0PerThreadRead  = AK0Number / KThreadRead;

            constexpr auto kfold = (AK1Number * M0 * sizeof(ADataType) > LdsBankSize)
                                       ? 1
                                       : LdsBankSize / (AK1Number * M0 * sizeof(ADataType));
            constexpr auto KThreadReadPerm =
                (kfold * K0PerThreadWrite / K0PerThreadRead) > 1
                    ? KThreadRead / (kfold * K0PerThreadWrite / K0PerThreadRead)
                    : KThreadRead;

            // 1<=mpair<=n0
            constexpr auto mpair =
                (AK1Number * MPerXdl * sizeof(ADataType) > (2 * LdsBankSize))
                    ? 1
                    : (((2 * LdsBankSize) / (AK1Number * MPerXdl * sizeof(ADataType))) > M0
                           ? M0
                           : (2 * LdsBankSize) / (AK1Number * MPerXdl * sizeof(ADataType)));

            constexpr auto a_lds_block_desc = make_naive_tensor_descriptor_packed(
                make_tuple(Number<KThreadWrite / kfold / KThreadReadPerm>{},
                           Number<K0PerThreadWrite>{},
                           Number<KThreadReadPerm * M1>{},
                           Number<kfold * M0 / mpair>{},
                           Number<mpair>{},
                           AK1Number));

            constexpr auto a_lds_block_desc_permuted = transform_tensor_descriptor(
                a_lds_block_desc,
                make_tuple(
                    make_pass_through_transform(Number<KThreadWrite / kfold / KThreadReadPerm>{}),
                    make_pass_through_transform(Number<K0PerThreadWrite>{}),
                    make_xor_with_modulo_transform(
                        make_tuple(Number<KThreadReadPerm * M1>{}, Number<kfold * M0 / mpair>{})),
                    make_pass_through_transform(Number<mpair>{}),
                    make_pass_through_transform(AK1Number)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}, Sequence<5>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}, Sequence<5>{}));

            constexpr auto a_lds_block_desc_unmerged = transform_tensor_descriptor(
                a_lds_block_desc_permuted,
                make_tuple(
                    make_pass_through_transform(Number<KThreadWrite / kfold / KThreadReadPerm>{}),
                    make_pass_through_transform(Number<K0PerThreadWrite>{}),
                    make_unmerge_transform(make_tuple(Number<KThreadReadPerm>{}, Number<M1>{})),
                    make_unmerge_transform(make_tuple(Number<kfold>{}, Number<M0 / mpair>{})),
                    make_pass_through_transform(Number<mpair>{}),
                    make_pass_through_transform(AK1Number)),
                make_tuple(Sequence<0>{},
                           Sequence<1>{},
                           Sequence<2>{},
                           Sequence<3>{},
                           Sequence<4>{},
                           Sequence<5>{}),
                make_tuple(Sequence<1>{},
                           Sequence<2>{},
                           Sequence<0, 3>{},
                           Sequence<4, 5>{},
                           Sequence<6>{},
                           Sequence<7>{}));

            constexpr auto a_lds_block_desc_ak0_m_ak1 = transform_tensor_descriptor(
                a_lds_block_desc_unmerged,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(Number<KThreadReadPerm>{},
                                          Number<KThreadWrite / kfold / KThreadReadPerm>{},
                                          Number<kfold>{},
                                          Number<K0PerThreadWrite>{})),
                           make_merge_transform_v3_division_mod(
                               make_tuple(Number<M0 / mpair>{}, Number<mpair>{}, Number<M1>{})),
                           make_pass_through_transform(AK1Number)),
                make_tuple(Sequence<0, 1, 4, 2>{}, Sequence<5, 6, 3>{}, Sequence<7>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            return a_lds_block_desc_ak0_m_ak1;
        }
    }

    template <typename DeviceArch>
    __device__ __host__ static constexpr auto
    GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1(DeviceArch)
    {
        constexpr index_t KPerBlockInByte = BKPerBlock * sizeof(BDataType) / BPackedSize;
        constexpr index_t MWave           = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWave           = NPerBlock / (NXdlPerWave * NPerXdl);
        constexpr index_t WaveSize        = BlockSize / (MWave * NWave);
        // B matrix in LDS memory, dst of blockwise copy
        if constexpr(DirectLoad &&
                     (is_same_v<DeviceArch, gfx950_t> || is_same_v<DeviceArch, gfx9_t>))
        {
            if constexpr(is_same_v<tensor_layout::gemm::RowMajor, BLayout>)
            {
                // FIXME: our support to non-K contiguous layout is limited, only work in some
                // specific setting
                return make_naive_tensor_descriptor_packed(
                    make_tuple(BK0Number, Number<NPerBlock>{}, BK1Number));
            }
            else
            {
                return make_naive_tensor_descriptor(
                    make_tuple(BK0Number, Number<NPerBlock>{}, BK1Number),
                    make_tuple(BK1Number, Number<BKPerBlock>{}, I1));
            }
        }
        else if constexpr(BBlockLdsExtraN || ForceNaiveLdsLayout)
        {
            // bank conflict when writting the data into LDS, but don't worry, we have whole entire
            // loop to hide it in v4. it may give you some benefit from less valu in compute address
            return make_naive_tensor_descriptor(
                make_tuple(BK0Number, Number<NPerBlock>{}, BK1Number),
                make_tuple(Number<NPerBlock + BBlockLdsExtraN>{} * BK1Number, BK1Number, I1));
        }
        else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
        {
            // NLdsLayer * K0 as logical Bank
            constexpr index_t LdsSize       = 32 * 4 / KPerBlockInByte;
            constexpr index_t NLdsLayer     = LdsSize < 1 ? 1 : LdsSize;
            constexpr auto b_lds_block_desc = make_naive_tensor_descriptor(
                make_tuple(
                    BK0Number * Number<NLdsLayer>{}, Number<NPerBlock / NLdsLayer>{}, BK1Number),
                make_tuple(BK1Number, Number<BKPerBlock * NLdsLayer>{}, I1));

            constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
                b_lds_block_desc,
                make_tuple(make_xor_with_modulo_transform(make_tuple(
                               Number<NPerBlock / NLdsLayer>{}, Number<BK0Number * NLdsLayer>{})),
                           make_pass_through_transform(BK1Number)),
                make_tuple(Sequence<1, 0>{}, Sequence<2>{}),
                make_tuple(Sequence<1, 0>{}, Sequence<2>{}));

            constexpr auto b_lds_block_desc_bk0_nldslayer_n_bk1 = transform_tensor_descriptor(
                b_lds_block_desc_permuted,
                make_tuple(make_unmerge_transform(make_tuple(BK0Number, Number<NLdsLayer>{})),
                           make_pass_through_transform(Number<NPerBlock / NLdsLayer>{}),
                           make_pass_through_transform(BK1Number)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}, Sequence<3>{}));

            constexpr auto b_lds_block_desc_bk0_n_bk1 = transform_tensor_descriptor(
                b_lds_block_desc_bk0_nldslayer_n_bk1,
                make_tuple(make_pass_through_transform(BK0Number),
                           make_merge_transform_v3_division_mod(
                               make_tuple(Number<NPerBlock / NLdsLayer>{}, Number<NLdsLayer>{})),
                           make_pass_through_transform(BK1Number)),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            return b_lds_block_desc_bk0_n_bk1;
        }
        else // RowMajor B
        {
            constexpr auto N0 = BBlockTransferThreadClusterLengths_BK0_N_BK1{}.At(I1);
            constexpr auto N1 = NPerBlock / N0;

            constexpr auto KThreadWrite     = BBlockTransferThreadClusterLengths_BK0_N_BK1{}.At(I0);
            constexpr auto K0PerThreadWrite = BK0Number / KThreadWrite;
            constexpr auto KThreadRead      = WaveSize / NPerXdl;
            constexpr auto K0PerThreadRead  = BK0Number / KThreadRead;

            constexpr auto kfold = (BK1Number * N0 * sizeof(BDataType) > 128)
                                       ? 1
                                       : 128 / (BK1Number * N0 * sizeof(BDataType));
            constexpr auto KThreadReadPerm =
                (kfold * K0PerThreadWrite / K0PerThreadRead) > 1
                    ? KThreadRead / (kfold * K0PerThreadWrite / K0PerThreadRead)
                    : KThreadRead;

            // 1<=npair<=n0
            constexpr auto npair = (BK1Number * NPerXdl * sizeof(BDataType) > 128)
                                       ? 1
                                       : ((128 / (BK1Number * NPerXdl * sizeof(BDataType))) > N0
                                              ? N0
                                              : 128 / (BK1Number * NPerXdl * sizeof(BDataType)));

            constexpr auto b_lds_block_desc = make_naive_tensor_descriptor_packed(
                make_tuple(Number<KThreadWrite / kfold / KThreadReadPerm>{},
                           Number<K0PerThreadWrite>{},
                           Number<KThreadReadPerm * N1>{},
                           Number<kfold * N0 / npair>{},
                           Number<npair>{},
                           BK1Number));

            constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
                b_lds_block_desc,
                make_tuple(
                    make_pass_through_transform(Number<KThreadWrite / kfold / KThreadReadPerm>{}),
                    make_pass_through_transform(Number<K0PerThreadWrite>{}),
                    make_xor_with_modulo_transform(
                        make_tuple(Number<KThreadReadPerm * N1>{}, Number<kfold * N0 / npair>{})),
                    make_pass_through_transform(Number<npair>{}),
                    make_pass_through_transform(BK1Number)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}, Sequence<5>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}, Sequence<5>{}));

            constexpr auto b_lds_block_desc_unmerged = transform_tensor_descriptor(
                b_lds_block_desc_permuted,
                make_tuple(
                    make_pass_through_transform(Number<KThreadWrite / kfold / KThreadReadPerm>{}),
                    make_pass_through_transform(Number<K0PerThreadWrite>{}),
                    make_unmerge_transform(make_tuple(Number<KThreadReadPerm>{}, Number<N1>{})),
                    make_unmerge_transform(make_tuple(Number<kfold>{}, Number<N0 / npair>{})),
                    make_pass_through_transform(Number<npair>{}),
                    make_pass_through_transform(BK1Number)),
                make_tuple(Sequence<0>{},
                           Sequence<1>{},
                           Sequence<2>{},
                           Sequence<3>{},
                           Sequence<4>{},
                           Sequence<5>{}),
                make_tuple(Sequence<1>{},
                           Sequence<2>{},
                           Sequence<0, 3>{},
                           Sequence<4, 5>{},
                           Sequence<6>{},
                           Sequence<7>{}));

            constexpr auto b_lds_block_desc_bk0_n_bk1 = transform_tensor_descriptor(
                b_lds_block_desc_unmerged,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(Number<KThreadReadPerm>{},
                                          Number<KThreadWrite / kfold / KThreadReadPerm>{},
                                          Number<kfold>{},
                                          Number<K0PerThreadWrite>{})),
                           make_merge_transform_v3_division_mod(
                               make_tuple(Number<N0 / npair>{}, Number<npair>{}, Number<N1>{})),
                           make_pass_through_transform(BK1Number)),
                make_tuple(Sequence<0, 1, 4, 2>{}, Sequence<5, 6, 3>{}, Sequence<7>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            return b_lds_block_desc_bk0_n_bk1;
        }
    }

    template <>
    __device__ __host__ constexpr auto
    GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1<gfx125_t>(gfx125_t)
    {
        constexpr index_t KPerBlockInByte = BKPerBlock * sizeof(BDataType) / BPackedSize;
        // NLdsLayer * K0 as logical Bank
        constexpr index_t LdsSize     = get_n_lds_banks(gfx125_t{}) * 4 / KPerBlockInByte;
        constexpr bool EnableLdsLayer = BBlockTransferThreadClusterLengths_BK0_N_BK1::Size() == 3 &&
                                        (BBlockTransferThreadClusterLengths_BK0_N_BK1{}[0] *
                                             BBlockTransferThreadClusterLengths_BK0_N_BK1{}[1] *
                                             BBlockTransferThreadClusterLengths_BK0_N_BK1{}[2] ==
                                         BlockSize);
        constexpr index_t NLdsLayer = (EnableLdsLayer == false) || (LdsSize < 1) ? 1 : LdsSize;
        constexpr index_t NPerThread =
            EnableLdsLayer ? NPerBlock / BBlockTransferThreadClusterLengths_BK0_N_BK1{}[1] : 1;

        constexpr index_t NPerThreadLayer = [&]() {
            if constexpr(DirectLoad || NPerThread == 1)
            {
                return 1;
            }
            // Disable MPerThreadLayer if it is non-power two.
            else if constexpr(math::next_power_of_two<NPerThread>() != NPerThread)
            {
                return 1;
            }
            else
            {
                return (NPerThread >= 16) ? 4 : NPerThread;
            }
        }();

        static_assert(NLdsLayer == 1 || NPerBlock % (NLdsLayer * NPerThreadLayer) == 0);
        // B matrix in LDS memory, dst of blockwise copy
        if constexpr(BBlockLdsExtraN || ForceNaiveLdsLayout || DirectLoad)
        {
            // 16 is the byte size of ds_load_b128 and ds_write_b128.
            constexpr auto PaddingSize = 16 / sizeof(BDataType);
            if constexpr(NLdsLayer == 1)
            {
                return make_naive_tensor_descriptor(
                    make_tuple(BK0Number, Number<NPerBlock>{}, BK1Number),
                    make_tuple(BK1Number, Number<BKPerBlock + PaddingSize>{}, I1));
            }
            else
            {
                constexpr auto b_lds_block_desc_bk0_n_unmerge_bk1 = make_naive_tensor_descriptor(
                    make_tuple(BK0Number,
                               Number<NPerBlock / NLdsLayer / NPerThreadLayer>{},
                               Number<NPerThreadLayer>{},
                               Number<NLdsLayer>{},
                               BK1Number),
                    make_tuple(BK1Number,
                               Number<(BKPerBlock * NLdsLayer + PaddingSize) * NPerThreadLayer>{},
                               Number<BKPerBlock * NLdsLayer + PaddingSize>{},
                               Number<BKPerBlock>{},
                               I1));

                return transform_tensor_descriptor(
                    b_lds_block_desc_bk0_n_unmerge_bk1,
                    make_tuple(make_pass_through_transform(BK0Number),
                               make_merge_transform_v3_division_mod(
                                   make_tuple(Number<NPerBlock / NLdsLayer / NPerThreadLayer>{},
                                              Number<NLdsLayer>{},
                                              Number<NPerThreadLayer>{})),
                               make_pass_through_transform(BK1Number)),
                    make_tuple(Sequence<0>{}, Sequence<1, 3, 2>{}, Sequence<4>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
            }
        }
        else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
        {
            constexpr auto b_lds_block_desc = make_naive_tensor_descriptor(
                make_tuple(
                    BK0Number * Number<NLdsLayer>{}, Number<NPerBlock / NLdsLayer>{}, BK1Number),
                make_tuple(BK1Number, Number<BKPerBlock * NLdsLayer>{}, I1));

            constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
                b_lds_block_desc,
                make_tuple(make_xor_with_modulo_transform(make_tuple(
                               Number<NPerBlock / NLdsLayer>{}, Number<BK0Number * NLdsLayer>{})),
                           make_pass_through_transform(BK1Number)),
                make_tuple(Sequence<1, 0>{}, Sequence<2>{}),
                make_tuple(Sequence<1, 0>{}, Sequence<2>{}));

            if constexpr(NLdsLayer == 1)
            {
                return b_lds_block_desc_permuted;
            }
            else
            {
                constexpr auto b_lds_block_desc_bk0_nldslayer_n_bk1 = transform_tensor_descriptor(
                    b_lds_block_desc_permuted,
                    make_tuple(make_unmerge_transform(make_tuple(Number<NLdsLayer>{}, BK0Number)),
                               make_unmerge_transform(
                                   make_tuple(Number<NPerBlock / NLdsLayer / NPerThreadLayer>{},
                                              Number<NPerThreadLayer>{})),
                               make_pass_through_transform(BK1Number)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<2, 0>{}, Sequence<1, 3>{}, Sequence<4>{}));

                constexpr auto b_lds_block_desc_bk0_n_bk1 = transform_tensor_descriptor(
                    b_lds_block_desc_bk0_nldslayer_n_bk1,
                    make_tuple(make_pass_through_transform(BK0Number),
                               make_merge_transform_v3_division_mod(
                                   make_tuple(Number<NPerBlock / NLdsLayer / NPerThreadLayer>{},
                                              Number<NLdsLayer>{},
                                              Number<NPerThreadLayer>{})),
                               make_pass_through_transform(BK1Number)),
                    make_tuple(Sequence<0>{}, Sequence<1, 2, 3>{}, Sequence<4>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

                return b_lds_block_desc_bk0_n_bk1;
            }
        }
        else // RowMajor B
        {
            constexpr index_t MWave    = MPerBlock / (MXdlPerWave * MPerXdl);
            constexpr index_t NWave    = NPerBlock / (NXdlPerWave * NPerXdl);
            constexpr index_t WaveSize = BlockSize / (MWave * NWave);

            constexpr auto LdsBankSize = get_n_lds_banks(gfx125_t{}) * 4;
            constexpr auto N0          = BBlockTransferThreadClusterLengths_BK0_N_BK1{}.At(I1);
            constexpr auto N1          = NPerBlock / N0;

            constexpr auto KThreadWrite     = BBlockTransferThreadClusterLengths_BK0_N_BK1{}.At(I0);
            constexpr auto K0PerThreadWrite = BK0Number / KThreadWrite;
            constexpr auto KThreadRead      = WaveSize / NPerXdl;
            constexpr auto K0PerThreadRead  = BK0Number / KThreadRead;

            constexpr auto kfold = (BK1Number * N0 * sizeof(BDataType) > LdsBankSize)
                                       ? 1
                                       : LdsBankSize / (BK1Number * N0 * sizeof(BDataType));

            constexpr auto KThreadReadPerm =
                (kfold * K0PerThreadWrite / K0PerThreadRead) > 1
                    ? KThreadRead / (kfold * K0PerThreadWrite / K0PerThreadRead)
                    : KThreadRead;

            // 1<=npair<=n0
            constexpr auto npair =
                (BK1Number * NPerXdl * sizeof(BDataType) > (2 * LdsBankSize))
                    ? 1
                    : (((2 * LdsBankSize) / (BK1Number * NPerXdl * sizeof(BDataType))) > N0
                           ? N0
                           : (2 * LdsBankSize) / (BK1Number * NPerXdl * sizeof(BDataType)));

            constexpr auto b_lds_block_desc = make_naive_tensor_descriptor_packed(
                make_tuple(Number<KThreadWrite / kfold / KThreadReadPerm>{},
                           Number<K0PerThreadWrite>{},
                           Number<KThreadReadPerm * N1>{},
                           Number<kfold * N0 / npair>{},
                           Number<npair>{},
                           BK1Number));

            constexpr auto b_lds_block_desc_permuted = transform_tensor_descriptor(
                b_lds_block_desc,
                make_tuple(
                    make_pass_through_transform(Number<KThreadWrite / kfold / KThreadReadPerm>{}),
                    make_pass_through_transform(Number<K0PerThreadWrite>{}),
                    make_xor_with_modulo_transform(
                        make_tuple(Number<KThreadReadPerm * N1>{}, Number<kfold * N0 / npair>{})),
                    make_pass_through_transform(Number<npair>{}),
                    make_pass_through_transform(BK1Number)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}, Sequence<5>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4>{}, Sequence<5>{}));

            constexpr auto b_lds_block_desc_unmerged = transform_tensor_descriptor(
                b_lds_block_desc_permuted,
                make_tuple(
                    make_pass_through_transform(Number<KThreadWrite / kfold / KThreadReadPerm>{}),
                    make_pass_through_transform(Number<K0PerThreadWrite>{}),
                    make_unmerge_transform(make_tuple(Number<KThreadReadPerm>{}, Number<N1>{})),
                    make_unmerge_transform(make_tuple(Number<kfold>{}, Number<N0 / npair>{})),
                    make_pass_through_transform(Number<npair>{}),
                    make_pass_through_transform(BK1Number)),
                make_tuple(Sequence<0>{},
                           Sequence<1>{},
                           Sequence<2>{},
                           Sequence<3>{},
                           Sequence<4>{},
                           Sequence<5>{}),
                make_tuple(Sequence<1>{},
                           Sequence<2>{},
                           Sequence<0, 3>{},
                           Sequence<4, 5>{},
                           Sequence<6>{},
                           Sequence<7>{}));

            constexpr auto b_lds_block_desc_bk0_n_bk1 = transform_tensor_descriptor(
                b_lds_block_desc_unmerged,
                make_tuple(make_merge_transform_v3_division_mod(
                               make_tuple(Number<KThreadReadPerm>{},
                                          Number<KThreadWrite / kfold / KThreadReadPerm>{},
                                          Number<kfold>{},
                                          Number<K0PerThreadWrite>{})),
                           make_merge_transform_v3_division_mod(
                               make_tuple(Number<N0 / npair>{}, Number<npair>{}, Number<N1>{})),
                           make_pass_through_transform(BK1Number)),
                make_tuple(Sequence<0, 1, 4, 2>{}, Sequence<5, 6, 3>{}, Sequence<7>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            return b_lds_block_desc_bk0_n_bk1;
        }
    }

    template <typename DeviceArch>
    __device__ __host__ static constexpr auto
    GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(DeviceArch)
    {
        constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWave = NPerBlock / (NXdlPerWave * NPerXdl);

        constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            make_naive_tensor_descriptor_packed(
                make_tuple(I1,
                           Number<CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl>{},
                           I1,
                           Number<CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>{}));

        return c_shuffle_block_desc_mblock_mperblock_nblock_nperblock;
    }

    template <>
    __device__ __host__ constexpr auto
    GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock<gfx125_t>(gfx125_t)
    {
        constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWave = NPerBlock / (NXdlPerWave * NPerXdl);

        constexpr index_t CShuffleM = CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl;
        constexpr index_t CShuffleN = CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl;
        constexpr index_t LdsSize =
            get_n_lds_banks(gfx125_t{}) * 4 / CShuffleN / sizeof(CShuffleDataType);
        constexpr index_t CShuffleLdsLayer = LdsSize < 1 ? 1 : LdsSize;

        constexpr index_t CShuffleNStride = CShuffleN + 16 / sizeof(CShuffleDataType);
        constexpr index_t CShuffleMPerThread =
            CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::Size() > 2
                ? CShuffleM /
                      CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock{}[I1]
                : 1;

        constexpr bool SupportLdsLayer = (CShuffleM % (CShuffleMPerThread * CShuffleLdsLayer)) == 0;
        if constexpr(CShuffleLdsLayer == 1 || CShuffleMPerThread == 1 || SupportLdsLayer == false)
        {
            constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
                make_naive_tensor_descriptor(
                    make_tuple(I1, Number<CShuffleM>{}, I1, Number<CShuffleN>{}),
                    make_tuple(Number<CShuffleM * CShuffleNStride>{},
                               Number<CShuffleNStride>{},
                               Number<CShuffleNStride>{},
                               I1));
            return c_shuffle_block_desc_mblock_mperblock_nblock_nperblock;
        }
        else
        {

            constexpr auto c_shuffle_block_desc_mblock_m_unmerged_nblock_nperblock =
                make_naive_tensor_descriptor(
                    make_tuple(I1,
                               Number<CShuffleM / CShuffleMPerThread / CShuffleLdsLayer>{},
                               Number<CShuffleMPerThread>{},
                               Number<CShuffleLdsLayer>{},
                               I1,
                               Number<CShuffleN>{}),
                    make_tuple(Number<CShuffleM * CShuffleNStride>{},
                               Number<CShuffleLdsLayer * CShuffleMPerThread * CShuffleNStride>{},
                               Number<CShuffleLdsLayer * CShuffleNStride>{},
                               Number<CShuffleNStride>{},
                               Number<CShuffleNStride>{},
                               I1));
            constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
                transform_tensor_descriptor(
                    c_shuffle_block_desc_mblock_m_unmerged_nblock_nperblock,
                    make_tuple(make_pass_through_transform(I1),
                               make_merge_transform_v3_division_mod(make_tuple(
                                   Number<CShuffleM / CShuffleMPerThread / CShuffleLdsLayer>{},
                                   Number<CShuffleLdsLayer>{},
                                   Number<CShuffleMPerThread>{})),
                               make_pass_through_transform(I1),
                               make_pass_through_transform(Number<CShuffleN>{})),
                    make_tuple(Sequence<0>{}, Sequence<1, 3, 2>{}, Sequence<4>{}, Sequence<5>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

            return c_shuffle_block_desc_mblock_mperblock_nblock_nperblock;
        }
    }

    template <typename DeviceArch>
    __host__ __device__ static constexpr auto
    GetCBlockDescriptor_MBlock_NXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl(DeviceArch)
    {
        constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWave = NPerBlock / (NXdlPerWave * NPerXdl);

        constexpr auto
            c_block_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl =
                make_naive_tensor_descriptor_packed(
                    make_tuple(I1,
                               Number<CShuffleMXdlPerWavePerShuffle>{},
                               Number<MWave * MPerXdl>{},
                               I1,
                               Number<CShuffleNXdlPerWavePerShuffle>{},
                               Number<NWave * NPerXdl>{}));

        return c_block_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl;
    }

    template <>
    __host__ __device__ constexpr auto
    GetCBlockDescriptor_MBlock_NXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl<gfx125_t>(
        gfx125_t)
    {
        constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWave = NPerBlock / (NXdlPerWave * NPerXdl);

        constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(gfx125_t{});
        return transform_tensor_descriptor(
            c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
            make_tuple(make_pass_through_transform(I1),
                       make_unmerge_transform(make_tuple(Number<CShuffleMXdlPerWavePerShuffle>{},
                                                         Number<MWave * MPerXdl>{})),
                       make_pass_through_transform(I1),
                       make_unmerge_transform(make_tuple(Number<CShuffleNXdlPerWavePerShuffle>{},
                                                         Number<NWave * NPerXdl>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}, Sequence<4, 5>{}));
    }

    template <typename ABlockDescriptor_AK0PerBlock_MPerBlock_AK1>
    __host__ __device__ static constexpr auto GetABlockDescriptor_AKB_AK0PerBlock_MPerBlock_AK1(
        const ABlockDescriptor_AK0PerBlock_MPerBlock_AK1&)
    {
        return transform_tensor_descriptor(
            ABlockDescriptor_AK0PerBlock_MPerBlock_AK1{},
            make_tuple(make_unmerge_transform(make_tuple(I1, AK0Number)),
                       make_pass_through_transform(Number<MPerBlock>{}),
                       make_pass_through_transform(AK1Number)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}));
    }

    template <typename BBlockDescriptor_BK0PerBlock_NPerBlock_BK1>
    __host__ __device__ static constexpr auto GetBBlockDescriptor_BKB_BK0PerBlock_NPerBlock_BK1(
        const BBlockDescriptor_BK0PerBlock_NPerBlock_BK1&)
    {
        return transform_tensor_descriptor(
            BBlockDescriptor_BK0PerBlock_NPerBlock_BK1{},
            make_tuple(make_unmerge_transform(make_tuple(I1, BK0Number)),
                       make_pass_through_transform(Number<NPerBlock>{}),
                       make_pass_through_transform(BK1Number)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0, 1>{}, Sequence<2>{}, Sequence<3>{}));
    }

    __host__ __device__ static constexpr auto
    GetCBlockDescriptor_MShuffle_MPerShuffle_NShuffle_NPerShuffle()
    {
        constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWave = NPerBlock / (NXdlPerWave * NPerXdl);

        return make_naive_tensor_descriptor_packed(
            make_tuple(Number<MXdlPerWave / CShuffleMXdlPerWavePerShuffle>{},
                       Number<CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl>{},
                       Number<NXdlPerWave / CShuffleNXdlPerWavePerShuffle>{},
                       Number<CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>{}));
    }

    template <bool BPreshuffle = false, index_t NumLdsBuffer = 1, typename DeviceArch>
    __device__ __host__ static constexpr index_t GetSharedMemoryNumberOfByte(DeviceArch)
    {
        // LDS allocation for A and B: be careful of alignment
        constexpr auto a_block_desc_ak0_m_ak1 =
            GetABlockDescriptor_AK0PerBlock_MPerBlock_AK1(DeviceArch{});
        constexpr auto b_block_desc_bk0_n_bk1 =
            GetBBlockDescriptor_BK0PerBlock_NPerBlock_BK1(DeviceArch{});

        // lds max alignment
        constexpr auto max_lds_align = math::lcm(AK1Number, BK1Number);

        constexpr auto a_block_space_size_aligned = math::integer_least_multiple(
            a_block_desc_ak0_m_ak1.GetElementSpaceSize(), max_lds_align);

        constexpr auto b_block_space_size_aligned =
            BPreshuffle ? 0
                        : math::integer_least_multiple(b_block_desc_bk0_n_bk1.GetElementSpaceSize(),
                                                       max_lds_align);

        // LDS allocation for C shuffle in LDS
        constexpr auto c_block_size = [&]() {
            if constexpr(CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::
                             Size() == 0)
            {
                return 0;
            }
            else if constexpr(
                CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::Size() == 6)
            {
                constexpr auto
                    c_block_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl =
                        GetCBlockDescriptor_MBlock_NXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl(
                            DeviceArch{});
                return c_block_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl
                    .GetElementSpaceSize();
            }
            else
            {
                constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
                    GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(DeviceArch{});
                return c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize();
            }
        }();

        return math::max((a_block_space_size_aligned * sizeof(ADataType) / APackedSize +
                          b_block_space_size_aligned * sizeof(BDataType) / BPackedSize) *
                             NumLdsBuffer,
                         c_block_size * sizeof(CShuffleDataType));
    }

    template <bool TransposeC, typename BlockwiseGemmPipe>
    __device__ static constexpr auto GetCThreadDescriptor()
    {
        if constexpr(TransposeC)
        {
            if constexpr(IsMxGemm)
            {
                return BlockwiseGemmPipe::GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5();
            }
            else
            {
                return BlockwiseGemmPipe::GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();
            }
        }
        else
        {
            if constexpr(IsMxGemm)
            {
                return BlockwiseGemmPipe::GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_M3_M4_M5_N3();
            }
            else
            {
                return BlockwiseGemmPipe::GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();
            }
        }
    }

    template <bool TransposeC, typename BlockwiseGemmPipe, typename CBlockDescriptor>
    __device__ static constexpr auto GetCBlockThreadDescriptor()
    {
        static_assert(
            CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::Size() == 4 ||
                IsMxGemm == false,
            "wrong!");
        if constexpr(TransposeC)
        {
            if constexpr(IsMxGemm)
            {
                constexpr auto MXdlPack = BlockwiseGemmPipe::MXdlPack;
                constexpr auto NXdlPack = BlockwiseGemmPipe::NXdlPack;
                // c_block_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_tmp is only used to get lengths
                constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_tmp =
                    BlockwiseGemmPipe::GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5();

                constexpr auto M1 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_tmp.GetLength(I2);
                constexpr auto N1 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_tmp.GetLength(I3);
                constexpr auto M2 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_tmp.GetLength(I4);
                constexpr auto N2 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_tmp.GetLength(I5);
                constexpr auto M3 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_tmp.GetLength(I6);
                constexpr auto N3 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_tmp.GetLength(I7);
                constexpr auto N4 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_tmp.GetLength(I8);
                constexpr auto N5 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_tmp.GetLength(I9);

                return transform_tensor_descriptor(
                    CBlockDescriptor{},
                    make_tuple(
                        make_freeze_transform(I0),
                        make_unmerge_transform(make_tuple(
                            Number<CShuffleMXdlPerWavePerShuffle / MXdlPack>{}, // M0 (MXdlPerWave)
                                                                                // per shuffle
                            M1,                                                 // M1 = MWave
                            M2,                                                 // M2 = MXdlPack
                            M3)),                                               // M3 = MPerXdl
                        make_freeze_transform(I0),
                        make_unmerge_transform(make_tuple(
                            Number<CShuffleNXdlPerWavePerShuffle / NXdlPack>{}, // N0 (NXdlPerWave)
                                                                                // per shuffle
                            N1,                                                 // N1 = NWave
                            N2,                                                 // N2 = NXdlPack
                            N3, // N2 * N3 * N4 = NPerXdl
                            N4,
                            N5))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<>{},
                               Sequence<0, 2, 4, 6>{},
                               Sequence<>{},
                               Sequence<1, 3, 5, 7, 8, 9>{}));
            }
            else
            {
                // c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp is only used to get lengths
                constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp =
                    BlockwiseGemmPipe::GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();

                constexpr auto M1 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I2);
                constexpr auto N1 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I3);
                constexpr auto M2 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I4);
                constexpr auto N2 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I5);
                constexpr auto N3 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I6);
                constexpr auto N4 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I7);

                if constexpr(CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::
                                 Size() == 6)
                {
                    return transform_tensor_descriptor(
                        CBlockDescriptor{},
                        make_tuple(make_freeze_transform(I0),
                                   make_pass_through_transform(
                                       Number<CShuffleMXdlPerWavePerShuffle>{}), // M0 (MXdlPerWave)
                                                                                 // per shuffle
                                   make_unmerge_transform(make_tuple(M1,         // M1 = MWave
                                                                     M2)),       // M2 = MPerXdl
                                   make_freeze_transform(I0),
                                   make_pass_through_transform(
                                       Number<CShuffleNXdlPerWavePerShuffle>{}), // N0 (NXdlPerWave)
                                                                                 // per shuffle
                                   make_unmerge_transform(make_tuple(N1,         // N1 = NWave
                                                                     N2, // N2 * N3 * N4 = NPerXdl
                                                                     N3,
                                                                     N4))),
                        make_tuple(Sequence<0>{},
                                   Sequence<1>{},
                                   Sequence<2>{},
                                   Sequence<3>{},
                                   Sequence<4>{},
                                   Sequence<5>{}),
                        make_tuple(Sequence<>{},
                                   Sequence<0>{},
                                   Sequence<2, 4>{},
                                   Sequence<>{},
                                   Sequence<1>{},
                                   Sequence<3, 5, 6, 7>{}));
                }
                else
                {
                    return transform_tensor_descriptor(
                        CBlockDescriptor{},
                        make_tuple(make_freeze_transform(I0),
                                   make_unmerge_transform(make_tuple(
                                       Number<CShuffleMXdlPerWavePerShuffle>{}, // M0 (MXdlPerWave)
                                                                                // per shuffle
                                       M1,                                      // M1 = MWave
                                       M2)),                                    // M2 = MPerXdl
                                   make_freeze_transform(I0),
                                   make_unmerge_transform(make_tuple(
                                       Number<CShuffleNXdlPerWavePerShuffle>{}, // N0 (NXdlPerWave)
                                                                                // per shuffle
                                       N1,                                      // N1 = NWave
                                       N2, // N2 * N3 * N4 = NPerXdl
                                       N3,
                                       N4))),
                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                        make_tuple(Sequence<>{},
                                   Sequence<0, 2, 4>{},
                                   Sequence<>{},
                                   Sequence<1, 3, 5, 6, 7>{}));
                }
            }
        }
        else
        {
            if constexpr(IsMxGemm)
            {
                constexpr auto MXdlPack = BlockwiseGemmPipe::MXdlPack;
                constexpr auto NXdlPack = BlockwiseGemmPipe::NXdlPack;
                static_assert(
                    CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::Size() ==
                    4);
                constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp =
                    BlockwiseGemmPipe::GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_M3_M4_M5_N3();

                constexpr auto M1 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I2);
                constexpr auto N1 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I3);
                constexpr auto M2 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I4);
                constexpr auto N2 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I5);
                constexpr auto M3 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I6);
                constexpr auto M4 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I7);
                constexpr auto M5 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I8);
                constexpr auto N3 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I9);
                return transform_tensor_descriptor(
                    CBlockDescriptor{},
                    make_tuple(
                        make_freeze_transform(I0),
                        make_unmerge_transform(make_tuple(
                            Number<CShuffleMXdlPerWavePerShuffle / MXdlPack>{}, // M0 (MXdlPerWave)
                                                                                // per shuffle
                            M1,                                                 // M1 = MWave
                            M2,                                                 // M2 = MXdlPack
                            M3, // M3 * M4 * M5 = MPerXdl
                            M4,
                            M5)),
                        make_freeze_transform(I0),
                        make_unmerge_transform(make_tuple(
                            Number<CShuffleNXdlPerWavePerShuffle / NXdlPack>{}, // N0 (NXdlPerWave)
                                                                                // per shuffle
                            N1,                                                 // N1 = NWave
                            N2,                                                 // N2 = NXdlPack
                            N3))),                                              // N3 = NPerXdl
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<>{},
                               Sequence<0, 2, 4, 6, 7, 8>{},
                               Sequence<>{},
                               Sequence<1, 3, 5, 9>{}));
            }
            else
            {
                // c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp is only used to get lengths
                constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp =
                    BlockwiseGemmPipe::GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

                constexpr auto M1 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I2);
                constexpr auto N1 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I3);
                constexpr auto M2 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I4);
                constexpr auto M3 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I5);
                constexpr auto M4 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I6);
                constexpr auto N2 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I7);

                if constexpr(CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::
                                 Size() == 6)
                {
                    return transform_tensor_descriptor(
                        CBlockDescriptor{},
                        make_tuple(make_freeze_transform(I0), // freeze mblock
                                   make_pass_through_transform(
                                       Number<CShuffleMXdlPerWavePerShuffle>{}), // M0 (MXdlPerWave)
                                                                                 // per shuffle
                                   make_unmerge_transform(make_tuple(
                                       M1, M2, M3, M4)),      // M1 = MWave, M2 * M3 * M4 = MPerXdl
                                   make_freeze_transform(I0), // freeze nblock
                                   make_pass_through_transform(
                                       Number<CShuffleNXdlPerWavePerShuffle>{}), // N0 (NXdlPerWave)
                                                                                 // per shuffle
                                   make_unmerge_transform(
                                       make_tuple(N1, N2))), // M1 = MWave, M2 * M3 * M4 = MPerXdl
                        make_tuple(Sequence<0>{},
                                   Sequence<1>{},
                                   Sequence<2>{},
                                   Sequence<3>{},
                                   Sequence<4>{},
                                   Sequence<5>{}),
                        make_tuple(Sequence<>{},
                                   Sequence<0>{},
                                   Sequence<2, 4, 5, 6>{},
                                   Sequence<>{},
                                   Sequence<1>{},
                                   Sequence<3, 7>{})

                    );
                }
                else
                {
                    return transform_tensor_descriptor(
                        CBlockDescriptor{},
                        make_tuple(make_freeze_transform(I0), // freeze mblock
                                   make_unmerge_transform(
                                       make_tuple(CShuffleMXdlPerWavePerShuffle,
                                                  M1,
                                                  M2,
                                                  M3,
                                                  M4)),       // M1 = MWave, M2 * M3 * M4 = MPerXdl
                                   make_freeze_transform(I0), // freeze nblock
                                   make_unmerge_transform(
                                       make_tuple(CShuffleNXdlPerWavePerShuffle,
                                                  N1,
                                                  N2))), // M1 = MWave, M2 * M3 * M4 = MPerXdl
                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                        make_tuple(Sequence<>{},
                                   Sequence<0, 2, 4, 5, 6>{},
                                   Sequence<>{},
                                   Sequence<1, 3, 7>{}));
                }
            }
        }
    }

    template <bool TransposeC,
              typename BlockwiseGemmPipe,
              typename CThreadDescriptor,
              typename CBlockThreadDescriptor,
              typename CDEElementwiseOperation>
    __device__ static auto
    GetCThreadCopyVgprToLds(const BlockwiseGemmPipe& blockwise_gemm,
                            const CThreadDescriptor&,
                            const CBlockThreadDescriptor& c_block_thread_desc,
                            const CDEElementwiseOperation& cde_element_op)
    {
        const auto c_thread_mtx_on_block =
            blockwise_gemm.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

        const index_t m_thread_data_on_block = c_thread_mtx_on_block[I0];
        const index_t n_thread_data_on_block = c_thread_mtx_on_block[I1];

        if constexpr(TransposeC)
        {
            if constexpr(IsMxGemm)
            {
                constexpr auto MXdlPack = BlockwiseGemmPipe::MXdlPack;
                constexpr auto NXdlPack = BlockwiseGemmPipe::NXdlPack;
                // TODO: hacky, fix it!
                // c_block_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_tmp is only used to get lengths
                constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_tmp =
                    BlockwiseGemmPipe::GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5();
                constexpr auto M0 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_tmp.GetLength(I0);
                constexpr auto N0 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_tmp.GetLength(I1);
                constexpr auto M1 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_tmp.GetLength(I2);
                constexpr auto N1 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_tmp.GetLength(I3);
                constexpr auto M2 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_tmp.GetLength(I4);
                constexpr auto N2 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_tmp.GetLength(I5);
                constexpr auto M3 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_tmp.GetLength(I6);
                constexpr auto N3 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_tmp.GetLength(I7);
                constexpr auto N4 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_tmp.GetLength(I8);
                constexpr auto N5 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_tmp.GetLength(I9);

                // calculate origin of thread output tensor on global memory
                //     blockwise GEMM c matrix starting index
                const auto m_thread_data_on_block_to_m0_m1_m2_m3_adaptor =
                    make_single_stage_tensor_adaptor(
                        make_tuple(make_merge_transform(make_tuple(M0, M1, M2, M3))),
                        make_tuple(Sequence<0, 1, 2, 3>{}),
                        make_tuple(Sequence<0>{}));

                const auto m_thread_data_on_block_idx =
                    m_thread_data_on_block_to_m0_m1_m2_m3_adaptor.CalculateBottomIndex(
                        make_multi_index(m_thread_data_on_block));

                const auto n_thread_data_on_block_to_n0_n1_n2_n3_n4_n5_adaptor =
                    make_single_stage_tensor_adaptor(
                        make_tuple(make_merge_transform(make_tuple(N0, N1, N2, N3, N4, N5))),
                        make_tuple(Sequence<0, 1, 2, 3, 4, 5>{}),
                        make_tuple(Sequence<0>{}));

                const auto n_thread_data_on_block_idx =
                    n_thread_data_on_block_to_n0_n1_n2_n3_n4_n5_adaptor.CalculateBottomIndex(
                        make_multi_index(n_thread_data_on_block));

                // shuffle: threadwise copy C from VGPR to LDS
                return ThreadwiseTensorSliceTransfer_v1r3<
                    AccDataType,
                    CShuffleDataType,
                    CThreadDescriptor,
                    CBlockThreadDescriptor,
                    CDEElementwiseOperation,
                    Sequence<CShuffleMXdlPerWavePerShuffle / MXdlPack,
                             CShuffleNXdlPerWavePerShuffle / NXdlPack,
                             I1,
                             I1,
                             M2,
                             N2,
                             I1,
                             N3,
                             I1,
                             N5>,
                    Sequence<0, 1, 2, 3, 4, 5, 6, 7, 8, 9>,
                    9,
                    1,
                    InMemoryDataOperationEnum::Set,
                    1,
                    true>{c_block_thread_desc,
                          make_multi_index(0,
                                           0,
                                           m_thread_data_on_block_idx[I1],
                                           n_thread_data_on_block_idx[I1],
                                           m_thread_data_on_block_idx[I2],
                                           n_thread_data_on_block_idx[I2],
                                           m_thread_data_on_block_idx[I3],
                                           n_thread_data_on_block_idx[I3],
                                           n_thread_data_on_block_idx[I4],
                                           n_thread_data_on_block_idx[I5]),
                          cde_element_op};
            }
            else
            {

                // TODO: hacky, fix it!
                // c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp is only used to get lengths
                constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp =
                    BlockwiseGemmPipe::GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();

                constexpr auto M0 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I0);
                constexpr auto N0 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I1);
                constexpr auto M1 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I2);
                constexpr auto N1 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I3);
                constexpr auto M2 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I4);
                constexpr auto N2 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I5);
                constexpr auto N3 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I6);
                constexpr auto N4 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I7);

                // calculate origin of thread output tensor on global memory
                //     blockwise GEMM c matrix starting index
                const auto m_thread_data_on_block_to_m0_m1_m2_adaptor =
                    make_single_stage_tensor_adaptor(
                        make_tuple(make_merge_transform(make_tuple(M0, M1, M2))),
                        make_tuple(Sequence<0, 1, 2>{}),
                        make_tuple(Sequence<0>{}));

                const auto m_thread_data_on_block_idx =
                    m_thread_data_on_block_to_m0_m1_m2_adaptor.CalculateBottomIndex(
                        make_multi_index(m_thread_data_on_block));

                const auto n_thread_data_on_block_to_n0_n1_n2_n3_n4_adaptor =
                    make_single_stage_tensor_adaptor(
                        make_tuple(make_merge_transform(make_tuple(N0, N1, N2, N3, N4))),
                        make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                        make_tuple(Sequence<0>{}));

                const auto n_thread_data_on_block_idx =
                    n_thread_data_on_block_to_n0_n1_n2_n3_n4_adaptor.CalculateBottomIndex(
                        make_multi_index(n_thread_data_on_block));

                // shuffle: threadwise copy C from VGPR to LDS
                return ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                                          CShuffleDataType,
                                                          CThreadDescriptor,
                                                          CBlockThreadDescriptor,
                                                          CDEElementwiseOperation,
                                                          Sequence<CShuffleMXdlPerWavePerShuffle,
                                                                   CShuffleNXdlPerWavePerShuffle,
                                                                   I1,
                                                                   I1,
                                                                   I1,
                                                                   N2,
                                                                   I1,
                                                                   N4>,
                                                          Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                                          7,
                                                          1,
                                                          InMemoryDataOperationEnum::Set,
                                                          1,
                                                          true>{
                    c_block_thread_desc,
                    make_multi_index(0,
                                     0,
                                     m_thread_data_on_block_idx[I1],
                                     n_thread_data_on_block_idx[I1],
                                     m_thread_data_on_block_idx[I2],
                                     n_thread_data_on_block_idx[I2],
                                     n_thread_data_on_block_idx[I3],
                                     n_thread_data_on_block_idx[I4]),
                    cde_element_op};
            }
        }
        else
        {
            if constexpr(IsMxGemm)
            {
                constexpr auto MXdlPack = BlockwiseGemmPipe::MXdlPack;
                constexpr auto NXdlPack = BlockwiseGemmPipe::NXdlPack;
                constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3_tmp =
                    BlockwiseGemmPipe::GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_M3_M4_M5_N3();

                constexpr auto M0 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3_tmp.GetLength(I0);
                constexpr auto N0 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3_tmp.GetLength(I1);
                constexpr auto M1 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3_tmp.GetLength(I2);
                constexpr auto N1 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3_tmp.GetLength(I3);
                constexpr auto M2 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3_tmp.GetLength(I4);
                constexpr auto N2 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3_tmp.GetLength(I5);
                constexpr auto M3 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3_tmp.GetLength(I6);
                constexpr auto M4 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3_tmp.GetLength(I7);
                constexpr auto M5 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3_tmp.GetLength(I8);
                constexpr auto N3 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3_tmp.GetLength(I9);
                // calculate origin of thread output tensor on global memory
                //     blockwise GEMM c matrix starting index
                const auto m_thread_data_on_block_to_m0_m1_m2_m3_m4_adaptor =
                    make_single_stage_tensor_adaptor(
                        make_tuple(make_merge_transform(make_tuple(M0, M1, M2, M3, M4, M5))),
                        make_tuple(Sequence<0, 1, 2, 3, 4, 5>{}),
                        make_tuple(Sequence<0>{}));

                const auto m_thread_data_on_block_idx =
                    m_thread_data_on_block_to_m0_m1_m2_m3_m4_adaptor.CalculateBottomIndex(
                        make_multi_index(m_thread_data_on_block));

                const auto n_thread_data_on_block_to_n0_n1_n2_adaptor =
                    make_single_stage_tensor_adaptor(
                        make_tuple(make_merge_transform(make_tuple(N0, N1, N2, N3))),
                        make_tuple(Sequence<0, 1, 2, 3>{}),
                        make_tuple(Sequence<0>{}));

                const auto n_thread_data_on_block_idx =
                    n_thread_data_on_block_to_n0_n1_n2_adaptor.CalculateBottomIndex(
                        make_multi_index(n_thread_data_on_block));

                // shuffle: threadwise copy C from VGPR to LDS
                return ThreadwiseTensorSliceTransfer_v1r3<
                    AccDataType,
                    CShuffleDataType,
                    CThreadDescriptor,
                    CBlockThreadDescriptor,
                    CDEElementwiseOperation,
                    Sequence<CShuffleMXdlPerWavePerShuffle / MXdlPack,
                             CShuffleNXdlPerWavePerShuffle / NXdlPack,
                             I1,
                             I1,
                             M2,
                             N2,
                             M3,
                             I1,
                             M5,
                             I1>,
                    Sequence<0, 1, 2, 3, 4, 5, 6, 7, 8, 9>,
                    9,
                    1,
                    InMemoryDataOperationEnum::Set,
                    1,
                    true>{c_block_thread_desc,
                          make_multi_index(0,
                                           0,
                                           m_thread_data_on_block_idx[I1],
                                           n_thread_data_on_block_idx[I1],
                                           m_thread_data_on_block_idx[I2],
                                           n_thread_data_on_block_idx[I2],
                                           m_thread_data_on_block_idx[I3],
                                           m_thread_data_on_block_idx[I4],
                                           m_thread_data_on_block_idx[I5],
                                           n_thread_data_on_block_idx[I3]),
                          cde_element_op};
            }
            else
            {
                // TODO: hacky, fix it!
                // c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp is only used to get lengths
                constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp =
                    BlockwiseGemmPipe::GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

                constexpr auto M0 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I0);
                constexpr auto N0 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I1);
                constexpr auto M1 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I2);
                constexpr auto N1 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I3);
                constexpr auto M2 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I4);
                constexpr auto M3 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I5);
                constexpr auto M4 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I6);
                constexpr auto N2 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I7);

                // calculate origin of thread output tensor on global memory
                //     blockwise GEMM c matrix starting index

                const auto m_thread_data_on_block_to_m0_m1_m2_m3_m4_adaptor =
                    make_single_stage_tensor_adaptor(
                        make_tuple(make_merge_transform(make_tuple(M0, M1, M2, M3, M4))),
                        make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                        make_tuple(Sequence<0>{}));

                const auto m_thread_data_on_block_idx =
                    m_thread_data_on_block_to_m0_m1_m2_m3_m4_adaptor.CalculateBottomIndex(
                        make_multi_index(m_thread_data_on_block));

                const auto n_thread_data_on_block_to_n0_n1_n2_adaptor =
                    make_single_stage_tensor_adaptor(
                        make_tuple(make_merge_transform(make_tuple(N0, N1, N2))),
                        make_tuple(Sequence<0, 1, 2>{}),
                        make_tuple(Sequence<0>{}));

                const auto n_thread_data_on_block_idx =
                    n_thread_data_on_block_to_n0_n1_n2_adaptor.CalculateBottomIndex(
                        make_multi_index(n_thread_data_on_block));

                // shuffle: threadwise copy C from VGPR to LDS
                return ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                                          CShuffleDataType,
                                                          CThreadDescriptor,
                                                          CBlockThreadDescriptor,
                                                          CDEElementwiseOperation,
                                                          Sequence<CShuffleMXdlPerWavePerShuffle,
                                                                   CShuffleNXdlPerWavePerShuffle,
                                                                   I1,
                                                                   I1,
                                                                   M2,
                                                                   I1,
                                                                   M4,
                                                                   I1>,
                                                          Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                                          7,
                                                          1,
                                                          InMemoryDataOperationEnum::Set,
                                                          1,
                                                          true>{
                    c_block_thread_desc,
                    make_multi_index(0,
                                     0,
                                     m_thread_data_on_block_idx[I1],
                                     n_thread_data_on_block_idx[I1],
                                     m_thread_data_on_block_idx[I2],
                                     m_thread_data_on_block_idx[I3],
                                     m_thread_data_on_block_idx[I4],
                                     n_thread_data_on_block_idx[I2]),
                    cde_element_op};
            }
        }
    }

    template <bool TransposeC, typename BlockwiseGemmPipe>
    __device__ static constexpr auto GetCThreadWiseSpaceFillingCurve()
    {
        if constexpr(TransposeC)
        {
            if constexpr(IsMxGemm)
            {
                constexpr auto MXdlPack = BlockwiseGemmPipe::MXdlPack;
                constexpr auto NXdlPack = BlockwiseGemmPipe::NXdlPack;
                constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_tmp =
                    BlockwiseGemmPipe::GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_M3_N3_N4_N5();

                constexpr auto N3 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_tmp.GetLength(I7);
                constexpr auto N5 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_n3_n4_n5_tmp.GetLength(I9);

                return SpaceFillingCurve<Sequence<MXdlPerWave / MXdlPack,
                                                  NXdlPerWave / NXdlPack,
                                                  1,
                                                  1,
                                                  MXdlPack,
                                                  NXdlPack,
                                                  1,
                                                  N3,
                                                  1,
                                                  N5>,
                                         Sequence<0, 1, 2, 3, 4, 5, 6, 7, 8, 9>,
                                         Sequence<CShuffleMXdlPerWavePerShuffle / MXdlPack,
                                                  CShuffleNXdlPerWavePerShuffle / NXdlPack,
                                                  1,
                                                  1,
                                                  MXdlPack,
                                                  NXdlPack,
                                                  1,
                                                  N3,
                                                  1,
                                                  N5>>{};
            }
            else
            {
                constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp =
                    BlockwiseGemmPipe::GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4();

                constexpr auto N2 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I5);
                constexpr auto N4 = c_block_desc_m0_n0_m1_n1_m2_n2_n3_n4_tmp.GetLength(I7);

                return SpaceFillingCurve<Sequence<MXdlPerWave, NXdlPerWave, 1, 1, 1, N2, 1, N4>,
                                         Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                         Sequence<CShuffleMXdlPerWavePerShuffle,
                                                  CShuffleNXdlPerWavePerShuffle,
                                                  1,
                                                  1,
                                                  1,
                                                  N2,
                                                  1,
                                                  N4>>{};
            }
        }
        else
        {
            if constexpr(IsMxGemm)
            {
                constexpr auto MXdlPack = BlockwiseGemmPipe::MXdlPack;
                constexpr auto NXdlPack = BlockwiseGemmPipe::NXdlPack;
                constexpr auto c_block_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3_tmp =
                    BlockwiseGemmPipe::GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_M3_M4_M5_N3();

                constexpr auto M3 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3_tmp.GetLength(I6);
                constexpr auto M5 = c_block_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3_tmp.GetLength(I8);

                return SpaceFillingCurve<Sequence<MXdlPerWave / MXdlPack,
                                                  NXdlPerWave / NXdlPack,
                                                  1,
                                                  1,
                                                  MXdlPack,
                                                  NXdlPack,
                                                  M3,
                                                  1,
                                                  M5,
                                                  1>,
                                         Sequence<0, 1, 2, 3, 4, 5, 6, 7, 8, 9>,
                                         Sequence<CShuffleMXdlPerWavePerShuffle / MXdlPack,
                                                  CShuffleNXdlPerWavePerShuffle / NXdlPack,
                                                  1,
                                                  1,
                                                  MXdlPack,
                                                  NXdlPack,
                                                  M3,
                                                  1,
                                                  M5,
                                                  1>>{};
            }
            else
            {

                constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp =
                    BlockwiseGemmPipe::GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

                constexpr auto M2 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I4);
                constexpr auto M4 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2_tmp.GetLength(I6);
                return SpaceFillingCurve<Sequence<MXdlPerWave, NXdlPerWave, 1, 1, M2, 1, M4, 1>,
                                         Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
                                         Sequence<CShuffleMXdlPerWavePerShuffle,
                                                  CShuffleNXdlPerWavePerShuffle,
                                                  1,
                                                  1,
                                                  M2,
                                                  1,
                                                  M4,
                                                  1>>{};
            }
        }
    }
    template <InMemoryDataOperationEnum CGlobalMemoryDataOperation,
              bool TransposeC,
              typename CThreadTransferSrcDstAccessOrder,
              index_t CThreadTransferSrcDstVectorDim,
              typename BlockwiseGemmPipe,
              typename CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2,
              typename CThreadBuffer,
              typename CDEElementwiseOperation>
    __device__ static void RunEpilogueNoShuffle(
        BlockwiseGemmPipe& blockwise_gemm,
        const CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2& c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2,
        CThreadBuffer& c_thread_buf,
        index_t block_m_id,
        index_t block_n_id,
        EDataType* p_c_grid,
        const CDEElementwiseOperation& cde_element_op)
    {
        static_assert(IsMxGemm == false);

        auto c_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_grid, c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetElementSpaceSize());
        static_assert(TransposeC == false);
        const index_t m_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_m_id * MPerBlock);

        const index_t n_block_data_idx_on_grid =
            __builtin_amdgcn_readfirstlane(block_n_id * NPerBlock);

        constexpr auto c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
            BlockwiseGemmPipe::GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

        constexpr auto c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2 =
            BlockwiseGemmPipe::GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2();

        constexpr auto M0 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I0);
        constexpr auto N0 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I1);
        constexpr auto M1 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I2);
        constexpr auto N1 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I3);
        constexpr auto M2 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I4);
        constexpr auto M3 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I5);
        constexpr auto M4 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I6);
        constexpr auto N2 = c_block_desc_m0_n0_m1_n1_m2_m3_m4_n2.GetLength(I7);

        // calculate origin of thread output tensor on global memory
        //     blockwise GEMM c matrix starting index
        const auto c_thread_mtx_on_block =
            blockwise_gemm.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

        const index_t m_thread_data_on_grid = m_block_data_idx_on_grid + c_thread_mtx_on_block[I0];

        const index_t n_thread_data_on_grid = n_block_data_idx_on_grid + c_thread_mtx_on_block[I1];

        const auto m_thread_data_on_grid_to_m0_m1_m2_m3_m4_adaptor =
            make_single_stage_tensor_adaptor(
                make_tuple(make_merge_transform(make_tuple(M0, M1, M2, M3, M4))),
                make_tuple(Sequence<0, 1, 2, 3, 4>{}),
                make_tuple(Sequence<0>{}));

        const auto m_thread_data_on_grid_idx =
            m_thread_data_on_grid_to_m0_m1_m2_m3_m4_adaptor.CalculateBottomIndex(
                make_multi_index(m_thread_data_on_grid));

        const auto n_thread_data_on_grid_to_n0_n1_n2_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(make_tuple(N0, N1, N2))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));

        const auto n_thread_data_on_grid_idx =
            n_thread_data_on_grid_to_n0_n1_n2_adaptor.CalculateBottomIndex(
                make_multi_index(n_thread_data_on_grid));

        auto c_thread_copy =
            ThreadwiseTensorSliceTransfer_v1r3<AccDataType,
                                               EDataType,
                                               decltype(c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2),
                                               decltype(c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2),
                                               CDEElementwiseOperation,
                                               Sequence<M0, N0, I1, I1, M2, I1, M4, I1>,
                                               CThreadTransferSrcDstAccessOrder,
                                               CThreadTransferSrcDstVectorDim,
                                               CShuffleBlockTransferScalarPerVector_NPerBlock,
                                               CGlobalMemoryDataOperation,
                                               1,
                                               true>{
                c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                make_multi_index(m_thread_data_on_grid_idx[I0],
                                 n_thread_data_on_grid_idx[I0],
                                 m_thread_data_on_grid_idx[I1],
                                 n_thread_data_on_grid_idx[I1],
                                 m_thread_data_on_grid_idx[I2],
                                 m_thread_data_on_grid_idx[I3],
                                 m_thread_data_on_grid_idx[I4],
                                 n_thread_data_on_grid_idx[I2]),
                cde_element_op};

        c_thread_copy.Run(c_thread_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                          make_tuple(I0, I0, I0, I0, I0, I0, I0, I0),
                          c_thread_buf,
                          c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                          c_grid_buf);
    }

    template <InMemoryDataOperationEnum CGlobalMemoryDataOperation,
              bool DoElementwiseBeforeCShuffle,
              bool TransposeC,
              typename BlockwiseGemmPipe,
              typename CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
              typename CThreadBuffer,
              typename CDEElementwiseOperation>
    __device__ static void RunEpilogue(const BlockwiseGemmPipe& blockwise_gemm_pipeline,
                                       const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock&
                                           c_grid_desc_mblock_mperblock_nblock_nperblock,
                                       CThreadBuffer& c_thread_buf,
                                       index_t block_m_id,
                                       index_t block_n_id,
                                       void* p_shared,
                                       EDataType* p_c_grid,
                                       const CDEElementwiseOperation& cde_element_op)
    {
        static_assert(MXdlPerWave % CShuffleMXdlPerWavePerShuffle == 0 &&
                          NXdlPerWave % CShuffleNXdlPerWavePerShuffle == 0,
                      "wrong!");

        auto c_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_grid, c_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

        constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWave = NPerBlock / (NXdlPerWave * NPerXdl);
        tensor_operation::element_wise::PassThrough pass_through{};
        const auto& vpgr_to_lds_element_op = [&] {
            if constexpr(DoElementwiseBeforeCShuffle)
            {
                return cde_element_op;
            }
            else
            {
                return pass_through;
            }
        };
        const auto& lds_to_global_element_op = [&] {
            if constexpr(!DoElementwiseBeforeCShuffle)
            {
                return cde_element_op;
            }
            else
            {
                return pass_through;
            }
        };

        constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(get_device_arch());
        auto c_shuffle_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<CShuffleDataType*>(p_shared),
            c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

        constexpr auto c_thread_desc       = GetCThreadDescriptor<TransposeC, BlockwiseGemmPipe>();
        constexpr auto c_block_thread_desc = GetCBlockThreadDescriptor<
            TransposeC,
            BlockwiseGemmPipe,
            decltype(c_shuffle_block_desc_mblock_mperblock_nblock_nperblock)>();

        auto c_thread_copy_vgpr_to_lds = GetCThreadCopyVgprToLds<TransposeC>(
            blockwise_gemm_pipeline, c_thread_desc, c_block_thread_desc, vpgr_to_lds_element_op());

        // const auto c_thread_mtx_on_block =
        //     blockwise_gemm_pipeline.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

        // const index_t m_thread_data_on_block = c_thread_mtx_on_block[I0];
        // const index_t n_thread_data_on_block = c_thread_mtx_on_block[I1];

        // shuffle: blockwise copy C from LDS to global
        auto c_shuffle_block_copy_lds_to_global = ThreadGroupTensorSliceTransfer_v6r1<
            ThisThreadBlock, // ThreadGroup
            conditional_t<!DoElementwiseBeforeCShuffle,
                          CDEElementwiseOperation,
                          tensor_operation::element_wise::PassThrough>,
            CGlobalMemoryDataOperation, // DstInMemOp,
            Sequence<1,
                     CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                     1,
                     CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>, // BlockSliceLengths,
            CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
            Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
            CShuffleDataType,     // typename SrcData,
            EDataType,            // typename DstData,
            decltype(c_shuffle_block_desc_mblock_mperblock_nblock_nperblock),
            decltype(c_grid_desc_mblock_mperblock_nblock_nperblock),
            Sequence<0, 1, 2, 3>,                           // typename DimAccessOrder,
            3,                                              // index_t VectorDim,
            CShuffleBlockTransferScalarPerVector_NPerBlock, // index_t ScalarPerVector,
            true,  // bool ThreadTransferSrcResetCoordinateAfterRun,
            false> // bool ThreadTransferDstResetCoordinateAfterRun>
            {c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
             make_multi_index(0, 0, 0, 0),
             c_grid_desc_mblock_mperblock_nblock_nperblock,
             make_multi_index(block_m_id, 0, block_n_id, 0),
             lds_to_global_element_op()};

        // space filling curve for threadwise C in VGPR
        constexpr auto sfc_c_vgpr =
            GetCThreadWiseSpaceFillingCurve<TransposeC, BlockwiseGemmPipe>();

        // space filling curve for shuffled blockwise C in global mem
        constexpr auto sfc_c_global =
            SpaceFillingCurve<Sequence<1, MPerBlock, 1, NPerBlock>,
                              Sequence<0, 2, 1, 3>,
                              Sequence<1,
                                       CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                                       1,
                                       CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>>{};

        constexpr index_t num_access = sfc_c_vgpr.GetNumOfAccess();

        static_assert(num_access == sfc_c_global.GetNumOfAccess(), "wrong!");

        static_for<0, num_access, 1>{}([&](auto access_id) {
            // make sure it's safe to write to LDS
            block_sync_lds();

            // each thread write its data from VGPR to LDS
            c_thread_copy_vgpr_to_lds.Run(c_thread_desc,
                                          sfc_c_vgpr.GetIndexTupleOfNumber(access_id),
                                          c_thread_buf,
                                          c_block_thread_desc,
                                          c_shuffle_block_buf);

            // make sure it's safe to read from LDS
            block_sync_lds();

            // each block copy its data from LDS to global
            c_shuffle_block_copy_lds_to_global.Run(
                c_shuffle_block_desc_mblock_mperblock_nblock_nperblock,
                c_shuffle_block_buf,
                c_grid_desc_mblock_mperblock_nblock_nperblock,
                c_grid_buf);

            if constexpr(access_id < num_access - 1)
            {
                constexpr auto c_global_step = sfc_c_global.GetForwardStep(access_id);

                // move on C
                c_shuffle_block_copy_lds_to_global.MoveDstSliceWindow(
                    c_grid_desc_mblock_mperblock_nblock_nperblock, c_global_step);
            }
        });
    }

    template <InMemoryDataOperationEnum CGlobalMemoryDataOperation,
              bool DoElementwiseBeforeCShuffle,
              bool TransposeC,
              bool IsLegacy,
              index_t NumDTensor_  = NumDTensor,
              typename DsDataType_ = DsDataType,
              typename BlockwiseGemmPipe,
              typename DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
              typename CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
              typename CThreadBuffer,
              typename DsGridPointer,
              typename CDEElementwiseOperation>
    __device__ static void
    RunMultiDEpilogue(BlockwiseGemmPipe& blockwise_gemm_pipeline,
                      const DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock&
                          ds_grid_desc_mblock_mperblock_nblock_nperblock,
                      const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock&
                          c_grid_desc_mblock_mperblock_nblock_nperblock,
                      CThreadBuffer& c_thread_buf,
                      index_t block_m_id,
                      index_t block_n_id,
                      void* p_shared,
                      DsGridPointer& p_ds_grid,
                      EDataType* p_c_grid,
                      const CDEElementwiseOperation& cde_element_op)
    {
        static_assert(MXdlPerWave % CShuffleMXdlPerWavePerShuffle == 0 &&
                          NXdlPerWave % CShuffleNXdlPerWavePerShuffle == 0,
                      "wrong!");

        tensor_operation::element_wise::PassThrough pass_through{};
        const auto& vpgr_to_lds_element_op = [&] {
            if constexpr(DoElementwiseBeforeCShuffle)
            {
                return cde_element_op;
            }
            else
            {
                return pass_through;
            }
        };
        const auto& lds_to_global_element_op = [&] {
            if constexpr(!DoElementwiseBeforeCShuffle)
            {
                return cde_element_op;
            }
            else
            {
                return pass_through;
            }
        };

        auto c_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_grid, c_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

        constexpr index_t MWave = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWave = NPerBlock / (NXdlPerWave * NPerXdl);
        constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(get_device_arch());

        auto c_shuffle_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<CShuffleDataType*>(p_shared),
            c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

        constexpr auto c_thread_desc       = GetCThreadDescriptor<TransposeC, BlockwiseGemmPipe>();
        constexpr auto c_block_thread_desc = GetCBlockThreadDescriptor<
            TransposeC,
            BlockwiseGemmPipe,
            decltype(c_shuffle_block_desc_mblock_mperblock_nblock_nperblock)>();

        auto c_thread_copy_vgpr_to_lds = GetCThreadCopyVgprToLds<TransposeC>(
            blockwise_gemm_pipeline, c_thread_desc, c_block_thread_desc, vpgr_to_lds_element_op());

        // const auto c_thread_mtx_on_block =
        //     blockwise_gemm_pipeline.CalculateCThreadOriginDataIndex(I0, I0, I0, I0);

        // const index_t m_thread_data_on_block = c_thread_mtx_on_block[I0];
        // const index_t n_thread_data_on_block = c_thread_mtx_on_block[I1];

        const auto ds_grid_buf = generate_tuple(
            [&](auto i) {
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    p_ds_grid[i],
                    ds_grid_desc_mblock_mperblock_nblock_nperblock[i].GetElementSpaceSize());
            },
            Number<NumDTensor_>{});

        // tuple of reference to C/Ds tensor descriptors
        const auto c_ds_desc_refs = concat_tuple_of_reference(
            tie(c_shuffle_block_desc_mblock_mperblock_nblock_nperblock),
            generate_tie([&](auto i) -> const auto& // return type should be reference
                         { return ds_grid_desc_mblock_mperblock_nblock_nperblock[i]; },
                         Number<NumDTensor_>{}));

        // tuple of reference to C/Ds tensor descriptors
        const auto c_ds_buf_refs = concat_tuple_of_reference(
            tie(c_shuffle_block_buf),
            generate_tie([&](auto i) -> const auto& // return type should be reference
                         { return ds_grid_buf[i]; },
                         Number<NumDTensor_>{}));

        // tuple of starting index of C/Ds blockwise copy
        const auto idx_c_ds_block_begin = container_concat(
            make_tuple(make_multi_index(0, 0, 0, 0)),
            generate_tuple([&](auto) { return make_multi_index(block_m_id, 0, block_n_id, 0); },
                           Number<NumDTensor_>{}));

        // shuffle: blockwise copy C from LDS to global
        auto cde_block_copy_lds_and_global = [&]() {
            if constexpr(IsLegacy)
            {
                return ThreadGroupTensorSliceTransfer_v7<
                    ThisThreadBlock,
                    decltype(container_concat(make_tuple(CShuffleDataType{}), DsDataType_{})),
                    Tuple<EDataType>,
                    decltype(c_ds_desc_refs),
                    decltype(tie(c_grid_desc_mblock_mperblock_nblock_nperblock)),
                    conditional_t<!DoElementwiseBeforeCShuffle,
                                  CDEElementwiseOperation,
                                  tensor_operation::element_wise::PassThrough>,
                    Sequence<static_cast<index_t>(CGlobalMemoryDataOperation)>, // FIXME: make
                                                                                // Sequence support
                                                                                // arbitray type
                    Sequence<1,
                             CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                             1,
                             CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>, // BlockSliceLengths,
                    CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                    Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
                    Sequence<0, 1, 2, 3>, // typename DimAccessOrder,
                    3,                    // index_t VectorDim,
                    CShuffleBlockTransferScalarPerVector_NPerBlock,
                    sequence_merge_t<Sequence<true>,
                                     uniform_sequence_gen_t<
                                         NumDTensor_,
                                         false>>, // ThreadTransferSrcResetCoordinateAfterRunFlags
                    Sequence<false>>              // ThreadTransferDstResetCoordinateAfterRunFlags
                    {c_ds_desc_refs,
                     idx_c_ds_block_begin,
                     tie(c_grid_desc_mblock_mperblock_nblock_nperblock),
                     make_tuple(make_multi_index(block_m_id, 0, block_n_id, 0)),
                     lds_to_global_element_op()};
            }
            else if constexpr(CDEShuffleBlockTransferScalarPerVectors::Size() == 1 &&
                              NumDTensor_ != 0)
            {
                return ThreadGroupTensorSliceTransfer_v7r2<
                    ThisThreadBlock,
                    decltype(container_concat(make_tuple(CShuffleDataType{}), DsDataType_{})),
                    Tuple<EDataType>,
                    decltype(c_ds_desc_refs),
                    decltype(tie(c_grid_desc_mblock_mperblock_nblock_nperblock)),
                    conditional_t<!DoElementwiseBeforeCShuffle,
                                  CDEElementwiseOperation,
                                  tensor_operation::element_wise::PassThrough>,
                    Sequence<static_cast<index_t>(CGlobalMemoryDataOperation)>, // FIXME: make
                                                                                // Sequence support
                                                                                // arbitray type
                    Sequence<1,
                             CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                             1,
                             CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>, // BlockSliceLengths,
                    CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                    Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
                    Sequence<0, 1, 2, 3>, // typename SrcDimAccessOrder,
                    Sequence<0, 1, 2, 3>, // typename DstDimAccessOrder,
                    3,                    // index_t SrcVectorDim,
                    3,                    // index_t DstVectorDim,
                    CShuffleBlockTransferScalarPerVector_NPerBlock,
                    CShuffleBlockTransferScalarPerVector_NPerBlock,
                    sequence_merge_t<Sequence<true>,
                                     uniform_sequence_gen_t<
                                         NumDTensor_,
                                         false>>, // ThreadTransferSrcResetCoordinateAfterRunFlags
                    Sequence<false>>              // ThreadTransferDstResetCoordinateAfterRunFlags
                    {c_ds_desc_refs,
                     idx_c_ds_block_begin,
                     tie(c_grid_desc_mblock_mperblock_nblock_nperblock),
                     make_tuple(make_multi_index(block_m_id, 0, block_n_id, 0)),
                     lds_to_global_element_op()};
            }
            else
            {
                return ThreadGroupTensorSliceTransfer_v7r3<
                    ThisThreadBlock,
                    decltype(container_concat(make_tuple(CShuffleDataType{}), DsDataType_{})),
                    Tuple<EDataType>,
                    decltype(c_ds_desc_refs),
                    decltype(tie(c_grid_desc_mblock_mperblock_nblock_nperblock)),
                    conditional_t<!DoElementwiseBeforeCShuffle,
                                  CDEElementwiseOperation,
                                  tensor_operation::element_wise::PassThrough>,
                    Sequence<static_cast<index_t>(CGlobalMemoryDataOperation)>, // FIXME: make
                                                                                // Sequence support
                                                                                // arbitray type
                    Sequence<1,
                             CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                             1,
                             CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>, // BlockSliceLengths,
                    CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                    Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
                    Sequence<0, 1, 2, 3>, // typename SrcDimAccessOrder,
                    Sequence<0, 1, 2, 3>, // typename DstDimAccessOrder,
                    3,                    // index_t SrcVectorDim,
                    3,                    // index_t DstVectorDim,
                    CDEShuffleBlockTransferScalarPerVectors,
                    CShuffleBlockTransferScalarPerVector_NPerBlock,
                    sequence_merge_t<Sequence<true>,
                                     uniform_sequence_gen_t<
                                         NumDTensor_,
                                         false>>, // ThreadTransferSrcResetCoordinateAfterRunFlags
                    Sequence<false>>              // ThreadTransferDstResetCoordinateAfterRunFlags
                    {c_ds_desc_refs,
                     idx_c_ds_block_begin,
                     tie(c_grid_desc_mblock_mperblock_nblock_nperblock),
                     make_tuple(make_multi_index(block_m_id, 0, block_n_id, 0)),
                     lds_to_global_element_op()};
            }
        }();

        // space filling curve for threadwise C in VGPR
        constexpr auto sfc_c_vgpr =
            GetCThreadWiseSpaceFillingCurve<TransposeC, BlockwiseGemmPipe>();
        // space filling curve for shuffled blockwise C in global mem
        constexpr auto sfc_c_global =
            SpaceFillingCurve<Sequence<1, MPerBlock, 1, NPerBlock>,
                              Sequence<0, 2, 1, 3>,
                              Sequence<1,
                                       CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                                       1,
                                       CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>>{};

        constexpr index_t num_access = sfc_c_vgpr.GetNumOfAccess();

        static_assert(num_access == sfc_c_global.GetNumOfAccess(), "wrong!");

        static_for<0, num_access, 1>{}([&](auto access_id) {
            // make sure it's safe to write to LDS
            block_sync_lds();

            // each thread write its data from VGPR to LDS
            c_thread_copy_vgpr_to_lds.Run(c_thread_desc,
                                          sfc_c_vgpr.GetIndexTupleOfNumber(access_id),
                                          c_thread_buf,
                                          c_block_thread_desc,
                                          c_shuffle_block_buf);

            // make sure it's safe to read from LDS
            block_sync_lds();

            // each block copy its data from LDS to global
            cde_block_copy_lds_and_global.Run(c_ds_desc_refs,
                                              c_ds_buf_refs,
                                              tie(c_grid_desc_mblock_mperblock_nblock_nperblock),
                                              tie(c_grid_buf));

            if constexpr(access_id < num_access - 1)
            {
                constexpr auto cde_lds_and_global_step = sfc_c_global.GetForwardStep(access_id);

                // move on Ds
                static_for<0, NumDTensor_, 1>{}([&](auto i) {
                    cde_block_copy_lds_and_global.MoveSrcSliceWindow(
                        c_ds_desc_refs, i + I1, cde_lds_and_global_step);
                });

                // move on E
                cde_block_copy_lds_and_global.MoveDstSliceWindow(
                    tie(c_grid_desc_mblock_mperblock_nblock_nperblock),
                    I0,
                    cde_lds_and_global_step);
            }
        });
    }

    template <InMemoryDataOperationEnum CGlobalMemoryDataOperation,
              bool TransposeC,
              bool IsInputGemm,
              typename IndexType,
              typename BlockwiseGemmPipe,
              typename CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
              typename DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
              typename CThreadBuffer,
              typename DsGridPointer,
              typename CDEElementwiseOperation>
    __device__ static void RunMoeEpilogue(BlockwiseGemmPipe& blockwise_gemm_pipeline,
                                          const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock&
                                              c_grid_desc_mblock_mperblock_nblock_nperblock,
                                          const DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock&
                                              ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                          CThreadBuffer& c_thread_buf,
                                          index_t block_m_id,
                                          index_t block_n_id,
                                          void* p_shared,
                                          const index_t* p_sorted_token_ids,
                                          EDataType* p_c_grid,
                                          DsGridPointer& p_ds_grid,
                                          const CDEElementwiseOperation& cde_element_op,
                                          index_t problemTopK,
                                          index_t problemN)
    {

        static_assert(MXdlPerWave % CShuffleMXdlPerWavePerShuffle == 0 &&
                          NXdlPerWave % CShuffleNXdlPerWavePerShuffle == 0,
                      "wrong!");

        constexpr index_t MWave      = MPerBlock / (MXdlPerWave * MPerXdl);
        constexpr index_t NWave      = NPerBlock / (NXdlPerWave * NPerXdl);
        constexpr auto c_thread_desc = GetCThreadDescriptor<TransposeC, BlockwiseGemmPipe>();

        constexpr auto c_shuffle_block_desc_mblock_mperblock_nblock_nperblock =
            GetCShuffleBlockDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(get_device_arch());
        auto c_shuffle_block_buf = make_dynamic_buffer<AddressSpaceEnum::Lds>(
            static_cast<CShuffleDataType*>(p_shared),
            c_shuffle_block_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());

        constexpr auto c_block_thread_desc = GetCBlockThreadDescriptor<
            TransposeC,
            BlockwiseGemmPipe,
            decltype(c_shuffle_block_desc_mblock_mperblock_nblock_nperblock)>();

        auto c_thread_copy_vgpr_to_lds =
            GetCThreadCopyVgprToLds<TransposeC>(blockwise_gemm_pipeline,
                                                c_thread_desc,
                                                c_block_thread_desc,
                                                ck::tensor_operation::element_wise::PassThrough{});

        const auto ds_grid_buf = generate_tuple(
            [&](auto i) {
                using DDataType       = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
                const DDataType* ptr_ = p_ds_grid[i];
                // hack logic here to support different kind of strides. todo fix it.
                // ascale t, 1; bscale E, N, 1, move ptr to E
                return make_dynamic_buffer<AddressSpaceEnum::Global>(
                    ptr_, ds_grid_desc_mblock_mperblock_nblock_nperblock[i].GetElementSpaceSize());
            },
            Number<NumDTensor>{});

        // tuple of reference to C/Ds tensor descriptors
        const auto c_ds_desc_refs = concat_tuple_of_reference(
            tie(c_shuffle_block_desc_mblock_mperblock_nblock_nperblock),
            generate_tie([&](auto i) -> const auto& // return type should be reference
                         { return ds_grid_desc_mblock_mperblock_nblock_nperblock[i]; },
                         Number<NumDTensor>{}));

        // tuple of reference to C/Ds tensor descriptors
        const auto c_ds_buf_refs = concat_tuple_of_reference(
            tie(c_shuffle_block_buf),
            generate_tie([&](auto i) -> const auto& // return type should be reference
                         { return ds_grid_buf[i]; },
                         Number<NumDTensor>{}));

        // tuple of starting index of C/Ds blockwise copy
        const auto idx_c_ds_block_begin =
            container_concat(make_tuple(make_multi_index(0, 0, 0, 0)),
                             generate_tuple(
                                 [&](auto) {
                                     return make_multi_index(block_m_id, 0, block_n_id, 0);
                                     // return make_multi_index(block_work_idx[I0], 0,
                                     // block_work_idx[I1], 0);
                                 },
                                 Number<NumDTensor>{}));

        const auto e_grid_desc_mblock_mperblock_nblock_nperblock =
            c_grid_desc_mblock_mperblock_nblock_nperblock;

        using CDEBlockTransferCluster =
            CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock;
        const auto EGlobalMemoryDataOperation = CGlobalMemoryDataOperation;
        constexpr index_t scatter_weight_idx =
            TransposeC ? 1 : 3; // IsInputGemm ? 1 : 1; // hack fix felix
        auto cde_block_copy_lds_and_global = ThreadGroupTensorSliceTransfer_v7r3_scatter<
            ThisThreadBlock,
            decltype(container_concat(make_tuple(CShuffleDataType{}), DsDataType{})),
            Tuple<EDataType>,
            decltype(c_ds_desc_refs),
            decltype(tie(e_grid_desc_mblock_mperblock_nblock_nperblock)),
            CDEElementwiseOperation,
            Sequence<static_cast<index_t>(EGlobalMemoryDataOperation)>, // FIXME: make Sequence
                                                                        // support arbitray type
            Sequence<1,
                     CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                     1,
                     CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>, // BlockSliceLengths,
            CDEBlockTransferCluster,
            Sequence<0, 1, 2, 3>, // typename ThreadClusterArrangeOrder,
            Sequence<0, 1, 2, 3>, // typename SrcDimAccessOrder,
            Sequence<0, 1, 2, 3>, // typename DstDimAccessOrder,
            3,                    // index_t SrcVectorDim,
            3,                    // index_t DstVectorDim,
            CDEShuffleBlockTransferScalarPerVectors,
            CShuffleBlockTransferScalarPerVector_NPerBlock,
            sequence_merge_t<
                Sequence<true>,
                uniform_sequence_gen_t<NumDTensor,
                                       false>>, // ThreadTransferSrcResetCoordinateAfterRunFlags
            Sequence<false>,                    // ThreadTransferDstResetCoordinateAfterRunFlags
            IndexType,
            1,                 // ScatterDim
            true,              // OutputScatter: false, only use scatter weights
            scatter_weight_idx // ScatterWeightIdx: ascale
            >{c_ds_desc_refs,
              idx_c_ds_block_begin,
              tie(e_grid_desc_mblock_mperblock_nblock_nperblock),
              make_tuple(make_multi_index(0, 0, block_n_id, 0)),
              cde_element_op};

        auto c_grid_buf = make_dynamic_buffer<AddressSpaceEnum::Global>(
            p_c_grid, c_grid_desc_mblock_mperblock_nblock_nperblock.GetElementSpaceSize());
        // space filling curve for threadwise C in VGPR
        constexpr auto sfc_c_vgpr =
            GetCThreadWiseSpaceFillingCurve<TransposeC, BlockwiseGemmPipe>();

        constexpr index_t num_access = sfc_c_vgpr.GetNumOfAccess();

        // space filling curve for shuffled blockwise C/D/E
        constexpr auto sfc_cde_block =
            SpaceFillingCurve<Sequence<1, MPerBlock, 1, NPerBlock>,
                              Sequence<0, 2, 1, 3>,
                              Sequence<1,
                                       CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl,
                                       1,
                                       CShuffleNXdlPerWavePerShuffle * NWave * NPerXdl>>{};

        static_assert(num_access == sfc_cde_block.GetNumOfAccess(), "wrong!");
        constexpr auto EMThreads =
            CDEBlockTransferCluster{}.At(I0) * CDEBlockTransferCluster{}.At(I1);
        constexpr auto EMRepeats = CShuffleMXdlPerWavePerShuffle * MWave * MPerXdl / EMThreads;
        constexpr auto ENThreads =
            CDEBlockTransferCluster{}.At(I2) * CDEBlockTransferCluster{}.At(I3);
        static_for<0, num_access, 1>{}([&](auto access_id) {
            // make sure it's safe to write to LDS
            StaticallyIndexedArray<IndexType, EMRepeats> scatter_offsets;

            auto dstidx = sfc_cde_block.GetIndex(access_id);
            const index_t c_token_pos =
                block_m_id * MPerBlock + threadIdx.x / ENThreads * EMRepeats + dstidx(I1);
            static_for<0, EMRepeats, 1>{}([&](auto m0) {
                const index_t fused_token = p_sorted_token_ids[c_token_pos + m0];
                index_t token_offset      = fused_token & 0xffffff;
                if constexpr(IsInputGemm)
                {
                    token_offset = token_offset * problemTopK + (fused_token >> 24);
                }
                scatter_offsets(m0) = token_offset * problemN;
            });

            block_sync_lds();

            // each thread write its data from VGPR to LDS
            c_thread_copy_vgpr_to_lds.Run(c_thread_desc,
                                          sfc_c_vgpr.GetIndexTupleOfNumber(access_id),
                                          c_thread_buf,
                                          c_block_thread_desc,
                                          c_shuffle_block_buf);

            // make sure it's safe to read from LDS
            block_sync_lds();

            // each block copy its data from LDS to global
            cde_block_copy_lds_and_global.Run(c_ds_desc_refs,
                                              c_ds_buf_refs,
                                              tie(e_grid_desc_mblock_mperblock_nblock_nperblock),
                                              tie(c_grid_buf),
                                              scatter_offsets);

            if constexpr(access_id < num_access - 1)
            {
                constexpr auto cde_lds_and_global_step = sfc_cde_block.GetForwardStep(access_id);

                // move on Ds
                static_for<0, NumDTensor, 1>{}([&](auto i) {
                    cde_block_copy_lds_and_global.MoveSrcSliceWindow(
                        c_ds_desc_refs, i + I1, cde_lds_and_global_step);
                });

                // move on E
                cde_block_copy_lds_and_global.MoveDstSliceWindow(
                    tie(e_grid_desc_mblock_mperblock_nblock_nperblock),
                    I0,
                    cde_lds_and_global_step);
            }
        });
    }
};

} // namespace ck
