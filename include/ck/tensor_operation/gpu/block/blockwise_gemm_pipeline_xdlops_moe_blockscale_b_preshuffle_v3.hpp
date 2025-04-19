// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_base.hpp"

namespace ck {

// Compute optimized pipeline
// GlobalPrefetchStages: 2
// LocalPreFillStages: 1
// LocalPreFetchStages: 1
// LocalSharedMemoryBuffer: 1

template <BlockGemmPipelineScheduler BlkGemmPipelineVer,
          index_t BlockSize,
          typename ADataType,
          typename BDataType,
          typename ComputeDataType,
          typename AccDataType,
          typename ATileDesc,
          typename BTileDesc,
          typename AMmaTileDesc,
          typename BMmaTileDesc,
          index_t ABlockTransferSrcScalarPerVector,
          index_t BBlockTransferSrcScalarPerVector,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MScaleBlock,
          index_t NScaleBlock,
          index_t KScaleBlock,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPacks>
struct BlockwiseGemmXdlops_pipeline_moe_blockscale_bpreshuffle_v3
{
};

template <index_t BlockSize,
          typename ADataType,
          typename BDataType,
          typename ComputeDataType,
          typename AccDataType,
          typename ATileDesc,
          typename BTileDesc,
          typename AMmaTileDesc,
          typename BMmaTileDesc,
          index_t ABlockTransferSrcScalarPerVector,
          index_t BBlockTransferSrcScalarPerVector,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MScaleBlock,
          index_t NScaleBlock,
          index_t KScaleBlock,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MRepeat,
          index_t NRepeat,
          index_t KPack
          // ,bool TransposeC //disable transposec right now...
          >
struct BlockwiseGemmXdlops_pipeline_moe_blockscale_bpreshuffle_v3<
    BlockGemmPipelineScheduler::Intrawave,
    BlockSize,
    ADataType,
    BDataType,
    ComputeDataType,
    AccDataType,
    ATileDesc,
    BTileDesc,
    AMmaTileDesc,
    BMmaTileDesc,
    ABlockTransferSrcScalarPerVector,
    BBlockTransferSrcScalarPerVector,
    MPerBlock,
    NPerBlock,
    KPerBlock,
    MScaleBlock,
    NScaleBlock,
    KScaleBlock,
    MPerXDL,
    NPerXDL,
    MRepeat,
    NRepeat,
    KPack> : BlockwiseGemmXdlops_pipeline_base<BlockSize,
                                               ADataType,
                                               BDataType,
                                               ComputeDataType,
                                               AccDataType,
                                               ATileDesc,
                                               BTileDesc,
                                               AMmaTileDesc,
                                               BMmaTileDesc,
                                               ABlockTransferSrcScalarPerVector,
                                               BBlockTransferSrcScalarPerVector,
                                               MPerBlock,
                                               NPerBlock,
                                               KPerBlock,
                                               MPerXDL,
                                               NPerXDL,
                                               MRepeat,
                                               NRepeat,
                                               KPack,
                                               true>

{
    using Base = BlockwiseGemmXdlops_pipeline_base<BlockSize,
                                                   ADataType,
                                                   BDataType,
                                                   ComputeDataType,
                                                   AccDataType,
                                                   ATileDesc,
                                                   BTileDesc,
                                                   AMmaTileDesc,
                                                   BMmaTileDesc,
                                                   ABlockTransferSrcScalarPerVector,
                                                   BBlockTransferSrcScalarPerVector,
                                                   MPerBlock,
                                                   NPerBlock,
                                                   KPerBlock,
                                                   MPerXDL,
                                                   NPerXDL,
                                                   MRepeat,
                                                   NRepeat,
                                                   KPack,
                                                   true>;
    using Base::A_K1;
    using Base::B_K1;
    using Base::I0;
    using Base::I1;
    using Base::I2;
    using Base::KRepeat;
    using Base::xdlops_gemm;
    using typename Base::HotLoopInstList;

    using Base::a_block_desc_m0_m1_m2_k;
    using Base::CalculateCThreadOriginDataIndex;
    using Base::CalculateCThreadOriginDataIndex8D;
    using Base::GetCBlockDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::GetCBlockDescriptor_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::GetCBlockDescriptor_M0_N0_M1_N1_M2_N2_N3_N4;
    using Base::GetCThreadBuffer;
    using Base::GetCThreadDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::GetCThreadDescriptor_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::GetCThreadDescriptor_M0_N0_M1_N1_M2_N2_N3_N4;
    using Base::MakeCGridDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2;
    using Base::MWaves;

    static constexpr index_t PrefetchStages        = 2;
    static constexpr index_t PrefillStages         = 1;
    static constexpr index_t GlobalBufferNum       = 1;
    static constexpr index_t HotloopLocalBufSwitch = MRepeat % 2 == 0 ? 0 : 1;

    template <typename TileDesc_M0_M1_M2_K>
    __host__ __device__ static constexpr auto MakeAGemmMmaTileDescriptor(const TileDesc_M0_M1_M2_K&)
    {
        constexpr index_t M0 = TileDesc_M0_M1_M2_K{}.GetLength(Number<0>{});
        constexpr index_t M1 = TileDesc_M0_M1_M2_K{}.GetLength(Number<1>{});
        constexpr index_t M2 = TileDesc_M0_M1_M2_K{}.GetLength(Number<2>{});
        constexpr index_t K2 = KPack;
        constexpr index_t K1 = 64 / NPerXDL;
        constexpr index_t K0 = KRepeat;

        return transform_tensor_descriptor(
            TileDesc_M0_M1_M2_K{},
            make_tuple(
                make_pass_through_transform(Number<M0>{}),
                make_pass_through_transform(Number<M1>{}),
                make_pass_through_transform(Number<M2>{}),
                make_unmerge_transform(make_tuple(Number<K0>{}, Number<K1>{}, Number<K2>{}))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3, 4, 5>{}));
    }

    static constexpr auto a_block_desc_m0_m1_m2_k0_k1_k2 =
        MakeAGemmMmaTileDescriptor(a_block_desc_m0_m1_m2_k);

    __host__ __device__ static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > PrefetchStages;
    }

    __host__ __device__ static constexpr TailNumber BlockLoopTailNum(index_t num_loop)
    {
        return num_loop % 2 == 0 ? TailNumber::Even : TailNumber::Odd;
    }

    template <typename Stage>
    __device__ static constexpr auto HotLoopScheduler(Stage stage)
    {
        constexpr auto num_ds_read_inst_a     = HotLoopInstList::A_LDS_Read_Inst_Num;
        constexpr auto num_ds_write_inst_a    = HotLoopInstList::A_LDS_Write_Inst_Num;
        constexpr auto num_buffer_load_inst_a = HotLoopInstList::A_Buffer_Load_Inst_Num;
        constexpr auto num_buffer_load_inst_b = MWaves * HotLoopInstList::B_Buffer_Load_Inst_Num;

        constexpr auto num_mfma = HotLoopInstList::C_MFMA_Inst_Num;

        constexpr auto staged_num_ds_read_inst_a = num_ds_read_inst_a / MRepeat;
        constexpr auto staged_num_mfma           = num_mfma / MRepeat;

        constexpr auto staged_num_mfma_per_ds_read_a = staged_num_mfma / staged_num_ds_read_inst_a;

        constexpr auto num_pk_fma_per_kscaleblock = MPerXDL == 16 ? 2 : 8;
        constexpr auto num_mfma_per_kscaleblock   = MPerXDL == 16 ? KPerBlock / 32 : KPerBlock / 16;

        if constexpr(stage.value == 0)
        {
            // B VMEM access.
            constexpr auto staged_num_buffer_load_b_per_ds_read_a =
                num_buffer_load_inst_b / staged_num_ds_read_inst_a;
            constexpr auto staged_num_mfma_per_buffer_load_b =
                staged_num_mfma / num_buffer_load_inst_b;
            // B global
            static_for<0, staged_num_ds_read_inst_a, 1>{}([&](auto i_inst) {
                static_for<0, staged_num_buffer_load_b_per_ds_read_a - 1, 1>{}([&](auto ibuf_inst) {
                    static_for<0, staged_num_mfma_per_buffer_load_b, 1>{}([&](auto imfma) {
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA

                        /* Judging issue v_pk_fma */
                        if constexpr((i_inst * staged_num_mfma_per_buffer_load_b *
                                          staged_num_buffer_load_b_per_ds_read_a +
                                      ibuf_inst * staged_num_mfma_per_buffer_load_b + imfma + 1) %
                                         num_mfma_per_kscaleblock ==
                                     0)
                        {
                            __builtin_amdgcn_sched_group_barrier(
                                0x800, num_pk_fma_per_kscaleblock, 0); // PK_FMA
                        }
                    });
                    __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                });

                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA

                /* Judging issue v_pk_fma */
                if constexpr((i_inst * staged_num_mfma_per_buffer_load_b *
                                  staged_num_buffer_load_b_per_ds_read_a +
                              (staged_num_buffer_load_b_per_ds_read_a - 1) *
                                  staged_num_mfma_per_buffer_load_b +
                              1) %
                                 num_mfma_per_kscaleblock ==
                             0)
                {
                    __builtin_amdgcn_sched_group_barrier(
                        0x800, num_pk_fma_per_kscaleblock, 0); // PK_FMA
                }
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read

                static_for<0, staged_num_mfma_per_buffer_load_b - 1, 1>{}([&](auto imfma) {
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA

                    /* Judging issue v_pk_fma */
                    if constexpr((i_inst * staged_num_mfma_per_buffer_load_b *
                                      staged_num_buffer_load_b_per_ds_read_a +
                                  (staged_num_buffer_load_b_per_ds_read_a - 1) *
                                      staged_num_mfma_per_buffer_load_b +
                                  imfma + 2) %
                                     num_mfma_per_kscaleblock ==
                                 0)
                    {
                        __builtin_amdgcn_sched_group_barrier(
                            0x800, num_pk_fma_per_kscaleblock, 0); // PK_FMA
                    }
                });
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
            });

            __builtin_amdgcn_sched_barrier(0);
        }
        else if constexpr(stage.value == 1)
        {
            // A LDS write access.
            constexpr auto staged_num_mfma_per_ds_write_a =
                math::integer_divide_ceil(staged_num_mfma, num_ds_write_inst_a);

            constexpr auto stage_more_mfma =
                staged_num_mfma - (staged_num_mfma_per_ds_write_a - 1) * num_ds_write_inst_a;

            // A local write
            static_for<0, num_ds_write_inst_a, 1>{}([&](auto i_inst) {
                if constexpr(i_inst.value < stage_more_mfma)
                {
                    if(i_inst.value < staged_num_ds_read_inst_a)
                    {
                        static_for<0, staged_num_mfma_per_ds_write_a - 1, 1>{}([&](auto i_mfma) {
                            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA

                            /* Judging issue v_pk_fma */
                            if constexpr((i_inst * staged_num_mfma_per_ds_write_a + i_mfma + 1) %
                                             num_mfma_per_kscaleblock ==
                                         0)
                            {
                                __builtin_amdgcn_sched_group_barrier(
                                    0x800, num_pk_fma_per_kscaleblock, 0); // PK_FMA
                            }
                        });
                        __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS Write
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA

                        /* Judging issue v_pk_fma */
                        if constexpr(((i_inst + 1) * staged_num_mfma_per_ds_write_a) %
                                         num_mfma_per_kscaleblock ==
                                     0)
                        {
                            __builtin_amdgcn_sched_group_barrier(
                                0x800, num_pk_fma_per_kscaleblock, 0); // PK_FMA
                        }

                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    }
                    else
                    {
                        static_for<0, staged_num_mfma_per_ds_write_a, 1>{}([&](auto i_mfma) {
                            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA

                            /* Judging issue v_pk_fma */
                            if constexpr((i_inst * staged_num_mfma_per_ds_write_a + i_mfma + 1) %
                                             num_mfma_per_kscaleblock ==
                                         0)
                            {
                                __builtin_amdgcn_sched_group_barrier(
                                    0x800, num_pk_fma_per_kscaleblock, 0); // PK_FMA
                            }
                        });

                        __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS Write
                    }
                }
                else
                {
                    if(i_inst.value < staged_num_ds_read_inst_a)
                    {
                        static_for<0, staged_num_mfma_per_ds_write_a - 2, 1>{}([&](auto i_mfma) {
                            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA

                            /* Judging issue v_pk_fma */
                            if constexpr((stage_more_mfma * staged_num_mfma_per_ds_write_a +
                                          (i_inst - stage_more_mfma) *
                                              (staged_num_mfma_per_ds_write_a - 1) +
                                          i_mfma + 1) %
                                             num_mfma_per_kscaleblock ==
                                         0)
                            {
                                __builtin_amdgcn_sched_group_barrier(
                                    0x800, num_pk_fma_per_kscaleblock, 0); // PK_FMA
                            }
                        });

                        __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS Write
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA

                        /* Judging issue v_pk_fma */
                        if constexpr((stage_more_mfma * staged_num_mfma_per_ds_write_a +
                                      (i_inst - stage_more_mfma + 1) *
                                          (staged_num_mfma_per_ds_write_a - 1)) %
                                         num_mfma_per_kscaleblock ==
                                     0)
                        {
                            __builtin_amdgcn_sched_group_barrier(
                                0x800, num_pk_fma_per_kscaleblock, 0); // PK_FMA
                        }
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    }
                    else
                    {
                        static_for<0, staged_num_mfma_per_ds_write_a - 1, 1>{}([&](auto i_mfma) {
                            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA

                            /* Judging issue v_pk_fma */
                            if constexpr((stage_more_mfma * staged_num_mfma_per_ds_write_a +
                                          (i_inst - stage_more_mfma) *
                                              (staged_num_mfma_per_ds_write_a - 1) +
                                          i_mfma + 1) %
                                             num_mfma_per_kscaleblock ==
                                         0)
                            {
                                __builtin_amdgcn_sched_group_barrier(
                                    0x800, num_pk_fma_per_kscaleblock, 0); // PK_FMA
                            }
                        });

                        __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS Write
                    }
                }
            });

            __builtin_amdgcn_sched_barrier(0);
        }
        else if constexpr(stage.value == 2)
        {
            // A VMEM access.
            constexpr auto staged_num_mfma_per_buffer_load_a =
                math::integer_divide_ceil(staged_num_mfma, num_buffer_load_inst_a);

            constexpr auto stage_more_mfma =
                staged_num_mfma - (staged_num_mfma_per_buffer_load_a - 1) * num_buffer_load_inst_a;

            // A global
            static_for<0, num_buffer_load_inst_a, 1>{}([&](auto i_inst) {
                if constexpr(i_inst.value < stage_more_mfma)
                {
                    if(i_inst.value < staged_num_ds_read_inst_a)
                    {
                        static_for<0, staged_num_mfma_per_buffer_load_a - 1, 1>{}([&](auto i_mfma) {
                            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA

                            /* Judging issue v_pk_fma */
                            if constexpr((i_inst * staged_num_mfma_per_buffer_load_a + i_mfma + 1) %
                                             num_mfma_per_kscaleblock ==
                                         0)
                            {
                                __builtin_amdgcn_sched_group_barrier(
                                    0x800, num_pk_fma_per_kscaleblock, 0); // PK_FMA
                            }
                        });
                        __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA

                        /* Judging issue v_pk_fma */
                        if constexpr(((i_inst + 1) * staged_num_mfma_per_buffer_load_a) %
                                         num_mfma_per_kscaleblock ==
                                     0)
                        {
                            __builtin_amdgcn_sched_group_barrier(
                                0x800, num_pk_fma_per_kscaleblock, 0); // PK_FMA
                        }
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    }
                    else
                    {
                        static_for<0, staged_num_mfma_per_buffer_load_a, 1>{}([&](auto i_mfma) {
                            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA

                            /* Judging issue v_pk_fma */
                            if constexpr((i_inst * staged_num_mfma_per_buffer_load_a + i_mfma + 1) %
                                             num_mfma_per_kscaleblock ==
                                         0)
                            {
                                __builtin_amdgcn_sched_group_barrier(
                                    0x800, num_pk_fma_per_kscaleblock, 0); // PK_FMA
                            }
                        });
                        __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                    }
                }
                else
                {
                    if(i_inst.value < staged_num_ds_read_inst_a)
                    {
                        static_for<0, staged_num_mfma_per_buffer_load_a - 2, 1>{}([&](auto i_mfma) {
                            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA

                            /* Judging issue v_pk_fma */
                            if constexpr((stage_more_mfma * staged_num_mfma_per_buffer_load_a +
                                          (i_inst - stage_more_mfma) *
                                              (staged_num_mfma_per_buffer_load_a - 1) +
                                          i_mfma + 1) %
                                             num_mfma_per_kscaleblock ==
                                         0)
                            {
                                __builtin_amdgcn_sched_group_barrier(
                                    0x800, num_pk_fma_per_kscaleblock, 0); // PK_FMA
                            }
                        });
                        __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA

                        /* Judging issue v_pk_fma */
                        if constexpr((stage_more_mfma * staged_num_mfma_per_buffer_load_a +
                                      (i_inst - stage_more_mfma + 1) *
                                          (staged_num_mfma_per_buffer_load_a - 1)) %
                                         num_mfma_per_kscaleblock ==
                                     0)
                        {
                            __builtin_amdgcn_sched_group_barrier(
                                0x800, num_pk_fma_per_kscaleblock, 0); // PK_FMA
                        }
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    }
                    else
                    {
                        static_for<0, staged_num_mfma_per_buffer_load_a - 1, 1>{}([&](auto i_mfma) {
                            __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA

                            /* Judging issue v_pk_fma */
                            if constexpr((stage_more_mfma * staged_num_mfma_per_buffer_load_a +
                                          (i_inst - stage_more_mfma) *
                                              (staged_num_mfma_per_buffer_load_a - 1) +
                                          i_mfma + 1) %
                                             num_mfma_per_kscaleblock ==
                                         0)
                            {
                                __builtin_amdgcn_sched_group_barrier(
                                    0x800, num_pk_fma_per_kscaleblock, 0); // PK_FMA
                            }
                        });

                        __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                    }
                }
            });

            __builtin_amdgcn_sched_barrier(0);
        }
        else
        {
            // A local Read
            static_for<0, staged_num_ds_read_inst_a, 1>{}([&](auto i_inst) {
                static_for<0, staged_num_mfma_per_ds_read_a, 1>{}([&](auto i_mfma) {
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA

                    /* Judging issue v_pk_fma */
                    if constexpr((i_inst * staged_num_mfma_per_ds_read_a + i_mfma + 1) %
                                     num_mfma_per_kscaleblock ==
                                 0)
                    {
                        __builtin_amdgcn_sched_group_barrier(
                            0x800, num_pk_fma_per_kscaleblock, 0); // PK_FMA
                    }
                });
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
            });

            __builtin_amdgcn_sched_barrier(0);
        }
    }

    template <typename Stage>
    __device__ static constexpr auto EpilogueScheduler_1(Stage stage)
    {
        constexpr auto num_ds_read_inst_a     = HotLoopInstList::A_LDS_Read_Inst_Num;
        constexpr auto num_ds_write_inst_a    = HotLoopInstList::A_LDS_Write_Inst_Num;
        constexpr auto num_buffer_load_inst_b = MWaves * HotLoopInstList::B_Buffer_Load_Inst_Num;

        constexpr auto num_mfma = HotLoopInstList::C_MFMA_Inst_Num;

        constexpr auto staged_num_ds_read_inst_a = num_ds_read_inst_a / MRepeat;
        constexpr auto staged_num_mfma           = num_mfma / MRepeat;

        constexpr auto staged_num_mfma_per_ds_read_a = staged_num_mfma / staged_num_ds_read_inst_a;

        if constexpr(stage.value == 0)
        {
            constexpr auto staged_num_buffer_load_b_per_ds_read_a =
                num_buffer_load_inst_b / staged_num_ds_read_inst_a;
            constexpr auto staged_num_mfma_per_buffer_load_b =
                staged_num_mfma / num_buffer_load_inst_b;
            // B global
            static_for<0, staged_num_ds_read_inst_a, 1>{}([&](auto i_inst) {
                ignore = i_inst;

                static_for<0, staged_num_buffer_load_b_per_ds_read_a, 1>{}([&](auto ibuf_inst) {
                    ignore = ibuf_inst;
                    __builtin_amdgcn_sched_group_barrier(
                        0x008, staged_num_mfma_per_buffer_load_b, 0);  // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                });

                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                __builtin_amdgcn_sched_group_barrier(
                    0x008, staged_num_mfma_per_buffer_load_b - 1, 0); // MFMA
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);    // VMEM read
            });

            __builtin_amdgcn_sched_barrier(0);
        }
        else if constexpr(stage.value == 1)
        {
            constexpr auto staged_num_mfma_per_ds_write_a =
                math::integer_divide_ceil(staged_num_mfma, num_ds_write_inst_a);

            constexpr auto stage_more_mfma =
                staged_num_mfma - (staged_num_mfma_per_ds_write_a - 1) * num_ds_write_inst_a;

            // A local write
            static_for<0, num_ds_write_inst_a, 1>{}([&](auto i_inst) {
                if constexpr(i_inst.value < stage_more_mfma)
                {
                    if(i_inst.value < staged_num_ds_read_inst_a)
                    {
                        __builtin_amdgcn_sched_group_barrier(
                            0x008, staged_num_mfma_per_ds_write_a - 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS Write
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    }
                    else
                    {
                        __builtin_amdgcn_sched_group_barrier(
                            0x008, staged_num_mfma_per_ds_write_a, 0);     // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS Write
                    }
                }
                else
                {
                    if(i_inst.value < staged_num_ds_read_inst_a)
                    {
                        __builtin_amdgcn_sched_group_barrier(
                            0x008, staged_num_mfma_per_ds_write_a - 2, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS Write
                        __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
                    }
                    else
                    {
                        __builtin_amdgcn_sched_group_barrier(
                            0x008, staged_num_mfma_per_ds_write_a - 1, 0); // MFMA
                        __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS Write
                    }
                }
            });
            __builtin_amdgcn_sched_barrier(0);
        }
        else
        {
            // A local Read
            static_for<0, staged_num_ds_read_inst_a, 1>{}([&](auto i_inst) {
                ignore = i_inst;
                __builtin_amdgcn_sched_group_barrier(
                    0x008, staged_num_mfma_per_ds_read_a, 0);      // MFMA
                __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
            });

            __builtin_amdgcn_sched_barrier(0);
        }
    }

    __device__ static constexpr auto EpilogueScheduler_2()
    {
        constexpr auto num_ds_read_inst_a = HotLoopInstList::A_LDS_Read_Inst_Num;

        constexpr auto num_mfma = HotLoopInstList::C_MFMA_Inst_Num;

        constexpr auto staged_num_ds_read_inst_a = num_ds_read_inst_a / MRepeat;
        constexpr auto staged_num_mfma           = num_mfma / MRepeat;

        constexpr auto staged_num_mfma_per_ds_read_a = staged_num_mfma / staged_num_ds_read_inst_a;

        // A local Read
        static_for<0, staged_num_ds_read_inst_a, 1>{}([&](auto i_inst) {
            ignore = i_inst;
            __builtin_amdgcn_sched_group_barrier(0x008, staged_num_mfma_per_ds_read_a, 0); // MFMA
            __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS read
        });

        __builtin_amdgcn_sched_barrier(0);
    }

    template <bool HasMainLoop,
              int NumKBlockPerScale,
              TailNumber TailNum,
              typename AGridDesc,
              typename ABlockDesc,
              typename ABlockTransfer,
              typename AGridBuffer,
              typename ABlockBuffer,
              typename ABlockTransferStep,
              typename BGridDesc,
              typename BBlockDesc,
              typename BBlockTransfer,
              typename BGridBuffer,
              typename BBlockBuffer,
              typename BBlockTransferStep,
              typename CScaleThreadDesc,
              typename CThreadBuffer,
              typename AScaleGridBuffer,
              typename AScaleGridDesc,
              typename AScaleThreadDesc,
              typename AScaleThreadTransfer,
              typename AScaleThreadTransferStep,
              typename BScaleGridBuffer,
              typename BScaleGridDesc,
              typename BScaleThreadDesc,
              typename BScaleThreadTransfer,
              typename BScaleThreadTransferStep>
    __device__ void Run(
        // ABlockCopy
        const AGridDesc& a_grid_desc,
        const ABlockDesc& a_block_desc,
        ABlockTransfer& a_blockwise_copy,
        const AGridBuffer& a_grid_buf,
        ABlockBuffer& a_block_buf,
        const ABlockTransferStep& a_block_copy_step,
        // BBlockCopy
        const BGridDesc& b_grid_desc,
        const BBlockDesc& b_block_desc,
        BBlockTransfer& b_blockwise_copy,
        const BGridBuffer& b_grid_buf,
        BBlockBuffer& b_block_buf,
        const BBlockTransferStep& b_block_copy_step,
        // CThread
        const CScaleThreadDesc& c_scale_thread_desc,
        CThreadBuffer& c_thread_buf,
        // AScaleThreadCopy
        const AScaleGridDesc& a_scale_grid_desc,
        const AScaleThreadDesc& a_scale_thread_desc,
        AScaleThreadTransfer& a_scale_thread_copy,
        const AScaleGridBuffer& a_scale_grid_buf,
        const AScaleThreadTransferStep& a_scale_thread_copy_step,
        // BScaleThreadCopy
        const BScaleGridDesc& b_scale_grid_desc,
        const BScaleThreadDesc& b_scale_thread_desc,
        BScaleThreadTransfer& b_scale_thread_copy,
        const BScaleGridBuffer& b_scale_grid_buf,
        const BScaleThreadTransferStep& b_scale_thread_copy_step,
        // num_loop
        index_t num_loop) const
    {
        ignore = b_block_desc;
        ignore = b_block_buf;
        __builtin_amdgcn_sched_barrier(0);
        static_assert(CScaleThreadDesc{}.GetLength(Number<0>{}) == 1,
                      "Pipeline v3 only support scaleblocksliceK=1");
        static_assert(CScaleThreadDesc{}.GetLength(Number<2>{}) == 1,
                      "Pipeline v3 only support scaleblocksliceN=1");
        // assume kperblock = scaleblockk
        auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeDataType>(
            a_thread_desc_.GetElementSpaceSize());
        auto b_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, ComputeDataType>(
            b_thread_desc_.GetElementSpaceSize());
        StaticallyIndexedArray<decltype(b_thread_buf), Number<2>{}> b_thread_bufs;
        constexpr auto b_block_origin_idx = make_tuple(I0, I0, I0, I0);
        auto a_scale_thread_buf           = make_static_buffer<AddressSpaceEnum::Vgpr, AccDataType>(
            a_scale_thread_desc.GetElementSpaceSize());
        auto b_scale_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, AccDataType>(
            b_scale_thread_desc.GetElementSpaceSize());
        auto c_scale_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, AccDataType>(
            c_scale_thread_desc.GetElementSpaceSize());

        // Global prefetch A1 B1, AScale1 BScale1
        b_blockwise_copy.Run(b_grid_desc,
                             b_grid_buf,
                             b_block_desc_n0_n1_k0_k1,
                             b_block_origin_idx,
                             b_thread_bufs(I0));
        b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);

        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
        __builtin_amdgcn_sched_barrier(0);

        static_for<0, MRepeat, 1>{}([&](auto m0) {
            a_scale_thread_copy.Run(a_scale_grid_desc,
                                    a_scale_grid_buf,
                                    a_scale_thread_desc,
                                    make_tuple(m0, I0),
                                    a_scale_thread_buf);
            a_scale_thread_copy.MoveSrcSliceWindow(a_scale_grid_desc,
                                                   a_scale_thread_copy_step.At(Number<0>{}));
        });

        if constexpr(NumKBlockPerScale == 1)
        {
            a_scale_thread_copy.MoveSrcSliceWindow(a_scale_grid_desc,
                                                   a_scale_thread_copy_step.At(Number<2>{}));
        }
        else
        {
            a_scale_thread_copy.MoveSrcSliceWindow(a_scale_grid_desc,
                                                   a_scale_thread_copy_step.At(Number<1>{}));
        }

        b_scale_thread_copy.Run(b_scale_grid_desc,
                                b_scale_grid_buf,
                                b_scale_thread_desc,
                                make_tuple(I0, I0),
                                b_scale_thread_buf);

        b_scale_thread_copy.MoveSrcSliceWindow(b_scale_grid_desc, b_scale_thread_copy_step);

        static_for<0, MRepeat, 1>{}([&](auto m0) {
            c_scale_thread_buf(m0) = a_scale_thread_buf[m0] * b_scale_thread_buf[I0];
        });

        // Local prefill A1
        a_blockwise_copy.RunWrite(a_block_desc, a_block_buf.At(I0));

        // Global prefetch A2, AScale2 BScale2
        a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
        a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);

        static_for<0, MRepeat, 1>{}([&](auto m0) {
            a_scale_thread_copy.Run(a_scale_grid_desc,
                                    a_scale_grid_buf,
                                    a_scale_thread_desc,
                                    make_tuple(m0, I0),
                                    a_scale_thread_buf);
            a_scale_thread_copy.MoveSrcSliceWindow(a_scale_grid_desc,
                                                   a_scale_thread_copy_step.At(Number<0>{}));
        });

        if constexpr(NumKBlockPerScale == 1)
        {
            a_scale_thread_copy.MoveSrcSliceWindow(a_scale_grid_desc,
                                                   a_scale_thread_copy_step.At(Number<2>{}));
        }
        else
        {
            a_scale_thread_copy.MoveSrcSliceWindow(a_scale_grid_desc,
                                                   a_scale_thread_copy_step.At(Number<1>{}));
        }

        b_scale_thread_copy.Run(b_scale_grid_desc,
                                b_scale_grid_buf,
                                b_scale_thread_desc,
                                make_tuple(I0, I0),
                                b_scale_thread_buf);

        b_scale_thread_copy.MoveSrcSliceWindow(b_scale_grid_desc, b_scale_thread_copy_step);

        // Initialize C
        c_thread_buf.Clear();

        StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                                  AccDataType,
                                  1,
                                  xdlops_gemm.GetRegSizePerXdlops(),
                                  true>
            c_thread_buf_per_scale;

        // Local prefetch A1
        block_sync_lds();
        static_for<0, KRepeat, 1>{}([&](auto k0) {
            a_thread_copy_.Run(a_block_desc_m0_m1_m2_k0_k1_k2,
                               make_tuple(I0, I0, I0, k0, I0, I0),
                               a_block_buf.At(I0),
                               a_thread_desc_,
                               make_tuple(I0, I0, I0, k0, I0, I0),
                               a_thread_buf);
        });

        __builtin_amdgcn_sched_barrier(0);

        // main body
        if constexpr(HasMainLoop)
        {
            index_t i = 0;
            do
            {
                auto LoopFunc = [&](auto mfma_reg_buf, auto local_read_buf) {
                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        if constexpr(m0.value == 0)
                        {
                            b_blockwise_copy.Run(b_grid_desc,
                                                 b_grid_buf,
                                                 b_block_desc_n0_n1_k0_k1,
                                                 b_block_origin_idx,
                                                 b_thread_bufs(local_read_buf));
                            b_blockwise_copy.MoveSrcSliceWindow(b_grid_desc, b_block_copy_step);
                        }
                        else if constexpr(m0.value == 1)
                        {
                            a_blockwise_copy.RunWrite(a_block_desc, a_block_buf.At(local_read_buf));
                        }
                        else if constexpr(m0.value == 2)
                        {
                            a_blockwise_copy.RunRead(a_grid_desc, a_grid_buf);
                            a_blockwise_copy.MoveSrcSliceWindow(a_grid_desc, a_block_copy_step);
                        }

                        vector_type<AccDataType, 2> c_scale_thread_vec;
                        c_scale_thread_vec.template AsType<AccDataType>()(Number<0>{}) =
                            c_scale_thread_buf[m0];
                        c_scale_thread_vec.template AsType<AccDataType>()(Number<1>{}) =
                            c_scale_thread_buf[m0];

                        static_for<0, NRepeat, 1>{}([&](auto n0) {
                            static_for<0, xdlops_gemm.GetRegSizePerXdlops(), 1>{}([&](auto t) {
                                c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{})
                                    .template AsType<AccDataType>()(Number<t>{}) = 0;
                            });
                            static_for<0, KRepeat, 1>{}([&](auto k0) {
                                vector_type<ComputeDataType, KPack> a_thread_vec;
                                vector_type<ComputeDataType, KPack> b_thread_vec;

                                static_for<0, KPack, 1>{}([&](auto ik) {
                                    a_thread_vec.template AsType<ComputeDataType>()(ik) =
                                        a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                            make_tuple((m0 + HotloopLocalBufSwitch * mfma_reg_buf) %
                                                           2,
                                                       I0,
                                                       I0,
                                                       k0,
                                                       I0,
                                                       ik))>{}];
                                    b_thread_vec.template AsType<ComputeDataType>()(ik) =
                                        b_thread_bufs[mfma_reg_buf]
                                                     [Number<b_thread_desc_.CalculateOffset(
                                                         make_tuple(n0, I0, k0, ik))>{}];
                                });

                                using mfma_input_type =
                                    typename vector_type<ComputeDataType,
                                                         xdlops_gemm.K1PerXdlops>::type;

                                xdlops_gemm.template Run<>(
                                    a_thread_vec.template AsType<mfma_input_type>(),
                                    b_thread_vec.template AsType<mfma_input_type>(),
                                    c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{}));
                            });

                            constexpr index_t c_offset =
                                c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                            static_for<0, xdlops_gemm.GetRegSizePerXdlops() / 2, 1>{}([&](auto t) {
                                using pk_fma_type = typename vector_type<AccDataType, 2>::type;

                                c_thread_buf.GetVectorTypeReference(Number<c_offset>{})
                                    .template AsType<pk_fma_type>()(t) = __builtin_elementwise_fma(
                                    c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{})
                                        .template AsType<pk_fma_type>()[t],
                                    c_scale_thread_vec.template AsType<pk_fma_type>()[Number<0>{}],
                                    c_thread_buf.GetVectorTypeReference(Number<c_offset>{})
                                        .template AsType<pk_fma_type>()[t]);
                            });
                        });

                        if constexpr(m0.value == MRepeat - 1)
                        {
                            block_sync_lds();

                            static_for<0, KRepeat, 1>{}([&](auto k0) {
                                a_thread_copy_.Run(
                                    a_block_desc_m0_m1_m2_k0_k1_k2,
                                    make_tuple(Number<(m0 + 1) % MRepeat>{}, I0, I0, k0, I0, I0),
                                    a_block_buf.At(local_read_buf),
                                    a_thread_desc_,
                                    make_tuple(
                                        Number<(m0 + 1 + HotloopLocalBufSwitch * mfma_reg_buf) %
                                               2>{},
                                        I0,
                                        I0,
                                        k0,
                                        I0,
                                        I0),
                                    a_thread_buf);
                            });
                        }
                        else
                        {
                            static_for<0, KRepeat, 1>{}([&](auto k0) {
                                a_thread_copy_.Run(
                                    a_block_desc_m0_m1_m2_k0_k1_k2,
                                    make_tuple(Number<(m0 + 1) % MRepeat>{}, I0, I0, k0, I0, I0),
                                    a_block_buf.At(mfma_reg_buf),
                                    a_thread_desc_,
                                    make_tuple(
                                        Number<(m0 + 1 + HotloopLocalBufSwitch * mfma_reg_buf) %
                                               2>{},
                                        I0,
                                        I0,
                                        k0,
                                        I0,
                                        I0),
                                    a_thread_buf);
                            });
                        }

                        HotLoopScheduler(m0);
                    });

                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        c_scale_thread_buf(m0) = a_scale_thread_buf[m0] * b_scale_thread_buf[I0];
                    });

                    static_for<0, MRepeat, 1>{}([&](auto m0) {
                        a_scale_thread_copy.Run(a_scale_grid_desc,
                                                a_scale_grid_buf,
                                                a_scale_thread_desc,
                                                make_tuple(m0, I0),
                                                a_scale_thread_buf);
                        a_scale_thread_copy.MoveSrcSliceWindow(
                            a_scale_grid_desc, a_scale_thread_copy_step.At(Number<0>{}));
                    });

                    if constexpr(NumKBlockPerScale == 1)
                    {
                        a_scale_thread_copy.MoveSrcSliceWindow(
                            a_scale_grid_desc, a_scale_thread_copy_step.At(Number<2>{}));
                    }
                    else
                    {
                        a_scale_thread_copy.MoveSrcSliceWindow(
                            a_scale_grid_desc, a_scale_thread_copy_step.At(Number<1>{}));
                    }

                    b_scale_thread_copy.Run(b_scale_grid_desc,
                                            b_scale_grid_buf,
                                            b_scale_thread_desc,
                                            make_tuple(I0, I0),
                                            b_scale_thread_buf);

                    b_scale_thread_copy.MoveSrcSliceWindow(b_scale_grid_desc,
                                                           b_scale_thread_copy_step);

                    __builtin_amdgcn_sched_group_barrier(0x020, MRepeat + 1, 0); // VMEM read
                    __builtin_amdgcn_sched_barrier(0);
                };

                LoopFunc(I0, I1);
                LoopFunc(I1, I0);

                i += 2;
            } while(i < (num_loop - 2));
        }

        // tail
        if constexpr(TailNum == TailNumber::Even)
        {
            static_for<0, MRepeat, 1>{}([&](auto m0) {
                if constexpr(m0.value == 0)
                {
                    b_blockwise_copy.Run(b_grid_desc,
                                         b_grid_buf,
                                         b_block_desc_n0_n1_k0_k1,
                                         b_block_origin_idx,
                                         b_thread_bufs(I1));
                }
                else if constexpr(m0.value == MRepeat - 1)
                {
                    a_blockwise_copy.RunWrite(a_block_desc, a_block_buf.At(I1));
                }

                vector_type<AccDataType, 2> c_scale_thread_vec;
                c_scale_thread_vec.template AsType<AccDataType>()(Number<0>{}) =
                    c_scale_thread_buf[m0];
                c_scale_thread_vec.template AsType<AccDataType>()(Number<1>{}) =
                    c_scale_thread_buf[m0];

                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    static_for<0, xdlops_gemm.GetRegSizePerXdlops(), 1>{}([&](auto t) {
                        c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{})
                            .template AsType<AccDataType>()(Number<t>{}) = 0;
                    });
                    static_for<0, KRepeat, 1>{}([&](auto k0) {
                        vector_type<ComputeDataType, KPack> a_thread_vec;
                        vector_type<ComputeDataType, KPack> b_thread_vec;

                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<ComputeDataType>()(ik) =
                                a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0 % 2, I0, I0, k0, I0, ik))>{}];
                            b_thread_vec.template AsType<ComputeDataType>()(ik) =
                                b_thread_bufs[I0][Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, I0, k0, ik))>{}];
                        });

                        using mfma_input_type =
                            typename vector_type<ComputeDataType, xdlops_gemm.K1PerXdlops>::type;

                        xdlops_gemm.template Run<>(
                            a_thread_vec.template AsType<mfma_input_type>(),
                            b_thread_vec.template AsType<mfma_input_type>(),
                            c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{}));
                    });

                    constexpr index_t c_offset =
                        c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                    static_for<0, xdlops_gemm.GetRegSizePerXdlops() / 2, 1>{}([&](auto t) {
                        using pk_fma_type = typename vector_type<AccDataType, 2>::type;

                        c_thread_buf.GetVectorTypeReference(Number<c_offset>{})
                            .template AsType<pk_fma_type>()(t) = __builtin_elementwise_fma(
                            c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{})
                                .template AsType<pk_fma_type>()[t],
                            c_scale_thread_vec.template AsType<pk_fma_type>()[Number<0>{}],
                            c_thread_buf.GetVectorTypeReference(Number<c_offset>{})
                                .template AsType<pk_fma_type>()[t]);
                    });
                });

                if constexpr(m0.value == MRepeat - 1)
                {
                    block_sync_lds();

                    static_for<0, KRepeat, 1>{}([&](auto k0) {
                        a_thread_copy_.Run(
                            a_block_desc_m0_m1_m2_k0_k1_k2,
                            make_tuple(Number<(m0 + 1) % MRepeat>{}, I0, I0, k0, I0, I0),
                            a_block_buf.At(I1),
                            a_thread_desc_,
                            make_tuple(Number<(m0 + 1) % 2>{}, I0, I0, k0, I0, I0),
                            a_thread_buf);
                    });
                }
                else
                {
                    static_for<0, KRepeat, 1>{}([&](auto k0) {
                        a_thread_copy_.Run(
                            a_block_desc_m0_m1_m2_k0_k1_k2,
                            make_tuple(Number<(m0 + 1) % MRepeat>{}, I0, I0, k0, I0, I0),
                            a_block_buf.At(I0),
                            a_thread_desc_,
                            make_tuple(Number<(m0 + 1) % 2>{}, I0, I0, k0, I0, I0),
                            a_thread_buf);
                    });
                }

                HotLoopScheduler(m0);
            });

            static_for<0, MRepeat, 1>{}([&](auto m0) {
                c_scale_thread_buf(m0) = a_scale_thread_buf[m0] * b_scale_thread_buf[I0];
            });

            static_for<0, MRepeat, 1>{}([&](auto m0) {
                vector_type<AccDataType, 2> c_scale_thread_vec;
                c_scale_thread_vec.template AsType<AccDataType>()(Number<0>{}) =
                    c_scale_thread_buf[m0];
                c_scale_thread_vec.template AsType<AccDataType>()(Number<1>{}) =
                    c_scale_thread_buf[m0];

                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    static_for<0, xdlops_gemm.GetRegSizePerXdlops(), 1>{}([&](auto t) {
                        c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{})
                            .template AsType<AccDataType>()(Number<t>{}) = 0;
                    });
                    static_for<0, KRepeat, 1>{}([&](auto k0) {
                        vector_type<ComputeDataType, KPack> a_thread_vec;
                        vector_type<ComputeDataType, KPack> b_thread_vec;

                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<ComputeDataType>()(ik) =
                                a_thread_buf[Number<a_thread_desc_.CalculateOffset(make_tuple(
                                    (m0 + HotloopLocalBufSwitch) % 2, I0, I0, k0, I0, ik))>{}];
                            b_thread_vec.template AsType<ComputeDataType>()(ik) =
                                b_thread_bufs[I1][Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, I0, k0, ik))>{}];
                        });

                        using mfma_input_type =
                            typename vector_type<ComputeDataType, xdlops_gemm.K1PerXdlops>::type;

                        xdlops_gemm.template Run<>(
                            a_thread_vec.template AsType<mfma_input_type>(),
                            b_thread_vec.template AsType<mfma_input_type>(),
                            c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{}));
                    });
                    constexpr index_t c_offset =
                        c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                    static_for<0, xdlops_gemm.GetRegSizePerXdlops() / 2, 1>{}([&](auto t) {
                        using pk_fma_type = typename vector_type<AccDataType, 2>::type;

                        c_thread_buf.GetVectorTypeReference(Number<c_offset>{})
                            .template AsType<pk_fma_type>()(t) = __builtin_elementwise_fma(
                            c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{})
                                .template AsType<pk_fma_type>()[t],
                            c_scale_thread_vec.template AsType<pk_fma_type>()[Number<0>{}],
                            c_thread_buf.GetVectorTypeReference(Number<c_offset>{})
                                .template AsType<pk_fma_type>()[t]);
                    });
                });

                if constexpr(m0.value != (MRepeat - 1))
                {
                    static_for<0, KRepeat, 1>{}([&](auto k0) {
                        a_thread_copy_.Run(
                            a_block_desc_m0_m1_m2_k0_k1_k2,
                            make_tuple(Number<m0 + 1>{}, I0, I0, k0, I0, I0),
                            a_block_buf.At(I1),
                            a_thread_desc_,
                            make_tuple(
                                Number<(m0 + 1 + HotloopLocalBufSwitch) % 2>{}, I0, I0, k0, I0, I0),
                            a_thread_buf);
                    });

                    EpilogueScheduler_2();
                }
            });
            // Let's leak last MFMA block to epilogue region, cover the potential lds-shuffle
            // latency
            // // __builtin_amdgcn_sched_barrier(0);
        }
        else
        {
            static_for<0, MRepeat, 1>{}([&](auto m0) {
                vector_type<AccDataType, 2> c_scale_thread_vec;
                c_scale_thread_vec.template AsType<AccDataType>()(Number<0>{}) =
                    c_scale_thread_buf[m0];
                c_scale_thread_vec.template AsType<AccDataType>()(Number<1>{}) =
                    c_scale_thread_buf[m0];

                static_for<0, NRepeat, 1>{}([&](auto n0) {
                    static_for<0, xdlops_gemm.GetRegSizePerXdlops(), 1>{}([&](auto t) {
                        c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{})
                            .template AsType<AccDataType>()(Number<t>{}) = 0;
                    });
                    static_for<0, KRepeat, 1>{}([&](auto k0) {
                        vector_type<ComputeDataType, KPack> a_thread_vec;
                        vector_type<ComputeDataType, KPack> b_thread_vec;

                        static_for<0, KPack, 1>{}([&](auto ik) {
                            a_thread_vec.template AsType<ComputeDataType>()(ik) =
                                a_thread_buf[Number<a_thread_desc_.CalculateOffset(
                                    make_tuple(m0 % 2, I0, I0, k0, I0, ik))>{}];
                            b_thread_vec.template AsType<ComputeDataType>()(ik) =
                                b_thread_bufs[I0][Number<b_thread_desc_.CalculateOffset(
                                    make_tuple(n0, I0, k0, ik))>{}];
                        });

                        using mfma_input_type =
                            typename vector_type<ComputeDataType, xdlops_gemm.K1PerXdlops>::type;

                        xdlops_gemm.template Run<>(
                            a_thread_vec.template AsType<mfma_input_type>(),
                            b_thread_vec.template AsType<mfma_input_type>(),
                            c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{}));
                    });
                    constexpr index_t c_offset =
                        c_thread_desc_.CalculateOffset(make_tuple(m0, n0, 0));

                    static_for<0, xdlops_gemm.GetRegSizePerXdlops() / 2, 1>{}([&](auto t) {
                        using pk_fma_type = typename vector_type<AccDataType, 2>::type;

                        c_thread_buf.GetVectorTypeReference(Number<c_offset>{})
                            .template AsType<pk_fma_type>()(t) = __builtin_elementwise_fma(
                            c_thread_buf_per_scale.GetVectorTypeReference(Number<0>{})
                                .template AsType<pk_fma_type>()[t],
                            c_scale_thread_vec.template AsType<pk_fma_type>()[Number<0>{}],
                            c_thread_buf.GetVectorTypeReference(Number<c_offset>{})
                                .template AsType<pk_fma_type>()[t]);
                    });
                });

                if constexpr(m0.value != (MRepeat - 1))
                {
                    static_for<0, KRepeat, 1>{}([&](auto k0) {
                        a_thread_copy_.Run(a_block_desc_m0_m1_m2_k0_k1_k2,
                                           make_tuple(Number<m0 + 1>{}, I0, I0, k0, I0, I0),
                                           a_block_buf.At(I0),
                                           a_thread_desc_,
                                           make_tuple(Number<(m0 + 1) % 2>{}, I0, I0, k0, I0, I0),
                                           a_thread_buf);
                    });

                    EpilogueScheduler_2();
                }
            });
        }
    }

    protected:
    // MRepeat MWave MLane KRepeat KLane KPack
    // KRepeat -> MRepeat-> Mwave->KLane->MLane->KPack
    // Reduce the vgpr usage here.
    static constexpr auto a_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(I2, I1, I1, Number<KRepeat>{}, I1, Number<KPack>{}));

    using AThreadCopy = ThreadwiseTensorSliceTransfer_v4<ADataType,
                                                         ComputeDataType,
                                                         decltype(a_block_desc_m0_m1_m2_k0_k1_k2),
                                                         decltype(a_thread_desc_),
                                                         Sequence<1, 1, 1, 1, 1, KPack>,
                                                         Sequence<0, 1, 2, 3, 4, 5>,
                                                         5,
                                                         A_K1,
                                                         A_K1>;

    AThreadCopy a_thread_copy_{Base::CalculateAThreadOriginDataIndex6D()};

    static constexpr auto b_thread_desc_ = make_naive_tensor_descriptor_packed(
        make_tuple(Number<NRepeat>{}, I1, Number<KRepeat>{}, Number<KPack>{}));

    static constexpr BTileDesc b_block_desc_n0_n1_k0_k1;
    using Base::c_thread_desc_;
};

} // namespace ck
