// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <sstream>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_microscale_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/host/concat.hpp"

namespace ck_tile {

//  A Tile Window: global memory
//  B Tile Window: global memory
//  C Distributed tensor: register

template <typename Problem, typename Policy = GemmMicroscalePipelineAgBgCrPolicy>
struct MicroscaleGemmPipelineAgBgCrCompV3 : public BaseGemmPipelineAgBgCrCompV3<Problem>
{
    using Base             = BaseGemmPipelineAgBgCrCompV3<Problem>;
    using PipelineImplBase = GemmMicroscalePipelineAgBgCrImplBase<Problem, Policy>;

    using ADataType = remove_cvref_t<typename Problem::ADataType>;
    using BDataType = remove_cvref_t<typename Problem::BDataType>;

    using BDqDataType = remove_cvref_t<typename Problem::ADataType>;

    static constexpr bool IsCastBeforeLDS = Problem::BCastPolicy == CastPolicy::BeforeLDSWrite;

    using BLDSType = std::conditional_t<IsCastBeforeLDS, BDqDataType, BDataType>;

    using BQDataType      = remove_cvref_t<typename Problem::BQDataType>;
    using CDataType       = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape  = remove_cvref_t<typename Problem::BlockGemmShape>;
    using BQuantGroupSize = remove_cvref_t<typename Problem::BQuantGroupSize>;

    static_assert(BQuantGroupSize::kM == 1, "only N/K blocks for BQuant kernel!");

    using I0 = number<0>;
    using I1 = number<1>;
    using I2 = number<2>;

    static constexpr index_t APackedSize =
        ck_tile::numeric_traits<remove_cvref_t<ADataType>>::PackedSize;

    static constexpr index_t BPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<BDataType>>::PackedSize;

    static constexpr index_t BQPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<BQDataType>>::PackedSize;

    static constexpr index_t BLDSPackedSize =
        ck_tile::numeric_traits<remove_cvref_t<BLDSType>>::PackedSize;

    using ALayout  = remove_cvref_t<typename Problem::ALayout>;
    using BQLayout = remove_cvref_t<typename Problem::BQLayout>;
    using BLayout  = remove_cvref_t<typename Problem::BLayout>;
    using CLayout  = remove_cvref_t<typename Problem::CLayout>;

    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;

    static constexpr index_t BlockSize = Problem::kBlockSize;
    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    static constexpr index_t NPerBlockBQ = BlockGemmShape::kN / BQuantGroupSize::kN;
    static constexpr index_t KPerBlockBQ = BlockGemmShape::kK / BQuantGroupSize::kK;

    static constexpr index_t GetVectorSizeA() { return Policy::template GetVectorSizeA<Problem>(); }
    static constexpr index_t GetVectorSizeB() { return Policy::template GetVectorSizeB<Problem>(); }
    static constexpr index_t GetVectorSizeC() { return Policy::template GetVectorSizeC<Problem>(); }
    static constexpr index_t GetVectorSizeBQ()
    {
        return Policy::template GetVectorSizeBQ<Problem>();
    }

    static constexpr index_t GetSmemPackA() { return Policy::template GetSmemPackA<Problem>(); }
    static constexpr index_t GetSmemPackB() { return Policy::template GetSmemPackB<Problem>(); }

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr bool DoubleSmemBuffer = Problem::DoubleSmemBuffer;

    static constexpr bool HasHotLoop = Problem::HasHotLoop;
    static constexpr auto TailNum    = Problem::TailNum;
    static constexpr auto Scheduler  = Problem::Scheduler;

    static constexpr auto is_a_load_tr_v = bool_constant<PipelineImplBase::is_a_load_tr>{};
    static constexpr auto is_b_load_tr_v = bool_constant<PipelineImplBase::is_b_load_tr>{};

    using Base::PrefetchStages;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(I0{});
        constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(I1{});
        return concat('_', "mxfp4gemm_pipeline_AgBgCrCompV3", 
                      concat('x', MPerBlock, NPerBlock, KPerBlock),  BlockSize,
                      concat('x', WaveNumM, WaveNumN),
                      concat('x', kPadM, kPadN, kPadK),
                      concat('x', kPadM, kPadN, kPadK), BQuantGroupSize::GetName());
        // clang-format on
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    CK_TILE_HOST static std::string Print()
    {
        constexpr index_t MPerXDL = BlockGemm::WarpGemm::kM;
        constexpr index_t NPerXDL = BlockGemm::WarpGemm::kN;
        constexpr index_t KPerXDL = BlockGemm::WarpGemm::WarpGemmAttribute::Impl::kK;

        constexpr index_t WaveSize = 64;
        constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(I0{});
        constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(I1{});

        constexpr index_t A_LDS_Read_Width = GetSmemPackA();
        constexpr index_t B_LDS_Read_Width = GetSmemPackB();

        constexpr index_t A_LDS_Write_Width = GetSmemPackA();
        constexpr index_t B_LDS_Write_Width = GetSmemPackB();

        constexpr index_t A_Buffer_Load_Inst_Num =
            MPerBlock * KPerBlock / (BlockSize * GetVectorSizeA());
        constexpr index_t B_Buffer_Load_Inst_Num =
            NPerBlock * KPerBlock / (BlockSize * GetVectorSizeB());
        constexpr index_t BQ_Buffer_Load_Inst_Num =
            NPerBlock * KPerBlockBQ / (BlockSize * GetVectorSizeBQ());

        constexpr index_t A_LDS_Write_Inst_Num =
            MPerBlock * KPerBlock / (BlockSize * A_LDS_Write_Width);
        constexpr index_t B_LDS_Write_Inst_Num =
            NPerBlock * KPerBlock / (BlockSize * B_LDS_Write_Width);

        constexpr index_t A_LDS_Read_Inst_Num =
            WaveNumN * MPerBlock * KPerBlock / (BlockSize * A_LDS_Read_Width);
        constexpr index_t B_LDS_Read_Inst_Num =
            WaveNumM * NPerBlock * KPerBlock / (BlockSize * B_LDS_Read_Width);

        constexpr index_t C_MFMA_Inst_Num = MPerBlock * NPerBlock * KPerBlock /
                                            (BlockSize / WaveSize) / (MPerXDL * NPerXDL * KPerXDL);

        auto str = std::stringstream{};

        str << "A/B vector size: " << GetVectorSizeA() << ", " << GetVectorSizeB() << ", "
            << "BQ vector size: " << GetVectorSizeBQ() << "\n"
            << "A/B LDS read/write width: " << A_LDS_Read_Width << ", " << B_LDS_Read_Width << "\n"
            << "A/B buffer load inst: " << A_Buffer_Load_Inst_Num << ", " << B_Buffer_Load_Inst_Num
            << ", " << "BQ buffer load inst: " << BQ_Buffer_Load_Inst_Num << "\n"
            << "A/B LDS write inst: " << A_LDS_Write_Inst_Num << ", " << B_LDS_Write_Inst_Num
            << "\n"
            << "A/B LDS read inst: " << A_LDS_Read_Inst_Num << ", " << B_LDS_Read_Inst_Num << "\n"
            << "C MFMA inst: " << C_MFMA_Inst_Num << "\n"
            << "BQuantGroupSize: " << BQuantGroupSize::GetName() << "\n"
            << "KPack: " << BlockGemm::Traits::KPack << "\n"
            << "PrefetchStages: " << PrefetchStages << "\n";
        return str.str();
    }

    template <GemmPipelineScheduler Scheduler>
    struct PipelineImpl : public PipelineImplBase
    {
    };

    template <>
    struct PipelineImpl<GemmPipelineScheduler::Intrawave> : public PipelineImplBase
    {
        using Base = PipelineImplBase;

        static constexpr bool is_a_col_major =
            std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
        static constexpr bool is_b_row_major =
            std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;

        CK_TILE_DEVICE static constexpr auto HotLoopScheduler()
        {
            constexpr index_t MPerXDL = BlockGemm::WarpGemm::kM;
            constexpr index_t NPerXDL = BlockGemm::WarpGemm::kN;
            constexpr index_t KPerXDL = BlockGemm::WarpGemm::WarpGemmAttribute::Impl::kK;

            constexpr index_t WaveSize = 64;
            constexpr index_t WaveNumM = BlockGemmShape::BlockWarps::at(I0{});
            constexpr index_t WaveNumN = BlockGemmShape::BlockWarps::at(I1{});

            // Below should be equal to AK1|BK1
            constexpr index_t A_LDS_Read_Width = GetSmemPackA();
            constexpr index_t B_LDS_Read_Width = GetSmemPackB();

            constexpr index_t A_LDS_Write_Width = GetSmemPackA();
            constexpr index_t B_LDS_Write_Width = GetSmemPackB();

            constexpr index_t A_Buffer_Load_Inst_Num =
                MPerBlock * KPerBlock / (BlockSize * GetVectorSizeA());
            constexpr index_t B_Buffer_Load_Inst_Num =
                NPerBlock * KPerBlock / (BlockSize * GetVectorSizeB());

            constexpr index_t A_LDS_Write_Inst_Num =
                MPerBlock * KPerBlock / (BlockSize * A_LDS_Write_Width);
            constexpr index_t B_LDS_Write_Inst_Num =
                NPerBlock * KPerBlock / (BlockSize * B_LDS_Write_Width);

            constexpr index_t A_LDS_Read_Inst_Num =
                WaveNumN * MPerBlock * KPerBlock / (BlockSize * A_LDS_Read_Width);
            constexpr index_t B_LDS_Read_Inst_Num =
                WaveNumM * NPerBlock * KPerBlock / (BlockSize * B_LDS_Read_Width);

            constexpr index_t C_MFMA_Inst_Num = MPerBlock * NPerBlock * KPerBlock /
                                                (BlockSize / WaveSize) /
                                                (MPerXDL * NPerXDL * KPerXDL);

            // A/B split schedule
            // compiler is likely to use ds_read2 when instruction width smaller than 16bytes
            constexpr auto num_ds_read_inst_a =
                A_LDS_Read_Width * sizeof(ADataType) / APackedSize == 16 ? A_LDS_Read_Inst_Num
                                                                         : A_LDS_Read_Inst_Num / 2;
            constexpr auto num_ds_read_inst_b =
                B_LDS_Read_Width * sizeof(BLDSType) / BLDSPackedSize == 16
                    ? B_LDS_Read_Inst_Num
                    : B_LDS_Read_Inst_Num / 2;

            constexpr auto num_ds_write_inst_a = A_LDS_Write_Inst_Num;
            constexpr auto num_ds_write_inst_b = B_LDS_Write_Inst_Num;

            constexpr auto num_buffer_load_inst_a = A_Buffer_Load_Inst_Num;
            constexpr auto num_buffer_load_inst_b = B_Buffer_Load_Inst_Num;

            constexpr auto num_mfma_inst = C_MFMA_Inst_Num;

            constexpr auto mfma_cycle = NPerXDL == 16 ? 16 : 32;
            constexpr auto ds_read_a_issue_cycle =
                A_LDS_Read_Width * sizeof(ADataType) / APackedSize == 16 ? 8 : 4;
            constexpr auto ds_read_b_issue_cycle =
                B_LDS_Read_Width * sizeof(BLDSType) / BLDSPackedSize == 16 ? 8 : 4;
            constexpr auto ds_read_a_mfma_rate =
                (mfma_cycle - 4 + 2 * ds_read_a_issue_cycle - 1) / (2 * ds_read_a_issue_cycle);
            constexpr auto ds_read_b_mfma_rate =
                (mfma_cycle - 4 + 2 * ds_read_b_issue_cycle - 1) / (2 * ds_read_b_issue_cycle);

            constexpr auto num_dsread_a_mfma =
                (num_ds_read_inst_a + ds_read_a_mfma_rate - 1) / ds_read_a_mfma_rate;
            constexpr auto num_dsread_b_mfma =
                (num_ds_read_inst_b + ds_read_b_mfma_rate - 1) / ds_read_b_mfma_rate;

            // stage 1
            // Separate this part?
            // constexpr auto num_mfma_per_ds_read = sizeof(ComputeDataType) / sizeof(ADataType) >
            //                                               sizeof(ComputeDataType) /
            //                                               sizeof(BDataType)
            //                                           ? sizeof(ComputeDataType) /
            //                                           sizeof(ADataType) : sizeof(ComputeDataType)
            //                                           / sizeof(BDataType);
            constexpr auto num_mfma_stage1 =
                num_mfma_inst - (num_dsread_a_mfma + num_dsread_b_mfma);
            constexpr auto num_mfma_per_issue =
                num_mfma_stage1 / (num_buffer_load_inst_a + num_buffer_load_inst_b);
            constexpr auto num_dswrite_per_issue_a = num_ds_write_inst_a / num_buffer_load_inst_a;
            constexpr auto num_dswrite_per_issue_b = num_ds_write_inst_b / num_buffer_load_inst_b;

            static_for<0, num_buffer_load_inst_a, 1>{}([&](auto i) {
                ignore = i;
                static_for<0, num_dswrite_per_issue_a, 1>{}([&](auto idswrite) {
                    ignore = idswrite;
                    __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                });
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(
                    0x008, num_mfma_per_issue - num_dswrite_per_issue_a, 0); // MFMA
            });
            static_for<0, num_buffer_load_inst_b, 1>{}([&](auto i) {
                ignore = i;
                static_for<0, num_dswrite_per_issue_b, 1>{}([&](auto idswrite) {
                    ignore = idswrite;
                    __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS write
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
                });
                __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); // VMEM read
                __builtin_amdgcn_sched_group_barrier(
                    0x008, num_mfma_per_issue - num_dswrite_per_issue_b, 0); // MFMA
            });

            // stage 2
            static_for<0, num_dsread_a_mfma, 1>{}([&](auto i) {
                if constexpr((num_ds_read_inst_a - (i + 1) * ds_read_a_mfma_rate) >=
                             ds_read_a_mfma_rate)
                {
                    __builtin_amdgcn_sched_group_barrier(0x100, ds_read_a_mfma_rate, 0); // DS read
                }
                else
                {
                    __builtin_amdgcn_sched_group_barrier(
                        0x100,
                        num_ds_read_inst_a - (num_dsread_a_mfma - 1) * ds_read_a_mfma_rate,
                        0); // DS read
                }
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });

            static_for<0, num_dsread_b_mfma, 1>{}([&](auto i) {
                if constexpr((num_ds_read_inst_b - (i + 1) * ds_read_b_mfma_rate) >=
                             ds_read_b_mfma_rate)
                {
                    __builtin_amdgcn_sched_group_barrier(0x100, ds_read_b_mfma_rate, 0); // DS read
                }
                else
                {
                    __builtin_amdgcn_sched_group_barrier(
                        0x100,
                        num_ds_read_inst_b - (num_dsread_b_mfma - 1) * ds_read_b_mfma_rate,
                        0); // DS read
                }
                __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); // MFMA
            });
        }

        template <typename TileType, typename CastTileType, typename ScaleTileType>
        CK_TILE_DEVICE static void ScaleTile(const TileType& block_tile,
                                             CastTileType& block_tile_cast,
                                             const ScaleTileType& scale_tile)
        {
            if constexpr(IsCastBeforeLDS)
            {
                constexpr auto b_block = TileType::get_distributed_spans();

                // Internally this is using V_CVT_SCALEF32_PK_BF16_FP4 or V_CVT_SCALEF32_PK_FP16_FP4
                // on gfx950
                auto pk_mxfp4_to_compute_v2 = [](auto pk_mxfp4, float fscale) {
                    if constexpr(std::is_same_v<BDqDataType, half_t>)
                    {
                        return pk_fp4_to_fp16x2(pk_mxfp4, fscale);
                    }
                    else if constexpr(std::is_same_v<BDqDataType, bf16_t>)
                    {
                        return pk_fp4_to_bf16x2(pk_mxfp4, fscale);
                    }
                    else
                    {
                        static_assert(false, "unsupported compute type");
                    }
                };

                constexpr index_t BQuantGroupSizeIdx0 =
                    std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>
                        ? BQuantGroupSize::kN
                        : BQuantGroupSize::kK;
                constexpr index_t BQuantGroupSizeIdx1 =
                    std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>
                        ? BQuantGroupSize::kK
                        : BQuantGroupSize::kN;

                // The input indices are with respect to B block tile. If B and Bq have different
                // layouts, the indices must be swapped
                auto make_bq_index = [](auto idx0, auto idx1) {
                    if constexpr(std::is_same_v<BLayout, BQLayout>)
                    {
                        return make_tuple(
                            tile_distributed_index<idx0.impl_.at(0) / BQuantGroupSizeIdx0>{},
                            tile_distributed_index<idx1.impl_.at(0) / BQuantGroupSizeIdx1>{});
                    }
                    else
                    {
                        return make_tuple(
                            tile_distributed_index<idx1.impl_.at(0) / BQuantGroupSizeIdx0>{},
                            tile_distributed_index<idx0.impl_.at(0) / BQuantGroupSizeIdx1>{});
                    }
                };

                sweep_tile_span(b_block[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(b_block[number<1>{}], [&](auto idx1) {
                        if constexpr(std::is_same_v<BDataType, ck_tile::pk_fp4_t>)
                        {
                            if constexpr(idx1.impl_.at(0) % BPackedSize == 0)
                            {
                                constexpr auto idx1_lo = tile_distributed_index<idx1.impl_.at(0)>{};
                                constexpr auto idx1_hi =
                                    tile_distributed_index<idx1.impl_.at(0) + 1>{};

                                constexpr auto i_j_idx_lo = make_tuple(idx0, idx1_lo);
                                constexpr auto i_j_idx_hi = make_tuple(idx0, idx1_hi);

                                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                                auto b_pack            = block_tile[i_j_idx];

                                constexpr auto i_j_idx_scale_lo = make_bq_index(idx0, idx1_lo);
                                constexpr auto i_j_idx_scale_hi = make_bq_index(idx0, idx1_hi);

                                // If the scale is the same for packed values, use pk cvt scale
                                // instructions, otherwise scale and cast element by element
                                if constexpr(i_j_idx_scale_lo[I0{}].impl_.at(0) ==
                                                 i_j_idx_scale_hi[I0{}].impl_.at(0) &&
                                             i_j_idx_scale_lo[I1{}].impl_.at(0) ==
                                                 i_j_idx_scale_hi[I1{}].impl_.at(0))
                                {
                                    float scale = float(scale_tile[i_j_idx_scale_lo]);
                                    auto cvt    = pk_mxfp4_to_compute_v2(b_pack, scale);

                                    block_tile_cast(i_j_idx_lo) = cvt.x;
                                    block_tile_cast(i_j_idx_hi) = cvt.y;
                                }
                                else
                                {
                                    float scale_lo = float(scale_tile[i_j_idx_scale_lo]);
                                    auto b_f4_lo =
                                        type_convert<pk_fp4_t>(b_pack.unpack(number<0>{}));
                                    block_tile_cast(i_j_idx_lo) = type_convert<BDqDataType>(
                                        type_convert<float>(b_f4_lo) * scale_lo);

                                    float scale_hi = float(scale_tile[i_j_idx_scale_hi]);
                                    auto b_f4_hi =
                                        type_convert<pk_fp4_t>(b_pack.unpack(number<1>{}));
                                    block_tile_cast(i_j_idx_hi) = type_convert<BDqDataType>(
                                        type_convert<float>(b_f4_hi) * scale_hi);
                                }
                            }
                        }
                        else
                        {
                            constexpr auto i_j_idx       = make_tuple(idx0, idx1);
                            constexpr auto i_j_idx_scale = make_bq_index(idx0, idx1);
                            float scale                  = float(scale_tile[i_j_idx_scale]);

                            auto b_pack = block_tile[i_j_idx];
                            block_tile_cast(i_j_idx) =
                                type_convert<BDqDataType>(type_convert<float>(b_pack) * scale);
                        }
                    });
                });
            }
        }

        template <typename WindowType, typename TileType, typename ElementwiseFunc>
        CK_TILE_DEVICE void ALocalPrefill(WindowType& lds_window,
                                          const TileType& block_tile,
                                          const ElementwiseFunc& element_func) const
        {
            if constexpr(is_a_col_major && !is_a_load_tr_v())
            {
                auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                    Policy::template MakeShuffledARegTileDistribution<Problem>());
                transpose_tile2d(a_shuffle_tmp, block_tile);
                Base::LocalPrefill(lds_window, a_shuffle_tmp, element_func);
            }
            else
            {
                Base::LocalPrefill(lds_window, block_tile, element_func);
            }
        }

        template <typename WindowType,
                  typename TileType,
                  typename TileTypeCast,
                  typename ElementwiseFunc>
        CK_TILE_DEVICE void BLocalPrefill(WindowType& lds_window,
                                          const TileType& block_tile,
                                          const TileTypeCast& block_tile_cast,
                                          const ElementwiseFunc& element_func) const
        {
            // Fill LDS and apply the scale if IsCastBeforeLDS
            auto get_b_block_tile = [](auto& b_block_tile_orig, auto& b_block_tile_cast) {
                if constexpr(IsCastBeforeLDS)
                {
                    return b_block_tile_cast;
                }
                else
                {
                    return b_block_tile_orig;
                }
            };

            if constexpr(is_b_row_major && !is_b_load_tr_v())
            {
                auto b_shuffle_tmp = make_static_distributed_tensor<BLDSType>(
                    Policy::template MakeShuffledBRegTileDistribution<Problem>());
                transpose_tile2d(b_shuffle_tmp, get_b_block_tile(block_tile, block_tile_cast));
                Base::LocalPrefill(lds_window, b_shuffle_tmp, element_func);
            }
            else
            {
                Base::LocalPrefill(
                    lds_window, get_b_block_tile(block_tile, block_tile_cast), element_func);
            }
        }

        template <typename BlockGemmType,
                  typename AWindowType,
                  typename BWindowType,
                  typename QTileType>
        CK_TILE_DEVICE void LocalPrefetch(BlockGemmType& block_gemm,
                                          const AWindowType& a_lds_window,
                                          const BWindowType& b_lds_window,
                                          const QTileType& q_block_tile) const
        {
            // Load from LDS
            // It can apply the scale and cast if we scale after reading from LDS
            if constexpr(IsCastBeforeLDS)
            {
                block_gemm.LocalPrefetch(
                    a_lds_window, b_lds_window, is_a_load_tr_v, is_b_load_tr_v);
            }
            else
            {
                block_gemm.LocalPrefetch(
                    a_lds_window, b_lds_window, q_block_tile, is_a_load_tr_v, is_b_load_tr_v);
            }
        }

        template <bool HasHotLoop,
                  TailNumber TailNum,
                  typename ADramBlockWindowTmp,
                  typename BDramBlockWindowTmp,
                  typename BQDramBlockWindowTmp,
                  typename AElementFunction,
                  typename BElementFunction>
        CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                       const AElementFunction& a_element_func,
                                       const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                       const BElementFunction& b_element_func,
                                       const BQDramBlockWindowTmp& bq_dram_block_window_tmp,
                                       index_t num_loop,
                                       void* p_smem) const
        {
            // -----------------------------------------------------------------------------------------
            // Pipeline checks
            static_assert(
                std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BDataType,
                                   remove_cvref_t<typename BDramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BQDataType,
                                   remove_cvref_t<typename BQDramBlockWindowTmp::DataType>>,
                "A/B/BQ Dram block window should have the same data type as appropriate "
                "([A|B|BQ]DataType) defined in Problem definition!");

            constexpr bool is_bq_col_major =
                std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>;

            static_assert(is_bq_col_major
                              ? (NPerBlockBQ == BQDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 KPerBlockBQ == BQDramBlockWindowTmp{}.get_window_lengths()[I1{}])
                              : (KPerBlockBQ == BQDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 NPerBlockBQ == BQDramBlockWindowTmp{}.get_window_lengths()[I1{}]),
                          "Bq block window has incorrect lengths for defined BqLayout!");

            static_assert(is_a_col_major
                              ? (KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}])
                              : (MPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 KPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}]),
                          "A block window has incorrect lengths for defined ALayout!");
            static_assert(is_b_row_major
                              ? (KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1{}])
                              : (NPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 KPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1{}]),
                          "B block window has incorrect lengths for defined BLayout!");

            // ------------------------------------------------------------------------------------
            // Definitions of all needed tiles
            // int b_block_stride = 0;
            // A/B tiles in LDS
            auto&& [a_lds_block, b_lds_block] = Base::GetABLdsTensorViews(p_smem);

            // Tile distribution for load from lds
            constexpr auto a_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
            constexpr auto b_lds_load_tile_distr =
                make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());

            // A DRAM tile window for load
            // A LDS tile window for store
            // A LDS tile for block GEMM
            auto&& [a_copy_dram_window, a_copy_lds_window, a_lds_gemm_window] =
                Base::GetAWindows(a_dram_block_window_tmp, a_lds_block, a_lds_load_tile_distr);

            // B DRAM tile window for load, (kN, kK/2)
            // B LDS tile window for store, (kN, kK)
            // B LDS tile for block GEMM
            auto&& [b_copy_dram_window, b_copy_lds_window, b_lds_gemm_window] =
                Base::GetBWindows(b_dram_block_window_tmp, b_lds_block, b_lds_load_tile_distr);

            // B scale DRAM tile window for load
            auto bq_copy_dram_window = Base::GetBQDramLoadWindow(bq_dram_block_window_tmp);

            auto bq_block_tile = decltype(load_tile(bq_copy_dram_window)){};

            // This defines the scaled and casted block tile for B matrix.
            // Effectively, it is used only if we scale and cast before writing to LDS.
            auto bdq_block_tile = make_static_distributed_tensor<BDqDataType>(
                Policy::template MakeBRegTileDistribution<Problem>());

            // Block GEMM
            auto block_gemm       = BlockGemm();
            auto c_block_tile     = block_gemm.MakeCBlockTile();
            using ABlockTileDistr = decltype(a_copy_dram_window.get_tile_distribution());
            using BBlockTileDistr = decltype(b_copy_dram_window.get_tile_distribution());

            using ABlockTile =
                decltype(make_static_distributed_tensor<ADataType>(ABlockTileDistr{}));
            using BBlockTile =
                decltype(make_static_distributed_tensor<BDataType>(BBlockTileDistr{}));

            ABlockTile a_block_tile;
            BBlockTile b_block_tile;

            using ADramTileWindowStep  = typename ADramBlockWindowTmp::BottomTensorIndex;
            using BDramTileWindowStep  = typename BDramBlockWindowTmp::BottomTensorIndex;
            using BQDramTileWindowStep = typename BQDramBlockWindowTmp::BottomTensorIndex;

            constexpr ADramTileWindowStep a_dram_tile_window_step =
                is_a_col_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
            constexpr BDramTileWindowStep b_dram_tile_window_step =
                is_b_row_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);

            constexpr BQDramTileWindowStep b_scale_dram_tile_window_step =
                std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>
                    ? make_array(0, KPerBlock / BQuantGroupSize::kK)
                    : make_array(KPerBlock / BQuantGroupSize::kK, 0);
            // -----------------------------------------------------------------------------------------
            // Gemm pipeline start

            // prefetch stages

            // Vmem -> Vgpr 0
            Base::GlobalPrefetch(a_block_tile, a_copy_dram_window, a_dram_tile_window_step);
            Base::GlobalPrefetch(b_block_tile, b_copy_dram_window, b_dram_tile_window_step);

            // Vmem -> Vgpr 0 (Q matrix)
            // Scale and cast tile before writing to LDS (if IsCastBeforeLDS)
            bq_block_tile = load_tile(bq_copy_dram_window);
            move_tile_window(bq_copy_dram_window, b_scale_dram_tile_window_step);
            ScaleTile(b_block_tile, bdq_block_tile, bq_block_tile);

            // initialize C tile to zero
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);
            block_sync_lds();

            // Vgpr -> LDS 0
            ALocalPrefill(a_copy_lds_window, a_block_tile, a_element_func);
            BLocalPrefill(b_copy_lds_window, b_block_tile, bdq_block_tile, b_element_func);

            // Vmem -> Vgpr 1
            Base::GlobalPrefetch(a_block_tile, a_copy_dram_window, a_dram_tile_window_step);
            Base::GlobalPrefetch(b_block_tile, b_copy_dram_window, b_dram_tile_window_step);

            // If we scale and cast before writing to LDS,
            // we need to read another tile of Q matrix from Vmem, then scale and cast tile
            if constexpr(IsCastBeforeLDS)
            {
                bq_block_tile = load_tile(bq_copy_dram_window);
                move_tile_window(bq_copy_dram_window, b_scale_dram_tile_window_step);
            }
            ScaleTile(b_block_tile, bdq_block_tile, bq_block_tile);

            block_sync_lds();

            // LDS -> Vgpr 0
            LocalPrefetch(block_gemm, a_lds_gemm_window, b_lds_gemm_window, bq_block_tile);

            __builtin_amdgcn_sched_barrier(0);

            // main body
            if constexpr(HasHotLoop)
            {
                index_t i = 0;
                do
                {
                    block_sync_lds();

                    // Vgpr -> LDS
                    ALocalPrefill(a_copy_lds_window, a_block_tile, a_element_func);
                    BLocalPrefill(b_copy_lds_window, b_block_tile, bdq_block_tile, b_element_func);

                    // Vmem -> Vgpr
                    Base::GlobalPrefetch(a_block_tile, a_copy_dram_window, a_dram_tile_window_step);
                    Base::GlobalPrefetch(b_block_tile, b_copy_dram_window, b_dram_tile_window_step);

                    // Vmem -> Vgpr (Q matrix)
                    // Scale and cast tile before writing to LDS (if IsCastBeforeLDS)
                    bq_block_tile = load_tile(bq_copy_dram_window);
                    move_tile_window(bq_copy_dram_window, b_scale_dram_tile_window_step);
                    ScaleTile(b_block_tile, bdq_block_tile, bq_block_tile);

                    // Consume tile
                    block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
                    block_sync_lds();

                    // LDS -> Vgpr
                    LocalPrefetch(block_gemm, a_lds_gemm_window, b_lds_gemm_window, bq_block_tile);

                    HotLoopScheduler();
                    __builtin_amdgcn_sched_barrier(0);

                    i += 1;
                } while(i < (num_loop - 1));
            }

            // tail
            if constexpr((TailNum == TailNumber::Full) || (TailNum == TailNumber::Odd))
            {
                // Leak last MFMA block to epilogue region, cover the potential lds-shuffle
                // latency
                block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
            }
            else
            {
                // If we scale and cast after reading from LDS,
                // we didn't read the second tile of Q matrix from Vmem during prefetch stages,
                // so we need to read the last tile here.
                // This is not a problem because we have all block_gemm instructions to hide the
                // latency.
                if constexpr(!IsCastBeforeLDS)
                {
                    bq_block_tile = load_tile(bq_copy_dram_window);
                    move_tile_window(bq_copy_dram_window, b_scale_dram_tile_window_step);
                }

                // Consume second to last tile
                block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
                block_sync_lds();

                // Vgpr -> LDS last tile
                ALocalPrefill(a_copy_lds_window, a_block_tile, a_element_func);
                BLocalPrefill(b_copy_lds_window, b_block_tile, bdq_block_tile, b_element_func);

                block_sync_lds();

                // LDS -> Vgpr last tile
                LocalPrefetch(block_gemm, a_lds_gemm_window, b_lds_gemm_window, bq_block_tile);

                // Consume last tile
                block_gemm(c_block_tile, a_lds_gemm_window, b_lds_gemm_window);
                block_sync_lds();
            }
            __builtin_amdgcn_sched_barrier(0);
            return c_block_tile;
        }
    };

    /**
     * @brief This function runs the pipeline using compile-time known hot loop and tail number.
     * @param num_loop The number of loop iterations. This is determined at runtime due to e.g.
     * SplitK.
     * @note This is used by the kernel variants that are able to determine
     *       hot loop and tail number on the host side, e.g. non-persistent gemm kernel.
     */
    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename BQDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const BQDramBlockWindowTmp& bq_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* p_smem,
                                   index_t n = 0) const
    {
        ck_tile::ignore = n;
        return PipelineImpl<Scheduler>{}.template operator()<HasHotLoop, TailNum>(
            a_dram_block_window_tmp,
            identity{},
            b_dram_block_window_tmp,
            identity{},
            bq_dram_block_window_tmp,
            num_loop,
            p_smem);
    }
};

} // namespace ck_tile
