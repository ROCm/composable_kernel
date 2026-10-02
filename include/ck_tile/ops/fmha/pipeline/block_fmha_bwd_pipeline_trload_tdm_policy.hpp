// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_trload_default_policy.hpp"
#include "ck_tile/ops/fmha/pipeline/fmha_bwd_tdm_padding.hpp"

#include "ck_tile/core/utility/debug.hpp"

namespace ck_tile {

// Policy for the TDM-capable gfx1250 bwd decode pipeline.
//
// Everything not listed here is inherited from the trload policy; this policy
// replaces the operand descriptors and DRAM distributions with the plain boxes
// TDM writes, and re-does the smem budget and the hot-loop scheduler.
struct BlockFmhaBwdPipelineTrLoadTdmPolicy : BlockFmhaBwdPipelineTrLoadDefaultPolicy
{
    // ---- operand staging hooks, TDM form ------------------------------------
    //
    // TDM writes global->LDS without going through a register tile, so the DRAM
    // view is used as-is and the transfers retire on TENSORcnt rather than vmcnt.
    static constexpr bool kUsesTdm = true;

    template <typename T, typename TensorView>
    CK_TILE_HOST_DEVICE static constexpr auto MakeXDramStagingView(const TensorView& naive_view)
    {
        return naive_view;
    }

    template <typename T, index_t KPerBlock>
    CK_TILE_DEVICE static TDMConfig MakeTdmConfig()
    {
        constexpr auto cfg = GetTdmPaddingConfig<T, KPerBlock>();
        TDMConfig c;
        c.pad_enable              = cfg[number<0>{}];
        c.pad_config.pad_amount   = cfg[number<1>{}];
        c.pad_config.pad_interval = cfg[number<2>{}];
        return c;
    }

    template <typename T, index_t KPerBlock, typename LdsWindow, typename DramWindow>
    CK_TILE_DEVICE static void LoadBlockToLds(LdsWindow&& lds_window, const DramWindow& dram_window)
    {
#if defined(__gfx125__)
        load_tile_tdm(MakeTdmConfig<T, KPerBlock>(), lds_window, dram_window);
#else
        async_load_tile(lds_window, dram_window);
#endif
    }

    CK_TILE_DEVICE static void WaitBlockToLds()
    {
#if defined(__gfx125__)
        // TDM commits on TENSORcnt, which block_sync_lds alone does not fence.
        s_wait_tensorcnt_barrier<0>();
        block_sync_lds();
#else
        s_waitcnt</*vmcnt=*/0>();
#endif
    }

    CK_TILE_DEVICE static void WaitAllMem()
    {
#if defined(__gfx125__)
        s_wait_tensorcnt_barrier<0>();
        s_waitcnt</*vmcnt=*/0>();
        block_sync_lds();
#else
        __builtin_amdgcn_s_waitcnt(0);
#endif
    }

    template <typename Problem, index_t Rows, index_t Cols>
    CK_TILE_HOST_DEVICE static constexpr auto MakeTdmDramTileDistribution()
    {
        constexpr index_t warpNum = Problem::BlockFmhaShape::NumWarps;
        static_assert(Rows % warpNum == 0, "rows must divide by the warp count");
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<warpNum, Rows / warpNum>, sequence<Cols>>,
                                       tuple<sequence<1>>,
                                       tuple<sequence<0>>,
                                       sequence<1, 2>,
                                       sequence<1, 0>>{},
            bool_constant<true>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        return MakeTdmDramTileDistribution<Problem,
                                           Problem::BlockFmhaShape::kN0,
                                           Problem::BlockFmhaShape::kQKHeaddim>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVDramTileDistribution()
    {
        return MakeTdmDramTileDistribution<Problem,
                                           Problem::BlockFmhaShape::kN0,
                                           Problem::BlockFmhaShape::kVHeaddim>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQDramTileDistribution()
    {
        return MakeTdmDramTileDistribution<Problem,
                                           Problem::BlockFmhaShape::kM0,
                                           Problem::BlockFmhaShape::kQKHeaddim>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOGradDramTileDistribution()
    {
        return MakeTdmDramTileDistribution<Problem,
                                           Problem::BlockFmhaShape::kM0,
                                           Problem::BlockFmhaShape::kVHeaddim>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKTRegBlockDescriptor()
    {
        using BlockGemm = remove_cvref_t<decltype(GetSGradKTBlockGemm<Problem>())>;
        using WarpGemm  = typename BlockGemm::WarpGemm;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm4BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm4BlockWarps::at(number<1>{});

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kN0;

        constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WarpGemm::kN);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        constexpr auto kt_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<MWarp>,
            tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>, // 2 4, 4
            tuple<sequence<0, 1>>,
            tuple<sequence<0, 1>>,
            sequence<1, 2>,
            sequence<0, 0>>{};

        constexpr auto kt_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            kt_block_outer_dstr_encoding, typename WarpGemm::BWarpDstrEncoding{});

        auto output =
            make_static_tile_distribution(typename InputTileDistributionTraits<
                                          decltype(kt_block_dstr_encode),
                                          typename Problem::KDataType>::TransposedDstrEncode{});
        return output;
    }

    // Row pad, in elements, that breaks LDS bank conflicts between the TDM
    // write and the transposed read. 8 elements is 16 B, one ds_read_b128 unit,
    // so rows stay aligned for both.
    static constexpr index_t kTdmLdsPad = 8;

    template <typename T, index_t MNPerBlock, index_t KPerBlock>
    CK_TILE_HOST_DEVICE static constexpr auto MakeXLdsTdmBlockDescriptor()
    {
        constexpr index_t kKPack = 16 / sizeof(T);
        return make_naive_tensor_descriptor(
            make_tuple(number<MNPerBlock>{}, number<KPerBlock>{}),
            make_tuple(number<KPerBlock + kTdmLdsPad>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});
    }

    template <typename T, index_t KPerBlock>
    CK_TILE_HOST_DEVICE static constexpr auto GetTdmPaddingConfig()
    {
        return detail::make_fmha_bwd_tdm_padding_config<T, KPerBlock, kTdmLdsPad>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsWriteBlockDescriptor()
    {
        return MakeXLdsTdmBlockDescriptor<typename Problem::KDataType,
                                          Problem::BlockFmhaShape::kN0,
                                          Problem::BlockFmhaShape::kQKHeaddim>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsWriteBlockDescriptor()
    {
        return MakeXLdsTdmBlockDescriptor<typename Problem::VDataType,
                                          Problem::BlockFmhaShape::kN0,
                                          Problem::BlockFmhaShape::kVHeaddim>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsWriteBlockDescriptor()
    {
        return MakeXLdsTdmBlockDescriptor<typename Problem::QDataType,
                                          Problem::BlockFmhaShape::kM0,
                                          Problem::BlockFmhaShape::kQKHeaddim>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOGradLdsWriteBlockDescriptor()
    {
        return MakeXLdsTdmBlockDescriptor<typename Problem::OGradDataType,
                                          Problem::BlockFmhaShape::kM0,
                                          Problem::BlockFmhaShape::kVHeaddim>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsReadBlockDescriptor()
    {
        return MakeXLdsTdmBlockDescriptor<typename Problem::KDataType,
                                          Problem::BlockFmhaShape::kN0,
                                          Problem::BlockFmhaShape::kQKHeaddim>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsReadBlockDescriptor()
    {
        return MakeXLdsTdmBlockDescriptor<typename Problem::VDataType,
                                          Problem::BlockFmhaShape::kN0,
                                          Problem::BlockFmhaShape::kVHeaddim>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsReadBlockDescriptor()
    {
        return MakeXLdsTdmBlockDescriptor<typename Problem::QDataType,
                                          Problem::BlockFmhaShape::kM0,
                                          Problem::BlockFmhaShape::kQKHeaddim>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOGradLdsReadBlockDescriptor()
    {
        return MakeXLdsTdmBlockDescriptor<typename Problem::OGradDataType,
                                          Problem::BlockFmhaShape::kM0,
                                          Problem::BlockFmhaShape::kVHeaddim>();
    }

    template <typename BlockGemm>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBiasSTileDistribution()
    {
        using c_block_tensor_type = decltype(BlockGemm{}.MakeCBlockTile());
        return c_block_tensor_type::get_tile_distribution();
    }

    // TDM padding only advances the LDS destination address between rows, so
    // nothing is written past the last row and the descriptor's element space
    // is the whole allocation.
    template <typename T, index_t MNPerBlock, index_t KPerBlock>
    CK_TILE_HOST_DEVICE static constexpr index_t GetTdmBlockSmemSize()
    {
        return sizeof(T) *
               MakeXLdsTdmBlockDescriptor<T, MNPerBlock, KPerBlock>().get_element_space_size();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeQ()
    {
        return GetTdmBlockSmemSize<typename Problem::QDataType,
                                   Problem::BlockFmhaShape::kM0,
                                   Problem::BlockFmhaShape::kQKHeaddim>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeK()
    {
        return GetTdmBlockSmemSize<typename Problem::KDataType,
                                   Problem::BlockFmhaShape::kN0,
                                   Problem::BlockFmhaShape::kQKHeaddim>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeV()
    {
        return GetTdmBlockSmemSize<typename Problem::VDataType,
                                   Problem::BlockFmhaShape::kN0,
                                   Problem::BlockFmhaShape::kVHeaddim>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSizeOGrad()
    {
        return GetTdmBlockSmemSize<typename Problem::OGradDataType,
                                   Problem::BlockFmhaShape::kM0,
                                   Problem::BlockFmhaShape::kVHeaddim>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        constexpr index_t smem_size_q    = GetSmemSizeQ<Problem>();
        constexpr index_t smem_size_lse  = GetSmemSizeLSE<Problem>();
        constexpr index_t smem_size_k    = GetSmemSizeK<Problem>();
        constexpr index_t smem_size_v    = GetSmemSizeV<Problem>();
        constexpr index_t smem_size_do   = GetSmemSizeOGrad<Problem>();
        constexpr index_t smem_size_d    = GetSmemSizeD<Problem>();
        constexpr index_t smem_size_ds   = GetSmemSizeSGrad<Problem>();
        constexpr index_t smem_size_bias = GetSmemSizeBias<Problem>();

        constexpr index_t smem_size_stage0 = smem_size_k + smem_size_v;
        constexpr index_t smem_size_stage1 = smem_size_q * 2 + smem_size_do * 2 +
                                             smem_size_lse * 2 + smem_size_d * 2 +
                                             max(smem_size_bias, smem_size_ds);
        return max(smem_size_stage0, smem_size_stage1);
    }

    template <typename Problem>
    class HotLoopScheduler
    {
        static constexpr index_t kBlockSize = Problem::kBlockSize;
        static constexpr index_t kM0        = Problem::BlockFmhaShape::kM0;
        static constexpr index_t kN0        = Problem::BlockFmhaShape::kN0;
        static constexpr index_t kQKHeaddim = Problem::BlockFmhaShape::kQKHeaddim;
        static constexpr index_t kVHeaddim  = Problem::BlockFmhaShape::kVHeaddim;
        static constexpr index_t kK0        = Problem::BlockFmhaShape::kK0;
        static constexpr index_t kK2        = Problem::BlockFmhaShape::kK2;
        static constexpr index_t kK4        = Problem::BlockFmhaShape::kK4;

        static constexpr index_t WarpGemmM =
            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{});
        static constexpr index_t WarpGemmN =
            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<1>{});
        static constexpr index_t WarpGemmK =
            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<2>{});
        static constexpr index_t Gemm4MWarp =
            Problem::BlockFmhaShape::Gemm4BlockWarps::at(number<0>{});
        static constexpr index_t Gemm4NWarp =
            Problem::BlockFmhaShape::Gemm4BlockWarps::at(number<1>{});

        static constexpr index_t blockWarps = kBlockSize / get_warp_size();
        using GemmDataType                  = typename Problem::GemmDataType;

        // Compute
        static constexpr index_t Gemm0MFMA =
            kM0 * kN0 * kK0 / (blockWarps * WarpGemmM * WarpGemmN * WarpGemmK);
        static constexpr index_t Gemm1MFMA =
            kN0 * kVHeaddim * kM0 / (blockWarps * WarpGemmM * WarpGemmN * WarpGemmK);
        static constexpr index_t Gemm2MFMA =
            kM0 * kN0 * kK2 / (blockWarps * WarpGemmM * WarpGemmN * WarpGemmK);
        static constexpr index_t Gemm3MFMA =
            kN0 * kQKHeaddim * kM0 / (blockWarps * WarpGemmM * WarpGemmN * WarpGemmK);
        static constexpr index_t Gemm4MFMA =
            kM0 * kQKHeaddim * kN0 / (blockWarps * WarpGemmM * WarpGemmN * WarpGemmK);

        // VMEM
        static constexpr index_t Q_VMEM_READ =
            kM0 * kQKHeaddim / kBlockSize / GetAlignmentQ<Problem>();
        static constexpr index_t OGrad_VMEM_READ =
            kM0 * kVHeaddim / kBlockSize / GetAlignmentOGrad<Problem>();
        static constexpr index_t LSE_VMEM_READ = 1;
        static constexpr index_t D_VMEM_READ   = 1;

        static constexpr index_t DQ_VMEM_WRITE = kM0 * kQKHeaddim / kBlockSize; // atomic add

        // LDS Read
        static constexpr index_t OGradT_LDS_READ =
            kM0 * kVHeaddim / get_warp_size() / GetTransposedAlignmentOGrad<Problem>();
        static constexpr index_t QT_LDS_READ =
            kM0 * kQKHeaddim / get_warp_size() / GetTransposedAlignmentQ<Problem>();
        static constexpr index_t SGradT_LDS_READ_P1 =
            kM0 * kK4 / (get_warp_size() * Gemm4MWarp) / GetTransposedAlignmentX<GemmDataType>();
        static constexpr index_t SGradT_LDS_READ_P2 =
            kM0 * kN0 / (get_warp_size() * Gemm4MWarp) / GetTransposedAlignmentX<GemmDataType>() -
            SGradT_LDS_READ_P1;
        static constexpr index_t Q_LDS_READ =
            kM0 * kK0 / get_warp_size() / GetAlignmentQ<Problem>();
        static constexpr index_t LSE_LDS_READ = kM0 / (4 * 4);
        static constexpr index_t D_LDS_READ   = LSE_LDS_READ;
        static constexpr index_t OGrad_LDS_READ =
            kM0 * kK2 / kBlockSize / GetAlignmentOGrad<Problem>();

        // LDS Write
        static constexpr index_t Q_LDS_WRITE =
            kM0 * kQKHeaddim / Problem::kBlockSize / GetAlignmentQ<Problem>();
        static constexpr index_t QT_LDS_WRITE =
            kM0 * kQKHeaddim / kBlockSize / GetTransposedAlignmentQ<Problem>();
        static constexpr index_t OGrad_LDS_WRITE =
            kM0 * kVHeaddim / kBlockSize / GetAlignmentOGrad<Problem>();
        static constexpr index_t OGradT_LDS_WRITE =
            kM0 * kVHeaddim / kBlockSize / GetTransposedAlignmentOGrad<Problem>();
        static constexpr index_t SGradT_LDS_WRITE = kM0 * kN0 / kBlockSize;

        // Emit stream `Count`'s I-th share of barriers when walking N steps.
        // The streams are interleaved by walking max(counts) steps and giving
        // each its proportional share, so a stream of `Count` barriers has its
        // k-th barrier (k starts at 1) emitted at zero-based step
        // ceil(k * N / Count) - 1. N >= Count keeps that to at most one barrier
        // per stream per step. Counts are preserved, but positions and
        // cross-stream interleaving need not match the old lcm expansion.
        template <index_t N, index_t Count, index_t Mask, index_t I>
        CK_TILE_DEVICE static constexpr void EmitShare()
        {
            constexpr index_t lo = (I * Count) / N;
            constexpr index_t hi = ((I + 1) * Count) / N;
            static_assert(Count <= N, "step grid must be at least as long as any stream");
            if constexpr(hi > lo)
                __builtin_amdgcn_sched_group_barrier(Mask, 1, 0);
        }

        public:
        static constexpr index_t TOTAL_VMEM_READ =
            Q_VMEM_READ + OGrad_VMEM_READ + LSE_VMEM_READ + D_VMEM_READ + DQ_VMEM_WRITE;

        CK_TILE_DEVICE static constexpr void SchedulerGemm0()
        {
            // Mem: Q, LSE, OGrad, D global load, OGrad^T LDS load
            // Comp: Q x K
            constexpr index_t VMEM_READ_INST =
                Q_VMEM_READ + OGrad_VMEM_READ + LSE_VMEM_READ + D_VMEM_READ;
            constexpr index_t MFMA_INST     = Gemm0MFMA;
            constexpr index_t LDS_READ_INST = OGradT_LDS_READ + LSE_LDS_READ + D_LDS_READ;

            constexpr index_t N = max(VMEM_READ_INST, max(MFMA_INST, LDS_READ_INST));
            static_for<0, N, 1>{}([&](auto i) {
                EmitShare<N, VMEM_READ_INST, 0x020, decltype(i)::value>(); // VMEM read
                EmitShare<N, MFMA_INST, 0x008, decltype(i)::value>();      // MFMA
                EmitShare<N, LDS_READ_INST, 0x100, decltype(i)::value>();  // DS read
            });
        }

        CK_TILE_DEVICE static constexpr void SchedulerGemm12()
        {
            // Mem:  Q^T LDS load
            // Comp: PT x OGrad
            constexpr index_t LDS_READ_INST = QT_LDS_READ;
            constexpr index_t MFMA_INST     = Gemm1MFMA + Gemm2MFMA;

            constexpr index_t N = max(MFMA_INST, LDS_READ_INST);
            static_for<0, N, 1>{}([&](auto i) {
                EmitShare<N, MFMA_INST, 0x008, decltype(i)::value>();     // MFMA
                EmitShare<N, LDS_READ_INST, 0x100, decltype(i)::value>(); // DS read
            });
        }

        CK_TILE_DEVICE static constexpr void SchedulerGemm3()
        {
            // Mem: LSE/D LDS store, SGradT LDS store, SGrad, Q, LSE LDS load.
            // Comp: SGradT x QT
            constexpr index_t LDS_WRITE_INST = SGradT_LDS_WRITE;
            constexpr index_t LDS_READ_INST  = SGradT_LDS_READ_P1 + Q_LDS_READ;
            constexpr index_t MFMA_INST      = Gemm3MFMA;

            constexpr index_t lds_rw_inst = LDS_WRITE_INST + LDS_READ_INST;
            constexpr index_t N           = max(MFMA_INST, lds_rw_inst);

            static_for<0, N, 1>{}([&](auto i) {
                EmitShare<N, MFMA_INST, 0x008, decltype(i)::value>(); // MFMA
                constexpr index_t lo = (decltype(i)::value * lds_rw_inst) / N;
                constexpr index_t hi = ((decltype(i)::value + 1) * lds_rw_inst) / N;
                if constexpr(hi > lo)
                {
                    if constexpr(lo < LDS_WRITE_INST)
                        __builtin_amdgcn_sched_group_barrier(0x200, 1, 0); // DS Write
                    else
                        __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); // DS Read
                }
            });
        }

        CK_TILE_DEVICE static constexpr void SchedulerGemm4()
        {
            // Mem: SGrad, OGrad, D LDS load.
            // Comp: SGrad x KT
            constexpr index_t LDS_READ_INST = SGradT_LDS_READ_P2 + OGrad_LDS_READ;
            constexpr index_t MFMA_INST     = Gemm4MFMA;

            constexpr index_t N = max(MFMA_INST, LDS_READ_INST);
            static_for<0, N, 1>{}([&](auto i) {
                EmitShare<N, MFMA_INST, 0x008, decltype(i)::value>();     // MFMA
                EmitShare<N, LDS_READ_INST, 0x100, decltype(i)::value>(); // DS read
            });
        }
    };
};

} // namespace ck_tile
