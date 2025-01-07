// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/fused_moe/pipeline/flatmm_uk_pipeline_policy.hpp"

namespace ck_tile {

/*
This pipeline deal with a gemm(actually 2 gemm) with one very small(token), one very big(weight)
we need to design the pipeline such that all waves along gemm-N dim (gemm-m only 1 wave)

    <----- gemm-N ------>
    +----+----+----+----+
    | w0 | w1 | w2 | w3 | gemm-m
    +----+----+----+----+
*/
template <typename Problem_, typename Policy_ = GemmPipelineFlatmmPolicy>
struct GemmPipeline_FlatmmUk
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    using BlockShape = typename Problem::BlockShape; // this is FusedMoeGemmShape

    using ADataType            = typename Problem::ADataType;
    using GDataType            = typename Problem::GDataType;
    using DDataType            = typename Problem::AccDataType;
    using AccDataType          = typename Problem::AccDataType;
    using ODataType            = typename Problem::ODataType;
    using AScaleDataType       = typename Problem::AScaleDataType;
    using GScaleDataType       = typename Problem::GScaleDataType;
    using DScaleDataType       = typename Problem::DScaleDataType;
    using YSmoothScaleDataType = typename Problem::YSmoothScaleDataType;
    using TopkWeightDataType   = typename Problem::TopkWeightDataType;
    using IndexDataType        = typename Problem::IndexDataType;
    using YDataType            = typename Problem::YDataType;

    using Traits = typename Problem::Traits;

    static constexpr bool IsGateOnly          = Traits::IsGateOnly;
    static constexpr bool UseSmoothQuant      = Traits::UseSmoothQuant;
    static constexpr bool PadHiddenSize       = Traits::PadHiddenSize;
    static constexpr bool PadIntermediateSize = Traits::PadIntermediateSize;

    static constexpr index_t kAlignmentA = Policy::template GetAlignment_A<Problem>();
    static constexpr index_t kAlignmentG = Policy::template GetAlignment_G<Problem>();
    static constexpr index_t kAlignmentD = Policy::template GetAlignment_D<Problem>();
    static constexpr index_t kAlignmentO = Policy::template GetAlignment_O<Problem>();

    static constexpr index_t SLD_A = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::SLD_A);
    static constexpr index_t GLD_A = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::GLD_A);
    static constexpr index_t GLD_B = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::GLD_B);
    static constexpr index_t GST_O = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::GST_O);

    static constexpr index_t kBlockPerCu = []() {
        if constexpr(Problem::kBlockPerCu != -1)
            return Problem::kBlockPerCu;
        else
        {
            // minimize occupancy
            return 2;
        }
    }();

    static constexpr const char* name = "flatmm_uk";

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        constexpr index_t smem_0 = Policy::template GetUK_0<Problem>().GetSmemSize();
        constexpr index_t smem_bridge =
            BlockShape::Block_M0 * BlockShape::Block_N0 * sizeof(YDataType);
        return max(smem_0, smem_bridge);
    }

    // this is the thread-offset along row/col
    CK_TILE_HOST_DEVICE static auto GetACoord()
    {
        constexpr auto a_dist = Policy::template MakeGlobalTileDistribution_A<Problem>();
        const auto a_coord    = a_dist.calculate_index();
        return a_coord;
    }

    // this is the thread-offset along row/col
    CK_TILE_HOST_DEVICE static auto GetOCoord()
    {
        constexpr auto o_dist = Policy::template MakeOGlobalTileDistribution<Problem>();
        const auto o_coord    = o_dist.calculate_index();
        return o_coord;
    }

    CK_TILE_DEVICE constexpr auto GetNumRowCoords_A()
    {
        constexpr index_t KLans   = BlockShape::Block_K0 / kAlignmentA;
        constexpr index_t MLans   = BlockShape::BlockSize / KLans;
        constexpr index_t MRepeat = BlockShape::Block_M0 / MLans;

        return MRepeat;
    }

    // TODO: properlly support scatter/gather
    CK_TILE_DEVICE auto GetRowCoords_A(index_t base_offset)
    {
        constexpr index_t KLans   = BlockShape::Block_K0 / kAlignmentA;
        constexpr index_t MLans   = BlockShape::BlockSize / KLans;
        constexpr index_t MRepeat = BlockShape::Block_M0 / MLans;

        auto base_coord = threadIdx.x / KLans + base_offset;

        array<index_t, MRepeat> coords;
        static_for<0, MRepeat, 1>{}([&](auto i) { coords.at(i) = base_coord + i * MLans; });

        return coords;
    }
    CK_TILE_DEVICE auto GetRowCoords_O2(index_t base_offset)
    {
        constexpr index_t NLans   = BlockShape::Block_N0 / kAlignmentO;
        constexpr index_t MLans   = BlockShape::BlockSize / NLans;
        constexpr index_t MRepeat = BlockShape::Block_M0 / MLans;

        auto base_coord = threadIdx.x / NLans + base_offset;

        array<index_t, MRepeat> coords;
        static_for<0, MRepeat, 1>{}([&](auto i) { coords.at(i) = base_coord + i * MLans; });

        return coords;
    }

    template <typename ROW_COORDS>
    CK_TILE_DEVICE auto GetRowID(const ROW_COORDS coords, const IndexDataType* sorted_token_ids_ptr)
    {
        constexpr index_t n_size = coords.size();

        array<index_t, n_size> row_ids;
        static_for<0, n_size, 1>{}([&](auto i) {
            row_ids.at(i) = sorted_token_ids_ptr[coords[i]]; // base_coord + i * MLans;
        });

        return row_ids;
    }

    template <typename ROW_COORDS>
    CK_TILE_DEVICE auto GetWeightScale(const ROW_COORDS coords,
                                       const TopkWeightDataType* sorted_weight_ptr)
    {
        constexpr index_t n_size = coords.size();

        array<TopkWeightDataType, n_size> w;
        static_for<0, n_size, 1>{}([&](auto i) {
            w.at(i) = sorted_weight_ptr[coords[i]]; // base_coord + i * MLans;
        });

        return w;
    }

    // TODO: this row id is before shuffle atomic, need use acc distribution
    CK_TILE_DEVICE auto GetRowCoords_O(index_t base_offset)
    {
        constexpr index_t MLanes   = BlockShape::Warp_M1;
        constexpr index_t Repeat_M = BlockShape::Repeat_M1;

        auto base_coord = threadIdx.x % MLanes + base_offset;

        array<index_t, Repeat_M> coords;
        static_for<0, Repeat_M, 1>{}([&](auto i) { coords.at(i) = base_coord + i * MLanes; });

        return coords;
    }

    template <typename Karg>
    CK_TILE_DEVICE auto operator()(const Karg& kargs, CK_TILE_LDS_ADDR void* smem)
    {
#if 0
        if(threadIdx.x == 0 && blockIdx.x == 0 && threadIdx.y == 0 && blockIdx.y == 0)
        {
            printf("[PIPE] GemmPipeline_FlatmmUk =====\n");
        }

        [[maybe_unused]] uint32_t tidx = threadIdx.x; // 0~255
        [[maybe_unused]] uint32_t tidy = threadIdx.y; // 0~0
        [[maybe_unused]] uint32_t bidx = blockIdx.x;  // 0~1
        [[maybe_unused]] uint32_t bidy = blockIdx.y;  // 0~51
        [[maybe_unused]] uint32_t bdmx = blockDim.x;  // 256
        [[maybe_unused]] uint32_t bdmy = blockDim.y;  // 1
        [[maybe_unused]] uint32_t gdmx = gridDim.x;   // 2
        [[maybe_unused]] uint32_t gdmy = gridDim.y; // 52
        [[maybe_unused]] uint32_t gid = ((bdmx * bdmy) * gdmx) * bidy 
                                        + (bdmx * bdmy) * bidx 
                                        + bdmx * tidy
                                        + tidx;
#endif
        [[maybe_unused]] int* dbg_int    = static_cast<int*>(kargs.dbg_int_ptr);
        [[maybe_unused]] short* dbg_bf16 = static_cast<short*>(kargs.dbg_bf16_ptr);
        [[maybe_unused]] float* dbg_fp32 = static_cast<float*>(kargs.dbg_fp32_ptr);

        ck_tile::index_t shared_intermediate_size_0 = kargs.intermediate_size;     // N
        index_t nr_0           = shared_intermediate_size_0 / BlockShape::Warp_N0; // divide N in W
        index_t kr_0           = kargs.hidden_size / BlockShape::Warp_K0;          // divide K in W
        index_t interm_idx_nr0 = __builtin_amdgcn_readfirstlane(
            blockIdx.x * BlockShape::Block_Nr0); // intermediate_tile_id * Block_N / (N in W)

        // ----------------------------------------------------------------------------
        // a
        auto a_res =
            make_wave_buffer_resource(reinterpret_cast<const ADataType*>(kargs.a_ptr),
                                      kargs.num_tokens * kargs.hidden_size * sizeof(ADataType));
        auto row_ids_a = GetRowCoords_A(blockIdx.y * BlockShape::Block_M0);
        auto a_coords  = generate_tuple(
            [&](auto i) {
                return row_ids_a[i] * kargs.hidden_size +
                       threadIdx.x % (BlockShape::Block_K0 / kAlignmentA) * kAlignmentA;
            },
            number<row_ids_a.size()>{});

        // ----------------------------------------------------------------------------
        // b
        auto b_win = [&]() {
            const GDataType* b_ptr = reinterpret_cast<const GDataType*>(kargs.b_ptr) +
                                     interm_idx_nr0 * kr_0 * BlockShape::Block_W0;
            auto b_view_ = make_naive_tensor_view<address_space_enum::global>(
                b_ptr,
                make_tuple(nr_0, kr_0, number<BlockShape::Block_W0>{}),
                make_tuple(kr_0 * BlockShape::Block_W0, number<BlockShape::Block_W0>{}, 1),
                number<kAlignmentG>{},
                number<1>{});

            auto b_window_ = make_tile_window_linear_raw(
                b_view_,
                make_tuple(number<BlockShape::Block_Nr0>{},
                           number<BlockShape::Block_Kr0>{},
                           number<BlockShape::Block_W0>{}),
                {0, 0, 0},
                Policy::template MakeGlobalTileDistribution_G<Problem>(),
                sequence<0, 1, 1>{});
            return b_window_;
        }();
        auto b_res    = b_win.get_bottom_tensor_view().get_buffer_view().cached_buf_res_;
        auto b_coords = generate_tuple([&](auto i) { return b_win.cached_coords_[i].get_offset(); },
                                       number<decltype(b_win)::NumAccess_NonLinear>{});

        // ----------------------------------------------------------------------------
        // core
        auto uk_0  = Policy::template GetUK_0<Problem>();
        auto acc_0 = uk_0(a_res,
                          a_coords,
                          b_res,
                          b_coords,
                          smem,
                          kargs.hidden_size,
                          BlockShape::Block_K0, // tile offset for B matrix each unroll
                          BlockShape::Block_Kr0 *
                              BlockShape::Block_W0, // tile offset for B matrix each unroll
                          dbg_int,
                          dbg_bf16,
                          dbg_fp32);

        // ----------------------------------------------------------------------------
        {
            int tid         = threadIdx.x;
            float srdfp32   = 0.f;
            float* smemfp32 = static_cast<float*>(smem);

            // ----------------------------------------------------------------------------
            // store to lds
            for(uint32_t accIdx = 0; accIdx < 16; accIdx++)
            {
                float* accSmem = smemfp32 + 4 * blockDim.x * accIdx;
                for(int xyzw = 0; xyzw < 4; xyzw++)
                {
                    accSmem[tid * 4 + xyzw] = acc_0.get_thread_buffer()[accIdx * 4 + xyzw];
                }
            }
            block_sync_lds();

            // ----------------------------------------------------------------------------
            // read from lds
            int sldIdx = 0;
            // int MLn = 15;
            // int Nln = tid / MLn;
            int tidInWave = tid % 64;
            int waveId    = tid / 64;
            // sldIdx = (tid64 % 16 * 16 + tid64 / 16) % 64
            //   + tid / 64;
            sldIdx = (tidInWave % 16 * 16 + tidInWave / 16) + waveId * 4;

            const int accNLane   = 16;
            const int NLaneCnt   = BlockShape::Block_N0 / 4; // xyzw 512 / 4 = 128
            const int accBlkSize = blockDim.x;

            int accInnerId  = tid % accNLane;        // 0~15
            int accNIdx     = tid / NLaneCnt;        // 0~127 = 0; 128~255 = 1
            int acc01BlkIdx = tid % NLaneCnt / 16;   // 0 ~ 7
            int accBlkIdx   = acc01BlkIdx * 2;       // 0, 2, 4, ..., 14
            int acc4Id      = accBlkIdx * accBlkSize //
                         + accNIdx * accBlkSize + accInnerId * 16;
            sldIdx = acc4Id;

            float* d_buf     = static_cast<float*>(kargs.d_ptr);
            int c_blk_offset = blockIdx.y * BlockShape::Block_M0 * kargs.intermediate_size / 4 +
                               blockIdx.x * BlockShape::Block_N0 / 4;

            for(uint32_t accIdx = 0; accIdx < 16; accIdx++)
            {
                for(int xyzw = 0; xyzw < 4; xyzw++)
                {
                    srdfp32 = smemfp32[accIdx * (1 * 4) + sldIdx * 4 + xyzw];
                    acc_0.get_thread_buffer()[accIdx * 4 + xyzw] = srdfp32;
                }

                // ----------------------------------------------------------------------------
                // store to vmem
                int c_m_idx_offset = (accIdx + accNIdx * 16) * kargs.intermediate_size / 4;
                int c_idx_offset   = c_blk_offset + c_m_idx_offset + (tid % NLaneCnt);

                for(int xyzw = 0; xyzw < 4; xyzw++)
                {
                    srdfp32                        = acc_0.get_thread_buffer()[accIdx * 4 + xyzw];
                    d_buf[c_idx_offset * 4 + xyzw] = srdfp32;
                }
            }
        }

#if 0
        // ----------------------------------------------------------------------------
        // debug
        for(uint32_t dbgi = 0; dbgi < 64; dbgi++)
        {
            dbg_fp32[gid * 64 + dbgi] = acc_0.get_thread_buffer()[dbgi];
        }
#endif
    }
};

} // namespace ck_tile
