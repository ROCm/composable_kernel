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

    using BlockShape = typename Problem::BlockShape; // this is FlatmmShape

    using ADataType            = typename Problem::ADataType;
    using BDataType            = typename Problem::GDataType;
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

    static constexpr index_t kAlignmentA = Policy::template GetAlignment_A<Problem>(); // buffer_load_dword   for A element cnt
    static constexpr index_t kAlignmentB = Policy::template GetAlignment_B<Problem>(); // buffer_load_dwordx4 for B element cnt
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
        return 64 * 1024;
        constexpr index_t smem = Policy::template GetUK<Problem>().GetSmemSize();
        constexpr index_t smem_bridge =
            BlockShape::Block_M * BlockShape::Block_N * sizeof(YDataType);
        return max(smem, smem_bridge);        
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
        constexpr index_t KLans   = BlockShape::Block_K / kAlignmentA;
        constexpr index_t MLans   = BlockShape::BlockSize / KLans;
        constexpr index_t MRepeat = BlockShape::Block_M / MLans;

        return MRepeat;
    }

    // TODO: properlly support scatter/gather
    CK_TILE_DEVICE auto GetRowCoords_A(index_t base_offset)
    {
        constexpr index_t KLans   = BlockShape::Block_K / kAlignmentA;
        constexpr index_t MLans   = BlockShape::BlockSize / KLans;
        constexpr index_t MRepeat = BlockShape::Block_M / MLans;

        auto base_coord = threadIdx.x / KLans + base_offset;

        array<index_t, MRepeat> coords;
        static_for<0, MRepeat, 1>{}([&](auto i) { coords.at(i) = base_coord + i * MLans; });
#if 0
        if(threadIdx.x == 0 && blockIdx.x == 0 && threadIdx.y == 0 && blockIdx.y == 0)
        {
            printf("[PIPE] GetRowCoords_A():\n");
            printf("[PIPE] kAlignmentA = %d, KLans = %d, MLans = %d, MRepeat = %d\n",
                static_cast<int>(kAlignmentA),  // buffer_load_dword   for A element cnt
                static_cast<int>(KLans),        // how many cols will be load once
                static_cast<int>(MLans),        // how many rows will be load once
                static_cast<int>(MRepeat));     // how many times need to load
            printf("[PIPE] coord.size() = %d\n", coords.size());
        }
#endif        
        return coords;
    }
    CK_TILE_DEVICE auto GetRowCoords_O2(index_t base_offset)
    {
        constexpr index_t NLans   = BlockShape::Block_N / kAlignmentO;
        constexpr index_t MLans   = BlockShape::BlockSize / NLans;
        constexpr index_t MRepeat = BlockShape::Block_M / MLans;

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

    template <typename ROW_IDS>
    CK_TILE_DEVICE auto GetAScale(const ROW_IDS row_ids_mma, const AScaleDataType* a_scale_ptr )
    {
        constexpr index_t n_size = row_ids_mma.size();

        array<TopkWeightDataType, n_size> w;
        static_for<0, n_size, 1>{}([&](auto i) {
            auto row_id = row_ids_mma[i] & 0xffffff;
            w.at(i) = a_scale_ptr[row_id];
            /*if (row_id >= num_tokens_)
            {
                w.at(i) = 0.f;
            } else {
                //w.at(i) = 1.f;
                // auto itp_k = row_ids_mma[i] >> 24;
                w.at(i) = a_scale_ptr[row_id];
            }*/
        });

        return w;
    }

    template <typename Karg>
    CK_TILE_DEVICE auto operator()(const Karg& kargs, CK_TILE_LDS_ADDR void* smem)
    {
#if 0
        if(threadIdx.x == 0 && blockIdx.x == 0 && threadIdx.y == 0 && blockIdx.y == 0)
        {
            printf("[PIPE] GemmPipeline_FlatmmUk =====\n");
            printf("[PIPE] GetSmemSize = %d (Byte)\n", static_cast<int>(GetSmemSize()));
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
        [[maybe_unused]] char* dbg_fp8   = static_cast<char*>(kargs.dbg_fp8_ptr);
        [[maybe_unused]] short* dbg_f16  = static_cast<short*>(kargs.dbg_f16_ptr);
        [[maybe_unused]] float* dbg_fp32 = static_cast<float*>(kargs.dbg_fp32_ptr);

        // ----------------------------------------------------------------------------
        // a
        auto a_res =
            make_wave_buffer_resource(reinterpret_cast<const ADataType*>(kargs.a_ptr),
                                      kargs.num_tokens * kargs.hidden_size * sizeof(ADataType));
        auto row_ids_a = GetRowCoords_A(blockIdx.y * BlockShape::Block_M);
        auto a_coords  = generate_tuple( // load_dwordx1 offset per thread by Byte
            [&](auto i) {
                return row_ids_a[i] * kargs.hidden_size +
                       threadIdx.x % (BlockShape::Block_K / kAlignmentA) * kAlignmentA;
            },
            number<row_ids_a.size()>{});
#if 0
        for(int i = 0; i < row_ids_a.size(); i++)
        {
            const ADataType* a_ptr = reinterpret_cast<const ADataType*>(kargs.a_ptr);
            int idx = row_ids_a[i] * kargs.hidden_size + threadIdx.x % (BlockShape::Block_K / kAlignmentA) * kAlignmentA;
            dbg_int[gid * 64  + i] = idx;
            dbg_fp32[gid * 64 + i] = type_convert<float>(a_ptr[idx]);
        }
#endif        
        // ----------------------------------------------------------------------------
        // b
        index_t nr = kargs.intermediate_size / BlockShape::Wave_N;  // divide N in W
        index_t kr = kargs.hidden_size / BlockShape::Wave_K;        // divide K in W
        index_t interm_idx_nr = __builtin_amdgcn_readfirstlane(blockIdx.x * BlockShape::Block_Nr); // current block base row idx

        auto b_win = [&]() {
            const BDataType* b_ptr = reinterpret_cast<const BDataType*>(kargs.b_ptr) +
                                     interm_idx_nr * kr * BlockShape::Block_W;
            auto b_view_ = make_naive_tensor_view<address_space_enum::global>(
                b_ptr,
                make_tuple(nr, kr, number<BlockShape::Block_W>{}),
                make_tuple(kr * BlockShape::Block_W, number<BlockShape::Block_W>{}, 1),
                number<kAlignmentB>{},
                number<1>{});

            auto b_window_ = make_tile_window_linear_raw(
                b_view_,
                make_tuple(number<BlockShape::Block_Nr>{},
                           number<BlockShape::Block_Kr>{},
                           number<BlockShape::Block_W>{}),
                {0, 0, 0},
                Policy::template MakeGlobalTileDistribution_G<Problem>(),
                sequence<0, 1, 1>{});
            return b_window_;
        }();
        auto b_res    = b_win.get_bottom_tensor_view().get_buffer_view().cached_buf_res_;
        auto b_coords = generate_tuple([&](auto i) { return b_win.cached_coords_[i].get_offset(); },
                                       number<decltype(b_win)::NumAccess_NonLinear>{});

#if 0
        if(threadIdx.x == 0 && blockIdx.x == 0 && threadIdx.y == 0 && blockIdx.y == 0)
        {
            printf("[PIPE] b_win: nr = %d, kr = %d, interm_idx_nr = %d, cached_coords_.size() = %d\n", 
                nr, kr, interm_idx_nr,
                b_win.cached_coords_.size());
        }
        dbg_int[gid * 64  + 0] = nr;
        dbg_int[gid * 64  + 1] = kr;
        dbg_int[gid * 64  + 2] = interm_idx_nr;
        dbg_int[gid * 64  + 3] = interm_idx_nr * kr * BlockShape::Block_W;
        dbg_int[gid * 64  + 4] = b_win.cached_coords_[0].get_offset();
        dbg_int[gid * 64  + 5] = b_win.cached_coords_[1].get_offset();
#endif        

        // ----------------------------------------------------------------------------
        // a_scale
        [[maybe_unused]] const float* sa_ptr = static_cast<const float*>(kargs.sa_ptr);
        int saLaneInBlk = threadIdx.x % BlockShape::Wave_M;
        int saBlk = blockIdx.y;
        int saIdx0 = saBlk * BlockShape::Block_M + saLaneInBlk;
        int saIdx1 = saBlk * BlockShape::Block_M + BlockShape::Wave_M + saLaneInBlk;
        array<TopkWeightDataType, BlockShape::Repeat_M> a_scale;
        a_scale[0] = sa_ptr[saIdx0];
        a_scale[1] = sa_ptr[saIdx1];

        // ----------------------------------------------------------------------------
        // b_scale
        [[maybe_unused]] const float* sb_ptr = static_cast<const float*>(kargs.sb_ptr);
        const int acc4xyzw = 4;
        int sbLaneInBlk = threadIdx.x / BlockShape::Wave_N * acc4xyzw; // t0~255 -> 0:63
        int sbBlk = blockIdx.x;
        int sbIdx0 = sbBlk * BlockShape::Block_N + sbLaneInBlk + 0;
        //int sbIdx1 = sbBlk * BlockShape::Block_N + sbLaneInBlk + 1;
        //int sbIdx2 = sbBlk * BlockShape::Block_N + sbLaneInBlk + 2;
        //int sbIdx3 = sbBlk * BlockShape::Block_N + sbLaneInBlk + 3;
        int sbIdxStep = blockDim.x / BlockShape::Wave_N * acc4xyzw;

        /*dbg_fp32[gid * 64 + 0] = BlockShape::Wave_N * 1.0f;
        dbg_fp32[gid * 64 + 1] = sbLaneInBlk * 1.0f;
        dbg_fp32[gid * 64 + 2] = sbBlk * 1.0f;
        dbg_fp32[gid * 64 + 3] = sbIdx0 * 1.0f;
        dbg_fp32[gid * 64 + 4] = sbIdx1 * 1.0f;
        dbg_fp32[gid * 64 + 5] = sbIdx2 * 1.0f;
        dbg_fp32[gid * 64 + 6] = sbIdx3 * 1.0f;
        dbg_fp32[gid * 64 + 7] = sbIdxStep * 1.0f;
        for(int dbgi = 0; dbgi < 8; dbgi++)
        {
            for(int wyzw = 0; wyzw < 4; wyzw++)
            {
                dbg_fp32[gid * 64 + dbgi * 4 + 0] = sb_ptr[sbIdx0 + dbgi * sbIdxStep];
                dbg_fp32[gid * 64 + dbgi * 4 + 1] = sb_ptr[sbIdx1 + dbgi * sbIdxStep];
                dbg_fp32[gid * 64 + dbgi * 4 + 2] = sb_ptr[sbIdx2 + dbgi * sbIdxStep];
                dbg_fp32[gid * 64 + dbgi * 4 + 3] = sb_ptr[sbIdx3 + dbgi * sbIdxStep];
            }
        }*/

        // ----------------------------------------------------------------------------
        // core
        auto uk = Policy::template GetUK<Problem>();
        auto acc =
            uk(a_res,
               a_coords,
               b_res,
               b_coords,
               a_scale,
               sb_ptr,
               sbIdx0,
               sbIdxStep,
               smem,
               kargs.hidden_size,
               BlockShape::Block_K,                        // tile offset for B matrix each unroll
               BlockShape::Block_Kr * BlockShape::Block_W, // tile offset for B matrix each unroll
               dbg_int,
               dbg_fp8,
               dbg_f16,
               dbg_fp32);

        // ----------------------------------------------------------------------------
#if 1        
        {
            int tid           = threadIdx.x;
            ODataType srdfp16    = 0.f;
            ODataType* smemfp16  = static_cast<ODataType*>(smem);
            const float * accfp32 = static_cast<const float*>(&(acc.get_thread_buffer()[0]));

            // ----------------------------------------------------------------------------
            // transpose in lds: acc4MCnt(row cnt) * acc4NCnt(col cnt) -> acc4NCnt(row cnt) * acc4MCnt(col cnt)
            const int accXyzw        = 4;
            const int acc4MCnt       = BlockShape::Wave_M;
            const int acc4NCnt       = BlockShape::Wave_N;
            const int acc4CntPerThd  = BlockShape::Repeat_M * BlockShape::Repeat_N;
            const int accLdsPadByte = 4 * 4; // 4DWord
            const int acc4NCntWithPad = acc4NCnt + accLdsPadByte / (4 * sizeof(ODataType)); // BlockShape::Wave_M
            const int accSmemBlkSz = accXyzw * acc4MCnt * acc4NCntWithPad;
            const int d4BlkMCnt      = BlockShape::Repeat_M;
            const int d4RowCntPerBlk = BlockShape::Wave_M;
            const int d4ColCntPerBlk = BlockShape::Wave_N;

            int acc4_ori_col_in_blk   = tid % acc4NCnt; // row id in origin blk
            int acc4_ori_row_in_blk   = tid / acc4NCnt; // col id in origin blk
            int acc4_trans_col_in_blk = tid / acc4MCnt;
            int acc4_trans_row_in_blk = tid % acc4MCnt;

            // ----------------------------------------------------------------------------
            // store to lds
            for(uint32_t acc4Idx = 0; acc4Idx < acc4CntPerThd; acc4Idx++)
            {                
                ODataType* accBlkSmem = smemfp16 + acc4Idx * accSmemBlkSz;
                int acc4_id_in_blk = acc4_ori_row_in_blk * acc4NCntWithPad + acc4_ori_col_in_blk;

                float t;
                array<ODataType, accXyzw> accfp16_;
                t = accfp32[accXyzw * acc4Idx + 0];
                accfp16_[0] = type_convert<ODataType>(t);
                t = accfp32[accXyzw * acc4Idx + 1];
                accfp16_[1] = type_convert<ODataType>(t);
                t = accfp32[accXyzw * acc4Idx + 2];
                accfp16_[2] = type_convert<ODataType>(t);
                t = accfp32[accXyzw * acc4Idx + 3];
                accfp16_[3] = type_convert<ODataType>(t);

                float * accBlkSmem_ = reinterpret_cast<float*>(&accBlkSmem[acc4_id_in_blk * accXyzw]);
                float * accfp32_ = reinterpret_cast<float*>(&accfp16_[0]);
                accBlkSmem_[0] = accfp32_[0];
                accBlkSmem_[1] = accfp32_[1];

                /*for(int xyzw = 0; xyzw < accXyzw; xyzw++)
                {
                    float srdfp32 = accfp32[accXyzw * acc4Idx + xyzw];
                    srdfp16 = type_convert<ODataType>(srdfp32);
                    accBlkSmem[acc4_id_in_blk * accXyzw + xyzw] = srdfp16;

                    //dbg_fp32[gid * 64 + accXyzw * acc4Idx + xyzw] = srdfp32;
                    //dbg_f16[gid * 64 + accXyzw * acc4Idx + xyzw] = srdfp16;
                    //dbg_int [gid * 64 + accXyzw * acc4Idx + xyzw] = (accXyzw * acc4Idx * blockDim.x) + (tid * accXyzw + xyzw);
                }*/
            }
            block_sync_lds();

            // ----------------------------------------------------------------------------
            // read from lds, store to vmem
            ODataType* d_b16_buf = static_cast<ODataType*>(kargs.d_f16_ptr);
            ODataType* d_grp_mem = d_b16_buf +
                                blockIdx.y * BlockShape::Block_M * kargs.intermediate_size +
                                blockIdx.x * BlockShape::Block_N;

            for(uint32_t acc4Idx = 0; acc4Idx < acc4CntPerThd; acc4Idx++)
            {
                ODataType* accBlkSmem = smemfp16 + acc4Idx * accSmemBlkSz;
                int acc4_id_in_blk = acc4_trans_row_in_blk * acc4NCntWithPad + acc4_trans_col_in_blk;

                int d4_blk_row_id = acc4Idx % d4BlkMCnt;
                int d4_blk_col_id = acc4Idx / d4BlkMCnt;

                int d4_row_in_blk = tid / acc4MCnt;
                int d4_col_in_blk = tid % acc4MCnt;
                
                int d4_row = d4RowCntPerBlk * d4_blk_row_id + d4_row_in_blk;
                int d4_col = d4ColCntPerBlk * d4_blk_col_id + d4_col_in_blk;
                int d4_offset = d4_row * kargs.intermediate_size / accXyzw + d4_col;
                
                for(int xyzw = 0; xyzw < accXyzw; xyzw++)
                {
                    srdfp16 = accBlkSmem[acc4_id_in_blk * accXyzw + xyzw];
                    d_grp_mem[d4_offset * accXyzw + xyzw] = srdfp16;
                    
                    //dbg_f16[gid * 64 + accXyzw * acc4Idx + xyzw] = srdfp16;
                    //dbg_int [gid * 64 + accXyzw * acc4Idx + xyzw] = (d4_blk_row_id * acc4BlkNCnt + d4_blk_col_id);
                }
            } 
        }
#else
        (void)acc;        
#endif        
    }
};

} // namespace ck_tile
