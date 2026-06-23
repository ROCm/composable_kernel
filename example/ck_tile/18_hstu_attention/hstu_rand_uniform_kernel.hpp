// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/fmha/block/block_dropout.hpp>
#include <ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2.hpp>
#include <ck_tile/ops/gemm/block/block_gemm_problem.hpp>

template <typename RandValOutputDataType, bool kIsJagged>
struct HstuRandUniformKernel
{
    // M/N tile size should be a multiplier of 32
    static constexpr ck_tile::index_t kMPerBlock = 128;
    static constexpr ck_tile::index_t kNPerBlock = 64;

    using BlockTile = ck_tile::sequence<kMPerBlock, kNPerBlock, 32>;
#if defined(__gfx11__) || defined(__gfx12__)
    using WarpTile = ck_tile::sequence<16, 16, 16>;
#else
    // either 32x32 or 16x16 warp-gemm is ok for wave64
    using WarpTile = ck_tile::sequence<32, 32, 8>;
#endif
    using BlockWarps = ck_tile::sequence<4, 1, 1>;

    using BlockGemmTileShape = ck_tile::TileGemmShape<BlockTile, BlockWarps, WarpTile>;

    static constexpr ck_tile::index_t kBlockSize =
        BlockGemmTileShape::NumWarps * ck_tile::get_warp_size();
    static constexpr ck_tile::index_t kBlockPerCu = 1;

    __device__ static constexpr auto GetBlockGemm()
    {
        using namespace ck_tile;

        using BlockGemmProblem_ = ck_tile::BlockGemmProblem<ck_tile::fp16_t,
                                                            ck_tile::fp16_t,
                                                            float,
                                                            kBlockSize,
                                                            BlockGemmTileShape>;

        auto warp_gemm = ck_tile::WarpGemmDispatcher<ck_tile::fp16_t,
                                                     ck_tile::fp16_t,
                                                     float,
                                                     WarpTile::at(number<0>{}),
                                                     WarpTile::at(number<1>{}),
                                                     WarpTile::at(number<2>{}),
                                                     false,
                                                     false,
                                                     false>{};

        using BlockGemmPolicy = BlockGemmARegBSmemCRegV2CustomPolicy<ck_tile::fp16_t,
                                                                     ck_tile::fp16_t,
                                                                     float,
                                                                     BlockWarps,
                                                                     decltype(warp_gemm)>;

        return ck_tile::BlockGemmARegBSmemCRegV2<BlockGemmProblem_, BlockGemmPolicy>{};
    };

    using BlockGemm = decltype(GetBlockGemm());

    using MyBlockDropout = ck_tile::BlockDropout;

    static constexpr bool kPadSeqLenQ = true;
    static constexpr bool kPadSeqLenK = true;

    // kargs use aggregate initializer, so no constructor will provided
    // use inheritance to minimize karg size
    // user need to use MakeKargs() function to create kargs.
    struct HstuRandUniformCommonKargs
    {
        void* rand_val_ptr;

        ck_tile::index_t seqlen_q;
        ck_tile::index_t seqlen_k;

        ck_tile::index_t num_heads;
        ck_tile::index_t num_batches;

        ck_tile::index_t stride_seqlen_q;

        ck_tile::index_t stride_nhead;

        uint64_t seed   = 1;
        uint64_t offset = 0;
    };

    struct HstuRandUniformBatchedKargs : HstuRandUniformCommonKargs
    {
        ck_tile::index_t stride_batch;
    };

    struct HstuRandUniformJaggedKargs : HstuRandUniformCommonKargs
    {
        const int32_t* seqstart_q_ptr;
    };

    using Kargs =
        std::conditional_t<kIsJagged, HstuRandUniformJaggedKargs, HstuRandUniformBatchedKargs>;

    template <bool Cond = !kIsJagged>
    __host__ static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(void* rand_val_ptr,
              ck_tile::index_t seqlen_q,
              ck_tile::index_t seqlen_k,
              ck_tile::index_t num_heads,
              ck_tile::index_t num_batches,
              ck_tile::index_t stride_seqlen_q,
              ck_tile::index_t stride_nhead,
              ck_tile::index_t stride_batch,
              std::tuple<uint64_t, uint64_t> drop_seed_offset)
    {
        Kargs kargs{{rand_val_ptr,
                     seqlen_q,
                     seqlen_k,
                     num_heads,
                     num_batches,
                     stride_seqlen_q,
                     stride_nhead,
                     std::get<0>(drop_seed_offset),
                     std::get<1>(drop_seed_offset)},
                    stride_batch};

        return kargs;
    }

    template <bool Cond = kIsJagged>
    __host__ static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(void* rand_val_ptr,
              ck_tile::index_t num_heads,
              ck_tile::index_t num_batches,
              ck_tile::index_t stride_seqlen_q,
              ck_tile::index_t stride_nhead,
              const void* seqstart_q_ptr,
              const void* seqstart_k_ptr,
              std::tuple<uint64_t, uint64_t> drop_seed_offset)
    {
        Kargs kargs{{rand_val_ptr,
                     -1,  // seqlen_q will be update in the kernel
                     -1, // seqlen_k will be update in the kernel
                     num_heads,
                     num_batches,
                     stride_seqlen_q,
                     stride_nhead,
                     std::get<0>(drop_seed_offset),
                     std::get<1>(drop_seed_offset)},
                    reinterpret_cast<const int32_t*>(seqstart_q_ptr),
                    reinterpret_cast<const int32_t*>(seqstart_q_ptr)};

        return kargs;
    }

    __host__ static constexpr auto GridSize(ck_tile::index_t batch_size_,
                                            ck_tile::index_t nhead_,
                                            ck_tile::index_t seqlen_q_,
                                            ck_tile::index_t seqlen_k_)
    {
        (void)seqlen_k_; // not used at present

        // at present, seqlen_k is not splitted by thread-groups
        return dim3(ck_tile::integer_divide_ceil(seqlen_q_, kMPerBlock), nhead_, batch_size_);
    }

    __device__ static constexpr auto GetTileIndex(ck_tile::index_t seqlen_q_,
                                                  ck_tile::index_t seqlen_k_)
    {
        (void)seqlen_q_; // not used at present
        (void)seqlen_k_; // not used at present

        const ck_tile::index_t i_block = blockIdx.x;
        const ck_tile::index_t i_nhead = blockIdx.y;
        const ck_tile::index_t i_batch = blockIdx.z;

        return ck_tile::make_tuple(i_block, i_nhead, i_batch);
    }

    __host__ static constexpr auto BlockSize()
    {
        if(ck_tile::is_wave32())
            return dim3(kBlockSize / ck_tile::get_warp_size() * 32);
        else
            return dim3(kBlockSize);
    }

    __device__ static constexpr ck_tile::index_t GetSmemSize()
    {
        return MyBlockDropout::MakeRandValLdsBlockDescriptor<BlockGemm>().get_element_space_size();
    }

    template <typename RandValDramBlockWindowTmp>
    __device__ void main_loop(const MyBlockDropout& dropout,
                              void* randval_smem_ptr,
                              const ck_tile::index_t num_total_loop,
                              RandValDramBlockWindowTmp& randval_dram_block_window_tmp) const
    {
        auto randval_dram_window =
            MyBlockDropout::MakeRandvalDramWindow<BlockGemm>(randval_dram_block_window_tmp, 0);

        ck_tile::index_t i_total_loops = 0;

        auto null_tile_window = ck_tile::make_null_tile_window(ck_tile::make_tuple());

        do
        {
            auto seq_offset = i_total_loops * kNPerBlock;

            // randval_dram_window is moved inside BlockDropout::Run()
            dropout.template Run<BlockGemm, float, RandValOutputDataType>(
                reinterpret_cast<char*>(randval_smem_ptr),
                seq_offset,
                null_tile_window,
                randval_dram_window);

        } while(++i_total_loops < num_total_loop);
    }

    __device__ void operator()(Kargs kargs) const
    {
        using namespace ck_tile;

        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        // divide problem
        const auto [i_tile_m, i_nhead, i_batch] = GetTileIndex(kargs.seqlen_q, kargs.seqlen_k);

        const index_t i_m0 = __builtin_amdgcn_readfirstlane(i_tile_m * kMPerBlock);

        long_index_t batch_offset_randval = 0;

        if constexpr(kIsJagged)
        {
            // get starting offset for each batch
            const long_index_t query_start = kargs.seqstart_q_ptr[i_batch];

            batch_offset_randval = query_start * kargs.stride_seqlen_q;

            // get real # queries & # keys under group mode
            const auto adjusted_seqstart_q_ptr = kargs.seqstart_q_ptr + i_batch;
            kargs.seqlen_q = adjusted_seqstart_q_ptr[1] - adjusted_seqstart_q_ptr[0];

            if(kargs.seqlen_q <= i_m0)
            {
                return;
            }

            const auto adjusted_seqstart_k_ptr = kargs.seqstart_k_ptr + i_batch;
            kargs.seqlen_k = adjusted_seqstart_k_ptr[1] - adjusted_seqstart_k_ptr[0];
        }
        else
        {
            batch_offset_randval = static_cast<long_index_t>(i_batch) * kargs.stride_batch;
        }

        constexpr auto randval_dram_window_lengths =
            make_tuple(number<kMPerBlock>{}, number<kNPerBlock>{});

        RandValOutputDataType* rand_val_ptr =
            reinterpret_cast<RandValOutputDataType*>(kargs.rand_val_ptr) +
            static_cast<long_index_t>(i_nhead) * kargs.stride_nhead + batch_offset_randval;

        const auto randval_dram = [&]() {
            const auto randval_dram_naive = make_naive_tensor_view<address_space_enum::global>(
                rand_val_ptr,
                make_tuple(kargs.seqlen_q, kargs.seqlen_k),
                make_tuple(kargs.stride_seqlen_q, 1),
                number<1>{},
                number<1>{});

            return pad_tensor_view(randval_dram_naive,
                                   randval_dram_window_lengths,
                                   ck_tile::sequence<kPadSeqLenQ, kPadSeqLenK>{});
        }();

        auto randval_dram_block_window_tmp =
            make_tile_window(randval_dram, randval_dram_window_lengths, {i_m0, 0});

        MyBlockDropout dropout(i_batch,
                               i_nhead,
                               kargs.num_heads,
                               kargs.seed,
                               kargs.offset,
                               0.0f /*rp_undrop_, not used*/,
                               0 /*p_undrop_in_uint8_t, not used*/,
                               true);

        const auto num_total_loop = ck_tile::integer_divide_ceil(kargs.seqlen_k, kNPerBlock);
        main_loop(dropout, smem_ptr, num_total_loop, randval_dram_block_window_tmp);
    }
};
