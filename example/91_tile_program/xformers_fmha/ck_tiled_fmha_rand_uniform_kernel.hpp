// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck/ck.hpp>
#include <ck/tensor/tensor_view.hpp>
#include <ck/tile_program/tile/tile_gemm_shape.hpp>
#include <ck/tile_program/block_tile/block_gemm_problem.hpp>
#include <ck/tile_program/tile/tile_window.hpp>
#include <ck/tile_program/block_tile/block_gemm_areg_bsmem_creg_v2.hpp>
#include <ck/tile_program/block_tile/block_dropout.hpp>

#include <ck/tile_program/tile/static_distributed_tensor.hpp>
#include <ck/tile_program/tile/distributed_tile_sweep.hpp>
#include <ck/tile_program/tile/load_tile.hpp>
#include <ck/tile_program/tile/store_tile.hpp>

template <ck::index_t MPerBlockTile,
          ck::index_t NPerBlockTile,
          ck::index_t KPerBlockTile,
          typename RandValOutputDataType,
          bool kIsGroupMode>
struct FmhaRandUniformKernel
{
    static constexpr ck::index_t kBlockSize  = 256;
    static constexpr ck::index_t kBlockPerCu = 1;

    __device__ static constexpr auto GetBlockGemm()
    {
        using namespace ck::tile_program;
        using namespace ck::tile_program::block;

        using BlockGemmProblem_ =
            BlockGemmProblem<ck::half_t,
                             ck::half_t,
                             float,
                             kBlockSize,
                             TileGemmShape<MPerBlockTile, NPerBlockTile, KPerBlockTile>>;

        // using the default policy, which use M32xN32xK8 warp_tile
        return BlockGemmARegBSmemCRegV2<BlockGemmProblem_>{};
    };

    using BlockGemm = decltype(GetBlockGemm());

    static constexpr bool kPadSeqLenQ = true;
    static constexpr bool kPadSeqLenK = true;

    using BlockGemmShape = ck::remove_cvref_t<typename BlockGemm::BlockGemmShape>;
    static constexpr ck::index_t kMPerBlock = BlockGemmShape::kM;
    static constexpr ck::index_t kNPerBlock = BlockGemmShape::kN;

    // kargs use aggregate initializer, so no constructor will provided
    // use inheritance to minimize karg size
    // user need to use MakeKargs() function to create kargs.
    struct FmhaRandUniformCommonKargs
    {
        void* rand_val_ptr;

        ck::index_t seqlen_q;
        ck::index_t seqlen_k;

        ck::index_t num_heads;
        ck::index_t num_batches;

        ck::index_t stride_seqlen_q;
        ck::index_t stride_seqlen_k;

        ck::index_t stride_nhead;

        uint64_t seed   = 1;
        uint64_t offset = 0;
    };

    struct FmhaRandUniformBatchModeKargs : FmhaRandUniformCommonKargs
    {
        ck::index_t stride_batch;
    };

    struct FmhaRandUniformGroupModeKargs : FmhaRandUniformCommonKargs
    {
        const int32_t* seqstart_q_ptr;
    };

    using Kargs = std::
        conditional_t<kIsGroupMode, FmhaRandUniformGroupModeKargs, FmhaRandUniformBatchModeKargs>;

    template <bool Cond = !kIsGroupMode>
    __host__ static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(void* rand_val_ptr,
              ck::index_t seqlen_q,
              ck::index_t seqlen_k,
              ck::index_t num_heads,
              ck::index_t num_batches,
              ck::index_t stride_seqlen_q,
              ck::index_t stride_seqlen_k,
              ck::index_t stride_nhead,
              ck::index_t stride_batch,
              std::tuple<uint64_t, uint64_t> drop_seed_offset)
    {
        Kargs kargs{{rand_val_ptr,
                     seqlen_q,
                     seqlen_k,
                     num_heads,
                     num_batches,
                     stride_seqlen_q,
                     stride_seqlen_k,
                     stride_nhead,
                     std::get<0>(drop_seed_offset),
                     std::get<1>(drop_seed_offset)},
                    stride_batch};

        return kargs;
    }

    template <bool Cond = kIsGroupMode>
    __host__ static constexpr std::enable_if_t<Cond, Kargs>
    MakeKargs(void* rand_val_ptr,
              ck::index_t seqlen_q,
              ck::index_t seqlen_k,
              ck::index_t num_heads,
              ck::index_t num_batches,
              ck::index_t stride_seqlen_q,
              ck::index_t stride_seqlen_k,
              ck::index_t stride_nhead,
              const void* seqstart_q_ptr,
              std::tuple<uint64_t, uint64_t> drop_seed_offset)
    {
        Kargs kargs{{rand_val_ptr,
                     seqlen_q,
                     seqlen_k,
                     num_heads,
                     num_batches,
                     stride_seqlen_q,
                     stride_seqlen_k,
                     stride_nhead,
                     std::get<0>(drop_seed_offset),
                     std::get<1>(drop_seed_offset)},
                    reinterpret_cast<const int32_t*>(seqstart_q_ptr),
                    seqstart_q_ptr};

        return kargs;
    }

    __host__ static constexpr auto GridSize(ck::index_t batch_size_,
                                            ck::index_t nhead_,
                                            ck::index_t seqlen_q_,
                                            ck::index_t seqlen_k_)
    {
        // at present, seqlen_k is not splitted by thread-groups
        return dim3(ck::math::integer_divide_ceil(seqlen_q_, kMPerBlock), nhead_, batch_size_);
    }

    __device__ static constexpr auto GetTileIndex(ck::index_t seqlen_q_, ck::index_t seqlen_k_)
    {
        // at present, seqlen_k is not splitted by thread-groups
        const ck::index_t i_block = blockIdx.x;
        const ck::index_t i_nhead = blockIdx.y;
        const ck::index_t i_batch = blockIdx.z;

        return ck::make_tuple(i_block, i_nhead, i_batch);
    }

    __host__ static constexpr auto BlockSize() { return dim3(kBlockSize); }

    __device__ static constexpr ck::index_t GetSmemSize()
    {
        return ck::tile_program::block::BlockDropout::MakeRandValLdsBlockDescriptor<BlockGemm>()
            .GetElementSpaceSize();
    }

    template <typename RandValDramBlockWindowTmp>
    __device__ void main_loop(const Kargs& kargs,
                              const ck::philox& ph,
                              void* randval_smem_ptr,
                              RandValDramBlockWindowTmp& randval_dram_block_window_tmp) const
    {
        using namespace ck;
        using namespace ck::tile_program;
        using namespace ck::tile_program::block;

        auto randval_dram_window =
            BlockDropout::MakeRandvalDramWindow<BlockGemm>(randval_dram_block_window_tmp, 0);

        const auto num_total_loop = math::integer_divide_ceil(kargs.seqlen_k, kNPerBlock);
        index_t i_total_loops     = 0;

        do
        {
            constexpr auto config =
                BlockGemm::Policy::template GetWarpGemmMWarpNWarp<typename BlockGemm::Problem>();
            using WG                    = remove_cvref_t<decltype(config.template At<0>())>;
            constexpr index_t MWarp     = config.template At<1>();
            constexpr index_t NWarp     = config.template At<2>();
            constexpr index_t kMPerStep = MWarp * WG::kM;
            constexpr index_t kNPerStep = NWarp * WG::kN;

            // randval tile in LDS
            auto randval_lds = make_tensor_view<AddressSpaceEnum::Lds>(
                reinterpret_cast<uint8_t*>(randval_smem_ptr),
                BlockDropout::MakeRandValLdsBlockDescriptor<BlockGemm>());

            auto randval_lds_window = make_tile_window(
                randval_lds,
                BlockDropout::MakeRandValLdsBlockDescriptor<BlockGemm>().GetLengths(),
                {0, 0});

            // register distribute
            auto randval_dist_generated = make_static_distributed_tensor<uint8_t>(
                BlockDropout::MakeRandValTileDistribution<BlockGemm>());

            static_assert(randval_dist_generated.kThreadElementSpaceSize == 16);

            auto randval_lds_read_window =
                make_tile_window(randval_lds_window.GetBottomTensorView(),
                                 randval_lds_window.GetWindowLengths(),
                                 randval_lds_window.GetWindowOrigin(),
                                 BlockDropout::MakeRandValLdsShuffleTileDistribution<BlockGemm>());

            const int start_m0_idx = randval_dram_window.GetWindowOrigin().At(Number<0>{});
            const int start_n0_idx = i_total_loops * kNPerBlock;

            static_for<0, kMPerBlock / kMPerStep, 1>{}([&](auto i_m0) {
                static_for<0, kNPerBlock / kNPerStep, 1>{}([&](auto i_n0) {
                    int block_row_start = (start_m0_idx / WG::kM) + (i_m0 * MWarp) + get_warp_id();
                    int block_col_start = start_n0_idx / WG::kN + i_n0;
                    uint2 rowcol        = make_uint2(block_row_start, block_col_start);

                    // generate random number
                    uint8_t random_uint8_t[16];
                    ph.get_random_16x8(random_uint8_t,
                                       reinterpret_cast<unsigned long long&>(rowcol));

                    constexpr auto randval_dist_generated_spans =
                        decltype(randval_dist_generated)::GetDistributedSpans();
                    int i_random_idx = 0;
                    sweep_tile_span(randval_dist_generated_spans[Number<0>{}], [&](auto idx0) {
                        sweep_tile_span(randval_dist_generated_spans[Number<1>{}], [&](auto idx1) {
                            constexpr auto i_j_idx          = make_tuple(idx0, idx1);
                            randval_dist_generated(i_j_idx) = random_uint8_t[i_random_idx++];
                        });
                    });
                    // save to LDS
                    store_tile(randval_lds_window, randval_dist_generated);
                    block_sync_lds();
                    // read from LDS to register
                    auto randval = load_tile(randval_lds_read_window);
                    // save to Global
                    const auto randval_store = cast_tile<RandValOutputDataType>(randval);
                    store_tile(randval_dram_window, randval_store);
                    move_tile_window(randval_dram_window, {0, kNPerStep});
                });
                move_tile_window(randval_dram_window, {kMPerStep, -kNPerBlock});
            });

            move_tile_window(randval_dram_window, {-kMPerBlock, kNPerBlock});

        } while(++i_total_loops < num_total_loop);
    }

    __device__ void operator()(Kargs kargs) const
    {
        using namespace ck;
        using namespace ck::tile_program;

        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        // divide problem
        const auto [i_tile_m, i_nhead, i_batch] = GetTileIndex(kargs.seqlen_q, kargs.seqlen_k);

        const index_t i_m0 = __builtin_amdgcn_readfirstlane(i_tile_m * kMPerBlock);

        long_index_t batch_offset_randval = 0;

        if constexpr(kIsGroupMode)
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
        }
        else
        {
            batch_offset_randval = static_cast<long_index_t>(i_batch) * kargs.stride_batch;
        }

        constexpr auto randval_dram_window_lengths =
            make_tuple(Number<kMPerBlock>{}, Number<kNPerBlock>{});

        RandValOutputDataType* rand_val_ptr =
            reinterpret_cast<RandValOutputDataType*>(kargs.rand_val_ptr) +
            static_cast<long_index_t>(i_nhead) * kargs.stride_nhead + batch_offset_randval;

        const auto randval_dram = [&]() {
            const auto randval_dram_naive = make_naive_tensor_view<AddressSpaceEnum::Global>(
                rand_val_ptr,
                make_tuple(kargs.seqlen_q, kargs.seqlen_k),
                make_tuple(kargs.stride_seqlen_q, kargs.stride_seqlen_k),
                Number<1>{},
                Number<1>{});

            return pad_tensor_view(randval_dram_naive,
                                   randval_dram_window_lengths,
                                   Sequence<kPadSeqLenQ, kPadSeqLenK>{});
        }();

        auto randval_dram_block_window_tmp =
            make_tile_window(randval_dram, randval_dram_window_lengths, {i_m0, 0});

        ck::philox ph(kargs.seed,
                      kargs.offset + (i_batch * kargs.num_heads + i_nhead) * get_warp_size() +
                          get_lane_id());

        main_loop(kargs, ph, smem_ptr, randval_dram_block_window_tmp);
    }
};
