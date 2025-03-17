#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/host/concat.hpp"

namespace ck_tile {
// A Tile Window: global memory
// B Tile Window: global memory
// C Distributed Tensor: register

template <typename Problem>
struct BaseGemmPipelineAgBgCrCompV5
{
    static constexpr index_t PrefetchStages  = 1;
    static constexpr index_t PrefillStages   = 1;
    static constexpr index_t GlobalBufferNum = 2;

    CK_TILE_HOST_DEVICE static constexpr auto TransposeC() { return Problem::TransposeC; }

    CK_TILE_HOST_DEVICE static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return num_loop > 0;
    }

    CK_TILE_HOST_DEVICE static constexpr TailNumber GetBlockLoopTailNum(index_t num_loop)
    {
        if(num_loop > PrefetchStages)
        {
            return TailNumber::One;
        }
        else
        {
            return TailNumber::Two;
        }
    }
};

template <typename Problem, typename Policy = GemmPipelineAgBgCrCompV4DefaultPolicy>
struct GemmPipelineAgBgCrCompV5 : public BaseGemmPipelineAgBgCrCompV5<Problem>
{
    using Base             = BaseGemmPipelineAgBgCrCompV5<Problem>;
    using PipelineImplBase = GemmPipelineAgBgCrImplBase<Problem, Policy>;

    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::BDataType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    using ALayout = remove_cvref_t<typename Problem::ALayout>;
    using BLayout = remove_cvref_t<typename Problem::BLayout>;
    using CLayout = remove_cvref_t<typename Problem::CLayout>;

    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;
    using I0        = number<0>;
    using I1        = number<1>;
    using I2        = number<2>;

    static constexpr index_t BlockSize = Problem::kBlockSize;

    static constexpr index_t MPerBlock = BlockGemmShape::kM;
    static constexpr index_t NPerBlock = BlockGemmShape::kN;
    static constexpr index_t KPerBlock = BlockGemmShape::kK;

    static constexpr index_t GetVectorSizeA() { return Policy::template GetVectorSizeA<Problem>(); }
    static constexpr index_t GetVectorSizeB() { return Policy::template GetVectorSizeB<Problem>(); }
    static constexpr index_t GetVectorSizeC() { return Policy::template GetVectorSizeC<Problem>(); }

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr bool DoubleSmemBuffer = Problem::DoubleSmemBuffer;

    static constexpr bool HasHotLoop = Problem::HasHotLoop;
    static constexpr auto TailNum    = Problem::TailNum;
    static constexpr auto Scheduler  = Problem::Scheduler;

    static constexpr index_t NumWarps =
        BlockGemmShape::NumWarps; // reduce_on_sequence(BlockGemmShape::BlockWarps{},
                                  // multiplies{}, number<1>{});
    static constexpr index_t KTileSize = BlockGemmShape::WarpTile::at(I2{});
    // static constexpr index_t KPerBlock = BlockGemmShape::kK;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    CK_TILE_HOST_DEVICE static constexpr auto IsTransposeC()
    {
        return Policy::template IsTransposeC<Problem>();
    }

    template <GemmPipelineScheduler Scheduler>
    struct PipelineImpl : public PipelineImplBase
    {
    };

    template <>
    struct PipelineImpl<GemmPipelineScheduler::Intrawave> : public PipelineImplBase
    {
        using Base = PipelineImplBase;

        template <bool HasHotLoop,
                  TailNumber TailNum,
                  typename ADramBlockWindowTmp,
                  typename AElementFunction,
                  typename BDramBlockWindowTmp,
                  typename BElementFunction>
        CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                       const AElementFunction& a_element_func,
                                       const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                       const BElementFunction& b_element_func,
                                       index_t num_loop,
                                       void* __restrict__ p_smem_0) const
        {
            static_assert(
                std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BDataType,
                                   remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
                "Data Type conflict on A and B matrix input data type.");

            static_assert(
                KPerBlock % ((NumWarps / 2) * KTileSize) == 0,
                "Ping Pong Warps, TileSize and Block Size for K dimensions does not match.");

            constexpr bool is_a_col_major =
                std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
            constexpr bool is_b_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;

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

            static constexpr index_t num_stages_ = 2;
            // This is used to identify the register tile on which a warp always operates on. 
            // For instance, warp-0 always uses a_block_tile_0 for reading in one cycle
            // and execution in the next cycle.
            index_t group_id                     = get_warp_id() % num_stages_;

            // op_id indicated one of the steps (0 - Read, 1 - Gemm Execution)
            // Each warp performs read in one cycle and in the next cycles performs GEMM operation
            // on the same block_tile that it has read in the previous cycle.
            index_t op_id                        = get_warp_id() % num_stages_;

            // global memory structures here.
            auto a_copy_dram_window =
                make_tile_window_linear(a_dram_block_window_tmp.get_bottom_tensor_view(),
                                        make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                                        a_dram_block_window_tmp.get_window_origin(),
                                        Policy::template MakeADramTileDistribution<Problem>());

            // B DRAM tile window for load
            auto b_copy_dram_window =
                make_tile_window_linear(b_dram_block_window_tmp.get_bottom_tensor_view(),
                                        make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                                        b_dram_block_window_tmp.get_window_origin(),
                                        Policy::template MakeBDramTileDistribution<Problem>());

            // DRAM window steps.
            using ADramTileWindowStep = typename ADramBlockWindowTmp::BottomTensorIndex;
            using BDramTileWindowStep = typename BDramBlockWindowTmp::BottomTensorIndex;
            constexpr ADramTileWindowStep a_dram_tile_window_step =
                is_a_col_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
            constexpr BDramTileWindowStep b_dram_tile_window_step =
                is_b_row_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);

            // LDS tiles here.
            constexpr auto ALdsTileDistr = decltype(make_static_tile_distribution(
                BlockGemm::MakeABlockDistributionEncode())){};
            constexpr auto BLdsTileDistr = decltype(make_static_tile_distribution(
                BlockGemm::MakeBBlockDistributionEncode())){};

            using ALdsTile = decltype(make_static_distributed_tensor<ADataType>(ALdsTileDistr));
            using BLdsTile = decltype(make_static_distributed_tensor<BDataType>(BLdsTileDistr));

            // array<ALdsTile, num_stages_> a_tiles;
            // array<BLdsTile, num_stages_> b_tiles;
            ALdsTile a_tile_0, a_tile_1;
            BLdsTile b_tile_0, b_tile_1;

            // LDS structures for temporary stroage
            // Loads from DRAM to LDS has more memory bandwidth compared to DRAM to Registers
            auto&& [a_lds_block, b_lds_block] = Base::GetABLdsTensorViews(p_smem_0);

            auto a_copy_lds_window = make_tile_window(
                a_lds_block, make_tuple(number<MPerBlock>{}, number<KPerBlock>{}), {0, 0});
            auto b_copy_lds_window = make_tile_window(
                b_lds_block, make_tuple(number<NPerBlock>{}, number<KPerBlock>{}), {0, 0});

            auto a_lds_window =
                make_tile_window_linear(a_lds_block,
                                        make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                                        {0, 0},
                                        ALdsTileDistr);
            auto b_lds_window =
                make_tile_window_linear(b_lds_block,
                                        make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                                        {0, 0},
                                        BLdsTileDistr);

            // Register tile for A and B.
            constexpr auto ABlockTileDistr = a_copy_dram_window.get_tile_distribution();
            constexpr auto BBlockTileDistr = b_copy_dram_window.get_tile_distribution();
            using ABlockTile = decltype(make_static_distributed_tensor<ADataType>(ABlockTileDistr));
            using BBlockTile = decltype(make_static_distributed_tensor<BDataType>(BBlockTileDistr));
            ABlockTile a_global_load_tile;
            BBlockTile b_global_load_tile;

            // Block GEMM
            auto block_gemm   = BlockGemm();
            auto c_block_tile = block_gemm.MakeCBlockTile();

            // initialize C
            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            // define ping, pong steps here as lambda functions.
            auto MemoryOpsStep = [&](auto idx) {
                // Memory read half here.
                Base::GlobalPrefetch(
                    a_global_load_tile, a_copy_dram_window, a_dram_tile_window_step);
                Base::GlobalPrefetch(
                    b_global_load_tile, b_copy_dram_window, b_dram_tile_window_step);

                // LDS write 0
                if constexpr(is_a_col_major)
                {
                    auto a_shuffle_tmp = make_static_distributed_tensor<ADataType>(
                        Policy::template MakeShuffledARegTileDistribution<Problem>());
                    transpose_tile2d(a_shuffle_tmp, a_global_load_tile);
                    Base::LocalPrefill(a_copy_lds_window, a_shuffle_tmp, a_element_func);
                }
                else
                {
                    Base::LocalPrefill(a_copy_lds_window, a_global_load_tile, a_element_func);
                }
                if constexpr(is_b_row_major)
                {
                    auto b_shuffle_tmp = make_static_distributed_tensor<BDataType>(
                        Policy::template MakeShuffledBRegTileDistribution<Problem>());
                    transpose_tile2d(b_shuffle_tmp, b_global_load_tile);
                    Base::LocalPrefill(b_copy_lds_window, b_shuffle_tmp, b_element_func);
                }
                else
                {
                    Base::LocalPrefill(b_copy_lds_window, b_global_load_tile, b_element_func);
                }

                // transfer from LDS to registers
                if(idx == 0)
                {
                    Base::LocalPrefetch(a_tile_0, a_lds_window);
                    Base::LocalPrefetch(b_tile_0, b_lds_window);
                }
                else
                {
                    Base::LocalPrefetch(a_tile_1, a_lds_window);
                    Base::LocalPrefetch(b_tile_1, b_lds_window);
                }
            };

            auto ComputeStep = [&](auto idx) {
                if(idx == 0)
                {
                    block_gemm(c_block_tile, a_tile_0, b_tile_0);
                    // tile_elementwise_inout([](auto& c) { c += 1; }, c_block_tile);
                }
                else
                {
                    block_gemm(c_block_tile, a_tile_1, b_tile_1);
                    // tile_elementwise_inout([](auto& c) { c += 1; }, c_block_tile);
                }
            };

            if(op_id == 0)
            {
                MemoryOpsStep(group_id);
            }            
            // start the main loop.
            index_t num_compute_steps = __builtin_amdgcn_readfirstlane(num_loop) * 2 - 1;

            while(num_compute_steps > 0)
            {
                // Synchronize all threads in a thread block
                block_sync_lds();
                op_id = (op_id + 1) % num_stages_;

                if(op_id == 0)
                {
                    MemoryOpsStep(group_id);
                }
                else
                {
                    ComputeStep(group_id);
                }

                num_compute_steps -= 1;
            }
            

            // Handle Tail Number here.
            block_sync_lds();
            if(op_id == 0)
            {
                ComputeStep(group_id);
            }

            block_sync_lds();
            return c_block_tile;
        }
    };

    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const AElementFunction& a_element_func,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const BElementFunction& b_element_func,
                                   index_t num_loop,
                                   void* p_smem_0) const
    {
        return PipelineImpl<Scheduler>{}.template operator()<HasHotLoop, TailNum>(
            a_dram_block_window_tmp,
            a_element_func,
            b_dram_block_window_tmp,
            b_element_func,
            num_loop,
            p_smem_0);
    }

    public:
    template <typename ADramBlockWindowTmp, typename BDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const index_t num_loop,
                                   void* __restrict__ p_smem_0) const
    {
        return PipelineImpl<Scheduler>{}.template operator()<HasHotLoop, TailNum>(
            a_dram_block_window_tmp,
            [](const ADataType& a) { return a; },
            b_dram_block_window_tmp,
            [](const BDataType& b) { return b; },
            num_loop,
            p_smem_0);
    }
};

} // namespace ck_tile
