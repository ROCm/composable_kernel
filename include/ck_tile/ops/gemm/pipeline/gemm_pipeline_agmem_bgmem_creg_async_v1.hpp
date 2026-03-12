// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp"
#include "ck_tile/host/concat.hpp"
#include "gemm_pipeline_agmem_bgmem_creg_v1.hpp"

namespace ck_tile {

//  A Tile Window: global memory
//  B Tile Window: global memory
//  C Distributed tensor: register
template <typename Problem, typename Policy = GemmPipelineAgBgCrCompAsyncDefaultPolicy>
struct GemmPipelineAGmemBGmemCRegAsyncV1 : public BaseGemmPipelineAGmemBGmemCRegV1<Problem>
{
    using Base             = BaseGemmPipelineAGmemBGmemCRegV1<Problem>;
    using PipelineImplBase = GemmPipelineAgBgCrImplBase<Problem, Policy>;

    using AsDataType = remove_cvref_t<typename Problem::AsDataTypeTuple>;
    using BsDataType = remove_cvref_t<typename Problem::BsDataTypeTuple>;
    using CDataType  = remove_cvref_t<typename Problem::CDataType>;

    using AElementWise   = remove_cvref_t<typename Problem::AElementWise>;
    using BElementWise   = remove_cvref_t<typename Problem::BElementWise>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;

    using AsLayout = remove_cvref_t<typename Problem::AsLayoutTuple>;
    using BsLayout = remove_cvref_t<typename Problem::BsLayoutTuple>;
    using CLayout  = remove_cvref_t<typename Problem::CLayout>;

    using ALayout = remove_cvref_t<std::tuple_element_t<0, AsLayout>>;
    using BLayout = remove_cvref_t<std::tuple_element_t<0, BsLayout>>;

    using ADataType = remove_cvref_t<std::tuple_element_t<0, AsDataType>>;
    using BDataType = remove_cvref_t<std::tuple_element_t<0, BsDataType>>;

    using BlockGemm = remove_cvref_t<decltype(Policy::template GetBlockGemm<Problem>())>;

    using I0 = number<0>;
    using I1 = number<1>;
    using I2 = number<2>;

    static constexpr index_t BlockSize = Problem::kBlockSize;

    static constexpr index_t kMPerBlock = BlockGemmShape::kM;
    static constexpr index_t kNPerBlock = BlockGemmShape::kN;
    static constexpr index_t kKPerBlock = BlockGemmShape::kK;

    static constexpr bool Async = true;

    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeA()
    {
        return Policy::template GetVectorSizeA<Problem, IsWave32Host>();
    }
    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeB()
    {
        return Policy::template GetVectorSizeB<Problem, IsWave32Host>();
    }
    static constexpr index_t GetVectorSizeC() { return Policy::template GetVectorSizeC<Problem>(); }

    static constexpr index_t GetSmemPackA() { return Policy::template GetSmemPackA<Problem>(); }
    static constexpr index_t GetSmemPackB() { return Policy::template GetSmemPackB<Problem>(); }

    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;

    static constexpr bool Preshuffle = Problem::Preshuffle;

    static constexpr auto Scheduler = Problem::Scheduler;

    static constexpr index_t NumWaveGroups = Problem::NumWaveGroups;

    static constexpr index_t kLdsAlignmentInBytes = 16;

    static constexpr auto is_a_load_tr_v = bool_constant<PipelineImplBase::is_a_load_tr>{};
    static constexpr auto is_b_load_tr_v = bool_constant<PipelineImplBase::is_b_load_tr>{};

    [[nodiscard]] CK_TILE_HOST static const std::string GetPipelineName()
    {
        // clang-format off
        return "BASIC_ASYNC_V1";
        // clang-format on
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "pipeline_AGmemBGmemCRegAsyncV1", 
                      concat('x', kMPerBlock, kNPerBlock, kKPerBlock,  BlockSize),
                      concat('x', GetVectorSizeA(), GetVectorSizeB(), GetVectorSizeC()),
                      concat('x', kPadM, kPadN, kPadK));
        // clang-format on
    }

    // For the basic gemm pipelien DoubleSmemBuffer set to be false naturally.
    static constexpr bool DoubleSmemBuffer = false;

    CK_TILE_HOST_DEVICE static constexpr auto TransposeC() { return Problem::TransposeC; }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
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
                  typename AsDramBlockWindowTmp,
                  typename BsDramBlockWindowTmp,
                  typename AElementFunction,
                  typename BElementFunction,
                  typename std::enable_if_t<is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                                is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                            bool>* = nullptr>
        CK_TILE_HOST_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                            const AElementFunction& a_element_func,
                                            const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                            const BElementFunction& b_element_func,
                                            index_t num_loop,
                                            void* p_smem) const
        {
            // TODO support multi-ABD
            static_assert(1 == std::tuple_size_v<AsDramBlockWindowTmp>);
            static_assert(1 == std::tuple_size_v<BsDramBlockWindowTmp>);
            using ADramBlockWindowTmp =
                remove_cvref_t<std::tuple_element_t<number<0>{}, AsDramBlockWindowTmp>>;
            using BDramBlockWindowTmp =
                remove_cvref_t<std::tuple_element_t<number<0>{}, BsDramBlockWindowTmp>>;
            // TODO currently fused elementwise are not supported
            ignore = a_element_func;
            ignore = b_element_func;
            static_assert(std::is_same_v<AElementFunction, element_wise::PassThrough>);
            static_assert(std::is_same_v<BElementFunction, element_wise::PassThrough>);
            static_assert(
                std::is_same_v<ADataType, remove_cvref_t<typename ADramBlockWindowTmp::DataType>> &&
                    std::is_same_v<BDataType,
                                   remove_cvref_t<typename BDramBlockWindowTmp::DataType>>,
                "Data Type conflict on A and B matrix input data type.");

            constexpr bool is_a_col_major =
                std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor>;
            constexpr bool is_b_row_major = std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>;

            static_assert(is_a_col_major
                              ? (kKPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 kMPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}])
                              : (kMPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 kKPerBlock == ADramBlockWindowTmp{}.get_window_lengths()[I1{}]),
                          "A block window has incorrect lengths for defined ALayout!");
            static_assert(is_b_row_major
                              ? (kKPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 kNPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1{}])
                              : (kNPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I0{}] &&
                                 kKPerBlock == BDramBlockWindowTmp{}.get_window_lengths()[I1{}]),
                          "B block window has incorrect lengths for defined BLayout!");

            ////////////// global window & register /////////////////
            // A DRAM tile window(s) for load
            auto a_tile_windows =
                make_tile_window(a_dram_block_window_tmp[I0{}].get_bottom_tensor_view(),
                                 make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                                 a_dram_block_window_tmp[I0{}].get_window_origin(),
                                 Policy::template MakeADramTileDistribution<Problem>());
            // B DRAM window(s) for load
            auto b_tile_windows =
                make_tile_window(b_dram_block_window_tmp[I0{}].get_bottom_tensor_view(),
                                 make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
                                 b_dram_block_window_tmp[I0{}].get_window_origin(),
                                 Policy::template MakeBDramTileDistribution<Problem>());

            // this pipeline has a pair of LDS buffers per logical tile
            auto&& [a_lds_block, b_lds_block] = Base::GetABLdsTensorViews(p_smem);

            // set up LDS tile shapes
            constexpr auto a_lds_shape = []() {
                if constexpr(is_a_load_tr_v)
                    return make_tuple(number<kKPerBlock>{}, number<kMPerBlock>{});
                else
                    return make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{});
            }();

            constexpr auto b_lds_shape = []() {
                if constexpr(is_b_load_tr_v)
                    return make_tuple(number<kKPerBlock>{}, number<kNPerBlock>{});
                else
                    return make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{});
            }();

            // LDS tile windows for storing, one per LDS buffer
            auto a_copy_lds_window = make_tile_window(a_lds_block, a_lds_shape, {0, 0});
            auto b_copy_lds_window = make_tile_window(b_lds_block, b_lds_shape, {0, 0});

            // Block GEMM
            auto block_gemm = BlockGemm();

            // Acc register tile
            auto c_block_tile = block_gemm.MakeCBlockTile();

            using ADramTileWindowStep = typename ADramBlockWindowTmp::BottomTensorIndex;
            using BDramTileWindowStep = typename BDramBlockWindowTmp::BottomTensorIndex;

            constexpr ADramTileWindowStep a_dram_tile_window_step =
                is_a_col_major ? make_array(kKPerBlock, 0) : make_array(0, kKPerBlock);
            constexpr BDramTileWindowStep b_dram_tile_window_step =
                is_b_row_major ? make_array(kKPerBlock, 0) : make_array(0, kKPerBlock);

            // tile distribution for the register tiles
            constexpr auto ALdsTileDistr =
                make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
            constexpr auto BLdsTileDistr =
                make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());

            using ALdsTile = decltype(make_static_distributed_tensor<ADataType>(ALdsTileDistr));
            using BLdsTile = decltype(make_static_distributed_tensor<BDataType>(BLdsTileDistr));

            // register tiles; double buffering -> a register tile corresponds to a LDS tile window
            ALdsTile a_block_tile;
            BLdsTile b_block_tile;

            constexpr auto a_lds_input_tile_distr = [ALdsTileDistr]() {
                if constexpr(is_a_load_tr_v)
                    return make_static_tile_distribution(
                        typename InputTileDistributionTraits<
                            typename decltype(ALdsTileDistr)::DstrEncode,
                            typename Problem::ADataType>::TransposedDstrEncode{});
                else
                    return ALdsTileDistr;
            }();
            constexpr auto b_lds_input_tile_distr = [BLdsTileDistr]() {
                if constexpr(is_b_load_tr_v)
                    return make_static_tile_distribution(
                        typename InputTileDistributionTraits<
                            typename decltype(BLdsTileDistr)::DstrEncode,
                            typename Problem::BDataType>::TransposedDstrEncode{});
                else
                    return BLdsTileDistr;
            }();

            // LDS tile windows for reading;
            // they share the data pointer with the LDS windows for storing
            // but also associate with a distribution to produce a register tile when reading
            auto a_lds_ld_window =
                make_tile_window(a_lds_block, a_lds_shape, {0, 0}, a_lds_input_tile_distr);
            auto b_lds_ld_window =
                make_tile_window(b_lds_block, b_lds_shape, {0, 0}, b_lds_input_tile_distr);

            static_assert((!(is_tile_window_linear_v<decltype(a_lds_ld_window)>) &&
                           !(is_tile_window_linear_v<decltype(b_lds_ld_window)>)),
                          "LDS windows must not be linear");

            // Global Prefetch
            Base::GlobalPrefetchAsync(a_copy_lds_window, a_tile_windows, a_dram_tile_window_step);
            Base::GlobalPrefetchAsync(b_copy_lds_window, b_tile_windows, b_dram_tile_window_step);

            tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);

            block_sync_lds_direct_load();

            if constexpr(HasHotLoop)
            {
                index_t iCounter = num_loop - 1;
                while(iCounter > 0)
                {
                    Base::LocalPrefetch(a_block_tile, a_lds_ld_window, is_a_load_tr_v);
                    Base::LocalPrefetch(b_block_tile, b_lds_ld_window, is_b_load_tr_v);

                    block_sync_lds();

                    Base::GlobalPrefetchAsync(
                        a_copy_lds_window, a_tile_windows, a_dram_tile_window_step);
                    Base::GlobalPrefetchAsync(
                        b_copy_lds_window, b_tile_windows, b_dram_tile_window_step);

                    // GEMM i
                    block_gemm(c_block_tile, a_block_tile, b_block_tile);

                    block_sync_lds_direct_load();

                    iCounter--;
                }
            }

            // tail
            {
                Base::LocalPrefetch(a_block_tile, a_lds_ld_window, is_a_load_tr_v);
                Base::LocalPrefetch(b_block_tile, b_lds_ld_window, is_b_load_tr_v);
                // GEMM num_loop - 1
                block_gemm(c_block_tile, a_block_tile, b_block_tile);
            }

            return c_block_tile;
        }
    };

    template <typename AsDramBlockWindowTmp,
              typename BsDramBlockWindowTmp,
              typename std::enable_if_t<is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                            is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto RunPipeline  = [&](auto hot_loop_) {
            constexpr bool hot_loop = hot_loop_.value;
            return PipelineImpl<Scheduler>{}.template operator()<hot_loop>(
                a_dram_block_window_tmp,
                element_wise::PassThrough{},
                b_dram_block_window_tmp,
                element_wise::PassThrough{},
                num_loop,
                p_smem);
        };
        return Base::TailHandler(RunPipeline, has_hot_loop);
    }

    template <typename ADramBlockWindowTmp,
              typename BDramBlockWindowTmp,
              typename std::enable_if_t<!is_detected<is_tuple, ADramBlockWindowTmp>::value &&
                                            !is_detected<is_tuple, BDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BDramBlockWindowTmp& b_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* p_smem) const
    {
        return operator()(ck_tile::make_tuple(a_dram_block_window_tmp),
                          ck_tile::make_tuple(b_dram_block_window_tmp),
                          num_loop,
                          p_smem);
    }

    template <typename AsDramBlockWindowTmp,
              typename BsDramBlockWindowTmp,
              typename AElementFunction,
              typename BElementFunction,
              typename std::enable_if_t<is_detected<is_tuple, AsDramBlockWindowTmp>::value &&
                                            is_detected<is_tuple, BsDramBlockWindowTmp>::value,
                                        bool>* = nullptr>
    CK_TILE_HOST_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                        const AElementFunction& a_element_func,
                                        const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                        const BElementFunction& b_element_func,
                                        index_t num_loop,
                                        void* p_smem) const
    {
        const bool has_hot_loop = Base::BlockHasHotloop(num_loop);
        const auto RunPipeline  = [&](auto hot_loop_) {
            constexpr bool hot_loop = hot_loop_.value;
            return PipelineImpl<Scheduler>{}.template operator()<hot_loop>(a_dram_block_window_tmp,
                                                                           a_element_func,
                                                                           b_dram_block_window_tmp,
                                                                           b_element_func,
                                                                           num_loop,
                                                                           p_smem);
        };
        return Base::TailHandler(RunPipeline, has_hot_loop);
    }
};

} // namespace ck_tile
