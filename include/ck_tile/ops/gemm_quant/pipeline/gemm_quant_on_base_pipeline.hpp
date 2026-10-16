// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm_quant/pipeline/tile_gemm_quant_traits.hpp"

// Runs a group-quant GEMM (AQuant / BQuant / ABQuant) on a plain GEMM pipeline that accepts
// per-iteration scale windows (e.g. GemmPipelineAgBgCrCompAsync). The plain pipeline keeps its own
// data movement and LDS layout; only its block gemm is replaced by the quant block gemm, and the
// quant scale windows are streamed through the pipeline's scale-window path.

namespace ck_tile {

// Replaces the policy of a pipeline template Pipeline<Problem, Policy>.
template <typename Pipeline, typename NewPolicy>
struct rebind_policy;

template <template <typename, typename> class Pipeline,
          typename Problem,
          typename Policy,
          typename NewPolicy>
struct rebind_policy<Pipeline<Problem, Policy>, NewPolicy>
{
    using type    = Pipeline<Problem, NewPolicy>;
    using problem = Problem;
    using policy  = Policy;
};

// Adapts a quant block gemm to the register-tile call of the plain pipelines:
// (c, a_reg, b_reg, scale_a, scale_b), where an absent scale is a null tensor.
template <typename QuantBlockGemm>
struct BlockGemmQuantRegAdaptor : QuantBlockGemm
{
    using QuantBlockGemm::LocalPrefetch;

    // LDS -> register-tile read of the TDM pipelines. The quant policy runs them with one sub tile
    // per K block, so the LDS windows never slide.
    template <WindowSlideMode Mode,
              typename ADstBlockTile,
              typename BDstBlockTile,
              typename ASmemBlockWindow,
              typename BSmemBlockWindow,
              bool ALoadTranspose,
              bool BLoadTranspose>
    CK_TILE_DEVICE void LocalPrefetch(ADstBlockTile& a_dst_block_tile,
                                      BDstBlockTile& b_dst_block_tile,
                                      ASmemBlockWindow& a_block_window,
                                      BSmemBlockWindow& b_block_window,
                                      bool_constant<ALoadTranspose>,
                                      bool_constant<BLoadTranspose>)
    {
        static_assert(Mode == WindowSlideMode::Stay, "quant block gemm needs one sub tile");
        if constexpr(ALoadTranspose)
            a_dst_block_tile = load_tile_transpose(a_block_window);
        else
            load_tile(a_dst_block_tile, a_block_window);
        if constexpr(BLoadTranspose)
            b_dst_block_tile = load_tile_transpose(b_block_window);
        else
            load_tile(b_dst_block_tile, b_block_window);
    }

    template <index_t SubTileIdx = 0,
              typename CBlockTensor,
              typename ABlockTensor,
              typename BBlockTensor,
              typename ScaleATensor,
              typename ScaleBTensor>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ABlockTensor& a_block_tensor,
                                   const BBlockTensor& b_block_tensor,
                                   const ScaleATensor& scale_a,
                                   const ScaleBTensor& scale_b)
    {
        // The quant block gemm consumes its own register tiles, which share the distribution
        // of the tiles the pipeline loaded from LDS.
        this->block_gemm_impl_.a_warp_tile_.get_thread_buffer() =
            a_block_tensor.get_thread_buffer();
        this->block_gemm_impl_.b_warp_tile_.get_thread_buffer() =
            b_block_tensor.get_thread_buffer();
        auto sa = scale_a;
        auto sb = scale_b;
        if constexpr(is_null_tensor_v<ScaleATensor>)
            QuantBlockGemm::operator()(c_block_tensor, sb, a_block_tensor, b_block_tensor);
        else if constexpr(is_null_tensor_v<ScaleBTensor>)
            QuantBlockGemm::operator()(c_block_tensor, sa, a_block_tensor, b_block_tensor);
        else
            QuantBlockGemm::operator()(c_block_tensor, sa, sb, a_block_tensor, b_block_tensor);
    }
};

// For plain pipelines without a scale-window path (e.g. Mem, CompV4, CompTDM): the block gemm
// carries the scale windows and loads the scales of the current K block on each (c, a, b) call. The
// pipelines call the block gemm exactly once per K block in K order. a / b are either LDS windows
// (already read by LocalPrefetch) or register tiles.
template <typename QuantBlockGemm,
          typename AQWindow,
          typename BQWindow,
          typename Policy,
          typename Problem>
struct BlockGemmQuantStreamAdaptor : BlockGemmQuantRegAdaptor<QuantBlockGemm>
{
    AQWindow aq_window_;
    BQWindow bq_window_;

    CK_TILE_DEVICE BlockGemmQuantStreamAdaptor(const AQWindow& aq, const BQWindow& bq)
        : aq_window_{aq}, bq_window_{bq}
    {
    }

    template <typename CBlockTensor, typename ABlock, typename BBlock>
    CK_TILE_DEVICE void operator()(CBlockTensor& c, const ABlock& a, const BBlock& b)
    {
        constexpr auto steps = Policy::template GetScaleDramTileWindowSteps<Problem>();
        auto scale_a         = load_tile(aq_window_);
        auto scale_b         = load_tile(bq_window_);
        move_tile_window(aq_window_, steps[number<0>{}]);
        move_tile_window(bq_window_, steps[number<1>{}]);
        if constexpr(is_tile_window_with_static_distribution_v<ABlock>)
        {
            if constexpr(is_null_tensor_v<decltype(scale_a)>)
                QuantBlockGemm::operator()(c, scale_b, a, b);
            else if constexpr(is_null_tensor_v<decltype(scale_b)>)
                QuantBlockGemm::operator()(c, scale_a, a, b);
            else
                QuantBlockGemm::operator()(c, scale_a, scale_b, a, b);
        }
        else
            BlockGemmQuantRegAdaptor<QuantBlockGemm>::operator()(c, a, b, scale_a, scale_b);
    }
};

// Policy of the plain pipeline with the quant block gemm and quant scale-window steps.
// The quant block gemm covers a whole K block, so sub-tiled pipelines run with one sub tile.
template <typename BasePolicy, typename QuantPolicy, QuantType QT>
struct GemmQuantBasePolicy : BasePolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetPipelineSubTileNum()
    {
        return number<1>{};
    }

    template <typename Problem, bool = false>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        return BlockGemmQuantRegAdaptor<
            remove_cvref_t<decltype(QuantPolicy::template GetBlockGemm<Problem>())>>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetScaleDramTileWindowSteps()
    {
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        constexpr auto aq_step      = [] {
            if constexpr(QT == QuantType::BQuantGrouped)
                return make_array(0, 0);
            else
            {
                constexpr index_t KPerBlockAQ = KPerBlock / Problem::AQuantGroupSize::kK;
                if constexpr(std::is_same_v<typename Problem::AQLayout,
                                                 tensor_layout::gemm::ColumnMajor>)
                    return make_array(KPerBlockAQ, 0);
                else
                    return make_array(0, KPerBlockAQ);
            }
        }();
        constexpr auto bq_step = [] {
            if constexpr(QT == QuantType::AQuantGrouped)
                return make_array(0, 0);
            else
            {
                constexpr index_t KPerBlockBQ = KPerBlock / Problem::BQuantGroupSize::kK;
                if constexpr(std::is_same_v<typename Problem::BQLayout,
                                            tensor_layout::gemm::RowMajor>)
                    return make_array(KPerBlockBQ, 0);
                else
                    return make_array(0, KPerBlockBQ);
            }
        }();
        return make_tuple(aq_step, bq_step);
    }
};

// Quant pipeline (e.g. BQuantGemmPipelineAgBgCrCompV3) whose main loop runs on BasePipeline.
// Keeps the quant pipeline's interface (quant block sizes, scale vector sizes) and takes the
// data-movement interface (vector sizes, LDS size, loop structure) from BasePipeline.
template <QuantType QT, typename QuantPipeline, typename BasePipeline>
struct GemmQuantOnBasePipeline : QuantPipeline
{
    using Problem         = typename rebind_policy<QuantPipeline, void>::problem;
    using QuantPolicy     = typename rebind_policy<QuantPipeline, void>::policy;
    using NullWindow      = decltype(make_null_tile_window(make_tuple(number<0>{}, number<0>{})));
    using RebasedPipeline = typename rebind_policy<
        BasePipeline,
        GemmQuantBasePolicy<typename rebind_policy<BasePipeline, void>::policy, QuantPolicy, QT>>::
        type;

    // GetScaleDramTileWindowSteps only models the non-preshuffled K movement, and the kernel-call
    // overloads below drop the runtime m / n arguments. Preshuffled quant scales need the
    // M/N-dependent movement the dedicated CompV3 quant pipelines implement, so reject them here
    // instead of silently reading the wrong group scales.
    static_assert(!Problem::Traits::APreshuffleQuant && !Problem::Traits::BPreshuffleQuant,
                  "GemmQuantOnBasePipeline does not support preshuffled quant scales. Use the "
                  "dedicated quant CompV3 pipeline for APreshuffleQuant / BPreshuffleQuant.");

    // The Interwave loops of the shared pipelines call the block gemm as
    // (c, a_window, b_window, a_load_tr, b_load_tr). BlockGemmQuantStreamAdaptor only provides the
    // (c, a, b) form and hides the inherited overloads, so such a problem would not compile.
    static_assert(RebasedPipeline::Scheduler != GemmPipelineScheduler::Interwave,
                  "GemmQuantOnBasePipeline does not support the Interwave scheduler. Use "
                  "Intrawave, or a dedicated quant pipeline.");

    template <typename P>
    using scale_window_path_t = decltype(P::HasScaleWindowPath);
    // Pipelines that stream per-K-block scale windows themselves (e.g. CompAsync)
    static constexpr bool HasScaleWindowPath =
        is_detected<scale_window_path_t, BasePipeline>::value;
    static constexpr index_t PrefetchStages = BasePipeline::PrefetchStages;
    static constexpr bool DoubleSmemBuffer  = BasePipeline::DoubleSmemBuffer;
    static constexpr index_t GetVectorSizeA() { return BasePipeline::GetVectorSizeA(); }
    static constexpr index_t GetVectorSizeB() { return BasePipeline::GetVectorSizeB(); }
    static constexpr index_t GetSmemPackB() { return BasePipeline::GetSmemPackB(); }
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return RebasedPipeline::GetSmemSize();
    }
    CK_TILE_HOST_DEVICE static constexpr bool BlockHasHotloop(index_t num_loop)
    {
        return BasePipeline::BlockHasHotloop(num_loop);
    }
    CK_TILE_HOST_DEVICE static constexpr TailNumber GetBlockLoopTailNum(index_t num_loop)
    {
        return BasePipeline::GetBlockLoopTailNum(num_loop);
    }
    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        return concat('_', QuantPipeline::GetName(), BasePipeline::GetName());
    }

    template <typename ADramWindow,
              typename BDramWindow,
              typename AQDramWindow,
              typename BQDramWindow>
    CK_TILE_DEVICE auto Run(const ADramWindow& a,
                            const BDramWindow& b,
                            const AQDramWindow& aq,
                            const BQDramWindow& bq,
                            index_t num_loop,
                            void* p_smem) const
    {
        auto aq_window = [&] {
            if constexpr(QT == QuantType::BQuantGrouped)
                return NullWindow{};
            else
                return GemmAQuantPipelineAgBgCrImplBase<Problem, QuantPolicy>{}.GetAQDramLoadWindow(
                    aq);
        }();
        auto bq_window = [&] {
            if constexpr(QT == QuantType::AQuantGrouped)
                return NullWindow{};
            else
                return GemmBQuantPipelineAgBgCrImplBase<Problem, QuantPolicy>{}.GetBQDramLoadWindow(
                    bq);
        }();
        if constexpr(HasScaleWindowPath)
            return RebasedPipeline{}(make_tuple(a),
                                     element_wise::PassThrough{},
                                     make_tuple(b),
                                     element_wise::PassThrough{},
                                     make_tuple(aq_window),
                                     make_tuple(bq_window),
                                     num_loop,
                                     p_smem);
        else
        {
            using Policy    = typename rebind_policy<RebasedPipeline, void>::policy;
            using BlockGemm = BlockGemmQuantStreamAdaptor<
                remove_cvref_t<decltype(QuantPolicy::template GetBlockGemm<Problem>())>,
                decltype(aq_window),
                decltype(bq_window),
                Policy,
                Problem>;
            const auto run = [&](auto hot_loop_, auto tail_num_) {
                return typename RebasedPipeline::template PipelineImpl<RebasedPipeline::Scheduler>{}
                    .template operator()<hot_loop_.value, tail_num_.value>(
                        make_tuple(a),
                        element_wise::PassThrough{},
                        make_tuple(b),
                        element_wise::PassThrough{},
                        num_loop,
                        p_smem,
                        BlockGemm{aq_window, bq_window});
            };
            return RebasedPipeline::TailHandler(
                run, BlockHasHotloop(num_loop), GetBlockLoopTailNum(num_loop));
        }
    }

    // AQuant kernel call
    template <typename ADramWindow, typename BDramWindow, typename AQDramWindow>
    CK_TILE_DEVICE auto operator()(const ADramWindow& a,
                                   const BDramWindow& b,
                                   const AQDramWindow& aq,
                                   index_t num_loop,
                                   void* p_smem,
                                   index_t = 0) const
    {
        if constexpr(QT == QuantType::AQuantGrouped)
            return Run(a, b, aq, NullWindow{}, num_loop, p_smem);
        else
            return Run(a, b, NullWindow{}, aq, num_loop, p_smem);
    }

    // BQuant kernel call with runtime loop structure (already derived from num_loop by Run)
    template <typename ADramWindow, typename BDramWindow, typename BQDramWindow>
    CK_TILE_DEVICE auto operator()(const ADramWindow& a,
                                   const BDramWindow& b,
                                   const BQDramWindow& bq,
                                   index_t num_loop,
                                   bool,
                                   TailNumber,
                                   void* p_smem,
                                   index_t = 0) const
    {
        return Run(a, b, NullWindow{}, bq, num_loop, p_smem);
    }

    // ABQuant kernel calls
    template <typename ADramWindow,
              typename BDramWindow,
              typename AQDramWindow,
              typename BQDramWindow>
    CK_TILE_DEVICE auto operator()(const ADramWindow& a,
                                   const BDramWindow& b,
                                   const AQDramWindow& aq,
                                   const BQDramWindow& bq,
                                   index_t num_loop,
                                   void* p_smem,
                                   index_t = 0,
                                   index_t = 0) const
    {
        return Run(a, b, aq, bq, num_loop, p_smem);
    }

    template <typename ADramWindow,
              typename BDramWindow,
              typename AQDramWindow,
              typename BQDramWindow>
    CK_TILE_DEVICE auto operator()(const ADramWindow& a,
                                   const BDramWindow& b,
                                   const AQDramWindow& aq,
                                   const BQDramWindow& bq,
                                   index_t num_loop,
                                   bool,
                                   TailNumber,
                                   void* p_smem,
                                   index_t = 0,
                                   index_t = 0) const
    {
        return Run(a, b, aq, bq, num_loop, p_smem);
    }
};

} // namespace ck_tile
