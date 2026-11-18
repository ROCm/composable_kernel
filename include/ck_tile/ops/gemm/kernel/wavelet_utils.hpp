// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/utility/ignore.hpp"

namespace ck_tile {

// Kernel-side support for wavelet and named-barrier GEMM pipelines, shared by every kernel that
// can run one. A wavelet pipeline launches load waves beyond the math waves its BlockGemm maps
// onto the C tile (LaunchBlockSize > BlockSize). Only the math waves hold an accumulator and run
// the epilogue; the load waves run a matching barrier sequence instead. A named-barrier pipeline
// takes a token its kernel minted once per launch (RunGemmPipeline forwards it). Other pipelines
// expose none of these members, and the helpers below detect that via SFINAE so each kernel
// dispatches without duplicating the machinery.
namespace impl {
template <typename T, typename = void>
struct has_launch_block_size : std::false_type
{
};
template <typename T>
struct has_launch_block_size<T, std::void_t<decltype(T::LaunchBlockSize)>> : std::true_type
{
};

template <typename T, typename = void>
struct has_is_wavelet : std::false_type
{
};
template <typename T>
struct has_is_wavelet<T, std::void_t<decltype(T::IsWavelet)>> : std::true_type
{
};

template <typename T, typename = void>
struct pipeline_barrier_pipeline
{
    using type = void;
};
template <typename T>
struct pipeline_barrier_pipeline<T, std::void_t<typename T::BarrierPipeline>>
{
    using type = typename T::BarrierPipeline;
};
} // namespace impl

// Block size to launch with: wavelet pipelines need LaunchBlockSize (load + math waves);
// all others fall back to BlockSize.
template <typename Pipeline>
inline constexpr index_t GemmPipelineLaunchBlockSize = []() {
    if constexpr(impl::has_launch_block_size<Pipeline>::value)
        return Pipeline::LaunchBlockSize;
    else
        return Pipeline::BlockSize;
}();

// True when the pipeline uses wavelet load/math wave specialization.
template <typename Pipeline>
inline constexpr bool is_wavelet_pipeline = []() {
    if constexpr(impl::has_is_wavelet<Pipeline>::value)
        return Pipeline::IsWavelet;
    else
        return false;
}();

// The named barrier pipeline a GEMM pipeline synchronises over, or void when it uses none. The
// hardware allows one named barrier array per kernel, so the kernel running such a pipeline
// declares it as its own `barrier_pipeline` and arms it with init().
template <typename Pipeline>
using barrier_pipeline_of_t = typename impl::pipeline_barrier_pipeline<Pipeline>::type;

// Run GemmPipeline over one C tile. A pipeline with named barriers takes the token its kernel
// minted with BarrierPipeline::init(); every other pipeline takes none, and @p barriers is a
// placeholder the kernel passes uniformly. This overload resolves the hot loop and tail from
// @p num_loop at run time, so callers whose K-loop count varies per tile (split-K, Stream-K) need
// not dispatch on them.
template <typename GemmPipeline,
          typename AsBlockWindow,
          typename AElementWise,
          typename BsBlockWindow,
          typename BElementWise,
          typename BarrierToken>
CK_TILE_DEVICE auto RunGemmPipeline(const AsBlockWindow& as_block_window,
                                    const AElementWise& a_element_func,
                                    const BsBlockWindow& bs_block_window,
                                    const BElementWise& b_element_func,
                                    index_t num_loop,
                                    void* smem_ptr,
                                    const BarrierToken& barriers)
{
    if constexpr(std::is_void_v<barrier_pipeline_of_t<GemmPipeline>>)
    {
        ignore = barriers;
        return GemmPipeline{}(
            as_block_window, a_element_func, bs_block_window, b_element_func, num_loop, smem_ptr);
    }
    else
    {
        return GemmPipeline{}(as_block_window,
                              a_element_func,
                              bs_block_window,
                              b_element_func,
                              num_loop,
                              smem_ptr,
                              barriers);
    }
}

// True on the waves that hold the C accumulator: the math waves of a wavelet pipeline, every
// wave of any other pipeline. Wave-uniform, so it may guard code that issues barriers only when
// both sides of the branch issue the same ones.
template <typename GemmPipeline>
CK_TILE_DEVICE bool IsMathWave()
{
    if constexpr(is_wavelet_pipeline<GemmPipeline>)
        return GemmPipeline::IsMathWave();
    else
        return true;
}

// Run the epilogue with wavelet load/math wave dispatch. For wavelet pipelines only the math
// waves run @p epilogue_body (which writes the C tile); load waves run a matching barrier
// sequence (RunBarrierStub) to avoid an LDS-sync deadlock. Non-wavelet pipelines run
// @p epilogue_body directly. The body is kernel-specific (split-K dispatch, window
// construction), so it is passed in rather than shared.
template <typename GemmPipeline, typename EpiloguePipeline, typename EpilogueBody>
CK_TILE_DEVICE void RunWaveletAwareEpilogue(EpilogueBody&& epilogue_body)
{
    if constexpr(is_wavelet_pipeline<GemmPipeline>)
    {
        if(GemmPipeline::IsMathWave())
            epilogue_body();
        else
            EpiloguePipeline::RunBarrierStub();
    }
    else
    {
        epilogue_body();
    }
}

} // namespace ck_tile
