// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file benchmark_cshuffle_lds.hpp
 * @brief LDS benchmark setup for CShuffleEpilogue.
 *
 * Provides Setup adapters that extract LDS descriptor and distribution
 * from CShuffleEpilogue for use with generic tile benchmark kernels.
 */

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/utility/tile_load_store_microkernels.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"

namespace ck_tile {

/**
 * @brief Create CShuffleEpilogue type from benchmark parameters.
 */
template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename ODataType,
          index_t kM,
          index_t kN,
          index_t MWave,
          index_t NWave,
          index_t MPerXdl,
          index_t NPerXdl,
          index_t KPerXdl>
using BenchmarkEpilogue = CShuffleEpilogue<CShuffleEpilogueProblem<ADataType,
                                                                   BDataType,
                                                                   tuple<>,
                                                                   AccDataType,
                                                                   ODataType,
                                                                   tuple<>,
                                                                   tensor_layout::gemm::RowMajor,
                                                                   element_wise::PassThrough,
                                                                   kM,
                                                                   kN,
                                                                   MWave,
                                                                   NWave,
                                                                   MPerXdl,
                                                                   NPerXdl,
                                                                   KPerXdl,
                                                                   false>>;

/**
 * @brief Setup for LDS store benchmark - adapts CShuffleEpilogue for tile benchmark.
 */
template <typename Epilogue>
struct LdsStoreSetup
{
    using ODataType                     = typename Epilogue::ODataType;
    static constexpr index_t kBlockSize = Epilogue::kBlockSize;
    static constexpr index_t kBytes =
        Epilogue::MPerIterationShuffle * Epilogue::NPerIterationShuffle * sizeof(ODataType);
    static constexpr auto lds_desc =
        Epilogue::template MakeLdsBlockDescriptor<typename Epilogue::Problem>();
    static constexpr auto distr =
        make_static_tile_distribution(Epilogue::MakeLdsDistributionEncode());

    CK_TILE_DEVICE static auto create()
    {
        alignas(16) __shared__ char smem[Epilogue::GetSmemSize()];

        auto lds_view =
            make_tensor_view<address_space_enum::lds>(reinterpret_cast<ODataType*>(smem), lds_desc);

        auto window = make_tile_window(lds_view,
                                       make_tuple(number<Epilogue::MPerIterationShuffle>{},
                                                  number<Epilogue::NPerIterationShuffle>{}),
                                       {0, 0},
                                       distr);

        auto tile = make_static_distributed_tensor<ODataType>(distr);

        return make_tuple(window, tile);
    }
};

/**
 * @brief Setup for LDS load benchmark - adapts CShuffleEpilogue for tile benchmark.
 */
template <typename Epilogue>
struct LdsLoadSetup
{
    using ODataType                     = typename Epilogue::ODataType;
    static constexpr index_t kBlockSize = Epilogue::kBlockSize;
    static constexpr index_t kBytes =
        Epilogue::MPerIterationShuffle * Epilogue::NPerIterationShuffle * sizeof(ODataType);
    static constexpr auto lds_desc =
        Epilogue::template MakeLdsBlockDescriptor<typename Epilogue::Problem>();

    using ReadPattern =
        tile_distribution_encoding_pattern_2d<Epilogue::kBlockSize,
                                              Epilogue::MPerIterationShuffle,
                                              Epilogue::NPerIterationShuffle,
                                              Epilogue::GetVectorSizeC(),
                                              tile_distribution_pattern::thread_raked>;
    static constexpr auto read_distr = ReadPattern::make_2d_static_tile_distribution();

    CK_TILE_DEVICE static auto create()
    {
        alignas(16) __shared__ char smem[Epilogue::GetSmemSize()];

        auto lds_view =
            make_tensor_view<address_space_enum::lds>(reinterpret_cast<ODataType*>(smem), lds_desc);

        return make_tile_window(lds_view,
                                make_tuple(number<Epilogue::MPerIterationShuffle>{},
                                           number<Epilogue::NPerIterationShuffle>{}),
                                {0, 0},
                                read_distr);
    }
};

} // namespace ck_tile
