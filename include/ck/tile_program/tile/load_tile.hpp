// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_description/tensor_space_filling_curve.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_window.hpp"
#include "ck/tile_program/tile/static_distributed_tensor.hpp"

namespace ck {
namespace tile_program {

// detail used by tile-programming APIs(), not supposed to be used directly
namespace detail {

// "Y dimension": Y dimensions inside TileWindowWithStaticDistribution
// input:
//   y_slice_origin: starting slice origin of Y dimension
//   y_slice_lengths: slice lengths of Y dimensionr
// output:
//   A StaticBuffer holding slice of thread data, and data layout is hardcoded to be in the order of
//   [Y0, Y1, Y2, ...]
template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          typename YIndex,
          index_t... YSliceLengths>
__device__ auto load_sliced_thread_data_from_tile_window(
    TileWindowWithStaticDistribution<BottomTensorView_, WindowLengths_, TileDistribution_>&
        tile_window,
    const YIndex& ys_slice_origin,
    Sequence<YSliceLengths...>)
{
    using DataType         = remove_cvref_t<typename BottomTensorView_::DataType>;
    using BottomTensorView = remove_cvref_t<BottomTensorView_>;
    using WindowLengths    = remove_cvref_t<WindowLengths_>;
    using TileDstr         = remove_cvref_t<TileDistribution_>;
    using TileWindow = TileWindowWithStaticDistribution<BottomTensorView, WindowLengths, TileDstr>;

    constexpr auto tile_dstr = TileDstr{};

    constexpr index_t NDimP = TileDstr::GetNumOfDimensionP();
    constexpr index_t NDimY = TileDstr::GetNumOfDimensionY();

    static_assert(NDimY == YIndex::Size() && NDimY == sizeof...(YSliceLengths),
                  "wrong! inconsistent # of dimension");

    static_assert(TileWindow::HasStaticTileDistribution(),
                  "wrong! assume static tile distribution");

    constexpr auto y_slice_lengths = Sequence<YSliceLengths...>{};

    constexpr index_t thread_element_size =
        container_reduce(y_slice_lengths, math::multiplies{}, 1);

    StaticBuffer<AddressSpaceEnum::Vgpr, DataType, thread_element_size, true> thread_buf;

    constexpr auto tmp = [&y_slice_lengths]() {
        const auto [ys_vector_lengths, ys_vector_strides] =
            TileWindow::GetWindowAdaptorYsSafeVectorLengthStrides();

        index_t VectorDimY      = 0;
        index_t ScalarPerVector = 1;

        for(index_t i = 0; i < NDimY; ++i)
        {
            if(ys_vector_strides[i] == 1 && ys_vector_lengths[i] > ScalarPerVector)
            {
                ScalarPerVector = math::gcd(ys_vector_lengths[i], y_slice_lengths[i]);
                VectorDimY      = i;
            }
        }

        return make_tuple(VectorDimY, ScalarPerVector);
    }();

    constexpr index_t VectorDimY      = tmp.template At<0>();
    constexpr index_t ScalarPerVector = tmp.template At<1>();

    // FIXME
    using DimAccessOrder = typename arithmetic_sequence_gen<0, NDimY, 1>::type;

    constexpr auto scalars_per_access_arr = generate_array(
        [&](auto i) { return (i == VectorDimY) ? ScalarPerVector : 1; }, Number<NDimY>{});

    constexpr auto scalars_per_access = TO_SEQUENCE(scalars_per_access_arr, NDimY);

    using vector_type_t = vector_type_maker_t<DataType, ScalarPerVector>;
    using vector_t      = typename vector_type_t::type;

    using SFC_Ys =
        SpaceFillingCurve<decltype(y_slice_lengths), DimAccessOrder, decltype(scalars_per_access)>;

    constexpr index_t num_access = SFC_Ys::GetNumOfAccess();

    static_assert(num_access > 0, "wrong! num_access should be larger than 0");

    // move to slice origin
    const auto ps_ys_slice_origin = container_concat(Array<index_t, NDimP>{0}, ys_slice_origin);

    tile_window.MoveWindowAdaptorAndBottomTensorThreadCoordinate(ps_ys_slice_origin);

    // loop over thread tensor space [y0, y1, ...]
    static_for<0, num_access, 1>{}([&](auto iAccess) {
        // read from bottom tensor
        const vector_t vec_value =
            tile_window.GetBottomTensorView().template GetVectorizedElements<vector_t>(
                tile_window.GetBottomTensorThreadCoordinate());

        const vector_type_t vec{vec_value};

        // data index [y0, y1, ...]
        constexpr auto idx_ys_start = SFC_Ys::GetIndex(iAccess);

        // write into distributed tensor
        static_for<0, ScalarPerVector, 1>{}([&](auto j) {
            constexpr auto idx_ys = generate_array(
                [&](auto jj) {
                    return jj == VectorDimY ? (idx_ys_start[jj] + j) : idx_ys_start[jj];
                },
                Number<NDimY>{});

            constexpr index_t d = tile_dstr.GetYs2DDescriptor().CalculateOffset(idx_ys);

            thread_buf.template At<d>() = vec.template AsType<DataType>()[j];
        });

        // move thread coordinate
        if constexpr(iAccess.value != num_access - 1)
        {
            constexpr auto idx_diff_ys = SFC_Ys::GetForwardStep(iAccess);

            constexpr auto idx_diff_ps_ys = container_concat(Array<index_t, NDimP>{0}, idx_diff_ys);

            tile_window.MoveWindowAdaptorAndBottomTensorThreadCoordinate(idx_diff_ps_ys);
        }
    });

    // move thread coordinate back to origin
    {
        constexpr auto idx_diff_ys = SFC_Ys::GetStepBetween(Number<num_access - 1>{}, Number<0>{});

        constexpr auto idx_diff_ps_ys = container_concat(Array<index_t, NDimP>{0}, idx_diff_ys);

        tile_window.MoveWindowAdaptorAndBottomTensorThreadCoordinate(idx_diff_ps_ys);
    }

    // move back to origin
    tile_window.MoveWindowAdaptorAndBottomTensorThreadCoordinate(MultiIndex<NDimP + NDimY>{0} -
                                                                 ps_ys_slice_origin);

    return thread_buf;
}

} // namespace detail

// FIXME: host dummy function for tile program
template <typename BottomTensorView_, typename WindowLengths_, typename TileDistribution_>
__host__ auto load_tile(
    const TileWindowWithStaticDistribution<BottomTensorView_, WindowLengths_, TileDistribution_>&
        tile_window)
{
    using DataType         = remove_cvref_t<typename BottomTensorView_::DataType>;
    using BottomTensorView = remove_cvref_t<BottomTensorView_>;
    using WindowLengths    = remove_cvref_t<WindowLengths_>;
    using TileDstr         = remove_cvref_t<TileDistribution_>;
    using TileWindow = TileWindowWithStaticDistribution<BottomTensorView, WindowLengths, TileDstr>;

    static_assert(is_known_at_compile_time<WindowLengths>::value,
                  "wrong! lengths should be static");
    static_assert(TileWindow::HasStaticTileDistribution(), "wrong!");

    return make_static_distributed_tensor<DataType>(tile_window.GetTileDistribution());
}

template <typename BottomTensorView_, typename WindowLengths_, typename TileDistribution_>
__device__ auto
load_tile(TileWindowWithStaticDistribution<BottomTensorView_, WindowLengths_, TileDistribution_>&
              tile_window)
{
    using DataType         = remove_cvref_t<typename BottomTensorView_::DataType>;
    using BottomTensorView = remove_cvref_t<BottomTensorView_>;
    using WindowLengths    = remove_cvref_t<WindowLengths_>;
    using TileDstr         = remove_cvref_t<TileDistribution_>;
    using TileWindow = TileWindowWithStaticDistribution<BottomTensorView, WindowLengths, TileDstr>;

    static_assert(is_known_at_compile_time<WindowLengths>::value,
                  "wrong! lengths should be static");
    static_assert(TileWindow::HasStaticTileDistribution(), "wrong!");

    constexpr auto tile_dstr = TileDstr{};

    constexpr index_t NDimY = tile_dstr.GetYs2DDescriptor().GetNumOfDimension();

    auto dstr_tensor = make_static_distributed_tensor<DataType>(tile_dstr);

    dstr_tensor.GetThreadBuffer() = detail::load_sliced_thread_data_from_tile_window(
        tile_window, MultiIndex<NDimY>{0}, to_sequence(tile_dstr.GetYs2DDescriptor().GetLengths()));

    return dstr_tensor;
}

} // namespace tile_program
} // namespace ck
