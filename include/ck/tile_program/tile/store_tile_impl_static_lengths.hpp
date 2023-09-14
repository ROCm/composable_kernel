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

namespace ck {
namespace tile_program {

template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          typename DataType_>
__device__ void
store_tile(TileWindowWithStaticLengths<BottomTensorView_, WindowLengths_>& tile_window_tmp,
           const StaticDistributedTensor<DataType_, TileDistribution_>& dstr_tensor)
{
    using DataType         = remove_cvref_t<typename BottomTensorView_::DataType>;
    using BottomTensorView = remove_cvref_t<BottomTensorView_>;
    using WindowLengths    = remove_cvref_t<WindowLengths_>;
    using TileDstr         = remove_cvref_t<TileDistribution_>;
    using TileWindow = TileWindowWithStaticDistribution<BottomTensorView, WindowLengths, TileDstr>;

    static_assert(is_same_v<remove_cvref_t<DataType_>, DataType>, "wrong!");

    constexpr auto tile_dstr = TileDstr{};

    auto tile_window = make_tile_window(tile_window_tmp.GetBottomTensorView(),
                                        tile_window_tmp.GetWindowLengths(),
                                        tile_window_tmp.GetWindowOrigin(),
                                        tile_dstr);

    constexpr auto thread_tensor_lengths_ys =
        to_sequence(tile_dstr.GetYs2DDescriptor().GetLengths());

    constexpr index_t NDimP = TileDstr::GetNumOfDimensionP();
    constexpr index_t NDimY = TileDstr::GetNumOfDimensionY();

    constexpr auto tmp = []() {
        const auto [ys_vector_lengths, ys_vector_strides] =
            TileWindow::GetWindowAdaptorYsSafeVectorLengthStrides();

        index_t VectorDimY      = 0;
        index_t ScalarPerVector = 1;

        for(index_t i = 0; i < NDimY; ++i)
        {
            if(ys_vector_strides[i] == 1 && ys_vector_lengths[i] > ScalarPerVector)
            {
                ScalarPerVector = ys_vector_lengths[i];
                VectorDimY      = i;
            }
        }

        return make_tuple(VectorDimY, ScalarPerVector);
    }();

    constexpr index_t VectorDimY      = tmp.template At<0>();
    constexpr index_t ScalarPerVector = tmp.template At<1>();

    // FIXME:
    using DimAccessOrder = typename arithmetic_sequence_gen<0, NDimY, 1>::type;

    constexpr auto scalars_per_access_arr = generate_array(
        [&](auto i) { return (i == VectorDimY) ? ScalarPerVector : 1; }, Number<NDimY>{});

    constexpr auto scalars_per_access = TO_SEQUENCE(scalars_per_access_arr, NDimY);

    using vector_type_t = vector_type_maker_t<DataType, ScalarPerVector>;
    using vector_t      = typename vector_type_t::type;

    using SFC_Ys = SpaceFillingCurve<decltype(thread_tensor_lengths_ys),
                                     DimAccessOrder,
                                     decltype(scalars_per_access)>;

    constexpr index_t num_access = SFC_Ys::GetNumOfAccess();

    static_assert(num_access > 0, "wrong! num_access should be larger than 0");

    // loop over thread tensor space [y0, y1, ...]
    static_for<0, num_access, 1>{}([&](auto iAccess) {
        // data index [y0, y1, ...]
        constexpr auto idx_ys_start = SFC_Ys::GetIndex(iAccess);

        // read from distributed tensor
        vector_type_t vec;

        static_for<0, ScalarPerVector, 1>{}([&](auto j) {
            constexpr auto idx_ys = generate_array(
                [&](auto jj) {
                    return jj == VectorDimY ? (idx_ys_start[jj] + j) : idx_ys_start[jj];
                },
                Number<NDimY>{});

            constexpr index_t d = tile_dstr.GetYs2DDescriptor().CalculateOffset(idx_ys);

            vec.template AsType<DataType>()(j) = dstr_tensor.GetThreadBuffer().template At<d>();
        });

        const vector_t vec_value = vec.template AsType<vector_t>().template At<0>();

        // write into bottom tensor
        tile_window.GetBottomTensorView().template SetVectorizedElements<vector_t>(
            tile_window.GetBottomTensorThreadCoordinate(), vec_value);

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
}

} // namespace tile_program
} // namespace ck
