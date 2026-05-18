// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/utility.hpp"
#include "ck_tile/core/arch/amd_buffer_addressing.hpp"
#include "ck_tile/core/algorithm/space_filling_curve.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/tensor/static_distributed_tensor.hpp"
#include "ck_tile/core/tensor/tensor_adaptor.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

/**
 * @brief This class provides tile (windowed) view and access to the device memory.
 *
 * @note This tile window does not support single issue you need to use tile_window_linear
 *       structure for this purpose
 *
 * @tparam BottomTensorView_        Class describing & holding device tensor memory.
 * @tparam WindowLengths_           Spatial sizes of windowed view on tensor.
 * @tparam StaticTileDistribution_  Thread distribution (mapping) into Tile dimensions.
 * @tparam StaticPageIndexArray_    Array type holding page indices for scatter/gather.
 * @tparam StaticValidArray_        Array type holding validity flags (nullptr_t if unused).
 * @tparam HsGatherDim              H-space dimension index used for gather lookup (default: 0).
 * @tparam NumCoord                 Number of pre-computed coordinates for pipelining (default: 1).
 * @tparam YsGatherDims             Sequence of Y-space dimension indices used for page lookup.
 *                                  For single dimension: sequence<0> (default).
 *                                  For multiple dimensions: sequence<dim0, dim1, ...> where
 *                                  the combined index is computed as:
 *                                  idx[dim0] + idx[dim1] * len[dim0] + idx[dim2] * len[dim0] *
 * len[dim1] + ...
 */
template <typename BottomTensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          typename StaticPageIndexArray_,
          typename StaticValidArray_,
          index_t HsGatherDim   = 0,
          index_t NumCoord      = 1,
          typename YsGatherDims = sequence<0>>
struct tile_scatter_gather
{
    using BottomTensorView = remove_reference_t<BottomTensorView_>;
    using WindowLengths    = remove_cvref_t<WindowLengths_>;
    using TileDstr         = remove_cvref_t<StaticTileDistribution_>;
    using PageIdxArray     = remove_cvref_t<StaticPageIndexArray_>;
    using ValidArray       = remove_cvref_t<StaticValidArray_>;
    using WindowAdaptor    = typename TileDstr::PsYs2XsAdaptor;
    using BottomTensorDesc = typename BottomTensorView::TensorDesc;

    using DataType = remove_cvref_t<typename BottomTensorView::DataType>;

    static constexpr index_t NDimWindowAdaptorTop = WindowAdaptor::get_num_of_top_dimension();
    static constexpr index_t NDimBottomTensor     = BottomTensorDesc::get_num_of_dimension();

    static constexpr index_t NDimP = TileDstr::get_num_of_dimension_p();
    static constexpr index_t NDimY = TileDstr::get_num_of_dimension_y();

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static_assert(NumCoord == 1);

    // TODO: check WindowLengths and StaticTileDistribution are consistent

    static_assert(ck_tile::is_known_at_compile_time<WindowLengths>::value,
                  "wrong! lengths should be static");
    static_assert(TileDstr::is_static(), "wrong!");

    static_assert(NDimBottomTensor == WindowAdaptor::get_num_of_bottom_dimension(),
                  "wrong! inconsistent # of diemsnions");

    using AdaptorTopIndex   = array<index_t, NDimWindowAdaptorTop>;
    using BottomTensorIndex = array<index_t, NDimBottomTensor>;

    using WindowAdaptorCoord =
        decltype(make_tensor_adaptor_coordinate(WindowAdaptor{}, AdaptorTopIndex{}));

    using BottomTensorCoord =
        decltype(make_tensor_coordinate(BottomTensorDesc{}, BottomTensorIndex{}));

    /**
     * @brief Check if a given Y-space dimension index is a gather dimension.
     *
     * Gather dimensions are those specified in YsGatherDims template parameter.
     * When computing forward_step_scatter, gather dimensions are set to 0
     * because page offset lookup handles address calculation for these dimensions.
     *
     * @param i Y-space dimension index to check
     * @return true if dimension i is in YsGatherDims, false otherwise
     */
    CK_TILE_DEVICE static constexpr bool is_gather_dim(index_t i)
    {
        return sequence_any_of(YsGatherDims{}, [i](auto k) { return i == k; });
    }

    /**
     * @brief Compute the linearized gather index from Y-space indices for page lookup.
     *
     * This function converts multi-dimensional Y-space indices (specified by YsGatherDims)
     * into a single linearized index used to look up the page offset in page_idx_ array.
     *
     * For single gather dimension (YsGatherDims::size() == 1):
     *   Simply returns idx_ys_start[YsGatherDims::at(0)]
     *
     * For multiple gather dimensions (e.g., YsGatherDims = sequence<0, 2>):
     *   Computes: idx[dim0] + idx[dim1] * len[dim0] + idx[dim2] * len[dim0] * len[dim1] + ...
     *   This is row-major linearization where earlier dimensions are inner (faster-varying).
     *
     * @tparam YsIndex Type of the Y-space index tuple/array
     * @param idx_ys_start Current Y-space indices from space-filling curve iteration
     * @return Linearized index for page_idx_ array lookup
     */
    template <typename YsIndex>
    CK_TILE_DEVICE static constexpr auto get_gather_index(const YsIndex& idx_ys_start)
    {
        // TODO: Consider making ys_lengths_ part of public API or adding accessor
        static_assert(sizeof(TileDstr::DstrEncode::detail::ys_lengths_) > 0,
                      "Relies on internal detail::ys_lengths_");

        constexpr index_t num_gather_dims = YsGatherDims::size();

        if constexpr(num_gather_dims == 1)
        {
            return idx_ys_start[number<YsGatherDims::at(0)>{}];
        }
        else
        {
            // Recursive lambda to compute index as a compile-time number
            // Uses row-major linearization: idx[0] + idx[1] * len[0] + idx[2] * len[0] * len[1] +
            // ...
            auto recurse = [&](auto self, auto i_constant) {
                constexpr index_t i   = decltype(i_constant)::value;
                constexpr index_t dim = YsGatherDims::at(i);
                auto current_val      = idx_ys_start[number<dim>{}];

                if constexpr(i + 1 < num_gather_dims)
                {
                    constexpr index_t len = TileDstr::DstrEncode::detail::ys_lengths_[dim];
                    return current_val + self(self, number<i + 1>{}) * number<len>{};
                }
                else
                {
                    return current_val;
                }
            };
            return recurse(recurse, number<0>{});
        }
    }

    struct load_store_traits
    {
        private:
        static constexpr auto get_vector_dim_y_scalar_per_vector()
        {
            const auto [ys_vector_lengths, ys_vector_strides] =
                tile_scatter_gather::get_window_adaptor_ys_safe_vector_length_strides();

            index_t VectorDimY_      = 0;
            index_t ScalarPerVector_ = 1;

            for(index_t i = 0; i < NDimY; ++i)
            {
                if(ys_vector_strides[i] == 1 && ys_vector_lengths[i] > ScalarPerVector_)
                {
                    ScalarPerVector_ = ys_vector_lengths[i];
                    VectorDimY_      = i;
                }
            }

            return make_tuple(VectorDimY_, ScalarPerVector_);
        }

        public:
        static constexpr index_t PackedSize =
            ck_tile::numeric_traits<remove_cvref_t<DataType>>::PackedSize;
        static constexpr index_t VectorDimY = get_vector_dim_y_scalar_per_vector().template at<0>();
        static constexpr index_t ScalarPerVector =
            get_vector_dim_y_scalar_per_vector().template at<1>();

        // using vector_type_t = vector_type_maker_t<DataType, ScalarPerVector>;
        // using vector_t      = typename vector_type_t::type;
        using vector_t = thread_buffer<DataType, ScalarPerVector / PackedSize>;

        private:
        static constexpr auto scalars_per_access_ = [] {
            constexpr auto scalars_per_access_arr = generate_array(
                [&](auto i) { return (i == VectorDimY) ? ScalarPerVector : 1; }, number<NDimY>{});

            /// TODO: add non-automatic storage argument support to macro TO_SEQUENCE()
            constexpr auto NDimY_ = NDimY;

            return TO_SEQUENCE(scalars_per_access_arr, NDimY_);
        }();

        static constexpr auto get_space_filling_curve()
        {
            constexpr auto tile_dstr = TileDstr{};

            constexpr auto thread_tensor_lengths_ys =
                to_sequence(tile_dstr.get_ys_to_d_descriptor().get_lengths());

            // FIXME: need logic to judge dim access order
            using DimAccessOrder = typename arithmetic_sequence_gen<0, NDimY, 1>::type;

            return space_filling_curve<decltype(thread_tensor_lengths_ys),
                                       DimAccessOrder,
                                       decltype(scalars_per_access_)>{};
        }

        public:
        using SFC_Ys = decltype(get_space_filling_curve());

        static constexpr index_t NumAccess = SFC_Ys::get_num_of_access();

        static_assert(0 < NumAccess, "Wrong! NumAccess should be larger than 0");
        static_assert(NumAccess % NumCoord == 0, "wrong! # of access is not divisible by NumCoord");
    };

    static constexpr index_t NumAccessPerCoord = load_store_traits::NumAccess / NumCoord;

    CK_TILE_DEVICE constexpr tile_scatter_gather() = default;

    CK_TILE_DEVICE constexpr tile_scatter_gather(const BottomTensorView& bottom_tensor_view,
                                                 const WindowLengths& window_lengths,
                                                 const BottomTensorIndex& window_origin,
                                                 const TileDstr& tile_distribution,
                                                 const PageIdxArray& page_idx,
                                                 const ValidArray& valids)
        : bottom_tensor_view_{bottom_tensor_view},
          window_lengths_{window_lengths},
          window_origin_{window_origin},
          tile_dstr_{tile_distribution},
          page_idx_{page_idx},
          valids_{valids},
          pre_computed_coords_{}
    {
#if 0 // debug
      // TODO: this use more register for FA, but less register for GEMM
      // need investigation
      // only support warp-tile and block-tile
        static_assert(NDimP == 1 or NDimP == 2, "wrong!");

        WindowAdaptorCoord window_adaptor_thread_coord_tmp;

        if constexpr(NDimP == 1)
        {
            window_adaptor_thread_coord_tmp = make_tensor_adaptor_coordinate(
                tile_distribution.get_ps_ys_to_xs_adaptor(), AdaptorTopIndex{get_lane_id(), 0});
        }
        else if constexpr(NDimP == 2)
        {
            window_adaptor_thread_coord_tmp =
                make_tensor_adaptor_coordinate(tile_distribution.get_ps_ys_to_xs_adaptor(),
                                               AdaptorTopIndex{get_warp_id(), get_lane_id(), 0});
        }
#else
        // TODO: this use less register for FA, but more register for GEMM
        // need investigation
        const auto window_adaptor_thread_coord_tmp = make_tensor_adaptor_coordinate(
            tile_distribution.get_ps_ys_to_xs_adaptor(),
            container_concat(get_partition_index(tile_distribution), array<index_t, NDimY>{0}));
#endif

        BottomTensorIndex bottom_tensor_thread_origin_idx_tmp =
            window_origin + window_adaptor_thread_coord_tmp.get_bottom_index();
        bottom_tensor_thread_origin_idx_tmp(HsGatherDim) = 0;
        const auto bottom_tensor_thread_coord_tmp        = make_tensor_coordinate(
            bottom_tensor_view_.get_tensor_descriptor(), bottom_tensor_thread_origin_idx_tmp);

        // pre-compute NumCoord (WindowAdaptorCoord, BottomTensorCoord) bundles to speed up
        // future load/store() calls (might allocate more registers)
        using Traits = load_store_traits;
        using SFC_Ys = typename Traits::SFC_Ys;

        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            auto window_adaptor_thread_coord = window_adaptor_thread_coord_tmp;
            auto bottom_tensor_thread_coord  = bottom_tensor_thread_coord_tmp;

            constexpr auto idx_diff_ys =
                SFC_Ys::get_step_between(number<0>{}, number<iCoord * NumAccessPerCoord>{});

            constexpr auto idx_diff_ps_ys = container_concat(
                generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}), idx_diff_ys);

            move_window_adaptor_and_bottom_tensor_thread_coordinate(
                window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);

            pre_computed_coords_(iCoord) =
                make_tuple(window_adaptor_thread_coord, bottom_tensor_thread_coord);
        });
        if constexpr(BottomTensorView::buffer_view::get_address_space() ==
                     address_space_enum::global)
        {
            auto partition_index = get_partition_index(tile_distribution);

            auto use_lane_id_0                              = partition_index;
            use_lane_id_0[1]                                = 0;
            const auto window_adaptor_thread_coord_tmp_warp = make_tensor_adaptor_coordinate(
                tile_distribution.get_ps_ys_to_xs_adaptor(),
                container_concat(use_lane_id_0, array<index_t, NDimY>{0}));

            BottomTensorIndex bottom_tensor_thread_origin_idx_tmp_warp =
                window_origin + window_adaptor_thread_coord_tmp_warp.get_bottom_index();
            bottom_tensor_thread_origin_idx_tmp_warp(HsGatherDim) = 0;
            const auto bottom_tensor_thread_coord_tmp_warp =
                make_tensor_coordinate(bottom_tensor_view_.get_tensor_descriptor(),
                                       bottom_tensor_thread_origin_idx_tmp_warp);

            // pre-compute NumCoord (WindowAdaptorCoord, BottomTensorCoord) bundles to speed up
            // future load/store() calls (might allocate more registers)
            static_for<0, NumCoord, 1>{}([&](auto iCoord) {
                auto window_adaptor_thread_coord = window_adaptor_thread_coord_tmp_warp;
                auto bottom_tensor_thread_coord  = bottom_tensor_thread_coord_tmp_warp;

                constexpr auto idx_diff_ys =
                    SFC_Ys::get_step_between(number<0>{}, number<iCoord * NumAccessPerCoord>{});

                constexpr auto idx_diff_ps_ys = container_concat(
                    generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                    idx_diff_ys);

                move_window_adaptor_and_bottom_tensor_thread_coordinate(
                    window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);

                pre_computed_warp_coords_(iCoord) =
                    make_tuple(window_adaptor_thread_coord, bottom_tensor_thread_coord);
            });
        }
    }

    CK_TILE_DEVICE static constexpr index_t get_num_of_dimension() { return NDimBottomTensor; }

    CK_TILE_DEVICE static constexpr bool has_static_tile_distribution()
    {
        return TileDstr::is_static();
    }

    CK_TILE_DEVICE constexpr auto get_window_lengths() const { return window_lengths_; }

    CK_TILE_DEVICE constexpr auto get_tile_distribution() const { return tile_dstr_; }

    CK_TILE_DEVICE constexpr auto get_bottom_tensor_view() const { return bottom_tensor_view_; }

    CK_TILE_DEVICE constexpr auto get_window_origin() const { return window_origin_; }

    CK_TILE_DEVICE constexpr void
    set_bottom_tensor_view_data_ptr(typename BottomTensorView::DataType* data)
    {
        bottom_tensor_view_.buf_.p_data_ = data;
    }

    // move thread's window adaptor coordinate and bottom tensor coordinate
    // [p0, p1, ..., y0, y1, ...] ==> [x0, x1, ...] ==> [x0', x1', ...] ==> [offset]
    template <typename ATopIndex>
    CK_TILE_DEVICE void move_window_adaptor_and_bottom_tensor_thread_coordinate(
        WindowAdaptorCoord& window_adaptor_thread_coord,
        BottomTensorCoord& bottom_tensor_thread_coord,
        const ATopIndex& idx_diff_adaptor_top) const
    {
        array<index_t, NDimBottomTensor> idx_diff_adaptor_bottom;

        move_tensor_adaptor_coordinate(tile_dstr_.get_ps_ys_to_xs_adaptor(),
                                       window_adaptor_thread_coord,
                                       idx_diff_adaptor_top,
                                       idx_diff_adaptor_bottom);

        move_tensor_coordinate(bottom_tensor_view_.get_tensor_descriptor(),
                               bottom_tensor_thread_coord,
                               idx_diff_adaptor_bottom);
    }

    // return vector dimension among [y0, y1, ...]
    CK_TILE_DEVICE static constexpr auto get_window_adaptor_ys_safe_vector_length_strides()
    {
        // bottom tensor top dimension vector lengths and strides
        const auto [bottom_tensor_top_dim_vector_lengths, bottom_tensor_top_dim_vector_strides] =
            BottomTensorDesc::get_top_dimension_safe_vector_length_strides();

        // window vector lengths/strides
        const auto window_adaptor_bottom_dim_vector_lengths = bottom_tensor_top_dim_vector_lengths;
        const auto window_adaptor_bottom_dim_vector_strides = bottom_tensor_top_dim_vector_strides;

        // window adaptor [p0, p1, ..., y0, y1, ...]
        array<index_t, WindowAdaptor::get_num_of_hidden_dimension()> window_adaptor_vector_lengths{
            -1};
        array<index_t, WindowAdaptor::get_num_of_hidden_dimension()> window_adaptor_vector_strides{
            -1};

        constexpr auto window_adaptor_bottom_dims =
            WindowAdaptor::get_bottom_dimension_hidden_ids();

        set_container_subset(window_adaptor_vector_lengths,
                             window_adaptor_bottom_dims,
                             window_adaptor_bottom_dim_vector_lengths);
        set_container_subset(window_adaptor_vector_strides,
                             window_adaptor_bottom_dims,
                             window_adaptor_bottom_dim_vector_strides);

        const auto [window_adaptor_ps_ys_vector_lengths, window_adaptor_ps_ys_vector_strides] =
            WindowAdaptor{}.get_top_dimension_safe_vector_length_strides(
                window_adaptor_vector_lengths, window_adaptor_vector_strides);

        // [y0, y1, ...]
        constexpr auto y_dims = typename arithmetic_sequence_gen<TileDstr::get_num_of_dimension_p(),
                                                                 NDimWindowAdaptorTop,
                                                                 1>::type{};

        return make_tuple(get_container_subset(window_adaptor_ps_ys_vector_lengths, y_dims),
                          get_container_subset(window_adaptor_ps_ys_vector_strides, y_dims));
    }

    CK_TILE_DEVICE constexpr auto get_num_of_access() const { return load_store_traits::NumAccess; }

    template <index_t i_access_unsupport_ = -1, bool oob_conditional_check = true>
    CK_TILE_DEVICE auto load(number<i_access_unsupport_>          = {},
                             bool_constant<oob_conditional_check> = {}) const
    {
        constexpr auto tile_dstr = TileDstr{};
        auto dst_tensor          = make_static_distributed_tensor<DataType>(tile_dstr);
        load(dst_tensor, number<i_access_unsupport_>{}, bool_constant<oob_conditional_check>{});
        return dst_tensor;
    }

    template <typename DistributedTensor,
              index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true>
    CK_TILE_DEVICE auto load(DistributedTensor& dst_tensor,
                             number<i_access_unsupport_>          = {},
                             bool_constant<oob_conditional_check> = {}) const
    {
        using Traits   = load_store_traits;
        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        constexpr auto tile_dstr = TileDstr{};

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);
                constexpr auto idx_gather   = get_gather_index(idx_ys_start);
                const auto page_offset      = page_idx_[idx_gather];

                // read from bottom tensor
                const vector_t vec_value = [&]() {
                    if constexpr(std::is_same_v<ValidArray, std::nullptr_t>)
                    {
                        return get_bottom_tensor_view().template get_vectorized_elements<vector_t>(
                            bottom_tensor_thread_coord,
                            page_offset,
                            bool_constant<oob_conditional_check>{});
                    }
                    else
                    {
                        return get_bottom_tensor_view().template get_vectorized_elements<vector_t>(
                            bottom_tensor_thread_coord,
                            page_offset,
                            valids_[idx_gather],
                            bool_constant<oob_conditional_check>{});
                    }
                }();
#if 1
                // write into distributed tensor
                static_for<0, Traits::ScalarPerVector, Traits::PackedSize>{}([&](auto j) {
                    constexpr auto idx_ys = generate_tuple(
                        [&](auto jj) {
                            return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                            : idx_ys_start[jj];
                        },
                        number<NDimY>{});

                    constexpr index_t d =
                        tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys) /
                        Traits::PackedSize;

                    dst_tensor.get_thread_buffer().template at<d>() =
                        vec_value.template get_as<DataType>()[j / Traits::PackedSize];
                });
#else
                constexpr index_t d =
                    tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys_start);
                static_assert(d % Traits::ScalarPerVector == 0);

                dst_tensor.get_thread_buffer().template get_as<vector_t>()(
                    number<d / Traits::ScalarPerVector>{}) = bit_cast<vector_t>(vec_value);
#endif
                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto forward_step_scatter = generate_tuple(
                        [&](auto i) { return is_gather_dim(i) ? 0 : idx_diff_ys[i]; },
                        number<NDimY>{});

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                        forward_step_scatter);

                    move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    template <typename LdsTileWindow_,
              index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true>
    CK_TILE_DEVICE auto async_load(LdsTileWindow_&& lds_tile,
                                   number<i_access_unsupport_>          = {},
                                   bool_constant<oob_conditional_check> = {}) const
    {
        using LdsTileWindow = remove_cvref_t<LdsTileWindow_>;
        using LdsDataType   = typename LdsTileWindow::DataType;
        using Traits        = load_store_traits;
        using vector_t      = typename Traits::vector_t;
        using SFC_Ys        = typename Traits::SFC_Ys;

        constexpr auto tile_dstr = TileDstr{};

        // Precompute invariant values outside loops
        const auto window_origin       = lds_tile.get_window_origin();
        const auto& bottom_tensor_view = lds_tile.get_bottom_tensor_view();
        const auto& tensor_descriptor  = bottom_tensor_view.get_tensor_descriptor();
        auto smem_base_ptr             = bottom_tensor_view.get_buffer_view().p_data_;

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            auto lds_window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto lds_bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // Use precomputed window origin
                auto lds_bottom_tensor_thread_idx =
                    window_origin + lds_window_adaptor_thread_coord.get_bottom_index();
                // Use precomputed tensor descriptor
                const auto lds_coord =
                    make_tensor_coordinate(tensor_descriptor, lds_bottom_tensor_thread_idx);
                // Calculate SMEM address using base pointer
                CK_TILE_LDS_ADDR LdsDataType* smem = smem_base_ptr + lds_coord.get_offset();

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);
                constexpr auto idx_gather   = get_gather_index(idx_ys_start);
                const auto page_offset      = page_idx_[idx_gather];

                // merge page_offset into bottom_coord
                auto mixed_bottom_thread_coord = bottom_tensor_thread_coord;
                mixed_bottom_thread_coord.get_hidden_index()[number<0>{}] += page_offset;

                // read from bottom tensor
                if constexpr(std::is_same_v<ValidArray, std::nullptr_t>)
                    this->get_bottom_tensor_view().template async_get_vectorized_elements<vector_t>(
                        smem,
                        mixed_bottom_thread_coord,
                        number<0>{},
                        bool_constant<oob_conditional_check>{});
                else
                    this->get_bottom_tensor_view().template async_get_vectorized_elements<vector_t>(
                        smem,
                        mixed_bottom_thread_coord,
                        number<0>{},
                        valids_[idx_gather],
                        bool_constant<oob_conditional_check>{});

                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto forward_step_scatter = generate_tuple(
                        [&](auto i) { return is_gather_dim(i) ? 0 : idx_diff_ys[i]; },
                        number<NDimY>{});

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                        forward_step_scatter);
                    // lds_diff doesn't need to mask the difference of the gather-dim.
                    constexpr auto lds_idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                        idx_diff_ys);

                    move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                    move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        lds_window_adaptor_thread_coord,
                        lds_bottom_tensor_thread_coord,
                        lds_idx_diff_ps_ys);
                }
            });
        });
    }

    // TODO: currently async load only implemented in inline asm
    template <typename LdsTileWindow_,
              index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true,
              bool pre_nop                = false>
    CK_TILE_DEVICE auto async_load_raw(LdsTileWindow_&& lds_tile,
                                       number<i_access_unsupport_>          = {},
                                       bool_constant<oob_conditional_check> = {},
                                       bool_constant<pre_nop>               = {}) const
    {
        using LdsTileWindow = remove_cvref_t<LdsTileWindow_>;
        // using LdsTensorView = typename LdsTileWindow::BottomTensorView;
        using LdsDataType = typename LdsTileWindow::DataType;
        // using LdsDescriptor = typename LdsTileWindow::BottomTensorDesc;

        // issues * warps * lanes
        static_assert(LdsTileWindow::get_num_of_dimension() == 3); // TODO: hard coded

        const index_t size_per_buf =
            lds_tile.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                make_tuple(number<0>{}, number<0>{}, number<0>{})) *
            sizeof(LdsDataType);

        const index_t size_per_wave =
            lds_tile.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                make_tuple(number<0>{}, number<1>{}, number<0>{})) *
                sizeof(LdsDataType) -
            size_per_buf;

        const index_t size_per_issue =
            lds_tile.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                make_tuple(number<1>{}, number<0>{}, number<0>{})) *
                sizeof(LdsDataType) -
            size_per_buf;

        const index_t m0_init_value = size_per_buf + size_per_wave * get_warp_id();
        m0_set_with_memory(
            amd_wave_read_first_lane(m0_init_value)); // This should be wave independent

        using Traits = load_store_traits;

        // using vector_type_t = typename Traits::vector_type_t;
        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        LdsDataType* smem = lds_tile.get_bottom_tensor_view().get_buffer_view().p_data_;

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess  = number<iCoord * NumAccessPerCoord + iCoordAccess>{};
                constexpr auto pre_nop_ = [&]() {
                    if constexpr(pre_nop && iCoord == 0 && iCoordAccess == 0)
                        return bool_constant<true>{};
                    else
                        return bool_constant<false>{};
                }();

                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);
                constexpr auto idx_gather   = get_gather_index(idx_ys_start);
                const auto page_offset      = page_idx_[idx_gather];

                // read from bottom tensor
                if constexpr(std::is_same_v<ValidArray, std::nullptr_t>)
                {
                    get_bottom_tensor_view().template async_get_vectorized_elements_raw<vector_t>(
                        smem, bottom_tensor_thread_coord, page_offset, 0, pre_nop_);
                }
                else
                {
                    get_bottom_tensor_view().template async_get_vectorized_elements_raw<vector_t>(
                        smem,
                        bottom_tensor_thread_coord,
                        page_offset,
                        valids_[idx_gather],
                        0,
                        pre_nop_);
                }

                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto forward_step_scatter = generate_tuple(
                        [&](auto i) { return is_gather_dim(i) ? 0 : idx_diff_ys[i]; },
                        number<NDimY>{});

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                        forward_step_scatter);

                    move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);

                    m0_inc_with_memory(size_per_issue);
                }
            });
        });
    }

    // ------------------------------------------------------------------
    // Variant of async_load_raw that lazily re-anchors the wave-uniform SRD
    // base pointer so per-lane voffsets stay within int32 range even when
    // the total cache pool exceeds 4 GB. For every load issue:
    //
    //   1. read the per-lane absolute page offset (long_index_t, in
    //      elements of DataType);
    //   2. take lane-0's value as a wave-uniform anchor candidate via
    //      amd_wave_read_first_lane();
    //   3. if (wave_anchor - cur_anchor_) is outside [0, kRebaseThreshold)
    //      shift the SRD base pointer to p_data_orig_ + wave_anchor and
    //      reinit the buffer resource; update cur_anchor_ accordingly;
    //   4. issue the buffer_load with voffset = (lane_page_offset -
    //      cur_anchor_), which is guaranteed to fit in int32 (after the
    //      *sizeof(T) byte scaling inside amd_async_buffer_load_with_oob_raw).
    //
    // Correctness precondition: within a single issue every lane of the
    // wave must map to the same physical page block, i.e.
    //   WaveSpanInN <= runtime page_size
    // Under this precondition the per-lane spread relative to the
    // wave-uniform anchor is bounded by page_size * row_stride * sizeof(T),
    // which fits comfortably in the half-INT32 element-window we leave
    // (kRebaseThreshold below). When the precondition does not hold use
    // async_load_raw_long instead.
    //
    // Fast path (no overflow this issue): one wave-read, one 64-bit
    // subtract, one compare-branch. Branch is wave-uniform; rebase rate is
    // low so the branch is well predicted by the SIMD scheduler.
    //
    // This method is non-const because it mutates bottom_tensor_view_
    // (rebase) and cur_anchor_ (anchor tracking). Use after
    // init_raw_lazy_rebase().
    template <typename LdsTileWindow_,
              index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true,
              bool pre_nop                = false>
    CK_TILE_DEVICE auto async_load_raw_lazy_rebase(
        LdsTileWindow_&& lds_tile,
        number<i_access_unsupport_>          = {},
        bool_constant<oob_conditional_check> = {},
        bool_constant<pre_nop>               = {})
    {
        using LdsTileWindow = remove_cvref_t<LdsTileWindow_>;
        using LdsDataType   = typename LdsTileWindow::DataType;

        // issues * warps * lanes
        static_assert(LdsTileWindow::get_num_of_dimension() == 3); // TODO: hard coded

        const index_t size_per_buf =
            lds_tile.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                make_tuple(number<0>{}, number<0>{}, number<0>{})) *
            sizeof(LdsDataType);

        const index_t size_per_wave =
            lds_tile.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                make_tuple(number<0>{}, number<1>{}, number<0>{})) *
                sizeof(LdsDataType) -
            size_per_buf;

        const index_t size_per_issue =
            lds_tile.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                make_tuple(number<1>{}, number<0>{}, number<0>{})) *
                sizeof(LdsDataType) -
            size_per_buf;

        const index_t m0_init_value = size_per_buf + size_per_wave * get_warp_id();
        m0_set_with_memory(amd_wave_read_first_lane(m0_init_value));

        using Traits   = load_store_traits;
        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        LdsDataType* smem = lds_tile.get_bottom_tensor_view().get_buffer_view().p_data_;

        // The buffer-load builtin scales the element offset by sizeof(DataType)
        // and feeds the result to a 32-bit voffset. To keep the byte offset
        // within INT32_MAX *for any active lane in the wave*, leave a margin
        // of half the element window for per-lane spread relative to lane-0.
        constexpr long_index_t kInt32ElemWindow =
            static_cast<long_index_t>(INT32_MAX) / static_cast<long_index_t>(sizeof(DataType));
        constexpr long_index_t kRebaseThreshold = kInt32ElemWindow / 2;

        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess  = number<iCoord * NumAccessPerCoord + iCoordAccess>{};
                constexpr auto pre_nop_ = [&]() {
                    if constexpr(pre_nop && iCoord == 0 && iCoordAccess == 0)
                        return bool_constant<true>{};
                    else
                        return bool_constant<false>{};
                }();

                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);
                constexpr auto idx_gather   = get_gather_index(idx_ys_start);

                // Per-lane absolute page offset (in elements of DataType).
                const long_index_t lane_page_offset =
                    static_cast<long_index_t>(page_idx_[idx_gather]);

                // Wave-uniform anchor candidate: lane-0's value (or first
                // active lane). Promoted to SGPRs by the readfirstlane.
                const long_index_t wave_anchor = amd_wave_read_first_lane(lane_page_offset);

                // Lazy rebase: only when the wave-uniform anchor has drifted
                // outside the current int32 voffset window around cur_anchor_.
                const long_index_t rel = wave_anchor - cur_anchor_;
                if(rel < 0 || rel >= kRebaseThreshold)
                {
                    cur_anchor_                      = wave_anchor;
                    bottom_tensor_view_.buf_.p_data_ = p_data_orig_ + cur_anchor_;
                    using BufSizeT =
                        remove_cvref_t<decltype(bottom_tensor_view_.buf_.buffer_size_)>;
                    bottom_tensor_view_.buf_.buffer_size_ =
                        static_cast<BufSizeT>(buffer_size_orig_ - cur_anchor_);
                    bottom_tensor_view_.init_raw();
                }

                // Per-lane voffset relative to (possibly new) cur_anchor_.
                // Fits in int32 by construction (kRebaseThreshold + spread).
                const index_t lane_voffset =
                    static_cast<index_t>(lane_page_offset - cur_anchor_);

                // read from bottom tensor
                if constexpr(std::is_same_v<ValidArray, std::nullptr_t>)
                {
                    get_bottom_tensor_view().template async_get_vectorized_elements_raw<vector_t>(
                        smem, bottom_tensor_thread_coord, lane_voffset, 0, pre_nop_);
                }
                else
                {
                    get_bottom_tensor_view().template async_get_vectorized_elements_raw<vector_t>(
                        smem,
                        bottom_tensor_thread_coord,
                        lane_voffset,
                        valids_[idx_gather],
                        0,
                        pre_nop_);
                }

                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto forward_step_scatter = generate_tuple(
                        [&](auto i) { return is_gather_dim(i) ? 0 : idx_diff_ys[i]; },
                        number<NDimY>{});

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                        forward_step_scatter);

                    move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);

                    m0_inc_with_memory(size_per_issue);
                }
            });
        });
    }

    // TODO: fix with swizzle
    template <typename LdsTileWindow_,
              index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true,
              bool static_move_ys         = false,
              typename = std::enable_if_t<std::is_class_v<remove_cvref_t<LdsTileWindow_>>>>
    CK_TILE_DEVICE void async_load_with_offset(index_t offset,
                                               LdsTileWindow_&& lds_tile,
                                               number<i_access_unsupport_>          = {},
                                               bool_constant<oob_conditional_check> = {},
                                               bool_constant<static_move_ys>        = {}) const
    {
        using LdsTileWindow = remove_cvref_t<LdsTileWindow_>;
        using LdsDataType   = typename LdsTileWindow::DataType;

        using Traits = load_store_traits;

        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        // Precompute invariant values outside loops
        const auto window_origin       = lds_tile.get_window_origin();
        const auto& bottom_tensor_view = lds_tile.get_bottom_tensor_view();
        const auto& tensor_descriptor  = bottom_tensor_view.get_tensor_descriptor();
        auto lds_base_ptr              = bottom_tensor_view.get_buffer_view().p_data_;

        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            auto window_adaptor_warp_coord = pre_computed_warp_coords_[iCoord][I0];
            auto bottom_tensor_warp_coord  = pre_computed_warp_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                constexpr auto idx_ys_offset = [&]() {
                    constexpr auto idx_off_ys = SFC_Ys::get_step_between(number<0>{}, iAccess);
                    constexpr auto adapter_ys_offset = make_tensor_adaptor_coordinate(
                        StaticTileDistribution_{}.get_ps_ys_to_xs_adaptor(),
                        container_concat(array<index_t, NDimP>{0},
                                         to_array<index_t, idx_off_ys.size()>(idx_off_ys)));
                    return adapter_ys_offset.get_bottom_index();
                }();
                const auto lds_ys_offset = [&]() {
                    if constexpr(static_move_ys)
                    {
                        const auto coord_ys_offset =
                            make_tensor_coordinate(tensor_descriptor, idx_ys_offset);
                        return coord_ys_offset.get_offset();
                    }
                    else
                        return 0;
                }();

                // Use precomputed window origin & tensor descriptor
                auto lds_bottom_tensor_thread_idx =
                    window_origin + window_adaptor_warp_coord.get_bottom_index();
                const auto lds_coord =
                    make_tensor_coordinate(tensor_descriptor, lds_bottom_tensor_thread_idx);

                // Calculate SMEM address using base pointer
                CK_TILE_LDS_ADDR LdsDataType* smem = lds_base_ptr +
                                                     lds_coord.get_offset() / Traits::PackedSize +
                                                     lds_ys_offset / Traits::PackedSize;

                const auto dram_ys_offset = [&]() {
                    if constexpr(static_move_ys)
                    {
                        const auto coord_ys_offset = make_tensor_coordinate(
                            this->get_bottom_tensor_view().get_tensor_descriptor(), idx_ys_offset);
                        return coord_ys_offset.get_offset();
                    }
                    else
                        return 0;
                }();

                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);
                constexpr auto idx_gather   = get_gather_index(idx_ys_start);
                const auto page_offset      = page_idx_[idx_gather];

                auto mixed_bottom_thread_coord = bottom_tensor_thread_coord;
                mixed_bottom_thread_coord.get_hidden_index()[number<0>{}] += page_offset;

                if constexpr(std::is_same_v<ValidArray, std::nullptr_t>)
                {
                    this->get_bottom_tensor_view().template async_get_vectorized_elements<vector_t>(
                        smem,
                        mixed_bottom_thread_coord,
                        offset + dram_ys_offset,
                        bool_constant<oob_conditional_check>{});
                }
                else
                {
                    this->get_bottom_tensor_view().template async_get_vectorized_elements<vector_t>(
                        smem,
                        mixed_bottom_thread_coord,
                        offset + dram_ys_offset,
                        valids_[idx_gather],
                        bool_constant<oob_conditional_check>{});
                }

                // Move thread coordinate if not last access
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto forward_step_scatter = generate_tuple(
                        [&](auto i) { return is_gather_dim(i) ? 0 : idx_diff_ys[i]; },
                        number<NDimY>{});

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                        forward_step_scatter);

                    if constexpr(!static_move_ys)
                        move_window_adaptor_and_bottom_tensor_thread_coordinate(
                            window_adaptor_thread_coord,
                            bottom_tensor_thread_coord,
                            idx_diff_ps_ys);

                    if constexpr(!static_move_ys)
                        move_window_adaptor_and_bottom_tensor_thread_coordinate(
                            window_adaptor_warp_coord, bottom_tensor_warp_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    template <index_t i_access_unsupport_ = -1, bool oob_conditional_check = true>
    CK_TILE_DEVICE void update(const static_distributed_tensor<DataType, TileDstr>& dstr_tensor,
                               number<i_access_unsupport_>          = {},
                               bool_constant<oob_conditional_check> = {}) const
    {
        using Traits = load_store_traits;

        // using vector_type_t = typename Traits::vector_type_t;
        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        constexpr auto tile_dstr = TileDstr{};

        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);
                constexpr auto idx_gather   = get_gather_index(idx_ys_start);
                const auto page_offset      = page_idx_[idx_gather];

                // read from distributed tensor
                vector_t vec_value;

                static_for<0, Traits::ScalarPerVector, Traits::PackedSize>{}([&](auto j) {
                    constexpr auto idx_ys = generate_tuple(
                        [&](auto jj) {
                            return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                            : idx_ys_start[jj];
                        },
                        number<NDimY>{});

                    constexpr index_t d =
                        tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys) /
                        Traits::PackedSize;

                    vec_value.template get_as<DataType>()(j / Traits::PackedSize) =
                        dstr_tensor.get_thread_buffer().template at<d>();
                });

                // write into bottom tensor
                if constexpr(std::is_same_v<ValidArray, std::nullptr_t>)
                {
                    get_bottom_tensor_view().template update_vectorized_elements<vector_t>(
                        bottom_tensor_thread_coord,
                        page_offset,
                        vec_value,
                        bool_constant<oob_conditional_check>{});
                }
                else
                {
                    get_bottom_tensor_view().template update_vectorized_elements<vector_t>(
                        bottom_tensor_thread_coord,
                        page_offset,
                        valids_[idx_gather],
                        vec_value,
                        bool_constant<oob_conditional_check>{});
                }

                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto forward_step_scatter = generate_tuple(
                        [&](auto i) { return is_gather_dim(i) ? 0 : idx_diff_ys[i]; },
                        number<NDimY>{});

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                        forward_step_scatter);

                    move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    template <index_t i_access_unsupport_ = -1, bool oob_conditional_check = true>
    CK_TILE_DEVICE void store(const static_distributed_tensor<DataType, TileDstr>& dstr_tensor,
                              number<i_access_unsupport_>          = {},
                              bool_constant<oob_conditional_check> = {}) const
    {
        using Traits = load_store_traits;

        // using vector_type_t = typename Traits::vector_type_t;
        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        constexpr auto tile_dstr = TileDstr{};
        // printf("off %d\n", page_idx_[I0]);
        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);
                constexpr auto idx_gather   = get_gather_index(idx_ys_start);
                const auto page_offset      = page_idx_[idx_gather];

                // printf("idx_ys_start[0], idx_ys_start[1](%d, %d) \n",
                // get_gather_index(idx_ys_start)+0, idx_ys_start[number<1>{}]+0);

                // read from distributed tensor
                // vector_type_t vec;
                vector_t vec_value;

                static_for<0, Traits::ScalarPerVector, Traits::PackedSize>{}([&](auto j) {
                    constexpr auto idx_ys = generate_tuple(
                        [&](auto jj) {
                            return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                            : idx_ys_start[jj];
                        },
                        number<NDimY>{});

                    constexpr index_t d =
                        tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys) /
                        Traits::PackedSize;
                    // printf("thread_idx_m: %d j: %d\n", idx_ys[number<0>{}] + 0, 0+j);
                    vec_value.template get_as<DataType>()(j / Traits::PackedSize) =
                        dstr_tensor.get_thread_buffer().template at<d>();
                });

                // const vector_t vec_value = vec.template get_as<vector_t>().template at<0>();

                // write into bottom tensor
                if constexpr(std::is_same_v<ValidArray, std::nullptr_t>)
                {
                    get_bottom_tensor_view().template set_vectorized_elements<vector_t>(
                        bottom_tensor_thread_coord,
                        page_offset,
                        vec_value,
                        bool_constant<oob_conditional_check>{});
                }
                else
                {
                    get_bottom_tensor_view().template set_vectorized_elements<vector_t>(
                        bottom_tensor_thread_coord,
                        page_offset,
                        valids_[idx_gather],
                        vec_value,
                        bool_constant<oob_conditional_check>{});
                }

                // printf("coord_offset:%d,   scatter_offset:%d \n",
                // bottom_tensor_thread_coord.get_offset(), offset); move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto forward_step_scatter = generate_tuple(
                        [&](auto i) { return is_gather_dim(i) ? 0 : idx_diff_ys[i]; },
                        number<NDimY>{});

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                        forward_step_scatter);

                    move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    // move thread's botom tensor coordiante
    // [x0', x1', ... ] ==> [offset]
    // also move window-origin
    CK_TILE_DEVICE void move(const BottomTensorIndex& step)
    {
        window_origin_ += step;
        BottomTensorIndex step_new = step;
        step_new(HsGatherDim)      = 0;
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            move_tensor_coordinate(bottom_tensor_view_.get_tensor_descriptor(),
                                   pre_computed_coords_(iCoord)(I1),
                                   step_new);
        });
        if constexpr(BottomTensorView::buffer_view::get_address_space() ==
                     address_space_enum::global)
        {
            static_for<0, NumCoord, 1>{}([&](auto iCoord) {
                move_tensor_coordinate(bottom_tensor_view_.get_tensor_descriptor(),
                                       pre_computed_warp_coords_(iCoord)(I1),
                                       step_new);
            });
        }
    }

    // ---------------------------------------------------------------------
    // async_load_raw_long: variant of async_load_raw that issues the per-tile
    // gather load via `amd_async_global_load_lds_raw` (i.e. AMDGCN
    // `global_load_lds_dwordx*`) rather than `buffer_load_dword_lds`.
    //
    // Identical iteration structure, m0/LDS-slot bookkeeping, and SFC walk
    // as async_load_raw — only the HW load instruction is swapped. The page
    // indirection is folded into a per-lane 64-bit base pointer, lifting
    // both 4 GB limits in the buffer_load path (SRD `size` field is uint32_t,
    // per-lane voffset is int32). PageIdxArray's element type can therefore
    // be `long_index_t` (caller's responsibility).
    //
    // Why not per-issue SRD rebase? In the K/V tile distributions emitted
    // by CK-UA today, a single wave-wide buffer_load_dword* spans
    // (LaneGroups) different N-positions, which for the prefill configs
    // (NumWarps≥2) can map to several different pages within one issue.
    // For paged-KV caches > 4 GB, those pages can be ≫ 4 GB apart in the
    // global K buffer, exceeding the 32-bit voffset / 32-bit SRD-size
    // range. Only a per-lane 64-bit base pointer (i.e. global_load_lds)
    // can address all those lanes from a single instruction.
    //
    // OOB note: this path drops the SRD's hardware OOB clamp. Caller must
    // ensure `page_idx_` only references live pages (true in the paged-KV
    // use-case where block_tables are populated from a valid allocator).
    // ---------------------------------------------------------------------
    template <typename LdsTileWindow_,
              index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true,
              bool pre_nop                = false>
    CK_TILE_DEVICE auto async_load_raw_long(LdsTileWindow_&& lds_tile,
                                            number<i_access_unsupport_>          = {},
                                            bool_constant<oob_conditional_check> = {},
                                            bool_constant<pre_nop> = {}) const
    {
        using LdsTileWindow = remove_cvref_t<LdsTileWindow_>;
        using LdsDataType   = typename LdsTileWindow::DataType;

        static_assert(LdsTileWindow::get_num_of_dimension() == 3); // TODO: hard coded

        // The per-tile LDS layout in elements (not bytes). The new
        // `global_load_lds_*` path differs from the `buffer_load_dword_lds`
        // path here: the LLVM intrinsic implicitly sets `m0` from its
        // `lptr` argument every call, so the manual `m0_set / m0_inc`
        // bookkeeping used by `async_load_raw` would be silently
        // overwritten. Instead, we compute the per-issue LDS element offset
        // and add it to the LDS base pointer on each call — the compiler
        // emits a fresh `s_mov_b32 m0, ...` per load with the right value.
        const index_t elems_per_buf =
            lds_tile.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                make_tuple(number<0>{}, number<0>{}, number<0>{}));

        const index_t elems_per_wave =
            lds_tile.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                make_tuple(number<0>{}, number<1>{}, number<0>{})) -
            elems_per_buf;

        const index_t elems_per_issue =
            lds_tile.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
                make_tuple(number<1>{}, number<0>{}, number<0>{})) -
            elems_per_buf;

        using Traits   = load_store_traits;
        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        // bf16/fp16/etc. element-typed global ptr base for this tile-window.
        const DataType* base_data_ptr = bottom_tensor_view_.get_buffer_view().p_data_;
        // Element count in the underlying buffer — used to clamp per-lane
        // pointers that go past the live range, mimicking the SRD's OOB
        // semantics on the original `buffer_load_dword_lds` path.
        const long_index_t buf_elems = static_cast<long_index_t>(
            bottom_tensor_view_.get_buffer_view().buffer_size_);
        LdsDataType*    lds_base = lds_tile.get_bottom_tensor_view().get_buffer_view().p_data_;

        // Wave / warp-group offset into LDS, computed once.
        const index_t lds_wave_elems = elems_per_buf + elems_per_wave * get_warp_id();

        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess  = number<iCoord * NumAccessPerCoord + iCoordAccess>{};
                constexpr auto pre_nop_ = [&]() {
                    if constexpr(pre_nop && iCoord == 0 && iCoordAccess == 0)
                        return bool_constant<true>{};
                    else
                        return bool_constant<false>{};
                }();

                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);
                constexpr auto idx_gather   = get_gather_index(idx_ys_start);
                // page_idx_ element type can be long_index_t — the pointer
                // arithmetic below stays 64-bit by promotion.
                const auto page_offset = page_idx_[idx_gather];

                // Per-lane 64-bit GLOBAL base pointer. coord.get_offset() is
                // the within-bottom-tensor element offset (intra-tile,
                // int32-safe by construction); page_offset is the
                // page-indirected element offset (potentially > INT32_MAX).
                // Pointer arithmetic on DataType* advances by sizeof(DataType)
                // and uses 64-bit ptrdiff_t internally.
                const long_index_t lane_elem_off =
                    static_cast<long_index_t>(bottom_tensor_thread_coord.get_offset()) +
                    static_cast<long_index_t>(page_offset);
                // Clamp to in-buffer range to keep `global_load_lds` from
                // faulting on tail-padded pages (the original buffer_load
                // SRD silently returned 0 for the same OOB voffsets). The
                // attention mask zeroes the contribution from these lanes
                // at softmax, so the value read here is irrelevant.
                constexpr index_t bytes_per_load_ = sizeof(vector_t);
                constexpr index_t elems_per_load_ = bytes_per_load_ / sizeof(DataType);
                const bool in_range =
                    (lane_elem_off >= 0) &&
                    (lane_elem_off + elems_per_load_ <= buf_elems);
                const long_index_t safe_off = in_range ? lane_elem_off : 0;
                const DataType* per_lane_ptr = base_data_ptr + safe_off;

                // Per-issue LDS write target. Wave-uniform; intrinsic emits
                // `s_mov_b32 m0, <this>` and the dwordx4 lds-direct write
                // lands at m0 + (lane_id * bytes_per_lane). For NumCoord==1
                // (the only case exercised by the UA pipeline today) the
                // two formulas coincide; we use the monotonically-increasing
                // one so each issue lands in its own LDS slot.
                constexpr index_t kIssue = iCoord * NumAccessPerCoord + iCoordAccess;
                LdsDataType* lds_ptr = lds_base + lds_wave_elems + elems_per_issue * kIssue;

                amd_async_global_load_lds_raw<DataType, elems_per_load_, /*byte_offset_imm=*/0>(
                    lds_ptr, per_lane_ptr, pre_nop_);

                // move thread coordinate (no m0_inc — see header comment above)
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto forward_step_scatter = generate_tuple(
                        [&](auto i) { return is_gather_dim(i) ? 0 : idx_diff_ys[i]; },
                        number<NDimY>{});

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}),
                        forward_step_scatter);

                    move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    CK_TILE_DEVICE void update_page_idx(const PageIdxArray& new_idx) { page_idx_ = new_idx; }

    CK_TILE_DEVICE void update_valids(const ValidArray& new_valids)
    {
        if constexpr(std::is_same_v<ValidArray, std::nullptr_t> == false)
        {
            valids_ = new_valids;
        }
    }

    CK_TILE_DEVICE void update_page_idx_and_valids(const PageIdxArray& new_idx,
                                                   const ValidArray& new_valids)
    {
        update_page_idx(new_idx);
        update_valids(new_valids);
    }

    CK_TILE_DEVICE void set_window_origin(const BottomTensorIndex& new_window_origin)
    {
        window_origin_ = new_window_origin;

#if 0 // debug
      // TODO: this use more register for FA, but less register for GEMM
      // need investigation
      // only support warp-tile and block-tile
        static_assert(NDimP == 1 or NDimP == 2, "wrong!");

        WindowAdaptorCoord window_adaptor_thread_coord_tmp;

        if constexpr(NDimP == 1)
        {
            window_adaptor_thread_coord_tmp = make_tensor_adaptor_coordinate(
                tile_dstr_.get_ps_ys_to_xs_adaptor(), AdaptorTopIndex{get_lane_id(), 0});
        }
        else if constexpr(NDimP == 2)
        {
            window_adaptor_thread_coord_tmp =
                make_tensor_adaptor_coordinate(tile_dstr_.get_ps_ys_to_xs_adaptor(),
                                               AdaptorTopIndex{get_warp_id(), get_lane_id(), 0});
        }
#else
        // TODO: this use less register for FA, but more register for GEMM
        // need investigation
        const auto window_adaptor_thread_coord_tmp = make_tensor_adaptor_coordinate(
            tile_dstr_.get_ps_ys_to_xs_adaptor(),
            container_concat(get_partition_index(tile_dstr_), array<index_t, NDimY>{0}));
#endif

        BottomTensorIndex bottom_tensor_thread_origin_idx_tmp =
            window_origin_ + window_adaptor_thread_coord_tmp.get_bottom_index();

        bottom_tensor_thread_origin_idx_tmp(HsGatherDim) = 0;
        const auto bottom_tensor_thread_coord_tmp        = make_tensor_coordinate(
            bottom_tensor_view_.get_tensor_descriptor(), bottom_tensor_thread_origin_idx_tmp);

        // pre-compute NumCoord (WindowAdaptorCoord, BottomTensorCoord) bundles to speed up
        // future load/store() calls (might allocate more registers)
        using Traits = load_store_traits;
        using SFC_Ys = typename Traits::SFC_Ys;

        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            auto window_adaptor_thread_coord = window_adaptor_thread_coord_tmp;
            auto bottom_tensor_thread_coord  = bottom_tensor_thread_coord_tmp;

            constexpr auto idx_diff_ys =
                SFC_Ys::get_step_between(number<0>{}, number<iCoord * NumAccessPerCoord>{});

            constexpr auto idx_diff_ps_ys = container_concat(
                generate_tuple([&](auto) { return number<0>{}; }, number<NDimP>{}), idx_diff_ys);

            move_window_adaptor_and_bottom_tensor_thread_coordinate(
                window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);

            pre_computed_coords_(iCoord) =
                make_tuple(window_adaptor_thread_coord, bottom_tensor_thread_coord);
        });
    }

    CK_TILE_HOST_DEVICE void init_raw() { bottom_tensor_view_.init_raw(); }

    // Companion to init_raw(): capture the original SRD base / size so that
    // async_load_raw_lazy_rebase() can shift the wave-uniform base pointer
    // on demand and later recompute the buffer resource (init_raw) without
    // losing the underlying pool layout. Reset the anchor to 0 (no shift).
    // Call this once per window instead of init_raw() when the per-issue
    // page offsets may exceed INT32_MAX (i.e. when the cache pool size in
    // bytes can overflow int32 voffsets).
    CK_TILE_HOST_DEVICE void init_raw_lazy_rebase()
    {
        p_data_orig_      = bottom_tensor_view_.buf_.p_data_;
        buffer_size_orig_ = static_cast<long_index_t>(bottom_tensor_view_.buf_.buffer_size_);
        cur_anchor_       = 0;
        bottom_tensor_view_.init_raw();
    }

    // this is the bottom tensor view
    // [x0', x1', ...] ==> [offset]
    BottomTensorView bottom_tensor_view_;

    //
    WindowLengths window_lengths_;

    // origin ([x0', x1', ...]) of window on bottom tensor
    BottomTensorIndex window_origin_;

    // Tile tensor distribution, which contains:
    //   1. adaptor for window: [p0, p1, ..., y0, y1, ...] ==> [x0, x1, ...]
    //   2. thread descriptor for thread tensor in register: [y0, y1, ...] ==> [d]
    TileDstr tile_dstr_;

    PageIdxArray page_idx_;
    ValidArray valids_;

    // this contains:
    //   per-thread coordinate for window adaptor
    //   per-thread coordinate for bottom tensor
    array<tuple<WindowAdaptorCoord, BottomTensorCoord>, NumCoord> pre_computed_coords_;
    std::conditional_t<BottomTensorView::buffer_view::get_address_space() ==
                           address_space_enum::global,
                       array<tuple<WindowAdaptorCoord, BottomTensorCoord>, NumCoord>,
                       std::byte>
        pre_computed_warp_coords_;

    // State used by async_load_raw_lazy_rebase(). Populated by
    // init_raw_lazy_rebase(); ignored by all other load paths.
    //   p_data_orig_      : original SRD base pointer (never mutated post-init)
    //   buffer_size_orig_ : original SRD size in elements of DataType
    //   cur_anchor_       : current wave-uniform SRD shift (in elements,
    //                       relative to p_data_orig_); kept in SGPRs as the
    //                       value is only ever assigned from
    //                       amd_wave_read_first_lane(...). When non-zero,
    //                       bottom_tensor_view_.buf_.p_data_ ==
    //                       p_data_orig_ + cur_anchor_.
    typename BottomTensorView::buffer_view::type* p_data_orig_ = nullptr;
    long_index_t buffer_size_orig_                             = 0;
    long_index_t cur_anchor_                                   = 0;
};

// TODO: use strategy
/**
 * @brief Factory function to create tile_scatter_gather with multi-dimensional gather support.
 *
 * This overload accepts a sequence<YsGatherDims...> to specify multiple Y-space dimensions
 * for page lookup. Use this when the tile distribution decomposes the paged dimension
 * into multiple Y-space dimensions (e.g., VECTORIZED_LAYOUT V tensor with K decomposition
 * {K2, K0, K1} where both Y0 and Y2 contribute to page index).
 *
 * @tparam HsGatherDim      H-space dimension for gather
 * @tparam NumCoord         Number of pre-computed coordinates
 * @tparam YsGatherDims     Parameter pack specifying which Y-dimensions are used for page lookup
 *
 * @param tensor_view       The underlying tensor view for device memory access
 * @param window_lengths    Static window sizes for each dimension
 * @param origin            Window origin coordinates on the bottom tensor
 * @param tile_distribution Thread-to-tile mapping distribution
 * @param page_idx          Array of page offsets (in bytes) for scatter/gather
 */
template <typename TensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          typename StaticPageIndexArray_,
          index_t HsGatherDim,
          index_t NumCoord,
          index_t... YsGatherDims>
CK_TILE_DEVICE constexpr auto
make_tile_scatter_gather(const TensorView_& tensor_view,
                         const WindowLengths_& window_lengths,
                         const multi_index<TensorView_::get_num_of_dimension()>& origin,
                         const StaticTileDistribution_& tile_distribution,
                         const StaticPageIndexArray_& page_idx,
                         number<HsGatherDim>,
                         number<NumCoord>,
                         sequence<YsGatherDims...>)
{
    return tile_scatter_gather<remove_cvref_t<TensorView_>,
                               remove_cvref_t<WindowLengths_>,
                               remove_cvref_t<StaticTileDistribution_>,
                               remove_cvref_t<StaticPageIndexArray_>,
                               std::nullptr_t,
                               HsGatherDim,
                               NumCoord,
                               sequence<YsGatherDims...>>{
        tensor_view, window_lengths, origin, tile_distribution, page_idx, nullptr};
}

// Legacy overload (compatible with original API)
template <typename TensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          typename StaticPageIndexArray_,
          index_t HsGatherDim = 0,
          index_t NumCoord    = 1>
CK_TILE_DEVICE constexpr auto
make_tile_scatter_gather(const TensorView_& tensor_view,
                         const WindowLengths_& window_lengths,
                         const multi_index<TensorView_::get_num_of_dimension()>& origin,
                         const StaticTileDistribution_& tile_distribution,
                         const StaticPageIndexArray_& page_idx,
                         number<HsGatherDim> = {},
                         number<NumCoord>    = {})
{
    return tile_scatter_gather<remove_cvref_t<TensorView_>,
                               remove_cvref_t<WindowLengths_>,
                               remove_cvref_t<StaticTileDistribution_>,
                               remove_cvref_t<StaticPageIndexArray_>,
                               std::nullptr_t,
                               HsGatherDim,
                               NumCoord,
                               sequence<0>>{
        tensor_view, window_lengths, origin, tile_distribution, page_idx, nullptr};
}

template <typename TensorView,
          typename WindowLengths,
          typename StaticTileDistribution,
          typename StaticPageIndexArray,
          index_t HsGatherDim>
CK_TILE_DEVICE constexpr auto make_tile_scatter_gather(
    const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
    const multi_index<TensorView::get_num_of_dimension()>& origin,
    const StaticTileDistribution& tile_distribution,
    const StaticPageIndexArray& page_idx,
    number<HsGatherDim> = {})
{
    return make_tile_scatter_gather(tile_window.get_bottom_tensor_view(),
                                    tile_window.get_window_lengths(),
                                    origin,
                                    tile_distribution,
                                    page_idx,
                                    number<HsGatherDim>{});
}

template <typename TensorView,
          typename WindowLengths,
          typename StaticTileDistribution,
          typename StaticPageIndexArray,
          index_t HsGatherDim>
CK_TILE_DEVICE constexpr auto make_tile_scatter_gather(
    const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
    const StaticTileDistribution& tile_distribution,
    const StaticPageIndexArray& page_idx,
    number<HsGatherDim> = {})
{
    return make_tile_scatter_gather(tile_window.get_bottom_tensor_view(),
                                    tile_window.get_window_lengths(),
                                    tile_window.get_window_origin(),
                                    tile_distribution,
                                    page_idx,
                                    number<HsGatherDim>{});
}

template <typename TensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          typename StaticPageIndexArray_,
          typename StaticValidArray_,
          index_t HsGatherDim = 0,
          index_t NumCoord    = 1>
CK_TILE_DEVICE constexpr auto
make_tile_scatter_gather(const TensorView_& tensor_view,
                         const WindowLengths_& window_lengths,
                         const multi_index<TensorView_::get_num_of_dimension()>& origin,
                         const StaticTileDistribution_& tile_distribution,
                         const StaticPageIndexArray_& page_idx,
                         const StaticValidArray_& valids,
                         number<HsGatherDim> = {},
                         number<NumCoord>    = {})
{
    return tile_scatter_gather<remove_cvref_t<TensorView_>,
                               remove_cvref_t<WindowLengths_>,
                               remove_cvref_t<StaticTileDistribution_>,
                               remove_cvref_t<StaticPageIndexArray_>,
                               remove_cvref_t<StaticValidArray_>,
                               HsGatherDim,
                               NumCoord>{
        tensor_view, window_lengths, origin, tile_distribution, page_idx, valids};
}

template <typename TensorView,
          typename WindowLengths,
          typename StaticTileDistribution,
          typename StaticPageIndexArray,
          typename StaticValidArray,
          index_t HsGatherDim>
CK_TILE_DEVICE constexpr auto make_tile_scatter_gather(
    const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
    const multi_index<TensorView::get_num_of_dimension()>& origin,
    const StaticTileDistribution& tile_distribution,
    const StaticPageIndexArray& page_idx,
    const StaticValidArray& valids,
    number<HsGatherDim> = {})
{
    return make_tile_scatter_gather(tile_window.get_bottom_tensor_view(),
                                    tile_window.get_window_lengths(),
                                    origin,
                                    tile_distribution,
                                    page_idx,
                                    valids,
                                    number<HsGatherDim>{});
}

template <typename TensorView,
          typename WindowLengths,
          typename StaticTileDistribution,
          typename StaticPageIndexArray,
          typename StaticValidArray,
          index_t HsGatherDim>
CK_TILE_DEVICE constexpr auto make_tile_scatter_gather(
    const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
    const StaticTileDistribution& tile_distribution,
    const StaticPageIndexArray& page_idx,
    const StaticValidArray& valids,
    number<HsGatherDim> = {})
{
    return make_tile_scatter_gather(tile_window.get_bottom_tensor_view(),
                                    tile_window.get_window_lengths(),
                                    tile_window.get_window_origin(),
                                    tile_distribution,
                                    page_idx,
                                    valids,
                                    number<HsGatherDim>{});
}

template <typename NewTensorView_,
          typename OldTensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          typename StaticPageIndexArray_,
          typename StaticValidArray_,
          index_t HsGatherDim = 0,
          index_t NumCoord    = 1>
CK_TILE_DEVICE auto replace_bottom_tensor_view(const NewTensorView_& new_tensor_view,
                                               const tile_scatter_gather<OldTensorView_,
                                                                         WindowLengths_,
                                                                         StaticTileDistribution_,
                                                                         StaticPageIndexArray_,
                                                                         StaticValidArray_,
                                                                         HsGatherDim,
                                                                         NumCoord>& tile_window)
{
    return make_tile_scatter_gather(new_tensor_view,
                                    tile_window.window_lengths_,
                                    tile_window.window_origin_,
                                    tile_window.tile_dstr_,
                                    tile_window.page_idx_,
                                    tile_window.valids_);
}

} // namespace ck_tile
