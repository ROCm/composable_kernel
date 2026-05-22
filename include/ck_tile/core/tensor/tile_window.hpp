// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/arch/utility.hpp"
#include "ck_tile/core/algorithm/space_filling_curve.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/tensor/null_tile_window.hpp"
#include "ck_tile/core/tensor/static_distributed_tensor.hpp"
#include "ck_tile/core/tensor/tensor_adaptor.hpp"
#include "ck_tile/core/tensor/tensor_view.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/core/tensor/tile_window_base.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

#include "ck_tile/core/utility/data_cache_prefetch.hpp"

namespace ck_tile {

/**
 * @brief This class provides tile (windowed) view and access to the device memory.
 *
 * @note This tile window does not support single issue you need to use tile_window_linear
 *       structure for this purpose
 *
 * @tparam BottomTensorView_        Class describing & holding device tensor memory.
 * @tparam WindowLengths_           Spatial sizes of windowed view on tensor.
 * @tparam StaticTileDistribution_  Thread distribution (mapping) into Tile dimensions
 * @tparam NumCoord                 TBD
 */
// Tile window with pre-computed per-thread coordinates (pre_computed_coords_).
// Construction pre-computes XOR address coordinates (~96 VALU), but subsequent
// .load()/.store() calls reuse the pre-computed coordinates with no reconstruction.
// Prefer this type for windows accessed repeatedly in a loop.
template <typename BottomTensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          index_t NumCoord>
struct tile_window_with_static_distribution
    : public tile_window_with_tile_dstr_base<
          tile_window_with_static_distribution<BottomTensorView_,
                                               WindowLengths_,
                                               StaticTileDistribution_,
                                               NumCoord>,
          BottomTensorView_,
          WindowLengths_,
          StaticTileDistribution_>
{
    using Base = tile_window_with_tile_dstr_base<
        tile_window_with_static_distribution<BottomTensorView_,
                                             WindowLengths_,
                                             StaticTileDistribution_,
                                             NumCoord>,
        BottomTensorView_,
        WindowLengths_,
        StaticTileDistribution_>;

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static_assert(NumCoord == 1);

    static_assert(Base::Traits::NumAccess % NumCoord == 0,
                  "wrong! # of access is not divisible by NumCoord");
    static constexpr index_t NumAccessPerCoord = Base::Traits::NumAccess / NumCoord;

    CK_TILE_DEVICE constexpr tile_window_with_static_distribution() = default;

    CK_TILE_DEVICE constexpr tile_window_with_static_distribution(
        const typename Base::BottomTensorView& bottom_tensor_view,
        const typename Base::WindowLengths& window_lengths,
        const typename Base::BottomTensorIndex& window_origin,
        const typename Base::TileDstr& tile_distribution,
        decltype(get_partition_index(tile_distribution)) partition_index)
        : pre_computed_coords_{}
    {

        this->window_origin_      = window_origin;
        this->window_lengths_     = window_lengths;
        this->bottom_tensor_view_ = bottom_tensor_view;
        this->tile_dstr_          = tile_distribution;

        pre_computed_coords_ =
            prepare_coords(bottom_tensor_view, window_origin, tile_distribution, partition_index);
        if constexpr(Base::BottomTensorView::buffer_view::get_address_space() ==
                     address_space_enum::global)
        {
            auto use_lane_id_0 = partition_index;
            use_lane_id_0[1]   = 0;

            pre_computed_warp_coords_ =
                prepare_coords(bottom_tensor_view, window_origin, tile_distribution, use_lane_id_0);
        }
    }

    CK_TILE_DEVICE constexpr tile_window_with_static_distribution(
        const typename Base::BottomTensorView& bottom_tensor_view,
        const typename Base::WindowLengths& window_lengths,
        const typename Base::BottomTensorIndex& window_origin,
        const typename Base::TileDstr& tile_distribution)
        : tile_window_with_static_distribution(bottom_tensor_view,
                                               window_lengths,
                                               window_origin,
                                               tile_distribution,
                                               get_partition_index(tile_distribution))
    {
    }

    CK_TILE_DEVICE constexpr auto
    prepare_coords(const typename Base::BottomTensorView& bottom_tensor_view,
                   const typename Base::BottomTensorIndex& window_origin,
                   const typename Base::TileDstr& tile_distribution,
                   decltype(get_partition_index(tile_distribution)) partition_index) const
    {
        array<tuple<typename Base::WindowAdaptorCoord, typename Base::BottomTensorCoord>, NumCoord>
            coords;

        const auto window_adaptor_thread_coord_tmp = make_tensor_adaptor_coordinate(
            tile_distribution.get_ps_ys_to_xs_adaptor(),
            container_concat(partition_index, multi_index<Base::NDimY>{0}));

        typename Base::BottomTensorIndex bottom_tensor_thread_origin_idx_tmp =
            window_origin + window_adaptor_thread_coord_tmp.get_bottom_index();

        const auto bottom_tensor_thread_coord_tmp = make_tensor_coordinate(
            bottom_tensor_view.get_tensor_descriptor(), bottom_tensor_thread_origin_idx_tmp);

        // pre-compute NumCoord (WindowAdaptorCoord, BottomTensorCoord) bundles to speed up
        // future load/store() calls (might allocate more registers)
        using Traits = typename Base::Traits;
        using SFC_Ys = typename Traits::SFC_Ys;

        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            auto window_adaptor_thread_coord = window_adaptor_thread_coord_tmp;
            auto bottom_tensor_thread_coord  = bottom_tensor_thread_coord_tmp;

            constexpr auto idx_diff_ys =
                SFC_Ys::get_step_between(number<0>{}, number<iCoord * NumAccessPerCoord>{});

            constexpr auto idx_diff_ps_ys = container_concat(
                generate_tuple([&](auto) { return number<0>{}; }, number<Base::NDimP>{}),
                idx_diff_ys);

            Base::move_window_adaptor_and_bottom_tensor_thread_coordinate(
                window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);

            coords(iCoord) = make_tuple(window_adaptor_thread_coord, bottom_tensor_thread_coord);
        });

        return coords;
    }

    template <index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true,
              bool static_move_ys         = false>
    CK_TILE_DEVICE auto load(number<i_access_unsupport_>          = {},
                             bool_constant<oob_conditional_check> = {},
                             bool_constant<static_move_ys>        = {}) const
    {
        return load_with_offset(0,
                                number<i_access_unsupport_>{},
                                bool_constant<oob_conditional_check>{},
                                bool_constant<static_move_ys>{});
    }

    template <index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true,
              bool static_move_ys         = false,
              typename offset_t           = index_t>
    CK_TILE_DEVICE auto load_with_offset(offset_t offset,
                                         number<i_access_unsupport_>          = {},
                                         bool_constant<oob_conditional_check> = {},
                                         bool_constant<static_move_ys>        = {}) const
    {
        constexpr auto tile_dstr = typename Base::TileDstr{};
        auto dst_tensor = make_static_distributed_tensor<typename Base::DataType>(tile_dstr);
        load_with_offset(offset,
                         dst_tensor,
                         number<i_access_unsupport_>{},
                         bool_constant<oob_conditional_check>{},
                         bool_constant<static_move_ys>{});
        return dst_tensor;
    }

    /**
     * @brief Load tile with elementwise function
     *
     * @note Load tile with elementwise - during value loading, an
     *       elementwise function is executed for each A0, A1, ... AN.
     *       The values A0, A1, ... AN are read by the same thread. In this way, we
     *       reduce the amount of information loaded into the registers.
     *       The same thread, during vectorized reading, accesses the same set of
     *       data from A0, A1, A2, ... AN.
     */
    template <typename... TileWindow_,
              typename ElementWise_,
              index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true>
    CK_TILE_DEVICE auto load(const ck_tile::tuple<TileWindow_...>& tile_windows,
                             ElementWise_ elementwise,
                             number<i_access_unsupport_>          = {},
                             bool_constant<oob_conditional_check> = {}) const
    {
        constexpr auto tile_dstr = typename Base::TileDstr{};
        auto dst_tensor = make_static_distributed_tensor<typename Base::DataType>(tile_dstr);
        load(dst_tensor,
             tile_windows,
             elementwise,
             number<i_access_unsupport_>{},
             bool_constant<oob_conditional_check>{});
        return dst_tensor;
    }

    template <typename DistributedTensor,
              typename... TileWindow_,
              typename ElementWise_,
              index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true>
    CK_TILE_DEVICE void load(DistributedTensor& dst_tensor,
                             const ck_tile::tuple<TileWindow_...>& tile_windows,
                             ElementWise_ elementwise,
                             number<i_access_unsupport_>          = {},
                             bool_constant<oob_conditional_check> = {}) const
    {

        using Traits   = typename Base::Traits;
        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        constexpr auto tile_dstr   = typename Base::TileDstr{};
        constexpr auto sizeOfTuple = remove_cvref_t<decltype(tile_windows)>::size();
        //  loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord =
                tile_windows[number<0>{}].pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord =
                tile_windows[number<0>{}].pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);

                // read from bottom tensor
                const auto idx_vec_value = generate_tuple(
                    [&](auto jj) {
                        return tile_windows[number<jj>{}]
                            .get_bottom_tensor_view()
                            .template get_vectorized_elements<vector_t>(
                                bottom_tensor_thread_coord,
                                0,
                                bool_constant<oob_conditional_check>{});
                    },
                    number<sizeOfTuple>{});

                // write into distributed tensor
                static_for<0, Traits::ScalarPerVector, Traits::PackedSize>{}([&](auto j) {
                    constexpr auto idx_ys = generate_tuple(
                        [&](auto jj) {
                            return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                            : idx_ys_start[jj];
                        },
                        number<Base::NDimY>{});

                    constexpr index_t d =
                        tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys) /
                        Traits::PackedSize;

                    ck_tile::apply(
                        [&](auto&&... t) {
                            elementwise(dst_tensor.get_thread_buffer().template at<d>(),
                                        t.template get_as<
                                            typename Base::DataType>()[j / Traits::PackedSize]...);
                        },
                        idx_vec_value);
                });
                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<Base::NDimP>{}),
                        idx_diff_ys);

                    Base::move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    template <typename DistributedTensor,
              index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true,
              bool static_move_ys         = false>
    CK_TILE_DEVICE void load(DistributedTensor& dst_tensor,
                             number<i_access_unsupport_>          = {},
                             bool_constant<oob_conditional_check> = {},
                             bool_constant<static_move_ys>        = {}) const
    {
        load_with_offset(0,
                         dst_tensor,
                         number<i_access_unsupport_>{},
                         bool_constant<oob_conditional_check>{},
                         bool_constant<static_move_ys>{});
    }

    template <typename offset_t>
    CK_TILE_DEVICE constexpr auto get_load_offset(offset_t = {}) const
    {
        constexpr auto bottom_tensor_idx_off = to_multi_index(offset_t{});
        const auto bottom_tensor_coord_off   = make_tensor_coordinate(
            this->bottom_tensor_view_.get_tensor_descriptor(), bottom_tensor_idx_off);
        return amd_wave_read_first_lane(bottom_tensor_coord_off.get_offset());
    }

    template <typename DataType,
              typename StaticTileDistribution,
              index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true,
              bool static_move_ys         = false,
              typename offset_t>
    CK_TILE_DEVICE void load_with_offset( //
        offset_t offset,
        static_distributed_tensor<DataType, StaticTileDistribution>& dst_tensor,
        number<i_access_unsupport_>          = {},
        bool_constant<oob_conditional_check> = {},
        bool_constant<static_move_ys>        = {}) const
    {
        using Traits   = typename Base::Traits;
        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        constexpr auto tile_dstr = typename Base::TileDstr{};

        const index_t linear_off = [&]() {
            if constexpr(std::is_integral_v<offset_t>)
                return offset;
            else if constexpr(is_constant_v<offset_t>)
                return offset_t::value;
            else
                return get_load_offset(offset_t{});
        }();

        // this is an optimization used in gfx125 where lds descriptor don't include xor swizzle
        if constexpr((Base::BottomTensorView::buffer_view::get_address_space() ==
                      address_space_enum::lds) &&
                     (!remove_cvref_t<
                         decltype(typename Base::BottomTensorView{}.get_tensor_descriptor())>::
                          template has_transform<coord_transform_enum::xor_t>()))
        {
            static_assert(
                []() constexpr {
                    [[maybe_unused]] constexpr auto desc =
                        typename Base::BottomTensorView{}.get_tensor_descriptor();
                    return true;
                }(),
                "BottomTensorView::get_tensor_descriptor() must be constexpr for LDS");
            // For LDS, compute offsets at compile time to optimize LDS access
            static_for<0, NumCoord, 1>{}([&](auto iCoord) {
                /// TODO: use structure binding (to be captured later) if compiled in C++20
                const auto& bottom_tensor_thread_coord = pre_computed_coords_[iCoord][I1];

                static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                    constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                    // data index [y0, y1, ...]
                    constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);

                    // Compute compile-time offset from access 0 to current access
                    constexpr auto lds_access_offset = [&]() {
                        constexpr auto idx_off_ys = SFC_Ys::get_step_between(number<0>{}, iAccess);
                        constexpr auto adapter_ys_offset = make_tensor_adaptor_coordinate(
                            tile_dstr.get_ps_ys_to_xs_adaptor(),
                            container_concat(array<index_t, Base::NDimP>{0},
                                             to_array<index_t, idx_off_ys.size()>(idx_off_ys)));
                        constexpr auto coord_ys_offset = make_tensor_coordinate(
                            typename Base::BottomTensorView{}.get_tensor_descriptor(),
                            adapter_ys_offset.get_bottom_index());
                        return coord_ys_offset.get_offset();
                    }();

                    // read from bottom tensor with compile-time offset
                    const vector_t vec_value =
                        this->get_bottom_tensor_view()
                            .template get_vectorized_elements<vector_t, lds_access_offset>(
                                bottom_tensor_thread_coord,
                                linear_off,
                                bool_constant<oob_conditional_check>{});
                    // write into distributed tensor
                    static_for<0, Traits::ScalarPerVector, Traits::PackedSize>{}([&](auto j) {
                        constexpr auto idx_ys = generate_tuple(
                            [&](auto jj) {
                                return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                                : idx_ys_start[jj];
                            },
                            number<Base::NDimY>{});

                        constexpr index_t d =
                            tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys) /
                            Traits::PackedSize;

                        dst_tensor.get_thread_buffer().template at<d>() =
                            vec_value
                                .template get_as<typename Base::DataType>()[j / Traits::PackedSize];
                    });
                });
            });
        }
        else
        {
            // loop over thread tensor space [y0, y1, ...]
            static_for<0, NumCoord, 1>{}([&](auto iCoord) {
                /// TODO: use structure binding (to be captured later) if compiled in C++20
                auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
                auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

                static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                    constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                    constexpr auto idx_ys_offset = [&]() {
                        constexpr auto idx_off_ys = SFC_Ys::get_step_between(number<0>{}, iAccess);
                        constexpr auto adapter_ys_offset = make_tensor_adaptor_coordinate(
                            StaticTileDistribution_{}.get_ps_ys_to_xs_adaptor(),
                            container_concat(array<index_t, Base::NDimP>{0},
                                             to_array<index_t, idx_off_ys.size()>(idx_off_ys)));
                        return adapter_ys_offset.get_bottom_index();
                    }();
                    const auto ys_offset = [&]() {
                        if constexpr(static_move_ys)
                        {
                            const auto coord_ys_offset = make_tensor_coordinate(
                                this->get_bottom_tensor_view().get_tensor_descriptor(),
                                idx_ys_offset);
                            return coord_ys_offset.get_offset();
                        }
                        else
                            return 0;
                    }();

                    // data index [y0, y1, ...]
                    constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);

                    // read from bottom tensor
                    const vector_t vec_value =
                        this->get_bottom_tensor_view().template get_vectorized_elements<vector_t>(
                            bottom_tensor_thread_coord,
                            linear_off + ys_offset,
                            bool_constant<oob_conditional_check>{});
                    // write into distributed tensor
                    static_for<0, Traits::ScalarPerVector, Traits::PackedSize>{}([&](auto j) {
                        constexpr auto idx_ys = generate_tuple(
                            [&](auto jj) {
                                return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                                : idx_ys_start[jj];
                            },
                            number<Base::NDimY>{});

                        constexpr index_t d =
                            tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys) /
                            Traits::PackedSize;

                        dst_tensor.get_thread_buffer().template at<d>() =
                            vec_value
                                .template get_as<typename Base::DataType>()[j / Traits::PackedSize];
                    });
                    // move thread coordinate
                    if constexpr(!static_move_ys && iCoordAccess != (NumAccessPerCoord - 1))
                    {
                        constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                        constexpr auto idx_diff_ps_ys =
                            container_concat(generate_tuple([&](auto) { return number<0>{}; },
                                                            number<Base::NDimP>{}),
                                             idx_diff_ys);

                        Base::move_window_adaptor_and_bottom_tensor_thread_coordinate(
                            window_adaptor_thread_coord,
                            bottom_tensor_thread_coord,
                            idx_diff_ps_ys);
                    }
                });
            });
        }
    }

    template <typename DstTile,
              index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true,
              bool pre_nop                = false>
    CK_TILE_DEVICE void load_raw(DstTile& dst_tensor,
                                 number<i_access_unsupport_>          = {},
                                 bool_constant<oob_conditional_check> = {},
                                 bool_constant<pre_nop>               = {}) const
    {
        using Traits   = typename Base::Traits;
        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;
        static constexpr index_t YElementSize =
            typename Base::TileDstr{}.get_ys_to_d_descriptor().get_element_space_size();
        static_assert(YElementSize % (Traits::PackedSize * Traits::ScalarPerVector) == 0);
        using vectorized_tbuf =
            array<vector_t, YElementSize / (Traits::PackedSize * Traits::ScalarPerVector)>;

        constexpr auto tile_dstr = typename Base::TileDstr{};

        auto& dst_vec_tbuf = reinterpret_cast<vectorized_tbuf&>(dst_tensor.get_thread_buffer());

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

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);
                constexpr index_t d =
                    tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys_start) /
                    Traits::PackedSize;
                static_assert(d % Traits::ScalarPerVector == 0);

                this->get_bottom_tensor_view().template get_vectorized_elements_raw<vector_t>(
                    dst_vec_tbuf.template at<d / Traits::ScalarPerVector>(),
                    bottom_tensor_thread_coord,
                    0 /**/,
                    bool_constant<oob_conditional_check>{},
                    pre_nop_);
#if CK_TILE_WORKAROUND_ROCM_6_1_SCRATCH_MEMORY_ISSUE || \
    CK_TILE_WORKAROUND_ROCM_6_2_SCRATCH_MEMORY_ISSUE
                asm volatile(
                    ""); // this is starting from rocm-6.2, but same sympton, reuse this flag
#endif
                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<Base::NDimP>{}),
                        idx_diff_ys);

                    Base::move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    // TODO: currently async load only implemented in inline asm
    template <typename LdsTileWindow_,
              index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true,
              bool pre_nop                = false>
    CK_TILE_DEVICE void async_load_raw(LdsTileWindow_&& lds_tile,
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

        // Use VALU so the compiler can optimize redundant/repeated computations
        const index_t m0_init_value =
            size_per_buf + size_per_wave * get_warp_id(/*ReturnSgpr=*/bool_constant<false>{});
        m0_set_with_memory(
            amd_wave_read_first_lane(m0_init_value)); // This should be wave independent

        using Traits = typename Base::Traits;

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

                // read from bottom tensor
                this->get_bottom_tensor_view().template async_get_vectorized_elements_raw<vector_t>(
                    smem, bottom_tensor_thread_coord, 0, pre_nop_);

                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<Base::NDimP>{}),
                        idx_diff_ys);

                    Base::move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);

                    m0_inc_with_memory(size_per_issue);
                }
            });
        });
    }

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
        using Traits        = typename Base::Traits;

        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        // Precompute invariant values outside loops
        const auto window_origin       = lds_tile.get_window_origin();
        const auto& bottom_tensor_view = lds_tile.get_bottom_tensor_view();
        const auto& tensor_descriptor  = bottom_tensor_view.get_tensor_descriptor();
        auto lds_base_ptr              = bottom_tensor_view.get_buffer_view().p_data_;
#if defined(__gfx125__)
        // this is an optimization used in gfx125 where lds descriptor don't include xor swizzle
        if constexpr(!remove_cvref_t<decltype(tensor_descriptor)>::template has_transform<
                         coord_transform_enum::xor_t>() &&
                     static_move_ys == false)
        {
            static_assert(
                []() constexpr {
                    [[maybe_unused]] constexpr auto desc =
                        LdsTileWindow{}.get_bottom_tensor_view().get_tensor_descriptor();
                    return true;
                }(),
                "LdsTileWindow::get_tensor_descriptor() must be constexpr");

            // For LDS, compute offsets at compile time to optimize LDS access
            static_for<0, NumCoord, 1>{}([&](auto iCoord) {
                /// TODO: use structure binding (to be captured later) if compiled in C++20
                auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
                auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];
                constexpr index_t dram_ys_offset = 0;
                constexpr index_t lds_ys_offset  = 0;

                auto lds_bottom_tensor_thread_idx =
                    window_origin + window_adaptor_thread_coord.get_bottom_index();
                const auto lds_origin_coord =
                    make_tensor_coordinate(tensor_descriptor, lds_bottom_tensor_thread_idx);

                // Calculate SMEM address using base pointer
                CK_TILE_LDS_ADDR LdsDataType* smem =
                    lds_base_ptr + lds_origin_coord.get_offset() / Traits::PackedSize +
                    lds_ys_offset / Traits::PackedSize;

                static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                    constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                    // Compute compile-time offset from access 0 to current access
                    constexpr auto tile_dstr         = typename Base::TileDstr{};
                    constexpr auto lds_access_offset = [&]() {
                        constexpr auto idx_off_ys = SFC_Ys::get_step_between(number<0>{}, iAccess);
                        constexpr auto adapter_ys_offset = make_tensor_adaptor_coordinate(
                            tile_dstr.get_ps_ys_to_xs_adaptor(),
                            container_concat(array<index_t, Base::NDimP>{0},
                                             to_array<index_t, idx_off_ys.size()>(idx_off_ys)));
                        constexpr auto coord_ys_offset = make_tensor_coordinate(
                            LdsTileWindow{}.get_bottom_tensor_view().get_tensor_descriptor(),
                            adapter_ys_offset.get_bottom_index());
                        return coord_ys_offset.get_offset();
                    }();

                    this->get_bottom_tensor_view().template async_get_vectorized_elements<vector_t>(
                        smem,
                        bottom_tensor_thread_coord,
                        offset + dram_ys_offset,
                        number<lds_access_offset>{},
                        bool_constant<oob_conditional_check>{});

                    // Move thread coordinate if not last access
                    if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                    {
                        constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);
                        constexpr auto idx_diff_ps_ys =
                            container_concat(generate_tuple([&](auto) { return number<0>{}; },
                                                            number<Base::NDimP>{}),
                                             idx_diff_ys);

                        Base::move_window_adaptor_and_bottom_tensor_thread_coordinate(
                            window_adaptor_thread_coord,
                            bottom_tensor_thread_coord,
                            idx_diff_ps_ys);
                    }
                });
            });
        }
        else
#endif
        {
            static_for<0, NumCoord, 1>{}([&](auto iCoord) {
                auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
                auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];
#if !defined(__gfx125__)
                auto window_adaptor_warp_coord = pre_computed_warp_coords_[iCoord][I0];
                auto bottom_tensor_warp_coord  = pre_computed_warp_coords_[iCoord][I1];
#endif
                static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                    constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                    constexpr auto idx_ys_offset = [&]() {
                        constexpr auto idx_off_ys = SFC_Ys::get_step_between(number<0>{}, iAccess);
                        constexpr auto adapter_ys_offset = make_tensor_adaptor_coordinate(
                            StaticTileDistribution_{}.get_ps_ys_to_xs_adaptor(),
                            container_concat(array<index_t, Base::NDimP>{0},
                                             to_array<index_t, idx_off_ys.size()>(idx_off_ys)));
                        return adapter_ys_offset.get_bottom_index();
                    }();
                    constexpr auto lds_ys_offset = [&]() {
                        if constexpr(static_move_ys)
                        {
                            const auto coord_ys_offset = make_tensor_coordinate(
                                decltype(tensor_descriptor){}, idx_ys_offset);
                            return coord_ys_offset.get_offset();
                        }
                        else
                            return 0;
                    }();

                    // Use precomputed window origin & tensor descriptor
#if defined(__gfx125__)
                    auto lds_bottom_tensor_thread_idx =
                        window_origin + window_adaptor_thread_coord.get_bottom_index();
#else // else branch for gfx950
                    auto lds_bottom_tensor_thread_idx =
                        window_origin + window_adaptor_warp_coord.get_bottom_index();
#endif
                    const auto lds_coord =
                        make_tensor_coordinate(tensor_descriptor, lds_bottom_tensor_thread_idx);

                    // Calculate SMEM address using base pointer
                    CK_TILE_LDS_ADDR LdsDataType* smem =
                        lds_base_ptr + lds_coord.get_offset() / Traits::PackedSize +
                        lds_ys_offset / Traits::PackedSize;

                    const auto dram_ys_offset = [&]() {
                        if constexpr(static_move_ys)
                        {
                            const auto coord_ys_offset = make_tensor_coordinate(
                                this->get_bottom_tensor_view().get_tensor_descriptor(),
                                idx_ys_offset);
                            return coord_ys_offset.get_offset();
                        }
                        else
                            return 0;
                    }();

                    if constexpr(!static_move_ys)
                        this->get_bottom_tensor_view()
                            .template async_get_vectorized_elements<vector_t>(
                                smem,
                                bottom_tensor_thread_coord,
                                offset + dram_ys_offset,
                                bool_constant<oob_conditional_check>{});
                    else
                    {
                        this->get_bottom_tensor_view()
                            .template async_get_vectorized_elements<vector_t>(
                                smem,
                                bottom_tensor_thread_coord.get_offset() + offset,
                                dram_ys_offset,
                                number<0>{},
                                bool_constant<oob_conditional_check>{});
                    }
                    // Move thread coordinate if not last access
                    if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                    {
                        constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);
                        constexpr auto idx_diff_ps_ys =
                            container_concat(generate_tuple([&](auto) { return number<0>{}; },
                                                            number<Base::NDimP>{}),
                                             idx_diff_ys);

                        if constexpr(!static_move_ys)
                            Base::move_window_adaptor_and_bottom_tensor_thread_coordinate(
                                window_adaptor_thread_coord,
                                bottom_tensor_thread_coord,
                                idx_diff_ps_ys);
#if !defined(__gfx125__)
                        if constexpr(!static_move_ys)
                            Base::move_window_adaptor_and_bottom_tensor_thread_coordinate(
                                window_adaptor_warp_coord,
                                bottom_tensor_warp_coord,
                                idx_diff_ps_ys);
#endif
                    }
                });
            });
        }
    }

    template <typename TDMConfig_,
              typename LdsTileWindow_,
              typename GatherIndexView_,
              index_t i_access_ = -1>
    CK_TILE_DEVICE auto tdm_load_to_lds(const TDMConfig_& tdm_config,
                                        LdsTileWindow_&& lds_tile,
                                        const GatherIndexView_& gather_index_view,
                                        number<i_access_> = {}) const
    {
        using LdsTileWindow = remove_cvref_t<LdsTileWindow_>;
        using LdsDataType   = typename LdsTileWindow::DataType;

        static_assert(std::is_same_v<LdsDataType, typename Base::DataType>,
                      "LdsDataType must match tile window's DataType");

        using Traits             = typename Base::Traits;
        constexpr auto tile_dstr = typename Base::TileDstr{};

        static constexpr index_t num_tensor_dims = BottomTensorView_::get_num_of_dimension();

        const auto lds_window_origin       = lds_tile.get_window_origin();
        const auto& lds_bottom_tensor_view = lds_tile.get_bottom_tensor_view();
        const auto& lds_tensor_descriptor  = lds_bottom_tensor_view.get_tensor_descriptor();
        auto smem_base_ptr                 = lds_bottom_tensor_view.get_buffer_view().p_data_;

        const auto& glb_tensor_descriptor = this->get_bottom_tensor_view().get_tensor_descriptor();

        // Use cached computation for global strides
        auto&& global_strides = get_cached_global_strides();

        auto process_coord = [&](auto iCoord) {
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0]; // without origin
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1]; // with origin

            auto lds_bottom_tensor_thread_idx =
                lds_window_origin + window_adaptor_thread_coord.get_bottom_index();

            // tdm's box dim is reversed from tile distribution
            constexpr auto raw_box_dim =
                to_sequence(tile_dstr.get_ys_to_d_descriptor().get_lengths()).reverse();

            constexpr auto box_dim = raw_box_dim.modify(
                number<0>{}, number<raw_box_dim.at(number<0>{}) / Traits::PackedSize>{});
            // Use precomputed tensor descriptor
            const auto lds_coord =
                make_tensor_coordinate(lds_tensor_descriptor, lds_bottom_tensor_thread_idx);

            // Calculate SMEM address using base pointer
            CK_TILE_LDS_ADDR LdsDataType* smem =
                smem_base_ptr + lds_coord.get_offset() / Traits::PackedSize;

            // Calculate remaining tensor dimensions, clamping negative values to 0
            // This prevents out-of-bounds access when window_origin + bottom_index > tensor_length
            auto&& tensor_dims = to_array<index_t, Base::NDimBottomTensor>(tuple_reverse(
                transform_tuples([](auto x) { return max(index_t{0}, x); },
                                 glb_tensor_descriptor.get_lengths() - this->get_window_origin() -
                                     window_adaptor_thread_coord.get_bottom_index())));
            tensor_dims[0] /= Traits::PackedSize;
            // Assert that both window origins have the same dimensionality
            static_assert(
                std::is_same<std::remove_cv_t<std::remove_reference_t<decltype(lds_window_origin)>>,
                             std::remove_cv_t<std::remove_reference_t<
                                 decltype(this->get_window_origin())>>>::value,
                "Window origin types mismatch - dimensions must be consistent!");
            // if GatherIndexView_ is null_tile_window, then we are doing TDM load
            if constexpr(is_null_tile_window_v<GatherIndexView_>)
            {
                this->get_bottom_tensor_view()
                    .template get_tdm_elements<TDMConfig_,
                                               remove_cvref_t<decltype(box_dim)>,
                                               num_tensor_dims>(tdm_config,
                                                                smem,
                                                                bottom_tensor_thread_coord,
                                                                tensor_dims,
                                                                global_strides,
                                                                number<num_tensor_dims>{});
            }
            // if GatherIndexView_ is not null_tile_view, then we are doing TDM gather
            else
            {
                constexpr index_t RowNumPerTDMIter =
                    std::is_same_v<typename GatherIndexView_::DataType, uint16_t> ? 16 : 8;
                constexpr index_t NumIterations = i_access_ / RowNumPerTDMIter;

                static_for<0, NumIterations, 1>{}([&](auto iIter) {
                    this->get_bottom_tensor_view()
                        .template get_tdm_elements<TDMConfig_,
                                                   remove_cvref_t<decltype(box_dim)>,
                                                   num_tensor_dims>(
                            tdm_config,
                            smem,
                            bottom_tensor_thread_coord,
                            tensor_dims,
                            global_strides,
                            number<num_tensor_dims>{},
                            gather_index_view.get_bottom_tensor_view(),
                            number<iIter * RowNumPerTDMIter>{});
                });
            }
        };

        if constexpr(is_null_tile_window_v<GatherIndexView_>)
        {
            ignore = gather_index_view;
        }

        static_for<0, NumCoord, 1>{}(process_coord);
    }

#if defined(__gfx125__)
    template <DataCachePrefetchKind PrefetchKind = DataCachePrefetchKind::None>
    static constexpr index_t getCachelineSize()
    {
        static_assert(PrefetchKind != DataCachePrefetchKind::None,
                      "getCachelineSize() called with DataCachePrefetchKind::None; "
                      "prefetching must target L1 or L2");
        if constexpr(PrefetchKind == DataCachePrefetchKind::L1)
            return 32; // L1 cacheline size in bytes for gfx125
        else
            return 256; // L2 cacheline size in bytes for gfx125
    }
#endif

    // NOTE:
    // We assume that the prefetch_for_tdm call starts with coordinates aligned to cacheline size
    // i.e for 32 byte cacheline they're aligned to 32. We also assume the step coordinate that is
    // moving in contiguous dimension is at the last dimension of the tile distribution (i.e x
    // dimension in row-major layout), and we only consider the step in that dimension for prefetch
    // coverage calculation.
    template <DataCachePrefetchKind PrefetchKind = DataCachePrefetchKind::None,
              typename DramTileWindowStep>
    CK_TILE_DEVICE constexpr index_t
    prefetch_for_tdm_covers_more_calls([[maybe_unused]] const DramTileWindowStep& step)
    {
        if constexpr(PrefetchKind == DataCachePrefetchKind::None)
            return 0;
#if defined(__gfx125__)
        // TODO: move it somewhere and call when we need these values
        constexpr index_t cacheline_size = getCachelineSize<PrefetchKind>();

        using Traits             = typename Base::Traits;
        constexpr auto tile_dstr = typename Base::TileDstr{};

        // Get tile dimensions
        constexpr auto raw_box_dim =
            to_sequence(tile_dstr.get_ys_to_d_descriptor().get_lengths()).reverse();

        const index_t x_step = step.at(number<DramTileWindowStep{}.size() - 1>{});
        if(x_step == 0)
            return 0; // if step is 0, it means we are not moving in that dimension, so prefetch
                      // won't cover more calls

        const index_t bytes_per_x_step =
            x_step * Traits::PackedSize * sizeof(typename Base::DataType);

        constexpr index_t cacheline_part_covered_by_prefetch_for_tdm =
            raw_box_dim.at(number<0>{}) * sizeof(typename Base::DataType);

        const index_t additional_prefetches_covered =
            max(0,
                (cacheline_size - cacheline_part_covered_by_prefetch_for_tdm) /
                    bytes_per_x_step); // we don't want negatives
        return additional_prefetches_covered;
#else
        return 0;
#endif
    }
    // Prefetch DRAM memory that would be accessed by TDM load
    // Similar to tdm_load_to_lds but issues cache prefetch hints instead of loading to LDS
    // We try to fill entire wave with multiple rows and columns per single call to prefetch
    // For OOB we set is_valid to false
    // For now TDMConfig_ is unused, but we keep it for future use when maybe TDM will have prefetch
    // config
    template <DataCachePrefetchKind PrefetchKind = DataCachePrefetchKind::None, typename TDMConfig_>
    CK_TILE_DEVICE void prefetch_for_tdm([[maybe_unused]] const TDMConfig_& tdm_config) const
    {
        if constexpr(PrefetchKind == DataCachePrefetchKind::None)
            return;
#if defined(__gfx125__)
        // TODO: move it somewhere and call when we need these values
        constexpr index_t cacheline_size   = getCachelineSize<PrefetchKind>();
        constexpr auto preferred_coherence = PrefetchKind == DataCachePrefetchKind::L1
                                                 ? amd_buffer_coherence_enum::CU_RT
                                                 : amd_buffer_coherence_enum::SE_RT;

        using Traits             = typename Base::Traits;
        constexpr auto tile_dstr = typename Base::TileDstr{};

        // Use cached computation for global strides (same as tdm_load_to_lds)
        auto&& global_strides = get_cached_global_strides();

        const auto& glb_tensor_descriptor = this->get_bottom_tensor_view().get_tensor_descriptor();

        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0]; // without origin
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1]; // with origin

            // Get tile dimensions
            constexpr auto raw_box_dim =
                to_sequence(tile_dstr.get_ys_to_d_descriptor().get_lengths()).reverse();
            constexpr index_t x_len = raw_box_dim.at(number<0>{}) / Traits::PackedSize;
            constexpr index_t y_len = (raw_box_dim.size() > 1 ? raw_box_dim.at(number<1>{}) : 1);

            // Calculate remaining tensor dimensions, clamping negative values to 0
            // This prevents out-of-bounds access when window_origin + bottom_index > tensor_length
            auto&& tensor_dims = to_array<index_t, Base::NDimBottomTensor>(tuple_reverse(
                transform_tuples([](auto x) { return max(index_t{0}, x); },
                                 glb_tensor_descriptor.get_lengths() - this->get_window_origin() -
                                     window_adaptor_thread_coord.get_bottom_index())));
            tensor_dims[0] /= Traits::PackedSize;

            // Prefetch across the 2D tile using strides
            // Distribute column prefetches across lanes - each lane prefetches different x
            // positions
            constexpr index_t col_prefetch_stride =
                max(1,
                    static_cast<index_t>(
                        cacheline_size /
                        (Traits::PackedSize *
                         sizeof(typename Base::DataType)))); // prefetch every cacheline bytes in
                                                             // packed element units
            constexpr index_t num_lanes = get_warp_size();

            // Calculate how many lanes needed to cover one row
            constexpr index_t num_unique_x  = max(1, x_len / col_prefetch_stride);
            constexpr index_t lanes_per_row = num_unique_x < num_lanes ? num_unique_x : num_lanes;
            constexpr index_t num_rows_parallel =
                num_lanes / lanes_per_row; // how many rows we can process in parallel

            // Determine which row and column offset this lane handles
            const index_t y_lane_offset = (get_lane_id() / lanes_per_row) % y_len;
            const index_t x_lane_offset = (get_lane_id() % lanes_per_row) * col_prefetch_stride;

            // Get base offset for this thread's starting position
            const auto base_offset = bottom_tensor_thread_coord.get_offset();

            constexpr index_t num_x_iterations =
                integer_divide_ceil(x_len, lanes_per_row * col_prefetch_stride);
            constexpr index_t num_y_iterations = integer_divide_ceil(y_len, num_rows_parallel);
            constexpr auto box_dim             = [&]() {
                if constexpr(raw_box_dim.size() > 1)
                {
                    return raw_box_dim.modify(number<0>{}, number<num_x_iterations>{})
                        .modify(number<1>{}, number<num_y_iterations>{});
                }
                else
                {
                    return raw_box_dim.modify(number<0>{}, number<num_x_iterations>{});
                }
            }();

            // Create reverse iteration order: dimension 0 moves fastest
            constexpr auto reverse_order =
                typename arithmetic_sequence_gen<box_dim.size() - 1, -1, -1>::type{};
            static_ford<decltype(box_dim), remove_cvref_t<decltype(reverse_order)>>{}(
                [&](auto box_dim_idx) {
                    const index_t x =
                        x_lane_offset + box_dim_idx[I0] * lanes_per_row * col_prefetch_stride;
                    index_t prefetch_offset = base_offset + x * Traits::PackedSize;
                    bool is_valid           = x < tensor_dims[0];

                    if constexpr(box_dim.size() > 1)
                    {
                        const index_t y = y_lane_offset + box_dim_idx[I1] * num_rows_parallel;
                        prefetch_offset += y * global_strides[0];
                        is_valid = is_valid && y < tensor_dims[1];
                    }

                    static_for<2, box_dim.size(), 1>{}([&](auto i) {
                        prefetch_offset += box_dim_idx[i] * global_strides[i - 1];
                        is_valid = is_valid && box_dim_idx[i] < tensor_dims[i];
                    });

                    using DataType = typename Base::DataType;
                    this->get_bottom_tensor_view()
                        .get_buffer_view()
                        .template prefetch<DataType, preferred_coherence>(
                            0, prefetch_offset, is_valid);
                });
        });
#endif
    }

    // NOTE:
    // We assume that the prefetch_for_flat call starts with coordinates aligned to cacheline size
    // i.e for 32 byte cacheline they're aligned to 32. We also assume the step coordinate that is
    // moving in contiguous dimension is at the last dimension of the tile distribution (i.e x
    // dimension in row-major layout), and we only consider the step in that dimension for prefetch
    // coverage calculation.
    // NWaveN_/NWaveK_ are accepted for API symmetry but do not affect coverage.
    template <DataCachePrefetchKind PrefetchKind = DataCachePrefetchKind::None,
              index_t NWaveN_                    = 1,
              index_t NWaveK_                    = 1,
              typename DramTileWindowStep>
    CK_TILE_DEVICE constexpr index_t
    prefetch_for_flat_covers_more_calls([[maybe_unused]] const DramTileWindowStep& step) const
    {
        if constexpr(PrefetchKind == DataCachePrefetchKind::None)
            return 0;
#if defined(__gfx125__)
        constexpr index_t cacheline_size = getCachelineSize<PrefetchKind>();
        using Traits                     = typename Base::Traits;

        const index_t x_step = step.at(number<DramTileWindowStep{}.size() - 1>{});
        if(x_step == 0)
            return 0;

        const index_t bytes_per_x_step =
            x_step * Traits::PackedSize * sizeof(typename Base::DataType);

        // bytes covered by the full K extent of the window
        constexpr auto win_lengths = typename Base::WindowLengths{};
        constexpr index_t x_len_bytes =
            win_lengths.at(number<1>{}) * sizeof(typename Base::DataType);
        // how many bytes the last cacheline extends past the window's K end
        constexpr index_t cacheline_overhang =
            (cacheline_size - x_len_bytes % cacheline_size) % cacheline_size;

        const index_t additional_prefetches_covered =
            max(0, static_cast<index_t>(cacheline_overhang) / bytes_per_x_step);
        return additional_prefetches_covered;
#else
        return 0;
#endif
    }

    // NWaveN_: number of N-direction warps per block (e.g. BlockWarps::at(I1)).
    // NWaveK_: number of K-direction warps per block (e.g. BlockWarps::at(I2)).
    // NWaveN/MWaveK used to partition the tile among warps, but only in the N dimension, so they
    // don't affect coverage calculation. They are used here to determine which rows each warp
    // should prefetch to minimize cross-warp redundancy(i.e. to not prefetch the same data in each
    // warp).
    template <DataCachePrefetchKind PrefetchKind = DataCachePrefetchKind::None,
              index_t NWaveN_                    = 1,
              index_t NWaveK_                    = 1>
    CK_TILE_DEVICE void prefetch_for_flat() const
    {
        if constexpr(PrefetchKind == DataCachePrefetchKind::None)
            return;
#if defined(__gfx125__)
        constexpr index_t cacheline_size   = getCachelineSize<PrefetchKind>();
        constexpr auto preferred_coherence = PrefetchKind == DataCachePrefetchKind::L1
                                                 ? amd_buffer_coherence_enum::CU_RT
                                                 : amd_buffer_coherence_enum::SE_RT;

        using Traits = typename Base::Traits;

        auto&& global_strides             = get_cached_global_strides();
        const auto& glb_tensor_descriptor = this->get_bottom_tensor_view().get_tensor_descriptor();

        // Use window lengths (X-space) instead of ys_to_d lengths (Y-space)
        constexpr auto win_lengths = typename Base::WindowLengths{};
        constexpr index_t x_len    = win_lengths.at(number<1>{}) / Traits::PackedSize;
        constexpr index_t y_len    = win_lengths.at(number<0>{});

        // Partition N-rows among N-warps using ceil-div so every warp gets at least one row even
        // when y_len < NWaveN_.  The actual rows covered by a warp are clamped against y_len in
        // the is_valid predicate, so warps whose base exceeds y_len simply issue no prefetches.
        constexpr index_t y_per_wave = max(index_t{1}, integer_divide_ceil(y_len, NWaveN_));

        // n_wave_id = which N-warp this thread belongs to.
        const index_t n_wave_id   = (get_warp_id() / NWaveK_) % NWaveN_;
        const index_t y_wave_base = n_wave_id * y_per_wave;

        // Base from window origin (warp-level, same for all lanes), not per-thread coords.
        const auto win_origin_coord =
            make_tensor_coordinate(glb_tensor_descriptor, this->get_window_origin());
        const auto base_offset = win_origin_coord.get_offset() / Traits::PackedSize;

        // OOB: remaining tensor extents measured from window origin
        auto&& tensor_dims = to_array<index_t, Base::NDimBottomTensor>(tuple_reverse(
            transform_tuples([](auto x) { return max(index_t{0}, x); },
                             glb_tensor_descriptor.get_lengths() - this->get_window_origin())));
        tensor_dims[0] /= Traits::PackedSize;

        // Distribute cache-line prefetches across warp lanes
        constexpr index_t col_prefetch_stride =
            max(1,
                static_cast<index_t>(cacheline_size /
                                     (Traits::PackedSize * sizeof(typename Base::DataType))));
        constexpr index_t num_lanes         = get_warp_size();
        constexpr index_t num_unique_x      = max(1, x_len / col_prefetch_stride);
        constexpr index_t lanes_per_row     = num_unique_x < num_lanes ? num_unique_x : num_lanes;
        constexpr index_t num_rows_parallel = num_lanes / lanes_per_row;

        // Lane offset within this warp's N-stripe [0, y_per_wave).
        // y_per_wave >= 1 by construction so the modulus is safe.
        const index_t y_lane_offset = (get_lane_id() / lanes_per_row) % y_per_wave;
        const index_t x_lane_offset = (get_lane_id() % lanes_per_row) * col_prefetch_stride;

        constexpr index_t num_x_iterations =
            integer_divide_ceil(x_len, lanes_per_row * col_prefetch_stride);
        constexpr index_t num_y_iterations = integer_divide_ceil(y_per_wave, num_rows_parallel);

        constexpr auto box_dim = sequence<num_x_iterations, num_y_iterations>{};
        constexpr auto reverse_order =
            typename arithmetic_sequence_gen<box_dim.size() - 1, -1, -1>::type{};

        static_ford<decltype(box_dim), remove_cvref_t<decltype(reverse_order)>>{}(
            [&](auto box_dim_idx) {
                const index_t x =
                    x_lane_offset + box_dim_idx[I0] * lanes_per_row * col_prefetch_stride;
                const index_t y = y_wave_base + y_lane_offset + box_dim_idx[I1] * num_rows_parallel;

                index_t prefetch_offset = base_offset + x + y * global_strides[0];
                bool is_valid           = x < tensor_dims[0] && y < tensor_dims[1];

                static_for<2, box_dim.size(), 1>{}([&](auto i) {
                    prefetch_offset += box_dim_idx[i] * global_strides[i - 1];
                    is_valid = is_valid && box_dim_idx[i] < tensor_dims[i];
                });

                using DataType = typename Base::DataType;
                this->get_bottom_tensor_view()
                    .get_buffer_view()
                    .template prefetch<DataType, preferred_coherence>(0, prefetch_offset, is_valid);
            });
#endif
    }

    template <typename TDMConfig_, typename LdsTileWindow_, index_t i_access_unsupport_ = -1>
    CK_TILE_DEVICE auto tdm_store_from_lds(const TDMConfig_& tdm_config,
                                           const LdsTileWindow_& lds_tile,
                                           number<i_access_unsupport_> = {}) const
    {
        using LdsTileWindow = remove_cvref_t<LdsTileWindow_>;
        using LdsDataType   = typename LdsTileWindow::DataType;

        static_assert(std::is_same_v<LdsDataType, typename Base::DataType>,
                      "LdsDataType must match tile window's DataType");
        using Traits             = typename Base::Traits;
        constexpr auto tile_dstr = typename Base::TileDstr{};

        static constexpr index_t num_tensor_dims = BottomTensorView_::get_num_of_dimension();

        const auto& glb_tensor_descriptor = this->get_bottom_tensor_view().get_tensor_descriptor();

        const auto lds_window_origin       = lds_tile.get_window_origin();
        const auto& lds_bottom_tensor_view = lds_tile.get_bottom_tensor_view();
        const auto& lds_tensor_descriptor  = lds_bottom_tensor_view.get_tensor_descriptor();
        auto smem_base_ptr                 = lds_bottom_tensor_view.get_buffer_view().p_data_;

        // Use cached computation for global strides
        auto&& global_strides = get_cached_global_strides();

        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0]; // without origin
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1]; // with origin

            auto lds_bottom_tensor_thread_idx =
                lds_window_origin + window_adaptor_thread_coord.get_bottom_index();

            // Calculate remaining tensor dimensions, clamping negative values to 0
            // This prevents out-of-bounds access when window_origin + bottom_index >
            // tensor_length
            auto&& tensor_dims = to_array<index_t, Base::NDimBottomTensor>(tuple_reverse(
                transform_tuples([](auto x) { return max(index_t{0}, x); },
                                 glb_tensor_descriptor.get_lengths() - this->get_window_origin() -
                                     window_adaptor_thread_coord.get_bottom_index())));
            tensor_dims[0] /= Traits::PackedSize;

            constexpr auto raw_box_dim =
                to_sequence(tile_dstr.get_ys_to_d_descriptor().get_lengths()).reverse();

            constexpr auto box_dim = raw_box_dim.modify(
                number<0>{}, number<raw_box_dim.at(number<0>{}) / Traits::PackedSize>{});
            // Use precomputed tensor descriptor
            const auto lds_coord =
                make_tensor_coordinate(lds_tensor_descriptor, lds_bottom_tensor_thread_idx);

            // Calculate SMEM address using base pointer
            CK_TILE_LDS_ADDR LdsDataType* smem =
                smem_base_ptr + lds_coord.get_offset() / Traits::PackedSize;
            // Assert that both window origins have the same dimensionality
            static_assert(
                std::is_same<std::remove_cv_t<std::remove_reference_t<decltype(lds_window_origin)>>,
                             std::remove_cv_t<std::remove_reference_t<
                                 decltype(this->get_window_origin())>>>::value,
                "Window origin types mismatch - dimensions must be consistent!");

            this->get_bottom_tensor_view()
                .template store_tdm_elements<TDMConfig_,
                                             remove_cvref_t<decltype(box_dim)>,
                                             num_tensor_dims>(tdm_config,
                                                              smem,
                                                              bottom_tensor_thread_coord,
                                                              tensor_dims,
                                                              global_strides,
                                                              number<num_tensor_dims>{});
        });
    }

    template <typename Policy, index_t i_access_unsupport_ = -1, bool oob_conditional_check = true>
    CK_TILE_DEVICE auto load_transpose(number<i_access_unsupport_>          = {},
                                       bool_constant<oob_conditional_check> = {}) const
    {
        return this->template load_transpose_with_offset<Policy>(
            0, number<i_access_unsupport_>{}, bool_constant<oob_conditional_check>{});
    }

    template <typename Policy, index_t i_access_unsupport_ = -1, bool oob_conditional_check = true>
    CK_TILE_DEVICE auto load_transpose_with_offset(index_t offset,
                                                   number<i_access_unsupport_>          = {},
                                                   bool_constant<oob_conditional_check> = {}) const
    {
        constexpr auto tile_dstr = typename Base::TileDstr{};
        auto dst_tensor = make_static_distributed_tensor<typename Base::DataType>(tile_dstr);
        this->template load_transpose_with_offset<Policy>(offset,
                                                          dst_tensor,
                                                          number<i_access_unsupport_>{},
                                                          bool_constant<oob_conditional_check>{});
        return dst_tensor;
    }

    template <typename Policy,
              typename DistributedTensor,
              index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true>
    CK_TILE_DEVICE void load_transpose_with_offset(index_t offset,
                                                   DistributedTensor& dst_tensor,
                                                   number<i_access_unsupport_>          = {},
                                                   bool_constant<oob_conditional_check> = {}) const
    {
        using Traits   = typename Base::Traits;
        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        constexpr auto tile_dstr = typename Base::TileDstr{};

        constexpr auto group_func = Policy::group_func;

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);

                // read from bottom tensor
                const vector_t vec_value =
                    this->get_bottom_tensor_view()
                        .template get_transpose_vectorized_elements<vector_t>(
                            bottom_tensor_thread_coord, offset);
                // write into distributed tensor
                static_for<0, Traits::ScalarPerVector, Traits::PackedSize>{}([&](auto j) {
                    constexpr auto orig_idx_ys = generate_tuple(
                        [&](auto jj) {
                            return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                            : idx_ys_start[jj];
                        },
                        number<Base::NDimY>{});

                    constexpr auto grouped_idx_ys = group_func(orig_idx_ys);

                    constexpr index_t linear_distributed_index =
                        tile_dstr.get_ys_to_d_descriptor().calculate_offset(grouped_idx_ys) /
                        Traits::PackedSize;

                    dst_tensor.get_thread_buffer().template at<linear_distributed_index>() =
                        vec_value
                            .template get_as<typename Base::DataType>()[j / Traits::PackedSize];
                });
                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<Base::NDimP>{}),
                        idx_diff_ys);

                    Base::move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    template <index_t i_access_unsupport_ = -1,
              bool oob_conditional_check  = true,
              bool static_move_ys         = false>
    CK_TILE_DEVICE void store(const static_distributed_tensor<typename Base::DataType,
                                                              typename Base::TileDstr>& dstr_tensor,
                              number<i_access_unsupport_>          = {},
                              bool_constant<oob_conditional_check> = {},
                              bool_constant<static_move_ys>        = {}) const
    {
        using Traits = typename Base::Traits;

        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        constexpr auto tile_dstr = typename Base::TileDstr{};

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                constexpr auto idx_ys_offset = [&]() {
                    constexpr auto idx_off_ys = SFC_Ys::get_step_between(number<0>{}, iAccess);
                    constexpr auto adapter_ys_offset = make_tensor_adaptor_coordinate(
                        StaticTileDistribution_{}.get_ps_ys_to_xs_adaptor(),
                        container_concat(array<index_t, Base::NDimP>{0},
                                         to_array<index_t, idx_off_ys.size()>(idx_off_ys)));
                    return adapter_ys_offset.get_bottom_index();
                }();
                const auto ys_offset = [&]() {
                    if constexpr(static_move_ys)
                    {
                        const auto coord_ys_offset = make_tensor_coordinate(
                            this->get_bottom_tensor_view().get_tensor_descriptor(), idx_ys_offset);
                        return coord_ys_offset.get_offset();
                    }
                    else
                        return 0;
                }();

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);

                // read from distributed tensor
                // vector_type_t vec;
                vector_t vec_value;

                static_for<0, Traits::ScalarPerVector, Traits::PackedSize>{}([&](auto j) {
                    constexpr auto idx_ys = generate_tuple(
                        [&](auto jj) {
                            return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                            : idx_ys_start[jj];
                        },
                        number<Base::NDimY>{});

                    constexpr index_t d =
                        tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys) /
                        Traits::PackedSize;

                    vec_value.template get_as<typename Base::DataType>()(j / Traits::PackedSize) =
                        dstr_tensor.get_thread_buffer().template at<d>();
                });

                // const vector_t vec_value = vec.template get_as<vector_t>().template at<0>();

                // write into bottom tensor
                this->get_bottom_tensor_view().template set_vectorized_elements<vector_t>(
                    bottom_tensor_thread_coord,
                    ys_offset,
                    vec_value,
                    bool_constant<oob_conditional_check>{});

                // move thread coordinate
                if constexpr(!static_move_ys && iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<Base::NDimP>{}),
                        idx_diff_ys);

                    Base::move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    template <index_t i_access_unsupport_ = -1>
    CK_TILE_DEVICE void
    store_raw(const static_distributed_tensor<typename Base::DataType, typename Base::TileDstr>&
                  dstr_tensor,
              number<i_access_unsupport_> = {}) const
    {
        using Traits = typename Base::Traits;

        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        constexpr auto tile_dstr                    = typename Base::TileDstr{};
        static constexpr bool oob_conditional_check = true;

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);

                // read from distributed tensor
                vector_t vec_value;
                static_for<0, Traits::ScalarPerVector, Traits::PackedSize>{}([&](auto j) {
                    constexpr auto idx_ys = generate_tuple(
                        [&](auto jj) {
                            return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                            : idx_ys_start[jj];
                        },
                        number<Base::NDimY>{});
                    constexpr index_t d =
                        tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys) /
                        Traits::PackedSize;
                    vec_value.template get_as<typename Base::DataType>()(j / Traits::PackedSize) =
                        dstr_tensor.get_thread_buffer().template at<d>();
                });

                // write into bottom tensor
                this->get_bottom_tensor_view()
                    .template set_vectorized_elements_raw<vector_t, oob_conditional_check>(
                        bottom_tensor_thread_coord, 0, vec_value);

                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<Base::NDimP>{}),
                        idx_diff_ys);

                    Base::move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    template <index_t i_access_unsupport_ = -1, bool oob_conditional_check = true>
    CK_TILE_DEVICE void
    update(const static_distributed_tensor<typename Base::DataType, typename Base::TileDstr>&
               dstr_tensor,
           number<i_access_unsupport_>          = {},
           bool_constant<oob_conditional_check> = {}) const
    {
        using Traits = typename Base::Traits;

        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        constexpr auto tile_dstr = typename Base::TileDstr{};

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);

                // read from distributed tensor
                vector_t vec_value;

                static_for<0, Traits::ScalarPerVector, Traits::PackedSize>{}([&](auto j) {
                    constexpr auto idx_ys = generate_tuple(
                        [&](auto jj) {
                            return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                            : idx_ys_start[jj];
                        },
                        number<Base::NDimY>{});

                    constexpr index_t d =
                        tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys) /
                        Traits::PackedSize;

                    vec_value.template get_as<typename Base::DataType>()(j / Traits::PackedSize) =
                        dstr_tensor.get_thread_buffer().template at<d>();
                });

                // write into bottom tensor
                this->get_bottom_tensor_view().template update_vectorized_elements<vector_t>(
                    bottom_tensor_thread_coord,
                    0,
                    vec_value,
                    bool_constant<oob_conditional_check>{});

                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<Base::NDimP>{}),
                        idx_diff_ys);

                    Base::move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    template <index_t i_access_unsupport_ = -1, bool oob_conditional_check = true, bool pre_nop>
    CK_TILE_DEVICE void
    update_raw(const static_distributed_tensor<typename Base::DataType, typename Base::TileDstr>&
                   dstr_tensor,
               number<i_access_unsupport_>          = {},
               bool_constant<oob_conditional_check> = {},
               bool_constant<pre_nop>               = {}) const
    {
        using Traits = typename Base::Traits;

        using vector_t = typename Traits::vector_t;
        using SFC_Ys   = typename Traits::SFC_Ys;

        constexpr auto tile_dstr = typename Base::TileDstr{};

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::get_index(iAccess);

                // read from distributed tensor
                vector_t vec_value;

                static_for<0, Traits::ScalarPerVector, Traits::PackedSize>{}([&](auto j) {
                    constexpr auto idx_ys = generate_tuple(
                        [&](auto jj) {
                            return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                            : idx_ys_start[jj];
                        },
                        number<Base::NDimY>{});

                    constexpr index_t d =
                        tile_dstr.get_ys_to_d_descriptor().calculate_offset(idx_ys) /
                        Traits::PackedSize;

                    vec_value.template get_as<typename Base::DataType>()(j / Traits::PackedSize) =
                        dstr_tensor.get_thread_buffer().template at<d>();
                });

                // write into bottom tensor
                this->get_bottom_tensor_view().template update_vectorized_elements_raw<vector_t>(
                    bottom_tensor_thread_coord,
                    0,
                    vec_value,
                    bool_constant<oob_conditional_check>{},
                    bool_constant<pre_nop>{});

                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::get_forward_step(iAccess);

                    constexpr auto idx_diff_ps_ys = container_concat(
                        generate_tuple([&](auto) { return number<0>{}; }, number<Base::NDimP>{}),
                        idx_diff_ys);

                    Base::move_window_adaptor_and_bottom_tensor_thread_coordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    // Custom move behavior
    CK_TILE_DEVICE void move_extended(const typename Base::BottomTensorIndex& step)
    {
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            move_tensor_coordinate(this->bottom_tensor_view_.get_tensor_descriptor(),
                                   pre_computed_coords_(iCoord)(I1),
                                   step);
        });

        if constexpr(Base::BottomTensorView::buffer_view::get_address_space() ==
                     address_space_enum::global)
        {
            static_for<0, NumCoord, 1>{}([&](auto iCoord) {
                move_tensor_coordinate(this->bottom_tensor_view_.get_tensor_descriptor(),
                                       pre_computed_warp_coords_(iCoord)(I1),
                                       step);
            });
        }
    }

    CK_TILE_DEVICE void set_window_origin_extended(const typename Base::BottomTensorIndex&)
    {
        // TODO: this use less register for FA, but more register for GEMM
        // need investigation
        const auto window_adaptor_thread_coord_tmp =
            make_tensor_adaptor_coordinate(this->tile_dstr_.get_ps_ys_to_xs_adaptor(),
                                           container_concat(get_partition_index(this->tile_dstr_),
                                                            array<index_t, Base::NDimY>{0}));

        typename Base::BottomTensorIndex bottom_tensor_thread_origin_idx_tmp =
            this->window_origin_ + window_adaptor_thread_coord_tmp.get_bottom_index();

        const auto bottom_tensor_thread_coord_tmp = make_tensor_coordinate(
            this->bottom_tensor_view_.get_tensor_descriptor(), bottom_tensor_thread_origin_idx_tmp);

        // pre-compute NumCoord (WindowAdaptorCoord, BottomTensorCoord) bundles to speed up
        // future load/store() calls (might allocate more registers)
        using Traits = typename Base::Traits;
        using SFC_Ys = typename Traits::SFC_Ys;

        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            auto window_adaptor_thread_coord = window_adaptor_thread_coord_tmp;
            auto bottom_tensor_thread_coord  = bottom_tensor_thread_coord_tmp;

            constexpr auto idx_diff_ys =
                SFC_Ys::get_step_between(number<0>{}, number<iCoord * NumAccessPerCoord>{});

            constexpr auto idx_diff_ps_ys = container_concat(
                generate_tuple([&](auto) { return number<0>{}; }, number<Base::NDimP>{}),
                idx_diff_ys);

            Base::move_window_adaptor_and_bottom_tensor_thread_coordinate(
                window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);

            pre_computed_coords_(iCoord) =
                make_tuple(window_adaptor_thread_coord, bottom_tensor_thread_coord);
        });
    }

    private:
    // Cached computation for global strides
    CK_TILE_DEVICE auto get_cached_global_strides() const
    {
        if(!tensor_cache_initialized_)
        {
            using Traits = typename Base::Traits;
            const auto& glb_tensor_descriptor =
                this->get_bottom_tensor_view().get_tensor_descriptor();
            cached_global_strides_ = to_array<index_t, Base::NDimBottomTensor>(
                transform_tuples([](auto x) { return max(x / Traits::PackedSize, index_t{1}); },
                                 tuple_reverse(container_reverse_inclusive_scan(
                                     glb_tensor_descriptor.get_lengths(), multiplies<>{}, 1))));
            tensor_cache_initialized_ = true;
        }

        return cached_global_strides_;
    }

    // this contains:
    //   per-thread coordinate for window adaptor
    //   per-thread coordinate for bottom tensor
    array<tuple<typename Base::WindowAdaptorCoord, typename Base::BottomTensorCoord>, NumCoord>
        pre_computed_coords_;

    // Cached tensor computation variables
    mutable bool tensor_cache_initialized_ = false;
    mutable typename Base::BottomTensorIndex cached_global_strides_;
    // pre_computed_warp_coords_ exists only in the global memory tile_window
    std::conditional_t<
        Base::BottomTensorView::buffer_view::get_address_space() == address_space_enum::global,
        array<tuple<typename Base::WindowAdaptorCoord, typename Base::BottomTensorCoord>, NumCoord>,
        std::byte>
        pre_computed_warp_coords_;
};

// TODO: use strategy
template <typename TensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          index_t NumCoord = 1>
CK_TILE_DEVICE constexpr auto
make_tile_window(const TensorView_& tensor_view,
                 const WindowLengths_& window_lengths,
                 const multi_index<TensorView_::get_num_of_dimension()>& origin,
                 const StaticTileDistribution_& tile_distribution,
                 number<NumCoord> = {})
{
    return tile_window_with_static_distribution<remove_cvref_t<TensorView_>,
                                                remove_cvref_t<WindowLengths_>,
                                                remove_cvref_t<StaticTileDistribution_>,
                                                NumCoord>{
        tensor_view, window_lengths, origin, tile_distribution};
}

template <typename TensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          index_t NumCoord = 1,
          typename         = std::enable_if_t<is_tensor_view_v<TensorView_> &&
                                              is_tile_distribution_v<StaticTileDistribution_>>>
CK_TILE_DEVICE constexpr auto
make_tile_window(const TensorView_& tensor_view,
                 const WindowLengths_& window_lengths,
                 const multi_index<TensorView_::get_num_of_dimension()>& origin,
                 const StaticTileDistribution_& tile_distribution,
                 decltype(get_partition_index(tile_distribution)) partition_index,
                 number<NumCoord> = {})
{
    return tile_window_with_static_distribution<remove_cvref_t<TensorView_>,
                                                remove_cvref_t<WindowLengths_>,
                                                remove_cvref_t<StaticTileDistribution_>,
                                                NumCoord>{
        tensor_view, window_lengths, origin, tile_distribution, partition_index};
}

// this version can't be called in a constexpr context
template <typename TensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          index_t NumCoord = 1>
CK_TILE_DEVICE auto
make_tile_window_raw(const TensorView_& tensor_view,
                     const WindowLengths_& window_lengths,
                     const multi_index<TensorView_::get_num_of_dimension()>& origin,
                     const StaticTileDistribution_& tile_distribution,
                     number<NumCoord> = {})
{
    auto w = tile_window_with_static_distribution<remove_cvref_t<TensorView_>,
                                                  remove_cvref_t<WindowLengths_>,
                                                  remove_cvref_t<StaticTileDistribution_>,
                                                  NumCoord>{
        tensor_view, window_lengths, origin, tile_distribution};
    w.init_raw();
    return w;
}

template <typename TensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          index_t NumCoord>
CK_TILE_DEVICE void move_tile_window(
    tile_window_with_static_distribution<TensorView_,
                                         WindowLengths_,
                                         StaticTileDistribution_,
                                         NumCoord>& window,
    const typename tile_window_with_static_distribution<TensorView_,
                                                        WindowLengths_,
                                                        StaticTileDistribution_,
                                                        NumCoord>::BottomTensorIndex& step)
{
    window.move(step);
}

template <typename TensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          index_t NumCoord>
CK_TILE_DEVICE void move_tile_window(
    tuple<tile_window_with_static_distribution<TensorView_,
                                               WindowLengths_,
                                               StaticTileDistribution_,
                                               NumCoord>>& window,
    const typename tile_window_with_static_distribution<TensorView_,
                                                        WindowLengths_,
                                                        StaticTileDistribution_,
                                                        NumCoord>::BottomTensorIndex& step)
{
    using T = tuple<tile_window_with_static_distribution<TensorView_,
                                                         WindowLengths_,
                                                         StaticTileDistribution_,
                                                         NumCoord>>;

    static constexpr auto N = T::size();
    static_for<0, N, 1>{}([&](auto Is) { window[number<Is>{}].move(step); });
}

template <typename TileWindowWithStaticDistributionType,
          typename StepType,
          typename std::enable_if_t<
              is_detected<is_tuple, TileWindowWithStaticDistributionType>::value>* = nullptr>
CK_TILE_DEVICE void move_tile_window(TileWindowWithStaticDistributionType& window, StepType& step)
{
    static constexpr auto N = TileWindowWithStaticDistributionType::size();
    static_for<0, N, 1>{}([&](auto Is) { window[number<Is>{}].move(step); });
}

/**
 * @brief This class provides description of tile windowed view on the device memory.
 *
 * @note This class does not provide any functions to read or modify device memory.
 *
 * @tparam BottomTensorView_    Class describing & holding device tensor memory.
 * @tparam WindowLengths_       Spatial sizes of windowed view on tensor.
 *
 * Tile window without pre-computed coordinates. Each store_tile() call constructs a
 * tile_window_with_static_distribution internally, paying the full coordinate computation
 * cost. Suitable for one-shot operations or when VGPR budget is too tight to hold the
 * pre-computed coordinates for the window's lifetime.
 */
template <typename BottomTensorView_, typename WindowLengths_>
struct tile_window_with_static_lengths
    : public tile_window_base<tile_window_with_static_lengths<BottomTensorView_, WindowLengths_>,
                              BottomTensorView_,
                              WindowLengths_>
{
    using Base =
        tile_window_base<tile_window_with_static_lengths<BottomTensorView_, WindowLengths_>,
                         BottomTensorView_,
                         WindowLengths_>;

    CK_TILE_DEVICE constexpr tile_window_with_static_lengths() = default;

    CK_TILE_DEVICE constexpr tile_window_with_static_lengths(
        const typename Base::BottomTensorView& bottom_tensor_view,
        const typename Base::WindowLengths& window_lengths,
        const typename Base::BottomTensorIndex& window_origin)
    {
        this->window_origin_      = window_origin;
        this->window_lengths_     = window_lengths;
        this->bottom_tensor_view_ = bottom_tensor_view;
    }

    /**
     * @brief Print tile window elements for debugging.
     *
     * @tparam DataType Element data type (e.g., fp16_t, float, bf8_t)
     * @param start_i Starting row (inclusive)
     * @param end_i   Ending row (exclusive)
     * @param start_j Starting column (inclusive)
     * @param end_j   Ending column (exclusive)
     * @param label   Optional output label
     *
     * @note Tested on fp16. Custom types may need adjustments.
     * @example tile_window.template print_tile_window_range<fp16_t>(0, 4, 0, 8, "A");
     */
    template <typename DataType>
    CK_TILE_DEVICE void print_tile_window_range(index_t start_i,
                                                index_t end_i,
                                                index_t start_j,
                                                index_t end_j,
                                                const char* label = "") const
    {
        const auto& tensor_view  = this->get_bottom_tensor_view();
        const auto window_origin = this->get_window_origin();

        printf("%s Window Range [%d:%d, %d:%d] (origin: %d, %d):\n",
               label,
               start_i,
               end_i - 1,
               start_j,
               end_j - 1,
               window_origin[0],
               window_origin[1]);

        for(index_t i = start_i; i < end_i; i++)
        {
            for(index_t j = start_j; j < end_j; j++)
            {
                // Create coordinate for this element relative to window origin
                auto coord =
                    make_tensor_coordinate(tensor_view.get_tensor_descriptor(),
                                           make_tuple(window_origin[0] + i, window_origin[1] + j));

                // Get the element using thread buffer type directly
                using ThreadBuf = thread_buffer<DataType, 2>;
                auto buf        = tensor_view.template get_vectorized_elements<ThreadBuf>(coord, 0);
                auto value      = buf.at(number<0>{}); // Extract first element from thread buffer
                printf("  %s[%d,%d] = %f", label, i, j, type_convert<float>(value));
            }
            printf("\n");
        }
        printf("\n");
    }
};

template <typename TensorView_,
          typename WindowLengths_,
          typename = std::enable_if_t<is_tensor_view_v<TensorView_>>>
CK_TILE_DEVICE constexpr auto
make_tile_window(const TensorView_& tensor_view,
                 const WindowLengths_& window_lengths,
                 const multi_index<TensorView_::get_num_of_dimension()>& origin)
{
    static_assert(ck_tile::is_known_at_compile_time<WindowLengths_>::value,
                  "wrong! lengths should be static");

    return tile_window_with_static_lengths<remove_cvref_t<TensorView_>,
                                           remove_cvref_t<WindowLengths_>>{
        tensor_view, window_lengths, origin};
}

// duplicate tile window and replace its origin
template <typename TensorView, typename WindowLengths>
CK_TILE_DEVICE constexpr auto
make_tile_window(const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
                 const multi_index<TensorView::get_num_of_dimension()>& origin)
{
    return tile_window_with_static_lengths<TensorView, WindowLengths>{
        tile_window.get_bottom_tensor_view(), tile_window.get_window_lengths(), origin};
}

template <typename TensorView, typename WindowLengths, typename StaticTileDistribution>
CK_TILE_DEVICE constexpr auto
make_tile_window(const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
                 const multi_index<TensorView::get_num_of_dimension()>& origin,
                 const StaticTileDistribution& tile_distribution)
{
    return make_tile_window(tile_window.get_bottom_tensor_view(),
                            tile_window.get_window_lengths(),
                            origin,
                            tile_distribution);
}

template <typename TensorView, typename WindowLengths, typename StaticTileDistribution>
CK_TILE_DEVICE constexpr auto
make_tile_window(const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
                 const StaticTileDistribution& tile_distribution)
{
    return make_tile_window(tile_window.get_bottom_tensor_view(),
                            tile_window.get_window_lengths(),
                            tile_window.get_window_origin(),
                            tile_distribution);
}

template <typename TensorView,
          typename WindowLengths,
          typename StaticTileDistribution,
          typename = std::enable_if_t<is_tile_distribution_v<StaticTileDistribution>>>
CK_TILE_DEVICE constexpr auto
make_tile_window(const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
                 const StaticTileDistribution& tile_distribution,
                 decltype(get_partition_index(tile_distribution)) partition_index)
{
    return make_tile_window(tile_window.get_bottom_tensor_view(),
                            tile_window.get_window_lengths(),
                            tile_window.get_window_origin(),
                            tile_distribution,
                            partition_index);
}

template <typename TensorView, typename WindowLengths, typename StaticTileDistribution>
CK_TILE_DEVICE constexpr auto
make_tile_window_raw(const tile_window_with_static_lengths<TensorView, WindowLengths>& tile_window,
                     const StaticTileDistribution& tile_distribution)
{
    auto w = make_tile_window(tile_window, tile_distribution);
    w.init_raw();
    return w;
}

template <typename TensorView_, typename WindowLengths_>
CK_TILE_DEVICE void move_tile_window(
    tile_window_with_static_lengths<TensorView_, WindowLengths_>& window,
    const typename tile_window_with_static_lengths<TensorView_, WindowLengths_>::BottomTensorIndex&
        step)
{
    window.move(step);
}

template <typename NewTensorView_,
          typename OldTensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          index_t NumCoord = 1>
CK_TILE_DEVICE auto
replace_bottom_tensor_view(const NewTensorView_& new_tensor_view,
                           const tile_window_with_static_distribution<OldTensorView_,
                                                                      WindowLengths_,
                                                                      StaticTileDistribution_,
                                                                      NumCoord>& tile_window)
{
    return make_tile_window(new_tensor_view,
                            tile_window.get_window_lengths(),
                            tile_window.get_window_origin(),
                            tile_window.get_tile_distribution());
}

template <typename NewTensorView_, typename OldTensorView_, typename WindowLengths_>
CK_TILE_DEVICE auto replace_bottom_tensor_view(
    const NewTensorView_& new_tensor_view,
    const tile_window_with_static_lengths<OldTensorView_, WindowLengths_>& tile_window)
{
    return make_tile_window(
        new_tensor_view, tile_window.get_window_lengths(), tile_window.get_window_origin());
}

/**
 * @brief Type trait to determine if a type is a tile window with static distribution.
 *
 * Defaults to `false_type`. Specializations define when the trait evaluates to `true`.
 *
 * @tparam T The type to check.
 */
template <typename T>
struct is_tile_window_with_static_distribution : std::false_type
{
};

/**
 * @brief Specialization for `tile_window_with_static_distribution` to evaluate to `true_type`.
 *
 * @tparam BottomTensorView_ Bottom tensor view type of the tile window.
 * @tparam WindowLengths_ Static window lengths.
 * @tparam StaticTileDistribution_ Tile distribution policy.
 * @tparam NumCoord Number of coordinate dimensions.
 */
template <typename BottomTensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          index_t NumCoord>
struct is_tile_window_with_static_distribution<
    tile_window_with_static_distribution<BottomTensorView_,
                                         WindowLengths_,
                                         StaticTileDistribution_,
                                         NumCoord>> : std::true_type
{
};

/**
 * @brief Helper variable template to check if a type is a tile window with static distribution.
 *
 * Equivalent to `is_tile_window_with_static_distribution<T>::value`.
 *
 * @tparam T The type to check.
 */
template <typename T>
inline constexpr bool is_tile_window_with_static_distribution_v =
    is_tile_window_with_static_distribution<T>::value;

/**
 * @brief Type trait to determine if a type is a tile window with static lengths.
 *
 * Defaults to `false_type`. Specializations define when the trait evaluates to `true`.
 *
 * @tparam T The type to check.
 */
template <typename T>
struct is_tile_window_with_static_lengths : std::false_type
{
};

/**
 * @brief Specialization for `tile_window_with_static_lengths` to evaluate to `true_type`.
 *
 * @tparam BottomTensorView_ Bottom tensor view type of the tile window.
 * @tparam WindowLengths_ Static window lengths.
 */
template <typename BottomTensorView_, typename WindowLengths_>
struct is_tile_window_with_static_lengths<
    tile_window_with_static_lengths<BottomTensorView_, WindowLengths_>> : std::true_type
{
};

/**
 * @brief Helper variable template to check if a type is a tile window with static lengths.
 *
 * Equivalent to `is_tile_window_with_static_lengths<T>::value`.
 *
 * @tparam T The type to check.
 */
template <typename T>
inline constexpr bool is_tile_window_with_static_lengths_v =
    is_tile_window_with_static_lengths<T>::value;

} // namespace ck_tile
