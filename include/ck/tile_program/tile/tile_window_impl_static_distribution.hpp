// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_description/tensor_adaptor_coordinate.hpp"
#include "ck/tensor_description/tensor_space_filling_curve.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/static_tile_distribution_helper.hpp"
#include "ck/tile_program/tile/static_distributed_tensor.hpp"

namespace ck {
namespace tile_program {

template <typename BottomTensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          index_t NumCoord>
struct TileWindowWithStaticDistribution
{
    using BottomTensorView = remove_reference_t<BottomTensorView_>;
    using WindowLengths    = remove_cvref_t<WindowLengths_>;
    using TileDstr         = remove_cvref_t<StaticTileDistribution_>;

    using WindowAdaptor    = typename TileDstr::PsYs2XsAdaptor;
    using BottomTensorDesc = typename BottomTensorView::TensorDesc;

    using DataType = remove_cvref_t<typename BottomTensorView::DataType>;

    static constexpr index_t NDimWindowAdaptorTop = WindowAdaptor::GetNumOfTopDimension();
    static constexpr index_t NDimBottomTensor     = BottomTensorDesc::GetNumOfDimension();

    static constexpr index_t NDimP = TileDstr::GetNumOfDimensionP();
    static constexpr index_t NDimY = TileDstr::GetNumOfDimensionY();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    // TODO: check WindowLengths and StaticTileDistribution are consistent

    static_assert(is_known_at_compile_time<WindowLengths>::value,
                  "wrong! lengths should be static");
    static_assert(TileDstr::IsStatic(), "wrong!");

    static_assert(NDimBottomTensor == WindowAdaptor::GetNumOfBottomDimension(),
                  "wrong! inconsistent # of diemsnions");

    using AdaptorTopIndex   = Array<index_t, NDimWindowAdaptorTop>;
    using BottomTensorIndex = Array<index_t, NDimBottomTensor>;

    using WindowAdaptorCoord =
        decltype(make_tensor_adaptor_coordinate(WindowAdaptor{}, AdaptorTopIndex{}));

    using BottomTensorCoord =
        decltype(make_tensor_coordinate(BottomTensorDesc{}, BottomTensorIndex{}));

    struct LoadStoreTraits
    {
        private:
        static constexpr auto GetVectorDimYScalarPerVector()
        {
            const auto [ys_vector_lengths, ys_vector_strides] =
                TileWindowWithStaticDistribution::GetWindowAdaptorYsSafeVectorLengthStrides();

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
        static constexpr index_t VectorDimY      = GetVectorDimYScalarPerVector().template At<0>();
        static constexpr index_t ScalarPerVector = GetVectorDimYScalarPerVector().template At<1>();

        using vector_type_t = vector_type_maker_t<DataType, ScalarPerVector>;
        using vector_t      = typename vector_type_t::type;

        private:
        static constexpr auto scalars_per_access_ = [] {
            constexpr auto scalars_per_access_arr = generate_array(
                [&](auto i) { return (i == VectorDimY) ? ScalarPerVector : 1; }, Number<NDimY>{});

            /// TODO: add non-automatic storage argument support to macro TO_SEQUENCE()
            constexpr auto NDimY_ = NDimY;

            return TO_SEQUENCE(scalars_per_access_arr, NDimY_);
        }();

        static constexpr auto GetSpaceFillingCurve()
        {
            constexpr auto tile_dstr = TileDstr{};

            constexpr auto thread_tensor_lengths_ys =
                to_sequence(tile_dstr.GetYs2DDescriptor().GetLengths());

            // FIXME: need logic to judge dim access order
            using DimAccessOrder = typename arithmetic_sequence_gen<0, NDimY, 1>::type;

            return SpaceFillingCurve<decltype(thread_tensor_lengths_ys),
                                     DimAccessOrder,
                                     decltype(scalars_per_access_)>{};
        }

        public:
        using SFC_Ys = decltype(GetSpaceFillingCurve());

        static constexpr index_t NumAccess = SFC_Ys::GetNumOfAccess();

        static_assert(0 < NumAccess, "Wrong! NumAccess should be larger than 0");
        static_assert(NumAccess % NumCoord == 0, "wrong! # of access is not divisible by NumCoord");
    };

    static constexpr index_t NumAccessPerCoord = LoadStoreTraits::NumAccess / NumCoord;

    __device__ constexpr TileWindowWithStaticDistribution() = default;

    __device__ constexpr TileWindowWithStaticDistribution(
        const BottomTensorView& bottom_tensor_view,
        const WindowLengths& window_lengths,
        const BottomTensorIndex& window_origin,
        const TileDstr& tile_distribution)
        : bottom_tensor_view_{bottom_tensor_view},
          window_lengths_{window_lengths},
          window_origin_{window_origin},
          tile_dstr_{tile_distribution},
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
                tile_distribution.GetPsYs2XsAdaptor(), AdaptorTopIndex{get_lane_id(), 0});
        }
        else if constexpr(NDimP == 2)
        {
            window_adaptor_thread_coord_tmp =
                make_tensor_adaptor_coordinate(tile_distribution.GetPsYs2XsAdaptor(),
                                               AdaptorTopIndex{get_warp_id(), get_lane_id(), 0});
        }
#else
        // TODO: this use less register for FA, but more register for GEMM
        // need investigation
        const auto window_adaptor_thread_coord_tmp = make_tensor_adaptor_coordinate(
            tile_distribution.GetPsYs2XsAdaptor(),
            container_concat(detail::get_partition_index(tile_distribution),
                             Array<index_t, NDimY>{0}));
#endif

        BottomTensorIndex bottom_tensor_thread_origin_idx_tmp =
            window_origin + window_adaptor_thread_coord_tmp.GetBottomIndex();

        const auto bottom_tensor_thread_coord_tmp = make_tensor_coordinate(
            bottom_tensor_view_.GetTensorDescriptor(), bottom_tensor_thread_origin_idx_tmp);

        // pre-compute NumCoord (WindowAdaptorCoord, BottomTensorCoord) bundles to speed up
        // future Load/Store() calls (might allocate more registers)
        using Traits = LoadStoreTraits;
        using SFC_Ys = typename Traits::SFC_Ys;

        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            auto window_adaptor_thread_coord = window_adaptor_thread_coord_tmp;
            auto bottom_tensor_thread_coord  = bottom_tensor_thread_coord_tmp;

            constexpr auto idx_diff_ys =
                SFC_Ys::GetStepBetween(Number<0>{}, Number<iCoord * NumAccessPerCoord>{});

            constexpr auto idx_diff_ps_ys = container_concat(Array<index_t, NDimP>{0}, idx_diff_ys);

            MoveWindowAdaptorAndBottomTensorThreadCoordinate(
                window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);

            pre_computed_coords_(iCoord) =
                make_tuple(window_adaptor_thread_coord, bottom_tensor_thread_coord);
        });
    }

    __device__ static constexpr index_t GetNumOfDimension() { return NDimBottomTensor; }

    __device__ static constexpr bool HasStaticTileDistribution() { return TileDstr::IsStatic(); }

    __device__ constexpr auto GetWindowLengths() const { return window_lengths_; }

    __device__ constexpr auto GetTileDistribution() const { return tile_dstr_; }

    __device__ constexpr auto GetBottomTensorView() const { return bottom_tensor_view_; }

    __device__ constexpr auto GetWindowOrigin() const { return window_origin_; }

    // move thread's window adaptor coordinate and bottom tensor coordinate
    // [p0, p1, ..., y0, y1, ...] ==> [x0, x1, ...] ==> [x0', x1', ...] ==> [offset]
    __device__ void MoveWindowAdaptorAndBottomTensorThreadCoordinate(
        WindowAdaptorCoord& window_adaptor_thread_coord,
        BottomTensorCoord& bottom_tensor_thread_coord,
        const AdaptorTopIndex& idx_diff_adaptor_top) const
    {
        Array<index_t, NDimBottomTensor> idx_diff_adaptor_bottom;

        move_tensor_adaptor_coordinate(tile_dstr_.GetPsYs2XsAdaptor(),
                                       window_adaptor_thread_coord,
                                       idx_diff_adaptor_top,
                                       idx_diff_adaptor_bottom);

        move_tensor_coordinate(bottom_tensor_view_.GetTensorDescriptor(),
                               bottom_tensor_thread_coord,
                               idx_diff_adaptor_bottom);
    }

    // return vector dimension among [y0, y1, ...]
    __device__ static constexpr auto GetWindowAdaptorYsSafeVectorLengthStrides()
    {
        // bottom tensor top dimension vector lengths and strides
        const auto [bottom_tensor_top_dim_vector_lengths, bottom_tensor_top_dim_vector_strides] =
            BottomTensorDesc::GetTopDimensionSafeVectorLengthStrides();

        // window vector lengths/strides
        const auto window_adaptor_bottom_dim_vector_lengths = bottom_tensor_top_dim_vector_lengths;
        const auto window_adaptor_bottom_dim_vector_strides = bottom_tensor_top_dim_vector_strides;

        // window adaptor [p0, p1, ..., y0, y1, ...]
        Array<index_t, WindowAdaptor::GetNumOfHiddenDimension()> window_adaptor_vector_lengths{-1};
        Array<index_t, WindowAdaptor::GetNumOfHiddenDimension()> window_adaptor_vector_strides{-1};

        constexpr auto window_adaptor_bottom_dims = WindowAdaptor::GetBottomDimensionHiddenIds();

        set_container_subset(window_adaptor_vector_lengths,
                             window_adaptor_bottom_dims,
                             window_adaptor_bottom_dim_vector_lengths);
        set_container_subset(window_adaptor_vector_strides,
                             window_adaptor_bottom_dims,
                             window_adaptor_bottom_dim_vector_strides);

        const auto [window_adaptor_ps_ys_vector_lengths, window_adaptor_ps_ys_vector_strides] =
            WindowAdaptor{}.GetTopDimensionSafeVectorLengthStrides(window_adaptor_vector_lengths,
                                                                   window_adaptor_vector_strides);

        // [y0, y1, ...]
        constexpr auto y_dims = typename arithmetic_sequence_gen<TileDstr::GetNumOfDimensionP(),
                                                                 NDimWindowAdaptorTop,
                                                                 1>::type{};

        return make_tuple(get_container_subset(window_adaptor_ps_ys_vector_lengths, y_dims),
                          get_container_subset(window_adaptor_ps_ys_vector_strides, y_dims));
    }

    __device__ constexpr auto GetNumAccess() const { return LoadStoreTraits::NumAccess; }

    template <bool use_inline_asm = false>
    __device__ auto Load(bool_constant<use_inline_asm> = {}) const
    {
        using Traits = LoadStoreTraits;

        using vector_type_t = typename Traits::vector_type_t;
        using vector_t      = typename vector_type_t::type;
        using SFC_Ys        = typename Traits::SFC_Ys;

        constexpr auto tile_dstr = TileDstr{};

        auto dst_tensor = make_static_distributed_tensor<DataType>(tile_dstr);

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = Number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::GetIndex(iAccess);

                // read from bottom tensor
                const vector_t vec_value =
                    GetBottomTensorView().template GetVectorizedElements<vector_t>(
                        bottom_tensor_thread_coord, bool_constant<use_inline_asm>{});

                const vector_type_t vec{vec_value};

                // write into distributed tensor
                static_for<0, Traits::ScalarPerVector, 1>{}([&](auto j) {
                    constexpr auto idx_ys = generate_array(
                        [&](auto jj) {
                            return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                            : idx_ys_start[jj];
                        },
                        Number<NDimY>{});

                    constexpr index_t d = tile_dstr.GetYs2DDescriptor().CalculateOffset(idx_ys);

                    dst_tensor.GetThreadBuffer().template At<d>() =
                        vec.template AsType<DataType>()[j];
                });

                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::GetForwardStep(iAccess);

                    constexpr auto idx_diff_ps_ys =
                        container_concat(Array<index_t, NDimP>{0}, idx_diff_ys);

                    MoveWindowAdaptorAndBottomTensorThreadCoordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });

        return dst_tensor;
    }

    __device__ auto MakeLoadTile()
    {
        constexpr auto tile_dstr = TileDstr{};
        return make_static_distributed_tensor<DataType>(tile_dstr);
    }

    template <typename DstTile>
    __device__ void LoadRaw(DstTile& dst_tensor) const
    {
        using Traits = LoadStoreTraits;

        using vector_type_t = typename Traits::vector_type_t;
        using vector_t      = typename vector_type_t::type;
        using SFC_Ys        = typename Traits::SFC_Ys;
        static constexpr index_t YElementSize =
            TileDstr{}.GetYs2DDescriptor().GetElementSpaceSize();
        static_assert(YElementSize % Traits::ScalarPerVector == 0);
        using vectorized_tbuf = StaticBuffer<AddressSpaceEnum::Vgpr,
                                             vector_t,
                                             YElementSize / Traits::ScalarPerVector,
                                             true>;

        constexpr auto tile_dstr          = TileDstr{};
        constexpr bool use_buffer_load_if = true;

        // auto dst_tensor = make_static_distributed_tensor<DataType>(tile_dstr);
        auto& dst_vec_tbuf = reinterpret_cast<vectorized_tbuf&>(dst_tensor.GetThreadBuffer());

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = Number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::GetIndex(iAccess);
                constexpr index_t d = tile_dstr.GetYs2DDescriptor().CalculateOffset(idx_ys_start);
                static_assert(d % Traits::ScalarPerVector == 0);

                GetBottomTensorView().template GetVectorizedElementsRaw<vector_t>(
                    dst_vec_tbuf.template At<d / Traits::ScalarPerVector>(),
                    bottom_tensor_thread_coord,
                    bool_constant<use_buffer_load_if>{});
#if 0
                // read from bottom tensor
                const vector_t vec_value =
                    GetBottomTensorView().template GetVectorizedElements<vector_t>(
                        bottom_tensor_thread_coord, bool_constant<use_inline_asm>{});

                const vector_type_t vec{vec_value};

                // write into distributed tensor
                static_for<0, Traits::ScalarPerVector, 1>{}([&](auto j) {
                    constexpr auto idx_ys = generate_array(
                        [&](auto jj) {
                            return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                            : idx_ys_start[jj];
                        },
                        Number<NDimY>{});

                    constexpr index_t d = tile_dstr.GetYs2DDescriptor().CalculateOffset(idx_ys);

                    dst_tensor.GetThreadBuffer().template At<d>() =
                        vec.template AsType<DataType>()[j];
                });
#endif

                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::GetForwardStep(iAccess);

                    constexpr auto idx_diff_ps_ys =
                        container_concat(Array<index_t, NDimP>{0}, idx_diff_ys);

                    MoveWindowAdaptorAndBottomTensorThreadCoordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });

        // return dst_tensor;
    }

    // TODO: currently async load only implemented in inline asm
    template <typename LdsTileWindow_, bool use_inline_asm = true>
    __device__ auto AsyncLoad(LdsTileWindow_&& lds_tile, bool_constant<use_inline_asm> = {}) const
    {
        using LdsTileWindow = remove_cvref_t<LdsTileWindow_>;
        // using LdsTensorView = typename LdsTileWindow::BottomTensorView;
        using LdsDataType = typename LdsTileWindow::DataType;
        // using LdsDescriptor = typename LdsTileWindow::BottomTensorDesc;

        // issues * warps * lanes
        static_assert(LdsTileWindow::GetNumOfDimension() == 3); // TODO: hard coded

        const index_t size_per_buf =
            lds_tile.GetBottomTensorView().GetTensorDescriptor().CalculateOffset(
                make_tuple(Number<0>{}, Number<0>{}, Number<0>{})) *
            sizeof(LdsDataType);

        const index_t size_per_wave =
            lds_tile.GetBottomTensorView().GetTensorDescriptor().CalculateOffset(
                make_tuple(Number<0>{}, Number<1>{}, Number<0>{})) *
                sizeof(LdsDataType) -
            size_per_buf;

        const index_t size_per_issue =
            lds_tile.GetBottomTensorView().GetTensorDescriptor().CalculateOffset(
                make_tuple(Number<1>{}, Number<0>{}, Number<0>{})) *
                sizeof(LdsDataType) -
            size_per_buf;

        const index_t m0_init_value = size_per_buf + size_per_wave * get_warp_id();
        m0_set_with_memory(m0_init_value); // This should be wave independent

        using Traits = LoadStoreTraits;

        using vector_type_t = typename Traits::vector_type_t;
        using vector_t      = typename vector_type_t::type;
        using SFC_Ys        = typename Traits::SFC_Ys;

        LdsDataType* smem = lds_tile.GetBottomTensorView().GetBufferView().p_data_;

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            // TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = Number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // read from bottom tensor
                GetBottomTensorView().template AsyncGetVectorizedElements<vector_t>(
                    smem, bottom_tensor_thread_coord);

                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::GetForwardStep(iAccess);

                    constexpr auto idx_diff_ps_ys =
                        container_concat(Array<index_t, NDimP>{0}, idx_diff_ys);

                    MoveWindowAdaptorAndBottomTensorThreadCoordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);

                    m0_inc_with_memory(size_per_issue);
                }
            });
        });
    }

    __device__ void Store(const StaticDistributedTensor<DataType, TileDstr>& dstr_tensor) const
    {
        using Traits = LoadStoreTraits;

        using vector_type_t = typename Traits::vector_type_t;
        using vector_t      = typename vector_type_t::type;
        using SFC_Ys        = typename Traits::SFC_Ys;

        constexpr auto tile_dstr = TileDstr{};

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = Number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::GetIndex(iAccess);

                // read from distributed tensor
                vector_type_t vec;

                static_for<0, Traits::ScalarPerVector, 1>{}([&](auto j) {
                    constexpr auto idx_ys = generate_array(
                        [&](auto jj) {
                            return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                            : idx_ys_start[jj];
                        },
                        Number<NDimY>{});

                    constexpr index_t d = tile_dstr.GetYs2DDescriptor().CalculateOffset(idx_ys);

                    vec.template AsType<DataType>()(j) =
                        dstr_tensor.GetThreadBuffer().template At<d>();
                });

                const vector_t vec_value = vec.template AsType<vector_t>().template At<0>();

                // write into bottom tensor
                GetBottomTensorView().template SetVectorizedElements<vector_t>(
                    bottom_tensor_thread_coord, vec_value);

                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::GetForwardStep(iAccess);

                    constexpr auto idx_diff_ps_ys =
                        container_concat(Array<index_t, NDimP>{0}, idx_diff_ys);

                    MoveWindowAdaptorAndBottomTensorThreadCoordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    __device__ void StoreRaw(const StaticDistributedTensor<DataType, TileDstr>& dstr_tensor) const
    {
        using Traits = LoadStoreTraits;

        using vector_type_t = typename Traits::vector_type_t;
        using vector_t      = typename vector_type_t::type;
        using SFC_Ys        = typename Traits::SFC_Ys;

        constexpr auto tile_dstr                  = TileDstr{};
        static constexpr bool use_buffer_store_if = true;

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord][I0];
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord][I1];

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = Number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::GetIndex(iAccess);

                // read from distributed tensor
                vector_type_t vec;

                static_for<0, Traits::ScalarPerVector, 1>{}([&](auto j) {
                    constexpr auto idx_ys = generate_array(
                        [&](auto jj) {
                            return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                            : idx_ys_start[jj];
                        },
                        Number<NDimY>{});

                    constexpr index_t d = tile_dstr.GetYs2DDescriptor().CalculateOffset(idx_ys);

                    vec.template AsType<DataType>()(j) =
                        dstr_tensor.GetThreadBuffer().template At<d>();
                });

                const vector_t vec_value = vec.template AsType<vector_t>().template At<0>();

                // write into bottom tensor
                GetBottomTensorView()
                    .template SetVectorizedElementsRaw<vector_t, use_buffer_store_if>(
                        bottom_tensor_thread_coord, vec_value);

                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::GetForwardStep(iAccess);

                    constexpr auto idx_diff_ps_ys =
                        container_concat(Array<index_t, NDimP>{0}, idx_diff_ys);

                    MoveWindowAdaptorAndBottomTensorThreadCoordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    // move thread's botom tensor coordiante
    // [x0', x1', ... ] ==> [offset]
    // also move window-origin
    __device__ void Move(const BottomTensorIndex& step)
    {
        window_origin_ += step;

        static_for<0, NumCoord, 1>{}([&](auto iCoord) {
            move_tensor_coordinate(
                bottom_tensor_view_.GetTensorDescriptor(), pre_computed_coords_(iCoord)(I1), step);
        });
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

    // this contains:
    //   per-thread coordinate for window adaptor
    //   per-thread coordinate for bottom tensor
    Array<Tuple<WindowAdaptorCoord, BottomTensorCoord>, NumCoord> pre_computed_coords_;
};

// TODO: use strategy
template <typename TensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          index_t NumCoord = 1>
__device__ constexpr auto
make_tile_window(const TensorView_& tensor_view,
                 const WindowLengths_& window_lengths,
                 const MultiIndex<TensorView_::GetNumOfDimension()>& origin,
                 const StaticTileDistribution_& tile_distribution,
                 Number<NumCoord> = {})
{
    return TileWindowWithStaticDistribution<remove_cvref_t<TensorView_>,
                                            remove_cvref_t<WindowLengths_>,
                                            remove_cvref_t<StaticTileDistribution_>,
                                            NumCoord>{
        tensor_view, window_lengths, origin, tile_distribution};
}

template <typename TensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          index_t NumCoord>
__device__ void
move_tile_window(TileWindowWithStaticDistribution<TensorView_,
                                                  WindowLengths_,
                                                  StaticTileDistribution_,
                                                  NumCoord>& window,
                 const typename TileWindowWithStaticDistribution<TensorView_,
                                                                 WindowLengths_,
                                                                 StaticTileDistribution_,
                                                                 NumCoord>::BottomTensorIndex& step)
{
    window.Move(step);
}

} // namespace tile_program
} // namespace ck
