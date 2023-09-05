// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

namespace ck {

template <index_t NDimHidden, typename BottomDimensionHiddenIds, typename TopDimensionHiddenIds>
struct TensorAdaptorCoordinate
{
    static constexpr index_t ndim_bottom_ = BottomDimensionHiddenIds::Size();
    static constexpr index_t ndim_top_    = TopDimensionHiddenIds::Size();

    using HiddenIndex = MultiIndex<NDimHidden>;
    using BottomIndex = MultiIndex<ndim_bottom_>;
    using TopIndex    = MultiIndex<ndim_top_>;

    public:
    __host__ __device__ constexpr TensorAdaptorCoordinate() = default;

    __host__ __device__ constexpr TensorAdaptorCoordinate(const HiddenIndex& idx_hidden)
        : idx_hidden_{idx_hidden}
    {
    }

    __host__ __device__ constexpr auto GetTopIndex() const
    {
        return get_container_subset(idx_hidden_, TopDimensionHiddenIds{});
    }

    __host__ __device__ constexpr auto GetBottomIndex() const
    {
        return get_container_subset(idx_hidden_, BottomDimensionHiddenIds{});
    }

    __host__ __device__ constexpr const auto& GetHiddenIndex() const { return idx_hidden_; }

    __host__ __device__ constexpr auto& GetHiddenIndex() { return idx_hidden_; }

    //
    HiddenIndex idx_hidden_;
};

template <typename Adaptor, typename TopIndex>
__host__ __device__ constexpr auto make_tensor_adaptor_coordinate(const Adaptor& adaptor,
                                                                  const TopIndex& idx_top)
{
    static_assert(Adaptor::GetNumOfTopDimension() == TopIndex::Size(),
                  "wrong! # of dimension inconsistent");

    constexpr index_t ntransform  = Adaptor::GetNumOfTransform();
    constexpr index_t ndim_hidden = Adaptor::GetNumOfHiddenDimension();
    constexpr auto bottom_dim_ids = Adaptor::GetBottomDimensionHiddenIds();
    constexpr auto top_dim_ids    = Adaptor::GetTopDimensionHiddenIds();

    MultiIndex<ndim_hidden> idx_hidden;

    // initialize visible index
    set_container_subset(idx_hidden, top_dim_ids, idx_top);

    // calculate hidden index
    static_for<ntransform, 0, -1>{}([&adaptor, &idx_hidden](auto itran_p1) {
        auto itran              = itran_p1 - Number<1>{};
        const auto& tran        = adaptor.GetTransforms().At(itran);
        constexpr auto dims_low = Adaptor::GetLowerDimensionHiddenIdss().At(itran);
        constexpr auto dims_up  = Adaptor::GetUpperDimensionHiddenIdss().At(itran);

        const auto idx_up = get_container_subset(idx_hidden, dims_up);

        MultiIndex<dims_low.Size()> idx_low;

        tran.CalculateLowerIndex(idx_low, idx_up);

        set_container_subset(idx_hidden, dims_low, idx_low);
    });

    return TensorAdaptorCoordinate<ndim_hidden,
                                   remove_cvref_t<decltype(bottom_dim_ids)>,
                                   remove_cvref_t<decltype(top_dim_ids)>>{idx_hidden};
}

template <bool JudgeDoTransforms = true,
          typename Adaptor,
          typename AdaptorCoord,
          typename TopIndex,
          typename BottomIndex>
__host__ __device__ constexpr void move_tensor_adaptor_coordinate(const Adaptor& adaptor,
                                                                  AdaptorCoord& coord,
                                                                  const TopIndex& idx_diff_top,
                                                                  BottomIndex& idx_diff_bottom)
{
    constexpr index_t ndim_hidden = Adaptor::GetNumOfHiddenDimension();
    constexpr index_t ndim_top    = Adaptor::GetNumOfTopDimension();
    //  constexpr index_t ndim_bottom = Adaptor::GetNumOfBottomDimension();
    constexpr index_t ntransform = Adaptor::GetNumOfTransform();

    //  STATIC_ASSERT(TopIndex::Size() == ndim_top && BottomIndex::Size() == ndim_bottom, "");

    // judge whether calculation of lower diff is needed for each transform
    // use index_t for boolean type
    auto do_transforms = make_zero_multi_index<ntransform>();

    if constexpr(JudgeDoTransforms)
    {
        auto is_non_zero_diff = make_zero_multi_index<ndim_hidden>();

        // decide do_transform by checkout non-zero index diff components
        MultiIndex<ndim_top> non_zero_diff_pick_top;

        static_for<0, ndim_top, 1>{}(
            [&](auto i) { non_zero_diff_pick_top(i) = (idx_diff_top[i] != 0); });

        set_container_subset(
            is_non_zero_diff, Adaptor::GetTopDimensionHiddenIds(), non_zero_diff_pick_top);

        static_for<ntransform - 1, -1, -1>{}([&](auto itran) {
            constexpr auto dims_low = Adaptor::GetLowerDimensionHiddenIdss().At(itran);
            constexpr auto dims_up  = Adaptor::GetUpperDimensionHiddenIdss().At(itran);

            const auto non_zero_diff_pick_up = get_container_subset(is_non_zero_diff, dims_up);

            MultiIndex<dims_low.Size()> non_zero_diff_pick_low;

            // if any of upper index diff components is non-zero, then
            //   1) Need to do this transform
            //   2) all components of lower index diff will assume to be non-zero and need to be
            //   computed
            const bool idx_diff_up_has_non_zero = container_reduce(
                non_zero_diff_pick_up, [](auto a, auto b) constexpr { return a or b; }, false);

            do_transforms(itran) = idx_diff_up_has_non_zero;

            static_for<0, dims_low.Size(), 1>{}(
                [&](auto i) { non_zero_diff_pick_low(i) = idx_diff_up_has_non_zero; });

            set_container_subset(is_non_zero_diff, dims_low, non_zero_diff_pick_low);
        });
    }
    else
    {
        static_for<ntransform - 1, -1, -1>{}([&](auto itran) { do_transforms(itran) = 1; });
    }

    // this is what needs to be calculated
    auto idx_diff_hidden = make_zero_multi_index<ndim_hidden>();

    // initialize top index diff
    set_container_subset(idx_diff_hidden, Adaptor::GetTopDimensionHiddenIds(), idx_diff_top);

    // this is what needs to be updated
    auto& idx_hidden = coord.GetHiddenIndex();

    // update top index
    auto idx_hidden_pick_top =
        get_container_subset(idx_hidden, Adaptor::GetTopDimensionHiddenIds());

    idx_hidden_pick_top += idx_diff_top;

    set_container_subset(idx_hidden, Adaptor::GetTopDimensionHiddenIds(), idx_hidden_pick_top);

    // update rest of hidden index
    static_for<ntransform - 1, -1, -1>{}([&](auto itran) {
        if(do_transforms[itran])
        {
            const auto& tran        = adaptor.GetTransforms().At(itran);
            constexpr auto dims_low = Adaptor::GetLowerDimensionHiddenIdss().At(itran);
            constexpr auto dims_up  = Adaptor::GetUpperDimensionHiddenIdss().At(itran);

            const auto idx_up_new  = get_container_subset(idx_hidden, dims_up);
            auto idx_low           = get_container_subset(idx_hidden, dims_low);
            const auto idx_diff_up = get_container_subset(idx_diff_hidden, dims_up);

            MultiIndex<dims_low.Size()> idx_diff_low;

            tran.UpdateLowerIndex(idx_diff_low, idx_diff_up, idx_low, idx_up_new);

            set_container_subset(idx_diff_hidden, dims_low, idx_diff_low);
            set_container_subset(idx_hidden, dims_low, idx_low);
        }
    });

    // set bottom index diff
    idx_diff_bottom = get_container_subset(idx_diff_hidden, Adaptor::GetBottomDimensionHiddenIds());
}

template <bool JudgeDoTransforms = true, typename Adaptor, typename AdaptorCoord, typename TopIndex>
__host__ __device__ constexpr void move_tensor_adaptor_coordinate(const Adaptor& adaptor,
                                                                  AdaptorCoord& coord,
                                                                  const TopIndex& idx_diff_top)
{
    constexpr index_t ndim_bottom = Adaptor::GetNumOfBottomDimension();

    MultiIndex<ndim_bottom> tmp;

    move_tensor_adaptor_coordinate<JudgeDoTransforms>(adaptor, coord, idx_diff_top, tmp);
}

template <typename Adaptor, typename AdaptorCoord>
__host__ __device__ constexpr bool
adaptor_coordinate_is_valid_assuming_top_index_is_valid(const Adaptor& adaptor,
                                                        const AdaptorCoord& coord)
{
    bool valid = true;

    constexpr index_t ntransform = Adaptor::GetNumOfTransform();

    const auto& idx_hidden = coord.GetHiddenIndex();

    static_for<ntransform - 1, -1, -1>{}([&adaptor, &idx_hidden, &valid](auto itran) {
        const auto tran = adaptor.GetTransforms().At(itran);

        // check validity, only if current transformation does not always has a valid mapping
        if constexpr(!decltype(tran)::IsValidUpperIndexAlwaysMappedToValidLowerIndex())
        {
            const auto idx_up =
                get_container_subset(idx_hidden, Adaptor::GetUpperDimensionHiddenIdss().At(itran));

            // Comment: using valid = valid && .. will result in weird control flow in ISA
            valid &= tran.IsValidUpperIndexMappedToValidLowerIndex(idx_up);
        }
    });

    return valid;
}

template <typename Adaptor, typename AdpatorCoord>
__host__ __device__ constexpr bool adaptor_coordinate_is_valid(const Adaptor& adaptor,
                                                               const AdpatorCoord& coord)
{
    // check top index
    const auto& idx_top = coord.GetTopIndex();

    bool is_top_index_valid = true;

    static_for<0, Adaptor::GetNumOfDimension(), 1>{}(
        [&is_top_index_valid, &idx_top, &adaptor](auto i) {
            is_top_index_valid =
                is_top_index_valid && (idx_top[i] >= 0 && idx_top[i] < adaptor.GetLength(i));
        });

    // check other hidden index
    return is_top_index_valid &&
           adaptor_coordinate_is_valid_assuming_top_index_is_valid(adaptor, coord);
}

} // namespace ck
