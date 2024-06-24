// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename ComputeDataType_, typename XDataType_>
struct ThreadWelford
{
    using XDataType       = remove_cvref_t<XDataType_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;

    template <typename T>
    CK_TILE_DEVICE void Update(T& mean, T& var, T x)
    {
        if(ck_tile::isnan(x))
        {
            mean = x;
            var  = x;
        }
        else
        {
            T delta = x - mean;
            mean += delta / cur_count_;
            T delta2 = x - mean;
            var += delta * delta2;
        }
    }

    // [CAUSION] - max_count_ is to deal with the padding problem
    // max_count_ is depend on caller, eg: naive and splitN welford will have different
    // calculation of max_count_
    CK_TILE_DEVICE constexpr ThreadWelford(int max_count) : cur_count_(0), max_count_(max_count) {}

    template <typename XDistributedTensor_,
              typename MeanDistributedTensor_,
              typename VarDistributedTensor_>
    CK_TILE_DEVICE void operator()(const XDistributedTensor_& x_tensor,
                                   MeanDistributedTensor_& mean_tensor,
                                   VarDistributedTensor_& var_tensor)
    {
        constexpr auto I0 = number<0>{};
        constexpr auto I1 = number<1>{};

        constexpr auto spans = XDistributedTensor_::get_distributed_spans();

        sweep_tile_span(spans[I1], [&](auto dstr_idx_i1) {
            if(cur_count_ < max_count_)
            {
                ++cur_count_;

                sweep_tile_span(spans[I0], [&](auto dstr_idx_i0) {
                    constexpr auto in_dstr_idx  = make_tuple(dstr_idx_i0, dstr_idx_i1);
                    constexpr auto out_dstr_idx = make_tuple(dstr_idx_i0);

                    auto x = ck_tile::type_convert<ComputeDataType>(x_tensor[in_dstr_idx]);

                    Update(mean_tensor(out_dstr_idx), var_tensor(out_dstr_idx), x);
                });
            }
        });
    }

    template <typename XDistributedTensor_>
    CK_TILE_DEVICE static auto MakeInitialMeanVarDistributedTensor()
    {
        static_assert(std::is_same_v<XDataType, typename XDistributedTensor_::DataType>, "wrong!");

        constexpr auto reduce_dims = sequence<1>{};

        constexpr auto dstr =
            make_static_tile_distribution(detail::make_reduce_tile_distribution_encoding(
                XDistributedTensor_::get_tile_distribution()
                    .get_static_tile_distribution_encoding(),
                reduce_dims));

        auto tensor = make_static_distributed_tensor<ComputeDataType>(dstr);
        clear_tile(tensor);

        return tensor;
    }

    template <typename XDistributedTensor_>
    CK_TILE_DEVICE auto operator()(const XDistributedTensor_& x_tensor)
    {
        auto mean_tensor = MakeInitialMeanVarDistributedTensor<XDistributedTensor_>();
        auto var_tensor  = MakeInitialMeanVarDistributedTensor<XDistributedTensor_>();

        (*this)(x_tensor, mean_tensor, var_tensor);

        return ck_tile::make_tuple(mean_tensor, var_tensor);
    }

    int cur_count_;
    int max_count_;
};

} // namespace ck_tile
