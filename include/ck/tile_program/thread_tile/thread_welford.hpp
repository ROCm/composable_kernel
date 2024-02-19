// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/math_v2.hpp"

#include "ck/tile_program/tile/static_distributed_tensor.hpp"
#include "ck/tile_program/tile/static_tile_distribution_helper.hpp"
#include "ck/tile_program/tile/distributed_tile_sweep.hpp"

namespace ck {
namespace tile_program {
namespace thread {

template <typename ComputeDataType_, typename XDataType_>
struct ThreadWelford
{
    using XDataType       = remove_cvref_t<XDataType_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;

    template <typename T>
    __device__ inline void Update(T& mean, T& var, T x)
    {
        using ck::math::isnan;

        if(isnan(x))
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
    __device__ constexpr ThreadWelford(int max_count) : cur_count_(0), max_count_(max_count) {}

    template <typename MeanDistributedTensor_,
              typename VarDistributedTensor_,
              typename XDistributedTensor_>
    __device__ void operator()(MeanDistributedTensor_& mean_tensor,
                               VarDistributedTensor_& var_tensor,
                               const XDistributedTensor_& x_tensor)
    {
        using ck::math::isnan;

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        constexpr auto spans = XDistributedTensor_::GetDistributedSpans();

        sweep_tile_span(spans[I1], [&](auto dstr_idx_i1) {
            if(cur_count_ < max_count_)
            {
                ++cur_count_;

                sweep_tile_span(spans[I0], [&](auto dstr_idx_i0) {
                    constexpr auto in_dstr_idx  = make_tuple(dstr_idx_i0, dstr_idx_i1);
                    constexpr auto out_dstr_idx = make_tuple(dstr_idx_i0);

                    auto x = ck::type_convert<ComputeDataType>(x_tensor[in_dstr_idx]);

                    Update(mean_tensor(out_dstr_idx), var_tensor(out_dstr_idx), x);
                });
            }
        });
    }

    template <typename XDistributedTensor_>
    __device__ auto operator()(const XDistributedTensor_& x_tensor)
    {
        static_assert(is_same_v<XDataType, typename XDistributedTensor_::DataType>, "wrong!");

        constexpr auto reduce_dims = Sequence<1>{};

        constexpr auto mean_dstr = make_static_tile_distribution(
            ck::tile_program::detail::make_reduce_tile_distribution_encoding(
                XDistributedTensor_::GetTileDistribution().GetStaticTileDistributionEncoding(),
                reduce_dims));

        constexpr auto var_dstr = make_static_tile_distribution(
            ck::tile_program::detail::make_reduce_tile_distribution_encoding(
                XDistributedTensor_::GetTileDistribution().GetStaticTileDistributionEncoding(),
                reduce_dims));

        auto mean_tensor = make_static_distributed_tensor<ComputeDataType>(mean_dstr);
        auto var_tensor  = make_static_distributed_tensor<ComputeDataType>(var_dstr);

        // init mean & var tensor
        tile_elementwise_inout([&](auto& mean) { mean = type_convert<ComputeDataType>(0); },
                               mean_tensor);
        tile_elementwise_inout([&](auto& var) { var = type_convert<ComputeDataType>(0); },
                               var_tensor);

        (*this)(mean_tensor, var_tensor, x_tensor);

        return ck::make_tuple(mean_tensor, var_tensor);
    }

    int cur_count_;
    int max_count_;
};

} // namespace thread
} // namespace tile_program
} // namespace ck
