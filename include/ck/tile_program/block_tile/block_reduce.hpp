// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"

#include "ck/tile_program/tile/static_distributed_tensor.hpp"
#include "ck/tile_program/tile/static_tile_distribution_helper.hpp"
#include "ck/tile_program/tile/distributed_tile_sweep.hpp"

namespace ck {
namespace tile_program {
namespace block {

// synchronize reduce result (cross lane reduction and broadcast on replicated dimension)
template <typename AccDistributedTensor_, typename ReduceFunc>
__device__ void block_tile_reduce_sync(AccDistributedTensor_& acc_tensor,
                                       const ReduceFunc& reduce_func)
{
    using Dstr             = typename AccDistributedTensor_::StaticTileDistribution;
    using DstrEncode       = typename Dstr::DstrEncode;
    using DstrEncodeDetail = typename DstrEncode::Detail;

    constexpr index_t NDimP = Dstr::GetNumOfDimensionP();
    constexpr index_t NDimR = Dstr::GetNumOfDimensionR();

    constexpr index_t idim_p_lane = NDimP - 1;

    const auto ps_idx = make_array<index_t>(get_block_id(), get_lane_id());
    const auto rs_idx = acc_tensor.GetTileDistribution().CalculateRsIndexFromPsIndex(ps_idx);

    constexpr index_t thread_buf_size = AccDistributedTensor_::GetThreadBufferSize();

    // loop over thread data
    static_for<0, thread_buf_size, 1>{}([&](auto i) {
        auto v_local = acc_tensor.GetThreadBuffer()[i];

        // cross-lane reduce for replication
        // only reduce on R dimension correspond to lane
        // (lane id maps to this R dimension)
        static_for<0, NDimR, 1>{}([&](auto idim_r) {
            // FIXME: nasty to use does_p_own_r_
            if constexpr(DstrEncodeDetail::does_p_own_r_[idim_p_lane][idim_r])
            {
                constexpr index_t r_length = DstrEncode::rs_lengths_[idim_r];

                constexpr index_t lid_over_rid_derivative =
                    DstrEncodeDetail::ps_over_rs_derivative_[idim_p_lane][idim_r];

                static_assert(math::is_power_of_two_integer(r_length),
                              "wrong! only support power of 2 reduction");

                constexpr index_t nstage = math::integer_log2_floor(r_length);

                // reduction sweep forward
                static_for<0, nstage, 1>{}([&](auto istage) {
                    constexpr index_t lid_delta =
                        lid_over_rid_derivative * (1 << (nstage - istage - 1));

                    // pull data from remote lane
                    const auto v_remote = warp_shuffle_down(v_local, lid_delta);

                    // reduce
                    v_local = reduce_func(v_local, v_remote);
                });
            }
        });

        // cross-lane broadcast for replication
        // only broadcast on R dimension correspond to lane
        // (lane id maps to this R dimension)
        static_for<0, NDimR, 1>{}([&](auto idim_r) {
            // FIXME: nasty to use does_p_own_r_
            if constexpr(DstrEncodeDetail::does_p_own_r_[idim_p_lane][idim_r])
            {
                const index_t r_id = rs_idx[idim_r];

                constexpr index_t r_length = DstrEncode::rs_lengths_[idim_r];

                constexpr index_t lid_over_rid_derivative =
                    DstrEncodeDetail::ps_over_rs_derivative_[NDimP - 1][idim_r];

                static_assert(math::is_power_of_two_integer(r_length),
                              "wrong! only support power of 2 reduction");

                constexpr index_t nstage = math::integer_log2_floor(r_length);

                // broadcast sweep backward
                static_for<0, nstage, 1>{}([&](auto istage) {
                    // do I hold reduced data?
                    const bool do_i_hold_reduced_data = r_id < (1 << istage);

                    constexpr index_t lid_delta = lid_over_rid_derivative * (1 << istage);

                    // pull data from remote lane
                    const auto v_remote = warp_shuffle_up(v_local, lid_delta);

                    // decide whether to update local data with remote data
                    v_local = do_i_hold_reduced_data ? v_local : v_remote;
                });
            }
        });

        acc_tensor.GetThreadBuffer()(i) = v_local;
    });
}

// FIXME: this is for 2D to 1D reduce only, need to support n-D
template <typename AccDistributedTensor_,
          typename InDistributedTensor_,
          index_t... InReduceDims,
          typename ReduceFunc>
__device__ void block_tile_reduce(AccDistributedTensor_& acc_tensor,
                                  const InDistributedTensor_& in_tensor,
                                  Sequence<InReduceDims...>,
                                  const ReduceFunc& reduce_func)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};

#if 0
    constexpr auto in_reduce_dims = Sequence<InReduceDims...>{};

    constexpr index_t ndim_in        = InDistributedTensor_::GetNumOfDimension();
    constexpr index_t ndim_in_reduce = in_reduce_dims.Size();
    constexpr index_t ndim_in_free   = ndim_in - ndim_in_reduce;

    constexpr auto in_free_dims_arr = [&] {
        Array<bool, ndim_free> is_free_dims{true};

        for(index_t i = 0; i < ndim_reduce; i++)
        {
            is_free_dims(in_reduce_dims[i]) = false;
        }

        Array<index_t, ndim_free> in_free_dims{-1};

        index_t cnt = 0;

        for(index_t i = 0; i < ndim_in; i++)
        {
            if(is_free_dims[i])
            {
                in_free_dims(cnt) = i;

                cnt++
            }
        }

        return is_free_dims;
    }();

    constexpr auto in_free_dims = TO_SEQUENCE(is_free_dims_arr, ndim_in_free);
#else

    constexpr auto spans = InDistributedTensor_::GetDistributedSpans();

    // in-thread reduction
    // FIXME: hard coded to be 2D to 1D reduction
    sweep_tile_span(spans[I0], [&](auto dstr_idx_i0) {
        constexpr auto acc_dstr_idx = make_tuple(dstr_idx_i0);

        auto acc = acc_tensor[acc_dstr_idx];

        // FIXME
        sweep_tile_span(spans[I1], [&](auto dstr_idx_i1) {
            constexpr auto in_dstr_idx = make_tuple(dstr_idx_i0, dstr_idx_i1);

            const auto in = in_tensor[in_dstr_idx];

            acc = reduce_func(acc, in);
        });

        acc_tensor(acc_dstr_idx) = acc;
    });
#endif
}

template <typename AccDataType_,
          typename InDistributedTensor_,
          index_t... InReduceDims,
          typename ReduceFunc,
          typename InDataType_>
__device__ auto block_tile_reduce(const InDistributedTensor_& in_tensor,
                                  Sequence<InReduceDims...> in_reduce_dims,
                                  const ReduceFunc& reduce_func,
                                  const InDataType_& reduce_init)
{
    using InDataType  = typename InDistributedTensor_::DataType;
    using AccDataType = remove_cvref_t<AccDataType_>;

    static_assert(is_same_v<InDataType, remove_cvref_t<InDataType_>>, "wrong!");

    // declare acc_tensor
    constexpr auto acc_dstr = make_static_tile_distribution(
        ck::tile_program::detail::make_reduce_tile_distribution_encoding(
            InDistributedTensor_::GetTileDistribution().GetStaticTileDistributionEncoding(),
            Sequence<InReduceDims...>{}));

    auto acc_tensor = make_static_distributed_tensor<AccDataType>(acc_dstr);

    // init acc_tensor
    tile_elementwise_inout([&](auto& acc) { acc = type_convert<AccDataType>(reduce_init); },
                           acc_tensor);

    // warp reduce
    block_tile_reduce(acc_tensor, in_tensor, in_reduce_dims, reduce_func);

    return acc_tensor;
}

} // namespace block
} // namespace tile_program
} // namespace ck
