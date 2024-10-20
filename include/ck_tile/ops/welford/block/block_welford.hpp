// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/welford/thread/thread_welford.hpp"

namespace ck_tile {

template <typename Problem_, typename Policy_ = void>
struct BlockWelford
{
    using Problem         = remove_cvref_t<Problem_>;
    using XDataType       = typename Problem::XDataType;
    using ComputeDataType = typename Problem::ComputeDataType;

    CK_TILE_DEVICE constexpr BlockWelford() {}

    // [CAUSION] - max_count_ is to deal with the padding problem
    // max_count_ is depend on caller, eg: naive and splitN welford will have different
    // calculation of max_count_
    // -> use block_welford_calculate_max_count to compute
    template <typename XDistributedTensor_,
              typename MeanDistributedTensor_,
              typename VarDistributedTensor_>
    CK_TILE_DEVICE void operator()(const XDistributedTensor_& x_tensor,
                                   MeanDistributedTensor_& mean_tensor,
                                   VarDistributedTensor_& var_tensor,
                                   int& cur_count_, // -> prefer init as zero
                                   const int& max_count_)
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

                    welford_update(
                        mean_tensor(out_dstr_idx), var_tensor(out_dstr_idx), x, cur_count_);
                });
            }
        });
    }

    template <typename XDistributedTensor_>
    CK_TILE_DEVICE static auto MakeMeanVarBlockTile()
    {
        static_assert(std::is_same_v<XDataType, typename XDistributedTensor_::DataType>, "wrong!");

        constexpr auto reduce_dims = sequence<1>{};

        constexpr auto dstr =
            make_static_tile_distribution(detail::make_reduce_tile_distribution_encoding(
                XDistributedTensor_::get_tile_distribution()
                    .get_static_tile_distribution_encoding(),
                reduce_dims));

        auto tensor = make_static_distributed_tensor<ComputeDataType>(dstr);

        return tensor;
    }

    template <typename XDistributedTensor_>
    CK_TILE_DEVICE auto
    operator()(const XDistributedTensor_& x_tensor, int& cur_count_, const int& max_count_)
    {
        auto mean_tensor = MakeMeanVarBlockTile<XDistributedTensor_>();
        auto var_tensor  = MakeMeanVarBlockTile<XDistributedTensor_>();
        clear_tile(mean_tensor);
        clear_tile(var_tensor);

        (*this)(x_tensor, mean_tensor, var_tensor, cur_count_, max_count_);

        return ck_tile::make_tuple(mean_tensor, var_tensor);
    }
};

template <typename Problem_, typename Policy_ = void>
struct BlockWelfordSync
{
    using Problem = remove_cvref_t<Problem_>;

    template <typename MeanDistributedTensor_, typename VarDistributedTensor_>
    CK_TILE_DEVICE void
    operator()(MeanDistributedTensor_& mean_tensor, VarDistributedTensor_& var_tensor, int& count)
    {
        using Dstr             = typename MeanDistributedTensor_::StaticTileDistribution;
        using DstrEncode       = typename Dstr::DstrEncode;
        using DstrEncodeDetail = typename DstrEncode::detail;

        static_assert(std::is_same_v<Dstr, typename VarDistributedTensor_::StaticTileDistribution>,
                      "wrong!");

        constexpr index_t NDimP = Dstr::get_num_of_dimension_p();
        constexpr index_t NDimR = Dstr::get_num_of_dimension_r();

        constexpr index_t idim_p_lane = NDimP - 1;

        // const auto ps_idx = make_array<index_t>(get_warp_id(), get_lane_id());
        // const auto rs_idx =
        //     mean_tensor.get_tile_distribution().calculate_rs_index_from_ps_index(ps_idx);

        constexpr index_t thread_buf_size = MeanDistributedTensor_::get_thread_buffer_size();
        static_assert(thread_buf_size == VarDistributedTensor_::get_thread_buffer_size());

        const int original_count = count;

        // loop over thread data
        static_for<0, thread_buf_size, 1>{}([&](auto i) {
            auto v_local_mean  = mean_tensor.get_thread_buffer()[i];
            auto v_local_var   = var_tensor.get_thread_buffer()[i];
            auto v_local_count = original_count;

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

                    static_assert(is_power_of_two_integer(r_length),
                                  "wrong! only support power of 2 reduction");

                    constexpr index_t nstage = integer_log2_floor(r_length);

                    // reduction sweep forward
                    static_for<0, nstage, 1>{}([&](auto istage) {
                        // xor
                        index_t src_lane =
                            (__lane_id()) ^
                            (number<lid_over_rid_derivative << istage.value>{}.value);

                        // pull data from remote lane
                        const auto v_remote_mean  = warp_shuffle(v_local_mean, src_lane);
                        const auto v_remote_var   = warp_shuffle(v_local_var, src_lane);
                        const auto v_remote_count = warp_shuffle(v_local_count, src_lane);

                        // welford merge
                        welford_merge(v_local_mean,
                                      v_local_var,
                                      v_local_count,
                                      v_remote_mean,
                                      v_remote_var,
                                      v_remote_count);
                    });
                }
            });

            mean_tensor.get_thread_buffer()(i) = v_local_mean;
            var_tensor.get_thread_buffer()(i)  = v_local_var;

            count = v_local_count;
        });
    }
};

template <typename Problem_, typename Policy_ = void>
struct BlockWelfordCrossWarpSync
{
    using Problem    = remove_cvref_t<Problem_>;
    using BlockShape = typename Problem::BlockShape;

    template <typename MeanDistributedTensor_>
    CK_TILE_DEVICE static constexpr index_t GetReduceWarps()
    {
        constexpr index_t num_reduce_warps = [&]() {
            using Dstr             = typename MeanDistributedTensor_::StaticTileDistribution;
            using DstrEncode       = typename Dstr::DstrEncode;
            using DstrEncodeDetail = typename DstrEncode::detail;

            constexpr index_t NDimR = Dstr::get_num_of_dimension_r();

            constexpr index_t idim_p_warp = 0;

            index_t len_ = 1;
            static_for<0, NDimR, 1>{}([&](auto idim_r) {
                if constexpr(DstrEncodeDetail::does_p_own_r_[idim_p_warp][idim_r])
                {
                    constexpr index_t r_length = DstrEncode::rs_lengths_[idim_r];
                    len_ *= r_length;
                }
            });
            return len_;
        }();
        return num_reduce_warps;
    }

    // return in byte
    template <typename MeanDistributedTensor_>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        // constexpr auto num_reduce_warps = GetReduceWarps<MeanDistributedTensor_>();

        // data need to exchange is very small, we just pack mean+var+count -> 4dword
        constexpr index_t thread_buf_size = MeanDistributedTensor_::get_thread_buffer_size();

        // we need to store all data from every wave into smem
        // e.g. 2x2 reduce along N
        //     -------------> reduce N
        //    | w0 | w1 |   ___>      | w01 |
        //    | w2 | w3 |             | w23 |
        //
        //   -> store data from every wave into LDS
        //
        //
        //     -------------> reduce N
        //    | w0 | w1 | w2 | w3 |   ----->  | w0123 |
        //
        //   -> also store data from every wave into LDS
        constexpr index_t num_warps = BlockShape::BlockSize / warpSize;
        return num_warps * 4 * thread_buf_size * sizeof(float);
    }

    template <typename MeanDistributedTensor_, typename VarDistributedTensor_>
    CK_TILE_DEVICE void operator()(MeanDistributedTensor_& mean_tensor,
                                   VarDistributedTensor_& var_tensor,
                                   int& count,
                                   void* smem)
    {
        using DataType = typename MeanDistributedTensor_::DataType;
        using Dstr     = typename MeanDistributedTensor_::StaticTileDistribution;
        // using DstrEncode       = typename Dstr::DstrEncode;
        // using DstrEncodeDetail = typename DstrEncode::detail;

        static_assert(std::is_same_v<Dstr, typename VarDistributedTensor_::StaticTileDistribution>,
                      "wrong!");

        constexpr index_t thread_buf_size = MeanDistributedTensor_::get_thread_buffer_size();
        static_assert(thread_buf_size == VarDistributedTensor_::get_thread_buffer_size());

        // Note: we always pack everything into fp32x4
        fp32x4_t* smem_ptr              = reinterpret_cast<fp32x4_t*>(smem);
        const index_t lane_id           = get_lane_id();
        const index_t warp_id           = get_warp_id();
        constexpr auto num_reduce_warps = GetReduceWarps<MeanDistributedTensor_>();
        constexpr index_t num_warps     = BlockShape::BlockSize / warpSize;
        const index_t smem_offset       = warp_id;

        // skip if nonthing to do
        if constexpr(num_reduce_warps == 1)
            return;

        // store into smem only for lane-0 within one warp
        if(lane_id == 0)
        {
            static_for<0, thread_buf_size, 1>{}([&](auto i) {
                fp32x4_t local_scratch_;
                local_scratch_[0] = bit_cast<float>(mean_tensor.get_thread_buffer()[i]);
                local_scratch_[1] = bit_cast<float>(var_tensor.get_thread_buffer()[i]);
                local_scratch_[2] = bit_cast<float>(count);

                smem_ptr[smem_offset + i * num_warps] = local_scratch_;
            });
        }
        block_sync_lds();

        // load from smem. here we let everythread to do compute :)
        index_t local_warp_id = warp_id / num_reduce_warps;
        index_t local_smem_os = local_warp_id * num_reduce_warps;
        fp32x4_t all_scratch[thread_buf_size * num_reduce_warps];
        static_for<0, thread_buf_size, 1>{}([&](auto i_0) {
            static_for<0, num_reduce_warps, 1>{}([&](auto i_1) {
                all_scratch[i_0 * num_warps + i_1] =
                    smem_ptr[i_0 * num_reduce_warps + local_smem_os + i_1];
            });
        });
        block_sync_lds(); // TODO: we don't need sync here

        // const int original_count = count;

        static_for<0, thread_buf_size, 1>{}([&](auto i_0) {
            // TODO: use descriptor for this
            auto v_local       = all_scratch[i_0 * num_warps];
            auto v_local_mean  = bit_cast<DataType>(v_local[0]);
            auto v_local_var   = bit_cast<DataType>(v_local[1]);
            auto v_local_count = bit_cast<int>(v_local[2]);

            // further reduce mean/var
            static_for<0, num_reduce_warps - 1, 1>{}([&](auto i_1_n1) {
                constexpr auto i_1        = number<i_1_n1 + 1>{};
                const fp32x4_t v_remote   = all_scratch[i_0 * num_warps + i_1];
                const auto v_remote_mean  = bit_cast<DataType>(v_remote[0]);
                const auto v_remote_var   = bit_cast<DataType>(v_remote[1]);
                const auto v_remote_count = bit_cast<int>(v_remote[2]);

                welford_merge(v_local_mean,
                              v_local_var,
                              v_local_count,
                              v_remote_mean,
                              v_remote_var,
                              v_remote_count);
            });

            mean_tensor.get_thread_buffer()(i_0) = v_local_mean;
            var_tensor.get_thread_buffer()(i_0)  = v_local_var;

            count = v_local_count;
        });
    }
};

// compute the max count for a last dim reduce
// everything may have vector/repeat, so the max count could be uneven
// TODO: specify which dim to compute and proper set the problem
// TODO: BlockShape we reuse layernorm_fwd_shape :)
template <typename BlockShape>
CK_TILE_DEVICE constexpr index_t block_tile_welford_calculate_max_count(int row_size)
{
    using S                   = BlockShape;
    index_t LastloopN         = row_size % S::Block_N == 0 ? S::Block_N : row_size % S::Block_N;
    constexpr index_t NThread = S::BlockWarps_N * S::Thread_N;
    index_t iNLane            = get_thread_id() % NThread;
    index_t iN0               = LastloopN / (S::Vector_N * S::Thread_N);
    index_t iN1               = (LastloopN % (S::Vector_N * S::Thread_N)) / S::Vector_N;
    index_t N2                = (LastloopN % (S::Vector_N * S::Thread_N)) % S::Vector_N;
    index_t iN3               = iNLane < iN1 ? S::Vector_N : iNLane == iN1 ? N2 : 0;
    return iN0 * S::Vector_N + iN3;
}

// Note: this function must be called after all the computation
template <typename VarDistributedTensor_>
CK_TILE_DEVICE constexpr void block_tile_welford_post_scale_var(VarDistributedTensor_& var_tensor,
                                                                int count)
{
    using DataType = typename VarDistributedTensor_::DataType;
    tile_elementwise_inout([&count](auto& x) { x = x / type_convert<DataType>(count); },
                           var_tensor);
}
} // namespace ck_tile
