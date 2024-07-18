// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_description/tensor_space_filling_curve.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/static_distributed_tensor.hpp"

namespace ck {
namespace tile_program {
namespace detail {

template <typename OutTensor, typename InTensor>
__device__ void shuffle_distributed_tensor_impl_in_thread(OutTensor& out_tensor,
                                                          const InTensor& in_tensor)
{
    constexpr auto I0 = Number<0>{};

    using DataType = typename InTensor::DataType;

    constexpr auto y_in_desc  = InTensor::GetTileDistribution().GetYs2DDescriptor();
    constexpr auto y_out_desc = OutTensor::GetTileDistribution().GetYs2DDescriptor();

    // y_dim_out_to_in
    constexpr auto get_rh_major_minor_to_y = [](auto dstr_tensor) {
        using DstrEncode = typename decltype(dstr_tensor.GetTileDistribution())::DstrEncode;

        Map<Array<index_t, 2>, index_t> rh_major_minor_to_y_;

        static_for<0, DstrEncode::NDimY, 1>{}([&](auto i) {
            constexpr index_t rh_major = DstrEncode::ys_to_rhs_major_[i];
            constexpr index_t rh_minor = DstrEncode::ys_to_rhs_minor_[i];

            rh_major_minor_to_y_({rh_major, rh_minor}) = i;
        });

        return rh_major_minor_to_y_;
    };

    constexpr auto rh_major_minor_to_y_in  = get_rh_major_minor_to_y(InTensor{});
    constexpr auto rh_major_minor_to_y_out = get_rh_major_minor_to_y(OutTensor{});

    constexpr auto y_dim_out_to_in = [&] {
        Map<index_t, index_t> y_dim_out_to_in_;

        for(const auto& [rh_major_minor, y_out] : rh_major_minor_to_y_out)
        {
            y_dim_out_to_in_(y_out) = rh_major_minor_to_y_in[rh_major_minor];
        }

        return y_dim_out_to_in_;
    }();

    //
    constexpr index_t NDimY = InTensor::GetTileDistribution().GetNumOfDimensionY();

    constexpr auto y_lengths = to_sequence(y_in_desc.GetLengths());

    // input and output vector dim in the order of input Y dims
    constexpr index_t y_dim_vec_in  = NDimY - 1;
    constexpr index_t y_dim_vec_out = y_dim_out_to_in[NDimY - 1];

    // vector lengths
    constexpr index_t vec_length_in  = y_lengths[y_dim_vec_in];
    constexpr index_t vec_length_out = y_lengths[y_dim_vec_out];

    // # of vectors
    constexpr index_t num_vec_in  = vec_length_out;
    constexpr index_t num_vec_out = vec_length_in;

    using InVec  = vector_type<DataType, vec_length_in>;
    using OutVec = vector_type<DataType, vec_length_out>;

    using InVecType  = typename InVec::type;
    using OutVecType = typename OutVec::type;

    // SFC
    constexpr auto scalars_per_access_arr = generate_array(
        [&](auto i) { return (i == y_dim_vec_in or i == y_dim_vec_out) ? y_lengths[i] : 1; },
        Number<NDimY>{});

    constexpr auto scalars_per_access = TO_SEQUENCE(scalars_per_access_arr, NDimY);

    using SFC_Y = SpaceFillingCurve<decltype(y_lengths),
                                    typename arithmetic_sequence_gen<0, NDimY, 1>::type,
                                    decltype(scalars_per_access)>;

    constexpr index_t num_access = SFC_Y::GetNumOfAccess();

    static_assert(num_access > 0, "wrong! num_access should be larger than 0");

    // in/out vectors to be transposed
    StaticallyIndexedArray<InVec, num_vec_in> in_vectors;
    StaticallyIndexedArray<OutVec, num_vec_out> out_vectors;

#if 0
    print(y_dim_out_to_in);
    printf("\n");
    printf("y_dim_vec_in %d\n", y_dim_vec_in);
    printf("y_dim_vec_out %d\n", y_dim_vec_out);
    printf("num_vec_in %d\n", num_vec_in);
    printf("num_vec_out %d\n", num_vec_out);
#endif

    // loop over SFC and do transpose
    static_for<0, num_access, 1>{}([&](auto iAccess) {
        // data index [y0, y1, ...] in the order of input tensor
        constexpr auto idx_y_start = SFC_Y::GetIndex(iAccess);

        // get input vectors
        static_for<0, num_vec_in, 1>{}([&](auto i) {
            constexpr auto idx_y_in = generate_array(
                [&](auto ii) {
                    return ii == y_dim_vec_out ? idx_y_start[ii] + i : idx_y_start[ii];
                },
                Number<NDimY>{});

            constexpr index_t in_offset = y_in_desc.CalculateOffset(idx_y_in);

            in_vectors(i).template AsType<InVecType>()(I0) =
                in_tensor.GetThreadBuffer().template GetAsType<InVecType>(Number<in_offset>{});
        });

        // transpose
        transpose_vectors<DataType, num_vec_in, num_vec_out>{}(in_vectors, out_vectors);

        // set output vectors
        static_for<0, num_vec_out, 1>{}([&](auto i) {
            constexpr auto idx_y_out_tmp = generate_array(
                [&](auto ii) { return ii == y_dim_vec_in ? idx_y_start[ii] + i : idx_y_start[ii]; },
                Number<NDimY>{});

            constexpr auto idx_y_out =
                container_reorder_given_new2old(idx_y_out_tmp, y_dim_out_to_in);

            constexpr index_t out_offset = y_out_desc.CalculateOffset(idx_y_out);

            out_tensor.GetThreadBuffer().template SetAsType<OutVecType>(
                Number<out_offset>{}, out_vectors[i].template AsType<OutVecType>()[I0]);
        });
    });
}

} // namespace detail
} // namespace tile_program
} // namespace ck
