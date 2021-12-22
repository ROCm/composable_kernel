/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef CK_GRIDWISE_GENERIC_REDUCTION_WRAPPER_COMMON_HPP
#define CK_GRIDWISE_GENERIC_REDUCTION_WRAPPER_COMMON_HPP

#include "sequence.hpp"
#include "tensor_descriptor_helper.hpp"

namespace wrapper {

using namespace ck;

template <index_t... Ns>
__device__ static constexpr auto make_tuple_from_seq(Sequence<Ns...>)
{
    return make_tuple(Ns...);
};

template <index_t dims, typename invariantDims, typename toReduceDims>
struct get_ref_2d_desc_types
{
    static constexpr auto ref_toReduceDimLengths =
        typename uniform_sequence_gen<toReduceDims::Size(), 8>::type{};
    static constexpr auto ref_invariantDimLengths =
        typename uniform_sequence_gen<invariantDims::Size(), 8>::type{};

    static constexpr auto ref_lengths = typename uniform_sequence_gen<dims, 8>::type{};

    // don't have to use accurate strides to get an expected referrence type
    static constexpr auto ref_desc = make_naive_tensor_descriptor(make_tuple_from_seq(ref_lengths),
                                                                  make_tuple_from_seq(ref_lengths));

    static constexpr auto ref_2d_desc = transform_tensor_descriptor(
        ref_desc,
        make_tuple(ck::make_merge_transform(make_tuple_from_seq(ref_invariantDimLengths)),
                   ck::make_merge_transform(make_tuple_from_seq(ref_toReduceDimLengths))),
        make_tuple(invariantDims{}, toReduceDims{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}));

    using refType_2dDesc = decltype(ref_2d_desc);
};

template <typename invariantDims, typename toReduceDims>
struct get_ref_2d_desc_types<2, invariantDims, toReduceDims>
{
    static constexpr auto ref_toReduceDimLengths =
        typename uniform_sequence_gen<toReduceDims::Size(), 8>::type{};
    static constexpr auto ref_invariantDimLengths =
        typename uniform_sequence_gen<invariantDims::Size(), 8>::type{};

    static constexpr auto ref_lengths = typename uniform_sequence_gen<2, 8>::type{};

    // don't have to use accurate strides to get an expected referrence type
    static constexpr auto ref_desc = make_naive_tensor_descriptor(make_tuple_from_seq(ref_lengths),
                                                                  make_tuple_from_seq(ref_lengths));

    using refType_2dDesc = decltype(ref_desc);
};

template <index_t dims>
struct get_ref_1d_desc_types
{
    static constexpr auto ref_lengths = typename uniform_sequence_gen<dims, 8>::type{};

    static constexpr auto ref_desc = make_naive_tensor_descriptor(make_tuple_from_seq(ref_lengths),
                                                                  make_tuple_from_seq(ref_lengths));

    static constexpr auto ref_1d_desc = transform_tensor_descriptor(
        ref_desc,
        make_tuple(make_merge_transform(make_tuple_from_seq(ref_lengths))),
        make_tuple(typename arithmetic_sequence_gen<0, dims, 1>::type{}),
        make_tuple(Sequence<0>{}));

    using refType_1dDesc = decltype(ref_1d_desc);
};

template <>
struct get_ref_1d_desc_types<1>
{
    static constexpr auto ref_lengths = typename uniform_sequence_gen<1, 8>::type{};

    static constexpr auto ref_desc = make_naive_tensor_descriptor(make_tuple_from_seq(ref_lengths),
                                                                  make_tuple_from_seq(ref_lengths));

    using refType_1dDesc = decltype(ref_desc);
};

} // end of namespace wrapper

#endif
