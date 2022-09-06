// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <numeric>
#include <type_traits>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/device_elementwise.hpp"
#include "ck/utility/type.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

using F16 = ck::half_t;

struct ExecutionConfig final
{
    bool do_verification = true;
    bool time_kernel     = false;
};

struct Problem final
{
    std::array<std::size_t, 4> shape = {4, 16, 32, 32};
    std::array<std::size_t, 4> axes  = {0, 2, 3, 1};
};

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

namespace detail {

template <typename T, typename = void>
struct is_iterator : std::false_type
{
};

template <typename T>
struct is_iterator<T,
                   std::void_t<decltype(*std::declval<T>()),
                               decltype(++std::declval<std::add_lvalue_reference_t<T>>()),
                               decltype(std::declval<std::add_lvalue_reference_t<T>>()++)>>
    : std::true_type
{
};

template <typename T>
inline constexpr bool is_iterator_v = is_iterator<T>::value;

struct Placeholder final
{
    template <typename T>
    constexpr inline operator T() const noexcept;
};

template <typename Iterator, typename = void>
struct is_output_iterator : std::false_type
{
};

template <typename Iterator>
struct is_output_iterator<
    Iterator,
    std::void_t<decltype(*std::declval<Iterator>() = std::declval<Placeholder>())>>
    : std::bool_constant<is_iterator_v<Iterator>>
{
};

template <typename T>
inline constexpr bool is_output_iterator_v = is_output_iterator<T>::value;

template <typename Iterator, typename = void>
struct is_bidirectional_iterator : std::false_type
{
};

template <typename Iterator>
struct is_bidirectional_iterator<
    Iterator,
    std::void_t<decltype(--std::declval<std::add_lvalue_reference_t<Iterator>>()),
                decltype(std::declval<std::add_lvalue_reference_t<Iterator>>()--)>>
    : std::bool_constant<is_iterator_v<Iterator>>
{
};

template <typename Iterator>
inline constexpr bool is_bidirectional_iterator_v = is_bidirectional_iterator<Iterator>::value;

template <typename Iterator, typename = void>
struct is_random_access_iterator : std::false_type
{
};

template <typename Iterator>
struct is_random_access_iterator<Iterator,
                                 std::void_t<decltype(std::declval<Iterator>() + 1),
                                             decltype(std::declval<Iterator>() - 1),
                                             decltype(std::declval<Iterator>()[1])>>
    : std::bool_constant<is_iterator_v<Iterator>>
{
};

template <typename Iterator>
inline constexpr bool is_random_access_iterator_v = is_random_access_iterator<Iterator>::value;

template <typename T, typename = void>
struct is_range : std::false_type
{
};

template <typename T>
struct is_range<T,
                std::void_t<decltype(begin(std::declval<T>())), decltype(end(std::declval<T>()))>>
    : std::bool_constant<is_iterator_v<ck::remove_cvref_t<decltype(begin(std::declval<T>()))>>>
{
};

template <typename T>
inline constexpr bool is_range_v = is_range<T>::value;

template <typename Range, typename = void>
struct is_sized_range : std::false_type
{
};

template <typename Range>
struct is_sized_range<Range, std::void_t<decltype(size(std::declval<Range>()))>>
    : std::bool_constant<is_range_v<Range>>
{
};

template <typename Range>
inline constexpr bool is_sized_range_v = is_sized_range<Range>::value;

template <typename Range, typename = void>
struct is_bidirectional_range : std::false_type
{
};

template <typename Range>
struct is_bidirectional_range<Range, std::void_t<>>
    : std::bool_constant<
          is_range_v<Range> &&
          is_bidirectional_iterator_v<ck::remove_cvref_t<decltype(begin(std::declval<Range>()))>>>
{
};

template <typename Range>
inline constexpr bool is_bidirectional_range_v = is_bidirectional_range<Range>::value;

template <typename Range, typename = void>
struct is_random_access_range : std::false_type
{
};

template <typename Range>
struct is_random_access_range<Range, std::void_t<>>
    : std::bool_constant<
          is_range_v<Range> &&
          is_random_access_iterator_v<ck::remove_cvref_t<decltype(begin(std::declval<Range>()))>>>
{
};

template <typename Range>
inline constexpr bool is_random_access_range_v = is_random_access_range<Range>::value;

} // namespace detail

template <typename Axes>
inline std::enable_if_t<detail::is_random_access_range_v<Axes>, bool>
is_valid_axes(const Axes& axes)
{
    using std::empty;
    if(empty(axes))
    {
        return false;
    }

    using std::begin, std::end;
    std::vector<std::size_t> sorted_axes(begin(axes), end(axes));

    std::sort(begin(sorted_axes), end(sorted_axes));
    const auto last = std::unique(begin(sorted_axes), end(sorted_axes));

    return (last == end(sorted_axes)) && (*begin(sorted_axes) == 0) &&
           (*std::prev(last) == size(axes) - 1);
}

inline bool parse_cmd_args(int argc, char* argv[], ExecutionConfig& config, Problem& problem)
{
    constexpr int num_execution_config_args = 2;
    constexpr int num_problem_args          = 8;

    assert(num_problem_args == size(problem.shape) + size(problem.axes));

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 1 + num_execution_config_args)
    {
        config.do_verification = std::stoi(argv[1]);
        config.time_kernel     = std::stoi(argv[2]);
    }
    else if(argc == 1 + num_execution_config_args + num_problem_args)
    {
        config.do_verification = std::stoi(argv[1]);
        config.time_kernel     = std::stoi(argv[2]);

        // read shape
        for(std::size_t idx = 0; idx < size(problem.shape); ++idx)
        {
            problem.shape[idx] = std::stoi(argv[idx + 3]);
        }

        // read axes
        for(std::size_t idx = 0; idx < size(problem.axes); ++idx)
        {
            problem.axes[idx] = std::stoi(argv[idx + size(problem.shape) + 3]);
        }

        if(!is_valid_axes(problem.axes))
        {
            std::cerr << "invalid axes: ";
            std::copy(begin(problem.axes),
                      end(problem.axes),
                      std::ostream_iterator<std::size_t>(std::cerr, " "));
            std::cerr << std::endl;
            return false;
        }
    }
    else
    {
        std::cerr << "arg1: verification (0=no, 1=yes)" << std::endl
                  << "arg2: time kernel (0=no, 1=yes)" << std::endl
                  << "arg3 ~ arg6: shape for 4D tensor" << std::endl
                  << "arg7 ~ arg10: axes to permute" << std::endl;
        return false;
    }

    return true;
}

template <typename Shape>
inline std::enable_if_t<detail::is_range_v<Shape>, bool> is_valid_shape(const Shape& shape)
{
    using std::begin, std::end;
    using std::empty;
    return !empty(shape) && std::all_of(begin(shape), end(shape), [](auto dim) { return 0 < dim; });
}

template <typename Shape, typename Indices>
inline std::enable_if_t<detail::is_sized_range_v<Shape> && detail::is_sized_range_v<Indices>, bool>
is_valid_indices(const Shape& shape, const Indices& indices)
{
    assert(is_valid_shape(shape));

    using std::empty;
    if(empty(indices))
    {
        return false;
    }

    using std::size;
    if(size(shape) != size(indices))
    {
        return false;
    }

    using std::begin, std::end;

    auto dim = begin(shape);
    auto idx = begin(indices);
    for(; dim != end(shape) && idx != end(indices); ++dim, ++idx)
    {
        if(*dim <= *idx)
        {
            return false;
        }
    }

    return true;
}

template <typename Shape, typename Axes, typename OutputIterator>
inline std::enable_if_t<detail::is_random_access_range_v<Shape> &&
                            detail::is_sized_range_v<Shape> && detail::is_sized_range_v<Axes> &&
                            detail::is_output_iterator_v<OutputIterator>,
                        OutputIterator>
transpose_shape(const Shape& shape, const Axes& axes, OutputIterator iter)
{
    using std::size;
    assert(size(shape) == size(axes) &&);
    assert(is_valid_shape(shape) && is_valid_axes(axes));

    for(const auto axis : axes)
    {
        *iter++ = shape[axis];
    }

    return iter;
}

template <typename Shape, typename Indices>
std::enable_if_t<detail::is_bidirectional_range_v<Shape> && detail::is_sized_range_v<Shape> &&
                     detail::is_bidirectional_range_v<Indices> && detail::is_sized_range_v<Indices>,
                 bool>
advance_indices(const Shape& shape, Indices& indices)
{
    assert(is_valid_shape(shape));
    assert(is_valid_indices(indices));
    assert(size(shape) == size(indices));

    bool carry = true;

    using std::rbegin, std::rend;
    auto dim = rbegin(shape);
    auto idx = rbegin(indices);
    for(; carry && dim != rend(shape) && idx != rend(indices); ++dim, ++idx)
    {
        assert(*idx < *dim);

        *idx  = (*idx + carry);
        carry = ((*idx == *dim) ? (*idx = 0, true) : false);
    }

    return !carry;
}

template <typename Src, typename Functor, typename Dest>
std::enable_if_t<std::is_invocable_v<Functor,
                                     std::add_lvalue_reference_t<Dest>,
                                     std::add_lvalue_reference_t<Src>>>
host_elementwise_permute(const Tensor<Src>& src, Functor functor, Tensor<Dest>& dest)
{
    const auto& shape            = src.mDesc.GetLengths();
    const auto& transposed_shape = dest.mDesc.GetLengths();
    assert(is_valid_shape(shape) && is_valid_shape(transposed_shape));

    static_assert(detail::is_sized_range_v<ck::remove_cvref_t<decltype(shape)>> &&
                  detail::is_sized_range_v<ck::remove_cvref_t<decltype(transposed_shape)>>);

    using std::size;
    assert(size(shape) == 4 && size(transposed_shape) == 4);

    std::array<std::size_t, 4> dims{};
    do
    {
        Dest b_val = 0;
        functor(b_val, src(dims[0], dims[1], dims[2], dims[3]));
        dest(dims[0], dims[2], dims[3], dims[1]) = b_val;
    } while(advance_indices(shape, dims));
}
