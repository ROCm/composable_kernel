// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <numeric>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/device_elementwise.hpp"

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

template <typename RandomAccessRange>
inline bool is_valid_axes(RandomAccessRange axes)
{
    using std::empty;
    if(empty(axes))
    {
        return false;
    }

    using std::begin, std::end;
    std::vector<std::size_t> copy(begin(axes), end(axes));

    std::sort(begin(copy), end(copy));
    const auto last = std::unique(begin(copy), end(copy));

    return (last == end(copy)) && (*begin(copy) == 0) && (*std::prev(last) == size(axes) - 1);
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
