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

inline bool parse_cmd_args(int argc, char* argv[], ExecutionConfig& config, Problem& problem)
{
    constexpr int num_execution_config_args = 2;
    constexpr int num_problem_args          = 8;

    assert(num_problem_args == problem.shape.size() + problem.axes.size());

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
        for(std::size_t idx = 0; idx < problem.shape.size(); ++idx)
        {
            problem.shape[idx] = std::stoi(argv[idx + 3]);
        }

        // read axes
        for(std::size_t idx = 0; idx < problem.axes.size(); ++idx)
        {
            problem.axes[idx] = std::stoi(argv[idx + problem.shape.size() + 3]);
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
