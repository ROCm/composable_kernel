// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_fwd_multiple_d_multiple_r_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"

using FP16 = ck::half_t;
using FP32 = float;

template <typename Ret>
struct type_function
{
    using type = Ret;
};

template <ck::index_t>
struct ALayoutSelector;

template <ck::index_t>
struct BLayoutSelector;

template <ck::index_t>
struct DELayoutSelector;

template <ck::index_t>
struct RLayoutSelector;

namespace ctl = ck::tensor_layout::convolution;

template <>
struct ALayoutSelector<1> : type_function<ctl::GNWC>
{
};

template <>
struct BLayoutSelector<1> : type_function<ctl::GKXC>
{
};

template <>
struct DELayoutSelector<1> : type_function<ctl::GNWK>
{
};

template <>
struct RLayoutSelector<1> : type_function<ctl::GNW>
{
};

template <>
struct ALayoutSelector<2> : type_function<ctl::GNHWC>
{
};

template <>
struct BLayoutSelector<2> : type_function<ctl::GKYXC>
{
};

template <>
struct DELayoutSelector<2> : type_function<ctl::GNHWK>
{
};

template <>
struct RLayoutSelector<2> : type_function<ctl::GNHW>
{
};

template <>
struct ALayoutSelector<3> : type_function<ctl::GNDHWC>
{
};

template <>
struct BLayoutSelector<3> : type_function<ctl::GKZYXC>
{
};

template <>
struct DELayoutSelector<3> : type_function<ctl::GNDHWK>
{
};

template <>
struct RLayoutSelector<3> : type_function<ctl::GNDHW>
{
};

template <ck::index_t NDimSpatial>
using ALayout = typename ALayoutSelector<NDimSpatial>::type;

template <ck::index_t NDimSpatial>
using BLayout = typename BLayoutSelector<NDimSpatial>::type;

template <ck::index_t NDimSpatial>
using DELayout = typename DELayoutSelector<NDimSpatial>::type;

template <ck::index_t NDimSpatial>
using RLayout = typename RLayoutSelector<NDimSpatial>::type;

struct ExecutionConfig final
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
};

inline void print_help_msg()
{
    std::cerr << "arg1: verification (0=no, 1=yes)\n"
              << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n"
              << "arg3: time kernel (0=no, 1=yes)\n"
              << ck::utils::conv::get_conv_param_parser_helper_msg() << std::endl;
}

inline bool parse_cmd_args(int argc,
                           char* argv[],
                           ck::utils::conv::ConvParam& problem_size,
                           ExecutionConfig& config)
{
    if(argc == 1)
    {
        // use default
    }
    else if(argc == 4)
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
    }
    else if((5 + 4) < argc && ((argc - (5 + 4)) % 3 == 0))
    {
        config.do_verification = std::stoi(argv[1]);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);

        const ck::index_t num_dim_spatial = std::stoi(argv[4]);
        problem_size = ck::utils::conv::parse_conv_param(num_dim_spatial, 5, argv);
    }
    else
    {
        print_help_msg();
        return false;
    }

    return true;
}

inline HostTensorDescriptor
make_r0_host_tensor_descriptor(const ck::utils::conv::ConvParam& problem_size)
{
    std::vector<ck::index_t> dimensions{problem_size.G_, problem_size.N_};

    std::copy(begin(problem_size.output_spatial_lengths_),
              end(problem_size.output_spatial_lengths_),
              std::back_inserter(dimensions));

    return HostTensorDescriptor(dimensions);
}

template <typename Lengths, typename Strides>
void unpack_host_tensor_descriptor(const HostTensorDescriptor& descriptor,
                                   Lengths& lengths,
                                   Strides& strides)
{
    assert(size(descriptor.GetLengths()) == size(lengths));
    std::copy_n(begin(descriptor.GetLengths()), size(descriptor.GetLengths()), begin(lengths));

    assert(size(descriptor.GetStrides()) == size(strides));
    std::copy_n(begin(descriptor.GetStrides()), size(descriptor.GetStrides()), begin(strides));
}

template <typename Range, typename OutputIterator>
auto copy(const Range& range, OutputIterator iter)
    -> decltype(std::copy(std::begin(range), std::end(range), iter))
{
    return std::copy(std::begin(range), std::end(range), iter);
}
