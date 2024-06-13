#pragma once

#include "ck/library/utility/convolution_parameter.hpp"

namespace ck {
namespace profiler {

template <ck::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename OutElementOp,
          typename AComputeType = InDataType,
          typename BComputeType = AComputeType>
bool profile_grouped_conv_fwd_outelementop_impl(int do_verification,
                                                int init_method,
                                                bool do_log,
                                                bool time_kernel,
                                                const ck::utils::conv::ConvParam& conv_param)
{
    // TODO: Implement the profiling logic here
    return false;
}

} // namespace profiler
} // namespace ck
