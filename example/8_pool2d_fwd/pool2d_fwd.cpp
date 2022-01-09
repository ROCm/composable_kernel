#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>
#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "device_tensor.hpp"
#include "device_operation/include/device_pool2d_fwd_nhwc_nhwc.hpp"
#include "element_wise_operation.hpp"

using InDataType  = ck::half_t;
using OutDataType = ck::half_t;
using AccDataType = float;

using InLayout  = ck::tensor_layout::pool::NHWC;
using OutLayout = ck::tensor_layout::pool::NHWC;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

// TODO: reimplement reduction as elementwise operator
static constexpr auto Max = ReduceTensorOp_t::MAX;

using DevicePoolFwdInstance =
    ck::tensor_operation::device::DevicePool2dFwd_Input_N_Hi_Wi_C_Output_N_Ho_Wo_C<InDataType,
                                                                                   OutDataType,
                                                                                   AccDataType,
                                                                                   Max,
                                                                                   InElementOp,
                                                                                   OutElementOp,
                                                                                   256,
                                                                                   256,
                                                                                   1,
                                                                                   1,
                                                                                   1>;

int main(int argc, char* argv[]) {}
