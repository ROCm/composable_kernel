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
#include "gemm_common.hpp"
#include "host_gemm.hpp"
#include "device_tensor.hpp"
#include "device_base.hpp"
#include "device_gemm_xdl.hpp"
#include "gemm_profiler.hpp"

int main(int argc, char* argv[])
{
    // Currently ADataType and BDataType need to be the same
    using ADataType   = ck::half_t;
    using BDataType   = ck::half_t;
    using CDataType   = ck::half_t;
    using AccDataType = float;

    // NT problem
    using ALayout = ck::tensor_layout::RowMajor;
    using BLayout = ck::tensor_layout::ColumnMajor;
    using CLayout = ck::tensor_layout::RowMajor;

    ck::profiler::
        profile_gemm<ADataType, BDataType, CDataType, AccDataType, ALayout, BLayout, CLayout>(argc,
                                                                                              argv);

    return 1;
}
