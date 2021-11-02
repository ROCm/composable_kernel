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
    enum GemmLayout
    {
        MK_KN, // 0: NN
        MK_NK, // 1: NT
        KM_KN, // 2: TN
        KM_NK, // 3: TT
    };

    // Currently ADataType and BDataType need to be the same
    using ADataType   = ck::half_t;
    using BDataType   = ck::half_t;
    using CDataType   = ck::half_t;
    using AccDataType = float;

    if(argc != 12)
    {
        printf("arg1 to 5: layout, do_verification, init_method, do_log, nrepeat\n");
        printf("arg6 to 11: M, N, K, StrideA, StrideB, StrideC\n");
        exit(1);
    }

    const auto layout          = static_cast<GemmLayout>(std::stoi(argv[1]));
    const bool do_verification = std::stoi(argv[2]);
    const int init_method      = std::stoi(argv[3]);
    const bool do_log          = std::stoi(argv[4]);
    const int nrepeat          = std::stoi(argv[5]);

    const int M = std::stoi(argv[6]);
    const int N = std::stoi(argv[7]);
    const int K = std::stoi(argv[8]);

    const int StrideA = std::stoi(argv[9]);
    const int StrideB = std::stoi(argv[10]);
    const int StrideC = std::stoi(argv[11]);

    if(layout == GemmLayout::MK_NK)
    {
        ck::profiler::profile_gemm<ADataType,
                                   BDataType,
                                   CDataType,
                                   AccDataType,
                                   ck::tensor_layout::RowMajor,
                                   ck::tensor_layout::ColumnMajor,
                                   ck::tensor_layout::RowMajor>(
            do_verification, init_method, do_log, nrepeat, M, N, K, StrideA, StrideB, StrideC);
    }
    else if(layout == GemmLayout::KM_KN)
    {
        ck::profiler::profile_gemm<ADataType,
                                   BDataType,
                                   CDataType,
                                   AccDataType,
                                   ck::tensor_layout::ColumnMajor,
                                   ck::tensor_layout::RowMajor,
                                   ck::tensor_layout::RowMajor>(
            do_verification, init_method, do_log, nrepeat, M, N, K, StrideA, StrideB, StrideC);
    }
    else
    {
        throw std::runtime_error("wrong! this GEMM layout not implemented");
    }

    return 1;
}
