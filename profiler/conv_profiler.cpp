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
#include "host_conv.hpp"
#include "device_tensor.hpp"
#include "device_conv_xdl.hpp"
#include "profile_conv.hpp"

struct ConvDataType
{
    F32_F32_F32,     // 0
        F16_F16_F16, // 1
};

{
    if(argc != 24)
    {
        printf("arg1: data type (0=fp32, 1=fp16)\n");
        printf("arg2: input tensor layout (0=NHWC)\n");
        printf("arg3: weight tensor layout (0=KYXC)\n");
        printf("arg4: output tensor layout (0=NHWK)\n");
        printf("arg5: verification (0=no, 1=yes)\n");
        printf("arg6: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg7: print matrix value (0=no, 1=yes)\n");
        printf("arg8: run kernel # of times (>1)\n");
        printf("arg9 to 23: N, K, C, Y, X, Hi, Wi, Sy, Sx, Dy, Dx, LeftPy, LeftPx, RightPy, "
               "RightPx\n");
        exit(1);
    }

    const int data_type = static_cast<ConvDataType>(std::stoi(argv[1]));
#if 0
    const int in_layout        = static_cast<GemmMatrixLayout>(std::stoi(argv[2]));
    const int wei_layout       = static_cast<GemmMatrixLayout>(std::stoi(argv[2]));
    const int out_layout       = static_cast<GemmMatrixLayout>(std::stoi(argv[2]));
#endif
    const bool do_verification = std::stoi(argv[3]);
    const int init_method      = std::stoi(argv[4]);
    const bool do_log          = std::stoi(argv[5]);
    const int nrepeat          = std::stoi(argv[6]);

    const ck::index_t N  = std::stoi(argv[7]);
    const ck::index_t K  = std::stoi(argv[8]);
    const ck::index_t C  = std::stoi(argv[9]);
    const ck::index_t Y  = std::stoi(argv[10]);
    const ck::index_t X  = std::stoi(argv[11]);
    const ck::index_t Hi = std::stoi(argv[12]);
    const ck::index_t Wi = std::stoi(argv[13]);

    const ck::index_t conv_stride_h   = std::stoi(argv[14]);
    const ck::index_t conv_stride_w   = std::stoi(argv[15]);
    const ck::index_t conv_dilation_h = std::stoi(argv[16]);
    const ck::index_t conv_dilation_w = std::stoi(argv[17]);
    const ck::index_t in_left_pad_h   = std::stoi(argv[18]);
    const ck::index_t in_left_pad_w   = std::stoi(argv[19]);
    const ck::index_t in_right_pad_h  = std::stoi(argv[20]);
    const ck::index_t in_right_pad_w  = std::stoi(argv[21]);

    if(data_type == ConvDataType::F32_F32_F32)
    {
        ck::profiler::profile_conv<float,
                                   float,
                                   float,
                                   ck::tensor_layout::ColumnMajor,
                                   ck::tensor_layout::ColumnMajor,
                                   ck::tensor_layout::RowMajor>(
            do_verification, init_method, do_log, nrepeat, M, N, K, StrideA, StrideB, StrideC);
    }
    else
    {
        throw std::runtime_error("wrong! this GEMM data_type & layout is not implemented");
    }

    return 1;
}
