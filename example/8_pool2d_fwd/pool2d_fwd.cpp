#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "device_tensor.hpp"
#include "tensor_layout.hpp"
#include "reduction_operator.hpp"
#include "device_operation/include/device_pool2d_fwd_nhwc_nhwc.hpp"

using namespace ck;
using namespace ck::tensor_operation::device;

using InDataType  = ck::half_t;
using OutDataType = ck::half_t;
using AccDataType = float;

using InLayout  = ck::tensor_layout::pool::NHWC;
using OutLayout = ck::tensor_layout::pool::NHWC;

// TODO: reimplement reduction as elementwise operator
static constexpr auto ReduceOpId = ck::ReduceTensorOp_t::MAX;
// static constexpr auto ReduceOpId = ck::ReduceTensorOp_t::AVG;
static constexpr bool NeedIndices = false;

using ReduceOperation = typename reduce_binary_operator<AccDataType, ReduceOpId>::opType;
using InElementwiseOperation =
    typename reduce_unary_operator<AccDataType, ReduceOpId, true, true>::InElementwiseOperation;
using AccElementwiseOperation =
    typename reduce_unary_operator<AccDataType, ReduceOpId, true, true>::AccElementwiseOperation;

using DevicePoolFwdInstance =
    DevicePool2dFwd_Input_N_Hi_Wi_C_Output_N_Ho_Wo_C<InDataType,  // InDataType
                                                     OutDataType, // OutDataType
                                                     AccDataType, // AccDataType
                                                     ReduceOperation,
                                                     InElementwiseOperation,
                                                     AccElementwiseOperation,
                                                     NeedIndices,
                                                     256, // BlockSize
                                                     512, // ReduceMPerBlock
                                                     1,   // ReduceKPerBlock
                                                     2,   // ReduceMPerThread
                                                     1>;  // ReduceKPerThread

template <typename TIn, typename TOut>
void max_pool_host_verify(const Tensor<TIn>& in,
                          Tensor<TOut>& out,
                          const std::vector<ck::index_t>& window_spatial_lengths,
                          const std::vector<ck::index_t>& window_strides,
                          const std::vector<ck::index_t>& in_left_pads,
                          const std::vector<ck::index_t>& in_right_pads)
{
    auto f_nchw = [&](auto n, auto c, auto ho, auto wo) {
        TIn v = std::numeric_limits<int>::min();

        for(int y = 0; y < window_spatial_lengths[0]; ++y)
        {
            int hi = ho * window_strides[0] + y - in_left_pads[0];
            for(int x = 0; x < window_spatial_lengths[1]; ++x)
            {
                int wi = wo * window_strides[1] + x - in_left_pads[1];
                if(hi >= 0 && hi < in.mDesc.GetLengths()[2] && wi >= 0 &&
                   wi < in.mDesc.GetLengths()[3])
                {
                    TIn in_v = in(n, c, hi, wi);
                    v        = v > in_v ? v : in_v;
                }
            }
        }

        out(n, c, ho, wo) = v;
    };

    make_ParallelTensorFunctor(f_nchw,
                               out.mDesc.GetLengths()[0],
                               out.mDesc.GetLengths()[1],
                               out.mDesc.GetLengths()[2],
                               out.mDesc.GetLengths()[3])(std::thread::hardware_concurrency());
}

template <typename TIn, typename TOut>
void average_pool_host_verify(const Tensor<TIn>& in,
                              Tensor<TOut>& out,
                              const std::vector<ck::index_t>& window_spatial_lengths,
                              const std::vector<ck::index_t>& window_strides,
                              const std::vector<ck::index_t>& in_left_pads,
                              const std::vector<ck::index_t>& in_right_pads)
{
    auto f_nchw = [&](auto n, auto c, auto ho, auto wo) {
        TIn v = std::numeric_limits<int>::min();

        for(int y = 0; y < window_spatial_lengths[0]; ++y)
        {
            int hi = ho * window_strides[0] + y - in_left_pads[0];
            for(int x = 0; x < window_spatial_lengths[1]; ++x)
            {
                int wi = wo * window_strides[1] + x - in_left_pads[1];
                if(hi >= 0 && hi < in.mDesc.GetLengths()[2] && wi >= 0 &&
                   wi < in.mDesc.GetLengths()[3])
                {
                    v += in(n, c, hi, wi);
                }
            }
        }

        out(n, c, ho, wo) = v / (window_spatial_lengths[0] * window_spatial_lengths[1]);
    };

    make_ParallelTensorFunctor(f_nchw,
                               out.mDesc.GetLengths()[0],
                               out.mDesc.GetLengths()[1],
                               out.mDesc.GetLengths()[2],
                               out.mDesc.GetLengths()[3])(std::thread::hardware_concurrency());
}

int main(int argc, char* argv[])
{
    bool do_verification = 0;
    int init_method      = 0;
    int nrepeat          = 5;

    // Pool shape
    ck::index_t N               = 128;
    ck::index_t C               = 192;
    ck::index_t Y               = 3;
    ck::index_t X               = 3;
    ck::index_t Hi              = 71;
    ck::index_t Wi              = 71;
    ck::index_t window_stride_h = 2;
    ck::index_t window_stride_w = 2;
    ck::index_t in_left_pad_h   = 1;
    ck::index_t in_left_pad_w   = 1;
    ck::index_t in_right_pad_h  = 1;
    ck::index_t in_right_pad_w  = 1;

    if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        nrepeat         = std::stoi(argv[3]);
    }
    else if(argc == 16)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        nrepeat         = std::stoi(argv[3]);

        N               = std::stoi(argv[4]);
        C               = std::stoi(argv[5]);
        Y               = std::stoi(argv[6]);
        X               = std::stoi(argv[7]);
        Hi              = std::stoi(argv[8]);
        Wi              = std::stoi(argv[9]);
        window_stride_h = std::stoi(argv[10]);
        window_stride_w = std::stoi(argv[11]);
        in_left_pad_h   = std::stoi(argv[12]);
        in_left_pad_w   = std::stoi(argv[13]);
        in_right_pad_h  = std::stoi(argv[14]);
        in_right_pad_w  = std::stoi(argv[15]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: run kernel # of times (>1)\n");
        printf("arg4 to 15: N, C, Y, X, Hi, Wi, Sy, Sx, LeftPy, LeftPx, RightPy, "
               "RightPx\n");
        exit(0);
    }

    const ck::index_t Ho = (Hi + in_left_pad_h + in_right_pad_h - Y) / window_stride_h + 1;
    const ck::index_t Wo = (Wi + in_left_pad_w + in_right_pad_w - X) / window_stride_w + 1;

    const std::vector<ck::index_t> window_spatial_lengths{{Y, X}};
    const std::vector<ck::index_t> window_strides{{window_stride_h, window_stride_w}};
    const std::vector<ck::index_t> input_left_pads{{in_left_pad_h, in_left_pad_w}};
    const std::vector<ck::index_t> input_right_pads{{in_right_pad_h, in_right_pad_w}};

    // tensor layout
    auto f_host_tensor_descriptor =
        [](std::size_t N_, std::size_t C_, std::size_t H, std::size_t W, auto layout) {
            if constexpr(ck::is_same<decltype(layout), ck::tensor_layout::pool::NCHW>::value)
            {
                return HostTensorDescriptor(std::vector<std::size_t>({N_, C_, H, W}),
                                            std::vector<std::size_t>({C_ * H * W, H * W, W, 1}));
            }
            else if constexpr(ck::is_same<decltype(layout), ck::tensor_layout::pool::NHWC>::value)
            {
                return HostTensorDescriptor(std::vector<std::size_t>({N_, C_, H, W}),
                                            std::vector<std::size_t>({C_ * H * W, 1, W * C_, C_}));
            }
        };

    Tensor<InDataType> in_n_c_hi_wi(f_host_tensor_descriptor(N, C, Hi, Wi, InLayout{}));
    Tensor<OutDataType> out_n_c_ho_wo_host_result(
        f_host_tensor_descriptor(N, C, Ho, Wo, OutLayout{}));
    Tensor<OutDataType> out_n_c_ho_wo_device_result(
        f_host_tensor_descriptor(N, C, Ho, Wo, OutLayout{}));

    std::cout << "in_n_c_hi_wi: " << in_n_c_hi_wi.mDesc << std::endl;
    std::cout << "out_n_c_ho_wo: " << out_n_c_ho_wo_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1: in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5}); break;
    default: in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 1.0});
    }

    DeviceMem in_device_buf(sizeof(InDataType) * in_n_c_hi_wi.mDesc.GetElementSpace());
    DeviceMem out_device_buf(sizeof(OutDataType) *
                             out_n_c_ho_wo_device_result.mDesc.GetElementSpace());

    in_device_buf.ToDevice(in_n_c_hi_wi.mData.data());

    auto pool     = DevicePoolFwdInstance{};
    auto invoker  = pool.MakeInvoker();
    auto argument = pool.MakeArgument(static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
                                      static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                                      N,
                                      C,
                                      std::vector<ck::index_t>{{Hi, Wi}},
                                      std::vector<ck::index_t>{{Y, X}},
                                      std::vector<ck::index_t>{{Ho, Wo}},
                                      window_strides,
                                      input_left_pads,
                                      input_right_pads,
                                      InElementwiseOperation{},
                                      AccElementwiseOperation{});

    if(!pool.IsSupportedArgument(argument))
    {
        throw std::runtime_error("wrong! device_op with the specified compilation parameters does "
                                 "not support this problem");
    }

    float ave_time = invoker.Run(argument, nrepeat);

    std::size_t flop = std::size_t(2) * N * C * Ho * Wo * Y * X;

    std::size_t num_btype =
        sizeof(InDataType) * (N * C * Hi * Wi) + sizeof(OutDataType) * (N * C * Ho * Wo);

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    if(do_verification)
    {
        max_pool_host_verify(in_n_c_hi_wi,
                             out_n_c_ho_wo_host_result,
                             window_spatial_lengths,
                             window_strides,
                             input_left_pads,
                             input_right_pads);

        out_device_buf.FromDevice(out_n_c_ho_wo_device_result.mData.data());

        check_error(out_n_c_ho_wo_host_result, out_n_c_ho_wo_device_result);
    }
}
