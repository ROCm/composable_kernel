#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>

#include "check_err.hpp"
#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "host_reduce_util.hpp"
#include "device_tensor.hpp"
#include "tensor_layout.hpp"
#include "reduction_operator.hpp"
#include "device_pool2d_fwd_nhwc_nhwc.hpp"

using InDataType  = ck::half_t;
using OutDataType = ck::half_t;
using AccDataType = float;

using InLayout  = ck::tensor_layout::convolution::NHWC;
using OutLayout = ck::tensor_layout::convolution::NHWC;

#if 1
static constexpr auto ReduceOpId = ck::ReduceTensorOp::MAX;
#else
static constexpr auto ReduceOpId = ck::ReduceTensorOp::AVG;
#endif

static constexpr bool NeedIndices  = false;
static constexpr bool PropagateNan = false;

using DevicePoolFwdInstance =
    ck::tensor_operation::device::DevicePool2dFwd_Input_N_Hi_Wi_C_Output_N_Ho_Wo_C<
        InDataType,  // InDataType
        OutDataType, // OutDataType
        AccDataType, // AccDataType
        ReduceOpId,
        NeedIndices,
        64, // BlockSize
        64, // ReduceMThreadClusterSize
        1,  // ReduceKThreadClusterSize
        4,  // ReduceMThreadSliceSize
        1,  // ReduceKThreadSliceSize
        4>; // InSrcOutDstVectorSize

template <typename InDataType,
          typename OutDataType,
          typename AccDataType,
          ck::ReduceTensorOp ReduceOpId,
          bool PropagateNan,
          bool NeedIndices>
static void pool_host_verify(const Tensor<InDataType>& in,
                             Tensor<OutDataType>& out,
                             Tensor<int>& out_indices,
                             const std::array<ck::index_t, 2>& window_spatial_lengths,
                             const std::array<ck::index_t, 2>& window_strides,
                             const std::array<ck::index_t, 2>& in_left_pads,
                             const std::array<ck::index_t, 2>& /*in_right_pads*/)
{
    using namespace ck::host_reduce;

    const int divider = window_spatial_lengths[0] * window_spatial_lengths[1];

    const auto PreUnaryOp = PreUnaryOpFn<AccDataType, ReduceOpId>(divider);
    const auto PosUnaryOp = PosUnaryOpFn<AccDataType, ReduceOpId>(divider);

    if constexpr(!NeedIndices)
    {
        auto opReduce = ReduceOpFn<AccDataType, ReduceOpId>();

        auto f_nchw = [&](auto n, auto c, auto ho, auto wo) {
            auto accuVal = ReduceOpZeroVal<AccDataType, ReduceOpId>();

            for(int y = 0; y < window_spatial_lengths[0]; ++y)
            {
                int hi = ho * window_strides[0] + y - in_left_pads[0];
                for(int x = 0; x < window_spatial_lengths[1]; ++x)
                {
                    int wi = wo * window_strides[1] + x - in_left_pads[1];
                    if(hi >= 0 && hi < in.mDesc.GetLengths()[2] && wi >= 0 &&
                       wi < in.mDesc.GetLengths()[3])
                    {
                        AccDataType currVal = static_cast<AccDataType>(in(n, c, hi, wi));

                        PreUnaryOp(currVal);

                        binop_with_nan_check<AccDataType, PropagateNan>(opReduce, accuVal, currVal);
                    }
                }
            }

            PosUnaryOp(accuVal);

            out(n, c, ho, wo) = accuVal;
        };

        make_ParallelTensorFunctor(f_nchw,
                                   out.mDesc.GetLengths()[0],
                                   out.mDesc.GetLengths()[1],
                                   out.mDesc.GetLengths()[2],
                                   out.mDesc.GetLengths()[3])(std::thread::hardware_concurrency());
    }
    else
    {
        auto opReduce = ReduceOpFn2<AccDataType, ReduceOpId>();

        auto f_nchw = [&](auto n, auto c, auto ho, auto wo) {
            auto accuVal  = ReduceOpZeroVal<AccDataType, ReduceOpId>();
            int accuIndex = 0;

            for(int y = 0; y < window_spatial_lengths[0]; ++y)
            {
                int hi = ho * window_strides[0] + y - in_left_pads[0];
                for(int x = 0; x < window_spatial_lengths[1]; ++x)
                {
                    int wi = wo * window_strides[1] + x - in_left_pads[1];
                    if(hi >= 0 && hi < in.mDesc.GetLengths()[2] && wi >= 0 &&
                       wi < in.mDesc.GetLengths()[3])
                    {
                        AccDataType currVal = static_cast<AccDataType>(in(n, c, hi, wi));
                        int currIndex       = y * window_spatial_lengths[1] + x;

                        PreUnaryOp(currVal);

                        binop_with_nan_check2<AccDataType, PropagateNan>(
                            opReduce, accuVal, currVal, accuIndex, currIndex);
                    }
                }
            }

            PosUnaryOp(accuVal);

            out(n, c, ho, wo)         = accuVal;
            out_indices(n, c, ho, wo) = accuIndex;
        };

        make_ParallelTensorFunctor(f_nchw,
                                   out.mDesc.GetLengths()[0],
                                   out.mDesc.GetLengths()[1],
                                   out.mDesc.GetLengths()[2],
                                   out.mDesc.GetLengths()[3])(std::thread::hardware_concurrency());
    };
}

int main(int argc, char* argv[])
{
    using namespace ck::host_reduce;

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

    const std::array<ck::index_t, 2> window_spatial_lengths{{Y, X}};
    const std::array<ck::index_t, 2> window_strides{{window_stride_h, window_stride_w}};
    const std::array<ck::index_t, 2> input_left_pads{{in_left_pad_h, in_left_pad_w}};
    const std::array<ck::index_t, 2> input_right_pads{{in_right_pad_h, in_right_pad_w}};

    // tensor layout
    auto f_host_tensor_descriptor =
        [](std::size_t N_, std::size_t C_, std::size_t H, std::size_t W, auto layout) {
            if constexpr(ck::is_same<decltype(layout), ck::tensor_layout::convolution::NCHW>::value)
            {
                return HostTensorDescriptor(std::vector<std::size_t>({N_, C_, H, W}),
                                            std::vector<std::size_t>({C_ * H * W, H * W, W, 1}));
            }
            else if constexpr(ck::is_same<decltype(layout),
                                          ck::tensor_layout::convolution::NHWC>::value)
            {
                return HostTensorDescriptor(std::vector<std::size_t>({N_, C_, H, W}),
                                            std::vector<std::size_t>({C_ * H * W, 1, W * C_, C_}));
            }
        };

    Tensor<InDataType> in_n_c_hi_wi(f_host_tensor_descriptor(N, C, Hi, Wi, InLayout{}));
    Tensor<OutDataType> out_n_c_ho_wo_host(f_host_tensor_descriptor(N, C, Ho, Wo, OutLayout{}));
    Tensor<int> out_indices_n_c_ho_wo_host(f_host_tensor_descriptor(N, C, Ho, Wo, OutLayout{}));
    Tensor<OutDataType> out_n_c_ho_wo_device(f_host_tensor_descriptor(N, C, Ho, Wo, OutLayout{}));
    Tensor<int> out_indices_n_c_ho_wo_device(f_host_tensor_descriptor(N, C, Ho, Wo, OutLayout{}));

    std::cout << "in_n_c_hi_wi: " << in_n_c_hi_wi.mDesc << std::endl;
    std::cout << "out_n_c_ho_wo: " << out_n_c_ho_wo_host.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1: in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_1<InDataType>{1}); break;
    case 2: in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5}); break;
    default: in_n_c_hi_wi.GenerateTensorValue(GeneratorTensor_3<InDataType>{-5.0, 5.0});
    }

    DeviceMem in_device_buf(sizeof(InDataType) * in_n_c_hi_wi.mDesc.GetElementSpace());
    DeviceMem out_device_buf(sizeof(OutDataType) * out_n_c_ho_wo_device.mDesc.GetElementSpace());
    DeviceMem out_indices_device_buf(sizeof(int) *
                                     out_indices_n_c_ho_wo_device.mDesc.GetElementSpace());

    in_device_buf.ToDevice(in_n_c_hi_wi.mData.data());

    auto pool        = DevicePoolFwdInstance{};
    auto invoker_ptr = pool.MakeInvokerPointer();
    auto argument_ptr =
        pool.MakeArgumentPointer(static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
                                 static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                                 static_cast<int*>(out_indices_device_buf.GetDeviceBuffer()),
                                 N,
                                 C,
                                 std::array<ck::index_t, 2>{{Hi, Wi}},
                                 std::array<ck::index_t, 2>{{Y, X}},
                                 std::array<ck::index_t, 2>{{Ho, Wo}},
                                 window_strides,
                                 input_left_pads,
                                 input_right_pads);

    if(!pool.IsSupportedArgument(argument_ptr.get()))
    {
        throw std::runtime_error("wrong! device_op with the specified compilation parameters does "
                                 "not support this problem");
    }

    float ave_time = invoker_ptr->Run(argument_ptr.get(), nrepeat);

    std::size_t flop = std::size_t(2) * N * C * Ho * Wo * Y * X;

    std::size_t num_btype =
        sizeof(InDataType) * (N * C * Hi * Wi) + sizeof(OutDataType) * (N * C * Ho * Wo);

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    if(do_verification)
    {
        pool_host_verify<InDataType,
                         OutDataType,
                         AccDataType,
                         ReduceOpId,
                         PropagateNan,
                         NeedIndices>(in_n_c_hi_wi,
                                      out_n_c_ho_wo_host,
                                      out_indices_n_c_ho_wo_host,
                                      window_spatial_lengths,
                                      window_strides,
                                      input_left_pads,
                                      input_right_pads);

        out_device_buf.FromDevice(out_n_c_ho_wo_device.mData.data());

        ck::utils::check_err(out_n_c_ho_wo_device.mData, out_n_c_ho_wo_host.mData);

        if constexpr(NeedIndices)
        {
            out_indices_device_buf.FromDevice(out_indices_n_c_ho_wo_device.mData.data());

            //          ck::utils::check_err(out_indices_n_c_ho_wo_device.mData,
            //          out_indices_n_c_ho_wo_host.mData);;
        };
    }
}
