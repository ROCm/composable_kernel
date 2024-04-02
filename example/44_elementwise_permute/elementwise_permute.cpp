// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_elementwise_impl.hpp"

#include "ck/library/reference_tensor_operation/cpu/reference_elementwise.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

using F16 = ck::half_t;
using F32 = float;

using ADataType = F16;
using BDataType = F16;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using DeviceElementwisePermuteInstance =
    ck::tensor_operation::device::DeviceElementwiseImpl<ck::Tuple<ADataType>, // InDataTypeTuple
                                                        ck::Tuple<BDataType>, // OutDataTypeTuple
                                                        PassThrough,          // ElementwiseOp
                                                        5,                    // NumDim
                                                        8,                    // MPerThread
                                                        ck::Sequence<1>,  // InScalarPerVectorSeq
                                                        ck::Sequence<1>>; // OutScalarPerVectorSeq

int main()
{
    bool do_verification = true;
    bool time_kernel     = true;

    std::vector<std::size_t> ncdhw = {16, 8, 8, 8, 8};
    std::vector<std::size_t> ndhwc = {16, 8, 8, 8, 8};
    std::array<ck::index_t, 5> ab_lengths;

    std::array<ck::index_t, 5> a_strides = {
        static_cast<int>(ncdhw[1] * ncdhw[2] * ncdhw[3] * ncdhw[4]),
        static_cast<int>(ncdhw[3] * ncdhw[4]),
        static_cast<int>(ncdhw[4]),
        1,
        static_cast<int>(ncdhw[2] * ncdhw[3] * ncdhw[4])};

    std::array<ck::index_t, 5> b_strides = {
        static_cast<int>(ndhwc[1] * ndhwc[2] * ndhwc[3] * ndhwc[4]),
        static_cast<int>(ndhwc[2] * ndhwc[3] * ndhwc[4]),
        static_cast<int>(ndhwc[3] * ndhwc[4]),
        static_cast<int>(ndhwc[4]),
        1};
    ck::ranges::copy(ncdhw, ab_lengths.begin());

    std::array<Tensor<ADataType>, 1> as = {Tensor<ADataType>(ab_lengths, a_strides)};
    Tensor<ADataType>& a                = as[0];
    Tensor<BDataType> b(ab_lengths, b_strides);

    a.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});

    DeviceMem a_device_buf(sizeof(ADataType) * a.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a.mData.data());

    std::array<const void*, 1> input = {a_device_buf.GetDeviceBuffer()};
    std::array<void*, 1> output      = {b_device_buf.GetDeviceBuffer()};

    auto broadcastPermute = DeviceElementwisePermuteInstance{};
    auto argument         = broadcastPermute.MakeArgumentPointer(
        ab_lengths, {a_strides}, {b_strides}, input, output, PassThrough{});

    if(!broadcastPermute.IsSupportedArgument(argument.get()))
    {
        throw std::runtime_error(
            "The runtime parameters seems not supported by the device instance, exiting!");
    };

    std::cout << "A (ncdhw): " << a.mDesc << std::endl;
    std::cout << "B (ndhwc): " << b.mDesc << std::endl;

    auto broadcastPermute_invoker_ptr = broadcastPermute.MakeInvokerPointer();
    float ave_time =
        broadcastPermute_invoker_ptr->Run(argument.get(), StreamConfig{nullptr, time_kernel});
    std::size_t flop = std::size_t(2) * ncdhw[0] * ncdhw[1] * ncdhw[2] * ncdhw[3] * ncdhw[4];

    std::size_t num_btype =
        sizeof(ADataType) * (ncdhw[0] * ncdhw[1] * ncdhw[2] * ncdhw[3] * ncdhw[4]) +
        sizeof(BDataType) * (ncdhw[0] * ncdhw[1] * ncdhw[2] * ncdhw[3] * ncdhw[4]);

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    bool pass = true;

    if(do_verification)
    {
        Tensor<BDataType> host_b(ab_lengths, b_strides);
        using ReferenceElementwiseInstance =
            ck::tensor_operation::host::ReferenceElementwise<1, ADataType, BDataType, PassThrough>;
        auto ref_elementwise = ReferenceElementwiseInstance{};
        auto ref_invoker     = ref_elementwise.MakeInvoker();

        auto ref_argument = ref_elementwise.MakeArgument(as, host_b, PassThrough{});
        ref_invoker.Run(ref_argument);

        b_device_buf.FromDevice(b.mData.data());
        pass &=
            ck::utils::check_err(b.mData, host_b.mData, "Error: Incorrect results b", 1e-3, 1e-3);
    }

    return pass ? 0 : 1;
}
