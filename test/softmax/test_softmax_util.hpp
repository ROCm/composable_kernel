// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>
#include <iostream>
#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/utility/number.hpp"
#include "ck/tensor_operation/gpu/device/device_softmax.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_softmax.hpp"

namespace ck {

template <typename Range>
std::string serialize_range(const Range& range)
{
    std::stringstream ss;
    for(auto& r : range)
    {
        ss << r << ", ";
    }
    std::string str = ss.str();
    return std::string(str.begin(), str.end() - 2);
}

template <typename Tuple>
class TestSoftmax : public ::testing::Test
{
    protected:
    using InDataType                            = std::tuple_element_t<0, Tuple>;
    using AccDataType                           = std::tuple_element_t<1, Tuple>;
    using OutDataType                           = std::tuple_element_t<2, Tuple>;
    static constexpr index_t Rank               = std::tuple_element_t<3, Tuple>{}.value;
    static constexpr index_t NumReduceDim       = std::tuple_element_t<4, Tuple>{}.value;
    static constexpr index_t BlockSize          = std::tuple_element_t<5, Tuple>{}.value;
    static constexpr index_t MThreadClusterSize = std::tuple_element_t<6, Tuple>{}.value;
    static constexpr index_t KThreadClusterSize = std::tuple_element_t<7, Tuple>{}.value;
    static constexpr index_t MThreadSliceSize   = std::tuple_element_t<8, Tuple>{}.value;
    static constexpr index_t KThreadSliceSize   = std::tuple_element_t<9, Tuple>{}.value;
    static constexpr index_t InSrcVectorDim     = std::tuple_element_t<10, Tuple>{}.value;
    static constexpr index_t InSrcVectorSize    = std::tuple_element_t<11, Tuple>{}.value;
    static constexpr index_t OutDstVectorSize   = std::tuple_element_t<12, Tuple>{}.value;

    using ReferenceInstance =
        tensor_operation::host::ReferenceSoftmax<InDataType, OutDataType, AccDataType>;

    using DeviceInstance = tensor_operation::device::DeviceSoftmax<InDataType,
                                                                   AccDataType,
                                                                   OutDataType,
                                                                   Rank,
                                                                   NumReduceDim,
                                                                   BlockSize,
                                                                   MThreadClusterSize,
                                                                   KThreadClusterSize,
                                                                   MThreadSliceSize,
                                                                   KThreadSliceSize,
                                                                   InSrcVectorDim,
                                                                   InSrcVectorSize,
                                                                   OutDstVectorSize>;

    TestSoftmax() : ref_instance_invoker_(ReferenceInstance{}.MakeInvoker()) {}

    void RunSingle(std::vector<index_t> in_length, AccDataType alpha, AccDataType beta)
    {
        std::vector<index_t> reduce_dims(NumReduceDim);
        std::iota(reduce_dims.begin(), reduce_dims.end(), Rank - NumReduceDim);

        Tensor<InDataType> in(in_length);
        Tensor<OutDataType> out(in_length);

        in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        out.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});

        Tensor<OutDataType> out_ref(out);

        DeviceMem in_dev(sizeof(InDataType) * in.mDesc.GetElementSpaceSize());
        DeviceMem out_dev(sizeof(OutDataType) * out.mDesc.GetElementSpaceSize());
        in_dev.ToDevice(in.mData.data());
        out_dev.ToDevice(out.mData.data());

        std::vector<index_t> i_in_lengths(in.mDesc.GetLengths().begin(),
                                          in.mDesc.GetLengths().end());
        std::vector<index_t> i_in_strides(in.mDesc.GetStrides().begin(),
                                          in.mDesc.GetStrides().end());

        auto device_instance = DeviceInstance{};
        auto argument_ptr    = device_instance.MakeArgumentPointer(i_in_lengths,
                                                                i_in_strides,
                                                                reduce_dims,
                                                                &alpha,
                                                                &beta,
                                                                in_dev.GetDeviceBuffer(),
                                                                out_dev.GetDeviceBuffer());

        if(!device_instance.IsSupportedArgument(argument_ptr.get()))
        {
            // std::cout << "Skipped due to unsupported argument: "
            //           << "input lengths = [" << serialize_range(in_length) << "], "
            //           << "scaler = [" << alpha << ", " << beta << "]." << std::endl;
            return;
        }

        auto invoker_ptr = device_instance.MakeInvokerPointer();
        invoker_ptr->Run(argument_ptr.get());

        ref_instance_invoker_.Run({in, out_ref, alpha, beta, reduce_dims});

        out_dev.FromDevice(out.mData.data());

        bool pass;

        if(std::is_same<InDataType, int8_t>::value)
        {
            EXPECT_TRUE(pass = ck::utils::check_err(
                            out.mData, out_ref.mData, "Error: Incorrect results!", 0, 1));
        }
        else
        {
            EXPECT_TRUE(pass = ck::utils::check_err(out.mData, out_ref.mData));
        }

        if(!pass)
        {
            FAIL() << "Failure in input lengths = [" << serialize_range(in_length) << "], "
                   << "scaler = [" << alpha << ", " << beta << "].";
        }
    }

    void Run()
    {
        for(auto in_length : this->in_lengths_)
        {
            for(auto scale : this->scales_)
            {
                this->RunSingle(in_length, scale[0], scale[1]);
            }
        }
    }

    std::vector<std::vector<index_t>> in_lengths_ = {
        {1, 8, 128}, {2, 128, 1024}, {3, 9, 1032}, {4, 4, 2048}, {8, 1, 8192}};
    std::vector<std::vector<AccDataType>> scales_ = {{1, 0}, {1, 1}, {0, 1}, {2, 2}};

    typename ReferenceInstance::Invoker ref_instance_invoker_;
};
} // namespace ck
