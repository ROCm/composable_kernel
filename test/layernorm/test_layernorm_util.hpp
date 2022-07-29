// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>
#include <iostream>
#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/utility/number.hpp"
#include "ck/tensor_operation/gpu/device/device_layernorm.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_layernorm.hpp"

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
class TestLayernorm : public ::testing::Test
{
    protected:
    using XDataType                             = std::tuple_element_t<0, Tuple>;
    using GammaDataType                         = std::tuple_element_t<1, Tuple>;
    using BetaDataType                          = std::tuple_element_t<2, Tuple>;
    using AccDataType                           = std::tuple_element_t<3, Tuple>;
    using YDataType                             = std::tuple_element_t<4, Tuple>;
    static constexpr index_t Rank               = std::tuple_element_t<5, Tuple>{}.value;
    static constexpr index_t NumReduceDim       = std::tuple_element_t<6, Tuple>{}.value;
    static constexpr index_t BlockSize          = std::tuple_element_t<7, Tuple>{}.value;
    static constexpr index_t MThreadClusterSize = std::tuple_element_t<8, Tuple>{}.value;
    static constexpr index_t KThreadClusterSize = std::tuple_element_t<9, Tuple>{}.value;
    static constexpr index_t MThreadSliceSize   = std::tuple_element_t<10, Tuple>{}.value;
    static constexpr index_t KThreadSliceSize   = std::tuple_element_t<11, Tuple>{}.value;
    static constexpr index_t XYSrcVectorDim     = std::tuple_element_t<12, Tuple>{}.value;
    static constexpr index_t XSrcVectorSize     = std::tuple_element_t<13, Tuple>{}.value;
    static constexpr index_t GammaSrcVectorSize = std::tuple_element_t<14, Tuple>{}.value;
    static constexpr index_t BetaSrcVectorSize  = std::tuple_element_t<15, Tuple>{}.value;
    static constexpr index_t YDstVectorSize     = std::tuple_element_t<16, Tuple>{}.value;

    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    using ReferenceInstance = tensor_operation::host::ReferenceLayernorm<XDataType,
                                                                         GammaDataType,
                                                                         BetaDataType,
                                                                         YDataType,
                                                                         AccDataType,
                                                                         PassThrough,
                                                                         Rank,
                                                                         NumReduceDim>;

    using DeviceInstance = tensor_operation::device::DeviceLayernorm<XDataType,
                                                                     GammaDataType,
                                                                     BetaDataType,
                                                                     AccDataType,
                                                                     YDataType,
                                                                     PassThrough,
                                                                     Rank,
                                                                     NumReduceDim,
                                                                     BlockSize,
                                                                     MThreadClusterSize,
                                                                     KThreadClusterSize,
                                                                     MThreadSliceSize,
                                                                     KThreadSliceSize,
                                                                     XYSrcVectorDim,
                                                                     XSrcVectorSize,
                                                                     GammaSrcVectorSize,
                                                                     BetaSrcVectorSize,
                                                                     YDstVectorSize>;

    TestLayernorm() : ref_instance_invoker_(ReferenceInstance{}.MakeInvoker()) {}

    void RunSingle(std::vector<index_t> lengths, std::vector<index_t> reduceDims)
    {
        std::vector<index_t> reduceLength(reduceDims.size());
        for(int i = 0; i < NumReduceDim; ++i)
        {
            reduceLength[i] = lengths[reduceDims[i]];
        }

        Tensor<XDataType> x(lengths);
        Tensor<GammaDataType> gamma(reduceLength);
        Tensor<BetaDataType> beta(reduceLength);
        Tensor<YDataType> y(lengths);
        Tensor<YDataType> y_ref(lengths);

        x.GenerateTensorValue(GeneratorTensor_3<XDataType>{0, 1.0});
        gamma.GenerateTensorValue(GeneratorTensor_3<GammaDataType>{0.0, 1.0});
        beta.GenerateTensorValue(GeneratorTensor_3<BetaDataType>{0.0, 1.0});

        DeviceMem x_dev(sizeof(XDataType) * x.mDesc.GetElementSpaceSize());
        DeviceMem gamma_dev(sizeof(GammaDataType) * gamma.mDesc.GetElementSpaceSize());
        DeviceMem beta_dev(sizeof(BetaDataType) * beta.mDesc.GetElementSpaceSize());
        DeviceMem y_dev(sizeof(YDataType) * y.mDesc.GetElementSpaceSize());

        x_dev.ToDevice(x.mData.data());
        gamma_dev.ToDevice(gamma.mData.data());
        beta_dev.ToDevice(beta.mData.data());

        auto device_instance = DeviceInstance{};
        auto argument_ptr    = device_instance.MakeArgumentPointer(
            lengths,
            std::vector<ck::index_t>{x.mDesc.GetStrides().begin(), x.mDesc.GetStrides().end()},
            std::vector<ck::index_t>{gamma.mDesc.GetStrides().begin(),
                                     gamma.mDesc.GetStrides().end()},
            std::vector<ck::index_t>{beta.mDesc.GetStrides().begin(),
                                     beta.mDesc.GetStrides().end()},
            reduceDims,
            1e-4,
            x_dev.GetDeviceBuffer(),
            gamma_dev.GetDeviceBuffer(),
            beta_dev.GetDeviceBuffer(),
            y_dev.GetDeviceBuffer(),
            PassThrough{});

        if(!device_instance.IsSupportedArgument(argument_ptr.get()))
        {
            return;
        }

        auto invoker_ptr = device_instance.MakeInvokerPointer();
        invoker_ptr->Run(argument_ptr.get());

        ref_instance_invoker_.Run(
            {x, gamma, beta, y_ref, PassThrough{}, lengths, reduceDims, 1e-4});

        y_dev.FromDevice(y.mData.data());

        bool pass;

        if(std::is_same<XDataType, int8_t>::value)
        {
            EXPECT_TRUE(pass = ck::utils::check_err(
                            y.mData, y_ref.mData, "Error: Incorrect results!", 0, 1));
        }
        else
        {
            EXPECT_TRUE(pass = ck::utils::check_err(
                            y.mData, y_ref.mData, "Error: Incorrect results d1", 1e-3, 1e-3));
        }

        if(!pass)
        {
            FAIL() << "Failure in input lengths = [" << serialize_range(lengths) << "], "
                   << "reduce dim = [" << serialize_range(reduceDims) << "].";
        }
    }

    void Run()
    {
        for(auto length : this->lengths_)
        {
            this->RunSingle(length, reduceDims_[0]);
        }
    }

    std::vector<std::vector<index_t>> lengths_ = {
        {4, 256}, {8, 511}, {9, 1032}, {4, 2048}, {1, 8192}, {4000, 2000}};

    std::vector<std::vector<index_t>> reduceDims_ = {{1}};

    typename ReferenceInstance::Invoker ref_instance_invoker_;
};
} // namespace ck
