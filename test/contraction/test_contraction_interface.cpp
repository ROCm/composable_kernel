// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"

#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/device_contraction_multiple_d.hpp"

#include "ck/library/tensor_operation_instance/gpu/contraction_bilinear.hpp"

#include "ck/library/utility/device_memory.hpp"

using Pass     = ck::tensor_operation::element_wise::PassThrough;
using Bilinear = ck::tensor_operation::element_wise::Bilinear;

using F32 = float;
using F64 = double;

template <typename DataTypeA,
          typename DataTypeB,
          typename DataTypeC,
          typename DataTypeD,
          ck::index_t NumDim>
class ContractionDeviceWrapper
{

    protected:
    using DeviceOp = ck::tensor_operation::device::DeviceContractionMultipleD<NumDim,
                                                                              NumDim,
                                                                              NumDim,
                                                                              DataTypeA,
                                                                              DataTypeB,
                                                                              ck::Tuple<DataTypeC>,
                                                                              DataTypeD,
                                                                              Pass,
                                                                              Pass,
                                                                              Bilinear>;

    public:
    ContractionDeviceWrapper(std::vector<ck::index_t>& Dims, std::vector<ck::index_t>& Strides)
        : InputDims_(Dims), OutputDims_(Dims), InputStrides_(Strides), OutputStrides_(Strides)
    {
    }
    ContractionDeviceWrapper(std::vector<ck::index_t>& InDims,
                             std::vector<ck::index_t>& OutDims,
                             std::vector<ck::index_t>& InStrides,
                             std::vector<ck::index_t>& OutStrides)
        : InputDims_(InDims),
          OutputDims_(OutDims),
          InputStrides_(InStrides),
          OutputStrides_(OutStrides)
    {
    }

    std::vector<ck::index_t>& InputDims_;
    std::vector<ck::index_t>& OutputDims_;
    std::vector<ck::index_t>& InputStrides_;
    std::vector<ck::index_t>& OutputStrides_;
    bool IsSupported() const
    {

        bool supported     = false;
        const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
            DeviceOp>::GetInstances();

        for(auto& op_ptr : op_ptrs)
        {
            auto argument_ptr =
                op_ptr->MakeArgumentPointer(nullptr,
                                            nullptr,
                                            std::array<const void*, 1>{nullptr},
                                            nullptr,
                                            InputStrides_,
                                            InputStrides_,
                                            InputStrides_,
                                            InputStrides_,
                                            std::array<std::vector<ck::index_t>, 1>{InputStrides_},
                                            std::array<std::vector<ck::index_t>, 1>{InputStrides_},
                                            OutputDims_,
                                            OutputStrides_,
                                            Pass{},
                                            Pass{},
                                            Bilinear{1.f, 1.f});

            supported = supported || op_ptr->IsSupportedArgument(argument_ptr.get());
        }
        return supported;
    }
};

TEST(TestContractionInterface, IncorrectNumDims)
{
    std::vector<std::vector<ck::index_t>> Dims    = {{4, 4}, {4, 4, 4, 4}, {4, 4, 4, 4, 4, 4}};
    std::vector<std::vector<ck::index_t>> Strides = {{1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1, 1, 1}};
    ContractionDeviceWrapper<F32, F32, F32, F32, 1> wrapper_1d(Dims[0], Strides[0]);
    ContractionDeviceWrapper<F32, F32, F32, F32, 2> wrapper_2d(Dims[1], Strides[1]);
    ContractionDeviceWrapper<F32, F32, F32, F32, 3> wrapper_3d(Dims[2], Strides[2]);
    EXPECT_FALSE(wrapper_1d.IsSupported());
    EXPECT_TRUE(wrapper_2d.IsSupported());
    EXPECT_FALSE(wrapper_3d.IsSupported());
}

TEST(TestContractionInterface, IncorrectDataTypes)
{
    std::vector<ck::index_t> Dims    = {4, 4, 4, 4};
    std::vector<ck::index_t> Strides = {64, 16, 4, 1};
    ContractionDeviceWrapper<F32, F32, F64, F64, 2> wrapper_1(Dims, Strides);
    ContractionDeviceWrapper<F64, F64, F32, F32, 2> wrapper_2(Dims, Strides);
    EXPECT_FALSE(wrapper_1.IsSupported());
    EXPECT_FALSE(wrapper_2.IsSupported());
}

TEST(TestContractionInterface, GridwiseGemm)
{
    std::vector<ck::index_t> InDims     = {1, 2, 3, 4};
    std::vector<ck::index_t> InStrides  = {24, 12, 4, 1};
    std::vector<ck::index_t> OutDims    = {4, 3, 2, 1};
    std::vector<ck::index_t> OutStrides = {6, 2, 1, 1};
    ContractionDeviceWrapper<F32, F32, F32, F32, 2> wrapper(InDims, OutDims, InStrides, OutStrides);

    EXPECT_FALSE(wrapper.IsSupported());
}

TEST(TestContractionInterface, MemoryAccess)
{
    std::vector<ck::index_t> Dims    = {4, 4, 4, 4};
    std::vector<ck::index_t> Strides = {4, 16, 64, 256};
    ContractionDeviceWrapper<F32, F32, F32, F32, 2> wrapper(Dims, Strides);

    EXPECT_FALSE(wrapper.IsSupported());
}
