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
          int NumDim>
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
    bool IsSupported() const
    {
        std::vector<ck::index_t> dummy_dims(NumDim * 2, 4);
        std::vector<ck::index_t> dummy_strides(NumDim * 2, 1);

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
                                            dummy_dims,
                                            dummy_strides,
                                            dummy_dims,
                                            dummy_strides,
                                            std::array<std::vector<ck::index_t>, 1>{dummy_dims},
                                            std::array<std::vector<ck::index_t>, 1>{dummy_strides},
                                            dummy_dims,
                                            dummy_strides,
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
    ContractionDeviceWrapper<F32, F32, F32, F32, 1> wrapper_1d;
    ContractionDeviceWrapper<F32, F32, F32, F32, 2> wrapper_2d;
    ContractionDeviceWrapper<F32, F32, F32, F32, 3> wrapper_3d;
    EXPECT_FALSE(wrapper_1d.IsSupported());
    EXPECT_TRUE(wrapper_2d.IsSupported());
    EXPECT_FALSE(wrapper_3d.IsSupported());
}

TEST(TestContractionInterface, IncorrectDataTypes)
{
    ContractionDeviceWrapper<F32, F32, F64, F64, 2> wrapper_1;
    ContractionDeviceWrapper<F64, F64, F32, F32, 2> wrapper_2;
    EXPECT_FALSE(wrapper_1.IsSupported());
    EXPECT_FALSE(wrapper_2.IsSupported());
}

// TEST(TestContractionInterface, CornerCases)
// {
//     EXPECT_FALSE()
// }
