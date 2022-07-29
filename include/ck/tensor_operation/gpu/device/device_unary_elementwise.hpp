// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <vector>

#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_unary_elementwise_1d.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ADataType,
          typename BDataType,
          typename ElementwiseFunctor,
          index_t Dim,
          index_t ScalarPerVector>
struct DeviceUnaryElementwise : public BaseOperator
{
    static constexpr auto I0 = Number<0>{};

    template <typename Desc_M0>
    static auto PadDescriptor_M0_1d(Desc_M0 desc_m0, index_t gridSize, index_t blockSize)
    {
        const auto m0           = desc_m0.GetLength(I0);
        const index_t loop_step = gridSize * blockSize * ScalarPerVector;
        const auto pad          = math::integer_least_multiple(m0, loop_step) - m0;
        const auto desc_m0_pad =
            transform_tensor_descriptor(desc_m0,
                                        make_tuple(make_right_pad_transform(m0, pad)),
                                        make_tuple(Sequence<0>{}),
                                        make_tuple(Sequence<0>{}));
        return desc_m0_pad;
    }

    static auto MakeDescriptor_M0(const std::vector<index_t>& shape,
                                  const std::vector<index_t>& stride,
                                  index_t gridSize,
                                  index_t blockSize)
    {
        auto tupleOfShape  = generate_tuple([&](auto I) { return shape[I]; }, Number<Dim>{});
        auto tupleOfStride = generate_tuple([&](auto I) { return stride[I]; }, Number<Dim>{});

        // nd desc - [s0, s1, s2, ...]
        const auto desc = make_naive_tensor_descriptor(tupleOfShape, tupleOfStride);

        // merge nd to 1d desc - [s0 * s1 * ...]
        if constexpr(Dim > 1)
        {
            const auto desc_m0 = transform_tensor_descriptor(
                desc,
                make_tuple(make_merge_transform(tupleOfShape)),
                make_tuple(generate_sequence_v2([&](auto I) { return I; }, Number<Dim>{})),
                make_tuple(Sequence<0>{}));

            return PadDescriptor_M0_1d(desc_m0, gridSize, blockSize);
        }
        else
            return PadDescriptor_M0_1d(desc, gridSize, blockSize);
    }

    using GridDesc_M0      = decltype(MakeDescriptor_M0({1, 1}, {1, 1}, 1, 1));
    using GridwiseUEltwise = GridwiseUnaryElementwise_1D<ADataType,
                                                         BDataType,
                                                         GridDesc_M0,
                                                         ElementwiseFunctor,
                                                         ScalarPerVector>;

    struct Argument : public BaseArgument
    {
        Argument(const ADataType* p_a,
                 BDataType* p_b,
                 const std::vector<index_t>& shape,
                 const std::vector<index_t>& stride_a,
                 const std::vector<index_t>& stride_b,
                 ElementwiseFunctor functor)
            : p_a_(p_a),
              p_b_(p_b),
              shape_(shape),
              functor_(functor),
              blockSize_(256) // FIXME - Calculate the grid size by number of CU in the future
        {
            index_t tensor_size =
                std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>{});
            gridSize_       = GridwiseUEltwise::CalculateGridSize(tensor_size);
            a_grid_desc_m0_ = MakeDescriptor_M0(shape, stride_a, gridSize_, blockSize_);
            b_grid_desc_m0_ = MakeDescriptor_M0(shape, stride_b, gridSize_, blockSize_);
        }

        const ADataType* p_a_;
        BDataType* p_b_;
        std::vector<int> shape_;
        GridDesc_M0 a_grid_desc_m0_;
        GridDesc_M0 b_grid_desc_m0_;
        ElementwiseFunctor functor_;
        index_t blockSize_;
        index_t gridSize_;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            const auto kernel = kernel_unary_elementwise_1d<GridwiseUEltwise,
                                                            ADataType,
                                                            BDataType,
                                                            GridDesc_M0,
                                                            ElementwiseFunctor>;

            float elapsed_time = launch_and_time_kernel(stream_config,
                                                        kernel,
                                                        dim3(arg.gridSize_),
                                                        dim3(arg.blockSize_),
                                                        0,
                                                        arg.p_a_,
                                                        arg.p_b_,
                                                        arg.a_grid_desc_m0_,
                                                        arg.b_grid_desc_m0_,
                                                        arg.functor_);
            return elapsed_time;
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        const Argument* pArg = dynamic_cast<const Argument*>(p_arg);

        if(pArg == nullptr)
            return false;

        if(pArg->shape_.back() % ScalarPerVector != 0)
            return false;

        return true;
    };

    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      void* p_b,
                                                      std::vector<index_t> shape,
                                                      std::vector<index_t> stride_a,
                                                      std::vector<index_t> stride_b,
                                                      ElementwiseFunctor functor)
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<BDataType*>(p_b),
                                          shape,
                                          stride_a,
                                          stride_b,
                                          functor);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() { return std::make_unique<Invoker>(); }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceBinaryElementwise"
            << "<"
            << "ScalarPerVector = " << ScalarPerVector
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
