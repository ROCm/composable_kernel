// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <array>
#include <memory>
#include <utility>

#include "ck/utility/math.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_permute.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"

#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

namespace detail {
template <typename Derived>
struct DevicePermuteBase : BaseOperator
{
    bool IsSupportedArgument(const BaseArgument* arg) override final
    {
        const auto* argument = dynamic_cast<const typename Derived::Argument*>(arg);
        if(!argument)
        {
            return false;
        }

        return Derived::IsSupportedArgument(*argument);
    }

    template <typename... Args>
    static auto MakeArgument(Args&&... args)
    {
        return typename Derived::Argument{std::forward<Args>(args)...};
    }

    template <typename... Args>
    static auto MakeArgumentPointer(Args&&... args)
    {
        return std::make_unique<typename Derived::Argument>(std::forward<Args>(args)...);
    }

    static auto MakeInvoker() { return typename Derived::Invoker{}; }

    static auto MakeInvokerPointer() { return std::make_unique<typename Derived::Invoker>(); };
};

template <typename Derived, typename Argument>
struct InvokerBase : BaseInvoker
{
    float Run(const BaseArgument* arg,
              const StreamConfig& stream_config = StreamConfig{}) override final
    {
        const auto* argument = dynamic_cast<const Argument*>(arg);
        if(!argument)
        {
            return 0.f;
        }

        return Derived::Run(*argument, stream_config);
    }
};
} // namespace detail

// Swap last 2 dimensions
// input shape: [d[0], d[1], d[2], ..., d[NumDim-3], d[NumDim-2], d[NumDim-1]]
//                                                                ^^^^^^^^^^^
// output shape: [d[0], d[1], d[2], ..., d[NumDim-3], d[NumDim-1], d[NumDim-2]]
//                                                    ^^^^^^^^^^^
template <typename InDataType,
          typename OutDataType,
          typename ElementwiseOperation,
          index_t NumDim,
          index_t MPerThread,
          index_t InScalarPerVector,
          index_t OutScalarPerVector>
struct DevicePermute : detail::DevicePermuteBase<DevicePermute<InDataType,
                                                               OutDataType,
                                                               ElementwiseOperation,
                                                               NumDim,
                                                               MPerThread,
                                                               InScalarPerVector,
                                                               OutScalarPerVector>>
{
    static_assert(3 <= NumDim, "Only accept at least 3D dimension tensor");

    using InDataTypePointer  = const InDataType*;
    using OutDataTypePointer = OutDataType*;

    template <typename Desc_M>
    static auto PadDescriptor_M_1d(Desc_M desc_m, index_t gridSize, index_t blockSize)
    {
        constexpr auto I0 = Number<0>{};

        const auto m            = desc_m.GetLength(I0);
        const index_t loop_step = gridSize * blockSize * MPerThread;
        const auto pad          = math::integer_least_multiple(m, loop_step) - m;
        const auto desc_m_pad =
            transform_tensor_descriptor(desc_m,
                                        make_tuple(make_right_pad_transform(m, pad)),
                                        make_tuple(Sequence<0>{}),
                                        make_tuple(Sequence<0>{}));
        return desc_m_pad;
    }

    static auto MakeDescriptor_M(const std::array<index_t, NumDim>& lengths,
                                 const std::array<index_t, NumDim>& stride,
                                 index_t gridSize,
                                 index_t blockSize)
    {
        auto tupleOfShape  = generate_tuple([&](auto I) { return lengths[I]; }, Number<NumDim>{});
        auto tupleOfStride = generate_tuple([&](auto I) { return stride[I]; }, Number<NumDim>{});

        // nd desc - [s0, s1, s2, ...]
        const auto desc = make_naive_tensor_descriptor(tupleOfShape, tupleOfStride);

        // merge nd to 1d desc - [s0 * s1 * ...]
        const auto desc_m = transform_tensor_descriptor(
            desc,
            make_tuple(make_merge_transform(tupleOfShape)),
            make_tuple(generate_sequence_v2([&](auto I) { return I; }, Number<NumDim>{})),
            make_tuple(Sequence<0>{}));

        return PadDescriptor_M_1d(desc_m, gridSize, blockSize);
    }

    using InGrid1dDesc  = decltype(MakeDescriptor_M({1, 1}, {1, 1}, 1, 1));
    using OutGrid1dDesc = decltype(MakeDescriptor_M({1, 1}, {1, 1}, 1, 1));

    using GridwisePermute = GridwisePermute<InGrid1dDesc,
                                            OutGrid1dDesc,
                                            InDataTypePointer,
                                            OutDataTypePointer,
                                            ElementwiseOperation,
                                            MPerThread,
                                            InScalarPerVector,
                                            OutScalarPerVector>;

    struct Argument : public BaseArgument
    {
        Argument(const std::array<index_t, NumDim> inLengths,
                 const std::array<index_t, NumDim> inStrides,
                 const std::array<index_t, NumDim> outLengths,
                 const std::array<index_t, NumDim> outStrides,
                 const void* in_dev_buffer,
                 void* out_dev_buffer,
                 ElementwiseOperation elementwise_op)
            : blockSize_(256),
              gridSize_(120), // FIXME - Calculate the grid size by number of CU in the future
              in_dev_buffer_(static_cast<InDataTypePointer>(in_dev_buffer)),
              out_dev_buffer_(static_cast<OutDataTypePointer>(out_dev_buffer)),
              in_grid_1d_desc_(MakeDescriptor_M(inLengths, inStrides, gridSize_, blockSize_)),
              out_grid_1d_desc_(MakeDescriptor_M(inLengths, inStrides, gridSize_, blockSize_)),
              inLengths_(inLengths),
              inStrides_(inStrides),
              outLengths_(outLengths),
              outStrides_(outStrides),
              elementwise_op_(elementwise_op)
        {
        }

        index_t blockSize_;
        index_t gridSize_;

        InDataTypePointer in_dev_buffer_;
        OutDataTypePointer out_dev_buffer_;
        InGrid1dDesc in_grid_1d_desc_;
        OutGrid1dDesc out_grid_1d_desc_;

        std::array<index_t, NumDim> inLengths_;
        std::array<index_t, NumDim> inStrides_;
        std::array<index_t, NumDim> outLengths_;
        std::array<index_t, NumDim> outStrides_;

        ElementwiseOperation elementwise_op_;
    };

    struct Invoker : detail::InvokerBase<Invoker, Argument>
    {
        static float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            const auto kernel = kernel_permute<GridwisePermute,
                                               InGrid1dDesc,
                                               OutGrid1dDesc,
                                               InDataTypePointer,
                                               OutDataTypePointer,
                                               ElementwiseOperation>;

            float elapsed_time = launch_and_time_kernel(stream_config,
                                                        kernel,
                                                        dim3(arg.gridSize_),
                                                        dim3(arg.blockSize_),
                                                        0,
                                                        arg.in_grid_1d_desc_,
                                                        arg.out_grid_1d_desc_,
                                                        arg.in_dev_buffer_,
                                                        arg.out_dev_buffer_,
                                                        arg.elementwise_op_);
            return elapsed_time;
        }
    };

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(arg.inLengths_.back() % MPerThread != 0)
        {
            return false;
        }

        // check if only swap last 2 dimensions
        if(!(std::equal(begin(arg.inLengths_),
                        std::prev(end(arg.inLengths_), 2),
                        begin(arg.outLengths_)) &&
             std::tie(*rbegin(arg.inLengths_), *std::next(rbegin(arg.inLengths_))) ==
                 std::tie(*std::next(rbegin(arg.outLengths_)), *rbegin(arg.outLengths_))))
        {
            return false;
        }

        auto IsScalarPerVectorValid = [&](const std::array<index_t, NumDim>& lengths,
                                          const std::array<index_t, NumDim>& strides,
                                          index_t scalarPerVector) {
            if(strides.back() == 1 && lengths.back() % scalarPerVector == 0)
                return true;

            if(strides.back() != 1 && scalarPerVector == 1)
                return true;

            return false;
        };

        bool valid = true;
        if(!IsScalarPerVectorValid(arg.inLengths_, arg.inStrides_, InScalarPerVector))
        {
            valid = false;
        }

        if(!IsScalarPerVectorValid(arg.outLengths_, arg.outStrides_, OutScalarPerVector))
        {
            valid = false;
        }

        return valid;
    };
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
