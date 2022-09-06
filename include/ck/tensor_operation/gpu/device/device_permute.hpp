// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <array>
#include <memory>
#include <utility>

#include "ck/utility/math.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_elementwise_1d.hpp"
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
            return false;
        }

        return Derived::Run(*argument, stream_config);
    }
};
} // namespace detail

template <typename InDataType,
          typename OutDataType,
          typename ElementwiseOperation,
          index_t NumDim,
          index_t MPerThread,
          typename InScalarPerVectorSeq,
          typename OutScalarPerVectorSeq>
struct DevicePermute : detail::DevicePermuteBase<DevicePermute<InDataType,
                                                               OutDataType,
                                                               ElementwiseOperation,
                                                               NumDim,
                                                               MPerThread,
                                                               InScalarPerVectorSeq,
                                                               OutScalarPerVectorSeq>>
{
    static constexpr int NumInput  = 1;
    static constexpr int NumOutput = 1;

    static_assert(NumInput == InScalarPerVectorSeq::Size() &&
                      NumOutput == OutScalarPerVectorSeq::Size(),
                  "Tuple size is inconsistent with the number of in/out!");

    using InDataTypePointerTuple  = Tuple<const InDataType*>;
    using OutDataTypePointerTuple = Tuple<OutDataType*>;

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
        if constexpr(NumDim > 1)
        {
            const auto desc_m = transform_tensor_descriptor(
                desc,
                make_tuple(make_merge_transform(tupleOfShape)),
                make_tuple(generate_sequence_v2([&](auto I) { return I; }, Number<NumDim>{})),
                make_tuple(Sequence<0>{}));

            return PadDescriptor_M_1d(desc_m, gridSize, blockSize);
        }
        else
            return PadDescriptor_M_1d(desc, gridSize, blockSize);
    }

    static auto GenerateInOutGrid1dDesc()
    {
        if constexpr(NumDim > 1)
        {
            return MakeDescriptor_M({1, 1}, {1, 1}, 1, 1);
        }
        else
        {
            return MakeDescriptor_M({1}, {1}, 1, 1);
        };
    };

    using InGrid1dDescTuple  = Tuple<decltype(GenerateInOutGrid1dDesc())>;
    using OutGrid1dDescTuple = Tuple<decltype(GenerateInOutGrid1dDesc())>;

    using GridwiseElementwise = GridwiseElementwise_1D<InGrid1dDescTuple,
                                                       OutGrid1dDescTuple,
                                                       InDataTypePointerTuple,
                                                       OutDataTypePointerTuple,
                                                       ElementwiseOperation,
                                                       MPerThread,
                                                       InScalarPerVectorSeq,
                                                       OutScalarPerVectorSeq>;

    struct Argument : public BaseArgument
    {
        Argument(const std::array<index_t, NumDim> lengths,
                 const std::array<index_t, NumDim> inStrides,
                 const std::array<index_t, NumDim> outStrides,
                 const void* in_dev_buffer,
                 void* out_dev_buffer,
                 ElementwiseOperation elementwise_op)
            : blockSize_(256),
              gridSize_(120), // FIXME - Calculate the grid size by number of CU in the future
              lengths_(lengths),
              inStridesArray_({inStrides}),
              outStridesArray_({outStrides}),
              elementwise_op_(elementwise_op)
        {
            in_dev_buffers_ = generate_tuple(
                [&](auto) {
                    using DataType = InDataType;
                    return static_cast<const DataType*>(in_dev_buffer);
                },
                Number<NumInput>{});

            out_dev_buffers_ = generate_tuple(
                [&](auto) {
                    using DataType = OutDataType;
                    return static_cast<DataType*>(out_dev_buffer);
                },
                Number<NumOutput>{});

            in_grid_1d_desc_tuple_ = generate_tuple(
                [&](auto) { return MakeDescriptor_M(lengths, inStrides, gridSize_, blockSize_); },
                Number<NumInput>{});

            out_grid_1d_desc_tuple_ = generate_tuple(
                [&](auto) { return MakeDescriptor_M(lengths, outStrides, gridSize_, blockSize_); },
                Number<NumOutput>{});
        }

        index_t blockSize_;
        index_t gridSize_;

        InDataTypePointerTuple in_dev_buffers_;
        OutDataTypePointerTuple out_dev_buffers_;
        InGrid1dDescTuple in_grid_1d_desc_tuple_;
        OutGrid1dDescTuple out_grid_1d_desc_tuple_;

        std::array<index_t, NumDim> lengths_;
        std::array<std::array<index_t, NumDim>, NumInput> inStridesArray_;
        std::array<std::array<index_t, NumDim>, NumOutput> outStridesArray_;

        ElementwiseOperation elementwise_op_;
    };

    struct Invoker : detail::InvokerBase<Invoker, Argument>
    {
        static float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            const auto kernel = kernel_elementwise_1d<GridwiseElementwise,
                                                      InGrid1dDescTuple,
                                                      OutGrid1dDescTuple,
                                                      InDataTypePointerTuple,
                                                      OutDataTypePointerTuple,
                                                      ElementwiseOperation>;

            float elapsed_time = launch_and_time_kernel(stream_config,
                                                        kernel,
                                                        dim3(arg.gridSize_),
                                                        dim3(arg.blockSize_),
                                                        0,
                                                        arg.in_grid_1d_desc_tuple_,
                                                        arg.out_grid_1d_desc_tuple_,
                                                        arg.in_dev_buffers_,
                                                        arg.out_dev_buffers_,
                                                        arg.elementwise_op_);
            return elapsed_time;
        }
    };

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(arg.lengths_.back() % MPerThread != 0)
            return false;

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
        static_for<0, NumInput, 1>{}([&](auto I) {
            if(!IsScalarPerVectorValid(
                   arg.lengths_, arg.inStridesArray_[I.value], InScalarPerVectorSeq::At(I)))
                valid = false;
        });

        static_for<0, NumOutput, 1>{}([&](auto I) {
            if(!IsScalarPerVectorValid(
                   arg.lengths_, arg.outStridesArray_[I.value], OutScalarPerVectorSeq::At(I)))
                valid = false;
        });

        return valid;
    };
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
