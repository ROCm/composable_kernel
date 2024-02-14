// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/math.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/tensor_operation/gpu/device/device_elementwise.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_elementwise_3d.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"

#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/stream_utility.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
template <typename InDataTypeTuple,
          typename OutDataTypeTuple,
          typename ElementwiseOperation,
          index_t NumDim_m, // choose how to set dims
          index_t NumDim_n,
          index_t NumDim_k,
          index_t MPerThread,
          index_t NPerThread,
          index_t KPerThread,
          typename InScalarPerVectorSeq,
          typename OutScalarPerVectorSeq>
struct DeviceElementwise3dImpl : public DeviceElementwise<InDataTypeTuple,
                                                          OutDataTypeTuple,
                                                          ElementwiseOperation,
                                                          NumDim_m + NumDim_n + NumDim_k>
{
    static constexpr index_t NumDim = NumDim_m + NumDim_n + NumDim_k;

    static constexpr int NumInput  = InDataTypeTuple::Size();
    static constexpr int NumOutput = OutDataTypeTuple::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};

    static_assert(NumInput == InScalarPerVectorSeq::Size() &&
                      NumOutput == OutScalarPerVectorSeq::Size(),
                  "Tuple size is inconsistent with the number of in/out!");

    static auto GenerateInDataTypePointerTuple()
    {
        return generate_tuple(
            [&](auto I) {
                using DataType = remove_cvref_t<decltype(InDataTypeTuple{}[I])>;

                return static_cast<const DataType*>(nullptr);
            },
            Number<NumInput>{});
    }

    static auto GenerateOutDataTypePointerTuple()
    {
        return generate_tuple(
            [&](auto I) {
                using DataType = remove_cvref_t<decltype(OutDataTypeTuple{}[I])>;

                return static_cast<DataType*>(nullptr);
            },
            Number<NumOutput>{});
    }

    using InDataTypePointerTuple  = decltype(GenerateInDataTypePointerTuple());
    using OutDataTypePointerTuple = decltype(GenerateOutDataTypePointerTuple());

    template <typename Desc_MNK>
    static auto PadDescriptor_MNK(Desc_MNK desc_mnk,
                                  index_t gridSize,
                                  index_t blockSize,
                                  index_t num_threads_m,
                                  index_t num_threads_n,
                                  index_t num_threads_k)
    {
        std::ignore = blockSize;
        std::ignore = gridSize;

        const auto m = desc_mnk.GetLength(I0);
        const auto n = desc_mnk.GetLength(I1);
        const auto k = desc_mnk.GetLength(I2);

        const index_t loop_step_m = num_threads_m * MPerThread;
        const index_t loop_step_n = num_threads_n * NPerThread;
        const index_t loop_step_k = num_threads_k * KPerThread;

        const auto pad_m = math::integer_least_multiple(m, loop_step_m) - m;
        const auto pad_n = math::integer_least_multiple(n, loop_step_n) - n;
        const auto pad_k = math::integer_least_multiple(k, loop_step_k) - k;

        const auto desc_mnk_pad =
            transform_tensor_descriptor(desc_mnk,
                                        make_tuple(make_right_pad_transform(m, pad_m),
                                                   make_right_pad_transform(n, pad_n),
                                                   make_right_pad_transform(k, pad_k)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
        return desc_mnk_pad;
    }

    static auto MakeDescriptor_MNK(const std::array<index_t, NumDim>& lengths,
                                   const std::array<index_t, NumDim>& stride,
                                   index_t gridSize,
                                   index_t blockSize,
                                   index_t num_threads_m,
                                   index_t num_threads_n,
                                   index_t num_threads_k)
    {
        auto tupleOfShape  = generate_tuple([&](auto I) { return lengths[I]; }, Number<NumDim>{});
        auto tupleOfStride = generate_tuple([&](auto I) { return stride[I]; }, Number<NumDim>{});

        // nd desc - [s0, s1, s2, ...]
        const auto desc = make_naive_tensor_descriptor(tupleOfShape, tupleOfStride);

        constexpr auto mDimIds = typename arithmetic_sequence_gen<0, NumDim_m, 1>::type();
        constexpr auto nDimIds =
            typename arithmetic_sequence_gen<NumDim_m, NumDim_m + NumDim_n, 1>::type();
        constexpr auto kDimIds =
            typename arithmetic_sequence_gen<NumDim_m + NumDim_n, NumDim, 1>::type();

        const auto mLengths = get_container_subset(tupleOfShape, mDimIds);
        const auto nLengths = get_container_subset(tupleOfShape, nDimIds);
        const auto kLengths = get_container_subset(tupleOfShape, kDimIds);

        // merge nd to 3d desc - [s0 * s1 * ...]
        if constexpr(NumDim > 3)
        {
            const auto desc_mnk = transform_tensor_descriptor(
                desc,
                make_tuple(make_merge_transform(mLengths),
                           make_merge_transform(nLengths),
                           make_merge_transform(kLengths)),
                make_tuple(mDimIds, nDimIds, kDimIds),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            return PadDescriptor_MNK(
                desc_mnk, gridSize, blockSize, num_threads_m, num_threads_n, num_threads_k);
        }
        else
            return PadDescriptor_MNK(
                desc, gridSize, blockSize, num_threads_m, num_threads_n, num_threads_k);
    }

    template <index_t TupleSize>
    static auto GenerateInOutGrid3dDescTuple(Number<TupleSize>)
    {
        return generate_tuple(
            [&](auto) {
                if constexpr(NumDim > 3)
                {
                    return MakeDescriptor_MNK({1, 1, 1}, {1, 1, 1}, 1, 1, 1, 1, 1);
                }
                else
                {
                    return MakeDescriptor_MNK({1}, {1}, 1, 1, 1, 1, 1);
                };
            },
            Number<TupleSize>{});
    }

    using OutGrid3dDescTuple = decltype(GenerateInOutGrid3dDescTuple(Number<NumOutput>{}));
    using InGrid3dDescTuple  = decltype(GenerateInOutGrid3dDescTuple(Number<NumInput>{}));

    using GridwiseElementwise = GridwiseElementwise_3D<InGrid3dDescTuple,
                                                       OutGrid3dDescTuple,
                                                       InDataTypePointerTuple,
                                                       OutDataTypePointerTuple,
                                                       ElementwiseOperation,
                                                       MPerThread,
                                                       NPerThread,
                                                       KPerThread,
                                                       InScalarPerVectorSeq,
                                                       OutScalarPerVectorSeq>;

    struct Argument : public BaseArgument
    {
        Argument(const std::array<index_t, NumDim> lengths,
                 const std::array<std::array<index_t, NumDim>, NumInput> inStridesArray,
                 const std::array<std::array<index_t, NumDim>, NumOutput> outStridesArray,
                 const std::array<const void*, NumInput> in_dev_buffers,
                 const std::array<void*, NumOutput> out_dev_buffers,
                 ElementwiseOperation elementwise_op)

            : lengths_(lengths),
              inStridesArray_(inStridesArray),
              outStridesArray_(outStridesArray),
              elementwise_op_(elementwise_op),
              blockSize_(256)
        {
            static_assert(NumDim_m > 0, "");
            static_assert(NumDim_n > 0, "");
            static_assert(NumDim_k > 0, "");

            in_dev_buffers_ = generate_tuple(
                [&](auto I) {
                    using DataType = remove_cvref_t<decltype(InDataTypeTuple{}[I])>;
                    return static_cast<const DataType*>(in_dev_buffers[I.value]);
                },
                Number<NumInput>{});

            out_dev_buffers_ = generate_tuple(
                [&](auto I) {
                    using DataType = remove_cvref_t<decltype(OutDataTypeTuple{}[I])>;
                    return static_cast<DataType*>(out_dev_buffers[I.value]);
                },
                Number<NumOutput>{});
        }

        InDataTypePointerTuple in_dev_buffers_;
        OutDataTypePointerTuple out_dev_buffers_;

        std::array<index_t, NumDim> lengths_;
        std::array<std::array<index_t, NumDim>, NumInput> inStridesArray_;
        std::array<std::array<index_t, NumDim>, NumOutput> outStridesArray_;

        ElementwiseOperation elementwise_op_;
        index_t blockSize_;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            index_t gridSize      = getAvailableComputeUnitCount(stream_config) * arg.blockSize_;
            index_t num_threads_m = gridSize / (16 * 16);
            index_t num_threads_n = 16;
            index_t num_threads_k = 16;

            auto in_grid_3d_desc_tuple = generate_tuple(
                [&](auto I) {
                    return MakeDescriptor_MNK(arg.lengths_,
                                              arg.inStridesArray_[I.value],
                                              gridSize,
                                              arg.blockSize_,
                                              num_threads_m,
                                              num_threads_n,
                                              num_threads_k);
                },
                Number<NumInput>{});

            auto out_grid_3d_desc_tuple = generate_tuple(
                [&](auto I) {
                    return MakeDescriptor_MNK(arg.lengths_,
                                              arg.outStridesArray_[I.value],
                                              gridSize,
                                              arg.blockSize_,
                                              num_threads_m,
                                              num_threads_n,
                                              num_threads_k);
                },
                Number<NumOutput>{});

            const auto kernel = kernel_elementwise_3d<GridwiseElementwise,
                                                      InGrid3dDescTuple,
                                                      OutGrid3dDescTuple,
                                                      InDataTypePointerTuple,
                                                      OutDataTypePointerTuple,
                                                      ElementwiseOperation>;

            float elapsed_time = launch_and_time_kernel(stream_config,
                                                        kernel,
                                                        dim3(gridSize),
                                                        dim3(arg.blockSize_),
                                                        0,
                                                        in_grid_3d_desc_tuple,
                                                        out_grid_3d_desc_tuple,
                                                        arg.in_dev_buffers_,
                                                        arg.out_dev_buffers_,
                                                        arg.elementwise_op_,
                                                        num_threads_m,
                                                        num_threads_n,
                                                        num_threads_k);
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
        if((ck::get_device_name() == "gfx940" || ck::get_device_name() == "gfx941" ||
            ck::get_device_name() == "gfx942" || ck::get_device_name() == "gfx950" ))
        {
            return false;
        }

        const Argument* pArg = dynamic_cast<const Argument*>(p_arg);

        if(pArg == nullptr)
            return false;

        if(pArg->lengths_.back() % MPerThread != 0)
            return false;

        auto IsScalarPerVectorValid = [&](const std::array<index_t, NumDim>& lengths,
                                          const std::array<index_t, NumDim>& strides,
                                          index_t scalarPerVector,
                                          index_t vectorDim) {
            if(strides[vectorDim] == 1 &&
               (lengths[vectorDim] % scalarPerVector == 0 ||
                lengths[vectorDim] % scalarPerVector == lengths[vectorDim]))
            {
                return true;
            }

            if(strides[vectorDim] >= scalarPerVector)
            {
                return true;
            }
            return false;
        };

        bool valid = true;
        static_for<0, NumInput, 1>{}([&](auto I) {
            valid = valid && IsScalarPerVectorValid(pArg->lengths_,
                                                    pArg->inStridesArray_[I.value],
                                                    InScalarPerVectorSeq::At(I),
                                                    NumDim_m - 1);
        });

        static_for<0, NumOutput, 1>{}([&](auto I) {
            valid = valid && IsScalarPerVectorValid(pArg->lengths_,
                                                    pArg->outStridesArray_[I.value],
                                                    OutScalarPerVectorSeq::At(I),
                                                    NumDim - 1);
        });

        return valid;
    }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::array<index_t, NumDim> lengths,
                        const std::array<std::array<index_t, NumDim>, NumInput> inStridesArray,
                        const std::array<std::array<index_t, NumDim>, NumOutput> outStridesArray,
                        const std::array<const void*, NumInput> in_dev_buffers,
                        const std::array<void*, NumOutput> out_dev_buffers,
                        ElementwiseOperation elementwise_op) override
    {
        return std::make_unique<Argument>(lengths,
                                          inStridesArray,
                                          outStridesArray,
                                          in_dev_buffers,
                                          out_dev_buffers,
                                          elementwise_op);
    }

    static auto MakeInvoker() { return Invoker{}; }
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    }
}; // namespace device

} // namespace device
} // namespace tensor_operation
} // namespace ck
