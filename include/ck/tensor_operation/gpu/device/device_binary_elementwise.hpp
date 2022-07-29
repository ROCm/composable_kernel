// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <vector>

#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/tensor_operation/gpu/device/device_elementwise.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_binary_elementwise_1d.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ComputeDataType,
          typename ElementwiseFunctor,
          index_t NDim,
          index_t MPerThread,
          index_t AScalarPerVector,
          index_t BScalarPerVector,
          index_t CScalarPerVector>
struct DeviceBinaryElementwise : public DeviceElementwise<2, 1, NDim, ElementwiseFunctor>
{
    static constexpr auto I0 = Number<0>{};

    template <typename Desc_M>
    static auto PadDescriptor_M_1d(Desc_M desc_m, index_t gridSize, index_t blockSize)
    {
        const auto M            = desc_m.GetLength(I0);
        const index_t loop_step = gridSize * blockSize * MPerThread;
        const auto pad          = math::integer_least_multiple(M, loop_step) - M;
        const auto desc_m_pad =
            transform_tensor_descriptor(desc_m,
                                        make_tuple(make_right_pad_transform(M, pad)),
                                        make_tuple(Sequence<0>{}),
                                        make_tuple(Sequence<0>{}));
        return desc_m_pad;
    }

    static auto MakeDescriptor_M(const std::vector<index_t>& lengths,
                                 const std::vector<index_t>& strides,
                                 index_t gridSize,
                                 index_t blockSize)
    {
        auto tupleOfShape  = generate_tuple([&](auto I) { return lengths[I]; }, Number<NDim>{});
        auto tupleOfStride = generate_tuple([&](auto I) { return strides[I]; }, Number<NDim>{});

        // nd desc - [s0, s1, s2, ...]
        const auto desc = make_naive_tensor_descriptor(tupleOfShape, tupleOfStride);

        // merge nd to 1d desc - [s0 * s1 * ...]
        if constexpr(NDim > 1)
        {
            const auto desc_m = transform_tensor_descriptor(
                desc,
                make_tuple(make_merge_transform(tupleOfShape)),
                make_tuple(generate_sequence_v2([&](auto I) { return I; }, Number<NDim>{})),
                make_tuple(Sequence<0>{}));

            return PadDescriptor_M_1d(desc_m, gridSize, blockSize);
        }
        else
            return PadDescriptor_M_1d(desc, gridSize, blockSize);
    }

    using AGridDesc_M        = decltype(MakeDescriptor_M({1, 1}, {1, 1}, 1, 1));
    using BGridDesc_M        = decltype(MakeDescriptor_M({1, 1}, {1, 1}, 1, 1));
    using CGridDesc_M        = decltype(MakeDescriptor_M({1, 1}, {1, 1}, 1, 1));
    using GridwiseBinEltwise = GridwiseBinaryElementwise_1D<ADataType,
                                                            BDataType,
                                                            CDataType,
                                                            ComputeDataType,
                                                            AGridDesc_M,
                                                            BGridDesc_M,
                                                            CGridDesc_M,
                                                            ElementwiseFunctor,
                                                            MPerThread,
                                                            AScalarPerVector,
                                                            BScalarPerVector,
                                                            CScalarPerVector>;

    struct Argument : public BaseArgument
    {
        Argument(const ADataType* p_a,
                 const BDataType* p_b,
                 CDataType* p_c,
                 const std::vector<index_t>& lengths,
                 const std::vector<index_t>& a_strides,
                 const std::vector<index_t>& b_strides,
                 const std::vector<index_t>& c_strides,
                 ElementwiseFunctor functor)
            : p_a_(p_a),
              p_b_(p_b),
              p_c_(p_c),
              lengths_(lengths),
              a_strides_(a_strides),
              b_strides_(b_strides),
              c_strides_(c_strides),
              functor_(functor),
              blockSize_(256),
              gridSize_(120) // FIXME - Calculate the grid size by number of CU in the future
        {
            a_grid_desc_m_ = MakeDescriptor_M(lengths, a_strides, gridSize_, blockSize_);
            b_grid_desc_m_ = MakeDescriptor_M(lengths, b_strides, gridSize_, blockSize_);
            c_grid_desc_m_ = MakeDescriptor_M(lengths, c_strides, gridSize_, blockSize_);
        }

        const ADataType* p_a_;
        const BDataType* p_b_;
        CDataType* p_c_;
        std::vector<int> lengths_;
        AGridDesc_M a_grid_desc_m_;
        BGridDesc_M b_grid_desc_m_;
        CGridDesc_M c_grid_desc_m_;
        std::vector<index_t> a_strides_;
        std::vector<index_t> b_strides_;
        std::vector<index_t> c_strides_;
        ElementwiseFunctor functor_;
        index_t blockSize_;
        index_t gridSize_;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            const auto kernel = kernel_binary_elementwise_1d<GridwiseBinEltwise,
                                                             ADataType,
                                                             BDataType,
                                                             CDataType,
                                                             AGridDesc_M,
                                                             BGridDesc_M,
                                                             CGridDesc_M,
                                                             ElementwiseFunctor>;

            float elapsed_time = launch_and_time_kernel(stream_config,
                                                        kernel,
                                                        dim3(arg.gridSize_),
                                                        dim3(arg.blockSize_),
                                                        0,
                                                        arg.p_a_,
                                                        arg.p_b_,
                                                        arg.p_c_,
                                                        arg.a_grid_desc_m_,
                                                        arg.b_grid_desc_m_,
                                                        arg.c_grid_desc_m_,
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

        if(pArg->lengths_.size() != NDim)
            return false;

        if(pArg->lengths_.back() % MPerThread != 0)
            return false;

        auto IsScalarPerVectorValid = [](bool isLastDimensionCoalesced, int scalarPerVector) {
            bool ret = true;

            if(!isLastDimensionCoalesced)
                ret = scalarPerVector == 1;
            else
                ret = MPerThread % scalarPerVector == 0;

            return ret;
        };

        if(!IsScalarPerVectorValid(pArg->a_strides_.back() == 1, AScalarPerVector))
            return false;

        if(!IsScalarPerVectorValid(pArg->b_strides_.back() == 1, BScalarPerVector))
            return false;

        if(!IsScalarPerVectorValid(pArg->c_strides_.back() == 1, CScalarPerVector))
            return false;

        return true;
    };

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::array<const void*, 2> p_inputs,
                        std::array<void*, 1> p_outputs,
                        std::vector<index_t> lengths,
                        std::vector<std::vector<index_t>> input_strides,
                        std::vector<std::vector<index_t>> output_strides,
                        ElementwiseFunctor functor) override
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_inputs[0]),
                                          static_cast<const BDataType*>(p_inputs[1]),
                                          static_cast<CDataType*>(p_outputs[0]),
                                          lengths,
                                          input_strides[0],
                                          input_strides[1],
                                          output_strides[0],
                                          functor);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceBinaryElementwise"
            << "<"
            << "NDim = " << NDim
            << "MPerThread = " << MPerThread
            << "AScalarPerVector = " << AScalarPerVector
            << "BScalarPerVector = " << BScalarPerVector
            << "CScalarPerVector = " << CScalarPerVector
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
