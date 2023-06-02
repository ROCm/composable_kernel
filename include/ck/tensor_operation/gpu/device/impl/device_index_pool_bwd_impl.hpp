// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"
#include "ck/tensor_operation/gpu/device/device_index_pool_bwd.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_put_element_1d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// output[indices] = input
template <typename DOutDataType,
          typename IndexDataType,
          typename DInDataType,
          ck::index_t InVectorSize>
struct DeviceIndexPoolBwdImpl : public DeviceIndexPoolBwd<DOutDataType, IndexDataType, DInDataType>
{
    static_assert(is_same_v<DInDataType, float> || is_same_v<DInDataType, double>,
                  "Data type is not supported!");

    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    template <typename Desc_M>
    static auto PadDescriptor_M_1d(Desc_M desc_m, index_t gridSize, index_t blockSize)
    {
        constexpr auto I0 = Number<0>{};

        const auto m            = desc_m.GetLength(I0);
        const index_t loop_step = gridSize * blockSize * InVectorSize;
        const auto pad          = math::integer_least_multiple(m, loop_step) - m;
        const auto desc_m_pad =
            transform_tensor_descriptor(desc_m,
                                        make_tuple(make_right_pad_transform(m, pad)),
                                        make_tuple(Sequence<0>{}),
                                        make_tuple(Sequence<0>{}));
        return desc_m_pad;
    }

    static auto MakeDescriptor_M(index_t length, index_t gridSize, index_t blockSize)
    {
        const auto desc_m = make_naive_tensor_descriptor_packed(make_tuple(length));
        return PadDescriptor_M_1d(desc_m, gridSize, blockSize);
    }

    using OutGrid1dDesc = decltype(MakeDescriptor_M(1, 1, 1));

    using GridwisePutElementSet = GridwisePutElement_1D<OutGrid1dDesc,
                                                        DOutDataType,
                                                        IndexDataType,
                                                        DInDataType,
                                                        PassThrough,
                                                        InMemoryDataOperationEnum::Set,
                                                        InVectorSize>;

    using GridwisePutElementAtomicAdd = GridwisePutElement_1D<OutGrid1dDesc,
                                                              DOutDataType,
                                                              IndexDataType,
                                                              DInDataType,
                                                              PassThrough,
                                                              InMemoryDataOperationEnum::AtomicAdd,
                                                              InVectorSize>;

    struct Argument : public BaseArgument
    {
        Argument(const DOutDataType* p_dout,
                 const IndexDataType* p_indices,
                 DInDataType* p_din,
                 index_t dout_length,
                 const std::vector<ck::index_t>& window_lengths,
                 const std::vector<ck::index_t>& window_strides)
            : p_dout_{p_dout},
              p_indices_{p_indices},
              p_din_{p_din},
              blockSize_{256},
              gridSize_{104}, // FIXME - Calculate the grid size by number of CU in the future
              windowOverlap_{false}
        {
            dout_grid_desc_ = MakeDescriptor_M(dout_length, gridSize_, blockSize_);

            for(size_t i = 0; i < window_lengths.size(); ++i)
            {
                windowOverlap_ |= window_lengths.at(i) > window_strides.at(i);
            }
        }

        const DOutDataType* p_dout_;
        const IndexDataType* p_indices_;
        DInDataType* p_din_;
        index_t blockSize_;
        index_t gridSize_;
        bool windowOverlap_;
        OutGrid1dDesc dout_grid_desc_;
    };

    struct Invoker : public BaseInvoker
    {
        constexpr auto KernelSelector(bool windowOverlap)
        {
            if(windowOverlap)
                return kernel_put_element_1d<GridwisePutElementAtomicAdd,
                                             OutGrid1dDesc,
                                             DOutDataType,
                                             IndexDataType,
                                             DInDataType,
                                             PassThrough>;
            else
                return kernel_put_element_1d<GridwisePutElementSet,
                                             OutGrid1dDesc,
                                             DOutDataType,
                                             IndexDataType,
                                             DInDataType,
                                             PassThrough>;
        }

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            const auto kernel = KernelSelector(arg.windowOverlap_);

            float elapsed_time = launch_and_time_kernel(stream_config,
                                                        kernel,
                                                        dim3(arg.gridSize_),
                                                        dim3(arg.blockSize_),
                                                        0,
                                                        arg.dout_grid_desc_,
                                                        arg.p_dout_,
                                                        arg.p_indices_,
                                                        arg.p_din_,
                                                        PassThrough{});

            return elapsed_time;
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        const Argument* pArg = dynamic_cast<const Argument*>(p_arg);
        // TODO
        ignore = pArg;
        return true;
    }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_dout,
                        const void* p_indices,
                        void* p_din,
                        index_t dout_length,
                        index_t,
                        std::vector<ck::index_t> window_lengths,
                        std::vector<ck::index_t> window_strides) override
    {
        return std::make_unique<Argument>(static_cast<const DOutDataType*>(p_dout),
                                          static_cast<const IndexDataType*>(p_indices),
                                          static_cast<DInDataType*>(p_din),
                                          dout_length,
                                          window_lengths,
                                          window_strides);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
