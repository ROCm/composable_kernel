// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/device_index_pool_bwd.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_put_element_1d.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_elementwise_1d.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/stream_utility.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// output[indices] = input
template <typename DOutDataType,
          typename IndexDataType,
          typename DInDataType,
          ck::index_t InOutVectorSize>
struct DeviceIndexPoolBwdImpl : public DeviceIndexPoolBwd<DOutDataType, IndexDataType, DInDataType>
{
    using DInDataType_AutomicAddPreCast =
        conditional_t<is_same_v<DInDataType, float> || is_same_v<DInDataType, double>,
                      DInDataType,
                      float>;

    using PassThrough  = ck::tensor_operation::element_wise::PassThrough;
    using UnaryConvert = ck::tensor_operation::element_wise::UnaryConvert;

    static constexpr auto I0 = Number<0>{};

    template <typename Desc_M>
    static auto PadDescriptor_M_1d(Desc_M desc_m, index_t loop_step)
    {
        const auto m   = desc_m.GetLength(I0);
        const auto pad = math::integer_least_multiple(m, loop_step) - m;
        const auto desc_m_pad =
            transform_tensor_descriptor(desc_m,
                                        make_tuple(make_right_pad_transform(m, pad)),
                                        make_tuple(Sequence<0>{}),
                                        make_tuple(Sequence<0>{}));
        return desc_m_pad;
    }

    static auto MakeDescriptor_M(index_t length, index_t loop_step)
    {
        const auto desc_m = make_naive_tensor_descriptor_packed(make_tuple(length));
        return PadDescriptor_M_1d(desc_m, loop_step);
    }

    using InOutGrid1dDesc = decltype(MakeDescriptor_M(1, 1));

    using GridwisePutElementSet = GridwisePutElement_1D<InOutGrid1dDesc,
                                                        DOutDataType,
                                                        IndexDataType,
                                                        DInDataType,
                                                        PassThrough,
                                                        InMemoryDataOperationEnum::Set,
                                                        InOutVectorSize>;

    using GridwisePutElementAtomicAdd = GridwisePutElement_1D<InOutGrid1dDesc,
                                                              DOutDataType,
                                                              IndexDataType,
                                                              DInDataType_AutomicAddPreCast,
                                                              PassThrough,
                                                              InMemoryDataOperationEnum::AtomicAdd,
                                                              InOutVectorSize>;

    using GridwiseCasting = GridwiseElementwise_1D<Tuple<InOutGrid1dDesc>,
                                                   Tuple<InOutGrid1dDesc>,
                                                   Tuple<const DInDataType_AutomicAddPreCast*>,
                                                   Tuple<DInDataType*>,
                                                   UnaryConvert,
                                                   InOutVectorSize,
                                                   Sequence<InOutVectorSize>,
                                                   Sequence<InOutVectorSize>>;

    struct Argument : public BaseArgument
    {
        Argument(const DOutDataType* p_dout,
                 const IndexDataType* p_indices,
                 DInDataType* p_din,
                 index_t dout_length,
                 index_t din_length,
                 const std::vector<ck::index_t>& window_lengths,
                 const std::vector<ck::index_t>& window_strides)
            : p_dout_{p_dout},
              p_indices_{p_indices},
              p_din_{p_din},
              dout_length_raw_{dout_length},
              din_length_raw_{din_length},
              blockSize_{256},
              windowOverlap_{false}
        {
            for(size_t i = 0; i < window_lengths.size(); ++i)
            {
                windowOverlap_ |= window_lengths.at(i) > window_strides.at(i);
            }
        }

        const DOutDataType* p_dout_;
        const IndexDataType* p_indices_;
        DInDataType* p_din_;
        index_t dout_length_raw_;
        index_t din_length_raw_;
        index_t blockSize_;
        bool windowOverlap_;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            index_t gridSize               = getAvailableComputeUnitCount(stream_config);
            index_t loop_step              = gridSize * arg.blockSize_ * InOutVectorSize;
            InOutGrid1dDesc din_grid_desc  = MakeDescriptor_M(arg.din_length_raw_, loop_step);
            InOutGrid1dDesc dout_grid_desc = MakeDescriptor_M(arg.dout_length_raw_, loop_step);

            if constexpr(is_same_v<DInDataType, float> || is_same_v<DInDataType, double>)
            {
                hip_check_error(hipMemsetAsync(arg.p_din_,
                                               0,
                                               arg.din_length_raw_ * sizeof(DInDataType),
                                               stream_config.stream_id_));

                if(arg.windowOverlap_)
                {
                    const auto put_kernel = kernel_put_element_1d<GridwisePutElementAtomicAdd,
                                                                  InOutGrid1dDesc,
                                                                  DOutDataType,
                                                                  IndexDataType,
                                                                  DInDataType,
                                                                  PassThrough>;

                    return launch_and_time_kernel(stream_config,
                                                  put_kernel,
                                                  dim3(gridSize),
                                                  dim3(arg.blockSize_),
                                                  0,
                                                  dout_grid_desc,
                                                  arg.p_dout_,
                                                  arg.p_indices_,
                                                  arg.p_din_,
                                                  PassThrough{});
                }
                else
                {
                    const auto put_kernel = kernel_put_element_1d<GridwisePutElementSet,
                                                                  InOutGrid1dDesc,
                                                                  DOutDataType,
                                                                  IndexDataType,
                                                                  DInDataType,
                                                                  PassThrough>;

                    return launch_and_time_kernel(stream_config,
                                                  put_kernel,
                                                  dim3(gridSize),
                                                  dim3(arg.blockSize_),
                                                  0,
                                                  dout_grid_desc,
                                                  arg.p_dout_,
                                                  arg.p_indices_,
                                                  arg.p_din_,
                                                  PassThrough{});
                }
            }
            else
            {
                if(arg.windowOverlap_)
                {
                    if(arg.p_workspace_ == nullptr)
                        throw std::runtime_error("wrong! WorkSpace pointer has not been set");

                    hip_check_error(
                        hipMemsetAsync(arg.p_workspace_,
                                       0,
                                       arg.din_length_raw_ * sizeof(DInDataType_AutomicAddPreCast),
                                       stream_config.stream_id_));

                    const auto put_kernel = kernel_put_element_1d<GridwisePutElementAtomicAdd,
                                                                  InOutGrid1dDesc,
                                                                  DOutDataType,
                                                                  IndexDataType,
                                                                  DInDataType_AutomicAddPreCast,
                                                                  PassThrough>;

                    const auto cast_kernel =
                        kernel_elementwise_1d<GridwiseCasting,
                                              Tuple<InOutGrid1dDesc>,
                                              Tuple<InOutGrid1dDesc>,
                                              Tuple<const DInDataType_AutomicAddPreCast*>,
                                              Tuple<DInDataType*>,
                                              UnaryConvert>;

                    float elapsed_time = launch_and_time_kernel(
                        stream_config,
                        put_kernel,
                        dim3(gridSize),
                        dim3(arg.blockSize_),
                        0,
                        dout_grid_desc,
                        arg.p_dout_,
                        arg.p_indices_,
                        static_cast<DInDataType_AutomicAddPreCast*>(arg.p_workspace_),
                        PassThrough{});

                    elapsed_time += launch_and_time_kernel(
                        stream_config,
                        cast_kernel,
                        dim3(gridSize),
                        dim3(arg.blockSize_),
                        0,
                        ck::make_tuple(din_grid_desc),
                        ck::make_tuple(din_grid_desc),
                        static_cast<DInDataType_AutomicAddPreCast*>(arg.p_workspace_),
                        arg.p_din_,
                        UnaryConvert{});

                    return elapsed_time;
                }
                else
                {
                    const auto put_kernel = kernel_put_element_1d<GridwisePutElementSet,
                                                                  InOutGrid1dDesc,
                                                                  DOutDataType,
                                                                  IndexDataType,
                                                                  DInDataType,
                                                                  PassThrough>;

                    hip_check_error(hipMemsetAsync(arg.p_din_,
                                                   0,
                                                   arg.din_length_raw_ * sizeof(DInDataType),
                                                   stream_config.stream_id_));

                    return launch_and_time_kernel(stream_config,
                                                  put_kernel,
                                                  dim3(gridSize),
                                                  dim3(arg.blockSize_),
                                                  0,
                                                  dout_grid_desc,
                                                  arg.p_dout_,
                                                  arg.p_indices_,
                                                  arg.p_din_,
                                                  PassThrough{});
                }
            }
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    size_t GetWorkSpaceSize(const BaseArgument* pArg) const override
    {
        const Argument* pArg_ = dynamic_cast<const Argument*>(pArg);

        bool needCast = pArg_->windowOverlap_ &&
                        !(is_same_v<DInDataType, float> || is_same_v<DInDataType, double>);

        if(!needCast)
            return 0;
        else
            return pArg_->din_length_raw_ * sizeof(DInDataType_AutomicAddPreCast);
    };

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        const Argument* pArg = dynamic_cast<const Argument*>(p_arg);
        if(pArg->din_length_raw_ % InOutVectorSize != 0 ||
           pArg->dout_length_raw_ % InOutVectorSize != 0)
        {
            return false;
        }
        return true;
    }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_dout,
                        const void* p_indices,
                        void* p_din,
                        index_t dout_length,
                        index_t din_length,
                        std::vector<ck::index_t> window_lengths,
                        std::vector<ck::index_t> window_strides) override
    {
        // Assume p_dout, p_indices, p_din are packed memory space, dout_length and din_length are
        // physical size of the packed tensor
        return std::make_unique<Argument>(static_cast<const DOutDataType*>(p_dout),
                                          static_cast<const IndexDataType*>(p_indices),
                                          static_cast<DInDataType*>(p_din),
                                          dout_length,
                                          din_length,
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
