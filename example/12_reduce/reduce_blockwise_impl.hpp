// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>

#include "ck/ck.hpp"
#include "ck/utility/reduction_enums.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"
#include "ck/tensor_operation/gpu/device/device_reduce_multiblock.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/host_common_util.hpp"
#include "ck/library/utility/host_reduction.hpp"

#include "reduce_example_common.hpp"

template <typename InOutDataType,
          typename AccDataType,
          ck::ReduceTensorOp ReduceOpId,
          ck::index_t Rank,
          ck::index_t NumReduceDim,
          bool PropagateNan,
          bool OutputIndex>
int reduce_blockwise_impl(bool do_verification,
                          int init_method,
                          bool time_kernel,
                          const std::vector<size_t>& inLengths,
                          const std::vector<int>& reduceDims,
                          float alpha,
                          float beta)

{
    using namespace ck;
    using namespace ck::tensor_operation::device;

    constexpr bool op_support_indices =
        (ReduceOpId == ReduceTensorOp::MIN || ReduceOpId == ReduceTensorOp::MAX ||
         ReduceOpId == ReduceTensorOp::AMAX);

    constexpr bool invalid_reduce_1 = OutputIndex && !op_support_indices;

    // 1) If InOutDataType is half_t, must use half_t as AccDataType for indexable reduction
    // operations 2) If InOutDataType is half_t, must use float as AccDataType for non-indexable
    // reduction operations
    constexpr bool invalid_reduce_2 =
        std::is_same<InOutDataType, half_t>::value &&
        ((!op_support_indices && !std::is_same<AccDataType, float>::value) ||
         (op_support_indices && !std::is_same<AccDataType, half_t>::value));

    // 1) If InOutDataType is float, must use float as AccDataType for indexable reduction
    // operations
    constexpr bool invalid_reduce_3 =
        std::is_same<InOutDataType, float>::value &&
        (op_support_indices && !std::is_same<AccDataType, float>::value);

    // 1) If InOutDataType is int8_t or int4_t, must use int8_t as AccDataType for indexable
    // reduction operations 2) If InOutDataType is int8_t or int4_t, must use int32_t as AccDataType
    // for non-indexable reduction operations
    constexpr bool invalid_reduce_4 =
        (std::is_same<InOutDataType, int8_t>::value ||
         std::is_same<InOutDataType, int4_t>::value) &&
        ((!op_support_indices && !std::is_same<AccDataType, int32_t>::value) ||
         (op_support_indices && !std::is_same<AccDataType, int8_t>::value));

    // 1) If InOutDataType is int8_t or int4_t, the supported operation must be either indexable
    // operations or ADD/AVG
    constexpr bool invalid_reduce_5 = (std::is_same<InOutDataType, int8_t>::value ||
                                       std::is_same<InOutDataType, int4_t>::value) &&
                                      (!op_support_indices && ReduceOpId != ReduceTensorOp::ADD &&
                                       ReduceOpId != ReduceTensorOp::AVG);

    // 1) If InOutDataType is bhalf_t, must use float as AccDataType for all reduction operations
    constexpr bool invalid_reduce_6 =
        std::is_same<InOutDataType, bhalf_t>::value && !std::is_same<AccDataType, float>::value;

    constexpr bool invalid_reduce = (invalid_reduce_1 || invalid_reduce_2 || invalid_reduce_3 ||
                                     invalid_reduce_4 || invalid_reduce_5 || invalid_reduce_6);

    if(invalid_reduce)
    {
        std::cerr << "The reduction setting is invalid, exiting!" << std::endl;
        return (-1);
    };

    using ReduceOperation = typename reduce_binary_operator<ReduceOpId>::opType;
    using InElementwiseOperation =
        typename reduce_unary_operator<ReduceOpId, true, true>::InElementwiseOperation;
    using AccElementwiseOperation =
        typename reduce_unary_operator<ReduceOpId, true, true>::AccElementwiseOperation;

    using InOutDataTypeInDevice = typename std::
        conditional<std::is_same<InOutDataType, int4_t>::value, int8_t, InOutDataType>::type;

    using DeviceReduceInstance =
        ck::tensor_operation::device::DeviceReduceMultiBlock<InOutDataTypeInDevice,
                                                             AccDataType,
                                                             InOutDataTypeInDevice,
                                                             Rank,
                                                             NumReduceDim,
                                                             ReduceOperation,
                                                             InElementwiseOperation,
                                                             AccElementwiseOperation,
                                                             InMemoryDataOperationEnum::Set,
                                                             PropagateNan,
                                                             OutputIndex,
                                                             false, // HaveIndexInputIfOutputIndex
                                                             256,   // BlockSize
                                                             4,     // MThreadClusterSize
                                                             64,    // KThreadClusterSize
                                                             1,     // MThreadSliceSize
                                                             1,     // KThreadSliceSize
                                                             0,     // InSrcVectorDim
                                                             1,     // InSrceVectorSize
                                                             1>;    // OutDstVectorSize

    Tensor<InOutDataType> in(inLengths);

    std::vector<size_t> outLengths;

    std::vector<int> invariantDims = get_invariant_dims<Rank, NumReduceDim>(reduceDims);

    if(invariantDims.empty())
        outLengths.push_back(1);
    else
        for(auto dim : invariantDims)
            outLengths.push_back(inLengths[dim]);

    Tensor<InOutDataType> out_ref(outLengths);
    Tensor<InOutDataType> out(outLengths);
    Tensor<int> out_indices_ref(outLengths);
    Tensor<int> out_indices(outLengths);

    auto inStrides  = in.mDesc.GetStrides();
    auto outStrides = out.mDesc.GetStrides();

    size_t invariant_total_length = out.mDesc.GetElementSize();
    size_t reduce_total_length    = in.mDesc.GetElementSize() / invariant_total_length;

    std::size_t num_thread = 1;

    if(do_verification)
    {
        switch(init_method)
        {
        case 0: break;
        case 1:
            in.GenerateTensorValue(GeneratorTensor_1<InOutDataType>{1}, num_thread);
            if(beta != 0.0f)
                out_ref.GenerateTensorValue(GeneratorTensor_1<InOutDataType>{1}, num_thread);
            break;
        case 2:
            in.GenerateTensorValue(GeneratorTensor_2<InOutDataType>{-5, 5}, num_thread);
            if(beta != 0.0f)
                out_ref.GenerateTensorValue(GeneratorTensor_2<InOutDataType>{-5, 5}, num_thread);
            break;
        default:
            in.GenerateTensorValue(GeneratorTensor_3<InOutDataType>{-5.0, 5.0}, num_thread);
            if(beta != 0.0f)
                out_ref.GenerateTensorValue(GeneratorTensor_3<InOutDataType>{-5.0, 5.0},
                                            num_thread);
        }

        if(beta != 0.0f)
            for(size_t i = 0; i < out_ref.mDesc.GetElementSpaceSize(); i++)
                out.mData[i] = out_ref.mData[i];
    };

    // these buffers are usually provided by the user application
    DeviceMem in_dev(sizeof(InOutDataTypeInDevice) * in.mDesc.GetElementSpaceSize());
    DeviceMem out_dev(sizeof(InOutDataTypeInDevice) * out.mDesc.GetElementSpaceSize());

    if(std::is_same<InOutDataType, int4_t>::value)
    {
        std::vector<InOutDataTypeInDevice> tmp_buf(in.mData.size());

        std::copy_n(in.mData.data(), in.mData.size(), tmp_buf.data());
        in_dev.ToDevice(tmp_buf.data());
    }
    else
        in_dev.ToDevice(in.mData.data());

    if(beta != 0.0f)
    {
        if(std::is_same<InOutDataType, int4_t>::value)
        {
            std::vector<InOutDataTypeInDevice> tmp_buf(in.mData.size());

            std::copy_n(out.mData.data(), out.mData.size(), tmp_buf.data());
            out_dev.ToDevice(tmp_buf.data());
        }
        else
            out_dev.ToDevice(out.mData.data());
    };

    size_t indicesSizeInBytes = OutputIndex ? out.mDesc.GetElementSize() * sizeof(int32_t) : 0;

    DeviceMem out_index_dev(indicesSizeInBytes);

    InElementwiseOperation in_elementwise_op;
    AccElementwiseOperation acc_elementwise_op;

    std::tie(in_elementwise_op, acc_elementwise_op) =
        reduce_unary_operator<ReduceOpId, true, true>::GetElementwiseOperator(
            static_cast<int32_t>(reduce_total_length));

    if(do_verification)
    {
        ReductionHost<InOutDataType,
                      AccDataType,
                      InOutDataType,
                      ReduceOperation,
                      InElementwiseOperation,
                      AccElementwiseOperation,
                      Rank,
                      NumReduceDim,
                      PropagateNan,
                      OutputIndex>
            hostReduce(in.mDesc, out_ref.mDesc, invariantDims, reduceDims);

        hostReduce.Run(alpha,
                       in.mData.data(),
                       beta,
                       out_ref.mData.data(),
                       out_indices_ref.mData.data(),
                       in_elementwise_op,
                       acc_elementwise_op);
    };

    std::vector<ck::index_t> i_inLengths;
    std::vector<ck::index_t> i_inStrides;
    std::vector<ck::index_t> i_outLengths;
    std::vector<ck::index_t> i_outStrides;

    i_inLengths.assign(inLengths.begin(), inLengths.end());
    i_inStrides.assign(inStrides.begin(), inStrides.end());
    i_outLengths.assign(outLengths.begin(), outLengths.end());
    i_outStrides.assign(outStrides.begin(), outStrides.end());

    auto reduce = DeviceReduceInstance{};

    auto argument_ptr = reduce.MakeArgumentPointer(i_inLengths,
                                                   i_inStrides,
                                                   i_outLengths,
                                                   i_outStrides,
                                                   reduceDims,
                                                   alpha,
                                                   beta,
                                                   in_dev.GetDeviceBuffer(),
                                                   nullptr,
                                                   out_dev.GetDeviceBuffer(),
                                                   out_index_dev.GetDeviceBuffer(),
                                                   in_elementwise_op,
                                                   acc_elementwise_op);

    if(!reduce.IsSupportedArgument(argument_ptr.get()))
    {
        std::cerr
            << "The runtime parameters seems not supported by the DeviceReduce instance, exiting!"
            << std::endl;

        return (-2);
    };

    std::string reduce_name = reduce.GetTypeString();

    auto invoker_ptr = reduce.MakeInvokerPointer();

    float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

    std::size_t num_bytes = invariant_total_length * reduce_total_length * sizeof(InOutDataType) +
                            invariant_total_length * sizeof(InOutDataType);

    float gb_per_sec = num_bytes / 1.E6 / avg_time;

    std::cout << "Perf: " << avg_time << " ms, " << gb_per_sec << " GB/s, " << reduce_name
              << std::endl;

    bool pass = true;

    if(do_verification)
    {
        if(std::is_same<InOutDataType, int4_t>::value)
        {
            std::vector<InOutDataTypeInDevice> tmp_buf(out.mData.size());

            out_dev.FromDevice(tmp_buf.data());

            std::copy_n(tmp_buf.data(), out.mData.size(), out.mData.data());
        }
        else
            out_dev.FromDevice(out.mData.data());

        pass = pass && ck::utils::check_err(out.mData, out_ref.mData);

        if(OutputIndex)
        {
            out_index_dev.FromDevice(out_indices.mData.data());
            pass = pass && ck::utils::check_err(out_indices.mData, out_indices_ref.mData);
        };
    };

    return (pass ? 0 : 1);
}
