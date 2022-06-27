// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_operation/gpu/device/device_multiple_reduce.hpp"

#include "ck/library/tensor_operation_instance/gpu/reduce/device_dual_reduce_instance.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/host_tensor/device_memory.hpp"
#include "ck/library/host_tensor/host_common_util.hpp"
#include "ck/library/host_tensor/host_reduction.hpp"
#include "ck/library/host_tensor/host_tensor.hpp"
#include "ck/library/host_tensor/host_tensor_generator.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

template <int Rank, int NumReduceDim, bool PropagateNan>
struct ReduceDescription
{
    static constexpr int Rank_          = Rank;
    static constexpr int NumReduceDim_  = NumReduceDim;
    static constexpr bool PropagateNan_ = PropagateNan;
};

using reduce_description_instances = std::tuple<ReduceDescription<4, 3, false>>;

template <typename DescriptionType>
bool description_match(const DescriptionType& description,
                       int Rank,
                       const std::vector<int>& reduceDims,
                       bool PropagateNan)
{
    if(description.Rank_ != Rank || description.PropagateNan_ != PropagateNan)
        return (false);

    if(DescriptionType::NumReduceDim_ != reduceDims.size())
        return (false);

    bool result = true;

    return (result);
};

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

namespace ck {
namespace profiler {

template <index_t Rank, index_t NumReduceDim>
static inline std::vector<int> get_invariant_dims(const std::vector<int>& reduceDims)
{
    assert(NumReduceDim == reduceDims.size());

    int reduceFlag = 0;

    // flag the bits for the reduceDims
    for(int i = 0; i < NumReduceDim; i++)
    {
        reduceFlag |= 1 << reduceDims[i];
    };

    std::vector<int> invariantDims;

    // collect invariant dimensions
    for(int i = 0; i < Rank; i++)
        if((reduceFlag & (1 << i)) == 0)
        {
            invariantDims.push_back(i);
        };

    return invariantDims;
};

template <typename InDataType,
          typename AccDataType,
          typename OutDataType1,
          typename OutDataType2,
          typename ReduceOperation,
          typename InElementwiseOperation1,
          typename AccElementwiseOperation1,
          typename InElementwiseOperation2,
          typename AccElementwiseOperation2,
          int Rank,
          int NumReduceDim,
          bool PropagateNan>
bool profile_dual_reduce_impl_impl(bool do_verification,
                                   int init_method,
                                   bool do_dumpout,
                                   bool time_kernel,
                                   const std::vector<size_t>& inLengths,
                                   const std::vector<int>& reduceDims,
                                   float alpha,
                                   float beta,
                                   InElementwiseOperation1 in_elementwise_op1,
                                   AccElementwiseOperation2 acc_elementwise_op1,
                                   InElementwiseOperation2 in_elementwise_op2,
                                   AccElementwiseOperation2 acc_elementwise_op2)
{
    using namespace ck::tensor_operation::device;
    using namespace ck::tensor_operation::device::device_reduce_instance;
    using ck::host_common::dumpBufferToFile;
    using ck::reduce::InMemoryDataOperatonSupportedOnDataType;

    constexpr bool out_support_atomic_add =
        InMemoryDataOperatonSupportedOnDataType<InMemoryDataOperationEnum::AtomicAdd,
                                                OutDataType1>::value &&
        InMemoryDataOperatonSupportedOnDataType<InMemoryDataOperationEnum::AtomicAdd,
                                                OutDataType2>::value;

    constexpr bool op_support_atomic_add =
        ReduceOperation::IsCompatibleInMemoryDataOperation(InMemoryDataOperationEnum::AtomicAdd);

    constexpr bool use_atomic_add = (out_support_atomic_add && op_support_atomic_add);

    bool pass = true;

    Tensor<InDataType> in(inLengths);

    std::vector<size_t> outLengths;

    const auto invariantDims = get_invariant_dims<Rank, NumReduceDim>(reduceDims);

    if(reduceDims.size() == Rank)
        outLengths.push_back(1);
    else
        for(auto dim : invariantDims)
            outLengths.push_back(inLengths[dim]);

    Tensor<OutDataType1> out1_ref(outLengths);
    Tensor<OutDataType1> out1(outLengths);
    Tensor<OutDataType1> out2_ref(outLengths);
    Tensor<OutDataType1> out2(outLengths);

    auto inStrides  = in.mDesc.GetStrides();
    auto outStrides = out1.mDesc.GetStrides();

    size_t invariant_total_length = out1.mDesc.GetElementSize();
    size_t reduce_total_length    = in.mDesc.GetElementSize() / invariant_total_length;

    std::size_t num_thread = 1;

    if(do_verification)
    {
        switch(init_method)
        {
        case 0: break;
        case 1:
            in.GenerateTensorValue(GeneratorTensor_1<InDataType>{1}, num_thread);
            if(beta != 0.0f)
            {
                out1_ref.GenerateTensorValue(GeneratorTensor_1<OutDataType1>{1}, num_thread);
                out2_ref.GenerateTensorValue(GeneratorTensor_1<OutDataType2>{1}, num_thread);
            };
            break;
        case 2:
            in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5}, num_thread);
            if(beta != 0.0f)
            {
                out1_ref.GenerateTensorValue(GeneratorTensor_2<OutDataType1>{-5, 5}, num_thread);
                out2_ref.GenerateTensorValue(GeneratorTensor_2<OutDataType2>{-5, 5}, num_thread);
            };
            break;
        default:
            in.GenerateTensorValue(GeneratorTensor_3<InDataType>{-5.0, 5.0}, num_thread);
            if(beta != 0.0f)
            {
                out1_ref.GenerateTensorValue(GeneratorTensor_3<OutDataType1>{-5.0, 5.0},
                                             num_thread);
                out2_ref.GenerateTensorValue(GeneratorTensor_3<OutDataType2>{-5.0, 5.0},
                                             num_thread);
            };
        }

        if(beta != 0.0f)
        {
            for(size_t i = 0; i < out1_ref.mDesc.GetElementSpace(); i++)
            {
                out1.mData[i] = out1_ref.mData[i];
                out2.mData[i] = out2_ref.mData[i];
            };
        };
    };

    // these buffers are usually provided by the user application
    DeviceMem in_dev(sizeof(InDataType) * in.mDesc.GetElementSpace());
    DeviceMem out1_dev(sizeof(OutDataType1) * out1.mDesc.GetElementSpace());
    DeviceMem out2_dev(sizeof(OutDataType1) * out2.mDesc.GetElementSpace());

    in_dev.ToDevice(in.mData.data());

    if(beta != 0.0f)
    {
        out1_dev.ToDevice(out1.mData.data());
        out2_dev.ToDevice(out2.mData.data());
    };

    float best_avg_time   = 0;
    float best_gb_per_sec = 0;

    using DeviceDualReduceInstPtr =
        DeviceMultipleReducePtr<2,
                                Tuple<InElementwiseOperation1, InElementwiseOperation2>,
                                Tuple<AccElementwiseOperation1, AccElementwiseOperation2>>;

    std::vector<DeviceDualReduceInstPtr> reduce_ptrs;

    add_device_dual_reduce_instance_blockwise<InDataType,
                                              AccDataType,
                                              OutDataType1,
                                              OutDataType2,
                                              ReduceOperation,
                                              InElementwiseOperation1,
                                              InElementwiseOperation2,
                                              AccElementwiseOperation1,
                                              AccElementwiseOperation2,
                                              PropagateNan,
                                              Rank,
                                              NumReduceDim>(reduce_ptrs);

    if constexpr(use_atomic_add)
    {
        add_device_dual_reduce_instance_multiblock_atomic_add<InDataType,
                                                              AccDataType,
                                                              OutDataType1,
                                                              OutDataType2,
                                                              ReduceOperation,
                                                              InElementwiseOperation1,
                                                              InElementwiseOperation2,
                                                              AccElementwiseOperation1,
                                                              AccElementwiseOperation2,
                                                              PropagateNan,
                                                              Rank,
                                                              NumReduceDim>(reduce_ptrs);
    }

    if(reduce_ptrs.empty())
    {
        throw std::runtime_error("Wrong! No device REDUCE instance found");
    };

    if(do_verification)
    {
        ReductionHost<InDataType,
                      AccDataType,
                      OutDataType1,
                      ReduceOperation,
                      InElementwiseOperation1,
                      AccElementwiseOperation1,
                      Rank,
                      NumReduceDim,
                      PropagateNan,
                      false>
            hostReduce1(in.mDesc, out1_ref.mDesc, invariantDims, reduceDims);

        hostReduce1.Run(alpha,
                        in.mData.data(),
                        beta,
                        out1_ref.mData.data(),
                        nullptr,
                        in_elementwise_op1,
                        acc_elementwise_op1);

        ReductionHost<InDataType,
                      AccDataType,
                      OutDataType2,
                      ReduceOperation,
                      InElementwiseOperation2,
                      AccElementwiseOperation2,
                      Rank,
                      NumReduceDim,
                      PropagateNan,
                      false>
            hostReduce2(in.mDesc, out2_ref.mDesc, invariantDims, reduceDims);

        hostReduce2.Run(alpha,
                        in.mData.data(),
                        beta,
                        out2_ref.mData.data(),
                        nullptr,
                        in_elementwise_op2,
                        acc_elementwise_op2);
    };

    std::vector<ck::index_t> i_inLengths;
    std::vector<ck::index_t> i_inStrides;
    std::vector<ck::index_t> i_outLengths;
    std::vector<ck::index_t> i_outStrides;

    i_inLengths.assign(inLengths.begin(), inLengths.end());
    i_inStrides.assign(inStrides.begin(), inStrides.end());
    i_outLengths.assign(outLengths.begin(), outLengths.end());
    i_outStrides.assign(outStrides.begin(), outStrides.end());

    for(auto& reduce_ptr : reduce_ptrs)
    {
        std::array<float, 2> alpha_values = {alpha, alpha};
        std::array<float, 2> beta_values  = {beta, beta};

        auto argument_ptr = reduce_ptr->MakeArgumentPointer(
            i_inLengths,
            i_inStrides,
            i_outLengths,
            i_outStrides,
            reduceDims,
            alpha_values,
            beta_values,
            in_dev.GetDeviceBuffer(),
            {out1_dev.GetDeviceBuffer(), out2_dev.GetDeviceBuffer()},
            make_tuple(in_elementwise_op1, in_elementwise_op2),
            make_tuple(acc_elementwise_op1, acc_elementwise_op2));

        if(!reduce_ptr->IsSupportedArgument(argument_ptr.get()))
            continue;

        std::string reduce_name = reduce_ptr->GetTypeString();

        auto invoker_ptr = reduce_ptr->MakeInvokerPointer();

        float avg_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

        std::size_t num_bytes =
            invariant_total_length * reduce_total_length * sizeof(InDataType) +
            invariant_total_length * (sizeof(OutDataType1) + sizeof(OutDataType2));

        float gb_per_sec = num_bytes / 1.E6 / avg_time;

        if(time_kernel)
            std::cout << "Perf: " << avg_time << " ms, " << gb_per_sec << " GB/s, " << reduce_name
                      << std::endl;

        if(gb_per_sec > best_gb_per_sec)
        {
            best_avg_time   = avg_time;
            best_gb_per_sec = gb_per_sec;
        }

        if(do_verification)
        {
            bool single_pass = true;

            out1_dev.FromDevice(out1.mData.data());
            out2_dev.FromDevice(out2.mData.data());
            single_pass &= ck::utils::check_err(out1.mData, out1_ref.mData);
            single_pass &= ck::utils::check_err(out2.mData, out2_ref.mData);

            if(!single_pass)
            {
                std::cout << "Fail Info: " << reduce_ptr->GetTypeString() << std::endl;
            }

            pass = pass && single_pass;
        };

        if(do_dumpout)
        {
            dumpBufferToFile("dump_in.bin", in.mData.data(), in.mDesc.GetElementSize());
            dumpBufferToFile("dump_out1.bin", out1.mData.data(), out1.mDesc.GetElementSize());
            dumpBufferToFile(
                "dump_out1_host.bin", out1_ref.mData.data(), out1_ref.mDesc.GetElementSize());
            dumpBufferToFile(
                "dump_out2_host.bin", out2_ref.mData.data(), out2_ref.mDesc.GetElementSize());
        };

        if(time_kernel)
            std::cout << "Best Perf: " << best_avg_time << " ms, " << best_gb_per_sec << " GB/s"
                      << std::endl;
    };

    return pass;
};

template <typename InDataType, typename AccDataType, typename OutDataType1, typename OutDataType2>
bool profile_dual_reduce_impl_for_mean_meansquare(bool do_verification,
                                                  int init_method,
                                                  bool do_dumpout,
                                                  bool time_kernel,
                                                  const std::vector<size_t>& inLengths,
                                                  const std::vector<int>& reduceDims,
                                                  bool PropagateNan,
                                                  float alpha,
                                                  float beta)
{
    using ck::tensor_operation::device::device_reduce_instance::Divide;
    using ck::tensor_operation::device::device_reduce_instance::PassThrough;
    using ck::tensor_operation::device::device_reduce_instance::Square;
    using ck::tensor_operation::device::device_reduce_instance::Sum;

    bool matched = false;
    bool pass    = true;

    using tuple_of_description_instances =
        tensor_operation::device::device_reduce_instance::reduce_description_instances;

    const auto tuple_object = tuple_of_description_instances{};

    static_for<0, std::tuple_size<tuple_of_description_instances>::value, 1>{}([&](auto i) {
        if(matched)
            return;

        using descType = remove_cvref_t<decltype(std::get<i>(tuple_object))>;

        if(!description_match(descType{}, inLengths.size(), reduceDims, PropagateNan))
            return;

        size_t reduceLength = 1;

        for(auto dim : reduceDims)
            reduceLength *= inLengths[dim];

        pass = pass && profile_dual_reduce_impl_impl<InDataType,
                                                     AccDataType,
                                                     OutDataType1,
                                                     OutDataType2,
                                                     Sum,
                                                     PassThrough,
                                                     Divide,
                                                     Square,
                                                     Divide,
                                                     descType::Rank_,
                                                     descType::NumReduceDim_,
                                                     descType::PropagateNan_>(
                           do_verification,
                           init_method,
                           do_dumpout,
                           time_kernel,
                           inLengths,
                           reduceDims,
                           alpha,
                           beta,
                           PassThrough{},
                           Divide{static_cast<int32_t>(reduceLength)},
                           Square{},
                           Divide{static_cast<int32_t>(reduceLength)});

        matched = true;
    });

    return pass;
};

} // namespace profiler
} // namespace ck
