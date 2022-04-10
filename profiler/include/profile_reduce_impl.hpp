#pragma once

#include "check_err.hpp"
#include "device_reduce.hpp"
#include "device_reduce_instance.hpp"
#include "reduction_enums.hpp"
#include "host_reduction.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

template <int Rank, int NumReduceDim, int ReduceOpId, int NanOpt, int IndicesOpt>
struct ReduceDescription
{
    static constexpr int Rank_         = Rank;
    static constexpr int NumReduceDim_ = NumReduceDim;
    static constexpr int ReduceOpId_   = ReduceOpId;
    static constexpr int NanOpt_       = NanOpt;
    static constexpr int IndicesOpt_   = IndicesOpt;
};

using reduce_description_instances = std::tuple<ReduceDescription<4, 3, 0, 0, 0>, // for ADD
                                                ReduceDescription<4, 4, 0, 0, 0>,
                                                ReduceDescription<4, 1, 0, 0, 0>,
                                                ReduceDescription<2, 1, 0, 0, 0>,

                                                ReduceDescription<4, 3, 5, 0, 0>, // for AVG
                                                ReduceDescription<4, 4, 5, 0, 0>,
                                                ReduceDescription<4, 1, 5, 0, 0>,
                                                ReduceDescription<2, 1, 5, 0, 0>,

                                                ReduceDescription<4, 3, 7, 0, 0>, // for NORM2
                                                ReduceDescription<4, 4, 7, 0, 0>,
                                                ReduceDescription<4, 1, 7, 0, 0>,
                                                ReduceDescription<2, 1, 7, 0, 0>,

                                                ReduceDescription<4, 3, 2, 0, 0>, // for MIN
                                                ReduceDescription<4, 4, 2, 0, 0>,
                                                ReduceDescription<4, 1, 2, 0, 0>,
                                                ReduceDescription<2, 1, 2, 0, 0>,
                                                ReduceDescription<4, 3, 3, 0, 0>, // for MAX
                                                ReduceDescription<4, 4, 3, 0, 0>,
                                                ReduceDescription<4, 1, 3, 0, 0>,
                                                ReduceDescription<2, 1, 3, 0, 0>,
                                                ReduceDescription<4, 3, 4, 0, 0>, // for AMAX
                                                ReduceDescription<4, 4, 4, 0, 0>,
                                                ReduceDescription<4, 1, 4, 0, 0>,
                                                ReduceDescription<2, 1, 4, 0, 0>,

                                                ReduceDescription<4, 3, 2, 0, 1>, // for MIN
                                                ReduceDescription<4, 4, 2, 0, 1>,
                                                ReduceDescription<4, 1, 2, 0, 1>,
                                                ReduceDescription<2, 1, 2, 0, 1>,
                                                ReduceDescription<4, 3, 3, 0, 1>, // for MAX
                                                ReduceDescription<4, 4, 3, 0, 1>,
                                                ReduceDescription<4, 1, 3, 0, 1>,
                                                ReduceDescription<2, 1, 3, 0, 1>,
                                                ReduceDescription<4, 3, 4, 0, 1>, // for AMAX
                                                ReduceDescription<4, 4, 4, 0, 1>,
                                                ReduceDescription<4, 1, 4, 0, 1>,
                                                ReduceDescription<2, 1, 4, 0, 1>>;

template <typename DescriptionType>
bool description_match(const DescriptionType& description,
                       int Rank,
                       const std::vector<int>& reduceDims,
                       ReduceTensorOp ReduceOpId,
                       NanPropagation NanOpt,
                       ReduceTensorIndices IndicesOpt)
{
    if(description.Rank_ != Rank || description.ReduceOpId_ != static_cast<int>(ReduceOpId) ||
       description.NanOpt_ != static_cast<int>(NanOpt) ||
       description.IndicesOpt_ != static_cast<int>(IndicesOpt))
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

template <typename T>
static void dumpBufferToFile(const char* fileName, T* data, size_t dataNumItems)
{
    std::ofstream outFile(fileName, std::ios::binary);
    if(outFile)
    {
        outFile.write(reinterpret_cast<char*>(data), dataNumItems * sizeof(T));
        outFile.close();
        std::cout << "Write output to file " << fileName << std::endl;
    }
    else
    {
        std::cout << "Could not open file " << fileName << " for writing" << std::endl;
    }
};

// map the data type used by the GPU kernels to the corresponding type used by the host codes
template <typename InType>
struct type_mapping
{
    using OutType = InType;
};

template <>
struct type_mapping<ck::half_t>
{
    using OutType = half_float::half;
};

template <typename InDataType,
          typename AccDataType,
          typename OutDataType,
          int Rank,
          int NumReduceDim,
          ReduceTensorOp ReduceOpId,
          NanPropagation NanOpt,
          ReduceTensorIndices IndicesOpt>
void profile_reduce_impl_impl(bool do_verification,
                              int init_method,
                              bool do_log,
                              bool do_dumpout,
                              int nrepeat,
                              const std::vector<size_t>& inLengths,
                              const std::vector<int>& reduceDims,
                              float alpha,
                              float beta)
{
    using namespace ck::tensor_operation::device;
    using namespace ck::tensor_operation::device::device_reduce_instance;
    using namespace ck::host_reduce;

    constexpr bool op_support_indices =
        (ReduceOpId == ReduceTensorOp::MIN || ReduceOpId == ReduceTensorOp::MAX ||
         ReduceOpId == ReduceTensorOp::AMAX);

    constexpr bool NeedIndices =
        (op_support_indices && (IndicesOpt != ReduceTensorIndices::NO_INDICES));

    constexpr bool PropagateNan = (NanOpt == NanPropagation::PROPAGATE_NAN);

    constexpr bool out_support_atomic_add = std::is_same<OutDataType, float>::value;
    constexpr bool op_support_atomic_add =
        !op_support_indices && ReduceOpId != ReduceTensorOp::NORM2;
    constexpr bool use_atomic_add = (out_support_atomic_add && op_support_atomic_add);

    // 1) If InDataType is half_t, must use half_t as AccDataType for indexable reduction operations
    // 2) If InDataType is half_t, must use float as AccDataType for non-indexable reduction
    // operations
    constexpr bool invalid_reduce_1 =
        std::is_same<InDataType, half_t>::value &&
        ((!op_support_indices && !std::is_same<AccDataType, float>::value) ||
         (op_support_indices && !std::is_same<AccDataType, half_t>::value));

    // 1) If InDataType is float, must use float as AccDataType for indexable reduction operations
    constexpr bool invalid_reduce_2 =
        std::is_same<InDataType, float>::value &&
        (op_support_indices && !std::is_same<AccDataType, float>::value);

    // 1) The indices can only be used when the reduction operation is indexable
    constexpr bool invalid_reduce_3 =
        (!op_support_indices && IndicesOpt != ReduceTensorIndices::NO_INDICES);

    // 1) If InDataType is int8_t, must use int8_t as AccDataType for indexable reduction operations
    // 2) If InDataType is int8_t, must use int32_t as AccDataType for non-indexable reduction
    // operations
    constexpr bool invalid_reduce_4 =
        std::is_same<InDataType, int8_t>::value &&
        ((!op_support_indices && !std::is_same<AccDataType, int32_t>::value) ||
         (op_support_indices && !std::is_same<AccDataType, int8_t>::value));

    // 1) If InDataType is int8_t, the supported operation must be either indexable operations or
    // ADD/AVG
    constexpr bool invalid_reduce_5 = std::is_same<InDataType, int8_t>::value &&
                                      (!op_support_indices && ReduceOpId != ReduceTensorOp::ADD &&
                                       ReduceOpId != ReduceTensorOp::AVG);

    // 1) If InDataType is bhalf_t, must use float as AccDataType for all reduction operations
    constexpr bool invalid_reduce_6 =
        std::is_same<InDataType, bhalf_t>::value && !std::is_same<AccDataType, float>::value;

    constexpr bool invalid_reduce = (invalid_reduce_1 || invalid_reduce_2 || invalid_reduce_3 ||
                                     invalid_reduce_4 || invalid_reduce_5 || invalid_reduce_6);

    if constexpr(!invalid_reduce)
    {
        Tensor<InDataType> in(inLengths);

        std::vector<size_t> outLengths;

        const auto invariantDims = get_invariant_dims<Rank, NumReduceDim>(reduceDims);

        if(reduceDims.size() == Rank)
            outLengths.push_back(1);
        else
            for(auto dim : invariantDims)
                outLengths.push_back(inLengths[dim]);

        Tensor<OutDataType> out_ref(outLengths);
        Tensor<OutDataType> out(outLengths);
        Tensor<int32_t> out_indices_ref(outLengths);
        Tensor<int32_t> out_indices(outLengths);

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
                in.GenerateTensorValue(GeneratorTensor_1<InDataType>{1}, num_thread);
                if(beta != 0.0f)
                    out_ref.GenerateTensorValue(GeneratorTensor_1<InDataType>{1}, num_thread);
                break;
            case 2:
                in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5}, num_thread);
                if(beta != 0.0f)
                    out_ref.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5}, num_thread);
                break;
            default:
                in.GenerateTensorValue(GeneratorTensor_3<InDataType>{-5.0, 5.0}, num_thread);
                if(beta != 0.0f)
                    out_ref.GenerateTensorValue(GeneratorTensor_3<InDataType>{-5.0, 5.0},
                                                num_thread);
            }

            if(beta != 0.0f)
                for(size_t i = 0; i < out_ref.mDesc.GetElementSpace(); i++)
                    out.mData[i] = out_ref.mData[i];
        };

        // these buffers are usually provided by the user application
        DeviceMem in_dev(sizeof(InDataType) * in.mDesc.GetElementSpace());
        DeviceMem out_dev(sizeof(OutDataType) * out.mDesc.GetElementSpace());

        in_dev.ToDevice(in.mData.data());

        if(beta != 0.0f)
            out_dev.ToDevice(out.mData.data());

        size_t indicesSizeInBytes = NeedIndices ? out.mDesc.GetElementSize() * sizeof(int) : 0;

        DeviceMem out_indices_dev(indicesSizeInBytes);

        float best_avg_time   = 0;
        float best_gb_per_sec = 0;

        using InElementwiseOperation_0 =
            typename reduce_unary_operator<AccDataType, ReduceOpId, true, true>::
                InElementwiseOperation;
        using AccElementwiseOperation_0 =
            typename reduce_unary_operator<AccDataType, ReduceOpId, true, true>::
                AccElementwiseOperation;
        using InElementwiseOperation_1 =
            typename reduce_unary_operator<AccDataType, ReduceOpId, true, false>::
                InElementwiseOperation;
        using AccElementwiseOperation_1 =
            typename reduce_unary_operator<AccDataType, ReduceOpId, true, false>::
                AccElementwiseOperation;
        using InElementwiseOperation_2 =
            typename reduce_unary_operator<AccDataType, ReduceOpId, false, true>::
                InElementwiseOperation;
        using AccElementwiseOperation_2 =
            typename reduce_unary_operator<AccDataType, ReduceOpId, false, true>::
                AccElementwiseOperation;

        using DeviceReduceInstPtr0 =
            DeviceReducePtr<InElementwiseOperation_0, AccElementwiseOperation_0>;
        using DeviceReduceInstPtr1 =
            DeviceReducePtr<InElementwiseOperation_1, AccElementwiseOperation_1>;
        using DeviceReduceInstPtr2 =
            DeviceReducePtr<InElementwiseOperation_2, AccElementwiseOperation_2>;

        std::vector<DeviceReduceInstPtr0> reduce0_ptrs;
        std::vector<DeviceReduceInstPtr1> reduce1_ptrs;
        std::vector<DeviceReduceInstPtr2> reduce2_ptrs;

        add_device_reduce_instance_threadwise<InDataType,
                                              AccDataType,
                                              OutDataType,
                                              Rank,
                                              NumReduceDim,
                                              ReduceOpId,
                                              NanOpt,
                                              IndicesOpt>(reduce0_ptrs);

        add_device_reduce_instance_blockwise<InDataType,
                                             AccDataType,
                                             OutDataType,
                                             Rank,
                                             NumReduceDim,
                                             ReduceOpId,
                                             NanOpt,
                                             IndicesOpt>(reduce0_ptrs);

        if constexpr(use_atomic_add)
        {
            add_device_reduce_instance_multiblock_atomic_add<InDataType,
                                                             AccDataType,
                                                             OutDataType,
                                                             Rank,
                                                             NumReduceDim,
                                                             ReduceOpId,
                                                             NanOpt,
                                                             IndicesOpt>(reduce0_ptrs);
        }
        else
        {
            add_device_reduce_instance_multiblock_partial_reduce<InDataType,
                                                                 AccDataType,
                                                                 OutDataType,
                                                                 Rank,
                                                                 NumReduceDim,
                                                                 ReduceOpId,
                                                                 NanOpt,
                                                                 IndicesOpt>(reduce1_ptrs);
        };

        // used for secondary reduction
        if constexpr(!use_atomic_add)
        {
            add_device_reduce_instance_blockwise_second_call<AccDataType,
                                                             AccDataType,
                                                             OutDataType,
                                                             Rank,
                                                             NumReduceDim,
                                                             ReduceOpId,
                                                             NanOpt,
                                                             IndicesOpt>(reduce2_ptrs);
        };

        if(reduce0_ptrs.empty() && reduce1_ptrs.empty())
        {
            throw std::runtime_error("Wrong! No device REDUCE instance found");
        };

        if(do_verification)
        {
            using HostInDataType  = typename type_mapping<InDataType>::OutType;
            using HostOutDataType = typename type_mapping<OutDataType>::OutType;
            using HostAccDataType = typename type_mapping<AccDataType>::OutType;

            ReductionHost<HostInDataType,
                          HostAccDataType,
                          HostOutDataType,
                          ReduceOpId,
                          Rank,
                          NumReduceDim,
                          PropagateNan,
                          NeedIndices>
                hostReduce(in.mDesc, out_ref.mDesc, invariantDims, reduceDims);

            hostReduce.Run(alpha,
                           reinterpret_cast<const HostInDataType*>(in.mData.data()),
                           beta,
                           reinterpret_cast<HostOutDataType*>(out_ref.mData.data()),
                           out_indices_ref.mData.data());
        };

        const auto i_inLengths  = to_int_vector(inLengths);
        const auto i_inStrides  = to_int_vector(inStrides);
        const auto i_outLengths = to_int_vector(outLengths);
        const auto i_outStrides = to_int_vector(outStrides);

        for(auto& reduce_ptr : reduce0_ptrs)
        {
            auto wsSizeInBytes = reduce_ptr->GetWorkspaceSizeInBytes(i_inLengths, reduceDims);

            DeviceMem ws_dev(wsSizeInBytes);

            InElementwiseOperation_0 in_elementwise_op_0(static_cast<int32_t>(reduce_total_length));
            AccElementwiseOperation_0 acc_elementwise_op_0(
                static_cast<int32_t>(reduce_total_length));

            auto argument_ptr = reduce_ptr->MakeArgumentPointer(i_inLengths,
                                                                i_inStrides,
                                                                i_outLengths,
                                                                i_outStrides,
                                                                reduceDims,
                                                                alpha,
                                                                beta,
                                                                in_dev.GetDeviceBuffer(),
                                                                out_dev.GetDeviceBuffer(),
                                                                out_indices_dev.GetDeviceBuffer(),
                                                                ws_dev.GetDeviceBuffer(),
                                                                in_elementwise_op_0,
                                                                acc_elementwise_op_0);

            if(!reduce_ptr->IsSupportedArgument(argument_ptr.get()))
                continue;

            std::string reduce_name = reduce_ptr->GetTypeString();

            auto invoker_ptr = reduce_ptr->MakeInvokerPointer();

            float avg_time = invoker_ptr->Run(argument_ptr.get(), nrepeat);

            std::size_t num_bytes =
                invariant_total_length * reduce_total_length * sizeof(InDataType) +
                invariant_total_length * sizeof(OutDataType);

            float gb_per_sec = num_bytes / 1.E6 / avg_time;

            std::cout << "Perf: " << avg_time << " ms, " << gb_per_sec << " GB/s, " << reduce_name
                      << std::endl;

            if(gb_per_sec > best_gb_per_sec)
            {
                best_avg_time   = avg_time;
                best_gb_per_sec = gb_per_sec;
            }

            if(do_verification)
            {
                out_dev.FromDevice(out.mData.data());
                ck::utils::check_err(out.mData, out_ref.mData);

                if(NeedIndices)
                {
                    out_indices_dev.FromDevice(out_indices.mData.data());
                    ck::utils::check_err(out_indices.mData, out_indices_ref.mData);
                    ;
                };

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "out_host  : ", out_ref.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "out_device: ", out.mData, ",") << std::endl;
                };
            };

            if(do_dumpout)
            {
                dumpBufferToFile("dump_in.bin", in.mData.data(), in.mDesc.GetElementSize());
                dumpBufferToFile("dump_out.bin", out.mData.data(), out.mDesc.GetElementSize());
                dumpBufferToFile(
                    "dump_out_host.bin", out_ref.mData.data(), out_ref.mDesc.GetElementSize());
                if(NeedIndices)
                {
                    dumpBufferToFile("dump_indices.bin",
                                     out_indices.mData.data(),
                                     out_indices.mDesc.GetElementSize());
                    dumpBufferToFile("dump_indices_host.bin",
                                     out_indices_ref.mData.data(),
                                     out_indices_ref.mDesc.GetElementSize());
                };
            };
        };

        for(auto& reduce_ptr : reduce1_ptrs)
        {
            auto wsSizeInBytes = reduce_ptr->GetWorkspaceSizeInBytes(i_inLengths, reduceDims);

            DeviceMem ws_dev(wsSizeInBytes);

            InElementwiseOperation_1 in_elementwise_op_1(static_cast<int32_t>(reduce_total_length));
            AccElementwiseOperation_1 acc_elementwise_op_1(
                static_cast<int32_t>(reduce_total_length));

            auto argument_ptr = reduce_ptr->MakeArgumentPointer(i_inLengths,
                                                                i_inStrides,
                                                                i_outLengths,
                                                                i_outStrides,
                                                                reduceDims,
                                                                alpha,
                                                                beta,
                                                                in_dev.GetDeviceBuffer(),
                                                                out_dev.GetDeviceBuffer(),
                                                                out_indices_dev.GetDeviceBuffer(),
                                                                ws_dev.GetDeviceBuffer(),
                                                                in_elementwise_op_1,
                                                                acc_elementwise_op_1);

            if(!reduce_ptr->IsSupportedArgument(argument_ptr.get()))
                continue;

            std::string reduce_name = reduce_ptr->GetTypeString();

            auto invoker_ptr = reduce_ptr->MakeInvokerPointer();

            float avg_time = invoker_ptr->Run(argument_ptr.get(), nrepeat);

            std::size_t num_bytes =
                invariant_total_length * reduce_total_length * sizeof(InDataType) +
                invariant_total_length * sizeof(OutDataType);

            std::vector<int> inLengths2 = reduce_ptr->GetWorkspace2dLengths(argument_ptr.get());
            std::vector<int> inStrides2{inLengths2[1], 1};

            for(auto& reduce2_ptr : reduce2_ptrs)
            {
                InElementwiseOperation_2 in_elementwise_op_2(
                    static_cast<int32_t>(reduce_total_length));
                AccElementwiseOperation_2 acc_elementwise_op_2(
                    static_cast<int32_t>(reduce_total_length));

                auto argument2_ptr =
                    reduce2_ptr->MakeArgumentPointer(inLengths2,
                                                     inStrides2,
                                                     i_outLengths,
                                                     i_outStrides,
                                                     reduceDims,
                                                     alpha,
                                                     beta,
                                                     ws_dev.GetDeviceBuffer(),
                                                     out_dev.GetDeviceBuffer(),
                                                     out_indices_dev.GetDeviceBuffer(),
                                                     ws_dev.GetDeviceBuffer(),
                                                     in_elementwise_op_2,
                                                     acc_elementwise_op_2);

                if(!reduce2_ptr->IsSupportedArgument(argument2_ptr.get()))
                    continue;

                std::string reduce2_name = reduce2_ptr->GetTypeString();

                auto invoker2_ptr = reduce2_ptr->MakeInvokerPointer();

                float avg_time_2 = invoker2_ptr->Run(argument2_ptr.get(), nrepeat);

                std::size_t num_bytes_2 =
                    static_cast<size_t>(inLengths2[0]) * inLengths2[1] * sizeof(AccDataType);

                float gb_per_sec = (num_bytes + num_bytes_2) / 1.E6 / (avg_time + avg_time_2);

                std::cout << "Perf: " << (avg_time + avg_time_2) << " ms, " << gb_per_sec
                          << " GB/s, " << reduce_name << " => " << reduce2_name << std::endl;

                if(gb_per_sec > best_gb_per_sec)
                {
                    best_avg_time   = avg_time + avg_time_2;
                    best_gb_per_sec = gb_per_sec;
                }

                if(do_verification)
                {
                    out_dev.FromDevice(out.mData.data());
                    ck::utils::check_err(out.mData, out_ref.mData);

                    if(NeedIndices)
                    {
                        out_indices_dev.FromDevice(out_indices.mData.data());
                        ck::utils::check_err(out_indices.mData, out_indices_ref.mData);
                        ;
                    };

                    if(do_log)
                    {
                        LogRangeAsType<float>(std::cout << "out_host  : ", out_ref.mData, ",")
                            << std::endl;
                        LogRangeAsType<float>(std::cout << "out_device: ", out.mData, ",")
                            << std::endl;
                    }
                }

                if(do_dumpout)
                {
                    dumpBufferToFile("dump_in.bin", in.mData.data(), in.mDesc.GetElementSize());
                    dumpBufferToFile("dump_out.bin", out.mData.data(), out.mDesc.GetElementSize());
                    dumpBufferToFile(
                        "dump_out_host.bin", out_ref.mData.data(), out_ref.mDesc.GetElementSize());
                    if(NeedIndices)
                    {
                        dumpBufferToFile("dump_indices.bin",
                                         out_indices.mData.data(),
                                         out_indices.mDesc.GetElementSize());
                        dumpBufferToFile("dump_indices_host.bin",
                                         out_indices_ref.mData.data(),
                                         out_indices_ref.mDesc.GetElementSize());
                    };
                };
            };
        };

        std::cout << "Best Perf: " << best_avg_time << " ms, " << best_gb_per_sec << " GB/s"
                  << std::endl;
    }
    else
    {
        std::cout << "The requested reduction operation is not supported, please check !!!"
                  << std::endl;
    };
};

template <typename InDataType, typename AccDataType, typename OutDataType>
void profile_reduce_impl(bool do_verification,
                         int init_method,
                         bool do_log,
                         bool do_dumpout,
                         int nrepeat,
                         const std::vector<size_t>& inLengths,
                         const std::vector<int>& reduceDims,
                         ReduceTensorOp ReduceOpId,
                         NanPropagation NanOpt,
                         ReduceTensorIndices IndicesOpt,
                         float alpha,
                         float beta)
{
    bool matched = false;

    using tuple_of_description_instances =
        tensor_operation::device::device_reduce_instance::reduce_description_instances;

    const auto tuple_object = tuple_of_description_instances{};

    static_for<0, std::tuple_size<tuple_of_description_instances>::value, 1>{}([&](auto i) {
        if(matched)
            return;

        using descType = remove_cvref_t<decltype(std::get<i>(tuple_object))>;

        if(!description_match(
               descType{}, inLengths.size(), reduceDims, ReduceOpId, NanOpt, IndicesOpt))
            return;

        profile_reduce_impl_impl<InDataType,
                                 AccDataType,
                                 OutDataType,
                                 descType::Rank_,
                                 descType::NumReduceDim_,
                                 static_cast<ReduceTensorOp>(descType::ReduceOpId_),
                                 static_cast<NanPropagation>(descType::NanOpt_),
                                 static_cast<ReduceTensorIndices>(descType::IndicesOpt_)>(
            do_verification,
            init_method,
            do_log,
            do_dumpout,
            nrepeat,
            inLengths,
            reduceDims,
            alpha,
            beta);

        matched = true;
    });
};

} // namespace profiler
} // namespace ck
