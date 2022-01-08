#pragma once
#include "device_reduce.hpp"
#include "device_reduce_instance.hpp"
#include "reduction_enums.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_reduce_instance {

template <int rank, typename toReduceDims, int reduceOp, int nanOpt, int indicesOpt>
struct ReduceDescription
{
    static constexpr int rank_       = rank;
    static constexpr int reduceOp_   = reduceOp;
    static constexpr int nanOpt_     = nanOpt;
    static constexpr int indicesOpt_ = indicesOpt;

    using toReduceDims_ = toReduceDims;
};

using reduce_description_instances =
    std::tuple<ReduceDescription<4, Sequence<0, 1, 2>, 0, 0, 0>, // for ADD
               ReduceDescription<4, Sequence<0>, 0, 0, 0>,
               ReduceDescription<2, Sequence<1>, 0, 0, 0>,

               ReduceDescription<4, Sequence<0, 1, 2>, 5, 0, 0>, // for AVG
               ReduceDescription<4, Sequence<0>, 5, 0, 0>,
               ReduceDescription<2, Sequence<1>, 5, 0, 0>,

               ReduceDescription<4, Sequence<0, 1, 2>, 7, 0, 0>, // for NORM2
               ReduceDescription<4, Sequence<0>, 7, 0, 0>,
               ReduceDescription<2, Sequence<1>, 7, 0, 0>,

               ReduceDescription<4, Sequence<0, 1, 2>, 2, 0, 0>, // for MIN
               ReduceDescription<4, Sequence<0>, 2, 0, 0>,
               ReduceDescription<2, Sequence<1>, 2, 0, 0>,
               ReduceDescription<4, Sequence<0, 1, 2>, 3, 0, 0>, // for MAX
               ReduceDescription<4, Sequence<0>, 3, 0, 0>,
               ReduceDescription<2, Sequence<1>, 3, 0, 0>,
               ReduceDescription<4, Sequence<0, 1, 2>, 4, 0, 0>, // for AMAX
               ReduceDescription<4, Sequence<0>, 4, 0, 0>,
               ReduceDescription<2, Sequence<1>, 4, 0, 0>,

               ReduceDescription<4, Sequence<0, 1, 2>, 2, 0, 1>, // for MIN
               ReduceDescription<4, Sequence<0>, 2, 0, 1>,
               ReduceDescription<2, Sequence<1>, 2, 0, 1>,
               ReduceDescription<4, Sequence<0, 1, 2>, 3, 0, 1>, // for MAX
               ReduceDescription<4, Sequence<0>, 3, 0, 1>,
               ReduceDescription<2, Sequence<1>, 3, 0, 1>,
               ReduceDescription<4, Sequence<0, 1, 2>, 4, 0, 1>, // for AMAX
               ReduceDescription<4, Sequence<0>, 4, 0, 1>,
               ReduceDescription<2, Sequence<1>, 4, 0, 1>>;

template <typename DescriptionType>
bool description_match(const DescriptionType& description,
                       int rank,
                       const std::vector<int>& toReduceDims,
                       ReduceTensorOp_t reduceOp,
                       NanPropagation_t nanOpt,
                       ReduceTensorIndices_t indicesOpt)
{
    if(description.rank_ != rank || description.reduceOp_ != static_cast<int>(reduceOp) ||
       description.nanOpt_ != static_cast<int>(nanOpt) ||
       description.indicesOpt_ != static_cast<int>(indicesOpt))
        return (false);

    if(DescriptionType::toReduceDims_::Size() != toReduceDims.size())
        return (false);

    bool result = true;

    static_for<0, DescriptionType::toReduceDims_::Size(), 1>{}([&](auto i) {
        if(DescriptionType::toReduceDims_::At(i) != toReduceDims[i])
            result = false;
    });

    return (result);
};

} // namespace device_reduce_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

namespace ck {
namespace profiler {

template <int rank, typename toReduceDims>
static std::vector<int> get_toReduce_dims()
{
    std::vector<int> resDims;

    static_for<0, toReduceDims::Size(), 1>{}(
        [&](auto i) { resDims.push_back(toReduceDims::At(i)); });

    return (resDims);
};

template <int rank, typename toReduceDims>
static std::vector<int> get_invariant_dims()
{
    std::vector<int> resDims;
    unsigned int incFlag = 0;

    static_for<0, toReduceDims::Size(), 1>{}(
        [&](auto i) { incFlag = incFlag | (0x1 << toReduceDims::At(i)); });

    for(int dim = 0; dim < rank; dim++)
    {
        if(incFlag & (0x1 << dim))
            continue;
        resDims.push_back(dim);
    };

    return (resDims);
};

static std::vector<int> to_int_vector(const std::vector<size_t>& inData)
{
    std::vector<int> outData;

    for(auto elem : inData)
        outData.push_back(static_cast<int>(elem));

    return (outData);
};

static void check_indices(const Tensor<int>& ref, const Tensor<int>& result)
{
    bool has_error  = false;
    int error_count = 0;

    for(int i = 0; i < ref.mData.size(); ++i)
    {
        if(ref.mData[i] != result.mData[i])
        {
            std::cerr << std::endl
                      << "Indices different at position " << i << " (ref: " << ref.mData[i]
                      << ", result: " << result.mData[i] << ")" << std::endl;
            has_error = true;
            error_count++;
            if(error_count == 20)
                break;
        };
    }

    if(!has_error)
        std::cout << std::endl << "Indices result is completely acccurate!" << std::endl;
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

template <typename inType,
          typename compType,
          typename outType,
          int rank,
          typename toReduceDims_,
          ReduceTensorOp_t reduceOp,
          NanPropagation_t nanOpt,
          ReduceTensorIndices_t indicesOpt>
void profile_reduce_impl(bool do_verification,
                         int init_method,
                         bool do_log,
                         bool do_dumpout,
                         int nrepeat,
                         const std::vector<size_t>& inLengths,
                         float alpha,
                         float beta)
{
    using namespace ck::tensor_operation::device;
    using namespace ck::tensor_operation::device::device_reduce_instance;

    constexpr bool op_support_indices =
        (reduceOp == ReduceTensorOp_t::MIN || reduceOp == ReduceTensorOp_t::MAX ||
         reduceOp == ReduceTensorOp_t::AMAX);

    constexpr bool need_indices =
        (op_support_indices && (indicesOpt != ReduceTensorIndices_t::NO_INDICES));

    constexpr bool out_support_atomic_add =
        (std::is_same<outType, float>::value || std::is_same<outType, double>::value);
    constexpr bool op_support_atomic_add =
        !op_support_indices && reduceOp != ReduceTensorOp_t::NORM2;
    constexpr bool use_atomic_add = (out_support_atomic_add && op_support_atomic_add);

    // if input is half type, no reason to use float for indiced reduction operation and must use
    // float for non-indiced reduction operation for accuracy
    constexpr bool invalid_reduce_1 =
        std::is_same<inType, half_t>::value &&
        ((!op_support_indices && !std::is_same<compType, float>::value) ||
         (op_support_indices && !std::is_same<compType, half_t>::value));

    // if input is float type, no reason to use double for indiced reduction operation
    constexpr bool invalid_reduce_2 = std::is_same<inType, float>::value &&
                                      (op_support_indices && !std::is_same<compType, float>::value);

    // indices option can only be used when it is really needed
    constexpr bool invalid_reduce_3 =
        (!op_support_indices && indicesOpt != ReduceTensorIndices_t::NO_INDICES);

    constexpr bool invalid_reduce = (invalid_reduce_1 || invalid_reduce_2 || invalid_reduce_3);

    Tensor<inType> in(inLengths);

    const std::vector<int> invariantDims = get_invariant_dims<rank, toReduceDims_>();
    const std::vector<int> toReduceDims  = get_toReduce_dims<rank, toReduceDims_>();

    std::vector<size_t> outLengths;

    if(invariantDims.empty())
        outLengths.push_back(1);
    else
        for(auto dim : invariantDims)
            outLengths.push_back(inLengths[dim]);

    Tensor<outType> out_ref(outLengths);
    Tensor<outType> out(outLengths);
    Tensor<int> out_indices_ref(outLengths);
    Tensor<int> out_indices(outLengths);

    auto inStrides  = in.mDesc.GetStrides();
    auto outStrides = out.mDesc.GetStrides();

    size_t dim0_total_length = out.mDesc.GetElementSize();
    size_t dim1_total_length = in.mDesc.GetElementSize() / dim0_total_length;

    std::size_t num_thread = std::thread::hardware_concurrency();

    if(do_verification)
    {
        switch(init_method)
        {
        case 0:
            in.GenerateTensorValue(GeneratorTensor_1<inType>{}, num_thread);
            if(beta != 0.0f)
                out_ref.GenerateTensorValue(GeneratorTensor_1<inType>{}, num_thread);
            break;
        case 1:
            in.GenerateTensorValue(GeneratorTensor_2<inType>{-99, 99}, num_thread);
            if(beta != 0.0f)
                out_ref.GenerateTensorValue(GeneratorTensor_2<inType>{-5, 5}, num_thread);
            break;
        default:
            in.GenerateTensorValue(GeneratorTensor_2<inType>{1, 5}, num_thread);
            if(beta != 0.0f)
                out_ref.GenerateTensorValue(GeneratorTensor_2<inType>{1, 5}, num_thread);
        }

        if(beta != 0.0f)
            for(size_t i = 0; i < out_ref.mDesc.GetElementSpace(); i++)
                out.mData[i] = out_ref.mData[i];
    };

    // these buffers are usually provided by the user application
    DeviceMem in_dev(sizeof(inType) * in.mDesc.GetElementSpace());
    DeviceMem out_dev(sizeof(outType) * out.mDesc.GetElementSpace());

    in_dev.ToDevice(in.mData.data());

    if(beta != 0.0f)
        out_dev.ToDevice(out.mData.data());

    size_t indicesSizeInBytes = need_indices ? out.mDesc.GetElementSize() * sizeof(int) : 0;

    DeviceMem out_indices_dev(indicesSizeInBytes);

    float best_avg_time   = 0;
    float best_gb_per_sec = 0;

    using DeviceReduceInstPtr  = DeviceReducePtr<inType,
                                                compType,
                                                outType,
                                                rank,
                                                toReduceDims_,
                                                reduceOp,
                                                nanOpt,
                                                indicesOpt>;
    using DeviceReduceInstPtr2 = DeviceReducePtr<compType,
                                                 compType,
                                                 outType,
                                                 rank,
                                                 toReduceDims_,
                                                 reduceOp,
                                                 nanOpt,
                                                 indicesOpt>;

    std::vector<DeviceReduceInstPtr> reduce_ptrs;
    std::vector<DeviceReduceInstPtr2> reduce2_ptrs;

    if constexpr(!invalid_reduce)
    {
        add_device_reduce_instance_threadwise<inType,
                                              compType,
                                              outType,
                                              rank,
                                              toReduceDims_,
                                              reduceOp,
                                              nanOpt,
                                              indicesOpt>(reduce_ptrs);

        add_device_reduce_instance_blockwise<inType,
                                             compType,
                                             outType,
                                             rank,
                                             toReduceDims_,
                                             reduceOp,
                                             nanOpt,
                                             indicesOpt>(reduce_ptrs);

        if constexpr(use_atomic_add)
            add_device_reduce_instance_multiblock_atomic_add<inType,
                                                             compType,
                                                             outType,
                                                             rank,
                                                             toReduceDims_,
                                                             reduceOp,
                                                             nanOpt,
                                                             indicesOpt>(reduce_ptrs);
        else
            add_device_reduce_instance_multiblock_two_call<inType,
                                                           compType,
                                                           outType,
                                                           rank,
                                                           toReduceDims_,
                                                           reduceOp,
                                                           nanOpt,
                                                           indicesOpt>(reduce_ptrs);

        // used for secondary reduction
        if constexpr(!use_atomic_add)
            add_device_reduce_instance_blockwise_second_call<compType,
                                                             compType,
                                                             outType,
                                                             rank,
                                                             toReduceDims_,
                                                             reduceOp,
                                                             nanOpt,
                                                             indicesOpt>(reduce2_ptrs);
    };

    if(reduce_ptrs.empty())
    {
        throw std::runtime_error("Wrong! No device REDUCE instance found");
    };

    if(do_verification)
    {
        ReductionHost<inType, compType, outType> hostReduce(
            reduceOp, nanOpt, indicesOpt, in.mDesc, out_ref.mDesc, invariantDims, toReduceDims);

        hostReduce.Run(
            alpha, in.mData.data(), beta, out_ref.mData.data(), out_indices_ref.mData.data());
    };

    for(auto& reduce_ptr : reduce_ptrs)
    {
        const auto i_inLengths  = to_int_vector(inLengths);
        const auto i_inStrides  = to_int_vector(inStrides);
        const auto i_outLengths = to_int_vector(outLengths);
        const auto i_outStrides = to_int_vector(outStrides);

        auto wsSizeInBytes = reduce_ptr->getWorkspaceSize(i_inLengths);

        DeviceMem ws_dev(wsSizeInBytes);

        auto argument_ptr = reduce_ptr->MakeArgumentPointer(i_inLengths,
                                                            i_inStrides,
                                                            i_outLengths,
                                                            i_outStrides,
                                                            alpha,
                                                            beta,
                                                            in_dev.GetDeviceBuffer(),
                                                            out_dev.GetDeviceBuffer(),
                                                            out_indices_dev.GetDeviceBuffer(),
                                                            ws_dev.GetDeviceBuffer());

        if(!reduce_ptr->IsSupportedArgument(argument_ptr.get()))
            continue;

        std::string reduce_name = reduce_ptr->GetTypeString();

        auto invoker_ptr = reduce_ptr->MakeInvokerPointer();

        float avg_time = invoker_ptr->Run(argument_ptr.get(), nrepeat);

        std::size_t num_bytes = dim0_total_length * dim1_total_length * sizeof(inType) +
                                dim0_total_length * sizeof(outType);

        if(reduce_ptr->hasFurtherCall())
        {
            std::vector<int> inLengths2 = reduce_ptr->getWorkspace2dLengths(argument_ptr.get());
            std::vector<int> inStrides2{inLengths2[1], 1};
            int origReduceLen = reduce_ptr->getOrigReduceLength(argument_ptr.get());

            for(auto& reduce2_ptr : reduce2_ptrs)
            {
                auto argument2_ptr =
                    reduce2_ptr->MakeArgumentPointer(inLengths2,
                                                     inStrides2,
                                                     i_outLengths,
                                                     i_outStrides,
                                                     alpha,
                                                     beta,
                                                     ws_dev.GetDeviceBuffer(),
                                                     out_dev.GetDeviceBuffer(),
                                                     out_indices_dev.GetDeviceBuffer(),
                                                     ws_dev.GetDeviceBuffer());

                if(!reduce2_ptr->IsSupportedArgument(argument2_ptr.get()))
                    continue;

                std::string reduce2_name = reduce2_ptr->GetTypeString();

                reduce2_ptr->setOrigReduceLength(argument2_ptr.get(), origReduceLen);

                auto invoker2_ptr = reduce2_ptr->MakeInvokerPointer();

                float avg_time_2 = invoker2_ptr->Run(argument2_ptr.get(), nrepeat);

                std::size_t num_bytes_2 =
                    static_cast<size_t>(inLengths2[0]) * inLengths2[1] * sizeof(compType);

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
                    check_error(out_ref, out);

                    if(need_indices)
                    {
                        out_indices_dev.FromDevice(out_indices.mData.data());
                        check_indices(out_indices_ref, out_indices);
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
                    if(need_indices)
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
        }
        else
        {
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
                check_error(out_ref, out);

                if(need_indices)
                {
                    out_indices_dev.FromDevice(out_indices.mData.data());
                    check_indices(out_indices_ref, out_indices);
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
                if(need_indices)
                {
                    dumpBufferToFile("dump_indices.bin",
                                     out_indices.mData.data(),
                                     out_indices.mDesc.GetElementSize());
                    dumpBufferToFile("dump_indices_host.bin",
                                     out_indices_ref.mData.data(),
                                     out_indices_ref.mDesc.GetElementSize());
                };
            };
        }
    };

    std::cout << "Best Perf: " << best_avg_time << " ms, " << best_gb_per_sec << " GB/s"
              << std::endl;
};

template <typename inType, typename compType, typename outType>
void profile_reduce(bool do_verification,
                    int init_method,
                    bool do_log,
                    bool do_dumpout,
                    int nrepeat,
                    const std::vector<size_t>& inLengths,
                    const std::vector<int>& toReduceDims,
                    ReduceTensorOp_t reduceOp,
                    NanPropagation_t nanOpt,
                    ReduceTensorIndices_t indicesOpt,
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
               descType{}, inLengths.size(), toReduceDims, reduceOp, nanOpt, indicesOpt))
            return;

        profile_reduce_impl<inType,
                            compType,
                            outType,
                            descType::rank_,
                            typename descType::toReduceDims_,
                            static_cast<ReduceTensorOp_t>(descType::reduceOp_),
                            static_cast<NanPropagation_t>(descType::nanOpt_),
                            static_cast<ReduceTensorIndices_t>(descType::indicesOpt_)>(
            do_verification, init_method, do_log, do_dumpout, nrepeat, inLengths, alpha, beta);

        matched = true;
    });
};

} // namespace profiler
} // namespace ck
