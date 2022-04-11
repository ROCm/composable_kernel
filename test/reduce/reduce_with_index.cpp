#include "getopt.h"
#include "device_reduce_instance.hpp"
#include "reduction_enums.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "host_reduction.hpp"
#include "check_err.hpp"
#include "reduce_util.hpp"

using namespace ck;

namespace {

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

constexpr int Rank = 4;

constexpr ReduceTensorOp ReduceOpId      = ReduceTensorOp::AMAX;
constexpr NanPropagation NanOpt          = NanPropagation::PROPAGATE_NAN;
constexpr bool PropagateNan              = false;
constexpr ReduceTensorIndices IndicesOpt = ReduceTensorIndices::FLATTENED_INDICES;
constexpr bool NeedIndices               = true;

template <typename InDataType,
          typename AccDataType,
          typename OutDataType,
          int Rank,
          int NumReduceDim>
bool test_reduce_with_index_impl(int init_method,
                                 const std::vector<size_t>& inLengths,
                                 const std::vector<int>& reduceDims,
                                 float alpha,
                                 float beta)
{
    using namespace ck::tensor_operation::device;
    using namespace ck::tensor_operation::device::device_reduce_instance;
    using namespace ck::host_reduce;

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

    // only used when the OutDataType is bhalf_t
    Tensor<float> out_ref_fp32(outLengths);
    Tensor<float> out_fp32(outLengths);

    auto inStrides  = in.mDesc.GetStrides();
    auto outStrides = out.mDesc.GetStrides();

    size_t invariant_total_length = out.mDesc.GetElementSize();
    size_t reduce_total_length    = in.mDesc.GetElementSize() / invariant_total_length;

    std::size_t num_thread = 1;

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
            out_ref.GenerateTensorValue(GeneratorTensor_3<InDataType>{-5.0, 5.0}, num_thread);
    }

    if(beta != 0.0f)
        for(size_t i = 0; i < out_ref.mDesc.GetElementSpace(); i++)
            out.mData[i] = out_ref.mData[i];

    // these buffers are usually provided by the user application
    DeviceMem in_dev(sizeof(InDataType) * in.mDesc.GetElementSpace());
    DeviceMem out_dev(sizeof(OutDataType) * out.mDesc.GetElementSpace());

    in_dev.ToDevice(in.mData.data());

    if(beta != 0.0f)
        out_dev.ToDevice(out.mData.data());

    size_t indicesSizeInBytes = NeedIndices ? out.mDesc.GetElementSize() * sizeof(int) : 0;

    DeviceMem out_indices_dev(indicesSizeInBytes);

    using InElementwiseOperation_0 =
        typename reduce_unary_operator<AccDataType, ReduceOpId, true, true>::InElementwiseOperation;
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

    add_device_reduce_instance_multiblock_partial_reduce<InDataType,
                                                         AccDataType,
                                                         OutDataType,
                                                         Rank,
                                                         NumReduceDim,
                                                         ReduceOpId,
                                                         NanOpt,
                                                         IndicesOpt>(reduce1_ptrs);

    add_device_reduce_instance_blockwise_second_call<AccDataType,
                                                     AccDataType,
                                                     OutDataType,
                                                     Rank,
                                                     NumReduceDim,
                                                     ReduceOpId,
                                                     NanOpt,
                                                     IndicesOpt>(reduce2_ptrs);

    if(reduce0_ptrs.empty() && reduce1_ptrs.empty())
    {
        throw std::runtime_error("Wrong! No device REDUCE instance found");
    };

    bool result = true;

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

    const auto i_inLengths  = to_int_vector(inLengths);
    const auto i_inStrides  = to_int_vector(inStrides);
    const auto i_outLengths = to_int_vector(outLengths);
    const auto i_outStrides = to_int_vector(outStrides);

    for(auto& reduce_ptr : reduce0_ptrs)
    {
        auto wsSizeInBytes = reduce_ptr->GetWorkspaceSizeInBytes(i_inLengths, reduceDims);

        DeviceMem ws_dev(wsSizeInBytes);

        InElementwiseOperation_0 in_elementwise_op_0(static_cast<int32_t>(reduce_total_length));
        AccElementwiseOperation_0 acc_elementwise_op_0(static_cast<int32_t>(reduce_total_length));

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

        auto invoker_ptr = reduce_ptr->MakeInvokerPointer();

        (void)invoker_ptr->Run(argument_ptr.get());

        out_dev.FromDevice(out.mData.data());

        bool single_result = true;

        if constexpr(std::is_same<OutDataType, ck::half_t>::value ||
                     std::is_same<OutDataType, ck::bhalf_t>::value)
        {
            reduce_util::to_f32_vector(out, out_fp32);
            reduce_util::to_f32_vector(out_ref, out_ref_fp32);
            single_result = ck::utils::check_err(
                out_fp32.mData, out_ref_fp32.mData, "Error: incorrect data result!");
        }
        else
        {
            single_result =
                ck::utils::check_err(out.mData, out_ref.mData, "Error: incorrect data result!");
        };

        if(NeedIndices)
        {
            out_indices_dev.FromDevice(out_indices.mData.data());
            single_result = single_result && ck::utils::check_err(out_indices_ref.mData,
                                                                  out_indices.mData,
                                                                  "Error: incorrect index result!");
        };

        if(!single_result)
        {
            std::cout << "Fail Info: " << reduce_ptr->GetTypeString() << std::endl;
            result = false;
        }
    };

    for(auto& reduce_ptr : reduce1_ptrs)
    {
        auto wsSizeInBytes = reduce_ptr->GetWorkspaceSizeInBytes(i_inLengths, reduceDims);

        DeviceMem ws_dev(wsSizeInBytes);

        InElementwiseOperation_1 in_elementwise_op_1(static_cast<int32_t>(reduce_total_length));
        AccElementwiseOperation_1 acc_elementwise_op_1(static_cast<int32_t>(reduce_total_length));

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

        (void)invoker_ptr->Run(argument_ptr.get());

        std::vector<int> inLengths2 = reduce_ptr->GetWorkspace2dLengths(argument_ptr.get());
        std::vector<int> inStrides2{inLengths2[1], 1};

        for(auto& reduce2_ptr : reduce2_ptrs)
        {
            InElementwiseOperation_2 in_elementwise_op_2(static_cast<int32_t>(reduce_total_length));
            AccElementwiseOperation_2 acc_elementwise_op_2(
                static_cast<int32_t>(reduce_total_length));

            auto argument2_ptr = reduce2_ptr->MakeArgumentPointer(inLengths2,
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

            (void)invoker2_ptr->Run(argument2_ptr.get());

            out_dev.FromDevice(out.mData.data());

            bool single_result = true;

            if constexpr(std::is_same<OutDataType, ck::half_t>::value ||
                         std::is_same<OutDataType, ck::bhalf_t>::value)
            {
                reduce_util::to_f32_vector(out, out_fp32);
                reduce_util::to_f32_vector(out_ref, out_ref_fp32);
                single_result = ck::utils::check_err(
                    out_fp32.mData, out_ref_fp32.mData, "Error: incorrect data result!");
            }
            else
            {
                single_result =
                    ck::utils::check_err(out.mData, out_ref.mData, "Error: incorrect data result!");
            };

            if(NeedIndices)
            {
                out_indices_dev.FromDevice(out_indices.mData.data());
                single_result =
                    single_result && ck::utils::check_err(out_indices_ref.mData,
                                                          out_indices.mData,
                                                          "Error: incorrect index result!");
            };

            if(!single_result)
            {
                std::cout << "Fail Info: " << reduce_ptr->GetTypeString() << " => "
                          << reduce2_ptr->GetTypeString() << std::endl;
                result = false;
            }
        };
    };

    return (result);
};

} // anonymous namespace

static struct option long_options[] = {{"inLengths", required_argument, nullptr, 'D'},
                                       {"reduceDimensions", required_argument, nullptr, 'R'},
                                       {"scales", required_argument, nullptr, 'S'},
                                       {"help", no_argument, nullptr, '?'},
                                       {nullptr, 0, nullptr, 0}};

class SimpleAppArgs
{
    template <typename T>
    static T getSingleValueFromString(const std::string& valueStr)
    {
        std::istringstream iss(valueStr);

        T ret;

        iss >> ret;

        return (ret);
    };

    template <typename T>
    static std::vector<T> getTypeValuesFromString(const char* cstr_values)
    {
        std::string valuesStr(cstr_values);

        std::vector<T> values;
        std::size_t pos = 0;
        std::size_t new_pos;

        new_pos = valuesStr.find(',', pos);
        while(new_pos != std::string::npos)
        {
            const std::string sliceStr = valuesStr.substr(pos, new_pos - pos);

            T val = getSingleValueFromString<T>(sliceStr);

            values.push_back(val);

            pos     = new_pos + 1;
            new_pos = valuesStr.find(',', pos);
        };

        std::string sliceStr = valuesStr.substr(pos);
        T val                = getSingleValueFromString<T>(sliceStr);

        values.push_back(val);

        return (values);
    };

    private:
    int option_index = 0;

    public:
    std::vector<size_t> inLengths;
    std::vector<int> reduceDims;
    std::vector<float> scales;

    int data_type;
    int init_method = 1;

    public:
    void show_usage(const char* cmd)
    {
        std::cout << "Usage of " << cmd << std::endl;
        std::cout << "--inLengths or -D, comma separated list of input tensor dimension lengths "
                     "(only 4-d tensor supported)"
                  << std::endl;
        std::cout << "--reduceDimensions or -R comma seperated list of dimension indexes to reduce "
                     "(only 1 or 3 or 4 dimensions supported)"
                  << std::endl;
        std::cout << "--scales or -S, comma separated two float values for alpha and beta"
                  << std::endl;
        std::cout << "Arg1 -- data type (1: fp32, 3: int8, 5: bp16, 6: fp64)" << std::endl;
        std::cout << "Arg2 -- init method(0=no init, 1=single integer value, 2=scope integer "
                     "value, 3=decimal value)"
                  << std::endl;
    };

    int processArgs(int argc, char* argv[])
    {
        unsigned int ch;

        while(1)
        {
            ch = getopt_long(argc, argv, "D:R:S:", long_options, &option_index);
            if(ch == -1)
                break;
            switch(ch)
            {
            case 'D':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                inLengths = getTypeValuesFromString<size_t>(optarg);
                break;
            case 'R':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                reduceDims = getTypeValuesFromString<int>(optarg);
                break;
            case 'S':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                scales = getTypeValuesFromString<float>(optarg);
                break;
            case '?':
                if(std::string(long_options[option_index].name) == "help")
                {
                    show_usage(argv[0]);
                    return (-1);
                };
                break;
            default: show_usage(argv[0]); return (-1);
            };
        };

        if(optind + 2 > argc)
            throw std::runtime_error("Invalid cmd-line arguments, more argumetns are needed!");

        data_type   = std::atoi(argv[optind++]);
        init_method = std::atoi(argv[optind]);

        if(scales.empty())
        {
            scales.push_back(1.0f);
            scales.push_back(0.0f);
        };

        if(inLengths.size() != 4 ||
           (reduceDims.size() != 1 && reduceDims.size() != 3 && reduceDims.size() != 4))
            return (-1);

        if(data_type != 0 && data_type != 1 && data_type != 3 && data_type != 5)
            return (-1);

        return (0);
    };
};

bool test_reduce_with_index(int data_type,
                            int init_method,
                            std::vector<int> reduceDims,
                            std::vector<size_t> inLengths,
                            float alpha,
                            float beta)
{
    bool result = true;

    if(data_type == 0)
    {
        switch(reduceDims.size())
        {
        case 1:
            result = test_reduce_with_index_impl<float, float, float, Rank, 1>(
                init_method, inLengths, reduceDims, alpha, beta);
            break;
        case 3:
            result = test_reduce_with_index_impl<float, float, float, Rank, 3>(
                init_method, inLengths, reduceDims, alpha, beta);
            break;
        case 4:
            result = test_reduce_with_index_impl<float, float, float, Rank, 4>(
                init_method, inLengths, reduceDims, alpha, beta);
            break;
        };
    }
    else if(data_type == 1)
    {
        switch(reduceDims.size())
        {
        case 1:
            result = test_reduce_with_index_impl<ck::half_t, ck::half_t, ck::half_t, Rank, 1>(
                init_method, inLengths, reduceDims, alpha, beta);
            break;
        case 3:
            result = test_reduce_with_index_impl<ck::half_t, ck::half_t, ck::half_t, Rank, 3>(
                init_method, inLengths, reduceDims, alpha, beta);
            break;
        case 4:
            result = test_reduce_with_index_impl<ck::half_t, ck::half_t, ck::half_t, Rank, 4>(
                init_method, inLengths, reduceDims, alpha, beta);
            break;
        };
    }
    else if(data_type == 3)
    {
        switch(reduceDims.size())
        {
        case 1:
            result = test_reduce_with_index_impl<int8_t, int8_t, int8_t, Rank, 1>(
                init_method, inLengths, reduceDims, alpha, beta);
            break;
        case 3:
            result = test_reduce_with_index_impl<int8_t, int8_t, int8_t, Rank, 3>(
                init_method, inLengths, reduceDims, alpha, beta);
            break;
        case 4:
            result = test_reduce_with_index_impl<int8_t, int8_t, int8_t, Rank, 4>(
                init_method, inLengths, reduceDims, alpha, beta);
            break;
        };
    }
    else if(data_type == 5)
    {
        switch(reduceDims.size())
        {
        case 1:
            result = test_reduce_with_index_impl<ck::bhalf_t, float, ck::bhalf_t, Rank, 1>(
                init_method, inLengths, reduceDims, alpha, beta);
            break;
        case 3:
            result = test_reduce_with_index_impl<ck::bhalf_t, float, ck::bhalf_t, Rank, 3>(
                init_method, inLengths, reduceDims, alpha, beta);
            break;
        case 4:
            result = test_reduce_with_index_impl<ck::bhalf_t, float, ck::bhalf_t, Rank, 4>(
                init_method, inLengths, reduceDims, alpha, beta);
            break;
        };
    }

    return (result);
};

int main(int argc, char* argv[])
{
    SimpleAppArgs args;

    bool result = true;

    if(argc == 1)
    {
        int data_type   = 1;
        int init_method = 2;
        std::vector<size_t> inLengths{64, 4, 280, 80};
        std::vector<std::vector<int>> v_reduceDims{
            {0, 1, 2, 3}, {0, 1, 2}, {1, 2, 3}, {0, 1, 3}, {0, 2, 3}, {0}, {1}, {2}, {3}};

        for(auto& reduceDims : v_reduceDims)
            result = result && test_reduce_with_index(
                                   data_type, init_method, reduceDims, inLengths, 1.0f, 0.0f);
    }
    else
    {
        if(args.processArgs(argc, argv) < 0)
        {
            throw std::runtime_error(
                "Invalid input arguments, test_reduce_with_index could not be executed!");
        };

        result = test_reduce_with_index(args.data_type,
                                        args.init_method,
                                        args.reduceDims,
                                        args.inLengths,
                                        args.scales[0],
                                        args.scales[1]);
    }

    std::cout << "test_reduce_with_index ..... " << (result ? "SUCCESS" : "FAILURE") << std::endl;

    return (result ? 0 : -1);
}
