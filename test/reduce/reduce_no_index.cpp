#include "getopt.h"
#include "device_reduce_instance.hpp"
#include "reduction_enums.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "host_reduction.hpp"
#include "test_util.hpp"
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

constexpr ReduceTensorOp_t ReduceOpId      = ReduceTensorOp_t::AVG;
constexpr NanPropagation_t NanOpt          = NanPropagation_t::PROPAGATE_NAN;
constexpr bool PropagateNan                = false;
constexpr ReduceTensorIndices_t IndicesOpt = ReduceTensorIndices_t::NO_INDICES;
constexpr bool NeedIndices                 = false;

template <typename InDataType,
          typename AccDataType,
          typename OutDataType,
          int Rank,
          int NumReduceDim>
bool test_reduce_no_index(int init_method,
                          const std::vector<size_t>& inLengths,
                          const std::vector<int>& reduceDims,
                          float alpha,
                          float beta)
{
    using namespace ck::tensor_operation::device;
    using namespace ck::tensor_operation::device::device_reduce_instance;
    using namespace ck::host_reduce;

    constexpr bool out_support_atomic_add = std::is_same<OutDataType, float>::value;
    constexpr bool op_support_atomic_add  = true;
    constexpr bool use_atomic_add         = (out_support_atomic_add && op_support_atomic_add);

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

    // only used when the OutDataType is bhalf_t
    Tensor<float> out_ref_fp32(outLengths);
    Tensor<float> out_fp32(outLengths);

    auto inStrides  = in.mDesc.GetStrides();
    auto outStrides = out.mDesc.GetStrides();

    size_t invariant_total_length = out.mDesc.GetElementSize();
    size_t reduce_total_length    = in.mDesc.GetElementSize() / invariant_total_length;

    std::size_t num_thread = std::thread::hardware_concurrency();

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

    bool result = true;

    using hInDataType  = typename type_mapping<InDataType>::OutType;
    using hOutDataType = typename type_mapping<OutDataType>::OutType;
    using hAccDataType = typename type_mapping<AccDataType>::OutType;

    ReductionHost<hInDataType,
                  hAccDataType,
                  hOutDataType,
                  ReduceOpId,
                  Rank,
                  NumReduceDim,
                  PropagateNan,
                  NeedIndices>
        hostReduce(in.mDesc, out_ref.mDesc, invariantDims, reduceDims);

    hostReduce.Run(alpha,
                   reinterpret_cast<const hInDataType*>(in.mData.data()),
                   beta,
                   reinterpret_cast<hOutDataType*>(out_ref.mData.data()),
                   nullptr);

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
                                                            nullptr,
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
            single_result = test_util::check_err(
                out_fp32.mData, out_ref_fp32.mData, "Error: incorrect data result!");
        }
        else
        {
            single_result =
                test_util::check_err(out.mData, out_ref.mData, "Error: incorrect data result!");
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
                                                            nullptr,
                                                            ws_dev.GetDeviceBuffer(),
                                                            in_elementwise_op_1,
                                                            acc_elementwise_op_1);

        if(!reduce_ptr->IsSupportedArgument(argument_ptr.get()))
            continue;

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
                                                                  nullptr,
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
                single_result = test_util::check_err(
                    out_fp32.mData, out_ref_fp32.mData, "Error: incorrect data result!");
            }
            else
            {
                single_result =
                    test_util::check_err(out.mData, out_ref.mData, "Error: incorrect data result!");
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
        std::cout << "Arg1 -- data type (0: fp16, 1: fp32, 3: int8, 5: bp16, 6: fp64)" << std::endl;
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

int main(int argc, char* argv[])
{
    SimpleAppArgs args;

    if(args.processArgs(argc, argv) < 0)
    {
        throw std::runtime_error(
            "Invalid input arguments, test_reduce_no_index could not be executed!");
    };

    bool result = true;

    if(args.data_type == 0)
    {
        switch(args.reduceDims.size())
        {
        case 1:
            result = test_reduce_no_index<float, float, float, Rank, 1>(
                args.init_method, args.inLengths, args.reduceDims, args.scales[0], args.scales[1]);
            break;
        case 3:
            result = test_reduce_no_index<float, float, float, Rank, 3>(
                args.init_method, args.inLengths, args.reduceDims, args.scales[0], args.scales[1]);
            break;
        case 4:
            result = test_reduce_no_index<float, float, float, Rank, 4>(
                args.init_method, args.inLengths, args.reduceDims, args.scales[0], args.scales[1]);
            break;
        };
    }
    else if(args.data_type == 1)
    {
        switch(args.reduceDims.size())
        {
        case 1:
            result = test_reduce_no_index<ck::half_t, float, ck::half_t, Rank, 1>(
                args.init_method, args.inLengths, args.reduceDims, args.scales[0], args.scales[1]);
            break;
        case 3:
            result = test_reduce_no_index<ck::half_t, float, ck::half_t, Rank, 3>(
                args.init_method, args.inLengths, args.reduceDims, args.scales[0], args.scales[1]);
            break;
        case 4:
            result = test_reduce_no_index<ck::half_t, float, ck::half_t, Rank, 4>(
                args.init_method, args.inLengths, args.reduceDims, args.scales[0], args.scales[1]);
            break;
        };
    }
    else if(args.data_type == 3)
    {
        switch(args.reduceDims.size())
        {
        case 1:
            result = test_reduce_no_index<int8_t, int32_t, int8_t, Rank, 1>(
                args.init_method, args.inLengths, args.reduceDims, args.scales[0], args.scales[1]);
            break;
        case 3:
            result = test_reduce_no_index<int8_t, int32_t, int8_t, Rank, 3>(
                args.init_method, args.inLengths, args.reduceDims, args.scales[0], args.scales[1]);
            break;
        case 4:
            result = test_reduce_no_index<int8_t, int32_t, int8_t, Rank, 4>(
                args.init_method, args.inLengths, args.reduceDims, args.scales[0], args.scales[1]);
            break;
        };
    }
    else if(args.data_type == 5)
    {
        switch(args.reduceDims.size())
        {
        case 1:
            result = test_reduce_no_index<ck::bhalf_t, float, ck::bhalf_t, Rank, 1>(
                args.init_method, args.inLengths, args.reduceDims, args.scales[0], args.scales[1]);
            break;
        case 3:
            result = test_reduce_no_index<ck::bhalf_t, float, ck::bhalf_t, Rank, 3>(
                args.init_method, args.inLengths, args.reduceDims, args.scales[0], args.scales[1]);
            break;
        case 4:
            result = test_reduce_no_index<ck::bhalf_t, float, ck::bhalf_t, Rank, 4>(
                args.init_method, args.inLengths, args.reduceDims, args.scales[0], args.scales[1]);
            break;
        };
    }

    std::cout << "test_reduce_no_index ..... " << (result ? "SUCCESS" : "FAILURE") << std::endl;

    return (result ? 0 : -1);
}
