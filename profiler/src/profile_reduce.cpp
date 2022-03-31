#include <iostream>
#include <fstream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <getopt.h>

#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "device_tensor.hpp"
#include "reduction_enums.hpp"

#include "profile_reduce_impl.hpp"

using namespace std;

using ck::NanPropagation;
using ck::ReduceTensorIndices;
using ck::ReduceTensorOp;

static struct option long_options[] = {{"inLengths", required_argument, nullptr, 'D'},
                                       {"reduceDims", required_argument, nullptr, 'R'},
                                       {"reduceOp", required_argument, nullptr, 'O'},
                                       {"compType", required_argument, nullptr, 'C'},
                                       {"outType", required_argument, nullptr, 'W'},
                                       {"nanOpt", required_argument, nullptr, 'N'},
                                       {"indicesOpt", required_argument, nullptr, 'I'},
                                       {"scales", required_argument, nullptr, 'S'},
                                       {"half", no_argument, nullptr, '?'},
                                       {"double", no_argument, nullptr, '?'},
                                       {"int8", no_argument, nullptr, '?'},
                                       {"bf16", no_argument, nullptr, '?'},
                                       {"dumpout", required_argument, nullptr, 'o'},
                                       {"verify", required_argument, nullptr, 'v'},
                                       {"log", required_argument, nullptr, 'l'},
                                       {"help", no_argument, nullptr, '?'},
                                       {nullptr, 0, nullptr, 0}};

template <typename T>
static T getSingleValueFromString(const string& valueStr)
{
    std::istringstream iss(valueStr);

    T val;

    iss >> val;

    return (val);
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
}

enum struct AppDataType
{
    appHalf     = 0,
    appFloat    = 1,
    appInt32    = 2,
    appInt8     = 3,
    appInt8x4   = 4,
    appBFloat16 = 5,
    appDouble   = 6,
};

static void check_reduce_dims(const int rank, const std::vector<int>& reduceDims)
{
    for(auto dim : reduceDims)
    {
        if(dim < 0 || dim >= rank)
            throw std::runtime_error("Invalid dimension index specified for Reducing");
    };

    unsigned int flag = 0;

    for(auto dim : reduceDims)
    {
        if(flag & (0x1 << dim))
            throw std::runtime_error("All toReduce dimensions should be different!");
        flag = flag | (0x1 << dim);
    };
};

class AppArgs
{
    private:
    int option_index = 0;

    public:
    bool use_half   = false;
    bool use_double = false;
    bool use_int8   = false;
    bool use_bf16   = false;

    std::vector<size_t> inLengths;
    std::vector<size_t> outLengths;
    std::vector<int> reduceDims;

    std::vector<float> scales;

    ReduceTensorOp reduceOp = ReduceTensorOp::ADD;
    AppDataType compTypeId  = AppDataType::appFloat;
    AppDataType outTypeId   = AppDataType::appFloat;

    bool compType_assigned = false;
    bool outType_assigned  = false;

    NanPropagation nanOpt          = NanPropagation::NOT_PROPAGATE_NAN;
    ReduceTensorIndices indicesOpt = ReduceTensorIndices::NO_INDICES;
    bool do_log                    = false;
    bool do_verification           = false;
    bool do_dumpout                = false;

    int init_method;
    int nrepeat;

    bool need_indices = false;

    AppArgs()  = default;
    ~AppArgs() = default;

    void show_usage(const char* cmd)
    {
        std::cout << "Usage of " << cmd << std::endl;
        std::cout << "--inLengths or -D, comma separated list of input tensor dimension lengths"
                  << std::endl;
        std::cout << "--reduceDims or -R, comma separated list of to-reduce dimensions"
                  << std::endl;
        std::cout << "--reduceOp or -O, enum value indicating the reduction operations"
                  << std::endl;
        std::cout << "--compType or -C, enum value indicating the type of accumulated values used "
                     "during the reduction"
                  << std::endl;
        std::cout << "--outType or -W, optional enum value indicating the type of the reduced "
                     "output, which could be float when the input data is half"
                  << std::endl;
        std::cout << "--nanOpt or -N, enum value indicates the selection for NanOpt" << std::endl;
        std::cout << "--indicesOpt or -I, enum value indicates the selection for IndicesOpt"
                  << std::endl;
        std::cout << "--scales or -S, comma separated two float values for alpha and beta"
                  << std::endl;
        std::cout << "--half, use fp16 for the input and output tensor data types" << std::endl;
        std::cout << "--double, use fp64 for the input and output tensor data types" << std::endl;
        std::cout << "--int8, use int8 for the input and output tensor data types" << std::endl;
        std::cout << "--bf16, use bfloat16 for the input and output tensor data types" << std::endl;
        std::cout << "--verify or -v, 1/0 to indicate whether to verify the reduction result by "
                     "comparing with the host-based reduction"
                  << std::endl;
        std::cout << "--dumpout or -o, 1/0 to indicate where to save the reduction result to files "
                     "for further analysis"
                  << std::endl;
        std::cout << "--log or -l, 1/0 to indicate whether to log some information" << std::endl;
    };

    int processArgs(int argc, char* argv[])
    {
        unsigned int ch;

        optind++; // to skip the "reduce" module name

        while(1)
        {
            ch = getopt_long(argc, argv, "D:R:O:C:W:N:I:S:v:o:l:", long_options, &option_index);
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
            case 'O':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                reduceOp = static_cast<ReduceTensorOp>(std::atoi(optarg));
                break;
            case 'C':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                compTypeId        = static_cast<AppDataType>(std::atoi(optarg));
                compType_assigned = true;
                break;
            case 'W':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                outTypeId        = static_cast<AppDataType>(std::atoi(optarg));
                outType_assigned = true;
                break;
            case 'N':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                nanOpt = static_cast<NanPropagation>(std::atoi(optarg));
                break;
            case 'I':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                indicesOpt = static_cast<ReduceTensorIndices>(std::atoi(optarg));
                break;
            case 'S':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                scales = getTypeValuesFromString<float>(optarg);

                if(scales.size() != 2)
                    throw std::runtime_error("Invalid option format!");
                break;
            case 'v':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                do_verification = static_cast<bool>(std::atoi(optarg));
                break;
            case 'o':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                do_dumpout = static_cast<bool>(std::atoi(optarg));
                break;
            case 'l':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                do_log = static_cast<bool>(std::atoi(optarg));
                break;
            case '?':
                if(std::string(long_options[option_index].name) == "half")
                    use_half = true;
                else if(std::string(long_options[option_index].name) == "double")
                    use_double = true;
                else if(std::string(long_options[option_index].name) == "int8")
                    use_int8 = true;
                else if(std::string(long_options[option_index].name) == "bf16")
                    use_bf16 = true;
                else if(std::string(long_options[option_index].name) == "help")
                {
                    show_usage(argv[0]);
                    return (-1);
                };
                break;

            default:
                show_usage(argv[0]);
                std::cerr << "Invalid cmd-line options!" << std::endl;
                return (-1);
            };
        };

        if(optind + 2 > argc)
            throw std::runtime_error("Invalid cmd-line arguments, more argumetns are needed!");

        init_method = std::atoi(argv[optind++]);
        nrepeat     = std::atoi(argv[optind]);

        if(scales.empty())
        {
            scales.push_back(1.0f);
            scales.push_back(0.0f);
        };

        if(reduceOp == ReduceTensorOp::MIN || reduceOp == ReduceTensorOp::MAX ||
           reduceOp == ReduceTensorOp::AMAX)
        {
            if(indicesOpt != ReduceTensorIndices::NO_INDICES)
                need_indices = true;

            // for indexable operations, no need to assign compType and outType, just let them be
            // same as inType
            compType_assigned = false;
            outType_assigned  = false;
        };

        return (0);
    };

}; // end of class AppArgs

int profile_reduce(int argc, char* argv[])
{
    using namespace ck::profiler;

    AppArgs args;

    if(args.processArgs(argc, argv) < 0)
        return (-1);

    int rank = args.inLengths.size();

    check_reduce_dims(rank, args.reduceDims);

    if(args.reduceOp == ReduceTensorOp::MUL || args.reduceOp == ReduceTensorOp::NORM1)
        throw std::runtime_error("MUL and NORM1 are not supported by composable kernel!");

    if(args.use_half)
    {
        if(!args.compType_assigned)
            args.compTypeId = AppDataType::appHalf;

        if(args.outType_assigned &&
           (args.outTypeId != AppDataType::appHalf && args.outTypeId != AppDataType::appFloat))
            args.outTypeId = AppDataType::appFloat;

        if(!args.outType_assigned)
            args.outTypeId = AppDataType::appHalf;

        if(args.compTypeId == AppDataType::appHalf)
        {
            profile_reduce_impl<ck::half_t, ck::half_t, ck::half_t>(args.do_verification,
                                                                    args.init_method,
                                                                    args.do_log,
                                                                    args.do_dumpout,
                                                                    args.nrepeat,
                                                                    args.inLengths,
                                                                    args.reduceDims,
                                                                    args.reduceOp,
                                                                    args.nanOpt,
                                                                    args.indicesOpt,
                                                                    args.scales[0],
                                                                    args.scales[1]);
        }
        else if(args.compTypeId == AppDataType::appFloat)
        {
            profile_reduce_impl<ck::half_t, float, ck::half_t>(args.do_verification,
                                                               args.init_method,
                                                               args.do_log,
                                                               args.do_dumpout,
                                                               args.nrepeat,
                                                               args.inLengths,
                                                               args.reduceDims,
                                                               args.reduceOp,
                                                               args.nanOpt,
                                                               args.indicesOpt,
                                                               args.scales[0],
                                                               args.scales[1]);
        }
        else
            throw std::runtime_error("Invalid compType assignment!");
    }
    else if(args.use_double)
    {
        profile_reduce_impl<double, double, double>(args.do_verification,
                                                    args.init_method,
                                                    args.do_log,
                                                    args.do_dumpout,
                                                    args.nrepeat,
                                                    args.inLengths,
                                                    args.reduceDims,
                                                    args.reduceOp,
                                                    args.nanOpt,
                                                    args.indicesOpt,
                                                    args.scales[0],
                                                    args.scales[1]);
    }
    else if(args.use_int8)
    {
        if(!args.compType_assigned)
            args.compTypeId = AppDataType::appInt8;

        if(args.outType_assigned &&
           (args.outTypeId != AppDataType::appInt8 && args.outTypeId != AppDataType::appInt32))
            args.outTypeId = AppDataType::appInt32;

        if(!args.outType_assigned)
            args.outTypeId = AppDataType::appInt8;

        if(args.compTypeId == AppDataType::appInt8)
        {
            profile_reduce_impl<int8_t, int8_t, int8_t>(args.do_verification,
                                                        args.init_method,
                                                        args.do_log,
                                                        args.do_dumpout,
                                                        args.nrepeat,
                                                        args.inLengths,
                                                        args.reduceDims,
                                                        args.reduceOp,
                                                        args.nanOpt,
                                                        args.indicesOpt,
                                                        args.scales[0],
                                                        args.scales[1]);
        }
        else if(args.compTypeId == AppDataType::appInt32)
        {
            profile_reduce_impl<int8_t, int32_t, int8_t>(args.do_verification,
                                                         args.init_method,
                                                         args.do_log,
                                                         args.do_dumpout,
                                                         args.nrepeat,
                                                         args.inLengths,
                                                         args.reduceDims,
                                                         args.reduceOp,
                                                         args.nanOpt,
                                                         args.indicesOpt,
                                                         args.scales[0],
                                                         args.scales[1]);
        }
        else
            throw std::runtime_error("Invalid compType assignment!");
    }
    else if(args.use_bf16)
    {
        if(args.outType_assigned &&
           (args.outTypeId != AppDataType::appBFloat16 && args.outTypeId != AppDataType::appFloat))
            args.outTypeId = AppDataType::appFloat;

        if(!args.outType_assigned)
            args.outTypeId = AppDataType::appBFloat16;

        profile_reduce_impl<ck::bhalf_t, float, ck::bhalf_t>(args.do_verification,
                                                             args.init_method,
                                                             args.do_log,
                                                             args.do_dumpout,
                                                             args.nrepeat,
                                                             args.inLengths,
                                                             args.reduceDims,
                                                             args.reduceOp,
                                                             args.nanOpt,
                                                             args.indicesOpt,
                                                             args.scales[0],
                                                             args.scales[1]);
    }
    else
    {
        if(args.compTypeId == AppDataType::appFloat)
        {
            profile_reduce_impl<float, float, float>(args.do_verification,
                                                     args.init_method,
                                                     args.do_log,
                                                     args.do_dumpout,
                                                     args.nrepeat,
                                                     args.inLengths,
                                                     args.reduceDims,
                                                     args.reduceOp,
                                                     args.nanOpt,
                                                     args.indicesOpt,
                                                     args.scales[0],
                                                     args.scales[1]);
        }
        else if(args.compTypeId == AppDataType::appDouble)
        {
            profile_reduce_impl<float, double, float>(args.do_verification,
                                                      args.init_method,
                                                      args.do_log,
                                                      args.do_dumpout,
                                                      args.nrepeat,
                                                      args.inLengths,
                                                      args.reduceDims,
                                                      args.reduceOp,
                                                      args.nanOpt,
                                                      args.indicesOpt,
                                                      args.scales[0],
                                                      args.scales[1]);
        }
        else
            throw std::runtime_error("Invalid compType assignment!");
    };

    return (0);
};
