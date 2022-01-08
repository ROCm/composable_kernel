#include <iostream>
#include <fstream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <vector>
#include <stdexcept>
#include <half.hpp>
#include <getopt.h>

#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "device_tensor.hpp"
#include "host_generic_reduction.hpp"

#include "reduction_enums.hpp"

#include "profile_reduce.hpp"

using namespace std;

using ck::NanPropagation_t;
using ck::ReduceTensorIndices_t;
using ck::ReduceTensorOp_t;

static struct option long_options[] = {{"inLengths", required_argument, nullptr, 'D'},
                                       {"toReduceDims", required_argument, nullptr, 'R'},
                                       {"reduceOp", required_argument, nullptr, 'O'},
                                       {"compType", required_argument, nullptr, 'C'},
                                       {"outType", required_argument, nullptr, 'W'},
                                       {"nanOpt", required_argument, nullptr, 'N'},
                                       {"indicesOpt", required_argument, nullptr, 'I'},
                                       {"scales", required_argument, nullptr, 'S'},
                                       {"half", no_argument, nullptr, '?'},
                                       {"double", no_argument, nullptr, '?'},
                                       {"dumpout", required_argument, nullptr, 'o'},
                                       {"verify", required_argument, nullptr, 'v'},
                                       {"log", required_argument, nullptr, 'l'},
                                       {"help", no_argument, nullptr, '?'},
                                       {nullptr, 0, nullptr, 0}};

template <typename T>
static T getSingleValueFromString(const string& valueStr);

template <>
int getSingleValueFromString<int>(const string& valueStr)
{
    return (std::stoi(valueStr));
};

template <>
size_t getSingleValueFromString<size_t>(const string& valueStr)
{
    return (std::stol(valueStr));
};

template <>
float getSingleValueFromString<float>(const string& valueStr)
{
    return (std::stof(valueStr));
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

typedef enum
{
    appHalf     = 0,
    appFloat    = 1,
    appInt32    = 2,
    appInt8     = 3,
    appInt8x4   = 4,
    appBFloat16 = 5,
    appDouble   = 6,
} appDataType_t;

static void check_reduce_dims(const int rank, const std::vector<int>& toReduceDims)
{
    for(auto dim : toReduceDims)
    {
        if(dim < 0 || dim >= rank)
            throw std::runtime_error("Invalid dimension index specified for Reducing");
    };

    unsigned int flag = 0;

    for(auto dim : toReduceDims)
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

    std::vector<size_t> inLengths;
    std::vector<size_t> outLengths;
    std::vector<int> toReduceDims;

    std::vector<float> scales;

    ReduceTensorOp_t reduceOp = ReduceTensorOp_t::ADD;
    appDataType_t compTypeId  = appFloat;
    appDataType_t outTypeId   = appFloat;

    bool compType_assigned = false;
    bool outType_assigned  = false;

    NanPropagation_t nanOpt          = NanPropagation_t::NOT_PROPAGATE_NAN;
    ReduceTensorIndices_t indicesOpt = ReduceTensorIndices_t::NO_INDICES;
    bool do_log                      = false;
    bool do_verification             = false;
    bool do_dumpout                  = false;

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
        std::cout << "--toReduceDims or -R, comma separated list of to-reduce dimensions"
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

                toReduceDims = getTypeValuesFromString<int>(optarg);
                break;
            case 'O':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                reduceOp = static_cast<ReduceTensorOp_t>(std::atoi(optarg));
                break;
            case 'C':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                compTypeId        = static_cast<appDataType_t>(std::atoi(optarg));
                compType_assigned = true;
                break;
            case 'W':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                outTypeId        = static_cast<appDataType_t>(std::atoi(optarg));
                outType_assigned = true;
                break;
            case 'N':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                nanOpt = static_cast<NanPropagation_t>(std::atoi(optarg));
                break;
            case 'I':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                indicesOpt = static_cast<ReduceTensorIndices_t>(std::atoi(optarg));
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

        if(reduceOp == ReduceTensorOp_t::MIN || reduceOp == ReduceTensorOp_t::MAX ||
           reduceOp == ReduceTensorOp_t::AMAX)
        {
            if(indicesOpt != ReduceTensorIndices_t::NO_INDICES)
                need_indices = true;

            // for indexable operations, no need to assign compType and outType, just let them be
            // same as inType
            compType_assigned = false;
            outType_assigned  = false;
        };

        return (0);
    };

}; // end of class AppArgs

int reduce_profiler(int argc, char* argv[])
{
    using namespace ck::profiler;

    AppArgs args;

    if(args.processArgs(argc, argv) < 0)
        return (-1);

    int rank = args.inLengths.size();

    check_reduce_dims(rank, args.toReduceDims);

    if(args.reduceOp == ReduceTensorOp_t::MUL || args.reduceOp == ReduceTensorOp_t::NORM1)
        throw std::runtime_error("MUL and NORM1 are not supported by composable kernel!");

    if(args.use_half)
    {
        if(!args.compType_assigned)
            args.compTypeId = appHalf;

        if(args.outType_assigned && (args.outTypeId != appHalf && args.outTypeId != appFloat))
            args.outTypeId = appFloat;

        if(!args.outType_assigned)
            args.outTypeId = appHalf;

        if(args.compTypeId == appHalf)
        {
            profile_reduce<ck::half_t, ck::half_t, ck::half_t>(args.do_verification,
                                                               args.init_method,
                                                               args.do_log,
                                                               args.do_dumpout,
                                                               args.nrepeat,
                                                               args.inLengths,
                                                               args.toReduceDims,
                                                               args.reduceOp,
                                                               args.nanOpt,
                                                               args.indicesOpt,
                                                               args.scales[0],
                                                               args.scales[1]);
        }
        else if(args.compTypeId == appFloat)
        {
            profile_reduce<ck::half_t, float, ck::half_t>(args.do_verification,
                                                          args.init_method,
                                                          args.do_log,
                                                          args.do_dumpout,
                                                          args.nrepeat,
                                                          args.inLengths,
                                                          args.toReduceDims,
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
        // profile_reduce<double, double, double>(args,do_verification, args.init_method,
        // args.do_log, args.nrepeat,
        //                   args.inLengths, args.toReduceDims, args.reduceOp, args.nanOpt,
        //                   args.indicesOpt, args.scales[0], args.scales[1]);
    }
    else
    {
        if(args.compTypeId == appFloat)
        {
            profile_reduce<float, float, float>(args.do_verification,
                                                args.init_method,
                                                args.do_log,
                                                args.do_dumpout,
                                                args.nrepeat,
                                                args.inLengths,
                                                args.toReduceDims,
                                                args.reduceOp,
                                                args.nanOpt,
                                                args.indicesOpt,
                                                args.scales[0],
                                                args.scales[1]);
        }
        else if(args.compTypeId == appDouble)
        {
            profile_reduce<float, double, float>(args.do_verification,
                                                 args.init_method,
                                                 args.do_log,
                                                 args.do_dumpout,
                                                 args.nrepeat,
                                                 args.inLengths,
                                                 args.toReduceDims,
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
