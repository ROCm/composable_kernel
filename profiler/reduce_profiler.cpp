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
#include "conv_common.hpp"
#include "host_conv.hpp"
#include "device_tensor.hpp"
#include "host_generic_reduction.hpp"

#include "reduction_enums.hpp"

using namespace std;

static struct option long_options[] = {{"inLengths", required_argument, NULL, 'D'},
                                       {"toReduceDims", required_argument, NULL, 'R'},
                                       {"reduceOp", required_argument, NULL, 'O'},
                                       {"compType", required_argument, NULL, 'C'},
                                       {"outType", required_argument, NULL, 'W'},
                                       {"nanOpt", required_argument, NULL, 'N'},
                                       {"indicesOpt", required_argument, NULL, 'I'},
                                       {"scales", required_argument, NULL, 'S'},
                                       {"half", no_argument, NULL, '?'},
                                       {"double", no_argument, NULL, '?'},
                                       {"dumpout", required_argument, NULL, 'o'},
                                       {"verify", required_argument, NULL, 'v'},
                                       {"log", required_argument, NULL, 'l'},
                                       {"help", no_argument, NULL, '?'},
                                       {0, 0, 0, 0}};
static int option_index             = 0;

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

static void show_usage(const char* cmd)
{
    std::cout << "Usage of " << cmd << std::endl;
    std::cout << "--inLengths or -D, comma separated list of input tensor dimension lengths"
              << std::endl;
    std::cout << "--toReduceDims or -R, comma separated list of to-reduce dimensions" << std::endl;
    std::cout << "--reduceOp or -O, enum value indicating the reduction operations" << std::endl;
    std::cout << "--compType or -C, enum value indicating the type of accumulated values used "
                 "during the reduction"
              << std::endl;
    std::cout << "--outType or -W, optional enum value indicating the type of the reduced output, "
                 "which could be float when the input data is half"
              << std::endl;
    std::cout << "--nanOpt or -N, enum value indicates the selection for NanOpt" << std::endl;
    std::cout << "--indicesOpt or -I, enum value indicates the selection for IndicesOpt"
              << std::endl;
    std::cout << "--scales or -S, comma separated two float values for alpha and beta" << std::endl;
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

static void check_reduce_dims(const int rank, const std::vector<int>& toReduceDims)
{
    for(auto dim : toReduceDims)
    {
        if(dim < 0 || dim >= rank)
            throw std::runtime_error("Invalid dimension index specified for Reducing");
    };

    unsigned int flag = 0;

    for(auto dim : toReduceDims) {
        if ( flag & (0x1 << dim) ) 
	     throw std::runtime_error("All toReduce dimensions should be different!"); 	
        flag = flag | (0x1 << dim); 
    }; 
};

static bool use_half   = false;
static bool use_double = false;

static vector<size_t> inLengths;
static vector<size_t> outLengths;
static vector<int> toReduceDims;

static vector<float> scales;

static ReduceTensorOp_t reduceOp        = ReduceTensorOp_t::REDUCE_TENSOR_ADD;
static appDataType_t compTypeId         = appFloat;
static appDataType_t outTypeId          = appFloat;
static bool compType_assigned           = false;
static bool outType_assigned            = false;
static NanPropagation_t nanOpt          = NanPropagation_t::NOT_PROPAGATE_NAN;
static ReduceTensorIndices_t indicesOpt = ReduceTensorIndices_t::REDUCE_TENSOR_NO_INDICES;
static bool do_log                  = false;
static bool do_verification             = false;
static bool do_dumpout                  = false;

static int init_method;
static int nrepeat;

static bool need_indices = false;

static void check_cmdline_arguments(int argc, char* argv[])
{
    unsigned int ch;

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
                exit(0);
            };
            break;

        default: show_usage(argv[0]); throw std::runtime_error("Invalid cmd-line options!");
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

    if(reduceOp == ReduceTensorOp_t::REDUCE_TENSOR_MIN ||
       reduceOp == ReduceTensorOp_t::REDUCE_TENSOR_MAX ||
       reduceOp == ReduceTensorOp_t::REDUCE_TENSOR_AMAX)
    {

        if(indicesOpt != ReduceTensorIndices_t::REDUCE_TENSOR_NO_INDICES)
            need_indices = true;

        // for indexable operations, no need to assign compType and outType, just let them be same
        // as inType
        compType_assigned = false;
        outType_assigned  = false;
    };
};

int reduce_profiler(int argc, char* argv[])
{
    using namespace ck;
    using half = half_float::half;

    check_cmdline_arguments(argc, argv);

    int rank = inLengths.size(); 

    check_reduce_dims(rank, toReduceDims);

    if(use_half)
    {
        if(!compType_assigned)
            compTypeId = appHalf;

        if(outType_assigned && (outTypeId != appHalf && outTypeId != appFloat))
            outTypeId = appFloat;

        if(!outType_assigned)
            outTypeId = appHalf;

        if(compTypeId == appHalf)
        {
            if(outTypeId == appHalf)
                profile_reduce<half_float::half, half_float::half, half_float::half>(do_verification, init_method, do_log, nrepeat,
				                                                     inLengths, toReduceDims, reduceOp, nanOpt, indicesOpt, scales[0], scales[1]);
            else
                profile_reduce<half_float::half, half_float::half, float>(do_verification, init_method, do_log, nrepeat, 
				                                                     inLengths, toReduceDims, reduceOp, nanOpt, indicesOpt, scales[0], scales[1]);
        }
        else if(compTypeId == appFloat)
        {
            if(outTypeId == appHalf)
                profile_reduce<half_float::half, float, half_float::half>(do_verification, init_method, do_log, nrepeat, 
				                                                     inLengths, toReduceDims, reduceOp, nanOpt, indicesOpt, scales[0], scales[1]);
            else
                profile_reduce<half_float::half, float, float>(do_verification, init_method, do_log, nrepeat, 
				                                                     inLengths, toReduceDims, reduceOp, nanOpt, indicesOpt, scales[0], scales[1]);
        }
        else
            throw std::runtime_error("Invalid compType assignment!");
    }
    else if(use_double)
        profile_reduce<double, double, double>(do_verification, init_method, do_log, nrepeat, 
				                                                     inLengths, toReduceDims, reduceOp, nanOpt, indicesOpt, scales[0], scales[1]);
    else
    {
        if(compTypeId == appFloat)
            profile_reduce<float, float, float>(do_verification, init_method, do_log, nrepeat, 
				                                                     inLengths, toReduceDims, reduceOp, nanOpt, indicesOpt, scales[0], scales[1]);
        else if(compTypeId == appDouble)
            profile_reduce<float, double, float>(do_verification, init_method, do_log, nrepeat, 
				                                                     inLengths, toReduceDims, reduceOp, nanOpt, indicesOpt, scales[0], scales[1]);
        else
            throw std::runtime_error("Invalid compType assignment!");
    };
};

}
