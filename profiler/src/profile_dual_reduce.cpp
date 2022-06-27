#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <getopt.h>

#include "ck/library/host_tensor/host_common_util.hpp"

#include "profiler/include/profile_dual_reduce_impl.hpp"
#include "profiler/include/data_type_enum.hpp"

using namespace std;

static struct option long_options[] = {{"inLengths", required_argument, nullptr, 'D'},
                                       {"reduceDims", required_argument, nullptr, 'R'},
                                       {"compType", required_argument, nullptr, 'C'},
                                       {"outType", required_argument, nullptr, 'W'},
                                       {"nanOpt", required_argument, nullptr, 'N'},
                                       {"scales", required_argument, nullptr, 'S'},
                                       {"half", no_argument, nullptr, '?'},
                                       {"dumpout", required_argument, nullptr, 'o'},
                                       {"verify", required_argument, nullptr, 'v'},
                                       {"help", no_argument, nullptr, '?'},
                                       {nullptr, 0, nullptr, 0}};

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

class ReduceProfilerArgs
{
    private:
    int option_index = 0;

    public:
    bool use_half = false;

    std::vector<size_t> inLengths;
    std::vector<size_t> outLengths;
    std::vector<int> reduceDims;

    std::vector<float> scales;

    ck::DataTypeEnum compTypeId = ck::DataTypeEnum::Float;
    ck::DataTypeEnum outTypeId  = ck::DataTypeEnum::Float;

    bool compType_assigned = false;
    bool outType_assigned  = false;

    int nanOpt           = 0;
    bool do_verification = false;
    bool do_dumpout      = false;

    int init_method;
    bool time_kernel;

    ReduceProfilerArgs()  = default;
    ~ReduceProfilerArgs() = default;

    void show_usage(const char* cmd)
    {
        std::cout << "Usage of " << cmd << std::endl;
        std::cout << "--inLengths or -D, comma separated list of input tensor dimension lengths"
                  << std::endl;
        std::cout << "--reduceDims or -R, comma separated list of to-reduce dimensions"
                  << std::endl;
        std::cout << "--compType or -C, enum value indicating the type of accumulated values used "
                     "during the reduction"
                  << std::endl;
        std::cout << "--outType or -W, optional enum value indicating the type of the reduced "
                     "output, which could be float when the input data is half"
                  << std::endl;
        std::cout
            << "--nanOpt or -N, 1/0 value indicates the selection to use or not use Nan-Propagation"
            << std::endl;
        std::cout << "--scales or -S, comma separated two float values for alpha and beta"
                  << std::endl;
        std::cout << "--half, use fp16 for the input and output tensor data types" << std::endl;
        std::cout << "--verify or -v, 1/0 to indicate whether to verify the reduction result by "
                     "comparing with the host-based reduction"
                  << std::endl;
        std::cout << "--dumpout or -o, 1/0 to indicate where to save the reduction result to files "
                     "for further analysis"
                  << std::endl;
    };

    int processArgs(int argc, char* argv[])
    {
        using ck::host_common::getTypeValuesFromString;

        int ch;

        optind++; // to skip the "reduce" module name

        while(1)
        {
            ch = getopt_long(argc, argv, "D:R:C:W:N:S:v:o:", long_options, &option_index);
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
            case 'C':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                compTypeId        = static_cast<ck::DataTypeEnum>(std::atoi(optarg));
                compType_assigned = true;
                break;
            case 'W':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                outTypeId        = static_cast<ck::DataTypeEnum>(std::atoi(optarg));
                outType_assigned = true;
                break;
            case 'N':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                nanOpt = std::atoi(optarg);
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
            case '?':
                if(std::string(long_options[option_index].name) == "half")
                    use_half = true;
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
        time_kernel = static_cast<bool>(std::atoi(argv[optind]));

        if(scales.empty())
        {
            scales.push_back(1.0f);
            scales.push_back(0.0f);
        };

        return (0);
    };

}; // end of class AppArgs

int profile_dual_reduce(int argc, char* argv[])
{
    using ck::DataTypeEnum;
    using ck::profiler::profile_dual_reduce_impl_for_mean_meansquare;

    ReduceProfilerArgs args;

    if(args.processArgs(argc, argv) < 0)
        return (-1);

    int rank = args.inLengths.size();

    check_reduce_dims(rank, args.reduceDims);

    if(args.use_half)
    {
        if(!args.compType_assigned)
            args.compTypeId = DataTypeEnum::Float;

        if(args.outType_assigned && (args.outTypeId != DataTypeEnum::Float))
            args.outTypeId = DataTypeEnum::Float;

        if(!args.outType_assigned)
            args.outTypeId = DataTypeEnum::Float;

        profile_dual_reduce_impl_for_mean_meansquare<ck::half_t, float, float, float>(
            args.do_verification,
            args.init_method,
            args.do_dumpout,
            args.time_kernel,
            args.inLengths,
            args.reduceDims,
            static_cast<bool>(args.nanOpt),
            args.scales[0],
            args.scales[1]);
    }
    else
    {
        if(args.compTypeId != DataTypeEnum::Float && args.compTypeId != DataTypeEnum::Double)
            args.compTypeId = DataTypeEnum::Float;

        if(args.compTypeId == DataTypeEnum::Float)
        {
            profile_dual_reduce_impl_for_mean_meansquare<float, float, float, float>(
                args.do_verification,
                args.init_method,
                args.do_dumpout,
                args.time_kernel,
                args.inLengths,
                args.reduceDims,
                static_cast<bool>(args.nanOpt),
                args.scales[0],
                args.scales[1]);
        }
        else if(args.compTypeId == DataTypeEnum::Double)
        {
            profile_dual_reduce_impl_for_mean_meansquare<float, double, double, double>(
                args.do_verification,
                args.init_method,
                args.do_dumpout,
                args.time_kernel,
                args.inLengths,
                args.reduceDims,
                static_cast<bool>(args.nanOpt),
                args.scales[0],
                args.scales[1]);
        }
        else
            throw std::runtime_error("Invalid compType assignment!");
    };

    return (0);
};
