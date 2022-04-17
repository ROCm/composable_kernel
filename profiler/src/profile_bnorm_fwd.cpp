#include <vector>
#include <cstdlib>
#include <stdexcept>
#include <sstream>
#include <getopt.h>

#include "profile_bnorm_fwd_impl.hpp"

using namespace std;

static struct option long_options[] = {{"inOutLengths", required_argument, nullptr, 'D'},
                                       {"half", no_argument, nullptr, '?'},
                                       {"double", no_argument, nullptr, '?'},
                                       {"int8", no_argument, nullptr, '?'},
                                       {"bf16", no_argument, nullptr, '?'},
                                       {"dumpout", required_argument, nullptr, 'o'},
                                       {"verify", required_argument, nullptr, 'v'},
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

class BatchNormFwdProfilerArgs
{
    private:
    int option_index = 0;

    public:
    bool use_half   = false;
    bool use_double = false;
    bool use_int8   = false;
    bool use_bf16   = false;

    std::vector<size_t> inOutLengths;

    bool do_verification = false;
    bool do_dumpout      = false;

    bool saveMeanAndInvVariance;
    bool updateMovingAverage;

    int init_method;
    int nrepeat;

    BatchNormFwdProfilerArgs()  = default;
    ~BatchNormFwdProfilerArgs() = default;

    void show_usage(const char* cmd)
    {
        std::cout << "Usage of " << cmd << std::endl;
        std::cout
            << "--inOutLengths or -D, comma separated list of input/output tensor dimension lengths"
            << std::endl;
        std::cout << "--half, use fp16 for the input and output tensor data types" << std::endl;
        std::cout << "--double, use fp64 for the input and output tensor data types" << std::endl;
        std::cout << "--int8, use int8 for the input and output tensor data types" << std::endl;
        std::cout << "--bf16, use bfloat16 for the input and output tensor data types" << std::endl;
        std::cout << "--verify or -v, 1/0 to indicate whether to verify the batch-norm result by "
                     "comparing with the host-based batch-norm"
                  << std::endl;
        std::cout
            << "--dumpout or -o, 1/0 to indicate where to save the batch-norm result to files "
               "for further analysis"
            << std::endl;
        std::cout << "Arg1 -- 1/0 to indicate whether to save the calculated mean and invVariance"
                  << std::endl;
        std::cout << "Arg2 -- 1/0 to indicate whether to update the moving average of the mean and "
                     "variance"
                  << std::endl;
        std::cout << "Arg3 -- init method used for bnScale and bnBias (0=no init, 1=single integer "
                     "value, 2=scope integer "
                     "value, 3=decimal value)"
                  << std::endl;
        std::cout << "Arg4 -- number of repeats to run the kernel" << std::endl;
    };

    int processArgs(int argc, char* argv[])
    {
        unsigned int ch;

        optind++; // to skip the "bnorm_fwd" module name

        while(1)
        {
            ch = getopt_long(argc, argv, "D:v:o:", long_options, &option_index);
            if(ch == -1)
                break;
            switch(ch)
            {
            case 'D':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                inOutLengths = getTypeValuesFromString<size_t>(optarg);
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

        if(optind + 4 > argc)
            throw std::runtime_error("Invalid cmd-line arguments, more argumetns are needed!");

        saveMeanAndInvVariance = std::atoi(argv[optind++]);
        updateMovingAverage    = std::atoi(argv[optind++]);

        init_method = std::atoi(argv[optind++]);
        nrepeat     = std::atoi(argv[optind]);

        return (0);
    };

}; // end of class AppArgs

int profile_bnorm_fwd(int argc, char* argv[])
{
    using namespace ck::profiler;
    const double exponentialAverageFactor = 0.2;

    BatchNormFwdProfilerArgs args;

    if(args.processArgs(argc, argv) < 0)
        return (-1);

    int rank = args.inOutLengths.size();

    if(rank != 4)
    {
        throw std::runtime_error(
            "The input/out tensor lengths must have 4 dimensions for NHWC layout!");
    }

    // currently only NHWC layout and spatial batch-norm mode supported
    std::vector<size_t> scaleBiasMeanVarLengths = {args.inOutLengths[3]};

    if(args.use_half)
    {
        const double epsilon = 0.0001;

        profile_bnorm_fwd_impl<ck::half_t, float>(args.do_verification,
                                                  args.init_method,
                                                  args.do_dumpout,
                                                  args.nrepeat,
                                                  args.inOutLengths,
                                                  scaleBiasMeanVarLengths,
                                                  args.saveMeanAndInvVariance,
                                                  args.updateMovingAverage,
                                                  epsilon,
                                                  exponentialAverageFactor,
                                                  1.0f,
                                                  0.0f);
    }
    else if(args.use_double)
    {
        const double epsilon = std::numeric_limits<double>::epsilon();

        profile_bnorm_fwd_impl<double, double>(args.do_verification,
                                               args.init_method,
                                               args.do_dumpout,
                                               args.nrepeat,
                                               args.inOutLengths,
                                               scaleBiasMeanVarLengths,
                                               args.saveMeanAndInvVariance,
                                               args.updateMovingAverage,
                                               epsilon,
                                               exponentialAverageFactor,
                                               1.0f,
                                               0.0f);
    }
    else if(args.use_int8)
    {
        const double epsilon = std::numeric_limits<float>::epsilon();

        profile_bnorm_fwd_impl<int8_t, float>(args.do_verification,
                                              args.init_method,
                                              args.do_dumpout,
                                              args.nrepeat,
                                              args.inOutLengths,
                                              scaleBiasMeanVarLengths,
                                              args.saveMeanAndInvVariance,
                                              args.updateMovingAverage,
                                              epsilon,
                                              exponentialAverageFactor,
                                              1.0f,
                                              0.0f);
    }
    else if(args.use_bf16)
    {
        const double epsilon = 0.0001;

        profile_bnorm_fwd_impl<ck::bhalf_t, float>(args.do_verification,
                                                   args.init_method,
                                                   args.do_dumpout,
                                                   args.nrepeat,
                                                   args.inOutLengths,
                                                   scaleBiasMeanVarLengths,
                                                   args.saveMeanAndInvVariance,
                                                   args.updateMovingAverage,
                                                   epsilon,
                                                   exponentialAverageFactor,
                                                   1.0f,
                                                   0.0f);
    }
    else
    {
        const double epsilon = std::numeric_limits<float>::epsilon();

        profile_bnorm_fwd_impl<float, float>(args.do_verification,
                                             args.init_method,
                                             args.do_dumpout,
                                             args.nrepeat,
                                             args.inOutLengths,
                                             scaleBiasMeanVarLengths,
                                             args.saveMeanAndInvVariance,
                                             args.updateMovingAverage,
                                             epsilon,
                                             exponentialAverageFactor,
                                             1.0f,
                                             0.0f);
    };

    return (0);
};
