#include <iostream>
#include <fstream>
#include <getopt.h>
#include "profile_bnorm_fwd_impl.hpp"

using namespace ck;

static struct option long_options[] = {{"inOutLengths", required_argument, nullptr, 'D'},
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
    std::vector<size_t> inOutLengths;

    int data_type;
    int init_method = 1;

    public:
    void show_usage(const char* cmd)
    {
        std::cout << "Usage of " << cmd << std::endl;
        std::cout << "--inOutLengths or -D, comma separated list of input/output tensor dimension "
                     "lengths "
                     "(only 4-d tensor supported)"
                  << std::endl;
        std::cout << "Arg1 -- data type (0: fp16, 1: fp32, 3: int8, 5: bp16, 6: fp64)" << std::endl;
        std::cout << "Arg2 -- init method used for bnScale and bnBias (0=no init, 1=single integer "
                     "value, 2=scope integer "
                     "value, 3=decimal value)"
                  << std::endl;
    };

    int processArgs(int argc, char* argv[])
    {
        unsigned int ch;

        while(1)
        {
            ch = getopt_long(argc, argv, "D:", long_options, &option_index);
            if(ch == -1)
                break;
            switch(ch)
            {
            case 'D':
                if(!optarg)
                    throw std::runtime_error("Invalid option format!");

                inOutLengths = getTypeValuesFromString<size_t>(optarg);
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

        if(inOutLengths.size() != 4)
            return (-1);

        if(data_type != 0 && data_type != 1 && data_type != 3 && data_type != 5 && data_type != 6)
            return (-1);

        return (0);
    };
};

bool test_bnorm_fwd(int data_type,
                    int init_method,
                    const std::vector<size_t> inOutLengths,
                    const std::vector<size_t> scaleBiasMeanVarLengths,
                    bool saveMeanAndInvVariance,
                    bool updateMovingAverage,
                    float alpha,
                    float beta)
{
    using ck::profiler::profile_bnorm_fwd_impl;

    bool result                           = true;
    const double exponentialAverageFactor = 0.2;

    if(data_type == 0)
    {
        const double epsilon = std::numeric_limits<float>::epsilon();

        result = profile_bnorm_fwd_impl<ck::half_t, float>(true,
                                                           init_method,
                                                           false,
                                                           0,
                                                           inOutLengths,
                                                           scaleBiasMeanVarLengths,
                                                           saveMeanAndInvVariance,
                                                           updateMovingAverage,
                                                           epsilon,
                                                           exponentialAverageFactor,
                                                           alpha,
                                                           beta);
    }
    else if(data_type == 1)
    {
        const double epsilon = std::numeric_limits<float>::epsilon();

        result = profile_bnorm_fwd_impl<float, float>(true,
                                                      init_method,
                                                      false,
                                                      0,
                                                      inOutLengths,
                                                      scaleBiasMeanVarLengths,
                                                      saveMeanAndInvVariance,
                                                      updateMovingAverage,
                                                      epsilon,
                                                      exponentialAverageFactor,
                                                      alpha,
                                                      beta);
    }
    else if(data_type == 3)
    {
        const double epsilon = std::numeric_limits<float>::epsilon();

        result = profile_bnorm_fwd_impl<int8_t, float>(true,
                                                       init_method,
                                                       false,
                                                       0,
                                                       inOutLengths,
                                                       scaleBiasMeanVarLengths,
                                                       saveMeanAndInvVariance,
                                                       updateMovingAverage,
                                                       epsilon,
                                                       exponentialAverageFactor,
                                                       alpha,
                                                       beta);
    }
    else if(data_type == 5)
    {
        const double epsilon = std::numeric_limits<float>::epsilon();

        result = profile_bnorm_fwd_impl<ck::bhalf_t, float>(true,
                                                            init_method,
                                                            false,
                                                            0,
                                                            inOutLengths,
                                                            scaleBiasMeanVarLengths,
                                                            saveMeanAndInvVariance,
                                                            updateMovingAverage,
                                                            epsilon,
                                                            exponentialAverageFactor,
                                                            alpha,
                                                            beta);
    }
    else if(data_type == 6)
    {
        const double epsilon = std::numeric_limits<double>::epsilon();

        result = profile_bnorm_fwd_impl<double, double>(true,
                                                        init_method,
                                                        false,
                                                        0,
                                                        inOutLengths,
                                                        scaleBiasMeanVarLengths,
                                                        saveMeanAndInvVariance,
                                                        updateMovingAverage,
                                                        epsilon,
                                                        exponentialAverageFactor,
                                                        alpha,
                                                        beta);
    };

    return result;
};

int main(int argc, char* argv[])
{
    SimpleAppArgs args;

    bool result = true;

    if(argc == 1)
    {
        std::vector<int> data_types = {0, 1};
        int init_method             = 1;

        std::vector<std::vector<size_t>> v_inOutLengths{{256, 14, 14, 1024},
                                                        {256, 28, 28, 128},
                                                        {256, 58, 58, 128},
                                                        {256, 7, 7, 2048},
                                                        {256, 14, 14, 256},
                                                        {256, 30, 30, 256},
                                                        {256, 56, 56, 256},
                                                        {256, 16, 16, 512}};
        for(auto data_type : data_types)
        {
            for(auto inOutLengths : v_inOutLengths)
            {
                std::vector<size_t> scaleBiasMeanVarLengths = {inOutLengths[3]};
                result                                      = result && test_bnorm_fwd(data_type,
                                                  init_method,
                                                  inOutLengths,
                                                  scaleBiasMeanVarLengths,
                                                  true,
                                                  false,
                                                  1.0f,
                                                  0.0f);
            };
        };
    }
    else
    {
        if(args.processArgs(argc, argv) < 0)
        {
            throw std::runtime_error(
                "Invalid input arguments, test_bnorm_fwd could not be executed!");
        };

        std::vector<size_t> scaleBiasMeanVarLengths = {args.inOutLengths[3]};

        result = test_bnorm_fwd(args.data_type,
                                args.init_method,
                                args.inOutLengths,
                                scaleBiasMeanVarLengths,
                                true,
                                false,
                                1.0f,
                                0.0f);
    }

    std::cout << "test_bnorm_fwd ..... " << (result ? "SUCCESS" : "FAILURE") << std::endl;

    return (result ? 0 : -1);
}
