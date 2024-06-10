#include "profiler_operation_registry.hpp"
#include "ck/library/utility/convolution_parameter.hpp"

#include <iostream>

#define OP_NAME "grouped_conv_fwd_outelementop"
#define OP_DESC "Grouped Convolution Forward+Elementwise Operation"

static void print_helper_msg()
{
    // clang-format off
    std::cout
        << "arg1: tensor operation (" OP_NAME ": " OP_DESC ")\n"
        << "arg2: data type (0: Input fp8, Weight fp8, Output fp8\n"
        << "                 1: Input bf8, Weight bf8, Output fp8\n"
        << "                 2: Input fp8, Weight bf8, Output fp8\n"
        << "                 3: Input bf8, Weight fp8, Output fp8)\n"
        << ck::utils::conv::get_conv_param_parser_helper_msg() << std::endl;
    // clang-format on
}

int grouped_conv_fwd_outelementop(int argc, char* argv[])
{
    std::cout << "argc: " << argc << "\targv[0]: " << argv[0] << std::endl;
    print_helper_msg();
    return 1;
}

REGISTER_PROFILER_OPERATION(OP_NAME, OP_DESC, grouped_conv_fwd_outelementop);
