// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <iostream>

#include "profiler_operation_registry.hpp"

static void print_helper_message()
{
    std::cout << "arg1: tensor operation " << ProfilerOperationRegistry::GetInstance() << std::endl;
}

int main(int argc, char* argv[])
{
    // This hack saves a bunch of time, but don't use it if rocprofiler is being used
    const char *p = getenv("CK_QUICK_EXIT");
    bool quick_exit;
    quick_exit = p != nullptr;
    int exit_code = 0;
    if(argc == 1)
    {
        print_helper_message();
        quick_exit = true;
    }
    else if(const auto operation = ProfilerOperationRegistry::GetInstance().Get(argv[1]);
            operation.has_value())
    {
        exit_code = (*operation)(argc, argv);
        quick_exit = (exit_code != 0);
    }
    else
    {
        std::cerr << "cannot find operation: " << argv[1] << std::endl;
        exit_code = EXIT_FAILURE;
        quick_exit = true;
    }
    if (quick_exit)
      std::quick_exit(exit_code);
    return exit_code;
}
