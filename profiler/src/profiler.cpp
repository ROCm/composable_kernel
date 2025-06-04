// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <tuple>
#include "profiler/argparse_wrapper.hpp"

#include "profiler/io_profiler.hpp"
#include "profiler_operation_registry.hpp"

static void print_helper_message()
{
    std::cout << "arg1: tensor operation " << ProfilerOperationRegistry::GetInstance() << std::endl;
}

// Parse global arguments and set up output configuration
// Returns a tuple containing:
// 1. The operation name if one was provided
// 2. New argc value
// 3. New argv array with processed arguments removed
std::tuple<std::string, int, char**> parse_global_arguments(int argc, char* argv[])
{
    // Initialize argparse
    argparse::ArgumentParser program("ckProfiler", "1.0");

    // Add global arguments for output type
    ck::profiler::io::AddOutputArguments(program);

    // Extract the operation name before parsing (since we want to support operations with hyphens)
    std::string operation_name;
    if(argc > 1 && argv[1][0] != '-')
    {
        operation_name = argv[1];
    }

    try
    {
        // Parse known arguments (this ignores unknown arguments)
        program.parse_known_args(argc, argv);

        // Configure output based on parsed arguments
        ck::profiler::io::ParseOutputArguments(program);
    }
    catch(const std::exception& err)
    {
        std::cerr << "Error parsing arguments: " << err.what() << std::endl;
        // Keep default console output
    }

    // Now remove the processed arguments to create a clean argv for the operation
    // First, identify which arguments to keep
    std::vector<int> args_to_keep;
    args_to_keep.push_back(0); // Always keep program name

    // If we have an operation name, keep it
    if(!operation_name.empty() && argc > 1)
    {
        args_to_keep.push_back(1); // Keep the operation name
    }

    // Check remaining arguments
    for(int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        // Skip the operation name (already handled)
        if(i == 1 && !operation_name.empty())
        {
            continue;
        }

        // Skip output arguments and their values
        if(arg == "-o" || arg == "--output")
        {
            i++; // Skip the value too
            continue;
        }

        // Keep all other arguments
        args_to_keep.push_back(i);
    }

    // Create new argv array
    int new_argc    = args_to_keep.size();
    char** new_argv = new char*[new_argc];

    for(int i = 0; i < new_argc; i++)
    {
        // Make a copy of each string we want to keep
        int original_index = args_to_keep[i];
        size_t len         = std::strlen(argv[original_index]) + 1; // +1 for null terminator
        new_argv[i]        = new char[len];
        std::strcpy(new_argv[i], argv[original_index]);
    }

    return {operation_name, new_argc, new_argv};
}

int main(int argc, char* argv[])
{
    // Check if no arguments were provided
    if(argc == 1)
    {
        print_helper_message();
        return 0;
    }

    // Parse global arguments and get operation name and filtered arguments
    std::string operation_name;
    int new_argc;
    char** new_argv;
    std::tie(operation_name, new_argc, new_argv) = parse_global_arguments(argc, argv);

    // Helper function to clean up memory
    auto cleanup = [new_argv, new_argc]() {
        if(new_argv)
        {
            for(int i = 0; i < new_argc; i++)
            {
                delete[] new_argv[i];
            }
            delete[] new_argv;
        }
    };

    // Check if we have an operation to run
    if(operation_name.empty())
    {
        print_helper_message();
        cleanup();
        return 0;
    }

    // Look up the operation and run it if found
    if(const auto operation = ProfilerOperationRegistry::GetInstance().Get(operation_name);
       operation.has_value())
    {
        int result = (*operation)(new_argc, new_argv);
        cleanup();
        return result;
    }
    else
    {
        std::cerr << "cannot find operation: " << operation_name << std::endl;
        cleanup();
        return EXIT_FAILURE;
    }
}
