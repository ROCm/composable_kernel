// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>

namespace ck_tile {
namespace dispatcher {
namespace utils {

/**
 * Simple command-line argument parser for examples.
 *
 * Usage:
 *   ExampleArgs args("Example 01: Basic GEMM", "Demonstrates basic GEMM usage");
 *   args.add_flag("--list", "List all kernel sets");
 *   args.add_option("--dtype", "fp16", "Data type (fp16, bf16, fp32)");
 *   args.add_option("--size", "1024", "Problem size MxNxK");
 *
 *   if (!args.parse(argc, argv)) return 0;  // --help was printed
 *
 *   bool do_list = args.has("--list");
 *   std::string dtype = args.get("--dtype");
 *   int size = args.get_int("--size");
 */
class ExampleArgs
{
    public:
    ExampleArgs(const std::string& name, const std::string& description = "")
        : name_(name), description_(description)
    {
        // Always add --help
        add_flag("--help", "Show this help message");
        add_flag("-h", "Show this help message");
    }

    // Add a boolean flag (no value)
    void add_flag(const std::string& name, const std::string& help)
    {
        flags_[name] = false;
        help_[name]  = help;
        order_.push_back(name);
    }

    // Add an option with a default value
    void
    add_option(const std::string& name, const std::string& default_val, const std::string& help)
    {
        options_[name]  = default_val;
        defaults_[name] = default_val;
        help_[name]     = help;
        order_.push_back(name);
    }

    // Parse arguments. Returns false if --help was requested.
    bool parse(int argc, char* argv[])
    {
        for(int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];

            // Check for --help
            if(arg == "--help" || arg == "-h")
            {
                print_help();
                return false;
            }

            // Check for flags
            if(flags_.find(arg) != flags_.end())
            {
                flags_[arg] = true;
                continue;
            }

            // Check for options (--name=value or --name value)
            std::string name, value;
            size_t eq_pos = arg.find('=');
            if(eq_pos != std::string::npos)
            {
                name  = arg.substr(0, eq_pos);
                value = arg.substr(eq_pos + 1);
            }
            else if(options_.find(arg) != options_.end() && i + 1 < argc)
            {
                name  = arg;
                value = argv[++i];
            }
            else
            {
                // Positional argument - store as _pos_N
                std::string pos_name = "_pos_" + std::to_string(positional_.size());
                positional_.push_back(arg);
                continue;
            }

            if(options_.find(name) != options_.end())
            {
                options_[name] = value;
            }
        }
        return true;
    }

    // Check if a flag is set
    bool has(const std::string& name) const
    {
        auto it = flags_.find(name);
        return it != flags_.end() && it->second;
    }

    // Get an option value as string
    std::string get(const std::string& name) const
    {
        auto it = options_.find(name);
        return it != options_.end() ? it->second : "";
    }

    // Get an option value as string with default
    std::string get(const std::string& name, const std::string& default_val) const
    {
        auto it = options_.find(name);
        return it != options_.end() ? it->second : default_val;
    }

    // Get an option value as int
    int get_int(const std::string& name, int default_val = 0) const
    {
        std::string val = get(name);
        if(val.empty())
            return default_val;
        try
        {
            return std::stoi(val);
        }
        catch(...)
        {
            return default_val;
        }
    }

    // Get an option value as float
    float get_float(const std::string& name, float default_val = 0.0f) const
    {
        std::string val = get(name);
        if(val.empty())
            return default_val;
        try
        {
            return std::stof(val);
        }
        catch(...)
        {
            return default_val;
        }
    }

    // Get positional arguments
    const std::vector<std::string>& positional() const { return positional_; }

    // Print help message
    void print_help() const
    {
        std::cout << "\n";
        std::cout << "  " << name_ << "\n";
        if(!description_.empty())
        {
            std::cout << "  " << description_ << "\n";
        }
        std::cout << "\n";
        std::cout << "Usage:\n";
        std::cout << "  ./example [OPTIONS]\n";
        std::cout << "\n";
        std::cout << "Options:\n";

        // Find max option name length for alignment
        size_t max_len = 0;
        for(const auto& name : order_)
        {
            if(name == "-h")
                continue; // Skip -h, show --help only
            max_len = std::max(max_len, name.length());
        }

        // Print options in order
        for(const auto& name : order_)
        {
            if(name == "-h")
                continue;

            std::cout << "  " << std::left << std::setw(max_len + 2) << name;

            auto help_it = help_.find(name);
            if(help_it != help_.end())
            {
                std::cout << help_it->second;
            }

            // Show default value for options
            auto def_it = defaults_.find(name);
            if(def_it != defaults_.end() && !def_it->second.empty())
            {
                std::cout << " (default: " << def_it->second << ")";
            }

            std::cout << "\n";
        }
        std::cout << "\n";
    }

    private:
    std::string name_;
    std::string description_;
    std::map<std::string, bool> flags_;
    std::map<std::string, std::string> options_;
    std::map<std::string, std::string> defaults_;
    std::map<std::string, std::string> help_;
    std::vector<std::string> order_;
    std::vector<std::string> positional_;
};

} // namespace utils
} // namespace dispatcher
} // namespace ck_tile
