// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <numeric>

// Local KernelTraits -- kept here to avoid touching tile_engine/ops/common/utils.hpp
// (shared-utils changes were the root cause of the revert in PR #9759).
struct ContractionMultiABDKernelTraits
{
    std::string pipeline;
    std::string epilogue;
    std::string scheduler;
    bool pad_m;
    bool pad_n;
    bool pad_k;
    bool persistent;

    ContractionMultiABDKernelTraits()
        : pipeline("compv3"),
          epilogue("cshuffle"),
          scheduler("intrawave"),
          pad_m(false),
          pad_n(false),
          pad_k(false),
          persistent(false)
    {
    }
};

// Parse comma-separated dimension string, e.g. "4,256" -> {4, 256}
inline std::vector<int> parse_dims(const std::string& dims_str)
{
    std::vector<int> dims;
    if(dims_str.empty())
        return dims;
    std::stringstream ss(dims_str);
    std::string token;
    while(std::getline(ss, token, ','))
        dims.push_back(std::stoi(token));
    return dims;
}

inline int product(const std::vector<int>& v)
{
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<int>{});
}

inline std::string dims_to_str(const std::vector<int>& dims)
{
    std::string s;
    for(size_t i = 0; i < dims.size(); ++i)
    {
        if(i > 0)
            s += ',';
        s += std::to_string(dims[i]);
    }
    return s;
}

struct ContractionMultiABDProblem
{
    std::vector<int> g_dims;
    std::vector<int> m_dims;
    std::vector<int> n_dims;
    std::vector<int> k_dims;

    int num_a_tensors = 1;
    int num_b_tensors = 1;
    int num_d_tensors = 1;

    std::string dtype;
    std::string layout; // 3-char: a+b+e

    int G_total() const { return product(g_dims); }
    int M_total() const { return product(m_dims); }
    int N_total() const { return product(n_dims); }
    int K_total() const { return product(k_dims); }
};
