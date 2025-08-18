// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstring>
#include <string>
#include <tuple>

#include "ck_tile/host.hpp"

// Common utilities for single kernel benchmarks

template <typename T>
struct DataTypeTraits;

template <>
struct DataTypeTraits<float>
{
    static constexpr const char* name = "fp32";
};

template <>
struct DataTypeTraits<double>
{
    static constexpr const char* name = "fp64";
};

template <>
struct DataTypeTraits<ck_tile::half_t>
{
    static constexpr const char* name = "fp16";
};

template <>
struct DataTypeTraits<ck_tile::bf16_t>
{
    static constexpr const char* name = "bf16";
};

template <>
struct DataTypeTraits<ck_tile::fp8_t>
{
    static constexpr const char* name = "fp8";
};

template <>
struct DataTypeTraits<ck_tile::bf8_t>
{
    static constexpr const char* name = "bf8";
};

template <>
struct DataTypeTraits<ck_tile::int8_t>
{
    static constexpr const char* name = "int8";
};

template <>
struct DataTypeTraits<ck_tile::int32_t>
{
    static constexpr const char* name = "int32";
};

// Helper functions for tensor initialization
template <typename DataType>
void initialize_tensor_random(void* ptr, size_t num_elements)
{
    auto* data = reinterpret_cast<DataType*>(ptr);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for(size_t i = 0; i < num_elements; ++i)
    {
        data[i] = static_cast<DataType>(dis(gen));
    }
}

template <typename DataType>
void initialize_tensor_linear(void* ptr, size_t num_elements)
{
    auto* data = reinterpret_cast<DataType*>(ptr);
    for(size_t i = 0; i < num_elements; ++i)
    {
        data[i] = static_cast<DataType>(i);
    }
}

template <typename DataType>
void initialize_tensor_constant(void* ptr, size_t num_elements, DataType value)
{
    auto* data = reinterpret_cast<DataType*>(ptr);
    for(size_t i = 0; i < num_elements; ++i)
    {
        data[i] = value;
    }
}
