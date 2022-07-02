// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include "ck/ck.hpp"

namespace ck {

namespace host_common {

template <typename T>
static inline void dumpBufferToFile(const char* fileName, T* data, size_t dataNumItems)
{
    std::ofstream outFile(fileName, std::ios::binary);
    if(outFile)
    {
        outFile.write(reinterpret_cast<char*>(data), dataNumItems * sizeof(T));
        outFile.close();
        std::cout << "Write output to file " << fileName << std::endl;
    }
    else
    {
        std::cout << "Could not open file " << fileName << " for writing" << std::endl;
    }
};

template <typename T>
static inline T getSingleValueFromString(const std::string& valueStr)
{
    std::istringstream iss(valueStr);

    T val;

    iss >> val;

    return (val);
};

template <typename T>
static inline std::vector<T> getTypeValuesFromString(const char* cstr_values)
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

} // namespace host_common
} // namespace ck
