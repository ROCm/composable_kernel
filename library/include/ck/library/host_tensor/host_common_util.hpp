/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef GUARD_HOST_COMMON_UTIL_HPP
#define GUARD_HOST_COMMON_UTIL_HPP

#include <half.hpp>
#include <vector>
#include <fstream>
#include "data_type.hpp"

namespace ck {

namespace host_common {

static inline std::vector<index_t> to_int_vector(const std::vector<size_t>& inData)
{
    std::vector<index_t> outData;

    for(auto elem : inData)
        outData.push_back(static_cast<index_t>(elem));

    return (outData);
};

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

}; // namespace host_common

}; // namespace ck

#endif
