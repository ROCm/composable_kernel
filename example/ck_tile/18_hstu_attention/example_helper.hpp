
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <stdexcept>

template <typename T>
static void dumpBufferToFile(const char* fileName, T* data, size_t dataNumItems)
{
    std::ofstream outFile(fileName, std::ios::binary);
    if(outFile)
    {
        outFile.write(reinterpret_cast<char*>(data), dataNumItems * sizeof(T));
        outFile.close();
        printf("Write output to file %s\n", fileName);
    }
    else
    {
        printf("Could not open file %s for writing\n", fileName);
    }
}

template <typename T>
static void readDataToBufferFromFile(T* data, size_t dataNumItems, const std::string& fileName)
{
    std::ifstream infile(fileName, std::ios::binary);
    if(infile)
    {
        try
        {
            infile.read(reinterpret_cast<char*>(data), dataNumItems * sizeof(T));
            infile.close();
        }
        catch(const std::runtime_error& e)
        {
            throw e;
        };
    }
    else
    {
        throw std::runtime_error("could not open the file for reading");
    }
}

static std::vector<int> get_integers_from_string(std::string srcStr)
{
    std::vector<int> integers;
    std::size_t pos = 0;
    std::size_t new_pos;

    new_pos = srcStr.find(',', pos);
    while(new_pos != std::string::npos)
    {
        std::string sliceStr = srcStr.substr(pos, new_pos - pos);

        int len = std::stoi(sliceStr);

        integers.push_back(len);

        pos     = new_pos + 1;
        new_pos = srcStr.find(',', pos);
    };

    std::string sliceStr = srcStr.substr(pos);

    if(!sliceStr.empty())
    {
        int len = std::stoi(sliceStr);

        integers.push_back(len);
    };

    return (integers);
};

static std::vector<float> get_floats_from_string(std::string srcStr)
{
    std::vector<float> values;
    std::size_t pos = 0;
    std::size_t new_pos;

    new_pos = srcStr.find(',', pos);
    while(new_pos != std::string::npos)
    {
        std::string sliceStr = srcStr.substr(pos, new_pos - pos);

        float val = std::stof(sliceStr);

        values.push_back(val);

        pos     = new_pos + 1;
        new_pos = srcStr.find(',', pos);
    };

    std::string sliceStr = srcStr.substr(pos);

    if(!sliceStr.empty())
    {
        float val = std::stof(sliceStr);

        values.push_back(val);
    };

    return (values);
};

template <typename T>
static void supplement_array_by_last_element(std::vector<T>& arr, int target_num_elements)
{
    if(static_cast<int>(arr.size()) < target_num_elements)
    {
        T last_val = arr.back();

        for(int i = arr.size(); i < target_num_elements; i++)
            arr.push_back(last_val);
    };
};
