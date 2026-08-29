// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "ck_tile/host/host_tensor.hpp"

namespace ck_tile {

// read data from a file, as dtype
// the file could dumped from torch as (targeting tensor is t here)
// numpy.savetxt("f.txt", t.view(-1).numpy())
// numpy.savetxt("f.txt", t.cpu().view(-1).numpy()) # from cuda to cpu to save
// numpy.savetxt("f.txt", t.cpu().view(-1).numpy(), fmt="%d")   # save as int
// will output f.txt, each line is a value
// dtype=float or int, internally will cast to real type
template <typename T>
void loadtxt(HostTensor<T>& tensor,
             const std::string& file_name,
             const std::string& dtype = "float")
{
    std::ifstream file(file_name);

    if(file.is_open())
    {
        std::string line;

        index_t cnt = 0;
        while(std::getline(file, line))
        {
            if(cnt >= static_cast<index_t>(tensor.mData.size()))
            {
                throw std::runtime_error(std::string("data read from file:") + file_name +
                                         " is too big");
            }

            if(dtype == "float")
            {
                tensor.mData[cnt] = type_convert<T>(std::stof(line));
            }
            else if(dtype == "int" || dtype == "int32")
            {
                tensor.mData[cnt] = type_convert<T>(std::stoi(line));
            }
            else
            {
                throw std::runtime_error(std::string("loadtxt: unsupported dtype:") + dtype);
            }
            cnt++;
        }
        file.close();
        if(cnt < static_cast<index_t>(tensor.mData.size()))
        {
            std::cerr << "Warning! reading from file:" << file_name
                      << ", does not match the size of this tensor" << std::endl;
        }
    }
    else
    {
        throw std::runtime_error(std::string("unable to open file:") + file_name);
    }
}

// Flat dump: one value per line, no shape info. Readable by numpy:
// torch.from_numpy(np.loadtxt('f.txt', dtype=np.int32/np.float32...)).view([...]).contiguous()
template <typename T>
void savetxt(const HostTensor<T>& tensor,
             const std::string& file_name,
             const std::string& dtype = "float")
{
    std::ofstream file(file_name);

    if(dtype != "float" && dtype != "int" && dtype != "int8_t")
    {
        throw std::runtime_error(std::string("savetxt: unsupported dtype:") + dtype);
    }

    if(file.is_open())
    {
        for(const auto& itm : tensor.mData)
        {
            if(dtype == "float")
                file << type_convert<float>(itm) << '\n';
            else if(dtype == "int")
                file << type_convert<int>(itm) << '\n';
            else if(dtype == "int8_t")
                file << static_cast<int>(type_convert<ck_tile::int8_t>(itm)) << '\n';
        }
        file.close();
    }
    else
    {
        throw std::runtime_error(std::string("unable to open file:") + file_name);
    }
}

// 2-D matrix dump: space-separated columns, one row per line. Human-readable.
// Unlike savetxt (flat, one value per line), this preserves the matrix layout.
// Output is capped to max_rows x max_cols (default 256x256). Pass 0 to dump all.
template <typename T>
void save_matrix_txt(const HostTensor<T>& tensor,
                     const std::string& file_name,
                     const std::string& dtype = "float",
                     std::size_t max_rows     = 256,
                     std::size_t max_cols     = 256)
{
    if(tensor.mDesc.get_num_of_dimension() != 2)
    {
        throw std::runtime_error("save_matrix_txt: tensor must be 2-D, got " +
                                 std::to_string(tensor.mDesc.get_num_of_dimension()) + "-D");
    }

    enum class DType
    {
        Float,
        Int,
        Int8
    };
    DType dt;
    if(dtype == "float")
        dt = DType::Float;
    else if(dtype == "int")
        dt = DType::Int;
    else if(dtype == "int8_t")
        dt = DType::Int8;
    else
        throw std::runtime_error(std::string("save_matrix_txt: unsupported dtype:") + dtype);

    const auto rows = tensor.mDesc.get_lengths()[0];
    const auto cols = tensor.mDesc.get_lengths()[1];

    const auto out_rows = (max_rows == 0) ? rows : std::min(rows, max_rows);
    const auto out_cols = (max_cols == 0) ? cols : std::min(cols, max_cols);

    std::ofstream file(file_name);
    if(!file.is_open())
    {
        throw std::runtime_error(std::string("unable to open file:") + file_name);
    }

    if(out_rows < rows || out_cols < cols)
    {
        file << "# shape: " << rows << "x" << cols << ", showing: " << out_rows << "x" << out_cols
             << "\n";
    }

    for(std::size_t r = 0; r < out_rows; r++)
    {
        for(std::size_t c = 0; c < out_cols; c++)
        {
            if(c > 0)
                file << " ";

            const auto& val = tensor(r, c);
            switch(dt)
            {
            case DType::Int: file << type_convert<int>(val); break;
            case DType::Int8: file << static_cast<int>(type_convert<ck_tile::int8_t>(val)); break;
            case DType::Float: file << type_convert<float>(val); break;
            }
        }
        file << "\n";
    }
    file.close();
}

} // namespace ck_tile
