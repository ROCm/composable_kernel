// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/host.hpp"

constexpr int MaxSendGPUNum = 7;

struct transfer_receive_basic_args
{
    const void* p_reduce;
    std::array<const void*, MaxSendGPUNum> p_receive_list;
    const void* p_output;
    ck_tile::index_t host_gpu;
    ck_tile::index_t device_id;
    ck_tile::index_t M;
    ck_tile::index_t N;
};

struct transfer_send_basic_args
{
    const void* p_reduce;
    ck_tile::index_t host_gpu;
    ck_tile::index_t device_id;
    ck_tile::index_t M;
    ck_tile::index_t N;
};

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("gpu_nums", "2", "number of gpu")
        .insert("transfer_size", "2048", "transfer memory size")
        .insert("pr", "fp16", "input data type. fp16/fp32 (representing 8/16/32 bit data)")
        .insert(
            "output_type", "float", "output data type. fp16/fp32 (representing 8/16/32 bit data)")
        .insert("M", "1024", "transfer memory first dimension")
        .insert("N", "2", "transfer memory second dimension")
        .insert("op_type", "reduce_add", "Operation type between different GPUs")
        .insert("host_gpu", "0", "host gpu #")
        .insert("v", "1", "cpu validation or not")
        .insert("warmup", "50", "number of iterations before benchmark the kernel")
        .insert("repeat", "100", "number of iterations to benchmark the kernel");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}
