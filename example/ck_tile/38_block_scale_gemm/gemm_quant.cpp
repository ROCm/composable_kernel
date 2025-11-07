// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

// This example demonstrates 2D block scale quantization (N×K) for BQuant
// using non-preshuffled configuration.
// NOTE: Once more 2d support is ready, we can migrate all 2d quant types to this example
// This is currently done separately to avoid too verbose dispatching.

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <stdexcept>
#include <string>
#include <tuple>

#include "ck_tile/core/config.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/permute_pk_int4.hpp"
#include "ck_tile/host/tensor_shuffle_utils.hpp"
#include "gemm_utils.hpp"

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("h", "false", "print help message")
        .insert("m", "3840", "m dimension")
        .insert("n", "4096", "n dimension")
        .insert("k", "2048", "k dimension")
        .insert("a_layout", "R", "A tensor data layout - Row by default")
        .insert("b_layout", "C", "B tensor data layout - Column by default")
        .insert("bq_layout", "C", "Bq tensor data layout - Column by default")
        .insert("c_layout", "R", "C tensor data layout - Row by default")
        .insert("stride_a", "0", "Tensor A stride")
        .insert("stride_q", "0", "Tensor AQ stride")
        .insert("stride_b", "0", "Tensor B stride")
        .insert("stride_c", "0", "Tensor C stride")
        .insert("v", "1", "0. No validation, 1. Validation on CPU, 2. Validation on GPU")
        .insert("prec",
                "fp8",
                "data type. For AQuant: fp8/bf8/i4fp8/i4bf8, For Bquant: fp8/bf8/fp8i4/bf8i4")
        .insert("warmup", "50", "number of iterations before benchmark the kernel")
        .insert("repeat", "1000", "number of iterations to benchmark the kernel")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer")
        .insert("split_k", "1", "splitK value")
        .insert("device", "0", "device id that will be used to run the kernel, default 0")
        .insert("init", "0", "0:random, 1:linear, 2:constant(1)")
        .insert("flush_cache", "true", "flush cache before running the kernel, default to true")
        .insert("rotating_count", "1000", "rotating count, defaults to 1")
        .insert("quant_mode", "bquant", "Choose aquant (default), bquant, tensor or rowcol")
        .insert("preshuffleb", "false", "Enable preshuffle tensor B, default false")
        .insert("group_size",
                "1x1x128",
                "Quantization group size as MxNxK, e.g., 1x1x128, 1x32x128, 1x64x128");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

void aquant_quantgrouped_instance_factory(
    std::unordered_map<size_t, std::function<int(ck_tile::ArgParser&)>>& lut);
void bquant_quantgrouped_fp8_instance_factory(
    std::unordered_map<size_t, std::function<int(ck_tile::ArgParser&)>>& lut);
void bquant_quantgrouped_bf8_instance_factory(
    std::unordered_map<size_t, std::function<int(ck_tile::ArgParser&)>>& lut);
void bquant_quantgrouped_fp8i4_instance_factory(
    std::unordered_map<size_t, std::function<int(ck_tile::ArgParser&)>>& lut);
void bquant_quantgrouped_bf8i4_instance_factory(
    std::unordered_map<size_t, std::function<int(ck_tile::ArgParser&)>>& lut);
void bquant_quantgrouped_preshuffleb_instance_factory(
    std::unordered_map<size_t, std::function<int(ck_tile::ArgParser&)>>& lut);

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result || arg_parser.get_bool("h"))
    {
        arg_parser.print();
        return -1;
    }

    auto device_id = arg_parser.get_int("device");
    std::printf("Device ID: %d\n", device_id);

    hipError_t err = hipSetDevice(device_id);
    if(err != hipSuccess)
    {
        std::cerr << "hipSetDevice failed with error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    std::unordered_map<size_t, std::function<int(ck_tile::ArgParser&)>> lut;
    aquant_quantgrouped_instance_factory(lut);
    bquant_quantgrouped_fp8_instance_factory(lut);
    bquant_quantgrouped_bf8_instance_factory(lut);
    bquant_quantgrouped_fp8i4_instance_factory(lut);
    bquant_quantgrouped_bf8i4_instance_factory(lut);
    bquant_quantgrouped_preshuffleb_instance_factory(lut);

    std::string data_type  = arg_parser.get_str("prec");
    std::string quant_mode = arg_parser.get_str("quant_mode");
    std::string preshuffleb =
        arg_parser.get_bool("preshuffleb") ? "preshuffleb" : "non-preshuffleb";
    std::string group_size_str = arg_parser.get_str("group_size");

    auto key = hash_multiple_strings({data_type, quant_mode, preshuffleb, group_size_str});
    if(lut.find(key) != lut.end())
    {
        return lut[key](arg_parser);
    }
    else
    {
        std::cerr
            << "Error: Combination of prec, quant_mode, preshuffleb, and group_size not supported."
            << " (prec: " << data_type << ", quant_mode: " << quant_mode
            << ", preshuffleb: " << preshuffleb << ", group_size: " << group_size_str << ")"
            << std::endl;
        return -1;
    }
}
