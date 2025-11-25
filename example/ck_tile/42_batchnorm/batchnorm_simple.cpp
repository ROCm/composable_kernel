// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/host.hpp"
#include "ck_tile/ops/batchnorm.hpp"
#include <cstring>

// Simple POC for batchnorm forward pass
// Tests basic functionality with a small tensor

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("n", "2", "batch size")
        .insert("c", "4", "number of channels")
        .insert("h", "8", "height")
        .insert("w", "8", "width")
        .insert("e", "1e-5", "epsilon")
        .insert("v", "1", "cpu validation or not")
        .insert("prec", "fp16", "precision")
        .insert("warmup", "5", "cold iter")
        .insert("repeat", "20", "hot iter");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

// CPU reference implementation
template <typename XDataType, typename YDataType, typename ComputeDataType>
void reference_batchnorm_fwd(const ck_tile::HostTensor<XDataType>& x,
                             ck_tile::HostTensor<YDataType>& y,
                             ck_tile::index_t N,
                             ck_tile::index_t C,
                             ck_tile::index_t H,
                             ck_tile::index_t W,
                             ComputeDataType epsilon)
{
    const ck_tile::index_t spatial_size = H * W;
    
    // Process each (N, C) combination
    for(ck_tile::index_t n = 0; n < N; ++n)
    {
        for(ck_tile::index_t c = 0; c < C; ++c)
        {
            // Compute mean
            ComputeDataType sum = 0;
            for(ck_tile::index_t h = 0; h < H; ++h)
            {
                for(ck_tile::index_t w = 0; w < W; ++w)
                {
                    ck_tile::index_t idx = n * C * H * W + c * H * W + h * W + w;
                    sum += ck_tile::type_convert<ComputeDataType>(x.mData[idx]);
                }
            }
            ComputeDataType mean = sum / static_cast<ComputeDataType>(spatial_size);
            
            // Compute variance
            ComputeDataType var_sum = 0;
            for(ck_tile::index_t h = 0; h < H; ++h)
            {
                for(ck_tile::index_t w = 0; w < W; ++w)
                {
                    ck_tile::index_t idx = n * C * H * W + c * H * W + h * W + w;
                    ComputeDataType val = ck_tile::type_convert<ComputeDataType>(x.mData[idx]);
                    ComputeDataType diff = val - mean;
                    var_sum += diff * diff;
                }
            }
            ComputeDataType variance = var_sum / static_cast<ComputeDataType>(spatial_size);
            
            // Normalize
            ComputeDataType inv_std = static_cast<ComputeDataType>(1.0) / 
                ck_tile::sqrt(variance + epsilon);
            
            for(ck_tile::index_t h = 0; h < H; ++h)
            {
                for(ck_tile::index_t w = 0; w < W; ++w)
                {
                    ck_tile::index_t idx = n * C * H * W + c * H * W + h * W + w;
                    ComputeDataType val = ck_tile::type_convert<ComputeDataType>(x.mData[idx]);
                    ComputeDataType normalized = (val - mean) * inv_std;
                    y.mData[idx] = ck_tile::type_convert<YDataType>(normalized);
                }
            }
        }
    }
}

template <typename DataType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    using XDataType       = DataType;
    using ComputeDataType = float;
    using YDataType       = DataType;

    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t C = arg_parser.get_int("c");
    ck_tile::index_t H = arg_parser.get_int("h");
    ck_tile::index_t W = arg_parser.get_int("w");
    float epsilon      = arg_parser.get_float("e");
    int do_validation  = arg_parser.get_int("v");
    int warmup         = arg_parser.get_int("warmup");
    int repeat         = arg_parser.get_int("repeat");

    std::cout << "Batchnorm POC: N=" << N << ", C=" << C << ", H=" << H << ", W=" << W 
              << ", epsilon=" << epsilon << std::endl;

    // Allocate host tensors
    ck_tile::index_t total_size = N * C * H * W;
    ck_tile::HostTensor<XDataType> x_host({N, C, H, W});
    ck_tile::HostTensor<YDataType> y_host_ref({N, C, H, W});
    ck_tile::HostTensor<YDataType> y_host_dev({N, C, H, W});

    // Fill input with random data
    ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(x_host);

    // Allocate device memory
    ck_tile::DeviceMem x_buf(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host_dev.get_element_space_size_in_bytes());

    x_buf.ToDevice(x_host.data());

    // Define kernel configuration
    using BlockWarps = ck_tile::sequence<4, 1>;
    using BlockTile  = ck_tile::sequence<1, 256>;  // Simplified for POC
    using WarpTile   = ck_tile::sequence<1, 256>;
    using Vector     = ck_tile::sequence<1, 1>;

    using Shape = ck_tile::BatchnormShape<BlockWarps, BlockTile, WarpTile, Vector>;
    using Problem = ck_tile::BatchnormProblem<XDataType, ComputeDataType, YDataType, Shape>;
    using Kernel = ck_tile::BatchnormFwd<Problem>;

    const ck_tile::index_t kBlockSize = Kernel::BlockSize();
    const ck_tile::index_t kGridSize = N * C;  // One block per (N, C) pair

    std::cout << "Kernel config: BlockSize=" << kBlockSize << ", GridSize=" << kGridSize << std::endl;

    if(!Kernel::IsSupportedArgument(N, C, H, W))
    {
        std::cout << "Arguments not supported!" << std::endl;
        return false;
    }

    // Launch kernel
    float ave_time = ck_tile::launch_kernel(
        ck_tile::stream_config{nullptr, true, 0, warmup, repeat},
        ck_tile::make_kernel<1>(
            Kernel{},
            kGridSize,
            kBlockSize,
            0,
            static_cast<const XDataType*>(x_buf.GetDeviceBuffer()),
            static_cast<YDataType*>(y_buf.GetDeviceBuffer()),
            N,
            C,
            H,
            W,
            static_cast<ComputeDataType>(epsilon)));

    std::size_t num_bytes = sizeof(XDataType) * total_size + sizeof(YDataType) * total_size;
    float gb_per_sec = num_bytes / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

    bool pass = true;

    if(do_validation)
    {
        // Compute reference
        reference_batchnorm_fwd<XDataType, YDataType, ComputeDataType>(
            x_host, y_host_ref, N, C, H, W, epsilon);
        
        // Get device result
        y_buf.FromDevice(y_host_dev.mData.data());
        
        // Check error
        pass = ck_tile::check_err(y_host_dev, y_host_ref, "Error: Incorrect results!", 1e-2, 1e-2);
        
        std::cout << "Validation: " << (pass ? "PASSED" : "FAILED") << std::endl;
    }

    return pass;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    const std::string data_type = arg_parser.get_str("prec");

    if(data_type == "fp16")
    {
        return run<ck_tile::half_t>(arg_parser) ? 0 : -2;
    }
    else if(data_type == "bf16")
    {
        return run<ck_tile::bf16_t>(arg_parser) ? 0 : -2;
    }
    else if(data_type == "fp32")
    {
        return run<float>(arg_parser) ? 0 : -2;
    }
    else
    {
        std::cout << "Unsupported data type: " << data_type << std::endl;
        return -3;
    }
}
