// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/host.hpp"
#include "ck_tile/ops/batchnorm.hpp"
#include <cstring>
#include <iomanip>

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
template <typename XDataType, typename YDataType, typename ComputeDataType, typename GammaDataType, typename BetaDataType>
void reference_batchnorm_fwd(const ck_tile::HostTensor<XDataType>& x,
                             const ck_tile::HostTensor<GammaDataType>* gamma,
                             const ck_tile::HostTensor<BetaDataType>* beta,
                             ck_tile::HostTensor<YDataType>& y,
                             ck_tile::index_t N,
                             ck_tile::index_t C,
                             ck_tile::index_t H,
                             ck_tile::index_t W,
                             ComputeDataType epsilon)
{
    const ck_tile::index_t spatial_size = H * W;
    const ck_tile::index_t per_channel_size = N * spatial_size;
    
    // Process each channel (compute statistics across ALL samples and spatial positions)
    for(ck_tile::index_t c = 0; c < C; ++c)
    {
        // Compute mean across all N samples and H×W positions for this channel
        ComputeDataType sum = 0;
        for(ck_tile::index_t n = 0; n < N; ++n)
        {
            for(ck_tile::index_t h = 0; h < H; ++h)
            {
                for(ck_tile::index_t w = 0; w < W; ++w)
                {
                    ck_tile::index_t idx = n * C * H * W + c * H * W + h * W + w;
                    sum += ck_tile::type_convert<ComputeDataType>(x.mData[idx]);
                }
            }
        }
        ComputeDataType mean = sum / static_cast<ComputeDataType>(per_channel_size);
        
        // Compute variance across all N samples and H×W positions for this channel
        ComputeDataType var_sum = 0;
        for(ck_tile::index_t n = 0; n < N; ++n)
        {
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
        }
        ComputeDataType variance = var_sum / static_cast<ComputeDataType>(per_channel_size);
        
        // Load gamma and beta for this channel
        ComputeDataType gamma_val = static_cast<ComputeDataType>(1.0);
        ComputeDataType beta_val = static_cast<ComputeDataType>(0.0);
        
        if(gamma != nullptr)
        {
            gamma_val = ck_tile::type_convert<ComputeDataType>(gamma->mData[c]);
        }
        if(beta != nullptr)
        {
            beta_val = ck_tile::type_convert<ComputeDataType>(beta->mData[c]);
        }
        
        // Compute inverse standard deviation
        ComputeDataType inv_std = static_cast<ComputeDataType>(1.0) / 
            ck_tile::sqrt(variance + epsilon);
        
        // Normalize all values in this channel with scale and bias
        for(ck_tile::index_t n = 0; n < N; ++n)
        {
            for(ck_tile::index_t h = 0; h < H; ++h)
            {
                for(ck_tile::index_t w = 0; w < W; ++w)
                {
                    ck_tile::index_t idx = n * C * H * W + c * H * W + h * W + w;
                    ComputeDataType val = ck_tile::type_convert<ComputeDataType>(x.mData[idx]);
                    ComputeDataType normalized = gamma_val * ((val - mean) * inv_std) + beta_val;
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
    ck_tile::HostTensor<ComputeDataType> gamma_host({C});
    ck_tile::HostTensor<ComputeDataType> beta_host({C});
    ck_tile::HostTensor<YDataType> y_host_ref({N, C, H, W});
    ck_tile::HostTensor<YDataType> y_host_dev({N, C, H, W});

    // Fill input with random data
    ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(x_host);
    ck_tile::FillUniformDistribution<ComputeDataType>{0.8f, 1.2f}(gamma_host);  // Scale around 1.0
    ck_tile::FillUniformDistribution<ComputeDataType>{-0.5f, 0.5f}(beta_host);  // Bias around 0.0

    // Allocate device memory
    ck_tile::DeviceMem x_buf(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem gamma_buf(gamma_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem beta_buf(beta_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host_dev.get_element_space_size_in_bytes());

    x_buf.ToDevice(x_host.data());
    gamma_buf.ToDevice(gamma_host.data());
    beta_buf.ToDevice(beta_host.data());

    // Define kernel configuration
    using BlockWarps = ck_tile::sequence<4, 1>;
    using BlockTile  = ck_tile::sequence<1, 256>;  // Simplified for POC
    using WarpTile   = ck_tile::sequence<1, 256>;
    using Vector     = ck_tile::sequence<1, 1>;

    using Shape = ck_tile::BatchnormShape<BlockWarps, BlockTile, WarpTile, Vector>;
    
    // Define traits (compile-time configuration)
    using Traits = ck_tile::BatchnormFwdTraits<false, false>;  // No save, no update
    
    // Define problem with all types
    using Problem = ck_tile::BatchnormProblem<XDataType,       // input type
                                              ComputeDataType, // gamma type
                                              ComputeDataType, // beta type
                                              ComputeDataType, // compute type
                                              YDataType,       // output type
                                              ComputeDataType, // mean/var type
                                              Shape,
                                              Traits>;
    using Kernel = ck_tile::BatchnormFwd<Problem>;

    // Prepare host arguments
    // Note: save/update behavior is determined by Traits (compile-time), not runtime args
    ck_tile::BatchnormFwdHostArgs hargs{
        x_buf.GetDeviceBuffer(),      // p_x
        gamma_buf.GetDeviceBuffer(),  // p_gamma
        beta_buf.GetDeviceBuffer(),   // p_beta
        y_buf.GetDeviceBuffer(),      // p_y
        nullptr,                      // p_running_mean (not used, Traits::kUpdateMovingAverage=false)
        nullptr,                      // p_running_var
        nullptr,                      // p_save_mean (not used, Traits::kSaveMeanInvStd=false)
        nullptr,                      // p_save_inv_std
        epsilon,                      // epsilon
        0.1f,                         // momentum
        N, C, H, W                    // dimensions
    };

    // Validate arguments
    if(!Kernel::IsSupportedArgument(hargs))
    {
        std::cout << "Arguments not supported!" << std::endl;
        return false;
    }

    // Get grid and block size
    const auto grid_size = Kernel::GridSize(hargs);
    const auto block_size = Kernel::BlockSize();
    
    std::cout << "Kernel config: BlockSize=" << block_size << ", GridSize=" << grid_size.x 
              << " (one block per channel, reducing over N×H×W=" << N*H*W << " elements)" << std::endl;

    // Make kernel arguments
    auto kargs = Kernel::MakeKernelArgs(hargs);

    // Launch kernel
    float ave_time = ck_tile::launch_kernel(
        ck_tile::stream_config{nullptr, true, 0, warmup, repeat},
        ck_tile::make_kernel<1>(Kernel{}, grid_size, block_size, 0, kargs));

    std::size_t num_bytes = sizeof(XDataType) * total_size + sizeof(YDataType) * total_size;
    float gb_per_sec = num_bytes / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

    bool pass = true;

    if(do_validation)
    {
        // Compute reference with gamma and beta
        reference_batchnorm_fwd<XDataType, YDataType, ComputeDataType, ComputeDataType, ComputeDataType>(
            x_host, &gamma_host, &beta_host, y_host_ref, N, C, H, W, epsilon);
        
        // Get device result
        y_buf.FromDevice(y_host_dev.mData.data());
        
        // Print sample outputs for each channel (2 samples per channel)
        std::cout << "\n=== Sample Outputs (first 2 values per channel) ===" << std::endl;
        for(ck_tile::index_t c = 0; c < C; ++c)
        {
            std::cout << "Channel " << c << ":" << std::endl;
            
            // Print 2 sample values from first sample (n=0)
            for(ck_tile::index_t sample = 0; sample < 2 && sample < H * W; ++sample)
            {
                ck_tile::index_t idx = 0 * C * H * W + c * H * W + sample;
                float ref_val = ck_tile::type_convert<float>(y_host_ref.mData[idx]);
                float dev_val = ck_tile::type_convert<float>(y_host_dev.mData[idx]);
                std::cout << "  Sample[" << sample << "]: "
                          << "Ref=" << std::fixed << std::setprecision(6) << ref_val
                          << ", Kernel=" << dev_val
                          << ", Diff=" << std::abs(ref_val - dev_val) << std::endl;
            }
        }
        std::cout << std::endl;
        
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
