// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/host.hpp"
#include "ck_tile/ops/batchnorm.hpp"
#include <cstring>
#include <iomanip>

// Batch normalization forward pass - NHWC layout
// NOTE: Using NHWC (not NCHW) for contiguous channel access

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
        .insert("warmup", "10", "cold iter")
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
                             ck_tile::HostTensor<ComputeDataType>* save_mean,
                             ck_tile::HostTensor<ComputeDataType>* save_inv_std,
                             ck_tile::HostTensor<ComputeDataType>* running_mean,
                             ck_tile::HostTensor<ComputeDataType>* running_var,
                             ComputeDataType momentum,
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
                    ck_tile::index_t idx = n*H*W*C + h*W*C + w*C + c;  // NHWC indexing
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
                    ck_tile::index_t idx = n*H*W*C + h*W*C + w*C + c;  // NHWC
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
        
        // Save mean and inv_std if requested
        if(save_mean != nullptr)
        {
            save_mean->mData[c] = mean;
        }
        if(save_inv_std != nullptr)
        {
            save_inv_std->mData[c] = inv_std;
        }
        
        // Update running statistics if requested
        if(running_mean != nullptr && running_var != nullptr)
        {
            running_mean->mData[c] = (1.0f - momentum) * running_mean->mData[c] + momentum * mean;
            running_var->mData[c] = (1.0f - momentum) * running_var->mData[c] + momentum * variance;
        }
        
        // Normalize all values in this channel with scale and bias
        for(ck_tile::index_t n = 0; n < N; ++n)
        {
            for(ck_tile::index_t h = 0; h < H; ++h)
            {
                for(ck_tile::index_t w = 0; w < W; ++w)
                {
                    ck_tile::index_t idx = n*H*W*C + h*W*C + w*C + c;  // NHWC
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

    // Allocate host tensors in NHWC layout
    ck_tile::index_t total_size = N * C * H * W;
    ck_tile::HostTensor<XDataType> x_host({N, H, W, C});  // NHWC!
    ck_tile::HostTensor<ComputeDataType> gamma_host({C});
    ck_tile::HostTensor<ComputeDataType> beta_host({C});
    ck_tile::HostTensor<YDataType> y_host_ref({N, H, W, C});  // NHWC!
    ck_tile::HostTensor<YDataType> y_host_dev({N, H, W, C});  // NHWC!
    
    // Allocate buffers for optional features
    ck_tile::HostTensor<ComputeDataType> save_mean_host({C});
    ck_tile::HostTensor<ComputeDataType> save_inv_std_host({C});
    ck_tile::HostTensor<ComputeDataType> running_mean_host({C});
    ck_tile::HostTensor<ComputeDataType> running_var_host({C});
    
    // Initialize running statistics
    ck_tile::FillUniformDistribution<ComputeDataType>{0.0f, 0.0f}(running_mean_host);  // Start at 0
    ck_tile::FillUniformDistribution<ComputeDataType>{1.0f, 1.0f}(running_var_host);   // Start at 1

    // Fill input with random data
    ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(x_host);
    
    // Fill gamma and beta with random values (test scale/bias)
    ck_tile::FillUniformDistribution<ComputeDataType>{0.8f, 1.2f}(gamma_host);  // Scale around 1
    ck_tile::FillUniformDistribution<ComputeDataType>{-0.5f, 0.5f}(beta_host);  // Bias around 0

    // Allocate device memory
    ck_tile::DeviceMem x_buf(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem gamma_buf(gamma_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem beta_buf(beta_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host_dev.get_element_space_size_in_bytes());
    ck_tile::DeviceMem save_mean_buf(save_mean_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem save_inv_std_buf(save_inv_std_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem running_mean_buf(running_mean_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem running_var_buf(running_var_host.get_element_space_size_in_bytes());

    x_buf.ToDevice(x_host.data());
    gamma_buf.ToDevice(gamma_host.data());
    beta_buf.ToDevice(beta_host.data());
    running_mean_buf.ToDevice(running_mean_host.data());
    running_var_buf.ToDevice(running_var_host.data());

    // Define kernel configuration using Generic2dBlockShape
    // Vector_N controls vectorization: higher = fewer iterations, more elements per thread
    // Block_N = ThreadPerBlock_N × Vector_N (must match tile size needed)
    using BlockTile = ck_tile::sequence<1, 256>;      // Block size: 1 channel, 128 spatial
    using ThreadPerBlock = ck_tile::sequence<1, 128>;   // 64 threads
    using Vector = ck_tile::sequence<1, 1>;            // Vector_N=2 (try 1,2,4,8)
    
    // With Vector_N=2: 64 threads × 2 elements = 128 elements per tile
    // With Vector_N=4: Need ThreadPerBlock=32 for 32×4=128
    // Experiment to find optimal!

    using Shape = ck_tile::BatchnormShape<BlockTile, ThreadPerBlock, Vector>;
    
    // Feature flags - change these to enable/disable testing different features
    constexpr bool kSaveMeanInvStd = true;       // Set true to test save for backward
    constexpr bool kUpdateMovingAverage = true;  // Set true to test running stats
    
    using Traits = ck_tile::BatchnormFwdTraits<kSaveMeanInvStd, kUpdateMovingAverage>;
    
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
        x_buf.GetDeviceBuffer(),              // p_x
        gamma_buf.GetDeviceBuffer(),          // p_gamma
        beta_buf.GetDeviceBuffer(),           // p_beta
        y_buf.GetDeviceBuffer(),              // p_y
        running_mean_buf.GetDeviceBuffer(),   // p_running_mean (now used!)
        running_var_buf.GetDeviceBuffer(),    // p_running_var
        save_mean_buf.GetDeviceBuffer(),      // p_save_mean (now used!)
        save_inv_std_buf.GetDeviceBuffer(),   // p_save_inv_std
        epsilon,                              // epsilon
        0.1f,                                 // momentum
        N, C, H, W                            // dimensions
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
        // Compute reference (will also save/update statistics)
        ck_tile::HostTensor<ComputeDataType> save_mean_ref({C});
        ck_tile::HostTensor<ComputeDataType> save_inv_std_ref({C});
        ck_tile::HostTensor<ComputeDataType> running_mean_ref({C});
        ck_tile::HostTensor<ComputeDataType> running_var_ref({C});
        
        // Copy initial running stats
        std::copy(running_mean_host.mData.begin(), running_mean_host.mData.end(), running_mean_ref.mData.begin());
        std::copy(running_var_host.mData.begin(), running_var_host.mData.end(), running_var_ref.mData.begin());
        
        reference_batchnorm_fwd<XDataType, YDataType, ComputeDataType, ComputeDataType, ComputeDataType>(
            x_host, &gamma_host, &beta_host, y_host_ref, 
            &save_mean_ref, &save_inv_std_ref,
            &running_mean_ref, &running_var_ref,
            0.1f, N, C, H, W, epsilon);
        
        // Get device result
        y_buf.FromDevice(y_host_dev.mData.data());
        
        // Print sample outputs for each channel (1 random sample per channel)
        std::cout << "\n=== Sample Outputs (1 random sample per channel) ===" << std::endl;
        std::srand(42);  // Fixed seed for reproducibility
        for(ck_tile::index_t c = 0; c < C; ++c)
        {
            // Pick random n, h, w for this channel
            ck_tile::index_t rand_n = std::rand() % N;
            ck_tile::index_t rand_h = std::rand() % H;
            ck_tile::index_t rand_w = std::rand() % W;
            
            ck_tile::index_t idx = rand_n*H*W*C + rand_h*W*C + rand_w*C + c;  // NHWC
            float ref_val = ck_tile::type_convert<float>(y_host_ref.mData[idx]);
            float dev_val = ck_tile::type_convert<float>(y_host_dev.mData[idx]);
            
            std::cout << "Ch" << std::setw(3) << c 
                      << " [n=" << std::setw(4) << rand_n 
                      << ",h=" << std::setw(4) << rand_h 
                      << ",w=" << std::setw(4) << rand_w << "]: "
                      << "Ref=" << std::fixed << std::setprecision(6) << std::setw(10) << ref_val
                      << " Kernel=" << std::setw(10) << dev_val
                      << " Diff=" << std::setw(10) << std::abs(ref_val - dev_val) << std::endl;
        }
        std::cout << std::endl;
        
        // Check output
        pass = ck_tile::check_err(y_host_dev, y_host_ref, "Error: Incorrect results!", 1e-2, 1e-2);
        
        // Conditionally verify features based on what's enabled
        if constexpr(kSaveMeanInvStd)
        {
            save_mean_buf.FromDevice(save_mean_host.mData.data());
            save_inv_std_buf.FromDevice(save_inv_std_host.mData.data());
            
            bool save_pass = ck_tile::check_err(save_mean_host, save_mean_ref, "Error: Saved mean incorrect!", 1e-3, 1e-3);
            save_pass = save_pass && ck_tile::check_err(save_inv_std_host, save_inv_std_ref, "Error: Saved inv_std incorrect!", 1e-3, 1e-3);
            
            std::cout << "\n=== Saved Statistics ===" << std::endl;
            for(ck_tile::index_t c = 0; c < std::min(C, ck_tile::index_t(4)); ++c)
            {
                std::cout << "Ch" << std::setw(2) << c 
                          << " mean: Ref=" << std::setw(10) << save_mean_ref.mData[c]
                          << " Dev=" << std::setw(10) << save_mean_host.mData[c]
                          << " | inv_std: Ref=" << std::setw(10) << save_inv_std_ref.mData[c]
                          << " Dev=" << std::setw(10) << save_inv_std_host.mData[c] << std::endl;
            }
            pass = pass && save_pass;
        }
        
        if constexpr(kUpdateMovingAverage)
        {
            if(repeat == 1)
            {
                running_mean_buf.FromDevice(running_mean_host.mData.data());
                running_var_buf.FromDevice(running_var_host.mData.data());
                
                bool running_pass = ck_tile::check_err(running_mean_host, running_mean_ref, "Error: Running mean incorrect!", 1e-3, 1e-3);
                running_pass = running_pass && ck_tile::check_err(running_var_host, running_var_ref, "Error: Running var incorrect!", 1e-3, 1e-3);
                
                std::cout << "\n=== Running Statistics ===" << std::endl;
                for(ck_tile::index_t c = 0; c < std::min(C, ck_tile::index_t(4)); ++c)
                {
                    std::cout << "Ch" << std::setw(2) << c 
                              << " mean: Ref=" << std::setw(10) << running_mean_ref.mData[c]
                              << " Dev=" << std::setw(10) << running_mean_host.mData[c]
                              << " | var: Ref=" << std::setw(10) << running_var_ref.mData[c]
                              << " Dev=" << std::setw(10) << running_var_host.mData[c] << std::endl;
                }
                pass = pass && running_pass;
            }
            else
            {
                std::cout << "\nNOTE: Running statistics validation requires -warmup=0 -repeat=1" << std::endl;
                std::cout << "(Multiple iterations accumulate running stats, making validation incorrect)" << std::endl;
            }
        }
        std::cout << std::endl;
        
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
