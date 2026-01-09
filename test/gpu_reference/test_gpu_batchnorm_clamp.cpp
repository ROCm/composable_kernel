// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <cmath>

#include "ck/ck.hpp"
#include "ck/utility/type_convert.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/gpu/naive_conv_fwd_gpu.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

using namespace ck;
using BaseConv = ck::tensor_layout::convolution::BaseConvolutionLayout;

// CPU reference implementation matching profiler's ref_bnorm_clamp_infer
template <typename DataType>
void cpu_batchnorm_clamp_ref(Tensor<DataType>& out,
                             const Tensor<DataType>& in,
                             const Tensor<DataType>& mean,
                             const Tensor<DataType>& variance,
                             const Tensor<DataType>& scale,
                             const Tensor<DataType>& shift,
                             float floor,
                             float ceil,
                             float epsilon)
{
    using Clamp = tensor_operation::element_wise::Clamp;

    auto func = [&](auto g, auto n, auto k, auto h, auto w) {
        const float x = type_convert<float>(in(g, n, k, h, w));
        const float invVariance =
            1.0f / std::sqrt(epsilon + type_convert<float>(variance(g, n, k, h, w)));
        const float norm_x = (x - type_convert<float>(mean(g, n, k, h, w))) * invVariance;
        float y            = type_convert<float>(scale(g, n, k, h, w)) * norm_x +
                  type_convert<float>(shift(g, n, k, h, w));
        Clamp{floor, ceil}(y, y);
        out(g, n, k, h, w) = type_convert<DataType>(y);
    };

    make_ParallelTensorFunctor(func,
                               out.GetLengths()[0],
                               out.GetLengths()[1],
                               out.GetLengths()[2],
                               out.GetLengths()[3],
                               out.GetLengths()[4])(std::thread::hardware_concurrency());
}

void run_batchnorm_test(index_t G,
                        index_t N,
                        index_t K,
                        index_t H,
                        index_t W,
                        bool use_gk_broadcast,
                        float epsilon,
                        float floor,
                        float ceil_val,
                        float tolerance = 1e-3f)
{
    const long_index_t total_elements = G * N * K * H * W;
    const long_index_t param_size     = use_gk_broadcast ? G * K : total_elements;

    // Create tensor descriptors
    auto out_desc =
        HostTensorDescriptor({G, N, K, H, W}, {N * K * H * W, K * H * W, H * W, W, 1}, BaseConv{});

    // Parameter descriptor: GK broadcast or full size
    auto param_desc = use_gk_broadcast
                          ? HostTensorDescriptor({G, 1, K, 1, 1}, {K, 0, 1, 0, 0}, BaseConv{})
                          : out_desc;

    Tensor<float> in_tensor(out_desc);
    Tensor<float> out_cpu_tensor(out_desc);
    Tensor<float> out_gpu_tensor(out_desc);
    Tensor<float> mean_tensor(param_desc);
    Tensor<float> variance_tensor(param_desc);
    Tensor<float> scale_tensor(param_desc);
    Tensor<float> shift_tensor(param_desc);

    // Initialize tensors
    in_tensor.GenerateTensorValue(GeneratorTensor_2<float>{-5, 5});
    mean_tensor.GenerateTensorValue(GeneratorTensor_2<float>{-5, 5});
    variance_tensor.GenerateTensorValue(GeneratorTensor_2<float>{0, 5});
    scale_tensor.GenerateTensorValue(GeneratorTensor_2<float>{-5, 5});
    shift_tensor.GenerateTensorValue(GeneratorTensor_2<float>{-5, 5});

    // CPU reference
    cpu_batchnorm_clamp_ref(out_cpu_tensor,
                            in_tensor,
                            mean_tensor,
                            variance_tensor,
                            scale_tensor,
                            shift_tensor,
                            floor,
                            ceil_val,
                            epsilon);

    // GPU version
    DeviceMem d_in(total_elements * sizeof(float));
    DeviceMem d_mean(param_size * sizeof(float));
    DeviceMem d_variance(param_size * sizeof(float));
    DeviceMem d_scale(param_size * sizeof(float));
    DeviceMem d_shift(param_size * sizeof(float));
    DeviceMem d_out(total_elements * sizeof(float));

    d_in.ToDevice(in_tensor.mData.data());
    d_mean.ToDevice(mean_tensor.mData.data());
    d_variance.ToDevice(variance_tensor.mData.data());
    d_scale.ToDevice(scale_tensor.mData.data());
    d_shift.ToDevice(shift_tensor.mData.data());

    // Setup strides
    std::vector<index_t> tensor_lengths = {G, N, K, H, W};
    std::vector<index_t> tensor_strides = {N * K * H * W, K * H * W, H * W, W, 1};

    // Parameter strides: GK broadcast uses zero strides for N, H, W
    std::vector<index_t> param_strides =
        use_gk_broadcast ? std::vector<index_t>{K, 0, 1, 0, 0} : tensor_strides;

    ref::naive_batchnorm_clamp_infer_gpu(
        reinterpret_cast<float*>(d_out.GetDeviceBuffer()),
        reinterpret_cast<const float*>(d_in.GetDeviceBuffer()),
        reinterpret_cast<const float*>(d_mean.GetDeviceBuffer()),
        reinterpret_cast<const float*>(d_variance.GetDeviceBuffer()),
        reinterpret_cast<const float*>(d_scale.GetDeviceBuffer()),
        reinterpret_cast<const float*>(d_shift.GetDeviceBuffer()),
        tensor_lengths,
        param_strides,
        tensor_strides,
        total_elements,
        epsilon,
        floor,
        ceil_val);

    d_out.FromDevice(out_gpu_tensor.mData.data());

    // Verify results
    bool pass       = true;
    float max_diff  = 0.0f;
    int error_count = 0;

    for(long_index_t i = 0; i < total_elements; ++i)
    {
        float diff = std::abs(out_gpu_tensor.mData[i] - out_cpu_tensor.mData[i]);
        max_diff   = std::max(max_diff, diff);

        if(diff > tolerance)
        {
            error_count++;
            pass = false;
        }
    }

    EXPECT_TRUE(pass) << "GPU batchnorm produced " << error_count << " errors out of "
                      << total_elements << " elements (max diff: " << max_diff << ")";
}

TEST(GpuBatchnormClamp, SmallDimensions)
{
    run_batchnorm_test(
        /* G */ 2,
        /* N */ 3,
        /* K */ 4,
        /* H */ 2,
        /* W */ 2,
        /* use_gk_broadcast */ true,
        /* epsilon */ 1e-4f,
        /* floor */ 0.0f,
        /* ceil */ 100.0f);
}

TEST(GpuBatchnormClamp, RealDimensions_GKBroadcast)
{
    // Dimensions from test_grouped_convnd_fwd_bias_bnorm_clamp
    run_batchnorm_test(
        /* G */ 2,
        /* N */ 32,
        /* K */ 256,
        /* H */ 4,
        /* W */ 4,
        /* use_gk_broadcast */ true,
        /* epsilon */ 1e-4f,
        /* floor */ 0.0f,
        /* ceil */ 2048.0f);
}

TEST(GpuBatchnormClamp, FullStrides_NoBroadcast)
{
    run_batchnorm_test(
        /* G */ 2,
        /* N */ 32,
        /* K */ 256,
        /* H */ 4,
        /* W */ 4,
        /* use_gk_broadcast */ false,
        /* epsilon */ 1e-4f,
        /* floor */ 0.0f,
        /* ceil */ 2048.0f);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
