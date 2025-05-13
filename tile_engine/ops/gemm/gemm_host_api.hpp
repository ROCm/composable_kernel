// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>

#include <hip/hip_runtime.h>

#include "ck_tile/ops/gemm.hpp"

template <typename T>
struct DataTypeTraits;

template <>
struct DataTypeTraits<float>
{
    static constexpr const char* name = "fp32";
};

template <>
struct DataTypeTraits<double>
{
    static constexpr const char* name = "fp64";
};

template <>
struct DataTypeTraits<ck_tile::half_t>
{
    static constexpr const char* name = "fp16";
};

template <>
struct DataTypeTraits<ck_tile::bf16_t>
{
    static constexpr const char* name = "bf16";
};

template <>
struct DataTypeTraits<ck_tile::fp8_t>
{
    static constexpr const char* name = "fp8";
};

template <>
struct DataTypeTraits<ck_tile::bf8_t>
{
    static constexpr const char* name = "bf8";
};

template <>
struct DataTypeTraits<ck_tile::pk_int4_t>
{
    static constexpr const char* name = "pk_int4_t";
};

/// @brief Defines the configuration parameters for a GEMM operation, enabling the selection of a
/// specific kernel instance based on the provided settings.
struct KernelTraits
{
    /// @brief The name of the pipeline.
    std::string pipeline;
    /// @brief The name of the scheduler (e.g., "intrawave", "interwave").
    std::string scheduler;
    /// @brief The name of the epilogue (e.g., "cshuffle", "default").
    std::string epilogue;
    /// @brief Indicates whether padding is applied to the M dimension.
    bool pad_m;
    /// @brief Indicates whether padding is applied to the N dimension.
    bool pad_n;
    /// @brief Indicates whether padding is applied to the K dimension.
    bool pad_k;
};

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
    // Calculate thresholds
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));
    // Calculate error due to split_k accumulation
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
        max_accumulated_value, kbatch);
    // Use higher threshold
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

inline auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3840", "The value for m dimension. Default is 3840.")
        .insert("n", "4096", "The value for n dimension. Default is 4096.")
        .insert("k", "2048", "The value for k dimension. Default is 2048.")
        .insert("stride_a", "0", "The stride value for tensor A. Default is 0.")
        .insert("stride_b", "0", "The stride value for tensor B. Default is 0.")
        .insert("stride_c", "0", "The stride value for tensor C  Default is 0.")
        .insert("split_k", "1", "The split value for k dimension. Default is 1.")
        .insert("v",
                "2",
                "The type of validation. Set to 0 for no validation, 1 for validation on CPU, or 2 "
                "for validation on GPU. Default is 2, validation on GPU.")
        .insert(
            "warmup", "50", "The number of iterations before benchmark the kernel. Default is 50.")
        .insert(
            "repeat", "100", "The number of iterations to benchmark the kernel. Default is 100.")
        .insert(
            "timer",
            "gpu",
            "The type of timer. Possible values are gpu timer or cpu timer. Default is gpu timer.")
        .insert("init",
                "0",
                "The method of tensor initialization. Set to 0 for random, to 1 for linear, or 2 "
                "for constant(1). Default is 0, random.")
        .insert("metric",
                "0",
                "Metric with which to measure kernel performance. Set to 0 for latency, 1 for "
                "tflops, or 2 for bandwidth. Default is 0, latency.")
        .insert("structured_sparsity",
                "false",
                "Whether use sparsity kernel or not. Possible values are true or false. Default is "
                "false")
        .insert(
            "pipeline",
            "compv3",
            "The type of pipeline. Possible values are compv3, compv4 or mem. Default is compv3.")
        .insert("scheduler",
                "intrawave",
                "The type of pipeline. Possible values are compv3, compv4 or mem. Default is "
                "compv3.")
        .insert(
            "epilogue",
            "cshuffle",
            "The type of epilogue. Possible values are cshuffle or default. Default is csshuffle.")
        .insert("pad_m",
                "false",
                "Whether pad or not in m direction. Possible values are true or false. Default is "
                "false.")
        .insert("pad_n",
                "false",
                "Whether pad or not in n direction. Possible values are true or false. Default is "
                "false.")
        .insert("pad_k",
                "false",
                "Whether pad or not in k direction. Possible values are true or false. Default is "
                "false.");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename Tensor>
void permute_vectors_i4x4_b(Tensor& tensor)
{
    const ck_tile::index_t K = tensor.get_length(0);
    const ck_tile::index_t N = tensor.get_length(1);
    // vector pk_i4x4 permute
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < K; j += 8)
        {
            int8_t input[8];

            for(int k = 0; k < 4; k++)
            {
                int8_t i4x2      = tensor(j + k * 2, i).data;
                input[k * 2 + 0] = (i4x2 >> 4) & 0xf;
                input[k * 2 + 1] = (i4x2 >> 0) & 0xf;
            }

            // permute 01234567->20643175
            {
                int8_t hi   = input[2];
                int8_t lo   = input[0];
                int8_t i4x2 = (hi << 4) | lo;

                tensor(j + 0, i) = i4x2;
            }

            {
                int8_t hi   = input[6];
                int8_t lo   = input[4];
                int8_t i4x2 = (hi << 4) | lo;

                tensor(j + 2, i) = i4x2;
            }

            {
                int8_t hi   = input[3];
                int8_t lo   = input[1];
                int8_t i4x2 = (hi << 4) | lo;

                tensor(j + 4, i) = i4x2;
            }

            {
                int8_t hi   = input[7];
                int8_t lo   = input[5];
                int8_t i4x2 = (hi << 4) | lo;

                tensor(j + 6, i) = i4x2;
            }
        }
    }
}

/// @brief Function to compare the results of the device and host computations
bool compare(ck_tile::index_t K,
             ck_tile::index_t kbatch,
             ck_tile::HostTensor<CDataType>& c_m_n_dev_result,
             ck_tile::HostTensor<CDataType>& c_m_n_host_result)
{
    const float max_accumulated_value =
        *std::max_element(c_m_n_host_result.mData.begin(), c_m_n_host_result.mData.end());
    const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
        K, kbatch, max_accumulated_value);
    bool pass = ck_tile::check_err(c_m_n_dev_result,
                                   c_m_n_host_result,
                                   "Error: Incorrect results!",
                                   rtol_atol.at(ck_tile::number<0>{}),
                                   rtol_atol.at(ck_tile::number<1>{}));

    std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
              << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{}) << std::endl;
    std::cout << "The verification result is:" << (pass ? "correct" : "fail") << std::endl;

    return pass;
}

/// @brief Function to get the kernel output with reference implementation on CPU/GPU
template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
void gemm_host_reference(int verify,
                         ck_tile::HostTensor<ADataType>& a_m_k,
                         ck_tile::HostTensor<BDataType>& b_k_n,
                         ck_tile::HostTensor<CDataType>& c_m_n_host_result,
                         ck_tile::DeviceMem& a_m_k_dev_buf,
                         ck_tile::DeviceMem& b_k_n_dev_buf,
                         ck_tile::index_t M,
                         ck_tile::index_t N,
                         ck_tile::index_t K,
                         ck_tile::index_t stride_A,
                         ck_tile::index_t stride_B,
                         ck_tile::index_t stride_C)
{
    if(verify == 1)
    {
        c_m_n_host_result.SetZero();

        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_m_k, b_k_n, c_m_n_host_result);
    }
    else if(verify == 2)
    {
        if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
        {
            // Restore input for B for gpu reference
            b_k_n_dev_buf.ToDevice(b_k_n.data());
        }

        ck_tile::DeviceMem c_m_n_gpu_buf_ref(c_m_n_host_result.get_element_space_size_in_bytes());
        c_m_n_host_result.SetZero();
        c_m_n_gpu_buf_ref.SetZero();

        ADataType* d_A = static_cast<ADataType*>(a_m_k_dev_buf.GetDeviceBuffer());
        BDataType* d_B = static_cast<BDataType*>(b_k_n_dev_buf.GetDeviceBuffer());
        CDataType* d_C = static_cast<CDataType*>(c_m_n_gpu_buf_ref.GetDeviceBuffer());

        ck_tile::reference_gemm_gpu<ADataType,
                                    BDataType,
                                    AccDataType,
                                    CDataType,
                                    ALayout,
                                    BLayout,
                                    CLayout>(d_A, d_B, d_C, M, N, K, stride_A, stride_B, stride_C);

        c_m_n_gpu_buf_ref.FromDevice(c_m_n_host_result.data());
    }
}
