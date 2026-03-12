// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/dispatcher/problem.hpp"
#include <hip/hip_runtime.h>
#include <cmath>
#include <vector>

namespace ck_tile {
namespace dispatcher {
namespace validation {

/// Reference CPU GEMM implementation for validation
template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType>
void reference_gemm_cpu(const ADataType* a,
                        const BDataType* b,
                        CDataType* c,
                        int M,
                        int N,
                        int K,
                        int stride_a,
                        int stride_b,
                        int stride_c,
                        bool transpose_a = false,
                        bool transpose_b = false)
{
    for(int m = 0; m < M; ++m)
    {
        for(int n = 0; n < N; ++n)
        {
            AccDataType acc = 0;

            for(int k = 0; k < K; ++k)
            {
                // Get A element
                int a_idx         = transpose_a ? (k * stride_a + m) : (m * stride_a + k);
                AccDataType a_val = static_cast<AccDataType>(a[a_idx]);

                // Get B element
                int b_idx         = transpose_b ? (n * stride_b + k) : (k * stride_b + n);
                AccDataType b_val = static_cast<AccDataType>(b[b_idx]);

                acc += a_val * b_val;
            }

            // Write C element
            int c_idx = m * stride_c + n;
            c[c_idx]  = static_cast<CDataType>(acc);
        }
    }
}

/// Validate kernel output against reference
template <typename CDataType>
bool validate_output(const CDataType* result,
                     const CDataType* reference,
                     int size,
                     float rtol = 1e-3f,
                     float atol = 1e-5f)
{
    int errors                    = 0;
    const int max_errors_to_print = 10;

    for(int i = 0; i < size; ++i)
    {
        float res_val = static_cast<float>(result[i]);
        float ref_val = static_cast<float>(reference[i]);

        float abs_diff = std::abs(res_val - ref_val);
        float abs_ref  = std::abs(ref_val);

        bool is_valid = (abs_diff <= atol) || (abs_diff <= rtol * abs_ref);

        if(!is_valid)
        {
            if(errors < max_errors_to_print)
            {
                printf("Mismatch at index %d: result=%.6f, reference=%.6f, diff=%.6e\n",
                       i,
                       res_val,
                       ref_val,
                       abs_diff);
            }
            errors++;
        }
    }

    if(errors > 0)
    {
        printf("Validation failed: %d/%d elements mismatched (%.2f%%)\n",
               errors,
               size,
               100.0f * errors / size);
        return false;
    }

    return true;
}

/// Validate kernel with reference implementation
template <typename ADataType, typename BDataType, typename CDataType, typename AccDataType>
bool validate_gemm_kernel(const void* a_dev_ptr,
                          const void* b_dev_ptr,
                          const void* c_dev_ptr,
                          const Problem& problem,
                          float rtol = 1e-3f,
                          float atol = 1e-5f)
{
    const int M = problem.M;
    const int N = problem.N;
    const int K = problem.K;

    // Allocate host memory
    std::vector<ADataType> a_host(M * K);
    std::vector<BDataType> b_host(K * N);
    std::vector<CDataType> c_host(M * N);
    std::vector<CDataType> c_ref(M * N);

    // Copy from device
    hipMemcpy(a_host.data(), a_dev_ptr, M * K * sizeof(ADataType), hipMemcpyDeviceToHost);
    hipMemcpy(b_host.data(), b_dev_ptr, K * N * sizeof(BDataType), hipMemcpyDeviceToHost);
    hipMemcpy(c_host.data(), c_dev_ptr, M * N * sizeof(CDataType), hipMemcpyDeviceToHost);

    // Compute reference
    reference_gemm_cpu<ADataType, BDataType, CDataType, AccDataType>(a_host.data(),
                                                                     b_host.data(),
                                                                     c_ref.data(),
                                                                     M,
                                                                     N,
                                                                     K,
                                                                     K, // stride_a (row-major)
                                                                     N, // stride_b (row-major)
                                                                     N, // stride_c (row-major)
                                                                     false,
                                                                     false);

    // Validate
    return validate_output(c_host.data(), c_ref.data(), M * N, rtol, atol);
}

/// Validator class for kernel instances
class KernelValidator
{
    public:
    KernelValidator(float rtol = 1e-3f, float atol = 1e-5f) : rtol_(rtol), atol_(atol) {}

    /// Validate a kernel instance
    template <typename KernelInstance>
    bool validate(KernelInstance& kernel,
                  const void* a_ptr,
                  const void* b_ptr,
                  const void* c_ptr,
                  const Problem& problem)
    {
        // Use kernel's validate method if available
        return kernel.validate(a_ptr, b_ptr, c_ptr, problem, rtol_, atol_);
    }

    /// Set tolerances
    void set_tolerances(float rtol, float atol)
    {
        rtol_ = rtol;
        atol_ = atol;
    }

    /// Get tolerances
    std::pair<float, float> get_tolerances() const { return {rtol_, atol_}; }

    private:
    float rtol_;
    float atol_;
};

/// Helper to generate random test data
template <typename T>
void generate_random_data(T* data, int size, float min_val = -1.0f, float max_val = 1.0f)
{
    for(int i = 0; i < size; ++i)
    {
        float rand_val = min_val + (max_val - min_val) * (rand() / (float)RAND_MAX);
        data[i]        = static_cast<T>(rand_val);
    }
}

/// Helper to allocate and initialize test tensors
template <typename T>
struct TestTensor
{
    T* host_ptr;
    T* device_ptr;
    int size;

    TestTensor(int size_) : size(size_)
    {
        host_ptr = new T[size];
        hipMalloc(&device_ptr, size * sizeof(T));
    }

    ~TestTensor()
    {
        delete[] host_ptr;
        hipFree(device_ptr);
    }

    void randomize(float min_val = -1.0f, float max_val = 1.0f)
    {
        generate_random_data(host_ptr, size, min_val, max_val);
        hipMemcpy(device_ptr, host_ptr, size * sizeof(T), hipMemcpyHostToDevice);
    }

    void copy_to_device()
    {
        hipMemcpy(device_ptr, host_ptr, size * sizeof(T), hipMemcpyHostToDevice);
    }

    void copy_from_device()
    {
        hipMemcpy(host_ptr, device_ptr, size * sizeof(T), hipMemcpyDeviceToHost);
    }

    void zero() { hipMemset(device_ptr, 0, size * sizeof(T)); }
};

} // namespace validation
} // namespace dispatcher
} // namespace ck_tile
