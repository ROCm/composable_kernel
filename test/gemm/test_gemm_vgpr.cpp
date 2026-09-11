// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * \brief Test that verifies availability of >256 VGPRs by running a kernel that uses large number
 * of VGPRs (~800) to perform a simple matrix multiplication operation.
 *
 * The test runs the kernel with different matrix sizes (8x8 and 16x16 per thread). The kernel
 * stores the input matrices in VGPRs, performs matrix multiplication, and writes the result back to
 * global memory. The host code verifies the correctness of the result by comparing it with a
 * reference implementation.
 *
 * \note This example must be compiled with the following flag to see the resource allocations:
 *  "-Rpass-analysis=kernel-resource-usage"
 *
 * On gfx1200 with ROCm 6.4.1, the kernel will show register spilling due to limited VGPRs:
 * \verbatim
 * 8x8 matrix per thread:
 * SGPRs: 8
 * VGPRs: 105
 * ScratchSize [bytes/lane]: 0
 * Dynamic Stack: False
 * Occupancy [waves/SIMD]: 12
 * SGPRs Spill: 0
 * VGPRs Spill: 0
 * LDS Size [bytes/block]: 0
 *
 * 16x16 matrix per thread:
 * SGPRs: 36
 * VGPRs: 256
 * ScratchSize [bytes/lane]: 54144
 * Dynamic Stack: False
 * Occupancy [waves/SIMD]: 5
 * SGPRs Spill: 0
 * VGPRs Spill: 3771
 * LDS Size [bytes/block]: 0
 * \endverbatim
 *
 * On gfx1250, the test will not show register spilling due to increased VGPRs:
 * \verbatim
 * 8x8 matrix per thread:
 * TotalSGPRs: 8
 * VGPRs: 135
 * ScratchSize [bytes/lane]: 0
 * Dynamic Stack: False
 * Occupancy [waves/SIMD]: 7
 * SGPRs Spill: 0
 * VGPRs Spill: 0
 * LDS Size [bytes/block]: 0
 *
 * 16x16 matrix per thread:
 * TotalSGPRs: 44
 * VGPRs: 787
 * ScratchSize [bytes/lane]: 50304
 * Dynamic Stack: False
 * Occupancy [waves/SIMD]: 1
 * SGPRs Spill: 0
 * VGPRs Spill: 0
 * LDS Size [bytes/block]: 0
 * \endverbatim
 *
 * \note The register allocations above can be influenced by compiler version and code
 * changes/optimizations.
 */

#include "gtest/gtest.h"

#include "ck/host_utility/hip_check_error.hpp"
#include "ck/utility/common_header.hpp"

using namespace std;

template <int MatSize>
__global__ void __launch_bounds__(64, 1) test_largevgpr(const float* a, const float* b, float* c)
{

    // store data in VGPRs
    typedef float mat_t __attribute__((ext_vector_type(MatSize * MatSize)));

    int num = hipThreadIdx_x;
    mat_t mata;
    mat_t matb;
    mat_t matc(0.0f);

    const float* p_a = a + num * MatSize * MatSize;
    const float* p_b = b + num * MatSize * MatSize;
    float* p_c       = c + num * MatSize * MatSize;

    for(uint32_t i = 0; i < MatSize; i++)
    {
        for(uint32_t j = 0; j < MatSize; j++)
        {
            mata[i * MatSize + j] = *(p_a + i * MatSize + j);
            matb[i * MatSize + j] = *(p_b + i * MatSize + j);
        }
    }

    for(uint32_t i = 0; i < MatSize; i++)
    {
        for(uint32_t j = 0; j < MatSize; j++)
        {
            for(uint32_t k = 0; k < MatSize; k++)
            {
                matc[i * MatSize + j] += mata[i * MatSize + k] * matb[k * MatSize + j];
            }
        }
    }
    for(uint32_t i = 0; i < MatSize; i++)
    {
        for(uint32_t j = 0; j < MatSize; j++)
        {
            *(p_c + i * MatSize + j) = matc[i * MatSize + j];
        }
    }
}

template <int MatSize>
void verify_largevgpr()
{
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> c;
    std::vector<float> ref;
    a.resize(MatSize * MatSize * 32);
    b.resize(MatSize * MatSize * 32);
    c.resize(MatSize * MatSize * 32);
    ref.resize(MatSize * MatSize * 32);

    constexpr int max_value = 7;
    constexpr int min_value = -7;
    for(size_t i = 0; i < MatSize * MatSize * 32; i++)
    {
        a[i]   = static_cast<float>((std::rand() % (max_value - min_value)) + min_value);
        b[i]   = static_cast<float>((std::rand() % (max_value - min_value)) + min_value);
        ref[i] = 0;
    }

    for(uint32_t t = 0; t < 32; t++)
    {
        const float* p_a = a.data() + t * MatSize * MatSize;
        const float* p_b = b.data() + t * MatSize * MatSize;
        float* p_ref     = ref.data() + t * MatSize * MatSize;
        for(uint32_t i = 0; i < MatSize; i++)
        {
            for(uint32_t j = 0; j < MatSize; j++)
            {
                for(uint32_t k = 0; k < MatSize; k++)
                {
                    *(p_ref + i * MatSize + j) +=
                        *(p_a + i * MatSize + k) * *(p_b + k * MatSize + j);
                }
            }
        }
    }

    float* device_a;
    float* device_b;
    float* device_c;

    HIP_CHECK_ERROR(hipMalloc(reinterpret_cast<void**>(&device_a), a.size() * sizeof(float)));
    HIP_CHECK_ERROR(hipMalloc(reinterpret_cast<void**>(&device_b), b.size() * sizeof(float)));
    HIP_CHECK_ERROR(hipMalloc(reinterpret_cast<void**>(&device_c), c.size() * sizeof(float)));

    HIP_CHECK_ERROR(hipMemcpy(device_a, a.data(), a.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK_ERROR(hipMemcpy(device_b, b.data(), b.size() * sizeof(float), hipMemcpyHostToDevice));

    hipLaunchKernelGGL(
        test_largevgpr<MatSize>, dim3(1), dim3(32), 0, nullptr, device_a, device_b, device_c);

    HIP_CHECK_ERROR(hipMemcpy(c.data(), device_c, c.size() * sizeof(float), hipMemcpyDeviceToHost));

    bool pass = true;
    for(size_t i = 0; i < MatSize * MatSize * 32; i++)
    {
        if(fabs(c[i] - ref[i]) > 0.0001)
        {
            pass = false;
            std::cout << "mismatch on index " << i << ": " << c[i] << " != " << ref[i] << std::endl;
            break;
        }
    }

    HIP_CHECK_ERROR(hipFree(device_a));
    HIP_CHECK_ERROR(hipFree(device_b));
    HIP_CHECK_ERROR(hipFree(device_c));
    EXPECT_TRUE(pass);
}

TEST(GEMMVGPR, M8x8) { verify_largevgpr<8>(); }

TEST(GEMMVGPR, M16x16) { verify_largevgpr<16>(); }
