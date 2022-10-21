// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/utility/amd_wmma.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

namespace ck {
__global__ void matmul(const half_t* a, const half_t* b, float* c)
{
    const int lIdx = threadIdx.x;

    // a and b fragments are stored in 8 VGPRs each, in packed format, so 16 elements each for a and
    // b a_frag will store one column of the 16x16 matrix tile b_frag will store one row of the
    // 16x16 matrix tile
    half16_t a_frag = {};
    half16_t b_frag = {};
    // initialize c fragment to 0
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, float, 1, 8, true> c_thread_buf_;

    // lane is (0-31) mod 16 instead of 0-31 due to matrix replication in gfx11
    // see https://atlvsp3.amd.com/sp3_gfx11_5_instructions.pdf page 482
    // TODO: remove this dependency in gfx12 https://ontrack-internal.amd.com/browse/DEGFXSP3-101
    const int lane = lIdx % 16;

    for(int ele = 0; ele < 16; ++ele)
    {
        b_frag[ele] = b[16 * lane + ele];
    }
    // follow origin design
    for(int ele = 0; ele < 16; ++ele)
    {
        a_frag[ele] = a[16 * lane + ele];
    }

    // sync threads, similar to mma_sync
    __syncthreads();
    intrin_wmma_f32_16x16x16_f16_w32<16, 16>::Run(
        a_frag, b_frag, c_thread_buf_.GetVectorTypeReference(Number<0>{}));
    __syncthreads();
    // wait for results, similar to mma_sync
    static_for<0, 8, 1>{}([&](auto ele) {
        const int r = ele * 2 + (lIdx / 16);
        // store results from unpacked c_thread_buf_ output
        c[16 * r + lane] = c_thread_buf_[Number<ele>{}];
    });
}

__global__ void matmul_swizzle_a(const half_t* a, const half_t* b, float* c)
{
    const int lIdx = threadIdx.x;

    half16_t a_frag = {};
    half16_t b_frag = {};
    StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr, float, 1, 8, true> c_thread_buf_;

    const int lane = lIdx % 16;

    for(int ele = 0; ele < 16; ++ele)
    {
        b_frag[ele] = b[16 * lane + ele];
    }

    const int offset_m = (((lane & 1) << 3) | (lane >> 1));
    for(int ele = 0; ele < 16; ++ele)
    {
        a_frag[ele] = a[16 * offset_m + ele];
    }

    __syncthreads();
    intrin_wmma_f32_16x16x16_f16_w32<16, 16>::Run(
        a_frag, b_frag, c_thread_buf_.GetVectorTypeReference(Number<0>{}));
    __syncthreads();

    static_for<0, 8, 1>{}([&](auto ele) {
        const int blk                   = lIdx / 16;
        const int r                     = ele;
        c[16 * 8 * blk + 16 * r + lane] = c_thread_buf_[Number<ele>{}];
    });
}
} // namespace ck

int main(int, char*[])
{
    std::vector<float> host_a(16 * 16);
    std::vector<float> host_b(16 * 16);
    std::vector<float> host_c(16 * 16);
    std::vector<float> wmma_c(16 * 16);
    std::vector<float> wmma_c_swizzle_a(16 * 16);
    uint64_t num_element = 256;

    // generate matrix a
    for(int i_m = 0; i_m < 16; i_m++)
    {
        for(int i_k = 0; i_k < 16; i_k++)
        {
            host_a[i_m * 16 + i_k] = float(i_m + 1) / 99.0 + (float(i_k + 1) / 100);
            // host_a[i_m * 16 + i_k] = float(i_k);
        }
    }

    // generate matrix b
    for(int i_n = 0; i_n < 16; i_n++)
    {
        for(int i_k = 0; i_k < 16; i_k++)
        {
            host_b[i_n * 16 + i_k] = float(i_n + 1) / 98.0 + (float(i_k + 1) / 100);
            // host_b[i_n * 16 + i_k] = 1.0;
        }
    }

    // run mk_nk_mn gemm on cpu
    for(int i_m = 0; i_m < 16; i_m++)
    {
        for(int i_n = 0; i_n < 16; i_n++)
        {
            for(int i_k = 0; i_k < 16; i_k++)
            {
                host_c[i_m * 16 + i_n] += host_a[i_m * 16 + i_k] * host_b[i_n * 16 + i_k];
            }
        }
    }

    DeviceMem device_a(sizeof(ck::half_t) * num_element);
    DeviceMem device_b(sizeof(ck::half_t) * num_element);
    DeviceMem device_c(sizeof(float) * num_element);

    std::vector<ck::half_t> fp16_a(16 * 16);
    std::vector<ck::half_t> fp16_b(16 * 16);
    // convert fp32 a and b into fp16 on host
    for(int i = 0; i < 16 * 16; i++)
    {
        fp16_a[i] = __float2half_rn(host_a[i]);
        fp16_b[i] = __float2half_rn(host_b[i]);
    }

    device_a.ToDevice(fp16_a.data());
    device_b.ToDevice(fp16_b.data());

    // run single wave wmma on GPU
    ck::matmul<<<1, 32>>>(static_cast<const ck::half_t*>(device_a.GetDeviceBuffer()),
                          static_cast<const ck::half_t*>(device_b.GetDeviceBuffer()),
                          static_cast<float*>(device_c.GetDeviceBuffer()));

    device_c.FromDevice(wmma_c.data());

    bool res = ck::utils::check_err(wmma_c, host_c, "Error: Incorrect results!", 1e-2);

    // run single wave wmma_swizzle_a on GPU
    ck::matmul_swizzle_a<<<1, 32>>>(static_cast<const ck::half_t*>(device_a.GetDeviceBuffer()),
                                    static_cast<const ck::half_t*>(device_b.GetDeviceBuffer()),
                                    static_cast<float*>(device_c.GetDeviceBuffer()));
    device_c.FromDevice(wmma_c_swizzle_a.data());

    bool res_swizzle_a =
        ck::utils::check_err(wmma_c_swizzle_a, host_c, "Error: Incorrect results!", 1e-2);

    if(res && res_swizzle_a)
    {
        std::cout << "test single wave wmma: Pass" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "test single wave wmma: Fail" << std::endl;
        return -1;
    }
}
