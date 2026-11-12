// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "ck/host_utility/hip_check_error.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/stream_config.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_dl.hpp"

namespace {

using Row         = ck::tensor_layout::gemm::RowMajor;
using Col         = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

// Keep the tile and transfer parameters of example/01_gemm/gemm_dl_fp32.cpp.
template <typename A       = float,
          typename B       = float,
          typename C       = float,
          typename Acc     = float,
          typename ALayout = Col,
          typename BLayout = Row,
          typename CLayout = Row>
using Gemm = ck::tensor_operation::device::DeviceGemmDl<
    A,
    B,
    C,
    Acc,
    ALayout,
    BLayout,
    CLayout,
    PassThrough,
    PassThrough,
    PassThrough,
    ck::tensor_operation::device::GemmSpecialization::Default,
    256,
    128,
    128,
    16,
    1,
    4,
    4,
    1,
    S<8, 2>,
    S<8, 2>,
    S<2, 1, 4, 1>,
    S<8, 1, 32, 1>,
    S<0, 3, 1, 2>,
    S<0, 3, 1, 2>,
    S<1, 1, 4, 1>,
    S<0, 3, 1, 2>,
    S<1, 1, 4, 1>,
    S<2, 1, 4, 1>,
    S<8, 1, 32, 1>,
    S<0, 3, 1, 2>,
    S<0, 3, 1, 2>,
    S<1, 1, 4, 1>,
    S<0, 3, 1, 2>,
    S<1, 1, 4, 1>,
    S<0, 1, 2, 3, 4, 5>,
    5,
    4>;

struct Problem
{
    ck::index_t m;
    ck::index_t n;
    ck::index_t k;
    ck::index_t stride_a;
    ck::index_t stride_b;
    ck::index_t stride_c;
};

bool run_problem(const Problem& p)
{
    const auto admission = Gemm<>::MakeArgument(nullptr,
                                                nullptr,
                                                nullptr,
                                                p.m,
                                                p.n,
                                                p.k,
                                                p.stride_a,
                                                p.stride_b,
                                                p.stride_c,
                                                PassThrough{},
                                                PassThrough{},
                                                PassThrough{});
    if(!Gemm<>::IsSupportedArgument(admission))
    {
        std::cerr << "FAIL: valid problem rejected" << std::endl;
        return false;
    }

    constexpr float sentinel = 123456.75f;
    std::vector<float> a(p.k * p.stride_a, sentinel);
    std::vector<float> b(p.k * p.stride_b, sentinel);
    std::vector<float> c(p.m * p.stride_c, sentinel);
    for(ck::index_t m = 0; m < p.m; ++m)
    {
        for(ck::index_t k = 0; k < p.k; ++k)
        {
            a[m + k * p.stride_a] = ((5 * m + 3 * k) % 17 - 8) / 8.0f;
        }
        for(ck::index_t n = 0; n < p.n; ++n)
        {
            c[m * p.stride_c + n] = std::numeric_limits<float>::quiet_NaN();
        }
    }
    for(ck::index_t k = 0; k < p.k; ++k)
    {
        for(ck::index_t n = 0; n < p.n; ++n)
        {
            b[k * p.stride_b + n] = ((7 * k + 11 * n) % 19 - 9) / 8.0f;
        }
    }

    ck::DeviceMem device_a(a.size() * sizeof(float));
    ck::DeviceMem device_b(b.size() * sizeof(float));
    ck::DeviceMem device_c(c.size() * sizeof(float));
    device_a.ToDevice(a.data());
    device_b.ToDevice(b.data());
    device_c.ToDevice(c.data());
    auto argument = Gemm<>::MakeArgument(static_cast<const float*>(device_a.GetDeviceBuffer()),
                                         static_cast<const float*>(device_b.GetDeviceBuffer()),
                                         static_cast<float*>(device_c.GetDeviceBuffer()),
                                         p.m,
                                         p.n,
                                         p.k,
                                         p.stride_a,
                                         p.stride_b,
                                         p.stride_c,
                                         PassThrough{},
                                         PassThrough{},
                                         PassThrough{});
    Gemm<>::MakeInvoker().Run(argument, StreamConfig{nullptr, false});
    HIP_CHECK_ERROR(hipGetLastError());
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    device_c.FromDevice(c.data());

    // Products are multiples of 1/64; with K <= 64 every partial sum is exact in FP32.
    // Exact comparison also catches a no-op kernel because active C starts as NaN.
    for(ck::index_t m = 0; m < p.m; ++m)
    {
        for(ck::index_t n = 0; n < p.stride_c; ++n)
        {
            float expected = sentinel;
            if(n < p.n)
            {
                expected = 0;
                for(ck::index_t k = 0; k < p.k; ++k)
                {
                    expected += a[m + k * p.stride_a] * b[k * p.stride_b + n];
                }
            }
            const float actual = c[m * p.stride_c + n];
            if(!std::isfinite(actual) || actual != expected)
            {
                std::cerr << "FAIL: C[" << m << ", " << n << "] = " << actual << ", expected "
                          << expected << std::endl;
                return false;
            }
        }
    }
    std::cout << "PASS" << std::endl;
    return true;
}

template <typename Operation = Gemm<>>
bool expect_rejection(const char* name,
                      ck::index_t m = 128,
                      ck::index_t n = 128,
                      ck::index_t k = 64)
{
    // Admission checks do not launch kernels or dereference the data pointers.
    auto argument       = Operation::MakeArgument(nullptr,
                                            nullptr,
                                            nullptr,
                                            m,
                                            n,
                                            k,
                                            256,
                                            256,
                                            256,
                                            PassThrough{},
                                            PassThrough{},
                                            PassThrough{});
    const bool rejected = !Operation::IsSupportedArgument(argument);
    std::cout << name << ": " << (rejected ? "PASS (rejected)" : "FAIL (accepted)") << std::endl;
    return rejected;
}

} // namespace

int main()
{
    try
    {
        int device = 0;
        hipDeviceProp_t properties{};
        HIP_CHECK_ERROR(hipGetDevice(&device));
        HIP_CHECK_ERROR(hipGetDeviceProperties(&properties, device));
        int runtime_version = 0;
        int driver_version  = 0;
        HIP_CHECK_ERROR(hipRuntimeGetVersion(&runtime_version));
        HIP_CHECK_ERROR(hipDriverGetVersion(&driver_version));
        const auto arch = ck::get_device_name();
        std::cout << "Device " << device << ": " << properties.name << ", native architecture "
                  << properties.gcnArchName << ", HIP runtime version " << runtime_version
                  << ", HIP driver version " << driver_version << std::endl;
        if(arch != "gfx90c" && arch != "gfx906" && !ck::is_gfx103_supported() &&
           !ck::is_gfx11_supported() && !ck::is_gfx117_supported() && !ck::is_gfx12_supported())
        {
            std::cout << "SKIP: native device does not support this DL GEMM test" << std::endl;
            return 77;
        }

        // K=16/32/48/64 covers every main-loop/single-or-double-tail combination.
        const Problem problems[] = {{128, 128, 16, 128, 128, 128},
                                    {128, 128, 32, 128, 128, 128},
                                    {128, 128, 48, 128, 128, 128},
                                    {128, 128, 64, 128, 128, 128},
                                    {256, 128, 64, 256, 128, 128},
                                    {128, 256, 64, 128, 256, 256},
                                    {128, 128, 64, 132, 132, 132}};
        bool pass                = true;
        for(int repeat = 0; repeat < 10; ++repeat)
        {
            for(const auto& problem : problems)
            {
                std::cout << "Run " << repeat + 1 << "/10 M=" << problem.m << " N=" << problem.n
                          << " K=" << problem.k << " strides=" << problem.stride_a << ","
                          << problem.stride_b << "," << problem.stride_c << ": " << std::flush;
                pass = run_problem(problem) && pass;
            }
        }

        pass = expect_rejection("M vector alignment", 130) && pass;
        pass = expect_rejection("N vector alignment", 128, 130) && pass;
        pass = expect_rejection("M block alignment", 132) && pass;
        pass = expect_rejection("N block alignment", 128, 132) && pass;
        pass = expect_rejection("K block alignment", 128, 128, 17) && pass;
        if(arch == "gfx90c")
        {
            pass = expect_rejection<Gemm<float, float, float, float, Row>>("Row-major A") && pass;
            pass = expect_rejection<Gemm<float, float, float, float, Col, Col>>("Column-major B") &&
                   pass;
            pass = expect_rejection<Gemm<float, float, float, float, Col, Row, Col>>(
                       "Column-major C") &&
                   pass;
            pass = expect_rejection<Gemm<ck::half_t>>("FP16 A") && pass;
            pass = expect_rejection<Gemm<float, ck::half_t>>("FP16 B") && pass;
            pass = expect_rejection<Gemm<float, float, ck::half_t>>("FP16 C") && pass;
            pass =
                expect_rejection<Gemm<float, float, float, ck::half_t>>("FP16 accumulator") && pass;
        }
        return pass ? 0 : 1;
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAIL: " << error.what() << std::endl;
        return 1;
    }
}
