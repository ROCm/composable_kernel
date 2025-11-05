// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>
#include <chrono>

#include "ck/ck.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/host_utility/device_prop.hpp"

#include <hip/hip_runtime.h>

#if __clang_major__ >= 20
#include "ck/utility/amd_buffer_addressing_builtins.hpp"
#else
#include "ck/utility/amd_buffer_addressing.hpp"
#endif

#include "s_prefetch_op_util.hpp"

template <typename T>
bool run_test()
{
    bool pass = true;

    const auto s_prefetch_kernel = ck::s_prefetch_op_util::kernel_with_scalar_prefetch<T>;
    const auto s_buffer_prefetch_kernel =
        ck::s_prefetch_op_util::kernel_with_scalar_buffer_prefetch<T>;

    const auto prefetch_kernel_container =
        std::make_tuple(s_prefetch_kernel, s_buffer_prefetch_kernel);

    ck::static_for<0, 2, 1>{}([&](auto i) {
        std::string kernel_name = (i == 1 ? "s_buffer_prefetch" : "s_prefetch");
        pass &= ck::s_prefetch_op_util::test_constant_prefetch_impl<
            decltype(std::get<ck::Number<i>{}>(prefetch_kernel_container)),
            T>(std::get<ck::Number<i>{}>(prefetch_kernel_container), kernel_name);
    });

    return pass;
}

int main(int, char*[])
{
    if(!ck::is_gfx12_supported())
    {
        std::cout << "This feature is not supported by current HW, skipping tests." << std::endl;
        return 0;
    }

    bool pass = true;

    std::cout << "=== Testing Constant Cache Prefetch ===" << std::endl;

    // Test different data types
    pass &= run_test<float>();
    pass &= run_test<double>();

    std::cout << "TestConstantPrefetch ..... " << (pass ? "SUCCESS" : "FAILURE") << std::endl;
    return pass ? 0 : 1;
}
