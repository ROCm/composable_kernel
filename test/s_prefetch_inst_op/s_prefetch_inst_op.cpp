// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"

#include "s_prefetch_inst_op_util.hpp"

template <typename T, uint32_t NUM_THREADS, uint32_t NUM_SCALARS>
bool run_test(bool time_kernels)
{
    return ck::s_prefetch_inst_op_util::test_inst_prefetch_impl<T, NUM_THREADS, NUM_SCALARS>(
        time_kernels, "s_prefetch_inst_pc_rel");
}

int main(int argc, char* argv[])
{
    if(!ck::is_gfx12_supported())
    {
        std::cout << "instruction cache prefetch is not supported by current HW, skipping tests."
                  << std::endl;
        return 0;
    }

    bool time_kernels = false;
    if(argc == 2)
    {
        time_kernels = std::stoi(argv[1]);
    }

    bool pass = true;

    std::cout << "=== Testing Instruction Prefetch ===" << std::endl;

    pass &= run_test<float, 4096, 16384>(time_kernels);

    std::cout << "TestInstPrefetch ..... " << (pass ? "SUCCESS" : "FAILURE") << std::endl;
    return pass ? 0 : 1;
}
