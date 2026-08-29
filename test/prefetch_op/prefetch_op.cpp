// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"

#include "prefetch_op_util.hpp"

template <typename T, uint32_t NUM_THREADS, uint32_t NUM_SCALARS, bool IS_L1_PREFETCH>
bool run_test(bool time_kernels)
{
    bool pass = true;

#if defined(__gfx125__)
    const auto coherence =
        IS_L1_PREFETCH ? ck::AmdBufferCoherenceEnum::CU_RT : ck::AmdBufferCoherenceEnum::SE_RT;
    using global_prefetch_op = ck::GlobalPrefetchDataOp<coherence>;
    using flat_prefetch_op   = ck::FlatPrefetchDataOp<coherence>;
#else
    using global_prefetch_op = ck::GlobalPrefetchDataOp<>;
    using flat_prefetch_op   = ck::FlatPrefetchDataOp<>;
#endif

    const auto global_prefetch_kernel =
        ck::prefetch_op_util::kernel_with_prefetch<T, NUM_THREADS, NUM_SCALARS, global_prefetch_op>;
    const auto flat_prefetch_kernel = ck::prefetch_op_util::
        kernel_with_prefetch_and_shared_mem<T, NUM_THREADS, NUM_SCALARS, flat_prefetch_op>;

    const auto prefetch_kernel_container =
        std::make_tuple(global_prefetch_kernel, flat_prefetch_kernel);

    ck::static_for<0, 2, 1>{}([&](auto i) {
        std::string kernel_name = (i == 1 ? "flat_prefetch" : "global_prefetch");

        auto kernel = std::get<ck::Number<i>{}>(prefetch_kernel_container);

        pass &=
            ck::prefetch_op_util::test_prefetch_impl<decltype(kernel), T, NUM_THREADS, NUM_SCALARS>(
                time_kernels, kernel, kernel_name);
    });

    return pass;
}

int main(int argc, char* argv[])
{
    if(!ck::is_gfx125_supported())
    {
        std::cout << "This feature is not supported by current HW, skipping tests." << std::endl;
        return 0;
    }

    bool time_kernels = false;

    if(argc == 2)
    {
        time_kernels = std::stoi(argv[1]);
    }

    bool pass = true;

    std::cout << "=== Testing L2 Global Cache Prefetch ===" << std::endl;

    pass &= run_test<float, 4096, 1024, false>(time_kernels);
    pass &= run_test<double, 4096, 512, false>(time_kernels);

    std::cout << "=== Testing L1 Global Cache Prefetch ===" << std::endl;

    pass &= run_test<float, 4096, 1024, true>(time_kernels);
    pass &= run_test<double, 4096, 512, true>(time_kernels);

    std::cout << "TestGlobalPrefetch ..... " << (pass ? "SUCCESS" : "FAILURE") << std::endl;
    return pass ? 0 : 1;
}
