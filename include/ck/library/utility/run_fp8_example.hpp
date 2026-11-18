// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/host_utility/device_prop.hpp"

namespace ck {
namespace utils {

// Select the complete example, including host input generation and reference
// computation, before allocating or interpreting any FP8 tensors.
template <template <typename, typename> class Example>
int run_fp8_example(int argc, char* argv[])
{
#if CK_USE_OCP_FP8 && CK_USE_FNUZ_FP8 && defined(__gfx942__)
    // Emit only the matching native kernels in each device compilation pass.
    // The host pass below instantiates both entry points for runtime dispatch.
    return Example<f8_fnuz_t, bf8_fnuz_t>{}.Run(argc, argv);
#else
#if CK_USE_OCP_FP8 && CK_USE_FNUZ_FP8 && !defined(__HIP_DEVICE_COMPILE__)
    if(get_device_name() == "gfx942")
        return Example<f8_fnuz_t, bf8_fnuz_t>{}.Run(argc, argv);
#endif
#if CK_USE_OCP_FP8
    return Example<f8_ocp_t, bf8_ocp_t>{}.Run(argc, argv);
#else
    return Example<f8_fnuz_t, bf8_fnuz_t>{}.Run(argc, argv);
#endif
#endif
}

} // namespace utils
} // namespace ck
