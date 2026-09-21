// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "../../contraction/contraction_instance_common.hpp"

// Instantiate contraction device operation and register via add_device_* function.
// See contraction_instance_common.hpp for macro definition and parameter documentation.
// clang-format off
CK_CONTRACTION_INSTANCE(device_contraction_mn_instance,
    scale, Scale, 2, f16_f16_f16_mnn,
    F16, F16, F32, F16, Empty_Tuple, F16, F16)
// clang-format on
