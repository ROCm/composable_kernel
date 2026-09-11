// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "../../contraction/contraction_instance_common.hpp"

// Instantiate contraction device operation and register via add_device_* function.
// See contraction_instance_common.hpp for macro definition and parameter documentation.
// clang-format off
CK_CONTRACTION_INSTANCE(device_contraction_kn_instance,
    bilinear, Bilinear, 6, f32_f32_f32_f32_compute_f16_knnn,
    F32, F32, F32, F32, F32_Tuple, F32, F16)
// clang-format on
