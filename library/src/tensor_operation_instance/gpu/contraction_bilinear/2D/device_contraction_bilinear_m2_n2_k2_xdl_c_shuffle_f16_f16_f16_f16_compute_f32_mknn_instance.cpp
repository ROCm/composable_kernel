// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "../../contraction/contraction_instance_common.hpp"

// Instantiate contraction device operation and register via add_device_* function.
// See contraction_instance_common.hpp for macro definition and parameter documentation.
// clang-format off
CK_CONTRACTION_INSTANCE(device_contraction_mk_instance,
    bilinear, Bilinear, 2, f16_f16_f16_f16_compute_f32_mknn,
    F16, F16, F32, F16, F16_Tuple, F16, F32)
// clang-format on
