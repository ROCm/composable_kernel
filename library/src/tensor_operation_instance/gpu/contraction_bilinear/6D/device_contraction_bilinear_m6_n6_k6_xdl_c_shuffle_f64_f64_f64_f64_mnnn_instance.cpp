// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "../../contraction/contraction_instance_common.hpp"

// Instantiate contraction device operation and register via add_device_* function.
// See contraction_instance_common.hpp for macro definition and parameter documentation.
// clang-format off
CK_CONTRACTION_INSTANCE(device_contraction_f64_mn_instance,
    bilinear, Bilinear, 6, f64_f64_f64_f64_mnnn,
    F64, F64, F64, F64, F64_Tuple, F64, F64)
// clang-format on
