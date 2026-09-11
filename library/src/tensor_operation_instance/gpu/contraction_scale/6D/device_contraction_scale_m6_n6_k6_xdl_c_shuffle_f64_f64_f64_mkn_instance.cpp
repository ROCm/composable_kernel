// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "../../contraction/contraction_instance_common.hpp"

// Instantiate contraction device operation and register via add_device_* function.
// See contraction_instance_common.hpp for macro definition and parameter documentation.
// clang-format off
CK_CONTRACTION_INSTANCE(device_contraction_f64_mk_instance,
    scale, Scale, 6, f64_f64_f64_mkn,
    F64, F64, F64, F64, Empty_Tuple, F64, F64)
// clang-format on
