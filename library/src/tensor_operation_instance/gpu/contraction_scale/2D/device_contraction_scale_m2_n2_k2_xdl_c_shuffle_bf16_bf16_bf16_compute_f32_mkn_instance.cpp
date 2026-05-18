// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "../../contraction/contraction_instance_common.hpp"

// Instantiate contraction device operation and register via add_device_* function.
// See contraction_instance_common.hpp for macro definition and parameter documentation.
// clang-format off
CK_CONTRACTION_INSTANCE(device_contraction_mk_instance,
    scale, Scale, 2, bf16_bf16_bf16_compute_f32_mkn,
    BF16, BF16, F32, BF16, Empty_Tuple, BF16, F32)
// clang-format on
