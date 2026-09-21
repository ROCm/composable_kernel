// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Forward declarations of registration functions for generated convolution kernels.
// The implementations are auto-generated and compiled in their respective OBJECT libraries.

#pragma once

#include <string>

namespace ck_tile {
namespace dispatcher {

class GroupedConvRegistry;

void register_all_grouped_conv_bwd_weight_kernels(GroupedConvRegistry& registry,
                                                  const std::string& arch);

void register_all_grouped_conv_bwd_weight_kernels(const std::string& arch);

void register_all_grouped_conv_fwd_kernels(GroupedConvRegistry& registry, const std::string& arch);

void register_all_grouped_conv_fwd_kernels(const std::string& arch);

void register_all_grouped_conv_bwd_data_kernels(GroupedConvRegistry& registry,
                                                const std::string& arch);

void register_all_grouped_conv_bwd_data_kernels(const std::string& arch);

} // namespace dispatcher
} // namespace ck_tile
