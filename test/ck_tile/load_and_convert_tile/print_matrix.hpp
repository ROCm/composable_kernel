// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/host.hpp"

namespace ck_tile {

// Helper to print matrix (for debugging)
template <typename T>
void print_matrix(const ck_tile::HostTensor<T>& mat,
                  const std::string& name = "Matrix",
                  const int width         = 3,
                  const int precision     = 3)
{
    const auto lens              = mat.get_lengths();
    const ck_tile::index_t rows  = lens[0];
    const ck_tile::index_t cols  = lens[1];
    const ck_tile::index_t limit = 32;

    std::cout << name << " (" << rows << "x" << cols << "):\n";
    for(ck_tile::index_t i = 0; i < std::min(rows, ck_tile::index_t(limit)); ++i)
    {
        for(ck_tile::index_t j = 0; j < std::min(cols, ck_tile::index_t(limit)); ++j)
        {
            std::cout << std::setw(width) << std::setprecision(precision)
                      << ck_tile::type_convert<float>(mat(i, j)) << " ";
        }
        if(cols > limit)
            std::cout << "...";
        std::cout << "\n";
    }
    if(rows > limit)
        std::cout << "...\n";
    std::cout << "\n";
}

} // namespace ck_tile
