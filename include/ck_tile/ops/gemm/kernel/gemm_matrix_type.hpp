// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck_tile {
    enum struct MatrixALayout {
        MK,  // Row-major layout for matrix A (default)
        KM   // Column-major layout for matrix A
    };

    enum struct MatrixBLayout {
        NK,  // Row-major layout for matrix B (default)
        KN   // Column-major layout for matrix B
    };

    enum struct MatrixCLayout {
        MN,  // Row-major layout for matrix C (default)
        NM   // Column-major layout for matrix C
    };

    // Function to convert string to MatrixALayout
    inline MatrixALayout parse_layout_a(const std::string& layout) {
        if (layout == "KM") return MatrixALayout::KM;
        return MatrixALayout::MK;  // Default to MK if not specified as KM
    }

    // Function to convert string to MatrixBLayout
    inline MatrixBLayout parse_layout_b(const std::string& layout) {
        if (layout == "KN") return MatrixBLayout::KN;
        return MatrixBLayout::NK;  // Default to NK if not specified as KN
    }

    // Function to convert string to MatrixBLayout
    inline MatrixCLayout parse_layout_c(const std::string& layout) {
        if (layout == "NM") return MatrixCLayout::NM;
        return MatrixCLayout::MN;  // Default to MN if not specified as NM
    }
}  // namespace ck_tile
