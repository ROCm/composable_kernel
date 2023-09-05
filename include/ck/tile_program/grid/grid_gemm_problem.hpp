// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
namespace tile_program {
namespace grid {

template <typename ADataType_,
          typename BDataType_,
          typename AccDataType_,
          typename CDataType_,
          typename AElementFunction_,
          typename BElementFunction_,
          typename CElementFunction_>
struct GridGemmProblem
{
    using ADataType   = ADataType_;
    using BDataType   = BDataType_;
    using AccDataType = AccDataType_;
    using CDataType   = CDataType_;

    using AElementFunction = AElementFunction_;
    using BElementFunction = BElementFunction_;
    using CElementFunction = CElementFunction_;
};

} // namespace grid
} // namespace tile_program
} // namespace ck
