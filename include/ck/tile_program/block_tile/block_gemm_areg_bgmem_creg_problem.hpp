// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/type.hpp"

namespace ck {
namespace tile_program {
namespace block {

// Problem Description for BlockGemmARegBGmemCReg
template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          index_t kBlockSize_,
          typename BlockGemmShape_>
struct BlockGemmARegBGmemCRegProblem
{
    using ADataType      = remove_cvref_t<ADataType_>;
    using BDataType      = remove_cvref_t<BDataType_>;
    using CDataType      = remove_cvref_t<CDataType_>;
    using BlockGemmShape = remove_cvref_t<BlockGemmShape_>;

    static constexpr index_t kBlockSize = kBlockSize_;
};

} // namespace block
} // namespace tile_program
} // namespace ck
