// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"

namespace ck {
namespace wrapper {

/**
 * \brief Traits for blockwise gemm xdl.
 *
 * \tparam MPerXDL M per Xdl.
 * \tparam NPerXDL N per Xdl.
 * \tparam MXdlPerWave M Xdl Per Wave
 * \tparam NXdlPerWave N Xdl Per Wave
 * \tparam K1 K1
 */
template <index_t MPerXDL, index_t NPerXDL, index_t MXdlPerWave, index_t NXdlPerWave, index_t K1>
struct BlockwisGemmXdlTraits
{
    static constexpr index_t MPerXDL_     = MPerXDL;
    static constexpr index_t NPerXDL_     = NPerXDL;
    static constexpr index_t MXdlPerWave_ = MXdlPerWave;
    static constexpr index_t NXdlPerWave_ = NXdlPerWave;
    static constexpr index_t K1_          = K1;
};

struct BlockwisGemmXdlTraits_32x32PerXdl_4x2XdlPerWave_4K1 : BlockwisGemmXdlTraits<32, 32, 4, 2, 4>
{
};
struct BlockwisGemmXdlTraits_32x32PerXdl_2x4XdlPerWave_4K1 : BlockwisGemmXdlTraits<32, 32, 2, 4, 4>
{
};
struct BlockwisGemmXdlTraits_32x32PerXdl_2x2XdlPerWave_4K1 : BlockwisGemmXdlTraits<32, 32, 2, 2, 4>
{
};

} // namespace wrapper
} // namespace ck
