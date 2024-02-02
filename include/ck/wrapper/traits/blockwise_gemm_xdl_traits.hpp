// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"

namespace ck {
namespace wrapper {

/**
 * \brief Traits for blockwise gemm xdl.
 *
 * \tparam MPerXDLValue The MFMA instruction size in M dimension.
 * \tparam NPerXDLValue The MFMA instruction size in N dimension.
 * \tparam MXdlPerWaveValue  The number of MFMA instructions run by single
 * wave in M dimension.
 * \tparam NXdlPerWaveValue  The number of MFMA instructions run by single
 * wave in N dimension.
 * \tparam K1Value The number of K-dim elements that are packed together as
 * a separate logical dimension. Usually aligns with vector load size.
 */
template <index_t MPerXDLValue,
          index_t NPerXDLValue,
          index_t MXdlPerWaveValue,
          index_t NXdlPerWaveValue,
          index_t K1Value>
struct BlockwisGemmXdlTraits
{
    static constexpr index_t MPerXDL     = MPerXDLValue;
    static constexpr index_t NPerXDL     = NPerXDLValue;
    static constexpr index_t MXdlPerWave = MXdlPerWaveValue;
    static constexpr index_t NXdlPerWave = NXdlPerWaveValue;
    static constexpr index_t K1          = K1Value;
};

// K1 = 4
struct BlockwisGemmXdlTraits_32x32Xdl_4x2XdlPerWave_4K1 : BlockwisGemmXdlTraits<32, 32, 4, 2, 4>
{
};
struct BlockwisGemmXdlTraits_32x32Xdl_2x4XdlPerWave_4K1 : BlockwisGemmXdlTraits<32, 32, 2, 4, 4>
{
};
struct BlockwisGemmXdlTraits_32x32Xdl_2x2XdlPerWave_4K1 : BlockwisGemmXdlTraits<32, 32, 2, 2, 4>
{
};
// K1 = 8
struct BlockwisGemmXdlTraits_32x32Xdl_4x2XdlPerWave_8K1 : BlockwisGemmXdlTraits<32, 32, 4, 2, 8>
{
};
struct BlockwisGemmXdlTraits_32x32Xdl_2x4XdlPerWave_8K1 : BlockwisGemmXdlTraits<32, 32, 2, 4, 8>
{
};
struct BlockwisGemmXdlTraits_32x32Xdl_2x2XdlPerWave_8K1 : BlockwisGemmXdlTraits<32, 32, 2, 2, 8>
{
};
// K1 = 16
struct BlockwisGemmXdlTraits_32x32Xdl_4x2XdlPerWave_16K1 : BlockwisGemmXdlTraits<32, 32, 4, 2, 16>
{
};
struct BlockwisGemmXdlTraits_32x32Xdl_2x4XdlPerWave_16K1 : BlockwisGemmXdlTraits<32, 32, 2, 4, 16>
{
};
struct BlockwisGemmXdlTraits_32x32Xdl_2x2XdlPerWave_16K1 : BlockwisGemmXdlTraits<32, 32, 2, 2, 16>
{
};

} // namespace wrapper
} // namespace ck
