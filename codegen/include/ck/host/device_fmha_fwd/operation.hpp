// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <vector>
#include <string>
#include "ck/host/types.hpp"
#include "ck/host/device_fmha_fwd/problem.hpp"

namespace ck {
namespace host {
namespace device_fmha_fwd {

// Derived from fmha_fwd.py FmhaFwdTileSize.
struct TileConfig
{
    // Block tile
    std::size_t bm0;
    std::size_t bn0;
    std::size_t bk0;
    std::size_t bn1;
    std::size_t bk1;
    std::size_t bk0max;

    // Gemm0 block warps
    std::size_t rm0;
    std::size_t rn0;
    std::size_t rk0;

    // Gemm1 block warps
    std::size_t rm1;
    std::size_t rn1;
    std::size_t rk1;

    // Gemm0 warp tile
    std::size_t wm0;
    std::size_t wn0;
    std::size_t wk0;

    // Gemm1 warp tile
    std::size_t wm1;
    std::size_t wn1;
    std::size_t wk1;
};

struct PipelineConfig
{
    std::string name;
    bool pad_m;
    bool pad_n;
    bool pad_k;
    bool pad_o;
};

struct Operation
{
    TileConfig tile = {};

    std::string pipeline = "qr_async";

    bool is_causal     = false;
    bool is_v_rowmajor = true;
    bool has_bias      = false;
    DataType dtype     = DataType::Half;

    bool pad_m = true; // pad seqlen_q
    bool pad_n = true; // pad seqlen_k
    bool pad_k = true; // pad hdim_q
    bool pad_o = true; // pad hdim_v

    static std::vector<Operation> CreateOperations(const Problem& prob, const std::string& arch);

    Solution ToSolution() const;
};

struct HdimBucketResult
{
    std::size_t bucket_hdim   = 0;
    std::size_t bucket_hdim_v = 0;
    std::vector<TileConfig> tiles;
};

HdimBucketResult
GetTileConfigsForHdim(const std::string& arch, DataType dtype, std::size_t K, std::size_t O);

bool IsSupportedArch(const std::string& arch);

} // namespace device_fmha_fwd
} // namespace host
} // namespace ck
