// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/device_fmha_fwd/operation.hpp"
#include "ck/host/device_fmha_fwd/problem.hpp"
#include "ck/host/stringutils.hpp"
#include <map>
#include <string>
#include <vector>

namespace ck {
namespace host {
namespace device_fmha_fwd {

static const char* const FmhaFwdWrapperTemplate =
    "ck_tile::FmhaFwdWrapper<${DataType}, "
    "${BM0}, ${BN0}, ${BK0}, ${BN1}, ${BK1}, ${BK0Max}, "
    "${RM0}, ${RN0}, ${RK0}, ${RM1}, ${RN1}, ${RK1}, "
    "${WM0}, ${WN0}, ${WK0}, ${WM1}, ${WN1}, ${WK1}, "
    "${IsCausal}, ${IsVRowMajor}, ${HasBias}, "
    "${PadM}, ${PadN}, ${PadK}, ${PadO}, "
    "ck_tile::FmhaPipelineTag::${PipelineTag}>";

static bool IsGfx950(const std::string& arch) { return arch.find("gfx950") == 0; }
static bool IsGfx12(const std::string& arch) { return arch.find("gfx12") == 0; }

using TileMap = std::map<std::pair<std::size_t, std::size_t>, std::vector<TileConfig>>;

// gfx9 fp16/bf16 tile configurations
//
// Constraints that must be satisfied:
//   - rn0 = rk0 = rn1 = rk1 = 1 (only M-dimension warp distribution supported)
//   - rm0 == rm1 (BlockGemm requires identical thread buffer sizes between GEMM0/GEMM1)
//   - bk0max >= 2 * bk0 (k0_loops >= 2 required for correct pipelining)
//   - bk0 >= wk0 (block K must be at least warp K; for fp16 min wk0 is 16)
//   - bn1 = hdim_v (output head dimension processed per block)
//   - bk1 = 32 (fixed for softmax/attention score reduction pipelining)
//   - (wm0, wn0, wk0) and (wm1, wn1, wk1) must be valid MFMA sizes for the dtype
//   - rm0=8 not supported when bn1 is not power-of-2 (V tensor distribution alignment)
//
// Valid fp16 MFMA sizes: (32,32,16), (16,16,16), (16,16,32), (4,64,16), (64,4,16)
// However, not all are usable in this kernel:
//   - (64,4,16), (4,64,16): warp_gemm_dispatcher has no template specialization
//   - (32,32,8): produces invalid results (likely internal kernel issue)
//   - (16,16,32): requires bk0 >= 2*wk0 (bk0 >= 64), only usable when bk0max >= 128
//
// clang-format off
static const TileMap gfx9_fp16_tiles = {
    //             bm0, bn0, bk0, bn1, bk1,bk0max,rm0,rn0,rk0,rm1,rn1,rk1, wm0,wn0,wk0, wm1,wn1,wk1
    {{32, 32},   {{128,  64,  16,  32,  32,  32,   4,  1,  1,   4,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 64,  64,  16,  32,  32,  32,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 64,  64,  16,  32,  32,  32,   2,  1,  1,   2,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 32,  64,  16,  32,  32,  32,   2,  1,  1,   2,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 16,  32,  16,  32,  32,  32,   1,  1,  1,   1,  1,  1,  16, 16, 16,  16, 16, 16},
                  {128,  64,  16,  32,  32,  32,   8,  1,  1,   8,  1,  1,  16, 16, 16,  16, 16, 16}}},

    {{64, 64},   {{128,  64,  32,  64,  32,  64,   4,  1,  1,   4,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 64,  64,  32,  64,  32,  64,   2,  1,  1,   2,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 32,  64,  32,  64,  32,  64,   2,  1,  1,   2,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 64,  64,  32,  64,  32,  64,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16},
                  {128,  64,  32,  64,  32,  64,   8,  1,  1,   8,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 16,  64,  32,  64,  32,  64,   1,  1,  1,   1,  1,  1,  16, 16, 16,  16, 16, 16}}},

    {{80, 96},   {{128, 128,  16,  96,  32,  80,   4,  1,  1,   4,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 16, 128,  16,  96,  32,  80,   1,  1,  1,   1,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 32, 128,  16,  96,  32,  80,   2,  1,  1,   2,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 64, 128,  16,  96,  32,  80,   2,  1,  1,   2,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 64, 128,  16,  96,  32,  80,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16}}},

    {{96, 128},  {{128, 128,  32, 128,  32,  96,   4,  1,  1,   4,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 16, 128,  32, 128,  32,  96,   1,  1,  1,   1,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 32, 128,  32, 128,  32,  96,   2,  1,  1,   2,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 64, 128,  32, 128,  32,  96,   2,  1,  1,   2,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 64, 128,  32, 128,  32,  96,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16},
                  {128, 128,  32, 128,  32,  96,   8,  1,  1,   8,  1,  1,  16, 16, 16,  16, 16, 16}}},

    {{128, 128}, {{ 64, 128,  32, 128,  32, 128,   4,  1,  1,   4,  1,  1,  16, 16, 32,  16, 16, 16},
                  {128,  64,  32, 128,  16, 128,   4,  1,  1,   4,  1,  1,  32, 32, 16,  32, 32, 16},
                  {128, 128,  32, 128,  32, 128,   4,  1,  1,   4,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 32, 128,  32, 128,  32, 128,   2,  1,  1,   2,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 64, 128,  32, 128,  32, 128,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16},
                  {128, 128,  32, 128,  32, 128,   8,  1,  1,   8,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 64, 128,  32, 128,  32, 128,   2,  1,  1,   2,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 32, 128,  64, 128,  32, 128,   2,  1,  1,   2,  1,  1,  16, 16, 32,  16, 16, 16},
                  { 64, 128,  64, 128,  32, 128,   4,  1,  1,   4,  1,  1,  16, 16, 32,  16, 16, 16},
                  {128, 128,  64, 128,  32, 128,   8,  1,  1,   8,  1,  1,  16, 16, 32,  16, 16, 16}}},

    {{192, 128}, {{128, 128,  32, 128,  32, 192,   4,  1,  1,   4,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 64, 128,  32, 128,  32, 192,   4,  1,  1,   4,  1,  1,  16, 16, 32,  16, 16, 16},
                  { 16, 128,  32, 128,  32, 192,   1,  1,  1,   1,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 32, 128,  32, 128,  32, 192,   2,  1,  1,   2,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 64, 128,  32, 128,  32, 192,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16},
                  {128, 128,  32, 128,  32, 192,   8,  1,  1,   8,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 64, 128,  32, 128,  32, 192,   2,  1,  1,   2,  1,  1,  32, 32, 16,  32, 32, 16},
                  {256, 128,  32, 128,  32, 192,   8,  1,  1,   8,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 32, 128,  32, 128,  32, 192,   2,  1,  1,   2,  1,  1,  16, 16, 32,  16, 16, 16},
                  {128, 128,  32, 128,  32, 192,   8,  1,  1,   8,  1,  1,  16, 16, 32,  16, 16, 16},
                  { 32, 128,  64, 128,  32, 192,   2,  1,  1,   2,  1,  1,  16, 16, 32,  16, 16, 16},
                  { 64, 128,  64, 128,  32, 192,   4,  1,  1,   4,  1,  1,  16, 16, 32,  16, 16, 16},
                  {128, 128,  64, 128,  32, 192,   8,  1,  1,   8,  1,  1,  16, 16, 32,  16, 16, 16}}},

    {{192, 192}, {{128, 128,  32, 192,  32, 192,   4,  1,  1,   4,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 64, 128,  32, 192,  32, 192,   2,  1,  1,   2,  1,  1,  32, 32, 16,  32, 32, 16},
                  {256, 128,  32, 192,  32, 192,   8,  1,  1,   8,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 16, 128,  32, 192,  32, 192,   1,  1,  1,   1,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 32, 128,  32, 192,  32, 192,   2,  1,  1,   2,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 64, 128,  32, 192,  32, 192,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16},
                  {128, 128,  32, 192,  32, 192,   8,  1,  1,   8,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 32, 128,  32, 192,  32, 192,   2,  1,  1,   2,  1,  1,  16, 16, 32,  16, 16, 16},
                  {128, 128,  32, 192,  32, 192,   8,  1,  1,   8,  1,  1,  16, 16, 32,  16, 16, 16}}},

    {{256, 256}, {{128, 128,  32, 256,  32, 256,   4,  1,  1,   4,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 64, 128,  32, 256,  32, 256,   2,  1,  1,   2,  1,  1,  32, 32, 16,  32, 32, 16},
                  {256, 128,  32, 256,  32, 256,   8,  1,  1,   8,  1,  1,  32, 32, 16,  32, 32, 16},
                  { 16, 128,  32, 256,  32, 256,   1,  1,  1,   1,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 32, 128,  32, 256,  32, 256,   2,  1,  1,   2,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 64, 128,  32, 256,  32, 256,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16},
                  {128, 128,  32, 256,  32, 256,   8,  1,  1,   8,  1,  1,  16, 16, 16,  16, 16, 16},
                  { 32, 128,  32, 256,  32, 256,   2,  1,  1,   2,  1,  1,  16, 16, 32,  16, 16, 16},
                  {128, 128,  32, 256,  32, 256,   8,  1,  1,   8,  1,  1,  16, 16, 32,  16, 16, 16}}},
};

// TODO WIP - Currently not used, additional configs will be added later
// gfx12 fp16/bf16 tiles from KernelComponentFactoryGfx12::get_hdim_tile_size_dict
static const TileMap gfx12_fp16_tiles = {
    //             bm0, bn0, bk0, bn1, bk1,bk0max,rm0,rn0,rk0,rm1,rn1,rk1, wm0,wn0,wk0, wm1,wn1,wk1
    {{32, 32},   {{ 64,  64,  16,  32,  32,  32,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16}}},
    {{64, 64},   {{ 64,  64,  32,  64,  32,  64,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16}}},
    {{128, 128}, {{ 64,  64,  32, 128,  32, 128,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16}}},
    {{192, 128}, {{ 64,  64,  32, 128,  32, 256,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16}}},
    {{256, 256}, {{ 64,  64,  32, 256,  32, 256,   4,  1,  1,   4,  1,  1,  16, 16, 16,  16, 16, 16}}},
};
// clang-format on

HdimBucketResult
GetTileConfigsForHdim(const std::string& arch, DataType dtype, std::size_t K, std::size_t O)
{
    HdimBucketResult result;

    if(dtype != DataType::Half)
        return result;

    const TileMap& tile_map = IsGfx12(arch) ? gfx12_fp16_tiles : gfx9_fp16_tiles;

    for(const auto& [key, tiles] : tile_map)
    {
        if(K <= key.first && O <= key.second)
        {
            result.bucket_hdim   = key.first;
            result.bucket_hdim_v = key.second;
            result.tiles         = tiles;
            return result;
        }
    }

    return result;
}

static std::vector<PipelineConfig> GetPipelinesGfx12()
{
    // QR pipeline is handled separately in CreateOperations with exact padding
    return {};
}

static std::vector<PipelineConfig>
GetPipelinesGfx9(std::size_t bucket_hdim, std::size_t bucket_hdim_v, bool has_bias)
{
    // QR pipeline is handled separately in CreateOperations with exact padding
    std::vector<PipelineConfig> configs;

    // QR_ASYNC pipeline requires pad_m=true, pad_k=true, pad_o=true (enforced by static_assert
    // in BlockFmhaPipelineQRKSVSAsync). Only pad_n is variable.
    if(!has_bias)
    {
        configs.push_back({"qr_async", true, false, true, true}); // pad_n=false
        configs.push_back({"qr_async", true, true, true, true});  // pad_n=true
    }

    // Note: qr_async_trload requires gfx950+ (uses buffer_load_dwordx3/x4 instructions)

    return configs;
}

// TODO WIP - Currently not used, additional configs will be added later
static std::vector<PipelineConfig>
GetPipelinesGfx950(std::size_t bucket_hdim, std::size_t bucket_hdim_v, bool has_bias)
{
    auto configs = GetPipelinesGfx9(bucket_hdim, bucket_hdim_v, has_bias);

    bool is_hdim_256 = (bucket_hdim == 256 && bucket_hdim_v == 256);
    if(!is_hdim_256 && !has_bias)
    {
        configs.push_back({"qr_async_trload", false, false, false, false});
        configs.push_back({"qr_async_trload", false, false, true, true});
    }

    return configs;
}

static std::vector<PipelineConfig> GetPipelineConfigs(const std::string& arch,
                                                      std::size_t bucket_hdim,
                                                      std::size_t bucket_hdim_v,
                                                      bool has_bias)
{
    if(IsGfx12(arch))
        return GetPipelinesGfx12();
    if(IsGfx950(arch))
        return GetPipelinesGfx950(bucket_hdim, bucket_hdim_v, has_bias);
    return GetPipelinesGfx9(bucket_hdim, bucket_hdim_v, has_bias);
}

static bool IsPaddingCompatible(const PipelineConfig& config,
                                const Problem& prob,
                                const TileConfig& tile,
                                std::size_t bucket_hdim,
                                std::size_t bucket_hdim_v)
{
    bool needs_pad_m = (prob.M % tile.bm0 != 0);
    bool needs_pad_n = (prob.N % tile.bn0 != 0);
    bool needs_pad_k = (prob.K != bucket_hdim);
    bool needs_pad_o = (prob.O != bucket_hdim_v);

    // +------------+----------+------------+
    // | config.pad | needs_pad| compatible |
    // +------------+----------+------------+
    // |   false    |  false   |    true    |
    // |   false    |  true    |    false   |
    // |   true     |  false   |    true    |
    // |   true     |  true    |    true    |
    // +------------+----------+------------+
    //
    return (config.pad_m || !needs_pad_m) && (config.pad_n || !needs_pad_n) &&
           (config.pad_k || !needs_pad_k) && (config.pad_o || !needs_pad_o);
}

std::vector<Operation> Operation::CreateOperations(const Problem& prob, const std::string& arch)
{
    std::vector<Operation> result;

    auto bucket = GetTileConfigsForHdim(arch, prob.dtype, prob.K, prob.O);
    auto pipelines =
        GetPipelineConfigs(arch, bucket.bucket_hdim, bucket.bucket_hdim_v, prob.has_bias);

    for(const auto& tile : bucket.tiles)
    {
        // Compute exact padding needs for this tile
        bool needs_pad_m = (prob.M % tile.bm0 != 0);
        bool needs_pad_n = (prob.N % tile.bn0 != 0);
        bool needs_pad_k = (prob.K != bucket.bucket_hdim);
        bool needs_pad_o = (prob.O != bucket.bucket_hdim_v);

        // QR pipeline: create one operation with exact padding
        {
            Operation op;
            op.tile          = tile;
            op.pipeline      = "qr";
            op.is_causal     = prob.is_causal;
            op.is_v_rowmajor = prob.is_v_rowmajor;
            op.has_bias      = prob.has_bias;
            op.dtype         = prob.dtype;
            op.pad_m         = needs_pad_m;
            op.pad_n         = needs_pad_n;
            op.pad_k         = needs_pad_k;
            op.pad_o         = needs_pad_o;
            result.push_back(op);
        }

        // Async pipelines: use predefined configs with filters
        for(const auto& pipeline : pipelines)
        {
            if(prob.dtype == DataType::Half && (prob.K % 8 != 0 || prob.O % 8 != 0))
                continue;
            // Single-warp configs (rm0=1) produce incorrect results with async pipelines
            if(tile.rm0 == 1)
                continue;
            // (96, 128) bucket: rm0 >= 4 with pad_n=false produces incorrect results
            if(bucket.bucket_hdim == 96 && bucket.bucket_hdim_v == 128)
            {
                if(!pipeline.pad_n && tile.rm0 >= 4)
                    continue;
            }
            // (128, 128) bucket filters for async pipelines:
            //   - bn0=64, bk1=16 config produces invalid results
            //   - bk0=64 configs (MFMA 16x16x32) produce invalid results
            if(bucket.bucket_hdim == 128 && bucket.bucket_hdim_v == 128)
            {
                if(tile.bn0 == 64 && tile.bk1 == 16)
                    continue;
                if(tile.bk0 == 64)
                    continue;
            }
            // (192, 128) bucket filters for async pipelines
            if(bucket.bucket_hdim == 192 && bucket.bucket_hdim_v == 128)
            {
                // bk0=64 configs produce invalid results
                if(tile.bk0 == 64)
                    continue;
                // pad_n=false fails for wm0=32 (MFMA 32x32x16) or rm0>=4
                if(!pipeline.pad_n && (tile.wm0 == 32 || tile.rm0 >= 4))
                    continue;
            }
            // (192, 192) bucket filters for async pipelines
            if(bucket.bucket_hdim == 192 && bucket.bucket_hdim_v == 192)
            {
                // rm0=8 with wm0=32 always fails (even with pad_n=true)
                if(tile.rm0 == 8 && tile.wm0 == 32)
                    continue;
                // pad_n=false fails except for (rm0=2, wm0=32) and (rm0=8, wk0=16)
                if(!pipeline.pad_n)
                {
                    bool is_valid =
                        (tile.rm0 == 2 && tile.wm0 == 32) || (tile.rm0 == 8 && tile.wk0 == 16);
                    if(!is_valid)
                        continue;
                }
            }

            if(!IsPaddingCompatible(pipeline, prob, tile, bucket.bucket_hdim, bucket.bucket_hdim_v))
                continue;

            Operation op;
            op.tile          = tile;
            op.pipeline      = pipeline.name;
            op.is_causal     = prob.is_causal;
            op.is_v_rowmajor = prob.is_v_rowmajor;
            op.has_bias      = prob.has_bias;
            op.dtype         = prob.dtype;
            op.pad_m         = pipeline.pad_m;
            op.pad_n         = pipeline.pad_n;
            op.pad_k         = pipeline.pad_k;
            op.pad_o         = pipeline.pad_o;
            result.push_back(op);
        }
    }

    return result;
}

static std::string ToDataTypeString(DataType dtype)
{
    switch(dtype)
    {
    case DataType::Half: return "ck_tile::fp16_t";
    case DataType::Float: return "float";
    default: return "ck_tile::fp16_t";
    }
}

Solution Operation::ToSolution() const
{
    std::unordered_map<std::string, std::string> values = {
        {"DataType", ToDataTypeString(dtype)},

        {"BM0", std::to_string(tile.bm0)},
        {"BN0", std::to_string(tile.bn0)},
        {"BK0", std::to_string(tile.bk0)},
        {"BN1", std::to_string(tile.bn1)},
        {"BK1", std::to_string(tile.bk1)},
        {"BK0Max", std::to_string(tile.bk0max)},

        {"RM0", std::to_string(tile.rm0)},
        {"RN0", std::to_string(tile.rn0)},
        {"RK0", std::to_string(tile.rk0)},

        {"RM1", std::to_string(tile.rm1)},
        {"RN1", std::to_string(tile.rn1)},
        {"RK1", std::to_string(tile.rk1)},

        {"WM0", std::to_string(tile.wm0)},
        {"WN0", std::to_string(tile.wn0)},
        {"WK0", std::to_string(tile.wk0)},

        {"WM1", std::to_string(tile.wm1)},
        {"WN1", std::to_string(tile.wn1)},
        {"WK1", std::to_string(tile.wk1)},

        {"IsCausal", is_causal ? "true" : "false"},
        {"IsVRowMajor", is_v_rowmajor ? "true" : "false"},
        {"HasBias", has_bias ? "true" : "false"},

        {"PadM", pad_m ? "true" : "false"},
        {"PadN", pad_n ? "true" : "false"},
        {"PadK", pad_k ? "true" : "false"},
        {"PadO", pad_o ? "true" : "false"},

        {"PipelineTag",
         pipeline == "qr_async_trload" ? "QR_ASYNC_TRLOAD"
                                       : (pipeline == "qr_async" ? "QR_ASYNC" : "QR")},
    };

    return Solution{InterpolateString(FmhaFwdWrapperTemplate, values), std::move(values)};
}

} // namespace device_fmha_fwd
} // namespace host
} // namespace ck
