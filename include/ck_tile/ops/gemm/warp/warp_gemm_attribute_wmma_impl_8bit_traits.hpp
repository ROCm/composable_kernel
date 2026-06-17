// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "warp_gemm_attribute_wmma_impl_base_traits.hpp"
#include "warp_gemm_params.hpp"
namespace ck_tile {

struct WmmaScale16Tag;

// int8 specialization - GFX11
template <>
struct WmmaTraits<gfx11_t, int8_t, int8_t, int32_t, 16, 16, 16>
    : WmmaTraitsBase<gfx11_t, int8_t, int8_t, int32_t, 16>
{
    using ArchType = gfx11_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx11__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, // neg_a
                                                          bit_cast<int32x4_t>(a_vec),
                                                          true, // neg_b
                                                          bit_cast<int32x4_t>(b_vec),
                                                          bit_cast<int32x8_t>(c_vec),
                                                          P::clamp);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

// int8 specialization - GFX12
template <>
struct WmmaTraits<gfx120_t, int8_t, int8_t, int32_t, 16, 16, 16>
    : WmmaTraitsBase<gfx12_t, int8_t, int8_t, int32_t, 16>
{
    using ArchType = gfx120_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx120__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, // neg_a
                                                                bit_cast<int32x2_t>(a_vec),
                                                                true, // neg_b
                                                                bit_cast<int32x2_t>(b_vec),
                                                                bit_cast<int32x8_t>(c_vec),
                                                                P::clamp);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

// fp8/bf8 specialization - GFX12
template <>
struct WmmaTraits<gfx120_t, fp8_t, fp8_t, float, 16, 16, 16>
    : WmmaTraitsBase<gfx12_t, fp8_t, fp8_t, float, 16>
{
    using ArchType = gfx120_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx120__
        return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
            bit_cast<int32x2_t>(a_vec), bit_cast<int32x2_t>(b_vec), bit_cast<fp32x8_t>(c_vec));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0.f};
#endif
    }
};

template <>
struct WmmaTraits<gfx120_t, bf8_t, bf8_t, float, 16, 16, 16>
    : WmmaTraitsBase<gfx12_t, bf8_t, bf8_t, float, 16>
{
    using ArchType = gfx120_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx120__
        return __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(
            bit_cast<int32x2_t>(a_vec), bit_cast<int32x2_t>(b_vec), bit_cast<fp32x8_t>(c_vec));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx120_t, fp8_t, bf8_t, float, 16, 16, 16>
    : WmmaTraitsBase<gfx12_t, fp8_t, bf8_t, float, 16>
{
    using ArchType = gfx120_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx120__
        return __builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12(
            bit_cast<int32x2_t>(a_vec), bit_cast<int32x2_t>(b_vec), bit_cast<fp32x8_t>(c_vec));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx120_t, bf8_t, fp8_t, float, 16, 16, 16>
    : WmmaTraitsBase<gfx12_t, bf8_t, fp8_t, float, 16>
{
    using ArchType = gfx120_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx120__
        return __builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12(
            bit_cast<int32x2_t>(a_vec), bit_cast<int32x2_t>(b_vec), bit_cast<fp32x8_t>(c_vec));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0.f};
#endif
    }
};

// iu8 specialization - GFX125
template <>
struct WmmaTraits<gfx125_t, int8_t, int8_t, int32_t, 16, 16, 64>
    : WmmaTraitsBase<gfx12_t, int8_t, int8_t, int32_t, 64>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_i32_16x16x64_iu8(true, // neg_a
                                                      bit_cast<int32x8_t>(a_vec),
                                                      true, // neg_b
                                                      bit_cast<int32x8_t>(b_vec),
                                                      bit_cast<int32x8_t>(c_vec),
                                                      P::reuse_a,
                                                      P::reuse_b);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx125_t, uint8_t, uint8_t, uint32_t, 16, 16, 64>
    : WmmaTraitsBase<gfx12_t, uint8_t, uint8_t, uint32_t, 64>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_i32_16x16x64_iu8(false, // neg_a
                                                      bit_cast<int32x8_t>(a_vec),
                                                      false, // neg_b
                                                      bit_cast<int32x8_t>(b_vec),
                                                      bit_cast<int32x8_t>(c_vec),
                                                      P::reuse_a,
                                                      P::reuse_b);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

// fp8/bf8 specialization - GFX125
template <>
struct WmmaTraits<gfx125_t, fp8_t, fp8_t, float, 16, 16, 64>
    : WmmaTraitsBase<gfx12_t, fp8_t, fp8_t, float, 64>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8(bit_cast<int32x8_t>(a_vec),
                                                          bit_cast<int32x8_t>(b_vec),
                                                          0,
                                                          bit_cast<fp32x8_t>(c_vec),
                                                          P::reuse_a,
                                                          P::reuse_b);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx125_t, bf8_t, bf8_t, float, 16, 16, 64>
    : WmmaTraitsBase<gfx12_t, bf8_t, bf8_t, float, 64>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_f32_16x16x64_bf8_bf8(bit_cast<int32x8_t>(a_vec),
                                                          bit_cast<int32x8_t>(b_vec),
                                                          0,
                                                          bit_cast<fp32x8_t>(c_vec),
                                                          P::reuse_a,
                                                          P::reuse_b);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx125_t, fp8_t, bf8_t, float, 16, 16, 64>
    : WmmaTraitsBase<gfx12_t, fp8_t, bf8_t, float, 64>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_f32_16x16x64_fp8_bf8(bit_cast<int32x8_t>(a_vec),
                                                          bit_cast<int32x8_t>(b_vec),
                                                          0,
                                                          bit_cast<fp32x8_t>(c_vec),
                                                          P::reuse_a,
                                                          P::reuse_b);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx125_t, bf8_t, fp8_t, float, 16, 16, 64>
    : WmmaTraitsBase<gfx12_t, bf8_t, fp8_t, float, 64>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_f32_16x16x64_bf8_fp8(bit_cast<int32x8_t>(a_vec),
                                                          bit_cast<int32x8_t>(b_vec),
                                                          0,
                                                          bit_cast<fp32x8_t>(c_vec),
                                                          P::reuse_a, // matrix_a_reuse
                                                          P::reuse_b);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx125_t, fp8_t, fp8_t, fp16_t, 16, 16, 64>
    : WmmaTraitsBase<gfx12_t, fp8_t, fp8_t, fp16_t, 64>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_f16_16x16x64_fp8_fp8(bit_cast<int32x8_t>(a_vec),
                                                          bit_cast<int32x8_t>(b_vec),
                                                          0,
                                                          bit_cast<fp16x8_t>(c_vec),
                                                          P::reuse_a,
                                                          P::reuse_b);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx125_t, fp8_t, bf8_t, fp16_t, 16, 16, 64>
    : WmmaTraitsBase<gfx12_t, fp8_t, bf8_t, fp16_t, 64>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_f16_16x16x64_fp8_bf8(bit_cast<int32x8_t>(a_vec),
                                                          bit_cast<int32x8_t>(b_vec),
                                                          0,
                                                          bit_cast<fp16x8_t>(c_vec),
                                                          P::reuse_a,
                                                          P::reuse_b);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};
template <>
struct WmmaTraits<gfx125_t, bf8_t, fp8_t, fp16_t, 16, 16, 64>
    : WmmaTraitsBase<gfx12_t, bf8_t, fp8_t, fp16_t, 64>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_f16_16x16x64_bf8_fp8(bit_cast<int32x8_t>(a_vec),
                                                          bit_cast<int32x8_t>(b_vec),
                                                          0,
                                                          bit_cast<fp16x8_t>(c_vec),
                                                          P::reuse_a,
                                                          P::reuse_b);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx125_t, bf8_t, bf8_t, fp16_t, 16, 16, 64>
    : WmmaTraitsBase<gfx12_t, bf8_t, bf8_t, fp16_t, 64>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_f16_16x16x64_bf8_bf8(bit_cast<int32x8_t>(a_vec),
                                                          bit_cast<int32x8_t>(b_vec),
                                                          0,
                                                          bit_cast<fp16x8_t>(c_vec),
                                                          P::reuse_a,
                                                          P::reuse_b);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

// f8f6f4 specialization - GFX125
enum F8F6F4OpDataTypeEnum
{
    E4M3, // 0x0
    E5M2, // 0x1
    E2M3, // 0x2
    E3M2, // 0x3
    E2M1, // 0x4
};

// Traits for MX data types used in f8f6f4 intrinsics
template <typename T>
struct MXDataTypeTrait;

template <>
struct MXDataTypeTrait<fp8_t>
{
    static constexpr F8F6F4OpDataTypeEnum OpDataType = F8F6F4OpDataTypeEnum::E4M3;
    using VecType                                    = int32x16_t;

    CK_TILE_DEVICE static int32x16_t to_wmma_vec(const int32x16_t& vec) { return vec; }
};

template <>
struct MXDataTypeTrait<bf8_t>
{
    static constexpr F8F6F4OpDataTypeEnum OpDataType = F8F6F4OpDataTypeEnum::E5M2;
    using VecType                                    = int32x16_t;

    CK_TILE_DEVICE static int32x16_t to_wmma_vec(const int32x16_t& vec) { return vec; }
};

template <>
struct MXDataTypeTrait<pk_fp4_t>
{
    static constexpr F8F6F4OpDataTypeEnum OpDataType = F8F6F4OpDataTypeEnum::E2M1;
    using VecType                                    = int32x8_t;

    CK_TILE_DEVICE static int32x16_t to_wmma_vec(const int32x8_t& vec)
    {
        return int32x16_t{
            vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], vec[7], 0, 0, 0, 0, 0, 0, 0, 0};
    }
};

// pk_fp6x16_t (legacy): 16 fp6 e2m3 values packed into 3 int32 (96 bits).
// At 16x16x128 each lane holds 64 fp6 elements = 4 packs = 12 int32
// (f6x16xN_tt<4, f6_kind::fp6>, whose storage is int32_t data[12]),
// padded with 4 zero lanes to fit the 16-wide f8f6f4 wmma input.
template <>
struct MXDataTypeTrait<pk_fp6x16_t>
{
    static constexpr F8F6F4OpDataTypeEnum OpDataType = F8F6F4OpDataTypeEnum::E2M3;
    using VecType                                    = f6x16xN_tt<4, f6_kind::fp6>;

    CK_TILE_DEVICE static int32x16_t to_wmma_vec(const f6x16xN_tt<4, f6_kind::fp6>& vec)
    {
        return int32x16_t{vec.data[0],
                          vec.data[1],
                          vec.data[2],
                          vec.data[3],
                          vec.data[4],
                          vec.data[5],
                          vec.data[6],
                          vec.data[7],
                          vec.data[8],
                          vec.data[9],
                          vec.data[10],
                          vec.data[11],
                          0,
                          0,
                          0,
                          0};
    }
};

template <>
struct WmmaTraits<gfx125_t, bf8_t, bf8_t, float, 16, 16, 128>
    : WmmaTraitsBase<gfx12_t, bf8_t, bf8_t, float, 128>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_f32_16x16x128_bf8_bf8(bit_cast<int32x16_t>(a_vec),
                                                           bit_cast<int32x16_t>(b_vec),
                                                           0,
                                                           bit_cast<fp32x8_t>(c_vec),
                                                           P::reuse_a,  // matrix_a_reuse
                                                           P::reuse_b); // matrix_b_reuse
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx125_t, fp8_t, bf8_t, float, 16, 16, 128>
    : WmmaTraitsBase<gfx12_t, fp8_t, bf8_t, float, 128>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_f32_16x16x128_fp8_bf8(bit_cast<int32x16_t>(a_vec),
                                                           bit_cast<int32x16_t>(b_vec),
                                                           0,
                                                           bit_cast<fp32x8_t>(c_vec),
                                                           P::reuse_a,  // matrix_a_reuse
                                                           P::reuse_b); // matrix_b_reuse
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx125_t, bf8_t, fp8_t, float, 16, 16, 128>
    : WmmaTraitsBase<gfx12_t, bf8_t, fp8_t, float, 128>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_f32_16x16x128_bf8_fp8(bit_cast<int32x16_t>(a_vec),
                                                           bit_cast<int32x16_t>(b_vec),
                                                           0,
                                                           bit_cast<fp32x8_t>(c_vec),
                                                           P::reuse_a,  // matrix_a_reuse
                                                           P::reuse_b); // matrix_b_reuse
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

// scale16 specializations: fp8xfp8, bf8xbf8, fp8xbf8, bf8xfp8
// Override kAK1PerLane/kBK1PerLane to 16 for scale16 register layout -> sequence<4,2,16>
template <>
struct WmmaTraits<gfx125_t, fp8_t, fp8_t, float, 16, 16, 128, WmmaScale16Tag>
    : WmmaTraitsBase<gfx12_t, fp8_t, fp8_t, float, 128>
{
    using ArchType         = gfx125_t;
    using MXTypeEnableType = WmmaScale16Tag;

    static constexpr index_t kAK1PerLane = 16;
    static constexpr index_t kAK0PerLane = kK / (kAK1PerLane * kABKLane);
    static constexpr index_t kBK1PerLane = 16;
    static constexpr index_t kBK0PerLane = kK / (kBK1PerLane * kABKLane);

    template <typename... Params>
    CK_TILE_DEVICE static CVecType wmma_intrinsic(const AVecType&, const BVecType&, const CVecType&)
    {
        static_assert(sizeof...(Params) < 0, "scale16 WmmaTraits requires int64_t scale arguments");
        return CVecType{0};
    }

    template <typename... Params>
    CK_TILE_DEVICE static CVecType wmma_intrinsic(const AVecType& a_vec,
                                                  const int64_t& a_scale,
                                                  const BVecType& b_vec,
                                                  const int64_t& b_scale,
                                                  const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P       = WarpGemmParamsParser<Params...>;
        using ATraits = MXDataTypeTrait<fp8_t>;
        using BTraits = MXDataTypeTrait<fp8_t>;
        return __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(
            ATraits::OpDataType,
            ATraits::to_wmma_vec(bit_cast<typename ATraits::VecType>(a_vec)),
            BTraits::OpDataType,
            BTraits::to_wmma_vec(bit_cast<typename BTraits::VecType>(b_vec)),
            0,
            bit_cast<fp32x8_t>(c_vec),
            P::op_sel_a,
            P::scale_a,
            a_scale,
            P::op_sel_b,
            P::scale_b,
            b_scale,
            0,
            0);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = a_scale;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = b_scale;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx125_t, bf8_t, bf8_t, float, 16, 16, 128, WmmaScale16Tag>
    : WmmaTraitsBase<gfx12_t, bf8_t, bf8_t, float, 128>
{
    using ArchType         = gfx125_t;
    using MXTypeEnableType = WmmaScale16Tag;

    static constexpr index_t kAK1PerLane = 16;
    static constexpr index_t kAK0PerLane = kK / (kAK1PerLane * kABKLane);
    static constexpr index_t kBK1PerLane = 16;
    static constexpr index_t kBK0PerLane = kK / (kBK1PerLane * kABKLane);

    template <typename... Params>
    CK_TILE_DEVICE static CVecType wmma_intrinsic(const AVecType&, const BVecType&, const CVecType&)
    {
        static_assert(sizeof...(Params) < 0, "scale16 WmmaTraits requires int64_t scale arguments");
        return CVecType{0};
    }

    template <typename... Params>
    CK_TILE_DEVICE static CVecType wmma_intrinsic(const AVecType& a_vec,
                                                  const int64_t& a_scale,
                                                  const BVecType& b_vec,
                                                  const int64_t& b_scale,
                                                  const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P       = WarpGemmParamsParser<Params...>;
        using ATraits = MXDataTypeTrait<bf8_t>;
        using BTraits = MXDataTypeTrait<bf8_t>;
        return __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(
            ATraits::OpDataType,
            ATraits::to_wmma_vec(bit_cast<typename ATraits::VecType>(a_vec)),
            BTraits::OpDataType,
            BTraits::to_wmma_vec(bit_cast<typename BTraits::VecType>(b_vec)),
            0,
            bit_cast<fp32x8_t>(c_vec),
            P::op_sel_a,
            P::scale_a,
            a_scale,
            P::op_sel_b,
            P::scale_b,
            b_scale,
            0,
            0);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = a_scale;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = b_scale;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx125_t, fp8_t, bf8_t, float, 16, 16, 128, WmmaScale16Tag>
    : WmmaTraitsBase<gfx12_t, fp8_t, bf8_t, float, 128>
{
    using ArchType         = gfx125_t;
    using MXTypeEnableType = WmmaScale16Tag;

    static constexpr index_t kAK1PerLane = 16;
    static constexpr index_t kAK0PerLane = kK / (kAK1PerLane * kABKLane);
    static constexpr index_t kBK1PerLane = 16;
    static constexpr index_t kBK0PerLane = kK / (kBK1PerLane * kABKLane);

    template <typename... Params>
    CK_TILE_DEVICE static CVecType wmma_intrinsic(const AVecType&, const BVecType&, const CVecType&)
    {
        static_assert(sizeof...(Params) < 0, "scale16 WmmaTraits requires int64_t scale arguments");
        return CVecType{0};
    }

    template <typename... Params>
    CK_TILE_DEVICE static CVecType wmma_intrinsic(const AVecType& a_vec,
                                                  const int64_t& a_scale,
                                                  const BVecType& b_vec,
                                                  const int64_t& b_scale,
                                                  const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P       = WarpGemmParamsParser<Params...>;
        using ATraits = MXDataTypeTrait<fp8_t>;
        using BTraits = MXDataTypeTrait<bf8_t>;
        return __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(
            ATraits::OpDataType,
            ATraits::to_wmma_vec(bit_cast<typename ATraits::VecType>(a_vec)),
            BTraits::OpDataType,
            BTraits::to_wmma_vec(bit_cast<typename BTraits::VecType>(b_vec)),
            0,
            bit_cast<fp32x8_t>(c_vec),
            P::op_sel_a,
            P::scale_a,
            a_scale,
            P::op_sel_b,
            P::scale_b,
            b_scale,
            0,
            0);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = a_scale;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = b_scale;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx125_t, bf8_t, fp8_t, float, 16, 16, 128, WmmaScale16Tag>
    : WmmaTraitsBase<gfx12_t, bf8_t, fp8_t, float, 128>
{
    using ArchType         = gfx125_t;
    using MXTypeEnableType = WmmaScale16Tag;

    static constexpr index_t kAK1PerLane = 16;
    static constexpr index_t kAK0PerLane = kK / (kAK1PerLane * kABKLane);
    static constexpr index_t kBK1PerLane = 16;
    static constexpr index_t kBK0PerLane = kK / (kBK1PerLane * kABKLane);

    template <typename... Params>
    CK_TILE_DEVICE static CVecType wmma_intrinsic(const AVecType&, const BVecType&, const CVecType&)
    {
        static_assert(sizeof...(Params) < 0, "scale16 WmmaTraits requires int64_t scale arguments");
        return CVecType{0};
    }

    template <typename... Params>
    CK_TILE_DEVICE static CVecType wmma_intrinsic(const AVecType& a_vec,
                                                  const int64_t& a_scale,
                                                  const BVecType& b_vec,
                                                  const int64_t& b_scale,
                                                  const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P       = WarpGemmParamsParser<Params...>;
        using ATraits = MXDataTypeTrait<bf8_t>;
        using BTraits = MXDataTypeTrait<fp8_t>;
        return __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(
            ATraits::OpDataType,
            ATraits::to_wmma_vec(bit_cast<typename ATraits::VecType>(a_vec)),
            BTraits::OpDataType,
            BTraits::to_wmma_vec(bit_cast<typename BTraits::VecType>(b_vec)),
            0,
            bit_cast<fp32x8_t>(c_vec),
            P::op_sel_a,
            P::scale_a,
            a_scale,
            P::op_sel_b,
            P::scale_b,
            b_scale,
            0,
            0);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = a_scale;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = b_scale;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

// 32x16x128 f4 specialization - GFX125
template <>
struct WmmaTraits<gfx125_t, pk_fp4_t, pk_fp4_t, float, 32, 16, 128>
    : WmmaTraitsBase<gfx12_t, pk_fp4_t, pk_fp4_t, float, 128, false, 32, 16>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        return __builtin_amdgcn_wmma_f32_32x16x128_f4(
            bit_cast<int32x16_t>(a_vec), bit_cast<int32x8_t>(b_vec), 0, bit_cast<fp32x16_t>(c_vec));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <bool IsScale16>
struct WmmaTraitsGfx125PkFp4F32_32x32x128
    : WmmaTraitsBase<gfx12_t, pk_fp4_t, pk_fp4_t, float, 128, false, 32, 32>
{
    using ArchType  = gfx125_t;
    using ScaleType = std::conditional_t<IsScale16, int64_t, int32_t>;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType wmma_intrinsic(const AVecType& a_vec,
                                                  const ScaleType& a_scale,
                                                  const BVecType& b_vec,
                                                  const ScaleType& b_scale,
                                                  const CVecType& c_vec)
    {
#ifdef __gfx125__
        using ASliceType = ext_vector_t<pk_fp4_t, sizeof(AVecType) / sizeof(pk_fp4_t)>;
        using BSliceType = ext_vector_t<pk_fp4_t, sizeof(BVecType) / sizeof(pk_fp4_t) / kCNBlock>;
        using CSliceType = fp32x16_t;

        using a_buf = thread_buffer<ASliceType, 1>;
        using b_buf = thread_buffer<BSliceType, kCNBlock>;
        using c_buf = thread_buffer<CSliceType, kCNBlock>;

        static_assert(sizeof(CVecType) == sizeof(c_buf),
                      "CVecType and c_buf must have the same size");
        static_assert(sizeof(AVecType) == sizeof(a_buf),
                      "AVecType and a_buf must have the same size");
        static_assert(sizeof(BVecType) == sizeof(b_buf),
                      "BVecType and b_buf must have the same size");

        auto&& a_buffer = bit_cast<a_buf>(a_vec);
        auto&& b_buffer = bit_cast<b_buf>(b_vec);
        auto&& c_result = bit_cast<c_buf>(c_vec);

        const auto& a_slice = a_buffer.template get_as<ASliceType>()[0];

        using P = WarpGemmParamsParser<Params...>;

        static_for<0, kCNBlock, 1>{}([&](auto n) {
            const auto& b_slice = b_buffer.template get_as<BSliceType>()[n];
            auto& c_slice       = c_result.template get_as<CSliceType>()[n];

            if constexpr(IsScale16)
            {
                c_slice = __builtin_amdgcn_wmma_scale16_f32_32x16x128_f4(
                    bit_cast<int32x16_t>(a_slice),
                    bit_cast<int32x8_t>(b_slice),
                    0,
                    c_slice,
                    1,          // OPSEL[0] - fixed to 1 for F4
                    P::scale_a, // OPSEL_HI[0] - scale data type for A
                    a_scale,
                    n.value,    // OPSEL[1] - select B scale (iterates over N blocks)
                    P::scale_b, // OPSEL_HI[1] - scale data type for B
                    b_scale,
                    0,  // NEG
                    0); // NEG_HI
            }
            else
            {
                c_slice = __builtin_amdgcn_wmma_scale_f32_32x16x128_f4(
                    bit_cast<int32x16_t>(a_slice),
                    bit_cast<int32x8_t>(b_slice),
                    0,
                    c_slice,
                    1,          // OPSEL[0] - fixed to 1 for F4
                    P::scale_a, // OPSEL_HI[0] - scale data type for A
                    a_scale,
                    n.value,    // OPSEL[1] - select B scale (iterates over N blocks)
                    P::scale_b, // OPSEL_HI[1] - scale data type for B
                    b_scale,
                    0,  // NEG
                    0); // NEG_HI
            }
        });

        return bit_cast<CVecType>(c_result);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = a_scale;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = b_scale;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        // Pass default scale values 1.0f
        Packed4Scale_E8M0 pkscale(1.0f, 1.0f, 1.0f, 1.0f);
        const auto default_scale = static_cast<ScaleType>(pkscale);
        return wmma_intrinsic(a_vec, default_scale, b_vec, default_scale, c_vec);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <>
struct WmmaTraits<gfx125_t, pk_fp4_t, pk_fp4_t, float, 32, 32, 128>
    : WmmaTraitsGfx125PkFp4F32_32x32x128<false>
{
};

template <>
struct WmmaTraits<gfx125_t, pk_fp4_t, pk_fp4_t, float, 32, 32, 128, WmmaScale16Tag>
    : WmmaTraitsGfx125PkFp4F32_32x32x128<true>
{
};

// Unified WmmaTraits for f8f6f4 combinations
template <typename AType, typename BType>
struct WmmaTraits<
    gfx125_t,
    AType,
    BType,
    float,
    16,
    16,
    128,
    std::enable_if_t<std::is_same_v<AType, pk_fp4_t> || std::is_same_v<BType, pk_fp4_t> ||
                     std::is_same_v<AType, pk_fp6x16_t> || std::is_same_v<BType, pk_fp6x16_t>>>
    : WmmaTraitsBase<gfx12_t, AType, BType, float, 128, true>
{
    using Base     = WmmaTraitsBase<gfx12_t, AType, BType, float, 128, true>;
    using ArchType = gfx125_t;

    using AVecType = typename Base::AVecType;
    using BVecType = typename Base::BVecType;
    using CVecType = typename Base::CVecType;

    using ATraits = MXDataTypeTrait<AType>;
    using BTraits = MXDataTypeTrait<BType>;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        return __builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(
            ATraits::OpDataType,
            ATraits::to_wmma_vec(bit_cast<typename ATraits::VecType>(a_vec)),
            BTraits::OpDataType,
            BTraits::to_wmma_vec(bit_cast<typename BTraits::VecType>(b_vec)),
            0,
            bit_cast<fp32x8_t>(c_vec));
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }

    template <typename... Params>
    CK_TILE_DEVICE static CVecType wmma_intrinsic(const AVecType& a_vec,
                                                  const int32_t& a_scale,
                                                  const BVecType& b_vec,
                                                  const int32_t& b_scale,
                                                  const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P = WarpGemmParamsParser<Params...>;

        return __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
            ATraits::OpDataType,
            ATraits::to_wmma_vec(bit_cast<typename ATraits::VecType>(a_vec)),
            BTraits::OpDataType,
            BTraits::to_wmma_vec(bit_cast<typename BTraits::VecType>(b_vec)),
            0,
            c_vec,
            P::op_sel_a, // OPSEL[0]
            0,           // Scale Type for A
            a_scale,
            P::op_sel_b, // OPSEL[1]
            0,           // Scale Type for B
            b_scale,
            0,  // NEG
            0); // NEG_HI
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = a_scale;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = b_scale;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

// fp8_t fp8_t specialization - GFX125
template <>
struct WmmaTraits<gfx125_t, fp8_t, fp8_t, float, 16, 16, 128>
    : WmmaTraitsBase<gfx12_t, fp8_t, fp8_t, float, 128>
{
    using ArchType = gfx125_t;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P = WarpGemmParamsParser<Params...>;
        return __builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8(bit_cast<int32x16_t>(a_vec),
                                                           bit_cast<int32x16_t>(b_vec),
                                                           0,
                                                           bit_cast<fp32x8_t>(c_vec),
                                                           P::reuse_a,  // matrix_a_reuse
                                                           P::reuse_b); // matrix_b_reuse
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }

    template <typename... Params>
    CK_TILE_DEVICE static CVecType wmma_intrinsic(const AVecType& a_vec,
                                                  const int32_t& a_scale,
                                                  const BVecType& b_vec,
                                                  const int32_t& b_scale,
                                                  const CVecType& c_vec)
    {
#ifdef __gfx125__
        using P       = WarpGemmParamsParser<Params...>;
        using ATraits = MXDataTypeTrait<fp8_t>;
        using BTraits = MXDataTypeTrait<fp8_t>;

        return __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
            ATraits::OpDataType,
            ATraits::to_wmma_vec(bit_cast<typename ATraits::VecType>(a_vec)),
            BTraits::OpDataType,
            BTraits::to_wmma_vec(bit_cast<typename BTraits::VecType>(b_vec)),
            0,
            c_vec,
            P::op_sel_a, // OPSEL[0]
            0,           // OPSEL_HI[0]
            a_scale,
            P::op_sel_b, // OPSEL[1]
            0,           // OPSEL_HI[1]
            b_scale,
            0,  // NEG
            0); // NEG_HI
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = a_scale;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = b_scale;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

template <typename AType, typename BType>
struct WmmaTraits<gfx125_t, AType, BType, float, 32, 32, 128>
    : WmmaTraitsBase<gfx12_t, AType, BType, float, 128, true, 32, 32>
{
    using Base     = WmmaTraitsBase<gfx12_t, AType, BType, float, 128, true, 32, 32>;
    using ArchType = gfx125_t;

    using AVecType = typename Base::AVecType;
    using BVecType = typename Base::BVecType;
    using CVecType = typename Base::CVecType;

    using ATraits = MXDataTypeTrait<AType>;
    using BTraits = MXDataTypeTrait<BType>;

    using Base::kCMBlock;
    using Base::kCNBlock;

    template <typename... Params>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx125__
        constexpr index_t kASliceSize = sizeof(AVecType) / sizeof(AType) / kCMBlock;
        constexpr index_t kBSliceSize = sizeof(BVecType) / sizeof(BType) / kCNBlock;

        using ASliceType = ext_vector_t<AType, kASliceSize>;
        using BSliceType = ext_vector_t<BType, kBSliceSize>;
        using CSliceType = fp32x8_t;

        using a_buf = thread_buffer<ASliceType, kCMBlock>;
        using b_buf = thread_buffer<BSliceType, kCNBlock>;
        using c_buf = thread_buffer<CSliceType, kCMBlock * kCNBlock>;

        static_assert(sizeof(CVecType) == sizeof(c_buf),
                      "CVecType and c_buf must have the same size");
        static_assert(sizeof(AVecType) == sizeof(a_buf),
                      "AVecType and a_buf must have the same size");
        static_assert(sizeof(BVecType) == sizeof(b_buf),
                      "BVecType and b_buf must have the same size");

        auto&& a_buffer = bit_cast<a_buf>(a_vec);
        auto&& b_buffer = bit_cast<b_buf>(b_vec);
        auto&& c_result = bit_cast<c_buf>(c_vec);

        static_for<0, kCNBlock, 1>{}([&](auto n) {
            static_for<0, kCMBlock, 1>{}([&](auto m) {
                constexpr index_t c_idx = n * kCMBlock + m;

                const auto& a_slice = a_buffer.template get_as<ASliceType>()[m];
                const auto& b_slice = b_buffer.template get_as<BSliceType>()[n];
                auto& c_slice       = c_result.template get_as<CSliceType>()[number<c_idx>{}];

                c_slice = __builtin_amdgcn_wmma_f32_16x16x128_f8f6f4(
                    ATraits::OpDataType,
                    ATraits::to_wmma_vec(bit_cast<typename ATraits::VecType>(a_slice)),
                    BTraits::OpDataType,
                    BTraits::to_wmma_vec(bit_cast<typename BTraits::VecType>(b_slice)),
                    0,
                    c_slice);
            });
        });

        return bit_cast<CVecType>(c_result);

#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }

    template <typename... Params>
    CK_TILE_DEVICE static CVecType wmma_intrinsic(const AVecType& a_vec,
                                                  const int32_t& a_scale,
                                                  const BVecType& b_vec,
                                                  const int32_t& b_scale,
                                                  const CVecType& c_vec)
    {
#ifdef __gfx125__
        constexpr index_t kASliceSize = sizeof(AVecType) / sizeof(AType) / kCMBlock;
        constexpr index_t kBSliceSize = sizeof(BVecType) / sizeof(BType) / kCNBlock;

        using ASliceType = ext_vector_t<AType, kASliceSize>;
        using BSliceType = ext_vector_t<BType, kBSliceSize>;
        using CSliceType = fp32x8_t;

        using a_buf = thread_buffer<ASliceType, kCMBlock>;
        using b_buf = thread_buffer<BSliceType, kCNBlock>;
        using c_buf = thread_buffer<CSliceType, kCMBlock * kCNBlock>;

        static_assert(sizeof(CVecType) == sizeof(c_buf),
                      "CVecType and c_buf must have the same size");
        static_assert(sizeof(AVecType) == sizeof(a_buf),
                      "AVecType and a_buf must have the same size");
        static_assert(sizeof(BVecType) == sizeof(b_buf),
                      "BVecType and b_buf must have the same size");

        auto&& a_buffer = bit_cast<a_buf>(a_vec);
        auto&& b_buffer = bit_cast<b_buf>(b_vec);
        auto&& c_result = bit_cast<c_buf>(c_vec);

        using P = WarpGemmParamsParser<Params...>;

        static_for<0, kCNBlock, 1>{}([&](auto n) {
            static_for<0, kCMBlock, 1>{}([&](auto m) {
                constexpr index_t c_idx = n * kCMBlock + m;

                const auto& a_slice = a_buffer.template get_as<ASliceType>()[m];
                const auto& b_slice = b_buffer.template get_as<BSliceType>()[n];
                auto& c_slice       = c_result.template get_as<CSliceType>()[number<c_idx>{}];

                c_slice = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
                    ATraits::OpDataType,
                    ATraits::to_wmma_vec(bit_cast<typename ATraits::VecType>(a_slice)),
                    BTraits::OpDataType,
                    BTraits::to_wmma_vec(bit_cast<typename BTraits::VecType>(b_slice)),
                    0,
                    c_slice,
                    m.value,    // OPSEL[0]
                    P::scale_a, // OPSEL_HI[0]
                    a_scale,
                    n.value,    // OPSEL[1]
                    P::scale_b, // OPSEL_HI[1]
                    b_scale,
                    0,  // NEG
                    0); // NEG_HI
            });
        });

        return bit_cast<CVecType>(c_result);

#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = a_scale;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = b_scale;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

// 32x32x128 f8f6f4 scale16 specialization - GFX125
// Decomposes the 32x32 tile into a 2x2 grid of 16x16x128 scale16 intrinsic calls.
// a_scale/b_scale are per-lane int64_t registers (lane L already holds row L's K-scales); the
// same register value is passed to every sub-call, and SCALE_OPSEL = m.value / n.value selects
// which 16-lane group (lanes 0-15 vs 16-31) the hardware reads each sub-tile's scale from.
template <typename AType, typename BType>
struct WmmaTraits<gfx125_t, AType, BType, float, 32, 32, 128, WmmaScale16Tag>
    : WmmaTraitsBase<gfx12_t, AType, BType, float, 128, true, 32, 32>
{
    using Base             = WmmaTraitsBase<gfx12_t, AType, BType, float, 128, true, 32, 32>;
    using ArchType         = gfx125_t;
    using MXTypeEnableType = WmmaScale16Tag;

    using AVecType = typename Base::AVecType;
    using BVecType = typename Base::BVecType;
    using CVecType = typename Base::CVecType;

    using ATraits = MXDataTypeTrait<AType>;
    using BTraits = MXDataTypeTrait<BType>;

    using Base::kCMBlock;
    using Base::kCNBlock;

    // scale16 register layout -> sequence<4,2,16>
    static constexpr index_t kAK1PerLane = 16;
    static constexpr index_t kAK0PerLane = Base::kK / (kAK1PerLane * Base::kABKLane);
    static constexpr index_t kBK1PerLane = 16;
    static constexpr index_t kBK0PerLane = Base::kK / (kBK1PerLane * Base::kABKLane);

    template <typename... Params>
    CK_TILE_DEVICE static CVecType wmma_intrinsic(const AVecType&, const BVecType&, const CVecType&)
    {
        static_assert(sizeof...(Params) < 0,
                      "32x32x128 scale16 WmmaTraits requires int64_t scale arguments");
        return CVecType{0};
    }

    template <typename... Params>
    CK_TILE_DEVICE static CVecType wmma_intrinsic(const AVecType& a_vec,
                                                  const int64_t& a_scale,
                                                  const BVecType& b_vec,
                                                  const int64_t& b_scale,
                                                  const CVecType& c_vec)
    {
#ifdef __gfx125__
        constexpr index_t kASliceSize = sizeof(AVecType) / sizeof(AType) / kCMBlock;
        constexpr index_t kBSliceSize = sizeof(BVecType) / sizeof(BType) / kCNBlock;

        using ASliceType = ext_vector_t<AType, kASliceSize>;
        using BSliceType = ext_vector_t<BType, kBSliceSize>;
        using CSliceType = fp32x8_t;

        using a_buf = thread_buffer<ASliceType, kCMBlock>;
        using b_buf = thread_buffer<BSliceType, kCNBlock>;
        using c_buf = thread_buffer<CSliceType, kCMBlock * kCNBlock>;

        static_assert(sizeof(CVecType) == sizeof(c_buf),
                      "CVecType and c_buf must have the same size");
        static_assert(sizeof(AVecType) == sizeof(a_buf),
                      "AVecType and a_buf must have the same size");
        static_assert(sizeof(BVecType) == sizeof(b_buf),
                      "BVecType and b_buf must have the same size");

        auto&& a_buffer = bit_cast<a_buf>(a_vec);
        auto&& b_buffer = bit_cast<b_buf>(b_vec);
        auto&& c_result = bit_cast<c_buf>(c_vec);

        using P = WarpGemmParamsParser<Params...>;

        // SCALE_OPSEL selects which 16-lane half provides the scale:
        //   SCALE_OPSEL=0 -> lanes 0..15, SCALE_OPSEL=1 -> lanes 16..31
        // Sub-iteration m: A-scale from lane group m; sub-iteration n: B-scale from lane group n.
        static_for<0, kCNBlock, 1>{}([&](auto n) {
            static_for<0, kCMBlock, 1>{}([&](auto m) {
                constexpr index_t c_idx = n * kCMBlock + m;

                const auto& a_slice = a_buffer.template get_as<ASliceType>()[m];
                const auto& b_slice = b_buffer.template get_as<BSliceType>()[n];
                auto& c_slice       = c_result.template get_as<CSliceType>()[number<c_idx>{}];

                c_slice = __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(
                    ATraits::OpDataType,
                    ATraits::to_wmma_vec(bit_cast<typename ATraits::VecType>(a_slice)),
                    BTraits::OpDataType,
                    BTraits::to_wmma_vec(bit_cast<typename BTraits::VecType>(b_slice)),
                    0,
                    c_slice,
                    m.value,    // OPSEL[0]  - select A scale lane group
                    P::scale_a, // OPSEL_HI[0]
                    a_scale,
                    n.value,    // OPSEL[1]  - select B scale lane group
                    P::scale_b, // OPSEL_HI[1]
                    b_scale,
                    0,  // NEG
                    0); // NEG_HI
            });
        });

        return bit_cast<CVecType>(c_result);

#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = a_scale;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = b_scale;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};

} // namespace ck_tile
