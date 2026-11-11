// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_config.hpp"

namespace ck_tile {

// Geometries the hand-written FP8 pieces are implemented for; kept apart from the 16-bit list.
template <typename Geometry>
struct FmhaTdmSchedFp8SupportedGeometry
{
    using Q = typename Geometry::QDataType;
    static_assert(std::is_same_v<Q, fp8_t> && std::is_same_v<typename Geometry::KDataType, Q> &&
                      std::is_same_v<typename Geometry::VDataType, Q> &&
                      std::is_same_v<typename Geometry::PDataType, Q>,
                  "qr_tdm_fp8_sched requires homogeneous FP8 Q/K/V/P");
    static_assert(
        Geometry::kM == 128 && Geometry::kN == 128 &&
            (((Geometry::kHeadDimQK == 64 || Geometry::kHeadDimQK == 128 ||
               Geometry::kHeadDimQK == 256) &&
              Geometry::kDv == Geometry::kHeadDimQK) ||
             (Geometry::kHeadDimQK == 192 && Geometry::kDv == 128)),
        "qr_tdm_fp8_sched supports D64/V64, D128/V128, D192/V128 and D256/V256 with N128");
    static_assert(Geometry::kWaves == 4 && Geometry::kWaveSize == 32,
                  "qr_tdm_fp8_sched requires four wave32 waves");
    static_assert(Geometry::kQkWmmaK == (Geometry::kHeadDimQK % 128 != 0 ? 64 : 128) &&
                      Geometry::kPvWmmaK == 128,
                  "qr_tdm_fp8_sched needs a K64 QK WMMA at D64/D192, else K128, and K128 PV WMMA");
    static_assert(Geometry::kQkStages == Geometry::kN / 32 &&
                      Geometry::kPvStages == Geometry::kN / 128,
                  "qr_tdm_fp8_sched requires 32-row QK stages and 128-row PV stages");
    static_assert(Geometry::PRepackLayout::kInThread,
                  "gemm_0 C^T and gemm_1 A lane layouts differ; P cannot be re-packed in-thread");
    static constexpr bool value = true;
};

// One wavefront per workgroup with 32 query rows; the per-wavefront code is the four-wave one.
template <typename Geometry>
struct FmhaTdmSchedFp8OneWaveGeometry
{
    using Q = typename Geometry::QDataType;
    static_assert(std::is_same_v<Q, fp8_t> && std::is_same_v<typename Geometry::KDataType, Q> &&
                      std::is_same_v<typename Geometry::VDataType, Q> &&
                      std::is_same_v<typename Geometry::PDataType, Q>,
                  "qr_tdm_fp8_sched requires homogeneous FP8 Q/K/V/P");
    static_assert(Geometry::kM == 32 && Geometry::kWaves == 1 && Geometry::kWaveSize == 32 &&
                      Geometry::kN == 128 &&
                      (((Geometry::kHeadDimQK == 64 || Geometry::kHeadDimQK == 128 ||
                         Geometry::kHeadDimQK == 256) &&
                        Geometry::kDv == Geometry::kHeadDimQK) ||
                       (Geometry::kHeadDimQK == 192 && Geometry::kDv == 128)),
                  "one-wavefront qr_tdm_fp8_sched supports M32 D64/D128/D192/D256 with N128");
    static_assert(Geometry::kQkWmmaK == (Geometry::kHeadDimQK % 128 != 0 ? 64 : 128) &&
                      Geometry::kPvWmmaK == 128 && Geometry::kQkStages == Geometry::kN / 32 &&
                      Geometry::kPvStages == Geometry::kN / 128,
                  "one-wavefront qr_tdm_fp8_sched keeps the four-wavefront WMMA stages");
    static_assert(Geometry::PRepackLayout::kInThread,
                  "gemm_0 C^T and gemm_1 A lane layouts differ; P cannot be re-packed in-thread");
    static constexpr bool value = true;
};

template <typename Geometry>
inline constexpr bool kFmhaTdmSchedFp8GeometryOk =
    std::conditional_t<Geometry::kWaves == 1,
                       FmhaTdmSchedFp8OneWaveGeometry<Geometry>,
                       FmhaTdmSchedFp8SupportedGeometry<Geometry>>::value;

// TDM pads each 64/128/256 B K/V row by 16 B so the 16 rows of one 8 B LDS read hit distinct banks.
template <typename Geometry>
struct FmhaTdmSchedFp8Layout
{
    static_assert(kFmhaTdmSchedFp8GeometryOk<Geometry>);
    static constexpr index_t kRowPadBytes     = 16;
    static constexpr index_t kKPhysicalStride = Geometry::kHeadDimQK + kRowPadBytes;
    // TDM pads only at power-of-two intervals; D192 K boxes span the pad instead (zero filled).
    static constexpr bool kKPadByTdm = (Geometry::kHeadDimQK & (Geometry::kHeadDimQK - 1)) == 0;
    static constexpr index_t kVPhysicalStride = Geometry::kDv + kRowPadBytes;
};

// The low 32 bits of a generic LDS pointer are its LDS offset; no null check.
CK_TILE_DEVICE uint32_t fmha_tdm_sched_fp8_lds_offset(const void* p)
{
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p));
}

template <typename T>
CK_TILE_DEVICE __attribute__((address_space(3))) T* fmha_tdm_sched_fp8_lds_ptr(uint32_t offset)
{
    return reinterpret_cast<__attribute__((address_space(3))) T*>(static_cast<uintptr_t>(offset));
}

// A full DS counter field means no wait; clamp a tail count to one that still emits a real wait.
template <index_t Count>
inline constexpr index_t kFmhaTdmSchedFp8DsWait =
    Count < waitcnt_arg::kMaxLgkmCnt ? Count : waitcnt_arg::kMaxLgkmCnt - 1;

} // namespace ck_tile
