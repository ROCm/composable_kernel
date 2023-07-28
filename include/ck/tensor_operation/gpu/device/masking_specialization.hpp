// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {

enum struct MaskingSpecialization
{
    MaskDisabled,
    MaskUpperTriangleFromTopLeft,
    MaskUpperTriangleFromBottomRight
};

inline std::string getMaskingSpecializationString(const MaskingSpecialization& s)
{
    switch(s)
    {
    case MaskingSpecialization::MaskDisabled: return "MaskDisabled";
    case MaskingSpecialization::MaskUpperTriangleFromTopLeft: return "MaskUpperTriangleFromTopLeft";
    case MaskingSpecialization::MaskUpperTriangleFromBottomRight:
        return "MaskUpperTriangleFromBottomRight";
    default: return "Unrecognized specialization!";
    }
}

struct MaskDisabledPredicate
{
    __host__ __device__ constexpr bool operator()(index_t /*m*/, index_t /*n*/) const
    {
        return false;
    };

    __host__ __device__ constexpr bool
        IsTileSkippable(index_t /*m*/, index_t /*n*/, index_t /*m_tile*/, index_t /*n_tile*/) const
    {
        return false;
    }
};

struct MaskUpperTriangleFromTopLeftPredicate
{
    __host__ __device__ constexpr bool operator()(index_t m, index_t n) const { return n > m; }

    __host__ __device__ constexpr bool
    IsTileSkippable(index_t m, index_t n, index_t m_tile, index_t /*n_tile*/) const
    {
        return operator()(m + m_tile - 1, n);
    }
};

// eg: m = 3, n = 5 => offset = 2
//    so matrix(n > m + offset) = 0
//      1  2  3  4  5
//    1 *  *  *  0  0
//    2 *  *  *  *  0
//    3 *  *  *  *  *
struct MaskUpperTriangleFromBottomRightPredicate
{
    MaskUpperTriangleFromBottomRightPredicate() : diagonal_offset_(0) {}
    __host__ __device__ void SetDiagonalOffset(const index_t diagonal_offset)
    {
        diagonal_offset_ = diagonal_offset;
    }
    __host__ __device__ constexpr bool operator()(index_t m, index_t n) const
    {
        return n > (m + diagonal_offset_);
    }

    __host__ __device__ constexpr bool IsTileSkippable(index_t m_tile_orig,
                                                       index_t n_tile_orig,
                                                       index_t m_tile_size,
                                                       index_t /*n_tile_size*/) const
    {
        return operator()(m_tile_orig + m_tile_size - 1, n_tile_orig);
    }

    private:
    index_t diagonal_offset_;
};

// to track the points which need to be set to -inf on C0
// Note: no need to reset M padding value, because they will not be stored out.
template <typename MaskOutPredicate>
struct C0MatrixMask_impl
{
    C0MatrixMask_impl(index_t MRaw, index_t NRaw) : NRaw_(NRaw), predicate_(MaskOutPredicate{})
    {
        if constexpr(std::is_same<MaskOutPredicate,
                                  MaskUpperTriangleFromBottomRightPredicate>::value)
        {
            if(NRaw > MRaw)
                predicate_.SetDiagonalOffset(NRaw - MRaw);
        }
    }

    __host__ __device__ constexpr bool IsNOutOfBound(/*index_t m, */ index_t n) const
    {
        return n >= NRaw_;
    }

    __host__ __device__ constexpr bool IsMaskedElement(index_t m, index_t n) const
    {
        return predicate_(m, n) || IsNOutOfBound(n);
    }

    __host__ __device__ constexpr bool
    IsTileSkippable(index_t m, index_t n, index_t m_tile, index_t n_tile) const
    {
        return predicate_.IsTileSkippable(m, n, m_tile, n_tile);
    }

    private:
    // index_t MRaw_;
    index_t NRaw_;
    MaskOutPredicate predicate_;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
