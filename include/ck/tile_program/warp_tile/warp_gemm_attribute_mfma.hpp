// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/warp_tile/warp_gemm_attribute_mfma_impl.hpp"

namespace ck {
namespace tile_program {
namespace warp {

template <typename WarpGemmAttributeMfmaImpl_>
struct WarpGemmAtrributeMfma
{
    using Impl = remove_cvref_t<WarpGemmAttributeMfmaImpl_>;

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    using AVecType = typename Impl::AVecType;
    using BVecType = typename Impl::BVecType;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t kM = Impl::kM;
    static constexpr index_t kN = Impl::kN;
    static constexpr index_t kK = Impl::kK;

    using AWarpDstrEncoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<Impl::kAMLane>, Sequence<Impl::kABKLane, Impl::kABKPerLane>>,
        Tuple<Sequence<2, 1>>,
        Tuple<Sequence<0, 0>>,
        Sequence<2>,
        Sequence<1>>;

    using BWarpDstrEncoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<Impl::kBNLane>, Sequence<Impl::kABKLane, Impl::kABKPerLane>>,
        Tuple<Sequence<2, 1>>,
        Tuple<Sequence<0, 0>>,
        Sequence<2>,
        Sequence<1>>;

    using CWarpDstrEncoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>,
              Sequence<Impl::kCNLane>>,
        Tuple<Sequence<1, 2>>,
        Tuple<Sequence<1, 0>>,
        Sequence<1, 1>,
        Sequence<0, 2>>;

    // c_vec += a_vec * b_vec
    __device__ void operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        Impl{}(c_vec, a_vec, b_vec);
    }

    // c_vec = a_vec * b_vec
    __device__ CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        return Impl{}(a_vec, b_vec);
    }
};

template <typename WarpGemmAttributeMfmaImpl_, index_t kKIter>
struct WarpGemmAtrributeMfmaIterateK
{
    static_assert(kKIter > 0, "wrong!");

    using Impl = remove_cvref_t<WarpGemmAttributeMfmaImpl_>;

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    using AVecType = typename vector_type_maker<typename Impl::AVecType, kKIter>::type::type;
    using BVecType = typename vector_type_maker<typename Impl::BVecType, kKIter>::type::type;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t kM = Impl::kM;
    static constexpr index_t kN = Impl::kN;
    static constexpr index_t kK = Impl::kK * kKIter;

    using AWarpDstrEncoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<Impl::kAMLane>, Sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
        Tuple<Sequence<2, 1>>,
        Tuple<Sequence<0, 0>>,
        Sequence<2>,
        Sequence<1>>;

    using BWarpDstrEncoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<Impl::kBNLane>, Sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
        Tuple<Sequence<2, 1>>,
        Tuple<Sequence<0, 0>>,
        Sequence<2>,
        Sequence<1>>;

    using CWarpDstrEncoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>,
              Sequence<Impl::kCNLane>>,
        Tuple<Sequence<1, 2>>,
        Tuple<Sequence<1, 0>>,
        Sequence<1, 1>,
        Sequence<0, 2>>;

    // c_vec += a_vec * b_vec
    __device__ void operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        const auto a_vector = typename vector_type_maker<AVecType, 1>::type{a_vec};
        const auto b_vector = typename vector_type_maker<BVecType, 1>::type{b_vec};

        static_for<0, kKIter, 1>{}([&](auto iKIter) {
            Impl{}(c_vec,
                   a_vector.template AsType<typename Impl::AVecType>()[iKIter],
                   b_vector.template AsType<typename Impl::BVecType>()[iKIter]);
        });
    }

    // c_vec = a_vec * b_vec
    __device__ CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        const auto a_vector = typename vector_type_maker<AVecType, 1>::type{a_vec};
        const auto b_vector = typename vector_type_maker<BVecType, 1>::type{b_vec};

        constexpr auto I0 = Number<0>{};

        // c = a * b
        auto c_vec = Impl{}(a_vector.template AsType<typename Impl::AVecType>()[I0],
                            b_vector.template AsType<typename Impl::BVecType>()[I0]);

        // c += a * b
        static_for<1, kKIter, 1>{}([&](auto iKIter) {
            Impl{}(c_vec,
                   a_vector.template AsType<typename Impl::AVecType>()[iKIter],
                   b_vector.template AsType<typename Impl::BVecType>()[iKIter]);
        });

        return c_vec;
    }
};

template <typename WarpGemmAttributeMfmaImpl_>
struct WarpGemmAtrributeMfmaTransposedCDistribution
{
    using Impl = remove_cvref_t<WarpGemmAttributeMfmaImpl_>;

    using ADataType = typename Impl::BDataType;
    using BDataType = typename Impl::ADataType;
    using CDataType = typename Impl::CDataType;

    using AVecType = typename Impl::BVecType;
    using BVecType = typename Impl::AVecType;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t kM = Impl::kN;
    static constexpr index_t kN = Impl::kM;
    static constexpr index_t kK = Impl::kK;

    using AWarpDstrEncoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<Impl::kBNLane>, Sequence<Impl::kABKLane, Impl::kABKPerLane>>,
        Tuple<Sequence<2, 1>>,
        Tuple<Sequence<0, 0>>,
        Sequence<2>,
        Sequence<1>>;

    using BWarpDstrEncoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<Impl::kAMLane>, Sequence<Impl::kABKLane, Impl::kABKPerLane>>,
        Tuple<Sequence<2, 1>>,
        Tuple<Sequence<0, 0>>,
        Sequence<2>,
        Sequence<1>>;

    using CWarpDstrEncoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<Impl::kCNLane>,
              Sequence<Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>>,
        Tuple<Sequence<2, 1>>,
        Tuple<Sequence<1, 0>>,
        Sequence<2, 2>,
        Sequence<0, 2>>;

    // c_vec += a_vec * b_vec
    __device__ void operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        // swap A and B
        Impl{}(c_vec, b_vec, a_vec);
    }

    // c_vec = a_vec * b_vec
    __device__ CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        // swap A and B
        return Impl{}(b_vec, a_vec);
    }
};

template <typename WarpGemmAttributeMfmaImpl_>
struct WarpGemmAtrributeMfmaTransposedCDistribution_SwizzleB
{
    using Impl = remove_cvref_t<WarpGemmAttributeMfmaImpl_>;

    using ADataType = typename Impl::BDataType;
    using BDataType = typename Impl::ADataType;
    using CDataType = typename Impl::CDataType;

    using AVecType = typename Impl::BVecType;
    using BVecType = typename Impl::AVecType;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t kM = Impl::kN;
    static constexpr index_t kN = Impl::kM;
    static constexpr index_t kK = Impl::kK;

    using AWarpDstrEncoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<Impl::kBNLane>, Sequence<Impl::kABKLane, Impl::kABKPerLane>>,
        Tuple<Sequence<2, 1>>,
        Tuple<Sequence<0, 0>>,
        Sequence<2>,
        Sequence<1>>;

    using BWarpDstrEncoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<Impl::kAMLane / (Impl::kABKPerLane * Impl::kABKLane * 2),
                       Impl::kABKLane,
                       2,
                       Impl::kABKPerLane>,
              Sequence<Impl::kABKLane, Impl::kABKPerLane>>,
        Tuple<Sequence<2, 1, 1, 1, 1>>,
        Tuple<Sequence<0, 0, 2, 1, 3>>,
        Sequence<2>,
        Sequence<1>>;

    using CWarpDstrEncoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<Impl::kCNLane>,
              Sequence<Impl::kCM0PerLane / 2, Impl::kCMLane, Impl::kCM1PerLane * 2>>,
        Tuple<Sequence<2, 1>>,
        Tuple<Sequence<1, 0>>,
        Sequence<2, 2>,
        Sequence<0, 2>>;

    // c_vec += a_vec * b_vec
    __device__ void operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        // swap A and B
        Impl{}(c_vec, b_vec, a_vec);
    }

    // c_vec = a_vec * b_vec
    __device__ CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        // swap A and B
        return Impl{}(b_vec, a_vec);
    }
};

template <typename WarpGemmAttributeMfmaImpl_, index_t kKIter>
struct WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution
{
    using Impl = remove_cvref_t<WarpGemmAttributeMfmaImpl_>;

    // swap A and B
    using ADataType = typename Impl::BDataType;
    using BDataType = typename Impl::ADataType;
    using CDataType = typename Impl::CDataType;

    using AVecType = typename vector_type_maker<typename Impl::BVecType, kKIter>::type::type;
    using BVecType = typename vector_type_maker<typename Impl::AVecType, kKIter>::type::type;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t kM = Impl::kN;
    static constexpr index_t kN = Impl::kM;
    static constexpr index_t kK = Impl::kK * kKIter;

    using AWarpDstrEncoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<Impl::kBNLane>, Sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
        Tuple<Sequence<2, 1>>,
        Tuple<Sequence<0, 0>>,
        Sequence<2>,
        Sequence<1>>;

    using BWarpDstrEncoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<Impl::kAMLane>, Sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
        Tuple<Sequence<2, 1>>,
        Tuple<Sequence<0, 0>>,
        Sequence<2>,
        Sequence<1>>;

    using CWarpDstrEncoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<Impl::kCNLane>,
              Sequence<Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>>,
        Tuple<Sequence<2, 1>>,
        Tuple<Sequence<1, 0>>,
        Sequence<2, 2>,
        Sequence<0, 2>>;

    // c_vec += a_vec * b_vec
    __device__ void operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        const auto a_vector = typename vector_type_maker<AVecType, 1>::type{a_vec};

        const auto b_vector = typename vector_type_maker<BVecType, 1>::type{b_vec};

        // swap A and B, value and type
        static_for<0, kKIter, 1>{}([&](auto iKIter) {
            Impl{}(c_vec,
                   b_vector.template AsType<typename Impl::AVecType>()[iKIter],
                   a_vector.template AsType<typename Impl::BVecType>()[iKIter]);
        });
    }

    // c_vec = a_vec * b_vec
    __device__ CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        const auto a_vector = typename vector_type_maker<AVecType, 1>::type{a_vec};
        const auto b_vector = typename vector_type_maker<BVecType, 1>::type{b_vec};

        constexpr auto I0 = Number<0>{};

        // swap A and B, value and type
        auto c_vec = Impl{}(b_vector.template AsType<typename Impl::AVecType>()[I0],
                            a_vector.template AsType<typename Impl::BVecType>()[I0]);

        static_for<1, kKIter, 1>{}([&](auto iKIter) {
            Impl{}(c_vec,
                   b_vector.template AsType<typename Impl::AVecType>()[iKIter],
                   a_vector.template AsType<typename Impl::BVecType>()[iKIter]);
        });

        return c_vec;
    }
};

template <typename WarpGemmAttributeMfmaImpl_, index_t kKIter, index_t SFactor_ = 2>
struct WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution_SwizzleB
{
    using Impl = remove_cvref_t<WarpGemmAttributeMfmaImpl_>;

    // swap A and B
    using ADataType = typename Impl::BDataType;
    using BDataType = typename Impl::ADataType;
    using CDataType = typename Impl::CDataType;

    using AVecType = typename vector_type_maker<typename Impl::BVecType, kKIter>::type::type;
    using BVecType = typename vector_type_maker<typename Impl::AVecType, kKIter>::type::type;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t kM      = Impl::kN;
    static constexpr index_t kN      = Impl::kM;
    static constexpr index_t kK      = Impl::kK * kKIter;
    static constexpr index_t SFactor = SFactor_; // group how many CM1 together

    using AWarpDstrEncoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<Impl::kBNLane>, Sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
        Tuple<Sequence<2, 1>>,
        Tuple<Sequence<0, 0>>,
        Sequence<2>,
        Sequence<1>>;
#if 0
    using BWarpDstrEncoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<Impl::kAMLane / (Impl::kABKPerLane * Impl::kABKLane * 2),
                       Impl::kABKLane,
                       2,
                       Impl::kABKPerLane>,
              Sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
        Tuple<Sequence<2, 1, 1, 1, 1>>,
        Tuple<Sequence<0, 0, 2, 1, 3>>,
        Sequence<2>,
        Sequence<1>>;

    using CWarpDstrEncoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<Impl::kCNLane>,
              Sequence<Impl::kCM0PerLane / 2, Impl::kCMLane, Impl::kCM1PerLane * 2>>,
        Tuple<Sequence<2, 1>>,
        Tuple<Sequence<1, 0>>,
        Sequence<2, 2>,
        Sequence<0, 2>>;
#else
    using BWarpDstrEncoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<Impl::kAMLane / (SFactor * Impl::kCMLane * Impl::kCM1PerLane),
                       SFactor,
                       Impl::kCMLane,
                       Impl::kCM1PerLane>,
              Sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
        Tuple<Sequence<2, 1, 1, 1, 1>>,
        Tuple<Sequence<0, 0, 2, 1, 3>>,
        Sequence<2>,
        Sequence<1>>;

    using CWarpDstrEncoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<Impl::kCNLane>,
              Sequence<Impl::kCM0PerLane / SFactor, Impl::kCMLane, Impl::kCM1PerLane * SFactor>>,
        Tuple<Sequence<2, 1>>,
        Tuple<Sequence<1, 0>>,
        Sequence<2, 2>,
        Sequence<0, 2>>;
#endif
    // c_vec += a_vec * b_vec
    __device__ void operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        const auto a_vector = typename vector_type_maker<AVecType, 1>::type{a_vec};

        const auto b_vector = typename vector_type_maker<BVecType, 1>::type{b_vec};

        // swap A and B, value and type
        static_for<0, kKIter, 1>{}([&](auto iKIter) {
            Impl{}(c_vec,
                   b_vector.template AsType<typename Impl::AVecType>()[iKIter],
                   a_vector.template AsType<typename Impl::BVecType>()[iKIter]);
        });
    }

    // c_vec = a_vec * b_vec
    __device__ CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        const auto a_vector = typename vector_type_maker<AVecType, 1>::type{a_vec};
        const auto b_vector = typename vector_type_maker<BVecType, 1>::type{b_vec};

        constexpr auto I0 = Number<0>{};

        // swap A and B, value and type
        auto c_vec = Impl{}(b_vector.template AsType<typename Impl::AVecType>()[I0],
                            a_vector.template AsType<typename Impl::BVecType>()[I0]);

        static_for<1, kKIter, 1>{}([&](auto iKIter) {
            Impl{}(c_vec,
                   b_vector.template AsType<typename Impl::AVecType>()[iKIter],
                   a_vector.template AsType<typename Impl::BVecType>()[iKIter]);
        });

        return c_vec;
    }
};

template <typename WarpGemmAttributeMfmaImpl_, index_t kKIter>
struct WarpGemmAtrributeMfmaIterateK_SwizzleA
{
    using Impl = remove_cvref_t<WarpGemmAttributeMfmaImpl_>;

    using ADataType = typename Impl::ADataType;
    using BDataType = typename Impl::BDataType;
    using CDataType = typename Impl::CDataType;

    using AVecType = typename vector_type_maker<typename Impl::AVecType, kKIter>::type::type;
    using BVecType = typename vector_type_maker<typename Impl::BVecType, kKIter>::type::type;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t kM = Impl::kN;
    static constexpr index_t kN = Impl::kM;
    static constexpr index_t kK = Impl::kK * kKIter;

    using AWarpDstrEncoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<Impl::kAMLane / (Impl::kABKPerLane * Impl::kABKLane * 2),
                       Impl::kABKLane,
                       2,
                       Impl::kABKPerLane>,
              Sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
        Tuple<Sequence<2, 1, 1, 1, 1>>,
        Tuple<Sequence<0, 0, 2, 1, 3>>,
        Sequence<2>,
        Sequence<1>>;

    using BWarpDstrEncoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<Impl::kBNLane>, Sequence<Impl::kABKLane, Impl::kABKPerLane * kKIter>>,
        Tuple<Sequence<2, 1>>,
        Tuple<Sequence<0, 0>>,
        Sequence<2>,
        Sequence<1>>;

    using CWarpDstrEncoding = StaticTileDistributionEncoding<
        Sequence<>,
        Tuple<Sequence<Impl::kCM0PerLane / 2, Impl::kCMLane, Impl::kCM1PerLane * 2>,
              Sequence<Impl::kCNLane>>,
        Tuple<Sequence<1, 2>>,
        Tuple<Sequence<1, 0>>,
        Sequence<1, 1>,
        Sequence<0, 2>>;

    // c_vec += a_vec * b_vec
    __device__ void operator()(CVecType& c_vec, const AVecType& a_vec, const BVecType& b_vec) const
    {
        const auto a_vector = typename vector_type_maker<AVecType, 1>::type{a_vec};
        const auto b_vector = typename vector_type_maker<BVecType, 1>::type{b_vec};

        static_for<0, kKIter, 1>{}([&](auto iKIter) {
            Impl{}(c_vec,
                   a_vector.template AsType<typename Impl::AVecType>()[iKIter],
                   b_vector.template AsType<typename Impl::BVecType>()[iKIter]);
        });
    }

    // c_vec = a_vec * b_vec
    __device__ CVecType operator()(const AVecType& a_vec, const BVecType& b_vec) const
    {
        const auto a_vector = typename vector_type_maker<AVecType, 1>::type{a_vec};
        const auto b_vector = typename vector_type_maker<BVecType, 1>::type{b_vec};

        constexpr auto I0 = Number<0>{};

        auto c_vec = Impl{}(a_vector.template AsType<typename Impl::AVecType>()[I0],
                            b_vector.template AsType<typename Impl::BVecType>()[I0]);

        static_for<1, kKIter, 1>{}([&](auto iKIter) {
            Impl{}(c_vec,
                   a_vector.template AsType<typename Impl::AVecType>()[iKIter],
                   b_vector.template AsType<typename Impl::BVecType>()[iKIter]);
        });

        return c_vec;
    }
};

} // namespace warp
} // namespace tile_program
} // namespace ck
