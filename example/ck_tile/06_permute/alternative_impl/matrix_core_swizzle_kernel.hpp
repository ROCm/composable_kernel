// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"

enum class matrix_core_inst_enum
{
    MFMA_32x32x8_F16  = 0,
    MFMA_16x16x16_F16 = 1,
};

namespace detail {
template <matrix_core_inst_enum>
struct to_warp_gemm;

template <>
struct to_warp_gemm<matrix_core_inst_enum::MFMA_32x32x8_F16>
{
    using type = ck_tile::WarpGemmMfmaF16F16F32M32N32K8;
};

template <>
struct to_warp_gemm<matrix_core_inst_enum::MFMA_16x16x16_F16>
{
    using type = ck_tile::WarpGemmMfmaF16F16F32M16N16K16;
};
} // namespace detail
template <matrix_core_inst_enum Inst>
using to_warp_gemm_t = typename detail::to_warp_gemm<Inst>::type;

enum class matrix_core_permute_style
{
    permute_b_n0_k0_n1_k1_n2_k2 = 0, // 0,1,4,2,5,3,6
    permute_b_n0_n1_k0_k1_n2_k2 = 1, // 0,1,2,4,5,3,6
};

// assume this is B matrix, originally we have batch*n*k
// now batch* n0*n1*n2*k0*k1*k2 -> batch* n0*k0*n1*k1*n2*k2
// assume using 32x32x8-f16, 4 waves and extend the KPerLane to 8xfp16(dwordx4)
//
//                                      4(waves)  32(mfma_m lane)
//                                          |      |
// batch* n0*n1*n2*k0*k1*k2 -> batch* n0*k0*n1*k1*n2*k2 -> 8(thread loading)
//                                    nr  kr    |
//        nr  4  32 kr 2  8                     2(klane)
//
// permute: 0,1,4,2,5,3,6
// or
// batch* n0*n1*n2*k0*k1*k2 -> batch* n0*n1*k0*k1*n2*k2 -> 8(thread loading)
// permute: 0,1,2,4,5,3,6
//
// this kernel only deal with fp16/bf16 data(16bit), and use 2d block size to do the swizzling
// for simplicity, only consider n/k is multiple of block-size

// independend host arg with no template
struct matrix_core_swizzle_host_args
{
    const void* p_src;
    void* p_dst;
    int32_t batch;
    int32_t n;
    int32_t k;
};

// NOTE: this kernel could follow the style of generic permute kernel
// but here we pass in fixed layout as template arg and generate different kernel instance
// purposely
template <int BLOCK_SIZE_ = 256,
          int NPerBlock_  = 256,
          int KPerBlock_  = 128,
          matrix_core_permute_style pstyle_ =
              matrix_core_permute_style::permute_b_n0_k0_n1_k1_n2_k2,
          matrix_core_inst_enum Inst_ = matrix_core_inst_enum::MFMA_32x32x8_F16>
struct matrix_core_swizzle_kernel
{
    using karg = matrix_core_swizzle_host_args;
    using harg = matrix_core_swizzle_host_args;

    static constexpr int BLOCK_SIZE                   = BLOCK_SIZE_;
    static constexpr int NPerBlock                    = NPerBlock_;
    static constexpr int KPerBlock                    = KPerBlock_;
    static constexpr matrix_core_permute_style pstyle = pstyle_;
    static constexpr matrix_core_inst_enum Inst       = Inst_;

    static constexpr ck_tile::index_t Alignment = 8;
    karg a;
    dim3 grids;

    using WarpGemm = to_warp_gemm_t<Inst>;

    __host__ matrix_core_swizzle_kernel(harg h)
    {
        a                   = h;
        ck_tile::index_t ns = (h.n + NPerBlock - 1) / NPerBlock;
        ck_tile::index_t ks = (h.k + KPerBlock - 1) / KPerBlock;
        grids               = dim3(ks, ns, h.batch);
    }

    __host__ bool is_applicable(harg h) { return h.n % NPerBlock == 0 && h.k % KPerBlock == 0; }

    __host__ void operator()(const ck_tile::stream_config& s) const
    {
        ck_tile::kentry<BLOCK_SIZE, 1, kernel><<<grids, BLOCK_SIZE, 0, s.stream_id_>>>(a);
    }

    struct kernel
    {
        __device__ static constexpr auto get_src_dist()
        {
            using namespace ck_tile;
            constexpr index_t K2 = Alignment;
            constexpr index_t N2 = WarpGemm::WarpGemmAttribute::Impl::kAMLane;
            constexpr index_t K1 = WarpGemm::WarpGemmAttribute::Impl::kABKLane;
            constexpr index_t N1 = BLOCK_SIZE / get_warp_size();

            static_assert(NPerBlock % (N1 * N2) == 0);
            static_assert(KPerBlock % (K1 * K2) == 0);

            constexpr index_t K0 = KPerBlock / (K1 * K2);
            constexpr index_t N0 = NPerBlock / (N1 * N2);

            // clang-format off
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<1>,// 0
                    //             1              2            3             4             5             6
                    tuple<sequence<N0>, sequence<N1>, sequence<N2>, sequence<K0>, sequence<K1>, sequence<K2>>,

                    //            N1           K1  N2
                    tuple<sequence<2>, sequence<5, 3>>,
                    tuple<sequence<0>, sequence<0, 0>>,

                    //       N0 K0 K2
                    sequence<1, 4, 6>,
                    sequence<0, 0, 0>>{});
            // clang-format on
        }
        __device__ static constexpr auto get_dst_dist()
        {
            using namespace ck_tile;
            constexpr index_t K2 = Alignment;
            constexpr index_t N2 = WarpGemm::WarpGemmAttribute::Impl::kAMLane;
            constexpr index_t K1 = WarpGemm::WarpGemmAttribute::Impl::kABKLane;
            constexpr index_t N1 = BLOCK_SIZE / get_warp_size();

            static_assert(NPerBlock % (N1 * N2) == 0);
            static_assert(KPerBlock % (K1 * K2) == 0);

            constexpr index_t K0 = KPerBlock / (K1 * K2);
            constexpr index_t N0 = NPerBlock / (N1 * N2);

            if constexpr(pstyle == matrix_core_permute_style::permute_b_n0_k0_n1_k1_n2_k2)
            {
                // clang-format off
                return make_static_tile_distribution(
                    tile_distribution_encoding<
                        sequence<1>,// 0
                        //             1              2            3             4             5             6
                        tuple<sequence<N0>, sequence<K0>, sequence<N1>, sequence<K1>, sequence<N2>, sequence<K2>>,

                        //            N1           K1  N2
                        tuple<sequence<3>, sequence<4, 5>>,
                        tuple<sequence<0>, sequence<0, 0>>,

                        //       N0 K0 K2
                        sequence<1, 2, 6>,
                        sequence<0, 0, 0>>{});
                // clang-format on
            }
            else
            {
                // clang-format off
                return make_static_tile_distribution(
                    tile_distribution_encoding<
                        sequence<1>,// 0
                        //             1              2            3             4             5             6
                        tuple<sequence<N0>, sequence<N1>, sequence<K0>, sequence<K1>, sequence<N2>, sequence<K2>>,

                        //            N1           K1  N2
                        tuple<sequence<2>, sequence<4, 5>>,
                        tuple<sequence<0>, sequence<0, 0>>,

                        //       N0 K0 K2
                        sequence<1, 3, 6>,
                        sequence<0, 0, 0>>{});
                // clang-format on
            }
        }

        __device__ void operator()(karg a_)
        {
            using namespace ck_tile;
            index_t i_k = blockIdx.x;
            index_t i_n = blockIdx.y;
            index_t i_b = blockIdx.z;

            constexpr index_t k2 = Alignment;
            constexpr index_t n2 = WarpGemm::WarpGemmAttribute::Impl::kAMLane;
            constexpr index_t k1 = WarpGemm::WarpGemmAttribute::Impl::kABKLane;
            constexpr index_t n1 = BLOCK_SIZE / get_warp_size();
            const index_t k0     = a_.k / (k1 * k2);
            const index_t n0     = a_.n / (n1 * n2);

            constexpr index_t k2_tile = Alignment;
            constexpr index_t n2_tile = WarpGemm::WarpGemmAttribute::Impl::kAMLane;
            constexpr index_t k1_tile = WarpGemm::WarpGemmAttribute::Impl::kABKLane;
            constexpr index_t n1_tile = BLOCK_SIZE / get_warp_size();
            constexpr index_t k0_tile = KPerBlock / (k1_tile * k2_tile);
            constexpr index_t n0_tile = NPerBlock / (n1_tile * n2_tile);

            const fp16_t* p_src = reinterpret_cast<const fp16_t*>(a_.p_src) + i_b * a_.k * a_.n;
            fp16_t* p_dst       = reinterpret_cast<fp16_t*>(a_.p_dst) + i_b * a_.k * a_.n;

            const auto src_view = [&]() {
                const auto tmp = make_naive_tensor_view_packed<address_space_enum::global>(
                    p_src,
                    make_tuple(n0, n1, n2, k0, k1, k2),
                    number<Alignment>{}); // control vector load
                return tmp;
            }();

            const auto src_window = make_tile_window(src_view,
                                                     make_tuple(number<n0_tile>{},
                                                                number<n1_tile>{},
                                                                number<n2_tile>{},
                                                                number<k0_tile>{},
                                                                number<k1_tile>{},
                                                                number<k2_tile>{}),
                                                     {i_n * n0_tile, 0, 0, i_k * k0_tile, 0, 0},
                                                     get_src_dist());

            auto dst_view = [&]() {
                if constexpr(pstyle == matrix_core_permute_style::permute_b_n0_k0_n1_k1_n2_k2)
                {
                    auto tmp = make_naive_tensor_view_packed<address_space_enum::global>(
                        p_dst,
                        make_tuple(n0, k0, n1, k1, n2, k2),
                        number<Alignment>{}); // control vector load
                    return tmp;
                }
                else
                {
                    auto tmp = make_naive_tensor_view_packed<address_space_enum::global>(
                        p_dst,
                        make_tuple(n0, n1, k0, k1, n2, k2),
                        number<Alignment>{}); // control vector load
                    return tmp;
                }
            }();

            auto dst_window = [&]() {
                if constexpr(pstyle == matrix_core_permute_style::permute_b_n0_k0_n1_k1_n2_k2)
                {
                    return make_tile_window(dst_view,
                                            make_tuple(number<n0_tile>{},
                                                       number<k0_tile>{},
                                                       number<n1_tile>{},
                                                       number<k1_tile>{},
                                                       number<n2_tile>{},
                                                       number<k2_tile>{}),
                                            {i_n * n0_tile, i_k * k0_tile, 0, 0, 0, 0},
                                            get_dst_dist());
                }
                else
                {
                    return make_tile_window(dst_view,
                                            make_tuple(number<n0_tile>{},
                                                       number<n1_tile>{},
                                                       number<k0_tile>{},
                                                       number<k1_tile>{},
                                                       number<n2_tile>{},
                                                       number<k2_tile>{}),
                                            {i_n * n0_tile, 0, i_k * k0_tile, 0, 0, 0},
                                            get_dst_dist());
                }
            }();

            // actual load store
            auto src_tile = load_tile(src_window);

            // now we only swap the distribution from src to dst, no extra movement occurs
            auto dst_tile                = make_static_distributed_tensor<fp16_t>(get_dst_dist());
            dst_tile.get_thread_buffer() = src_tile.get_thread_buffer();

            // final store
            store_tile(dst_window, dst_tile);
        }
    };
};
