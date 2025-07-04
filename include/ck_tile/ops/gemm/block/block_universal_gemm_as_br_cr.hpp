// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_breg_creg_v1_default_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include "block_universal_gemm_base.hpp"

namespace ck_tile {

// A is block window on shared memory
// B is block distributed tensor
// C is block distributed tensor
template <typename Problem_, typename Policy_ = BlockGemmASmemBRegCRegV1DefaultPolicy>
struct BlockUniversalGemmAsBrCr : public BlockUniversalGemmBase<Problem_, Policy_>
{
    private:
    using Base = BlockUniversalGemmBase<Problem_, Policy_>;
    using typename Base::ADataType;
    using typename Base::BDataType;
    using typename Base::CDataType;
    using typename Base::ComputeDataType;
    using typename Base::WarpGemm;
    using typename Base::AWarpTensor;
    using typename Base::BWarpTensor;
    using typename Base::CWarpTensor;
    using Base::MIterPerWarp;
    using Base::NIterPerWarp;
    using Base::KIterPerWarp;
    using Base::a_warp_y_index_zeros;
    using Base::b_warp_y_index_zeros;
    using Base::c_warp_y_index_zeros;
    using Base::a_warp_y_lengths;
    using Base::b_warp_y_lengths;
    using Base::c_warp_y_lengths;
    using Base::Scheduler;
    using Base::load_interleaved_pk_type;

    using GemmTraits = typename Base::template GemmTraits_<Problem_, Policy_>;

    public:
    using Base::MakeABlockDistributionEncode;
    using Base::MakeBBlockDistributionEncode;
    using Base::MakeCBlockTile;

    private:
    template <GemmPipelineScheduler scheduler, typename GemmTraits_>
    struct BlockGemmImpl
    {
    };

    template <typename GemmTraits_>
    struct BlockGemmImpl<GemmPipelineScheduler::Default, GemmTraits_>
    {
        static constexpr auto ALdsTileDistr =
            decltype(make_static_tile_distribution(MakeABlockDistributionEncode())){};
        static constexpr auto BLdsTileDistr =
            decltype(make_static_tile_distribution(MakeBBlockDistributionEncode())){};

        using ALdsTile = decltype(make_static_distributed_tensor<ComputeDataType>(ALdsTileDistr));
        using BLdsTile = decltype(make_static_distributed_tensor<ComputeDataType>(BLdsTileDistr));

        ALdsTile a_warp_tile_;
        BLdsTile b_warp_tile_;

        // C += A * B
        template <typename CBlockTensor, typename ASmemBlockWindow, typename BRegBlockTensor>
        CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                       const ASmemBlockWindow& a_block_window,
                                       const BRegBlockTensor& b_block_tensor)
        {
            static_assert(std::is_same_v<CDataType, typename CBlockTensor::DataType>,
                          "The CDataType as defined in traits should be the same as correspoinding "
                          "C block tensor data type!");
            static_assert(std::is_same_v<ADataType, typename ASmemBlockWindow::DataType> &&
                              std::is_same_v<BDataType, typename BRegBlockTensor::DataType>,
                          "The ADataType and BDataType as defined in "
                          "traits should be the same as correspoinding block window data type!");

            if constexpr(std::is_same_v<ADataType, pk_int4_t>)
            {
                load_interleaved_pk_type(a_warp_tile_, a_block_window);
            }
            else
            {
                load_tile(a_warp_tile_, a_block_window);
            }
            b_warp_tile_.get_thread_buffer() = b_block_tensor.get_thread_buffer();

            // hot loop:
            static_for<0, GemmTraits_::KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    // read A warp tensor from A block tensor
                    AWarpTensor a_warp_tensor;

                    a_warp_tensor.get_thread_buffer() = a_warp_tile_.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // read B warp tensor from B block tensor
                        BWarpTensor b_warp_tensor;

                        b_warp_tensor.get_thread_buffer() = b_warp_tile_.get_y_sliced_thread_data(
                            merge_sequences(sequence<nIter, kIter>{}, b_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

                        // read C warp tensor from C block tensor-
                        CWarpTensor c_warp_tensor;

                        c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                        // warp GEMM
                        WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                        // write C warp tensor into C block tensor
                        c_block_tensor.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());
                    });
                });
            });
        }
    };

    template <typename GemmTraits_>
    struct BlockGemmImpl<GemmPipelineScheduler::Intrawave, GemmTraits_>
    {
        static constexpr auto ALdsTileDistr =
            decltype(make_static_tile_distribution(MakeABlockDistributionEncode())){};
        static constexpr auto BLdsTileDistr =
            decltype(make_static_tile_distribution(MakeBBlockDistributionEncode())){};

        using ALdsTile = decltype(make_static_distributed_tensor<ComputeDataType>(ALdsTileDistr));
        using BLdsTile = decltype(make_static_distributed_tensor<ComputeDataType>(BLdsTileDistr));

        ALdsTile a_warp_tile_;
        BLdsTile b_warp_tile_;

        template <typename ASmemBlockWindow>
        CK_TILE_DEVICE void LocalPrefetchA(const ASmemBlockWindow& a_block_window)
        {
            if constexpr(std::is_same_v<ADataType, pk_int4_t>)
            {
                load_interleaved_pk_type(a_warp_tile_, a_block_window);
            }
            else
            {
                load_tile(a_warp_tile_, a_block_window);
            }
        }

        // C += A * B
        template <typename CBlockTensor, typename ASmemBlockWindow, typename BRegBlockTensor>
        CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                       [[maybe_unused]] ASmemBlockWindow& a_block_window,
                                       [[maybe_unused]] BRegBlockTensor& b_block_tensor)
        {
            static_assert(std::is_same_v<CDataType, typename CBlockTensor::DataType>,
                          "The CDataType as defined in traits should be the same as correspoinding "
                          "C block tensor data type!");

            b_warp_tile_.get_thread_buffer() = b_block_tensor.get_thread_buffer();
            // hot loop:
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    // read A warp tensor from A block tensor
                    AWarpTensor a_warp_tensor;

                    a_warp_tensor.get_thread_buffer() = a_warp_tile_.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // read B warp tensor from B block tensor
                        BWarpTensor b_warp_tensor;

                        b_warp_tensor.get_thread_buffer() = b_warp_tile_.get_y_sliced_thread_data(
                            merge_sequences(sequence<nIter, kIter>{}, b_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));

                        // read C warp tensor from C block tensor
                        CWarpTensor c_warp_tensor;

                        c_warp_tensor.get_thread_buffer() = c_block_tensor.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                        // warp GEMM
                        WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                        // write C warp tensor into C block tensor
                        c_block_tensor.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());
                    });
                });
            });
        }
    };

    template <typename GemmTraits_>
    struct BlockGemmImpl<GemmPipelineScheduler::Interwave, GemmTraits_>
    {
        static constexpr index_t KPerThread     = GemmTraits_::KPerThread;
        static constexpr index_t NumMacClusters = GemmTraits_::InterWaveSchedulingMacClusters;
        static constexpr index_t KPerInnerLoop =
            ck_tile::max(KPerThread / NumMacClusters, WarpGemm::kKPerThread);
        static constexpr index_t KRepeat        = KPerThread / KPerInnerLoop;
        static constexpr index_t KInnerLoopIter = KPerInnerLoop / WarpGemm::kKPerThread;

        static constexpr auto ALdsTileDistr =
            decltype(make_static_tile_distribution(MakeABlockDistributionEncode())){};
        using ALdsTile = decltype(make_static_distributed_tensor<ComputeDataType>(ALdsTileDistr));
        ALdsTile a_warp_tile_;

        template <index_t KIdx, typename ASmemBlockWindow>
        CK_TILE_DEVICE void LocalPrefetchA(const ASmemBlockWindow& a_block_window)
        {
            constexpr auto a_lds_load_tile_distr =
                make_static_tile_distribution(MakeABlockDistributionEncode());

            auto a_lds_gemm_window = make_tile_window(
                a_block_window.get_bottom_tensor_view(),
                make_tuple(number<GemmTraits::MPerBlock>{}, number<KPerInnerLoop>{}),
                {0, KIdx * KPerInnerLoop},
                a_lds_load_tile_distr);

            if constexpr(std::is_same_v<ADataType, pk_int4_t>)
            {
                load_interleaved_pk_type(a_warp_tile_, a_block_window);
            }
            else
            {
                load_tile(a_warp_tile_, a_lds_gemm_window);
            }
        }

        // C += A * B
        template <typename CBlockTensor, typename ASmemBlockWindow, typename BRegBlockTensor>
        CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                       const ASmemBlockWindow& a_block_window,
                                       const BRegBlockTensor& b_block_tensor)
        {
            static_assert(std::is_same_v<CDataType, typename CBlockTensor::DataType>,
                          "The CDataType as defined in traits should be the same as correspoinding "
                          "C block tensor data type!");

            // hot loop:
            static_for<0, KRepeat, 1>{}([&](auto kIter) {
                LocalPrefetchA<kIter.value>(a_block_window);
                __builtin_amdgcn_sched_barrier(0);
                // NOTE: Synchronize threads in a workgroup at the start of each MAC
                // cluster, but except the first, as we can shorten non-MAC cluster a bit
                // and there's no observable negative impact. The desired effect is waves in
                // a workgroup executing MAC in sync. This avoids some out-of-sync waves
                // hijacking MAC resource from other workgroups and reducing the chance of
                // latency hiding by waiting for the rest of the workgroup at the eventual
                // sync point.
                if constexpr(kIter.value != 0 || KRepeat == 1)
                {
                    __builtin_amdgcn_s_barrier();
                    __builtin_amdgcn_sched_barrier(0);
                }

                static_for<0, KInnerLoopIter, 1>{}([&](auto kInnerIter) {
                    static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                        // read A warp tensor from A block tensor
                        AWarpTensor a_warp_tensor;

                        a_warp_tensor.get_thread_buffer() = a_warp_tile_.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, kInnerIter>{}, a_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));
                        static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                            // read B warp tensor from B block tensor
                            BWarpTensor b_warp_tensor;

                            b_warp_tensor.get_thread_buffer() =
                                b_block_tensor.get_y_sliced_thread_data(
                                    merge_sequences(sequence<nIter, kInnerIter>{},
                                                    b_warp_y_index_zeros),
                                    merge_sequences(sequence<1, 1>{}, b_warp_y_lengths));
                            // read C warp tensor from C block tensor-
                            CWarpTensor c_warp_tensor;

                            c_warp_tensor.get_thread_buffer() =
                                c_block_tensor.get_y_sliced_thread_data(
                                    merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                                    merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));

                            // The block_sync_lds() here performs double duty:
                            // A) safeguard against data hazard because barrier from
                            // blockwise_gemm is moved here B) reduce VMEM FIFO congestion
                            // by applying small delays to different wavefronts It is
                            // performed near the end of MAC cluster to minimize lgkmcnt
                            // penalty
                            if constexpr(kIter.value == KRepeat - 1 &&
                                         kInnerIter.value == KInnerLoopIter - 1 &&
                                         mIter.value == MIterPerWarp - 1 &&
                                         nIter.value == NIterPerWarp - 1)
                            {
                                __builtin_amdgcn_sched_barrier(0);
                                block_sync_lds();
                                __builtin_amdgcn_sched_barrier(0);
                            }
                            // warp GEMM
                            WarpGemm{}(c_warp_tensor, a_warp_tensor, b_warp_tensor);

                            // write C warp tensor into C block tensor
                            c_block_tensor.set_y_sliced_thread_data(
                                merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                                merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                                c_warp_tensor.get_thread_buffer());

                            if constexpr(kInnerIter.value == 0 && mIter.value == 0 &&
                                         nIter.value == 0)
                            {
                                __builtin_amdgcn_sched_barrier(0);
                                __builtin_amdgcn_s_setprio(1);
                                __builtin_amdgcn_sched_barrier(0);
                            }
                        });
                    });
                });

                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_setprio(0);
                __builtin_amdgcn_sched_barrier(0);
            });
        }
    };

    public:
    template <typename ASmemBlockWindow>
    CK_TILE_DEVICE void LocalPrefetchA(const ASmemBlockWindow& a_block_window)
    {
        block_gemm_impl_.LocalPrefetchA(a_block_window);
    }

    // C += A * B
    template <typename CBlockTensor, typename ASmemBlockWindow, typename BRegBlockTensor>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ASmemBlockWindow& a_block_window,
                                   const BRegBlockTensor& b_block_tensor)
    {
        block_gemm_impl_(c_block_tensor, a_block_window, b_block_tensor);
    }

    // C = A * B
    template <typename ASmemBlockWindow, typename BRegBlockTensor>
    CK_TILE_DEVICE auto operator()(const ASmemBlockWindow& a_block_window,
                                   const BRegBlockTensor& b_block_tensor)
    {
        auto c_block_tensor = Base::MakeCBlockTile();
        block_gemm_impl_(c_block_tensor, a_block_window, b_block_tensor);
        return c_block_tensor;
    }

    private:
    BlockGemmImpl<Scheduler, GemmTraits> block_gemm_impl_{};
};

} // namespace ck_tile
