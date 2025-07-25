// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v1_default_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include "block_universal_gemm_base.hpp"

namespace ck_tile {

// A is block distributed tensor
// B is block distributed tensor
// C is block distributed tensor
template <typename Problem_, typename Policy_ = BlockGemmARegBSmemCRegV1DefaultPolicy>
struct BlockUniversalGemmArBrCr : public BlockUniversalGemmBase<Problem_, Policy_>
{
    private:
    using Base = BlockUniversalGemmBase<Problem_, Policy_>;
    using Base::a_warp_y_index_zeros;
    using Base::a_warp_y_lengths;
    using Base::b_warp_y_index_zeros;
    using Base::b_warp_y_lengths;
    using Base::c_warp_y_index_zeros;
    using Base::c_warp_y_lengths;
    using Base::KIterPerWarp;
    using Base::MIterPerWarp;
    using Base::NIterPerWarp;
    using Base::Scheduler;
    using typename Base::ADataType;
    using typename Base::AWarpTensor;
    using typename Base::BDataType;
    using typename Base::BWarpTensor;
    using typename Base::CDataType;
    using typename Base::CWarpTensor;

    using GemmTraits = typename Base::template GemmTraits_<Problem_, Policy_>;

    public:
    using Base::MakeABlockDistributionEncode;
    using Base::MakeBBlockDistributionEncode;
    using Base::MakeCBlockTile;
    using typename Base::WarpGemm;

    private:
    template <GemmPipelineScheduler scheduler, typename GemmTraits_>
    struct BlockGemmImpl
    {
    };

    template <typename GemmTraits_>
    struct BlockGemmImpl<GemmPipelineScheduler::Default, GemmTraits_>
    {
        // C += A * B
        template <typename CBlockTensor, typename ARegBlockTensor, typename BRegBlockTensor>
        CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                       const ARegBlockTensor& a_block_tensor,
                                       const BRegBlockTensor& b_block_tensor)
        {
            static_assert(std::is_same_v<CDataType, typename CBlockTensor::DataType>,
                          "The CDataType as defined in traits should be the same as correspoinding "
                          "C block tensor data type!");
            static_assert(std::is_same_v<ADataType, typename ARegBlockTensor::DataType> &&
                              std::is_same_v<BDataType, typename BRegBlockTensor::DataType>,
                          "The ADataType and BDataType as defined in "
                          "traits should be the same as correspoinding block window data type!");

            // hot loop:
            static_for<0, GemmTraits_::KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    // read A warp tensor from A block tensor
                    AWarpTensor a_warp_tensor;

                    a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // read B warp tensor from B block tensor
                        BWarpTensor b_warp_tensor;

                        b_warp_tensor.get_thread_buffer() = b_block_tensor.get_y_sliced_thread_data(
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
        // C += A * B
        template <typename CBlockTensor, typename ARegBlockTensor, typename BRegBlockTensor>
        CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                       ARegBlockTensor& a_block_tensor,
                                       BRegBlockTensor& b_block_tensor)
        {
            static_assert(std::is_same_v<CDataType, typename CBlockTensor::DataType>,
                          "The CDataType as defined in traits should be the same as correspoinding "
                          "C block tensor data type!");

            // hot loop:
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    // read A warp tensor from A block tensor
                    AWarpTensor a_warp_tensor;

                    a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, kIter>{}, a_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, a_warp_y_lengths));

                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        // read B warp tensor from B block tensor
                        BWarpTensor b_warp_tensor;

                        b_warp_tensor.get_thread_buffer() = b_block_tensor.get_y_sliced_thread_data(
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

        // C += A * B
        template <typename CBlockTensor, typename ARegBlockTensor, typename BRegBlockTensor>
        CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                       const ARegBlockTensor& a_block_tensor,
                                       const BRegBlockTensor& b_block_tensor)
        {
            static_assert(std::is_same_v<CDataType, typename CBlockTensor::DataType>,
                          "The CDataType as defined in traits should be the same as correspoinding "
                          "C block tensor data type!");

            // hot loop:
            static_for<0, KRepeat, 1>{}([&](auto kIter) {
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

                        a_warp_tensor.get_thread_buffer() = a_block_tensor.get_y_sliced_thread_data(
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
                            }
                        });
                    });
                });
            });
        }
    };

    public:
    // C += A * B
    template <typename CBlockTensor, typename ARegBlockTensor, typename BRegBlockTensor>
    CK_TILE_DEVICE void operator()(CBlockTensor& c_block_tensor,
                                   const ARegBlockTensor& a_block_tensor,
                                   const BRegBlockTensor& b_block_tensor)
    {

        block_gemm_impl_(c_block_tensor, a_block_tensor, b_block_tensor);
    }

    // C = A * B
    template <typename ARegBlockTensor, typename BRegBlockTensor>
    CK_TILE_DEVICE auto operator()(const ARegBlockTensor& a_block_tensor,
                                   const BRegBlockTensor& b_block_tensor)
    {
        auto c_block_tensor = Base::MakeCBlockTile();

        block_gemm_impl_(c_block_tensor, a_block_tensor, b_block_tensor);

        return c_block_tensor;
    }

    private:
    BlockGemmImpl<Scheduler, GemmTraits> block_gemm_impl_{};
};

} // namespace ck_tile
