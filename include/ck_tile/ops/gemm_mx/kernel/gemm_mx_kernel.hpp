// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp"

namespace ck_tile {


template <typename ScaleM = MXScalePointer<-1>, typename ScaleN = MXScalePointer<-1>, index_t NumATensor = 1, index_t NumBTensor = 1, index_t NumDTensor = 0>
struct MXGemmKernelArgs : UniversalGemmKernelArgs<NumATensor, NumBTensor, NumDTensor>
{
    using Base = UniversalGemmKernelArgs<NumATensor, NumBTensor, NumDTensor>;

    CK_TILE_HOST MXGemmKernelArgs(const std::array<const void*, NumATensor>& as_ptr_,
                                  const std::array<const void*, NumBTensor>& bs_ptr_,
                                  const std::array<const void*, NumDTensor>& ds_ptr_,
                                  void* e_ptr_,
                                  index_t k_batch_,
                                  index_t M_,
                                  index_t N_,
                                  index_t K_,
                                  const std::array<index_t, NumATensor>& stride_As_,
                                  const std::array<index_t, NumBTensor>& stride_Bs_,
                                  const std::array<index_t, NumDTensor>& stride_Ds_,
                                  index_t stride_E_,
                                  ScaleM scale_m_ptr_,
                                  ScaleN scale_n_ptr_)
        : Base{as_ptr_,
               bs_ptr_,
               ds_ptr_,
               e_ptr_,
               M_,
               N_,
               K_,
               stride_As_,
               stride_Bs_,
               stride_Ds_,
               stride_E_,
               k_batch_},
          scale_m_ptr(scale_m_ptr_),
          scale_n_ptr(scale_n_ptr_)
    {
    }

    ScaleM scale_m_ptr;
    ScaleN scale_n_ptr;
};

template <typename TilePartitioner_, typename MXGemmPipeline_, typename EpiloguePipeline_>
struct MXGemmKernel : UniversalGemmKernel<TilePartitioner_, MXGemmPipeline_, EpiloguePipeline_>
{
    using Underlying = UniversalGemmKernel<TilePartitioner_, MXGemmPipeline_, EpiloguePipeline_>;

    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using MXGemmPipeline = remove_cvref_t<MXGemmPipeline_>;
    using BlockGemmShape =
        remove_cvref_t<typename MXGemmPipeline::BlockGemmShape>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;
    using ALayout          = remove_cvref_t<typename MXGemmPipeline::ALayout>;
    using BLayout          = remove_cvref_t<typename MXGemmPipeline::BLayout>;
    using ELayout          = remove_cvref_t<typename MXGemmPipeline::CLayout>;
    using DsLayout         = remove_cvref_t<typename EpiloguePipeline::DsLayout>;
    using DsDataType       = remove_cvref_t<typename EpiloguePipeline::DsDataType>;
    static constexpr index_t KernelBlockSize  = MXGemmPipeline::BlockSize;
    static constexpr bool UsePersistentKernel = MXGemmPipeline::UsePersistentKernel;

    // Below type is actually accumulation data type - the output of block GEMM.
    using EDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    static constexpr auto I0 = number<0>();
    static constexpr auto I1 = number<1>();
    static constexpr auto I2 = number<2>();
    static constexpr auto I3 = number<3>();
    static constexpr auto I4 = number<4>();
    static constexpr auto I5 = number<5>();

    static constexpr index_t NumATensor = Underlying::AsDataType::size();
    static constexpr index_t NumBTensor = Underlying::BsDataType::size();
    static constexpr index_t NumDTensor = Underlying::DsDataType::size();

    using ADataType = remove_cvref_t<std::tuple_element_t<I0, typename Underlying::AsDataType>>;
    using BDataType = remove_cvref_t<std::tuple_element_t<I0, typename Underlying::BsDataType>>;

    static constexpr auto MThreadPerXdl = BlockGemmShape::WarpTile::at(number<0>{});
    static constexpr auto NThreadPerXdl = BlockGemmShape::WarpTile::at(number<1>{});
    static constexpr auto KThreadPerXdl = 64 / MThreadPerXdl;

    static constexpr auto APackedSize = numeric_traits<ADataType>::PackedSize;
    static constexpr auto BPackedSize = numeric_traits<BDataType>::PackedSize;

    /// @brief The e8m0 scales are packed into int32/float32 such that
    /// in one element contains a 2x2 block of scales (two rows, two lements in K dim)
    static constexpr auto MXdlPack = MXGemmPipeline::MXdlPack;
    static constexpr auto NXdlPack = MXGemmPipeline::NXdlPack;
    static constexpr auto KXdlPack = MXGemmPipeline::KXdlPack;

    static constexpr int kBlockPerCu = 1;

    static_assert(DsLayout::size() == DsDataType::size(),
                  "The size of DsLayout and DsDataType should be the same");

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "mx_gemm", gemm_prec_str<ADataType, BDataType>, MXGemmPipeline::GetName());
        // clang-format on
    }

    template <typename ScaleM, typename ScaleN>
    using KernelArgs = MXGemmKernelArgs<ScaleM, ScaleN, NumATensor, NumBTensor, NumDTensor>;

    template <typename ScaleM, typename ScaleN>
    CK_TILE_HOST static auto MakeKernelArgs(const std::array<const void*, NumATensor>& as_ptr,
                                            const std::array<const void*, NumBTensor>& bs_ptr,
                                            const std::array<const void*, NumDTensor>& ds_ptr,
                                            void* e_ptr,
                                            index_t k_batch,
                                            index_t M,
                                            index_t N,
                                            index_t K,
                                            const std::array<index_t, NumATensor>& stride_As,
                                            const std::array<index_t, NumBTensor>& stride_Bs,
                                            const std::array<index_t, NumDTensor>& stride_Ds,
                                            index_t stride_E,
                                            ScaleM scale_m_ptr,
                                            ScaleN scale_n_ptr)
    {
        return KernelArgs<ScaleM, ScaleN>(as_ptr,
                                          bs_ptr,
                                          ds_ptr,
                                          e_ptr,
                                          k_batch,
                                          M,
                                          N,
                                          K,
                                          stride_As,
                                          stride_Bs,
                                          stride_Ds,
                                          stride_E,
                                          scale_m_ptr,
                                          scale_n_ptr);
    }

    template <class ScaleM, class ScaleN>
    CK_TILE_HOST static constexpr auto
    GridSize(const KernelArgs<ScaleM, ScaleN>& kargs)
    {
        hipDeviceProp_t prop;
        int deviceId = 0; // default device

        int dync_smem_size       = 0;
        int maxActiveBlocksPerCU = 0;

        if(hipGetDeviceProperties(&prop, deviceId) != hipSuccess)
            throw std::runtime_error(std::string("hipGetDeviceProperties failed: ") +
                                        hipGetErrorName(hipGetLastError()));

        if(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                &maxActiveBlocksPerCU,
                reinterpret_cast<void*>(
                    kentry<1, MXGemmKernel, remove_cvref_t<decltype(kargs)>>),
                KernelBlockSize,
                dync_smem_size) != hipSuccess)
            throw std::runtime_error(
                std::string("hipOccupancyMaxActiveBlocksPerMultiprocessor failed: ") +
                hipGetErrorName(hipGetLastError()));

        const int persistent_block_size = prop.multiProcessorCount * maxActiveBlocksPerCU;
        const int total_work_tile_cnt   = TilePartitioner::GridSize(kargs.M, kargs.N);

        return dim3(min(persistent_block_size, total_work_tile_cnt), 1, 1);
    }

    using SplitKBatchOffset = typename Underlying::SplitKBatchOffset;

    template <memory_operation_enum DstInMemOp = memory_operation_enum::set, typename ScaleM, typename ScaleN>
    CK_TILE_DEVICE static auto
    MakeGemmTensorViews(const std::array<const ADataType*, NumATensor>& as_ptr,
                        const std::array<const BDataType*, NumBTensor>& bs_ptr,
                        const std::array<const void*, NumDTensor>& ds_ptr,
                        EDataType* e_ptr,
                        const KernelArgs<ScaleM, ScaleN>& kargs,
                        const SplitKBatchOffset& splitk_batch_offset)
    {
        // Get tensor views from the UniversalGemmKernel
        const auto& gemm_tensor_views_tuple =
            Underlying::template MakeGemmTensorViews<DstInMemOp>(
                as_ptr, bs_ptr, ds_ptr, e_ptr, kargs, splitk_batch_offset.splitted_k);

        auto scale_a = kargs.scale_m_ptr;
        auto scale_b = kargs.scale_n_ptr;

        static_assert(ScaleM::GranularityK == ScaleN::GranularityK, "M and N scales must have same K granularity!");
        static constexpr int BlockScaleSize = ScaleM::GranularityK;
        const auto&& scale_packs_m = integer_divide_ceil(kargs.M, (MXdlPack * MThreadPerXdl));
        const auto&& scale_packs_n = integer_divide_ceil(kargs.N, (NXdlPack * NThreadPerXdl));
        const auto&& scale_packs_k = kargs.K / BlockScaleSize / (KXdlPack * KThreadPerXdl);

        // A scale tensor view
        const auto& scale_a_tensor_view = [&]() {
            // Pack 2x2 e8m0 over M/K dimension into 1 int32_t to trigger dword width load
            const auto scale_a_naive_desc = make_naive_tensor_descriptor_packed(
                make_tuple(scale_packs_m, scale_packs_k, KThreadPerXdl, MThreadPerXdl));
            const auto scale_a_desc = transform_tensor_descriptor(
                scale_a_naive_desc,
                make_tuple(make_merge_transform(make_tuple(scale_packs_m, MThreadPerXdl)),
                           make_merge_transform(make_tuple(scale_packs_k, KThreadPerXdl))),
                make_tuple(sequence<0, 3>{}, sequence<1, 2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return make_tensor_view<address_space_enum::global>(
                reinterpret_cast<const int32_t*>(scale_a.ptr), scale_a_desc);
        }();

        // B scale tensor view
        const auto& scale_b_tensor_view = [&]() {
            const auto scale_b_naive_desc = make_naive_tensor_descriptor_packed(
                make_tuple(scale_packs_n, scale_packs_k, KThreadPerXdl, NThreadPerXdl));
            const auto scale_b_desc = transform_tensor_descriptor(
                scale_b_naive_desc,
                make_tuple(make_merge_transform(make_tuple(scale_packs_n, NThreadPerXdl)),
                           make_merge_transform(make_tuple(scale_packs_k, KThreadPerXdl))),
                make_tuple(sequence<0, 3>{}, sequence<1, 2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return make_tensor_view<address_space_enum::global>(
                reinterpret_cast<const int32_t*>(scale_b.ptr), scale_b_desc);
        }();

        return concat_tuple(gemm_tensor_views_tuple, make_tuple(scale_a_tensor_view, scale_b_tensor_view));
    }

    template <typename TensorView>
    CK_TILE_DEVICE static auto MakeGemmPadViews(const TensorView& views)
    {
        const auto& padded_views = Underlying::template MakeGemmPadViews<TensorView>(views);

        return make_tuple(
            padded_views.at(I0), padded_views.at(I1), padded_views.at(I2), padded_views.at(I3), views.at(I4), views.at(I5));
    }

    template <typename PadView>
    CK_TILE_DEVICE static auto
    MakeGemmTileWindows(const PadView& views, const index_t i_m, const index_t i_n)
    {
        const auto& tile_windows = Underlying::template MakeGemmTileWindows<PadView>(views, i_m, i_n);

        static constexpr int BlockScaleSize = 32;

        // We are packing 2x2 (MXdlPack x KXdlPack) scales (e8m0) into one int32 element
        auto scale_a_block_window = make_tile_window(
            views.at(I4),
            make_tuple(number<TilePartitioner::MPerBlock / MXdlPack>{},
                       number<TilePartitioner::KPerBlock / (BlockScaleSize * KXdlPack)>{}),
            {i_m / MXdlPack, 0});

        // We are packing 2x2 (NXdlPack x KXdlPack) scales (e8m0) into one int32 element
        auto scale_b_block_window = make_tile_window(
            views.at(I5),
            make_tuple(number<TilePartitioner::NPerBlock / NXdlPack>{},
                       number<TilePartitioner::KPerBlock / (BlockScaleSize * KXdlPack)>{}),
            {i_n / NXdlPack, 0});

        return make_tuple(tile_windows.at(I0),
                          tile_windows.at(I1),
                          tile_windows.at(I2),
                          tile_windows.at(I3),
                          scale_a_block_window,
                          scale_b_block_window);
    }

    template <class ScaleM, class ScaleN, bool UseDefaultScheduler = true>
    CK_TILE_DEVICE static void
    RunMxGemm(const std::array<const ADataType*, NumATensor>& as_ptr,
              const std::array<const BDataType*, NumBTensor>& bs_ptr,
              const std::array<const void*, NumDTensor>& ds_ptr,
              EDataType* e_ptr,
              void* smem_ptr_ping,
              void* smem_ptr_pong,
              const KernelArgs<ScaleM, ScaleN>& kargs,
              const SplitKBatchOffset& splitk_batch_offset,
              const index_t block_idx_m,
              const index_t block_idx_n)
    {
        // Create Gemm tensor views, pad views and tile windows
        const auto& gemm_tensor_views_tuple =
            MakeGemmTensorViews<EpiloguePipeline::MemoryOperation>(
                as_ptr, bs_ptr, ds_ptr, e_ptr, kargs, splitk_batch_offset);
        const auto& gemm_pad_views = MakeGemmPadViews(gemm_tensor_views_tuple);
        auto gemm_tile_windows     = MakeGemmTileWindows(gemm_pad_views, block_idx_m, block_idx_n);

        const index_t num_loop = TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k);

        // Run GEMM cooperatively by whole workgroup.
        const auto& a_block_window       = gemm_tile_windows.at(I0);
        const auto& b_block_window       = gemm_tile_windows.at(I1);
        const auto& d_block_window       = gemm_tile_windows.at(I2);
        const auto& scale_a_block_window = gemm_tile_windows.at(I4);
        const auto& scale_b_block_window = gemm_tile_windows.at(I5);

        static_assert(ScaleM::GranularityK == ScaleN::GranularityK // have the same granK
                          || ScaleM::GranularityMN == -1           // or ScaleA is disable
                          || ScaleN::GranularityMN == -1,          // or ScaleB is disable
                      "ScaleM and ScaleN should have the same GranularityK");

        const auto& c_block_tile = MXGemmPipeline{}(a_block_window,
                                                      b_block_window,
                                                      scale_a_block_window,
                                                      scale_b_block_window,
                                                      num_loop,
                                                      smem_ptr_ping,
                                                      smem_ptr_pong);

        // Run Epilogue Pipeline
        auto& c_block_window = gemm_tile_windows.at(I3);
        EpiloguePipeline{}(c_block_window, c_block_tile, d_block_window, smem_ptr_ping);
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemPingSize()
    {
        return max(MXGemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemPongSize()
    {
        return MXGemmPipeline::GetSmemSize();
    }

    template <class ScaleM, class ScaleN>
    CK_TILE_DEVICE void operator()(KernelArgs<ScaleM, ScaleN> kargs,
                                   int partition_idx = get_block_id()) const
    {
        const int total_work_tile_cnt = amd_wave_read_first_lane(TilePartitioner::GridSize(kargs.M, kargs.N));

        do
        {
            const auto [iM, iN] =
                TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(partition_idx);
            const index_t i_m = amd_wave_read_first_lane(iM * TilePartitioner::MPerBlock);
            const index_t i_n = amd_wave_read_first_lane(iN * TilePartitioner::NPerBlock);

            const SplitKBatchOffset splitk_batch_offset(kargs);
            // options
            EDataType* e_ptr = static_cast<EDataType*>(kargs.e_ptr);

            // options
            std::array<const ADataType*, NumATensor> as_ptr;
            static_for<0, NumATensor, 1>{}([&](auto i) {
                as_ptr[i] = static_cast<const ADataType*>(kargs.as_ptr[i]) +
                            splitk_batch_offset.as_k_split_offset[i] / APackedSize;
            });

            std::array<const BDataType*, NumBTensor> bs_ptr;
            static_for<0, NumBTensor, 1>{}([&](auto i) {
                bs_ptr[i] = static_cast<const BDataType*>(kargs.bs_ptr[i]) +
                            splitk_batch_offset.bs_k_split_offset[i] / BPackedSize;
            });

            // allocate LDS
            __shared__ char smem_ptr_ping[GetSmemPingSize()];
            __shared__ char smem_ptr_pong[GetSmemPongSize()];

            if constexpr(!(EpiloguePipeline::MemoryOperation == memory_operation_enum::atomic_add &&
                           EpiloguePipeline::GetVectorSizeC() % 2 != 0 &&
                           is_any_of<EDataType, fp16_t, bf16_t>::value))
            {
                constexpr auto scheduler_type = (MXGemmPipeline::NumWaveGroups == 1);
                RunMxGemm<ScaleM, ScaleN, scheduler_type>(as_ptr,
                                                          bs_ptr,
                                                          kargs.ds_ptr,
                                                          e_ptr,
                                                          smem_ptr_ping,
                                                          smem_ptr_pong,
                                                          kargs,
                                                          splitk_batch_offset,
                                                          i_m,
                                                          i_n);
            }
            else
            {
                static_assert(false,
                              "Unimplemented: atomic_add with odd vector size for fp16/bf16");
            }
            partition_idx += gridDim.x;
        } while(UsePersistentKernel && partition_idx < total_work_tile_cnt);
    }
};

} // namespace ck_tile
