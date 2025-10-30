// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

#include "ck_tile/ops/flatmm/kernel/flatmm_kernel.hpp"

namespace ck_tile {

template <typename TilePartitioner_, typename MXFlatmmPipeline_, typename EpiloguePipeline_>
struct MXFlatmmKernel : FlatmmKernel<TilePartitioner_, MXFlatmmPipeline_, EpiloguePipeline_>
{
    using Underlying = FlatmmKernel<TilePartitioner_, MXFlatmmPipeline_, EpiloguePipeline_>;

    using TilePartitioner = remove_cvref_t<TilePartitioner_>;
    using FlatmmPipeline  = remove_cvref_t<MXFlatmmPipeline_>;
    using BlockGemmShape =
        remove_cvref_t<typename MXFlatmmPipeline_::BlockGemmShape>; // TileFlatmmShape
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;
    using ALayout          = remove_cvref_t<typename FlatmmPipeline::ALayout>;
    using BLayout          = remove_cvref_t<typename FlatmmPipeline::BLayout>;
    using ELayout          = remove_cvref_t<typename FlatmmPipeline::CLayout>;
    using DsLayout         = remove_cvref_t<typename EpiloguePipeline::DsLayout>;
    using DsDataType       = remove_cvref_t<typename EpiloguePipeline::DsDataType>;
    static constexpr index_t KernelBlockSize  = FlatmmPipeline::BlockSize;
    static constexpr bool UsePersistentKernel = FlatmmPipeline::UsePersistentKernel;

    using ADataType = remove_cvref_t<typename FlatmmPipeline::ADataType>;
    using BDataType = remove_cvref_t<typename FlatmmPipeline::BDataType>;
    // Below type is actually accumulation data type - the output of block GEMM.
    using EDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    static constexpr int MThreadPerXdl = BlockGemmShape::WarpTile::at(number<0>{});
    static constexpr int NThreadPerXdl = BlockGemmShape::WarpTile::at(number<1>{});
    static constexpr int KThreadPerXdl = 64 / MThreadPerXdl;

    static constexpr int APackedSize = numeric_traits<ADataType>::PackedSize;
    static constexpr int BPackedSize = numeric_traits<BDataType>::PackedSize;

    static constexpr int MXdlPack = FlatmmPipeline::MXdlPack;
    static constexpr int NXdlPack = FlatmmPipeline::NXdlPack;
    static constexpr int KXdlPack = FlatmmPipeline::KXdlPack;

    static constexpr index_t NumDTensor = DsDataType::size();

    static constexpr auto I0 = number<0>();
    static constexpr auto I1 = number<1>();
    static constexpr auto I2 = number<2>();
    static constexpr auto I3 = number<3>();
    static constexpr auto I4 = number<4>();
    static constexpr auto I5 = number<5>();

    static_assert(DsLayout::size() == DsDataType::size(),
                  "The size of DsLayout and DsDataType should be the same");
    // using KernelArgs = FlatmmKernelArgs<DsLayout::size()>;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "mx_flatmm_gemm", gemm_prec_str<ADataType, BDataType>, FlatmmPipeline::GetName());
        // clang-format on
    }

    template <class ScaleM, class ScaleN>
    CK_TILE_HOST static constexpr auto
    GridSize(const FlatmmKernelArgs<ScaleM, ScaleN, DsDataType::size()>& kargs)
    {
        if constexpr(UsePersistentKernel)
        {
            hipDeviceProp_t prop;
            int deviceId = 0; // default device

            constexpr int block_size = MXFlatmmKernel::BlockSize().x;
            int dync_smem_size       = 0;
            int maxActiveBlocksPerCU = 0;

            if(hipGetDeviceProperties(&prop, deviceId) != hipSuccess)
                throw std::runtime_error(std::string("hipGetDeviceProperties failed: ") +
                                         hipGetErrorName(hipGetLastError()));

            if(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                   &maxActiveBlocksPerCU,
                   reinterpret_cast<void*>(
                       kentry<1, MXFlatmmKernel, remove_cvref_t<decltype(kargs)>>),
                   block_size,
                   dync_smem_size) != hipSuccess)
                throw std::runtime_error(
                    std::string("hipOccupancyMaxActiveBlocksPerMultiprocessor failed: ") +
                    hipGetErrorName(hipGetLastError()));

            const int persistent_block_size = prop.multiProcessorCount * maxActiveBlocksPerCU;
            const int total_work_tile_cnt   = TilePartitioner::GridSize(kargs.M, kargs.N);

            // std::cout << "maxActiveBlocksPerCU: " << maxActiveBlocksPerCU
            //           << ", persistent_block_size: " << persistent_block_size
            //           << ", total_work_tile_cnt: " << total_work_tile_cnt << std::endl;

            if(kargs.k_batch != 1)
                throw std::runtime_error("Wrong! k_batch != 1 not supported in persistent kernel");
            return dim3(min(persistent_block_size, total_work_tile_cnt), 1, kargs.k_batch);
        }
        else
        {
            return dim3(TilePartitioner::GridSize(kargs.M, kargs.N), 1, kargs.k_batch);
        }
    }

    using SplitKBatchOffset = typename Underlying::SplitKBatchOffset;

    template <memory_operation_enum DstInMemOp = memory_operation_enum::set, class KernelArgs>
    CK_TILE_DEVICE static auto
    MakeGemmTensorViews(const ADataType* a_ptr,
                        const BDataType* b_flat_ptr,
                        const std::array<const void*, NumDTensor>& ds_ptr,
                        EDataType* e_ptr,
                        const KernelArgs& kargs,
                        const SplitKBatchOffset& splitk_batch_offset)
    {
        const auto& a_tensor_view = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_ptr,
                    make_tuple(kargs.M, splitk_batch_offset.splitted_k),
                    make_tuple(kargs.stride_A, 1),
                    number<FlatmmPipeline::GetVectorSizeA()>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_ptr,
                    make_tuple(splitk_batch_offset.splitted_k, kargs.M),
                    make_tuple(kargs.stride_A, 1),
                    number<FlatmmPipeline::GetVectorSizeA()>{},
                    number<1>{});
            }
        }();

        index_t kFlatK = kargs.K * BlockGemmShape::WarpTile::at(I1);
        index_t kFlatN = kargs.N * kargs.K / kFlatK;

        const auto& b_flat_tensor_view = [&]() {
            return make_naive_tensor_view<address_space_enum::global>(
                b_flat_ptr,
                make_tuple(kFlatN, kFlatK),
                make_tuple(kFlatK, 1),
                number<FlatmmPipeline::GetVectorSizeB()>{},
                number<1>{});
        }();

        const auto& ds_tensor_view = generate_tuple(
            [&](auto i) {
                using DiLayout   = remove_cvref_t<std::tuple_element_t<i.value, DsLayout>>;
                using DDataType_ = remove_cvref_t<std::tuple_element_t<i.value, DsDataType>>;
                if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
                {
                    return make_naive_tensor_view<address_space_enum::global>(
                        static_cast<const DDataType_*>(ds_ptr[i]),
                        make_tuple(kargs.M, kargs.N),
                        make_tuple(kargs.stride_Ds[i], 1),
                        number<EpiloguePipeline::GetVectorSizeD(i)>{},
                        number<1>{});
                }
                else
                {
                    return make_naive_tensor_view<address_space_enum::global>(
                        static_cast<const DDataType_*>(ds_ptr[i]),
                        make_tuple(kargs.N, kargs.M),
                        make_tuple(kargs.stride_Ds[i], 1),
                        number<EpiloguePipeline::GetVectorSizeD(i)>{},
                        number<1>{});
                }
            },
            number<NumDTensor>{});

        // TODO: enable vector write for C in ColMajor
        const auto& e_tensor_view = [&]() {
            if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    e_ptr,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(kargs.stride_E, 1),
                    number<EpiloguePipeline::GetVectorSizeC()>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    e_ptr,
                    make_tuple(kargs.N, kargs.M),
                    make_tuple(kargs.stride_E, 1),
                    number<1>{},
                    number<1>{});
            }
        }();

        auto scale_a = kargs.scale_m_ptr;
        auto scale_b = kargs.scale_n_ptr;

        static constexpr int BlockScaleSize = 32; // decltype(scale_n)::GranularityK;
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
            const auto scale_b_navie_desc = make_naive_tensor_descriptor_packed(
                make_tuple(scale_packs_n, scale_packs_k, KThreadPerXdl, NThreadPerXdl));
            const auto scale_b_desc = transform_tensor_descriptor(
                scale_b_navie_desc,
                make_tuple(make_merge_transform(make_tuple(scale_packs_n, NThreadPerXdl)),
                           make_merge_transform(make_tuple(scale_packs_k, KThreadPerXdl))),
                make_tuple(sequence<0, 3>{}, sequence<1, 2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return make_tensor_view<address_space_enum::global>(
                reinterpret_cast<const int32_t*>(scale_b.ptr), scale_b_desc);
        }();

        return make_tuple(a_tensor_view,
                          b_flat_tensor_view,
                          ds_tensor_view,
                          e_tensor_view,
                          scale_a_tensor_view,
                          scale_b_tensor_view);
    }

    template <typename TensorView>
    CK_TILE_DEVICE static auto MakeGemmPadViews(const TensorView& views)
    {
        const auto& a_pad_view = [&]() {
            const auto& a_tensor_view = views.at(I0);
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(a_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::KPerBlock>{}),
                                       sequence<false, FlatmmPipeline::kPadK>{});
            }
            else
            {
                return pad_tensor_view(a_tensor_view,
                                       make_tuple(number<TilePartitioner::KPerBlock>{},
                                                  number<TilePartitioner::MPerBlock>{}),
                                       sequence<false, FlatmmPipeline::kPadM>{});
            }
        }();

        const auto& b_flat_tensor_view = views.at(I1);

        const auto& ds_pad_view = generate_tuple(
            [&](auto i) {
                const auto& d_tensor_view = views.at(I2);
                using DiLayout            = remove_cvref_t<std::tuple_element_t<i.value, DsLayout>>;
                if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
                {
                    return pad_tensor_view(d_tensor_view[i],
                                           make_tuple(number<TilePartitioner::MPerBlock>{},
                                                      number<TilePartitioner::NPerBlock>{}),
                                           sequence<false, FlatmmPipeline::kPadN>{});
                }
                else
                {
                    return pad_tensor_view(d_tensor_view[i],
                                           make_tuple(number<TilePartitioner::NPerBlock>{},
                                                      number<TilePartitioner::MPerBlock>{}),
                                           sequence<false, FlatmmPipeline::kPadM>{});
                }
            },
            number<NumDTensor>{});

        // TODO vector write in for C in ColMajor
        const auto& e_pad_view = [&]() {
            const auto& e_tensor_view = views.at(I3);
            if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(e_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<false, FlatmmPipeline::kPadN>{});
            }
            else
            {
                return pad_tensor_view(e_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<FlatmmPipeline::kPadM, false>{});
            }
        }();

        return make_tuple(
            a_pad_view, b_flat_tensor_view, ds_pad_view, e_pad_view, views.at(I4), views.at(I5));
    }

    template <typename PadView>
    CK_TILE_DEVICE static auto
    MakeGemmTileWindows(const PadView& views, const index_t i_m, const index_t i_n)
    {
        const auto& a_pad_view      = views.at(I0);
        const auto& b_flat_pad_view = views.at(I1);
        const auto& ds_pad_view     = views.at(I2);
        const auto& e_pad_view      = views.at(I3);

        const auto& a_block_window = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_tile_window(a_pad_view,
                                        make_tuple(number<TilePartitioner::MPerBlock>{},
                                                   number<TilePartitioner::KPerBlock>{}),
                                        {i_m, 0});
            }
            else
            {
                return make_tile_window(a_pad_view,
                                        make_tuple(number<TilePartitioner::KPerBlock>{},
                                                   number<TilePartitioner::MPerBlock>{}),
                                        {0, i_m});
            }
        }();

        const auto& b_flat_block_window =
            make_tile_window(b_flat_pad_view,
                             make_tuple(number<FlatmmPipeline::flatNPerWarp>{},
                                        number<FlatmmPipeline::flatKPerWarp>{}),
                             {static_cast<int>(i_n / BlockGemmShape::WarpTile::at(I1)), 0});

        const auto ds_block_window = generate_tuple(
            [&](auto i) {
                using DiLayout = remove_cvref_t<std::tuple_element_t<i.value, DsLayout>>;
                if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
                {
                    return make_tile_window(ds_pad_view[i],
                                            make_tuple(number<TilePartitioner::MPerBlock>{},
                                                       number<TilePartitioner::NPerBlock>{}),
                                            {i_m, i_n});
                }
                else
                {
                    return make_tile_window(ds_pad_view[i],
                                            make_tuple(number<TilePartitioner::NPerBlock>{},
                                                       number<TilePartitioner::MPerBlock>{}),
                                            {i_n, i_m});
                }
            },
            number<NumDTensor>{});

        auto e_block_window = make_tile_window(
            e_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {i_m, i_n});

        static constexpr int BlockScaleSize = 32;

        auto scale_a_block_window = make_tile_window(
            views.at(I4),
            make_tuple(number<TilePartitioner::MPerBlock / MXdlPack>{},
                       number<TilePartitioner::KPerBlock / (BlockScaleSize * KXdlPack)>{}),
            {i_m / MXdlPack, 0});

        auto scale_b_block_window = make_tile_window(
            views.at(I5),
            make_tuple(number<TilePartitioner::NPerBlock / NXdlPack>{},
                       number<TilePartitioner::KPerBlock / (BlockScaleSize * KXdlPack)>{}),
            {i_n / NXdlPack, 0});

        return make_tuple(a_block_window,
                          b_flat_block_window,
                          ds_block_window,
                          e_block_window,
                          scale_a_block_window,
                          scale_b_block_window);
    }

    template <class ScaleM, class ScaleN, bool UseDefaultScheduler = true>
    CK_TILE_DEVICE static void
    RunFlatmm(const ADataType* a_ptr,
              const BDataType* b_flat_ptr,
              const std::array<const void*, NumDTensor>& ds_ptr,
              EDataType* e_ptr,
              void* smem_ptr_ping,
              void* smem_ptr_pong,
              const FlatmmKernelArgs<ScaleM, ScaleN, DsDataType::size()>& kargs,
              const SplitKBatchOffset& splitk_batch_offset,
              const index_t block_idx_m,
              const index_t block_idx_n)
    {
        // Create Gemm tensor views, pad views and tile windows
        const auto& gemm_tensor_views_tuple =
            MakeGemmTensorViews<EpiloguePipeline::MemoryOperation>(
                a_ptr, b_flat_ptr, ds_ptr, e_ptr, kargs, splitk_batch_offset);
        const auto& gemm_pad_views = MakeGemmPadViews(gemm_tensor_views_tuple);
        auto gemm_tile_windows     = MakeGemmTileWindows(gemm_pad_views, block_idx_m, block_idx_n);

        const index_t num_loop = TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k);

        // Run GEMM cooperatively by whole workgroup.
        const auto& a_block_window       = gemm_tile_windows.at(I0);
        const auto& b_flat_block_window  = gemm_tile_windows.at(I1);
        const auto& d_block_window       = gemm_tile_windows.at(I2);
        const auto& scale_a_block_window = gemm_tile_windows.at(I4);
        const auto& scale_b_block_window = gemm_tile_windows.at(I5);

        static_assert(ScaleM::GranularityK == ScaleN::GranularityK // have the same granK
                          || ScaleM::GranularityMN == -1           // or ScaleA is disable
                          || ScaleN::GranularityMN == -1,          // or ScaleB is disable
                      "ScaleM and ScaleN should have the same GranularityK");
        constexpr bool DoEpiScale =
            (ScaleM::GranularityMN != -1 && ScaleM::GranularityK == 0) || // per token
            (ScaleN::GranularityMN != -1 && ScaleN::GranularityK == 0);   // per channel

        auto a_block_window_with_distr =
            ck_tile::make_tile_window(a_block_window.get_bottom_tensor_view(),
                                      a_block_window.get_window_lengths(),
                                      a_block_window.get_window_origin(),
                                      FlatmmPipeline::GetADramTileDistribution());
        const auto& c_block_tile = FlatmmPipeline{}(a_block_window_with_distr,
                                                    b_flat_block_window,
                                                    scale_a_block_window,
                                                    scale_b_block_window,
                                                    num_loop,
                                                    smem_ptr_ping,
                                                    smem_ptr_pong);

        // Run Epilogue Pipeline
        if constexpr(DoEpiScale)
        {
            auto& c_block_window = gemm_tile_windows.at(I3);
            EpiloguePipeline{}(c_block_window,
                               c_block_tile,
                               d_block_window,
                               smem_ptr_ping,
                               kargs.scale_m_ptr + block_idx_m,
                               kargs.scale_n_ptr + block_idx_n);
        }
        else if(UseDefaultScheduler || (get_warp_id() == 0))
        {
            // Run Epilogue Pipeline
            auto& c_block_window = gemm_tile_windows.at(I3);
            EpiloguePipeline{}(c_block_window, c_block_tile, d_block_window, smem_ptr_ping);
        }
    }

    template <class ScaleM, class ScaleN>
    CK_TILE_DEVICE void operator()(FlatmmKernelArgs<ScaleM, ScaleN, DsDataType::size()> kargs,
                                   int partition_idx = blockIdx.x) const
    {
        int total_work_tile_cnt = TilePartitioner::GridSize(kargs.M, kargs.N);

        do
        {
            const auto [iM, iN] =
                TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(partition_idx);
            const index_t i_m = amd_wave_read_first_lane(iM * TilePartitioner::MPerBlock);
            const index_t i_n = amd_wave_read_first_lane(iN * TilePartitioner::NPerBlock);

            const SplitKBatchOffset splitk_batch_offset(kargs);
            // options
            const ADataType* a_ptr = static_cast<const ADataType*>(kargs.a_ptr) +
                                     splitk_batch_offset.a_k_split_offset / APackedSize;
            const BDataType* b_flat_ptr = static_cast<const BDataType*>(kargs.b_ptr) +
                                          splitk_batch_offset.b_k_split_offset / BPackedSize;
            EDataType* e_ptr = static_cast<EDataType*>(kargs.e_ptr);

            // allocate LDS
            __shared__ char smem_ptr_ping[Underlying::GetSmemPingSize()];
            __shared__ char smem_ptr_pong[Underlying::GetSmemPongSize()];

            if constexpr(!(EpiloguePipeline::MemoryOperation == memory_operation_enum::atomic_add &&
                           EpiloguePipeline::GetVectorSizeC() % 2 != 0 &&
                           is_any_of<EDataType, fp16_t, bf16_t>::value))
            {
                constexpr auto scheduler_type = (FlatmmPipeline::NumWaveGroups == 1);
                RunFlatmm<ScaleM, ScaleN, scheduler_type>(a_ptr,
                                                          b_flat_ptr,
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
