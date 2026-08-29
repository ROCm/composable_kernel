// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp"

namespace ck_tile {

template <index_t NumATensor = 1, index_t NumBTensor = 1, index_t NumDTensor = 0>
struct MxGemmHostArgs : public UniversalGemmHostArgs<NumATensor, NumBTensor, NumDTensor>
{
    using BaseHostArgs = UniversalGemmHostArgs<NumATensor, NumBTensor, NumDTensor>;

    CK_TILE_HOST explicit MxGemmHostArgs(const std::array<const void*, NumATensor>& as_ptr_,
                                         const std::array<const void*, NumATensor>& as_scale_ptr_,
                                         const std::array<const void*, NumBTensor>& bs_ptr_,
                                         const std::array<const void*, NumBTensor>& bs_scale_ptr_,
                                         const std::array<const void*, NumDTensor>& ds_ptr_,
                                         void* e_ptr_,
                                         index_t k_batch_,
                                         index_t M_,
                                         index_t N_,
                                         index_t K_,
                                         const std::array<index_t, NumATensor>& stride_As_,
                                         const std::array<index_t, NumBTensor>& stride_Bs_,
                                         const std::array<index_t, NumDTensor>& stride_Ds_,
                                         index_t stride_E_)
        : BaseHostArgs(as_ptr_,
                       bs_ptr_,
                       ds_ptr_,
                       e_ptr_,
                       k_batch_,
                       M_,
                       N_,
                       K_,
                       stride_As_,
                       stride_Bs_,
                       stride_Ds_,
                       stride_E_),
          as_scale_ptr(as_scale_ptr_),
          bs_scale_ptr(bs_scale_ptr_)
    {
    }

    const std::array<const void*, NumATensor> as_scale_ptr;
    const std::array<const void*, NumBTensor> bs_scale_ptr;
};

template <index_t NumATensor = 1, index_t NumBTensor = 1, index_t NumDTensor = 0>
struct MxGemmKernelArgs : public UniversalGemmKernelArgs<NumATensor, NumBTensor, NumDTensor>
{
    const std::array<const void*, NumATensor> as_scale_ptr;
    const std::array<const void*, NumBTensor> bs_scale_ptr;
};

template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct MxGemmKernel
    : public UniversalGemmKernel<TilePartitioner_,
                                 GemmPipeline_,
                                 EpiloguePipeline_,
                                 MxGemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>>
{
    using BaseKernel =
        UniversalGemmKernel<TilePartitioner_,
                            GemmPipeline_,
                            EpiloguePipeline_,
                            MxGemmKernel<TilePartitioner_, GemmPipeline_, EpiloguePipeline_>>;

    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using MxGemmPipeline   = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;

    using BaseKernel::PersistentKernel;
    using typename BaseKernel::AsLayout;
    using typename BaseKernel::BsLayout;
    using typename BaseKernel::CLayout;
    using typename BaseKernel::DsLayout;

    using typename BaseKernel::ADataType;
    using typename BaseKernel::BDataType;
    using typename BaseKernel::EDataType;

    using BaseKernel::NumATensor;
    using BaseKernel::NumBTensor;
    using BaseKernel::NumDTensor;

    using BaseKernel::GetBlockId;
    using BaseKernel::GetGridSize;
    using BaseKernel::GetNumTiles;
    using BaseKernel::GetSmemSize;
    using typename BaseKernel::SplitKBatchOffset;

    using BaseKernel::APackedSize;
    using BaseKernel::BPackedSize;

    using BaseKernel::I1;

    using AElementWise = remove_cvref_t<typename MxGemmPipeline::AElementWise>;
    using BElementWise = remove_cvref_t<typename MxGemmPipeline::BElementWise>;

    using BlockGemmShape = remove_cvref_t<typename MxGemmPipeline::BlockGemmShape>;

    static constexpr int MThreadPerXdl = BlockGemmShape::WarpTile::at(number<0>{});
    static constexpr int NThreadPerXdl = BlockGemmShape::WarpTile::at(number<1>{});

    static constexpr int BlockScaleSize = MxGemmPipeline::ScaleBlockSize;
    using ScalePtrType                  = const int32_t*;
    // Padding flags pulled from pipeline so the kernel can pad the (unscaled) C and scale views
    // consistently with the A/B views that the pipeline already pads via
    // Underlying::MakeA/BBlockWindows.
    static constexpr bool kPadM = MxGemmPipeline::kPadM;
    static constexpr bool kPadN = MxGemmPipeline::kPadN;
    static constexpr bool kPadK = MxGemmPipeline::kPadK;

    // ------------------------------------------------------------------
    // Compile-time padding-support invariants for the MX comp-async pipeline.
    //
    //   - K padding is NOT supported: async_load_tile issues vector buffer reads whose
    //     OOB check is per-vector-start, so a vector that straddles the K pad boundary
    //     pulls in data from the adjacent row / next K tile rather than zero. The packed
    //     scale tile has the same vector-load property. Until the async path learns how
    //     to do per-element pad masking, we forbid kPadK at compile time.
    //
    //   - kPadM / kPadN are supported only when the GEMM has at least one full block
    //     along that dimension; the CShuffleEpilogue's LDS shuffle uses thread positions
    //     that do not all participate when the entire dimension is smaller than a tile
    //     (resulting in zeros being written into in-range output rows). The "entire
    //     dimension < tile" case is rejected at runtime in IsSupportedArgument; we
    //     cannot statically catch it because M and N are runtime values.
    // ------------------------------------------------------------------
    static_assert(!kPadK,
                  "MX GEMM (comp-async pipeline): K padding (kPadK = true) is not supported. "
                  "The async vector loads do not mask elements that straddle the K pad "
                  "boundary, so partial K tiles produce silently wrong results. Choose K so "
                  "that K is a multiple of KPerBlock * k_batch.");

    // Single source of truth for the split-K atomic-add precondition, shared by the runtime
    // check in IsSupportedArgument and the atomic_add dispatch in operator(). Split-K
    // accumulates each k_id's partial C tile with atomic_add; the CShuffle epilogue can only
    // emit atomic_add for fp16/bf16 outputs when the C vector size is even. For an odd vector
    // size that combination is not instantiated, so such a config cannot run split-K. For all
    // shipped tile shapes GetVectorSizeC() is even, so this is defensive rather than reachable.
    static constexpr bool kSplitKAtomicAddSupported =
        EpiloguePipeline::GetVectorSizeC() % 2 == 0 || !is_any_of<EDataType, fp16_t, bf16_t>::value;

    static constexpr index_t MXdlPackEff = MxGemmPipeline::MXdlPackEff;
    static constexpr index_t NXdlPackEff = MxGemmPipeline::NXdlPackEff;
    static constexpr index_t KXdlPackEff = MxGemmPipeline::KXdlPackEff;

    using KernelArgs = MxGemmKernelArgs<NumATensor, NumBTensor, NumDTensor>;

    CK_TILE_HOST static constexpr KernelArgs
    MakeKernelArgs(const MxGemmHostArgs<NumATensor, NumBTensor, NumDTensor>& hostArgs)
    {
        return KernelArgs{{hostArgs.as_ptr,
                           hostArgs.bs_ptr,
                           hostArgs.ds_ptr,
                           hostArgs.e_ptr,
                           hostArgs.M,
                           hostArgs.N,
                           hostArgs.K,
                           hostArgs.stride_As,
                           hostArgs.stride_Bs,
                           hostArgs.stride_Ds,
                           hostArgs.stride_E,
                           hostArgs.k_batch,
                           hostArgs.async_input_scheduler},
                          hostArgs.as_scale_ptr,
                          hostArgs.bs_scale_ptr};
    }

    CK_TILE_HOST static bool IsSupportedArgument(const KernelArgs& kargs)
    {
        const bool log = ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING));

        if(kargs.k_batch < 1)
        {
            if(log)
                CK_TILE_ERROR("MX GEMM: k_batch must be >= 1.");
            return false;
        }

        // Split-K derives this k_id's logical K start from the row-major SplitKBatchOffset
        // (as_k_split_offset[0]) to offset the packed-scale / flat-B windows; for column-major A
        // that field is stride-scaled, so split-K with non-row-major A is not yet supported.
        // (k_batch == 1 is unaffected -- the offset is 0 and unused.) When col-major A lands for
        // non-preshuffle, extend the split-K K-offset here instead of this reject.
        using ALayout = remove_cvref_t<std::tuple_element_t<0, AsLayout>>;
        if constexpr(!std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.k_batch > 1)
            {
                if(log)
                    CK_TILE_ERROR("MX GEMM: split-K (k_batch > 1) currently requires row-major A.");
                return false;
            }
        }

        // Scales are granular in K: each packed int32_t covers BlockScaleSize * KXdlPackEff
        // consecutive K elements. Every split-K boundary must land on that granularity so that
        // each split can compute a packed-scale K offset. K1 is the WarpTile K, which is a
        // multiple of that granularity for all shipped configs, but be defensive.
        constexpr index_t scale_granularity_k = BlockScaleSize * KXdlPackEff;
        if(kargs.k_batch > 1)
        {
            // splitk_batch_offset allocates K in units of K1 (warp-tile K). If K1 itself is
            // not a multiple of the scale granularity, split-K is not safe.
            constexpr index_t K1 = BlockGemmShape::WarpTile::at(number<2>{});
            static_assert(K1 % scale_granularity_k == 0,
                          "MX GEMM: WarpTile K must be a multiple of BlockScaleSize * KXdlPack "
                          "to support split-K.");
            // Defensive runtime check: K must split evenly along K1 boundaries so that each
            // k_id consumes a whole number of warp-tile K chunks (and therefore a whole
            // number of packed-scale K elements).
            if(kargs.K % (K1 * kargs.k_batch) != 0)
            {
                if(log)
                    CK_TILE_ERROR("MX GEMM: with k_batch > 1, K must be a multiple of WarpTile_K * "
                                  "k_batch so that every split lands on a packed-scale boundary.");
                return false;
            }
        }

        // Delegate the remaining shape/vector-size checks to the universal kernel.
        return BaseKernel::IsSupportedArgument(kargs);
    }

    template <typename KernelArgs>
    CK_TILE_DEVICE static auto
    MakeScaleABlockWindow(const std::array<ScalePtrType, NumATensor>& as_scale_ptr,
                          const KernelArgs& kargs,
                          index_t block_idx_m,
                          const index_t k_elem_offset = 0)
    {
        const auto&& scale_packs_m = integer_divide_ceil(kargs.M, MThreadPerXdl * MXdlPackEff);
        const auto&& scale_packs_k = kargs.K / BlockScaleSize / KXdlPackEff;

        // For split-K (k_batch > 1) advance the scale origin into this k_id's packed-K slice.
        const index_t k_scale_offset = k_elem_offset / BlockScaleSize / KXdlPackEff;

        // Scale16: descriptor order [packs_m, MThreadPerXdl, packs_k] -- K contiguous per M-row,
        //          no pre-shuffle needed (natural row-major layout matches).
        // Scale32: descriptor order [packs_m, packs_k, MThreadPerXdl] -- original layout,
        //          requires pre-shuffle to match.
        const auto scale_a_naive_desc = [&]() {
            if constexpr(BlockScaleSize == 16)
                return make_naive_tensor_descriptor_packed(
                    make_tuple(scale_packs_m, MThreadPerXdl, scale_packs_k));
            else
                return make_naive_tensor_descriptor_packed(
                    make_tuple(scale_packs_m, scale_packs_k, MThreadPerXdl));
        }();
        const auto scale_a_desc = [&]() {
            if constexpr(BlockScaleSize == 16)
                return transform_tensor_descriptor(
                    scale_a_naive_desc,
                    make_tuple(make_merge_transform(make_tuple(scale_packs_m, MThreadPerXdl)),
                               make_pass_through_transform(scale_packs_k)),
                    make_tuple(sequence<0, 1>{}, sequence<2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            else
                return transform_tensor_descriptor(
                    scale_a_naive_desc,
                    make_tuple(make_merge_transform(make_tuple(scale_packs_m, MThreadPerXdl)),
                               make_pass_through_transform(scale_packs_k)),
                    make_tuple(sequence<0, 2>{}, sequence<1>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
        }();
        const auto& scale_a_tensor_view = generate_tuple(
            [&](auto i) {
                return make_tensor_view<address_space_enum::global>(as_scale_ptr[i], scale_a_desc);
            },
            number<NumATensor>{});

        // Pad the scale view so partial trailing tiles along M are handled safely (OOB scale
        // loads return zero; with A also zero on the padded region the contribution is zero
        // regardless of scale value). kPadK is statically disabled, so K never actually pads.
        const auto& scale_a_pad_view = generate_tuple(
            [&](auto i) {
                return pad_tensor_view(
                    scale_a_tensor_view[i],
                    make_tuple(number<TilePartitioner::MPerBlock / MXdlPackEff>{},
                               number<TilePartitioner::KPerBlock / BlockScaleSize / KXdlPackEff>{}),
                    sequence<kPadM, kPadK>{});
            },
            number<NumATensor>{});

        const auto& scale_a_block_window = generate_tuple(
            [&](auto i) {
                return make_tile_window(
                    scale_a_pad_view[i],
                    make_tuple(
                        number<TilePartitioner::MPerBlock / MXdlPackEff>{},
                        number<TilePartitioner::KPerBlock / (BlockScaleSize * KXdlPackEff)>{}),
                    {block_idx_m / MXdlPackEff, k_scale_offset});
            },
            number<NumATensor>{});

        return scale_a_block_window;
    }

    template <typename KernelArgs>
    CK_TILE_DEVICE static auto
    MakeScaleBBlockWindow(const std::array<ScalePtrType, NumBTensor>& bs_scale_ptr,
                          const KernelArgs& kargs,
                          index_t block_idx_n,
                          const index_t k_elem_offset = 0)
    {
        const auto&& scale_packs_n = integer_divide_ceil(kargs.N, NThreadPerXdl * NXdlPackEff);
        const auto&& scale_packs_k = kargs.K / BlockScaleSize / KXdlPackEff;

        // For split-K (k_batch > 1) advance the scale origin into this k_id's packed-K slice.
        const index_t k_scale_offset = k_elem_offset / BlockScaleSize / KXdlPackEff;

        const auto scale_b_naive_desc = [&]() {
            if constexpr(BlockScaleSize == 16)
                return make_naive_tensor_descriptor_packed(
                    make_tuple(scale_packs_n, NThreadPerXdl, scale_packs_k));
            else
                return make_naive_tensor_descriptor_packed(
                    make_tuple(scale_packs_n, scale_packs_k, NThreadPerXdl));
        }();
        const auto scale_b_desc = [&]() {
            if constexpr(BlockScaleSize == 16)
                return transform_tensor_descriptor(
                    scale_b_naive_desc,
                    make_tuple(make_merge_transform(make_tuple(scale_packs_n, NThreadPerXdl)),
                               make_pass_through_transform(scale_packs_k)),
                    make_tuple(sequence<0, 1>{}, sequence<2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            else
                return transform_tensor_descriptor(
                    scale_b_naive_desc,
                    make_tuple(make_merge_transform(make_tuple(scale_packs_n, NThreadPerXdl)),
                               make_pass_through_transform(scale_packs_k)),
                    make_tuple(sequence<0, 2>{}, sequence<1>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
        }();
        const auto& scale_b_tensor_view = generate_tuple(
            [&](auto i) {
                return make_tensor_view<address_space_enum::global>(bs_scale_ptr[i], scale_b_desc);
            },
            number<NumBTensor>{});

        // Pad the scale view so partial trailing tiles along N are handled safely (OOB scale
        // loads return zero; with B also zero on the padded region the contribution is zero
        // regardless of scale value). kPadK is statically disabled, so K never actually pads.
        const auto& scale_b_pad_view = generate_tuple(
            [&](auto i) {
                return pad_tensor_view(
                    scale_b_tensor_view[i],
                    make_tuple(number<TilePartitioner::NPerBlock / NXdlPackEff>{},
                               number<TilePartitioner::KPerBlock / BlockScaleSize / KXdlPackEff>{}),
                    sequence<kPadN, kPadK>{});
            },
            number<NumBTensor>{});

        const auto& scale_b_block_window = generate_tuple(
            [&](auto i) {
                return make_tile_window(
                    scale_b_pad_view[i],
                    make_tuple(
                        number<TilePartitioner::NPerBlock / NXdlPackEff>{},
                        number<TilePartitioner::KPerBlock / (BlockScaleSize * KXdlPackEff)>{}),
                    {block_idx_n / NXdlPackEff, k_scale_offset});
            },
            number<NumBTensor>{});
        return scale_b_block_window;
    }

    CK_TILE_DEVICE static auto
    MakeBFlatBlockWindows(const std::array<const BDataType*, NumBTensor>& bs_ptr,
                          const KernelArgs& kargs,
                          const index_t i_n,
                          const index_t k_elem_offset = 0)
    {
        static_assert(NumBTensor == 1, "MX GEMM preshuffle currently supports one B tensor");

        constexpr index_t kKPerBlock    = MxGemmPipeline::kKPerBlock;
        constexpr index_t kNWarpTile    = BlockGemmShape::WarpTile::at(I1);
        constexpr index_t flatKPerBlock = kKPerBlock * kNWarpTile;
        const index_t kFlatKBlocks      = kargs.K / kKPerBlock;
        const index_t kFlatN            = kargs.N / kNWarpTile;

        const index_t k_flat_offset = (k_elem_offset / kKPerBlock) * flatKPerBlock;

        auto b_flat_tensor_view = [&]() {
            static_assert(flatKPerBlock % MxGemmPipeline::GetVectorSizeB() == 0,
                          "wrong! vector size for preshuffled B tensor");
            auto naive_desc = make_naive_tensor_descriptor_packed(
                make_tuple(kFlatN, kFlatKBlocks, number<flatKPerBlock>{}));
            auto desc = transform_tensor_descriptor(
                naive_desc,
                make_tuple(make_pass_through_transform(kFlatN),
                           make_merge_transform_v3_division_mod(
                               make_tuple(kFlatKBlocks, number<flatKPerBlock>{}))),
                make_tuple(sequence<0>{}, sequence<1, 2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
            return make_tensor_view<address_space_enum::global>(bs_ptr[number<0>{}], desc);
        }();

        return generate_tuple(
            [&](auto) {
                return make_tile_window(b_flat_tensor_view,
                                        make_tuple(number<MxGemmPipeline::flatNPerWarp>{},
                                                   number<MxGemmPipeline::flatKPerWarp>{}),
                                        {static_cast<int>(i_n / BlockGemmShape::WarpTile::at(I1)),
                                         static_cast<int>(k_flat_offset)});
            },
            number<NumBTensor>{});
    }

    template <memory_operation_enum DstInMemOp>
    CK_TILE_DEVICE static void RunGemm(const std::array<const ADataType*, NumATensor>& as_ptr,
                                       const std::array<const BDataType*, NumBTensor>& bs_ptr,
                                       const std::array<const void*, NumDTensor>& ds_ptr,
                                       EDataType* e_ptr,
                                       void* smem_ptr,
                                       KernelArgs kargs,
                                       const SplitKBatchOffset& splitk_batch_offset,
                                       const index_t block_idx_m,
                                       const index_t block_idx_n,
                                       const index_t k_elem_offset = 0)
    {
        std::array<ScalePtrType, NumATensor> as_scale_ptr;
        std::array<const ADataType*, NumATensor> as_ptr_;
        index_t block_idx_m_;
        // Large tensor support (when M is large, N and K are relatively small)
        using ALayout = remove_cvref_t<std::tuple_element_t<0, AsLayout>>;
        constexpr bool offset_ptrs_by_tile_coords =
            std::is_same_v<tensor_layout::gemm::RowMajor, ALayout> &&
            std::is_same_v<tensor_layout::gemm::RowMajor, CLayout> && !BaseKernel::ClusterLaunch;

        if constexpr(offset_ptrs_by_tile_coords)
        {
            static_for<0, NumATensor, 1>{}([&](auto i) {
                as_ptr_[i] = as_ptr[i] + static_cast<std::ptrdiff_t>(block_idx_m) *
                                             kargs.stride_As[i] / APackedSize;
            });
            e_ptr += static_cast<std::ptrdiff_t>(block_idx_m) * kargs.stride_E;
            static_for<0, NumATensor, 1>{}([&](auto i) {
                as_scale_ptr[i] = reinterpret_cast<ScalePtrType>(kargs.as_scale_ptr[i]) +
                                  static_cast<std::ptrdiff_t>(block_idx_m / MXdlPackEff) *
                                      (kargs.K / BlockScaleSize / KXdlPackEff);
            });

            kargs.M      = std::min(kargs.M - block_idx_m, TilePartitioner::MPerBlock);
            block_idx_m_ = 0;
        }
        else
        {
            static_for<0, NumATensor, 1>{}([&](auto i) {
                as_scale_ptr[i] = reinterpret_cast<ScalePtrType>(kargs.as_scale_ptr[i]);
            });
            static_for<0, NumATensor, 1>{}([&](auto i) { as_ptr_[i] = as_ptr[i]; });
            block_idx_m_ = block_idx_m;
        }

        std::array<ScalePtrType, NumBTensor> bs_scale_ptr;
        static_for<0, NumBTensor, 1>{}([&](auto i) {
            bs_scale_ptr[i] = reinterpret_cast<ScalePtrType>(kargs.bs_scale_ptr[i]);
        });

        // cluster launch pads grid to cluster boundaries; skip out-of-bound blocks
        if constexpr(BaseKernel::ClusterLaunch)
        {
            if(block_idx_m_ >= kargs.M || block_idx_n >= kargs.N)
                return;
        }

        // The preshuffle A async-load (MakeMX_AAsyncLoadBytesDramWindow) rebuilds the A
        // view with a packed descriptor, i.e. it assumes the leading (M) stride equals
        // the view's K extent. That only holds when the extent equals stride_A, which is
        // the case for k_batch == 1 (splitted_k == K) but NOT for split-K (splitted_k < K):
        // a packed extent of splitted_k would stride M by splitted_k instead of stride_A
        // and read the wrong rows (only row 0 lands correctly). Use the full K extent so
        // the packed M stride matches stride_A. The as_ptr K-offset already selects this
        // k_id's slice and num_loop bounds the blocks read, so reads stay within
        // [as_k_split_offset, as_k_split_offset + splitted_k) <= K (in-allocation).
        const auto& as_block_window = [&]() {
            if constexpr(MxGemmPipeline::Preshuffle)
            {
                return BaseKernel::MakeABlockWindows(as_ptr_, kargs, kargs.K, block_idx_m_);
            }
            else
            {
                return BaseKernel::MakeABlockWindows(
                    as_ptr_, kargs, splitk_batch_offset.splitted_k, block_idx_m_);
            }
        }();
        const auto& bs_block_window = [&]() {
            if constexpr(MxGemmPipeline::Preshuffle)
            {
                return MakeBFlatBlockWindows(bs_ptr, kargs, block_idx_n, k_elem_offset);
            }
            else
            {
                return BaseKernel::MakeBBlockWindows(
                    bs_ptr, kargs, splitk_batch_offset.splitted_k, block_idx_n);
            }
        }();
        const auto& ds_block_window =
            BaseKernel::MakeDBlockWindows(ds_ptr, kargs, block_idx_m_, block_idx_n);

        // Create scale block windows. For split-K (k_batch > 1), k_elem_offset advances the
        // scale origin into the correct packed-K slice for this k_id; otherwise it is zero.
        const auto& scale_a_block_window =
            MakeScaleABlockWindow(as_scale_ptr, kargs, block_idx_m_, k_elem_offset);
        const auto& scale_b_block_window =
            MakeScaleBBlockWindow(bs_scale_ptr, kargs, block_idx_n, k_elem_offset);

        const index_t num_loop =
            amd_wave_read_first_lane(TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k));

        const auto& c_block_tile = MxGemmPipeline{}.template operator()(as_block_window,
                                                                        AElementWise{},
                                                                        bs_block_window,
                                                                        BElementWise{},
                                                                        scale_a_block_window,
                                                                        scale_b_block_window,
                                                                        num_loop,
                                                                        smem_ptr);

        // Dispatch epilogue: when k_batch > 1 each split accumulates a partial result into
        // the same C tile, so we need atomic add (universal_gemm_kernel pattern). The
        // fp16/bf16 even-vector-size precondition is captured once in kSplitKAtomicAddSupported
        // and also rejected up front in IsSupportedArgument.
        // if(k_batch == 1)
        auto c_block_window = BaseKernel::template MakeCBlockWindows<DstInMemOp>(
            e_ptr, kargs, block_idx_m_, block_idx_n);
        EpiloguePipeline{}(c_block_window, c_block_tile, ds_block_window, smem_ptr);
    }

    CK_TILE_DEVICE static void RunGemm(const std::array<const ADataType*, NumATensor>& as_ptr,
                                       const std::array<const BDataType*, NumBTensor>& bs_ptr,
                                       const std::array<const void*, NumDTensor>& ds_ptr,
                                       EDataType* e_ptr,
                                       void* smem_ptr,
                                       const KernelArgs& kargs,
                                       const SplitKBatchOffset& splitk_batch_offset,
                                       const index_t block_idx_m,
                                       const index_t block_idx_n)
    {
        if(kargs.k_batch == 1)
        {
            RunGemm<memory_operation_enum::set>(as_ptr,
                                                bs_ptr,
                                                ds_ptr,
                                                e_ptr,
                                                smem_ptr,
                                                kargs,
                                                splitk_batch_offset,
                                                block_idx_m,
                                                block_idx_n);
        }
        else
        {
            // This k_id's logical K-element start. For row-major A, as_k_split_offset[0] is exactly
            // that offset, so reuse it rather than recomputing the split formula; the packed-scale
            // and flat-B K offsets are derived from it. Split-K with non-row-major A is rejected in
            // IsSupportedArgument; for k_batch == 1 this value is 0 and unused for any layout.
            const index_t k_elem_offset =
                amd_wave_read_first_lane(splitk_batch_offset.as_k_split_offset[number<0>{}]);
            RunGemm<memory_operation_enum::atomic_add>(as_ptr,
                                                       bs_ptr,
                                                       ds_ptr,
                                                       e_ptr,
                                                       smem_ptr,
                                                       kargs,
                                                       splitk_batch_offset,
                                                       block_idx_m,
                                                       block_idx_n,
                                                       k_elem_offset);
        }
    }
};

} // namespace ck_tile
