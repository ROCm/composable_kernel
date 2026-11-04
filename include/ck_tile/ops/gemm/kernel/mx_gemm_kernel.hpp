// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp"

namespace ck_tile {

/**
 * @brief Host arguments of MxGemmKernel: the UniversalGemmHostArgs plus the A and B scale pointers.
 * @note On the gfx125 TDM pipelines, IsMxTdmShapeSupported describes the scale layout the kernel
 *       reads, the K values each configuration accepts, and the scale padding a K tail (K not a
 *       multiple of KPerBlock) needs.
 */
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

namespace detail {
// The operand types MxGemmKernel takes K tails for on the TDM pipelines.
template <typename T>
inline constexpr bool is_mx_tdm_k_tail_operand_v =
    std::is_same_v<T, fp8_t> || std::is_same_v<T, pk_fp4_t>;
} // namespace detail

/**
 * @brief Whether these operands, layouts and scales are K-tail configurations, for which
 *        MxGemmKernel takes K tails on the gfx125 TDM pipelines (see IsMxTdmShapeSupported).
 * @details The K-tail configurations: fp8_t or pk_fp4_t on each side (any pair), 32-element
 *          scales packed four to an int32, a row-major A and a column-major B, so that K is the
 *          contiguous dimension of both. IsMxTdmShapeSupported also requires a KPerBlock that is
 *          a multiple of ScaleBlockSize * KXdlPack; MxGroupedGemmKernel takes no K tails.
 * @tparam ADataType      Element type of A (cv and reference qualifiers are ignored).
 * @tparam BDataType      Element type of B (cv and reference qualifiers are ignored).
 * @tparam ALayout        Layout of A.
 * @tparam BLayout        Layout of B.
 * @tparam ScaleBlockSize Elements per scale.
 * @tparam KXdlPack       Scale blocks per packed int32 of scales.
 */
template <typename ADataType,
          typename BDataType,
          typename ALayout,
          typename BLayout,
          index_t ScaleBlockSize,
          index_t KXdlPack>
inline constexpr bool is_mx_tdm_k_tail_config_v =
    detail::is_mx_tdm_k_tail_operand_v<remove_cvref_t<ADataType>> &&
    detail::is_mx_tdm_k_tail_operand_v<remove_cvref_t<BDataType>> && ScaleBlockSize == 32 &&
    KXdlPack == 4 && std::is_same_v<ALayout, tensor_layout::gemm::RowMajor> &&
    std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>;

/**
 * @brief Host shape checks for MX GEMM on the gfx125 TDM pipelines.
 * @details For TDM pipelines UniversalGemmKernel::IsSupportedArgument checks only tensor size and
 *          split-K and skips its shape checks, so MxGemmKernel applies these checks instead.
 *
 *          Scale layout. The A scales are an M' x S matrix and the B scales an N' x S matrix, one
 *          row per row of A or column of B, where M' (N') is M (N) rounded up to a multiple of
 *          the warp tile M (N) (MxGemmKernel::MThreadPerXdl and NThreadPerXdl); the rows past M
 *          (N) are padding rows. S is ceil(K / ScaleBlockSize) scale blocks rounded up to a
 *          multiple of KXdlPack, one int32 scale pack per ScaleBlockSize * KXdlPack elements of K;
 *          the columns past ceil(K / ScaleBlockSize) are padding columns. Both matrices are
 *          pre-shuffled with preShuffleScaleBuffer_gfx1250<ScaleType, ScaleBlockSize, true>
 *          (ck_tile/host/mx_processing.hpp), passing the padded row count, S and the warp tile
 *          M (N), and the device buffers hold every padding row and column, because the kernel
 *          can read any of them.
 *
 *          K tails. A K tail is a K that is not a multiple of KPerBlock, so the last K tile is
 *          partial. With AllowKTail and a KPerBlock that is a multiple of
 *          ScaleBlockSize * KXdlPack, the K-tail configurations (is_mx_tdm_k_tail_config_v)
 *          accept a K that is a multiple of the PackedSize of A and of B (any K for fp8_t, an even
 *          K for pk_fp4_t); it need not be a multiple of ScaleBlockSize, of
 *          ScaleBlockSize * KXdlPack or of KPerBlock. The TDM writes zeros to LDS for the A and B
 *          elements past K, so a partial last K tile needs no masking. When K is not a multiple of
 *          ScaleBlockSize, the last scale block covers only the remaining K % ScaleBlockSize
 *          elements. Every scale must be finite, padding rows and columns included: the kernel
 *          multiplies the zeros past K by the padding columns of the last scale pack and, when a K
 *          tile holds more than one scale pack, by scales of later rows of A (columns of B) (see
 *          MxGemmKernel::GetScalePacksK). NaN x 0 is NaN, so a NaN scale there turns the result of
 *          its own row (column), or of an earlier one, into NaN. Pad with a finite value, for
 *          example 0x00, which is finite for e8m0_t, e5m3_t and e4m3_t (0x7F is 1.0 for e8m0_t but
 *          NaN for e4m3_t). The scale types default-construct to their NaN encoding, so padding
 *          left at its default value is not finite.
 *
 *          The checks reject:
 *          - K <= 0;
 *          - K > INT_MAX - 2 * max(KPerBlock, ScaleBlockSize * KXdlPack): the rounded-up K loop and
 *            scale pack counts, or the K window offset the pipelines reach after the last K tile,
 *            (GetLoopNum(K) + 1) * KPerBlock, would overflow index_t;
 *          - without K-tail support (AllowKTail false, a configuration outside
 *            is_mx_tdm_k_tail_config_v, or a KPerBlock that is not a multiple of
 *            ScaleBlockSize * KXdlPack): K not a multiple of ScaleBlockSize * KXdlPack or of
 *            KPerBlock (a caller pads K to a multiple of both, with zero A and B elements and
 *            finite scales);
 *          - for a packed A or B (PackedSize > 1, such as pk_fp4_t or pk_fp6x16_t), a stride or a
 *            contiguous extent that is not a whole number of packs: the TDM load
 *            (tile_window::tdm_load_to_lds) divides both by PackedSize and rounds down. The
 *            contiguous extent is K for a row-major A and a column-major B, M for a column-major
 *            A and N for a row-major B.
 *          Apart from that, M and N are not checked: TDM clips partial tiles in hardware.
 *          MxGemmKernel::IsTdmShapeSupported adds one check that depends on M and N: the B-scale
 *          matrix (N' x S), and the A-scale matrix (M' x S, or one M tile of it when the kernel
 *          shifts the A-scale pointer per M tile), must each be smaller than 2^31 bytes.
 *          The function takes the operand types and tile constants rather than a kernel, so it
 *          can also be checked for operand types that the TDM pipelines cannot instantiate yet.
 * @tparam ADataType      Type of A.
 * @tparam BDataType      Type of B.
 * @tparam ALayout        Layout of A (tensor_layout::gemm::RowMajor or ColumnMajor).
 * @tparam BLayout        Layout of B (tensor_layout::gemm::RowMajor or ColumnMajor).
 * @tparam KPerBlock      K tile size.
 * @tparam ScaleBlockSize Elements per scale (16 or 32).
 * @tparam KXdlPack       Scale blocks per packed int32 of scales (the pipeline's KXdlPackEff).
 * @tparam AllowKTail     Whether the K-tail configurations accept K that is not a multiple of
 *                        ScaleBlockSize * KXdlPack or of KPerBlock, when KPerBlock is a multiple
 *                        of ScaleBlockSize * KXdlPack. The default, false, keeps
 *                        those checks for every configuration. MxGemmKernel::IsSupportedArgument
 *                        passes true; MxGroupedGemmKernel::IsSupportedArgument also checks every
 *                        group with false.
 * @param M        Rows of A and C.
 * @param N        Columns of B and C.
 * @param K        Reduction length.
 * @param stride_A Stride of A in elements, as in MxGemmHostArgs.
 * @param stride_B Stride of B in elements, as in MxGemmHostArgs.
 * @return true if the shape passes these checks.
 */
template <typename ADataType,
          typename BDataType,
          typename ALayout,
          typename BLayout,
          index_t KPerBlock,
          index_t ScaleBlockSize,
          index_t KXdlPack,
          bool AllowKTail = false>
CK_TILE_HOST bool
IsMxTdmShapeSupported(index_t M, index_t N, index_t K, index_t stride_A, index_t stride_B)
{
    static_assert(KPerBlock > 0 && ScaleBlockSize > 0 && KXdlPack > 0);
    using A = remove_cvref_t<ADataType>;
    using B = remove_cvref_t<BDataType>;
    constexpr bool kKTailSupported =
        AllowKTail && is_mx_tdm_k_tail_config_v<A, B, ALayout, BLayout, ScaleBlockSize, KXdlPack> &&
        KPerBlock % (ScaleBlockSize * KXdlPack) == 0;
    const auto reject = [](const auto&... reason) {
        if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
        {
            CK_TILE_ERROR("MX GEMM with a TDM pipeline ", reason..., "!");
        }
        return false;
    };
    if(K <= 0)
    {
        return reject("requires K > 0");
    }
    // GetLoopNum and MxGemmKernel::GetScalePacksK round K up as (K + d - 1) / d with d = KPerBlock
    // and d = ScaleBlockSize * KXdlPack respectively, and the pipelines, which load one K tile past
    // the last when the loop count is odd, move their K windows up to
    // (GetLoopNum(K) + 1) * KPerBlock <= K + 2 * KPerBlock - 1; none of them may overflow index_t.
    constexpr index_t kKRoundUp = max(KPerBlock, ScaleBlockSize * KXdlPack);
    constexpr index_t kMaxK     = numeric<index_t>::max() - 2 * kKRoundUp;
    if(K > kMaxK)
    {
        return reject("requires K <= INT_MAX - 2 * max(KPerBlock, ScaleBlockSize * KXdlPack) = ",
                      kMaxK);
    }
    if constexpr(!kKTailSupported)
    {
        if(K % (ScaleBlockSize * KXdlPack) != 0)
        {
            return reject("requires K to be a multiple of ScaleBlockSize * KXdlPack = ",
                          ScaleBlockSize * KXdlPack);
        }
        if(K % KPerBlock != 0)
        {
            return reject("requires K to be a multiple of KPerBlock = ", KPerBlock);
        }
    }
    if constexpr(numeric_traits<A>::PackedSize > 1)
    {
        constexpr index_t pack = numeric_traits<A>::PackedSize;
        if(stride_A % pack != 0)
        {
            return reject("requires the stride of a packed A to be a multiple of its pack size ",
                          pack);
        }
        if(std::is_same_v<ALayout, tensor_layout::gemm::ColumnMajor> && M % pack != 0)
        {
            return reject(
                "requires M of a column-major packed A to be a multiple of its pack size ", pack);
        }
        if(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor> && K % pack != 0)
        {
            return reject("requires K of a row-major packed A to be a multiple of its pack size ",
                          pack);
        }
    }
    if constexpr(numeric_traits<B>::PackedSize > 1)
    {
        constexpr index_t pack = numeric_traits<B>::PackedSize;
        if(stride_B % pack != 0)
        {
            return reject("requires the stride of a packed B to be a multiple of its pack size ",
                          pack);
        }
        if(std::is_same_v<BLayout, tensor_layout::gemm::RowMajor> && N % pack != 0)
        {
            return reject("requires N of a row-major packed B to be a multiple of its pack size ",
                          pack);
        }
        if(std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor> && K % pack != 0)
        {
            return reject(
                "requires K of a column-major packed B to be a multiple of its pack size ", pack);
        }
    }
    return true;
}

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
    // size that combination is not instantiated, so such a config cannot run split-K.
    // Epilogues must declare atomic support; TDM stores cannot reduce split-K outputs.
    template <typename T>
    using AtomicAddRequiresEvenVectorSize = decltype(T::kAtomicAddRequiresEvenVectorSize);

    static constexpr bool kSplitKAtomicAddSupported = [] {
        if constexpr(is_detected<AtomicAddRequiresEvenVectorSize, EpiloguePipeline>::value)
            return !EpiloguePipeline::kAtomicAddRequiresEvenVectorSize ||
                   EpiloguePipeline::GetVectorSizeC() % 2 == 0;
        else
            return false; // Unmarked epilogues (including TDM) cannot reduce split-K outputs.
    }();

    static constexpr index_t MXdlPackEff = MxGemmPipeline::MXdlPackEff;
    static constexpr index_t NXdlPackEff = MxGemmPipeline::NXdlPackEff;
    static constexpr index_t KXdlPackEff = MxGemmPipeline::KXdlPackEff;

    // Protects the K loop of the TDM pipelines' block GEMM (BlockGemmARegBRegCRegV1): it runs
    // KPerBlock / (WarpTile K) warp GEMMs per K tile (KIterPerWarp) and drops any remainder, so a
    // K tile that is not a whole number of warp tiles would skip K values in every K tile.
    // IsMxTdmShapeSupported checks K, not KPerBlock against the warp tile.
    static_assert(!BaseKernel::kIsTdmPipeline ||
                      TilePartitioner::KPerBlock % BlockGemmShape::WarpTile::at(number<2>{}) == 0,
                  "MX GEMM (TDM pipelines): KPerBlock must be a multiple of WarpTile K.");

    // Large tensor support (when M is large, N and K are relatively small): RunGemm shifts the
    // A / E / A-scale base pointers by the M tile and clamps kargs.M, so those descriptors span
    // at most one M tile however large M is.
    static constexpr bool kOffsetPtrsByTileCoords =
        std::is_same_v<tensor_layout::gemm::RowMajor,
                       remove_cvref_t<std::tuple_element_t<0, AsLayout>>> &&
        std::is_same_v<tensor_layout::gemm::RowMajor, CLayout> && !BaseKernel::ClusterLaunch;

    // The shift leaves ds_ptr at its base while clamping kargs.M, so the D windows would be built
    // at origin 0 and every workgroup would read D rows [0, MPerBlock). Before enabling D here,
    // shift ds_ptr alongside e_ptr and fold the Ds layouts into kOffsetPtrsByTileCoords.
    static_assert(!kOffsetPtrsByTileCoords || NumDTensor == 0,
                  "MX GEMM: the per-M-tile base-pointer shift does not offset the D pointers.");

    // The shift shifts every A pointer by stride_As[i], which is a row offset only for RowMajor A,
    // but the predicate above inspects AsLayout[0] alone.
    static_assert(!kOffsetPtrsByTileCoords || NumATensor == 1,
                  "MX GEMM: the per-M-tile base-pointer shift assumes a single RowMajor A.");

    CK_TILE_HOST_DEVICE static constexpr bool IsLargeTensorMOffsettingSupported()
    {
        return kOffsetPtrsByTileCoords;
    }

    // Number of int32 scale packs in each row of the packed scale views. Each pack covers
    // BlockScaleSize * KXdlPackEff elements of K; the count rounds up, so it includes a partial
    // last pack when K is not a multiple of BlockScaleSize * KXdlPackEff.
    //
    // On a K tail the kernel reads scales that multiply only the zeros the TDM loads for A and B
    // past K. NaN x 0 is NaN, so every scale the caller passes, padding included, must be finite.
    // Those reads are:
    // - the K padding columns, if any, of the last pack, GetScalePacksK(K) - 1;
    // - on the TDM pipelines, when a K tile holds more than one scale pack and K is not a
    //   multiple of KPerBlock, the packs p >= P = GetScalePacksK(K) that the last K tile can
    //   read, because the scale views are not padded in K. For row r of the view, pack p is pack
    //   p % P of row r + (p / P) * MThreadPerXdl (for B, NThreadPerXdl): a scale of a later row,
    //   which can be one of the caller's M or N padding rows. Reads past the end of the view (for
    //   an A view shifted to its M tile, the end of that tile) return 0.
    // The pipelines also load one scale tile past the last K tile when the loop count is odd;
    // those values are never used.
    CK_TILE_HOST_DEVICE static constexpr index_t GetScalePacksK(index_t K)
    {
        return integer_divide_ceil(K, BlockScaleSize * KXdlPackEff);
    }

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

        if(kargs.k_batch > 1 && !kSplitKAtomicAddSupported)
        {
            if(log)
                CK_TILE_ERROR(
                    "MX GEMM: split-K is unsupported by this epilogue/output vector size.");
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

        if constexpr(BaseKernel::kIsTdmPipeline)
        {
            // UniversalGemmKernel::IsSupportedArgument skips its shape checks for TDM pipelines.
            // The device check comes last so that the argument checks also run on a host without
            // a gfx125 device.
            if(!BaseKernel::IsTdmArgumentSupported("MX GEMM",
                                                   kargs.k_batch,
                                                   kargs.K,
                                                   kargs.as_ptr[0],
                                                   kargs.bs_ptr[0],
                                                   kargs.e_ptr) ||
               !IsTdmShapeSupported(kargs) || !BaseKernel::IsTdmDeviceSupported("MX GEMM"))
            {
                return false;
            }
        }

        // Delegate the remaining shape/vector-size checks to the universal kernel.
        return BaseKernel::IsSupportedArgument(kargs);
    }

    /**
     * @brief Runs the shape checks of IsSupportedArgument: IsMxTdmShapeSupported and
     *        IsScaleViewSizeSupported.
     * @details Skips the split-K, pointer-alignment and device checks and the A, B, D and E size
     *          checks of UniversalGemmKernel::IsSupportedArgument.
     * @tparam AllowKTail Passed to IsMxTdmShapeSupported. The default, true, gives the checks of
     *                    IsSupportedArgument; false keeps the K checks for every configuration, as
     *                    MxGroupedGemmKernel::IsSupportedArgument requires.
     * @param kargs Kernel arguments; M, N, K and the A and B strides are read.
     * @return true if the shape in kargs passes these checks.
     */
    template <bool AllowKTail = true>
    CK_TILE_HOST static bool IsTdmShapeSupported(const KernelArgs& kargs)
    {
        static_assert(BaseKernel::kIsTdmPipeline,
                      "MX GEMM: IsTdmShapeSupported applies to the TDM pipelines only.");
        static_assert(NumATensor == 1 && NumBTensor == 1,
                      "MX GEMM: the TDM pipelines take a single A and a single B tensor.");
        using ALayout = remove_cvref_t<std::tuple_element_t<0, AsLayout>>;
        using BLayout = remove_cvref_t<std::tuple_element_t<0, BsLayout>>;
        // IsMxTdmShapeSupported goes first, so a shape it rejects is logged with its own reason.
        return IsMxTdmShapeSupported<ADataType,
                                     BDataType,
                                     ALayout,
                                     BLayout,
                                     TilePartitioner::KPerBlock,
                                     BlockScaleSize,
                                     KXdlPackEff,
                                     AllowKTail>(
                   kargs.M, kargs.N, kargs.K, kargs.stride_As[0], kargs.stride_Bs[0]) &&
               IsScaleViewSizeSupported(kargs);
    }

    private:
    /**
     * @brief Whether the packed scale views of MakeScaleABlockWindow and MakeScaleBBlockWindow
     *        are smaller than 2^31 bytes.
     * @details The scales are read with buffer loads, which compute byte offsets in index_t, so
     *          the kernel keeps each scale view below 2^31 bytes, the limit
     *          UniversalGemmKernel::IsSupportedArgument applies to A, B, D and E but not to the
     *          scales. A scale view holds GetScalePacksK(K) int32 packs in each of its
     *          ceil(M / (MThreadPerXdl * MXdlPackEff)) * MThreadPerXdl rows (for B: N,
     *          NThreadPerXdl and NXdlPackEff); with MXdlPackEff and NXdlPackEff of 1 on the TDM
     *          pipelines, these are the M' (N') rows of IsMxTdmShapeSupported. A scale view can be
     *          larger than A (B): with a K tail, for an fp8_t B with K < 4, a column of B holds
     *          fewer than 4 bytes and its row of scales holds 4; and as N is rounded up to the
     *          warp tile, with 16-element scales, a 32-wide warp tile and N = 1, the B-scale view
     *          holds 2K bytes for a K-byte B (likewise for A and M). The A-scale view spans one M
     *          tile when RunGemm shifts the A-scale pointer to it (kOffsetPtrsByTileCoords), and
     *          all of M otherwise, as with cluster launch or a column-major A or C.
     * @param kargs Kernel arguments; M, N and K are read.
     * @return true if both scale views are smaller than 2^31 bytes.
     */
    CK_TILE_HOST static bool IsScaleViewSizeSupported(const KernelArgs& kargs)
    {
        // The byte size of a scale view: the rows and packs of its descriptor (see
        // MakeScaleABlockWindow), all in 64 bits, so that no rounding up or product overflows for
        // any M, N and K.
        constexpr long_index_t kScalePackK = BlockScaleSize * KXdlPackEff;
        constexpr long_index_t kScaleBytes = sizeof(remove_pointer_t<ScalePtrType>);
        const long_index_t packs_k =
            integer_divide_ceil(static_cast<long_index_t>(kargs.K), kScalePackK);
        const auto view_bytes = [&](index_t extent, index_t thread_per_xdl, index_t xdl_pack) {
            const long_index_t row_groups =
                integer_divide_ceil(static_cast<long_index_t>(extent),
                                    static_cast<long_index_t>(thread_per_xdl) * xdl_pack);
            return row_groups * thread_per_xdl * packs_k * kScaleBytes;
        };
        const auto reject = [](const char* operand, long_index_t bytes) {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR("MX GEMM with a TDM pipeline requires the packed ",
                              operand,
                              "-scale view to be smaller than 2^31 bytes, but it holds ",
                              bytes,
                              " bytes!");
            }
            return false;
        };
        constexpr long_index_t kSizeLimit = long_index_t{1} << 31;
        const long_index_t a_bytes =
            view_bytes(BaseKernel::ClampMToOffsettedTile(kargs.M, 0), MThreadPerXdl, MXdlPackEff);
        if(a_bytes >= kSizeLimit)
        {
            return reject("A", a_bytes);
        }
        const long_index_t b_bytes = view_bytes(kargs.N, NThreadPerXdl, NXdlPackEff);
        if(b_bytes >= kSizeLimit)
        {
            return reject("B", b_bytes);
        }
        return true;
    }

    public:
    template <typename KernelArgs>
    CK_TILE_DEVICE static auto
    MakeScaleABlockWindow(const std::array<ScalePtrType, NumATensor>& as_scale_ptr,
                          const KernelArgs& kargs,
                          index_t block_idx_m,
                          const index_t k_elem_offset = 0)
    {
        // IsScaleViewSizeSupported recomputes this view's lengths in 64 bits to bound its byte
        // size; change both together.
        const auto&& scale_packs_m = integer_divide_ceil(kargs.M, MThreadPerXdl * MXdlPackEff);
        const auto&& scale_packs_k = GetScalePacksK(kargs.K);

        // For split-K (k_batch > 1) advance the scale origin into this k_id's packed-K slice.
        const index_t k_scale_offset = k_elem_offset / BlockScaleSize / KXdlPackEff;

        // Unified coalesced scale layout for all block sizes: descriptor order
        // [packs_m, packs_k, MThreadPerXdl] (M-thread fastest) so consecutive lanes read
        // consecutive addresses. The host pre-shuffle writes scales in this order.
        const auto scale_a_naive_desc = make_naive_tensor_descriptor_packed(
            make_tuple(scale_packs_m, scale_packs_k, MThreadPerXdl));
        const auto scale_a_desc = transform_tensor_descriptor(
            scale_a_naive_desc,
            make_tuple(make_merge_transform(make_tuple(scale_packs_m, MThreadPerXdl)),
                       make_pass_through_transform(scale_packs_k)),
            make_tuple(sequence<0, 2>{}, sequence<1>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
        const auto& scale_a_tensor_view = generate_tuple(
            [&](auto i) {
                return make_tensor_view<address_space_enum::global>(as_scale_ptr[i], scale_a_desc);
            },
            number<NumATensor>{});

        // Pad the scale view so partial trailing tiles along M are handled safely (OOB scale
        // loads return zero, a finite scale, so with A also zero on the padded region the
        // contribution is zero). kPadK is statically disabled, so K never actually pads; see
        // GetScalePacksK for the scale reads past K on a K tail.
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
        // IsScaleViewSizeSupported recomputes this view's lengths in 64 bits to bound its byte
        // size; change both together.
        const auto&& scale_packs_n = integer_divide_ceil(kargs.N, NThreadPerXdl * NXdlPackEff);
        const auto&& scale_packs_k = GetScalePacksK(kargs.K);

        // For split-K (k_batch > 1) advance the scale origin into this k_id's packed-K slice.
        const index_t k_scale_offset = k_elem_offset / BlockScaleSize / KXdlPackEff;

        const auto scale_b_naive_desc = make_naive_tensor_descriptor_packed(
            make_tuple(scale_packs_n, scale_packs_k, NThreadPerXdl));
        const auto scale_b_desc = transform_tensor_descriptor(
            scale_b_naive_desc,
            make_tuple(make_merge_transform(make_tuple(scale_packs_n, NThreadPerXdl)),
                       make_pass_through_transform(scale_packs_k)),
            make_tuple(sequence<0, 2>{}, sequence<1>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
        const auto& scale_b_tensor_view = generate_tuple(
            [&](auto i) {
                return make_tensor_view<address_space_enum::global>(bs_scale_ptr[i], scale_b_desc);
            },
            number<NumBTensor>{});

        // Pad the scale view so partial trailing tiles along N are handled safely (OOB scale
        // loads return zero, a finite scale, so with B also zero on the padded region the
        // contribution is zero). kPadK is statically disabled, so K never actually pads; see
        // GetScalePacksK for the scale reads past K on a K tail.
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

        if constexpr(kOffsetPtrsByTileCoords)
        {
            static_for<0, NumATensor, 1>{}([&](auto i) {
                as_ptr_[i] = as_ptr[i] + static_cast<std::ptrdiff_t>(block_idx_m) *
                                             kargs.stride_As[i] / APackedSize;
            });
            e_ptr += static_cast<std::ptrdiff_t>(block_idx_m) * kargs.stride_E;
            static_for<0, NumATensor, 1>{}([&](auto i) {
                as_scale_ptr[i] = reinterpret_cast<ScalePtrType>(kargs.as_scale_ptr[i]) +
                                  static_cast<std::ptrdiff_t>(block_idx_m / MXdlPackEff) *
                                      GetScalePacksK(kargs.K);
            });

            kargs.M      = BaseKernel::ClampMToOffsettedTile(kargs.M, block_idx_m);
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
        else if constexpr(kSplitKAtomicAddSupported)
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
