// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/convolution_parameter.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

#if __clang_major__ >= 23
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"
#endif
namespace ck_tile {

enum class GroupedConvDirection
{
    FORWARD,
    BACKWARD_DATA,
    BACKWARD_WEIGHT
};

/// @brief The Grouped Conv kernel host arguments.
///
/// @par Overview
///      This structure is passed to Grouped Convolution Kernels when creating kernel
///      arguments object. It contain all necessary information required to
///      build proper kernel argument and launch kernel on GPU.
template <typename InPtr, typename WeiPtr, typename OutPtr, typename CDElementwise>
struct GroupedConvHostArgs : public conv::ConvParam
{
    CK_TILE_HOST GroupedConvHostArgs() = delete;
    CK_TILE_HOST GroupedConvHostArgs(ConvParam conv_param,
                                     InPtr in_ptr_,
                                     WeiPtr wei_ptr_,
                                     const std::vector<const void*> ds_ptr_,
                                     OutPtr out_ptr_,
                                     index_t k_batch_,
                                     CDElementwise elfunc_ = CDElementwise{})
        : conv::ConvParam(conv_param),
          in_ptr(in_ptr_),
          wei_ptr(wei_ptr_),
          ds_ptr(ds_ptr_),
          out_ptr(out_ptr_),
          k_batch(k_batch_),
          elfunc(elfunc_)
    {
    }

    InPtr in_ptr;
    WeiPtr wei_ptr;
    const std::vector<const void*> ds_ptr;
    OutPtr out_ptr;
    index_t k_batch;
    const CDElementwise elfunc;
};

using PassThrough = ck_tile::element_wise::PassThrough;

template <typename CDElementwise = PassThrough>
using GroupedConvFwdHostArgs = GroupedConvHostArgs<const void*, const void*, void*, CDElementwise>;
using GroupedConvBwdWeightHostArgs =
    GroupedConvHostArgs<const void*, void*, const void*, PassThrough>;
using GroupedConvBwdDataHostArgs =
    GroupedConvHostArgs<void*, const void*, const void*, PassThrough>;

template <index_t NDimSpatial_,
          ConvolutionSpecialization ConvSpecialization_,
          typename InLayout_,
          typename WeiLayout_,
          typename DsLayout_,
          typename OutLayout_,
          index_t VectorSizeA_          = 1,
          index_t VectorSizeB_          = 1,
          index_t VectorSizeC_          = 1,
          index_t NumGroupsToMerge_     = 1,
          bool EnableSplitImage_        = false,
          bool ExplicitGemm_            = false,
          typename DepthwiseTraitsType_ = void>
struct GroupedConvTraits
{
    private:
    static constexpr auto generate_implicit_gemm_layout()
    {
        return generate_tuple([](auto) { return ck_tile::tensor_layout::gemm::RowMajor{}; },
                              number<DsLayout_::size()>{});
    }

    public:
    // Fixed values for Implicit GEMM
    struct FixedGemmParams
    {
        static constexpr ck_tile::index_t TilePartitionerGroupNum = 8;
        static constexpr ck_tile::index_t TilePartitionerM01      = 4;
        static constexpr bool kPadM                               = true;
        static constexpr bool kPadN                               = true;
        static constexpr bool kPadK                               = true;
        static constexpr bool TransposeC                          = false;
        static constexpr bool FixedVectorSize                     = true;
        static constexpr bool UseStructuredSparsity               = false;
        static constexpr bool Persistent                          = false;
        using ELayout = ck_tile::tensor_layout::gemm::RowMajor;
    };
    // Compile time parameters
    static constexpr index_t NumGroupsToMerge = NumGroupsToMerge_;
    static constexpr bool EnableSplitImage    = EnableSplitImage_;
    static constexpr bool ExplicitGemm        = ExplicitGemm_;
    static constexpr bool IsDepthwise         = !std::is_void_v<DepthwiseTraitsType_>;
    using DepthwiseTraits                     = DepthwiseTraitsType_;
    static constexpr index_t NDimSpatial      = NDimSpatial_;
    static constexpr ConvolutionSpecialization ConvSpecialization = ConvSpecialization_;
    using InLayout                                                = InLayout_;
    using WeiLayout                                               = WeiLayout_;
    using DsLayout                                                = DsLayout_;
    using OutLayout                                               = OutLayout_;

    // Forward Gemm Layouts
    using AsLayoutFwd = std::conditional_t<NumGroupsToMerge == 1,
                                           ck_tile::tensor_layout::gemm::RowMajor,
                                           ck_tile::tensor_layout::gemm::ColumnMajor>;
    using BsLayoutFwd = ck_tile::tensor_layout::gemm::ColumnMajor;
    using CLayoutFwd  = ck_tile::tensor_layout::gemm::RowMajor;
    // Backward Data Gemm Layouts
    using AsLayoutBwdData = ck_tile::tensor_layout::gemm::RowMajor;
    using BsLayoutBwdData = ck_tile::tensor_layout::gemm::RowMajor;
    using CLayoutBwdData  = ck_tile::tensor_layout::gemm::RowMajor;
    // Backward Weight Gemm Layouts
    using AsLayoutBwdWeight = ck_tile::tensor_layout::gemm::ColumnMajor;
    using BsLayoutBwdWeight = ck_tile::tensor_layout::gemm::RowMajor;
    using CLayoutBwdWeight  = ck_tile::tensor_layout::gemm::RowMajor;

    template <GroupedConvDirection Direction>
    struct GemmLayouts
    {
        static_assert(false, "Unsupported direction.");
    };

    template <>
    struct GemmLayouts<GroupedConvDirection::FORWARD>
    {
        using AsLayout = AsLayoutFwd;
        using BsLayout = BsLayoutFwd;
        using CLayout  = CLayoutFwd;
    };

    template <>
    struct GemmLayouts<GroupedConvDirection::BACKWARD_DATA>
    {
        using AsLayout = AsLayoutBwdData;
        using BsLayout = BsLayoutBwdData;
        using CLayout  = CLayoutBwdData;
    };

    template <>
    struct GemmLayouts<GroupedConvDirection::BACKWARD_WEIGHT>
    {
        using AsLayout = AsLayoutBwdWeight;
        using BsLayout = BsLayoutBwdWeight;
        using CLayout  = CLayoutBwdWeight;
    };

    template <ck_tile::index_t NumWaveGroups = 1>
    using GroupedConvImplicitGemmTraitsFwd =
        TileGemmTraits<true, true, true, AsLayoutFwd, BsLayoutFwd, CLayoutFwd, NumWaveGroups>;
    template <ck_tile::index_t NumWaveGroups = 1>
    using GroupedConvImplicitGemmTraitsBwdData = TileGemmTraits<true,
                                                                true,
                                                                true,
                                                                AsLayoutBwdData,
                                                                BsLayoutBwdData,
                                                                CLayoutBwdData,
                                                                NumWaveGroups>;
    template <ck_tile::index_t NumWaveGroups = 1>
    using GroupedConvImplicitGemmTraitsBwdWeight  = TileGemmTraits<true,
                                                                   true,
                                                                   true,
                                                                   AsLayoutBwdWeight,
                                                                   BsLayoutBwdWeight,
                                                                   CLayoutBwdWeight,
                                                                   NumWaveGroups>;
    static constexpr ck_tile::index_t VectorSizeA = VectorSizeA_;
    static constexpr ck_tile::index_t VectorSizeB = VectorSizeB_;
    static constexpr ck_tile::index_t VectorSizeC = VectorSizeC_;
    static constexpr ck_tile::index_t NumDTensor  = DsLayout::size();
    using ImplicitGemmDsLayout                    = decltype(generate_implicit_gemm_layout());
};

template <typename InDataType_,
          typename WeiDataType_,
          typename AccDataType_,
          typename OutDataType_,
          index_t BlockSize_,
          index_t TileH_,
          index_t TileW_,
          index_t FilterH_,
          index_t FilterW_,
          index_t StrideH_,
          index_t StrideW_,
          index_t DilationH_,
          index_t DilationW_,
          index_t PadH_,
          index_t PadW_,
          index_t NBatch_,
          index_t SubTileH_,
          index_t SubTileW_,
          index_t InVectorSize_,
          index_t OutVectorSize_>
struct DepthwiseConvFwdTraits
{
    using InDataType  = InDataType_;
    using WeiDataType = WeiDataType_;
    using AccDataType = AccDataType_;
    using OutDataType = OutDataType_;

    static constexpr index_t NDimSpatial = 2;

    static constexpr index_t BlockSize = BlockSize_;
    static constexpr index_t WaveSize  = BlockSize;

    static constexpr index_t TileOutH = TileH_;
    static constexpr index_t TileOutW = TileW_;
    static constexpr index_t TileInH  = TileOutH * StrideH_;
    static constexpr index_t TileInW  = TileOutW * StrideW_;

    static constexpr index_t FilterH = FilterH_;
    static constexpr index_t FilterW = FilterW_;

    static constexpr index_t StrideH   = StrideH_;
    static constexpr index_t StrideW   = StrideW_;
    static constexpr index_t DilationH = DilationH_;
    static constexpr index_t DilationW = DilationW_;
    static constexpr index_t PadH      = PadH_;
    static constexpr index_t PadW      = PadW_;

    static constexpr index_t LdsTileH = TileInH + 2 * PadH;
    static constexpr index_t LdsTileW = TileInW + 2 * PadW;

    static constexpr index_t NBatch = NBatch_;

    static constexpr index_t SubTileH = SubTileH_;
    static constexpr index_t SubTileW = SubTileW_;

    static constexpr index_t InVectorSize  = InVectorSize_;
    static constexpr index_t OutVectorSize = OutVectorSize_;
    // Hardcoded to 2: enables v_dot2 (fp16x2) on FP16 and even/odd weight packing for
    // 2-column-per-step processing in RunConvolution when StrideW=1
    static constexpr index_t WeiVectorSize = 2;

    static constexpr index_t HRepeats      = integer_divide_ceil(TileOutH, SubTileH);
    static constexpr index_t WRepeats      = integer_divide_ceil(TileOutW, SubTileW);
    static constexpr index_t TotalSubTiles = HRepeats * WRepeats;
    static constexpr index_t TilePerWave   = WaveSize / TotalSubTiles;
    static constexpr index_t ThreadPerTile = WaveSize / TilePerWave;

    // LdsStride must satisfy: LdsStride - LdsTileW >= PadW (padding vector overflow guard)
    static constexpr index_t LdsStrideBase = integer_least_multiple(LdsTileW, InVectorSize);
    static constexpr index_t LdsStrideMin  = LdsTileW + PadW;
    static constexpr index_t LdsStride     = (LdsStrideBase >= LdsStrideMin)
                                                 ? LdsStrideBase
                                                 : integer_least_multiple(LdsStrideMin, InVectorSize);

    static constexpr index_t LdsTileSize  = LdsTileH * LdsStride;
    static constexpr index_t LdsInputSize = LdsTileSize * TilePerWave * sizeof(InDataType);
    static constexpr index_t LdsSize      = LdsInputSize;

    using InVector  = ext_vector_t<InDataType, InVectorSize>;
    using OutVector = ext_vector_t<OutDataType, OutVectorSize>;
    using WeiVector = ext_vector_t<WeiDataType, WeiVectorSize>;

    // Capped at 4 for LDS access: 4 * sizeof(fp32) = 16 bytes = ds_read_b128 max width.
    // Conservative for FP16 (could be 8), but keeps the code uniform across data types.
    static constexpr index_t InVectorSizeInternal  = (InVectorSize < 4) ? InVectorSize : 4;
    static constexpr index_t OutVectorSizeInternal = (OutVectorSize < 4) ? OutVectorSize : 4;

    using InVectorInternal  = ext_vector_t<InDataType, InVectorSizeInternal>;
    using OutVectorInternal = ext_vector_t<OutDataType, OutVectorSizeInternal>;
    using AccVectorInternal = ext_vector_t<AccDataType, OutVectorSizeInternal>;

    static_assert(std::is_same_v<InDataType, fp16_t> || std::is_same_v<InDataType, bf16_t> ||
                      std::is_same_v<InDataType, float>,
                  "Only fp16, bf16 and float are supported currently");
    static_assert(BlockSize == 64 || BlockSize == 128 || BlockSize == 256,
                  "BlockSize must be 64, 128, or 256");
    static_assert(TotalSubTiles <= WaveSize, "TotalSubTiles must not exceed WaveSize");
    static_assert(DilationH == 1 && DilationW == 1, "Only dilation=1 is supported currently");
    static_assert(FilterH == FilterW, "Only square filters are supported currently");
    static_assert(FilterH % 2 == 1, "Only odd filter sizes are supported (3, 5, 7, 9)");
    static_assert((InVectorSize & (InVectorSize - 1)) == 0 &&
                      (OutVectorSize & (OutVectorSize - 1)) == 0,
                  "InVectorSize and OutVectorSize must be powers of 2");
    static_assert(SubTileH <= TileOutH && SubTileW <= TileOutW,
                  "SubTile dimensions must not exceed Tile output dimensions");
};

/// @brief Helper struct for split-image piece information
///
/// @par Overview
///      Stores metadata for a single spatial piece in split-image convolution.
///      Used to track block ranges and spatial coordinates for each piece.
struct SplitImagePieceInfo
{
    ck_tile::index_t block_start, block_end;    ///< GPU block range for this piece
    ck_tile::index_t d_start, h_start, w_start; ///< Spatial start coordinates (output space)
    ck_tile::index_t d_size, h_size, w_size;    ///< Spatial dimensions of this piece
};

/// @brief Calculate piece information for split-image convolution
///
/// @par Overview
///      Computes spatial coordinates, dimensions, and GPU block range for a single
///      piece in split-image convolution. Handles edge pieces that may have different
///      sizes due to non-uniform division.
///
/// @tparam TilePartitioner Type providing MPerBlock and NPerBlock constants
///
/// @param piece_idx Index of the piece to calculate (0-based)
/// @param num_d_pieces Number of pieces in D dimension
/// @param num_h_pieces Number of pieces in H dimension
/// @param num_w_pieces Number of pieces in W dimension
/// @param base_piece_d Base size of each D piece (may differ for last piece)
/// @param base_piece_h Base size of each H piece (may differ for last piece)
/// @param base_piece_w Base size of each W piece (may differ for last piece)
/// @param total_d Total D dimension size (output space)
/// @param total_h Total H dimension size (output space)
/// @param total_w Total W dimension size (output space)
/// @param N Batch size
/// @param K Output channels
/// @param total_blocks Accumulated block count from previous pieces
///
/// @return SplitImagePieceInfo containing all metadata for this piece
template <typename TilePartitioner>
CK_TILE_HOST SplitImagePieceInfo calculate_spatial_piece(ck_tile::index_t piece_idx,
                                                         ck_tile::index_t num_d_pieces,
                                                         ck_tile::index_t num_h_pieces,
                                                         ck_tile::index_t num_w_pieces,
                                                         ck_tile::index_t base_piece_d,
                                                         ck_tile::index_t base_piece_h,
                                                         ck_tile::index_t base_piece_w,
                                                         ck_tile::index_t total_d,
                                                         ck_tile::index_t total_h,
                                                         ck_tile::index_t total_w,
                                                         ck_tile::index_t N,
                                                         ck_tile::index_t K,
                                                         ck_tile::index_t total_blocks)
{
    // Unflatten piece index into 3D coordinates (W-major, then H, then D)
    const ck_tile::index_t w_idx = piece_idx % num_w_pieces;
    const ck_tile::index_t h_idx = (piece_idx / num_w_pieces) % num_h_pieces;
    const ck_tile::index_t d_idx = piece_idx / (num_w_pieces * num_h_pieces);

    // Calculate spatial start positions
    const ck_tile::index_t w_start = w_idx * base_piece_w;
    const ck_tile::index_t h_start = h_idx * base_piece_h;
    const ck_tile::index_t d_start = d_idx * base_piece_d;

    // Calculate piece sizes (last piece may be larger to cover remainder)
    const ck_tile::index_t w_size =
        (w_idx == num_w_pieces - 1) ? (total_w - w_start) : base_piece_w;
    const ck_tile::index_t h_size =
        (h_idx == num_h_pieces - 1) ? (total_h - h_start) : base_piece_h;
    const ck_tile::index_t d_size =
        (d_idx == num_d_pieces - 1) ? (total_d - d_start) : base_piece_d;

    // Calculate GEMM dimensions for this piece
    const ck_tile::index_t piece_gemm_m = N * d_size * h_size * w_size;
    const ck_tile::index_t piece_gemm_n = K;

    // Calculate GPU grid size for this piece
    const ck_tile::index_t piece_grid =
        ((piece_gemm_m + TilePartitioner::MPerBlock - 1) / TilePartitioner::MPerBlock) *
        ((piece_gemm_n + TilePartitioner::NPerBlock - 1) / TilePartitioner::NPerBlock);

    return {
        total_blocks, total_blocks + piece_grid, d_start, h_start, w_start, d_size, h_size, w_size};
}

} // namespace ck_tile

#if __clang_major__ >= 23
#pragma clang diagnostic pop
#endif
