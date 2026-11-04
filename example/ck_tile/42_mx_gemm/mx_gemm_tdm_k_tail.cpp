// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file
 * @brief MX GEMM on the gfx1250 TDM V2 pipeline with a K that is not a multiple of the K tile or
 *        of the 32-element scale block.
 * @details C = (A * scale_a) x (B * scale_b) with row-major A (M x K), column-major B (K x N),
 *          fp16 C, and one E8M0 scale per 32 consecutive elements of K. The block tile is
 *          256 x 128 x 384 on 2 x 2 waves with 16x16x128 warp tiles. The default K = 900 leaves
 *          a last K tile of 132 elements and a last scale block of 4 elements, and the default
 *          M = N = 1000 leaves partial tiles in M and N.
 *
 *          MxGemmKernel accepts such K only with fp8 or fp4 on each side, 32-element scales, a
 *          row-major A and a column-major B (K contiguous in both) and a K tile that is a
 *          multiple of 128 (four 32-element scales) and of the warp tile K; otherwise
 *          IsSupportedArgument requires K to be a multiple of the K tile and of
 *          ScaleBlockSize * 4. The 384-deep K tile holds three scale packs, so it shows the reads
 *          of scales of later rows described below; it is not a tuned configuration, and the
 *          printed time does not represent the pipeline.
 *
 *          The host code shows the caller's side of the scale layout documented on
 *          ck_tile::IsMxTdmShapeSupported (ck_tile/ops/gemm/kernel/mx_gemm_kernel.hpp): each
 *          scale tensor has one row per row of A (column of B), rounded up to the warp tile, and
 *          ceil(K / 32) scale columns rounded up to a whole number of 4-scale int32 packs. Every
 *          scale must be finite, padding included (written here as raw 0x00): on a K tail the
 *          kernel multiplies the zero A and B elements past K by the padding columns of the last
 *          scale pack and, because a 384-deep K tile holds three scale packs, by scales of later
 *          rows, so one NaN scale can turn other rows of C into NaN. The tensors are pre-shuffled
 *          with preShuffleScaleBuffer_gfx1250, and IsSupportedArgument decides whether the kernel
 *          accepts M, N and K (for fp4xfp4, K must be even: two elements per byte).
 */

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/mx_gemm_kernel.hpp"

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

struct MxGemmTdmKTailConfig
{
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 384;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 128;

    static constexpr ck_tile::index_t ScaleBlockSize = 32;
    // Scales per int32 scale pack (KXdlPack in IsMxTdmShapeSupported): the scale columns are
    // padded to a whole number of packs.
    static constexpr ck_tile::index_t ScalesPerPack = 4;
};

template <typename ADataType,
          typename BDataType,
          typename AScaleDataType,
          typename BScaleDataType,
          typename AccDataType,
          typename CDataType>
struct MxGemmTdmKTailKernel
{
    using Config = MxGemmTdmKTailConfig;

    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<Config::M_Tile, Config::N_Tile, Config::K_Tile>,
        ck_tile::sequence<Config::M_Warp, Config::N_Warp, Config::K_Warp>,
        ck_tile::sequence<Config::M_Warp_Tile, Config::N_Warp_Tile, Config::K_Warp_Tile>>;

    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape, 8, 4>;

    // The TDM clips its loads at M, N and K, so the tile views need no padding.
    static constexpr bool kPadM            = false;
    static constexpr bool kPadN            = false;
    static constexpr bool kPadK            = false;
    static constexpr bool DoubleSmemBuffer = true;
    static constexpr bool TransposeC       = true;

    using Traits = ck_tile::TileGemmUniversalTraits<kPadM,
                                                    kPadN,
                                                    kPadK,
                                                    DoubleSmemBuffer,
                                                    Row,
                                                    Col,
                                                    Row,
                                                    TransposeC,
                                                    false,  // UseStructuredSparsity
                                                    false,  // UsePersistentKernel
                                                    1,      // NumWaveGroups
                                                    false>; // Preshuffle

    using Problem = ck_tile::MxGemmPipelineProblem<ADataType,
                                                   BDataType,
                                                   AccDataType,
                                                   GemmShape,
                                                   Traits,
                                                   ck_tile::GemmPipelineScheduler::Intrawave,
                                                   ck_tile::element_wise::PassThrough,
                                                   ck_tile::element_wise::PassThrough,
                                                   ADataType,
                                                   BDataType,
                                                   AScaleDataType,
                                                   BScaleDataType,
                                                   Config::ScaleBlockSize>;

    using Pipeline = ck_tile::GemmPipelineAgBgCrCompTDMV2<Problem>;

    using Epilogue =
        ck_tile::TdmEpilogue<ck_tile::CShuffleEpilogueProblem<ADataType,
                                                              BDataType,
                                                              ck_tile::tuple<>, // DsDataType
                                                              AccDataType,
                                                              CDataType,
                                                              ck_tile::tuple<>, // DsLayout
                                                              Row,
                                                              ck_tile::element_wise::PassThrough,
                                                              TilePartitioner::MPerBlock,
                                                              TilePartitioner::NPerBlock,
                                                              Config::M_Warp,
                                                              Config::N_Warp,
                                                              Config::M_Warp_Tile,
                                                              Config::N_Warp_Tile,
                                                              Config::K_Warp_Tile,
                                                              Problem::TransposeC,
                                                              1,     // NumWaveGroups
                                                              false, // FixedVectorSize
                                                              1,     // VectorSizeC
                                                              1,     // BlockedXDLN_PerWarp
                                                              DoubleSmemBuffer,
                                                              ADataType,
                                                              BDataType,
                                                              true>>; // TilesPacked

    using type = ck_tile::MxGemmKernel<TilePartitioner, Pipeline, Epilogue>;
};

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "1000", "m dimension")
        .insert("n", "1000", "n dimension")
        .insert("k",
                "900",
                "k dimension: any K > 0 for fp8xfp8, an even K for fp4xfp4; need not be a "
                "multiple of the K tile (384) or of the scale block (32)")
        .insert("mx_prec", "fp8xfp8", "A x B precision: fp8xfp8 (fp8) or fp4xfp4 (fp4)")
        .insert("v", "1", "0: no check, 1: check against the CPU reference")
        .insert("warmup", "50", "number of warmup iterations")
        .insert("repeat", "100", "number of timed iterations")
        .insert("init",
                "0",
                "0: random integers in [-5, 5] and random scales in [0.125, 2]; 1: all ones, "
                "scales 1.0");
    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

// Fills A or B with integers in [-5, 5] (init 0) or with ones (init 1). FP4 stores each integer
// as the nearest FP4 value (E2M1 has no 5), and a packed FP4 byte gets two independent values.
template <typename T>
void fill_operand(ck_tile::HostTensor<T>& t, int init, std::uint32_t seed)
{
    if constexpr(std::is_same_v<T, ck_tile::pk_fp4_t>)
    {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dis(-5.f, 5.f);
        const auto value = [&]() { return init == 1 ? 1.f : std::round(dis(gen)); };
        for(auto& elem : t.mData)
        {
            const auto lo = ck_tile::float_to_mxfp4(value(), 1.f);
            const auto hi = ck_tile::float_to_mxfp4(value(), 1.f);
            elem          = ck_tile::pk_fp4_t::_pack(lo, hi);
        }
    }
    else if(init == 1)
    {
        ck_tile::FillConstant<T>{ck_tile::type_convert<T>(1.f)}(t);
    }
    else
    {
        ck_tile::FillUniformDistributionIntegerValue<T>{-5, 5, seed}(t);
    }
}

// The scale tensor for the kernel: `rows` rows (padded to the warp tile) of `cols` scales
// (ceil(K / 32) padded to whole packs), row-major with K the fast dimension. The real scales
// fill the top-left corner (the lengths of `real`); every other scale is padding.
template <typename ScaleType>
ck_tile::HostTensor<ScaleType> make_padded_scales(const ck_tile::HostTensor<ScaleType>& real,
                                                  ck_tile::index_t rows,
                                                  ck_tile::index_t cols)
{
    const auto real_rows = static_cast<ck_tile::index_t>(real.get_lengths()[0]);
    const auto real_cols = static_cast<ck_tile::index_t>(real.get_lengths()[1]);
    if(rows < real_rows || cols < real_cols)
    {
        throw std::invalid_argument("make_padded_scales: the padded tensor is smaller than the "
                                    "scales");
    }
    ck_tile::HostTensor<ScaleType> padded(
        {static_cast<std::size_t>(rows), static_cast<std::size_t>(cols)},
        {static_cast<std::size_t>(cols), std::size_t{1}});
    // On a K tail the kernel multiplies the zero A and B elements past K by the padding columns of
    // the last scale pack and by scales of later rows, padding rows included
    // (MxGemmKernel::GetScalePacksK). NaN x 0 is NaN, so every scale must be finite. A
    // default-constructed ScaleType is NaN; raw 0x00 is finite for every MX scale type (2^-127 for
    // E8M0).
    std::fill(padded.mData.begin(), padded.mData.end(), ScaleType{typename ScaleType::type{0}});
    for(ck_tile::index_t r = 0; r < real_rows; ++r)
        for(ck_tile::index_t c = 0; c < real_cols; ++c)
            padded(r, c) = real(r, c);
    return padded;
}

template <typename ADataType, typename BDataType>
int run_mx_gemm_tdm_k_tail(ck_tile::ArgParser& arg_parser)
{
    using namespace ck_tile;

    using AScaleDataType = e8m0_t;
    using BScaleDataType = e8m0_t;
    using AccDataType    = float;
    using CDataType      = half_t;
    using Config         = MxGemmTdmKTailConfig;
    using Kernel         = typename MxGemmTdmKTailKernel<ADataType,
                                                         BDataType,
                                                         AScaleDataType,
                                                         BScaleDataType,
                                                         AccDataType,
                                                         CDataType>::type;

    const index_t M     = arg_parser.get_int("m");
    const index_t N     = arg_parser.get_int("n");
    const index_t K     = arg_parser.get_int("k");
    const int init      = arg_parser.get_int("init");
    const int n_warmup  = arg_parser.get_int("warmup");
    const int n_repeat  = arg_parser.get_int("repeat");
    const bool do_check = arg_parser.get_int("v") == 1;
    if(M <= 0 || N <= 0 || K <= 0)
    {
        std::cerr << "m, n and k must be positive." << std::endl;
        return EXIT_FAILURE;
    }
    if(M > numeric<index_t>::max() - Config::M_Tile)
    {
        std::cerr << "m is too large." << std::endl;
        return EXIT_FAILURE;
    }
    if(n_warmup < 0 || n_repeat < 1)
    {
        std::cerr << "warmup must be >= 0 and repeat >= 1." << std::endl;
        return EXIT_FAILURE;
    }

    // A, B and C have no padding between rows: each row of A and each column of B holds K
    // contiguous elements, and each row of C holds N.
    const index_t stride_a = K;
    const index_t stride_b = K;
    const index_t stride_c = N;

    // The kernel arguments for the given buffers.
    const auto make_kargs =
        [&](const void* a, const void* scale_a, const void* b, const void* scale_b, void* c) {
            return Kernel::MakeKernelArgs(MxGemmHostArgs<1, 1, 0>({a},
                                                                  {scale_a},
                                                                  {b},
                                                                  {scale_b},
                                                                  {},
                                                                  c,
                                                                  1, // k_batch
                                                                  M,
                                                                  N,
                                                                  K,
                                                                  {stride_a},
                                                                  {stride_b},
                                                                  {},
                                                                  stride_c));
        };
    // Reports that the kernel rejects the arguments, with the kernel's reason.
    const auto report_unsupported = [&](const auto& kargs) {
        if(!EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
        {
            // Check again with logging on, so that the kernel prints why it rejects the arguments.
            UpdateEnvVar(CK_TILE_ENV(CK_TILE_LOGGING), true);
            static_cast<void>(Kernel::IsSupportedArgument(kargs));
        }
        std::cerr << "The kernel does not support M = " << M << ", N = " << N << ", K = " << K
                  << " (stride A = " << stride_a << ", stride B = " << stride_b << ")."
                  << std::endl;
    };

    // Check the shape before allocating: with null pointers IsSupportedArgument checks the sizes
    // and the strides (a null pointer passes its pointer-alignment check). The final arguments are
    // checked again before the launch.
    if(const auto shape_kargs = make_kargs(nullptr, nullptr, nullptr, nullptr, nullptr);
       !Kernel::IsSupportedArgument(shape_kargs))
    {
        report_unsupported(shape_kargs);
        return EXIT_FAILURE;
    }

    // One scale per ScaleBlockSize elements of K; when K % ScaleBlockSize != 0, the last block
    // covers only the remaining K % ScaleBlockSize elements. The kernel reads whole int32 packs of
    // ScalesPerPack scales and rows grouped by the warp tile, so the scale tensors are padded in
    // both dimensions.
    const index_t num_scale_blocks = integer_divide_ceil(K, Config::ScaleBlockSize);
    const index_t num_scale_cols = integer_least_multiple(num_scale_blocks, Config::ScalesPerPack);
    const index_t scale_rows_a   = integer_least_multiple(M, Config::M_Warp_Tile);
    const index_t scale_rows_b   = integer_least_multiple(N, Config::N_Warp_Tile);

    const index_t num_k_tiles = integer_divide_ceil(K, Config::K_Tile);
    std::cout << "M = " << M << ", N = " << N << ", K = " << K << "\n"
              << "  K tiles of " << Config::K_Tile << ": " << num_k_tiles
              << ", elements in the last: " << K - (num_k_tiles - 1) * Config::K_Tile << "\n"
              << "  scale blocks of " << Config::ScaleBlockSize << ": " << num_scale_blocks
              << ", elements in the last: " << K - (num_scale_blocks - 1) * Config::ScaleBlockSize
              << "\n"
              << "  scale columns per row: " << num_scale_cols
              << " (padding: " << num_scale_cols - num_scale_blocks
              << "), int32 scale packs per row: " << num_scale_cols / Config::ScalesPerPack << "\n"
              << "  scale rows: A " << scale_rows_a << " (padding: " << scale_rows_a - M << "), B "
              << scale_rows_b << " (padding: " << scale_rows_b - N << ")" << std::endl;

    HostTensor<ADataType> a_host(host_tensor_descriptor(M, K, stride_a, bool_constant<true>{}));
    HostTensor<BDataType> b_host(host_tensor_descriptor(K, N, stride_b, bool_constant<false>{}));
    HostTensor<CDataType> c_host(host_tensor_descriptor(M, N, stride_c, bool_constant<true>{}));

    // The real scales: scale_a is M x num_scale_blocks and scale_b is N x num_scale_blocks, both
    // with K the fast dimension.
    HostTensor<AScaleDataType> scale_a(
        {static_cast<std::size_t>(M), static_cast<std::size_t>(num_scale_blocks)},
        {static_cast<std::size_t>(num_scale_blocks), std::size_t{1}});
    HostTensor<BScaleDataType> scale_b(
        {static_cast<std::size_t>(N), static_cast<std::size_t>(num_scale_blocks)},
        {static_cast<std::size_t>(num_scale_blocks), std::size_t{1}});

    // Fixed seeds, so that every run uses the same data.
    constexpr std::uint32_t seed = 11939;
    fill_operand(a_host, init, seed);
    fill_operand(b_host, init, seed + 1);
    if(init == 1)
    {
        FillConstant<AScaleDataType>{AScaleDataType{1.f}}(scale_a);
        FillConstant<BScaleDataType>{BScaleDataType{1.f}}(scale_b);
    }
    else
    {
        FillUniformScaleDistribution<AScaleDataType>{0.125f, 2.f, seed + 2}(scale_a);
        FillUniformScaleDistribution<BScaleDataType>{0.125f, 2.f, seed + 3}(scale_b);
    }

    // The scale tensors the kernel reads: padded, then pre-shuffled into the device layout,
    // passing the padded row count, the padded column count and the warp tile M (N).
    const auto scale_a_padded = make_padded_scales(scale_a, scale_rows_a, num_scale_cols);
    const auto scale_b_padded = make_padded_scales(scale_b, scale_rows_b, num_scale_cols);
    HostTensor<AScaleDataType> scale_a_shuffled(scale_a_padded.get_lengths(),
                                                scale_a_padded.get_strides());
    HostTensor<BScaleDataType> scale_b_shuffled(scale_b_padded.get_lengths(),
                                                scale_b_padded.get_strides());
    preShuffleScaleBuffer_gfx1250<AScaleDataType, Config::ScaleBlockSize, true>(
        scale_a_padded.data(),
        scale_a_shuffled.data(),
        scale_rows_a,
        num_scale_cols,
        Config::M_Warp_Tile);
    preShuffleScaleBuffer_gfx1250<BScaleDataType, Config::ScaleBlockSize, true>(
        scale_b_padded.data(),
        scale_b_shuffled.data(),
        scale_rows_b,
        num_scale_cols,
        Config::N_Warp_Tile);

    DeviceMem a_dev(a_host.get_element_space_size_in_bytes());
    DeviceMem b_dev(b_host.get_element_space_size_in_bytes());
    DeviceMem c_dev(c_host.get_element_space_size_in_bytes());
    DeviceMem scale_a_dev(scale_a_shuffled.get_element_space_size_in_bytes());
    DeviceMem scale_b_dev(scale_b_shuffled.get_element_space_size_in_bytes());
    a_dev.ToDevice(a_host.data());
    b_dev.ToDevice(b_host.data());
    scale_a_dev.ToDevice(scale_a_shuffled.data());
    scale_b_dev.ToDevice(scale_b_shuffled.data());
    c_dev.SetZero();

    const auto kargs = make_kargs(a_dev.GetDeviceBuffer(),
                                  scale_a_dev.GetDeviceBuffer(),
                                  b_dev.GetDeviceBuffer(),
                                  scale_b_dev.GetDeviceBuffer(),
                                  c_dev.GetDeviceBuffer());
    if(!Kernel::IsSupportedArgument(kargs))
    {
        report_unsupported(kargs);
        return EXIT_FAILURE;
    }

    const dim3 grids  = Kernel::GridSize(M, N, 1);
    const dim3 blocks = Kernel::BlockSize();
    std::cout << "Grid {" << grids.x << ", " << grids.y << ", " << grids.z << "}, block {"
              << blocks.x << ", " << blocks.y << ", " << blocks.z << "}" << std::endl;

    const float ave_time = launch_kernel(stream_config{nullptr, true, 1, n_warmup, n_repeat},
                                         make_kernel<1>(Kernel{}, grids, blocks, 0, kargs));

    const std::size_t flop     = std::size_t(2) * M * N * K;
    const std::size_t num_byte = a_host.get_element_space_size_in_bytes() +
                                 b_host.get_element_space_size_in_bytes() +
                                 c_host.get_element_space_size_in_bytes() +
                                 scale_a_shuffled.get_element_space_size_in_bytes() +
                                 scale_b_shuffled.get_element_space_size_in_bytes();
    std::cout << "Time " << ave_time << " ms, " << static_cast<float>(flop) / 1.e9f / ave_time
              << " TFlops, " << static_cast<float>(num_byte) / 1.e6f / ave_time << " GB/s"
              << std::endl;

    if(!do_check)
        return EXIT_SUCCESS;

    c_dev.FromDevice(c_host.data());

    // The reference takes scale_a as M x num_scale_blocks and scale_b as num_scale_blocks x N.
    // scale_b_ref has a column-major descriptor, so its memory layout equals that of scale_b and a
    // plain copy fills it.
    HostTensor<BScaleDataType> scale_b_ref(
        {static_cast<std::size_t>(num_scale_blocks), static_cast<std::size_t>(N)},
        {std::size_t{1}, static_cast<std::size_t>(num_scale_blocks)});
    std::copy(scale_b.mData.begin(), scale_b.mData.end(), scale_b_ref.mData.begin());

    HostTensor<CDataType> c_ref(host_tensor_descriptor(M, N, stride_c, bool_constant<true>{}));
    c_ref.SetZero();
    reference_mx_gemm<ADataType, BDataType, AScaleDataType, BScaleDataType, AccDataType, CDataType>(
        a_host, b_host, c_ref, scale_a, scale_b_ref, Config::ScaleBlockSize);

    float max_abs = 0.f;
    for(const auto& v : c_ref.mData)
        max_abs = std::max(max_abs, std::abs(type_convert<float>(v)));
    if(!std::isfinite(max_abs))
    {
        std::cerr << "The result exceeds the range of the fp16 C." << std::endl;
        return EXIT_FAILURE;
    }
    // The products of A, B and the scales are exact in AccDataType, so the kernel and the reference
    // differ only in the fp32 summation order and the rounding to CDataType.
    const auto rtol = get_relative_threshold<CDataType, CDataType, AccDataType>(K);
    const auto atol =
        get_absolute_threshold<CDataType, CDataType, AccDataType>(std::max(max_abs, 1.f), K);
    const bool pass = check_err(c_host, c_ref, "Error: Incorrect results!", rtol, atol);
    std::cout << "Relative error threshold " << rtol << ", absolute error threshold " << atol
              << ": " << (pass ? "correct" : "incorrect") << std::endl;
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return EXIT_FAILURE;

    const std::string prec = arg_parser.get_str("mx_prec");
    try
    {
        if(prec == "fp8" || prec == "fp8xfp8")
            return run_mx_gemm_tdm_k_tail<ck_tile::fp8_t, ck_tile::fp8_t>(arg_parser);
        if(prec == "fp4" || prec == "fp4xfp4")
            return run_mx_gemm_tdm_k_tail<ck_tile::pk_fp4_t, ck_tile::pk_fp4_t>(arg_parser);
        std::cerr << "Unsupported precision " << prec << "; use fp8xfp8 or fp4xfp4." << std::endl;
        return EXIT_FAILURE;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
