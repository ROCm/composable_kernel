// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// This example runs MX flat GEMM using the TDM v1 pipeline with data cache prefetch.
// Instead of using the WeightPreshufflePipelineAGmemBGmemCRegTDM (preshuffle TDM pipeline),
// it uses the compute TDM v1 pipeline (GemmPipelineAgBgCrCompTDMV1) which supports
// hardware data cache prefetch on gfx1250.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>
#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/mx_gemm_kernel.hpp"

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

// MX GEMM config using TDM v1 pipeline with data cache prefetch
template <typename PrecType,
          ck_tile::DataCachePrefetchKind DataCachePrefetchA_ = ck_tile::DataCachePrefetchKind::L2,
          ck_tile::DataCachePrefetchKind DataCachePrefetchB_ = DataCachePrefetchA_>
struct MXGemmConfigTDMV1Prefetch
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    // gfx1250 TDM v1 MX scale distribution requires 32x32 warp tiles
    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 128;

    static constexpr bool kPadM = true;
    static constexpr bool kPadN = true;
    static constexpr bool kPadK = false;

    static constexpr bool TransposeC       = true;
    static constexpr bool DoubleSmemBuffer = true;

    static constexpr ck_tile::DataCachePrefetchKind DataCachePrefetchA = DataCachePrefetchA_;
    static constexpr ck_tile::DataCachePrefetchKind DataCachePrefetchB = DataCachePrefetchB_;

    static constexpr ck_tile::index_t ScaleBlockSize = 32;
};

template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AScaleDataType,
          typename BScaleDataType,
          bool CompareWithNoPrefetch>
float invoke_mx_gemm_tdm_v1(ck_tile::DeviceMem& a_dev_buf,
                            ck_tile::DeviceMem& b_dev_buf,
                            ck_tile::DeviceMem& c_dev_buf,
                            ck_tile::DeviceMem& scale_a_dev_buf,
                            ck_tile::DeviceMem& scale_b_dev_buf,
                            ck_tile::index_t M,
                            ck_tile::index_t N,
                            ck_tile::index_t K,
                            ck_tile::index_t stride_A,
                            ck_tile::index_t stride_B,
                            ck_tile::index_t stride_C,
                            int n_warmup,
                            int n_repeat)
{
    using namespace ck_tile;

    constexpr index_t M_Tile = GemmConfig::M_Tile;
    constexpr index_t N_Tile = GemmConfig::N_Tile;
    constexpr index_t K_Tile = GemmConfig::K_Tile;

    constexpr index_t M_Warp = GemmConfig::M_Warp;
    constexpr index_t N_Warp = GemmConfig::N_Warp;
    constexpr index_t K_Warp = GemmConfig::K_Warp;

    constexpr index_t M_Warp_Tile = GemmConfig::M_Warp_Tile;
    constexpr index_t N_Warp_Tile = GemmConfig::N_Warp_Tile;
    constexpr index_t K_Warp_Tile = GemmConfig::K_Warp_Tile;

    using GemmShape = TileGemmShape<sequence<M_Tile, N_Tile, K_Tile>,
                                    sequence<M_Warp, N_Warp, K_Warp>,
                                    sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

    using TilePartitioner = GemmSpatiallyLocalTilePartitioner<GemmShape, 8, 4>;

    using GemmUniversalTraits = TileGemmUniversalTraits<GemmConfig::kPadM,
                                                        GemmConfig::kPadN,
                                                        GemmConfig::kPadK,
                                                        GemmConfig::DoubleSmemBuffer,
                                                        ALayout,
                                                        BLayout,
                                                        CLayout,
                                                        GemmConfig::TransposeC,
                                                        false, // UseStructuredSparsity
                                                        false, // UsePersistentKernel
                                                        1,     // NumWaveGroups
                                                        false, // Preshuffle
                                                        16,    // VectorSize
                                                        GemmConfig::DataCachePrefetchA,
                                                        GemmConfig::DataCachePrefetchB>;

    using AComputeDataType = ADataType;
    using BComputeDataType = BDataType;

    using UniversalGemmProblem = MxGemmPipelineProblem<ADataType,
                                                       BDataType,
                                                       AccDataType,
                                                       GemmShape,
                                                       GemmUniversalTraits,
                                                       GemmPipelineScheduler::Intrawave,
                                                       element_wise::PassThrough,
                                                       element_wise::PassThrough,
                                                       AComputeDataType,
                                                       BComputeDataType,
                                                       AScaleDataType,
                                                       BScaleDataType,
                                                       GemmConfig::ScaleBlockSize>;

    using GemmPipeline = GemmPipelineAgBgCrCompTDMV1<
        UniversalGemmProblem,
        GemmPipelineAgBgCrCompTDMDefaultPolicy<false, // WaveSpecialized
                                               GemmConfig::DataCachePrefetchA,
                                               GemmConfig::DataCachePrefetchB>>;

    using GemmEpilogue = TdmEpilogue<CShuffleEpilogueProblem<ADataType,
                                                             BDataType,
                                                             tuple<>, // DsDataType
                                                             AccDataType,
                                                             CDataType,
                                                             tuple<>, // DsLayout
                                                             CLayout,
                                                             element_wise::PassThrough,
                                                             TilePartitioner::MPerBlock,
                                                             TilePartitioner::NPerBlock,
                                                             M_Warp,
                                                             N_Warp,
                                                             M_Warp_Tile,
                                                             N_Warp_Tile,
                                                             K_Warp_Tile,
                                                             UniversalGemmProblem::TransposeC,
                                                             1,     // NumWaveGroups
                                                             false, // FixedVectorSize
                                                             1,     // VectorSizeC
                                                             1,     // BlockedXDLN_PerWarp
                                                             GemmConfig::DoubleSmemBuffer,
                                                             AComputeDataType,
                                                             BComputeDataType>>;

    using Kernel = MxGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;

    constexpr index_t ScaleBlockSize = GemmConfig::ScaleBlockSize;

    MxGemmHostArgs<1, 1, 0> args({a_dev_buf.GetDeviceBuffer()},
                                 {scale_a_dev_buf.GetDeviceBuffer()},
                                 {b_dev_buf.GetDeviceBuffer()},
                                 {scale_b_dev_buf.GetDeviceBuffer()},
                                 {},
                                 c_dev_buf.GetDeviceBuffer(),
                                 1, // k_batch
                                 M,
                                 N,
                                 K,
                                 {stride_A},
                                 {stride_B},
                                 {},
                                 stride_C);

    auto kargs = Kernel::MakeKernelArgs(args);

    const dim3 grids  = Kernel::GridSize(M, N, 1);
    const dim3 blocks = Kernel::BlockSize();

    if(!Kernel::IsSupportedArgument(kargs))
    {
        std::cerr << "Wrong! Arguments not supported! Skipping kernel!\n";
        return -1.f;
    }

    auto kind_str = [](ck_tile::DataCachePrefetchKind k) {
        return k == ck_tile::DataCachePrefetchKind::L1   ? "L1"
               : k == ck_tile::DataCachePrefetchKind::L2 ? "L2"
                                                         : "None";
    };
    std::cout << "Launching MX GEMM TDM V1 kernel with data cache prefetch" << " (A "
              << kind_str(GemmConfig::DataCachePrefetchA) << " / B "
              << kind_str(GemmConfig::DataCachePrefetchB) << ")\n"
              << "  Grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
              << ", Blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
              << std::endl;

    float ave_time =
        launch_kernel(stream_config{nullptr, true, 1, n_warmup, n_repeat, true, true, 50},
                      make_kernel<1>(Kernel{}, grids, blocks, 0, kargs));

    constexpr int APackedSize = numeric_traits<ADataType>::PackedSize;
    constexpr int BPackedSize = numeric_traits<BDataType>::PackedSize;

    std::size_t flop     = std::size_t(2) * M * N * K;
    std::size_t num_byte = sizeof(ADataType) * M * K / APackedSize +
                           sizeof(BDataType) * N * K / BPackedSize + sizeof(CDataType) * M * N +
                           sizeof(AScaleDataType) * M * (K / ScaleBlockSize) +
                           sizeof(BScaleDataType) * N * (K / ScaleBlockSize);

    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "  M=" << M << " N=" << N << " K=" << K << " : " << ave_time << " ms, " << tflops
              << " TFlops, " << gb_per_sec << " GB/s" << std::endl;

    return ave_time;
}

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "128", "m dimension")
        .insert("n", "128", "n dimension")
        .insert("k", "256", "k dimension")
        .insert("stride_a", "0", "Tensor A stride")
        .insert("stride_b", "0", "Tensor B stride")
        .insert("stride_c", "0", "Tensor C stride")
        .insert("v", "1", "0. No validation, 1. Validation on CPU")
        .insert("mx_prec", "fp4xfp4", "support: fp4xfp4, fp8xfp8")
        .insert("warmup", "50", "number of warmup iterations")
        .insert("repeat", "100", "number of benchmark iterations")
        .insert("compare", "0", "0: prefetch only, 1: compare with/without prefetch")
        .insert("prefetch_a_l1", "0", "0: prefetch A to L2, 1: prefetch A to L1")
        .insert("prefetch_b_l1", "0", "0: prefetch B to L2, 1: prefetch B to L1")
        .insert("init", "0", "0: random, 1: constant(1)");
    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

/// @brief Pre-shuffle scale buffer for gfx1250 wmma mx scale instruction.
template <typename ScaleType, ck_tile::index_t ScaleBlockSize, bool KStride>
void preShuffleScaleBuffer(const ScaleType* src,
                           ScaleType* dst,
                           ck_tile::index_t MN,
                           ck_tile::index_t K)
{
    static_assert(ScaleBlockSize == 32 && sizeof(ScaleType) == 1,
                  "Only 8-bit scale with ScaleBlockSize=32 supported");

    constexpr ck_tile::index_t MPerXdlops = 16;
    constexpr ck_tile::index_t KPerXdlops = 128;

    int MNPack = 2;
    int KPack  = 1;

    int MNStep = MPerXdlops;
    int KStep  = KPerXdlops / ScaleBlockSize;

    int K0 = K / KPack / KStep;

    for(int mn = 0; mn < MN; ++mn)
    {
        int iMNRepeat = mn / (MNStep * MNPack);
        int tempmn    = mn % (MNStep * MNPack);

        for(int k = 0; k < K; ++k)
        {
            int iKRepeat = k / (KStep * KPack);
            int tempk    = k % (KStep * KPack);

            int outputIndex = (iMNRepeat * MNPack * MNStep) * (KStep * KPack * K0) +
                              (iKRepeat * KStep * KPack) * (MNStep * MNPack) +
                              tempmn * (KStep * KPack) + tempk;

            if constexpr(KStride)
                dst[outputIndex] = src[mn * K + k];
            else
                dst[outputIndex] = src[k * MN + mn];
        }
    }
}

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename AScaleDataType,
          typename BScaleDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
int run_mx_gemm_tdm_v1_prefetch(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    using namespace ck_tile;

    constexpr index_t ScaleBlockSize = 32;

    index_t M = arg_parser.get_int("m");
    index_t N = arg_parser.get_int("n");
    index_t K = arg_parser.get_int("k");

    index_t stride_A = arg_parser.get_int("stride_a");
    index_t stride_B = arg_parser.get_int("stride_b");
    index_t stride_C = arg_parser.get_int("stride_c");

    index_t init_method  = arg_parser.get_int("init");
    index_t n_warmup     = arg_parser.get_int("warmup");
    index_t n_repeat     = arg_parser.get_int("repeat");
    bool compare         = arg_parser.get_int("compare") == 1;
    auto prefetch_kind_a = arg_parser.get_int("prefetch_a_l1") == 1 ? DataCachePrefetchKind::L1
                                                                    : DataCachePrefetchKind::L2;
    auto prefetch_kind_b = arg_parser.get_int("prefetch_b_l1") == 1 ? DataCachePrefetchKind::L1
                                                                    : DataCachePrefetchKind::L2;

    stride_A = get_default_stride(M, K, stride_A, is_row_major(ALayout{}));
    stride_B = get_default_stride(K, N, stride_B, is_row_major(BLayout{}));
    stride_C = get_default_stride(M, N, stride_C, is_row_major(CLayout{}));

    if(K % ScaleBlockSize != 0)
        throw std::runtime_error("K must be multiple of ScaleBlockSize");

    HostTensor<ADataType> a_host(host_tensor_descriptor(M, K, stride_A, is_row_major(ALayout{})));
    HostTensor<BDataType> b_host(host_tensor_descriptor(K, N, stride_B, is_row_major(BLayout{})));
    HostTensor<CDataType> c_rslt_host(
        host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

    index_t scale_K_dim = K / ScaleBlockSize;
    // Pad M to M_Warp_Tile boundary for scale_a (required by hardware layout)
    constexpr index_t M_Warp_Tile = MXGemmConfigTDMV1Prefetch<ADataType>::M_Warp_Tile;
    index_t scale_padded_M        = integer_least_multiple(M, M_Warp_Tile);

    // scale_a: (padded_M, K/ScaleBlockSize) row-major
    HostTensor<AScaleDataType> scale_a(
        {static_cast<std::size_t>(scale_padded_M), static_cast<std::size_t>(scale_K_dim)},
        {static_cast<std::size_t>(scale_K_dim), std::size_t{1}});
    // scale_b: (N, K/ScaleBlockSize) row-major -> K is the fast-changing dimension
    HostTensor<BScaleDataType> scale_b(
        {static_cast<std::size_t>(N), static_cast<std::size_t>(scale_K_dim)},
        {static_cast<std::size_t>(scale_K_dim), std::size_t{1}});

    if(init_method == 0)
    {
        FillUniformDistribution<>{0.0f, 1.0f}(a_host);
        FillUniformDistribution<>{-0.5f, 0.5f}(b_host);
        FillUniformDistribution<>{-2.f, 2.f}(scale_a);
        FillUniformDistribution<>{-2.f, 2.f}(scale_b);
    }
    else
    {
        FillUniformDistribution<>{1.f, 1.f}(a_host);
        FillUniformDistribution<>{1.f, 1.f}(b_host);
        FillUniformDistribution<>{1.f, 1.f}(scale_a);
        FillUniformDistribution<>{1.f, 1.f}(scale_b);
    }

    // Pre-shuffle scales for hardware (gfx1250 wmma layout)
    HostTensor<AScaleDataType> scale_a_shuffled(
        {static_cast<std::size_t>(scale_padded_M), static_cast<std::size_t>(scale_K_dim)},
        {static_cast<std::size_t>(scale_K_dim), std::size_t{1}});
    HostTensor<BScaleDataType> scale_b_shuffled(
        {static_cast<std::size_t>(N), static_cast<std::size_t>(scale_K_dim)},
        {static_cast<std::size_t>(scale_K_dim), std::size_t{1}});

    // Both scale_a and scale_b are row-major (N/M, K/ScaleBlockSize) with K as fast dim
    preShuffleScaleBuffer<AScaleDataType, ScaleBlockSize, true>(
        scale_a.data(), scale_a_shuffled.data(), scale_padded_M, scale_K_dim);
    preShuffleScaleBuffer<BScaleDataType, ScaleBlockSize, true>(
        scale_b.data(), scale_b_shuffled.data(), N, scale_K_dim);

    DeviceMem a_dev_buf(a_host.get_element_space_size_in_bytes());
    DeviceMem b_dev_buf(b_host.get_element_space_size_in_bytes());
    DeviceMem c_dev_buf(c_rslt_host.get_element_space_size_in_bytes());
    DeviceMem scale_a_dev_buf(scale_a_shuffled.get_element_space_size_in_bytes());
    DeviceMem scale_b_dev_buf(scale_b_shuffled.get_element_space_size_in_bytes());

    a_dev_buf.ToDevice(a_host.data());
    b_dev_buf.ToDevice(b_host.data());
    scale_a_dev_buf.ToDevice(scale_a_shuffled.data());
    scale_b_dev_buf.ToDevice(scale_b_shuffled.data());
    c_rslt_host.SetZero();

    // Run with data cache prefetch enabled
    using Kind    = DataCachePrefetchKind;
    auto kind_str = [](Kind k) { return k == Kind::L1 ? "L1" : "L2"; };

    std::cout << "\n=== Running MX GEMM with TDM V1 Pipeline - DataCache Prefetch ENABLED (A "
              << kind_str(prefetch_kind_a) << " / B " << kind_str(prefetch_kind_b) << ") ===\n"
              << std::endl;

    auto run_prefetch = [&](auto prefetch_a_tag, auto prefetch_b_tag) {
        using Config = MXGemmConfigTDMV1Prefetch<ADataType,
                                                 decltype(prefetch_a_tag)::value,
                                                 decltype(prefetch_b_tag)::value>;
        return invoke_mx_gemm_tdm_v1<Config,
                                     ADataType,
                                     BDataType,
                                     AccDataType,
                                     CDataType,
                                     ALayout,
                                     BLayout,
                                     CLayout,
                                     AScaleDataType,
                                     BScaleDataType,
                                     false>(a_dev_buf,
                                            b_dev_buf,
                                            c_dev_buf,
                                            scale_a_dev_buf,
                                            scale_b_dev_buf,
                                            M,
                                            N,
                                            K,
                                            stride_A,
                                            stride_B,
                                            stride_C,
                                            n_warmup,
                                            n_repeat);
    };

    float ave_time_prefetch = 0.f;
    ignore                  = ave_time_prefetch;
    if(prefetch_kind_a == Kind::L1 && prefetch_kind_b == Kind::L1)
    {
        ave_time_prefetch = run_prefetch(std::integral_constant<Kind, Kind::L1>{},
                                         std::integral_constant<Kind, Kind::L1>{});
    }
    else if(prefetch_kind_a == Kind::L1 && prefetch_kind_b == Kind::L2)
    {
        ave_time_prefetch = run_prefetch(std::integral_constant<Kind, Kind::L1>{},
                                         std::integral_constant<Kind, Kind::L2>{});
    }
    else if(prefetch_kind_a == Kind::L2 && prefetch_kind_b == Kind::L1)
    {
        ave_time_prefetch = run_prefetch(std::integral_constant<Kind, Kind::L2>{},
                                         std::integral_constant<Kind, Kind::L1>{});
    }
    else
    {
        ave_time_prefetch = run_prefetch(std::integral_constant<Kind, Kind::L2>{},
                                         std::integral_constant<Kind, Kind::L2>{});
    }

    c_dev_buf.FromDevice(c_rslt_host.data());

    // Optionally run without prefetch for comparison
    if(compare)
    {
        std::cout << "\n=== Running MX GEMM with TDM V1 Pipeline - DataCache Prefetch DISABLED "
                     "===\n"
                  << std::endl;

        DeviceMem c_dev_buf_noprefetch(c_rslt_host.get_element_space_size_in_bytes());

        using ConfigNoPrefetch = MXGemmConfigTDMV1Prefetch<ADataType, Kind::None, Kind::None>;
        invoke_mx_gemm_tdm_v1<ConfigNoPrefetch,
                              ADataType,
                              BDataType,
                              AccDataType,
                              CDataType,
                              ALayout,
                              BLayout,
                              CLayout,
                              AScaleDataType,
                              BScaleDataType,
                              false>(a_dev_buf,
                                     b_dev_buf,
                                     c_dev_buf_noprefetch,
                                     scale_a_dev_buf,
                                     scale_b_dev_buf,
                                     M,
                                     N,
                                     K,
                                     stride_A,
                                     stride_B,
                                     stride_C,
                                     n_warmup,
                                     n_repeat);

        std::cout << "\n=== Comparison Summary ===" << std::endl;
        std::cout << "Check timing above to compare performance with/without data cache prefetch."
                  << std::endl;
    }

    // Validation
    bool pass = true;
    if(arg_parser.get_int("v") == 1)
    {
        HostTensor<CDataType> c_ref_host(
            host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        c_ref_host.SetZero();

        // reference_mx_gemm expects scale_a(M, K/ScaleBlockSize) and scale_b(K/ScaleBlockSize, N)
        // Truncate scale_a from padded M to actual M
        HostTensor<AScaleDataType> scale_a_ref(
            {static_cast<std::size_t>(M), static_cast<std::size_t>(scale_K_dim)},
            {static_cast<std::size_t>(scale_K_dim), std::size_t{1}});
        for(index_t m = 0; m < M; ++m)
            for(index_t k = 0; k < scale_K_dim; ++k)
                scale_a_ref(m, k) = scale_a(m, k);

        // scale_b is (N, K/ScaleBlockSize) row-major; reference expects (K/ScaleBlockSize, N)
        // col-major -> same memory layout, just different descriptor
        HostTensor<BScaleDataType> scale_b_ref(
            {static_cast<std::size_t>(scale_K_dim), static_cast<std::size_t>(N)},
            {std::size_t{1}, static_cast<std::size_t>(scale_K_dim)});
        std::copy(scale_b.mData.begin(), scale_b.mData.end(), scale_b_ref.mData.begin());

        reference_mx_gemm<ADataType,
                          BDataType,
                          AScaleDataType,
                          BScaleDataType,
                          AccDataType,
                          CDataType>(a_host, b_host, c_ref_host, scale_a_ref, scale_b_ref);

        const float rtol = 1e-2;
        const float atol = 1e-2;

        pass = check_err(c_rslt_host, c_ref_host, "Error: Incorrect results!", rtol, atol);

        std::cout << "Relative error threshold: " << rtol << " Absolute error threshold: " << atol
                  << std::endl;
        std::cout << "Verification: " << (pass ? "PASSED" : "FAILED") << std::endl;
    }

    return pass ? 0 : -1;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return EXIT_FAILURE;

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    std::string prec = arg_parser.get_str("mx_prec");

    try
    {
        if(prec == "fp8" || prec == "fp8xfp8")
        {
            return run_mx_gemm_tdm_v1_prefetch<ck_tile::fp8_t,
                                               ck_tile::fp8_t,
                                               float,
                                               ck_tile::half_t,
                                               ck_tile::e8m0_t,
                                               ck_tile::e8m0_t,
                                               Row,
                                               Col,
                                               Row>(argc, argv);
        }
        else if(prec == "fp4" || prec == "fp4xfp4")
        {
            return run_mx_gemm_tdm_v1_prefetch<ck_tile::pk_fp4_t,
                                               ck_tile::pk_fp4_t,
                                               float,
                                               ck_tile::half_t,
                                               ck_tile::e8m0_t,
                                               ck_tile::e8m0_t,
                                               Row,
                                               Col,
                                               Row>(argc, argv);
        }
        else
        {
            std::cerr << "Unsupported precision: " << prec << ". Supported: fp8, fp4" << std::endl;
            return EXIT_FAILURE;
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
