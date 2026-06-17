// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include <sstream>
#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/mx_gemm_kernel.hpp"
#include "ck_tile/core/numeric/math.hpp"

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

template <typename PrecType, ck_tile::index_t M_Warp_Tile>
constexpr ck_tile::index_t get_k_warp_tile()
{
#if CK_TILE_USE_WMMA
#if defined(CK_USE_GFX1250)
    // is_8bit: all 8-bit types (including non-MX int8). is_mxtype: types with MX scale support.
    constexpr bool is_8bit = std::is_same_v<PrecType, ck_tile::fp8_t> ||
                             std::is_same_v<PrecType, ck_tile::bf8_t> ||
                             std::is_same_v<PrecType, ck_tile::int8_t>;
    constexpr bool is_mxtype = std::is_same_v<PrecType, ck_tile::fp8_t> ||
                               std::is_same_v<PrecType, ck_tile::bf8_t> ||
                               std::is_same_v<PrecType, ck_tile::pk_fp4_t>;
    if constexpr(is_mxtype && (M_Warp_Tile == 32 || M_Warp_Tile == 16))
    {
        return 128;
    }
    else
    {
        return is_8bit ? 64 : 32;
    }
#else
    return 16;
#endif
#else
    if constexpr(M_Warp_Tile == 32)
        return 16;
    else
        return 32;
#endif
}

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
        max_accumulated_value, kbatch);
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

enum struct MxGemmPipelineType
{
    CompTDMV1,
    CompTDMV2
};

template <MxGemmPipelineType PT, typename Problem>
struct MxGemmPipelineTypeSelector;

template <typename Problem>
struct MxGemmPipelineTypeSelector<MxGemmPipelineType::CompTDMV1, Problem>
{
    using base_pipeline = ck_tile::BaseGemmPipelineAgBgCrCompTDM<Problem>;
    using pipeline      = ck_tile::GemmPipelineAgBgCrCompTDMV1<Problem>;

    static constexpr auto GetName() { return "GemmPipelineAgBgCrCompTDMV1"; }
};

template <typename Problem>
struct MxGemmPipelineTypeSelector<MxGemmPipelineType::CompTDMV2, Problem>
{
    using base_pipeline = ck_tile::BaseGemmPipelineAgBgCrCompTDM<Problem>;
    using pipeline      = ck_tile::GemmPipelineAgBgCrCompTDMV2<Problem>;

    static constexpr auto GetName() { return "GemmPipelineAgBgCrCompTDMV2"; }
};

template <MxGemmPipelineType PT, typename Problem>
struct MxGemmEpilogueTypeSelector
{
    using epilogue = ck_tile::TdmEpilogue<Problem>;
};

template <MxGemmPipelineType PT>
struct MxGemmPipelineDefaultParams
{
    static constexpr bool PadM       = false;
    static constexpr bool PadN       = false;
    static constexpr bool PadK       = false;
    static constexpr bool Preshuffle = false;
};

/// @brief Pre-shuffle scale buffer for gfx1250 wmma mx scale instruction.
///
/// Reorganizes the scale data from row-major (MN x K) layout to the hardware-specific
/// layout expected by the gfx1250 wmma instruction.
///
/// @tparam ScaleType Scale data type (e.g., e8m0_t)
/// @tparam ScaleBlockSize The block size for microscaling (e.g., 32)
/// @tparam KStride Whether K is the fast-moving dimension
template <typename ScaleType, ck_tile::index_t ScaleBlockSize, bool KStride>
void preShuffleScaleBuffer_gfx1250(const ScaleType* src,
                                   ScaleType* dst,
                                   ck_tile::index_t MN,
                                   ck_tile::index_t K)
{
    static_assert((ScaleBlockSize == 32 || ScaleBlockSize == 16) && sizeof(ScaleType) == 1,
                  "wrong! only support 8-bit scale with ScaleBlockSize=32 or 16");

    // ScaleBlockSize == 16: the natural row-major scale layout already matches the gfx1250
    // wmma scale distribution (one e8m0 per 16 K-elements lands warp-aligned), so the
    // device-side shuffle is the identity transform for all K.
    if constexpr(ScaleBlockSize == 16)
    {
        for(ck_tile::index_t mn = 0; mn < MN; ++mn)
            for(ck_tile::index_t k = 0; k < K; ++k)
            {
                if constexpr(KStride)
                    dst[mn * K + k] = src[mn * K + k];
                else
                    dst[mn * K + k] = src[k * MN + mn];
            }
        return;
    }

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
            {
                dst[outputIndex] = src[mn * K + k];
            }
            else
                dst[outputIndex] = src[k * MN + mn];
        }
    }
}

template <typename Tuple, typename Derived>
class TestCkTileMxGemmPipeline : public ::testing::Test
{
    public:
    using ALayout                      = std::tuple_element_t<0, Tuple>;
    using BLayout                      = std::tuple_element_t<1, Tuple>;
    using CLayout                      = std::tuple_element_t<2, Tuple>;
    using ADataType                    = std::tuple_element_t<3, Tuple>;
    using BDataType                    = std::tuple_element_t<4, Tuple>;
    using AScaleDataType               = std::tuple_element_t<5, Tuple>;
    using BScaleDataType               = std::tuple_element_t<6, Tuple>;
    using AccDataType                  = std::tuple_element_t<7, Tuple>;
    using CDataType                    = std::tuple_element_t<8, Tuple>;
    static constexpr auto Scheduler    = std::tuple_element_t<14, Tuple>::value;
    static constexpr auto PipelineType = std::tuple_element_t<15, Tuple>::value;

    static constexpr ck_tile::index_t M_Tile = std::tuple_element_t<9, Tuple>{};
    static constexpr ck_tile::index_t N_Tile = std::tuple_element_t<10, Tuple>{};
    static constexpr ck_tile::index_t K_Tile = std::tuple_element_t<11, Tuple>{};

    static constexpr ck_tile::index_t M_Warp_Tile = std::tuple_element_t<12, Tuple>{};
    static constexpr ck_tile::index_t N_Warp_Tile = std::tuple_element_t<13, Tuple>{};
    static constexpr ck_tile::index_t K_Warp_Tile = ck_tile::max(
        get_k_warp_tile<ADataType, M_Warp_Tile>(), get_k_warp_tile<BDataType, N_Warp_Tile>());

    using AComputeDataType = ADataType;
    using BComputeDataType = BDataType;

    using DsLayout   = ck_tile::tuple<>;
    using DsDataType = ck_tile::tuple<>;

    static constexpr bool Persistent = false;
    static constexpr bool ClusterLaunch =
        ck_tile::tuple_element_or_default_t<Tuple, 17, std::false_type>::value;

    static constexpr ck_tile::index_t ScaleBlockSize = std::tuple_element_t<16, Tuple>{};

    protected:
    template <bool PadM, bool PadN, bool PadK, bool Preshuffle>
    void invoke_mx_gemm(const ck_tile::MxGemmHostArgs<1, 1, 0>& args,
                        const ck_tile::stream_config& s)
    {
        constexpr ck_tile::index_t M_Warp = 2;
        constexpr ck_tile::index_t N_Warp = 2;
        constexpr ck_tile::index_t K_Warp = 1;

        // if cluster launch is enabled, set cluster dim to 2x2x1
        constexpr ck_tile::index_t kClusterSizeM =
            std::conditional_t<ClusterLaunch, ck_tile::number<2>, ck_tile::number<1>>{};
        constexpr ck_tile::index_t kClusterSizeN =
            std::conditional_t<ClusterLaunch, ck_tile::number<2>, ck_tile::number<1>>{};
        constexpr ck_tile::index_t kClusterSizeK =
            std::conditional_t<ClusterLaunch, ck_tile::number<1>, ck_tile::number<1>>{};

        constexpr bool kPadM      = PadM;
        constexpr bool kPadN      = PadN;
        constexpr bool kPadK      = PadK;
        constexpr bool preshuffle = Preshuffle;

        constexpr bool DoubleSmemBuffer = true; // TDM pipeline requires double smem buffer

#if defined(CK_USE_GFX1250)
        constexpr bool TransposeC =
            std::is_same_v<CLayout, ck_tile::tensor_layout::gemm::RowMajor> &&
            M_Warp_Tile == N_Warp_Tile;
#else
        constexpr bool TransposeC = false;
#endif
        static constexpr bool StructuredSparsity = false;
        static constexpr bool NumWaveGroup       = 1;

        constexpr int kBlockPerCu                         = 1;
        constexpr ck_tile::index_t TileParitionerGroupNum = 8;
        constexpr ck_tile::index_t TileParitionerM01      = 4;

        using GemmShape = std::conditional_t<
            ClusterLaunch,
            ck_tile::ClusterTileGemmShape<
                ck_tile::sequence<kClusterSizeM, kClusterSizeN, kClusterSizeK>,
                ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>,
            ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                   ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                                   ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>>;

        using TilePartitioner =
            std::conditional_t<ClusterLaunch,
                               ck_tile::GemmClusterTilePartitioner<GemmShape>,
                               ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                                          TileParitionerGroupNum,
                                                                          TileParitionerM01>>;

        using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<kPadM,
                                                                     kPadN,
                                                                     kPadK,
                                                                     DoubleSmemBuffer,
                                                                     ALayout,
                                                                     BLayout,
                                                                     CLayout,
                                                                     TransposeC,
                                                                     StructuredSparsity,
                                                                     Persistent,
                                                                     NumWaveGroup,
                                                                     preshuffle>;

        using UniversalGemmProblem =
            ck_tile::MxGemmPipelineProblem<ADataType,
                                           BDataType,
                                           AccDataType,
                                           GemmShape,
                                           GemmUniversalTraits,
                                           Scheduler,
                                           ck_tile::element_wise::PassThrough,
                                           ck_tile::element_wise::PassThrough,
                                           AComputeDataType,
                                           BComputeDataType,
                                           AScaleDataType,
                                           BScaleDataType,
                                           ScaleBlockSize>;

        using GemmPipeline =
            typename MxGemmPipelineTypeSelector<PipelineType, UniversalGemmProblem>::pipeline;

        using GemmEpilogue = typename MxGemmEpilogueTypeSelector<
            PipelineType,
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             DsDataType,
                                             AccDataType,
                                             CDataType,
                                             DsLayout,
                                             CLayout,
                                             ck_tile::element_wise::PassThrough,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             M_Warp,
                                             N_Warp,
                                             M_Warp_Tile,
                                             N_Warp_Tile,
                                             K_Warp_Tile,
                                             UniversalGemmProblem::TransposeC,
                                             1,                /*kNumWaveGroups_*/
                                             false,            /*FixedVectorSize_*/
                                             1,                /*VectorSizeC_*/
                                             1,                /*BlockedXDLN_PerWarp_*/
                                             DoubleSmemBuffer, /*DoubleSmemBuffer*/
                                             AComputeDataType, /*AComputeDataType_*/
                                             BComputeDataType /*BComputeDataType_*/>>::epilogue;

        using Kernel = ck_tile::MxGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
        auto kargs   = Kernel::MakeKernelArgs(args);

        const dim3 blocks = Kernel::BlockSize();
        const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping mx_gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching MxGemm kernel with args:" << " grid: {" << grids.x << ", "
                      << grids.y << ", " << grids.z << "}" << ", blocks: {" << blocks.x << ", "
                      << blocks.y << ", " << blocks.z << "}" << std::endl;
        }

        if constexpr(ClusterLaunch)
        {
            dim3 clusters = Kernel::ClusterSize();
            ck_tile::launch_kernel(
                s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, clusters, grids, blocks, 0, kargs));
        }
        else
        {
            ck_tile::launch_kernel(
                s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
        }
    }

    public:
    void SetUp() override
    {
        if constexpr(!Derived::check_data_type())
        {
            GTEST_SKIP() << "Unsupported data type combination for mx_gemm pipeline test.";
        }
    }

    template <bool PadM       = MxGemmPipelineDefaultParams<PipelineType>::PadM,
              bool PadN       = MxGemmPipelineDefaultParams<PipelineType>::PadN,
              bool PadK       = MxGemmPipelineDefaultParams<PipelineType>::PadK,
              bool Preshuffle = MxGemmPipelineDefaultParams<PipelineType>::Preshuffle>
    void Run(const int M,
             const int N,
             const int K,
             const int StrideA = 0,
             const int StrideB = 0,
             const int StrideC = 0)
    {
        if constexpr(Derived::check_data_type())
        {
            RunSingle<PadM, PadN, PadK, Preshuffle>(M, N, K, StrideA, StrideB, StrideC, 1);
        }
    }

    template <bool PadM, bool PadN, bool PadK, bool Preshuffle>
    void RunSingle(const int M,
                   const int N,
                   const int K,
                   const int StrideA,
                   const int StrideB,
                   const int StrideC,
                   int kbatch = 1)
    {
        using namespace ck_tile;

        // K must be a multiple of ScaleBlockSize
        if(K % ScaleBlockSize != 0)
        {
            GTEST_SKIP() << "K must be multiple of ScaleBlockSize for MX GEMM";
        }

        index_t stride_A = get_default_stride(M, K, StrideA, is_row_major(ALayout{}));
        index_t stride_B = get_default_stride(K, N, StrideB, is_row_major(BLayout{}));
        index_t stride_C = get_default_stride(M, N, StrideC, is_row_major(CLayout{}));

        // Create host tensors for A, B, C
        HostTensor<ADataType> a_m_k(
            host_tensor_descriptor(M, K, stride_A, is_row_major(ALayout{})));
        HostTensor<BDataType> b_k_n(
            host_tensor_descriptor(K, N, stride_B, is_row_major(BLayout{})));
        HostTensor<CDataType> c_m_n_dev_result(
            host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

        // Create host tensors for scale_a and scale_b
        // scale_a: (M, K/ScaleBlockSize) row-major
        // scale_b: (N, K/ScaleBlockSize) col-major
        const index_t num_scale_k = K / ScaleBlockSize;
        // Pre-shuffle interleaves 2 K-lanes (MNPack=2) with MPerXdlops=16 stride,
        // so M must be padded to at least MNPack * MPerXdlops = 32.
        constexpr index_t ScaleShuffleAlign = 32;
        const index_t scale_padded_M        = integer_least_multiple(
            static_cast<index_t>(M),
            static_cast<index_t>(ck_tile::max(M_Warp_Tile, ScaleShuffleAlign)));

        HostTensor<AScaleDataType> scale_a(
            {static_cast<std::size_t>(scale_padded_M), static_cast<std::size_t>(num_scale_k)},
            {static_cast<std::size_t>(num_scale_k), static_cast<std::size_t>(1)});

        // scale_b uses N as first dimension (col-major like B)
        HostTensor<BScaleDataType> scale_b(
            {static_cast<std::size_t>(N), static_cast<std::size_t>(num_scale_k)},
            {static_cast<std::size_t>(num_scale_k), static_cast<std::size_t>(1)});

        // Fill data
        // For pk_fp4_t each byte packs two 4-bit elements; the generic filler
        // converts a single float and duplicates it into both nibbles.
        // Generate two independent random values per byte instead.
        if constexpr(std::is_same_v<ADataType, pk_fp4_t>)
        {
            std::mt19937 gen(11939);
            std::uniform_real_distribution<float> dis(-5.f, 5.f);
            for(auto& elem : a_m_k.mData)
            {
                auto lo = float_to_mxfp4(std::round(dis(gen)), 1.f);
                auto hi = float_to_mxfp4(std::round(dis(gen)), 1.f);
                elem    = pk_fp4_t::_pack(lo, hi);
            }
        }
        else
        {
            FillUniformDistributionIntegerValue<ADataType>{-5, 5, 11939}(a_m_k);
        }
        if constexpr(std::is_same_v<BDataType, pk_fp4_t>)
        {
            std::mt19937 gen(11940);
            std::uniform_real_distribution<float> dis(-5.f, 5.f);
            for(auto& elem : b_k_n.mData)
            {
                auto lo = float_to_mxfp4(std::round(dis(gen)), 1.f);
                auto hi = float_to_mxfp4(std::round(dis(gen)), 1.f);
                elem    = pk_fp4_t::_pack(lo, hi);
            }
        }
        else
        {
            FillUniformDistributionIntegerValue<BDataType>{-5, 5, 11940}(b_k_n);
        }

        {
            // Fill scale tensors with values uniformly drawn from [0.125, 2.0] = [2^-3, 2^1].
            // This spans 5 exponent bands centred around 1.0, keeping scales numerically
            // well-behaved without saturating the accumulator.
            //
            // Per-type raw byte ranges produced (raw bytes sampled uniformly within each):
            //   e8m0_t (bias=127, mant=0): raw in [124, 128] -> floats {0.125, 0.25, 0.5, 1.0, 2.0}
            //   e4m3_t (bias=7,   mant=3): raw in [32,  64]  -> floats  0.125 .. 2.0
            //   e5m3_t (bias=15,  mant=3): raw in [96,  128] -> floats  0.125 .. 2.0
            // No generated value exceeds 2.0 for any type.
            // A and B use different seeds so their scale values are uncorrelated.
            ck_tile::FillUniformScaleDistribution<AScaleDataType>{0.125f, 2.0f, 11941}(scale_a);
            ck_tile::FillUniformScaleDistribution<BScaleDataType>{0.125f, 2.0f, 11943}(scale_b);
        }

        // Pre-shuffle scale buffers for the hardware
        HostTensor<AScaleDataType> scale_a_shuffled(
            {static_cast<std::size_t>(scale_padded_M), static_cast<std::size_t>(num_scale_k)},
            {static_cast<std::size_t>(num_scale_k), static_cast<std::size_t>(1)});

        HostTensor<BScaleDataType> scale_b_shuffled(
            {static_cast<std::size_t>(N), static_cast<std::size_t>(num_scale_k)},
            {static_cast<std::size_t>(num_scale_k), static_cast<std::size_t>(1)});

        // Pre-shuffle for gfx1250 (WaveSize=32, WMMA)
        // Scales start in natural tensor layout and are pre-shuffled into the device layout
        // for both scale block sizes (the shuffle is the identity for ScaleBlockSize==16,
        // whose natural layout already matches the warp scale distribution).
        preShuffleScaleBuffer_gfx1250<AScaleDataType, ScaleBlockSize, true>(
            scale_a.mData.data(), scale_a_shuffled.mData.data(), scale_padded_M, num_scale_k);
        preShuffleScaleBuffer_gfx1250<BScaleDataType, ScaleBlockSize, true>(
            scale_b.mData.data(), scale_b_shuffled.mData.data(), N, num_scale_k);

        // Allocate device memory
        DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
        DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
        DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());
        DeviceMem scale_a_dev_buf(scale_a_shuffled.get_element_space_size_in_bytes());
        DeviceMem scale_b_dev_buf(scale_b_shuffled.get_element_space_size_in_bytes());

        // Upload data to device
        a_m_k_dev_buf.ToDevice(a_m_k.data());
        b_k_n_dev_buf.ToDevice(b_k_n.data());
        c_m_n_dev_buf.SetZero();
        c_m_n_dev_result.SetZero();
        scale_a_dev_buf.ToDevice(scale_a_shuffled.data());
        scale_b_dev_buf.ToDevice(scale_b_shuffled.data());

        // Create MxGemmHostArgs
        ck_tile::MxGemmHostArgs<1, 1, 0> args(
            {static_cast<const void*>(a_m_k_dev_buf.GetDeviceBuffer())},
            {static_cast<const void*>(scale_a_dev_buf.GetDeviceBuffer())},
            {static_cast<const void*>(b_k_n_dev_buf.GetDeviceBuffer())},
            {static_cast<const void*>(scale_b_dev_buf.GetDeviceBuffer())},
            {},
            c_m_n_dev_buf.GetDeviceBuffer(),
            kbatch,
            M,
            N,
            K,
            {stride_A},
            {stride_B},
            {},
            stride_C);

        invoke_mx_gemm<PadM, PadN, PadK, Preshuffle>(args, stream_config{nullptr, false});

        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());

        // Host reference computation using reference_mx_gemm
        // reference_mx_gemm expects scale_a(M, K/ScaleBlockSize) and scale_b(K/ScaleBlockSize, N)
        // We need to create scale_b in (K/ScaleBlockSize, N) format for the reference
        HostTensor<BScaleDataType> scale_b_ref(
            {static_cast<std::size_t>(num_scale_k), static_cast<std::size_t>(N)},
            {static_cast<std::size_t>(1), static_cast<std::size_t>(num_scale_k)});
        // Copy scale_b data (our scale_b is (N, num_scale_k) row-major,
        // reference expects (num_scale_k, N) col-major, which is the same memory layout)
        std::copy(scale_b.mData.begin(), scale_b.mData.end(), scale_b_ref.mData.begin());

        // Truncate scale_a to actual M (not padded)
        HostTensor<AScaleDataType> scale_a_ref(
            {static_cast<std::size_t>(M), static_cast<std::size_t>(num_scale_k)},
            {static_cast<std::size_t>(num_scale_k), static_cast<std::size_t>(1)});
        for(int m = 0; m < M; ++m)
        {
            for(int k = 0; k < num_scale_k; ++k)
            {
                scale_a_ref(m, k) = scale_a(m, k);
            }
        }

        HostTensor<CDataType> c_m_n_host_ref(
            host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        reference_mx_gemm<ADataType,
                          BDataType,
                          AScaleDataType,
                          BScaleDataType,
                          AccDataType,
                          CDataType>(a_m_k, b_k_n, c_m_n_host_ref, scale_a_ref, scale_b_ref);

        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, kbatch, max_accumulated_value);
        bool pass = check_err(c_m_n_dev_result,
                              c_m_n_host_ref,
                              "Error: Incorrect results!",
                              rtol_atol.at(number<0>{}),
                              rtol_atol.at(number<1>{}));
        EXPECT_TRUE(pass);
    }
};
