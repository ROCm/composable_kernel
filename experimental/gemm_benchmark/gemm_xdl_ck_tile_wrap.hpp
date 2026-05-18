// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// #define CK_TILE_FORCE_SINGLE_TAIL_HANDLER 1
#include "ck/tensor_operation/gpu/device/device_gemm_v2.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_mx.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/flatmm.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include "ck_tile/ops/gemm/kernel/mx_gemm_kernel.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"
#include "../../example/ck_tile/03_gemm/gemm_utils.hpp"
#include "../../example/ck_tile/03_gemm/run_gemm_example.inc"
#include "../../example/ck_tile/03_gemm/universal_gemm_invoker.hpp"
#include "ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1.hpp"

static constexpr ck::index_t DisableGfx9I4ToF32 = 0x2000000;

template <ck_tile::GemmPipeline PipelineId>
struct FlatMMPipelineTypeTraits;

template <>
struct FlatMMPipelineTypeTraits<ck_tile::GemmPipeline::PRESHUFFLE_FLATMM>
{
    template <typename ADataType,
              typename BDataType,
              typename CDataType,
              typename BlockGemmShape,
              typename Traits,
              ck_tile::GemmPipelineScheduler Scheduler,
              ck_tile::amd_buffer_coherence_enum BMemNTType,
              bool BPreShufflePermute,
              typename ComputeDataType>
    using PipelineProblem = ck_tile::FlatmmPipelineProblem<ADataType,
                                                           BDataType,
                                                           CDataType,
                                                           BlockGemmShape,
                                                           Traits,
                                                           Scheduler,
                                                           true,
                                                           ck_tile::TailNumber::Full,
                                                           BMemNTType,
                                                           BPreShufflePermute,
                                                           ComputeDataType>;

    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::FlatmmPipelineAGmemBGmemCRegV1<PipelineProblem>;

    template <typename TilePartitioner, typename FlatmmPipeline, typename EpiloguePipeline>
    using GemmKernel = ck_tile::FlatmmKernel<TilePartitioner, FlatmmPipeline, EpiloguePipeline>;
};

template <>
struct FlatMMPipelineTypeTraits<ck_tile::GemmPipeline::PRESHUFFLE_MX_TDM>
{
    template <typename ADataType,
              typename BDataType,
              typename CDataType,
              typename BlockGemmShape,
              typename Traits,
              ck_tile::GemmPipelineScheduler Scheduler,
              ck_tile::amd_buffer_coherence_enum BMemNTType,
              bool BPreShufflePermute,
              typename ComputeDataType>
    using PipelineProblem = ck_tile::MXFlatmmPipelineProblem<ADataType,
                                                             BDataType,
                                                             CDataType,
                                                             BlockGemmShape,
                                                             Traits,
                                                             Scheduler,
                                                             true,
                                                             ck_tile::TailNumber::Full,
                                                             BMemNTType,
                                                             BPreShufflePermute,
                                                             ComputeDataType>;

    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::WeightPreshufflePipelineAGmemBGmemCRegTDM<PipelineProblem>;

    template <typename TilePartitioner, typename MXFlatmmPipeline, typename EpiloguePipeline>
    using GemmKernel = ck_tile::MXFlatmmKernel<TilePartitioner, MXFlatmmPipeline, EpiloguePipeline>;
};

template <>
struct FlatMMPipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_TDM_V1>
{
    template <typename ADataType,
              typename BDataType,
              typename CDataType,
              typename BlockGemmShape,
              typename Traits,
              ck_tile::GemmPipelineScheduler Scheduler,
              ck_tile::amd_buffer_coherence_enum BMemNTType,
              bool BPreShufflePermute,
              typename ComputeDataType>
    using PipelineProblem = ck_tile::MxGemmPipelineProblem<ADataType,
                                                           BDataType,
                                                           CDataType,
                                                           BlockGemmShape,
                                                           Traits,
                                                           Scheduler,
                                                           ck_tile::element_wise::PassThrough,
                                                           ck_tile::element_wise::PassThrough,
                                                           ComputeDataType,
                                                           ComputeDataType,
                                                           ck_tile::e8m0_bexp_t,
                                                           ck_tile::e8m0_bexp_t>;

    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompTDMV1<PipelineProblem>;

    template <typename TilePartitioner, typename MXFlatmmPipeline, typename EpiloguePipeline>
    using GemmKernel = ck_tile::MxGemmKernel<TilePartitioner, MXFlatmmPipeline, EpiloguePipeline>;
};

template <>
struct FlatMMPipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_TDM_V2>
    : public FlatMMPipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_TDM_V1>
{
    template <typename PipelineProblem>
    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompTDMV2<PipelineProblem>;
};

struct FlatMMInvoker
{
    template <typename FlatmmConfig,
              typename ADataType,
              typename BDataType,
              typename DsDatatype,
              typename AccDataType,
              typename CDataType,
              typename ALayout,
              typename BLayout,
              typename DsLayout,
              typename ELayout,
              bool persistent,
              typename CDEElementWise,
              typename CompuateType,
              typename FlatMMHostArg>
    static float
    gemm(const FlatMMHostArg& args, const ck_tile::stream_config& s, bool check_arg_only = false)
    {
        constexpr bool ClusterLaunch =
            FlatmmConfig::kClusterSizeM > 1 || FlatmmConfig::kClusterSizeN > 1;

        using CodegenFlatmmShape = std::conditional_t<
            ClusterLaunch,
            ck_tile::ClusterTileGemmShape<
                ck_tile::sequence<FlatmmConfig::kClusterSizeM, FlatmmConfig::kClusterSizeN, 1>,
                ck_tile::sequence<FlatmmConfig::M_Tile, FlatmmConfig::N_Tile, FlatmmConfig::K_Tile>,
                ck_tile::sequence<FlatmmConfig::M_Warp, FlatmmConfig::N_Warp, FlatmmConfig::K_Warp>,
                ck_tile::sequence<FlatmmConfig::M_Warp_Tile,
                                  FlatmmConfig::N_Warp_Tile,
                                  FlatmmConfig::K_Warp_Tile>>,
            ck_tile::TileGemmShape<
                ck_tile::sequence<FlatmmConfig::M_Tile, FlatmmConfig::N_Tile, FlatmmConfig::K_Tile>,
                ck_tile::sequence<FlatmmConfig::M_Warp, FlatmmConfig::N_Warp, FlatmmConfig::K_Warp>,
                ck_tile::sequence<FlatmmConfig::M_Warp_Tile,
                                  FlatmmConfig::N_Warp_Tile,
                                  FlatmmConfig::K_Warp_Tile>>>;

        using TilePartitioner = std::conditional_t<
            ClusterLaunch,
            ck_tile::GemmClusterTilePartitioner<CodegenFlatmmShape>,
            ck_tile::GemmSpatiallyLocalTilePartitioner<CodegenFlatmmShape,
                                                       FlatmmConfig::TileParitionerGroupNum,
                                                       FlatmmConfig::TileParitionerM01>>;

        using CodegenGemmTraits =
            ck_tile::TileGemmUniversalTraits<FlatmmConfig::kPadM,
                                             FlatmmConfig::kPadN,
                                             FlatmmConfig::kPadK,
                                             FlatmmConfig::DoubleSmemBuffer,
                                             ALayout,
                                             BLayout,
                                             ELayout,
                                             FlatmmConfig::TransposeC,
                                             FlatmmConfig::UseStructuredSparsity,
                                             persistent,
                                             FlatmmConfig::NumWaveGroups,
                                             FlatmmConfig::Preshuffle>;
        float ave_time{0};

        constexpr auto scheduler = FlatmmConfig::Scheduler;

        using CodegenPipelineProblem =
            FlatMMPipelineTypeTraits<FlatmmConfig::Pipeline>::template PipelineProblem<
                ADataType,
                BDataType,
                AccDataType,
                CodegenFlatmmShape,
                CodegenGemmTraits,
                scheduler,
                ck_tile::amd_buffer_coherence_enum::coherence_default,
                false,
                CompuateType>;
        using CodegenFlatmmPipeline = FlatMMPipelineTypeTraits<
            FlatmmConfig::Pipeline>::template GemmPipeline<CodegenPipelineProblem>;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             DsDatatype,
                                             AccDataType,
                                             CDataType,
                                             DsLayout,
                                             ELayout,
                                             CDEElementWise,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             FlatmmConfig::M_Warp,
                                             FlatmmConfig::N_Warp,
                                             FlatmmConfig::M_Warp_Tile,
                                             FlatmmConfig::N_Warp_Tile,
                                             FlatmmConfig::K_Warp_Tile,
                                             CodegenPipelineProblem::TransposeC,
                                             FlatmmConfig::NumWaveGroups,
                                             false,
                                             1,
                                             false,
                                             FlatmmConfig::BlockedXDLN_PerWarp,
                                             FlatmmConfig::DoubleSmemBuffer,
                                             CompuateType>>;

        // ToDo: Will add the codegen part to test different pipeline policies in GEMM.
        // Now we only use the BlockGemmASmemBSmemCRegV1DefaultPolicy.
        using Kernel = FlatMMPipelineTypeTraits<FlatmmConfig::Pipeline>::
            template GemmKernel<TilePartitioner, CodegenFlatmmPipeline, GemmEpilogue>;

        auto kargs       = Kernel::MakeKernelArgs(args);
        const dim3 grids = [&]() {
            if constexpr(FlatmmConfig::Pipeline == ck_tile::GemmPipeline::PRESHUFFLE_FLATMM ||
                         FlatmmConfig::Pipeline == ck_tile::GemmPipeline::PRESHUFFLE_MX_TDM)
            {
                return Kernel::GridSize(kargs);
            }
            else
            {
                return Kernel::GridSize(args.M, args.N, args.k_batch);
            }
        }();
        const dim3 blocks = Kernel::BlockSize();
        if(check_arg_only)
        {
            ave_time = Kernel::IsSupportedArgument(kargs) ? 1.0f : 0.0f;
            return ave_time;
        }

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args:" << CodegenFlatmmShape::GetName() << "\n"
                      << "Shape: " << CodegenFlatmmShape::GetName() << "\n"
                      << "problem: " << CodegenPipelineProblem::GetName() << "\n"
                      << "pipeline: " << CodegenFlatmmPipeline::GetName() << "\n"
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
        }

        std::unique_ptr<ck_tile::RotatingMemWrapper<ADataType, BDataType>> rotating_mem_ptr;
        std::function<void()> preprocess;
        auto clear_gemm_output = [&]() {
            if(args.k_batch > 1)
                hipGetErrorString(hipMemsetAsync(
                    args.e_ptr, 0, args.M * args.N * sizeof(CDataType), s.stream_id_));
        };

        if(s.flush_cache_)
        {
            std::cout << "Flushing cache..." << std::endl;
            static constexpr ck_tile::index_t APackedSize =
                std::is_same_v<ADataType, ck_tile::pk_int4_t> ? 2 : 1;
            static constexpr ck_tile::index_t BPackedSize =
                std::is_same_v<BDataType, ck_tile::pk_int4_t> ? 2 : 1;
            const void* a_ptr         = nullptr;
            const void* b_ptr         = nullptr;
            ck_tile::index_t stride_A = 0;
            ck_tile::index_t stride_B = 0;
            if constexpr(FlatmmConfig::Pipeline == ck_tile::GemmPipeline::PRESHUFFLE_FLATMM ||
                         FlatmmConfig::Pipeline == ck_tile::GemmPipeline::PRESHUFFLE_MX_TDM)
            {
                a_ptr    = kargs.a_ptr;
                b_ptr    = kargs.b_ptr;
                stride_A = args.stride_A;
                stride_B = args.stride_B;
            }
            else
            {
                a_ptr    = kargs.as_ptr[0];
                b_ptr    = kargs.bs_ptr[0];
                stride_A = args.stride_As[0];
                stride_B = args.stride_Bs[0];
            }

            ck_tile::HostTensor<ADataType> a_m(
                ck_tile::host_tensor_descriptor(args.M, args.K, stride_A, is_row_major(ALayout{})));
            ck_tile::HostTensor<BDataType> b_n(
                ck_tile::host_tensor_descriptor(args.K, args.N, stride_B, is_row_major(BLayout{})));

            auto size_a_buffer = a_m.get_element_space_size_in_bytes() / APackedSize;
            auto size_b_buffer = b_n.get_element_space_size_in_bytes() / BPackedSize;

            rotating_mem_ptr = std::make_unique<ck_tile::RotatingMemWrapper<ADataType, BDataType>>(
                a_ptr, b_ptr, s.rotating_count_, size_a_buffer, size_b_buffer);
            rotating_mem_ptr->Print();

            preprocess = [&]() {
                ck_tile::flush_icache();
                rotating_mem_ptr->Next();
                clear_gemm_output();
            };
        }
        else
        {
            preprocess = clear_gemm_output;
        }

        if constexpr(ClusterLaunch)
        {
            dim3 clusters = Kernel::ClusterSize();
            return ck_tile::launch_kernel_time_mask(
                s,
                preprocess,
                ck_tile::make_kernel<FlatmmConfig::kBlockPerCu>(
                    Kernel{}, clusters, grids, blocks, 0, kargs));
        }
        else
        {
            return ck_tile::launch_kernel_time_mask(
                s,
                preprocess,
                ck_tile::make_kernel<FlatmmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
        }
    }
};

namespace ck {

template <typename CkDataType>
constexpr auto GetCkTileDataType()
{
    if constexpr(is_same_v<CkDataType, ck::half_t>)
    {
        return ck_tile::fp16_t{};
    }
    else if constexpr(is_same_v<CkDataType, ck::bhalf_t>)
    {
        return ck_tile::bf16_t{};
    }
    else if constexpr(is_same_v<CkDataType, ck::f8_t>)
    {
        return ck_tile::fp8_t{};
    }
    else if constexpr(is_same_v<CkDataType, ck::bf8_t>)
    {
        return ck_tile::bf8_t{};
    }
    else if constexpr(is_same_v<CkDataType, ck::pk_i4_t>)
    {
        return ck_tile::pk_int4_t{};
    }
    else if constexpr(is_same_v<CkDataType, ck::f4x2_pk_t>)
    {
        return ck_tile::pk_fp4_t{};
    }
    else if constexpr(is_same_v<CkDataType, ck::e8m0_bexp_t>)
    {
        return ck_tile::e8m0_bexp_t{};
    }
    else
    {
        return CkDataType{};
    }
}

namespace tensor_operation {
namespace device {
using AScaleDataTypeCk              = ck::e8m0_bexp_t;
using BScaleDataTypeCk              = ck::e8m0_bexp_t;
constexpr index_t ScaleGranularityK = 32;

template <typename ALayoutCk,
          typename BLayoutCk,
          typename CLayoutCk,
          typename ADataTypeCk,
          typename BDataTypeCk,
          typename CDataTypeCk,
          typename GemmAccDataTypeCk,
          typename CShuffleDataTypeCk,
          typename AElementwiseOperationCk,
          typename BElementwiseOperationCk,
          typename CElementwiseOperationCk,
          typename GemmSpec,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t KPerXDL,
          index_t MWarp,
          index_t NWarp,
          index_t KWarp,
          index_t CShuffleNXdlPerWavePerShuffle = 1,
          typename ComputeDataTypeCk            = ADataTypeCk,
          index_t ClusterSizeM                  = 1,
          index_t ClusterSizeN                  = 1,
          ck_tile::GemmPipelineScheduler PipelineScheduler =
              ck_tile::GemmPipelineScheduler::Intrawave,
          ck_tile::GemmPipeline PipelineVer = ck_tile::GemmPipeline::COMPUTE_V3,
          index_t MinimumOccupancy          = 0>
struct DeviceGemm_Xdl_CkTileWrap : public
#if defined(CK_TILE_WARP_ENABLE_MX)
                                   DeviceGemmMX<ALayoutCk,
#if defined(CK_TILE_WRAP_ENABLE_BPRESHUFFLE)
                                                ck::tensor_layout::gemm::MFMA,
#else
                                                BLayoutCk,
#endif
                                                CLayoutCk,
                                                ADataTypeCk,
                                                AScaleDataTypeCk,
                                                BDataTypeCk,
                                                BScaleDataTypeCk,
                                                CDataTypeCk,
                                                ScaleGranularityK,
                                                AElementwiseOperationCk,
                                                BElementwiseOperationCk,
                                                CElementwiseOperationCk>
#else
#if defined(CK_TILE_WRAP_ENABLE_BPRESHUFFLE)
                                   DeviceGemmV2BPreshuffle
#else
                                   DeviceGemmV2
#endif
                                   <ALayoutCk,
                                    BLayoutCk,
                                    CLayoutCk,
                                    ADataTypeCk,
                                    BDataTypeCk,
                                    CDataTypeCk,
                                    AElementwiseOperationCk,
                                    BElementwiseOperationCk,
                                    CElementwiseOperationCk>
#endif
{
    template <typename CkGemmLayout>
    static constexpr auto GetCkTileGemmLayout()
    {
        if constexpr(is_same_v<CkGemmLayout, ck::tensor_layout::gemm::RowMajor>)
        {
            return ck_tile::tensor_layout::gemm::RowMajor{};
        }
        else if constexpr(is_same_v<CkGemmLayout, ck::tensor_layout::gemm::ColumnMajor>)
        {
            return ck_tile::tensor_layout::gemm::ColumnMajor{};
        }
        else
        {
            static_assert(false);
        }
    }

    template <typename DataType>
    static constexpr auto GetPackedSize()
    {
        if constexpr(is_same_v<DataType, ck_tile::pk_int4_t> ||
                     is_same_v<DataType, ck_tile::pk_fp4_t>)
            return 2;
        else
            return 1;
    }

    template <typename CkElementwiseOperation>
    static constexpr auto GetCkTileElementwiseOperation()
    {
        if constexpr(is_same_v<CkElementwiseOperation,
                               ck::tensor_operation::element_wise::PassThrough>)
        {
            return ck_tile::element_wise::PassThrough{};
        }
        else
        {
            static_assert(0);
            return ck_tile::element_wise::PassThrough{};
        }
    }
    using ALayout          = decltype(GetCkTileGemmLayout<ALayoutCk>());
    using BLayout          = decltype(GetCkTileGemmLayout<BLayoutCk>());
    using CLayout          = decltype(GetCkTileGemmLayout<CLayoutCk>());
    using ADataType        = decltype(GetCkTileDataType<ADataTypeCk>());
    using BDataType        = decltype(GetCkTileDataType<BDataTypeCk>());
    using CDataType        = decltype(GetCkTileDataType<CDataTypeCk>());
    using GemmAccDataType  = decltype(GetCkTileDataType<GemmAccDataTypeCk>());
    using CShuffleDataType = decltype(GetCkTileDataType<CShuffleDataTypeCk>());
    using ComputeDataType  = decltype(GetCkTileDataType<ComputeDataTypeCk>());
    using AElementwiseOperation =
        decltype(GetCkTileElementwiseOperation<AElementwiseOperationCk>());
    using BElementwiseOperation =
        decltype(GetCkTileElementwiseOperation<BElementwiseOperationCk>());
    using CElementwiseOperation =
        decltype(GetCkTileElementwiseOperation<CElementwiseOperationCk>());

    struct GemmConfig
    {
        static constexpr auto I0 = Number<0>{};
        static constexpr auto I1 = Number<1>{};
        static constexpr auto I2 = Number<2>{};

        static constexpr bool kPadM = GemmSpec()[0];
        static constexpr bool kPadN = GemmSpec()[1];
        static constexpr bool kPadK = GemmSpec()[2];

        static constexpr ck_tile::index_t M_Tile = MPerBlock;
        static constexpr ck_tile::index_t N_Tile = NPerBlock;
        static constexpr ck_tile::index_t K_Tile = KPerBlock;

        static constexpr ck_tile::index_t M_Warp = MWarp;
        static constexpr ck_tile::index_t N_Warp = NWarp;
        static constexpr ck_tile::index_t K_Warp = KWarp;

        static constexpr ck_tile::index_t M_Warp_Tile = MPerXDL;
        static constexpr ck_tile::index_t N_Warp_Tile = NPerXDL;
        static constexpr ck_tile::index_t K_Warp_Tile = KPerXDL;

        static constexpr bool TransposeC =
            std::is_same_v<CLayout, ck_tile::tensor_layout::gemm::RowMajor>;
        static constexpr bool UseStructuredSparsity = false;
        static constexpr bool UseDataCachePrefetch  = false;
        static constexpr bool DataCachePrefetchToL1 = false;

        static constexpr auto Scheduler = PipelineScheduler;
        // COMPUTE_V3 is mapped to BASIC_V2 in universal_gemm_invoker.hpp
        static constexpr ck_tile::GemmPipeline Pipeline =
            (PipelineVer == ck_tile::GemmPipeline::COMPUTE_V3) ? ck_tile::GemmPipeline::BASIC_V2
                                                               : PipelineVer;
        static constexpr int kBlockPerCu =
            MinimumOccupancy
                ? MinimumOccupancy
                : (PipelineScheduler == ck_tile::GemmPipelineScheduler::Interwave ? 2 : 1);
        static constexpr ck_tile::index_t NumWaveGroups =
            Pipeline == ck_tile::GemmPipeline::COMPUTE_V5 ? 2 : 1;
        static constexpr bool DoubleSmemBuffer =
            Pipeline == ck_tile::GemmPipeline::COMPUTE_V4 ||
            Pipeline == ck_tile::GemmPipeline::COMPUTE_ASYNC ||
            Pipeline == ck_tile::GemmPipeline::COMPUTE_ASYNC_V2 ||
            Pipeline == ck_tile::GemmPipeline::COMPUTE_TDM_V1 ||
            Pipeline == ck_tile::GemmPipeline::COMPUTE_TDM_V2 ||
            Pipeline == ck_tile::GemmPipeline::PRESHUFFLE_V2 ||
            Pipeline == ck_tile::GemmPipeline::PRESHUFFLE_FLATMM ||
            Pipeline == ck_tile::GemmPipeline::PRESHUFFLE_TDM ||
            Pipeline == ck_tile::GemmPipeline::PRESHUFFLE_MX_TDM ||
            Pipeline == ck_tile::GemmPipeline::COMPUTE_MX_TDM;

        static constexpr bool PermuteA   = false;
        static constexpr bool PermuteB   = false;
        static constexpr bool Preshuffle = Pipeline == ck_tile::GemmPipeline::PRESHUFFLE_V2 ||
                                           Pipeline == ck_tile::GemmPipeline::PRESHUFFLE_FLATMM ||
                                           Pipeline == ck_tile::GemmPipeline::PRESHUFFLE_TDM ||
                                           Pipeline == ck_tile::GemmPipeline::PRESHUFFLE_MX_TDM;
        static constexpr bool TiledMMAPermuteN = false;

        static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
        static constexpr ck_tile::index_t TileParitionerM01      = 4;

        static constexpr ck_tile::index_t kClusterSizeM       = ClusterSizeM;
        static constexpr ck_tile::index_t kClusterSizeN       = ClusterSizeN;
        static constexpr ck_tile::index_t BlockedXDLN_PerWarp = CShuffleNXdlPerWavePerShuffle;
    };

    template <typename DeviceArch_>
    static constexpr index_t GetEstimateVgprCount(DeviceArch_)
    {
        constexpr index_t WaveSize =
            (is_same_v<DeviceArch_, gfx950_t> || is_same_v<DeviceArch_, gfx9_t>) ? 64 : 32;
        constexpr index_t AVgprSize = MPerBlock * KPerBlock / MWarp / WaveSize * sizeof(ADataType) /
                                      GetPackedSize<ADataType>() / sizeof(uint32_t);
        constexpr index_t BVgprSize = NPerBlock * KPerBlock / NWarp / WaveSize * sizeof(BDataType) /
                                      GetPackedSize<BDataType>() / sizeof(uint32_t);
        constexpr index_t AccVgprSize = MPerBlock * NPerBlock / (MWarp * NWarp * WaveSize) *
                                        sizeof(GemmAccDataType) / sizeof(uint32_t);
        if constexpr(PipelineVer == ck_tile::GemmPipeline::BASIC_V1)
        {
            return AVgprSize + BVgprSize + AccVgprSize;
        }
        else if constexpr((PipelineVer == ck_tile::GemmPipeline::BASIC_V2) ||
                          (PipelineVer == ck_tile::GemmPipeline::COMPUTE_V3) ||
                          (PipelineVer == ck_tile::GemmPipeline::MEMORY) ||
                          (PipelineVer == ck_tile::GemmPipeline::COMPUTE_ASYNC) ||
                          (PipelineVer == ck_tile::GemmPipeline::COMPUTE_ASYNC_V2) ||
                          (PipelineVer == ck_tile::GemmPipeline::PRESHUFFLE_V2) ||
                          (PipelineVer == ck_tile::GemmPipeline::PRESHUFFLE_FLATMM) ||
                          (PipelineVer == ck_tile::GemmPipeline::PRESHUFFLE_MX_TDM))
        {
            return 2 * (AVgprSize + BVgprSize) + AccVgprSize;
        }
        else if constexpr(PipelineVer == ck_tile::GemmPipeline::COMPUTE_V4)
        {
            return 3 * (AVgprSize + BVgprSize) + AccVgprSize;
        }
        else if constexpr(PipelineVer == ck_tile::GemmPipeline::COMPUTE_TDM_V1 ||
                          PipelineVer == ck_tile::GemmPipeline::COMPUTE_TDM_V2 ||
                          PipelineVer == ck_tile::GemmPipeline::COMPUTE_MX_TDM)
        {
            constexpr index_t MaxKSubTile = KPerBlock / KPerXDL;
            return math::min(2 * (AVgprSize + BVgprSize),
                             math::max(2 * (AVgprSize + BVgprSize) / MaxKSubTile, 256)) +
                   AccVgprSize;
        }
        else if constexpr(PipelineVer == ck_tile::GemmPipeline::PRESHUFFLE_FLATMM ||
                          PipelineVer == ck_tile::GemmPipeline::PRESHUFFLE_TDM ||
                          PipelineVer == ck_tile::GemmPipeline::PRESHUFFLE_V2)
        {
            return AVgprSize + 2 * BVgprSize + AccVgprSize;
        }
        else
        {
            // invalid pipeline version
            static_assert(0);
        }
    }

    static constexpr index_t GetEstimateSmemSize()
    {
        constexpr index_t MSize =
            MPerBlock * KPerBlock * sizeof(ComputeDataType) / GetPackedSize<ComputeDataType>();
        constexpr index_t NSize =
            NPerBlock * KPerBlock * sizeof(ComputeDataType) / GetPackedSize<ComputeDataType>();
        if constexpr(PipelineVer == ck_tile::GemmPipeline::COMPUTE_V4 ||
                     PipelineVer == ck_tile::GemmPipeline::COMPUTE_ASYNC ||
                     PipelineVer == ck_tile::GemmPipeline::COMPUTE_ASYNC_V2 ||
                     PipelineVer == ck_tile::GemmPipeline::COMPUTE_TDM_V1 ||
                     PipelineVer == ck_tile::GemmPipeline::COMPUTE_TDM_V2 ||
                     PipelineVer == ck_tile::GemmPipeline::COMPUTE_MX_TDM)
        {
            return 2 * (MSize + NSize);
        }
        else if constexpr(PipelineVer == ck_tile::GemmPipeline::PRESHUFFLE_V2 ||
                          PipelineVer == ck_tile::GemmPipeline::PRESHUFFLE_FLATMM ||
                          PipelineVer == ck_tile::GemmPipeline::PRESHUFFLE_TDM ||
                          PipelineVer == ck_tile::GemmPipeline::PRESHUFFLE_MX_TDM)
        {
            return 2 * MSize;
        }
        else
        {
            return MSize + NSize;
        }
    }

#if CK_TILE_USE_WMMA
#if defined(CK_USE_GFX1250)
    using DeviceArch = gfx125_t;
#else
    using DeviceArch = gfx120_t;
#endif
#else
#if defined(CK_GFX950_SUPPORT)
    using DeviceArch = gfx950_t;
#else
    using DeviceArch = gfx9_t;
#endif
#endif

    template <typename DeviceArch_>
    static constexpr bool IsValidCompilationParameter(DeviceArch_ arch)
    {
        static_assert(((NPerBlock / NWarp / NPerXDL) % CShuffleNXdlPerWavePerShuffle) == 0);
        if constexpr(GemmConfig::Pipeline == ck_tile::GemmPipeline::COMPUTE_TDM_V2 ||
                     GemmConfig::Pipeline == ck_tile::GemmPipeline::COMPUTE_TDM_V1 ||
                     GemmConfig::Pipeline == ck_tile::GemmPipeline::PRESHUFFLE_TDM ||
                     GemmConfig::Pipeline == ck_tile::GemmPipeline::PRESHUFFLE_MX_TDM ||
                     GemmConfig::Pipeline == ck_tile::GemmPipeline::COMPUTE_MX_TDM)
        {
            if constexpr(!(is_same_v<DeviceArch_, gfx125_t>))
            {
                return false;
            }
        }
        if constexpr(GemmConfig::Pipeline == ck_tile::GemmPipeline::COMPUTE_ASYNC ||
                     GemmConfig::Pipeline == ck_tile::GemmPipeline::COMPUTE_ASYNC_V2)
        {
            if constexpr(!(is_same_v<DeviceArch_, gfx125_t> || is_same_v<DeviceArch_, gfx950_t> ||
                           is_same_v<DeviceArch_, gfx9_t>))
            {
                return false;
            }
        }

        if constexpr(GemmConfig::Pipeline == ck_tile::GemmPipeline::COMPUTE_TDM_V2)
        {
            if constexpr(GemmConfig::M_Warp * GemmConfig::N_Warp != 4)
            {
                return false;
            }
        }
        if constexpr(GemmConfig::Pipeline == ck_tile::GemmPipeline::BASIC_V1)
        {
            if constexpr(is_same_v<ALayoutCk, ck::tensor_layout::gemm::ColumnMajor> ||
                         is_same_v<BLayoutCk, ck::tensor_layout::gemm::RowMajor>)
            {
                return false;
            }
        }
        if constexpr(GemmConfig::Pipeline == ck_tile::GemmPipeline::COMPUTE_V4 ||
                     GemmConfig::Pipeline == ck_tile::GemmPipeline::MEMORY ||
                     GemmConfig::Pipeline == ck_tile::GemmPipeline::COMPUTE_ASYNC ||
                     GemmConfig::Pipeline == ck_tile::GemmPipeline::COMPUTE_ASYNC_V2 ||
                     GemmConfig::Pipeline == ck_tile::GemmPipeline::COMPUTE_TDM_V2 ||
                     GemmConfig::Pipeline == ck_tile::GemmPipeline::COMPUTE_TDM_V1)
        {
            if constexpr(is_same_v<BDataType, ck_tile::pk_int4_t>)
            {
                return false;
            }
        }

        if constexpr(GemmConfig::Pipeline == ck_tile::GemmPipeline::PRESHUFFLE_FLATMM)
        {
            if constexpr(GemmConfig::M_Warp != 1)
            {
                return false;
            }
        }
        if constexpr(GemmConfig::Pipeline == ck_tile::GemmPipeline::PRESHUFFLE_MX_TDM ||
                     GemmConfig::Pipeline == ck_tile::GemmPipeline::COMPUTE_MX_TDM)
        {
            if constexpr(!(GemmConfig::M_Warp_Tile == 32 && GemmConfig::N_Warp_Tile == 32))
            {
                return false;
            }
        }
        if constexpr(MinimumOccupancy != 0)
        {
            constexpr auto EstimateVgprCount = GetEstimateVgprCount(arch);
            constexpr auto AvailableVgprCount =
                math::min(get_vgpr_count_per_simd(arch) / MinimumOccupancy /
                              (math::integer_divide_ceil(MWarp * NWarp, 4)),
                          get_max_vgpr_count(arch));
            if constexpr(EstimateVgprCount > (AvailableVgprCount + AvailableVgprCount / 4))
            {
                return false;
            }
        }

        constexpr index_t LdsSize = GetEstimateSmemSize();
        if constexpr(LdsSize > get_lds_size(arch))
        {
            return false;
        }
        return true;
    }
#if defined(CK_TILE_WARP_ENABLE_MX)
    using AScaleDataType = decltype(GetCkTileDataType<AScaleDataTypeCk>());
    using BScaleDataType = decltype(GetCkTileDataType<BScaleDataTypeCk>());
    using ScaleAPointer  = ck_tile::FlatmmScalePointer<1, ScaleGranularityK, AScaleDataType>;
    using ScaleBPointer  = ck_tile::FlatmmScalePointer<1, ScaleGranularityK, BScaleDataType>;
#else
    using AScaleDataType = float;
    using BScaleDataType = float;
    using ScaleAPointer  = ck_tile::FlatmmScalePointer<-1, 0, AScaleDataType>;
    using ScaleBPointer  = ck_tile::FlatmmScalePointer<-1, 0, BScaleDataType>;
#endif
    struct Argument : public tensor_operation::device::BaseArgument
    {
        __host__ Argument(const ADataTypeCk* p_a_grid_,
                          const BDataTypeCk* p_b_grid_,
                          CDataTypeCk* p_c_grid_,
                          index_t M_,
                          index_t N_,
                          index_t K_,
                          index_t StrideA_,
                          index_t StrideB_,
                          index_t StrideC_,
                          index_t k_batch_,
                          const AScaleDataTypeCk* p_a_scale_ = nullptr,
                          const BScaleDataTypeCk* p_b_scale_ = nullptr,
                          index_t StrideScaleA_              = 0,
                          index_t StrideScaleB_              = 0)
#if defined(CK_TILE_WARP_ENABLE_MX)
            : host_arg({p_a_grid_},
                       {p_a_scale_},
                       {p_b_grid_},
                       {p_b_scale_},
                       {},
                       p_c_grid_,
                       k_batch_,
                       M_,
                       N_,
                       K_,
                       {StrideA_},
                       {StrideB_},
                       {},
                       StrideC_),
#else
            : host_arg(p_a_grid_,
                       p_b_grid_,
                       p_c_grid_,
                       k_batch_,
                       M_,
                       N_,
                       K_,
                       StrideA_,
                       StrideB_,
                       StrideC_),
#endif
              host_scale_arg(p_a_grid_,
                             p_b_grid_,
                             {},
                             p_c_grid_,
                             k_batch_,
                             M_,
                             N_,
                             K_,
                             StrideA_,
                             StrideB_,
                             {},
                             StrideC_,
                             ScaleAPointer(reinterpret_cast<const AScaleDataType*>(p_a_scale_),
                                           M_ * StrideScaleA_),
                             ScaleBPointer(reinterpret_cast<const BScaleDataType*>(p_b_scale_),
                                           N_ * StrideScaleB_))
        {
        }
#if defined(CK_TILE_WARP_ENABLE_MX)
        ck_tile::MxGemmHostArgs<> host_arg;
#else
        ck_tile::GemmHostArgs host_arg;
#endif
        ck_tile::ScaleFlatmmHostArgs<ScaleAPointer, ScaleBPointer> host_scale_arg;
    };
    using GemmInvoker =
        std::conditional_t<PipelineVer == ck_tile::GemmPipeline::PRESHUFFLE_FLATMM ||
                               PipelineVer == ck_tile::GemmPipeline::PRESHUFFLE_MX_TDM,
                           FlatMMInvoker,
#if defined(CK_TILE_WARP_ENABLE_MX)
                           FlatMMInvoker>;
#else
                           UniversalInvoker>;
#endif
    static constexpr auto& GetHostArg(const Argument& arg)
    {
        if constexpr(PipelineVer == ck_tile::GemmPipeline::PRESHUFFLE_FLATMM ||
                     PipelineVer == ck_tile::GemmPipeline::PRESHUFFLE_MX_TDM)
        {
            return arg.host_scale_arg;
        }
        else
        {
            return arg.host_arg;
        }
    }
    struct Invoker : public BaseInvoker
    {

        float Run(const Argument& arg, const StreamConfig& s = StreamConfig{})
        {
            if constexpr(IsValidCompilationParameter(DeviceArch{}))
            {
                return GemmInvoker::template gemm<GemmConfig,
                                                  ADataType,
                                                  BDataType,
                                                  ck_tile::tuple<>,
                                                  GemmAccDataType,
                                                  CDataType,
                                                  ALayout,
                                                  BLayout,
                                                  ck_tile::tuple<>,
                                                  CLayout,
                                                  false,
                                                  ck_tile::element_wise::PassThrough,
                                                  ComputeDataType>(
                    GetHostArg(arg),
                    ck_tile::stream_config{s.stream_id_,
                                           s.time_kernel_,
                                           s.log_level_,
                                           s.cold_niters_,
                                           s.nrepeat_,
                                           true,
                                           s.flush_cache,
                                           s.rotating_count});
            }
            else
            {
                return 0;
            }
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static bool IsSupportedArgument(const Argument& arg)
    {
        if constexpr(IsValidCompilationParameter(DeviceArch{}))
        {
            return GemmInvoker::template gemm<GemmConfig,
                                              ADataType,
                                              BDataType,
                                              ck_tile::tuple<>,
                                              GemmAccDataType,
                                              CDataType,
                                              ALayout,
                                              BLayout,
                                              ck_tile::tuple<>,
                                              CLayout,
                                              false,
                                              ck_tile::element_wise::PassThrough,
                                              ComputeDataType>(
                       GetHostArg(arg), ck_tile::stream_config{}, true) != 0.0f;
        }
        else
        {
            return false;
        }
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }
#if !defined(CK_TILE_WARP_ENABLE_MX)
    index_t GetKPerBlock() override { return KPerBlock | DisableGfx9I4ToF32; }

    bool GetPermuteA() override { return false; }
    bool GetPermuteB() override { return false; }
#endif
#if defined(CK_TILE_WRAP_ENABLE_BPRESHUFFLE) && !defined(CK_TILE_WARP_ENABLE_MX)
    int GetPreShuffleParameters() override { return NPerXDL; }
#endif

#if defined(CK_TILE_WARP_ENABLE_MX)
    static auto MakeArgument(const ADataTypeCk* p_a,
                             const AScaleDataTypeCk* p_a_scale,
                             const BDataTypeCk* p_b,
                             const BScaleDataTypeCk* p_b_scale,
                             CDataTypeCk* p_c,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t StrideA,
                             index_t StrideScaleA,
                             index_t StrideB,
                             index_t StrideScaleB,
                             index_t StrideC,
                             index_t KBatch,
                             AElementwiseOperationCk,
                             BElementwiseOperationCk,
                             CElementwiseOperationCk)
    {
        return Argument{p_a,
                        p_b,
                        p_c,
                        M,
                        N,
                        K,
                        StrideA,
                        StrideB,
                        StrideC,
                        KBatch,
                        p_a_scale,
                        p_b_scale,
                        StrideScaleA,
                        StrideScaleB};
    }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_a_scale,
                                                      const void* p_b,
                                                      const void* p_b_scale,
                                                      void* p_c,
                                                      index_t M,
                                                      index_t N,
                                                      index_t K,
                                                      index_t StrideA,
                                                      index_t StrideScaleA,
                                                      index_t StrideB,
                                                      index_t StrideScaleB,
                                                      index_t StrideC,
                                                      index_t KBatch,
                                                      AElementwiseOperationCk,
                                                      BElementwiseOperationCk,
                                                      CElementwiseOperationCk) override
    {
        return std::make_unique<Argument>(static_cast<const ADataTypeCk*>(p_a),
                                          static_cast<const BDataTypeCk*>(p_b),
                                          static_cast<CDataTypeCk*>(p_c),
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideC,
                                          KBatch,
                                          static_cast<const AScaleDataTypeCk*>(p_a_scale),
                                          static_cast<const BScaleDataTypeCk*>(p_b_scale),
                                          StrideScaleA,
                                          StrideScaleB);
    }
#else
    static auto MakeArgument(const ADataTypeCk* p_a,
                             const BDataTypeCk* p_b,
                             CDataTypeCk* p_c,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t StrideA,
                             index_t StrideB,
                             index_t StrideC,
                             index_t KBatch,
                             AElementwiseOperationCk,
                             BElementwiseOperationCk,
                             CElementwiseOperationCk)
    {
        return Argument{p_a, p_b, p_c, M, N, K, StrideA, StrideB, StrideC, KBatch};
    }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      void* p_c,
                                                      index_t M,
                                                      index_t N,
                                                      index_t K,
                                                      index_t StrideA,
                                                      index_t StrideB,
                                                      index_t StrideC,
                                                      index_t KBatch,
                                                      AElementwiseOperationCk,
                                                      BElementwiseOperationCk,
                                                      CElementwiseOperationCk) override
    {
        return std::make_unique<Argument>(static_cast<const ADataTypeCk*>(p_a),
                                          static_cast<const BDataTypeCk*>(p_b),
                                          static_cast<CDataTypeCk*>(p_c),
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideC,
                                          KBatch);
    }
#endif
    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        std::map<ck_tile::GemmPipelineScheduler, std::string> PipelineSchedulerToString{
            {ck_tile::GemmPipelineScheduler::Intrawave, "Intrawave"},
            {ck_tile::GemmPipelineScheduler::Interwave, "Interwave"},
            {ck_tile::GemmPipelineScheduler::Default, "Default"}};

        std::map<ck_tile::GemmPipeline, std::string> PipelineToString{
            {ck_tile::GemmPipeline::COMPUTE_ASYNC, "COMPUTE_ASYNC"},
            {ck_tile::GemmPipeline::COMPUTE_V3, "COMPUTE_V3"},
            {ck_tile::GemmPipeline::COMPUTE_V4, "COMPUTE_V4"},
            {ck_tile::GemmPipeline::COMPUTE_V5, "COMPUTE_V5"},
            {ck_tile::GemmPipeline::COMPUTE_V6, "COMPUTE_V6"},
            {ck_tile::GemmPipeline::MEMORY, "MEMORY"},
            {ck_tile::GemmPipeline::BASIC_V1, "BASIC_V1"},
            {ck_tile::GemmPipeline::BASIC_V2, "BASIC_V2"},
            {ck_tile::GemmPipeline::PRESHUFFLE_V2, "PRESHUFFLE_V2"},
            {ck_tile::GemmPipeline::COMPUTE_TDM_V1, "COMPUTE_TDM_V1"},
            {ck_tile::GemmPipeline::COMPUTE_TDM_V2, "COMPUTE_TDM_V2"},
            {ck_tile::GemmPipeline::COMPUTE_ASYNC_V2, "COMPUTE_ASYNC_V2"},
            {ck_tile::GemmPipeline::PRESHUFFLE_FLATMM, "PRESHUFFLE_FLATMM"},
            {ck_tile::GemmPipeline::PRESHUFFLE_TDM, "PRESHUFFLE_TDM"},
            {ck_tile::GemmPipeline::PRESHUFFLE_MX_TDM, "PRESHUFFLE_MX_TDM"},
            {ck_tile::GemmPipeline::COMPUTE_MX_TDM, "COMPUTE_MX_TDM"}};

        auto str = std::stringstream();
        // clang-format off
        str << "DeviceGemm_Xdl_CkTileWrap"
            << "<"
            << std::string(ALayoutCk::name)[0]
            << std::string(BLayoutCk::name)[0]
            << std::string(CLayoutCk::name)[0] << ", "
            << get_type_name<ADataTypeCk>() << ", "
            << get_type_name<BDataTypeCk>() << ", "
            << get_type_name<GemmAccDataTypeCk>() << ", "
            << get_type_name<CDataTypeCk>() << ", "
            << "GemmSepc<" << GemmSpec{}[0] << ", " << GemmSpec{}[1] << ", " << GemmSpec{}[2] << ">, "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << MPerXDL << ", "
            << NPerXDL << ", "
            << KPerXDL << ", "
            << MWarp << ", "
            << NWarp << ", "
            << KWarp << ", "
            << PipelineSchedulerToString[PipelineScheduler]  << ", "
            << PipelineToString[PipelineVer] << ">";
        // clang-format on

        return str.str();
    }
    REGISTER_EXTRA_PRINTING_METHODS
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
