// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>
#include <type_traits>

#include "ck_tile/host.hpp"
#include "mx_flatmm.hpp"

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

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
          typename ScaleM,
          typename ScaleN,
          bool persistent,
          typename CDEElementWise>
float mx_flatmm_calc(const ck_tile::ScaleFlatmmHostArgs<ScaleM, ScaleN>& args,
                     const ck_tile::stream_config& s)
{
    using CodegenFlatmmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<FlatmmConfig::M_Tile, FlatmmConfig::N_Tile, FlatmmConfig::K_Tile>,
        ck_tile::sequence<FlatmmConfig::M_Warp, FlatmmConfig::N_Warp, FlatmmConfig::K_Warp>,
        ck_tile::sequence<FlatmmConfig::M_Warp_Tile,
                          FlatmmConfig::N_Warp_Tile,
                          FlatmmConfig::K_Warp_Tile>>;

    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<CodegenFlatmmShape,
                                                   FlatmmConfig::TileParitionerGroupNum,
                                                   FlatmmConfig::TileParitionerM01>;

    using Traits = ck_tile::TileGemmTraits<FlatmmConfig::kPadM,
                                           FlatmmConfig::kPadN,
                                           FlatmmConfig::kPadK,
                                           ALayout,
                                           BLayout,
                                           ELayout,
                                           FlatmmConfig::NumWaveGroups>;

    using CodegenGemmTraits = ck_tile::TileGemmUniversalTraits<FlatmmConfig::kPadM,
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
                                                               true>;

    using ComputeDataType = ADataType;
    static_assert(sizeof(ComputeDataType) >= sizeof(BDataType),
                  "mixed_prec_flatmm requires ADataType is a wider type than BDataType");

    using GemmPipelineProblem = ck_tile::GemmPipelineProblem<ComputeDataType,
                                                             ComputeDataType,
                                                             AccDataType,
                                                             CodegenFlatmmShape,
                                                             Traits>;

    using BaseGemmPipeline = ck_tile::BaseFlatmmPipelineAGmemBGmemCRegV1<GemmPipelineProblem>;

    const ck_tile::index_t k_grain     = args.k_batch * FlatmmConfig::K_Tile;
    const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain * FlatmmConfig::K_Tile;
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);
    float ave_time{0};

    const auto Run = [&](const auto has_hot_loop_,
                         const auto tail_number_,
                         const auto memory_operation_) {
        constexpr bool has_hot_loop_v   = has_hot_loop_.value;
        constexpr auto tail_number_v    = tail_number_.value;
        constexpr auto scheduler        = FlatmmConfig::Scheduler;
        constexpr auto memory_operation = memory_operation_.value;

        constexpr int BlockedXDLN_PerWarp = 2; // determined by scale shuffle pattern

        using CodegenPipelineProblem = ck_tile::MXFlatmmPipelineProblem<ADataType,
                                                                        BDataType,
                                                                        AccDataType,
                                                                        CodegenFlatmmShape,
                                                                        CodegenGemmTraits,
                                                                        scheduler,
                                                                        has_hot_loop_v,
                                                                        tail_number_v>;

        using CodegenMXFlatmmPipeline =
            ck_tile::MXF4FlatmmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ComputeDataType,
                                             ComputeDataType,
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
                                             memory_operation,
                                             FlatmmConfig::NumWaveGroups,
                                             false, // FixedVectorSize
                                             1,     // VectorSizeC
                                             FlatmmConfig::TiledMMAPermuteN,
                                             BlockedXDLN_PerWarp>>;

        using Kernel =
            ck_tile::MXFlatmmKernel<TilePartitioner, CodegenMXFlatmmPipeline, GemmEpilogue>;

        auto kargs = Kernel::MakeKernelArgs(args);

        const dim3 grids      = Kernel::GridSize(kargs);
        constexpr dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args:" << CodegenFlatmmShape::GetName() << "\n"
                      << "Shape: " << CodegenFlatmmShape::GetName() << "\n"
                      << "problem: " << CodegenPipelineProblem::GetName() << "\n"
                      << "pipeline: " << CodegenMXFlatmmPipeline::GetName() << "\n"
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
        }

        // Declare rotating_mem_ptr here so it stays in scope until it is needed
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
            constexpr ck_tile::index_t APackedSize = ck_tile::numeric_traits<ADataType>::PackedSize;
            constexpr ck_tile::index_t BPackedSize = ck_tile::numeric_traits<BDataType>::PackedSize;

            ck_tile::HostTensor<ADataType> a_m(ck_tile::host_tensor_descriptor(
                args.M, args.K, args.stride_A, is_row_major(ALayout{})));
            ck_tile::HostTensor<BDataType> b_n(ck_tile::host_tensor_descriptor(
                args.K, args.N, args.stride_B, is_row_major(BLayout{})));

            auto size_a_buffer = a_m.get_element_space_size_in_bytes() / APackedSize;
            auto size_b_buffer = b_n.get_element_space_size_in_bytes() / BPackedSize;

            rotating_mem_ptr = std::make_unique<ck_tile::RotatingMemWrapper<ADataType, BDataType>>(
                kargs.a_ptr, kargs.b_ptr, s.rotating_count_, size_a_buffer, size_b_buffer);
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

        ave_time = ck_tile::launch_kernel_time_mask(
            s,
            preprocess,
            ck_tile::make_kernel<FlatmmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
        return ave_time;
    };

    const auto RunSplitk = [&](const auto has_hot_loop_, const auto tail_number_) {
        if(args.k_batch == 1)
        {
            Run(has_hot_loop_,
                tail_number_,
                ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                           ck_tile::memory_operation_enum::set>{});
        }
        else
        {
            Run(has_hot_loop_,
                tail_number_,
                ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                           ck_tile::memory_operation_enum::atomic_add>{});
        }
    };
    BaseGemmPipeline::TailHandler(RunSplitk, has_hot_loop, tail_num);
    return ave_time;
}

template <typename FlatmmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDatatype,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename ScaleA,
          typename ScaleB,
          bool UsePersistentKernel = false,
          typename CDEElementWise  = ck_tile::element_wise::PassThrough>
float invoke_mx_flatmm(ck_tile::DeviceMem& a_dev_buf,
                       ck_tile::DeviceMem& b_shuffle_dev_buf,
                       ck_tile::DeviceMem& c_dev_buf,
                       ck_tile::index_t M,
                       ck_tile::index_t N,
                       ck_tile::index_t K,
                       ck_tile::index_t stride_A,
                       ck_tile::index_t stride_B,
                       ck_tile::index_t stride_C,
                       ck_tile::index_t kbatch,
                       ScaleA scale_a,
                       ScaleB scale_b,
                       int n_warmup,
                       int n_repeat)
{
    ck_tile::ScaleFlatmmHostArgs<ScaleA, ScaleB> args = {a_dev_buf.GetDeviceBuffer(),
                                                         b_shuffle_dev_buf.GetDeviceBuffer(),
                                                         {},
                                                         c_dev_buf.GetDeviceBuffer(),
                                                         kbatch,
                                                         M,
                                                         N,
                                                         K,
                                                         stride_A,
                                                         stride_B,
                                                         {},
                                                         stride_C,
                                                         scale_a,
                                                         scale_b};

    float ave_time = mx_flatmm_calc<FlatmmConfig,
                                    ADataType,
                                    BDataType,
                                    DsDatatype,
                                    AccDataType,
                                    CDataType,
                                    ALayout,
                                    BLayout,
                                    DsLayout,
                                    CLayout,
                                    ScaleA,
                                    ScaleB,
                                    UsePersistentKernel,
                                    CDEElementWise>(
        args, ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat, true, true, 50});

    constexpr int APackedSize = ck_tile::numeric_traits<ADataType>::PackedSize;
    constexpr int BPackedSize = ck_tile::numeric_traits<BDataType>::PackedSize;

    std::size_t flop     = std::size_t(2) * M * N * K + std::size_t(2) * M * N * K / 32;
    std::size_t num_byte = sizeof(ADataType) * M * K / APackedSize +
                           sizeof(BDataType) * N * K / BPackedSize + sizeof(CDataType) * M * N +
                           sizeof(ck_tile::e8m0_t) * M * K / 32 +
                           sizeof(ck_tile::e8m0_t) * N * K / 32;
    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Run MXFP4_Flatmm kernel " //
              << " M =" << M << " N =" << N << " K =" << K << " StrideA =" << stride_A
              << " StrideB =" << stride_B << " StrideC =" << stride_C << " : " << ave_time
              << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, " << std::endl;

    return ave_time;
}

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "32", "m dimension")
        .insert("n", "128", "n dimension")
        .insert("k", "256", "k dimension")
        .insert("a_layout", "R", "A tensor data layout - Row by default")
        .insert("b_layout", "C", "B tensor data layout - Row by default")
        .insert("c_layout", "R", "C tensor data layout - Row by default")
        .insert("stride_a", "0", "Tensor A stride")
        .insert("stride_b", "0", "Tensor B stride")
        .insert("stride_c", "0", "Tensor C stride")
        .insert("v", "1", "0. No validation, 1. Validation on CPU, 2. Validation on GPU")
        .insert(
            "mx_prec", "fp4xfp4", "data type for activation and weight, support: fp6xfp6, fp8xfp8")
        .insert("warmup", "50", "number of iterations before benchmark the kernel")
        .insert("repeat", "100", "number of iterations to benchmark the kernel")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer")
        .insert("split_k", "1", "splitK value")
        .insert("init", "0", "0:random, 1:constant(1)")
        .insert("persistent", "0", "0: no persistent, 1: persistent kernel")
        .insert("warp_tile",
                "0",
                "0: 16x16, 1: 32x32, 2: 16x16x128 (950 only), 3: 32x32x64 (950 only)");
    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <class FlatmmConfig, class IterSrc, class IterDst>
void preShuffleWeight(const IterSrc src, IterDst dst, int N, int K)
{
    int KPack = 16;
    int NLane = FlatmmConfig::N_Warp_Tile;
    int KLane = 64 / NLane;
    int K_pk  = K / 2;
    int K0    = K_pk / (KLane * KPack);
    // K -> K0 KLane KPack
    // N -> N0 NLane
    // N, K -> N0 K0 KLane NLane KPack
    int tempk;
    for(int n = 0; n < N; ++n)
    {
        for(int k = 0; k < K_pk; ++k)
        {
            int n0 = n / NLane;
            int n1 = n % NLane;

            int k0 = k / (KLane * KPack);
            tempk  = k % (KLane * KPack);
            int k1 = tempk / KPack;
            int k2 = tempk % KPack;

            int outputIndex = n0 * KPack * NLane * KLane * K0 + k0 * KPack * NLane * KLane +
                              k1 * KPack * NLane + n1 * KPack + k2;

            dst[outputIndex] = src[n * K_pk + k];
        }
    }
}

template <class FlatmmConfig, bool KLast, typename Src>
auto preShuffleScale(Src& src)
{
    using dtype      = typename Src::Data::value_type;
    auto src_lengths = src.get_lengths();
    const auto MN    = KLast ? src_lengths[0] : src_lengths[1];
    const auto K     = KLast ? src_lengths[1] : src_lengths[0];

    size_t MNXdlPack   = 2;
    size_t KXdlPack    = 2;
    size_t XdlMNThread = FlatmmConfig::N_Warp_Tile; // 16
    size_t XdlKThread  = 64 / XdlMNThread;

    const auto MN_Paded = ck_tile::integer_least_multiple(MN, XdlMNThread * MNXdlPack);

    ck_tile::HostTensor<dtype> shuffled(ck_tile::HostTensorDescriptor({MN_Paded * K}, {1}));

    size_t K0 = K / KXdlPack / XdlKThread; // KRepeat

    // The 4 16x128 building blocks will be packed into 1 32x256 for F4
    // The 8 16x16x128 mfma will be packed into 1 32x32x256 for F4

    // unfold the MN32xK(256/32) scale buffer
    //    4            16             2           2
    // To XdlKThread-> XdlMNThread -> KXdlPack -> MNXdlPack
    // Then, MNRepeat->KRepeat

    for(size_t n = 0; n < MN_Paded; ++n)
    {
        for(size_t k = 0; k < K; ++k)
        {
            auto n0    = n / (XdlMNThread * MNXdlPack); // i MNRepeat
            auto tempn = n % (XdlMNThread * MNXdlPack);
            auto n1    = tempn % XdlMNThread; // i XdlMNThread
            auto n2    = tempn / XdlMNThread; // i MNXdlPack

            auto k0    = k / (XdlKThread * KXdlPack); // i KRepeat
            auto tempk = k % (XdlKThread * KXdlPack);
            auto k1    = tempk % XdlKThread; // i XdlKThread
            auto k2    = tempk / XdlKThread; // i KXdlPack

            auto outputIndex = n0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread * K0 +
                               k0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread +
                               k1 * MNXdlPack * KXdlPack * XdlMNThread + n1 * MNXdlPack * KXdlPack +
                               k2 * MNXdlPack + n2;

            if constexpr(KLast)
                shuffled(outputIndex) = n < MN ? src(n, k) : dtype{};
            else
                shuffled(outputIndex) = n < MN ? src(k, n) : dtype{};
        }
    }
    return shuffled;
}

#include "run_mx_flatmm.inc"

template <typename FlatmmConfig>
int run_mx_flatmm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    std::string mx_prec  = arg_parser.get_str("mx_prec");
    std::string a_layout = arg_parser.get_str("a_layout");
    std::string b_layout = arg_parser.get_str("b_layout");
    int persistent_opt   = arg_parser.get_int("persistent");

    if(a_layout == "R" && b_layout == "C")
    {
        if(mx_prec == "fp4xfp4")
        {
            if(persistent_opt == 0)
            {
                run_mx_flatmm_with_layouts<ck_tile::pk_fp4_t,
                                           ck_tile::pk_fp4_t,
                                           ck_tile::fp16_t,
                                           FlatmmConfig,
                                           false>(argc, argv, Row{}, Col{}, Row{});
            }
            else
            {
                run_mx_flatmm_with_layouts<ck_tile::pk_fp4_t,
                                           ck_tile::pk_fp4_t,
                                           ck_tile::fp16_t,
                                           FlatmmConfig,
                                           true>(argc, argv, Row{}, Col{}, Row{});
            }
        }
        else if(mx_prec == "fp6xfp6")
        {
            throw std::runtime_error("Only support fp4xfp4 now!");
        }
        else if(mx_prec == "fp8xfp8")
        {
            throw std::runtime_error("Only support fp4xfp4 now!");
        }
        else
        {
            throw std::runtime_error("Unsupported data_type!");
        }
    }
    else
    {
        throw std::runtime_error("Unsupported data layout configuration for A,B and C tensors!");
    }
    return -1;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return EXIT_FAILURE;
    try
    {
        int warp_tile = arg_parser.get_int("warp_tile");
        if(warp_tile == 0)
        {
            return !run_mx_flatmm_example<MXfp4_FlatmmConfig16>(argc, argv);
        }
        else if(warp_tile == 1)
        {
            throw std::runtime_error("Only support MFMA_16x16x128 now!");
        }
        else
        {
            throw std::runtime_error("Unsupported warp_tile!");
        }
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
