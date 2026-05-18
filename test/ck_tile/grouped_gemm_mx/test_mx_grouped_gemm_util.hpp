// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include <iomanip>
#include <iostream>
#include <sstream>
#include <type_traits>
#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/mx_grouped_gemm_kernel.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

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

/**
 * @brief Pre-shuffle scale buffer for gfx1250 wmma mx scale instruction.
 *
 * Reorganizes the scale data from row-major (MN x K) layout to the hardware-specific
 * layout expected by the gfx1250 wmma instruction.
 *
 * @tparam ScaleType Scale data type (e.g., e8m0_t)
 * @tparam ScaleBlockSize The block size for microscaling (e.g., 32)
 * @tparam KStride Whether K is the fast-moving dimension
 */
template <typename ScaleType, ck_tile::index_t ScaleBlockSize, bool KStride>
void preShuffleScaleBuffer_gfx1250(const ScaleType* src,
                                   ScaleType* dst,
                                   ck_tile::index_t MN,
                                   ck_tile::index_t K)
{
    static_assert(ScaleBlockSize == 32 && sizeof(ScaleType) == 1,
                  "wrong! only support 8-bit scale with ScaleBlockSize=32");

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

template <typename Tuple>
class TestCkTileMxGroupedGemm : public ::testing::Test
{
    protected:
    using ALayout                                    = std::tuple_element_t<0, Tuple>;
    using BLayout                                    = std::tuple_element_t<1, Tuple>;
    using CLayout                                    = std::tuple_element_t<2, Tuple>;
    using ADataType                                  = std::tuple_element_t<3, Tuple>;
    using BDataType                                  = std::tuple_element_t<4, Tuple>;
    using AScaleDataType                             = std::tuple_element_t<5, Tuple>;
    using BScaleDataType                             = std::tuple_element_t<6, Tuple>;
    using AccDataType                                = std::tuple_element_t<7, Tuple>;
    using CDataType                                  = std::tuple_element_t<8, Tuple>;
    using PersistentType                             = std::tuple_element_t<9, Tuple>;
    static constexpr bool Persistent                 = PersistentType::value;
    static constexpr auto Scheduler                  = std::tuple_element_t<10, Tuple>::value;
    static constexpr auto PipelineType               = std::tuple_element_t<11, Tuple>::value;
    static constexpr ck_tile::index_t ScaleBlockSize = std::tuple_element_t<12, Tuple>::value;

    // No D tensors for this test
    using DsLayout   = ck_tile::tuple<>;
    using DsDataType = ck_tile::tuple<>;

    // Compute types match the data types for this pipeline
    using AComputeDataType = ADataType;
    using BComputeDataType = BDataType;

    struct GroupedGemKernelParam_Wmma
    {
        static const bool kPadM = false;
        static const bool kPadN = false;
        static const bool kPadK = false;

        static const int kBlockPerCu         = 1;
        static const ck_tile::index_t M_Tile = 64;
        static const ck_tile::index_t N_Tile = 64;
        static const ck_tile::index_t K_Tile = 128;

        static const ck_tile::index_t M_Warp = 2;
        static const ck_tile::index_t N_Warp = 2;
        static const ck_tile::index_t K_Warp = 1;

        static const ck_tile::index_t M_Warp_Tile     = 32;
        static const ck_tile::index_t N_Warp_Tile     = 32;
        static constexpr ck_tile::index_t K_Warp_Tile = 128;
    };

    using mx_grouped_gemm_kargs = ck_tile::MxGroupedGemmHostArgs<>;
    std::size_t get_workspace_size(const std::vector<mx_grouped_gemm_kargs>& gemm_descs)
    {
        return gemm_descs.size() * sizeof(ck_tile::MxGemmTransKernelArg<>);
    }

    template <typename GroupedGemKernelParam, typename ALayout, typename BLayout, typename CLayout>
    bool invoke_mx_grouped_gemm(const std::vector<mx_grouped_gemm_kargs>& gemm_descs,
                                const ck_tile::stream_config& s,
                                void* kargs_ptr)
    {
        constexpr bool preshuffle       = false;
        constexpr bool DoubleSmemBuffer = true; // TDM pipeline requires double smem buffer
        constexpr bool TransposeC =
            std::is_same_v<CLayout, ck_tile::tensor_layout::gemm::RowMajor> &&
            GroupedGemKernelParam::M_Warp_Tile == GroupedGemKernelParam::N_Warp_Tile;
        static constexpr bool StructuredSparsity = false;
        static constexpr bool NumWaveGroup       = 1;

        constexpr ck_tile::index_t TileParitionerGroupNum = 8;
        constexpr ck_tile::index_t TileParitionerM01      = 4;

        using GemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<GroupedGemKernelParam::M_Tile,
                                                     GroupedGemKernelParam::N_Tile,
                                                     GroupedGemKernelParam::K_Tile>,
                                   ck_tile::sequence<GroupedGemKernelParam::M_Warp,
                                                     GroupedGemKernelParam::N_Warp,
                                                     GroupedGemKernelParam::K_Warp>,
                                   ck_tile::sequence<GroupedGemKernelParam::M_Warp_Tile,
                                                     GroupedGemKernelParam::N_Warp_Tile,
                                                     GroupedGemKernelParam::K_Warp_Tile>>;
        using TilePartitioner = ck_tile::
            GemmSpatiallyLocalTilePartitioner<GemmShape, TileParitionerGroupNum, TileParitionerM01>;

        using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<GroupedGemKernelParam::kPadM,
                                                                     GroupedGemKernelParam::kPadN,
                                                                     GroupedGemKernelParam::kPadK,
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
                                           BScaleDataType>;

        /* make pipeline selective */
        using GemmPipeline =
            typename MxGemmPipelineTypeSelector<PipelineType, UniversalGemmProblem>::pipeline;

        using GemmEpilogue = ck_tile::TdmEpilogue<
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
                                             GroupedGemKernelParam::M_Warp,
                                             GroupedGemKernelParam::N_Warp,
                                             GroupedGemKernelParam::M_Warp_Tile,
                                             GroupedGemKernelParam::N_Warp_Tile,
                                             GroupedGemKernelParam::K_Warp_Tile,
                                             UniversalGemmProblem::TransposeC,
                                             1,                /*kNumWaveGroups_*/
                                             false,            /*FixedVectorSize_*/
                                             1,                /*VectorSizeC_*/
                                             1,                /*BlockedXDLN_PerWarp_*/
                                             DoubleSmemBuffer, /*DoubleSmemBuffer*/
                                             AComputeDataType, /*AComputeDataType_*/
                                             BComputeDataType /*BComputeDataType_*/>>;

        using Kernel = ck_tile::MxGroupedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;

        auto kargs = Kernel::MakeKargs(gemm_descs);
        if(!Kernel::IsSupportedArgument(kargs))
        {
            ADD_FAILURE() << "Kernel " << Kernel::GetName()
                          << " does not support the given arguments"
                             " (set CK_TILE_LOGGING=1 for details)";
            return false;
        }

        const dim3 grids  = Kernel::GridSize(kargs);
        const dim3 blocks = Kernel::BlockSize();
        if(kargs.empty())
            return true;

        ck_tile::hip_check_error(
            hipMemcpyWithStream(kargs_ptr,
                                kargs.data(),
                                kargs.size() * sizeof(ck_tile::MxGemmTransKernelArg<>),
                                hipMemcpyHostToDevice,
                                s.stream_id_));

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel: " << Kernel::GetName() << " with args:" << " grid: {"
                      << grids.x << ", " << grids.y << ", " << grids.z << "}" << ", blocks: {"
                      << blocks.x << ", " << blocks.y << ", " << blocks.z << "}" << std::endl;
        }

        ck_tile::ignore =
            ck_tile::launch_kernel(s,
                                   ck_tile::make_kernel<GroupedGemKernelParam::kBlockPerCu>(
                                       Kernel{},
                                       grids,
                                       blocks,
                                       0,
                                       ck_tile::cast_pointer_to_constant_address_space(kargs_ptr),
                                       kargs.size()));
        return true;
    }

    auto calculate_rtol_atol(const ck_tile::index_t K,
                             const ck_tile::index_t kbatch,
                             const float max_accumulated_value)
    {
        using ComputeType =
            std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;
        // Calculate thresholds
        const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
            ck_tile::integer_divide_ceil(K, kbatch));
        auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
            max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));
        // Calculate error due to split_k accumulation
        const auto rtol_split_k =
            ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
        auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
            max_accumulated_value, kbatch);

        // Extra tolerance for BF16: hardware vs software conversion can differ by ~1 ULP.
        if constexpr(std::is_same_v<CDataType, ck_tile::bf16_t>)
        {
            atol += 0.6f;
            atol_split_k += 0.6f;
        }

        return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
    }

    static constexpr bool check_data_type()
    {

        // Validate scale type / data type combination
        constexpr bool a_is_f4      = std::is_same_v<ADataType, ck_tile::pk_fp4_t>;
        constexpr bool b_is_f4      = std::is_same_v<BDataType, ck_tile::pk_fp4_t>;
        constexpr bool a_scale_e8m0 = std::is_same_v<AScaleDataType, ck_tile::e8m0_t>;
        constexpr bool b_scale_e8m0 = std::is_same_v<BScaleDataType, ck_tile::e8m0_t>;
        if constexpr(!a_is_f4 && !a_scale_e8m0)
            return false;
        if constexpr(!b_is_f4 && !b_scale_e8m0)
            return false;

            // Check hardware WMMA support for the fixed warp tile (32x32x128)
#if defined(CK_USE_GFX1250)
        return ck_tile::has_wmma_traits_v<ck_tile::gfx125_t,
                                          ADataType,
                                          BDataType,
                                          AccDataType,
                                          GroupedGemKernelParam_Wmma::M_Warp_Tile,
                                          GroupedGemKernelParam_Wmma::N_Warp_Tile,
                                          GroupedGemKernelParam_Wmma::K_Warp_Tile>;
#else
        return false;
#endif
    }

    void SetUp() override
    {
        if constexpr(!check_data_type())
        {
            GTEST_SKIP() << "Unsupported data type / layout combination for mx_grouped_gemm.";
        }
    }

    public:
    void Run(const std::vector<int>& Ms,
             const std::vector<int>& Ns,
             const std::vector<int>& Ks,
             const int kbatch      = 1,
             const int group_count = 16)
    {
        if constexpr(!check_data_type())
            return;

        using namespace ck_tile::literals;

        auto f_host_tensor_descriptor = [](std::size_t row,
                                           std::size_t col,
                                           std::size_t stride,
                                           auto layout) {
            if constexpr(std::is_same_v<decltype(layout), ck_tile::tensor_layout::gemm::RowMajor>)
            {
                return ck_tile::HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return ck_tile::HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

        auto f_get_default_stride =
            [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
                if(stride == 0)
                {
                    if constexpr(std::is_same_v<decltype(layout),
                                                ck_tile::tensor_layout::gemm::RowMajor>)
                    {
                        return col;
                    }
                    else
                    {
                        return row;
                    }
                }
                else
                    return stride;
            };

        std::vector<ck_tile::HostTensor<ADataType>> a_m_k_tensors;
        std::vector<ck_tile::HostTensor<BDataType>> b_k_n_tensors;
        std::vector<ck_tile::HostTensor<CDataType>> c_m_n_tensors;

        a_m_k_tensors.reserve(group_count);
        b_k_n_tensors.reserve(group_count);
        c_m_n_tensors.reserve(group_count);

        /* Scale */
        std::vector<ck_tile::HostTensor<AScaleDataType>> scale_a_tensors;
        std::vector<ck_tile::HostTensor<BScaleDataType>> scale_b_tensors;
        scale_a_tensors.reserve(group_count);
        scale_b_tensors.reserve(group_count);

        /* Scale Reference */
        std::vector<ck_tile::HostTensor<AScaleDataType>> scale_a_ref_tensors;
        std::vector<ck_tile::HostTensor<BScaleDataType>> scale_b_ref_tensors;
        scale_a_ref_tensors.reserve(group_count);
        scale_b_ref_tensors.reserve(group_count);

        /* Device */
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> a_m_k_dev_buf;
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> b_k_n_dev_buf;
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> c_m_n_dev_buf;

        a_m_k_dev_buf.reserve(group_count);
        b_k_n_dev_buf.reserve(group_count);
        c_m_n_dev_buf.reserve(group_count);

        /* Scale */
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> scale_a_dev_buf;
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> scale_b_dev_buf;
        scale_a_dev_buf.reserve(group_count);
        scale_b_dev_buf.reserve(group_count);

        std::vector<mx_grouped_gemm_kargs> gemm_descs;
        gemm_descs.reserve(group_count);

        std::vector<int> stride_As(group_count);
        std::vector<int> stride_Bs(group_count);
        std::vector<int> stride_Cs(group_count);

        for(int i = 0; i < group_count; ++i)
        {
            const ck_tile::index_t M = Ms[i];
            const ck_tile::index_t N = Ns[i];
            const ck_tile::index_t K = Ks[i];

            stride_As[i] = f_get_default_stride(M, K, 0, ALayout{});
            stride_Bs[i] = f_get_default_stride(K, N, 0, BLayout{});
            stride_Cs[i] = f_get_default_stride(M, N, 0, CLayout{});

            a_m_k_tensors.push_back(ck_tile::HostTensor<ADataType>(
                f_host_tensor_descriptor(M, K, stride_As[i], ALayout{})));
            b_k_n_tensors.push_back(ck_tile::HostTensor<BDataType>(
                f_host_tensor_descriptor(K, N, stride_Bs[i], BLayout{})));
            c_m_n_tensors.push_back(ck_tile::HostTensor<CDataType>(
                f_host_tensor_descriptor(M, N, stride_Cs[i], CLayout{})));

            std::cout << "gemm[" << i << "]" << " a_m_k: " << a_m_k_tensors[i].mDesc
                      << " b_k_n: " << b_k_n_tensors[i].mDesc
                      << " c_m_n: " << c_m_n_tensors[i].mDesc << " KBatch: " << kbatch << std::endl;

            ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k_tensors[i]);
            ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_k_n_tensors[i]);

            // K must be a multiple of ScaleBlockSize
            if(K % ScaleBlockSize != 0)
            {
                GTEST_SKIP() << "K must be multiple of ScaleBlockSize for MX GEMM";
            }
            const ck_tile::index_t num_scale_k = K / ScaleBlockSize;
            if(num_scale_k % (GroupedGemKernelParam_Wmma::K_Warp_Tile / ScaleBlockSize) != 0)
            {
                GTEST_SKIP() << "K must be a multiple of K_Warp_Tile ("
                             << GroupedGemKernelParam_Wmma::K_Warp_Tile
                             << ") for MX GEMM. Pad the scale data.";
            }
            const ck_tile::index_t scale_padded_M = ck_tile::integer_least_multiple(
                static_cast<ck_tile::index_t>(M),
                static_cast<ck_tile::index_t>(GroupedGemKernelParam_Wmma::M_Warp_Tile));

            ck_tile::HostTensor<AScaleDataType> scale_a(
                {static_cast<std::size_t>(scale_padded_M), static_cast<std::size_t>(num_scale_k)},
                {static_cast<std::size_t>(num_scale_k), static_cast<std::size_t>(1)});

            // scale_b uses N as first dimension (col-major like B)
            ck_tile::HostTensor<BScaleDataType> scale_b(
                {static_cast<std::size_t>(N), static_cast<std::size_t>(num_scale_k)},
                {static_cast<std::size_t>(num_scale_k), static_cast<std::size_t>(1)});

            // Fill scaled data
            {
                std::mt19937 gen(std::chrono::steady_clock::now().time_since_epoch().count());
                std::uniform_real_distribution<float> dist(0.25, 4.f);
                for(auto& s : scale_a.mData)
                {
                    auto v = dist(gen);
                    s      = AScaleDataType{v};
                }
                for(auto& s : scale_b.mData)
                {
                    auto v = dist(gen);
                    s      = BScaleDataType{v};
                }
            }

            // Record in reference scale vector for validation
            {
                scale_b_ref_tensors.push_back(
                    ck_tile::HostTensor<BScaleDataType>(f_host_tensor_descriptor(
                        num_scale_k, N, num_scale_k, ck_tile::tensor_layout::gemm::ColumnMajor{})));
                scale_a_ref_tensors.push_back(
                    ck_tile::HostTensor<AScaleDataType>(f_host_tensor_descriptor(
                        M, num_scale_k, num_scale_k, ck_tile::tensor_layout::gemm::RowMajor{})));

                // Copy scale_b data (our scale_b is (N, num_scale_k) row-major,
                // reference expects (num_scale_k, N) col-major, which is the same memory layout)
                std::copy(scale_b.mData.begin(),
                          scale_b.mData.end(),
                          scale_b_ref_tensors[i].mData.begin());

                // Truncate scale_a to actual M (not padded)
                for(int m = 0; m < M; ++m)
                {
                    for(int k = 0; k < num_scale_k; ++k)
                    {
                        scale_a_ref_tensors[i](m, k) = scale_a(m, k);
                    }
                }
            }

            // Pre-shuffle scale buffers for the hardware
            ck_tile::HostTensor<AScaleDataType> scale_a_shuffled(
                {static_cast<std::size_t>(scale_padded_M), static_cast<std::size_t>(num_scale_k)},
                {static_cast<std::size_t>(num_scale_k), static_cast<std::size_t>(1)});

            ck_tile::HostTensor<BScaleDataType> scale_b_shuffled(
                {static_cast<std::size_t>(N), static_cast<std::size_t>(num_scale_k)},
                {static_cast<std::size_t>(num_scale_k), static_cast<std::size_t>(1)});

            std::cout << " scale_a: [scale_padded_M = " << scale_padded_M
                      << ", num_scale_k = " << num_scale_k << "]." << std::endl;
            std::cout << " scale_b: [N = " << N << ", num_scale_k = " << num_scale_k << "]."
                      << std::endl;

            // Pre-shuffle for gfx1250 (WaveSize=32, WMMA)
            preShuffleScaleBuffer_gfx1250<AScaleDataType, ScaleBlockSize, true>(
                scale_a.mData.data(), scale_a_shuffled.mData.data(), scale_padded_M, num_scale_k);

            // For B scale: B is ColMajor, so scale_b is organized as (N, K/ScaleBlockSize)
            // where N is the fast-changing dimension for col-major B
            preShuffleScaleBuffer_gfx1250<BScaleDataType, ScaleBlockSize, true>(
                scale_b.mData.data(), scale_b_shuffled.mData.data(), N, num_scale_k);

            scale_a_tensors.push_back(scale_a_shuffled);
            scale_b_tensors.push_back(scale_b_shuffled);

            a_m_k_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
                a_m_k_tensors[i].get_element_space_size_in_bytes()));
            b_k_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
                b_k_n_tensors[i].get_element_space_size_in_bytes()));
            c_m_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
                c_m_n_tensors[i].get_element_space_size_in_bytes()));

            a_m_k_dev_buf[i]->ToDevice(a_m_k_tensors[i].data());
            b_k_n_dev_buf[i]->ToDevice(b_k_n_tensors[i].data());
            c_m_n_dev_buf[i]->SetZero();
            c_m_n_tensors[i].SetZero();

            scale_a_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
                scale_a_shuffled.get_element_space_size_in_bytes()));
            scale_b_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
                scale_b_shuffled.get_element_space_size_in_bytes()));
            scale_a_dev_buf[i]->ToDevice(scale_a_shuffled.data());
            scale_b_dev_buf[i]->ToDevice(scale_b_shuffled.data());

            const void* p_a       = a_m_k_dev_buf[i]->GetDeviceBuffer();
            const void* p_b       = b_k_n_dev_buf[i]->GetDeviceBuffer();
            void* p_c             = c_m_n_dev_buf[i]->GetDeviceBuffer();
            const void* p_scale_a = scale_a_dev_buf[i]->GetDeviceBuffer();
            const void* p_scale_b = scale_b_dev_buf[i]->GetDeviceBuffer();

            gemm_descs.push_back(mx_grouped_gemm_kargs(p_a,
                                                       p_scale_a,
                                                       p_b,
                                                       p_scale_b,
                                                       {/*ds_ptr*/},
                                                       p_c,
                                                       kbatch,
                                                       M,
                                                       N,
                                                       K,
                                                       stride_As[i],
                                                       stride_Bs[i],
                                                       {/*stride_Ds*/},
                                                       stride_Cs[i]));
        }

        ck_tile::DeviceMem gemm_workspace;
        gemm_workspace.Realloc(get_workspace_size(gemm_descs));

        if(!invoke_mx_grouped_gemm<GroupedGemKernelParam_Wmma, ALayout, BLayout, CLayout>(
               gemm_descs,
               ck_tile::stream_config{nullptr, false, 1},
               gemm_workspace.GetDeviceBuffer()))
        {
            return;
        }

        // Copy results back to host for validation
        for(int i = 0; i < group_count; i++)
        {
            c_m_n_dev_buf[i]->FromDevice(c_m_n_tensors[i].data());
        }

        bool pass{true};

        for(int i = 0; i < group_count; ++i)
        {
            ck_tile::HostTensor<CDataType> c_m_n_host_ref(
                f_host_tensor_descriptor(Ms[i], Ns[i], stride_Cs[i], CLayout{}));
            c_m_n_host_ref.SetZero();

            ck_tile::reference_mx_gemm<ADataType,
                                       BDataType,
                                       AScaleDataType,
                                       BScaleDataType,
                                       AccDataType,
                                       CDataType>(a_m_k_tensors[i],
                                                  b_k_n_tensors[i],
                                                  c_m_n_host_ref,
                                                  scale_a_ref_tensors[i],
                                                  scale_b_ref_tensors[i]);

            const float max_accumulated_value =
                *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
            const auto rtol_atol = calculate_rtol_atol(Ks[i], kbatch, max_accumulated_value);
            pass &= ck_tile::check_err(c_m_n_tensors[i],
                                       c_m_n_host_ref,
                                       "Error: Incorrect results!",
                                       rtol_atol.at(ck_tile::number<0>{}),
                                       rtol_atol.at(ck_tile::number<1>{}));
        }
        EXPECT_TRUE(pass);
    }
};
