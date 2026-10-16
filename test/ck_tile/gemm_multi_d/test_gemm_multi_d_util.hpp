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
#include "ck_tile/ops/gemm/kernel/gemm_multi_d_kernel.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

struct ElementWiseAddAdd
{
    template <typename E, typename C, typename D0, typename D1>
    CK_TILE_HOST_DEVICE auto operator()(E& e, const C& c, const D0& d0, const D1& d1) const -> void
    {
        const float x0_f = ck_tile::type_convert<float>(c) + ck_tile::type_convert<float>(d0) +
                           ck_tile::type_convert<float>(d1);

        e = ck_tile::type_convert<E>(x0_f);
    }
};

struct MultiplyMultiply
{
    template <typename E, typename C, typename D0, typename D1>
    CK_TILE_HOST_DEVICE auto operator()(E& e, const C& c, const D0& d0, const D1& d1) const -> void
    {
        const float x0_f = ck_tile::type_convert<float>(c) * ck_tile::type_convert<float>(d0) *
                           ck_tile::type_convert<float>(d1);

        e = ck_tile::type_convert<E>(x0_f);
    }
};

// Epilogue/pipeline selectors for tuple element 12 of TestCkTileGemmMultiD:
//   std::true_type  -> CompV3 pipeline + CShuffleEpilogue
//   std::false_type -> CompV3 pipeline + DefaultGemm2DEpilogue
//   MultiDTdmV1     -> CompTDMV1 pipeline + TdmMultiDEpilogue (gfx1250 only)
//   MultiDTdmV2     -> CompTDMV2 pipeline + TdmMultiDEpilogue (gfx1250 only)
// For the TDM selectors, tuple element 13 is TransposeC (default std::false_type) and element 14
// is the block tile as ck_tile::sequence<M, N, K> (default 128x128x64).
struct MultiDTdmV1
{
};
struct MultiDTdmV2
{
};

template <typename T>
inline constexpr bool is_multi_d_tdm_v =
    std::is_same_v<T, MultiDTdmV1> || std::is_same_v<T, MultiDTdmV2>;

// How the D tensors are filled by TestCkTileGemmMultiD::Run.
enum class MultiDFill
{
    Uniform,          // D0, D1 uniform in [-1, 1]
    RampD0IdentityD1, // D0(m, n) = small ramp in (m, n), D1 = identity of CDElementWiseFn
};

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename EDataType,
          typename DsDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeTypeAB =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;

    using ComputeType =
        std::conditional_t<sizeof(ComputeTypeAB) < sizeof(DsDataType), ComputeTypeAB, DsDataType>;

    // Calculate thresholds
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, EDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, EDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));
    // Calculate error due to split_k accumulation
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<EDataType, EDataType, EDataType>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<EDataType, EDataType, EDataType>(
        max_accumulated_value, kbatch);
    // Use higher threshold
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

template <typename Tuple>
class TestCkTileGemmMultiD : public ::testing::Test
{
    protected:
    using ALayout           = std::tuple_element_t<0, Tuple>;
    using BLayout           = std::tuple_element_t<1, Tuple>;
    using D0Layout          = std::tuple_element_t<2, Tuple>;
    using D1Layout          = std::tuple_element_t<3, Tuple>;
    using ELayout           = std::tuple_element_t<4, Tuple>;
    using ADataType         = std::tuple_element_t<5, Tuple>;
    using BDataType         = std::tuple_element_t<6, Tuple>;
    using D0DataType        = std::tuple_element_t<7, Tuple>;
    using D1DataType        = std::tuple_element_t<8, Tuple>;
    using AccDataType       = std::tuple_element_t<9, Tuple>;
    using EDataType         = std::tuple_element_t<10, Tuple>;
    using CDElementWiseFn   = std::tuple_element_t<11, Tuple>;
    using UseCshuffleEpilog = std::tuple_element_t<12, Tuple>;
    using DsLayout          = ck_tile::tuple<D0Layout, D1Layout>;

    using DsDataType = ck_tile::tuple<D0DataType, D1DataType>;
    using ComputeType =
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>;

    static constexpr bool kIsTdm = is_multi_d_tdm_v<UseCshuffleEpilog>;
    static constexpr bool kTdmTransposeC =
        ck_tile::tuple_element_or_default_t<Tuple, 13, std::false_type>::value;
    using TdmBlockTile =
        ck_tile::tuple_element_or_default_t<Tuple, 14, ck_tile::sequence<128, 128, 64>>;

#if CK_TILE_USE_WMMA
    struct GemmWarpConfig_Wmma
    {
        static constexpr ck_tile::index_t M_Tile      = 128;
        static constexpr ck_tile::index_t N_Tile      = 128;
        static constexpr ck_tile::index_t K_Tile      = 64;
        static constexpr ck_tile::index_t M_Warp_Tile = 16;
        static constexpr ck_tile::index_t N_Warp_Tile = 16;
        static constexpr ck_tile::index_t K_Warp_Tile =
            ck_tile::get_k_warp_tile<ComputeType, M_Warp_Tile>();
    };
#else
    struct GemmWarpConfig_Mfma
    {
        static constexpr ck_tile::index_t M_Tile      = 256;
        static constexpr ck_tile::index_t N_Tile      = 256;
        static constexpr ck_tile::index_t K_Tile      = 64;
        static constexpr ck_tile::index_t M_Warp_Tile = 32;
        static constexpr ck_tile::index_t N_Warp_Tile = 32;
        static constexpr ck_tile::index_t K_Warp_Tile = 16;
    };
#endif

    template <typename GemmWarpConfig,
              typename ADataType,
              typename BDataType,
              typename DsDataType,
              typename AccDataType,
              typename EDataType,
              typename ALayout,
              typename BLayout,
              typename DsLayout,
              typename ELayout,
              typename CDEElementWise = ck_tile::element_wise::PassThrough>
    void invoke_gemm_multi_d(const ck_tile::GemmMultiDHostArgs<DsDataType::size()>& args,
                             const ck_tile::stream_config& s)
    {
        constexpr ck_tile::index_t M_Tile = GemmWarpConfig::M_Tile;
        constexpr ck_tile::index_t N_Tile = GemmWarpConfig::N_Tile;
        constexpr ck_tile::index_t K_Tile = GemmWarpConfig::K_Tile;

        constexpr ck_tile::index_t M_Warp = 2;
        constexpr ck_tile::index_t N_Warp = 2;
        constexpr ck_tile::index_t K_Warp = 1;

        constexpr ck_tile::index_t M_Warp_Tile = GemmWarpConfig::M_Warp_Tile;
        constexpr ck_tile::index_t N_Warp_Tile = GemmWarpConfig::N_Warp_Tile;
        constexpr ck_tile::index_t K_Warp_Tile = GemmWarpConfig::K_Warp_Tile;

        constexpr bool DoubleSmemBuffer = false;

        constexpr bool kPadM = false;
        constexpr bool kPadN = false;
        constexpr bool kPadK = false;

        constexpr bool TransposeC = false;

        constexpr int kBlockPerCu                         = 1;
        constexpr ck_tile::index_t TileParitionerGroupNum = 8;
        constexpr ck_tile::index_t TileParitionerM01      = 4;

        using GemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                   ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                                   ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;
        using TilePartitioner = ck_tile::
            GemmSpatiallyLocalTilePartitioner<GemmShape, TileParitionerGroupNum, TileParitionerM01>;

        using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<kPadM,
                                                                     kPadN,
                                                                     kPadK,
                                                                     DoubleSmemBuffer,
                                                                     ALayout,
                                                                     BLayout,
                                                                     ELayout,
                                                                     TransposeC>;

        constexpr auto scheduler = ck_tile::GemmPipelineScheduler::Intrawave;

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                           BDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           scheduler>;

        using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<UniversalGemmProblem>;

        using DefaultGemmEpilogue = ck_tile::DefaultGemm2DEpilogue<
            ck_tile::DefaultGemm2DEpilogueProblem<ADataType,
                                                  BDataType,
                                                  DsDataType,
                                                  AccDataType,
                                                  EDataType,
                                                  DsLayout,
                                                  ELayout,
                                                  CDEElementWise,
                                                  TilePartitioner::MPerBlock,
                                                  TilePartitioner::NPerBlock,
                                                  kPadM,
                                                  kPadN,
                                                  M_Warp_Tile,
                                                  N_Warp_Tile,
                                                  K_Warp_Tile,
                                                  UniversalGemmProblem::TransposeC,
                                                  true>>;

        using CShuffleGemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             DsDataType,
                                             AccDataType,
                                             EDataType,
                                             DsLayout,
                                             ELayout,
                                             CDEElementWise,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             M_Warp,
                                             N_Warp,
                                             M_Warp_Tile,
                                             N_Warp_Tile,
                                             K_Warp_Tile,
                                             UniversalGemmProblem::TransposeC>>;

        using GemmEpilogue =
            std::conditional_t<UseCshuffleEpilog::value, CShuffleGemmEpilogue, DefaultGemmEpilogue>;

        using Kernel = ck_tile::GemmKernelMultiD<TilePartitioner, GemmPipeline, GemmEpilogue>;
        auto kargs   = Kernel::MakeKernelArgs(args);

        const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
        const dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args: " << Kernel::GetName() << '\n'
                      << "shape: " << GemmShape::GetName() << '\n'
                      << "pipeline: " << GemmPipeline::GetName() << '\n'
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
        }

        ck_tile::ignore = ck_tile::launch_kernel(
            s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
    }

    // TDM kernel: CompTDMV1/V2 pipeline, no padding (TDM clips A/B/E), double LDS buffer,
    // spatially local partitioner without cluster, TdmMultiDEpilogue for the D fusion.
    template <typename TdmSelector = UseCshuffleEpilog>
    struct TdmKernelBuilder
    {
        static_assert(is_multi_d_tdm_v<TdmSelector>);

        static constexpr ck_tile::index_t M_Tile = TdmBlockTile::at(ck_tile::number<0>{});
        static constexpr ck_tile::index_t N_Tile = TdmBlockTile::at(ck_tile::number<1>{});
        static constexpr ck_tile::index_t K_Tile = TdmBlockTile::at(ck_tile::number<2>{});

        static constexpr ck_tile::index_t M_Warp = 2;
        static constexpr ck_tile::index_t N_Warp = 2;
        static constexpr ck_tile::index_t K_Warp = 1;

        static constexpr ck_tile::index_t M_Warp_Tile = 16;
        static constexpr ck_tile::index_t N_Warp_Tile = 16;
        static constexpr ck_tile::index_t K_Warp_Tile =
            ck_tile::get_k_warp_tile<ComputeType, M_Warp_Tile>();

        static constexpr bool DoubleSmemBuffer = true;
        static constexpr bool kPadM            = false;
        static constexpr bool kPadN            = false;
        static constexpr bool kPadK            = false;
        static constexpr bool TransposeC       = kTdmTransposeC;

        using GemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                   ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                                   ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;
        using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape, 8, 4>;

        using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<kPadM,
                                                                     kPadN,
                                                                     kPadK,
                                                                     DoubleSmemBuffer,
                                                                     ALayout,
                                                                     BLayout,
                                                                     ELayout,
                                                                     TransposeC>;

        using UniversalGemmProblem =
            ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                  BDataType,
                                                  AccDataType,
                                                  GemmShape,
                                                  GemmUniversalTraits,
                                                  ck_tile::GemmPipelineScheduler::Intrawave>;

        using GemmPipeline =
            std::conditional_t<std::is_same_v<TdmSelector, MultiDTdmV1>,
                               ck_tile::GemmPipelineAgBgCrCompTDMV1<UniversalGemmProblem>,
                               ck_tile::GemmPipelineAgBgCrCompTDMV2<UniversalGemmProblem>>;

        using GemmEpilogue = ck_tile::TdmMultiDEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             DsDataType,
                                             AccDataType,
                                             EDataType,
                                             DsLayout,
                                             ELayout,
                                             CDElementWiseFn,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             M_Warp,
                                             N_Warp,
                                             M_Warp_Tile,
                                             N_Warp_Tile,
                                             K_Warp_Tile,
                                             UniversalGemmProblem::TransposeC,
                                             1,     /*kNumWaveGroups_*/
                                             false, /*FixedVectorSize_*/
                                             1,     /*VectorSizeC_*/
                                             1,     /*BlockedXDLN_PerWarp_*/
                                             DoubleSmemBuffer>>;

        using Kernel = ck_tile::GemmKernelMultiD<TilePartitioner, GemmPipeline, GemmEpilogue>;
    };

    template <typename TdmSelector = UseCshuffleEpilog>
    using TdmKernel = typename TdmKernelBuilder<TdmSelector>::Kernel;

    void invoke_gemm_multi_d_tdm(const ck_tile::GemmMultiDHostArgs<DsDataType::size()>& args,
                                 const ck_tile::stream_config& s)
    {
        using Kernel = TdmKernel<>;
        auto kargs   = Kernel::MakeKernelArgs(args);

        const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
        const dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args: " << Kernel::GetName() << '\n'
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
        }

        ck_tile::ignore =
            ck_tile::launch_kernel(s, ck_tile::make_kernel<1>(Kernel{}, grids, blocks, 0, kargs));
    }

    public:
    void SetUp() override
    {
        if constexpr(kIsTdm)
        {
            if(!ck_tile::is_gfx125_supported())
            {
                // Off gfx125 the host-side argument check must reject the TDM kernel.
                EXPECT_FALSE(IsTdmSupported(256, 256, 256, 1));
                GTEST_SKIP() << "TDM multi-D GEMM requires gfx1250.";
            }
        }
    }

    // Host-only check of Kernel::IsSupportedArgument for the TDM kernel (no launch).
    bool IsTdmSupported(const int M, const int N, const int K, const int k_batch)
    {
        static_assert(kIsTdm, "IsTdmSupported is only meaningful for TDM kernels");
        using Kernel                                                = TdmKernel<>;
        std::array<const void*, DsDataType::size()> ds_ptr          = {nullptr, nullptr};
        std::array<ck_tile::index_t, DsDataType::size()> strides_ds = {N, N};
        ck_tile::GemmMultiDHostArgs<DsDataType::size()> args(
            nullptr, nullptr, ds_ptr, nullptr, k_batch, M, N, K, K, K, strides_ds, N);
        return Kernel::IsSupportedArgument(Kernel::MakeKernelArgs(args));
    }

    bool Run(const int M,
             const int N,
             const int K,
             const int k_batch,
             int StrideA                         = 0,
             int StrideB                         = 0,
             int StrideD0                        = 0,
             int StrideD1                        = 0,
             int StrideE                         = 0,
             MultiDFill d_fill                   = MultiDFill::Uniform,
             bool expect_differs_from_plain_gemm = false)
    {
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

        StrideA  = f_get_default_stride(M, K, StrideA, ALayout{});
        StrideB  = f_get_default_stride(K, N, StrideB, BLayout{});
        StrideD0 = f_get_default_stride(M, N, StrideD0, D0Layout{});
        StrideD1 = f_get_default_stride(M, N, StrideD1, D1Layout{});
        StrideE  = f_get_default_stride(M, N, StrideE, ELayout{});

        ck_tile::HostTensor<ADataType> a_m_k_tesnor(
            f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
        ck_tile::HostTensor<BDataType> b_k_n_tensors(
            f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
        ck_tile::HostTensor<D0DataType> d0_m_n_tensors(
            f_host_tensor_descriptor(M, N, StrideD0, D0Layout{}));
        ck_tile::HostTensor<D1DataType> d1_m_n_tensors(
            f_host_tensor_descriptor(M, N, StrideD1, D1Layout{}));
        ck_tile::HostTensor<EDataType> e_m_n_device_result(
            f_host_tensor_descriptor(M, N, StrideE, ELayout{}));

        ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(a_m_k_tesnor);
        ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(b_k_n_tensors);
        if(d_fill == MultiDFill::Uniform)
        {
            ck_tile::FillUniformDistribution<D0DataType>{-1.f, 1.f}(d0_m_n_tensors);
            ck_tile::FillUniformDistribution<D1DataType>{-1.f, 1.f}(d1_m_n_tensors);
        }
        else
        {
            // Distinct D0 value per (m, n) so a wrong D row/column is visible. D1 is the identity
            // of the elementwise op (1 for multiply, 0 for add), so E depends on D0 only.
            d0_m_n_tensors.ForEach([](auto& self, const auto& idx) {
                const float v = 0.125f * static_cast<float>((idx[0] * 7 + idx[1] * 3) % 17) + 0.5f;
                self(idx)     = ck_tile::type_convert<D0DataType>(v);
            });
            if constexpr(std::is_same_v<CDElementWiseFn, MultiplyMultiply>)
            {
                d1_m_n_tensors.ForEach([](auto& self, const auto& idx) {
                    self(idx) = ck_tile::type_convert<D1DataType>(1.f);
                });
            }
            else
            {
                d1_m_n_tensors.SetZero();
            }
        }

        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k_tesnor.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n_tensors.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d0_m_n_dev_buf(d0_m_n_tensors.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d1_m_n_dev_buf(d1_m_n_tensors.get_element_space_size_in_bytes());
        ck_tile::DeviceMem e_m_n_dev_buf(e_m_n_device_result.get_element_space_size_in_bytes());

        a_m_k_dev_buf.ToDevice(a_m_k_tesnor.mData.data());
        b_k_n_dev_buf.ToDevice(b_k_n_tensors.mData.data());
        d0_m_n_dev_buf.ToDevice(d0_m_n_tensors.mData.data());
        d1_m_n_dev_buf.ToDevice(d1_m_n_tensors.mData.data());

        e_m_n_dev_buf.SetZero();
        e_m_n_device_result.SetZero();
        if constexpr(kIsTdm)
        {
            // Pre-fill every logical E element with a sentinel so a kernel that skips the E store
            // cannot pass a test whose expected output is zero. Stride gaps stay zero, matching
            // the host reference.
            e_m_n_device_result.ForEach([](auto& self, const auto& idx) {
                self(idx) = ck_tile::type_convert<EDataType>(-777.f);
            });
            e_m_n_dev_buf.ToDevice(e_m_n_device_result.data());
            e_m_n_device_result.SetZero();
        }

        std::array<const void*, DsDataType::size()> ds_ptr_buf = {d0_m_n_dev_buf.GetDeviceBuffer(),
                                                                  d1_m_n_dev_buf.GetDeviceBuffer()};
        std::array<ck_tile::index_t, DsDataType::size()> stridesDs = {StrideD0, StrideD1};

        ck_tile::GemmMultiDHostArgs<DsDataType::size()> args({a_m_k_dev_buf.GetDeviceBuffer(),
                                                              b_k_n_dev_buf.GetDeviceBuffer(),
                                                              ds_ptr_buf,
                                                              e_m_n_dev_buf.GetDeviceBuffer(),
                                                              k_batch,
                                                              M,
                                                              N,
                                                              K,
                                                              StrideA,
                                                              StrideB,
                                                              stridesDs,
                                                              StrideE});
        if constexpr(kIsTdm)
        {
            invoke_gemm_multi_d_tdm(args, ck_tile::stream_config{nullptr, false});
        }
        else
        {
#if CK_TILE_USE_WMMA
            invoke_gemm_multi_d<GemmWarpConfig_Wmma,
                                ADataType,
                                BDataType,
                                DsDataType,
                                AccDataType,
                                EDataType,
                                ALayout,
                                BLayout,
                                DsLayout,
                                ELayout,
                                CDElementWiseFn>(args, ck_tile::stream_config{nullptr, false});
#else
            invoke_gemm_multi_d<GemmWarpConfig_Mfma,
                                ADataType,
                                BDataType,
                                DsDataType,
                                AccDataType,
                                EDataType,
                                ALayout,
                                BLayout,
                                DsLayout,
                                ELayout,
                                CDElementWiseFn>(args, ck_tile::stream_config{nullptr, false});
#endif
        }

        std::cout << "Run kernel with M =" << M << " N =" << N << " K =" << K
                  << " StrideA =" << StrideA << " StrideB =" << StrideB << " StrideE =" << StrideE
                  << " StrideD0 =" << StrideD0 << " StrideD1 =" << StrideD1 << std::endl;

        e_m_n_dev_buf.FromDevice(e_m_n_device_result.data());
        bool pass = true;

        ck_tile::HostTensor<EDataType> e_m_n_host_ref(
            f_host_tensor_descriptor(M, N, StrideE, ELayout{}));
        e_m_n_host_ref.SetZero();

        ck_tile::reference_gemm_multiple_d<ADataType,
                                           BDataType,
                                           DsDataType,
                                           AccDataType,
                                           EDataType,
                                           CDElementWiseFn>(
            a_m_k_tesnor, b_k_n_tensors, {d0_m_n_tensors, d1_m_n_tensors}, e_m_n_host_ref);

        const float max_accumulated_value =
            *std::max_element(e_m_n_host_ref.mData.begin(), e_m_n_host_ref.mData.end());
        const auto rtol_atol =
            calculate_rtol_atol<ADataType, BDataType, AccDataType, EDataType, DsDataType>(
                K, k_batch, max_accumulated_value);
        pass = ck_tile::check_err(e_m_n_device_result,
                                  e_m_n_host_ref,
                                  "Error: Incorrect results!",
                                  rtol_atol.at(ck_tile::number<0>{}),
                                  rtol_atol.at(ck_tile::number<1>{}));
        std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                  << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                  << std::endl;

        if(expect_differs_from_plain_gemm)
        {
            // The D fusion must have an effect: compare against the GEMM without Ds.
            ck_tile::HostTensor<EDataType> e_m_n_plain(
                f_host_tensor_descriptor(M, N, StrideE, ELayout{}));
            e_m_n_plain.SetZero();
            ck_tile::reference_gemm<ADataType, BDataType, AccDataType, EDataType>(
                a_m_k_tesnor, b_k_n_tensors, e_m_n_plain);
            const double atol       = rtol_atol.at(ck_tile::number<1>{});
            std::size_t num_diff    = 0;
            std::size_t num_nonzero = 0;
            for(int m = 0; m < M; ++m)
            {
                for(int n = 0; n < N; ++n)
                {
                    const double dev   = ck_tile::type_convert<float>(e_m_n_device_result(m, n));
                    const double plain = ck_tile::type_convert<float>(e_m_n_plain(m, n));
                    if(std::abs(dev - plain) > atol)
                    {
                        ++num_diff;
                    }
                    if(dev != 0.0)
                    {
                        ++num_nonzero;
                    }
                }
            }
            std::cout << "Elements differing from the GEMM without Ds: " << num_diff
                      << ", non-zero elements: " << num_nonzero << std::endl;
            // Most elements must be affected by the D fusion, and E must not be all zero.
            const std::size_t num_elems = static_cast<std::size_t>(M) * N;
            pass = pass && (num_diff >= num_elems / 2) && (num_nonzero >= num_elems / 2);
        }

        return pass;
    }
};
