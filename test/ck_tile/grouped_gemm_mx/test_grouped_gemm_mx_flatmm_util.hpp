// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <type_traits>
#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/reference/reference_gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/flatmm.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "mx_flatmm_arch_traits.hpp"

// Initialization proxy: pk_fp6x16_t lacks converters for FillUniformDistribution.
// Initialize through pk_fp6_t and memcpy the raw bytes.
template <typename T>
using mx_flatmm_init_proxy_t =
    std::conditional_t<std::is_same_v<T, ck_tile::pk_fp6x16_t>, ck_tile::pk_fp6_t, T>;

template <ck_tile::index_t NLane, typename dtype>
auto preShuffleWeight(ck_tile::HostTensor<dtype>& src)
{
    auto src_lengths          = src.get_lengths();
    const int K               = src_lengths[0];
    const int N               = src_lengths[1];
    constexpr int packed_size = ck_tile::numeric_traits<dtype>::PackedSize;

    int KPack = std::is_same_v<dtype, ck_tile::pk_fp6x16_t> ? 32 : 16 * packed_size;

    int KLane = ck_tile::get_warp_size() / NLane;
    int K0    = K / (KLane * KPack);

    ck_tile::HostTensor<dtype> shuffled(ck_tile::HostTensorDescriptor({N * K}, {1}));

    for(int n = 0; n < N; ++n)
    {
        for(int k = 0; k < K; k += packed_size)
        {
            int n0 = n / NLane;
            int n1 = n % NLane;

            int k0    = k / (KLane * KPack);
            int tempk = k % (KLane * KPack);
            int k1    = tempk / KPack;
            int k2    = tempk % KPack;

            int outputIndex = n0 * KPack * NLane * KLane * K0 + k0 * KPack * NLane * KLane +
                              k1 * KPack * NLane + n1 * KPack + k2;

            shuffled(outputIndex) = src(k, n);
        }
    }
    return shuffled;
}

// Tuple layout: <ADataType, BDataType, CDataType, MXFlatmmArchTraits>
template <typename Tuple>
class TestGroupedGemmMXFlatmm : public ::testing::Test
{
    protected:
    using ADataType          = std::tuple_element_t<0, Tuple>;
    using BDataType          = std::tuple_element_t<1, Tuple>;
    using CDataType          = std::tuple_element_t<2, Tuple>;
    using MXFlatmmArchTraits = std::tuple_element_t<3, Tuple>;

    using AInitType = mx_flatmm_init_proxy_t<ADataType>;
    using BInitType = mx_flatmm_init_proxy_t<BDataType>;

    static_assert(ck_tile::numeric_traits<AInitType>::PackedSize ==
                      ck_tile::numeric_traits<ADataType>::PackedSize,
                  "init proxy must share PackedSize with ADataType");
    static_assert(ck_tile::numeric_traits<BInitType>::PackedSize ==
                      ck_tile::numeric_traits<BDataType>::PackedSize,
                  "init proxy must share PackedSize with BDataType");

    using FlatmmConfig = typename MXFlatmmArchTraits::Config;
    using AccDataType  = float;
    using ScaleType    = ck_tile::e8m0_t;

    using ALayout = ck_tile::tensor_layout::gemm::RowMajor;
    using BLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck_tile::tensor_layout::gemm::RowMajor;

    using DsLayout   = ck_tile::tuple<>;
    using DsDataType = ck_tile::tuple<>;

    static constexpr int ScaleGranularityM = 1;
    static constexpr int ScaleGranularityN = 1;
    static constexpr int ScaleGranularityK = 32;

    using ScaleA = ck_tile::FlatmmScalePointer<ScaleGranularityM, ScaleGranularityK, ScaleType>;
    using ScaleB = ck_tile::FlatmmScalePointer<ScaleGranularityN, ScaleGranularityK, ScaleType>;

    static constexpr ck_tile::index_t NumDTensor = 0;

    static auto copy_proxy_to_real(auto& dst, const auto& src)
    {
        using DstType = std::remove_cvref_t<decltype(*dst.data())>;
        using SrcType = std::remove_cvref_t<decltype(*src.data())>;
        if constexpr(std::is_same_v<DstType, SrcType>)
        {
            dst = src;
        }
        else
        {
            // Only relevant for fp6x16_t -> fp6_t conversion.
            // size of fp6x16_t is 96 bits, size of fp6_t is 128 bits.
            static_assert(sizeof(DstType) <= sizeof(SrcType),
                          "DstType must be smaller or equal to SrcType");
            const auto n = dst.get_element_space_size();
            for(std::size_t i = 0; i < n; ++i)
            {
                std::memcpy(&dst.mData[i], &src.mData[i], sizeof(DstType));
            }
        }
    }

    public:
    void Run(const std::vector<int>& Ms,
             const std::vector<int>& Ns,
             const std::vector<int>& Ks,
             const int group_count,
             const int init_method = 1)
    {
        using namespace ck_tile::literals;

        constexpr int APackedSize = ck_tile::numeric_traits<ADataType>::PackedSize;
        constexpr int BPackedSize = ck_tile::numeric_traits<BDataType>::PackedSize;

        constexpr bool a_row_major = true;
        constexpr bool b_row_major = false;
        constexpr bool c_row_major = true;

        // Per-group host tensors
        std::vector<ck_tile::HostTensor<ADataType>> a_tensors;
        std::vector<ck_tile::HostTensor<BDataType>> b_origin_tensors;
        std::vector<ck_tile::HostTensor<CDataType>> c_tensors;
        std::vector<ck_tile::HostTensor<ScaleType>> scale_a_tensors;
        std::vector<ck_tile::HostTensor<ScaleType>> scale_b_tensors;

        // Per-group shuffled tensors
        std::vector<ck_tile::HostTensor<BDataType>> b_shuffled_tensors;
        std::vector<ck_tile::HostTensor<ScaleType>> scale_a_shuffled_tensors;
        std::vector<ck_tile::HostTensor<ScaleType>> scale_b_shuffled_tensors;

        // Per-group device buffers
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> a_dev_bufs;
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> b_dev_bufs;
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> c_dev_bufs;
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> scale_a_dev_bufs;
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> scale_b_dev_bufs;

        // Per-group host args arrays (struct-of-arrays for GroupedFlatmmHostArgs)
        std::vector<ck_tile::index_t> h_Ms(group_count);
        std::vector<ck_tile::index_t> h_Ns(group_count);
        std::vector<ck_tile::index_t> h_Ks(group_count);
        std::vector<const void*> h_a_ptrs(group_count);
        std::vector<const void*> h_b_ptrs(group_count);
        std::vector<void*> h_c_ptrs(group_count);
        std::vector<ck_tile::index_t> h_stride_As(group_count);
        std::vector<ck_tile::index_t> h_stride_Bs(group_count);
        std::vector<ck_tile::index_t> h_stride_Cs(group_count);
        std::vector<ScaleA> h_scale_as(group_count);
        std::vector<ScaleB> h_scale_bs(group_count);

        for(int i = 0; i < group_count; ++i)
        {
            const ck_tile::index_t M = Ms[i];
            const ck_tile::index_t N = Ns[i];
            const ck_tile::index_t K = Ks[i];

            ASSERT_EQ(K % ScaleGranularityK, 0) << "K must be a multiple of ScaleGranularityK (32)";
            ASSERT_EQ(K % APackedSize, 0) << "K must be a multiple of A PackedSize";
            ASSERT_EQ(K % BPackedSize, 0) << "K must be a multiple of B PackedSize";

            const ck_tile::index_t stride_A =
                ck_tile::get_default_stride(M, K, 0, ck_tile::bool_constant<a_row_major>{});
            const ck_tile::index_t stride_B =
                ck_tile::get_default_stride(K, N, 0, ck_tile::bool_constant<b_row_major>{});
            const ck_tile::index_t stride_C =
                ck_tile::get_default_stride(M, N, 0, ck_tile::bool_constant<c_row_major>{});

            const auto scale_stride_A =
                ck_tile::get_default_stride(M / ScaleGranularityM,
                                            K / ScaleGranularityK,
                                            0,
                                            ck_tile::bool_constant<a_row_major>{});
            const auto scale_stride_B =
                ck_tile::get_default_stride(K / ScaleGranularityK,
                                            N / ScaleGranularityN,
                                            0,
                                            ck_tile::bool_constant<b_row_major>{});

            // Create host tensors
            ck_tile::HostTensor<ADataType> a_host(ck_tile::host_tensor_descriptor(
                M, K, stride_A, ck_tile::bool_constant<a_row_major>{}));
            ck_tile::HostTensor<BDataType> b_origin_host(ck_tile::host_tensor_descriptor(
                K, N, stride_B, ck_tile::bool_constant<b_row_major>{}));
            ck_tile::HostTensor<CDataType> c_host(ck_tile::host_tensor_descriptor(
                M, N, stride_C, ck_tile::bool_constant<c_row_major>{}));

            ck_tile::HostTensor<ScaleType> scale_a(
                ck_tile::host_tensor_descriptor(M / ScaleGranularityM,
                                                K / ScaleGranularityK,
                                                scale_stride_A,
                                                ck_tile::bool_constant<a_row_major>{}));
            ck_tile::HostTensor<ScaleType> scale_b(
                ck_tile::host_tensor_descriptor(K / ScaleGranularityK,
                                                N / ScaleGranularityN,
                                                scale_stride_B,
                                                ck_tile::bool_constant<b_row_major>{}));

            // Initialize via proxy types
            ck_tile::HostTensor<AInitType> a_init(ck_tile::host_tensor_descriptor(
                M, K, stride_A, ck_tile::bool_constant<a_row_major>{}));
            ck_tile::HostTensor<BInitType> b_init(ck_tile::host_tensor_descriptor(
                K, N, stride_B, ck_tile::bool_constant<b_row_major>{}));

            if(init_method == 0)
            {
                // Random tensor and scale values. Give each group a distinct seed
                // so same-dim groups are not bitwise-identical.
                const uint32_t base_seed = 11939;
                const uint32_t seed      = base_seed + static_cast<uint32_t>(i);
                ck_tile::FillUniformDistribution<>{0.0f, 1.0f, seed}(a_init);
                ck_tile::FillUniformDistribution<>{-2.f, 2.f, seed}(scale_a);
                ck_tile::FillUniformDistribution<>{-.5f, .5f, seed}(b_init);
                ck_tile::FillUniformDistribution<>{-2.f, 2.f, seed}(scale_b);
            }
            else if(init_method == 1)
            {
                // Constant tensor and scale values
                ck_tile::FillUniformDistribution<>{2.f, 2.f}(a_init);
                ck_tile::FillUniformDistribution<>{0.5f, 0.5f}(scale_a);
                ck_tile::FillUniformDistribution<>{0.5f, 0.5f}(b_init);
                ck_tile::FillUniformDistribution<>{2.f, 2.f}(scale_b);
            }
            else
            {
                FAIL() << "Unexpected init_method: " << init_method;
            }

            copy_proxy_to_real(a_host, a_init);
            copy_proxy_to_real(b_origin_host, b_init);

            // Pre-shuffle B and scales
            auto b_shuffled       = preShuffleWeight<MXFlatmmArchTraits::GetNLane()>(b_origin_host);
            auto scale_a_shuffled = MXFlatmmArchTraits::template preShuffleScale<true>(scale_a);
            auto scale_b_shuffled = MXFlatmmArchTraits::template preShuffleScale<false>(scale_b);

            // Allocate device buffers
            auto a_dev =
                std::make_unique<ck_tile::DeviceMem>(a_host.get_element_space_size_in_bytes());
            auto b_dev =
                std::make_unique<ck_tile::DeviceMem>(b_shuffled.get_element_space_size_in_bytes());
            auto c_dev =
                std::make_unique<ck_tile::DeviceMem>(c_host.get_element_space_size_in_bytes());
            auto sa_dev = std::make_unique<ck_tile::DeviceMem>(
                scale_a_shuffled.get_element_space_size_in_bytes());
            auto sb_dev = std::make_unique<ck_tile::DeviceMem>(
                scale_b_shuffled.get_element_space_size_in_bytes());

            a_dev->ToDevice(a_host.data());
            b_dev->ToDevice(b_shuffled.data());
            c_host.SetZero();
            c_dev->ToDevice(c_host.data());
            sa_dev->ToDevice(scale_a_shuffled.data());
            sb_dev->ToDevice(scale_b_shuffled.data());

            // Fill host args arrays
            h_Ms[i]        = M;
            h_Ns[i]        = N;
            h_Ks[i]        = K;
            h_a_ptrs[i]    = a_dev->GetDeviceBuffer();
            h_b_ptrs[i]    = b_dev->GetDeviceBuffer();
            h_c_ptrs[i]    = c_dev->GetDeviceBuffer();
            h_stride_As[i] = stride_A;
            h_stride_Bs[i] = stride_B;
            h_stride_Cs[i] = stride_C;
            h_scale_as[i] =
                ScaleA{static_cast<ScaleType*>(sa_dev->GetDeviceBuffer()), M / ScaleGranularityM};
            h_scale_bs[i] =
                ScaleB{static_cast<ScaleType*>(sb_dev->GetDeviceBuffer()), N / ScaleGranularityN};

            // Store for later validation
            a_tensors.push_back(std::move(a_host));
            b_origin_tensors.push_back(std::move(b_origin_host));
            c_tensors.push_back(std::move(c_host));
            scale_a_tensors.push_back(std::move(scale_a));
            scale_b_tensors.push_back(std::move(scale_b));
            b_shuffled_tensors.push_back(std::move(b_shuffled));
            scale_a_shuffled_tensors.push_back(std::move(scale_a_shuffled));
            scale_b_shuffled_tensors.push_back(std::move(scale_b_shuffled));
            a_dev_bufs.push_back(std::move(a_dev));
            b_dev_bufs.push_back(std::move(b_dev));
            c_dev_bufs.push_back(std::move(c_dev));
            scale_a_dev_bufs.push_back(std::move(sa_dev));
            scale_b_dev_bufs.push_back(std::move(sb_dev));
        }

        // Copy each host vector to a device buffer and hand the kernel the device pointers.
        auto to_device = [](const auto& host_vec) {
            using ElemType = std::remove_cvref_t<decltype(host_vec[0])>;
            auto dev = std::make_unique<ck_tile::DeviceMem>(host_vec.size() * sizeof(ElemType));
            dev->ToDevice(host_vec.data());
            return dev;
        };

        auto d_Ms        = to_device(h_Ms);
        auto d_Ns        = to_device(h_Ns);
        auto d_Ks        = to_device(h_Ks);
        auto d_a_ptrs    = to_device(h_a_ptrs);
        auto d_b_ptrs    = to_device(h_b_ptrs);
        auto d_c_ptrs    = to_device(h_c_ptrs);
        auto d_stride_As = to_device(h_stride_As);
        auto d_stride_Bs = to_device(h_stride_Bs);
        auto d_stride_Cs = to_device(h_stride_Cs);
        auto d_scale_as  = to_device(h_scale_as);
        auto d_scale_bs  = to_device(h_scale_bs);

        // Build grouped host args from the device-resident metadata arrays
        ck_tile::GroupedFlatmmHostArgs<ScaleA, ScaleB, NumDTensor> host_args{
            static_cast<ck_tile::index_t>(group_count),
            static_cast<ck_tile::index_t*>(d_Ms->GetDeviceBuffer()),
            static_cast<ck_tile::index_t*>(d_Ns->GetDeviceBuffer()),
            static_cast<ck_tile::index_t*>(d_Ks->GetDeviceBuffer()),
            static_cast<const void**>(d_a_ptrs->GetDeviceBuffer()),
            static_cast<ck_tile::index_t*>(d_stride_As->GetDeviceBuffer()),
            static_cast<const void**>(d_b_ptrs->GetDeviceBuffer()),
            static_cast<ck_tile::index_t*>(d_stride_Bs->GetDeviceBuffer()),
            {},
            {},
            static_cast<void**>(d_c_ptrs->GetDeviceBuffer()),
            static_cast<ck_tile::index_t*>(d_stride_Cs->GetDeviceBuffer()),
            1,
            static_cast<ScaleA*>(d_scale_as->GetDeviceBuffer()),
            static_cast<ScaleB*>(d_scale_bs->GetDeviceBuffer())};

        // --- Instantiate and launch the GroupedMXFlatmmKernel ---
        //
        using FlatmmShape = ck_tile::TileGemmShape<
            ck_tile::sequence<FlatmmConfig::M_Tile, FlatmmConfig::N_Tile, FlatmmConfig::K_Tile>,
            ck_tile::sequence<FlatmmConfig::M_Warp, FlatmmConfig::N_Warp, FlatmmConfig::K_Warp>,
            ck_tile::sequence<FlatmmConfig::M_Warp_Tile,
                              FlatmmConfig::N_Warp_Tile,
                              FlatmmConfig::K_Warp_Tile>>;

        using TilePartitioner =
            ck_tile::GemmSpatiallyLocalTilePartitioner<FlatmmShape,
                                                       FlatmmConfig::TileParitionerGroupNum,
                                                       FlatmmConfig::TileParitionerM01>;

        using GemmTraits = ck_tile::TileGemmUniversalTraits<FlatmmConfig::kPadM,
                                                            FlatmmConfig::kPadN,
                                                            FlatmmConfig::kPadK,
                                                            FlatmmConfig::DoubleSmemBuffer,
                                                            ALayout,
                                                            BLayout,
                                                            CLayout,
                                                            FlatmmConfig::TransposeC,
                                                            FlatmmConfig::UseStructuredSparsity,
                                                            /*Persistent=*/false,
                                                            FlatmmConfig::NumWaveGroups,
                                                            /*UseAsyncCopy=*/true>;

        // Both MX FLATMM pipelines dispatch (HasHotLoop, TailNum) at runtime
        // based on num_loop, so the values baked into the pipeline problem below
        // are not load-bearing (kept at library defaults). Groups within a
        // single call may use any mix of K values.
        using MXPipelineProblem =
            ck_tile::MXFlatmmPipelineProblem<ADataType,
                                             BDataType,
                                             AccDataType,
                                             FlatmmShape,
                                             GemmTraits,
                                             ck_tile::GemmPipelineScheduler::Default,
                                             /*HasHotLoop=*/true,
                                             ck_tile::TailNumber::Full>;

        using MXFlatmmPipeline =
            typename MXFlatmmArchTraits::template MXFlatmmPipeline<MXPipelineProblem>;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
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
                                             FlatmmConfig::M_Warp,
                                             FlatmmConfig::N_Warp,
                                             FlatmmConfig::M_Warp_Tile,
                                             FlatmmConfig::N_Warp_Tile,
                                             FlatmmConfig::K_Warp_Tile,
                                             FlatmmConfig::TransposeC,
                                             FlatmmConfig::NumWaveGroups,
                                             false,
                                             1,
                                             MXFlatmmArchTraits::BlockedXDLN_PerWarp,
                                             FlatmmConfig::DoubleSmemBuffer>>;

        using Kernel =
            ck_tile::GroupedMXFlatmmKernel<TilePartitioner, MXFlatmmPipeline, GemmEpilogue>;

        auto kernel_args  = Kernel::MakeKernelArgs(host_args);
        const dim3 grids  = Kernel::GridSize(host_args);
        const dim3 blocks = Kernel::BlockSize();

        std::cout << "Launching kernel: " << Kernel::GetName() << " grid: {" << grids.x << ", "
                  << grids.y << ", " << grids.z << "}, blocks: {" << blocks.x
                  << "}, group_count: " << group_count << std::endl;

        auto s          = ck_tile::stream_config{nullptr, false, 0, 0, 1};
        ck_tile::ignore = ck_tile::launch_kernel(s,
                                                 ck_tile::make_kernel<FlatmmConfig::kBlockPerCu>(
                                                     Kernel{}, grids, blocks, 0, kernel_args));

        // Copy results back and validate per group
        bool pass = true;
        for(int i = 0; i < group_count; ++i)
        {
            c_dev_bufs[i]->FromDevice(c_tensors[i].data());

            ck_tile::HostTensor<CDataType> c_ref(ck_tile::host_tensor_descriptor(
                Ms[i],
                Ns[i],
                ck_tile::get_default_stride(Ms[i], Ns[i], 0, ck_tile::bool_constant<c_row_major>{}),
                ck_tile::bool_constant<c_row_major>{}));
            c_ref.SetZero();

            ck_tile::reference_mx_gemm<ADataType,
                                       BDataType,
                                       ScaleType,
                                       ScaleType,
                                       AccDataType,
                                       CDataType>(
                a_tensors[i], b_origin_tensors[i], c_ref, scale_a_tensors[i], scale_b_tensors[i]);

            // Constant init (init_method==1) produces an exact integer K result;
            // use near-exact tolerance so a dropped/double-counted K-tile cannot
            // hide inside the K-scaled relative slack. Random init keeps 1e-2.
            const float rtol = (init_method == 1) ? 0.f : 1e-2f;
            const float atol = (init_method == 1) ? 1.f : 1e-2f;
            bool group_pass =
                ck_tile::check_err(c_tensors[i],
                                   c_ref,
                                   std::string("Group ") + std::to_string(i) + " result mismatch",
                                   rtol,
                                   atol);
            pass &= group_pass;
        }
        EXPECT_TRUE(pass);
    }
};
