// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <sstream>
#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp"

template <typename Tuple>
class TestCkTileGroupedGemm : public ::testing::Test
{
    protected:
    using ALayout     = std::tuple_element_t<0, Tuple>;
    using BLayout     = std::tuple_element_t<1, Tuple>;
    using CLayout     = std::tuple_element_t<2, Tuple>;
    using ADataType   = std::tuple_element_t<3, Tuple>;
    using BDataType   = std::tuple_element_t<4, Tuple>;
    using AccDataType = std::tuple_element_t<5, Tuple>;
    using CDataType   = std::tuple_element_t<6, Tuple>;

    struct batched_gemm_kargs : public ck_tile::BatchedGemmHostArgs
    {
    };

    template <typename ALayout, typename BLayout, typename CLayout>
    void invoke_grouped_gemm(std::vector<const void*>& a_m_k_dev_buf,
                             std::vector<const void*>& b_k_n_dev_buf,
                             std::vector<void*>& c_m_n_dev_buf,
                             const std::vector<ck_tile::GroupedGemmDesc>& gemm_descs,
                             const ck_tile::stream_config& s)
    {
        constexpr bool kPadM        = false;
        constexpr bool kPadN        = false;
        constexpr bool kPadK        = false;
        constexpr bool kTilePermute = false;

        constexpr ck_tile::index_t kOutputRank = 2;

        constexpr int kBlockPerCu = 1;

        // This part comes from the Codegen
        constexpr ck_tile::index_t M_Tile = 128;
        constexpr ck_tile::index_t N_Tile = 128;
        constexpr ck_tile::index_t K_Tile = 32;

        constexpr ck_tile::index_t M_Warp = 2;
        constexpr ck_tile::index_t N_Warp = 2;
        constexpr ck_tile::index_t K_Warp = 1;

        constexpr ck_tile::index_t M_Warp_Tile = 32;
        constexpr ck_tile::index_t N_Warp_Tile = 32;
        constexpr ck_tile::index_t K_Warp_Tile = 8;

        constexpr bool CShuffleEpilogue =
            std::is_same_v<CLayout, ck_tile::tensor_layout::gemm::ColumnMajor>;

        using CodegenGemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                   ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                                   ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

        using TilePartitioner = ck_tile::GemmTile1DPartitioner<CodegenGemmShape>;

        using GemmEpilogue = std::conditional_t<
            CShuffleEpilogue,
            ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<AccDataType,
                                                                       CDataType,
                                                                       kPadM,
                                                                       kPadN,
                                                                       kTilePermute,
                                                                       kOutputRank,
                                                                       1,
                                                                       0,
                                                                       TilePartitioner::MPerBlock,
                                                                       TilePartitioner::NPerBlock>>,
            ck_tile::Default2DEpilogue<
                ck_tile::Default2DEpilogueProblem<AccDataType, CDataType, kPadM, kPadN>>>;

        using CodegenGemmTraits =
            ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout>;

        using CodegenPipelineProblem = ck_tile::GemmPipelineProblem<ADataType,
                                                                    BDataType,
                                                                    AccDataType,
                                                                    CodegenGemmShape,
                                                                    CodegenGemmTraits>;

        using CodegenGemmPolicy = ck_tile::UniversalGemmPipelineAgBgCrPolicy;
        using CodegenGemmPipeline =
            ck_tile::GemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem, CodegenGemmPolicy>;

        using Kernel =
            ck_tile::GroupedGemmKernel<TilePartitioner, CodegenGemmPipeline, GemmEpilogue>;

        auto arguments = Kernel::MakeKargs(a_m_k_dev_buf, b_k_n_dev_buf, c_m_n_dev_buf, gemm_descs);

        std::size_t workspace_size = Kernel::GetWorkSpaceSize(&arguments);
        std::size_t kargs_size     = Kernel::GetDeviceKernelArgSize(&arguments);

        ck_tile::DeviceMem gemm_workspace, gemm_kargs;

        if(kargs_size > 0)
        {
            gemm_kargs.Realloc(kargs_size);
            Kernel::SetDeviceKernelArgs(&arguments, gemm_kargs.GetDeviceBuffer());
        }
        if(workspace_size > 0 && workspace_size != kargs_size)
        {
            gemm_workspace.Realloc(workspace_size);
            Kernel::SetWorkSpacePointer(&arguments, gemm_workspace.GetDeviceBuffer());
        }

        const dim3 grids      = Kernel::GridSize(arguments);
        constexpr dim3 blocks = Kernel::BlockSize();

        ck_tile::hip_check_error(hipMemcpyWithStream(
            arguments.p_workspace_,
            arguments.gemm_kernel_args_.data(),
            arguments.gemm_kernel_args_.size() * sizeof(typename Kernel::GemmTransKernelArg),
            hipMemcpyHostToDevice,
            s.stream_id_));

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args:"
                      << " grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
        }

        ck_tile::launch_kernel(
            s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, arguments));
    }

    public:
    void Run(const std::vector<int>& Ms,
             const std::vector<int>& Ns,
             const std::vector<int>& Ks,
             std::vector<int>& stride_As,
             std::vector<int>& stride_Bs,
             std::vector<int>& stride_Cs,
             const int group_count = 16)
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

        std::vector<ck_tile::HostTensor<ADataType>> a_m_k_tensors;
        std::vector<ck_tile::HostTensor<BDataType>> b_k_n_tensors;
        std::vector<ck_tile::HostTensor<CDataType>> c_m_n_tensors;

        a_m_k_tensors.reserve(group_count);
        b_k_n_tensors.reserve(group_count);
        c_m_n_tensors.reserve(group_count);

        std::vector<ck_tile::GroupedGemmDesc> gemm_descs;
        gemm_descs.reserve(group_count);

        for(int i = 0; i < group_count; ++i)
        {
            const ck_tile::index_t M = Ms[i];
            const ck_tile::index_t N = Ns[i];
            const ck_tile::index_t K = Ks[i];

            stride_As[i] = f_get_default_stride(M, N, stride_As[i], ALayout{});
            stride_Bs[i] = f_get_default_stride(K, N, stride_Bs[i], BLayout{});
            stride_Cs[i] = f_get_default_stride(M, N, stride_Cs[i], CLayout{});

            a_m_k_tensors.push_back(ck_tile::HostTensor<ADataType>(
                f_host_tensor_descriptor(M, K, stride_As[i], ALayout{})));
            b_k_n_tensors.push_back(ck_tile::HostTensor<BDataType>(
                f_host_tensor_descriptor(K, N, stride_Bs[i], BLayout{})));
            c_m_n_tensors.push_back(ck_tile::HostTensor<CDataType>(
                f_host_tensor_descriptor(M, N, stride_Cs[i], CLayout{})));

            std::cout << "gemm[" << i << "]"
                      << " a_m_k: " << a_m_k_tensors[i].mDesc
                      << " b_k_n: " << b_k_n_tensors[i].mDesc
                      << " c_m_n: " << c_m_n_tensors[i].mDesc << std::endl;

            ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(a_m_k_tensors[i]);
            ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(b_k_n_tensors[i]);

            gemm_descs.push_back({M, N, K, stride_As[i], stride_Bs[i], stride_Cs[i]});
        }

        std::vector<std::unique_ptr<ck_tile::DeviceMem>> a_m_k_dev_buf;
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> b_k_n_dev_buf;
        std::vector<std::unique_ptr<ck_tile::DeviceMem>> c_m_n_dev_buf;

        a_m_k_dev_buf.reserve(group_count);
        b_k_n_dev_buf.reserve(group_count);
        c_m_n_dev_buf.reserve(group_count);

        std::vector<const void*> p_a, p_b;
        std::vector<void*> p_c;

        for(int i = 0; i < group_count; ++i)
        {
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

            p_a.push_back(a_m_k_dev_buf[i]->GetDeviceBuffer());
            p_b.push_back(b_k_n_dev_buf[i]->GetDeviceBuffer());
            p_c.push_back(c_m_n_dev_buf[i]->GetDeviceBuffer());
        }

        invoke_grouped_gemm<ALayout, BLayout, CLayout>(
            p_a, p_b, p_c, gemm_descs, ck_tile::stream_config{nullptr, false});

        for(int i = 0; i < group_count; i++)
        {
            c_m_n_dev_buf[i]->FromDevice(c_m_n_tensors[i].data());
        }
        bool pass = true;
        for(int i = 0; i < group_count; ++i)
        {
            ck_tile::HostTensor<CDataType> c_m_n_host_ref(
                f_host_tensor_descriptor(Ms[i], Ns[i], stride_Cs[i], CLayout{}));
            c_m_n_host_ref.SetZero();
            ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
                a_m_k_tensors[i], b_k_n_tensors[i], c_m_n_host_ref);
            pass &= ck_tile::check_err(c_m_n_tensors[i], c_m_n_host_ref);
        }
        EXPECT_TRUE(pass);
    }
};
