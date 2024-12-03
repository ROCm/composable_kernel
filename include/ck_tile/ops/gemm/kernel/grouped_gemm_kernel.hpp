// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <string>

#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/literals.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/host.hpp"

namespace ck_tile {

struct GroupedGemmDesc
{
    ck_tile::index_t M;
    ck_tile::index_t N;
    ck_tile::index_t K;
    ck_tile::index_t stride_A;
    ck_tile::index_t stride_B;
    ck_tile::index_t stride_C;
};

template <typename TilePartitioner_, typename GemmPipeline_, typename EpiloguePipeline_>
struct GroupedGemmKernel
{
    using TilePartitioner                    = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline                       = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline                   = remove_cvref_t<EpiloguePipeline_>;
    using ALayout                            = remove_cvref_t<typename GemmPipeline::ALayout>;
    using BLayout                            = remove_cvref_t<typename GemmPipeline::BLayout>;
    using CLayout                            = remove_cvref_t<typename GemmPipeline::CLayout>;
    static constexpr index_t KernelBlockSize = GemmPipeline::BlockSize;

    using ADataType = remove_cvref_t<typename GemmPipeline::ADataType>;
    using BDataType = remove_cvref_t<typename GemmPipeline::BDataType>;
    using CDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    __host__ static constexpr auto BlockSize() { return dim3(KernelBlockSize); }

    struct GroupedKernelArgument
    {
        const void* a_ptr;
        const void* b_ptr;
        void* c_ptr;
        ck_tile::index_t M;
        ck_tile::index_t N;
        ck_tile::index_t K;
        ck_tile::index_t stride_A;
        ck_tile::index_t stride_B;
        ck_tile::index_t stride_C;
    };

    struct GemmTransKernelArg
    {
        GroupedKernelArgument group_karg;
        ck_tile::index_t block_start;
        ck_tile::index_t block_end;
        ck_tile::index_t group_count;
        ck_tile::index_t block_n0_size;

        GemmTransKernelArg() = default;
        GemmTransKernelArg(GroupedKernelArgument&& karg,
                           index_t bl_start,
                           index_t bl_end,
                           index_t gcount,
                           index_t n0_size)
            : group_karg{karg},
              block_start{bl_start},
              block_end{bl_end},
              group_count{gcount},
              block_n0_size{n0_size}
        {
        }
    };

    struct Argument
    {
        Argument(std::vector<const void*>& p_As,
                 std::vector<const void*>& p_Bs,
                 std::vector<void*>& p_Es,
                 const std::vector<GroupedGemmDesc>& gemm_descs)
        {
            index_t group_count_ = ck_tile::type_convert<ck_tile::index_t>(gemm_descs.size());
            grid_size_           = 0;
            if(!(group_count_ == ck_tile::type_convert<ck_tile::index_t>(p_As.size()) &&
                 group_count_ == ck_tile::type_convert<ck_tile::index_t>(p_Bs.size()) &&
                 group_count_ == ck_tile::type_convert<ck_tile::index_t>(p_Es.size())))
            {
                throw std::runtime_error("wrong! group_count_ != p_As/b/c.size");
            }
            gemm_kernel_args_.reserve(group_count_);

            for(std::size_t i = 0; i < gemm_descs.size(); ++i)
            {
                const index_t M = gemm_descs[i].M;
                const index_t N = gemm_descs[i].N;
                const index_t K = gemm_descs[i].K;

                if(M == 0)
                {
                    continue;
                }

                const index_t stride_a = gemm_descs[i].stride_A;
                const index_t stride_b = gemm_descs[i].stride_B;
                const index_t stride_c = gemm_descs[i].stride_C;

                const index_t MPerBlock = TilePartitioner::MPerBlock;
                const index_t NPerBlock = TilePartitioner::NPerBlock;

                const auto M0 = ck_tile::integer_divide_ceil(M, MPerBlock);
                const auto N0 = ck_tile::integer_divide_ceil(N, NPerBlock);

                const index_t grid_size_one = M0 * N0 * 1;

                const index_t block_start = grid_size_;
                const index_t block_end   = grid_size_ + grid_size_one;

                grid_size_ += grid_size_one;

                auto karg = GroupedKernelArgument{type_convert<const ADataType*>(p_As[i]),
                                                  type_convert<const BDataType*>(p_Bs[i]),
                                                  type_convert<CDataType*>(p_Es[i]),
                                                  M,
                                                  N,
                                                  K,
                                                  stride_a,
                                                  stride_b,
                                                  stride_c};

                gemm_kernel_args_.emplace_back(
                    std::move(karg), block_start, block_end, group_count_, N0);
            }
        }

        std::vector<GemmTransKernelArg> gemm_kernel_args_;
        index_t grid_size_;
        void* p_workspace_ = nullptr;
    };

    __host__ static constexpr auto GridSize(const Argument& args)
    {
        return dim3(args.grid_size_, 1, 1);
    }

    CK_TILE_HOST static auto MakeKargs(std::vector<const void*>& p_As,
                                       std::vector<const void*>& p_Bs,
                                       std::vector<void*>& p_Es,
                                       const std::vector<GroupedGemmDesc>& gemm_descs)
    {
        return Argument{p_As, p_Bs, p_Es, gemm_descs};
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_DEVICE void Run(const GroupedKernelArgument& kargs,
                            const index_t block_start,
                            const index_t block_n0_size) const
    {
        const auto [i_m, i_n] = TilePartitioner{}(block_start, block_n0_size);
        // options
        const ADataType* a_start = static_cast<const ADataType*>(kargs.a_ptr);
        const BDataType* b_start = static_cast<const BDataType*>(kargs.b_ptr);
        // Convert pointers to tensor views
        auto a_tensor_view = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_start,
                    make_tuple(kargs.M, kargs.K),
                    make_tuple(kargs.stride_A, 1),
                    number<GemmPipeline::VectorSizeA>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_start,
                    make_tuple(kargs.M, kargs.K),
                    make_tuple(1, kargs.stride_A),
                    number<1>{},
                    number<1>{});
            }
        }();

        auto b_tensor_view = [&]() {
            if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    b_start,
                    make_tuple(kargs.N, kargs.K),
                    make_tuple(1, kargs.stride_B),
                    number<1>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    b_start,
                    make_tuple(kargs.N, kargs.K),
                    make_tuple(kargs.stride_B, 1),
                    number<GemmPipeline::VectorSizeB>{},
                    number<1>{});
            }
        }();

        auto a_pad_view = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(a_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::KPerBlock>{}),
                                       sequence<false, GemmPipeline::kPadK>{});
            }
            else
            {
                return pad_tensor_view(a_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::KPerBlock>{}),
                                       sequence<GemmPipeline::kPadM, false>{});
            }
        }();
        // clang-format on

        auto a_block_window = make_tile_window(
            a_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::KPerBlock>{}),
            {i_m, 0});

        auto b_pad_view = [&]() {
            if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::ColumnMajor>)
            {
                return pad_tensor_view(b_tensor_view,
                                       make_tuple(number<TilePartitioner::NPerBlock>{},
                                                  number<TilePartitioner::KPerBlock>{}),
                                       sequence<false, GemmPipeline::kPadK>{});
            }
            else
            {
                return pad_tensor_view(b_tensor_view,
                                       make_tuple(number<TilePartitioner::NPerBlock>{},
                                                  number<TilePartitioner::KPerBlock>{}),
                                       sequence<GemmPipeline::kPadN, false>{});
            }
        }();

        auto b_block_window = make_tile_window(
            b_pad_view,
            make_tuple(number<TilePartitioner::NPerBlock>{}, number<TilePartitioner::KPerBlock>{}),
            {i_n, 0});

        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        const index_t num_loop = TilePartitioner::GetLoopNum(kargs.K);

        // Run GEMM cooperatively by whole wokrgroup.
        auto c_block_tile =
            GemmPipeline{}.template operator()(a_block_window, b_block_window, num_loop, smem_ptr);

        CDataType* c_start = static_cast<CDataType*>(kargs.c_ptr);
        auto c_tensor_view = [&]() {
            if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    c_start,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(kargs.stride_C, 1),
                    number<GemmPipeline::VectorSizeC>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    c_start,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(1, kargs.stride_C),
                    number<1>{},
                    number<1>{});
            }
        }();

        auto c_pad_view = [&]() {
            if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(c_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<false, GemmPipeline::kPadN>{});
            }
            else
            {
                return pad_tensor_view(c_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<GemmPipeline::kPadM, false>{});
            }
        }();
        auto CBlockWindow_pad = make_tile_window(
            c_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {i_m, i_n});

        EpiloguePipeline{}(CBlockWindow_pad, c_block_tile);
    }

    CK_TILE_DEVICE void operator()(const Argument& kargs) const
    {
        const index_t block_id   = ck_tile::get_block_1d_id();
        const auto gemm_desc_ptr = reinterpret_cast<const GemmTransKernelArg*>(kargs.p_workspace_);

        index_t left     = 0;
        index_t right    = gemm_desc_ptr->group_count;
        index_t group_id = index_t((left + right) / 2);

        while((!(block_id >= gemm_desc_ptr[group_id].block_start &&
                 block_id < gemm_desc_ptr[group_id].block_end)) &&
              left <= right)
        {
            if(block_id < gemm_desc_ptr[group_id].block_start)
            {
                right = group_id;
            }
            else
            {
                left = group_id;
            }
            group_id = index_t((left + right) / 2);
        }

        Run(gemm_desc_ptr[group_id].group_karg,
            gemm_desc_ptr[group_id].block_start,
            gemm_desc_ptr[group_id].block_n0_size);
    }

    static void SetWorkSpacePointer(Argument* p_arg, void* p_workspace)
    {
        p_arg->p_workspace_ = p_workspace;
    }

    static void SetDeviceKernelArgs(Argument* p_arg, void* p_dev_kernel_args)
    {
        return SetWorkSpacePointer(p_arg, p_dev_kernel_args);
    }

    static size_t GetDeviceKernelArgSize(const Argument* p_arg) { return GetWorkSpaceSize(p_arg); }

    static size_t GetWorkSpaceSize(const Argument* p_arg)
    {
        if(p_arg)
        {
            return p_arg->gemm_kernel_args_.size() * sizeof(GemmTransKernelArg);
        }
        else
            throw std::runtime_error("The argument pointer is not an object of "
                                     "GroupedGemmKernel::Argument structure!");
    }
};

} // namespace ck_tile
