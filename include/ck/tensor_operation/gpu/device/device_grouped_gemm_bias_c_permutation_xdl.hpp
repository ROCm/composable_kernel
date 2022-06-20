#pragma once

#include <iostream>
#include <sstream>
#include "device.hpp"
#include "device_base.hpp"
#include "device_grouped_gemm_bias_c_permutation.hpp"
#include "common_header.hpp"
#include "tensor_layout.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm_multiple_d_xdl_cshuffle.hpp"
#include "gemm_specialization.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseGemm,
          typename GemmDesc,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_grouped_gemm_bias_transpose_xdl(
            const void CK_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
            const index_t group_count,
            const AElementwiseOperation a_element_op,
            const BElementwiseOperation b_element_op,
            const CElementwiseOperation c_element_op)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__))
    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    const index_t block_id = get_block_1d_id();

    const auto gemm_desc_ptr =
        reinterpret_cast<const GemmDesc*>(cast_pointer_to_generic_address_space(gemm_descs_const));

    index_t group_id = 0;
    for(index_t i = 0; i < group_count; i++)
    {
        group_id =
            (block_id >= gemm_desc_ptr[i].BlockStart_ && block_id < gemm_desc_ptr[i].BlockEnd_)
                ? i
                : group_id;
    }

    GridwiseGemm::template Run<HasMainKBlockLoop>(
        gemm_desc_ptr[group_id].a_ptr_,
        gemm_desc_ptr[group_id].b_ptr_,
        gemm_desc_ptr[group_id].d_ptr_,
        gemm_desc_ptr[group_id].e_ptr_,
        p_shared,
        a_element_op,
        b_element_op,
        c_element_op,
        gemm_desc_ptr[group_id].a_grid_desc_k0_m_k1_,
        gemm_desc_ptr[group_id].b_grid_desc_k0_n_k1_,
        gemm_desc_ptr[group_id].ds_grid_desc_mblock_mperblock_nblock_nperblock_,
        gemm_desc_ptr[group_id].e_grid_desc_mblock_mperblock_nblock_nperblock_,
        gemm_desc_ptr[group_id].block_2_ctile_map_);
#else
    ignore = gemm_descs_const;
    ignore = group_count;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = c_element_op;
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
}

template <
          typename ALayout,
          typename BLayout,
typename ADataType,
          typename BDataType,
          typename DDataType,
          typename EDataType,
          typename AccDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          GemmSpecialization GemmSpec,
          ck::index_t NumPrefetch,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          ck::index_t AK1,
          ck::index_t BK1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_K1,
          bool ABlockLdsAddExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_K1,
          bool BBlockLdsAddExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEBlockTransferScalarPerVector_NPerBlock,
          LoopScheduler LoopSched = make_default_loop_scheduler()>
struct DeviceGroupedGemmBiasCPermuteXdl
    : public DeviceGroupedGemmBiasCPermute<AElementwiseOperation,
                                           BElementwiseOperation,
                                           CElementwiseOperation>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static auto MakeAGridDescriptor_K0_M_K1(index_t M, index_t K, index_t StrideA)
    {
        assert(K % BK1 == 0);

        const index_t K0 = K / AK1;

        const auto a_grid_desc_m_k = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(StrideA, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(I1, StrideA));
            }
        }();

        if constexpr(GemmSpec == GemmSpecialization::MNPadding)
        {
            const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;

            return transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(K0, AK1)),
                           make_right_pad_transform(M, PadM)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(K0, AK1)),
                           make_pass_through_transform(M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
    }

    static auto MakeBGridDescriptor_K0_N_K1(index_t K, index_t N, index_t StrideB)
    {
        assert(K % BK1 == 0);

        const index_t K0 = K / BK1;

        const auto b_grid_desc_k_n = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(K, N), make_tuple(StrideB, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(K, N), make_tuple(I1, StrideB));
            }
        }();

        if constexpr(GemmSpec == GemmSpecialization::MNPadding)
        {
            const auto PadN = (NPerBlock - N % NPerBlock) % NPerBlock;

            return transform_tensor_descriptor(
                b_grid_desc_k_n,
                make_tuple(make_unmerge_transform(make_tuple(K0, BK1)),
                           make_right_pad_transform(N, PadN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                b_grid_desc_k_n,
                make_tuple(make_unmerge_transform(make_tuple(K0, BK1)),
                           make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
    }

    static auto MakeCGridDescriptor_M_N(index_t M0,
                                        index_t M1,
                                        index_t N0,
                                        index_t N1,
                                        index_t StrideM0,
                                        index_t StrideM1,
                                        index_t StrideN0,
                                        index_t StrideN1)
    {
        const auto c_grid_desc_m_n = [&]() {
            const auto c_grid_desc_m0_m1_n0_n1 = make_naive_tensor_descriptor(
                make_tuple(M0, M1, N0, N1), make_tuple(StrideM0, StrideM1, StrideN0, StrideN1));

            return transform_tensor_descriptor(c_grid_desc_m0_m1_n0_n1,
                                               make_tuple(make_merge_transform(make_tuple(M0, M1)),
                                                          make_merge_transform(make_tuple(N0, N1))),
                                               make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}));
        }();

        const index_t M = M0 * M1;
        const index_t N = N0 * N1;

        if constexpr(GemmSpec == GemmSpecialization::MNPadding)
        {
            const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;
            const auto PadN = (NPerBlock - N % NPerBlock) % NPerBlock;

            return transform_tensor_descriptor(
                c_grid_desc_m_n,
                make_tuple(make_right_pad_transform(M, PadM), make_right_pad_transform(N, PadN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
        else
        {

            return transform_tensor_descriptor(
                c_grid_desc_m_n,
                make_tuple(make_pass_through_transform(M), make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
    }

    using AGridDesc_K0_M_K1 = decltype(MakeAGridDescriptor_K0_M_K1(1, 1, 1));
    using BGridDesc_K0_N_K1 = decltype(MakeBGridDescriptor_K0_N_K1(1, 1, 1));
    using EGridDesc_M_N     = decltype(MakeCGridDescriptor_M_N(1, 1, 1, 1, 1, 1, 1, 1));

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemmMultipleD_k0mk1_k0nk1_mn_xdl_cshuffle<
        ADataType, // TODO: distinguish A/B datatype
        AccDataType,
        EDataType,            // CShuffleDataType,
        ck::Tuple<EDataType>, // DsDataType,
        EDataType,            // EDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        InMemoryDataOperationEnum::Set,
        AGridDesc_K0_M_K1,
        BGridDesc_K0_N_K1,
        EGridDesc_M_N,
        NumPrefetch, // NumGemmKPrefetchStage
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsAddExtraM,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false, // BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsAddExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEBlockTransferScalarPerVector_NPerBlock,
        LoopSched>;

    struct GroupedGemmBlock2ETileMap
    {
        using UnderlyingBlock2CTileMap = typename GridwiseGemm::DefaultBlock2ETileMap;
        static_assert(
            std::is_same<decltype(GridwiseGemm::MakeDefaultBlock2ETileMap(EGridDesc_M_N{})),
                         typename GridwiseGemm::DefaultBlock2ETileMap>::value,
            "Wrong! Should be the same type name");
        GroupedGemmBlock2ETileMap()
        {
            block_2_ctile_map_ = GridwiseGemm::MakeDefaultBlock2ETileMap(EGridDesc_M_N{});
            BlockStart_        = -1;
        }

        GroupedGemmBlock2ETileMap(const EGridDesc_M_N& c_grid_desc_m_n, ck::index_t BlockStart)
        {
            block_2_ctile_map_ = GridwiseGemm::MakeDefaultBlock2ETileMap(c_grid_desc_m_n);
            BlockStart_        = BlockStart;
        }

        template <typename TopIdx>
        __host__ __device__ constexpr auto CalculateBottomIndex(const TopIdx& idx_top) const
        {
            return block_2_ctile_map_.CalculateBottomIndex(
                make_multi_index(idx_top[I0] - BlockStart_));
        }

        template <typename CTileIdx, typename CTileDim>
        __host__ __device__ bool ValidCTileIndex(const CTileIdx& c_tile_idx,
                                                 const CTileDim& c_tile_dim) const
        {
            return block_2_ctile_map_.ValidCTileIndex(c_tile_idx, c_tile_dim);
        }

        __host__ bool CheckValidity(const EGridDesc_M_N& c_grid_desc_m_n) const
        {
            return block_2_ctile_map_.CheckValidity(c_grid_desc_m_n);
        }

        typename GridwiseGemm::DefaultBlock2ETileMap block_2_ctile_map_;
        ck::index_t BlockStart_;
    };

    struct GemmBiasTransKernelArg
    {
        AGridDesc_K0_M_K1 a_grid_desc_k0_m_k1_;
        BGridDesc_K0_N_K1 b_grid_desc_k0_n_k1_;
        EGridDesc_M_N e_grid_desc_m_n_;

        ck::StaticallyIndexedArray<
            typename GridwiseGemm::EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
            1>
            ds_grid_desc_mblock_mperblock_nblock_nperblock_;

        typename GridwiseGemm::EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            e_grid_desc_mblock_mperblock_nblock_nperblock_;

        GroupedGemmBlock2ETileMap block_2_ctile_map_;

        const ADataType* a_ptr_;
        const BDataType* b_ptr_;
        const DDataType* d_ptr_;
        EDataType* e_ptr_;

        ck::index_t BlockStart_, BlockEnd_;
    };

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(std::vector<const void*>& p_As,
                 std::vector<const void*>& p_Bs,
                 std::vector<const void*>& p_Ds,
                 std::vector<void*>& p_Es,
                 std::vector<GemmBiasCPermuteDesc>& gemm_bias_transpose_desc,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
            : a_element_op_{a_element_op}, b_element_op_{b_element_op}, c_element_op_{c_element_op}
        {
            grid_size_ = 0;

            gemm_descs_args_workspace_ = nullptr;

            group_count_ = ck::type_convert<ck::index_t>(gemm_bias_transpose_desc.size());

            if(!(group_count_ == ck::type_convert<ck::index_t>(p_As.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_Bs.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_Ds.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_Es.size())))
            {
                throw std::runtime_error("wrong! group_count_ != p_As/b/c.size");
            }

            gemm_desc_kernel_arg_.reserve(group_count_);

            for(std::size_t i = 0; i < gemm_bias_transpose_desc.size(); i++)
            {
                const index_t M = gemm_bias_transpose_desc[i].M_;
                const index_t N = gemm_bias_transpose_desc[i].N_;
                const index_t K = gemm_bias_transpose_desc[i].K_;

                const index_t StrideA = gemm_bias_transpose_desc[i].stride_A_;
                const index_t StrideB = gemm_bias_transpose_desc[i].stride_B_;

                if(!(M == gemm_bias_transpose_desc[i].M0_ * gemm_bias_transpose_desc[i].M1_ &&
                     N == gemm_bias_transpose_desc[i].N0_ * gemm_bias_transpose_desc[i].N1_))
                {
                    throw std::runtime_error("wrong! M != M0 * M1 or N != N0 * N1");
                }

                const auto a_grid_desc_k0_m_k1_ =
                    DeviceGroupedGemmBiasCPermuteXdl::MakeAGridDescriptor_K0_M_K1(M, K, StrideA);
                const auto b_grid_desc_k0_n_k1_ =
                    DeviceGroupedGemmBiasCPermuteXdl::MakeBGridDescriptor_K0_N_K1(K, N, StrideB);

                const auto e_grid_desc_m_n_ =
                    DeviceGroupedGemmBiasCPermuteXdl::MakeCGridDescriptor_M_N(
                        gemm_bias_transpose_desc[i].M0_,
                        gemm_bias_transpose_desc[i].M1_,
                        gemm_bias_transpose_desc[i].N0_,
                        gemm_bias_transpose_desc[i].N1_,
                        gemm_bias_transpose_desc[i].stride_E_M0_,
                        gemm_bias_transpose_desc[i].stride_E_M1_,
                        gemm_bias_transpose_desc[i].stride_E_N0_,
                        gemm_bias_transpose_desc[i].stride_E_N1_);

                const auto d_grid_desc_m_n_ =
                    DeviceGroupedGemmBiasCPermuteXdl::MakeCGridDescriptor_M_N(
                        gemm_bias_transpose_desc[i].M0_,
                        gemm_bias_transpose_desc[i].M1_,
                        gemm_bias_transpose_desc[i].N0_,
                        gemm_bias_transpose_desc[i].N1_,
                        gemm_bias_transpose_desc[i].stride_D_M0_,
                        gemm_bias_transpose_desc[i].stride_D_M1_,
                        gemm_bias_transpose_desc[i].stride_D_N0_,
                        gemm_bias_transpose_desc[i].stride_D_N1_);

                const index_t grid_size_grp =
                    GroupedGemmBlock2ETileMap(e_grid_desc_m_n_, 0)
                        .block_2_ctile_map_.CalculateGridSize(e_grid_desc_m_n_);

                const index_t BlockStart = grid_size_;
                const index_t BlockEnd   = grid_size_ + grid_size_grp;

                grid_size_ += grid_size_grp;

                const auto block_2_ctile_map_ =
                    GroupedGemmBlock2ETileMap(e_grid_desc_m_n_, BlockStart);

                if(GridwiseGemm::CheckValidity(a_grid_desc_k0_m_k1_,
                                               b_grid_desc_k0_n_k1_,
                                               e_grid_desc_m_n_,
                                               block_2_ctile_map_))
                {
                    StaticallyIndexedArray<
                        typename GridwiseGemm::EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                        1>
                        ds_grid_desc_mblock_mperblock_nblock_nperblock_;

                    ds_grid_desc_mblock_mperblock_nblock_nperblock_(I0) =
                        GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                            d_grid_desc_m_n_);

                    auto e_grid_desc_mblock_mperblock_nblock_nperblock_ =
                        GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                            e_grid_desc_m_n_);

                    gemm_desc_kernel_arg_.push_back(
                        GemmBiasTransKernelArg{a_grid_desc_k0_m_k1_,
                                               b_grid_desc_k0_n_k1_,
                                               e_grid_desc_m_n_,
                                               ds_grid_desc_mblock_mperblock_nblock_nperblock_,
                                               e_grid_desc_mblock_mperblock_nblock_nperblock_,
                                               block_2_ctile_map_,
                                               static_cast<const ADataType*>(p_As[i]),
                                               static_cast<const BDataType*>(p_Bs[i]),
                                               static_cast<const DDataType*>(p_Ds[i]),
                                               static_cast<EDataType*>(p_Es[i]),
                                               BlockStart,
                                               BlockEnd});
                }
            }
        }

        //  private:
        index_t group_count_;
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;

        std::vector<GemmBiasTransKernelArg> gemm_desc_kernel_arg_;

        void* gemm_descs_args_workspace_;

        index_t grid_size_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceGroupedGemmBiasCPermuteXdl::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            bool has_main_k_block_loop = true;

            for(std::size_t i = 0; i < arg.gemm_desc_kernel_arg_.size(); i++)
            {
                std::cout << "group: " << i << " arg.a_grid_desc_k0_m_k1_{"
                          << arg.gemm_desc_kernel_arg_[i].a_grid_desc_k0_m_k1_.GetLength(I0) << ", "
                          << arg.gemm_desc_kernel_arg_[i].a_grid_desc_k0_m_k1_.GetLength(I1) << ", "
                          << arg.gemm_desc_kernel_arg_[i].a_grid_desc_k0_m_k1_.GetLength(I2) << "}";

                std::cout << ", arg.b_grid_desc_k0_n_k1_{"
                          << arg.gemm_desc_kernel_arg_[i].b_grid_desc_k0_n_k1_.GetLength(I0) << ", "
                          << arg.gemm_desc_kernel_arg_[i].b_grid_desc_k0_n_k1_.GetLength(I1) << ", "
                          << arg.gemm_desc_kernel_arg_[i].b_grid_desc_k0_n_k1_.GetLength(I2) << "}";

                std::cout << ", arg.e_grid_desc_m_n_{ "
                          << arg.gemm_desc_kernel_arg_[i].e_grid_desc_m_n_.GetLength(I0) << ", "
                          << arg.gemm_desc_kernel_arg_[i].e_grid_desc_m_n_.GetLength(I1) << "}"
                          << std::endl;

                if(!GridwiseGemm::CheckValidity(arg.gemm_desc_kernel_arg_[i].a_grid_desc_k0_m_k1_,
                                                arg.gemm_desc_kernel_arg_[i].b_grid_desc_k0_n_k1_,
                                                arg.gemm_desc_kernel_arg_[i].e_grid_desc_m_n_,
                                                arg.gemm_desc_kernel_arg_[i].block_2_ctile_map_))
                {
                    throw std::runtime_error(
                        "wrong! GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3 has invalid setting");
                }

                const auto K = arg.gemm_desc_kernel_arg_[i].a_grid_desc_k0_m_k1_.GetLength(I0) *
                               arg.gemm_desc_kernel_arg_[i].a_grid_desc_k0_m_k1_.GetLength(I2);

                if(GridwiseGemm::CalculateHasMainKBlockLoop(K) != has_main_k_block_loop)
                {
                    throw std::runtime_error("wrong! not all gemm has_main_k_block_loop");
                }
            }

            hipGetErrorString(
                hipMemcpy(arg.gemm_descs_args_workspace_,
                          arg.gemm_desc_kernel_arg_.data(),
                          arg.gemm_desc_kernel_arg_.size() * sizeof(GemmBiasTransKernelArg),
                          hipMemcpyHostToDevice));

            float ave_time = 0;

            auto launch_kernel = [&](auto has_main_k_block_loop_) {
                const auto kernel = kernel_grouped_gemm_bias_transpose_xdl<GridwiseGemm,
                                                                           GemmBiasTransKernelArg,
                                                                           AElementwiseOperation,
                                                                           BElementwiseOperation,
                                                                           CElementwiseOperation,
                                                                           has_main_k_block_loop_>;

                return launch_and_time_kernel(
                    stream_config,
                    kernel,
                    dim3(arg.grid_size_),
                    dim3(BlockSize),
                    0,
                    cast_pointer_to_constant_address_space(arg.gemm_descs_args_workspace_),
                    arg.gemm_desc_kernel_arg_.size(),
                    arg.a_element_op_,
                    arg.b_element_op_,
                    arg.c_element_op_);
            };

            if(has_main_k_block_loop)
            {
                ave_time = launch_kernel(integral_constant<bool, true>{});
            }
            else
            {
                ave_time = launch_kernel(integral_constant<bool, false>{});
            }

            return ave_time;
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(ck::type_convert<ck::index_t>(arg.gemm_desc_kernel_arg_.size()) != arg.group_count_)
            return false;
        else
            return true;
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(std::vector<const void*>& p_As,
                             std::vector<const void*>& p_Bs,
                             std::vector<const void*>& p_Ds,
                             std::vector<void*>& p_Es,
                             std::vector<GemmBiasCPermuteDesc> gemm_bias_transpose_desc,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op)
    {
        return Argument{p_As,
                        p_Bs,
                        p_Ds,
                        p_Es,
                        gemm_bias_transpose_desc,
                        a_element_op,
                        b_element_op,
                        c_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(std::vector<const void*>& p_As,
                        std::vector<const void*>& p_Bs,
                        std::vector<const void*>& p_Ds,
                        std::vector<void*>& p_Es,
                        std::vector<GemmBiasCPermuteDesc>& gemm_bias_transpose_desc,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CElementwiseOperation c_element_op,
                        index_t /* KBatch */ = 1) override
    {
        return std::make_unique<Argument>(p_As,
                                          p_Bs,
                                          p_Ds,
                                          p_Es,
                                          gemm_bias_transpose_desc,
                                          a_element_op,
                                          b_element_op,
                                          c_element_op);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceGroupedGemmBiasCPermuteXdl"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1 << ", "
            << MPerXDL << ", "
            << NPerXDL << ", "
            << MXdlPerWave << ", "
            << NXdlPerWave
            << ">";
        // clang-format on

        return str.str();
    }

    size_t GetWorkSpaceSize(const BaseArgument* p_arg) const override
    {
        return dynamic_cast<const Argument*>(p_arg)->group_count_ * sizeof(GemmBiasTransKernelArg);
    }

    void SetWorkSpacePointer(BaseArgument* p_arg, void* workspace_ptr) const override
    {
        dynamic_cast<Argument*>(p_arg)->gemm_descs_args_workspace_ = workspace_ptr;
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
