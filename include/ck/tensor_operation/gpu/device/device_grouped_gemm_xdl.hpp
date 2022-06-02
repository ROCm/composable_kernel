#ifndef DEVICE_GROUPED_GEMM_XDL_HPP
#define DEVICE_GROUPED_GEMM_XDL_HPP

#include <iostream>
#include <sstream>
#include "device.hpp"
#include "device_base.hpp"
#include "device_gemm.hpp"
#include "common_header.hpp"
#include "tensor_layout.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm_xdlops_v2r3.hpp"
#include "gemm_specialization.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename GemmDesc,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_grouped_gemm_xdlops_v2r3(const void CK_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
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
        gemm_desc_ptr[group_id].a_ptr,
        gemm_desc_ptr[group_id].b_ptr,
        gemm_desc_ptr[group_id].c_ptr,
        p_shared,
        gemm_desc_ptr[group_id].a_grid_desc_k0_m_k1_,
        gemm_desc_ptr[group_id].b_grid_desc_k0_n_k1_,
        gemm_desc_ptr[group_id].c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_,
        a_element_op,
        b_element_op,
        c_element_op,
        gemm_desc_ptr[group_id].grouped_gemm_block_2_ctile_map_);
#else
    ignore = gemm_descs_const;
    ignore = group_count;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = c_element_op;
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
}

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          GemmSpecialization GemmSpec,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t K0PerBlock,
          ck::index_t K1,
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
          ck::index_t CThreadTransferSrcDstVectorDim,
          ck::index_t CThreadTransferDstScalarPerVector,
          ck::index_t NumPrefetch   = 1,
          ck::index_t MaxGroupCount = 10>
struct DeviceGroupedGemmXdl
    : public DeviceGroupedGemm<AElementwiseOperation, BElementwiseOperation, CElementwiseOperation>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static constexpr auto K1Number = Number<K1>{};

    static auto MakeAGridDescriptor_K0_M_K1(index_t M, index_t K, index_t StrideA)
    {
        assert(K % K1 == 0);

        const index_t K0 = K / K1;

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
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                           make_right_pad_transform(M, PadM)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                a_grid_desc_m_k,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                           make_pass_through_transform(M)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
    }

    static auto MakeBGridDescriptor_K0_N_K1(index_t K, index_t N, index_t StrideB)
    {
        assert(K % K1 == 0);

        const index_t K0 = K / K1;

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
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                           make_right_pad_transform(N, PadN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
        else
        {
            return transform_tensor_descriptor(
                b_grid_desc_k_n,
                make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                           make_pass_through_transform(N)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
    }

    static auto MakeCGridDescriptor_M_N(index_t M, index_t N, index_t StrideC)
    {
        const auto c_grid_desc_m_n = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(StrideC, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(I1, StrideC));
            }
        }();

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
    using CGridDesc_M_N     = decltype(MakeCGridDescriptor_M_N(1, 1, 1));

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3<
        BlockSize,
        ADataType, // TODO: distinguish A/B datatype
        AccDataType,
        CDataType,
        InMemoryDataOperationEnum::Set,
        AGridDesc_K0_M_K1,
        BGridDesc_K0_N_K1,
        CGridDesc_M_N,
        AElementwiseOperation,
        BElementwiseOperation,
        CElementwiseOperation,
        MPerBlock,
        NPerBlock,
        K0PerBlock,
        MPerXDL,
        NPerXDL,
        K1,
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
        Sequence<0, 2, 4, 5, 6, 1, 3, 7>, // CThreadTransferSrcDstAccessOrder,
        CThreadTransferSrcDstVectorDim,
        CThreadTransferDstScalarPerVector,
        NumPrefetch>;

    struct GroupedGemmBlock2CTileMap
    {
        using UnderlyingBlock2CTileMap = typename GridwiseGemm::DefaultBlock2CTileMap;
        static_assert(
            std::is_same<decltype(GridwiseGemm::MakeDefaultBlock2CTileMap(CGridDesc_M_N{}, 1, 1)),
                         typename GridwiseGemm::DefaultBlock2CTileMap>::value,
            "Wrong! Should be the same type name");
        GroupedGemmBlock2CTileMap()
        {
            block_2_ctile_map_ = GridwiseGemm::MakeDefaultBlock2CTileMap(CGridDesc_M_N{}, 1, 1);
            BlockStart_        = -1;
        }

        GroupedGemmBlock2CTileMap(const CGridDesc_M_N& c_grid_desc_m_n,
                                  index_t M01,
                                  index_t N01,
                                  ck::index_t BlockStart)
        {
            block_2_ctile_map_ = GridwiseGemm::MakeDefaultBlock2CTileMap(c_grid_desc_m_n, M01, N01);
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

        __host__ bool CheckValidity(const CGridDesc_M_N& c_grid_desc_m_n) const
        {
            return block_2_ctile_map_.CheckValidity(c_grid_desc_m_n);
        }

        typename GridwiseGemm::DefaultBlock2CTileMap block_2_ctile_map_;
        ck::index_t BlockStart_;
    };

    struct GemmDescKernelArg
    {
        AGridDesc_K0_M_K1 a_grid_desc_k0_m_k1_;
        BGridDesc_K0_N_K1 b_grid_desc_k0_n_k1_;
        CGridDesc_M_N c_grid_desc_m_n_;

        typename GridwiseGemm::CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2
            c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_;

        GroupedGemmBlock2CTileMap grouped_gemm_block_2_ctile_map_;

        const ADataType* a_ptr;
        const BDataType* b_ptr;
        CDataType* c_ptr;

        ck::index_t BlockStart_, BlockEnd_;
    };

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(std::vector<const void*>& p_a,
                 std::vector<const void*>& p_b,
                 std::vector<void*>& p_c,
                 std::vector<GemmShape>& gemm_shapes,
                 index_t M01,
                 index_t N01,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
            : M01_{M01},
              N01_{N01},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op}
        {
            grid_size_ = 0;

            gemm_descs_args_workspace_ = nullptr;

            group_count_ = ck::type_convert<ck::index_t>(gemm_shapes.size());

            if(!(group_count_ == ck::type_convert<ck::index_t>(p_a.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_b.size()) &&
                 group_count_ == ck::type_convert<ck::index_t>(p_c.size())))
            {
                throw std::runtime_error("wrong! group_count_ != P_a/b/c.size");
            }

            gemm_desc_kernel_arg_.reserve(group_count_);

            for(std::size_t i = 0; i < gemm_shapes.size(); i++)
            {
                const index_t M = gemm_shapes[i].M;
                const index_t N = gemm_shapes[i].N;
                const index_t K = gemm_shapes[i].K;

                const index_t StrideA = gemm_shapes[i].StrideA;
                const index_t StrideB = gemm_shapes[i].StrideB;
                const index_t StrideC = gemm_shapes[i].StrideC;

                const auto a_grid_desc_k0_m_k1_ =
                    DeviceGroupedGemmXdl::MakeAGridDescriptor_K0_M_K1(M, K, StrideA);
                const auto b_grid_desc_k0_n_k1_ =
                    DeviceGroupedGemmXdl::MakeBGridDescriptor_K0_N_K1(K, N, StrideB);
                const auto c_grid_desc_m_n_ =
                    DeviceGroupedGemmXdl::MakeCGridDescriptor_M_N(M, N, StrideC);

                const index_t grid_size_grp =
                    GroupedGemmBlock2CTileMap(c_grid_desc_m_n_, M01, N01, 0)
                        .block_2_ctile_map_.CalculateGridSize(c_grid_desc_m_n_);

                const index_t BlockStart = grid_size_;
                const index_t BlockEnd   = grid_size_ + grid_size_grp;

                grid_size_ += grid_size_grp;

                const auto grouped_gemm_block_2_ctile_map_ =
                    GroupedGemmBlock2CTileMap(c_grid_desc_m_n_, M01, N01, BlockStart);

                if(GridwiseGemm::CheckValidity(a_grid_desc_k0_m_k1_,
                                               b_grid_desc_k0_n_k1_,
                                               c_grid_desc_m_n_,
                                               grouped_gemm_block_2_ctile_map_))
                {
                    const auto c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_ =
                        GridwiseGemm::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(c_grid_desc_m_n_);

                    gemm_desc_kernel_arg_.push_back(
                        GemmDescKernelArg{a_grid_desc_k0_m_k1_,
                                          b_grid_desc_k0_n_k1_,
                                          c_grid_desc_m_n_,
                                          c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_,
                                          grouped_gemm_block_2_ctile_map_,
                                          static_cast<const ADataType*>(p_a[i]),
                                          static_cast<const BDataType*>(p_b[i]),
                                          static_cast<CDataType*>(p_c[i]),
                                          BlockStart,
                                          BlockEnd});
                }
            }
        }

        //  private:
        index_t M01_;
        index_t N01_;
        index_t group_count_;
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;

        std::vector<GemmDescKernelArg> gemm_desc_kernel_arg_;

        void* gemm_descs_args_workspace_;

        index_t grid_size_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceGroupedGemmXdl::Argument;

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

                std::cout << ", arg.c_grid_desc_m_n_{ "
                          << arg.gemm_desc_kernel_arg_[i].c_grid_desc_m_n_.GetLength(I0) << ", "
                          << arg.gemm_desc_kernel_arg_[i].c_grid_desc_m_n_.GetLength(I1) << "}"
                          << std::endl;

                if(!GridwiseGemm::CheckValidity(
                       arg.gemm_desc_kernel_arg_[i].a_grid_desc_k0_m_k1_,
                       arg.gemm_desc_kernel_arg_[i].b_grid_desc_k0_n_k1_,
                       arg.gemm_desc_kernel_arg_[i].c_grid_desc_m_n_,
                       arg.gemm_desc_kernel_arg_[i].grouped_gemm_block_2_ctile_map_))
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
                          arg.gemm_desc_kernel_arg_.size() * sizeof(GemmDescKernelArg),
                          hipMemcpyHostToDevice));

            float ave_time = 0;

            if(has_main_k_block_loop)
            {
                const auto kernel =
                    kernel_grouped_gemm_xdlops_v2r3<GridwiseGemm,
                                                    ADataType, // TODO: distiguish A/B datatype
                                                    CDataType,
                                                    GemmDescKernelArg,
                                                    AElementwiseOperation,
                                                    BElementwiseOperation,
                                                    CElementwiseOperation,
                                                    true>;

                ave_time = launch_and_time_kernel(
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
            }
            else
            {
                const auto kernel =
                    kernel_grouped_gemm_xdlops_v2r3<GridwiseGemm,
                                                    ADataType, // TODO: distiguish A/B datatype
                                                    CDataType,
                                                    GemmDescKernelArg,
                                                    AElementwiseOperation,
                                                    BElementwiseOperation,
                                                    CElementwiseOperation,
                                                    false>;

                ave_time = launch_and_time_kernel(
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

    static auto MakeArgument(std::vector<const void*>& p_a,
                             std::vector<const void*>& p_b,
                             std::vector<void*>& p_c,
                             std::vector<GemmShape> gemm_shapes,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op)
    {
        return Argument{p_a, p_b, p_c, gemm_shapes, 1, 1, a_element_op, b_element_op, c_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(std::vector<const void*>& p_a,
                                                      std::vector<const void*>& p_b,
                                                      std::vector<void*>& p_c,
                                                      std::vector<GemmShape>& gemm_shapes,
                                                      AElementwiseOperation a_element_op,
                                                      BElementwiseOperation b_element_op,
                                                      CElementwiseOperation c_element_op,
                                                      index_t /* KBatch */ = 1) override
    {
        return std::make_unique<Argument>(
            p_a, p_b, p_c, gemm_shapes, 1, 1, a_element_op, b_element_op, c_element_op);
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
        str << "DeviceGroupedGemmXdl"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << K0PerBlock << ", "
            << K1 << ", "
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
        return dynamic_cast<const Argument*>(p_arg)->group_count_ * sizeof(GemmDescKernelArg);
    }

    void SetWorkSpacePointer(BaseArgument* p_arg, void* workspace_ptr) const override
    {
        dynamic_cast<Argument*>(p_arg)->gemm_descs_args_workspace_ = workspace_ptr;
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
