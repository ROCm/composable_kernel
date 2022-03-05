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
#include "gridwise_grouped_gemm_xdlops_v2r3.hpp"
#include "gemm_specialization.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

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
          GemmSpecialization_t GemmSpecialization,
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
          ck::index_t NumPrefetch = 1,
          ck::index_t GroupCount  = 1>
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

        if constexpr(GemmSpecialization == GemmSpecialization_t::MNPadding)
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

        if constexpr(GemmSpecialization == GemmSpecialization_t::MNPadding)
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

        if constexpr(GemmSpecialization == GemmSpecialization_t::MNPadding)
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
    using GridwiseGemm = GridwiseGroupedGemm_k0mk1_k0nk1_mn_xdlops_v2r3<
        BlockSize,
        ADataType, // TODO: distinguish A/B datatype
        AccDataType,
        CDataType,
        InMemoryDataOperationEnum_t::Set,
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

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const ADataType* p_a_grid,
                 const BDataType* p_b_grid,
                 CDataType* p_c_grid,
                 std::vector<gemm_desc> gemm_shapes,
                 index_t M01,
                 index_t N01,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
            : p_a_grid_{p_a_grid},
              p_b_grid_{p_b_grid},
              p_c_grid_{p_c_grid},
              M01_{M01},
              N01_{N01},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op}
        {
            const index_t i = 0;

            const index_t M = gemm_shapes[Number<0>{}].M;
            const index_t N = gemm_shapes[Number<0>{}].N;
            const index_t K = gemm_shapes[Number<0>{}].K;

            const index_t StrideA = gemm_shapes[Number<0>{}].StrideA;
            const index_t StrideB = gemm_shapes[Number<0>{}].StrideB;
            const index_t StrideC = gemm_shapes[Number<0>{}].StrideC;

            a_grid_desc_k0_m_k1_(Number<0>{}) =
                DeviceGroupedGemmXdl::MakeAGridDescriptor_K0_M_K1(M, K, StrideA);
            b_grid_desc_k0_n_k1_(Number<0>{}) =
                DeviceGroupedGemmXdl::MakeBGridDescriptor_K0_N_K1(K, N, StrideB);
            c_grid_desc_m_n_(Number<0>{}) =
                DeviceGroupedGemmXdl::MakeCGridDescriptor_M_N(M, N, StrideC);

            if(GridwiseGemm::CheckValidity(a_grid_desc_k0_m_k1_[Number<0>{}],
                                           b_grid_desc_k0_n_k1_[Number<0>{}],
                                           c_grid_desc_m_n_[Number<0>{}],
                                           M01_,
                                           N01_))
            {
                c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_(Number<0>{}) =
                    GridwiseGemm::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(
                        c_grid_desc_m_n_[Number<0>{}]);

                block_2_ctile_map_(Number<0>{}) = GridwiseGemm::MakeDefaultBlock2CTileMap(
                    c_grid_desc_m_n_[Number<0>{}], M01, N01);
            }
        }

        //  private:
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        CDataType* p_c_grid_;
        StaticallyIndexedArray<AGridDesc_K0_M_K1, GroupCount> a_grid_desc_k0_m_k1_;
        StaticallyIndexedArray<BGridDesc_K0_N_K1, GroupCount> b_grid_desc_k0_n_k1_;
        StaticallyIndexedArray<CGridDesc_M_N, GroupCount> c_grid_desc_m_n_;
        StaticallyIndexedArray<typename GridwiseGemm::CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2, GroupCount>
            c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_;
        StaticallyIndexedArray<typename GridwiseGemm::DefaultBlock2CTileMap, GroupCount>
            block_2_ctile_map_;
        index_t M01_;
        index_t N01_;
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceGroupedGemmXdl::Argument;

        float Run(const Argument& arg, int nrepeat = 1)
        {
            std::cout << "Gemm " << i << ":" << std::endl;
            std::cout << "arg.a_grid_desc_k0_m_k1_{"
                      << arg.a_grid_desc_k0_m_k1_[Number<0>{}].GetLength(I0) << ", "
                      << arg.a_grid_desc_k0_m_k1_[Number<0>{}].GetLength(I1) << ", "
                      << arg.a_grid_desc_k0_m_k1_[Number<0>{}].GetLength(I2) << "}" << std::endl;

            std::cout << "arg.b_grid_desc_k0_n_k1_{"
                      << arg.b_grid_desc_k0_n_k1_[Number<0>{}].GetLength(I0) << ", "
                      << arg.b_grid_desc_k0_n_k1_[Number<0>{}].GetLength(I1) << ", "
                      << arg.b_grid_desc_k0_n_k1_[Number<0>{}].GetLength(I2) << "}" << std::endl;

            std::cout << "arg.c_grid_desc_m_n_{ " << arg.c_grid_desc_m_n_[Number<0>{}].GetLength(I0)
                      << ", " << arg.c_grid_desc_m_n_[Number<0>{}].GetLength(I1) << "}"
                      << std::endl;

            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_k0_m_k1_[Number<0>{}],
                                            arg.b_grid_desc_k0_n_k1_[Number<0>{}],
                                            arg.c_grid_desc_m_n_[Number<0>{}],
                                            arg.M01_,
                                            arg.N01_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3 has invalid setting");
            }

            const index_t grid_size =
                GridwiseGemm::CalculateGridSize(arg.c_grid_desc_m_n_[Number<0>{}]);

            const auto K0 = arg.a_grid_desc_k0_m_k1_[Number<0>{}].GetLength(I0);

            const bool has_main_k0_block_loop = GridwiseGemm::CalculateHasMainK0BlockLoop(K0);

            float ave_time = 0;

#if 1
            if(has_main_k0_block_loop)
            {
                const auto kernel = kernel_gemm_xdlops_v2r3<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    remove_reference_t<
                        StaticallyIndexedArray<DeviceGroupedGemmXdl::AGridDesc_K0_M_K1,
                                               GroupCount>>,
                    remove_reference_t<
                        StaticallyIndexedArray<DeviceGroupedGemmXdl::BGridDesc_K0_N_K1,
                                               GroupCount>>,
                    remove_reference_t<StaticallyIndexedArray<
                        typename GridwiseGemm::CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2,
                        GroupCount>>,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CElementwiseOperation,
                    remove_reference_t<
                        StaticallyIndexedArray<typename GridwiseGemm::DefaultBlock2CTileMap,
                                               GroupCount>>,
                    true>;

                ave_time = launch_and_time_kernel(kernel,
                                                  nrepeat,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.a_grid_desc_k0_m_k1_,
                                                  arg.b_grid_desc_k0_n_k1_,
                                                  arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_,
                                                  arg.a_element_op_,
                                                  arg.b_element_op_,
                                                  arg.c_element_op_,
                                                  arg.block_2_ctile_map_);
            }
            else
            {
                const auto kernel = kernel_gemm_xdlops_v2r3<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    remove_reference_t<
                        StaticallyIndexedArray<DeviceGroupedGemmXdl::AGridDesc_K0_M_K1,
                                               GroupCount>>,
                    remove_reference_t<
                        StaticallyIndexedArray<DeviceGroupedGemmXdl::BGridDesc_K0_N_K1,
                                               GroupCount>>,
                    remove_reference_t<StaticallyIndexedArray<
                        typename GridwiseGemm::CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2,
                        GroupCount>>,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CElementwiseOperation,
                    remove_reference_t<
                        StaticallyIndexedArray<typename GridwiseGemm::DefaultBlock2CTileMap,
                                               GroupCount>>,
                    false>;

                ave_time = launch_and_time_kernel(kernel,
                                                  nrepeat,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.a_grid_desc_k0_m_k1_,
                                                  arg.b_grid_desc_k0_n_k1_,
                                                  arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_,
                                                  arg.a_element_op_,
                                                  arg.b_element_op_,
                                                  arg.c_element_op_,
                                                  arg.block_2_ctile_map_);
            }

#endif
            return ave_time;
        }

        // polymorphic
        float Run(const BaseArgument* p_arg, int nrepeat = 1) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), nrepeat);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        const index_t i = 0;
        return GridwiseGemm::CheckValidity(arg.a_grid_desc_k0_m_k1_[Number<0>{}],
                                           arg.b_grid_desc_k0_n_k1_[Number<0>{}],
                                           arg.c_grid_desc_m_n_[Number<0>{}],
                                           arg.M01_,
                                           arg.N01_);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const ADataType* p_a,
                             const BDataType* p_b,
                             CDataType* p_c,
                             std::vector<gemm_desc> gemm_shapes,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op)
    {
        return Argument{p_a, p_b, p_c, gemm_shapes, 1, 1, a_element_op, b_element_op, c_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      void* p_c,
                                                      std::vector<gemm_desc> gemm_shapes,
                                                      AElementwiseOperation a_element_op,
                                                      BElementwiseOperation b_element_op,
                                                      CElementwiseOperation c_element_op,
                                                      index_t /* KBatch */ = 1) override
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
                                          static_cast<CDataType*>(p_c),
                                          gemm_shapes,
                                          1,
                                          1,
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
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
