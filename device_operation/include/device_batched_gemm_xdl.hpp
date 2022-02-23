#ifndef DEVICE_BATCHED_GEMM_XDL_HPP
#define DEVICE_BATCHED_GEMM_XDL_HPP

#include <iostream>
#include <sstream>
#include "device.hpp"
#include "device_base.hpp"
#include "device_gemm.hpp"
#include "common_header.hpp"
#include "tensor_layout.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_batched_gemm_xdlops_v2r3.hpp"

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
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t K0PerBlock,
          ck::index_t K1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_G_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_K1,
          bool ABlockLdsAddExtraM,
          typename BBlockTransferThreadClusterLengths_G_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_K1,
          bool BBlockLdsAddExtraN,
          ck::index_t CThreadTransferSrcDstVectorDim,
          ck::index_t CThreadTransferDstScalarPerVector>
struct DeviceBatchedGemmXdl
    : public DeviceGemm<AElementwiseOperation, BElementwiseOperation, CElementwiseOperation>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    static constexpr auto K1Number = Number<K1>{};

    static auto
    MakeAGridDescriptor_G_K0_M_K1(index_t BatchCount, index_t M, index_t K, index_t StrideA)
    {
        assert(K % K1 == 0);

        const index_t K0 = K / K1;

        const auto a_grid_desc_g_m_k = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(BatchCount, M, K),
                                                    make_tuple(M * StrideA, StrideA, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(BatchCount, M, K),
                                                    make_tuple(K * StrideA, I1, StrideA));
            }
        }();

        const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;

        const auto a_grid_desc_g_k0_mp_k1 =
            transform_tensor_descriptor(a_grid_desc_g_m_k,
                                        make_tuple(make_pass_through_transform(BatchCount),
                                                   make_unmerge_transform(make_tuple(K0, K1Number)),
                                                   make_right_pad_transform(M, PadM)),
                                        make_tuple(Sequence<0>{}, Sequence<2>{}, Sequence<1>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2>{}));

        return a_grid_desc_g_k0_mp_k1;
    }

    static auto
    MakeBGridDescriptor_G_K0_N_K1(index_t BatchCount, index_t K, index_t N, index_t StrideB)
    {
        assert(K % K1 == 0);

        const index_t K0 = K / K1;

        const auto b_grid_desc_g_k_n = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(BatchCount, K, N),
                                                    make_tuple(K * StrideB, StrideB, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(BatchCount, K, N),
                                                    make_tuple(N * StrideB, I1, StrideB));
            }
        }();

        const auto PadN = (NPerBlock - N % NPerBlock) % NPerBlock;

        const auto b_grid_desc_g_k0_np_k1 =
            transform_tensor_descriptor(b_grid_desc_g_k_n,
                                        make_tuple(make_pass_through_transform(BatchCount),
                                                   make_unmerge_transform(make_tuple(K0, K1Number)),
                                                   make_right_pad_transform(N, PadN)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2>{}));

        return b_grid_desc_g_k0_np_k1;
    }

    static auto MakeCGridDescriptor_G_M_N(index_t BatchCount, index_t M, index_t N, index_t StrideC)
    {
        const auto c_grid_desc_g_m_n = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(BatchCount, M, N),
                                                    make_tuple(M * StrideC, StrideC, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(BatchCount, M, N),
                                                    make_tuple(N * StrideC, I1, StrideC));
            }
        }();

        const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;
        const auto PadN = (NPerBlock - N % NPerBlock) % NPerBlock;

        const auto c_grid_desc_g_mp_np =
            transform_tensor_descriptor(c_grid_desc_g_m_n,
                                        make_tuple(make_pass_through_transform(BatchCount),
                                                   make_right_pad_transform(M, PadM),
                                                   make_right_pad_transform(N, PadN)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        return c_grid_desc_g_mp_np;
    }

    using AGridDesc_G_K0_M_K1 = decltype(MakeAGridDescriptor_G_K0_M_K1(1, 1, 1, 1));
    using BGridDesc_G_K0_N_K1 = decltype(MakeBGridDescriptor_G_K0_N_K1(1, 1, 1, 1));
    using CGridDesc_G_M_N     = decltype(MakeCGridDescriptor_G_M_N(1, 1, 1, 1));

    // GridwiseBatchedGemm
    using GridwiseBatchedGemm = GridwiseBatchedGemm_gk0mk1_gk0nk1_gmn_xdlops_v2r3<
        BlockSize,
        ADataType, // TODO: distinguish A/B datatype
        AccDataType,
        CDataType,
        InMemoryDataOperationEnum_t::Set,
        AGridDesc_G_K0_M_K1,
        BGridDesc_G_K0_N_K1,
        CGridDesc_G_M_N,
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
        ABlockTransferThreadClusterLengths_G_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsAddExtraM,
        BBlockTransferThreadClusterLengths_G_K0_N_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false, // BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsAddExtraN,
        Sequence<0, 1, 3, 5, 6, 7, 2, 4, 8>, // CThreadTransferSrcDstAccessOrder,
        CThreadTransferSrcDstVectorDim,
        CThreadTransferDstScalarPerVector>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const ADataType* p_a_grid,
                 const BDataType* p_b_grid,
                 CDataType* p_c_grid,
                 index_t M,
                 index_t N,
                 index_t K,
                 index_t StrideA,
                 index_t StrideB,
                 index_t StrideC,
                 index_t M01,
                 index_t N01,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op,
                 index_t BatchCount)
            : p_a_grid_{p_a_grid},
              p_b_grid_{p_b_grid},
              p_c_grid_{p_c_grid},
              a_grid_desc_g_k0_m_k1_{},
              b_grid_desc_g_k0_n_k1_{},
              c_grid_desc_g_m_n_{},
              c_grid_desc_g_m0_n0_m1_n1_m2_m3_m4_n2_{},
              block_2_ctile_map_{},
              M01_{M01},
              N01_{N01},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op}
        {
            a_grid_desc_g_k0_m_k1_ =
                DeviceBatchedGemmXdl::MakeAGridDescriptor_G_K0_M_K1(BatchCount, M, K, StrideA);
            b_grid_desc_g_k0_n_k1_ =
                DeviceBatchedGemmXdl::MakeBGridDescriptor_G_K0_N_K1(BatchCount, K, N, StrideB);
            c_grid_desc_g_m_n_ =
                DeviceBatchedGemmXdl::MakeCGridDescriptor_G_M_N(BatchCount, M, N, StrideC);

            if(GridwiseBatchedGemm::CheckValidity(
                   a_grid_desc_g_k0_m_k1_, b_grid_desc_g_k0_n_k1_, c_grid_desc_g_m_n_, M01_, N01_))
            {
                c_grid_desc_g_m0_n0_m1_n1_m2_m3_m4_n2_ =
                    GridwiseBatchedGemm::MakeCGridDescriptor_G_M0_N0_M1_N1_M2_M3_M4_N2(
                        c_grid_desc_g_m_n_);

                block_2_ctile_map_ =
                    GridwiseBatchedGemm::MakeDefaultBlock2CTileMap(c_grid_desc_g_m_n_, M01, N01);
            }
        }

        //  private:
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        CDataType* p_c_grid_;
        AGridDesc_G_K0_M_K1 a_grid_desc_g_k0_m_k1_;
        BGridDesc_G_K0_N_K1 b_grid_desc_g_k0_n_k1_;
        CGridDesc_G_M_N c_grid_desc_g_m_n_;
        typename GridwiseBatchedGemm::CGridDesc_G_M0_N0_M1_N1_M2_M3_M4_N2
            c_grid_desc_g_m0_n0_m1_n1_m2_m3_m4_n2_;
        typename GridwiseBatchedGemm::DefaultBlock2CTileMap block_2_ctile_map_;
        index_t M01_;
        index_t N01_;
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceBatchedGemmXdl::Argument;

        float Run(const Argument& arg, int nrepeat = 1)
        {
            {
                std::cout << "arg.a_grid_desc_g_k0_m_k1_{"
                          << arg.a_grid_desc_g_k0_m_k1_.GetLength(I0) << ", "
                          << arg.a_grid_desc_g_k0_m_k1_.GetLength(I1) << ", "
                          << arg.a_grid_desc_g_k0_m_k1_.GetLength(I2) << ", "
                          << arg.a_grid_desc_g_k0_m_k1_.GetLength(I3) << "}" << std::endl;

                std::cout << "arg.b_grid_desc_g_k0_n_k1_{"
                          << arg.b_grid_desc_g_k0_n_k1_.GetLength(I0) << ", "
                          << arg.b_grid_desc_g_k0_n_k1_.GetLength(I1) << ", "
                          << arg.b_grid_desc_g_k0_n_k1_.GetLength(I2) << ", "
                          << arg.b_grid_desc_g_k0_n_k1_.GetLength(I3) << "}" << std::endl;

                std::cout << "arg.c_grid_desc_g_m_n_{" << arg.c_grid_desc_g_m_n_.GetLength(I0)
                          << ", " << arg.c_grid_desc_g_m_n_.GetLength(I1) << ", "
                          << arg.c_grid_desc_g_m_n_.GetLength(I2) << "}" << std::endl;
            }

            if(!GridwiseBatchedGemm::CheckValidity(arg.a_grid_desc_g_k0_m_k1_,
                                                   arg.b_grid_desc_g_k0_n_k1_,
                                                   arg.c_grid_desc_g_m_n_,
                                                   arg.M01_,
                                                   arg.N01_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseBatchedGemm_km_kn_m0m1n0n1_xdlops_v2r3 has invalid setting");
            }

            const index_t grid_size =
                GridwiseBatchedGemm::CalculateGridSize(arg.c_grid_desc_g_m_n_);

            const auto K0 = arg.a_grid_desc_g_k0_m_k1_.GetLength(I1);

            const bool has_main_k0_block_loop =
                GridwiseBatchedGemm::CalculateHasMainK0BlockLoop(K0);

            float ave_time = 0;

            if(has_main_k0_block_loop)
            {
                const auto kernel = kernel_batched_gemm_xdlops_v2r3<
                    GridwiseBatchedGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    remove_reference_t<DeviceBatchedGemmXdl::AGridDesc_G_K0_M_K1>,
                    remove_reference_t<DeviceBatchedGemmXdl::BGridDesc_G_K0_N_K1>,
                    remove_reference_t<
                        typename GridwiseBatchedGemm::CGridDesc_G_M0_N0_M1_N1_M2_M3_M4_N2>,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CElementwiseOperation,
                    remove_reference_t<typename GridwiseBatchedGemm::DefaultBlock2CTileMap>,
                    true>;

                ave_time = launch_and_time_kernel(kernel,
                                                  nrepeat,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.a_grid_desc_g_k0_m_k1_,
                                                  arg.b_grid_desc_g_k0_n_k1_,
                                                  arg.c_grid_desc_g_m0_n0_m1_n1_m2_m3_m4_n2_,
                                                  arg.a_element_op_,
                                                  arg.b_element_op_,
                                                  arg.c_element_op_,
                                                  arg.block_2_ctile_map_);
            }
            else
            {
                const auto kernel = kernel_batched_gemm_xdlops_v2r3<
                    GridwiseBatchedGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    remove_reference_t<DeviceBatchedGemmXdl::AGridDesc_G_K0_M_K1>,
                    remove_reference_t<DeviceBatchedGemmXdl::BGridDesc_G_K0_N_K1>,
                    remove_reference_t<
                        typename GridwiseBatchedGemm::CGridDesc_G_M0_N0_M1_N1_M2_M3_M4_N2>,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CElementwiseOperation,
                    remove_reference_t<typename GridwiseBatchedGemm::DefaultBlock2CTileMap>,
                    false>;

                ave_time = launch_and_time_kernel(kernel,
                                                  nrepeat,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.a_grid_desc_g_k0_m_k1_,
                                                  arg.b_grid_desc_g_k0_n_k1_,
                                                  arg.c_grid_desc_g_m0_n0_m1_n1_m2_m3_m4_n2_,
                                                  arg.a_element_op_,
                                                  arg.b_element_op_,
                                                  arg.c_element_op_,
                                                  arg.block_2_ctile_map_);
            }

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
        return GridwiseBatchedGemm::CheckValidity(arg.a_grid_desc_g_k0_m_k1_,
                                                  arg.b_grid_desc_g_k0_n_k1_,
                                                  arg.c_grid_desc_g_m_n_,
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
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t StrideA,
                             index_t StrideB,
                             index_t StrideC,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op,
                             index_t BatchCount)
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
                        1,
                        1,
                        a_element_op,
                        b_element_op,
                        c_element_op,
                        BatchCount};
    }

    static auto MakeInvoker() { return Invoker{}; }

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
                                                      AElementwiseOperation a_element_op,
                                                      BElementwiseOperation b_element_op,
                                                      CElementwiseOperation c_element_op,
                                                      index_t BatchCount) override
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
                                          static_cast<CDataType*>(p_c),
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideC,
                                          1,
                                          1,
                                          a_element_op,
                                          b_element_op,
                                          c_element_op,
                                          BatchCount);
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
        str << "DeviceBatchedGemmXdl"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << K0PerBlock
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
