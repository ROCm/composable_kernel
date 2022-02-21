#ifndef DEVICE_GEMM_XDL_C_SHUFFLE_HPP
#define DEVICE_GEMM_XDL_C_SHUFFLE_HPP

#include <iostream>
#include <sstream>
#include "device.hpp"
#include "device_base.hpp"
#include "device_gemm.hpp"
#include "device_gemm_xdl.hpp"
#include "common_header.hpp"
#include "tensor_layout.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm_xdlops_v3r1.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <
    typename ADataType,
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
    typename CBlockTransferClusterLengths_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl,
    index_t CBlockTransferScalarPerVector_NWaveNPerXdl>
struct DeviceGemmXdl_C_Shuffle
    : public DeviceGemm<AElementwiseOperation, BElementwiseOperation, CElementwiseOperation>
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

        const auto a_grid_desc_k0_m_k1 =
            transform_tensor_descriptor(a_grid_desc_m_k,
                                        make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                                                   make_pass_through_transform(M)),
                                        make_tuple(Sequence<1>{}, Sequence<0>{}),
                                        make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

        return a_grid_desc_k0_m_k1;
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

        const auto b_grid_desc_k0_n_k1 =
            transform_tensor_descriptor(b_grid_desc_k_n,
                                        make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                                                   make_pass_through_transform(N)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                                        make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

        return b_grid_desc_k0_n_k1;
    }

    static auto MakeCGridDescriptor_M_N(index_t M, index_t N, index_t StrideC)
    {
        if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
        {
            return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(StrideC, I1));
        }
        else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, CLayout>::value)
        {
            return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(I1, StrideC));
        }
    }

    using AGridDesc_K0_M_K1 = decltype(MakeAGridDescriptor_K0_M_K1(1, 1, 1));
    using BGridDesc_K0_N_K1 = decltype(MakeBGridDescriptor_K0_N_K1(1, 1, 1));
    using CGridDesc_M_N     = decltype(MakeCGridDescriptor_M_N(1, 1, 1));

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v3r1<
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
        false,
        ABlockLdsAddExtraM,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false,
        BBlockLdsAddExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CBlockTransferClusterLengths_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl,
        CBlockTransferScalarPerVector_NWaveNPerXdl>;

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
                 CElementwiseOperation c_element_op)
            : p_a_grid_{p_a_grid},
              p_b_grid_{p_b_grid},
              p_c_grid_{p_c_grid},
              a_grid_desc_k0_m_k1_{},
              b_grid_desc_k0_n_k1_{},
              c_grid_desc_m_n_{},
              c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_{},
              block_2_ctile_map_{},
              M01_{M01},
              N01_{N01},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op}
        {
            a_grid_desc_k0_m_k1_ =
                DeviceGemmXdl_C_Shuffle::MakeAGridDescriptor_K0_M_K1(M, K, StrideA);
            b_grid_desc_k0_n_k1_ =
                DeviceGemmXdl_C_Shuffle::MakeBGridDescriptor_K0_N_K1(K, N, StrideB);
            c_grid_desc_m_n_ = DeviceGemmXdl_C_Shuffle::MakeCGridDescriptor_M_N(M, N, StrideC);

            if(GridwiseGemm::CheckValidity(
                   a_grid_desc_k0_m_k1_, b_grid_desc_k0_n_k1_, c_grid_desc_m_n_, M01_, N01_))
            {
                c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_ =
                    GridwiseGemm::
                        MakeCGridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl(
                            c_grid_desc_m_n_);

                block_2_ctile_map_ = GridwiseGemm::MakeDefaultBlock2CTileMap(c_grid_desc_m_n_, M01, N01);
            }
        }

        //  private:
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        CDataType* p_c_grid_;
        AGridDesc_K0_M_K1 a_grid_desc_k0_m_k1_;
        BGridDesc_K0_N_K1 b_grid_desc_k0_n_k1_;
        CGridDesc_M_N c_grid_desc_m_n_;
        typename GridwiseGemm::
            CGridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl
                c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_;
        typename GridwiseGemm::DefaultBlock2CTileMap block_2_ctile_map_;
        index_t M01_;
        index_t N01_;
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceGemmXdl_C_Shuffle::Argument;

        float Run(const Argument& arg, int nrepeat = 1)
        {
            {
                std::cout << "arg.a_grid_desc_k0_m_k1_{" << arg.a_grid_desc_k0_m_k1_.GetLength(I0)
                          << ", " << arg.a_grid_desc_k0_m_k1_.GetLength(I1) << ", "
                          << arg.a_grid_desc_k0_m_k1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.b_grid_desc_k0_n_k1_{" << arg.b_grid_desc_k0_n_k1_.GetLength(I0)
                          << ", " << arg.b_grid_desc_k0_n_k1_.GetLength(I1) << ", "
                          << arg.b_grid_desc_k0_n_k1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.c_grid_desc_m_n_{ " << arg.c_grid_desc_m_n_.GetLength(I0) << ", "
                          << arg.c_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;
            }

            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_k0_m_k1_,
                                            arg.b_grid_desc_k0_n_k1_,
                                            arg.c_grid_desc_m_n_,
                                            arg.M01_,
                                            arg.N01_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemm_km_kn_m0m1n0n1_xdlops_v2r3 has invalid setting");
            }

            const index_t grid_size = GridwiseGemm::CalculateGridSize(arg.c_grid_desc_m_n_);

            const auto K0 = arg.a_grid_desc_k0_m_k1_.GetLength(I0);

            const bool has_main_k0_block_loop = GridwiseGemm::CalculateHasMainK0BlockLoop(K0);

            float ave_time = 0;

            if(has_main_k0_block_loop)
            {
                const auto kernel = kernel_gemm_xdlops_v3r1<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    remove_reference_t<DeviceGemmXdl_C_Shuffle::AGridDesc_K0_M_K1>,
                    remove_reference_t<DeviceGemmXdl_C_Shuffle::BGridDesc_K0_N_K1>,
                    remove_reference_t<
                        typename GridwiseGemm::
                            CGridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl>,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CElementwiseOperation,
                    remove_reference_t<typename GridwiseGemm::DefaultBlock2CTileMap>,
                    true>;

                ave_time = launch_and_time_kernel(
                    kernel,
                    nrepeat,
                    dim3(grid_size),
                    dim3(BlockSize),
                    0,
                    arg.p_a_grid_,
                    arg.p_b_grid_,
                    arg.p_c_grid_,
                    arg.a_grid_desc_k0_m_k1_,
                    arg.b_grid_desc_k0_n_k1_,
                    arg.c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_,
                    arg.a_element_op_,
                    arg.b_element_op_,
                    arg.c_element_op_,
                    arg.block_2_ctile_map_);
            }
            else
            {
                const auto kernel = kernel_gemm_xdlops_v3r1<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    remove_reference_t<DeviceGemmXdl_C_Shuffle::AGridDesc_K0_M_K1>,
                    remove_reference_t<DeviceGemmXdl_C_Shuffle::BGridDesc_K0_N_K1>,
                    remove_reference_t<
                        typename GridwiseGemm::
                            CGridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl>,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CElementwiseOperation,
                    remove_reference_t<typename GridwiseGemm::DefaultBlock2CTileMap>,
                    false>;

                ave_time = launch_and_time_kernel(
                    kernel,
                    nrepeat,
                    dim3(grid_size),
                    dim3(BlockSize),
                    0,
                    arg.p_a_grid_,
                    arg.p_b_grid_,
                    arg.p_c_grid_,
                    arg.a_grid_desc_k0_m_k1_,
                    arg.b_grid_desc_k0_n_k1_,
                    arg.c_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_,
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
        return GridwiseGemm::CheckValidity(arg.a_grid_desc_k0_m_k1_,
                                           arg.b_grid_desc_k0_n_k1_,
                                           arg.c_grid_desc_m_n_,
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
                             CElementwiseOperation c_element_op)
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
                        c_element_op};
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
                                                      index_t /* KBatch */ = 1) override
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
        str << "DeviceGemmXdl_C_Shuffle"
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
