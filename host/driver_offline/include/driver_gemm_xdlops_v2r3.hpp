#ifndef DRIVER_GEMM_XDLOPS_V2R3_HPP
#define DRIVER_GEMM_XDLOPS_V2R3_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm_xdlops_v2r3.hpp"
#include "element_wise_operation.hpp"

template <ck::index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          ck::InMemoryDataOperationEnum_t CGlobalMemoryDataOperation,
          typename AGridDesc_K0_M_K1,
          typename BGridDesc_K0_N_K,
          typename CMNGridDesc,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t KPerBlock,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t K1,
          ck::index_t MRepeat,
          ck::index_t NRepeat,
          typename ABlockTransferThreadSliceLengths_K0_M_K1,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_K1,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          typename BBlockTransferThreadSliceLengths_K0_N_K1,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_K1,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          typename CThreadTransferSrcDstAccessOrder,
          ck::index_t CThreadTransferSrcDstVectorDim,
          ck::index_t CThreadTransferDstScalarPerVector,
          typename AGridStepHacks,
          typename BGridStepHacks,
          typename CGridStepHacks,
          typename AGridMoveSliceWindowStepHacks,
          typename BGridMoveSliceWindowStepHacks,
          bool CAccessOrderMRepeatNRepeat,
          bool ABlockLdsAddExtraM,
          bool BBlockLdsAddExtraN>
__host__ float driver_gemm_xdlops_v2r3(const FloatAB* p_a_grid,
                                       const FloatAB* p_b_grid,
                                       FloatC* p_c_grid,
                                       const AGridDesc_K0_M_K1& a_grid_desc_k0_m_k1,
                                       const BGridDesc_K0_N_K& b_grid_desc_k0_n_k1,
                                       const CMNGridDesc& c_grid_desc_m_n,
                                       ck::index_t M01,
                                       ck::index_t N01,
                                       AGridStepHacks,
                                       BGridStepHacks,
                                       CGridStepHacks,
                                       AGridMoveSliceWindowStepHacks,
                                       BGridMoveSliceWindowStepHacks,
                                       ck::index_t nrepeat)
{
    using namespace ck;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};

    using ElementwiseOperation = ck::tensor_operation::element_wise::PassThrough;

    using GridwiseGemm =
        GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3<BlockSize,
                                                FloatAB,
                                                FloatAcc,
                                                FloatC,
                                                CGlobalMemoryDataOperation,
                                                AGridDesc_K0_M_K1,
                                                BGridDesc_K0_N_K,
                                                CMNGridDesc,
                                                ElementwiseOperation,
                                                ElementwiseOperation,
                                                ElementwiseOperation,
                                                MPerBlock,
                                                NPerBlock,
                                                KPerBlock,
                                                MPerXDL,
                                                NPerXDL,
                                                K1,
                                                MRepeat,
                                                NRepeat,
                                                ABlockTransferThreadClusterLengths_K0_M_K1,
                                                ABlockTransferThreadClusterArrangeOrder,
                                                ABlockTransferSrcAccessOrder,
                                                ABlockTransferSrcVectorDim,
                                                ABlockTransferSrcScalarPerVector,
                                                ABlockTransferDstScalarPerVector_K1,
                                                AThreadTransferSrcResetCoordinateAfterRun,
                                                ABlockLdsAddExtraM,
                                                BBlockTransferThreadClusterLengths_K0_N_K1,
                                                BBlockTransferThreadClusterArrangeOrder,
                                                BBlockTransferSrcAccessOrder,
                                                BBlockTransferSrcVectorDim,
                                                BBlockTransferSrcScalarPerVector,
                                                BBlockTransferDstScalarPerVector_K1,
                                                BThreadTransferSrcResetCoordinateAfterRun,
                                                BBlockLdsAddExtraN,
                                                CThreadTransferSrcDstAccessOrder,
                                                CThreadTransferSrcDstVectorDim,
                                                CThreadTransferDstScalarPerVector>;

    {
        std::cout << "a_grid_desc_k0_m_k1{" << a_grid_desc_k0_m_k1.GetLength(I0) << ", "
                  << a_grid_desc_k0_m_k1.GetLength(I1) << ", " << a_grid_desc_k0_m_k1.GetLength(I2)
                  << "}" << std::endl;

        std::cout << "b_grid_desc_k0_n_k1{" << b_grid_desc_k0_n_k1.GetLength(I0) << ", "
                  << b_grid_desc_k0_n_k1.GetLength(I1) << ", " << b_grid_desc_k0_n_k1.GetLength(I2)
                  << "}" << std::endl;

        std::cout << "c_grid_desc_m_n{ " << c_grid_desc_m_n.GetLength(I0) << ", "
                  << c_grid_desc_m_n.GetLength(I1) << "}" << std::endl;
    }

    if(!GridwiseGemm::CheckValidity(
           a_grid_desc_k0_m_k1, b_grid_desc_k0_n_k1, c_grid_desc_m_n, M01, N01))
    {
        throw std::runtime_error(
            "wrong! GridwiseGemm_km_kn_m0m1n0n1_xdlops_v2r3 has invalid setting");
    }

    const auto c_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc =
        GridwiseGemm::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(c_grid_desc_m_n);

    using CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2 = decltype(c_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc);

    const auto block_2_ctile_map = GridwiseGemm::MakeDefaultBlock2CTileMap(c_grid_desc_m_n, M01, N01);

    using Block2CTileMap = decltype(block_2_ctile_map);

    const index_t grid_size = GridwiseGemm::CalculateGridSize(c_grid_desc_m_n);

    const auto K0 = a_grid_desc_k0_m_k1.GetLength(I0);

    const bool has_main_k0_block_loop = GridwiseGemm::CalculateHasMainK0BlockLoop(K0);

    float ave_time = 0;

    auto element_op_ = ElementwiseOperation{};

#if CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VALUE
    if(has_main_k0_block_loop)
    {
        const auto kernel =
            kernel_gemm_xdlops_v2r3<GridwiseGemm,
                                    FloatAB,
                                    FloatC,
                                    remove_reference_t<AGridDesc_K0_M_K1>,
                                    remove_reference_t<BGridDesc_K0_N_K>,
                                    remove_reference_t<CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2>,
                                    ElementwiseOperation,
                                    ElementwiseOperation,
                                    ElementwiseOperation,
                                    remove_reference_t<Block2CTileMap>,
                                    true>;

        ave_time = launch_and_time_kernel(kernel,
                                          nrepeat,
                                          dim3(grid_size),
                                          dim3(BlockSize),
                                          0,
                                          p_a_grid,
                                          p_b_grid,
                                          p_c_grid,
                                          a_grid_desc_k0_m_k1,
                                          b_grid_desc_k0_n_k1,
                                          c_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc,
                                          element_op_,
                                          element_op_,
                                          element_op_,
                                          block_2_ctile_map);
    }
    else
    {
        const auto kernel =
            kernel_gemm_xdlops_v2r3<GridwiseGemm,
                                    FloatAB,
                                    FloatC,
                                    remove_reference_t<AGridDesc_K0_M_K1>,
                                    remove_reference_t<BGridDesc_K0_N_K>,
                                    remove_reference_t<CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2>,
                                    ElementwiseOperation,
                                    ElementwiseOperation,
                                    ElementwiseOperation,
                                    remove_reference_t<Block2CTileMap>,
                                    false>;

        ave_time = launch_and_time_kernel(kernel,
                                          nrepeat,
                                          dim3(grid_size),
                                          dim3(BlockSize),
                                          0,
                                          p_a_grid,
                                          p_b_grid,
                                          p_c_grid,
                                          a_grid_desc_k0_m_k1,
                                          b_grid_desc_k0_n_k1,
                                          c_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc,
                                          element_op_,
                                          element_op_,
                                          element_op_,
                                          block_2_ctile_map);
    }
#elif CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VOID_POINTER
    DeviceMem a_grid_desc_k0_m_k1_dev_buf(sizeof(AGridDesc_K0_M_K1));
    DeviceMem b_grid_desc_k0_n_k1_dev_buf(sizeof(BGridDesc_K0_N_K));
    DeviceMem c_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc_dev_buf(
        sizeof(CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2));
    DeviceMem block_2_ctile_map_dev_buf(sizeof(Block2CTileMap));

    a_grid_desc_k0_m_k1_dev_buf.ToDevice(&a_grid_desc_k0_m_k1);
    b_grid_desc_k0_n_k1_dev_buf.ToDevice(&b_grid_desc_k0_n_k1);
    c_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc_dev_buf.ToDevice(&c_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc);
    block_2_ctile_map_dev_buf.ToDevice(&block_2_ctile_map);

    if(has_main_k0_block_loop)
    {
        const auto kernel =
            kernel_gemm_xdlops_v2r3<GridwiseGemm,
                                    FloatAB,
                                    FloatC,
                                    remove_reference_t<AGridDesc_K0_M_K1>,
                                    remove_reference_t<BGridDesc_K0_N_K>,
                                    remove_reference_t<CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2>,
                                    remove_reference_t<Block2CTileMap>,
                                    true>;

        ave_time = launch_and_time_kernel(
            kernel,
            nrepeat,
            dim3(grid_size),
            dim3(BlockSize),
            0,
            p_a_grid,
            p_b_grid,
            p_c_grid,
            cast_pointer_to_constant_address_space(a_grid_desc_k0_m_k1_dev_buf.GetDeviceBuffer()),
            cast_pointer_to_constant_address_space(b_grid_desc_k0_n_k1_dev_buf.GetDeviceBuffer()),
            cast_pointer_to_constant_address_space(
                c_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc_dev_buf.GetDeviceBuffer()),
            cast_pointer_to_constant_address_space(block_2_ctile_map_dev_buf.GetDeviceBuffer()));
    }
    else
    {
        const auto kernel =
            kernel_gemm_xdlops_v2r3<GridwiseGemm,
                                    FloatAB,
                                    FloatC,
                                    remove_reference_t<AGridDesc_K0_M_K1>,
                                    remove_reference_t<BGridDesc_K0_N_K>,
                                    remove_reference_t<CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2>,
                                    remove_reference_t<Block2CTileMap>,
                                    false>;

        ave_time = launch_and_time_kernel(
            kernel,
            nrepeat,
            dim3(grid_size),
            dim3(BlockSize),
            0,
            p_a_grid,
            p_b_grid,
            p_c_grid,
            cast_pointer_to_constant_address_space(a_grid_desc_k0_m_k1_dev_buf.GetDeviceBuffer()),
            cast_pointer_to_constant_address_space(b_grid_desc_k0_n_k1_dev_buf.GetDeviceBuffer()),
            cast_pointer_to_constant_address_space(
                c_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc_dev_buf.GetDeviceBuffer()),
            cast_pointer_to_constant_address_space(block_2_ctile_map_dev_buf.GetDeviceBuffer()));
    }
}
#endif
    return ave_time;
}
#endif
