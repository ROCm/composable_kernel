#ifndef DRIVER_BATCHED_GEMM_XDLOPS_V2R3_HPP
#define DRIVER_BATCHED_GEMM_XDLOPS_V2R3_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_batched_gemm_xdlops_v2r3.hpp"

template <ck::index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          ck::InMemoryDataOperationEnum_t CGlobalMemoryDataOperation,
          typename ASK0MK1GridDesc,
          typename BK0NK1GridDesc,
          typename CSMNGridDesc,
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
__host__ float driver_batched_gemm_xdlops_v2r3(const FloatAB* p_a_grid,
                                               const FloatAB* p_b_grid,
                                               FloatC* p_c_grid,
                                               const ASK0MK1GridDesc& a_s_k0_m_k1_grid_desc,
                                               const BK0NK1GridDesc& b_k0_n_k1_grid_desc,
                                               const CSMNGridDesc& c_s_m_n_grid_desc,
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
    constexpr auto I3 = Number<3>{};

    using GridwiseBatchedGemm =
        GridwiseBatchedGemm_sk0mk1_k0nk1_smn_xdlops_v2r3<BlockSize,
                                                         FloatAB,
                                                         FloatAcc,
                                                         FloatC,
                                                         CGlobalMemoryDataOperation,
                                                         ASK0MK1GridDesc,
                                                         BK0NK1GridDesc,
                                                         CSMNGridDesc,
                                                         MPerBlock,
                                                         NPerBlock,
                                                         KPerBlock,
                                                         MPerXDL,
                                                         NPerXDL,
                                                         K1,
                                                         MRepeat,
                                                         NRepeat,
                                                         ABlockTransferThreadSliceLengths_K0_M_K1,
                                                         ABlockTransferThreadClusterLengths_K0_M_K1,
                                                         ABlockTransferThreadClusterArrangeOrder,
                                                         ABlockTransferSrcAccessOrder,
                                                         ABlockTransferSrcVectorDim,
                                                         ABlockTransferSrcScalarPerVector,
                                                         ABlockTransferDstScalarPerVector_K1,
                                                         AThreadTransferSrcResetCoordinateAfterRun,
                                                         BBlockTransferThreadSliceLengths_K0_N_K1,
                                                         BBlockTransferThreadClusterLengths_K0_N_K1,
                                                         BBlockTransferThreadClusterArrangeOrder,
                                                         BBlockTransferSrcAccessOrder,
                                                         BBlockTransferSrcVectorDim,
                                                         BBlockTransferSrcScalarPerVector,
                                                         BBlockTransferDstScalarPerVector_K1,
                                                         BThreadTransferSrcResetCoordinateAfterRun,
                                                         CThreadTransferSrcDstAccessOrder,
                                                         CThreadTransferSrcDstVectorDim,
                                                         CThreadTransferDstScalarPerVector,
                                                         AGridStepHacks,
                                                         BGridStepHacks,
                                                         CGridStepHacks,
                                                         AGridMoveSliceWindowStepHacks,
                                                         BGridMoveSliceWindowStepHacks,
                                                         CAccessOrderMRepeatNRepeat,
                                                         ABlockLdsAddExtraM,
                                                         BBlockLdsAddExtraN>;

    {
        std::cout << "a_s_k0_m_k1_grid_desc{" << a_s_k0_m_k1_grid_desc.GetLength(I0) << ", "
                  << a_s_k0_m_k1_grid_desc.GetLength(I1) << ", "
                  << a_s_k0_m_k1_grid_desc.GetLength(I2) << ", "
                  << a_s_k0_m_k1_grid_desc.GetLength(I3) << "}" << std::endl;

        std::cout << "b_k0_n_k1_grid_desc{" << b_k0_n_k1_grid_desc.GetLength(I0) << ", "
                  << b_k0_n_k1_grid_desc.GetLength(I1) << ", " << b_k0_n_k1_grid_desc.GetLength(I2)
                  << "}" << std::endl;

        std::cout << "c_s_m_n_grid_desc{ " << c_s_m_n_grid_desc.GetLength(I0) << ", "
                  << c_s_m_n_grid_desc.GetLength(I1) << ", " << c_s_m_n_grid_desc.GetLength(I2)
                  << "}" << std::endl;
    }

    if(!GridwiseBatchedGemm::CheckValidity(
           a_s_k0_m_k1_grid_desc, b_k0_n_k1_grid_desc, c_s_m_n_grid_desc, M01, N01))
    {
        throw std::runtime_error(
            "wrong! GridwiseGemm_km_kn_m0m1n0n1_xdlops_v2r3 has invalid setting");
    }

    const auto a_batch_stride =
        a_s_k0_m_k1_grid_desc.CalculateOffset(make_multi_index(1, 0, 0, 0)) -
        a_s_k0_m_k1_grid_desc.CalculateOffset(make_multi_index(0, 0, 0, 0));
    const auto c_batch_stride = c_s_m_n_grid_desc.CalculateOffset(make_multi_index(1, 0, 0)) -
                                c_s_m_n_grid_desc.CalculateOffset(make_multi_index(0, 0, 0));
    const auto c_s_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc =
        GridwiseBatchedGemm::MakeCSM0N0M1N1M2M3M4N2GridDescriptor(c_s_m_n_grid_desc);

    // printf("%s: %d %ld %ld\n", __FILE__, __LINE__, c_s_m_n_grid_desc.GetElementSpaceSize(),
    // c_s_m_n_grid_desc.GetElementSize()); printf("%s: %d %ld %ld\n", __FILE__, __LINE__,
    // c_s_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc.GetElementSpaceSize(),
    // c_s_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc.GetElementSize()); constexpr auto I4 = Number<4>{};
    // constexpr auto I5 = Number<5>{};
    // constexpr auto I6 = Number<6>{};
    // constexpr auto I7 = Number<7>{};
    // constexpr auto I8 = Number<8>{};
    // printf("%s: %d %d %d %d %d %d %d %d %d %d\n", __FILE__, __LINE__,
    //        c_s_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc.GetLength(I0),
    //        c_s_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc.GetLength(I1),
    //        c_s_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc.GetLength(I2),
    //        c_s_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc.GetLength(I3),
    //        c_s_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc.GetLength(I4),
    //        c_s_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc.GetLength(I5),
    //        c_s_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc.GetLength(I6),
    //        c_s_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc.GetLength(I7),
    //        c_s_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc.GetLength(I8));

    using CSM0N0M1N1M2M3M4N2GridDesc = decltype(c_s_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc);

    const auto c_block_cluster_adaptor =
        GridwiseBatchedGemm::MakeCBlockClusterAdaptor(c_s_m_n_grid_desc, M01, N01);

    using CBlockClusterAdaptor = decltype(c_block_cluster_adaptor);

    const index_t grid_size = GridwiseBatchedGemm::CalculateGridSize(c_s_m_n_grid_desc);

    const auto kernel =
        kernel_batched_gemm_xdlops_v2r3<GridwiseBatchedGemm,
                                        FloatAB,
                                        FloatC,
                                        remove_reference_t<ASK0MK1GridDesc>,
                                        remove_reference_t<BK0NK1GridDesc>,
                                        remove_reference_t<CSM0N0M1N1M2M3M4N2GridDesc>,
                                        remove_reference_t<CBlockClusterAdaptor>>;

#if CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VALUE
    float ave_time = launch_and_time_kernel(kernel,
                                            nrepeat,
                                            dim3(grid_size),
                                            dim3(BlockSize),
                                            0,
                                            p_a_grid,
                                            p_b_grid,
                                            p_c_grid,
                                            a_batch_stride,
                                            c_batch_stride,
                                            a_s_k0_m_k1_grid_desc,
                                            b_k0_n_k1_grid_desc,
                                            c_s_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc,
                                            c_block_cluster_adaptor);

#elif CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VOID_POINTER
    DeviceMem a_s_k0_m_k1_grid_desc_dev_buf(sizeof(ASK0MK1GridDesc));
    DeviceMem b_k0_n_k1_grid_desc_dev_buf(sizeof(BK0NK1GridDesc));
    DeviceMem c_s_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc_dev_buf(sizeof(CSM0N0M1N1M2M3M4N2GridDesc));
    DeviceMem c_block_cluster_adaptor_dev_buf(sizeof(CBlockClusterAdaptor));

    a_s_k0_m_k1_grid_desc_dev_buf.ToDevice(&a_s_k0_m_k1_grid_desc);
    b_k0_n_k1_grid_desc_dev_buf.ToDevice(&b_k0_n_k1_grid_desc);
    c_s_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc_dev_buf.ToDevice(&c_s_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc);
    c_block_cluster_adaptor_dev_buf.ToDevice(&c_block_cluster_adaptor);

    float ave_time = launch_and_time_kernel(
        kernel,
        nrepeat,
        dim3(grid_size),
        dim3(BlockSize),
        0,
        p_a_grid,
        p_b_grid,
        p_c_grid,
        cast_pointer_to_constant_address_space(a_s_k0_m_k1_grid_desc_dev_buf.GetDeviceBuffer()),
        cast_pointer_to_constant_address_space(b_k0_n_k1_grid_desc_dev_buf.GetDeviceBuffer()),
        cast_pointer_to_constant_address_space(
            c_s_m0_n0_m1_n1_m2_m3_m4_n2_grid_desc_dev_buf.GetDeviceBuffer()),
        cast_pointer_to_constant_address_space(c_block_cluster_adaptor_dev_buf.GetDeviceBuffer()));
#endif
    return ave_time;
}
#endif
