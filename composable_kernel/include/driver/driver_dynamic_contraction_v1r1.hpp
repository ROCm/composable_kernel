#ifndef CK_DRIVER_DYNAMIC_CONTRACTION_V1R1_HPP
#define CK_DRIVER_DYNAMIC_CONTRACTION_V1R1_HPP

#include "common_header.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "gridwise_dynamic_contraction_v1r1.hpp"

namespace ck {

template <index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          InMemoryDataOperation CGlobalMemoryDataOperation,
          typename AGKGM0GM1GridDesc,
          typename BGKGN0GN1GridDesc,
          typename CGM0GM1GN0GN1GridDesc,
          index_t GM1PerBlockGM11,
          index_t GN1PerBlockGN11,
          index_t KPerBlock,
          index_t M1PerThread,
          index_t N1PerThread,
          index_t KPerThread,
          index_t M1N1ThreadClusterM10,
          index_t M1N1ThreadClusterN10,
          index_t M1N1ThreadClusterM11,
          index_t M1N1ThreadClusterN11,
          typename ABlockTransferThreadSliceLengths_GK_GM0_GM10_GM11,
          typename ABlockTransferThreadClusterLengths_GK_GM0_GM10_GM11,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_GM11,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          typename BBlockTransferThreadSliceLengths_GK_GN0_GN10_GN11,
          typename BBlockTransferThreadClusterLengths_GK_GN0_GN10_GN11,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_GN11,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          typename CThreadTransferSrcDstAccessOrder,
          index_t CThreadTransferSrcDstVectorDim,
          index_t CThreadTransferDstScalarPerVector,
          typename AGridIteratorHacks,
          typename BGridIteratorHacks,
          typename CGridIteratorHacks,
          typename AGridMoveSliceWindowIteratorHacks,
          typename BGridMoveSliceWindowIteratorHacks>
__host__ float
driver_dynamic_contraction_v1r1(const FloatAB* p_a_grid,
                                const FloatAB* p_b_grid,
                                FloatC* p_c_grid,
                                const AGKGM0GM1GridDesc& a_gk_gm0_gm1_grid_desc,
                                const BGKGN0GN1GridDesc& b_gk_gn0_gn1_grid_desc,
                                const CGM0GM1GN0GN1GridDesc& c_gm0_gm1_gn0_gn1_grid_desc,
                                AGridIteratorHacks,
                                BGridIteratorHacks,
                                CGridIteratorHacks,
                                AGridMoveSliceWindowIteratorHacks,
                                BGridMoveSliceWindowIteratorHacks,
                                index_t nrepeat)

{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};
    constexpr auto I5 = Number<5>{};

    // GEMM
    using GridwiseContraction = GridwiseDynamicContraction_km0m1_kn0n1_m0m1n0n1_v1r1<
        BlockSize,
        FloatAB,
        FloatAcc,
        FloatC,
        CGlobalMemoryDataOperation,
        AGKGM0GM1GridDesc,
        BGKGN0GN1GridDesc,
        CGM0GM1GN0GN1GridDesc,
        GM1PerBlockGM11,
        GN1PerBlockGN11,
        KPerBlock,
        M1PerThread,
        N1PerThread,
        KPerThread,
        M1N1ThreadClusterM10,
        M1N1ThreadClusterN10,
        M1N1ThreadClusterM11,
        M1N1ThreadClusterN11,
        ABlockTransferThreadSliceLengths_GK_GM0_GM10_GM11,
        ABlockTransferThreadClusterLengths_GK_GM0_GM10_GM11,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_GM11,
        AThreadTransferSrcResetCoordinateAfterRun,
        BBlockTransferThreadSliceLengths_GK_GN0_GN10_GN11,
        BBlockTransferThreadClusterLengths_GK_GN0_GN10_GN11,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_GN11,
        BThreadTransferSrcResetCoordinateAfterRun,
        CThreadTransferSrcDstAccessOrder,
        CThreadTransferSrcDstVectorDim,
        CThreadTransferDstScalarPerVector,
        AGridIteratorHacks,
        BGridIteratorHacks,
        CGridIteratorHacks,
        AGridMoveSliceWindowIteratorHacks,
        BGridMoveSliceWindowIteratorHacks>;

    const auto K = a_gk_gm0_gm1_grid_desc.GetLength(I0);

    if(!GridwiseContraction::CheckValidity(
           a_gk_gm0_gm1_grid_desc, b_gk_gn0_gn1_grid_desc, c_gm0_gm1_gn0_gn1_grid_desc))
    {
        throw std::runtime_error(
            "wrong! GridwiseDynamicContraction_km_kn0n1_mn0n1_v1r1 has invalid setting");
    }

    const auto a_gk_gm0_gm10_gm11_grid_desc =
        GridwiseContraction::MakeAGKGM0GM10GM11GridDescriptor(a_gk_gm0_gm1_grid_desc);
    const auto b_gk_gn0_gn10_gn11_grid_desc =
        GridwiseContraction::MakeBGKGN0GN10GN11GridDescriptor(b_gk_gn0_gn1_grid_desc);

    using AGKGM0GM10GM11GridDesc = decltype(a_gk_gm0_gm10_gm11_grid_desc);
    using BGKGN0GN10GN11GridDesc = decltype(b_gk_gn0_gn10_gn11_grid_desc);

    // c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc
    const auto c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc =
        GridwiseContraction::MakeCGM10BM0BM1GN10BN0BN1GridDescriptor(c_gm0_gm1_gn0_gn1_grid_desc);

    using CGM10BM0BM1GN10BN0BN1GridDesc = decltype(c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc);

    // c_blockid_to_gm10_gn10_block_cluster_adaptor
    const auto c_blockid_to_gm10_gn10_block_cluster_adaptor =
        GridwiseContraction::MakeCBlockIdToGM10GN10BlockClusterAdaptor(c_gm0_gm1_gn0_gn1_grid_desc);

    using CBlockIdToGM10GN10BlockClusterAdaptor =
        decltype(c_blockid_to_gm10_gn10_block_cluster_adaptor);

    const index_t grid_size = GridwiseContraction::CalculateGridSize(c_gm0_gm1_gn0_gn1_grid_desc);

    const bool has_main_k_block_loop = GridwiseContraction::CalculateHasMainKBlockLoop(K);

    const bool has_double_tail_k_block_loop =
        GridwiseContraction::CalculateHasDoubleTailKBlockLoop(K);

    {
        std::cout << "a_gk_gm0_gm10_gm11_grid_desc{" << a_gk_gm0_gm10_gm11_grid_desc.GetLength(I0)
                  << ", " << a_gk_gm0_gm10_gm11_grid_desc.GetLength(I1) << ", "
                  << a_gk_gm0_gm10_gm11_grid_desc.GetLength(I2) << ", "
                  << a_gk_gm0_gm10_gm11_grid_desc.GetLength(I3) << "}" << std::endl;

        std::cout << "b_gk_gn0_gn10_gn11_grid_desc{" << b_gk_gn0_gn10_gn11_grid_desc.GetLength(I0)
                  << ", " << b_gk_gn0_gn10_gn11_grid_desc.GetLength(I1) << ", "
                  << b_gk_gn0_gn10_gn11_grid_desc.GetLength(I2) << ", "
                  << b_gk_gn0_gn10_gn11_grid_desc.GetLength(I3) << "}" << std::endl;

        std::cout << "c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc{ "
                  << c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc.GetLength(I0) << ", "
                  << c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc.GetLength(I1) << ", "
                  << c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc.GetLength(I2) << ", "
                  << c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc.GetLength(I3) << ", "
                  << c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc.GetLength(I4) << ", "
                  << c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc.GetLength(I5) << "}" << std::endl;
    }

    float ave_time = 0;

    if(has_main_k_block_loop && has_double_tail_k_block_loop)
    {
        const auto kernel = kernel_dynamic_contraction_v1r1<
            GridwiseContraction,
            FloatAB,
            FloatC,
            remove_reference_t<AGKGM0GM10GM11GridDesc>,
            remove_reference_t<BGKGN0GN10GN11GridDesc>,
            remove_reference_t<CGM10BM0BM1GN10BN0BN1GridDesc>,
            remove_reference_t<CBlockIdToGM10GN10BlockClusterAdaptor>,
            true,
            true>;

        ave_time = launch_and_time_kernel(kernel,
                                          nrepeat,
                                          dim3(grid_size),
                                          dim3(BlockSize),
                                          0,
                                          0,
                                          p_a_grid,
                                          p_b_grid,
                                          p_c_grid,
                                          a_gk_gm0_gm10_gm11_grid_desc,
                                          b_gk_gn0_gn10_gn11_grid_desc,
                                          c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc,
                                          c_blockid_to_gm10_gn10_block_cluster_adaptor);
    }
    else if(has_main_k_block_loop && !has_double_tail_k_block_loop)
    {
        const auto kernel = kernel_dynamic_contraction_v1r1<
            GridwiseContraction,
            FloatAB,
            FloatC,
            remove_reference_t<AGKGM0GM10GM11GridDesc>,
            remove_reference_t<BGKGN0GN10GN11GridDesc>,
            remove_reference_t<CGM10BM0BM1GN10BN0BN1GridDesc>,
            remove_reference_t<CBlockIdToGM10GN10BlockClusterAdaptor>,
            true,
            false>;

        ave_time = launch_and_time_kernel(kernel,
                                          nrepeat,
                                          dim3(grid_size),
                                          dim3(BlockSize),
                                          0,
                                          0,
                                          p_a_grid,
                                          p_b_grid,
                                          p_c_grid,
                                          a_gk_gm0_gm10_gm11_grid_desc,
                                          b_gk_gn0_gn10_gn11_grid_desc,
                                          c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc,
                                          c_blockid_to_gm10_gn10_block_cluster_adaptor);
    }
    else if(!has_main_k_block_loop && has_double_tail_k_block_loop)
    {
        const auto kernel = kernel_dynamic_contraction_v1r1<
            GridwiseContraction,
            FloatAB,
            FloatC,
            remove_reference_t<AGKGM0GM10GM11GridDesc>,
            remove_reference_t<BGKGN0GN10GN11GridDesc>,
            remove_reference_t<CGM10BM0BM1GN10BN0BN1GridDesc>,
            remove_reference_t<CBlockIdToGM10GN10BlockClusterAdaptor>,
            false,
            true>;

        ave_time = launch_and_time_kernel(kernel,
                                          nrepeat,
                                          dim3(grid_size),
                                          dim3(BlockSize),
                                          0,
                                          0,
                                          p_a_grid,
                                          p_b_grid,
                                          p_c_grid,
                                          a_gk_gm0_gm10_gm11_grid_desc,
                                          b_gk_gn0_gn10_gn11_grid_desc,
                                          c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc,
                                          c_blockid_to_gm10_gn10_block_cluster_adaptor);
    }
    else
    {
        const auto kernel = kernel_dynamic_contraction_v1r1<
            GridwiseContraction,
            FloatAB,
            FloatC,
            remove_reference_t<AGKGM0GM10GM11GridDesc>,
            remove_reference_t<BGKGN0GN10GN11GridDesc>,
            remove_reference_t<CGM10BM0BM1GN10BN0BN1GridDesc>,
            remove_reference_t<CBlockIdToGM10GN10BlockClusterAdaptor>,
            false,
            false>;

        ave_time = launch_and_time_kernel(kernel,
                                          nrepeat,
                                          dim3(grid_size),
                                          dim3(BlockSize),
                                          0,
                                          0,
                                          p_a_grid,
                                          p_b_grid,
                                          p_c_grid,
                                          a_gk_gm0_gm10_gm11_grid_desc,
                                          b_gk_gn0_gn10_gn11_grid_desc,
                                          c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc,
                                          c_blockid_to_gm10_gn10_block_cluster_adaptor);
    }

    return ave_time;
}

} // namespace ck
#endif
