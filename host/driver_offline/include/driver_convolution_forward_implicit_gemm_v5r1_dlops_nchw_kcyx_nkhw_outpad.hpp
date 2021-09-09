#ifndef DRIVER_CONVOLUTION_FORWARD_IMPLICIT_GEMM_V5R1_DLOPS_NCHW_KCYX_NKHW_OUTPAD_HPP
#define DRIVER_CONVOLUTION_FORWARD_IMPLICIT_GEMM_V5R1_DLOPS_NCHW_KCYX_NKHW_OUTPAD_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm_dlops_v2.hpp"

template <ck::index_t BlockSize,
          typename FloatAB,
          typename FloatAcc,
          typename FloatC,
          ck::index_t KPerBlock,
          ck::index_t HoPerBlock,
          ck::index_t WoPerBlock,
          ck::index_t EPerBlock,
          ck::index_t KPerThread,
          ck::index_t HoPerThread,
          ck::index_t WoPerThread,
          ck::index_t EPerThread,
          typename ABlockTransferThreadSliceLengths_E_K,
          typename ABlockTransferThreadClusterLengths_E_K,
          ck::index_t ABlockTransferSrcScalarPerVector_E,
          ck::index_t ABlockTransferDstScalarPerVector_K,
          ck::index_t BThreadTransferSrcScalarPerVector_E,
          ck::index_t CThreadTransferDstScalarPerVector_K>
struct DriverDynamicConvolutionForwardImplicitGemmDlops_v5r1_nchw_kcyx_nkhw_outpad
{
    template <typename... Wei,
              typename... In,
              typename... Out,
              typename ConvStrides,
              typename ConvDilations,
              typename InLeftPads,
              typename InRightPads>
    __host__ float Run(const ck::TensorDescriptor<Wei...>& wei_k_c0_y_x_c1_global_desc,
                       const ck::TensorDescriptor<In...>& in_n_c0_hi_wi_c1_global_desc,
                       const ck::TensorDescriptor<Out...>& out_n_k0_ho_wo_k1_global_desc,
                       const ConvStrides& conv_strides,
                       const ConvDilations& conv_dilations,
                       const InLeftPads& in_left_pads,
                       const InRightPads& in_right_pads,
                       const FloatAB* __restrict__ p_a_grid,
                       const FloatAB* __restrict__ p_b_grid,
                       FloatC* __restrict__ p_c_grid) const
    {
        using namespace ck;

        const int nrepeat = 100;

        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
        constexpr auto I4 = Number<4>{};

        const auto N  = in_n_c0_hi_wi_c1_global_desc.GetLength(I0);
        const auto C0 = in_n_c0_hi_wi_c1_global_desc.GetLength(I1);
        const auto Hi = in_n_c0_hi_wi_c1_global_desc.GetLength(I2);
        const auto Wi = in_n_c0_hi_wi_c1_global_desc.GetLength(I3);
        const auto C1 = in_n_c0_hi_wi_c1_global_desc.GetLength(I4);

        const auto K0 = out_n_k0_ho_wo_k1_global_desc.GetLength(I1);
        const auto Ho = out_n_k0_ho_wo_k1_global_desc.GetLength(I2);
        const auto Wo = out_n_k0_ho_wo_k1_global_desc.GetLength(I3);
        const auto K1 = out_n_k0_ho_wo_k1_global_desc.GetLength(I4);

        const auto K = wei_k_c0_y_x_c1_global_desc.GetLength(I0);
        const auto Y = wei_k_c0_y_x_c1_global_desc.GetLength(I2);
        const auto X = wei_k_c0_y_x_c1_global_desc.GetLength(I3);

        const auto ConvStrideH = conv_strides[I0];
        const auto ConvStrideW = conv_strides[I1];

        const auto ConvDilationH = conv_dilations[I0];
        const auto ConvDilationW = conv_dilations[I1];

        const auto Hop = (Ho + HoPerBlock - 1) / HoPerBlock * HoPerBlock;
        const auto Wop = (Wo + WoPerBlock - 1) / WoPerBlock * WoPerBlock;

        const auto OutRightPadH = Hop - Ho;
        const auto OutRightPadW = Wop - Wo;

        const auto InLeftPadH = in_left_pads[I0];
        const auto InLeftPadW = in_left_pads[I1];

        const auto InRightPadH = in_right_pads[I0] + OutRightPadH * ConvStrideH;
        const auto InRightPadW = in_right_pads[I1] + OutRightPadW * ConvStrideW;

        std::cerr << "OutRightPadH = " << OutRightPadH << " OutRightPadW = " << OutRightPadW
                  << std::endl;
        std::cerr << "InRightPadH = " << InRightPadH << " InRightPadW = " << InRightPadW
                  << std::endl;

        // weight tensor
        const auto a_e_k_grid_desc = transform_tensor_descriptor(
            make_naive_tensor_descriptor_packed(make_tuple(K, C0 * Y * X * C1)),
            make_tuple(make_pass_through_transform(K),
                       make_pass_through_transform(C0 * Y * X * C1)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<1>{}, Sequence<0>{}));

        // input tensor
        const auto in_n_c0_hip_wip_c1_global_desc = transform_tensor_descriptor(
            in_n_c0_hi_wi_c1_global_desc,
            make_tuple(make_pass_through_transform(N),
                       make_pass_through_transform(C0),
                       make_pad_transform(Hi, InLeftPadH, InRightPadH),
                       make_pad_transform(Wi, InLeftPadW, InRightPadW),
                       make_pass_through_transform(C1)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

        const auto in_n_c0_y_ho_x_wo_c1_global_desc = transform_tensor_descriptor(
            in_n_c0_hip_wip_c1_global_desc,
            make_tuple(
                make_pass_through_transform(N),
                make_pass_through_transform(C0),
                make_embed_transform(make_tuple(Y, Hop), make_tuple(ConvDilationH, ConvStrideH)),
                make_embed_transform(make_tuple(X, Wop), make_tuple(ConvDilationW, ConvStrideW)),
                make_pass_through_transform(C1)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
            make_tuple(
                Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}, Sequence<6>{}));

        const auto b_e_n_ho_wo_grid_desc = transform_tensor_descriptor(
            in_n_c0_y_ho_x_wo_c1_global_desc,
            make_tuple(make_merge_transform(make_tuple(C0, Y, X, C1)),
                       make_pass_through_transform(N),
                       make_pass_through_transform(Hop),
                       make_pass_through_transform(Wop)),
            make_tuple(Sequence<1, 2, 4, 6>{}, Sequence<0>{}, Sequence<3>{}, Sequence<5>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        // output tensor
        const auto c_k_n_hop_wop_grid_desc = transform_tensor_descriptor(
            make_naive_tensor_descriptor_packed(make_tuple(N, K0, Ho, Wo, K1)),
            make_tuple(make_merge_transform(make_tuple(K0, K1)),
                       make_pass_through_transform(N),
                       make_pad_transform(Ho, 0, OutRightPadH),
                       make_pad_transform(Wo, 0, OutRightPadW)),
            make_tuple(Sequence<1, 4>{}, Sequence<0>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        const auto E = C0 * Y * X * C1;

        std::cerr << "Hop = " << Hop << " Wop = " << Wop << std::endl;

        if(!((K % KPerBlock) == 0 && (Hop % HoPerBlock) == 0 && (Wop % WoPerBlock) == 0 &&
             (E % EPerBlock) == 0))
        {
            throw std::runtime_error("wrong! GEMM size no divisible");
        }

        // hack to control index calculation when iterating over a_k_m_global tensor
        constexpr auto a_e_k_global_step_hacks =
            make_tuple(make_tuple(Sequence<0, 0, 0>{}, Sequence<0, 0, 0>{}),
                       make_tuple(Sequence<0, 0, 0>{}, Sequence<0, 0, 0>{}));

        constexpr auto a_e_k_global_move_slice_window_step_hack = Sequence<0, 0, 0>{};

        constexpr auto b_e_n_ho_wo_global_step_hacks =
            make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}),
                       make_tuple(Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{}));

        constexpr auto b_e_n_ho_wo_global_move_slice_window_step_hack =
            Sequence<0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>{};

        // hack to control index calculation when iterating over c_m0_m1_n0_n1_global tensor
        // hack for NKHW format
        constexpr auto c_k_n_ho_wo_global_tensor_step_hacks =
            make_tuple(make_tuple(Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{}),
                       make_tuple(Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{},
                                  Sequence<0, 0, 0, 0, 0>{}));

        // GEMM
        using GridwiseGemm = GridwiseGemmDlops_km_kn_mn_v3<
            BlockSize,
            FloatAB,
            FloatAcc,
            FloatC,
            InMemoryDataOperationEnum_t::Set,
            decltype(a_e_k_grid_desc),
            decltype(b_e_n_ho_wo_grid_desc),
            decltype(c_k_n_hop_wop_grid_desc),
            KPerBlock,
            HoPerBlock,
            WoPerBlock,
            EPerBlock,
            KPerThread,
            HoPerThread,
            WoPerThread,
            EPerThread,
            ABlockTransferThreadSliceLengths_E_K,
            ABlockTransferThreadClusterLengths_E_K,
            Sequence<1, 0>,
            Sequence<1, 0>,
            0,
            ABlockTransferSrcScalarPerVector_E,
            ABlockTransferDstScalarPerVector_K,
            false, // don't move back src coordinate after threadwise copy
            Sequence<0, 2, 3, 1>,
            0,
            BThreadTransferSrcScalarPerVector_E,
            false, // don't move back src coordinate after threadwise copy, which will be fused with
                   // MoveSrcSliceWindow() to save addr computation
            Sequence<0, 2, 3, 1>,
            0,
            CThreadTransferDstScalarPerVector_K,
            decltype(a_e_k_global_step_hacks),
            decltype(b_e_n_ho_wo_global_step_hacks),
            decltype(c_k_n_ho_wo_global_tensor_step_hacks),
            decltype(a_e_k_global_move_slice_window_step_hack),
            decltype(b_e_n_ho_wo_global_move_slice_window_step_hack)>;

        using AEKGridDesc       = decltype(a_e_k_grid_desc);
        using BENHoWoGridDesc   = decltype(b_e_n_ho_wo_grid_desc);
        using CKNHopWopGridDesc = decltype(c_k_n_hop_wop_grid_desc);

        const auto grid_size = (K / KPerBlock) * (Hop / HoPerBlock) * (Wop / WoPerBlock) * N;

        const bool has_main_k_block_loop = (E + EPerBlock) / (2 * EPerBlock) > 1;

        const bool has_double_tail_k_block_loop = (E / EPerBlock) % 2 == 0;

        const auto c_blockid_to_k_n_ho_wo_block_cluster_adaptor =
            make_single_stage_tensor_adaptor(make_tuple(make_merge_transform(make_tuple(I0, I0))),
                                             make_tuple(Sequence<0, 1>{}),
                                             make_tuple(Sequence<0>{}));

        using CBlockIdToKNHopWopBlockClusterAdaptor =
            decltype(c_blockid_to_k_n_ho_wo_block_cluster_adaptor);

#if CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VALUE
        float ave_time = 0;

        if(has_main_k_block_loop && has_double_tail_k_block_loop)
        {
            const auto kernel =
                kernel_gemm_dlops_v2<GridwiseGemm,
                                     FloatAB,
                                     FloatC,
                                     remove_reference_t<AEKGridDesc>,
                                     remove_reference_t<BENHoWoGridDesc>,
                                     remove_reference_t<CKNHopWopGridDesc>,
                                     remove_reference_t<CBlockIdToKNHopWopBlockClusterAdaptor>,
                                     true,
                                     true>;

            ave_time = launch_and_time_kernel(kernel,
                                              nrepeat,
                                              dim3(grid_size),
                                              dim3(BlockSize),
                                              0,
                                              p_a_grid,
                                              p_b_grid,
                                              p_c_grid,
                                              a_e_k_grid_desc,
                                              b_e_n_ho_wo_grid_desc,
                                              c_k_n_hop_wop_grid_desc,
                                              c_blockid_to_k_n_ho_wo_block_cluster_adaptor);
        }
        else if(has_main_k_block_loop && !has_double_tail_k_block_loop)
        {
            const auto kernel =
                kernel_gemm_dlops_v2<GridwiseGemm,
                                     FloatAB,
                                     FloatC,
                                     remove_reference_t<AEKGridDesc>,
                                     remove_reference_t<BENHoWoGridDesc>,
                                     remove_reference_t<CKNHopWopGridDesc>,
                                     remove_reference_t<CBlockIdToKNHopWopBlockClusterAdaptor>,
                                     true,
                                     false>;

            ave_time = launch_and_time_kernel(kernel,
                                              nrepeat,
                                              dim3(grid_size),
                                              dim3(BlockSize),
                                              0,
                                              p_a_grid,
                                              p_b_grid,
                                              p_c_grid,
                                              a_e_k_grid_desc,
                                              b_e_n_ho_wo_grid_desc,
                                              c_k_n_hop_wop_grid_desc,
                                              c_blockid_to_k_n_ho_wo_block_cluster_adaptor);
        }
        else if(!has_main_k_block_loop && has_double_tail_k_block_loop)
        {
            const auto kernel =
                kernel_gemm_dlops_v2<GridwiseGemm,
                                     FloatAB,
                                     FloatC,
                                     remove_reference_t<AEKGridDesc>,
                                     remove_reference_t<BENHoWoGridDesc>,
                                     remove_reference_t<CKNHopWopGridDesc>,
                                     remove_reference_t<CBlockIdToKNHopWopBlockClusterAdaptor>,
                                     false,
                                     true>;

            ave_time = launch_and_time_kernel(kernel,
                                              nrepeat,
                                              dim3(grid_size),
                                              dim3(BlockSize),
                                              0,
                                              p_a_grid,
                                              p_b_grid,
                                              p_c_grid,
                                              a_e_k_grid_desc,
                                              b_e_n_ho_wo_grid_desc,
                                              c_k_n_hop_wop_grid_desc,
                                              c_blockid_to_k_n_ho_wo_block_cluster_adaptor);
        }
        else
        {
            const auto kernel =
                kernel_gemm_dlops_v2<GridwiseGemm,
                                     FloatAB,
                                     FloatC,
                                     remove_reference_t<AEKGridDesc>,
                                     remove_reference_t<BENHoWoGridDesc>,
                                     remove_reference_t<CKNHopWopGridDesc>,
                                     remove_reference_t<CBlockIdToKNHopWopBlockClusterAdaptor>,
                                     false,
                                     false>;

            ave_time = launch_and_time_kernel(kernel,
                                              nrepeat,
                                              dim3(grid_size),
                                              dim3(BlockSize),
                                              0,
                                              p_a_grid,
                                              p_b_grid,
                                              p_c_grid,
                                              a_e_k_grid_desc,
                                              b_e_n_ho_wo_grid_desc,
                                              c_k_n_hop_wop_grid_desc,
                                              c_blockid_to_k_n_ho_wo_block_cluster_adaptor);
        }

        return ave_time;
#elif CK_EXPERIMENTAL_PASS_TENSOR_DESCRIPTOR_BY_VOID_POINTER
        DeviceMem a_e_k_grid_desc_dev_buf(sizeof(AEKGridDesc));
        DeviceMem b_e_n_ho_wo_grid_desc_dev_buf(sizeof(BENHoWoGridDesc));
        DeviceMem c_k_n_hop_wop_grid_desc_dev_buf(sizeof(CKNHopWopGridDesc));
        DeviceMem c_blockid_to_k_n_ho_wo_block_cluster_adaptor_dev_buf(
            sizeof(CBlockIdToKNHopWopBlockClusterAdaptor));

        a_e_k_grid_desc_dev_buf.ToDevice(&a_e_k_grid_desc);
        b_e_n_ho_wo_grid_desc_dev_buf.ToDevice(&b_e_n_ho_wo_grid_desc);
        c_k_n_hop_wop_grid_desc_dev_buf.ToDevice(&c_k_n_hop_wop_grid_desc);
        c_blockid_to_k_n_ho_wo_block_cluster_adaptor_dev_buf.ToDevice(
            &c_blockid_to_k_n_ho_wo_block_cluster_adaptor);

        float ave_time = 0;

        if(has_main_k_block_loop && has_double_tail_k_block_loop)
        {
            const auto kernel =
                kernel_gemm_dlops_v2<GridwiseGemm,
                                     FloatAB,
                                     FloatC,
                                     remove_reference_t<AEKGridDesc>,
                                     remove_reference_t<BENHoWoGridDesc>,
                                     remove_reference_t<CKNHopWopGridDesc>,
                                     remove_reference_t<CBlockIdToKNHopWopBlockClusterAdaptor>,
                                     true,
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
                cast_pointer_to_constant_address_space(a_e_k_grid_desc_dev_buf.GetDeviceBuffer()),
                cast_pointer_to_constant_address_space(
                    b_e_n_ho_wo_grid_desc_dev_buf.GetDeviceBuffer()),
                cast_pointer_to_constant_address_space(
                    c_k_n_hop_wop_grid_desc_dev_buf.GetDeviceBuffer()),
                cast_pointer_to_constant_address_space(
                    c_blockid_to_k_n_ho_wo_block_cluster_adaptor_dev_buf.GetDeviceBuffer()));
        }
        else if(has_main_k_block_loop && !has_double_tail_k_block_loop)
        {
            const auto kernel =
                kernel_gemm_dlops_v2<GridwiseGemm,
                                     FloatAB,
                                     FloatC,
                                     remove_reference_t<AEKGridDesc>,
                                     remove_reference_t<BENHoWoGridDesc>,
                                     remove_reference_t<CKNHopWopGridDesc>,
                                     remove_reference_t<CBlockIdToKNHopWopBlockClusterAdaptor>,
                                     true,
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
                cast_pointer_to_constant_address_space(a_e_k_grid_desc_dev_buf.GetDeviceBuffer()),
                cast_pointer_to_constant_address_space(
                    b_e_n_ho_wo_grid_desc_dev_buf.GetDeviceBuffer()),
                cast_pointer_to_constant_address_space(
                    c_k_n_hop_wop_grid_desc_dev_buf.GetDeviceBuffer()),
                cast_pointer_to_constant_address_space(
                    c_blockid_to_k_n_ho_wo_block_cluster_adaptor_dev_buf.GetDeviceBuffer()));
        }
        else if(!has_main_k_block_loop && has_double_tail_k_block_loop)
        {
            const auto kernel =
                kernel_gemm_dlops_v2<GridwiseGemm,
                                     FloatAB,
                                     FloatC,
                                     remove_reference_t<AEKGridDesc>,
                                     remove_reference_t<BENHoWoGridDesc>,
                                     remove_reference_t<CKNHopWopGridDesc>,
                                     remove_reference_t<CBlockIdToKNHopWopBlockClusterAdaptor>,
                                     false,
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
                cast_pointer_to_constant_address_space(a_e_k_grid_desc_dev_buf.GetDeviceBuffer()),
                cast_pointer_to_constant_address_space(
                    b_e_n_ho_wo_grid_desc_dev_buf.GetDeviceBuffer()),
                cast_pointer_to_constant_address_space(
                    c_k_n_hop_wop_grid_desc_dev_buf.GetDeviceBuffer()),
                cast_pointer_to_constant_address_space(
                    c_blockid_to_k_n_ho_wo_block_cluster_adaptor_dev_buf.GetDeviceBuffer()));
        }
        else
        {
            const auto kernel =
                kernel_gemm_dlops_v2<GridwiseGemm,
                                     FloatAB,
                                     FloatC,
                                     remove_reference_t<AEKGridDesc>,
                                     remove_reference_t<BENHoWoGridDesc>,
                                     remove_reference_t<CKNHopWopGridDesc>,
                                     remove_reference_t<CBlockIdToKNHopWopBlockClusterAdaptor>,
                                     false,
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
                cast_pointer_to_constant_address_space(a_e_k_grid_desc_dev_buf.GetDeviceBuffer()),
                cast_pointer_to_constant_address_space(
                    b_e_n_ho_wo_grid_desc_dev_buf.GetDeviceBuffer()),
                cast_pointer_to_constant_address_space(
                    c_k_n_hop_wop_grid_desc_dev_buf.GetDeviceBuffer()),
                cast_pointer_to_constant_address_space(
                    c_blockid_to_k_n_ho_wo_block_cluster_adaptor_dev_buf.GetDeviceBuffer()));
        }

        return ave_time;
#endif
    }
};
#endif
