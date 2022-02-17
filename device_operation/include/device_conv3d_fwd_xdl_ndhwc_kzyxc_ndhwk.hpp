#ifndef DEVICE_CONV3D_FWD_XDL_HPP
#define DEVICE_CONV3D_FWD_XDL_HPP

#include <iostream>
#include <memory>
#include <sstream>
#include "device.hpp"
#include "device_conv_fwd.hpp"
#include "common_header.hpp"
#include "tensor_layout.hpp"
#include "convolution_forward_specialization.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "transform_forward_convolution3d_into_gemm_v4r4r4_ndhwc_kzyxc_ndhwk.hpp"
#include "gridwise_gemm_xdlops_v2r3.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// specialization for #D conv: in[n, di, hi, wi, c] * wei[k, z, y, x, c] = out[n, do, ho, wo, k]
template <typename InDataType,
          typename WeiDataType, // WeiDataType must be the same as InDataType
          typename OutDataType,
          typename AccDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          ConvolutionForwardSpecialization_t ConvForwardSpecialization,
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
          ck::index_t CThreadTransferDstScalarPerVector>
struct DeviceConv3dFwdXdl_Input_N_Di_Hi_Wi_C_Weight_K_Z_Y_X_C_Output_N_Do_Ho_Wo_K
    : public DeviceConvFwd<InElementwiseOperation, WeiElementwiseOperation, OutElementwiseOperation>

{
    using DeviceOp = DeviceConv3dFwdXdl_Input_N_Di_Hi_Wi_C_Weight_K_Z_Y_X_C_Output_N_Do_Ho_Wo_K;

    using ADataType = InDataType;
    using BDataType = WeiDataType;
    using CDataType = OutDataType;
    // TODO make A/B datatype different
    using ABDataType = InDataType;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    static index_t GetMaxAllowableSubBatchSize(const index_t N,
                                               const index_t K,
                                               const index_t C,
                                               std::vector<ck::index_t> input_spatial_lengths,
                                               std::vector<ck::index_t> output_spatial_lengths)
    {
        const index_t Di = input_spatial_lengths[0];
        const index_t Hi = input_spatial_lengths[1];
        const index_t Wi = input_spatial_lengths[2];

        const index_t Do = output_spatial_lengths[0];
        const index_t Ho = output_spatial_lengths[1];
        const index_t Wo = output_spatial_lengths[2];

        // N1 should satisfy that
        //   1) N % N1 = 0;
        //   2) N1 * (Do * Ho * Wo * K) < (2^31 - 1)
        //   3) N1 * (Di * Hi * Wi * C) < (2^31 - 1)
        //
        // Do NOT confuse (B, N1) in this function with (B, N1) in gridewise GEMM.
        auto N1 = N + 1;

        const auto stride =
            math::max(long_index_t(Do) * Ho * Wo * K, long_index_t(Di) * Hi * Wi * C);
        const index_t max_stride = NumericLimits<index_t>::Max();

        for(index_t n0 = 1; n0 <= N; ++n0)
        {
            index_t n1 = N / n0;
            if(n0 * n1 == N && long_index_t(n1) * long_index_t(stride) < max_stride)
            {
                N1 = n1;
                break;
            }
        }

        const auto B = N / N1;
        if(B * N1 != N)
        {
            throw std::runtime_error(__func__ +
                                     std::string(": failed to find num_subbatches for conv3d.\n"));
        }

        return N1;
    }

    static auto
    MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(const index_t N,
                                                    const index_t K,
                                                    const index_t C,
                                                    std::vector<ck::index_t> input_spatial_lengths,
                                                    std::vector<ck::index_t> filter_spatial_lengths,
                                                    std::vector<ck::index_t> output_spatial_lengths,
                                                    std::vector<ck::index_t> conv_filter_strides,
                                                    std::vector<ck::index_t> conv_filter_dilations,
                                                    std::vector<ck::index_t> input_left_pads,
                                                    std::vector<ck::index_t> input_right_pads)
    {
        assert(input_spatial_lengths.size() > 2);
        assert(filter_spatial_lengths.size() > 2);
        assert(conv_filter_strides.size() > 2);
        assert(conv_filter_dilations.size() > 2);
        assert(input_left_pads.size() > 2);
        assert(input_right_pads.size() > 2);

        const index_t Di = input_spatial_lengths[0];
        const index_t Hi = input_spatial_lengths[1];
        const index_t Wi = input_spatial_lengths[2];
        const index_t Z  = filter_spatial_lengths[0];
        const index_t Y  = filter_spatial_lengths[1];
        const index_t X  = filter_spatial_lengths[2];

        const index_t Do = output_spatial_lengths[0];
        const index_t Ho = output_spatial_lengths[1];
        const index_t Wo = output_spatial_lengths[2];

        if constexpr(ConvForwardSpecialization ==
                     ConvolutionForwardSpecialization_t::Filter1x1Stride1Pad0)
        {
            static_assert(ConvForwardSpecialization == -1, "Not implemented!");
        }
        else if constexpr(ConvForwardSpecialization ==
                          ConvolutionForwardSpecialization_t::Filter1x1Pad0)
        {

            static_assert(ConvForwardSpecialization == -1, "Not implemented!");
        }
        else
        {
            const index_t subbatch_size =
                GetMaxAllowableSubBatchSize(N, K, C, input_spatial_lengths, output_spatial_lengths);
            printf("subbatch_size = %d\n", subbatch_size);

            const auto in_desc_n_di_hi_wi_c =
                make_naive_tensor_descriptor_packed(make_tuple(subbatch_size, Di, Hi, Wi, C));
            const auto wei_desc_k_z_y_x_c =
                make_naive_tensor_descriptor_packed(make_tuple(K, Z, Y, X, C));
            const auto out_desc_n_do_ho_wo_k =
                make_naive_tensor_descriptor_packed(make_tuple(subbatch_size, Do, Ho, Wo, K));

            const auto descs =
                transform_forward_convolution3d_into_gemm_v4r4r4_nhwc_kyxc_nhwk_pad(
                    in_desc_n_di_hi_wi_c,
                    wei_desc_k_z_y_x_c,
                    out_desc_n_do_ho_wo_k,
                    make_tuple(
                        conv_filter_strides[0], conv_filter_strides[1], conv_filter_strides[2]),
                    make_tuple(conv_filter_dilations[0],
                               conv_filter_dilations[1],
                               conv_filter_dilations[2]),
                    make_tuple(input_left_pads[0], input_left_pads[1], input_left_pads[2]),
                    make_tuple(input_right_pads[0], input_right_pads[1], input_right_pads[2]),
                    Number<K1>{});

            return descs;
        }
    }

    using ABCGridDescs = remove_cvref_t<decltype(MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(
        1, 1, 1, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}))>;

    using AGridDesc_K0_M_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I0])>;
    using BGridDesc_K0_N_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I1])>;
    using CGridDesc_M_N   = remove_cvref_t<decltype(ABCGridDescs{}[I2])>;

    using GridwiseGemm = GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3<
        BlockSize,
        InDataType,
        AccDataType,
        OutDataType,
        InMemoryDataOperationEnum_t::Set,
        AGridDesc_K0_M_K1,
        BGridDesc_K0_N_K1,
        CGridDesc_M_N,
        InElementwiseOperation,
        WeiElementwiseOperation,
        OutElementwiseOperation,
        MPerBlock,
        NPerBlock,
        K0PerBlock,
        MPerXDL,
        NPerXDL,
        K1,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        Sequence<1, 0, 2>, // ABlockTransferThreadClusterArrangeOrder,
        Sequence<1, 0, 2>, // ABlockTransferSrcAccessOrder,
        2,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsAddExtraM,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        Sequence<1, 0, 2>, // ABlockTransferThreadClusterArrangeOrder,
        Sequence<1, 0, 2>, // ABlockTransferSrcAccessOrder,
        2,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false, // BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsAddExtraN,
        Sequence<2, 3, 0, 1, 7, 5, 4, 6>,
        7,
        CThreadTransferDstScalarPerVector>;

    using CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2 =
        decltype(GridwiseGemm::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(CGridDesc_M_N{}));
    using Block2CTileMap = decltype(GridwiseGemm::MakeBlock2CTileMap(CGridDesc_M_N{}, 1, 1));

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const InDataType* p_in,
                 const WeiDataType* p_wei,
                 OutDataType* p_out,
                 const index_t N,
                 const index_t K,
                 const index_t C,
                 std::vector<ck::index_t> input_spatial_lengths,
                 std::vector<ck::index_t> filter_spatial_lengths,
                 std::vector<ck::index_t> output_spatial_lengths,
                 std::vector<ck::index_t> conv_filter_strides,
                 std::vector<ck::index_t> conv_filter_dilations,
                 std::vector<ck::index_t> input_left_pads,
                 std::vector<ck::index_t> input_right_pads,
                 index_t M01,
                 index_t N01,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op)
            : p_a_grid_{p_in},
              p_b_grid_{p_wei},
              p_c_grid_{p_out},
              M01_{M01},
              N01_{N01},
              in_element_op_{in_element_op},
              wei_element_op_{wei_element_op},
              out_element_op_{out_element_op}
        {
            const index_t subbatch_size =
                GetMaxAllowableSubBatchSize(N, K, C, input_spatial_lengths, output_spatial_lengths);
            num_subbatches_ = N / subbatch_size;
            const auto descs =
                MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(subbatch_size,
                                                                K,
                                                                C,
                                                                input_spatial_lengths,
                                                                filter_spatial_lengths,
                                                                output_spatial_lengths,
                                                                conv_filter_strides,
                                                                conv_filter_dilations,
                                                                input_left_pads,
                                                                input_right_pads);

            a_grid_desc_k0_m_k1_ = descs[I0];
            b_grid_desc_k0_n_k1_ = descs[I1];
            c_grid_desc_m_n_     = descs[I2];

            a_batch_stride_ = a_grid_desc_k0_m_k1_.GetElementSize();
            b_batch_stride_ = b_grid_desc_k0_n_k1_.GetElementSize();
            c_batch_stride_ = c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_.GetElementSize();

            if(GridwiseGemm::CheckValidity(
                   a_grid_desc_k0_m_k1_, b_grid_desc_k0_n_k1_, c_grid_desc_m_n_, M01_, N01_))
            {
                c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_ =
                    GridwiseGemm::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(c_grid_desc_m_n_);

                block_2_ctile_map_ = GridwiseGemm::MakeBlock2CTileMap(c_grid_desc_m_n_, M01, N01);
            }
        }

        //  private:
        const InDataType* p_a_grid_;
        const WeiDataType* p_b_grid_;
        OutDataType* p_c_grid_;
        index_t num_subbatches_;
        index_t a_batch_stride_;
        index_t b_batch_stride_;
        index_t c_batch_stride_;
        AGridDesc_K0_M_K1 a_grid_desc_k0_m_k1_;
        BGridDesc_K0_N_K1 b_grid_desc_k0_n_k1_;
        CGridDesc_M_N c_grid_desc_m_n_;
        CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2 c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_;
        Block2CTileMap block_2_ctile_map_;
        index_t M01_;
        index_t N01_;
        InElementwiseOperation in_element_op_;
        WeiElementwiseOperation wei_element_op_;
        OutElementwiseOperation out_element_op_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, int nrepeat = 1)
        {
            {
                std::cout << "a_grid_desc_k0_m_k1{" << arg.a_grid_desc_k0_m_k1_.GetLength(I0)
                          << ", " << arg.a_grid_desc_k0_m_k1_.GetLength(I1) << ", "
                          << arg.a_grid_desc_k0_m_k1_.GetLength(I2) << "}" << std::endl;

                std::cout << "b_grid_desc_k0_n_k1{" << arg.b_grid_desc_k0_n_k1_.GetLength(I0)
                          << ", " << arg.b_grid_desc_k0_n_k1_.GetLength(I1) << ", "
                          << arg.b_grid_desc_k0_n_k1_.GetLength(I2) << "}" << std::endl;

                std::cout << "c_grid_desc_m_n{ " << arg.c_grid_desc_m_n_.GetLength(I0) << ", "
                          << arg.c_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;
            }

            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_k0_m_k1_,
                                            arg.b_grid_desc_k0_n_k1_,
                                            arg.c_grid_desc_m_n_,
                                            arg.M01_,
                                            arg.N01_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseBatchedGemm_bk0mk1_k0nk1_bmn_xdlops_v2r3r2_xdlops_v2r3 has "
                    "invalid setting");
            }

            const index_t grid_size = GridwiseGemm::CalculateGridSize(arg.c_grid_desc_m_n_);

            const auto K0 = arg.a_grid_desc_k0_m_k1_.GetLength(I0);

            const bool has_main_k0_block_loop = GridwiseGemm::CalculateHasMainK0BlockLoop(K0);

            float ave_time = 0;
            if(has_main_k0_block_loop)
            {
                const auto kernel = kernel_batched_gemm_xdlops_v2r3<
                    GridwiseGemm,
                    InDataType,
                    OutDataType,
                    remove_reference_t<AGridDesc_K0_M_K1>,
                    remove_reference_t<BGridDesc_K0_N_K1>,
                    remove_reference_t<CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2>,
                    InElementwiseOperation,
                    WeiElementwiseOperation,
                    OutElementwiseOperation,
                    remove_reference_t<Block2CTileMap>,
                    true>;
                ave_time = launch_and_time_kernel(kernel,
                                                  nrepeat,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.num_subbatches_,
                                                  arg.a_batch_stride_,
                                                  arg.b_batch_stride_,
                                                  arg.c_batch_stride_,
                                                  arg.a_grid_desc_k0_m_k1_,
                                                  arg.b_grid_desc_k0_n_k1_,
                                                  arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_,
                                                  arg.in_element_op_,
                                                  arg.wei_element_op_,
                                                  arg.out_element_op_,
                                                  arg.block_2_ctile_map_);
            }
            else
            {
                const auto kernel = kernel_batched_gemm_xdlops_v2r3<
                    GridwiseGemm,
                    InDataType,
                    OutDataType,
                    remove_reference_t<AGridDesc_K0_M_K1>,
                    remove_reference_t<BGridDesc_K0_N_K1>,
                    remove_reference_t<CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2>,
                    InElementwiseOperation,
                    WeiElementwiseOperation,
                    OutElementwiseOperation,
                    remove_reference_t<Block2CTileMap>,
                    false>;

                ave_time = launch_and_time_kernel(kernel,
                                                  nrepeat,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.num_subbatches_,
                                                  arg.a_batch_stride_,
                                                  arg.b_batch_stride_,
                                                  arg.c_batch_stride_,
                                                  arg.a_grid_desc_k0_m_k1_,
                                                  arg.b_grid_desc_k0_n_k1_,
                                                  arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_,
                                                  arg.in_element_op_,
                                                  arg.wei_element_op_,
                                                  arg.out_element_op_,
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

    static auto MakeArgument(const InDataType* p_in,
                             const WeiDataType* p_wei,
                             OutDataType* p_out,
                             const index_t N,
                             const index_t K,
                             const index_t C,
                             std::vector<ck::index_t> input_spatial_lengths,
                             std::vector<ck::index_t> filter_spatial_lengths,
                             std::vector<ck::index_t> output_spatial_lengths,
                             std::vector<ck::index_t> conv_filter_strides,
                             std::vector<ck::index_t> conv_filter_dilations,
                             std::vector<ck::index_t> input_left_pads,
                             std::vector<ck::index_t> input_right_pads,
                             InElementwiseOperation in_element_op,
                             WeiElementwiseOperation wei_element_op,
                             OutElementwiseOperation out_element_op)
    {
        return Argument{p_in,
                        p_wei,
                        p_out,
                        N,
                        K,
                        C,
                        input_spatial_lengths,
                        filter_spatial_lengths,
                        output_spatial_lengths,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        1,
                        1,
                        in_element_op,
                        wei_element_op,
                        out_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in,
                        const void* p_wei,
                        void* p_out,
                        const index_t N,
                        const index_t K,
                        const index_t C,
                        std::vector<ck::index_t> input_spatial_lengths,
                        std::vector<ck::index_t> filter_spatial_lengths,
                        std::vector<ck::index_t> output_spatial_lengths,
                        std::vector<ck::index_t> conv_filter_strides,
                        std::vector<ck::index_t> conv_filter_dilations,
                        std::vector<ck::index_t> input_left_pads,
                        std::vector<ck::index_t> input_right_pads,
                        InElementwiseOperation in_element_op,
                        WeiElementwiseOperation wei_element_op,
                        OutElementwiseOperation out_element_op) override

    {
        return std::make_unique<Argument>(static_cast<const InDataType*>(p_in),
                                          static_cast<const WeiDataType*>(p_wei),
                                          static_cast<OutDataType*>(p_out),
                                          N,
                                          K,
                                          C,
                                          input_spatial_lengths,
                                          filter_spatial_lengths,
                                          output_spatial_lengths,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          1,
                                          1,
                                          in_element_op,
                                          wei_element_op,
                                          out_element_op);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceConv3dFwdXdl_Input_N_Di_Hi_Wi_C_Weight_K_Z_Y_X_C_Output_N_Do_Ho_Wo_K"
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

