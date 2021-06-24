#include "device.hpp"
#include "host_tensor.hpp"
#include "dynamic_tensor_descriptor.hpp"
#include "dynamic_tensor_descriptor_helper.hpp"
#include "transform_forward_convolution_into_gemm_v4r5_nchw_kcyx_nkhw.hpp"

#include "olc_driver_common.hpp"
#include "conv_tunables.hpp"

#include "handle.hpp"

namespace detail_dyn_conv_fwd_v4r5_nchw_kcyx_nkhw {

template <typename TInWei, typename TAcc, typename TOut>
static std::string get_network_config_string_from_types()
{
    std::string out;

    out += static_cast<char>(Driver::get_typeid_from_type<TInWei>()) +
           static_cast<char>(Driver::get_typeid_from_type<TAcc>()) +
           static_cast<char>(Driver::get_typeid_from_type<TOut>());

    return (out);
};

static std::string
get_network_config_string_from_tunable(const tunable_dyn_conv_fwd_v4r5_nchw_kcyx_nkhw* pt)
{
    std::string out("TUN_");

    out += std::to_string(pt->BlockSize) + "_";

    out += std::to_string(pt->GM1PerBlockGM11) + "x" + std::to_string(pt->GN1PerBlockGN11) + "x" +
           std::to_string(pt->KPerBlock) + "_";
    out += std::to_string(pt->M1PerThread) + "x" + std::to_string(pt->N1PerThread) + "x" +
           std::to_string(pt->KPerThread) + "_";
    out += std::to_string(pt->M1N1ThreadClusterM10) + "x" +
           std::to_string(pt->M1N1ThreadClusterN10) + "x" +
           std::to_string(pt->M1N1ThreadClusterM11) + "x" +
           std::to_string(pt->M1N1ThreadClusterN11) + "_";

    out += std::to_string(pt->ABlockTransferThreadSliceLengths_GK_GM0_GM10_GM11[0]) + "x" +
           std::to_string(pt->ABlockTransferThreadSliceLengths_GK_GM0_GM10_GM11[1]) + "x" +
           std::to_string(pt->ABlockTransferThreadSliceLengths_GK_GM0_GM10_GM11[2]) + "x" +
           std::to_string(pt->ABlockTransferThreadSliceLengths_GK_GM0_GM10_GM11[3]) + "_";

    out += std::to_string(pt->ABlockTransferThreadClusterLengths_GK_GM0_GM10_GM11[0]) + "x" +
           std::to_string(pt->ABlockTransferThreadClusterLengths_GK_GM0_GM10_GM11[1]) + "x" +
           std::to_string(pt->ABlockTransferThreadClusterLengths_GK_GM0_GM10_GM11[2]) + "x" +
           std::to_string(pt->ABlockTransferThreadClusterLengths_GK_GM0_GM10_GM11[3]) + "_";

    out += std::to_string(pt->ABlockTransferThreadClusterArrangeOrder[0]) + "x" +
           std::to_string(pt->ABlockTransferThreadClusterArrangeOrder[1]) + "x" +
           std::to_string(pt->ABlockTransferThreadClusterArrangeOrder[2]) + "x" +
           std::to_string(pt->ABlockTransferThreadClusterArrangeOrder[3]) + "_";

    out += std::to_string(pt->ABlockTransferSrcAccessOrder[0]) + "x" +
           std::to_string(pt->ABlockTransferSrcAccessOrder[1]) + "x" +
           std::to_string(pt->ABlockTransferSrcAccessOrder[2]) + "x" +
           std::to_string(pt->ABlockTransferSrcAccessOrder[3]) + "_";

    out += std::to_string(pt->ABlockTransferSrcVectorDim) + "_";
    out += std::to_string(pt->ABlockTransferSrcScalarPerVector) + "_";
    out += std::to_string(pt->ABlockTransferDstScalarPerVector_GM11) + "_";
    out += std::to_string(pt->AThreadTransferSrcResetCoordinateAfterRun) + "_";

    out += std::to_string(pt->BBlockTransferThreadSliceLengths_GK_GN0_GN10_GN11[0]) + "x" +
           std::to_string(pt->BBlockTransferThreadSliceLengths_GK_GN0_GN10_GN11[1]) + "x" +
           std::to_string(pt->BBlockTransferThreadSliceLengths_GK_GN0_GN10_GN11[2]) + "x" +
           std::to_string(pt->BBlockTransferThreadSliceLengths_GK_GN0_GN10_GN11[3]);

    out += std::to_string(pt->BBlockTransferThreadClusterLengths_GK_GN0_GN10_GN11[0]) + "x" +
           std::to_string(pt->BBlockTransferThreadClusterLengths_GK_GN0_GN10_GN11[1]) + "x" +
           std::to_string(pt->BBlockTransferThreadClusterLengths_GK_GN0_GN10_GN11[2]) + "x" +
           std::to_string(pt->BBlockTransferThreadClusterLengths_GK_GN0_GN10_GN11[3]) + "_";

    out += std::to_string(pt->BBlockTransferThreadClusterArrangeOrder[0]) + "x" +
           std::to_string(pt->BBlockTransferThreadClusterArrangeOrder[1]) + "x" +
           std::to_string(pt->BBlockTransferThreadClusterArrangeOrder[2]) + "x" +
           std::to_string(pt->BBlockTransferThreadClusterArrangeOrder[3]) + "_";

    out += std::to_string(pt->BBlockTransferSrcAccessOrder[0]) + "x" +
           std::to_string(pt->BBlockTransferSrcAccessOrder[1]) + "x" +
           std::to_string(pt->BBlockTransferSrcAccessOrder[2]) + "x" +
           std::to_string(pt->BBlockTransferSrcAccessOrder[3]) + "_";

    out += std::to_string(pt->BBlockTransferSrcVectorDim) + "_";
    out += std::to_string(pt->BBlockTransferSrcScalarPerVector) + "_";
    out += std::to_string(pt->BBlockTransferDstScalarPerVector_GN11) + "_";
    out += std::to_string(pt->BThreadTransferSrcResetCoordinateAfterRun) + "_";

    out += std::to_string(pt->CThreadTransferSrcDstAccessOrder[0]) + "x" +
           std::to_string(pt->CThreadTransferSrcDstAccessOrder[1]) + "x" +
           std::to_string(pt->CThreadTransferSrcDstAccessOrder[2]) + "x" +
           std::to_string(pt->CThreadTransferSrcDstAccessOrder[3]) + "x" +
           std::to_string(pt->CThreadTransferSrcDstAccessOrder[4]) + "x" +
           std::to_string(pt->CThreadTransferSrcDstAccessOrder[5]) + "_";

    out += std::to_string(pt->CThreadTransferSrcDstVectorDim) + "_";
    out += std::to_string(pt->CThreadTransferDstScalarPerVector);

    return (out);
};

template <typename TInWei, typename TAcc, typename TOut>
static std::string get_definition_string_from_types()
{
    std::string out;

    out += " -DCK_PARAM_IN_WEI_DATATYPE=" + std::to_string(Driver::get_typeid_from_type<TInWei>()) +
           " -DCK_PARAM_CONV_COMPTYPE=" + std::to_string(Driver::get_typeid_from_type<TAcc>()) +
           " -DCK_PARAM_OUT_DATATYPE=" + std::to_string(Driver::get_typeid_from_type<TOut>());

    return (out);
};

static std::string
get_definition_string_from_tunable(const tunable_dyn_conv_fwd_v4r5_nchw_kcyx_nkhw* pt)
{
    std::string out;

    out += " -DCK_PARAM_BlockSize=" + std::to_string(pt->BlockSize);

    out += " -DCK_PARAM_GM1PerBlockGM11=" + std::to_string(pt->GM1PerBlockGM11) +
           " -DCK_PARAM_GN1PerBlockGN11=" + std::to_string(pt->GN1PerBlockGN11) +
           " -DCK_PARAM_KPerBlock=" + std::to_string(pt->KPerBlock);
    out += " -DCK_PARAM_M1PerThread=" + std::to_string(pt->M1PerThread) +
           " -DCK_PARAM_N1PerThread=" + std::to_string(pt->N1PerThread) +
           " -DCK_PARAM_KPerThread=" + std::to_string(pt->KPerThread);

    out += " -DCK_PARAM_M1N1ThreadClusterM10=" + std::to_string(pt->M1N1ThreadClusterM10) +
           " -DCK_PARAM_M1N1ThreadClusterN10=" + std::to_string(pt->M1N1ThreadClusterN10) +
           " -DCK_PARAM_M1N1ThreadClusterM11=" + std::to_string(pt->M1N1ThreadClusterM11) +
           " -DCK_PARAM_M1N1ThreadClusterN11=" + std::to_string(pt->M1N1ThreadClusterN11);

    out += " -DCK_PARAM_ABlockTransferThreadSliceLengths_GK_GM0_GM10_GM11=" +
           std::to_string(pt->ABlockTransferThreadSliceLengths_GK_GM0_GM10_GM11[0]) + "," +
           std::to_string(pt->ABlockTransferThreadSliceLengths_GK_GM0_GM10_GM11[1]) + "," +
           std::to_string(pt->ABlockTransferThreadSliceLengths_GK_GM0_GM10_GM11[2]) + "," +
           std::to_string(pt->ABlockTransferThreadSliceLengths_GK_GM0_GM10_GM11[3]);

    out += " -DCK_PARAM_ABlockTransferThreadClusterLengths_GK_GM0_GM10_GM11=" +
           std::to_string(pt->ABlockTransferThreadClusterLengths_GK_GM0_GM10_GM11[0]) + "," +
           std::to_string(pt->ABlockTransferThreadClusterLengths_GK_GM0_GM10_GM11[1]) + "," +
           std::to_string(pt->ABlockTransferThreadClusterLengths_GK_GM0_GM10_GM11[2]) + "," +
           std::to_string(pt->ABlockTransferThreadClusterLengths_GK_GM0_GM10_GM11[3]);

    out += " -DCK_PARAM_ABlockTransferThreadClusterArrangeOrder=" +
           std::to_string(pt->ABlockTransferThreadClusterArrangeOrder[0]) + "," +
           std::to_string(pt->ABlockTransferThreadClusterArrangeOrder[1]) + "," +
           std::to_string(pt->ABlockTransferThreadClusterArrangeOrder[2]) + "," +
           std::to_string(pt->ABlockTransferThreadClusterArrangeOrder[3]);

    out += " -DCK_PARAM_ABlockTransferSrcAccessOrder=" +
           std::to_string(pt->ABlockTransferSrcAccessOrder[0]) + "," +
           std::to_string(pt->ABlockTransferSrcAccessOrder[1]) + "," +
           std::to_string(pt->ABlockTransferSrcAccessOrder[2]) + "," +
           std::to_string(pt->ABlockTransferSrcAccessOrder[3]);

    out +=
        " -DCK_PARAM_ABlockTransferSrcVectorDim=" + std::to_string(pt->ABlockTransferSrcVectorDim);
    out += " -DCK_PARAM_ABlockTransferSrcScalarPerVector=" +
           std::to_string(pt->ABlockTransferSrcScalarPerVector);
    out += " -DCK_PARAM_ABlockTransferDstScalarPerVector_GM11=" +
           std::to_string(pt->ABlockTransferDstScalarPerVector_GM11);
    out += " -DCK_PARAM_AThreadTransferSrcResetCoordinateAfterRun=" +
           std::to_string(pt->AThreadTransferSrcResetCoordinateAfterRun);

    out += " -DCK_PARAM_BBlockTransferThreadSliceLengths_GK_GN0_GN10_GN11=" +
           std::to_string(pt->BBlockTransferThreadSliceLengths_GK_GN0_GN10_GN11[0]) + "," +
           std::to_string(pt->BBlockTransferThreadSliceLengths_GK_GN0_GN10_GN11[1]) + "," +
           std::to_string(pt->BBlockTransferThreadSliceLengths_GK_GN0_GN10_GN11[2]) + "," +
           std::to_string(pt->BBlockTransferThreadSliceLengths_GK_GN0_GN10_GN11[3]);

    out += " -DCK_PARAM_BBlockTransferThreadClusterLengths_GK_GN0_GN10_GN11=" +
           std::to_string(pt->BBlockTransferThreadClusterLengths_GK_GN0_GN10_GN11[0]) + "," +
           std::to_string(pt->BBlockTransferThreadClusterLengths_GK_GN0_GN10_GN11[1]) + "," +
           std::to_string(pt->BBlockTransferThreadClusterLengths_GK_GN0_GN10_GN11[2]) + "," +
           std::to_string(pt->BBlockTransferThreadClusterLengths_GK_GN0_GN10_GN11[3]);

    out += " -DCK_PARAM_BBlockTransferThreadClusterArrangeOrder=" +
           std::to_string(pt->BBlockTransferThreadClusterArrangeOrder[0]) + "," +
           std::to_string(pt->BBlockTransferThreadClusterArrangeOrder[1]) + "," +
           std::to_string(pt->BBlockTransferThreadClusterArrangeOrder[2]) + "," +
           std::to_string(pt->BBlockTransferThreadClusterArrangeOrder[3]);

    out += " -DCK_PARAM_BBlockTransferSrcAccessOrder=" +
           std::to_string(pt->BBlockTransferSrcAccessOrder[0]) + "," +
           std::to_string(pt->BBlockTransferSrcAccessOrder[1]) + "," +
           std::to_string(pt->BBlockTransferSrcAccessOrder[2]) + "," +
           std::to_string(pt->BBlockTransferSrcAccessOrder[3]);

    out +=
        " -DCK_PARAM_BBlockTransferSrcVectorDim=" + std::to_string(pt->BBlockTransferSrcVectorDim);
    out += " -DCK_PARAM_BBlockTransferSrcScalarPerVector=" +
           std::to_string(pt->BBlockTransferSrcScalarPerVector);
    out += " -DCK_PARAM_BBlockTransferDstScalarPerVector_GN11=" +
           std::to_string(pt->BBlockTransferDstScalarPerVector_GN11);
    out += " -DCK_PARAM_BThreadTransferSrcResetCoordinateAfterRun=" +
           std::to_string(pt->BThreadTransferSrcResetCoordinateAfterRun);

    out += " -DCK_PARAM_CThreadTransferSrcDstAccessOrder=" +
           std::to_string(pt->CThreadTransferSrcDstAccessOrder[0]) + "," +
           std::to_string(pt->CThreadTransferSrcDstAccessOrder[1]) + "," +
           std::to_string(pt->CThreadTransferSrcDstAccessOrder[2]) + "," +
           std::to_string(pt->CThreadTransferSrcDstAccessOrder[3]) + "," +
           std::to_string(pt->CThreadTransferSrcDstAccessOrder[4]) + "," +
           std::to_string(pt->CThreadTransferSrcDstAccessOrder[5]);

    out += " -DCK_PARAM_CThreadTransferSrcDstVectorDim=" +
           std::to_string(pt->CThreadTransferSrcDstVectorDim);
    out += " -DCK_PARAM_CThreadTransferDstScalarPerVector=" +
           std::to_string(pt->CThreadTransferDstScalarPerVector);

    return (out);
};

} // namespace detail_dyn_conv_fwd_v4r5_nchw_kcyx_nkhw

template <typename TInWei,
          typename TAcc,
          typename TOut,
          typename InLengths,
          typename WeiLengths,
          typename OutLengths,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads>
void device_dynamic_convolution_forward_implicit_gemm_v4r5_nchw_kcyx_nkhw_olc(
    olCompile::Handle* handle,
    const InLengths& in_n_c_hi_wi_lengths,
    const WeiLengths& wei_k_c_y_x_lengths,
    const OutLengths& out_n_k_ho_wo_lengths,
    const ConvStrides& conv_strides,
    const ConvDilations& conv_dilations,
    const InLeftPads& in_left_pads,
    const InRightPads& in_right_pads,
    const Tensor<TInWei>& in_n_c_hi_wi,
    const Tensor<TInWei>& wei_k_c_y_x,
    Tensor<TOut>& out_n_k_ho_wo,
    const tunable_dyn_conv_fwd_v4r5_nchw_kcyx_nkhw* tunable,
    ck::index_t nrepeat)
{
    using namespace ck;
    using namespace detail_dyn_conv_fwd_v4r5_nchw_kcyx_nkhw;
    using size_t = std::size_t;

    constexpr index_t N0 = 4; // this could not be a tunable so far

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // The follow codes are only used for computing the grid_size, hasMainKBlockLoop,
    // hasDoubleTailKBlockLoop

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    const auto in_n_c_hi_wi_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(in_n_c_hi_wi_lengths);
    const auto wei_k_c_y_x_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(wei_k_c_y_x_lengths);
    const auto out_n_k_ho_wo_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(out_n_k_ho_wo_lengths);

    const auto descs = transform_forward_convolution_into_contraction_v4r5_nchw_kcyx_nkhw_pad<N0>(
        wei_k_c_y_x_desc,
        in_n_c_hi_wi_desc,
        out_n_k_ho_wo_desc,
        conv_strides,
        conv_dilations,
        in_left_pads,
        in_right_pads);

    const auto a_gk_gm0_gm1_grid_desc      = descs[I0];
    const auto c_gm0_gm1_gn0_gn1_grid_desc = descs[I2];

    const auto GM1 = c_gm0_gm1_gn0_gn1_grid_desc.GetLength(I1);
    const auto GN1 = c_gm0_gm1_gn0_gn1_grid_desc.GetLength(I3);
    const auto GK  = a_gk_gm0_gm1_grid_desc.GetLength(I0);

    const index_t grid_size = (GM1 / tunable->GM1PerBlockGM11) * (GN1 / tunable->GN1PerBlockGN11);
    const bool hasMainKBlockLoop       = ((GK + tunable->KPerBlock) / (2 * tunable->KPerBlock) > 1);
    const bool hasDoubleTailKBlockLoop = ((GK / tunable->KPerBlock) % 2 == 0);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////

    // these buffers are usually provided by the user application
    DeviceMem in_n_c_hi_wi_dev_buf(sizeof(TInWei) * in_n_c_hi_wi.mDesc.GetElementSpace());
    DeviceMem wei_k_c_y_x_dev_buf(sizeof(TInWei) * wei_k_c_y_x.mDesc.GetElementSpace());
    DeviceMem out_n_k_ho_wo_dev_buf(sizeof(TOut) * out_n_k_ho_wo.mDesc.GetElementSpace());

    in_n_c_hi_wi_dev_buf.ToDevice(in_n_c_hi_wi.mData.data());
    wei_k_c_y_x_dev_buf.ToDevice(wei_k_c_y_x.mData.data());
    out_n_k_ho_wo_dev_buf.ToDevice(out_n_k_ho_wo.mData.data());

    // these are workspace buffers that should be expressed to the user by the corresponding
    // workspace API
    DeviceMem workspace_buf(4096);

    void* a_gk_gm0_gm10_gm11_grid_desc_dev_buf = workspace_buf.GetDeviceBuffer();
    void* b_gk_gn0_gn10_gn11_grid_desc_dev_buf =
        static_cast<void*>(static_cast<unsigned char*>(workspace_buf.GetDeviceBuffer()) + 1024);
    void* c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc_dev_buf =
        static_cast<void*>(static_cast<unsigned char*>(workspace_buf.GetDeviceBuffer()) + 2048);
    void* c_blockid_to_gm10_gn10_block_cluster_adaptor_dev_buf =
        static_cast<void*>(static_cast<unsigned char*>(workspace_buf.GetDeviceBuffer()) + 3072);

    const std::vector<size_t> vld  = {static_cast<size_t>(tunable->BlockSize), 1, 1};
    const std::vector<size_t> vgd1 = {static_cast<size_t>(tunable->BlockSize), 1, 1};
    const std::vector<size_t> vgd2 = {static_cast<size_t>(grid_size * tunable->BlockSize), 1, 1};

    std::string program_name = "dynamic_convolution_forward_implicit_gemm_v4r5_nchw_kcyx_nkhw.cpp";
    std::string algo_name    = "implicit_gemm_conv_fwd_v4r4_nchw";

    std::string param = " -std=c++17 ";
    std::string network_config;

    param += get_definition_string_from_types<TInWei, TAcc, TOut>() +
             " -DCK_PARAM_HAS_MAIN_KBLOCK_LOOP=" + std::to_string(hasMainKBlockLoop) +
             " -DCK_PARAM_HAS_DOUBLE_TAIL_KBLOCK_LOOP=" + std::to_string(hasDoubleTailKBlockLoop) +
             " -DCK_PARAM_N0=" + std::to_string(N0) + " " +
             get_definition_string_from_tunable(tunable);
    network_config = get_network_config_string_from_types<TInWei, TAcc, TOut>() + "_V" +
                     std::to_string(hasDoubleTailKBlockLoop) + "_" + std::to_string(N0) + "_" +
                     get_network_config_string_from_tunable(tunable);

    std::vector<float> kernel1_times;
    std::vector<float> kernel2_times;

    for(index_t i = 0; i < nrepeat; ++i)
    {
        KernelTimer timer1, timer2;
        std::string kernel_name;

        kernel_name = "dynamic_convolution_forward_implicit_gemm_v4r5_nchw_kcyx_nkhw_prepare";
        auto network_config_1 = network_config + "_1";

        timer1.Start();
        handle->AddKernel(algo_name, network_config_1, program_name, kernel_name, vld, vgd1, param)(
            static_cast<index_t>(in_n_c_hi_wi_lengths[I0]),
            static_cast<index_t>(in_n_c_hi_wi_lengths[I1]),
            static_cast<index_t>(in_n_c_hi_wi_lengths[I2]),
            static_cast<index_t>(in_n_c_hi_wi_lengths[I3]),
            static_cast<index_t>(wei_k_c_y_x_lengths[I0]),
            static_cast<index_t>(wei_k_c_y_x_lengths[I2]),
            static_cast<index_t>(wei_k_c_y_x_lengths[I3]),
            conv_strides[I0],
            conv_strides[I1],
            conv_dilations[I0],
            conv_dilations[I1],
            in_left_pads[I0],
            in_left_pads[I1],
            in_right_pads[I0],
            in_right_pads[I1],
            a_gk_gm0_gm10_gm11_grid_desc_dev_buf,
            b_gk_gn0_gn10_gn11_grid_desc_dev_buf,
            c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc_dev_buf,
            c_blockid_to_gm10_gn10_block_cluster_adaptor_dev_buf);
        timer2.End();

        kernel_name           = "dynamic_convolution_forward_implicit_gemm_v4r5_nchw_kcyx_nkhw";
        auto network_config_2 = network_config + "_2";

        timer2.Start();
        handle->AddKernel(algo_name, network_config_2, program_name, kernel_name, vld, vgd2, param)(
            reinterpret_cast<const TInWei*>(wei_k_c_y_x_dev_buf.GetDeviceBuffer()),
            reinterpret_cast<const TInWei*>(in_n_c_hi_wi_dev_buf.GetDeviceBuffer()),
            reinterpret_cast<TOut*>(out_n_k_ho_wo_dev_buf.GetDeviceBuffer()),
            (const void*)(a_gk_gm0_gm10_gm11_grid_desc_dev_buf),
            (const void*)(b_gk_gn0_gn10_gn11_grid_desc_dev_buf),
            (const void*)(c_gm10_bm0_bm1_gn10_bn0_bn1_grid_desc_dev_buf),
            (const void*)(c_blockid_to_gm10_gn10_block_cluster_adaptor_dev_buf));
        timer2.End();

        kernel1_times.push_back(timer1.GetElapsedTime());
        kernel2_times.push_back(timer2.GetElapsedTime());
    }

    {
        auto ave_time1 = Driver::get_effective_average(kernel1_times);
        auto ave_time2 = Driver::get_effective_average(kernel2_times);

        const auto N = in_n_c_hi_wi_lengths[I0];
        const auto C = in_n_c_hi_wi_lengths[I1];

        const auto K  = out_n_k_ho_wo_lengths[I1];
        const auto Ho = out_n_k_ho_wo_lengths[I2];
        const auto Wo = out_n_k_ho_wo_lengths[I3];

        const auto Y = wei_k_c_y_x_lengths[I2];
        const auto X = wei_k_c_y_x_lengths[I3];

        float perf = (float)(std::size_t(2) * N * K * Ho * Wo * C * Y * X) /
                     (std::size_t(1000) * 1000 * 1000) / (ave_time1 + ave_time2);

        std::cout << "Average time : " << ave_time1 + ave_time2 << " ms(" << ave_time1 << ", "
                  << ave_time2 << "), " << perf << " TFlop/s" << std::endl;
    };

    // copy result back to host
    out_n_k_ho_wo_dev_buf.FromDevice(out_n_k_ho_wo.mData.data());
}
