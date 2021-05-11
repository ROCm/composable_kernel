#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "driver_dynamic_convolution_forward_implicit_gemm_v4r4_nchw_kcyx_nkhw.hpp"

template <class TInWei,
          ck::index_t InWeiVectorSize,
          class TAcc,
          class TOut,
          class InDesc,
          class WeiDesc,
          class OutDesc,
          class ConvStrides,
          class ConvDilations,
          class InLeftPads,
          class InRightPads>
void device_dynamic_convolution_forward_implicit_gemm_v4r4_nchw_kcyx_nkhw(
    InDesc,
    const Tensor<TInWei>& in_n_c_hi_wi,
    WeiDesc,
    const Tensor<TInWei>& wei_k_c_y_x,
    OutDesc,
    Tensor<TOut>& out_n_k_ho_wo,
    ConvStrides,
    ConvDilations,
    InLeftPads,
    InRightPads,
    ck::index_t nrepeat)
{
    using namespace ck;

    std::cout << __func__ << std::endl;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};
    constexpr auto I5 = Number<5>{};
    constexpr auto I6 = Number<6>{};
    constexpr auto I7 = Number<7>{};
    constexpr auto I8 = Number<8>{};

    DeviceMem in_n_c_hi_wi_device_buf(sizeof(TInWei) * in_n_c_hi_wi.mDesc.GetElementSpace());
    DeviceMem wei_k_c_y_x_device_buf(sizeof(TInWei) * wei_k_c_y_x.mDesc.GetElementSpace());
    DeviceMem out_n_k_ho_wo_device_buf(sizeof(TOut) * out_n_k_ho_wo.mDesc.GetElementSpace());

    in_n_c_hi_wi_device_buf.ToDevice(in_n_c_hi_wi.mData.data());
    wei_k_c_y_x_device_buf.ToDevice(wei_k_c_y_x.mData.data());
    out_n_k_ho_wo_device_buf.ToDevice(out_n_k_ho_wo.mData.data());

#if 1
    // run-time variables
    const auto in_n_c_hi_wi_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(to_multi_index(InDesc::GetLengths()));
    const auto wei_k_c_y_x_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(to_multi_index(WeiDesc::GetLengths()));
    const auto out_n_k_ho_wo_desc =
        make_dynamic_naive_tensor_descriptor_packed_v2(to_multi_index(OutDesc::GetLengths()));

    const auto conv_strides   = to_multi_index(ConvStrides{});
    const auto conv_dilations = to_multi_index(ConvDilations{});
    const auto in_left_pads   = to_multi_index(InLeftPads{});
    const auto in_right_pads  = to_multi_index(InRightPads{});
#else
    // compile-time variables
    const auto in_n_c_hi_wi_desc = make_dynamic_naive_tensor_descriptor_packed_v2(
        sequence_to_tuple_of_number(InDesc::GetLengths()));
    const auto wei_k_c_y_x_desc = make_dynamic_naive_tensor_descriptor_packed_v2(
        sequence_to_tuple_of_number(WeiDesc::GetLengths()));
    const auto out_n_k_ho_wo_desc = make_dynamic_naive_tensor_descriptor_packed_v2(
        sequence_to_tuple_of_number(OutDesc::GetLengths()));

    const auto conv_strides   = sequence_to_tuple_of_number(ConvStrides{});
    const auto conv_dilations = sequence_to_tuple_of_number(ConvDilations{});
    const auto in_left_pads   = sequence_to_tuple_of_number(InLeftPads{});
    const auto in_right_pads  = sequence_to_tuple_of_number(InRightPads{});
#endif

#if 0
    // cdata = 16, BlockSize = 64, 16x64x4
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlock = 16;
    constexpr index_t GemmNPerBlock = 64;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerThread = 2;
    constexpr index_t GemmNPerThread = 2;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 8;

    constexpr index_t ThreadGemmDataPerReadM = 2;
    constexpr index_t ThreadGemmDataPerReadN = 2;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<1, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<4, 16>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<4, 1>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<1, 64>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN = 1;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmN1 = 2;
#elif 0
    // cdata = 32, BlockSize 64, 16x128x4
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlock = 16;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerThread = 2;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 8;

    constexpr index_t ThreadGemmDataPerReadM = 2;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<1, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<4, 16>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<4, 2>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<1, 64>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN = 1;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmN1 = 4;
#elif 0
    // cdata = 64, BlockSize 64, 16x256x2
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlock = 16;
    constexpr index_t GemmNPerBlock = 256;
    constexpr index_t GemmKPerBlock = 2;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 1;
    constexpr index_t GemmNLevel1Cluster = 16;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<1, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<2, 16>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<2, 4>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<1, 64>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN = 1;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmN1 = 4;
#elif 0
    // cdata = 64, BlockSize 64, 16x256x4
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlock = 16;
    constexpr index_t GemmNPerBlock = 256;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 1;
    constexpr index_t GemmNLevel1Cluster = 16;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<1, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<4, 16>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<4, 4>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<1, 64>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN = 1;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmN1 = 4;
#elif 0
    // cdata = 16, BlockSize = 64, 16x64x4
    // GemmBBlockCopySrcDataPerRead_GemmN = 4
    // GemmCThreadCopyDstDataPerWrite_GemmN1 = 2
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlock = 16;
    constexpr index_t GemmNPerBlock = 64;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerThread = 2;
    constexpr index_t GemmNPerThread = 2;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 8;

    constexpr index_t ThreadGemmDataPerReadM = 2;
    constexpr index_t ThreadGemmDataPerReadN = 2;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<1, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<4, 16>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<1, 4>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<4, 16>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 4;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmN1 = 2;
#elif 0
    // cdata = 32, BlockSize = 64, 16x128x4
    // GemmBBlockCopySrcDataPerRead_GemmN = 4
    // GemmCThreadCopyDstDataPerWrite_GemmN1 = 4
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlock = 16;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerThread = 2;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 8;

    constexpr index_t ThreadGemmDataPerReadM = 2;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<1, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<4, 16>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<2, 4>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<2, 32>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 4;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmN1 = 4;
#elif 0
    // cdata = 64, BlockSize = 128, 32x256x8
    constexpr index_t BlockSize = 128;

    constexpr index_t GemmMPerBlock = 32;
    constexpr index_t GemmNPerBlock = 256;
    constexpr index_t GemmKPerBlock = 8;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 16;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<2, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<4, 32>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<8, 2>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<1, 128>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN = 1;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmN1 = 1;
#elif 0
    // cdata = 64, BlockSize = 256, 128x128x2
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 2;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 8;
    constexpr index_t GemmNLevel1Cluster = 8;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<1, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<2, 128>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 1;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<1, 1>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<2, 128>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN = 1;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmN1 = 1;
#elif 0
    // cdata = 64, BlockSize = 256, 128x128x4
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 8;
    constexpr index_t GemmNLevel1Cluster = 8;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<2, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<2, 128>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 2;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<2, 1>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<2, 128>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN = 1;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmN1 = 1;
#elif 1
    // cdata = 64, BlockSize = 256, 128x128x8
    // b thread copy 4x1
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 8;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 8;
    constexpr index_t GemmNLevel1Cluster = 8;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<4, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<2, 128>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 4;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<4, 1>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<2, 128>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN = 1;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmN1 = 1;
#elif 1
    // cdata = 64, BlockSize = 256, 128x128x8
    // b thread copy 2x2
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 8;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 8;
    constexpr index_t GemmNLevel1Cluster = 8;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<4, 1>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<2, 128>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 2;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<2, 2>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<4, 64>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN = 1;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 1;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmN1 = 1;
#elif 1
    // cdata = 64, BlockSize = 256, 128x128x16
    // GemmBBlockCopySrcDataPerRead_GemmN = 4
    // GemmCThreadCopyDstDataPerWrite_GemmN1 = 4
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 16;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 4;

    using GemmABlockTransferThreadSliceLengths_GemmK_GemmM   = Sequence<4, 2>;
    using GemmABlockTransferThreadClusterLengths_GemmK_GemmM = Sequence<4, 64>;

    constexpr index_t GemmABlockTransferSrcScalarPerVector_GemmK = 4;
    constexpr index_t GemmABlockTransferDstScalarPerVector_GemmM = 1;

    using GemmBBlockTransferThreadSliceLengths_GemmK_GemmN   = Sequence<2, 4>;
    using GemmBBlockTransferThreadClusterLengths_GemmK_GemmN = Sequence<8, 32>;

    constexpr index_t GemmBBlockTransferSrcScalarPerVector_GemmN = 4;
    constexpr index_t GemmBBlockTransferDstScalarPerVector_GemmN = 4;

    constexpr index_t GemmCThreadTransferDstScalarPerVector_GemmN1 = 4;
#endif

    constexpr index_t GemmM1 = GemmMPerThread * GemmMLevel0Cluster * GemmMLevel1Cluster;
    constexpr index_t GemmN1 = GemmNPerThread * GemmNLevel0Cluster * GemmNLevel1Cluster;

    const auto descs =
#if 1
        transform_forward_convolution_into_gemm_v4r4_nchw_kcyx_nkhw_pad
#elif 0
        transform_forward_convolution_into_gemm_v4r4_nchw_kcyx_nkhw_no_pad
#else
        transform_forward_convolution_into_gemm_v4r4_nchw_kcyx_nkhw_1x1
#endif
        <GemmMPerBlock, GemmNPerBlock, GemmM1, GemmN1>(wei_k_c_y_x_desc,
                                                       in_n_c_hi_wi_desc,
                                                       out_n_k_ho_wo_desc,
                                                       conv_strides,
                                                       conv_dilations,
                                                       in_left_pads,
                                                       in_right_pads);

    for(index_t i = 0; i < 5; ++i)
    {
        float ave_time = launch_kernel_dynamic_gemm_v1<
            BlockSize,
            typename vector_type<TInWei, InWeiVectorSize>::type,
            TAcc,
            TOut,
            InMemoryDataOperation::Set,
            decltype(descs[I0]),
            decltype(descs[I1]),
            decltype(descs[I2]),
            decltype(descs[I3]),
            GemmMPerBlock,
            GemmNPerBlock,
            GemmKPerBlock,
            GemmMPerThread,
            GemmNPerThread,
            GemmKPerThread,
            GemmMLevel0Cluster,
            GemmNLevel0Cluster,
            GemmMLevel1Cluster,
            GemmNLevel1Cluster,
            GemmABlockTransferThreadSliceLengths_GemmK_GemmM,
            GemmABlockTransferThreadClusterLengths_GemmK_GemmM,
            Sequence<1, 0>,
            Sequence<1, 0>,
            0,
            GemmABlockTransferSrcScalarPerVector_GemmK,
            GemmABlockTransferDstScalarPerVector_GemmM,
            false, // don't move back src coordinate after threadwise copy
            GemmBBlockTransferThreadSliceLengths_GemmK_GemmN,
            GemmBBlockTransferThreadClusterLengths_GemmK_GemmN,
            Sequence<0, 1>,
            Sequence<0, 1>,
            1,
            GemmBBlockTransferSrcScalarPerVector_GemmN,
            GemmBBlockTransferDstScalarPerVector_GemmN,
            false, // don't move back src coordinate after threadwise copy, which will be fused with
                   // MoveSrcSliceWindow() to save addr computation
            Sequence<2, 3, 0, 1>,
            3,
            GemmCThreadTransferDstScalarPerVector_GemmN1,
            decltype(descs[I4]),
            decltype(descs[I5]),
            decltype(descs[I6]),
            decltype(descs[I7]),
            decltype(descs[I8])>(static_cast<typename vector_type<TInWei, InWeiVectorSize>::type*>(
                                     wei_k_c_y_x_device_buf.GetDeviceBuffer()),
                                 static_cast<typename vector_type<TInWei, InWeiVectorSize>::type*>(
                                     in_n_c_hi_wi_device_buf.GetDeviceBuffer()),
                                 static_cast<TOut*>(out_n_k_ho_wo_device_buf.GetDeviceBuffer()),
                                 descs[I0],
                                 descs[I1],
                                 descs[I2],
                                 descs[I3],
                                 descs[I4],
                                 descs[I5],
                                 descs[I6],
                                 descs[I7],
                                 descs[I8],
                                 nrepeat);

        float perf = (float)calculate_convolution_flops(
                         in_n_c_hi_wi_desc, wei_k_c_y_x_desc, out_n_k_ho_wo_desc) /
                     (std::size_t(1000) * 1000 * 1000) / ave_time;

        std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s" << std::endl;
    }

    // copy result back to host
    out_n_k_ho_wo_device_buf.FromDevice(out_n_k_ho_wo.mData.data());
}
