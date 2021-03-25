#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "gridwise_operation_wrapper.hpp"
#include "gridwise_convolution_forward_implicit_gemm_v4r4_nchw_kcyx_nkhw.hpp"

template <class T,
          class InDesc,
          class WeiDesc,
          class OutDesc,
          class ConvStrides,
          class ConvDilations,
          class InLeftPads,
          class InRightPads>
void device_convolution_forward_implicit_gemm_v4r4_nchw_kcyx_nkhw(InDesc,
                                                                  const Tensor<T>& in_nchw,
                                                                  WeiDesc,
                                                                  const Tensor<T>& wei_kcyx,
                                                                  OutDesc,
                                                                  Tensor<T>& out_nkhw,
                                                                  ConvStrides,
                                                                  ConvDilations,
                                                                  InLeftPads,
                                                                  InRightPads,
                                                                  ck::index_t nrepeat)
{
    std::cout << "device_convolution_forward_implicit_gemm_v4r4_nchw_kcyx_nkhw" << std::endl;

    using namespace ck;

    using TDevice = typename conditional<is_same<half_float::half, T>::value, half_t, T>::type;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    constexpr auto in_nchw_desc =
        make_native_tensor_descriptor(InDesc::GetLengths(), InDesc::GetStrides());
    constexpr auto wei_kcyx_desc =
        make_native_tensor_descriptor(WeiDesc::GetLengths(), WeiDesc::GetStrides());
    constexpr auto out_nkhw_desc =
        make_native_tensor_descriptor(OutDesc::GetLengths(), OutDesc::GetStrides());

    constexpr index_t N  = out_nkhw_desc.GetLength(I0);
    constexpr index_t K  = out_nkhw_desc.GetLength(I1);
    constexpr index_t Ho = out_nkhw_desc.GetLength(I2);
    constexpr index_t Wo = out_nkhw_desc.GetLength(I3);

    std::size_t data_sz = sizeof(T);
    DeviceMem in_nchw_device_buf(data_sz * in_nchw.mDesc.GetElementSpace());
    DeviceMem wei_kcyx_device_buf(data_sz * wei_kcyx.mDesc.GetElementSpace());
    DeviceMem out_nkhw_device_buf(data_sz * out_nkhw.mDesc.GetElementSpace());

    in_nchw_device_buf.ToDevice(in_nchw.mData.data());
    wei_kcyx_device_buf.ToDevice(wei_kcyx.mData.data());
    out_nkhw_device_buf.ToDevice(out_nkhw.mData.data());

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

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<1, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<4, 16>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 1;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<4, 1>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<1, 64>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 2;
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
    // cdata = 64, BlockSize = 256, 64x256x8
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 64;
    constexpr index_t GemmNPerBlock = 256;
    constexpr index_t GemmKPerBlock = 8;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 16;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<2, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<4, 64>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 2;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<8, 1>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<1, 256>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
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

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<1, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<2, 128>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 1;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<1, 1>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<2, 128>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
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

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<2, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<2, 128>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 2;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<2, 1>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<2, 128>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
#elif 1
    // cdata = 64, BlockSize = 256, 128x128x8
    // b threadwise copy 4x1
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

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<4, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<2, 128>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 4;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<4, 1>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<2, 128>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
#elif 0
    // cdata = 64, BlockSize = 256, 128x128x8
    // b threadwise copy 2x2
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

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<4, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<2, 128>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 4;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<2, 2>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<4, 64>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
#elif 0
    // cdata = 64, BlockSize = 256, 128x128x8
    // vector 4
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

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<4, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<2, 128>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 4;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<1, 4>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<8, 32>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 4;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 4;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 4;
#elif 0
    // cdata = 64, BlockSize = 256, 128x128x16
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

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<4, 2>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<4, 64>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 4;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<4, 2>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<4, 64>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
#elif 0
    // cdata = 64, BlockSize = 256, 128x128x8
    // GemmBBlockCopySrcDataPerRead_GemmN = 4
    // GemmCThreadCopyDstDataPerWrite_GemmN1 = 4
    constexpr index_t BlockSize = 256;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 8;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 4;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<4, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<2, 128>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 4;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<1, 4>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<8, 32>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 4;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 4;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 4;
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

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<4, 2>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<4, 64>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 4;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<2, 4>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<8, 32>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 4;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 4;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 4;
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

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<1, 8>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<16, 16>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 1;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 4;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<2, 4>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<8, 32>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 4;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 4;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 4;
#elif 0
    // cdata = 64, BlockSize = 128, 128x64x4
    constexpr index_t BlockSize = 128;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 64;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 4;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<4, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<1, 128>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 4;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<2, 1>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<2, 64>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
#elif 0
    // cdata = 64, BlockSize = 128, 128x64x4
    // GemmBBlockCopySrcDataPerRead_GemmN = 2
    // GemmCThreadCopyDstDataPerWrite_GemmN1 = 2
    constexpr index_t BlockSize = 128;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 64;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 8;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<4, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<1, 128>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 4;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<1, 2>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<4, 32>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 2;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 2;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 2;
#elif 0
    // cdata = 64, BlockSize = 128, 128x64x8
    constexpr index_t BlockSize = 128;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 64;
    constexpr index_t GemmKPerBlock = 8;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 8;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<4, 2>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<2, 64>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 4;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 2;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<4, 1>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<2, 64>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
#elif 0
    // cdata = 64, BlockSize = 128, 128x64x8
    // GemmBBlockCopySrcDataPerRead_GemmN = 4
    // GemmCThreadCopyDstDataPerWrite_GemmN1 = 4
    constexpr index_t BlockSize = 128;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 64;
    constexpr index_t GemmKPerBlock = 8;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 8;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<4, 2>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<2, 64>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 4;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 2;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<1, 4>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<8, 16>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 4;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 4;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 4;
#elif 0
    // cdata = 64, BlockSize = 128, 128x64x16
    constexpr index_t BlockSize = 128;

    constexpr index_t GemmMPerBlock = 128;
    constexpr index_t GemmNPerBlock = 64;
    constexpr index_t GemmKPerBlock = 16;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 4;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<4, 4>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<4, 32>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 4;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 4;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<8, 1>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<2, 64>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
#elif 0
    // cdata = 64, BlockSize = 128, 64x128x4
    constexpr index_t BlockSize = 128;

    constexpr index_t GemmMPerBlock = 64;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 4;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<2, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<2, 64>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 2;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<4, 1>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<1, 128>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
#elif 0
    // BlockSize = 128, 64x128x4
    // GemmBBlockCopySrcDataPerRead_GemmN = 4
    // GemmCThreadCopyDstDataPerWrite_GemmN1 = 4
    constexpr index_t BlockSize = 128;

    constexpr index_t GemmMPerBlock = 64;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 8;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<2, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<2, 64>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 2;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<1, 4>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<4, 32>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 4;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 4;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 4;
#elif 0
    // cdata = 64, BlockSize = 128, 64x128x8
    constexpr index_t BlockSize = 128;

    constexpr index_t GemmMPerBlock = 64;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 8;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 8;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<4, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<2, 64>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 4;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<8, 1>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<1, 128>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
#elif 0
    // BlockSize = 128, 64x128x8
    // GemmBBlockCopySrcDataPerRead_GemmN = 4
    // GemmCThreadCopyDstDataPerWrite_GemmN1 = 4
    constexpr index_t BlockSize = 128;

    constexpr index_t GemmMPerBlock = 64;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 8;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 8;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<4, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<2, 64>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 4;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<2, 4>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<4, 32>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 4;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 4;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 4;
#elif 0
    // BlockSize = 128, 64x128x16
    constexpr index_t BlockSize = 128;

    constexpr index_t GemmMPerBlock = 64;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 16;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 4;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<8, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<2, 64>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 4;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<16, 1>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<1, 128>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
#elif 0
    // cdata = 64, BlockSize = 64, 64x64x4
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlock = 64;
    constexpr index_t GemmNPerBlock = 64;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 2;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<4, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<1, 64>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 4;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<4, 1>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<1, 64>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
#elif 0
    // cdata = 64, BlockSize = 64, 64x64x8
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlock = 64;
    constexpr index_t GemmNPerBlock = 64;
    constexpr index_t GemmKPerBlock = 8;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 2;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<4, 2>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<2, 32>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 4;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 2;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<8, 1>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<1, 64>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
#elif 0
    // cdata = 64, BlockSize =  64, 32x128x2
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlock = 32;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 2;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 8;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<1, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<2, 32>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 1;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<2, 2>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<1, 64>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
#elif 0
    // cdata = 64, BlockSize =  64, 32x128x4
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlock = 32;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 4;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 8;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<2, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<2, 32>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 2;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<4, 2>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<1, 64>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
#elif 0
    // cdata = 64, BlockSize = 32, 32x64x3
    constexpr index_t BlockSize = 32;

    constexpr index_t GemmMPerBlock = 32;
    constexpr index_t GemmNPerBlock = 64;
    constexpr index_t GemmKPerBlock = 3;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 4;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<3, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<1, 32>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 1;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<3, 2>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<1, 32>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 2;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
#elif 0
    // cdata = 64, BlockSize = 64, 32x128x3
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlock = 32;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 3;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 8;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<3, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<1, 32>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 1;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<3, 2>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<1, 64>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 2;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
#elif 0
    // cdata = 64, BlockSize = 64, 64x64x3
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlock = 64;
    constexpr index_t GemmNPerBlock = 64;
    constexpr index_t GemmKPerBlock = 3;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 4;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<3, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<1, 64>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 1;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<3, 1>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<1, 64>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
#elif 0
    // cdata = 64, BlockSize =  64, 32x128x8
    constexpr index_t BlockSize = 64;

    constexpr index_t GemmMPerBlock = 32;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 8;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 4;

    constexpr index_t ThreadGemmDataPerReadM = 4;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<4, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<2, 32>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 4;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<8, 2>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<1, 64>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
#elif 1
    // cdata = 32, BlockSize = 128, 32x128x8
    constexpr index_t BlockSize = 128;

    constexpr index_t GemmMPerBlock = 32;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 8;

    constexpr index_t GemmMPerThread = 2;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 4;

    constexpr index_t ThreadGemmDataPerReadM = 2;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<2, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<4, 32>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 2;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<8, 1>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<1, 128>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
#elif 0
    // cdata = 32, BlockSize = 128, 32x128x16
    constexpr index_t BlockSize = 128;

    constexpr index_t GemmMPerBlock = 32;
    constexpr index_t GemmNPerBlock = 128;
    constexpr index_t GemmKPerBlock = 16;

    constexpr index_t GemmMPerThread = 2;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 4;

    constexpr index_t ThreadGemmDataPerReadM = 2;
    constexpr index_t ThreadGemmDataPerReadN = 4;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM   = Sequence<4, 1>;
    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM = Sequence<4, 32>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK  = 4;
    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM = 1;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN   = Sequence<16, 1>;
    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN = Sequence<1, 128>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmN  = 1;
    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN = 1;

    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 = 1;
#endif

    constexpr index_t GemmM = K;
    constexpr index_t GemmN = N * Ho * Wo;

    constexpr index_t GridSize = math::integer_divide_ceil(GemmM, GemmMPerBlock) *
                                 math::integer_divide_ceil(GemmN, GemmNPerBlock);

    printf("%s: BlockSize %u, GridSize %u \n", __func__, BlockSize, GridSize);

    using gridwise_conv = GridwiseConvolutionForwardImplicitGemm_v4r4_nchw_kcyx_nkhw<
        GridSize,
        BlockSize,
        TDevice,
        TDevice,
        decltype(in_nchw_desc),
        decltype(wei_kcyx_desc),
        decltype(out_nkhw_desc),
        ConvStrides,
        ConvDilations,
        InLeftPads,
        InRightPads,
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
        ThreadGemmDataPerReadM,
        ThreadGemmDataPerReadN,
        GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
        GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
        GemmABlockCopySrcDataPerRead_GemmK,
        GemmABlockCopyDstDataPerWrite_GemmM,
        GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
        GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
        GemmBBlockCopySrcDataPerRead_GemmN,
        GemmBBlockCopyDstDataPerWrite_GemmN,
        GemmCThreadCopyDstDataPerWrite_GemmN1>;

    for(index_t i = 0; i < 5; ++i)
    {
        std::cout << "Start running " << nrepeat << " times..." << std::endl;

        KernelTimer timer;
        timer.Start();

        for(index_t j = 0; j < nrepeat; ++j)
        {
            launch_kernel(run_gridwise_operation<gridwise_conv,
                                                 const TDevice* const __restrict__,
                                                 const TDevice* const __restrict__,
                                                 TDevice* const __restrict__>,
                          dim3(GridSize),
                          dim3(BlockSize),
                          0,
                          0,
                          static_cast<TDevice*>(in_nchw_device_buf.GetDeviceBuffer()),
                          static_cast<TDevice*>(wei_kcyx_device_buf.GetDeviceBuffer()),
                          static_cast<TDevice*>(out_nkhw_device_buf.GetDeviceBuffer()));
        }

        timer.End();

        float ave_time = timer.GetElapsedTime() / nrepeat;

        float perf = (float)calculate_convolution_flops(InDesc{}, WeiDesc{}, OutDesc{}) /
                     (std::size_t(1000) * 1000 * 1000) / ave_time;

        std::cout << "Average time : " << ave_time << " ms, " << perf << " TFlop/s" << std::endl;
    }

    out_nkhw_device_buf.FromDevice(out_nkhw.mData.data());
}
