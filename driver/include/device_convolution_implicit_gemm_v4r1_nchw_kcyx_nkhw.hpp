#pragma once
#include <unistd.h>
#include "device.hpp"
#include "host_tensor.hpp"
#include "gridwise_operation_wrapper.hpp"
#include "gridwise_convolution_implicit_gemm_v4r1_nchw_kcyx_nkhw_lds_double_buffer.hpp"

template <typename T,
          typename InDesc,
          typename WeiDesc,
          typename OutDesc,
          typename ConvStrides,
          typename ConvDilations,
          typename LeftPads,
          typename RightPads>
void device_convolution_implicit_gemm_v4r1_nchw_kcyx_nkhw(InDesc,
                                                          const Tensor<T>& in_nchw,
                                                          WeiDesc,
                                                          const Tensor<T>& wei_kcyx,
                                                          OutDesc,
                                                          Tensor<T>& out_nkhw,
                                                          ConvStrides,
                                                          ConvDilations,
                                                          LeftPads,
                                                          RightPads,
                                                          ck::index_t nrepeat)
{
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
    // cdata = 64, BlockSize = 256,  64x256x8
    constexpr index_t BlockSize = 256;

    constexpr index_t KPerBlock =  64;
    constexpr index_t BPerBlock = 32;
    constexpr index_t EPerBlock = 8;

    constexpr index_t GemmNRepeat = 2;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 16;

    constexpr index_t GemmDataPerReadA   = 4;
    constexpr index_t GemmDataPerReadB   = 4;

    using InBlockCopySubLengths_E_N1_B_N2      = Sequence<1, 2, 1, 4>;
    using InBlockCopyClusterLengths_E_N1_B_N2  = Sequence<8, 1, 32, 1>;
    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 2, 1, 3>; // [E, B, N1, N2]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [E, N1, B, N2]

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 4;

    using WeiBlockCopySubLengths_E_K            = Sequence<2, 1>;
    using WeiBlockCopyClusterLengths_E_K        = Sequence<4, 64>;
    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 2;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 1;
#elif 0
    // cdata = 64, BlockSize = 256, 128x128x4
    constexpr index_t BlockSize = 256;

    constexpr index_t KPerBlock = 128;
    constexpr index_t BPerBlock = 16;
    constexpr index_t EPerBlock = 4;

    constexpr index_t GemmNRepeat = 2;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 8;
    constexpr index_t GemmNLevel1Cluster = 8;

    constexpr index_t GemmDataPerReadA = 4;
    constexpr index_t GemmDataPerReadB = 4;

    using InBlockCopySubLengths_E_N1_B_N2      = Sequence<1, 1, 1, 2>;
    using InBlockCopyClusterLengths_E_N1_B_N2  = Sequence<4, 2, 16, 2>;
    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 2, 1, 3>; // [E, B, N1, N2]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [E, N1, B, N2]

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 2;

    using WeiBlockCopySubLengths_E_K            = Sequence<2, 1>;
    using WeiBlockCopyClusterLengths_E_K        = Sequence<2, 128>;
    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 2;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 1;
#elif 1
    // cdata = 64, BlockSize = 256, 128x128x8
    constexpr index_t BlockSize = 256;

    constexpr index_t KPerBlock = 128;
    constexpr index_t BPerBlock = 16;
    constexpr index_t EPerBlock = 8;

    constexpr index_t GemmNRepeat = 2;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 8;
    constexpr index_t GemmNLevel1Cluster = 8;

    constexpr index_t GemmDataPerReadA = 4;
    constexpr index_t GemmDataPerReadB = 4;

    using InBlockCopySubLengths_E_N1_B_N2      = Sequence<1, 1, 1, 4>;
    using InBlockCopyClusterLengths_E_N1_B_N2  = Sequence<8, 2, 16, 1>;
    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 2, 1, 3>; // [E, B, N1, N2]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [E, N1, B, N2]

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 4;

    using WeiBlockCopySubLengths_E_K            = Sequence<4, 1>;
    using WeiBlockCopyClusterLengths_E_K        = Sequence<2, 128>;
    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 4;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 1;
#elif 0
    // cdata = 64, BlockSize = 256, 128x128x16
    constexpr index_t BlockSize = 256;

    constexpr index_t KPerBlock = 128;
    constexpr index_t BPerBlock = 16;
    constexpr index_t EPerBlock = 16;

    constexpr index_t GemmNRepeat = 2;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 4;

    constexpr index_t GemmDataPerReadA = 4;
    constexpr index_t GemmDataPerReadB = 4;

    using InBlockCopySubLengths_E_N1_B_N2      = Sequence<1, 2, 1, 4>;
    using InBlockCopyClusterLengths_E_N1_B_N2  = Sequence<16, 1, 16, 1>;
    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 2, 1, 3>; // [E, B, N1, N2]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [E, N1, B, N2]

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 4;

    using WeiBlockCopySubLengths_E_K            = Sequence<4, 2>;
    using WeiBlockCopyClusterLengths_E_K        = Sequence<4, 64>;
    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 4;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 2;
#elif 0
    // cdata = 4, BlockSize = 256, 128x128x16
    // for 1x1
    constexpr index_t BlockSize = 256;

    constexpr index_t KPerBlock = 128;
    constexpr index_t BPerBlock = 16;
    constexpr index_t EPerBlock = 16;

    constexpr index_t GemmNRepeat = 2;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 4;

    constexpr index_t GemmDataPerReadA = 4;
    constexpr index_t GemmDataPerReadB = 4;

    using InBlockCopySubLengths_E_N1_B_N2      = Sequence<4, 1, 1, 2>;
    using InBlockCopyClusterLengths_E_N1_B_N2  = Sequence<4, 2, 16, 2>;
    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 2, 1, 3>; // [E, B, N1, N2]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [E, N1, B, N2]

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 2;

    using WeiBlockCopySubLengths_E_K            = Sequence<4, 2>;
    using WeiBlockCopyClusterLengths_E_K        = Sequence<4, 64>;
    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 4;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 2;
#elif 0
    // cdata = 64, BlockSize = 128, 64x128x4
    constexpr index_t BlockSize = 128;

    constexpr index_t KPerBlock = 64;
    constexpr index_t BPerBlock = 16;
    constexpr index_t EPerBlock = 4;

    constexpr index_t GemmNRepeat = 2;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 8;

    constexpr index_t GemmDataPerReadA = 4;
    constexpr index_t GemmDataPerReadB = 4;

    using InBlockCopySubLengths_E_N1_B_N2      = Sequence<1, 1, 1, 4>;
    using InBlockCopyClusterLengths_E_N1_B_N2  = Sequence<4, 2, 16, 1>;
    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 2, 1, 3>; // [E, B, N1, N2]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [E, N1, B, N2]

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 4;

    using WeiBlockCopySubLengths_E_K            = Sequence<2, 1>;
    using WeiBlockCopyClusterLengths_E_K        = Sequence<2, 64>;
    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 2;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 1;
#elif 0
    // cdata = 64, BlockSize = 128, 64x128x8
    constexpr index_t BlockSize = 128;

    constexpr index_t KPerBlock = 64;
    constexpr index_t BPerBlock = 16;
    constexpr index_t EPerBlock = 8;

    constexpr index_t GemmNRepeat = 2;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 8;

    constexpr index_t GemmDataPerReadA = 4;
    constexpr index_t GemmDataPerReadB = 4;

    using InBlockCopySubLengths_E_N1_B_N2      = Sequence<1, 2, 1, 4>;
    using InBlockCopyClusterLengths_E_N1_B_N2  = Sequence<8, 1, 16, 1>;
    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 2, 1, 3>; // [E, B, N1, N2]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [E, N1, B, N2]

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 4;

    using WeiBlockCopySubLengths_E_K            = Sequence<4, 1>;
    using WeiBlockCopyClusterLengths_E_K        = Sequence<2, 64>;
    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 4;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 1;
#elif 0
    // cdata = 64, BlockSize = 128, 64x128x16
    constexpr index_t BlockSize = 128;

    constexpr index_t KPerBlock = 64;
    constexpr index_t BPerBlock = 16;
    constexpr index_t EPerBlock = 16;

    constexpr index_t GemmNRepeat = 2;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 4;

    constexpr index_t GemmDataPerReadA = 4;
    constexpr index_t GemmDataPerReadB = 4;

    using InBlockCopySubLengths_E_N1_B_N2      = Sequence<2, 2, 1, 4>;
    using InBlockCopyClusterLengths_E_N1_B_N2  = Sequence<8, 1, 16, 1>;
    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 2, 1, 3>; // [E, B, N1, N2]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [E, N1, B, N2]

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 4;

    using WeiBlockCopySubLengths_E_K            = Sequence<4, 2>;
    using WeiBlockCopyClusterLengths_E_K        = Sequence<4, 32>;
    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 4;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 2;
#elif 0
    // cdata = 64, BlockSize = 128, 128x64x4
    constexpr index_t BlockSize = 128;

    constexpr index_t KPerBlock = 128;
    constexpr index_t BPerBlock = 8;
    constexpr index_t EPerBlock = 4;

    constexpr index_t GemmNRepeat = 2;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 8;
    constexpr index_t GemmNLevel1Cluster = 4;

    constexpr index_t GemmDataPerReadA = 4;
    constexpr index_t GemmDataPerReadB = 4;

    using InBlockCopySubLengths_E_N1_B_N2      = Sequence<1, 1, 1, 2>;
    using InBlockCopyClusterLengths_E_N1_B_N2  = Sequence<4, 2, 8, 2>;
    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 2, 1, 3>; // [E, B, N1, N2]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [E, N1, B, N2]

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 2;

    using WeiBlockCopySubLengths_E_K            = Sequence<2, 2>;
    using WeiBlockCopyClusterLengths_E_K        = Sequence<2, 64>;
    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 2;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 2;
#elif 0
    // cdata = 64, BlockSize = 128, 128x64x8
    constexpr index_t BlockSize = 128;

    constexpr index_t KPerBlock = 128;
    constexpr index_t BPerBlock = 8;
    constexpr index_t EPerBlock = 8;

    constexpr index_t GemmNRepeat = 2;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 8;
    constexpr index_t GemmNLevel1Cluster = 4;

    constexpr index_t GemmDataPerReadA = 4;
    constexpr index_t GemmDataPerReadB = 4;

    using InBlockCopySubLengths_E_N1_B_N2      = Sequence<1, 1, 1, 4>;
    using InBlockCopyClusterLengths_E_N1_B_N2  = Sequence<8, 2, 8, 1>;
    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 2, 1, 3>; // [E, B, N1, N2]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [E, N1, B, N2]

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 4;

    using WeiBlockCopySubLengths_E_K            = Sequence<4, 2>;
    using WeiBlockCopyClusterLengths_E_K        = Sequence<2, 64>;
    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 4;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 2;
#elif 0
    // cdata = 64, BlockSize = 128, 128x64x16
    constexpr index_t BlockSize = 128;

    constexpr index_t KPerBlock = 128;
    constexpr index_t BPerBlock = 8;
    constexpr index_t EPerBlock = 16;

    constexpr index_t GemmNRepeat = 2;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 8;
    constexpr index_t GemmNLevel1Cluster = 4;

    constexpr index_t GemmDataPerReadA = 4;
    constexpr index_t GemmDataPerReadB = 4;

    using InBlockCopySubLengths_E_N1_B_N2      = Sequence<1, 2, 1, 4>;
    using InBlockCopyClusterLengths_E_N1_B_N2  = Sequence<16, 1, 8, 1>;
    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 2, 1, 3>; // [E, B, N1, N2]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [E, N1, B, N2]

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 4;

    using WeiBlockCopySubLengths_E_K            = Sequence<4, 4>;
    using WeiBlockCopyClusterLengths_E_K        = Sequence<4, 32>;
    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 4;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 4;
#elif 0
    // cdata = 64, BlockSize = 64, 64x64x8
    constexpr index_t BlockSize = 64;

    constexpr index_t KPerBlock = 64;
    constexpr index_t BPerBlock = 8;
    constexpr index_t EPerBlock = 8;

    constexpr index_t GemmNRepeat = 2;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 2;

    constexpr index_t GemmDataPerReadA = 4;
    constexpr index_t GemmDataPerReadB = 4;

    using InBlockCopySubLengths_E_N1_B_N2      = Sequence<1, 2, 1, 4>;
    using InBlockCopyClusterLengths_E_N1_B_N2  = Sequence<8, 1, 8, 1>;
    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 2, 1, 3>; // [E, B, N1, N2]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [E, N1, B, N2]

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 4;

    using WeiBlockCopySubLengths_E_K            = Sequence<4, 2>;
    using WeiBlockCopyClusterLengths_E_K        = Sequence<2, 32>;
    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 4;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 1;
#elif 0
    // cdata = 64, BlockSize = 32, 32x64x3
    constexpr index_t BlockSize = 32;

    constexpr index_t KPerBlock = 32;
    constexpr index_t BPerBlock = 8;
    constexpr index_t EPerBlock = 3;

    constexpr index_t GemmNRepeat = 2;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 1;
    constexpr index_t GemmNLevel1Cluster = 2;

    constexpr index_t GemmDataPerReadA = 4;
    constexpr index_t GemmDataPerReadB = 4;

    using InBlockCopySubLengths_E_N1_B_N2      = Sequence<3, 1, 1, 2>;
    using InBlockCopyClusterLengths_E_N1_B_N2  = Sequence<1, 2, 8, 2>;
    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 2, 1, 3>; // [E, B, N1, N2]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [E, N1, B, N2]

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 2;

    using WeiBlockCopySubLengths_E_K            = Sequence<3, 1>;
    using WeiBlockCopyClusterLengths_E_K        = Sequence<1, 32>;
    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 1;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 1;
#elif 0
    // cdata = 64, BlockSize = 64, 32x128x3
    constexpr index_t BlockSize = 64;

    constexpr index_t KPerBlock = 32;
    constexpr index_t BPerBlock = 16;
    constexpr index_t EPerBlock = 3;

    constexpr index_t GemmNRepeat = 2;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 1;
    constexpr index_t GemmNLevel1Cluster = 4;

    constexpr index_t GemmDataPerReadA = 4;
    constexpr index_t GemmDataPerReadB = 4;

    using InBlockCopySubLengths_E_N1_B_N2      = Sequence<3, 1, 1, 2>;
    using InBlockCopyClusterLengths_E_N1_B_N2  = Sequence<1, 2, 16, 2>;
    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 2, 1, 3>; // [E, B, N1, N2]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [E, N1, B, N2]

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 2;

    using WeiBlockCopySubLengths_E_K            = Sequence<3, 1>;
    using WeiBlockCopyClusterLengths_E_K        = Sequence<1, 32>;
    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 1;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 1;
#elif 0
    // cdata = 64, BlockSize = 64, 64x64x3
    constexpr index_t BlockSize = 64;

    constexpr index_t KPerBlock = 64;
    constexpr index_t BPerBlock = 8;
    constexpr index_t EPerBlock = 3;

    constexpr index_t GemmNRepeat = 2;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 4;

    constexpr index_t GemmDataPerReadA = 4;
    constexpr index_t GemmDataPerReadB = 4;

    using InBlockCopySubLengths_E_N1_B_N2      = Sequence<3, 1, 1, 1>;
    using InBlockCopyClusterLengths_E_N1_B_N2  = Sequence<1, 2, 8, 4>;
    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 2, 1, 3>; // [E, B, N1, N2]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [E, N1, B, N2]

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 1;

    using WeiBlockCopySubLengths_E_K            = Sequence<3, 1>;
    using WeiBlockCopyClusterLengths_E_K        = Sequence<1, 64>;
    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 1;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 1;
#elif 0
    // cdata = 64, BlockSize = 64, 32x128x4
    constexpr index_t BlockSize = 64;

    constexpr index_t KPerBlock = 32;
    constexpr index_t BPerBlock = 16;
    constexpr index_t EPerBlock = 4;

    constexpr index_t GemmNRepeat = 2;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 2;
    constexpr index_t GemmNLevel0Cluster = 2;
    constexpr index_t GemmMLevel1Cluster = 2;
    constexpr index_t GemmNLevel1Cluster = 8;

    constexpr index_t GemmDataPerReadA = 4;
    constexpr index_t GemmDataPerReadB = 4;

    using InBlockCopySubLengths_E_N1_B_N2      = Sequence<1, 2, 1, 4>;
    using InBlockCopyClusterLengths_E_N1_B_N2  = Sequence<4, 1, 16, 1>;
    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 2, 1, 3>; // [E, B, N1, N2]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [E, N1, B, N2]

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 4;

    using WeiBlockCopySubLengths_E_K            = Sequence<2, 1>;
    using WeiBlockCopyClusterLengths_E_K        = Sequence<2, 32>;
    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 2;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 1;
#elif 0
    // cdata = 64, BlockSize = 64, 32x128x8
    constexpr index_t BlockSize = 64;

    constexpr index_t KPerBlock = 32;
    constexpr index_t BPerBlock = 16;
    constexpr index_t EPerBlock = 8;

    constexpr index_t GemmNRepeat = 2;

    constexpr index_t GemmMPerThread = 4;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 1;
    constexpr index_t GemmNLevel1Cluster = 4;

    constexpr index_t GemmDataPerReadA = 4;
    constexpr index_t GemmDataPerReadB = 4;

    using InBlockCopySubLengths_E_N1_B_N2      = Sequence<2, 2, 1, 4>;
    using InBlockCopyClusterLengths_E_N1_B_N2  = Sequence<4, 1, 16, 1>;
    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 2, 1, 3>; // [E, B, N1, N2]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [E, N1, B, N2]

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 4;

    using WeiBlockCopySubLengths_E_K            = Sequence<4, 1>;
    using WeiBlockCopyClusterLengths_E_K        = Sequence<2, 32>;
    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 4;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 1;
#elif 0
    // cdata = 32, BlockSize = 256, 64x128x8
    constexpr index_t BlockSize = 256;

    constexpr index_t KPerBlock = 64;
    constexpr index_t BPerBlock = 16;
    constexpr index_t EPerBlock = 8;

    constexpr index_t GemmNRepeat = 2;

    constexpr index_t GemmMPerThread = 2;
    constexpr index_t GemmNPerThread = 4;
    constexpr index_t GemmKPerThread = 1;

    constexpr index_t GemmMLevel0Cluster = 4;
    constexpr index_t GemmNLevel0Cluster = 4;
    constexpr index_t GemmMLevel1Cluster = 4;
    constexpr index_t GemmNLevel1Cluster = 4;

    constexpr index_t GemmDataPerReadA = 2;
    constexpr index_t GemmDataPerReadB = 4;

    using InBlockCopySubLengths_E_N1_B_N2      = Sequence<1, 1, 1, 4>;
    using InBlockCopyClusterLengths_E_N1_B_N2  = Sequence<8, 2, 16, 1>;
    using InBlockCopyThreadClusterArrangeOrder = Sequence<0, 1, 3, 2>; // [E, N1, N2, B]
    using InBlockCopySrcAccessOrder            = Sequence<0, 2, 1, 3>; // [E, B, N1, N2]
    using InBlockCopyDstAccessOrder            = Sequence<0, 1, 2, 3>; // [E, N1, B, N2]

    constexpr index_t InBlockCopySrcDataPerRead_B   = 1;
    constexpr index_t InBlockCopyDstDataPerWrite_N2 = 4;

    using WeiBlockCopySubLengths_E_K            = Sequence<2, 1>;
    using WeiBlockCopyClusterLengths_E_K        = Sequence<4, 64>;
    using WeiBlockCopyThreadClusterArrangeOrder = Sequence<1, 0>; // [K, E]
    using WeiBlockCopySrcAccessOrder            = Sequence<1, 0>; // [K, E]
    using WeiBlockCopyDstAccessOrder            = Sequence<0, 1>; // [E, K]

    constexpr index_t WeiBlockCopySrcDataPerRead_E  = 2;
    constexpr index_t WeiBlockCopyDstDataPerWrite_K = 1;
#endif

    constexpr index_t N1 = GemmNRepeat;
    constexpr index_t N2 = GemmNPerThread;

    constexpr index_t B = (N * Ho * Wo) / (N1 * N2);

    constexpr index_t GridSize =
        ((B + BPerBlock - 1) / BPerBlock) * ((K + KPerBlock - 1) / KPerBlock);

    printf("%s: BlockSize %u, GridSize %u \n", __func__, BlockSize, GridSize);

    using gridwise_conv = GridwiseConvolutionImplicitGemm_v4r1_nchw_kcyx_nkhw_lds_double_buffer<
        GridSize,
        BlockSize,
        T,
        T,
        decltype(in_nchw_desc),
        decltype(wei_kcyx_desc),
        decltype(out_nkhw_desc),
        ConvStrides,
        ConvDilations,
        LeftPads,
        RightPads,
        BPerBlock,
        KPerBlock,
        EPerBlock,
        GemmNRepeat,
        GemmMPerThread,
        GemmNPerThread,
        GemmKPerThread,
        GemmMLevel0Cluster,
        GemmNLevel0Cluster,
        GemmMLevel1Cluster,
        GemmNLevel1Cluster,
        GemmDataPerReadA,
        GemmDataPerReadB,
        InBlockCopySubLengths_E_N1_B_N2,
        InBlockCopyClusterLengths_E_N1_B_N2,
        InBlockCopyThreadClusterArrangeOrder,
        InBlockCopySrcAccessOrder,
        InBlockCopyDstAccessOrder,
        InBlockCopySrcDataPerRead_B,
        InBlockCopyDstDataPerWrite_N2,
        WeiBlockCopySubLengths_E_K,
        WeiBlockCopyClusterLengths_E_K,
        WeiBlockCopyThreadClusterArrangeOrder,
        WeiBlockCopySrcAccessOrder,
        WeiBlockCopyDstAccessOrder,
        WeiBlockCopySrcDataPerRead_E,
        WeiBlockCopyDstDataPerWrite_K>;

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
