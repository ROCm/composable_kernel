// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_xdl_waveletmodel_cshuffle_v3.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using namespace ck::tensor_layout::convolution;

using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using F32  = float;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvBwdWeightDefault =
    ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Default;

static constexpr auto ConvBwdWeightFilter1x1Stride1Pad0 =
    ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0;

// Wave-specialized (wavelet) conv bwd weight instances for F16.
// All configs use TileLoad=256 (4 load waves) + TileMath=256 (4 math waves) = 512 threads.
// MPerXDL=NPerXDL=32 (32x32 MFMA), K0PerBlock=32, K1=8, NumGemmKPrefetchStage=1.
//
// AK0PerBlock = K0PerBlock / K1 = 32 / 8 = 4. Transfer cluster K0 dim must be <= 4.
// All A/B transfers use S<4, 32, 2>=256 (K0c=4, Mc/Nc=32, K1c=2).
// SrcScalar = MPerBlock/32 (A) or NPerBlock/32 (B). DstScalar_K1 = K1/K1c = 4.

template <ck::index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename ELayout,
          ConvolutionBackwardWeightSpecialization ConvSpec>
using device_grouped_conv_bwd_weight_wavelet_xdl_c_shuffle_f16_instances = std::tuple<
    // clang-format off
    //                                                                                                                                                                 |Prefetch|TileLoad|TileMath| M  |  N | K0|K1| MXdl|NXdl|MXdl|NXdl|
    //                                                                                                                                                                 | Stage  | Thrds  | Thrds  |    |    |   |  | XDL | XDL| /Wv| /Wv| A cluster     |A ArrangeOrd|A SrcAccOrd | AVDim|ASSca|ADSca|AExM| B cluster     |B ArrangeOrd|B SrcAccOrd | BVDim|BSSca|BDSca|BExN|CShM|CShN|  C cluster       |CSca|
    // M=128 N=128: 2x2 waves, 64 AccVGPRs/wave
    DeviceGroupedConvBwdWeight_Xdl_WaveletModel_CShuffleV3<NDimSpatial, ALayout, BLayout, ELayout, F16, F16, F16, F32, PassThrough, PassThrough, PassThrough, ConvSpec,    1,      256,     256,   128, 128,  32, 8,  32,  32,   2,   2, S<4, 32, 2>, S<2, 0, 1>, S<1, 0, 2>,     1,    4,    4, false, S<4, 32, 2>, S<2, 0, 1>, S<1, 0, 2>,     1,    4,    4, false,   1,   1, S<1, 32, 1, 8>,    2>,
    // M=64 N=64: 2x2 waves, 16 AccVGPRs/wave — smallest tile, targets Group 3 shapes
    DeviceGroupedConvBwdWeight_Xdl_WaveletModel_CShuffleV3<NDimSpatial, ALayout, BLayout, ELayout, F16, F16, F16, F32, PassThrough, PassThrough, PassThrough, ConvSpec,    1,      256,     256,    64,  64,  32, 8,  32,  32,   1,   1, S<4, 32, 2>, S<2, 0, 1>, S<1, 0, 2>,     1,    2,    4, false, S<4, 32, 2>, S<2, 0, 1>, S<1, 0, 2>,     1,    2,    4, false,   1,   1, S<1, 32, 1, 8>,    2>,
    // M=128 N=64: 2x2 waves, 32 AccVGPRs/wave — asymmetric M>N
    DeviceGroupedConvBwdWeight_Xdl_WaveletModel_CShuffleV3<NDimSpatial, ALayout, BLayout, ELayout, F16, F16, F16, F32, PassThrough, PassThrough, PassThrough, ConvSpec,    1,      256,     256,   128,  64,  32, 8,  32,  32,   2,   1, S<4, 32, 2>, S<2, 0, 1>, S<1, 0, 2>,     1,    4,    4, false, S<4, 32, 2>, S<2, 0, 1>, S<1, 0, 2>,     1,    2,    4, false,   1,   1, S<1, 32, 1, 8>,    2>,
    // M=64 N=128: 2x2 waves, 32 AccVGPRs/wave — asymmetric N>M
    DeviceGroupedConvBwdWeight_Xdl_WaveletModel_CShuffleV3<NDimSpatial, ALayout, BLayout, ELayout, F16, F16, F16, F32, PassThrough, PassThrough, PassThrough, ConvSpec,    1,      256,     256,    64, 128,  32, 8,  32,  32,   1,   2, S<4, 32, 2>, S<2, 0, 1>, S<1, 0, 2>,     1,    2,    4, false, S<4, 32, 2>, S<2, 0, 1>, S<1, 0, 2>,     1,    4,    4, false,   1,   1, S<1, 32, 1, 8>,    2>
    // clang-format on
    >;

// 2-way (4,2) wave-specialized instances for F16.
// TileLoad=256 (4 load waves) + TileMath=128 (2 math waves) = 384 threads (6 waves).
// Same load cluster as (4,4): S<4, 32, 2>=256.
// Math waves halved → each wave handles more MXdl/NXdl work → more AccVGPRs.
template <ck::index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename ELayout,
          ConvolutionBackwardWeightSpecialization ConvSpec>
using device_grouped_conv_bwd_weight_wavelet_4w2_xdl_c_shuffle_f16_instances = std::tuple<
    // clang-format off
    //                                                                                                                                                                 |Prefetch|TileLoad|TileMath| M  |  N | K0|K1| MXdl|NXdl|MXdl|NXdl|
    //                                                                                                                                                                 | Stage  | Thrds  | Thrds  |    |    |   |  | XDL | XDL| /Wv| /Wv| A cluster     |A ArrangeOrd|A SrcAccOrd | AVDim|ASSca|ADSca|AExM| B cluster     |B ArrangeOrd|B SrcAccOrd | BVDim|BSSca|BDSca|BExN|CShM|CShN|  C cluster       |CSca|
    // M=64 N=64: 1x2 waves, 32 AccVGPRs/wave
    DeviceGroupedConvBwdWeight_Xdl_WaveletModel_CShuffleV3<NDimSpatial, ALayout, BLayout, ELayout, F16, F16, F16, F32, PassThrough, PassThrough, PassThrough, ConvSpec,    1,      256,     128,    64,  64,  32, 8,  32,  32,   2,   1, S<4, 32, 2>, S<2, 0, 1>, S<1, 0, 2>,     1,    2,    4, false, S<4, 32, 2>, S<2, 0, 1>, S<1, 0, 2>,     1,    2,    4, false,   1,   1, S<1, 16, 1, 8>,    2>,
    // M=128 N=64: 2x1 waves, 64 AccVGPRs/wave
    DeviceGroupedConvBwdWeight_Xdl_WaveletModel_CShuffleV3<NDimSpatial, ALayout, BLayout, ELayout, F16, F16, F16, F32, PassThrough, PassThrough, PassThrough, ConvSpec,    1,      256,     128,   128,  64,  32, 8,  32,  32,   2,   2, S<4, 32, 2>, S<2, 0, 1>, S<1, 0, 2>,     1,    4,    4, false, S<4, 32, 2>, S<2, 0, 1>, S<1, 0, 2>,     1,    2,    4, false,   1,   1, S<1, 16, 1, 8>,    2>,
    // M=64 N=128: 1x2 waves, 64 AccVGPRs/wave
    DeviceGroupedConvBwdWeight_Xdl_WaveletModel_CShuffleV3<NDimSpatial, ALayout, BLayout, ELayout, F16, F16, F16, F32, PassThrough, PassThrough, PassThrough, ConvSpec,    1,      256,     128,    64, 128,  32, 8,  32,  32,   2,   2, S<4, 32, 2>, S<2, 0, 1>, S<1, 0, 2>,     1,    2,    4, false, S<4, 32, 2>, S<2, 0, 1>, S<1, 0, 2>,     1,    4,    4, false,   1,   1, S<1, 16, 1, 8>,    2>
    // clang-format on
    >;

// BF16 wavelet instances — same tile configs as F16, with BF16 in/out and F32 compute.
template <ck::index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename ELayout,
          ConvolutionBackwardWeightSpecialization ConvSpec>
using device_grouped_conv_bwd_weight_wavelet_xdl_c_shuffle_bf16_instances = std::tuple<
    // clang-format off
    //                                                                                                                                                                   |Prefetch|TileLoad|TileMath|  M |  N | K0|K1| MXdl|NXdl|MXdl|NXdl|
    //                                                                                                                                                                   | Stage  | Thrds  | Thrds  |    |    |   |  |  XDL| XDL| /Wv| /Wv| A cluster     |A ArrangeOrd|A SrcAccOrd | AVDim|ASSca|ADSca|AExM| B cluster     |B ArrangeOrd|B SrcAccOrd | BVDim|BSSca|BDSca|BExN|CShM|CShN|  C cluster       |CSca|
    // M=128 N=128: 2x2 waves, 64 AccVGPRs/wave
    DeviceGroupedConvBwdWeight_Xdl_WaveletModel_CShuffleV3<NDimSpatial, ALayout, BLayout, ELayout, BF16, BF16, BF16, F32, PassThrough, PassThrough, PassThrough, ConvSpec,    1,      256,     256,   128, 128,  32, 8,  32,  32,   2,   2, S<4, 32, 2>, S<2, 0, 1>, S<1, 0, 2>,     1,    4,    4, false, S<4, 32, 2>, S<2, 0, 1>, S<1, 0, 2>,     1,    4,    4, false,   1,   1, S<1, 32, 1, 8>,    2>,
    // M=64 N=64: 2x2 waves, 16 AccVGPRs/wave
    DeviceGroupedConvBwdWeight_Xdl_WaveletModel_CShuffleV3<NDimSpatial, ALayout, BLayout, ELayout, BF16, BF16, BF16, F32, PassThrough, PassThrough, PassThrough, ConvSpec,    1,      256,     256,    64,  64,  32, 8,  32,  32,   1,   1, S<4, 32, 2>, S<2, 0, 1>, S<1, 0, 2>,     1,    2,    4, false, S<4, 32, 2>, S<2, 0, 1>, S<1, 0, 2>,     1,    2,    4, false,   1,   1, S<1, 32, 1, 8>,    2>,
    // M=128 N=64: 2x2 waves, 32 AccVGPRs/wave
    DeviceGroupedConvBwdWeight_Xdl_WaveletModel_CShuffleV3<NDimSpatial, ALayout, BLayout, ELayout, BF16, BF16, BF16, F32, PassThrough, PassThrough, PassThrough, ConvSpec,    1,      256,     256,   128,  64,  32, 8,  32,  32,   2,   1, S<4, 32, 2>, S<2, 0, 1>, S<1, 0, 2>,     1,    4,    4, false, S<4, 32, 2>, S<2, 0, 1>, S<1, 0, 2>,     1,    2,    4, false,   1,   1, S<1, 32, 1, 8>,    2>,
    // M=64 N=128: 2x2 waves, 32 AccVGPRs/wave
    DeviceGroupedConvBwdWeight_Xdl_WaveletModel_CShuffleV3<NDimSpatial, ALayout, BLayout, ELayout, BF16, BF16, BF16, F32, PassThrough, PassThrough, PassThrough, ConvSpec,    1,      256,     256,    64, 128,  32, 8,  32,  32,   1,   2, S<4, 32, 2>, S<2, 0, 1>, S<1, 0, 2>,     1,    2,    4, false, S<4, 32, 2>, S<2, 0, 1>, S<1, 0, 2>,     1,    4,    4, false,   1,   1, S<1, 32, 1, 8>,    2>
    // clang-format on
    >;

// BF16 (4,2) wavelet instances — same tile configs as F16 (4,2).
template <ck::index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename ELayout,
          ConvolutionBackwardWeightSpecialization ConvSpec>
using device_grouped_conv_bwd_weight_wavelet_4w2_xdl_c_shuffle_bf16_instances = std::tuple<
    // clang-format off
    //                                                                                                                                                                   |Prefetch|TileLoad|TileMath|  M |  N | K0|K1| MXdl|NXdl|MXdl|NXdl|
    //                                                                                                                                                                   | Stage  | Thrds  | Thrds  |    |    |   |  |  XDL| XDL| /Wv| /Wv| A cluster     |A ArrangeOrd|A SrcAccOrd | AVDim|ASSca|ADSca|AExM| B cluster     |B ArrangeOrd|B SrcAccOrd | BVDim|BSSca|BDSca|BExN|CShM|CShN|  C cluster       |CSca|
    // M=64 N=64: 1x2 waves, 32 AccVGPRs/wave
    DeviceGroupedConvBwdWeight_Xdl_WaveletModel_CShuffleV3<NDimSpatial, ALayout, BLayout, ELayout, BF16, BF16, BF16, F32, PassThrough, PassThrough, PassThrough, ConvSpec,    1,      256,     128,    64,  64,  32, 8,  32,  32,   2,   1, S<4, 32, 2>, S<2, 0, 1>, S<1, 0, 2>,     1,    2,    4, false, S<4, 32, 2>, S<2, 0, 1>, S<1, 0, 2>,     1,    2,    4, false,   1,   1, S<1, 16, 1, 8>,    2>,
    // M=128 N=64: 2x1 waves, 64 AccVGPRs/wave
    DeviceGroupedConvBwdWeight_Xdl_WaveletModel_CShuffleV3<NDimSpatial, ALayout, BLayout, ELayout, BF16, BF16, BF16, F32, PassThrough, PassThrough, PassThrough, ConvSpec,    1,      256,     128,   128,  64,  32, 8,  32,  32,   2,   2, S<4, 32, 2>, S<2, 0, 1>, S<1, 0, 2>,     1,    4,    4, false, S<4, 32, 2>, S<2, 0, 1>, S<1, 0, 2>,     1,    2,    4, false,   1,   1, S<1, 16, 1, 8>,    2>,
    // M=64 N=128: 1x2 waves, 64 AccVGPRs/wave
    DeviceGroupedConvBwdWeight_Xdl_WaveletModel_CShuffleV3<NDimSpatial, ALayout, BLayout, ELayout, BF16, BF16, BF16, F32, PassThrough, PassThrough, PassThrough, ConvSpec,    1,      256,     128,    64, 128,  32, 8,  32,  32,   2,   2, S<4, 32, 2>, S<2, 0, 1>, S<1, 0, 2>,     1,    2,    4, false, S<4, 32, 2>, S<2, 0, 1>, S<1, 0, 2>,     1,    4,    4, false,   1,   1, S<1, 16, 1, 8>,    2>
    // clang-format on
    >;

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
