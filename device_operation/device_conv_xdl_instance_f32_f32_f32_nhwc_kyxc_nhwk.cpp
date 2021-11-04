#include <stdlib.h>
#include "config.hpp"
#include "device_gemm_xdl.hpp"
#include "device_gemm_xdl_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_conv_xdl_instance {

using F16 = ck::half_t;
using F32 = float;

using NHWC = ck::tensor_layout::convolution::NHWC;
using KYXC = ck::tensor_layout::convolution::KYXC;
using NHWK = ck::tensor_layout::convolution::NHWK;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

// Compilation parameters for a[m, k] * b[n, k] = c[m, n]
using device_conv_xdl_instances_f32_f32_f32_nhwc_kyxc_nhwk =
    std::tuple<
        // clang-format off
        //##########| InData| WeiData| OutData| AccData|     In|    Wei|    Out| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer|  BBlockTransfer|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| CThreadTransfer| CThreadTransfer| ABlockLds| BBlockLds|
        //##########|   Type|    Type|    Type|    Type| Layout| Layout| Layout|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|     ThreadSlice|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|     ThreadSlice|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| SrcDstVectorDim|       DstScalar| AddExtraM| AddExtraN|
        //##########|       |        |        |        |       |       |       |      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_N_K1| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1| Lengths_K0_N_K1| Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|                |       PerVector|          |          |
        //##########|       |        |        |        |       |       |       |      |      |      |      |   |     |     |     |     |                |                |               |               |               |               |               |                |                |               |               |              |               |               |                |                |          |          |
        DeviceConvXdl<   F32,     F32,     F32,     F32,   NHWC,   KYXC,   NHWK,   256,   256,   128,     4,  4,   32,   32,    4,    2,      S<1, 4, 4>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      S<1, 2, 4>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,               7,               1,      true,      true>,
        DeviceConvXdl<   F32,     F32,     F32,     F32,   NHWC,   KYXC,   NHWK,   256,   128,   256,     4,  4,   32,   32,    2,    4,      S<1, 2, 4>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      S<1, 4, 4>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,               7,               1,      true,      true>,
        DeviceConvXdl<   F32,     F32,     F32,     F32,   NHWC,   KYXC,   NHWK,   128,   128,   128,     4,  4,   32,   32,    4,    2,      S<1, 4, 4>,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      S<1, 4, 4>,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,               7,               1,      true,      true>,
        DeviceConvXdl<   F32,     F32,     F32,     F32,   NHWC,   KYXC,   NHWK,   256,   128,   128,     4,  4,   32,   32,    2,    2,      S<1, 2, 4>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      S<1, 2, 4>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,               7,               1,      true,      true>,
        DeviceConvXdl<   F32,     F32,     F32,     F32,   NHWC,   KYXC,   NHWK,   128,   128,    64,     4,  4,   32,   32,    2,    2,      S<1, 4, 4>,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      S<1, 2, 4>,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,               7,               1,      true,      true>,
        DeviceConvXdl<   F32,     F32,     F32,     F32,   NHWC,   KYXC,   NHWK,   128,    64,   128,     4,  4,   32,   32,    2,    2,      S<1, 2, 4>,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      S<1, 4, 4>,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,               7,               1,      true,      true>,
        DeviceConvXdl<   F32,     F32,     F32,     F32,   NHWC,   KYXC,   NHWK,   256,   128,    64,     4,  4,   32,   32,    2,    1,      S<1, 2, 4>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      S<1, 1, 4>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,               7,               1,      true,      true>,
        DeviceConvXdl<   F32,     F32,     F32,     F32,   NHWC,   KYXC,   NHWK,   256,    64,   128,     4,  4,   32,   32,    1,    2,      S<1, 1, 4>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      S<1, 2, 4>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,               7,               1,      true,      true>
        // clang-format on
        >;

template <>
void add_device_conv_xdl_instance<F32, F32, F32, NHWC, KYXC, NHWK>(
    std::vector<DeviceConvPtr>& device_conv_instances)
{
    using DeviceConvs = device_gemm_xdl_instances_f32_f32_f32_nhwc_kyxc_nhwk;

    const auto device_convs = DeviceConvs{};

    ck::static_for<0, std::tuple_size_v<DeviceConvs>, 1>{}([&](auto i) {
        using Conv = remove_cvref_t<decltype(std::get<i>(device_convs))>;

        auto conv = Conv{};

        device_conv_instances.push_back(std::make_unique<Conv>(conv));
    });
}

} // namespace device_conv_xdl_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
