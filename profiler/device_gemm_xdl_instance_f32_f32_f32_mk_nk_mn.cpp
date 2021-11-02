#include <stdlib.h>
#include <half.hpp>
#include "config.hpp"
#include "device_gemm_xdl.hpp"
#include "device_gemm_xdl_instance.hpp"

namespace ck {
namespace profiler {
namespace device_gemm_xdl_instance {

// Compilation parameters for a[m, k] * b[n, k] = c[m, n]
using device_gemm_xdl_instance_f32_f32_f32_mk_nk_mn =
    std::tuple<
        // clang-format off
        //########################################| AData| BData| CData| AccData| ALayout| BLayout| CLayout| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer|  BBlockTransfer|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| CThreadTransfer| CThreadTransfer| ABlockLds| BBlockLds|
        //########################################|  Type|  Type|  Type|    Type|        |        |        |  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|     ThreadSlice|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|     ThreadSlice|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| SrcDstVectorDim|       DstScalar| AddExtraM| AddExtraN|
        //########################################|      |      |      |        |        |        |        |      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_N_K1| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1| Lengths_K0_N_K1| Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|                |       PerVector|          |          |
        //########################################|      |      |      |        |        |        |        |      |      |      |      |   |     |     |     |     |                |                |               |               |               |               |               |                |                |               |               |              |               |               |                |                |          |          |
        ck::tensor_operation::device::DeviceGemmXdl<  F32,   F32,   F32,     F32,     Row,     Col,     Row,   256,   256,   128,     4,  4,   32,   32,    4,    2,      S<1, 4, 4>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      S<1, 2, 4>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,               7,               1,      true,      true>,
        ck::tensor_operation::device::DeviceGemmXdl<  F32,   F32,   F32,     F32,     Row,     Col,     Row,   256,   128,   256,     4,  4,   32,   32,    2,    4,      S<1, 2, 4>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      S<1, 4, 4>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,               7,               1,      true,      true>,
        ck::tensor_operation::device::DeviceGemmXdl<  F32,   F32,   F32,     F32,     Row,     Col,     Row,   128,   128,   128,     4,  4,   32,   32,    4,    2,      S<1, 4, 4>,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      S<1, 4, 4>,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,               7,               1,      true,      true>,
        ck::tensor_operation::device::DeviceGemmXdl<  F32,   F32,   F32,     F32,     Row,     Col,     Row,   256,   128,   128,     4,  4,   32,   32,    2,    2,      S<1, 2, 4>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      S<1, 2, 4>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,               7,               1,      true,      true>,
        ck::tensor_operation::device::DeviceGemmXdl<  F32,   F32,   F32,     F32,     Row,     Col,     Row,   128,   128,    64,     4,  4,   32,   32,    2,    2,      S<1, 4, 4>,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      S<1, 2, 4>,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,               7,               1,      true,      true>,
        ck::tensor_operation::device::DeviceGemmXdl<  F32,   F32,   F32,     F32,     Row,     Col,     Row,   128,    64,   128,     4,  4,   32,   32,    2,    2,      S<1, 2, 4>,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      S<1, 4, 4>,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,               7,               1,      true,      true>,
        ck::tensor_operation::device::DeviceGemmXdl<  F32,   F32,   F32,     F32,     Row,     Col,     Row,   256,   128,    64,     4,  4,   32,   32,    2,    1,      S<1, 2, 4>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      S<1, 1, 4>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,               7,               1,      true,      true>,
        ck::tensor_operation::device::DeviceGemmXdl<  F32,   F32,   F32,     F32,     Row,     Col,     Row,   256,    64,   128,     4,  4,   32,   32,    1,    2,      S<1, 1, 4>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      S<1, 2, 4>,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,               7,               1,      true,      true>
        // clang-format on
        >;

template <>
void add_device_gemm_xdl_instance<F32, F32, F32, Row, Col, Row>(
    std::vector<DeviceGemmXdlBaseOpPtr>& device_op_instances)
{
    using DeviceGemms = device_gemm_xdl_instance::device_gemm_xdl_instance_f32_f32_f32_mk_nk_mn;

    const auto device_gemms = DeviceGemms{};

    ck::static_for<0, std::tuple_size_v<DeviceGemms>, 1>{}([&](auto i) {
        using Gemm = remove_cvref_t<decltype(std::get<i>(device_gemms))>;

        auto gemm = Gemm{};

        device_op_instances.push_back(std::make_unique<Gemm>(gemm));
    });
}

} // namespace device_gemm_xdl_instance
} // namespace profiler
} // namespace ck
