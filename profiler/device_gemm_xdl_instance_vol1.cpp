#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>
#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "gemm_common.hpp"
#include "host_gemm.hpp"
#include "device_tensor.hpp"
#include "device_base.hpp"
#include "device_gemm_xdl.hpp"
#include "device_gemm_xdl_instance.hpp"

namespace ck {
namespace profiler {
namespace device_gemm_xdl_instance {

// Compilation parameters for a[m, k] * b[n, k] = c[m, n]
using device_gemm_xdl_instance_f16_f16_f16_mk_nk_mn_vol1 = std::tuple<
    // clang-format off
    //########################################| AData| BData| CData| AccData| ALayout| BLayout| CLayout| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl| ABlockTransferThread|   ABlockTransferThread| ABlockTransferThread| ABlockTransfer|       ABlock|          ABlock| ABlockTransfer|  BBlockTransfer|  BBlockTransfer|  BBlockTransfer|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| CThreadTransfer| CThreadTransfer| ABlockLds| BBlockLds|
    //########################################|  Type|  Type|  Type|    Type|        |        |        |  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per| SliceLengths_K0_M_K1| ClusterLengths_K0_M_K1|  ClusterArrangeOrder| SrcAccessOrder|  TransferSrc|     TransferSrc|   DstScalarPer|     ThreadSlice|   ThreadCluster|   ThreadCluster|  SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| SrcDstVectorDim|       DstScalar| AddExtraM| AddExtraN|
    //########################################|      |      |      |        |        |        |        |      |      |      |      |   |     |     | Wave| Wave|                     |                       |                     |               |    VectorDim| ScalarPerVector|               |                |       Vector_K1| Lengths_K0_N_K1| Lengths_K0_N_K1|   ArrangeOrder|      PerVector|   PerVector_K1|                |       PerVector|          |          |
    ck::tensor_operation::device::DeviceGemmXdl<  F16,   F16,   F16,     F32,     Row,      Col,    Row,   256,   256,   128,     4,  8,   32,   32,    4,    2,           S<1, 4, 8>,            S<4, 64, 1>,           S<1, 0, 2>,     S<1, 0, 2>,            2,               8,              8,      S<1, 2, 8>,     S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,              2,              8,              8,               7,               1,      true,      true>,
    ck::tensor_operation::device::DeviceGemmXdl<  F16,   F16,   F16,     F32,     Row,      Col,    Row,   256,   128,   128,     4,  8,   32,   32,    2,    2,           S<1, 2, 8>,            S<4, 64, 1>,           S<1, 0, 2>,     S<1, 0, 2>,            2,               8,              8,      S<1, 2, 8>,     S<4, 64, 1>,      S<1, 0, 2>,      S<1, 0, 2>,              2,              8,              8,               7,               1,      true,      true>
    // clang-format on
    >;

void add_device_gemms_xdl_instnace_f16_f16_f16_mk_nk_mn_vol1(
    std::vector<DeviceOpCombo>& device_op_combos,
    ck::half_t* p_a,
    ck::half_t* p_b,
    ck::half_t* p_c,
    int M,
    int N,
    int K,
    int StrideA,
    int StrideB,
    int StrideC)
{
    using DeviceGemms =
        device_gemm_xdl_instance::device_gemm_xdl_instance_f16_f16_f16_mk_nk_mn_vol1;

    const auto device_gemms = DeviceGemms{};

    ck::static_for<0, std::tuple_size_v<DeviceGemms>, 1>{}([&](auto i) {
        using Gemm         = remove_cvref_t<decltype(std::get<i>(device_gemms))>;
        using GemmInvoker  = typename Gemm::Invoker;
        using GemmArgument = typename Gemm::Argument;

        auto gemm     = Gemm{};
        auto invoker  = gemm.MakeInvoker();
        auto argument = gemm.MakeArgument(p_a, p_b, p_c, M, N, K, StrideA, StrideB, StrideC);

        device_op_combos.push_back(std::make_tuple(std::make_unique<Gemm>(gemm),
                                                   std::make_unique<GemmInvoker>(invoker),
                                                   std::make_unique<GemmArgument>(argument)));
    });
}

} // namespace device_gemm_xdl_instance
} // namespace profiler
} // namespace ck
