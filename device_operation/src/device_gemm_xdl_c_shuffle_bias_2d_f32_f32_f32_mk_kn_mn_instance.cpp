#include <stdlib.h>
#include "config.hpp"
#include "device_gemm_xdl_c_shuffle_bias_2d.hpp"
#include "element_wise_operation.hpp"
#include "device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_gemm_instance {

using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough  = ck::tensor_operation::element_wise::PassThrough;
using AlphaBetaAdd = ck::tensor_operation::element_wise::AlphaBetaAdd;

// Compilation parameters for a[m, k] * b[k, n] = c[m, n]
using device_gemm_xdl_c_shuffle_bias_2d_f32_f32_f32_mk_kn_mn_instances = std::tuple<
    // clang-format off
        //#############################|AData| BData| CData| AccData| ALayout| BLayout| CLayout|           A|           B|            C| Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|     CBlockTransferClusterLengths|  CBlockTransfer|
        //#############################| Type|  Type|  Type|    Type|        |        |        | Elementwise| Elementwise|  Elementwise|  Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave| _MBlock_MXdlPerWave_MWaveMPerXdl| ScalarPerVector|
        //#############################|     |      |      |        |        |        |        |   Operation|   Operation|    Operation|      |      |      |      |   |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle| _NBlock_NXdlPerWave_NWaveNPerXdl|   _NWaveNPerXdl|
        //#############################|     |      |      |        |        |        |        |            |            |             |      |      |      |      |   |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                                 |                |
        DeviceGemmXdl_C_Shuffle_Bias_2d<  F32,   F32,   F32,     F32,     Row,      Row,    Row, PassThrough, PassThrough, AlphaBetaAdd,   256,   256,   128,     4,  4,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              4,      true,           1,           1,             S<1, 1, 32, 1, 1, 8>,               4>,
        DeviceGemmXdl_C_Shuffle_Bias_2d<  F32,   F32,   F32,     F32,     Row,      Row,    Row, PassThrough, PassThrough, AlphaBetaAdd,   256,   128,   256,     4,  4,   32,   32,    2,    4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              4,      true,           1,           1,             S<1, 1, 32, 1, 1, 8>,               4>,
        DeviceGemmXdl_C_Shuffle_Bias_2d<  F32,   F32,   F32,     F32,     Row,      Row,    Row, PassThrough, PassThrough, AlphaBetaAdd,   128,   128,   128,     4,  4,   32,   32,    4,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              4,      true,           1,           1,             S<1, 1, 16, 1, 1, 8>,               4>,
        DeviceGemmXdl_C_Shuffle_Bias_2d<  F32,   F32,   F32,     F32,     Row,      Row,    Row, PassThrough, PassThrough, AlphaBetaAdd,   256,   128,   128,     4,  4,   32,   32,    2,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              4,      true,           1,           1,             S<1, 1, 32, 1, 1, 8>,               4>,
        DeviceGemmXdl_C_Shuffle_Bias_2d<  F32,   F32,   F32,     F32,     Row,      Row,    Row, PassThrough, PassThrough, AlphaBetaAdd,   128,   128,    64,     4,  4,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              4,      true,           1,           1,             S<1, 1, 32, 1, 1, 4>,               4>,
        DeviceGemmXdl_C_Shuffle_Bias_2d<  F32,   F32,   F32,     F32,     Row,      Row,    Row, PassThrough, PassThrough, AlphaBetaAdd,   128,    64,   128,     4,  4,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              4,      true,           1,           1,             S<1, 1, 16, 1, 1, 8>,               4>,
        DeviceGemmXdl_C_Shuffle_Bias_2d<  F32,   F32,   F32,     F32,     Row,      Row,    Row, PassThrough, PassThrough, AlphaBetaAdd,   256,   128,    64,     4,  4,   32,   32,    2,    1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              4,      true,           1,           1,             S<1, 1, 16, 1, 1, 4>,               4>,
        DeviceGemmXdl_C_Shuffle_Bias_2d<  F32,   F32,   F32,     F32,     Row,      Row,    Row, PassThrough, PassThrough, AlphaBetaAdd,   256,    64,   128,     4,  4,   32,   32,    1,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,      true,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              4,      true,           1,           1,             S<1, 1, 32, 1, 1, 8>,               4>
    // clang-format on
    >;

void add_device_gemm_xdl_c_shuffle_bias_2d_f32_f32_f32_mk_kn_mn_instances(
    std::vector<DeviceGemmBiasPtr<PassThrough, PassThrough, AlphaBetaAdd>>& instances)
{
    add_device_operation_instances(
        instances, device_gemm_xdl_c_shuffle_bias_2d_f32_f32_f32_mk_kn_mn_instances{});
}

} // namespace device_gemm_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
