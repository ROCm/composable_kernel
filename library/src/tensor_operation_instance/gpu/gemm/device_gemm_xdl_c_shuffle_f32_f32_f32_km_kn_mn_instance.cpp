#include <stdlib.h>
#include "config.hpp"
#include "device_gemm_xdl_c_shuffle.hpp"
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

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

// Compilation parameters for a[k, m] * b[k, n] = c[m, n]
using device_gemm_xdl_c_shuffle_f32_f32_f32_km_kn_mn_instances = std::tuple<
    // clang-format off
        //#####################|AData| BData| CData| AccData| CShuffle| ALayout| BLayout| CLayout|           A|           B|           C| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|     CBlockTransferClusterLengths|  CBlockTransfer|
        //#####################| Type|  Type|  Type|    Type| DataType|        |        |        | Elementwise| Elementwise| Elementwise|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave| _MBlock_MXdlPerWave_MWaveMPerXdl| ScalarPerVector|
        //#####################|     |      |      |        |         |        |        |        |   Operation|   Operation|   Operation|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle| _NBlock_NXdlPerWave_NWaveNPerXdl|   _NWaveNPerXdl|
        //#####################|     |      |      |        |         |        |        |        |            |            |            |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                                 |                |
        DeviceGemmXdl_C_Shuffle<  F32,   F32,   F32,     F32,      F32,     Col,      Row,    Row, PassThrough, PassThrough, PassThrough,   256,   256,   128,    32,   2,   2,   32,   32,    4,    2,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              2,     false,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,     false,           1,           1,             S<1, 1, 32, 1, 1, 8>,               4>,
        DeviceGemmXdl_C_Shuffle<  F32,   F32,   F32,     F32,      F32,     Col,      Row,    Row, PassThrough, PassThrough, PassThrough,   256,   256,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              8,      true,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,      true,           1,           1,             S<1, 1, 32, 1, 1, 8>,               4>,
        DeviceGemmXdl_C_Shuffle<  F32,   F32,   F32,     F32,      F32,     Col,      Row,    Row, PassThrough, PassThrough, PassThrough,   256,   128,   256,    32,   4,   4,   32,   32,    2,    4,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              2,     false,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,     false,           1,           1,             S<1, 1, 32, 1, 1, 8>,               4>,
        DeviceGemmXdl_C_Shuffle<  F32,   F32,   F32,     F32,      F32,     Col,      Row,    Row, PassThrough, PassThrough, PassThrough,   256,   128,   256,    32,   8,   8,   32,   32,    2,    4,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              2,              8,      true,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              8,      true,           1,           1,             S<1, 1, 32, 1, 1, 8>,               4>,
        DeviceGemmXdl_C_Shuffle<  F32,   F32,   F32,     F32,      F32,     Col,      Row,    Row, PassThrough, PassThrough, PassThrough,   128,   128,   128,    32,   2,   2,   32,   32,    4,    2,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              2,     false,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,     false,           1,           1,             S<1, 1, 16, 1, 1, 8>,               4>,
        DeviceGemmXdl_C_Shuffle<  F32,   F32,   F32,     F32,      F32,     Col,      Row,    Row, PassThrough, PassThrough, PassThrough,   128,   128,   128,    32,   8,   8,   32,   32,    4,    2,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              8,      true,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              8,      true,           1,           1,             S<1, 1, 16, 1, 1, 8>,               4>,
        DeviceGemmXdl_C_Shuffle<  F32,   F32,   F32,     F32,      F32,     Col,      Row,    Row, PassThrough, PassThrough, PassThrough,   256,   128,   128,    32,   2,   2,   32,   32,    2,    2,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              2,     false,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,     false,           1,           1,             S<1, 1, 32, 1, 1, 8>,               4>,
        DeviceGemmXdl_C_Shuffle<  F32,   F32,   F32,     F32,      F32,     Col,      Row,    Row, PassThrough, PassThrough, PassThrough,   256,   128,   128,    32,   8,   8,   32,   32,    2,    2,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              2,              8,      true,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,      true,           1,           1,             S<1, 1, 32, 1, 1, 8>,               4>,
        DeviceGemmXdl_C_Shuffle<  F32,   F32,   F32,     F32,      F32,     Col,      Row,    Row, PassThrough, PassThrough, PassThrough,   128,   128,    64,    32,   2,   2,   32,   32,    2,    2,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              2,     false,     S<4, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,     false,           1,           1,             S<1, 1, 32, 1, 1, 4>,               4>,
        DeviceGemmXdl_C_Shuffle<  F32,   F32,   F32,     F32,      F32,     Col,      Row,    Row, PassThrough, PassThrough, PassThrough,   128,   128,    64,    32,   8,   8,   32,   32,    2,    2,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              8,      true,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,      true,           1,           1,             S<1, 1, 32, 1, 1, 4>,               4>,
        DeviceGemmXdl_C_Shuffle<  F32,   F32,   F32,     F32,      F32,     Col,      Row,    Row, PassThrough, PassThrough, PassThrough,   128,    64,   128,    32,   2,   2,   32,   32,    2,    2,     S<8, 16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              2,     false,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,     false,           1,           1,             S<1, 1, 16, 1, 1, 8>,               4>,
        DeviceGemmXdl_C_Shuffle<  F32,   F32,   F32,     F32,      F32,     Col,      Row,    Row, PassThrough, PassThrough, PassThrough,   128,    64,   128,    32,   8,   8,   32,   32,    2,    2,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              2,              8,      true,     S<4, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              8,      true,           1,           1,             S<1, 1, 16, 1, 1, 8>,               4>,
        DeviceGemmXdl_C_Shuffle<  F32,   F32,   F32,     F32,      F32,     Col,      Row,    Row, PassThrough, PassThrough, PassThrough,   256,   128,    64,    32,   2,   2,   32,   32,    2,    1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              2,     false,     S<16,16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,     false,           1,           1,             S<1, 1, 16, 1, 1, 4>,               4>,
        DeviceGemmXdl_C_Shuffle<  F32,   F32,   F32,     F32,      F32,     Col,      Row,    Row, PassThrough, PassThrough, PassThrough,   256,   128,    64,    32,   8,   8,   32,   32,    2,    1,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              2,              8,      true,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              1,              8,      true,           1,           1,             S<1, 1, 16, 1, 1, 4>,               4>,
        DeviceGemmXdl_C_Shuffle<  F32,   F32,   F32,     F32,      F32,     Col,      Row,    Row, PassThrough, PassThrough, PassThrough,   256,    64,   128,    32,   2,   2,   32,   32,    1,    2,     S<16,16, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              2,     false,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              2,     false,           1,           1,             S<1, 1, 32, 1, 1, 8>,               4>,
        DeviceGemmXdl_C_Shuffle<  F32,   F32,   F32,     F32,      F32,     Col,      Row,    Row, PassThrough, PassThrough, PassThrough,   256,    64,   128,    32,   8,   8,   32,   32,    1,    2,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              1,              8,      true,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,      true,           1,           1,             S<1, 1, 32, 1, 1, 8>,               4>
    // clang-format on
    >;

void add_device_gemm_xdl_c_shuffle_f32_f32_f32_km_kn_mn_instances(
    std::vector<DeviceGemmPtr<PassThrough, PassThrough, PassThrough>>& instances)
{
    add_device_operation_instances(instances,
                                   device_gemm_xdl_c_shuffle_f32_f32_f32_km_kn_mn_instances{});
}

} // namespace device_gemm_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
