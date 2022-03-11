#include <algorithm>
#include <cstdlib>
#include <half.hpp>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

#include "gemm_util.hpp"
#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_gemm.hpp"
#include "device_tensor.hpp"
#include "device_gemm_xdl.hpp"
#include "device_gemm_xdl_c_shuffle.hpp"
#include "element_wise_operation.hpp"
#include "gemm_specialization.hpp"

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using DeviceGemmPtr_ =
    ck::tensor_operation::device::DeviceGemmPtr<ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough>;

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_gemm_instance {
void add_device_gemm_xdl_f16_f16_f16_mk_nk_mn_instances(std::vector<DeviceGemmPtr_>&);
}
} // namespace device
} // namespace tensor_operation
} // namespace ck

int main()
{
    using ADataType = ck::half_t;
    using BDataType = ck::half_t;
    using CDataType = ck::half_t;

    using ALayout = ck::tensor_layout::gemm::RowMajor;
    using BLayout = ck::tensor_layout::gemm::ColumnMajor;
    using CLayout = ck::tensor_layout::gemm::RowMajor;

    std::vector<DeviceGemmPtr_> gemmPtrs;
    ck::tensor_operation::device::device_gemm_instance::
        add_device_gemm_xdl_f16_f16_f16_mk_nk_mn_instances(gemmPtrs);

    bool res = true;

    for(auto& gemmPtr : gemmPtrs)
    {
        res &= ck::gemm_util::TestGemm<DeviceGemmPtr_,
                                       ADataType,
                                       BDataType,
                                       CDataType,
                                       ALayout,
                                       BLayout,
                                       CLayout,
                                       PassThrough,
                                       PassThrough,
                                       PassThrough>{}(gemmPtr);
    }

    std::cout << "TestGemm ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;
}
