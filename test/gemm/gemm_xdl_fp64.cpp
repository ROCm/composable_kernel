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
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "host_gemm.hpp"
#include "device_tensor.hpp"
#include "device_gemm_xdl.hpp"
#include "element_wise_operation.hpp"
#include "reference_gemm.hpp"
#include "gemm_specialization.hpp"

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using DeviceGemmNoOpPtr =
    ck::tensor_operation::device::DeviceGemmPtr<ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough>;

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_gemm_instance {
void add_device_gemm_xdl_f64_f64_f64_km_kn_mn_instances(std::vector<DeviceGemmNoOpPtr>&);
void add_device_gemm_xdl_f64_f64_f64_km_nk_mn_instances(std::vector<DeviceGemmNoOpPtr>&);
void add_device_gemm_xdl_f64_f64_f64_mk_nk_mn_instances(std::vector<DeviceGemmNoOpPtr>&);
void add_device_gemm_xdl_f64_f64_f64_mk_kn_mn_instances(std::vector<DeviceGemmNoOpPtr>&);

} // namespace device_gemm_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

inline std::string get_device_name()
{
    hipDeviceProp_t props{};
    int device;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
    {
        return std::string();
    }

    status = hipGetDeviceProperties(&props, device);
    if(status != hipSuccess)
    {
        return std::string();
    }
    const std::string name(props.gcnArchName);

    return name;
}

int main()
{
    if(get_device_name().find("gfx90a") == std::string::npos)
    {
        std::cout << "TestGemm ..... SUCCESS" << std::endl;
        return 0;
    }
    using ADataType   = double;
    using BDataType   = double;
    using CDataType   = double;
    using AccDataType = double;

    using RowMajor    = ck::tensor_layout::gemm::RowMajor;
    using ColumnMajor = ck::tensor_layout::gemm::ColumnMajor;

    bool res = true;
    std::vector<DeviceGemmNoOpPtr> gemmPtrs;
    ck::tensor_operation::device::device_gemm_instance::
        add_device_gemm_xdl_f64_f64_f64_km_kn_mn_instances(gemmPtrs);

    for(auto& gemmPtr : gemmPtrs)
    {
        res &= ck::gemm_util::TestGemm<DeviceGemmNoOpPtr,
                                       ADataType,
                                       BDataType,
                                       CDataType,
                                       AccDataType,
                                       ColumnMajor,
                                       RowMajor,
                                       RowMajor,
                                       PassThrough,
                                       PassThrough,
                                       PassThrough>{}(gemmPtr);
    }

    gemmPtrs.clear();
    ck::tensor_operation::device::device_gemm_instance::
        add_device_gemm_xdl_f64_f64_f64_km_nk_mn_instances(gemmPtrs);

    for(auto& gemmPtr : gemmPtrs)
    {
        res &= ck::gemm_util::TestGemm<DeviceGemmNoOpPtr,
                                       ADataType,
                                       BDataType,
                                       CDataType,
                                       AccDataType,
                                       ColumnMajor,
                                       ColumnMajor,
                                       RowMajor,
                                       PassThrough,
                                       PassThrough,
                                       PassThrough>{}(gemmPtr);
    }

    gemmPtrs.clear();
    ck::tensor_operation::device::device_gemm_instance::
        add_device_gemm_xdl_f64_f64_f64_mk_kn_mn_instances(gemmPtrs);

    for(auto& gemmPtr : gemmPtrs)
    {
        res &= ck::gemm_util::TestGemm<DeviceGemmNoOpPtr,
                                       ADataType,
                                       BDataType,
                                       CDataType,
                                       AccDataType,
                                       RowMajor,
                                       RowMajor,
                                       RowMajor,
                                       PassThrough,
                                       PassThrough,
                                       PassThrough>{}(gemmPtr);
    }

    gemmPtrs.clear();
    ck::tensor_operation::device::device_gemm_instance::
        add_device_gemm_xdl_f64_f64_f64_mk_nk_mn_instances(gemmPtrs);

    for(auto& gemmPtr : gemmPtrs)
    {
        res &= ck::gemm_util::TestGemm<DeviceGemmNoOpPtr,
                                       ADataType,
                                       BDataType,
                                       CDataType,
                                       AccDataType,
                                       RowMajor,
                                       ColumnMajor,
                                       RowMajor,
                                       PassThrough,
                                       PassThrough,
                                       PassThrough>{}(gemmPtr);
    }
    std::cout << "TestGemm ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;
    return res ? 0 : 1;
}
