#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>

#include "check_err.hpp"
#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "host_gemm.hpp"
#include "device_tensor.hpp"
#include "device_grouped_gemm_xdl.hpp"
#include "element_wise_operation.hpp"
#include "reference_gemm.hpp"
#include "gemm_specialization.hpp"

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using DeviceGroupedGemmPtr_ = ck::tensor_operation::device::DeviceGroupedGemmPtr<
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough>;

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_grouped_gemm_instance {
void add_device_grouped_gemm_xdl_f16_f16_f16_mk_nk_mn_instances(
    std::vector<DeviceGroupedGemmPtr_>&);
}
} // namespace device
} // namespace tensor_operation
} // namespace ck

namespace {

using ADataType   = ck::half_t;
using BDataType   = ck::half_t;
using CDataType   = ck::half_t;
using AccDataType = float;

using ALayout = ck::tensor_layout::gemm::RowMajor;
using BLayout = ck::tensor_layout::gemm::ColumnMajor;
using CLayout = ck::tensor_layout::gemm::RowMajor;

bool TestGroupedGemm(DeviceGroupedGemmPtr_& groupedGemmPtr)
{
    int group_count = rand() % 10 + 1;

    // GEMM shape
    std::vector<ck::tensor_operation::device::GemmShape> gemm_shapes;
    std::vector<const void*> p_a, p_b;
    std::vector<void*> p_c;

    gemm_shapes.reserve(group_count);

    for(int i = 0; i < group_count; i++)
    {
        int M = 256 + 256 * (rand() % 10);
        int N = 256 + 256 * (rand() % 10);
        int K = 128 + 128 * (rand() % 10);

        int AStride = std::is_same<ck::tensor_layout::gemm::RowMajor, ALayout>::value ? K : M;
        int BStride = std::is_same<ck::tensor_layout::gemm::RowMajor, BLayout>::value ? N : K;
        int CStride = std::is_same<ck::tensor_layout::gemm::RowMajor, CLayout>::value ? N : M;

        gemm_shapes.push_back({M, N, K, AStride, BStride, CStride});
    }

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                            std::vector<std::size_t>({stride, 1}));
            }
            else
            {
                return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                            std::vector<std::size_t>({1, stride}));
            }
        };

    std::vector<Tensor<ADataType>> a_tensors;
    ;
    std::vector<Tensor<BDataType>> b_tensors;
    std::vector<Tensor<CDataType>> c_host_tensors;
    std::vector<Tensor<CDataType>> c_device_tensors;

    a_tensors.reserve(group_count);
    b_tensors.reserve(group_count);
    c_host_tensors.reserve(group_count);
    c_device_tensors.reserve(group_count);

    using DeviceMemPtr = std::unique_ptr<DeviceMem>;

    std::vector<DeviceMemPtr> a_tensors_device, b_tensors_device, c_tensors_device;

    a_tensors_device.reserve(group_count);
    b_tensors_device.reserve(group_count);
    c_tensors_device.reserve(group_count);

    for(std::size_t i = 0; i < gemm_shapes.size(); i++)
    {
        a_tensors.emplace_back(Tensor<ADataType>(f_host_tensor_descriptor(
            gemm_shapes[i].M, gemm_shapes[i].K, gemm_shapes[i].StrideA, ALayout{})));
        b_tensors.emplace_back(Tensor<BDataType>(f_host_tensor_descriptor(
            gemm_shapes[i].K, gemm_shapes[i].N, gemm_shapes[i].StrideB, BLayout{})));
        c_host_tensors.emplace_back(Tensor<CDataType>(f_host_tensor_descriptor(
            gemm_shapes[i].M, gemm_shapes[i].N, gemm_shapes[i].StrideC, CLayout{})));
        c_device_tensors.emplace_back(Tensor<CDataType>(f_host_tensor_descriptor(
            gemm_shapes[i].M, gemm_shapes[i].N, gemm_shapes[i].StrideC, CLayout{})));

        a_tensors[i].GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b_tensors[i].GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
    }

    for(std::size_t i = 0; i < gemm_shapes.size(); i++)
    {
        a_tensors_device.emplace_back(
            std::make_unique<DeviceMem>(sizeof(ADataType) * a_tensors[i].mDesc.GetElementSize()));
        b_tensors_device.emplace_back(
            std::make_unique<DeviceMem>(sizeof(BDataType) * b_tensors[i].mDesc.GetElementSize()));
        c_tensors_device.emplace_back(std::make_unique<DeviceMem>(
            sizeof(CDataType) * c_device_tensors[i].mDesc.GetElementSize()));

        a_tensors_device[i]->ToDevice(a_tensors[i].mData.data());
        b_tensors_device[i]->ToDevice(b_tensors[i].mData.data());

        p_a.push_back(a_tensors_device[i]->GetDeviceBuffer());
        p_b.push_back(b_tensors_device[i]->GetDeviceBuffer());
        p_c.push_back(c_tensors_device[i]->GetDeviceBuffer());
    }

    auto a_element_op = PassThrough{};
    auto b_element_op = PassThrough{};
    auto c_element_op = PassThrough{};

    // do GEMM
    auto invoker_ptr = groupedGemmPtr->MakeInvokerPointer();

    auto argument_ptr = groupedGemmPtr->MakeArgumentPointer(
        p_a, p_b, p_c, gemm_shapes, a_element_op, b_element_op, c_element_op);

    DeviceMem gemm_desc_workspace(groupedGemmPtr->GetWorkSpaceSize(argument_ptr.get()));

    groupedGemmPtr->SetWorkSpacePointer(argument_ptr.get(), gemm_desc_workspace.GetDeviceBuffer());

    invoker_ptr->Run(argument_ptr.get());

    for(std::size_t i = 0; i < gemm_shapes.size(); i++)
    {
        c_tensors_device[i]->FromDevice(c_device_tensors[i].mData.data());

        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                                BDataType,
                                                                                CDataType,
                                                                                AccDataType,
                                                                                PassThrough,
                                                                                PassThrough,
                                                                                PassThrough>;

        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(a_tensors[i],
                                                  b_tensors[i],
                                                  c_host_tensors[i],
                                                  a_element_op,
                                                  b_element_op,
                                                  c_element_op);

        if(!groupedGemmPtr->IsSupportedArgument(argument_ptr.get()))
        {
            return false;
        }

        ref_invoker.Run(ref_argument);

        bool res = ck::utils::check_err(c_host_tensors[i].mData, c_device_tensors[i].mData);

        std::cout << "group_id: " << i << (res ? " SUCCESS" : " FAILURE") << std::endl;

        if(!res)
            return false;
    }

    return true;
}

} // anonymous namespace

int main()
{
    std::vector<DeviceGroupedGemmPtr_> groupedGemmPtrs;
    ck::tensor_operation::device::device_grouped_gemm_instance::
        add_device_grouped_gemm_xdl_f16_f16_f16_mk_nk_mn_instances(groupedGemmPtrs);

    bool res = true;

    for(auto& gemmPtr : groupedGemmPtrs)
    {
        res &= TestGroupedGemm(gemmPtr);
    }

    std::cout << "TestGroupedGemm ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;

    return res ? 0 : 1;
}
