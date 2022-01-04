#include <iostream>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include "config.hpp"
#include "print.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "device_tensor.hpp"
#include "device_gemm_instance.hpp"
#include "host_gemm.hpp"
#include "tensor_layout.hpp"
#include "device_gemm_splitk_xdl_instance.hpp"
#include "device_gemm_splitk_xdl.hpp"

enum GemmMatrixLayout
{
    MK_KN_MN, // 0
    MK_NK_MN, // 1
    KM_KN_MN, // 2
    KM_NK_MN, // 3
};
using DeviceGemmNoOpPtr =
    ck::tensor_operation::device::DeviceGemmPtr<ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough>;
using GEMM_PTR = std::vector<DeviceGemmNoOpPtr>;

static std::vector<std::vector<bool>>& GetLayoutType()
{
    static std::vector<std::vector<bool>> LayOut = {{0, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, 1, 0}};
    return LayOut;
}
static void add_device_gemm_instance_mk_kn_mn(GEMM_PTR& gemm_ptrs)
{
    ck::tensor_operation::device::device_gemm_instance::add_device_splitk_gemm_instance<
        float,
        float,
        float,
        ck::tensor_layout::gemm::RowMajor,
        ck::tensor_layout::gemm::RowMajor,
        ck::tensor_layout::gemm::RowMajor>(gemm_ptrs);
}
static void add_device_gemm_instance_mk_nk_mn(GEMM_PTR& gemm_ptrs)
{
    ck::tensor_operation::device::device_gemm_instance::add_device_splitk_gemm_instance<
        float,
        float,
        float,
        ck::tensor_layout::gemm::RowMajor,
        ck::tensor_layout::gemm::ColumnMajor,
        ck::tensor_layout::gemm::RowMajor>(gemm_ptrs);
}
static void add_device_gemm_instance_km_kn_mn(GEMM_PTR& gemm_ptrs)
{
    ck::tensor_operation::device::device_gemm_instance::add_device_splitk_gemm_instance<
        float,
        float,
        float,
        ck::tensor_layout::gemm::ColumnMajor,
        ck::tensor_layout::gemm::RowMajor,
        ck::tensor_layout::gemm::RowMajor>(gemm_ptrs);
}
static void add_device_gemm_instance_km_nk_mn(GEMM_PTR& gemm_ptrs)
{
    ck::tensor_operation::device::device_gemm_instance::add_device_splitk_gemm_instance<
        float,
        float,
        float,
        ck::tensor_layout::gemm::ColumnMajor,
        ck::tensor_layout::gemm::ColumnMajor,
        ck::tensor_layout::gemm::RowMajor>(gemm_ptrs);
}

static auto& GetAddDeviceGemmInstance()
{
    static std::vector<void (*)(GEMM_PTR&)> AddDeviceGemmInstance = {
        add_device_gemm_instance_mk_kn_mn,
        add_device_gemm_instance_mk_nk_mn,
        add_device_gemm_instance_km_kn_mn,
        add_device_gemm_instance_km_nk_mn};
    return AddDeviceGemmInstance;
}

static void add_device_gemm_instance(GEMM_PTR& gemm_ptrs, int layout)
{
    GetAddDeviceGemmInstance()[layout](gemm_ptrs);
}

template <typename T>
static bool check_out(const Tensor<T>& ref, const Tensor<T>& result)
{
    float max_diff = 1e-6;

    for(int i = 0; i < ref.mData.size(); ++i)
    {
        float diff = std::abs(double(ref.mData[i]) - double(result.mData[i]));
        if(max_diff < diff)
        {
            return false;
        }
    }

    return true;
}
int main(int argc, char* argv[])
{
    if(argc != 9)
    {
        printf("arg1: matrix layout (0: A[m, k] * B[k, n] = C[m, n];\n");
        printf("                     1: A[m, k] * B[n, k] = C[m, n];\n");
        printf("                     2: A[k, n] * B[k, n] = C[m, n];\n");
        printf("                     3: A[k, n] * B[n, k] = C[m, n])\n");
        printf("arg2 to 7: M, N, K, StrideA, StrideB, StrideC KBatch\n");
        return 1;
    }

    const int layout = static_cast<GemmMatrixLayout>(std::stoi(argv[1]));

    const int M = std::stoi(argv[2]);
    const int N = std::stoi(argv[3]);
    const int K = std::stoi(argv[4]);

    const int StrideA = std::stoi(argv[5]);
    const int StrideB = std::stoi(argv[6]);
    const int StrideC = std::stoi(argv[7]);
    const int KBatch  = std::stoi(argv[8]);

    if(layout > 3 || layout < 0)
    {
        printf("arg1 must be 0 ,1 ,2 or 3 \n");
        return 1;
    }
    auto LayOut = GetLayoutType();

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, bool isRevert) {
            if(isRevert)
            {
                return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                            std::vector<std::size_t>({1, stride}));
            }
            else
            {
                return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                            std::vector<std::size_t>({stride, 1}));
            }
        };
    Tensor<float> a_m_k(f_host_tensor_descriptor(M, K, StrideA, LayOut[layout][0]));
    Tensor<float> b_k_n(f_host_tensor_descriptor(K, N, StrideB, LayOut[layout][1]));
    Tensor<float> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, LayOut[layout][2]));
    Tensor<float> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, LayOut[layout][2]));

    // init data
    std::size_t num_thread = std::thread::hardware_concurrency();
    a_m_k.GenerateTensorValue(GeneratorTensor_2<float>{-5, 5}, num_thread);
    b_k_n.GenerateTensorValue(GeneratorTensor_2<float>{-5, 5}, num_thread);
    // set zero to c_device_buf
    c_m_n_device_result.GenerateTensorValue(GeneratorTensor_0<float>{}, num_thread);

    host_gemm_mk_kn_mn(a_m_k,
                       b_k_n,
                       c_m_n_host_result,
                       ck::tensor_operation::element_wise::PassThrough{},
                       ck::tensor_operation::element_wise::PassThrough{},
                       ck::tensor_operation::element_wise::PassThrough{});

    DeviceMem a_device_buf(sizeof(float) * a_m_k.mDesc.GetElementSpace());
    DeviceMem b_device_buf(sizeof(float) * b_k_n.mDesc.GetElementSpace());
    DeviceMem c_device_buf(sizeof(float) * c_m_n_device_result.mDesc.GetElementSpace());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());
    c_device_buf.ToDevice(c_m_n_device_result.mData.data());

    // add device GEMM instances
    GEMM_PTR gemm_ptrs;
    add_device_gemm_instance(gemm_ptrs, layout);

    bool success = false;
    for(auto& gemm_ptr : gemm_ptrs)
    {
        auto argument_ptr =
            gemm_ptr->MakeArgumentPointer(static_cast<float*>(a_device_buf.GetDeviceBuffer()),
                                          static_cast<float*>(b_device_buf.GetDeviceBuffer()),
                                          static_cast<float*>(c_device_buf.GetDeviceBuffer()),
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideC,
                                          ck::tensor_operation::element_wise::PassThrough{},
                                          ck::tensor_operation::element_wise::PassThrough{},
                                          ck::tensor_operation::element_wise::PassThrough{},
                                          KBatch);

        auto invoker_ptr = gemm_ptr->MakeInvokerPointer();
        if(gemm_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            invoker_ptr->Run(argument_ptr.get(), 0);

            c_device_buf.FromDevice(c_m_n_device_result.mData.data());
            if(!check_out(c_m_n_host_result, c_m_n_device_result))
            {
                success = false;
                break;
            }
            success = true;
        }
    }
    if(success)
    {
        std::cout << "test split k : Pass" << std::endl;
    }
    else
    {
        std::cout << "test split k: Fail " << std::endl;
    }
    return 0;
}
