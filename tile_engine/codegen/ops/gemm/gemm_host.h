#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>


template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3840", "m dimension")
        .insert("n", "4096", "n dimension")
        .insert("k", "2048", "k dimension")
        .insert("stride_a", "0", "Tensor A stride")
        .insert("stride_b", "0", "Tensor B stride")
        .insert("stride_c", "0", "Tensor C stride")
        .insert("split_k", "1", "splitK value")
        .insert("v", "2", "0. No validation, 1. Validation on CPU, 2. Validation on GPU")
        .insert("warmup", "50", "number of iterations before benchmark the kernel")
        .insert("repeat", "100", "number of iterations to benchmark the kernel")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer")
        .insert("init", "0", "0:random, 1:linear, 2:constant(1)");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType, typename ALayout, typename BLayout, typename CLayout>
int init_host_tensor(                           const ck_tile::ArgParser& arg_parser
                                                ck_tile::HostTensor<ADataType>& a_m_k,
                                                ck_tile::HostTensor<BDataType>& b_k_n,
                                                ck_tile::HostTensor<CDataType>& c_m_n_dev_result,
                                                ck_tile::DeviceMem& a_m_k_dev_buf,
                                                ck_tile::DeviceMem& b_k_n_dev_buf,
                                                ck_tile::DeviceMem& c_m_n_dev_buf,
                                                const ALayout a_layout = ALayout{},
                                                const BLayout b_layout = BLayout{},
                                                [[maybe_unused]] const CLayout c_layout = CLayout{})
{
    ck_tile::index_t M = arg_parser.get_int("m");
    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t K = arg_parser.get_int("k");

    ck_tile::index_t stride_A = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_B = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_C = arg_parser.get_int("stride_c");

    ck_tile::index_t kbatch      = arg_parser.get_int("split_k");
    int n_warmup                 = arg_parser.get_int("warmup");
    int n_repeat                 = arg_parser.get_int("repeat");
    ck_tile::index_t init_method = arg_parser.get_int("init");

    stride_A = ck_tile::get_default_stride(M, K, stride_A, is_row_major(a_layout));
    stride_B = ck_tile::get_default_stride(K, N, stride_B, is_row_major(b_layout));
    stride_C = ck_tile::get_default_stride(M, N, stride_C, is_row_major(CLayout{}));


    a_m_k(ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(a_layout)));
    b_k_n(ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(b_layout)));
    c_m_n_dev_result(ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

    if(init_method == 0)
    {
        ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k);
        ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_k_n);
    }
    else if(init_method == 1)
    {
        ck_tile::FillMonotonicSeq<ADataType>{}(a_m_k);
        ck_tile::FillMonotonicSeq<BDataType>{}(b_k_n);
    }
    else if(init_method == 2)
    {
        ck_tile::FillConstant<ADataType>{static_cast<ADataType>(1)}(a_m_k);
        ck_tile::FillConstant<BDataType>{static_cast<BDataType>(1)}(b_k_n);
    }
    else
    {
        a_m_k.SetZero();
        b_k_n.SetZero();
    }

    a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
    b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
    c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

    a_m_k_dev_buf.ToDevice(a_m_k.data());
    b_k_n_dev_buf.ToDevice(b_k_n.data());
    c_m_n_dev_buf.SetZero();
    c_m_n_dev_result.SetZero(); //TODO:: Can we create it later on after kernel call.

    //TODO:: return or pass them as reference
    return 1;
}

//verification code
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType,, typename ALayout, typename BLayout, typename CLayout>
void do_verify(
                ck_tile::HostTensor<ADataType>& a_m_k,
                ck_tile::HostTensor<BDataType>& b_k_n,  
                ck_tile::HostTensor<CDataType>& c_m_n_dev_result,      
                ck_tile::DeviceMem& a_m_k_dev_buf,
                ck_tile::DeviceMem& b_k_n_dev_buf,
                ck_tile::index_t M,
                ck_tile::index_t N,
                ck_tile::index_t K,
                ck_tile::index_t stride_A,
                ck_tile::index_t stride_B,
                ck_tile::index_t stride_C,
                ck_tile::index_t kbatch)
    /*a_host_tensor, b_host_tensor, c_host_tensor(copied from device tensor for validation on cpu), m, n, k, k+batch, stride_C, */) {

    if(arg_parser.get_int("v") == 1)
    {
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        c_m_n_host_ref.SetZero();

        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_m_k, b_k_n, c_m_n_host_ref);
        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, kbatch, max_accumulated_value);
        pass = ck_tile::check_err(c_m_n_dev_result,
                                c_m_n_host_ref,
                                "Error: Incorrect results!",
                                rtol_atol.at(ck_tile::number<0>{}),
                                rtol_atol.at(ck_tile::number<1>{}));

        std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                << std::endl;
        std::cout << "The CPU verification result is:" << (pass ? "correct" : "fail") << std::endl;
    }
    else if(arg_parser.get_int("v") == 2)
    {
        ck_tile::HostTensor<CDataType> c_m_n_gpu_ref(
            ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));
        ck_tile::DeviceMem c_m_n_gpu_buf_ref(c_m_n_gpu_ref.get_element_space_size_in_bytes());
        c_m_n_gpu_ref.SetZero();
        c_m_n_gpu_buf_ref.SetZero();

        ADataType* d_A;
        BDataType* d_B;
        CDataType* d_C;

        ck_tile::hip_check_error(hipMalloc(&d_A, M * K * sizeof(ADataType)));
        ck_tile::hip_check_error(hipMalloc(&d_B, N * K * sizeof(BDataType)));
        ck_tile::hip_check_error(hipMalloc(&d_C, M * N * sizeof(CDataType)));

        ck_tile::hip_check_error(hipMemcpy(d_A,
                                        a_m_k_dev_buf.GetDeviceBuffer(),
                                        M * K * sizeof(ADataType),
                                        hipMemcpyHostToDevice));
        ck_tile::hip_check_error(hipMemcpy(d_B,
                                        b_k_n_dev_buf.GetDeviceBuffer(),
                                        N * K * sizeof(BDataType),
                                        hipMemcpyHostToDevice));

        ck_tile::reference_gemm_gpu<ADataType,
                                    BDataType,
                                    AccDataType,
                                    CDataType,
                                    ALayout,
                                    BLayout,
                                    CLayout>(d_A, d_B, d_C, M, N, K, stride_A, stride_B, stride_C);

        ck_tile::hip_check_error(hipMemcpy(c_m_n_gpu_buf_ref.GetDeviceBuffer(),
                                        d_C,
                                        M * N * sizeof(CDataType),
                                        hipMemcpyDeviceToHost));

        ck_tile::hip_check_error(hipFree(d_A));
        ck_tile::hip_check_error(hipFree(d_B));
        ck_tile::hip_check_error(hipFree(d_C));

        c_m_n_gpu_buf_ref.FromDevice(c_m_n_gpu_ref.data());
        const float max_accumulated_value =
            *std::max_element(c_m_n_gpu_ref.mData.begin(), c_m_n_gpu_ref.mData.end());
        const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
            K, kbatch, max_accumulated_value);
        pass = ck_tile::check_err(c_m_n_dev_result,
                                c_m_n_gpu_ref,
                                "Error: Incorrect results!",
                                rtol_atol.at(ck_tile::number<0>{}),
                                rtol_atol.at(ck_tile::number<1>{}));

        std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                << " Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                << std::endl;
        std::cout << "The GPU verification result is: " << (pass ? "correct" : "fail") << std::endl;
    }

}

