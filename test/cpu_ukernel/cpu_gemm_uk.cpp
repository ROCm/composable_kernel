#include <iostream>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <tuple>
#include <memory>
#include <half.hpp>
#include "host_tensor.hpp"
#include "device.hpp"
#include "config.hpp"
#include "print.hpp"
#include "cpuid.hpp"
#include "threadwise_gemm_avx2.hpp"

#define ITERATE_THREAD_GEMM_AVX2_MXN_6X16_INSTANCE(FA, FB, FC, TA, TB, NT)   \
    ck::cpu::ThreadwiseGemmAvx2_MxN_6x16<FA, FB, FC, 6, 16, TA, TB, NT>,     \
        ck::cpu::ThreadwiseGemmAvx2_MxN_6x16<FA, FB, FC, 5, 16, TA, TB, NT>, \
        ck::cpu::ThreadwiseGemmAvx2_MxN_6x16<FA, FB, FC, 4, 16, TA, TB, NT>, \
        ck::cpu::ThreadwiseGemmAvx2_MxN_6x16<FA, FB, FC, 3, 16, TA, TB, NT>, \
        ck::cpu::ThreadwiseGemmAvx2_MxN_6x16<FA, FB, FC, 2, 16, TA, TB, NT>, \
        ck::cpu::ThreadwiseGemmAvx2_MxN_6x16<FA, FB, FC, 1, 16, TA, TB, NT>, \
        ck::cpu::ThreadwiseGemmAvx2_MxN_6x16<FA, FB, FC, 6, 8, TA, TB, NT>,  \
        ck::cpu::ThreadwiseGemmAvx2_MxN_6x16<FA, FB, FC, 5, 8, TA, TB, NT>,  \
        ck::cpu::ThreadwiseGemmAvx2_MxN_6x16<FA, FB, FC, 4, 8, TA, TB, NT>,  \
        ck::cpu::ThreadwiseGemmAvx2_MxN_6x16<FA, FB, FC, 3, 8, TA, TB, NT>,  \
        ck::cpu::ThreadwiseGemmAvx2_MxN_6x16<FA, FB, FC, 2, 8, TA, TB, NT>,  \
        ck::cpu::ThreadwiseGemmAvx2_MxN_6x16<FA, FB, FC, 1, 8, TA, TB, NT>

//#define ITERATE_THREAD_GEMM_AVX2_MXN_6X16_INSTANCE(FA, FB, FC, TA, TB, NT)  \
//     ck::cpu::ThreadwiseGemmAvx2_MxN_6x16<FA, FB, FC,  6, 16,  TA,  TB,  NT>

#define ITERATE_THREAD_GEMM_AVX2_MXN_4X24_INSTANCE(FA, FB, FC, TA, TB, NT)   \
    ck::cpu::ThreadwiseGemmAvx2_MxN_4x24<FA, FB, FC, 4, 24, TA, TB, NT>,     \
        ck::cpu::ThreadwiseGemmAvx2_MxN_4x24<FA, FB, FC, 3, 24, TA, TB, NT>, \
        ck::cpu::ThreadwiseGemmAvx2_MxN_4x24<FA, FB, FC, 2, 24, TA, TB, NT>, \
        ck::cpu::ThreadwiseGemmAvx2_MxN_4x24<FA, FB, FC, 1, 24, TA, TB, NT>, \
        ck::cpu::ThreadwiseGemmAvx2_MxN_4x24<FA, FB, FC, 4, 16, TA, TB, NT>, \
        ck::cpu::ThreadwiseGemmAvx2_MxN_4x24<FA, FB, FC, 3, 16, TA, TB, NT>, \
        ck::cpu::ThreadwiseGemmAvx2_MxN_4x24<FA, FB, FC, 2, 16, TA, TB, NT>, \
        ck::cpu::ThreadwiseGemmAvx2_MxN_4x24<FA, FB, FC, 1, 16, TA, TB, NT>, \
        ck::cpu::ThreadwiseGemmAvx2_MxN_4x24<FA, FB, FC, 4, 8, TA, TB, NT>,  \
        ck::cpu::ThreadwiseGemmAvx2_MxN_4x24<FA, FB, FC, 3, 8, TA, TB, NT>,  \
        ck::cpu::ThreadwiseGemmAvx2_MxN_4x24<FA, FB, FC, 2, 8, TA, TB, NT>,  \
        ck::cpu::ThreadwiseGemmAvx2_MxN_4x24<FA, FB, FC, 1, 8, TA, TB, NT>

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

// using AType = half_float::half;
// using BType = half_float::half;
using AType = float;
using BType = float;
using CType = float;

template <typename ALayout, typename BLayout>
using thread_gemm_avx2_mxn_6x16_instances = std::tuple<
    // clang-format off
    //                                        FloatA FloatB FloatC  ALayout  BLayout NTStore
    ITERATE_THREAD_GEMM_AVX2_MXN_6X16_INSTANCE( AType, BType, CType, ALayout, BLayout, false),
    ITERATE_THREAD_GEMM_AVX2_MXN_6X16_INSTANCE( AType, BType, CType, ALayout, BLayout, false),
    ITERATE_THREAD_GEMM_AVX2_MXN_6X16_INSTANCE( AType, BType, CType, ALayout, BLayout, false),
    ITERATE_THREAD_GEMM_AVX2_MXN_6X16_INSTANCE( AType, BType, CType, ALayout, BLayout, false)

    // ITERATE_THREAD_GEMM_AVX2_MXN_6X16_INSTANCE(AType, BType, CType,    ALayout,    BLayout, false)
    // clang-format on
    >;

template <typename ALayout, typename BLayout>
using thread_gemm_avx2_mxn_4x24_instances = std::tuple<
    // clang-format off
    //                                        FloatA FloatB FloatC  ALayout  BLayout NTStore
    ITERATE_THREAD_GEMM_AVX2_MXN_4X24_INSTANCE( AType, BType, CType, ALayout, BLayout, false),
    ITERATE_THREAD_GEMM_AVX2_MXN_4X24_INSTANCE( AType, BType, CType, ALayout, BLayout, false),
    ITERATE_THREAD_GEMM_AVX2_MXN_4X24_INSTANCE( AType, BType, CType, ALayout, BLayout, false),
    ITERATE_THREAD_GEMM_AVX2_MXN_4X24_INSTANCE( AType, BType, CType, ALayout, BLayout, false)
    // clang-format on
    >;

void dump_cache_hierarchy()
{
    auto dump_cache_type = [&](const ck::cpu::cpuid_cache_type& type) {
        if(type == ck::cpu::cpuid_cache_type_dcache)
            printf("data cache");
        else if(type == ck::cpu::cpuid_cache_type_icache)
            printf("inst cache");
        else if(type == ck::cpu::cpuid_cache_type_unified)
            printf("unif cache");
    };
    auto dump_cache_detail = [&](const ck::cpu::cpuid_cache_detail& detail) {
        dump_cache_type(static_cast<const ck::cpu::cpuid_cache_type>(detail.type));
        printf(" size:%u, cache_line:%u, associativity:%u, sets:%u, partitions:%u, shared by "
               "procs:%u(%u)\n",
               detail.size,
               detail.cache_line_size,
               detail.associativity,
               detail.sets,
               detail.partitions,
               detail.shared_by_procs,
               detail.cores_per_socket);
    };

    ck::cpu::cpuid_cache_hierarchy cache = ck::cpu::cpuid_query_cache();
    if(cache.l1d.size != 0)
    {
        printf("l1 ");
        dump_cache_detail(cache.l1d);
    }
    if(cache.l1i.size != 0)
    {
        printf("l1 ");
        dump_cache_detail(cache.l1i);
    }
    if(cache.l2.size != 0)
    {
        printf("l2 ");
        dump_cache_detail(cache.l2);
    }
    if(cache.l3.size != 0)
    {
        printf("l3 ");
        dump_cache_detail(cache.l3);
    }
    if(cache.l4.size != 0)
    {
        printf("l4 ");
        dump_cache_detail(cache.l4);
    }
}

template <typename T>
void rand_vector(T* v, int elem)
{
    int i;

    static int flag = 0;
    if(!flag)
    {
        srand(time(nullptr));
        flag = 1;
    }

    for(i = 0; i < elem; i++)
    {
        v[i] = (static_cast<T>(rand() % 100)) / 100.0f;
    }
}

bool valid_vector(const float* ref, const float* rhs, uint32_t elem)
{
    float rtol   = 1e-5;
    float atol   = 1e-8;
    uint32_t err = 0;
    for(uint32_t i = 0; i < elem; i++)
    {
        float diff = std::abs(ref[i] - rhs[i]);
        if(diff > atol + rtol * std::abs(ref[i]))
        {
            printf("diff at %u, ref:%f, rhs:%f\n", i, ref[i], rhs[i]);
            err++;
        }
    }

    return err == 0;
}

template <typename FloatA, typename FloatB, typename ALayout, typename BLayout>
void ref_cpu_gemm_uk(
    const FloatA* a, const FloatB* b, float* c, float alpha, uint32_t m, uint32_t n, uint32_t k)
{
    auto f_host_2d_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if(std::is_same<decltype(layout), Row>::value)
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

    auto f_host_vectored_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t vec, std::size_t stride) {
            // only valid in row major. stride is for each row, contains vector size
            return HostTensorDescriptor(std::vector<std::size_t>({row, col, vec}),
                                        std::vector<std::size_t>({stride, vec, 1}));
        };

    std::size_t lda = std::is_same<Row, ALayout>::value ? k : m;     // in unit of element
    std::size_t ldb = std::is_same<Row, BLayout>::value ? n : k * 8; // in unit of element
    std::size_t ldc = n;
    HostTensorDescriptor a_m_k = f_host_2d_tensor_descriptor(m, n, lda, ALayout{});
    HostTensorDescriptor b_k_n = std::is_same<Row, BLayout>::value
                                     ? f_host_2d_tensor_descriptor(k, n, ldb, BLayout{})
                                     : f_host_vectored_tensor_descriptor(n / 8, k, 8, ldb);
    HostTensorDescriptor c_m_n = f_host_2d_tensor_descriptor(m, n, ldc, Row{});

    for(uint32_t im = 0; im < m; im++)
    {
        for(uint32_t in = 0; in < n; in++)
        {
            float acc = .0f;
            for(uint32_t ik = 0; ik < k; ik++)
            {
                acc += static_cast<float>(a[a_m_k.GetOffsetFromMultiIndex(im, ik)]) *
                       (std::is_same<Row, BLayout>::value
                            ? static_cast<float>(b[b_k_n.GetOffsetFromMultiIndex(ik, in)])
                            : static_cast<float>(
                                  b[b_k_n.GetOffsetFromMultiIndex(in / 8, ik, in % 8)]));
            }
            acc *= alpha;
            c[c_m_n.GetOffsetFromMultiIndex(im, in)] = acc;
        }
    }
}

template <typename FloatA, typename FloatB, typename ALayout, typename BLayout, typename ukenrel_t>
void test_ukernel(ukenrel_t uk,
                  FloatA* mat_a,
                  FloatB* mat_b,
                  float* mat_c,
                  float alpha,
                  uint32_t m,
                  uint32_t n,
                  uint32_t k)
{
    ck::cpu::ThreadwiseGemmParam param;
    param.p_a   = mat_a;
    param.p_b   = mat_b;
    param.p_c   = mat_c;
    param.Kr    = k;
    param.lda   = (std::is_same<Row, ALayout>::value ? k : m) * sizeof(FloatA);
    param.ldb   = (std::is_same<Row, BLayout>::value ? n : k * 8) * sizeof(FloatB);
    param.ldc   = n * sizeof(float);
    param.alpha = alpha;

    auto invoke_uk = [&]() {
        if constexpr(std::is_same<Row, ALayout>::value && std::is_same<Row, BLayout>::value)
        {
            assert(m % uk.Mr_ == 0 && n == uk.Nr_);
            FloatA* p_a = mat_a;
            float* p_c  = mat_c;
            param.p_a   = p_a;
            param.p_c   = p_c;
            for(uint32_t i_m = 0; i_m < m; i_m += uk.Mr_)
            {
                uk.Run(&param);
                p_a += uk.Mr_ * k;
                p_c += uk.Mr_ * n;
                param.p_a = p_a;
                param.p_c = p_c;
            }
        }
        else if constexpr(std::is_same<Row, ALayout>::value && std::is_same<Col, BLayout>::value)
        {
            assert(m % uk.Mr_ == 0 && n % uk.Nr_ == 0);
            FloatA* p_a = mat_a;
            float* p_c  = mat_c;
            param.p_a   = p_a;
            param.p_b   = mat_b;
            param.p_c   = p_c;
            for(uint32_t i_m = 0; i_m < m; i_m += uk.Mr_)
            {
                float* p_c_n  = p_c;
                FloatB* p_b_n = mat_b;
                for(uint32_t i_n = 0; i_n < n; i_n += uk.Nr_)
                {
                    uk.Run(&param);
                    p_b_n += uk.Nr_ * k; // Nr_/8*k*8
                    p_c_n += uk.Nr_;
                    param.p_b = p_b_n;
                    param.p_c = p_c_n;
                }
                p_a += uk.Mr_ * k;
                p_c += uk.Mr_ * n;
                param.p_a = p_a;
                param.p_b = mat_b;
                param.p_c = p_c;
            }
        }
        else if constexpr(std::is_same<Col, ALayout>::value && std::is_same<Row, BLayout>::value)
        {
            assert(m == uk.Mr_ && n == uk.Nr_);
            uk.Run(&param);
        }
        else
        {
            assert(m % uk.Mr_ == 0 && n % uk.Nr_ == 0);
            FloatB* p_b = mat_b;
            float* p_c  = mat_c;
            param.p_b   = p_b;
            param.p_c   = p_c;
            for(uint32_t i_n = 0; i_n < n; i_n += uk.Nr_)
            {
                uk.Run(&param);
                p_b += uk.Nr_ * k; // Nr_/8*k*8
                p_c += uk.Nr_;
                param.p_b = p_b;
                param.p_c = p_c;
            }
        }
    };

    printf("gemm_uk_%dx%d_%c%c: ", uk.Mr_, uk.Nr_, ALayout::name[0], BLayout::name[0]);
    fflush(stdout);
    // printf("%s: ", typeid(uk).name());fflush(stdout);
    memset(mat_c, 0, m * n * sizeof(float));

    int repeat = 7e10 / (2 * m * n * k);

    for(int i = 0; i < (repeat / 5); i++)
    {
        invoke_uk();
    }

    WallTimer timer;

    timer.Start();
    for(int i = 0; i < repeat; i++)
    {
        invoke_uk();
    }
    timer.End();

    float us     = timer.GetElapsedTime() * 1e3 / repeat;
    float gflops = static_cast<float>(2 * m * n * k) * 1e-3 / us;

    memset(mat_c, 0, m * n * sizeof(float));
    invoke_uk();

    printf("m:%u, n:%u, k:%u, alpha:%f, cost:%lfus, GFLOPS:%lf, ", m, n, k, alpha, us, gflops);
    fflush(stdout);
}

// implement small ukernel on L1
template <typename FloatA, typename FloatB, typename ALayout, typename BLayout>
void test_cpu_ukernel(float alpha, uint32_t m, uint32_t n, uint32_t k)
{

    DeviceAlignedMemCPU a_mem(m * k * sizeof(FloatA), 32);
    DeviceAlignedMemCPU b_mem(k * n * sizeof(FloatB), 32);
    DeviceAlignedMemCPU c_mem(m * n * sizeof(float), 32);
    DeviceAlignedMemCPU c_mem_ref(m * n * sizeof(float), 32);

    c_mem_ref.SetZero();
    rand_vector(reinterpret_cast<FloatA*>(a_mem.mpDeviceBuf), m * k);
    rand_vector(reinterpret_cast<FloatB*>(b_mem.mpDeviceBuf), k * n);

    ref_cpu_gemm_uk<FloatA, FloatB, ALayout, BLayout>(
        reinterpret_cast<FloatA*>(a_mem.mpDeviceBuf),
        reinterpret_cast<FloatB*>(b_mem.mpDeviceBuf),
        reinterpret_cast<float*>(c_mem_ref.mpDeviceBuf),
        alpha,
        m,
        n,
        k);

    // using thread_gemm_instance = thread_gemm_avx2_mxn_6x16_instances<ALayout, BLayout>;
    using thread_gemm_instance = thread_gemm_avx2_mxn_4x24_instances<ALayout, BLayout>;
    bool found                 = false;

    ck::static_for<0, std::tuple_size_v<thread_gemm_instance>, 1>{}([&](auto i) {
        using uk_type = std::tuple_element_t<i, thread_gemm_instance>;
        if(m % uk_type::Mr_ != 0 || n % uk_type::Nr_ != 0)
            return;
        if((m != uk_type::Mr_ && std::is_same<typename uk_type::ALayout_, Col>::value) ||
           (n != uk_type::Nr_ && std::is_same<typename uk_type::BLayout_, Row>::value))
            // only k is the fast changing dim of A/B can we do muldiplt m, n
            return;

        if(found)
            return;

        test_ukernel<FloatA, FloatB, ALayout, BLayout>(uk_type{},
                                                       reinterpret_cast<FloatA*>(a_mem.mpDeviceBuf),
                                                       reinterpret_cast<FloatB*>(b_mem.mpDeviceBuf),
                                                       reinterpret_cast<float*>(c_mem.mpDeviceBuf),
                                                       alpha,
                                                       m,
                                                       n,
                                                       k);

        bool is_valid = valid_vector(reinterpret_cast<float*>(c_mem_ref.mpDeviceBuf),
                                     reinterpret_cast<float*>(c_mem.mpDeviceBuf),
                                     m * n);
        printf("vald:%s\n", is_valid ? "y" : "n");
        found = true;
    });
}

int main(int argc, char** argv)
{
    int m       = 4;
    int n       = 24;
    int k       = 64;
    float alpha = 1.0f;
    if(argc > 3)
    {
        m = std::atoi(argv[1]);
        n = std::atoi(argv[2]);
        k = std::atoi(argv[3]);
    }
    if(argc > 4)
    {
        alpha = std::atof(argv[4]);
    }
    dump_cache_hierarchy();
    test_cpu_ukernel<AType, BType, Row, Row>(alpha, m, n, k);
    test_cpu_ukernel<AType, BType, Row, Col>(alpha, m, n, k);
    test_cpu_ukernel<AType, BType, Col, Row>(alpha, m, n, k);
    test_cpu_ukernel<AType, BType, Col, Col>(alpha, m, n, k);
}
