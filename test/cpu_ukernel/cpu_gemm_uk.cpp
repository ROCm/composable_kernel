#include <iostream>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <tuple>
#include <memory>
#include <chrono>
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

// #define ITERATE_THREAD_GEMM_AVX2_MXN_6X16_INSTANCE(FA, FB, FC, TA, TB, NT)  \
//     ck::cpu::ThreadwiseGemmAvx2_MxN_6x16<FA, FB, FC,  6, 16,  TA,  TB,  NT>

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using thread_gemm_avx2_mxn_6x16_instances = std::tuple<
    // clang-format off
    //                                        FloatA FloatB FloatC ALayout  BLayout NTStore
    ITERATE_THREAD_GEMM_AVX2_MXN_6X16_INSTANCE(float, float, float,    Row,    Row, false),
    ITERATE_THREAD_GEMM_AVX2_MXN_6X16_INSTANCE(float, float, float,    Row,    Col, false),
    ITERATE_THREAD_GEMM_AVX2_MXN_6X16_INSTANCE(float, float, float,    Col,    Row, false),
    ITERATE_THREAD_GEMM_AVX2_MXN_6X16_INSTANCE(float, float, float,    Col,    Col, false)

    // ITERATE_THREAD_GEMM_AVX2_MXN_6X16_INSTANCE(float, float, float,    Row,    Col, false)
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

void* __aligned_malloc(size_t required_bytes, size_t alignment)
{
    if(alignment == 0 || (alignment & (alignment - 1))) // check pow of 2
        return nullptr;
    void* p1;  // original block
    void** p2; // aligned block
    int offset = alignment - 1 + sizeof(void*);
    if((p1 = malloc(required_bytes + offset)) == nullptr)
    {
        return nullptr;
    }
    p2     = reinterpret_cast<void**>((reinterpret_cast<size_t>(p1) + offset) & ~(alignment - 1));
    p2[-1] = p1;
    return p2;
}

void __aligned_free(void* p) { free((reinterpret_cast<void**>(p))[-1]); }

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

template <typename data_type, typename ALayout, typename BLayout>
void ref_cpu_gemm_uk(const data_type* a,
                     const data_type* b,
                     float* c,
                     float alpha,
                     uint32_t m,
                     uint32_t n,
                     uint32_t k)
{
    auto a_offset = [&](uint32_t im, uint32_t ik) {
        if constexpr(std::is_same<Row, ALayout>::value)
        {
            return im * k + ik;
        }
        else
        {
            return ik * m + im;
        }
    };

    auto b_offset = [&](uint32_t ik, uint32_t in) {
        if constexpr(std::is_same<Row, BLayout>::value)
        {
            return ik * n + in;
        }
        else
        {
            // n*k*n8
            return (in / 8) * k * 8 + ik * 8 + in % 8;
        }
    };

    auto c_offset = [&](uint32_t im, uint32_t in) { return im * n + in; };

    for(uint32_t im = 0; im < m; im++)
    {
        for(uint32_t in = 0; in < n; in++)
        {
            float acc = .0f;
            for(uint32_t ik = 0; ik < k; ik++)
            {
                acc += a[a_offset(im, ik)] * b[b_offset(ik, in)];
            }
            acc *= alpha;
            c[c_offset(im, in)] = acc;
        }
    }
}

template <typename data_type, typename ALayout, typename BLayout, typename ukenrel_t>
void test_ukernel(ukenrel_t uk,
                  data_type* mat_a,
                  data_type* mat_b,
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
    param.lda   = (std::is_same<Row, ALayout>::value ? k : m) * sizeof(data_type);
    param.ldb   = (std::is_same<Row, BLayout>::value ? n : k * 8) * sizeof(data_type);
    param.ldc   = n * sizeof(float);
    param.alpha = alpha;

    printf("gemm_uk_%dx%d_%c%c: ", uk.Mr_, uk.Nr_, ALayout::name[0], BLayout::name[0]);
    fflush(stdout);
    // printf("%s: ", typeid(uk).name());fflush(stdout);
    memset(mat_c, 0, m * n * sizeof(float));

    int repeat = 7e10 / (2 * m * n * k);

    for(int i = 0; i < (repeat / 5); i++)
    {
        uk.Run(&param);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < repeat; i++)
    {
        uk.Run(&param);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double us = static_cast<double>(
                    std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) /
                repeat;
    double gflops = static_cast<double>(2 * m * n * k) * 1e-3 / us;

    memset(mat_c, 0, m * n * sizeof(float));
    uk.Run(&param);

    printf("m:%u, n:%u, k:%u, alpha:%f, cost:%lfus, GFLOPS:%lf, ", m, n, k, alpha, us, gflops);
    fflush(stdout);
}

// implement small ukernel on L1
template <typename data_type, typename ALayout, typename BLayout>
void test_cpu_ukernel(float alpha, uint32_t m, uint32_t n, uint32_t k)
{
    data_type* mat_a =
        reinterpret_cast<data_type*>(__aligned_malloc(m * k * sizeof(data_type), 32));
    data_type* mat_b =
        reinterpret_cast<data_type*>(__aligned_malloc(k * n * sizeof(data_type), 32));
    float* mat_c = reinterpret_cast<float*>(__aligned_malloc(m * n * sizeof(float), 32));

    float* mat_c_ref = reinterpret_cast<float*>(__aligned_malloc(m * n * sizeof(float), 32));
    memset(mat_c_ref, 0, m * n * sizeof(float));

    rand_vector(mat_a, m * k);
    rand_vector(mat_b, k * n);

    ref_cpu_gemm_uk<data_type, ALayout, BLayout>(mat_a, mat_b, mat_c_ref, alpha, m, n, k);

    ck::static_for<0, std::tuple_size_v<thread_gemm_avx2_mxn_6x16_instances>, 1>{}([&](auto i) {
        using uk_type = std::tuple_element_t<i, thread_gemm_avx2_mxn_6x16_instances>;
        if constexpr(!std::is_same<typename uk_type::ALayout_, ALayout>::value ||
                     !std::is_same<typename uk_type::BLayout_, BLayout>::value)
        {
            return;
        }
        if(uk_type::Mr_ != m || uk_type::Nr_ != n)
            return;

        test_ukernel<data_type, ALayout, BLayout>(uk_type{}, mat_a, mat_b, mat_c, alpha, m, n, k);

        bool is_valid = valid_vector(mat_c_ref, mat_c, m * n);
        printf("vald:%s\n", is_valid ? "y" : "n");

        // return ;
    });

    __aligned_free(mat_a);
    __aligned_free(mat_b);
    __aligned_free(mat_c);
    __aligned_free(mat_c_ref);
}

int main(int argc, char** argv)
{
    int m       = 6;
    int n       = 16;
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
    test_cpu_ukernel<float, Row, Row>(alpha, m, n, k);
    test_cpu_ukernel<float, Row, Col>(alpha, m, n, k);
    test_cpu_ukernel<float, Col, Row>(alpha, m, n, k);
    test_cpu_ukernel<float, Col, Col>(alpha, m, n, k);
}
