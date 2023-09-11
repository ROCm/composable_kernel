// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#define USING_MASK 0

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <fstream>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/batched_mha_bwd_qloop_light_v2.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_permute.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16   = ck::half_t;
using BF16  = ck::bhalf_t;
using F32   = float;
using U16   = unsigned short;
using INT32 = int32_t;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Scale       = ck::tensor_operation::element_wise::Scale;

using QKVElementOp = PassThrough;
using YElementOp   = PassThrough;

using InputDataType   = F16;
using OutputDataType  = F16;
using GemmDataType    = F16;
using AccDataType     = F32;
using ShuffleDataType = F32;
using LSEDataType     = F32;
using DDataType       = F32;
using ZDataType       = U16; // INT32

static constexpr ck::index_t NumDimG = 2;
static constexpr ck::index_t NumDimM = 1;
static constexpr ck::index_t NumDimN = 1;
static constexpr ck::index_t NumDimK = 1;
static constexpr ck::index_t NumDimO = 1;

static constexpr ck::index_t CShuffleBlockTransferScalarPerVector_NPerBlock = 8;

#if USING_MASK
static constexpr auto MaskingSpec =
    ck::tensor_operation::device::MaskingSpecialization::MaskUpperTriangleFromTopLeft;
#else
static constexpr auto MaskingSpec =
    ck::tensor_operation::device::MaskingSpecialization::MaskDisabled;
#endif

struct SimpleDeviceMem
{
    SimpleDeviceMem() = delete;

    SimpleDeviceMem(std::size_t mem_size) : p_mem_{}
    {
        (void)hipMalloc(static_cast<void**>(&p_mem_), mem_size);
    }

    void* GetDeviceBuffer() { return p_mem_; }

    ~SimpleDeviceMem() { (void)hipFree(p_mem_); }

    void* p_mem_;
};

int main(int argc, char* argv[])
{
    ck::index_t M  = 512;
    ck::index_t N  = 512;
    ck::index_t K  = 128; // K & O should 64 < K & O <=128
    ck::index_t O  = 128; // K & O should 64 < K & O <=128
    ck::index_t G0 = 4;
    ck::index_t G1 = 6;

    bool input_permute  = false;
    bool output_permute = false;

    float p_drop                    = 0.0;
    const unsigned long long seed   = 1;
    const unsigned long long offset = 0;

    float p_dropout               = 1 - p_drop;
    ZDataType p_dropout_in_16bits = ZDataType(std::floor(p_dropout * 65535.0));
    float rp_dropout              = 1.0 / p_dropout;
    float alpha                   = 1.f / std::sqrt(K);

    std::cout << "M: " << M << std::endl;
    std::cout << "N: " << N << std::endl;
    std::cout << "K: " << K << std::endl;
    std::cout << "O: " << O << std::endl;
    std::cout << "G0: " << G0 << std::endl;
    std::cout << "G1: " << G1 << std::endl;
    std::cout << "alpha: " << alpha << std::endl;
    std::cout << "input_permute: " << input_permute << std::endl;
    std::cout << "output_permute: " << output_permute << std::endl;
    std::cout << "p_drop: " << p_drop << std::endl;
    std::cout << "seed: " << seed << std::endl;
    std::cout << "offset: " << offset << std::endl;

    const ck::index_t BatchCount = G0 * G1;

    std::vector<ck::index_t> q_gs_ms_ks_lengths{G0, G1, M, K};
    std::vector<ck::index_t> q_gs_ms_ks_strides =
        input_permute
            ? std::vector<ck::index_t>{M * G1 * K, K, G1 * K, 1} // Q layout [G0, M, G1, K]
            : std::vector<ck::index_t>{G1 * M * K, M * K, K, 1}; // Q layout [G0, G1, M, K]

    std::vector<ck::index_t> k_gs_ns_ks_lengths{G0, G1, N, K};
    std::vector<ck::index_t> k_gs_ns_ks_strides =
        input_permute
            ? std::vector<ck::index_t>{N * G1 * K, K, G1 * K, 1} // K layout [G0, N, G1, K]
            : std::vector<ck::index_t>{G1 * N * K, N * K, K, 1}; // K layout [G0, G1, N, K]

    std::vector<ck::index_t> v_gs_os_ns_lengths{G0, G1, O, N};
    std::vector<ck::index_t> v_gs_os_ns_strides =
        input_permute
            ? std::vector<ck::index_t>{N * G1 * O, O, 1, G1 * O} // V layout [G0, N, G1, O]
            : std::vector<ck::index_t>{G1 * N * O, N * O, 1, O}; // V layout [G0, G1, N, O]

    std::vector<ck::index_t> y_gs_ms_os_lengths{G0, G1, M, O};
    std::vector<ck::index_t> y_gs_ms_os_strides =
        output_permute
            ? std::vector<ck::index_t>{M * G1 * O, O, G1 * O, 1} // Y layout [G0, M, G1, O]
            : std::vector<ck::index_t>{G1 * M * O, M * O, O, 1}; // Y layout [G0, G1, M, O]

    std::vector<ck::index_t> z_gs_ms_ns_lengths{G0, G1, M, N};
    std::vector<ck::index_t> z_gs_ms_ns_strides =
        input_permute
            ? std::vector<ck::index_t>{M * G1 * N, N, G1 * N, 1} // Z layout [G0, M, G1, N]
            : std::vector<ck::index_t>{G1 * M * N, M * N, N, 1}; // Z layout [G0, G1, M, N]
    // The softmax stat log-sum-exp (LSE) is used to speed up softmax calculation in backward pass
    // Pi = exp(Si) / sum(exp(S0) + exp(S1) + ...)
    //    = exp(Si) / exp(log(sum(exp() + ...)))
    //    = exp(Si - log(sum(exp() + ...)))
    //               ^^^^^^^^^^^^^^^^^^^^^
    //                       LSE
    std::vector<ck::index_t> lse_gs_ms_lengths{G0, G1, M};
    std::vector<ck::index_t> lse_gs_ms_strides{G1 * M, M, 1}; // LSE layout [G0, G1, M]

    SimpleDeviceMem q_device_buf(sizeof(InputDataType) * G0 * G1 * M * K);
    SimpleDeviceMem k_device_buf(sizeof(InputDataType) * G0 * G1 * N * K);
    SimpleDeviceMem z_device_buf(sizeof(ZDataType) * G0 * G1 * M * N);
    SimpleDeviceMem v_device_buf(sizeof(InputDataType) * G0 * G1 * O * N);
    SimpleDeviceMem y_device_buf(sizeof(InputDataType) * G0 * G1 * M * O);
    SimpleDeviceMem lse_device_buf(sizeof(LSEDataType) * G0 * G1 * M);
    SimpleDeviceMem qgrad_device_buf(sizeof(OutputDataType) * G0 * G1 * M * K);
    SimpleDeviceMem kgrad_device_buf(sizeof(OutputDataType) * G0 * G1 * N * K);
    SimpleDeviceMem vgrad_device_buf(sizeof(OutputDataType) * G0 * G1 * O * N);
    SimpleDeviceMem ygrad_device_buf(sizeof(InputDataType) * G0 * G1 * M * O);
    SimpleDeviceMem d_device_buf(sizeof(DDataType) * G0 * G1 * M);

    using DeviceOp =
        ck::tensor_operation::device::DeviceBatchedMultiheadAttentionBackwardQloopLightV2<
            2,
            1,
            1,
            1,
            1,
            InputDataType,
            OutputDataType,
            ZDataType,
            LSEDataType,
            DDataType,
            void,
            void,
            QKVElementOp,
            QKVElementOp,
            Scale,
            QKVElementOp,
            YElementOp,
            MaskingSpec>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    std::string best_op_name;
    int best_op_id        = -1;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    // profile device op instances
    std::cout << "Run all instances and do timing" << std::endl;

    for(int i = 0; i < op_ptrs.size(); ++i)
    {
        auto& op_ptr      = op_ptrs[i];
        auto argument_ptr = op_ptr->MakeArgumentPointer(
            q_device_buf.GetDeviceBuffer(),
            k_device_buf.GetDeviceBuffer(),
            nullptr, // set to nullptr
            v_device_buf.GetDeviceBuffer(),
            y_device_buf.GetDeviceBuffer(),
            lse_device_buf.GetDeviceBuffer(),
            d_device_buf.GetDeviceBuffer(),
            ygrad_device_buf.GetDeviceBuffer(),
            qgrad_device_buf.GetDeviceBuffer(),
            kgrad_device_buf.GetDeviceBuffer(),
            vgrad_device_buf.GetDeviceBuffer(),
            {}, // std::array<void*, 1> p_acc0_biases;
            {}, // std::array<void*, 1> p_acc1_biases;
            q_gs_ms_ks_lengths,
            q_gs_ms_ks_strides,
            k_gs_ns_ks_lengths,
            k_gs_ns_ks_strides,
            z_gs_ms_ns_lengths,
            z_gs_ms_ns_strides,
            v_gs_os_ns_lengths,
            v_gs_os_ns_strides,
            y_gs_ms_os_lengths,
            y_gs_ms_os_strides,
            lse_gs_ms_lengths,
            {}, // std::array<std::vector<ck::index_t>, 1>{acc0_biases_gs_ms_ns_lengths},
            {}, // std::array<std::vector<ck::index_t>, 1>{acc0_biases_gs_ms_ns_strides},
            {}, // std::array<std::vector<ck::index_t>, 1>{acc1_biases_gs_ms_os_lengths},
            {}, // std::array<std::vector<ck::index_t>, 1>{acc1_biases_gs_ms_os_strides},
            QKVElementOp{},
            QKVElementOp{},
            Scale{alpha},
            QKVElementOp{},
            YElementOp{},
            p_drop,
            std::tuple<unsigned long long, unsigned long long>(seed, offset));

        auto invoker_ptr    = op_ptr->MakeInvokerPointer();
        std::string op_name = op_ptr->GetTypeString();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            float ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true});

            std::size_t flop = (size_t(3) * M * N * K + size_t(2) * M * N * O) * 2 * BatchCount;

            std::size_t num_btype =
                (sizeof(InputDataType) * M * K + sizeof(InputDataType) * K * N +
                 sizeof(InputDataType) * N * O + sizeof(InputDataType) * M * O * size_t(2) +
                 sizeof(OutputDataType) * M * K + sizeof(OutputDataType) * K * N +
                 sizeof(OutputDataType) * N * O) *
                    BatchCount +
                sizeof(LSEDataType) * M * BatchCount;

            float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

            float gb_per_sec = num_btype / 1.E6 / ave_time;

            std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                      << " GB/s, " << op_name << std::endl;

            if(tflops > best_tflops)
            {
                best_op_id      = i;
                best_op_name    = op_name;
                best_tflops     = tflops;
                best_ave_time   = ave_time;
                best_gb_per_sec = gb_per_sec;
            }
        }
        else
        {
            std::cout << op_name << " does not support this problem" << std::endl;
        }
    }

    // run the best instance
    {
        auto& op_ptr = op_ptrs[best_op_id];
        std::cout << "Run the best instance without timing: " << op_ptr->GetTypeString()
                  << std::endl;
        auto argument_ptr = op_ptr->MakeArgumentPointer(
            q_device_buf.GetDeviceBuffer(),
            k_device_buf.GetDeviceBuffer(),
            nullptr, // set to nullptr
            v_device_buf.GetDeviceBuffer(),
            y_device_buf.GetDeviceBuffer(),
            lse_device_buf.GetDeviceBuffer(),
            d_device_buf.GetDeviceBuffer(),
            ygrad_device_buf.GetDeviceBuffer(),
            qgrad_device_buf.GetDeviceBuffer(),
            kgrad_device_buf.GetDeviceBuffer(),
            vgrad_device_buf.GetDeviceBuffer(),
            {}, // std::array<void*, 1> p_acc0_biases;
            {}, // std::array<void*, 1> p_acc1_biases;
            q_gs_ms_ks_lengths,
            q_gs_ms_ks_strides,
            k_gs_ns_ks_lengths,
            k_gs_ns_ks_strides,
            z_gs_ms_ns_lengths,
            z_gs_ms_ns_strides,
            v_gs_os_ns_lengths,
            v_gs_os_ns_strides,
            y_gs_ms_os_lengths,
            y_gs_ms_os_strides,
            lse_gs_ms_lengths,
            {}, // std::array<std::vector<ck::index_t>, 1>{acc0_biases_gs_ms_ns_lengths},
            {}, // std::array<std::vector<ck::index_t>, 1>{acc0_biases_gs_ms_ns_strides},
            {}, // std::array<std::vector<ck::index_t>, 1>{acc1_biases_gs_ms_os_lengths},
            {}, // std::array<std::vector<ck::index_t>, 1>{acc1_biases_gs_ms_os_strides},
            QKVElementOp{},
            QKVElementOp{},
            Scale{alpha},
            QKVElementOp{},
            YElementOp{},
            p_drop,
            std::tuple<unsigned long long, unsigned long long>(seed, offset));

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        }

        std::cout << "Done" << std::endl;
    }

    return 0;
}
