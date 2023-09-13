import enum
import os.path
import shutil
import functools
import operator
import collections
import subprocess
import re
import gemm_op
from gemm_op import *
import user

def SubstituteTemplate(template, values):
    text = template
    changed = True
    while changed:
        changed = False
        for key, value in values.items():
            regex = "\\$\\{%s\\}" % key
            newtext = re.sub(regex, value, text)
            if newtext != text:
                changed = True
            text = newtext
    return text


class EmitGemmInstance:
    def __init__(self):
        self.gemm_devop_template =     """
#pragma once

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_dl.hpp"

using ADataType   = ck::half_t;
using BDataType   = ck::half_t;
using CDataType   = ck::half_t;
using AccDataType = float;

using ALayout = Col;
using BLayout = Row;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

using DeviceGemmInstance = ck::tensor_operation::device::DeviceGemmDl<
            ${type_a},
            ${type_b},
            ${type_c},
            ${type_acc},
            ${layout_a},
            ${layout_b},
            ${layout_c},
            ${elementwise_op_a},
            ${elementwise_op_b},
            ${elementwise_op_c},
            ${Gemm_spec},
            ${block_size},
            ${mperblock},
            ${nperblock},
            ${k0perblock},
            ${k1},
            ${m1perthread},
            ${n1perthread},
            ${kperthread},
            ${m1n1_thcluster_m1xs},
            ${m1n1_thcluster_n1xs},
            ${ABT_thread_slice_lengths_K0_M0_M1_K1},
            ${ABT_thread_cluster_lengths_K0_M0_M1_K1},
            ${ABT_thread_cluster_arrange_order},
            ${ABT_src_access_order},
            ${ABT_src_vec_tensor_lengths_K0_M0_M1_K1},
            ${ABT_src_vec_tensor_cont_dim_order},
            ${ABT_dst_vec_tensor_lengths_K0_M0_M1_K1},
            ${BBT_thread_slice_lengths_K0_N0_N1_K1},
            ${BBT_thread_cluster_lengths_K0_N0_N1_K1},
            ${BBT_thread_cluster_arrange_order},
            ${BBT_src_access_order},
            ${BBT_src_vec_tensor_lengths_K0_N0_N1_K1},
            ${BBT_src_vec_tensor_cont_dim_order},
            ${BBT_dst_vec_tensor_lengths_K0_N0_N1_K1},
            ${CTT_src_dst_access_order},
            ${CTT_src_dst_vec_dim},
            ${CTT_dst_scalar_per_vector}>;

    using ReferenceGemmInstance = ck::tensor_operation::host::
        ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;


bool run_gemm_${name}(const ProblemSize& problem_size, const ExecutionConfig& config)
{
    using namespace ck::literals;

    auto& [M, N, K, StrideA, StrideB, StrideC] = problem_size;

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    Tensor<${type_a}> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ${layout_a}{}));
    Tensor<${type_b}> b_k_n(f_host_tensor_descriptor(K, N, StrideB, ${layout_b}{}));

    switch(config.init_method)
    {
    case 0: break;
    case 1:
        ck::utils::FillUniformDistributionIntegerValue<${type_a}>{-5.f, 5.f}(a_m_k);
        ck::utils::FillUniformDistributionIntegerValue<${type_b}>{-5.f, 5.f}(b_k_n);
        break;
    default:
        ck::utils::FillUniformDistribution<${type_a}>{-1.f, 1.f}(a_m_k);
        ck::utils::FillUniformDistribution<${type_b}>{-1.f, 1.f}(b_k_n);
    }

    Tensor<${type_c}> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<${type_c}> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;

    DeviceMem a_m_k_device_buf(sizeof(${type_a}) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_k_n_device_buf(sizeof(${type_b}) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(${type_c}) * c_m_n_device_result.mDesc.GetElementSpaceSize());

    a_m_k_device_buf.ToDevice(a_m_k.mData.data());
    b_k_n_device_buf.ToDevice(b_k_n.mData.data());


    auto a_element_op = ${elementwise_op_a}{};
    auto b_element_op = ${elementwise_op_b}{};
    auto c_element_op = ${elementwise_op_c}{};

    // do GEMM
    auto gemm     = DeviceGemmInstance{};
    auto invoker  = gemm.MakeInvoker();
    auto argument = gemm.MakeArgument(

        static_cast<${type_a}*>(a_m_k_device_buf.GetDeviceBuffer()),
        static_cast<${type_b}*>(b_k_n_device_buf.GetDeviceBuffer()),
        static_cast<${type_c}*>(c_m_n_device_buf.GetDeviceBuffer()),
        M,
        N,
        K,
        StrideA,
        StrideB,
        StrideC,
        a_element_op,
        b_element_op,
        c_element_op);

    if(!gemm.IsSupportedArgument(argument))
    {
        std::cerr << gemm.GetTypeString() << " does not support this problem" << std::endl;

        return true;
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});

    std::size_t flop = 2_uz * M * N * K;
    std::size_t num_btype =
        sizeof(${type_a}) * M * K + sizeof(${type_b}) * K * N + sizeof(${type_c}) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << gemm.GetTypeString() << std::endl;

    if(config.do_verification)
    {
        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n, c_m_n_host_result, a_element_op, b_element_op, c_element_op);

        ref_invoker.Run(ref_argument);

        c_m_n_device_buf.FromDevice(c_m_n_device_result.mData.data());

        return ck::utils::check_err(c_m_n_device_result, c_m_n_host_result);
    }

    return true;
}

bool run_gemm_${name}(int argc, char* argv[])
{
    ProblemSize problem_size;
    ExecutionConfig config;

    return !parse_cmd_args(argc, argv, problem_size, config) || run_gemm_${name}(problem_size, config);
}

"""
    def emit(self,operation):
        name = (str(operation.tile_desc.block_size) + "_" + str(operation.tile_desc.m_per_block) + "_" + str(operation.tile_desc.n_per_block)+ "_" + str(operation.tile_desc.k_per_block) + "_" + str(operation.tile_desc.k1))
        values = {
            'name' : name,
            'type_a' : operation.A.element,
            'type_b' : operation.B.element,
            'type_c' : operation.C.element,
            'type_acc' : 'float',
            'layout_a' : operation.A.layout,
            'layout_b' : operation.B.layout,
            'layout_c' : operation.C.layout,
            'elementwise_op_a' : operation.a_elem_op,
            'elementwise_op_b' : operation.b_elem_op,
            'elementwise_op_c' : operation.epilogue_functor,
            'Gemm_spec' : operation.gemm_specialization,
            'block_size' : str(operation.tile_desc.block_size),
            'mperblock' : str(operation.tile_desc.m_per_block),
            'nperblock' : str(operation.tile_desc.n_per_block),
            'k0perblock' : str(operation.tile_desc.k_per_block),
            'k1' : str(operation.tile_desc.k1),
            'm1perthread' : str(operation.tile_desc.m_per_thread),
            'n1perthread' : str(operation.tile_desc.n_per_thread),
            'kperthread' : str(operation.tile_desc.k_per_thread),
            'm1n1_thcluster_m1xs' : operation.tile_desc.m1n1_thcluster_m1xs,
            'm1n1_thcluster_n1xs' : operation.tile_desc.m1n1_thcluster_n1xs,
            'ABT_thread_slice_lengths_K0_M0_M1_K1' : operation.a_block_transfer.thread_slice_length,
            'ABT_thread_cluster_lengths_K0_M0_M1_K1' : operation.a_block_transfer.thread_cluster_length,
            'ABT_thread_cluster_arrange_order' : operation.a_block_transfer.thread_cluster_arrange_order,
            'ABT_src_access_order' : operation.a_block_transfer.src_access_order,
            'ABT_src_vec_tensor_lengths_K0_M0_M1_K1' : operation.a_block_transfer.src_vec_tensor_lengths,
            'ABT_src_vec_tensor_cont_dim_order' : operation.a_block_transfer.src_vec_tensor_cont_dim_order,
            'ABT_dst_vec_tensor_lengths_K0_M0_M1_K1' : operation.a_block_transfer.dst_vec_tensor_lengths,
            'BBT_thread_slice_lengths_K0_N0_N1_K1' : operation.b_block_transfer.thread_slice_length,
            'BBT_thread_cluster_lengths_K0_N0_N1_K1' : operation.b_block_transfer.thread_cluster_length,
            'BBT_thread_cluster_arrange_order' :  operation.b_block_transfer.thread_cluster_arrange_order,
            'BBT_src_access_order' : operation.b_block_transfer.src_access_order,
            'BBT_src_vec_tensor_lengths_K0_N0_N1_K1' : operation.b_block_transfer.src_vec_tensor_lengths,
            'BBT_src_vec_tensor_cont_dim_order' : operation.b_block_transfer.src_vec_tensor_cont_dim_order,
            'BBT_dst_vec_tensor_lengths_K0_N0_N1_K1': operation.b_block_transfer.dst_vec_tensor_lengths,
            'CTT_src_dst_access_order' : operation.c_block_transfer.src_dst_access_order,
            'CTT_src_dst_vec_dim' : str(operation.c_block_transfer.src_dst_vec_dim),
            'CTT_dst_scalar_per_vector' : str(operation.c_block_transfer.dst_scalar_per_vector),
        }
        template = self.gemm_devop_template
        # name = (str(operation.tile_desc.block_size) + "_" + str(operation.tile_desc.m_per_block) + "_" + str(operation.tile_desc.n_per_block)
        # + "_" + str(operation.tile_desc.k_per_block) + "_" + str(operation.tile_desc.k1))
        
        cf = open("%s.cpp" % name,'w')
        cf.write(SubstituteTemplate(template, values))
        cf.close()
        