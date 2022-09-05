// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <getopt.h>

#include "ck/ck.hpp"
// #include "ck/tensor_operation/gpu/device/device_sparse_embedding3_forward_layernorm.hpp"
#include "ck/tensor_operation/gpu/device/device_sparse_embedding3_forward_layernorm.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_common_util.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_sparse_embedding3_forward_layernorm.hpp"

using EmbType = float;
using IndexType = int64_t;
using GammaDataType = float;
using BetaDataType = float;
using AccDataType = float;
using OutType = float;

using DeviceInstance_fp32_e1024 = ck::tensor_operation::device::DeviceSparseEmbedding3ForwardLayernorm<
                                                                        EmbType,
                                                                        IndexType,
                                                                        GammaDataType,
                                                                        BetaDataType,
                                                                        AccDataType,
                                                                        OutType,
                                                                        256,        // BlockSize
                                                                        256,        // DimClusterSize
                                                                        1,          // RowClusterSize
                                                                        1,          // DimPerBlock
                                                                        1024,       // RowPerBlock
                                                                        1,          // DimThreadSize
                                                                        4>          // RowVectorSize

using DeviceInstance_fp32_e768 = ck::tensor_operation::device::DeviceSparseEmbedding3ForwardLayernorm<
                                                                        EmbType,
                                                                        IndexType,
                                                                        GammaDataType,
                                                                        BetaDataType,
                                                                        AccDataType,
                                                                        OutType,
                                                                        256,        // BlockSize
                                                                        256,        // DimClusterSize
                                                                        1,          // RowClusterSize
                                                                        1,          // DimPerBlock
                                                                        768,        // RowPerBlock
                                                                        1,          // DimThreadSize
                                                                        1>          // RowVectorSize

template<typename emb_type, ck::index_t dim>
struct emb_kernel{
};

template<>
struct emb_kernel<float, 768>{
    using kernel_type = DeviceInstance_fp32_e768;
};

template<>
struct emb_kernel<float, 1024>{
    using kernel_type = DeviceInstance_fp32_e1024;
};



int main()
{
    bool time_kernel = false;

    constexpr auto num_rows = 65536;
    constexpr auto dims = ck::Sequence<768, 1024>{};
    constexpr auto index_length = 32;
    constexpr AccDataType epsilon = 1e-4;

    auto f_host_tensor_desc_1d = [](std::size_t len_){
        return HostTensorDescriptor(std::vector<std::size_t>({len_}));
    };

    auto f_host_tensor_desc_2d = [](std::size_t rows_, std::size_t cols_){
        return HostTensorDescriptor(std::vector<std::size_t>({rows_, cols_}));
    };

    using ReferenceInstance = ck::tensor_operation::host::ReferenceSparseEmbedding3ForwardLayernorm<EmbType,
                                                                                                    IndexType,
                                                                                                    GammaDataType,
                                                                                                    BetaDataType,
                                                                                                    AccDataType,
                                                                                                    OutType>;

    static_for<0, dims.Size(), 1>{}([&](auto I){
        constexpr auto current_dim = dims.At(Number<I>{});
        Tensor<EmbType> emb_a(f_host_tensor_desc_2d(num_rows, current_dim));
        Tensor<EmbType> emb_b(f_host_tensor_desc_2d(num_rows, current_dim));
        Tensor<EmbType> emb_c(f_host_tensor_desc_2d(num_rows, current_dim));

        Tensor<IndexType> index_a(f_host_tensor_desc_1d(index_length));
        Tensor<IndexType> index_b(f_host_tensor_desc_1d(index_length));
        Tensor<IndexType> index_c(f_host_tensor_desc_1d(index_length));

        Tensor<GammaDataType> gamma(f_host_tensor_desc_1d(current_dim));
        Tensor<BetaDataType> beta(f_host_tensor_desc_1d(current_dim));

        Tensor<OutType> out(f_host_tensor_desc_2d(index_length, current_dim));

        emb_a.GenerateTensorValue(GeneratorTensor_3<EmbType>{0.0, 1.0});
        emb_b.GenerateTensorValue(GeneratorTensor_3<EmbType>{0.0, 1.0});
        emb_c.GenerateTensorValue(GeneratorTensor_3<EmbType>{0.0, 1.0});

        index_a.GenerateTensorValue(GeneratorTensor_2<IndexType>{0, num_rows});
        index_b.GenerateTensorValue(GeneratorTensor_2<IndexType>{0, num_rows});
        index_c.GenerateTensorValue(GeneratorTensor_2<IndexType>{0, num_rows});

        DeviceMem emb_a_dev(sizeof(EmbType) * emb_a.mDesc.GetElementSpaceSize());
        DeviceMem emb_b_dev(sizeof(EmbType) * emb_b.mDesc.GetElementSpaceSize());
        DeviceMem emb_c_dev(sizeof(EmbType) * emb_c.mDesc.GetElementSpaceSize());

        DeviceMem index_a_dev(sizeof(IndexType) * index_a.mDesc.GetElementSpaceSize());
        DeviceMem index_b_dev(sizeof(IndexType) * index_b.mDesc.GetElementSpaceSize());
        DeviceMem index_c_dev(sizeof(IndexType) * index_c.mDesc.GetElementSpaceSize());

        DeviceMem gamma_dev(sizeof(GammaDataType) * gamma.mDesc.GetElementSpaceSize());
        DeviceMem beta_dev(sizeof(BetaDataType) * beta.mDesc.GetElementSpaceSize());

        DeviceMem out_dev(sizeof(OutType) * out.mDesc.GetElementSpaceSize());

        emb_a_dev.ToDevice(emb_a.mData.data());
        emb_b_dev.ToDevice(emb_b.mData.data());
        emb_c_dev.ToDevice(emb_c.mData.data());

        index_a_dev.ToDevice(index_a.mData.data());
        index_b_dev.ToDevice(index_b.mData.data());
        index_c_dev.ToDevice(index_c.mData.data());

        gamma_dev.ToDevice(gamma.mData.data());
        beta_dev.ToDevice(beta.mData.data());

        auto device_instance = emb_kernel<EmbType, current_dim>{};
        auto argument_ptr    = device_instance.MakeArgumentPointer(
                                    out_dev.GetDeviceBuffer(),
                                    emb_a_dev.GetDeviceBuffer(),
                                    emb_b_dev.GetDeviceBuffer(),
                                    emb_c_dev.GetDeviceBuffer(),
                                    index_a_dev.GetDeviceBuffer(),
                                    index_b_dev.GetDeviceBuffer(),
                                    index_c_dev.GetDeviceBuffer(),
                                    gamma_dev.GetDeviceBuffer(),
                                    beta_dev.GetDeviceBuffer(),
                                    num_rows,
                                    current_dim,
                                    index_length,
                                    epsilon);

        if(!device_instance.IsSupportedArgument(argument_ptr.get()))
        {
            std::cout << "The runtime parameters are not supported" << std::endl;
            return 1;
        };

        auto invoker_ptr = device_instance.MakeInvokerPointer();
        invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

        // reference
        bool pass = true;
        {
            Tensor<OutType> out_from_dev(f_host_tensor_desc_2d(index_length, current_dim));
            ReferenceInstance ref;
            auto ref_argument =
                ref.MakeArgument(out, emb_a, emb_b, emb_c, index_a, index_b, index_c, gamma, beta, num_rows,
                                    current_dim,
                                    index_length,
                                    epsilon);
            auto ref_invoker = ref.MakeInvoker();
            ref_invoker.Run(ref_argument);

            out_dev.FromDevice(out_from_dev.mData.data());
            pass &=
                ck::utils::check_err(out_from_dev.mData, out.mData, "Error: Incorrect results d1", 1e-3, 1e-3);
        }
    });

    return 0;
}
