// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <numeric>
#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <vector>
#include <gtest/gtest.h>

#include "ck/library/utility/host_tensor.hpp"

#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

#include "ck/host_utility/kernel_launch.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/wrapper/layout.hpp"
#include "ck/wrapper/tensor.hpp"
#include "ck/wrapper/operations/copy.hpp"
#include "ck/wrapper/operations/gemm.hpp"

template <typename DataType>
void CheckResult(const std::vector<DataType>& a_data,
                 const std::vector<DataType>& b_data,
                 std::vector<DataType>& c_m_n_device_result,
                 const ck::index_t M,
                 const ck::index_t N,
                 const ck::index_t K)
{
    using PassThrough           = ck::tensor_operation::element_wise::PassThrough;
    using ReferenceGemmInstance = ck::tensor_operation::host::
        ReferenceGemm<DataType, DataType, DataType, float, PassThrough, PassThrough, PassThrough>;

    Tensor<DataType> a_m_k(HostTensorDescriptor({M, K}));
    Tensor<DataType> b_k_n(HostTensorDescriptor({K, N}, {1, K}));
    Tensor<DataType> c_m_n_host_result(HostTensorDescriptor({M, N}));

    a_m_k.mData = a_data;
    b_k_n.mData = b_data;

    auto ref_op       = ReferenceGemmInstance{};
    auto ref_invoker  = ref_op.MakeInvoker();
    auto ref_argument = ref_op.MakeArgument(
        a_m_k, b_k_n, c_m_n_host_result, PassThrough{}, PassThrough{}, PassThrough{});

    ref_invoker.Run(ref_argument);
    EXPECT_TRUE(ck::utils::check_err(c_m_n_device_result, c_m_n_host_result.mData));
}

template <typename DataType,
          typename GemmTraits,
          ck::index_t scalar_per_vector,
          typename BlockShape,
          typename ThreadLayoutShape>
__global__ void DeviceGemm(const void* p_a,
                           const void* p_b,
                           void* p_c,
                           const ck::index_t M,
                           const ck::index_t N,
                           const ck::index_t K,
                           const BlockShape tile_shape,
                           const ThreadLayoutShape thread_layout)
{
    constexpr auto MPerBlock = ck::wrapper::size<0>(tile_shape);
    constexpr auto NPerBlock = ck::wrapper::size<1>(tile_shape);
    constexpr auto KPerBlock = ck::wrapper::size<2>(tile_shape);

    const auto a_global_layout =
        ck::wrapper::make_layout(ck::make_tuple(M, K), ck::make_tuple(K, 1));
    const auto b_global_layout =
        ck::wrapper::make_layout(ck::make_tuple(N, K), ck::make_tuple(K, 1));
    const auto c_global_layout =
        ck::wrapper::make_layout(ck::make_tuple(M, N), ck::make_tuple(N, 1));

    constexpr auto a_tile_layout = ck::wrapper::make_layout(
        ck::make_tuple(MPerBlock, KPerBlock), ck::make_tuple(KPerBlock, ck::Number<1>{}));
    constexpr auto b_tile_layout = ck::wrapper::make_layout(
        ck::make_tuple(NPerBlock, KPerBlock), ck::make_tuple(KPerBlock, ck::Number<1>{}));
    constexpr auto c_tile_layout = ck::wrapper::make_layout(
        ck::make_tuple(MPerBlock, NPerBlock), ck::make_tuple(NPerBlock, ck::Number<1>{}));

    auto a_global_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Global>(
        static_cast<const DataType*>(p_a), a_global_layout);
    auto b_global_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Global>(
        static_cast<const DataType*>(p_b), b_global_layout);
    auto c_global_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Global>(
        static_cast<DataType*>(p_c), c_global_layout);

    auto a_padded_global_tensor = ck::wrapper::pad(a_global_tensor, shape(a_tile_layout));
    auto b_padded_global_tensor = ck::wrapper::pad(b_global_tensor, shape(b_tile_layout));
    auto c_padded_global_tensor = ck::wrapper::pad(c_global_tensor, shape(c_tile_layout));

    __shared__ DataType lds_a[ck::wrapper::size(a_tile_layout)];
    __shared__ DataType lds_b[ck::wrapper::size(b_tile_layout)];

    auto a_lds_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Lds>(
        static_cast<DataType*>(lds_a), a_tile_layout);
    auto b_lds_tensor = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Lds>(
        static_cast<DataType*>(lds_b), b_tile_layout);

    const ck::index_t block_idx      = static_cast<ck::index_t>(blockIdx.x);
    using DimAccessOrder             = ck::Tuple<ck::Number<0>, ck::Number<1>>;
    constexpr ck::index_t vector_dim = 1;

    auto c_global_local_tile = ck::wrapper::make_local_tile(
        c_padded_global_tensor,
        tile_shape,
        block_idx,
        make_tuple(ck::Number<1>{}, ck::Number<1>{}, ck::wrapper::slice(KPerBlock)));
    auto c_global_local_partition =
        ck::wrapper::make_blockwise_gemm_xdl_c_local_partition<DataType,
                                                               decltype(a_tile_layout),
                                                               decltype(b_tile_layout),
                                                               ck::wrapper::size(thread_layout),
                                                               GemmTraits>(c_global_local_tile);
    auto c_vgpr_reg = ck::wrapper::make_blockwise_gemm_xdl_c_vgpr<DataType,
                                                                  decltype(a_tile_layout),
                                                                  decltype(b_tile_layout),
                                                                  ck::wrapper::size(thread_layout),
                                                                  GemmTraits>();
    ck::wrapper::clear(c_vgpr_reg);

    const ck::index_t num_loop = ck::math::integer_divide_ceil(K, KPerBlock);
    ck::index_t i              = 0;
    do
    {
        const auto k_slice = ck::wrapper::slice(i * KPerBlock, (i + 1) * KPerBlock);
        auto a_padded_global_tensor_k_slice = a_padded_global_tensor(ck::wrapper::slice(), k_slice);
        auto b_padded_global_tensor_k_slice = b_padded_global_tensor(ck::wrapper::slice(), k_slice);
        auto a_global_local_tile            = ck::wrapper::make_local_tile(
            a_padded_global_tensor_k_slice,
            tile_shape,
            block_idx,
            make_tuple(ck::Number<1>{}, ck::wrapper::slice(N), ck::Number<1>{}));
        auto b_global_local_tile = ck::wrapper::make_local_tile(
            b_padded_global_tensor_k_slice,
            tile_shape,
            block_idx,
            make_tuple(ck::wrapper::slice(M), ck::Number<1>{}, ck::Number<1>{}));

        ck::wrapper::blockwise_copy<DimAccessOrder, vector_dim, scalar_per_vector>(
            a_global_local_tile, a_lds_tensor, thread_layout);
        ck::wrapper::blockwise_copy<DimAccessOrder, vector_dim, scalar_per_vector>(
            b_global_local_tile, b_lds_tensor, thread_layout);
        ck::block_sync_lds();
        ck::wrapper::blockwise_gemm_xdl<DataType, ck::wrapper::size(thread_layout), GemmTraits>(
            a_lds_tensor, b_lds_tensor, c_vgpr_reg);

        ++i;
    } while(i < num_loop);

    ck::wrapper::copy(c_vgpr_reg, c_global_local_partition);
}

template <typename DataType,
          typename GemmTraits,
          ck::index_t scalar_per_vector,
          typename BlockShape,
          typename ThreadLayoutShape>
void PerformGemm(const ck::index_t M,
                 const ck::index_t N,
                 const ck::index_t K,
                 const BlockShape& tile_shape,
                 const ThreadLayoutShape& thread_layout)
{
    // Global memory buffers
    DeviceMem a_mem(M * K * sizeof(DataType));
    DeviceMem b_mem(K * N * sizeof(DataType));
    DeviceMem c_mem(M * N * sizeof(DataType));

    std::vector<DataType> a_data(M * K);
    std::vector<DataType> b_data(K * N);
    ck::utils::FillUniformDistributionIntegerValue<DataType>{-5.f, 5.f}(a_data);
    ck::utils::FillUniformDistributionIntegerValue<DataType>{-5.f, 5.f}(b_data);

    a_mem.ToDevice(a_data.data());
    b_mem.ToDevice(b_data.data());
    c_mem.SetZero();

    const ck::index_t grid_size =
        ck::math::integer_divide_ceil(M, ck::wrapper::size<0>(tile_shape)) *
        ck::math::integer_divide_ceil(N, ck::wrapper::size<1>(tile_shape));

    const auto kernel =
        DeviceGemm<DataType, GemmTraits, scalar_per_vector, BlockShape, ThreadLayoutShape>;
    launch_and_time_kernel(StreamConfig{nullptr},
                           kernel,
                           dim3(grid_size),
                           dim3(ck::wrapper::size(thread_layout)),
                           0,
                           a_mem.GetDeviceBuffer(),
                           b_mem.GetDeviceBuffer(),
                           c_mem.GetDeviceBuffer(),
                           M,
                           N,
                           K,
                           tile_shape,
                           thread_layout);

    std::vector<DataType> c_data(M * N);
    c_mem.FromDevice(c_data.data());

    CheckResult<DataType>(a_data, b_data, c_data, M, N, K);
}

TEST(TestGemm, Float)
{
    using DataType           = float;
    const auto thread_layout = ck::make_tuple(ck::Number<16>{}, ck::Number<16>{});
    const auto tile_shape = ck::make_tuple(ck::Number<128>{}, ck::Number<128>{}, ck::Number<64>{});
    PerformGemm<DataType, ck::wrapper::BlockwisGemmXdlTraits_32x32Xdl_2x2XdlPerWave_4K1, 4>(
        512, 512, 128, tile_shape, thread_layout);
    // Irregular case
    PerformGemm<DataType, ck::wrapper::BlockwisGemmXdlTraits_32x32Xdl_2x2XdlPerWave_4K1, 1>(
        129, 129, 67, tile_shape, thread_layout);
}

TEST(TestGemm, Int8)
{
    using DataType           = int8_t;
    const auto thread_layout = ck::make_tuple(ck::Number<64>{}, ck::Number<4>{});
    const auto tile_shape = ck::make_tuple(ck::Number<128>{}, ck::Number<128>{}, ck::Number<64>{});
    PerformGemm<DataType, ck::wrapper::BlockwisGemmXdlTraits_32x32Xdl_2x2XdlPerWave_16K1, 16>(
        512, 512, 128, tile_shape, thread_layout);
    // Irregular case
    PerformGemm<DataType, ck::wrapper::BlockwisGemmXdlTraits_32x32Xdl_2x2XdlPerWave_16K1, 1>(
        129, 129, 67, tile_shape, thread_layout);
}

TEST(TestGemm, Half)
{
    using DataType           = ck::half_t;
    const auto thread_layout = ck::make_tuple(ck::Number<32>{}, ck::Number<8>{});
    const auto tile_shape = ck::make_tuple(ck::Number<128>{}, ck::Number<128>{}, ck::Number<64>{});
    PerformGemm<DataType, ck::wrapper::BlockwisGemmXdlTraits_32x32Xdl_2x2XdlPerWave_8K1, 8>(
        512, 512, 128, tile_shape, thread_layout);
    // Irregular case
    PerformGemm<DataType, ck::wrapper::BlockwisGemmXdlTraits_32x32Xdl_2x2XdlPerWave_8K1, 1>(
        129, 129, 67, tile_shape, thread_layout);
}

TEST(TestGemm, Float_2x4_4x2_XdlPerWave)
{
    using DataType                            = float;
    const auto thread_layout_4x2_xdl_per_wave = ck::make_tuple(ck::Number<16>{}, ck::Number<8>{});
    const auto thread_layout_2x4_xdl_per_wave = ck::make_tuple(ck::Number<8>{}, ck::Number<16>{});
    const auto tile_shape = ck::make_tuple(ck::Number<128>{}, ck::Number<128>{}, ck::Number<64>{});
    PerformGemm<DataType, ck::wrapper::BlockwisGemmXdlTraits_32x32Xdl_4x2XdlPerWave_4K1, 4>(
        512, 512, 128, tile_shape, thread_layout_4x2_xdl_per_wave);
    PerformGemm<DataType, ck::wrapper::BlockwisGemmXdlTraits_32x32Xdl_2x4XdlPerWave_4K1, 4>(
        512, 512, 128, tile_shape, thread_layout_2x4_xdl_per_wave);
}
