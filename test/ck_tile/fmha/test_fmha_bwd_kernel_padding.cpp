// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <vector>
#include "ck_tile/host.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "example/ck_tile/01_fmha/fmha_bwd.hpp"
#include "example/ck_tile/01_fmha/fmha_bwd_runner.hpp" // for get_elimit
#include "gtest/gtest.h"

namespace {

using bf16 = ck_tile::bf16_t;
using ck_tile::DeviceMem;

const ck_tile::stream_config kStreamConfig{
    nullptr, // stream_id_
    false,   // time_kernel_
    1,       // log_level_
    0,       // cold_niters_
    1,       // nrepeat_
    true,    // is_gpu_timer_
    false,   // flush_cache_
    1,       // rotating_count_
};

template <typename T>
std::vector<T> MakeVectorFromFunction(size_t count, std::function<float(size_t)> fn)
{
    std::vector<T> data(count);
    for(size_t i = 0; i < count; ++i)
    {
        data[i] = static_cast<T>(fn(i));
    }
    return data;
}

template <typename T>
std::vector<float> ToFloatVector(const std::vector<T>& src)
{
    std::vector<float> dst(src.size());
    for(size_t i = 0; i < src.size(); ++i)
    {
        dst[i] = ck_tile::type_convert<float>(src[i]);
    }
    return dst;
}

template <typename T>
std::vector<T> CopyDeviceToHost(const ck_tile::DeviceMem& dev, size_t element_count)
{
    std::vector<T> host(element_count);
    if(element_count > 0)
    {
        dev.FromDevice(host.data());
    }
    return host;
}

float SentinelValue() { return -999.f; }

} // namespace

// Typed tests over {fp32, fp16, bf16}
template <typename DataTypeConfig>
class FmhaBwdKernelPaddingTyped : public ::testing::Test
{
};

using KernelPaddingTypes = ::testing::Types<FmhaBwdFp32, FmhaBwdFp16, FmhaBwdBf16>;
TYPED_TEST_SUITE(FmhaBwdKernelPaddingTyped, KernelPaddingTypes);

TYPED_TEST(FmhaBwdKernelPaddingTyped, OGradDotO_GroupPaddingRespectsLogicalLengths)
{
    constexpr ck_tile::index_t batch      = 2;
    constexpr ck_tile::index_t nhead      = 1;
    constexpr ck_tile::index_t hdim       = 128;
    constexpr ck_tile::index_t phys_rows0 = 8;
    constexpr ck_tile::index_t phys_rows1 = 8;
    constexpr ck_tile::index_t max_phys   = phys_rows0; // both batches equal

    const std::vector<int32_t> seqstart_q_host{0, phys_rows0, phys_rows0 + phys_rows1};
    const std::vector<int32_t> seqlen_q_host{5, 3};

    const ck_tile::index_t total_rows = seqstart_q_host.back();

    // Types per config
    using TypeConfig = FmhaBwdTypeConfig<TypeParam>;
    using OType      = typename TypeConfig::ODataType;
    using DOType     = typename TypeConfig::OGradDataType;
    using DType      = typename TypeConfig::DDataType;   // float under bf16 config
    using AccType    = typename TypeConfig::AccDataType; // float

    // Host tensors laid out as [b, h, s, d] with b=1, h=1 under group mode
    ck_tile::HostTensor<OType> o_host({1, nhead, total_rows, hdim});
    ck_tile::HostTensor<DOType> do_host({1, nhead, total_rows, hdim});
    ck_tile::HostTensor<DType> d_init_host({1, nhead, total_rows});

    // Initialize O/dO with constants using FillConstant (no manual bf16 casts)
    const float o_const  = 0.25f;
    const float do_const = 0.5f;
    ck_tile::FillConstant<OType>{ck_tile::type_convert<OType>(o_const)}(o_host);
    ck_tile::FillConstant<DOType>{ck_tile::type_convert<DOType>(do_const)}(do_host);
    ck_tile::FillConstant<DType>{ck_tile::type_convert<DType>(SentinelValue())}(d_init_host);

    // Prepare expected D via runner-style CPU reference, sentinel elsewhere
    std::vector<float> expected(static_cast<size_t>(total_rows), SentinelValue());
    for(ck_tile::index_t b = 0; b < batch; ++b)
    {
        const ck_tile::index_t start = seqstart_q_host[b];
        const ck_tile::index_t len   = seqlen_q_host[b];
        for(ck_tile::index_t row = 0; row < len; ++row)
        {
            AccType acc = 0;
            for(ck_tile::index_t c = 0; c < hdim; ++c)
            {
                // o_host/do_host are [1, nhead, s, d]
                const auto o_val  = ck_tile::type_convert<AccType>(o_host(0, 0, start + row, c));
                const auto do_val = ck_tile::type_convert<AccType>(do_host(0, 0, start + row, c));
                acc += do_val * o_val;
            }
            expected[start + row] = ck_tile::type_convert<float>(acc);
        }
    }
    std::vector<float> sentinel_ref(static_cast<size_t>(total_rows), SentinelValue());

    // Device buffers
    ck_tile::DeviceMem o_dev(o_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem do_dev(do_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem d_dev(d_init_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem seqstart_dev(seqstart_q_host.size() * sizeof(int32_t));
    ck_tile::DeviceMem seqlen_dev(seqlen_q_host.size() * sizeof(int32_t));

    o_dev.ToDevice(o_host.data());
    do_dev.ToDevice(do_host.data());
    d_dev.ToDevice(d_init_host.data());
    seqstart_dev.ToDevice(seqstart_q_host.data());
    seqlen_dev.ToDevice(seqlen_q_host.data());

    fmha_bwd_args args{};
    args.q_ptr                = nullptr;
    args.k_ptr                = nullptr;
    args.v_ptr                = nullptr;
    args.bias_ptr             = nullptr;
    args.o_ptr                = o_dev.GetDeviceBuffer();
    args.lse_ptr              = nullptr;
    args.do_ptr               = do_dev.GetDeviceBuffer();
    args.d_ptr                = d_dev.GetDeviceBuffer();
    args.rand_val_ptr         = nullptr;
    args.dq_ptr               = nullptr;
    args.dk_ptr               = nullptr;
    args.dv_ptr               = nullptr;
    args.dbias_ptr            = nullptr;
    args.dq_acc_ptr           = nullptr;
    args.seqstart_q_ptr       = seqstart_dev.GetDeviceBuffer();
    args.seqstart_k_ptr       = nullptr;
    args.seqlen_k_ptr         = nullptr;
    args.seqlen_q_ptr         = seqlen_dev.GetDeviceBuffer();
    args.seqlen_q             = 0;
    args.seqlen_k             = 0;
    args.batch                = batch;
    args.max_seqlen_q         = max_phys;
    args.max_seqlen_k         = 0;
    args.hdim_q               = hdim;
    args.hdim_v               = hdim;
    args.nhead_q              = nhead;
    args.nhead_k              = nhead;
    args.scale                = 1.0f;
    args.stride_q             = 0;
    args.stride_k             = 0;
    args.stride_v             = 0;
    args.stride_bias          = 0;
    args.stride_o             = hdim;
    args.stride_randval       = 0;
    args.stride_do            = hdim;
    args.stride_dq_acc        = 0;
    args.stride_dq            = 0;
    args.stride_dk            = 0;
    args.stride_dv            = 0;
    args.stride_dbias         = 0;
    args.nhead_stride_q       = 0;
    args.nhead_stride_k       = 0;
    args.nhead_stride_v       = 0;
    args.nhead_stride_bias    = 0;
    args.nhead_stride_o       = max_phys * hdim;
    args.nhead_stride_randval = 0;
    args.nhead_stride_do      = max_phys * hdim;
    args.nhead_stride_lsed    = max_phys;
    args.nhead_stride_dq_acc  = 0;
    args.nhead_stride_dq      = 0;
    args.nhead_stride_dk      = 0;
    args.nhead_stride_dv      = 0;
    args.nhead_stride_dbias   = 0;
    args.batch_stride_q       = 0;
    args.batch_stride_k       = 0;
    args.batch_stride_v       = 0;
    args.batch_stride_bias    = 0;
    args.batch_stride_o       = 0;
    args.batch_stride_randval = 0;
    args.batch_stride_do      = 0;
    args.batch_stride_lsed    = 0;
    args.batch_stride_dq_acc  = 0;
    args.batch_stride_dq      = 0;
    args.batch_stride_dk      = 0;
    args.batch_stride_dv      = 0;
    args.batch_stride_dbias   = 0;
    args.split_stride_dq_acc  = 0;
    args.window_size_left     = -1;
    args.window_size_right    = 0;
    args.mask_type            = static_cast<ck_tile::index_t>(mask_enum::no_mask);
    args.p_drop               = 0.0f;
    args.p_undrop             = 1.0f;
    args.drop_seed_offset     = std::make_pair(uint64_t{0}, uint64_t{0});

    using DotTileTraits = ck_tile::TileFmhaBwdOGradDotOTraits<true, true, 2>;
    using DotProblem =
        ck_tile::BlockFmhaBwdOGradDotOPipelineProblem<typename TypeConfig::ODataType,
                                                      typename TypeConfig::OGradDataType,
                                                      typename TypeConfig::DDataType,
                                                      64,
                                                      hdim,
                                                      true,
                                                      DotTileTraits>;
    using DotPipeline = ck_tile::BlockFmhaBwdOGradDotO<DotProblem>;
    using DotKernel   = ck_tile::FmhaBwdOGradDotOKernel<DotPipeline>;

    auto [dot_kargs, dot_grids] = fmha_bwd_dot_do_o_create_kargs_and_grids<DotKernel>(args);
    const dim3 dot_blocks       = DotKernel::BlockSize();
    constexpr ck_tile::index_t kDotBlockPerCu = DotKernel::kBlockPerCu;
    auto dot_kernel =
        ck_tile::make_kernel<kDotBlockPerCu>(DotKernel{}, dot_grids, dot_blocks, 0, dot_kargs);
    dot_kernel(kStreamConfig);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    auto d_result_host = CopyDeviceToHost<float>(d_dev, total_rows);

    auto [rtol_doto, atol_doto] = get_elimit<TypeParam>(hdim, hdim);
    for(size_t i = 0; i < d_result_host.size(); ++i)
    {
        SCOPED_TRACE(::testing::Message() << "index=" << i);
        if(std::fabs(expected[i] - sentinel_ref[i]) < 1e-6f)
        {
            EXPECT_FLOAT_EQ(d_result_host[i], sentinel_ref[i]);
        }
        else
        {
            EXPECT_NEAR(d_result_host[i], expected[i], static_cast<float>(atol_doto));
        }
    }
}

TYPED_TEST(FmhaBwdKernelPaddingTyped, OGradDotO_VariedPhysicalAndZeroLogical)
{
    constexpr ck_tile::index_t batch    = 3;
    constexpr ck_tile::index_t nhead    = 1;
    constexpr ck_tile::index_t hdim     = 64;
    constexpr ck_tile::index_t phys_r0  = 5;
    constexpr ck_tile::index_t phys_r1  = 7;
    constexpr ck_tile::index_t phys_r2  = 4;
    constexpr ck_tile::index_t max_phys = phys_r1;

    const std::vector<int32_t> seqstart_q_host{
        0, phys_r0, phys_r0 + phys_r1, phys_r0 + phys_r1 + phys_r2};
    const std::vector<int32_t> seqlen_q_host{3, 0, 4};
    const ck_tile::index_t total_rows = seqstart_q_host.back();

    using TypeConfig = FmhaBwdTypeConfig<TypeParam>;
    using OType      = typename TypeConfig::ODataType;
    using DOType     = typename TypeConfig::OGradDataType;
    using DType      = typename TypeConfig::DDataType;

    ck_tile::HostTensor<OType> o_host({1, nhead, total_rows, hdim});
    ck_tile::HostTensor<DOType> do_host({1, nhead, total_rows, hdim});
    ck_tile::HostTensor<DType> d_init_host({1, nhead, total_rows});
    ck_tile::FillConstant<OType>{ck_tile::type_convert<OType>(1.0f)}(o_host);
    ck_tile::FillConstant<DOType>{ck_tile::type_convert<DOType>(2.0f)}(do_host);
    ck_tile::FillConstant<DType>{ck_tile::type_convert<DType>(SentinelValue())}(d_init_host);

    std::vector<float> expected(static_cast<size_t>(total_rows), SentinelValue());
    const float dot = 2.0f * 1.0f * static_cast<float>(hdim);
    for(ck_tile::index_t b = 0; b < batch; ++b)
    {
        const ck_tile::index_t start = seqstart_q_host[b];
        const ck_tile::index_t len   = seqlen_q_host[b];
        for(ck_tile::index_t row = 0; row < len; ++row)
            expected[start + row] = dot;
    }
    std::vector<float> sentinel_ref(static_cast<size_t>(total_rows), SentinelValue());

    ck_tile::DeviceMem o_dev(o_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem do_dev(do_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem d_dev(d_init_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem seqstart_dev(seqstart_q_host.size() * sizeof(int32_t));
    ck_tile::DeviceMem seqlen_dev(seqlen_q_host.size() * sizeof(int32_t));
    o_dev.ToDevice(o_host.data());
    do_dev.ToDevice(do_host.data());
    d_dev.ToDevice(d_init_host.data());
    seqstart_dev.ToDevice(seqstart_q_host.data());
    seqlen_dev.ToDevice(seqlen_q_host.data());

    fmha_bwd_args args{};
    args.o_ptr             = o_dev.GetDeviceBuffer();
    args.do_ptr            = do_dev.GetDeviceBuffer();
    args.d_ptr             = d_dev.GetDeviceBuffer();
    args.seqstart_q_ptr    = seqstart_dev.GetDeviceBuffer();
    args.seqlen_q_ptr      = seqlen_dev.GetDeviceBuffer();
    args.batch             = batch;
    args.max_seqlen_q      = max_phys;
    args.hdim_v            = hdim;
    args.nhead_q           = nhead;
    args.nhead_k           = nhead;
    args.stride_o          = hdim;
    args.stride_do         = hdim;
    args.nhead_stride_o    = max_phys * hdim;
    args.nhead_stride_do   = max_phys * hdim;
    args.nhead_stride_lsed = max_phys;
    args.p_undrop          = 1.0f;

    using DotTileTraits = ck_tile::TileFmhaBwdOGradDotOTraits<true, true, 2>;
    using DotProblem =
        ck_tile::BlockFmhaBwdOGradDotOPipelineProblem<typename TypeConfig::ODataType,
                                                      typename TypeConfig::OGradDataType,
                                                      typename TypeConfig::DDataType,
                                                      64,
                                                      hdim,
                                                      true,
                                                      DotTileTraits>;
    using DotPipeline = ck_tile::BlockFmhaBwdOGradDotO<DotProblem>;
    using DotKernel   = ck_tile::FmhaBwdOGradDotOKernel<DotPipeline>;

    auto [dot_kargs, dot_grids] = fmha_bwd_dot_do_o_create_kargs_and_grids<DotKernel>(args);
    const dim3 dot_blocks       = DotKernel::BlockSize();
    constexpr ck_tile::index_t kDotBlockPerCu = DotKernel::kBlockPerCu;
    auto dot_kernel =
        ck_tile::make_kernel<kDotBlockPerCu>(DotKernel{}, dot_grids, dot_blocks, 0, dot_kargs);
    dot_kernel(kStreamConfig);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    auto d_result_host = CopyDeviceToHost<float>(d_dev, total_rows);
    auto [rtol, atol]  = get_elimit<TypeParam>(hdim, hdim);
    for(size_t i = 0; i < d_result_host.size(); ++i)
    {
        SCOPED_TRACE(::testing::Message() << "index=" << i);
        if(std::fabs(expected[i] - sentinel_ref[i]) < 1e-6f)
            EXPECT_FLOAT_EQ(d_result_host[i], sentinel_ref[i]);
        else
            EXPECT_NEAR(d_result_host[i], expected[i], static_cast<float>(atol));
    }
}

TYPED_TEST(FmhaBwdKernelPaddingTyped, OGradDotO_VariedPhysical_NoLogicalPtr)
{
    constexpr ck_tile::index_t batch    = 3;
    constexpr ck_tile::index_t nhead    = 1;
    constexpr ck_tile::index_t hdim     = 64;
    constexpr ck_tile::index_t phys_r0  = 5;
    constexpr ck_tile::index_t phys_r1  = 7;
    constexpr ck_tile::index_t phys_r2  = 4;
    constexpr ck_tile::index_t max_phys = phys_r1;

    const std::vector<int32_t> seqstart_q_host{
        0, phys_r0, phys_r0 + phys_r1, phys_r0 + phys_r1 + phys_r2};
    const ck_tile::index_t total_rows = seqstart_q_host.back();

    using TypeConfig = FmhaBwdTypeConfig<TypeParam>;
    using OType      = typename TypeConfig::ODataType;
    using DOType     = typename TypeConfig::OGradDataType;
    using DType      = typename TypeConfig::DDataType;

    ck_tile::HostTensor<OType> o_host({1, nhead, total_rows, hdim});
    ck_tile::HostTensor<DOType> do_host({1, nhead, total_rows, hdim});
    ck_tile::HostTensor<DType> d_init_host({1, nhead, total_rows});
    ck_tile::FillConstant<OType>{ck_tile::type_convert<OType>(1.0f)}(o_host);
    ck_tile::FillConstant<DOType>{ck_tile::type_convert<DOType>(2.0f)}(do_host);
    ck_tile::FillConstant<DType>{ck_tile::type_convert<DType>(SentinelValue())}(d_init_host);

    std::vector<float> expected(static_cast<size_t>(total_rows), SentinelValue());
    const float dot = 2.0f * 1.0f * static_cast<float>(hdim);
    // seqlen_q_ptr is null; logical lengths equal physical lengths per group
    for(int r = 0; r < phys_r0; ++r)
        expected[0 + r] = dot;
    for(int r = 0; r < phys_r1; ++r)
        expected[phys_r0 + r] = dot;
    for(int r = 0; r < phys_r2; ++r)
        expected[phys_r0 + phys_r1 + r] = dot;
    std::vector<float> sentinel_ref(static_cast<size_t>(total_rows), SentinelValue());

    ck_tile::DeviceMem o_dev(o_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem do_dev(do_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem d_dev(d_init_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem seqstart_dev(seqstart_q_host.size() * sizeof(int32_t));
    o_dev.ToDevice(o_host.data());
    do_dev.ToDevice(do_host.data());
    d_dev.ToDevice(d_init_host.data());
    seqstart_dev.ToDevice(seqstart_q_host.data());

    fmha_bwd_args args{};
    args.o_ptr             = o_dev.GetDeviceBuffer();
    args.do_ptr            = do_dev.GetDeviceBuffer();
    args.d_ptr             = d_dev.GetDeviceBuffer();
    args.seqstart_q_ptr    = seqstart_dev.GetDeviceBuffer();
    args.seqlen_q_ptr      = nullptr; // no logical len ptr
    args.batch             = batch;
    args.max_seqlen_q      = max_phys;
    args.hdim_v            = hdim;
    args.nhead_q           = nhead;
    args.nhead_k           = nhead;
    args.stride_o          = hdim;
    args.stride_do         = hdim;
    args.nhead_stride_o    = max_phys * hdim;
    args.nhead_stride_do   = max_phys * hdim;
    args.nhead_stride_lsed = max_phys;
    args.p_undrop          = 1.0f;

    using DotTileTraits = ck_tile::TileFmhaBwdOGradDotOTraits<true, true, 2>;
    using DotProblem =
        ck_tile::BlockFmhaBwdOGradDotOPipelineProblem<typename TypeConfig::ODataType,
                                                      typename TypeConfig::OGradDataType,
                                                      typename TypeConfig::DDataType,
                                                      64,
                                                      hdim,
                                                      true,
                                                      DotTileTraits>;
    using DotPipeline = ck_tile::BlockFmhaBwdOGradDotO<DotProblem>;
    using DotKernel   = ck_tile::FmhaBwdOGradDotOKernel<DotPipeline>;

    auto [dot_kargs, dot_grids] = fmha_bwd_dot_do_o_create_kargs_and_grids<DotKernel>(args);
    const dim3 dot_blocks       = DotKernel::BlockSize();
    constexpr ck_tile::index_t kDotBlockPerCu = DotKernel::kBlockPerCu;
    auto dot_kernel =
        ck_tile::make_kernel<kDotBlockPerCu>(DotKernel{}, dot_grids, dot_blocks, 0, dot_kargs);
    dot_kernel(kStreamConfig);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    auto d_result_host = CopyDeviceToHost<float>(d_dev, total_rows);
    auto [rtol, atol]  = get_elimit<TypeParam>(hdim, hdim);
    for(size_t i = 0; i < d_result_host.size(); ++i)
    {
        SCOPED_TRACE(::testing::Message() << "index=" << i);
        if(std::fabs(expected[i] - sentinel_ref[i]) < 1e-6f)
            EXPECT_FLOAT_EQ(d_result_host[i], sentinel_ref[i]);
        else
            EXPECT_NEAR(d_result_host[i], expected[i], static_cast<float>(atol));
    }
}

TYPED_TEST(FmhaBwdKernelPaddingTyped, ConvertQGrad_GroupPaddingAndZeroLength)
{
    constexpr ck_tile::index_t batch = 3;
    constexpr ck_tile::index_t nhead = 1;
    constexpr ck_tile::index_t hdim  = 128;

    const std::vector<int32_t> seqstart_q_host{0, 6, 6, 10}; // physical lengths: 6,0,4
    const std::vector<int32_t> seqlen_q_host{4, 0, 3};
    const std::vector<int32_t> seqstart_k_host{0, 7, 15, 18};
    const std::vector<int32_t> seqlen_k_host{5, 8, 3};

    const ck_tile::index_t total_rows_q = seqstart_q_host.back();

    using TypeConfigC = FmhaBwdTypeConfig<TypeParam>;
    using AccType     = typename TypeConfigC::AccDataType;   // float
    using QGradType   = typename TypeConfigC::QGradDataType; // bf16

    ck_tile::HostTensor<AccType> dq_acc_host({1, nhead, total_rows_q, hdim});
    ck_tile::HostTensor<QGradType> dq_host_init({1, nhead, total_rows_q, hdim});

    const float dq_acc_const = 1.25f;
    ck_tile::FillConstant<AccType>{ck_tile::type_convert<AccType>(dq_acc_const)}(dq_acc_host);
    ck_tile::FillConstant<QGradType>{ck_tile::type_convert<QGradType>(SentinelValue())}(
        dq_host_init);

    const float dq_sentinel_val =
        ck_tile::type_convert<float>(ck_tile::type_convert<QGradType>(SentinelValue()));
    std::vector<float> dq_sentinel_ref(static_cast<size_t>(total_rows_q * hdim), dq_sentinel_val);
    std::vector<float> expected = dq_sentinel_ref;
    for(ck_tile::index_t b = 0; b < batch; ++b)
    {
        const ck_tile::index_t q_start = seqstart_q_host[b];
        const ck_tile::index_t q_len   = seqlen_q_host[b];
        for(ck_tile::index_t row = 0; row < q_len; ++row)
        {
            for(ck_tile::index_t c = 0; c < hdim; ++c)
            {
                const size_t idx = (q_start + row) * hdim + c;
                // dq_acc_host is [1, nhead, s, d]
                expected[idx] = ck_tile::type_convert<float>(dq_acc_host(0, 0, q_start + row, c));
            }
        }
    }

    ck_tile::DeviceMem dq_acc_dev(dq_acc_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem dq_dev(dq_host_init.get_element_space_size_in_bytes());
    ck_tile::DeviceMem seqstart_q(seqstart_q_host.size() * sizeof(int32_t));
    ck_tile::DeviceMem seqstart_k(seqstart_k_host.size() * sizeof(int32_t));
    ck_tile::DeviceMem seqlen_q_dev(seqlen_q_host.size() * sizeof(int32_t));
    ck_tile::DeviceMem seqlen_k_dev(seqlen_k_host.size() * sizeof(int32_t));

    dq_acc_dev.ToDevice(dq_acc_host.data());
    dq_dev.ToDevice(dq_host_init.data());
    seqstart_q.ToDevice(seqstart_q_host.data());
    seqstart_k.ToDevice(seqstart_k_host.data());
    seqlen_q_dev.ToDevice(seqlen_q_host.data());
    seqlen_k_dev.ToDevice(seqlen_k_host.data());

    fmha_bwd_args args{};
    args.dq_acc_ptr          = dq_acc_dev.GetDeviceBuffer();
    args.dq_ptr              = dq_dev.GetDeviceBuffer();
    args.seqstart_q_ptr      = seqstart_q.GetDeviceBuffer();
    args.seqstart_k_ptr      = seqstart_k.GetDeviceBuffer();
    args.seqlen_q_ptr        = seqlen_q_dev.GetDeviceBuffer();
    args.seqlen_k_ptr        = seqlen_k_dev.GetDeviceBuffer();
    args.batch               = batch;
    args.nhead_q             = nhead;
    args.nhead_k             = nhead;
    args.hdim_q              = hdim;
    args.hdim_v              = hdim;
    args.max_seqlen_q        = 6;
    args.max_seqlen_k        = 8;
    args.stride_dq_acc       = hdim;
    args.stride_dq           = hdim;
    args.nhead_stride_dq_acc = hdim * args.max_seqlen_q;
    args.nhead_stride_dq     = hdim * args.max_seqlen_q;
    args.nhead_stride_q      = 0;
    args.nhead_stride_k      = 0;
    args.nhead_stride_v      = 0;
    args.nhead_stride_o      = 0;
    args.batch_stride_dq_acc = 0;
    args.batch_stride_dq     = 0;
    args.split_stride_dq_acc = args.max_seqlen_q * args.stride_dq_acc;
    args.window_size_left    = -1;
    args.window_size_right   = 0;
    args.mask_type           = static_cast<ck_tile::index_t>(mask_enum::no_mask);
    args.p_drop              = 0.0f;
    args.p_undrop            = 1.0f;
    args.drop_seed_offset    = std::make_pair(uint64_t{0}, uint64_t{0});

    using TypeConfig        = FmhaBwdTypeConfig<TypeParam>;
    using ConvertTileTraits = ck_tile::TileFmhaBwdConvertQGradTraits<true, true, 2>;
    using ConvertProblem =
        ck_tile::BlockFmhaBwdConvertQGradPipelineProblem<typename TypeConfig::AccDataType,
                                                         typename TypeConfig::QGradDataType,
                                                         256,
                                                         64,
                                                         0,
                                                         hdim,
                                                         true,
                                                         false,
                                                         ConvertTileTraits>;
    using ConvertPipeline = ck_tile::BlockFmhaBwdConvertQGrad<ConvertProblem>;
    using ConvertKernel   = ck_tile::FmhaBwdConvertQGradKernel<ConvertPipeline>;

    auto [convert_kargs, convert_grids] =
        fmha_bwd_convert_dq_create_kargs_and_grids<ConvertKernel>(args);
    const dim3 convert_blocks                     = ConvertKernel::BlockSize();
    constexpr ck_tile::index_t kConvertBlockPerCu = ConvertKernel::kBlockPerCu;
    auto convert_kernel                           = ck_tile::make_kernel<kConvertBlockPerCu>(
        ConvertKernel{}, convert_grids, convert_blocks, 0, convert_kargs);
    convert_kernel(kStreamConfig);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    using QGradOutT       = typename TypeConfigC::QGradDataType;
    auto dq_result_host_t = CopyDeviceToHost<QGradOutT>(dq_dev, total_rows_q * hdim);
    auto dq_result_host   = ToFloatVector(dq_result_host_t);

    auto [rtol_gpad, atol_gpad] = get_elimit<TypeParam>(hdim, hdim);
    for(size_t i = 0; i < dq_result_host.size(); ++i)
    {
        SCOPED_TRACE(::testing::Message() << "index=" << i);
        if(std::fabs(expected[i] - dq_sentinel_ref[i]) < 1e-6f)
        {
            EXPECT_FLOAT_EQ(dq_result_host[i], dq_sentinel_ref[i]);
        }
        else
        {
            EXPECT_NEAR(dq_result_host[i], expected[i], static_cast<float>(atol_gpad));
        }
    }
}

TYPED_TEST(FmhaBwdKernelPaddingTyped, ConvertQGrad_DeterministicPaddingUsesLogicalLengths)
{
    constexpr ck_tile::index_t batch        = 1;
    constexpr ck_tile::index_t nhead        = 1;
    constexpr ck_tile::index_t hdim         = 128;
    constexpr ck_tile::index_t phys_rows    = 8;
    constexpr ck_tile::index_t logical_rows = 5;
    constexpr ck_tile::index_t phys_k       = 24;
    constexpr ck_tile::index_t logical_k    = 20;
    constexpr ck_tile::index_t kN0          = 16;
    constexpr ck_tile::index_t nsplits      = (logical_k + kN0 - 1) / kN0;

    const std::vector<int32_t> seqstart_q_host{0, phys_rows};
    const std::vector<int32_t> seqlen_q_host{logical_rows};
    const std::vector<int32_t> seqstart_k_host{0, phys_k};
    const std::vector<int32_t> seqlen_k_host{logical_k};
    const ck_tile::index_t total_rows_q = seqstart_q_host.back();

    using TypeConfigD = FmhaBwdTypeConfig<TypeParam>;
    using AccTypeDet  = typename TypeConfigD::AccDataType;   // float
    using QGradTypeD  = typename TypeConfigD::QGradDataType; // bf16

    ck_tile::HostTensor<AccTypeDet> dq_acc_host({nsplits, 1, nhead, phys_rows, hdim});
    dq_acc_host.ForEach([&](auto& self, auto idx) {
        const float s = static_cast<float>(idx[0]);
        // Use split-dependent constant to avoid per-element variance and rounding interplay
        self(idx) = ck_tile::type_convert<AccTypeDet>(1.0f + 0.1f * s);
    });

    const float dq_sentinel_val_det =
        ck_tile::type_convert<float>(ck_tile::type_convert<QGradTypeD>(SentinelValue()));
    std::vector<float> expected(total_rows_q * hdim, dq_sentinel_val_det);
    // Expected is the sum over splits of the constant (1.0 + 0.1*s)
    for(ck_tile::index_t row = 0; row < logical_rows; ++row)
        for(ck_tile::index_t c = 0; c < hdim; ++c)
        {
            float acc = 0.f;
            for(ck_tile::index_t s = 0; s < nsplits; ++s)
            {
                acc += (1.0f + 0.1f * static_cast<float>(s));
            }
            expected[row * hdim + c] = acc;
        }

    ck_tile::HostTensor<QGradTypeD> dq_init({1, nhead, total_rows_q, hdim});
    ck_tile::FillConstant<QGradTypeD>{ck_tile::type_convert<QGradTypeD>(SentinelValue())}(dq_init);

    DeviceMem dq_acc_dev(dq_acc_host.get_element_space_size_in_bytes());
    DeviceMem dq_dev(dq_init.get_element_space_size_in_bytes());
    DeviceMem seqstart_q(seqstart_q_host.size() * sizeof(int32_t));
    DeviceMem seqstart_k(seqstart_k_host.size() * sizeof(int32_t));
    DeviceMem seqlen_q_dev(seqlen_q_host.size() * sizeof(int32_t));
    DeviceMem seqlen_k_dev(seqlen_k_host.size() * sizeof(int32_t));

    dq_acc_dev.ToDevice(dq_acc_host.data());
    dq_dev.ToDevice(dq_init.data());
    seqstart_q.ToDevice(seqstart_q_host.data());
    seqstart_k.ToDevice(seqstart_k_host.data());
    seqlen_q_dev.ToDevice(seqlen_q_host.data());
    seqlen_k_dev.ToDevice(seqlen_k_host.data());

    fmha_bwd_args args{};
    args.dq_acc_ptr          = dq_acc_dev.GetDeviceBuffer();
    args.dq_ptr              = dq_dev.GetDeviceBuffer();
    args.seqstart_q_ptr      = seqstart_q.GetDeviceBuffer();
    args.seqstart_k_ptr      = seqstart_k.GetDeviceBuffer();
    args.seqlen_q_ptr        = seqlen_q_dev.GetDeviceBuffer();
    args.seqlen_k_ptr        = seqlen_k_dev.GetDeviceBuffer();
    args.batch               = batch;
    args.nhead_q             = nhead;
    args.nhead_k             = nhead;
    args.hdim_q              = hdim;
    args.hdim_v              = hdim;
    args.max_seqlen_q        = phys_rows;
    args.max_seqlen_k        = phys_k;
    args.stride_dq_acc       = hdim;
    args.stride_dq           = hdim;
    args.nhead_stride_dq_acc = phys_rows * hdim;
    args.nhead_stride_dq     = phys_rows * hdim;
    args.split_stride_dq_acc = phys_rows * hdim;
    args.window_size_left    = -1;
    args.window_size_right   = 0;
    args.mask_type           = static_cast<ck_tile::index_t>(mask_enum::no_mask);
    args.p_drop              = 0.0f;
    args.p_undrop            = 1.0f;
    args.drop_seed_offset    = std::make_pair(uint64_t{0}, uint64_t{0});

    using TypeConfig    = FmhaBwdTypeConfig<TypeParam>;
    using TileTraitsDet = ck_tile::TileFmhaBwdConvertQGradTraits<true, true, 2>;
    using PipelineProblemDet =
        ck_tile::BlockFmhaBwdConvertQGradPipelineProblem<typename TypeConfig::AccDataType,
                                                         typename TypeConfig::QGradDataType,
                                                         256,
                                                         64,
                                                         kN0,
                                                         hdim,
                                                         true,
                                                         true,
                                                         TileTraitsDet>;
    using PipelineDet      = ck_tile::BlockFmhaBwdConvertQGrad<PipelineProblemDet>;
    using ConvertKernelDet = ck_tile::FmhaBwdConvertQGradKernel<PipelineDet>;

    auto [convert_kargs, convert_grids] =
        fmha_bwd_convert_dq_create_kargs_and_grids<ConvertKernelDet>(args);
    const dim3 convert_blocks                     = ConvertKernelDet::BlockSize();
    constexpr ck_tile::index_t kConvertBlockPerCu = ConvertKernelDet::kBlockPerCu;
    auto convert_kernel                           = ck_tile::make_kernel<kConvertBlockPerCu>(
        ConvertKernelDet{}, convert_grids, convert_blocks, 0, convert_kargs);
    convert_kernel(kStreamConfig);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    using QGradOutTD      = typename TypeConfigD::QGradDataType;
    auto dq_result_host_t = CopyDeviceToHost<QGradOutTD>(dq_dev, total_rows_q * hdim);
    auto dq_result_host   = ToFloatVector(dq_result_host_t);

    const float dq_sentinel_val2 = dq_sentinel_val_det;

    auto [rtol_det, atol_det] = get_elimit<TypeParam>(hdim, hdim);
    for(size_t i = 0; i < dq_result_host.size(); ++i)
    {
        SCOPED_TRACE(::testing::Message() << "index=" << i);
        if(std::fabs(expected[i] - dq_sentinel_val2) < 1e-6f)
        {
            EXPECT_FLOAT_EQ(dq_result_host[i], dq_sentinel_val2);
        }
        else
        {
            EXPECT_NEAR(dq_result_host[i], expected[i], static_cast<float>(atol_det));
        }
    }
}
