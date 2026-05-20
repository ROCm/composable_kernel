// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_wmma_fixed_nk.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/number.hpp"
#include "ck/utility/tuple.hpp"
#include "profiler/profile_grouped_gemm_fixed_nk_bias_impl.hpp"
#include "gtest/gtest.h"

#include <iostream>
#include <tuple>
#include <type_traits>
#include <vector>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-invalidation"
static ck::index_t param_mask = 0xffffff;

using FP32 = float;
using FP16 = ck::half_t;
using BF16 = ck::bhalf_t;
using Row  = ck::tensor_layout::gemm::RowMajor;
using Col  = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using SplitKAdd   = ck::tensor_operation::element_wise::SplitKAdd;

using CDEElementOp = SplitKAdd;

// clang-format off
using KernelTypes = ::testing::Types<
    std::tuple<FP16, FP16, ck::Tuple<FP16>, FP16, Row, Row, ck::Tuple<Row>, Row>,
    std::tuple<FP16, FP16, ck::Tuple<FP16>, FP16, Row, Col, ck::Tuple<Row>, Row>
>;
// clang-format on

template <typename Tuple>
class TestGroupedGemmFixedNKBias : public testing::Test
{
    protected:
    using ADataType   = std::tuple_element_t<0, Tuple>;
    using BDataType   = std::tuple_element_t<1, Tuple>;
    using DsDataType  = std::tuple_element_t<2, Tuple>;
    using EDataType   = std::tuple_element_t<3, Tuple>;
    using ALayout     = std::tuple_element_t<4, Tuple>;
    using BLayout     = std::tuple_element_t<5, Tuple>;
    using DsLayout    = std::tuple_element_t<6, Tuple>;
    using ELayout     = std::tuple_element_t<7, Tuple>;
    using AccDataType = FP32;

    public:
    static constexpr bool verify_     = true;
    static constexpr int init_method_ = 1; // integer value initialization
    static constexpr bool log_        = false;
    static constexpr bool bench_      = false; // measure kernel performance
    static constexpr int n_warmup_    = 0;
    static constexpr int n_iter_      = 1;

    std::vector<int> k_batches_;

    bool IsSplitKSupported()
    {
        // gfx11 does not support split-K due to missing atomic add for fp16/bf16
        // Technically, we could still use split-K for fp32, but we currently don't have
        // instances for it so we disable it entirely
        constexpr bool require_16bit_atomic_add =
            std::is_same_v<EDataType, FP16> || std::is_same_v<EDataType, BF16>;
        bool missing_atomic_add = require_16bit_atomic_add && ck::is_gfx11_supported();

        // CDE element operators are not supported in combination with split K
        constexpr bool has_cde_element_operator =
            !std::is_same_v<CDEElementOp, PassThrough> && !std::is_same_v<CDEElementOp, SplitKAdd>;

        return !missing_atomic_add && !has_cde_element_operator;
    }

    void SetUp() override
    {
        if(!IsSplitKSupported())
        {
            k_batches_ = {1};
        }
        else
        {
            k_batches_ = {1, 2, 3, 4, 8};
        }
    }

    private:
    template <typename Layout>
    void SetStrides(std::vector<int>& strides,
                    const std::vector<int>& rows,
                    const std::vector<int>& cols) const
    {
        if(std::is_same_v<Layout, Row>)
        {
            for(const auto c : cols)
            {
                strides.emplace_back(c);
            }
        }
        else if(std::is_same_v<Layout, Col>)
        {
            for(const auto r : rows)
            {
                strides.emplace_back(r);
            }
        }
    }

    template <typename Layouts>
    void SetTupleStrides(std::vector<int>& strides,
                         const std::vector<int>& rows,
                         const std::vector<int>& cols) const
    {
        if constexpr(Layouts::Size() > 0)
        {
            // As of now multi ABD implementation supports only tensors with matching layouts.
            using Layout = ck::remove_cvref_t<ck::tuple_element_t<ck::Number<0>{}, Layouts>>;
            SetStrides<Layout>(strides, rows, cols);
        }
    }

    public:
    void Run(const std::vector<int>& Ms,
             const std::vector<int>& Ns,
             const std::vector<int>& Ks,
             const std::vector<int>& StrideAs = {},
             const std::vector<int>& StrideBs = {},
             const std::vector<int>& StrideDs = {},
             const std::vector<int>& StrideEs = {})
    {
        std::vector<int> stride_as = StrideAs;
        std::vector<int> stride_bs = StrideBs;
        std::vector<int> stride_ds = StrideDs;
        std::vector<int> stride_es = StrideEs;

        if(stride_as.empty())
        {
            SetStrides<ALayout>(stride_as, Ms, Ks);
        }
        if(stride_bs.empty())
        {
            SetStrides<BLayout>(stride_bs, Ks, Ns);
        }
        if(stride_ds.empty())
        {
            SetTupleStrides<DsLayout>(stride_ds, Ms, Ns);
        }
        if(stride_es.empty())
        {
            SetStrides<ELayout>(stride_es, Ms, Ns);
        }

        std::vector<int> k_batches;
        for(size_t i = 0; i < k_batches_.size(); i++)
        {
            if(param_mask & (1 << i))
            {
                k_batches.push_back(k_batches_[i]);
            }
        }

        RunSingle(Ms, Ns, Ks, stride_as, stride_bs, stride_ds, stride_es, k_batches);
    }

    void RunSingle(const std::vector<int>& Ms,
                   const std::vector<int>& Ns,
                   const std::vector<int>& Ks,
                   const std::vector<int>& StrideAs,
                   const std::vector<int>& StrideBs,
                   const std::vector<int>& StrideDs,
                   const std::vector<int>& StrideEs,
                   const std::vector<int>& kbatches)
    {
        bool pass = ck::profiler::profile_grouped_gemm_fixed_nk_bias_impl<ADataType,
                                                                          BDataType,
                                                                          DsDataType,
                                                                          EDataType,
                                                                          AccDataType,
                                                                          ALayout,
                                                                          BLayout,
                                                                          DsLayout,
                                                                          ELayout>(verify_,
                                                                                   init_method_,
                                                                                   log_,
                                                                                   bench_,
                                                                                   Ms,
                                                                                   Ns,
                                                                                   Ks,
                                                                                   StrideAs,
                                                                                   StrideBs,
                                                                                   StrideDs,
                                                                                   StrideEs,
                                                                                   kbatches,
                                                                                   n_warmup_,
                                                                                   n_iter_);
        EXPECT_TRUE(pass);
    }
};

TYPED_TEST_SUITE(TestGroupedGemmFixedNKBias, KernelTypes);

TYPED_TEST(TestGroupedGemmFixedNKBias, TinyCases)
{
    const std::vector<int> Ms{2, 1};
    constexpr int N = 768;
    constexpr int K = 544;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);

    this->Run(Ms, Ns, Ks);
}

TYPED_TEST(TestGroupedGemmFixedNKBias, SmallCases)
{
    const std::vector<int> Ms{2, 1, 3, 4, 5};
    constexpr int N = 768;
    constexpr int K = 544;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);

    this->Run(Ms, Ns, Ks);
}

TYPED_TEST(TestGroupedGemmFixedNKBias, MidCases)
{
    const std::vector<int> Ms{167, 183, 177, 153, 139, 204};
    constexpr int N = 768;
    constexpr int K = 544;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);

    this->Run(Ms, Ns, Ks);
}

TYPED_TEST(TestGroupedGemmFixedNKBias, Regular)
{
    const std::vector<int> Ms{64, 128, 256};
    constexpr int N = 768;
    constexpr int K = 320;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);

    this->Run(Ms, Ns, Ks);
}

TYPED_TEST(TestGroupedGemmFixedNKBias, MNKPadded)
{
    const std::vector<int> Ms{127, 150, 188, 210};
    constexpr int N = 136;
    constexpr int K = 280;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);

    this->Run(Ms, Ns, Ks);
}

TYPED_TEST(TestGroupedGemmFixedNKBias, TestLargeKBatch)
{
    // In some cases Split K is not supported. Running this test would fail since no instance will
    // be supported, so we skip the test
    if(!this->IsSplitKSupported())
    {
        GTEST_SKIP() << "Split-K not supported for for the current configuration (FP16/BF16 on "
                        "GFX11, or using CDE element-wise operation)";
    }

    const std::vector<int> Ms{188, 210};
    constexpr int N = 768;
    constexpr int K = 4096;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);

    this->k_batches_ = {32, 64};

    this->Run(Ms, Ns, Ks);
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    if(argc == 1) {}
    else if(argc == 2)
    {
        param_mask = strtol(argv[1], nullptr, 0);
    }
    else
    {
        std::cout << "Usage of " << argv[0] << std::endl;
        std::cout << "Arg1: param_mask " << std::endl;
    }
    return RUN_ALL_TESTS();
}
#pragma clang diagnostic pop
