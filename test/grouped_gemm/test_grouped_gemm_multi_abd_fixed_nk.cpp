// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <tuple>
#include <vector>
#include <type_traits>

#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/ck.hpp"
#include "ck/utility/type.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "profiler/profile_grouped_gemm_multi_abd_fixed_nk_impl.hpp"

#include "gtest/gtest.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wno-unknown-warning-option"
#pragma clang diagnostic ignored "-Wlifetime-safety-invalidation"
using FP32 = float;
using FP16 = ck::half_t;
using BF16 = ck::bhalf_t;
using I8   = int8_t;
using Row  = ck::tensor_layout::gemm::RowMajor;
using Col  = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using AddFastGelu = ck::tensor_operation::element_wise::AddFastGelu;
using Add         = ck::tensor_operation::element_wise::Add;
using Multiply    = ck::tensor_operation::element_wise::Multiply;
using FastGelu    = ck::tensor_operation::element_wise::FastGelu;

// clang-format off
using KernelTypes = ::testing::Types<
    std::tuple<ck::Tuple<BF16>, ck::Tuple<I8, BF16>, ck::Tuple<BF16>, BF16, ck::Tuple<Row>, ck::Tuple<Col, Col>, ck::Tuple<Row>, Row, AddFastGelu>,
    std::tuple<ck::Tuple<BF16>, ck::Tuple<I8, BF16>, ck::Tuple<BF16>, BF16, ck::Tuple<Col>, ck::Tuple<Row, Row>, ck::Tuple<Row>, Row, AddFastGelu>,
    std::tuple<ck::Tuple<BF16>, ck::Tuple<I8, BF16>, ck::Tuple<BF16>, BF16, ck::Tuple<Row>, ck::Tuple<Row, Row>, ck::Tuple<Row>, Row, AddFastGelu>,
    std::tuple<ck::Tuple<BF16>, ck::Tuple<I8, BF16>, ck::Tuple<BF16>, BF16, ck::Tuple<Row>, ck::Tuple<Col, Col>, ck::Tuple<Row>, Row, Add>,
    std::tuple<ck::Tuple<BF16>, ck::Tuple<I8, BF16>, ck::Tuple<BF16>, BF16, ck::Tuple<Col>, ck::Tuple<Row, Row>, ck::Tuple<Row>, Row, Add>,
    std::tuple<ck::Tuple<BF16>, ck::Tuple<I8, BF16>, ck::Tuple<BF16>, BF16, ck::Tuple<Row>, ck::Tuple<Row, Row>, ck::Tuple<Row>, Row, Add>,
    std::tuple<ck::Tuple<BF16>, ck::Tuple<I8, BF16>, ck::Tuple<>,     BF16, ck::Tuple<Row>, ck::Tuple<Col, Col>, ck::Tuple<>,    Row, PassThrough>,
    std::tuple<ck::Tuple<BF16>, ck::Tuple<I8, BF16>, ck::Tuple<>,     BF16, ck::Tuple<Col>, ck::Tuple<Row, Row>, ck::Tuple<>,    Row, PassThrough>,
    std::tuple<ck::Tuple<BF16>, ck::Tuple<I8, BF16>, ck::Tuple<>,     BF16, ck::Tuple<Row>, ck::Tuple<Row, Row>, ck::Tuple<>,    Row, PassThrough>,
    std::tuple<ck::Tuple<BF16>, ck::Tuple<I8, BF16>, ck::Tuple<>,     BF16, ck::Tuple<Row>, ck::Tuple<Col, Col>, ck::Tuple<>,    Row, FastGelu>,
    std::tuple<ck::Tuple<BF16>, ck::Tuple<I8, BF16>, ck::Tuple<>,     BF16, ck::Tuple<Col>, ck::Tuple<Row, Row>, ck::Tuple<>,    Row, FastGelu>,
    std::tuple<ck::Tuple<BF16>, ck::Tuple<I8, BF16>, ck::Tuple<>,     BF16, ck::Tuple<Row>, ck::Tuple<Row, Row>, ck::Tuple<>,    Row, FastGelu>
>;
// clang-format on

template <typename Tuple>
class TestGroupedGemmMultiABDFixedNK : public testing::Test
{
    protected:
    using AsDataType   = std::tuple_element_t<0, Tuple>;
    using BsDataType   = std::tuple_element_t<1, Tuple>;
    using DsDataType   = std::tuple_element_t<2, Tuple>;
    using EDataType    = std::tuple_element_t<3, Tuple>;
    using AccDataType  = float;
    using AsLayout     = std::tuple_element_t<4, Tuple>;
    using BsLayout     = std::tuple_element_t<5, Tuple>;
    using DsLayout     = std::tuple_element_t<6, Tuple>;
    using ELayout      = std::tuple_element_t<7, Tuple>;
    using AElementOp   = PassThrough;
    using BElementOp   = Multiply;
    using CDEElementOp = std::tuple_element_t<8, Tuple>;

    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    public:
    static constexpr bool verify_     = true;
    static constexpr int init_method_ = 1; // integer value initialization
    static constexpr bool log_        = false;
    static constexpr bool bench_      = false; // measure kernel performance
    static constexpr int n_warmup_    = 0;
    static constexpr int n_iter_      = 1;

    std::vector<int> k_batches_ = {1};

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
             const std::vector<int>& StrideE  = {})
    {
        std::vector<int> stride_as = StrideAs;
        std::vector<int> stride_bs = StrideBs;
        std::vector<int> stride_ds = StrideDs;
        std::vector<int> stride_e  = StrideE;

        if(stride_as.empty())
        {
            SetTupleStrides<AsLayout>(stride_as, Ms, Ks);
        }
        if(stride_bs.empty())
        {
            SetTupleStrides<BsLayout>(stride_bs, Ks, Ns);
        }
        if(stride_ds.empty())
        {
            SetTupleStrides<DsLayout>(stride_ds, Ms, Ns);
        }
        if(stride_e.empty())
        {
            SetStrides<ELayout>(stride_e, Ms, Ns);
        }

        RunSingle(Ms, Ns, Ks, stride_as, stride_bs, stride_ds, stride_e);
    }

    void RunSingle(const std::vector<int>& Ms,
                   const std::vector<int>& Ns,
                   const std::vector<int>& Ks,
                   const std::vector<int>& StrideAs,
                   const std::vector<int>& StrideBs,
                   const std::vector<int>& StrideDs,
                   const std::vector<int>& StrideE)
    {
        bool pass =
            ck::profiler::profile_grouped_gemm_multi_abd_fixed_nk_impl<AsDataType,
                                                                       BsDataType,
                                                                       DsDataType,
                                                                       EDataType,
                                                                       AccDataType,
                                                                       AsLayout,
                                                                       BsLayout,
                                                                       DsLayout,
                                                                       ELayout,
                                                                       AElementOp,
                                                                       BElementOp,
                                                                       CDEElementOp>(verify_,
                                                                                     init_method_,
                                                                                     log_,
                                                                                     bench_,
                                                                                     Ms,
                                                                                     Ns,
                                                                                     Ks,
                                                                                     StrideAs,
                                                                                     StrideBs,
                                                                                     StrideDs,
                                                                                     StrideE,
                                                                                     k_batches_,
                                                                                     n_warmup_,
                                                                                     n_iter_);
        EXPECT_TRUE(pass);
    }
};

TYPED_TEST_SUITE(TestGroupedGemmMultiABDFixedNK, KernelTypes);

TYPED_TEST(TestGroupedGemmMultiABDFixedNK, TinyCases)
{
    const std::vector<int> Ms{3, 4};
    constexpr int N = 8;
    constexpr int K = 64;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);

    this->Run(Ms, Ns, Ks);
}

TYPED_TEST(TestGroupedGemmMultiABDFixedNK, SmallCases)
{
    const std::vector<int> Ms{3, 5, 16, 7, 8};
    constexpr int N = 768;
    constexpr int K = 544;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);

    this->Run(Ms, Ns, Ks);
}

TYPED_TEST(TestGroupedGemmMultiABDFixedNK, MidCases)
{
    const std::vector<int> Ms{167, 183, 177, 153, 139, 204};
    constexpr int N = 768;
    constexpr int K = 544;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);

    this->Run(Ms, Ns, Ks);
}

TYPED_TEST(TestGroupedGemmMultiABDFixedNK, Regular)
{
    const std::vector<int> Ms{64, 128, 256};
    constexpr int N = 768;
    constexpr int K = 320;

    const std::vector<int> Ns(Ms.size(), N);
    const std::vector<int> Ks(Ms.size(), K);

    this->Run(Ms, Ns, Ks);
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#pragma clang diagnostic pop
