// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <string>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <vector>
#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "example/ck_tile/17_grouped_gemm/grouped_gemm_multi_d.hpp"
#include "profiler/profile_grouped_gemm_tile_loop_generic_impl.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-invalidation"
extern ck::index_t param_mask;
extern ck::index_t instance_index;

namespace ck {
namespace test {

template <typename Tuple, bool FailIfNoSupportedInstances = false>
class TestGroupedGemmTileLoop : public testing::Test
{
    protected:
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    using ALayout      = tuple_element_t<0, Tuple>;
    using BLayout      = tuple_element_t<1, Tuple>;
    using DsLayout     = tuple_element_t<2, Tuple>;
    using ELayout      = tuple_element_t<3, Tuple>;
    using ADataType    = tuple_element_t<4, Tuple>;
    using BDataType    = tuple_element_t<5, Tuple>;
    using DsDataType   = tuple_element_t<6, Tuple>;
    using EDataType    = tuple_element_t<7, Tuple>;
    using AElementOp   = tuple_element_or_t<8, Tuple, PassThrough>;
    using BElementOp   = tuple_element_or_t<9, Tuple, PassThrough>;
    using CDEElementOp = tuple_element_or_t<10, Tuple, PassThrough>;

    using Row = ck::tensor_layout::gemm::RowMajor;
    using Col = ck::tensor_layout::gemm::ColumnMajor;

    static constexpr auto NumDTensor = DsLayout::Size();

    public:
    static constexpr bool verify_     = true;
    static constexpr int init_method_ = 1; // integer value initialization
    static constexpr bool log_        = false;
    static constexpr bool bench_      = false; // measure kernel performance
    static constexpr int n_warmup_    = 0;
    static constexpr int n_iter_      = 1;

    bool fail_if_no_supported_instances_ = FailIfNoSupportedInstances;

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

    public:
    void Run(const std::vector<int>& Ms,
             const std::vector<int>& Ns,
             const std::vector<int>& Ks,
             const std::vector<int>& StrideAs                         = {},
             const std::vector<int>& StrideBs                         = {},
             const std::vector<std::array<int, NumDTensor>>& StrideDs = {},
             const std::vector<int>& StrideEs                         = {})
    {
        std::vector<int> stride_as                         = StrideAs;
        std::vector<int> stride_bs                         = StrideBs;
        std::vector<std::array<int, NumDTensor>> stride_ds = StrideDs;
        std::vector<int> stride_es                         = StrideEs;

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
            for(size_t group = 0; group < Ms.size(); ++group)
            {
                std::array<int, NumDTensor> d_strides;
                static_for<0, NumDTensor, 1>{}([&](auto i) {
                    using DLayout = tuple_element_t<i, DsLayout>;

                    if(std::is_same_v<DLayout, Row>)
                    {
                        d_strides[i] = Ns[group];
                    }
                    else if(std::is_same_v<DLayout, Col>)
                    {
                        d_strides[i] = Ms[group];
                    }
                });

                stride_ds.emplace_back(d_strides);
            }
        }

        if(stride_es.empty())
        {
            SetStrides<ELayout>(stride_es, Ms, Ns);
        }

        RunSingle(Ms, Ns, Ks, stride_as, stride_bs, stride_ds, stride_es);
    }

    void RunSingle(const std::vector<int>& Ms,
                   const std::vector<int>& Ns,
                   const std::vector<int>& Ks,
                   const std::vector<int>& StrideAs,
                   const std::vector<int>& StrideBs,
                   const std::vector<std::array<int, NumDTensor>>& StrideDs,
                   const std::vector<int>& StrideEs)
    {
        bool pass =
            ck::profiler::profile_grouped_gemm_tile_loop_generic_impl<ADataType,
                                                                      BDataType,
                                                                      DsDataType,
                                                                      EDataType,
                                                                      ALayout,
                                                                      BLayout,
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
                                                                                    StrideEs,
                                                                                    n_warmup_,
                                                                                    n_iter_);
        EXPECT_TRUE(pass);
    }
};

} // namespace test
} // namespace ck
#pragma clang diagnostic pop
