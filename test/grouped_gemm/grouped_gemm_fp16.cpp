// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <random>

#include "profiler/profile_grouped_gemm_impl.hpp"

namespace {

using ADataType   = ck::half_t;
using BDataType   = ck::half_t;
using CDataType   = ck::half_t;
using AccDataType = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <typename ALayout, typename BLayout, typename CLayout>
bool TestGroupedGemm()
{

    std::mt19937 gen(19391);
    std::uniform_int_distribution<> distrib(1, 10);
    int group_count = distrib(gen);

    // GEMM shape
    std::vector<ck::tensor_operation::device::GemmDesc> gemm_descs;
    std::vector<const void*> p_a, p_b;
    std::vector<void*> p_c;

    std::vector<int> Ms, Ns, Ks, StrideAs, StrideBs, StrideCs;

    for(int i = 0; i < group_count; i++)
    {
        Ms.push_back(256 + 256 * distrib(gen));
        Ns.push_back(256 + 256 * distrib(gen));
        Ks.push_back(128 + 128 * distrib(gen));

        StrideAs.push_back(std::is_same<Row, ALayout>::value ? Ks[i] : Ms[i]);
        StrideBs.push_back(std::is_same<Row, BLayout>::value ? Ns[i] : Ks[i]);
        StrideCs.push_back(std::is_same<Row, CLayout>::value ? Ns[i] : Ms[i]);
    }

    return ck::profiler::profile_grouped_gemm_impl<ADataType,
                                                   BDataType,
                                                   CDataType,
                                                   AccDataType,
                                                   ALayout,
                                                   BLayout,
                                                   CLayout>(
        true, 1, false, 1, Ms, Ns, Ks, StrideAs, StrideBs, StrideCs);
}

} // anonymous namespace

int main()
{
    bool res = true;

    res = res && TestGroupedGemm<Row, Row, Row>();
    res = res && TestGroupedGemm<Row, Col, Row>();
    res = res && TestGroupedGemm<Col, Row, Row>();
    res = res && TestGroupedGemm<Col, Col, Row>();

    std::cout << "TestGroupedGemm ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;

    return res ? 0 : 1;
}
