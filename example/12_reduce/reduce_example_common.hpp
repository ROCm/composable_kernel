// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"

template <ck::index_t Rank, ck::index_t NumReduceDim>
std::vector<int> get_invariant_dims(const std::vector<int>& reduceDims)
{
    assert(NumReduceDim == reduceDims.size());

    int reduceFlag = 0;

    // flag the bits for the reduceDims
    for(int i = 0; i < NumReduceDim; i++)
    {
        reduceFlag |= 1 << reduceDims[i];
    };

    std::vector<int> invariantDims;

    // collect invariant dimensions
    for(int i = 0; i < Rank; i++)
        if((reduceFlag & (1 << i)) == 0)
        {
            invariantDims.push_back(i);
        };

    return invariantDims;
};

template <ck::index_t Rank, ck::index_t NumReduceDim>
struct ReduceShape
{
    static constexpr ck::index_t Rank_         = Rank;
    static constexpr ck::index_t NumReduceDim_ = NumReduceDim;
};

using reduce_shape_instances = std::tuple<ReduceShape<3, 1>,
                                          ReduceShape<3, 2>,
                                          ReduceShape<4, 1>,
                                          ReduceShape<4, 2>,
                                          ReduceShape<4, 3>,
                                          ReduceShape<5, 1>,
                                          ReduceShape<5, 2>,
                                          ReduceShape<5, 3>,
                                          ReduceShape<5, 4>>;
