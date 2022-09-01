// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/tensor_operation/gpu/device/device_sparse_embedding3_forward.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename EmbType,
          typename IndexType,
          typename GammaDataType,
          typename BetaDataType,
          typename OutType,
          ck::index_t RowPerBlock,  // MxN, along M
          ck::index_t DimPerBlock,  // MxN, along N
          typename LayernormOperation>
struct DeviceSparseEmbedding3ForwardLayernorm : public BaseOperator
{

    struct Argument : public BaseArgument
    {
        Argument(   OutType* p_output,
                    const EmbType* p_emb_a,
                    const EmbType* p_emb_b,
                    const EmbType* p_emb_c,
                    const IndexType* p_index_a,
                    const IndexType* p_index_b,
                    const IndexType* p_index_c,
                    const GammaDataType * p_gamma,
                    const BetaDataType * p_beta,
                    ck::index_t NumRows,
                    ck::index_t EmbeddingDim,
                    ck::index_t IndexLength):
                            p_emb_a_(p_emb_a),
                            p_emb_b_(p_emb_b),
                            p_emb_c_(p_emb_c),
                            p_index_a_(p_index_a),
                            p_index_b_(p_index_b),
                            p_index_c_(p_index_c),
                            p_gamma_(p_gamma),
                            p_beta_(p_beta),
                            NumRows_(NumRows),
                            EmbeddingDim_(EmbeddingDim),
                            IndexLength_(IndexLength) {}
        p_output* p_output;
        const EmbType* p_emb_a_;
        const EmbType* p_emb_b_;
        const EmbType* p_emb_c_;
        const IndexType* p_index_a_;
        const IndexType* p_index_b_;
        const IndexType* p_index_c_;
        const GammaDataType * p_gamma_;
        const BetaDataType * p_beta_;
        ck::index_t NumRows_;
        ck::index_t EmbeddingDim_;
        ck::index_t IndexLength_;
    };

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const EmbType* p_emb_a,
                        const EmbType* p_emb_b,
                        const EmbType* p_emb_c,
                        const IndexType* p_index_a,
                        const IndexType* p_index_b,
                        const IndexType* p_index_c,
                        const GammaDataType * p_gamma,
                        const BetaDataType * p_beta,
                        ck::index_t NumRows,
                        ck::index_t EmbeddingDim,
                        ck::index_t IndexLength)
    {
        std::make_unique<Argument>(
                    p_emb_a,
                    p_emb_b,
                    p_emb_c,
                    p_index_a,
                    p_index_b,
                    p_index_c,
                    p_gamma,
                    p_beta,
                    NumRows,
                    EmbeddingDim,
                    IndexLength);
    }

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        };
    };

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() 
    {
        return std::make_unique<Invoker>();
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceSparseEmbedding3ForwardLayernorm<"<< ",";
        // clang-format on

        return str.str();
    }
};


}
}
}
