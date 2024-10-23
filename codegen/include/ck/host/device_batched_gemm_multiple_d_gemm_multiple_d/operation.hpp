// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <vector>
#include <string>
#include "ck/host/types.hpp"
#include "ck/host/operation/gemm.hpp"
#include "ck/host/device_batched_gemm_multiple_d_gemm_multiple_d/problem.hpp"

namespace ck {
namespace host {
namespace device_batched_gemm_multiple_d_gemm_multiple_d {

// defines all values needed for an instance
struct Operation_Xdl_CShuffle
{
    // returns a vector of instances, only given fusion operators: will use default problem spec
    static std::vector<std::vector<Operation_Xdl_CShuffle>>
    CreateOperations(const std::string& prologue, const std::string& epilogue);
    // returns a vector of instances, given a problem spec and fusion operators
    static std::vector<Operation_Xdl_CShuffle>
    CreateOperations(const Problem& prob, const std::string& prologue, const std::string& epilogue);
    TensorDesc A0{};
    TensorDesc B0{};
    std::vector<TensorDesc> D0s = {};
    TensorDesc B1{};
    std::vector<TensorDesc> D1s = {};
    TensorDesc E1{};
    DataType acc_type        = DataType::Float;
    DataType cshuffle_type   = DataType::Float;
    std::string a0_elem_op   = PassThrough;
    std::string b0_elem_op   = PassThrough;
    std::string cde0_elem_op = PassThrough;
    std::string b1_elem_op   = PassThrough;
    std::string cde1_elem_op = PassThrough;
    std::string prologue     = "";
    std::string epilogue     = "";
    // tuning parameters
    operation::PaddingDesc padding_desc{};
    operation::TileDescGemmGemm tile_desc{};
    operation::BlockTransferDesc a0_block_transfer{};
    operation::BlockTransferDesc b0_block_transfer{};
    operation::BlockTransferDesc cde0_block_transfer{};
    operation::BlockTransferDesc b1_block_transfer{};
    operation::CShuffleDesc cshuffle{};
    operation::CBlockTransferDesc cde1_block_transfer{};

    // functions to update fusion operators if provided
    void update_prologue(const std::string& prologue);
    void update_epilogue(const std::string& epilogue);
    /**constexpr**/ bool
    IsSupported(std::size_t MRaw_, std::size_t NRaw_, std::size_t KRaw_, std::size_t Gemm1NRaw_);
    // returns a templated instance
    Solution ToSolution() const;
};

} // namespace device_batched_gemm_multiple_d_gemm_multiple_d
} // namespace host
} // namespace ck
