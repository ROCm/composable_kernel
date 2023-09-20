// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdlib>
#include <vector>
#include <memory>
#include <sstream>
#include <iterator>
#include <numeric>
#include "ck/host/common.hpp"

namespace ck {
namespace host {
namespace device_batched_gemm_softmax_gemm {

struct Problem
{
    std::size_t M                    = 0;
    std::size_t N                    = 0;
    std::size_t K                    = 0;
    std::size_t O                    = 0;
    bool TransA                      = false;
    bool TransB                      = false;
    bool TransB1                     = false;
    bool TransC                      = false;
    DataType ADataType               = DataType::Half;
    DataType BDataType               = DataType::Half;
    DataType B1DataType              = DataType::Half;
    DataType CDataType               = DataType::Half;
    std::string AElementOp           = "ck::tensor_operation::element_wise::PassThrough";
    std::string BElementOp           = "ck::tensor_operation::element_wise::PassThrough";
    std::string B1ElementOp          = "ck::tensor_operation::element_wise::PassThrough";
    std::string CElementOp           = "ck::tensor_operation::element_wise::PassThrough";
    std::string AccElementOp         = "ck::tensor_operation::element_wise::Scale";

    std::string GetIncludeHeader() const;

    std::vector<Solution> GetSolutions(const std::string& arch) const;

    private:
    std::vector<std::string> GetInstances(const std::string& arch) const;

    Solution MakeSolution(std::size_t idx, const std::string& arch) const;

    static const std::size_t DeviceBatchedGemmSoftmaxGemm_Xdl_CShuffle_idx = 0;
    static const std::size_t ALayout_idx = 1;
    static const std::size_t B0Layout_idx = 2;
    static const std::size_t B1Layout_idx = 3;
    static const std::size_t CLayout_idx = 4;
    static const std::size_t ADataType_idx = 5;
    static const std::size_t B0DataType_idx = 6;
    static const std::size_t B1DataType_idx = 7;
    static const std::size_t CDataType_idx = 8;
    static const std::size_t AccDataType_idx = 9;
    static const std::size_t CShuffleDataType_idx = 10;
    static const std::size_t AElementwiseOperation_idx = 11;
    static const std::size_t B0ElementwiseOperation_idx = 12;
    static const std::size_t Acc0ElementwiseOperation_idx = 13;
    static const std::size_t B1ElementwiseOperation_idx = 14;
    static const std::size_t CElementwiseOperation_idx = 15;
    static const std::size_t GEMMSpecialization_idx = 16;
    static const std::size_t NumGemmKPrefetchStage_idx = 17;
    static const std::size_t BlockSize_idx = 18;
    static const std::size_t Gemm01MPerBlock_idx = 19;
    static const std::size_t Gemm0NPerBlock_idx = 20;
    static const std::size_t Gemm0KPerBlock_idx = 21;
    static const std::size_t Gemm1NPerBlock_idx = 22;
    static const std::size_t Gemm1KPerBlock_idx = 23;
    static const std::size_t AK1_idx = 24;
    static const std::size_t BK1_idx = 25;
    static const std::size_t B1K1_idx = 26;
    static const std::size_t MPerXDL_idx = 27;
    static const std::size_t NPerXDL_idx = 28;
    static const std::size_t Gemm0MXdlPerWave_idx = 29;
    static const std::size_t Gemm0NXdlPerWave_idx = 30;
    static const std::size_t Gemm1NXdlPerWave_idx = 31;
    static const std::size_t ABlockTransferThreadClusterLengths_K0_M_K1_idx = 32;
    static const std::size_t ABlockTransferThreadClusterArrangeOrder_idx = 33;
    static const std::size_t ABlockTransferSrcAccessOrder_idx = 34;
    static const std::size_t ABlockTransferSrcVectorDim_idx = 35;
    static const std::size_t ABlockTransferSrcScalarPerVector_idx = 36;
    static const std::size_t ABlockTransferDstScalarPerVector_K1_idx = 37;
    static const std::size_t ABlockLdsAddExtraM_idx = 38;
    static const std::size_t B0BlockTransferThreadClusterLengths_K0_N_K1_idx = 39;
    static const std::size_t B0BlockTransferThreadClusterArrangeOrder_idx = 40;
    static const std::size_t B0BlockTransferSrcAccessOrder_idx = 41;
    static const std::size_t B0BlockTransferSrcVectorDim_idx = 42;
    static const std::size_t B0BlockTransferSrcScalarPerVector_idx = 43;
    static const std::size_t B0BlockTransferDstScalarPerVector_K1_idx = 44;
    static const std::size_t B0BlockLdsAddExtraN_idx = 45;
    static const std::size_t B1BlockTransferThreadClusterLengths_K0_N_K1_idx = 46;
    static const std::size_t B1BlockTransferThreadClusterArrangeOrder_idx = 47;
    static const std::size_t B1BlockTransferSrcAccessOrder_idx = 48;
    static const std::size_t B1BlockTransferSrcVectorDim_idx = 49;
    static const std::size_t B1BlockTransferSrcScalarPerVector_idx = 50;
    static const std::size_t B1BlockTransferDstScalarPerVector_K1_idx = 51;
    static const std::size_t B1BlockLdsAddExtraN_idx = 52;
    static const std::size_t CShuffleMXdlPerWavePerShuffle_idx = 53;
    static const std::size_t CShuffleNXdlPerWavePerShuffle_idx = 54;
    static const std::size_t CBlockTransferClusterLengths_MBlock_MWaveMPerXdl_NBlock_NWaveNPerXdl_idx = 55;
    static const std::size_t CBlockTransferScalarPerVector_NWaveNPerXdl_idx = 56;
    static const std::size_t MaskOutUpperTriangle_idx = 57;
};

} // namespace device_batched_gemm_softmax_gemm
} // namespace host
} // namespace ck
