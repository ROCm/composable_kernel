// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iomanip>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "common/utils.hpp"

// Data types and Layouts are defined by the generated kernel headers
// No hardcoded type definitions here to avoid conflicts
struct GemmProblem
{
    int split_k_;
    int m_, n_, k_;
    int stride_a_, stride_b_, stride_c_;

    std::string dtype_a_, dtype_b_, dtype_acc_, dtype_c_;
    std::string layout_a_, layout_b_, layout_c_;

    bool structured_sparsity_;

    friend std::ostream& operator<<([[clang::lifetimebound]] std::ostream& os,
                                    const GemmProblem& problem)
    {
        os << "{\n"
           << "   \"split_k\":" << problem.split_k_ << ",\n"
           << "   \"m\":" << problem.m_ << ",\n"
           << "   \"n\":" << problem.n_ << ",\n"
           << "   \"k\":" << problem.k_ << ",\n"
           << "   \"stride_a\":" << problem.stride_a_ << ",\n"
           << "   \"stride_b\":" << problem.stride_b_ << ",\n"
           << "   \"stride_c\":" << problem.stride_c_ << ",\n"
           << "   \"dtype_a\":\"" << problem.dtype_a_ << "\",\n"
           << "   \"dtype_b\":\"" << problem.dtype_b_ << "\",\n"
           << "   \"dtype_acc\":\"" << problem.dtype_acc_ << "\",\n"
           << "   \"dtype_c\":\"" << problem.dtype_c_ << "\",\n"
           << "   \"layout_a\":\"" << problem.layout_a_ << "\",\n"
           << "   \"layout_b\":\"" << problem.layout_b_ << "\",\n"
           << "   \"layout_c\":\"" << problem.layout_c_ << "\",\n"
           << "   \"structured_sparsity\":" << (problem.structured_sparsity_ ? "true" : "false")
           << "\n"
           << "}";
        return os;
    }
};

// Detect Problem::DsDataType, default to void when absent
template <class T, class = void>
struct get_DsDataType
{
    using type = void;
};

template <class T>
struct get_DsDataType<T, std::void_t<typename T::DsDataType>>
{
    using type = typename T::DsDataType;
};

// Detect Problem::D0DataType, default to void when absent
template <class T, class = void>
struct get_D0DataType
{
    using type = void;
};

template <class T>
struct get_D0DataType<T, std::void_t<typename T::D0DataType>>
{
    using type = typename T::D0DataType;
};

/// @brief Generic compare: all types provided explicitly as template parameters
template <typename AType, typename BType, typename AccType, typename OutType, typename DType = void>
bool compare(std::string instanceName,
             ck_tile::index_t K,
             ck_tile::index_t kbatch,
             ck_tile::HostTensor<OutType>& out_dev_result,
             ck_tile::HostTensor<OutType>& out_host_result)
{
    const float max_accumulated_value =
        *std::max_element(out_host_result.mData.begin(), out_host_result.mData.end());
    auto rtol_atol = [&] {
        if constexpr(std::is_void_v<DType>)
        {
            return calculate_rtol_atol<AType, BType, AccType, OutType>(
                K, kbatch, max_accumulated_value);
        }
        else
        {
            return calculate_rtol_atol<AType, BType, DType, AccType, OutType>(
                K, kbatch, max_accumulated_value);
        }
    }();
    bool pass = ck_tile::check_err(out_dev_result,
                                   out_host_result,
                                   "Error: Incorrect results!",
                                   rtol_atol.at(ck_tile::number<0>{}),
                                   rtol_atol.at(ck_tile::number<1>{}));

    std::cout << "For " << instanceName << " Relative error threshold is "
              << rtol_atol.at(ck_tile::number<0>{}) << " Absolute error threshold is "
              << rtol_atol.at(ck_tile::number<1>{}) << std::endl;
    std::cout << "The verification result is:" << (pass ? "correct" : "fail") << std::endl;

    return pass;
}

/// @brief Backward-compatible compare: deduces types from global aliases and Problem trait
template <typename Problem, typename OutType>
bool compare(std::string instanceName,
             ck_tile::index_t K,
             ck_tile::index_t kbatch,
             ck_tile::HostTensor<OutType>& out_dev_result,
             ck_tile::HostTensor<OutType>& out_host_result)
{
    using DDataType = typename get_D0DataType<Problem>::type;
    return compare<ADataType, BDataType, AccDataType, OutType, DDataType>(
        instanceName, K, kbatch, out_dev_result, out_host_result);
}
