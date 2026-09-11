// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Common utilities shared by all CK Tile grouped convolution profiler headers
// (both builder-based and dispatcher-based, all directions).

#pragma once

#include <algorithm>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <vector>

#include "../../experimental/builder/test/utils/conv_algorithm_type_utils.hpp"
#include "ck_tile/builder/testing/conv/reference.hpp"
#include "ck_tile/builder/conv_builder.hpp"

namespace ck_tile::builder::profiling {

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;

/// Deduce the C++ data type from a SIGNATURE's data_type field.
template <auto SIGNATURE>
using DeduceDataType =
    std::conditional_t<SIGNATURE.data_type == ckb::DataType::FP32,
                       float,
                       std::conditional_t<SIGNATURE.data_type == ckb::DataType::FP16,
                                          ck_tile::half_t,
                                          ck_tile::bfloat16_t>>;

/// Compute reference outputs using the builder reference implementation.
/// Returns a unique_ptr-like handle to reference outputs (from ckt::alloc_outputs).
template <auto SIGNATURE>
auto compute_reference(const ckt::Args<SIGNATURE>& args, const ckt::Inputs<SIGNATURE>& inputs)
{
    auto reference = ckt::alloc_outputs(args);
    using ReferenceInstance =
        typename ckb::ConvBuilder<SIGNATURE, ckt::ConvAlgorithm_Reference{}>::Instance;
    auto ref_conv   = ReferenceInstance{};
    auto ref_result = ckt::run(ref_conv, args, inputs, reference.get());
    (void)ref_result;
    return reference;
}

/// Tag type to select which convolution buffer to validate.
enum class ConvBuffer
{
    Output, // Forward: compare output
    Input,  // Backward data: compare input (dX)
    Weight, // Backward weight: compare weight (dW)
};

/// Unified CPU validation that copies GPU data to host and runs check_err.
/// Parameterized by ConvBuffer to select which buffer to compare.
template <auto SIGNATURE, ConvBuffer BUFFER>
void run_cpu_validation(const ckt::Args<SIGNATURE>& args,
                        const ckt::Outputs<SIGNATURE>& outputs,
                        const ckt::Outputs<SIGNATURE>& reference)
{
    using DataType        = DeduceDataType<SIGNATURE>;
    const auto conv_param = args.to_ck_tile_conv_param();

    std::size_t bytes_num;
    const void* ref_ptr;
    const void* out_ptr;
    const char* err_msg;

    if constexpr(BUFFER == ConvBuffer::Output)
    {
        bytes_num = conv_param.template GetOutputByte<DataType>();
        ref_ptr   = reference.output;
        out_ptr   = outputs.output;
        err_msg   = "Error: Incorrect results!";
    }
    else if constexpr(BUFFER == ConvBuffer::Input)
    {
        bytes_num = conv_param.template GetInputByte<DataType>();
        ref_ptr   = reference.input;
        out_ptr   = outputs.input;
        err_msg   = "\tError: Incorrect results!";
    }
    else // ConvBuffer::Weight
    {
        bytes_num = conv_param.template GetWeightByte<DataType>();
        ref_ptr   = reference.weight;
        out_ptr   = outputs.weight;
        err_msg   = "\tError: Incorrect results!";
    }

    std::vector<DataType> out(bytes_num / sizeof(DataType));
    std::vector<DataType> ref(bytes_num / sizeof(DataType));
    HIP_CHECK_ERROR(hipMemcpy(ref.data(), ref_ptr, bytes_num, hipMemcpyDeviceToHost));
    HIP_CHECK_ERROR(hipMemcpy(out.data(), out_ptr, bytes_num, hipMemcpyDeviceToHost));
    ck_tile::check_err(out, ref, err_msg);
}

/// Run validation report and print errors if any. Returns true if valid.
/// Calls run_cpu_validation on failure for detailed error output.
template <auto SIGNATURE, ConvBuffer BUFFER>
bool validate_and_report(const ckt::Args<SIGNATURE>& args,
                         const ckt::Outputs<SIGNATURE>& outputs,
                         const ckt::Outputs<SIGNATURE>& reference,
                         double rtol,
                         double atol)
{
    ckt::ValidationReport report;
    ckt::Outputs<SIGNATURE>::reflect(
        args, [&](std::string_view name, const auto& desc, void* ckt::Outputs<SIGNATURE>::*ptr) {
            report.check(name, desc, outputs.*ptr, reference.*ptr, rtol, atol);
        });

    if(report.get_errors().empty())
        return true;

    for(const auto& error : report.get_errors())
    {
        std::cout << "\tNumber of incorrect values: " << error.wrong_elements
                  << " Is all zero:" << error.is_all_zero() << " max err: " << error.max_error
                  << std::endl;
        run_cpu_validation<SIGNATURE, BUFFER>(args, outputs, reference);
    }
    return false;
}

} // namespace ck_tile::builder::profiling
