// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/dispatcher/kernel_instance.hpp"
#include "ck_tile/dispatcher/kernel_key.hpp"
#include "ck_tile/dispatcher/problem.hpp"
#include <string>

namespace ck_tile {
namespace dispatcher {
namespace test {

/// Mock kernel instance for testing dispatcher functionality
/// Supports configurable behavior for testing different scenarios
class MockKernelInstance : public KernelInstance
{
    public:
    /// Constructor
    /// @param key Kernel configuration key
    /// @param name Human-readable kernel name
    /// @param supports_all Whether this kernel supports all problems (default: true)
    explicit MockKernelInstance(const KernelKey& key,
                                const std::string& name,
                                bool supports_all = true)
        : key_(key), name_(name), supports_all_(supports_all), execution_count_(0)
    {
    }

    const KernelKey& get_key() const override { return key_; }

    bool supports(const Problem& problem) const override
    {
        if(supports_all_)
        {
            return problem.is_valid();
        }
        // For testing: only support problems where M/N/K are divisible by tile sizes
        return problem.is_valid() && (problem.M % key_.algorithm.tile_shape.m == 0) &&
               (problem.N % key_.algorithm.tile_shape.n == 0) &&
               (problem.K % key_.algorithm.tile_shape.k == 0);
    }

    std::string get_name() const override { return name_; }

    float run(const void* a_ptr,
              const void* b_ptr,
              void* c_ptr,
              const void** d_ptrs,
              const Problem& problem,
              void* stream) const override
    {
        execution_count_++;
        // Simulate execution time (1ms for testing)
        return 1.0f;
    }

    bool validate(const void* a_ptr,
                  const void* b_ptr,
                  const void* c_ptr,
                  const void** d_ptrs,
                  const Problem& problem,
                  float tolerance) const override
    {
        // Mock validation always passes
        return true;
    }

    /// Get execution count (for testing)
    int get_execution_count() const { return execution_count_; }

    /// Reset execution count
    void reset_execution_count() { execution_count_ = 0; }

    /// Set whether this kernel supports all problems
    void set_supports_all(bool supports_all) { supports_all_ = supports_all; }

    private:
    KernelKey key_;
    std::string name_;
    bool supports_all_;
    mutable int execution_count_;
};

/// Helper function to create a test kernel key
inline KernelKey make_test_key(std::uint16_t tile_m        = 256,
                               std::uint16_t tile_n        = 256,
                               std::uint16_t tile_k        = 32,
                               const std::string& gfx_arch = "gfx942")
{
    KernelKey key;
    key.signature.dtype_a             = DataType::FP16;
    key.signature.dtype_b             = DataType::FP16;
    key.signature.dtype_c             = DataType::FP16;
    key.signature.dtype_acc           = DataType::FP32;
    key.signature.layout_a            = LayoutTag::RowMajor;
    key.signature.layout_b            = LayoutTag::ColMajor;
    key.signature.layout_c            = LayoutTag::RowMajor;
    key.signature.transpose_a         = false;
    key.signature.transpose_b         = false;
    key.signature.grouped             = false;
    key.signature.split_k             = 1;
    key.signature.elementwise_op      = "PassThrough";
    key.signature.num_d_tensors       = 0;
    key.signature.structured_sparsity = false;

    key.algorithm.tile_shape.m      = tile_m;
    key.algorithm.tile_shape.n      = tile_n;
    key.algorithm.tile_shape.k      = tile_k;
    key.algorithm.wave_shape.m      = 2;
    key.algorithm.wave_shape.n      = 2;
    key.algorithm.wave_shape.k      = 1;
    key.algorithm.warp_tile_shape.m = 32;
    key.algorithm.warp_tile_shape.n = 32;
    key.algorithm.warp_tile_shape.k = 16;
    key.algorithm.pipeline          = Pipeline::CompV4;
    key.algorithm.scheduler         = Scheduler::Intrawave;
    key.algorithm.epilogue          = Epilogue::CShuffle;
    key.algorithm.block_size        = 256;
    key.algorithm.double_buffer     = true;
    key.algorithm.persistent        = false;
    key.algorithm.preshuffle        = false;
    key.algorithm.transpose_c       = false;
    key.algorithm.num_wave_groups   = 1;

    key.gfx_arch = gfx_arch;

    return key;
}

} // namespace test
} // namespace dispatcher
} // namespace ck_tile
