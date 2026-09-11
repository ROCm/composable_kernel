// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/dispatcher/backends/tile_backend.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include <type_traits>

namespace ck_tile {
namespace dispatcher {
namespace backends {

/// Helper to register a CK Tile generated kernel
/// This should be called from generated code for each kernel
template <typename SelectedKernel>
void register_tile_kernel(Registry& registry, const std::string& kernel_name)
{
    // Extract metadata from SelectedKernel static members
    KernelKey key;

    // Signature
    key.signature.dtype_a   = static_cast<DataType>(SelectedKernel::ADataType);
    key.signature.dtype_b   = static_cast<DataType>(SelectedKernel::BDataType);
    key.signature.dtype_c   = static_cast<DataType>(SelectedKernel::CDataType);
    key.signature.dtype_acc = static_cast<DataType>(SelectedKernel::AccDataType);

    key.signature.layout_a = static_cast<LayoutTag>(SelectedKernel::ALayout);
    key.signature.layout_b = static_cast<LayoutTag>(SelectedKernel::BLayout);
    key.signature.layout_c = static_cast<LayoutTag>(SelectedKernel::CLayout);

    key.signature.transpose_a = false; // Extract from kernel if available
    key.signature.transpose_b = false;
    key.signature.grouped     = false;
    key.signature.split_k     = 1;

    key.signature.elementwise_op      = "PassThrough"; // Extract if available
    key.signature.num_d_tensors       = 0;
    key.signature.structured_sparsity = SelectedKernel::UseStructuredSparsity;

    // Algorithm
    key.algorithm.tile_shape.m = SelectedKernel::TileM;
    key.algorithm.tile_shape.n = SelectedKernel::TileN;
    key.algorithm.tile_shape.k = SelectedKernel::TileK;

    key.algorithm.wave_shape.m = SelectedKernel::WarpPerBlock_M;
    key.algorithm.wave_shape.n = SelectedKernel::WarpPerBlock_N;
    key.algorithm.wave_shape.k = SelectedKernel::WarpPerBlock_K;

    key.algorithm.warp_tile_shape.m = SelectedKernel::WarpTileM;
    key.algorithm.warp_tile_shape.n = SelectedKernel::WarpTileN;
    key.algorithm.warp_tile_shape.k = SelectedKernel::WarpTileK;

    // Extract pipeline, epilogue, scheduler from traits
    key.algorithm.pipeline  = Pipeline::CompV4;  // Extract from kernel
    key.algorithm.epilogue  = Epilogue::Default; // Extract from kernel
    key.algorithm.scheduler = Scheduler::Auto;   // Extract from kernel

    key.algorithm.block_size      = SelectedKernel::BlockSize;
    key.algorithm.double_buffer   = SelectedKernel::DoubleSmemBuffer;
    key.algorithm.persistent      = SelectedKernel::UsePersistentKernel;
    key.algorithm.preshuffle      = false; // Extract if available
    key.algorithm.transpose_c     = SelectedKernel::TransposeC;
    key.algorithm.num_wave_groups = 1; // Extract if available

    key.gfx_arch = 942; // Extract from build configuration

    // Create kernel instance
    auto kernel_instance = std::make_shared<TileKernelInstance<SelectedKernel>>(key, kernel_name);

    // Register with high priority (Tile kernels preferred)
    registry.register_kernel(kernel_instance, Registry::Priority::High);
}

/// Macro to simplify kernel registration in generated code
#define CK_TILE_REGISTER_KERNEL(SelectedKernel, KernelName, Registry) \
    ::ck_tile::dispatcher::backends::register_tile_kernel<SelectedKernel>(Registry, KernelName)

/// Helper to register multiple kernels from a list
template <typename... Kernels>
struct KernelRegistrar
{
    static void register_all(Registry& registry)
    {
        // This would be specialized for each kernel set
        // For now, empty implementation
    }
};

/// Auto-registration helper
/// Place this in generated files to automatically register kernels
template <typename SelectedKernel>
struct AutoRegister
{
    AutoRegister(const std::string& kernel_name)
    {
        auto& registry = Registry::instance();
        register_tile_kernel<SelectedKernel>(registry, kernel_name);
    }
};

/// Macro for auto-registration
#define CK_TILE_AUTO_REGISTER(SelectedKernel, KernelName)                \
    static ::ck_tile::dispatcher::backends::AutoRegister<SelectedKernel> \
        auto_register_##SelectedKernel{KernelName};

} // namespace backends
} // namespace dispatcher
} // namespace ck_tile
