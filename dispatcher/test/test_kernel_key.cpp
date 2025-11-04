// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/// Unit tests for KernelKey

#include "ck_tile/dispatcher/kernel_key.hpp"
#include <cassert>
#include <iostream>

using namespace ck_tile::dispatcher;

void test_kernel_key_construction()
{
    std::cout << "Test: KernelKey construction... ";
    
    KernelKey key;
    key.signature.dtype_a = DataType::FP16;
    key.signature.dtype_b = DataType::FP16;
    key.signature.dtype_c = DataType::FP16;
    key.signature.dtype_acc = DataType::FP32;
    key.signature.elementwise_op = "PassThrough";
    key.signature.num_d_tensors = 0;
    
    key.algorithm.tile_shape.m = 256;
    key.algorithm.tile_shape.n = 256;
    key.algorithm.tile_shape.k = 32;
    
    key.gfx_arch = 942;
    
    assert(key.signature.dtype_a == DataType::FP16);
    assert(key.algorithm.tile_shape.m == 256);
    assert(key.gfx_arch == 942);
    
    std::cout << "PASSED\n";
}

void test_kernel_key_equality()
{
    std::cout << "Test: KernelKey equality... ";
    
    KernelKey key1, key2;
    
    // Set same values
    key1.signature.dtype_a = DataType::FP16;
    key1.algorithm.tile_shape.m = 256;
    key1.gfx_arch = 942;
    
    key2.signature.dtype_a = DataType::FP16;
    key2.algorithm.tile_shape.m = 256;
    key2.gfx_arch = 942;
    
    assert(key1 == key2);
    assert(!(key1 != key2));
    
    // Change one value
    key2.algorithm.tile_shape.m = 128;
    assert(key1 != key2);
    assert(!(key1 == key2));
    
    std::cout << "PASSED\n";
}

void test_encode_identifier()
{
    std::cout << "Test: encode_identifier... ";
    
    KernelKey key;
    key.signature.split_k = 1;
    key.signature.elementwise_op = "PassThrough";
    key.signature.num_d_tensors = 0;
    key.algorithm.tile_shape.m = 256;
    key.algorithm.tile_shape.n = 256;
    key.algorithm.tile_shape.k = 32;
    key.algorithm.wave_shape.m = 2;
    key.algorithm.wave_shape.n = 2;
    key.algorithm.wave_shape.k = 1;
    key.algorithm.warp_tile_shape.m = 32;
    key.algorithm.warp_tile_shape.n = 32;
    key.algorithm.warp_tile_shape.k = 16;
    key.algorithm.persistent = true;
    key.algorithm.preshuffle = false;
    key.structured_sparsity = false;
    
    std::string id = key.encode_identifier();
    
    // Check that identifier contains expected components
    assert(id.find("256x256x32") != std::string::npos);  // tile shape
    assert(id.find("2x2x1") != std::string::npos);       // wave shape
    assert(id.find("32x32x16") != std::string::npos);    // warp tile shape
    assert(id.find("persist") != std::string::npos);     // persistent flag
    
    std::cout << "PASSED (id=" << id << ")\n";
}

void test_encode_identifier_with_fusion()
{
    std::cout << "Test: encode_identifier with fusion... ";
    
    KernelKey key;
    key.signature.split_k = 1;
    key.signature.elementwise_op = "Relu";
    key.signature.num_d_tensors = 2;
    key.algorithm.tile_shape.m = 128;
    key.algorithm.tile_shape.n = 128;
    key.algorithm.tile_shape.k = 64;
    key.algorithm.wave_shape.m = 2;
    key.algorithm.wave_shape.n = 2;
    key.algorithm.wave_shape.k = 1;
    key.algorithm.warp_tile_shape.m = 16;
    key.algorithm.warp_tile_shape.n = 16;
    key.algorithm.warp_tile_shape.k = 32;
    key.algorithm.persistent = false;
    key.structured_sparsity = false;
    
    std::string id = key.encode_identifier();
    
    // Check fusion-specific components
    assert(id.find("Relu") != std::string::npos);
    assert(id.find("_d2") != std::string::npos);
    assert(id.find("nopers") != std::string::npos);
    
    std::cout << "PASSED (id=" << id << ")\n";
}

int main()
{
    std::cout << "=== KernelKey Unit Tests ===\n\n";
    
    test_kernel_key_construction();
    test_kernel_key_equality();
    test_encode_identifier();
    test_encode_identifier_with_fusion();
    
    std::cout << "\n=== All KernelKey tests PASSED ===\n";
    return 0;
}

