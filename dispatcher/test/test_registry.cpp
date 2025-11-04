// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/// Unit tests for Registry

#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/kernel_key.hpp"
#include <cassert>
#include <iostream>

using namespace ck_tile::dispatcher;

// Mock kernel instance for testing
class MockKernelInstance : public KernelInstance {
public:
    MockKernelInstance(const KernelKey& key, const std::string& name)
        : key_(key), name_(name) {}
    
    const KernelKey& get_key() const override { return key_; }
    bool supports(const Problem&) const override { return true; }
    std::string get_name() const override { return name_; }
    
    float run(const void*, const void*, void*, const void**, const Problem&, void*) const override {
        return 0.0f;
    }
    
    bool validate(const void*, const void*, const void*, const void**, const Problem&, float) const override {
        return true;
    }

private:
    KernelKey key_;
    std::string name_;
};

KernelKey make_test_key(int tile_m)
{
    KernelKey key;
    key.signature.dtype_a = DataType::FP16;
    key.signature.elementwise_op = "PassThrough";
    key.signature.num_d_tensors = 0;
    key.algorithm.tile_shape.m = tile_m;
    key.algorithm.tile_shape.n = 256;
    key.algorithm.tile_shape.k = 32;
    key.algorithm.wave_shape.m = 2;
    key.algorithm.wave_shape.n = 2;
    key.algorithm.wave_shape.k = 1;
    key.algorithm.warp_tile_shape.m = 32;
    key.algorithm.warp_tile_shape.n = 32;
    key.algorithm.warp_tile_shape.k = 16;
    key.algorithm.persistent = false;
    key.gfx_arch = 942;
    return key;
}

void test_registry_registration()
{
    std::cout << "Test: Registry registration... ";
    
    Registry registry;
    
    auto key = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "test_kernel");
    
    bool registered = registry.register_kernel(kernel);
    assert(registered);
    assert(registry.size() == 1);
    
    std::cout << "PASSED\n";
}

void test_registry_lookup()
{
    std::cout << "Test: Registry lookup... ";
    
    Registry registry;
    
    auto key = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "test_kernel");
    registry.register_kernel(kernel);
    
    // Lookup by key
    auto found = registry.lookup(key);
    assert(found != nullptr);
    assert(found->get_name() == "test_kernel");
    
    // Lookup by identifier
    std::string id = key.encode_identifier();
    auto found2 = registry.lookup(id);
    assert(found2 != nullptr);
    assert(found2->get_name() == "test_kernel");
    
    // Lookup non-existent
    auto key2 = make_test_key(128);
    auto not_found = registry.lookup(key2);
    assert(not_found == nullptr);
    
    std::cout << "PASSED\n";
}

void test_registry_priority()
{
    std::cout << "Test: Registry priority... ";
    
    Registry registry;
    
    auto key = make_test_key(256);
    auto kernel1 = std::make_shared<MockKernelInstance>(key, "kernel_low");
    auto kernel2 = std::make_shared<MockKernelInstance>(key, "kernel_high");
    
    // Register with low priority
    registry.register_kernel(kernel1, Registry::Priority::Low);
    
    // Try to register with normal priority (should replace)
    bool replaced = registry.register_kernel(kernel2, Registry::Priority::Normal);
    assert(replaced);
    
    auto found = registry.lookup(key);
    assert(found->get_name() == "kernel_high");
    
    // Try to register with low priority again (should fail)
    auto kernel3 = std::make_shared<MockKernelInstance>(key, "kernel_low2");
    bool not_replaced = registry.register_kernel(kernel3, Registry::Priority::Low);
    assert(!not_replaced);
    
    found = registry.lookup(key);
    assert(found->get_name() == "kernel_high");
    
    std::cout << "PASSED\n";
}

void test_registry_get_all()
{
    std::cout << "Test: Registry get_all... ";
    
    Registry registry;
    
    auto key1 = make_test_key(256);
    auto key2 = make_test_key(128);
    auto kernel1 = std::make_shared<MockKernelInstance>(key1, "kernel1");
    auto kernel2 = std::make_shared<MockKernelInstance>(key2, "kernel2");
    
    registry.register_kernel(kernel1);
    registry.register_kernel(kernel2);
    
    auto all = registry.get_all();
    assert(all.size() == 2);
    
    std::cout << "PASSED\n";
}

void test_registry_filter()
{
    std::cout << "Test: Registry filter... ";
    
    Registry registry;
    
    // Create kernels with different tile sizes
    for (int tile_m : {128, 256, 512}) {
        auto key = make_test_key(tile_m);
        auto kernel = std::make_shared<MockKernelInstance>(
            key, "kernel_" + std::to_string(tile_m));
        registry.register_kernel(kernel);
    }
    
    // Filter for large tiles (>= 256)
    auto large_tiles = registry.filter([](const KernelInstance& k) {
        return k.get_key().algorithm.tile_shape.m >= 256;
    });
    
    assert(large_tiles.size() == 2);
    
    std::cout << "PASSED\n";
}

void test_registry_clear()
{
    std::cout << "Test: Registry clear... ";
    
    Registry registry;
    
    auto key = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "test_kernel");
    registry.register_kernel(kernel);
    
    assert(registry.size() == 1);
    
    registry.clear();
    assert(registry.size() == 0);
    
    std::cout << "PASSED\n";
}

int main()
{
    std::cout << "=== Registry Unit Tests ===\n\n";
    
    test_registry_registration();
    test_registry_lookup();
    test_registry_priority();
    test_registry_get_all();
    test_registry_filter();
    test_registry_clear();
    
    std::cout << "\n=== All Registry tests PASSED ===\n";
    return 0;
}

