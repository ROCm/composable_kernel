// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * Example: Automatic JSON Export on Registration
 * 
 * Demonstrates how to enable automatic JSON export so the registry
 * automatically exports kernel metadata whenever kernels are registered.
 * 
 * Two modes:
 * 1. Export on program exit (default) - Exports once when program ends
 * 2. Export on every registration - Exports after each kernel registration
 * 
 * Usage:
 *   ./auto_export_example [mode]
 *   
 *   mode: "exit" (default) or "every"
 */

#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/json_export.hpp"
#include <iostream>
#include <string>

using namespace ck_tile::dispatcher;

int main(int argc, char* argv[]) {
    std::cout << "=== Automatic JSON Export Example ===\n\n";
    
    // Parse mode
    std::string mode = "exit";
    if (argc > 1) {
        mode = argv[1];
    }
    
    bool export_on_every = (mode == "every");
    
    // Get registry
    auto& registry = Registry::instance();
    
    // Enable auto-export
    std::string output_file = "auto_export_kernels.json";
    std::cout << "Enabling auto-export to: " << output_file << "\n";
    std::cout << "Mode: " << (export_on_every ? "Export on every registration" : "Export on program exit") << "\n\n";
    
    registry.enable_auto_export(output_file, true, export_on_every);
    
    // Verify it's enabled
    if (registry.is_auto_export_enabled()) {
        std::cout << "✓ Auto-export is enabled\n\n";
    }
    
    // Simulate kernel registration
    std::cout << "Current kernel count: " << registry.size() << "\n";
    
    if (registry.size() == 0) {
        std::cout << "\n[INFO] No kernels registered in this example.\n";
        std::cout << "In a real application, kernels would be registered via:\n";
        std::cout << "  registry.register_kernel(kernel_instance, Priority::Normal);\n\n";
        
        std::cout << "When kernels are registered:\n";
        if (export_on_every) {
            std::cout << "  - JSON file is updated after EACH registration\n";
            std::cout << "  - Useful for debugging and development\n";
            std::cout << "  - Higher I/O overhead\n";
        } else {
            std::cout << "  - JSON file is written ONCE on program exit\n";
            std::cout << "  - Efficient for production use\n";
            std::cout << "  - Lower I/O overhead\n";
        }
    } else {
        std::cout << "\n✓ Registry has " << registry.size() << " kernels\n";
        
        if (export_on_every) {
            std::cout << "\nWith 'every' mode:\n";
            std::cout << "  - JSON was exported after each registration\n";
            std::cout << "  - Check " << output_file << " - it should exist now\n";
        } else {
            std::cout << "\nWith 'exit' mode:\n";
            std::cout << "  - JSON will be exported when this program exits\n";
            std::cout << "  - File will appear when main() returns\n";
        }
    }
    
    // Demonstrate disabling
    std::cout << "\n--- Demonstrating disable ---\n";
    registry.disable_auto_export();
    
    if (!registry.is_auto_export_enabled()) {
        std::cout << "✓ Auto-export is now disabled\n";
    }
    
    // Re-enable for exit
    std::cout << "\n--- Re-enabling for exit ---\n";
    registry.enable_auto_export(output_file, true, false);
    std::cout << "✓ Auto-export re-enabled for program exit\n";
    
    std::cout << "\n=== Example Complete ===\n";
    std::cout << "Watch for: " << output_file << " to be created on exit\n";
    
    // When this function returns, the Registry singleton will be destroyed
    // and auto-export will trigger (since we re-enabled it)
    return 0;
}

