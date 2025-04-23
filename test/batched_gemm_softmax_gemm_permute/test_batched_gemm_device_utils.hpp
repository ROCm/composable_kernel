// test_batched_gemm_device_utils.hpp
#pragma once

#include <hip/hip_runtime.h>
#include <string>

namespace ck {
namespace test {

struct DeviceResources {
    int computeUnits;
    size_t totalMemory;
    std::string deviceName;
    // Add other relevant properties as needed
};

inline DeviceResources GetDeviceResources() {
    DeviceResources res;
    hipDeviceProp_t props;
    
    // Fix the unused result error by storing the return value
    hipError_t status = hipGetDeviceProperties(&props, 0); // Use current device
    if (status != hipSuccess) {
        // Handle error (optional)
        res.computeUnits = 0;
        res.totalMemory = 0;
        res.deviceName = "Unknown";
        return res;
    }
    
    res.computeUnits = props.multiProcessorCount;
    res.totalMemory = props.totalGlobalMem;
    res.deviceName = props.name;
    
    return res;
}

// Device capability tiers
enum class DeviceCapabilityTier {
    LOW,      // MI308 and similar
    MEDIUM,   // Mid-range devices
    HIGH      // MI300 and high-end devices
};

inline DeviceCapabilityTier DetermineDeviceTier() {
    DeviceResources res = GetDeviceResources();
    
    // Adjust these thresholds based on your device specifics
    if (res.computeUnits < 80) { 
        return DeviceCapabilityTier::LOW;
    } else if (res.computeUnits < 120) {
        return DeviceCapabilityTier::MEDIUM;
    } else {
        return DeviceCapabilityTier::HIGH;
    }
}

} // namespace test
} // namespace ck
