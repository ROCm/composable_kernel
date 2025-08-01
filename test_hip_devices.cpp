#include <hip/hip_runtime.h>
#include <iostream>

int main() {
    int deviceCount = 0;
    hipError_t error = hipGetDeviceCount(&deviceCount);
    
    if (error != hipSuccess) {
        std::cout << "HIP Error: " << hipGetErrorString(error) << std::endl;
        return 1;
    }
    
    std::cout << "Number of HIP devices: " << deviceCount << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name 
                  << " (Compute Capability: " << prop.major << "." << prop.minor << ")" << std::endl;
    }
    
    return 0;
}
