// Simple HIP kernel with explicit LDS reads to test rocgdb

#include <hip/hip_runtime.h>
#include <iostream>

// Very simple kernel that reads from LDS
__global__ void simple_lds_read_kernel(float* output)
{
    // Declare LDS buffer
    __shared__ float lds[256];

    int tid = threadIdx.x;

    // Write to LDS
    lds[tid] = tid * 2.0f;

    __syncthreads();

    // Read from LDS (different pattern to create some bank conflicts)
    int read_idx = (tid + 32) % 256;
    float value = lds[read_idx];

    __syncthreads();

    // Write to global memory
    output[tid] = value;
}

int main()
{
    const int N = 256;

    float* d_output;
    hipMalloc(&d_output, N * sizeof(float));

    std::cout << "Launching simple LDS read kernel..." << std::endl;

    simple_lds_read_kernel<<<1, 256>>>(d_output);

    hipDeviceSynchronize();

    // Copy back and verify
    float h_output[N];
    hipMemcpy(h_output, d_output, N * sizeof(float), hipMemcpyDeviceToHost);

    std::cout << "First 10 values: ";
    for(int i = 0; i < 10; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    hipFree(d_output);

    std::cout << "Done!" << std::endl;

    return 0;
}
