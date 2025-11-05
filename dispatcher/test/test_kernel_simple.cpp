#include <iostream>
#include <hip/hip_runtime.h>
#include <vector>

// Kernel header will be auto-included via -include flag in CMakeLists.txt
// #include "tile_engine_kernel_128x128x64.hpp"

#define HIP_CHECK(call) { hipError_t err = call; if(err != hipSuccess) { std::cerr << "Error\n"; exit(1); } }

int main() {
    const int M = 4, N = 4, K = 4;  // Tiny for manual verification
    
    // Host data - simple values
    std::vector<ADataType> a_host(M*K), b_host(K*N), c_result(M*N);
    
    // A = all 1s, B = all 1s, C should be K (4) for each element
    for(int i = 0; i < M*K; i++) a_host[i] = ADataType(1.0f);
    for(int i = 0; i < K*N; i++) b_host[i] = BDataType(1.0f);
    
    // GPU
    ADataType *a, *b;
    CDataType *c;
    HIP_CHECK(hipMalloc(&a, M*K*sizeof(ADataType)));
    HIP_CHECK(hipMalloc(&b, K*N*sizeof(BDataType)));
    HIP_CHECK(hipMalloc(&c, M*N*sizeof(CDataType)));
    
    HIP_CHECK(hipMemcpy(a, a_host.data(), M*K*sizeof(ADataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(b, b_host.data(), K*N*sizeof(BDataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(c, 0, M*N*sizeof(CDataType)));
    
    // Execute
    ck_tile::GemmHostArgs args;
    args.a_ptr = a;
    args.b_ptr = b;
    args.c_ptr = c;
    args.M = M;
    args.N = N;
    args.K = K;
    args.stride_A = K;
    args.stride_B = N;
    args.stride_C = N;
    args.k_batch = 1;
    
    ck_tile::stream_config stream;
    stream.time_kernel_ = true;
    stream.cold_niters_ = 1;
    stream.nrepeat_ = 1;
    stream.is_gpu_timer_ = true;
    
    std::cout << "Input: A=all 1s, B=all 1s\n";
    std::cout << "Expected: C=all " << K << "s (since each element is sum of " << K << " 1*1)\n\n";
    
    float time = SelectedKernel::launch(args, stream);
    std::cout << "Executed in " << time << " ms\n\n";
    
    // Copy result
    HIP_CHECK(hipMemcpy(c_result.data(), c, M*N*sizeof(CDataType), hipMemcpyDeviceToHost));
    
    // Check
    std::cout << "GPU Result (first 16 elements):\n";
    for(int i = 0; i < std::min(16, M*N); i++) {
        std::cout << "  C[" << i << "] = " << float(c_result[i]) << " (expected " << K << ")\n";
    }
    
    // Validate
    int correct = 0;
    for(int i = 0; i < M*N; i++) {
        if(std::abs(float(c_result[i]) - float(K)) < 0.1f) correct++;
    }
    
    std::cout << "\n" << correct << "/" << M*N << " elements correct\n";
    
    if(correct == M*N) {
        std::cout << "[OK] Kernel computes correctly!\n";
    } else {
        std::cout << "[FAIL] Kernel output incorrect!\n";
    }
    
    HIP_CHECK(hipFree(a)); HIP_CHECK(hipFree(b)); HIP_CHECK(hipFree(c));
    return (correct == M*N) ? 0 : 1;
}
