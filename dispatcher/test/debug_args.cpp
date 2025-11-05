// Debug: Print GemmHostArgs to see exact values
#include <iostream>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

int main() {
    const int M = 128, N = 128, K = 128;
    
    std::cout << "For RCR layout (Row-major A, Column-major B, Row-major C):\n";
    std::cout << "M=" << M << ", N=" << N << ", K=" << K << "\n\n";
    
    std::cout << "A is MxK (128x128) row-major:\n";
    std::cout << "  stride_A = K = " << K << " (leading dimension = num columns)\n\n";
    
    std::cout << "B is KxN (128x128) column-major:\n";
    std::cout << "  stride_B = K = " << K << " (leading dimension = num rows)\n\n";
    
    std::cout << "C is MxN (128x128) row-major:\n";
    std::cout << "  stride_C = N = " << N << " (leading dimension = num columns)\n\n";
    
    std::cout << "tile_engine calculation:\n";
    bool is_a_row = true;  // RowMajor
    bool is_b_row = false; // ColumnMajor
    bool is_c_row = true;  // RowMajor
    
    auto stride_a = is_a_row ? K : M;  // row-major: col, col-major: row
    auto stride_b = is_b_row ? N : K;  // row-major: col, col-major: row  
    auto stride_c = is_c_row ? N : M;  // row-major: col, col-major: row
    
    std::cout << "  stride_A = " << stride_a << "\n";
    std::cout << "  stride_B = " << stride_b << "\n";
    std::cout << "  stride_C = " << stride_c << "\n";
    
    return 0;
}
