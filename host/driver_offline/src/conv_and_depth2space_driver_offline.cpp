#include <atomic>
#include <iostream>

int main(int argc, char** argv) {
    using namespace ck;

    using data_t = float;
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};
    constexpr auto I4 = Number<4>{};
    constexpr auto I5 = Number<5>{};

#define __case 0
#ifdef  __case == 0
    const index_t N = 1;
    const index_t HoBs = 2;
    const index_t WoBs = 2;
    const index_t C = 3;
    constexpr index_t BlokSize = 2;
#endif
    
    std::vector<std::size_t> depth2space_lengths = {N, HoBs, WoBz, C};
    Tensor<data_t> depth2space(depth2space_lengths);
    depth2space(0, 0, 0, 0) = 0.0f;
    depth2space(0, 0, 0, 1) = 1.0f;
    depth2space(0, 0, 0, 2) = 2.0f;
    depth2space(0, 0, 1, 0) = 3.0f;
    depth2space(0, 0, 1, 1) = 4.0f;
    depth2space(0, 0, 1, 2) = 5.0f;
    depth2space(0, 1, 0, 0) = 6.0f;
    depth2space(0, 1, 0, 1) = 7.0f;
    depth2space(0, 1, 0, 2) = 8.0f;
    depth2space(0, 1, 1, 0) = 9.0f;
    depth2space(0, 1, 1, 1) = 10.0f;
    depth2space(0, 1, 1, 2) = 11.0f;

    depth2space_desc = make_naive_tensor_descriptor_packed(depth2space_lengths);
    const auto conv_desc = transform_depth2space_to_convolution<BlockSize>(depth2space_lengths);

}
