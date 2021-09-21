#include "common_header.hpp"
#include "tensor_descriptor_helper.hpp"
#include <atomic>
#include <iostream>
#include <vector>
#include <numeric>

int main(int argc, char** argv) {
    using namespace ck;
    using data_t = float;

    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    {
        const index_t N = 1;
        const index_t HoBs = 2;
        const index_t WoBs = 2;
        const index_t C = 3;
        constexpr index_t BlockSize = 2;

        std::vector<data_t> data(N*HoBs*WoBs*C);
        // std::iota(data.begin(), data.end(), 0.0f);
        const auto depth2space_lengths = make_tuple(N, HoBs, WoBs, C);
        const auto depth2space_desc = make_naive_tensor_descriptor_packed(depth2space_lengths);
        const auto conv_desc = transform_depth2space_to_convolution_nhwc<BlockSize>(depth2space_desc);

        // set values in conv
        int cnt = 1;
        for (int n=0; n<conv_desc.GetLength(I0); ++n) 
        {
            for (int ho=0; ho<conv_desc.GetLength(I1); ++ho) 
            {
                for (int wo=0; wo<conv_desc.GetLength(I2); ++wo) 
                {
                    for (int c=0; c<conv_desc.GetLength(I3); ++c) 
                    {
                        auto offset = conv_desc.CalculateOffset(make_multi_index(n, ho, wo, c));
                        *(data.begin() + offset) = cnt;
                        ++cnt;

                    }

                }

            }
        }

        // check value in depth2space
        for (int n=0; n<depth2space_desc.GetLength(I0); ++n) 
        {
            for (int ho=0; ho<depth2space_desc.GetLength(I1); ++ho) 
            {
                for (int wo=0; wo<depth2space_desc.GetLength(I2); ++wo) 
                {
                    for (int k=0; k<depth2space_desc.GetLength(I3); ++k) 
                    {
                        auto offset = depth2space_desc.CalculateOffset(make_multi_index(n, ho, wo, k));
                        std::cout << "[" << n << ", " << ho << ", " << wo << ", " << k << "], offset = " << offset  << ", val = " << *(data.begin() + offset) << std::endl;

                    }

                }

            }
        }
    }

    {
        const index_t N = 1;
        const index_t HoBs = 4;
        const index_t WoBs = 4;
        const index_t C = 1;
        constexpr index_t BlockSize = 2;

        std::vector<data_t> data(N*HoBs*WoBs*C);
        std::iota(data.begin(), data.end(), 0.0f);
        const auto depth2space_lengths = make_tuple(N, HoBs, WoBs, C);
        const auto depth2space_desc = make_naive_tensor_descriptor_packed(depth2space_lengths);
        const auto conv_desc = transform_depth2space_to_convolution_nhwc<BlockSize>(depth2space_desc);

        int cnt = 1;
        for (int n=0; n<conv_desc.GetLength(I0); ++n) 
        {
            for (int ho=0; ho<conv_desc.GetLength(I1); ++ho) 
            {
                for (int wo=0; wo<conv_desc.GetLength(I2); ++wo) 
                {
                    for (int k=0; k<conv_desc.GetLength(I3); ++k) 
                    {
                        auto offset = conv_desc.CalculateOffset(make_multi_index(n, ho, wo, k));
                        *(data.begin() + offset) = cnt;
                        ++cnt;

                    }

                }

            }
        }

        // check value in depth2space
        for (int n=0; n<depth2space_desc.GetLength(I0); ++n) 
        {
            for (int ho=0; ho<depth2space_desc.GetLength(I1); ++ho) 
            {
                for (int wo=0; wo<depth2space_desc.GetLength(I2); ++wo) 
                {
                    for (int k=0; k<depth2space_desc.GetLength(I3); ++k) 
                    {
                        auto offset = depth2space_desc.CalculateOffset(make_multi_index(n, ho, wo, k));
                        std::cout << "[" << n << ", " << ho << ", " << wo << ", " << k << "], offset = " << offset  << ", val = " << *(data.begin() + offset) << std::endl;

                    }

                }

            }
        }
    }
    // conv_desc.Print();

}
