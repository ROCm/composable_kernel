#include "common_header.hpp"
#include "tensor_descriptor_helper.hpp"

#include <iostream>
#include <vector>
#include <numeric>
#include <limits>

int main(int argc, char** argv) {
    using namespace ck;
    using data_t = float;
    const double tol = std::numeric_limits<data_t>::epsilon();

    auto set_values = [](const auto& conv_desc, std::vector<data_t>& data)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        int cnt = 0;
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

    };

    auto check_values = [](const auto& depth2space_desc, const std::vector<data_t>& data, const std::vector<data_t>& expected)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        const double tol = std::numeric_limits<data_t>::epsilon();

        for (int n=0; n<depth2space_desc.GetLength(I0); ++n) 
        {
            const auto offset_n = n*depth2space_desc.GetLength(I1) * depth2space_desc.GetLength(I2) * depth2space_desc.GetLength(I3);
            for (int ho=0; ho<depth2space_desc.GetLength(I1); ++ho) 
            {
                const auto offset_ho = ho* depth2space_desc.GetLength(I2) * depth2space_desc.GetLength(I3);
                for (int wo=0; wo<depth2space_desc.GetLength(I2); ++wo) 
                {
                    const auto offset_wo = wo * depth2space_desc.GetLength(I3);
                    for (int k=0; k<depth2space_desc.GetLength(I3); ++k) 
                    {
                        auto offset = depth2space_desc.CalculateOffset(make_multi_index(n, ho, wo, k));
                        const auto idx = offset_n + offset_ho + offset_wo + k;
                        assert(std::abs(*(data.begin() + offset) - expected[idx]) < tol);
                        const auto diff = std::abs(*(data.begin() + offset) - expected[idx]);
                        if (diff > tol)
                        {
                            std::cout << "[" << n << ", " << ho << ", " << wo << ", " << k << "], offset = " << offset  << ", val = " << *(data.begin() + offset) << std::endl;
                        }
                    }
                }
            }
        }
    };

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
        set_values(conv_desc, data);

        // check values in depth2space
        std::vector<data_t> expected(data.size());
        std::iota(expected.begin(), expected.end(), static_cast<data_t>(0));

        check_values(depth2space_desc, data, expected);
    }

    {
        const index_t N = 1;
        const index_t HoBs = 4;
        const index_t WoBs = 4;
        const index_t C = 1;
        constexpr index_t BlockSize = 2;

        std::vector<data_t> data(N*HoBs*WoBs*C);
        const auto depth2space_lengths = make_tuple(N, HoBs, WoBs, C);
        const auto depth2space_desc = make_naive_tensor_descriptor_packed(depth2space_lengths);
        const auto conv_desc = transform_depth2space_to_convolution_nhwc<BlockSize>(depth2space_desc);

        // set values in conv
        set_values(conv_desc, data);

        // check values in depth2space
        std::vector<data_t> expected = { 0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15};
        check_values(depth2space_desc, data, expected);
    }

    {
        const index_t N = 1;
        const index_t HoBs = 4;
        const index_t WoBs = 6;
        const index_t C = 3;
        constexpr index_t BlockSize = 2;

        std::vector<data_t> data(N*HoBs*WoBs*C);
        // std::iota(data.begin(), data.end(), 0.0f);
        const auto depth2space_lengths = make_tuple(N, HoBs, WoBs, C);
        const auto depth2space_desc = make_naive_tensor_descriptor_packed(depth2space_lengths);
        const auto conv_desc = transform_depth2space_to_convolution_nhwc<BlockSize>(depth2space_desc);

        // set values in conv
        set_values(conv_desc, data);

        // check values in depth2space
        std::vector<data_t> expected = {
            0, 1, 2, 3, 4, 5, 
            12, 13, 14, 15, 16, 17,
            24, 25, 26, 27, 28, 29,
            6, 7, 8, 9, 10, 11,
            18, 19, 20, 21, 22, 23,
            30, 31, 32, 33, 34, 35,
            36, 37, 38, 39, 40, 41,
            48, 49, 50, 51, 52, 53,
            60, 61, 62, 63, 64, 65,
            42, 43, 44, 45, 46, 47,
            54, 55, 56, 57, 58, 59,
            66, 67, 68, 69, 70, 71};

        check_values(depth2space_desc, data, expected);
    }

    return 0;
}
