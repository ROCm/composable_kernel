#include "config.hpp"
#include "integral_constant.hpp"
#include "number.hpp"
#include "type.hpp"
#include "common_header.hpp"
#include "tensor_descriptor_helper.hpp"

#include <iostream>
#include <vector>
#include <numeric>
#include <limits>

using namespace ck;

int test_slice_transform();

int main()
{
    {
        constexpr auto Rs = Number64<0>{} + Number64<1>{};
        static_assert(is_known_at_compile_time<remove_cvref_t<decltype(Rs)>>::value, "");
    }
    {
        constexpr auto Rs = Number<0>{} + Number64<1>{};
        static_assert(is_known_at_compile_time<remove_cvref_t<decltype(Rs)>>::value, "");
    }
    {
        constexpr auto Rs = Number64<0>{} + Number<1>{};
        static_assert(is_known_at_compile_time<remove_cvref_t<decltype(Rs)>>::value, "");
    }

    return 0;
}

int test_slice_transform()
{
    using data_t = float;

    auto set_values = [](const auto& desc, std::vector<data_t>& data)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};

        int cnt = 0;
        for (int i0=0; i0<desc.GetLength(I0); ++i0)
        {
            for (int i1=0; i1<desc.GetLength(I1); ++i1)
            {
                for (int i2=0; i2<desc.GetLength(I2); ++i2)
                {
                    auto offset = desc.CalculateOffset(make_multi_index(i0, i1, i2));
                    *(data.begin() + offset) = cnt;
                    ++cnt;
                }
            }
        }

    };

    auto check_values = [](const auto& desc2d, const std::vector<data_t>& data, const std::vector<data_t>& expected)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        const double tol = std::numeric_limits<data_t>::epsilon();

        for (int i0=0; i0<desc2d.GetLength(I0); ++i0)
        {
            const auto offset_i0 = i0*desc2d.GetLength(I1);
            for (int i1=0; i1<desc2d.GetLength(I1); ++i1)
            {
                auto offset = desc2d.CalculateOffset(make_multi_index(i0, i1));
                const auto idx = offset_i0 + i1;
                assert(std::abs(*(data.begin() + offset) - expected[idx]) < tol);
                const auto diff = std::abs(*(data.begin() + offset) - expected[idx]);
                if (diff > tol)
                {
                    std::cout << "[" << i0 << ", " << i1 << "], offset = " << offset  << ", val = " << *(data.begin() + offset) << std::endl;
                }
            }
        }
    };

    const index_t dim0 = 3;
    const index_t dim1 = 4;
    const index_t dim2 = 5;

    std::vector<data_t> data_3d(dim0 * dim1 * dim2);
    const auto desc_3d = make_naive_tensor_descriptor_packed(make_tuple(dim0, dim1, dim2));


    set_values(desc_3d, data_3d);

    const auto desc_slice = transform_tensor_descriptor(
        desc_3d,
        make_tuple(make_slice_transform(dim0, 1, 2),
                   make_pass_through_transform(dim1),
                   make_pass_through_transform(dim2)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

    const auto desc_2d = transform_tensor_descriptor(
        desc_slice,
        make_tuple(make_merge_transform(make_tuple(1, dim1)),
                   make_pass_through_transform(dim2)),
        make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}));


    std::vector<data_t> expected(dim1*dim2);
    std::iota(expected.begin(), expected.end(), static_cast<data_t>(dim1*dim2));

    check_values(desc_2d, data_3d, expected);

    return 0;
}
