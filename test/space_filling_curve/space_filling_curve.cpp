#include <vector>
#include <iostream>
#include <numeric>
#include <cassert>

#include "tensor_space_filling_curve.hpp"

using namespace ck;

void traverse_using_space_filling_curve();

int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    {
        traverse_using_space_filling_curve();
        auto err = hipDeviceSynchronize();
        (void)err;
        assert(err == hipSuccess);
    }
    return 0;
}

void traverse_using_space_filling_curve()
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};

    using TensorLengths     = Sequence<4, 10, 9>;
    using DimAccessOrder    = Sequence<2, 0, 1>;
    using ScalarsPerAccess  = Sequence<1, 2, 3>;
    using SpaceFillingCurve = SpaceFillingCurve<TensorLengths, DimAccessOrder, ScalarsPerAccess>;

    constexpr auto expected = make_tuple(make_tuple(0, 0, 0),
                                         make_tuple(0, 2, 0),
                                         make_tuple(0, 4, 0),
                                         make_tuple(0, 6, 0),
                                         make_tuple(0, 8, 0),
                                         make_tuple(1, 8, 0),
                                         make_tuple(1, 6, 0),
                                         make_tuple(1, 4, 0),
                                         make_tuple(1, 2, 0),
                                         make_tuple(1, 0, 0),
                                         make_tuple(2, 0, 0),
                                         make_tuple(2, 2, 0),
                                         make_tuple(2, 4, 0),
                                         make_tuple(2, 6, 0),
                                         make_tuple(2, 8, 0),
                                         make_tuple(3, 8, 0),
                                         make_tuple(3, 6, 0),
                                         make_tuple(3, 4, 0),
                                         make_tuple(3, 2, 0),
                                         make_tuple(3, 0, 0),
                                         make_tuple(3, 0, 3),
                                         make_tuple(3, 2, 3),
                                         make_tuple(3, 4, 3),
                                         make_tuple(3, 6, 3),
                                         make_tuple(3, 8, 3),
                                         make_tuple(2, 8, 3),
                                         make_tuple(2, 6, 3),
                                         make_tuple(2, 4, 3),
                                         make_tuple(2, 2, 3),
                                         make_tuple(2, 0, 3),
                                         make_tuple(1, 0, 3),
                                         make_tuple(1, 2, 3),
                                         make_tuple(1, 4, 3),
                                         make_tuple(1, 6, 3),
                                         make_tuple(1, 8, 3),
                                         make_tuple(0, 8, 3),
                                         make_tuple(0, 6, 3),
                                         make_tuple(0, 4, 3),
                                         make_tuple(0, 2, 3),
                                         make_tuple(0, 0, 3),
                                         make_tuple(0, 0, 6),
                                         make_tuple(0, 2, 6),
                                         make_tuple(0, 4, 6),
                                         make_tuple(0, 6, 6),
                                         make_tuple(0, 8, 6),
                                         make_tuple(1, 8, 6),
                                         make_tuple(1, 6, 6),
                                         make_tuple(1, 4, 6),
                                         make_tuple(1, 2, 6),
                                         make_tuple(1, 0, 6),
                                         make_tuple(2, 0, 6),
                                         make_tuple(2, 2, 6),
                                         make_tuple(2, 4, 6),
                                         make_tuple(2, 6, 6),
                                         make_tuple(2, 8, 6),
                                         make_tuple(3, 8, 6),
                                         make_tuple(3, 6, 6),
                                         make_tuple(3, 4, 6),
                                         make_tuple(3, 2, 6),
                                         make_tuple(3, 0, 6));

    constexpr index_t num_accesses = SpaceFillingCurve::GetNumOfAccess();

    static_assert(num_accesses == reduce_on_sequence(TensorLengths{} / ScalarsPerAccess{},
                                                     math::multiplies{},
                                                     Number<1>{}));

    static_for<1, num_accesses, 1>{}([&](auto i) {
        constexpr auto idx_curr = SpaceFillingCurve::GetIndex(i);

        static_assert(idx_curr[I0] == expected[i][I0]);
        static_assert(idx_curr[I1] == expected[i][I1]);
        static_assert(idx_curr[I2] == expected[i][I2]);

        constexpr auto backward_step = SpaceFillingCurve::GetBackwardStep(i);
        constexpr auto expected_step = expected[i - I1] - expected[i];
        static_assert(backward_step[I0] == expected_step[I0]);
        static_assert(backward_step[I1] == expected_step[I1]);
        static_assert(backward_step[I2] == expected_step[I2]);
    });

    static_for<0, num_accesses - 1, 1>{}([&](auto i) {
        constexpr auto idx_curr = SpaceFillingCurve::GetIndex(i);

        static_assert(idx_curr[I0] == expected[i][I0]);
        static_assert(idx_curr[I1] == expected[i][I1]);
        static_assert(idx_curr[I2] == expected[i][I2]);

        constexpr auto forward_step  = SpaceFillingCurve::GetForwardStep(i);
        constexpr auto expected_step = expected[i + I1] - expected[i];
        static_assert(forward_step[I0] == expected_step[I0]);
        static_assert(forward_step[I1] == expected_step[I1]);
        static_assert(forward_step[I2] == expected_step[I2]);
    });
}
