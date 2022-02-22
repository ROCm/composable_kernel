#include <vector>
#include <iostream>
#include <numeric>
#include <cassert>

#include "tensor_space_filling_curve.hpp"

using namespace ck;

__global__ void traverse_using_space_filling_curve();

int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    {
        traverse_using_space_filling_curve<<<1, 1>>>();
        auto err = hipDeviceSynchronize();
        (void)err;
        assert(err == hipSuccess);
    }
    return 0;
}

__global__ void traverse_using_space_filling_curve()
{
    constexpr auto I0 = Number<0>{};

    using SliceLengths      = Sequence<4, 3, 6>;
    using DimAccessOrder    = Sequence<2, 1, 0>;
    using ScalarsPerAccess  = Sequence<2, 1, 2>;
    using SpaceFillingCurve = SpaceFillingCurve<SliceLengths, DimAccessOrder, ScalarsPerAccess>;

    constexpr index_t num_accesses = SpaceFillingCurve::GetNumOfAccess();
    static_assert(num_accesses == reduce_on_sequence(SliceLengths{} / ScalarsPerAccess{},
                                                     math::multiplies{},
                                                     Number<1>{}));

    auto idx0 = SpaceFillingCurve::GetIndex(I0);

    static_for<1, num_accesses, 1>{}([&](auto i) {
        constexpr auto idx1 = SpaceFillingCurve::GetIndex(i);
        const auto idx_diff = idx1 - idx0;
        index_t diff        = 0;
        static_for<0, SliceLengths::Size(), 1>{}([&](auto idim) {
            diff += abs(idx_diff[idim]) / ScalarsPerAccess{}[idim];

        });

        if(diff != 1)
        {
            constexpr auto I1 = Number<1>{};
            constexpr auto I2 = Number<2>{};

            printf("Error: \n");
            printf("idx(%d) = [%d, %d, %d]\n", i.value - 1, idx0[I0], idx0[I1], idx0[I2]);
            printf("idx(%d) = [%d, %d, %d]\n", i.value, idx1[I0], idx1[I1], idx1[I2]);
            printf("delta_idx / ScalarsPerAccess = [%d, %d, %d]\n",
                   idx_diff[I0] / ScalarsPerAccess{}[I0],
                   idx_diff[I1] / ScalarsPerAccess{}[I1],
                   idx_diff[I2] / ScalarsPerAccess{}[I2]);
        }
        idx0 = idx1;
    });
}

