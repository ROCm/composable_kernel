#include <vector>
#include <iostream>
#include <numeric>

#include "tensor_descriptor_helper.hpp"
#include "tensor_space_filling_curve.hpp"

using namespace ck;

template <typename TensorDesc>
__global__ void test_iterator(const TensorDesc desc);

int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;
    constexpr int N0 = 16;
    constexpr int N1 = 8;
    constexpr int N2 = 16;

    constexpr auto I0 = Number<0>{};
    // constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    // // constexpr auto I3 = Number<3>{};

#if 0
    std::vector<int> data(N0 * N1 * N2);
    std::iota(data.begin(), data.end(), 0);
    auto desc = make_naive_tensor_descriptor_packed(make_tuple(N0, N1, N2));

    {
        constexpr int nDim = decltype(desc)::GetNumOfVisibleDimension();
        using Index        = MultiIndex<nDim>;
        using TensorCoord  = decltype(make_tensor_coordinate(desc, Index{}));
        // TensorCoord coord  = make_tensor_coordinate(desc, make_tuple(1, 2, 3));

        // printf("%d\n", coord);
    }

    {
        printf("\n");
        auto desc0 = transform_tensor_descriptor(
            desc,
            make_tuple(make_pass_through_transform(N0), make_merge_transform(make_tuple(N1, N2))),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        constexpr int nDim = decltype(desc0)::GetNumOfVisibleDimension();
        static_assert(nDim == 2);
        using Index        = MultiIndex<nDim>;
        using TensorCoord  = decltype(make_tensor_coordinate(desc0, Index{}));
        TensorCoord coord  = make_tensor_coordinate(desc0, make_tuple(1, 2));

        printf("coord.index = [%d, %d]\n", coord.GetIndex()[I0], coord.GetIndex()[I1]);
        // std::cout << "coord.index = " << coord.GetIndex() << std::endl;
        const auto step = make_tensor_coordinate_step(desc0, Index(1, 1));
        printf("step.index = [%d, %d]\n", step.GetIndexDiff()[I0], step.GetIndexDiff()[I1]);

        // move_tensor_coordinate(desc0, coord, Index(1, 1));
        move_tensor_coordinate(desc0, coord, step);
        printf("coord.index = [%d, %d]\n", coord.GetIndex()[I0], coord.GetIndex()[I1]);
    }

    {
        printf("\n");
        auto desc0 = transform_tensor_descriptor(
            desc,
            make_tuple(make_pass_through_transform(N0), make_slice_transform(N1, 0, N1-1), make_slice_transform(N2, 0, N2-1)),

            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        constexpr int nDim = decltype(desc0)::GetNumOfVisibleDimension();
        static_assert(nDim == 3);
        using Index        = MultiIndex<nDim>;
        using TensorCoord  = decltype(make_tensor_coordinate(desc0, Index{}));
        TensorCoord coord  = make_tensor_coordinate(desc0, make_tuple(0, N1-2, N2-2));

        printf("coord.index = [%d, %d, %d]\n", coord.GetIndex()[I0], coord.GetIndex()[I1], coord.GetIndex()[I2]);
        // std::cout << "coord.index = " << coord.GetIndex() << std::endl;
        const auto step = make_tensor_coordinate_step(desc0, Index(0, 1, 1));
        printf("step.index = [%d, %d, %d]\n", step.GetIndexDiff()[I0], step.GetIndexDiff()[I1], step.GetIndexDiff()[I2]);

        // move_tensor_coordinate(desc0, coord, Index(1, 1));
        move_tensor_coordinate(desc0, coord, step);
        printf("coord.index = [%d, %d, %d]\n", coord.GetIndex()[I0], coord.GetIndex()[I1], coord.GetIndex()[I2]);
        printf("%d\n", to_multi_index(coord.GetIndex()) + Index{1, 1, 1});

    }

#endif

    {
        constexpr auto desc = make_naive_tensor_descriptor_packed(
            make_tuple(Number<N0>{}, Number<N1>{}, Number<N2>{}));
        printf("\n");
        auto desc0 = transform_tensor_descriptor(
            desc,
            make_tuple(make_slice_transform(Number<N0>{}, I0, Number<N0>{} / I2),
                       make_slice_transform(Number<N1>{}, I0, Number<N1>{} / I2),
                       make_slice_transform(Number<N2>{}, I0, Number<N2>{} / I2)),

            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

        test_iterator<<<1, 1>>>(desc0);
        hipDeviceSynchronize();
    }
    return 0;
}

template <typename TensorDesc>
__global__ void test_iterator(const TensorDesc desc)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};

    using SliceLengths     = Sequence<4, 3, 6>;
    using DimAccessOrder   = Sequence<2, 1, 0>;
    using ScalarsPerAccess = Sequence<2, 1, 2>;
    using SpaceFillingCurve =
        SpaceFillingCurve<decltype(desc), SliceLengths, DimAccessOrder, ScalarsPerAccess>;

    constexpr index_t num_accesses = SpaceFillingCurve::GetNumOfAccess();
    static_assert(num_accesses == reduce_on_sequence(SliceLengths{} / ScalarsPerAccess{}, math::multiplies{}, Number<1>{}));

    auto idx0 = SpaceFillingCurve::GetIndex(I0);

    printf("i = %d, idx = [%d, %d, %d]\n", 0, idx0[I0], idx0[I1], idx0[I2]);
    static_for<1, num_accesses, 1>{}([&](auto i) {
        constexpr auto idx1 = SpaceFillingCurve::GetIndex(i);
        const auto idx_diff = idx1 - idx0;
        index_t diff        = 0;
        static_for<0, TensorDesc::GetNumOfVisibleDimension(), 1>{}([&](auto idim) {
            diff += abs(idx_diff[idim]) / ScalarsPerAccess{}[idim];

        });
        printf(
            "i = %d, idx = [%d, %d, %d], diff = %d\n", i.value, idx1[I0], idx1[I1], idx1[I2], diff);
        idx0 = idx1;
    });
}

#if 0
/*
 * Naive tensor shape (16, 8, 16), desc shape (8, 4, 8)
 */
template <typename TensorDesc>
__global__ void test_iterator(const TensorDesc desc)
{
    __shared__ int sdata[1024 * 2];

    for(int ii = 0; ii < 1024 * 2; ++ii)
    {
        sdata[ii] = ii;
    }

    auto tensor_buf   = make_dynamic_buffer<AddressSpaceEnum_t::Lds>(static_cast<int*>(sdata),
                                                                   desc.GetElementSpaceSize());
    using Iterator    = TensorIterator<int,
                                    decltype(desc),
                                    decltype(tensor_buf),
                                    Sequence<4, 3, 6>,
                                    Sequence<0, 1, 2>,
                                    Sequence<0, 0, 2>>;
    auto iter = Iterator(tensor_buf);

    for(int ii = 0; ii < 8*4*16 / (2); ++ii)
    // for(int ii = 0; ii < 4; ++ii)
    {
        const auto val = iter.Get(ii);
        // printf("sdata[%d] = %d, %d, %d, %d\n", ii, val[0], val[1], val[2], val[3]);
        printf("sdata[%d] = %d, %d\n", ii, val[0], val[1]);
        ++iter;
    }
}
#endif
