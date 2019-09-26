#ifndef CK_TENSOR_VISIT_HPP
#define CK_TENSOR_VISIT_HPP

#include "common_header.hpp"
#include "dimension.hpp"
#include "dimension_transform.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_coordinate.hpp"

namespace ck {

template <class TensorDescriptor>
struct TensorVisit
{
    using Index      = typename TensorDescriptor::Index;
    using Coordinate = typename TensorCoordinate<TensorDescriptor>::type;

    __host__ __device__ static void Run_v1(Index idx_begin)
    {
        const auto coord_begin = Coordinate(idx_begin);

        ford<TensorDescriptor::GetLengths()>{}(
            [&](auto idx_diff) { index_t offset = (coord_begin + idx_diff).GetOffset(); });
    }

    __host__ __device__ static void Run_v2(Index idx_begin)
    {
        const auto coord_begin = Coordinate(idx_begin);

        ford<TensorDescriptor::GetLengths()>{}([&](auto idx_diff) {
            index_t offset_diff = coord_begin.GetOffsetDiff(idx_diff);
            index_t offset      = coord_begin.GetOffset() + offset_diff;
        });
    }

    __host__ __device__ static void Run_v3(Index idx_begin)
    {
        const auto coord_begin = Coordinate(idx_begin);

        constexpr auto linear_dimensions    = TensorDescriptor::GetLinearDimensions();
        constexpr auto nonlinear_dimensions = TensorDescriptor::GetNonLinearDimensions();

        constexpr auto lengths = TensorDescriptor::GetLengths();

        constexpr auto linear_dimension_lengths_hack =
            lambda_HackLengths{}(lengths, linear_dimensions);
        constexpr auto nonlinear_dimension_lengths_hack =
            lambda_HackLengths{}(lengths, nonlinear_dimensions);

        ford<nonlinear_dimension_lengths_hack>{}([&](auto idx_diff_nonlinear_hack) {
            // run-time component
            index_t offset_diff_nonlinear = coord_begin.GetOffsetDiff(idx_diff_nonlinear_hack);

            ford<linear_dimension_lengths_hack>{}([&](auto idx_diff_linear_hack) {
                // compile-time component
                index_t offset_diff_linear = coord_begin.GetOffsetDiff(idx_diff_linear_hack);

                index_t offset =
                    coord_begin.GetOffset() + offset_diff_nonlinear + offset_diff_linear;
            });
        });
    }

    __host__ __device__ static void Run_v4(Index idx_begin)
    {
        const auto coord_begin = Coordinate(idx_begin);

        constexpr auto linear_dimensions = TensorDescriptor::GetLinearDimensions();

        constexpr auto nonlinear_independent_dimension_groups =
            TensorDescriptor::GetNonLinearIndependentDimensionGroups();

        constexpr auto lengths = TensorDescriptor::GetLengths();

        constexpr auto linear_dimension_lengths = lambda_HackLengths{}(lengths, linear_dimensions);

        // run-time component
        index_t offset_diff_nonlinear = 0;

        template <index_t NGroup>
        struct f_recursion
        {
            template <index_t IGroup>
            __host__ __device__ void Run(Number<IGroup>)
            {
                constexpr auto nonlinear_independent_dimensions_igroup =
                    nonlinear_independent_dimension_groups.Get(igroup);

                constexpr auto nonlinear_independent_lengths_igroup =
                    lambda_HackLengths{}(lengths, nonlinear_independent_dimensions_igroup);

                ford<nonlinear_independent_lengths_igroup>{}(
                    [&](auto idx_diff_nonlinear_igroup_hack) {
                        // run-time component
                        offset_diff_nonlinear +=
                            coord_begin.GetOffsetDiff(idx_diff_nonlinear_igroup_hack);

                        Run(Number<IGroup + 1>{});
                    });
            };

            // inner-most work
            template <>
            __host__ __device__ void Run(Number<NGroup>)
            {
                ford<linear_dimension_lengths>{}([&](auto idx_diff_linear_hack) {
                    // compile-time component
                    index_t offset_diff_linear = coord_begin.GetOffsetDiff(idx_diff_linear_hack);

                    index_t offset =
                        coord_begin.GetOffset() + offset_diff_nonlinear + offset_diff_linear;
                });
            }
        };

        // run-time component
        index_t offset_diff_nonlinear = 0;

        f_recursion<nonlinear_independent_dimension_groups.GetSize()>{}.Run();
    }
};

} // namespace ck
#endif
