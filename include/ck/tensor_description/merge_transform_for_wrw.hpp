#pragma once

#include "common_header.hpp"
#include "multi_index_transform.hpp"

namespace ck {

// Implementation of "Merge" transformation primitive that uses division and mod. It is supposed to
// be used for low_lengths that are known at compile time and are power of 2, otherwise performance
// will be very bad
template <typename LowLengths>
struct Merge_v3_division_mod_for_wrw
{
    static constexpr index_t NDimLow = LowLengths::Size();

    using LowerIndex = MultiIndex<NDimLow>;
    using UpperIndex = MultiIndex<1>;

    using LowLengthsScan =
        decltype(container_reverse_exclusive_scan(LowLengths{}, math::multiplies{}, Number<1>{}));

    using UpLengths =
        decltype(make_tuple(container_reduce(LowLengths{}, math::multiplies{}, Number<1>{})));

    LowLengths low_lengths_;
    LowLengthsScan low_lengths_scan_;
    UpLengths up_lengths_;

    __host__ __device__ constexpr Merge_v3_division_mod_for_wrw() = default;

    __host__ __device__ constexpr Merge_v3_division_mod_for_wrw(const LowLengths& low_lengths)
        : low_lengths_{low_lengths},
          low_lengths_scan_{
              container_reverse_exclusive_scan(low_lengths, math::multiplies{}, Number<1>{})},
          up_lengths_{make_tuple(container_reduce(low_lengths, math::multiplies{}, Number<1>{}))}
    {
        static_assert(LowerIndex::Size() == NDimLow, "wrong!");
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return NDimLow; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        index_t tmp = idx_up[Number<0>{}];

        // division and mod
        static_for<0, NDimLow - 1, 1>{}([&](auto i) {
            idx_low(i) = tmp / this->low_lengths_scan_[i];
            tmp %= this->low_lengths_scan_[i];
        });

        idx_low(Number<NDimLow - 1>{}) = tmp;
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              const UpIdxDiff& idx_up_diff,
                                              LowIdx& idx_low,
                                              const UpIdx& idx_up_new,
                                              Number<Hack>) const
    {
        static_assert(LowIdxDiff::Size() == NDimLow && UpIdxDiff::Size() == 1 &&
                          LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0   = Number<0>{};
        constexpr auto INm1 = Number<NDimLow - 1>{};

        index_t tmp = idx_up_new[I0];

        //if(get_block_1d_id() == 0 && get_thread_local_1d_id() == 0){
        //    //printf("%d, %d, %d\n", __LINE__, tmp, tmp2);
        //    //printf("%d, %d, %d\n", 
        //    //        __LINE__, 
        //    //        static_cast<index_t>(this->low_lengths_scan_.At(Number<0>())),
        //    //        static_cast<index_t>(this->low_lengths_scan_.At(Number<1>())));
        //    printf("%d, %d, %d, %d, %d, %d\n", __LINE__, NDimLow, idx_low.At(Number<0>()), idx_low.At(Number<1>()), idx_diff_low.At(Number<0>()), idx_diff_low.At(Number<1>()));
        //}

        //static_for<0, NDimLow - 1, 1>{}([&](auto i) {
        //    const index_t tmp2 = idx_low[i];
        //    idx_low(i)         = tmp / this->low_lengths_scan_[i];
        //    idx_diff_low(i)    = idx_low[i] - tmp2;
        //    tmp %= this->low_lengths_scan_[i];
        //});

        //const index_t tmp2 = idx_low[INm1];
        //idx_low(INm1)      = tmp;
        //idx_diff_low(INm1) = idx_low[INm1] - tmp2;

        idx_low(INm1)      = tmp;
        idx_diff_low(INm1) = idx_up_diff[I0];

        //if(get_block_1d_id() == 0 && get_thread_local_1d_id() == 0){
        //    //printf("%d, %d, %d\n", __LINE__, tmp, tmp2);
        //    printf("%d, %d, %d\n", 
        //            __LINE__, 
        //            static_cast<index_t>(this->low_lengths_scan_.At(Number<0>())),
        //            static_cast<index_t>(this->low_lengths_scan_.At(Number<1>())));
        //    printf("%d, %d, %d, %d, %d, %d\n", __LINE__, NDimLow, idx_low.At(Number<0>()), idx_low.At(Number<1>()), idx_diff_low.At(Number<0>()), idx_diff_low.At(Number<1>()));
        //}
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return false; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<LowLengths>::value &&
               is_known_at_compile_time<LowLengthsScan>::value &&
               is_known_at_compile_time<UpLengths>::value;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("Merge_v3_direct_division_mod_wrw, ");
        printf("low_lengths_ ");
        print_multi_index(low_lengths_);
        printf("low_lengths_scan_ ");
        print_multi_index(low_lengths_scan_);
        printf("up_lengths_ ");
        print_multi_index(up_lengths_);
        printf("}");
    }
};

template <typename LowLengths>
__host__ __device__ constexpr auto
make_merge_transform_v3_division_mod_for_wrw(const LowLengths& low_lengths)
{
    return Merge_v3_division_mod_for_wrw<LowLengths>{low_lengths};
}

}
