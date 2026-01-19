# Build Time Optimization

This document describes techniques for reducing C++ template instantiation overhead in the Composable Kernel codebase.

## Why Build Time Matters

Composable Kernel relies heavily on C++ template metaprogramming to achieve GPU kernels with no runtime abstraction penalty. However, deep template instantiation can significantly impact build times. A single translation unit may trigger hundreds of thousands of template instantiations, with each instantiation adding to compile time.

## Measuring Build Time

Use Clang's `-ftime-trace` flag to generate JSON build traces:

```bash
# Build with time trace enabled
cmake -DCMAKE_CXX_FLAGS="-ftime-trace -ftime-trace-granularity=1" ..
ninja example_gemm_xdl_fp16

# Find the trace file
find . -name "*.json" -path "*/CMakeFiles/*"
```

The trace file can be viewed in Chrome's `chrome://tracing` or analyzed with tools like [ClangBuildAnalyzer](https://github.com/aras-p/ClangBuildAnalyzer).

Key metrics to monitor:

- **Template instantiation count**: Total number of unique template instantiations
- **Template instantiation depth**: Maximum recursion depth during instantiation
- **Wall-clock time**: Actual time spent instantiating templates

The `script/tools/ck-build-analysis` script automates trace collection and analysis:

```bash
script/tools/ck-build-analysis example_gemm_xdl_fp16 --granularity=1
```

## Optimization Techniques

### 1. Replace O(N) Recursion with O(1) Pack Expansion

Recursive template patterns create O(N) instantiation depth. Use compiler intrinsics and fold expressions for O(1) depth.

**Before** (O(N) recursive instantiation):

```cpp
template <index_t N, typename F, index_t... Is>
struct sequence_gen_impl
{
    using type = typename sequence_gen_impl<N-1, F, F{}(Number<N-1>{}), Is...>::type;
};

template <typename F, index_t... Is>
struct sequence_gen_impl<0, F, Is...>
{
    using type = Sequence<Is...>;
};
```

**After** (O(1) using compiler intrinsic):

```cpp
template <index_t N, typename F>
struct sequence_gen
{
    template <index_t... Is>
    static constexpr auto make(std::integer_sequence<index_t, Is...>)
    {
        return Sequence<F{}(Number<Is>{})...>{};
    }
    using type = decltype(make(__make_integer_seq<std::integer_sequence, index_t, N>{}));
};
```

The `__make_integer_seq` intrinsic (available in Clang and MSVC) generates integer sequences with O(1) template depth.

### 2. Replace Lambdas with Named Functors

Each lambda expression creates a unique closure type, causing separate template instantiations at every call site.

**Before** (lambda creates unique instantiations):

```cpp
// Called in multiple places - each creates new instantiations
auto result = transform_tensor_descriptor(
    desc,
    make_tuple(make_pass_through_transform(Length)),
    make_tuple(Sequence<0>{}),
    make_tuple(Sequence<0>{}));

// The lambda inside transform_tensor_descriptor:
generate_tuple([](auto i) { return Sequence<i>{}; }, Number<N>{});
```

**After** (named functor shares instantiations):

```cpp
// Define functor once
struct generate_identity_sequence
{
    template <index_t I>
    __host__ __device__ constexpr auto operator()(Number<I>) const
    {
        return Sequence<I>{};
    }
};

// Use everywhere - shares instantiations
generate_tuple(generate_identity_sequence{}, Number<N>{});
```

This reduced `transform_tensor_descriptor` instantiations from 388 to 32 (92% reduction).

#### container_concat optimization

The same pattern applies to utility functions like `container_concat`:

**Before**:

```cpp
template <typename... X, typename... Y>
__host__ __device__ constexpr auto container_concat(const Tuple<X...>& tx, const Tuple<Y...>& ty)
{
    return unpack2([](auto&&... zs) { return make_tuple(forward<decltype(zs)>(zs)...); }, tx, ty);
}
```

**After**:

```cpp
struct make_tuple_functor
{
    template <typename... Ts>
    __host__ __device__ constexpr auto operator()(Ts&&... xs) const
    {
        return make_tuple(forward<Ts>(xs)...);
    }
};

template <typename... X, typename... Y>
__host__ __device__ constexpr auto container_concat(const Tuple<X...>& tx, const Tuple<Y...>& ty)
{
    return unpack2(make_tuple_functor{}, tx, ty);
}
```

This reduced `container_concat` instantiations from 186 to 93 (50% reduction).

#### make_uniform_tuple helper

For patterns that create tuples with repeated values, use dedicated helpers instead of lambdas:

**Before**:

```cpp
// Creates unique lambda type at each call site
generate_tuple([](auto) { return some_value; }, Number<N>{});
```

**After**:

```cpp
// Defined once, shared across all call sites
template <index_t N, typename T>
__host__ __device__ constexpr auto make_uniform_tuple(T&& value)
{
    return detail::make_uniform_tuple_impl(static_cast<T&&>(value), make_index_sequence<N>{});
}

// Usage
make_uniform_tuple<N>(some_value);
```

#### sequence_map_inverse optimization

The `sequence_map_inverse` template inverts a permutation sequence. The original implementation used O(N) recursive template instantiations.

**Before** (O(N) recursive instantiation):

```cpp
template <index_t Target, typename Seq, index_t Pos>
struct find_source_index_impl
{
    static constexpr index_t value =
        (Seq::template At<Pos>() == Target) ? Pos : find_source_index_impl<Target, Seq, Pos+1>::value;
};
```

**After** (O(1) using constexpr array lookup):

```cpp
namespace detail {
template <index_t Target, index_t... Is>
__host__ __device__ constexpr index_t find_source_index(Sequence<Is...>)
{
    constexpr index_t values[] = {Is...};
    for(index_t i = 0; i < sizeof...(Is); ++i)
        if(values[i] == Target) return i;
    return 0;
}

template <typename SeqMap, index_t... Positions>
__host__ __device__ constexpr auto invert_permutation_impl(Sequence<Positions...>)
{
    return Sequence<find_source_index<Positions>(SeqMap{})...>{};
}
} // namespace detail

template <typename SeqMap>
struct sequence_map_inverse
{
    using type = decltype(detail::invert_permutation_impl<SeqMap>(
        typename arithmetic_sequence_gen<0, SeqMap::Size(), 1>::type{}));
};
```

This reduced instantiations from 45 to 10 (78% reduction) and wall-clock time by 95%.

#### calculate_element_space_size optimization

Computing element space size for tensor descriptors can use a fold expression instead of recursive template instantiation.

**Before** (recursive or loop-based approach):

```cpp
// Implicit recursion through generate_tuple and container_reduce
const auto element_space_size = container_reduce(
    generate_tuple([&](auto i) {
        return (lengths[i] - I1) * strides[i];
    }, Number<N>{}),
    math::plus{}, LongNumber<1>{});
```

**After** (O(1) using fold expression):

```cpp
namespace detail {
template <typename... Lengths, typename... Strides, index_t... Is>
__host__ __device__ constexpr auto compute_element_space_size(
    const Tuple<Lengths...>& lengths,
    const Tuple<Strides...>& strides,
    Sequence<Is...>)
{
    return (LongNumber<1>{} + ... +
            ((lengths[Number<Is>{}] - Number<1>{}) * strides[Number<Is>{}]));
}
} // namespace detail
```

This reduced instantiations from 24 to 10 (58% reduction) and wall-clock time by 73%.

### 3. Use Constexpr Arrays Instead of Template Recursion

Replace recursive template searches with constexpr functions using arrays.

**Before** (O(N) recursive template search):

```cpp
template <index_t Target, typename FirstSeq, typename... RestSeqs>
struct find_in_tuple_of_sequences_impl
{
    static constexpr index_t pos = sequence_find<Target>(FirstSeq{});
    static constexpr bool found_here = (pos >= 0);

    using next = find_in_tuple_of_sequences_impl<Target, RestSeqs...>;

    static constexpr index_t itran = found_here ? 0 : 1 + next::itran;
    static constexpr index_t idim_up = found_here ? pos : next::idim_up;
};
```

**After** (O(1) pack expansion with constexpr array):

```cpp
template <index_t Target, typename... Seqs>
struct FindInTupleOfSequencesCompute
{
    static constexpr auto compute()
    {
        if constexpr(sizeof...(Seqs) == 0) {
            return ResultData{0, 0, false};
        } else {
            // Pack expansion creates array - O(1) template depth
            constexpr index_t indices[] = {sequence_find_value<Target>(Seqs{})...};
            for(index_t i = 0; i < sizeof...(Seqs); ++i)
                if(indices[i] >= 0) return ResultData{i, indices[i], true};
            return ResultData{0, 0, false};
        }
    }
};
```

This reduced instantiations by 50% and wall-clock time by 69%.

### 4. Avoid Unnecessary Template Parameter Variations

Templates with many parameter combinations cause combinatorial explosion.

- Cache template results where possible
- Use type erasure for runtime-only variations
- Consider `if constexpr` to reduce branch template instantiations

## Case Studies

The following PRs demonstrate these techniques applied to Composable Kernel:

- **sequence_gen optimization**: Replaced O(N) recursion with `__make_integer_seq` intrinsic
- **transform_tensor_descriptor**: Replaced lambdas with named functors (92% instantiation reduction)
- **container_concat**: Replaced lambdas with named functors (50% instantiation reduction)
- **sequence_map_inverse**: Replaced O(N) recursion with pack expansion (78% instantiation, 95% time reduction)
- **calculate_element_space_size**: Replaced implicit recursion with fold expression (58% instantiation, 73% time reduction)
- **find_in_tuple_of_sequences**: Replaced recursive search with pack expansion (50% reduction)
- **sequence_merge**: Replaced O(log N) recursion with O(1) fold expression

See tracking issue [#3575](https://github.com/ROCm/composable_kernel/issues/3575) for the full list of PRs.

## Tools and Commands

Identify optimization targets:

```bash
# Run analysis on a specific target
script/tools/ck-build-analysis example_convnd_fwd_xdl_fp16 --granularity=1

# View the generated report
cat build_time_analysis_report.md
```

The report shows template instantiation counts, wall-clock times, and identifies the most expensive templates.
