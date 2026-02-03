# Build Time Optimization

Tracking issue: [#3575](https://github.com/ROCm/composable_kernel/issues/3575)

This document describes techniques for reducing C++ template instantiation overhead in the Composable Kernel codebase.

## Why Build Time Matters

Composable Kernel relies heavily on C++ template metaprogramming to achieve GPU kernels with no runtime abstraction penalty. However, deep template instantiation can significantly impact build times. A single translation unit may trigger hundreds of thousands of template instantiations, with each instantiation adding to compile time.

## Key Types

This codebase uses compile-time types to enable zero-overhead abstractions:

- `Number<N>` - compile-time integer, enables static dispatch and compile-time arithmetic
- `Sequence<Is...>` - compile-time integer sequence, used for dimension ordering and index manipulation
- `Tuple<Ts...>` - heterogeneous container holding different types, used for tensor descriptors and transforms

These types allow the compiler to fully unroll loops, eliminate branches, and inline all operations - producing GPU kernels with no runtime abstraction cost.

## Optimization Techniques

### 1. Replace Recursive Templates with Pack Expansion

Recursive template patterns create O(N) instantiation depth - the compiler must instantiate each level before proceeding to the next:

```
sequence_gen_impl<5, F>
  → sequence_gen_impl<4, F>
    → sequence_gen_impl<3, F>
      → ...
```

Using `__make_integer_seq` (Clang/MSVC) combined with pack expansion reduces this to constant depth - the compiler generates the entire sequence in one step internally, without recursive template instantiation.

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

**After** (constant depth using compiler intrinsic + pack expansion):

```cpp
namespace detail {

template <typename T, T... Is>
struct sequence_gen_helper
{
    // Apply functor F to all indices via pack expansion
    // F{}(Number<0>{}), F{}(Number<1>{}), ..., F{}(Number<N-1>{})
    template <typename F>
    using apply = Sequence<F{}(Number<Is>{})...>;
};

} // namespace detail

template <index_t N, typename F>
struct sequence_gen
{
    // __make_integer_seq<sequence_gen_helper, index_t, N> produces
    // sequence_gen_helper<index_t, 0, 1, ..., N-1> with constant depth
    using type =
        typename __make_integer_seq<detail::sequence_gen_helper, index_t, N>::template apply<F>;
};
```

Note: This document assumes C++17 or later. While `std::make_integer_sequence` (introduced in C++14) is the standard library facility for generating integer sequences, it only produces `std::integer_sequence<T, ...>`. We use `__make_integer_seq` directly because it accepts any template as its first argument, enabling this pattern where the helper class receives the index pack directly.

### 2. Replace Lambdas with Named Functors

Each lambda expression creates a unique closure type, causing separate template instantiations at every call site. Named functors share a single type across all uses.

**Before** (lambda creates unique instantiations at each call site):

```cpp
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

This significantly reduces template instantiations for `transform_tensor_descriptor`.

**Example: container_concat**

```cpp
// Before: lambda creates unique type per call site
// (unpack2 applies a functor to all elements from both tuples)
template <typename... X, typename... Y>
__host__ __device__ constexpr auto container_concat(const Tuple<X...>& tx, const Tuple<Y...>& ty)
{
    return unpack2([](auto&&... zs) { return make_tuple(forward<decltype(zs)>(zs)...); }, tx, ty);
}

// After: named functor shares instantiations
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

This reduces `container_concat` template instantiations.

**Example: make_uniform_tuple**

For patterns that create tuples with repeated values:

```cpp
// Before: unique lambda type at each call site
generate_tuple([](auto) { return some_value; }, Number<N>{});

// After: dedicated helper function
template <index_t N, typename T>
__host__ __device__ constexpr auto make_uniform_tuple(T&& value)
{
    return detail::make_uniform_tuple_impl(static_cast<T&&>(value), make_index_sequence<N>{});
}

// Usage
make_uniform_tuple<N>(some_value);
```

### 3. Use Constexpr Loops Instead of Template Recursion

Template recursion creates N template instantiations for N iterations. A constexpr loop executes at compile time but only requires a single template instantiation. While both are O(N) in complexity, constexpr loops are significantly faster because they avoid the overhead of template instantiation.

**Before** (O(N) template instantiations):

```cpp
// Simplified example - actual CK code used more complex recursive patterns
template <index_t Target, typename Seq, index_t Pos, bool AtEnd>
struct find_source_index_impl
{
    static constexpr index_t value =
        (Seq::template At<Pos>() == Target) ? Pos : find_source_index_impl<Target, Seq, Pos+1, (Pos+1 == Seq::Size())>::value;
};

template <index_t Target, typename Seq, index_t Pos>
struct find_source_index_impl<Target, Seq, Pos, true>
{
    static constexpr index_t value = -1; // not found
};
```

**After** (single instantiation with constexpr loop):

```cpp
template <index_t Target, index_t... Is>
__host__ __device__ constexpr index_t find_source_index(Sequence<Is...>)
{
    // Simplified example - actual implementation handles empty sequences
    constexpr index_t values[] = {Is...};
    for(index_t i = 0; i < sizeof...(Is); ++i)
        if(values[i] == Target) return i;
    return -1; // not found
}
```

This significantly reduces `sequence_map_inverse` instantiations and compile time.

### 4. Use Fold Expressions for Accumulation

Fold expressions (C++17) can replace recursive template patterns for accumulation operations.

**Before** (uses helper utilities that hide template recursion: `generate_tuple` recursively constructs a tuple of N elements, and `container_reduce` recursively reduces that tuple):

```cpp
const auto element_space_size = container_reduce(
    generate_tuple([&](auto i) {
        return (lengths[i] - Number<1>{}) * strides[i];
    }, Number<N>{}),
    math::plus{}, Number<1>{});
```

**After** (single fold expression):

```cpp
template <typename... Lengths, typename... Strides, index_t... Is>
__host__ __device__ constexpr auto compute_element_space_size(
    const Tuple<Lengths...>& lengths,
    const Tuple<Strides...>& strides,
    Sequence<Is...>)
{
    return (LongNumber<1>{} + ... +
            ((lengths[Number<Is>{}] - Number<1>{}) * strides[Number<Is>{}]));
}
```

This reduces `calculate_element_space_size` instantiations and compile time.
