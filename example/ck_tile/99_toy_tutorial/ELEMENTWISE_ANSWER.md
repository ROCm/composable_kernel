# Does `tile_elementwise` Work on `thread_buffer`?

## Short Answer: NO

`tile_elementwise_in` and `tile_elementwise_inout` are designed for **distributed tensors/tiles**, NOT raw `thread_buffer`.

## What Are They For?

These functions work on **distributed tiles** - high-level tensor objects that:
- Manage thread buffers internally
- Know about tile distribution across threads
- Are created by `load_tile()` or `make_static_distributed_tensor()`

## Example: Using tile_elementwise (CORRECT)

```cpp
// This works - operating on distributed tiles
auto input_tile = load_tile(input_window);  // Returns distributed_tensor

auto output_tile = tile_elementwise_in(
    [&](const auto& val) {
        return ck_tile::exp(val);  // Applied to each element
    },
    input_tile  // Works because this is a distributed_tensor!
);

store_tile(output_window, output_tile);
```

## What Happens Inside?

Looking at the implementation (`tile_elementwise.hpp:40-60`):

```cpp
template <typename InElementFunc, typename... InTensor>
CK_TILE_DEVICE auto tile_elementwise_in(const InElementFunc& in_element_func,
                                        const InTensor&... in_dstr_tensors)
{
    // Gets the thread_buffer from the distributed tensor
    constexpr index_t thread_buffer_size =
        __type_pack_element<0, InTensor...>::get_thread_buffer_size();

    auto out_dstr_tensor = make_static_distributed_tensor<OutDataType>(in_tile_dstr);

    // Applies function to each element in the thread buffer
    static_for<0, thread_buffer_size, 1>{}([&](auto i) {
        out_dstr_tensor.get_thread_buffer()(i) =
            in_element_func(in_dstr_tensors.get_thread_buffer()[i]...);
    });

    return out_dstr_tensor;
}
```

**Key insight**: It calls `get_thread_buffer()` on the distributed tensor, then uses `static_for` internally!

## For Raw thread_buffer: Use static_for Directly

```cpp
// For thread_buffer<float, 4>, use static_for directly:
thread_buffer<float, 4> y;
thread_buffer<float, 4> exp_y;

static_for<0, 4, 1>{}([&](auto i) {
    exp_y[i] = ck_tile::exp(y[i]);
});
```

This is exactly what `tile_elementwise` does internally!

## Summary Table

| Type | Use | Function |
|------|-----|----------|
| `thread_buffer<T, N>` | Raw register buffer | `static_for` or `#pragma unroll` |
| `distributed_tensor<...>` | High-level tile | `tile_elementwise_in/inout` |
| Loaded from memory | Use `load_tile()` first | Then `tile_elementwise_*` |

## Complete Working Example

```cpp
// Method 1: Raw thread_buffer (what you have)
template <typename DataType>
__global__ void kernel(DataType* output, const DataType* input, int size)
{
    thread_buffer<DataType, 4> y;
    // ... load data ...

    // Use static_for (this is the right way!)
    thread_buffer<DataType, 4> exp_y;
    static_for<0, 4, 1>{}([&](auto i) {
        exp_y[i] = ck_tile::exp(y[i]);
    });

    // ... store data ...
}

// Method 2: Using distributed tensors (higher level)
template <typename Problem>
__global__ void kernel_with_tiles(/* ... */)
{
    // Create tile window
    auto input_window = make_tile_window(/* ... */);

    // Load creates a distributed_tensor
    auto input_tile = load_tile(input_window);

    // Now tile_elementwise works!
    auto output_tile = tile_elementwise_in(
        [](auto x) { return ck_tile::exp(x); },
        input_tile
    );

    // Store back
    store_tile(output_window, output_tile);
}
```

## Recommendation

For your use case (applying `exp` to a `thread_buffer<float, 4>`):

**Use `static_for`** - It's simple, direct, and exactly what the high-level functions use internally!

```cpp
thread_buffer<float, 4> y;
thread_buffer<float, 4> exp_y;

static_for<0, 4, 1>{}([&](auto i) {
    exp_y[i] = ck_tile::exp(y[i]);
});
```

✓ No repetition
✓ Fully unrolled at compile time
✓ Clean, readable code
✓ Part of CK Tile's core utilities
