# thread_buffer Usage Guide: Applying Operations Without Repetition

## The Question

How to apply `ck_tile::exp()` to a `thread_buffer<float, 4>` without repeating the operation 4 times?

```cpp
// Instead of this repetitive code:
thread_buffer<float, 4> y;
thread_buffer<float, 4> exp_y;

exp_y[0] = ck_tile::exp(y[0]);
exp_y[1] = ck_tile::exp(y[1]);
exp_y[2] = ck_tile::exp(y[2]);
exp_y[3] = ck_tile::exp(y[3]);
```

## Solution 1: Using `static_for` (RECOMMENDED)

**Best for fixed-size buffers** - Fully unrolled at compile time, no runtime overhead.

```cpp
thread_buffer<float, 4> y;
thread_buffer<float, 4> exp_y;

static_for<0, 4, 1>{}([&](auto i) {
    exp_y[i] = ck_tile::exp(y[i]);
});
```

### Why `static_for`?
- **Compile-time unrolling**: Generates 4 separate instructions, just like manual repetition
- **Clean syntax**: Write the operation once
- **Type-safe**: Uses lambdas with perfect forwarding
- **Part of CK Tile**: Already available in `ck_tile/core/utility/functional.hpp`

## Solution 2: Using `#pragma unroll` Loop

**Best for runtime-sized buffers** - Familiar syntax, compiler handles optimization.

```cpp
thread_buffer<float, 4> y;
thread_buffer<float, 4> exp_y;

#pragma unroll
for (int i = 0; i < 4; i++) {
    exp_y[i] = ck_tile::exp(y[i]);
}
```

### Why `#pragma unroll`?
- **Familiar loop syntax**: Easy to read and understand
- **Compiler directive**: Hints to compiler to unroll the loop
- **Works with runtime sizes**: Unlike `static_for`
- **Standard practice**: Common in GPU kernels

## Solution 3: For Distributed Tensors (Advanced)

If you're working with CK Tile's distributed tensors, use the built-in helpers:

```cpp
#include "ck_tile/core/tensor/tile_elementwise.hpp"

// For in-place operation on tensors
tile_elementwise_inout([](auto& x) { x = ck_tile::exp(x); }, my_tensor);

// For creating new tensor with operation
auto exp_tensor = tile_elementwise_in([](auto x) { return ck_tile::exp(x); }, my_tensor);
```

### When to use?
- Working with `distributed_tensor` types
- Need automatic distribution handling
- Part of larger tile operations

## Complete Example

Here's a complete kernel using `static_for`:

```cpp
template <typename DataType>
__global__ void exp_kernel(DataType* output, const DataType* input, int size)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = 4;

    if (tid * stride < size)
    {
        // Load 4 elements
        thread_buffer<DataType, 4> y;
        for (int i = 0; i < 4; i++)
        {
            int idx = tid * stride + i;
            y[i] = (idx < size) ? input[idx] : 0.0f;
        }

        // Apply exp using static_for - NO REPETITION!
        thread_buffer<DataType, 4> exp_y;
        static_for<0, 4, 1>{}([&](auto i) {
            exp_y[i] = ck_tile::exp(y[i]);
        });

        // Store results
        for (int i = 0; i < 4; i++)
        {
            int idx = tid * stride + i;
            if (idx < size)
                output[idx] = exp_y[i];
        }
    }
}
```

## Comparison with ext_vector_type

Your original question compared to `__attribute__((ext_vector_type(4)))`:

```cpp
// Original C++ style:
using CVec = float __attribute__((ext_vector_type(4)));
const auto &[y_0, y_1, y_2, y_3] = y;
CVec exp_y{
    ck_tile::exp(y_0),
    ck_tile::exp(y_1),
    ck_tile::exp(y_2),
    ck_tile::exp(y_3),
};

// CK Tile equivalent (cleaner!):
thread_buffer<float, 4> y;
thread_buffer<float, 4> exp_y;

static_for<0, 4, 1>{}([&](auto i) {
    exp_y[i] = ck_tile::exp(y[i]);
});
```

## What About `get_as`?

You can use `get_as<fp32x4_t>()` to convert between `thread_buffer` and vector types:

```cpp
thread_buffer<float, 4> y;

// Get the underlying fp32x4_t vector
fp32x4_t y_vec = y.get_as<fp32x4_t>()[0];

// Now y_vec is float __attribute__((ext_vector_type(4)))
// But you still need to apply exp element-wise!
```

**Note**: `ck_tile::exp` doesn't have a vectorized version for `fp32x4_t`, so you'd still need element-wise application.

## Summary

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| `static_for` | Fixed-size buffers | Compile-time, clean syntax | Compile-time size only |
| `#pragma unroll` | Runtime-sized loops | Familiar syntax, flexible | Compiler-dependent |
| `tile_elementwise_*` | Distributed tensors | Automatic distribution | Overkill for simple buffers |
| Manual repetition | Very small (2-3 elements) | Explicit, simple | Repetitive, error-prone |

**Recommendation**: Use `static_for<0, N, 1>{}` with a lambda for fixed-size `thread_buffer` operations.

## See Also

- `tutorial_thread_buffer_methods.cpp` - Comparison of all methods
- `tutorial_thread_buffer_exp_simple.cpp` - CPU-side demonstration
- `tutorial_thread_buffer_exp.cpp` - Full GPU kernel example
- `include/ck_tile/core/utility/functional.hpp` - `static_for` implementation
- `include/ck_tile/core/tensor/tile_elementwise.hpp` - Tensor-level operations
