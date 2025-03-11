# Batched Transpose

This folder contains examples for batched transpose using `ck_tile` tile-programming implementation. Currently, it supports transpose with:

- **2D transpose:** Swaps width and height dimensions.
- **4D transpose:** Converts batched data layout from **NCHW** to **NHWC**.

Now the transpose read with single data point. We would soon put it in vectorized transpose.

## Build

```sh
# In the root of `composable_kernel`
mkdir build && cd build

# You can replace <arch> with the appropriate architecture (e.g., gfx90a or gfx942) or leave it blank
sh ../script/cmake-ck-dev.sh  ../ <arch>

# Build the transpose executables
make tile_example_batched_transpose_2d -j
make tile_example_batched_transpose_4d -j
```

This will generate the executable:

```
build/bin/tile_example_batched_transpose_2d
build/bin/tile_example_batched_transpose_4d
```

## Example

### 2D Transpose

```sh
./bin/tile_example_batched_transpose_2d -H 16 -W 16
```

#### Arguments:

- `-H` : Input height size (default: 16)
- `-W` : Input width size (default: 16)

### 4D Transpose (NCHW -> NHWC)

```sh
./bin/tile_example_batched_transpose_4d -N 2 -C 16 -H 1 -W 16 -layout_in NCHW -layout_out NHWC
```

#### Arguments:

- `-N`  : Input batch size (default: 2)
- `-C`  : Input channel size (default: 16)
- `-H`  : Input height size (default: 1)
- `-W`  : Input width size (default: 16)
- `-layout_in`  : Input tensor layout (default: NCHW)
- `-layout_out` : Output tensor layout (default: NHWC)
- `-v`  : Enable CPU validation (default: 1)
- `-seed` : Random seed (-1 means random every time, default: -1)
