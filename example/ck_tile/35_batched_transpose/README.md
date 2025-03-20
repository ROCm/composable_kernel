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

This will generate the executables:

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

```
args:
          -v    whether do CPU validation or not (default:1)
         -pr    input data type. int8/fp16/fp32 (representing 8/16/32 bit data) (default:fp16)
          -N    input batch size.  (default:2)
          -H    input height size. (default:16)
          -W    input width size.  (default:16)
       -seed    seed to be used, -1 means random every time (default:-1)
```

### 4D Transpose (NCHW -> NHWC)

```sh
./bin/tile_example_batched_transpose_4d -N 2 -C 16 -H 1 -W 16 -layout_in NCHW -layout_out NHWC
```

#### Arguments:

```
args:
          -v    whether do CPU validation or not (default:1)
         -pr    input data type. int8/fp16/fp32 (representing 8/16/32 bit data) (default:fp16)
          -N    input batch size.  (default:2)
          -C    input channel size. (default:16)
          -H    input height size. (default:1)
          -W    input width size.  (default:16)
  -layout_in    input tensor data layout - NCHW by default (default:NCHW)
 -layout_out    output tensor data layout - NHWC by default  (default:NHWC)
       -seed    seed to be used, -1 means random every time (default:-1)
```
