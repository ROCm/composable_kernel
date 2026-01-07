# Builder profiler for Convolution

This directory contains the profiler created with builder for CK Tile.


## Overview

Instances are generated using `generate_instances.py`. This script is called with cmake files generation. Interface is the same as for ckProfiler. Example of the usage:
```bash
# arg1: tensor operation (grouped_conv_fwd : Grouped Convolution Forward)
# arg2: data type (0: Input fp32, Weight fp32, Output fp32
#                  1: Input fp16, Weight fp16, Output fp16
#                  2: Input bf16, Weight bf16, Output bf16
#                  3: Input int8, Weight int8, Output int8
#                  4: Input fp8, Weight fp8, Output fp8
#                  5: Input bf8, Weight bf8, Output fp8
#                  6: Input fp8, Weight bf8, Output fp8
#                  7: Input bf8, Weight fp8, Output fp8
#                  8: Input fp32, Weight fp32, Output fp32, Compute tf32)
# arg3: tensor layout (0: Input[G, N, Hi, Wi, C], Weight[G, K, Y, X, C], Output[G, N, Ho, Wo, K]
#                      1: Input[N, Hi, Wi, G, C], Weight[G, K, Y, X, C], Output[N, Ho, Wo, G, K]
#                      2: Input[N, G, C, Hi, Wi], Weight[G, K, Y, X, C], Output[N, G, K, Ho, Wo]
#                      3: Input[N, G, C, Hi, Wi], Weight[G, K, C, Y, X], Output[N, G, K, Ho, Wo])
# arg4: indexing data type (0: 32-bit, 1: 64-bit)
# arg5: verification (0: no, 1: yes)
# arg6: initialization (0: no init, 1: integer value, 2: decimal value)
# arg7: print tensor value (0: no; 1: yes)
# arg8: time kernel (0: no, 1: yes)
# Following arguments (depending on number of spatial dims):
#   Number of spatial dimensions (1=Conv1d, 2=Conv2d, 3=Conv3d)
#   G, N, K, C, 
#   <filter spatial dimensions>, (ie Y, X for 2D)
#   <input image spatial dimensions>, (ie Hi, Wi for 2D)
#   <strides>, (ie Sy, Sx for 2D)
#   <dilations>, (ie Dy, Dx for 2D)
#   <left padding>, (ie LeftPy, LeftPx for 2D)
#   <right padding>, (ie RightPy, RightPx for 2D)

 ################                             op   datatype  layout  indexing  verify  init  log  time  Ndims  G  N   K   C  Y  X  Hi  Wi  Sy  Sx  Dy  Dx  LeftPy  LeftPx  RightPy  RightPx
./bin/profile_ckb_tile_conv_fwd grouped_conv_fwd          1       0         0       1     1    0     1      2 32  4 192 192  3  3  28  28   1   1   1   1       1       1        1        1

```
