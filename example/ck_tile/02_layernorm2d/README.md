# Layernorm2D forward

This folder contains example for Layernorm2D forward using `ck_tile` tile-programming implementation.

# Implementation and feature support

## welford online algorithm
We use welfold algorithm to update `mean`/`variance` block by block. For `N <=4096` case we can compute `mean`/`var`/`normalization` within one loop, we call it `one-pass`. For large N case, it is hard to keep `mean`/`var` inside register/LDS and then computation `normalization`, so we need to load input twice, first time to compute `mean`/`var` block-by-block, then load input another time to compute the `normalization`. We call it `two-pass`.

## mean/variance save
In training case the mean/variance need to store out (TBD, not supported yet)

## prenorm/postnorm

![](misc/pnorm.png)

since [prenorm/postnorm](https://arxiv.org/pdf/1906.01787) is quite common in LLM blocks, this example boosts this feature by kernel fusion. Note that `prenorm`/`postnorm` always need to do elementwise-add a `shortcut` before the actual layernorm computation, and optionally store out the result to global. You can use `-fadd=1` to test `pre-add+store`, or `-fadd=2` to test `pre-add` without store out.

## dynamic quantization
we support dynamic quantization for `int8` output, by setting `-fsweep=1` and `-prec_o=int8`. In this case the output will doing a rowwise dynamic quantization like below:
```
# assume output int8, hidden_states is [m, n] shape and in fp16/bf16
# [m, 1]
per_token_amax, _ = torch.max(
     input=torch.abs(hidden_states), 
     dim=-1, 
     keepdim=True
)
per_token_scale = per_token_amax.to(dtype=torch.float32) / 127.0

# quant hidden_states
hidden_states = (hidden_states / per_token_scale).to(dtype=torch.int8)

return hidden_states, per_token_scale
# hidden_states now is int8 will feed to next layer as intput
# per_token_scale will be used as dequant factor later layer
```

## build
```
# in the root of ck_tile
mkdir build && cd build
sh ../script/cmake-ck-dev.sh  ../ <arch>  # you can replace this <arch> to gfx90a, gfx942...
make tile_example_layernorm2d_fwd -j
```
This will result in an executable `build/bin/tile_example_layernorm2d_fwd`

## example
```
args:
          -m    m dimension (default:3328)
          -n    n dimension (default:4096)
     -stride    stride per row, if -1 then equal to n (default:-1)
          -e    epsilon (default:1e-5)
    -save_mv    save mean/variance(invstd) or not. set to 1 in training case (default:0)
          -v    cpu validation or not (default:1)
      -kname    print kernel name or not (default:1)
     -prec_i    input precision (default:fp16)
     -prec_o    output precision, set auto will be the same as input (default:auto)
     -prec_s    output quant scale type, set auto will be the same as input. used when fsweep=1 (default:auto)
       -fadd    fused-add, 0:no fused add, 1:preadd+store, 2:preadd only (default:0)
     -fsweep    fused-sweep, 0:no, 1:fused-dynamic-quant (default:0)
     -warmup    cold iter (default:5)
     -repeat    hot iter (default:20)
```
