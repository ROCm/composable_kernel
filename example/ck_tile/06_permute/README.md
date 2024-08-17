# permute

This folder contains example for permute kernel, which is similiar to [torch.permute](https://pytorch.org/docs/stable/generated/torch.permute.html) (combined with [torch.contiguous](https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html)). Currently we implement a generic permute kernel that support up to rank 8 arbitrary permutation with a single kernel instance. Performance is not the first consideration, we prefer a simple and general kernel implementation using `ck_tile` in this example.


```
args:
          -v    weather do CPU validation or not (default:1)
       -prec    data type. fp16/bf16/fp32 (default:fp16)
      -shape    the shape of the input tensor (default:2,3,4)
       -perm    permute perm (default:2,1,0)
```

## build
```
# in the root of ck_tile
mkdir build && cd build
sh ../script/cmake-ck-dev.sh  ../ <arch>  # you can replace this <arch> to gfx90a, gfx942...
make tile_example_permute -j
```
This will result in an executable `build/bin/tile_example_permute`


## some examples
```
# torch
x=torch.tensor.randn(2,3,4,6)
y=x.permute(0,3,2,1).contiguous()

# ck_tile
./build/bin/tile_example_permute -shape=2,3,4,6 -perm=0,3,2,1
```

or you can try the smoke_test
```
# in the root of ck_tile, after you build this example
sh example/ck_tile/06_permute/script/smoke_test.sh
```
