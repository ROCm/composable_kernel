# fused multi-head attention

This folder contains example for fmha(fused multi-head attention) using ck tile-programming implementation. It is a good example to demonstrate the usage of tile-programming API, as well as illustrate the new approach to construct a kernel template and instantiate it(them) while keeping compile time fast.

## build
```
# in the root of ck
mkdir build && cd build
sh ../script/cmake-ck-dev.sh  ../ <arch>  # you can replace this <arch> to gfx90a, gfx942...
make example_fmha_fwd -j
```
This will result in an executable `build/bin/example_fmha_fwd`

## kernel
The kernel template is `fmha_fwd_kernel.hpp`, this is the grid-wise op in old ck's terminology. We put it here purposely, to demonstrate one can construct a kernel by using various internal component from ck. We may still have an implementation under ck's include path (in the future) for the kernel template.

There are 3 template parameters for this kernel template.
* `TilePartitioner` is used to map the workgroup to corresponding tile, `fmha_fwd_tile_partitioner.hpp` in this folder served as this purpose.
* `FmhaPipeline` is one of the block_tile_pipeline(under `include/ck/tile_program/block_tile_pipeline`) which is a performance critical component. Indeed, we did a lot of optimization and trials to optimize the pipeline and may still workout more performance pipeline and update into that folder. People only need to replace this pipeline type and would be able to enjoy the benefit of different performant implementations (stay tuned for updated pipeline(s)).
* `EpiloguePipeline` will modify and store out the result in the last phase. People usually will do lot of post-fusion at this stage, so we also abstract this concept. Currently we didn't do much thing at the epilogue stage but leave the room for future possible support.

## codegen
To speed up compile time, we instantiate the kernels into separate file. In this way we can benefit from parallel building from CMake/Make system. This is achieved by `generate.py` script. Besides, you can look into this script to learn how to instantiate a kernel instance step by step, which is described in `FMHA_FWD_KERNEL_BODY` variable.

## executable
`example_fmha_fwd` is the example executable, implemented in `fmha_fwd.cpp`. You can type `./bin/example_fmha_fwd -?` to list all supported args
```
args:
          -v    weather do CPU validation or not (default:1)
       -mode    kernel mode. 0:batch, 1:group (default:0)
          -b    batch size (default:2)
          -h    num of head, for q (default:8)
        -h_k    num of head, for k/v, 0 means equal to h (default:0)
                 if not equal to h, then this is GQA/MQA case
          -s    seqlen_q (default:3328)
        -s_k    seqlen_k, negative value means equal to s (default:-1)
          -d    head dim for q, k (default:128)
        -d_v    head dim for v, 0 means equal to d (default:0)
      -scale    scale factor. 0 means equal to 1/sqrt(seqlen) (default:0)
      -iperm    permute input (default:1)
                 if true, will be b*h*s*d, else b*s*h*d
      -operm    permute output (default:1)
       -bias    add bias or not (default:0)
       -prec    data type. fp16/bf16/fp8/bf8 (default:fp16)
       -mask    0: no mask, 1: top-left, 2:bottom-right (default:0)
                 't:l,r', top-left local-attn with left right size
                 'b:l,r', bottom-r local-attn with left right size
                 'g:y,x', generic attention mask coordinate with y/x size
    -vlayout    r for row-major(seqlen*hdim), c for col-major(hdim*seqlen) (default:r)
        -lse    0 not store lse, 1 store lse (default:0)
       -init    init method. 0:random int, 1:random float, 2:trig float (default:1)
```
Example: `./bin/example_fmha_fwd -b=1 -h=16 -s=16384 -d=128` will run a fmha case with batch=1, nhead=16, sequence length=16384, hdim=128, fp16 case.

## support features
Currently we are still in rapid development stage, so more features/optimizations will be coming soon.

### hdim
Currently we support `32/64/128/256` hdim for `fp16`/`bf16`, within which `64`/`128` is better optimized. We may consider optimize other hdim performance if have more request. We also have an experimental support for arbitrary hdim(even odd number), one can change the return value of `get_pad()` inside `generate.py` to achieve this. (Note: we may change the method or optimize arbitrary hdim support in the future)

### group/batch mode
Currently we support both batch and group mode, by setting `-mode` = `0` or `1`, where in group mode we support each batch can have different seqlen

### MQA/GQA
By setting `-h`(nhead for q) and `-h_k`(nhead for k/v) with different number, you can achieve MQA/GQA. Please pay attention that `h % h_K == 0` when you set different numbers.

### input/output permute, and `b*s*3*h*d`
If you look at the kernel argument inside `fmha_fwd_kernel.hpp`, we support providing arbitrary stride for seqlen(stride_q/k/v), nhead, batch of q/k/v matrix, hence it is very flexible to support `b*h*s*d` or `b*s*h*d` input/output permute. The `-iperm=0/1`, `-operm=0/1` is a convenient way to achieve this through the executable. We didn't provide a command-line arg to test `b*s*3*h*d` layout which is by default used by torch/FA, but it's trivial to achieve this if one set the proper `stride_q/k/v` value as `3*h*d`.

### attention bias
Attention bias is supported with the layout of `b*h*s*s` and bias value in float number.

### lse
For training kernels, "log sum exp" need to store out in forward and used in backward. We support this by setting `-lse=1`

### vlayout
We support v matrix in both row-major(`seqlen*hdim`) and col-major(`hdim*seqlen`). Since the accumulate(reduce) dimension for V is along `seqlen`, for current AMD's mfma layout which expect each thread to have contiguous register holding pixels along reduce dimension, it's easier to support col-major V layout. However, the performance of col-major is not necessarily faster than row-major, there are many factors that may affect the overall performance. We still provide the `-vlayout=r/c` here to switch/test between different layouts.

### generic attention mask coordinate
We unify the mask expression into generic attention mask coordinate, providing an uniformed approach to describe causal top-left, causal bottom-right, local attention.
![](misc/gamc.png)

(more description to be added)

### dropout
TBD

## FP8 experimental support
As described in [this blog](https://blog.hippoml.com/8bit-hippoattention-up-to-3x-faster-compared-to-flashattentionv2-8f9def90b482), we have an experimental support for fp8 fmha kernels, you can evaluate the performance by setting the arg `-prec=fp8` to the `example_fmha_fwd`, on a gfx940/941/942 machine and ROCm 6.0+. Currently if you not explicitly setting `-v=0`(which will disable CPU verification), it will printout an error as much as `0.05`. We are still WIP to tune the kernel performance as well as the precision, so stay tuned for the updated performance(pipeline)
Currently we only support `-vlayout=c` for fp8, which is `hdim*seqlen` for V matrix. row major for V matrix support will come later.
