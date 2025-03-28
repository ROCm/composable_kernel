# HSTU attention operator

  HSTU-attention operator is an operator which takes tensor `q: [batches, seqlen, nhead, hdim_qk]`,  `k: [batches, seqlen, nhead, hdim_qk`, 
  `v: [batches, seqlen, nhead, hdim_v]` and some parameters for defining the functional masking as inputs, and do the following:

   * Multiply `q: [batches, seqlen, nhead, hdim_qk]` with `k: [batches, seqlen, nhead, hdim_k]` to get temporary tensor `s: [batches, nhead, seqlen, seqlen]`
   * Update `s` by filtering its values according to a special functional mask, which includes the logics of lower-triangular and diagonal window causal mask
     as well assequence mask
   * Do element-wise SiLu on the `lower seqlen` dimension of `s` to get temporary tensor `p: [batches, nhead, seqlen, seqlen]`
   * Multiply `p : [batches, nhead, seqlen, seqlen]` with `v: [batches, seqlen, nhead, hdim_v]` to get final output `o: [batches, seqlen_q, nhead, headsz_v]` 
   * Jagged inputs are also supported, where each batch has separate seqlen defined by the `sequence_offsets[]`
  

## implementation

   The operator is implemented using a fused kernel in the example:

   *  Tensor S and Tensor P only exist in VGPRs as per-workgroup tiles, no global memory access is needed

## build

   ``` bash
   #> mkdir build
   #> cd build
   #> ../script/cmake-ck-dev.sh .. gfx942              ; use #> rocminfo |grep "gfx"   to check your gpu arch
   #> make -j tile_example_hstu_attention
   ```

## test/verify

   ``` bash
   #>  build/bin/tile_example_hstu_attention -v=1 -prec=fp16 -b=10 -nidx=9 -nhead=4 -hsizeq=64 -hsizev=64 -seqq=13 -seqk=512 -init=u -seed=123 -perf=0 -maskmax=0
   #>  . example/ck_tile/07_hstu_attention/test_hstu_attention.sh
   ```

   Check the example file `example_hstu_attention.cpp` for an understanding of the command-line arguments. Which is like the following:

  ``` C++
    arg_parser.insert("v", "1", "weather do CPU validation or not")
        .insert("prec", "fp16", "data type. fp16/bf16")
        .insert("b", "12", "batch size")
        .insert("nidx", "9", "number of indices for accessing the batches")
        .insert("nhead", "4", "number of heads")
        .insert("hsizeq", "64", "headdim size of Q/K")
        .insert("hsizev", "64", "headdim size of V/O")
        .insert("seqq", "13", "length of the sequence dimension of query tensor")
        .insert("seqv", "1024", "length of the sequence dimension of key tensor")
        .insert("init", "u", "init method for input tensor values, u, uniform random float values, n, normalized random float values")
        .insert("seed", "13579", "seed by the uniform or normal distribution generator")
        .insert("perf", "0", "weather measure execution time or not")
        .insert("maskmax", "0", "used to set mask values to random [0, maskmax), maskmax should in [0, 128], 0 means set all values to 1");
  ```

