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
   #>  build/bin/tile_example_hstu_attention -v=1 -prec=bf16 -b=10 -jagged=1 -nhead=4 -hdim_qk=128 -hdim_v=128 -seqlen=750,730,733,860,870,788,760,821,833,779 -targets=5,5,6,6,5,6,5,6,4,6
       -causal=1 -local_len=5 -context_len=6 -minfull_len=6 
   #>  . example/ck_tile/07_hstu_attention/test_hstu_attention.sh
   ```

   Check the example file `example_hstu_attention.cpp` for an understanding of the command-line arguments. Which is like the following:

  ``` C++
    arg_parser.insert("v", "1", "weather do CPU validation or not")
        .insert("prec", "fp16", "data type. fp16/bf16")
        .insert("jagged", "0", "q/k/v batched sequence is jagged or not")
        .insert("b", "12", "batch size")
        .insert("nhead", "4", "number of heads")
        .insert("hdim_qk", "64", "headdim size of Q/K")
        .insert("hdim_v", "64", "headdim size of V/O")
        .insert("seqlen", "400", "seqlen of single or all batches for query and key/value tensor")
        .insert("targets", "16", "sequence length at the end of query/key token sequence that should be excluded from attention")
        .insert("causal", "1", "enable causal mask or not")
        .insert("local_len", "5", "length of the diagonal window for enabling masking, value 0 to disable")
        .insert("context_len", "6", "sequence length at the begin of the query sequence the should be included for attention")
        .insert("minfull_len", "6", "sequence length at the end of the query sequence that should be included for attention")
        .insert("seed", "13579", "seed by the uniform or normal distribution generator")
        .insert("perf", "0", "weather measure execution time or not");
  ```

