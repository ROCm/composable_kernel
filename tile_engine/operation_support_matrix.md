**CK Tile operation support by data type, layout, and GPU target:**

| Op | CK Tile Kernel | fp16 | fp8 | bf16 | bf8 | int8 | fp4 | fp6 | rcr | rrr | ccr | crr | 90a | 942 | 950 | 1201 |
|:--:|----------------|:----:|:---:|:----:|:---:|:----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:----:|
| GEMM | gemm_universal [1][2]<br>engine: gemm_universal/<br>example: 03_gemm/ | ✅ | ✅ | ✅ | ✅ | ❌ | | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| GEMM | gemm_multi_d [3]<br>engine: gemm_multi_d/<br>example: 19_gemm_multi_d/ | ✅ | | | | | | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| GEMM | gemm_preshuffle [4]<br>engine: gemm_preshuffle/ | ✅ | ✅ | ✅ | ✅ | | | | ✅ | | | | ✅ | ✅ | ✅ | ❌ |
| GEMM | streamk_gemm [5][6][7]<br>engine: gemm_streamk/<br>example: 40_streamk_gemm/ | ✅ | ✅ | ❌ | ❌ | | | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| GEMM | batched_gemm<br>example: 16_batched_gemm/ | ❌ | | | | | | | ❌ | | | | ❌ | ❌ | ❌ | ❌ |
| GEMM | batched_contraction<br>example: 41_batched_contraction/ | ❌ | | | | | | | ❌ | | | | ❌ | ❌ | ❌ | ❌ |
| GEMM | block_scale_gemm<br>example: 38_block_scale_gemm/ | | ❌ | ❌ | ❌ | | ❌ | | ❌ | | | | ❌ | ❌ | ❌ | ❌ |
| GEMM | flatmm<br>example: 18_flatmm/ | ❌ | ❌ | ❌ | ❌ | | ❌ | ❌ | ❌ | | | | ❌ | ❌ | ❌ | ❌ |
| GEMM | gemm_multi_abd<br>example: 22_gemm_multi_abd/ | ❌ | | | | | | | ❌ | | | | ❌ | ❌ | ❌ | ❌ |
| GEMM | gemm_quant | | ❌ | | ❌ | | | | ❌ | | | | ❌ | ❌ | ❌ | ❌ |
| GEMM | grouped_gemm<br>example: 17_grouped_gemm/ | ❌ | ❌ | ❌ | | | | | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| GEMM | grouped_gemm_quant | | ❌ | | ❌ | | | | ❌ | | | | ❌ | ❌ | ❌ | ❌ |
| Reduce | multi_reduce2d [8]<br>engine: reduce/<br>example: 05_reduce/ | ✅ | | ❌ | | | | | | | | | ❌ | ✅ | ✅ | ❌ |
| Reduce | reduce2d<br>example: 05_reduce/ | ❌ | | ❌ | | | | | | | | | ❌ | ❌ | ❌ | ❌ |
| Attention | fmha<br>example: 01_fmha/ | ❌ | ❌ | ❌ | ❌ | | | | | | | | ❌ | ❌ | ❌ | ❌ |
| Attention | sparse_attn<br>example: 50_sparse_attn/ | ❌ | | ❌ | | | | | | | | | ❌ | ❌ | ❌ | ❌ |
| Activation | softmax | ❌ | | ❌ | | | | | | | | | ❌ | ❌ | ❌ | ❌ |
| Activation | topk_softmax<br>example: 09_topk_softmax/ | ❌ | | ❌ | | | | | | | | | ❌ | ❌ | ❌ | ❌ |
| Conv | grouped_conv<br>example: 20_grouped_convolution/ | ❌ | | ❌ | | | | | | | | | ❌ | ❌ | ❌ | ❌ |
| Data Move | batched_transpose<br>example: 35_batched_transpose/ | ❌ | ❌ | ❌ | | | | | | | | | ❌ | ❌ | ❌ | ❌ |
| Data Move | image_to_column<br>example: 04_img2col/ | ❌ | | ❌ | | | | | | | | | ❌ | ❌ | ❌ | ❌ |
| Data Move | permute<br>example: 06_permute/ | ❌ | ❌ | | | | | | | | | | ❌ | ❌ | ❌ | ❌ |
| Elementwise | elementwise<br>example: 21_elementwise/ | ❌ | | ❌ | | | | | | | | | ❌ | ❌ | ❌ | ❌ |
| MoE | fused_moe<br>example: 15_fused_moe/ | ❌ | | ❌ | | | | | | | | | ❌ | ❌ | ❌ | ❌ |
| Norm | add_rmsnorm2d_rdquant<br>example: 11_add_rmsnorm2d_rdquant/ | ❌ | | ❌ | | | | | | | | | ❌ | ❌ | ❌ | ❌ |
| Norm | layernorm2d<br>example: 02_layernorm2d/ | ❌ | | ❌ | | | | | | | | | ❌ | ❌ | ❌ | ❌ |
| Norm | norm_reduce | ❌ | | ❌ | | | | | | | | | ❌ | ❌ | ❌ | ❌ |
| Norm | rmsnorm2d<br>example: 10_rmsnorm2d/ | ❌ | | ❌ | | | | | | | | | ❌ | ❌ | ❌ | ❌ |
| Pooling | pooling<br>example: 36_pooling/ | ❌ | | | | | | | | | | | ❌ | ❌ | ❌ | ❌ |
| Quant | smoothquant<br>example: 12_smoothquant/ | ❌ | | ❌ | | | | | | | | | ❌ | ❌ | ❌ | ❌ |

**Legend:**
- **CK Tile Kernel column:** First line is the kernel name. Lines prefixed with "engine:" show the tile engine directory under `ops/`. Lines prefixed with "example:" show the CK Tile example directory under `example/ck_tile/`.
- **Green cell** (✅): CK Tile implementation exists **and** the tile engine supports it.
- **Red cell** (❌): CK Tile implementation exists **but** the tile engine does **not** support it.
- **Grey cell** (blank): No CK Tile implementation exists for this combination.

**Notes:**
- All CK Tile GEMM and reduce kernels are architecturally generic (no compile-time GPU guards). The gfx filtering in the tile engine is a validation/testing scope decision, not a code limitation.
- [1] **gemm_universal:** CMake defaults to `fp8;fp16`. Building bf16/bf8 requires `-DGEMM_UNIVERSAL_DATATYPE="fp16;fp8;bf16;bf8"`.
- [2] **gemm_universal:** CK Tile supports int8 GEMM (with int32 output) but the tile engine has no int8 configuration.
- [3] **gemm_multi_d:** CK Tile kernel is type-generic but example and tile engine are fp16-only. Adding other types requires new tile engine configurations.
- [4] **gemm_preshuffle:** Only supports rcr layout (A=row, B=column, C=row) due to the pre-shuffle data format requirement.
- [5] **streamk_gemm:** CK Tile supports bf16 and bf8 for streamk, but the tile engine has no default tile configs for them.
- [6] **streamk_gemm:** Builder and default configs support all 4 layouts, but CMake defaults to `rcr` only. Building others requires `-DGEMM_STREAMK_LAYOUT="rcr;rrr;ccr;crr"`.
- [7] **streamk_gemm:** CK Tile kernels have no arch-specific guards; tile engine filtering is pending validation for gfx950/gfx1201.
- [8] **multi_reduce2d:** CK Tile's reduce example supports bf16 input but the tile engine only configures fp16. The reduce kernel adapts to wave32/wave64 at runtime via `is_wave32()`.
- Reduce operations do not use matrix layouts.

**Layout codes:** Each layout code specifies the memory layout of tensors A, B, and C as row-major (r) or column-major (c). For example, `rcr` means A is row-major, B is column-major, and C is row-major. For gemm_multi_d, the instance builder uses 4-character codes (e.g., `rcrr`) where the 4th character specifies the D tensor layout; in the table above, the 3-character A/B/C portion is shown since the D layout is always row-major (r) for all supported configurations.

**Data type mapping:** The column labels (fp16, fp8, bf16, bf8, int8, fp4, fp6) refer to the input configuration label passed to the tile engine or CK Tile example. Each label determines the actual types used for the source tensors (A, B), accumulator, and output tensor (C). For 16-bit and 8-bit float types, A and B use the label type, the accumulator is fp32, and the output type C matches the input type for fp16 and bf16 but is promoted to fp16 for fp8 and bf8 since 8-bit precision is insufficient for output storage. int8 uses int32 for both accumulation and output. fp4 is a mixed-precision weight type where B is fp4 and A uses the activation type (fp16 or bf16). fp6 is used by the microscaling (MX) flatmm pipeline where both A and B are fp6 with fp32 accumulation and fp32 output.

**Data type mapping per config label:**

| Config Label | A (source) | B (source) | Acc | C (output) |
|:------------:|:----------:|:----------:|:---:|:----------:|
| fp16 | fp16 | fp16 | fp32 | fp16 |
| bf16 | bf16 | bf16 | fp32 | bf16 |
| int8 | int8 | int8 | int32 | int32 |
| fp8 | fp8 | fp8 | fp32 | fp16 |
| bf8 | bf8 | bf8 | fp32 | fp16 |
| fp6 | fp6 | fp6 | fp32 | fp32 |
| fp4 | fp16 or bf16 | fp4 | fp32 | fp16 or bf16 |

For gemm_multi_d, the D tensors (D0, D1) use the same type as the config label (fp16).
