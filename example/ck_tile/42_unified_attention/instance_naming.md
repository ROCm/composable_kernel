| Suffix | Meaning |
|--------|---------|
| d64 | head dim 64 |
| bf16 / fp16 | dtype |
| mask | IsMasking=true |
| gqa8 | 8 queries per KV |
| bs32 | page block size 32 (BlockSize=32) |
| decode_t | tiny decode tier (unified_attention_decode_tiny_kernel_traits, kBlockM=16, kBlockQ=2) |
| _local | sliding-window / SWA (IsLocal=true) |
