# Architecture Components
Kernel: Define tensor view, tile window and the range of each block
pipeline policy:  Define LDS usage and thread mapping
pipeline: Contains concrete implementations of attention. Each pipeline combines kernel components with a specific policy to create executable solutions.

## Pipeline
The pipeline/ directory contains specialized implementations of attention algorithms, differentiated by their optimization strategies. Filename components indicate key implementation choices:
include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_nwarp_sshuffle_qr_ks_vs.hpp





nwarp_sshuffle means 
