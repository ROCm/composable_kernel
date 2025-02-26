# Architectures
Kernel: Define tensor view, tile window and the range of each block
pipeline policy:  Define LDS usage and thread mapping
pipeline: real implementation

## Pipeline
The pipeline folder contains different pipelines, each represting a distince algorithm or strategy, ex:
include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_nwarp_sshuffle_qr_ks_vs.hpp

nwarp_sshuffle means 
