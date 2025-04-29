for file in gemm_universal_*; do mv "$file" "${file/f16_f16_f16/fp16_fp16_fp16}"; done
