# GEMM Examples for Microscaling Formats

## example_gemm_mx_fp8

Custom verification parameters:
```bash
# arg1: verification (0=no, 1=CPU)
# arg2: initialization (0=constant values, 1=integer values, 2=decimal values)
# arg3: time kernel (0=no, 1=yes)
# arg4: verbosity (0=no info, 1=verbose info)
# arg5 to 10: M(128x), N(128x), K(64x), StrideA, StrideB, StrideC
# arg11: KBatch
./bin/example_gemm_mx_fp8 1 1 0 1
```

Custom tensor shapes:
```bash
./bin/example_gemm_mx_fp8 1 2 1 0 128  128  256 -1 -1 -1 1
```

Default invocation:
```bash
# Implies: ./bin/example_gemm_mx_fp8 1 2 0 0
./bin/example_gemm_mx_fp8
```