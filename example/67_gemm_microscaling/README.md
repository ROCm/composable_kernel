# GEMM Examples for Microscaling Formats

## example_gemm_mx_fp8

Custom verification parameters:
```bash
# arg1: verification (0=no, 1=CPU)
# arg2: initialization (0=constant values, 1=integer values, 2=decimal values)
# arg3: time kernel (0=no, 1=yes)
# arg4: verbosity (0=no info, 1=verbose info)
# arg5 to 10: M(256x), N(256x), K(512x), StrideA, StrideB, StrideC
# arg11: KBatch
# arg12: warmup runs pre-timing
# arg13: repeat run count for timing
./bin/example_gemm_mx_fp8 1 1 0 1
```

Custom tensor shapes:
```bash
./bin/example_gemm_mx_fp8 1 2 1 0 256  256  512 -1 -1 -1 1 10 10
```

Default invocation:
```bash
# Implies: ./bin/example_gemm_mx_fp8 1 2 0 0
./bin/example_gemm_mx_fp8
```