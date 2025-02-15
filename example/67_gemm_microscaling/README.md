# GEMM Examples for Microscaling Formats

## example_gemm_mx_fp8

```bash
# arg1: verification (0=no, 1=CPU)
# arg2: initialization (0=no init, 1=integer value, 2=decimal value)
# arg3: time kernel (0=no, 1=yes)
# arg4: verbosity (0=no info, 1=verbose info)
# arg5 to 10: M (16x), N(16x), K(16x), StrideA, StrideB, StrideC
./bin/example_gemm_mx_fp8 1 1 0 1
```

```bash
# Implies: ./bin/example_gemm_mx_fp8 1 2 0 0
./bin/example_gemm_mx_fp8
```