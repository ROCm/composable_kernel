# Test Plan: XOR Descriptor Copy-Only Test

## Goal
Test if XOR descriptor works for basic copy operations in Tutorial 10's context

## What We Know
- ✅ Tutorial 11b: XOR + copy distribution → WORKS
- ✅ Tutorial 09: Packed LDS + full GEMM → WORKS
- ✗ Tutorial 10: XOR LDS + full GEMM → FAILS

## Hypothesis
The XOR descriptor works for copying, but something in the GEMM computation logic breaks.

## Test Approach

Create a simplified version of Tutorial 10 that:
1. Loads A from global → stores to XOR-swizzled LDS
2. Loads A from XOR-swizzled LDS → stores back to global
3. Compares with input

Same for B matrix.

If this passes: XOR descriptor is fine, issue is in GEMM logic
If this fails: XOR descriptor has a context-specific bug in Tutorial 10

## Implementation

Modify Tutorial 10's operator() to:
- Skip all GEMM computation
- Just copy A: global → XOR LDS → global
- Just copy B: global → XOR LDS → global
- Verify correctness

This is essentially Tutorial 11b but with Tutorial 10's exact setup (distributions, tile sizes, etc.)

## Next Step After This Test

If copy-only passes but GEMM fails:
- Issue is in how GEMM reads from XOR-swizzled LDS
- OR issue is in GEMM computation with XOR-loaded data
- Need to test partial GEMM (load from XOR, compute, check accumulator)
