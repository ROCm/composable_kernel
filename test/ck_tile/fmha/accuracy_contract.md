# Scheduled FMHA accuracy checks

The forward tests compare decoded FP16/BF16 inputs with an independent FP32
QK, online softmax and PV implementation. The reference does not narrow P to
the device operand type. O and LSE are never replaced with device results.

## Strict diagnostics and dtype contract

`Compare` without an `AccuracyContract` keeps the original strict behavior:
max absolute O error <= 0.015625, minimum row cosine >= 0.99998, mean row cosine
>= 0.99999, and the existing finiteness/LSE checks. With an explicit contract,
`strict_passed` and all cosine metrics are still calculated and reported.
BF16 forward tests gate on the component contract; FP16 forward tests require
both the component contract and the strict check. Historical strict failures
are preserved in the external PR validation evidence.
Both contracts retain the original absolute O error cap of 0.015625. A large
contribution budget cannot override that cap.

The cosine constants originated in the local FMHA tests before this test
migration. They are not an upstream universal BF16 attention accuracy guarantee.
For comparison, the stock FMHA example uses dtype-specific elementwise
limits and a narrowed P/O reference. Passing that check is not the same claim
as passing this independent FP32 comparison.

## Component budget

For each selected output component, the reference also computes
`A[d] = sum_k p[k] * abs(V[k,d])`, with unquantized probabilities and the same
normalization, masks, descales and optional V=0 virtual sink as the O reference.
The additional accumulator uses double precision; it does not change the FP32
O/LSE calculation. Let `r` be FP32 reference O, `K` the number of valid keys,
`Td = ceil(key_length / key_tile_size)`,
`Tr = ceil(key_length / reference.key_chunk_size)`, and `u32 = 2^-24`.
The contract accepts N64/N128. `Compute` records its actual key chunk size so
the budget counts both device and reference online reductions, including host
tests that use smaller chunks. The double A accumulator
uses the FP32 oracle probabilities; it is not an independent FP64 softmax.

The test uses FP16 unit roundoff `u = 2^-11` or BF16 `u = 2^-8`, and
`gamma(n) = n*u32 / (1 - n*u32)`. BF16 `u = 2^-8` assumes round-to-nearest
narrowing of device P and O. The tested gfx1250 path uses ties-to-even
conversion, including the LLVM builtin BF16 cast when enabled. That builtin
path does not consult `CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT`; the effective
converter must be checked in the final build. A path that actually truncates
P or O needs a `u = 2^-7` rounding bound and separate contract validation; it
is outside this tested budget. For the tested FP32 score path:

```text
eta  = max_k (abs(scale) * gamma(Dq) * sum_d abs(Q[d]*K[k,d])
              + u32 * abs(score[k]))
beta = 2*gamma(K) + gamma(3*Td + 8) + gamma(3*Tr + 8)
       + expm1(4*eta) + 4*u32
E    = beta*A + gamma(2)*abs(r) + u*A + floor*K*max_k(abs(V[k,d]))
limit = (1 + u)*E + u*abs(r) + floor
```

The factors of two account for comparing the device accumulation with a
separate FP32 reference, rather than treating either as exact. Score differences
between the two paths can reach `2*eta`; normalization adds another factor of
two in the probability ratio bound, giving `expm1(4*eta)`. The two chunk terms count their separate
online rescale/reduction operations. The final
`(1+u)` retains the cross term from rounding O after P and FP32 error.
`floor` is 2^-25 for FP16 round-to-nearest subnormal spacing, and 2^-126
for the BF16 path that permits flushing subnormals to zero. FP16 subnormal
behavior must be verified for the selected runtime; the floor does not assert
that every hardware/compiler path preserves FP16 subnormals.

A component passes when `abs(actual-r) <= limit` and the original absolute
error cap is satisfied. Nonfinite O always fails.
LSE keeps `atol=0.02`, `rtol=0.002`; -infinity is allowed only on empty rows
without a finite virtual sink. The comparison reports the maximum component
ratio, failing element/row counts, global relative L2, worst row relative L2,
row norm drift, per-head projected gain and the original cosine results.

This is a **tested numerical budget**, not an operation-by-operation proof of
all hardware implementations. In particular, `4*u32` for exp accuracy, folded
log2 scale/FMA differences, and the online reduction allowance are explicit
policy assumptions. The supported validation domain consists of the finite,
bounded decoded inputs used by these tests. Overflow, nonfinite budgets and
invalid metadata fail rather than creating an infinite tolerance. Wide score
ranges and underflow require additional evidence; do not infer their behavior
from the bounded-input tests.
The fixed absolute cap is also specific to this bounded test domain: for
example, BF16 output rounding alone can reach 0.015625 at `abs(O)=4`.
It is not a universal guarantee for arbitrary magnitudes of V or descales.
The formal properties use `abs(V)<=1.75`; the full performance matrix input
range is checked separately before applying this contract.

For FP16, the per-key floor is deliberately conservative even when a
probability is normal. At K=32768 it contributes at most
`32768*2^-25*max(abs(V))` before the final O rounding; this can dominate
the dtype rounding term. FP16 still requires the original strict check.
Conversion/WMMA subnormal handling requires native controls and code-object
evidence; a compiler flush flag alone is insufficient evidence.

## Gain and cancellation controls

A large `A/abs(r)` represents cancellation. Consequently neither a global
relative L2 bound of `uP+uO` nor an averaging-based gain threshold is a universal
property of signed attention inputs. Deterministic rounding may be correlated
across identical rows. Those metrics remain diagnostics and are compared with
same-input stock `qr_tdm` controls in the PR evidence.

The formal generated-dispatch tests run each of their 40 shapes with signed,
constant and nonnegative V, with LSE off/on, for FP16 and BF16: 240 tests per
dtype, 480 total. This covers the existing batch/group, layouts, masks, tails,
virtual sinks and selector boundaries without duplicating kernel geometry.
Constant channels use several exactly representable signed mantissas based on
`{0.75, -1.25, 1.5, -1.75}`, with head variation. Without a virtual sink, exact
arithmetic returns the constant on each nonempty attention row, independent
of QK. The device result is checked against the component budget; in particular,
FP16 subnormal P rounding at extreme score differences can cause a deviation
from the exact constant. Empty rows return zero; a finite virtual sink with V=0
reduces the constant by the probability mass assigned to real keys, which the
independent oracle computes.
For this no-cancellation property, the component budget also detects gain
errors. Several mantissas are needed: BF16(1.01) equals 1.0078125 and an
all-ones V alone would not reject that post-rounded gain.

Gain detection also depends on K and the score budget `eta`. The host
`LargestKeyGainSensitivity` control covers K=32768, Dq=192, N64/N128, with
every Q/K component equal to 0.125 and the signed constant channels above.
It rejects both post-BF16-rounded +/-1% gains. This is a specific property
control, not a guarantee for larger score roundoff or arbitrary QK inputs.
CPU geometry controls with concentrated tail/boundary weights include
inputs where a 1% gain remains inside the component budget. Small missing
tile contributions and mild rescale errors can also remain inside it.
Those measured sensitivity limits are preserved in the external evidence.

Host unit tests verify exact outputs, cancellation with repeated rows,
post-rounded gain, a single over-budget component, the absolute error cap,
zero outputs, NaN, LSE mismatches and invalid contribution metadata.
External CPU fault models additionally check skipped PV contributions, stale
rescale and causal boundary faults. Same-input native captures measure the
actual kernel outputs separately. A CPU fault model pass is not a native
regression test pass.

The contract does not establish end-to-end model quality. Full PR validation,
FP16/BF16 coverage, same-input stock controls, the largest-K fault checks and
reviewer acceptance remain separate gates.
