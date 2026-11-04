# comp_tdm_v3 bf16 GEMM: correctness check and benchmark

`tile_example_gemm_comp_tdm_v3_bf16_bench` runs `GemmPipelineAgBgCrCompTDMV3` with bf16 A and B,
fp32 accumulation and bf16 C (row-major) on gfx1250, for any M x N x K. Its main use is a quick
correctness check over a list of shapes: one process per shape, with the exit code as the result.

## Build

```bash
cmake -DGPU_TARGETS=gfx1250 ..
make tile_example_gemm_comp_tdm_v3_bf16_bench
```

## Instances

Chosen at run time; each combination is one compiled kernel.

| Option | Values | Default |
|---|---|---|
| `-layout` | `rcr`, `rrr`, `crr`, `ccr` (A, B; C is row-major) | `rcr` |
| `-epilogue` | `tdm`, `cshuffle` | `tdm` |
| `-tile` | `256x256x64` (16 waves, 4x4x1), `128x128x64` (4 waves, 2x2x1) | `256x256x64` |
| `-vector_c` | `auto`, `8`, `2`, `1` (CShuffle C store vector) | `auto` |
| `-k_batch` | split K into this many ranges (CShuffle only) | `1` |

All instances use the warp tile 16x16x32, a double LDS buffer and the default policy. TDM clips
every A/B box to the real extent, so no instance needs M, N or K to be a multiple of the tile.

- `tdm`: TdmEpilogue writes C with one TDM store that clips to the real M x N, so it serves every
  shape.
- `cshuffle`: CShuffleEpilogue with kPadM and kPadN, storing C in vectors of `-vector_c`
  elements, which must divide N. `auto` picks the widest of 8, 2 and 1 that divides N, so CShuffle
  also serves every shape; a forced width that does not divide N is rejected (exit code 2).
- `-k_batch` > 1 (split-K): the K ranges run as separate workgroups and each adds its bf16
  partial C into C atomically, so C is zeroed before every launch (inside the timed loop) and the
  result depends on the order of the adds. Only CShuffle can do this, with a C vector of at least 2
  (N even under `auto`): TdmEpilogue overwrites C, so it rejects any `-k_batch` > 1 (exit code 2).

## Modes

| `-mode` | What it does |
|---|---|
| `verify-int` (default) | Integers in [-5, 5]; C must equal the exact sum rounded once to bf16, bit for bit (exact for K up to 671088). |
| `verify-rand` | normal(0, `-sigma`) rounded to bf16; checked against a double reference with a bf16 and fp32-accumulation tolerance. |
| `verify-sparse` | About sqrt(64/K) of the elements nonzero, each -2, -1, 1 or 2, so sum\|a*b\| stays below 256 at almost every output: every partial and running sum is exact in bf16, so those outputs must be exact at any `-k_batch`. |
| `peak` | `-rounds` samples, each after `-gap_ms` idle and `-warmup` launches, timing `-repeat` launches. |
| `sustained` | 50 warmup, 100 timed launches, no gap, one sample. |

The verify modes launch the kernel `-verify_launches` times and require identical outputs
(`-k_batch` 1). With `-k_batch` > 1, `verify-int` and `verify-rand` (and the rare sparse output
above 256) allow each split's bf16 rounding: |got - ref| <= |ref| * 2^-7 + k_batch * sum|a*b| * 2^-7. The
reference checks every output when M*N*K <= 2^33, and otherwise a sample of points in every
256x256 output tile plus its corners (`-check`, `-check_per_tile`).

## Output and exit code

One JSON line per result: `verify`, `sample` / `summary` (timed modes) or `reject`.

| Exit code | Meaning |
|---|---|
| 0 | pass |
| 1 | wrong result |
| 2 | rejected by `IsSupportedArgument`, or bad arguments |
| 3 | error |

## Examples

```bash
B=./bin/tile_example_gemm_comp_tdm_v3_bf16_bench
$B                                                   # 4096^3, RCR, verify-int
$B -m=333 -n=517 -k=1000 -layout=crr -mode=verify-rand
$B -mode=peak                                        # 4096^3 timing
$B -m=192 -n=448 -k=524288 -layout=crr -tile=128x128x64 -epilogue=cshuffle -k_batch=16 \
   -mode=verify-sparse                                # split-K, exact check

# a shape list, one "M N K layout" per line
while read m n k l; do
  $B -m=$m -n=$n -k=$k -layout=$l -mode=verify-int > /dev/null; echo "$m $n $k $l rc=$?"
done < shapes.txt
```
