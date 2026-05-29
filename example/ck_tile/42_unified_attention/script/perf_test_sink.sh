#!/bin/bash
# perf_test_sink.sh — perf gate that locks in the "zero-overhead sinks"
# contract.
#
# Two regressions this script catches:
#
#   1. The no-sink instance's runtime growing past `BASELINE_TOLERANCE`
#      (default 5%) of the captured baseline. This fires on
#      catastrophic regressions of the kHasSink=false code path:
#      barriers added in the K/V loop, extra HBM round-trips per tile,
#      tile-storage allocated inside the iteration, register-spill
#      cliffs from an over-grown pipeline operator(), etc.
#      → time(no-sink) / BASELINE_NO_SINK must be ≤ BASELINE_TOLERANCE.
#
#   2. The sink-aware path adding more than `SINK_OVERHEAD_MAX` (default
#      10%) over the no-sink leg on the same shape. Same kinds of
#      catastrophic regressions, but on the kHasSink=true path
#      specifically — e.g. someone moves the per-row sink-init sweep
#      inside the K/V loop, or makes `sink_ptr_pre_offset[r %
#      num_qpkv]` a non-coalesced indirection.
#      → time(sink) / time(no-sink) must be ≤ SINK_OVERHEAD_MAX.
#
# Both gates run on the two prefill shapes that mirror `perf_test_swa.sh`'s
# coverage:
#
#   * d=128 MHA prefill        (-d=128 -h_k=8 -nqpkv=1)   q = kv = 8192
#   * d=64  GQA-8 prefill      (-d=64  -h_k=1 -nqpkv=8)   q = kv = 8192
#
# Baselines were captured on MI355 (HIP_VISIBLE_DEVICES=6) right after
# the kernel-side sink path landed. The procedure was:
#
#   for shape in {d128_MHA, d64_GQA8}:
#     for i in 1..3:
#       run -mask=b -verify=0 -warmup=10 -repeat=30 -varlen=0
#       extract "X.XX ms" from the bench line
#     baseline = mean of the 3 runs
#
# Observed numbers (ms) on MI355:
#
#   d128 MHA no-sink : 0.4573, 0.4606, 0.4610     mean ≈ 0.460
#   d64  GQA-8 no-sink: 0.3370, 0.3388, 0.3381     mean ≈ 0.338
#
# Both shapes show sink-on overhead of <2.5% on the same MI355, so the
# 10% SINK_OVERHEAD gate has ~4× headroom over the observed noise floor.
#
# What this gate does NOT catch — and why we are OK with that:
#
#   The natural "branch-leak" regression mode (someone replaces
#   `if constexpr (kHasSink)` with something that lets the per-row
#   sweep run on no-sink instances) is empirically below the gate's
#   noise floor on prefill. The sink init's cost is ≈ 1 fp32 division
#   per thread (kBlockM=256, ~256 threads/CTA → ~1 row/thread); the
#   K/V loop runs ~10^6 cycles. Forced sanity provocations on MI355:
#
#     * `if constexpr(kHasSink)` → `if constexpr(true)` plus a null-guard
#       on `sink_ptr_pre_offset`:               no-sink stayed at 0.454 ms
#                                              (1.3% under baseline) ❌
#     * 100× redundant sweep in the lambda:    no-sink crept to 0.350 ms
#       (compiler dead-coded most writes)      (3.5% over baseline) ❌
#     * 50× `block_sync_lds()` before the
#       set_tile:                              no-sink at 0.461 ms
#                                              (0.2% over baseline) ❌
#
#   In all three cases the gate stays GREEN. That's because the gate
#   is calibrated against the *kernel wall time*, not the init phase
#   in isolation, and the init phase is fundamentally < 1 µs out of
#   ~460 µs. The gate's actual value is catching regressions in the
#   one place that dominates wall time: the K/V loop. The "did the
#   if-constexpr collapse correctly" check belongs to a static / SASS
#   diff, not a runtime gate. We accept that scope.
#
# Run with HIP_VISIBLE_DEVICES set to your assigned GPU (defaults to 6
# on the shared MI355 dev node). Build with
# `ninja -j 50 tile_example_unified_attention` from the CK build dir
# (do not use `cmake --build`; see Sink-implementation-steps.md hard
# constraints). Exit code is the number of failed assertions across
# both shapes (0 = all PASS).

set -uo pipefail

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-6}"

EXE_NAME=tile_example_unified_attention
EXE="${EXE:-$(find . -name "$EXE_NAME" -type f -executable 2>/dev/null | head -n 1)}"
if [ -z "${EXE:-}" ] || [ ! -x "$EXE" ]; then
    echo "ERROR: $EXE_NAME not found. Set EXE=/path/to/$EXE_NAME or run from build dir." >&2
    exit 2
fi
echo "Using EXE=$EXE"
echo "Using HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"

# Allow override via env vars; defaults match the perf plan thresholds.
BASELINE_TOLERANCE="${BASELINE_TOLERANCE:-1.05}"  # no-sink within 5% of baseline
SINK_OVERHEAD_MAX="${SINK_OVERHEAD_MAX:-1.10}"    # sink-on within 10% of no-sink

# Bench-only config — same shape as perf_test_swa.sh's so the two
# scripts can share GPU-warmup state when run back-to-back.
COMMON="-prec=bf16 -seed=17 -verify=0 -warmup=10 -repeat=30 -varlen=0 -nb=512"

# Each row is "NAME|BASELINE_MS|EXTRA_ARGS". BASELINE_MS is the
# captured no-sink runtime (see procedure above). Adding new shapes:
# capture the baseline with `EXE COMMON SHAPE_ARGS -mask=b` averaged
# over 3 runs and paste the mean here.
SHAPES=(
    "prefill_d128 MHA   q=kv=8192 |0.460 |-page_blk_size=128 -d=128 -h_k=8 -nqpkv=1 -b=1 -s=8192 -s_k=8192 -query_lens=8192 -kv_lens=8192"
    "prefill_d64  GQA-8 q=kv=8192 |0.338 |-page_blk_size=128 -d=64  -h_k=1 -nqpkv=8 -b=1 -s=8192 -s_k=8192 -query_lens=8192 -kv_lens=8192"
)

# Extract "X.XXX ms" from a benchmark output line. Same helper as
# perf_test_swa.sh so the two scripts stay in lockstep.
extract_ms() {
    grep -oE '[0-9]+\.[0-9]+ ms' "$1" | head -1 | awk '{print $1}'
}

n_pass=0
n_fail=0

for row in "${SHAPES[@]}"; do
    name="${row%%|*}"
    rest="${row#*|}"
    baseline="${rest%%|*}"
    baseline="${baseline// /}"
    args="${rest#*|}"

    printf '== %s\n' "$name"

    nosink_log=$(mktemp)
    sink_log=$(mktemp)
    "$EXE" $COMMON $args -mask=b                  > "$nosink_log" 2>&1 || true
    "$EXE" $COMMON $args -mask=b -sink=random:17  > "$sink_log"   2>&1 || true

    t_nosink=$(extract_ms "$nosink_log")
    t_sink=$(extract_ms "$sink_log")

    if [ -z "$t_nosink" ] || [ -z "$t_sink" ]; then
        echo "   FAIL: could not parse timing"
        echo "   no-sink tail:"
        tail -3 "$nosink_log" | sed 's/^/      /'
        echo "   sink tail:"
        tail -3 "$sink_log"   | sed 's/^/      /'
        n_fail=$((n_fail + 1))
        rm -f "$nosink_log" "$sink_log"
        continue
    fi
    rm -f "$nosink_log" "$sink_log"

    # awk handles fp arithmetic; bash itself is integer-only.
    baseline_ratio=$(awk -v t="$t_nosink" -v b="$baseline" 'BEGIN{printf "%.3f", t/b}')
    overhead_ratio=$(awk -v s="$t_sink"   -v n="$t_nosink" 'BEGIN{printf "%.3f", s/n}')

    baseline_pass=$(awk -v r="$baseline_ratio" -v g="$BASELINE_TOLERANCE" \
                        'BEGIN{print (r+0 <= g+0) ? 1 : 0}')
    overhead_pass=$(awk -v r="$overhead_ratio" -v g="$SINK_OVERHEAD_MAX" \
                        'BEGIN{print (r+0 <= g+0) ? 1 : 0}')

    printf '   no-sink :  %s ms   (baseline %s ms, ratio %sx, gate ≤ %s)\n' \
           "$t_nosink" "$baseline" "$baseline_ratio" "$BASELINE_TOLERANCE"
    printf '   sink    :  %s ms   (sink/nosink %sx, gate ≤ %s)\n' \
           "$t_sink"   "$overhead_ratio" "$SINK_OVERHEAD_MAX"

    shape_fail=0
    if [ "$baseline_pass" = "1" ]; then
        echo "   PASS  baseline (no-sink instance unchanged)"
    else
        echo "   FAIL  baseline — kHasSink=false instance got slower"
        shape_fail=$((shape_fail + 1))
    fi

    if [ "$overhead_pass" = "1" ]; then
        echo "   PASS  overhead (sink init is near-zero cost)"
    else
        echo "   FAIL  overhead — sink-aware init got expensive"
        shape_fail=$((shape_fail + 1))
    fi

    if [ "$shape_fail" = "0" ]; then
        n_pass=$((n_pass + 1))
    else
        n_fail=$((n_fail + shape_fail))
    fi
done

echo
echo "Summary:"
printf '  PASS (shapes) : %d\n' "$n_pass"
printf '  FAIL (assertions across shapes) : %d\n' "$n_fail"

exit "$n_fail"
