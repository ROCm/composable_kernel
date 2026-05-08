#!/bin/bash
# perf_test_swa.sh - Perf gating test for SWA in CK-tile unified attention.
#
# Asserts that the SWA KV-block iteration clip (the unified_attention_kernel.hpp
# "Sliding-window-attention: tighten the KV-block iteration..." block) is
# actually firing and skipping out-of-window KV blocks.
#
# Strategy: on a long-context prefill shape (kv=8192, q=128) with a small SWA
# window (128), the kernel should iterate ~3 KV sub-blocks per Q-tile instead
# of ~128 for plain causal. We assert >= MIN_SPEEDUP wall-clock speedup.
#
# Measured speedup on MI350 today (gfx950) is 12-20x; we assert 5x to leave
# generous headroom for other GPUs / contention while still catching a Step D
# regression that re-iterates the full KV.
#
# Run with:
#   ./perf_test_swa.sh
#
# Exit code:
#   0 = SWA met the speedup threshold for both shape families.
#   1 = SWA failed to meet the threshold somewhere.
#   2 = environment error (binary not found, parse failure, etc.)

set -uo pipefail

EXE_NAME=tile_example_unified_attention
EXE="${EXE:-$(find . -name $EXE_NAME -type f | head -n 1)}"
if [ -z "${EXE:-}" ] || [ ! -x "$EXE" ]; then
    echo "ERROR: $EXE_NAME not found. Set EXE=/path/to/$EXE_NAME or run from build dir." >&2
    exit 2
fi

# Speedup threshold; the actual measurement is 10-20x, so 5x is a generous
# regression guard.
MIN_SPEEDUP="${MIN_SPEEDUP:-5.0}"

# verify=0 because we trust smoke_test_swa.sh / edge_test_swa.sh for numerics
# and want time_kernel_=true to actually measure clock time, not host-ref work.
COMMON="-prec=bf16 -seed=17 -verify=0 -warmup=5 -repeat=20 -varlen=0 -nb=1024 -page_blk_size=128"

# Long-context prefill: kv=8192 with a 128-token query. This is the regime
# where SWA (window=128) is most lopsided vs causal (full lower triangle).
SHAPE_A="-d=128 -h_k=8 -nqpkv=1 -b=2 -s=128 -s_k=8192 -query_lens=128,128 -kv_lens=8192,8192"
SHAPE_B="-d=64  -h_k=1 -nqpkv=8 -b=2 -s=128 -s_k=8192 -query_lens=128,128 -kv_lens=8192,8192"

# Parse "<...>, <ms> ms, <...>" out of the kernel summary line.
# We grep -oP a single match to avoid partial reads of a large output.
extract_ms() {
    grep -oP '\d+\.\d+(?= ms,)' | head -n 1
}

run_one() {
    local label="$1"; shift
    local out
    out=$("$EXE" $COMMON "$@" 2>&1)
    local ms
    ms=$(echo "$out" | extract_ms)
    if [ -z "$ms" ]; then
        echo "ERROR: failed to extract ms from output of '$label'" >&2
        echo "$out" | tail -10 >&2
        exit 2
    fi
    printf '%s\n' "$ms"
}

# Returns 0 if "$1 / $2 >= $MIN_SPEEDUP", 1 otherwise.
check_speedup() {
    awk -v c="$1" -v s="$2" -v m="$MIN_SPEEDUP" \
        'BEGIN { sp = c / s; if (sp >= m) exit 0; else exit 1 }'
}

n_fail=0
overall_status=0

run_one_shape() {
    local shape_name="$1"
    local shape_args="$2"

    echo "=== $shape_name ==="
    local t_causal t_swa
    t_causal=$(run_one "$shape_name causal" $shape_args -mask=b)
    t_swa=$(run_one    "$shape_name swa"    $shape_args -mask=xb:128)

    local speedup
    speedup=$(awk -v c="$t_causal" -v s="$t_swa" 'BEGIN { printf "%.2f", c / s }')

    printf '  causal       : %8s ms\n' "$t_causal"
    printf '  swa xb:128   : %8s ms\n' "$t_swa"
    printf '  speedup      : %sx (threshold %sx)\n' "$speedup" "$MIN_SPEEDUP"

    if check_speedup "$t_causal" "$t_swa"; then
        echo "  PASS"
    else
        echo "  FAIL: SWA was not >= ${MIN_SPEEDUP}x faster than causal."
        echo "        Most likely culprit: Step D (KV-block iteration clip in"
        echo "        unified_attention_kernel.hpp) was disabled or regressed,"
        echo "        leaving the SWA path iterating the full KV like causal."
        n_fail=$((n_fail + 1))
        overall_status=1
    fi
    echo
}

run_one_shape "d=128 MHA, q=128, kv=8192" "$SHAPE_A"
run_one_shape "d=64  GQA-8 (h_k=1), q=128, kv=8192" "$SHAPE_B"

if [ $overall_status -eq 0 ]; then
    echo "All perf gates passed."
fi

exit $overall_status
