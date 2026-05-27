#!/bin/bash
# perf_test_swa.sh — perf gate that locks in Step D's KV-tile clip.
#
# Step D (the IsLocal=true tile-range clip in unified_attention_kernel.hpp::
# operator()) is what makes long-context SWA fast. Without it the SWA path
# is correctness-preserving — the per-pixel mask still zeros tokens outside
# the window — but the kernel iterates the *full* causal KV range and
# multiplies most cells by zero. On q=kv=8192 / window=128 that's a ≥10×
# slowdown.
#
# This script catches future regressions that silently disable Step D
# (e.g. mask coords flipped, the `IsLocal && IsMasking` guard turned off,
# `_max_seq_prefix_len` envelope opened back up to seq_len, …). It asserts
#
#     time(-mask=b)  /  time(-mask=xb:128)  ≥  RATIO_MIN
#
# on the two shape families that exercise the Phase 3 SWA instances:
#
#   * d=128 MHA prefill        (-d=128 -h_k=8 -nqpkv=1)
#   * d=64  GQA-8 prefill      (-d=64  -h_k=1 -nqpkv=8)
#
# Both run with q=kv=8192 — the largest comfortable shape on a single CTA's
# worth of pages — and window=128. The threshold below is set with headroom:
# observed ratios on MI355 are 10–21×, so 5× catches a regression long
# before the kernel becomes "as slow as causal".
#
# Run with HIP_VISIBLE_DEVICES set to your assigned GPU (defaults to 6).
# Exit code is the number of shapes that failed the threshold.

set -uo pipefail

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-6}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXE_NAME=tile_example_unified_attention
EXE="${EXE:-$(find . -name "$EXE_NAME" -type f -executable 2>/dev/null | head -n 1)}"
if [ -z "${EXE:-}" ] || [ ! -x "$EXE" ]; then
    echo "ERROR: $EXE_NAME not found. Set EXE=/path/to/$EXE_NAME or run from build dir." >&2
    exit 2
fi
echo "Using EXE=$EXE"
echo "Using HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"

# Speedup threshold: 5× headroom on the observed ~10–21× ratios. Below
# this means Step D was disabled or the SWA envelope blew open.
RATIO_MIN="${RATIO_MIN:-5.0}"

# Bench-only config — verify off, ample warmup, 30 timed repeats so the
# faster runs are dominated by the kernel and not the launch overhead.
COMMON="-prec=bf16 -seed=17 -verify=0 -warmup=10 -repeat=30 -varlen=0 \
  -nb=128 -page_blk_size=128 -b=1 -s=8192 -s_k=8192 \
  -query_lens=8192 -kv_lens=8192"

# Each row is "NAME|EXTRA_ARGS". Both -mask=b and -mask=xb:128 are appended
# per run by the loop; we never time the kernel with verify on.
SHAPES=(
    "d128 MHA  prefill q=kv=8192 win=128 |-d=128 -h_k=8 -nqpkv=1"
    "d64  GQA-8 prefill q=kv=8192 win=128|-d=64  -h_k=1 -nqpkv=8"
)

# Extract the "X.XXX ms" from a single benchmark line such as
#   "[bf16|] b:1, h:8/8, ..., 0.45200819 ms, 304.10 TFlops, ..."
# Returns the first such number or the empty string on failure.
extract_ms() {
    grep -oE '[0-9]+\.[0-9]+ ms' "$1" | head -1 | awk '{print $1}'
}

n_pass=0
n_fail=0

for row in "${SHAPES[@]}"; do
    name="${row%%|*}"
    args="${row#*|}"

    printf '== %s\n' "$name"

    causal_log=$(mktemp)
    swa_log=$(mktemp)
    "$EXE" $COMMON $args -mask=b      > "$causal_log" 2>&1 || true
    "$EXE" $COMMON $args -mask=xb:128 > "$swa_log"    2>&1 || true

    t_causal=$(extract_ms "$causal_log")
    t_swa=$(extract_ms "$swa_log")

    if [ -z "$t_causal" ] || [ -z "$t_swa" ]; then
        echo "   FAIL: could not parse timing"
        echo "   causal tail:"
        tail -3 "$causal_log" | sed 's/^/      /'
        echo "   swa tail:"
        tail -3 "$swa_log"    | sed 's/^/      /'
        n_fail=$((n_fail + 1))
        rm -f "$causal_log" "$swa_log"
        continue
    fi
    rm -f "$causal_log" "$swa_log"

    # awk handles fp arithmetic; bash itself is integer-only.
    ratio=$(awk -v c="$t_causal" -v s="$t_swa" 'BEGIN{printf "%.2f", c/s}')
    passed=$(awk -v r="$ratio" -v m="$RATIO_MIN" 'BEGIN{print (r+0 >= m+0) ? 1 : 0}')

    printf '   causal:  %s ms\n'   "$t_causal"
    printf '   xb:128:  %s ms\n'   "$t_swa"
    printf '   ratio :  %sx  (gate ≥ %s)\n' "$ratio" "$RATIO_MIN"
    if [ "$passed" = "1" ]; then
        echo "   PASS"
        n_pass=$((n_pass + 1))
    else
        echo "   FAIL — Step D regression suspected (SWA too slow vs causal)"
        n_fail=$((n_fail + 1))
    fi
done

echo
echo "Summary:"
printf '  PASS : %d\n' "$n_pass"
printf '  FAIL : %d\n' "$n_fail"

exit "$n_fail"
