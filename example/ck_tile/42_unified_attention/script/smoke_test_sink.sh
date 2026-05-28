#!/bin/bash
# smoke_test_sink.sh - RED/GREEN tests for learnable per-Q-head attention
# sinks (GPT-OSS / vLLM convention) in the CK-tile unified attention kernel.
#
# Each test entry is "EXPECT|NAME|EXTRA_ARGS" where EXPECT is GREEN or RED.
#   GREEN: the test must currently pass; failing it is a regression.
#   RED:   the test must currently fail; passing it means the device-side
#          sink branch (`kHasSink` in the kernel) has landed and the
#          test should be moved to GREEN.
#
# Current baseline — the kernel-side sink path is wired in the *prefill*
# tier only (prefill_d128 + prefill_d64, bf16/fp16). Decode tiers still
# fall through to the no-sink kernel, and `dispatch_variant` fast-fails
# (returns {false, -1.f}) whenever `args.sink_ptr != nullptr` AND the
# selected variant has no sink instance compiled yet — the same SWA-Phase-
# 8-trap-style prophylaxis that keeps users from silently getting the
# wrong output. As a side effect, *every* `is_sink=true` call on a
# decode tier currently fails the dispatcher, which matches the host
# reference (with sinks applied) for `const:0/1.0/random/CSV` (still
# RED — kernel never ran) and *no longer* matches it for `const:-1e4`
# (formerly GREEN under the no-op CLI; now RED because the dispatcher
# refuses the call). The decode-tier rows below are RED until the
# decode sink instances ship in a later phase.
#
# Test mix:
#   - 3 GREEN no-sink baselines (causal-only on baseA + baseB, plus
#     `sink≈-inf` on baseB which dispatches to prefill_d64's sink
#     instance and produces ≈ the no-sink output).
#   - 4 GREEN sink-prefill cases (baseB × {const:0, const:1.0, random,
#     CSV} — each hits prefill_d64's sink instance and matches the
#     host reference within bf16 tolerance).
#   - 5 RED decode-tier cases (baseA × {sink≈-inf, const:0, const:1.0,
#     random, CSV} — every baseA dispatches to decode_d128_m128, which
#     has no sink instance yet; dispatcher fast-fails for all 5).
#
# Run with HIP_VISIBLE_DEVICES set to your assigned GPU. Defaults to 6 on
# the shared dev node. Example:
#   ./smoke_test_sink.sh
#   HIP_VISIBLE_DEVICES=7 ./smoke_test_sink.sh
#
# Exit code is the number of unexpected outcomes (0 = all matched expectation).

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

# Deterministic, verification-only fixture (matches smoke_test_swa.sh).
# - bf16 + seed=17 keeps both backends inside the bf16 atol=1e-2 envelope on
#   the no-sink and "sink-vanishes" cases without single-element noise.
# - varlen=0 + explicit query_lens/kv_lens makes shapes fully reproducible.
# - warmup=0, repeat=1 keeps each test under a second.
COMMON="-prec=bf16 -seed=17 -verify=1 -warmup=0 -repeat=1 -varlen=0 -nb=1024 -page_blk_size=128"

# Two known-good baselines (same two as smoke_test_swa.sh) so the sink
# tests share the same shapes the SWA pipeline already validates. Both
# exercise the dispatcher's classical-causal path (mask=b → mask_type=2,
# window=(-1, 0, false), no SWA, no sink).
BASELINE_A="-d=128 -h_k=8 -nqpkv=1 -b=4 -s=512 -s_k=512 -query_lens=128,128,128,128 -kv_lens=128,128,128,128"
BASELINE_B="-d=64  -h_k=1 -nqpkv=8 -b=4 -s=512 -s_k=512 -query_lens=400,256,512,128 -kv_lens=400,256,512,128"

TESTS=(
    # --- GREEN no-sink baselines ---
    # Plain causal, no sink flag. These guard the harness-only commit
    # against accidentally breaking the existing causal verification.
    "GREEN|baseA causal           |$BASELINE_A -mask=b"
    "GREEN|baseB causal           |$BASELINE_B -mask=b"

    # `-sink=const:-1e4` collapses the kernel's sink contribution to
    # ~exp(-1e4 - m) ≈ 0. With Phase 4's prefill sink instance in place,
    # baseB dispatches to prefill_d64's sink kernel and produces ≈ the
    # no-sink output — matching the host reference within bf16 noise.
    # baseA hits decode_d128_m128 (no sink instance yet), so its sink≈-inf
    # case is RED below and will flip GREEN once decode sink instances
    # ship.
    "GREEN|baseB sink=-inf-ish    |$BASELINE_B -mask=b -sink=const:-1e4"

    # --- GREEN sink-aware (prefill tier, Phase 4): baseB dispatches to
    #     prefill_d64's sink instance which seeds the online softmax with
    #     m = sink_raw / sm_scale, l = 1, o_acc = 0 — matching the host
    #     reference within bf16 tolerance for every sink magnitude. ---

    # Zero sink: one virtual key with weight exp(0 - m_max). The kernel's
    # sink-aware softmax produces the same result as the host reference's
    # sink-aware softmax. Bit-equivalent to "kernel and reference both
    # apply the same sink correctly".
    "GREEN|baseB sink=0           |$BASELINE_B -mask=b -sink=const:0.0"

    # Small positive sink: heavier mass absorbed by the sink. Still
    # bit-equivalent.
    "GREEN|baseB sink=1.0         |$BASELINE_B -mask=b -sink=const:1.0"

    # Per-head random sinks via the 'random:N' draw. Exercises the
    # per-Q-head indexing in both the kernel and the reference (each head
    # sees a distinct logit) without depending on a CSV length matching
    # nhead_q.
    "GREEN|baseB sink=random:17   |$BASELINE_B -mask=b -sink=random:17"

    # Explicit per-head CSV — proves the CSV parser path (baseB has
    # h_k*nqpkv = 1*8 = 8 heads).
    "GREEN|baseB sink=CSV[8]      |$BASELINE_B -mask=b -sink=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8"

    # --- RED decode-tier cases (Phase 6 will flip these GREEN). All
    #     baseA configs (-d=128 -h_k=8 -nqpkv=1 -query_lens=128,...) hit
    #     decode_d128_m128 because avg_rows=128. No decode_*_sink
    #     instance is compiled yet, so `dispatch_sink<decode_*, ...>`
    #     returns {false, 0.f} → the example prints "faild to run
    #     unified_attention()" → rc=1 → RED. ---

    "RED  |baseA sink=-inf-ish    |$BASELINE_A -mask=b -sink=const:-1e4"
    "RED  |baseA sink=0           |$BASELINE_A -mask=b -sink=const:0.0"
    "RED  |baseA sink=1.0         |$BASELINE_A -mask=b -sink=const:1.0"
    "RED  |baseA sink=random:17   |$BASELINE_A -mask=b -sink=random:17"
    "RED  |baseA sink=CSV[8]      |$BASELINE_A -mask=b -sink=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8"
)

n_green_pass=0
n_green_fail=0   # regressions
n_red_pass=0     # unexpected sink passes (move to GREEN)
n_red_fail=0     # expected RED

for entry in "${TESTS[@]}"; do
    expect="${entry%%|*}"
    expect="${expect// /}"
    rest="${entry#*|}"
    name="${rest%%|*}"
    args="${rest#*|}"

    printf '== [%-5s] %-22s :: %s\n' "$expect" "$name" "$args"
    set +e
    "$EXE" $COMMON $args > /tmp/sink_test_out.$$ 2>&1
    ret=$?
    set -e

    if [ "$expect" = "GREEN" ]; then
        if [ $ret -eq 0 ]; then
            echo "   PASS (as expected)"
            n_green_pass=$((n_green_pass + 1))
        else
            echo "   REGRESSION: expected GREEN but failed (rc=$ret). Tail of output:"
            tail -3 /tmp/sink_test_out.$$ | sed 's/^/      /'
            n_green_fail=$((n_green_fail + 1))
        fi
    else
        if [ $ret -ne 0 ]; then
            echo "   FAIL (RED, as expected)"
            n_red_fail=$((n_red_fail + 1))
        else
            echo "   UNEXPECTED PASS: sink support may have landed. Move this test to GREEN."
            n_red_pass=$((n_red_pass + 1))
        fi
    fi
    rm -f /tmp/sink_test_out.$$
done

echo
echo "Summary:"
printf '  GREEN passed (good)              : %d\n' $n_green_pass
printf '  GREEN failed (REGRESSION)        : %d\n' $n_green_fail
printf '  RED   failed (expected today)    : %d\n' $n_red_fail
printf '  RED   passed (flip to GREEN now) : %d\n' $n_red_pass

# Exit code = number of unexpected outcomes.
exit $((n_green_fail + n_red_pass))
