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
# Current baseline — the kernel still ignores sinks; only the host
# reference applies them. The test mix exercises both halves:
#   - 4 GREEN baselines: causal-only with no sink, and the "sink ≈ -inf"
#     case which collapses the reference to no-sink (so kernel & ref agree).
#   - 8 RED cases: active sink values (const:0, const:1.0, random, CSV) —
#     the host reference applies them, the kernel doesn't, so verification
#     diverges by design until the kernel-side sink path is wired up.
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
    # --- GREEN baselines ---
    # Plain causal, no sink flag. These guard the harness-only commit
    # against accidentally breaking the existing causal verification.
    "GREEN|baseA causal           |$BASELINE_A -mask=b"
    "GREEN|baseB causal           |$BASELINE_B -mask=b"

    # `-sink=const:-1e4` collapses the reference's sink contribution to
    # ~exp(-1e4 - m) ≈ 0, so the sink-aware host reference equals the
    # no-sink output — which is also what the (sink-blind) kernel
    # produces. Useful as a regression guard for "the sink CLI didn't
    # accidentally turn into a no-op vs. enabling something dangerous".
    "GREEN|baseA sink=-inf-ish    |$BASELINE_A -mask=b -sink=const:-1e4"
    "GREEN|baseB sink=-inf-ish    |$BASELINE_B -mask=b -sink=const:-1e4"

    # --- RED (kernel ignores sink; reference applies it; comparison
    #     fails until the kHasSink kernel branch ships). ---

    # Zero sink: still adds one virtual key of weight exp(0 - m_max),
    # which is a non-trivial fraction of the existing softmax mass,
    # so the no-sink kernel output differs from the reference.
    "RED  |baseA sink=0           |$BASELINE_A -mask=b -sink=const:0.0"
    "RED  |baseB sink=0           |$BASELINE_B -mask=b -sink=const:0.0"

    # Small positive sink: same idea, heavier mass absorbed.
    "RED  |baseA sink=1.0         |$BASELINE_A -mask=b -sink=const:1.0"
    "RED  |baseB sink=1.0         |$BASELINE_B -mask=b -sink=const:1.0"

    # Per-head random sinks via the 'random:N' draw. Exercises the
    # per-Q-head indexing in the reference (each head sees a distinct
    # logit) without depending on a CSV length matching nhead_q.
    "RED  |baseA sink=random:17   |$BASELINE_A -mask=b -sink=random:17"
    "RED  |baseB sink=random:17   |$BASELINE_B -mask=b -sink=random:17"

    # Explicit per-head CSV — proves the CSV parser path (baseA has
    # h_k*nqpkv = 8*1 = 8 heads; baseB has 1*8 = 8). Length must match
    # nhead_q exactly.
    "RED  |baseA sink=CSV[8]      |$BASELINE_A -mask=b -sink=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8"
    "RED  |baseB sink=CSV[8]      |$BASELINE_B -mask=b -sink=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8"
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
