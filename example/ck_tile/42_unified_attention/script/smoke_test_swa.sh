#!/bin/bash
# smoke_test_swa.sh - Phase 1 RED tests for Sliding Window Attention (SWA)
# in the CK-tile unified attention kernel.
#
# Each test entry is "EXPECT|EXTRA_ARGS" where EXPECT is GREEN or RED.
#   GREEN: the test must currently pass; failing it is a regression.
#   RED:   the test must currently fail; passing it means SWA support landed
#          and the test should be moved to GREEN.
#
# Run with HIP_VISIBLE_DEVICES set to your assigned GPU. Example:
#   HIP_VISIBLE_DEVICES=7 ./smoke_test_swa.sh
#
# Exit code is the number of unexpected outcomes (0 = all matched expectation).

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXE_NAME=tile_example_unified_attention
EXE="${EXE:-$(find . -name $EXE_NAME -type f | head -n 1)}"
if [ -z "${EXE:-}" ] || [ ! -x "$EXE" ]; then
    echo "ERROR: $EXE_NAME not found. Set EXE=/path/to/$EXE_NAME or run from build dir." >&2
    exit 2
fi

# Deterministic, verification-only fixture.
# - bf16 + seed=13 chosen so that both baselines pass causal (-mask=b) without
#   tripping pre-existing single-element bf16 rounding noise.
# - varlen=0 with explicit query_lens/kv_lens makes shapes fully reproducible.
# - warmup=0, repeat=1 keeps each test under a second.
COMMON="-prec=bf16 -seed=13 -verify=1 -warmup=0 -repeat=1 -varlen=0 -nb=1024 -page_blk_size=128"

# Two known-good baselines from the existing causal verification path.
BASELINE_A="-d=128 -h_k=8 -nqpkv=1 -b=4 -s=512 -s_k=512 -query_lens=128,128,128,128 -kv_lens=128,128,128,128"
BASELINE_B="-d=64  -h_k=1 -nqpkv=8 -b=4 -s=512 -s_k=512 -query_lens=400,256,512,128 -kv_lens=400,256,512,128"

TESTS=(
    # Causal regression guards (must pass).
    "GREEN|baseA causal     |$BASELINE_A -mask=b"
    "GREEN|baseB causal     |$BASELINE_B -mask=b"

    # SWA via xformer-style window. Today the kernel does not honor the SWA
    # lower bound (its KV-block iteration is implicitly causal), so these fail.
    "RED  |baseA xb:64      |$BASELINE_A -mask=xb:64"
    "RED  |baseA xb:128     |$BASELINE_A -mask=xb:128"
    "RED  |baseB xb:64      |$BASELINE_B -mask=xb:64"
    "RED  |baseB xb:128     |$BASELINE_B -mask=xb:128"

    # SWA via FA-style explicit left/right window.
    "RED  |baseA b:64,0     |$BASELINE_A -mask=b:64,0"
    "RED  |baseB b:64,0     |$BASELINE_B -mask=b:64,0"
)

n_green_pass=0
n_green_fail=0   # regressions
n_red_pass=0    # unexpected SWA passes (move to GREEN)
n_red_fail=0    # expected RED

for entry in "${TESTS[@]}"; do
    expect="${entry%%|*}"
    expect="${expect// /}"
    rest="${entry#*|}"
    name="${rest%%|*}"
    args="${rest#*|}"

    printf '== [%-5s] %-22s :: %s\n' "$expect" "$name" "$args"
    set +e
    "$EXE" $COMMON $args > /tmp/swa_test_out.$$ 2>&1
    ret=$?
    set -e

    if [ "$expect" = "GREEN" ]; then
        if [ $ret -eq 0 ]; then
            echo "   PASS (as expected)"
            n_green_pass=$((n_green_pass + 1))
        else
            echo "   REGRESSION: expected GREEN but failed (rc=$ret). Tail of output:"
            tail -3 /tmp/swa_test_out.$$ | sed 's/^/      /'
            n_green_fail=$((n_green_fail + 1))
        fi
    else
        if [ $ret -ne 0 ]; then
            echo "   FAIL (RED, as expected)"
            n_red_fail=$((n_red_fail + 1))
        else
            echo "   UNEXPECTED PASS: SWA support may have landed. Move this test to GREEN."
            n_red_pass=$((n_red_pass + 1))
        fi
    fi
    rm -f /tmp/swa_test_out.$$
done

echo
echo "Summary:"
printf '  GREEN passed (good)              : %d\n' $n_green_pass
printf '  GREEN failed (REGRESSION)        : %d\n' $n_green_fail
printf '  RED   failed (expected today)    : %d\n' $n_red_fail
printf '  RED   passed (flip to GREEN now) : %d\n' $n_red_pass

# Exit code = number of unexpected outcomes.
exit $((n_green_fail + n_red_pass))
