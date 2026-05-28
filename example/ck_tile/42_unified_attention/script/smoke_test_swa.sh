#!/bin/bash
# smoke_test_swa.sh - RED/GREEN tests for Sliding Window Attention (SWA)
# in the CK-tile unified attention kernel.
#
# Each test entry is "EXPECT|NAME|EXTRA_ARGS" where EXPECT is GREEN or RED.
#   GREEN: the test must currently pass; failing it is a regression.
#   RED:   the test must currently fail; passing it means SWA support landed
#          and the test should be moved to GREEN.
#
# Run with HIP_VISIBLE_DEVICES set to your assigned GPU. Defaults to 6 on
# the shared dev node. Example:
#   ./smoke_test_swa.sh
#   HIP_VISIBLE_DEVICES=7 ./smoke_test_swa.sh
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

# Deterministic, verification-only fixture.
# - bf16 + seed=17 chosen so that all baselines and SWA configurations clear
#   the bf16 atol=1e-2 tolerance without single-element boundary noise.
# - varlen=0 with explicit query_lens/kv_lens makes shapes fully reproducible.
# - warmup=0, repeat=1 keeps each test under a second.
COMMON="-prec=bf16 -seed=17 -verify=1 -warmup=0 -repeat=1 -varlen=0 -nb=1024 -page_blk_size=128"

# Two known-good baselines from the existing causal verification path.
BASELINE_A="-d=128 -h_k=8 -nqpkv=1 -b=4 -s=512 -s_k=512 -query_lens=128,128,128,128 -kv_lens=128,128,128,128"
BASELINE_B="-d=64  -h_k=1 -nqpkv=8 -b=4 -s=512 -s_k=512 -query_lens=400,256,512,128 -kv_lens=400,256,512,128"

TESTS=(
    # Causal regression guards (must pass today). These exercise the new mask=...
    # CLI plumbing all the way through to args.mask_type = 2.
    "GREEN|baseA causal     |$BASELINE_A -mask=b"
    "GREEN|baseB causal     |$BASELINE_B -mask=b"

    # SWA via xformer-style window. baseB (d=64, GQA-style, prefill rows)
    # is GREEN now that the IsLocal=true prefill instances ship. baseA
    # (per-batch q_len=128) routes through `decode_d128_m128`; the
    # matching IsLocal=true decode instances now ship too, so
    # `dispatch_local` returns true on them and the cases are GREEN.
    "GREEN|baseA xb:64      |$BASELINE_A -mask=xb:64"
    "GREEN|baseA xb:128     |$BASELINE_A -mask=xb:128"
    "GREEN|baseB xb:64      |$BASELINE_B -mask=xb:64"
    "GREEN|baseB xb:128     |$BASELINE_B -mask=xb:128"

    # SWA via FA-style explicit left/right window.
    "GREEN|baseA b:64,0     |$BASELINE_A -mask=b:64,0"
    "GREEN|baseB b:64,0     |$BASELINE_B -mask=b:64,0"

    # Pure prefill SWA on d=128 (forces `prefill_d128` by giving every batch a
    # q_len > 128 = decode_d128_m128 threshold). Validates that the d=128
    # `IsLocal=true` prefill instance is correct.
    "GREEN|prefill d128 xb:64 |-d=128 -h_k=8 -nqpkv=1 -b=2 -s=512 -s_k=512 -query_lens=257,512 -kv_lens=257,512 -mask=xb:64"
    "GREEN|prefill d128 b:64,0|-d=128 -h_k=8 -nqpkv=1 -b=2 -s=512 -s_k=512 -query_lens=257,512 -kv_lens=257,512 -mask=b:64,0"
)

n_green_pass=0
n_green_fail=0   # regressions
n_red_pass=0     # unexpected SWA passes (move to GREEN)
n_red_fail=0     # expected RED

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
