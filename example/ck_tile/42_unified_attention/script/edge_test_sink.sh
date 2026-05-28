#!/bin/bash
# edge_test_sink.sh - edge-case tests for the kernel-side per-Q-head
# attention sink path on the prefill tier (prefill_d64 / prefill_d128).
#
# These cases probe corners that the smoke test does not:
#   1. sink ≈ -infinity  (collapses to no-sink output; sanity guard
#      against a runaway sign / overflow in the m-init).
#   2. sink ≈ +infinity  (sink absorbs all softmax mass; output → 0).
#   3. sink = 0          (sink contributes weight 1 per virtual key).
#   4. per-head random   (exercises the per-row head indexing in both
#                         kernel and reference; default GQA-broadcast
#                         check baked in because baseB has 8 Q-heads).
#   5. per-head CSV      (the host-side CSV parser path).
#   6. random + d128     (covers prefill_d128 with non-trivial per-head
#                         sinks).
#
# The "all-window-masked + sink" fixture from the plan (combine the SWA
# `b:0,0` window with `sink=const:0.0`) needs sink_local instances —
# those land in a later phase. The dispatcher currently fast-fails the
# (sink && is_local) combo to avoid silently routing to a non-sink
# instance; that fixture is documented in the script body but skipped
# below.
#
# Run with HIP_VISIBLE_DEVICES set; defaults to 6 on the shared dev
# node.
#
# Exit code = number of FAIL'd tests (0 = all PASS).

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

# Deterministic, verification-only fixtures.
COMMON="-prec=bf16 -seed=17 -verify=1 -warmup=0 -repeat=1 -varlen=0 -nb=1024 -page_blk_size=128"

# baselineB: prefill_d64 (h_k=1, nqpkv=8 → GQA-8 fan-out). Same fixture
# as smoke_test_sink.sh so the two scripts cross-validate.
BASELINE_B="-d=64  -h_k=1 -nqpkv=8 -b=4 -s=512 -s_k=512 -query_lens=400,256,512,128 -kv_lens=400,256,512,128"

# baselineD: prefill_d128 with non-trivial query lengths so the variant
# selector lands on prefill_d128 (not decode_d128_m128 which has no
# sink instance yet).
BASELINE_D="-d=128 -h_k=8 -nqpkv=1 -b=4 -s=512 -s_k=512 -query_lens=300,300,300,300 -kv_lens=300,300,300,300"

TESTS=(
    # 1. Sink ≈ -infinity: kernel must collapse the sink contribution to
    #    ~0 (m_init = -1e4/sm_scale dominated by max_j S_raw_j after the
    #    first iteration), reproducing the no-sink output. Both kernel
    #    and reference apply the same near-zero sink, so they match.
    "baseB sink≈-inf       |$BASELINE_B -mask=b -sink=const:-1e4"

    # 2. Sink ≈ +infinity: kernel must absorb almost all the softmax
    #    mass onto the sink (m_init = 1e4/sm_scale dominates max_j S_raw_j),
    #    driving the post-normalization V-weighted output to ≈ 0. The
    #    reference does the same, so they match within bf16 noise (the
    #    output is dominated by zero ± rounding).
    "baseB sink≈+inf       |$BASELINE_B -mask=b -sink=const:+1e4"

    # 3. Sink = 0: one virtual key with weight exp(0 - m_max); ~0.1-1%
    #    of the existing softmax mass. Output is materially different
    #    from no-sink but well-defined; both kernel and reference must
    #    agree.
    "baseB sink=0          |$BASELINE_B -mask=b -sink=const:0.0"

    # 4. Per-head random sinks (GQA-8 fan-out): baseB has 8 Q-heads, so
    #    `random:17` draws 8 distinct sink values. The kernel's per-row
    #    indexing `sink_ptr_pre_offset[r % num_queries_per_kv]` must
    #    match the reference's `sink[q_head_idx]` lookup — the same head
    #    must receive the same sink across every q-token in its group.
    "baseB sink=random:17  |$BASELINE_B -mask=b -sink=random:17"

    # 5. Explicit per-head CSV (8 distinct sinks). Same indexing check
    #    as (4), but with the CSV parser path instead of the seeded
    #    'random:N' draw.
    "baseB sink=CSV[8]     |$BASELINE_B -mask=b -sink=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8"

    # 6. prefill_d128 with per-head random sinks (h_k=8, nqpkv=1 → 8
    #    Q-heads, one Q per KV group). Validates the d=128 sink
    #    instance and the GQA-1 indexing edge.
    "baseD d128 sink=rand  |$BASELINE_D -mask=b -sink=random:17"
)

# (Skipped) All-window-masked + sink — needs sink_local instances which
# land in a later phase. Dispatcher currently fast-fails (sink &&
# is_local) → {false, -1.f}, so the test would FAIL. Re-enable when
# Phase 9 ships:
#   $BASELINE_B -mask=b:0,0 -sink=const:0.0
# Expected outcome at that point: kernel writes 0 for sink-only Q-tiles
# (not NaN), reference does the same.

n_pass=0
n_fail=0

for entry in "${TESTS[@]}"; do
    name="${entry%%|*}"
    name="${name// /}"
    args="${entry#*|}"

    printf '== %-22s :: %s\n' "$name" "$args"
    set +e
    "$EXE" $COMMON $args > /tmp/sink_edge.$$ 2>&1
    ret=$?
    set -e

    if [ $ret -eq 0 ]; then
        echo "   PASS"
        n_pass=$((n_pass + 1))
    else
        echo "   FAIL (rc=$ret). Tail of output:"
        tail -3 /tmp/sink_edge.$$ | sed 's/^/      /'
        n_fail=$((n_fail + 1))
    fi
    rm -f /tmp/sink_edge.$$
done

echo
echo "Summary:"
printf '  PASS : %d\n' $n_pass
printf '  FAIL : %d\n' $n_fail

exit $n_fail
