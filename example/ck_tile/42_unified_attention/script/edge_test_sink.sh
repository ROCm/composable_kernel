#!/bin/bash
# edge_test_sink.sh - edge-case tests for the kernel-side per-Q-head
# attention sink path. Covers the prefill tier (prefill_d{64,128}), the
# decode tier (decode_d{64,128}_m{16,32/64,128}), and the SWA × sink
# combo (IsLocal=true + kHasSink=true) — all in one matrix.
#
# Probed corners:
#   1. sink ≈ -infinity  (collapses to no-sink output; sanity guard
#      against a runaway sign / overflow in the m-init).
#   2. sink ≈ +infinity  (sink absorbs all softmax mass; output → 0).
#   3. sink = 0          (sink contributes weight 1 per virtual key).
#   4. per-head random   (exercises the per-row head indexing in both
#                         kernel and reference; default GQA-broadcast
#                         check baked in because baseB has 8 Q-heads).
#   5. per-head CSV      (the host-side CSV parser path).
#   6. random + d128     (prefill_d128 with non-trivial per-head sinks).
#   7. GPT-OSS canonical shape — decode_d64_m16 (q=1 generation,
#      d=64, GQA-8). Three variants: sink=random + SWA, sink=random + no
#      SWA, and sink=const:-1e4 (reduces to no-sink decode case).
#   8. Non-page-aligned stress — page_blk_size ∈ {64, 128} on
#      prefill_d{64,128} with random sinks. Sinks don't touch the
#      page-table arithmetic, so this is a pure regression guard. No
#      kernel-side change was needed for paging — paging is computed
#      from `kargs.page_size` and the block table; the sink init lives
#      in the per-row online softmax setup.
#   9. SWA × sink combo — the (kHasSink=true, IsLocal=true) cell.
#      Includes the all-window-masked Q-tile case (-mask=b:0,0)
#      which hits the pipeline's no-work early-exit path; with sink
#      that path writes lse = sm_scale * sink_raw, output = 0 (not NaN).
#
# Split-KV × sink edge cases are deliberately omitted from this script:
# example 42's CLI doesn't yet expose -num_splits, so the example main
# always runs num_splits=1. The kernel-side gate
# (`i_split == 0 ? sink_ptr : nullptr`) and the pipeline-side null-guard
# are wired in and will be exercised by aiter's Python binding the
# moment a split-KV launch lands. When example 42 grows a -num_splits
# CLI, add decode_d64_m16 + -num_splits=8 ± sink cases here.
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
# selector lands on prefill_d128 (not decode_d128_m128). The decode_
# d128_m128 sink path is covered separately by the decode-tier cases
# below; this fixture stays focused on the prefill_d128 sink instance.
BASELINE_D="-d=128 -h_k=8 -nqpkv=1 -b=4 -s=512 -s_k=512 -query_lens=300,300,300,300 -kv_lens=300,300,300,300"

# decodeOSS: GPT-OSS canonical shape — q=1 generation step, d=64,
# GQA-8, batch=32. Average rows per Q-tile = 1 * 8 = 8 ≤ 16, so the
# variant selector lands on decode_d64_m16 (the "_t" tiny-decode tier).
# This is *the* shape that motivated the whole sink rollout.
DECODE_OSS="-d=64 -h_k=8 -nqpkv=8 -b=32 -s=1 -s_k=128 -varlen=0"

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

    # ---- GPT-OSS canonical shapes (decode_d64_m16) ----

    # 7. The canonical GPT-OSS-with-sink call: q=1 generation step on a
    #    GQA-8 d=64 model, with SWA window=128 and a per-head random
    #    sink. Routes to decode_d64_m16 × SWA × sink.
    "ossDecode swa+sink    |$DECODE_OSS -mask=t:128,0 -sink=random:17"

    # 8. Same shape, no SWA — exercises decode_d64_m16 sink instance
    #    *without* the IsLocal=true branch.
    "ossDecode sink only   |$DECODE_OSS -mask=b -sink=random:17"

    # 9. Same shape, sink=const:-1e4 — reduces to the no-sink decode
    #    case bit-for-bit (up to fp32 ordering); a sanity check that
    #    a near-zero sink mass doesn't perturb the reference shape's
    #    output. Verification must still PASS (both kernel and
    #    reference apply the same near-zero sink).
    "ossDecode sink≈-inf   |$DECODE_OSS -mask=b -sink=const:-1e4"

    # ---- Non-page-aligned page sizes ----
    # Stress the page-table arithmetic with non-default page block
    # sizes alongside an active sink. The sink doesn't touch paging,
    # so this is a regression guard — failures here would point at a
    # paging bug being uncovered by, not caused by, the sink path.

    "baseB pgblk=64 rand   |$BASELINE_B -mask=b -sink=random:17 -page_blk_size=64"
    "baseB pgblk=128 rand  |$BASELINE_B -mask=b -sink=random:17 -page_blk_size=128"
    "baseD pgblk=64 rand   |$BASELINE_D -mask=b -sink=random:17 -page_blk_size=64"
    "baseD pgblk=128 rand  |$BASELINE_D -mask=b -sink=random:17 -page_blk_size=128"

    # ---- SWA × sink combo (IsLocal=true + kHasSink=true) ----
    # The 16 _local_sink.cpp instance files (8 variants × {bf16, fp16})
    # populate the (sink && is_local) cell of the dispatcher. The two
    # `if constexpr` branches (SWA Step-D clip in the kernel, sink init
    # in the pipeline) are orthogonal and compose.

    # 10. baseB + SWA window=64, random sinks. The window is small
    #     enough that some Q-tiles will have non-trivial KV overlap
    #     and others will exit early — both paths now write a
    #     sink-aware partial.
    "baseB swa64+rand      |$BASELINE_B -mask=t:64,0 -sink=random:17"

    # 11. baseD prefill_d128 + SWA window=128, const sink. Same idea
    #     on the d=128 path.
    "baseD swa128+const    |$BASELINE_D -mask=t:128,0 -sink=const:0.5"

    # 12. All-window-masked + sink (the case the plan calls out
    #     explicitly): SWA window collapses to zero overlap on every
    #     Q-tile, so the pipeline's no-work early-exit fires for every
    #     row. With sink, that early-exit writes lse = sm_scale *
    #     sink_raw and o_acc = 0 — the output normalizes to exactly 0
    #     (not NaN, not -inf). Reference does the same.
    "baseB swa00+const0    |$BASELINE_B -mask=b:0,0 -sink=const:0.0"
)

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
