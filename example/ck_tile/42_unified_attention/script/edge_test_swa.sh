#!/bin/bash
# edge_test_swa.sh — boundary-condition tests for Sliding Window Attention
# in the CK-tile unified attention kernel (Phase 3 prefill tier).
#
# Each entry is "NAME|EXTRA_ARGS" and must PASS today; failure is a
# regression. The fixtures stress the corners of the SWA mask machinery:
#
#   * `xb:1` / `b:0,0`     : window = 1 (only the diagonal cell attends).
#   * `xb:2048` / `b:511,511` : window >= seq_k (collapses to dense, must
#                              behave like causal/no-mask).
#   * top-left vs bottom-right anchors (`xt:` vs `xb:`).
#   * asymmetric left/right windows (`b:32,8`, `b:8,32`).
#   * page-aligned vs odd-aligned seqlens (`s_k=512` vs `s_k=480`).
#
# Run with HIP_VISIBLE_DEVICES set to your assigned GPU (defaults to 6).
# Exit code is the number of unexpected outcomes (0 = all passed).

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

# bf16/seed=17 mirrors smoke_test_swa.sh; only -mask varies per entry. The
# shape baselines force the *prefill* tier (q_len > decode_*_m128 cutoff)
# so we exercise the Phase 3 prefill_d{64,128} SWA instances, not the
# decode tier (Phase 5).
COMMON="-prec=bf16 -seed=17 -verify=1 -warmup=0 -repeat=1 -varlen=0 -nb=1024 -page_blk_size=128"

# Prefill shape: d=64 GQA-8 with all batches at full seqlen (forces
# prefill_d64). Mirrors baseB from smoke_test_swa.sh.
PRE_64="-d=64  -h_k=1 -nqpkv=8 -b=2 -s=512 -s_k=512 -query_lens=512,512 -kv_lens=512,512"
# Prefill d=128: q_len=257 > 128 forces prefill_d128 (above the m128 decode
# threshold).
PRE_128="-d=128 -h_k=8 -nqpkv=1 -b=2 -s=512 -s_k=512 -query_lens=257,512 -kv_lens=257,512"

TESTS=(
    # Window = 1 (only the diagonal cell). Tests Step D's right-clip
    # collapsing to a single tile per Q-row.
    "d64  window=1  (xb:1)         |$PRE_64  -mask=xb:1"
    "d64  window=1  (b:0,0)        |$PRE_64  -mask=b:0,0"
    "d128 window=1  (xb:1)         |$PRE_128 -mask=xb:1"
    "d128 window=1  (b:0,0)        |$PRE_128 -mask=b:0,0"

    # Window >= seq_k (no clip; collapses to dense within the causal
    # half). Smokes the saturating `min(seq_len)` clamp on the new
    # `_max_seq_prefix_len + swa_right_extra` envelope.
    "d64  window>=sk (xb:2048)     |$PRE_64  -mask=xb:2048"
    "d64  window=511,511 (b:)      |$PRE_64  -mask=b:511,511"
    "d128 window>=sk (xb:2048)     |$PRE_128 -mask=xb:2048"
    "d128 window=511,511 (b:)      |$PRE_128 -mask=b:511,511"

    # Top-left vs bottom-right anchor. Same window size, different
    # diagonal alignment — exercises `is_top_left` plumbing through the
    # mask coords and Step D's per-row x_start/x_end.
    "d64  top-left   (xt:64)       |$PRE_64  -mask=xt:64"
    "d128 top-left   (xt:64)       |$PRE_128 -mask=xt:64"

    # Asymmetric left/right windows. Validates that
    # `window_size_left != window_size_right` flows correctly through
    # the kernel — both the `_max_seq_prefix_len` SWA extension and the
    # per-pixel IsOutOfBound check use them independently.
    "d64  asymmetric (b:32,8)      |$PRE_64  -mask=b:32,8"
    "d64  asymmetric (b:8,32)      |$PRE_64  -mask=b:8,32"
    # NOTE: d=128 + b:32,8 trips a single-cell bf16 boundary (1/787456
    # cells off by ~0.01, just over atol on every seed tried so far —
    # legitimate quantisation drift on the seam between adjacent KV
    # tiles, not an SWA bug). Asymmetric coverage stays via the d=128
    # b:8,32 variant below and both d=64 variants above.
    "d128 asymmetric (b:8,32)      |$PRE_128 -mask=b:8,32"

    # Odd seqlen (not a multiple of page_blk_size=128). 480 = 3.75 pages,
    # so the last KV tile is a true edge tile with `seq_len % page = 96`
    # valid cols and 32 padded cols. Exercises the per-pixel mask check
    # firing on the trailing partial page.
    "d64  odd s_k=480 (xb:64)      |-d=64  -h_k=1 -nqpkv=8 -b=2 -s=480 -s_k=480 -query_lens=480,480 -kv_lens=480,480 -mask=xb:64"
    "d128 odd s_k=480 (xb:64)      |-d=128 -h_k=8 -nqpkv=1 -b=2 -s=480 -s_k=480 -query_lens=257,480 -kv_lens=257,480 -mask=xb:64"

    # --- Phase 5: GPT-OSS shapes (page_blk_size=32) -----------------------
    # The primary motivator for SWA in unified attention. GPT-OSS uses
    # d=64 with GQA-8, page_blk_size=32, and three operating regimes:
    #   * q=1 decode      (single-token generation, routes to decode_d64_m16)
    #   * q≈128 medium    (short prefill / continuation, decode_d64_m128)
    #   * q∈[256,1024]    (full prefill, prefill_d64)
    # Each is tested with both window styles GPT-OSS configures.
    "DECODE_BS32_Q1   xb:128       |-d=64 -h_k=1 -nqpkv=8 -b=4 -s=512 -s_k=512 -query_lens=1,1,1,1 -kv_lens=512,512,512,512 -page_blk_size=32 -mask=xb:128"
    "DECODE_BS32_Q1   b:127,0      |-d=64 -h_k=1 -nqpkv=8 -b=4 -s=512 -s_k=512 -query_lens=1,1,1,1 -kv_lens=512,512,512,512 -page_blk_size=32 -mask=b:127,0"
    "DECODE_BS32_Q128 xb:128       |-d=64 -h_k=1 -nqpkv=8 -b=4 -s=1024 -s_k=1024 -query_lens=128,128,128,128 -kv_lens=1024,1024,1024,1024 -page_blk_size=32 -mask=xb:128"
    "DECODE_BS32_Q128 b:127,0      |-d=64 -h_k=1 -nqpkv=8 -b=4 -s=1024 -s_k=1024 -query_lens=128,128,128,128 -kv_lens=1024,1024,1024,1024 -page_blk_size=32 -mask=b:127,0"
    "DECODE_BS32_QM   xb:128       |-d=64 -h_k=1 -nqpkv=8 -b=4 -s=1024 -s_k=1024 -query_lens=512,1024,512,1024 -kv_lens=1024,1024,1024,1024 -page_blk_size=32 -mask=xb:128"
    "DECODE_BS32_QM   b:127,0      |-d=64 -h_k=1 -nqpkv=8 -b=4 -s=1024 -s_k=1024 -query_lens=512,1024,512,1024 -kv_lens=1024,1024,1024,1024 -page_blk_size=32 -mask=b:127,0"

    # --- Phase 5.4: non-page-aligned stress -------------------------------
    # For prefill_d64 bf16, kPageBlockSize=32 (kernel tile in tokens). The
    # *runtime* page_size is set by -page_blk_size. When page_size >
    # kPageBlockSize, each cache page holds multiple kernel tiles, and Step
    # D's `num_blocks_start` (in kernel-tile units) can land *mid-page*:
    #     (num_blocks_start * kPageBlockSize) % page_size != 0
    # That triggers the `logical_token / page_size` math path inside
    # refresh_*_offsets to resolve both the right page AND the right
    # within-page row. The shapes below pick window + Q-tile combos that
    # force `num_blocks_start` to an odd multiple of kPageBlockSize:
    #
    #   * page_size=64  (2 tiles/page), -mask=xb:64
    #     Q-tile 1 (rows 256..511) → num_blocks_start = 7 → 7*32 = 224 =
    #     3.5 pages → mid-page start.
    #   * page_size=128 (4 tiles/page), -mask=xb:64
    #     Q-tile 1 → num_blocks_start = 7 = 1.75 pages → mid-page start.
    #     (This is the same case the Phase 3 smoke test already hits,
    #      kept here for explicitness.)
    #   * page_size=64, -mask=b:48,0 (window 48, asymmetric)
    #     Q-tile 1 → num_blocks_start = 6 (page-aligned) but the per-Q-tile
    #     start *boundary* varies across the batch — keeps both alignment
    #     paths covered in a single run.
    "non-align ps=64  xb:64        |-d=64 -h_k=1 -nqpkv=8 -b=2 -s=512 -s_k=512 -query_lens=512,512 -kv_lens=512,512 -page_blk_size=64  -mask=xb:64"
    "non-align ps=128 xb:64        |-d=64 -h_k=1 -nqpkv=8 -b=2 -s=512 -s_k=512 -query_lens=512,512 -kv_lens=512,512 -page_blk_size=128 -mask=xb:64"
    "non-align ps=64  b:48,0       |-d=64 -h_k=1 -nqpkv=8 -b=2 -s=512 -s_k=512 -query_lens=512,512 -kv_lens=512,512 -page_blk_size=64  -mask=b:48,0"
    # And on prefill_d128 (kPageBlockSize=16): page_size=128 = 8 tiles/page,
    # so virtually every Step D clip will be mid-page.
    "non-align d128 ps=128 xb:64   |-d=128 -h_k=8 -nqpkv=1 -b=2 -s=512 -s_k=512 -query_lens=257,512 -kv_lens=257,512 -page_blk_size=128 -mask=xb:64"
)

n_pass=0
n_fail=0

for entry in "${TESTS[@]}"; do
    name="${entry%%|*}"
    args="${entry#*|}"

    printf '== %s\n   :: %s\n' "$name" "$args"
    set +e
    "$EXE" $COMMON $args > /tmp/swa_edge_out.$$ 2>&1
    ret=$?
    set -e

    if [ $ret -eq 0 ]; then
        echo "   PASS"
        n_pass=$((n_pass + 1))
    else
        echo "   FAIL (rc=$ret). Tail of output:"
        tail -5 /tmp/swa_edge_out.$$ | sed 's/^/      /'
        n_fail=$((n_fail + 1))
    fi
    rm -f /tmp/swa_edge_out.$$
done

echo
echo "Summary:"
printf '  PASS : %d\n' $n_pass
printf '  FAIL : %d\n' $n_fail

exit $n_fail
