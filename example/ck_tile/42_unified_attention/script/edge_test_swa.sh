#!/bin/bash
# edge_test_swa.sh - Numerical edge cases for SWA in CK-tile unified attention.
#
# Tests the corner cases that random-shape sweeps in smoke_test_swa.sh miss:
#   - window=1 (every Q row attends to its own K position only)
#   - window > seq_k with right=0 (degenerates to plain causal)
#   - explicit b:0,0 (alternative spelling of diagonal-only)
#   - decode shapes (q=1, kv>>1) — exercises the SWA path on a single-token Q.
#     For page_blk_size>=64 (Edge 4) we route to the large-tier kernel, which is
#     wasteful but correct. For page_blk_size==32 (Edge 5) we route to the
#     decode-tier IsLocal=true instances added in Phase 5 — that's the GPT-OSS
#     production path.
#
# Same convention as smoke_test_swa.sh: every test must pass, exit code is the
# number of failures.

set -uo pipefail

EXE_NAME=tile_example_unified_attention
EXE="${EXE:-$(find . -name $EXE_NAME -type f | head -n 1)}"
if [ -z "${EXE:-}" ] || [ ! -x "$EXE" ]; then
    echo "ERROR: $EXE_NAME not found. Set EXE=/path/to/$EXE_NAME or run from build dir." >&2
    exit 2
fi

# Same deterministic fixture as smoke_test_swa.sh.
COMMON="-prec=bf16 -seed=17 -verify=1 -warmup=0 -repeat=1 -varlen=0 -nb=1024 -page_blk_size=128"

BASELINE_A="-d=128 -h_k=8 -nqpkv=1 -b=4 -s=512 -s_k=512 -query_lens=128,128,128,128 -kv_lens=128,128,128,128"
BASELINE_B="-d=64  -h_k=1 -nqpkv=8 -b=4 -s=512 -s_k=512 -query_lens=400,256,512,128 -kv_lens=400,256,512,128"

# Decode-shape fixtures (q=1, kv=512). SWA dispatcher forces tile_tier::large so
# these go through the prefill kernel even though they're decode shapes — the
# point of these tests is that the large-tier IsLocal=true kernel still produces
# correct numerics on a single-token query.
DECODE_A="-d=128 -h_k=8 -nqpkv=1 -b=4 -s=1 -s_k=512 -query_lens=1,1,1,1 -kv_lens=512,512,512,512"
DECODE_B="-d=64  -h_k=1 -nqpkv=8 -b=4 -s=1 -s_k=512 -query_lens=1,1,1,1 -kv_lens=512,512,512,512"

# Decode + page_blk_size=32 fixtures for d=64 GQA-8. These exercise the NEW
# decode-tier IsLocal=true instances added in Phase 5:
#   - DECODE_BS32_Q1 (q=1)            → tiny+bs32 local (kBlockM=16, kBlockQ=2)
#   - DECODE_BS32_QM (q in [256,1024]) → medium+bs32 local (kBlockM=128, kBlockQ=16)
# Use the GPT-OSS-shaped window (left=127, right=0) to mirror the production
# workload that motivated Phase 5.
DECODE_BS32_Q1="-d=64 -h_k=1 -nqpkv=8 -b=4 -s=1 -s_k=512 -query_lens=1,1,1,1 -kv_lens=512,512,512,512"
DECODE_BS32_QM="-d=64 -h_k=1 -nqpkv=8 -b=2 -s=1024 -s_k=1024 -query_lens=1024,512 -kv_lens=1024,512"

TESTS=(
    # Edge 1: window=1 — diagonal-only attention. Smallest non-zero window.
    #         xb:1 decodes to left=0, right=0 via window/2 split.
    "baseA xb:1           |$BASELINE_A -mask=xb:1"
    "baseB xb:1           |$BASELINE_B -mask=xb:1"

    # Edge 2: window > seq_k with right=0 — IsLocal=true kernel must still
    #         produce the same answer as the IsLocal=false causal kernel
    #         (verified independently against the reference).
    "baseA b:8192,0       |$BASELINE_A -mask=b:8192,0"
    "baseB b:8192,0       |$BASELINE_B -mask=b:8192,0"

    # Edge 3: alternative diagonal-only spelling via explicit b:0,0.
    #         Must produce identical numerics to xb:1 above.
    "baseA b:0,0          |$BASELINE_A -mask=b:0,0"
    "baseB b:0,0          |$BASELINE_B -mask=b:0,0"

    # Edge 4: decode shapes (single-token query). The SWA mask trims the K range
    #         to a 64-wide window at the bottom-right corner of the (1, 512)
    #         attention matrix, so most of the kv tail is masked out.
    "decode q=1 d128 xb:64    |$DECODE_A -mask=xb:64"
    "decode q=1 d64  xb:64    |$DECODE_B -mask=xb:64"

    # Edge 5 (Phase 5): GPT-OSS-shaped d64 GQA-8 SWA on page_blk_size=32. These
    # MUST hit the new decode-tier IsLocal=true instances; if a regression takes
    # them back to the bs64-only fallback they fail with "no matching kernel
    # instance" or wrong numerics.
    "decode q=1 d64 bs32 b:127,0    |$DECODE_BS32_Q1 -page_blk_size=32 -mask=b:127,0"
    "decode q=1 d64 bs32 xb:128     |$DECODE_BS32_Q1 -page_blk_size=32 -mask=xb:128"
    "shortpf  d64 bs32 b:127,0      |$DECODE_BS32_QM -page_blk_size=32 -mask=b:127,0"
    "shortpf  d64 bs32 xb:128       |$DECODE_BS32_QM -page_blk_size=32 -mask=xb:128"
)

n_pass=0
n_fail=0

for entry in "${TESTS[@]}"; do
    name="${entry%%|*}"
    args="${entry#*|}"

    printf '== %-32s :: %s\n' "$name" "$args"
    set +e
    "$EXE" $COMMON $args > /tmp/swa_edge_out.$$ 2>&1
    ret=$?
    set -e

    if [ $ret -eq 0 ]; then
        echo "   PASS"
        n_pass=$((n_pass + 1))
    else
        echo "   FAIL (rc=$ret). Tail of output:"
        tail -3 /tmp/swa_edge_out.$$ | sed 's/^/      /'
        n_fail=$((n_fail + 1))
    fi
    rm -f /tmp/swa_edge_out.$$
done

echo
echo "Summary:"
printf '  passed : %d\n' $n_pass
printf '  failed : %d\n' $n_fail

exit $n_fail
