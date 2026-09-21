#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Smoke test for tile_example_mx_flatmm.
#
# Sweeps mx_prec x warp_tile x (M, N, K) shapes and writes one log per combo so
# subsequent invocations don't overwrite earlier results.
#
# Optional environment overrides:
#   EXE       absolute path to tile_example_mx_flatmm (default: discovered via `find .`)
#   WORKDIR   directory to run in / write logs to (default: $PWD)

set -u

EXE="${EXE:-$(find . -name tile_example_mx_flatmm -type f | head -n 1)}"
WORKDIR="${WORKDIR:-$PWD}"

if [[ -z "$EXE" || ! -x "$EXE" ]]; then
    echo "ERROR: could not locate tile_example_mx_flatmm executable." >&2
    echo "       Set EXE=/abs/path/tile_example_mx_flatmm and re-run." >&2
    exit 2
fi

cd "$WORKDIR" || { echo "ERROR: cannot cd to WORKDIR=$WORKDIR" >&2; exit 2; }

echo "EXE     : $EXE"
echo "WORKDIR : $WORKDIR"
echo

COMMON='-v=1 -init=0 -warmup=0 -repeat=1 -verbose=1'

PRECS=(fp8xfp8 fp4xfp4 fp4xfp8 fp8xfp4 fp6xfp6)
WARP_TILES=(0 1)

# Shape constraints per active FlatmmConfig (kPad{M,N,K}=false everywhere):
#
#   (prec, wt)               Config (mx_flatmm_arch_traits.hpp)   M_Tile N_Tile K_Tile
#   ----------------------   ----------------------------------   ------ ------ ------
#   (fp4xfp4,         0)     MXfp4_FlatmmConfig16                  128    512    256
#   ({fp8,fp6,mixed}, 0)     MXFlatmmConfigBase16                  128    256    256
#   ({fp4,fp8,mixed}, 1)     MXFlatmmConfigBase32TDM               128    256    256
#
# Default shapes satisfy N_Tile=256 (M%128, N%256, K%256). The fp4xfp4 wt=0
# path needs N%512 since MXfp4_FlatmmConfig16 doubles N_Tile.
SHAPES_DEFAULT=(
    "128 256 256"
    "512 768 1024"
)
SHAPES_FP4FP4_WT0=(
    "128 512 256"
    "512 1024 1024"
)

declare -a RESULTS

run_one() {
    local prec=$1 wt=$2 M=$3 N=$4 K=$5
    local log="mx_flatmm_${prec}_wt${wt}_m${M}_n${N}_k${K}.log"
    local label="prec=${prec} wt=${wt} M=${M} N=${N} K=${K}"

    echo "[RUN ] $label -> $log"
    "$EXE" -m=$M -n=$N -k=$K -mx_prec=$prec -warp_tile=$wt $COMMON >& "$log"
    local rc=$?

    if [[ $rc -eq 0 ]]; then
        echo "[PASS] $label"
        RESULTS+=("PASS  $label  log=$log")
    else
        echo "[FAIL] $label  (rc=$rc)"
        RESULTS+=("FAIL  $label  rc=$rc  log=$log")
    fi
    return $rc
}

fails=0
total=0
for prec in "${PRECS[@]}"; do
    for wt in "${WARP_TILES[@]}"; do
        # FP6 is not supported on the GFX1250 TDM pipeline (warp_tile=1).
        if [[ "$prec" == "fp6xfp6" && "$wt" == "1" ]]; then
            echo "[SKIP] prec=$prec wt=$wt (TDM does not support FP6)"
            RESULTS+=("SKIP  prec=$prec wt=$wt  (TDM does not support FP6)")
            continue
        fi
        # Pick shape set matching the active FlatmmConfig for this (prec, wt).
        if [[ "$prec" == "fp4xfp4" && "$wt" == "0" ]]; then
            shapes=("${SHAPES_FP4FP4_WT0[@]}")
        else
            shapes=("${SHAPES_DEFAULT[@]}")
        fi
        for shape in "${shapes[@]}"; do
            total=$((total + 1))
            # shellcheck disable=SC2086
            run_one "$prec" "$wt" $shape || fails=$((fails + 1))
        done
    done
done

echo
echo "===== smoke_test_mx summary ====="
for line in "${RESULTS[@]}"; do
    echo "$line"
done
echo "---------------------------------"
echo "Total: $total   Failures: $fails"
echo "================================="

exit $fails
