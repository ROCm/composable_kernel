# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ck_tile_gemm_split_trait(<trait> <out_pipeline> <out_epilogue> <out_scheduler>)
#
# Split a Tile Engine GEMM trait-combo string
#   <pipeline>_<epilogue>_<scheduler>_<pad_m>_<pad_n>_<pad_k>[_<persistent>]
# into its pipeline / epilogue / scheduler fields. Multi-token pipeline names
# (comp_async_eight_waves, comp_tdm_v2, comp_async, comp_tdm, weight_preshuffle)
# are matched first; every other trait falls back to the legacy "_" split, so
# existing trait strings resolve exactly as before.
#
# Python mirror: trait_parse.py (split_trait).
function(ck_tile_gemm_split_trait trait out_pipeline out_epilogue out_scheduler)
    set(_ck_rest "${trait}")
    set(_ck_pipeline "")
    if("${trait}" MATCHES "^(comp_async_eight_waves|comp_tdm_v2|comp_async|comp_tdm|weight_preshuffle)_(.*)$")
        set(_ck_pipeline "${CMAKE_MATCH_1}")
        set(_ck_rest "${CMAKE_MATCH_2}")
    endif()

    string(REPLACE "_" ";" _ck_parts "${_ck_rest}")
    if(_ck_pipeline STREQUAL "")
        list(GET _ck_parts 0 _ck_pipeline)
        list(REMOVE_AT _ck_parts 0)
    endif()
    list(GET _ck_parts 0 _ck_epilogue)
    list(GET _ck_parts 1 _ck_scheduler)

    set(${out_pipeline} "${_ck_pipeline}" PARENT_SCOPE)
    set(${out_epilogue} "${_ck_epilogue}" PARENT_SCOPE)
    set(${out_scheduler} "${_ck_scheduler}" PARENT_SCOPE)
endfunction()
