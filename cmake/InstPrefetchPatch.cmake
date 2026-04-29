# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Two-pass instruction-prefetch offset patching
#
# Requires:  -DENABLE_INST_PREFETCH_PATCH=ON
#
# Usage in a target CMakeLists.txt (after add_executable / add_example_executable):
#   ck_inst_prefetch_patch(<target-name>)
#
# The PRE_LINK hook runs patch_prefetch_offset.py with --skip-build-round1
# (round 1 = the normal cmake build that just compiled the .o) so the script
# parses the .s file, patches the .o in-place, and the linker then consumes
# the patched .o.  PRE_LINK fires after all sources are compiled but BEFORE
# linking, which is critical — POST_BUILD would run after linking, so the
# executable would contain the original unpatched koffset=0 values.
# ---------------------------------------------------------------------------
if(NOT DEFINED CK_INST_PREFETCH_PATCH_DEFINED)
    set(CK_INST_PREFETCH_PATCH_DEFINED TRUE)
    option(ENABLE_INST_PREFETCH_PATCH
        "Enable two-pass s_prefetch_inst_pc_rel offset patching."
        OFF)
    option(INST_PREFETCH_PATCH_DUMP_INTERMEDIATES
        "Dump intermediate files (merged tables, objdump text) during prefetch patching."
        OFF)

    function(ck_inst_prefetch_patch TARGET_NAME)
        if(NOT ENABLE_INST_PREFETCH_PATCH)
            return()
        endif()
        # Ensure the target produces .s files needed by the patching script.
        target_compile_options(${TARGET_NAME} PRIVATE --save-temps -Wno-gnu-line-marker)
        set(_log_file "${CMAKE_BINARY_DIR}/prefetch_patch_${TARGET_NAME}.log")
        cmake_host_system_information(RESULT _nproc QUERY NUMBER_OF_LOGICAL_CORES)
        set(_extra_args "")
        if(INST_PREFETCH_PATCH_DUMP_INTERMEDIATES)
            list(APPEND _extra_args --dump-intermediates)
        endif()
        add_custom_command(
            TARGET  ${TARGET_NAME}
            PRE_LINK
            COMMAND ${CMAKE_COMMAND} -E echo
                    "[inst-prefetch-patch] Running patch_prefetch_offset.py for ${TARGET_NAME} (log: ${_log_file})"
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_SOURCE_DIR}/script/patch_prefetch_offset.py
                    --build-dir      ${CMAKE_BINARY_DIR}
                    --target         ${TARGET_NAME}
                    --jobs           ${_nproc}
                    --skip-build-round1
                    ${_extra_args}
            COMMENT "[inst-prefetch-patch] Patching prefetch offsets for ${TARGET_NAME} — log: ${_log_file}"
            VERBATIM
        )
    endfunction()
endif()
