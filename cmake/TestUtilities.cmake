# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Helper function to conditionally link device_conv libraries only if they exist as targets.
# This is useful when device_conv libraries may be filtered out based on GPU targets,
# DTYPES, or build configuration flags.
#
# Usage:
#   target_link_device_conv_libraries_if_exist(my_test PRIVATE utility device_conv2d_nhwgc_operations ...)
#
# Only device_conv* libraries are checked with if(TARGET).
# All other libraries (utility, gtest_main, etc.) are always linked.
function(target_link_device_conv_libraries_if_exist TARGET_NAME VISIBILITY)
    set(_libs_to_link)
    foreach(lib ${ARGN})
        if(lib MATCHES "^device_conv")
            # Only add device_conv libraries if they exist
            if(TARGET ${lib})
                list(APPEND _libs_to_link ${lib})
            endif()
        else()
            # Always add non-device_conv libraries
            list(APPEND _libs_to_link ${lib})
        endif()
    endforeach()
    # Single target_link_libraries call with all libraries
    if(_libs_to_link)
        target_link_libraries(${TARGET_NAME} ${VISIBILITY} ${_libs_to_link})
    endif()
endfunction()
