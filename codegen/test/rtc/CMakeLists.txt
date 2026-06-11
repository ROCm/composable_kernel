# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

option(USE_HIPRTC_FOR_CODEGEN_TESTS "Whether to enable hipRTC for codegen tests." ON)

find_package(hip REQUIRED)
if(USE_HIPRTC_FOR_CODEGEN_TESTS)
    find_package(hiprtc REQUIRED)
endif()

file(GLOB RTC_SOURCES CONFIGURE_DEPENDS src/*.cpp)
add_library(ck_rtc ${RTC_SOURCES})
target_include_directories(ck_rtc PUBLIC include)
target_link_libraries(ck_rtc PUBLIC hip::host -lstdc++fs)

if(USE_HIPRTC_FOR_CODEGEN_TESTS)
    target_link_libraries(ck_rtc PUBLIC hiprtc::hiprtc)
    target_compile_definitions(ck_rtc PUBLIC HIPRTC_FOR_CODEGEN_TESTS)
    message(STATUS "CK codegen tests: hipRTC enabled")
else()
    message(STATUS "CK codegen tests: hipRTC disabled")
endif()
