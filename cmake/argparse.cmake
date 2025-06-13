include(FetchContent)

set(ARGPARSE_DIR "" CACHE STRING "Location of local argparse repo to build against")

if(ARGPARSE_DIR)
    set(FETCHCONTENT_SOURCE_DIR_ARGPARSE ${ARGPARSE_DIR} CACHE STRING "argparse source directory override")
endif()

FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
    GIT_TAG v2.9 # Using a stable version
)

FetchContent_MakeAvailable(argparse)

# Create a proper modern CMake INTERFACE target
if(NOT TARGET argparse::argparse)
    add_library(argparse::argparse INTERFACE)
    target_include_directories(argparse::argparse
        INTERFACE
        $<BUILD_INTERFACE:${argparse_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )

    # Disable the specific warning for code that includes argparse
    target_compile_options(argparse::argparse INTERFACE
        $<$<CXX_COMPILER_ID:Clang,AppleClang,GNU>:-Wno-missing-noreturn>
    )
endif()

message(STATUS "Argparse source dir: ${argparse_SOURCE_DIR}")
