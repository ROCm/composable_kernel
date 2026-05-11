# Tutorial 08: Simple LDS Staging
# Demonstrates basic LDS (Local Data Share / shared memory) usage
# Direct continuation of Tutorial 07 with minimal changes

# Create executable for LDS staging tutorial
add_executable(aa_tutorial_08_lds_staging simple_lds_staging.cpp)

# Set properties
target_include_directories(aa_tutorial_08_lds_staging PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Message for build output
message(STATUS "Added Tutorial 08: Simple LDS Staging - Global -> LDS -> Registers -> MFMA with basic shared memory")
