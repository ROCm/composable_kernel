add_executable(example_batchnorm_simple EXCLUDE_FROM_ALL batchnorm_simple.cpp)
set(EXAMPLE_BATCHNORM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_BATCHNORM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()

target_compile_options(example_batchnorm_simple PRIVATE ${EXAMPLE_BATCHNORM_COMPILE_OPTIONS})

# Full version (to be implemented later)
# add_executable(example_batchnorm EXCLUDE_FROM_ALL batchnorm.cpp)
# target_compile_options(example_batchnorm PRIVATE ${EXAMPLE_BATCHNORM_COMPILE_OPTIONS})
