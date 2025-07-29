add_executable(copy_basic EXCLUDE_FROM_ALL copy_basic.cpp)

target_compile_options(copy_basic PRIVATE
  -mllvm -enable-noalias-to-md-conversion=0
)