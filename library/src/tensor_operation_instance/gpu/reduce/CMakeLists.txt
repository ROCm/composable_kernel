# device_reduce_instance
set(DEVICE_REDUCE_INSTANCE_SOURCE
   device_reduce_instance_blockwise_f16_f16_f16.cpp;
   device_reduce_instance_blockwise_f16_f32_f16.cpp;
   device_reduce_instance_blockwise_f32_f32_f32.cpp;
   device_reduce_instance_blockwise_f32_f64_f32.cpp;
   device_reduce_instance_blockwise_f64_f64_f64.cpp;
   device_reduce_instance_blockwise_i8_i32_i8.cpp;
   device_reduce_instance_blockwise_i8_i8_i8.cpp;   
   device_reduce_instance_blockwise_b16_f32_b16.cpp;
   device_reduce_instance_threadwise_f16_f16_f16.cpp;
   device_reduce_instance_threadwise_f16_f32_f16.cpp;
   device_reduce_instance_threadwise_f32_f32_f32.cpp;
   device_reduce_instance_threadwise_f32_f64_f32.cpp;
   device_reduce_instance_threadwise_f64_f64_f64.cpp;
   device_reduce_instance_threadwise_i8_i32_i8.cpp;
   device_reduce_instance_threadwise_i8_i8_i8.cpp;
   device_reduce_instance_threadwise_b16_f32_b16.cpp;
   device_reduce_instance_blockwise_second_call_f16_f16_f16.cpp;
   device_reduce_instance_blockwise_second_call_f32_f32_f16.cpp;
   device_reduce_instance_blockwise_second_call_f32_f32_f32.cpp;
   device_reduce_instance_blockwise_second_call_f64_f64_f32.cpp;
   device_reduce_instance_blockwise_second_call_f64_f64_f64.cpp;
   device_reduce_instance_blockwise_second_call_i32_i32_i8.cpp;
   device_reduce_instance_blockwise_second_call_i8_i8_i8.cpp;
   device_reduce_instance_blockwise_second_call_f32_f32_b16.cpp;
   device_reduce_instance_multiblock_atomic_add_f16_f32_f32.cpp;
   device_reduce_instance_multiblock_atomic_add_f32_f32_f32.cpp;
   device_reduce_instance_multiblock_atomic_add_f32_f64_f32.cpp;
   device_reduce_instance_multiblock_atomic_add_b16_f32_f32.cpp;
   device_reduce_instance_multiblock_partial_reduce_f16_f16_f16.cpp;
   device_reduce_instance_multiblock_partial_reduce_f16_f32_f16.cpp;
   device_reduce_instance_multiblock_partial_reduce_f32_f32_f32.cpp;
   device_reduce_instance_multiblock_partial_reduce_f32_f64_f32.cpp;
   device_reduce_instance_multiblock_partial_reduce_f64_f64_f64.cpp;
   device_reduce_instance_multiblock_partial_reduce_i8_i32_i8.cpp;
   device_reduce_instance_multiblock_partial_reduce_i8_i8_i8.cpp;
   device_reduce_instance_multiblock_partial_reduce_b16_f32_b16.cpp;
)

add_library(device_reduce_instance SHARED ${DEVICE_REDUCE_INSTANCE_SOURCE}) 
target_compile_features(device_reduce_instance PUBLIC)
set_target_properties(device_reduce_instance PROPERTIES POSITION_INDEPENDENT_CODE ON)
install(TARGETS device_reduce_instance LIBRARY DESTINATION lib) 

clang_tidy_check(device_reduce_instance)
