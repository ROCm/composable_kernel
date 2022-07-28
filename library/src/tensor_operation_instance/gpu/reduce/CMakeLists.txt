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
   device_reduce_instance_multiblock_atomic_add_f16_f32_f32.cpp;
   device_reduce_instance_multiblock_atomic_add_f32_f32_f32.cpp;
   device_reduce_instance_multiblock_atomic_add_f32_f64_f32.cpp;
   device_reduce_instance_multiblock_atomic_add_f64_f64_f64.cpp;
   device_reduce_instance_multiblock_atomic_add_b16_f32_f32.cpp;
)

add_instance_library(device_reduce_instance OBJECT ${DEVICE_REDUCE_INSTANCE_SOURCE})
