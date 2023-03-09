set(CONV2D_PERLAYER_QUANT_SRC
    conv2d_fwd/device_conv2d_dl_perlayer_quantization_int8_instance.cpp
    conv2d_fwd/device_conv2d_xdl_perlayer_quantization_int8_instance.cpp
)

set(CONV2D_PERCHANNEL_QUANT_SRC
    conv2d_fwd/device_conv2d_dl_perchannel_quantization_int8_instance.cpp
    conv2d_fwd/device_conv2d_xdl_perchannel_quantization_int8_instance.cpp
)

set(CONV2D_BIAS_PERLAYER_QUANT_SRC
    conv2d_fwd/device_conv2d_dl_bias_perlayer_quantization_int8_instance.cpp
    conv2d_fwd/device_conv2d_xdl_bias_perlayer_quantization_int8_instance.cpp
)

set(CONV2D_BIAS_PERCHANNEL_QUANT_SRC
    conv2d_fwd/device_conv2d_dl_bias_perchannel_quantization_int8_instance.cpp
    conv2d_fwd/device_conv2d_xdl_bias_perchannel_quantization_int8_instance.cpp
)


add_instance_library(device_quantization_instance
    ${CONV2D_PERLAYER_QUANT_SRC}
    ${CONV2D_PERCHANNEL_QUANT_SRC}
    ${CONV2D_BIAS_PERLAYER_QUANT_SRC}
    ${CONV2D_BIAS_PERCHANNEL_QUANT_SRC}
)
