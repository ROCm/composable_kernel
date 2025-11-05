// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

namespace ck {
namespace s_prefetch_op_util {

// Prefetch to constant cache using AMD builtin with chunks_to_prefetch(1..32: 1 chunk = 128B)
template <typename T>
__device__ __forceinline__ void prefetch_to_constant_cache(const T* addr,
                                                           unsigned int chunks_to_prefetch)
{
#if defined(__gfx12__)
    assert(chunks_to_prefetch > 0 && chunks_to_prefetch <= 32);
    __builtin_amdgcn_s_prefetch_data(addr, chunks_to_prefetch - 1); // we need to pass 0..31
#else
    // ignore - not supported
    (void)addr;
    (void)chunks_to_prefetch;
#endif
}

// Prefetch to constant cache using AMD builtin with chunks_to_prefetch(1..32: 1 chunk = 128B)
template <unsigned int offset>
__device__ __forceinline__ void prefetch_to_constant_cache(__amdgpu_buffer_rsrc_t buf_res,
                                                           unsigned int chunks_to_prefetch)
{
#if defined(__gfx12__)
    assert(chunks_to_prefetch > 0 && chunks_to_prefetch <= 32);
    __builtin_amdgcn_s_buffer_prefetch_data(buf_res, offset, chunks_to_prefetch - 1);
#else
    // ignore - not supported
    (void)buf_res;
    (void)chunks_to_prefetch;
#endif
}

template <typename T>
__global__ void kernel_with_scalar_prefetch(const T* src,
                                            T* dst,
                                            const void CK_CONSTANT_ADDRESS_SPACE* scalar_data,
                                            index_t num_elements,
                                            index_t num_scalars)
{
    index_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    const T CK_CONSTANT_ADDRESS_SPACE* scalar_elems =
        static_cast<const T CK_CONSTANT_ADDRESS_SPACE*>(scalar_data);

    // Calculate number of 128B chunks needed to cover num_scalars elements
    constexpr index_t chunk_size_bytes   = 128;
    constexpr index_t elements_per_chunk = chunk_size_bytes / sizeof(T);
    unsigned int chunks_needed = (num_scalars + elements_per_chunk - 1) / elements_per_chunk;

    // Prefetch all scalar data at once using chunks parameter
    if(threadIdx.x == 0)
    {
        prefetch_to_constant_cache(scalar_elems, chunks_needed);
    }

    T sum = 0;
    if(tid < num_elements)
    {
        sum = src[tid]; // load from global mem to make sure prefetch finished
    }
    __syncthreads(); // waits on loads from global mem
    if(tid < num_elements)
    {
        // Access prefetched scalar data
        for(index_t i = 0; i < num_scalars; i++)
        {
            sum += scalar_elems[i]; // should be fast due to scalars being preloaded
        }

        dst[tid] = sum;
    }
}

template <typename T>
__global__ void
kernel_with_scalar_buffer_prefetch(const T* src,
                                   T* dst,
                                   const void CK_CONSTANT_ADDRESS_SPACE* scalar_data,
                                   index_t num_elements,
                                   index_t num_scalars)
{
    index_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    const T CK_CONSTANT_ADDRESS_SPACE* scalar_elems =
        static_cast<const T CK_CONSTANT_ADDRESS_SPACE*>(scalar_data);

    // Calculate number of 128B chunks needed to cover num_scalars elements
    constexpr index_t chunk_size_bytes   = 128;
    constexpr index_t elements_per_chunk = chunk_size_bytes / sizeof(T);
    unsigned int chunks_needed = (num_scalars + elements_per_chunk - 1) / elements_per_chunk;

    __amdgpu_buffer_rsrc_t src_wave_buffer_resource =
        make_wave_buffer_resource_new(scalar_elems, num_scalars);

    // Prefetch all scalar data at once using chunks parameter
    if(threadIdx.x == 0)
    {
        prefetch_to_constant_cache<0>(src_wave_buffer_resource, chunks_needed);
    }

    T sum = 0;
    if(tid < num_elements)
    {
        sum = src[tid]; // load from global mem to make sure prefetch finished
    }
    __syncthreads(); // waits on loads from global mem
    if(tid < num_elements)
    {
        // Access prefetched scalar data
        for(index_t i = 0; i < num_scalars; i++)
        {
            sum += amd_s_buffer_load_impl<T, 1>(
                src_wave_buffer_resource,
                i * sizeof(T)); // should be fast due to scalars being preloaded
        }

        dst[tid] = sum;
    }
}

template <typename PrefetchKernel, typename T>
bool test_constant_prefetch_impl(const PrefetchKernel& prefetch_kernel,
                                 const std::string& kernel_name)
{
    constexpr index_t num_elements = 512;
    constexpr index_t num_scalars  = 512;
    constexpr index_t block_size   = 256;
    constexpr index_t grid_size    = (num_elements + block_size - 1) / block_size;

    std::cout << "Testing " << kernel_name << " to constant cache for type: " << typeid(T).name()
              << std::endl;
    std::cout << "Elements: " << num_elements << ", Scalars: " << num_scalars << std::endl;

    // Host data
    std::vector<T> h_src(num_elements);
    std::vector<T> h_scalar(num_scalars);
    std::vector<T> h_dst_with_prefetch_chunks(num_elements);
    std::vector<T> h_expected(num_elements);

    // Initialize data
    for(index_t i = 0; i < num_elements; i++)
    {
        h_src[i] = static_cast<T>(i % 100);
    }

    T scalar_sum = 0;
    for(index_t i = 0; i < num_scalars; i++)
    {
        h_scalar[i] = static_cast<T>(i + 1);
        scalar_sum += h_scalar[i];
    }

    // Expected results
    for(index_t i = 0; i < num_elements; i++)
    {
        h_expected[i] = h_src[i] + scalar_sum;
    }

    // Device memory
    DeviceMem d_src(sizeof(T) * num_elements);
    DeviceMem d_scalar(sizeof(T) * num_scalars);
    DeviceMem d_dst_with_prefetch_chunks(sizeof(T) * num_elements);

    d_src.ToDevice(h_src.data());
    d_scalar.ToDevice(h_scalar.data());

    hipStream_t stream;
    hip_check_error(hipStreamCreate(&stream));

    prefetch_kernel<<<grid_size, block_size, 0, stream>>>(
        static_cast<const T*>(d_src.GetDeviceBuffer()),
        static_cast<T*>(d_dst_with_prefetch_chunks.GetDeviceBuffer()),
        cast_pointer_to_constant_address_space(d_scalar.GetDeviceBuffer()),
        num_elements,
        num_scalars);

    hip_check_error(hipStreamSynchronize(stream));

    // Copy results back
    d_dst_with_prefetch_chunks.FromDevice(h_dst_with_prefetch_chunks.data());

    // Verify results
    bool pass = ck::utils::check_err(h_dst_with_prefetch_chunks, h_expected);

    std::cout << (pass ? "PASS" : "FAIL") << std::endl;

    hip_check_error(hipStreamDestroy(stream));

    return pass;
}

} // namespace s_prefetch_op_util
} // namespace ck
