#include <hip/hip_runtime.h>

#include <iostream>
#include <vector>

// clang-format off
// /opt/rocm/llvm/bin/clang++ -O3 -x hip --save-temps --offload-arch=gfx950 -o test-f8f4 test-f8f4.cpp && ./test-f8f4
// clang-format on

#define HIP_CHECK(call)                                                                    \
    do                                                                                     \
    {                                                                                      \
        hipError_t err = call;                                                             \
        if(err != hipSuccess)                                                              \
        {                                                                                  \
            printf("HIP error %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(err)); \
            exit(1);                                                                       \
        }                                                                                  \
    } while(0)

using fp16_t = _Float16;

template <int pk_size>
struct pk_f6_t
{
    static constexpr int num_bits_elem     = 6;
    using element_type                     = uint32_t; // element storage fundamental type
    static constexpr int packed_size       = pk_size;
    static constexpr int num_bits_vec_elem = sizeof(element_type) * 8; // 32-bit uint for storage
    static_assert((packed_size * num_bits_elem) % num_bits_vec_elem == 0,
                  "Packed elements must fit exactly into the element storage.");
    static constexpr int vector_size = (packed_size * num_bits_elem) / num_bits_vec_elem;
    // using storage_type = element_type __attribute__((ext_vector_type(vector_size)));
    // storage_type data_{storage_type(0)}; // packed data
    element_type data_[vector_size]; // packed data
    using type = pk_f6_t<packed_size>;
    void pack(const uint32_t x, const int i)
    {
        uint32_t bits        = static_cast<uint32_t>(x) & 0x3F;
        const int bit_pos    = i * num_bits_elem;
        const int arr_index  = bit_pos / num_bits_vec_elem;
        const int bit_offset = bit_pos % num_bits_vec_elem;
        const int overhang   = bit_offset + num_bits_elem - num_bits_vec_elem;
        uint32_t old_value   = data_[arr_index];

        // insert bits into the current 32-bit block
        old_value |= (bits << bit_offset);
        data_[arr_index] = old_value;

        // if it crosses into the next block, shift the remainder
        if(overhang > 0 && (arr_index + 1) < vector_size)
        {
            uint32_t next_value = data_[arr_index + 1];
            next_value |= (bits >> (num_bits_elem - overhang));
            data_[arr_index + 1] = next_value;
        }
    }

    template <typename type>
    static inline uint32_t unpack(const type& pk, const int i)
    {
        const int bit_pos    = i * num_bits_elem;
        const int arr_idx    = bit_pos / num_bits_vec_elem;
        const int bit_offset = bit_pos % num_bits_vec_elem;
        const int overhang   = bit_offset + num_bits_elem - num_bits_vec_elem;

        uint32_t bits = pk.data_[arr_idx] >> bit_offset;
        if(overhang > 0 && (arr_idx + 1) < vector_size)
        {
            bits |= (pk.data_[arr_idx + 1] & ((1u << overhang) - 1)) << (num_bits_elem - overhang);
        }

        return bits & 0x3F;
    }

    inline uint32_t unpack(const int i) const { return unpack(*this, i); }

    static float fp6_e2m3_to_float(uint32_t fp6_bits)
    {
        fp6_bits = fp6_bits & 0x3F;

        uint32_t sign     = (fp6_bits >> 5) & 0x1; // bit 5
        uint32_t exponent = (fp6_bits >> 3) & 0x3; // bits 4-3
        uint32_t mantissa = fp6_bits & 0x7;        // bits 2-0

        float result;
        if(exponent == 0 && mantissa == 0)
        {
            result = 0.f;
        }
        else if(exponent != 0)
        {
            result               = std::pow(2, exponent - 1);
            float mantissa_value = 1.0f + mantissa / 8.0f;
            result *= mantissa_value;
        }
        else
        {
            result = mantissa / 8.0f;
        }
        return sign == 1 ? -1 * result : result;
    }
};

using f6x16_pk_t = pk_f6_t<16>;

__global__ void kernel1(const int32_t* a, const int32_t* b, float* c)
{
    const int l             = threadIdx.x;
    using i32x8_t           = int32_t __attribute__((ext_vector_type(8)));
    int k_dim_offset        = l / 16 * 6;
    int mn_dim_offset       = l % 16;
    int total_k_dim_dw_size = 128 * 6 / 8 / 4;
    int thr_base_offset     = mn_dim_offset * total_k_dim_dw_size + k_dim_offset;
    // clang-format off
    i32x8_t a_vec{a[thr_base_offset],a[thr_base_offset+1],a[thr_base_offset+2],a[thr_base_offset+3],a[thr_base_offset+4],a[thr_base_offset+5],0,0};
    i32x8_t b_vec{b[thr_base_offset],b[thr_base_offset+1],b[thr_base_offset+2],b[thr_base_offset+3],b[thr_base_offset+4],b[thr_base_offset+5],0,0};
    // clang-format on

    // printf("thread_idx: %d, base_offset: %d,  value: %d %d %d %d %d %d\n",
    //        l,
    //        thr_base_offset,
    //        a[thr_base_offset],
    //        a[thr_base_offset + 1],
    //        a[thr_base_offset + 2],
    //        a[thr_base_offset + 3],
    //        a[thr_base_offset + 4],
    //        a[thr_base_offset + 5]);

    using fp32x4_t = float __attribute__((ext_vector_type(4)));
    fp32x4_t c_vec{0};
    c_vec =
        __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a_vec, b_vec, c_vec, 2, 2, 0, 127, 0, 127);

    // printf("thread_idx: %d, base_offset: %d, float value: %f %f %f %f\n",
    //        l,
    //        thr_base_offset,
    //        c_vec[0],
    //        c_vec[1],
    //        c_vec[2],
    //        c_vec[3]);
    int c_m               = l % 16;
    int c_n               = l / 16 * 4;
    c[c_m * 16 + c_n + 0] = c_vec[0], c[c_m * 16 + c_n + 1] = c_vec[1];
    c[c_m * 16 + c_n + 2] = c_vec[2], c[c_m * 16 + c_n + 3] = c_vec[3];
}

int main(int argc, char const* argv[])
{

    f6x16_pk_t h_a[16 * (128 / 16)];
    f6x16_pk_t h_b[16 * (128 / 16)];

    float ref_a[16 * 128];
    float ref_b[16 * 128];
    std::vector<float> h_c(16 * 16);
    std::vector<float> h_cc(16 * 16);

    for(int i = 0; i < 16; i++)
    {
        for(int j = 0; j < 128; j += 16)
        {
            for(int k = 0; k < 16; k++)
            {
                uint32_t value = rand() & 0x3f;
                h_a[i * (128 / 16) + j / 16].pack(value, k);
                h_b[i * (128 / 16) + j / 16].pack(value, k);
                ref_a[i * 128 + j + k] = f6x16_pk_t::fp6_e2m3_to_float(value);
                ref_b[i * 128 + j + k] = f6x16_pk_t::fp6_e2m3_to_float(value);
                // std::cout << ref_a[i * 128 + j + k] << "vs"
                //           << f6x16_pk_t::fp6_e2m3_to_float(h_a[i * (128 / 16) + j /
                //           16].unpack(k))
                //           << std::endl;
            }
        }
    }

    for(int m = 0; m < 16; m++)
    {
        for(int n = 0; n < 16; n++)
        {
            h_c[m * 16 + n] = 0;
            for(int k = 0; k < 128; k++)
            {
                h_c[m * 16 + n] += ref_a[m * 128 + k] * ref_b[n * 128 + k];
            }
            // std::cout << h_c[m * 16 + n] << " ";
        }
        // std::cout << std::endl;
    }

    int32_t* d_a;
    int32_t* d_b;
    float* d_c;

    HIP_CHECK(hipMalloc(&d_a, 16 * 128 / 16 * sizeof(f6x16_pk_t)));
    HIP_CHECK(hipMalloc(&d_b, 16 * 128 / 16 * sizeof(f6x16_pk_t)));
    HIP_CHECK(hipMalloc(&d_c, 16 * 16 * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_a, h_a, 16 * 128 / 16 * sizeof(f6x16_pk_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b, 16 * 128 / 16 * sizeof(f6x16_pk_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_c, 0, 16 * 16 * sizeof(float)));

    kernel1<<<1, 64>>>(d_a, d_b, d_c);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpy(h_cc.data(), d_c, 16 * 16 * sizeof(float), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));

    for(int i = 0; i < 16 * 16; i++)
    {
        std::cout << h_c[i] << "vs" << static_cast<float>(h_cc[i]) << std::endl;
        // printf("%d: %f\n", i, static_cast<float>(h_c[i]));
    }
    return 0;
}