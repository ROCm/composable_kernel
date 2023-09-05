#include <cstring>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/tensor/tensor_view.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/host_utility/device_prop.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

#include "softmax.hpp"

template <typename ADataType, typename AccDataType, typename BDataType>
void reference_softmax(const Tensor<ADataType>& a_m_n, Tensor<BDataType>& b_m_n)
{
    auto f = [&](auto m) {
        const int N = a_m_n.mDesc.GetLengths()[1];

        AccDataType v_max = ck::NumericLimits<ADataType>::Lowest();

        // max
        for(int n = 0; n < N; ++n)
        {
            const ADataType v_a = a_m_n(m, n);

            v_max = v_max < v_a ? v_a : v_max;
        }

        AccDataType v_exp_sum = 0;

        // sum
        for(int n = 0; n < N; ++n)
        {
            const ADataType v_a = a_m_n(m, n);

            v_exp_sum += ck::math::exp(v_a - v_max);
        }

        // elementwise
        for(int n = 0; n < N; ++n)
        {
            const ADataType v_a = a_m_n(m, n);

            b_m_n(m, n) = ck::math::exp(v_a - v_max) / v_exp_sum;
        }
    };

    make_ParallelTensorFunctor(f, b_m_n.mDesc.GetLengths()[0])(std::thread::hardware_concurrency());
}

int main(int argc, char* argv[])
{
    using ADataType   = float;
    using AccDataType = float;
    using BDataType   = float;

    ck::index_t M = 3328;
    ck::index_t N = 4096;

    if(argc == 3)
    {
        M = std::stoi(argv[1]);
        N = std::stoi(argv[2]);
    }

    std::array<ck::index_t, 2> a_lengths{M, N};
    std::array<ck::index_t, 2> a_strides{N, 1};

    std::array<ck::index_t, 2> b_lengths{M, N};
    std::array<ck::index_t, 2> b_strides{N, 1};

    // host verify
    Tensor<ADataType> a_host(a_lengths, a_strides);
    Tensor<BDataType> b_host_ref(b_lengths, b_strides);
    Tensor<BDataType> b_host_dev(b_lengths, b_strides);

    ck::utils::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f}(a_host);

    // reference
    reference_softmax<ADataType, AccDataType, BDataType>(a_host, b_host_ref);

    DeviceMem a_buf(sizeof(ADataType) * a_host.GetElementSpaceSize());
    DeviceMem b_buf(sizeof(BDataType) * b_host_ref.GetElementSpaceSize());

    a_buf.ToDevice(a_host.mData.data());

    constexpr ck::index_t kMPerBlock = 128;
    constexpr ck::index_t kNPerBlock = 128;

    constexpr ck::index_t kBlockSize = 256;
    ck::index_t kGridSize            = (M / kMPerBlock);

    std::cout << "grid size " << kGridSize << std::endl;

    const auto kernel =
        Softmax<ADataType, AccDataType, BDataType, kBlockSize, kMPerBlock, kNPerBlock>{};

    float ave_time = launch(ProgramServer{},
                            kernel,
                            kGridSize,
                            kBlockSize,
                            static_cast<ADataType*>(a_buf.GetDeviceBuffer()),
                            static_cast<BDataType*>(b_buf.GetDeviceBuffer()),
                            M,
                            N);

    b_buf.FromDevice(b_host_dev.mData.data());

    std::size_t num_btype = sizeof(ADataType) * M * N + sizeof(BDataType) * M * N;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

    return !ck::utils::check_err(b_host_dev, b_host_ref);
}
