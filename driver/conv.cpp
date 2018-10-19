#include <iostream>
#include "tensor.hpp"

template <typename T>
void direct_convolution(const Tensor<T>& in,
                        const Tensor<T>& wei,
                        Tensor<T>& out,
                        std::size_t num_thread)
{
    auto f = [&](auto n, auto k, auto ho, auto wo) {
        double v = 0;
        for(int c = 0; c < wei.mDesc.GetLengths()[1]; ++c)
        {
            for(int y = 0; y < wei.mDesc.GetLengths()[2]; ++y)
            {
                int hi = ho + y;
                for(int x = 0; x < wei.mDesc.GetLengths()[3]; ++x)
                {
                    int wi = wo + x;
                    v += in(n, c, hi, wi) * wei(k, c, y, x);
                }
            }
        }
        out(n, k, ho, wo) = v;
    };

    auto f_par = make_ParallelTensorFunctor(f,
                                            out.mDesc.GetLengths()[0],
                                            out.mDesc.GetLengths()[1],
                                            out.mDesc.GetLengths()[2],
                                            out.mDesc.GetLengths()[3]);

    f_par(num_thread);
}

template <class T>
struct Generator
{

    template <class... Is>
    T operator()(Is... is)
    {
        return 1;
    }
};

int main()
{
    Tensor<float> in({3, 16, 128, 128});
    Tensor<float> wei({4, 16, 3, 3});
    Tensor<float> out({3, 4, 126, 126});

    int num_thread = std::thread::hardware_concurrency();

    std::cout << __func__ << ": num_thread " << num_thread << std::endl;

    in.GenerateTensorValue(Generator<float>{}, num_thread);
    wei.GenerateTensorValue(Generator<float>{}, num_thread);

    direct_convolution(in, wei, out, num_thread);

    std::cout << __func__ << ": done" << std::endl;

    LogRange(std::cout, in.mData, ",") << std::endl;
    LogRange(std::cout, wei.mData, ",") << std::endl;
    LogRange(std::cout, out.mData, ",") << std::endl;
}
