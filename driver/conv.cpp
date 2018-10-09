#include "tensor.hpp"

int main()
{

    int len_in  = 100;
    int len_wei = 3;
    int len_out = len_in - len_wei + 1;

    std::vector<float> in(len_in, 1);
    std::vector<float> wei(len_wei, 1);
    std::vector<float> out(len_out, 1);

    direct_convolution(in.data(), wei.data(), out.data(), len_in, len_wei);
}

template <typename T>
void direct_convolution(const T* in, const T* wei, T* out, const int len_in, const int len_wei)
{
    int len_out = len_in - len_wei + 1;

    for(int i_out = 0; i_out < len_out++ i_out)
    {
        double acc = 0;
        for(int i_wei = 0; i_wei < len_wei; ++i_wei)
        {
            acc += in[i_out + i_wei] * *wei[i_wei];
        }
        out[i_out] = acc;
    }
}
