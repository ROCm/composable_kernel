// 输入tile到输出tile的映射计算示例（包含dilation）
#include <iostream>
#include <algorithm>

struct TileRange
{
    int h_min, h_max, w_min, w_max;
    int size_h() const { return h_max - h_min + 1; }
    int size_w() const { return w_max - w_min + 1; }
};

// 计算有效滤波器大小
int effective_filter_size(int filter_size, int dilation)
{
    return (filter_size - 1) * dilation + 1;
}

// 计算输入tile影响的输出tile范围（包含dilation）
TileRange calculate_output_tile_with_dilation(int input_h_start,
                                              int input_w_start,
                                              int input_tile_size,
                                              int filter_h,
                                              int filter_w,
                                              int dilation_h,
                                              int dilation_w,
                                              int pad_h,
                                              int pad_w,
                                              int stride_h,
                                              int stride_w)
{
    TileRange output_range;

    // 计算有效滤波器大小
    int eff_filter_h = effective_filter_size(filter_h, dilation_h);
    int eff_filter_w = effective_filter_size(filter_w, dilation_w);

    // 输入tile范围
    int input_h_end = input_h_start + input_tile_size - 1;
    int input_w_end = input_w_start + input_tile_size - 1;

    // 计算输出范围 - 使用向上取整的整数除法
    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };

    // 最小输出位置：当滤波器在最右下角时
    output_range.h_min = std::max(0, ceil_div(input_h_start + pad_h - eff_filter_h + 1, stride_h));
    output_range.w_min = std::max(0, ceil_div(input_w_start + pad_w - eff_filter_w + 1, stride_w));

    // 最大输出位置：当滤波器在最左上角时
    output_range.h_max = (input_h_end + pad_h) / stride_h;
    output_range.w_max = (input_w_end + pad_w) / stride_w;

    return output_range;
}

// 验证函数：检查输入位置是否会影响输出位置（包含dilation）
bool input_affects_output_with_dilation(int input_h,
                                        int input_w,
                                        int output_h,
                                        int output_w,
                                        int filter_h,
                                        int filter_w,
                                        int dilation_h,
                                        int dilation_w,
                                        int pad_h,
                                        int pad_w,
                                        int stride_h,
                                        int stride_w)
{
    // 计算滤波器在输入上的采样位置
    for(int kh = 0; kh < filter_h; kh++)
    {
        for(int kw = 0; kw < filter_w; kw++)
        {
            int sample_h = output_h * stride_h - pad_h + kh * dilation_h;
            int sample_w = output_w * stride_w - pad_w + kw * dilation_w;

            if(sample_h == input_h && sample_w == input_w)
            {
                return true;
            }
        }
    }
    return false;
}

int main()
{
    // 卷积参数
    int filter_h = 3, filter_w = 3;
    int pad_h = 1, pad_w = 1;
    int stride_h = 1, stride_w = 1;
    int input_tile_size = 4;

    // 测试不同的dilation值
    int dilations[] = {1, 2, 3};

    for(int dilation : dilations)
    {
        std::cout << "=== 4x4 Input Tile Mapping with Dilation=" << dilation << " ===" << std::endl;
        std::cout << "Filter: " << filter_h << "x" << filter_w << std::endl;
        std::cout << "Effective Filter: " << effective_filter_size(filter_h, dilation) << "x"
                  << effective_filter_size(filter_w, dilation) << std::endl;
        std::cout << "Padding: " << pad_h << "x" << pad_w << std::endl;
        std::cout << "Stride: " << stride_h << "x" << stride_w << std::endl << std::endl;

        // 测试几个不同的输入tile位置
        int test_positions[][2] = {{0, 0}, {2, 2}, {4, 4}};

        for(auto& pos : test_positions)
        {
            int h_start = pos[0], w_start = pos[1];

            TileRange output_tile = calculate_output_tile_with_dilation(h_start,
                                                                        w_start,
                                                                        input_tile_size,
                                                                        filter_h,
                                                                        filter_w,
                                                                        dilation,
                                                                        dilation,
                                                                        pad_h,
                                                                        pad_w,
                                                                        stride_h,
                                                                        stride_w);

            std::cout << "Input Tile [" << h_start << ":" << h_start + input_tile_size - 1 << ", "
                      << w_start << ":" << w_start + input_tile_size - 1 << "]" << std::endl;
            std::cout << "  -> Output Tile [" << output_tile.h_min << ":" << output_tile.h_max
                      << ", " << output_tile.w_min << ":" << output_tile.w_max << "]" << std::endl;
            std::cout << "  -> Output Size: " << output_tile.size_h() << "x" << output_tile.size_w()
                      << std::endl;

            // 验证几个关键点
            std::cout << "  -> Detailed mapping for output positions:" << std::endl;
            for(int h = output_tile.h_min; h <= std::min(output_tile.h_max, output_tile.h_min + 2);
                h++)
            {
                for(int w = output_tile.w_min;
                    w <= std::min(output_tile.w_max, output_tile.w_min + 2);
                    w++)
                {
                    std::cout << "     Output(" << h << "," << w << ") samples from input: ";

                    // 显示这个输出位置采样的所有输入位置
                    for(int kh = 0; kh < filter_h; kh++)
                    {
                        for(int kw = 0; kw < filter_w; kw++)
                        {
                            int sample_h = h * stride_h - pad_h + kh * dilation;
                            int sample_w = w * stride_w - pad_w + kw * dilation;

                            // 检查是否在当前tile范围内
                            if(sample_h >= h_start && sample_h < h_start + input_tile_size &&
                               sample_w >= w_start && sample_w < w_start + input_tile_size)
                            {
                                std::cout << "(" << sample_h << "," << sample_w << ") ";
                            }
                        }
                    }
                    std::cout << std::endl;
                }
            }
            std::cout << std::endl;
        }

        // 显示dilation的影响
        std::cout << "=== Dilation Effect Analysis ===" << std::endl;
        std::cout << "For Input Tile [0:3, 0:3] with different dilations:" << std::endl;

        for(int d = 1; d <= 3; d++)
        {
            TileRange range = calculate_output_tile_with_dilation(
                0, 0, 4, filter_h, filter_w, d, d, pad_h, pad_w, stride_h, stride_w);
            std::cout << "  Dilation " << d << ": Output [" << range.h_min << ":" << range.h_max
                      << ", " << range.w_min << ":" << range.w_max << "] (size: " << range.size_h()
                      << "x" << range.size_w() << ")" << std::endl;
        }
        std::cout << std::endl;
    }

    return 0;
}