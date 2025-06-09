// 4x4 Input Tile到Output Tile映射 - Stride=2情况
#include <iostream>
#include <algorithm>
#include <vector>

struct TileRange
{
    int h_min, h_max, w_min, w_max;
    int size_h() const { return h_max - h_min + 1; }
    int size_w() const { return w_max - w_min + 1; }
    bool is_valid() const { return h_min <= h_max && w_min <= w_max; }
};

// 计算4x4输入tile影响的输出tile范围
TileRange calculate_output_tile_stride2(int input_h_start, int input_w_start)
{
    // 卷积参数
    const int input_tile_size = 4;
    const int kernel_h = 3, kernel_w = 3;
    const int stride_h = 2, stride_w = 2;
    const int pad_h = 1, pad_w = 1;

    TileRange output_range;

    // 输入tile范围: [h_start, h_start+3] x [w_start, w_start+3]
    int input_h_end = input_h_start + input_tile_size - 1;
    int input_w_end = input_w_start + input_tile_size - 1;

    // 向上取整除法
    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };

    // 最小输出位置：当滤波器在最右下角时 (kh=2, kw=2)
    output_range.h_min = std::max(0, ceil_div(input_h_start + pad_h - (kernel_h - 1), stride_h));
    output_range.w_min = std::max(0, ceil_div(input_w_start + pad_w - (kernel_w - 1), stride_w));

    // 最大输出位置：当滤波器在最左上角时 (kh=0, kw=0)
    output_range.h_max = (input_h_end + pad_h) / stride_h;
    output_range.w_max = (input_w_end + pad_w) / stride_w;

    return output_range;
}

// 检查输入位置是否影响输出位置
bool input_affects_output_stride2(int input_h, int input_w, int output_h, int output_w)
{
    const int kernel_h = 3, kernel_w = 3;
    const int stride_h = 2, stride_w = 2;
    const int pad_h = 1, pad_w = 1;

    // 计算该输出位置的滤波器在输入上的采样范围
    for(int kh = 0; kh < kernel_h; kh++)
    {
        for(int kw = 0; kw < kernel_w; kw++)
        {
            int sample_h = output_h * stride_h - pad_h + kh;
            int sample_w = output_w * stride_w - pad_w + kw;

            if(sample_h == input_h && sample_w == input_w)
            {
                return true;
            }
        }
    }
    return false;
}

// 获取输出位置采样的所有输入位置
std::vector<std::pair<int, int>> get_sampled_positions(int output_h, int output_w)
{
    const int kernel_h = 3, kernel_w = 3;
    const int stride_h = 2, stride_w = 2;
    const int pad_h = 1, pad_w = 1;

    std::vector<std::pair<int, int>> positions;

    for(int kh = 0; kh < kernel_h; kh++)
    {
        for(int kw = 0; kw < kernel_w; kw++)
        {
            int sample_h = output_h * stride_h - pad_h + kh;
            int sample_w = output_w * stride_w - pad_w + kw;
            positions.push_back({sample_h, sample_w});
        }
    }
    return positions;
}

int main()
{
    std::cout << "=== 4x4 Input Tile映射分析 (Stride=2) ===" << std::endl;
    std::cout << "参数: kernel=3x3, stride=2x2, pad=1x1" << std::endl;
    std::cout << std::endl;

    // 测试几个不同的输入tile位置
    std::vector<std::pair<int, int>> test_tiles = {{0, 0}, {2, 2}, {4, 4}, {6, 6}, {8, 8}};

    for(auto [h_start, w_start] : test_tiles)
    {
        std::cout << "输入Tile [" << h_start << ":" << h_start + 3 << ", " << w_start << ":"
                  << w_start + 3 << "]" << std::endl;

        TileRange output_tile = calculate_output_tile_stride2(h_start, w_start);

        if(output_tile.is_valid())
        {
            std::cout << "  -> 输出Tile [" << output_tile.h_min << ":" << output_tile.h_max << ", "
                      << output_tile.w_min << ":" << output_tile.w_max << "]" << std::endl;
            std::cout << "  -> 输出大小: " << output_tile.size_h() << "x" << output_tile.size_w()
                      << std::endl;

            // 详细分析每个输出位置
            std::cout << "  -> 详细映射关系:" << std::endl;
            for(int h = output_tile.h_min; h <= output_tile.h_max; h++)
            {
                for(int w = output_tile.w_min; w <= output_tile.w_max; w++)
                {
                    std::cout << "     Output(" << h << "," << w << ") 采样输入位置: ";

                    auto sampled = get_sampled_positions(h, w);
                    bool first   = true;
                    for(auto [sh, sw] : sampled)
                    {
                        // 检查是否在当前输入tile范围内
                        if(sh >= h_start && sh < h_start + 4 && sw >= w_start && sw < w_start + 4)
                        {
                            if(!first)
                                std::cout << ", ";
                            std::cout << "(" << sh << "," << sw << ")";
                            first = false;
                        }
                    }
                    std::cout << std::endl;
                }
            }
        }
        else
        {
            std::cout << "  -> 无效的输出tile (该输入tile不影响任何输出)" << std::endl;
        }
        std::cout << std::endl;
    }

    // 特殊分析：stride=2的影响
    std::cout << "=== Stride=2的影响分析 ===" << std::endl;
    std::cout << "对比同一输入tile在不同stride下的输出范围:" << std::endl;

    // 输入tile [0:3, 0:3]
    int h_start = 0, w_start = 0;

    // Stride=1的情况 (理论计算)
    std::cout << "输入Tile [0:3, 0:3]:" << std::endl;
    std::cout << "  Stride=1: 输出范围大约是 [0:4, 0:4] (5x5)" << std::endl;

    // Stride=2的实际情况
    TileRange stride2_output = calculate_output_tile_stride2(0, 0);
    std::cout << "  Stride=2: 输出范围是 [" << stride2_output.h_min << ":" << stride2_output.h_max
              << ", " << stride2_output.w_min << ":" << stride2_output.w_max << "] ("
              << stride2_output.size_h() << "x" << stride2_output.size_w() << ")" << std::endl;

    std::cout << std::endl;
    std::cout << "=== 关键观察 ===" << std::endl;
    std::cout << "1. Stride=2时，输出尺寸大约是输入的一半" << std::endl;
    std::cout << "2. 4x4输入tile通常影响2x2或3x3的输出tile" << std::endl;
    std::cout << "3. 输出位置之间有间隔，不是连续的密集映射" << std::endl;

    return 0;
}