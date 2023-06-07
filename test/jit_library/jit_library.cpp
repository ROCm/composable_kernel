#include "ck/host/device_gemm_multiple_d.hpp"
#include <iostream>

bool test_Problem()
{
    auto problem = ck::host::device_gemm_multiple_d::Problem{
        256,
        256,
        256,
        false,
        true,
        false,
        {},
        ck::host::DataType::Half,
        ck::host::DataType::Half,
        ck::host::DataType::Half,
        {},
        "ck::tensor_operation::element_wise::Passthrough",
        "ck::tensor_operation::element_wise::Passthrough",
        "ck::tensor_operation::element_wise::Passthrough"};

    const auto include_header = problem.GetIncludeHeader();
    const auto solutions      = problem.GetSolutions("gfx90a");
    const auto& solution      = solutions.at(0);
    const auto template_str   = solution.template_str;
    const auto grid_size      = solution.grid_size;
    const auto block_size     = solution.block_size;

    bool pass = true;

    pass &= include_header ==
            "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp";
    pass &= solutions.size() == 42;
    pass &= template_str ==
            "ck::tensor_operation::device::DeviceGemmMultipleD_Xdl_CShuffle< "
            "ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::ColumnMajor, ck::Tuple<>, "
            "ck::tensor_layout::gemm::RowMajor, ck::half_t, ck::half_t, float, float, ck::Tuple<>, "
            "ck::half_t, ck::tensor_operation::element_wise::Passthrough, "
            "ck::tensor_operation::element_wise::Passthrough, "
            "ck::tensor_operation::element_wise::Passthrough, "
            "ck::tensor_operation::device::GemmSpecialization::Default, 1, 256, 256, 128, 32, 8, "
            "8, 32, 32, 4, 2, ck::Sequence<4,64,1>, ck::Sequence<1,0,2>, ck::Sequence<1,0,2>, 2, "
            "8, 8, 1, ck::Sequence<4,64,1>, ck::Sequence<1,0,2>, ck::Sequence<1,0,2>, 2, 8, 8, 1, "
            "1, 1, ck::Sequence<1,32,1,8>, 8, ck::LoopScheduler::Default, ck::PipelineVersion::v1>";
    pass &= grid_size == 2;
    pass &= block_size == 256;

    return pass;
}

bool test_GetGemmSpec()
{
    bool pass = true;
    {
        // PadMNK
        auto problem = ck::host::device_gemm_multiple_d::Problem{
            255,
            255,
            255,
            false,
            true,
            false,
            {},
            ck::host::DataType::Half,
            ck::host::DataType::Half,
            ck::host::DataType::Half,
            {},
            "ck::tensor_operation::element_wise::Passthrough",
            "ck::tensor_operation::element_wise::Passthrough",
            "ck::tensor_operation::element_wise::Passthrough"};
        const auto solutions    = problem.GetSolutions("gfx90a");
        const auto& solution    = solutions.at(0);
        const auto template_str = solution.template_str;

        pass &= template_str.find("GemmSpecialization::MNKPadding") != std::string::npos;
    }
    {
        // Default
        auto problem = ck::host::device_gemm_multiple_d::Problem{
            256,
            256,
            256,
            false,
            true,
            false,
            {},
            ck::host::DataType::Half,
            ck::host::DataType::Half,
            ck::host::DataType::Half,
            {},
            "ck::tensor_operation::element_wise::Passthrough",
            "ck::tensor_operation::element_wise::Passthrough",
            "ck::tensor_operation::element_wise::Passthrough"};
        const auto solutions    = problem.GetSolutions("gfx90a");
        const auto& solution    = solutions.at(0);
        const auto template_str = solution.template_str;

        pass &= template_str.find("GemmSpecialization::Default") != std::string::npos;
    }

    return pass;
}

bool test_GetInstances()
{
    bool pass = true;
    {
        // Col Col Fp16
        auto problem = ck::host::device_gemm_multiple_d::Problem{
            256,
            256,
            256,
            true,
            true,
            false,
            {},
            ck::host::DataType::Half,
            ck::host::DataType::Half,
            ck::host::DataType::Half,
            {},
            "ck::tensor_operation::element_wise::Passthrough",
            "ck::tensor_operation::element_wise::Passthrough",
            "ck::tensor_operation::element_wise::Passthrough"};
        pass &= problem.GetSolutions("gfx90a").size() == 51;
    }
    {
        // Col Row Fp16
        auto problem = ck::host::device_gemm_multiple_d::Problem{
            256,
            256,
            256,
            true,
            false,
            false,
            {},
            ck::host::DataType::Half,
            ck::host::DataType::Half,
            ck::host::DataType::Half,
            {},
            "ck::tensor_operation::element_wise::Passthrough",
            "ck::tensor_operation::element_wise::Passthrough",
            "ck::tensor_operation::element_wise::Passthrough"};
        pass &= problem.GetSolutions("gfx90a").size() == 51;
    }
    {
        // Row Col Fp16
        auto problem = ck::host::device_gemm_multiple_d::Problem{
            256,
            256,
            256,
            false,
            true,
            false,
            {},
            ck::host::DataType::Half,
            ck::host::DataType::Half,
            ck::host::DataType::Half,
            {},
            "ck::tensor_operation::element_wise::Passthrough",
            "ck::tensor_operation::element_wise::Passthrough",
            "ck::tensor_operation::element_wise::Passthrough"};
        pass &= problem.GetSolutions("gfx90a").size() == 42;
    }
    {
        // Row Row Int8
        auto problem = ck::host::device_gemm_multiple_d::Problem{
            256,
            256,
            256,
            false,
            false,
            false,
            {},
            ck::host::DataType::Int8,
            ck::host::DataType::Int8,
            ck::host::DataType::Half,
            {},
            "ck::tensor_operation::element_wise::Passthrough",
            "ck::tensor_operation::element_wise::Passthrough",
            "ck::tensor_operation::element_wise::Passthrough"};
        pass &= problem.GetSolutions("gfx90a").size() == 48;
    }
    {
        // Col Col Int8
        auto problem = ck::host::device_gemm_multiple_d::Problem{
            256,
            256,
            256,
            true,
            true,
            false,
            {},
            ck::host::DataType::Int8,
            ck::host::DataType::Int8,
            ck::host::DataType::Half,
            {},
            "ck::tensor_operation::element_wise::Passthrough",
            "ck::tensor_operation::element_wise::Passthrough",
            "ck::tensor_operation::element_wise::Passthrough"};
        pass &= problem.GetSolutions("gfx90a").size() == 48;
    }
    {
        // Col Row Int8
        auto problem = ck::host::device_gemm_multiple_d::Problem{
            256,
            256,
            256,
            true,
            false,
            false,
            {},
            ck::host::DataType::Int8,
            ck::host::DataType::Int8,
            ck::host::DataType::Half,
            {},
            "ck::tensor_operation::element_wise::Passthrough",
            "ck::tensor_operation::element_wise::Passthrough",
            "ck::tensor_operation::element_wise::Passthrough"};
        pass &= problem.GetSolutions("gfx90a").size() == 48;
    }
    {
        // Row Col Int8
        auto problem = ck::host::device_gemm_multiple_d::Problem{
            256,
            256,
            256,
            false,
            true,
            false,
            {},
            ck::host::DataType::Int8,
            ck::host::DataType::Int8,
            ck::host::DataType::Half,
            {},
            "ck::tensor_operation::element_wise::Passthrough",
            "ck::tensor_operation::element_wise::Passthrough",
            "ck::tensor_operation::element_wise::Passthrough"};
        pass &= problem.GetSolutions("gfx90a").size() == 39;
    }
    {
        // Row Row Int8
        auto problem = ck::host::device_gemm_multiple_d::Problem{
            256,
            256,
            256,
            false,
            false,
            false,
            {},
            ck::host::DataType::Int8,
            ck::host::DataType::Int8,
            ck::host::DataType::Half,
            {},
            "ck::tensor_operation::element_wise::Passthrough",
            "ck::tensor_operation::element_wise::Passthrough",
            "ck::tensor_operation::element_wise::Passthrough"};
        pass &= problem.GetSolutions("gfx90a").size() == 48;
    }

    return pass;
}

bool test_MakeLayoutsTuple()
{
    bool pass = true;
    {
        // Empty Tuple
        auto problem = ck::host::device_gemm_multiple_d::Problem{
            256,
            256,
            256,
            false,
            false,
            false,
            {},
            ck::host::DataType::Half,
            ck::host::DataType::Half,
            ck::host::DataType::Half,
            {ck::host::DataType::Half},
            "ck::tensor_operation::element_wise::Passthrough",
            "ck::tensor_operation::element_wise::Passthrough",
            "ck::tensor_operation::element_wise::Passthrough"};
        const auto solutions    = problem.GetSolutions("gfx90a");
        const auto& solution    = solutions.at(0);
        const auto template_str = solution.template_str;
        pass &= template_str.find("ck::Tuple<>") != std::string::npos;
    }
    {
        // RowColRow Tuple
        auto problem = ck::host::device_gemm_multiple_d::Problem{
            256,
            256,
            256,
            false,
            false,
            false,
            {false, true, false},
            ck::host::DataType::Half,
            ck::host::DataType::Half,
            ck::host::DataType::Half,
            {ck::host::DataType::Half},
            "ck::tensor_operation::element_wise::Passthrough",
            "ck::tensor_operation::element_wise::Passthrough",
            "ck::tensor_operation::element_wise::Passthrough"};
        const auto solutions    = problem.GetSolutions("gfx90a");
        const auto& solution    = solutions.at(0);
        const auto template_str = solution.template_str;
        pass &= template_str.find(
                    "ck::Tuple<ck::tensor_layout::gemm::RowMajor, "
                    "ck::tensor_layout::gemm::ColumnMajor, ck::tensor_layout::gemm::RowMajor>") !=
                std::string::npos;
    }

    return pass;
}

bool test_MakeTypeTuple()
{
    bool pass = true;
    {
        // Empty Tuple
        auto problem = ck::host::device_gemm_multiple_d::Problem{
            256,
            256,
            256,
            false,
            false,
            false,
            {true},
            ck::host::DataType::Half,
            ck::host::DataType::Half,
            ck::host::DataType::Half,
            {},
            "ck::tensor_operation::element_wise::Passthrough",
            "ck::tensor_operation::element_wise::Passthrough",
            "ck::tensor_operation::element_wise::Passthrough"};
        const auto solutions    = problem.GetSolutions("gfx90a");
        const auto& solution    = solutions.at(0);
        const auto template_str = solution.template_str;
        pass &= template_str.find("ck::Tuple<>") != std::string::npos;
    }
    {
        // Half Int8 Tuple
        auto problem = ck::host::device_gemm_multiple_d::Problem{
            256,
            256,
            256,
            false,
            false,
            false,
            {},
            ck::host::DataType::Half,
            ck::host::DataType::Half,
            ck::host::DataType::Half,
            {ck::host::DataType::Half, ck::host::DataType::Int8},
            "ck::tensor_operation::element_wise::Passthrough",
            "ck::tensor_operation::element_wise::Passthrough",
            "ck::tensor_operation::element_wise::Passthrough"};
        const auto solutions    = problem.GetSolutions("gfx90a");
        const auto& solution    = solutions.at(0);
        const auto template_str = solution.template_str;
        pass &= template_str.find("ck::Tuple<ck::half_t, int8_t>") != std::string::npos;
    }
    return pass;
}

int main()
{
    bool pass = true;
    pass &= test_Problem();
    pass &= test_GetGemmSpec();
    pass &= test_GetInstances();
    pass &= test_MakeLayoutsTuple();
    pass &= test_MakeTypeTuple();

    if(pass)
    {
        std::cout << "Test jit library: Pass" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "Test jit library: Fail" << std::endl;
        return -1;
    }
}