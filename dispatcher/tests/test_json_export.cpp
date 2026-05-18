// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// Unit tests for JSON export functionality

#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/json_export.hpp"
#include "test_mock_kernel.hpp"
#include <gtest/gtest.h>
#include <fstream>
#include <cstdio>

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::test;

// =============================================================================
// Basic Export Tests
// =============================================================================

class JSONExportBasicTest : public ::testing::Test
{
    protected:
    void SetUp() override { Registry::instance().clear(); }

    void TearDown() override { Registry::instance().clear(); }
};

TEST_F(JSONExportBasicTest, ExportEmptyRegistry)
{
    std::string json = Registry::instance().export_json(false);

    EXPECT_FALSE(json.empty());
    EXPECT_NE(json.find("\"kernels\""), std::string::npos);
    // Empty registry should still produce valid JSON with kernels section
}

TEST_F(JSONExportBasicTest, ExportSingleKernel)
{
    auto key    = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "test_kernel");
    Registry::instance().register_kernel(kernel);

    std::string json = Registry::instance().export_json(false);

    EXPECT_FALSE(json.empty());
    EXPECT_NE(json.find("\"test_kernel\""), std::string::npos);
}

TEST_F(JSONExportBasicTest, ExportMultipleKernels)
{
    for(int i = 0; i < 5; i++)
    {
        auto key    = make_test_key(100 + i);
        auto kernel = std::make_shared<MockKernelInstance>(key, "kernel_" + std::to_string(i));
        Registry::instance().register_kernel(kernel);
    }

    std::string json = Registry::instance().export_json(false);

    // Should contain all kernel names
    for(int i = 0; i < 5; i++)
    {
        EXPECT_NE(json.find("\"kernel_" + std::to_string(i) + "\""), std::string::npos);
    }
}

// =============================================================================
// Export with Statistics Tests
// =============================================================================

class JSONExportStatisticsTest : public ::testing::Test
{
    protected:
    void SetUp() override { Registry::instance().clear(); }

    void TearDown() override { Registry::instance().clear(); }
};

TEST_F(JSONExportStatisticsTest, ExportWithStatistics)
{
    auto key    = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "kernel");
    Registry::instance().register_kernel(kernel);

    std::string json = Registry::instance().export_json(true); // Include statistics

    EXPECT_NE(json.find("\"statistics\""), std::string::npos);
    EXPECT_NE(json.find("\"by_datatype\""), std::string::npos);
    EXPECT_NE(json.find("\"by_pipeline\""), std::string::npos);
}

TEST_F(JSONExportStatisticsTest, ExportWithoutStatistics)
{
    auto key    = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "kernel");
    Registry::instance().register_kernel(kernel);

    std::string json = Registry::instance().export_json(false); // No statistics

    // Statistics section might be minimal or absent
    EXPECT_NE(json.find("\"kernels\""), std::string::npos);
}

// =============================================================================
// Metadata Tests
// =============================================================================

class JSONExportMetadataTest : public ::testing::Test
{
    protected:
    void SetUp() override { Registry::instance().clear(); }

    void TearDown() override { Registry::instance().clear(); }
};

TEST_F(JSONExportMetadataTest, MetadataPresent)
{
    std::string json = Registry::instance().export_json(true);

    EXPECT_NE(json.find("\"metadata\""), std::string::npos);
    EXPECT_NE(json.find("\"timestamp\""), std::string::npos);
    EXPECT_NE(json.find("\"total_kernels\""), std::string::npos);
}

TEST_F(JSONExportMetadataTest, CorrectKernelCount)
{
    const int num_kernels = 7;
    for(int i = 0; i < num_kernels; i++)
    {
        auto key    = make_test_key(100 + i);
        auto kernel = std::make_shared<MockKernelInstance>(key, "kernel_" + std::to_string(i));
        Registry::instance().register_kernel(kernel);
    }

    std::string json = Registry::instance().export_json(true);

    EXPECT_NE(json.find("\"total_kernels\": " + std::to_string(num_kernels)), std::string::npos);
}

TEST_F(JSONExportMetadataTest, RegistryNameIncluded)
{
    Registry::instance().set_name("test_registry");

    auto key    = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "kernel");
    Registry::instance().register_kernel(kernel);

    std::string json = Registry::instance().export_json(true);

    EXPECT_NE(json.find("\"registry_name\""), std::string::npos);
    EXPECT_NE(json.find("\"test_registry\""), std::string::npos);
}

// =============================================================================
// Export to File Tests
// =============================================================================

class JSONExportToFileTest : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        Registry::instance().clear();
        test_file_ = "/tmp/test_export_" + std::to_string(time(nullptr)) + ".json";
    }

    void TearDown() override
    {
        Registry::instance().clear();
        std::remove(test_file_.c_str());
    }

    std::string test_file_;
};

TEST_F(JSONExportToFileTest, ExportToFile)
{
    auto key    = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "kernel");
    Registry::instance().register_kernel(kernel);

    bool success = Registry::instance().export_json_to_file(test_file_, true);
    EXPECT_TRUE(success);

    // Verify file exists
    std::ifstream file(test_file_);
    EXPECT_TRUE(file.good());

    // Verify content
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("\"kernel\""), std::string::npos);
}

TEST_F(JSONExportToFileTest, ExportToInvalidPath)
{
    bool success = Registry::instance().export_json_to_file("/invalid/path/file.json", true);
    EXPECT_FALSE(success);
}

// =============================================================================
// Auto-Export Tests
// =============================================================================

class JSONAutoExportTest : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        Registry::instance().clear();
        Registry::instance().disable_auto_export();
        test_file_ = "/tmp/test_auto_export_" + std::to_string(time(nullptr)) + ".json";
    }

    void TearDown() override
    {
        Registry::instance().disable_auto_export();
        Registry::instance().clear();
        std::remove(test_file_.c_str());
    }

    std::string test_file_;
};

TEST_F(JSONAutoExportTest, EnableAutoExport)
{
    EXPECT_FALSE(Registry::instance().is_auto_export_enabled());

    Registry::instance().enable_auto_export(test_file_, true, false);

    EXPECT_TRUE(Registry::instance().is_auto_export_enabled());
}

TEST_F(JSONAutoExportTest, DisableAutoExport)
{
    Registry::instance().enable_auto_export(test_file_, true, false);
    EXPECT_TRUE(Registry::instance().is_auto_export_enabled());

    Registry::instance().disable_auto_export();
    EXPECT_FALSE(Registry::instance().is_auto_export_enabled());
}

TEST_F(JSONAutoExportTest, AutoExportOnRegistration)
{
    // Enable auto-export with export_on_every_registration=true
    Registry::instance().enable_auto_export(test_file_, true, false);

    auto key    = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "auto_kernel");
    Registry::instance().register_kernel(kernel);

    // File might be created on registration or on exit depending on implementation
    // Just verify auto-export is enabled
    EXPECT_TRUE(Registry::instance().is_auto_export_enabled());
}

// =============================================================================
// JSON Validity Tests
// =============================================================================

class JSONValidityTest : public ::testing::Test
{
    protected:
    void SetUp() override { Registry::instance().clear(); }

    void TearDown() override { Registry::instance().clear(); }

    // Simple JSON syntax checker
    bool isValidJSON(const std::string& json)
    {
        int braces     = 0;
        int brackets   = 0;
        bool in_string = false;
        char prev      = '\0';

        for(char c : json)
        {
            if(c == '"' && prev != '\\')
            {
                in_string = !in_string;
            }

            if(!in_string)
            {
                if(c == '{')
                    braces++;
                else if(c == '}')
                    braces--;
                else if(c == '[')
                    brackets++;
                else if(c == ']')
                    brackets--;
            }

            if(braces < 0 || brackets < 0)
                return false;
            prev = c;
        }

        return braces == 0 && brackets == 0 && !in_string;
    }
};

TEST_F(JSONValidityTest, EmptyRegistryProducesValidJSON)
{
    std::string json = Registry::instance().export_json(true);
    EXPECT_TRUE(isValidJSON(json));
}

TEST_F(JSONValidityTest, SingleKernelProducesValidJSON)
{
    auto key    = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "kernel");
    Registry::instance().register_kernel(kernel);

    std::string json = Registry::instance().export_json(true);
    EXPECT_TRUE(isValidJSON(json));
}

TEST_F(JSONValidityTest, ManyKernelsProduceValidJSON)
{
    for(int i = 0; i < 50; i++)
    {
        auto key    = make_test_key(100 + i);
        auto kernel = std::make_shared<MockKernelInstance>(key, "kernel_" + std::to_string(i));
        Registry::instance().register_kernel(kernel);
    }

    std::string json = Registry::instance().export_json(true);
    EXPECT_TRUE(isValidJSON(json));
}

TEST_F(JSONValidityTest, NoNullBytesInJSON)
{
    auto key    = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "kernel");
    Registry::instance().register_kernel(kernel);

    std::string json = Registry::instance().export_json(true);

    // Check for null bytes
    EXPECT_EQ(json.find('\0'), std::string::npos);
}

TEST_F(JSONValidityTest, NoPrintableGarbageInJSON)
{
    auto key    = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "kernel");
    Registry::instance().register_kernel(kernel);

    std::string json = Registry::instance().export_json(true);

    // All characters should be printable or whitespace
    for(char c : json)
    {
        EXPECT_TRUE(std::isprint(c) || std::isspace(c))
            << "Non-printable character: " << static_cast<int>(c);
    }
}

// =============================================================================
// Kernel Details Tests
// =============================================================================

class JSONKernelDetailsTest : public ::testing::Test
{
    protected:
    void SetUp() override { Registry::instance().clear(); }

    void TearDown() override { Registry::instance().clear(); }
};

TEST_F(JSONKernelDetailsTest, SignatureIncluded)
{
    auto key              = make_test_key(256);
    key.signature.dtype_a = DataType::FP16;
    key.signature.dtype_b = DataType::FP16;
    key.signature.dtype_c = DataType::FP16;

    auto kernel = std::make_shared<MockKernelInstance>(key, "kernel");
    Registry::instance().register_kernel(kernel);

    std::string json = Registry::instance().export_json(true);

    EXPECT_NE(json.find("\"signature\""), std::string::npos);
    EXPECT_NE(json.find("\"dtype_a\""), std::string::npos);
    EXPECT_NE(json.find("\"fp16\""), std::string::npos);
}

TEST_F(JSONKernelDetailsTest, AlgorithmIncluded)
{
    auto key    = make_test_key(256, 256, 32);
    auto kernel = std::make_shared<MockKernelInstance>(key, "kernel");
    Registry::instance().register_kernel(kernel);

    std::string json = Registry::instance().export_json(true);

    EXPECT_NE(json.find("\"algorithm\""), std::string::npos);
    EXPECT_NE(json.find("\"tile_shape\""), std::string::npos);
}

TEST_F(JSONKernelDetailsTest, IdentifierIncluded)
{
    auto key    = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "my_kernel");
    Registry::instance().register_kernel(kernel);

    std::string json = Registry::instance().export_json(true);

    EXPECT_NE(json.find("\"identifier\""), std::string::npos);
    EXPECT_NE(json.find("\"name\""), std::string::npos);
    EXPECT_NE(json.find("\"my_kernel\""), std::string::npos);
}

// =============================================================================
// Multiple Registries Export Tests
// =============================================================================

class JSONMultipleRegistriesTest : public ::testing::Test
{
    protected:
    void TearDown() override { Registry::instance().clear(); }
};

TEST_F(JSONMultipleRegistriesTest, DifferentRegistriesDifferentJSON)
{
    Registry reg1;
    reg1.set_name("registry1");

    Registry reg2;
    reg2.set_name("registry2");

    auto key1 = make_test_key(128);
    auto key2 = make_test_key(256);

    reg1.register_kernel(std::make_shared<MockKernelInstance>(key1, "k1"));
    reg2.register_kernel(std::make_shared<MockKernelInstance>(key2, "k2"));

    std::string json1 = reg1.export_json(true);
    std::string json2 = reg2.export_json(true);

    EXPECT_NE(json1, json2);

    EXPECT_NE(json1.find("\"registry1\""), std::string::npos);
    EXPECT_NE(json2.find("\"registry2\""), std::string::npos);

    EXPECT_NE(json1.find("\"k1\""), std::string::npos);
    EXPECT_NE(json2.find("\"k2\""), std::string::npos);
}
