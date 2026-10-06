// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <non_ascii_test.hpp>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

int main()
{
    const auto data = non_ascii_test();
    const auto it   = data.find("non_ascii_test.txt");
    if(it == data.end())
    {
        std::cerr << "Embedded file 'non_ascii_test.txt' not found\n";
        return 1;
    }

    const std::string_view content = it->second;
    if(content.empty())
    {
        std::cerr << "Embedded content is empty\n";
        return 1;
    }
    if(content.find("\xe2\x80\x94") == std::string_view::npos)
    {
        std::cerr << "Expected UTF-8 em dash bytes not found in embedded content\n";
        return 1;
    }

    std::ifstream file(std::string(SOURCE_DIR) + "/non_ascii_test.txt", std::ios::binary);
    if(!file.is_open())
    {
        std::cerr << "Failed to open source file non_ascii_test.txt\n";
        return 1;
    }

    std::vector<char> original((std::istreambuf_iterator<char>(file)),
                               std::istreambuf_iterator<char>());

    if(content.size() != original.size())
    {
        std::cerr << "Size mismatch: embedded=" << content.size() << " original=" << original.size()
                  << "\n";
        return 1;
    }
    if(std::memcmp(content.data(), original.data(), content.size()) != 0)
    {
        std::cerr << "Content mismatch between embedded and original file\n";
        return 1;
    }

    std::cout << "Non-ASCII embed regression test passed." << std::endl;
    return 0;
}
