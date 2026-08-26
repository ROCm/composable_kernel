// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Single-kernel benchmark entry point for batched_contraction_multi_abd.
 *
 * The kernel header is force-included at compile time via:
 *   -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE
 *
 * That header exports to global scope:
 *   SelectedKernel, KERNEL_NAME,
 *   AsDataType, BsDataType, DsDataType, EDataType, AccDataType,
 *   ALayout, BLayout, ELayout, DsLayout,
 *   NumATensors, NumBTensors, NumDTensors,
 *   NumDimsG, NumDimsM, NumDimsN, NumDimsK
 */

#include <iostream>
#include <stdexcept>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "contraction_multi_abd_benchmark.hpp"

inline auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser p;
    // Defaults must match the shipped kernel defaults (NUM_DIM_G=1, M=2, N=2, K=1),
    // otherwise a no-argument run always aborts on the dimension-count check below.
    p.insert("g_dims", "2", "G (batch) dimensions, comma-separated")
        .insert("m_dims", "4,256", "M dimensions, comma-separated")
        .insert("n_dims", "16,128", "N dimensions, comma-separated")
        .insert("k_dims", "64", "K dimensions, comma-separated")
        .insert("verify", "0", "Verify output vs CPU reference (1=yes, 0=no)")
        .insert("warmup", "50", "Warmup iterations")
        .insert("repeat", "100", "Benchmark iterations")
        .insert("timer", "true", "Use GPU timer (true/false)")
        .insert("log", "false", "Log kernel launch info (true/false)");
    bool ok = p.parse(argc, argv);
    return std::make_pair(ok, p);
}

int main(int argc, char* argv[])
{
    try
    {
        auto [ok, parser] = create_args(argc, argv);
        if(!ok)
            return EXIT_FAILURE;

        ContractionMultiABDProblem problem;
        problem.g_dims = parse_dims(parser.get_str("g_dims"));
        problem.m_dims = parse_dims(parser.get_str("m_dims"));
        problem.n_dims = parse_dims(parser.get_str("n_dims"));
        problem.k_dims = parse_dims(parser.get_str("k_dims"));

        // Validate counts match the compiled kernel
        if(static_cast<int>(problem.g_dims.size()) != NumDimsG ||
           static_cast<int>(problem.m_dims.size()) != NumDimsM ||
           static_cast<int>(problem.n_dims.size()) != NumDimsN ||
           static_cast<int>(problem.k_dims.size()) != NumDimsK)
        {
            std::cerr << "Dimension count mismatch: kernel compiled with G=" << NumDimsG
                      << " M=" << NumDimsM << " N=" << NumDimsN << " K=" << NumDimsK << "\n";
            return EXIT_FAILURE;
        }

        // Every extent must be strictly positive. A zero or negative value parses
        // fine but produces zero-sized (or, for a negative pair whose product is
        // positive, undersized) buffers, and a zero K divides by zero when the
        // reference inputs are initialized to 1/K.
        auto check_positive = [](const auto& dims, const char* name) {
            for(size_t i = 0; i < dims.size(); ++i)
            {
                if(dims[i] <= 0)
                {
                    std::cerr << name << "[" << i << "]=" << dims[i]
                              << " is not strictly positive.\n";
                    return false;
                }
            }
            return true;
        };

        if(!check_positive(problem.g_dims, "g_dims") || !check_positive(problem.m_dims, "m_dims") ||
           !check_positive(problem.n_dims, "n_dims") || !check_positive(problem.k_dims, "k_dims"))
        {
            return EXIT_FAILURE;
        }

        run_contraction_multi_abd_benchmark(problem,
                                            parser.get_int("warmup"),
                                            parser.get_int("repeat"),
                                            parser.get_bool("verify"),
                                            parser.get_bool("log"),
                                            parser.get_bool("timer"));

        return EXIT_SUCCESS;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
