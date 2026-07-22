// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/batched_contraction_multi_abd.hpp"
#include "common/utils.hpp"
#include "contraction_multi_abd_common.hpp"

namespace contraction_multi_abd_detail {

inline void
print_dims(std::ostream& os, const std::string& name, const std::vector<ck_tile::index_t>& dims)
{
    os << "   \"" << name << "\":[";
    for(size_t i = 0; i < dims.size(); ++i)
    {
        os << dims[i];
        if(i < dims.size() - 1)
            os << ",";
    }
    os << "]";
}

} // namespace contraction_multi_abd_detail

struct ContractionMultiABDProblem
{
    std::vector<ck_tile::index_t> g_dims_;
    std::vector<ck_tile::index_t> m_dims_;
    std::vector<ck_tile::index_t> n_dims_;
    std::vector<ck_tile::index_t> k_dims_;

    ck_tile::index_t g_total_;
    ck_tile::index_t m_total_;
    ck_tile::index_t n_total_;
    ck_tile::index_t k_total_;

    friend std::ostream& operator<<(std::ostream& os, const ContractionMultiABDProblem& p)
    {
        os << "{\n";
        contraction_multi_abd_detail::print_dims(os, "g_dims", p.g_dims_);
        os << ",\n";
        contraction_multi_abd_detail::print_dims(os, "m_dims", p.m_dims_);
        os << ",\n";
        contraction_multi_abd_detail::print_dims(os, "n_dims", p.n_dims_);
        os << ",\n";
        contraction_multi_abd_detail::print_dims(os, "k_dims", p.k_dims_);
        os << ",\n"
           << "   \"g_total\":" << p.g_total_ << ",\n"
           << "   \"m_total\":" << p.m_total_ << ",\n"
           << "   \"n_total\":" << p.n_total_ << ",\n"
           << "   \"k_total\":" << p.k_total_ << "\n"
           << "}";
        return os;
    }
};
