// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ostream>
#include <string>

#include "ck/ck.hpp"
#include "ck/tile_program/block_tile/block_masking.hpp"

enum class mask_enum
{
    no_mask = 0,
    causal_top_left,
    causal_bottom_right,
    window_generic,
};

struct mask_info
{
    mask_enum type;
    ck::index_t y, x;

    void serialize(std::ostream& os) const
    {
        if(type == mask_enum::no_mask)
            os << "n";
        else if(type == mask_enum::causal_top_left)
            os << "tl";
        else if(type == mask_enum::causal_bottom_right)
            os << "br";
        else
        {
            os << "g(" << y << "/" << x << ")";
        }
    }
    static mask_info decode(std::string str, ck::index_t seqlen_q, ck::index_t seqlen_k)
    {
        ck::index_t x_total = seqlen_k;
        ck::index_t y_total = seqlen_q;
        mask_info tmp;
        auto found_0 = str.find(':');
        if(found_0 != std::string::npos)
        {
            std::string t = str.substr(0, found_0);
            std::string v = str.substr(found_0 + 1);
            auto found_1  = v.find(",");
            if(found_1 == std::string::npos)
            {
                printf("not supported value %s, %s\n", v.c_str(), str.c_str());
                assert(0);
            }
            tmp.type       = mask_enum::window_generic;
            ck::index_t v0 = atoi(v.substr(0, found_1).c_str());
            ck::index_t v1 = atoi(v.substr(found_1 + 1).c_str());
            // TODO: some validation
            if(t == "t")
            {
                auto r = ck::make_generic_attention_mask_coordinates_from_lr_window(
                    v0, v1, y_total, x_total, true);
                tmp.y = r.At(ck::Number<0>{});
                tmp.x = r.At(ck::Number<1>{});
            }
            else if(t == "b")
            {
                auto r = ck::make_generic_attention_mask_coordinates_from_lr_window(
                    v0, v1, y_total, x_total, false);
                tmp.y = r.At(ck::Number<0>{});
                tmp.x = r.At(ck::Number<1>{});
            }
            else if(t == "g")
            {
                tmp.y = v0;
                tmp.x = v1;
            }
            else
            {
                printf("not supported type %s, %s\n", t.c_str(), str.c_str());
                assert(0);
            }
        }
        else
        {
            // should be 0, 1, 2
            tmp.type = static_cast<mask_enum>(atoi(str.c_str()));
            if(tmp.type == mask_enum::causal_top_left)
            {
                tmp.y = seqlen_q;
                tmp.x = 1;
            }
            else if(tmp.type == mask_enum::causal_bottom_right)
            {
                tmp.y = seqlen_q;
                tmp.x = seqlen_k - seqlen_q + 1;
            }
        }
        return tmp;
    }

    friend std::ostream& operator<<(std::ostream& os, const mask_info& mi);
};

inline std::ostream& operator<<(std::ostream& os, const mask_info& mi)
{
    mi.serialize(os);
    return os;
}
