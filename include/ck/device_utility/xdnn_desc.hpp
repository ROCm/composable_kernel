#pragma once

#include <string>
#include <vector>
#include <functional>
#include <string.h>

#define XDNN_OK 0
#define XDNN_FAIL 1

namespace ck {

int sanitize_desc(int& ndims,
                  std::vector<std::reference_wrapper<int64_t>> d,
                  std::vector<std::reference_wrapper<int64_t>> h,
                  std::vector<std::reference_wrapper<int64_t>> w,
                  const std::vector<int64_t>& def_values,
                  bool must_have_spatial)
{
    size_t N = d.size();
    assert(h.size() == N && w.size() == N && def_values.size() == N);

    ndims = 5;

    // check output spatial values
    const bool no_d = d[0].get() == 0;
    const bool no_h = h[0].get() == 0;
    const bool no_w = w[0].get() == 0;

    if(no_d)
        ndims--;
    if(no_d && no_h)
        ndims--;
    if(no_d && no_h && no_w)
        ndims--;
    if(must_have_spatial && ndims <= 2)
        return XDNN_FAIL;

    if(ndims == 5)
    {
        if(no_h && no_w)
        {
            // User specified values for the d dimension but not values for h
            // and w dimensions. Propagate d values to h and w dimensions.
            for(size_t n = 0; n < N; ++n)
                w[n].get() = h[n].get() = d[n].get();
        }
        else if(!no_h && !no_w)
        {
            // User specified them all, good to go.
        }
        else
        {
            // Problem is not cubic and one of h or w dimension is missing.
            return XDNN_FAIL;
        }
    }
    else if(ndims == 4 && no_w)
    {
        // User specified values for the h dimension but not values for the w
        // dimension. Propagate h values to the w dimension.
        for(size_t n = 0; n < N; ++n)
            w[n].get() = h[n].get();
    }

    for(size_t n = 0; n < N; ++n)
    {
        if(ndims < 5)
            d[n].get() = def_values[n];
        if(ndims < 4)
            h[n].get() = def_values[n];
        if(ndims < 3)
            w[n].get() = def_values[n];
    }

    return XDNN_OK;
}

struct desc_t
{
    int64_t g, mb;
    int64_t ic, id, ih, iw;
    int64_t oc, od, oh, ow;
    int64_t kd, kh, kw;
    int64_t sd, sh, sw;
    int64_t pd, ph, pw;
    int64_t pd_r, ph_r, pw_r; // End side padding for each dimension
    int64_t dd, dh, dw;
    bool has_groups;

    const char* name;
    int ndims;

    // Initialize dependent opposite-side paddings values
    // from the shape parameters
    void init_pad_r(bool is_deconv)
    {
        pw_r = opp_pad(is_deconv, iw, ow, kw, sw, pw, dw);
        ph_r = opp_pad(is_deconv, ih, oh, kh, sh, ph, dh);
        pd_r = opp_pad(is_deconv, id, od, kd, sd, pd, dd);
    }

    int64_t desc_nelems(int arg, int mask) const;

    private:
    int64_t
    opp_pad(bool is_deconv, int64_t i, int64_t o, int64_t k, int64_t s, int64_t p, int64_t d) const
    {
        return is_deconv ? (i - 1) * s - o + ((k - 1) * (d + 1) + 1) - p
                         : (o - 1) * s - i + ((k - 1) * (d + 1) + 1) - p;
    }
};

static inline int str2desc(desc_t* desc, const char* str, bool is_deconv = false)
{
    /* canonical form:
     * gXmbX_icXidXihXiwX_ocXodXohXowX_kdXkhXkwX_sdXshXswX_pdXphXpwX_ddXdhXdwXnS
     *
     * where X is number, S - string
     * note: symbol `_` is ignored
     *
     * implicit rules:
     *  - if smaller dimensions are not specified => square or cubic form;
     *  - if output is undefined => compute output;
     *  - if padding is undefined => compute trivial padding;
     */

    desc_t d{0};
    d.g  = 1;
    d.mb = 2;
    d.sd = d.sh = d.sw = 1;
    d.pd = d.ph = d.pw = -1;

    const char* s = str;
    assert(s);

#define CASE_NN(prb, c)                                              \
    do                                                               \
    {                                                                \
        if(!strncmp(prb, s, strlen(prb)))                            \
        {                                                            \
            ok = 1;                                                  \
            s += strlen(prb);                                        \
            char* end_s;                                             \
            d.c = strtol(s, &end_s, 10);                             \
            s += (end_s - s);                                        \
            /* check any # groups, including one, works correctly */ \
            if(!strncmp(prb, "g", 1))                                \
                d.has_groups = true;                                 \
            if(d.c < 0)                                              \
                return XDNN_FAIL;                                    \
            /* printf("@@@debug: %s: %d\n", prb, d. c); */           \
        }                                                            \
    } while(0)
#define CASE_N(c) CASE_NN(#c, c)
    while(*s)
    {
        int ok = 0;
        CASE_N(g);
        CASE_N(mb);
        CASE_N(ic);
        CASE_N(id);
        CASE_N(ih);
        CASE_N(iw);
        CASE_N(oc);
        CASE_N(od);
        CASE_N(oh);
        CASE_N(ow);
        CASE_N(kd);
        CASE_N(kh);
        CASE_N(kw);
        CASE_N(sd);
        CASE_N(sh);
        CASE_N(sw);
        CASE_N(pd);
        CASE_N(ph);
        CASE_N(pw);
        CASE_N(dd);
        CASE_N(dh);
        CASE_N(dw);
        if(*s == 'n')
        {
            d.name = s + 1;
            break;
        }
        if(*s == '_')
            ++s;
        if(!ok)
            return XDNN_FAIL;
    }
#undef CASE_NN
#undef CASE_N

    if(d.has_groups && d.g <= 0)
        return XDNN_FAIL;
    if(d.ic == 0 || d.oc == 0)
        return XDNN_FAIL;
    if(d.sd <= 0 || d.sh <= 0 || d.sw <= 0)
        return XDNN_FAIL;

    auto compute_out = [](bool is_deconv, int64_t i, int64_t k, int64_t s, int64_t p, int64_t d) {
        if(is_deconv)
            return (i - 1) * s + (k - 1) * (d + 1) - 2 * p + 1;
        else
            return (i - ((k - 1) * (d + 1) + 1) + 2 * p) / s + 1;
    };
    auto compute_pad = [](bool is_deconv, int64_t o, int64_t i, int64_t k, int64_t s, int64_t d) {
        if(is_deconv)
            return ((i - 1) * s - o + ((k - 1) * (d + 1) + 1)) / 2;
        else
            return ((o - 1) * s - i + ((k - 1) * (d + 1) + 1)) / 2;
    };

    const bool no_d = (d.id | d.kd | d.od | d.dd) == 0 && d.sd == 1 && d.pd < 1;
    const bool no_h = (d.ih | d.kh | d.oh | d.dh) == 0 && d.sh == 1 && d.ph < 1;
    const bool no_w = (d.iw | d.kw | d.ow | d.dw) == 0 && d.sw == 1 && d.pw < 1;

    // printf("no_h:%d, no_w:%d, d.iw:%d\n", no_h, no_w, d.iw);

    if(!no_d)
    {
        if(!d.id || !d.kd)
            return XDNN_FAIL;
        if(!d.od)
        {
            if(d.pd < 0)
                d.pd = 0;
            d.od = compute_out(is_deconv, d.id, d.kd, d.sd, d.pd, d.dd);
            if(d.od <= 0)
                return XDNN_FAIL;
        }
        else if(d.pd < 0)
            d.pd = compute_pad(is_deconv, d.od, d.id, d.kd, d.sd, d.dd);
    }

    if(!no_h)
    {
        if(!d.ih || !d.kh)
            return XDNN_FAIL;
        if(!d.oh)
        {
            if(d.ph < 0)
                d.ph = 0;
            d.oh = compute_out(is_deconv, d.ih, d.kh, d.sh, d.ph, d.dh);
            if(d.oh <= 0)
                return XDNN_FAIL;
        }
        else if(d.ph < 0)
            d.ph = compute_pad(is_deconv, d.oh, d.ih, d.kh, d.sh, d.dh);
    }

    if(!no_w)
    {
        if(!d.iw || !d.kw)
            return XDNN_FAIL;
        if(!d.ow)
        {
            if(d.pw < 0)
                d.pw = 0;
            d.ow = compute_out(is_deconv, d.iw, d.kw, d.sw, d.pw, d.dw);
            if(d.ow <= 0)
                return XDNN_FAIL;
        }
        else if(d.pw < 0)
            d.pw = compute_pad(is_deconv, d.ow, d.iw, d.kw, d.sw, d.dw);
    }

    if(sanitize_desc(d.ndims,
                     {d.od, d.id, d.kd, d.sd, d.pd, d.dd},
                     {d.oh, d.ih, d.kh, d.sh, d.ph, d.dh},
                     {d.ow, d.iw, d.kw, d.sw, d.pw, d.dw},
                     {1, 1, 1, 1, 0, 0},
                     true) != XDNN_OK)
        return XDNN_FAIL;

    d.init_pad_r(is_deconv);

    // TODO: this is difference CK~OneDNN
    d.dh++;
    d.dw++;
    d.dd++;

    *desc = d;

    return XDNN_OK;
}

} // namespace ck
