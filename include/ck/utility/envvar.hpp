#ifndef __ENVVAR_HPP
#define __ENVVAR_HPP

#include <cstdlib>

namespace ck {

static inline int getenv_int(const char* var_name, int default_int)
{
    char* v = ::getenv(var_name);
    int r   = default_int;
    if(v)
        r = ::atoi(v);
    return r;
}

static inline char* getenv_str(const char* var_name, char* default_str)
{
    char* v = ::getenv(var_name);
    if(v)
        return v;
    return default_str;
}

} // namespace ck
#endif
