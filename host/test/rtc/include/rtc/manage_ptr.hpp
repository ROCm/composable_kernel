#ifndef GUARD_HOST_TEST_RTC_INCLUDE_RTC_MANAGE_POINTER
#define GUARD_HOST_TEST_RTC_INCLUDE_RTC_MANAGE_POINTER

#include <type_traits>
#include <memory>

namespace rtc {
template <class F, F f>
struct manage_deleter
{
    template <class T>
    void operator()(T* x) const
    {
        if(x != nullptr)
        {
            (void)f(x);
        }
    }
};

struct null_deleter
{
    template <class T>
    void operator()(T*) const
    {
    }
};

template <class T, class F, F f>
using manage_ptr = std::unique_ptr<T, manage_deleter<F, f>>;

#define RTC_MANAGE_PTR(T, F) rtc::manage_ptr<std::remove_pointer_t<T>, decltype(&F), &F>

} // namespace rtc

#endif
