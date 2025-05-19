
template <std::size_t n,
          typename... Tf,
          typename... Ta,
          typename std::enable_if<n == sizeof...(Ta)>::type* = nullptr>
void mypArgs(const std::tuple<Ta...>&, void*)
{
}

template <std::size_t n,
          typename TTf,
          typename... Tf,
          typename... Ta,
          typename std::enable_if<n != sizeof...(Ta)>::type* = nullptr>
void mypArgs(const std::tuple<Ta...>& actuals, void** _vargs)
{
    using TTa = typename std::tuple_element<n, std::tuple<Ta...>>::type;
    #if 0
    // When using the kernel<<<...>>>(args...) syntax, hipcc allows references.
    // This assertion is obsolete.
    static_assert(!std::is_reference<TTf>{},
                  "A __global__ function cannot have a reference as one of its "
                  "arguments.");
    #endif 
#if defined(HIP_STRICT)
    static_assert(std::is_trivially_copyable<T>{},
                  "Only TriviallyCopyable types can be arguments to a __global__ "
                  "function");
#endif
    if constexpr(std::is_same<TTa, TTf>::value || std::is_convertible<TTa, TTf>::value)
        _vargs[n] = const_cast<void*>(reinterpret_cast<const void*>(&std::get<n>(actuals)));
    else
        static_assert("Incompatible types between formal and actual");

    return mypArgs<n + 1, Tf...>(actuals, _vargs);
}

template <typename... Formals, typename... Actuals>
void myvalidateArgsCountType(void (*kernel)(Formals...),
                             std::tuple<Actuals...> actuals,
                             void** _vars)
{
    (void)kernel;
    static_assert(sizeof...(Formals) == sizeof...(Actuals), "Argument Count Mismatch");
    mypArgs<0, Formals...>(actuals, _vars);
}

/**
 * @brief Launches kernel from the pointer address, with arguments and shared memory on stream.
 *
 * @param [in] function_address pointer to the Kernel to launch.
 * @param [in] numBlocks number of blocks.
 * @param [in] dimBlocks dimension of a block.
 * @param [in] args pointer to kernel arguments.
 * @param [in] sharedMemBytes  Amount of dynamic shared memory to allocate for this kernel.
 * HIP-Clang compiler provides support for extern shared declarations.
 * @param [in] stream  Stream where the kernel should be dispatched.
 * May be 0, in which case the default stream is used with associated synchronization rules.
 * @param [in] startEvent  If non-null, specified event will be updated to track the start time of
 * the kernel launch. The event must be created before calling this API.
 * @param [in] stopEvent  If non-null, specified event will be updated to track the stop time of
 * the kernel launch. The event must be created before calling this API.
 * @param [in] flags  The value of hipExtAnyOrderLaunch, signifies if kernel can be
 * launched in any order.
 * @returns #hipSuccess, #hipInvalidDeviceId, #hipErrorNotInitialized, #hipErrorInvalidValue.
 *
 */
extern "C" hipError_t hipExtLaunchKernel(const void* function_address,
                                         dim3 numBlocks,
                                         dim3 dimBlocks,
                                         void** args,
                                         size_t sharedMemBytes,
                                         hipStream_t stream,
                                         hipEvent_t startEvent,
                                         hipEvent_t stopEvent,
                                         int flags);

/**
 * @brief Launches kernel with dimention parameters and shared memory on stream with templated
 * kernel and arguments.
 *
 * @param [in] kernel  Kernel to launch.
 * @param [in] numBlocks  const number of blocks.
 * @param [in] dimBlocks  const dimension of a block.
 * @param [in] sharedMemBytes  Amount of dynamic shared memory to allocate for this kernel.
 * HIP-Clang compiler provides support for extern shared declarations.
 * @param [in] stream  Stream where the kernel should be dispatched.
 * May be 0, in which case the default stream is used with associated synchronization rules.
 * @param [in] startEvent  If non-null, specified event will be updated to track the start time of
 * the kernel launch. The event must be created before calling this API.
 * @param [in] stopEvent  If non-null, specified event will be updated to track the stop time of
 * the kernel launch. The event must be created before calling this API.
 * @param [in] flags  The value of hipExtAnyOrderLaunch, signifies if kernel can be
 * launched in any order.
 * @param [in] args  templated kernel arguments.
 *
 */
template <typename... Args, typename F = void (*)(Args...)>
inline void myhipExtLaunchKernelGGL(F kernel,
                                    const dim3& numBlocks,
                                    const dim3& dimBlocks,
                                    std::uint32_t sharedMemBytes,
                                    hipStream_t stream,
                                    hipEvent_t startEvent,
                                    hipEvent_t stopEvent,
                                    std::uint32_t flags,
                                    Args... args)
{
    constexpr size_t count = sizeof...(Args);
    auto tup_              = std::tuple<Args...>{args...};
    void* _Args[count];
    myvalidateArgsCountType(kernel, tup_, _Args);

    auto k = reinterpret_cast<void*>(kernel);
    (void)hipExtLaunchKernel(k,
                             numBlocks,
                             dimBlocks,
                             _Args,
                             sharedMemBytes,
                             stream,
                             startEvent,
                             stopEvent,
                             static_cast<int>(flags));
}
