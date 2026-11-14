#pragma once

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

namespace hft::core {

/**
 * CPU Core Pinning and NUMA Affinity Controller.
 * Binds critical processing threads to dedicated physical cores, preventing OS context switches.
 */
class CpuPinning {
public:
    static inline bool pin_thread(int core_id) noexcept {
#if defined(__linux__)
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        pthread_t current_thread = pthread_self();
        return pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) == 0;
#elif defined(_WIN32)
        DWORD_PTR mask = (1ULL << core_id);
        DWORD_PTR result = SetThreadAffinityMask(GetCurrentThread(), mask);
        return result != 0;
#else
        (void)core_id;
        return false;
#endif
    }
};

} // namespace hft::core
