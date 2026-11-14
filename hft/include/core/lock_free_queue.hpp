#pragma once

#include <atomic>
#include <cstddef>
#include <new>
#include <type_traits>
#include <utility>

namespace hft::core {

#ifndef __cpp_lib_hardware_interference_size
constexpr size_t CACHE_LINE_SIZE = 64;
#else
constexpr size_t CACHE_LINE_SIZE = std::hardware_destructive_interference_size;
#endif

/**
 * High-Performance Single-Producer Single-Consumer (SPSC) Lock-Free Ring Buffer.
 * - Cache-line aligned read and write pointers to avoid false sharing.
 * - Zero dynamic allocations after initialization.
 * - Memory barriers guarantee sequential consistency across CPU cores.
 */
template <typename T, size_t Capacity>
class SpscLockFreeQueue {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable for zero-copy memory transfers");

public:
    SpscLockFreeQueue() : head_(0), tail_(0) {}
    ~SpscLockFreeQueue() = default;

    SpscLockFreeQueue(const SpscLockFreeQueue&) = delete;
    SpscLockFreeQueue& operator=(const SpscLockFreeQueue&) = delete;

    template <typename... Args>
    bool emplace(Args&&... args) noexcept {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        const size_t current_head = head_.load(std::memory_order_acquire);

        if ((current_tail - current_head) >= Capacity) [[unlikely]] {
            return false; // Queue is full
        }

        new (&buffer_[current_tail & BUFFER_MASK]) T(std::forward<Args>(args)...);
        tail_.store(current_tail + 1, std::memory_order_release);
        return true;
    }

    bool push(const T& item) noexcept {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        const size_t current_head = head_.load(std::memory_order_acquire);

        if ((current_tail - current_head) >= Capacity) [[unlikely]] {
            return false; // Queue is full
        }

        buffer_[current_tail & BUFFER_MASK] = item;
        tail_.store(current_tail + 1, std::memory_order_release);
        return true;
    }

    bool pop(T& out_item) noexcept {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        const size_t current_tail = tail_.load(std::memory_order_acquire);

        if (current_head == current_tail) [[unlikely]] {
            return false; // Queue is empty
        }

        out_item = buffer_[current_head & BUFFER_MASK];
        head_.store(current_head + 1, std::memory_order_release);
        return true;
    }

    [[nodiscard]] bool empty() const noexcept {
        return head_.load(std::memory_order_relaxed) == tail_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] size_t size() const noexcept {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t tail = tail_.load(std::memory_order_relaxed);
        return (tail >= head) ? (tail - head) : 0;
    }

private:
    static constexpr size_t BUFFER_MASK = Capacity - 1;

    alignas(CACHE_LINE_SIZE) std::atomic<size_t> tail_{0};
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> head_{0};
    alignas(CACHE_LINE_SIZE) T buffer_[Capacity];
};

} // namespace hft::core
