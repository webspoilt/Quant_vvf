#pragma once

#include <cstddef>
#include <cstdint>
#include <new>

namespace hft::core {

/**
 * Pre-allocated Linear Arena Memory Pool
 * Guarantees zero heap allocation (malloc/free) in the execution path.
 */
template <size_t PoolSize>
class PreallocatedMemoryArena {
public:
    PreallocatedMemoryArena() noexcept : offset_(0) {}

    void* allocate(size_t bytes, size_t alignment = 64) noexcept {
        size_t current = reinterpret_cast<size_t>(storage_ + offset_);
        size_t aligned = (current + (alignment - 1)) & ~(alignment - 1);
        size_t new_offset = (aligned - reinterpret_cast<size_t>(storage_)) + bytes;

        if (new_offset > PoolSize) [[unlikely]] {
            return nullptr; // Out of arena space
        }

        offset_ = new_offset;
        return reinterpret_cast<void*>(aligned);
    }

    void reset() noexcept {
        offset_ = 0;
    }

    [[nodiscard]] size_t used_bytes() const noexcept {
        return offset_;
    }

    [[nodiscard]] size_t available_bytes() const noexcept {
        return PoolSize > offset_ ? PoolSize - offset_ : 0;
    }

private:
    alignas(64) uint8_t storage_[PoolSize];
    size_t offset_{0};
};

} // namespace hft::core
