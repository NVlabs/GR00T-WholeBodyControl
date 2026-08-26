#pragma once

#include <vector>
#include <stdexcept>
#include <cuda_runtime_api.h>

// Allocator using pinned memory allocations to bypass the extra memcpy from pageable host memory.
//
// Host data allocations (CPU) are pageable by default and the GPU cannot access data directly
// from pageable host memory. When a data transfer from pageable host memory to device memory
// is invoked, the CUDA driver must first allocate a temporary page-locked, or “pinned”, host array,
// copy the host data to the pinned array, and then transfer the data from the pinned array to device
// memory.

template<typename T>
class PinnedAllocator
{
public:
    using value_type = T;

    PinnedAllocator() noexcept = default;

    template <typename U>
    PinnedAllocator(const PinnedAllocator<U>&) noexcept {}

    T* allocate(size_t n)
    {
        T* tmp;
        auto error = cudaMallocHost((void**)&tmp, n * sizeof(T));
        if (error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));

        return tmp;
    }

    void deallocate(T* p, size_t n)
    {
        if (p)
        {
            auto error = cudaFreeHost(p);
            if (error != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(error));
        }
    }
};

// Stateless allocator type, equality operators always return true...
template <class T, class U>
bool operator==(PinnedAllocator<T> const&, PinnedAllocator<U> const&) { return true; }
template <class T, class U>
bool operator!=(PinnedAllocator<T> const&, PinnedAllocator<U> const&) { return false; }

template<typename T>
using TPinnedVector = std::vector<T, PinnedAllocator<T>>;
