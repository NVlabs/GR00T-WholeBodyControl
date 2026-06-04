/**
 * @file utils.hpp
 * @brief Core utility types: thread-safe DataBuffer and CRC32 checksum.
 *
 * DataBuffer<T> is the primary mechanism for passing data between threads in
 * the G1 deployment stack.  It uses a shared_mutex (multiple-reader /
 * single-writer) and stores data via shared_ptr for lock-free read-side
 * copies.
 *
 * TimestampedData<T> pairs a shared_ptr<const T> with a steady-clock
 * timestamp so consumers can check data freshness.
 */

#ifndef UTILS_HPP
#define UTILS_HPP

#include <memory>
#include <chrono>
#include <shared_mutex>
#include <cstdint>
#include <cmath>
#include <array>

/**
 * @brief Immutable data snapshot with a monotonic timestamp.
 * @tparam T Type of the stored payload.
 *
 * Returned by DataBuffer::GetDataWithTime().  The `data` shared_ptr is
 * safe to hold beyond the buffer's lock scope.
 */
template <typename T> 
struct TimestampedData {
    std::shared_ptr<const T> data;
    std::chrono::steady_clock::time_point timestamp;

    TimestampedData()
      : data(nullptr),
        timestamp {} {}

    TimestampedData(std::shared_ptr<const T> d, std::chrono::steady_clock::time_point t)
      : data(d),
        timestamp(t) {}

    bool HasData() const { return data != nullptr; }

    double GetAgeMs() const {
      if (!HasData()) return -1.0;
      auto now = std::chrono::steady_clock::now();
      auto age = std::chrono::duration_cast<std::chrono::microseconds>(now - timestamp);
      return age.count() / 1000.0;
    }
};

/**
 * @brief Thread-safe data buffer with timestamp tracking
 * @tparam T Type of data to store
 */
template <typename T> 
class DataBuffer {
  public:
    void SetData(const T& newData) {
      std::unique_lock<std::shared_mutex> lock(mutex);
      data = std::make_shared<T>(newData);
      last_update_time = std::chrono::steady_clock::now();
    }
    
    void SetData(T&& newData) {
      std::unique_lock<std::shared_mutex> lock(mutex);
      data = std::make_shared<T>(std::move(newData));
      last_update_time = std::chrono::steady_clock::now();
    }

    // Return data with timestamp
    TimestampedData<T> GetDataWithTime() {
      std::shared_lock<std::shared_mutex> lock(mutex);
      if (data) { return TimestampedData<T>(data, last_update_time); }
      return TimestampedData<T>(); // Empty
    }

    // Const overload for read-only access
    TimestampedData<T> GetDataWithTime() const {
      std::shared_lock<std::shared_mutex> lock(mutex);
      if (data) { return TimestampedData<T>(data, last_update_time); }
      return TimestampedData<T>(); // Empty
    }

    void Clear() {
      std::unique_lock<std::shared_mutex> lock(mutex);
      data = nullptr;
      last_update_time = std::chrono::steady_clock::time_point {}; // Reset timestamp
    }
    
  private:
    std::shared_ptr<T> data;
    std::chrono::steady_clock::time_point last_update_time;
    mutable std::shared_mutex mutex;
};

/**
 * @brief Calculate CRC32 checksum for data validation
 * @param ptr Pointer to data array
 * @param len Length of data array
 * @return CRC32 checksum
 */
inline uint32_t Crc32Core(uint32_t* ptr, uint32_t len) {
  uint32_t xbit = 0;
  uint32_t data = 0;
  uint32_t CRC32 = 0xFFFFFFFF;
  const uint32_t dwPolynomial = 0x04c11db7;
  for (uint32_t i = 0; i < len; i++) {
    xbit = 1 << 31;
    data = ptr[i];
    for (uint32_t bits = 0; bits < 32; bits++) {
      if (CRC32 & 0x80000000) {
        CRC32 <<= 1;
        CRC32 ^= dwPolynomial;
      } else CRC32 <<= 1;
      if (data & xbit) CRC32 ^= dwPolynomial;

      xbit >>= 1;
    }
  }
  return CRC32;
}

class CounterDebouncer {
  public:
    CounterDebouncer(
      int threshold = 20,
      int max_counter = 100,
      int increment = 10,
      int decrement = 1)
      : threshold_(threshold),
      max_counter_(max_counter),
      increment_(increment),
      decrement_(decrement),
      counter_(0)
    {}

    bool update(bool value) {
      if (value) {
        counter_ += increment_;
        if (counter_ > max_counter_) {
          counter_ = max_counter_;
        }
      } else {
        counter_ -= decrement_;
        if (counter_ < 0) {
          counter_ = 0;
        }
      }

      return state();
    }

    bool state() const {
      return counter_ >= threshold_;
    }
  private:
    int threshold_{};
    int increment_{};
    int decrement_{};
    int max_counter_{};
    int counter_{};
};

/**
 * @brief Rolling statistics buffer using Welford's algorithm for numerically stable
 *        online computation of mean and variance.
 *
 * Maintains a circular buffer of up to BUFFER_CAPACITY doubles. Uses Welford's
 * algorithm which is more numerically stable than naive sum/sum-of-squares methods,
 * especially for large datasets or values with large magnitudes.
 *
 * When the buffer is full, new values overwrite the oldest values using the
 * sliding window variant of Welford's algorithm.
 *
 * @tparam BUFFER_CAPACITY Maximum number of values to store (default 1000).
 */
template <size_t BUFFER_CAPACITY = 1000>
class RollingStats {
  public:
    RollingStats()
      : buffer_{},
        size_(0),
        head_(0),
        mean_(0.0),
        m2_(0.0)
    {}

    /**
     * @brief Push a new value into the rolling buffer using Welford's algorithm.
     *
     * If the buffer is not full, uses standard Welford's online algorithm.
     * If the buffer is full, uses the sliding window variant that removes
     * the oldest value's contribution before adding the new value.
     *
     * @param value The new value to add.
     */
    void push(double value) {
        if (size_ < BUFFER_CAPACITY) {
            // Buffer not full: standard Welford's algorithm
            buffer_[head_] = value;
            size_++;

            // Welford's online update:
            // delta = x - mean
            // mean += delta / n
            // delta2 = x - mean (after mean update)
            // M2 += delta * delta2
            double delta = value - mean_;
            mean_ += delta / static_cast<double>(size_);
            double delta2 = value - mean_;
            m2_ += delta * delta2;

            head_ = (head_ + 1) % BUFFER_CAPACITY;
        } else {
            // Buffer full: sliding window Welford's update
            double old_value = buffer_[head_];
            buffer_[head_] = value;
            head_ = (head_ + 1) % BUFFER_CAPACITY;

            // Sliding window update:
            // 1. Compute deltas relative to old mean
            // 2. Update mean: new_mean = old_mean + (x_new - x_old) / n
            // 3. Update M2: M2 += (x_new - old_mean)(x_new - new_mean)
            //                   - (x_old - old_mean)(x_old - new_mean)
            double delta_old = old_value - mean_;
            double delta_new = value - mean_;
            mean_ += (value - old_value) / static_cast<double>(size_);
            double delta_old_new = old_value - mean_;
            double delta_new_new = value - mean_;
            m2_ += (delta_new * delta_new_new) - (delta_old * delta_old_new);
        }
    }

    /**
     * @brief Get the mean of all values in the buffer.
     * @return The mean, or 0.0 if the buffer is empty.
     */
    double mean() const {
        return mean_;
    }

    /**
     * @brief Get the population variance of all values in the buffer.
     *
     * Computed as M2 / n using Welford's algorithm.
     *
     * @return The variance, or 0.0 if the buffer is empty.
     */
    double variance() const {
        if (size_ == 0) return 0.0;
        double var = m2_ / static_cast<double>(size_);
        // Guard against small negative values due to floating-point error
        return var < 0.0 ? 0.0 : var;
    }

    /**
     * @brief Get the population standard deviation of all values in the buffer.
     * @return The standard deviation, or 0.0 if the buffer is empty.
     */
    double stddev() const {
        return std::sqrt(variance());
    }

    /**
     * @brief Get the current number of values in the buffer.
     * @return Size (0 to BUFFER_CAPACITY).
     */
    size_t size() const { return size_; }

    /**
     * @brief Check if the buffer is full.
     * @return true if size == BUFFER_CAPACITY.
     */
    bool full() const { return size_ == BUFFER_CAPACITY; }

    /**
     * @brief Clear all values and reset statistics.
     */
    void clear() {
        size_ = 0;
        head_ = 0;
        mean_ = 0.0;
        m2_ = 0.0;
    }

  private:
    std::array<double, BUFFER_CAPACITY> buffer_;
    size_t size_;   ///< Number of values currently in the buffer (0 to BUFFER_CAPACITY)
    size_t head_;   ///< Next write position in the circular buffer
    double mean_;   ///< Running mean (Welford's algorithm)
    double m2_;     ///< Sum of squared differences from the mean (Welford's algorithm)
};

#endif // UTILS_HPP
