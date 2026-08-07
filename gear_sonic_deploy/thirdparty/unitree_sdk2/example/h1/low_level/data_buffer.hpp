#pragma once

#include <deque>
#include <memory>
#include <mutex>
#include <shared_mutex>

template <typename T> class DataBuffer {
public:
  void SetData(const T &newData) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    data = std::make_shared<T>(newData);
  }

  std::shared_ptr<const T> GetData() {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return data ? data : nullptr;
  }

  void Clear() {
    std::unique_lock<std::shared_mutex> lock(mutex);
    data = nullptr;
  }

private:
  std::shared_ptr<T> data;
  std::shared_mutex mutex;
};
