#ifndef OMG_DDS_CORE_EXTERNAL_HPP_
#define OMG_DDS_CORE_EXTERNAL_HPP_

/*
 * Copyright(c) 2022 ZettaScale Technology and others
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <memory>
#include <dds/core/Exception.hpp>

namespace dds
{
namespace core
{

template <typename T>
class external {
  public:
    external() = default;
    external(T* p, bool locked = false): ptr_(p), locked_(locked) {;}
    external(std::shared_ptr<T> p): ptr_(p) {;}
    external(const external& other);
    ~external() = default;
    external& operator=(const external& other);
    T& operator*();
    const T& operator*() const;
    T* get() {return ptr_.get();}
    const T* get() const {return ptr_.get();}
    std::shared_ptr<T> get_shared_ptr() {return ptr_;}
    T* operator->() {return get();}
    const T* operator->() const {return get();}
    bool operator==(const external<T>& other) const {return other.ptr_ == ptr_;}
    bool operator!=(const external<T>& other) const {return !(*this == other);}
    operator bool() const {return static_cast<bool>(ptr_);}
    bool is_locked() const {return locked_;}
    void lock();
  private:
    std::shared_ptr<T> ptr_;
    bool locked_ = false;
};

template <typename T>
external<T>::external(const external<T>& other)
{
  if (other.is_locked())
    ptr_ = std::make_shared<T>(*other);  //if other is locked, this implies that it can be dereferenced, and deep copy
  else
    ptr_ = other.ptr_;  //unlocked means shallow copy
}

template <typename T>
external<T>& external<T>::operator=(const external<T>& other)
{
  if (is_locked())
    throw InvalidDataError("attempting to assign to locked external field");  //assignments to locked externals are not allowed

  if (other.is_locked()) {
    if (ptr_)
      *ptr_ = *other;  //copy over existing object
    else
      ptr_ = std::make_shared<T>(*other); //deep copy into new object
  } else {
    ptr_ = other.ptr_;  //shallow copy
  }

  return *this;
}

template <typename T>
T& external<T>::operator*()
{
  auto p = get();
  if (!p)
    throw NullReferenceError("attempting to dereference unset external field");  //dereferencing unset externals is not allowed

  return *p;
}

template <typename T>
const T& external<T>::operator*() const
{
  auto p = get();
  if (!p)
    throw NullReferenceError("attempting to dereference unset external field");  //dereferencing unset externals is not allowed

  return *p;
}

template <typename T>
void external<T>::lock()
{
  if (!*this)
    throw NullReferenceError("attempting to lock unset external field");  //locking unbound externals is not allowed

  locked_ = true;
}

}
}

#endif /* !defined(OMG_DDS_CORE_EXTERNAL_HPP_) */
