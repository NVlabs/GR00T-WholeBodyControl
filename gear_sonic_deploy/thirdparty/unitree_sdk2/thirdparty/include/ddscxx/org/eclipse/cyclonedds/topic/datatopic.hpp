/*
 * Copyright(c) 2020 to 2022 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef DDSCXXDATATOPIC_HPP_
#define DDSCXXDATATOPIC_HPP_

#include <memory>
#include <string>
#include <cstring>
#include <vector>
#include <atomic>

#include "dds/ddsrt/md5.h"
#include "org/eclipse/cyclonedds/core/cdr/basic_cdr_ser.hpp"
#include "org/eclipse/cyclonedds/core/cdr/extended_cdr_v1_ser.hpp"
#include "org/eclipse/cyclonedds/core/cdr/extended_cdr_v2_ser.hpp"
#include "org/eclipse/cyclonedds/core/cdr/fragchain.hpp"
#include "org/eclipse/cyclonedds/topic/TopicTraits.hpp"
#include "org/eclipse/cyclonedds/topic/hash.hpp"

#ifdef DDSCXX_HAS_SHM
#include "dds/ddsi/ddsi_shm_transport.h"
#endif

constexpr size_t CDR_HEADER_SIZE = 4U;
#define BO_LITTLE   0X01
#define PLAIN_CDR   0x00
#define PL_CDR      0x02
#define PLAIN_CDR2  0x06
#define D_CDR       0x08
#define PL_CDR2     0x0A

// macro to check if a pointer is nullptr and return false
#define CHECK_FOR_NULL(val)  if ((val) == nullptr) return false;

using org::eclipse::cyclonedds::core::cdr::endianness;
using org::eclipse::cyclonedds::core::cdr::native_endianness;
using org::eclipse::cyclonedds::core::cdr::cdr_stream;
using org::eclipse::cyclonedds::core::cdr::basic_cdr_stream;
using org::eclipse::cyclonedds::core::cdr::xcdr_v1_stream;
using org::eclipse::cyclonedds::core::cdr::xcdr_v2_stream;
using org::eclipse::cyclonedds::core::cdr::extensibility;
using org::eclipse::cyclonedds::core::cdr::encoding_version;
using org::eclipse::cyclonedds::topic::TopicTraits;

template<typename T, class S>
bool get_serialized_size(const T& sample, bool as_key, size_t &sz);

template<typename T>
bool to_key(const T& tokey, ddsi_keyhash_t& hash)
{
  if (TopicTraits<T>::isKeyless())
  {
    memset(&(hash.value), 0x0, sizeof(hash.value));  //just set all key bytes to 0 as all instances have the same hash value, and hashing is pointless
    return true;
  } else
  {
    basic_cdr_stream str(endianness::big_endian);
    size_t sz = 0;
    if (!get_serialized_size<T, basic_cdr_stream>(tokey, true, sz)) {
      assert(false);
      return false;
    }
    size_t padding = 0;
    if (sz < 16)
      padding = (16 - sz % 16)%16;
    std::vector<unsigned char> buffer(sz + padding);
    if (padding)
      memset(buffer.data() + sz, 0x0, padding);
    str.set_buffer(buffer.data(), sz);
    if (!write(str, tokey, true)) {
      assert(false);
      return false;
    }
    static thread_local bool (*fptr)(const std::vector<unsigned char>&, ddsi_keyhash_t&) = NULL;
    if (fptr == NULL)
    {
      max(str, tokey, true);
      if (str.position() <= 16)
      {
        //bind to unmodified function which just copies buffer into the keyhash
        fptr = &org::eclipse::cyclonedds::topic::simple_key;
      }
      else
      {
        //bind to MD5 hash function
        fptr = &org::eclipse::cyclonedds::topic::complex_key;
      }
    }
    return (*fptr)(buffer, hash);
  }
}

static inline void* calc_offset(void* ptr, ptrdiff_t n)
{
  return static_cast<void*>(static_cast<unsigned char*>(ptr) + n);
}

static inline const void* calc_offset(const void* ptr, ptrdiff_t n)
{
  return static_cast<const void*>(static_cast<const unsigned char*>(ptr) + n);
}

template<typename T,
         class S,
         std::enable_if_t<std::is_same<basic_cdr_stream, S>::value, bool> = true >
bool write_header(void *buffer)
{
  CHECK_FOR_NULL(buffer);
  memset(buffer, 0x0, 4);

  auto ptr = static_cast<unsigned char*>(calc_offset(buffer, 1));

  assert(TopicTraits<T>::getExtensibility() == extensibility::ext_final);
  *ptr = PLAIN_CDR;

  if (native_endianness() == endianness::little_endian)
    *ptr |= BO_LITTLE;

  return true;
}

template<typename T,
         class S,
         std::enable_if_t<std::is_same<xcdr_v2_stream, S>::value, bool> = true >
bool write_header(void *buffer)
{
  CHECK_FOR_NULL(buffer);
  memset(buffer, 0x0, 4);

  auto ptr = static_cast<unsigned char*>(calc_offset(buffer, 1));

  switch (TopicTraits<T>::getExtensibility()) {
    case extensibility::ext_final:
      *ptr = PLAIN_CDR2;
      break;
    case extensibility::ext_appendable:
      *ptr = D_CDR;
      break;
    case extensibility::ext_mutable:
      *ptr = PL_CDR2;
      break;
  }

  if (native_endianness() == endianness::little_endian)
    *ptr |= BO_LITTLE;

  return true;
}

template<typename T,
         class S,
         std::enable_if_t<std::is_same<xcdr_v1_stream, S>::value, bool> = true >
bool write_header(void *buffer)
{
  CHECK_FOR_NULL(buffer);
  memset(buffer, 0x0, 4);

  auto ptr = static_cast<unsigned char*>(calc_offset(buffer, 1));

  switch (TopicTraits<T>::getExtensibility()) {
    case extensibility::ext_final:
    case extensibility::ext_appendable:
      *ptr = PLAIN_CDR;
      break;
    case extensibility::ext_mutable:
      *ptr = PL_CDR;
      break;
  }

  if (native_endianness() == endianness::little_endian)
    *ptr |= BO_LITTLE;

  return true;
}

template<typename T>
bool finish_header(void *buffer, size_t bytes_written)
{
  CHECK_FOR_NULL(buffer);
  auto alignbytes = static_cast<unsigned char>(4 % (4 - bytes_written % 4));
  auto ptr = static_cast<unsigned char*>(calc_offset(buffer, 3));

  *ptr = alignbytes;

  return true;
}

template<typename T>
bool read_header(const void *buffer, encoding_version &ver, endianness &end)
{
  CHECK_FOR_NULL(buffer);
  auto ptr = static_cast<const unsigned char*>(calc_offset(buffer, 1));

  if (*ptr & BO_LITTLE)
    end = endianness::little_endian;
  else
    end = endianness::big_endian;

  auto field = *ptr & ~BO_LITTLE;
  switch (TopicTraits<T>::getExtensibility()) {
    case extensibility::ext_final:
      switch (field) {
        case PLAIN_CDR:
          if (TopicTraits<T>::allowableEncodings() & DDS_DATA_REPRESENTATION_FLAG_XCDR1)
            ver = encoding_version::basic_cdr;
          else
            ver = encoding_version::xcdr_v1;
          break;
        case PLAIN_CDR2:
          ver = encoding_version::xcdr_v2;
          break;
        default:
          return false;
      }
      break;
    case extensibility::ext_appendable:
      switch (field) {
        case PL_CDR:
          ver = encoding_version::xcdr_v1;
          break;
        case D_CDR:
          ver = encoding_version::xcdr_v2;
          break;
        default:
          return false;
      }
      break;
    case extensibility::ext_mutable:
      switch (field) {
        case PL_CDR:
          ver = encoding_version::xcdr_v1;
          break;
        case PL_CDR2:
          ver = encoding_version::xcdr_v2;
          break;
        default:
          return false;
      }
      break;
    default:
      return false;
  }

  return true;
}

template<typename T, class S, bool K>
bool get_serialized_fixed_size(const T& sample, size_t &sz)
{
  static thread_local size_t serialized_size = 0;
  static thread_local std::mutex mtx;
  static thread_local std::atomic_bool initialized {false};
  if (initialized.load(std::memory_order_relaxed)) {
    sz = serialized_size;
    return true;
  }
  std::lock_guard<std::mutex> lock(mtx);
  if (initialized.load(std::memory_order_relaxed)) {
    sz = serialized_size;
    return true;
  }
  S str;
  if (!move(str, sample, K))
    return false;
  serialized_size = str.position();
  initialized.store(true, std::memory_order_release);
  sz = serialized_size;
  return true;
}

template<typename T, class S>
bool get_serialized_size(const T& sample, bool as_key, size_t &sz)
{
  if (TopicTraits<T>::isSelfContained()) {
    if ((as_key && !get_serialized_fixed_size<T,S,true>(sample, sz)) ||
        (!as_key && !get_serialized_fixed_size<T,S,false>(sample, sz)))
      return false;
  } else {
    S str;
    if (!move(str, sample, as_key))
      return false;
    sz = str.position();
  }

  return true;
}

template<typename T, class S>
bool serialize_into(void *buffer,
                    size_t buf_sz,
                    const T &sample,
                    bool as_key)
{
  CHECK_FOR_NULL(buffer);
  assert(buf_sz >= CDR_HEADER_SIZE);

  S str;
  str.set_buffer(calc_offset(buffer, CDR_HEADER_SIZE), buf_sz-CDR_HEADER_SIZE);
  return (write_header<T,S>(buffer)
        && write(str, sample, as_key)
        && finish_header<T>(buffer, buf_sz));
}

/// \brief De-serialize the buffer into the sample
/// \param[in] buffer The buffer to be de-serialized
/// \param[out] sample Type to which the buffer will be de-serialized
/// \param[in] data_kind The data kind (data, or key)
/// \tparam T The sample type
/// \return True if the deserialization is successful
///         False if the deserialization failed
template <typename T>
bool deserialize_sample_from_buffer(void *buffer,
                                    size_t buf_sz,
                                    T &sample,
                                    const ddsi_serdata_kind data_kind=SDK_DATA)
{
  CHECK_FOR_NULL(buffer);
  assert(data_kind != SDK_EMPTY);

  encoding_version ver;
  endianness end;
  if (!read_header<T>(buffer, ver, end))
    return false;

  switch (ver) {
    case encoding_version::basic_cdr:
      {
        basic_cdr_stream str(end);
        str.set_buffer(calc_offset(buffer, CDR_HEADER_SIZE), buf_sz-CDR_HEADER_SIZE);
        return read(str, sample, data_kind == SDK_KEY);
      }
      break;
    case encoding_version::xcdr_v1:
      {
        xcdr_v1_stream str(end);
        str.set_buffer(calc_offset(buffer, CDR_HEADER_SIZE), buf_sz-CDR_HEADER_SIZE);
        return read(str, sample, data_kind == SDK_KEY);
      }
      break;
    case encoding_version::xcdr_v2:
      {
        xcdr_v2_stream str(end);
        str.set_buffer(calc_offset(buffer, CDR_HEADER_SIZE), buf_sz-CDR_HEADER_SIZE);
        return read(str, sample, data_kind == SDK_KEY);
      }
      break;
    default:
      return false;
  }
}

template <typename T> class ddscxx_serdata;

template <typename T>
bool serdata_eqkey(const ddsi_serdata* a, const ddsi_serdata* b)
{
  auto s_a = static_cast<const ddscxx_serdata<T>*>(a);
  auto s_b = static_cast<const ddscxx_serdata<T>*>(b);

  return 0 == memcmp(s_a->key().value, s_b->key().value, 16);
}

template <typename T>
uint32_t serdata_size(const ddsi_serdata* dcmn)
{
  return static_cast<uint32_t>(static_cast<const ddscxx_serdata<T>*>(dcmn)->size());
}

template <typename T>
ddsi_serdata *serdata_from_ser(
  const ddsi_sertype* type,
  enum ddsi_serdata_kind kind,
  const struct nn_rdata* fragchain,
  size_t size)
{
  auto d = new ddscxx_serdata<T>(type, kind);
  d->resize(size);
  auto cursor = static_cast<unsigned char*>(d->data());
  org::eclipse::cyclone::core::cdr::serdata_from_ser_copyin_fragchain (cursor, fragchain, size);

  if (d->getT())
  {
    d->key_md5_hashed() = to_key(*d->getT(), d->key());
    d->populate_hash();
  }
  else
  {
    delete d;
    d = nullptr;
  }

  return d;
}

template <typename T>
ddsi_serdata *serdata_from_ser_iov(
  const ddsi_sertype* type,
  enum ddsi_serdata_kind kind,
  ddsrt_msg_iovlen_t niov,
  const ddsrt_iovec_t* iov,
  size_t size)
{
  auto d = new ddscxx_serdata<T>(type, kind);
  d->resize(size);

  size_t off = 0;
  auto cursor = static_cast<unsigned char*>(d->data());
  for (ddsrt_msg_iovlen_t i = 0; i < niov && off < size; i++)
  {
    size_t n_bytes = iov[i].iov_len;
    if (n_bytes + off > size) n_bytes = size - off;
    memcpy(cursor, iov[i].iov_base, n_bytes);
    cursor += n_bytes;
    off += n_bytes;
  }

  T* ptr = d->getT();
  if (ptr) {
    d->key_md5_hashed() = to_key(*ptr, d->key());
    d->populate_hash();
  } else {
    delete d;
    d = nullptr;
  }

  return d;

}

template <typename T>
ddsi_serdata *serdata_from_keyhash(
  const ddsi_sertype* type,
  const struct ddsi_keyhash* keyhash)
{
  (void)keyhash;
  (void)type;
  //replace with (if key_size_max <= 16) then populate the data class with the key hash
  return nullptr;
}

template <typename T, class S>
ddsi_serdata *serdata_from_sample(
  const ddsi_sertype* typecmn,
  enum ddsi_serdata_kind kind,
  const void* sample)
{
  assert(kind != SDK_EMPTY);
  auto d = new ddscxx_serdata<T>(typecmn, kind);
  const auto& msg = *static_cast<const T*>(sample);
  size_t sz = 0;

  if (!get_serialized_size<T,S>(msg, kind == SDK_KEY, sz))
    goto failure;

  sz += CDR_HEADER_SIZE;
  d->resize(sz);

  if (!serialize_into<T,S>(d->data(), sz, msg, kind == SDK_KEY))
    goto failure;

  d->key_md5_hashed() = to_key(msg, d->key());
  d->setT(&msg);
  d->populate_hash();
  return d;

failure:
  if (d)
    delete d;
  return nullptr;
}

template <typename T>
void serdata_to_ser(const ddsi_serdata* dcmn, size_t off, size_t sz, void* buf)
{
  auto d = static_cast<const ddscxx_serdata<T>*>(dcmn);
  memcpy(buf, calc_offset(d->data(), static_cast<ptrdiff_t>(off)), sz);
}

template <typename T>
ddsi_serdata *serdata_to_ser_ref(
  const ddsi_serdata* dcmn, size_t off,
  size_t sz, ddsrt_iovec_t* ref)
{
  auto d = static_cast<const ddscxx_serdata<T>*>(dcmn);
  ref->iov_base = calc_offset(d->data(), static_cast<ptrdiff_t>(off));
  ref->iov_len = static_cast<ddsrt_iov_len_t>(sz);
  return ddsi_serdata_ref(d);
}

template <typename T>
void serdata_to_ser_unref(ddsi_serdata* dcmn, const ddsrt_iovec_t* ref)
{
  static_cast<void>(ref);    // unused
  ddsi_serdata_unref(static_cast<ddscxx_serdata<T>*>(dcmn));
}

template <typename T>
bool serdata_to_sample(
  const ddsi_serdata* dcmn, void* sample, void** bufptr,
  void* buflim)
{
  (void)bufptr;
  (void)buflim;

  auto typed_sample_ptr = static_cast<T*>(sample);
  // cast away const, with the reasoning that we don't modify the underlying ddsi_serdata which
  // is actually const, we only modify the ddscxx_serdata non const contents
  auto d = const_cast<ddscxx_serdata<T>*>(static_cast<const ddscxx_serdata<T>*>(dcmn));

  auto t_ptr = d->getT();
  if (!t_ptr)
    return false;

  *typed_sample_ptr = *t_ptr;
  return true;
}

template <typename T, class S>
ddsi_serdata *serdata_to_untyped(const ddsi_serdata* dcmn)
{
  /* Cast away const: the serialized ddsi_serdata itself is not touched: only its C++ representation
   * in the C++ wrapper may initialized if this was not done before. So conceptually the const for
   * ddsi_serdata is not violated.
   */
  auto d = const_cast<ddscxx_serdata<T>*>(static_cast<const ddscxx_serdata<T>*>(dcmn));
  auto d1 = new ddscxx_serdata<T>(d->type, SDK_KEY);
  d1->type = nullptr;

  auto t = d->getT();
  size_t sz = 0;
  if (t == nullptr || !get_serialized_size<T,S>(*t, true, sz))
    goto failure;

  sz += CDR_HEADER_SIZE;
  d1->resize(sz);

  if (!serialize_into<T,S>(d1->data(), sz, *t, true))
    goto failure;

  d1->key_md5_hashed() = to_key(*t, d1->key());
  d1->hash = d->hash;

  return d1;

failure:
  delete d1;
  return nullptr;
}

template <typename T>
bool serdata_untyped_to_sample(
  const ddsi_sertype* type,
  const ddsi_serdata* dcmn, void* sample,
  void** bufptr, void* buflim)
{
  (void)type;
  (void)bufptr;
  (void)buflim;

  auto d = static_cast<const ddscxx_serdata<T>*>(dcmn);
  T* ptr = static_cast<T*>(sample);

  return deserialize_sample_from_buffer(d->data(), d->size(), *ptr, SDK_KEY);
}

template <typename T>
void serdata_free(ddsi_serdata* dcmn)
{
  auto* d = static_cast<ddscxx_serdata<T>*>(dcmn);

#ifdef DDSCXX_HAS_SHM
  if (d->iox_chunk && d->iox_subscriber)
  {
    // Explicit cast to iox_subscriber is required here, since the C++ binding has no notion of
    // iox subscriber, but the underlying C API expects this to be a typed iox_subscriber.
    // TODO (Sumanth), Fix this when we cleanup the interfaces to not use iceoryx directly in
    //  the C++ plugin
    free_iox_chunk(static_cast<iox_sub_t *>(d->iox_subscriber), &d->iox_chunk);
  }
#endif
  delete d;
}

template <typename T>
size_t serdata_print(
  const ddsi_sertype* tpcmn, const ddsi_serdata* dcmn, char* buf, size_t bufsize)
{
  (void)tpcmn;
  (void)dcmn;
  //implementation to follow!!!
  if (bufsize > 0)
    buf[0] = 0x0;
  return 0;
}

template <typename T>
void serdata_get_keyhash(
  const ddsi_serdata* d, struct ddsi_keyhash* buf,
  bool force_md5)
{
  auto ptr = static_cast<const ddscxx_serdata<T>*>(d);
  assert(buf);
  if (force_md5 && !ptr->key_md5_hashed())
  {
    ddsrt_md5_state_t md5st;
    ddsrt_md5_init(&md5st);
    ddsrt_md5_append(&md5st, static_cast<const ddsrt_md5_byte_t*>(ptr->key().value), 16);
    ddsrt_md5_finish(&md5st, static_cast<ddsrt_md5_byte_t*>(buf->value));
  }
  else
  {
    memcpy(buf->value, ptr->key().value, 16);
  }
}

#ifdef DDSCXX_HAS_SHM
template<typename T>
uint32_t serdata_iox_size(const struct ddsi_serdata* d)
{
  assert(sizeof(T) == d->type->iox_size);
  return d->type->iox_size;
}

template<typename T>
ddsi_serdata * serdata_from_iox_buffer(
    const struct ddsi_sertype * typecmn, enum ddsi_serdata_kind kind,
    void * sub, void * iox_buffer)
{
  try {
    auto d = new ddscxx_serdata<T>(typecmn, kind);

    // serdata from the loaned sample (when using iceoryx)
    d->iox_chunk = iox_buffer;

    // Update the iox subscriber, when constructing the serdata in the case of sample received
    // from iceoryx
    if (sub != nullptr) {
      d->iox_subscriber = sub;
    }

    // key handling
    const auto& msg = *static_cast<const T*>(d->iox_chunk);
    d->key_md5_hashed() = to_key(msg, d->key());
    d->populate_hash();

    return d;
  }
  catch (std::exception&) {
    return nullptr;
  }
}
#endif

template<typename T,
         class S >
struct ddscxx_serdata_ops: public ddsi_serdata_ops {
  ddscxx_serdata_ops(): ddsi_serdata_ops {
  &serdata_eqkey<T>,
  &serdata_size<T>,
  &serdata_from_ser<T>,
  &serdata_from_ser_iov<T>,
  &serdata_from_keyhash<T>,
  &serdata_from_sample<T, S>,
  &serdata_to_ser<T>,
  &serdata_to_ser_ref<T>,
  &serdata_to_ser_unref<T>,
  &serdata_to_sample<T>,
  &serdata_to_untyped<T, S>,
  &serdata_untyped_to_sample<T>,
  &serdata_free<T>,
  &serdata_print<T>,
  &serdata_get_keyhash<T>
#ifdef DDSCXX_HAS_SHM
  , &serdata_iox_size<T>
  , &serdata_from_iox_buffer<T>
#endif
  } { ; }
};

template <typename T>
class ddscxx_serdata : public ddsi_serdata {
  size_t m_size{ 0 };
  std::unique_ptr<unsigned char[]> m_data{ nullptr };
  ddsi_keyhash_t m_key;
  bool m_key_md5_hashed = false;
  std::atomic<T *> m_t{ nullptr };

public:
  bool hash_populated = false;
  ddscxx_serdata(const ddsi_sertype* type, ddsi_serdata_kind kind);
  ~ddscxx_serdata() { delete m_t.load(std::memory_order_acquire); }

  void resize(size_t requested_size);
  size_t size() const { return m_size; }
  void* data() const { return m_data.get(); }
  ddsi_keyhash_t& key() { return m_key; }
  const ddsi_keyhash_t& key() const { return m_key; }
  bool& key_md5_hashed() { return m_key_md5_hashed; }
  const bool& key_md5_hashed() const { return m_key_md5_hashed; }
  void populate_hash();
  T* setT(const T* toset);
  T* getT();

private:
  void deserialize_and_update_sample(uint8_t * buffer, size_t sz, T *& t);
  void update_sample_from_iox_chunk(T *& t);
};

template <typename T>
ddscxx_serdata<T>::ddscxx_serdata(const ddsi_sertype* type, ddsi_serdata_kind kind)
  : ddsi_serdata{}
{
  memset(m_key.value, 0x0, 16);
  ddsi_serdata_init(this, type, kind);
}

template <typename T>
void ddscxx_serdata<T>::resize(size_t requested_size)
{
  if (!requested_size) {
    m_size = 0;
    m_data.reset();
    return;
  }

  /* FIXME: CDR padding in DDSI makes me do this to avoid reading beyond the bounds
  when copying data to network.  Should fix Cyclone to handle that more elegantly.  */
  size_t n_pad_bytes = (0 - requested_size) % 4;
  m_data.reset(new unsigned char[requested_size + n_pad_bytes]);
  m_size = requested_size + n_pad_bytes;

  // zero the very end. The caller isn't necessarily going to overwrite it.
  std::memset(calc_offset(m_data.get(), static_cast<ptrdiff_t>(requested_size)), '\0', n_pad_bytes);
}

template <typename T>
void ddscxx_serdata<T>::populate_hash()
{
  if (hash_populated)
    return;

  key_md5_hashed() = to_key(*getT(), key());
  if (!key_md5_hashed())
  {
    ddsi_keyhash_t buf;
    ddsrt_md5_state_t md5st;
    ddsrt_md5_init(&md5st);
    ddsrt_md5_append(&md5st, static_cast<const ddsrt_md5_byte_t*>(key().value), 16);
    ddsrt_md5_finish(&md5st, static_cast<ddsrt_md5_byte_t*>(buf.value));
    memcpy(&(hash), buf.value, 4);
  }
  else
  {
    memcpy(&(hash), key().value, 4);
  }

  hash ^= type->serdata_basehash;
  hash_populated = true;
}

template <typename T>
T* ddscxx_serdata<T>::setT(const T* toset)
{
  assert(toset);
  T* t = m_t.load(std::memory_order_acquire);
  if (t == nullptr) {
    t = new T(*toset);
    T* exp = nullptr;
    if (!m_t.compare_exchange_strong(exp, t, std::memory_order_seq_cst)) {
      delete t;
      t = exp;
    }
  } else {
    *t = *toset;
  }
  return t;
}

template <typename T>
T* ddscxx_serdata<T>::getT() {
  // check if m_t is already set
  T *t = m_t.load(std::memory_order_acquire);
  // if m_t is not set
  if (t == nullptr) {
    // if the data is available on iox_chunk, update and get the sample
    update_sample_from_iox_chunk(t);
    // if its not possible to get the sample from iox_chunk
    if(t == nullptr) {
      // deserialize and get the sample
      deserialize_and_update_sample(static_cast<uint8_t *>(data()), size(), t);
    }
  }
  return t;
}

template <typename T>
void ddscxx_serdata<T>::deserialize_and_update_sample(uint8_t * buffer, size_t sz, T *& t) {
  t = new T();
  // if deserialization failed
  if(!deserialize_sample_from_buffer(buffer, sz, *t, kind)) {
    delete t;
    t = nullptr;
  }

  T* exp = nullptr;
  if (!m_t.compare_exchange_strong(exp, t, std::memory_order_seq_cst)) {
    delete t;
    t = exp;
  }
}

template <typename T>
void ddscxx_serdata<T>::update_sample_from_iox_chunk(T *& t) {
#ifdef DDSCXX_HAS_SHM
  // if data is available on the iox_chunk (and doesn't have a serialized representation)
  if (iox_chunk != nullptr && data() == nullptr) {
      auto iox_header = iceoryx_header_from_chunk(iox_chunk);
      // if the iox chunk has the data in serialized form
      if (iox_header->shm_data_state == IOX_CHUNK_CONTAINS_SERIALIZED_DATA) {
        deserialize_and_update_sample(static_cast<uint8_t *>(iox_chunk), iox_header->data_size, t);
      } else if (iox_header->shm_data_state == IOX_CHUNK_CONTAINS_RAW_DATA) {
        // get the chunk directly without any copy
        t = static_cast<T*>(this->iox_chunk);
      } else {
        // Data is in un-initialized state, which shouldn't happen
        t = nullptr;
      }
    } else {
    // data is not available on iox_chunk
    t = nullptr;
  }
#else
  t = nullptr;
#endif  // DDSCXX_HAS_SHM
}

template <typename T, class S>
class ddscxx_sertype;

template <typename T, class S>
void sertype_free(ddsi_sertype* tpcmn)
{
  auto tp = static_cast<ddscxx_sertype<T,S>*>(tpcmn);
  ddsi_sertype_fini(tpcmn);
  delete tp;
}

template <typename T>
void sertype_zero_samples(const ddsi_sertype*, void*, size_t)
{
  return;
}

template <typename T>
void sertype_realloc_samples(
  void** ptrs, const ddsi_sertype*, void*, size_t, size_t)
{
  /* For C++ we make one big assumption about the caller of this function:
   * it can only be invoked by the ddsi_sertype_alloc_sample, and will therefore
   * never be used to reallocate an existing sample collection. This is caused by
   * the fact that the C++ API lets either the user specify the exact dimensions
   * of his preallocated collection (in which case there is no need to realloc them),
   * or if the user didn't preallocate any memory it peeks at the available
   * samples prior to allocating the sample collection that is returned (so that
   * again there is no need to reallocate it).
   * Because of this, we can safely assume that sertype_realloc_samples can only
   * be invoked by ddsi_sertype_alloc_sample, in which case oldCount is always 0,
   * count is always 1 and the old pointer is always null.
   */
  ptrs[0] = new T();
}

template <typename T>
void sertype_free_samples(
  const ddsi_sertype*, void** ptrs, size_t, dds_free_op_t op)
{
  /* For C++ we make one big assumption about the caller of this function:
   * it can only be invoked by the ddsi_sertype_free_sample, and will therefore
   * never be used to free an existing sample collection. This is caused by
   * the fact that the C++ API lets either the user specify the exact dimensions
   * of his preallocated collection (in which case there is no need to release
   * it in the cyclone code base), or if the user didn't preallocate any memory it
   * returns a collection of samples that will be owned by the user (in which case
   * cyclone doesn't need to release the collection either).
   * Because of this, we can safely assume that sertype_free_samples can only
   * be invoked by ddsi_sertype_free_sample, in which case count is always 1,
   * and the op flags can either be set to DDS_FREE_ALL_BIT, or to DDS_FREE_CONTENTS_BIT.
   */
  T* ptr = reinterpret_cast<T *>(ptrs[0]);
  if (op & DDS_FREE_ALL_BIT) {
    delete ptr;
  } else {
    assert(op & DDS_FREE_CONTENTS_BIT);
    *ptr = T();
  }
}

template <typename T>
bool sertype_equal(
  const ddsi_sertype* acmn, const ddsi_sertype* bcmn)
{
  /* A bit of a guess: types with the same name & type name are really the same if they have
   the same type support identifier as well */
  (void)acmn;
  (void)bcmn;
  return true;
}

template <typename T>
uint32_t sertype_hash(const ddsi_sertype* tpcmn)
{
  (void)tpcmn;
  return 0x0;
}

template <typename T, class S>
size_t sertype_get_serialized_size(const ddsi_sertype*, const void * sample)
{
  const auto& msg = *static_cast<const T*>(sample);

  // get the serialized size of the sample (with out serializing)
  size_t sz = 0;
  if (!get_serialized_size<T,S>(msg, false, sz)) {
    // the max value is treated as an error in the Cyclone core
    return SIZE_MAX;
  }

  return sz + CDR_HEADER_SIZE;  // Include the additional bytes for the CDR header
}

template <typename T, class S>
bool sertype_serialize_into(const ddsi_sertype*,
                            const void * sample,
                            void * dst_buffer,
                            size_t sz)
{
  // cast to the type
  const auto& msg = *static_cast<const T*>(sample);

  return serialize_into<T,S>(dst_buffer, sz, msg, false);
}

template<typename T,
         class S >
struct ddscxx_sertype_ops: public ddsi_sertype_ops {
  ddscxx_sertype_ops(): ddsi_sertype_ops {
  ddsi_sertype_v0,
  nullptr,
  sertype_free<T,S>,
  sertype_zero_samples<T>,
  sertype_realloc_samples<T>,
  sertype_free_samples<T>,
  sertype_equal<T>,
  sertype_hash<T>,
  #ifdef DDSCXX_HAS_TYPE_DISCOVERY
  TopicTraits<T>::getTypeId,
  TopicTraits<T>::getTypeMap,
  TopicTraits<T>::getTypeInfo,
  #else
  nullptr,
  nullptr,
  nullptr,
  #endif //DDSCXX_HAS_TYPE_DISCOVERY
  TopicTraits<T>::deriveSertype,
  sertype_get_serialized_size<T,S>,
  sertype_serialize_into<T,S>
    } { ; }
};

template <typename T,
          class S >
class ddscxx_sertype : public ddsi_sertype {
public:
  static const ddscxx_sertype_ops<T,S> sertype_ops;
  static const ddscxx_serdata_ops<T,S> serdata_ops;
  ddscxx_sertype();
};

template <typename T, class S>
const ddscxx_sertype_ops<T,S> ddscxx_sertype<T,S>::sertype_ops;

template <typename T, class S>
const ddscxx_serdata_ops<T,S> ddscxx_sertype<T,S>::serdata_ops;

template <typename T, class S>
ddscxx_sertype<T,S>::ddscxx_sertype()
  : ddsi_sertype{}
{
  uint32_t flags = (TopicTraits<T>::isKeyless() ? DDSI_SERTYPE_FLAG_TOPICKIND_NO_KEY : 0);
#ifdef DDSCXX_HAS_SHM
  flags |= (TopicTraits<T>::isSelfContained() ?
      DDSI_SERTYPE_FLAG_FIXED_SIZE : 0);
#endif

  ddsi_sertype_init_flags(
      static_cast<ddsi_sertype*>(this),
      TopicTraits<T>::getTypeName(),
      &sertype_ops,
      &serdata_ops,
      flags);

  allowed_data_representation = TopicTraits<T>::allowableEncodings();

#ifdef DDSCXX_HAS_SHM
  // update the size of the type, if its fixed
  // this needs to be done after sertype init! TODO need an API in Cyclone DDS to set this
  this->iox_size =
      static_cast<uint32_t>(TopicTraits<T>::getSampleSize());
#endif
}

#endif  // DDSCXXDATATOPIC_HPP_
