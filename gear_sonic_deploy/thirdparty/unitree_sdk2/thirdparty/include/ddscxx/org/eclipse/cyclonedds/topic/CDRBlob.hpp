/*
 * Copyright(c) 2006 to 2020 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef CYCLONEDDS_TOPIC_CDRBLOB_HPP
#define CYCLONEDDS_TOPIC_CDRBLOB_HPP

#include <array>

namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace topic
{
enum class BlobKind
{
  Empty,
  KeyOnly,
  Data,
};

class CDRBlob
{
private:
  std::array<char, 4> encoding_ = { };
  BlobKind kind_ = BlobKind::Empty;
  std::vector<uint8_t> payload_;

public:
  CDRBlob() = default;

  explicit CDRBlob(
      const std::array<char, 4>& encoding,
      BlobKind kind,
      const std::vector<uint8_t>& payload) :
          encoding_(encoding),
          kind_(kind),
          payload_(payload) {}

  const std::array<char, 4>& encoding() const { return this->encoding_; }
  std::array<char, 4>& encoding() { return this->encoding_; }
  void encoding(const std::array<char, 4>& _val_) { this->encoding_ = _val_; }
  void encoding(std::array<char, 4>&& _val_) { this->encoding_ = _val_; }
  BlobKind kind() const { return this->kind_; }
  BlobKind& kind() { return this->kind_; }
  void kind(BlobKind _val_) { this->kind_ = _val_; }
  const std::vector<uint8_t>& payload() const { return this->payload_; }
  std::vector<uint8_t>& payload() { return this->payload_; }
  void payload(const std::vector<uint8_t>& _val_) { this->payload_ = _val_; }
  void payload(std::vector<uint8_t>&& _val_) { this->payload_ = _val_; }
};

}
}
}
}

#endif /* CYCLONEDDS_TOPIC_CDRBLOB_HPP */
