/*
 * Copyright(c) 2021 to 2022 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */
#ifndef EXTENDED_CDR_SERIALIZATION_V1_HPP_
#define EXTENDED_CDR_SERIALIZATION_V1_HPP_

#include "cdr_stream.hpp"

namespace org {
namespace eclipse {
namespace cyclonedds {
namespace core {
namespace cdr {

/**
 * @brief
 * Implementation of the extended cdr version1 stream.
 *
 * This type of cdr stream has a maximum alignment of 8 bytes.
 */
class OMG_DDS_API xcdr_v1_stream : public cdr_stream {
public:
  /**
   * @brief
   * Constructor.
   *
   * Basically a pass through for the cdr_stream base class.
   *
   * @param[in] end The endianness to set for the data stream, default to the local system endianness.
   * @param[in] ignore_faults Bitmask for ignoring faults, can be composed of bit fields from the serialization_status enumerator.
   */
  xcdr_v1_stream(endianness end = native_endianness(), uint64_t ignore_faults = 0x0) : cdr_stream(end, 8, ignore_faults) { ; }

  /**
   * @brief
   * Starts a new member.
   *
   * Determines whether a header is necessary for this entity through header_necessary, and if it is, handles the header.
   *
   * @param[in, out] prop Properties of the member to start.
   * @param[in] is_set Whether the entity represented by prop is present, if it is an optional entity.
   *
   * @return Whether the operation was completed succesfully.
   */
  bool start_member(entity_properties_t &prop, bool is_set = true);

  /**
   * @brief
   * Finishes a member.
   *
   * Determines whether a header is necessary for this entity through header_necessary, and if it is, completes the previous header.
   *
   * @param[in, out] prop Properties of the member to finish.
   * @param[in] is_set Whether the entity represented by prop is present, if it is an optional entity.
   *
   * @return Whether the operation was completed succesfully.
   */
  bool finish_member(entity_properties_t &prop, bool is_set = true);

  /**
   * @brief
   * Returns the next entity to be processed at this level.
   *
   * Depending on the data structure and the streaming mode, either a header is read from the stream, or a
   * properties entry is pulled from the tree.
   *
   * @param[in, out] prop The property tree to get the next entity from.
   *
   * @return The next entity to be processed, or a nullptr if the current tree level does not hold more entities that match this tree.
   */
  entity_properties_t* next_entity(entity_properties_t *prop);

  /**
   * @brief
   * Returns the first entity to be processed at this level.
   *
   * Depending on the data structure and the streaming mode, either a header is read from the stream, or a
   * properties entry is pulled from the tree.
   *
   * @param[in, out] prop The property tree to get the next entity from.
   *
   * @return The first entity to be processed, or a nullptr if the current tree level does not hold any entities that match this tree.
   */
  entity_properties_t *first_entity(entity_properties_t *prop);

  /**
   * @brief
   * Finishes the current struct.
   *
   * Adds the final parameter list entry if necessary when writing to the stream.
   *
   * @param[in, out] props The property tree to get the next entity from.
   *
   * @return Whether the struct is complete and correct.
   */
  bool finish_struct(entity_properties_t &props);

private:

  static const uint16_t pid_mask;                         /**< the mask for non-extended parameter list ids*/
  static const uint16_t pid_extended;                     /**<  indicating an extended entry*/
  static const uint16_t pid_list_end;                     /**<  guardian entry indicating end of parameter list*/
  static const uint16_t pid_ignore;                       /**<  ignore this entry*/
  static const uint16_t pid_flag_impl_extension;          /**<  bit flag indicating implementation specific extension*/
  static const uint16_t pid_flag_must_understand;         /**< bit flag indicating that this entry must be parsed successfully or the entire sample must be discarded*/
  static const uint32_t pl_extended_mask;                 /**<  mask for extended parameter list ids*/
  static const uint32_t pl_extended_flag_impl_extension;  /**< bit flag indicating implementation specific extension*/
  static const uint32_t pl_extended_flag_must_understand; /**< bit flag indicating that this entry must be parsed successfully or the entire sample must be discarded*/

  /**
   * @brief
   * Returns the next entity to be processed.
   *
   * For optional members and members of mutable structures, a parameter list header field is necessary preceding
   * the field contents itself.
   *
   * @param[in] props The properties of the entity.
   *
   * @return Whether a header is necessary for the entity.
   */
  bool header_necessary(const entity_properties_t &props);

  /**
   * @brief
   * Determines whether a parameter list is necessary.
   *
   * @param[in] props The entity whose members might be represented by a parameter list.
   *
   * @return Whether a parameter list is necessary for the entity.
   */
  bool list_necessary(const entity_properties_t &props);

  /**
   * @brief
   * Reads a header field from the stream.
   *
   * If header_necessary returns true for a field, then this function needs to be called first to read the
   * header from stream and to allow the streamer to determine what to do with the field.
   *
   * @param[out] out The header to read into.
   * @param[out] is_final Whether the final field has been read.
   *
   * @return Whether the header was read succesfully.
   */
  bool read_header(entity_properties_t &out, bool &is_final);

  /**
   * @brief
   * Writes a header field to the stream.
   *
   * If header_necessary returns true for a field, then this function needs to be called first to write the
   * header to the stream before the contents of the field are written.
   *
   * @param[in, out] props The properties of the entity.
   *
   * @return Whether the header was read succesfully.
   */
  bool write_header(entity_properties_t &props);

  /**
   * @brief
   * Finishes a header field in the stream.
   *
   * Goes back to the offset of the length field that was unfinished in 
   *
   * @param[in, out] props The properties of the entity.
   *
   * @return Whether the header was read succesfully.
   */
  bool finish_write_header(entity_properties_t &props);

  /**
   * @brief
   * Writes the terminating entry in a parameter list.
   *
   * @return Whether the header was read succesfully.
   */
  bool write_final_list_entry();

  /**
   * @brief
   * Moves the cursor as if writing the terminating entry in a parameter list.
   *
   * @return Whether the header was read succesfully.
   */
  bool move_final_list_entry();

  /**
   * @brief
   * Moves the stream offset by the amount that would have been written by write_header.
   *
   * This function needs to be called first to move the stream by the same amount the header would
   * have taken up, if it would have been written.
   *
   * @param[in] props The entity to move the cursor by.
   *
   * @return Whether the header was read succesfully.
   */
  bool move_header(const entity_properties_t &props);

  /**
   * @brief
   * Determines whether to use extended format header.
   *
   * An extended header is necessary for entities with a size larger than 65535 bytes or entities with
   * a member id larger than 16128.
   *
   * @param[in] props The entity to check.
   *
   * @return Whether an extended format header is necessary.
   */
  static bool extended_header(const entity_properties_t &props);
};

/**
 * @brief
 * Enumerated type stream manipulation functions.
 * Depending on the number coverage of the enum, it will be written
 * to the stream as an uint8_t, an uint16_t or a uint32_t.
 *
 * These are "endpoints" for write functions, since compound
 * (sequence/array/constructed type) functions will decay to these
 * calls.
 */

/**
 * @brief
 * Reads the value of the enum from the stream.
 *
 * @param[in, out] str The stream which is read from.
 * @param[out] toread The variable to read into.
 * @param[in] N The number of entities to read.
 *
 * @return Whether the operation was completed succesfully.
 */
template<typename T, std::enable_if_t<std::is_enum<T>::value && !std::is_arithmetic<T>::value, bool> = true >
bool read(xcdr_v1_stream& str, T& toread, size_t N = 1)
{
  switch (str.is_key() ? bb_32_bits : get_bit_bound<T>())
  {
    case bb_8_bits:
      return read_enum_impl<xcdr_v1_stream,T,uint8_t>(str, toread, N);
      break;
    case bb_16_bits:
      return read_enum_impl<xcdr_v1_stream,T,uint16_t>(str, toread, N);
      break;
    case bb_32_bits:
      return read_enum_impl<xcdr_v1_stream,T,uint32_t>(str, toread, N);
      break;
    default:
      assert(false);
  }
  return true;
}

/**
 * @brief
 * Writes the value of the enum to the stream.
 *
 * @param[in, out] str The stream which is written to.
 * @param[in] towrite The variable to write.
 * @param[in] N The number of entities to write.
 *
 * @return Whether the operation was completed succesfully.
 */
template<typename T, std::enable_if_t<std::is_enum<T>::value && !std::is_arithmetic<T>::value, bool> = true >
bool write(xcdr_v1_stream& str, const T& towrite, size_t N = 1)
{
  switch (str.is_key() ? bb_32_bits : get_bit_bound<T>())
  {
    case bb_8_bits:
      return write_enum_impl<xcdr_v1_stream,T,uint8_t>(str, towrite, N);
      break;
    case bb_16_bits:
      return write_enum_impl<xcdr_v1_stream,T,uint16_t>(str, towrite, N);
      break;
    case bb_32_bits:
      return write_enum_impl<xcdr_v1_stream,T,uint32_t>(str, towrite, N);
      break;
    default:
      assert(false);
  }
  return true;
}

/**
 * @brief
 * Moves the cursor of the stream by the size the enum would take up.
 *
 * @param[in, out] str The stream whose cursor is moved.
 * @param[in] N The number of entities to move.
 *
 * @return Whether the operation was completed succesfully.
 */
template<typename T, std::enable_if_t<std::is_enum<T>::value && !std::is_arithmetic<T>::value, bool> = true >
bool move(xcdr_v1_stream& str, const T&, size_t N = 1)
{
  switch (str.is_key() ? bb_32_bits : get_bit_bound<T>())
  {
    case bb_8_bits:
      return move(str, int8_t(0), N);
      break;
    case bb_16_bits:
      return move(str, int16_t(0), N);
      break;
    case bb_32_bits:
      return move(str, int32_t(0), N);
      break;
    default:
      assert(false);
  }
  return true;
}

/**
 * @brief
 * Moves the cursor of the stream by the size the enum would take up (maximum size version).
 *
 * @param[in, out] str The stream whose cursor is moved.
 * @param[in] max_sz The variable to move the cursor by, no contents of this variable are used, it is just used to determine the template.
 * @param[in] N The number of entities at most to move.
 *
 * @return Whether the operation was completed succesfully.
 */
template<typename T, std::enable_if_t<std::is_enum<T>::value && !std::is_arithmetic<T>::value, bool> = true >
bool max(xcdr_v1_stream& str, const T& max_sz, size_t N = 1)
{
  return move(str, max_sz, N);
}

}
}
}
}
}  /* namespace org / eclipse / cyclonedds / core / cdr */
#endif
