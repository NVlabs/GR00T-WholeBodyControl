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
#ifndef EXTENDED_CDR_SERIALIZATION_V2_HPP_
#define EXTENDED_CDR_SERIALIZATION_V2_HPP_

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
class OMG_DDS_API xcdr_v2_stream : public cdr_stream {
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
  xcdr_v2_stream(endianness end = native_endianness(), uint64_t ignore_faults = 0x0) : cdr_stream(end, 4, ignore_faults) { ; }

  /**
   * @brief
   * Resets the state of the stream as before streaming began.
   *
   * Will reset the stack of delimiters in addition to what is reset by cdr_stream::reset.
   */
  void reset();

  /**
   * @brief
   * Starts a new member.
   *
   * Determines whether a header is necessary for this entity through em_header_necessary, and if it is, handles the header.
   *
   * @param[in, out] prop Properties of the member to start.
   * @param[in] is_set Whether the entity represented by prop is present, if it is an optional entity.
   */
  bool start_member(entity_properties_t &prop, bool is_set = true);

  /**
   * @brief
   * Finishes a member.
   *
   * Determines whether a header is necessary for this entity through em_header_necessary, and if it is, completes the previous header.
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
   * Start a new struct.
   *
   * This function is called by the generated streaming functions, and will start a parameter list, if that is relevant for it.
   *
   * @param[in, out] props The entity whose members might be represented by a parameter list.
   *
   * @return Whether the operation was completed succesfully.
   */
  bool start_struct(entity_properties_t &props);

  /**
   * @brief
   * Finish the current struct.
   *
   * This function is called by the generated streaming functions, and will finish the current parameter list, if that is relevant for it.
   *
   * @param[in, out] props The entity whose members might be represented by a parameter list.
   *
   * @return Whether the struct is complete and correct.
   */
  bool finish_struct(entity_properties_t &props);

  /**
   * @brief
   * Function declaration for starting an array or sequence of non-primitive types.
   *
   * This function inserts a d-header placeholder before the start of the consecutive objects.
   *
   *@param[in] is_array True when the consecutive entries is an array, false when it is a sequence.
   *@param[in] primitive Whether the consecutive entities are primitives (base types, not enums, strings, typedefs and arrays are resolved though)
   *
   * @return Whether the d-header was inserted correctly.
   */
  bool start_consecutive(bool is_array, bool primitive);

  /**
   * @brief
   * Function declaration for finishing an array or sequence of non-primitive types.
   *
   * This function finishes the d-header placeholder.
   *
   * @return Whether the d-header was finished correctly.
   */
  bool finish_consecutive();

private:
  typedef struct consecutives {
    consecutives(bool is_array = false, bool d_header_present = false) : is_array(is_array), d_header_present(d_header_present) {}
    bool is_array;
    bool d_header_present;
  } consecutives_t;

  static const uint32_t bytes_1;          /**< length field code indicating length is 1 byte*/
  static const uint32_t bytes_2;          /**< length field code indicating length is 2 bytes*/
  static const uint32_t bytes_4;          /**< length field code indicating length is 4 bytes*/
  static const uint32_t bytes_8;          /**< length field code indicating length is 8 bytes*/
  static const uint32_t nextint;          /**< length field code indicating length is the next integer field*/
  static const uint32_t nextint_times_1;  /**< same as nextint*/
  static const uint32_t nextint_times_4;  /**< length field code indicating length is the next integer field times 4*/
  static const uint32_t nextint_times_8;  /**< length field code indicating length is the next integer field times 8*/
  static const uint32_t lc_mask;          /**< mask for length field codes*/
  static const uint32_t id_mask;          /**< mask for member ids*/
  static const uint32_t must_understand;  /**< must understand member field flag*/

  DDSCXX_WARNING_MSVC_OFF(4251)
  custom_stack<consecutives_t, m_maximum_depth> m_consecutives; /**< stack of consecutive entries, uses to determine whether or not to finish a d_header*/
  custom_stack<size_t, m_maximum_depth> m_delimiters; /**< locations of sequence delimiters */
  DDSCXX_WARNING_MSVC_ON(4251)

  /**
   * @brief
   * Reads a D-header from the stream.
   *
   * Will put the entity size into the streams m_buffer_end stack
   *
   * @return Whether the read was succesful.
   */
  bool read_d_header();

  /**
   * @brief
   * Reads an EM-header from the stream.
   *
   * @param[out] props The entity to read the EM-header into.
   *
   * @return Whether the read was succesful.
   */
  bool read_em_header(entity_properties_t &props);

  /**
   * @brief
   * Writes a D-header placeholder to the stream.
   *
   * This will be filled with the value through the function finish_d_header when the struct is completed.
   *
   * @return Whether the write was succesful.
   */
  bool write_d_header();

  /**
   * @brief
   * Adds an optional flag to the stream.
   *
   * In the case of an optional field, but not a parameter list, the xcdrv2 spec states that this field should
   * preceded by a single boolean, indicating its presence or absence.
   *
   * @param[in] is_set Whether the entity represented by prop is present.
   *
   * @return Whether the read was succesful.
   */
  bool write_optional_tag(bool is_set);

  /**
   * @brief
   * Moves the stream cursor by the amount of an optional flag.
   *
   * In the case of an optional field, but not a parameter list, the xcdrv2 spec states that this field should
   * preceded by a single boolean, indicating its presence or absence.
   *
   * @return Whether the read was succesful.
   */
  bool move_optional_tag();

  /**
   * @brief
   * Writes an EM-header to the stream.
   *
   * @param[in, out] prop The entity to write the EM-header for.
   *
   * @return Whether the header was read succesfully.
   */
  bool write_em_header(entity_properties_t &prop);

  /**
   * @brief
   * Moves the stream's position by the amount that it would after writing the D-header.
   *
   * Moves the cursor by a 4 byte int.
   *
   * @return Whether the operation was completed succesfully.
   */
  bool move_d_header() {return move(*this, uint32_t(0));}

  /**
   * @brief
   * Moves the stream's position by the amount that it would after writing the EM-header.
   *
   * @return Whether the header was read succesfully.
   */
  bool move_em_header();

  /**
   * @brief
   * Finishes the write operation of the D-header.
   *
   * @return Whether the header was read succesfully.
   */
  bool finish_d_header();

  /**
   * @brief
   * Finishes the write operation of the EM-header.
   *
   * @return Whether the header was read succesfully.
   */
  bool finish_em_header();

  /**
   * @brief
   * Checks whether a D-header is necessary for the indicated entity.
   *
   * @param[in] props The entity whose properties to check.
   *
   * @return Whether the entity props needs a D-header
   */
  bool d_header_necessary(const entity_properties_t &props);

  /**
   * @brief
   * Checks whether a EM-header is necessary for the indicated entity.
   *
   * @param[in] props The entity whose properties to check.
   *
   * @return Whether the entity props needs a EM-header
   */
  bool em_header_necessary(const entity_properties_t &props);

  /**
   * @brief
   * Determines whether a parameter list is necessary.
   *
   * This function is called by the next_entity function, to determine whether or not
   * to read em headers from the stream.
   *
   * @param[in] props The entity whose members might be represented by a parameter list.
   *
   * @return Whether a list is necessary for this entity.
   */
  bool list_necessary(const entity_properties_t &props);
};

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
bool read(xcdr_v2_stream& str, T& toread, size_t N = 1) {
  switch (str.is_key() ? bb_32_bits : get_bit_bound<T>())
  {
    case bb_8_bits:
      return read_enum_impl<xcdr_v2_stream,T,uint8_t>(str, toread, N);
      break;
    case bb_16_bits:
      return read_enum_impl<xcdr_v2_stream,T,uint16_t>(str, toread, N);
      break;
    case bb_32_bits:
      return read_enum_impl<xcdr_v2_stream,T,uint32_t>(str, toread, N);
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
 * @param [in, out] str The stream which is written to.
 * @param [in] towrite The variable to write.
 * @param[in] N The number of entities to write.
 *
 * @return Whether the operation was completed succesfully.
 */
template<typename T, std::enable_if_t<std::is_enum<T>::value && !std::is_arithmetic<T>::value, bool> = true >
bool write(xcdr_v2_stream& str, const T& towrite, size_t N = 1) {
  switch (str.is_key() ? bb_32_bits : get_bit_bound<T>())
  {
    case bb_8_bits:
      return write_enum_impl<xcdr_v2_stream,T,uint8_t>(str, towrite, N);
      break;
    case bb_16_bits:
      return write_enum_impl<xcdr_v2_stream,T,uint16_t>(str, towrite, N);
      break;
    case bb_32_bits:
      return write_enum_impl<xcdr_v2_stream,T,uint32_t>(str, towrite, N);
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
bool move(xcdr_v2_stream& str, const T&, size_t N = 1) {
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
bool max(xcdr_v2_stream& str, const T& max_sz, size_t N = 1) {
  return move(str, max_sz, N);
}

}
}
}
}
}  /* namespace org / eclipse / cyclonedds / core / cdr */
#endif