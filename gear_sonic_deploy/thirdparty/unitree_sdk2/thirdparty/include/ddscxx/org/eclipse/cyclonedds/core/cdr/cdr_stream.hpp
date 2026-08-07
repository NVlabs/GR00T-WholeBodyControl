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
#ifndef CDR_STREAM_HPP_
#define CDR_STREAM_HPP_

#include "dds/ddsrt/endian.h"
#include <org/eclipse/cyclonedds/core/type_helpers.hpp>
#include <org/eclipse/cyclonedds/core/cdr/entity_properties.hpp>
#include <stdint.h>
#include <string>
#include <stdexcept>
#include <stack>
#include <cassert>
#include <dds/core/macros.hpp>

namespace org {
namespace eclipse {
namespace cyclonedds {
namespace core {
namespace cdr {

/**
 * @brief
 * Custom stack implementation.
 */
template<typename T, size_t N>
class custom_stack {
  static_assert(N > 0, "Stack capacity must be larger than 0");
  T data[N];
  size_t sz = 0;
  public:
  custom_stack() = default;
  custom_stack(const T &in) {data[0] = in; sz = 1;}
  T &top() {return data[sz-1];}
  const T& top() const {return data[sz-1];}
  void pop() {sz--;}
  void push(const T &in) {data[sz++] = in;}
  void reset() {sz = 0;}
  size_t size() const {return sz;}
};

/**
 * @brief
 * Enum conversion and validation function template forward declaration.
 *
 * This function is generated for each enumerated class encountered in the parsed .idl files.
 * Converts an integer value to the corresponding enum class value, or its default value
 * if there is no enum equivalent to the int.
 *
 * @param[in] in The integer to convert to the enumerated class.
 *
 * @return The enumerator representation of in.
 */
template<typename E>
E enum_conversion(uint32_t in);

/**
 * @brief
 * Byte swapping function, is only enabled for arithmetic (base) types.
 *
 * Determines the number of bytes to swap by the size of the template parameter.
 *
 * @param[in, out] toswap The entity whose bytes will be swapped.
 */
template<typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value> >
void byte_swap(T& toswap) {
    union { T a; uint16_t u2; uint32_t u4; uint64_t u8; } u;
    u.a = toswap;
    DDSCXX_WARNING_MSVC_OFF(6326)
    switch (sizeof(T)) {
    case 1:
        break;
    case 2:
        u.u2 = static_cast<uint16_t>((u.u2 & 0xFF00) >> 8)
             | static_cast<uint16_t>((u.u2 & 0x00FF) << 8);
        break;
    case 4:
        u.u4 = static_cast<uint32_t>((u.u4 & 0xFFFF0000) >> 16)
             | static_cast<uint32_t>((u.u4 & 0x0000FFFF) << 16);
        u.u4 = static_cast<uint32_t>((u.u4 & 0xFF00FF00) >> 8)
             | static_cast<uint32_t>((u.u4 & 0x00FF00FF) << 8);
        break;
    case 8:
        u.u8 = static_cast<uint64_t>((u.u8 & 0xFFFFFFFF00000000) >> 32)
             | static_cast<uint64_t>((u.u8 & 0x00000000FFFFFFFF) << 32);
        u.u8 = static_cast<uint64_t>((u.u8 & 0xFFFF0000FFFF0000) >> 16)
             | static_cast<uint64_t>((u.u8 & 0x0000FFFF0000FFFF) << 16);
        u.u8 = static_cast<uint64_t>((u.u8 & 0xFF00FF00FF00FF00) >> 8)
             | static_cast<uint64_t>((u.u8 & 0x00FF00FF00FF00FF) << 8);
        break;
    default:
        throw std::invalid_argument(std::string("attempted byteswap on variable of invalid size: ") + std::to_string(sizeof(T)));
    }
    DDSCXX_WARNING_MSVC_ON(6326)
    toswap = u.a;
}

/**
 * @brief
 * Endianness types.
 *
 * @enum endianness C++ implementation of cyclonedds's DDSRT_ENDIAN endianness defines
 *
 * @var endianness::little_endian Little endianness.
 * @var endianness::big_endian Big endianness.
 */
enum class endianness {
    little_endian = DDSRT_LITTLE_ENDIAN,
    big_endian = DDSRT_BIG_ENDIAN
};

/**
 * @brief
 * Returns the endianness of the local system.
 *
 * Takes the value from the DDSRT_ENDIAN definition and converts it to the c++ enum class value.
 *
 * @retval little_endian If the system is little endian.
 * @retval big_endian If the system is big endian.
 */
constexpr endianness native_endianness() { return endianness(DDSRT_ENDIAN); }

/**
 * @brief
 * Serialization status bitmasks.
 *
 * @enum serialization_status Describes the serialization status of a cdr stream.
 *
 * These are stored as an bitfields in an int in cdr streams, since more than one serialization fault can be encountered.
 *
 * @var serialization_status::move_bound_exceeded The serialization has encountered a field which has exceeded the bounds set for it.
 * @var serialization_status::write_bound_exceeded The serialization has encountered a field which has exceeded the bounds set for it.
 * @var serialization_status::read_bound_exceeded The serialization has encountered a field which has exceeded the bounds set for it.
 * @var serialization_status::illegal_field_value The serialization has encountered a field with a value which should never occur in a valid CDR stream.
 * @var serialization_status::invalid_pl_entry The serialization has encountered a parameter list id which is illegal (not extended id in reserved for OMG space).
 * @var serialization_status::unsupported_xtypes A streamer has attempted to stream a struct requiring xtypes but not supporting it itself.
 * @var serialization_status::must_understand_fail A struct being read contains a field that must be understood but does not recognize or have.
 */
enum serialization_status : uint64_t {
  move_bound_exceeded   = 0x1 << 0,
  write_bound_exceeded  = 0x1 << 1,
  read_bound_exceeded   = 0x1 << 2,
  invalid_pl_entry      = 0x1 << 3,
  illegal_field_value   = 0x1 << 4,
  unsupported_xtypes    = 0x1 << 5,
  must_understand_fail  = 0x1 << 6
};

/**
 * @brief
 * Base cdr_stream class.
 *
 * This class implements the base functions which all "real" cdr stream implementations will use.
 */
class OMG_DDS_API cdr_stream {
public:
    /**
     * @brief
     * Constructor.
     *
     * Sets the stream endianness to end, and maximum alignment to max_align.
     *
     * @param[in] end The endianness to set for the data stream, default to the local system endianness.
     * @param[in] max_align The maximum size that the stream will align CDR primitives to.
     * @param[in] ignore_faults Bitmask for ignoring faults, can be composed of bit fields from the serialization_status enumerator.
     */
    cdr_stream(endianness end, size_t max_align, uint64_t ignore_faults = 0x0) : m_stream_endianness(end), m_max_alignment(max_align), m_fault_mask(~ignore_faults), m_swap(native_endianness() != m_stream_endianness) { ; }

    /**
     * @brief
     * Returns the current stream alignment.
     *
     * @return The current stream alignment.
     */
    size_t alignment() const { return m_current_alignment; }

    /**
     * @brief
     * Sets the new stream alignment.
     *
     * Also returns the value the alignment has been set to.
     *
     * @param[in] newalignment The new alignment to set.
     *
     * @return The value the alignment has been set to.
     */
    size_t alignment(size_t newalignment) { return m_current_alignment = newalignment; }

    /**
     * @brief
     * Checks whether a delimited cdr stream is not being read out of bounds.
     *
     * This function will return true if N bytes can be read from the stream.
     *
     * @param[in] N The number of bytes requested.
     * @param[in] peek Whether this is true access, or just a "peek".
     *
     * @return Whether enough bytes are available for another header.
     */
    bool bytes_available(size_t N = 1, bool peek = false);

    /**
     * @brief
     * Returns the current cursor offset.
     *
     * @retval SIZE_MAX In this case, a maximum size calculation was being done, and the maximum size was determined to be unbounded.
     * @return The current cursor offset.
     */
    inline size_t position() const { return m_position; }

    /**
     * @brief
     * Sets the new cursor offset.
     *
     * Also returs the value the offset has been set to.
     *
     * @param[in] newposition The new offset to set.
     *
     * @return The value the offset has been set to.
     */
    size_t position(size_t newposition) { return m_position = newposition; }

    /**
     * @brief
     * Cursor move function.
     *
     * Moves the current position offset by incr_by if it is not at SIZE_MAX.
     * Returns the position value after this operation.
     *
     * @param[in] incr_by The amount to move the cursor position by.
     *
     * @return The cursor position after this operation.
     */
    size_t incr_position(size_t incr_by) { if (m_position != SIZE_MAX) m_position += incr_by; return m_position; }

    /**
     * @brief
     * Resets the state of the stream as before streaming began.
     *
     * Will set the current offset, alignment to 0, clear the stack and fault status.
     * Will retain the buffer pointer and size.
     */
    virtual void reset();

    /**
     * @brief
     * Buffer set function.
     *
     * Sets the buffer pointer to toset.
     * As a side effect, the current position and alignment are reset, since these are not associated with the new buffer.
     *
     * @param[in] toset The new pointer of the buffer to set.
     * @param[in] buffer_size The size of the buffer being set.
     */
    void set_buffer(void* toset, size_t buffer_size = SIZE_MAX);

    /**
     * @brief
     * Gets the current cursor pointer.
     *
     * If the current position is SIZE_MAX or the buffer pointer is not set, it returns nullptr.
     *
     * @retval nullptr If the current buffer is not set, or if the cursor offset is not valid.
     * @return The current cursor pointer.
     */
    inline char* get_cursor() const { return m_buffer + m_position; }

    /**
     * @brief
     * Const stream endianness getter (const).
     *
     * This is used to determine whether the data read or written from the stream needs to have their bytes swapped.
     *
     * @return The stream endianness.
     */
    const endianness& stream_endianness() const { return m_stream_endianness; }

    /**
     * @brief
     * Determines whether the local and stream endianness are the same.
     *
     * This is used to determine whether the data read or written from the stream needs to have their bytes swapped.
     *
     * @retval false If the stream endianness DOES match the local endianness.
     * @retval true If the stream endianness DOES NOT match the local endianness.
     */
    bool swap_endianness() const { return m_swap; }

    /**
     * @brief
     * Aligns the current stream to a new alignment.
     *
     * Aligns the current stream to newalignment, moves the cursor be at newalignment.
     * Aligns to maximum m_max_alignment (which is stream-type specific).
     * Zeroes the bytes the cursor is moved if add_zeroes is true.
     * Nothing happens if the stream is already aligned to newalignment.
     *
     * @param[in] newalignment The new alignment to align the stream to.
     * @param[in] add_zeroes Whether the bytes that the cursor moves need to be zeroed.
     *
     * @return Whether the cursor could be moved by the required amount.
     */
    bool align(size_t newalignment, bool add_zeroes);

    /**
     * @brief
     * Returns the current status of serialization.
     *
     * Can be a composition of multiple bit fields from serialization_status.
     *
     * @return The current status of serialization.
     */
    uint64_t status() const { return m_status; }

    /**
     * @brief
     * Serialization status update function.
     *
     * Adds to the current status of serialization and returns whether abort status has been reached.
     *
     * @param[in] toadd The serialization status error to add.
     *
     * @retval false If the serialization status of the stream HAS NOT YET reached one of the serialization errors which it is not set to ignore.
     * @retval true If the serialization status of the stream HAS reached one of the serialization errors which it is not set to ignore.
     */
    bool status(serialization_status toadd) { m_status |= static_cast<uint64_t>(toadd); return abort_status(); }

    /**
     * @brief
     * Returns true when the stream has encountered an error which it is not set to ignore.
     *
     * All streaming functions should become NOOPs after this status is encountered.
     *
     * @retval false If the serialization status of the stream HAS NOT YET reached one of the serialization errors which it is not set to ignore.
     * @retval true If the serialization status of the stream HAS reached one of the serialization errors which it is not set to ignore.
     */
    inline bool abort_status() const { return m_status & m_fault_mask; }

    /**
     * @brief
     * Type of streaming operation to be done.
     *
     * @var stream_mode::unset The stream mode is not set.
     * @var stream_mode::read Reads from the stream into an instance.
     * @var stream_mode::write Writes from the instance to the stream.
     * @var stream_mode::move Moves the cursor by the same amount as would has been done through stream_mode::write, without copying any data to the stream.
     * @var stream_mode::max Same as stream_mode::move, but by the maximum amount possible for an entity of that type.
     */
    enum class stream_mode {
      unset,
      read,
      write,
      move,
      max
    };

    /**
     * @brief
     * Returns whether the streaming is done only over the key values.
     *
     * @return Whether the streaming is done only over the key values.
     */
    inline bool is_key() const {return m_key;}

    /**
     * @brief
     * Function which sets the current streaming mode.
     *
     * This will impact which entities will be retrieved from the entity properties list.
     * This will also reset the current cursor position.
     *
     * @param[in] mode The streaming mode to set for the stream.
     * @param[in] key The key mode to set for the stream.
     */
    void set_mode(stream_mode mode, bool key) {m_mode = mode; m_key = key; reset();}

    /**
     * @brief
     * Function declaration for starting a new member.
     *
     * This function is called by next_entity for each entity which is iterated over.
     * Depending on the implementation and mode headers may be read from/written to the stream.
     * This function can be overridden in cdr streaming implementations.
     *
     * @param[in] prop Properties of the entity to start.
     * @param[in] is_set Whether the entity represented by prop is present, if it is an optional entity.
     *
     * @return Whether the operation was completed succesfully.
     */
    virtual bool start_member(entity_properties_t &prop, bool is_set = true) { prop.is_present = is_set; return true;}

    /**
     * @brief
     * Function declaration for finishing an existing member.
     *
     * This function is called by next_entity for each entity which is iterated over.
     * Depending on the implementation and mode header length fields may be completed.
     * This function can be overridden in cdr streaming implementations.
     *
     * @param[in] is_set Whether the entity represented by prop is present, if it is an optional entity.
     *
     * @return Whether the operation was completed succesfully.
     */
    virtual bool finish_member(entity_properties_t &, bool is_set = true) { (void) is_set; return true;}

    /**
     * @brief
     * Function declaration for retrieving the next entity to be operated on by the streamer.
     *
     * This function is called by the instance implementation switchbox and will return the next entity to operate on by calling next_prop.
     * This will also call the implementation specific push/pop entity functions to write/finish headers where necessary.
     *
     * @param[in, out] prop The property tree to get the next entity from.
     *
     * @return The next entity to be processed, or the final entity if the current tree level does not hold more entities.
     */
    virtual entity_properties_t* next_entity(entity_properties_t *prop);

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
    virtual entity_properties_t* first_entity(entity_properties_t *prop);

    /**
     * @brief
     * Function declaration for starting a parameter list.
     *
     * This function is called by the generated functions for the entity, and will trigger the necessary actions on starting a new struct.
     * I.E. starting a new parameter list, writing headers.
     *
     * @param[in,out] props The entity whose members might be represented by a parameter list.
     *
     * @return Whether the operation was completed succesfully.
     */
    virtual bool start_struct(entity_properties_t &props);

    /**
     * @brief
     * Function declaration for finishing a parameter list.
     *
     * This function is called by the generated functions for the entity, and will trigger the necessary actions on finishing the current struct.
     * I.E. finishing headers, writing length fields.
     *
     * @param[in,out] props The entity whose members might be represented by a parameter list.
     *
     * @return Whether the struct is complete and correct.
     */
    virtual bool finish_struct(entity_properties_t &props);

    /**
     * @brief
     * Function declaration for starting an array or sequence of non-primitive types.
     *
     * This function is used to keep track of whether delimiters need to be and have been written to the stream.
     * This function is an effective no-op for all streamers except xcdr_v2.
     *
     * @param[in] is_array True when the consecutive entries is an array, false when it is a sequence.
     * @param[in] primitive Whether the consecutive entities are primitives (base types, not enums, strings, typedefs and arrays are resolved though)
     *
     * @return Always true.
     */
    virtual bool start_consecutive(bool is_array, bool primitive) { (void) is_array; (void) primitive; return true;}

    /**
     * @brief
     * Function declaration for finishing an array or sequence of non-primitive types.
     *
     * This function is an effective no-op for all streamers except xcdr_v2.
     *
     * @return Always true.
     */
    virtual bool finish_consecutive() {return true;}

protected:

    /**
     * @brief Implementation for starting the recording the size and starting offset of a member.
     */
    inline void push_member_start() { m_e_sz.push(0); m_e_off.push(static_cast<uint32_t>(position())); }

    /**
     * @brief Implementation for finishing the recording the size and starting offset of a member.
     */
    inline void pop_member_start() { m_e_sz.pop(); m_e_off.pop(); }

    /**
     * @brief
     * Checks the struct for completeness.
     *
     * Checks whether all fields which must be understood are present.
     *
     * @param[in,out] props The struct whose start is recorded.
     */
    void check_struct_completeness(entity_properties_t &props);

    /**
     * @brief
     * Returns the previous entity at the current level (if any).
     *
     * @param[in] prop Entity to the current entity.
     *
     * @return Pointer to the previous entity, or nullptr if there is any.
     */
    entity_properties_t* previous_entity(entity_properties_t *prop);

    static const size_t m_maximum_depth = 32;     /**< the maximum depth of structures in the streamer*/

    endianness m_stream_endianness;               /**< the endianness of the stream*/
    size_t m_position = 0,                        /**< the current offset position in the stream*/
        m_max_alignment,                          /**< the maximum bytes that can be aligned to*/
        m_current_alignment = 1,                  /**< the current alignment*/
        m_buffer_size = 0;                        /**< the size of the current buffer*/
    char* m_buffer = nullptr;                     /**< the current buffer in use*/
    uint64_t m_status = 0,                        /**< the current status of streaming*/
             m_fault_mask;                        /**< the mask for statuses that will cause streaming
                                                       to be aborted*/
    stream_mode m_mode = stream_mode::unset;      /**< the current streaming mode*/
    bool m_key = false;                           /**< the current key mode*/
    bool m_swap = false;                          /**< whether to swap endianness*/

    DDSCXX_WARNING_MSVC_OFF(4251)
    custom_stack<size_t, m_maximum_depth> m_buffer_end; /**< the end of reading at the current level*/
    custom_stack<uint32_t, m_maximum_depth> m_e_off, /**< the offset of the entity at the current level*/
                                            m_e_sz; /**< the size of the entity at the current level*/
    DDSCXX_WARNING_MSVC_ON(4251)
};

/**
 * @brief
 * Primitive type stream manipulation functions.
 *
 * These are "endpoints" for write functions, since composit
 * (sequence/array/constructed type) functions will decay to these
 * calls.
 */

/**
 * @brief
 * Primitive type read function.
 *
 * Aligns the stream to the alignment of type T.
 * Reads the value from the current position of the stream str into
 * toread, will swap bytes if necessary.
 * Moves the cursor of the stream by the size of T.
 * This function is only enabled for arithmetic types and enums.
 *
 * @param[in, out] str The stream which is read from.
 * @param[out] toread The variable to read into.
 * @param[in] N The number of entities to read.
 *
 * @return Whether the operation was completed succesfully.
 */
template<typename S, typename T, std::enable_if_t<std::is_arithmetic<T>::value
                                               && !std::is_enum<T>::value
                                               && std::is_base_of<cdr_stream, S>::value, bool> = true >
bool read(S &str, T& toread, size_t N = 1)
{
  if (str.position() == SIZE_MAX
   || !str.align(sizeof(T), false)
   || !str.bytes_available(sizeof(T)*N))
    return false;

  auto from = reinterpret_cast<const T*>(str.get_cursor());
  T *to = &toread;

  assert(from);

  if (N == 1) {
    toread = *from;
    if (str.swap_endianness())
        byte_swap(toread);
  } else {
    memcpy(to,from,sizeof(T)*N);
    if (str.swap_endianness()) {
      for (size_t i = 0; i < N; i++, to++)
        byte_swap(*to);
    }
  }

  str.incr_position(sizeof(T)*N);

  return true;
}

/**
 * @brief
 * Enum type read function implementation.
 *
 * Uses the template parameter I to determine the stream-end read type,
 * this type is determined by the stream implementation.
 * Reads the enums as type I from the stream.
 * Each read entity is verified by the enum's conversion version.
 * This function is only enabled for enum types.
 *
 * @param[in, out] str The stream which is read from.
 * @param[out] toread The variable to read.
 * @param[in] N The number of entities to read.
 *
 * @return Whether the operation was completed succesfully.
 */
template<typename S, typename T, typename I, std::enable_if_t<std::is_integral<I>::value
                                               && std::is_enum<T>::value
                                               && std::is_base_of<cdr_stream, S>::value, bool> = true>
bool read_enum_impl(S& str, T& toread, size_t N)
{
  T *ptr = &toread;
  I holder = 0;
  for (size_t i = 0; i < N; i++, ptr++)
  {
    if (!read(str, holder))
      return false;
    *ptr = enum_conversion<T>(holder);
  }
  return true;
}

/**
 * @brief
 * Primitive type write function.
 *
 * Aligns str to the type to be written.
 * Writes towrite to str.
 * Swaps bytes written to str if the endiannesses do not match up.
 * Moves the cursor of str by the size of towrite.
 * This function is only enabled for arithmetic types.
 *
 * @param[in, out] str The stream which is written to.
 * @param[in] towrite The variable to write.
 * @param[in] N The number of entities to write.
 *
 * @return Whether the operation was completed succesfully.
 */
template<typename S, typename T, std::enable_if_t<std::is_arithmetic<T>::value
                                               && !std::is_enum<T>::value
                                               && std::is_base_of<cdr_stream, S>::value, bool> = true >
bool write(S& str, const T& towrite, size_t N = 1)
{
  if (str.position() == SIZE_MAX
   || !str.align(sizeof(T), true)
   || !str.bytes_available(sizeof(T)*N))
    return false;

  auto to = reinterpret_cast<T*>(str.get_cursor());

  assert(to);

  if (N == 1) {
    *to = towrite;
    if (str.swap_endianness())
      byte_swap(*to);
  } else {
    const T *from = &towrite;
    memcpy(to,from,sizeof(T)*N);
    if (str.swap_endianness()) {
      for (size_t i = 0; i < N; i++, to++)
        byte_swap(*to);
    }
  }

  str.incr_position(sizeof(T)*N);

  return true;
}

/**
 * @brief
 * Enum type write function implementation.
 *
 * Uses the template parameter I to determine the stream-end write type,
 * this type is determined by the stream implementation.
 * Writes the enums as type I to the stream.
 * If the enums have the same size as the integer stream type, they are written
 * as a block, otherwise they are copied one by one.
 * This function is only enabled for enum types.
 *
 * @param[in, out] str The stream which is written to.
 * @param[in] towrite The variable to write.
 * @param[in] N The number of entities to write.
 *
 * @return Whether the operation was completed succesfully.
 */
template<typename S, typename T, typename I, std::enable_if_t<std::is_integral<I>::value
                                               && std::is_enum<T>::value
                                               && std::is_base_of<cdr_stream, S>::value, bool> = true>
bool write_enum_impl(S& str, const T& towrite, size_t N)
{
  const T *ptr = &towrite;
  if (sizeof(T) == sizeof(I)) {
      if (!write(str, *reinterpret_cast<const I*>(ptr), N))
        return false;
  } else {
    for (size_t i = 0; i < N; i++, ptr++)
      if (!write(str, *reinterpret_cast<const I*>(ptr)))
        return false;
  }
  return true;
}

/**
 * @brief
 * Primitive type cursor move function.
 *
 * Used in determining the size of a type when written to the stream.
 * Aligns str to the size of toincr.
 * Moves the cursor of str by the size of toincr.
 * This function is only enabled for arithmetic types.
 *
 * @param[in, out] str The stream whose cursor is moved.
 * @param[in] N The number of entities to move.
 *
 * @return Whether the operation was completed succesfully.
 */
template<typename S, typename T, std::enable_if_t<std::is_arithmetic<T>::value
                                               && !std::is_enum<T>::value
                                               && std::is_base_of<cdr_stream, S>::value, bool> = true >
bool move(S& str, const T&, size_t N = 1)
{
  if (str.position() == SIZE_MAX)
    return true;

  if (!str.align(sizeof(T), false))
    return false;

  str.incr_position(sizeof(T)*N);

  return true;
}

/**
 * @brief
 * Primitive type max stream move function.
 *
 * Used in determining the maximum stream size of a constructed type.
 * Moves the cursor to the maximum position it could occupy after
 * writing max_sz to the stream.
 * Is in essence the same as the primitive type cursor move function,
 * but additionally checks for whether the cursor it at the "end",
 * which may happen if unbounded members (strings/sequences/...)
 * are part of the constructed type.
 * This function is only enabled for arithmetic types.
 *
 * @param[in, out] str The stream whose cursor is moved.
 * @param[in] max_sz The variable to move the cursor by, no contents of this variable are used, it is just used to determine the template.
 * @param[in] N The number of entities at most to move.
 *
 * @return Whether the operation was completed succesfully.
 */
template<typename S, typename T, std::enable_if_t<std::is_arithmetic<T>::value
                                               && !std::is_enum<T>::value
                                               && std::is_base_of<cdr_stream, S>::value, bool> = true >
bool max(S& str, const T& max_sz, size_t N = 1)
{
  return move(str, max_sz, N);
}

 /**
 * @brief
 * String type stream manipulation functions
 *
 * These are "endpoints" for write functions, since compound
 * (sequence/array/constructed type) functions will decay to these
 * calls.
 */

/**
 * @brief
 * Bounded string read function.
 *
 * Reads the length from str, but then initializes toread with at most N characters from it.
 * It does move the cursor by length read, since that is the number of characters in the stream.
 * If N is 0, then the string is taken to be unbounded.
 *
 * @param[in, out] str The stream to read from.
 * @param[out] toread The string to read to.
 * @param[in] N The maximum number of characters to read from the stream.
 *
 * @return Whether the operation was completed succesfully.
 */
template<typename S, typename T, std::enable_if_t<std::is_base_of<cdr_stream, S>::value, bool> = true >
bool read_string(S& str, T& toread, size_t N)
{
  if (str.position() == SIZE_MAX)
    return false;

  uint32_t string_length = 0;

  if (!read(str, string_length)
   || !str.bytes_available(string_length))
    return false;

  if (string_length == 0
   && str.status(serialization_status::illegal_field_value))
    return false;

  if (N && string_length > N + 1)
    return false;

  auto cursor = str.get_cursor();
  toread.assign(cursor, cursor + std::min<size_t>(string_length - 1, N ? N : SIZE_MAX));  //remove 1 for terminating NULL

  str.incr_position(string_length);

  //aligned to chars
  str.alignment(1);

  return true;
}

/**
 * @brief
 * Bounded string write function.
 *
 * Attempts to write the length of towrite to str, where the bound is checked.
 * Then writes the contents of towrite to str.
 * If N is 0, then the string is taken to be unbounded.
 *
 * @param[in, out] str The stream to write to.
 * @param[in] towrite The string to write.
 * @param[in] N The maximum number of characters to write to the stream.
 *
 * @return Whether the operation was completed succesfully.
 */
template<typename S, typename T, std::enable_if_t<std::is_base_of<cdr_stream, S>::value, bool> = true >
bool write_string(S& str, const T& towrite, size_t N)
{
  if (str.position() == SIZE_MAX)
    return false;

  size_t string_length = towrite.length() + 1;  //add 1 extra for terminating NULL

  if (N
   && string_length > N + 1
   && str.status(serialization_status::write_bound_exceeded))
      return false;

  if (!write(str, uint32_t(string_length))
   || !str.bytes_available(string_length))
    return false;

  memcpy(str.get_cursor(), towrite.c_str(), string_length);

  str.incr_position(string_length);

  //aligned to chars
  str.alignment(1);

  return true;
}

/**
 * @brief
 * Bounded string cursor move function.
 *
 * Attempts to move the cursor for the length field, where the bound is checked.
 * Then moves the cursor for the length of the string.
 * If N is 0, then the string is taken to be unbounded.
 *
 * @param[in, out] str The stream whose cursor is moved.
 * @param[in] toincr The string used to move the cursor.
 * @param[in] N The maximum number of characters in the string which the stream is moved by.
 *
 * @return Whether the operation was completed succesfully.
 */
template<typename S, typename T, std::enable_if_t<std::is_base_of<cdr_stream, S>::value, bool> = true >
bool move_string(S& str, const T& toincr, size_t N)
{
  if (str.position() == SIZE_MAX)
    return true;

  size_t string_length = toincr.length() + 1;  //add 1 extra for terminating NULL

  if (N
   && string_length > N + 1
   && str.status(serialization_status::move_bound_exceeded))
      return false;

  if (!move(str, uint32_t()))
    return false;

  str.incr_position(string_length);

  //aligned to chars
  str.alignment(1);

  return true;
}

/**
 * @brief
 * Bounded string cursor max move function.
 *
 * Similar to the string move function, with the additional checks that no move
 * is done if the cursor is already at its maximum position, and that the cursor
 * is set to its maximum position if the bound is equal to 0 (unbounded).
 *
 * @param[in, out] str The stream whose cursor is moved.
 * @param[in] max_sz The string used to move the cursor.
 * @param[in] N The maximum number of characters in the string which the stream is at most moved by.
 *
 * @return Whether the operation was completed succesfully.
 */
template<typename S, typename T, std::enable_if_t<std::is_base_of<cdr_stream, S>::value, bool> = true >
bool max_string(S& str, const T& max_sz, size_t N)
{
  if (N == 0)
    str.position(SIZE_MAX); //unbounded string, theoretical length unlimited
  else
    return move_string(str, max_sz, N);

  return true;
}

}
}
}
}
} /* namespace org / eclipse / cyclonedds / core / cdr */
#endif
