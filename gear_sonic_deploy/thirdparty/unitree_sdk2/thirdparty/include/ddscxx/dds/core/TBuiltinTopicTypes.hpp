/* Copyright 2010, Object Management Group, Inc.
* Copyright 2010, PrismTech, Corp.
* Copyright 2010, Real-Time Innovations, Inc.
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
#ifndef OMG_DDS_CORE_TBUILTIN_TOPIC_TYPES_HPP_
#define OMG_DDS_CORE_TBUILTIN_TOPIC_TYPES_HPP_


#if defined (OMG_DDS_X_TYPES_BUILTIN_TOPIC_TYPES_SUPPORT)
namespace dds
{
namespace core
{

template <typename DELEGATE>
class TBytesTopicType;

template <typename DELEGATE>
class TStringTopicType;

template <typename DELEGATE>
class TKeyedBytesTopicType;

template <typename DELEGATE>
class TKeyedStringTopicType;
}
}

template <typename DELEGATE>
class dds::core::TBytesTopicType
{
public:
    TBytesTopicType();
    TBytesTopicType(const std::vector<uint32_t>& data);
public:
    operator std::vector<uint32_t>& ();
public:
    const std::vector<uint8_t>& data();
    void data(const std::vector<uint8_t>& bytes);
};

template <typename DELEGATE>
class dds::core::TStringTopicType
{
public:
    TStringTopicType();
    TStringTopicType(const std::string& data);
public:
    operator std::string& ();
public:
    const std::string& data();
    void data(const std::string& bytes);
};


/**
 * This class represents a built-in topic type that can be used
 * to readily create Topics, DataReaders and DataWriters for this type.
 * No code generation is required when using this type.
 */
template <typename DELEGATE>
class dds::core::TKeyedStringTopicType<DELEGATE>
{
public:
    TKeyedStringTopicType();
    TKeyedStringTopicType(const std::string& key, const std::string& value);

public:
    const std::string& key() const;
    void key(const std::string& value);

    const std::string& value() const;
    void value(const std::string& value);
};

/**
 * This class represents a built-in topic type that can be used
 * to readily create Topics, DataReaders and DataWriters for this type.
 * No code generation is required when using this type.
 */
template <typename DELEGATE>
class dds::core::TKeyedBytesTopicType<DELEGATE>
{
public:
    TKeyedBytesTopicType();
    TKeyedBytesTopicType(const std::string& key, const std::vector<uint8_t>& bytes);

public:
    const std::string& key() const;
    void key(const std::string& value);

    const std::vector<uint8_t>& value() const;
    void value(const std::vector<uint8_t>& value);
};


#endif /* OMG_DDS_X_TYPES_BUILTIN_TOPIC_TYPES_SUPPORT */

#endif /* OMG_DDS_CORE_TBUILTIN_TOPIC_TYPES_HPP_ */
