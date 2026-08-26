/*
 * Copyright(c) 2006 to 2022 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */


/**
 * @file
 */

#ifndef CYCLONEDDS_TOPIC_TOPICTRAITS_HPP_
#define CYCLONEDDS_TOPIC_TOPICTRAITS_HPP_

#include "dds/ddsrt/heap.h"
#include "dds/ddsi/ddsi_serdata.h"
#include "org/eclipse/cyclonedds/core/cdr/cdr_enums.hpp"
#include "dds/features.hpp"

//forward declaration of c++ sertype wrapper
template <typename T, class S> class ddscxx_sertype;

namespace org
{
namespace eclipse
{
namespace cyclonedds
{

namespace core
{
namespace cdr
{
//forward declarations of streamer types
class basic_cdr_stream;
class xcdr_v1_stream;
class xcdr_v2_stream;
}
}

namespace topic
{

using core::cdr::extensibility;
using core::cdr::encoding_version;
using core::cdr::allowable_encodings_t;
using core::cdr::basic_cdr_stream;
using core::cdr::xcdr_v1_stream;
using core::cdr::xcdr_v2_stream;

template <class TOPIC> class TopicTraits
{
public:

    /**
     * @brief Returns whether TOPIC contains no key fields.
     *
     * Used in creating the CycloneDDS writer and equality comparisons with other topics.
     * This is one of the traits that is conditionally generated if there are any key fields.
     *
     * @return Whether TOPIC does not contain any fields marked as key fields.
     */
    static constexpr bool isKeyless()
    {
        return false;
    }

    /**
     * @brief Returns the name of the type of TOPIC.
     *
     * Used in creating the correct TopicDescription.
     * This trait is always generated for user-defined types, and this function is just a placeholder.
     *
     * @return The name of the type of TOPIC.
     */
    static constexpr const char *getTypeName()
    {
        return "";
    }

    /**
     * @brief Returns an instance of ddsi_sertype for TOPIC.
     *
     * Used by CycloneDDS-CXX to get a sertype, which contains the functions used by CycloneDDS which are specific to TOPIC.
     *
     * @param[in] kinds The serialization of the of the sertype to create.
     * @return A pointer to a new dssi_sertype.
     */
    static ddsi_sertype *getSerType(allowable_encodings_t kinds = allowableEncodings())
    {
        if (kinds & allowableEncodings() & DDS_DATA_REPRESENTATION_FLAG_XCDR1)
            return static_cast<ddsi_sertype*>(new ddscxx_sertype<TOPIC,basic_cdr_stream>());
        else if (kinds & allowableEncodings() & DDS_DATA_REPRESENTATION_FLAG_XCDR2)
            return static_cast<ddsi_sertype*>(new ddscxx_sertype<TOPIC,xcdr_v2_stream>());
        else
            return nullptr;
    }

    /**
     * @brief Returns the size of an instance of TOPIC.
     *
     * Used by shared memory implementation to determine the size of the block necessary to contain an instance of TOPIC.
     *
     * @return The size of an instance of TOPIC.
     */
    static constexpr size_t getSampleSize()
    {
        return sizeof(TOPIC);
    }

    /**
     * @brief Returns whether instances of TOPIC reference memory outside its own declaration.
     *
     * Used by shared memory implementation.
     * This trait will be generated as false if any strings or vectors are found anywhere in TOPIC's member tree.
     *
     * @return Whether TOPIC is a selfcontained type.
     */
    static constexpr bool isSelfContained()
    {
        return true;
    }

    /**
     * @brief Returns the allowable encodings for this topic.
     *
     * Used to determine which encoding type to write in combination with the data representation QoS.
     *
     * @return The allowable encodings of TOPIC.
     */
    static constexpr allowable_encodings_t allowableEncodings()
    {
        return 0xFFFFFFFFu;
    }

    /**
     * @brief Returns the xtypes extensibility of TOPIC.
     *
     * Used to determine which encoding type to write in the CDR header.
     * This trait will be generated if the extensibility of TOPIC differs from final.
     *
     * @return The extensibility of TOPIC.
     */
    static constexpr extensibility getExtensibility()
    {
        return extensibility::ext_final;
    }

#ifdef DDSCXX_HAS_TYPE_DISCOVERY
    /**
     * @brief Returns the typeid for TOPIC.
     *
     * Is a simple pass-through for the derivation of the typeid from the type information.
     * As C++ keeps the topic type as the template parameter, there is no need to look at the
     * sertype for this topic.
     *
     * @param[in] kind The kind of typeid.
     *
     * @return A pointer to the typeid of this topic.
     */
    static ddsi_typeid_t* getTypeId(const struct ddsi_sertype *, ddsi_typeid_kind_t kind)
    {
        auto ti = getTypeInfo(NULL);
        if (ti) {
            auto id = ddsi_typeinfo_typeid(ti, kind);
            ddsi_typeinfo_fini(ti);
            ddsrt_free(ti);
            return id;
        } else {
            return nullptr;
        }
    }

    /**
     * @brief Returns the type map for TOPIC.
     *
     * Takes the type map blob for this topic which is part of the generated type traits, and deserializes
     * the type map from this blob.
     *
     * @return A pointer to the typemap for this topic.
     */
    static ddsi_typemap_t* getTypeMap(const struct ddsi_sertype *)
    {
        ddsi_sertype_cdr_data_t cdr{type_map_blob_sz(), const_cast<uint8_t*>(type_map_blob())};
        return ddsi_typemap_deser(&cdr);
    }

    /**
     * @brief Returns the type info for TOPIC.
     *
     * Takes the type info blob for this topic which is part of the generated type traits, and deserializes
     * the type info from this blob.
     *
     * @return A pointer to the typeinfo for this topic.
     */
    static ddsi_typeinfo_t* getTypeInfo(const struct ddsi_sertype *)
    {
        ddsi_sertype_cdr_data_t cdr{type_info_blob_sz(), const_cast<uint8_t*>(type_info_blob())};
        return ddsi_typeinfo_deser(&cdr);
    }

    /**
     * @brief Returns size of the blob of the cdr serialized type map.
     *
     * This function is a placeholder, it will be instantiated for all valid topics.
     *
     * @return The size of the serialized type map blob.
     */
    static constexpr unsigned int type_map_blob_sz()
    {
        return static_cast<unsigned int>(-1);
    }

    /**
     * @brief Returns size of the blob of the cdr serialized type info.
     *
     * This function is a placeholder, it will be instantiated for all valid topics.
     *
     * @return The size of the serialized type info blob.
     */
    static constexpr unsigned int type_info_blob_sz()
    {
        return static_cast<unsigned int>(-1);
    }

    /**
     * @brief Returns pointer to the blob of the cdr serialized type map.
     *
     * This function is a placeholder, it will be instantiated for all valid topics.
     *
     * @return Pointer to the serialized type map blob.
     */
    static inline const uint8_t * type_map_blob()
    {
        return nullptr;
    }

    /**
     * @brief Returns pointer to the blob of the cdr serialized type info.
     *
     * This function is a placeholder, it will be instantiated for all valid topics.
     *
     * @return Pointer to the serialized type info blob.
     */
    static inline const uint8_t * type_info_blob()
    {
        return nullptr;
    }
#endif  //DDSCXX_HAS_TYPE_DISCOVERY

    /**
     * @brief Returns a pointer to the derived sertype.
     *
     * Returns a nullptr if no type can be derived succesfully.
     *
     * @param[in] data_representation The type of data representation to use.
     *
     * @return The pointer to the derived sertype.
     */
    static struct ddsi_sertype* deriveSertype(const struct ddsi_sertype *, dds_data_representation_id_t data_representation, dds_type_consistency_enforcement_qospolicy_t)
    {
        struct ddsi_sertype *ptr = nullptr;
        switch (data_representation) {
          case DDS_DATA_REPRESENTATION_XCDR1:
              ptr = getSerType(DDS_DATA_REPRESENTATION_FLAG_XCDR1);
              break;
          case DDS_DATA_REPRESENTATION_XCDR2:
              ptr = getSerType(DDS_DATA_REPRESENTATION_FLAG_XCDR2);
              break;
        }
        if (ptr) {
            uint32_t refc = ddsrt_atomic_ld32 (&ptr->flags_refc);
            ddsrt_atomic_st32 (&ptr->flags_refc, refc & ~DDSI_SERTYPE_REFC_MASK);
        }
        return ptr;
    }
};

}
}
}
}

#endif /* CYCLONEDDS_TOPIC_TOPICTRAITS_HPP_ */
