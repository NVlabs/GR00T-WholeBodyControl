#ifndef __UT_DDS_TRAINTS_HPP__
#define __UT_DDS_TRAINTS_HPP__

namespace unitree
{
namespace common
{
#include <org/eclipse/cyclonedds/topic/TopicTraits.hpp>

#define DdsGetTypeName(TYPE) \
    org::eclipse::cyclonedds::topic::TopicTraits<TYPE>::getTypeName()

#define DdsIsKeyless(TYPE) \
    org::eclipse::cyclonedds::topic::TopicTraits<TYPE>::isKeyless()

}
}
#endif//__UT_DDS_TRAINTS_HPP__
