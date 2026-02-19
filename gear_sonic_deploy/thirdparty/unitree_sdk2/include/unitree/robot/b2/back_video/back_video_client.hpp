#ifndef __UT_ROBOT_B2_BACK_VIDEO_CLIENT_HPP__
#define __UT_ROBOT_B2_BACK_VIDEO_CLIENT_HPP__

#include <unitree/robot/client/client.hpp>

namespace unitree
{
namespace robot
{
namespace b2
{
/*
 * BackVideoClient
 */
class BackVideoClient : public Client
{
public:
    explicit BackVideoClient();
    ~BackVideoClient();

    void Init();

    /*
     * @brief GetImageSample
     * @api: 1001
     */
    int32_t GetImageSample(std::vector<uint8_t>&);

};
}
}
}

#endif//__UT_ROBOT_B2_BACK_VIDEO_CLIENT_HPP__
