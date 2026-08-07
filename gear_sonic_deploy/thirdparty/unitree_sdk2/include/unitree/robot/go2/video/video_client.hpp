#ifndef __UT_ROBOT_GO2_VIDEO_CLIENT_HPP__
#define __UT_ROBOT_GO2_VIDEO_CLIENT_HPP__

#include <unitree/robot/client/client.hpp>

namespace unitree
{
namespace robot
{
namespace go2
{
/*
 * VideoClient
 */
class VideoClient : public Client
{
public:
    explicit VideoClient();
    ~VideoClient();

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

#endif//__UT_ROBOT_GO2_VIDEO_CLIENT_HPP__
