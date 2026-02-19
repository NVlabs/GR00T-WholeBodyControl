#ifndef __UT_ROBOT_GO2_VUI_CLIENT_HPP__
#define __UT_ROBOT_GO2_VUI_CLIENT_HPP__

#include <unitree/robot/client/client.hpp>

namespace unitree
{
namespace robot
{
namespace go2
{
/*
 * VuiClient
 */
class VuiClient : public Client
{
public:
    explicit VuiClient();
    ~VuiClient();

    void Init();

    /*
     * @brief SetSwitch
     * @api: 1001
     */
    int32_t SetSwitch(int enable);

    /*
     * @brief GetSwitch
     * @api: 1002
     */
    int32_t GetSwitch(int&);

    /*
     * @brief SetVolume
     * @api: 1003
     */
    int32_t SetVolume(int level);

    /*
     * @brief GetVolume
     * @api: 1004
     */
    int32_t GetVolume(int&);

    /*
     * @brief SetBrightness
     * @api: 1005
     */
    int32_t SetBrightness(int level);

    /*
     * @brief GetBrightness
     * @api: 1006
     */
    int32_t GetBrightness(int&);

};
}
}
}

#endif//__UT_ROBOT_GO2_VUI_CLIENT_HPP__
