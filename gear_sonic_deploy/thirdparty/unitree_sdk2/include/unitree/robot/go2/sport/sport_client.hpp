#ifndef __UT_ROBOT_GO2_SPORT_CLIENT_HPP__
#define __UT_ROBOT_GO2_SPORT_CLIENT_HPP__

#include <unitree/robot/client/client.hpp>

namespace unitree
{
namespace robot
{
namespace go2
{
/*
 * PathPoint
 */
struct stPathPoint
{
  float timeFromStart;
  float x;
  float y;
  float yaw;
  float vx;
  float vy;
  float vyaw;
};

typedef struct stPathPoint PathPoint;

/*
 * SportClient
 */
class SportClient : public Client
{
public:
    explicit SportClient(bool enableLease = false);
    ~SportClient();

    void Init();

    int32_t Damp();
    int32_t BalanceStand();
    int32_t StopMove();
    int32_t StandUp();
    int32_t StandDown();
    int32_t RecoveryStand();
    int32_t Euler(float roll, float pitch, float yaw);
    int32_t Move(float vx, float vy, float vyaw);
    int32_t Sit();
    int32_t RiseSit();
    int32_t SpeedLevel(int level);
    int32_t Hello();
    int32_t Stretch();
    int32_t SwitchJoystick(bool flag);
    int32_t Content();
    int32_t Heart();
    int32_t Pose(bool flag);
    int32_t Scrape();
    int32_t FrontFlip();
    int32_t FrontJump();
    int32_t FrontPounce();
    int32_t Dance1();
    int32_t Dance2();
    int32_t LeftFlip();
    int32_t BackFlip();
    int32_t HandStand(bool flag);
    int32_t FreeWalk();
    int32_t FreeBound(bool flag);
    int32_t FreeJump(bool flag);
    int32_t FreeAvoid(bool flag);
    int32_t ClassicWalk(bool flag);
    int32_t WalkUpright(bool flag);
    int32_t CrossStep(bool flag);
    int32_t AutoRecoverSet(bool flag);
    int32_t AutoRecoverGet(bool& flag);
    int32_t StaticWalk();
    int32_t TrotRun();
    int32_t EconomicGait();
    int32_t SwitchAvoidMode();



};
}
}
}

#endif//__UT_ROBOT_GO2_SPORT_CLIENT_HPP__
