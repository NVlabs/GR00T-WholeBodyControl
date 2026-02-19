#ifndef __UT_ROBOT_B2_SPORT_CLIENT_HPP__
#define __UT_ROBOT_B2_SPORT_CLIENT_HPP__

#include <unitree/robot/client/client.hpp>

namespace unitree
{
namespace robot
{
namespace b2
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
    
    int32_t Move(float vx, float vy, float vyaw);
    
    int32_t SwitchGait(int d);

    int32_t BodyHeight(float height);
    
    int32_t SpeedLevel(int level);
    
    int32_t TrajectoryFollow(std::vector<unitree::robot::b2::PathPoint> &path);

    int32_t ContinuousGait(bool flag);

    int32_t MoveToPos(float x, float y, float yaw);
    
    int32_t SwitchMoveMode(bool flag);
    
    int32_t VisionWalk(bool flag);

    int32_t HandStand(bool flag);

    int32_t AutoRecoverySet(bool flag);

    int32_t FreeWalk();

    int32_t ClassicWalk(bool flag);

    int32_t FastWalk(bool flag);

    int32_t Euler(float roll, float pitch, float yaw);
    
};
}
}
}

#endif//__UT_ROBOT_B2_SPORT_CLIENT_HPP__
