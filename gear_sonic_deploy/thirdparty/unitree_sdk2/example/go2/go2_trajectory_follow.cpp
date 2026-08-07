#include <unitree/robot/go2/sport/sport_client.hpp>
#include <unitree/common/thread/recurrent_thread.hpp>
#include <math.h>

class Custom
{
public:
  Custom() {}
  void control();

  unitree::robot::go2::SportClient tc;

  int c = 0;
  float dt = 0.002; // 0.001~0.01
};

void Custom::control()
{
    c++;
    
    int32_t ret;

    float vx = 0.3;
    float delta = 0.06;
    static float count = 0;
    count += dt;
    std::vector<unitree::robot::go2::PathPoint> path;
    for (int i=0; i<30; i++) {
      unitree::robot::go2::PathPoint p;
      float var = (count + i * delta);
      p.timeFromStart = i * delta;
      p.x = vx * var;
      p.y = 0.6 * sin(M_PI * vx * var);
      p.yaw = 2*0.6 * vx * M_PI * cos(M_PI * vx * var);
      p.vx = vx;
      p.vy = M_PI * vx * (0.6 * cos(M_PI * vx * var));
      p.vyaw = - M_PI * vx*2*0.6 * vx * M_PI * sin(M_PI * vx * var);
      path.push_back(p);
    }

    ret = tc.TrajectoryFollow(path);
    if(ret != 0){
      std::cout << "Call TrajectoryFollow: " << ret << std::endl;
    }

    std::cout << c << std::endl;
}

int main(int argc, char** argv)
{
  unitree::robot::ChannelFactory::Instance()->Init(0);

  Custom custom;
  custom.tc.SetTimeout(10.0f);
  custom.tc.Init();

  unitree::common::ThreadPtr threadPtr = unitree::common::CreateRecurrentThread(custom.dt * 1000000, std::bind(&Custom::control, &custom));

  while (1)
  {
    sleep(10);
  }

  return 0;
}
