#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/common/time/time_tool.hpp>
#include <unitree/common/thread/thread.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>

using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree::robot::b2;

#define TOPIC_LOWCMD "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"

constexpr double PosStopF = (2.146E+9f);
constexpr double VelStopF = (16000.0f);

class Custom
{
public:
    explicit Custom(){}
    ~Custom(){}

    void Init();
    void Start();

private:
    void InitLowCmd();
    void LowStateMessageHandler(const void* messages);
    void LowCmdWrite();
    int queryMotionStatus();
    std::string queryServiceName(std::string form,std::string name);
 
private:
    float Kp = 70.0;
    float Kd = 5.0;
    double time_consume = 0;
    int rate_count = 0;
    int sin_count = 0;
    int motiontime = 0;
    float dt = 0.002; // 0.001~0.01

    MotionSwitcherClient msc;

    unitree_go::msg::dds_::LowCmd_ low_cmd{};      // default init
    unitree_go::msg::dds_::LowState_ low_state{};  // default init

    /*publisher*/
    ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_publisher;
    /*subscriber*/
    ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber;

    /*LowCmd write thread*/
    ThreadPtr lowCmdWriteThreadPtr;

    float _targetPos_1[12] = {0.0, 1.36, -2.65, 0.0, 1.36, -2.65,
                              -0.2, 1.36, -2.65, 0.2, 1.36, -2.65};

    float _targetPos_2[12] = {0.0, 0.67, -1.3, 0.0, 0.67, -1.3,
                              0.0, 0.67, -1.3, 0.0, 0.67, -1.3};

    float _targetPos_3[12] = {-0.35, 1.36, -2.65, 0.35, 1.36, -2.65,
                              -0.5, 1.36, -2.65, 0.5, 1.36, -2.65};

    float _startPos[12];
    float _duration_1 = 500;   
    float _duration_2 = 500; 
    float _duration_3 = 2000;   
    float _duration_4 = 900;   
    float _percent_1 = 0;    
    float _percent_2 = 0;    
    float _percent_3 = 0;    
    float _percent_4 = 0;    

    bool firstRun = true;
    bool done = false;
};

uint32_t crc32_core(uint32_t* ptr, uint32_t len)
{
    unsigned int xbit = 0;
    unsigned int data = 0;
    unsigned int CRC32 = 0xFFFFFFFF;
    const unsigned int dwPolynomial = 0x04c11db7;

    for (unsigned int i = 0; i < len; i++)
    {
        xbit = 1 << 31;
        data = ptr[i];
        for (unsigned int bits = 0; bits < 32; bits++)
        {
            if (CRC32 & 0x80000000)
            {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            }
            else
            {
                CRC32 <<= 1;
            }

            if (data & xbit)
                CRC32 ^= dwPolynomial;
            xbit >>= 1;
        }
    }

    return CRC32;
}

void Custom::Init()
{
    InitLowCmd();

    /*create publisher*/
    lowcmd_publisher.reset(new ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
    lowcmd_publisher->InitChannel();

    /*create subscriber*/
    lowstate_subscriber.reset(new ChannelSubscriber<unitree_go::msg::dds_::LowState_>(TOPIC_LOWSTATE));
    lowstate_subscriber->InitChannel(std::bind(&Custom::LowStateMessageHandler, this, std::placeholders::_1), 1);

    /*init MotionSwitcherClient*/
    msc.SetTimeout(10.0f); 
    msc.Init();

    /*Shut down motion control-related service*/
    while(queryMotionStatus())
    {
        std::cout << "Try to deactivate the motion control-related service." << std::endl;
        int32_t ret = msc.ReleaseMode(); 
        if (ret == 0) {
            std::cout << "ReleaseMode succeeded." << std::endl;
        } else {
            std::cout << "ReleaseMode failed. Error code: " << ret << std::endl;
        }
        sleep(5);
    }
}

void Custom::InitLowCmd()
{
    low_cmd.head()[0] = 0xFE;
    low_cmd.head()[1] = 0xEF;
    low_cmd.level_flag() = 0xFF;
    low_cmd.gpio() = 0;

    for(int i=0; i<20; i++)
    {
        low_cmd.motor_cmd()[i].mode() = (0x01);   // motor switch to servo (PMSM) mode
        low_cmd.motor_cmd()[i].q() = (PosStopF);
        low_cmd.motor_cmd()[i].kp() = (0);
        low_cmd.motor_cmd()[i].dq() = (VelStopF);
        low_cmd.motor_cmd()[i].kd() = (0);
        low_cmd.motor_cmd()[i].tau() = (0);
    }
}

int Custom::queryMotionStatus()
{
    std::string robotForm,motionName;
    int motionStatus;
    int32_t ret = msc.CheckMode(robotForm,motionName);
    if (ret == 0) {
        std::cout << "CheckMode succeeded." << std::endl;
    } else {
        std::cout << "CheckMode failed. Error code: " << ret << std::endl;
    }
    if(motionName.empty())
    {
        std::cout << "The motion control-related service is deactivated." << std::endl;
        motionStatus = 0;
    }
    else
    {
        std::string serviceName = queryServiceName(robotForm,motionName);
        std::cout << "Service: "<< serviceName<< " is activate" << std::endl;
        motionStatus = 1;
    }
    return motionStatus;
}

std::string Custom::queryServiceName(std::string form,std::string name)
{
    if(form == "0")
    {
        if(name == "normal" ) return "sport_mode"; 
        if(name == "ai" ) return "ai_sport"; 
        if(name == "advanced" ) return "advanced_sport"; 
    }
    else
    {
        if(name == "ai-w" ) return "wheeled_sport(go2W)"; 
        if(name == "normal-w" ) return "wheeled_sport(b2W)";
    }
    return "";
}

void Custom::Start()
{
    /*loop publishing thread*/
    lowCmdWriteThreadPtr = CreateRecurrentThreadEx("writebasiccmd", UT_CPU_ID_NONE, 2000, &Custom::LowCmdWrite, this);
}

void Custom::LowStateMessageHandler(const void* message)
{
    low_state = *(unitree_go::msg::dds_::LowState_*)message;
}

void Custom::LowCmdWrite()
{
    if(_percent_4<1)
    {
        std::cout<<"Read sensor data example: "<<std::endl;
        std::cout<<"Joint 0 pos: "<<low_state.motor_state()[0].q()<<std::endl;
        std::cout<<"Imu accelerometer : "<<"x: "<<low_state.imu_state().accelerometer()[0]<<" y: "<<low_state.imu_state().accelerometer()[1]<<" z: "<<low_state.imu_state().accelerometer()[2]<<std::endl;
        std::cout<<"Foot force "<<low_state.foot_force()[0]<<std::endl;
        std::cout<<std::endl;
    }
    if((_percent_4 == 1) && ( done == false))
    {
        std::cout<<"The example is done! "<<std::endl;
        std::cout<<std::endl;
        done = true;
    }

    motiontime++;
    if(motiontime>=500)
    {
        if(firstRun)
        {
            for(int i = 0; i < 12; i++)
            {
                _startPos[i] = low_state.motor_state()[i].q();
            }
            firstRun = false;
        }

        _percent_1 += (float)1 / _duration_1;
        _percent_1 = _percent_1 > 1 ? 1 : _percent_1;
        if (_percent_1 < 1)
        {
            for (int j = 0; j < 12; j++)
            {
                low_cmd.motor_cmd()[j].q() = (1 - _percent_1) * _startPos[j] + _percent_1 * _targetPos_1[j];
                low_cmd.motor_cmd()[j].dq() = 0;
                low_cmd.motor_cmd()[j].kp() = Kp;
                low_cmd.motor_cmd()[j].kd() = Kd;
                low_cmd.motor_cmd()[j].tau() = 0;
            }
        
        }
        if ((_percent_1 == 1)&&(_percent_2 < 1))
        {
            _percent_2 += (float)1 / _duration_2;
            _percent_2 = _percent_2 > 1 ? 1 : _percent_2;

            for (int j = 0; j < 12; j++)
            {
                low_cmd.motor_cmd()[j].q() = (1 - _percent_2) * _targetPos_1[j] + _percent_2 * _targetPos_2[j];
                low_cmd.motor_cmd()[j].dq() = 0;
                low_cmd.motor_cmd()[j].kp() = Kp;
                low_cmd.motor_cmd()[j].kd() = Kd;
                low_cmd.motor_cmd()[j].tau() = 0;
            }
        }

        if ((_percent_1 == 1)&&(_percent_2 == 1)&&(_percent_3<1))
        {
            _percent_3 += (float)1 / _duration_3;
            _percent_3 = _percent_3 > 1 ? 1 : _percent_3;

            for (int j = 0; j < 12; j++)
            {
                low_cmd.motor_cmd()[j].q() =  _targetPos_2[j];
                low_cmd.motor_cmd()[j].dq() = 0;
                low_cmd.motor_cmd()[j].kp() = Kp;
                low_cmd.motor_cmd()[j].kd() = Kd;
                low_cmd.motor_cmd()[j].tau() = 0;
            }
            if(_percent_3<0.4)
            {
                for (int j = 12; j < 16; j++)
                {
                    low_cmd.motor_cmd()[j].q() =  0;
                    low_cmd.motor_cmd()[j].kp() = 0;
                    low_cmd.motor_cmd()[j].dq() = 3;

                    low_cmd.motor_cmd()[j].kd() = Kd;
                    low_cmd.motor_cmd()[j].tau() = 0;
                }
                            
            }
            else if((_percent_3>=0.4)&&(_percent_3<0.8))
            {
                for (int j = 12; j < 16; j++)
                {
                    low_cmd.motor_cmd()[j].q() =  0;
                    low_cmd.motor_cmd()[j].kp() = 0;
                    low_cmd.motor_cmd()[j].dq() = -3;

                    low_cmd.motor_cmd()[j].kd() = Kd;
                    low_cmd.motor_cmd()[j].tau() = 0;
                }
                            
            }
            else if(_percent_3>=0.8)
            {
                for (int j = 12; j < 16; j++)
                {
                    low_cmd.motor_cmd()[j].q() =  0;
                    low_cmd.motor_cmd()[j].kp() = 0;
                    low_cmd.motor_cmd()[j].dq() = 0;

                    low_cmd.motor_cmd()[j].kd() = Kd;
                    low_cmd.motor_cmd()[j].tau() = 0;
                } 
            }
            
        }
        if ((_percent_1 == 1)&&(_percent_2 == 1)&&(_percent_3==1)&&((_percent_4<=1)))
        {
            _percent_4 += (float)1 / _duration_4;
            _percent_4 = _percent_4 > 1 ? 1 : _percent_4;
            for (int j = 0; j < 12; j++)
            {
                low_cmd.motor_cmd()[j].q() = (1 - _percent_4) * _targetPos_2[j] + _percent_4 * _targetPos_3[j];
                low_cmd.motor_cmd()[j].dq() = 0;
                low_cmd.motor_cmd()[j].kp() = Kp;
                low_cmd.motor_cmd()[j].kd() = Kd;
                low_cmd.motor_cmd()[j].tau() = 0;
            }
        }
        low_cmd.crc() = crc32_core((uint32_t *)&low_cmd, (sizeof(unitree_go::msg::dds_::LowCmd_)>>2)-1);
    
        lowcmd_publisher->Write(low_cmd);
    }
   
}

int main(int argc, const char** argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " networkInterface" << std::endl;
        exit(-1); 
    }

    std::cout << "WARNING: Make sure the robot is hung up or lying on the ground." << std::endl
            << "Press Enter to continue..." << std::endl;
    std::cin.ignore();

    ChannelFactory::Instance()->Init(0, argv[1]);

    Custom custom;
    custom.Init();
    custom.Start();
  
    while (1)
    {
        sleep(10);
    }

    return 0;
}
