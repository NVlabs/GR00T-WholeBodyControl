#include <chrono>
#include <thread>
#include <unitree/idl/hg/HandState_.hpp> //replace your sdk path
#include <unitree/idl/hg/HandCmd_.hpp> //replace your sdk path
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <iostream>
#include <unistd.h>
#include <atomic>
#include <mutex>
#include <cmath>
#include <termios.h>
#include <unistd.h>
#include <eigen3/Eigen/Dense>


enum State {
    INIT,
    ROTATE,
    GRIP,
    STOP,
    PRINT
};

// set URDF Limits
const float maxLimits_left[7]=  {  1.05 ,  1.05  , 1.75 ,   0   ,  0    , 0     , 0   }; // set max motor value
const float minLimits_left[7]=  { -1.05 , -0.724 ,   0  , -1.57 , -1.75 , -1.57  ,-1.75}; 
const float maxLimits_right[7]= {  1.05 , 0.742  ,   0  ,  1.57 , 1.75  , 1.57  , 1.75}; 
const float minLimits_right[7]= { -1.05 , -1.05  , -1.75,    0  ,  0    ,   0   ,0    }; 

// Initing the dds configuration
std::string dds_namespace = "rt/dex3/left";
std::string sub_namespace = "rt/dex3/left/state";
unitree::robot::ChannelPublisherPtr<unitree_hg::msg::dds_::HandCmd_> handcmd_publisher;
unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::HandState_> handstate_subscriber;
unitree_hg::msg::dds_::HandCmd_ msg;
unitree_hg::msg::dds_::HandState_ state;
std::atomic<State> currentState(INIT);
std::mutex stateMutex;

#define MOTOR_MAX 7
#define SENSOR_MAX 9
uint8_t hand_id = 0;

typedef struct {
    uint8_t id     : 4;
    uint8_t status : 3;
    uint8_t timeout: 1;
} RIS_Mode_t;

// stateToString Method
const char* stateToString(State state) {
    switch (state) {
        case INIT: return "INIT";
        case ROTATE: return "ROTATE";
        case GRIP: return "GRIP";
        case STOP: return "STOP";
        case PRINT: return "PRINT";
        default: return "UNKNOWN";
    }
}

// Monitor user's input
char getNonBlockingInput() {
    struct termios oldt, newt;
    char ch;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt); 
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar(); 

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt); 
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    return ch;
}

void userInputThread() {
    while (true) {
        char ch = getNonBlockingInput();
        if (ch == 'q') {
            std::cout << "Exiting..." << std::endl;
                currentState = STOP;
                break;
        } else if (ch == 'r') {
            currentState = ROTATE;
        } else if (ch == 'g') {
            currentState = GRIP;
        } else if (ch == 'p') {
            currentState = PRINT;
        } else if (ch == 's') {
            currentState = STOP;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); 
    }
}

// this method can send kp and kd to motors
void rotateMotors(bool isLeftHand) {
    static int _count = 1; 
    static int dir = 1;    
    const float* maxLimits = isLeftHand ? maxLimits_left : maxLimits_right;
    const float* minLimits = isLeftHand ? minLimits_left : minLimits_right;

    for (int i = 0; i < MOTOR_MAX; i++) {
        RIS_Mode_t ris_mode;
        ris_mode.id = i;        
        ris_mode.status = 0x01; 
   
        
        uint8_t mode = 0;
        mode |= (ris_mode.id & 0x0F);           
        mode |= (ris_mode.status & 0x07) << 4;    
        mode |= (ris_mode.timeout & 0x01) << 7;  
        msg.motor_cmd()[i].mode(mode);
        msg.motor_cmd()[i].tau(0);
        msg.motor_cmd()[i].kp(0.5);    
        msg.motor_cmd()[i].kd(0.1);    


        float range = maxLimits[i] - minLimits[i];
        float mid = (maxLimits[i] + minLimits[i]) / 2.0; 
        float amplitude = range / 2.0; 
        float q = mid + amplitude * sin(_count / 20000.0 * M_PI); 

        msg.motor_cmd()[i].q(q);
    }

    handcmd_publisher->Write(msg);
    _count += dir;


    if (_count >= 10000) {
        dir = -1;
    }
    if (_count <= -10000) {
        dir = 1;
    }

    usleep(100); 
}

// this method can send static position to motors
void gripHand(bool isLeftHand) {

    const float* maxLimits = isLeftHand ? maxLimits_left : maxLimits_right;
    const float* minLimits = isLeftHand ? minLimits_left : minLimits_right;

    for (int i = 0; i < MOTOR_MAX; i++) {
        RIS_Mode_t ris_mode;
        ris_mode.id = i;        
        ris_mode.status = 0x01; 
    
        
        uint8_t mode = 0;
        mode |= (ris_mode.id & 0x0F);            
        mode |= (ris_mode.status & 0x07) << 4;    
        mode |= (ris_mode.timeout & 0x01) << 7;   
        msg.motor_cmd()[i].mode(mode);
        msg.motor_cmd()[i].tau(0);

      
        float mid = (maxLimits[i] + minLimits[i]) / 2.0;


        msg.motor_cmd()[i].q(mid); 
        msg.motor_cmd()[i].dq(0);  
        msg.motor_cmd()[i].kp(1.5);      
        msg.motor_cmd()[i].kd(0.1);   
    }


    handcmd_publisher->Write(msg);
    usleep(1000000);
}

// this method can send dynamic position to motors
void stopMotors() {
    for (int i = 0; i < MOTOR_MAX; i++) {
        RIS_Mode_t ris_mode;
        ris_mode.id = i;       
        ris_mode.status = 0x01; 
        ris_mode.timeout = 0x01; 
        
        uint8_t mode = 0;
        mode |= (ris_mode.id & 0x0F);            
        mode |= (ris_mode.status & 0x07) << 4;  
        mode |= (ris_mode.timeout & 0x01) << 7;   
        msg.motor_cmd()[i].mode(mode);
        msg.motor_cmd()[i].tau(0);
        msg.motor_cmd()[i].dq(0); 
        msg.motor_cmd()[i].kp(0);
        msg.motor_cmd()[i].kd(0);
        msg.motor_cmd()[i].q(0); 

    }
    handcmd_publisher->Write(msg);
    usleep(1000000); 
}

// this method can subscribe dds and show the position for now
void printState(bool isLeftHand){
    Eigen::Matrix<float, 7, 1> q;

    const float* maxLimits = isLeftHand ? maxLimits_left : maxLimits_right;
    const float* minLimits = isLeftHand ? minLimits_left : minLimits_right;
    for(int i = 0; i < 7; i++) 
    {
        q(i) = state.motor_state()[i].q();
      
        q(i) = (q(i) - minLimits[i] ) / (maxLimits[i] - minLimits[i]);
        q(i) = std::clamp(q(i), 0.0f, 1.0f);
    }
    std::cout << "\033[2J\033[H"; 
    std::cout << "-- Hand State --\n";
    std::cout << "--- Current State: " << "Test" << " ---\n";
    std::cout << "Commands:\n";
    std::cout << "  r - Rotate\n";
    std::cout << "  g - Grip\n";
    std::cout << "  t - Test\n";
    std::cout << "  q - Quit\n";
    if(isLeftHand){
        std::cout << " L: " << q.transpose() << std::endl;
    }else std::cout << " R: " << q.transpose() << std::endl;
    usleep(0.1 * 1e6);

}

void StateHandler(const void *message) {
  state = *(unitree_hg::msg::dds_::HandState_ *)message;
}




int main(int argc, const char** argv)
{
    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     Dex3 Hand Example      \n\n";
    std::string input;
    std::cout << "Please input the hand id (L for left hand, R for right hand): ";
    std::cin >> input;

    if (input == "L") {
        hand_id = 0;
        dds_namespace = "rt/dex3/left";
        sub_namespace = "rt/lf/dex3/left/state";
    } else if (input == "R") {
        hand_id = 1;
        dds_namespace = "rt/dex3/right";
        sub_namespace = "rt/lf/dex3/right/state";
    } else {
        std::cout << "Invalid hand id. Please input 'L' or 'R'." << std::endl;
        return -1;
    }

    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " networkInterface" << std::endl;
        exit(-1); 
    }
    unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
    handcmd_publisher.reset(new unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::HandCmd_>(dds_namespace + "/cmd"));
    handstate_subscriber.reset(new unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::HandState_>(sub_namespace));
    handcmd_publisher->InitChannel();
    handstate_subscriber->InitChannel(
      std::bind(&StateHandler, std::placeholders::_1), 1);
    state.motor_state().resize(MOTOR_MAX);
    state.press_sensor_state().resize(SENSOR_MAX);
    msg.motor_cmd().resize(MOTOR_MAX);
    
    // handcmd_publisher->msg_.motor_cmd().resize(MOTOR_MAX);

   
    std::thread inputThread(userInputThread);
    State lastState = INIT; 
    while (true) {
        State state;
        {
            std::lock_guard<std::mutex> lock(stateMutex);
            state = currentState.load();
        }
                
        if (state != lastState) {
            std::cout << "\n--- Current State: " << stateToString(state) << " ---\n";
            std::cout << "Commands:\n";
            std::cout << "  r - Rotate\n";
            std::cout << "  g - Grip\n";
            std::cout << "  p - Print_state\n";
            std::cout << "  q - Quit\n";
            std::cout << "  s - Stop\n";
            lastState = state; 
        }

        switch (state) {
            case INIT:
                std::cout << "Initializing..." << std::endl;
                currentState = ROTATE;
                break;
            case ROTATE:
                rotateMotors(input == "L");
                break;
            case GRIP:
                gripHand(input == "L");
                break;
            case STOP:
                stopMotors();
                break;
            case PRINT:
                printState(input == "L");
                break;
            default:
                std::cout << "Invalid state!" << std::endl;
                inputThread.join();  
                break;
        }
    }

    return 0;
}