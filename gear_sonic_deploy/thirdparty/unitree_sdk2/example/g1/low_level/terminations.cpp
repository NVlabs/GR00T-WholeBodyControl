#include <unitree/robot/g1/common/terminations.hpp>
#include <boost/program_options.hpp>
#include <thread>

namespace po = boost::program_options;

using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;

int main(int argc, char** argv)
{
    // Parse command line arguments
    po::options_description desc("Unitree G1 termination functions testing.");
    desc.add_options()
        ("network,n", po::value<std::string>()->default_value(""), "dds network interface")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    std::cout << desc << std::endl;
    
    // DDS Init
    ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    auto lowstate_subscriber = std::make_shared<ChannelSubscriber<LowState_>>("rt/lowstate");
    LowState_ lowstate;
    lowstate_subscriber->InitChannel([&lowstate](const void* message) {
        lowstate = *(const LowState_*)message;
    });

    std::cout << "Checking terminations..." << std::endl;

    while (true)
    {
        if (g1::bad_orientation(lowstate, 1.0f)) { // Tip the robot over to test bad orientation
            std::cout << "Bad orientation detected!" << std::endl;
        }
        if (g1::lost_connection(lowstate_subscriber, 1000)) { // Unplug the network cable to test lost connection
            std::cout << "Lost connection!" << std::endl;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    return 0;
}
