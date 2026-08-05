#include <stddef.h>
#include <iostream>
#include <PXREARobotSDK.h>
#include <chrono>
#include <thread>

void OnPXREAClientCallback(void* context,PXREAClientCallbackType type,int status,void* userData)
{

    switch (type)
    {

    case PXREAServerConnect:
        std::cout <<"server connect"  << std::endl;
        break;
    case PXREAServerDisconnect:
        std::cout  <<"server disconnect"  << std::endl;
        break;
    case PXREADeviceFind:
        std::cout << "device find"<< (const char*)userData << std::endl;
        break;
    case PXREADeviceMissing:
        std::cout <<"device missing"<<(const char*)userData<<  std::endl;
        break;
    case PXREADeviceConnect:
        std::cout <<"device connect"<<(const char*)userData<<status<< std::endl;
        break;
    case PXREADeviceStateJson:
        auto& dsj = *((PXREADevStateJson*)userData);

        std::cout <<"device data"<< dsj.stateJson << std::endl;
        break;

    }
}






void testDeinit()
{
    PXREADeinit();
}

int main(int argc, char *argv[])
{
    PXREAInit(NULL, OnPXREAClientCallback, PXREAFullMask);

    while(1)
    {

        std::cout << " main loop alive." <<std::endl;
        std::this_thread::sleep_for (std::chrono::seconds(1));
    }


    return 0;
}
