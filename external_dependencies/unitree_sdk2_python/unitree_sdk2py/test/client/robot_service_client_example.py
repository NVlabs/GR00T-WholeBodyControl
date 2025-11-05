import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient

if __name__ == "__main__":
    ChannelFactoryInitialize(0, "enx000ec6768747")
    rsc = RobotStateClient()
    rsc.SetTimeout(3.0)
    rsc.Init()

    while True:
        print("##################GetServerApiVersion###################")
        code, serverAPiVersion = rsc.GetServerApiVersion()

        if code != 0:
            print("get server api error. code:", code)
        else:
            print("get server api version:", serverAPiVersion)

        time.sleep(3)

        print("##################ServiceList###################")
        code, lst = rsc.ServiceList()
        
        if code != 0:
            print("list sevrice error. code:", code)
        else:
            print("list service success. len:", len(lst))
            for s in lst:
                print("name:", s.name, ", protect:", s.protect, ", status:", s.status)

        time.sleep(3)

        print("##################ServiceSwitch###################")
        code = rsc.ServiceSwitch("sport_mode", False)
        if code != 0:
            print("service stop sport_mode error. code:", code)
        else:
            print("service stop sport_mode success. code:", code)

        time.sleep(1)

        code = rsc.ServiceSwitch("sport_mode", True)
        if code != 0:
            print("service start sport_mode error. code:", code)
        else:
            print("service start sport_mode success. code:", code)
        
        time.sleep(3)

