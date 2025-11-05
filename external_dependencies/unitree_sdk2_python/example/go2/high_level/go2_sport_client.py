import time
import sys
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.go2.sport.sport_client import (
    SportClient,
    PathPoint,
    SPORT_PATH_POINT_SIZE,
)
import math
from dataclasses import dataclass

@dataclass
class TestOption:
    name: str
    id: int

option_list = [
    TestOption(name="damp", id=0),         
    TestOption(name="stand_up", id=1),     
    TestOption(name="stand_down", id=2),   
    TestOption(name="move forward", id=3),         
    TestOption(name="move lateral", id=4),    
    TestOption(name="move rotate", id=5),  
    TestOption(name="stop_move", id=6),  
    TestOption(name="switch_gait", id=7),    
    TestOption(name="switch_gait", id=8),
    TestOption(name="balanced stand", id=9),     
    TestOption(name="recovery", id=10),
    TestOption(name="recovery", id=10),       
    TestOption(name="left flip", id=11),      
    TestOption(name="back flip", id=12),
    TestOption(name="free walk", id=13),  
    TestOption(name="free bound", id=14), 
    TestOption(name="free avoid", id=15),  
    TestOption(name="walk stair", id=16), 
    TestOption(name="walk upright", id=17),
    TestOption(name="cross step", id=18),
    TestOption(name="free jump", id=19)       
]

class UserInterface:
    def __init__(self):
        self.test_option_ = None

    def convert_to_int(self, input_str):
        try:
            return int(input_str)
        except ValueError:
            return None

    def terminal_handle(self):
        input_str = input("Enter id or name: \n")

        if input_str == "list":
            self.test_option_.name = None
            self.test_option_.id = None
            for option in option_list:
                print(f"{option.name}, id: {option.id}")
            return

        for option in option_list:
            if input_str == option.name or self.convert_to_int(input_str) == option.id:
                self.test_option_.name = option.name
                self.test_option_.id = option.id
                print(f"Test: {self.test_option_.name}, test_id: {self.test_option_.id}")
                return

        print("No matching test option found.")

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} networkInterface")
        sys.exit(-1)

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    ChannelFactoryInitialize(0, sys.argv[1])

    test_option = TestOption(name=None, id=None) 
    user_interface = UserInterface()
    user_interface.test_option_ = test_option

    sport_client = SportClient()  
    sport_client.SetTimeout(10.0)
    sport_client.Init()
    while True:

        user_interface.terminal_handle()

        print(f"Updated Test Option: Name = {test_option.name}, ID = {test_option.id}\n")

        if test_option.id == 0:
            sport_client.Damp()
        elif test_option.id == 1:
            sport_client.StandUp()
        elif test_option.id == 2:
            sport_client.StandDown()
        elif test_option.id == 3:
            sport_client.Move(0.3,0,0)
        elif test_option.id == 4:
            sport_client.Move(0,0.3,0)
        elif test_option.id == 5:
            sport_client.Move(0,0,0.5)
        elif test_option.id == 6:
            sport_client.StopMove()
        elif test_option.id == 7:
            sport_client.SwitchGait(0)
        elif test_option.id == 8:
            sport_client.SwitchGait(1)
        elif test_option.id == 9:
            sport_client.BalanceStand()
        elif test_option.id == 10:
            sport_client.RecoveryStand()
        elif test_option.id == 11:
            ret = sport_client.LeftFlip()
            print("ret: ",ret)
        elif test_option.id == 12:
            ret = sport_client.BackFlip()
            print("ret: ",ret)
        elif test_option.id == 13:
            ret = sport_client.FreeWalk(True)
            print("ret: ",ret)
        elif test_option.id == 14:
            ret = sport_client.FreeBound(True)
            print("ret: ",ret)
            time.sleep(2)
            ret = sport_client.FreeBound(False)
            print("ret: ",ret)
        elif test_option.id == 14:
            ret = sport_client.FreeBound(True)
            print("ret: ",ret)
            time.sleep(2)
            ret = sport_client.FreeBound(False)
            print("ret: ",ret)
        elif test_option.id == 15:
            ret = sport_client.FreeAvoid(True)
            print("ret: ",ret)
            time.sleep(2)
            ret = sport_client.FreeAvoid(False)
            print("ret: ",ret)
        elif test_option.id == 16:
            ret = sport_client.WalkStair(True)
            print("ret: ",ret)
            time.sleep(10)
            ret = sport_client.WalkStair(False)
            print("ret: ",ret)
        elif test_option.id == 17:
            ret = sport_client.WalkUpright(True)
            print("ret: ",ret)
            time.sleep(4)
            ret = sport_client.WalkUpright(False)
            print("ret: ",ret)
        elif test_option.id == 18:
            ret = sport_client.CrossStep(True)
            print("ret: ",ret)
            time.sleep(4)
            ret = sport_client.CrossStep(False)
            print("ret: ",ret)
        elif test_option.id == 19:
            ret = sport_client.FreeJump(True)
            print("ret: ",ret)
            time.sleep(4)
            ret = sport_client.FreeJump(False)
            print("ret: ",ret)

        time.sleep(1)
