import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient, PathPoint, SPORT_PATH_POINT_SIZE

if __name__ == "__main__":
    ChannelFactoryInitialize(0, "enp2s0")
    client = SportClient()
    client.SetTimeout(10.0)
    client.Init()

    print("##################GetServerApiVersion###################")
    code, serverAPiVersion = client.GetServerApiVersion()
    if code != 0:
        print("get server api error. code:", code)
    else:
        print("get server api version:", serverAPiVersion)

    if serverAPiVersion != client.GetApiVersion():
        print("api version not equal.")

    time.sleep(3)

    print("##################Trigger###################")
    code = client.Trigger()
    if code != 0:
        print("sport trigger error. code:", code)
    else:
        print("sport trigger success.")

    time.sleep(3)

    while True:
        print("##################RecoveryStand###################")
        code = client.RecoveryStand()
        
        if code != 0:
            print("sport recovery stand error. code:", code)
        else:
            print("sport recovery stand success.")

        time.sleep(3)

        print("##################StandDown###################")
        code = client.StandDown()
        if code != 0:
            print("sport stand down error. code:", code)
        else:
            print("sport stand down success.")

        time.sleep(3)

        print("##################Damp###################")
        code = client.Damp()
        if code != 0:
            print("sport damp error. code:", code)
        else:
            print("sport damp down success.")

        time.sleep(3)

        print("##################RecoveryStand###################")
        code = client.RecoveryStand()
        
        if code != 0:
            print("sport recovery stand error. code:", code)
        else:
            print("sport recovery stand success.")

        time.sleep(3)

        print("##################Sit###################")
        code = client.Sit()
        if code != 0:
            print("sport stand down error. code:", code)
        else:
            print("sport stand down success.")

        time.sleep(3)
        
        print("##################RiseSit###################")
        code = client.RiseSit()
        
        if code != 0:
            print("sport rise sit error. code:", code)
        else:
            print("sport rise sit success.")

        time.sleep(3)

        print("##################SetBodyHight###################")
        code = client.BodyHeight(0.18)
        
        if code != 0:
            print("sport body hight error. code:", code)
        else:
            print("sport body hight success.")

        time.sleep(3)

        print("##################GetState#################")
        keys = ["state", "bodyHeight", "footRaiseHeight", "speedLevel", "gait"]
        code, data = client.GetState(keys)
        
        if code != 0:
            print("sport get state error. code:", code)
        else:
            print("sport get state success. data:", data)

        time.sleep(3)
