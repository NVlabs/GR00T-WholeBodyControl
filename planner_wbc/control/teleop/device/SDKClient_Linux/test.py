import time

import ManusServer

ManusServer.init()

for i in range(10):
    time.sleep(1)
    output = ManusServer.get_latest_state()
    # print(output.keys())
    # if(output['3762867141_angle']!=[] and output['3822396207_angle']!=[]):
    #     print(output['3762867141_angle'])

ManusServer.shutdown()
