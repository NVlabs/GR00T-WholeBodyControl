import json, time
import xrobotoolkit_sdk as xrt
from datetime import datetime

xrt.init()  # 建链、启动心跳与服务端反馈流

dev_id = "TestDevice"  # your device ID in unity app

# send JSON

for i in range(3):
	print("start sending")
	start = int(time.time() * 1e3)
	cmd = {"functionName": "set_robot", "value": {"mode": "teach"}, "timestamp_ns": start}
	jsonfile= json.dumps(cmd)
	jsontime = int(time.time()*1e3)
	xrt.device_control_json(dev_id,jsonfile)
	end = int(time.time()*1e3)
	
	print("json process time", jsontime - start)

	print("current time difference ", end-start)
# send bytes
print("send_bytes_to_device rc =", xrt.send_bytes_to_device(dev_id, b"\xAA\x55\x10\x00"))





