# unitree_sdk2_python
unitree_sdk2 python 接口

# 安装
## 依赖
- python>=3.8
- cyclonedds==0.10.2
- numpy
- opencv-python

## 安装 unitree_sdk2_python
在终端中执行：
```bash
cd ~
sudo apt install python3-pip
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip3 install -e .
```
## FAQ
##### 1. `pip3 install -e .` 遇到报错
```bash
Could not locate cyclonedds. Try to set CYCLONEDDS_HOME or CMAKE_PREFIX_PATH
```
该错误提示找不到 cyclonedds 路径。首先编译安装cyclonedds：
```bash
cd ~
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x 
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install
```
进入 unitree_sdk2_python 目录，设置 `CYCLONEDDS_HOME` 为刚刚编译好的 cyclonedds 所在路径，再安装 unitree_sdk2_python
```bash
cd ~/unitree_sdk2_python
export CYCLONEDDS_HOME="~/cyclonedds/install"
pip3 install -e .
```

详细见：
https://pypi.org/project/cyclonedds/#installing-with-pre-built-binaries

# 使用
python sdk2 接口与 unitree_skd2的接口保持一致，通过请求响应或订阅发布topic实现机器人的状态获取和控制。相应的例程位于`/example`目录下。在运行例程前，需要根据文档 https://support.unitree.com/home/zh/developer/Quick_start 配置好机器人的网络连接。
## DDS通讯
在终端中执行：
```bash
python3 ./example/helloworld/publisher.py
```
打开新的终端，执行：
```bash
python3 ./example/helloworld/subscriber.py
```
可以看到终端输出的数据信息。`publisher.py` 和 `subscriber.py` 传输的数据定义在 `user_data.py` 中，用户可以根据需要自行定义需要传输的数据结构。

## 高层状态和控制
高层接口的数据结构和控制方式与unitree_sdk2一致。具体可见：https://support.unitree.com/home/zh/developer/sports_services
### 高层状态
终端中执行：
```bash
python3 ./example/high_level/read_highstate.py enp2s0
```
其中 `enp2s0` 为机器人所连接的网卡名称，请根据实际情况修改。
### 高层控制
终端中执行：
```bash
python3 ./example/high_level/sportmode_test.py enp2s0
```
其中 `enp2s0` 为机器人所连接的网卡名称，请根据实际情况修改。
该例程提供了几种测试方法，可根据测试需要选择:
```python
test.StandUpDown() # 站立趴下
# test.VelocityMove() # 速度控制
# test.BalanceAttitude() # 姿态控制
# test.TrajectoryFollow() # 轨迹跟踪
# test.SpecialMotions() # 特殊动作

```
## 底层状态和控制
底层接口的数据结构和控制方式与unitree_sdk2一致。具体可见：https://support.unitree.com/home/zh/developer/Basic_services
### 底层状态
终端中执行：
```bash
python3 ./example/low_level/lowlevel_control.py enp2s0
```
其中 `enp2s0` 为机器人所连接的网卡名称，请根据实际情况修改。程序会输出右前腿hip关节的状态、IMU和电池电压信息。

### 底层电机控制
首先使用 app 关闭高层运动服务(sport_mode)，否则会导致指令冲突。
终端中执行：
```bash
python3 ./example/low_level/lowlevel_control.py enp2s0
```
其中 `enp2s0` 为机器人所连接的网卡名称，请根据实际情况修改。左后腿 hip 关节会保持在0角度 (安全起见，这里设置 kp=10, kd=1)，左后腿 calf 关节将持续输出 1Nm 的转矩。

## 遥控器状态获取
终端中执行：
```bash
python3 ./example/wireless_controller/wireless_controller.py enp2s0
```
其中 `enp2s0` 为机器人所连接的网卡名称，请根据实际情况修改。
终端将输出每一个按键的状态。对于遥控器按键的定义和数据结构可见： https://support.unitree.com/home/zh/developer/Get_remote_control_status

## 前置摄像头
使用opencv获取前置摄像头(确保在有图形界面的系统下运行, 按 ESC 退出程序): 
```bash
python3 ./example/front_camera/camera_opencv.py enp2s0
```
其中 `enp2s0` 为机器人所连接的网卡名称，请根据实际情况修改。

## 避障开关
```bash
python3 ./example/obstacles_avoid_switch/obstacles_avoid_switch.py enp2s0
```
其中 `enp2s0` 为机器人所连接的网卡名称，请根据实际情况修改。机器人将循环开启和关闭避障功能。关于避障服务，详细见 https://support.unitree.com/home/zh/developer/ObstaclesAvoidClient

## 灯光音量控制
```bash
python3 ./example/vui_client/vui_client_example.py enp2s0
```
其中 `enp2s0` 为机器人所连接的网卡名称，请根据实际情况修改。机器人将循环调节音量和灯光亮度。该接口详细见 https://support.unitree.com/home/zh/developer/VuiClient
