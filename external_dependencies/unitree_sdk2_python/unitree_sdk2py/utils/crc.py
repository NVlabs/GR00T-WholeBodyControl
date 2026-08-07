import struct
import cyclonedds
import cyclonedds.idl as idl

from .singleton import Singleton
from ..idl.unitree_go.msg.dds_ import LowCmd_
from ..idl.unitree_go.msg.dds_ import LowState_

from ..idl.unitree_hg.msg.dds_ import LowCmd_ as HGLowCmd_
from ..idl.unitree_hg.msg.dds_ import LowState_ as HGLowState_
import ctypes
import os
import platform

class CRC(Singleton):
    def __init__(self):
        #4 bytes aligned, little-endian format.
        #size 812
        self.__packFmtLowCmd = '<4B4IH2x' + 'B3x5f3I' * 20 + '4B' + '55Bx2I'
        #size 1180
        self.__packFmtLowState = '<4B4IH2x' + '13fb3x' + 'B3x7fb3x3I' * 20 + '4BiH4b15H' + '8hI41B3xf2b2x2f4h2I'
        #size 1004
        self.__packFmtHGLowCmd = '<2B2x' + 'B3x5fI' * 35 + '5I'
        #size 2092
        self.__packFmtHGLowState = '<2I2B2xI' + '13fh2x' + 'B3x4f2hf7I' * 35 + '40B5I'

        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.platform = platform.system()
        if self.platform == "Linux":
            if platform.machine()=="x86_64":
                self.crc_lib = ctypes.CDLL(script_dir + '/lib/crc_amd64.so')
            elif platform.machine()=="aarch64":
                self.crc_lib = ctypes.CDLL(script_dir + '/lib/crc_aarch64.so')

            self.crc_lib.crc32_core.argtypes = (ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32)
            self.crc_lib.crc32_core.restype = ctypes.c_uint32
    
    def Crc(self, msg: idl.IdlStruct):
        if msg.__idl_typename__ == 'unitree_go.msg.dds_.LowCmd_':
            return self.__Crc32(self.__PackLowCmd(msg))
        elif msg.__idl_typename__ == 'unitree_go.msg.dds_.LowState_':
            return self.__Crc32(self.__PackLowState(msg))
        if msg.__idl_typename__ == 'unitree_hg.msg.dds_.LowCmd_':
            return self.__Crc32(self.__PackHGLowCmd(msg))
        elif msg.__idl_typename__ == 'unitree_hg.msg.dds_.LowState_':
            return self.__Crc32(self.__PackHGLowState(msg))
        else:
            raise TypeError('unknown IDL message type to crc')

    def __PackLowCmd(self, cmd: LowCmd_):
        origData = []
        origData.extend(cmd.head)
        origData.append(cmd.level_flag)
        origData.append(cmd.frame_reserve)
        origData.extend(cmd.sn)
        origData.extend(cmd.version)
        origData.append(cmd.bandwidth)

        for i in range(20):
            origData.append(cmd.motor_cmd[i].mode)
            origData.append(cmd.motor_cmd[i].q)
            origData.append(cmd.motor_cmd[i].dq)
            origData.append(cmd.motor_cmd[i].tau)
            origData.append(cmd.motor_cmd[i].kp)
            origData.append(cmd.motor_cmd[i].kd)
            origData.extend(cmd.motor_cmd[i].reserve)

        origData.append(cmd.bms_cmd.off)
        origData.extend(cmd.bms_cmd.reserve)

        origData.extend(cmd.wireless_remote)
        origData.extend(cmd.led)
        origData.extend(cmd.fan)
        origData.append(cmd.gpio)
        origData.append(cmd.reserve)
        origData.append(cmd.crc)

        return self.__Trans(struct.pack(self.__packFmtLowCmd, *origData))

    def __PackLowState(self, state: LowState_):
        origData = []
        origData.extend(state.head)
        origData.append(state.level_flag)
        origData.append(state.frame_reserve)
        origData.extend(state.sn)
        origData.extend(state.version)
        origData.append(state.bandwidth)
        
        origData.extend(state.imu_state.quaternion)
        origData.extend(state.imu_state.gyroscope)
        origData.extend(state.imu_state.accelerometer)
        origData.extend(state.imu_state.rpy)
        origData.append(state.imu_state.temperature)
        
        for i in range(20):
            origData.append(state.motor_state[i].mode)
            origData.append(state.motor_state[i].q)
            origData.append(state.motor_state[i].dq)
            origData.append(state.motor_state[i].ddq)
            origData.append(state.motor_state[i].tau_est)
            origData.append(state.motor_state[i].q_raw)
            origData.append(state.motor_state[i].dq_raw)
            origData.append(state.motor_state[i].ddq_raw)
            origData.append(state.motor_state[i].temperature)
            origData.append(state.motor_state[i].lost)
            origData.extend(state.motor_state[i].reserve)

        origData.append(state.bms_state.version_high)
        origData.append(state.bms_state.version_low)
        origData.append(state.bms_state.status)
        origData.append(state.bms_state.soc)
        origData.append(state.bms_state.current)
        origData.append(state.bms_state.cycle)
        origData.extend(state.bms_state.bq_ntc)
        origData.extend(state.bms_state.mcu_ntc)
        origData.extend(state.bms_state.cell_vol)
        
        origData.extend(state.foot_force)
        origData.extend(state.foot_force_est)
        origData.append(state.tick)
        origData.extend(state.wireless_remote)
        origData.append(state.bit_flag)
        origData.append(state.adc_reel)
        origData.append(state.temperature_ntc1)
        origData.append(state.temperature_ntc2)
        origData.append(state.power_v)
        origData.append(state.power_a)
        origData.extend(state.fan_frequency)
        origData.append(state.reserve)
        origData.append(state.crc)

        return self.__Trans(struct.pack(self.__packFmtLowState, *origData))

    def __PackHGLowCmd(self, cmd: HGLowCmd_):
        origData = []
        origData.append(cmd.mode_pr)
        origData.append(cmd.mode_machine)

        for i in range(35):
            origData.append(cmd.motor_cmd[i].mode)
            origData.append(cmd.motor_cmd[i].q)
            origData.append(cmd.motor_cmd[i].dq)
            origData.append(cmd.motor_cmd[i].tau)
            origData.append(cmd.motor_cmd[i].kp)
            origData.append(cmd.motor_cmd[i].kd)
            origData.append(cmd.motor_cmd[i].reserve)

        origData.extend(cmd.reserve)
        origData.append(cmd.crc)

        return self.__Trans(struct.pack(self.__packFmtHGLowCmd, *origData))

    def __PackHGLowState(self, state: HGLowState_):
        origData = []
        origData.extend(state.version)
        origData.append(state.mode_pr)
        origData.append(state.mode_machine)
        origData.append(state.tick)
        
        origData.extend(state.imu_state.quaternion)
        origData.extend(state.imu_state.gyroscope)
        origData.extend(state.imu_state.accelerometer)
        origData.extend(state.imu_state.rpy)
        origData.append(state.imu_state.temperature)
        
        for i in range(35):
            origData.append(state.motor_state[i].mode)
            origData.append(state.motor_state[i].q)
            origData.append(state.motor_state[i].dq)
            origData.append(state.motor_state[i].ddq)
            origData.append(state.motor_state[i].tau_est)
            origData.extend(state.motor_state[i].temperature)
            origData.append(state.motor_state[i].vol)
            origData.extend(state.motor_state[i].sensor)
            origData.append(state.motor_state[i].motorstate)
            origData.extend(state.motor_state[i].reserve)

        origData.extend(state.wireless_remote)
        origData.extend(state.reserve)
        origData.append(state.crc)

        return self.__Trans(struct.pack(self.__packFmtHGLowState, *origData))

    def __Trans(self, packData):
        calcData = []
        calcLen = ((len(packData)>>2)-1)

        for i in range(calcLen):
            d = ((packData[i*4+3] << 24) | (packData[i*4+2] << 16) | (packData[i*4+1] << 8) | (packData[i*4]))
            calcData.append(d)

        return calcData

    def _crc_py(self, data):
        bit = 0
        crc = 0xFFFFFFFF
        polynomial = 0x04c11db7

        for i in range(len(data)):
            bit = 1 << 31
            current = data[i]

            for b in range(32):
                if crc & 0x80000000:
                    crc = (crc << 1) & 0xFFFFFFFF
                    crc ^= polynomial
                else:
                    crc = (crc << 1) & 0xFFFFFFFF

                if current & bit:
                    crc ^= polynomial

                bit >>= 1
        
        return crc

    def _crc_ctypes(self, data):
        uint32_array = (ctypes.c_uint32 * len(data))(*data)
        length = len(data)
        crc=self.crc_lib.crc32_core(uint32_array, length)
        return crc

    def __Crc32(self, data):
        if self.platform == "Linux":
            return self._crc_ctypes(data)
        else:
            return self._crc_py(data)
