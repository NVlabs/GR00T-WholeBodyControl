from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.utils.crc import CRC

crc = CRC()

"""
" LowCmd/LowState CRC
"""
cmd = unitree_go_msg_dds__LowCmd_()
cmd.crc = crc.Crc(cmd)

state = unitree_go_msg_dds__LowState_()
state.crc = crc.Crc(state)

print("CRC[LowCmd, LowState]: {}, {}".format(cmd.crc, state.crc))

"""
" LowCmd/LowState for HG CRC. ()
"""
cmd = unitree_hg_msg_dds__LowCmd_()
cmd.crc = crc.Crc(cmd)

state = unitree_hg_msg_dds__LowState_()
state.crc = crc.Crc(state)

print("CRC[HGLowCmd, HGLowState]: {}, {}".format(cmd.crc, state.crc))
