from .builtin_interfaces.msg.dds_ import *
from .std_msgs.msg.dds_ import *
from .geometry_msgs.msg.dds_ import *
from .nav_msgs.msg.dds_ import *
from .sensor_msgs.msg.dds_ import *
from .unitree_go.msg.dds_ import *
from .unitree_api.msg.dds_ import *

# IDL for unitree_hg
from .unitree_hg.msg.dds_ import LowCmd_ as HGLowCmd_
from .unitree_hg.msg.dds_ import LowState_ as HGLowState_
from .unitree_hg.msg.dds_ import MotorCmd_ as HGMotorCmd_
from .unitree_hg.msg.dds_ import MotorState_ as HGMotorState_
from .unitree_hg.msg.dds_ import BmsState_ as HGBmsState_
from .unitree_hg.msg.dds_ import IMUState_ as HGIMUState_
from .unitree_hg.msg.dds_ import MainBoardState_ as HGMainBoardState_
from .unitree_hg.msg.dds_ import PressSensorState_ as HGPressSensorState_
from .unitree_hg.msg.dds_ import HandCmd_ as HGHandCmd_
from .unitree_hg.msg.dds_ import HandState_ as HGHandState_
from .unitree_hg.msg.dds_ import OdoState_ as HGOdoState_

"""
" builtin_interfaces_msgs.msg.dds_ dafault
"""
def builtin_interfaces_msgs_msg_dds__Time_():
    return Time_(0, 0)


"""
" std_msgs.msg.dds_ dafault
"""
def std_msgs_msg_dds__Header_():
    return Header_(builtin_interfaces_msgs_msg_dds__Time_(), "")

def std_msgs_msg_dds__String_():
    return String_("")


"""
" geometry_msgs.msg.dds_ dafault
"""
def geometry_msgs_msg_dds__Point_():
    return Point_(0.0, 0.0, 0.0)

def geometry_msgs_msg_dds__Point32_():
    return Point32_(0.0, 0.0, 0.0)

def geometry_msgs_msg_dds__PointStamped_():
    return PointStamped_(std_msgs_msg_dds__Header_(), geometry_msgs_msg_dds__Point_())

def geometry_msgs_msg_dds__Quaternion_():
    return Quaternion_(0.0, 0.0, 0.0, 0.0)

def geometry_msgs_msg_dds__Vector3_():
    return Vector3_(0.0, 0.0, 0.0)

def geometry_msgs_msg_dds__Pose_():
    return Pose_(geometry_msgs_msg_dds__Point_(), geometry_msgs_msg_dds__Quaternion_())

def geometry_msgs_msg_dds__Pose2D_():
    return Pose2D_(0.0, 0.0, 0.0)

def geometry_msgs_msg_dds__PoseStamped_():
    return PoseStamped_(std_msgs_msg_dds__Header_(), geometry_msgs_msg_dds__Pose_())

def geometry_msgs_msg_dds__PoseWithCovariance_():
    return PoseWithCovariance_(geometry_msgs_msg_dds__Pose_(), [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ])

def geometry_msgs_msg_dds__PoseWithCovarianceStamped_():
    return PoseWithCovarianceStamped_(std_msgs_msg_dds__Header_(), geometry_msgs_msg_dds__PoseWithCovariance_())

def geometry_msgs_msg_dds__QuaternionStamped_():
    return QuaternionStamped_(std_msgs_msg_dds__Header_(), geometry_msgs_msg_dds__Quaternion_())

def geometry_msgs_msg_dds__Twist_():
    return Twist_(geometry_msgs_msg_dds__Vector3_(), geometry_msgs_msg_dds__Vector3_())

def geometry_msgs_msg_dds__TwistStamped_():
    return TwistStamped_(std_msgs_msg_dds__Header_(), geometry_msgs_msg_dds__Twist_())

def geometry_msgs_msg_dds__TwistWithCovariance_():
    return TwistWithCovariance_(geometry_msgs_msg_dds__Twist_(), [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ])

def geometry_msgs_msg_dds__TwistWithCovarianceStamped_():
    return TwistWithCovarianceStamped_(std_msgs_msg_dds__Header_(), geometry_msgs_msg_dds__TwistWithCovariance_())


"""
" nav_msgs.msg.dds_ dafault
"""
def nav_msgs_msg_dds__MapMetaData_():
    return MapMetaData_(builtin_interfaces_msgs_msg_dds__Time_(), 0, 0, geometry_msgs_msg_dds__Pose_())

def nav_msgs_msg_dds__OccupancyGrid_():
    return OccupancyGrid_(std_msgs_msg_dds__Header_(), nav_msgs_msg_dds__MapMetaData_(), [])

def nav_msgs_msg_dds__Odometry_():
    return Odometry_(std_msgs_msg_dds__Header_(), "", geometry_msgs_msg_dds__PoseWithCovariance_(),
            geometry_msgs_msg_dds__TwistWithCovariance_())


"""
" sensor_msgs.msg.dds_ dafault
"""
def sensor_msgs_msg_dds__PointField_Constants_PointField_():
    return PointField_("", 0, 0, 0)

def sensor_msgs_msg_dds__PointField_Constants_PointCloud2_():
    return PointCloud2_(std_msgs_msg_dds__Header_(), 0, 0, [], False, 0, 0, [], False)


"""
" unitree_go.msg.dds_ dafault
"""
def unitree_go_msg_dds__AudioData_():
    return AudioData_(0, [])

def unitree_go_msg_dds__BmsCmd_():
    return BmsCmd_(0, [0, 0, 0])

def unitree_go_msg_dds__BmsState_():
    return BmsState_(0, 0, 0, 0, 0, 0, [0, 0], [0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

def unitree_go_msg_dds__Error_():
    return Error_(0, 0)

def unitree_go_msg_dds__Go2FrontVideoData_():
    return Go2FrontVideoData_(0, [], [], [])

def unitree_go_msg_dds__HeightMap_():
    return HeightMap_(0.0, "", 0.0, 0, 0, [0.0, 0.0], [])

def unitree_go_msg_dds__IMUState_():
    return IMUState_([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0)

def unitree_go_msg_dds__InterfaceConfig_():
    return InterfaceConfig_(0, 0, [0, 0])

def unitree_go_msg_dds__LidarState_():
    return LidarState_(0.0, "", "", "", 0.0, 0.0, 0, 0.0, 0.0, 0, 0, 0.0, 0.0, [0.0, 0.0, 0.0], 0.0, 0, 0)

def unitree_go_msg_dds__MotorCmd_():
    return MotorCmd_(0, 0.0, 0.0, 0.0, 0.0, 0.0, [0, 0, 0])

def unitree_go_msg_dds__MotorState_():
    return MotorState_(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, [0, 0])

def unitree_go_msg_dds__LowCmd_():
    return LowCmd_([0, 0], 0, 0, [0, 0], [0, 0], 0, [unitree_go_msg_dds__MotorCmd_() for i in range(20)],
                unitree_go_msg_dds__BmsCmd_(),
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0], 0, 0, 0)

def unitree_go_msg_dds__LowState_():
    return LowState_([0, 0], 0, 0, [0, 0], [0, 0], 0, unitree_go_msg_dds__IMUState_(),
                [unitree_go_msg_dds__MotorState_() for i in range(20)],
                unitree_go_msg_dds__BmsState_(), [0, 0, 0, 0], [0, 0, 0, 0], 0,
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                0, 0, 0, 0, 0.0, 0.0, [0, 0, 0, 0], 0, 0)

def unitree_go_msg_dds__Req_():
    return Req_("", "")

def unitree_go_msg_dds__Res_():
    return Res_("", [], "")

def unitree_go_msg_dds__TimeSpec_():
    return TimeSpec_(0, 0)

def unitree_go_msg_dds__PathPoint_():
    return PathPoint_(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

def unitree_go_msg_dds__SportModeState_():
    return SportModeState_(unitree_go_msg_dds__TimeSpec_(), 0, unitree_go_msg_dds__IMUState_(),
                0, 0, 0, 0.0, [0.0, 0.0, 0.0], 0.0,
                [0.0, 0.0, 0.0], 0.0, [0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],[unitree_go_msg_dds__PathPoint_() for i in range(10)])

def unitree_go_msg_dds__UwbState_():
    return UwbState_([0, 0], 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [0.0, 0.0], 0, 0, 0)

def unitree_go_msg_dds__UwbSwitch_():
    return UwbSwitch_(0)

def unitree_go_msg_dds__WirelessController_():
    return WirelessController_(0.0, 0.0, 0.0, 0.0, 0)


"""
" unitree_hg.msg.dds_ dafault
"""
def unitree_hg_msg_dds__BmsCmd_():
    return HGBmsCmd_(0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

def unitree_hg_msg_dds__BmsState_():
    return HGBmsState_(0, 0, 0,
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0], 0, 0, 0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0, 0, [0, 0, 0, 0, 0], [0, 0, 0])

def unitree_hg_msg_dds__IMUState_():
    return HGIMUState_([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0)

def unitree_hg_msg_dds__MotorCmd_():
    return HGMotorCmd_(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

def unitree_hg_msg_dds__MotorState_():
    return HGMotorState_(0, 0.0, 0.0, 0.0, 0.0, [0, 0], 0.0, [0, 0], 0,  [0, 0, 0, 0])

def unitree_hg_msg_dds__MainBoardState_():
    return HGMainBoardState_([0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0, 0, 0])

def unitree_hg_msg_dds__LowCmd_():
    return HGLowCmd_(0, 0, [unitree_hg_msg_dds__MotorCmd_() for i in range(35)], [0, 0, 0, 0], 0)

def unitree_hg_msg_dds__LowState_():
    return HGLowState_([0, 0], 0, 0, 0, unitree_hg_msg_dds__IMUState_(),
                [unitree_hg_msg_dds__MotorState_() for i in range(35)],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0], 0)

def unitree_hg_msg_dds__PressSensorState_():
    return HGPressSensorState_([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0, 0)

def unitree_hg_msg_dds__HandCmd_():
    return HGHandCmd_([unitree_hg_msg_dds__MotorCmd_() for i in range(7)], [0, 0, 0, 0])

def unitree_hg_msg_dds__HandState_():
    return HGHandState_([unitree_hg_msg_dds__MotorState_() for i in range(7)], 
                        [unitree_hg_msg_dds__PressSensorState_() for i in range(7)],
                         unitree_hg_msg_dds__IMUState_(), 
                         0.0, 0.0, 0.0, 0.0, [0, 0], [0, 0])

def unitree_hg_msg_dds__OdoState_():
    return HGOdoState_([0, 0], 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0)


"""
" unitree_api.msg.dds_ dafault
"""
def unitree_api_msg_dds__RequestIdentity_():
    return RequestIdentity_(0, 0)

def unitree_api_msg_dds__RequestLease_():
    return RequestLease_(0, unitree_hg_msg_dds__IMUState_(), [], )

def unitree_api_msg_dds__RequestPolicy_():
    return RequestPolicy_(0, False)

def unitree_api_msg_dds__RequestHeader_():
    return RequestHeader_(unitree_api_msg_dds__RequestIdentity_(), unitree_api_msg_dds__RequestLease_(),
            unitree_api_msg_dds__RequestPolicy_())

def unitree_api_msg_dds__Request_():
    return Request_(unitree_api_msg_dds__RequestHeader_(), "", [])

def unitree_api_msg_dds__ResponseStatus_():
    return ResponseStatus_(0)

def unitree_api_msg_dds__ResponseHeader_():
    return ResponseHeader_(unitree_api_msg_dds__RequestIdentity_(), unitree_api_msg_dds__ResponseStatus_())

def unitree_api_msg_dds__Response_():
    return Response_(unitree_api_msg_dds__ResponseHeader_(), "", [], 0, 0, [0, 0])

