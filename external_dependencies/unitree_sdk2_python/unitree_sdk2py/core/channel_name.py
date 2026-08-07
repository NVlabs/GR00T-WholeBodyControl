from enum import Enum

"""
" Enum ChannelType
"""
class ChannelType(Enum):
    SEND = 0
    RECV = 1

"""
" function GetClientChannelName
"""
def GetClientChannelName(serviceName: str, channelType: ChannelType):
    name = "rt/api/" + serviceName
    
    if channelType == ChannelType.SEND:
        name += "/request"
    else:
        name += "/response"

    return name

"""
" function GetClientChannelName
"""
def GetServerChannelName(serviceName: str, channelType: ChannelType):
    name = "rt/api/" + serviceName
    
    if channelType == ChannelType.SEND:
        name += "/response"
    else:
        name += "/request"

    return name