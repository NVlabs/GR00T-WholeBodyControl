import QtQuick
import QtQuick.Window
import QtQuick.Controls
import pico.qmlcomponents 1.0
import QtMultimedia
Window {
    visible: true
    width: 800
    height: 200
    x: 0
    y: Screen.height / 2 - height / 2
    title: "数据录制"
    SDKCaller{
        id:sdkCaller
    }
    Component.onCompleted:
    {
        sdkCaller.testInit()
    }
    Component.onDestruction:
    {
        sdkCaller.testDeinit()
    }
    Connections{
        target: sdkCaller
        function getSN(dev)
        {
            for (var i=0; i < devModel.count; ++i)
            {
                if(devModel.get(i).name === dev)
                {
                    return i
                }
            }
            return -1
        }
        function onDeviceFind(dev){
            if(getSN(dev)<0)
            {
                devModel.append({"name": dev})
            }
        }
        function onDeviceMiss(dev){
            var i = getSN(dev)
            if(i>=0)
            {
                devModel.remove(i)
            }
        }
        function onDisConnect(){
            console.log("clear device sn")
            devModel.clear();
            devid.text = ""
        }
    }
    ListModel {
        id: devModel
    }
    Row{
        width: parent.width - 40
        height: parent.height - 40
        anchors.centerIn: parent
        spacing: 10
        Column{
            spacing: 5
            width: parent.width/2 - 5
            height: parent.height
            Rectangle{
                width: parent.width
                height: 30
                border.width: 1
                Row {
                    anchors.fill: parent
                    Repeater {
                        model: devModel
                        TextField {
                            selectByMouse: true
                            width: 100
                            height: 40
                            text: name
                            verticalAlignment: Text.AlignVCenter
                            onFocusChanged: {
                                if(focus)
                                {
                                    devid.text = text
                                }
                            }
                        }
                    }
                }
            }
            TextField{
                id:devid
                height: 30
                selectByMouse: true
                width: parent.width
                placeholderText: "请选择一台设备跟踪"
            }

            Text
            {
                text: "数据计数 " + sdkCaller.recordCount
            }
            Text
            {
                text: "录制状态 " + sdkCaller.recordState
            }
            Row
            {
                spacing: 5
                Button
                {
                    text: "开始"
                    onClicked: sdkCaller.startRecord()
                }

                Button
                {
                    text: "停止"
                    onClicked: sdkCaller.stopRecord()
                }

            }
        }
    }

}
