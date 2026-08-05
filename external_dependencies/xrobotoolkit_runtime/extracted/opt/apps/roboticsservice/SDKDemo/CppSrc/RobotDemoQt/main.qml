import QtQuick
import QtQuick.Window
import QtQuick.Controls
import pico.qmlcomponents 1.0
import QtMultimedia
Window {
    visible: true
    width: 800
    height: 600
    x: 0
    y: Screen.height / 2 - height / 2
    title: "PICO机器人助手Demo"
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
        }
    }
    ListModel {
        id: devModel
    }
    Column{
        spacing: 5
        width: parent.width - 40
        height: parent.height - 40
        anchors.centerIn: parent

        Row {
            spacing: 5

            Text {
                width: 60
                anchors.verticalCenter: parent.verticalCenter
                text: "设备列表:"
            }

            Rectangle {
                width: 150
                height: 80
                border.width: 1

                ListView {
                    anchors.fill: parent
                    model: devModel
                    clip: true

                    delegate: ItemDelegate {
                        width: parent.width
                        height: 20

                        Text {
                            anchors.verticalCenter: parent.verticalCenter
                            anchors.left: parent.left
                            anchors.leftMargin: 10
                            text: name
                        }

                        MouseArea {
                            anchors.fill: parent
                            onClicked: {
                                listView.currentIndex = index
                            }
                        }

                        highlighted: ListView.isCurrentItem
                    }

                    ScrollBar.vertical: ScrollBar {
                        active: true
                    }
                }
            }
        }

        Row
        {
            spacing: 5
            Text
            {
                width: 40
                anchors.verticalCenter: parent.verticalCenter
                text: "头:"
            }

            Rectangle {
                width: 500
                height: headText.contentHeight + 10
                border.width: 1
                border.color: "#CCCCCC"
                Text {
                    id: headText
                    width: parent.width - 10
                    anchors.centerIn: parent
                    wrapMode: Text.Wrap
                    text: sdkCaller.headMessage
                }
            }


        }

        Row
        {
            spacing: 5
            Text
            {
                width: 40
                anchors.verticalCenter: parent.verticalCenter
                text: "手柄:"
            }

            Rectangle {
                width: 500
                height: handlerText.contentHeight + 10
                border.width: 1
                border.color: "#CCCCCC"

                Text {
                    id: handlerText
                    anchors.fill: parent
                    anchors.margins: 5
                    wrapMode: Text.Wrap
                    text: sdkCaller.handlerMessage
                    verticalAlignment: Text.AlignVCenter
                }
            }

        }

        Row
        {
            spacing: 5
            Text
            {
                width: 40
                anchors.verticalCenter: parent.verticalCenter
                text: "手势:"
            }

            Rectangle {
                width: 500
                height: 200
                border.width: 1
                border.color: "#CCCCCC"

                Flickable {
                    anchors.fill: parent
                    anchors.margins: 1
                    contentWidth: gestureText.width
                    contentHeight: gestureText.height
                    clip: true

                    ScrollBar.vertical: ScrollBar {
                        policy: ScrollBar.AsNeeded
                    }

                    Text {
                        id: gestureText
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.margins: 2
                        wrapMode: Text.Wrap
                        text: sdkCaller.handMessage
                    }
                }
            }

        }
    }

}
