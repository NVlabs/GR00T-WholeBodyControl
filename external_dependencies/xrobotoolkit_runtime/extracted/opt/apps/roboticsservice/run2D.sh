#!/bin/bash
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DIR:$DIR/lib:$DIR/SDK/x64
export QT_PLUGIN_PATH=$DIR/plugins/:$QT_PLUGIN_PATH
export QT_QML_PATH=$DIR/qml/:$QT_QML_PATH
echo $LD_LIBRARY_PATH
echo $QT_PLUGIN_PATH
echo $QT_QML_PATH
cd $DIR
./RoboticsServiceProcess &
./RobotDemoQt

