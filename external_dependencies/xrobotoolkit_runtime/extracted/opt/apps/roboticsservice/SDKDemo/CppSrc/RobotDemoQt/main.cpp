//Main entry point for the robot demo application
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include "SDKCaller.h"


int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);

    QQmlApplicationEngine engine;
    const QUrl url(u"qrc:/RobotDemo/main.qml"_qs);
    qDebug() << "robot demo..." <<Qt::endl;
    QObject::connect(
        &engine,
        &QQmlApplicationEngine::objectCreationFailed,
        &app,
        []() { QCoreApplication::exit(-1); },
        Qt::QueuedConnection);
    qmlRegisterType<SDKCaller>("pico.qmlcomponents", 1, 0, "SDKCaller");
    engine.load(url);

    return app.exec();
}
