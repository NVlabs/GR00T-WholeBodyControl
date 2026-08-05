//Handler for processing and managing sensor data from the robot
#include <QDateTime>
#include <QString>
#include "SensorDataHandler.h"
#include "SensorDataSaver.h"

SensorDataHandler::SensorDataHandler(QObject *parent)
    : QObject{parent}
{
    auto saver = new SensorDataSaver();
    saver->moveToThread(&m_saveThread);
    connect(this,&SensorDataHandler::startLog,saver,&SensorDataSaver::startLog);
    connect(this,&SensorDataHandler::stopLog,saver,&SensorDataSaver::stopLog);
    connect(this,&SensorDataHandler::saveData,saver,&SensorDataSaver::saveData);
    connect(&m_saveThread,&QThread::finished,saver,&QObject::deleteLater);
    m_saveThread.start();

}

SensorDataHandler::~SensorDataHandler()
{
    m_saveThread.quit();
    m_saveThread.wait();
}



void SensorDataHandler::logData(QString sensorData)
{
    qint64 nowpt = QDateTime::currentDateTime().toMSecsSinceEpoch();
    emit saveData(nowpt,sensorData);
}

