//Saver class for storing sensor data to various log files
#include "SensorDataSaver.h"
#include <QDebug>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QDir>

SensorDataSaver::SensorDataSaver(QObject *parent)
    : QObject{parent}
{}

void SensorDataSaver::startLog(QString folderName)
{
    qDebug()<<"create folder"<<folderName<<Qt::endl;
    m_folderName = folderName;
    QDir().mkpath(m_folderName);
    m_headLog.open(m_folderName.toStdString()+"/"+"head.txt");
    m_handLog.open(m_folderName.toStdString()+"/"+"hand.txt");
    m_controllerLog.open(m_folderName.toStdString()+"/"+"controller.txt");
    m_bodyLog.open(m_folderName.toStdString()+"/"+"body.csv");
    m_bodyLog<<"localTimeStamp,remoteTimeStamp,";
    for(int i=0;i<24;++i)
    {
        m_bodyLog<<"bodyjoints["<<i<<"]-pos-x,";
        m_bodyLog<<"bodyjoints["<<i<<"]-pos-y,";
        m_bodyLog<<"bodyjoints["<<i<<"]-pos-z,";
        m_bodyLog<<"bodyjoints["<<i<<"]-rotate-x,";
        m_bodyLog<<"bodyjoints["<<i<<"]-rotate-y,";
        m_bodyLog<<"bodyjoints["<<i<<"]-rotate-z,";
        m_bodyLog<<"bodyjoints["<<i<<"]-rotate-w";
        m_bodyLog<<(i==23?"\n":",");
    }
    m_motionLog.open(m_folderName.toStdString()+"/"+"motion.txt");
}

void SensorDataSaver::stopLog()
{
    qDebug()<<"close log"<<Qt::endl;
    m_headLog.close();
    m_handLog.close();
    m_controllerLog.close();
    m_bodyLog.close();
    m_motionLog.close();
}


void SensorDataSaver::saveData(qint64 tmStamp, QString sensorData)
{
    QJsonDocument jsonDoc = QJsonDocument::fromJson(sensorData.toUtf8());
    if (jsonDoc.isObject()) {
        QJsonObject jsonObj = jsonDoc.object();

        auto tmStampRemote = jsonObj.value("timeStampNs").toInteger();
 
        if (jsonObj.contains("Head") && m_headLog.is_open()) {
            QJsonValue headValue = jsonObj.value("Head");
            QJsonObject headObj = headValue.toObject();
            m_headLog << QJsonDocument(headObj).toJson().constData() << std::endl;
        }

        if (jsonObj.contains("Hand") && m_handLog.is_open()) {
            QJsonValue handValue = jsonObj.value("Hand");
            QJsonObject handObj = handValue.toObject();
            m_handLog << QJsonDocument(handObj).toJson().constData() << std::endl;
        }

        if (jsonObj.contains("Controller") && m_controllerLog.is_open()) {
            QJsonValue controllerValue = jsonObj.value("Controller");
            QJsonObject controllerObj = controllerValue.toObject();
            m_controllerLog << QJsonDocument(controllerObj).toJson().constData() << std::endl;
        }

        if (jsonObj.contains("Body") && m_bodyLog.is_open())
        {
            m_bodyLog << tmStamp << "," << tmStampRemote << ",";
            QJsonArray jointsArray = jsonObj.value("Body").toObject().value("joints").toArray();
            QStringList pValues;
            for (const QJsonValue &jointValue : jointsArray)
            {
                QJsonObject jointObj = jointValue.toObject();
                QString pValue = jointObj["p"].toString();
                pValues.append(pValue);
            }
            QString pValuesString = pValues.join(",");
            m_bodyLog << pValuesString.toUtf8().constData() << std::endl;
        }

        if (jsonObj.contains("Motion") && m_motionLog.is_open()) {
            QJsonValue motionValue = jsonObj.value("Motion");
            QJsonObject motionObj = motionValue.toObject();
            m_motionLog << QJsonDocument(motionObj).toJson().constData() << std::endl;
        }
    }
}
