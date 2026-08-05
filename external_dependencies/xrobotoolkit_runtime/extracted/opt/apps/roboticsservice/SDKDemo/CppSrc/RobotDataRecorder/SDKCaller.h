//SDK caller class for handling robot SDK interactions and data recording
#ifndef SDKCALLER_H
#define SDKCALLER_H

#include<QDebug>
#include <QObject>
#include <QCoreApplication>
#include <PXREARobotSDK.h>
#include <QTimer>
#include <QJsonDocument>
#include <QJsonObject>
#include "SensorDataHandler.h"

void OnPXREAClientCallback(void* context,PXREAClientCallbackType type,int status,void* userData);

class SDKCaller : public QObject
{
    Q_OBJECT
    Q_PROPERTY(int recordCount READ getRecordCount WRITE setRecordCount NOTIFY recordCountChanged)
    Q_PROPERTY(QString recordState READ getRecordState WRITE setRecordState NOTIFY recordStateChanged)

public:
    explicit SDKCaller(QObject *parent = nullptr)
    {
        Q_UNUSED(parent)
        m_recordCount = 0;
        m_recordState = "未开始";
    }

    int getRecordCount() const
    {
        return m_recordCount;
    }

    void setRecordCount(int value)
    {
        if (m_recordCount != value) {
            m_recordCount = value;
            emit recordCountChanged();
        }
    }

    QString getRecordState() const
    {
        return m_recordState;
    }

    void setRecordState(const QString& value)
    {
        if (m_recordState != value) {
            m_recordState = value;
            emit recordStateChanged();
        }
    }
    void incrementRecordCount()
    {
        m_recordCount++;
        emit recordCountChanged();
    }

signals:
    void deviceFind(const QString& dev);
    void deviceMiss(const QString& dev);
    void disConnect();
    void headMessageChanged(QString);
    void handlerMessageChanged(QString);
    void handMessageChanged(QString);
    void recordCountChanged();
    void recordStateChanged();



private:
    SensorDataHandler m_sensorDataHandler;
    int m_recordCount;
    QString m_recordState;
    QString m_recordFolderName;

public:
    void logSensorData(QString senData)
    {
        m_sensorDataHandler.logData(senData);
    }

public slots:
    void testInit()
    {
        PXREAInit(this,OnPXREAClientCallback,PXREAFullMask);
    }
    void testDeinit()
    {
        PXREADeinit();
    }
    void testDeviceControlJson(const QString& devID,const QString& parameterJson)
    {
        PXREADeviceControlJson(devID.toUtf8().constData(),parameterJson.toUtf8().constData());
    }
    void startRecord()
    {
        QString folderName = QDateTime::currentDateTime().toString("yyyy-MM-dd_HH-mm-ss");
        m_sensorDataHandler.startLog(folderName);
        m_recordFolderName = folderName;
        setRecordState(m_recordFolderName + " 正在记录");
    }
    void stopRecord()
    {
        m_sensorDataHandler.stopLog();
        setRecordState(m_recordFolderName + " 已保存");
    }
};

inline void OnPXREAClientCallback(void* context,PXREAClientCallbackType type,int status,void* userData)
{
    auto& sdkCaller = (*(SDKCaller*)context);
    switch (type)
    {
    case PXREAServerConnect:
        qDebug() <<"server connect"  << Qt::endl;
        break;
    case PXREAServerDisconnect:
        qDebug() <<"server disconnect"  << Qt::endl;
        sdkCaller.disConnect();
        break;
    case PXREADeviceFind:
        qDebug() <<"device find"<<(const char*)userData<< Qt::endl;
        sdkCaller.deviceFind((const char*)userData);
        break;
    case PXREADeviceMissing:
        qDebug() <<"device missing"<<(const char*)userData<< Qt::endl;
        sdkCaller.deviceMiss((const char*)userData);
        break;
    case PXREADeviceConnect:
    {
        qDebug() <<"device connect"<<(const char*)userData<<status<< Qt::endl;
    }
    break;
    case PXREADeviceStateJson:
    {
        auto& dsj = *((PXREADevStateJson*)userData);
        QJsonDocument jsonDoc = QJsonDocument::fromJson(dsj.stateJson);
        QJsonObject jsonObj = jsonDoc.object();
        if(jsonObj["functionName"] == "Tracking")
        {
            sdkCaller.logSensorData(jsonObj.value("value").toString().replace("\\",""));
            sdkCaller.incrementRecordCount();
        }
        else
        {
            qDebug() <<"device state"<<dsj.devID<<dsj.stateJson<< Qt::endl;
        }
    }
    break;
    }
}

#endif // SDKCALLER_H
