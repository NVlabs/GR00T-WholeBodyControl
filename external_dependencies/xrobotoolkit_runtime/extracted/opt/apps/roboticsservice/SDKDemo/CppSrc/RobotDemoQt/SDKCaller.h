//SDK interface wrapper for robot control and data communication
#ifndef SDKCALLER_H
#define SDKCALLER_H

#include<QDebug>
#include <QObject>
#include <QCoreApplication>
#include <PXREARobotSDK.h>
#include <QTimer>
#include <QJsonDocument>
#include <QJsonObject>


#ifdef _WIN32
#define MY_PIPE R"(\\.\pipe\mypipe)"
#elif __linux__
#define MY_PIPE "mypipe"
#endif

void OnPXREAClientCallback(void* context,PXREAClientCallbackType type,int status,void* userData);

class SDKCaller : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QString headMessage READ getHeadMessage WRITE setHeadMessage NOTIFY headMessageChanged)
    Q_PROPERTY(QString handlerMessage READ getHandlerMessage WRITE setHandlerMessage NOTIFY handlerMessageChanged)
    Q_PROPERTY(QString handMessage READ getHandMessage WRITE setHandMessage NOTIFY handMessageChanged)

public:
    explicit SDKCaller(QObject *parent = nullptr)
    {
        Q_UNUSED(parent)
    }

    void setHeadMessage(QString str)
    {
        if(m_headMsg != str)
        {
            m_headMsg = str;
            emit headMessageChanged(str);
        }
    }

    QString getHeadMessage() const
    {
        return m_headMsg;
    }

    void setHandlerMessage(QString str)
    {
        if(m_handlerMsg != str)
        {
            m_handlerMsg = str;
            emit handlerMessageChanged(str);
        }
    }

    QString getHandlerMessage() const
    {
        return m_handlerMsg;
    }

    void setHandMessage(QString str)
    {
        if(m_handMsg != str)
        {
            m_handMsg = str;
            emit handMessageChanged(str);
        }
    }

    QString getHandMessage() const
    {
        return m_handMsg;
    }

signals:
    void deviceFind(const QString& dev);
    void deviceMiss(const QString& dev);
    void disConnect();
    void headMessageChanged(QString);
    void handlerMessageChanged(QString);
    void handMessageChanged(QString);

private:
    QString m_headMsg;
    QString m_handlerMsg;
    QString m_handMsg;

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
        // qDebug() <<"device state"<<dsj.devID<<dsj.stateJson<< Qt::endl;
        QJsonDocument jsonDoc = QJsonDocument::fromJson(dsj.stateJson);
        QJsonObject jsonObj = jsonDoc.object();
        if(jsonObj["functionName"] == "Tracking")
        {
            QJsonDocument valueDoc = QJsonDocument::fromJson(jsonObj["value"].toString().toUtf8());
            QJsonObject valueObj = valueDoc.object();
            if(valueObj.contains("Head")) {
                QJsonDocument headDoc(valueObj["Head"].toObject());
                sdkCaller.setHeadMessage(headDoc.toJson(QJsonDocument::Compact));
            }
            if(valueObj.contains("Controller")) {
                QJsonDocument controllerDoc(valueObj["Controller"].toObject());
                sdkCaller.setHandlerMessage(controllerDoc.toJson(QJsonDocument::Compact));
            }
            if(valueObj.contains("Hand")) {
                QJsonDocument handDoc(valueObj["Hand"].toObject());
                sdkCaller.setHandMessage(handDoc.toJson(QJsonDocument::Compact));
            }
        }
    }
    break;
    }
}

#endif // SDKCALLER_H
