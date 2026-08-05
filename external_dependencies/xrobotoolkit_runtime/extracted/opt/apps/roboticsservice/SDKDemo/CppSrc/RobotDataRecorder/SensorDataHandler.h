//Handler class for processing and managing sensor data
#ifndef SENSORDATAHANDLER_H
#define SENSORDATAHANDLER_H

#include <QObject>
#include <QThread>
#include <QTimer>

class SensorDataHandler : public QObject
{
    Q_OBJECT
public:
    explicit SensorDataHandler(QObject *parent = nullptr);
    ~SensorDataHandler();

signals:
    void startLog(QString foldername);
    void stopLog();
    void saveData(qint64 tmStamp,QString sensorData);
public slots:
    void logData(QString sensorData);
private:
    QThread m_saveThread;
};

#endif // SENSORDATAHANDLER_H
