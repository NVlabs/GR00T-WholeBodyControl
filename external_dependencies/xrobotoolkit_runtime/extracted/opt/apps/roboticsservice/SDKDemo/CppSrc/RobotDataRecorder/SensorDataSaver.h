//Saver class for writing sensor data to various log files
#ifndef SENSORDATASAVER_H
#define SENSORDATASAVER_H

#include <QObject>
#include <fstream>
class SensorDataSaver : public QObject
{
    Q_OBJECT
public:
    explicit SensorDataSaver(QObject *parent = nullptr);

public slots:
    void startLog(QString folderName);
    void stopLog();
    void saveData(qint64 tmStamp,QString sensorData);
private:
    QString m_folderName;
    std::ofstream m_headLog;
    std::ofstream m_handLog;
    std::ofstream m_controllerLog;
    std::ofstream m_bodyLog;
    std::ofstream m_motionLog;
};

#endif // SENSORDATASAVER_H
