//Video frame provider for handling and displaying video frames in QML
#ifndef VIDEOFRAMEPROVIDER_H
#define VIDEOFRAMEPROVIDER_H

#include <QObject>
#include <QDebug>
#include <fstream>
#include <QVideoFrame>
#include <QVideoSink>


//main.cpp qmlRegisterType<VideoFrameProvider>(uri, 1, 0, "VideoFrameProvider");
//qml define VideoFrameProvider instance
//define VideoOutput ,source using VideoFrameProvider instance
//pass VideoFrameProvider instance addr to cpp when startup
// invoke VideoFrameProvider presentFrame at runtime

class VideoFrameProvider: public QObject
{
    Q_OBJECT

    Q_PROPERTY(QVideoSink *videoSurface READ videoSurface WRITE setVideoSurface NOTIFY videoSurfaceChanged)

public:
    VideoFrameProvider(QObject *parent = nullptr) : QObject(parent)
    {
        m_width = 0;
        m_height = 0 ;
        m_pixFormat = QVideoFrameFormat::Format_Invalid;
        m_dumpFrame = false;
        m_dumpFileNo = 1;
    }
    ~VideoFrameProvider()
    {
        if(m_dumpFrame && m_dumpFile.is_open())
        {
            m_dumpFile.close();
        }
    }


    QVideoSink *videoSurface()
    {
        return m_surface;
    }

    void dumpFrame(const char* data,unsigned len)
    {
        m_dumpFile.write(data, len);
    }

    void setVideoSurface(QVideoSink *surface)
    {
        if (m_surface != surface)
        {
            m_surface = surface;
            emit videoSurfaceChanged();
            }
        }

    void presentQtFrame(QVideoFrame frame, int width, int height, QVideoFrameFormat::PixelFormat pixFormat)
    {
        if((width == m_width
             && height == m_height
             && pixFormat == m_pixFormat) == false)
        {
            qDebug() << "reset format to"  << width << height << pixFormat << Qt::endl;
            setFormat(width, height, pixFormat);
    }

        if (m_surface)
    {
            m_surface->setVideoFrame(frame);
        }
    }
    static void writeVideoFramePlane(uint8_t *dst,int dstlinesize, const uint8_t *src,  int xsize, int ysize)
    {
        if(dstlinesize != xsize)
        {
            for(int i = 0; i<ysize; ++i)
            {
                memcpy(dst + i*dstlinesize, src + i*xsize,xsize);
            }
        }
        else
        {
            memcpy(dst,src,xsize*ysize);
        }
    }
    void presentFrame(const char* data, int frameSize, int width, int height, int bytesPerLine, QVideoFrameFormat::PixelFormat pixFormat)
    {
        if((width == m_width
            && height == m_height
            && pixFormat == m_pixFormat) == false)
        {
            qDebug() << "reset format to" << frameSize << width << height << bytesPerLine << pixFormat << Qt::endl;
            setFormat(width, height, pixFormat);
            if(m_dumpFrame)
            {
                if(m_dumpFile.is_open())
                {
                    m_dumpFile.close();
                    ++m_dumpFileNo;
                }
                m_dumpFile.open(QString("dump%1.video").arg(m_dumpFileNo).toUtf8().constData(), std::ios::binary);
            }
        }

        QVideoFrame frame(QVideoFrameFormat(QSize(width, height), pixFormat));
        if (frame.map(QVideoFrame::WriteOnly))
        {
            if(m_dumpFrame)
            {
                dumpFrame(data,frameSize);
            }

            switch(pixFormat)
            {
            case QVideoFrameFormat::Format_YUV420P:
            {
                writeVideoFramePlane(frame.bits(0), frame.bytesPerLine(0), (const uint8_t *)data, width, height);
                writeVideoFramePlane(frame.bits(1), frame.bytesPerLine(1), (const uint8_t *)data + width*height, width/2, height/2);
                writeVideoFramePlane(frame.bits(2), frame.bytesPerLine(2), (const uint8_t *)data + width*height + width*height/4, width/2, height/2);
            }
            break;
            case QVideoFrameFormat::Format_BGRA8888:
            {
                writeVideoFramePlane(frame.bits(0), frame.bytesPerLine(0), (const uint8_t *)data, width*4, height);
            }
            break;
            default:
            break;
            }

            frame.unmap();

            if (m_surface)
            {
                m_surface->setVideoFrame(frame);
        }
    }
    }

    void presentVideoFrame(QVideoFrame frame)
    {
        if((m_width == frame.width()
            && m_height == frame.height()
            && m_pixFormat == frame.pixelFormat()) == false)
        {
            qDebug() << "reset format to" << frame.width() << frame.height() << frame.pixelFormat() << Qt::endl;
            setFormat(frame.width(),frame.height(),frame.pixelFormat());
        }

        if (m_surface)
        {
            m_surface->setVideoFrame(frame);
    }
    }

    void enableDumpFrame()
    {
        m_dumpFrame = true;
    }

public slots:
    quint64 getThis()
    {
        return (quint64)this;
    }

    void presentQtFrameSlot(QString devid, QVideoFrame rtcFrame, int width, int height, int bytesPerLine, int pixFormat)
    {
        Q_UNUSED(devid);
        presentQtFrame(rtcFrame, width, height, (QVideoFrameFormat::PixelFormat)pixFormat);
    }

    void presentFrameSlot(QString devid, QByteArray rtcFrame, int width, int height, int bytesPerLine, int pixFormat)
    {
        Q_UNUSED(devid);
        presentFrame(rtcFrame.data(), rtcFrame.size(), width, height, bytesPerLine, (QVideoFrameFormat::PixelFormat)pixFormat);
    }

signals:
    void videoSurfaceChanged();
    void firstRemoteFrameSignal(QString uid, int width, int height);

private:
    void setFormat(int width, int height, QVideoFrameFormat::PixelFormat pixFormat)
    {
        m_width = width;
        m_height = height;
        m_pixFormat = pixFormat;
        m_format = QVideoFrameFormat(QSize(width, height), pixFormat);
        if(m_format.isValid())
        {
            qDebug() << "vidoe format is valid" << Qt::endl;
            }
        else
        {
            qDebug() << "vidoe format is not valid" << Qt::endl;
        }

    }

private:
    bool m_dumpFrame;
    std::ofstream m_dumpFile;
    unsigned m_dumpFileNo;
    QVideoSink* m_surface{nullptr};
    QVideoFrameFormat m_format;
    unsigned m_width;
    unsigned m_height;
    QVideoFrameFormat::PixelFormat m_pixFormat;
};


#endif // VIDEOFRAMEPROVIDER_H
