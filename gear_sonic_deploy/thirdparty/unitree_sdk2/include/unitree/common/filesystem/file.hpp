#ifndef __UT_FILE_HPP__
#define __UT_FILE_HPP__

#include <unitree/common/filesystem/filesystem.hpp>

namespace unitree
{
namespace common
{
class File
{
public:
    File();
    File(const std::string& fileName);
    File(const std::string& fileName, int32_t flag);
    File(const std::string& fileName, int32_t flag, uint32_t mode);

    virtual ~File();

    int32_t GetFd() const;
    bool IsOpen() const;

    void Open();
    void Open(int32_t flag, uint32_t mode = UT_OPEN_MODE_NONE);
    void Open(const std::string& fileName, int32_t flag = UT_OPEN_FLAG_R,
        uint32_t mode = UT_OPEN_MODE_NONE);

    void Append(const char* s, int64_t len);
    int64_t Write(const char* s, int64_t len);
    int64_t Write(const std::string& s);

    int64_t Read(char* s, int64_t len);
    int64_t Read(std::string& s, int64_t len);
    int64_t ReadAll(std::string& s);

    void SeekBegin();
    void SeekEnd();
    void Seek(int64_t offset, int64_t whence);

    void Sync();
    void Truncate(int64_t len);

    void Close();

    int64_t Size() const;

private:
    void CheckOpen();
    void OpenInner();

private:
    std::string mFileName;
    int32_t mFd;
    int32_t mFlag;
    uint32_t mMode;
};

typedef std::shared_ptr<File> FilePtr;

class MMReadFile
{
public:
    MMReadFile();
    MMReadFile(const std::string& fileName);

    virtual ~MMReadFile();

    int32_t GetFd() const;
    bool IsOpen() const;

    void Open();
    void Open(const std::string& fileName);

    int64_t Size() const;
    int64_t Read(char* s, int64_t len);
    int64_t Read(std::string& s, int64_t len);
    int64_t ReadAll(std::string& s);

    void Seek(int64_t offset);

    void Close();

public:
    void* MMRead(int64_t len, int64_t& canreadlen);
    void MMClose(void* ptr, int64_t len);

private:
    void Init();

private:
    std::string mFileName;
    int32_t mFd;
    int64_t mOffset;
    int64_t mSize;
};

typedef std::shared_ptr<MMReadFile> MMReadFilePtr;

//Functions
bool ExistFile(const std::string& fileName);

std::string LoadFile(const std::string& fileName);
std::string LoadFileEx(const std::string& fileName);

void SaveFile(const std::string& fileName, const char* s, int64_t len,
    uint32_t mode = UT_OPEN_MODE_RW);

void SaveFile(const std::string& fileName, const std::string& s,
    uint32_t mode = UT_OPEN_MODE_RW);

void AppendFile(const std::string& fileName, const char* s, int64_t len,
    uint32_t mode = UT_OPEN_MODE_RW);

void AppendFile(const std::string& fileName, const std::string& s,
    uint32_t mode = UT_OPEN_MODE_RW);

int64_t GetFileSize(const std::string& fileName);

void RemoveFile(const std::string& fileName);

int64_t MMLoadFile(const std::string& fileName, std::string& s);

bool CopyFile(const std::string& fileName, const std::string& destFileName, bool deeply = false);
}
}

#endif//__UT_FILE_HPP__
