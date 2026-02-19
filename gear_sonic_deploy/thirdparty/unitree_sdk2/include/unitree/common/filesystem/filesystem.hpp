#ifndef __UT_FILE_SYSTEM_HPP__
#define __UT_FILE_SYSTEM_HPP__

#include <unitree/common/decl.hpp>

#define UT_FILE_READ_SIZE           65536 //64K
#define UT_FILE_READ_BIGGER_SIZE    262144 //256K

#define UT_FD_INVALID   -1
#define UT_FD_STDIN     STDIN_FILENO
#define UT_FD_STDOUT    STDOUT_FILENO
#define UT_FD_STDERR    STDERR_FILENO

#define UT_PATH_DELIM_CHAR  '/'
#define UT_PATH_DELIM_STR   "/"
#define UT_PATH_TRIM_DELIM_STR   " \t\r\n"

#define NOT_FLAG(f, flag)   \
    (((f) & (flag)) != (flag))

#define HAS_FLAG(f, flag)   \
    (((f) & (flag)) == (flag))
    
#define SET_FLAG(f, flag)   \
    (f) |= (flag)

#define IS_RDONLY_FLAG(f) \
    ((f & O_ACCMODE) == O_RDONLY)

namespace unitree
{
namespace common
{
enum
{
    UT_OPEN_FLAG_R         = O_RDONLY,
    UT_OPEN_FLAG_W         = O_WRONLY,
    UT_OPEN_FLAG_RW        = O_RDWR,
    UT_OPEN_FLAG_T         = O_TRUNC,
    UT_OPEN_FLAG_A         = O_APPEND,
    UT_OPEN_FLAG_S         = O_SYNC,
    UT_OPEN_FLAG_C         = O_CREAT,
    UT_OPEN_FLAG_CW        = O_CREAT | O_WRONLY,
    UT_OPEN_FLAG_CWT       = O_CREAT | O_WRONLY | O_TRUNC,
    UT_OPEN_FLAG_CA        = O_CREAT | O_RDWR | O_APPEND,
    UT_OPEN_FLAG_CS        = O_CREAT | O_RDWR | O_SYNC,
    UT_OPEN_FLAG_CAS       = O_CREAT | O_RDWR | O_APPEND | O_SYNC,
    UT_OPEN_FLAG_CTS       = O_CREAT | O_RDWR | O_TRUNC | O_SYNC
};

enum
{
    UT_OPEN_MODE_NONE = 0,
    UT_OPEN_MODE_RW   = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH,
    UT_OPEN_MODE_RWX  = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH
};

enum
{
    UT_OPEN_MASK_000 = 0,
    UT_OPEN_MASK_022 = 0022
};

typedef struct dirent DIRENT;

/*
 * FileSystemHelper
 */
class FileSystemHelper
{
public:
    static FileSystemHelper* Instance()
    {
        static FileSystemHelper inst;
        return &inst;
    }

    /*
     * File Functions
     */
    int32_t Open(const std::string& fileName, int32_t flag, uint32_t mode = 0);

    int64_t Write(int32_t fd, const char* s, int64_t len);
    int64_t Write(int32_t fd, const std::string& s);

    int64_t Read(int32_t fd, char* s, int64_t len);
    int64_t Read(int32_t fd, std::string& s, int64_t len);
    int64_t ReadAll(int32_t fd, std::string& s);

    int64_t Seek(int32_t fd, int64_t offset, int64_t whence);

    bool Stat(int32_t fd, struct stat& statbuf);
    void Truncate(int32_t fd, int64_t len);
    void Sync(int32_t fd);
    void Close(int32_t fd);

    /*
     * Directory Functions
     */
    DIR* Opendir(const std::string& dirName);
    DIRENT* Readdir(DIR* dir);
    void Closedir(DIR* dir);

    void Makedir(const std::string& dirName, uint32_t mode = UT_OPEN_MODE_RWX,
        bool ignoreExist = true);
    void MakedirRecurse(const std::string& dirName, uint32_t mode = UT_OPEN_MODE_RWX);

    /*
     * stat Function
     */
    bool Stat(const std::string& name, struct stat& statbuf);
    bool StatL(const std::string& name, struct stat& statbuf);
    bool IsFile(const struct stat& statbuf);
    bool IsDirectory(const struct stat& statbuf);
    bool IsSymlink(const struct stat& statbuf);
    bool ExistFile(const std::string& name);
    bool ExistDirectory(const std::string& name);
    bool Exist(const std::string& name);

    int64_t GetFileSize(const std::string& fileName);

    /*
     * Symlink Function
     */
    std::string GetSymlink(const std::string& symlinkName);
    bool Symlink(const std::string& targetFileName, const std::string& symlinkName);

    /*
     * Remove and Rename Function
     */
    void RemoveFile(const std::string& fileName, bool ignoreNoExist = true);
    void RemoveDirectory(const std::string& dirName, bool ignoreNoExist = true);

    void Remove(const std::string& name, bool ignoreNoExist = true);
    void Rename(const std::string& oldName, const std::string& newName);

    /*
     * Path proccess Function
     */
    std::string& NormalizePath(std::string& s, bool withEndDelim = true);
    std::string GetFatherDirectory(const std::string& s, bool withEndDelim = true);
    std::string GetFileName(const std::string& s);
    std::string GetLastName(const std::string& s);

    /*
     * Realpath
     */
    std::string GetRealName(const std::string& name);

    bool IsSame(const std::string& name1, const std::string& name2);

    /*
     * MMap Function
     */
    void* MmapRead(int32_t fd, int64_t offset, int64_t len);
    void Munmap(void *addr, int64_t len);

    /*
     * Mode and Owner
     */
    bool Chmod(const std::string& name, mode_t mode);
    bool Chown(const std::string& name, uid_t uid, gid_t gid);

    bool ChmodL(const std::string& name, mode_t mode);
    bool ChownL(const std::string& name, uid_t uid, gid_t gid);

    bool UTime(const std::string& name, struct utimbuf& times);
    bool UTime(const std::string& name, struct timeval times[2]);

    bool Chattr(const std::string& name, const struct stat& statbuf);

    /*
     * ParseModeString
     * @param s: like "0755", "0666", "766"
     * return mode
     */
    uint32_t ParseModeString(const std::string& s);

private:
    FileSystemHelper();
    ~FileSystemHelper();

private:
    int32_t mOldMask;
};

/*
 * FDCloser
 */
class FDCloser
{
public:
    explicit FDCloser() :
        mFd(UT_FD_INVALID)
    {}

    explicit FDCloser(int32_t fd) :
        mFd(fd)
    {}

    ~FDCloser()
    {
        Close();
    }

    void SetFd(int32_t fd)
    {
        //close old fd
        Close();
        //set new fd
        mFd = fd;
    }

private:
    void Close()
    {
        if (mFd > 0)
        {
            FileSystemHelper::Instance()->Close(mFd);
            mFd = UT_FD_INVALID;
        }
    }

private:
    int32_t mFd;
};

/*
 * DIRCloser
 */
class DIRCloser
{
public:
    explicit DIRCloser() :
        mDIR(NULL)
    {}

    explicit DIRCloser(DIR* dir) :
        mDIR(dir)
    {}

    ~DIRCloser()
    {
        Close();
    }

    void SetDir(DIR* dir)
    {
        //close old dir
        Close();
        //set new dir
        mDIR = dir;
    }

private:
    void Close()
    {
        if (mDIR != NULL)
        {
            FileSystemHelper::Instance()->Closedir(mDIR);
            mDIR = NULL;
        }
    }

private:
    DIR* mDIR;
};

//Function
void Remove(const std::string& name);
void Rename(const std::string& oldName, const std::string& newName);

std::string NormalizePath(const std::string& s, bool withEndDelim = false);
std::string GetFatherDirectory(const std::string& s, bool withEndDelim = false);
std::string GetFileName(const std::string& s);
std::string GetLastName(const std::string& s);

void Chattr(const std::string& name, const struct stat& statbuf);
void Copyattr(const std::string& name, const std::string& templateName);
}

}

#endif//__UT_FILE_SYSTEM_HPP__
