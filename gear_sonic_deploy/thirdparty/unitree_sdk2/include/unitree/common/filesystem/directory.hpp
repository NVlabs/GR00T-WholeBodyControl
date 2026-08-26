#ifndef __UT_DIRECTORY_HPP__
#define __UT_DIRECTORY_HPP__

#include <unitree/common/filesystem/filesystem.hpp>

namespace unitree
{
namespace common
{
typedef std::pair<int32_t,std::string> DIRENT_INFO;
typedef std::pair<std::string,std::string> SYMLNK_INFO;

class Directory
{
public:
    Directory();
    Directory(const std::string& dirName);

    ~Directory();

    void Open();
    void Open(const std::string& dirName);

    bool IsOpen();

    void ListFile(std::vector<std::string>& fileNameList, const std::string& regExpress);
    void ListFile(std::vector<std::string>& fileNameList, bool recurse = true, bool absolute = true);

    void ListDir(std::vector<std::string>& dirNameList, bool recurse = true, bool absolute = true);

    void List(std::list<std::string>& fileNameList, std::list<std::string>& dirNameList,
        std::list<SYMLNK_INFO>& symlinkList);

    void Cleanup();
    bool CopyTo(const std::string& dirName, bool deeply = false);

    void Close();

private:
    void CheckOpen();
    void OpenInner();

private:
    DIR *mDIR;
    std::string mDirName;
};

typedef std::shared_ptr<Directory> DirectoryPtr;

bool ExistDirectory(const std::string& dirName);

void CreateDirectory(const std::string& dirName, bool recurse = true,
    uint32_t mode = UT_OPEN_MODE_RWX);

void ListDirectory(const std::string& dirName, std::vector<std::string>& fileNameList,
    const std::string& regExpress);

void ListDirectory(const std::string& dirName, std::vector<std::string>& fileNameList, bool recurse = true, bool absolute = true);

void ListChildDirectory(const std::string& dirName, std::vector<std::string>& dirNameList, bool recurse = true, bool absolute = true);

void RemoveDirectory(const std::string& dirName, bool recurse = true);

void CopyDirectory(const std::string& dirName, const std::string& destDirName, bool deeply);

}
}
#endif//__UT_DIRECTORY_HPP__
