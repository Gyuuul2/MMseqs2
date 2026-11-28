#ifndef MMSEQS_FILEUTIL_H
#define MMSEQS_FILEUTIL_H

#include <cerrno>
#include <cstdio>
#include <ctime>
#include <list>
#include <string>
#include <vector>
#include <cstddef>
#include <utility>
#include <fcntl.h>
#include <sys/stat.h>
#include "Parameters.h"

class FileUtil {

public:
    static bool fileExists(const char *fileName);

    static bool directoryExists(const char *directoryName);

    static FILE* openFileOrDie(const char *fileName, const char *mode, bool shouldExist);

    static size_t countLines(const char *name);

    static bool makeDir(const char *dirName, const int mode = 0777);

    static void deleteTempFiles(const std::list<std::string> &tmpFiles);

    static std::string getRealPathFromSymLink(const std::string path);

    static std::string getHashFromSymLink(const std::string path);

    static void* mmapFile(FILE * file, size_t *dataSize);

    static void munmapData(void * ptr, size_t dataSize);

    static void writeFile(const std::string &pathToFile, const unsigned char *sh, size_t len);
    static void publishFile(const std::string &file, const std::string &content);

    static void sleepBeforePoll(unsigned int polls, unsigned int maxShift = 4);
    static bool hasFailedTooLong(time_t &failedSince, unsigned int &attempts);
    static bool waitBeforeRetry(const char *file, const char *error, time_t &failedSince, unsigned int &attempts);
    static bool canRetryRead(const std::string &file, int error, time_t &failedSince, unsigned int &attempts);
    static int openFile(const std::string &file, int flags = O_RDONLY);
    static int openListedFile(const std::string &file, int flags = O_RDONLY);
    static int openSharedFd(const std::string &file, int flags, size_t *size = NULL);
    static FILE *openSharedFile(const std::string &file, const char *mode);
    static bool statSharedFile(const std::string &file, struct stat *sb = NULL);
    static std::string readFile(const std::string &file);

    // ESTALE or EIO from NFS, a later try can succeed
    static bool isTransientError(int error) {
        return error == EIO || error == ESTALE;
    }
    // bytes done, fewer after an error or end of file (error 0)
    static size_t tryPreadAll(int fd, char *data, size_t size, size_t offset, int &readError);
    static size_t tryPwriteAll(int fd, const char *data, size_t size, size_t offset, int &writeError);
    // a readError of 0 or less is an end of file
    static bool waitBeforeReadRetry(int fd, const char *fileName, int readError, const char *reason, time_t &failedSince, unsigned int &attempts);
    static void preadAll(int fd, const char *fileName, char *data, size_t size, size_t offset);
    // for pread/pwrite only, the file position starts at 0 again
    static void reopenFile(int fd, const char *fileName);
    static void pwriteAll(int fd, const char *fileName, const char *data, size_t size, size_t offset);
    // free the disk blocks of a read range, fd open for writing
    static void punchHole(int fd, size_t offset, size_t size);
    static const size_t PUNCH_HOLE_ALIGNMENT = 1024 * 1024;
    // whole aligned blocks only, the edges may belong to other readers
    static void punchWholeBlocks(int fd, size_t offset, size_t size);

    static std::string dirName(const std::string &file);

    // whether polls for a file that another machine writes in the directory of file need refreshDirectory and the other NFS workarounds
    static bool needsDirectoryRefresh(const std::string &file);
    static void refreshDirectory(const std::string &file);

    static std::string baseName(const std::string &file);

    static size_t getFreeSpace(const char *dir);

    static std::string getCurrentWorkingDirectory();

    static void symlinkAlias(const std::string &file, const std::string &alias);
    static void symlinkAbs(const std::string &target, const std::string &link);

    static size_t getFileSize(const std::string &fileName);

    static char *allocPageBufferWithAdvice(int input_desc, size_t insize);

    static bool symlinkExists(const std::string &path);

    static void copyFile(const char *src, const char *dst);
    static void copyFile(const std::string& src, const std::string& dst);

    // sparse copy of fileRanges while they fit, copiedBytes is the copied or reserved prefix
    static std::string copyToLocalDir(const std::string &db, const std::string &localDir, int threads, std::vector<std::string> &copies, size_t &copiedBytes,
                                      const std::vector<std::vector<std::pair<size_t, size_t> > > &fileRanges, size_t maxBytes, bool isReservingOnly = false);

    // 0 or the errno of the failed write
    static int copyRange(int sourceFd, const char *sourceName, int copyFd, char *buffer, size_t bytes, size_t offset);

    static FILE *openAndDelete(const char *fileName, const char *mode);

    static std::vector<std::string> findDatafiles(const char * datafiles);

    static void remove(const char * file);
    // ignore ENOENT of a retransmitted NFS REMOVE
    static void removeSharedFile(const std::string &file);

    static void move(const char * src, const char * dst);

    static int parseDbType(const char *name);

    static std::string createTemporaryDirectory(const std::string& basePath, const std::string& subDirectory);

    static void fixRlimitNoFile();

    // stdio, logs and readers callers do not count
    static const size_t UNCOUNTED_OPEN_FILES = 64;

    static void checkRlimitNoFile(size_t minOpenFiles);

    // remove forbidden symbols in from filenames
    static std::string sanitizeFilename(std::string name);
};


#endif //MMSEQS_FILEUTIL_H
