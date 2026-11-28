#include "FileUtil.h"
#include "Util.h"
#include "Debug.h"
#include "simd.h"
#include "BatchReader.h"
#include "MMseqsNode.h"

#include <cstddef>
#include <cerrno>
#include <cstring>
#include <climits>
#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>

#include <sys/stat.h>
#include <dirent.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/resource.h>

bool FileUtil::fileExists(const char* fileName) {
    struct stat st;
    return stat(fileName, &st) == 0;
}

bool FileUtil::directoryExists(const char* directoryName) {
    struct stat st;
    return stat(directoryName, &st) == 0 && S_ISDIR(st.st_mode);
}

bool FileUtil::makeDir(const char* directoryName, const int mode ) {
    return mkdir(directoryName, mode) == 0;
}

void* FileUtil::mmapFile(FILE * file, size_t *dataSize){
    struct stat sb;
    if (fstat(fileno(file), &sb) < 0) {
        int errsv = errno;
        Debug(Debug::ERROR) << "Failed to fstat." << ". Error " << errsv << ".\n";
        EXIT(EXIT_FAILURE);
    }
    *dataSize = sb.st_size;
    int mode = PROT_READ;
    int fd =  fileno(file);

    void * ret = mmap(NULL, *dataSize, mode, MAP_PRIVATE, fd, 0);
    if(ret == MAP_FAILED){
        int errsv = errno;
        Debug(Debug::ERROR) << "Failed to mmap memory dataSize=" << *dataSize <<". Error " << errsv << ".\n";
        EXIT(EXIT_FAILURE);
    }
    return ret;
}

void FileUtil::munmapData(void * ptr, size_t dataSize){
    if(munmap(ptr, dataSize) < 0){
        Debug(Debug::ERROR) << "Failed to munmap memory\n";
        EXIT(EXIT_FAILURE);
    }
}

FILE* FileUtil::openFileOrDie(const char * fileName, const char * mode, bool shouldExist) {
    bool exists = FileUtil::fileExists(fileName);
    if (exists && !shouldExist) {
        errno = EEXIST;
        perror(fileName);
        EXIT(EXIT_FAILURE);
    }
    if (!exists && shouldExist) {
        errno = ENOENT;
        perror(fileName);
        EXIT(EXIT_FAILURE);
    }

    FILE* file;
    file = fopen(fileName, mode);
    if(file == NULL) { perror(fileName); EXIT(EXIT_FAILURE); }
    return file;
}
size_t FileUtil::countLines(const char* name) {
    FILE *fp = FileUtil::openFileOrDie(name, "r", true);
    size_t cnt = 0;
    while (!feof(fp)) {
        char ch = fgetc(fp);
        cnt += (ch == '\n') ? 1 : 0;
    }
    if (fclose(fp) != 0) {
        Debug(Debug::ERROR) << "Cannot close file " << name << "\n";
        EXIT(EXIT_FAILURE);
    }
    return cnt;
}


void FileUtil::deleteTempFiles(const std::list<std::string> &tmpFiles) {
    for (std::list<std::string>::const_iterator it = tmpFiles.begin(); it != tmpFiles.end(); it++) {
        Debug(Debug::INFO) << "Deleting " << *it << "\n";
        std::string file = *it;
        FileUtil::remove(file.c_str());
    }
}

void FileUtil::sleepBeforePoll(unsigned int polls, unsigned int maxShift) {
    usleep(10000u << std::min(polls, maxShift));
}

bool FileUtil::hasFailedTooLong(time_t &failedSince, unsigned int &attempts) {
    time_t now = time(NULL);
    if (attempts++ == 0) {
        failedSince = now;
    }
    return attempts >= 100 && now - failedSince >= 60;
}

bool FileUtil::waitBeforeRetry(const char *file, const char *error, time_t &failedSince, unsigned int &attempts) {
    if (hasFailedTooLong(failedSince, attempts)) {
        return false;
    }
    if (attempts > 1) {
        if (attempts == 2) {
            Debug(Debug::WARNING) << "Cannot access " << file << ": " << error << ", retrying\n";
        }
        sleepBeforePoll(attempts - 1, 7);
    }
    return true;
}

bool FileUtil::canRetryRead(const std::string &file, int error, time_t &failedSince, unsigned int &attempts) {
    bool isCachedMissing = error == ENOENT && FileUtil::needsDirectoryRefresh(file);
    return (FileUtil::isTransientError(error) || isCachedMissing) && FileUtil::waitBeforeRetry(file.c_str(), strerror(error), failedSince, attempts);
}

// NFS caches a missing name for up to 60 s
static bool isListed(const std::string &file) {
    if (FileUtil::needsDirectoryRefresh(file) == false) {
        return false;
    }
    static time_t lastListing = 0;
    time_t now = time(NULL);
    time_t last = __atomic_load_n(&lastListing, __ATOMIC_RELAXED);
    if (now == last || __sync_bool_compare_and_swap(&lastListing, last, now) == false) {
        return false;
    }
    std::string name = FileUtil::baseName(file);
    DIR *directory = opendir(FileUtil::dirName(file).c_str());
    if (directory == NULL) {
        return false;
    }
    bool isFound = false;
    for (struct dirent *entry; isFound == false && (entry = readdir(directory)) != NULL;) {
        isFound = name == entry->d_name;
    }
    closedir(directory);
    return isFound;
}

int FileUtil::openFile(const std::string &file, int flags) {
    int fd;
    do {
        fd = open(file.c_str(), flags, 0666);
    } while (fd < 0 && errno == EINTR);
    return fd;
}

int FileUtil::openListedFile(const std::string &file, int flags) {
    int fd = openFile(file, flags);
    int openError = errno;
    if (fd < 0 && openError == ENOENT && isListed(file)) {
        return openFile(file, flags);
    }
    errno = openError;
    return fd;
}

int FileUtil::openSharedFd(const std::string &file, int flags, size_t *size) {
    time_t failedSince = 0;
    unsigned int attempts = 0;
    for (;;) {
        int fd = openListedFile(file, flags | O_CLOEXEC);
        if (fd >= 0) {
            struct stat sb;
            if (size != NULL) {
                if (fstat(fd, &sb) != 0) {
                    Debug(Debug::ERROR) << "Cannot read " << file << "\n";
                    EXIT(EXIT_FAILURE);
                }
                *size = sb.st_size;
            }
            return fd;
        }
        int openError = errno;
        if (canRetryRead(file, openError, failedSince, attempts) == false) {
            Debug(Debug::ERROR) << "Cannot open " << file << ": " << strerror(openError) << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
}

// for the modes r, rb and r+
FILE *FileUtil::openSharedFile(const std::string &file, const char *mode) {
    FILE *handle = fdopen(openSharedFd(file, (strchr(mode, '+') != NULL) ? O_RDWR : O_RDONLY), mode);
    if (handle == NULL) {
        Debug(Debug::ERROR) << "Cannot open " << file << "\n";
        EXIT(EXIT_FAILURE);
    }
    return handle;
}

// missing names return false at once
bool FileUtil::statSharedFile(const std::string &file, struct stat *sb) {
    time_t failedSince = 0;
    unsigned int attempts = 0;
    struct stat st;
    for (;;) {
        if (stat(file.c_str(), (sb != NULL) ? sb : &st) == 0) {
            return true;
        }
        int statError = errno;
        if (statError == ENOENT || statError == ENOTDIR) {
            return false;
        }
        if (canRetryRead(file, statError, failedSince, attempts) == false) {
            Debug(Debug::ERROR) << "Cannot stat " << file << ": " << strerror(statError) << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
}

std::string FileUtil::readFile(const std::string &file) {
    size_t size;
    int fd = openSharedFd(file, O_RDONLY, &size);
    std::string content(size, '\0');
    FileUtil::preadAll(fd, file.c_str(), &content[0], content.size(), 0);
    if (close(fd) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << file << "\n";
        EXIT(EXIT_FAILURE);
    }
    return content;
}

size_t FileUtil::tryPreadAll(int fd, char *data, size_t size, size_t offset, int &readError) {
    for (size_t done = 0; done < size;) {
        ssize_t count = pread(fd, data + done, size - done, offset + done);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            readError = (count < 0) ? errno : 0;
            return done;
        }
        done += count;
    }
    readError = 0;
    return size;
}

bool FileUtil::waitBeforeReadRetry(int fd, const char *fileName, int readError, const char *reason, time_t &failedSince, unsigned int &attempts) {
    if ((readError <= 0 || readError == EAGAIN || isTransientError(readError)) == false
        || waitBeforeRetry(fileName, reason, failedSince, attempts) == false) {
        return false;
    }
    // first retry only rereads, another reader may have reopened it
    if (attempts > 1 && isTransientError(readError)) {
        reopenFile(fd, fileName);
    }
    return true;
}

void FileUtil::preadAll(int fd, const char *fileName, char *data, size_t size, size_t offset) {
    time_t failedSince = 0;
    unsigned int attempts = 0;
    int readError;
    for (size_t done = tryPreadAll(fd, data, size, offset, readError); done < size;) {
        const char *reason = (readError == 0) ? "end of file" : strerror(readError);
        if (waitBeforeReadRetry(fd, fileName, readError, reason, failedSince, attempts) == false) {
            Debug(Debug::ERROR) << "Cannot read " << fileName << " at offset " << offset + done
                                << " (" << size - done << " bytes remaining): " << reason << "\n";
            EXIT(EXIT_FAILURE);
        }
        size_t count = tryPreadAll(fd, data + done, size - done, offset + done, readError);
        attempts = (count > 0) ? 0 : attempts;
        done += count;
    }
}

void FileUtil::reopenFile(int fd, const char *fileName) {
    int fileFlags = fcntl(fd, F_GETFL);
    int fdFlags = fcntl(fd, F_GETFD);
    int reopenedFd = (fileFlags < 0 || fdFlags < 0) ? -1 : open(fileName, fileFlags);
    if (reopenedFd >= 0) {
        dup2(reopenedFd, fd);
        fcntl(fd, F_SETFD, fdFlags);
        close(reopenedFd);
    }
}

size_t FileUtil::tryPwriteAll(int fd, const char *data, size_t size, size_t offset, int &writeError) {
    for (size_t done = 0; done < size;) {
        ssize_t count = pwrite(fd, data + done, size - done, offset + done);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            writeError = (count < 0) ? errno : 0;
            return done;
        }
        done += count;
    }
    writeError = 0;
    return size;
}

void FileUtil::pwriteAll(int fd, const char *fileName, const char *data, size_t size, size_t offset) {
    time_t failedSince = 0;
    unsigned int attempts = 0;
    int writeError;
    for (size_t done = tryPwriteAll(fd, data, size, offset, writeError); done < size;) {
        // no retry on EIO, NFS may have lost written back data
        if ((writeError != EAGAIN && writeError != ESTALE) || waitBeforeRetry(fileName, strerror(writeError), failedSince, attempts) == false) {
            Debug(Debug::ERROR) << "Cannot write " << fileName << " at offset " << offset + done << ": " << (writeError == 0 ? "no bytes written" : strerror(writeError)) << "\n";
            EXIT(EXIT_FAILURE);
        }
        if (attempts > 1 && writeError == ESTALE) {
            reopenFile(fd, fileName);
        }
        size_t count = tryPwriteAll(fd, data + done, size - done, offset + done, writeError);
        attempts = (count > 0) ? 0 : attempts;
        done += count;
    }
}

void FileUtil::punchHole(int fd, size_t offset, size_t size) {
#ifdef FALLOC_FL_PUNCH_HOLE
    static std::atomic<bool> warned(false);
    if (size == 0 || fallocate(fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, offset, size) == 0) {
        return;
    }
    int error = errno;
    if (warned.exchange(true) == false) {
        Debug(Debug::WARNING) << "The file system cannot free the parts of temporary files that the nodes have read (" << strerror(error) << "). "
                              << "Each temporary file keeps its full size until its step deletes it, so the run needs more free temporary disk\n";
    }
#else
    (void) fd; (void) offset; (void) size;
#endif
}

void FileUtil::punchWholeBlocks(int fd, size_t offset, size_t size) {
    size_t start = (offset + PUNCH_HOLE_ALIGNMENT - 1) / PUNCH_HOLE_ALIGNMENT * PUNCH_HOLE_ALIGNMENT;
    size_t end = (offset + size) / PUNCH_HOLE_ALIGNMENT * PUNCH_HOLE_ALIGNMENT;
    if (end > start) {
        punchHole(fd, start, end - start);
    }
}

void FileUtil::writeFile(const std::string &pathToFile, const unsigned char *data, size_t len) {
    int fd = open(pathToFile.c_str(), O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR | S_IXUSR);
    if (fd == -1) {
        Debug(Debug::ERROR) << "Could not write file " << pathToFile << "!\n";
        EXIT(EXIT_FAILURE);
    }

    ssize_t res = write(fd, data, len);
    if (res == -1) {
        Debug(Debug::ERROR) << "Error writing file " << pathToFile << "!\n";
        EXIT(EXIT_FAILURE);
    }

    if (close(fd) != 0) {
        Debug(Debug::ERROR) << "Error closing file " << pathToFile << "!\n";
        EXIT(EXIT_FAILURE);
    }
}

void FileUtil::publishFile(const std::string &file, const std::string &content) {
    std::string tmpFile = file + ".tmp";
    FILE *handle = openAndDelete(tmpFile.c_str(), "w");
    if (fwrite(content.c_str(), sizeof(char), content.size(), handle) != content.size()) {
        Debug(Debug::ERROR) << "Cannot write " << tmpFile << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (fflush(handle) != 0 || fsync(fileno(handle)) != 0) {
        Debug(Debug::ERROR) << "Cannot flush " << tmpFile << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (fclose(handle) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << tmpFile << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (rename(tmpFile.c_str(), file.c_str()) != 0) {
        Debug(Debug::ERROR) << "Cannot rename " << tmpFile << " to " << file << "\n";
        EXIT(EXIT_FAILURE);
    }
}

// NFS caches missing names, unknown storage counts as NFS
bool FileUtil::needsDirectoryRefresh(const std::string &file) {
    static std::mutex directoriesMutex;
    static std::map<std::string, bool> directories;
    std::string directory = dirName(file);
    std::lock_guard<std::mutex> lock(directoriesMutex);
    std::map<std::string, bool>::const_iterator it = directories.find(directory);
    if (it != directories.end()) {
        return it->second;
    }
    bool needsRefresh = true;
#ifdef __linux__
    int fd = open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) {
        return true;
    }
    StorageKind storage = getStorageKind(fd);
    close(fd);
    needsRefresh = storage == STORAGE_NFS || storage == STORAGE_UNKNOWN;
#endif
    directories[directory] = needsRefresh;
    return needsRefresh;
}

// opening a directory makes NFS look up missing names again
void FileUtil::refreshDirectory(const std::string &file) {
    if (needsDirectoryRefresh(file) == false) {
        return;
    }
    DIR *directory = opendir(dirName(file).c_str());
    if (directory != NULL) {
        closedir(directory);
    }
}

std::string FileUtil::dirName(const std::string &file) {
    size_t pos = file.find_last_of("\\/");
    return (std::string::npos == pos)
           ? "."
           : file.substr(0, pos);
}

std::string FileUtil::baseName(const std::string &file) {
    size_t pos = file.find_last_of("\\/");
    return (std::string::npos == pos)
           ? file
           : file.substr(pos+1, file.length());
}

size_t FileUtil::getFreeSpace(const char *path) {
    struct statvfs stat;
    if (statvfs(path, &stat) != 0) {
        // error happens, just quits here
        return SIZE_MAX;
    }

    // the available size is f_bsize * f_bavail
    return stat.f_bavail * stat.f_frsize;
}


std::string FileUtil::getRealPathFromSymLink(const std::string path){
    char *p = realpath(path.c_str(), NULL);
    if (p == NULL) {
        Debug(Debug::ERROR) << "Could not get path of " << path << "!\n";
        EXIT(EXIT_FAILURE);
    }

    std::string name(p);
    free(p);
    return name;
}

std::string FileUtil::getHashFromSymLink(const std::string path){
    char *p = realpath(path.c_str(), NULL);
    if (p == NULL) {
        Debug(Debug::ERROR) << "Could not get path of " << path << "!\n";
        EXIT(EXIT_FAILURE);
    }

    std::string base = baseName(p);
    free(p);

    return base;
}

void FileUtil::symlinkAlias(const std::string &file, const std::string &alias) {
    char *p = realpath(file.c_str(), NULL);
    if (p == NULL) {
        Debug(Debug::ERROR) << "Could not get path of " << file << "!\n";
        EXIT(EXIT_FAILURE);
    }

    std::string path = dirName(p);
    std::string base = baseName(p);
    free(p);

    std::string pathToAlias = (path + "/" + alias);
    if (symlinkExists(pathToAlias) == true){
        FileUtil::remove(pathToAlias.c_str());
    }
    // symlinkat is not available in Conda macOS
    // Conda uses the macOS 10.9 SDK, and symlinkat was introduced in 10.10
    // We emulate symlinkat by manipulating the CWD instead
    std::string oldWd = FileUtil::getCurrentWorkingDirectory();
    if (chdir(path.c_str()) != 0) {
        Debug(Debug::ERROR) << "Could not change working directory to " << path << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (symlink(base.c_str(), alias.c_str()) != 0) {
        Debug(Debug::ERROR) << "Could not create symlink of " << file << "!\n";
        EXIT(EXIT_FAILURE);
    }
    if (chdir(oldWd.c_str()) != 0) {
        Debug(Debug::ERROR) << "Could not change working directory to " << oldWd << "\n";
        EXIT(EXIT_FAILURE);
    }
}

std::string FileUtil::getCurrentWorkingDirectory() {
    // CWD can be larger than PATH_MAX and allocating enough memory is somewhat tricky
    char* wd = NULL;
#ifdef PATH_MAX
    size_t bufferSize = PATH_MAX;
#else
    size_t bufferSize = 1024;
#endif
    do {
        if (wd != NULL) {
            free(wd);
            bufferSize *= 2;
        }
        errno = 0;
        wd = getcwd(NULL, bufferSize);
        if (wd == NULL && errno != ERANGE && errno != 0) {
            Debug(Debug::ERROR) << "Could not get current working directory\n";
            EXIT(EXIT_FAILURE);
        }
    } while (wd == NULL && errno == ERANGE);
    std::string cwd(wd);
    free(wd);
    return cwd;
}

void FileUtil::symlinkAbs(const std::string &target, const std::string &link) {
    if(FileUtil::fileExists(link.c_str())){
        FileUtil::remove(link.c_str());
    }
    char *t = realpath(target.c_str(), NULL);
    if (t == NULL) {
        Debug(Debug::ERROR) << "Could not get realpath of " << target << "!\n";
        EXIT(EXIT_FAILURE);
    }

    std::string realLink;
    char *l = realpath(link.c_str(), NULL);
    if (l == NULL) {
        std::string path = dirName(link);
        std::string base = baseName(link);
        l = realpath(path.c_str(), NULL);
        if (l == NULL) {
            Debug(Debug::ERROR) << "Could not get realpath of " << link << "!\n";
            EXIT(EXIT_FAILURE);
        } else {
            realLink = (std::string(l) + "/" + base);
        }
    } else {
        realLink = l;
        if (symlinkExists(realLink) == true){
            FileUtil::remove(realLink.c_str());
        }
    }

    if (symlink(t, realLink.c_str()) != 0) {
        Debug(Debug::ERROR) << "Could not create symlink of " << target << "!\n";
        EXIT(EXIT_FAILURE);
    }

    free(t);
    free(l);
}

size_t FileUtil::getFileSize(const std::string &fileName) {
    struct stat stat_buf;
    int rc = stat(fileName.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

char *FileUtil::allocPageBufferWithAdvice(int input_desc, size_t insize) {
    const size_t page_size = getpagesize();
    void *inbuf = mem_align(page_size, insize);
#if HAVE_POSIX_FADVISE
    if (posix_fadvise(input_desc, 0, 0, POSIX_FADV_SEQUENTIAL) != 0) {
        Debug(Debug::ERROR) << "posix_fadvise returned an error\n";
    }
#else
    (void)input_desc;
#endif
    return static_cast<char *>(inbuf);
}


bool FileUtil::symlinkExists(const std::string &path)  {
    struct stat buf;
    int result = lstat(path.c_str(), &buf);
    return (result == 0);
}

void FileUtil::copyFile(const char *src, const char *dst) {
    //https://stackoverflow.com/questions/10195343/copy-a-file-in-a-sane-safe-and-efficient-way
    char buf[BUFSIZ];
    size_t size;

    int source = open(src, O_RDONLY, 0);
    if (source == -1) {
        Debug(Debug::ERROR) << "Could not open file " << src << "!\n";
        EXIT(EXIT_FAILURE);
    }
    int dest = open(dst, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (dest == -1) {
        Debug(Debug::ERROR) << "Could not open file " << dst << "!\n";
        EXIT(EXIT_FAILURE);
    }
    while ((size = read(source, buf, BUFSIZ)) > 0) {
        size_t res = write(dest, buf, size);
        if (res != size) {
            Debug(Debug::ERROR) << "Error writing file " << dst << "!\n";
            EXIT(EXIT_FAILURE);
        }
    }
    close(source);
    close(dest);
}

void FileUtil::copyFile(const std::string& src, const std::string& dst) {
    copyFile(src.c_str(), dst.c_str());
}

FILE * FileUtil::openAndDelete(const char *fileName, const char *mode) {
    if(FileUtil::fileExists(fileName) == true){
        if(FileUtil::directoryExists(fileName)){
            Debug(Debug::ERROR) << "Can not open " << fileName << " for writing. It is a directory.\n";
            EXIT(EXIT_FAILURE);
        }else {
            FileUtil::remove(fileName);
        }
    }
    FILE * file = fopen(fileName, mode);
    if (file == NULL) {
        Debug(Debug::ERROR) << "Could not open " << fileName << " for writing!\n";
        EXIT(EXIT_FAILURE);
    }
    return file;
}

std::vector<std::string> FileUtil::findDatafiles(const char * datafiles){
    std::string baseName = std::string(datafiles);
    std::string checkName = baseName + ".0";
    std::vector<std::string> filenames;
    size_t cnt = 0;
    while(statSharedFile(checkName) == true){
        filenames.push_back(checkName);
        cnt++;
        checkName = baseName + "." + SSTR(cnt);
    }
    if(cnt == 0){
        if(statSharedFile(baseName)){
            filenames.push_back(baseName);
        }
    }
    return filenames;
}

void FileUtil::remove(const char * file ) {
    if (std::remove(file) != 0){
        Debug(Debug::ERROR) << "Could not delete " << file << "!\n";
        EXIT(EXIT_FAILURE);
    }
}

void FileUtil::removeSharedFile(const std::string &file) {
    if (std::remove(file.c_str()) != 0 && errno != ENOENT) {
        Debug(Debug::ERROR) << "Could not delete " << file << "!\n";
        EXIT(EXIT_FAILURE);
    }
}

void FileUtil::move(const char * src, const char * dst) {
    struct stat srcFileInfo;
    FILE * srcFile = FileUtil::openFileOrDie(src, "rw", true);
    if (fstat(fileno(srcFile), &srcFileInfo) < 0) {
        int errsv = errno;
        Debug(Debug::ERROR) << "Failed to fstat File=" << src << ". Error " << errsv << ".\n";
        EXIT(EXIT_FAILURE);
    }
    struct stat srcDirInfo;
    std::string dirName = FileUtil::dirName(dst);
    FILE * dstDir = FileUtil::openFileOrDie(dirName.c_str(), "r", true);
    if (fstat(fileno(dstDir), &srcDirInfo) < 0) {
        int errsv = errno;
        Debug(Debug::ERROR) << "Failed to fstat File=" << dirName << ". Error " << errsv << ".\n";
        EXIT(EXIT_FAILURE);
    }
    bool sameFileSystem = (srcDirInfo.st_dev == srcFileInfo.st_dev);
    if (fclose(srcFile) != 0) {
        Debug(Debug::ERROR) << "Cannot close file " << src << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (fclose(dstDir) != 0) {
        Debug(Debug::ERROR) << "Cannot close directory " << dirName << "\n";
        EXIT(EXIT_FAILURE);
    }
    if(sameFileSystem){
        if (std::rename(src, dst) != 0){
            Debug(Debug::ERROR) << "Could not copy file " << src << " to " << dst << "!\n";
            EXIT(EXIT_FAILURE);
        }
    }else{
        FileUtil::copyFile(src, dst);
        FileUtil::remove(src);
    }
}

int FileUtil::parseDbType(const char *name) {
    std::string dbTypeFile = std::string(name) + ".dbtype";
    struct stat sb;
    if (statSharedFile(dbTypeFile, &sb) == false) {
        return Parameters::DBTYPE_GENERIC_DB;
    }

    size_t fileSize = sb.st_size;
    if (fileSize != sizeof(int)) {
        Debug(Debug::ERROR) << "File size of " << dbTypeFile << " seems to be wrong!\n";
        Debug(Debug::ERROR) << "It should have 4 bytes but it has " <<  fileSize << " bytes.";
        EXIT(EXIT_FAILURE);
    }
    FILE *file = openSharedFile(dbTypeFile, "r");
    int dbtype;
    preadAll(fileno(file), dbTypeFile.c_str(), (char *) &dbtype, fileSize, 0);
    if (fclose(file) != 0) {
        Debug(Debug::ERROR) << "Cannot close file " << dbTypeFile << "\n";
        EXIT(EXIT_FAILURE);
    }
#if SIMDE_ENDIAN_ORDER == SIMDE_ENDIAN_BIG
    dbtype = __builtin_bswap32(dbtype);
#endif
    return dbtype;
}

std::string FileUtil::createTemporaryDirectory(const std::string& basePath, const std::string& subDirectory) {
    std::string tmpDir(basePath);
    if (FileUtil::directoryExists(tmpDir.c_str()) == false) {
        Debug(Debug::INFO) << "Path " << tmpDir << " does not exist or is not a directory.\n";
        if (FileUtil::makeDir(tmpDir.c_str()) == false) {
            Debug(Debug::ERROR) << "Cannot create temporary folder " << tmpDir << ".\n";
            EXIT(EXIT_FAILURE);
        } else {
            Debug(Debug::INFO) << "Created directory " << tmpDir << "\n";
        }
    }
    tmpDir += "/" + subDirectory;
    if (FileUtil::directoryExists(tmpDir.c_str()) == false) {
        if (FileUtil::makeDir(tmpDir.c_str()) == false) {
            Debug(Debug::ERROR) << "Cannot create temporary subfolder " << tmpDir << ".\n";
            EXIT(EXIT_FAILURE);
        }
    }
    FileUtil::symlinkAlias(tmpDir, "latest");
    return tmpDir;
}

void FileUtil::fixRlimitNoFile() {
    static bool increasedRlimitNoFile(false);
    if (increasedRlimitNoFile == false) {
        increasedRlimitNoFile = true;
        struct rlimit limit;
        if (getrlimit(RLIMIT_NOFILE, &limit) != 0) {
            Debug(Debug::WARNING) << "Could not increase maximum number of open files (getrlimit " << errno << "). Use ulimit manually\n";
            return;
        }
        limit.rlim_cur = std::min(std::max((rlim_t)8192, limit.rlim_cur), limit.rlim_max);
        limit.rlim_max = std::min(RLIM_INFINITY, limit.rlim_max);
        if (setrlimit(RLIMIT_NOFILE, &limit) != 0) {
            Debug(Debug::WARNING) << "Could not increase maximum number of open files (setrlimit " << errno << "). Use ulimit manually\n";
        }
    }
}

void FileUtil::checkRlimitNoFile(size_t minOpenFiles) {
    struct rlimit limit;
    if (getrlimit(RLIMIT_NOFILE, &limit) != 0) {
        Debug(Debug::WARNING) << "Could not check maximum number of open files (getrlimit " << errno << "). Use ulimit -n " << minOpenFiles << " manually\n";
        return;
    }
    struct rlimit raised = { (rlim_t) minOpenFiles, limit.rlim_max };
    if (minOpenFiles > limit.rlim_cur && setrlimit(RLIMIT_NOFILE, &raised) != 0) {
        Debug(Debug::ERROR) << "Raise the open file limit (ulimit -n) to at least " << minOpenFiles << "\n";
        EXIT(EXIT_FAILURE);
    }
}

std::string FileUtil::sanitizeFilename(std::string name){
    static const std::vector<std::pair<char, char>> symbolTable =
            {{'\\', '@'},
             {'/', '@'},
             {':', '@'},
             {'*', '@'},
             {'?', '@'},
             {'<', '@'},
             {'>', '@'},
             {'|', '!'}};

    std::vector<std::pair<char, char>>::const_iterator it;
    for (it = symbolTable.begin(); it != symbolTable.end(); ++it) {
        std::replace(name.begin(), name.end(), it->first, it->second);
    }
    return name;
}

static const size_t LOCAL_COPY_IO_BYTES = 64 * 1024 * 1024;

int FileUtil::copyRange(int sourceFd, const char *sourceName, int copyFd, char *buffer, size_t bytes, size_t offset) {
    preadAll(sourceFd, sourceName, buffer, bytes, offset);
    int writeError;
    if (tryPwriteAll(copyFd, buffer, bytes, offset, writeError) < bytes) {
        return (writeError != 0) ? writeError : EIO;
    }
    return 0;
}

std::string FileUtil::copyToLocalDir(const std::string &db, const std::string &localDir, int threads, std::vector<std::string> &copies, size_t &copiedBytes,
                                     const std::vector<std::vector<std::pair<size_t, size_t> > > &fileRanges, size_t maxBytes, bool isReservingOnly) {
    std::vector<std::string> dataFiles = findDatafiles(db.c_str());
    if (dataFiles.empty()) {
        Debug(Debug::ERROR) << "No datafile could be found for " << db << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (directoryExists(localDir.c_str()) == false && makeDir(localDir.c_str()) == false) {
        Debug(Debug::ERROR) << "Cannot create " << localDir << "\n";
        EXIT(EXIT_FAILURE);
    }
    std::string localDb = localDir + "/" + baseName(db);
    if (fileExists((localDb + ".dbtype").c_str())) {
        remove((localDb + ".dbtype").c_str());
    }
    std::vector<size_t> fileBytes(dataFiles.size());
    std::vector<std::pair<size_t, size_t> > chunks;
    std::vector<size_t> chunkBytes;
    for (size_t i = 0; i < dataFiles.size(); i++) {
        fileBytes[i] = getFileSize(dataFiles[i]);
        const std::vector<std::pair<size_t, size_t> > &ranges = fileRanges[i];
        for (size_t range = 0; range < ranges.size(); range++) {
            for (size_t offset = ranges[range].first; offset < ranges[range].second; offset += LOCAL_COPY_IO_BYTES) {
                chunks.emplace_back(i, offset);
                chunkBytes.push_back(std::min(LOCAL_COPY_IO_BYTES, ranges[range].second - offset));
            }
        }
    }
    std::vector<int> sourceFds(dataFiles.size());
    std::vector<int> copyFds(dataFiles.size());
    size_t firstCopy = copies.size();
    for (size_t i = 0; i < dataFiles.size(); i++) {
        // keep the suffix of the data file
        copies.push_back(localDb + dataFiles[i].substr(db.size()));
        sourceFds[i] = ::open(dataFiles[i].c_str(), O_RDONLY | O_CLOEXEC);
        copyFds[i] = ::open(copies.back().c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if (sourceFds[i] < 0 || copyFds[i] < 0 || ftruncate(copyFds[i], fileBytes[i]) != 0) {
            Debug(Debug::ERROR) << "Cannot copy " << dataFiles[i] << " to " << copies.back() << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
    // before reserving, publishFile exits on a full device
    std::string dbtype = readFile(db + ".dbtype");
    if (dbtype.size() != sizeof(int)) {
        Debug(Debug::ERROR) << "Invalid database type file: " << db << ".dbtype\n";
        EXIT(EXIT_FAILURE);
    }
    publishFile(localDb + ".dbtype", dbtype);
    copies.push_back(localDb + ".dbtype");
    // reserve in order up to the first chunk that does not fit
    size_t totalBytes = 0;
    size_t chunkCount = 0;
    for (; chunkCount < chunks.size() && chunkBytes[chunkCount] <= maxBytes - totalBytes; chunkCount++) {
#ifdef __linux__
        int fd = copyFds[chunks[chunkCount].first];
        int failed;
        do {
            failed = fallocate(fd, 0, chunks[chunkCount].second, chunkBytes[chunkCount]);
        } while (failed != 0 && errno == EINTR);
        int reserveError = errno;
        if (failed != 0 && reserveError != EOPNOTSUPP && reserveError != ENOSYS) {
            std::string copyEnd;
            MMseqsNode::addPhaseValue(copyEnd, "Local tmp", "cannot reserve " + copies[firstCopy + chunks[chunkCount].first] + " at offset " + SSTR(chunks[chunkCount].second) + ": " + strerror(reserveError)
                                      + ", the copy ends there");
            MMseqsNode::printPhaseValues(copyEnd);
            punchHole(fd, chunks[chunkCount].second, chunkBytes[chunkCount]);
            break;
        }
#endif
        totalBytes += chunkBytes[chunkCount];
    }
    chunks.resize(isReservingOnly ? 0 : chunkCount);
    // without fallocate the first chunk on a full device ends the copy
    size_t failedChunk = chunks.size();
    // up to 64 large reads from shared storage
    int copyThreads = std::max(1, std::min(std::min(threads, 64), (int) chunks.size()));
#pragma omp parallel num_threads(copyThreads)
    {
        std::vector<char> buffer(chunks.empty() ? 0 : LOCAL_COPY_IO_BYTES);
#pragma omp for schedule(dynamic, 1)
        for (size_t i = 0; i < chunks.size(); i++) {
            if (i > __atomic_load_n(&failedChunk, __ATOMIC_RELAXED)) {
                continue;
            }
            size_t file = chunks[i].first;
            int writeError = copyRange(sourceFds[file], dataFiles[file].c_str(), copyFds[file], buffer.data(), chunkBytes[i], chunks[i].second);
            if (writeError == ENOSPC || writeError == EDQUOT) {
                std::string copyEnd;
                MMseqsNode::addPhaseValue(copyEnd, "Local tmp", "cannot write " + copies[firstCopy + file] + " at offset " + SSTR(chunks[i].second) + ": " + strerror(writeError) + ", the copy ends there");
                MMseqsNode::printPhaseValues(copyEnd);
                size_t failed = __atomic_load_n(&failedChunk, __ATOMIC_RELAXED);
                while (i < failed && __atomic_compare_exchange_n(&failedChunk, &failed, i, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED) == false) {}
            } else if (writeError != 0) {
                Debug(Debug::ERROR) << "Cannot write " << copies[firstCopy + file] << " at offset " << chunks[i].second << ": " << strerror(writeError) << "\n";
                EXIT(EXIT_FAILURE);
            }
        }
    }
    // cut off everything from the failed chunk on
    if (failedChunk < chunks.size()) {
        totalBytes = 0;
        for (size_t i = 0; i < failedChunk; i++) {
            totalBytes += chunkBytes[i];
        }
        for (size_t file = chunks[failedChunk].first; file < dataFiles.size(); file++) {
            size_t keptBytes = (file == chunks[failedChunk].first) ? chunks[failedChunk].second : 0;
            if (ftruncate(copyFds[file], keptBytes) != 0 || ftruncate(copyFds[file], fileBytes[file]) != 0) {
                Debug(Debug::ERROR) << "Cannot truncate " << copies[firstCopy + file] << "\n";
                EXIT(EXIT_FAILURE);
            }
        }
    }
    for (size_t i = 0; i < dataFiles.size(); i++) {
        if (::close(sourceFds[i]) != 0 || ::close(copyFds[i]) != 0) {
            Debug(Debug::ERROR) << "Cannot copy " << dataFiles[i] << " to " << copies[firstCopy + i] << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
    copiedBytes = totalBytes;
    std::string copied;
    MMseqsNode::addPhaseValue(copied, "Local tmp", (isReservingOnly ? "reserved " : "copied ") + MMseqsNode::formatBytes(totalBytes) + " of " + db + " to " + localDb);
    MMseqsNode::printPhaseValues(copied);
    return localDb;
}
