#include "MMseqsNode.h"
#include "Debug.h"
#include "FileUtil.h"
#include "Parameters.h"
#include "Util.h"
#include "FastSort.h"
#include "DBReader.h"
#include "itoa.h"
#include "Timer.h"
#include "BatchReader.h"

#include <cerrno>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <map>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#ifdef OPENMP
#include <omp.h>
#endif

unsigned int MMseqsNode::rank = 0;
unsigned int MMseqsNode::count = 1;
const size_t MMseqsNode::EXCHANGE_FILE_HEADER_BYTES;
static std::string runId;
static size_t sentExchangeSize = 0;
static size_t sentOtherNodeExchangeSize = 0;
static size_t exchangeIndexOpenCount = 0;
static std::map<std::string, std::vector<std::pair<char *, size_t> > > ownExchangeRecords;
static const unsigned int MAX_READ_POLLS = 400;

void MMseqsNode::init() {
    setvbuf(stdout, NULL, _IOLBF, 0);
    // Open MPI, MPICH/Intel MPI, then SLURM
    const char *rankVariables[][2] = { { "OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE" }, { "PMI_RANK", "PMI_SIZE" }, { "SLURM_PROCID", "SLURM_STEP_NUM_TASKS" } };
    for (size_t i = 0; i < sizeof(rankVariables) / sizeof(rankVariables[0]); i++) {
        const char *procId = getenv(rankVariables[i][0]);
        const char *taskCount = getenv(rankVariables[i][1]);
        if (procId != NULL && taskCount != NULL) {
            rank = strtoul(procId, NULL, 10);
            count = strtoul(taskCount, NULL, 10);
            break;
        }
    }
    const char *runVariables[] = { "SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_RESTART_COUNT", "OMPI_MCA_ess_base_jobid", "PMIX_NAMESPACE" };
    runId.clear();
    for (size_t i = 0; i < sizeof(runVariables) / sizeof(runVariables[0]); i++) {
        const char *value = getenv(runVariables[i]);
        runId += std::string(value != NULL ? value : "") + "\t";
    }
    runId += "\n";
    if (count == 0 || rank >= count) {
        Debug(Debug::ERROR) << "Invalid node rank " << rank << " for " << count << " nodes\n";
        EXIT(EXIT_FAILURE);
    }
}

void MMseqsNode::publishFile(const std::string &file, const std::string &content) {
    FileUtil::publishFile(file, content);
}

void MMseqsNode::sleepBeforePoll(unsigned int polls, unsigned int maxShift) {
    usleep(10000u << std::min(polls, maxShift));
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

static int openFile(const std::string &file, int flags = O_RDONLY) {
    int fd;
    do {
        fd = open(file.c_str(), flags, 0666);
    } while (fd < 0 && errno == EINTR);
    return fd;
}

static int openListedFile(const std::string &file, int flags = O_RDONLY) {
    int fd = openFile(file, flags);
    int openError = errno;
    if (fd < 0 && openError == ENOENT && isListed(file)) {
        return openFile(file, flags);
    }
    errno = openError;
    return fd;
}

// NFS caches a missing name, so poll on an empty file
static std::string readPolledFile(const std::string &file) {
    int flags = FileUtil::needsDirectoryRefresh(file) ? (O_RDONLY | O_CREAT) : O_RDONLY;
    int fd = openFile(file, flags);
    int openError = (fd < 0) ? errno : 0;
    time_t start = time(NULL);
    for (unsigned int polls = 0; openError == ENOENT && (flags & O_CREAT) != 0; polls++) {
        if (time(NULL) - start >= 60) {
            Debug(Debug::ERROR) << "Cannot open " << file << ": " << strerror(openError) << ". Its directory must be shared by all nodes\n";
            EXIT(EXIT_FAILURE);
        }
        MMseqsNode::sleepBeforePoll(polls, 7);
        FileUtil::refreshDirectory(FileUtil::dirName(file));
        fd = openFile(file, flags);
        openError = (fd < 0) ? errno : 0;
    }
    if (fd < 0) {
        if (openError == ESTALE || openError == ENOENT) {
            return std::string();
        }
        Debug(Debug::ERROR) << "Cannot open " << file << ": " << strerror(openError) << "\n";
        EXIT(EXIT_FAILURE);
    }
    struct stat sb;
    bool isStale = false;
    if (fstat(fd, &sb) != 0) {
        int statError = errno;
        isStale = statError == ESTALE;
        if (isStale == false) {
            Debug(Debug::ERROR) << "Cannot read " << file << ": " << strerror(statError) << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
    std::string content(isStale ? 0 : sb.st_size, '\0');
    int readError;
    if (MMseqsNode::tryPreadAll(fd, &content[0], content.size(), 0, readError) == false) {
        if (readError != 0 && readError != ESTALE) {
            Debug(Debug::ERROR) << "Cannot read " << file << ": " << strerror(readError) << "\n";
            EXIT(EXIT_FAILURE);
        }
        isStale = true;
    }
    close(fd);
    return isStale ? std::string() : content;
}

static const size_t WAIT_WARNING_SECONDS = 600;

MMseqsNode::WaitWarning::WaitWarning() : start(time(NULL)), nextSeconds(WAIT_WARNING_SECONDS) {}

void MMseqsNode::WaitWarning::reset() {
    start = time(NULL);
    nextSeconds = WAIT_WARNING_SECONDS;
}

void MMseqsNode::WaitWarning::check(const std::string &what) {
    if (static_cast<size_t>(time(NULL) - start) >= nextSeconds) {
        Debug(Debug::ERROR) << "Node " << rank << " waited " << nextSeconds << " s for " << what << ". Check that no node has died and that all nodes share this directory\n";
        nextSeconds *= 2;
    }
}

static void sleepInWait(unsigned int polls) {
    MMseqsNode::sleepBeforePoll(polls, std::min(8u, 4 + polls / 32));
}

std::string MMseqsNode::readFile(const std::string &file) {
    return FileUtil::readFile(file);
}

std::string MMseqsNode::waitForRunFile(const std::string &file) {
    std::string content = readPolledFile(file);
    WaitWarning warning;
    for (unsigned int polls = 0; content.compare(0, runId.size(), runId) != 0; polls++) {
        sleepInWait(polls);
        warning.check(file);
        content = readPolledFile(file);
    }
    return content.substr(runId.size());
}

void MMseqsNode::publishRunFile(const std::string &file, const std::string &content) {
    publishFile(file, runId + content);
}

void MMseqsNode::waitForRunFilesChange(const std::string &prefix, std::string &content) {
    WaitWarning warning;
    for (unsigned int polls = 4;; polls++) {
        std::string nodeContents;
        for (unsigned int node = 0; node < count; node++) {
            std::string nodeContent = readPolledFile(prefix + SSTR(node));
            if (nodeContent.compare(0, runId.size(), runId) == 0) {
                nodeContents.append(nodeContent, runId.size(), std::string::npos);
            }
        }
        if (nodeContents != content) {
            content.swap(nodeContents);
            return;
        }
        sleepBeforePoll(polls, 7);
        warning.check(prefix);
    }
}

std::string MMseqsNode::gatherRunFiles(const std::string &prefix) {
    std::vector<std::string> nodeContents(count);
#pragma omp parallel for schedule(dynamic, 1)
    for (unsigned int node = 0; node < count; node++) {
        std::string file = prefix + SSTR(node);
        nodeContents[node] = waitForRunFile(file);
        FileUtil::remove(file.c_str());
    }
    std::string content;
    for (unsigned int node = 0; node < count; node++) {
        content += nodeContents[node];
    }
    return content;
}

void MMseqsNode::publishLog(FILE *file, const std::string &log, size_t key, size_t size, size_t takenItemCount) {
    if (fflush(file) != 0 || fsync(fileno(file)) != 0) {
        Debug(Debug::ERROR) << "Cannot flush " << log << "\n";
        EXIT(EXIT_FAILURE);
    }
    publishRunFile(log + ".progress", SSTR(key) + "\t" + SSTR(size) + "\t" + SSTR(takenItemCount) + "\n");
}

bool MMseqsNode::readLogProgress(const std::string &log, size_t &key, size_t &size, size_t *takenItemCount) {
    std::string file = log + ".progress";
    std::string content = readPolledFile(file);
    if (content.compare(0, runId.size(), runId) != 0) {
        return false;
    }
    const char *columns[3];
    if (Util::getWordsOfLine((char *) content.c_str() + runId.size(), columns, 3) < 3) {
        Debug(Debug::ERROR) << "Cannot read " << file << "\n";
        EXIT(EXIT_FAILURE);
    }
    key = Util::fast_atoi<size_t>(columns[0]);
    size = Util::fast_atoi<size_t>(columns[1]);
    if (takenItemCount != NULL) {
        *takenItemCount = Util::fast_atoi<size_t>(columns[2]);
    }
    return true;
}

void MMseqsNode::waitForLog(const std::string &log, size_t &key, size_t &size) {
    WaitWarning warning;
    for (unsigned int polls = 0; readLogProgress(log, key, size) == false; polls++) {
        sleepInWait(polls);
        warning.check(log + ".progress");
    }
}

bool MMseqsNode::readLog(const std::string &log, size_t from, size_t to, char *data, bool wait) {
    size_t size = to - from;
    if (size == 0) {
        return true;
    }
    // NFS can expose the progress before the log
    for (unsigned int polls = 0; polls < MAX_READ_POLLS; polls++) {
        int fd = openListedFile(log);
        int openError = (fd < 0) ? errno : 0;
        if (fd < 0) {
            if (openError != ENOENT && openError != ESTALE) {
                Debug(Debug::ERROR) << "Cannot open " << log << ": " << strerror(openError) << "\n";
                EXIT(EXIT_FAILURE);
            }
        } else {
            int readError;
            bool isRead = tryPreadAll(fd, data, size, from, readError);
            int closeError = (close(fd) != 0) ? errno : 0;
            if ((readError != 0 && readError != ESTALE) || (closeError != 0 && closeError != ESTALE && closeError != EINTR)) {
                Debug(Debug::ERROR) << "Cannot read " << log << ": " << strerror(readError != 0 ? readError : closeError) << "\n";
                EXIT(EXIT_FAILURE);
            }
            if (isRead && closeError == 0) {
                return true;
            }
        }
        if (wait == false) {
            return false;
        }
        sleepBeforePoll(polls);
    }
    Debug(Debug::ERROR) << "Cannot read " << log << " bytes " << from << " to " << to << " after " << MAX_READ_POLLS << " polls\n";
    EXIT(EXIT_FAILURE);
}

void MMseqsNode::removeLog(const std::string &log) {
    FileUtil::remove(log.c_str());
    FileUtil::remove((log + ".progress").c_str());
}

struct ExchangeFileHeader {
    uint64_t magic;
    uint64_t exchangeFileNumber;
};
static_assert(sizeof(ExchangeFileHeader) == MMseqsNode::EXCHANGE_FILE_HEADER_BYTES, "Unexpected exchange file header layout");

static uint64_t getExchangeFileMagic() {
    uint64_t magic = 0x4d4d533258434831ULL;
    // mix in the run id
    for (size_t i = 0; i < runId.size(); i++) {
        magic = (magic ^ static_cast<unsigned char>(runId[i])) * 1099511628211ULL;
    }
    return magic;
}

void MMseqsNode::publishExchangeFile(FILE *file, const std::string &fileName, size_t size, const std::string &indexFile, const std::vector<uint64_t> &line) {
    if (fclose(file) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << fileName << "\n";
        EXIT(EXIT_FAILURE);
    }
    __sync_fetch_and_add(&sentExchangeSize, size);
    __sync_fetch_and_add(&sentOtherNodeExchangeSize, size);
    FILE *handle = FileUtil::openFileOrDie(indexFile.c_str(), "a", true);
    uint64_t magic = getExchangeFileMagic();
    if (fwrite(line.data(), sizeof(uint64_t), line.size() - 1, handle) != line.size() - 1
        || fwrite(&magic, sizeof(magic), 1, handle) != 1 || fflush(handle) != 0 || fclose(handle) != 0) {
        Debug(Debug::ERROR) << "Cannot write " << indexFile << "\n";
        EXIT(EXIT_FAILURE);
    }
}

size_t MMseqsNode::readExchangeIndexLines(const std::string &indexFile, size_t firstLine, size_t lineWords, std::vector<uint64_t> &lines) {
    size_t lineSize = lineWords * sizeof(uint64_t);
    int fd = openListedFile(indexFile);
    __sync_fetch_and_add(&exchangeIndexOpenCount, 1);
    int openError = (fd < 0) ? errno : 0;
    if (fd < 0) {
        if (openError == ENOENT || openError == ESTALE) {
            return 0;
        }
        Debug(Debug::ERROR) << "Cannot open " << indexFile << ": " << strerror(openError) << "\n";
        EXIT(EXIT_FAILURE);
    }
    ssize_t readSize;
    do {
        readSize = pread(fd, lines.data(), lines.size() * sizeof(uint64_t), firstLine * lineSize);
    } while (readSize < 0 && errno == EINTR);
    int readError = (readSize < 0) ? errno : 0;
    int closeError = (close(fd) != 0) ? errno : 0;
    if (readError == ESTALE || closeError == ESTALE || closeError == EINTR) {
        return 0;
    }
    if (readError != 0 || closeError != 0) {
        Debug(Debug::ERROR) << "Cannot read " << indexFile << ": " << strerror(readError != 0 ? readError : closeError) << "\n";
        EXIT(EXIT_FAILURE);
    }
    size_t lineCount = 0;
    while ((lineCount + 1) * lineSize <= static_cast<size_t>(readSize) && lines[lineCount * lineWords] == firstLine + lineCount
           && lines[(lineCount + 1) * lineWords - 1] == getExchangeFileMagic()) {
        lineCount++;
    }
    return lineCount;
}

size_t MMseqsNode::waitForExchangeIndexLines(const std::string &indexFile, size_t &nextLine, size_t lineWords, std::vector<uint64_t> &lines, bool wait) {
    WaitWarning warning;
    for (unsigned int polls = 0;; polls++) {
        size_t lineCount = readExchangeIndexLines(indexFile, nextLine, lineWords, lines);
        if (lineCount > 0 || wait == false) {
            nextLine += lineCount;
            return lineCount;
        }
        sleepInWait(polls);
        warning.check(indexFile);
    }
}

FILE *MMseqsNode::createExchangeFile(const std::string &fileName, uint64_t exchangeFileNumber, const char *mode) {
    FILE *file = FileUtil::openAndDelete(fileName.c_str(), mode);
    ExchangeFileHeader header = { getExchangeFileMagic(), exchangeFileNumber };
    if (fwrite(&header, sizeof(header), 1, file) != 1 || fflush(file) != 0) {
        Debug(Debug::ERROR) << "Cannot write " << fileName << "\n";
        EXIT(EXIT_FAILURE);
    }
    return file;
}

void MMseqsNode::writeExchangeFile(const std::string &fileName, uint64_t exchangeFileNumber, const char *data, size_t size, const std::string &indexFile,
                                   const std::vector<uint64_t> &line, int threads) {
    FILE *file = createExchangeFile(fileName, exchangeFileNumber);
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (size_t offset = 0; offset < size; offset += BatchReader::MAX_READ_REQUEST_SIZE) {
        FileUtil::pwriteAll(fileno(file), fileName.c_str(), data + offset, std::min(BatchReader::MAX_READ_REQUEST_SIZE, size - offset), EXCHANGE_FILE_HEADER_BYTES + offset);
    }
    publishExchangeFile(file, fileName, EXCHANGE_FILE_HEADER_BYTES + size, indexFile, line);
}

int MMseqsNode::tryOpenExchangeFile(const std::string &file, size_t expectedSize, uint64_t expectedNumber,
                                    size_t &observedSize, uint64_t &observedNumber, int &lastError, int openFlags) {
    observedSize = SIZE_MAX;
    observedNumber = UINT64_MAX;
    lastError = 0;
    int fd = openListedFile(file, O_RDONLY | O_CLOEXEC | openFlags);
    if (fd < 0) {
        lastError = errno;
        return -1;
    }
    struct stat fileStat;
    int statResult;
    do {
        statResult = fstat(fd, &fileStat);
    } while (statResult < 0 && errno == EINTR);
    if (statResult == 0) {
        observedSize = static_cast<size_t>(fileStat.st_size);
        if (observedSize == expectedSize) {
            ExchangeFileHeader header;
            ssize_t headerSize;
            do {
                headerSize = pread(fd, &header, sizeof(header), 0);
            } while (headerSize < 0 && errno == EINTR);
            if (headerSize == static_cast<ssize_t>(sizeof(header))) {
                observedNumber = header.exchangeFileNumber;
            }
            if (headerSize == static_cast<ssize_t>(sizeof(header)) && header.magic == getExchangeFileMagic()
                && header.exchangeFileNumber == expectedNumber) {
                return fd;
            }
            if (headerSize < 0) {
                lastError = errno;
            }
        }
    } else {
        lastError = errno;
    }
    close(fd);
    return -1;
}

int MMseqsNode::waitForExchangeFile(const std::string &file, size_t size, uint64_t exchangeFileNumber) {
    WaitWarning warning;
    for (unsigned int polls = 0;; polls++) {
        size_t observedSize;
        uint64_t observedNumber;
        int lastError;
        int fd = tryOpenExchangeFile(file, size, exchangeFileNumber, observedSize, observedNumber, lastError, O_RDWR);
        if (fd >= 0) {
            return fd;
        }
        if (lastError != 0 && lastError != ENOENT && lastError != ESTALE) {
            Debug(Debug::ERROR) << "Cannot open " << file << ": " << strerror(lastError) << "\n";
            EXIT(EXIT_FAILURE);
        }
        sleepInWait(polls);
        warning.check(file);
    }
}

bool MMseqsNode::tryPreadAll(int fd, char *data, size_t size, size_t offset, int &readError) {
    for (size_t done = 0; done < size;) {
        ssize_t count = pread(fd, data + done, size - done, offset + done);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            readError = (count < 0) ? errno : 0;
            return false;
        }
        done += count;
    }
    readError = 0;
    return true;
}

void MMseqsNode::removeOnceRead(const std::string &file) {
    publishRunFile(file + ".read." + SSTR(rank), "");
    if (isMaster()) {
        gatherRunFiles(file + ".read.");
        FileUtil::remove(file.c_str());
    }
}

std::string MMseqsNode::barrier(const std::string &file, const std::string &content) {
    publishFile(file + "." + SSTR(rank), runId + content);
    if (isMaster()) {
        publishFile(file, runId + gatherRunFiles(file + "."));
    }
    return waitForRunFile(file);
}

void MMseqsNode::addPhaseValue(std::string &content, const std::string &name, const std::string &value) {
    content += name + "\t" + SSTR(rank) + "\t" + value + "\n";
}

void MMseqsNode::addPhaseTime(std::string &content, const std::string &phase, Timer &timer) {
    addPhaseSeconds(content, phase, timer.getTimediff());
    timer.reset();
}

void MMseqsNode::addPhaseSeconds(std::string &content, const std::string &phase, double seconds) {
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%.3f s", seconds);
    addPhaseValue(content, "Time for " + phase, buffer);
}

void MMseqsNode::printPhaseValues(const std::string &values) {
    if (Debug::debugLevel < Debug::INFO) {
        return;
    }
    std::string prefix = (count > 1) ? "[node " + SSTR(rank) + "] " : "";
    std::string output;
    for (char *line = (char *) values.c_str(); *line != '\0'; line = Util::skipLine(line)) {
        const char *columns[3];
        Util::getFieldsOfLine(line, columns, 3);
        output += prefix;
        output.append(columns[0], columns[1] - 1);
        output += ": ";
        output.append(columns[2], strchr(columns[2], '\n'));
        output += '\n';
    }
    Debug(Debug::INFO) << output;
}

MMseqsNode::WorkQueue::WorkQueue(const std::string &name, size_t itemCount) : name(name), itemCount(itemCount), nextItem(rank), nextOrderedItem(0), itemsLeft(count, 0), nextNode((rank + 1) % count) {
    for (size_t owner = 0; owner < count; owner++) {
        if (owner != rank && owner < itemCount) {
            itemsLeft[owner] = (itemCount - 1 - owner) / count + 1;
        }
    }
}

bool MMseqsNode::WorkQueue::takeItem(size_t item) {
    std::string file = name + "." + SSTR(item);
    int fd = open(file.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0666);
    if (fd < 0 && errno == EEXIST) {
        return false;
    }
    if (fd < 0 || write(fd, &rank, sizeof(rank)) != static_cast<ssize_t>(sizeof(rank)) || close(fd) != 0) {
        Debug(Debug::ERROR) << "Cannot create " << file << "\n";
        EXIT(EXIT_FAILURE);
    }
    return true;
}

size_t MMseqsNode::WorkQueue::takeNextItem() {
    for (; nextItem < itemCount; nextItem += count) {
        if (takeItem(nextItem)) {
            nextItem += count;
            return nextItem - count;
        }
    }
    for (size_t emptyNodes = 0; emptyNodes < count;) {
        size_t owner = nextNode;
        nextNode = (nextNode + 1) % count;
        if (itemsLeft[owner] == 0) {
            emptyNodes++;
            continue;
        }
        emptyNodes = 0;
        itemsLeft[owner]--;
        if (takeItem(owner + itemsLeft[owner] * count)) {
            return owner + itemsLeft[owner] * count;
        }
        itemsLeft[owner] = 0;
    }
    return itemCount;
}

size_t MMseqsNode::WorkQueue::takeNextItemInOrder() {
    size_t item = nextOrderedItem;
    while (item < itemCount && takeItem(item) == false) {
        item++;
    }
    __atomic_store_n(&nextOrderedItem, std::min(item + 1, itemCount), __ATOMIC_RELAXED);
    return item;
}

size_t MMseqsNode::WorkQueue::getTakenItemCount() const {
    return __atomic_load_n(&nextOrderedItem, __ATOMIC_RELAXED);
}

unsigned int MMseqsNode::WorkQueue::getNode(size_t item) const {
    unsigned int node = count;
    std::string file = name + "." + SSTR(item);
    int fd = openListedFile(file);
    if (fd >= 0) {
        if (pread(fd, &node, sizeof(node), 0) != static_cast<ssize_t>(sizeof(node))) {
            node = count;
        }
        close(fd);
    }
    return node;
}

void MMseqsNode::WorkQueue::removeTakenItemFiles() {
    for (size_t item = rank; item < itemCount; item += count) {
        std::string file = name + "." + SSTR(item);
        if (FileUtil::fileExists(file.c_str())) {
            FileUtil::remove(file.c_str());
        }
    }
}

size_t MMseqsNode::getMemoryLimit(const Parameters &par, size_t localCopyBytes) {
    const size_t memoryLimit = Util::computeMemory(par.splitMemoryLimit);
    if (par.localDbName.empty() == false) {
        if (par.localDbDir.empty() || FileUtil::isDbName(par.localDbName) == false) {
            Debug(Debug::ERROR) << "--local-db-name requires --local-db-dir and a database name without a directory\n";
            EXIT(EXIT_FAILURE);
        }
        const std::string localDb = FileUtil::getLocalDbPath(par.localDbDir + "/" + par.localDbName, rank);
        localCopyBytes = FileUtil::getLocalDbMemoryBytes(localDb);
    }
    if (localCopyBytes > 0 && localCopyBytes >= memoryLimit) {
        Debug(Debug::ERROR) << "The sequence copy on tmpfs needs " << localCopyBytes << " bytes, the available memory is " << memoryLimit << " bytes\n";
        EXIT(EXIT_FAILURE);
    }
    return memoryLimit - localCopyBytes;
}

size_t MMseqsNode::getSmallestMemoryLimit(const std::string &file, size_t memoryLimit, int threads, const std::string &localDirectory, size_t *localNodes) {
    std::string localDevice;
    int fd = localDirectory.empty() ? -1 : open(localDirectory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    struct stat sb;
    if (fd >= 0 && getStorageKind(fd) == STORAGE_LOCAL_FLASH && fstat(fd, &sb) == 0) {
        char host[256] = "";
        gethostname(host, sizeof(host) - 1);
        localDevice = "\t" + std::string(host) + "\t" + SSTR(sb.st_dev) + "\n";
    }
    if (fd >= 0) {
        close(fd);
    }
    std::string nodeLimits = barrier(file, SSTR(memoryLimit) + ((threads > 0) ? "\t" + SSTR(threads) : "") + (localDevice.empty() ? "\n" : localDevice));
    size_t smallestMemoryLimit = memoryLimit;
    size_t sameDevice = 0;
    for (char *line = (char *) nodeLimits.c_str(); *line != '\0'; line = Util::skipLine(line)) {
        const char *columns[2];
        Util::getWordsOfLine(line, columns, 2);
        if (threads > 0 && Util::fast_atoi<int>(columns[1]) != threads) {
            Debug(Debug::ERROR) << "Run all nodes with the same --threads\n";
            EXIT(EXIT_FAILURE);
        }
        smallestMemoryLimit = std::min(smallestMemoryLimit, Util::fast_atoi<size_t>(columns[0]));
        char *end = Util::skipLine(line);
        sameDevice += localDevice.empty() == false && static_cast<size_t>(end - line) >= localDevice.size() && localDevice.compare(0, std::string::npos, end - localDevice.size(), localDevice.size()) == 0;
    }
    if (localNodes != NULL) {
        *localNodes = sameDevice;
    }
    return smallestMemoryLimit;
}

// column-major, a reader takes its two columns in one read
size_t MMseqsNode::writeExchangeTable(int fd, const std::string &fileName, const std::vector<size_t> &flushOffsets, size_t columns, const std::vector<size_t> &tail, size_t tableOffset) {
    size_t flushCount = flushOffsets.size() / columns;
    std::vector<size_t> table(flushOffsets.size());
    for (size_t flush = 0; flush < flushCount; flush++) {
        for (size_t column = 0; column < columns; column++) {
            table[column * flushCount + flush] = flushOffsets[flush * columns + column];
        }
    }
    table.insert(table.end(), tail.begin(), tail.end());
    FileUtil::pwriteAll(fd, fileName.c_str(), (const char *) table.data(), table.size() * sizeof(size_t), tableOffset);
    return tableOffset + table.size() * sizeof(size_t);
}

void MMseqsNode::readExchangeTableColumns(int fd, const std::string &fileName, size_t tableOffset, size_t flushCount, size_t column, std::vector<size_t> &offsets) {
    offsets.resize(2 * flushCount);
    FileUtil::preadAll(fd, fileName.c_str(), (char *) offsets.data(), offsets.size() * sizeof(size_t), tableOffset + column * flushCount * sizeof(size_t));
}

// buffer offsets; the own records [ownBegin, ownEnd) are not in the file
void MMseqsNode::addExchangeFlush(std::vector<size_t> &indexData, size_t &writtenSize, const std::vector<size_t> &nodeOffsets, size_t ownBegin, size_t ownEnd, size_t lineBegin, size_t lineEnd) {
    lineBegin = getExchangeFileOffset(lineBegin, ownBegin, ownEnd);
    lineEnd = getExchangeFileOffset(lineEnd, ownBegin, ownEnd);
    std::vector<size_t> offsets(count + 1);
    for (size_t node = 0; node <= count; node++) {
        offsets[node] = std::min(std::max(getExchangeFileOffset(nodeOffsets[node], ownBegin, ownEnd), lineBegin), lineEnd) - lineBegin;
        indexData.push_back(writtenSize + offsets[node]);
    }
    writtenSize += offsets[count];
    __sync_fetch_and_add(&sentExchangeSize, offsets[count]);
    __sync_fetch_and_add(&sentOtherNodeExchangeSize, offsets[count] - (offsets[rank + 1] - offsets[rank]));
}

void MMseqsNode::keepOwnExchangeRecords(const std::string &exchange, char *records, size_t size) {
    ownExchangeRecords[exchange].push_back(std::make_pair(records, size));
    __sync_fetch_and_add(&sentExchangeSize, size);
}

void MMseqsNode::addSentExchangeBytes(size_t size, size_t otherNodeSize) {
    __sync_fetch_and_add(&sentExchangeSize, size);
    __sync_fetch_and_add(&sentOtherNodeExchangeSize, otherNodeSize);
}

std::string MMseqsNode::closeExchange(FILE *file, const std::string &fileName, const std::string &indexFile, const std::vector<size_t> &indexData) {
    size_t flushCount = indexData.size() / (count + 1);
    size_t writtenSize = (flushCount > 0) ? indexData.back() : 0;
    writeExchangeTable(fileno(file), fileName, indexData, count + 1, std::vector<size_t>(), writtenSize);
    if (fclose(file) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << fileName << "\n";
        EXIT(EXIT_FAILURE);
    }
    return barrier(indexFile, SSTR(writtenSize) + "\t" + SSTR(flushCount) + "\n");
}

void MMseqsNode::openExchange(const std::string &index, const std::string &prefix, std::vector<ExchangeReader> &readers, std::vector<FILE *> &files) {
    files.assign(count, NULL);
    std::vector<size_t> writtenSizes(count);
    std::vector<size_t> flushCounts(count);
    char *line = (char *) index.c_str();
    for (size_t node = 0; node < count; node++, line = Util::skipLine(line)) {
        const char *columns[2];
        Util::getWordsOfLine(line, columns, 2);
        writtenSizes[node] = Util::fast_atoi<size_t>(columns[0]);
        flushCounts[node] = Util::fast_atoi<size_t>(columns[1]);
    }
    std::vector<std::vector<size_t> > flushOffsets(count);
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t node = 0; node < count; node++) {
        if (flushCounts[node] == 0) {
            continue;
        }
        std::string file = prefix + "." + SSTR(node);
        // writable to punch holes
        files[node] = FileUtil::openFileOrDie(file.c_str(), "r+", true);
        readExchangeTableColumns(fileno(files[node]), file, writtenSizes[node], flushCounts[node], rank, flushOffsets[node]);
    }
    for (size_t node = 0; node < count; node++) {
        size_t nodeReaders = readers.size();
        for (size_t flush = 0; flush < flushCounts[node]; flush++) {
            size_t pos = flushOffsets[node][flush];
            size_t end = flushOffsets[node][flushCounts[node] + flush];
            if (pos < end) {
                ExchangeReader reader = { files[node], pos, end, std::vector<char>(), 0, 0, 0, 0, NULL, pos };
                readers.push_back(reader);
            }
        }
        if (files[node] != NULL && readers.size() == nodeReaders) {
            if (fclose(files[node]) != 0) {
                Debug(Debug::ERROR) << "Cannot close " << prefix << "." << node << "\n";
                EXIT(EXIT_FAILURE);
            }
            files[node] = NULL;
        }
    }
    std::map<std::string, std::vector<std::pair<char *, size_t> > >::iterator own = ownExchangeRecords.find(prefix);
    if (own != ownExchangeRecords.end()) {
        for (size_t i = 0; i < own->second.size(); i++) {
            ExchangeReader reader = { NULL, 0, own->second[i].second, std::vector<char>(), 0, 0, 0, 0, own->second[i].first, 0 };
            readers.push_back(reader);
        }
        ownExchangeRecords.erase(own);
    }
}

static size_t getUnreadSize(const MMseqsNode::ExchangeReader &reader) {
    return reader.end - reader.pos;
}

static const char *getReaderData(const MMseqsNode::ExchangeReader &reader) {
    return (reader.ownRecords != NULL) ? reader.ownRecords : reader.buffer.data();
}

static const size_t PUNCH_HOLE_ALIGNMENT = 1024 * 1024;
static const size_t EXCHANGE_BUFFER_SIZE = 1024 * 1024;

static void fillExchangeReader(MMseqsNode::ExchangeReader &reader, size_t size, size_t bufferSize, const std::string &prefix) {
    if (reader.ownRecords != NULL) {
        reader.bufferEnd = std::max(reader.bufferEnd, std::min(reader.end, reader.bufferPos + std::max(size, bufferSize)));
        reader.pos = reader.bufferEnd;
        return;
    }
    if (reader.bufferEnd - reader.bufferPos >= size || getUnreadSize(reader) == 0) {
        return;
    }
    size_t punchEnd = (reader.pos - reader.bufferEnd + reader.bufferPos) / PUNCH_HOLE_ALIGNMENT * PUNCH_HOLE_ALIGNMENT;
    if (punchEnd >= reader.punchEnd + 1024 * 1024 * 1024) {
        FileUtil::punchHole(fileno(reader.file), reader.punchEnd, punchEnd - reader.punchEnd);
        reader.punchEnd = punchEnd;
    }
    if (reader.bufferEnd > reader.bufferPos) {
        memmove(reader.buffer.data(), reader.buffer.data() + reader.bufferPos, reader.bufferEnd - reader.bufferPos);
    }
    reader.bufferEnd -= reader.bufferPos;
    reader.recordEnd -= std::min(reader.recordEnd, reader.bufferPos);
    reader.lastRecord -= std::min(reader.lastRecord, reader.bufferPos);
    reader.bufferPos = 0;
    size_t readerBufferSize = std::min(std::max(size, bufferSize), reader.bufferEnd + getUnreadSize(reader));
    if (reader.buffer.size() < readerBufferSize) {
        reader.buffer.resize(readerBufferSize);
    }
    std::vector<std::pair<size_t, size_t> > reads;
    while (reader.bufferEnd < reader.buffer.size() && getUnreadSize(reader) > 0) {
        size_t readSize = std::min(std::min(reader.buffer.size() - reader.bufferEnd, reader.end - reader.pos), BatchReader::MAX_READ_REQUEST_SIZE);
        reads.push_back(std::make_pair(reader.pos, reader.bufferEnd));
        reader.pos += readSize;
        reader.bufferEnd += readSize;
    }
    size_t readCount = reads.size();
    reads.push_back(std::make_pair(reader.pos, reader.bufferEnd));
#pragma omp parallel for schedule(dynamic, 1) if (omp_in_parallel() == 0)
    for (size_t i = 0; i < readCount; i++) {
        FileUtil::preadAll(fileno(reader.file), prefix.c_str(), reader.buffer.data() + reads[i].second, reads[i + 1].second - reads[i].second, reads[i].first);
    }
}

static void releaseExchangeReader(MMseqsNode::ExchangeReader &reader) {
    if (getUnreadSize(reader) == 0 && reader.bufferEnd > 0 && reader.bufferPos == reader.bufferEnd) {
        if (reader.ownRecords != NULL) {
            Util::releasePages(reader.ownRecords, reader.end);
            free(reader.ownRecords);
            reader.ownRecords = NULL;
        } else {
            FileUtil::punchHole(fileno(reader.file), reader.punchEnd, reader.end - reader.punchEnd);
            std::vector<char>().swap(reader.buffer);
        }
        reader.bufferPos = 0;
        reader.bufferEnd = 0;
        reader.recordEnd = 0;
        reader.lastRecord = 0;
    }
}

size_t MMseqsNode::readExchangeRecordsUpTo(std::vector<ExchangeReader> &readers, size_t bufferMemory, size_t (*recordSize)(const char *, size_t),
                                           bool (*compare)(const ClusterPair &, const ClusterPair &), const ClusterPair &last,
                                           std::vector<char> &records, std::vector<size_t> *recordOffsets, const std::string &prefix) {
    size_t bufferSize = std::max(EXCHANGE_BUFFER_SIZE, bufferMemory / 2 / std::max((size_t) 1, readers.size()));
    size_t threads = static_cast<size_t>(Parameters::getInstance().threads);
    bool isClusterPair = recordSize == NULL;
    std::vector<char> needsRead(readers.size());
    for (size_t i = 0; i < readers.size(); i++) {
        const ExchangeReader &reader = readers[i];
        needsRead[i] = getUnreadSize(reader) > 0 && (reader.bufferEnd - reader.bufferPos < bufferSize / 2 || reader.recordEnd == reader.bufferPos);
    }
#pragma omp parallel for schedule(dynamic, 1) if (readers.size() >= threads)
    for (size_t i = 0; i < readers.size(); i++) {
        if (needsRead[i]) {
            fillExchangeReader(readers[i], bufferSize, bufferSize, prefix);
        }
    }
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < readers.size(); i++) {
        ExchangeReader &reader = readers[i];
        if (needsRead[i] == false) {
            continue;
        }
        for (size_t fillSize = bufferSize; ; fillSize *= 2) {
            fillExchangeReader(reader, fillSize, fillSize, prefix);
            const char *data = getReaderData(reader);
            if (isClusterPair && reader.bufferEnd - reader.recordEnd >= sizeof(ClusterPair)) {
                reader.recordEnd += (reader.bufferEnd - reader.recordEnd) / sizeof(ClusterPair) * sizeof(ClusterPair);
                reader.lastRecord = reader.recordEnd - sizeof(ClusterPair);
            }
            for (size_t size; isClusterPair == false && (size = recordSize(data + reader.recordEnd, reader.bufferEnd - reader.recordEnd)) > 0; reader.recordEnd += size) {
                reader.lastRecord = reader.recordEnd;
            }
            if (reader.recordEnd > reader.bufferPos || getUnreadSize(reader) == 0) {
                break;
            }
        }
        if (getUnreadSize(reader) == 0 && reader.recordEnd != reader.bufferEnd) {
            Debug(Debug::ERROR) << "Cannot read " << prefix << ": a record ends after its part\n";
            EXIT(EXIT_FAILURE);
        }
    }
    ClusterPair lastReadRecord = last;
    for (size_t i = 0; i < readers.size(); i++) {
        if (getUnreadSize(readers[i]) > 0) {
            ClusterPair readerLast;
            memcpy(&readerLast, getReaderData(readers[i]) + readers[i].lastRecord, sizeof(ClusterPair));
            if (compare(readerLast, lastReadRecord)) {
                lastReadRecord = readerLast;
            }
        }
    }
    std::vector<size_t> readerEnds(readers.size());
    std::vector<size_t> readerByteOffsets(readers.size() + 1, 0);
    std::vector<size_t> readerRecordOffsets(readers.size() + 1, 0);
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < readers.size(); i++) {
        const ExchangeReader &reader = readers[i];
        const char *data = getReaderData(reader);
        size_t pos = reader.bufferPos;
        size_t recordCount = 0;
        if (isClusterPair) {
            const ClusterPair *pairs = (const ClusterPair *) (data + reader.bufferPos);
            recordCount = std::upper_bound(pairs, pairs + (reader.recordEnd - reader.bufferPos) / sizeof(ClusterPair), lastReadRecord, compare) - pairs;
            pos += recordCount * sizeof(ClusterPair);
        }
        for (ClusterPair pair; isClusterPair == false && pos < reader.recordEnd; pos += recordSize(data + pos, reader.recordEnd - pos), recordCount++) {
            memcpy(&pair, data + pos, sizeof(ClusterPair));
            if (compare(lastReadRecord, pair)) {
                break;
            }
        }
        readerEnds[i] = pos;
        readerByteOffsets[i + 1] = pos - reader.bufferPos;
        readerRecordOffsets[i + 1] = recordCount;
    }
    for (size_t i = 0; i < readers.size(); i++) {
        readerByteOffsets[i + 1] += readerByteOffsets[i];
        readerRecordOffsets[i + 1] += readerRecordOffsets[i];
    }
    if (records.size() < readerByteOffsets.back()) {
        records.resize(readerByteOffsets.back());
    }
    if (recordOffsets != NULL) {
        recordOffsets->resize(readerRecordOffsets.back() + 1);
        recordOffsets->back() = readerByteOffsets.back();
    }
    size_t threadCopySize = std::max(EXCHANGE_BUFFER_SIZE, readerByteOffsets.back() / threads + 1);
    size_t threadCopyCount = (readerByteOffsets.back() + threadCopySize - 1) / threadCopySize;
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t threadCopy = 0; threadCopy < threadCopyCount; threadCopy++) {
        size_t copyBegin = threadCopy * threadCopySize;
        size_t copyEnd = std::min(readerByteOffsets.back(), copyBegin + threadCopySize);
        for (size_t i = std::upper_bound(readerByteOffsets.begin(), readerByteOffsets.end(), copyBegin) - readerByteOffsets.begin() - 1; i < readers.size() && readerByteOffsets[i] < copyEnd; i++) {
            size_t from = std::max(copyBegin, readerByteOffsets[i]);
            size_t to = std::min(copyEnd, readerByteOffsets[i + 1]);
            memcpy(records.data() + from, getReaderData(readers[i]) + readers[i].bufferPos + (from - readerByteOffsets[i]), to - from);
        }
    }
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < readers.size(); i++) {
        ExchangeReader &reader = readers[i];
        if (recordOffsets != NULL) {
            size_t pos = readerByteOffsets[i];
            for (size_t record = readerRecordOffsets[i]; record < readerRecordOffsets[i + 1]; record++) {
                (*recordOffsets)[record] = pos;
                pos += isClusterPair ? sizeof(ClusterPair) : recordSize(records.data() + pos, readerByteOffsets[i + 1] - pos);
            }
        }
        reader.bufferPos = readerEnds[i];
        releaseExchangeReader(reader);
    }
    return readerRecordOffsets.back();
}

static bool isLongerLength(const std::pair<size_t, size_t> &bin, size_t length) {
    return bin.first > length;
}

size_t MMseqsNode::readExchangeCompressionUnits(std::vector<ExchangeReader> &readers, const std::vector<std::vector<std::pair<size_t, size_t> > > &readerLengthOffsets, size_t bufferMemory, size_t maxCompressionUnitSize,
                                                size_t bufferBegin, size_t bufferEnd, char *buffer, std::vector<std::pair<size_t, size_t> > *compressionUnits, const std::string &prefix) {
    if (readers.empty()) {
        return SIZE_MAX;
    }
    size_t bufferSize = std::max(4 * sizeof(uint64_t) + maxCompressionUnitSize, bufferMemory / std::max((size_t) 1, readers.size()));
    size_t nextOffset = SIZE_MAX;
#pragma omp parallel reduction(min:nextOffset)
    {
        std::vector<std::pair<size_t, size_t> > threadCompressionUnits;
        ZSTD_DCtx *dctx = ZSTD_createDCtx();
        if (dctx == NULL) {
            Debug(Debug::ERROR) << "ZSTD_createDCtx() error\n";
            EXIT(EXIT_FAILURE);
        }
#pragma omp for schedule(dynamic, 1) nowait
        for (size_t i = 0; i < readers.size(); i++) {
            ExchangeReader &reader = readers[i];
            // consecutive units mostly share their length
            std::vector<std::pair<size_t, size_t> >::const_iterator lengthOffset = readerLengthOffsets[i].end();
            uint64_t compressionUnit[4];
            for (;;) {
                fillExchangeReader(reader, sizeof(compressionUnit), bufferSize, prefix);
                if (reader.bufferEnd - reader.bufferPos < sizeof(compressionUnit)) {
                    break;
                }
                memcpy(compressionUnit, getReaderData(reader) + reader.bufferPos, sizeof(compressionUnit));
                if (lengthOffset == readerLengthOffsets[i].end() || lengthOffset->first != compressionUnit[0]) {
                    lengthOffset = std::lower_bound(readerLengthOffsets[i].begin(), readerLengthOffsets[i].end(), compressionUnit[0], isLongerLength);
                }
                size_t offset = lengthOffset->second + compressionUnit[1];
                if (offset >= bufferEnd) {
                    nextOffset = std::min(nextOffset, offset);
                    break;
                }
                fillExchangeReader(reader, sizeof(compressionUnit) + compressionUnit[3], bufferSize, prefix);
                if (reader.bufferEnd - reader.bufferPos < sizeof(compressionUnit) + compressionUnit[3]) {
                    Debug(Debug::ERROR) << "Cannot read " << prefix << ": a record ends after its part\n";
                    EXIT(EXIT_FAILURE);
                }
                const char *data = getReaderData(reader) + reader.bufferPos + sizeof(compressionUnit);
                if (compressionUnit[3] == compressionUnit[2]) {
                    memcpy(buffer + (offset - bufferBegin), data, compressionUnit[2]);
                } else if (ZSTD_decompressDCtx(dctx, buffer + (offset - bufferBegin), compressionUnit[2], data, compressionUnit[3]) != compressionUnit[2]) {
                    Debug(Debug::ERROR) << "Cannot decompress a record of " << prefix << "\n";
                    EXIT(EXIT_FAILURE);
                }
                if (compressionUnits != NULL) {
                    threadCompressionUnits.push_back(std::make_pair(offset, compressionUnit[2]));
                }
                reader.bufferPos += sizeof(compressionUnit) + compressionUnit[3];
            }
            releaseExchangeReader(reader);
        }
        ZSTD_freeDCtx(dctx);
        if (compressionUnits != NULL) {
#pragma omp critical
            compressionUnits->insert(compressionUnits->end(), threadCompressionUnits.begin(), threadCompressionUnits.end());
        }
    }
    return nextOffset;
}

void MMseqsNode::closeExchangeFiles(std::vector<FILE *> &files, const std::string &prefix) {
    for (size_t node = 0; node < files.size(); node++) {
        if (files[node] != NULL && fclose(files[node]) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << prefix << "." << node << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
}

static const size_t BALANCE_BINS_PER_NODE = 1024;

static size_t getBinKey(size_t keyEnd, size_t bin) {
    size_t bins = BALANCE_BINS_PER_NODE * MMseqsNode::count;
    return keyEnd / bins * bin + keyEnd % bins * bin / bins;
}

static void addBinOffsets(const ClusterPair *pairs, size_t pairCount, bool (*compare)(const ClusterPair &, const ClusterPair &), size_t keyEnd, std::vector<size_t> &binOffsets) {
    size_t bins = BALANCE_BINS_PER_NODE * MMseqsNode::count;
    size_t flushBegin = binOffsets.size();
    binOffsets.resize(flushBegin + bins + 1);
#pragma omp parallel for schedule(static)
    for (size_t bin = 0; bin < bins; bin++) {
        DBKeyType binKey = static_cast<DBKeyType>(getBinKey(keyEnd, bin));
        ClusterPair binFirstPair = { binKey, binKey };
        binOffsets[flushBegin + bin] = std::lower_bound(pairs, pairs + pairCount, binFirstPair, compare) - pairs;
    }
    binOffsets[flushBegin + bins] = pairCount;
}

static size_t writePairs(ClusterPair *pairs, size_t pairCount, bool (*compare)(const ClusterPair &, const ClusterPair &), const std::vector<size_t> &nodeKeyOffsets,
                         FILE *file, const std::string &fileName, std::vector<size_t> &indexData, size_t &writtenSize, size_t ownPairLimit) {
    std::vector<size_t> nodeOffsets(MMseqsNode::count + 1, pairCount * sizeof(ClusterPair));
    for (size_t node = 0; node < MMseqsNode::count; node++) {
        ClusterPair nodeFirstPair = { static_cast<DBKeyType>(nodeKeyOffsets[node]), static_cast<DBKeyType>(nodeKeyOffsets[node]) };
        nodeOffsets[node] = (std::lower_bound(pairs, pairs + pairCount, nodeFirstPair, compare) - pairs) * sizeof(ClusterPair);
    }
    size_t ownBegin = nodeOffsets[MMseqsNode::rank] / sizeof(ClusterPair);
    size_t ownEnd = ownBegin + std::min(nodeOffsets[MMseqsNode::rank + 1] / sizeof(ClusterPair) - ownBegin, ownPairLimit);
    if (fwrite(pairs, sizeof(ClusterPair), ownBegin, file) != ownBegin || fwrite(pairs + ownEnd, sizeof(ClusterPair), pairCount - ownEnd, file) != pairCount - ownEnd) {
        Debug(Debug::ERROR) << "Cannot write " << fileName << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (ownBegin > 0) {
        memmove(pairs, pairs + ownBegin, (ownEnd - ownBegin) * sizeof(ClusterPair));
    }
    MMseqsNode::addExchangeFlush(indexData, writtenSize, nodeOffsets, ownBegin * sizeof(ClusterPair), ownEnd * sizeof(ClusterPair), 0, pairCount * sizeof(ClusterPair));
    return ownEnd - ownBegin;
}

MMseqsNode::ClusterPairWriter::ClusterPairWriter(const std::string &exchange, size_t pairCapacity, bool (*compare)(const ClusterPair &, const ClusterPair &), const std::vector<size_t> &nodeKeyOffsets,
                                                 std::vector<size_t> *balancedKeyOffsets)
    : exchange(exchange), fileName(exchange + "." + SSTR(rank)), file(FileUtil::openAndDelete(fileName.c_str(), "w")), pairs(static_cast<ClusterPair *>(malloc(pairCapacity * sizeof(ClusterPair)))),
      pairCapacity(pairCapacity), pairCount(0), compare(compare), nodeKeyOffsets(nodeKeyOffsets), balancedKeyOffsets(balancedKeyOffsets), writtenSize(0), keptPairCount(0) {
    Util::checkAllocation(pairs, "Can not allocate pairs memory in ClusterPairWriter");
}

MMseqsNode::ClusterPairWriter::~ClusterPairWriter() {
    free(pairs);
}

void MMseqsNode::ClusterPairWriter::flush() {
    SORT_PARALLEL(pairs, pairs + pairCount, compare);
    if (balancedKeyOffsets != NULL) {
        addBinOffsets(pairs, pairCount, compare, nodeKeyOffsets[count], binOffsets);
        if (fwrite(pairs, sizeof(ClusterPair), pairCount, file) != pairCount) {
            Debug(Debug::ERROR) << "Cannot write " << fileName << "\n";
            EXIT(EXIT_FAILURE);
        }
        pairCount = 0;
        return;
    }
    size_t ownPairCount = writePairs(pairs, pairCount, compare, nodeKeyOffsets, file, fileName, indexData, writtenSize, pairCapacity - keptPairCount);
    if (ownPairCount > 0) {
        ClusterPair *ownPairs = static_cast<ClusterPair *>(malloc(ownPairCount * sizeof(ClusterPair)));
        Util::checkAllocation(ownPairs, "Can not allocate own pairs memory in ClusterPairWriter");
        memcpy(ownPairs, pairs, ownPairCount * sizeof(ClusterPair));
        keepOwnExchangeRecords(exchange, (char *) ownPairs, ownPairCount * sizeof(ClusterPair));
        keptPairCount += ownPairCount;
    }
    pairCount = 0;
}

std::string MMseqsNode::ClusterPairWriter::close(const uint64_t *singletonBits, size_t firstKey, size_t keyCount) {
    SORT_PARALLEL(pairs, pairs + pairCount, compare);
    if (balancedKeyOffsets != NULL) {
        addBinOffsets(pairs, pairCount, compare, nodeKeyOffsets[count], binOffsets);
        size_t bins = BALANCE_BINS_PER_NODE * count;
        size_t keyEnd = nodeKeyOffsets[count];
        std::vector<uint64_t> binLines(bins, 0);
#pragma omp parallel for schedule(static)
        for (size_t bin = 0; bin < bins; bin++) {
            for (size_t binOffset = bin; binOffset < binOffsets.size(); binOffset += bins + 1) {
                binLines[bin] += binOffsets[binOffset + 1] - binOffsets[binOffset];
            }
            size_t from = std::max(firstKey, getBinKey(keyEnd, bin));
            size_t to = std::min(firstKey + keyCount, getBinKey(keyEnd, bin + 1));
            if (singletonBits != NULL && from < to) {
                binLines[bin] += Util::countSetBits(singletonBits, from - firstKey, to - firstKey);
            }
        }
        std::string binFile = exchange + ".bins";
        publishRunFile(binFile + "." + SSTR(rank), std::string((const char *) binLines.data(), bins * sizeof(uint64_t)));
        std::vector<uint64_t> nodeBinOffsets(count + 1, bins);
        if (isMaster()) {
            std::fill(binLines.begin(), binLines.end(), 0);
#pragma omp parallel for schedule(dynamic, 1)
            for (unsigned int node = 0; node < count; node++) {
                std::string file = binFile + "." + SSTR(node);
                std::string nodeBinLines = waitForRunFile(file);
                FileUtil::remove(file.c_str());
#pragma omp critical
                for (size_t bin = 0; bin < bins; bin++) {
                    uint64_t lines;
                    memcpy(&lines, nodeBinLines.data() + bin * sizeof(uint64_t), sizeof(uint64_t));
                    binLines[bin] += lines;
                }
            }
            uint64_t totalLines = 0;
            for (size_t bin = 0; bin < bins; bin++) {
                totalLines += binLines[bin];
            }
            nodeBinOffsets[0] = 0;
            size_t bin = 0;
            uint64_t linesBefore = 0;
            for (size_t node = 1; node < count; node++) {
                while (linesBefore * count < totalLines * node) {
                    linesBefore += binLines[bin++];
                }
                nodeBinOffsets[node] = bin;
            }
            publishRunFile(binFile, std::string((const char *) nodeBinOffsets.data(), nodeBinOffsets.size() * sizeof(uint64_t)));
        }
        std::string cuts = waitForRunFile(binFile);
        removeOnceRead(binFile);
        memcpy(nodeBinOffsets.data(), cuts.data(), nodeBinOffsets.size() * sizeof(uint64_t));
        balancedKeyOffsets->resize(count + 1);
        for (size_t node = 0; node <= count; node++) {
            (*balancedKeyOffsets)[node] = getBinKey(keyEnd, nodeBinOffsets[node]);
        }
        std::vector<size_t> nodeOffsets(count + 1);
        for (size_t flushBegin = 0; flushBegin + bins + 1 < binOffsets.size(); flushBegin += bins + 1) {
            for (size_t node = 0; node <= count; node++) {
                nodeOffsets[node] = binOffsets[flushBegin + nodeBinOffsets[node]] * sizeof(ClusterPair);
            }
            addExchangeFlush(indexData, writtenSize, nodeOffsets, 0, 0, 0, nodeOffsets[count]);
        }
        std::vector<size_t>().swap(binOffsets);
    }
    size_t ownPairCount = 0;
    if (pairCount > 0) {
        ownPairCount = writePairs(pairs, pairCount, compare, (balancedKeyOffsets != NULL) ? *balancedKeyOffsets : nodeKeyOffsets, file, fileName, indexData, writtenSize, pairCount);
        pairCount = 0;
    }
    if (ownPairCount > 0) {
        pairs = static_cast<ClusterPair *>(realloc(pairs, ownPairCount * sizeof(ClusterPair)));
        Util::checkAllocation(pairs, "Can not shrink pairs memory in ClusterPairWriter");
        keepOwnExchangeRecords(exchange, (char *) pairs, ownPairCount * sizeof(ClusterPair));
    } else {
        Util::releasePages(pairs, pairCapacity * sizeof(ClusterPair));
        free(pairs);
    }
    pairs = NULL;
    return closeExchange(file, fileName, exchange + ".index", indexData);
}

static const size_t CLUSTER_SPLIT_MEMORY_DIVISOR = 16;
static const size_t RECEIVED_MEMORY_DIVISOR = 64;

static void writeKeyBits(const std::string &bitsFile, const uint64_t *bits, size_t firstKey, size_t keyCount) {
    FILE *file = FileUtil::openAndDelete(bitsFile.c_str(), "w");
    uint64_t header[2] = { firstKey, keyCount };
    size_t words = (keyCount + 63) / 64;
    if (fwrite(header, sizeof(uint64_t), 2, file) != 2 || fwrite(bits, sizeof(uint64_t), words, file) != words
        || fclose(file) != 0) {
        Debug(Debug::ERROR) << "Cannot write " << bitsFile << "\n";
        EXIT(EXIT_FAILURE);
    }
}

static bool readEntryKeys(DBReader<DBKeyType> &cluDb, DBReader<DBKeyType>::Index &entry, size_t memoryLimit, std::vector<char> &buffer, std::vector<DBKeyType> &keys) {
    buffer.resize(std::min(memoryLimit / CLUSTER_SPLIT_MEMORY_DIVISOR, BatchReader::MAX_INFLIGHT_READ_SIZE));
    // without the trailing '\0'
    size_t size = std::min(buffer.size() - 1, entry.length - 1);
    if (size == 0) {
        std::vector<char>().swap(buffer);
        return false;
    }
    cluDb.readData(entry.offset, size, buffer.data());
    size_t keySize = size;
    while (buffer[keySize - 1] != '\n') {
        keySize--;
    }
    buffer[keySize] = '\0';
    keys.clear();
    char keyBuffer[255];
    for (char *data = buffer.data(); *data != '\0'; data = Util::skipLine(data)) {
        Util::parseKey(data, keyBuffer);
        keys.push_back(Util::fast_atoi<DBKeyType>(keyBuffer));
    }
    entry.offset += keySize;
    entry.length -= keySize;
    return true;
}

// members with an entry have the repKey DB_KEY_INVALID
static void sendMembersWithoutEntry(MMseqsNode::ClusterPairWriter &pairWriter, const ClusterPair *pairs, size_t pairCount, uint64_t *representativeBits, size_t firstKey) {
    for (size_t i = 0; i < pairCount; i++) {
        if (pairs[i].repKey != DB_KEY_INVALID) {
            pairWriter.add(pairs[i].repKey, pairs[i].memberKey);
            representativeBits[(pairs[i].memberKey - firstKey) / 64] &= ~(1ULL << ((pairs[i].memberKey - firstKey) % 64));
        }
    }
}

// empty if its first entry is larger than splitSize
static size_t findSplitEnd(DBReader<DBKeyType> &cluDb, size_t splitBegin, size_t splitOffset, size_t splitSize, size_t splitEntries, size_t entryEnd) {
    size_t sizeEnd = (splitOffset + splitSize >= cluDb.getDataSize()) ? cluDb.getSize() : cluDb.findEntryByOffset(splitOffset + splitSize + 1) - 1;
    return std::min(std::min(sizeEnd, splitBegin + splitEntries), entryEnd);
}

std::string MMseqsNode::sendClusterPairs(const std::string &clu, bool withRepresentative, bool (*compare)(const ClusterPair &, const ClusterPair &), const std::vector<size_t> &nodeKeyOffsets,
                                         const std::string &exchange, const std::string &receivedIndex, const std::string &receivedPrefix,
                                         uint64_t *representativeBits, const std::vector<size_t> &cluKeyOffsets, size_t firstKey, size_t keyCount,
                                         std::vector<size_t> *balancedKeyOffsets) {
    std::vector<ExchangeReader> receivedReaders;
    std::vector<FILE *> receivedFiles;
    if (receivedPrefix.empty() == false) {
        openExchange(receivedIndex, receivedPrefix, receivedReaders, receivedFiles);
    }
    size_t memoryLimit = getMemoryLimit(Parameters::getInstance());
    ClusterPairWriter pairWriter(exchange, memoryLimit / 4 / sizeof(ClusterPair), compare, nodeKeyOffsets, balancedKeyOffsets);

    std::vector<DBKeyType> repKeys;
    std::vector<size_t> pairOffsets;
    std::vector<char> received;
    std::vector<char> entryBuffer;
    std::vector<DBKeyType> streamedKeys;
    for (size_t node = 0; node < count; node++) {
        if (cluKeyOffsets.empty() ? node != rank : std::max(firstKey, cluKeyOffsets[node]) >= std::min(firstKey + keyCount, cluKeyOffsets[node + 1])) {
            continue;
        }
        DBReader<DBKeyType> cluDb((clu + "." + SSTR(node)).c_str(), (clu + ".index." + SSTR(node)).c_str(), Parameters::getInstance().threads,
                                  DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
        cluDb.open(DBReader<DBKeyType>::NOSORT);
        size_t entryBegin = cluKeyOffsets.empty() ? 0 : cluDb.findEntryByKey(firstKey);
        size_t entryEnd = cluKeyOffsets.empty() ? cluDb.getSize() : cluDb.findEntryByKey(firstKey + keyCount);
        size_t splitSize = memoryLimit / CLUSTER_SPLIT_MEMORY_DIVISOR;
        size_t splitEntries = std::max((size_t) 1, memoryLimit / CLUSTER_SPLIT_MEMORY_DIVISOR / (2 * sizeof(uint64_t) + 2 * sizeof(size_t)));
        std::vector<DBKeyType> entryKeys;
        DBReader<DBKeyType>::Index streamedEntry;
        size_t splitOffset = (entryBegin < entryEnd) ? cluDb.getIndexEntry(entryBegin).offset : 0;
        for (size_t splitBegin = entryBegin, splitEnd = findSplitEnd(cluDb, entryBegin, splitOffset, splitSize, splitEntries, entryEnd), nextSplitEnd; splitBegin < entryEnd; splitBegin = splitEnd, splitEnd = nextSplitEnd) {
            bool isStreamed = splitEnd == splitBegin;
            if (isStreamed) {
                streamedEntry = cluDb.getIndexEntry(splitBegin);
                entryKeys.assign(1, streamedEntry.id);
                splitOffset += streamedEntry.length;
                splitEnd++;
            } else {
                cluDb.loadData(splitBegin, splitEnd);
                entryKeys.resize(splitEnd - splitBegin);
#pragma omp parallel for schedule(static) reduction(+:splitOffset)
                for (size_t i = splitBegin; i < splitEnd; i++) {
                    entryKeys[i - splitBegin] = cluDb.getDbKey(i);
                    splitOffset += cluDb.getEntryLen(i);
                }
            }
            nextSplitEnd = findSplitEnd(cluDb, splitEnd, splitOffset, splitSize, splitEntries, entryEnd);
            if (nextSplitEnd > splitEnd) {
                cluDb.prefetchData(splitEnd, nextSplitEnd);
            }
            repKeys = entryKeys;
            if (representativeBits != NULL) {
                for (size_t i = 0; i < entryKeys.size(); i++) {
                    representativeBits[(entryKeys[i] - firstKey) / 64] &= ~(1ULL << ((entryKeys[i] - firstKey) % 64));
                }
            }
            ClusterPair splitLast = { entryKeys.back(), entryKeys.back() };
            size_t receivedCount;
            while ((receivedCount = readExchangeRecordsUpTo(receivedReaders, memoryLimit / RECEIVED_MEMORY_DIVISOR, NULL, ClusterPair::compareByMember, splitLast, received, NULL, receivedPrefix)) > 0) {
                ClusterPair *receivedPairs = (ClusterPair *) received.data();
                SORT_PARALLEL(receivedPairs, receivedPairs + receivedCount, ClusterPair::compareByMember);
#pragma omp parallel for schedule(static)
                for (size_t i = 0; i < receivedCount; i++) {
                    DBKeyType memberKey = receivedPairs[i].memberKey;
                    size_t entry = std::lower_bound(entryKeys.begin(), entryKeys.end(), memberKey) - entryKeys.begin();
                    bool isEntry = entry < entryKeys.size() && entryKeys[entry] == memberKey;
                    if (isEntry == false && representativeBits == NULL) {
                        Debug(Debug::ERROR) << "Key " << memberKey << " of " << receivedPrefix << " is no entry of " << clu << "." << rank << "\n";
                        EXIT(EXIT_FAILURE);
                    }
                    if (isEntry) {
                        repKeys[entry] = receivedPairs[i].repKey;
                        receivedPairs[i].repKey = DB_KEY_INVALID;
                    }
                }
                if (representativeBits != NULL) {
                    sendMembersWithoutEntry(pairWriter, receivedPairs, receivedCount, representativeBits, firstKey);
                }
            }
            if (isStreamed) {
                while (readEntryKeys(cluDb, streamedEntry, memoryLimit, entryBuffer, streamedKeys)) {
                    for (size_t i = 0; i < streamedKeys.size(); i++) {
                        if (withRepresentative || streamedKeys[i] != entryKeys[0]) {
                            pairWriter.add(repKeys[0], streamedKeys[i]);
                        }
                    }
                }
                continue;
            }
            pairOffsets.resize(splitEnd - splitBegin + 1);
            pairOffsets[0] = 0;
#pragma omp parallel
            {
                int thread_idx = 0;
#ifdef OPENMP
                thread_idx = omp_get_thread_num();
#endif
#pragma omp for schedule(dynamic, 100)
                for (size_t i = splitBegin; i < splitEnd; i++) {
                    pairOffsets[i - splitBegin + 1] = Util::countLines(cluDb.getData(i, thread_idx), cluDb.getEntryLen(i)) - (withRepresentative ? 0 : 1);
                }
            }
            for (size_t i = 1; i < pairOffsets.size(); i++) {
                pairOffsets[i] += pairOffsets[i - 1];
            }
            for (size_t firstEntry = splitBegin; firstEntry < splitEnd;) {
                size_t firstPair = pairOffsets[firstEntry - splitBegin];
                size_t endEntry = splitBegin + (std::upper_bound(pairOffsets.begin() + (firstEntry - splitBegin) + 1, pairOffsets.end(),
                                                                       firstPair + pairWriter.pairCapacity - pairWriter.pairCount) - pairOffsets.begin()) - 1;
                if (endEntry == firstEntry) {
                    pairWriter.flush();
                    continue;
                }
#pragma omp parallel
                {
                    int thread_idx = 0;
#ifdef OPENMP
                    thread_idx = omp_get_thread_num();
#endif
                    char keyBuffer[255];
#pragma omp for schedule(dynamic, 100)
                    for (size_t i = firstEntry; i < endEntry; i++) {
                        DBKeyType entryKey = cluDb.getDbKey(i);
                        char *data = cluDb.getData(i, thread_idx);
                        size_t pairOffset = pairWriter.pairCount + pairOffsets[i - splitBegin] - firstPair;
                        while (*data != '\0') {
                            Util::parseKey(data, keyBuffer);
                            DBKeyType key = Util::fast_atoi<DBKeyType>(keyBuffer);
                            if (withRepresentative || key != entryKey) {
                                pairWriter.pairs[pairOffset].repKey = repKeys[i - splitBegin];
                                pairWriter.pairs[pairOffset].memberKey = key;
                                pairOffset++;
                            }
                            data = Util::skipLine(data);
                        }
                    }
                }
                pairWriter.pairCount += pairOffsets[endEntry - splitBegin] - firstPair;
                firstEntry = endEntry;
            }
        }
        cluDb.close();
    }
    ClusterPair lastPair = { DB_KEY_INVALID, DB_KEY_INVALID };
    size_t receivedCount;
    while ((receivedCount = readExchangeRecordsUpTo(receivedReaders, memoryLimit / RECEIVED_MEMORY_DIVISOR, NULL, ClusterPair::compareByMember, lastPair, received, NULL, receivedPrefix)) > 0) {
        if (representativeBits == NULL) {
            Debug(Debug::ERROR) << "Key " << ((ClusterPair *) received.data())->memberKey << " of " << receivedPrefix << " is no entry of " << clu << "." << rank << "\n";
            EXIT(EXIT_FAILURE);
        }
        sendMembersWithoutEntry(pairWriter, (ClusterPair *) received.data(), receivedCount, representativeBits, firstKey);
    }
    closeExchangeFiles(receivedFiles, receivedPrefix);
    if (balancedKeyOffsets != NULL && representativeBits != NULL) {
        writeKeyBits(exchange + ".bits." + SSTR(rank), representativeBits, firstKey, keyCount);
    }
    return pairWriter.close(representativeBits, firstKey, keyCount);
}

void MMseqsNode::readKeyBits(int fd, const std::string &fileName, size_t fileOffset, size_t fileFirstKey, size_t from, size_t to, size_t firstKey, uint64_t *bits) {
    if (from >= to) {
        return;
    }
    size_t firstWord = (from - fileFirstKey) / 64;
    std::vector<uint64_t> fileBits((to - 1 - fileFirstKey) / 64 + 1 - firstWord);
    FileUtil::preadAll(fd, fileName.c_str(), (char *) fileBits.data(), fileBits.size() * sizeof(uint64_t), fileOffset + firstWord * sizeof(uint64_t));
    for (size_t word = 0; word < fileBits.size(); word++) {
        size_t wordKey = fileFirstKey + (firstWord + word) * 64;
        uint64_t value = fileBits[word];
        value &= (wordKey < from) ? ~0ULL << (from - wordKey) : ~0ULL;
        value &= (to - wordKey < 64) ? (1ULL << (to - wordKey)) - 1 : ~0ULL;
        if (wordKey < firstKey) {
            value >>= firstKey - wordKey;
        }
        size_t bit = (wordKey < firstKey) ? 0 : wordKey - firstKey;
        bits[bit / 64] |= value << (bit % 64);
        if (bit % 64 != 0) {
            bits[bit / 64 + 1] |= value >> (64 - bit % 64);
        }
    }
}

uint64_t *MMseqsNode::readRepresentativeBits(const std::string &db, const std::vector<size_t> &fileKeyOffsets, size_t firstKey, size_t keyCount) {
    size_t words = (keyCount + 63) / 64 + 1;
    uint64_t *bits = new(std::nothrow) uint64_t[words];
    Util::checkAllocation(bits, "Can not allocate representative bit memory");
    memset(bits, 0, words * sizeof(uint64_t));
    for (size_t node = 0; node + 1 < fileKeyOffsets.size(); node++) {
        size_t from = std::max(firstKey, fileKeyOffsets[node]);
        size_t to = std::min(firstKey + keyCount, fileKeyOffsets[node + 1]);
        if (from >= to) {
            continue;
        }
        std::string bitsFile = db + ".bits." + SSTR(node);
        FILE *file = FileUtil::openFileOrDie(bitsFile.c_str(), "r", true);
        uint64_t keyRange[2];
        FileUtil::preadAll(fileno(file), bitsFile.c_str(), (char *) keyRange, sizeof(keyRange), 0);
        if (keyRange[0] > from || keyRange[0] + keyRange[1] < to) {
            Debug(Debug::ERROR) << "Cannot read " << bitsFile << "\n";
            EXIT(EXIT_FAILURE);
        }
        readKeyBits(fileno(file), bitsFile, sizeof(keyRange), keyRange[0], from, to, firstKey, bits);
        if (fclose(file) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << bitsFile << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
    return bits;
}

void MMseqsNode::writeRepresentativeBits(const std::string &db, DBReader<DBKeyType> &seqDbr, const uint64_t *bits, size_t firstKey, size_t keyCount) {
    writeKeyBits(db + ".bits." + SSTR(rank), bits, firstKey, keyCount);
    // add the representative count as a sixth column
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    std::string representativeIndex;
    for (size_t range = 0; range < seqDbr.getIndexSize(); range++) {
        if (seqDbr.getRangeDataFile(range) != rank) {
            continue;
        }
        size_t rangeEndKey = (range + 1 < seqDbr.getIndexSize()) ? index[range + 1].id : seqDbr.getSize();
        size_t rangeRepresentatives = Util::countSetBits(bits, index[range].id - firstKey, rangeEndKey - firstKey);
        representativeIndex += SSTR(index[range].id) + "\t" + SSTR(index[range].offset) + "\t" + SSTR(index[range].length) + "\t"
                             + SSTR(rangeEndKey - index[range].id) + "\t" + SSTR(rank) + "\t" + SSTR(rangeRepresentatives) + "\n";
    }
    publishRunFile(db + ".index." + SSTR(rank), representativeIndex);
    if (isMaster()) {
        publishFile(db + ".index", gatherRunFiles(db + ".index."));
    }
}

std::vector<size_t> MMseqsNode::getClusterKeyOffsets(const std::string &file, const std::string &clu, size_t keyCount, const std::string &firstKeyFile) {
    std::string cluIndex = firstKeyFile.empty() ? clu + ".index." + SSTR(rank) : firstKeyFile;
    FILE *handle = FileUtil::openFileOrDie(cluIndex.c_str(), "r", true);
    uint64_t firstCell[2];
    if (fread(firstCell, sizeof(uint64_t), 2, handle) != 2 || fclose(handle) != 0) {
        Debug(Debug::ERROR) << "Cannot read " << cluIndex << "\n";
        EXIT(EXIT_FAILURE);
    }
    std::string firstKeys = barrier(file, SSTR(firstCell[0]) + "\n");
    removeOnceRead(file);
    std::vector<size_t> offsets(count + 1, keyCount);
    char *line = (char *) firstKeys.c_str();
    for (size_t node = 0; node < count; node++, line = Util::skipLine(line)) {
        offsets[node] = Util::fast_atoi<size_t>(line);
    }
    for (size_t node = count; node > 0; node--) {
        offsets[node - 1] = std::min(offsets[node - 1], offsets[node]);
    }
    return offsets;
}

std::vector<size_t> MMseqsNode::getBalancedClusterKeyOffsets(const std::string &file, const std::string &clu, size_t keyCount) {
    if (isMaster()) {
        std::vector<size_t> partOffsets(count + 1, 0);
        for (size_t node = 0; node < count; node++) {
            DBReader<DBKeyType> cluDb((clu + "." + SSTR(node)).c_str(), (clu + ".index." + SSTR(node)).c_str(), 1, DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
            cluDb.open(DBReader<DBKeyType>::NOSORT);
            partOffsets[node + 1] = partOffsets[node] + cluDb.getDataSize();
            cluDb.close();
        }
        std::string firstKeys;
        size_t split = 1;
        for (size_t node = 0; node < count && split < count; node++) {
            if (partOffsets[count] * split / count >= partOffsets[node + 1]) {
                continue;
            }
            DBReader<DBKeyType> cluDb((clu + "." + SSTR(node)).c_str(), (clu + ".index." + SSTR(node)).c_str(), 1, DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
            cluDb.open(DBReader<DBKeyType>::NOSORT);
            for (; split < count && partOffsets[count] * split / count < partOffsets[node + 1]; split++) {
                size_t entry = cluDb.findEntryByOffset(partOffsets[count] * split / count - partOffsets[node]);
                size_t firstKey = (entry < cluDb.getSize()) ? static_cast<size_t>(cluDb.getIndexEntry(entry).id) : static_cast<size_t>(cluDb.getIndexEntry(cluDb.getSize() - 1).id) + 1;
                firstKeys += SSTR(firstKey) + "\n";
            }
            cluDb.close();
        }
        for (; split < count; split++) {
            firstKeys += SSTR(keyCount) + "\n";
        }
        publishRunFile(file, firstKeys);
    }
    std::string firstKeys = waitForRunFile(file);
    removeOnceRead(file);
    std::vector<size_t> offsets(count + 1, keyCount);
    offsets[0] = 0;
    char *line = (char *) firstKeys.c_str();
    for (size_t node = 1; node < count; node++, line = Util::skipLine(line)) {
        offsets[node] = Util::fast_atoi<size_t>(line);
    }
    return offsets;
}

void MMseqsNode::readClusterKeys(const std::string &clu, const std::vector<size_t> &cluKeyOffsets, size_t keyBegin, size_t keyEnd, std::vector<DBKeyType> &keys) {
    keys.clear();
    size_t cellsPerRead = 1024 * 1024 / (2 * sizeof(uint64_t));
    for (size_t node = 0; node < count; node++) {
        size_t from = std::max(keyBegin, cluKeyOffsets[node]);
        size_t to = std::min(keyEnd, cluKeyOffsets[node + 1]);
        if (from >= to) {
            continue;
        }
        std::string cluData = clu + "." + SSTR(node);
        std::string cluIndex = clu + ".index." + SSTR(node);
        DBReader<DBKeyType> cluDb(cluData.c_str(), cluIndex.c_str(), 1, DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
        cluDb.open(DBReader<DBKeyType>::NOSORT);
        size_t entryBegin = cluDb.findEntryByKey(from);
        size_t entryEnd = cluDb.findEntryByKey(to);
        cluDb.close();
        FILE *cluIndexFile = FileUtil::openFileOrDie(cluIndex.c_str(), "r", true);
        size_t nodeKeysBegin = keys.size();
        keys.resize(nodeKeysBegin + entryEnd - entryBegin);
#pragma omp parallel
        {
            std::vector<uint64_t> cells(2 * cellsPerRead);
#pragma omp for schedule(dynamic, 1)
            for (size_t entry = entryBegin; entry < entryEnd; entry += cellsPerRead) {
                size_t cellCount = std::min(cellsPerRead, entryEnd - entry);
                FileUtil::preadAll(fileno(cluIndexFile), cluIndex.c_str(), (char *) cells.data(), cellCount * 2 * sizeof(uint64_t), entry * 2 * sizeof(uint64_t));
                for (size_t i = 0; i < cellCount; i++) {
                    keys[nodeKeysBegin + entry - entryBegin + i] = cells[2 * i];
                }
            }
        }
        if (fclose(cluIndexFile) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << cluIndex << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
}

void MMseqsNode::writeClusterThreadEntries(FILE *dataFile, const std::string &dataFileName, FILE *indexFile, const std::string &indexFileName,
                                           std::vector<std::string> &threadData, std::vector<std::vector<uint64_t> > &threadEntries, size_t &dataSize, size_t &indexSize) {
    int threads = static_cast<int>(threadData.size());
    std::vector<size_t> threadDataOffsets(threads + 1);
    std::vector<size_t> threadEntryOffsets(threads + 1);
    threadDataOffsets[0] = dataSize;
    threadEntryOffsets[0] = indexSize;
    for (int thread = 0; thread < threads; thread++) {
        threadDataOffsets[thread + 1] = threadDataOffsets[thread] + threadData[thread].size();
        threadEntryOffsets[thread + 1] = threadEntryOffsets[thread] + threadEntries[thread].size() * sizeof(uint64_t);
    }
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (int thread = 0; thread < threads; thread++) {
        std::vector<uint64_t> &entries = threadEntries[thread];
        for (size_t i = 1; i < entries.size(); i += 2) {
            entries[i] += threadDataOffsets[thread];
        }
        FileUtil::pwriteAll(fileno(dataFile), dataFileName.c_str(), threadData[thread].data(), threadData[thread].size(), threadDataOffsets[thread]);
        FileUtil::pwriteAll(fileno(indexFile), indexFileName.c_str(), (const char *) entries.data(), entries.size() * sizeof(uint64_t), threadEntryOffsets[thread]);
    }
    dataSize = threadDataOffsets[threads];
    indexSize = threadEntryOffsets[threads];
}

void MMseqsNode::closeClusterFiles(FILE *dataFile, const std::string &dataFileName, FILE *indexFile, const std::string &indexFileName, size_t dataSize, size_t indexSize) {
    uint64_t endEntry[2] = { static_cast<uint64_t>(DB_KEY_INVALID), static_cast<uint64_t>(dataSize) };
    FileUtil::pwriteAll(fileno(indexFile), indexFileName.c_str(), (const char *) endEntry, sizeof(endEntry), indexSize);
    if (fclose(dataFile) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << dataFileName << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (fclose(indexFile) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << indexFileName << "\n";
        EXIT(EXIT_FAILURE);
    }
}

static size_t findMergedPairCount(const ClusterPair *pairs, size_t pairCount, const ClusterPair *singletons, size_t singletonCount, size_t pos) {
    size_t low = (pos > singletonCount) ? pos - singletonCount : 0;
    size_t high = std::min(pos, pairCount);
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        if (ClusterPair::compareByRep(pairs[mid], singletons[pos - mid - 1])) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

// rep of the last pair or singleton before the merged position
static DBKeyType getLastRepKey(const ClusterPair *pairs, const ClusterPair *singletons, size_t pairsBefore, size_t singletonsBefore) {
    bool isPair = singletonsBefore == 0 || (pairsBefore > 0 && ClusterPair::compareByRep(singletons[singletonsBefore - 1], pairs[pairsBefore - 1]));
    return isPair ? pairs[pairsBefore - 1].repKey : singletons[singletonsBefore - 1].repKey;
}

void MMseqsNode::writeClusterPairs(const std::string &index, const std::string &prefix, const std::string &clu, const uint64_t *singletonBits, size_t firstKey, size_t keyCount) {
    const Parameters &par = Parameters::getInstance();
    std::vector<ExchangeReader> repReaders;
    std::vector<FILE *> repFiles;
    openExchange(index, prefix, repReaders, repFiles);
    size_t memoryLimit = getMemoryLimit(par);
    std::string resultDb = clu + "." + SSTR(rank);
    std::string resultIndex = clu + ".index." + SSTR(rank);
    FILE *resultDataFile = FileUtil::openAndDelete(resultDb.c_str(), "w");
    FILE *resultIndexFile = FileUtil::openAndDelete(resultIndex.c_str(), "w");
    size_t resultDataSize = 0;
    size_t resultIndexSize = 0;
    std::vector<char> records;
    std::vector<ClusterPair> singletons;
    std::vector<size_t> threadSingletonOffsets(par.threads + 1, 0);
    std::vector<std::string> threadData(par.threads);
    std::vector<std::vector<uint64_t> > threadEntries(par.threads);
    DBKeyType previousRepKey = DB_KEY_INVALID;
    size_t splitKeys = std::max((size_t) 1, memoryLimit / 16 / sizeof(ClusterPair));
    size_t keyEnd = firstKey + keyCount;
    for (size_t splitBegin = firstKey; splitBegin < keyEnd; splitBegin += splitKeys) {
        size_t splitEnd = std::min(keyEnd, splitBegin + splitKeys);
        ClusterPair lastPair = { static_cast<DBKeyType>(splitEnd - 1), DB_KEY_INVALID };
        size_t nextSingletonKey = splitBegin;
        size_t pairCount;
        do {
            pairCount = readExchangeRecordsUpTo(repReaders, memoryLimit / 8, NULL, ClusterPair::compareByRep, lastPair, records, NULL, prefix);
            ClusterPair *pairs = (ClusterPair *) records.data();
            SORT_PARALLEL(pairs, pairs + pairCount, ClusterPair::compareByRep);
            singletons.clear();
            if (singletonBits != NULL) {
                size_t singletonEnd = (pairCount > 0) ? std::max(nextSingletonKey, (size_t) (pairs[pairCount - 1].repKey + 1)) : splitEnd;
                size_t bitBegin = nextSingletonKey - firstKey;
                size_t bitEnd = singletonEnd - firstKey;
                size_t wordBegin = bitBegin / 64;
                size_t wordCount = (bitEnd + 63) / 64 - wordBegin;
#pragma omp parallel for schedule(dynamic, 1)
                for (int thread = 0; thread < par.threads; thread++) {
                    threadSingletonOffsets[thread + 1] = Util::countSetBits(singletonBits, std::max(bitBegin, (wordBegin + wordCount * thread / par.threads) * 64),
                                                                            std::min(bitEnd, (wordBegin + wordCount * (thread + 1) / par.threads) * 64));
                }
                for (int thread = 0; thread < par.threads; thread++) {
                    threadSingletonOffsets[thread + 1] += threadSingletonOffsets[thread];
                }
                singletons.resize(threadSingletonOffsets[par.threads]);
#pragma omp parallel for schedule(dynamic, 1)
                for (int thread = 0; thread < par.threads; thread++) {
                    size_t singleton = threadSingletonOffsets[thread];
                    for (size_t word = wordBegin + wordCount * thread / par.threads; word < wordBegin + wordCount * (thread + 1) / par.threads; word++) {
                        uint64_t bits = singletonBits[word] & ((word == wordBegin) ? ~0ULL << (bitBegin % 64) : ~0ULL);
                        if (bitEnd - word * 64 < 64) {
                            bits &= (1ULL << (bitEnd - word * 64)) - 1;
                        }
                        for (; bits != 0; bits &= bits - 1) {
                            DBKeyType key = firstKey + word * 64 + __builtin_ctzll(bits);
                            singletons[singleton].repKey = key;
                            singletons[singleton].memberKey = key;
                            singleton++;
                        }
                    }
                }
                nextSingletonKey = std::max(nextSingletonKey, singletonEnd);
            }
            size_t singletonCount = singletons.size();
            size_t writeCount = pairCount + singletonCount;
            if (writeCount == 0) {
                break;
            }
#pragma omp parallel for schedule(dynamic, 1)
            for (int thread = 0; thread < par.threads; thread++) {
                std::string &data = threadData[thread];
                std::vector<uint64_t> &entries = threadEntries[thread];
                data.clear();
                entries.clear();
                char buffer[32];
                size_t begin = writeCount * thread / par.threads;
                size_t end = writeCount * (thread + 1) / par.threads;
                size_t pair = findMergedPairCount(pairs, pairCount, singletons.data(), singletonCount, begin);
                size_t singleton = begin - pair;
                size_t endPair = findMergedPairCount(pairs, pairCount, singletons.data(), singletonCount, end);
                size_t endSingleton = end - endPair;
                DBKeyType repKey = (begin > 0) ? getLastRepKey(pairs, singletons.data(), pair, singleton) : previousRepKey;
                while (pair < endPair || singleton < endSingleton) {
                    const ClusterPair &next = (singleton == endSingleton || (pair < endPair && ClusterPair::compareByRep(pairs[pair], singletons[singleton]))) ? pairs[pair++] : singletons[singleton++];
                    if (next.repKey != repKey) {
                        if (repKey != DB_KEY_INVALID) {
                            data.push_back('\0');
                        }
                        repKey = next.repKey;
                        entries.push_back(static_cast<uint64_t>(repKey));
                        entries.push_back(data.size());
                        char *tmpBuff = Itoa::u64toa_sse2(static_cast<uint64_t>(repKey), buffer);
                        *(tmpBuff - 1) = '\n';
                        data.append(buffer, tmpBuff - buffer);
                    }
                    if (next.memberKey != repKey) {
                        char *tmpBuff = Itoa::u64toa_sse2(static_cast<uint64_t>(next.memberKey), buffer);
                        *(tmpBuff - 1) = '\n';
                        data.append(buffer, tmpBuff - buffer);
                    }
                }
            }
            writeClusterThreadEntries(resultDataFile, resultDb, resultIndexFile, resultIndex, threadData, threadEntries, resultDataSize, resultIndexSize);
            previousRepKey = getLastRepKey(pairs, singletons.data(), pairCount, singletonCount);
        } while (pairCount > 0);
    }
    if (previousRepKey != DB_KEY_INVALID) {
        FileUtil::pwriteAll(fileno(resultDataFile), resultDb.c_str(), "\0", 1, resultDataSize);
        resultDataSize++;
    }
    closeClusterFiles(resultDataFile, resultDb, resultIndexFile, resultIndex, resultDataSize, resultIndexSize);
    closeExchangeFiles(repFiles, prefix);
}

void MMseqsNode::removeExchanges(const std::string &prefix, const std::string &lastExchange, const std::vector<std::string> &exchanges, std::string phaseValues) {
    addPhaseValue(phaseValues, "Exchange bytes sent", SSTR(sentExchangeSize));
    addPhaseValue(phaseValues, "Exchange bytes sent to other nodes", SSTR(sentOtherNodeExchangeSize));
    addPhaseValue(phaseValues, "Exchange index opens", SSTR(exchangeIndexOpenCount));
    printPhaseValues(phaseValues);
    barrier(prefix + ".merged");
    FileUtil::remove((lastExchange + "." + SSTR(rank)).c_str());
    for (size_t i = 0; isMaster() && i < exchanges.size(); i++) {
        FileUtil::remove((exchanges[i] + ".index").c_str());
    }
    removeOnceRead(prefix + ".merged");
}
