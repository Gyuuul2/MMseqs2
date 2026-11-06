#include "MMseqsNode.h"
#include "Debug.h"
#include "FileUtil.h"
#include "Parameters.h"
#include "Util.h"
#include "FastSort.h"
#include "DBReader.h"
#include "itoa.h"
#include "Timer.h"

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
const size_t MMseqsNode::EXCHANGE_BUFFER_SIZE;
const size_t MMseqsNode::EXCHANGE_SLICE_HEADER_BYTES;
const size_t MMseqsNode::EXCHANGE_WRITE_BYTES;
// the first line of every barrier file, so a file that a killed run left is told apart from one of this run
static std::string runId;
// the exchange bytes that this node sent, all of them and those for other nodes, reported by removeExchanges
static size_t sentExchangeBytes = 0;
static size_t sentOtherNodeExchangeBytes = 0;
// the records and their bytes that this node sent to itself in the last flush of an exchange, until openExchange reads them
static std::map<std::string, std::vector<std::pair<char *, size_t> > > ownExchangeRecords;
static const unsigned int MAX_READ_POLLS = 400;

void MMseqsNode::init() {
    setvbuf(stdout, NULL, _IOLBF, 0);
    // mpirun (Open MPI, then PMI of MPICH/Intel MPI) before SLURM, whose variables mpirun tasks inherit from the allocation
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
    const std::string tmpFile = file + ".tmp";
    FILE *handle = FileUtil::openAndDelete(tmpFile.c_str(), "w");
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

void MMseqsNode::sleepBeforePoll(unsigned int polls, unsigned int maxShift) {
    usleep(10000u << std::min(polls, maxShift));
}

// whether the directory of file lists it: reading the directory asks the server, while a lookup of a name that NFS once
// saw missing is answered from its cache for up to 60 s (Nextflow lists the work directory before it reads .exitcode, too); a listing refreshes
// the directory for all threads, so a process lists at most once a second and treats a file as not listed in between
static bool isListed(const std::string &file) {
    if (FileUtil::needsDirectoryRefresh(file) == false) {
        return false;
    }
    static time_t lastListing = 0;
    const time_t now = time(NULL);
    const time_t last = __atomic_load_n(&lastListing, __ATOMIC_RELAXED);
    if (now == last || __sync_bool_compare_and_swap(&lastListing, last, now) == false) {
        return false;
    }
    const std::string name = FileUtil::baseName(file);
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

static const time_t DIRECTORY_WAIT_SECONDS = 60;

static int openFile(const std::string &file, int flags = O_RDONLY) {
    int fd;
    do {
        fd = open(file.c_str(), flags, 0666);
    } while (fd < 0 && errno == EINTR);
    return fd;
}

// the content of file, or an empty string while it is not published; NFS only promises that open sees the latest state of a file that
// exists, a name it found missing can stay missing, so a poll creates the empty file that the rename of the publish replaces; a file
// that a later publish replaces while it is read fails with ESTALE and is read again by the next poll; other file systems see the rename at once
static std::string readPolledFile(const std::string &file) {
    const int flags = FileUtil::needsDirectoryRefresh(file) ? (O_RDONLY | O_CREAT) : O_RDONLY;
    int fd = openFile(file, flags);
    int openError = (fd < 0) ? errno : 0;
    // with O_CREAT, ENOENT means that the directory does not exist on this machine, or that NFS still remembers it missing
    const time_t start = time(NULL);
    for (unsigned int polls = 0; openError == ENOENT && (flags & O_CREAT) != 0; polls++) {
        if (time(NULL) - start >= DIRECTORY_WAIT_SECONDS) {
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
    std::string content;
    struct stat sb;
    bool isStale = false;
    if (fstat(fd, &sb) != 0) {
        const int statError = errno;
        isStale = statError == ESTALE;
        if (isStale == false) {
            Debug(Debug::ERROR) << "Cannot read " << file << ": " << strerror(statError) << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
    content.resize(isStale ? 0 : sb.st_size);
    for (size_t done = 0; done < content.size();) {
        const ssize_t count = pread(fd, &content[done], content.size() - done, done);
        const int readError = (count < 0) ? errno : 0;
        if (readError == EINTR) {
            continue;
        }
        if (readError != 0 && readError != ESTALE) {
            Debug(Debug::ERROR) << "Cannot read " << file << ": " << strerror(readError) << "\n";
            EXIT(EXIT_FAILURE);
        }
        if (count <= 0) {
            isStale = true;
            break;
        }
        done += count;
    }
    close(fd);
    return isStale ? std::string() : content;
}

MMseqsNode::WaitWarning::WaitWarning(size_t seconds) : start(time(NULL)), firstSeconds(seconds), nextSeconds(seconds) {}

void MMseqsNode::WaitWarning::reset() {
    start = time(NULL);
    nextSeconds = firstSeconds;
}

void MMseqsNode::WaitWarning::check(const std::string &what) {
    if (static_cast<size_t>(time(NULL) - start) >= nextSeconds) {
        Debug(Debug::ERROR) << "Node " << rank << " waited " << nextSeconds << " s for " << what << ". Check that no node has died and that all nodes share this directory\n";
        nextSeconds *= 2;
    }
}

// polls of a wait of up to 5 s are at most 160 ms apart, then twice as far every 32 polls, as all nodes wait in a long barrier
static void sleepInWait(unsigned int polls) {
    MMseqsNode::sleepBeforePoll(polls, std::min(8u, 4 + polls / 32));
}

// the content of file once it is published; a published file is never empty
static std::string waitForFile(const std::string &file) {
    std::string content = readPolledFile(file);
    MMseqsNode::WaitWarning warning;
    for (unsigned int polls = 0; content.empty(); polls++) {
        sleepInWait(polls);
        warning.check(file);
        content = readPolledFile(file);
    }
    return content;
}

std::string MMseqsNode::readFile(const std::string &file) {
    FILE *handle = FileUtil::openFileOrDie(file.c_str(), "r", true);
    struct stat sb;
    if (fstat(fileno(handle), &sb) != 0) {
        Debug(Debug::ERROR) << "Cannot read " << file << "\n";
        EXIT(EXIT_FAILURE);
    }
    std::string content(sb.st_size, '\0');
    const size_t readBytes = content.empty() ? 0 : fread(&content[0], sizeof(char), content.size(), handle);
    const int closeResult = fclose(handle);
    if (readBytes != content.size() || closeResult != 0) {
        Debug(Debug::ERROR) << "Cannot read " << file << "\n";
        EXIT(EXIT_FAILURE);
    }
    return content;
}

// a file of another run is read again until this run replaces it
std::string MMseqsNode::waitForRunFile(const std::string &file) {
    std::string content = waitForFile(file);
    WaitWarning warning;
    for (unsigned int polls = 0; content.compare(0, runId.size(), runId) != 0; polls++) {
        sleepInWait(polls);
        warning.check(file);
        content = waitForFile(file);
    }
    return content.substr(runId.size());
}

void MMseqsNode::publishRunFile(const std::string &file, const std::string &content) {
    publishFile(file, runId + content);
}

std::string MMseqsNode::gatherRunFiles(const std::string &prefix) {
    std::vector<std::string> nodeContents(count);
    // the threads wait for the files at once, so their reads do not add up node after node
#pragma omp parallel for schedule(dynamic, 1)
    for (unsigned int node = 0; node < count; node++) {
        const std::string file = prefix + SSTR(node);
        nodeContents[node] = waitForRunFile(file);
        FileUtil::remove(file.c_str());
    }
    std::string content;
    for (unsigned int node = 0; node < count; node++) {
        content += nodeContents[node];
    }
    return content;
}

void MMseqsNode::publishLog(FILE *file, const std::string &log, size_t key, size_t bytes, size_t orderedClaims) {
    if (fflush(file) != 0 || fsync(fileno(file)) != 0) {
        Debug(Debug::ERROR) << "Cannot flush " << log << "\n";
        EXIT(EXIT_FAILURE);
    }
    publishRunFile(log + ".progress", SSTR(key) + "\t" + SSTR(bytes) + "\t" + SSTR(orderedClaims) + "\n");
}

bool MMseqsNode::readLogProgress(const std::string &log, size_t &key, size_t &bytes, size_t *orderedClaims) {
    const std::string file = log + ".progress";
    // NFS may fail to read a progress file that was just replaced; the next poll reads it again
    const std::string content = readPolledFile(file);
    if (content.compare(0, runId.size(), runId) != 0) {
        return false;
    }
    const char *columns[3];
    if (Util::getWordsOfLine((char *) content.c_str() + runId.size(), columns, 3) < 3) {
        Debug(Debug::ERROR) << "Cannot read " << file << "\n";
        EXIT(EXIT_FAILURE);
    }
    key = Util::fast_atoi<size_t>(columns[0]);
    bytes = Util::fast_atoi<size_t>(columns[1]);
    if (orderedClaims != NULL) {
        *orderedClaims = Util::fast_atoi<size_t>(columns[2]);
    }
    return true;
}

void MMseqsNode::waitForLog(const std::string &log, size_t &key, size_t &bytes) {
    WaitWarning warning;
    for (unsigned int polls = 0; readLogProgress(log, key, bytes) == false; polls++) {
        sleepInWait(polls);
        warning.check(log + ".progress");
    }
}

bool MMseqsNode::readLog(const std::string &log, size_t from, size_t to, char *data, bool wait) {
    const size_t bytes = to - from;
    if (bytes == 0) {
        return true;
    }
    // NFS can expose the progress file before the flushed log is visible
    for (unsigned int polls = 0; polls < MAX_READ_POLLS; polls++) {
        int fd = openFile(log);
        int openError = (fd < 0) ? errno : 0;
        if (openError == ENOENT && isListed(log)) {
            fd = openFile(log);
            openError = (fd < 0) ? errno : 0;
        }
        if (fd < 0) {
            if (openError != ENOENT && openError != ESTALE) {
                Debug(Debug::ERROR) << "Cannot open " << log << ": " << strerror(openError) << "\n";
                EXIT(EXIT_FAILURE);
            }
        } else {
            size_t done = 0;
            int readError = 0;
            while (done < bytes) {
                const ssize_t count = pread(fd, data + done, bytes - done, from + done);
                if (count > 0) {
                    done += count;
                } else if (count < 0 && errno == EINTR) {
                    continue;
                } else if (count < 0) {
                    readError = errno;
                    break;
                } else {
                    break;
                }
            }
            const int closeError = (close(fd) != 0) ? errno : 0;
            if ((readError != 0 && readError != ESTALE) || (closeError != 0 && closeError != ESTALE && closeError != EINTR)) {
                Debug(Debug::ERROR) << "Cannot read " << log << ": " << strerror(readError != 0 ? readError : closeError) << "\n";
                EXIT(EXIT_FAILURE);
            }
            if (done == bytes && readError == 0 && closeError == 0) {
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

static uint64_t exchangeSliceMagic();

void MMseqsNode::publishExchangeFlush(FILE *file, const std::string &fileName, size_t bytes, const std::string &indexFile, const std::vector<uint64_t> &line) {
    if (fclose(file) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << fileName << "\n";
        EXIT(EXIT_FAILURE);
    }
    __sync_fetch_and_add(&sentExchangeBytes, bytes);
    __sync_fetch_and_add(&sentOtherNodeExchangeBytes, bytes);
    FILE *handle = FileUtil::openFileOrDie(indexFile.c_str(), "a", true);
    const uint64_t magic = exchangeSliceMagic();
    if (fwrite(line.data(), sizeof(uint64_t), line.size() - 1, handle) != line.size() - 1
        || fwrite(&magic, sizeof(magic), 1, handle) != 1 || fflush(handle) != 0 || fclose(handle) != 0) {
        Debug(Debug::ERROR) << "Cannot write " << indexFile << "\n";
        EXIT(EXIT_FAILURE);
    }
}

size_t MMseqsNode::readExchangeFlushes(const std::string &indexFile, size_t flush, size_t lineWords, std::vector<uint64_t> &lines) {
    const size_t lineBytes = lineWords * sizeof(uint64_t);
    int fd = openFile(indexFile);
    int openError = (fd < 0) ? errno : 0;
    // a listing refreshes an NFS negative lookup before the index is opened again
    if (openError == ENOENT && isListed(indexFile)) {
        fd = openFile(indexFile);
        openError = (fd < 0) ? errno : 0;
    }
    if (fd < 0) {
        if (openError == ENOENT || openError == ESTALE) {
            return 0;
        }
        Debug(Debug::ERROR) << "Cannot open " << indexFile << ": " << strerror(openError) << "\n";
        EXIT(EXIT_FAILURE);
    }
    ssize_t readBytes;
    do {
        readBytes = pread(fd, lines.data(), lines.size() * sizeof(uint64_t), flush * lineBytes);
    } while (readBytes < 0 && errno == EINTR);
    const int readError = (readBytes < 0) ? errno : 0;
    const int closeError = (close(fd) != 0) ? errno : 0;
    if (readError == ESTALE || closeError == ESTALE || closeError == EINTR) {
        return 0;
    }
    if (readError != 0 || closeError != 0) {
        Debug(Debug::ERROR) << "Cannot read " << indexFile << ": " << strerror(readError != 0 ? readError : closeError) << "\n";
        EXIT(EXIT_FAILURE);
    }
    size_t lineCount = 0;
    while ((lineCount + 1) * lineBytes <= static_cast<size_t>(readBytes) && lines[lineCount * lineWords] == flush + lineCount
           && lines[(lineCount + 1) * lineWords - 1] == exchangeSliceMagic()) {
        lineCount++;
    }
    return lineCount;
}

struct ExchangeSliceHeader {
    uint64_t magic;
    uint64_t generation;
};
static_assert(sizeof(ExchangeSliceHeader) == MMseqsNode::EXCHANGE_SLICE_HEADER_BYTES, "Unexpected exchange slice header layout");
static const uint64_t EXCHANGE_SLICE_MAGIC = 0x4d4d533258434831ULL;

static uint64_t exchangeSliceMagic() {
    uint64_t magic = EXCHANGE_SLICE_MAGIC;
    // the run id keeps a generation 0 file of an earlier job from matching this one
    for (size_t i = 0; i < runId.size(); i++) {
        magic = (magic ^ static_cast<unsigned char>(runId[i])) * 1099511628211ULL;
    }
    return magic;
}

void MMseqsNode::writeExchangeSliceHeader(FILE *file, const std::string &fileName, uint64_t generation) {
    const ExchangeSliceHeader header = { exchangeSliceMagic(), generation };
    if (fwrite(&header, sizeof(header), 1, file) != 1 || fflush(file) != 0) {
        Debug(Debug::ERROR) << "Cannot write " << fileName << "\n";
        EXIT(EXIT_FAILURE);
    }
}

void MMseqsNode::writeExchangeFlush(const std::string &fileName, uint64_t generation, const char *data, size_t bytes, const std::string &indexFile,
                                    const std::vector<uint64_t> &line, int threads) {
    FILE *file = FileUtil::openAndDelete(fileName.c_str(), "w");
    writeExchangeSliceHeader(file, fileName, generation);
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (size_t offset = 0; offset < bytes; offset += EXCHANGE_WRITE_BYTES) {
        FileUtil::pwriteAll(fileno(file), fileName.c_str(), data + offset, std::min(EXCHANGE_WRITE_BYTES, bytes - offset), EXCHANGE_SLICE_HEADER_BYTES + offset);
    }
    publishExchangeFlush(file, fileName, EXCHANGE_SLICE_HEADER_BYTES + bytes, indexFile, line);
}

int MMseqsNode::tryOpenExchangeSlice(const std::string &file, size_t expectedBytes, uint64_t expectedGeneration, time_t &lastDirectoryRefresh,
                                     size_t &observedBytes, uint64_t &observedGeneration, int &lastError) {
    observedBytes = SIZE_MAX;
    observedGeneration = UINT64_MAX;
    lastError = 0;
    int fd = openFile(file, O_RDONLY | O_CLOEXEC);
    int openError = (fd < 0) ? errno : 0;
    if (openError == ENOENT) {
        const time_t now = time(NULL);
        if (now != lastDirectoryRefresh) {
            lastDirectoryRefresh = now;
            if (isListed(file)) {
                fd = openFile(file, O_RDONLY | O_CLOEXEC);
                openError = (fd < 0) ? errno : 0;
            }
        }
    }
    if (fd < 0) {
        lastError = openError;
        return -1;
    }
    struct stat fileStat;
    int statResult;
    do {
        statResult = fstat(fd, &fileStat);
    } while (statResult < 0 && errno == EINTR);
    if (statResult == 0) {
        observedBytes = static_cast<size_t>(fileStat.st_size);
        if (observedBytes == expectedBytes) {
            ExchangeSliceHeader header;
            ssize_t headerBytes;
            do {
                headerBytes = pread(fd, &header, sizeof(header), 0);
            } while (headerBytes < 0 && errno == EINTR);
            if (headerBytes == static_cast<ssize_t>(sizeof(header))) {
                observedGeneration = header.generation;
            }
            if (headerBytes == static_cast<ssize_t>(sizeof(header)) && header.magic == exchangeSliceMagic()
                && header.generation == expectedGeneration) {
                return fd;
            }
            if (headerBytes < 0) {
                lastError = errno;
            }
        }
    } else {
        lastError = errno;
    }
    close(fd);
    return -1;
}

bool MMseqsNode::tryReadExchangeSlice(int fd, char *data, size_t bytes, size_t offset, int &readError) {
    // a published slice may still read short or turn stale on NFS; the caller opens it again
    for (size_t done = 0; done < bytes;) {
        const ssize_t count = pread(fd, data + done, bytes - done, offset + done);
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
    const std::string prefix = (count > 1) ? "[node " + SSTR(rank) + "] " : "";
    std::string output;
    for (char *line = (char *) values.c_str(); *line != '\0'; line = Util::skipLine(line)) {
        const char *columns[3];
        if (Util::getFieldsOfLine(line, columns, 3) < 3) {
            Debug(Debug::ERROR) << "Cannot read the phase values\n";
            EXIT(EXIT_FAILURE);
        }
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

// O_EXCL creates atomically on local file systems, GPFS, Lustre and NFS version 3 or later
bool MMseqsNode::WorkQueue::claim(size_t item) {
    const std::string file = name + "." + SSTR(item);
    const int fd = open(file.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0666);
    if (fd < 0 && errno == EEXIST) {
        return false;
    }
    if (fd < 0 || write(fd, &rank, sizeof(rank)) != static_cast<ssize_t>(sizeof(rank)) || close(fd) != 0) {
        Debug(Debug::ERROR) << "Cannot create " << file << "\n";
        EXIT(EXIT_FAILURE);
    }
    return true;
}

size_t MMseqsNode::WorkQueue::claimNext() {
    for (; nextItem < itemCount; nextItem += count) {
        if (claim(nextItem)) {
            nextItem += count;
            return nextItem - count;
        }
    }
    // a failed claim at the back of an owner means that its owner or another node is there
    for (size_t emptyNodes = 0; emptyNodes < count;) {
        const size_t owner = nextNode;
        nextNode = (nextNode + 1) % count;
        if (itemsLeft[owner] == 0) {
            emptyNodes++;
            continue;
        }
        emptyNodes = 0;
        itemsLeft[owner]--;
        if (claim(owner + itemsLeft[owner] * count)) {
            return owner + itemsLeft[owner] * count;
        }
        itemsLeft[owner] = 0;
    }
    return itemCount;
}

size_t MMseqsNode::WorkQueue::claimInOrder() {
    size_t item = nextOrderedItem;
    while (item < itemCount && claim(item) == false) {
        item++;
    }
    __atomic_store_n(&nextOrderedItem, std::min(item + 1, itemCount), __ATOMIC_RELAXED);
    return item;
}

size_t MMseqsNode::WorkQueue::getOrderedClaims() const {
    return __atomic_load_n(&nextOrderedItem, __ATOMIC_RELAXED);
}

unsigned int MMseqsNode::WorkQueue::getNode(size_t item) const {
    unsigned int node = count;
    const std::string file = name + "." + SSTR(item);
    int fd = openFile(file);
    if (fd < 0 && errno == ENOENT && isListed(file)) {
        fd = openFile(file);
    }
    if (fd >= 0) {
        if (pread(fd, &node, sizeof(node), 0) != static_cast<ssize_t>(sizeof(node))) {
            node = count;
        }
        close(fd);
    }
    return node;
}

void MMseqsNode::WorkQueue::removeClaims() {
    for (size_t item = rank; item < itemCount; item += count) {
        const std::string file = name + "." + SSTR(item);
        if (FileUtil::fileExists(file.c_str())) {
            FileUtil::remove(file.c_str());
        }
    }
}

size_t MMseqsNode::getSmallestMemoryLimit(const std::string &file, size_t memoryLimit, int threads) {
    const std::string nodeLimits = barrier(file, SSTR(memoryLimit) + ((threads > 0) ? "\t" + SSTR(threads) : "") + "\n");
    size_t smallestMemoryLimit = memoryLimit;
    for (char *line = (char *) nodeLimits.c_str(); *line != '\0'; line = Util::skipLine(line)) {
        const char *columns[2];
        Util::getWordsOfLine(line, columns, 2);
        if (threads > 0 && Util::fast_atoi<int>(columns[1]) != threads) {
            Debug(Debug::ERROR) << "Run all nodes with the same --threads\n";
            EXIT(EXIT_FAILURE);
        }
        smallestMemoryLimit = std::min(smallestMemoryLimit, Util::fast_atoi<size_t>(columns[0]));
    }
    return smallestMemoryLimit;
}

void MMseqsNode::addExchangeFlush(std::string &indexData, size_t &writtenBytes, const std::vector<size_t> &nodeOffsets) {
    for (size_t node = 0; node < nodeOffsets.size(); node++) {
        const uint64_t offset = writtenBytes + nodeOffsets[node];
        indexData.append(reinterpret_cast<const char *>(&offset), sizeof(offset));
    }
    writtenBytes += nodeOffsets.back();
    __sync_fetch_and_add(&sentExchangeBytes, nodeOffsets.back());
    __sync_fetch_and_add(&sentOtherNodeExchangeBytes, nodeOffsets.back() - (nodeOffsets[rank + 1] - nodeOffsets[rank]));
}

void MMseqsNode::keepOwnExchangeRecords(const std::string &exchange, char *records, size_t bytes) {
    ownExchangeRecords[exchange].push_back(std::make_pair(records, bytes));
    __sync_fetch_and_add(&sentExchangeBytes, bytes);
}

std::string MMseqsNode::closeExchange(FILE *file, const std::string &fileName, const std::string &indexFile, const std::string &indexData) {
    const size_t flushCount = indexData.size() / ((count + 1) * sizeof(uint64_t));
    // the offsets of a node in all flushes follow each other, so a node reads its starts and ends at once
    std::vector<uint64_t> nodeFlushOffsets(indexData.size() / sizeof(uint64_t));
    for (size_t flush = 0; flush < flushCount; flush++) {
        for (size_t node = 0; node <= count; node++) {
            memcpy(&nodeFlushOffsets[node * flushCount + flush], indexData.data() + (flush * (count + 1) + node) * sizeof(uint64_t), sizeof(uint64_t));
        }
    }
    const size_t writtenBytes = (flushCount > 0) ? nodeFlushOffsets.back() : 0;
    if (fseeko(file, writtenBytes, SEEK_SET) != 0 || fwrite(nodeFlushOffsets.data(), sizeof(uint64_t), nodeFlushOffsets.size(), file) != nodeFlushOffsets.size()) {
        Debug(Debug::ERROR) << "Cannot write " << fileName << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (fclose(file) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << fileName << "\n";
        EXIT(EXIT_FAILURE);
    }
    return barrier(indexFile, SSTR(writtenBytes) + "\t" + SSTR(flushCount) + "\n");
}

void MMseqsNode::openExchange(const std::string &index, const std::string &prefix, std::vector<ExchangeBlock> &blocks, std::vector<FILE *> &files) {
    files.assign(count, NULL);
    std::vector<size_t> writtenBytes(count);
    std::vector<size_t> flushCounts(count);
    char *line = (char *) index.c_str();
    for (size_t node = 0; node < count; node++, line = Util::skipLine(line)) {
        const char *columns[2];
        Util::getWordsOfLine(line, columns, 2);
        writtenBytes[node] = Util::fast_atoi<size_t>(columns[0]);
        flushCounts[node] = Util::fast_atoi<size_t>(columns[1]);
    }
    // the starts then the ends of the part of this node in every flush of a node
    std::vector<std::vector<uint64_t> > flushOffsets(count);
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t node = 0; node < count; node++) {
        if (flushCounts[node] == 0) {
            continue;
        }
        const std::string file = prefix + "." + SSTR(node);
        // read and write, so the bytes of a block are freed once read
        files[node] = FileUtil::openFileOrDie(file.c_str(), "r+", true);
        flushOffsets[node].resize(2 * flushCounts[node]);
        FileUtil::preadAll(fileno(files[node]), file.c_str(), reinterpret_cast<char *>(flushOffsets[node].data()), flushOffsets[node].size() * sizeof(uint64_t),
                           writtenBytes[node] + rank * flushCounts[node] * sizeof(uint64_t));
    }
    for (size_t node = 0; node < count; node++) {
        const size_t nodeBlocks = blocks.size();
        for (size_t flush = 0; flush < flushCounts[node]; flush++) {
            const size_t pos = flushOffsets[node][flush];
            const size_t end = flushOffsets[node][flushCounts[node] + flush];
            if (pos < end) {
                const ExchangeBlock block = { files[node], pos, end, std::vector<char>(), 0, 0, 0, 0, NULL, pos };
                blocks.push_back(block);
            }
        }
        if (files[node] != NULL && blocks.size() == nodeBlocks) {
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
            const ExchangeBlock block = { NULL, 0, own->second[i].second, std::vector<char>(), 0, 0, 0, 0, own->second[i].first, 0 };
            blocks.push_back(block);
        }
        ownExchangeRecords.erase(own);
    }
}

static size_t getUnreadBytes(const MMseqsNode::ExchangeBlock &block) {
    return block.end - block.pos;
}

static const char *getBlockData(const MMseqsNode::ExchangeBlock &block) {
    return (block.ownRecords != NULL) ? block.ownRecords : block.buffer.data();
}

// a block frees the bytes that it has read in steps of at least PUNCH_HOLE_STEP that end at whole PUNCH_HOLE_ALIGNMENT, then its rest once read,
// so a few punches free a block of any size while the other nodes read their blocks of the same file
static const size_t PUNCH_HOLE_STEP = 1024 * 1024 * 1024;
static const size_t PUNCH_HOLE_ALIGNMENT = 1024 * 1024;

// buffers at least bytes of the unread records of a block, fewer only at its end, reading up to bufferSize; the records that this node sent to itself stay where they are
static void bufferExchangeBlock(MMseqsNode::ExchangeBlock &block, size_t bytes, size_t bufferSize, const std::string &prefix) {
    if (block.ownRecords != NULL) {
        block.bufferEnd = std::max(block.bufferEnd, std::min(block.end, block.bufferPos + std::max(bytes, bufferSize)));
        block.pos = block.bufferEnd;
        return;
    }
    if (block.bufferEnd - block.bufferPos >= bytes || getUnreadBytes(block) == 0) {
        return;
    }
    // the bytes before bufferPos were returned by an earlier read and no other node reads them
    const size_t punchEnd = (block.pos - block.bufferEnd + block.bufferPos) / PUNCH_HOLE_ALIGNMENT * PUNCH_HOLE_ALIGNMENT;
    if (punchEnd >= block.punchEnd + PUNCH_HOLE_STEP) {
        FileUtil::punchHole(fileno(block.file), block.punchEnd, punchEnd - block.punchEnd);
        block.punchEnd = punchEnd;
    }
    if (block.bufferEnd > block.bufferPos) {
        memmove(block.buffer.data(), block.buffer.data() + block.bufferPos, block.bufferEnd - block.bufferPos);
    }
    block.bufferEnd -= block.bufferPos;
    block.recordEnd -= std::min(block.recordEnd, block.bufferPos);
    block.lastRecord -= std::min(block.lastRecord, block.bufferPos);
    block.bufferPos = 0;
    const size_t blockBufferSize = std::min(std::max(bytes, bufferSize), block.bufferEnd + getUnreadBytes(block));
    if (block.buffer.size() < blockBufferSize) {
        block.buffer.resize(blockBufferSize);
    }
    // outside a parallel region the threads read pieces of the block
    std::vector<std::pair<size_t, size_t> > reads;
    while (block.bufferEnd < block.buffer.size() && getUnreadBytes(block) > 0) {
        const size_t readBytes = std::min(std::min(block.buffer.size() - block.bufferEnd, block.end - block.pos), MAX_READ_REQUEST_SIZE);
        reads.push_back(std::make_pair(block.pos, block.bufferEnd));
        block.pos += readBytes;
        block.bufferEnd += readBytes;
    }
    const size_t readCount = reads.size();
    reads.push_back(std::make_pair(block.pos, block.bufferEnd));
#pragma omp parallel for schedule(dynamic, 1) if (omp_in_parallel() == 0)
    for (size_t i = 0; i < readCount; i++) {
        FileUtil::preadAll(fileno(block.file), prefix.c_str(), block.buffer.data() + reads[i].second, reads[i + 1].second - reads[i].second, reads[i].first);
    }
}

// frees a block once it has passed all its records: the records that this node sent to itself, or its buffer and its bytes in the file
static void releaseReadBlock(MMseqsNode::ExchangeBlock &block) {
    if (getUnreadBytes(block) == 0 && block.bufferEnd > 0 && block.bufferPos == block.bufferEnd) {
        if (block.ownRecords != NULL) {
            Util::releasePages(block.ownRecords, block.end);
            free(block.ownRecords);
            block.ownRecords = NULL;
        } else {
            FileUtil::punchHole(fileno(block.file), block.punchEnd, block.end - block.punchEnd);
            std::vector<char>().swap(block.buffer);
        }
        block.bufferPos = 0;
        block.bufferEnd = 0;
        block.recordEnd = 0;
        block.lastRecord = 0;
    }
}

size_t MMseqsNode::readExchangeRange(std::vector<ExchangeBlock> &blocks, size_t bufferMemory, size_t (*recordSize)(const char *, size_t),
                                     bool (*compare)(const ClusterPair &, const ClusterPair &), const ClusterPair &last,
                                     std::vector<char> &records, std::vector<size_t> *recordOffsets, const std::string &prefix) {
    // the buffers of the blocks and the copy of their records each take half of bufferMemory
    const size_t bufferSize = std::max(EXCHANGE_BUFFER_SIZE, bufferMemory / 2 / std::max<size_t>(1, blocks.size()));
    // with fewer blocks than threads, the threads share the reads and copies of a block
    const size_t threads = static_cast<size_t>(Parameters::getInstance().threads);
    const bool isClusterPair = recordSize == NULL;
    std::vector<char> isRead(blocks.size());
    for (size_t i = 0; i < blocks.size(); i++) {
        const ExchangeBlock &block = blocks[i];
        isRead[i] = getUnreadBytes(block) > 0 && (block.bufferEnd - block.bufferPos < bufferSize / 2 || block.recordEnd == block.bufferPos);
    }
#pragma omp parallel for schedule(dynamic, 1) if (blocks.size() >= threads)
    for (size_t i = 0; i < blocks.size(); i++) {
        if (isRead[i]) {
            bufferExchangeBlock(blocks[i], bufferSize, bufferSize, prefix);
        }
    }
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < blocks.size(); i++) {
        ExchangeBlock &block = blocks[i];
        if (isRead[i] == false) {
            continue;
        }
        // a record larger than the buffer doubles it
        for (size_t bytes = bufferSize; ; bytes *= 2) {
            bufferExchangeBlock(block, bytes, bytes, prefix);
            const char *data = getBlockData(block);
            if (isClusterPair && block.bufferEnd - block.recordEnd >= sizeof(ClusterPair)) {
                block.recordEnd += (block.bufferEnd - block.recordEnd) / sizeof(ClusterPair) * sizeof(ClusterPair);
                block.lastRecord = block.recordEnd - sizeof(ClusterPair);
            }
            for (size_t size; isClusterPair == false && (size = recordSize(data + block.recordEnd, block.bufferEnd - block.recordEnd)) > 0; block.recordEnd += size) {
                block.lastRecord = block.recordEnd;
            }
            if (block.recordEnd > block.bufferPos || getUnreadBytes(block) == 0) {
                break;
            }
        }
        if (getUnreadBytes(block) == 0 && block.recordEnd != block.bufferEnd) {
            Debug(Debug::ERROR) << "Cannot read " << prefix << ": a record ends after its block\n";
            EXIT(EXIT_FAILURE);
        }
    }
    // the records up to the last buffered record of every block that is not read to its end are all buffered
    ClusterPair rangeLast = last;
    for (size_t i = 0; i < blocks.size(); i++) {
        if (getUnreadBytes(blocks[i]) > 0) {
            ClusterPair blockLast;
            memcpy(&blockLast, getBlockData(blocks[i]) + blocks[i].lastRecord, sizeof(ClusterPair));
            if (compare(blockLast, rangeLast)) {
                rangeLast = blockLast;
            }
        }
    }
    std::vector<size_t> rangeEnds(blocks.size());
    std::vector<size_t> blockByteOffsets(blocks.size() + 1, 0);
    std::vector<size_t> blockRecordOffsets(blocks.size() + 1, 0);
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < blocks.size(); i++) {
        const ExchangeBlock &block = blocks[i];
        const char *data = getBlockData(block);
        size_t pos = block.bufferPos;
        size_t recordCount = 0;
        if (isClusterPair) {
            // the records of a block are in order
            const ClusterPair *pairs = reinterpret_cast<const ClusterPair *>(data + block.bufferPos);
            recordCount = std::upper_bound(pairs, pairs + (block.recordEnd - block.bufferPos) / sizeof(ClusterPair), rangeLast, compare) - pairs;
            pos += recordCount * sizeof(ClusterPair);
        }
        for (ClusterPair pair; isClusterPair == false && pos < block.recordEnd; pos += recordSize(data + pos, block.recordEnd - pos), recordCount++) {
            memcpy(&pair, data + pos, sizeof(ClusterPair));
            if (compare(rangeLast, pair)) {
                break;
            }
        }
        rangeEnds[i] = pos;
        blockByteOffsets[i + 1] = pos - block.bufferPos;
        blockRecordOffsets[i + 1] = recordCount;
    }
    for (size_t i = 0; i < blocks.size(); i++) {
        blockByteOffsets[i + 1] += blockByteOffsets[i];
        blockRecordOffsets[i + 1] += blockRecordOffsets[i];
    }
    if (records.size() < blockByteOffsets.back()) {
        records.resize(blockByteOffsets.back());
    }
    if (recordOffsets != NULL) {
        recordOffsets->resize(blockRecordOffsets.back() + 1);
        recordOffsets->back() = blockByteOffsets.back();
    }
    // the copies are cut into a piece per thread
    const size_t pieceSize = std::max<size_t>(EXCHANGE_BUFFER_SIZE, blockByteOffsets.back() / threads + 1);
    const size_t pieceCount = (blockByteOffsets.back() + pieceSize - 1) / pieceSize;
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t piece = 0; piece < pieceCount; piece++) {
        const size_t pieceBegin = piece * pieceSize;
        const size_t pieceEnd = std::min(blockByteOffsets.back(), pieceBegin + pieceSize);
        for (size_t i = std::upper_bound(blockByteOffsets.begin(), blockByteOffsets.end(), pieceBegin) - blockByteOffsets.begin() - 1; i < blocks.size() && blockByteOffsets[i] < pieceEnd; i++) {
            const size_t from = std::max(pieceBegin, blockByteOffsets[i]);
            const size_t to = std::min(pieceEnd, blockByteOffsets[i + 1]);
            memcpy(records.data() + from, getBlockData(blocks[i]) + blocks[i].bufferPos + (from - blockByteOffsets[i]), to - from);
        }
    }
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < blocks.size(); i++) {
        ExchangeBlock &block = blocks[i];
        if (recordOffsets != NULL) {
            size_t pos = blockByteOffsets[i];
            for (size_t record = blockRecordOffsets[i]; record < blockRecordOffsets[i + 1]; record++) {
                (*recordOffsets)[record] = pos;
                pos += isClusterPair ? sizeof(ClusterPair) : recordSize(records.data() + pos, blockByteOffsets[i + 1] - pos);
            }
        }
        block.bufferPos = rangeEnds[i];
        releaseReadBlock(block);
    }
    return blockRecordOffsets.back();
}

size_t MMseqsNode::readExchangeSegments(std::vector<ExchangeBlock> &blocks, size_t bufferMemory, size_t maxSegmentSize, uint64_t fileType, size_t windowBegin, size_t windowEnd,
                                        char *window, std::vector<std::pair<size_t, size_t> > &segments, const std::string &prefix) {
    const size_t bufferSize = std::max(4 * sizeof(uint64_t) + maxSegmentSize, bufferMemory / std::max<size_t>(1, blocks.size()));
    size_t nextOffset = SIZE_MAX;
#pragma omp parallel reduction(min:nextOffset)
    {
        std::vector<std::pair<size_t, size_t> > threadSegments;
        ZSTD_DCtx *dctx = ZSTD_createDCtx();
        if (dctx == NULL) {
            Debug(Debug::ERROR) << "ZSTD_createDCtx() error\n";
            EXIT(EXIT_FAILURE);
        }
#pragma omp for schedule(dynamic, 1) nowait
        for (size_t i = 0; i < blocks.size(); i++) {
            ExchangeBlock &block = blocks[i];
            uint64_t segment[4];
            for (;;) {
                bufferExchangeBlock(block, sizeof(segment), bufferSize, prefix);
                if (block.bufferEnd - block.bufferPos < sizeof(segment)) {
                    break;
                }
                memcpy(segment, getBlockData(block) + block.bufferPos, sizeof(segment));
                if (segment[0] != fileType || segment[1] >= windowEnd) {
                    if (segment[0] == fileType) {
                        nextOffset = std::min<size_t>(nextOffset, segment[1]);
                    }
                    break;
                }
                bufferExchangeBlock(block, sizeof(segment) + segment[3], bufferSize, prefix);
                if (block.bufferEnd - block.bufferPos < sizeof(segment) + segment[3]) {
                    Debug(Debug::ERROR) << "Cannot read " << prefix << ": a record ends after its block\n";
                    EXIT(EXIT_FAILURE);
                }
                const char *data = getBlockData(block) + block.bufferPos + sizeof(segment);
                if (segment[3] == segment[2]) {
                    memcpy(window + (segment[1] - windowBegin), data, segment[2]);
                } else if (ZSTD_decompressDCtx(dctx, window + (segment[1] - windowBegin), segment[2], data, segment[3]) != segment[2]) {
                    Debug(Debug::ERROR) << "Cannot decompress a record of " << prefix << "\n";
                    EXIT(EXIT_FAILURE);
                }
                threadSegments.push_back(std::make_pair(segment[1], segment[2]));
                block.bufferPos += sizeof(segment) + segment[3];
            }
            releaseReadBlock(block);
        }
        ZSTD_freeDCtx(dctx);
#pragma omp critical
        segments.insert(segments.end(), threadSegments.begin(), threadSegments.end());
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

// sorts a flush of pairs, appends it to file and adds its offsets; keeps up to ownPairLimit of the pairs for this node at the start of pairs instead and returns their count
static size_t writePairs(ClusterPair *pairs, size_t pairCount, bool (*compare)(const ClusterPair &, const ClusterPair &), const std::vector<size_t> &nodeKeyOffsets,
                         FILE *file, const std::string &fileName, std::string &indexData, size_t &writtenBytes, size_t ownPairLimit) {
    SORT_PARALLEL(pairs, pairs + pairCount, compare);
    std::vector<size_t> nodeOffsets(MMseqsNode::count + 1, pairCount * sizeof(ClusterPair));
    for (size_t node = 0; node < MMseqsNode::count; node++) {
        const ClusterPair nodeFirstPair = { static_cast<DBKeyType>(nodeKeyOffsets[node]), static_cast<DBKeyType>(nodeKeyOffsets[node]) };
        nodeOffsets[node] = (std::lower_bound(pairs, pairs + pairCount, nodeFirstPair, compare) - pairs) * sizeof(ClusterPair);
    }
    const size_t ownBegin = nodeOffsets[MMseqsNode::rank] / sizeof(ClusterPair);
    const size_t ownEnd = ownBegin + std::min(nodeOffsets[MMseqsNode::rank + 1] / sizeof(ClusterPair) - ownBegin, ownPairLimit);
    if (fwrite(pairs, sizeof(ClusterPair), ownBegin, file) != ownBegin || fwrite(pairs + ownEnd, sizeof(ClusterPair), pairCount - ownEnd, file) != pairCount - ownEnd) {
        Debug(Debug::ERROR) << "Cannot write " << fileName << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (ownBegin > 0) {
        memmove(pairs, pairs + ownBegin, (ownEnd - ownBegin) * sizeof(ClusterPair));
    }
    for (size_t node = MMseqsNode::rank + 1; node <= MMseqsNode::count; node++) {
        nodeOffsets[node] -= (ownEnd - ownBegin) * sizeof(ClusterPair);
    }
    MMseqsNode::addExchangeFlush(indexData, writtenBytes, nodeOffsets);
    return ownEnd - ownBegin;
}

MMseqsNode::ClusterPairWriter::ClusterPairWriter(const std::string &exchange, size_t pairCapacity, bool (*compare)(const ClusterPair &, const ClusterPair &), const std::vector<size_t> &nodeKeyOffsets)
    : exchange(exchange), fileName(exchange + "." + SSTR(rank)), file(FileUtil::openAndDelete(fileName.c_str(), "w")), pairs(static_cast<ClusterPair *>(malloc(pairCapacity * sizeof(ClusterPair)))),
      pairCapacity(pairCapacity), pairCount(0), compare(compare), nodeKeyOffsets(nodeKeyOffsets), writtenBytes(0), keptPairCount(0) {
    Util::checkAllocation(pairs, "Can not allocate pairs memory in ClusterPairWriter");
}

MMseqsNode::ClusterPairWriter::~ClusterPairWriter() {
    free(pairs);
}

void MMseqsNode::ClusterPairWriter::flush() {
    // the pairs for this node stay in memory while all that it keeps fit into as much as its buffer
    const size_t ownPairCount = writePairs(pairs, pairCount, compare, nodeKeyOffsets, file, fileName, indexData, writtenBytes, pairCapacity - keptPairCount);
    if (ownPairCount > 0) {
        ClusterPair *ownPairs = static_cast<ClusterPair *>(malloc(ownPairCount * sizeof(ClusterPair)));
        Util::checkAllocation(ownPairs, "Can not allocate own pairs memory in ClusterPairWriter");
        memcpy(ownPairs, pairs, ownPairCount * sizeof(ClusterPair));
        keepOwnExchangeRecords(exchange, reinterpret_cast<char *>(ownPairs), ownPairCount * sizeof(ClusterPair));
        keptPairCount += ownPairCount;
    }
    pairCount = 0;
}

std::string MMseqsNode::ClusterPairWriter::close() {
    size_t ownPairCount = 0;
    if (pairCount > 0) {
        ownPairCount = writePairs(pairs, pairCount, compare, nodeKeyOffsets, file, fileName, indexData, writtenBytes, pairCount);
        pairCount = 0;
    }
    // the buffer, cut to the pairs for this node, becomes their records
    if (ownPairCount > 0) {
        pairs = static_cast<ClusterPair *>(realloc(pairs, ownPairCount * sizeof(ClusterPair)));
        Util::checkAllocation(pairs, "Can not shrink pairs memory in ClusterPairWriter");
        keepOwnExchangeRecords(exchange, reinterpret_cast<char *>(pairs), ownPairCount * sizeof(ClusterPair));
    } else {
        Util::releasePages(pairs, pairCapacity * sizeof(ClusterPair));
        free(pairs);
    }
    pairs = NULL;
    return closeExchange(file, fileName, exchange + ".index", indexData);
}

// a loaded and a prefetched split of a cluster database take a quarter of the memory, half for entry bytes and half for index cells
static const size_t CLUSTER_SPLIT_MEMORY_DIVISOR = 16;
// the received records read at once
static const size_t RECEIVED_MEMORY_DIVISOR = 64;
// the pairs sent at once
static const size_t SENT_PAIRS_MEMORY_DIVISOR = 2;

static void writeKeyBits(const std::string &bitsFile, const uint64_t *bits, size_t firstKey, size_t keyCount) {
    FILE *file = FileUtil::openAndDelete(bitsFile.c_str(), "w");
    const uint64_t header[2] = { firstKey, keyCount };
    const size_t words = (keyCount + 63) / 64;
    if (fwrite(header, sizeof(uint64_t), 2, file) != 2 || fwrite(bits, sizeof(uint64_t), words, file) != words
        || fclose(file) != 0) {
        Debug(Debug::ERROR) << "Cannot write " << bitsFile << "\n";
        EXIT(EXIT_FAILURE);
    }
}

// reads the next keys of entry that fit a split and moves entry past them, false once all are read; a key cut off by the read is read again with the next
static bool readEntryKeys(DBReader<DBKeyType> &cluDb, DBReader<DBKeyType>::Index &entry, size_t memoryLimit, std::vector<char> &buffer, std::vector<DBKeyType> &keys) {
    buffer.resize(std::min(memoryLimit / CLUSTER_SPLIT_MEMORY_DIVISOR, MAX_INFLIGHT_READ_SIZE));
    // the entry ends with '\0' after its last newline
    const size_t bytes = std::min(buffer.size() - 1, entry.length - 1);
    if (bytes == 0) {
        std::vector<char>().swap(buffer);
        return false;
    }
    cluDb.readData(entry.offset, bytes, buffer.data());
    size_t keyBytes = bytes;
    while (buffer[keyBytes - 1] != '\n') {
        keyBytes--;
    }
    buffer[keyBytes] = '\0';
    keys.clear();
    char keyBuffer[255];
    for (char *data = buffer.data(); *data != '\0'; data = Util::skipLine(data)) {
        Util::parseKey(data, keyBuffer);
        keys.push_back(Util::fast_atoi<DBKeyType>(keyBuffer));
    }
    entry.offset += keyBytes;
    entry.length -= keyBytes;
    return true;
}

// loads the entries of keys [keyBegin, keyEnd) of the parts <clu>.<node>, whose keys start at cluKeyOffsets, or all of <clu>.<rank> without cluKeyOffsets, split by split,
// prefetching the next, and calls processSplit(cluDb, splitBegin, splitEnd, entryKeys, streamedEntry); a larger entry is a split of its own that is not loaded, streamedEntry for readEntryKeys, otherwise NULL
template<typename ProcessSplit>
static void readClusterSplits(const std::string &clu, const std::vector<size_t> &cluKeyOffsets, size_t keyBegin, size_t keyEnd, size_t memoryLimit, ProcessSplit processSplit) {
    for (size_t node = 0; node < MMseqsNode::count; node++) {
        if (cluKeyOffsets.empty() ? node != MMseqsNode::rank : std::max(keyBegin, cluKeyOffsets[node]) >= std::min(keyEnd, cluKeyOffsets[node + 1])) {
            continue;
        }
        DBReader<DBKeyType> cluDb((clu + "." + SSTR(node)).c_str(), (clu + ".index." + SSTR(node)).c_str(), Parameters::getInstance().threads,
                                  DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
        cluDb.open(DBReader<DBKeyType>::NOSORT);
        const size_t entryBegin = cluKeyOffsets.empty() ? 0 : cluDb.findEntryByKey(keyBegin);
        const size_t entryEnd = cluKeyOffsets.empty() ? cluDb.getSize() : cluDb.findEntryByKey(keyEnd);
        const size_t splitBytes = memoryLimit / CLUSTER_SPLIT_MEMORY_DIVISOR;
        const size_t splitEntries = std::max<size_t>(1, memoryLimit / CLUSTER_SPLIT_MEMORY_DIVISOR / (2 * sizeof(uint64_t) + 2 * sizeof(size_t)));
        // an empty split if its first entry is larger than splitBytes
        const auto findSplitEnd = [&](size_t splitBegin, size_t splitOffset) {
            const size_t bytesEnd = (splitOffset + splitBytes >= cluDb.getDataSize()) ? cluDb.getSize() : cluDb.findEntryByOffset(splitOffset + splitBytes + 1) - 1;
            return std::min(std::min(bytesEnd, splitBegin + splitEntries), entryEnd);
        };
        std::vector<DBKeyType> entryKeys;
        DBReader<DBKeyType>::Index entry;
        size_t splitOffset = (entryBegin < entryEnd) ? cluDb.getIndexEntry(entryBegin).offset : 0;
        for (size_t splitBegin = entryBegin, splitEnd = findSplitEnd(entryBegin, splitOffset), nextSplitEnd; splitBegin < entryEnd; splitBegin = splitEnd, splitEnd = nextSplitEnd) {
            const bool isStreamed = splitEnd == splitBegin;
            if (isStreamed) {
                entry = cluDb.getIndexEntry(splitBegin);
                entryKeys.assign(1, entry.id);
                splitOffset += entry.length;
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
            nextSplitEnd = findSplitEnd(splitEnd, splitOffset);
            if (nextSplitEnd > splitEnd) {
                cluDb.prefetchData(splitEnd, nextSplitEnd);
            }
            processSplit(cluDb, splitBegin, splitEnd, entryKeys, isStreamed ? &entry : NULL);
        }
        cluDb.close();
    }
}

std::string MMseqsNode::sendClusterPairs(const std::string &clu, bool withRepresentative, bool (*compare)(const ClusterPair &, const ClusterPair &), const std::vector<size_t> &nodeKeyOffsets,
                                         const std::string &exchange, const std::string &receivedIndex, const std::string &receivedPrefix,
                                         uint64_t *representativeBits, const std::vector<size_t> &cluKeyOffsets, size_t firstKey, size_t keyCount) {
    std::vector<ExchangeBlock> receivedBlocks;
    std::vector<FILE *> receivedFiles;
    if (receivedPrefix.empty() == false) {
        openExchange(receivedIndex, receivedPrefix, receivedBlocks, receivedFiles);
    }
    const size_t memoryLimit = Util::computeMemory(Parameters::getInstance().splitMemoryLimit);
    // half for the buffered pairs and half for those that the writer keeps for this node
    ClusterPairWriter pairWriter(exchange, memoryLimit / SENT_PAIRS_MEMORY_DIVISOR / 2 / sizeof(ClusterPair), compare, nodeKeyOffsets);

    std::vector<DBKeyType> repKeys;
    std::vector<size_t> pairOffsets;
    std::vector<char> received;
    const auto clearRepresentativeBit = [&](DBKeyType key) {
        representativeBits[(key - firstKey) / 64] &= ~(1ULL << ((key - firstKey) % 64));
    };
    // the members with an entry have the repKey DB_KEY_INVALID
    const auto sendMembersWithoutEntry = [&](const ClusterPair *pairs, size_t pairCount) {
        for (size_t i = 0; i < pairCount; i++) {
            if (pairs[i].repKey != DB_KEY_INVALID) {
                pairWriter.add(pairs[i].repKey, pairs[i].memberKey);
                clearRepresentativeBit(pairs[i].memberKey);
            }
        }
    };
    std::vector<char> entryBuffer;
    std::vector<DBKeyType> streamedKeys;
    readClusterSplits(clu, cluKeyOffsets, firstKey, firstKey + keyCount, memoryLimit, [&](DBReader<DBKeyType> &cluDb, size_t splitBegin, size_t splitEnd, const std::vector<DBKeyType> &entryKeys, DBReader<DBKeyType>::Index *streamedEntry) {
        repKeys = entryKeys;
        if (representativeBits != NULL) {
            for (size_t i = 0; i < entryKeys.size(); i++) {
                clearRepresentativeBit(entryKeys[i]);
            }
        }
        const ClusterPair splitLast = { entryKeys.back(), entryKeys.back() };
        size_t receivedCount;
        while ((receivedCount = readExchangeRange(receivedBlocks, memoryLimit / RECEIVED_MEMORY_DIVISOR, NULL, ClusterPair::compareByMember, splitLast, received, NULL, receivedPrefix)) > 0) {
            ClusterPair *receivedPairs = reinterpret_cast<ClusterPair *>(received.data());
            SORT_PARALLEL(receivedPairs, receivedPairs + receivedCount, ClusterPair::compareByMember);
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < receivedCount; i++) {
                const DBKeyType memberKey = receivedPairs[i].memberKey;
                const size_t entry = std::lower_bound(entryKeys.begin(), entryKeys.end(), memberKey) - entryKeys.begin();
                const bool isEntry = entry < entryKeys.size() && entryKeys[entry] == memberKey;
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
                sendMembersWithoutEntry(receivedPairs, receivedCount);
            }
        }
        if (streamedEntry != NULL) {
            while (readEntryKeys(cluDb, *streamedEntry, memoryLimit, entryBuffer, streamedKeys)) {
                for (size_t i = 0; i < streamedKeys.size(); i++) {
                    if (withRepresentative || streamedKeys[i] != entryKeys[0]) {
                        pairWriter.add(repKeys[0], streamedKeys[i]);
                    }
                }
            }
            return;
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
            const size_t firstPair = pairOffsets[firstEntry - splitBegin];
            const size_t endEntry = splitBegin + (std::upper_bound(pairOffsets.begin() + (firstEntry - splitBegin) + 1, pairOffsets.end(),
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
                    const DBKeyType entryKey = cluDb.getDbKey(i);
                    char *data = cluDb.getData(i, thread_idx);
                    size_t pairOffset = pairWriter.pairCount + pairOffsets[i - splitBegin] - firstPair;
                    while (*data != '\0') {
                        Util::parseKey(data, keyBuffer);
                        const DBKeyType key = Util::fast_atoi<DBKeyType>(keyBuffer);
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
    });
    const ClusterPair lastPair = { DB_KEY_INVALID, DB_KEY_INVALID };
    size_t receivedCount;
    while ((receivedCount = readExchangeRange(receivedBlocks, memoryLimit / RECEIVED_MEMORY_DIVISOR, NULL, ClusterPair::compareByMember, lastPair, received, NULL, receivedPrefix)) > 0) {
        if (representativeBits == NULL) {
            Debug(Debug::ERROR) << "Key " << reinterpret_cast<ClusterPair *>(received.data())->memberKey << " of " << receivedPrefix << " is no entry of " << clu << "." << rank << "\n";
            EXIT(EXIT_FAILURE);
        }
        sendMembersWithoutEntry(reinterpret_cast<ClusterPair *>(received.data()), receivedCount);
    }
    closeExchangeFiles(receivedFiles, receivedPrefix);
    return pairWriter.close();
}

void MMseqsNode::readKeyBits(int fd, const std::string &fileName, size_t fileOffset, size_t fileFirstKey, size_t from, size_t to, size_t firstKey, uint64_t *bits) {
    if (from >= to) {
        return;
    }
    const size_t firstWord = (from - fileFirstKey) / 64;
    std::vector<uint64_t> fileBits((to - 1 - fileFirstKey) / 64 + 1 - firstWord);
    FileUtil::preadAll(fd, fileName.c_str(), reinterpret_cast<char *>(fileBits.data()), fileBits.size() * sizeof(uint64_t), fileOffset + firstWord * sizeof(uint64_t));
    // a word of the file goes to bit (its first key - firstKey) of bits, without its keys outside [from, to)
    for (size_t word = 0; word < fileBits.size(); word++) {
        const size_t wordKey = fileFirstKey + (firstWord + word) * 64;
        uint64_t value = fileBits[word];
        value &= (wordKey < from) ? ~0ULL << (from - wordKey) : ~0ULL;
        value &= (to - wordKey < 64) ? (1ULL << (to - wordKey)) - 1 : ~0ULL;
        if (wordKey < firstKey) {
            value >>= firstKey - wordKey;
        }
        const size_t bit = (wordKey < firstKey) ? 0 : wordKey - firstKey;
        bits[bit / 64] |= value << (bit % 64);
        if (bit % 64 != 0) {
            bits[bit / 64 + 1] |= value >> (64 - bit % 64);
        }
    }
}

uint64_t *MMseqsNode::readRepresentativeBits(const std::string &db, const std::vector<size_t> &fileKeyOffsets, size_t firstKey, size_t keyCount) {
    const size_t words = (keyCount + 63) / 64 + 1;
    uint64_t *bits = new(std::nothrow) uint64_t[words];
    Util::checkAllocation(bits, "Can not allocate representative bit memory");
    memset(bits, 0, words * sizeof(uint64_t));
    for (size_t node = 0; node + 1 < fileKeyOffsets.size(); node++) {
        const size_t from = std::max(firstKey, fileKeyOffsets[node]);
        const size_t to = std::min(firstKey + keyCount, fileKeyOffsets[node + 1]);
        if (from >= to) {
            continue;
        }
        const std::string bitsFile = db + ".bits." + SSTR(node);
        FILE *file = FileUtil::openFileOrDie(bitsFile.c_str(), "r", true);
        uint64_t keyRange[2];
        FileUtil::preadAll(fileno(file), bitsFile.c_str(), reinterpret_cast<char *>(keyRange), sizeof(keyRange), 0);
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
    // the ranges of this node in the sequence index, with the number of its representatives as a sixth column
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    std::string representativeIndex;
    for (size_t range = 0; range < seqDbr.getIndexSize(); range++) {
        if (seqDbr.getIndexFile(range) != rank) {
            continue;
        }
        const size_t rangeEndKey = (range + 1 < seqDbr.getIndexSize()) ? index[range + 1].id : seqDbr.getSize();
        const size_t rangeRepresentatives = Util::countSetBits(bits, index[range].id - firstKey, rangeEndKey - firstKey);
        representativeIndex += SSTR(index[range].id) + "\t" + SSTR(index[range].offset) + "\t" + SSTR(index[range].length) + "\t"
                             + SSTR(rangeEndKey - index[range].id) + "\t" + SSTR(rank) + "\t" + SSTR(rangeRepresentatives) + "\n";
    }
    publishRunFile(db + ".index." + SSTR(rank), representativeIndex);
    if (isMaster()) {
        publishFile(db + ".index", gatherRunFiles(db + ".index."));
    }
}

std::vector<size_t> MMseqsNode::getClusterKeyOffsets(const std::string &file, const std::string &clu, size_t keyCount, const std::string &firstKeyFile) {
    const std::string cluIndex = firstKeyFile.empty() ? clu + ".index." + SSTR(rank) : firstKeyFile;
    FILE *handle = FileUtil::openFileOrDie(cluIndex.c_str(), "r", true);
    uint64_t firstCell[2];
    if (fread(firstCell, sizeof(uint64_t), 2, handle) != 2 || fclose(handle) != 0) {
        Debug(Debug::ERROR) << "Cannot read " << cluIndex << "\n";
        EXIT(EXIT_FAILURE);
    }
    const std::string firstKeys = barrier(file, SSTR(firstCell[0]) + "\n");
    removeOnceRead(file);
    std::vector<size_t> offsets(count + 1, keyCount);
    char *line = (char *) firstKeys.c_str();
    for (size_t node = 0; node < count; node++, line = Util::skipLine(line)) {
        offsets[node] = Util::fast_atoi<size_t>(line);
    }
    // a part without entries starts with its end cell, so it starts where the next part starts
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
                const size_t entry = cluDb.findEntryByOffset(partOffsets[count] * split / count - partOffsets[node]);
                const size_t firstKey = (entry < cluDb.getSize()) ? static_cast<size_t>(cluDb.getIndexEntry(entry).id) : static_cast<size_t>(cluDb.getIndexEntry(cluDb.getSize() - 1).id) + 1;
                firstKeys += SSTR(firstKey) + "\n";
            }
            cluDb.close();
        }
        for (; split < count; split++) {
            firstKeys += SSTR(keyCount) + "\n";
        }
        publishRunFile(file, firstKeys);
    }
    const std::string firstKeys = waitForRunFile(file);
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
    // the (key, offset) cells in key order that a thread reads at once
    const size_t pieceEntries = 1024 * 1024 / (2 * sizeof(uint64_t));
    for (size_t node = 0; node < count; node++) {
        const size_t from = std::max(keyBegin, cluKeyOffsets[node]);
        const size_t to = std::min(keyEnd, cluKeyOffsets[node + 1]);
        if (from >= to) {
            continue;
        }
        const std::string cluData = clu + "." + SSTR(node);
        const std::string cluIndex = clu + ".index." + SSTR(node);
        DBReader<DBKeyType> cluDb(cluData.c_str(), cluIndex.c_str(), 1, DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
        cluDb.open(DBReader<DBKeyType>::NOSORT);
        const size_t entryBegin = cluDb.findEntryByKey(from);
        const size_t entryEnd = cluDb.findEntryByKey(to);
        cluDb.close();
        FILE *cluIndexFile = FileUtil::openFileOrDie(cluIndex.c_str(), "r", true);
        const size_t nodeKeysBegin = keys.size();
        keys.resize(nodeKeysBegin + entryEnd - entryBegin);
#pragma omp parallel
        {
            std::vector<uint64_t> cells(2 * pieceEntries);
#pragma omp for schedule(dynamic, 1)
            for (size_t entry = entryBegin; entry < entryEnd; entry += pieceEntries) {
                const size_t cellCount = std::min(pieceEntries, entryEnd - entry);
                FileUtil::preadAll(fileno(cluIndexFile), cluIndex.c_str(), reinterpret_cast<char *>(cells.data()), cellCount * 2 * sizeof(uint64_t), entry * 2 * sizeof(uint64_t));
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

void MMseqsNode::writeClusterThreadParts(FILE *dataFile, const std::string &dataFileName, FILE *indexFile, const std::string &indexFileName,
                                         std::vector<std::string> &threadData, std::vector<std::vector<uint64_t> > &threadEntries, size_t &dataBytes, size_t &indexBytes) {
    const int threads = static_cast<int>(threadData.size());
    std::vector<size_t> threadDataOffsets(threads + 1);
    std::vector<size_t> threadEntryOffsets(threads + 1);
    threadDataOffsets[0] = dataBytes;
    threadEntryOffsets[0] = indexBytes;
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
        FileUtil::pwriteAll(fileno(indexFile), indexFileName.c_str(), reinterpret_cast<const char *>(entries.data()), entries.size() * sizeof(uint64_t), threadEntryOffsets[thread]);
    }
    dataBytes = threadDataOffsets[threads];
    indexBytes = threadEntryOffsets[threads];
}

void MMseqsNode::closeClusterFiles(FILE *dataFile, const std::string &dataFileName, FILE *indexFile, const std::string &indexFileName, size_t dataBytes, size_t indexBytes) {
    const uint64_t endEntry[2] = { static_cast<uint64_t>(DB_KEY_INVALID), static_cast<uint64_t>(dataBytes) };
    FileUtil::pwriteAll(fileno(indexFile), indexFileName.c_str(), reinterpret_cast<const char *>(endEntry), sizeof(endEntry), indexBytes);
    if (fclose(dataFile) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << dataFileName << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (fclose(indexFile) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << indexFileName << "\n";
        EXIT(EXIT_FAILURE);
    }
}

// how many of the sorted pairs come before position pos once they are merged with the sorted singletons in rep order
static size_t findMergedPairCount(const ClusterPair *pairs, size_t pairCount, const ClusterPair *singletons, size_t singletonCount, size_t pos) {
    size_t low = (pos > singletonCount) ? pos - singletonCount : 0;
    size_t high = std::min(pos, pairCount);
    while (low < high) {
        const size_t mid = low + (high - low) / 2;
        if (ClusterPair::compareByRep(pairs[mid], singletons[pos - mid - 1])) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

size_t MMseqsNode::writeClusterPairs(const std::string &index, const std::string &prefix, const std::string &clu, const uint64_t *singletonBits, size_t firstKey, size_t keyCount) {
    const Parameters &par = Parameters::getInstance();
    std::vector<ExchangeBlock> repBlocks;
    std::vector<FILE *> repFiles;
    openExchange(index, prefix, repBlocks, repFiles);
    const size_t memoryLimit = Util::computeMemory(par.splitMemoryLimit);
    const std::string resultDb = clu + "." + SSTR(rank);
    const std::string resultIndex = clu + ".index." + SSTR(rank);
    FILE *resultDataFile = FileUtil::openAndDelete(resultDb.c_str(), "w");
    FILE *resultIndexFile = FileUtil::openAndDelete(resultIndex.c_str(), "w");
    size_t resultDataBytes = 0;
    size_t resultIndexBytes = 0;
    std::vector<char> records;
    std::vector<ClusterPair> singletons;
    std::vector<size_t> threadSingletonOffsets(par.threads + 1, 0);
    std::vector<std::string> threadData(par.threads);
    // (key, offset) of the entries that start in the part of a thread
    std::vector<std::vector<uint64_t> > threadIndex(par.threads);
    DBKeyType previousRepKey = DB_KEY_INVALID;
    size_t lineCount = 0;
    // the keys go in splits, so the pairs of their singletons fit into memory
    const size_t splitKeys = std::max<size_t>(1, memoryLimit / 16 / sizeof(ClusterPair));
    const size_t keyEnd = firstKey + keyCount;
    for (size_t splitBegin = firstKey; splitBegin < keyEnd; splitBegin += splitKeys) {
        const size_t splitEnd = std::min(keyEnd, splitBegin + splitKeys);
        const ClusterPair lastPair = { static_cast<DBKeyType>(splitEnd - 1), DB_KEY_INVALID };
        size_t nextSingletonKey = splitBegin;
        size_t pairCount;
        do {
            pairCount = readExchangeRange(repBlocks, memoryLimit / 8, NULL, ClusterPair::compareByRep, lastPair, records, NULL, prefix);
            ClusterPair *pairs = reinterpret_cast<ClusterPair *>(records.data());
            SORT_PARALLEL(pairs, pairs + pairCount, ClusterPair::compareByRep);
            singletons.clear();
            if (singletonBits != NULL) {
                // the singletons up to the largest rep of the pairs go with them in rep order, and the rest of the split after the last pairs
                const size_t singletonEnd = (pairCount > 0) ? std::max<size_t>(nextSingletonKey, pairs[pairCount - 1].repKey + 1) : splitEnd;
                const size_t bitBegin = nextSingletonKey - firstKey;
                const size_t bitEnd = singletonEnd - firstKey;
                const size_t wordBegin = bitBegin / 64;
                const size_t wordCount = (bitEnd + 63) / 64 - wordBegin;
                const auto getSingletonWord = [&](size_t word) {
                    const uint64_t bits = singletonBits[word] & ((word == wordBegin) ? ~0ULL << (bitBegin % 64) : ~0ULL);
                    return (bitEnd - word * 64 < 64) ? bits & ((1ULL << (bitEnd - word * 64)) - 1) : bits;
                };
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
                        for (uint64_t bits = getSingletonWord(word); bits != 0; bits &= bits - 1) {
                            const DBKeyType key = firstKey + word * 64 + __builtin_ctzll(bits);
                            singletons[singleton].repKey = key;
                            singletons[singleton].memberKey = key;
                            singleton++;
                        }
                    }
                }
                nextSingletonKey = std::max(nextSingletonKey, singletonEnd);
            }
            const size_t singletonCount = singletons.size();
            const size_t writeCount = pairCount + singletonCount;
            if (writeCount == 0) {
                break;
            }
            // the rep of the last of the first pairsBefore pairs and singletonsBefore singletons in rep order
            const auto getLastRepKey = [&](size_t pairsBefore, size_t singletonsBefore) {
                const bool isPair = singletonsBefore == 0 || (pairsBefore > 0 && ClusterPair::compareByRep(singletons[singletonsBefore - 1], pairs[pairsBefore - 1]));
                return isPair ? pairs[pairsBefore - 1].repKey : singletons[singletonsBefore - 1].repKey;
            };
#pragma omp parallel for schedule(dynamic, 1)
            for (int thread = 0; thread < par.threads; thread++) {
                std::string &data = threadData[thread];
                std::vector<uint64_t> &threadEntries = threadIndex[thread];
                data.clear();
                threadEntries.clear();
                char buffer[32];
                const size_t begin = writeCount * thread / par.threads;
                const size_t end = writeCount * (thread + 1) / par.threads;
                size_t pair = findMergedPairCount(pairs, pairCount, singletons.data(), singletonCount, begin);
                size_t singleton = begin - pair;
                const size_t endPair = findMergedPairCount(pairs, pairCount, singletons.data(), singletonCount, end);
                const size_t endSingleton = end - endPair;
                DBKeyType repKey = (begin > 0) ? getLastRepKey(pair, singleton) : previousRepKey;
                while (pair < endPair || singleton < endSingleton) {
                    const ClusterPair &next = (singleton == endSingleton || (pair < endPair && ClusterPair::compareByRep(pairs[pair], singletons[singleton]))) ? pairs[pair++] : singletons[singleton++];
                    if (next.repKey != repKey) {
                        if (repKey != DB_KEY_INVALID) {
                            data.push_back('\0');
                        }
                        repKey = next.repKey;
                        threadEntries.push_back(static_cast<uint64_t>(repKey));
                        threadEntries.push_back(data.size());
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
            writeClusterThreadParts(resultDataFile, resultDb, resultIndexFile, resultIndex, threadData, threadIndex, resultDataBytes, resultIndexBytes);
            lineCount += writeCount;
            previousRepKey = getLastRepKey(pairCount, singletonCount);
        } while (pairCount > 0);
    }
    if (previousRepKey != DB_KEY_INVALID) {
        FileUtil::pwriteAll(fileno(resultDataFile), resultDb.c_str(), "\0", 1, resultDataBytes);
        resultDataBytes++;
    }
    closeClusterFiles(resultDataFile, resultDb, resultIndexFile, resultIndex, resultDataBytes, resultIndexBytes);
    closeExchangeFiles(repFiles, prefix);
    return lineCount;
}

void MMseqsNode::removeExchanges(const std::string &prefix, const std::string &lastExchange, const std::vector<std::string> &exchanges, std::string phaseValues) {
    addPhaseValue(phaseValues, "Exchange bytes sent", SSTR(sentExchangeBytes));
    addPhaseValue(phaseValues, "Exchange bytes sent to other nodes", SSTR(sentOtherNodeExchangeBytes));
    printPhaseValues(phaseValues);
    barrier(prefix + ".merged");
    FileUtil::remove((lastExchange + "." + SSTR(rank)).c_str());
    for (size_t i = 0; isMaster() && i < exchanges.size(); i++) {
        FileUtil::remove((exchanges[i] + ".index").c_str());
    }
    removeOnceRead(prefix + ".merged");
}
