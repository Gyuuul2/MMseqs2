#include "MMseqsNode.h"
#include "Debug.h"
#include "FileUtil.h"
#include "Parameters.h"
#include "Util.h"
#include "FastSort.h"
#include "DBReader.h"
#include "DBWriter.h"
#include "itoa.h"
#include "Timer.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <map>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#ifdef OPENMP
#include <omp.h>
#endif

unsigned int MMseqsNode::rank = 0;
unsigned int MMseqsNode::count = 1;
const size_t MMseqsNode::EXCHANGE_BUFFER_SIZE;
// the first line of every barrier file, so a file that a killed run left is told apart from one of this run
static std::string runId;
// the exchange bytes that this node sent, all of them and those for other nodes, reported by removeExchanges
static size_t sentExchangeBytes = 0;
static size_t sentOtherNodeExchangeBytes = 0;
// the records and their bytes that this node sent to itself in the last flush of an exchange, until openExchange reads them
static std::map<std::string, std::pair<char *, size_t> > ownExchangeRecords;

void MMseqsNode::init() {
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
    if (isMaster() == false) {
        Parameters &par = Parameters::getInstance();
        par.verbosity = Debug::ERROR;
        Debug::setDebugLevel(Debug::ERROR);
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

std::string MMseqsNode::waitForFile(const std::string &file) {
    for (unsigned int polls = 0; FileUtil::fileExists(file.c_str()) == false; polls++) {
        sleepBeforePoll(polls);
    }
    FILE *handle = FileUtil::openFileOrDie(file.c_str(), "r", true);
    struct stat sb;
    if (fstat(fileno(handle), &sb) != 0) {
        Debug(Debug::ERROR) << "Cannot read " << file << "\n";
        EXIT(EXIT_FAILURE);
    }
    std::string content(sb.st_size, '\0');
    if (fread(&content[0], sizeof(char), content.size(), handle) != content.size()) {
        Debug(Debug::ERROR) << "Cannot read " << file << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (fclose(handle) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << file << "\n";
        EXIT(EXIT_FAILURE);
    }
    return content;
}

// a file of another run is read again until this run replaces it
std::string MMseqsNode::waitForRunFile(const std::string &file) {
    std::string content = waitForFile(file);
    for (unsigned int polls = 0; content.compare(0, runId.size(), runId) != 0; polls++) {
        sleepBeforePoll(polls);
        content = waitForFile(file);
    }
    return content.substr(runId.size());
}

void MMseqsNode::publishRunFile(const std::string &file, const std::string &content) {
    publishFile(file, runId + content);
}

std::string MMseqsNode::gatherRunFiles(const std::string &prefix) {
    std::string content;
    for (unsigned int node = 0; node < MMseqsNode::count; node++) {
        const std::string file = prefix + SSTR(node);
        content += waitForRunFile(file);
        FileUtil::remove(file.c_str());
    }
    return content;
}

void MMseqsNode::publishLog(FILE *file, const std::string &log, size_t key, size_t bytes) {
    if (fflush(file) != 0 || fsync(fileno(file)) != 0) {
        Debug(Debug::ERROR) << "Cannot flush " << log << "\n";
        EXIT(EXIT_FAILURE);
    }
    publishRunFile(log + ".progress", SSTR(key) + "\t" + SSTR(bytes) + "\n");
}

bool MMseqsNode::readLogProgress(const std::string &log, size_t &key, size_t &bytes) {
    const std::string file = log + ".progress";
    if (FileUtil::fileExists(file.c_str()) == false) {
        return false;
    }
    // NFS may fail to read a progress file that was just replaced; the next poll reads it again
    FILE *handle = fopen(file.c_str(), "r");
    if (handle == NULL) {
        return false;
    }
    char buffer[256];
    const size_t size = fread(buffer, sizeof(char), sizeof(buffer) - 1, handle);
    const bool hasError = ferror(handle) != 0;
    fclose(handle);
    if (hasError) {
        return false;
    }
    const std::string content(buffer, size);
    if (content.compare(0, runId.size(), runId) != 0) {
        return false;
    }
    const char *columns[2];
    if (Util::getWordsOfLine((char *) content.c_str() + runId.size(), columns, 2) < 2) {
        Debug(Debug::ERROR) << "Cannot read " << file << "\n";
        EXIT(EXIT_FAILURE);
    }
    key = Util::fast_atoi<size_t>(columns[0]);
    bytes = Util::fast_atoi<size_t>(columns[1]);
    return true;
}

void MMseqsNode::waitForLog(const std::string &log, size_t &key, size_t &bytes) {
    for (unsigned int polls = 0; readLogProgress(log, key, bytes) == false; polls++) {
        sleepBeforePoll(polls);
    }
}

void MMseqsNode::readLog(const std::string &log, size_t from, size_t to, char *data) {
    FILE *file = FileUtil::openFileOrDie(log.c_str(), "r", true);
    FileUtil::preadAll(fileno(file), log.c_str(), data, to - from, from);
    if (fclose(file) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << log << "\n";
        EXIT(EXIT_FAILURE);
    }
}

void MMseqsNode::publishExchangeFlush(FILE *file, const std::string &fileName, size_t bytes, const std::string &indexFile, const std::vector<uint64_t> &line) {
    if (fflush(file) != 0 || fsync(fileno(file)) != 0 || fclose(file) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << fileName << "\n";
        EXIT(EXIT_FAILURE);
    }
    sentExchangeBytes += bytes;
    sentOtherNodeExchangeBytes += bytes;
    FILE *handle = FileUtil::openFileOrDie(indexFile.c_str(), "a", true);
    if (fwrite(line.data(), sizeof(uint64_t), line.size(), handle) != line.size() || fflush(handle) != 0 || fsync(fileno(handle)) != 0 || fclose(handle) != 0) {
        Debug(Debug::ERROR) << "Cannot write " << indexFile << "\n";
        EXIT(EXIT_FAILURE);
    }
}

bool MMseqsNode::readExchangeFlush(const std::string &indexFile, size_t flush, std::vector<uint64_t> &line) {
    const size_t lineBytes = line.size() * sizeof(uint64_t);
    const int fd = open(indexFile.c_str(), O_RDONLY);
    if (fd < 0) {
        Debug(Debug::ERROR) << "Cannot open " << indexFile << "\n";
        EXIT(EXIT_FAILURE);
    }
    const ssize_t readBytes = pread(fd, line.data(), lineBytes, flush * lineBytes);
    if (close(fd) != 0 || readBytes < 0) {
        Debug(Debug::ERROR) << "Cannot read " << indexFile << "\n";
        EXIT(EXIT_FAILURE);
    }
    return static_cast<size_t>(readBytes) == lineBytes && line[0] == flush;
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
    if (isMaster() == false) {
        return;
    }
    std::vector<std::string> names;
    std::vector<std::string> maxValues;
    std::vector<std::string> maxNodes;
    std::vector<std::string> minValues;
    std::vector<size_t> lineCounts;
    for (char *line = (char *) values.c_str(); *line != '\0'; line = Util::skipLine(line)) {
        const char *columns[3];
        if (Util::getFieldsOfLine(line, columns, 3) < 3) {
            Debug(Debug::ERROR) << "Cannot read the phase values\n";
            EXIT(EXIT_FAILURE);
        }
        const std::string name(columns[0], columns[1] - 1);
        const std::string node(columns[1], columns[2] - 1);
        const std::string value(columns[2], strchr(columns[2], '\n'));
        const size_t i = std::find(names.begin(), names.end(), name) - names.begin();
        if (i == names.size()) {
            names.push_back(name);
            maxValues.push_back(value);
            maxNodes.push_back(node);
            minValues.push_back(value);
            lineCounts.push_back(0);
        } else if (strtod(value.c_str(), NULL) > strtod(maxValues[i].c_str(), NULL)) {
            maxValues[i] = value;
            maxNodes[i] = node;
        } else if (strtod(value.c_str(), NULL) < strtod(minValues[i].c_str(), NULL)) {
            minValues[i] = value;
        }
        lineCounts[i]++;
    }
    for (size_t i = 0; i < names.size(); i++) {
        if (lineCounts[i] == 1) {
            Debug(Debug::INFO) << names[i] << ": " << maxValues[i] << ((count > 1) ? " (node " + maxNodes[i] + ")" : "") << "\n";
        } else {
            Debug(Debug::INFO) << names[i] << ": max " << maxValues[i] << " (node " << maxNodes[i] << ") min " << minValues[i] << "\n";
        }
    }
}

MMseqsNode::WorkQueue::WorkQueue(const std::string &name, size_t itemCount) : name(name), itemCount(itemCount), nextItem(rank), itemsLeft(count, 0), nextNode((rank + 1) % count) {
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
    if (fd < 0 || close(fd) != 0) {
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
    indexData += SSTR(rank);
    for (size_t node = 0; node < nodeOffsets.size(); node++) {
        indexData += "\t" + SSTR(writtenBytes + nodeOffsets[node]);
    }
    indexData += "\n";
    writtenBytes += nodeOffsets.back();
    sentExchangeBytes += nodeOffsets.back();
    sentOtherNodeExchangeBytes += nodeOffsets.back() - (nodeOffsets[rank + 1] - nodeOffsets[rank]);
}

std::string MMseqsNode::closeExchange(FILE *file, const std::string &fileName, const std::string &indexFile, const std::string &indexData) {
    if (fclose(file) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << fileName << "\n";
        EXIT(EXIT_FAILURE);
    }
    return barrier(indexFile, indexData);
}

void MMseqsNode::openExchange(const std::string &index, const std::string &prefix, std::vector<ExchangeBlock> &blocks, std::vector<FILE *> &files,
                              bool sendersInOrder) {
    files.assign(count, NULL);
    std::vector<const char *> columns(count + 2);
    for (char *line = (char *) index.c_str(); *line != '\0'; line = Util::skipLine(line)) {
        Util::getWordsOfLine(line, columns.data(), columns.size());
        const unsigned int node = Util::fast_atoi<unsigned int>(columns[0]);
        const size_t pos = Util::fast_atoi<size_t>(columns[1 + rank]);
        const size_t end = Util::fast_atoi<size_t>(columns[2 + rank]);
        if (pos == end) {
            continue;
        }
        if (files[node] == NULL) {
            files[node] = FileUtil::openFileOrDie((prefix + "." + SSTR(node)).c_str(), "r", true);
        }
        // the index lists the flushes of every node one after the other
        if (sendersInOrder && blocks.empty() == false && blocks.back().file == files[node]) {
            blocks.back().parts.push_back(std::make_pair(pos, end));
            blocks.back().partsBytes += end - pos;
            continue;
        }
        const ExchangeBlock block = { files[node], pos, end, std::vector<char>(), 0, 0, 0, 0, std::vector<std::pair<size_t, size_t> >(), 0, 0, NULL };
        blocks.push_back(block);
    }
    std::map<std::string, std::pair<char *, size_t> >::iterator own = ownExchangeRecords.find(prefix);
    if (own != ownExchangeRecords.end()) {
        const ExchangeBlock block = { NULL, 0, own->second.second, std::vector<char>(), 0, 0, 0, 0, std::vector<std::pair<size_t, size_t> >(), 0, 0, own->second.first };
        blocks.push_back(block);
        ownExchangeRecords.erase(own);
    }
}

static size_t getUnreadBytes(const MMseqsNode::ExchangeBlock &block) {
    return block.end - block.pos + block.partsBytes;
}

static const char *getBlockData(const MMseqsNode::ExchangeBlock &block) {
    return (block.ownRecords != NULL) ? block.ownRecords : block.buffer.data();
}

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
    memmove(block.buffer.data(), block.buffer.data() + block.bufferPos, block.bufferEnd - block.bufferPos);
    block.bufferEnd -= block.bufferPos;
    block.recordEnd -= std::min(block.recordEnd, block.bufferPos);
    block.lastRecord -= std::min(block.lastRecord, block.bufferPos);
    block.bufferPos = 0;
    const size_t blockBufferSize = std::min(std::max(bytes, bufferSize), block.bufferEnd + getUnreadBytes(block));
    if (block.buffer.size() < blockBufferSize) {
        block.buffer.resize(blockBufferSize);
    }
    // the parts of a block follow each other in its buffer, and a flush holds whole records; outside a parallel region the threads read pieces of it
    std::vector<std::pair<size_t, size_t> > reads;
    while (block.bufferEnd < block.buffer.size() && getUnreadBytes(block) > 0) {
        if (block.pos == block.end) {
            block.pos = block.parts[block.nextPart].first;
            block.end = block.parts[block.nextPart].second;
            block.partsBytes -= block.end - block.pos;
            block.nextPart++;
        }
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

// frees the records that this node sent to itself once a block has passed them all
static void releaseOwnRecords(MMseqsNode::ExchangeBlock &block) {
    if (block.ownRecords != NULL && block.bufferPos == block.end) {
        free(block.ownRecords);
        block.ownRecords = NULL;
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
    const bool isClusterPair = recordSize == ClusterPair::recordSize;
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
            for (size_t size; (size = recordSize(data + block.recordEnd, block.bufferEnd - block.recordEnd)) > 0; block.recordEnd += size) {
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
                pos += recordSize(records.data() + pos, blockByteOffsets[i + 1] - pos);
            }
        }
        block.bufferPos = rangeEnds[i];
        releaseOwnRecords(block);
    }
    return blockRecordOffsets.back();
}

size_t MMseqsNode::readExchangeSegments(std::vector<ExchangeBlock> &blocks, size_t bufferMemory, size_t maxSegmentSize, uint64_t fileType, size_t windowBegin, size_t windowEnd,
                                        char *window, std::vector<std::pair<size_t, size_t> > &segments, const std::string &prefix) {
    const size_t bufferSize = std::max(3 * sizeof(uint64_t) + maxSegmentSize, bufferMemory / std::max<size_t>(1, blocks.size()));
    size_t nextOffset = SIZE_MAX;
#pragma omp parallel reduction(min:nextOffset)
    {
        std::vector<std::pair<size_t, size_t> > threadSegments;
#pragma omp for schedule(dynamic, 1) nowait
        for (size_t i = 0; i < blocks.size(); i++) {
            ExchangeBlock &block = blocks[i];
            uint64_t segment[3];
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
                bufferExchangeBlock(block, sizeof(segment) + segment[2], bufferSize, prefix);
                if (block.bufferEnd - block.bufferPos < sizeof(segment) + segment[2]) {
                    Debug(Debug::ERROR) << "Cannot read " << prefix << ": a record ends after its block\n";
                    EXIT(EXIT_FAILURE);
                }
                memcpy(window + (segment[1] - windowBegin), getBlockData(block) + block.bufferPos + sizeof(segment), segment[2]);
                threadSegments.push_back(std::make_pair(segment[1], segment[2]));
                block.bufferPos += sizeof(segment) + segment[2];
            }
            releaseOwnRecords(block);
        }
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

void MMseqsNode::readExchangePairs(const std::string &index, const std::string &prefix, std::vector<ClusterPair> &pairs) {
    std::vector<ExchangeBlock> blocks;
    std::vector<FILE *> files;
    openExchange(index, prefix, blocks, files);
    std::vector<size_t> blockOffsets(blocks.size() + 1, 0);
    for (size_t i = 0; i < blocks.size(); i++) {
        blockOffsets[i + 1] = blockOffsets[i] + blocks[i].end - blocks[i].pos;
    }
    pairs.resize(blockOffsets.back() / sizeof(ClusterPair));
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < blocks.size(); i++) {
        FileUtil::preadAll(fileno(blocks[i].file), prefix.c_str(), reinterpret_cast<char *>(pairs.data()) + blockOffsets[i], blockOffsets[i + 1] - blockOffsets[i], blocks[i].pos);
    }
    closeExchangeFiles(files, prefix);
}

// sorts a flush of pairs, appends it to file and adds its index line; ownPairCount: keeps the pairs for this node at the start of pairs instead
static void writePairs(ClusterPair *pairs, size_t pairCount, bool (*compare)(const ClusterPair &, const ClusterPair &), const std::vector<size_t> &nodeKeyOffsets,
                       FILE *file, const std::string &fileName, std::string &indexData, size_t &writtenBytes, size_t *ownPairCount = NULL) {
    SORT_PARALLEL(pairs, pairs + pairCount, compare);
    std::vector<size_t> nodeOffsets(MMseqsNode::count + 1, pairCount * sizeof(ClusterPair));
    for (size_t node = 0; node < MMseqsNode::count; node++) {
        const ClusterPair nodeFirstPair = { static_cast<DBKeyType>(nodeKeyOffsets[node]), static_cast<DBKeyType>(nodeKeyOffsets[node]) };
        nodeOffsets[node] = (std::lower_bound(pairs, pairs + pairCount, nodeFirstPair, compare) - pairs) * sizeof(ClusterPair);
    }
    const size_t ownBegin = nodeOffsets[MMseqsNode::rank] / sizeof(ClusterPair);
    const size_t ownEnd = (ownPairCount != NULL) ? nodeOffsets[MMseqsNode::rank + 1] / sizeof(ClusterPair) : ownBegin;
    if (fwrite(pairs, sizeof(ClusterPair), ownBegin, file) != ownBegin || fwrite(pairs + ownEnd, sizeof(ClusterPair), pairCount - ownEnd, file) != pairCount - ownEnd) {
        Debug(Debug::ERROR) << "Cannot write " << fileName << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (ownPairCount != NULL && ownBegin > 0) {
        memmove(pairs, pairs + ownBegin, (ownEnd - ownBegin) * sizeof(ClusterPair));
    }
    if (ownPairCount != NULL) {
        *ownPairCount = ownEnd - ownBegin;
    }
    for (size_t node = MMseqsNode::rank + 1; node <= MMseqsNode::count; node++) {
        nodeOffsets[node] -= (ownEnd - ownBegin) * sizeof(ClusterPair);
    }
    MMseqsNode::addExchangeFlush(indexData, writtenBytes, nodeOffsets);
    sentExchangeBytes += (ownEnd - ownBegin) * sizeof(ClusterPair);
}

MMseqsNode::ClusterPairWriter::ClusterPairWriter(const std::string &exchange, size_t pairCapacity, bool (*compare)(const ClusterPair &, const ClusterPair &), const std::vector<size_t> &nodeKeyOffsets)
    : exchange(exchange), fileName(exchange + "." + SSTR(rank)), file(FileUtil::openAndDelete(fileName.c_str(), "w")), pairs(static_cast<ClusterPair *>(malloc(pairCapacity * sizeof(ClusterPair)))),
      pairCapacity(pairCapacity), pairCount(0), compare(compare), nodeKeyOffsets(nodeKeyOffsets), writtenBytes(0) {
    Util::checkAllocation(pairs, "Can not allocate pairs memory in ClusterPairWriter");
}

MMseqsNode::ClusterPairWriter::~ClusterPairWriter() {
    free(pairs);
}

void MMseqsNode::ClusterPairWriter::flush() {
    writePairs(pairs, pairCount, compare, nodeKeyOffsets, file, fileName, indexData, writtenBytes);
    pairCount = 0;
}

std::string MMseqsNode::ClusterPairWriter::close() {
    size_t ownPairCount = 0;
    if (pairCount > 0) {
        writePairs(pairs, pairCount, compare, nodeKeyOffsets, file, fileName, indexData, writtenBytes, &ownPairCount);
        pairCount = 0;
    }
    // the buffer, cut to the pairs for this node, becomes their records
    if (ownPairCount > 0) {
        pairs = static_cast<ClusterPair *>(realloc(pairs, ownPairCount * sizeof(ClusterPair)));
        Util::checkAllocation(pairs, "Can not shrink pairs memory in ClusterPairWriter");
        ownExchangeRecords[exchange] = std::make_pair(reinterpret_cast<char *>(pairs), ownPairCount * sizeof(ClusterPair));
    } else {
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

// loads <clu>.<rank> split by split, prefetching the next, and calls processSplit(cluDb, splitBegin, splitEnd, entryKeys); a larger entry is a split of its own
template<typename ProcessSplit>
static void readClusterSplits(const std::string &clu, size_t memoryLimit, ProcessSplit processSplit) {
    DBReader<DBKeyType> cluDb((clu + "." + SSTR(MMseqsNode::rank)).c_str(), (clu + ".index." + SSTR(MMseqsNode::rank)).c_str(), Parameters::getInstance().threads,
                              DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
    cluDb.open(DBReader<DBKeyType>::NOSORT);
    const size_t splitBytes = memoryLimit / CLUSTER_SPLIT_MEMORY_DIVISOR;
    const size_t splitEntries = std::max<size_t>(1, memoryLimit / CLUSTER_SPLIT_MEMORY_DIVISOR / (2 * sizeof(uint64_t) + 2 * sizeof(size_t)));
    const auto findSplitEnd = [&](size_t splitBegin, size_t splitOffset) {
        const size_t bytesEnd = (splitOffset + splitBytes >= cluDb.getDataSize()) ? cluDb.getSize() : std::max(splitBegin + 1, cluDb.findEntryByOffset(splitOffset + splitBytes + 1) - 1);
        return std::min(bytesEnd, splitBegin + splitEntries);
    };
    std::vector<DBKeyType> entryKeys;
    size_t splitOffset = 0;
    for (size_t splitBegin = 0, splitEnd = findSplitEnd(0, 0), nextSplitEnd; splitBegin < cluDb.getSize(); splitBegin = splitEnd, splitEnd = nextSplitEnd) {
        cluDb.loadData(splitBegin, splitEnd);
        entryKeys.resize(splitEnd - splitBegin);
#pragma omp parallel for schedule(static) reduction(+:splitOffset)
        for (size_t i = splitBegin; i < splitEnd; i++) {
            entryKeys[i - splitBegin] = cluDb.getDbKey(i);
            splitOffset += cluDb.getEntryLen(i);
        }
        nextSplitEnd = findSplitEnd(splitEnd, splitOffset);
        if (splitEnd < cluDb.getSize()) {
            cluDb.prefetchData(splitEnd, nextSplitEnd);
        }
        processSplit(cluDb, splitBegin, splitEnd, entryKeys);
    }
    cluDb.close();
}

std::string MMseqsNode::sendClusterPairs(const std::string &clu, bool withRepresentative, bool (*compare)(const ClusterPair &, const ClusterPair &), const std::vector<size_t> &nodeKeyOffsets,
                                         const std::string &exchange, const std::string &receivedIndex, const std::string &receivedPrefix) {
    std::vector<ExchangeBlock> receivedBlocks;
    std::vector<FILE *> receivedFiles;
    if (receivedPrefix.empty() == false) {
        openExchange(receivedIndex, receivedPrefix, receivedBlocks, receivedFiles);
    }
    const size_t memoryLimit = Util::computeMemory(Parameters::getInstance().splitMemoryLimit);
    ClusterPairWriter pairWriter(exchange, memoryLimit / SENT_PAIRS_MEMORY_DIVISOR / sizeof(ClusterPair), compare, nodeKeyOffsets);

    std::vector<DBKeyType> repKeys;
    std::vector<size_t> pairOffsets;
    std::vector<char> received;
    DBKeyType previousReceivedKey = DB_KEY_INVALID;
    readClusterSplits(clu, memoryLimit, [&](DBReader<DBKeyType> &cluDb, size_t splitBegin, size_t splitEnd, const std::vector<DBKeyType> &entryKeys) {
        repKeys = entryKeys;
        const ClusterPair splitLast = { entryKeys.back(), entryKeys.back() };
        size_t receivedCount;
        while ((receivedCount = readExchangeRange(receivedBlocks, memoryLimit / RECEIVED_MEMORY_DIVISOR, ClusterPair::recordSize, ClusterPair::compareByMember, splitLast, received, NULL, receivedPrefix)) > 0) {
            ClusterPair *receivedPairs = reinterpret_cast<ClusterPair *>(received.data());
            SORT_PARALLEL(receivedPairs, receivedPairs + receivedCount, ClusterPair::compareByMember);
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < receivedCount; i++) {
                const DBKeyType memberKey = receivedPairs[i].memberKey;
                const size_t entry = std::lower_bound(entryKeys.begin(), entryKeys.end(), memberKey) - entryKeys.begin();
                // a received member is an entry of this node once, and the entries are in key order
                if (entry == entryKeys.size() || entryKeys[entry] != memberKey || memberKey == ((i > 0) ? receivedPairs[i - 1].memberKey : previousReceivedKey)) {
                    Debug(Debug::ERROR) << "Key " << memberKey << " of " << receivedPrefix << " is no entry of " << clu << "." << rank << " in key order\n";
                    EXIT(EXIT_FAILURE);
                }
                repKeys[entry] = receivedPairs[i].repKey;
            }
            previousReceivedKey = receivedPairs[receivedCount - 1].memberKey;
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
            if (endEntry == firstEntry && pairWriter.pairCount > 0) {
                pairWriter.flush();
                continue;
            }
            // a cluster with more pairs than fit into memory is sent in several flushes
            if (endEntry == firstEntry) {
                const DBKeyType entryKey = cluDb.getDbKey(firstEntry);
                char keyBuffer[255];
                for (char *data = cluDb.getData(firstEntry, 0); *data != '\0'; data = Util::skipLine(data)) {
                    Util::parseKey(data, keyBuffer);
                    const DBKeyType key = Util::fast_atoi<DBKeyType>(keyBuffer);
                    if (withRepresentative == false && key == entryKey) {
                        continue;
                    }
                    pairWriter.add(repKeys[firstEntry - splitBegin], key);
                }
                firstEntry++;
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
    if (readExchangeRange(receivedBlocks, memoryLimit / RECEIVED_MEMORY_DIVISOR, ClusterPair::recordSize, ClusterPair::compareByMember, lastPair, received, NULL, receivedPrefix) > 0) {
        Debug(Debug::ERROR) << "Key " << reinterpret_cast<ClusterPair *>(received.data())->memberKey << " of " << receivedPrefix << " is no entry of " << clu << "." << rank << "\n";
        EXIT(EXIT_FAILURE);
    }
    closeExchangeFiles(receivedFiles, receivedPrefix);
    return pairWriter.close();
}

std::string MMseqsNode::sendClusterMembers(const std::string &clu, const std::string &requestsIndex, const std::string &requests, const std::string &members) {
    std::vector<ExchangeBlock> requestBlocks;
    std::vector<FILE *> requestFiles;
    openExchange(requestsIndex, requests, requestBlocks, requestFiles);
    const size_t memoryLimit = Util::computeMemory(Parameters::getInstance().splitMemoryLimit);
    const std::string membersFile = members + "." + SSTR(rank);
    FILE *membersHandle = FileUtil::openAndDelete(membersFile.c_str(), "w");
    std::string membersIndexData;
    size_t membersBytes = 0;
    const size_t memberCapacity = std::max<size_t>(1, memoryLimit / SENT_PAIRS_MEMORY_DIVISOR / sizeof(ClusterPair));
    std::vector<ClusterPair> nodeMembers;
    std::vector<size_t> nodeOffsets(count + 1);
    std::vector<size_t> nodePositions(count);
    // writes the first memberCount of nodeMembers, whose nodes start at nodeOffsets
    const auto writeMembers = [&](size_t memberCount) {
        if (fwrite(nodeMembers.data(), sizeof(ClusterPair), memberCount, membersHandle) != memberCount) {
            Debug(Debug::ERROR) << "Cannot write " << membersFile << "\n";
            EXIT(EXIT_FAILURE);
        }
        for (size_t node = 0; node <= count; node++) {
            nodeOffsets[node] *= sizeof(ClusterPair);
        }
        addExchangeFlush(membersIndexData, membersBytes, nodeOffsets);
    };
    std::vector<char> received;
    std::vector<size_t> requestEntries;
    std::vector<size_t> memberCounts;
    std::vector<size_t> memberOffsets;
    ClusterPair previousRequest = { DB_KEY_INVALID, DB_KEY_INVALID };
    readClusterSplits(clu, memoryLimit, [&](DBReader<DBKeyType> &cluDb, size_t splitBegin, size_t, const std::vector<DBKeyType> &entryKeys) {
        const ClusterPair splitLast = { entryKeys.back(), DB_KEY_INVALID };
        size_t requestCount;
        while ((requestCount = readExchangeRange(requestBlocks, memoryLimit / RECEIVED_MEMORY_DIVISOR, ClusterPair::recordSize, ClusterPair::compareByRep, splitLast, received, NULL, requests)) > 0) {
            // a node gets the members of a key once
            ClusterPair *requestPairs = reinterpret_cast<ClusterPair *>(received.data());
            SORT_PARALLEL(requestPairs, requestPairs + requestCount, ClusterPair::compareByRep);
            requestEntries.resize(requestCount);
            memberCounts.assign(requestCount, 0);
#pragma omp parallel
            {
                unsigned int thread_idx = 0;
#ifdef OPENMP
                thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
                char keyBuffer[255];
#pragma omp for schedule(dynamic, 100)
                for (size_t i = 0; i < requestCount; i++) {
                    const DBKeyType key = requestPairs[i].repKey;
                    const ClusterPair &previous = (i > 0) ? requestPairs[i - 1] : previousRequest;
                    if (key == previous.repKey && requestPairs[i].memberKey == previous.memberKey) {
                        continue;
                    }
                    const size_t entry = std::lower_bound(entryKeys.begin(), entryKeys.end(), key) - entryKeys.begin();
                    if (entry == entryKeys.size() || entryKeys[entry] != key) {
                        Debug(Debug::ERROR) << "Key " << key << " of " << requests << " is no entry of " << clu << "." << rank << "\n";
                        EXIT(EXIT_FAILURE);
                    }
                    requestEntries[i] = splitBegin + entry;
                    for (char *data = cluDb.getData(requestEntries[i], thread_idx); *data != '\0'; data = Util::skipLine(data)) {
                        Util::parseKey(data, keyBuffer);
                        memberCounts[i] += (Util::fast_atoi<DBKeyType>(keyBuffer) != key);
                    }
                }
            }
            previousRequest = requestPairs[requestCount - 1];
            for (size_t firstRequest = 0; firstRequest < requestCount;) {
                // a request with more members than fit into memory is sent in several flushes
                if (memberCounts[firstRequest] > memberCapacity) {
                    const DBKeyType key = requestPairs[firstRequest].repKey;
                    const size_t node = requestPairs[firstRequest].memberKey;
                    nodeMembers.resize(memberCapacity);
                    size_t memberCount = 0;
                    char keyBuffer[255];
                    for (char *data = cluDb.getData(requestEntries[firstRequest], 0); *data != '\0'; data = Util::skipLine(data)) {
                        Util::parseKey(data, keyBuffer);
                        const ClusterPair pair = { key, Util::fast_atoi<DBKeyType>(keyBuffer) };
                        if (pair.memberKey != key) {
                            nodeMembers[memberCount++] = pair;
                        }
                        if (memberCount == memberCapacity || (memberCount > 0 && *Util::skipLine(data) == '\0')) {
                            for (size_t i = 0; i <= count; i++) {
                                nodeOffsets[i] = (i <= node) ? 0 : memberCount;
                            }
                            writeMembers(memberCount);
                            memberCount = 0;
                        }
                    }
                    firstRequest++;
                    continue;
                }
                // the requests whose members fit go in requesting node order, each node's in key order
                size_t endRequest = firstRequest;
                for (size_t memberCount = 0; endRequest < requestCount && memberCount + memberCounts[endRequest] <= memberCapacity; endRequest++) {
                    memberCount += memberCounts[endRequest];
                }
                std::fill(nodeOffsets.begin(), nodeOffsets.end(), 0);
                for (size_t i = firstRequest; i < endRequest; i++) {
                    nodeOffsets[requestPairs[i].memberKey + 1] += memberCounts[i];
                }
                for (size_t node = 0; node < count; node++) {
                    nodeOffsets[node + 1] += nodeOffsets[node];
                    nodePositions[node] = nodeOffsets[node];
                }
                memberOffsets.resize(requestCount);
                for (size_t i = firstRequest; i < endRequest; i++) {
                    memberOffsets[i] = nodePositions[requestPairs[i].memberKey];
                    nodePositions[requestPairs[i].memberKey] += memberCounts[i];
                }
                if (nodeOffsets.back() > 0) {
                    nodeMembers.resize(nodeOffsets.back());
#pragma omp parallel
                    {
                        unsigned int thread_idx = 0;
#ifdef OPENMP
                        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
                        char keyBuffer[255];
#pragma omp for schedule(dynamic, 100)
                        for (size_t i = firstRequest; i < endRequest; i++) {
                            if (memberCounts[i] == 0) {
                                continue;
                            }
                            const DBKeyType key = requestPairs[i].repKey;
                            size_t member = memberOffsets[i];
                            for (char *data = cluDb.getData(requestEntries[i], thread_idx); *data != '\0'; data = Util::skipLine(data)) {
                                Util::parseKey(data, keyBuffer);
                                const ClusterPair pair = { key, Util::fast_atoi<DBKeyType>(keyBuffer) };
                                if (pair.memberKey != key) {
                                    nodeMembers[member++] = pair;
                                }
                            }
                        }
                    }
                    writeMembers(nodeOffsets.back());
                }
                firstRequest = endRequest;
            }
        }
    });
    const ClusterPair lastPair = { DB_KEY_INVALID, DB_KEY_INVALID };
    if (readExchangeRange(requestBlocks, memoryLimit / RECEIVED_MEMORY_DIVISOR, ClusterPair::recordSize, ClusterPair::compareByRep, lastPair, received, NULL, requests) > 0) {
        Debug(Debug::ERROR) << "Key " << reinterpret_cast<ClusterPair *>(received.data())->repKey << " of " << requests << " is no entry of " << clu << "." << rank << "\n";
        EXIT(EXIT_FAILURE);
    }
    closeExchangeFiles(requestFiles, requests);
    return closeExchange(membersHandle, membersFile, members + ".index", membersIndexData);
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
    // the bits are on disk before the index part that tells the master this node is done
    const std::string bitsFile = db + ".bits." + SSTR(rank);
    FILE *file = FileUtil::openAndDelete(bitsFile.c_str(), "w");
    const uint64_t header[2] = { firstKey, keyCount };
    const size_t words = (keyCount + 63) / 64;
    if (fwrite(header, sizeof(uint64_t), 2, file) != 2 || fwrite(bits, sizeof(uint64_t), words, file) != words
        || fflush(file) != 0 || fsync(fileno(file)) != 0 || fclose(file) != 0) {
        Debug(Debug::ERROR) << "Cannot write " << bitsFile << "\n";
        EXIT(EXIT_FAILURE);
    }
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

std::vector<size_t> MMseqsNode::getClusterKeyOffsets(const std::string &file, const std::string &clu, size_t keyCount) {
    const std::string cluIndex = clu + ".index." + SSTR(rank);
    FILE *handle = FileUtil::openFileOrDie(cluIndex.c_str(), "r", true);
    uint64_t firstCell[2];
    if (fread(firstCell, sizeof(uint64_t), 2, handle) != 2 || fclose(handle) != 0) {
        Debug(Debug::ERROR) << "Cannot read " << cluIndex << "\n";
        EXIT(EXIT_FAILURE);
    }
    const std::string firstKeys = barrier(file, SSTR(firstCell[0]) + "\n");
    // the master removes file once every node has read it
    publishRunFile(file + ".read." + SSTR(rank), "");
    if (isMaster()) {
        gatherRunFiles(file + ".read.");
        FileUtil::remove(file.c_str());
    }
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

void MMseqsNode::readClusterKeys(const std::string &clu, const std::vector<size_t> &cluKeyOffsets, size_t keyBegin, size_t keyEnd, std::vector<DBKeyType> &keys) {
    keys.clear();
    // (key, offset) cells in key order
    std::vector<uint64_t> cells(1024 * 1024 / sizeof(uint64_t));
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
        if (fseek(cluIndexFile, entryBegin * 2 * sizeof(uint64_t), SEEK_SET) != 0) {
            Debug(Debug::ERROR) << "Cannot read " << cluIndex << "\n";
            EXIT(EXIT_FAILURE);
        }
        for (size_t entry = entryBegin; entry < entryEnd;) {
            const size_t cellCount = fread(cells.data(), 2 * sizeof(uint64_t), std::min(cells.size() / 2, entryEnd - entry), cluIndexFile);
            if (cellCount == 0) {
                Debug(Debug::ERROR) << "Cannot read " << cluIndex << "\n";
                EXIT(EXIT_FAILURE);
            }
            for (size_t i = 0; i < cellCount; i++) {
                keys.push_back(cells[2 * i]);
            }
            entry += cellCount;
        }
        if (fclose(cluIndexFile) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << cluIndex << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
}

void MMseqsNode::writeClusterIndexEntry(FILE *file, const std::string &fileName, DBKeyType key, size_t offset) {
    const uint64_t cell[2] = { static_cast<uint64_t>(key), static_cast<uint64_t>(offset) };
    if (fwrite(cell, sizeof(uint64_t), 2, file) != 2) {
        Debug(Debug::ERROR) << "Cannot write " << fileName << "\n";
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

void MMseqsNode::writeClusterPairs(const std::string &index, const std::string &prefix, const std::string &clu, const uint64_t *singletonBits, size_t firstKey, size_t keyCount) {
    const Parameters &par = Parameters::getInstance();
    std::vector<ExchangeBlock> repBlocks;
    std::vector<FILE *> repFiles;
    openExchange(index, prefix, repBlocks, repFiles);
    const size_t memoryLimit = Util::computeMemory(par.splitMemoryLimit);
    const std::string resultDb = clu + "." + SSTR(rank);
    const std::string resultIndex = clu + ".index." + SSTR(rank);
    FILE *resultIndexFile = FileUtil::openAndDelete(resultIndex.c_str(), "w");
    DBWriter dbw(resultDb.c_str(), (resultDb + ".index").c_str(), 1, 0, Parameters::DBTYPE_OMIT_FILE);
    dbw.open();
    std::vector<char> records;
    std::vector<ClusterPair> singletons;
    std::vector<size_t> threadSingletonOffsets(par.threads + 1, 0);
    std::vector<std::string> threadData(par.threads);
    // (key, offset) of the entries that start in the part of a thread
    std::vector<std::vector<uint64_t> > threadIndex(par.threads);
    std::vector<size_t> threadDataOffsets(par.threads + 1);
    DBKeyType previousRepKey = DB_KEY_INVALID;
    // the keys go in splits, so the pairs of their singletons fit into memory
    const size_t splitKeys = std::max<size_t>(1, memoryLimit / 16 / sizeof(ClusterPair));
    const size_t keyEnd = firstKey + keyCount;
    for (size_t splitBegin = firstKey; splitBegin < keyEnd; splitBegin += splitKeys) {
        const size_t splitEnd = std::min(keyEnd, splitBegin + splitKeys);
        const ClusterPair lastPair = { static_cast<DBKeyType>(splitEnd - 1), DB_KEY_INVALID };
        size_t nextSingletonKey = splitBegin;
        size_t pairCount;
        do {
            pairCount = readExchangeRange(repBlocks, memoryLimit / 8, ClusterPair::recordSize, ClusterPair::compareByRep, lastPair, records, NULL, prefix);
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
                    size_t threadSingletons = 0;
                    for (size_t word = wordBegin + wordCount * thread / par.threads; word < wordBegin + wordCount * (thread + 1) / par.threads; word++) {
                        threadSingletons += __builtin_popcountll(getSingletonWord(word));
                    }
                    threadSingletonOffsets[thread + 1] = threadSingletons;
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
            threadDataOffsets[0] = dbw.getOffset(0);
            for (int thread = 0; thread < par.threads; thread++) {
                threadDataOffsets[thread + 1] = threadDataOffsets[thread] + threadData[thread].size();
            }
#pragma omp parallel for schedule(dynamic, 1)
            for (int thread = 0; thread < par.threads; thread++) {
                for (size_t i = 1; i < threadIndex[thread].size(); i += 2) {
                    threadIndex[thread][i] += threadDataOffsets[thread];
                }
            }
            for (int thread = 0; thread < par.threads; thread++) {
                dbw.writeAdd(threadData[thread].data(), threadData[thread].size(), 0);
                if (fwrite(threadIndex[thread].data(), sizeof(uint64_t), threadIndex[thread].size(), resultIndexFile) != threadIndex[thread].size()) {
                    Debug(Debug::ERROR) << "Cannot write " << resultIndex << "\n";
                    EXIT(EXIT_FAILURE);
                }
            }
            previousRepKey = getLastRepKey(pairCount, singletonCount);
        } while (pairCount > 0);
    }
    const ClusterPair lastPair = { DB_KEY_INVALID, DB_KEY_INVALID };
    if (readExchangeRange(repBlocks, memoryLimit / 8, ClusterPair::recordSize, ClusterPair::compareByRep, lastPair, records, NULL, prefix) > 0) {
        Debug(Debug::ERROR) << "Key " << reinterpret_cast<ClusterPair *>(records.data())->repKey << " of " << prefix << " is no key of node " << rank << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (previousRepKey != DB_KEY_INVALID) {
        dbw.writeAdd("\0", 1, 0);
    }
    closeClusterWriter(dbw, resultIndexFile, resultIndex);
    closeExchangeFiles(repFiles, prefix);
}

void MMseqsNode::closeClusterWriter(DBWriter &writer, FILE *indexFile, const std::string &indexFileName) {
    writeClusterIndexEntry(indexFile, indexFileName, DB_KEY_INVALID, writer.getOffset(0));
    if (fclose(indexFile) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << indexFileName << "\n";
        EXIT(EXIT_FAILURE);
    }
    writer.close(false, false);
    FileUtil::remove(writer.getIndexFileName());
}

void MMseqsNode::removeExchanges(const std::string &prefix, const std::string &lastExchange, const std::vector<std::string> &exchanges, std::string phaseValues) {
    if (ownExchangeRecords.empty() == false) {
        Debug(Debug::ERROR) << "The pairs that node " << rank << " sent to itself through " << ownExchangeRecords.begin()->first << " were not read\n";
        EXIT(EXIT_FAILURE);
    }
    addPhaseValue(phaseValues, "Exchange bytes sent", SSTR(sentExchangeBytes));
    addPhaseValue(phaseValues, "Exchange bytes sent to other nodes", SSTR(sentOtherNodeExchangeBytes));
    printPhaseValues(barrier(prefix + ".merged", phaseValues));
    FileUtil::remove((lastExchange + "." + SSTR(rank)).c_str());
    publishFile(prefix + ".done." + SSTR(rank), runId);
    if (isMaster()) {
        gatherRunFiles(prefix + ".done.");
        for (size_t i = 0; i < exchanges.size(); i++) {
            FileUtil::remove((exchanges[i] + ".index").c_str());
        }
        FileUtil::remove((prefix + ".merged").c_str());
    }
}
