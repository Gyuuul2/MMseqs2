#include "kmermatcher.h"
#include "Debug.h"
#include "FileUtil.h"
#include "FastSort.h"
#include "MMseqsNode.h"
#include "BatchReader.h"

#include <sys/resource.h>
#include <sys/stat.h>

#include <thread>
#include <algorithm>

#ifdef OPENMP
#include <omp.h>
#endif

static const size_t KMER_FILE_BUFFER_SIZE = 16 * 1024 * 1024;
static const size_t TASKS_PER_THREAD = 2;
static const size_t RANGES_PER_NODE = 8;
static const size_t MIN_TARGET_SPLITS_PER_NODE = 2;
static const size_t MAX_TARGET_SPLITS_PER_NODE = 8;
static const size_t SEND_KMER_SIZE = 12;

// codes of a k-mer file row
enum { KMER_MEMBER_CODES, KMER_REP_CANDIDATE_CODES, KMER_INDEX_CODES, KMER_CODE_COUNT };
// codes of a pair file row
enum { PAIR_INDEX_CODES, PAIR_TARGET_CODES, PAIR_CODE_COUNT };
enum { KMER_FILE_GROUPS, PAIR_FILE_GROUPS, FILE_GROUP_KINDS };

struct KmerFileRow {
    DBKeyType splitFirstKey;
    unsigned int idBits;
    size_t offsets[KMER_CODE_COUNT + 1];
    size_t groupCount;
};

struct PairFileRow {
    size_t offsets[PAIR_CODE_COUNT + 1];
    size_t partition;
    size_t recordCount;
    size_t targetCount;

    static bool comparePartition(const PairFileRow &row, size_t partition) {
        return row.partition < partition;
    }
};

struct PairRecord {
    DBKeyType rep;
    size_t targetOffset;
    size_t targetCount;

    static bool compareByRep(const PairRecord &first, const PairRecord &second) {
        return first.rep < second.rep;
    }

    static bool compareRepToKey(const PairRecord &record, size_t key) {
        return record.rep < key;
    }
};

struct PassRow {
    size_t file;
    size_t row;
    size_t bufferOffset;
};

struct KmerThreadTaskStart {
    size_t group;
    uint64_t hash;
    size_t bitPos[KMER_CODE_COUNT];
};

// start of hash sub range i in units of 2^shift, rounded up
static PackedKmerPosition::uint128 getHashSubRangeStart(size_t hashSubRange, size_t hashSubRangeCount, unsigned int shift) {
    PackedKmerPosition::uint128 start = (static_cast<PackedKmerPosition::uint128>(hashSubRange) << 64) / hashSubRangeCount;
    return (start + (static_cast<PackedKmerPosition::uint128>(1) << shift) - 1) >> shift;
}

// Rice parameter of the hash gaps
static unsigned int getHashDiffLowBits(size_t hashSubRange, size_t hashSubRangeEnd, size_t hashSubRangeCount, unsigned int shift, size_t groupCount) {
    PackedKmerPosition::uint128 width = getHashSubRangeStart(hashSubRangeEnd, hashSubRangeCount, shift) - getHashSubRangeStart(hashSubRange, hashSubRangeCount, shift);
    PackedKmerPosition::uint128 diff = width / std::max((size_t) 1, groupCount);
    return getBitWidth((diff > UINT64_MAX) ? UINT64_MAX : std::max((uint64_t) 1, (uint64_t) diff)) - 1;
}

struct RowValue {
    uint64_t value;
    unsigned int row;
    RowValue(uint64_t value, unsigned int row) : value(value), row(row) {}
    bool operator<(const RowValue &other) const {
        return value > other.value || (value == other.value && row > other.row);
    }
};

typedef std::priority_queue<RowValue> RowValueQueue;

struct KmerHashSubRangeRead {
    KmerCodeReader codeReads[KMER_CODE_COUNT];
    size_t codeStartBitPos[KMER_CODE_COUNT];
    const KmerFileRow *hashSubRangeRow;
    size_t taskRow;
    bool isWholeSequence;
    unsigned int hashDiffLowBits;
    size_t group;
    size_t groupEnd;
    uint64_t hash;
    size_t groupMemberCount;
    size_t groupRepCandidateCount;
    size_t pendingGroup;
    uint64_t pendingHash;
    size_t pendingIndexBitPos;
};

struct PairRowRead {
    KmerCodeReader codeReads[PAIR_CODE_COUNT];
    unsigned int repKeyDiffLowBits;
    size_t recordsLeft;
    size_t nextRepKey;
    size_t repKey;
    DBKeyType target;
    size_t targetsLeft;
    unsigned int targetGapBits;
};

struct PairKeyTaskStart {
    size_t recordsLeft;
    size_t nextRepKey;
    size_t bitPos[PAIR_CODE_COUNT];
};

struct PartitionCut {
    size_t key;
    size_t prefOffset;
    size_t cost;
};

struct KeyTaskCuts {
    KeyTaskCuts() : costStep(1) {}
    std::vector<PartitionCut> cuts;
    size_t costStep;

    void add(const PartitionCut &cut) {
        if (cut.cost / costStep == (cuts.empty() ? 0 : cuts.back().cost / costStep)) {
            return;
        }
        cuts.push_back(cut);
        while (cuts.size() >= 16) {
            costStep *= 2;
            size_t kept = 0;
            for (size_t i = 0; i < cuts.size(); i++) {
                if (cuts[i].cost / costStep != ((kept > 0) ? cuts[kept - 1].cost / costStep : 0)) {
                    cuts[kept++] = cuts[i];
                }
            }
            cuts.resize(kept);
        }
    }
};

struct PartitionMerge {
    std::vector<PairRowRead> rowReads;
    std::vector<PairKeyTaskStart> taskStarts;
    std::vector<size_t> keyOffsets;
    DBKeyType *targets;
    std::vector<size_t> keyTaskOffsets;
    std::vector<size_t> keyTaskDataOffsets;
    std::vector<size_t> keyTaskEntryOffsets;
    std::vector<size_t> keyTaskCostOffsets;
    std::vector<KeyTaskCuts> keyTaskCuts;
    size_t cost;
};

static size_t getPartitionMergeSize(size_t rowCount, size_t threads) {
    return rowCount * (sizeof(PairRowRead) + TASKS_PER_THREAD * threads * sizeof(PairKeyTaskStart)
                       + threads * (sizeof(PairRowRead) + 2 * sizeof(RowValue) + sizeof(unsigned int)));
}

static inline void queueNextKmerGroup(RowValueQueue &queue, unsigned int row, KmerHashSubRangeRead &read) {
    read.pendingGroup = read.group;
    read.pendingHash = read.hash;
    read.pendingIndexBitPos = getKmerCodeBitPos(read.codeReads[KMER_INDEX_CODES]);
    if (read.group == read.groupEnd) {
        return;
    }
    read.hash += getRice(read.codeReads[KMER_INDEX_CODES], read.hashDiffLowBits) + ((read.group > 0) ? 1 : 0);
    read.groupMemberCount = getEliasGamma(read.codeReads[KMER_INDEX_CODES]);
    read.groupRepCandidateCount = read.isWholeSequence ? 0 : getEliasGamma(read.codeReads[KMER_INDEX_CODES]);
    read.group++;
    queue.push(RowValue(read.hash, row));
}

struct ReceivedFile {
    std::string fileName;
    FILE *file;
    size_t rowTableOffset;
    size_t flushCount;
    char *data;
};

template <typename Row>
static size_t getRowEnd(const Row &row) {
    return row.offsets[sizeof(row.offsets) / sizeof(size_t) - 1];
}

template <typename Row>
static void readItemRows(const ReceivedFile &receivedFile, size_t item, std::vector<Row> &rows, std::vector<BatchReader::ReadRequest> *readRequests = NULL) {
    size_t flushCount = receivedFile.flushCount;
    std::vector<size_t> rowStarts;
    MMseqsNode::readExchangeTableColumns(fileno(receivedFile.file), receivedFile.fileName, receivedFile.rowTableOffset, flushCount, item, rowStarts);
    size_t rowCount = 0;
    for (size_t flush = 0; flush < flushCount; flush++) {
        rowCount += (rowStarts[flushCount + flush] - rowStarts[flush]) / sizeof(Row);
    }
    rows.resize(rowCount);
    size_t row = 0;
    for (size_t flush = 0; flush < flushCount; flush++) {
        size_t size = rowStarts[flushCount + flush] - rowStarts[flush];
        if (size == 0) {
            continue;
        }
        if (readRequests == NULL) {
            FileUtil::preadAll(fileno(receivedFile.file), receivedFile.fileName.c_str(), (char *) (rows.data() + row), size, rowStarts[flush]);
        } else {
            BatchReader::ReadRequest readRequest = {fileno(receivedFile.file), receivedFile.fileName.c_str(), (char *) (rows.data() + row), rowStarts[flush], size, size};
            readRequests->push_back(readRequest);
        }
        row += size / sizeof(Row);
    }
}

static void closeReceivedFiles(std::vector<ReceivedFile> &receivedFiles) {
    for (size_t i = 0; i < receivedFiles.size(); i++) {
        if (receivedFiles[i].file != NULL && fclose(receivedFiles[i].file) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << receivedFiles[i].fileName << "\n";
            EXIT(EXIT_FAILURE);
        }
        receivedFiles[i].file = NULL;
    }
}

struct ExchangeFileGroups {
    std::string prefix;
    std::string localTmp;
    size_t itemCount;
    size_t itemsPerGroup;
    size_t count;
    size_t localCount;
    size_t localNodes;
    std::vector<std::vector<uint64_t> > localLines;
    size_t lineWords;
    std::vector<size_t> nextIndexLines;
    std::vector<std::vector<size_t> > nodeFileSizes;
    size_t publishedLineCount;
    std::vector<std::string> names;
    std::vector<FILE *> files;
    std::vector<size_t> sizes;
    std::vector<std::vector<size_t> > flushRowOffsets;
    size_t flushCount;
    double waitSeconds;
};

static void initFileGroups(ExchangeFileGroups &groups, const std::string &prefix, size_t itemCount, size_t itemsPerGroup, size_t lineWords) {
    groups.prefix = prefix;
    groups.localCount = 0;
    groups.localNodes = 0;
    groups.itemCount = itemCount;
    groups.itemsPerGroup = itemsPerGroup;
    groups.count = (itemCount + itemsPerGroup - 1) / itemsPerGroup;
    groups.lineWords = lineWords;
    groups.nextIndexLines.assign(MMseqsNode::count, 0);
    groups.nodeFileSizes.assign(MMseqsNode::count, std::vector<size_t>(groups.count, SIZE_MAX));
    groups.publishedLineCount = 0;
    groups.flushCount = 0;
    groups.waitSeconds = 0;
}

static size_t getFileGroupItemEnd(const ExchangeFileGroups &groups, size_t group) {
    return std::min(groups.itemCount, (group + 1) * groups.itemsPerGroup);
}

static std::string getFileGroupName(const ExchangeFileGroups &groups, size_t node, size_t group) {
    return groups.prefix + "." + SSTR(node) + "." + SSTR(group);
}

static bool isLocalFileGroup(const ExchangeFileGroups &groups, size_t group) {
    return group + groups.localCount >= groups.count;
}

static std::string getLocalFileGroupName(const ExchangeFileGroups &groups, size_t group) {
    return groups.localTmp + "/" + FileUtil::baseName(getFileGroupName(groups, MMseqsNode::rank, group));
}

static FILE *copyLocalFileGroup(const ExchangeFileGroups &groups, size_t group, FILE *local, size_t size) {
    std::string name = getFileGroupName(groups, MMseqsNode::rank, group);
    FILE *shared = FileUtil::openAndDelete(name.c_str(), "w");
    char *buffer = new(std::nothrow) char[KMER_FILE_BUFFER_SIZE];
    Util::checkAllocation(buffer, "Can not allocate buffer memory in copyLocalFileGroup");
    for (size_t offset = 0; offset < size; offset += KMER_FILE_BUFFER_SIZE) {
        size_t copySize = std::min(KMER_FILE_BUFFER_SIZE, size - offset);
        FileUtil::preadAll(fileno(local), "local file group", buffer, copySize, offset);
        FileUtil::pwriteAll(fileno(shared), name.c_str(), buffer, copySize, offset);
    }
    delete[] buffer;
    if (fclose(local) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << getLocalFileGroupName(groups, group) << "\n";
        EXIT(EXIT_FAILURE);
    }
    FileUtil::remove(getLocalFileGroupName(groups, group).c_str());
    return shared;
}

static void fitLocalFileGroups(ExchangeFileGroups &groups, const std::vector<size_t> &flushStarts, const std::vector<size_t> &flushEnds, BatchWriter &batchWriter) {
    while (groups.localCount > 0) {
        size_t growth = 0;
        for (size_t i = groups.count - groups.localCount; i < groups.count; i++) {
            growth += flushEnds[i] - flushStarts[i];
        }
        // SIZE_MAX if statvfs failed
        size_t freeSize = FileUtil::getFreeSpace(groups.localTmp.c_str());
        if (freeSize != SIZE_MAX && 2 * growth * groups.localNodes <= freeSize) {
            return;
        }
        batchWriter.waitWrites();
        size_t first = groups.count - groups.localCount;
        MMseqsNode::reportUnusableLocalTmp("Cannot keep " + FileUtil::baseName(getFileGroupName(groups, MMseqsNode::rank, first)) + " in --local-tmp " + groups.localTmp + ": "
                                           + ((freeSize == SIZE_MAX) ? std::string("its free space is unknown") : "it has " + SSTR(freeSize) + " free bytes, the local file groups may need " + SSTR(2 * growth * groups.localNodes)));
        groups.files[first] = copyLocalFileGroup(groups, first, groups.files[first], flushStarts[first]);
        groups.names[first] = getFileGroupName(groups, MMseqsNode::rank, first);
        groups.localCount--;
    }
}

static void createFileGroups(ExchangeFileGroups &groups) {
    groups.names.resize(groups.count);
    groups.files.resize(groups.count);
    groups.sizes.assign(groups.count, MMseqsNode::EXCHANGE_FILE_HEADER_BYTES);
    groups.flushRowOffsets.assign(groups.count, std::vector<size_t>());
    for (size_t i = 0; i < groups.count; i++) {
        groups.names[i] = isLocalFileGroup(groups, i) ? getLocalFileGroupName(groups, i) : getFileGroupName(groups, MMseqsNode::rank, i);
        groups.files[i] = MMseqsNode::createExchangeFile(groups.names[i], i, "w+");
    }
}

static size_t closeFileGroups(ExchangeFileGroups &groups, const std::vector<size_t> &itemSizes) {
    size_t sent = 0;
    for (size_t i = 0; i < groups.count; i++) {
        size_t begin = i * groups.itemsPerGroup;
        size_t end = getFileGroupItemEnd(groups, i);
        std::vector<size_t> tail;
        if (itemSizes.empty() == false) {
            tail.assign(itemSizes.begin() + begin, itemSizes.begin() + end);
        }
        std::vector<uint64_t> line(groups.lineWords, 0);
        std::copy(tail.begin(), tail.end(), line.begin() + 3);
        tail.push_back(groups.flushCount);
        tail.push_back(groups.sizes[i]);
        size_t columns = (groups.flushCount > 0) ? groups.flushRowOffsets[i].size() / groups.flushCount : 1;
        size_t fileSize = MMseqsNode::writeExchangeTable(fileno(groups.files[i]), groups.names[i], groups.flushRowOffsets[i], columns, tail, groups.sizes[i]);
        line[0] = groups.publishedLineCount++;
        line[1] = i;
        line[2] = isLocalFileGroup(groups, i) ? 0 : fileSize;
        if (isLocalFileGroup(groups, i)) {
            groups.localLines.push_back(line);
        }
        MMseqsNode::publishExchangeFile(groups.files[i], groups.names[i], line[2], groups.prefix + ".index." + SSTR(MMseqsNode::rank), line);
        sent += groups.sizes[i] - MMseqsNode::EXCHANGE_FILE_HEADER_BYTES;
    }
    return sent;
}

static size_t readFileGroupIndex(ExchangeFileGroups &groups, unsigned int node, bool wait, std::vector<size_t> *itemSizes = NULL) {
    std::vector<uint64_t> lines(16 * groups.lineWords);
    size_t lineCount = MMseqsNode::waitForExchangeIndexLines(groups.prefix + ".index." + SSTR(node), groups.nextIndexLines[node], groups.lineWords, lines, wait);
    for (size_t i = 0; i < lineCount; i++) {
        uint64_t *line = lines.data() + i * groups.lineWords;
        size_t &fileSize = groups.nodeFileSizes[node][line[1]];
        for (size_t item = line[1] * groups.itemsPerGroup; fileSize == SIZE_MAX && itemSizes != NULL && item < getFileGroupItemEnd(groups, line[1]); item++) {
            (*itemSizes)[item] += line[3 + item - line[1] * groups.itemsPerGroup];
        }
        if (fileSize == SIZE_MAX || line[2] > 0) {
            fileSize = line[2];
        }
    }
    return lineCount;
}

static bool waitForFileGroup(ExchangeFileGroups &groups, size_t group, bool wait) {
    Timer timer;
    for (unsigned int node = 0; node < MMseqsNode::count; node++) {
        while (groups.nodeFileSizes[node][group] == 0 || groups.nodeFileSizes[node][group] == SIZE_MAX) {
            if (readFileGroupIndex(groups, node, wait) == 0) {
                return false;
            }
        }
    }
    groups.waitSeconds += timer.getTimediff();
    return true;
}

static void openFileGroups(const ExchangeFileGroups &groups, size_t group, std::vector<ReceivedFile> &receivedFiles, int threads) {
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (size_t node = 0; node < MMseqsNode::count; node++) {
        ReceivedFile &receivedFile = receivedFiles[node];
        receivedFile.fileName = getFileGroupName(groups, node, group);
        size_t fileSize = groups.nodeFileSizes[node][group];
        receivedFile.file = fdopen(MMseqsNode::waitForExchangeFile(receivedFile.fileName, fileSize, group), "r+");
        if (receivedFile.file == NULL) {
            Debug(Debug::ERROR) << "Cannot open " << receivedFile.fileName << "\n";
            EXIT(EXIT_FAILURE);
        }
        size_t tail[2];
        FileUtil::preadAll(fileno(receivedFile.file), receivedFile.fileName.c_str(), (char *) tail, sizeof(tail), fileSize - sizeof(tail));
        receivedFile.flushCount = tail[0];
        receivedFile.rowTableOffset = tail[1];
    }
}

static void followFileGroups(ExchangeFileGroups &groups) {
    size_t served = 0;
    for (size_t i = 0; i <= groups.count; i++) {
        for (; served < groups.localLines.size() && std::min(served + 2, groups.localLines[served][1]) <= i + 1; served++) {
            std::vector<uint64_t> &line = groups.localLines[served];
            std::string localName = getLocalFileGroupName(groups, line[1]);
            FILE *local = FileUtil::openFileOrDie(localName.c_str(), "r", true);
            struct stat sb;
            if (fstat(fileno(local), &sb) != 0) {
                Debug(Debug::ERROR) << "Cannot read " << localName << "\n";
                EXIT(EXIT_FAILURE);
            }
            line[0] = groups.publishedLineCount++;
            line[2] = sb.st_size;
            FILE *shared = copyLocalFileGroup(groups, line[1], local, line[2]);
            MMseqsNode::publishExchangeFile(shared, getFileGroupName(groups, MMseqsNode::rank, line[1]), line[2], groups.prefix + ".index." + SSTR(MMseqsNode::rank), line);
        }
        if (i == groups.count) {
            break;
        }
        MMseqsNode::waitForRunFile(groups.prefix + ".released." + SSTR(i));
        FileUtil::remove(getFileGroupName(groups, MMseqsNode::rank, i).c_str());
    }
}

static void addFinishedItems(const Parameters &par, std::string &progress, size_t kind, size_t group, size_t items) {
    progress += SSTR(kind) + "\t" + SSTR(group) + "\t" + SSTR(items) + "\n";
    MMseqsNode::publishRunFile(par.db2 + ".progress." + SSTR(MMseqsNode::rank), progress);
}

static void releaseFileGroups(const Parameters &par, ExchangeFileGroups &kmerGroups, ExchangeFileGroups &pairGroups) {
    ExchangeFileGroups *fileGroups[FILE_GROUP_KINDS] = { &kmerGroups, &pairGroups };
    std::vector<char> isReleased[FILE_GROUP_KINDS];
    size_t groupsLeft = 0;
    for (size_t kind = 0; kind < FILE_GROUP_KINDS; kind++) {
        isReleased[kind].assign(fileGroups[kind]->count, false);
        groupsLeft += fileGroups[kind]->count;
    }
    std::string progress;
    while (groupsLeft > 0) {
        MMseqsNode::waitForRunFilesChange(par.db2 + ".progress.", progress);
        std::vector<size_t> finishedItems[FILE_GROUP_KINDS];
        for (size_t kind = 0; kind < FILE_GROUP_KINDS; kind++) {
            finishedItems[kind].assign(fileGroups[kind]->count, 0);
        }
        for (char *line = (char *) progress.c_str(); *line != '\0'; line = Util::skipLine(line)) {
            const char *columns[3];
            Util::getWordsOfLine(line, columns, 3);
            finishedItems[Util::fast_atoi<size_t>(columns[0])][Util::fast_atoi<size_t>(columns[1])] += Util::fast_atoi<size_t>(columns[2]);
        }
        for (size_t kind = 0; kind < FILE_GROUP_KINDS; kind++) {
            ExchangeFileGroups &groups = *fileGroups[kind];
            for (size_t i = 0; i < groups.count; i++) {
                if (isReleased[kind][i] == false && finishedItems[kind][i] == getFileGroupItemEnd(groups, i) - i * groups.itemsPerGroup) {
                    MMseqsNode::publishRunFile(groups.prefix + ".released." + SSTR(i), "");
                    isReleased[kind][i] = true;
                    groupsLeft--;
                }
            }
        }
    }
}

template <typename Row>
static void readPassRows(const std::vector<ReceivedFile> &receivedFiles, const std::vector<std::vector<Row> > &fileRows, std::vector<PassRow> &passRows, char *buffer, bool isAsync,
                         BatchReader *&batchReader, int threads) {
    std::vector<size_t> readOrder(passRows.size());
    for (size_t i = 0; i < readOrder.size(); i++) {
        readOrder[i] = i;
    }
    SORT_PARALLEL(readOrder.begin(), readOrder.end(), [&](size_t first, size_t second) {
        return passRows[first].file < passRows[second].file || (passRows[first].file == passRows[second].file && passRows[first].row < passRows[second].row);
    });
    size_t bufferPos = 0;
    std::vector<BatchReader::ReadRequest> readRequests;
    for (size_t i = 0; i < readOrder.size(); i++) {
        PassRow &passRow = passRows[readOrder[i]];
        const Row &codeRow = fileRows[passRow.file][passRow.row];
        size_t size = getRowEnd(codeRow) - codeRow.offsets[0];
        const ReceivedFile &receivedFile = receivedFiles[passRow.file];
        if (size == 0 || receivedFile.data != NULL) {
            continue;
        }
        passRow.bufferOffset = bufferPos;
        BatchReader::ReadRequest *last = readRequests.empty() ? NULL : &readRequests.back();
        if (last != NULL && last->fileToRead == fileno(receivedFile.file) && last->fileOffset + last->bytesToRead == codeRow.offsets[0]) {
            last->bytesToRead += size;
            last->minBytesToRead += size;
        } else {
            BatchReader::ReadRequest readRequest = {fileno(receivedFile.file), receivedFile.fileName.c_str(), buffer + bufferPos, codeRow.offsets[0], size, size};
            readRequests.push_back(readRequest);
        }
        bufferPos += size;
    }
    if (readRequests.empty()) {
        return;
    }
    if (batchReader == NULL) {
        batchReader = new BatchReader(threads, readRequests[0].fileToRead);
    }
    if (isAsync) {
        batchReader->readAsync(readRequests);
    } else {
        batchReader->readSync(readRequests);
    }
}

template <typename Row>
static void punchPassRows(const std::vector<ReceivedFile> &receivedFiles, const std::vector<std::vector<Row> > &fileRows, const std::vector<PassRow> &passRows, size_t &releasedSize) {
    std::vector<std::pair<size_t, std::pair<size_t, size_t> > > fileRanges;
    for (size_t i = 0; i < passRows.size(); i++) {
        const Row &codeRow = fileRows[passRows[i].file][passRows[i].row];
        if (getRowEnd(codeRow) > codeRow.offsets[0]) {
            fileRanges.push_back(std::make_pair(passRows[i].file, std::make_pair(codeRow.offsets[0], getRowEnd(codeRow))));
        }
    }
    SORT_SERIAL(fileRanges.begin(), fileRanges.end());
    for (size_t i = 0; i < fileRanges.size();) {
        size_t file = fileRanges[i].first;
        // memory is aligned at its addresses
        size_t base = (size_t) receivedFiles[file].data;
        size_t alignment = (receivedFiles[file].data == NULL) ? (size_t) 1024 * 1024 : Util::getPageSize();
        size_t start = (base + fileRanges[i].second.first + alignment - 1) / alignment * alignment;
        size_t end = fileRanges[i].second.second;
        for (i++; i < fileRanges.size() && fileRanges[i].first == file && fileRanges[i].second.first == end; i++) {
            end = fileRanges[i].second.second;
        }
        end = (base + end) / alignment * alignment;
        if (end <= start) {
            continue;
        }
        if (receivedFiles[file].data == NULL) {
            FileUtil::punchHole(fileno(receivedFiles[file].file), start, end - start);
        } else {
            Util::releasePages((char *) start, end - start);
            releasedSize += end - start;
        }
    }
}

template <typename Row>
static KmerCodeReader getRowCodeReader(const ReceivedFile &receivedFile, const Row &codeRow, size_t bufferOffset, size_t code, char *passBuffer, char *codeReadBuffer, size_t codeReadBufferSize) {
    KmerCodeReader codeRead = {NULL, 0, 0, 0, -1, NULL, SIZE_MAX, 0, 0};
    if (receivedFile.data != NULL) {
        codeRead.buffer = receivedFile.data + codeRow.offsets[code];
    } else if (codeReadBufferSize == 0) {
        codeRead.buffer = passBuffer + bufferOffset + codeRow.offsets[code] - codeRow.offsets[0];
    } else {
        codeRead.buffer = codeReadBuffer;
        codeRead.fd = fileno(receivedFile.file);
        codeRead.fileName = receivedFile.fileName.c_str();
        codeRead.bufferSize = codeReadBufferSize;
        codeRead.fileEnd = codeRow.offsets[code + 1];
        refillKmerCodeBuffer(codeRead, codeRow.offsets[code]);
    }
    return codeRead;
}

static size_t planPasses(const std::vector<size_t> &itemSizes, size_t passLimit, std::vector<size_t> &passOffsets, std::vector<size_t> &passSizes) {
    passOffsets.assign(1, 0);
    passSizes.assign(1, 0);
    size_t maxPassSize = 0;
    for (size_t item = 0; item < itemSizes.size(); item++) {
        if (item > passOffsets.back() && passSizes.back() + itemSizes[item] > passLimit) {
            passOffsets.push_back(item);
            passSizes.push_back(0);
        }
        passSizes.back() += itemSizes[item];
        if (passSizes.back() <= passLimit) {
            maxPassSize = std::max(maxPassSize, passSizes.back());
        }
    }
    passOffsets.push_back(itemSizes.size());
    return maxPassSize;
}

static size_t getPassLimit(size_t memory) {
    return memory / 3 / sizeof(size_t) * sizeof(size_t);
}

struct KmerExchange {
    std::vector<size_t> sendSplitKeyOffsets;
    size_t sendSplitCount;
    std::vector<size_t> fileKeyOffsets;
    bool hasRepresentativeBits;
    size_t memoryLimit;
    size_t writeBehindSize;
    size_t keySplitMemoryLimit;
    size_t pairRowSize;
    unsigned int adjKeyBits;
    unsigned int adjKeyBase;
    unsigned int adjKeyCodeBits;
    unsigned int kmerHashShift;
    unsigned int globalIdBits;
    size_t maxRepSeqCount;
    size_t pairTargetCount;
    size_t selfMergedSendSplitCount;
    bool hasTargetSplits;
    size_t hashRangeCount;
    size_t hashSubRangesPerRange;
    size_t hashSubRangeCount;
    size_t rowsPerHashRange;
    size_t partitionsPerRange;
};

static bool getRowHashSubRanges(const KmerExchange &exchange, size_t hashRange, size_t rangeHashSubRange, size_t &first, size_t &end) {
    bool isWholeSequence = (rangeHashSubRange >= exchange.hashSubRangesPerRange);
    first = hashRange * exchange.hashSubRangesPerRange + (isWholeSequence ? 0 : rangeHashSubRange);
    end = first + (isWholeSequence ? exchange.hashSubRangesPerRange : 1);
    return isWholeSequence;
}

static inline void putAdjKeyCode(KmerCodeWriter &writer, const KmerExchange &exchange, unsigned int adjKey) {
    unsigned int adjKeyCode = 0;
    for (int j = 5; j >= 0; j--) {
        adjKeyCode = adjKeyCode * exchange.adjKeyBase + PackedKmerPosition::getAdjacentSeq(adjKey, j);
    }
    putBits(writer, adjKeyCode, exchange.adjKeyCodeBits);
}

static inline unsigned int getAdjKeyCode(KmerCodeReader &reader, const KmerExchange &exchange) {
    unsigned int adjKeyCode = getBits(reader, exchange.adjKeyCodeBits);
    unsigned int adjKey = 0;
    for (int j = 0; j < 6; j++) {
        adjKey |= (adjKeyCode % exchange.adjKeyBase) << (PackedKmerPosition::RESIDUE_BITS * j);
        adjKeyCode /= exchange.adjKeyBase;
    }
    return adjKey;
}

static size_t getPartitionFirstKey(const KmerExchange &exchange, size_t sendSplit, size_t partition) {
    return exchange.sendSplitKeyOffsets[sendSplit] + (partition * (exchange.sendSplitKeyOffsets[sendSplit + 1] - exchange.sendSplitKeyOffsets[sendSplit]) + exchange.partitionsPerRange - 1) / exchange.partitionsPerRange;
}

static void setSendSplit(const Parameters &par, const KmerExchange &exchange, size_t sendSplit) {
    sendSplitFirstKey = exchange.sendSplitKeyOffsets[sendSplit];
    delete[] representativeBits;
    representativeBits = NULL;
    if (exchange.hasRepresentativeBits) {
        representativeBits = MMseqsNode::readRepresentativeBits(par.db1, exchange.fileKeyOffsets, sendSplitFirstKey, exchange.sendSplitKeyOffsets[sendSplit + 1] - sendSplitFirstKey);
    }
}

static size_t getGroupIds(const PackedKmerPosition *hashSeqPair, size_t groupStart, size_t groupEnd, uint64_t *groupBuffer, size_t groupBufferWords, uint64_t *setBitCounts, size_t keySplitWords) {
    if (groupEnd - groupStart <= groupBufferWords) {
        for (size_t pos = groupStart; pos < groupEnd; pos++) {
            groupBuffer[pos - groupStart] = hashSeqPair[pos].getId();
        }
        SORT_SERIAL(groupBuffer, groupBuffer + (groupEnd - groupStart));
        return static_cast<size_t>(std::unique(groupBuffer, groupBuffer + (groupEnd - groupStart)) - groupBuffer);
    }
    memset(groupBuffer, 0, keySplitWords * sizeof(uint64_t));
    for (size_t pos = groupStart; pos < groupEnd; pos++) {
        DBKeyType id = hashSeqPair[pos].getId();
        groupBuffer[id / 64] |= 1ULL << (id % 64);
    }
    size_t memberCount = 0;
    for (size_t word = 0; word < keySplitWords; word++) {
        if (word % 8 == 0) {
            setBitCounts[word / 8] = memberCount;
        }
        memberCount += __builtin_popcountll(groupBuffer[word]);
    }
    return memberCount;
}

static void getPassRows(const std::vector<std::vector<KmerFileRow> > &fileRows, const std::vector<PassRow> &rowOrder, size_t passFirstHashSubRange, size_t passEndHashSubRange,
                        std::vector<PassRow> &rows, std::vector<size_t> &rowOffsets) {
    rows.clear();
    rowOffsets.assign(1, 0);
    for (size_t rangeHashSubRange = passFirstHashSubRange; rangeHashSubRange < passEndHashSubRange; rangeHashSubRange++) {
        for (size_t i = 0; i < rowOrder.size(); i++) {
            PassRow passRow = rowOrder[i];
            passRow.row += rangeHashSubRange;
            if (fileRows[passRow.file][passRow.row].groupCount > 0) {
                rows.push_back(passRow);
            }
        }
        rowOffsets.push_back(rows.size());
    }
}

struct OwnPairs {
    char *data;
    size_t size;
    size_t limit;
    std::vector<std::vector<PairFileRow> > sendSplitRows;
};

struct PairFile {
    ExchangeFileGroups *groups;
    BatchWriter *batchWriter;
    OwnPairs *ownPairs;
    std::vector<size_t> partitionSizes;
    size_t targetCount;
};

static unsigned int getRepKeyDiffLowBits(size_t partitionKeys, size_t recordCount) {
    return getBitWidth(std::max((size_t) 1, partitionKeys / std::max((size_t) 1, recordCount))) - 1;
}

struct RepTargetMerge {
    DBKeyType *repTargets;
    size_t repTargetCapacity;
    size_t sortedRepStart;
    size_t sortedRepTargetCount;
    std::vector<size_t> recordTargets;
    RowValueQueue targetQueue;
};

template <typename AddTarget>
static void mergeRepTargets(const PairRecord *records, size_t repStart, size_t repEnd, const DBKeyType *targets, RepTargetMerge &merge, AddTarget &addTarget) {
    DBKeyType rep = records[repStart].rep;
    if (repEnd - repStart == 1) {
        for (size_t target = 0; target < records[repStart].targetCount; target++) {
            if (targets[records[repStart].targetOffset + target] != rep) {
                addTarget(targets[records[repStart].targetOffset + target]);
            }
        }
        return;
    }
    if (merge.sortedRepStart != repStart) {
        size_t repTargetCount = 0;
        for (size_t record = repStart; record < repEnd; record++) {
            repTargetCount += records[record].targetCount;
        }
        if (repTargetCount <= merge.repTargetCapacity) {
            DBKeyType *repTargetsEnd = merge.repTargets;
            for (size_t record = repStart; record < repEnd; record++) {
                repTargetsEnd = std::copy(targets + records[record].targetOffset, targets + records[record].targetOffset + records[record].targetCount, repTargetsEnd);
            }
            SORT_SERIAL(merge.repTargets, repTargetsEnd);
            repTargetsEnd = std::unique(merge.repTargets, repTargetsEnd);
            merge.sortedRepTargetCount = static_cast<size_t>(std::remove(merge.repTargets, repTargetsEnd, rep) - merge.repTargets);
            merge.sortedRepStart = repStart;
        }
    }
    if (merge.sortedRepStart == repStart) {
        for (size_t target = 0; target < merge.sortedRepTargetCount; target++) {
            addTarget(merge.repTargets[target]);
        }
        return;
    }
    std::vector<size_t> &recordTargets = merge.recordTargets;
    RowValueQueue &targetQueue = merge.targetQueue;
    recordTargets.assign(repEnd - repStart, 0);
    for (size_t record = repStart; record < repEnd; record++) {
        targetQueue.push(RowValue(targets[records[record].targetOffset], record - repStart));
    }
    DBKeyType prevTarget = DB_KEY_INVALID;
    while (targetQueue.empty() == false) {
        RowValue target = targetQueue.top();
        targetQueue.pop();
        if (target.value != prevTarget && target.value != rep) {
            addTarget(target.value);
            prevTarget = target.value;
        }
        const PairRecord &record = records[repStart + target.row];
        recordTargets[target.row]++;
        if (recordTargets[target.row] < record.targetCount) {
            targetQueue.push(RowValue(targets[record.targetOffset + recordTargets[target.row]], target.row));
        }
    }
}

struct TargetGapCounter {
    DBKeyType maxGap;
    DBKeyType prevTarget;
    size_t count;

    void operator()(DBKeyType target) {
        if (count > 0) {
            maxGap = std::max(maxGap, target - prevTarget - 1);
        }
        prevTarget = target;
        count++;
    }
};

struct RepTargetWriter {
    KmerCodeWriter &indexWrite;
    KmerCodeWriter &targetWrite;
    unsigned int globalIdBits;
    size_t targetCount;
    unsigned int gapBits;
    unsigned int gapBitsWidth;
    size_t written;
    DBKeyType prevTarget;

    void operator()(DBKeyType id) {
        if (written == 0) {
            putBits(indexWrite, id, globalIdBits);
            if (targetCount > 1) {
                putBits(indexWrite, gapBits, gapBitsWidth);
            }
        } else {
            putBits(targetWrite, id - prevTarget - 1, gapBits);
        }
        prevTarget = id;
        written++;
    }
};

static unsigned int getRepTargetGapBits(const PairRecord *records, size_t repStart, size_t repEnd, const DBKeyType *targets, RepTargetMerge &merge, size_t &repTargetCount) {
    TargetGapCounter gaps = { 0, 0, 0 };
    mergeRepTargets(records, repStart, repEnd, targets, merge, gaps);
    repTargetCount = gaps.count;
    return getBitWidth(gaps.maxGap);
}

static void writePairFileRows(const KmerExchange &exchange, PairFile &pairFile, PairRecord *records, size_t recordCount, DBKeyType *targets,
                              char *writeBuffers, int threads) {
    if (recordCount == 0) {
        return;
    }
    // partitions are key ranges
    const std::vector<size_t> &sendSplitKeyOffsets = exchange.sendSplitKeyOffsets;
    std::vector<size_t> rowRecordOffsets(1, 0);
    std::vector<size_t> rowFirstKeys;
    std::vector<size_t> rowKeys;
    std::vector<PairFileRow> rows;
    std::vector<size_t> sendSplitRows;
    while (rowRecordOffsets.back() < recordCount) {
        size_t rep = records[rowRecordOffsets.back()].rep;
        size_t sendSplit = std::upper_bound(sendSplitKeyOffsets.begin(), sendSplitKeyOffsets.end(), rep) - sendSplitKeyOffsets.begin() - 1;
        size_t partition = (rep - sendSplitKeyOffsets[sendSplit]) * exchange.partitionsPerRange / (sendSplitKeyOffsets[sendSplit + 1] - sendSplitKeyOffsets[sendSplit]);
        rowFirstKeys.push_back(getPartitionFirstKey(exchange, sendSplit, partition));
        rowKeys.push_back(getPartitionFirstKey(exchange, sendSplit, partition + 1) - rowFirstKeys.back());
        rowRecordOffsets.push_back(std::lower_bound(records + rowRecordOffsets.back(), records + recordCount, rowFirstKeys.back() + rowKeys.back(), PairRecord::compareRepToKey) - records);
        while (sendSplitRows.size() <= sendSplit) {
            sendSplitRows.push_back(rows.size());
        }
        PairFileRow row = {{0, 0, 0}, partition, 0, 0};
        rows.push_back(row);
    }
    sendSplitRows.resize(exchange.sendSplitCount + 1, rows.size());
    unsigned int targetGapBitsWidth = getBitWidth(exchange.globalIdBits);
    // count the code size of every row
#pragma omp parallel num_threads(threads)
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        RepTargetMerge merge;
        merge.repTargets = (DBKeyType *) (writeBuffers + thread_idx * KMER_FILE_BUFFER_SIZE + KMER_FILE_BUFFER_SIZE / 2);
        merge.repTargetCapacity = KMER_FILE_BUFFER_SIZE / 2 / sizeof(DBKeyType);
        merge.sortedRepStart = SIZE_MAX;
#pragma omp for schedule(dynamic, 1)
        for (size_t row = 0; row < rows.size(); row++) {
            size_t rowEnd = rowRecordOffsets[row + 1];
            for (size_t record = rowRecordOffsets[row]; record < rowEnd; record++) {
                rows[row].recordCount += (record == rowRecordOffsets[row] || records[record].rep != records[record - 1].rep);
            }
            unsigned int repKeyDiffLowBits = getRepKeyDiffLowBits(rowKeys[row], rows[row].recordCount);
            size_t codeBits[PAIR_CODE_COUNT] = {0, 0};
            size_t nextRepKey = 0;
            for (size_t repStart = rowRecordOffsets[row]; repStart < rowEnd;) {
                size_t repEnd = repStart + 1;
                while (repEnd < rowEnd && records[repEnd].rep == records[repStart].rep) {
                    repEnd++;
                }
                for (size_t i = repStart + 16; i < std::min(repEnd + 16, rowEnd); i++) {
                    __builtin_prefetch(targets + records[i].targetOffset);
                }
                size_t repKey = records[repStart].rep - rowFirstKeys[row];
                size_t repTargetCount = 0;
                unsigned int targetGapBits = getRepTargetGapBits(records, repStart, repEnd, targets, merge, repTargetCount);
                codeBits[PAIR_INDEX_CODES] += ((repKey - nextRepKey) >> repKeyDiffLowBits) + 1 + repKeyDiffLowBits + 2 * getBitWidth(repTargetCount) - 1 + exchange.globalIdBits
                                              + ((repTargetCount > 1) ? targetGapBitsWidth : 0);
                codeBits[PAIR_TARGET_CODES] += (repTargetCount - 1) * targetGapBits;
                rows[row].targetCount += repTargetCount;
                nextRepKey = repKey + 1;
                repStart = repEnd;
            }
            for (size_t code = 0; code < PAIR_CODE_COUNT; code++) {
                rows[row].offsets[code + 1] = (codeBits[code] + CHAR_BIT - 1) / CHAR_BIT;
            }
        }
    }
    OwnPairs &ownPairs = *pairFile.ownPairs;
    size_t ownBytesBefore = ownPairs.size;
    std::vector<char> isOwnSendSplit(exchange.sendSplitCount, false);
    ExchangeFileGroups &groups = *pairFile.groups;
    std::vector<size_t> flushStarts(groups.sizes);
    std::vector<size_t> flushEnds(groups.sizes);
    for (size_t sendSplit = 0; sendSplit < exchange.sendSplitCount; sendSplit++) {
        size_t sendSplitSize = 0;
        for (size_t row = sendSplitRows[sendSplit]; row < sendSplitRows[sendSplit + 1]; row++) {
            for (size_t code = 0; code < PAIR_CODE_COUNT; code++) {
                sendSplitSize += rows[row].offsets[code + 1];
            }
        }
        isOwnSendSplit[sendSplit] = sendSplit % MMseqsNode::count == MMseqsNode::rank && sendSplit / MMseqsNode::count < exchange.selfMergedSendSplitCount
                                    && ownPairs.size + sendSplitSize <= ownPairs.limit;
        for (size_t row = sendSplitRows[sendSplit]; row < sendSplitRows[sendSplit + 1]; row++) {
            pairFile.targetCount += rows[row].targetCount;
            size_t &size = isOwnSendSplit[sendSplit] ? ownPairs.size : pairFile.groups->sizes[(sendSplit * exchange.partitionsPerRange + rows[row].partition) / pairFile.groups->itemsPerGroup];
            rows[row].offsets[0] = size;
            for (size_t code = 0; code < PAIR_CODE_COUNT; code++) {
                rows[row].offsets[code + 1] += rows[row].offsets[code];
            }
            size = rows[row].offsets[PAIR_CODE_COUNT];
            if (isOwnSendSplit[sendSplit] == false) {
                pairFile.partitionSizes[sendSplit * exchange.partitionsPerRange + rows[row].partition] += size - rows[row].offsets[0];
                flushEnds[(sendSplit * exchange.partitionsPerRange + rows[row].partition) / groups.itemsPerGroup] += size - rows[row].offsets[0] + sizeof(PairFileRow);
            }
        }
    }
    fitLocalFileGroups(groups, flushStarts, flushEnds, *pairFile.batchWriter);
    if (ownPairs.size > ownBytesBefore) {
        ownPairs.data = static_cast<char *>(realloc(ownPairs.data, ownPairs.size));
        Util::checkAllocation(ownPairs.data, "Can not allocate ownPairs memory in writePairFileRows");
    }
#pragma omp parallel num_threads(threads)
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        char *writeBuffer = writeBuffers + thread_idx * KMER_FILE_BUFFER_SIZE;
        RepTargetMerge merge;
        merge.repTargets = (DBKeyType *) (writeBuffer + KMER_FILE_BUFFER_SIZE / 2);
        merge.repTargetCapacity = KMER_FILE_BUFFER_SIZE / 2 / sizeof(DBKeyType);
        merge.sortedRepStart = SIZE_MAX;
#pragma omp for schedule(dynamic, 1)
        for (size_t row = 0; row < rows.size(); row++) {
            size_t sendSplit = std::upper_bound(sendSplitRows.begin(), sendSplitRows.end(), row) - sendSplitRows.begin() - 1;
            bool isOwnRow = isOwnSendSplit[sendSplit];
            size_t group = (sendSplit * exchange.partitionsPerRange + rows[row].partition) / pairFile.groups->itemsPerGroup;
            KmerCodeWriter codeWrites[PAIR_CODE_COUNT];
            for (size_t code = 0; code < PAIR_CODE_COUNT; code++) {
                size_t writePos = rows[row].offsets[code];
                codeWrites[code] = isOwnRow
                    ? KmerCodeWriter(ownPairs.data, ownPairs.size, writePos)
                    : KmerCodeWriter(fileno(pairFile.groups->files[group]), pairFile.groups->names[group].c_str(), writeBuffer + code * KMER_FILE_BUFFER_SIZE / 4, KMER_FILE_BUFFER_SIZE / 4, writePos, pairFile.batchWriter);
            }
            KmerCodeWriter &indexWrite = codeWrites[PAIR_INDEX_CODES];
            KmerCodeWriter &targetWrite = codeWrites[PAIR_TARGET_CODES];
            unsigned int repKeyDiffLowBits = getRepKeyDiffLowBits(rowKeys[row], rows[row].recordCount);
            size_t rowEnd = rowRecordOffsets[row + 1];
            size_t nextRepKey = 0;
            for (size_t repStart = rowRecordOffsets[row]; repStart < rowEnd;) {
                size_t repEnd = repStart + 1;
                while (repEnd < rowEnd && records[repEnd].rep == records[repStart].rep) {
                    repEnd++;
                }
                for (size_t i = repStart + 16; i < std::min(repEnd + 16, rowEnd); i++) {
                    __builtin_prefetch(targets + records[i].targetOffset);
                }
                size_t repKey = records[repStart].rep - rowFirstKeys[row];
                size_t repTargetCount = 0;
                unsigned int targetGapBits = getRepTargetGapBits(records, repStart, repEnd, targets, merge, repTargetCount);
                putRice(indexWrite, repKey - nextRepKey, repKeyDiffLowBits);
                putEliasGamma(indexWrite, repTargetCount);
                RepTargetWriter targetWriter = { indexWrite, targetWrite, exchange.globalIdBits, repTargetCount, targetGapBits, targetGapBitsWidth, 0, 0 };
                mergeRepTargets(records, repStart, repEnd, targets, merge, targetWriter);
                nextRepKey = repKey + 1;
                repStart = repEnd;
            }
            putBits(indexWrite, 0, (CHAR_BIT - indexWrite.bitCount) % CHAR_BIT);
            putBits(targetWrite, 0, (CHAR_BIT - targetWrite.bitCount) % CHAR_BIT);
            if (isOwnRow == false) {
                writeKmerCodeBuffer(indexWrite);
                writeKmerCodeBuffer(targetWrite);
            }
        }
    }
    for (size_t sendSplit = 0; sendSplit < exchange.sendSplitCount; sendSplit++) {
        if (isOwnSendSplit[sendSplit]) {
            ownPairs.sendSplitRows[sendSplit].insert(ownPairs.sendSplitRows[sendSplit].end(), rows.data() + sendSplitRows[sendSplit], rows.data() + sendSplitRows[sendSplit + 1]);
        }
    }
    for (size_t i = 0; i < groups.count; i++) {
        size_t groupBegin = i * groups.itemsPerGroup;
        size_t groupEnd = getFileGroupItemEnd(groups, i);
        for (size_t sendSplit = groupBegin / exchange.partitionsPerRange; sendSplit * exchange.partitionsPerRange < groupEnd; sendSplit++) {
            groups.flushRowOffsets[i].push_back(groups.sizes[i]);
            if (isOwnSendSplit[sendSplit]) {
                continue;
            }
            size_t first = std::max(groupBegin, sendSplit * exchange.partitionsPerRange) - sendSplit * exchange.partitionsPerRange;
            size_t end = std::min(groupEnd, (sendSplit + 1) * exchange.partitionsPerRange) - sendSplit * exchange.partitionsPerRange;
            PairFileRow *groupRows = std::lower_bound(rows.data() + sendSplitRows[sendSplit], rows.data() + sendSplitRows[sendSplit + 1], first, PairFileRow::comparePartition);
            size_t groupRowCount = std::lower_bound(groupRows, rows.data() + sendSplitRows[sendSplit + 1], end, PairFileRow::comparePartition) - groupRows;
            pairFile.batchWriter->write(fileno(groups.files[i]), groups.names[i].c_str(), (const char *) groupRows, groupRowCount * sizeof(PairFileRow), groups.sizes[i]);
            groups.sizes[i] += groupRowCount * sizeof(PairFileRow);
        }
        groups.flushRowOffsets[i].push_back(groups.sizes[i]);
    }
    groups.flushCount++;
}

struct PairOutput {
    PairRecord *records;
    DBKeyType *targets;
    size_t size;
    size_t regionSize;
    size_t recordsTaken;
    size_t targetsTaken;
    std::vector<std::pair<size_t, size_t> > recordRegions;
};

struct PairOutputRegion {
    size_t recordStart;
    size_t recordPos;
    size_t recordEnd;
    size_t targetPos;
    size_t targetEnd;
};

static bool takePairOutput(PairOutput &output, PairOutputRegion &region, size_t recordCount, size_t targetCount) {
    bool hasRecords = (region.recordEnd - region.recordPos >= recordCount);
    bool hasTargets = (region.targetEnd - region.targetPos >= targetCount);
    if (hasRecords && hasTargets) {
        return true;
    }
    bool isTaken = false;
#pragma omp critical(takePairOutput)
    {
        size_t reservedRecordCount = hasRecords ? 0 : std::max(recordCount, output.regionSize / sizeof(PairRecord));
        size_t reservedTargetCount = hasTargets ? 0 : std::max(targetCount, output.regionSize / sizeof(DBKeyType));
        if ((output.recordsTaken + reservedRecordCount) * sizeof(PairRecord) + (output.targetsTaken + reservedTargetCount) * sizeof(DBKeyType) > output.size) {
            reservedRecordCount = hasRecords ? 0 : recordCount;
            reservedTargetCount = hasTargets ? 0 : targetCount;
        }
        if ((output.recordsTaken + reservedRecordCount) * sizeof(PairRecord) + (output.targetsTaken + reservedTargetCount) * sizeof(DBKeyType) <= output.size) {
            if (reservedRecordCount > 0) {
                if (region.recordPos > region.recordStart) {
                    output.recordRegions.push_back(std::make_pair(region.recordStart, region.recordPos));
                }
                region.recordStart = output.recordsTaken;
                region.recordPos = output.recordsTaken;
                output.recordsTaken += reservedRecordCount;
                region.recordEnd = output.recordsTaken;
            }
            if (reservedTargetCount > 0) {
                output.targetsTaken += reservedTargetCount;
                region.targetPos = output.size / sizeof(DBKeyType) - output.targetsTaken;
                region.targetEnd = region.targetPos + reservedTargetCount;
            }
            isTaken = true;
        }
    }
    return isTaken;
}

static void flushPairOutput(const KmerExchange &exchange, PairFile &pairFile, PairOutput &output, std::vector<PairOutputRegion> &regions, char *writeBuffers, int threads) {
    for (size_t thread = 0; thread < regions.size(); thread++) {
        if (regions[thread].recordPos > regions[thread].recordStart) {
            output.recordRegions.push_back(std::make_pair(regions[thread].recordStart, regions[thread].recordPos));
        }
        regions[thread] = PairOutputRegion();
    }
    SORT_SERIAL(output.recordRegions.begin(), output.recordRegions.end());
    size_t recordCount = 0;
    for (size_t i = 0; i < output.recordRegions.size(); i++) {
        memmove(output.records + recordCount, output.records + output.recordRegions[i].first, (output.recordRegions[i].second - output.recordRegions[i].first) * sizeof(PairRecord));
        recordCount += output.recordRegions[i].second - output.recordRegions[i].first;
    }
    if (threads > 1) {
        SORT_PARALLEL(output.records, output.records + recordCount, PairRecord::compareByRep);
    } else {
        SORT_SERIAL(output.records, output.records + recordCount, PairRecord::compareByRep);
    }
    writePairFileRows(exchange, pairFile, output.records, recordCount, output.targets, writeBuffers, threads);
    output.recordsTaken = 0;
    output.targetsTaken = 0;
    output.recordRegions.clear();
}

static void getPairPassRows(const std::vector<std::vector<std::pair<size_t, size_t> > > &partitionRows, size_t passFirstPartition, size_t passEndPartition,
                            std::vector<PassRow> &rows, std::vector<size_t> &rowOffsets) {
    rows.clear();
    rowOffsets.assign(1, 0);
    for (size_t partition = passFirstPartition; partition < passEndPartition; partition++) {
        for (size_t row = 0; row < partitionRows[partition].size(); row++) {
            PassRow passRow = {partitionRows[partition][row].first, partitionRows[partition][row].second, 0};
            rows.push_back(passRow);
        }
        rowOffsets.push_back(rows.size());
    }
}

static inline void readPairRecord(PairRowRead &read, unsigned int globalIdBits) {
    KmerCodeReader &indexRead = read.codeReads[PAIR_INDEX_CODES];
    read.repKey = read.nextRepKey + getRice(indexRead, read.repKeyDiffLowBits);
    read.nextRepKey = read.repKey + 1;
    read.targetsLeft = getEliasGamma(indexRead) - 1;
    read.target = getBits(indexRead, globalIdBits);
    read.targetGapBits = (read.targetsLeft > 0) ? getBits(indexRead, getBitWidth(globalIdBits)) : 0;
    read.recordsLeft--;
}

struct KeyTargetCopy {
    DBKeyType *targets;
    size_t count;

    void operator()(DBKeyType target) {
        targets[count++] = target;
    }
};

struct PrefTargetWriter {
    bool hasEntry;
    KmerCodeWriter &dataWrite;
    DBReader<DBKeyType> &seqDbr;
    DBKeyType prevTarget;
    size_t targetRange;
    size_t residues;

    void operator()(DBKeyType target) {
        if (hasEntry) {
            putPrefKey(dataWrite, target - prevTarget);
            prevTarget = target;
            residues += seqDbr.getSeqLenFromRange(targetRange, target);
        }
    }
};

template <typename AddTarget>
static void mergeKeyTargets(std::vector<PairRowRead> &reads, RowValueQueue &repKeyQueue, RowValueQueue &targetQueue, std::vector<unsigned int> &keyRows, size_t key,
                            unsigned int globalIdBits, AddTarget &addTarget) {
    keyRows.clear();
    while (repKeyQueue.empty() == false && repKeyQueue.top().value == key) {
        keyRows.push_back(repKeyQueue.top().row);
        repKeyQueue.pop();
    }
    for (size_t row = 0; row < keyRows.size(); row++) {
        targetQueue.push(RowValue(reads[keyRows[row]].target, keyRows[row]));
    }
    DBKeyType prevTarget = DB_KEY_INVALID;
    while (targetQueue.empty() == false) {
        RowValue target = targetQueue.top();
        targetQueue.pop();
        if (target.value != prevTarget) {
            addTarget(target.value);
            prevTarget = target.value;
        }
        PairRowRead &read = reads[target.row];
        if (read.targetsLeft > 0) {
            read.target += getBits(read.codeReads[PAIR_TARGET_CODES], read.targetGapBits) + 1;
            read.targetsLeft--;
            targetQueue.push(RowValue(read.target, target.row));
        }
    }
    for (size_t row = 0; row < keyRows.size(); row++) {
        PairRowRead &read = reads[keyRows[row]];
        if (read.recordsLeft > 0) {
            readPairRecord(read, globalIdBits);
            repKeyQueue.push(RowValue(read.repKey, keyRows[row]));
        }
    }
}

static void getPartitionRowReads(const std::vector<ReceivedFile> &receivedPairFiles, const std::vector<std::vector<PairFileRow> > &pairFileRows, const PassRow *partitionPassRows,
                                 size_t partitionRowCount, size_t partitionKeys, char *codeBuffer, size_t codeReadBufferSize, std::vector<PairRowRead> &rowReads) {
    rowReads.assign(partitionRowCount, PairRowRead());
    for (size_t passRow = 0; passRow < partitionRowCount; passRow++) {
        const PassRow &partitionPassRow = partitionPassRows[passRow];
        const PairFileRow &pairRow = pairFileRows[partitionPassRow.file][partitionPassRow.row];
        PairRowRead &read = rowReads[passRow];
        for (size_t code = 0; code < PAIR_CODE_COUNT; code++) {
            read.codeReads[code] = getRowCodeReader(receivedPairFiles[partitionPassRow.file], pairRow, partitionPassRow.bufferOffset, code, codeBuffer,
                                                    codeBuffer + (passRow * PAIR_CODE_COUNT + code) * codeReadBufferSize, codeReadBufferSize);
        }
        read.repKeyDiffLowBits = getRepKeyDiffLowBits(partitionKeys, pairRow.recordCount);
        read.recordsLeft = pairRow.recordCount;
    }
}

static void countPartitionTargets(const KmerExchange &exchange, const std::vector<ReceivedFile> &receivedPairFiles, const std::vector<std::vector<PairFileRow> > &pairFileRows,
                                  const PassRow *partitionPassRows, size_t partitionRowCount, char *codeBuffer, size_t partitionKeys, size_t partitionTargets, PartitionMerge &merge, int threads) {
    std::vector<PairRowRead> &rowReads = merge.rowReads;
    std::vector<size_t> &keyOffsets = merge.keyOffsets;
    getPartitionRowReads(receivedPairFiles, pairFileRows, partitionPassRows, partitionRowCount, partitionKeys, codeBuffer, 0, rowReads);
    keyOffsets.resize(partitionKeys + 1);
    // prefix sum per thread, then over the threads
    std::vector<size_t> partTargets(threads + 1, 0);
#pragma omp parallel num_threads(threads)
    {
#pragma omp for schedule(static)
        for (size_t key = 0; key <= partitionKeys; key++) {
            keyOffsets[key] = 0;
        }
#pragma omp for schedule(dynamic, 1)
        for (size_t row = 0; row < rowReads.size(); row++) {
            PairRowRead read = rowReads[row];
            while (read.recordsLeft > 0) {
                readPairRecord(read, exchange.globalIdBits);
                if (threads == 1) {
                    keyOffsets[read.repKey + 1] += read.targetsLeft + 1;
                } else {
                    __sync_fetch_and_add(&keyOffsets[read.repKey + 1], read.targetsLeft + 1);
                }
            }
        }
#pragma omp for schedule(static)
        for (int part = 0; part < threads; part++) {
            size_t partBegin = 1 + partitionKeys * part / threads;
            size_t partEnd = 1 + partitionKeys * (part + 1) / threads;
            for (size_t key = partBegin + 1; key < partEnd; key++) {
                keyOffsets[key] += keyOffsets[key - 1];
            }
            partTargets[part + 1] = (partEnd > partBegin) ? keyOffsets[partEnd - 1] : 0;
        }
#pragma omp single
        for (int part = 0; part < threads; part++) {
            partTargets[part + 1] += partTargets[part];
        }
#pragma omp for schedule(static)
        for (int part = 1; part < threads; part++) {
            for (size_t key = 1 + partitionKeys * part / threads; key < 1 + partitionKeys * (part + 1) / threads; key++) {
                keyOffsets[key] += partTargets[part];
            }
        }
    }
    size_t keyTaskWeight = std::max((size_t) 1, (partitionTargets + partitionKeys + TASKS_PER_THREAD * threads - 1) / (TASKS_PER_THREAD * threads));
    merge.keyTaskOffsets.assign(1, 0);
    for (size_t task = 1; merge.keyTaskOffsets.back() < partitionKeys; task++) {
        size_t low = merge.keyTaskOffsets.back();
        size_t high = partitionKeys;
        while (low < high) {
            size_t mid = low + (high - low) / 2;
            if (keyOffsets[mid] + mid < task * keyTaskWeight) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        if (low > merge.keyTaskOffsets.back()) {
            merge.keyTaskOffsets.push_back(low);
        }
    }
    size_t keyTaskCount = merge.keyTaskOffsets.size() - 1;
    merge.taskStarts.resize(keyTaskCount * rowReads.size());
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (size_t row = 0; row < rowReads.size(); row++) {
        PairRowRead read = rowReads[row];
        PairKeyTaskStart start = {read.recordsLeft, 0, {0, 0}};
        merge.taskStarts[row] = start;
        for (size_t task = 1; task < keyTaskCount;) {
            start.recordsLeft = read.recordsLeft;
            start.nextRepKey = read.nextRepKey;
            start.bitPos[PAIR_INDEX_CODES] = getKmerCodeBitPos(read.codeReads[PAIR_INDEX_CODES]);
            if (read.recordsLeft > 0) {
                readPairRecord(read, exchange.globalIdBits);
            }
            for (; task < keyTaskCount && (start.recordsLeft == 0 || read.repKey >= merge.keyTaskOffsets[task]); task++) {
                merge.taskStarts[task * rowReads.size() + row] = start;
            }
            start.bitPos[PAIR_TARGET_CODES] += read.targetsLeft * read.targetGapBits;
        }
    }
}

static void mergePartitionTargets(const KmerExchange &exchange, DBReader<DBKeyType> &seqDbr, size_t partitionFirstKey, PartitionMerge &merge, DBKeyType *targets, int threads) {
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    const std::vector<size_t> &keyOffsets = merge.keyOffsets;
    merge.targets = targets;
    size_t keyTaskCount = merge.keyTaskOffsets.size() - 1;
    size_t rowCount = merge.rowReads.size();
    merge.keyTaskDataOffsets.assign(keyTaskCount + 1, 0);
    merge.keyTaskEntryOffsets.assign(keyTaskCount + 1, 0);
    merge.keyTaskCostOffsets.assign(keyTaskCount + 1, 0);
    merge.keyTaskCuts.assign(keyTaskCount, KeyTaskCuts());
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (size_t keyTask = 0; keyTask < keyTaskCount; keyTask++) {
        std::vector<PairRowRead> reads(merge.rowReads);
        RowValueQueue repKeyQueue;
        RowValueQueue targetQueue;
        std::vector<unsigned int> keyRows;
        for (size_t row = 0; row < rowCount; row++) {
            const PairKeyTaskStart &start = merge.taskStarts[keyTask * rowCount + row];
            PairRowRead &read = reads[row];
            for (size_t code = 0; code < PAIR_CODE_COUNT; code++) {
                if (start.bitPos[code] > 0) {
                    seekKmerCodeBits(read.codeReads[code], getKmerCodeBitPos(read.codeReads[code]) + start.bitPos[code]);
                }
            }
            read.recordsLeft = start.recordsLeft;
            read.nextRepKey = start.nextRepKey;
            if (read.recordsLeft > 0) {
                readPairRecord(read, exchange.globalIdBits);
                repKeyQueue.push(RowValue(read.repKey, row));
            }
        }
        for (size_t key = merge.keyTaskOffsets[keyTask]; key < merge.keyTaskOffsets[keyTask + 1]; key++) {
            DBKeyType *keyTargets = targets + keyOffsets[key];
            KeyTargetCopy targetCopy = { keyTargets, 0 };
            mergeKeyTargets(reads, repKeyQueue, targetQueue, keyRows, key, exchange.globalIdBits, targetCopy);
            size_t keyTargetCount = targetCopy.count;
            std::fill(keyTargets + keyTargetCount, targets + keyOffsets[key + 1], DB_KEY_INVALID);
            if (keyTargetCount == 0 || isRepresentative(partitionFirstKey + key) == false) {
                continue;
            }
            size_t entrySize = getPrefKeySize(partitionFirstKey + key) + 1;
            size_t targetRange = SIZE_MAX;
            size_t targetResidues = 0;
            for (size_t target = 0; target < keyTargetCount; target++) {
                entrySize += getPrefKeySize(keyTargets[target] - ((target > 0) ? keyTargets[target - 1] : 0));
                targetResidues += seqDbr.getSeqLenFromRange(targetRange, keyTargets[target]);
            }
            merge.keyTaskDataOffsets[keyTask + 1] += entrySize;
            merge.keyTaskEntryOffsets[keyTask + 1]++;
            merge.keyTaskCostOffsets[keyTask + 1] += (index[seqDbr.findIdRange(partitionFirstKey + key)].length - 2) * targetResidues;
            PartitionCut cut = { key + 1, merge.keyTaskDataOffsets[keyTask + 1], merge.keyTaskCostOffsets[keyTask + 1] };
            merge.keyTaskCuts[keyTask].add(cut);
        }
    }
    std::vector<PairRowRead>().swap(merge.rowReads);
    std::vector<PairKeyTaskStart>().swap(merge.taskStarts);
    for (size_t keyTask = 0; keyTask < keyTaskCount; keyTask++) {
        merge.keyTaskDataOffsets[keyTask + 1] += merge.keyTaskDataOffsets[keyTask];
        merge.keyTaskEntryOffsets[keyTask + 1] += merge.keyTaskEntryOffsets[keyTask];
        merge.keyTaskCostOffsets[keyTask + 1] += merge.keyTaskCostOffsets[keyTask];
    }
    merge.cost = merge.keyTaskCostOffsets[keyTaskCount];
}

struct PrefFiles {
    std::string dataName;
    std::string indexName;
    FILE *dataFile;
    FILE *indexFile;
    char *writeBuffers;
    BatchWriter *batchWriter;
};

static void writePartition(const PrefFiles &prefFiles, size_t partitionFirstKey, const PartitionMerge &merge, size_t dataOffset, size_t indexOffset, size_t firstWriteBuffer, int threads) {
    size_t keyTaskCount = merge.keyTaskOffsets.size() - 1;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (size_t keyTask = 0; keyTask < keyTaskCount; keyTask++) {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        char *writeBuffer = prefFiles.writeBuffers + 2 * (firstWriteBuffer + thread_idx) * KMER_FILE_BUFFER_SIZE;
        KmerCodeWriter dataWrite(fileno(prefFiles.dataFile), prefFiles.dataName.c_str(), writeBuffer, KMER_FILE_BUFFER_SIZE, dataOffset + merge.keyTaskDataOffsets[keyTask], prefFiles.batchWriter);
        KmerCodeWriter indexWrite(fileno(prefFiles.indexFile), prefFiles.indexName.c_str(), writeBuffer + KMER_FILE_BUFFER_SIZE, KMER_FILE_BUFFER_SIZE,
                                                    indexOffset + merge.keyTaskEntryOffsets[keyTask] * 2 * sizeof(uint64_t), prefFiles.batchWriter);
        for (size_t key = merge.keyTaskOffsets[keyTask]; key < merge.keyTaskOffsets[keyTask + 1]; key++) {
            if (merge.keyOffsets[key + 1] > merge.keyOffsets[key] && isRepresentative(partitionFirstKey + key)) {
                uint64_t indexCell[2] = {partitionFirstKey + key, dataWrite.writePos + dataWrite.bufferPos};
                putBytes(indexWrite, (const char *) indexCell, sizeof(indexCell));
                putPrefKey(dataWrite, partitionFirstKey + key);
                DBKeyType prevTarget = 0;
                for (size_t target = merge.keyOffsets[key]; target < merge.keyOffsets[key + 1] && merge.targets[target] != DB_KEY_INVALID; target++) {
                    putPrefKey(dataWrite, merge.targets[target] - prevTarget);
                    prevTarget = merge.targets[target];
                }
                putBytes(dataWrite, "", 1);
            }
        }
        writeKmerCodeBuffer(dataWrite);
        writeKmerCodeBuffer(indexWrite);
    }
}

// target split mode: the pref of a node has a data and an index file per target split, file node * target splits + target split
struct TargetSplitFiles {
    std::vector<size_t> keyOffsets;
    std::vector<std::string> dataNames;
    std::vector<std::string> indexNames;
    std::vector<FILE *> dataFiles;
    std::vector<size_t> dataSizes;
    // (first key, data offset) of every entry, written when the files are closed
    std::vector<std::vector<uint64_t> > indexCells;
    // the cells of the window being written
    std::vector<size_t> cellOffsets;
    std::vector<size_t> cellCosts;
};

// the targets of a rep in one target split
struct TargetSplitPiece {
    size_t key;
    size_t targetCount;
    const DBKeyType *first;
    const DBKeyType *last;
};

// an entry of every target split for the keys from firstKey; a piece of a rep in the entry is the rep key gap, the size class of all
// targets of the rep, its targets in the target split as gaps from the target split start and a 0
static void writeTargetSplits(const PrefFiles &prefFiles, TargetSplitFiles &files, DBReader<DBKeyType> &seqDbr, size_t firstKey, size_t keyCount, const size_t *keyOffsets,
                              const DBKeyType *targets, int threads) {
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    const size_t splitCount = files.dataSizes.size();
    for (size_t split = 0; split < splitCount; split++) {
        files.indexCells[split].push_back(firstKey);
        files.indexCells[split].push_back(files.dataSizes[split]);
    }
    // a task keeps its pieces by target split in the second write buffer of a thread, so it has at most that many targets or one key
    const size_t pieceCapacity = KMER_FILE_BUFFER_SIZE / sizeof(TargetSplitPiece);
    std::vector<size_t> prevKeys(splitCount, firstKey);
    std::vector<size_t> taskKeyOffsets;
    std::vector<size_t> taskPieceOffsets(threads * (splitCount + 1));
    size_t chunkFirstKey = 0;
    do {
        size_t taskTargets = std::min(pieceCapacity, (keyOffsets[keyCount] - keyOffsets[chunkFirstKey] + threads - 1) / threads);
        taskKeyOffsets.assign(1, chunkFirstKey);
        while (taskKeyOffsets.size() <= static_cast<size_t>(threads) && taskKeyOffsets.back() < keyCount) {
            size_t taskFirstKey = taskKeyOffsets.back();
            size_t taskEndKey = std::upper_bound(keyOffsets + taskFirstKey + 1, keyOffsets + keyCount + 1, keyOffsets[taskFirstKey] + taskTargets) - keyOffsets - 1;
            taskKeyOffsets.push_back(std::max(taskFirstKey + 1, taskEndKey));
        }
        size_t taskCount = taskKeyOffsets.size() - 1;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
        for (size_t task = 0; task < taskCount; task++) {
            TargetSplitPiece *pieces = (TargetSplitPiece *) (prefFiles.writeBuffers + (2 * task + 1) * KMER_FILE_BUFFER_SIZE);
            size_t *pieceOffsets = taskPieceOffsets.data() + task * (splitCount + 1);
            std::fill(pieceOffsets, pieceOffsets + splitCount + 1, 0);
            std::vector<size_t> nextPiece;
            // count the pieces of each target split, then fill them in key order
            for (int fill = 0; fill < 2; fill++) {
                for (size_t key = taskKeyOffsets[task]; key < taskKeyOffsets[task + 1]; key++) {
                    const DBKeyType *begin = targets + keyOffsets[key];
                    const DBKeyType *end = std::lower_bound(begin, targets + keyOffsets[key + 1], DB_KEY_INVALID);
                    if (begin == end || isRepresentative(firstKey + key) == false) {
                        continue;
                    }
                    for (const DBKeyType *first = begin; first < end;) {
                        size_t split = std::upper_bound(files.keyOffsets.begin(), files.keyOffsets.end(), *first) - files.keyOffsets.begin() - 1;
                        const DBKeyType *last = std::lower_bound(first, end, files.keyOffsets[split + 1]);
                        if (fill == 0) {
                            pieceOffsets[split + 1]++;
                        } else {
                            TargetSplitPiece piece = { key, static_cast<size_t>(end - begin), first, last };
                            pieces[nextPiece[split]++] = piece;
                        }
                        first = last;
                    }
                }
                if (fill == 0) {
                    for (size_t split = 0; split < splitCount; split++) {
                        pieceOffsets[split + 1] += pieceOffsets[split];
                    }
                    nextPiece.assign(pieceOffsets, pieceOffsets + splitCount);
                }
            }
        }
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
        for (size_t split = 0; split < splitCount; split++) {
            unsigned int thread_idx = 0;
#ifdef OPENMP
            thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
            KmerCodeWriter dataWrite(fileno(files.dataFiles[split]), files.dataNames[split].c_str(), prefFiles.writeBuffers + 2 * thread_idx * KMER_FILE_BUFFER_SIZE, KMER_FILE_BUFFER_SIZE,
                                     files.dataSizes[split], prefFiles.batchWriter);
            const DBKeyType splitFirstKey = files.keyOffsets[split];
            for (size_t task = 0; task < taskCount; task++) {
                const TargetSplitPiece *pieces = (const TargetSplitPiece *) (prefFiles.writeBuffers + (2 * task + 1) * KMER_FILE_BUFFER_SIZE);
                const size_t *pieceOffsets = taskPieceOffsets.data() + task * (splitCount + 1);
                for (size_t i = pieceOffsets[split]; i < pieceOffsets[split + 1]; i++) {
                    putPrefKey(dataWrite, firstKey + pieces[i].key - prevKeys[split]);
                    putPrefKey(dataWrite, getBitWidth(pieces[i].targetCount) - 1);
                    DBKeyType prevTarget = splitFirstKey - 1;
                    size_t targetRange = SIZE_MAX;
                    size_t targetResidues = 0;
                    for (const DBKeyType *target = pieces[i].first; target < pieces[i].last; target++) {
                        putPrefKey(dataWrite, *target - prevTarget);
                        prevTarget = *target;
                        targetResidues += seqDbr.getSeqLenFromRange(targetRange, *target);
                    }
                    putBytes(dataWrite, "", 1);
                    files.cellCosts[split] += (index[seqDbr.findIdRange(firstKey + pieces[i].key)].length - 2) * targetResidues;
                    prevKeys[split] = firstKey + pieces[i].key;
                }
            }
            if (taskKeyOffsets.back() == keyCount) {
                putBytes(dataWrite, "", 1);
            }
            writeKmerCodeBuffer(dataWrite);
            files.dataSizes[split] = dataWrite.writePos;
        }
        chunkFirstKey = taskKeyOffsets.back();
    } while (chunkFirstKey < keyCount);
}

// one row per (window, target split): first key, offset, size, cost, node, target split, window
static void addTargetSplitRows(TargetSplitFiles &files, size_t windowFirstKey, size_t window, std::string &partitionsData) {
    for (size_t split = 0; split < files.dataSizes.size(); split++) {
        partitionsData += SSTR(windowFirstKey) + "\t" + SSTR(files.cellOffsets[split]) + "\t" + SSTR(files.dataSizes[split] - files.cellOffsets[split]) + "\t" + SSTR(files.cellCosts[split])
                          + "\t" + SSTR(MMseqsNode::rank) + "\t" + SSTR(split) + "\t" + SSTR(window) + "\n";
        files.cellOffsets[split] = files.dataSizes[split];
        files.cellCosts[split] = 0;
    }
}

struct OwnKmers {
    char *data;
    size_t size;
    size_t limit;
    std::vector<std::vector<KmerFileRow> > hashRangeRows;
    std::vector<size_t> hashRanges;
};

static size_t countRepresentatives(size_t dbFrom, size_t from, size_t to) {
    return (representativeBits == NULL) ? to - from : Util::countSetBits(representativeBits, from - dbFrom, to - dbFrom);
}

static bool isKmerRecord(const PackedKmerPosition &kmer) {
    return kmer.isWholeSequence() == false;
}

static bool compareKmerHash(const PackedKmerPosition &kmer, uint64_t hash) {
    return static_cast<uint64_t>(kmer.getKmer()) < hash;
}

template <typename T, bool includeAdjacency, typename L>
static size_t sendKmers(Parameters &par, DBReader<DBKeyType> &seqDbr, BaseMatrix *subMat, const KmerExchange &exchange, MMseqsNode::WorkQueue &sendQueue,
                        size_t representativeBitsMemory, size_t &sentSplitCount, OwnKmers &ownKmers, ExchangeFileGroups &kmerGroups) {
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    float kmersPerSequenceScale = par.kmersPerSequenceScale.values.aminoacid();
    size_t readChunkLimit = std::min(BatchReader::MAX_INFLIGHT_READ_SIZE, exchange.keySplitMemoryLimit / 6);
    // the current and the prefetched input buffer, page aligned in every data file
    size_t readBufferBytes = 2 * (std::max(readChunkLimit, seqDbr.getMaxSeqLen()) + 2 * Util::getPageSize() * seqDbr.getDataFileCnt());
    // ids must fit next to the hash and the adjKey
    size_t keySplitKeyLimit = (1ULL << (128 - 1 - 64 - exchange.adjKeyBits)) - 1;
    createFileGroups(kmerGroups);
    std::vector<KmerFileRow> keySplitRows(exchange.hashRangeCount * exchange.rowsPerHashRange);
    std::vector<size_t> hashSubRangeOffsets(exchange.hashSubRangeCount + 1);
    std::vector<size_t> wholeSequenceOffsets(exchange.hashRangeCount + 1);
    char *kmerWriteBuffer = new(std::nothrow) char[par.threads * KMER_FILE_BUFFER_SIZE];
    Util::checkAllocation(kmerWriteBuffer, "Can not allocate kmerWriteBuffer memory in sendKmers");
    BatchWriter batchWriter(exchange.writeBehindSize, par.threads, fileno(kmerGroups.files[0]));
    PackedKmerPosition *hashSeqPair = NULL;
    size_t hashSeqPairCapacity = 0;
    std::vector<char> isOwnHashRange(exchange.hashRangeCount);
    for (size_t sendSplit = sendQueue.takeNextItem(); sendSplit < exchange.sendSplitCount; sendSplit = sendQueue.takeNextItem()) {
        sentSplitCount++;
        setSendSplit(par, exchange, sendSplit);
        size_t dbFrom = exchange.sendSplitKeyOffsets[sendSplit];
        size_t dbSize = exchange.sendSplitKeyOffsets[sendSplit + 1] - dbFrom;
        if (countRepresentatives(dbFrom, dbFrom, dbFrom + dbSize) == 0) {
            continue;
        }
        size_t firstRange = (dbSize > 0) ? seqDbr.findIdRange(dbFrom) : seqDbr.getIndexSize();
        size_t keySplitMemoryLimit = exchange.keySplitMemoryLimit - std::min(exchange.keySplitMemoryLimit, ownKmers.size + readBufferBytes);
        size_t kmerCapacity = keySplitMemoryLimit / sizeof(PackedKmerPosition);
        std::vector<size_t> keySplitOffsets;
        size_t maxKeySplitKmers = 0;
        size_t maxKeySplitKeys = 0;
        size_t groupBufferWords = 0;
        size_t fixedSize = 0;
        while (true) {
            keySplitOffsets.assign(1, dbFrom);
            size_t keySplitKmers = 0;
            maxKeySplitKmers = 0;
            for (size_t range = firstRange; range < seqDbr.getIndexSize() && index[range].id < dbFrom + dbSize; range++) {
                size_t rangeFrom = std::max(static_cast<size_t>(index[range].id), dbFrom);
                size_t rangeTo = std::min((range + 1 < seqDbr.getIndexSize()) ? static_cast<size_t>(index[range + 1].id) : seqDbr.getSize(), dbFrom + dbSize);
                size_t seqKmers = computeSeqKmerCount(static_cast<int>(index[range].length - 2), par.kmerSize, par.kmersPerSequence, kmersPerSequenceScale);
                if (seqKmers > kmerCapacity) {
                    Debug(Debug::ERROR) << "Not enough memory to extract the k-mers of a sequence of length " << index[range].length - 2 << ". Raise --split-memory-limit.\n";
                    EXIT(EXIT_FAILURE);
                }
                size_t rangeRepresentatives = countRepresentatives(dbFrom, rangeFrom, rangeTo);
                size_t cutKey = rangeFrom;
                while (keySplitKmers + rangeRepresentatives * seqKmers > kmerCapacity || rangeTo - keySplitOffsets.back() > keySplitKeyLimit) {
                    size_t fitCount = std::min(rangeRepresentatives, (kmerCapacity - keySplitKmers) / seqKmers);
                    size_t nextKeySplitKey = rangeTo;
                    if (fitCount < rangeRepresentatives && representativeBits == NULL) {
                        nextKeySplitKey = cutKey + fitCount;
                    } else if (fitCount < rangeRepresentatives) {
                        size_t skipCount = fitCount;
                        size_t bit = cutKey - dbFrom;
                        uint64_t word = representativeBits[bit / 64] & (~0ULL << (bit % 64));
                        while (static_cast<size_t>(__builtin_popcountll(word)) <= skipCount) {
                            skipCount -= __builtin_popcountll(word);
                            bit = bit / 64 * 64 + 64;
                            word = representativeBits[bit / 64];
                        }
                        for (; skipCount > 0; skipCount--) {
                            word &= word - 1;
                        }
                        nextKeySplitKey = dbFrom + bit / 64 * 64 + __builtin_ctzll(word);
                    }
                    if (nextKeySplitKey - keySplitOffsets.back() > keySplitKeyLimit) {
                        nextKeySplitKey = keySplitOffsets.back() + keySplitKeyLimit;
                        fitCount = countRepresentatives(dbFrom, cutKey, nextKeySplitKey);
                    }
                    maxKeySplitKmers = std::max(maxKeySplitKmers, keySplitKmers + fitCount * seqKmers);
                    rangeRepresentatives -= fitCount;
                    cutKey = nextKeySplitKey;
                    keySplitOffsets.push_back(cutKey);
                    keySplitKmers = 0;
                }
                keySplitKmers += rangeRepresentatives * seqKmers;
                maxKeySplitKmers = std::max(maxKeySplitKmers, keySplitKmers);
            }
            keySplitOffsets.push_back(dbFrom + dbSize);
            maxKeySplitKeys = 0;
            for (size_t keySplit = 0; keySplit + 1 < keySplitOffsets.size(); keySplit++) {
                maxKeySplitKeys = std::max(maxKeySplitKeys, keySplitOffsets[keySplit + 1] - keySplitOffsets[keySplit]);
            }
            // ids of a group, or a bit per key with set bit counts per 512
            groupBufferWords = (maxKeySplitKeys + 63) / 64 + (maxKeySplitKeys + 511) / 512;
            fixedSize = par.threads * groupBufferWords * sizeof(uint64_t);
            if (fixedSize + maxKeySplitKmers * sizeof(PackedKmerPosition) <= keySplitMemoryLimit) {
                break;
            }
            if (maxKeySplitKmers == 0) {
                Debug(Debug::ERROR) << "Not enough memory to run the kmermatcher. Memory limit: " << exchange.memoryLimit
                                    << " bytes, the group buffers need: " << representativeBitsMemory + fixedSize << " bytes.\n"
                                    << "Raise --split-memory-limit.\n";
                EXIT(EXIT_FAILURE);
            }
            kmerCapacity = std::min(maxKeySplitKmers - 1, (fixedSize < keySplitMemoryLimit) ? (keySplitMemoryLimit - fixedSize) / sizeof(PackedKmerPosition) : maxKeySplitKmers / 2);
        }
        size_t keySplitCount = keySplitOffsets.size() - 1;

        unsigned int idBits = getBitWidth(maxKeySplitKeys);
        PackedKmerPosition::idShift = 0;
        PackedKmerPosition::seqLenShift = idBits;
        PackedKmerPosition::adjKeyShift = idBits;
        PackedKmerPosition::kmerShift = idBits + exchange.adjKeyBits;
        PackedKmerPosition::maxSeqLen = 0;

        if (maxKeySplitKmers > hashSeqPairCapacity || fixedSize + hashSeqPairCapacity * sizeof(PackedKmerPosition) > keySplitMemoryLimit) {
            Util::releasePages(hashSeqPair, hashSeqPairCapacity * sizeof(PackedKmerPosition));
            delete[] hashSeqPair;
            hashSeqPairCapacity = maxKeySplitKmers;
            hashSeqPair = new(std::nothrow) PackedKmerPosition[hashSeqPairCapacity];
            Util::checkAllocation(hashSeqPair, "Can not allocate hashSeqPair memory in sendKmers");
        }
        uint64_t *groupBuffers = new(std::nothrow) uint64_t[par.threads * groupBufferWords];
        Util::checkAllocation(groupBuffers, "Can not allocate groupBuffers memory in sendKmers");
        for (size_t keySplit = 0; keySplit < keySplitCount; keySplit++) {
            if (countRepresentatives(dbFrom, keySplitOffsets[keySplit], keySplitOffsets[keySplit + 1]) == 0) {
                continue;
            }
            std::vector<size_t> readChunkOffsets(1, keySplitOffsets[keySplit]);
            size_t readChunkSize = 0;
            size_t keySplitFirstRange = (keySplitOffsets[keySplit] < keySplitOffsets[keySplit + 1]) ? seqDbr.findIdRange(keySplitOffsets[keySplit]) : seqDbr.getIndexSize();
            for (size_t range = keySplitFirstRange; range < seqDbr.getIndexSize() && index[range].id < keySplitOffsets[keySplit + 1]; range++) {
                size_t rangeFrom = std::max(static_cast<size_t>(index[range].id), keySplitOffsets[keySplit]);
                size_t rangeTo = std::min((range + 1 < seqDbr.getIndexSize()) ? static_cast<size_t>(index[range + 1].id) : seqDbr.getSize(), keySplitOffsets[keySplit + 1]);
                for (size_t key = rangeFrom; key < rangeTo; ) {
                    if (readChunkSize > 0 && readChunkSize + index[range].length > readChunkLimit) {
                        readChunkOffsets.push_back(key);
                        readChunkSize = 0;
                    }
                    size_t readChunkSeqCount = std::min(rangeTo - key, std::max((size_t) 1, (readChunkLimit - readChunkSize) / index[range].length));
                    key += readChunkSeqCount;
                    readChunkSize += readChunkSeqCount * index[range].length;
                }
            }
            readChunkOffsets.push_back(keySplitOffsets[keySplit + 1]);
            size_t kmerCount = fillKmerPositionArray<Parameters::DBTYPE_AMINO_ACIDS, T, includeAdjacency, L>(
                hashSeqPair, maxKeySplitKmers, seqDbr, par, subMat, true, 0, SIZE_MAX, NULL, readChunkOffsets).first;

            SORT_PARALLEL(hashSeqPair, hashSeqPair + kmerCount);

            // whole sequence records follow the k-mer records
            size_t wholeSequenceStart = std::partition_point(hashSeqPair, hashSeqPair + kmerCount, isKmerRecord) - hashSeqPair;
#pragma omp parallel for schedule(static) num_threads(par.threads)
            for (size_t hashSubRange = 0; hashSubRange < exchange.hashSubRangeCount; hashSubRange++) {
                uint64_t hashSubRangeStart = static_cast<uint64_t>(getHashSubRangeStart(hashSubRange, exchange.hashSubRangeCount, 0));
                hashSubRangeOffsets[hashSubRange] = std::lower_bound(hashSeqPair, hashSeqPair + wholeSequenceStart, hashSubRangeStart, compareKmerHash) - hashSeqPair;
                if (hashSubRange % exchange.hashSubRangesPerRange == 0) {
                    wholeSequenceOffsets[hashSubRange / exchange.hashSubRangesPerRange] = std::lower_bound(hashSeqPair + wholeSequenceStart, hashSeqPair + kmerCount, hashSubRangeStart, compareKmerHash) - hashSeqPair;
                }
            }
            hashSubRangeOffsets[exchange.hashSubRangeCount] = wholeSequenceStart;
            wholeSequenceOffsets[exchange.hashRangeCount] = kmerCount;
            size_t keySplitWords = (keySplitOffsets[keySplit + 1] - keySplitOffsets[keySplit] + 63) / 64;
            // count the code size of every row
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
            for (size_t row = 0; row < keySplitRows.size(); row++) {
                unsigned int thread_idx = 0;
#ifdef OPENMP
                thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
                uint64_t *groupBuffer = groupBuffers + thread_idx * groupBufferWords;
                uint64_t *setBitCounts = groupBuffer + (maxKeySplitKeys + 63) / 64;
                size_t hashSubRange = 0;
                size_t hashSubRangeEnd = 0;
                bool isWholeSequence = getRowHashSubRanges(exchange, row / exchange.rowsPerHashRange, row % exchange.rowsPerHashRange, hashSubRange, hashSubRangeEnd);
                size_t from = isWholeSequence ? wholeSequenceOffsets[row / exchange.rowsPerHashRange] : hashSubRangeOffsets[hashSubRange];
                size_t to = isWholeSequence ? wholeSequenceOffsets[row / exchange.rowsPerHashRange + 1] : hashSubRangeOffsets[hashSubRangeEnd];
                unsigned int hashShift = isWholeSequence ? 0 : exchange.kmerHashShift;
                KmerFileRow &hashSubRangeRow = keySplitRows[row];
                KmerFileRow emptyRow = {static_cast<DBKeyType>(keySplitOffsets[keySplit]), idBits, {0, 0, 0, 0}, 0};
                hashSubRangeRow = emptyRow;
                for (size_t pos = from; pos < to; pos++) {
                    hashSubRangeRow.groupCount += (pos == from || hashSeqPair[pos].getKmer() != hashSeqPair[pos - 1].getKmer());
                }
                unsigned int hashDiffLowBits = getHashDiffLowBits(hashSubRange, hashSubRangeEnd, exchange.hashSubRangeCount, hashShift, hashSubRangeRow.groupCount);
                uint64_t prevHash = static_cast<uint64_t>(getHashSubRangeStart(hashSubRange, exchange.hashSubRangeCount, hashShift));
                size_t codeBits[KMER_CODE_COUNT] = {0, 0, 0};
                for (size_t groupStart = from, groupEnd = from; groupStart < to; groupStart = groupEnd) {
                    uint64_t hash = static_cast<uint64_t>(hashSeqPair[groupStart].getKmer()) >> hashShift;
                    size_t repCandidateCount = 1;
                    for (groupEnd = groupStart + 1; groupEnd < to && hashSeqPair[groupEnd].getKmer() == hashSeqPair[groupStart].getKmer(); groupEnd++) {
                        repCandidateCount += (hashSeqPair[groupEnd].getAdjKey() != hashSeqPair[groupEnd - 1].getAdjKey());
                    }
                    size_t memberCount = getGroupIds(hashSeqPair, groupStart, groupEnd, groupBuffer, groupBufferWords, setBitCounts, keySplitWords);
                    codeBits[KMER_MEMBER_CODES] += memberCount * idBits;
                    codeBits[KMER_INDEX_CODES] += (((groupStart == from) ? hash - prevHash : hash - prevHash - 1) >> hashDiffLowBits) + 1 + hashDiffLowBits
                                                  + 2 * getBitWidth(memberCount) - 1;
                    if (isWholeSequence == false) {
                        codeBits[KMER_REP_CANDIDATE_CODES] += repCandidateCount * (exchange.adjKeyCodeBits + getBitWidth(memberCount - 1));
                        codeBits[KMER_INDEX_CODES] += 2 * getBitWidth(repCandidateCount) - 1;
                    }
                    prevHash = hash;
                }
                for (size_t code = 0; code < KMER_CODE_COUNT; code++) {
                    hashSubRangeRow.offsets[code + 1] = hashSubRangeRow.offsets[code] + (codeBits[code] + CHAR_BIT - 1) / CHAR_BIT;
                }
            }
            size_t ownKeySplitLimit = std::min(exchange.keySplitMemoryLimit - ownKmers.size - readBufferBytes - fixedSize - hashSeqPairCapacity * sizeof(PackedKmerPosition),
                                                     ownKmers.limit - std::min(ownKmers.limit, ownKmers.size));
            size_t ownKeySplitSize = 0;
            isOwnHashRange.assign(exchange.hashRangeCount, false);
            for (size_t i = 0; i < ownKmers.hashRanges.size(); i++) {
                size_t hashRange = ownKmers.hashRanges[i];
                size_t hashRangeSize = 0;
                for (size_t row = hashRange * exchange.rowsPerHashRange; row < (hashRange + 1) * exchange.rowsPerHashRange; row++) {
                    hashRangeSize += keySplitRows[row].offsets[KMER_CODE_COUNT];
                }
                if (ownKeySplitSize + hashRangeSize > ownKeySplitLimit) {
                    break;
                }
                isOwnHashRange[hashRange] = true;
                ownKeySplitSize += hashRangeSize;
            }
            std::vector<size_t> flushStarts(kmerGroups.sizes);
            for (size_t row = 0; row < keySplitRows.size(); row++) {
                size_t &size = isOwnHashRange[row / exchange.rowsPerHashRange] ? ownKmers.size : kmerGroups.sizes[row / exchange.rowsPerHashRange / MMseqsNode::count];
                for (size_t code = 0; code <= KMER_CODE_COUNT; code++) {
                    keySplitRows[row].offsets[code] += size;
                }
                size = keySplitRows[row].offsets[KMER_CODE_COUNT];
            }
            std::vector<size_t> flushEnds(kmerGroups.sizes);
            for (size_t hashRange = 0; hashRange < exchange.hashRangeCount; hashRange++) {
                const KmerFileRow *hashRangeRows = keySplitRows.data() + hashRange * exchange.rowsPerHashRange;
                if (isOwnHashRange[hashRange] == false && getRowEnd(hashRangeRows[exchange.rowsPerHashRange - 1]) > hashRangeRows[0].offsets[0]) {
                    flushEnds[hashRange / MMseqsNode::count] += exchange.rowsPerHashRange * sizeof(KmerFileRow);
                }
            }
            fitLocalFileGroups(kmerGroups, flushStarts, flushEnds, batchWriter);
            if (ownKeySplitSize > 0) {
                ownKmers.data = static_cast<char *>(realloc(ownKmers.data, ownKmers.size));
                Util::checkAllocation(ownKmers.data, "Can not allocate ownKmers memory in sendKmers");
            }
            // write the member ids, rep candidates and index codes
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
            for (size_t row = 0; row < keySplitRows.size(); row++) {
                unsigned int thread_idx = 0;
#ifdef OPENMP
                thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
                uint64_t *groupBuffer = groupBuffers + thread_idx * groupBufferWords;
                uint64_t *setBitCounts = groupBuffer + (maxKeySplitKeys + 63) / 64;
                size_t hashSubRange = 0;
                size_t hashSubRangeEnd = 0;
                bool isWholeSequence = getRowHashSubRanges(exchange, row / exchange.rowsPerHashRange, row % exchange.rowsPerHashRange, hashSubRange, hashSubRangeEnd);
                size_t from = isWholeSequence ? wholeSequenceOffsets[row / exchange.rowsPerHashRange] : hashSubRangeOffsets[hashSubRange];
                size_t to = isWholeSequence ? wholeSequenceOffsets[row / exchange.rowsPerHashRange + 1] : hashSubRangeOffsets[hashSubRangeEnd];
                unsigned int hashShift = isWholeSequence ? 0 : exchange.kmerHashShift;
                const KmerFileRow &hashSubRangeRow = keySplitRows[row];
                size_t group = row / exchange.rowsPerHashRange / MMseqsNode::count;
                size_t codeBufferSize = KMER_FILE_BUFFER_SIZE / KMER_CODE_COUNT;
                KmerCodeWriter codeWrites[KMER_CODE_COUNT];
                for (size_t code = 0; code < KMER_CODE_COUNT; code++) {
                    size_t writePos = hashSubRangeRow.offsets[code];
                    codeWrites[code] = isOwnHashRange[row / exchange.rowsPerHashRange]
                        ? KmerCodeWriter(ownKmers.data, ownKmers.size, writePos)
                        : KmerCodeWriter(fileno(kmerGroups.files[group]), kmerGroups.names[group].c_str(), kmerWriteBuffer + thread_idx * KMER_FILE_BUFFER_SIZE + code * codeBufferSize, codeBufferSize, writePos, &batchWriter);
                }
                unsigned int hashDiffLowBits = getHashDiffLowBits(hashSubRange, hashSubRangeEnd, exchange.hashSubRangeCount, hashShift, hashSubRangeRow.groupCount);
                uint64_t prevHash = static_cast<uint64_t>(getHashSubRangeStart(hashSubRange, exchange.hashSubRangeCount, hashShift));
                for (size_t groupStart = from, groupEnd = from; groupStart < to; groupStart = groupEnd) {
                    uint64_t hash = static_cast<uint64_t>(hashSeqPair[groupStart].getKmer()) >> hashShift;
                    size_t repCandidateCount = 1;
                    for (groupEnd = groupStart + 1; groupEnd < to && hashSeqPair[groupEnd].getKmer() == hashSeqPair[groupStart].getKmer(); groupEnd++) {
                        repCandidateCount += (hashSeqPair[groupEnd].getAdjKey() != hashSeqPair[groupEnd - 1].getAdjKey());
                    }
                    size_t memberCount = getGroupIds(hashSeqPair, groupStart, groupEnd, groupBuffer, groupBufferWords, setBitCounts, keySplitWords);
                    if (groupEnd - groupStart <= groupBufferWords) {
                        for (size_t member = 0; member < memberCount; member++) {
                            putBits(codeWrites[KMER_MEMBER_CODES], groupBuffer[member], idBits);
                        }
                    } else {
                        for (size_t word = 0; word < keySplitWords; word++) {
                            for (uint64_t bits = groupBuffer[word]; bits != 0; bits &= bits - 1) {
                                putBits(codeWrites[KMER_MEMBER_CODES], word * 64 + __builtin_ctzll(bits), idBits);
                            }
                        }
                    }
                    // a rep candidate is the first member of its adjKey
                    unsigned int memberIndexBits = getBitWidth(memberCount - 1);
                    for (size_t pos = groupStart; isWholeSequence == false && pos < groupEnd; pos++) {
                        if (pos > groupStart && hashSeqPair[pos].getAdjKey() == hashSeqPair[pos - 1].getAdjKey()) {
                            continue;
                        }
                        DBKeyType id = hashSeqPair[pos].getId();
                        size_t memberIndex = 0;
                        if (groupEnd - groupStart <= groupBufferWords) {
                            memberIndex = std::lower_bound(groupBuffer, groupBuffer + memberCount, id) - groupBuffer;
                        } else {
                            memberIndex = setBitCounts[id / 512] + __builtin_popcountll(groupBuffer[id / 64] & ((1ULL << (id % 64)) - 1));
                            for (size_t word = id / 512 * 8; word < id / 64; word++) {
                                memberIndex += __builtin_popcountll(groupBuffer[word]);
                            }
                        }
                        putAdjKeyCode(codeWrites[KMER_REP_CANDIDATE_CODES], exchange, hashSeqPair[pos].getAdjKey());
                        putBits(codeWrites[KMER_REP_CANDIDATE_CODES], memberIndex, memberIndexBits);
                    }
                    putRice(codeWrites[KMER_INDEX_CODES], (groupStart == from) ? hash - prevHash : hash - prevHash - 1, hashDiffLowBits);
                    putEliasGamma(codeWrites[KMER_INDEX_CODES], memberCount);
                    if (isWholeSequence == false) {
                        putEliasGamma(codeWrites[KMER_INDEX_CODES], repCandidateCount);
                    }
                    prevHash = hash;
                }
                // end every code at a byte boundary
                for (size_t code = 0; code < KMER_CODE_COUNT; code++) {
                    putBits(codeWrites[code], 0, (CHAR_BIT - codeWrites[code].bitCount) % CHAR_BIT);
                    if (codeWrites[code].fd != -1) {
                        writeKmerCodeBuffer(codeWrites[code]);
                    }
                }
            }
            for (size_t hashRange = 0; hashRange < exchange.hashRangeCount; hashRange++) {
                size_t group = hashRange / MMseqsNode::count;
                kmerGroups.flushRowOffsets[group].push_back(kmerGroups.sizes[group]);
                const KmerFileRow *hashRangeRows = keySplitRows.data() + hashRange * exchange.rowsPerHashRange;
                if (getRowEnd(hashRangeRows[exchange.rowsPerHashRange - 1]) > hashRangeRows[0].offsets[0]) {
                    if (isOwnHashRange[hashRange]) {
                        ownKmers.hashRangeRows[hashRange].insert(ownKmers.hashRangeRows[hashRange].end(), hashRangeRows, hashRangeRows + exchange.rowsPerHashRange);
                    } else {
                        batchWriter.write(fileno(kmerGroups.files[group]), kmerGroups.names[group].c_str(), (const char *) hashRangeRows, exchange.rowsPerHashRange * sizeof(KmerFileRow), kmerGroups.sizes[group]);
                        kmerGroups.sizes[group] += exchange.rowsPerHashRange * sizeof(KmerFileRow);
                    }
                }
                if (hashRange + 1 == getFileGroupItemEnd(kmerGroups, group)) {
                    kmerGroups.flushRowOffsets[group].push_back(kmerGroups.sizes[group]);
                }
            }
            kmerGroups.flushCount++;
        }
        delete[] groupBuffers;
    }
    seqDbr.releaseDataPages();
    batchWriter.waitWrites();
    delete[] kmerWriteBuffer;
    Util::releasePages(hashSeqPair, hashSeqPairCapacity * sizeof(PackedKmerPosition));
    delete[] hashSeqPair;
    return closeFileGroups(kmerGroups, std::vector<size_t>());
}

struct ReceivedHashRange {
    size_t hashRange;
    std::vector<ReceivedFile> files;
    std::vector<std::vector<KmerFileRow> > fileRows;
    std::vector<PassRow> rowOrder;
    std::vector<size_t> hashSubRangeReads;
    std::vector<size_t> hashSubRangeSizes;
    std::vector<size_t> passHashSubRangeOffsets;
    std::vector<size_t> passSizes;
    std::vector<size_t> passReceivedSizes;
    size_t maxTaskStarts;
    size_t codeBufferSize;
};

enum { TASK_DONE, TASK_OUTPUT_FULL, TASK_GROUP_TOO_BIG };

static void rewindGroupReads(std::vector<KmerHashSubRangeRead> &reads, const std::vector<unsigned int> &groupRows, size_t code, const std::vector<KmerCodeReader> &groupStarts) {
    for (size_t row = 0; row < groupRows.size(); row++) {
        KmerCodeReader &codeRead = reads[groupRows[row]].codeReads[code];
        if (codeRead.fileOffset != groupStarts[row].fileOffset) {
            refillKmerCodeBuffer(codeRead, groupStarts[row].fileOffset);
        }
        codeRead = groupStarts[row];
    }
}

struct CoverCheck {
    float covThr;
    int covMode;
    float repLength;
    bool isCovered;

    bool operator()(const DBReader<DBKeyType>::Index &range) const {
        return Util::canBeCovered(covThr, covMode, repLength, static_cast<float>(range.length - 2)) == isCovered;
    }
};

static void releaseOwnKmers(OwnKmers &ownKmers, OwnPairs &ownPairs, ReceivedHashRange *receivedHashRanges) {
    Util::releasePages(ownKmers.data, ownKmers.size);
    free(ownKmers.data);
    ownKmers.data = NULL;
    for (size_t i = 0; i < 2; i++) {
        if (receivedHashRanges[i].files.empty() == false) {
            receivedHashRanges[i].files[MMseqsNode::count].data = NULL;
        }
    }
    ownPairs.limit = ownKmers.size;
}

static size_t getTaskCount(const ReceivedHashRange &received, size_t rangeHashSubRange, size_t pass, size_t passLimit, int threads) {
    size_t passReceivedSize = std::max((size_t) 1, received.passReceivedSizes[pass]);
    return (received.passSizes[pass] > passLimit) ? 1 : std::max((size_t) 1, (TASKS_PER_THREAD * threads * received.hashSubRangeSizes[rangeHashSubRange] + passReceivedSize - 1) / passReceivedSize);
}

static bool planHashRange(const Parameters &par, const KmerExchange &exchange, ExchangeFileGroups &kmerGroups, const OwnKmers &ownKmers, size_t passLimit,
                          ReceivedHashRange &received, size_t hashRange, bool wait, size_t &receivedSize) {
    size_t group = hashRange / MMseqsNode::count;
    if (waitForFileGroup(kmerGroups, group, wait) == false) {
        return false;
    }
    received.hashRange = hashRange;
    received.files.assign(MMseqsNode::count + 1, ReceivedFile());
    openFileGroups(kmerGroups, group, received.files, par.threads);
    received.files[MMseqsNode::count].data = ownKmers.data;
    received.fileRows.resize(MMseqsNode::count + 1);
    received.fileRows[MMseqsNode::count] = ownKmers.hashRangeRows[hashRange];
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
    for (size_t sender = 0; sender < MMseqsNode::count; sender++) {
        readItemRows(received.files[sender], hashRange - group * kmerGroups.itemsPerGroup, received.fileRows[sender]);
    }
    received.rowOrder.clear();
    for (size_t sender = 0; sender < received.fileRows.size(); sender++) {
        for (size_t row = 0; row < received.fileRows[sender].size(); row += exchange.rowsPerHashRange) {
            PassRow passRow = {sender, row, 0};
            received.rowOrder.push_back(passRow);
        }
    }
    // key order of the rows gives id order of the members
    SORT_SERIAL(received.rowOrder.begin(), received.rowOrder.end(), [&](const PassRow &first, const PassRow &second) {
        return received.fileRows[first.file][first.row].splitFirstKey < received.fileRows[second.file][second.row].splitFirstKey;
    });
    received.hashSubRangeReads.assign(exchange.rowsPerHashRange, 0);
    std::vector<size_t> hashSubRangeCodeSizes(exchange.rowsPerHashRange, 0);
    received.hashSubRangeSizes.assign(exchange.rowsPerHashRange, 0);
    for (size_t sender = 0; sender < received.fileRows.size(); sender++) {
        const std::vector<KmerFileRow> &rows = received.fileRows[sender];
        for (size_t row = 0; row < rows.size(); row++) {
            size_t rangeHashSubRange = row % exchange.rowsPerHashRange;
            received.hashSubRangeReads[rangeHashSubRange] += (rows[row].groupCount > 0);
            received.hashSubRangeSizes[rangeHashSubRange] += getRowEnd(rows[row]) - rows[row].offsets[0];
            if (received.files[sender].data == NULL) {
                hashSubRangeCodeSizes[rangeHashSubRange] += getRowEnd(rows[row]) - rows[row].offsets[0];
            }
        }
    }
    size_t maxPassSize = planPasses(hashSubRangeCodeSizes, passLimit, received.passHashSubRangeOffsets, received.passSizes);
    bool hasRefilledPass = (*std::max_element(received.passSizes.begin(), received.passSizes.end()) > passLimit);
    received.codeBufferSize = hasRefilledPass ? passLimit : maxPassSize;
    received.passReceivedSizes.assign(received.passSizes.size(), 0);
    received.maxTaskStarts = 0;
    for (size_t pass = 0; pass < received.passSizes.size(); pass++) {
        for (size_t rangeHashSubRange = received.passHashSubRangeOffsets[pass]; rangeHashSubRange < received.passHashSubRangeOffsets[pass + 1]; rangeHashSubRange++) {
            received.passReceivedSizes[pass] += received.hashSubRangeSizes[rangeHashSubRange];
            receivedSize += received.hashSubRangeSizes[rangeHashSubRange];
        }
        size_t passTaskStarts = 0;
        for (size_t rangeHashSubRange = received.passHashSubRangeOffsets[pass]; rangeHashSubRange < received.passHashSubRangeOffsets[pass + 1]; rangeHashSubRange++) {
            passTaskStarts += (getTaskCount(received, rangeHashSubRange, pass, passLimit, par.threads) + 1) * received.hashSubRangeReads[rangeHashSubRange];
        }
        received.maxTaskStarts = std::max(received.maxTaskStarts, passTaskStarts);
    }
    return true;
}

static size_t takeHashRange(const OwnKmers &ownKmers, size_t &ownHashRanges, MMseqsNode::WorkQueue &hashQueue) {
    return (ownHashRanges < ownKmers.hashRanges.size()) ? ownKmers.hashRanges[ownHashRanges++] : hashQueue.takeNextItem();
}

static size_t receiveKmers(Parameters &par, DBReader<DBKeyType> &seqDbr, const KmerExchange &exchange, MMseqsNode::WorkQueue &hashQueue, size_t &receivedHashRangeCount,
                           size_t &pairSize, size_t &pairTargetCount, OwnKmers &ownKmers, OwnPairs &ownPairs, ExchangeFileGroups &kmerGroups, ExchangeFileGroups &pairGroups, size_t receivedFlushCount,
                           std::string &progress) {
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();

    PairFile pairFile;
    createFileGroups(pairGroups);
    pairFile.groups = &pairGroups;
    BatchWriter batchWriter(exchange.writeBehindSize, par.threads, fileno(pairGroups.files[0]));
    pairFile.batchWriter = &batchWriter;
    pairFile.ownPairs = &ownPairs;
    pairFile.partitionSizes.assign(exchange.sendSplitCount * exchange.partitionsPerRange, 0);
    pairFile.targetCount = 0;
    char *pairWriteBuffer = new(std::nothrow) char[par.threads * KMER_FILE_BUFFER_SIZE];
    Util::checkAllocation(pairWriteBuffer, "Can not allocate pairWriteBuffer memory in receiveKmers");
    ReceivedHashRange receivedHashRanges[2];

    std::vector<KmerThreadTaskStart> taskStarts;
    size_t receivedRowsMemory = 2 * receivedFlushCount * (exchange.rowsPerHashRange * sizeof(KmerFileRow) + sizeof(PassRow));
    size_t passLimit = getPassLimit(exchange.keySplitMemoryLimit - std::min(exchange.keySplitMemoryLimit, exchange.pairRowSize + ownKmers.size + receivedRowsMemory));
    size_t outputSize = passLimit;
    size_t halfRecords = outputSize / 2 / sizeof(PairRecord);
    size_t halfTargets = (outputSize - halfRecords * sizeof(PairRecord)) / sizeof(DBKeyType);
    if (halfRecords < exchange.maxRepSeqCount || halfTargets == 0) {
        Debug(Debug::ERROR) << "Not enough memory to receive the k-mers. Raise --split-memory-limit.\n";
        EXIT(EXIT_FAILURE);
    }
    PairOutput output;
    output.size = outputSize;
    output.regionSize = std::min(KMER_FILE_BUFFER_SIZE, outputSize / (16 * par.threads));
    output.recordsTaken = 0;
    output.targetsTaken = 0;
    char *outputBuffer = new(std::nothrow) char[outputSize];
    Util::checkAllocation(outputBuffer, "Can not allocate outputBuffer memory in receiveKmers");
    output.records = (PairRecord *) outputBuffer;
    output.targets = (DBKeyType *) outputBuffer;
    std::vector<PairOutputRegion> outputRegions(par.threads);
    char *passBuffer = NULL;
    size_t codeBufferSize = 0;
    BatchReader *batchReader = NULL;
    std::vector<PassRow> passRows[2];
    std::vector<size_t> passRowOffsets[2];

    size_t receivedSize = 0;

    size_t ownHashRanges = 0;
    size_t hashRange = takeHashRange(ownKmers, ownHashRanges, hashQueue);
    if (hashRange < exchange.hashRangeCount) {
        planHashRange(par, exchange, kmerGroups, ownKmers, passLimit, receivedHashRanges[0], hashRange, true, receivedSize);
    }
    size_t passCodeBuffer = 0;
    bool isPrefetched = false;
    for (size_t taken = 0; hashRange < exchange.hashRangeCount; taken++) {
        receivedHashRangeCount++;
        if (taken == ownKmers.hashRanges.size()) {
            releaseOwnKmers(ownKmers, ownPairs, receivedHashRanges);
        }
        const ReceivedHashRange &received = receivedHashRanges[taken % 2];
        ReceivedHashRange &nextReceived = receivedHashRanges[(taken + 1) % 2];
        size_t passCount = received.passSizes.size();
        if (passBuffer == NULL || received.codeBufferSize > codeBufferSize) {
            Util::releasePages(passBuffer, 2 * codeBufferSize);
            delete[] passBuffer;
            codeBufferSize = received.codeBufferSize;
            passBuffer = new(std::nothrow) char[2 * codeBufferSize];
            Util::checkAllocation(passBuffer, "Can not allocate passBuffer memory in receiveKmers");
        }
        taskStarts.reserve(received.maxTaskStarts);
        size_t nextHashRange = exchange.hashRangeCount;
        bool isNextPlanned = false;
        for (size_t pass = 0; pass < passCount; pass++) {
            size_t passFirstHashSubRange = received.passHashSubRangeOffsets[pass];
            size_t passHashSubRanges = received.passHashSubRangeOffsets[pass + 1] - passFirstHashSubRange;
            bool isRefilledPass = (received.passSizes[pass] > passLimit);
            char *codeBuffer = passBuffer + passCodeBuffer * codeBufferSize;
            size_t codeReadBufferSize = 0;
            if (isRefilledPass) {
                codeBuffer = passBuffer;
                codeReadBufferSize = 2 * passLimit / (received.hashSubRangeReads[passFirstHashSubRange] * KMER_CODE_COUNT);
                if (codeReadBufferSize == 0) {
                    Debug(Debug::ERROR) << "Not enough memory to receive the k-mers. Raise --split-memory-limit.\n";
                    EXIT(EXIT_FAILURE);
                }
                getPassRows(received.fileRows, received.rowOrder, passFirstHashSubRange, received.passHashSubRangeOffsets[pass + 1], passRows[passCodeBuffer], passRowOffsets[passCodeBuffer]);
            } else if (isPrefetched == false) {
                getPassRows(received.fileRows, received.rowOrder, passFirstHashSubRange, received.passHashSubRangeOffsets[pass + 1], passRows[passCodeBuffer], passRowOffsets[passCodeBuffer]);
                readPassRows(received.files, received.fileRows, passRows[passCodeBuffer], passBuffer + passCodeBuffer * codeBufferSize, false, batchReader, par.threads);
            } else {
                batchReader->waitAsyncRead();
            }
            if (pass + 1 == passCount) {
                nextHashRange = takeHashRange(ownKmers, ownHashRanges, hashQueue);
                isNextPlanned = nextHashRange < exchange.hashRangeCount && planHashRange(par, exchange, kmerGroups, ownKmers, passLimit, nextReceived, nextHashRange, false, receivedSize);
            }
            const ReceivedHashRange *prefetched = (pass + 1 < passCount) ? &received : (isNextPlanned ? &nextReceived : NULL);
            size_t prefetchedPass = (pass + 1 < passCount) ? pass + 1 : 0;
            isPrefetched = isRefilledPass == false && prefetched != NULL && prefetched->codeBufferSize <= codeBufferSize
                           && prefetched->passSizes[prefetchedPass] <= passLimit && prefetched->passSizes[prefetchedPass] > 0;
            if (isPrefetched) {
                getPassRows(prefetched->fileRows, prefetched->rowOrder, prefetched->passHashSubRangeOffsets[prefetchedPass], prefetched->passHashSubRangeOffsets[prefetchedPass + 1],
                            passRows[1 - passCodeBuffer], passRowOffsets[1 - passCodeBuffer]);
                readPassRows(prefetched->files, prefetched->fileRows, passRows[1 - passCodeBuffer], passBuffer + (1 - passCodeBuffer) * codeBufferSize, true, batchReader, par.threads);
            }
            const std::vector<PassRow> &hashSubRangeRows = passRows[passCodeBuffer];
            const std::vector<size_t> &hashSubRangeRowOffsets = passRowOffsets[passCodeBuffer];
            if (isRefilledPass == false) {
                passCodeBuffer = 1 - passCodeBuffer;
            }
            std::vector<size_t> hashSubRangeTaskOffsets(passHashSubRanges + 1, 0);
            std::vector<size_t> taskStartOffsets(passHashSubRanges + 1, 0);
            std::vector<size_t> taskHashSubRanges;
            for (size_t passHashSubRange = 0; passHashSubRange < passHashSubRanges; passHashSubRange++) {
                size_t taskCount = getTaskCount(received, passFirstHashSubRange + passHashSubRange, pass, passLimit, par.threads);
                hashSubRangeTaskOffsets[passHashSubRange + 1] = hashSubRangeTaskOffsets[passHashSubRange] + taskCount;
                taskStartOffsets[passHashSubRange + 1] = taskStartOffsets[passHashSubRange]
                                                          + (taskCount + 1) * (hashSubRangeRowOffsets[passHashSubRange + 1] - hashSubRangeRowOffsets[passHashSubRange]);
                taskHashSubRanges.insert(taskHashSubRanges.end(), taskCount, passHashSubRange);
            }
            size_t passTaskCount = taskHashSubRanges.size();
            taskStarts.resize(taskStartOffsets[passHashSubRanges]);
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
            for (size_t passRow = 0; passRow < hashSubRangeRows.size(); passRow++) {
                size_t passHashSubRange = std::upper_bound(hashSubRangeRowOffsets.begin(), hashSubRangeRowOffsets.end(), passRow) - hashSubRangeRowOffsets.begin() - 1;
                size_t taskCount = hashSubRangeTaskOffsets[passHashSubRange + 1] - hashSubRangeTaskOffsets[passHashSubRange];
                size_t rowCount = hashSubRangeRowOffsets[passHashSubRange + 1] - hashSubRangeRowOffsets[passHashSubRange];
                KmerThreadTaskStart *rowStarts = taskStarts.data() + taskStartOffsets[passHashSubRange] + passRow - hashSubRangeRowOffsets[passHashSubRange];
                const ReceivedFile &receivedFile = received.files[hashSubRangeRows[passRow].file];
                const KmerFileRow &hashSubRangeRow = received.fileRows[hashSubRangeRows[passRow].file][hashSubRangeRows[passRow].row];
                size_t hashSubRange = 0;
                size_t hashSubRangeEnd = 0;
                bool isWholeSequence = getRowHashSubRanges(exchange, received.hashRange, passFirstHashSubRange + passHashSubRange, hashSubRange, hashSubRangeEnd);
                unsigned int hashShift = isWholeSequence ? 0 : exchange.kmerHashShift;
                PackedKmerPosition::uint128 hashStart = getHashSubRangeStart(hashSubRange, exchange.hashSubRangeCount, hashShift);
                PackedKmerPosition::uint128 hashWidth = getHashSubRangeStart(hashSubRangeEnd, exchange.hashSubRangeCount, hashShift) - hashStart;
                KmerThreadTaskStart start = {0, static_cast<uint64_t>(hashStart), {0, 0, 0}};
                rowStarts[0] = start;
                if (taskCount > 1) {
                    KmerCodeReader indexRead = getRowCodeReader(receivedFile, hashSubRangeRow, hashSubRangeRows[passRow].bufferOffset, KMER_INDEX_CODES, codeBuffer, NULL, 0);
                    unsigned int hashDiffLowBits = getHashDiffLowBits(hashSubRange, hashSubRangeEnd, exchange.hashSubRangeCount, hashShift, hashSubRangeRow.groupCount);
                    size_t task = 1;
                    for (; start.group < hashSubRangeRow.groupCount; start.group++) {
                        start.bitPos[KMER_INDEX_CODES] = getKmerCodeBitPos(indexRead);
                        uint64_t hash = start.hash + getRice(indexRead, hashDiffLowBits) + ((start.group > 0) ? 1 : 0);
                        for (; task < taskCount && hash >= hashStart + hashWidth * task / taskCount; task++) {
                            rowStarts[task * rowCount] = start;
                        }
                        size_t memberCount = getEliasGamma(indexRead);
                        size_t repCandidateCount = isWholeSequence ? 0 : getEliasGamma(indexRead);
                        start.bitPos[KMER_MEMBER_CODES] += memberCount * hashSubRangeRow.idBits;
                        start.bitPos[KMER_REP_CANDIDATE_CODES] += repCandidateCount * (exchange.adjKeyCodeBits + getBitWidth(memberCount - 1));
                        start.hash = hash;
                    }
                    start.bitPos[KMER_INDEX_CODES] = getKmerCodeBitPos(indexRead);
                    for (; task < taskCount; task++) {
                        rowStarts[task * rowCount] = start;
                    }
                } else {
                    for (size_t code = 0; code < KMER_CODE_COUNT; code++) {
                        start.bitPos[code] = (hashSubRangeRow.offsets[code + 1] - hashSubRangeRow.offsets[code]) * CHAR_BIT;
                    }
                }
                start.group = hashSubRangeRow.groupCount;
                rowStarts[taskCount * rowCount] = start;
            }
            // take the largest tasks first
            std::vector<size_t> taskBits(passTaskCount, 0);
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
            for (size_t task = 0; task < passTaskCount; task++) {
                size_t passHashSubRange = taskHashSubRanges[task];
                size_t rowCount = hashSubRangeRowOffsets[passHashSubRange + 1] - hashSubRangeRowOffsets[passHashSubRange];
                const KmerThreadTaskStart *rowStarts = taskStarts.data() + taskStartOffsets[passHashSubRange] + (task - hashSubRangeTaskOffsets[passHashSubRange]) * rowCount;
                for (size_t row = 0; row < rowCount; row++) {
                    for (size_t code = 0; code < KMER_CODE_COUNT; code++) {
                        taskBits[task] += rowStarts[rowCount + row].bitPos[code] - rowStarts[row].bitPos[code];
                    }
                }
            }
            std::vector<size_t> roundTasks(passTaskCount);
            for (size_t task = 0; task < passTaskCount; task++) {
                roundTasks[task] = task;
            }
            SORT_SERIAL(roundTasks.begin(), roundTasks.end(), [&](size_t first, size_t second) {
                return taskBits[first] > taskBits[second] || (taskBits[first] == taskBits[second] && first < second);
            });
            std::vector<std::vector<KmerThreadTaskStart> > taskResumes(passTaskCount);
            std::vector<char> taskStates(passTaskCount, TASK_DONE);
            std::vector<size_t> tooBigTasks;
            for (bool isSerial = false; roundTasks.empty() == false;) {
#pragma omp parallel num_threads(isSerial ? 1 : par.threads)
                {
                    unsigned int thread_idx = 0;
#ifdef OPENMP
                    thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
                    PairOutputRegion &region = outputRegions[thread_idx];
                    std::vector<KmerHashSubRangeRead> reads;
                    std::vector<unsigned int> groupRows;
                    std::vector<KmerCodeReader> groupRepCandidateStarts;
                    std::vector<KmerCodeReader> groupMemberStarts;
                    std::vector<size_t> repCandidatesLeft;
                    std::vector<std::pair<size_t, size_t> > repMembers;
                    typedef std::pair<unsigned int, std::pair<size_t, size_t> > GroupRepCandidate;
                    std::vector<GroupRepCandidate> repCandidates;
                    repCandidates.reserve(KMER_FILE_BUFFER_SIZE / sizeof(GroupRepCandidate));
                    std::vector<DBKeyType> repSeqIds;
                    std::vector<std::pair<DBKeyType, DBKeyType> > repCoveredKeys;
                    RowValueQueue adjKeyQueue;
#pragma omp for schedule(dynamic, 1)
                    for (size_t i = 0; i < roundTasks.size(); i++) {
                        size_t task = roundTasks[i];
                        taskStates[task] = TASK_DONE;
                        size_t passHashSubRange = taskHashSubRanges[task];
                        size_t hashSubRange = 0;
                        size_t hashSubRangeEnd = 0;
                        bool isWholeSequence = getRowHashSubRanges(exchange, received.hashRange, passFirstHashSubRange + passHashSubRange, hashSubRange, hashSubRangeEnd);
                        unsigned int hashShift = isWholeSequence ? 0 : exchange.kmerHashShift;
                        size_t groupMaxRepSeqCount = isWholeSequence ? 1 : exchange.maxRepSeqCount;
                        size_t rowCount = hashSubRangeRowOffsets[passHashSubRange + 1] - hashSubRangeRowOffsets[passHashSubRange];
                        const KmerThreadTaskStart *rowStarts = taskStarts.data() + taskStartOffsets[passHashSubRange] + (task - hashSubRangeTaskOffsets[passHashSubRange]) * rowCount;
                        const KmerThreadTaskStart *rowResumes = taskResumes[task].empty() ? rowStarts : taskResumes[task].data();
                        reads.clear();
                        for (size_t row = 0; row < rowCount; row++) {
                            const KmerThreadTaskStart &start = rowResumes[row];
                            const KmerThreadTaskStart &end = rowStarts[rowCount + row];
                            if (start.group == end.group) {
                                continue;
                            }
                            const PassRow &passRow = hashSubRangeRows[hashSubRangeRowOffsets[passHashSubRange] + row];
                            const KmerFileRow &hashSubRangeRow = received.fileRows[passRow.file][passRow.row];
                            KmerHashSubRangeRead read;
                            for (size_t code = 0; code < KMER_CODE_COUNT; code++) {
                                read.codeReads[code] = getRowCodeReader(received.files[passRow.file], hashSubRangeRow, passRow.bufferOffset, code, codeBuffer,
                                                                        codeBuffer + (reads.size() * KMER_CODE_COUNT + code) * codeReadBufferSize, codeReadBufferSize);
                                read.codeStartBitPos[code] = getKmerCodeBitPos(read.codeReads[code]);
                                if (start.bitPos[code] > 0) {
                                    seekKmerCodeBits(read.codeReads[code], read.codeStartBitPos[code] + start.bitPos[code]);
                                }
                            }
                            read.hashSubRangeRow = &hashSubRangeRow;
                            read.taskRow = row;
                            read.isWholeSequence = isWholeSequence;
                            read.hashDiffLowBits = getHashDiffLowBits(hashSubRange, hashSubRangeEnd, exchange.hashSubRangeCount, hashShift, hashSubRangeRow.groupCount);
                            read.group = start.group;
                            read.groupEnd = end.group;
                            read.hash = start.hash;
                            reads.push_back(read);
                        }

                        // merge the groups of the rows by hash
                        RowValueQueue kmerHashQueue;
                        for (size_t row = 0; row < reads.size(); row++) {
                            queueNextKmerGroup(kmerHashQueue, row, reads[row]);
                        }
                        while (kmerHashQueue.empty() == false) {
                            uint64_t hash = kmerHashQueue.top().value;
                            groupRows.clear();
                            size_t memberCount = 0;
                            while (kmerHashQueue.empty() == false && kmerHashQueue.top().value == hash) {
                                groupRows.push_back(kmerHashQueue.top().row);
                                memberCount += reads[kmerHashQueue.top().row].groupMemberCount;
                                kmerHashQueue.pop();
                            }
                            if (memberCount < 2) {
                                KmerHashSubRangeRead &read = reads[groupRows[0]];
                                KmerCodeReader &memberRead = read.codeReads[KMER_MEMBER_CODES];
                                KmerCodeReader &repCandidateRead = read.codeReads[KMER_REP_CANDIDATE_CODES];
                                size_t memberBits = read.groupMemberCount * read.hashSubRangeRow->idBits;
                                size_t repCandidateBits = read.groupRepCandidateCount * (exchange.adjKeyCodeBits + getBitWidth(read.groupMemberCount - 1));
                                seekKmerCodeBits(memberRead, getKmerCodeBitPos(memberRead) + memberBits);
                                seekKmerCodeBits(repCandidateRead, getKmerCodeBitPos(repCandidateRead) + repCandidateBits);
                            } else {
                                if (takePairOutput(output, region, groupMaxRepSeqCount, memberCount) == false) {
                                    if (isSerial == false) {
                                        taskStates[task] = (groupMaxRepSeqCount * sizeof(PairRecord) + memberCount * sizeof(DBKeyType) > output.size) ? TASK_GROUP_TOO_BIG : TASK_OUTPUT_FULL;
                                        break;
                                    }
                                    flushPairOutput(exchange, pairFile, output, outputRegions, pairWriteBuffer, 1);
                                    takePairOutput(output, region, std::min(groupMaxRepSeqCount, halfRecords), std::min(memberCount, halfTargets));
                                }
                                groupRepCandidateStarts.clear();
                                groupMemberStarts.clear();
                                for (size_t row = 0; row < groupRows.size(); row++) {
                                    groupRepCandidateStarts.push_back(reads[groupRows[row]].codeReads[KMER_REP_CANDIDATE_CODES]);
                                    groupMemberStarts.push_back(reads[groupRows[row]].codeReads[KMER_MEMBER_CODES]);
                                }
                                repMembers.clear();
                                if (isWholeSequence) {
                                    repMembers.push_back(std::make_pair(0, 0));
                                } else {
                                    size_t groupRepCandidateCount = 0;
                                    for (size_t row = 0; row < groupRows.size(); row++) {
                                        groupRepCandidateCount += reads[groupRows[row]].groupRepCandidateCount;
                                    }
                                    if (groupRepCandidateCount <= KMER_FILE_BUFFER_SIZE / sizeof(GroupRepCandidate)) {
                                        repCandidates.clear();
                                        rewindGroupReads(reads, groupRows, KMER_REP_CANDIDATE_CODES, groupRepCandidateStarts);
                                        for (size_t row = 0; row < groupRows.size(); row++) {
                                            KmerHashSubRangeRead &read = reads[groupRows[row]];
                                            unsigned int memberIndexBits = getBitWidth(read.groupMemberCount - 1);
                                            for (size_t repCandidate = 0; repCandidate < read.groupRepCandidateCount; repCandidate++) {
                                                unsigned int adjKey = getAdjKeyCode(read.codeReads[KMER_REP_CANDIDATE_CODES], exchange);
                                                repCandidates.push_back(GroupRepCandidate(adjKey, std::pair<size_t, size_t>(row, getBits(read.codeReads[KMER_REP_CANDIDATE_CODES], memberIndexBits))));
                                            }
                                        }
                                        // one row lists its candidates in adjKey order
                                        if (groupRows.size() > 1) {
                                            SORT_SERIAL(repCandidates.begin(), repCandidates.end());
                                        }
                                        selectGroupRepSeqs(repMembers, groupMaxRepSeqCount, [&](const auto &visit) {
                                            for (size_t repCandidate = 0; repCandidate < repCandidates.size(); repCandidate++) {
                                                visit(repCandidates[repCandidate].first, repCandidates[repCandidate].second);
                                            }
                                        }, std::less<std::pair<size_t, size_t> >());
                                    } else {
                                        selectGroupRepSeqs(repMembers, groupMaxRepSeqCount, [&](const auto &visit) {
                                            rewindGroupReads(reads, groupRows, KMER_REP_CANDIDATE_CODES, groupRepCandidateStarts);
                                            if (groupRows.size() == 1) {
                                                KmerHashSubRangeRead &read = reads[groupRows[0]];
                                                unsigned int memberIndexBits = getBitWidth(read.groupMemberCount - 1);
                                                for (size_t repCandidate = 0; repCandidate < read.groupRepCandidateCount; repCandidate++) {
                                                    unsigned int adjKey = getAdjKeyCode(read.codeReads[KMER_REP_CANDIDATE_CODES], exchange);
                                                    visit(adjKey, std::pair<size_t, size_t>(0, getBits(read.codeReads[KMER_REP_CANDIDATE_CODES], memberIndexBits)));
                                                }
                                            } else {
                                                // merge the rows by adjKey
                                                repCandidatesLeft.resize(groupRows.size());
                                                for (size_t row = 0; row < groupRows.size(); row++) {
                                                    KmerHashSubRangeRead &read = reads[groupRows[row]];
                                                    repCandidatesLeft[row] = read.groupRepCandidateCount - 1;
                                                    adjKeyQueue.push(RowValue(getAdjKeyCode(read.codeReads[KMER_REP_CANDIDATE_CODES], exchange), row));
                                                }
                                                while (adjKeyQueue.empty() == false) {
                                                    RowValue repCandidate = adjKeyQueue.top();
                                                    adjKeyQueue.pop();
                                                    KmerHashSubRangeRead &read = reads[groupRows[repCandidate.row]];
                                                    size_t memberIndex = getBits(read.codeReads[KMER_REP_CANDIDATE_CODES], getBitWidth(read.groupMemberCount - 1));
                                                    visit(repCandidate.value, std::pair<size_t, size_t>(repCandidate.row, memberIndex));
                                                    if (repCandidatesLeft[repCandidate.row] > 0) {
                                                        repCandidatesLeft[repCandidate.row]--;
                                                        adjKeyQueue.push(RowValue(getAdjKeyCode(read.codeReads[KMER_REP_CANDIDATE_CODES], exchange), repCandidate.row));
                                                    }
                                                }
                                            }
                                        }, std::less<std::pair<size_t, size_t> >());
                                    }
                                }
                                repSeqIds.clear();
                                for (size_t rep = 0; rep < repMembers.size(); rep++) {
                                    KmerHashSubRangeRead &read = reads[groupRows[repMembers[rep].first]];
                                    KmerCodeReader &memberRead = read.codeReads[KMER_MEMBER_CODES];
                                    seekKmerCodeBits(memberRead, getKmerCodeBitPos(groupMemberStarts[repMembers[rep].first]) + repMembers[rep].second * read.hashSubRangeRow->idBits);
                                    repSeqIds.push_back(getBits(memberRead, read.hashSubRangeRow->idBits) + read.hashSubRangeRow->splitFirstKey);
                                }
                                repCoveredKeys.clear();
                                for (size_t rep = 0; rep < repSeqIds.size(); rep++) {
                                    size_t repRange = seqDbr.findIdRange(repSeqIds[rep]);
                                    float repLength = static_cast<float>(index[repRange].length - 2);
                                    // ranges are in length descending order
                                    CoverCheck isNotCovered = { par.covThr, par.covMode, repLength, false };
                                    CoverCheck isCovered = { par.covThr, par.covMode, repLength, true };
                                    const DBReader<DBKeyType>::Index *coveredFirstRange = std::partition_point(index, index + repRange, isNotCovered);
                                    const DBReader<DBKeyType>::Index *coveredEndRange = std::partition_point(index + repRange, index + seqDbr.getIndexSize(), isCovered);
                                    DBKeyType coveredTo = (coveredEndRange < index + seqDbr.getIndexSize()) ? coveredEndRange->id : static_cast<DBKeyType>(seqDbr.getSize());
                                    repCoveredKeys.push_back(std::make_pair(coveredFirstRange->id, coveredTo));
                                }
                                rewindGroupReads(reads, groupRows, KMER_MEMBER_CODES, groupMemberStarts);
                                for (size_t row = 0, member = 0; row < groupRows.size();) {
                                    if (region.recordEnd - region.recordPos < repSeqIds.size() || region.targetPos == region.targetEnd) {
                                        flushPairOutput(exchange, pairFile, output, outputRegions, pairWriteBuffer, 1);
                                        takePairOutput(output, region, halfRecords, halfTargets);
                                    }
                                    const DBKeyType *members = output.targets + region.targetPos;
                                    while (row < groupRows.size() && region.targetPos < region.targetEnd) {
                                        KmerHashSubRangeRead &read = reads[groupRows[row]];
                                        output.targets[region.targetPos++] = getBits(read.codeReads[KMER_MEMBER_CODES], read.hashSubRangeRow->idBits) + read.hashSubRangeRow->splitFirstKey;
                                        if (++member == read.groupMemberCount) {
                                            row++;
                                            member = 0;
                                        }
                                    }
                                    const DBKeyType *membersEnd = output.targets + region.targetPos;
                                    for (size_t rep = 0; rep < repSeqIds.size(); rep++) {
                                        const DBKeyType *covered = std::lower_bound(members, membersEnd, repCoveredKeys[rep].first);
                                        const DBKeyType *coveredEnd = std::lower_bound(covered, membersEnd, repCoveredKeys[rep].second);
                                        if (coveredEnd - covered > 1 || (coveredEnd - covered == 1 && *covered != repSeqIds[rep])) {
                                            PairRecord record = {repSeqIds[rep], static_cast<size_t>(covered - output.targets), static_cast<size_t>(coveredEnd - covered)};
                                            output.records[region.recordPos++] = record;
                                        }
                                    }
                                }
                            }
                            for (size_t row = 0; row < groupRows.size(); row++) {
                                queueNextKmerGroup(kmerHashQueue, groupRows[row], reads[groupRows[row]]);
                            }
                        }
                        // resume a stopped task where it stopped
                        if (taskStates[task] != TASK_DONE) {
                            if (taskResumes[task].empty()) {
                                taskResumes[task].assign(rowStarts, rowStarts + rowCount);
                            }
                            for (size_t row = 0; row < reads.size(); row++) {
                                const KmerHashSubRangeRead &read = reads[row];
                                KmerThreadTaskStart &resume = taskResumes[task][read.taskRow];
                                resume.group = read.pendingGroup;
                                resume.hash = read.pendingHash;
                                resume.bitPos[KMER_INDEX_CODES] = read.pendingIndexBitPos - read.codeStartBitPos[KMER_INDEX_CODES];
                                resume.bitPos[KMER_MEMBER_CODES] = getKmerCodeBitPos(read.codeReads[KMER_MEMBER_CODES]) - read.codeStartBitPos[KMER_MEMBER_CODES];
                                resume.bitPos[KMER_REP_CANDIDATE_CODES] = getKmerCodeBitPos(read.codeReads[KMER_REP_CANDIDATE_CODES]) - read.codeStartBitPos[KMER_REP_CANDIDATE_CODES];
                            }
                        }
                    }
                }
                std::vector<size_t> fullTasks;
                for (size_t i = 0; i < roundTasks.size(); i++) {
                    if (taskStates[roundTasks[i]] == TASK_OUTPUT_FULL) {
                        fullTasks.push_back(roundTasks[i]);
                    } else if (taskStates[roundTasks[i]] == TASK_GROUP_TOO_BIG) {
                        tooBigTasks.push_back(roundTasks[i]);
                    }
                }
                isSerial = fullTasks.empty();
                if (isSerial == false || tooBigTasks.empty() == false) {
                    flushPairOutput(exchange, pairFile, output, outputRegions, pairWriteBuffer, par.threads);
                }
                roundTasks.swap(isSerial ? tooBigTasks : fullTasks);
                if (isSerial) {
                    tooBigTasks.clear();
                }
            }
        }
        std::vector<PassRow> hashRangeRows;
        std::vector<size_t> hashRangeRowOffsets;
        getPassRows(received.fileRows, received.rowOrder, 0, exchange.rowsPerHashRange, hashRangeRows, hashRangeRowOffsets);
        punchPassRows(received.files, received.fileRows, hashRangeRows, ownPairs.limit);
        closeReceivedFiles(receivedHashRanges[taken % 2].files);
        addFinishedItems(par, progress, KMER_FILE_GROUPS, hashRange / MMseqsNode::count, 1);
        hashRange = nextHashRange;
        if (hashRange < exchange.hashRangeCount && isNextPlanned == false) {
            planHashRange(par, exchange, kmerGroups, ownKmers, passLimit, nextReceived, hashRange, true, receivedSize);
        }
    }
    releaseOwnKmers(ownKmers, ownPairs, receivedHashRanges);
    flushPairOutput(exchange, pairFile, output, outputRegions, pairWriteBuffer, par.threads);
    delete batchReader;
    batchWriter.waitWrites();
    Util::releasePages(outputBuffer, outputSize);
    delete[] outputBuffer;
    Util::releasePages(passBuffer, 2 * codeBufferSize);
    delete[] passBuffer;
    delete[] pairWriteBuffer;
    pairSize = closeFileGroups(pairGroups, pairFile.partitionSizes);
    pairTargetCount = pairFile.targetCount;
    return receivedSize;
}

static void addKeyTaskCuts(std::vector<PartitionCut> &partitionCuts, size_t partitionFirstKey, size_t taskFirstKey, size_t taskPrefOffset, size_t taskCost, const KeyTaskCuts &taskCuts) {
    PartitionCut taskStart = { partitionFirstKey + taskFirstKey, taskPrefOffset, taskCost };
    partitionCuts.push_back(taskStart);
    for (size_t i = 0; i < taskCuts.cuts.size(); i++) {
        PartitionCut cut = { partitionFirstKey + taskCuts.cuts[i].key, taskPrefOffset + taskCuts.cuts[i].prefOffset, taskCost + taskCuts.cuts[i].cost };
        partitionCuts.push_back(cut);
    }
}

static void addPartitionEnd(std::vector<PartitionCut> &partitionCuts, std::vector<size_t> &partitionCutOffsets, PackedKmerPosition::uint128 &mergedCost, size_t prefEnd, size_t cost) {
    PartitionCut end = { 0, prefEnd, cost };
    partitionCuts.push_back(end);
    partitionCutOffsets.push_back(partitionCuts.size());
    mergedCost += cost;
}

static std::string mergePairFiles(Parameters &par, DBReader<DBKeyType> &seqDbr, const KmerExchange &exchange, MMseqsNode::WorkQueue &mergeQueue, size_t &mergedSplitCount,
                                  OwnPairs &ownPairs, ExchangeFileGroups &pairGroups, std::string &progress, std::vector<size_t> &targetSplitBlockOffsets, size_t &prefSize) {
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    std::vector<ReceivedFile> receivedPairFiles(MMseqsNode::count + 1);
    receivedPairFiles[MMseqsNode::count].data = ownPairs.data;
    std::vector<std::vector<PairFileRow> > pairFileRows(MMseqsNode::count + 1);
    size_t partitionCount = exchange.sendSplitCount * exchange.partitionsPerRange;
    std::vector<size_t> partitionPairSizes(partitionCount, 0);
    for (unsigned int receiver = 0; receiver < MMseqsNode::count; receiver++) {
        for (size_t group = 0; group < pairGroups.count; group++) {
            while (pairGroups.nodeFileSizes[receiver][group] == SIZE_MAX) {
                readFileGroupIndex(pairGroups, receiver, true, &partitionPairSizes);
            }
        }
    }
    PrefFiles prefFiles;
    TargetSplitFiles targetSplitFiles;
    if (exchange.hasTargetSplits) {
        // runs of whole blocks, fewer while a rep has few targets
        size_t representativeCount = 0;
        for (size_t range = 0; range < seqDbr.getIndexSize(); range++) {
            representativeCount += seqDbr.getRangeRepresentativeCount(range);
        }
        const size_t alignNodeCount = MMseqsNode::count - MMseqsNode::getFirstAlignNode();
        const size_t targetsPerRep = exchange.pairTargetCount / std::max<size_t>(1, representativeCount);
        const size_t blockCount = seqDbr.getBlockKeyOffsets().size() - 1;
        const size_t targetSplitCount = std::min(blockCount, std::max(MIN_TARGET_SPLITS_PER_NODE * alignNodeCount, std::min(MAX_TARGET_SPLITS_PER_NODE * alignNodeCount, targetsPerRep / 4)));
        targetSplitBlockOffsets.resize(targetSplitCount + 1);
        targetSplitFiles.keyOffsets.resize(targetSplitCount + 1);
        for (size_t split = 0; split <= targetSplitCount; split++) {
            targetSplitBlockOffsets[split] = split * blockCount / targetSplitCount;
            targetSplitFiles.keyOffsets[split] = seqDbr.getBlockKeyOffsets()[targetSplitBlockOffsets[split]];
        }
        for (size_t split = 0; split < targetSplitCount; split++) {
            const size_t file = MMseqsNode::rank * targetSplitCount + split;
            targetSplitFiles.dataNames.push_back(par.db2 + "." + SSTR(file));
            targetSplitFiles.indexNames.push_back(par.db2 + ".index." + SSTR(file));
            targetSplitFiles.dataFiles.push_back(FileUtil::openAndDelete(targetSplitFiles.dataNames.back().c_str(), "w"));
        }
        targetSplitFiles.dataSizes.assign(targetSplitCount, 0);
        targetSplitFiles.indexCells.resize(targetSplitCount);
        targetSplitFiles.cellOffsets.assign(targetSplitCount, 0);
        targetSplitFiles.cellCosts.assign(targetSplitCount, 0);
    } else {
        prefFiles.dataName = par.db2 + "." + SSTR(MMseqsNode::rank);
        prefFiles.indexName = par.db2 + ".index." + SSTR(MMseqsNode::rank);
        prefFiles.dataFile = FileUtil::openAndDelete(prefFiles.dataName.c_str(), "w");
        prefFiles.indexFile = FileUtil::openAndDelete(prefFiles.indexName.c_str(), "w");
    }
    BatchWriter batchWriter(exchange.writeBehindSize, par.threads, fileno(exchange.hasTargetSplits ? targetSplitFiles.dataFiles[0] : prefFiles.dataFile));
    prefFiles.batchWriter = &batchWriter;
    size_t prefDataSize = 0;
    size_t prefIndexSize = 0;
    prefFiles.writeBuffers = new(std::nothrow) char[par.threads * 2 * KMER_FILE_BUFFER_SIZE];
    Util::checkAllocation(prefFiles.writeBuffers, "Can not allocate prefWriteBuffer memory in mergePairFiles");
    std::vector<PartitionCut> partitionCuts;
    std::vector<size_t> partitionCutOffsets(1, 0);
    PackedKmerPosition::uint128 mergedCost = 0;
    std::string partitionsData;
    char *pairPassBuffer = NULL;
    size_t pairPassBufferSize = 0;
    DBKeyType *pairPassTargets = NULL;
    size_t pairPassTargetsSize = 0;
    std::vector<size_t> partitionCosts(partitionCount, 0);
    size_t globalCost = 0;
    for (size_t partition = 0; partition < partitionCount; partition++) {
        size_t sendSplit = partition / exchange.partitionsPerRange;
        size_t rangePartition = partition % exchange.partitionsPerRange;
        if (sendSplit / MMseqsNode::count >= exchange.selfMergedSendSplitCount) {
            partitionCosts[partition] = partitionPairSizes[partition] + getPartitionFirstKey(exchange, sendSplit, rangePartition + 1) - getPartitionFirstKey(exchange, sendSplit, rangePartition);
        }
        globalCost += partitionCosts[partition];
    }
    size_t mergeSplitCount = exchange.sendSplitCount;
    size_t mergeSplitCost = std::max((size_t) 1, (globalCost + mergeSplitCount - 1) / mergeSplitCount);
    std::vector<size_t> splitPartitionOffsets(mergeSplitCount + 1, partitionCount);
    splitPartitionOffsets[0] = 0;
    size_t costBefore = 0;
    for (size_t partition = 0, split = 1; partition < partitionCount; partition++) {
        costBefore += partitionCosts[partition];
        for (; split < mergeSplitCount && costBefore >= split * mergeSplitCost; split++) {
            splitPartitionOffsets[split] = partition + 1;
        }
    }
    std::vector<size_t> splitCosts(mergeSplitCount, 0);
    std::vector<size_t> mergeOrder(mergeSplitCount);
    for (size_t split = 0; split < mergeSplitCount; split++) {
        for (size_t partition = splitPartitionOffsets[split]; partition < splitPartitionOffsets[split + 1]; partition++) {
            splitCosts[split] += partitionCosts[partition];
        }
        mergeOrder[split] = split;
    }
    for (size_t i = 0; i < mergeOrder.size(); i += MMseqsNode::count) {
        std::vector<size_t>::iterator end = mergeOrder.begin() + std::min(i + MMseqsNode::count, mergeOrder.size());
        SORT_SERIAL(mergeOrder.begin() + i, end, [&](size_t first, size_t second) {
            return splitCosts[first] > splitCosts[second] || (splitCosts[first] == splitCosts[second] && first < second);
        });
        if (i / MMseqsNode::count % 2 == 1) {
            std::reverse(mergeOrder.begin() + i, end);
        }
    }
    size_t releasedSize = 0;
    size_t bitsSendSplit = SIZE_MAX;
    for (size_t ownMergeRange = 0;; ownMergeRange++) {
        size_t splitBegin = 0;
        size_t splitEnd = 0;
        // a window is a run of keys that one node merges; the self-merged send splits follow the merge splits
        size_t window = 0;
        size_t windowFirstKey = SIZE_MAX;
        if (ownMergeRange < exchange.selfMergedSendSplitCount) {
            splitBegin = (MMseqsNode::rank + ownMergeRange * MMseqsNode::count) * exchange.partitionsPerRange;
            splitEnd = splitBegin + exchange.partitionsPerRange;
            window = mergeSplitCount + splitBegin / exchange.partitionsPerRange;
        } else {
            if (ownMergeRange == exchange.selfMergedSendSplitCount) {
                Util::releasePages(ownPairs.data, ownPairs.size);
                free(ownPairs.data);
                ownPairs.data = NULL;
                ownPairs.size = 0;
                receivedPairFiles[MMseqsNode::count].data = NULL;
            }
            size_t item = mergeQueue.takeNextItem();
            if (item == mergeSplitCount) {
                break;
            }
            size_t mergeSplit = mergeOrder[item];
            mergedSplitCount++;
            splitBegin = splitPartitionOffsets[mergeSplit];
            splitEnd = splitPartitionOffsets[mergeSplit + 1];
            window = mergeSplit;
        }
        for (size_t pieceBegin = splitBegin, pieceEnd = splitBegin; pieceBegin < splitEnd; pieceBegin = pieceEnd) {
            size_t sendSplit = pieceBegin / exchange.partitionsPerRange;
            size_t fileGroup = pieceBegin / pairGroups.itemsPerGroup;
            pieceEnd = std::min(splitEnd, std::min((sendSplit + 1) * exchange.partitionsPerRange, getFileGroupItemEnd(pairGroups, fileGroup)));
            if (ownMergeRange >= exchange.selfMergedSendSplitCount && sendSplit / MMseqsNode::count < exchange.selfMergedSendSplitCount) {
                if (windowFirstKey != SIZE_MAX) {
                    addTargetSplitRows(targetSplitFiles, windowFirstKey, window, partitionsData);
                    windowFirstKey = SIZE_MAX;
                }
                continue;
            }
            size_t firstPartition = pieceBegin - sendSplit * exchange.partitionsPerRange;
            size_t endPartition = pieceEnd - sendSplit * exchange.partitionsPerRange;
            if (exchange.hasTargetSplits && windowFirstKey == SIZE_MAX) {
                windowFirstKey = getPartitionFirstKey(exchange, sendSplit, firstPartition);
            }
            if (sendSplit != bitsSendSplit) {
                setSendSplit(par, exchange, sendSplit);
                bitsSendSplit = sendSplit;
            }
            waitForFileGroup(pairGroups, fileGroup, true);
            openFileGroups(pairGroups, fileGroup, receivedPairFiles, par.threads);
            std::vector<std::vector<BatchReader::ReadRequest> > rowReadRequests(MMseqsNode::count);
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
            for (size_t receiver = 0; receiver < MMseqsNode::count; receiver++) {
                readItemRows(receivedPairFiles[receiver], sendSplit - fileGroup * pairGroups.itemsPerGroup / exchange.partitionsPerRange, pairFileRows[receiver], &rowReadRequests[receiver]);
            }
            pairFileRows[MMseqsNode::count] = ownPairs.sendSplitRows[sendSplit];
            for (size_t receiver = 1; receiver < rowReadRequests.size(); receiver++) {
                rowReadRequests[0].insert(rowReadRequests[0].end(), rowReadRequests[receiver].begin(), rowReadRequests[receiver].end());
            }
            BatchReader *batchReader = new BatchReader(par.threads, fileno(receivedPairFiles[0].file));
            batchReader->readSync(rowReadRequests[0]);
            std::vector<size_t> partitionCodeSizes(exchange.partitionsPerRange, 0);
            std::vector<size_t> partitionTargets(exchange.partitionsPerRange, 0);
            std::vector<std::vector<std::pair<size_t, size_t> > > partitionRows(exchange.partitionsPerRange);
            std::vector<size_t> partitionRowCounts(exchange.partitionsPerRange, 0);
            for (size_t receiver = 0; receiver < receivedPairFiles.size(); receiver++) {
                const std::vector<PairFileRow> &rows = pairFileRows[receiver];
                for (size_t row = 0; row < rows.size(); row++) {
                    partitionRowCounts[rows[row].partition] += (rows[row].partition >= firstPartition && rows[row].partition < endPartition);
                }
            }
            for (size_t partition = 0; partition < exchange.partitionsPerRange; partition++) {
                partitionRows[partition].reserve(partitionRowCounts[partition]);
            }
            for (size_t receiver = 0; receiver < receivedPairFiles.size(); receiver++) {
                const std::vector<PairFileRow> &rows = pairFileRows[receiver];
                for (size_t row = 0; row < rows.size(); row++) {
                    size_t partition = rows[row].partition;
                    if (partition < firstPartition || partition >= endPartition) {
                        continue;
                    }
                    if (receivedPairFiles[receiver].data == NULL) {
                        partitionCodeSizes[partition] += getRowEnd(rows[row]) - rows[row].offsets[0];
                    }
                    partitionTargets[partition] += rows[row].targetCount;
                    partitionRows[partition].push_back(std::make_pair(receiver, row));
                }
            }
            size_t pairPassLimit = getPassLimit(exchange.keySplitMemoryLimit - ownPairs.size);
            std::vector<size_t> partitionSizes(endPartition - firstPartition);
            for (size_t partition = firstPartition; partition < endPartition; partition++) {
                size_t partitionKeys = getPartitionFirstKey(exchange, sendSplit, partition + 1) - getPartitionFirstKey(exchange, sendSplit, partition);
                partitionSizes[partition - firstPartition] = partitionCodeSizes[partition] + partitionTargets[partition] * sizeof(DBKeyType) + (partitionKeys + 1) * sizeof(size_t)
                                                             + getPartitionMergeSize(partitionRowCounts[partition], 1);
            }
            std::vector<size_t> passPartitionOffsets;
            std::vector<size_t> pairPassSizes;
            planPasses(partitionSizes, pairPassLimit, passPartitionOffsets, pairPassSizes);
            for (size_t pass = 0; pass < passPartitionOffsets.size(); pass++) {
                passPartitionOffsets[pass] += firstPartition;
            }
            size_t pairPassCount = pairPassSizes.size();
            std::vector<size_t> pairPassCodeSizes(pairPassCount, 0);
            size_t maxPairPassCodeSize = 0;
            size_t maxPairPassTargets = 0;
            bool hasRefilledPass = false;
            for (size_t pass = 0; pass < pairPassCount; pass++) {
                size_t pairPassTargetCount = 0;
                for (size_t partition = passPartitionOffsets[pass]; partition < passPartitionOffsets[pass + 1]; partition++) {
                    pairPassCodeSizes[pass] += partitionCodeSizes[partition];
                    pairPassTargetCount += partitionTargets[partition];
                }
                if (pairPassSizes[pass] <= pairPassLimit) {
                    maxPairPassCodeSize = std::max(maxPairPassCodeSize, pairPassCodeSizes[pass]);
                    maxPairPassTargets = std::max(maxPairPassTargets, pairPassTargetCount);
                } else {
                    hasRefilledPass = true;
                }
            }

            size_t neededBufferSize = std::max(2 * maxPairPassCodeSize, hasRefilledPass ? 2 * pairPassLimit : 0);
            if (neededBufferSize > pairPassBufferSize) {
                Util::releasePages(pairPassBuffer, pairPassBufferSize);
                delete[] pairPassBuffer;
                pairPassBufferSize = neededBufferSize;
                pairPassBuffer = new(std::nothrow) char[pairPassBufferSize];
                Util::checkAllocation(pairPassBuffer, "Can not allocate pairPassBuffer memory in mergePairFiles");
            }
            if (maxPairPassTargets > pairPassTargetsSize) {
                Util::releasePages(pairPassTargets, pairPassTargetsSize * sizeof(DBKeyType));
                delete[] pairPassTargets;
                pairPassTargetsSize = maxPairPassTargets;
                pairPassTargets = new(std::nothrow) DBKeyType[pairPassTargetsSize];
                Util::checkAllocation(pairPassTargets, "Can not allocate pairPassTargets memory in mergePairFiles");
            }
            std::vector<PassRow> pairPassRows[2];
            std::vector<size_t> pairPassRowOffsets[2];
            std::vector<PartitionMerge> partitionMerges;
            bool isPrefetched = false;
            for (size_t pass = 0; pass < pairPassCount; pass++) {
                size_t passFirstPartition = passPartitionOffsets[pass];
                size_t passPartitions = passPartitionOffsets[pass + 1] - passFirstPartition;
                bool isRefilledPass = (pairPassSizes[pass] > pairPassLimit);
                char *codeBuffer = pairPassBuffer + (pass % 2) * maxPairPassCodeSize;
                if (isRefilledPass) {
                    codeBuffer = pairPassBuffer;
                    getPairPassRows(partitionRows, passFirstPartition, passPartitionOffsets[pass + 1], pairPassRows[pass % 2], pairPassRowOffsets[pass % 2]);
                } else {
                    if (isPrefetched == false) {
                        getPairPassRows(partitionRows, passFirstPartition, passPartitionOffsets[pass + 1], pairPassRows[pass % 2], pairPassRowOffsets[pass % 2]);
                        readPassRows(receivedPairFiles, pairFileRows, pairPassRows[pass % 2], codeBuffer, false, batchReader, par.threads);
                    } else {
                        batchReader->waitAsyncRead();
                    }
                    isPrefetched = pass + 1 < pairPassCount && pairPassSizes[pass + 1] <= pairPassLimit && pairPassCodeSizes[pass + 1] > 0;
                    if (isPrefetched) {
                        getPairPassRows(partitionRows, passPartitionOffsets[pass + 1], passPartitionOffsets[pass + 2], pairPassRows[(pass + 1) % 2], pairPassRowOffsets[(pass + 1) % 2]);
                        readPassRows(receivedPairFiles, pairFileRows, pairPassRows[(pass + 1) % 2], pairPassBuffer + ((pass + 1) % 2) * maxPairPassCodeSize, true, batchReader, par.threads);
                    }
                }
                const std::vector<PassRow> &partitionPassRows = pairPassRows[pass % 2];
                const std::vector<size_t> &partitionPassRowOffsets = pairPassRowOffsets[pass % 2];
                if (passPartitions >= 2 * static_cast<size_t>(par.threads)) {
                    // many partitions: a thread merges a partition
                    std::vector<size_t> partitionOrder(passPartitions);
                    for (size_t passPartition = 0; passPartition < passPartitions; passPartition++) {
                        partitionOrder[passPartition] = passPartition;
                    }
                    SORT_SERIAL(partitionOrder.begin(), partitionOrder.end(), [&](size_t first, size_t second) {
                        return partitionTargets[passFirstPartition + first] > partitionTargets[passFirstPartition + second]
                               || (partitionTargets[passFirstPartition + first] == partitionTargets[passFirstPartition + second] && first < second);
                    });
                    partitionMerges.resize(passPartitions);
                    std::vector<size_t> partitionTargetOffsets(passPartitions + 1, 0);
                    for (size_t passPartition = 0; passPartition < passPartitions; passPartition++) {
                        partitionTargetOffsets[passPartition + 1] = partitionTargetOffsets[passPartition] + partitionTargets[passFirstPartition + passPartition];
                    }
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
                    for (size_t i = 0; i < passPartitions; i++) {
                        size_t passPartition = partitionOrder[i];
                        size_t partition = passFirstPartition + passPartition;
                        size_t partitionFirstKey = getPartitionFirstKey(exchange, sendSplit, partition);
                        countPartitionTargets(exchange, receivedPairFiles, pairFileRows, partitionPassRows.data() + partitionPassRowOffsets[passPartition],
                                              partitionPassRowOffsets[passPartition + 1] - partitionPassRowOffsets[passPartition], codeBuffer,
                                              getPartitionFirstKey(exchange, sendSplit, partition + 1) - partitionFirstKey, partitionTargets[partition], partitionMerges[passPartition], 1);
                        mergePartitionTargets(exchange, seqDbr, partitionFirstKey, partitionMerges[passPartition], pairPassTargets + partitionTargetOffsets[passPartition], 1);
                    }
                    if (exchange.hasTargetSplits) {
                        // the partitions in key order, the target splits in parallel
                        for (size_t passPartition = 0; passPartition < passPartitions; passPartition++) {
                            const PartitionMerge &merge = partitionMerges[passPartition];
                            writeTargetSplits(prefFiles, targetSplitFiles, seqDbr, getPartitionFirstKey(exchange, sendSplit, passFirstPartition + passPartition), merge.keyOffsets.size() - 1,
                                              merge.keyOffsets.data(), merge.targets, par.threads);
                            partitionMerges[passPartition] = PartitionMerge();
                        }
                        continue;
                    }
                    std::vector<size_t> partitionDataOffsets(passPartitions + 1, prefDataSize);
                    std::vector<size_t> partitionIndexOffsets(passPartitions + 1, prefIndexSize);
                    for (size_t passPartition = 0; passPartition < passPartitions; passPartition++) {
                        const PartitionMerge &merge = partitionMerges[passPartition];
                        partitionDataOffsets[passPartition + 1] = partitionDataOffsets[passPartition] + merge.keyTaskDataOffsets.back();
                        partitionIndexOffsets[passPartition + 1] = partitionIndexOffsets[passPartition] + merge.keyTaskEntryOffsets.back() * 2 * sizeof(uint64_t);
                        for (size_t keyTask = 0; keyTask + 1 < merge.keyTaskOffsets.size(); keyTask++) {
                            addKeyTaskCuts(partitionCuts, getPartitionFirstKey(exchange, sendSplit, passFirstPartition + passPartition), merge.keyTaskOffsets[keyTask],
                                           partitionDataOffsets[passPartition] + merge.keyTaskDataOffsets[keyTask], merge.keyTaskCostOffsets[keyTask], merge.keyTaskCuts[keyTask]);
                        }
                        addPartitionEnd(partitionCuts, partitionCutOffsets, mergedCost, partitionDataOffsets[passPartition + 1], merge.cost);
                    }
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
                    for (size_t i = 0; i < passPartitions; i++) {
                        unsigned int thread_idx = 0;
#ifdef OPENMP
                        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
                        size_t passPartition = partitionOrder[i];
                        writePartition(prefFiles, getPartitionFirstKey(exchange, sendSplit, passFirstPartition + passPartition), partitionMerges[passPartition],
                                       partitionDataOffsets[passPartition], partitionIndexOffsets[passPartition], thread_idx, 1);
                        partitionMerges[passPartition] = PartitionMerge();
                    }
                    prefDataSize = partitionDataOffsets[passPartitions];
                    prefIndexSize = partitionIndexOffsets[passPartitions];
                } else {
                    // few partitions: all threads merge a partition
                    for (size_t passPartition = 0; passPartition < passPartitions; passPartition++) {
                        size_t partition = passFirstPartition + passPartition;
                        size_t partitionFirstKey = getPartitionFirstKey(exchange, sendSplit, partition);
                        size_t partitionKeys = getPartitionFirstKey(exchange, sendSplit, partition + 1) - partitionFirstKey;
                        size_t partitionRowCount = partitionPassRowOffsets[passPartition + 1] - partitionPassRowOffsets[passPartition];
                        size_t partitionDataOffset = prefDataSize;
                        size_t partitionEntryCount = 0;
                        size_t partitionCost = 0;
                        if (isRefilledPass == false) {
                            int mergeThreads = par.threads;
                            while (mergeThreads > 1 && pairPassSizes[pass] - getPartitionMergeSize(partitionRowCount, 1) + getPartitionMergeSize(partitionRowCount, mergeThreads) > pairPassLimit) {
                                mergeThreads--;
                            }
                            partitionMerges.resize(1);
                            PartitionMerge &merge = partitionMerges[0];
                            countPartitionTargets(exchange, receivedPairFiles, pairFileRows, partitionPassRows.data() + partitionPassRowOffsets[passPartition], partitionRowCount, codeBuffer,
                                                  partitionKeys, partitionTargets[partition], merge, mergeThreads);
                            mergePartitionTargets(exchange, seqDbr, partitionFirstKey, merge, pairPassTargets, mergeThreads);
                            if (exchange.hasTargetSplits) {
                                writeTargetSplits(prefFiles, targetSplitFiles, seqDbr, partitionFirstKey, partitionKeys, merge.keyOffsets.data(), merge.targets, par.threads);
                                continue;
                            }
                            writePartition(prefFiles, partitionFirstKey, merge, prefDataSize, prefIndexSize, 0, par.threads);
                            for (size_t keyTask = 0; keyTask + 1 < merge.keyTaskOffsets.size(); keyTask++) {
                                addKeyTaskCuts(partitionCuts, partitionFirstKey, merge.keyTaskOffsets[keyTask], prefDataSize + merge.keyTaskDataOffsets[keyTask], merge.keyTaskCostOffsets[keyTask],
                                               merge.keyTaskCuts[keyTask]);
                            }
                            partitionEntryCount = merge.keyTaskEntryOffsets.back();
                            partitionCost = merge.cost;
                            prefDataSize += merge.keyTaskDataOffsets.back();
                        } else {
                            size_t codeReadBufferSize = 2 * pairPassLimit / (std::max((size_t) 1, partitionRowCount) * PAIR_CODE_COUNT);
                            if (codeReadBufferSize == 0) {
                                Debug(Debug::ERROR) << "Not enough memory to merge the pairs. Raise --split-memory-limit.\n";
                                EXIT(EXIT_FAILURE);
                            }
                            std::vector<PairRowRead> reads;
                            getPartitionRowReads(receivedPairFiles, pairFileRows, partitionPassRows.data() + partitionPassRowOffsets[passPartition], partitionRowCount, partitionKeys, codeBuffer,
                                                 codeReadBufferSize, reads);
                            RowValueQueue repKeyQueue;
                            RowValueQueue targetQueue;
                            std::vector<unsigned int> keyRows;
                            for (size_t row = 0; row < reads.size(); row++) {
                                readPairRecord(reads[row], exchange.globalIdBits);
                                repKeyQueue.push(RowValue(reads[row].repKey, row));
                            }
                            if (exchange.hasTargetSplits) {
                                // the keys and targets of 16 MiB besides the last key, then an entry in every target split
                                const size_t chunkSize = KMER_FILE_BUFFER_SIZE / sizeof(DBKeyType);
                                std::vector<size_t> chunkKeyOffsets;
                                std::vector<DBKeyType> chunkTargets;
                                chunkKeyOffsets.reserve(chunkSize);
                                chunkTargets.reserve(chunkSize);
                                chunkKeyOffsets.push_back(0);
                                size_t chunkFirstKey = repKeyQueue.empty() ? partitionKeys : static_cast<size_t>(repKeyQueue.top().value);
                                for (size_t key = chunkFirstKey; key < partitionKeys;) {
                                    const bool hasEntry = isRepresentative(partitionFirstKey + key);
                                    const auto addTarget = [&](DBKeyType target) {
                                        if (hasEntry) {
                                            chunkTargets.push_back(target);
                                        }
                                    };
                                    mergeKeyTargets(reads, repKeyQueue, targetQueue, keyRows, key, exchange.globalIdBits, addTarget);
                                    chunkKeyOffsets.push_back(chunkTargets.size());
                                    const size_t nextKey = repKeyQueue.empty() ? partitionKeys : static_cast<size_t>(repKeyQueue.top().value);
                                    if (nextKey == partitionKeys || chunkTargets.size() + nextKey - chunkFirstKey >= chunkSize) {
                                        writeTargetSplits(prefFiles, targetSplitFiles, seqDbr, partitionFirstKey + chunkFirstKey, key + 1 - chunkFirstKey, chunkKeyOffsets.data(), chunkTargets.data(),
                                                          par.threads);
                                        chunkFirstKey = nextKey;
                                        chunkKeyOffsets.assign(1, 0);
                                        chunkTargets.clear();
                                    } else {
                                        chunkKeyOffsets.resize(nextKey - chunkFirstKey + 1, chunkTargets.size());
                                    }
                                    key = nextKey;
                                }
                                continue;
                            }
                            size_t taskKeys = std::max((size_t) 1, partitionKeys / (TASKS_PER_THREAD * par.threads));
                            KeyTaskCuts taskCuts;
                            size_t taskFirstKey = 0;
                            size_t taskPrefOffset = partitionDataOffset;
                            size_t taskCost = 0;
                            KmerCodeWriter dataWrite(fileno(prefFiles.dataFile), prefFiles.dataName.c_str(), prefFiles.writeBuffers, KMER_FILE_BUFFER_SIZE, prefDataSize, &batchWriter);
                            KmerCodeWriter indexWrite(fileno(prefFiles.indexFile), prefFiles.indexName.c_str(), prefFiles.writeBuffers + KMER_FILE_BUFFER_SIZE, KMER_FILE_BUFFER_SIZE, prefIndexSize, &batchWriter);
                            for (size_t key = 0; key < partitionKeys;) {
                                if (key == taskFirstKey + taskKeys) {
                                    addKeyTaskCuts(partitionCuts, partitionFirstKey, taskFirstKey, taskPrefOffset, taskCost, taskCuts);
                                    taskCuts = KeyTaskCuts();
                                    taskFirstKey = key;
                                    taskPrefOffset = dataWrite.writePos + dataWrite.bufferPos;
                                    taskCost = partitionCost;
                                }
                                bool hasEntry = repKeyQueue.empty() == false && repKeyQueue.top().value == key && isRepresentative(partitionFirstKey + key);
                                if (hasEntry) {
                                    uint64_t indexCell[2] = {partitionFirstKey + key, dataWrite.writePos + dataWrite.bufferPos};
                                    putBytes(indexWrite, (const char *) indexCell, sizeof(indexCell));
                                    putPrefKey(dataWrite, partitionFirstKey + key);
                                }
                                PrefTargetWriter targetWriter = { hasEntry, dataWrite, seqDbr, 0, SIZE_MAX, 0 };
                                mergeKeyTargets(reads, repKeyQueue, targetQueue, keyRows, key, exchange.globalIdBits, targetWriter);
                                size_t targetResidues = targetWriter.residues;
                                if (hasEntry) {
                                    putBytes(dataWrite, "", 1);
                                    partitionEntryCount++;
                                    partitionCost += (index[seqDbr.findIdRange(partitionFirstKey + key)].length - 2) * targetResidues;
                                    PartitionCut cut = { key + 1, dataWrite.writePos + dataWrite.bufferPos - taskPrefOffset, partitionCost - taskCost };
                                    taskCuts.add(cut);
                                }
                                key = std::min(taskFirstKey + taskKeys, repKeyQueue.empty() ? partitionKeys : static_cast<size_t>(repKeyQueue.top().value));
                            }
                            writeKmerCodeBuffer(dataWrite);
                            writeKmerCodeBuffer(indexWrite);
                            prefDataSize = dataWrite.writePos;
                            addKeyTaskCuts(partitionCuts, partitionFirstKey, taskFirstKey, taskPrefOffset, taskCost, taskCuts);
                        }
                        addPartitionEnd(partitionCuts, partitionCutOffsets, mergedCost, prefDataSize, partitionCost);
                        prefIndexSize += partitionEntryCount * 2 * sizeof(uint64_t);
                    }
                }
            }
            delete batchReader;
            std::vector<PassRow> splitRows;
            std::vector<size_t> splitRowOffsets;
            getPairPassRows(partitionRows, firstPartition, endPartition, splitRows, splitRowOffsets);
            punchPassRows(receivedPairFiles, pairFileRows, splitRows, releasedSize);
            closeReceivedFiles(receivedPairFiles);
            addFinishedItems(par, progress, PAIR_FILE_GROUPS, fileGroup, pieceEnd - pieceBegin);
        }
        if (windowFirstKey != SIZE_MAX) {
            addTargetSplitRows(targetSplitFiles, windowFirstKey, window, partitionsData);
        }
    }
    batchWriter.waitWrites();
    Util::releasePages(pairPassBuffer, pairPassBufferSize);
    delete[] pairPassBuffer;
    Util::releasePages(pairPassTargets, pairPassTargetsSize * sizeof(DBKeyType));
    delete[] pairPassTargets;
    delete[] prefFiles.writeBuffers;
    for (size_t split = 0; split < targetSplitFiles.dataFiles.size(); split++) {
        FILE *indexFile = FileUtil::openAndDelete(targetSplitFiles.indexNames[split].c_str(), "w");
        size_t indexSize = targetSplitFiles.indexCells[split].size() * sizeof(uint64_t);
        FileUtil::pwriteAll(fileno(indexFile), targetSplitFiles.indexNames[split].c_str(), (const char *) targetSplitFiles.indexCells[split].data(), indexSize, 0);
        MMseqsNode::closeClusterFiles(targetSplitFiles.dataFiles[split], targetSplitFiles.dataNames[split], indexFile, targetSplitFiles.indexNames[split],
                                      targetSplitFiles.dataSizes[split], indexSize);
        prefSize += targetSplitFiles.dataSizes[split] + indexSize + 2 * sizeof(uint64_t);
    }
    if (exchange.hasTargetSplits == false) {
        MMseqsNode::closeClusterFiles(prefFiles.dataFile, prefFiles.dataName, prefFiles.indexFile, prefFiles.indexName, prefDataSize, prefIndexSize);
        prefSize = prefDataSize + prefIndexSize + 2 * sizeof(uint64_t);
    }
    size_t rowCost = std::max((size_t) 1, static_cast<size_t>(mergedCost / 16384));
    for (size_t partition = 0; partition + 1 < partitionCutOffsets.size(); partition++) {
        const PartitionCut *rowStart = &partitionCuts[partitionCutOffsets[partition]];
        for (size_t i = partitionCutOffsets[partition] + 1; i < partitionCutOffsets[partition + 1]; i++) {
            const PartitionCut &cut = partitionCuts[i];
            if (i + 1 < partitionCutOffsets[partition + 1] && cut.cost - rowStart->cost < rowCost) {
                continue;
            }
            if (cut.prefOffset > rowStart->prefOffset) {
                partitionsData += SSTR(rowStart->key) + "\t" + SSTR(rowStart->prefOffset) + "\t" + SSTR(cut.prefOffset - rowStart->prefOffset) + "\t" + SSTR(cut.cost - rowStart->cost)
                                  + "\t" + SSTR(MMseqsNode::rank) + "\n";
            }
            rowStart = &cut;
        }
    }
    return partitionsData;
}

template <typename T, bool includeAdjacency, typename L>
void kmermatcherLinfinity(Parameters &par, DBReader<DBKeyType> &seqDbr, BaseMatrix *subMat, size_t memoryLimit) {
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    float kmersPerSequenceScale = par.kmersPerSequenceScale.values.aminoacid();
    KmerExchange exchange;
    exchange.memoryLimit = memoryLimit;
    size_t totalKmerCount = 0;
    size_t totalSequenceSize = 0;
    for (size_t range = 0; range < seqDbr.getIndexSize(); range++) {
        size_t rangeKeys = ((range + 1 < seqDbr.getIndexSize()) ? static_cast<size_t>(index[range + 1].id) : seqDbr.getSize()) - index[range].id;
        totalKmerCount += seqDbr.getRangeRepresentativeCount(range) * computeSeqKmerCount(static_cast<int>(index[range].length - 2), par.kmerSize, par.kmersPerSequence, kmersPerSequenceScale);
        totalSequenceSize += rangeKeys * index[range].length;
    }
    exchange.sendSplitCount = (MMseqsNode::count > 1) ? RANGES_PER_NODE * MMseqsNode::count : 1;
    std::vector<size_t> &sendSplitKeyOffsets = exchange.sendSplitKeyOffsets;
    sendSplitKeyOffsets.assign(exchange.sendSplitCount + 1, seqDbr.getSize());
    sendSplitKeyOffsets[0] = 0;
    size_t sendSplitCost = std::max((size_t) 1, (totalKmerCount * SEND_KMER_SIZE + totalSequenceSize + exchange.sendSplitCount - 1) / exchange.sendSplitCount);
    size_t costBefore = 0;
    for (size_t range = 0, sendSplit = 1; range < seqDbr.getIndexSize(); range++) {
        size_t rangeKeys = ((range + 1 < seqDbr.getIndexSize()) ? static_cast<size_t>(index[range + 1].id) : seqDbr.getSize()) - index[range].id;
        size_t rangeKmers = seqDbr.getRangeRepresentativeCount(range) * computeSeqKmerCount(static_cast<int>(index[range].length - 2), par.kmerSize, par.kmersPerSequence, kmersPerSequenceScale);
        size_t rangeCost = rangeKmers * SEND_KMER_SIZE + rangeKeys * index[range].length;
        for (; sendSplit < exchange.sendSplitCount && costBefore + rangeCost >= sendSplit * sendSplitCost; sendSplit++) {
            size_t cutCost = sendSplit * sendSplitCost - costBefore;
            sendSplitKeyOffsets[sendSplit] = index[range].id + static_cast<size_t>((static_cast<PackedKmerPosition::uint128>(cutCost) * rangeKeys + rangeCost - 1) / rangeCost);
        }
        costBefore += rangeCost;
    }
    size_t maxSendSplitKeys = 0;
    for (size_t sendSplit = 0; sendSplit < exchange.sendSplitCount; sendSplit++) {
        maxSendSplitKeys = std::max(maxSendSplitKeys, sendSplitKeyOffsets[sendSplit + 1] - sendSplitKeyOffsets[sendSplit]);
    }
    exchange.fileKeyOffsets = seqDbr.getFileKeyOffsets(MMseqsNode::count);
    exchange.hasRepresentativeBits = FileUtil::fileExists((par.db1 + ".bits." + SSTR(MMseqsNode::rank)).c_str());
    // the pref is written by target split, runs of the blocks that align2clust reads; one align node reads the query split pref
    exchange.hasTargetSplits = par.align2clustMode == Parameters::ALIGN2CLUST_MODE_TARGET_SPLIT && MMseqsNode::count - MMseqsNode::getFirstAlignNode() > 1;
    if (exchange.hasTargetSplits) {
        seqDbr.setBlockCacheSize(0);
    }
    // the master writes the dbtype last
    if (MMseqsNode::isMaster() && FileUtil::fileExists((par.db2 + ".dbtype").c_str())) {
        FileUtil::remove((par.db2 + ".dbtype").c_str());
    }

    exchange.adjKeyBits = (includeAdjacency ? 6 : 0) * PackedKmerPosition::RESIDUE_BITS;
    exchange.adjKeyBase = includeAdjacency ? adjSubMat->alphabetSize : 1;
    exchange.adjKeyCodeBits = includeAdjacency ? getBitWidth(MathUtil::ipow<size_t>(exchange.adjKeyBase, 6) - 1) : 0;
    exchange.kmerHashShift = 64 - getKmerBits(subMat, par.kmerSize);

    exchange.hashRangeCount = std::max(RANGES_PER_NODE * MMseqsNode::count, TASKS_PER_THREAD * par.threads);
    ExchangeFileGroups kmerGroups;
    ExchangeFileGroups pairGroups;
    initFileGroups(kmerGroups, par.db2 + ".kmers", exchange.hashRangeCount, MMseqsNode::count, 4);

    struct rlimit openFileLimit = {0, 0};
    getrlimit(RLIMIT_NOFILE, &openFileLimit);
    size_t minOpenFiles = seqDbr.getDataFileCnt() + 64 + kmerGroups.count + 2 * MMseqsNode::count + 2 * (exchange.hasTargetSplits ? MAX_TARGET_SPLITS_PER_NODE * (MMseqsNode::count - MMseqsNode::getFirstAlignNode()) : 1);
    if (minOpenFiles > openFileLimit.rlim_cur) {
        Debug(Debug::ERROR) << "Raise the open file limit (ulimit -n) to at least " << minOpenFiles << "\n";
        EXIT(EXIT_FAILURE);
    }
    exchange.globalIdBits = getBitWidth(seqDbr.getSize() - 1);
    exchange.maxRepSeqCount = 1 + (includeAdjacency ? par.adjIteration : 0);
    exchange.selfMergedSendSplitCount = exchange.sendSplitCount / MMseqsNode::count * MMseqsNode::count / (2 * MMseqsNode::count - 1);
    size_t receiverKmerSize = (2 * exchange.globalIdBits + exchange.adjKeyCodeBits + CHAR_BIT - 1) / CHAR_BIT;
    size_t pairTargetSize = (exchange.globalIdBits + CHAR_BIT - 1) / CHAR_BIT + sizeof(DBKeyType);
    MMseqsNode::WorkQueue sendQueue(par.db2 + ".send_splits", exchange.sendSplitCount);
    MMseqsNode::WorkQueue hashQueue(par.db2 + ".hash_ranges", exchange.hashRangeCount);
    MMseqsNode::WorkQueue mergeQueue(par.db2 + ".merge_splits", exchange.sendSplitCount);
    sendQueue.removeTakenItemFiles();
    hashQueue.removeTakenItemFiles();
    mergeQueue.removeTakenItemFiles();
    std::string progressFile = par.db2 + ".progress." + SSTR(MMseqsNode::rank);
    std::vector<std::string> killedRunFiles(1, progressFile);
    for (size_t i = 0; i < kmerGroups.count && MMseqsNode::isMaster(); i++) {
        killedRunFiles.push_back(par.db2 + ".kmers.released." + SSTR(i));
        killedRunFiles.push_back(par.db2 + ".pairs.released." + SSTR(i));
    }
    for (size_t i = 0; i < killedRunFiles.size(); i++) {
        if (FileUtil::fileExists(killedRunFiles[i].c_str())) {
            FileUtil::remove(killedRunFiles[i].c_str());
        }
    }
    std::string indexFiles[FILE_GROUP_KINDS] = { par.db2 + ".kmers.index." + SSTR(MMseqsNode::rank), par.db2 + ".pairs.index." + SSTR(MMseqsNode::rank) };
    for (size_t i = 0; i < FILE_GROUP_KINDS; i++) {
        if (fclose(FileUtil::openAndDelete(indexFiles[i].c_str(), "w")) != 0) {
            Debug(Debug::ERROR) << "Cannot create " << indexFiles[i] << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
    size_t localNodes = 0;
    size_t minPassLimit = std::max((size_t) 1, getPassLimit(MMseqsNode::getSmallestMemoryLimit(par.db2 + ".memory", memoryLimit, par.threads, par.localTmp, &localNodes)));
    // localNodes counts this node too when its --local-tmp is local flash
    if (par.localTmp.empty() == false && localNodes == 0) {
        MMseqsNode::reportUnusableLocalTmp("Cannot keep the last exchange file groups in --local-tmp " + par.localTmp + ": it is not local flash");
    }
    size_t passes = std::max((size_t) 1, (totalKmerCount * receiverKmerSize + minPassLimit - 1) / minPassLimit);
    exchange.hashSubRangesPerRange = (std::max(passes, exchange.hashRangeCount) * TASKS_PER_THREAD * par.threads + exchange.hashRangeCount - 1) / exchange.hashRangeCount;
    exchange.hashSubRangeCount = exchange.hashRangeCount * exchange.hashSubRangesPerRange;
    exchange.rowsPerHashRange = exchange.hashSubRangesPerRange + 1;
    size_t representativeBitsMemory = exchange.hasRepresentativeBits ? ((maxSendSplitKeys + 63) / 64 + 1) * sizeof(uint64_t) : 0;
    exchange.writeBehindSize = memoryLimit / 8;
    // target splits add the block table and the keys and targets of a refilled pass
    size_t fixedMemory = representativeBitsMemory + 2 * par.threads * KMER_FILE_BUFFER_SIZE + exchange.writeBehindSize
                               + exchange.hashRangeCount * exchange.rowsPerHashRange * sizeof(KmerFileRow)
                               + (exchange.hasTargetSplits ? seqDbr.getBlockKeyOffsets().size() * sizeof(size_t) + 2 * KMER_FILE_BUFFER_SIZE : 0);
    exchange.keySplitMemoryLimit = (fixedMemory < memoryLimit) ? static_cast<size_t>(static_cast<double>(memoryLimit - fixedMemory) * 0.95) : 0;
    size_t rangePairPasses = std::max((size_t) 1, (totalKmerCount / exchange.sendSplitCount * exchange.maxRepSeqCount * pairTargetSize + minPassLimit - 1) / minPassLimit);
    exchange.partitionsPerRange = std::max((size_t) 4 * par.threads, 2 * rangePairPasses);
    exchange.pairRowSize = exchange.sendSplitCount * exchange.partitionsPerRange * sizeof(PairFileRow);
    size_t partitionCount = exchange.sendSplitCount * exchange.partitionsPerRange;
    size_t partitionsPerGroup = (partitionCount + kmerGroups.count - 1) / kmerGroups.count;
    initFileGroups(pairGroups, par.db2 + ".pairs", partitionCount, partitionsPerGroup, 4 + partitionsPerGroup);
    if (localNodes > 0) {
        size_t freeSize = FileUtil::getFreeSpace(par.localTmp.c_str());
        size_t share = (freeSize == SIZE_MAX) ? 0 : freeSize / localNodes;
        size_t kmerGroupSize = std::max((size_t) 1, totalKmerCount * receiverKmerSize / (MMseqsNode::count * kmerGroups.count));
        size_t pairGroupSize = std::max((size_t) 1, totalKmerCount * ((exchange.globalIdBits + CHAR_BIT - 1) / CHAR_BIT) / (MMseqsNode::count * pairGroups.count));
        kmerGroups.localCount = std::min(kmerGroups.count, share / (3 * kmerGroupSize));
        pairGroups.localCount = std::min(pairGroups.count, share / (3 * pairGroupSize));
        if (kmerGroups.localCount < kmerGroups.count || pairGroups.localCount < pairGroups.count) {
            MMseqsNode::reportUnusableLocalTmp("Cannot keep all exchange file groups in --local-tmp " + par.localTmp + ": its " + ((freeSize == SIZE_MAX) ? std::string("unknown free space holds ") : SSTR(share) + " free bytes per node hold ")
                                               + SSTR(kmerGroups.localCount) + " of " + SSTR(kmerGroups.count) + " k-mer and " + SSTR(pairGroups.localCount) + " of "
                                               + SSTR(pairGroups.count) + " pair file groups");
        }
    }
    kmerGroups.localTmp = par.localTmp;
    pairGroups.localTmp = par.localTmp;
    kmerGroups.localNodes = localNodes;
    pairGroups.localNodes = localNodes;
    std::string progress;

    Timer phaseTimer;
    std::string sendValues;
    size_t sentSplitCount = 0;
    OwnKmers ownKmers = {NULL, 0, getPassLimit(exchange.keySplitMemoryLimit - std::min(exchange.keySplitMemoryLimit, exchange.pairRowSize)),
                         std::vector<std::vector<KmerFileRow> >(exchange.hashRangeCount), std::vector<size_t>()};
    size_t sendSplitKmerSize = totalKmerCount / exchange.sendSplitCount * sizeof(PackedKmerPosition);
    size_t ownMemory = std::min(ownKmers.limit, exchange.keySplitMemoryLimit - std::min(exchange.keySplitMemoryLimit, sendSplitKmerSize));
    size_t ownHashRangeSize = std::max((size_t) 1, totalKmerCount * receiverKmerSize / (MMseqsNode::count * exchange.hashRangeCount));
    size_t ownItems = (exchange.hashRangeCount - MMseqsNode::rank + MMseqsNode::count - 1) / MMseqsNode::count;
    size_t ownHashRanges = std::min(ownItems * MMseqsNode::count / (2 * MMseqsNode::count - 1), ownMemory / ownHashRangeSize);
    for (size_t i = 0; i < ownHashRanges; i++) {
        ownKmers.hashRanges.push_back(MMseqsNode::rank + i * MMseqsNode::count);
    }
    if (MMseqsNode::isMaster()) {
        Debug(Debug::INFO) << "Send k-mers in " << exchange.sendSplitCount << " send splits\n";
    }
    size_t sentSize = sendKmers<T, includeAdjacency, L>(par, seqDbr, subMat, exchange, sendQueue, representativeBitsMemory, sentSplitCount, ownKmers, kmerGroups);
    std::thread kmerFollow(followFileGroups, std::ref(kmerGroups));
    // claim the own hash ranges
    for (size_t i = 0; i < ownKmers.hashRanges.size(); i++) {
        hashQueue.takeNextItem();
    }
    MMseqsNode::addPhaseTime(sendValues, "sending k-mers", phaseTimer);
    MMseqsNode::addPhaseValue(sendValues, "K-mer bytes sent", SSTR(sentSize + ownKmers.size));
    MMseqsNode::addPhaseValue(sendValues, "K-mer bytes kept in memory", SSTR(ownKmers.size));
    MMseqsNode::addPhaseValue(sendValues, "Send splits", SSTR(sentSplitCount));
    MMseqsNode::printPhaseValues(sendValues);
    std::string nodeFlushCounts = MMseqsNode::barrier(par.db2 + ".kmers.done", SSTR(kmerGroups.flushCount) + "\n");
    double sendWaitSeconds = phaseTimer.getTimediff();
    size_t receivedFlushCount = 0;
    for (char *line = (char *) nodeFlushCounts.c_str(); *line != '\0'; line = Util::skipLine(line)) {
        receivedFlushCount += Util::fast_atoi<size_t>(line);
    }
    // every node has passed the send splits
    sendQueue.removeTakenItemFiles();
    // every node has read the memory limits
    std::thread fileGroupRelease;
    if (MMseqsNode::isMaster()) {
        FileUtil::remove((par.db2 + ".memory").c_str());
        fileGroupRelease = std::thread(releaseFileGroups, std::cref(par), std::ref(kmerGroups), std::ref(pairGroups));
    }

    phaseTimer.reset();
    std::string receiveValues;
    MMseqsNode::addPhaseSeconds(receiveValues, "waiting for the other nodes to send k-mers", sendWaitSeconds);
    size_t receivedHashRangeCount = 0;
    size_t pairSize = 0;
    size_t pairTargetCount = 0;
    if (MMseqsNode::isMaster()) {
        Debug(Debug::INFO) << "Receive k-mers in " << exchange.hashRangeCount << " hash ranges\n";
    }
    OwnPairs ownPairs = {NULL, 0, 0, std::vector<std::vector<PairFileRow> >(exchange.sendSplitCount)};
    size_t receivedSize = receiveKmers(par, seqDbr, exchange, hashQueue, receivedHashRangeCount, pairSize, pairTargetCount, ownKmers, ownPairs, kmerGroups, pairGroups, receivedFlushCount, progress);
    std::thread pairFollow(followFileGroups, std::ref(pairGroups));
    MMseqsNode::addPhaseTime(receiveValues, "receiving k-mers", phaseTimer);
    MMseqsNode::addPhaseSeconds(receiveValues, "waiting for k-mer file groups", kmerGroups.waitSeconds);
    MMseqsNode::addPhaseValue(receiveValues, "K-mer bytes received", SSTR(receivedSize));
    MMseqsNode::addPhaseValue(receiveValues, "Pair bytes written", SSTR(pairSize));
    MMseqsNode::addPhaseValue(receiveValues, "Pair targets written", SSTR(pairTargetCount));
    MMseqsNode::addPhaseValue(receiveValues, "Hash ranges", SSTR(receivedHashRangeCount));
    MMseqsNode::printPhaseValues(receiveValues);
    kmerFollow.join();
    // the targets of the pair records, a few may repeat in other records
    exchange.pairTargetCount = 0;
    const std::string nodePairTargetCounts = MMseqsNode::barrier(par.db2 + ".pairs.done", SSTR(pairTargetCount) + "\n");
    for (char *line = (char *) nodePairTargetCounts.c_str(); *line != '\0'; line = Util::skipLine(line)) {
        exchange.pairTargetCount += Util::fast_atoi<size_t>(line);
    }
    std::string mergeValues;
    MMseqsNode::addPhaseTime(mergeValues, "waiting for the other nodes to receive k-mers", phaseTimer);
    // every node has passed the hash ranges
    hashQueue.removeTakenItemFiles();
    if (MMseqsNode::isMaster()) {
        FileUtil::remove((par.db2 + ".kmers.done").c_str());
    }

    size_t mergedSplitCount = 0;
    if (MMseqsNode::isMaster()) {
        Debug(Debug::INFO) << "Merge pairs in " << exchange.sendSplitCount << " splits\n";
    }
    std::vector<size_t> targetSplitBlockOffsets;
    size_t prefSize = 0;
    std::string partitionsData = mergePairFiles(par, seqDbr, exchange, mergeQueue, mergedSplitCount, ownPairs, pairGroups, progress, targetSplitBlockOffsets, prefSize);
    MMseqsNode::addPhaseTime(mergeValues, "merging pairs", phaseTimer);
    MMseqsNode::addPhaseSeconds(mergeValues, "waiting for pair file groups", pairGroups.waitSeconds);
    MMseqsNode::addPhaseValue(mergeValues, "Merge splits", SSTR(mergedSplitCount));
    MMseqsNode::addPhaseValue(mergeValues, "Prefilter bytes written", SSTR(prefSize));
    if (exchange.hasTargetSplits) {
        MMseqsNode::addPhaseValue(mergeValues, "Target splits", SSTR(targetSplitBlockOffsets.size() - 1) + " of " + SSTR(seqDbr.getBlockKeyOffsets().size() - 1) + " blocks");
    }
    MMseqsNode::printPhaseValues(mergeValues);
    pairFollow.join();
    MMseqsNode::publishRunFile(par.db2 + ".partitions." + SSTR(MMseqsNode::rank), partitionsData);
    if (MMseqsNode::isMaster()) {
        std::string partitions = MMseqsNode::gatherRunFiles(par.db2 + ".partitions.");
        std::string mergeWaitValues;
        MMseqsNode::addPhaseTime(mergeWaitValues, "waiting for the other nodes to merge pairs", phaseTimer);
        fileGroupRelease.join();
        MMseqsNode::publishRunFile(par.db2 + ".merged", "");
        for (size_t i = 0; i < kmerGroups.count; i++) {
            FileUtil::remove((kmerGroups.prefix + ".released." + SSTR(i)).c_str());
        }
        for (size_t i = 0; i < pairGroups.count; i++) {
            FileUtil::remove((pairGroups.prefix + ".released." + SSTR(i)).c_str());
        }
        MMseqsNode::printPhaseValues(mergeWaitValues);
        mergeQueue.removeTakenItemFiles();
        FileUtil::remove((par.db2 + ".pairs.done").c_str());
        // sort the partition rows of all nodes by key
        std::vector<std::pair<size_t, char *> > partitionRows;
        for (char *row = (char *) partitions.c_str(); *row != '\0'; row = Util::skipLine(row)) {
            partitionRows.push_back(std::make_pair(Util::fast_atoi<size_t>(row), row));
        }
        SORT_SERIAL(partitionRows.begin(), partitionRows.end());
        // target split mode: the first line holds the first block of every target split
        std::string sortedPartitions;
        for (size_t i = 0; i < targetSplitBlockOffsets.size(); i++) {
            sortedPartitions += SSTR(targetSplitBlockOffsets[i]) + ((i + 1 < targetSplitBlockOffsets.size()) ? "\t" : "\n");
        }
        for (size_t i = 0; i < partitionRows.size(); i++) {
            sortedPartitions.append(partitionRows[i].second, Util::skipLine(partitionRows[i].second) - partitionRows[i].second);
        }
        MMseqsNode::publishFile(par.db2 + ".partitions", sortedPartitions);
        MMseqsNode::removeOnceRead(par.db2 + ".merged");
        DBWriter::writeDbtypeFile(par.db2.c_str(), DBReader<DBKeyType>::setExtendedDbtype(Parameters::DBTYPE_PREFILTER_RES, Parameters::DBTYPE_EXTENDED_LINFINITY), false);
    } else {
        MMseqsNode::waitForRunFile(par.db2 + ".merged");
        MMseqsNode::removeOnceRead(par.db2 + ".merged");
        mergeQueue.removeTakenItemFiles();
    }
    // nodes without work have no progress file
    if (FileUtil::fileExists(progressFile.c_str())) {
        FileUtil::remove(progressFile.c_str());
    }
    for (size_t i = 0; i < FILE_GROUP_KINDS; i++) {
        FileUtil::remove(indexFiles[i].c_str());
    }
}

template void kmermatcherLinfinity<short, true, unsigned short>(Parameters &, DBReader<DBKeyType> &, BaseMatrix *, size_t);
template void kmermatcherLinfinity<short, false, unsigned short>(Parameters &, DBReader<DBKeyType> &, BaseMatrix *, size_t);
template void kmermatcherLinfinity<int, true, unsigned short>(Parameters &, DBReader<DBKeyType> &, BaseMatrix *, size_t);
template void kmermatcherLinfinity<int, false, unsigned short>(Parameters &, DBReader<DBKeyType> &, BaseMatrix *, size_t);
template void kmermatcherLinfinity<int, true, int>(Parameters &, DBReader<DBKeyType> &, BaseMatrix *, size_t);
template void kmermatcherLinfinity<int, false, int>(Parameters &, DBReader<DBKeyType> &, BaseMatrix *, size_t);
