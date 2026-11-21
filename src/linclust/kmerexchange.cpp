#include "kmermatcher.h"
#include "Debug.h"
#include "FileUtil.h"
#include "FastSort.h"
#include "MMseqsNode.h"
#include "BatchReader.h"

#include <sys/stat.h>

#include <thread>
#include <algorithm>

#ifdef OPENMP
#include <omp.h>
#endif

static const size_t KMER_FILE_BUFFER_SIZE = 16 * 1024 * 1024;
static const size_t TASKS_PER_THREAD = 2;
static const size_t RANGES_PER_NODE = 8;
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
    double writeWaitSeconds;
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
    groups.writeWaitSeconds = 0;
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
    std::string localName = getLocalFileGroupName(groups, group);
    FILE *shared = FileUtil::openAndDelete(name.c_str(), "w");
    char *buffer = new(std::nothrow) char[KMER_FILE_BUFFER_SIZE];
    Util::checkAllocation(buffer, "Can not allocate buffer memory in copyLocalFileGroup");
    for (size_t offset = 0; offset < size; offset += KMER_FILE_BUFFER_SIZE) {
        size_t copySize = std::min(KMER_FILE_BUFFER_SIZE, size - offset);
        FileUtil::preadAll(fileno(local), localName.c_str(), buffer, copySize, offset);
        FileUtil::pwriteAll(fileno(shared), name.c_str(), buffer, copySize, offset);
    }
    delete[] buffer;
    if (fclose(local) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << localName << "\n";
        EXIT(EXIT_FAILURE);
    }
    FileUtil::remove(localName.c_str());
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
                         BatchReader &batchReader) {
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
    if (isAsync) {
        batchReader.readAsync(readRequests);
    } else {
        batchReader.readSync(readRequests);
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
    KmerCodeReader codeRead = {NULL, 0, 0, 0, -1, NULL, codeRow.offsets[code + 1] - codeRow.offsets[code], 0, 0};
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
    size_t selfMergedSendSplitCount;
    bool hasTargetSplits;
    size_t alignGroupCount;
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
                    ? KmerCodeWriter(ownPairs.data, rows[row].offsets[code + 1], writePos)
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

struct KeyTargetAppend {
    bool hasEntry;
    std::vector<DBKeyType> &targets;

    void operator()(DBKeyType target) {
        if (hasEntry) {
            targets.push_back(target);
        }
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

// target split mode: file node * target splits + target split
struct TargetSplitFiles {
    std::vector<size_t> keyOffsets;
    std::vector<std::string> dataNames;
    std::vector<std::string> indexNames;
    std::vector<FILE *> dataFiles;
    std::vector<size_t> dataSizes;
    // (first key, data offset) of every entry, written when the files are closed
    std::vector<std::vector<uint64_t> > indexCells;
    double writeSeconds;
};

// the targets of a rep in one target split
struct RepTargetsPiece {
    size_t key;
    size_t targetCount;
    const DBKeyType *first;
    const DBKeyType *last;
};

// write an entry in every target split: rep key gap, size class, target gaps, 0
static void writeTargetSplits(const PrefFiles &prefFiles, TargetSplitFiles &files, DBReader<DBKeyType> &seqDbr, size_t firstKey, size_t keyCount, const size_t *keyOffsets,
                              const DBKeyType *targets, int threads, std::string &partitionsData) {
    Timer timer;
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    size_t splitCount = files.dataSizes.size();
    size_t entry = files.indexCells[0].size() / 2;
    size_t prefBegin = 0;
    for (size_t split = 0; split < splitCount; split++) {
        files.indexCells[split].push_back(firstKey);
        files.indexCells[split].push_back(files.dataSizes[split]);
        prefBegin += files.dataSizes[split];
    }
    std::vector<size_t> targetSplitCosts(splitCount, 0);
    // a task keeps its pieces in the second write buffer of a thread
    size_t pieceCapacity = KMER_FILE_BUFFER_SIZE / sizeof(RepTargetsPiece);
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
            RepTargetsPiece *pieces = (RepTargetsPiece *) (prefFiles.writeBuffers + (2 * task + 1) * KMER_FILE_BUFFER_SIZE);
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
                            RepTargetsPiece piece = { key, static_cast<size_t>(end - begin), first, last };
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
            DBKeyType splitFirstKey = files.keyOffsets[split];
            for (size_t task = 0; task < taskCount; task++) {
                const RepTargetsPiece *pieces = (const RepTargetsPiece *) (prefFiles.writeBuffers + (2 * task + 1) * KMER_FILE_BUFFER_SIZE);
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
                    targetSplitCosts[split] += (index[seqDbr.findIdRange(firstKey + pieces[i].key)].length - 2) * targetResidues;
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
    // a row per entry: first key, entry, pref bytes, cost, node
    size_t prefEnd = 0;
    size_t cost = 0;
    for (size_t split = 0; split < splitCount; split++) {
        prefEnd += files.dataSizes[split];
        cost += targetSplitCosts[split];
    }
    partitionsData += SSTR(firstKey) + "\t" + SSTR(entry) + "\t" + SSTR(prefEnd - prefBegin) + "\t" + SSTR(cost) + "\t" + SSTR(MMseqsNode::rank) + "\n";
    files.writeSeconds += timer.getTimediff();
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

// cuts [from, to) into key splits of at most kmerCapacity k-mers and keySplitKeyLimit keys, the representative bits start at dbFrom; returns the most k-mers of a key split
static size_t addKeySplits(const Parameters &par, DBReader<DBKeyType> &seqDbr, size_t dbFrom, size_t from, size_t to, size_t kmerCapacity, size_t keySplitKeyLimit, std::vector<size_t> &keySplitOffsets) {
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    float kmersPerSequenceScale = par.kmersPerSequenceScale.values.aminoacid();
    size_t firstRange = (to > from) ? seqDbr.findIdRange(from) : seqDbr.getIndexSize();
    size_t keySplitKmers = 0;
    size_t maxKeySplitKmers = 0;
    for (size_t range = firstRange; range < seqDbr.getIndexSize() && index[range].id < to; range++) {
        size_t rangeFrom = std::max(static_cast<size_t>(index[range].id), from);
        size_t rangeTo = std::min((range + 1 < seqDbr.getIndexSize()) ? static_cast<size_t>(index[range + 1].id) : seqDbr.getSize(), to);
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
    keySplitOffsets.push_back(to);
    return maxKeySplitKmers;
}

template <typename T, bool includeAdjacency, typename L>
static size_t sendKmers(Parameters &par, DBReader<DBKeyType> &seqDbr, BaseMatrix *subMat, const KmerExchange &exchange, MMseqsNode::WorkQueue &sendQueue,
                        size_t representativeBitsMemory, size_t &sentSplitCount, OwnKmers &ownKmers, ExchangeFileGroups &kmerGroups, double &extractSeconds) {
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
        size_t keySplitMemoryLimit = exchange.keySplitMemoryLimit - std::min(exchange.keySplitMemoryLimit, ownKmers.size + readBufferBytes);
        size_t kmerCapacity = keySplitMemoryLimit / sizeof(PackedKmerPosition);
        std::vector<size_t> keySplitOffsets;
        size_t maxKeySplitKmers = 0;
        size_t maxKeySplitKeys = 0;
        size_t groupBufferWords = 0;
        size_t fixedSize = 0;
        while (true) {
            keySplitOffsets.assign(1, dbFrom);
            maxKeySplitKmers = addKeySplits(par, seqDbr, dbFrom, dbFrom, dbFrom + dbSize, kmerCapacity, keySplitKeyLimit, keySplitOffsets);
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
            std::vector<size_t> readChunkOffsets = seqDbr.getSplitKeyOffsets(keySplitOffsets[keySplit], keySplitOffsets[keySplit + 1], readChunkLimit, 0);
            Timer timer;
            size_t kmerCount = fillKmerPositionArray<Parameters::DBTYPE_AMINO_ACIDS, T, includeAdjacency, L>(
                hashSeqPair, maxKeySplitKmers, seqDbr, par, subMat, true, 0, SIZE_MAX, NULL, readChunkOffsets).first;
            extractSeconds += timer.getTimediff();

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
                        ? KmerCodeWriter(ownKmers.data, hashSubRangeRow.offsets[code + 1], writePos)
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
    kmerGroups.writeWaitSeconds = batchWriter.getWaitSeconds();
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

// the keys of the ranges rep can cover, ranges are in length descending order
static std::pair<DBKeyType, DBKeyType> getCoveredKeys(const Parameters &par, DBReader<DBKeyType> &seqDbr, DBKeyType repKey) {
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    size_t repRange = seqDbr.findIdRange(repKey);
    float repLength = static_cast<float>(index[repRange].length - 2);
    CoverCheck isNotCovered = { par.covThr, par.covMode, repLength, false };
    CoverCheck isCovered = { par.covThr, par.covMode, repLength, true };
    const DBReader<DBKeyType>::Index *coveredFirstRange = std::partition_point(index, index + repRange, isNotCovered);
    const DBReader<DBKeyType>::Index *coveredEndRange = std::partition_point(index + repRange, index + seqDbr.getIndexSize(), isCovered);
    DBKeyType coveredTo = (coveredEndRange < index + seqDbr.getIndexSize()) ? coveredEndRange->id : static_cast<DBKeyType>(seqDbr.getSize());
    return std::make_pair(coveredFirstRange->id, coveredTo);
}

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
    PairFile pairFile;
    createFileGroups(pairGroups);
    pairFile.groups = &pairGroups;
    BatchWriter batchWriter(exchange.writeBehindSize, par.threads, fileno(pairGroups.files[0]));
    BatchReader batchReader(par.threads, fileno(pairGroups.files[0]));
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
                readPassRows(received.files, received.fileRows, passRows[passCodeBuffer], passBuffer + passCodeBuffer * codeBufferSize, false, batchReader);
            } else {
                batchReader.waitAsyncRead();
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
                readPassRows(prefetched->files, prefetched->fileRows, passRows[1 - passCodeBuffer], passBuffer + (1 - passCodeBuffer) * codeBufferSize, true, batchReader);
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
                                    repCoveredKeys.push_back(getCoveredKeys(par, seqDbr, repSeqIds[rep]));
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
    batchWriter.waitWrites();
    pairGroups.writeWaitSeconds = batchWriter.getWaitSeconds();
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

// runs of whole blocks, every align group has all target splits
static std::vector<size_t> getTargetSplitBlockOffsets(const KmerExchange &exchange, DBReader<DBKeyType> &seqDbr) {
    size_t alignNodeCount = (MMseqsNode::count - MMseqsNode::getFirstAlignNode() + exchange.alignGroupCount - 1) / exchange.alignGroupCount;
    size_t blockCount = seqDbr.getBlockKeyOffsets().size() - 1;
    size_t targetSplitCount = std::min(blockCount, MAX_TARGET_SPLITS_PER_NODE * alignNodeCount);
    std::vector<size_t> targetSplitBlockOffsets(targetSplitCount + 1);
    for (size_t split = 0; split <= targetSplitCount; split++) {
        targetSplitBlockOffsets[split] = split * blockCount / targetSplitCount;
    }
    return targetSplitBlockOffsets;
}

static std::string mergePairFiles(Parameters &par, DBReader<DBKeyType> &seqDbr, const KmerExchange &exchange, MMseqsNode::WorkQueue &mergeQueue, size_t &mergedSplitCount,
                                  OwnPairs &ownPairs, ExchangeFileGroups &pairGroups, std::string &progress, std::vector<size_t> &targetSplitBlockOffsets, size_t &prefSize,
                                  std::string &phaseValues) {
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
        targetSplitBlockOffsets = getTargetSplitBlockOffsets(exchange, seqDbr);
        size_t targetSplitCount = targetSplitBlockOffsets.size() - 1;
        targetSplitFiles.keyOffsets.resize(targetSplitCount + 1);
        for (size_t split = 0; split <= targetSplitCount; split++) {
            targetSplitFiles.keyOffsets[split] = seqDbr.getBlockKeyOffsets()[targetSplitBlockOffsets[split]];
        }
        for (size_t split = 0; split < targetSplitCount; split++) {
            size_t file = MMseqsNode::rank * targetSplitCount + split;
            targetSplitFiles.dataNames.push_back(par.db2 + "." + SSTR(file));
            targetSplitFiles.indexNames.push_back(par.db2 + ".index." + SSTR(file));
            targetSplitFiles.dataFiles.push_back(FileUtil::openAndDelete(targetSplitFiles.dataNames.back().c_str(), "w"));
        }
        targetSplitFiles.dataSizes.assign(targetSplitCount, 0);
        targetSplitFiles.indexCells.resize(targetSplitCount);
    } else {
        prefFiles.dataName = par.db2 + "." + SSTR(MMseqsNode::rank);
        prefFiles.indexName = par.db2 + ".index." + SSTR(MMseqsNode::rank);
        prefFiles.dataFile = FileUtil::openAndDelete(prefFiles.dataName.c_str(), "w");
        prefFiles.indexFile = FileUtil::openAndDelete(prefFiles.indexName.c_str(), "w");
    }
    BatchWriter batchWriter(exchange.writeBehindSize, par.threads, fileno(exchange.hasTargetSplits ? targetSplitFiles.dataFiles[0] : prefFiles.dataFile));
    BatchReader batchReader(par.threads, fileno(exchange.hasTargetSplits ? targetSplitFiles.dataFiles[0] : prefFiles.dataFile));
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
    targetSplitFiles.writeSeconds = 0;
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
    double openSeconds = 0;
    double passReadSeconds = 0;
    double passWaitSeconds = 0;
    double mergeSeconds = 0;
    size_t pieceCount = 0;
    size_t mergedPassCount = 0;
    size_t refilledPassCount = 0;
    for (size_t ownMergeRange = 0;; ownMergeRange++) {
        size_t splitBegin = 0;
        size_t splitEnd = 0;
        if (ownMergeRange < exchange.selfMergedSendSplitCount) {
            splitBegin = (MMseqsNode::rank + ownMergeRange * MMseqsNode::count) * exchange.partitionsPerRange;
            splitEnd = splitBegin + exchange.partitionsPerRange;
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
        }
        for (size_t pieceBegin = splitBegin, pieceEnd = splitBegin; pieceBegin < splitEnd; pieceBegin = pieceEnd) {
            size_t sendSplit = pieceBegin / exchange.partitionsPerRange;
            size_t fileGroup = pieceBegin / pairGroups.itemsPerGroup;
            pieceEnd = std::min(splitEnd, std::min((sendSplit + 1) * exchange.partitionsPerRange, getFileGroupItemEnd(pairGroups, fileGroup)));
            if (ownMergeRange >= exchange.selfMergedSendSplitCount && sendSplit / MMseqsNode::count < exchange.selfMergedSendSplitCount) {
                continue;
            }
            size_t firstPartition = pieceBegin - sendSplit * exchange.partitionsPerRange;
            size_t endPartition = pieceEnd - sendSplit * exchange.partitionsPerRange;
            if (sendSplit != bitsSendSplit) {
                setSendSplit(par, exchange, sendSplit);
                bitsSendSplit = sendSplit;
            }
            waitForFileGroup(pairGroups, fileGroup, true);
            Timer timer;
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
            batchReader.readSync(rowReadRequests[0]);
            openSeconds += timer.getTimediff();
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
                    refilledPassCount++;
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
            pieceCount++;
            mergedPassCount += pairPassCount;
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
                    timer.reset();
                    if (isPrefetched == false) {
                        getPairPassRows(partitionRows, passFirstPartition, passPartitionOffsets[pass + 1], pairPassRows[pass % 2], pairPassRowOffsets[pass % 2]);
                        readPassRows(receivedPairFiles, pairFileRows, pairPassRows[pass % 2], codeBuffer, false, batchReader);
                        passReadSeconds += timer.getTimediff();
                    } else {
                        batchReader.waitAsyncRead();
                        passWaitSeconds += timer.getTimediff();
                    }
                    isPrefetched = pass + 1 < pairPassCount && pairPassSizes[pass + 1] <= pairPassLimit && pairPassCodeSizes[pass + 1] > 0;
                    if (isPrefetched) {
                        getPairPassRows(partitionRows, passPartitionOffsets[pass + 1], passPartitionOffsets[pass + 2], pairPassRows[(pass + 1) % 2], pairPassRowOffsets[(pass + 1) % 2]);
                        readPassRows(receivedPairFiles, pairFileRows, pairPassRows[(pass + 1) % 2], pairPassBuffer + ((pass + 1) % 2) * maxPairPassCodeSize, true, batchReader);
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
                    timer.reset();
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
                    mergeSeconds += timer.getTimediff();
                    if (exchange.hasTargetSplits) {
                        // the partitions in key order, the target splits in parallel
                        for (size_t passPartition = 0; passPartition < passPartitions; passPartition++) {
                            const PartitionMerge &merge = partitionMerges[passPartition];
                            writeTargetSplits(prefFiles, targetSplitFiles, seqDbr, getPartitionFirstKey(exchange, sendSplit, passFirstPartition + passPartition), merge.keyOffsets.size() - 1,
                                              merge.keyOffsets.data(), merge.targets, par.threads, partitionsData);
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
                            timer.reset();
                            countPartitionTargets(exchange, receivedPairFiles, pairFileRows, partitionPassRows.data() + partitionPassRowOffsets[passPartition], partitionRowCount, codeBuffer,
                                                  partitionKeys, partitionTargets[partition], merge, mergeThreads);
                            mergePartitionTargets(exchange, seqDbr, partitionFirstKey, merge, pairPassTargets, mergeThreads);
                            mergeSeconds += timer.getTimediff();
                            if (exchange.hasTargetSplits) {
                                writeTargetSplits(prefFiles, targetSplitFiles, seqDbr, partitionFirstKey, partitionKeys, merge.keyOffsets.data(), merge.targets, par.threads, partitionsData);
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
                                // 16 MiB of keys and targets per entry
                                size_t chunkSize = KMER_FILE_BUFFER_SIZE / sizeof(DBKeyType);
                                std::vector<size_t> chunkKeyOffsets;
                                std::vector<DBKeyType> chunkTargets;
                                chunkKeyOffsets.reserve(chunkSize);
                                chunkTargets.reserve(chunkSize);
                                chunkKeyOffsets.push_back(0);
                                size_t chunkFirstKey = repKeyQueue.empty() ? partitionKeys : static_cast<size_t>(repKeyQueue.top().value);
                                for (size_t key = chunkFirstKey; key < partitionKeys;) {
                                    KeyTargetAppend targetAppend = { isRepresentative(partitionFirstKey + key), chunkTargets };
                                    mergeKeyTargets(reads, repKeyQueue, targetQueue, keyRows, key, exchange.globalIdBits, targetAppend);
                                    chunkKeyOffsets.push_back(chunkTargets.size());
                                    size_t nextKey = repKeyQueue.empty() ? partitionKeys : static_cast<size_t>(repKeyQueue.top().value);
                                    if (nextKey == partitionKeys || chunkTargets.size() + nextKey - chunkFirstKey >= chunkSize) {
                                        writeTargetSplits(prefFiles, targetSplitFiles, seqDbr, partitionFirstKey + chunkFirstKey, key + 1 - chunkFirstKey, chunkKeyOffsets.data(), chunkTargets.data(),
                                                          par.threads, partitionsData);
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
            std::vector<PassRow> splitRows;
            std::vector<size_t> splitRowOffsets;
            getPairPassRows(partitionRows, firstPartition, endPartition, splitRows, splitRowOffsets);
            punchPassRows(receivedPairFiles, pairFileRows, splitRows, releasedSize);
            closeReceivedFiles(receivedPairFiles);
            addFinishedItems(par, progress, PAIR_FILE_GROUPS, fileGroup, pieceEnd - pieceBegin);
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
    MMseqsNode::addPhaseSeconds(phaseValues, "opening pair pieces", openSeconds);
    MMseqsNode::addPhaseSeconds(phaseValues, "reading pair passes", passReadSeconds);
    MMseqsNode::addPhaseSeconds(phaseValues, "waiting for prefetched pair passes", passWaitSeconds);
    MMseqsNode::addPhaseSeconds(phaseValues, "merging partitions", mergeSeconds);
    MMseqsNode::addPhaseSeconds(phaseValues, "writing target splits", targetSplitFiles.writeSeconds);
    MMseqsNode::addPhaseSeconds(phaseValues, "waiting for prefilter writes summed over threads", batchWriter.getWaitSeconds());
    MMseqsNode::addPhaseValue(phaseValues, "Merge pieces", SSTR(pieceCount));
    MMseqsNode::addPhaseValue(phaseValues, "Merge passes", SSTR(mergedPassCount) + " (" + SSTR(refilledPassCount) + " refilled)");
    return partitionsData;
}

// MMSEQS_KMER_TARGET_PREF=1: every unit (a key split inside one target split) learns the reps of its k-mer groups and writes its own prefilter
// pass 1 sends fingerprints, their owners answer which are in other units; pass 2 sends the candidates of those groups, their owners answer
// the reps; pass 3 pairs every member with its reps and writes the unit file; the owners of the rep cells write the size classes
enum { KMER_GROUPS, WHOLE_SEQUENCE_GROUPS, GROUP_KINDS };

// a thread drops a pair it has just added, most pairs come from several k-mers
static const size_t RECENT_PAIR_SLOTS = 1 << 16;
// pass 3 threads claim the pair array in blocks
static const size_t PAIR_BLOCK_SIZE = 4096;

// a fingerprint keeps 8 bits more than the k-mer count needs, one in 2^8 collides with another k-mer and its group is sent in pass 2 for nothing
static const unsigned int FINGERPRINT_SPARE_BITS = 8;

struct PrefUnit {
    size_t firstKey;
    size_t endKey;
    unsigned int idBits;
};

// a node's exchange file of a step, as published at the barrier
struct PrefExchangeFile {
    size_t size;
    size_t tableOffset;
    size_t flushCount;
};

struct PieceCount {
    DBKeyType rep;
    size_t targetSplit;
    size_t targetCount;

    static bool compareByRep(const PieceCount &first, const PieceCount &second) {
        return first.rep < second.rep || (first.rep == second.rep && first.targetSplit < second.targetSplit);
    }

    static bool compareBySplit(const PieceCount &first, const PieceCount &second) {
        return first.targetSplit < second.targetSplit || (first.targetSplit == second.targetSplit && first.rep < second.rep);
    }
};

struct UnitKmers {
    UnitKmers() : kmers(NULL) {}
    PackedKmerPosition *kmers;
    std::vector<size_t> rangeOffsets[GROUP_KINDS];
};

// the groups of a unit that pass 3 needs, per hash range the sent k-mer and whole sequence groups, then those with another member in the unit
enum { SENT_KMER_GROUPS, SENT_WHOLE_SEQUENCE_GROUPS, LOCAL_KMER_GROUPS, LOCAL_WHOLE_SEQUENCE_GROUPS, GROUP_SEGMENTS };

// the needed groups of an own unit, a bit row per hash range (see putNeededRow), in memory, in the spill file in --local-tmp or else extracted again in pass 3
struct NeededGroups {
    NeededGroups() : spillOffset(SIZE_MAX) {}
    std::vector<char> data;
    size_t spillOffset;
    std::vector<size_t> rowOffsets;
};

struct TargetPref {
    size_t hashRangeCount;
    std::vector<size_t> hashRangeNodeOffsets;
    unsigned int codeShifts[GROUP_KINDS];
    unsigned int fingerprintShifts[GROUP_KINDS];
    std::vector<size_t> targetSplitBlockOffsets;
    std::vector<size_t> targetSplitKeyOffsets;
    std::vector<size_t> cellFirstKeys;
    std::vector<size_t> cellNodeOffsets;
    // own send splits in the order taken, with their units
    std::vector<size_t> sendSplits;
    std::vector<size_t> sendSplitUnitOffsets;
    // all units, node by node
    std::vector<PrefUnit> units;
    std::vector<size_t> nodeUnitOffsets;
    // unit files are numbered in key order
    std::vector<size_t> unitFiles;
    // own unit, hash range, kind
    std::vector<size_t> fingerprintCounts;
    // per range of the index the keys its sequences can cover
    std::vector<std::pair<DBKeyType, DBKeyType> > rangeCoveredKeys;
    size_t unitKmerCapacity;
    // per own unit its k-mers if all fit into unitKmerCapacity, else the units share extractedUnit
    std::vector<UnitKmers> keptUnits;
    size_t keptSize;
    bool isKeeping;
    UnitKmers extractedUnit;
    std::vector<NeededGroups> neededGroups;
    size_t neededSize;
    size_t neededLimit;
    // the spill file of the needed groups in --local-tmp, no spilling with localNodes 0
    std::string spillName;
    FILE *spillFile;
    size_t spillSize;
    size_t localNodes;
    size_t readChunkLimit;
    size_t passLimit;
    char *writeBuffers;
};

static size_t getOwnUnit(const TargetPref &pref, size_t unit) {
    return pref.nodeUnitOffsets[MMseqsNode::rank] + unit;
}

// the bytes held across the passes: kept k-mers, needed groups in memory, fingerprint counts and the extraction buffer
static size_t getHeldMemory(const TargetPref &pref) {
    return pref.keptSize * sizeof(PackedKmerPosition) + pref.neededSize + pref.fingerprintCounts.size() * sizeof(size_t)
           + ((pref.extractedUnit.kmers != NULL) ? pref.unitKmerCapacity * sizeof(PackedKmerPosition) : 0);
}

static void freeKeptUnits(TargetPref &pref) {
    for (size_t unit = 0; unit < pref.keptUnits.size(); unit++) {
        delete[] pref.keptUnits[unit].kmers;
        pref.keptUnits[unit].kmers = NULL;
    }
    pref.keptSize = 0;
}

static size_t getRiceSize(uint64_t value, unsigned int bits) {
    return (value >> bits) + 1 + bits;
}

static size_t getEliasGammaSize(size_t value) {
    return 2 * getBitWidth(value) - 1;
}

static unsigned int getRowLowBits(const TargetPref &pref, size_t hashRange, unsigned int shift, size_t groupCount) {
    return getHashDiffLowBits(hashRange, hashRange + 1, pref.hashRangeCount, shift, groupCount);
}

// rounded down, a fingerprint drops more bits than the hash
static uint64_t getRowFirstValue(const TargetPref &pref, size_t hashRange, unsigned int shift) {
    return static_cast<uint64_t>(((static_cast<PackedKmerPosition::uint128>(hashRange) << 64) / pref.hashRangeCount) >> shift);
}

static inline uint64_t getGroupValue(const PackedKmerPosition &kmer, unsigned int shift) {
    return static_cast<uint64_t>(kmer.getKmer()) >> shift;
}

static inline size_t getGroupEnd(const PackedKmerPosition *kmers, size_t groupStart, size_t to) {
    size_t groupEnd = groupStart + 1;
    while (groupEnd < to && kmers[groupEnd].getKmer() == kmers[groupStart].getKmer()) {
        groupEnd++;
    }
    return groupEnd;
}

static inline bool hasOtherMember(const PackedKmerPosition *kmers, size_t groupStart, size_t groupEnd) {
    for (size_t pos = groupStart + 1; pos < groupEnd; pos++) {
        if (kmers[pos].getId() != kmers[groupStart].getId()) {
            return true;
        }
    }
    return false;
}

static KmerCodeReader getMemoryCodeReader(char *data, size_t size) {
    KmerCodeReader reader = {data, 0, 0, 0, -1, NULL, size, 0, 0};
    return reader;
}

static void endCodeBits(KmerCodeWriter &writer) {
    putBits(writer, 0, (CHAR_BIT - writer.bitCount) % CHAR_BIT);
}

static std::string publishPrefExchangeFile(FILE *file, const std::string &fileName, std::vector<size_t> &flushOffsets, size_t columns, size_t size) {
    size_t flushCount = flushOffsets.size() / columns;
    size_t fileSize = MMseqsNode::writeExchangeTable(fileno(file), fileName, flushOffsets, columns, std::vector<size_t>(), size);
    MMseqsNode::publishExchangeFile(file, fileName, fileSize, "", std::vector<uint64_t>(1, 0));
    return SSTR(fileSize) + "\t" + SSTR(size) + "\t" + SSTR(flushCount) + "\n";
}

static std::vector<PrefExchangeFile> readPrefExchangeFiles(const std::string &barrierContent) {
    std::vector<PrefExchangeFile> files;
    for (char *line = (char *) barrierContent.c_str(); *line != '\0'; line = Util::skipLine(line)) {
        const char *columns[3];
        Util::getWordsOfLine(line, columns, 3);
        PrefExchangeFile file = { Util::fast_atoi<size_t>(columns[0]), Util::fast_atoi<size_t>(columns[1]), Util::fast_atoi<size_t>(columns[2]) };
        files.push_back(file);
    }
    return files;
}

static std::vector<int> openPrefExchangeFiles(const std::string &prefix, const std::vector<PrefExchangeFile> &files) {
    std::vector<int> fds(files.size());
    for (size_t node = 0; node < files.size(); node++) {
        fds[node] = MMseqsNode::waitForExchangeFile(prefix + "." + SSTR(node), files[node].size, 0);
    }
    return fds;
}

static void closePrefExchangeFiles(const std::string &prefix, std::vector<int> &fds) {
    for (size_t node = 0; node < fds.size(); node++) {
        if (close(fds[node]) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << prefix << "." << node << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
}

// the answers of all owners to an own unit, in hash range order
static void readUnitAnswers(const std::string &prefix, const std::vector<PrefExchangeFile> &files, const std::vector<int> &fds, size_t unit,
                            std::vector<char> &answers, BatchReader &batchReader) {
    std::vector<std::string> names(files.size());
    std::vector<std::vector<size_t> > offsets(files.size());
    size_t size = 0;
    for (size_t node = 0; node < files.size(); node++) {
        names[node] = prefix + "." + SSTR(node);
        MMseqsNode::readExchangeTableColumns(fds[node], names[node], files[node].tableOffset, files[node].flushCount, unit, offsets[node]);
        for (size_t flush = 0; flush < files[node].flushCount; flush++) {
            size += offsets[node][files[node].flushCount + flush] - offsets[node][flush];
        }
    }
    answers.resize(size + sizeof(uint64_t));
    std::vector<BatchReader::ReadRequest> readRequests;
    size_t bufferPos = 0;
    for (size_t node = 0; node < files.size(); node++) {
        for (size_t flush = 0; flush < files[node].flushCount; flush++) {
            size_t readSize = offsets[node][files[node].flushCount + flush] - offsets[node][flush];
            if (readSize > 0) {
                BatchReader::ReadRequest readRequest = {fds[node], names[node].c_str(), answers.data() + bufferPos, offsets[node][flush], readSize, readSize};
                readRequests.push_back(readRequest);
            }
            bufferPos += readSize;
        }
    }
    if (readRequests.empty() == false) {
        batchReader.readSync(readRequests);
    }
}

static void setUnitShifts(const KmerExchange &exchange, const PrefUnit &unit) {
    PackedKmerPosition::idShift = 0;
    PackedKmerPosition::seqLenShift = unit.idBits;
    PackedKmerPosition::adjKeyShift = unit.idBits;
    PackedKmerPosition::kmerShift = unit.idBits + exchange.adjKeyBits;
    PackedKmerPosition::maxSeqLen = 0;
}

template <typename T, bool includeAdjacency, typename L>
static void extractUnitKmers(Parameters &par, DBReader<DBKeyType> &seqDbr, BaseMatrix *subMat, const KmerExchange &exchange, const TargetPref &pref, const PrefUnit &unit,
                             UnitKmers &unitKmers, size_t capacity, double &extractSeconds) {
    setUnitShifts(exchange, unit);
    PackedKmerPosition *kmers = unitKmers.kmers;
    std::vector<size_t> readChunkOffsets = seqDbr.getSplitKeyOffsets(unit.firstKey, unit.endKey, pref.readChunkLimit, 0);
    Timer timer;
    size_t kmerCount = fillKmerPositionArray<Parameters::DBTYPE_AMINO_ACIDS, T, includeAdjacency, L>(kmers, capacity, seqDbr, par, subMat, true, 0, SIZE_MAX, NULL, readChunkOffsets).first;
    extractSeconds += timer.getTimediff();
    SORT_PARALLEL(kmers, kmers + kmerCount);
    size_t wholeSequenceStart = std::partition_point(kmers, kmers + kmerCount, isKmerRecord) - kmers;
    std::vector<size_t> *rangeOffsets = unitKmers.rangeOffsets;
    for (size_t kind = 0; kind < GROUP_KINDS; kind++) {
        rangeOffsets[kind].resize(pref.hashRangeCount + 1);
    }
#pragma omp parallel for schedule(static) num_threads(par.threads)
    for (size_t hashRange = 0; hashRange < pref.hashRangeCount; hashRange++) {
        uint64_t rangeStart = getRowFirstValue(pref, hashRange, 0);
        rangeOffsets[KMER_GROUPS][hashRange] = std::lower_bound(kmers, kmers + wholeSequenceStart, rangeStart, compareKmerHash) - kmers;
        rangeOffsets[WHOLE_SEQUENCE_GROUPS][hashRange] = std::lower_bound(kmers + wholeSequenceStart, kmers + kmerCount, rangeStart, compareKmerHash) - kmers;
    }
    rangeOffsets[KMER_GROUPS][pref.hashRangeCount] = wholeSequenceStart;
    rangeOffsets[WHOLE_SEQUENCE_GROUPS][pref.hashRangeCount] = kmerCount;
}

// the sorted k-mers of an own unit, kept since pass 1 or extracted again
template <typename T, bool includeAdjacency, typename L>
static const UnitKmers &getUnitKmers(Parameters &par, DBReader<DBKeyType> &seqDbr, BaseMatrix *subMat, const KmerExchange &exchange, TargetPref &pref, const PrefUnit &unit,
                                     size_t ownUnit, double &extractSeconds) {
    if (pref.keptUnits[ownUnit].kmers != NULL) {
        setUnitShifts(exchange, unit);
        return pref.keptUnits[ownUnit];
    }
    if (pref.extractedUnit.kmers == NULL) {
        pref.extractedUnit.kmers = new(std::nothrow) PackedKmerPosition[pref.unitKmerCapacity];
        Util::checkAllocation(pref.extractedUnit.kmers, "Can not allocate kmers memory in getUnitKmers");
    }
    extractUnitKmers<T, includeAdjacency, L>(par, seqDbr, subMat, exchange, pref, unit, pref.extractedUnit, pref.unitKmerCapacity, extractSeconds);
    return pref.extractedUnit;
}

// pass 1 keeps the k-mers of a unit while those of all units fit, else it frees them and every pass extracts again
template <typename T, bool includeAdjacency, typename L>
static const UnitKmers &extractNewUnitKmers(Parameters &par, DBReader<DBKeyType> &seqDbr, BaseMatrix *subMat, const KmerExchange &exchange, TargetPref &pref, const PrefUnit &unit,
                                            double &extractSeconds) {
    size_t ownUnit = pref.keptUnits.size();
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    float kmersPerSequenceScale = par.kmersPerSequenceScale.values.aminoacid();
    size_t maxKmerCount = 0;
    for (size_t range = seqDbr.findIdRange(unit.firstKey); range < seqDbr.getIndexSize() && index[range].id < unit.endKey; range++) {
        size_t rangeFrom = std::max(static_cast<size_t>(index[range].id), unit.firstKey);
        size_t rangeTo = std::min((range + 1 < seqDbr.getIndexSize()) ? static_cast<size_t>(index[range + 1].id) : seqDbr.getSize(), unit.endKey);
        maxKmerCount += countRepresentatives(sendSplitFirstKey, rangeFrom, rangeTo) * computeSeqKmerCount(static_cast<int>(index[range].length - 2), par.kmerSize, par.kmersPerSequence, kmersPerSequenceScale);
    }
    pref.keptUnits.push_back(UnitKmers());
    if (pref.isKeeping && pref.keptSize + maxKmerCount <= pref.unitKmerCapacity) {
        UnitKmers &kept = pref.keptUnits.back();
        kept.kmers = new(std::nothrow) PackedKmerPosition[maxKmerCount];
        Util::checkAllocation(kept.kmers, "Can not allocate kmers memory in extractNewUnitKmers");
        pref.keptSize += maxKmerCount;
        extractUnitKmers<T, includeAdjacency, L>(par, seqDbr, subMat, exchange, pref, unit, kept, maxKmerCount, extractSeconds);
        return kept;
    }
    if (pref.isKeeping) {
        freeKeptUnits(pref);
        pref.isKeeping = false;
    }
    return getUnitKmers<T, includeAdjacency, L>(par, seqDbr, subMat, exchange, pref, unit, ownUnit, extractSeconds);
}

// the rows of an own unit to its exchange file, a row per hash range of rowBits[hashRange] bits
template <typename WriteRow>
static void writeUnitRows(const Parameters &par, const TargetPref &pref, FILE *file, const std::string &fileName, size_t &fileSize, std::vector<size_t> &flushOffsets,
                          const std::vector<size_t> &rowBits, BatchWriter &batchWriter, WriteRow writeRow) {
    std::vector<size_t> rowOffsets(pref.hashRangeCount + 1);
    for (size_t hashRange = 0; hashRange <= pref.hashRangeCount; hashRange++) {
        rowOffsets[hashRange] = fileSize;
        flushOffsets.push_back(fileSize);
        if (hashRange < pref.hashRangeCount) {
            fileSize += (rowBits[hashRange] + CHAR_BIT - 1) / CHAR_BIT;
        }
    }
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
    for (size_t hashRange = 0; hashRange < pref.hashRangeCount; hashRange++) {
        if (rowBits[hashRange] == 0) {
            continue;
        }
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        KmerCodeWriter writer(fileno(file), fileName.c_str(), pref.writeBuffers + thread_idx * KMER_FILE_BUFFER_SIZE, KMER_FILE_BUFFER_SIZE, rowOffsets[hashRange], &batchWriter);
        writeRow(hashRange, writer);
        endCodeBits(writer);
        writeKmerCodeBuffer(writer);
    }
}

// pass 1: per row the k-mer and whole sequence group counts, then the fingerprint gaps of both kinds
template <typename T, bool includeAdjacency, typename L>
static std::string sendFingerprints(Parameters &par, DBReader<DBKeyType> &seqDbr, BaseMatrix *subMat, const KmerExchange &exchange, TargetPref &pref, MMseqsNode::WorkQueue &sendQueue,
                                    double &extractSeconds, size_t &sentSize) {
    const std::string fileName = par.db2 + ".fingerprints." + SSTR(MMseqsNode::rank);
    FILE *file = MMseqsNode::createExchangeFile(fileName, 0, "w");
    size_t fileSize = MMseqsNode::EXCHANGE_FILE_HEADER_BYTES;
    std::vector<size_t> flushOffsets;
    BatchWriter batchWriter(exchange.writeBehindSize, par.threads, fileno(file));
    std::string units;
    std::vector<size_t> rowBits(pref.hashRangeCount);
    // ids must fit next to the hash and the adjKey
    size_t keySplitKeyLimit = (1ULL << (128 - 1 - 64 - exchange.adjKeyBits)) - 1;
    pref.sendSplitUnitOffsets.assign(1, 0);
    std::vector<PrefUnit> ownUnits;
    for (size_t sendSplit = sendQueue.takeNextItem(); sendSplit < exchange.sendSplitCount; sendSplit = sendQueue.takeNextItem()) {
        setSendSplit(par, exchange, sendSplit);
        size_t dbFrom = exchange.sendSplitKeyOffsets[sendSplit];
        size_t dbTo = exchange.sendSplitKeyOffsets[sendSplit + 1];
        // a unit stays in one target split
        std::vector<size_t> keySplitOffsets(1, dbFrom);
        size_t split = std::upper_bound(pref.targetSplitKeyOffsets.begin(), pref.targetSplitKeyOffsets.end(), dbFrom) - pref.targetSplitKeyOffsets.begin() - 1;
        for (size_t from = dbFrom; from < dbTo; split++) {
            size_t to = std::min(dbTo, pref.targetSplitKeyOffsets[split + 1]);
            addKeySplits(par, seqDbr, dbFrom, from, to, pref.unitKmerCapacity, keySplitKeyLimit, keySplitOffsets);
            from = to;
        }
        pref.sendSplits.push_back(sendSplit);
        for (size_t keySplit = 0; keySplit + 1 < keySplitOffsets.size(); keySplit++) {
            if (keySplitOffsets[keySplit + 1] == keySplitOffsets[keySplit] || countRepresentatives(dbFrom, keySplitOffsets[keySplit], keySplitOffsets[keySplit + 1]) == 0) {
                continue;
            }
            PrefUnit unit = { keySplitOffsets[keySplit], keySplitOffsets[keySplit + 1], getBitWidth(keySplitOffsets[keySplit + 1] - keySplitOffsets[keySplit]) };
            ownUnits.push_back(unit);
            units += SSTR(unit.firstKey) + "\t" + SSTR(unit.endKey) + "\n";
            const UnitKmers &unitKmers = extractNewUnitKmers<T, includeAdjacency, L>(par, seqDbr, subMat, exchange, pref, unit, extractSeconds);
            const PackedKmerPosition *kmers = unitKmers.kmers;
            const std::vector<size_t> *rangeOffsets = unitKmers.rangeOffsets;
            size_t firstCount = pref.fingerprintCounts.size();
            pref.fingerprintCounts.resize(firstCount + pref.hashRangeCount * GROUP_KINDS);
            size_t *counts = pref.fingerprintCounts.data() + firstCount;
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
            for (size_t hashRange = 0; hashRange < pref.hashRangeCount; hashRange++) {
                size_t bits = 0;
                for (size_t kind = 0; kind < GROUP_KINDS; kind++) {
                    unsigned int shift = pref.fingerprintShifts[kind];
                    size_t from = rangeOffsets[kind][hashRange];
                    size_t to = rangeOffsets[kind][hashRange + 1];
                    size_t count = 0;
                    for (size_t pos = from; pos < to; pos++) {
                        count += (pos == from || getGroupValue(kmers[pos], shift) != getGroupValue(kmers[pos - 1], shift));
                    }
                    counts[hashRange * GROUP_KINDS + kind] = count;
                    unsigned int lowBits = getRowLowBits(pref, hashRange, shift, count);
                    uint64_t prev = getRowFirstValue(pref, hashRange, shift);
                    for (size_t pos = from; pos < to; pos++) {
                        uint64_t value = getGroupValue(kmers[pos], shift);
                        if (pos == from || value != getGroupValue(kmers[pos - 1], shift)) {
                            bits += getRiceSize((pos == from) ? value - prev : value - prev - 1, lowBits);
                            prev = value;
                        }
                    }
                }
                size_t groupCount = counts[hashRange * GROUP_KINDS] + counts[hashRange * GROUP_KINDS + 1];
                rowBits[hashRange] = (groupCount > 0) ? bits + getEliasGammaSize(counts[hashRange * GROUP_KINDS] + 1) + getEliasGammaSize(counts[hashRange * GROUP_KINDS + 1] + 1) : 0;
            }
            writeUnitRows(par, pref, file, fileName, fileSize, flushOffsets, rowBits, batchWriter, [&](size_t hashRange, KmerCodeWriter &writer) {
                putEliasGamma(writer, counts[hashRange * GROUP_KINDS] + 1);
                putEliasGamma(writer, counts[hashRange * GROUP_KINDS + 1] + 1);
                for (size_t kind = 0; kind < GROUP_KINDS; kind++) {
                    unsigned int shift = pref.fingerprintShifts[kind];
                    size_t from = rangeOffsets[kind][hashRange];
                    size_t to = rangeOffsets[kind][hashRange + 1];
                    unsigned int lowBits = getRowLowBits(pref, hashRange, shift, counts[hashRange * GROUP_KINDS + kind]);
                    uint64_t prev = getRowFirstValue(pref, hashRange, shift);
                    for (size_t pos = from; pos < to; pos++) {
                        uint64_t value = getGroupValue(kmers[pos], shift);
                        if (pos == from || value != getGroupValue(kmers[pos - 1], shift)) {
                            putRice(writer, (pos == from) ? value - prev : value - prev - 1, lowBits);
                            prev = value;
                        }
                    }
                }
            });
        }
        pref.sendSplitUnitOffsets.push_back(ownUnits.size());
    }
    seqDbr.releaseDataPages();
    batchWriter.waitWrites();
    sentSize = fileSize;
    return publishPrefExchangeFile(file, fileName, flushOffsets, pref.hashRangeCount + 1, fileSize) + units;
}

// a row of a sender unit in an owner's batch
struct PrefRowRead {
    KmerCodeReader reader;
    size_t unit;
    size_t groupCounts[GROUP_KINDS];
    size_t groupsLeft;
    uint64_t value;
    unsigned int lowBits;
    KmerCodeWriter answer;
};

// an owner reads the rows of its hash ranges from all units in batches and answers every row; answerBits bounds the answer of a row,
// with hasRowSizes an answer to a row with groups starts with its size. A batch and its rows take half of memoryLimit, the answers the rest
template <typename AnswerBits, typename AnswerHashRange>
static std::string answerHashRanges(const Parameters &par, const KmerExchange &exchange, const TargetPref &pref, const std::string &sentPrefix, const std::vector<PrefExchangeFile> &sentFiles,
                                    const std::string &answerPrefix, bool hasRowSizes, size_t memoryLimit, AnswerBits answerBits, AnswerHashRange answerHashRange,
                                    size_t &receivedSize, std::string &phaseValues) {
    size_t firstHashRange = pref.hashRangeNodeOffsets[MMseqsNode::rank];
    size_t ownHashRanges = pref.hashRangeNodeOffsets[MMseqsNode::rank + 1] - firstHashRange;
    size_t unitCount = pref.units.size();
    std::vector<int> fds = openPrefExchangeFiles(sentPrefix, sentFiles);
    std::vector<std::string> names(fds.size());
    // the offsets of the own hash ranges in every sender unit
    std::vector<size_t> rowOffsets(unitCount * (ownHashRanges + 1));
    for (size_t node = 0; node < fds.size(); node++) {
        names[node] = sentPrefix + "." + SSTR(node);
        size_t flushCount = sentFiles[node].flushCount;
        std::vector<size_t> table((ownHashRanges + 1) * flushCount);
        FileUtil::preadAll(fds[node], names[node].c_str(), (char *) table.data(), table.size() * sizeof(size_t), sentFiles[node].tableOffset + firstHashRange * flushCount * sizeof(size_t));
        for (size_t flush = 0; flush < flushCount; flush++) {
            for (size_t i = 0; i <= ownHashRanges; i++) {
                rowOffsets[(pref.nodeUnitOffsets[node] + flush) * (ownHashRanges + 1) + i] = table[i * flushCount + flush];
            }
        }
    }
    // per row its PrefRowRead, answer size, size field and offset
    std::vector<size_t> hashRangeSizes(ownHashRanges, unitCount * (sizeof(PrefRowRead) + 3 * sizeof(size_t)));
    for (size_t unit = 0; unit < unitCount; unit++) {
        for (size_t i = 0; i < ownHashRanges; i++) {
            hashRangeSizes[i] += rowOffsets[unit * (ownHashRanges + 1) + i + 1] - rowOffsets[unit * (ownHashRanges + 1) + i];
        }
    }
    // a thread merges the rows of a hash range in less than two PrefRowRead per unit
    size_t fixedMemory = rowOffsets.size() * sizeof(size_t) + 2 * par.threads * unitCount * sizeof(PrefRowRead);
    size_t ownerMemory = memoryLimit - std::min(memoryLimit, fixedMemory);
    std::vector<size_t> passOffsets;
    std::vector<size_t> passSizes;
    planPasses(hashRangeSizes, ownerMemory / 2, passOffsets, passSizes);
    const std::string fileName = answerPrefix + "." + SSTR(MMseqsNode::rank);
    FILE *file = MMseqsNode::createExchangeFile(fileName, 0, "w");
    size_t fileSize = MMseqsNode::EXCHANGE_FILE_HEADER_BYTES;
    std::vector<size_t> flushOffsets;
    BatchReader batchReader(par.threads, fds.empty() ? fileno(file) : fds[0]);
    // a sub pass is answered while the previous one is written
    BatchWriter batchWriter(exchange.writeBehindSize, par.threads, fileno(file));
    double readSeconds = 0;
    double answerSeconds = 0;
    double writeSeconds = 0;
    Timer timer;
    for (size_t pass = 0; pass + 1 < passOffsets.size(); pass++) {
        size_t passFirst = passOffsets[pass];
        size_t passEnd = passOffsets[pass + 1];
        size_t passHashRanges = passEnd - passFirst;
        // a read per sender unit, the rows of its hash ranges are adjacent
        std::vector<size_t> unitDataOffsets(unitCount + 1, 0);
        for (size_t unit = 0; unit < unitCount; unit++) {
            const size_t *offsets = rowOffsets.data() + unit * (ownHashRanges + 1);
            unitDataOffsets[unit + 1] = unitDataOffsets[unit] + offsets[passEnd] - offsets[passFirst];
        }
        timer.reset();
        std::vector<char> rowData(unitDataOffsets[unitCount] + sizeof(uint64_t));
        std::vector<BatchReader::ReadRequest> readRequests;
        for (size_t node = 0; node < fds.size(); node++) {
            for (size_t unit = pref.nodeUnitOffsets[node]; unit < pref.nodeUnitOffsets[node + 1]; unit++) {
                const size_t *offsets = rowOffsets.data() + unit * (ownHashRanges + 1);
                size_t readSize = offsets[passEnd] - offsets[passFirst];
                if (readSize > 0) {
                    BatchReader::ReadRequest readRequest = {fds[node], names[node].c_str(), rowData.data() + unitDataOffsets[unit], offsets[passFirst], readSize, readSize};
                    readRequests.push_back(readRequest);
                }
            }
        }
        if (readRequests.empty() == false) {
            batchReader.readSync(readRequests);
        }
        readSeconds += timer.getTimediff();
        receivedSize += unitDataOffsets[unitCount];
        // every row answers at most answerBits
        std::vector<PrefRowRead> rows(unitCount * passHashRanges);
        std::vector<size_t> answerSizes(unitCount * passHashRanges);
        std::vector<size_t> answerSizeBytes(unitCount * passHashRanges);
        std::vector<size_t> hashRangeAnswerSizes(passHashRanges, 0);
        for (size_t unit = 0; unit < unitCount; unit++) {
            const size_t *offsets = rowOffsets.data() + unit * (ownHashRanges + 1);
            for (size_t i = 0; i < passHashRanges; i++) {
                PrefRowRead &row = rows[unit * passHashRanges + i];
                row.unit = unit;
                row.groupCounts[KMER_GROUPS] = 0;
                row.groupCounts[WHOLE_SEQUENCE_GROUPS] = 0;
                if (offsets[passFirst + i + 1] > offsets[passFirst + i]) {
                    row.reader = getMemoryCodeReader(rowData.data() + unitDataOffsets[unit] + offsets[passFirst + i] - offsets[passFirst], offsets[passFirst + i + 1] - offsets[passFirst + i]);
                    row.groupCounts[KMER_GROUPS] = getEliasGamma(row.reader) - 1;
                    row.groupCounts[WHOLE_SEQUENCE_GROUPS] = getEliasGamma(row.reader) - 1;
                }
                size_t bits = answerBits(pref.units[unit], row.groupCounts);
                answerSizeBytes[unit * passHashRanges + i] = (hasRowSizes && row.groupCounts[KMER_GROUPS] + row.groupCounts[WHOLE_SEQUENCE_GROUPS] > 0) ? sizeof(uint64_t) : 0;
                answerSizes[unit * passHashRanges + i] = answerSizeBytes[unit * passHashRanges + i] + (bits + CHAR_BIT - 1) / CHAR_BIT;
                hashRangeAnswerSizes[i] += answerSizes[unit * passHashRanges + i];
            }
        }
        // the answers of the hash ranges in sub passes that fit next to the pass, a flush per sub pass
        std::vector<size_t> subPassOffsets;
        std::vector<size_t> subPassSizes;
        planPasses(hashRangeAnswerSizes, ownerMemory - std::min(ownerMemory, passSizes[pass]), subPassOffsets, subPassSizes);
        for (size_t subPass = 0; subPass + 1 < subPassOffsets.size(); subPass++) {
            timer.reset();
            size_t subPassFirst = subPassOffsets[subPass];
            size_t subPassHashRanges = subPassOffsets[subPass + 1] - subPassFirst;
            // answers unit by unit
            std::vector<size_t> answerOffsets(unitCount * subPassHashRanges + 1, 0);
            for (size_t unit = 0; unit < unitCount; unit++) {
                for (size_t i = 0; i < subPassHashRanges; i++) {
                    answerOffsets[unit * subPassHashRanges + i + 1] = answerOffsets[unit * subPassHashRanges + i] + answerSizes[unit * passHashRanges + subPassFirst + i];
                }
            }
            // not zero filled, only the pages of the written answers are touched
            char *answers = new(std::nothrow) char[answerOffsets.back() + sizeof(uint64_t)];
            Util::checkAllocation(answers, "Can not allocate answers memory in answerHashRanges");
            for (size_t unit = 0; unit < unitCount; unit++) {
                for (size_t i = 0; i < subPassHashRanges; i++) {
                    size_t row = unit * passHashRanges + subPassFirst + i;
                    size_t answer = unit * subPassHashRanges + i;
                    rows[row].answer = KmerCodeWriter(answers, answerOffsets[answer + 1], answerOffsets[answer] + answerSizeBytes[row]);
                }
            }
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
            for (size_t i = subPassFirst; i < subPassFirst + subPassHashRanges; i++) {
                std::vector<PrefRowRead *> hashRangeRows;
                for (size_t unit = 0; unit < unitCount; unit++) {
                    hashRangeRows.push_back(&rows[unit * passHashRanges + i]);
                }
                answerHashRange(firstHashRange + passFirst + i, hashRangeRows);
                for (size_t unit = 0; unit < unitCount; unit++) {
                    KmerCodeWriter &answer = hashRangeRows[unit]->answer;
                    endCodeBits(answer);
                    if (answerSizeBytes[unit * passHashRanges + i] > 0) {
                        uint64_t size = answer.bufferPos;
                        memcpy(answer.buffer - sizeof(uint64_t), &size, sizeof(uint64_t));
                    }
                }
            }
            answerSeconds += timer.getTimediff();
            timer.reset();
            // the answers of a unit to the hash ranges of the sub pass are adjacent in the file, moved down in the space of the unit and written behind
            size_t flushStart = flushOffsets.size();
            for (size_t unit = 0; unit < unitCount; unit++) {
                flushOffsets.push_back(fileSize);
                for (size_t i = 0; i < subPassHashRanges; i++) {
                    size_t row = unit * passHashRanges + subPassFirst + i;
                    fileSize += answerSizeBytes[row] + rows[row].answer.bufferPos;
                }
            }
            flushOffsets.push_back(fileSize);
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
            for (size_t unit = 0; unit < unitCount; unit++) {
                char *unitAnswers = answers + answerOffsets[unit * subPassHashRanges];
                size_t answerSize = 0;
                for (size_t i = 0; i < subPassHashRanges; i++) {
                    size_t row = unit * passHashRanges + subPassFirst + i;
                    size_t size = answerSizeBytes[row] + rows[row].answer.bufferPos;
                    memmove(unitAnswers + answerSize, answers + answerOffsets[unit * subPassHashRanges + i], size);
                    answerSize += size;
                }
                batchWriter.write(fileno(file), fileName.c_str(), unitAnswers, answerSize, flushOffsets[flushStart + unit]);
            }
            delete[] answers;
            writeSeconds += timer.getTimediff();
        }
    }
    timer.reset();
    batchWriter.waitWrites();
    writeSeconds += timer.getTimediff();
    closePrefExchangeFiles(sentPrefix, fds);
    std::string sent = sentPrefix.substr(par.db2.size() + 1);
    MMseqsNode::addPhaseSeconds(phaseValues, "reading " + sent, readSeconds);
    MMseqsNode::addPhaseSeconds(phaseValues, "answering " + sent, answerSeconds);
    MMseqsNode::addPhaseSeconds(phaseValues, "writing " + answerPrefix.substr(par.db2.size() + 1), writeSeconds);
    return publishPrefExchangeFile(file, fileName, flushOffsets, unitCount + 1, fileSize);
}

// the rows by value, smallest on top; the top row moves on to its next value in place
static void siftDownRowValue(std::vector<RowValue> &heap, size_t pos) {
    RowValue moved = heap[pos];
    for (size_t child = 2 * pos + 1; child < heap.size(); child = 2 * pos + 1) {
        if (child + 1 < heap.size() && heap[child] < heap[child + 1]) {
            child++;
        }
        if ((moved < heap[child]) == false) {
            break;
        }
        heap[pos] = heap[child];
        pos = child;
    }
    heap[pos] = moved;
}

// owner 1: a mark per fingerprint of a row, set if another unit has the fingerprint
static void markFingerprints(const TargetPref &pref, size_t hashRange, std::vector<PrefRowRead *> &rows) {
    std::vector<RowValue> heap;
    std::vector<unsigned int> valueRows;
    for (size_t kind = 0; kind < GROUP_KINDS; kind++) {
        unsigned int shift = pref.fingerprintShifts[kind];
        heap.clear();
        for (size_t row = 0; row < rows.size(); row++) {
            PrefRowRead &read = *rows[row];
            read.groupsLeft = read.groupCounts[kind];
            read.lowBits = getRowLowBits(pref, hashRange, shift, read.groupsLeft);
            read.value = getRowFirstValue(pref, hashRange, shift);
            if (read.groupsLeft > 0) {
                read.value += getRice(read.reader, read.lowBits);
                heap.push_back(RowValue(read.value, row));
            }
        }
        // RowValue orders a max heap by the smallest value
        std::make_heap(heap.begin(), heap.end());
        while (heap.empty() == false) {
            uint64_t value = heap[0].value;
            valueRows.clear();
            while (heap.empty() == false && heap[0].value == value) {
                PrefRowRead &read = *rows[heap[0].row];
                valueRows.push_back(heap[0].row);
                if (--read.groupsLeft > 0) {
                    read.value += getRice(read.reader, read.lowBits) + 1;
                    heap[0].value = read.value;
                } else {
                    heap[0] = heap.back();
                    heap.pop_back();
                }
                if (heap.empty() == false) {
                    siftDownRowValue(heap, 0);
                }
            }
            for (size_t i = 0; i < valueRows.size(); i++) {
                putBits(rows[valueRows[i]]->answer, valueRows.size() > 1, 1);
            }
        }
    }
}

// the marks of an own unit: per hash range the bits of its k-mer and its whole sequence fingerprints
struct UnitMarks {
    std::vector<char> data;
    std::vector<size_t> rowOffsets;

    void init(const TargetPref &pref, size_t ownUnit) {
        rowOffsets.assign(pref.hashRangeCount + 1, 0);
        const size_t *counts = pref.fingerprintCounts.data() + ownUnit * pref.hashRangeCount * GROUP_KINDS;
        for (size_t hashRange = 0; hashRange < pref.hashRangeCount; hashRange++) {
            rowOffsets[hashRange + 1] = rowOffsets[hashRange] + (counts[hashRange * GROUP_KINDS] + counts[hashRange * GROUP_KINDS + 1] + CHAR_BIT - 1) / CHAR_BIT;
        }
    }
};

// walks the groups of a row with the mark of their fingerprint
struct MarkedGroups {
    MarkedGroups(const PackedKmerPosition *kmers, size_t from, size_t to, unsigned int fingerprintShift, const KmerCodeReader &marks)
        : kmers(kmers), to(to), fingerprintShift(fingerprintShift), marks(marks), groupStart(from), groupEnd(from), isMarked(false), isFirst(true), fingerprint(0) {}

    bool next() {
        groupStart = groupEnd;
        if (groupStart == to) {
            return false;
        }
        groupEnd = getGroupEnd(kmers, groupStart, to);
        uint64_t groupFingerprint = getGroupValue(kmers[groupStart], fingerprintShift);
        if (isFirst || groupFingerprint != fingerprint) {
            isMarked = getBits(marks, 1);
            fingerprint = groupFingerprint;
            isFirst = false;
        }
        return true;
    }

    const PackedKmerPosition *kmers;
    size_t to;
    unsigned int fingerprintShift;
    KmerCodeReader marks;
    size_t groupStart;
    size_t groupEnd;
    bool isMarked;
    bool isFirst;
    uint64_t fingerprint;
};

// the [start, end) of the groups of a row that pass 3 needs, by GROUP_SEGMENTS
static void collectNeededGroups(const TargetPref &pref, const PackedKmerPosition *kmers, const std::vector<size_t> *rangeOffsets, const UnitMarks &marks, size_t hashRange,
                                std::vector<std::pair<size_t, size_t> > *neededGroups) {
    KmerCodeReader marksRead = getMemoryCodeReader((char *) marks.data.data() + marks.rowOffsets[hashRange], marks.rowOffsets[hashRange + 1] - marks.rowOffsets[hashRange]);
    for (size_t segment = 0; segment < GROUP_SEGMENTS; segment++) {
        neededGroups[segment].clear();
    }
    for (size_t kind = 0; kind < GROUP_KINDS; kind++) {
        MarkedGroups groups(kmers, rangeOffsets[kind][hashRange], rangeOffsets[kind][hashRange + 1], pref.fingerprintShifts[kind], marksRead);
        while (groups.next()) {
            if (groups.isMarked) {
                neededGroups[SENT_KMER_GROUPS + kind].push_back(std::make_pair(groups.groupStart, groups.groupEnd));
            } else if (hasOtherMember(kmers, groups.groupStart, groups.groupEnd)) {
                neededGroups[LOCAL_KMER_GROUPS + kind].push_back(std::make_pair(groups.groupStart, groups.groupEnd));
            }
        }
        marksRead = groups.marks;
    }
}

static void readUnitMarks(const TargetPref &pref, const std::string &prefix, const std::vector<PrefExchangeFile> &files, const std::vector<int> &fds, size_t ownUnit,
                          UnitMarks &marks, BatchReader &batchReader) {
    marks.init(pref, ownUnit);
    readUnitAnswers(prefix, files, fds, getOwnUnit(pref, ownUnit), marks.data, batchReader);
}

// pass 2 row: the counts of the sent k-mer and whole sequence groups, then per group its hash gap, a bit for another member in the unit and the
// candidates (first id of every adjKey); measures the row without writer
static size_t putCandidateRow(const KmerExchange &exchange, const TargetPref &pref, const PrefUnit &unit, const PackedKmerPosition *kmers,
                              const std::vector<std::pair<size_t, size_t> > *sentGroups, size_t hashRange, KmerCodeWriter *writer) {
    size_t sentCounts[GROUP_KINDS] = {sentGroups[KMER_GROUPS].size(), sentGroups[WHOLE_SEQUENCE_GROUPS].size()};
    if (sentCounts[KMER_GROUPS] + sentCounts[WHOLE_SEQUENCE_GROUPS] == 0) {
        return 0;
    }
    size_t bits = getEliasGammaSize(sentCounts[KMER_GROUPS] + 1) + getEliasGammaSize(sentCounts[WHOLE_SEQUENCE_GROUPS] + 1);
    if (writer != NULL) {
        putEliasGamma(*writer, sentCounts[KMER_GROUPS] + 1);
        putEliasGamma(*writer, sentCounts[WHOLE_SEQUENCE_GROUPS] + 1);
    }
    for (size_t kind = 0; kind < GROUP_KINDS; kind++) {
        unsigned int shift = pref.codeShifts[kind];
        unsigned int lowBits = getRowLowBits(pref, hashRange, shift, sentCounts[kind]);
        uint64_t prev = getRowFirstValue(pref, hashRange, shift);
        for (size_t group = 0; group < sentCounts[kind]; group++) {
            size_t groupStart = sentGroups[kind][group].first;
            size_t groupEnd = sentGroups[kind][group].second;
            uint64_t value = getGroupValue(kmers[groupStart], shift);
            uint64_t gap = (group == 0) ? value - prev : value - prev - 1;
            prev = value;
            size_t candidateCount = 1;
            for (size_t pos = groupStart + 1; kind == KMER_GROUPS && pos < groupEnd; pos++) {
                candidateCount += (kmers[pos].getAdjKey() != kmers[pos - 1].getAdjKey());
            }
            bits += getRiceSize(gap, lowBits) + 1 + candidateCount * unit.idBits;
            if (kind == KMER_GROUPS) {
                bits += getEliasGammaSize(candidateCount) + candidateCount * exchange.adjKeyCodeBits;
            }
            if (writer == NULL) {
                continue;
            }
            putRice(*writer, gap, lowBits);
            putBits(*writer, hasOtherMember(kmers, groupStart, groupEnd), 1);
            if (kind == KMER_GROUPS) {
                putEliasGamma(*writer, candidateCount);
            }
            for (size_t pos = groupStart; pos < groupEnd; pos++) {
                if (pos == groupStart || (kind == KMER_GROUPS && kmers[pos].getAdjKey() != kmers[pos - 1].getAdjKey())) {
                    if (kind == KMER_GROUPS) {
                        putAdjKeyCode(*writer, exchange, kmers[pos].getAdjKey());
                    }
                    putBits(*writer, kmers[pos].getId(), unit.idBits);
                }
            }
        }
    }
    return bits;
}

struct GroupCandidate {
    unsigned int adjKey;
    DBKeyType id;

    bool operator<(const GroupCandidate &other) const {
        return adjKey < other.adjKey || (adjKey == other.adjKey && id < other.id);
    }
};

// pass 3 row of a unit: the group counts by GROUP_SEGMENTS, per group its distinct member ids, the sent k-mer groups their candidates and the
// local k-mer groups their reps as member indices; a rep set depends only on the distinct candidate ids up to maxRepSeqCount, so only
// groups with more pick by adjKey
static size_t putNeededRow(const KmerExchange &exchange, const PrefUnit &unit, const PackedKmerPosition *kmers, const std::vector<std::pair<size_t, size_t> > *groups,
                           KmerCodeWriter *writer) {
    size_t groupCount = 0;
    for (size_t segment = 0; segment < GROUP_SEGMENTS; segment++) {
        groupCount += groups[segment].size();
    }
    if (groupCount == 0) {
        return 0;
    }
    size_t bits = 0;
    std::vector<DBKeyType> members;
    std::vector<GroupCandidate> candidates;
    std::vector<DBKeyType> reps;
    for (size_t segment = 0; segment < GROUP_SEGMENTS; segment++) {
        bits += getEliasGammaSize(groups[segment].size() + 1);
        if (writer != NULL) {
            putEliasGamma(*writer, groups[segment].size() + 1);
        }
    }
    for (size_t segment = 0; segment < GROUP_SEGMENTS; segment++) {
        for (size_t group = 0; group < groups[segment].size(); group++) {
            size_t groupStart = groups[segment][group].first;
            size_t groupEnd = groups[segment][group].second;
            members.clear();
            candidates.clear();
            bool isSorted = true;
            for (size_t pos = groupStart; pos < groupEnd; pos++) {
                members.push_back(kmers[pos].getId());
                isSorted &= (pos == groupStart || members[members.size() - 2] <= members.back());
                if (pos == groupStart || kmers[pos].getAdjKey() != kmers[pos - 1].getAdjKey()) {
                    GroupCandidate candidate = { kmers[pos].getAdjKey(), kmers[pos].getId() };
                    candidates.push_back(candidate);
                }
            }
            if (isSorted == false) {
                SORT_SERIAL(members.begin(), members.end());
            }
            members.erase(std::unique(members.begin(), members.end()), members.end());
            bits += getEliasGammaSize(members.size()) + members.size() * unit.idBits;
            if (writer != NULL) {
                putEliasGamma(*writer, members.size());
                for (size_t member = 0; member < members.size(); member++) {
                    putBits(*writer, members[member], unit.idBits);
                }
            }
            if (segment != SENT_KMER_GROUPS && segment != LOCAL_KMER_GROUPS) {
                continue;
            }
            reps.clear();
            if (segment == SENT_KMER_GROUPS) {
                for (size_t candidate = 0; candidate < candidates.size(); candidate++) {
                    reps.push_back(candidates[candidate].id);
                }
            } else {
                for (size_t candidate = 0; candidate < candidates.size(); candidate++) {
                    reps.push_back(candidates[candidate].id);
                }
                SORT_SERIAL(reps.begin(), reps.end());
                reps.erase(std::unique(reps.begin(), reps.end()), reps.end());
                if (reps.size() > exchange.maxRepSeqCount && writer == NULL) {
                    reps.resize(exchange.maxRepSeqCount);
                } else if (reps.size() > exchange.maxRepSeqCount) {
                    reps.clear();
                    selectGroupRepSeqs(reps, exchange.maxRepSeqCount, [&](const auto &visit) {
                        for (size_t j = 0; j < candidates.size(); j++) {
                            visit(candidates[j].adjKey, candidates[j].id);
                        }
                    }, std::less<DBKeyType>());
                }
            }
            unsigned int memberIndexBits = getBitWidth(members.size() - 1);
            bits += getEliasGammaSize(reps.size()) + reps.size() * memberIndexBits;
            if (writer != NULL) {
                putEliasGamma(*writer, reps.size());
                for (size_t rep = 0; rep < reps.size(); rep++) {
                    putBits(*writer, std::lower_bound(members.begin(), members.end(), reps[rep]) - members.begin(), memberIndexBits);
                }
            }
        }
    }
    return bits;
}

// the needed group rows of an own unit with neededBits[hashRange] bits: in memory while those of all units fit, else in the spill file, else none
static void storeNeededGroups(const Parameters &par, const KmerExchange &exchange, TargetPref &pref, NeededGroups &needed, const PrefUnit &unit,
                              const PackedKmerPosition *kmers, const std::vector<std::vector<std::pair<size_t, size_t> > > &neededGroups,
                              const std::vector<size_t> &neededBits, bool isPass3, BatchWriter &batchWriter) {
    std::vector<size_t> rowOffsets(pref.hashRangeCount + 1, 0);
    for (size_t hashRange = 0; hashRange < pref.hashRangeCount; hashRange++) {
        rowOffsets[hashRange + 1] = rowOffsets[hashRange] + (neededBits[hashRange] + CHAR_BIT - 1) / CHAR_BIT;
    }
    size_t size = rowOffsets.back();
    if (isPass3 == false && pref.neededSize + size > pref.neededLimit && pref.localNodes > 0) {
        // SIZE_MAX if statvfs failed
        size_t freeSize = FileUtil::getFreeSpace(par.localTmp.c_str());
        if (freeSize == SIZE_MAX || 2 * size * pref.localNodes > freeSize) {
            MMseqsNode::reportUnusableLocalTmp("Cannot keep the needed groups in --local-tmp " + par.localTmp + ": "
                                               + ((freeSize == SIZE_MAX) ? std::string("its free space is unknown") : "it has " + SSTR(freeSize) + " free bytes"));
            pref.localNodes = 0;
        } else {
            if (pref.spillFile == NULL) {
                pref.spillFile = FileUtil::openAndDelete(pref.spillName.c_str(), "w+");
            }
            needed.spillOffset = pref.spillSize;
            writeUnitRows(par, pref, pref.spillFile, pref.spillName, pref.spillSize, needed.rowOffsets, neededBits, batchWriter, [&](size_t hashRange, KmerCodeWriter &writer) {
                putNeededRow(exchange, unit, kmers, neededGroups.data() + hashRange * GROUP_SEGMENTS, &writer);
            });
            return;
        }
    }
    if (isPass3 == false && pref.neededSize + size > pref.neededLimit) {
        return;
    }
    pref.neededSize += isPass3 ? 0 : size;
    needed.rowOffsets.swap(rowOffsets);
    needed.data.resize(size + sizeof(uint64_t));
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
    for (size_t hashRange = 0; hashRange < pref.hashRangeCount; hashRange++) {
        if (neededBits[hashRange] == 0) {
            continue;
        }
        KmerCodeWriter writer(needed.data.data(), needed.rowOffsets[hashRange + 1], needed.rowOffsets[hashRange]);
        putNeededRow(exchange, unit, kmers, neededGroups.data() + hashRange * GROUP_SEGMENTS, &writer);
        endCodeBits(writer);
    }
}

template <typename T, bool includeAdjacency, typename L>
static std::string sendCandidates(Parameters &par, DBReader<DBKeyType> &seqDbr, BaseMatrix *subMat, const KmerExchange &exchange, TargetPref &pref,
                                  const std::vector<PrefExchangeFile> &markFiles, double &extractSeconds, size_t &sentSize) {
    const std::string fileName = par.db2 + ".candidates." + SSTR(MMseqsNode::rank);
    FILE *file = MMseqsNode::createExchangeFile(fileName, 0, "w");
    size_t fileSize = MMseqsNode::EXCHANGE_FILE_HEADER_BYTES;
    std::vector<size_t> flushOffsets;
    BatchWriter batchWriter(exchange.writeBehindSize, par.threads, fileno(file));
    std::vector<int> markFds = openPrefExchangeFiles(par.db2 + ".marks", markFiles);
    BatchReader batchReader(par.threads, fileno(file));
    std::vector<size_t> rowBits(pref.hashRangeCount);
    std::vector<size_t> neededBits(pref.hashRangeCount);
    std::vector<std::vector<std::pair<size_t, size_t> > > neededGroups(pref.hashRangeCount * GROUP_SEGMENTS);
    UnitMarks marks;
    pref.neededGroups.resize(pref.sendSplitUnitOffsets.back());
    for (size_t i = 0; i < pref.sendSplits.size(); i++) {
        setSendSplit(par, exchange, pref.sendSplits[i]);
        for (size_t ownUnit = pref.sendSplitUnitOffsets[i]; ownUnit < pref.sendSplitUnitOffsets[i + 1]; ownUnit++) {
            const PrefUnit &unit = pref.units[getOwnUnit(pref, ownUnit)];
            readUnitMarks(pref, par.db2 + ".marks", markFiles, markFds, ownUnit, marks, batchReader);
            const UnitKmers &unitKmers = getUnitKmers<T, includeAdjacency, L>(par, seqDbr, subMat, exchange, pref, unit, ownUnit, extractSeconds);
            const PackedKmerPosition *kmers = unitKmers.kmers;
            const std::vector<size_t> *rangeOffsets = unitKmers.rangeOffsets;
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
            for (size_t hashRange = 0; hashRange < pref.hashRangeCount; hashRange++) {
                collectNeededGroups(pref, kmers, rangeOffsets, marks, hashRange, neededGroups.data() + hashRange * GROUP_SEGMENTS);
                rowBits[hashRange] = putCandidateRow(exchange, pref, unit, kmers, neededGroups.data() + hashRange * GROUP_SEGMENTS, hashRange, NULL);
                neededBits[hashRange] = putNeededRow(exchange, unit, kmers, neededGroups.data() + hashRange * GROUP_SEGMENTS, NULL);
            }
            writeUnitRows(par, pref, file, fileName, fileSize, flushOffsets, rowBits, batchWriter, [&](size_t hashRange, KmerCodeWriter &writer) {
                putCandidateRow(exchange, pref, unit, kmers, neededGroups.data() + hashRange * GROUP_SEGMENTS, hashRange, &writer);
            });
            storeNeededGroups(par, exchange, pref, pref.neededGroups[ownUnit], unit, kmers, neededGroups, neededBits, false, batchWriter);
            if (pref.keptUnits[ownUnit].kmers != NULL) {
                delete[] pref.keptUnits[ownUnit].kmers;
                pref.keptUnits[ownUnit].kmers = NULL;
            }
        }
    }
    pref.keptSize = 0;
    seqDbr.releaseDataPages();
    batchWriter.waitWrites();
    closePrefExchangeFiles(par.db2 + ".marks", markFds);
    sentSize = fileSize;
    return publishPrefExchangeFile(file, fileName, flushOffsets, pref.hashRangeCount + 1, fileSize);
}

static size_t getRepAnswerBits(const KmerExchange &exchange, const size_t *groupCounts) {
    size_t repBits = 1 + std::max(exchange.globalIdBits, exchange.adjKeyCodeBits);
    return groupCounts[KMER_GROUPS] * (getEliasGammaSize(exchange.maxRepSeqCount + 1) + exchange.maxRepSeqCount * repBits)
           + groupCounts[WHOLE_SEQUENCE_GROUPS] * (getEliasGammaSize(2) + 1 + exchange.globalIdBits);
}

// owner 2: per group of a row the rep count, per rep a bit for the row's own candidate and its index, else the global id.
// The candidates of a group are sorted in the thread's write buffer if they fit, else the rows are merged again for every rep
static void pickReps(const KmerExchange &exchange, const TargetPref &pref, size_t hashRange, std::vector<PrefRowRead *> &rows) {
    unsigned int thread_idx = 0;
#ifdef OPENMP
    thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
    size_t candidateCapacity = KMER_FILE_BUFFER_SIZE / (2 * sizeof(GroupCandidate));
    GroupCandidate *candidates = (GroupCandidate *) (pref.writeBuffers + thread_idx * KMER_FILE_BUFFER_SIZE);
    GroupCandidate *sortedCandidates = candidates + candidateCapacity;
    std::vector<size_t> rowCandidateOffsets;
    std::vector<KmerCodeReader> candidateStarts;
    std::vector<std::pair<size_t, size_t> > rowOrder;
    std::vector<KmerCodeReader> candidateReads;
    std::vector<size_t> candidatesLeft;
    std::vector<size_t> repIndexes;
    RowValueQueue adjKeyQueue;
    std::vector<DBKeyType> reps;
    std::vector<unsigned int> valueRows;
    for (size_t kind = 0; kind < GROUP_KINDS; kind++) {
        unsigned int shift = pref.codeShifts[kind];
        RowValueQueue queue;
        for (size_t row = 0; row < rows.size(); row++) {
            PrefRowRead &read = *rows[row];
            read.groupsLeft = read.groupCounts[kind];
            read.lowBits = getRowLowBits(pref, hashRange, shift, read.groupsLeft);
            read.value = getRowFirstValue(pref, hashRange, shift);
            if (read.groupsLeft > 0) {
                read.value += getRice(read.reader, read.lowBits);
                queue.push(RowValue(read.value, row));
            }
        }
        while (queue.empty() == false) {
            uint64_t value = queue.top().value;
            valueRows.clear();
            while (queue.empty() == false && queue.top().value == value) {
                valueRows.push_back(queue.top().row);
                queue.pop();
            }
            bool hasTwoMembers = valueRows.size() > 1;
            rowCandidateOffsets.assign(1, 0);
            candidateStarts.clear();
            for (size_t i = 0; i < valueRows.size(); i++) {
                PrefRowRead &read = *rows[valueRows[i]];
                hasTwoMembers |= (getBits(read.reader, 1) != 0);
                size_t candidateCount = (kind == WHOLE_SEQUENCE_GROUPS) ? 1 : getEliasGamma(read.reader);
                rowCandidateOffsets.push_back(rowCandidateOffsets.back() + candidateCount);
                candidateStarts.push_back(read.reader);
            }
            reps.clear();
            if (rowCandidateOffsets.back() <= candidateCapacity) {
                for (size_t i = 0; i < valueRows.size(); i++) {
                    PrefRowRead &read = *rows[valueRows[i]];
                    const PrefUnit &unit = pref.units[read.unit];
                    for (size_t j = rowCandidateOffsets[i]; j < rowCandidateOffsets[i + 1]; j++) {
                        candidates[j].adjKey = (kind == WHOLE_SEQUENCE_GROUPS) ? 0 : getAdjKeyCode(read.reader, exchange);
                        candidates[j].id = unit.firstKey + getBits(read.reader, unit.idBits);
                    }
                }
                size_t candidateCount = rowCandidateOffsets.back();
                if (hasTwoMembers && kind == WHOLE_SEQUENCE_GROUPS) {
                    reps.push_back(std::min_element(candidates, candidates + candidateCount)->id);
                } else if (hasTwoMembers) {
                    // the first candidate of an adjKey has the smallest id
                    std::copy(candidates, candidates + candidateCount, sortedCandidates);
                    SORT_SERIAL(sortedCandidates, sortedCandidates + candidateCount);
                    selectGroupRepSeqs(reps, exchange.maxRepSeqCount, [&](const auto &visit) {
                        for (size_t j = 0; j < candidateCount; j++) {
                            visit(sortedCandidates[j].adjKey, sortedCandidates[j].id);
                        }
                    }, std::less<DBKeyType>());
                }
                for (size_t i = 0; i < valueRows.size(); i++) {
                    PrefRowRead &read = *rows[valueRows[i]];
                    putEliasGamma(read.answer, reps.size() + 1);
                    size_t first = rowCandidateOffsets[i];
                    size_t rowCandidateCount = rowCandidateOffsets[i + 1] - first;
                    for (size_t rep = 0; rep < reps.size(); rep++) {
                        size_t j = first;
                        while (j < first + rowCandidateCount && candidates[j].id != reps[rep]) {
                            j++;
                        }
                        putBits(read.answer, j < first + rowCandidateCount, 1);
                        if (j < first + rowCandidateCount) {
                            putBits(read.answer, j - first, getBitWidth(rowCandidateCount - 1));
                        } else {
                            putBits(read.answer, reps[rep], exchange.globalIdBits);
                        }
                    }
                }
            } else {
                if (hasTwoMembers && kind == WHOLE_SEQUENCE_GROUPS) {
                    DBKeyType rep = 0;
                    for (size_t i = 0; i < valueRows.size(); i++) {
                        const PrefUnit &unit = pref.units[rows[valueRows[i]]->unit];
                        KmerCodeReader candidateRead = candidateStarts[i];
                        DBKeyType id = unit.firstKey + getBits(candidateRead, unit.idBits);
                        rep = (i == 0) ? id : std::min(rep, id);
                    }
                    reps.push_back(rep);
                } else if (hasTwoMembers) {
                    // the rows of a group are units with disjoint ids and list their candidates by adjKey: merged in unit order they come by (adjKey, id)
                    rowOrder.clear();
                    for (size_t i = 0; i < valueRows.size(); i++) {
                        rowOrder.push_back(std::make_pair(pref.units[rows[valueRows[i]]->unit].firstKey, i));
                    }
                    SORT_SERIAL(rowOrder.begin(), rowOrder.end());
                    candidateReads.resize(rowOrder.size());
                    candidatesLeft.resize(rowOrder.size());
                    selectGroupRepSeqs(reps, exchange.maxRepSeqCount, [&](const auto &visit) {
                        for (size_t k = 0; k < rowOrder.size(); k++) {
                            size_t i = rowOrder[k].second;
                            candidateReads[k] = candidateStarts[i];
                            candidatesLeft[k] = rowCandidateOffsets[i + 1] - rowCandidateOffsets[i];
                            if (candidatesLeft[k] > 0) {
                                adjKeyQueue.push(RowValue(getAdjKeyCode(candidateReads[k], exchange), k));
                            }
                        }
                        while (adjKeyQueue.empty() == false) {
                            RowValue candidate = adjKeyQueue.top();
                            adjKeyQueue.pop();
                            const PrefUnit &unit = pref.units[rows[valueRows[rowOrder[candidate.row].second]]->unit];
                            visit(static_cast<unsigned int>(candidate.value), static_cast<DBKeyType>(unit.firstKey + getBits(candidateReads[candidate.row], unit.idBits)));
                            if (--candidatesLeft[candidate.row] > 0) {
                                adjKeyQueue.push(RowValue(getAdjKeyCode(candidateReads[candidate.row], exchange), candidate.row));
                            }
                        }
                    }, std::less<DBKeyType>());
                }
                for (size_t i = 0; i < valueRows.size(); i++) {
                    PrefRowRead &read = *rows[valueRows[i]];
                    const PrefUnit &unit = pref.units[read.unit];
                    size_t rowCandidateCount = rowCandidateOffsets[i + 1] - rowCandidateOffsets[i];
                    repIndexes.assign(reps.size(), SIZE_MAX);
                    for (size_t j = 0; j < rowCandidateCount; j++) {
                        if (kind == KMER_GROUPS) {
                            getAdjKeyCode(read.reader, exchange);
                        }
                        DBKeyType id = unit.firstKey + getBits(read.reader, unit.idBits);
                        for (size_t rep = 0; rep < reps.size(); rep++) {
                            if (repIndexes[rep] == SIZE_MAX && id == reps[rep]) {
                                repIndexes[rep] = j;
                            }
                        }
                    }
                    putEliasGamma(read.answer, reps.size() + 1);
                    for (size_t rep = 0; rep < reps.size(); rep++) {
                        putBits(read.answer, repIndexes[rep] != SIZE_MAX, 1);
                        if (repIndexes[rep] != SIZE_MAX) {
                            putBits(read.answer, repIndexes[rep], getBitWidth(rowCandidateCount - 1));
                        } else {
                            putBits(read.answer, reps[rep], exchange.globalIdBits);
                        }
                    }
                }
            }
            for (size_t i = 0; i < valueRows.size(); i++) {
                PrefRowRead &read = *rows[valueRows[i]];
                if (--read.groupsLeft > 0) {
                    read.value += getRice(read.reader, read.lowBits) + 1;
                    queue.push(RowValue(read.value, valueRows[i]));
                }
            }
        }
    }
}

// encodes whole rep cells of a unit file: per cell a 0 terminated entry, per rep the rep gap, the size class (0, the class files hold it), the target gaps and 0;
// pieces holds the piece counts of the cells, pieceCells their cells and starts
struct CellWriter {
    std::vector<char> data;
    std::vector<uint64_t> indexCells;
    size_t nextCell;
    bool isOpen;
    DBKeyType prevRep;
    std::vector<char> pieces;
    std::vector<std::pair<size_t, size_t> > pieceCells;
};

static void putPrefKey(std::vector<char> &data, uint64_t key) {
    char buffer[10];
    data.insert(data.end(), buffer, buffer + Util::putPrefKey(buffer, key));
}

// starts the cells up to cell, the cells before end empty
static void startCell(const TargetPref &pref, CellWriter &writer, size_t cell) {
    for (; writer.nextCell <= cell; writer.nextCell++) {
        if (writer.isOpen) {
            writer.data.push_back('\0');
        }
        writer.indexCells.push_back(pref.cellFirstKeys[writer.nextCell]);
        writer.indexCells.push_back(writer.data.size());
        writer.prevRep = pref.cellFirstKeys[writer.nextCell];
        writer.isOpen = true;
    }
}

static void closeCells(const TargetPref &pref, CellWriter &writer, size_t cellEnd) {
    if (writer.nextCell < cellEnd) {
        startCell(pref, writer, cellEnd - 1);
    }
    if (writer.isOpen) {
        writer.data.push_back('\0');
        writer.isOpen = false;
    }
}

static size_t getCell(const TargetPref &pref, DBKeyType key) {
    return std::upper_bound(pref.cellFirstKeys.begin(), pref.cellFirstKeys.end() - 1, key) - pref.cellFirstKeys.begin() - 1;
}

static bool compareClusterPairs(const ClusterPair &first, const ClusterPair &second) {
    return first.repKey < second.repKey || (first.repKey == second.repKey && first.memberKey < second.memberKey);
}

static bool isSameClusterPair(const ClusterPair &first, const ClusterPair &second) {
    return first.repKey == second.repKey && first.memberKey == second.memberKey;
}

// a part of a piece counts column: the piece bytes, the cost (rep length * target residues), the target split and the rep count, then per rep the rep gap and the target count
static void putPieceCounts(CellWriter &writer, size_t cell, size_t targetSplit, size_t pieceSize, size_t cost, size_t repCount, const std::vector<char> &repCounts) {
    writer.pieceCells.push_back(std::make_pair(cell, writer.pieces.size()));
    putPrefKey(writer.pieces, pieceSize);
    putPrefKey(writer.pieces, cost);
    putPrefKey(writer.pieces, targetSplit);
    putPrefKey(writer.pieces, repCount);
    writer.pieces.insert(writer.pieces.end(), repCounts.begin(), repCounts.end());
}

// the sorted unique pairs of a task, the writer continues its cells and adds a piece counts part per cell
static void writeTaskCells(const TargetPref &pref, DBReader<DBKeyType> &seqDbr, const PrefUnit &unit, const ClusterPair *pairs, size_t pairCount, CellWriter &writer) {
    writer.data.clear();
    writer.indexCells.clear();
    writer.pieces.clear();
    writer.pieceCells.clear();
    size_t targetSplit = std::upper_bound(pref.targetSplitKeyOffsets.begin(), pref.targetSplitKeyOffsets.end(), unit.firstKey) - pref.targetSplitKeyOffsets.begin() - 1;
    std::vector<char> repCounts;
    size_t countCell = SIZE_MAX;
    size_t pieceSize = 0;
    size_t cost = 0;
    size_t repCount = 0;
    DBKeyType prevCountRep = 0;
    for (size_t repStart = 0, repEnd = 0; repStart < pairCount; repStart = repEnd) {
        DBKeyType rep = pairs[repStart].repKey;
        for (repEnd = repStart + 1; repEnd < pairCount && pairs[repEnd].repKey == rep; repEnd++) {
        }
        size_t cell = getCell(pref, rep);
        if (cell != countCell) {
            if (repCount > 0) {
                putPieceCounts(writer, countCell, targetSplit, pieceSize, cost, repCount, repCounts);
            }
            countCell = cell;
            pieceSize = 0;
            cost = 0;
            repCount = 0;
            repCounts.clear();
            prevCountRep = pref.cellFirstKeys[cell];
        }
        startCell(pref, writer, cell);
        size_t pieceStart = writer.data.size();
        putPrefKey(writer.data, rep - writer.prevRep);
        putPrefKey(writer.data, 0);
        DBKeyType prevTarget = unit.firstKey - 1;
        size_t targetRange = SIZE_MAX;
        size_t targetResidues = 0;
        for (size_t i = repStart; i < repEnd; i++) {
            putPrefKey(writer.data, pairs[i].memberKey - prevTarget);
            prevTarget = pairs[i].memberKey;
            targetResidues += seqDbr.getSeqLenFromRange(targetRange, pairs[i].memberKey);
        }
        writer.data.push_back('\0');
        writer.prevRep = rep;
        pieceSize += writer.data.size() - pieceStart;
        cost += (seqDbr.getIndex()[seqDbr.findIdRange(rep)].length - 2) * targetResidues;
        repCount++;
        putPrefKey(repCounts, rep - prevCountRep);
        putPrefKey(repCounts, repEnd - repStart);
        prevCountRep = rep;
    }
    if (repCount > 0) {
        putPieceCounts(writer, countCell, targetSplit, pieceSize, cost, repCount, repCounts);
    }
}

static void appendCells(BatchWriter &batchWriter, FILE *dataFile, const std::string &dataName, const CellWriter &writer, std::vector<uint64_t> &indexCells, size_t &dataSize) {
    for (size_t i = 0; i < writer.indexCells.size(); i += 2) {
        indexCells.push_back(writer.indexCells[i]);
        indexCells.push_back(dataSize + writer.indexCells[i + 1]);
    }
    batchWriter.write(fileno(dataFile), dataName.c_str(), writer.data.data(), writer.data.size(), dataSize);
    dataSize += writer.data.size();
}

// pass 3: the reps of every group, from the owner if its fingerprint is in another unit, else from the unit; every member a target of its reps.
// Every thread keeps a run of pairs; tasks of rep cells merge the runs and encode in parallel; past the pair capacity a sweep ends at a cell and the unit is walked again
template <typename T, bool includeAdjacency, typename L>
static std::string writeUnitPrefs(Parameters &par, DBReader<DBKeyType> &seqDbr, BaseMatrix *subMat, const KmerExchange &exchange, TargetPref &pref,
                                  const std::vector<PrefExchangeFile> &markFiles, const std::vector<PrefExchangeFile> &repFiles,
                                  double &extractSeconds, size_t &prefSize, size_t &pairCount, size_t &sweepCount, std::string &phaseValues) {
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    const std::string fileName = par.db2 + ".piece_counts." + SSTR(MMseqsNode::rank);
    FILE *file = MMseqsNode::createExchangeFile(fileName, 0, "w");
    size_t fileSize = MMseqsNode::EXCHANGE_FILE_HEADER_BYTES;
    std::vector<size_t> flushOffsets;
    std::vector<int> markFds = openPrefExchangeFiles(par.db2 + ".marks", markFiles);
    std::vector<int> repFds = openPrefExchangeFiles(par.db2 + ".reps", repFiles);
    BatchReader batchReader(par.threads, fileno(file));
    UnitMarks marks;
    std::vector<char> repAnswers;
    std::vector<size_t> repRowOffsets(pref.hashRangeCount);
    size_t cellCount = pref.cellFirstKeys.size() - 1;
    std::vector<std::vector<ClusterPair> > threadRecentPairs(par.threads, std::vector<ClusterPair>(RECENT_PAIR_SLOTS));
    std::vector<CellWriter> taskWriters;
    double prepareSeconds = 0;
    double emitSeconds = 0;
    double sortSeconds = 0;
    double encodeSeconds = 0;
    double appendSeconds = 0;
    double waitSeconds = 0;
    Timer timer;
    BatchWriter batchWriter(exchange.writeBehindSize, par.threads, fileno(file));
    // the units with their needed groups in memory first, then the spilled ones, then those extracted again, which need the memory of the others
    std::vector<std::pair<size_t, size_t> > unitOrder;
    std::vector<size_t> unitSendSplits(pref.sendSplitUnitOffsets.back());
    for (size_t i = 0; i < pref.sendSplits.size(); i++) {
        for (size_t ownUnit = pref.sendSplitUnitOffsets[i]; ownUnit < pref.sendSplitUnitOffsets[i + 1]; ownUnit++) {
            const NeededGroups &needed = pref.neededGroups[ownUnit];
            unitSendSplits[ownUnit] = i;
            unitOrder.push_back(std::make_pair((needed.data.empty() == false) ? 0 : ((needed.spillOffset != SIZE_MAX) ? 1 : 2), ownUnit));
        }
    }
    SORT_SERIAL(unitOrder.begin(), unitOrder.end());
    std::vector<size_t> neededBits(pref.hashRangeCount);
    std::vector<std::vector<std::pair<size_t, size_t> > > extractedGroups;
    for (size_t order = 0; order < unitOrder.size(); order++) {
        size_t ownUnit = unitOrder[order].second;
        setSendSplit(par, exchange, pref.sendSplits[unitSendSplits[ownUnit]]);
        timer.reset();
        {
            size_t unitIndex = getOwnUnit(pref, ownUnit);
            const PrefUnit &unit = pref.units[unitIndex];
            NeededGroups &needed = pref.neededGroups[ownUnit];
            if (needed.data.empty() && needed.spillOffset != SIZE_MAX) {
                size_t first = needed.rowOffsets[0];
                needed.data.resize(needed.rowOffsets.back() - first + sizeof(uint64_t));
                std::vector<BatchReader::ReadRequest> readRequests;
                BatchReader::ReadRequest readRequest = {fileno(pref.spillFile), pref.spillName.c_str(), needed.data.data(), first, needed.rowOffsets.back() - first, needed.rowOffsets.back() - first};
                if (readRequest.bytesToRead > 0) {
                    readRequests.push_back(readRequest);
                    batchReader.readSync(readRequests);
                }
                for (size_t hashRange = 0; hashRange <= pref.hashRangeCount; hashRange++) {
                    needed.rowOffsets[hashRange] -= first;
                }
            } else if (needed.data.empty()) {
                readUnitMarks(pref, par.db2 + ".marks", markFiles, markFds, ownUnit, marks, batchReader);
                const UnitKmers &unitKmers = getUnitKmers<T, includeAdjacency, L>(par, seqDbr, subMat, exchange, pref, unit, ownUnit, extractSeconds);
                extractedGroups.resize(pref.hashRangeCount * GROUP_SEGMENTS);
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
                for (size_t hashRange = 0; hashRange < pref.hashRangeCount; hashRange++) {
                    collectNeededGroups(pref, unitKmers.kmers, unitKmers.rangeOffsets, marks, hashRange, extractedGroups.data() + hashRange * GROUP_SEGMENTS);
                    neededBits[hashRange] = putNeededRow(exchange, unit, unitKmers.kmers, extractedGroups.data() + hashRange * GROUP_SEGMENTS, NULL);
                }
                storeNeededGroups(par, exchange, pref, needed, unit, unitKmers.kmers, extractedGroups, neededBits, true, batchWriter);
            }
            readUnitAnswers(par.db2 + ".reps", repFiles, repFds, unitIndex, repAnswers, batchReader);
            // a row of the reps has a size if the unit sent a group
            for (size_t hashRange = 0, pos = 0; hashRange < pref.hashRangeCount; hashRange++) {
                bool hasSent = false;
                if (needed.rowOffsets[hashRange + 1] > needed.rowOffsets[hashRange]) {
                    KmerCodeReader neededRead = getMemoryCodeReader(needed.data.data() + needed.rowOffsets[hashRange], needed.rowOffsets[hashRange + 1] - needed.rowOffsets[hashRange]);
                    hasSent = getEliasGamma(neededRead) > 1;
                    hasSent |= getEliasGamma(neededRead) > 1;
                }
                uint64_t rowSize = 0;
                if (hasSent) {
                    memcpy(&rowSize, repAnswers.data() + pos, sizeof(uint64_t));
                    pos += sizeof(uint64_t);
                }
                repRowOffsets[hashRange] = pos;
                pos += rowSize;
            }
            if (unitOrder[order].first == 2) {
                // extracted again: the k-mers and read buffers are not needed any more
                delete[] pref.extractedUnit.kmers;
                pref.extractedUnit.kmers = NULL;
                seqDbr.releaseDataPages();
            }
            // the reps that can cover a key of the unit
            size_t firstRepRange = SIZE_MAX;
            size_t endRepRange = 0;
            for (size_t range = 0; range < seqDbr.getIndexSize(); range++) {
                if (pref.rangeCoveredKeys[range].first < unit.endKey && pref.rangeCoveredKeys[range].second > unit.firstKey) {
                    firstRepRange = std::min(firstRepRange, range);
                    endRepRange = range + 1;
                }
            }
            DBKeyType unitRepBegin = (firstRepRange < endRepRange) ? index[firstRepRange].id : 0;
            DBKeyType unitRepEnd = (firstRepRange < endRepRange) ? ((endRepRange < seqDbr.getIndexSize()) ? index[endRepRange].id : seqDbr.getSize()) : 0;
            std::string dataName = par.db2 + "." + SSTR(pref.unitFiles[unitIndex]);
            FILE *dataFile = FileUtil::openAndDelete(dataName.c_str(), "w");
            size_t dataSize = 0;
            std::vector<uint64_t> unitIndexCells;
            // the memory the passes do not hold takes the pairs of a sweep with their unit file bytes and piece counts, and the cells of a sweep with their piece count parts
            size_t usedMemory = getHeldMemory(pref) + ((unitOrder[order].first == 0) ? 0 : needed.data.size()) + repAnswers.size() + cellCount * (3 * sizeof(uint64_t) + 1 + 4 * getPrefKeySize(SIZE_MAX));
            size_t pairBytes = sizeof(ClusterPair) + 4 * getPrefKeySize(seqDbr.getSize()) + 2;
            // a rep keeps all its pairs of the unit in one sweep
            size_t pairCapacity = std::max((exchange.keySplitMemoryLimit - std::min(exchange.keySplitMemoryLimit, usedMemory)) / pairBytes, 2 * (unit.endKey - unit.firstKey));
            // chunks of hash ranges whose groups cannot add more than half of the capacity
            std::vector<size_t> chunkHashRanges(1, 0);
            std::vector<size_t> chunkPairBounds(1, 0);
            for (size_t hashRange = 0; hashRange < pref.hashRangeCount; hashRange++) {
                // a member takes at least idBits of a row
                size_t records = (needed.rowOffsets[hashRange + 1] - needed.rowOffsets[hashRange]) * CHAR_BIT / std::max(1u, unit.idBits);
                if (hashRange > chunkHashRanges.back() && chunkPairBounds.back() + records * exchange.maxRepSeqCount > pairCapacity / 2) {
                    chunkHashRanges.push_back(hashRange);
                    chunkPairBounds.push_back(0);
                }
                chunkPairBounds.back() += records * exchange.maxRepSeqCount;
            }
            chunkHashRanges.push_back(pref.hashRangeCount);
            size_t arraySize = pairCapacity / 2 + *std::max_element(chunkPairBounds.begin(), chunkPairBounds.end()) + par.threads * PAIR_BLOCK_SIZE;
            ClusterPair *pairs = new(std::nothrow) ClusterPair[arraySize];
            Util::checkAllocation(pairs, "Can not allocate pairs memory in writeUnitPrefs");
            const ClusterPair sentinel = { DB_KEY_INVALID, DB_KEY_INVALID };
            // the cells are written in order, a sweep may end and the next start inside a cell
            CellWriter unitWriter;
            unitWriter.nextCell = 0;
            unitWriter.isOpen = false;
            unitWriter.prevRep = 0;
            // the unit's piece counts are a flush, its parts are appended in cell order
            size_t countColumns = 0;
            prepareSeconds += timer.getTimediff();
            for (DBKeyType repBegin = unitRepBegin; repBegin < unitRepEnd; sweepCount++) {
                // a pair dropped at the end of a sweep may come again in the next
                for (size_t thread = 0; thread < threadRecentPairs.size(); thread++) {
                    std::fill(threadRecentPairs[thread].begin(), threadRecentPairs[thread].end(), sentinel);
                }
                DBKeyType repEnd = unitRepEnd;
                size_t pairEnd = 0;
                for (size_t chunk = 0; chunk + 1 < chunkHashRanges.size(); chunk++) {
                    if (pairEnd + chunkPairBounds[chunk + 1] + par.threads * PAIR_BLOCK_SIZE > arraySize) {
                        timer.reset();
                        SORT_PARALLEL(pairs, pairs + pairEnd, compareClusterPairs);
                        pairEnd = std::unique(pairs, std::lower_bound(pairs, pairs + pairEnd, sentinel, compareClusterPairs), isSameClusterPair) - pairs;
                        // the sweep ends at the rep where its pairs reach half of the capacity, but keeps one rep
                        ClusterPair firstRepEnd = { pairs[0].repKey, DB_KEY_INVALID };
                        const ClusterPair *cut = (pairEnd > pairCapacity / 2) ? std::max(pairs + pairCapacity / 2, std::upper_bound(pairs, pairs + pairEnd, firstRepEnd, compareClusterPairs)) : pairs + pairEnd;
                        if (cut < pairs + pairEnd) {
                            repEnd = cut->repKey;
                            ClusterPair end = { repEnd, 0 };
                            pairEnd = std::lower_bound(pairs, pairs + pairEnd, end, compareClusterPairs) - pairs;
                        }
                        sortSeconds += timer.getTimediff();
                    }
                    timer.reset();
                    size_t chunkStart = chunkHashRanges[chunk];
                    size_t chunkEnd = chunkHashRanges[chunk + 1];
#pragma omp parallel num_threads(par.threads)
                    {
                        unsigned int thread_idx = 0;
#ifdef OPENMP
                        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
                        // the thread claims blocks of the pair array, the rest of its last block is filled with the sentinel
                        size_t blockPos = 0;
                        size_t blockEnd = 0;
                        ClusterPair *recentPairs = threadRecentPairs[thread_idx].data();
                        std::vector<DBKeyType> reps;
                        std::vector<DBKeyType> members;
                        std::vector<size_t> candidateMembers;
                        DBKeyType coveredRep = DB_KEY_INVALID;
                        std::pair<DBKeyType, DBKeyType> coveredKeys;
#pragma omp for schedule(dynamic, 1)
                        for (size_t hashRange = chunkStart; hashRange < chunkEnd; hashRange++) {
                            if (needed.rowOffsets[hashRange + 1] == needed.rowOffsets[hashRange]) {
                                continue;
                            }
                            KmerCodeReader neededRead = getMemoryCodeReader(needed.data.data() + needed.rowOffsets[hashRange], needed.rowOffsets[hashRange + 1] - needed.rowOffsets[hashRange]);
                            KmerCodeReader repRead = getMemoryCodeReader(repAnswers.data() + repRowOffsets[hashRange], repAnswers.size() - repRowOffsets[hashRange]);
                            size_t segmentGroups[GROUP_SEGMENTS];
                            for (size_t segment = 0; segment < GROUP_SEGMENTS; segment++) {
                                segmentGroups[segment] = getEliasGamma(neededRead) - 1;
                            }
                            for (size_t segment = 0; segment < GROUP_SEGMENTS; segment++) {
                                for (size_t group = 0; group < segmentGroups[segment]; group++) {
                                    members.resize(getEliasGamma(neededRead));
                                    for (size_t member = 0; member < members.size(); member++) {
                                        members[member] = unit.firstKey + getBits(neededRead, unit.idBits);
                                    }
                                    unsigned int memberIndexBits = getBitWidth(members.size() - 1);
                                    candidateMembers.assign(1, 0);
                                    if (segment == SENT_KMER_GROUPS || segment == LOCAL_KMER_GROUPS) {
                                        candidateMembers.resize(getEliasGamma(neededRead));
                                        for (size_t candidate = 0; candidate < candidateMembers.size(); candidate++) {
                                            candidateMembers[candidate] = getBits(neededRead, memberIndexBits);
                                        }
                                    }
                                    reps.clear();
                                    if (segment < LOCAL_KMER_GROUPS) {
                                        // the owner's answer: per rep the own candidate or the global id
                                        size_t repCount = getEliasGamma(repRead) - 1;
                                        for (size_t rep = 0; rep < repCount; rep++) {
                                            bool isOwn = getBits(repRead, 1) != 0;
                                            reps.push_back(isOwn ? members[candidateMembers[getBits(repRead, getBitWidth(candidateMembers.size() - 1))]] : getBits(repRead, exchange.globalIdBits));
                                        }
                                    } else {
                                        for (size_t rep = 0; rep < candidateMembers.size(); rep++) {
                                            reps.push_back(members[candidateMembers[rep]]);
                                        }
                                    }
                                    for (size_t rep = 0; rep < reps.size(); rep++) {
                                        if (reps[rep] < repBegin || reps[rep] >= repEnd) {
                                            continue;
                                        }
                                        if (reps[rep] != coveredRep) {
                                            coveredRep = reps[rep];
                                            coveredKeys = pref.rangeCoveredKeys[seqDbr.findIdRange(coveredRep)];
                                        }
                                        for (size_t member = 0; member < members.size(); member++) {
                                            if (members[member] < coveredKeys.first || members[member] >= coveredKeys.second || members[member] == reps[rep]) {
                                                continue;
                                            }
                                            ClusterPair pair = { reps[rep], members[member] };
                                            ClusterPair &recent = recentPairs[(reps[rep] * 0x9E3779B97F4A7C15ULL + members[member]) & (RECENT_PAIR_SLOTS - 1)];
                                            if (isSameClusterPair(recent, pair)) {
                                                continue;
                                            }
                                            recent = pair;
                                            if (blockPos == blockEnd) {
                                                blockPos = __sync_fetch_and_add(&pairEnd, PAIR_BLOCK_SIZE);
                                                blockEnd = blockPos + PAIR_BLOCK_SIZE;
                                            }
                                            pairs[blockPos++] = pair;
                                        }
                                    }
                                }
                            }
                        }
                        std::fill(pairs + blockPos, pairs + blockEnd, sentinel);
                    }
                    emitSeconds += timer.getTimediff();
                }
                timer.reset();
                SORT_PARALLEL(pairs, pairs + pairEnd, compareClusterPairs);
                pairEnd = std::lower_bound(pairs, pairs + pairEnd, sentinel, compareClusterPairs) - pairs;
                // tasks of whole cells with about equal pairs, the first may continue and the last end inside a cell
                size_t firstInnerCell = getCell(pref, repBegin) + 1;
                size_t endCell = std::lower_bound(pref.cellFirstKeys.begin(), pref.cellFirstKeys.end(), static_cast<size_t>(repEnd)) - pref.cellFirstKeys.begin();
                size_t taskCount = std::max((size_t) 1, std::min(endCell - std::min(endCell, firstInnerCell) + 1, TASKS_PER_THREAD * par.threads));
                std::vector<DBKeyType> taskReps(1, repBegin);
                for (size_t task = 1; task < taskCount; task++) {
                    size_t low = std::max(firstInnerCell, getCell(pref, taskReps.back()) + 1);
                    size_t high = endCell;
                    while (low < high) {
                        size_t cell = low + (high - low) / 2;
                        ClusterPair first = { static_cast<DBKeyType>(pref.cellFirstKeys[cell]), 0 };
                        if (static_cast<size_t>(std::lower_bound(pairs, pairs + pairEnd, first, compareClusterPairs) - pairs) < task * pairEnd / taskCount) {
                            low = cell + 1;
                        } else {
                            high = cell;
                        }
                    }
                    if (low < endCell && pref.cellFirstKeys[low] < repEnd) {
                        taskReps.push_back(pref.cellFirstKeys[low]);
                    }
                }
                taskReps.push_back(repEnd);
                taskCount = taskReps.size() - 1;
                taskWriters.resize(taskCount);
                sortSeconds += timer.getTimediff();
                timer.reset();
                size_t sweepPairCount = 0;
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads) reduction(+:sweepPairCount)
                for (size_t task = 0; task < taskCount; task++) {
                    ClusterPair first = { taskReps[task], 0 };
                    ClusterPair end = { taskReps[task + 1], 0 };
                    ClusterPair *taskPairs = std::lower_bound(pairs, pairs + pairEnd, first, compareClusterPairs);
                    size_t taskPairCount = std::unique(taskPairs, std::lower_bound(taskPairs, pairs + pairEnd, end, compareClusterPairs), isSameClusterPair) - taskPairs;
                    CellWriter &writer = taskWriters[task];
                    writer.nextCell = (task == 0) ? unitWriter.nextCell : getCell(pref, taskReps[task]);
                    writer.isOpen = (task == 0) ? unitWriter.isOpen : false;
                    writer.prevRep = (task == 0) ? unitWriter.prevRep : taskReps[task];
                    writeTaskCells(pref, seqDbr, unit, taskPairs, taskPairCount, writer);
                    sweepPairCount += taskPairCount;
                    size_t taskEndCell = std::lower_bound(pref.cellFirstKeys.begin(), pref.cellFirstKeys.end(), static_cast<size_t>(taskReps[task + 1])) - pref.cellFirstKeys.begin();
                    if (pref.cellFirstKeys[taskEndCell] == taskReps[task + 1]) {
                        closeCells(pref, writer, taskEndCell);
                    } else {
                        startCell(pref, writer, taskEndCell - 1);
                    }
                }
                pairCount += sweepPairCount;
                encodeSeconds += timer.getTimediff();
                timer.reset();
                // the tasks follow each other in the unit file and the piece counts, the threads copy them to the writer
                std::vector<size_t> taskDataOffsets(taskCount);
                std::vector<size_t> taskPieceOffsets(taskCount);
                for (size_t task = 0; task < taskCount; task++) {
                    const CellWriter &writer = taskWriters[task];
                    for (size_t i = 0; i < writer.indexCells.size(); i += 2) {
                        unitIndexCells.push_back(writer.indexCells[i]);
                        unitIndexCells.push_back(dataSize + writer.indexCells[i + 1]);
                    }
                    for (size_t i = 0; i < writer.pieceCells.size(); i++) {
                        for (; countColumns <= writer.pieceCells[i].first; countColumns++) {
                            flushOffsets.push_back(fileSize + writer.pieceCells[i].second);
                        }
                    }
                    taskDataOffsets[task] = dataSize;
                    taskPieceOffsets[task] = fileSize;
                    dataSize += writer.data.size();
                    fileSize += writer.pieces.size();
                }
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
                for (size_t task = 0; task < taskCount; task++) {
                    CellWriter &writer = taskWriters[task];
                    batchWriter.write(fileno(dataFile), dataName.c_str(), writer.data.data(), writer.data.size(), taskDataOffsets[task]);
                    batchWriter.write(fileno(file), fileName.c_str(), writer.pieces.data(), writer.pieces.size(), taskPieceOffsets[task]);
                    std::vector<char>().swap(writer.data);
                    std::vector<char>().swap(writer.pieces);
                }
                appendSeconds += timer.getTimediff();
                unitWriter.nextCell = taskWriters[taskCount - 1].nextCell;
                unitWriter.isOpen = taskWriters[taskCount - 1].isOpen;
                unitWriter.prevRep = taskWriters[taskCount - 1].prevRep;
                repBegin = repEnd;
            }
            delete[] pairs;
            for (; countColumns <= cellCount; countColumns++) {
                flushOffsets.push_back(fileSize);
            }
            unitWriter.data.clear();
            unitWriter.indexCells.clear();
            closeCells(pref, unitWriter, cellCount);
            appendCells(batchWriter, dataFile, dataName, unitWriter, unitIndexCells, dataSize);
            // the unit file is closed once its writes are done
            timer.reset();
            batchWriter.waitWrites();
            waitSeconds += timer.getTimediff();
            std::string indexName = par.db2 + ".index." + SSTR(pref.unitFiles[unitIndex]);
            FILE *indexFile = FileUtil::openAndDelete(indexName.c_str(), "w");
            size_t indexSize = unitIndexCells.size() * sizeof(uint64_t);
            FileUtil::pwriteAll(fileno(indexFile), indexName.c_str(), (const char *) unitIndexCells.data(), indexSize, 0);
            MMseqsNode::closeClusterFiles(dataFile, dataName, indexFile, indexName, dataSize, indexSize);
            prefSize += dataSize + indexSize + 2 * sizeof(uint64_t);
            pref.neededSize -= (unitOrder[order].first == 0) ? needed.rowOffsets.back() : 0;
            std::vector<char>().swap(needed.data);
        }
    }
    if (flushOffsets.empty()) {
        flushOffsets.assign(cellCount + 1, fileSize);
    }
    closePrefExchangeFiles(par.db2 + ".marks", markFds);
    closePrefExchangeFiles(par.db2 + ".reps", repFds);
    MMseqsNode::addPhaseSeconds(phaseValues, "pass 3 preparing units", prepareSeconds);
    MMseqsNode::addPhaseSeconds(phaseValues, "pass 3 emitting pairs", emitSeconds);
    MMseqsNode::addPhaseSeconds(phaseValues, "pass 3 sorting pairs", sortSeconds);
    MMseqsNode::addPhaseSeconds(phaseValues, "pass 3 encoding cells", encodeSeconds);
    MMseqsNode::addPhaseSeconds(phaseValues, "pass 3 appending cells and piece counts", appendSeconds);
    MMseqsNode::addPhaseSeconds(phaseValues, "pass 3 waiting for writes", waitSeconds);
    return publishPrefExchangeFile(file, fileName, flushOffsets, pref.cellFirstKeys.size(), fileSize);
}

// the owner of rep cells sums the piece counts: per target split a class file with an entry per own cell and a piece per rep with targets in the split, and a .partitions row per cell.
// The cells of a pass are decoded, sorted and encoded in parallel, their class pieces split by split
static std::string writeSizeClasses(const Parameters &par, const KmerExchange &exchange, const TargetPref &pref, const std::vector<PrefExchangeFile> &countFiles, size_t &classSize,
                                    std::string &phaseValues) {
    const std::string prefix = par.db2 + ".piece_counts";
    size_t firstCell = pref.cellNodeOffsets[MMseqsNode::rank];
    size_t ownCells = pref.cellNodeOffsets[MMseqsNode::rank + 1] - firstCell;
    size_t targetSplitCount = pref.targetSplitKeyOffsets.size() - 1;
    std::vector<int> fds = openPrefExchangeFiles(prefix, countFiles);
    std::vector<std::string> names(fds.size());
    // per node and flush the offsets of the own cells
    std::vector<std::vector<size_t> > tables(fds.size());
    std::vector<size_t> cellSizes(ownCells, 0);
    for (size_t node = 0; node < fds.size(); node++) {
        names[node] = prefix + "." + SSTR(node);
        size_t flushCount = countFiles[node].flushCount;
        tables[node].resize((ownCells + 1) * flushCount);
        FileUtil::preadAll(fds[node], names[node].c_str(), (char *) tables[node].data(), tables[node].size() * sizeof(size_t), countFiles[node].tableOffset + firstCell * flushCount * sizeof(size_t));
        for (size_t flush = 0; flush < flushCount; flush++) {
            for (size_t cell = 0; cell < ownCells; cell++) {
                cellSizes[cell] += tables[node][(cell + 1) * flushCount + flush] - tables[node][cell * flushCount + flush];
            }
        }
    }
    // a rep count takes at least 2 bytes, decoded a PieceCount and a class piece of at most 12 bytes; per target split an entry end, its offset and its index cell
    for (size_t cell = 0; cell < ownCells; cell++) {
        cellSizes[cell] += cellSizes[cell] / 2 * (sizeof(PieceCount) + 12) + targetSplitCount * (1 + 3 * sizeof(size_t));
    }
    std::vector<size_t> passOffsets;
    std::vector<size_t> passSizes;
    planPasses(cellSizes, pref.passLimit, passOffsets, passSizes);
    std::vector<std::string> dataNames(targetSplitCount);
    std::vector<FILE *> dataFiles(targetSplitCount);
    std::vector<size_t> dataSizes(targetSplitCount, 0);
    std::vector<std::vector<uint64_t> > indexCells(targetSplitCount);
    for (size_t split = 0; split < targetSplitCount; split++) {
        dataNames[split] = par.db2 + "." + SSTR(pref.units.size() + MMseqsNode::rank * targetSplitCount + split);
        dataFiles[split] = FileUtil::openAndDelete(dataNames[split].c_str(), "w");
    }
    BatchWriter batchWriter(exchange.writeBehindSize, par.threads, fileno(dataFiles[0]));
    BatchReader batchReader(par.threads, fileno(dataFiles[0]));
    std::vector<char> counts;
    std::string rows;
    double readSeconds = 0;
    double decodeSeconds = 0;
    double sortSeconds = 0;
    double encodeSeconds = 0;
    double writeSeconds = 0;
    Timer timer;
    for (size_t pass = 0; pass + 1 < passOffsets.size(); pass++) {
        size_t passFirst = passOffsets[pass];
        size_t passEnd = passOffsets[pass + 1];
        size_t passCells = passEnd - passFirst;
        timer.reset();
        // per node and flush its start in counts
        std::vector<size_t> flushStarts;
        size_t readSize = 0;
        for (size_t node = 0; node < fds.size(); node++) {
            size_t flushCount = countFiles[node].flushCount;
            for (size_t flush = 0; flush < flushCount; flush++) {
                flushStarts.push_back(readSize);
                readSize += tables[node][passEnd * flushCount + flush] - tables[node][passFirst * flushCount + flush];
            }
        }
        counts.resize(readSize);
        std::vector<BatchReader::ReadRequest> readRequests;
        for (size_t node = 0, k = 0; node < fds.size(); node++) {
            size_t flushCount = countFiles[node].flushCount;
            for (size_t flush = 0; flush < flushCount; flush++, k++) {
                size_t start = tables[node][passFirst * flushCount + flush];
                size_t size = tables[node][passEnd * flushCount + flush] - start;
                if (size > 0) {
                    BatchReader::ReadRequest readRequest = {fds[node], names[node].c_str(), counts.data() + flushStarts[k], start, size, size};
                    readRequests.push_back(readRequest);
                }
            }
        }
        if (readRequests.empty() == false) {
            batchReader.readSync(readRequests);
        }
        readSeconds += timer.getTimediff();
        timer.reset();
        std::vector<size_t> cellPrefSizes(passCells, 0);
        std::vector<size_t> cellCosts(passCells, 0);
        // per cell its class pieces in split order, an entry per split
        std::vector<std::vector<char> > cellClasses(passCells);
        std::vector<size_t> splitOffsets(passCells * (targetSplitCount + 1));
        double threadDecodeSeconds = 0;
        double threadSortSeconds = 0;
        double threadEncodeSeconds = 0;
#pragma omp parallel num_threads(par.threads) reduction(+:threadDecodeSeconds, threadSortSeconds, threadEncodeSeconds)
        {
            std::vector<PieceCount> pieceCounts;
            Timer threadTimer;
#pragma omp for schedule(dynamic, 1)
            for (size_t cell = 0; cell < passCells; cell++) {
                threadTimer.reset();
                pieceCounts.clear();
                for (size_t node = 0, k = 0; node < fds.size(); node++) {
                    size_t flushCount = countFiles[node].flushCount;
                    const size_t *table = tables[node].data();
                    for (size_t flush = 0; flush < flushCount; flush++, k++) {
                        const char *data = counts.data() + flushStarts[k] + table[(passFirst + cell) * flushCount + flush] - table[passFirst * flushCount + flush];
                        const char *columnEnd = data + table[(passFirst + cell + 1) * flushCount + flush] - table[(passFirst + cell) * flushCount + flush];
                        while (data < columnEnd) {
                            cellPrefSizes[cell] += Util::getPrefKey(data);
                            cellCosts[cell] += Util::getPrefKey(data);
                            size_t targetSplit = Util::getPrefKey(data);
                            size_t repCount = Util::getPrefKey(data);
                            PieceCount pieceCount = { static_cast<DBKeyType>(pref.cellFirstKeys[firstCell + passFirst + cell]), targetSplit, 0 };
                            for (size_t i = 0; i < repCount; i++) {
                                pieceCount.rep += Util::getPrefKey(data);
                                pieceCount.targetCount = Util::getPrefKey(data);
                                pieceCounts.push_back(pieceCount);
                            }
                        }
                    }
                }
                threadDecodeSeconds += threadTimer.getTimediff();
                threadTimer.reset();
                SORT_SERIAL(pieceCounts.begin(), pieceCounts.end(), PieceCount::compareByRep);
                // the size class of a rep from its targets in all splits, kept once per split it has targets in
                size_t kept = 0;
                for (size_t pos = 0, repEnd = 0; pos < pieceCounts.size(); pos = repEnd) {
                    size_t targetCount = 0;
                    for (repEnd = pos; repEnd < pieceCounts.size() && pieceCounts[repEnd].rep == pieceCounts[pos].rep; repEnd++) {
                        targetCount += pieceCounts[repEnd].targetCount;
                    }
                    for (size_t i = pos; i < repEnd; i++) {
                        if (i == pos || pieceCounts[i].targetSplit != pieceCounts[i - 1].targetSplit) {
                            pieceCounts[kept] = pieceCounts[i];
                            pieceCounts[kept++].targetCount = targetCount;
                        }
                    }
                }
                pieceCounts.resize(kept);
                SORT_SERIAL(pieceCounts.begin(), pieceCounts.end(), PieceCount::compareBySplit);
                threadSortSeconds += threadTimer.getTimediff();
                threadTimer.reset();
                std::vector<char> &data = cellClasses[cell];
                size_t *offsets = splitOffsets.data() + cell * (targetSplitCount + 1);
                for (size_t split = 0, pos = 0; split < targetSplitCount; split++) {
                    offsets[split] = data.size();
                    DBKeyType prevRep = pref.cellFirstKeys[firstCell + passFirst + cell];
                    for (; pos < pieceCounts.size() && pieceCounts[pos].targetSplit == split; pos++) {
                        putPrefKey(data, pieceCounts[pos].rep - prevRep);
                        putPrefKey(data, getBitWidth(pieceCounts[pos].targetCount) - 1);
                        data.push_back('\0');
                        prevRep = pieceCounts[pos].rep;
                    }
                    data.push_back('\0');
                }
                offsets[targetSplitCount] = data.size();
                cellPrefSizes[cell] += data.size();
                threadEncodeSeconds += threadTimer.getTimediff();
            }
        }
        decodeSeconds += threadDecodeSeconds;
        sortSeconds += threadSortSeconds;
        encodeSeconds += threadEncodeSeconds;
        timer.reset();
        for (size_t split = 0; split < targetSplitCount; split++) {
            for (size_t cell = 0; cell < passCells; cell++) {
                const size_t *offsets = splitOffsets.data() + cell * (targetSplitCount + 1);
                indexCells[split].push_back(pref.cellFirstKeys[firstCell + passFirst + cell]);
                indexCells[split].push_back(dataSizes[split]);
                dataSizes[split] += offsets[split + 1] - offsets[split];
            }
        }
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
        for (size_t cell = 0; cell < passCells; cell++) {
            const size_t *offsets = splitOffsets.data() + cell * (targetSplitCount + 1);
            for (size_t split = 0; split < targetSplitCount; split++) {
                batchWriter.write(fileno(dataFiles[split]), dataNames[split].c_str(), cellClasses[cell].data() + offsets[split], offsets[split + 1] - offsets[split],
                                  indexCells[split][2 * (passFirst + cell) + 1]);
            }
        }
        for (size_t cell = 0; cell < passCells; cell++) {
            rows += SSTR(pref.cellFirstKeys[firstCell + passFirst + cell]) + "\t" + SSTR(firstCell + passFirst + cell) + "\t" + SSTR(cellPrefSizes[cell]) + "\t" + SSTR(cellCosts[cell])
                    + "\t" + SSTR(MMseqsNode::rank) + "\n";
        }
        writeSeconds += timer.getTimediff();
    }
    timer.reset();
    batchWriter.waitWrites();
    classSize = 0;
    for (size_t split = 0; split < targetSplitCount; split++) {
        std::string indexName = par.db2 + ".index." + SSTR(pref.units.size() + MMseqsNode::rank * targetSplitCount + split);
        FILE *indexFile = FileUtil::openAndDelete(indexName.c_str(), "w");
        size_t indexSize = indexCells[split].size() * sizeof(uint64_t);
        FileUtil::pwriteAll(fileno(indexFile), indexName.c_str(), (const char *) indexCells[split].data(), indexSize, 0);
        MMseqsNode::closeClusterFiles(dataFiles[split], dataNames[split], indexFile, indexName, dataSizes[split], indexSize);
        classSize += dataSizes[split] + indexSize + 2 * sizeof(uint64_t);
    }
    closePrefExchangeFiles(prefix, fds);
    writeSeconds += timer.getTimediff();
    MMseqsNode::addPhaseSeconds(phaseValues, "size classes reading piece counts", readSeconds);
    MMseqsNode::addPhaseSeconds(phaseValues, "size classes decoding summed over threads", decodeSeconds);
    MMseqsNode::addPhaseSeconds(phaseValues, "size classes sorting summed over threads", sortSeconds);
    MMseqsNode::addPhaseSeconds(phaseValues, "size classes encoding summed over threads", encodeSeconds);
    MMseqsNode::addPhaseSeconds(phaseValues, "size classes writing", writeSeconds);
    return rows;
}

template <typename T, bool includeAdjacency, typename L>
static void kmermatcherTargetPref(Parameters &par, DBReader<DBKeyType> &seqDbr, BaseMatrix *subMat, const KmerExchange &exchange, MMseqsNode::WorkQueue &sendQueue,
                                  size_t totalKmerCount, size_t smallestMemoryLimit, size_t localNodes) {
    TargetPref pref;
    size_t nodeCount = MMseqsNode::count;
    // an owner pass holds TASKS_PER_THREAD hash ranges per thread
    size_t minPassLimit = std::max((size_t) 1, getPassLimit(smallestMemoryLimit));
    size_t receiverKmerSize = (2 * exchange.globalIdBits + exchange.adjKeyCodeBits + CHAR_BIT - 1) / CHAR_BIT;
    size_t ownerPasses = std::max((size_t) 1, (totalKmerCount * receiverKmerSize / nodeCount + minPassLimit - 1) / minPassLimit);
    pref.hashRangeCount = nodeCount * std::max(RANGES_PER_NODE, ownerPasses * TASKS_PER_THREAD * par.threads);
    pref.hashRangeNodeOffsets.resize(nodeCount + 1);
    for (size_t node = 0; node <= nodeCount; node++) {
        pref.hashRangeNodeOffsets[node] = node * pref.hashRangeCount / nodeCount;
    }
    pref.codeShifts[KMER_GROUPS] = exchange.kmerHashShift;
    pref.codeShifts[WHOLE_SEQUENCE_GROUPS] = 0;
    unsigned int fingerprintBits = getBitWidth(totalKmerCount) + FINGERPRINT_SPARE_BITS;
    for (size_t kind = 0; kind < GROUP_KINDS; kind++) {
        unsigned int codeBits = 64 - pref.codeShifts[kind];
        pref.fingerprintShifts[kind] = pref.codeShifts[kind] + ((codeBits > fingerprintBits) ? codeBits - fingerprintBits : 0);
    }
    pref.targetSplitBlockOffsets = getTargetSplitBlockOffsets(exchange, seqDbr);
    for (size_t split = 0; split < pref.targetSplitBlockOffsets.size(); split++) {
        pref.targetSplitKeyOffsets.push_back(seqDbr.getBlockKeyOffsets()[pref.targetSplitBlockOffsets[split]]);
    }
    // the rep cells are the partitions of the pair merge
    size_t cellCount = exchange.sendSplitCount * exchange.partitionsPerRange;
    pref.cellFirstKeys.resize(cellCount + 1);
    for (size_t cell = 0; cell < cellCount; cell++) {
        pref.cellFirstKeys[cell] = getPartitionFirstKey(exchange, cell / exchange.partitionsPerRange, cell % exchange.partitionsPerRange);
    }
    pref.cellFirstKeys[cellCount] = seqDbr.getSize();
    pref.cellNodeOffsets.resize(nodeCount + 1);
    for (size_t node = 0; node <= nodeCount; node++) {
        pref.cellNodeOffsets[node] = node * cellCount / nodeCount;
    }
    pref.rangeCoveredKeys.resize(seqDbr.getIndexSize());
#pragma omp parallel for schedule(static) num_threads(par.threads)
    for (size_t range = 0; range < seqDbr.getIndexSize(); range++) {
        pref.rangeCoveredKeys[range] = getCoveredKeys(par, seqDbr, seqDbr.getIndex()[range].id);
    }
    pref.readChunkLimit = std::min(BatchReader::MAX_INFLIGHT_READ_SIZE, exchange.keySplitMemoryLimit / 6);
    // the current and the prefetched input buffer, page aligned in every data file
    size_t readBufferBytes = 2 * (std::max(pref.readChunkLimit, seqDbr.getMaxSeqLen()) + 2 * Util::getPageSize() * seqDbr.getDataFileCnt());
    size_t unitMemory = exchange.keySplitMemoryLimit - std::min(exchange.keySplitMemoryLimit, readBufferBytes);
    // two thirds for the k-mers of a unit, a third for the needed groups of the units
    pref.unitKmerCapacity = unitMemory / 3 * 2 / sizeof(PackedKmerPosition);
    pref.passLimit = std::max((size_t) 1, getPassLimit(exchange.keySplitMemoryLimit));
    pref.writeBuffers = new(std::nothrow) char[par.threads * KMER_FILE_BUFFER_SIZE];
    Util::checkAllocation(pref.writeBuffers, "Can not allocate writeBuffers memory in kmermatcherTargetPref");
    pref.keptSize = 0;
    pref.isKeeping = true;
    pref.neededSize = 0;
    pref.neededLimit = unitMemory / 3;
    pref.spillFile = NULL;
    pref.spillSize = 0;
    pref.localNodes = par.localTmp.empty() ? 0 : localNodes;
    pref.spillName = par.localTmp + "/" + FileUtil::baseName(par.db2) + ".needed." + SSTR(MMseqsNode::rank);
    if (MMseqsNode::isMaster()) {
        Debug(Debug::INFO) << "Send fingerprints in " << exchange.sendSplitCount << " send splits to " << pref.hashRangeCount << " hash ranges\n";
    }

    Timer phaseTimer;
    std::string values;
    double extractSeconds = 0;
    size_t sentSize = 0;
    std::string units = sendFingerprints<T, includeAdjacency, L>(par, seqDbr, subMat, exchange, pref, sendQueue, extractSeconds, sentSize);
    // pass 3 keeps the files of all own units open next to the marks and reps of all nodes, the size classes a file per target split next to the piece counts
    FileUtil::checkRlimitNoFile(2 * seqDbr.getDataFileCnt() + 64 + std::max(2 * nodeCount + pref.sendSplitUnitOffsets.back() + 2, nodeCount + pref.targetSplitKeyOffsets.size()));
    // the owners get the memory of the extraction buffer, pass 2 allocates it again
    delete[] pref.extractedUnit.kmers;
    pref.extractedUnit.kmers = NULL;
    MMseqsNode::addPhaseTime(values, "sending fingerprints", phaseTimer);
    MMseqsNode::addPhaseValue(values, "Fingerprint bytes sent", SSTR(sentSize));
    MMseqsNode::addPhaseValue(values, "Units", SSTR(pref.sendSplitUnitOffsets.back()) + (pref.isKeeping ? ", k-mers kept for the later passes" : ", k-mers extracted in every pass"));
    units = MMseqsNode::barrier(par.db2 + ".fingerprints.done", units);
    sendQueue.removeTakenItemFiles();
    if (MMseqsNode::isMaster()) {
        FileUtil::remove((par.db2 + ".memory").c_str());
    }
    // per node its fingerprint file, then a line per unit
    std::vector<PrefExchangeFile> fingerprintFiles;
    pref.nodeUnitOffsets.assign(1, 0);
    for (char *line = (char *) units.c_str(); *line != '\0';) {
        const char *columns[3];
        Util::getWordsOfLine(line, columns, 3);
        PrefExchangeFile file = { Util::fast_atoi<size_t>(columns[0]), Util::fast_atoi<size_t>(columns[1]), Util::fast_atoi<size_t>(columns[2]) };
        fingerprintFiles.push_back(file);
        line = Util::skipLine(line);
        for (size_t unit = 0; unit < file.flushCount; unit++, line = Util::skipLine(line)) {
            Util::getWordsOfLine(line, columns, 2);
            PrefUnit prefUnit = { Util::fast_atoi<size_t>(columns[0]), Util::fast_atoi<size_t>(columns[1]), 0 };
            prefUnit.idBits = getBitWidth(prefUnit.endKey - prefUnit.firstKey);
            pref.units.push_back(prefUnit);
        }
        pref.nodeUnitOffsets.push_back(pref.units.size());
    }
    std::vector<std::pair<size_t, size_t> > unitOrder;
    for (size_t unit = 0; unit < pref.units.size(); unit++) {
        unitOrder.push_back(std::make_pair(pref.units[unit].firstKey, unit));
    }
    SORT_SERIAL(unitOrder.begin(), unitOrder.end());
    pref.unitFiles.resize(pref.units.size());
    for (size_t i = 0; i < unitOrder.size(); i++) {
        pref.unitFiles[unitOrder[i].second] = i;
    }
    MMseqsNode::addPhaseTime(values, "waiting for the other nodes to send fingerprints", phaseTimer);

    size_t receivedSize = 0;
    std::string ownerValues;
    std::string answer = answerHashRanges(par, exchange, pref, par.db2 + ".fingerprints", fingerprintFiles, par.db2 + ".marks", false,
                                          exchange.keySplitMemoryLimit - std::min(exchange.keySplitMemoryLimit, getHeldMemory(pref)), [](const PrefUnit &, const size_t *groupCounts) {
        return groupCounts[KMER_GROUPS] + groupCounts[WHOLE_SEQUENCE_GROUPS];
    }, [&](size_t hashRange, std::vector<PrefRowRead *> &rows) {
        markFingerprints(pref, hashRange, rows);
    }, receivedSize, ownerValues);
    MMseqsNode::addPhaseTime(values, "marking fingerprints", phaseTimer);
    values += ownerValues;
    MMseqsNode::addPhaseValue(values, "Fingerprint bytes received", SSTR(receivedSize));
    std::vector<PrefExchangeFile> markFiles = readPrefExchangeFiles(MMseqsNode::barrier(par.db2 + ".marks.done", answer));
    FileUtil::remove((par.db2 + ".fingerprints." + SSTR(MMseqsNode::rank)).c_str());
    if (MMseqsNode::isMaster()) {
        FileUtil::remove((par.db2 + ".fingerprints.done").c_str());
    }
    MMseqsNode::addPhaseTime(values, "waiting for the other nodes to mark fingerprints", phaseTimer);

    std::string sent = sendCandidates<T, includeAdjacency, L>(par, seqDbr, subMat, exchange, pref, markFiles, extractSeconds, sentSize);
    // pass 3 reads the needed groups, or extracts again into extractedUnit
    delete[] pref.extractedUnit.kmers;
    pref.extractedUnit.kmers = NULL;
    size_t spilledUnits = 0;
    size_t extractedUnits = 0;
    for (size_t unit = 0; unit < pref.neededGroups.size(); unit++) {
        spilledUnits += pref.neededGroups[unit].spillOffset != SIZE_MAX;
        extractedUnits += pref.neededGroups[unit].data.empty() && pref.neededGroups[unit].spillOffset == SIZE_MAX;
    }
    MMseqsNode::addPhaseTime(values, "sending candidates", phaseTimer);
    MMseqsNode::addPhaseValue(values, "Needed groups for pass 3", SSTR(pref.neededSize) + " bytes in memory, " + SSTR(pref.spillSize) + " bytes in " + SSTR(spilledUnits) + " units spilled, "
                                                                   + SSTR(extractedUnits) + " units extracted again");
    MMseqsNode::addPhaseValue(values, "Candidate bytes sent", SSTR(sentSize));
    std::vector<PrefExchangeFile> candidateFiles = readPrefExchangeFiles(MMseqsNode::barrier(par.db2 + ".candidates.done", sent));
    if (MMseqsNode::isMaster()) {
        FileUtil::remove((par.db2 + ".marks.done").c_str());
    }
    MMseqsNode::addPhaseTime(values, "waiting for the other nodes to send candidates", phaseTimer);

    receivedSize = 0;
    ownerValues.clear();
    answer = answerHashRanges(par, exchange, pref, par.db2 + ".candidates", candidateFiles, par.db2 + ".reps", true,
                              exchange.keySplitMemoryLimit - std::min(exchange.keySplitMemoryLimit, getHeldMemory(pref)), [&](const PrefUnit &, const size_t *groupCounts) {
        return getRepAnswerBits(exchange, groupCounts);
    }, [&](size_t hashRange, std::vector<PrefRowRead *> &rows) {
        pickReps(exchange, pref, hashRange, rows);
    }, receivedSize, ownerValues);
    MMseqsNode::addPhaseTime(values, "picking reps", phaseTimer);
    values += ownerValues;
    MMseqsNode::addPhaseValue(values, "Candidate bytes received", SSTR(receivedSize));
    std::vector<PrefExchangeFile> repFiles = readPrefExchangeFiles(MMseqsNode::barrier(par.db2 + ".reps.done", answer));
    FileUtil::remove((par.db2 + ".candidates." + SSTR(MMseqsNode::rank)).c_str());
    if (MMseqsNode::isMaster()) {
        FileUtil::remove((par.db2 + ".candidates.done").c_str());
    }
    MMseqsNode::addPhaseTime(values, "waiting for the other nodes to pick reps", phaseTimer);

    size_t prefSize = 0;
    size_t pairCount = 0;
    size_t sweepCount = 0;
    std::string prefValues;
    sent = writeUnitPrefs<T, includeAdjacency, L>(par, seqDbr, subMat, exchange, pref, markFiles, repFiles, extractSeconds, prefSize, pairCount, sweepCount, prefValues);
    freeKeptUnits(pref);
    delete[] pref.extractedUnit.kmers;
    if (pref.spillFile != NULL) {
        if (fclose(pref.spillFile) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << pref.spillName << "\n";
            EXIT(EXIT_FAILURE);
        }
        FileUtil::remove(pref.spillName.c_str());
    }
    delete[] pref.writeBuffers;
    MMseqsNode::addPhaseTime(values, "writing unit prefilters", phaseTimer);
    values += prefValues;
    MMseqsNode::addPhaseSeconds(values, "reading sequences and extracting k-mers", extractSeconds);
    MMseqsNode::addPhaseValue(values, "Prefilter pairs", SSTR(pairCount) + " in " + SSTR(sweepCount) + " sweeps");
    std::vector<PrefExchangeFile> countFiles = readPrefExchangeFiles(MMseqsNode::barrier(par.db2 + ".piece_counts.done", sent));
    FileUtil::remove((par.db2 + ".marks." + SSTR(MMseqsNode::rank)).c_str());
    FileUtil::remove((par.db2 + ".reps." + SSTR(MMseqsNode::rank)).c_str());
    if (MMseqsNode::isMaster()) {
        FileUtil::remove((par.db2 + ".reps.done").c_str());
    }
    MMseqsNode::addPhaseTime(values, "waiting for the other nodes to write unit prefilters", phaseTimer);

    size_t classSize = 0;
    std::string classValues;
    std::string rows = writeSizeClasses(par, exchange, pref, countFiles, classSize, classValues);
    MMseqsNode::addPhaseTime(values, "writing size classes", phaseTimer);
    values += classValues;
    MMseqsNode::addPhaseValue(values, "Prefilter bytes written", SSTR(prefSize + classSize));
    MMseqsNode::printPhaseValues(values);
    MMseqsNode::publishRunFile(par.db2 + ".partitions." + SSTR(MMseqsNode::rank), rows);
    if (MMseqsNode::isMaster()) {
        rows = MMseqsNode::gatherRunFiles(par.db2 + ".partitions.");
        MMseqsNode::publishRunFile(par.db2 + ".merged", "");
        FileUtil::remove((par.db2 + ".piece_counts.done").c_str());
        // the owners' rows follow in cell order; the head: align groups and units, target split blocks, unit first keys, first cell of the class files of every node
        std::string partitions = SSTR(exchange.alignGroupCount) + "\t" + SSTR(pref.units.size()) + "\n";
        for (size_t split = 0; split < pref.targetSplitBlockOffsets.size(); split++) {
            partitions += SSTR(pref.targetSplitBlockOffsets[split]) + ((split + 1 < pref.targetSplitBlockOffsets.size()) ? "\t" : "\n");
        }
        for (size_t i = 0; i < unitOrder.size(); i++) {
            partitions += SSTR(unitOrder[i].first) + "\t";
        }
        partitions += SSTR(seqDbr.getSize()) + "\n";
        for (size_t node = 0; node <= nodeCount; node++) {
            partitions += SSTR(pref.cellNodeOffsets[node]) + ((node < nodeCount) ? "\t" : "\n");
        }
        FileUtil::publishFile(par.db2 + ".partitions", partitions + rows);
        MMseqsNode::removeOnceRead(par.db2 + ".merged");
        DBWriter::writeDbtypeFile(par.db2.c_str(), DBReader<DBKeyType>::setExtendedDbtype(Parameters::DBTYPE_PREFILTER_RES, Parameters::DBTYPE_EXTENDED_LINFINITY), false);
    } else {
        MMseqsNode::waitForRunFile(par.db2 + ".merged");
        MMseqsNode::removeOnceRead(par.db2 + ".merged");
    }
    FileUtil::remove((par.db2 + ".piece_counts." + SSTR(MMseqsNode::rank)).c_str());
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
    exchange.hasRepresentativeBits = MMseqsNode::statSharedFile(par.db1 + ".bits." + SSTR(MMseqsNode::rank));
    // A/B switch MMSEQS_KMER_TARGET_PREF=1: the units write their own prefilter, see kmermatcherTargetPref
    const char *targetPrefSwitch = getenv("MMSEQS_KMER_TARGET_PREF");
    const bool isTargetPref = targetPrefSwitch != NULL && strcmp(targetPrefSwitch, "1") == 0;
    // write the pref by target split for two or more align nodes
    exchange.hasTargetSplits = MMseqsNode::count - MMseqsNode::getFirstAlignNode() > 1 || isTargetPref;
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

    // align2clust opens up to five files per target split
    if (isTargetPref == false) {
        FileUtil::checkRlimitNoFile(2 * seqDbr.getDataFileCnt() + 64 + kmerGroups.count + 2 * MMseqsNode::count + 5 * (exchange.hasTargetSplits ? MAX_TARGET_SPLITS_PER_NODE * (MMseqsNode::count - MMseqsNode::getFirstAlignNode()) : 1));
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
    size_t smallestMemoryLimit = MMseqsNode::getSmallestMemoryLimit(par.db2 + ".memory", memoryLimit, par.threads, par.localTmp, &localNodes);
    size_t minPassLimit = std::max((size_t) 1, getPassLimit(smallestMemoryLimit));
    // --align2clust-split 0: floor(sqrt(align nodes)), fewer if a group node's share of the sequences does not fit a quarter of the smallest memory limit
    size_t alignNodeCount = MMseqsNode::count - MMseqsNode::getFirstAlignNode();
    exchange.alignGroupCount = (par.align2clustSplit > 0) ? par.align2clustSplit : std::min(static_cast<size_t>(std::sqrt(static_cast<double>(alignNodeCount))), alignNodeCount * (smallestMemoryLimit / 4) / seqDbr.getDataSize());
    exchange.alignGroupCount = std::min(alignNodeCount, std::max((size_t) 1, exchange.alignGroupCount));
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
    // plus the block table and a refilled pass
    size_t fixedMemory = representativeBitsMemory + 2 * par.threads * KMER_FILE_BUFFER_SIZE + exchange.writeBehindSize
                               + exchange.hashRangeCount * exchange.rowsPerHashRange * sizeof(KmerFileRow)
                               + (exchange.hasTargetSplits ? seqDbr.getBlockKeyOffsets().size() * sizeof(size_t) + 2 * KMER_FILE_BUFFER_SIZE : 0);
    exchange.keySplitMemoryLimit = (fixedMemory < memoryLimit) ? static_cast<size_t>(static_cast<double>(memoryLimit - fixedMemory) * 0.95) : 0;
    size_t rangePairPasses = std::max((size_t) 1, (totalKmerCount / exchange.sendSplitCount * exchange.maxRepSeqCount * pairTargetSize + minPassLimit - 1) / minPassLimit);
    exchange.partitionsPerRange = std::max((size_t) 4 * par.threads, 2 * rangePairPasses);
    if (isTargetPref) {
        kmermatcherTargetPref<T, includeAdjacency, L>(par, seqDbr, subMat, exchange, sendQueue, totalKmerCount, smallestMemoryLimit, localNodes);
        for (size_t i = 0; i < FILE_GROUP_KINDS; i++) {
            FileUtil::remove(indexFiles[i].c_str());
        }
        return;
    }
    exchange.pairRowSize = exchange.sendSplitCount * exchange.partitionsPerRange * sizeof(PairFileRow);
    size_t partitionCount = exchange.sendSplitCount * exchange.partitionsPerRange;
    size_t partitionsPerGroup = (partitionCount + kmerGroups.count - 1) / kmerGroups.count;
    initFileGroups(pairGroups, par.db2 + ".pairs", partitionCount, partitionsPerGroup, 4 + partitionsPerGroup);
    // the exchange file groups go to the shared folder at once, a local copy only delays the merging nodes
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
    double extractSeconds = 0;
    size_t sentSize = sendKmers<T, includeAdjacency, L>(par, seqDbr, subMat, exchange, sendQueue, representativeBitsMemory, sentSplitCount, ownKmers, kmerGroups, extractSeconds);
    std::thread kmerFollow(followFileGroups, std::ref(kmerGroups));
    // claim the own hash ranges
    for (size_t i = 0; i < ownKmers.hashRanges.size(); i++) {
        hashQueue.takeNextItem();
    }
    MMseqsNode::addPhaseTime(sendValues, "sending k-mers", phaseTimer);
    MMseqsNode::addPhaseSeconds(sendValues, "reading sequences and extracting k-mers", extractSeconds);
    MMseqsNode::addPhaseSeconds(sendValues, "waiting for k-mer writes summed over threads", kmerGroups.writeWaitSeconds);
    MMseqsNode::addPhaseValue(sendValues, "K-mer bytes sent", SSTR(sentSize + ownKmers.size));
    MMseqsNode::addPhaseValue(sendValues, "K-mer bytes kept in memory", SSTR(ownKmers.size));
    MMseqsNode::addPhaseValue(sendValues, "Send splits", SSTR(sentSplitCount));
    MMseqsNode::addPhaseValue(sendValues, "K-mer flushes", SSTR(kmerGroups.flushCount));
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
    MMseqsNode::addPhaseSeconds(receiveValues, "waiting for pair writes summed over threads", pairGroups.writeWaitSeconds);
    MMseqsNode::addPhaseValue(receiveValues, "K-mer bytes received", SSTR(receivedSize));
    MMseqsNode::addPhaseValue(receiveValues, "Pair bytes written", SSTR(pairSize));
    MMseqsNode::addPhaseValue(receiveValues, "Pair targets written", SSTR(pairTargetCount));
    MMseqsNode::addPhaseValue(receiveValues, "Hash ranges", SSTR(receivedHashRangeCount));
    MMseqsNode::printPhaseValues(receiveValues);
    kmerFollow.join();
    MMseqsNode::barrier(par.db2 + ".pairs.done");
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
    std::string mergePairValues;
    std::string partitionsData = mergePairFiles(par, seqDbr, exchange, mergeQueue, mergedSplitCount, ownPairs, pairGroups, progress, targetSplitBlockOffsets, prefSize, mergePairValues);
    MMseqsNode::addPhaseTime(mergeValues, "merging pairs", phaseTimer);
    MMseqsNode::addPhaseSeconds(mergeValues, "waiting for pair file groups", pairGroups.waitSeconds);
    mergeValues += mergePairValues;
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
        // first line: the align group count, second line: first block of every target split
        std::string sortedPartitions = SSTR(exchange.alignGroupCount) + "\n";
        for (size_t i = 0; i < targetSplitBlockOffsets.size(); i++) {
            sortedPartitions += SSTR(targetSplitBlockOffsets[i]) + ((i + 1 < targetSplitBlockOffsets.size()) ? "\t" : "\n");
        }
        for (size_t i = 0; i < partitionRows.size(); i++) {
            sortedPartitions.append(partitionRows[i].second, Util::skipLine(partitionRows[i].second) - partitionRows[i].second);
        }
        FileUtil::publishFile(par.db2 + ".partitions", sortedPartitions);
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
