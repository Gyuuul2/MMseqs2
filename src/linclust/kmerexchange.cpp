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

static const size_t KMER_FILE_BUFFER_BYTES = 16 * 1024 * 1024;
// the writes that wait for the file system while the threads go on take the memory limit divided by this
static const size_t WRITE_BEHIND_MEMORY_DIVISOR = 8;
// the bit buffer keeps less than a byte between codes, so this many bits always fit in one step
static const unsigned int KMER_FILE_STEP_BITS = 64 - CHAR_BIT;
// open files besides the k-mer and sequence data files: standard streams and barrier files, with headroom
static const size_t OTHER_OPEN_FILES = 64;
// the work that all threads share is cut into this many tasks per thread
static const size_t TASKS_PER_THREAD = 2;
// a send split is cut into at least this many partitions per thread
static const size_t PARTITIONS_PER_THREAD = 4;
// a thread of a receiving pass takes regions of at most this part of its share of the output, so the regions not yet filled when the output is full are a small part of it
static const size_t OUTPUT_REGIONS_PER_THREAD = 16;
// a partition row ends once its alignment cells reach 1 / this of what its node merged, four times the group ranges per aligning node of align2clust, which can only cut at rows
static const size_t PARTITION_ROWS_PER_NODE = 16384;
// a key task keeps fewer than twice this many places to cut a partition
static const size_t CUTS_PER_TASK = 8;
// the nodes take the send splits and the hash ranges one at a time, about this many each, so a node that is faster takes more
static const size_t RANGES_PER_NODE = 8;
// sending a k-mer takes about as long as reading this many bytes of sequence
static const size_t SEND_KMER_BYTES = 12;
// the bytes of an exchange file are freed in whole parts of this size
static const size_t PUNCH_ALIGNMENT = 1024 * 1024;

// the codes of a row of a k-mer file, one after the other
enum { KMER_MEMBER_CODES, KMER_REP_CANDIDATE_CODES, KMER_INDEX_CODES, KMER_CODE_COUNT };
// the codes of a row of a pair file, one after the other: per record its key gap, target count, first target and gap bits, then the gaps of its other targets
enum { PAIR_INDEX_CODES, PAIR_TARGET_CODES, PAIR_CODE_COUNT };

// a row of the k-mer file <pref>.kmers.<node> per key split and hash sub range, then a row per key split and hash range that holds its whole sequence hashes
struct KmerFileRow {
    DBKeyType splitFirstKey;
    unsigned int idBits;
    // where every code starts, then the end of the row
    size_t offsets[KMER_CODE_COUNT + 1];
    size_t groupCount;
};

// a row of the pair file <pref>.pairs.<node> per flush and partition with records, a record per representative with its targets in id order without repeats
struct PairFileRow {
    size_t offsets[PAIR_CODE_COUNT + 1];
    size_t partition;
    size_t recordCount;
    size_t targetCount;
};

// a representative and its targets of a group in the output buffer of a receiving pass, at targetOffset of the targets there
struct PairRecord {
    DBKeyType rep;
    size_t targetOffset;
    size_t targetCount;

    static bool compareByRep(const PairRecord &first, const PairRecord &second) {
        return first.rep < second.rep;
    }
};

// a row of a received file that a pass reads, and where its codes are in the pass buffer
struct PassRow {
    size_t file;
    size_t row;
    size_t bufferOffset;
};

// where a received row is at the start of a task of its hash sub range: its next group, the hash before it and the bit position of every code
struct KmerThreadTaskStart {
    size_t group;
    uint64_t hash;
    size_t bitPos[KMER_CODE_COUNT];
};

// with fd -1 a buffer in memory, otherwise a buffer that batchWriter writes at writePos of the file whenever it is full
struct KmerCodeWriter {
    int fd;
    const char *fileName;
    char *buffer;
    size_t bufferBytes;
    size_t bufferPos;
    size_t writePos;
    uint64_t bitBuffer;
    unsigned int bitCount;
    BatchWriter *batchWriter;
};

static KmerCodeWriter createFileCodeWriter(int fd, const char *fileName, char *buffer, size_t bufferBytes, size_t writePos, BatchWriter *batchWriter) {
    return KmerCodeWriter{fd, fileName, buffer, bufferBytes, 0, writePos, 0, 0, batchWriter};
}

static KmerCodeWriter createMemoryCodeWriter(char *data, size_t bytes, size_t writePos) {
    return KmerCodeWriter{-1, NULL, data + writePos, bytes - writePos, 0, writePos, 0, 0, NULL};
}

// with fd -1 a buffer in memory, otherwise a buffer of bufferBytes bytes at fileOffset that getBits refills up to fileEnd
struct KmerCodeReader {
    char *buffer;
    size_t bufferPos;
    uint64_t bitBuffer;
    unsigned int bitCount;
    int fd;
    const char *fileName;
    size_t bufferBytes;
    size_t fileOffset;
    size_t fileEnd;
};

static void writeKmerCodeBuffer(KmerCodeWriter &kmerCodeWrite) {
    kmerCodeWrite.batchWriter->write(kmerCodeWrite.fd, kmerCodeWrite.fileName, kmerCodeWrite.buffer, kmerCodeWrite.bufferPos, kmerCodeWrite.writePos);
    kmerCodeWrite.writePos += kmerCodeWrite.bufferPos;
    kmerCodeWrite.bufferPos = 0;
}

static void refillKmerCodeBuffer(KmerCodeReader &kmerCodeRead, size_t fileOffset) {
    FileUtil::preadAll(kmerCodeRead.fd, kmerCodeRead.fileName, kmerCodeRead.buffer, std::min(kmerCodeRead.bufferBytes, kmerCodeRead.fileEnd - fileOffset), fileOffset);
    kmerCodeRead.fileOffset = fileOffset;
    kmerCodeRead.bufferPos = 0;
}

static inline void putBits(KmerCodeWriter &kmerCodeWrite, uint64_t value, unsigned int width) {
    for (unsigned int done = 0; done < width; done += KMER_FILE_STEP_BITS) {
        const unsigned int bits = std::min(width - done, KMER_FILE_STEP_BITS);
        kmerCodeWrite.bitBuffer |= ((value >> done) & ((1ULL << bits) - 1)) << kmerCodeWrite.bitCount;
        for (kmerCodeWrite.bitCount += bits; kmerCodeWrite.bitCount >= CHAR_BIT; kmerCodeWrite.bitCount -= CHAR_BIT) {
            if (kmerCodeWrite.bufferPos == kmerCodeWrite.bufferBytes) {
                writeKmerCodeBuffer(kmerCodeWrite);
            }
            kmerCodeWrite.buffer[kmerCodeWrite.bufferPos++] = static_cast<char>(kmerCodeWrite.bitBuffer);
            kmerCodeWrite.bitBuffer >>= CHAR_BIT;
        }
    }
}

// the writer is at a byte boundary
static void putBytes(KmerCodeWriter &kmerCodeWrite, const char *bytes, size_t count) {
    for (size_t done = 0; done < count;) {
        if (kmerCodeWrite.bufferPos == kmerCodeWrite.bufferBytes) {
            writeKmerCodeBuffer(kmerCodeWrite);
        }
        const size_t copyBytes = std::min(count - done, kmerCodeWrite.bufferBytes - kmerCodeWrite.bufferPos);
        memcpy(kmerCodeWrite.buffer + kmerCodeWrite.bufferPos, bytes + done, copyBytes);
        kmerCodeWrite.bufferPos += copyBytes;
        done += copyBytes;
    }
}

static inline void putPrefKey(KmerCodeWriter &kmerCodeWrite, uint64_t key) {
    char buffer[10];
    putBytes(kmerCodeWrite, buffer, Util::putPrefKey(buffer, key));
}

static inline size_t getPrefKeyBytes(uint64_t key) {
    return (std::max(1u, getBitWidth(key)) + 6) / 7;
}

static inline uint64_t getBits(KmerCodeReader &kmerCodeRead, unsigned int width) {
    uint64_t value = 0;
    for (unsigned int done = 0; done < width; done += KMER_FILE_STEP_BITS) {
        const unsigned int bits = std::min(width - done, KMER_FILE_STEP_BITS);
        for (; kmerCodeRead.bitCount < bits; kmerCodeRead.bitCount += CHAR_BIT) {
            if (kmerCodeRead.bufferPos == kmerCodeRead.bufferBytes) {
                refillKmerCodeBuffer(kmerCodeRead, kmerCodeRead.fileOffset + kmerCodeRead.bufferBytes);
            }
            kmerCodeRead.bitBuffer |= static_cast<uint64_t>(static_cast<unsigned char>(kmerCodeRead.buffer[kmerCodeRead.bufferPos++])) << kmerCodeRead.bitCount;
        }
        value |= (kmerCodeRead.bitBuffer & ((1ULL << bits) - 1)) << done;
        kmerCodeRead.bitBuffer >>= bits;
        kmerCodeRead.bitCount -= bits;
    }
    return value;
}

// the zeros before the next one bit, which is read too; the bit buffer holds no bits above bitCount
static inline uint64_t getUnary(KmerCodeReader &kmerCodeRead) {
    uint64_t zeros = 0;
    while (kmerCodeRead.bitBuffer == 0) {
        zeros += kmerCodeRead.bitCount;
        if (kmerCodeRead.bufferPos == kmerCodeRead.bufferBytes) {
            refillKmerCodeBuffer(kmerCodeRead, kmerCodeRead.fileOffset + kmerCodeRead.bufferBytes);
        }
        kmerCodeRead.bitBuffer = static_cast<unsigned char>(kmerCodeRead.buffer[kmerCodeRead.bufferPos++]);
        kmerCodeRead.bitCount = CHAR_BIT;
    }
    const unsigned int bit = __builtin_ctzll(kmerCodeRead.bitBuffer);
    kmerCodeRead.bitBuffer >>= bit + 1;
    kmerCodeRead.bitCount -= bit + 1;
    return zeros + bit;
}

// the position of the next bit, from the start of the file or the buffer
static inline size_t getKmerCodeBitPos(const KmerCodeReader &kmerCodeRead) {
    return (kmerCodeRead.fileOffset + kmerCodeRead.bufferPos) * CHAR_BIT - kmerCodeRead.bitCount;
}

static inline void seekKmerCodeBits(KmerCodeReader &kmerCodeRead, size_t bitPos) {
    const size_t bytePos = bitPos / CHAR_BIT;
    if (bytePos >= kmerCodeRead.fileOffset && bytePos - kmerCodeRead.fileOffset < kmerCodeRead.bufferBytes) {
        kmerCodeRead.bufferPos = bytePos - kmerCodeRead.fileOffset;
    } else {
        refillKmerCodeBuffer(kmerCodeRead, bytePos);
    }
    kmerCodeRead.bitBuffer = 0;
    kmerCodeRead.bitCount = 0;
    getBits(kmerCodeRead, bitPos % CHAR_BIT);
}

// Elias gamma code of a value of at least 1
static inline void putEliasGamma(KmerCodeWriter &kmerCodeWrite, size_t value) {
    const unsigned int valueBits = getBitWidth(value) - 1;
    putBits(kmerCodeWrite, 0, valueBits);
    putBits(kmerCodeWrite, 1, 1);
    putBits(kmerCodeWrite, value ^ (1ULL << valueBits), valueBits);
}

static inline size_t getEliasGamma(KmerCodeReader &kmerCodeRead) {
    const unsigned int valueBits = getUnary(kmerCodeRead);
    return (1ULL << valueBits) | getBits(kmerCodeRead, valueBits);
}

// Rice code: the value >> bits in unary, then its lower bits
static inline void putRice(KmerCodeWriter &kmerCodeWrite, uint64_t value, unsigned int bits) {
    for (uint64_t quotient = value >> bits; quotient > 0;) {
        const unsigned int zeros = std::min<uint64_t>(quotient, KMER_FILE_STEP_BITS);
        putBits(kmerCodeWrite, 0, zeros);
        quotient -= zeros;
    }
    putBits(kmerCodeWrite, 1, 1);
    putBits(kmerCodeWrite, value, bits);
}

static inline uint64_t getRice(KmerCodeReader &kmerCodeRead, unsigned int bits) {
    const uint64_t quotient = getUnary(kmerCodeRead);
    return (quotient << bits) | getBits(kmerCodeRead, bits);
}

// hash sub range i of hashSubRangeCount starts at i * 2^64 / hashSubRangeCount rounded down, here in units of 2^shift rounded up
static KmerPositionBit::uint128 getHashSubRangeStart(size_t hashSubRange, size_t hashSubRangeCount, unsigned int shift) {
    const KmerPositionBit::uint128 start = (static_cast<KmerPositionBit::uint128>(hashSubRange) << 64) / hashSubRangeCount;
    return (start + (static_cast<KmerPositionBit::uint128>(1) << shift) - 1) >> shift;
}

// Rice parameter: the expected difference between the hashes of groupCount groups in the hash sub ranges [hashSubRange, hashSubRangeEnd), rounded down to a power of two
static unsigned int getHashDiffLowBits(size_t hashSubRange, size_t hashSubRangeEnd, size_t hashSubRangeCount, unsigned int shift, size_t groupCount) {
    const KmerPositionBit::uint128 width = getHashSubRangeStart(hashSubRangeEnd, hashSubRangeCount, shift) - getHashSubRangeStart(hashSubRange, hashSubRangeCount, shift);
    const KmerPositionBit::uint128 diff = width / std::max<size_t>(1, groupCount);
    return getBitWidth((diff > UINT64_MAX) ? UINT64_MAX : std::max<uint64_t>(1, static_cast<uint64_t>(diff))) - 1;
}

// a value of a row of a merge, the smallest value first in a priority_queue, at the same value the first row
struct RowValue {
    uint64_t value;
    unsigned int row;
    RowValue(uint64_t value, unsigned int row) : value(value), row(row) {}
    bool operator<(const RowValue &other) const {
        return value > other.value || (value == other.value && row > other.row);
    }
};

typedef std::priority_queue<RowValue> RowValueQueue;

// a received row of a hash sub range and its group read last, with where the codes start, its row in the task and where its group read last starts
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

// a row of a partition that the rep node reads: its code readers, the Rice parameter of its key gaps, its records left, and its record read last with the targets after the current one
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

// where a row of a partition is at the start of a key task: its records left, the key after its record before and the bit position of every code
struct PairKeyTaskStart {
    size_t recordsLeft;
    size_t nextRepKey;
    size_t bitPos[PAIR_CODE_COUNT];
};

// a place to cut a partition: the first key after it, the pref bytes and the alignment cells before it
struct PartitionCut {
    size_t key;
    size_t prefBytes;
    size_t cost;
};

// the cuts of a key task where its alignment cells passed a multiple of step, which doubles to keep fewer than twice CUTS_PER_TASK of them
struct KeyTaskCuts {
    KeyTaskCuts() : step(1) {}
    std::vector<PartitionCut> cuts;
    size_t step;

    void add(const PartitionCut &cut) {
        if (cut.cost / step == (cuts.empty() ? 0 : cuts.back().cost / step)) {
            return;
        }
        cuts.push_back(cut);
        while (cuts.size() >= 2 * CUTS_PER_TASK) {
            step *= 2;
            size_t kept = 0;
            for (size_t i = 0; i < cuts.size(); i++) {
                if (cuts[i].cost / step != ((kept > 0) ? cuts[kept - 1].cost / step : 0)) {
                    cuts[kept++] = cuts[i];
                }
            }
            cuts.resize(kept);
        }
    }
};

// a partition that the rep node merges: its row readers, where they start every key task, key targets, key tasks with their entry and index offsets, alignment cells and cuts, alignment cells
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

// a partition merge with threads holds per row its reader and its starts of the key tasks, and per thread a copy of the readers and their places in the queues
static size_t getPartitionMergeBytes(size_t rowCount, size_t threads) {
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

// the file of every node that a node reads for the whole phase: where its row table starts, its row flushes and, of a pair file, the code bytes of every partition
struct ReceivedFile {
    std::string fileName;
    FILE *file;
    size_t rowTableOffset;
    size_t flushCount;
    std::vector<size_t> itemBytes;
    // without a file, the rows that this node sent to itself in memory
    char *data;
};

template <typename Row>
static size_t getRowEnd(const Row &row) {
    return row.offsets[sizeof(row.offsets) / sizeof(size_t) - 1];
}

// writes at rowTableOffset the row starts of flushRowOffsets item after item, the flush ends last, then itemBytes, flushCount and rowTableOffset; closes the file
static void writeRowTable(FILE *file, const std::string &fileName, size_t itemCount, const std::vector<size_t> &flushRowOffsets, const std::vector<size_t> &itemBytes, size_t rowTableOffset) {
    const size_t flushCount = flushRowOffsets.size() / (itemCount + 1);
    std::vector<size_t> table(flushRowOffsets.size());
    for (size_t flush = 0; flush < flushCount; flush++) {
        for (size_t item = 0; item <= itemCount; item++) {
            table[item * flushCount + flush] = flushRowOffsets[flush * (itemCount + 1) + item];
        }
    }
    table.insert(table.end(), itemBytes.begin(), itemBytes.end());
    table.push_back(flushCount);
    table.push_back(rowTableOffset);
    FileUtil::pwriteAll(fileno(file), fileName.c_str(), reinterpret_cast<const char *>(table.data()), table.size() * sizeof(size_t), rowTableOffset);
    if (fclose(file) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << fileName << "\n";
        EXIT(EXIT_FAILURE);
    }
}

// opens the file to read it and free what is read, and reads its itemByteCount code bytes, the flush count and where the row table starts, written by writeRowTable
static void openReceivedFile(ReceivedFile &receivedFile, size_t itemByteCount) {
    receivedFile.file = FileUtil::openFileOrDie(receivedFile.fileName.c_str(), "r+", true);
    struct stat sb;
    const size_t tailBytes = (itemByteCount + 2) * sizeof(size_t);
    if (fstat(fileno(receivedFile.file), &sb) != 0 || static_cast<size_t>(sb.st_size) < tailBytes) {
        Debug(Debug::ERROR) << "Cannot read " << receivedFile.fileName << "\n";
        EXIT(EXIT_FAILURE);
    }
    std::vector<size_t> tail(itemByteCount + 2);
    FileUtil::preadAll(fileno(receivedFile.file), receivedFile.fileName.c_str(), reinterpret_cast<char *>(tail.data()), tailBytes, sb.st_size - tailBytes);
    receivedFile.itemBytes.assign(tail.begin(), tail.begin() + itemByteCount);
    receivedFile.flushCount = tail[itemByteCount];
    receivedFile.rowTableOffset = tail[itemByteCount + 1];
}

// the rows of item, flush after flush, read now or added to readRequests for the caller to read
template <typename Row>
static void readItemRows(const ReceivedFile &receivedFile, size_t item, std::vector<Row> &rows, std::vector<BatchReader::ReadRequest> *readRequests = NULL) {
    const size_t flushCount = receivedFile.flushCount;
    std::vector<size_t> rowStarts(2 * flushCount);
    FileUtil::preadAll(fileno(receivedFile.file), receivedFile.fileName.c_str(), reinterpret_cast<char *>(rowStarts.data()), rowStarts.size() * sizeof(size_t),
                       receivedFile.rowTableOffset + item * flushCount * sizeof(size_t));
    size_t rowCount = 0;
    for (size_t flush = 0; flush < flushCount; flush++) {
        rowCount += (rowStarts[flushCount + flush] - rowStarts[flush]) / sizeof(Row);
    }
    rows.resize(rowCount);
    size_t row = 0;
    for (size_t flush = 0; flush < flushCount; flush++) {
        const size_t bytes = rowStarts[flushCount + flush] - rowStarts[flush];
        if (readRequests == NULL) {
            FileUtil::preadAll(fileno(receivedFile.file), receivedFile.fileName.c_str(), reinterpret_cast<char *>(rows.data() + row), bytes, rowStarts[flush]);
        } else if (bytes > 0) {
            const BatchReader::ReadRequest readRequest = {fileno(receivedFile.file), receivedFile.fileName.c_str(), reinterpret_cast<char *>(rows.data() + row), rowStarts[flush], bytes, bytes};
            readRequests->push_back(readRequest);
        }
        row += bytes / sizeof(Row);
    }
}

static void closeReceivedFiles(std::vector<ReceivedFile> &receivedFiles) {
    for (size_t i = 0; i < receivedFiles.size(); i++) {
        if (receivedFiles[i].file != NULL && fclose(receivedFiles[i].file) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << receivedFiles[i].fileName << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
}

// reads the codes of passRows to buffer in file order and sets their bufferOffset
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
        const size_t bytes = getRowEnd(codeRow) - codeRow.offsets[0];
        const ReceivedFile &receivedFile = receivedFiles[passRow.file];
        if (bytes == 0 || receivedFile.data != NULL) {
            continue;
        }
        passRow.bufferOffset = bufferPos;
        BatchReader::ReadRequest *last = readRequests.empty() ? NULL : &readRequests.back();
        if (last != NULL && last->fileToRead == fileno(receivedFile.file) && last->fileOffset + last->bytesToRead == codeRow.offsets[0]) {
            last->bytesToRead += bytes;
            last->minBytesToRead += bytes;
        } else {
            const BatchReader::ReadRequest readRequest = {fileno(receivedFile.file), receivedFile.fileName.c_str(), buffer + bufferPos, codeRow.offsets[0], bytes, bytes};
            readRequests.push_back(readRequest);
        }
        bufferPos += bytes;
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

// frees the codes of passRows once processed, only whole PUNCH_ALIGNMENT file bytes or memory pages of rows that follow each other so no other row loses data; returns the file ranges freed and adds the memory freed to releasedBytes
template <typename Row>
static size_t punchPassRows(const std::vector<ReceivedFile> &receivedFiles, const std::vector<std::vector<Row> > &fileRows, const std::vector<PassRow> &passRows, size_t &releasedBytes) {
    std::vector<std::pair<size_t, std::pair<size_t, size_t> > > fileRanges;
    for (size_t i = 0; i < passRows.size(); i++) {
        const Row &codeRow = fileRows[passRows[i].file][passRows[i].row];
        if (getRowEnd(codeRow) > codeRow.offsets[0]) {
            fileRanges.push_back(std::make_pair(passRows[i].file, std::make_pair(codeRow.offsets[0], getRowEnd(codeRow))));
        }
    }
    SORT_SERIAL(fileRanges.begin(), fileRanges.end());
    size_t punchedRanges = 0;
    for (size_t i = 0; i < fileRanges.size();) {
        const size_t file = fileRanges[i].first;
        // memory is aligned at its addresses
        const size_t base = reinterpret_cast<size_t>(receivedFiles[file].data);
        const size_t alignment = (receivedFiles[file].data == NULL) ? PUNCH_ALIGNMENT : Util::getPageSize();
        const size_t start = (base + fileRanges[i].second.first + alignment - 1) / alignment * alignment;
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
            punchedRanges++;
        } else {
            Util::releasePages(reinterpret_cast<char *>(start), end - start);
            releasedBytes += end - start;
        }
    }
    return punchedRanges;
}

// a reader of a code of a row: in the memory of the received file, in the pass buffer at bufferOffset, or with codeReadBufferBytes through codeReadBuffer
template <typename Row>
static KmerCodeReader getRowCodeReader(const ReceivedFile &receivedFile, const Row &codeRow, size_t bufferOffset, size_t code, char *passBuffer, char *codeReadBuffer, size_t codeReadBufferBytes) {
    KmerCodeReader codeRead = {NULL, 0, 0, 0, -1, NULL, SIZE_MAX, 0, 0};
    if (receivedFile.data != NULL) {
        codeRead.buffer = receivedFile.data + codeRow.offsets[code];
    } else if (codeReadBufferBytes == 0) {
        codeRead.buffer = passBuffer + bufferOffset + codeRow.offsets[code] - codeRow.offsets[0];
    } else {
        codeRead.buffer = codeReadBuffer;
        codeRead.fd = fileno(receivedFile.file);
        codeRead.fileName = receivedFile.fileName.c_str();
        codeRead.bufferBytes = codeReadBufferBytes;
        codeRead.fileEnd = codeRow.offsets[code + 1];
        refillKmerCodeBuffer(codeRead, codeRow.offsets[code]);
    }
    return codeRead;
}

// cuts the items into passes whose bytes fit passLimit, an item that does not fit a pass of its own; returns the largest pass that fits
static size_t planPasses(const std::vector<size_t> &itemBytes, size_t passLimit, std::vector<size_t> &passOffsets, std::vector<size_t> &passBytes) {
    passOffsets.assign(1, 0);
    passBytes.assign(1, 0);
    size_t maxPassBytes = 0;
    for (size_t item = 0; item < itemBytes.size(); item++) {
        if (item > passOffsets.back() && passBytes.back() + itemBytes[item] > passLimit) {
            passOffsets.push_back(item);
            passBytes.push_back(0);
        }
        passBytes.back() += itemBytes[item];
        if (passBytes.back() <= passLimit) {
            maxPassBytes = std::max(maxPassBytes, passBytes.back());
        }
    }
    passOffsets.push_back(itemBytes.size());
    return maxPassBytes;
}

// a pass holds the codes it reads, the prefetched codes of the next pass and its output, a third of the memory each
static size_t getPassLimit(size_t memory) {
    return memory / 3 / sizeof(size_t) * sizeof(size_t);
}

// linfinity: the send splits, hash ranges and partitions that every node cuts the same, and the limits of this node
struct KmerExchange {
    // the keys of every send split, whose k-mers the node that takes it sends; the merge splits are key ranges of its partitions
    std::vector<size_t> sendSplitKeyOffsets;
    size_t sendSplitCount;
    // the keys of the data files, and whether a clusthashfast subset has the representative bits of their keys
    std::vector<size_t> fileKeyOffsets;
    bool hasRepresentativeBits;
    size_t memoryLimit;
    size_t writeBehindBytes;
    size_t keySplitMemoryLimit;
    // the rows of a flush of pair records, a row per partition at most, which a receiving node holds until it writes them
    size_t pairRowBytes;
    unsigned int adjKeyBits;
    // the residues of an adjKey are sent as the digits of a number of base adjKeyBase in adjKeyCodeBits
    unsigned int adjKeyBase;
    unsigned int adjKeyCodeBits;
    unsigned int kmerHashShift;
    unsigned int globalIdBits;
    size_t maxRepSeqs;
    // every node merges its first own send splits, rank + i * count for i below this, itself and before the merge splits that the nodes take
    size_t selfMergedSendSplitCount;
    size_t hashRangeCount;
    size_t hashSubRangesPerRange;
    size_t hashSubRangeCount;
    size_t rowsPerHashRange;
    size_t partitionsPerRange;
};

// the hash sub ranges [first, end) of row rangeHashSubRange of a hash range: one for a row of k-mers, all of the hash range for its row of whole sequence hashes, which returns true
static bool getRowHashSubRanges(const KmerExchange &exchange, size_t hashRange, size_t rangeHashSubRange, size_t &first, size_t &end) {
    const bool isWholeSequence = (rangeHashSubRange >= exchange.hashSubRangesPerRange);
    first = hashRange * exchange.hashSubRangesPerRange + (isWholeSequence ? 0 : rangeHashSubRange);
    end = first + (isWholeSequence ? exchange.hashSubRangesPerRange : 1);
    return isWholeSequence;
}

static inline void putAdjKeyCode(KmerCodeWriter &kmerCodeWrite, const KmerExchange &exchange, unsigned int adjKey) {
    unsigned int adjKeyCode = 0;
    for (int j = 5; j >= 0; j--) {
        adjKeyCode = adjKeyCode * exchange.adjKeyBase + KmerPositionBit::getAdjacentSeq(adjKey, j);
    }
    putBits(kmerCodeWrite, adjKeyCode, exchange.adjKeyCodeBits);
}

static inline unsigned int getAdjKeyCode(KmerCodeReader &kmerCodeRead, const KmerExchange &exchange) {
    unsigned int adjKeyCode = getBits(kmerCodeRead, exchange.adjKeyCodeBits);
    unsigned int adjKey = 0;
    for (int j = 0; j < 6; j++) {
        adjKey |= (adjKeyCode % exchange.adjKeyBase) << (KmerPositionBit::RESIDUE_BITS * j);
        adjKeyCode /= exchange.adjKeyBase;
    }
    return adjKey;
}

// every node cuts every send split into the same partitions, the pairs of a representative go to its partition
static size_t getPartitionFirstKey(const KmerExchange &exchange, size_t sendSplit, size_t partition) {
    return exchange.sendSplitKeyOffsets[sendSplit] + (partition * (exchange.sendSplitKeyOffsets[sendSplit + 1] - exchange.sendSplitKeyOffsets[sendSplit]) + exchange.partitionsPerRange - 1) / exchange.partitionsPerRange;
}

// a node sends and merges a send split with the representative bits of its keys
static void setSendSplit(const Parameters &par, const KmerExchange &exchange, size_t sendSplit) {
    localFirstKey = exchange.sendSplitKeyOffsets[sendSplit];
    delete[] representativeBits;
    representativeBits = NULL;
    if (exchange.hasRepresentativeBits) {
        representativeBits = MMseqsNode::readRepresentativeBits(par.db1, exchange.fileKeyOffsets, localFirstKey, exchange.sendSplitKeyOffsets[sendSplit + 1] - localFirstKey);
    }
}

// the distinct ids of [groupStart, groupEnd), sorted in groupBuffer or, if they do not fit, set as bits of the key split counted per 512; returns their count
static size_t getGroupIds(const KmerPositionBit *hashSeqPair, size_t groupStart, size_t groupEnd, uint64_t *groupBuffer, size_t groupBufferWords, uint64_t *setBitCounts, size_t keySplitWords) {
    if (groupEnd - groupStart <= groupBufferWords) {
        for (size_t pos = groupStart; pos < groupEnd; pos++) {
            groupBuffer[pos - groupStart] = hashSeqPair[pos].getId();
        }
        SORT_SERIAL(groupBuffer, groupBuffer + (groupEnd - groupStart));
        return static_cast<size_t>(std::unique(groupBuffer, groupBuffer + (groupEnd - groupStart)) - groupBuffer);
    }
    memset(groupBuffer, 0, keySplitWords * sizeof(uint64_t));
    for (size_t pos = groupStart; pos < groupEnd; pos++) {
        const DBKeyType id = hashSeqPair[pos].getId();
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

// the rows of a pass with groups, hash sub range after hash sub range of a hash range in id order, and where the rows of every hash sub range start
static void getPassRows(const KmerExchange &exchange, const std::vector<std::vector<KmerFileRow> > &fileRows, size_t passFirstHashSubRange, size_t passEndHashSubRange,
                        std::vector<PassRow> &rows, std::vector<size_t> &rowOffsets) {
    rows.clear();
    rowOffsets.assign(1, 0);
    for (size_t rangeHashSubRange = passFirstHashSubRange; rangeHashSubRange < passEndHashSubRange; rangeHashSubRange++) {
        for (size_t sender = 0; sender < fileRows.size(); sender++) {
            for (size_t row = rangeHashSubRange; row < fileRows[sender].size(); row += exchange.rowsPerHashRange) {
                if (fileRows[sender][row].groupCount > 0) {
                    const PassRow passRow = {sender, row, 0};
                    rows.push_back(passRow);
                }
            }
        }
        // the key splits are key ranges, so their rows in key order give the members of a group in id order whichever node sent them
        SORT_SERIAL(rows.begin() + rowOffsets.back(), rows.end(), [&](const PassRow &first, const PassRow &second) {
            return fileRows[first.file][first.row].splitFirstKey < fileRows[second.file][second.row].splitFirstKey;
        });
        rowOffsets.push_back(rows.size());
    }
}

// the rows that a node writes for its own merge send splits, which it merges itself, in memory instead of its pair file up to limit bytes
struct OwnPairs {
    char *data;
    size_t bytes;
    size_t limit;
    std::vector<std::vector<PairFileRow> > sendSplitRows;
};

// the pair file of this node and its size
struct PairFile {
    std::string name;
    FILE *file;
    BatchWriter *batchWriter;
    size_t bytes;
    OwnPairs *ownPairs;
    // the rows of every flush are written after its records as a flush: where the rows of every send split start in it, then its end, flush after flush
    std::vector<size_t> flushRowOffsets;
    // the code bytes of every partition, send split after send split
    std::vector<size_t> partitionBytes;
};

// the Rice parameter of the key gaps of a row: the keys of its partition per record, rounded down to a power of two
static unsigned int getRepKeyDiffLowBits(size_t partitionKeys, size_t recordCount) {
    return getBitWidth(std::max<size_t>(1, partitionKeys / std::max<size_t>(1, recordCount))) - 1;
}

// the targets of the records of a rep that a thread merges: up to repTargetCapacity sorted in repTargets for the records from sortedRepStart, more from the records by targetQueue
struct RepTargetMerge {
    DBKeyType *repTargets;
    size_t repTargetCapacity;
    size_t sortedRepStart;
    size_t sortedRepTargetCount;
    std::vector<size_t> recordTargets;
    RowValueQueue targetQueue;
};

// gives the targets of the records [repStart, repEnd) of a rep, each a part of the members of a group, to addTarget in id order without repeats and without the rep: sorted in repTargets if they fit, otherwise merged
template <typename AddTarget>
static void mergeRepTargets(const PairRecord *records, size_t repStart, size_t repEnd, const DBKeyType *targets, RepTargetMerge &merge, AddTarget addTarget) {
    const DBKeyType rep = records[repStart].rep;
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
        const RowValue target = targetQueue.top();
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

// the targets of a rep without repeats, and the bits of every gap between them, the width of its largest gap less one
static unsigned int getRepTargetGapBits(const PairRecord *records, size_t repStart, size_t repEnd, const DBKeyType *targets, RepTargetMerge &merge, size_t &repTargetCount) {
    DBKeyType maxGap = 0;
    DBKeyType prevTarget = 0;
    repTargetCount = 0;
    mergeRepTargets(records, repStart, repEnd, targets, merge, [&](DBKeyType target) {
        if (repTargetCount > 0) {
            maxGap = std::max<DBKeyType>(maxGap, target - prevTarget - 1);
        }
        prevTarget = target;
        repTargetCount++;
    });
    return getBitWidth(maxGap);
}

// writes the records, sorted by rep, as a row per partition to the pair file or the own pairs, then the rows as a flush; a thread writes through the first half of its write buffer and sorts in the second
static void writePairFileRows(const KmerExchange &exchange, PairFile &pairFile, PairRecord *records, size_t recordCount, DBKeyType *targets,
                              char *writeBuffers, int threads) {
    if (recordCount == 0) {
        return;
    }
    // the partitions are key ranges, so the records of a row follow each other
    const std::vector<size_t> &sendSplitKeyOffsets = exchange.sendSplitKeyOffsets;
    std::vector<size_t> rowRecordOffsets(1, 0);
    std::vector<size_t> rowFirstKeys;
    std::vector<size_t> rowKeys;
    std::vector<PairFileRow> rows;
    // the first row of every send split, then the row count
    std::vector<size_t> sendSplitRows;
    while (rowRecordOffsets.back() < recordCount) {
        const size_t rep = records[rowRecordOffsets.back()].rep;
        const size_t sendSplit = std::upper_bound(sendSplitKeyOffsets.begin(), sendSplitKeyOffsets.end(), rep) - sendSplitKeyOffsets.begin() - 1;
        const size_t partition = (rep - sendSplitKeyOffsets[sendSplit]) * exchange.partitionsPerRange / (sendSplitKeyOffsets[sendSplit + 1] - sendSplitKeyOffsets[sendSplit]);
        rowFirstKeys.push_back(getPartitionFirstKey(exchange, sendSplit, partition));
        rowKeys.push_back(getPartitionFirstKey(exchange, sendSplit, partition + 1) - rowFirstKeys.back());
        rowRecordOffsets.push_back(std::partition_point(records + rowRecordOffsets.back(), records + recordCount,
                                                        [&](const PairRecord &record) { return record.rep < rowFirstKeys.back() + rowKeys.back(); }) - records);
        while (sendSplitRows.size() <= sendSplit) {
            sendSplitRows.push_back(rows.size());
        }
        const PairFileRow row = {{0, 0, 0}, partition, 0, 0};
        rows.push_back(row);
    }
    sendSplitRows.resize(exchange.sendSplitCount + 1, rows.size());
    const unsigned int targetGapBitsWidth = getBitWidth(exchange.globalIdBits);
    // the bytes of every code of a row
#pragma omp parallel num_threads(threads)
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        RepTargetMerge merge;
        merge.repTargets = reinterpret_cast<DBKeyType *>(writeBuffers + thread_idx * KMER_FILE_BUFFER_BYTES + KMER_FILE_BUFFER_BYTES / 2);
        merge.repTargetCapacity = KMER_FILE_BUFFER_BYTES / 2 / sizeof(DBKeyType);
        merge.sortedRepStart = SIZE_MAX;
#pragma omp for schedule(dynamic, 1)
        for (size_t row = 0; row < rows.size(); row++) {
            const size_t rowEnd = rowRecordOffsets[row + 1];
            for (size_t record = rowRecordOffsets[row]; record < rowEnd; record++) {
                rows[row].recordCount += (record == rowRecordOffsets[row] || records[record].rep != records[record - 1].rep);
            }
            const unsigned int repKeyDiffLowBits = getRepKeyDiffLowBits(rowKeys[row], rows[row].recordCount);
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
                const size_t repKey = records[repStart].rep - rowFirstKeys[row];
                size_t repTargetCount = 0;
                const unsigned int targetGapBits = getRepTargetGapBits(records, repStart, repEnd, targets, merge, repTargetCount);
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
    // the rows of an own merge send split stay in memory while they fit
    OwnPairs &ownPairs = *pairFile.ownPairs;
    const size_t ownBytesBefore = ownPairs.bytes;
    std::vector<char> isOwnSendSplit(exchange.sendSplitCount, false);
    for (size_t sendSplit = 0; sendSplit < exchange.sendSplitCount; sendSplit++) {
        size_t sendSplitBytes = 0;
        for (size_t row = sendSplitRows[sendSplit]; row < sendSplitRows[sendSplit + 1]; row++) {
            for (size_t code = 0; code < PAIR_CODE_COUNT; code++) {
                sendSplitBytes += rows[row].offsets[code + 1];
            }
        }
        isOwnSendSplit[sendSplit] = sendSplit % MMseqsNode::count == MMseqsNode::rank && sendSplit / MMseqsNode::count < exchange.selfMergedSendSplitCount
                                    && ownPairs.bytes + sendSplitBytes <= ownPairs.limit;
        size_t &bytes = isOwnSendSplit[sendSplit] ? ownPairs.bytes : pairFile.bytes;
        for (size_t row = sendSplitRows[sendSplit]; row < sendSplitRows[sendSplit + 1]; row++) {
            rows[row].offsets[0] = bytes;
            for (size_t code = 0; code < PAIR_CODE_COUNT; code++) {
                rows[row].offsets[code + 1] += rows[row].offsets[code];
            }
            bytes = rows[row].offsets[PAIR_CODE_COUNT];
            if (isOwnSendSplit[sendSplit] == false) {
                pairFile.partitionBytes[sendSplit * exchange.partitionsPerRange + rows[row].partition] += bytes - rows[row].offsets[0];
            }
        }
    }
    if (ownPairs.bytes > ownBytesBefore) {
        ownPairs.data = static_cast<char *>(realloc(ownPairs.data, ownPairs.bytes));
        Util::checkAllocation(ownPairs.data, "Can not allocate ownPairs memory in writePairFileRows");
    }
    // a record: the gap to the key after the record before, the target count, the first target and the bits of the gaps to the other targets
#pragma omp parallel num_threads(threads)
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        char *writeBuffer = writeBuffers + thread_idx * KMER_FILE_BUFFER_BYTES;
        RepTargetMerge merge;
        merge.repTargets = reinterpret_cast<DBKeyType *>(writeBuffer + KMER_FILE_BUFFER_BYTES / 2);
        merge.repTargetCapacity = KMER_FILE_BUFFER_BYTES / 2 / sizeof(DBKeyType);
        merge.sortedRepStart = SIZE_MAX;
#pragma omp for schedule(dynamic, 1)
        for (size_t row = 0; row < rows.size(); row++) {
            const bool isOwnRow = isOwnSendSplit[std::upper_bound(sendSplitRows.begin(), sendSplitRows.end(), row) - sendSplitRows.begin() - 1];
            KmerCodeWriter codeWrites[PAIR_CODE_COUNT];
            for (size_t code = 0; code < PAIR_CODE_COUNT; code++) {
                const size_t writePos = rows[row].offsets[code];
                codeWrites[code] = isOwnRow
                    ? createMemoryCodeWriter(ownPairs.data, ownPairs.bytes, writePos)
                    : createFileCodeWriter(fileno(pairFile.file), pairFile.name.c_str(), writeBuffer + code * KMER_FILE_BUFFER_BYTES / 4, KMER_FILE_BUFFER_BYTES / 4, writePos, pairFile.batchWriter);
            }
            KmerCodeWriter &indexWrite = codeWrites[PAIR_INDEX_CODES];
            KmerCodeWriter &targetWrite = codeWrites[PAIR_TARGET_CODES];
            const unsigned int repKeyDiffLowBits = getRepKeyDiffLowBits(rowKeys[row], rows[row].recordCount);
            const size_t rowEnd = rowRecordOffsets[row + 1];
            size_t nextRepKey = 0;
            for (size_t repStart = rowRecordOffsets[row]; repStart < rowEnd;) {
                size_t repEnd = repStart + 1;
                while (repEnd < rowEnd && records[repEnd].rep == records[repStart].rep) {
                    repEnd++;
                }
                for (size_t i = repStart + 16; i < std::min(repEnd + 16, rowEnd); i++) {
                    __builtin_prefetch(targets + records[i].targetOffset);
                }
                const size_t repKey = records[repStart].rep - rowFirstKeys[row];
                size_t repTargetCount = 0;
                const unsigned int targetGapBits = getRepTargetGapBits(records, repStart, repEnd, targets, merge, repTargetCount);
                putRice(indexWrite, repKey - nextRepKey, repKeyDiffLowBits);
                putEliasGamma(indexWrite, repTargetCount);
                size_t repTarget = 0;
                DBKeyType prevTarget = 0;
                mergeRepTargets(records, repStart, repEnd, targets, merge, [&](DBKeyType target) {
                    if (repTarget == 0) {
                        putBits(indexWrite, target, exchange.globalIdBits);
                        if (repTargetCount > 1) {
                            putBits(indexWrite, targetGapBits, targetGapBitsWidth);
                        }
                    } else {
                        putBits(targetWrite, target - prevTarget - 1, targetGapBits);
                    }
                    prevTarget = target;
                    repTarget++;
                });
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
        pairFile.flushRowOffsets.push_back(pairFile.bytes);
        const PairFileRow *sendSplitRowsStart = rows.data() + sendSplitRows[sendSplit];
        const size_t sendSplitRowCount = sendSplitRows[sendSplit + 1] - sendSplitRows[sendSplit];
        if (isOwnSendSplit[sendSplit]) {
            ownPairs.sendSplitRows[sendSplit].insert(ownPairs.sendSplitRows[sendSplit].end(), sendSplitRowsStart, sendSplitRowsStart + sendSplitRowCount);
            continue;
        }
        pairFile.batchWriter->write(fileno(pairFile.file), pairFile.name.c_str(), reinterpret_cast<const char *>(sendSplitRowsStart), sendSplitRowCount * sizeof(PairFileRow), pairFile.bytes);
        pairFile.bytes += sendSplitRowCount * sizeof(PairFileRow);
    }
    pairFile.flushRowOffsets.push_back(pairFile.bytes);
}

// the output of a receiving node over its passes, written once full: threads take record regions from its start and target regions from its end, the filled record regions are moved together before sorting
struct PairOutput {
    PairRecord *records;
    DBKeyType *targets;
    size_t bytes;
    size_t regionBytes;
    size_t recordsTaken;
    size_t targetsTaken;
    std::vector<std::pair<size_t, size_t> > recordRegions;
    size_t flushes;
};

// the record region and the target region that a thread fills
struct PairOutputRegion {
    size_t recordStart;
    size_t recordPos;
    size_t recordEnd;
    size_t targetPos;
    size_t targetEnd;
};

// a thread keeps its regions while they hold recordCount records and targetCount targets more, otherwise takes new ones of regionBytes or, if these do not fit, of the size it needs; false if the output is full
static bool takePairOutput(PairOutput &output, PairOutputRegion &region, size_t recordCount, size_t targetCount) {
    const bool hasRecords = (region.recordEnd - region.recordPos >= recordCount);
    const bool hasTargets = (region.targetEnd - region.targetPos >= targetCount);
    if (hasRecords && hasTargets) {
        return true;
    }
    bool isTaken = false;
#pragma omp critical(takePairOutput)
    {
        size_t reservedRecordCount = hasRecords ? 0 : std::max(recordCount, output.regionBytes / sizeof(PairRecord));
        size_t reservedTargetCount = hasTargets ? 0 : std::max(targetCount, output.regionBytes / sizeof(DBKeyType));
        if ((output.recordsTaken + reservedRecordCount) * sizeof(PairRecord) + (output.targetsTaken + reservedTargetCount) * sizeof(DBKeyType) > output.bytes) {
            reservedRecordCount = hasRecords ? 0 : recordCount;
            reservedTargetCount = hasTargets ? 0 : targetCount;
        }
        if ((output.recordsTaken + reservedRecordCount) * sizeof(PairRecord) + (output.targetsTaken + reservedTargetCount) * sizeof(DBKeyType) <= output.bytes) {
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
                region.targetPos = output.bytes / sizeof(DBKeyType) - output.targetsTaken;
                region.targetEnd = region.targetPos + reservedTargetCount;
            }
            isTaken = true;
        }
    }
    return isTaken;
}

// writes the records of the output sorted by rep and empties the output and the regions of every thread
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
    output.flushes += (recordCount > 0);
    output.recordsTaken = 0;
    output.targetsTaken = 0;
    output.recordRegions.clear();
}

// the rows of a pass of the representative node, partition after partition, and where the rows of every partition start
static void getPairPassRows(const std::vector<std::vector<std::pair<size_t, size_t> > > &partitionRows, size_t passFirstPartition, size_t passEndPartition,
                            std::vector<PassRow> &rows, std::vector<size_t> &rowOffsets) {
    rows.clear();
    rowOffsets.assign(1, 0);
    for (size_t partition = passFirstPartition; partition < passEndPartition; partition++) {
        for (size_t row = 0; row < partitionRows[partition].size(); row++) {
            const PassRow passRow = {partitionRows[partition][row].first, partitionRows[partition][row].second, 0};
            rows.push_back(passRow);
        }
        rowOffsets.push_back(rows.size());
    }
}

// reads the next record of a row: its key, its first target and how many follow
static inline void readPairRecord(PairRowRead &read, unsigned int globalIdBits) {
    KmerCodeReader &indexRead = read.codeReads[PAIR_INDEX_CODES];
    read.repKey = read.nextRepKey + getRice(indexRead, read.repKeyDiffLowBits);
    read.nextRepKey = read.repKey + 1;
    read.targetsLeft = getEliasGamma(indexRead) - 1;
    read.target = getBits(indexRead, globalIdBits);
    read.targetGapBits = (read.targetsLeft > 0) ? getBits(indexRead, getBitWidth(globalIdBits)) : 0;
    read.recordsLeft--;
}

// gives the targets of key in the rows whose record has it to addTarget, merged by id without repeats, then queues the next records of these rows
template <typename AddTarget>
static void mergeKeyTargets(std::vector<PairRowRead> &reads, RowValueQueue &repKeyQueue, RowValueQueue &targetQueue, std::vector<unsigned int> &keyRows, size_t key,
                            unsigned int globalIdBits, AddTarget addTarget) {
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
        const RowValue target = targetQueue.top();
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

// the readers of the rows of a partition at their start, with their codes in codeBuffer or, with codeReadBufferBytes, in buffers from codeBuffer on
static void getPartitionRowReads(const std::vector<ReceivedFile> &receivedPairFiles, const std::vector<std::vector<PairFileRow> > &pairFileRows, const PassRow *partitionPassRows,
                                 size_t partitionRowCount, size_t partitionKeys, char *codeBuffer, size_t codeReadBufferBytes, std::vector<PairRowRead> &rowReads) {
    rowReads.assign(partitionRowCount, PairRowRead());
    for (size_t passRow = 0; passRow < partitionRowCount; passRow++) {
        const PassRow &partitionPassRow = partitionPassRows[passRow];
        const PairFileRow &pairRow = pairFileRows[partitionPassRow.file][partitionPassRow.row];
        PairRowRead &read = rowReads[passRow];
        for (size_t code = 0; code < PAIR_CODE_COUNT; code++) {
            read.codeReads[code] = getRowCodeReader(receivedPairFiles[partitionPassRow.file], pairRow, partitionPassRow.bufferOffset, code, codeBuffer,
                                                    codeBuffer + (passRow * PAIR_CODE_COUNT + code) * codeReadBufferBytes, codeReadBufferBytes);
        }
        read.repKeyDiffLowBits = getRepKeyDiffLowBits(partitionKeys, pairRow.recordCount);
        read.recordsLeft = pairRow.recordCount;
    }
}

// with threads: the row readers of a partition, the target count of every key, key tasks of equal work and where every row starts them
static void countPartitionTargets(const KmerExchange &exchange, const std::vector<ReceivedFile> &receivedPairFiles, const std::vector<std::vector<PairFileRow> > &pairFileRows,
                                  const PassRow *partitionPassRows, size_t partitionRowCount, char *codeBuffer, size_t partitionKeys, size_t partitionTargets, PartitionMerge &merge, int threads) {
    std::vector<PairRowRead> &rowReads = merge.rowReads;
    std::vector<size_t> &keyOffsets = merge.keyOffsets;
    getPartitionRowReads(receivedPairFiles, pairFileRows, partitionPassRows, partitionRowCount, partitionKeys, codeBuffer, 0, rowReads);
    keyOffsets.resize(partitionKeys + 1);
    // the target count of every key, then where its targets start, summed in a part of the keys per thread and then over the parts
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
                __sync_fetch_and_add(&keyOffsets[read.repKey + 1], read.targetsLeft + 1);
            }
        }
#pragma omp for schedule(static)
        for (int part = 0; part < threads; part++) {
            const size_t partBegin = 1 + partitionKeys * part / threads;
            const size_t partEnd = 1 + partitionKeys * (part + 1) / threads;
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
    // tasks of keys with about the same keys and targets: a task starts at the first key whose keys and targets before it reach a multiple of keyTaskWeight
    const size_t keyTaskWeight = std::max<size_t>(1, (partitionTargets + partitionKeys + TASKS_PER_THREAD * threads - 1) / (TASKS_PER_THREAD * threads));
    merge.keyTaskOffsets.assign(1, 0);
    for (size_t task = 1; merge.keyTaskOffsets.back() < partitionKeys; task++) {
        size_t low = merge.keyTaskOffsets.back();
        size_t high = partitionKeys;
        while (low < high) {
            const size_t mid = low + (high - low) / 2;
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
    // every row starts a key task at its first record with a key from the task on
    const size_t keyTaskCount = merge.keyTaskOffsets.size() - 1;
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

// with threads: merges the targets of every key from the rows by id without repeats, a key task at a time, and finds where the entries and index cells of every key task start
static void mergePartitionTargets(const KmerExchange &exchange, DBReader<DBKeyType> &seqDbr, size_t partitionFirstKey, PartitionMerge &merge, DBKeyType *targets, int threads) {
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    const std::vector<size_t> &keyOffsets = merge.keyOffsets;
    merge.targets = targets;
    const size_t keyTaskCount = merge.keyTaskOffsets.size() - 1;
    const size_t rowCount = merge.rowReads.size();
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
            size_t keyTargetCount = 0;
            mergeKeyTargets(reads, repKeyQueue, targetQueue, keyRows, key, exchange.globalIdBits, [&](DBKeyType target) { keyTargets[keyTargetCount++] = target; });
            if (keyOffsets[key] + keyTargetCount < keyOffsets[key + 1]) {
                keyTargets[keyTargetCount] = DB_KEY_INVALID;
            }
            if (keyTargetCount == 0 || isRepresentative(partitionFirstKey + key) == false) {
                continue;
            }
            size_t entryBytes = getPrefKeyBytes(partitionFirstKey + key) + 1;
            size_t targetRange = SIZE_MAX;
            size_t targetResidues = 0;
            for (size_t target = 0; target < keyTargetCount; target++) {
                entryBytes += getPrefKeyBytes(keyTargets[target] - ((target > 0) ? keyTargets[target - 1] : 0));
                targetResidues += seqDbr.getSeqLenFromRange(targetRange, keyTargets[target]);
            }
            merge.keyTaskDataOffsets[keyTask + 1] += entryBytes;
            merge.keyTaskEntryOffsets[keyTask + 1]++;
            merge.keyTaskCostOffsets[keyTask + 1] += (index[seqDbr.findIdRange(partitionFirstKey + key)].length - 2) * targetResidues;
            const PartitionCut cut = { key + 1, merge.keyTaskDataOffsets[keyTask + 1], merge.keyTaskCostOffsets[keyTask + 1] };
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

// the pref files of this node: the entries and their index cells, and a pair of write buffers per thread
struct PrefFiles {
    std::string dataName;
    std::string indexName;
    FILE *dataFile;
    FILE *indexFile;
    char *writeBuffers;
    BatchWriter *batchWriter;
};

// with threads: writes the entries of a merged partition at dataOffset and indexOffset of the pref files, a key task at a time
static void writePartition(const PrefFiles &prefFiles, size_t partitionFirstKey, const PartitionMerge &merge, size_t dataOffset, size_t indexOffset, size_t firstWriteBuffer, int threads) {
    const size_t keyTaskCount = merge.keyTaskOffsets.size() - 1;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (size_t keyTask = 0; keyTask < keyTaskCount; keyTask++) {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        char *writeBuffer = prefFiles.writeBuffers + 2 * (firstWriteBuffer + thread_idx) * KMER_FILE_BUFFER_BYTES;
        KmerCodeWriter dataWrite = createFileCodeWriter(fileno(prefFiles.dataFile), prefFiles.dataName.c_str(), writeBuffer, KMER_FILE_BUFFER_BYTES, dataOffset + merge.keyTaskDataOffsets[keyTask], prefFiles.batchWriter);
        KmerCodeWriter indexWrite = createFileCodeWriter(fileno(prefFiles.indexFile), prefFiles.indexName.c_str(), writeBuffer + KMER_FILE_BUFFER_BYTES, KMER_FILE_BUFFER_BYTES,
                                                    indexOffset + merge.keyTaskEntryOffsets[keyTask] * 2 * sizeof(uint64_t), prefFiles.batchWriter);
        for (size_t key = merge.keyTaskOffsets[keyTask]; key < merge.keyTaskOffsets[keyTask + 1]; key++) {
            if (merge.keyOffsets[key + 1] > merge.keyOffsets[key] && isRepresentative(partitionFirstKey + key)) {
                const uint64_t indexCell[2] = {partitionFirstKey + key, dataWrite.writePos + dataWrite.bufferPos};
                putBytes(indexWrite, reinterpret_cast<const char *>(indexCell), sizeof(indexCell));
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

// the rows that a node sends to itself for the first of its own hash ranges, which it takes itself, in memory instead of its k-mer file up to limit bytes
struct OwnKmers {
    char *data;
    size_t bytes;
    size_t limit;
    std::vector<std::vector<KmerFileRow> > hashRangeRows;
    std::vector<size_t> hashRanges;
};

// linfinity sending: the nodes take the send splits one at a time, the k-mers of a send split in key splits, written to <pref>.kmers.<node>
template <typename T, bool includeAdjacency, typename L>
static size_t sendKmers(Parameters &par, DBReader<DBKeyType> &seqDbr, BaseMatrix *subMat, const KmerExchange &exchange, MMseqsNode::WorkQueue &sendQueue,
                        size_t representativeBitsMemory, std::vector<size_t> &sendSplits, std::string &sendValues, OwnKmers &ownKmers) {
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    const float kmersPerSequenceScale = par.kmersPerSequenceScale.values.aminoacid();
    const size_t readChunkBytes = std::min(MAX_INFLIGHT_READ_BYTES, exchange.keySplitMemoryLimit / 4);
    // a key split spans so few keys that their ids fit next to the whole sequence bit, the hash and the adjKey
    const size_t keySplitKeyLimit = (1ULL << (128 - 1 - 64 - exchange.adjKeyBits)) - 1;
    const std::string kmerFile = par.db2 + ".kmers." + SSTR(MMseqsNode::rank);
    FILE *kmerFileHandle = FileUtil::openAndDelete(kmerFile.c_str(), "w");
    size_t kmerFileBytes = 0;
    // the rows of every key split are written after its codes as a flush: where the rows of every hash range start in it, then its end, flush after flush
    std::vector<size_t> flushRowOffsets;
    // the rows of a key split, hash range after hash range
    std::vector<KmerFileRow> keySplitRows(exchange.hashRangeCount * exchange.rowsPerHashRange);
    std::vector<size_t> hashSubRangeOffsets(exchange.hashSubRangeCount + 1);
    std::vector<size_t> wholeSequenceOffsets(exchange.hashSubRangeCount + 1);
    char *kmerWriteBuffer = new(std::nothrow) char[par.threads * KMER_FILE_BUFFER_BYTES];
    Util::checkAllocation(kmerWriteBuffer, "Can not allocate kmerWriteBuffer memory in sendKmers");
    BatchWriter batchWriter(exchange.writeBehindBytes, par.threads, fileno(kmerFileHandle));
    // the k-mers of the largest send split so far, kept for the next ones so their pages are not mapped again, while they fit with the group buffers
    KmerPositionBit *hashSeqPair = NULL;
    size_t hashSeqPairCapacity = 0;
    std::vector<char> isOwnKeySplitHashRange(exchange.hashRangeCount);
    for (size_t sendSplit = sendQueue.takeNextItem(); sendSplit < exchange.sendSplitCount; sendSplit = sendQueue.takeNextItem()) {
        sendSplits.push_back(sendSplit);
        setSendSplit(par, exchange, sendSplit);
        const size_t dbFrom = exchange.sendSplitKeyOffsets[sendSplit];
        const size_t dbSize = exchange.sendSplitKeyOffsets[sendSplit + 1] - dbFrom;
        const size_t firstRange = (dbSize > 0) ? seqDbr.findIdRange(dbFrom) : seqDbr.getIndexSize();
        const auto countRepresentatives = [&](size_t from, size_t to) {
            return (representativeBits == NULL) ? to - from : Util::countSetBits(representativeBits, from - dbFrom, to - dbFrom);
        };
        // the k-mers of a key split and a group buffer per thread fit the memory beside the own rows, with fewer k-mers per key split if needed
        const size_t keySplitMemoryLimit = exchange.keySplitMemoryLimit - ownKmers.bytes;
        size_t kmerCapacity = keySplitMemoryLimit / sizeof(KmerPositionBit);
        std::vector<size_t> keySplitOffsets;
        size_t maxKeySplitKmers = 0;
        size_t maxKeySplitKeys = 0;
        size_t groupBufferWords = 0;
        size_t fixedBytes = 0;
        while (true) {
            keySplitOffsets.assign(1, dbFrom);
            size_t keySplitKmers = 0;
            maxKeySplitKmers = 0;
            for (size_t range = firstRange; range < seqDbr.getIndexSize() && index[range].id < dbFrom + dbSize; range++) {
                const size_t rangeFrom = std::max(static_cast<size_t>(index[range].id), dbFrom);
                const size_t rangeTo = std::min((range + 1 < seqDbr.getIndexSize()) ? static_cast<size_t>(index[range + 1].id) : seqDbr.getSize(), dbFrom + dbSize);
                const size_t seqKmers = computeSeqKmerCount(static_cast<int>(index[range].length - 2), par.kmerSize, par.kmersPerSequence, kmersPerSequenceScale);
                if (seqKmers > kmerCapacity) {
                    Debug(Debug::ERROR) << "Not enough memory to extract the k-mers of a sequence of length " << index[range].length - 2 << ". Raise --split-memory-limit.\n";
                    EXIT(EXIT_FAILURE);
                }
                size_t rangeRepresentatives = countRepresentatives(rangeFrom, rangeTo);
                // the next key split starts at the first representative that does not fit, or where its keys would need more id bits
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
                        fitCount = countRepresentatives(cutKey, nextKeySplitKey);
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
            // a group buffer holds the ids of a group, or for a bigger group a bit per key of the key split and the count of set bits before every 512 of them
            groupBufferWords = (maxKeySplitKeys + 63) / 64 + (maxKeySplitKeys + 511) / 512;
            fixedBytes = par.threads * groupBufferWords * sizeof(uint64_t);
            if (fixedBytes + maxKeySplitKmers * sizeof(KmerPositionBit) <= keySplitMemoryLimit) {
                break;
            }
            if (maxKeySplitKmers == 0) {
                Debug(Debug::ERROR) << "Not enough memory to run the kmermatcher. Memory limit: " << exchange.memoryLimit
                                    << " bytes, the group buffers need: " << representativeBitsMemory + fixedBytes << " bytes.\n"
                                    << "Raise --split-memory-limit.\n";
                EXIT(EXIT_FAILURE);
            }
            // fewer k-mers per key split take fewer keys per group buffer
            kmerCapacity = std::min(maxKeySplitKmers - 1, (fixedBytes < keySplitMemoryLimit) ? (keySplitMemoryLimit - fixedBytes) / sizeof(KmerPositionBit) : maxKeySplitKmers / 2);
        }
        const size_t keySplitCount = keySplitOffsets.size() - 1;

        const unsigned int idBits = getBitWidth(maxKeySplitKeys);
        KmerPositionBit::idShift = 0;
        KmerPositionBit::seqLenShift = idBits;
        KmerPositionBit::adjKeyShift = idBits;
        KmerPositionBit::kmerShift = idBits + exchange.adjKeyBits;
        KmerPositionBit::maxSeqLen = 0;

        if (maxKeySplitKmers > hashSeqPairCapacity || fixedBytes + hashSeqPairCapacity * sizeof(KmerPositionBit) > keySplitMemoryLimit) {
            Util::releasePages(hashSeqPair, hashSeqPairCapacity * sizeof(KmerPositionBit));
            delete[] hashSeqPair;
            hashSeqPairCapacity = maxKeySplitKmers;
            hashSeqPair = new(std::nothrow) KmerPositionBit[hashSeqPairCapacity];
            Util::checkAllocation(hashSeqPair, "Can not allocate hashSeqPair memory in sendKmers");
        }
        uint64_t *groupBuffers = new(std::nothrow) uint64_t[par.threads * groupBufferWords];
        Util::checkAllocation(groupBuffers, "Can not allocate groupBuffers memory in sendKmers");
        for (size_t keySplit = 0; keySplit < keySplitCount; keySplit++) {
            std::vector<size_t> readChunkOffsets(1, keySplitOffsets[keySplit]);
            size_t readChunkUsedBytes = 0;
            const size_t keySplitFirstRange = (keySplitOffsets[keySplit] < keySplitOffsets[keySplit + 1]) ? seqDbr.findIdRange(keySplitOffsets[keySplit]) : seqDbr.getIndexSize();
            for (size_t range = keySplitFirstRange; range < seqDbr.getIndexSize() && index[range].id < keySplitOffsets[keySplit + 1]; range++) {
                const size_t rangeFrom = std::max(static_cast<size_t>(index[range].id), keySplitOffsets[keySplit]);
                const size_t rangeTo = std::min((range + 1 < seqDbr.getIndexSize()) ? static_cast<size_t>(index[range + 1].id) : seqDbr.getSize(), keySplitOffsets[keySplit + 1]);
                for (size_t key = rangeFrom; key < rangeTo; ) {
                    if (readChunkUsedBytes > 0 && readChunkUsedBytes + index[range].length > readChunkBytes) {
                        readChunkOffsets.push_back(key);
                        readChunkUsedBytes = 0;
                    }
                    const size_t readChunkSeqCount = std::min(rangeTo - key, std::max<size_t>(1, (readChunkBytes - readChunkUsedBytes) / index[range].length));
                    key += readChunkSeqCount;
                    readChunkUsedBytes += readChunkSeqCount * index[range].length;
                }
            }
            readChunkOffsets.push_back(keySplitOffsets[keySplit + 1]);
            const size_t kmerCount = fillKmerPositionArray<Parameters::DBTYPE_AMINO_ACIDS, T, includeAdjacency, L>(
                hashSeqPair, maxKeySplitKmers, seqDbr, par, subMat, true, 0, SIZE_MAX, NULL, readChunkOffsets).first;

            SORT_PARALLEL(hashSeqPair, hashSeqPair + kmerCount);

            // the whole sequence records follow the k-mer records, both in hash order
            const size_t wholeSequenceStart = std::partition_point(hashSeqPair, hashSeqPair + kmerCount,
                                                                   [](const KmerPositionBit &kmer) { return kmer.isWholeSequence() == false; }) - hashSeqPair;
            const auto compareHash = [](const KmerPositionBit &kmer, uint64_t hash) { return static_cast<uint64_t>(kmer.getKmer()) < hash; };
#pragma omp parallel for schedule(static) num_threads(par.threads)
            for (size_t hashSubRange = 0; hashSubRange < exchange.hashSubRangeCount; hashSubRange++) {
                const uint64_t hashSubRangeStart = static_cast<uint64_t>(getHashSubRangeStart(hashSubRange, exchange.hashSubRangeCount, 0));
                hashSubRangeOffsets[hashSubRange] = std::lower_bound(hashSeqPair, hashSeqPair + wholeSequenceStart, hashSubRangeStart, compareHash) - hashSeqPair;
                wholeSequenceOffsets[hashSubRange] = std::lower_bound(hashSeqPair + wholeSequenceStart, hashSeqPair + kmerCount, hashSubRangeStart, compareHash) - hashSeqPair;
            }
            hashSubRangeOffsets[exchange.hashSubRangeCount] = wholeSequenceStart;
            wholeSequenceOffsets[exchange.hashSubRangeCount] = kmerCount;
            const size_t keySplitWords = (keySplitOffsets[keySplit + 1] - keySplitOffsets[keySplit] + 63) / 64;
            // the code bytes of every row, as they are written below
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
                const bool isWholeSequence = getRowHashSubRanges(exchange, row / exchange.rowsPerHashRange, row % exchange.rowsPerHashRange, hashSubRange, hashSubRangeEnd);
                const size_t from = (isWholeSequence ? wholeSequenceOffsets : hashSubRangeOffsets)[hashSubRange];
                const size_t to = (isWholeSequence ? wholeSequenceOffsets : hashSubRangeOffsets)[hashSubRangeEnd];
                const unsigned int hashShift = isWholeSequence ? 0 : exchange.kmerHashShift;
                KmerFileRow &hashSubRangeRow = keySplitRows[row];
                const KmerFileRow emptyRow = {static_cast<DBKeyType>(keySplitOffsets[keySplit]), idBits, {0, 0, 0, 0}, 0};
                hashSubRangeRow = emptyRow;
                for (size_t pos = from; pos < to; pos++) {
                    hashSubRangeRow.groupCount += (pos == from || hashSeqPair[pos].getKmer() != hashSeqPair[pos - 1].getKmer());
                }
                const unsigned int hashDiffLowBits = getHashDiffLowBits(hashSubRange, hashSubRangeEnd, exchange.hashSubRangeCount, hashShift, hashSubRangeRow.groupCount);
                uint64_t prevHash = static_cast<uint64_t>(getHashSubRangeStart(hashSubRange, exchange.hashSubRangeCount, hashShift));
                size_t codeBits[KMER_CODE_COUNT] = {0, 0, 0};
                for (size_t groupStart = from, groupEnd = from; groupStart < to; groupStart = groupEnd) {
                    const uint64_t hash = static_cast<uint64_t>(hashSeqPair[groupStart].getKmer()) >> hashShift;
                    size_t repCandidateCount = 1;
                    for (groupEnd = groupStart + 1; groupEnd < to && hashSeqPair[groupEnd].getKmer() == hashSeqPair[groupStart].getKmer(); groupEnd++) {
                        repCandidateCount += (hashSeqPair[groupEnd].getAdjKey() != hashSeqPair[groupEnd - 1].getAdjKey());
                    }
                    const size_t memberCount = getGroupIds(hashSeqPair, groupStart, groupEnd, groupBuffer, groupBufferWords, setBitCounts, keySplitWords);
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
            // the own hash ranges keep their rows of the key split in memory up to the first that does not fit beside the key split
            const size_t ownKeySplitLimit = std::min(exchange.keySplitMemoryLimit - ownKmers.bytes - fixedBytes - hashSeqPairCapacity * sizeof(KmerPositionBit),
                                                     ownKmers.limit - std::min(ownKmers.limit, ownKmers.bytes));
            size_t ownKeySplitBytes = 0;
            isOwnKeySplitHashRange.assign(exchange.hashRangeCount, false);
            for (size_t i = 0; i < ownKmers.hashRanges.size(); i++) {
                const size_t hashRange = ownKmers.hashRanges[i];
                size_t hashRangeBytes = 0;
                for (size_t row = hashRange * exchange.rowsPerHashRange; row < (hashRange + 1) * exchange.rowsPerHashRange; row++) {
                    hashRangeBytes += keySplitRows[row].offsets[KMER_CODE_COUNT];
                }
                if (ownKeySplitBytes + hashRangeBytes > ownKeySplitLimit) {
                    break;
                }
                isOwnKeySplitHashRange[hashRange] = true;
                ownKeySplitBytes += hashRangeBytes;
            }
            // the rows of the hash ranges follow each other in the k-mer file or in memory
            for (size_t row = 0; row < keySplitRows.size(); row++) {
                size_t &bytes = isOwnKeySplitHashRange[row / exchange.rowsPerHashRange] ? ownKmers.bytes : kmerFileBytes;
                for (size_t code = 0; code <= KMER_CODE_COUNT; code++) {
                    keySplitRows[row].offsets[code] += bytes;
                }
                bytes = keySplitRows[row].offsets[KMER_CODE_COUNT];
            }
            if (ownKeySplitBytes > 0) {
                ownKmers.data = static_cast<char *>(realloc(ownKmers.data, ownKmers.bytes));
                Util::checkAllocation(ownKmers.data, "Can not allocate ownKmers memory in sendKmers");
            }
            // a row writes its codes in one pass over its groups: the ids of the members, the rep candidates and the index, each at its offset
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
                const bool isWholeSequence = getRowHashSubRanges(exchange, row / exchange.rowsPerHashRange, row % exchange.rowsPerHashRange, hashSubRange, hashSubRangeEnd);
                const size_t from = (isWholeSequence ? wholeSequenceOffsets : hashSubRangeOffsets)[hashSubRange];
                const size_t to = (isWholeSequence ? wholeSequenceOffsets : hashSubRangeOffsets)[hashSubRangeEnd];
                const unsigned int hashShift = isWholeSequence ? 0 : exchange.kmerHashShift;
                const KmerFileRow &hashSubRangeRow = keySplitRows[row];
                const size_t codeBufferSize = KMER_FILE_BUFFER_BYTES / KMER_CODE_COUNT;
                KmerCodeWriter codeWrites[KMER_CODE_COUNT];
                for (size_t code = 0; code < KMER_CODE_COUNT; code++) {
                    const size_t writePos = hashSubRangeRow.offsets[code];
                    codeWrites[code] = isOwnKeySplitHashRange[row / exchange.rowsPerHashRange]
                        ? createMemoryCodeWriter(ownKmers.data, ownKmers.bytes, writePos)
                        : createFileCodeWriter(fileno(kmerFileHandle), kmerFile.c_str(), kmerWriteBuffer + thread_idx * KMER_FILE_BUFFER_BYTES + code * codeBufferSize, codeBufferSize, writePos, &batchWriter);
                }
                const unsigned int hashDiffLowBits = getHashDiffLowBits(hashSubRange, hashSubRangeEnd, exchange.hashSubRangeCount, hashShift, hashSubRangeRow.groupCount);
                uint64_t prevHash = static_cast<uint64_t>(getHashSubRangeStart(hashSubRange, exchange.hashSubRangeCount, hashShift));
                for (size_t groupStart = from, groupEnd = from; groupStart < to; groupStart = groupEnd) {
                    const uint64_t hash = static_cast<uint64_t>(hashSeqPair[groupStart].getKmer()) >> hashShift;
                    size_t repCandidateCount = 1;
                    for (groupEnd = groupStart + 1; groupEnd < to && hashSeqPair[groupEnd].getKmer() == hashSeqPair[groupStart].getKmer(); groupEnd++) {
                        repCandidateCount += (hashSeqPair[groupEnd].getAdjKey() != hashSeqPair[groupEnd - 1].getAdjKey());
                    }
                    const size_t memberCount = getGroupIds(hashSeqPair, groupStart, groupEnd, groupBuffer, groupBufferWords, setBitCounts, keySplitWords);
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
                    // a rep candidate is the first member of an adjKey, so the smallest id, as its index in the members of the group; a whole sequence has none
                    const unsigned int memberIndexBits = getBitWidth(memberCount - 1);
                    for (size_t pos = groupStart; isWholeSequence == false && pos < groupEnd; pos++) {
                        if (pos > groupStart && hashSeqPair[pos].getAdjKey() == hashSeqPair[pos - 1].getAdjKey()) {
                            continue;
                        }
                        const DBKeyType id = hashSeqPair[pos].getId();
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
                // a code ends at a byte boundary, where the next one starts
                for (size_t code = 0; code < KMER_CODE_COUNT; code++) {
                    putBits(codeWrites[code], 0, (CHAR_BIT - codeWrites[code].bitCount) % CHAR_BIT);
                    if (codeWrites[code].fd != -1) {
                        writeKmerCodeBuffer(codeWrites[code]);
                    }
                }
            }
            for (size_t hashRange = 0; hashRange < exchange.hashRangeCount; hashRange++) {
                flushRowOffsets.push_back(kmerFileBytes);
                const KmerFileRow *hashRangeRows = keySplitRows.data() + hashRange * exchange.rowsPerHashRange;
                if (isOwnKeySplitHashRange[hashRange]) {
                    ownKmers.hashRangeRows[hashRange].insert(ownKmers.hashRangeRows[hashRange].end(), hashRangeRows, hashRangeRows + exchange.rowsPerHashRange);
                    continue;
                }
                batchWriter.write(fileno(kmerFileHandle), kmerFile.c_str(), reinterpret_cast<const char *>(hashRangeRows), exchange.rowsPerHashRange * sizeof(KmerFileRow), kmerFileBytes);
                kmerFileBytes += exchange.rowsPerHashRange * sizeof(KmerFileRow);
            }
            flushRowOffsets.push_back(kmerFileBytes);
        }
        delete[] groupBuffers;
    }
    seqDbr.releaseDataPages();
    batchWriter.waitWrites();
    MMseqsNode::addPhaseSeconds(sendValues, "waiting to write k-mers", batchWriter.getWaitSeconds() / par.threads);
    delete[] kmerWriteBuffer;
    Util::releasePages(hashSeqPair, hashSeqPairCapacity * sizeof(KmerPositionBit));
    delete[] hashSeqPair;
    writeRowTable(kmerFileHandle, kmerFile, exchange.hashRangeCount, flushRowOffsets, std::vector<size_t>(), kmerFileBytes);
    return kmerFileBytes;
}

// a hash range that a node receives: its rows of every k-mer file, the reads and bytes of every hash sub range, its passes with the bytes they read and receive, and their buffers
struct ReceivedHashRange {
    size_t hashRange;
    std::vector<std::vector<KmerFileRow> > fileRows;
    std::vector<size_t> hashSubRangeReads;
    std::vector<size_t> hashSubRangeBytes;
    std::vector<size_t> passHashSubRangeOffsets;
    std::vector<size_t> passBytes;
    std::vector<size_t> passReceivedBytes;
    size_t maxTaskStarts;
    size_t codeBufferBytes;
};

// a task of a receiving pass is done, or stops at a group whose records do not fit the output that is full, or that do not fit the empty output either
enum { TASK_DONE, TASK_OUTPUT_FULL, TASK_GROUP_TOO_BIG };

// linfinity receiving: the nodes take the hash ranges one at a time, in passes, and write the pairs of the chosen reps to <pref>.pairs.<node> or to ownPairs
static size_t receiveKmers(Parameters &par, DBReader<DBKeyType> &seqDbr, const KmerExchange &exchange, MMseqsNode::WorkQueue &hashQueue, std::vector<size_t> &hashRanges,
                           std::string &receiveValues, size_t &pairBytes, OwnKmers &ownKmers, OwnPairs &ownPairs) {
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();

    PairFile pairFile;
    pairFile.name = par.db2 + ".pairs." + SSTR(MMseqsNode::rank);
    pairFile.file = FileUtil::openAndDelete(pairFile.name.c_str(), "w+");
    BatchWriter batchWriter(exchange.writeBehindBytes, par.threads, fileno(pairFile.file));
    pairFile.batchWriter = &batchWriter;
    pairFile.bytes = 0;
    pairFile.ownPairs = &ownPairs;
    pairFile.partitionBytes.assign(exchange.sendSplitCount * exchange.partitionsPerRange, 0);
    char *pairWriteBuffer = new(std::nothrow) char[par.threads * KMER_FILE_BUFFER_BYTES];
    Util::checkAllocation(pairWriteBuffer, "Can not allocate pairWriteBuffer memory in receiveKmers");
    // the k-mer file of every sender, open for the whole phase, then the own rows in memory
    std::vector<ReceivedFile> receivedFiles(MMseqsNode::count + 1);
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
    for (size_t sender = 0; sender < MMseqsNode::count; sender++) {
        receivedFiles[sender].fileName = par.db2 + ".kmers." + SSTR(sender);
        openReceivedFile(receivedFiles[sender], 0);
    }
    receivedFiles[MMseqsNode::count].data = ownKmers.data;
    // once the own hash ranges are received, the pairs of the own merge send splits take the memory of the own k-mers
    const auto releaseOwnKmers = [&]() {
        Util::releasePages(ownKmers.data, ownKmers.bytes);
        free(ownKmers.data);
        ownKmers.data = NULL;
        receivedFiles[MMseqsNode::count].data = NULL;
        ownPairs.limit = ownKmers.bytes;
    };
    size_t receivedFlushCount = 0;
    for (size_t sender = 0; sender < MMseqsNode::count; sender++) {
        receivedFlushCount += receivedFiles[sender].flushCount;
    }

    std::vector<KmerThreadTaskStart> taskStarts;
    // a pass and the prefetched next one fit beside the rows of a flush, the own rows and the rows of this and the next hash range; a bigger hash sub range is a pass read through a refilled buffer
    const size_t receivedRowsMemory = 2 * receivedFlushCount * exchange.rowsPerHashRange * sizeof(KmerFileRow);
    const size_t passLimit = getPassLimit(exchange.keySplitMemoryLimit - std::min(exchange.keySplitMemoryLimit, exchange.pairRowBytes + ownKmers.bytes + receivedRowsMemory));
    // the records and targets of the passes until the output is full, written then and at the end
    const size_t outputBytes = passLimit;
    // a thread that is alone with a group too big for the output takes half of it for records, at least one per representative, and half for targets
    const size_t halfRecords = outputBytes / 2 / sizeof(PairRecord);
    const size_t halfTargets = (outputBytes - halfRecords * sizeof(PairRecord)) / sizeof(DBKeyType);
    if (halfRecords < exchange.maxRepSeqs || halfTargets == 0) {
        Debug(Debug::ERROR) << "Not enough memory to receive the k-mers. Raise --split-memory-limit.\n";
        EXIT(EXIT_FAILURE);
    }
    PairOutput output;
    output.bytes = outputBytes;
    output.regionBytes = std::min(KMER_FILE_BUFFER_BYTES, outputBytes / (OUTPUT_REGIONS_PER_THREAD * par.threads));
    output.recordsTaken = 0;
    output.targetsTaken = 0;
    output.flushes = 0;
    char *outputBuffer = new(std::nothrow) char[outputBytes];
    Util::checkAllocation(outputBuffer, "Can not allocate outputBuffer memory in receiveKmers");
    output.records = reinterpret_cast<PairRecord *>(outputBuffer);
    output.targets = reinterpret_cast<DBKeyType *>(outputBuffer);
    std::vector<PairOutputRegion> outputRegions(par.threads);
    // the groups and their k-mer bytes, and those of the groups with one member that give no pairs
    size_t receivedGroupCount = 0;
    size_t oneMemberGroupCount = 0;
    size_t oneMemberGroupBits = 0;
    size_t repCandidateBytes = 0;
    size_t bufferRefilledPassCount = 0;
    size_t oversizedGroupTaskCount = 0;
    size_t punchedRanges = 0;
    double readWaitSeconds = 0;
    // the buffer of the largest hash range so far, kept for the next ones so their pages are not mapped again
    char *passBuffer = NULL;
    size_t codeBufferBytes = 0;
    BatchReader *batchReader = NULL;
    // the rows of a pass with groups, hash sub range after hash sub range in id order, and where the rows of every hash sub range start, per code buffer
    std::vector<PassRow> passRows[2];
    std::vector<size_t> passRowOffsets[2];

    // a pass is cut into about TASKS_PER_THREAD tasks per thread of about equal bytes, of equal hash width per hash sub range; a buffer refilled pass is one task
    const auto getTaskCount = [&](const ReceivedHashRange &received, size_t rangeHashSubRange, size_t pass) {
        const size_t passReceivedBytes = std::max<size_t>(1, received.passReceivedBytes[pass]);
        return (received.passBytes[pass] > passLimit) ? 1 : std::max<size_t>(1, (TASKS_PER_THREAD * par.threads * received.hashSubRangeBytes[rangeHashSubRange] + passReceivedBytes - 1) / passReceivedBytes);
    };
    size_t receivedBytes = 0;
    const auto planHashRange = [&](ReceivedHashRange &received, size_t hashRange) {
        received.hashRange = hashRange;
        received.fileRows.resize(MMseqsNode::count + 1);
        received.fileRows[MMseqsNode::count] = ownKmers.hashRangeRows[hashRange];
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
        for (size_t sender = 0; sender < MMseqsNode::count; sender++) {
            readItemRows(receivedFiles[sender], hashRange, received.fileRows[sender]);
        }
        received.hashSubRangeReads.assign(exchange.rowsPerHashRange, 0);
        std::vector<size_t> hashSubRangeCodeBytes(exchange.rowsPerHashRange, 0);
        received.hashSubRangeBytes.assign(exchange.rowsPerHashRange, 0);
        for (size_t sender = 0; sender < received.fileRows.size(); sender++) {
            const std::vector<KmerFileRow> &rows = received.fileRows[sender];
            for (size_t row = 0; row < rows.size(); row++) {
                const size_t rangeHashSubRange = row % exchange.rowsPerHashRange;
                received.hashSubRangeReads[rangeHashSubRange] += (rows[row].groupCount > 0);
                received.hashSubRangeBytes[rangeHashSubRange] += getRowEnd(rows[row]) - rows[row].offsets[0];
                repCandidateBytes += rows[row].offsets[KMER_INDEX_CODES] - rows[row].offsets[KMER_REP_CANDIDATE_CODES];
                // a pass reads the rows of the files to its code buffer
                if (receivedFiles[sender].data == NULL) {
                    hashSubRangeCodeBytes[rangeHashSubRange] += getRowEnd(rows[row]) - rows[row].offsets[0];
                }
            }
        }
        const size_t maxPassBytes = planPasses(hashSubRangeCodeBytes, passLimit, received.passHashSubRangeOffsets, received.passBytes);
        const bool hasBufferRefilledPass = (*std::max_element(received.passBytes.begin(), received.passBytes.end()) > passLimit);
        received.codeBufferBytes = hasBufferRefilledPass ? passLimit : maxPassBytes;
        received.passReceivedBytes.assign(received.passBytes.size(), 0);
        // a pass holds a start per row and task
        received.maxTaskStarts = 0;
        for (size_t pass = 0; pass < received.passBytes.size(); pass++) {
            for (size_t rangeHashSubRange = received.passHashSubRangeOffsets[pass]; rangeHashSubRange < received.passHashSubRangeOffsets[pass + 1]; rangeHashSubRange++) {
                received.passReceivedBytes[pass] += received.hashSubRangeBytes[rangeHashSubRange];
                receivedBytes += received.hashSubRangeBytes[rangeHashSubRange];
            }
            size_t passTaskStarts = 0;
            for (size_t rangeHashSubRange = received.passHashSubRangeOffsets[pass]; rangeHashSubRange < received.passHashSubRangeOffsets[pass + 1]; rangeHashSubRange++) {
                passTaskStarts += (getTaskCount(received, rangeHashSubRange, pass) + 1) * received.hashSubRangeReads[rangeHashSubRange];
            }
            received.maxTaskStarts = std::max(received.maxTaskStarts, passTaskStarts);
        }
    };
    const auto readPass = [&](const ReceivedHashRange &received, size_t pass, size_t codeBuffer, bool isAsync) {
        getPassRows(exchange, received.fileRows, received.passHashSubRangeOffsets[pass], received.passHashSubRangeOffsets[pass + 1], passRows[codeBuffer], passRowOffsets[codeBuffer]);
        readPassRows(receivedFiles, received.fileRows, passRows[codeBuffer], passBuffer + codeBuffer * codeBufferBytes, isAsync, batchReader, par.threads);
    };

    // the hash ranges of the own rows first, which this node took while sending
    size_t ownHashRanges = 0;
    const auto takeHashRange = [&]() {
        return (ownHashRanges < ownKmers.hashRanges.size()) ? ownKmers.hashRanges[ownHashRanges++] : hashQueue.takeNextItem();
    };
    ReceivedHashRange receivedHashRanges[2];
    size_t hashRange = takeHashRange();
    if (hashRange < exchange.hashRangeCount) {
        planHashRange(receivedHashRanges[0], hashRange);
    }
    // the code buffer of the next pass that is not buffer refilled, and whether it is read already
    size_t passCodeBuffer = 0;
    bool isPrefetched = false;
    for (size_t taken = 0; hashRange < exchange.hashRangeCount; taken++) {
        hashRanges.push_back(hashRange);
        if (taken == ownKmers.hashRanges.size()) {
            releaseOwnKmers();
        }
        const ReceivedHashRange &received = receivedHashRanges[taken % 2];
        ReceivedHashRange &nextReceived = receivedHashRanges[(taken + 1) % 2];
        const size_t passCount = received.passBytes.size();
        // a hash range that needs a bigger buffer is not prefetched
        if (passBuffer == NULL || received.codeBufferBytes > codeBufferBytes) {
            Util::releasePages(passBuffer, 2 * codeBufferBytes);
            delete[] passBuffer;
            codeBufferBytes = received.codeBufferBytes;
            passBuffer = new(std::nothrow) char[2 * codeBufferBytes];
            Util::checkAllocation(passBuffer, "Can not allocate passBuffer memory in receiveKmers");
        }
        taskStarts.reserve(received.maxTaskStarts);
        size_t nextHashRange = exchange.hashRangeCount;
        for (size_t pass = 0; pass < passCount; pass++) {
            const size_t passFirstHashSubRange = received.passHashSubRangeOffsets[pass];
            const size_t passHashSubRanges = received.passHashSubRangeOffsets[pass + 1] - passFirstHashSubRange;
            // a buffer refilled pass is a single hash sub range that reads its codes through buffers of codeReadBufferBytes
            const bool isBufferRefilledPass = (received.passBytes[pass] > passLimit);
            char *codeBuffer = passBuffer + passCodeBuffer * codeBufferBytes;
            size_t codeReadBufferBytes = 0;
            if (isBufferRefilledPass) {
                bufferRefilledPassCount++;
                codeBuffer = passBuffer;
                codeReadBufferBytes = 2 * passLimit / (received.hashSubRangeReads[passFirstHashSubRange] * KMER_CODE_COUNT);
                if (codeReadBufferBytes == 0) {
                    Debug(Debug::ERROR) << "Not enough memory to receive the k-mers. Raise --split-memory-limit.\n";
                    EXIT(EXIT_FAILURE);
                }
                getPassRows(exchange, received.fileRows, passFirstHashSubRange, received.passHashSubRangeOffsets[pass + 1], passRows[passCodeBuffer], passRowOffsets[passCodeBuffer]);
            } else {
                Timer readTimer;
                if (isPrefetched == false) {
                    readPass(received, pass, passCodeBuffer, false);
                } else {
                    batchReader->waitAsyncRead();
                }
                readWaitSeconds += readTimer.getTimediff();
            }
            // the last pass takes the next hash range, so its first pass is read while this one is merged
            if (pass + 1 == passCount) {
                nextHashRange = takeHashRange();
                if (nextHashRange < exchange.hashRangeCount) {
                    planHashRange(nextReceived, nextHashRange);
                }
            }
            const ReceivedHashRange *prefetched = (pass + 1 < passCount) ? &received : ((nextHashRange < exchange.hashRangeCount) ? &nextReceived : NULL);
            const size_t prefetchedPass = (pass + 1 < passCount) ? pass + 1 : 0;
            isPrefetched = isBufferRefilledPass == false && prefetched != NULL && prefetched->codeBufferBytes <= codeBufferBytes
                           && prefetched->passBytes[prefetchedPass] <= passLimit && prefetched->passBytes[prefetchedPass] > 0;
            if (isPrefetched) {
                readPass(*prefetched, prefetchedPass, 1 - passCodeBuffer, true);
            }
            const std::vector<PassRow> &hashSubRangeRows = passRows[passCodeBuffer];
            const std::vector<size_t> &hashSubRangeRowOffsets = passRowOffsets[passCodeBuffer];
            if (isBufferRefilledPass == false) {
                passCodeBuffer = 1 - passCodeBuffer;
            }
            std::vector<size_t> hashSubRangeTaskOffsets(passHashSubRanges + 1, 0);
            // the task starts of a hash sub range: a start per row for every task and then the end of every row
            std::vector<size_t> taskStartOffsets(passHashSubRanges + 1, 0);
            std::vector<size_t> taskHashSubRanges;
            for (size_t passHashSubRange = 0; passHashSubRange < passHashSubRanges; passHashSubRange++) {
                const size_t taskCount = getTaskCount(received, passFirstHashSubRange + passHashSubRange, pass);
                hashSubRangeTaskOffsets[passHashSubRange + 1] = hashSubRangeTaskOffsets[passHashSubRange] + taskCount;
                taskStartOffsets[passHashSubRange + 1] = taskStartOffsets[passHashSubRange]
                                                          + (taskCount + 1) * (hashSubRangeRowOffsets[passHashSubRange + 1] - hashSubRangeRowOffsets[passHashSubRange]);
                taskHashSubRanges.insert(taskHashSubRanges.end(), taskCount, passHashSubRange);
            }
            const size_t passTaskCount = taskHashSubRanges.size();
            taskStarts.resize(taskStartOffsets[passHashSubRanges]);
            // every row decodes its index codes once to find its first group in every task; member and rep candidate codes have a fixed width per group
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
            for (size_t passRow = 0; passRow < hashSubRangeRows.size(); passRow++) {
                const size_t passHashSubRange = std::upper_bound(hashSubRangeRowOffsets.begin(), hashSubRangeRowOffsets.end(), passRow) - hashSubRangeRowOffsets.begin() - 1;
                const size_t taskCount = hashSubRangeTaskOffsets[passHashSubRange + 1] - hashSubRangeTaskOffsets[passHashSubRange];
                const size_t rowCount = hashSubRangeRowOffsets[passHashSubRange + 1] - hashSubRangeRowOffsets[passHashSubRange];
                KmerThreadTaskStart *rowStarts = taskStarts.data() + taskStartOffsets[passHashSubRange] + passRow - hashSubRangeRowOffsets[passHashSubRange];
                const ReceivedFile &receivedFile = receivedFiles[hashSubRangeRows[passRow].file];
                const KmerFileRow &hashSubRangeRow = received.fileRows[hashSubRangeRows[passRow].file][hashSubRangeRows[passRow].row];
                size_t hashSubRange = 0;
                size_t hashSubRangeEnd = 0;
                const bool isWholeSequence = getRowHashSubRanges(exchange, received.hashRange, passFirstHashSubRange + passHashSubRange, hashSubRange, hashSubRangeEnd);
                const unsigned int hashShift = isWholeSequence ? 0 : exchange.kmerHashShift;
                const KmerPositionBit::uint128 hashStart = getHashSubRangeStart(hashSubRange, exchange.hashSubRangeCount, hashShift);
                const KmerPositionBit::uint128 hashWidth = getHashSubRangeStart(hashSubRangeEnd, exchange.hashSubRangeCount, hashShift) - hashStart;
                KmerThreadTaskStart start = {0, static_cast<uint64_t>(hashStart), {0, 0, 0}};
                rowStarts[0] = start;
                if (taskCount > 1) {
                    KmerCodeReader indexRead = getRowCodeReader(receivedFile, hashSubRangeRow, hashSubRangeRows[passRow].bufferOffset, KMER_INDEX_CODES, codeBuffer, NULL, 0);
                    const unsigned int hashDiffLowBits = getHashDiffLowBits(hashSubRange, hashSubRangeEnd, exchange.hashSubRangeCount, hashShift, hashSubRangeRow.groupCount);
                    size_t task = 1;
                    for (; start.group < hashSubRangeRow.groupCount; start.group++) {
                        start.bitPos[KMER_INDEX_CODES] = getKmerCodeBitPos(indexRead);
                        const uint64_t hash = start.hash + getRice(indexRead, hashDiffLowBits) + ((start.group > 0) ? 1 : 0);
                        for (; task < taskCount && hash >= hashStart + hashWidth * task / taskCount; task++) {
                            rowStarts[task * rowCount] = start;
                        }
                        const size_t memberCount = getEliasGamma(indexRead);
                        const size_t repCandidateCount = isWholeSequence ? 0 : getEliasGamma(indexRead);
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
            // the code bits of every task, so the threads take the largest first
            std::vector<size_t> taskBits(passTaskCount, 0);
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
            for (size_t task = 0; task < passTaskCount; task++) {
                const size_t passHashSubRange = taskHashSubRanges[task];
                const size_t rowCount = hashSubRangeRowOffsets[passHashSubRange + 1] - hashSubRangeRowOffsets[passHashSubRange];
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
            // the output is written when full and stopped tasks go on where they stopped; a task with a group too big for the empty output goes on alone at the end, writing whenever full
            std::vector<std::vector<KmerThreadTaskStart> > taskResumes(passTaskCount);
            std::vector<char> taskStops(passTaskCount, TASK_DONE);
            std::vector<size_t> tooBigTasks;
            for (bool isAlone = false; roundTasks.empty() == false;) {
#pragma omp parallel num_threads(isAlone ? 1 : par.threads)
                {
                    unsigned int thread_idx = 0;
#ifdef OPENMP
                    thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
                    PairOutputRegion &region = outputRegions[thread_idx];
                    size_t threadGroups = 0;
                    size_t threadOneMemberGroups = 0;
                    size_t threadOneMemberGroupBits = 0;
                    std::vector<KmerHashSubRangeRead> reads;
                    std::vector<unsigned int> groupRows;
                    std::vector<KmerCodeReader> groupRepCandidateStarts;
                    std::vector<KmerCodeReader> groupMemberStarts;
                    std::vector<size_t> repCandidatesLeft;
                    // the representatives of a group in choice order as the position of their row in groupRows and their member index there
                    std::vector<std::pair<size_t, size_t> > repMembers;
                    // the rep candidates of a group as their adjKey and their member like in repMembers
                    typedef std::pair<unsigned int, std::pair<size_t, size_t> > GroupRepCandidate;
                    std::vector<GroupRepCandidate> repCandidates;
                    repCandidates.reserve(KMER_FILE_BUFFER_BYTES / sizeof(GroupRepCandidate));
                    std::vector<DBKeyType> repSeqIds;
                    std::vector<std::pair<DBKeyType, DBKeyType> > repCoveredKeys;
                    RowValueQueue adjKeyQueue;
                    // the readers of a code of the group rows go back to the group start
                    const auto rewindGroupReads = [&](size_t code, const std::vector<KmerCodeReader> &groupStarts) {
                        for (size_t row = 0; row < groupRows.size(); row++) {
                            KmerCodeReader &codeRead = reads[groupRows[row]].codeReads[code];
                            if (codeRead.fileOffset != groupStarts[row].fileOffset) {
                                refillKmerCodeBuffer(codeRead, groupStarts[row].fileOffset);
                            }
                            codeRead = groupStarts[row];
                        }
                    };
#pragma omp for schedule(dynamic, 1)
                    for (size_t i = 0; i < roundTasks.size(); i++) {
                        const size_t task = roundTasks[i];
                        taskStops[task] = TASK_DONE;
                        const size_t passHashSubRange = taskHashSubRanges[task];
                        size_t hashSubRange = 0;
                        size_t hashSubRangeEnd = 0;
                        const bool isWholeSequence = getRowHashSubRanges(exchange, received.hashRange, passFirstHashSubRange + passHashSubRange, hashSubRange, hashSubRangeEnd);
                        const unsigned int hashShift = isWholeSequence ? 0 : exchange.kmerHashShift;
                        const size_t groupMaxRepSeqs = isWholeSequence ? 1 : exchange.maxRepSeqs;
                        const size_t rowCount = hashSubRangeRowOffsets[passHashSubRange + 1] - hashSubRangeRowOffsets[passHashSubRange];
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
                                read.codeReads[code] = getRowCodeReader(receivedFiles[passRow.file], hashSubRangeRow, passRow.bufferOffset, code, codeBuffer,
                                                                        codeBuffer + (reads.size() * KMER_CODE_COUNT + code) * codeReadBufferBytes, codeReadBufferBytes);
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

                        // merges the groups of the rows by hash
                        RowValueQueue kmerHashQueue;
                        for (size_t row = 0; row < reads.size(); row++) {
                            queueNextKmerGroup(kmerHashQueue, row, reads[row]);
                        }
                        while (kmerHashQueue.empty() == false) {
                            const uint64_t hash = kmerHashQueue.top().value;
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
                                const size_t memberBits = read.groupMemberCount * read.hashSubRangeRow->idBits;
                                const size_t repCandidateBits = read.groupRepCandidateCount * (exchange.adjKeyCodeBits + getBitWidth(read.groupMemberCount - 1));
                                threadGroups++;
                                threadOneMemberGroups++;
                                threadOneMemberGroupBits += getKmerCodeBitPos(read.codeReads[KMER_INDEX_CODES]) - read.pendingIndexBitPos + memberBits + repCandidateBits;
                                seekKmerCodeBits(memberRead, getKmerCodeBitPos(memberRead) + memberBits);
                                seekKmerCodeBits(repCandidateRead, getKmerCodeBitPos(repCandidateRead) + repCandidateBits);
                            } else {
                                // the records and targets of the group at most: a record per representative and every member once
                                if (takePairOutput(output, region, groupMaxRepSeqs, memberCount) == false) {
                                    if (isAlone == false) {
                                        taskStops[task] = (groupMaxRepSeqs * sizeof(PairRecord) + memberCount * sizeof(DBKeyType) > output.bytes) ? TASK_GROUP_TOO_BIG : TASK_OUTPUT_FULL;
                                        break;
                                    }
                                    flushPairOutput(exchange, pairFile, output, outputRegions, pairWriteBuffer, 1);
                                    takePairOutput(output, region, std::min(groupMaxRepSeqs, halfRecords), std::min(memberCount, halfTargets));
                                }
                                threadGroups++;
                                groupRepCandidateStarts.clear();
                                groupMemberStarts.clear();
                                for (size_t row = 0; row < groupRows.size(); row++) {
                                    groupRepCandidateStarts.push_back(reads[groupRows[row]].codeReads[KMER_REP_CANDIDATE_CODES]);
                                    groupMemberStarts.push_back(reads[groupRows[row]].codeReads[KMER_MEMBER_CODES]);
                                }
                                // a rep candidate is the first member of its adjKey in a row, (row, member index) order is id order
                                repMembers.clear();
                                if (isWholeSequence) {
                                    repMembers.push_back(std::make_pair(0, 0));
                                } else {
                                    size_t groupRepCandidateCount = 0;
                                    for (size_t row = 0; row < groupRows.size(); row++) {
                                        groupRepCandidateCount += reads[groupRows[row]].groupRepCandidateCount;
                                    }
                                    // rep candidates up to a thread write buffer, the second one of fixedMemory that receiving does not allocate, are read once and sorted in the order of the row merge below
                                    if (groupRepCandidateCount <= KMER_FILE_BUFFER_BYTES / sizeof(GroupRepCandidate)) {
                                        repCandidates.clear();
                                        rewindGroupReads(KMER_REP_CANDIDATE_CODES, groupRepCandidateStarts);
                                        for (size_t row = 0; row < groupRows.size(); row++) {
                                            KmerHashSubRangeRead &read = reads[groupRows[row]];
                                            const unsigned int memberIndexBits = getBitWidth(read.groupMemberCount - 1);
                                            for (size_t repCandidate = 0; repCandidate < read.groupRepCandidateCount; repCandidate++) {
                                                const unsigned int adjKey = getAdjKeyCode(read.codeReads[KMER_REP_CANDIDATE_CODES], exchange);
                                                repCandidates.push_back(GroupRepCandidate(adjKey, std::pair<size_t, size_t>(row, getBits(read.codeReads[KMER_REP_CANDIDATE_CODES], memberIndexBits))));
                                            }
                                        }
                                        SORT_SERIAL(repCandidates.begin(), repCandidates.end());
                                        selectGroupRepSeqs(repMembers, groupMaxRepSeqs, [&](const auto &visit) {
                                            for (size_t repCandidate = 0; repCandidate < repCandidates.size(); repCandidate++) {
                                                visit(repCandidates[repCandidate].first, repCandidates[repCandidate].second);
                                            }
                                        }, std::less<std::pair<size_t, size_t> >());
                                    } else {
                                        selectGroupRepSeqs(repMembers, groupMaxRepSeqs, [&](const auto &visit) {
                                            rewindGroupReads(KMER_REP_CANDIDATE_CODES, groupRepCandidateStarts);
                                            if (groupRows.size() == 1) {
                                                KmerHashSubRangeRead &read = reads[groupRows[0]];
                                                const unsigned int memberIndexBits = getBitWidth(read.groupMemberCount - 1);
                                                for (size_t repCandidate = 0; repCandidate < read.groupRepCandidateCount; repCandidate++) {
                                                    const unsigned int adjKey = getAdjKeyCode(read.codeReads[KMER_REP_CANDIDATE_CODES], exchange);
                                                    visit(adjKey, std::pair<size_t, size_t>(0, getBits(read.codeReads[KMER_REP_CANDIDATE_CODES], memberIndexBits)));
                                                }
                                            } else {
                                                // the rows merged by adjKey, at the same adjKey in row order
                                                repCandidatesLeft.resize(groupRows.size());
                                                for (size_t row = 0; row < groupRows.size(); row++) {
                                                    KmerHashSubRangeRead &read = reads[groupRows[row]];
                                                    repCandidatesLeft[row] = read.groupRepCandidateCount - 1;
                                                    adjKeyQueue.push(RowValue(getAdjKeyCode(read.codeReads[KMER_REP_CANDIDATE_CODES], exchange), row));
                                                }
                                                while (adjKeyQueue.empty() == false) {
                                                    const RowValue repCandidate = adjKeyQueue.top();
                                                    adjKeyQueue.pop();
                                                    KmerHashSubRangeRead &read = reads[groupRows[repCandidate.row]];
                                                    const size_t memberIndex = getBits(read.codeReads[KMER_REP_CANDIDATE_CODES], getBitWidth(read.groupMemberCount - 1));
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
                                // the id of a member index is at its position in the fixed width member codes
                                repSeqIds.clear();
                                for (size_t rep = 0; rep < repMembers.size(); rep++) {
                                    KmerHashSubRangeRead &read = reads[groupRows[repMembers[rep].first]];
                                    KmerCodeReader &memberRead = read.codeReads[KMER_MEMBER_CODES];
                                    seekKmerCodeBits(memberRead, getKmerCodeBitPos(groupMemberStarts[repMembers[rep].first]) + repMembers[rep].second * read.hashSubRangeRow->idBits);
                                    repSeqIds.push_back(getBits(memberRead, read.hashSubRangeRow->idBits) + read.hashSubRangeRow->splitFirstKey);
                                }
                                // the keys [first, second) of the members with a length that a representative can cover
                                repCoveredKeys.clear();
                                for (size_t rep = 0; rep < repSeqIds.size(); rep++) {
                                    const size_t repRange = seqDbr.findIdRange(repSeqIds[rep]);
                                    const float repLength = static_cast<float>(index[repRange].length - 2);
                                    const auto isCovered = [&](const DBReader<DBKeyType>::Index &range) {
                                        return Util::canBeCovered(par.covThr, par.covMode, repLength, static_cast<float>(range.length - 2));
                                    };
                                    // ranges are in length descending order and every --cov-mode covers an interval of lengths that holds the representative length
                                    const DBReader<DBKeyType>::Index *coveredFirstRange = std::partition_point(index, index + repRange,
                                                                                                               [&](const DBReader<DBKeyType>::Index &range) { return isCovered(range) == false; });
                                    const DBReader<DBKeyType>::Index *coveredEndRange = std::partition_point(index + repRange, index + seqDbr.getIndexSize(), isCovered);
                                    const DBKeyType coveredTo = (coveredEndRange < index + seqDbr.getIndexSize()) ? coveredEndRange->id : static_cast<DBKeyType>(seqDbr.getSize());
                                    repCoveredKeys.push_back(std::make_pair(coveredFirstRange->id, coveredTo));
                                }
                                // the members go to the output once, in id order as the rows are key splits in key order; a record per representative is the part of the members it covers, with itself that writePairFileRows leaves out
                                rewindGroupReads(KMER_MEMBER_CODES, groupMemberStarts);
                                for (size_t row = 0, member = 0; row < groupRows.size();) {
                                    // a group that outgrows the output of a thread that is alone continues in the next flush
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
                                        if (coveredEnd - covered > (std::binary_search(covered, coveredEnd, repSeqIds[rep]) ? 1 : 0)) {
                                            const PairRecord record = {repSeqIds[rep], static_cast<size_t>(covered - output.targets), static_cast<size_t>(coveredEnd - covered)};
                                            output.records[region.recordPos++] = record;
                                        }
                                    }
                                }
                            }
                            for (size_t row = 0; row < groupRows.size(); row++) {
                                queueNextKmerGroup(kmerHashQueue, groupRows[row], reads[groupRows[row]]);
                            }
                        }
                        // a stopped task goes on at the group that it stopped at, where its rows have read only the index codes of their next group
                        if (taskStops[task] != TASK_DONE) {
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
#pragma omp atomic
                    receivedGroupCount += threadGroups;
#pragma omp atomic
                    oneMemberGroupCount += threadOneMemberGroups;
#pragma omp atomic
                    oneMemberGroupBits += threadOneMemberGroupBits;
                }
                std::vector<size_t> fullTasks;
                for (size_t i = 0; i < roundTasks.size(); i++) {
                    if (taskStops[roundTasks[i]] == TASK_OUTPUT_FULL) {
                        fullTasks.push_back(roundTasks[i]);
                    } else if (taskStops[roundTasks[i]] == TASK_GROUP_TOO_BIG) {
                        tooBigTasks.push_back(roundTasks[i]);
                    }
                }
                // the output is full, or the tasks that go on alone start with an empty output
                isAlone = fullTasks.empty();
                if (isAlone == false || tooBigTasks.empty() == false) {
                    flushPairOutput(exchange, pairFile, output, outputRegions, pairWriteBuffer, par.threads);
                }
                oversizedGroupTaskCount += isAlone ? tooBigTasks.size() : 0;
                roundTasks.swap(isAlone ? tooBigTasks : fullTasks);
                if (isAlone) {
                    tooBigTasks.clear();
                }
            }
        }
        // the receiving node of the hash range is the only reader of its codes; the pairs of the own merge send splits take the memory of the own rows
        std::vector<PassRow> hashRangeRows;
        std::vector<size_t> hashRangeRowOffsets;
        getPassRows(exchange, received.fileRows, 0, exchange.rowsPerHashRange, hashRangeRows, hashRangeRowOffsets);
        punchedRanges += punchPassRows(receivedFiles, received.fileRows, hashRangeRows, ownPairs.limit);
        hashRange = nextHashRange;
    }
    releaseOwnKmers();
    flushPairOutput(exchange, pairFile, output, outputRegions, pairWriteBuffer, par.threads);
    delete batchReader;
    closeReceivedFiles(receivedFiles);
    batchWriter.waitWrites();
    MMseqsNode::addPhaseSeconds(receiveValues, "waiting to read k-mers", readWaitSeconds);
    MMseqsNode::addPhaseSeconds(receiveValues, "waiting to write pairs", batchWriter.getWaitSeconds() / par.threads);
    MMseqsNode::addPhaseValue(receiveValues, "Pair flushes", SSTR(output.flushes));
    MMseqsNode::addPhaseValue(receiveValues, "Pair bytes kept in memory", SSTR(ownPairs.bytes));
    MMseqsNode::addPhaseValue(receiveValues, "Passes read through a refilled buffer", SSTR(bufferRefilledPassCount));
    MMseqsNode::addPhaseValue(receiveValues, "Thread tasks with a group too big for the output", SSTR(oversizedGroupTaskCount));
    MMseqsNode::addPhaseValue(receiveValues, "K-mer file ranges freed", SSTR(punchedRanges));
    MMseqsNode::addPhaseValue(receiveValues, "K-mer groups", SSTR(receivedGroupCount));
    MMseqsNode::addPhaseValue(receiveValues, "K-mer groups with one member", SSTR(oneMemberGroupCount));
    MMseqsNode::addPhaseValue(receiveValues, "K-mer bytes of groups with one member", SSTR((oneMemberGroupBits + CHAR_BIT - 1) / CHAR_BIT));
    MMseqsNode::addPhaseValue(receiveValues, "K-mer representative candidate bytes", SSTR(repCandidateBytes));
    Util::releasePages(outputBuffer, outputBytes);
    delete[] outputBuffer;
    Util::releasePages(passBuffer, 2 * codeBufferBytes);
    delete[] passBuffer;
    delete[] pairWriteBuffer;
    writeRowTable(pairFile.file, pairFile.name, exchange.sendSplitCount, pairFile.flushRowOffsets, pairFile.partitionBytes, pairFile.bytes);
    pairBytes = pairFile.bytes;
    return receivedBytes;
}

// linfinity rep node: merges the own merge send splits, then takes merge splits one at a time and merges their partitions from every pair file and ownPairs into <pref>.<node>; returns its partition rows
static std::string mergePairFiles(Parameters &par, DBReader<DBKeyType> &seqDbr, const KmerExchange &exchange, MMseqsNode::WorkQueue &mergeQueue, std::vector<size_t> &mergeSplits,
                                  std::string &mergeValues, OwnPairs &ownPairs) {
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    PrefFiles prefFiles;
    prefFiles.dataName = par.db2 + "." + SSTR(MMseqsNode::rank);
    prefFiles.indexName = par.db2 + ".index." + SSTR(MMseqsNode::rank);
    prefFiles.dataFile = FileUtil::openAndDelete(prefFiles.dataName.c_str(), "w");
    prefFiles.indexFile = FileUtil::openAndDelete(prefFiles.indexName.c_str(), "w");
    BatchWriter batchWriter(exchange.writeBehindBytes, par.threads, fileno(prefFiles.dataFile));
    prefFiles.batchWriter = &batchWriter;
    size_t prefDataBytes = 0;
    size_t prefIndexBytes = 0;
    prefFiles.writeBuffers = new(std::nothrow) char[par.threads * 2 * KMER_FILE_BUFFER_BYTES];
    Util::checkAllocation(prefFiles.writeBuffers, "Can not allocate prefWriteBuffer memory in mergePairFiles");
    // a partition is its start, the starts and cuts of its key tasks and its end, with the pref bytes before them in the pref file and the alignment cells before them in the partition
    std::vector<PartitionCut> partitionCuts;
    std::vector<size_t> partitionCutOffsets(1, 0);
    KmerPositionBit::uint128 mergedCost = 0;
    const auto addKeyTaskCuts = [&](size_t partitionFirstKey, size_t taskFirstKey, size_t taskPrefOffset, size_t taskCost, const KeyTaskCuts &taskCuts) {
        const PartitionCut taskStart = { partitionFirstKey + taskFirstKey, taskPrefOffset, taskCost };
        partitionCuts.push_back(taskStart);
        for (size_t i = 0; i < taskCuts.cuts.size(); i++) {
            const PartitionCut cut = { partitionFirstKey + taskCuts.cuts[i].key, taskPrefOffset + taskCuts.cuts[i].prefBytes, taskCost + taskCuts.cuts[i].cost };
            partitionCuts.push_back(cut);
        }
    };
    const auto addPartitionEnd = [&](size_t prefEnd, size_t cost) {
        const PartitionCut end = { 0, prefEnd, cost };
        partitionCuts.push_back(end);
        partitionCutOffsets.push_back(partitionCuts.size());
        mergedCost += cost;
    };
    // the buffer of the largest send split so far, kept for the next ones so their pages are not mapped again
    char *pairPassBuffer = NULL;
    size_t pairPassBufferBytes = 0;
    // the targets of the partitions of a pass, one after the other, kept like pairPassBuffer
    DBKeyType *pairPassTargets = NULL;
    size_t pairPassTargetsSize = 0;
    // the pair file of every receiving node, open for the whole phase, then the own pairs in memory, and their rows of a send split
    std::vector<ReceivedFile> receivedPairFiles(MMseqsNode::count + 1);
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
    for (size_t receiver = 0; receiver < MMseqsNode::count; receiver++) {
        receivedPairFiles[receiver].fileName = par.db2 + ".pairs." + SSTR(receiver);
        openReceivedFile(receivedPairFiles[receiver], exchange.sendSplitCount * exchange.partitionsPerRange);
    }
    receivedPairFiles[MMseqsNode::count].data = ownPairs.data;
    std::vector<std::vector<PairFileRow> > pairFileRows(MMseqsNode::count + 1);
    // the partitions of all but the own merge send splits are cut into as many merge splits as send splits, key ranges of about equal cost like decomposeDomainByAminoAcid; a partition costs its pair bytes plus one per key
    const size_t partitionCount = exchange.sendSplitCount * exchange.partitionsPerRange;
    std::vector<size_t> partitionPairBytes(partitionCount, 0);
    std::vector<size_t> partitionCosts(partitionCount, 0);
    size_t globalCost = 0;
    for (size_t partition = 0; partition < partitionCount; partition++) {
        for (size_t receiver = 0; receiver < MMseqsNode::count; receiver++) {
            partitionPairBytes[partition] += receivedPairFiles[receiver].itemBytes[partition];
        }
        const size_t sendSplit = partition / exchange.partitionsPerRange;
        const size_t rangePartition = partition % exchange.partitionsPerRange;
        if (sendSplit / MMseqsNode::count >= exchange.selfMergedSendSplitCount) {
            partitionCosts[partition] = partitionPairBytes[partition] + getPartitionFirstKey(exchange, sendSplit, rangePartition + 1) - getPartitionFirstKey(exchange, sendSplit, rangePartition);
        }
        globalCost += partitionCosts[partition];
    }
    for (size_t receiver = 0; receiver < receivedPairFiles.size(); receiver++) {
        std::vector<size_t>().swap(receivedPairFiles[receiver].itemBytes);
    }
    const size_t mergeSplitCount = exchange.sendSplitCount;
    const size_t mergeSplitCost = std::max<size_t>(1, (globalCost + mergeSplitCount - 1) / mergeSplitCount);
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
    // the nodes take the costliest splits first, so the ones taken last are small
    SORT_SERIAL(mergeOrder.begin(), mergeOrder.end(), [&](size_t first, size_t second) {
        return splitCosts[first] > splitCosts[second] || (splitCosts[first] == splitCosts[second] && first < second);
    });
    // own items are rank + j * count, so every other group of count items is reversed and no node takes the largest of every group
    for (size_t group = MMseqsNode::count; group < mergeOrder.size(); group += 2 * MMseqsNode::count) {
        std::reverse(mergeOrder.begin() + group, mergeOrder.begin() + std::min(group + MMseqsNode::count, mergeOrder.size()));
    }
    MMseqsNode::addPhaseValue(mergeValues, "Largest merge split cost to average", SSTR(static_cast<double>(splitCosts[mergeOrder[0]]) * mergeSplitCount / std::max<size_t>(1, globalCost)));
    size_t mergedPairBytes = 0;
    size_t punchedRanges = 0;
    double readWaitSeconds = 0;
    size_t releasedOwnPairBytes = 0;
    // the own merge send splits first, then the merge splits that this node takes without the own merge send splits of every node
    for (size_t ownMergeRange = 0;; ownMergeRange++) {
        size_t splitBegin = 0;
        size_t splitEnd = 0;
        if (ownMergeRange < exchange.selfMergedSendSplitCount) {
            splitBegin = (MMseqsNode::rank + ownMergeRange * MMseqsNode::count) * exchange.partitionsPerRange;
            splitEnd = splitBegin + exchange.partitionsPerRange;
        } else {
            // the own pairs are merged
            if (ownMergeRange == exchange.selfMergedSendSplitCount) {
                Util::releasePages(ownPairs.data, ownPairs.bytes);
                free(ownPairs.data);
                ownPairs.data = NULL;
                ownPairs.bytes = 0;
                receivedPairFiles[MMseqsNode::count].data = NULL;
            }
            const size_t item = mergeQueue.takeNextItem();
            if (item == mergeSplitCount) {
                break;
            }
            const size_t mergeSplit = mergeOrder[item];
            mergeSplits.push_back(mergeSplit);
            splitBegin = splitPartitionOffsets[mergeSplit];
            splitEnd = splitPartitionOffsets[mergeSplit + 1];
        }
        for (size_t sendSplit = splitBegin / exchange.partitionsPerRange; splitBegin < splitEnd && sendSplit * exchange.partitionsPerRange < splitEnd; sendSplit++) {
            if (ownMergeRange >= exchange.selfMergedSendSplitCount && sendSplit / MMseqsNode::count < exchange.selfMergedSendSplitCount) {
                continue;
            }
            // the partitions [firstPartition, endPartition) of the send split that the split holds
            const size_t firstPartition = std::max(splitBegin, sendSplit * exchange.partitionsPerRange) - sendSplit * exchange.partitionsPerRange;
            const size_t endPartition = std::min(splitEnd, (sendSplit + 1) * exchange.partitionsPerRange) - sendSplit * exchange.partitionsPerRange;
            for (size_t partition = firstPartition; partition < endPartition; partition++) {
                mergedPairBytes += partitionPairBytes[sendSplit * exchange.partitionsPerRange + partition];
            }
            setSendSplit(par, exchange, sendSplit);
            // the rows of the send split lie in every flush of a pair file, so all of them are read in one batch
            std::vector<std::vector<BatchReader::ReadRequest> > rowReadRequests(MMseqsNode::count);
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
            for (size_t receiver = 0; receiver < MMseqsNode::count; receiver++) {
                readItemRows(receivedPairFiles[receiver], sendSplit, pairFileRows[receiver], &rowReadRequests[receiver]);
            }
            pairFileRows[MMseqsNode::count] = ownPairs.sendSplitRows[sendSplit];
            for (size_t receiver = 1; receiver < rowReadRequests.size(); receiver++) {
                rowReadRequests[0].insert(rowReadRequests[0].end(), rowReadRequests[receiver].begin(), rowReadRequests[receiver].end());
            }
            BatchReader *batchReader = new BatchReader(par.threads, fileno(receivedPairFiles[0].file));
            Timer readTimer;
            batchReader->readSync(rowReadRequests[0]);
            readWaitSeconds += readTimer.getTimediff();
            std::vector<size_t> partitionCodeBytes(exchange.partitionsPerRange, 0);
            std::vector<size_t> partitionTargets(exchange.partitionsPerRange, 0);
            // the received file and row of every row of a partition
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
                    const size_t partition = rows[row].partition;
                    if (partition < firstPartition || partition >= endPartition) {
                        continue;
                    }
                    // a pass reads the rows of the files to its code buffer
                    if (receivedPairFiles[receiver].data == NULL) {
                        partitionCodeBytes[partition] += getRowEnd(rows[row]) - rows[row].offsets[0];
                    }
                    partitionTargets[partition] += rows[row].targetCount;
                    partitionRows[partition].push_back(std::make_pair(receiver, row));
                }
            }
            const size_t pairPassLimit = getPassLimit(exchange.keySplitMemoryLimit - ownPairs.bytes);
            // a partition takes its codes, its target ids, where the targets of every key start and its merge with a thread
            std::vector<size_t> partitionBytes(endPartition - firstPartition);
            for (size_t partition = firstPartition; partition < endPartition; partition++) {
                const size_t partitionKeys = getPartitionFirstKey(exchange, sendSplit, partition + 1) - getPartitionFirstKey(exchange, sendSplit, partition);
                partitionBytes[partition - firstPartition] = partitionCodeBytes[partition] + partitionTargets[partition] * sizeof(DBKeyType) + (partitionKeys + 1) * sizeof(size_t)
                                                             + getPartitionMergeBytes(partitionRowCounts[partition], 1);
            }
            // a pass and the prefetched codes of the next one fit their limit; a bigger partition is a pass that reads its codes through refilled buffers
            std::vector<size_t> passPartitionOffsets;
            std::vector<size_t> pairPassBytes;
            planPasses(partitionBytes, pairPassLimit, passPartitionOffsets, pairPassBytes);
            for (size_t pass = 0; pass < passPartitionOffsets.size(); pass++) {
                passPartitionOffsets[pass] += firstPartition;
            }
            const size_t pairPassCount = pairPassBytes.size();
            std::vector<size_t> pairPassCodeBytes(pairPassCount, 0);
            size_t maxPairPassCodeBytes = 0;
            size_t maxPairPassTargets = 0;
            bool hasBufferRefilledPairPass = false;
            for (size_t pass = 0; pass < pairPassCount; pass++) {
                size_t pairPassTargetCount = 0;
                for (size_t partition = passPartitionOffsets[pass]; partition < passPartitionOffsets[pass + 1]; partition++) {
                    pairPassCodeBytes[pass] += partitionCodeBytes[partition];
                    pairPassTargetCount += partitionTargets[partition];
                }
                if (pairPassBytes[pass] <= pairPassLimit) {
                    maxPairPassCodeBytes = std::max(maxPairPassCodeBytes, pairPassCodeBytes[pass]);
                    maxPairPassTargets = std::max(maxPairPassTargets, pairPassTargetCount);
                } else {
                    hasBufferRefilledPairPass = true;
                }
            }

            // two code buffers for a pass and the prefetched next one, or the buffers of a buffer refilled pass
            const size_t neededPairPassBufferBytes = std::max(2 * maxPairPassCodeBytes, hasBufferRefilledPairPass ? 2 * pairPassLimit : 0);
            if (neededPairPassBufferBytes > pairPassBufferBytes) {
                Util::releasePages(pairPassBuffer, pairPassBufferBytes);
                delete[] pairPassBuffer;
                pairPassBufferBytes = neededPairPassBufferBytes;
                pairPassBuffer = new(std::nothrow) char[pairPassBufferBytes];
                Util::checkAllocation(pairPassBuffer, "Can not allocate pairPassBuffer memory in mergePairFiles");
            }
            // the threads that place the targets touch their pages first
            if (maxPairPassTargets > pairPassTargetsSize) {
                Util::releasePages(pairPassTargets, pairPassTargetsSize * sizeof(DBKeyType));
                delete[] pairPassTargets;
                pairPassTargetsSize = maxPairPassTargets;
                pairPassTargets = new(std::nothrow) DBKeyType[pairPassTargetsSize];
                Util::checkAllocation(pairPassTargets, "Can not allocate pairPassTargets memory in mergePairFiles");
            }
            std::vector<PassRow> pairPassRows[2];
            std::vector<size_t> pairPassRowOffsets[2];
            const auto readPairPass = [&](size_t pass, bool isAsync) {
                getPairPassRows(partitionRows, passPartitionOffsets[pass], passPartitionOffsets[pass + 1], pairPassRows[pass % 2], pairPassRowOffsets[pass % 2]);
                readPassRows(receivedPairFiles, pairFileRows, pairPassRows[pass % 2], pairPassBuffer + (pass % 2) * maxPairPassCodeBytes, isAsync, batchReader, par.threads);
            };
            std::vector<PartitionMerge> partitionMerges;
            bool isPrefetched = false;
            for (size_t pass = 0; pass < pairPassCount; pass++) {
                const size_t passFirstPartition = passPartitionOffsets[pass];
                const size_t passPartitions = passPartitionOffsets[pass + 1] - passFirstPartition;
                // a buffer refilled pass is a single partition that reads its codes through refilled buffers and merges its rows by key straight to the pref files
                const bool isBufferRefilledPass = (pairPassBytes[pass] > pairPassLimit);
                char *codeBuffer = pairPassBuffer + (pass % 2) * maxPairPassCodeBytes;
                if (isBufferRefilledPass) {
                    codeBuffer = pairPassBuffer;
                    getPairPassRows(partitionRows, passFirstPartition, passPartitionOffsets[pass + 1], pairPassRows[pass % 2], pairPassRowOffsets[pass % 2]);
                } else {
                    readTimer.reset();
                    if (isPrefetched == false) {
                        readPairPass(pass, false);
                    } else {
                        batchReader->waitAsyncRead();
                    }
                    readWaitSeconds += readTimer.getTimediff();
                    isPrefetched = pass + 1 < pairPassCount && pairPassBytes[pass + 1] <= pairPassLimit && pairPassCodeBytes[pass + 1] > 0;
                    if (isPrefetched) {
                        readPairPass(pass + 1, true);
                    }
                }
                const std::vector<PassRow> &partitionPassRows = pairPassRows[pass % 2];
                const std::vector<size_t> &partitionPassRowOffsets = pairPassRowOffsets[pass % 2];
                if (passPartitions >= 2 * static_cast<size_t>(par.threads)) {
                    // a pass of many partitions: a thread merges a partition, the largest first, and then writes it after the partitions before it
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
                        const size_t passPartition = partitionOrder[i];
                        const size_t partition = passFirstPartition + passPartition;
                        const size_t partitionFirstKey = getPartitionFirstKey(exchange, sendSplit, partition);
                        countPartitionTargets(exchange, receivedPairFiles, pairFileRows, partitionPassRows.data() + partitionPassRowOffsets[passPartition],
                                              partitionPassRowOffsets[passPartition + 1] - partitionPassRowOffsets[passPartition], codeBuffer,
                                              getPartitionFirstKey(exchange, sendSplit, partition + 1) - partitionFirstKey, partitionTargets[partition], partitionMerges[passPartition], 1);
                        mergePartitionTargets(exchange, seqDbr, partitionFirstKey, partitionMerges[passPartition], pairPassTargets + partitionTargetOffsets[passPartition], 1);
                    }
                    std::vector<size_t> partitionDataOffsets(passPartitions + 1, prefDataBytes);
                    std::vector<size_t> partitionIndexOffsets(passPartitions + 1, prefIndexBytes);
                    for (size_t passPartition = 0; passPartition < passPartitions; passPartition++) {
                        const PartitionMerge &merge = partitionMerges[passPartition];
                        partitionDataOffsets[passPartition + 1] = partitionDataOffsets[passPartition] + merge.keyTaskDataOffsets.back();
                        partitionIndexOffsets[passPartition + 1] = partitionIndexOffsets[passPartition] + merge.keyTaskEntryOffsets.back() * 2 * sizeof(uint64_t);
                        for (size_t keyTask = 0; keyTask + 1 < merge.keyTaskOffsets.size(); keyTask++) {
                            addKeyTaskCuts(getPartitionFirstKey(exchange, sendSplit, passFirstPartition + passPartition), merge.keyTaskOffsets[keyTask],
                                           partitionDataOffsets[passPartition] + merge.keyTaskDataOffsets[keyTask], merge.keyTaskCostOffsets[keyTask], merge.keyTaskCuts[keyTask]);
                        }
                        addPartitionEnd(partitionDataOffsets[passPartition + 1], merge.cost);
                    }
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
                    for (size_t i = 0; i < passPartitions; i++) {
                        unsigned int thread_idx = 0;
#ifdef OPENMP
                        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
                        const size_t passPartition = partitionOrder[i];
                        writePartition(prefFiles, getPartitionFirstKey(exchange, sendSplit, passFirstPartition + passPartition), partitionMerges[passPartition],
                                       partitionDataOffsets[passPartition], partitionIndexOffsets[passPartition], thread_idx, 1);
                        partitionMerges[passPartition] = PartitionMerge();
                    }
                    prefDataBytes = partitionDataOffsets[passPartitions];
                    prefIndexBytes = partitionIndexOffsets[passPartitions];
                } else {
                    // a pass of few partitions: the partitions one after the other in one PartitionMerge that keeps its buffers, all threads merge a partition
                    for (size_t passPartition = 0; passPartition < passPartitions; passPartition++) {
                        const size_t partition = passFirstPartition + passPartition;
                        const size_t partitionFirstKey = getPartitionFirstKey(exchange, sendSplit, partition);
                        const size_t partitionKeys = getPartitionFirstKey(exchange, sendSplit, partition + 1) - partitionFirstKey;
                        const size_t partitionRowCount = partitionPassRowOffsets[passPartition + 1] - partitionPassRowOffsets[passPartition];
                        const size_t partitionDataOffset = prefDataBytes;
                        size_t partitionEntryCount = 0;
                        size_t partitionCost = 0;
                        if (isBufferRefilledPass == false) {
                            // the pass fits with the merge of the partition, fewer threads if their reader copies do not fit
                            int mergeThreads = par.threads;
                            while (mergeThreads > 1 && pairPassBytes[pass] - getPartitionMergeBytes(partitionRowCount, 1) + getPartitionMergeBytes(partitionRowCount, mergeThreads) > pairPassLimit) {
                                mergeThreads--;
                            }
                            partitionMerges.resize(1);
                            PartitionMerge &merge = partitionMerges[0];
                            countPartitionTargets(exchange, receivedPairFiles, pairFileRows, partitionPassRows.data() + partitionPassRowOffsets[passPartition], partitionRowCount, codeBuffer,
                                                  partitionKeys, partitionTargets[partition], merge, mergeThreads);
                            mergePartitionTargets(exchange, seqDbr, partitionFirstKey, merge, pairPassTargets, mergeThreads);
                            writePartition(prefFiles, partitionFirstKey, merge, prefDataBytes, prefIndexBytes, 0, par.threads);
                            for (size_t keyTask = 0; keyTask + 1 < merge.keyTaskOffsets.size(); keyTask++) {
                                addKeyTaskCuts(partitionFirstKey, merge.keyTaskOffsets[keyTask], prefDataBytes + merge.keyTaskDataOffsets[keyTask], merge.keyTaskCostOffsets[keyTask],
                                               merge.keyTaskCuts[keyTask]);
                            }
                            partitionEntryCount = merge.keyTaskEntryOffsets.back();
                            partitionCost = merge.cost;
                            prefDataBytes += merge.keyTaskDataOffsets.back();
                        } else {
                            const size_t codeReadBufferBytes = 2 * pairPassLimit / (std::max<size_t>(1, partitionRowCount) * PAIR_CODE_COUNT);
                            if (codeReadBufferBytes == 0) {
                                Debug(Debug::ERROR) << "Not enough memory to merge the pairs. Raise --split-memory-limit.\n";
                                EXIT(EXIT_FAILURE);
                            }
                            std::vector<PairRowRead> reads;
                            getPartitionRowReads(receivedPairFiles, pairFileRows, partitionPassRows.data() + partitionPassRowOffsets[passPartition], partitionRowCount, partitionKeys, codeBuffer,
                                                 codeReadBufferBytes, reads);
                            RowValueQueue repKeyQueue;
                            RowValueQueue targetQueue;
                            std::vector<unsigned int> keyRows;
                            for (size_t row = 0; row < reads.size(); row++) {
                                readPairRecord(reads[row], exchange.globalIdBits);
                                repKeyQueue.push(RowValue(reads[row].repKey, row));
                            }
                            // key tasks like those of a merge with all threads, so the partition gets as many places to cut
                            const size_t refilledPassTaskKeys = std::max<size_t>(1, partitionKeys / (TASKS_PER_THREAD * par.threads));
                            KeyTaskCuts refilledPassCuts;
                            size_t taskFirstKey = 0;
                            size_t taskPrefOffset = partitionDataOffset;
                            size_t taskCost = 0;
                            KmerCodeWriter refilledPassDataWrite = createFileCodeWriter(fileno(prefFiles.dataFile), prefFiles.dataName.c_str(), prefFiles.writeBuffers, KMER_FILE_BUFFER_BYTES, prefDataBytes, &batchWriter);
                            KmerCodeWriter refilledPassIndexWrite = createFileCodeWriter(fileno(prefFiles.indexFile), prefFiles.indexName.c_str(), prefFiles.writeBuffers + KMER_FILE_BUFFER_BYTES, KMER_FILE_BUFFER_BYTES, prefIndexBytes, &batchWriter);
                            for (size_t key = 0; key < partitionKeys; key++) {
                                if (key == taskFirstKey + refilledPassTaskKeys) {
                                    addKeyTaskCuts(partitionFirstKey, taskFirstKey, taskPrefOffset, taskCost, refilledPassCuts);
                                    refilledPassCuts = KeyTaskCuts();
                                    taskFirstKey = key;
                                    taskPrefOffset = refilledPassDataWrite.writePos + refilledPassDataWrite.bufferPos;
                                    taskCost = partitionCost;
                                }
                                const bool hasEntry = repKeyQueue.empty() == false && repKeyQueue.top().value == key && isRepresentative(partitionFirstKey + key);
                                if (hasEntry) {
                                    const uint64_t indexCell[2] = {partitionFirstKey + key, refilledPassDataWrite.writePos + refilledPassDataWrite.bufferPos};
                                    putBytes(refilledPassIndexWrite, reinterpret_cast<const char *>(indexCell), sizeof(indexCell));
                                    putPrefKey(refilledPassDataWrite, partitionFirstKey + key);
                                }
                                DBKeyType prevTarget = 0;
                                size_t targetRange = SIZE_MAX;
                                size_t targetResidues = 0;
                                mergeKeyTargets(reads, repKeyQueue, targetQueue, keyRows, key, exchange.globalIdBits, [&](DBKeyType target) {
                                    if (hasEntry) {
                                        putPrefKey(refilledPassDataWrite, target - prevTarget);
                                        prevTarget = target;
                                        targetResidues += seqDbr.getSeqLenFromRange(targetRange, target);
                                    }
                                });
                                if (hasEntry) {
                                    putBytes(refilledPassDataWrite, "", 1);
                                    partitionEntryCount++;
                                    partitionCost += (index[seqDbr.findIdRange(partitionFirstKey + key)].length - 2) * targetResidues;
                                    const PartitionCut cut = { key + 1, refilledPassDataWrite.writePos + refilledPassDataWrite.bufferPos - taskPrefOffset, partitionCost - taskCost };
                                    refilledPassCuts.add(cut);
                                }
                            }
                            writeKmerCodeBuffer(refilledPassDataWrite);
                            writeKmerCodeBuffer(refilledPassIndexWrite);
                            prefDataBytes = refilledPassDataWrite.writePos;
                            addKeyTaskCuts(partitionFirstKey, taskFirstKey, taskPrefOffset, taskCost, refilledPassCuts);
                        }
                        addPartitionEnd(prefDataBytes, partitionCost);
                        prefIndexBytes += partitionEntryCount * 2 * sizeof(uint64_t);
                    }
                }
            }
            delete batchReader;
            // the merging node of the partitions is the only reader of their pairs
            std::vector<PassRow> splitRows;
            std::vector<size_t> splitRowOffsets;
            getPairPassRows(partitionRows, firstPartition, endPartition, splitRows, splitRowOffsets);
            punchedRanges += punchPassRows(receivedPairFiles, pairFileRows, splitRows, releasedOwnPairBytes);
        }
    }
    closeReceivedFiles(receivedPairFiles);
    batchWriter.waitWrites();
    MMseqsNode::addPhaseSeconds(mergeValues, "waiting to read pairs", readWaitSeconds);
    MMseqsNode::addPhaseSeconds(mergeValues, "waiting to write the prefilter", batchWriter.getWaitSeconds() / par.threads);
    MMseqsNode::addPhaseValue(mergeValues, "Pair bytes merged", SSTR(mergedPairBytes));
    MMseqsNode::addPhaseValue(mergeValues, "Pair file ranges freed", SSTR(punchedRanges));
    Util::releasePages(pairPassBuffer, pairPassBufferBytes);
    delete[] pairPassBuffer;
    Util::releasePages(pairPassTargets, pairPassTargetsSize * sizeof(DBKeyType));
    delete[] pairPassTargets;
    delete[] prefFiles.writeBuffers;
    MMseqsNode::closeClusterFiles(prefFiles.dataFile, prefFiles.dataName, prefFiles.indexFile, prefFiles.indexName, prefDataBytes, prefIndexBytes);
    // a row per part of a partition with entries: firstKey, prefOffset, prefBytes, alignment cells and node; a part ends at the first cut where its cells reach 1 / PARTITION_ROWS_PER_NODE of what this node merged
    const size_t rowCost = std::max<size_t>(1, static_cast<size_t>(mergedCost / PARTITION_ROWS_PER_NODE));
    std::string partitionsData;
    for (size_t partition = 0; partition + 1 < partitionCutOffsets.size(); partition++) {
        const PartitionCut *rowStart = &partitionCuts[partitionCutOffsets[partition]];
        for (size_t i = partitionCutOffsets[partition] + 1; i < partitionCutOffsets[partition + 1]; i++) {
            const PartitionCut &cut = partitionCuts[i];
            if (i + 1 < partitionCutOffsets[partition + 1] && cut.cost - rowStart->cost < rowCost) {
                continue;
            }
            if (cut.prefBytes > rowStart->prefBytes) {
                partitionsData += SSTR(rowStart->key) + "\t" + SSTR(rowStart->prefBytes) + "\t" + SSTR(cut.prefBytes - rowStart->prefBytes) + "\t" + SSTR(cut.cost - rowStart->cost)
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
    const float kmersPerSequenceScale = par.kmersPerSequenceScale.values.aminoacid();
    KmerExchange exchange;
    exchange.memoryLimit = memoryLimit;
    // every node cuts the same send splits, hash ranges and partitions, sized by the k-mers of all representatives of the database
    size_t totalKmerCount = 0;
    size_t totalSequenceBytes = 0;
    for (size_t range = 0; range < seqDbr.getIndexSize(); range++) {
        const size_t rangeKeys = ((range + 1 < seqDbr.getIndexSize()) ? static_cast<size_t>(index[range + 1].id) : seqDbr.getSize()) - index[range].id;
        totalKmerCount += seqDbr.getRangeRepresentativeCount(range) * computeSeqKmerCount(static_cast<int>(index[range].length - 2), par.kmerSize, par.kmersPerSequence, kmersPerSequenceScale);
        totalSequenceBytes += rangeKeys * index[range].length;
    }
    // RANGES_PER_NODE send splits a node, one without other nodes to balance, cut like decomposeDomainByAminoAcid on sequence bytes plus SEND_KMER_BYTES a k-mer
    exchange.sendSplitCount = (MMseqsNode::count > 1) ? RANGES_PER_NODE * MMseqsNode::count : 1;
    std::vector<size_t> &sendSplitKeyOffsets = exchange.sendSplitKeyOffsets;
    sendSplitKeyOffsets.assign(exchange.sendSplitCount + 1, seqDbr.getSize());
    sendSplitKeyOffsets[0] = 0;
    const size_t sendSplitCost = std::max<size_t>(1, (totalKmerCount * SEND_KMER_BYTES + totalSequenceBytes + exchange.sendSplitCount - 1) / exchange.sendSplitCount);
    size_t costBefore = 0;
    for (size_t range = 0, sendSplit = 1; range < seqDbr.getIndexSize(); range++) {
        const size_t rangeKeys = ((range + 1 < seqDbr.getIndexSize()) ? static_cast<size_t>(index[range + 1].id) : seqDbr.getSize()) - index[range].id;
        const size_t rangeKmers = seqDbr.getRangeRepresentativeCount(range) * computeSeqKmerCount(static_cast<int>(index[range].length - 2), par.kmerSize, par.kmersPerSequence, kmersPerSequenceScale);
        const size_t rangeCost = rangeKmers * SEND_KMER_BYTES + rangeKeys * index[range].length;
        for (; sendSplit < exchange.sendSplitCount && costBefore + rangeCost >= sendSplit * sendSplitCost; sendSplit++) {
            const size_t cutCost = sendSplit * sendSplitCost - costBefore;
            sendSplitKeyOffsets[sendSplit] = index[range].id + static_cast<size_t>((static_cast<KmerPositionBit::uint128>(cutCost) * rangeKeys + rangeCost - 1) / rangeCost);
        }
        costBefore += rangeCost;
    }
    size_t maxSendSplitKeys = 0;
    for (size_t sendSplit = 0; sendSplit < exchange.sendSplitCount; sendSplit++) {
        maxSendSplitKeys = std::max(maxSendSplitKeys, sendSplitKeyOffsets[sendSplit + 1] - sendSplitKeyOffsets[sendSplit]);
    }
    // a clusthashfast subset: the representative bits of the keys to send and merge
    exchange.fileKeyOffsets = seqDbr.getFileKeyOffsets(MMseqsNode::count);
    exchange.hasRepresentativeBits = FileUtil::fileExists((par.db1 + ".bits." + SSTR(MMseqsNode::rank)).c_str());
    // a result is complete once the master writes its dbtype
    if (MMseqsNode::isMaster() && FileUtil::fileExists((par.db2 + ".dbtype").c_str())) {
        FileUtil::remove((par.db2 + ".dbtype").c_str());
    }

    struct rlimit openFileLimit = {0, 0};
    getrlimit(RLIMIT_NOFILE, &openFileLimit);
    // a node reads the exchange file of every node while it writes its own, or its pref data and index
    const size_t minOpenFiles = seqDbr.getDataFileCnt() + OTHER_OPEN_FILES + MMseqsNode::count + 2;
    if (minOpenFiles > openFileLimit.rlim_cur) {
        Debug(Debug::ERROR) << "Raise the open file limit (ulimit -n) to at least " << minOpenFiles << "\n";
        EXIT(EXIT_FAILURE);
    }

    // keys are in length descending order, so a record needs no length, and no position as the diagonal comes later
    exchange.adjKeyBits = (includeAdjacency ? 6 : 0) * KmerPositionBit::RESIDUE_BITS;
    exchange.adjKeyBase = includeAdjacency ? adjSubMat->alphabetSize : 1;
    exchange.adjKeyCodeBits = includeAdjacency ? getBitWidth(MathUtil::ipow<size_t>(exchange.adjKeyBase, 6) - 1) : 0;
    // a k-mer record holds its hash in the top kmerBits bits, the hash of a whole sequence takes all 64
    exchange.kmerHashShift = 64 - getKmerBits(subMat, par.kmerSize);

    // hash ranges are equal parts of the 64 bit hash, RANGES_PER_NODE a node and enough for the threads that write the k-mer file of a key split
    exchange.hashRangeCount = std::max(RANGES_PER_NODE * MMseqsNode::count, TASKS_PER_THREAD * par.threads);
    exchange.globalIdBits = getBitWidth(seqDbr.getSize() - 1);
    exchange.maxRepSeqs = 1 + (includeAdjacency ? par.adjIteration : 0);
    // as many as a node half as fast as the others merges, like the own hash ranges
    exchange.selfMergedSendSplitCount = exchange.sendSplitCount / MMseqsNode::count * MMseqsNode::count / (2 * MMseqsNode::count - 1);
    // a receiving node reads about a member id, a rep candidate and its index codes per record
    const size_t receiverKmerBytes = (2 * exchange.globalIdBits + exchange.adjKeyCodeBits + CHAR_BIT - 1) / CHAR_BIT;
    // a representative node reads a target id and sorts it, a record can be a target of every representative of its group
    const size_t pairTargetBytes = (exchange.globalIdBits + CHAR_BIT - 1) / CHAR_BIT + sizeof(DBKeyType);
    // the nodes take the items of these queues; every node removes the claims of its own items that a killed run left before the barrier of the memory limits
    MMseqsNode::WorkQueue sendQueue(par.db2 + ".send_splits", exchange.sendSplitCount);
    MMseqsNode::WorkQueue hashQueue(par.db2 + ".hash_ranges", exchange.hashRangeCount);
    MMseqsNode::WorkQueue mergeQueue(par.db2 + ".merge_splits", exchange.sendSplitCount);
    sendQueue.removeTakenItemFiles();
    hashQueue.removeTakenItemFiles();
    mergeQueue.removeTakenItemFiles();
    const size_t minPassLimit = std::max<size_t>(1, getPassLimit(MMseqsNode::getSmallestMemoryLimit(par.db2 + ".memory", memoryLimit, par.threads)));
    const size_t passes = std::max<size_t>(1, (totalKmerCount * receiverKmerBytes + minPassLimit - 1) / minPassLimit);
    // a pass holds about TASKS_PER_THREAD hash sub ranges per thread, so the threads take whole hash sub ranges without decoding a row to cut it
    exchange.hashSubRangesPerRange = (std::max(passes, exchange.hashRangeCount) * TASKS_PER_THREAD * par.threads + exchange.hashRangeCount - 1) / exchange.hashRangeCount;
    exchange.hashSubRangeCount = exchange.hashRangeCount * exchange.hashSubRangesPerRange;
    exchange.rowsPerHashRange = exchange.hashSubRangesPerRange + 1;
    const size_t representativeBitsMemory = exchange.hasRepresentativeBits ? ((maxSendSplitKeys + 63) / 64 + 1) * sizeof(uint64_t) : 0;
    exchange.writeBehindBytes = memoryLimit / WRITE_BEHIND_MEMORY_DIVISOR;
    // every thread writes through two buffers at most, a key split holds its rows until it writes them after its codes
    const size_t fixedMemory = representativeBitsMemory + 2 * par.threads * KMER_FILE_BUFFER_BYTES + exchange.writeBehindBytes
                               + exchange.hashRangeCount * exchange.rowsPerHashRange * sizeof(KmerFileRow);
    exchange.keySplitMemoryLimit = (fixedMemory < memoryLimit) ? static_cast<size_t>(static_cast<double>(memoryLimit - fixedMemory) * 0.95) : 0;
    const size_t rangePairPasses = std::max<size_t>(1, (totalKmerCount / exchange.sendSplitCount * exchange.maxRepSeqs * pairTargetBytes + minPassLimit - 1) / minPassLimit);
    // all threads merge a partition, which is about half a pass or smaller, so only a few are too big for a pass
    exchange.partitionsPerRange = std::max(PARTITIONS_PER_THREAD * par.threads, 2 * rangePairPasses);
    exchange.pairRowBytes = exchange.sendSplitCount * exchange.partitionsPerRange * sizeof(PairFileRow);

    Timer phaseTimer;
    std::string sendValues;
    std::vector<size_t> sendSplits;
    // the own rows of the first own hash ranges that fit and that a node half as fast as the others takes itself
    OwnKmers ownKmers = {NULL, 0, getPassLimit(exchange.keySplitMemoryLimit - std::min(exchange.keySplitMemoryLimit, exchange.pairRowBytes)),
                         std::vector<std::vector<KmerFileRow> >(exchange.hashRangeCount), std::vector<size_t>()};
    const size_t sendSplitKmerBytes = totalKmerCount / exchange.sendSplitCount * sizeof(KmerPositionBit);
    const size_t ownMemory = std::min(ownKmers.limit, exchange.keySplitMemoryLimit - std::min(exchange.keySplitMemoryLimit, sendSplitKmerBytes));
    const size_t ownHashRangeBytes = std::max<size_t>(1, totalKmerCount * receiverKmerBytes / (MMseqsNode::count * exchange.hashRangeCount));
    const size_t ownItems = (exchange.hashRangeCount - MMseqsNode::rank + MMseqsNode::count - 1) / MMseqsNode::count;
    const size_t ownHashRanges = std::min(ownItems * MMseqsNode::count / (2 * MMseqsNode::count - 1), ownMemory / ownHashRangeBytes);
    for (size_t i = 0; i < ownHashRanges; i++) {
        ownKmers.hashRanges.push_back(MMseqsNode::rank + i * MMseqsNode::count);
    }
    if (MMseqsNode::isMaster()) {
        Debug(Debug::INFO) << "Send k-mers in " << exchange.sendSplitCount << " send splits\n";
    }
    const size_t sentBytes = sendKmers<T, includeAdjacency, L>(par, seqDbr, subMat, exchange, sendQueue, representativeBitsMemory, sendSplits, sendValues, ownKmers);
    // no node takes hash ranges before the barrier, so takeNextItem takes the first own hash ranges in order
    for (size_t i = 0; i < ownKmers.hashRanges.size(); i++) {
        hashQueue.takeNextItem();
    }
    MMseqsNode::addPhaseTime(sendValues, "sending k-mers", phaseTimer);
    MMseqsNode::addPhaseValue(sendValues, "K-mer bytes sent", SSTR(sentBytes + ownKmers.bytes));
    MMseqsNode::addPhaseValue(sendValues, "K-mer bytes kept in memory", SSTR(ownKmers.bytes));
    MMseqsNode::addPhaseValue(sendValues, "Send splits", SSTR(sendSplits.size()));
    MMseqsNode::printPhaseValues(sendValues);
    MMseqsNode::barrier(par.db2 + ".kmers.done");
    const double sendWaitSeconds = phaseTimer.getTimediff();
    // every node has passed the send splits
    sendQueue.removeTakenItemFiles();
    // every node has read the memory limits
    if (MMseqsNode::isMaster()) {
        FileUtil::remove((par.db2 + ".memory").c_str());
    }

    phaseTimer.reset();
    std::string receiveValues;
    MMseqsNode::addPhaseSeconds(receiveValues, "waiting for the other nodes to send k-mers", sendWaitSeconds);
    std::vector<size_t> hashRanges;
    size_t pairBytes = 0;
    if (MMseqsNode::isMaster()) {
        Debug(Debug::INFO) << "Receive k-mers in " << exchange.hashRangeCount << " hash ranges\n";
    }
    OwnPairs ownPairs = {NULL, 0, 0, std::vector<std::vector<PairFileRow> >(exchange.sendSplitCount)};
    const size_t receivedBytes = receiveKmers(par, seqDbr, exchange, hashQueue, hashRanges, receiveValues, pairBytes, ownKmers, ownPairs);
    MMseqsNode::addPhaseTime(receiveValues, "receiving k-mers", phaseTimer);
    MMseqsNode::addPhaseValue(receiveValues, "K-mer bytes received", SSTR(receivedBytes));
    MMseqsNode::addPhaseValue(receiveValues, "Pair bytes written", SSTR(pairBytes));
    MMseqsNode::addPhaseValue(receiveValues, "Hash ranges", SSTR(hashRanges.size()));
    MMseqsNode::printPhaseValues(receiveValues);
    MMseqsNode::barrier(par.db2 + ".pairs.done");
    std::string mergeValues;
    MMseqsNode::addPhaseTime(mergeValues, "waiting for the other nodes to receive k-mers", phaseTimer);
    // every node has passed the hash ranges and read the k-mer file of this node
    hashQueue.removeTakenItemFiles();
    // freeing the blocks and cached pages of a large k-mer file takes long, so it is removed while the pairs are merged
    const std::string kmerFile = par.db2 + ".kmers." + SSTR(MMseqsNode::rank);
    std::thread kmerFileRemoval([&kmerFile]() { FileUtil::remove(kmerFile.c_str()); });
    if (MMseqsNode::isMaster()) {
        FileUtil::remove((par.db2 + ".kmers.done").c_str());
    }

    std::vector<size_t> mergeSplits;
    if (MMseqsNode::isMaster()) {
        Debug(Debug::INFO) << "Merge pairs in " << exchange.sendSplitCount << " splits\n";
    }
    const std::string partitionsData = mergePairFiles(par, seqDbr, exchange, mergeQueue, mergeSplits, mergeValues, ownPairs);
    kmerFileRemoval.join();
    MMseqsNode::addPhaseTime(mergeValues, "merging pairs", phaseTimer);
    MMseqsNode::addPhaseValue(mergeValues, "Merge splits", SSTR(mergeSplits.size()));
    MMseqsNode::printPhaseValues(mergeValues);
    MMseqsNode::publishRunFile(par.db2 + ".partitions." + SSTR(MMseqsNode::rank), partitionsData);
    // freeing a large file takes long, so every node removes its pair file while the master writes the partitions
    const std::string pairFile = par.db2 + ".pairs." + SSTR(MMseqsNode::rank);
    if (MMseqsNode::isMaster()) {
        // every node has merged its send splits and read the pair files
        const std::string partitions = MMseqsNode::gatherRunFiles(par.db2 + ".partitions.");
        std::string mergeWaitValues;
        MMseqsNode::addPhaseTime(mergeWaitValues, "waiting for the other nodes to merge pairs", phaseTimer);
        MMseqsNode::publishRunFile(par.db2 + ".merged", "");
        std::thread pairFileRemoval([&pairFile]() { FileUtil::remove(pairFile.c_str()); });
        MMseqsNode::printPhaseValues(mergeWaitValues);
        mergeQueue.removeTakenItemFiles();
        FileUtil::remove((par.db2 + ".pairs.done").c_str());
        // the partition rows of all nodes are sorted by key
        std::vector<std::pair<size_t, char *> > partitionRows;
        for (char *row = (char *) partitions.c_str(); *row != '\0'; row = Util::skipLine(row)) {
            partitionRows.push_back(std::make_pair(Util::fast_atoi<size_t>(row), row));
        }
        SORT_SERIAL(partitionRows.begin(), partitionRows.end());
        std::string sortedPartitions;
        for (size_t i = 0; i < partitionRows.size(); i++) {
            sortedPartitions.append(partitionRows[i].second, Util::skipLine(partitionRows[i].second) - partitionRows[i].second);
        }
        MMseqsNode::publishFile(par.db2 + ".partitions", sortedPartitions);
        MMseqsNode::removeOnceRead(par.db2 + ".merged");
        DBWriter::writeDbtypeFile(par.db2.c_str(), DBReader<DBKeyType>::setExtendedDbtype(Parameters::DBTYPE_PREFILTER_RES, Parameters::DBTYPE_EXTENDED_LINFINITY), false);
        pairFileRemoval.join();
    } else {
        // the master publishes this once every node has merged, so no node takes merge ranges or reads the pair file of this node any more
        MMseqsNode::waitForRunFile(par.db2 + ".merged");
        MMseqsNode::removeOnceRead(par.db2 + ".merged");
        mergeQueue.removeTakenItemFiles();
        FileUtil::remove(pairFile.c_str());
    }
}

template void kmermatcherLinfinity<short, true, unsigned short>(Parameters &, DBReader<DBKeyType> &, BaseMatrix *, size_t);
template void kmermatcherLinfinity<short, false, unsigned short>(Parameters &, DBReader<DBKeyType> &, BaseMatrix *, size_t);
template void kmermatcherLinfinity<int, true, unsigned short>(Parameters &, DBReader<DBKeyType> &, BaseMatrix *, size_t);
template void kmermatcherLinfinity<int, false, unsigned short>(Parameters &, DBReader<DBKeyType> &, BaseMatrix *, size_t);
template void kmermatcherLinfinity<int, true, int>(Parameters &, DBReader<DBKeyType> &, BaseMatrix *, size_t);
template void kmermatcherLinfinity<int, false, int>(Parameters &, DBReader<DBKeyType> &, BaseMatrix *, size_t);
