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

#ifndef SIZE_T_MAX
#define SIZE_T_MAX ((size_t) -1)
#endif

static const size_t KMER_FILE_BUFFER_SIZE = 16 * 1024 * 1024;
// the writes that wait for the file system while the threads go on take the memory limit divided by this
static const size_t WRITE_BEHIND_MEMORY_DIVISOR = 8;
// the bit buffer keeps less than a byte between codes, so this many bits always fit in one step
static const unsigned int KMER_FILE_STEP_BITS = 64 - CHAR_BIT;
// open files besides the k-mer and sequence data files: standard streams and barrier files, with headroom
static const size_t OTHER_OPEN_FILES = 64;
// the work that all threads share is cut into this many chunks per thread
static const size_t CHUNKS_PER_THREAD = 2;
// a send range is cut into at least this many partitions per thread
static const size_t PARTITIONS_PER_THREAD = 4;
// a thread of a receiving pass takes blocks of at most this part of its share of the output, so the blocks not yet filled when the output is full are a small part of it
static const size_t OUTPUT_BLOCKS_PER_THREAD = 16;
// a partition row ends once its alignment cells reach 1 / (nodes * this) of what its node merged
static const size_t PARTITIONS_PER_NODE = 4;
// a key chunk keeps fewer than twice this many places to cut a partition
static const size_t CUTS_PER_CHUNK = 8;
// the nodes take the send ranges and the hash ranges one at a time, about this many each, so a node that is faster takes more
static const size_t RANGES_PER_NODE = 8;
// sending a k-mer takes about as long as reading this many bytes of sequence
static const size_t SEND_KMER_BYTES = 12;
// the largest file on ext4, and on a Lustre object of a single stripe
static const size_t LARGE_FILE_BYTES = 16ULL << 40;
// the bytes of an exchange file are freed in whole parts of this size
static const size_t PUNCH_ALIGNMENT = 1024 * 1024;

// the codes of a row of a k-mer file, one after the other
enum { KMER_MEMBER_CODES, KMER_REP_CANDIDATE_CODES, KMER_INDEX_CODES, KMER_CODE_COUNT };
// the codes of a row of a pair file, one after the other: per record its key gap, target count, first target and gap bits, then the gaps of its other targets
enum { PAIR_INDEX_CODES, PAIR_TARGET_CODES, PAIR_CODE_COUNT };

// a row of the k-mer file <pref>.kmers.<node> per load and hash sub range, then a row per load and hash range that holds its whole sequence hashes
struct KmerHashSubRange {
    DBKeyType loadFirstKey;
    unsigned int idBits;
    // where every code starts, then the end of the row
    size_t offsets[KMER_CODE_COUNT + 1];
    size_t groupCount;
};

// a row of the pair file <pref>.pairs.<node> per flush and partition with records, a record per representative with its targets in id order without repeats
struct AssignedPairPartition {
    size_t offsets[PAIR_CODE_COUNT + 1];
    size_t partition;
    size_t recordCount;
    size_t targetCount;
};

// a representative and its targets of a group in the output buffer of a receiving pass, at targetOffset of the targets there
struct AssignedPairRecord {
    DBKeyType rep;
    size_t targetOffset;
    size_t targetCount;

    static bool compareByRep(const AssignedPairRecord &first, const AssignedPairRecord &second) {
        return first.rep < second.rep;
    }
};

// a row of a received file that a pass reads, and where its codes are in the pass buffer
struct PassRow {
    size_t file;
    size_t row;
    size_t bufferOffset;
};

// where a received row is at the start of a chunk of its hash sub range: its next group, the hash before it and the bit position of every code
struct KmerChunkStart {
    size_t group;
    uint64_t hash;
    size_t bitPos[KMER_CODE_COUNT];
};

// with fd -1 a buffer in memory, otherwise a buffer that batchWriter writes at writePos of the file whenever it is full
struct KmerCodeWriter {
    int fd;
    const char *fileName;
    char *buffer;
    size_t bufferSize;
    size_t bufferPos;
    size_t writePos;
    uint64_t bitBuffer;
    unsigned int bitCount;
    BatchWriter *batchWriter;
};

// with fd -1 a buffer in memory, otherwise a window of bufferSize bytes at fileOffset that getBits refills up to fileEnd
struct KmerCodeReader {
    char *buffer;
    size_t bufferPos;
    uint64_t bitBuffer;
    unsigned int bitCount;
    int fd;
    const char *fileName;
    size_t bufferSize;
    size_t fileOffset;
    size_t fileEnd;
};

static void writeKmerCodeBuffer(KmerCodeWriter &kmerCodeWrite) {
    kmerCodeWrite.batchWriter->write(kmerCodeWrite.fd, kmerCodeWrite.fileName, kmerCodeWrite.buffer, kmerCodeWrite.bufferPos, kmerCodeWrite.writePos);
    kmerCodeWrite.writePos += kmerCodeWrite.bufferPos;
    kmerCodeWrite.bufferPos = 0;
}

static void readKmerCodeWindow(KmerCodeReader &kmerCodeRead, size_t fileOffset) {
    FileUtil::preadAll(kmerCodeRead.fd, kmerCodeRead.fileName, kmerCodeRead.buffer, std::min(kmerCodeRead.bufferSize, kmerCodeRead.fileEnd - fileOffset), fileOffset);
    kmerCodeRead.fileOffset = fileOffset;
    kmerCodeRead.bufferPos = 0;
}

static inline void putBits(KmerCodeWriter &kmerCodeWrite, uint64_t value, unsigned int width) {
    for (unsigned int done = 0; done < width; done += KMER_FILE_STEP_BITS) {
        const unsigned int bits = std::min(width - done, KMER_FILE_STEP_BITS);
        kmerCodeWrite.bitBuffer |= ((value >> done) & ((1ULL << bits) - 1)) << kmerCodeWrite.bitCount;
        for (kmerCodeWrite.bitCount += bits; kmerCodeWrite.bitCount >= CHAR_BIT; kmerCodeWrite.bitCount -= CHAR_BIT) {
            if (kmerCodeWrite.bufferPos == kmerCodeWrite.bufferSize) {
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
        if (kmerCodeWrite.bufferPos == kmerCodeWrite.bufferSize) {
            writeKmerCodeBuffer(kmerCodeWrite);
        }
        const size_t copyBytes = std::min(count - done, kmerCodeWrite.bufferSize - kmerCodeWrite.bufferPos);
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
            if (kmerCodeRead.bufferPos == kmerCodeRead.bufferSize) {
                readKmerCodeWindow(kmerCodeRead, kmerCodeRead.fileOffset + kmerCodeRead.bufferSize);
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
        if (kmerCodeRead.bufferPos == kmerCodeRead.bufferSize) {
            readKmerCodeWindow(kmerCodeRead, kmerCodeRead.fileOffset + kmerCodeRead.bufferSize);
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
    if (bytePos >= kmerCodeRead.fileOffset && bytePos - kmerCodeRead.fileOffset < kmerCodeRead.bufferSize) {
        kmerCodeRead.bufferPos = bytePos - kmerCodeRead.fileOffset;
    } else {
        readKmerCodeWindow(kmerCodeRead, bytePos);
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

struct FileKmerHash {
    uint64_t hash;
    unsigned int row;
    FileKmerHash(uint64_t hash, unsigned int row) : hash(hash), row(row) {}
};

struct CompareKmerHashAndRow {
    bool operator()(const FileKmerHash &first, const FileKmerHash &second) const {
        return first.hash > second.hash || (first.hash == second.hash && first.row > second.row);
    }
};

typedef std::priority_queue<FileKmerHash, std::vector<FileKmerHash>, CompareKmerHashAndRow> KmerHashQueue;

struct FileKmerAdjKey {
    unsigned int adjKey;
    unsigned int row;
    FileKmerAdjKey(unsigned int adjKey, unsigned int row) : adjKey(adjKey), row(row) {}
};

struct CompareKmerAdjKeyAndRow {
    bool operator()(const FileKmerAdjKey &first, const FileKmerAdjKey &second) const {
        return first.adjKey > second.adjKey || (first.adjKey == second.adjKey && first.row > second.row);
    }
};

typedef std::priority_queue<FileKmerAdjKey, std::vector<FileKmerAdjKey>, CompareKmerAdjKeyAndRow> KmerAdjKeyQueue;

struct FileTargetId {
    DBKeyType targetId;
    unsigned int record;
    FileTargetId(DBKeyType targetId, unsigned int record) : targetId(targetId), record(record) {}
};

struct CompareTargetIdAndRecord {
    bool operator()(const FileTargetId &first, const FileTargetId &second) const {
        return first.targetId > second.targetId || (first.targetId == second.targetId && first.record > second.record);
    }
};

typedef std::priority_queue<FileTargetId, std::vector<FileTargetId>, CompareTargetIdAndRecord> TargetIdQueue;

struct FileRepKey {
    size_t repKey;
    unsigned int row;
    FileRepKey(size_t repKey, unsigned int row) : repKey(repKey), row(row) {}
};

struct CompareRepKeyAndRow {
    bool operator()(const FileRepKey &first, const FileRepKey &second) const {
        return first.repKey > second.repKey || (first.repKey == second.repKey && first.row > second.row);
    }
};

typedef std::priority_queue<FileRepKey, std::vector<FileRepKey>, CompareRepKeyAndRow> RepKeyQueue;

// a received row of a hash sub range and its group read last, with where the codes start, its row in the chunk and where its group read last starts
struct KmerHashSubRangeRead {
    KmerCodeReader codeReads[KMER_CODE_COUNT];
    size_t codeStartBitPos[KMER_CODE_COUNT];
    const KmerHashSubRange *hashSubRangeRow;
    size_t chunkRow;
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

// where a row of a partition is at the start of a key chunk: its records left, the key after its record before and the bit position of every code
struct PairChunkStart {
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

// the cuts of a key chunk where its alignment cells passed a multiple of step, which doubles to keep fewer than twice CUTS_PER_CHUNK of them
struct ChunkCuts {
    ChunkCuts() : step(1) {}
    std::vector<PartitionCut> cuts;
    size_t step;

    void add(const PartitionCut &cut) {
        if (cut.cost / step == (cuts.empty() ? 0 : cuts.back().cost / step)) {
            return;
        }
        cuts.push_back(cut);
        while (cuts.size() >= 2 * CUTS_PER_CHUNK) {
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

// a partition that the rep node merges: its row readers, where they start every key chunk, key targets, key chunks with their entry and index offsets, alignment cells and cuts, alignment cells
struct PartitionMerge {
    std::vector<PairRowRead> rowReads;
    std::vector<PairChunkStart> chunkStarts;
    std::vector<size_t> keyOffsets;
    DBKeyType *targets;
    std::vector<size_t> keyChunkOffsets;
    std::vector<size_t> keyChunkDataOffsets;
    std::vector<size_t> keyChunkEntryOffsets;
    std::vector<size_t> keyChunkCostOffsets;
    std::vector<ChunkCuts> keyChunkCuts;
    size_t cost;
};

// a partition merge with threads holds per row its reader and its starts of the key chunks, and per thread a copy of the readers and their places in the queues
static size_t getPartitionMergeBytes(size_t rowCount, size_t threads) {
    return rowCount * (sizeof(PairRowRead) + CHUNKS_PER_THREAD * threads * sizeof(PairChunkStart)
                       + threads * (sizeof(PairRowRead) + sizeof(FileRepKey) + sizeof(FileTargetId) + sizeof(unsigned int)));
}

static inline void queueNextKmerGroup(KmerHashQueue &queue, unsigned int row, KmerHashSubRangeRead &read) {
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
    queue.push(FileKmerHash(read.hash, row));
}

// the file of every node that a node reads for the whole phase: where its row table starts, its row blocks and, of a pair file, the code bytes of every partition
struct ReceivedFile {
    std::string fileName;
    FILE *file;
    size_t rowTableOffset;
    size_t blockCount;
    std::vector<size_t> itemBytes;
    // without a file, the rows that this node sent to itself in memory
    char *data;
};

template <typename Row>
static size_t getRowEnd(const Row &row) {
    return row.offsets[sizeof(row.offsets) / sizeof(size_t) - 1];
}

// writes at rowTableOffset where the rows of every item start in every block of rows, item after item, then the end of every block, from blockRowStarts that hold them
// block after block, then the code bytes of itemBytes, blockCount and rowTableOffset; closes the file
static void writeRowTable(FILE *file, const std::string &fileName, size_t itemCount, const std::vector<size_t> &blockRowStarts, const std::vector<size_t> &itemBytes, size_t rowTableOffset) {
    const size_t blockCount = blockRowStarts.size() / (itemCount + 1);
    std::vector<size_t> table(blockRowStarts.size());
    for (size_t block = 0; block < blockCount; block++) {
        for (size_t item = 0; item <= itemCount; item++) {
            table[item * blockCount + block] = blockRowStarts[block * (itemCount + 1) + item];
        }
    }
    table.insert(table.end(), itemBytes.begin(), itemBytes.end());
    table.push_back(blockCount);
    table.push_back(rowTableOffset);
    FileUtil::pwriteAll(fileno(file), fileName.c_str(), reinterpret_cast<const char *>(table.data()), table.size() * sizeof(size_t), rowTableOffset);
    if (fclose(file) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << fileName << "\n";
        EXIT(EXIT_FAILURE);
    }
}

// opens the file to read it and free what is read, and reads its itemByteCount code bytes, the block count and where the row table starts, written by writeRowTable
// with itemCount items
static void openReceivedFile(ReceivedFile &receivedFile, size_t itemCount, size_t itemByteCount) {
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
    receivedFile.blockCount = tail[itemByteCount];
    receivedFile.rowTableOffset = tail[itemByteCount + 1];
    // every node must cut as many items
    if (receivedFile.rowTableOffset + (itemCount + 1) * receivedFile.blockCount * sizeof(size_t) + tailBytes != static_cast<size_t>(sb.st_size)) {
        Debug(Debug::ERROR) << "Data file " << receivedFile.fileName << " has other items than " << itemCount << "\n";
        EXIT(EXIT_FAILURE);
    }
}

// the rows of item, block after block, read now or added to readRequests for the caller to read
template <typename Row>
static void readItemRows(const ReceivedFile &receivedFile, size_t item, std::vector<Row> &rows, std::vector<BatchReader::ReadRequest> *readRequests = NULL) {
    const size_t blockCount = receivedFile.blockCount;
    std::vector<size_t> rowStarts(2 * blockCount);
    FileUtil::preadAll(fileno(receivedFile.file), receivedFile.fileName.c_str(), reinterpret_cast<char *>(rowStarts.data()), rowStarts.size() * sizeof(size_t),
                       receivedFile.rowTableOffset + item * blockCount * sizeof(size_t));
    size_t rowCount = 0;
    for (size_t block = 0; block < blockCount; block++) {
        rowCount += (rowStarts[blockCount + block] - rowStarts[block]) / sizeof(Row);
    }
    rows.resize(rowCount);
    size_t row = 0;
    for (size_t block = 0; block < blockCount; block++) {
        const size_t bytes = rowStarts[blockCount + block] - rowStarts[block];
        if (readRequests == NULL) {
            FileUtil::preadAll(fileno(receivedFile.file), receivedFile.fileName.c_str(), reinterpret_cast<char *>(rows.data() + row), bytes, rowStarts[block]);
        } else if (bytes > 0) {
            const BatchReader::ReadRequest readRequest = {fileno(receivedFile.file), receivedFile.fileName.c_str(), reinterpret_cast<char *>(rows.data() + row), rowStarts[block], bytes, bytes};
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
    SORT_SERIAL(readOrder.begin(), readOrder.end(), [&](size_t first, size_t second) {
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

// frees the codes of passRows in their files once they are read and processed, rows that follow each other in a file at once and within whole
// PUNCH_ALIGNMENT bytes, so no file system block is zeroed; returns the ranges freed
template <typename Row>
static size_t punchPassRows(const std::vector<ReceivedFile> &receivedFiles, const std::vector<std::vector<Row> > &fileRows, const std::vector<PassRow> &passRows) {
    std::vector<std::pair<size_t, std::pair<size_t, size_t> > > fileRanges;
    for (size_t i = 0; i < passRows.size(); i++) {
        const Row &codeRow = fileRows[passRows[i].file][passRows[i].row];
        if (receivedFiles[passRows[i].file].data == NULL && getRowEnd(codeRow) > codeRow.offsets[0]) {
            fileRanges.push_back(std::make_pair(passRows[i].file, std::make_pair(codeRow.offsets[0], getRowEnd(codeRow))));
        }
    }
    SORT_SERIAL(fileRanges.begin(), fileRanges.end());
    size_t punchedRanges = 0;
    for (size_t i = 0; i < fileRanges.size();) {
        const size_t file = fileRanges[i].first;
        const size_t start = (fileRanges[i].second.first + PUNCH_ALIGNMENT - 1) / PUNCH_ALIGNMENT * PUNCH_ALIGNMENT;
        size_t end = fileRanges[i].second.second;
        for (i++; i < fileRanges.size() && fileRanges[i].first == file && fileRanges[i].second.first == end; i++) {
            end = fileRanges[i].second.second;
        }
        end = end / PUNCH_ALIGNMENT * PUNCH_ALIGNMENT;
        if (end > start) {
            FileUtil::punchHole(fileno(receivedFiles[file].file), start, end - start);
            punchedRanges++;
        }
    }
    return punchedRanges;
}

// a reader of a code of a row: in the memory of the received file, in the pass buffer at bufferOffset, or with windowBytes through window
template <typename Row>
static KmerCodeReader getRowCodeReader(const ReceivedFile &receivedFile, const Row &codeRow, size_t bufferOffset, size_t code, char *passBuffer, char *window, size_t windowBytes) {
    KmerCodeReader codeRead = {NULL, 0, 0, 0, -1, NULL, SIZE_T_MAX, 0, 0};
    if (receivedFile.data != NULL) {
        codeRead.buffer = receivedFile.data + codeRow.offsets[code];
    } else if (windowBytes == 0) {
        codeRead.buffer = passBuffer + bufferOffset + codeRow.offsets[code] - codeRow.offsets[0];
    } else {
        codeRead.buffer = window;
        codeRead.fd = fileno(receivedFile.file);
        codeRead.fileName = receivedFile.fileName.c_str();
        codeRead.bufferSize = windowBytes;
        codeRead.fileEnd = codeRow.offsets[code + 1];
        readKmerCodeWindow(codeRead, codeRow.offsets[code]);
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

// --createdb-mode 3: the send ranges, hash ranges and partitions that every node cuts the same, and the limits of this node
struct KmerExchange {
    // the keys of every send range, whose k-mers the node that takes it sends; the merge splits are key ranges of its partitions
    std::vector<size_t> sendRangeKeyOffsets;
    size_t sendRangeCount;
    // the keys of the data files, and whether a clusthashfast subset has the representative bits of their keys
    std::vector<size_t> fileKeyOffsets;
    bool hasRepresentativeBits;
    size_t memoryLimit;
    size_t writeBehindMemory;
    size_t loadMemoryLimit;
    // the rows of a flush of pair records, a row per partition at most, which a receiving node holds until it writes them
    size_t pairRowsMemory;
    unsigned int adjKeyBits;
    // the residues of an adjKey are sent as the digits of a number of base adjKeyBase in adjKeyCodeBits
    unsigned int adjKeyBase;
    unsigned int adjKeyCodeBits;
    unsigned int kmerHashShift;
    unsigned int globalIdBits;
    size_t maxRepSeqs;
    size_t hashRangeCount;
    size_t hashSubRangesPerRange;
    size_t hashSubRangeCount;
    size_t rowsPerHashRange;
    size_t partitionsPerRange;
};

// the hash sub ranges [first, end) of row rangeHashSubRange of a hash range: one for a row of k-mers, all of the hash range for its row of whole sequence hashes
static void getRowHashSubRanges(const KmerExchange &exchange, size_t hashRange, size_t rangeHashSubRange, size_t &first, size_t &end) {
    const bool isWholeSequence = (rangeHashSubRange >= exchange.hashSubRangesPerRange);
    first = hashRange * exchange.hashSubRangesPerRange + (isWholeSequence ? 0 : rangeHashSubRange);
    end = first + (isWholeSequence ? exchange.hashSubRangesPerRange : 1);
}

static inline unsigned int packAdjKey(const KmerExchange &exchange, unsigned int adjKey) {
    unsigned int adjKeyCode = 0;
    for (int j = 5; j >= 0; j--) {
        adjKeyCode = adjKeyCode * exchange.adjKeyBase + KmerPositionBit::getAdjacentSeq(adjKey, j);
    }
    return adjKeyCode;
}

static inline unsigned int unpackAdjKey(const KmerExchange &exchange, unsigned int adjKeyCode) {
    unsigned int adjKey = 0;
    for (int j = 0; j < 6; j++) {
        adjKey |= (adjKeyCode % exchange.adjKeyBase) << (KmerPositionBit::RESIDUE_BITS * j);
        adjKeyCode /= exchange.adjKeyBase;
    }
    return adjKey;
}

// every node cuts every send range into the same partitions, the pairs of a representative go to its partition
static size_t getPartitionFirstKey(const KmerExchange &exchange, size_t sendRange, size_t partition) {
    return exchange.sendRangeKeyOffsets[sendRange] + (partition * (exchange.sendRangeKeyOffsets[sendRange + 1] - exchange.sendRangeKeyOffsets[sendRange]) + exchange.partitionsPerRange - 1) / exchange.partitionsPerRange;
}

// a node sends and merges a send range with the representative bits of its keys
static void setSendRange(const Parameters &par, const KmerExchange &exchange, size_t sendRange) {
    localFirstKey = exchange.sendRangeKeyOffsets[sendRange];
    delete[] representativeBits;
    representativeBits = NULL;
    if (exchange.hasRepresentativeBits) {
        representativeBits = MMseqsNode::readRepresentativeBits(par.db1, exchange.fileKeyOffsets, localFirstKey, exchange.sendRangeKeyOffsets[sendRange + 1] - localFirstKey);
    }
}

// the distinct ids of [groupStart, groupEnd), sorted in groupBuffer or, if they do not fit, set as load bits counted per 512; returns their count
static size_t getGroupIds(const KmerPositionBit *hashSeqPair, size_t groupStart, size_t groupEnd, uint64_t *groupBuffer, size_t groupBufferWords, uint64_t *setBitCounts, size_t loadWords) {
    if (groupEnd - groupStart <= groupBufferWords) {
        for (size_t pos = groupStart; pos < groupEnd; pos++) {
            groupBuffer[pos - groupStart] = hashSeqPair[pos].getId();
        }
        SORT_SERIAL(groupBuffer, groupBuffer + (groupEnd - groupStart));
        return static_cast<size_t>(std::unique(groupBuffer, groupBuffer + (groupEnd - groupStart)) - groupBuffer);
    }
    memset(groupBuffer, 0, loadWords * sizeof(uint64_t));
    for (size_t pos = groupStart; pos < groupEnd; pos++) {
        const DBKeyType id = hashSeqPair[pos].getId();
        groupBuffer[id / 64] |= 1ULL << (id % 64);
    }
    size_t memberCount = 0;
    for (size_t word = 0; word < loadWords; word++) {
        if (word % 8 == 0) {
            setBitCounts[word / 8] = memberCount;
        }
        memberCount += __builtin_popcountll(groupBuffer[word]);
    }
    return memberCount;
}

// the rows of a pass with groups, hash sub range after hash sub range of a hash range in id order, and where the rows of every hash sub range start
static void getPassRows(const KmerExchange &exchange, const std::vector<std::vector<KmerHashSubRange> > &fileRows, size_t passFirstHashSubRange, size_t passEndHashSubRange,
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
        // the loads are key ranges, so their rows in key order give the members of a group in id order whichever node sent them
        SORT_SERIAL(rows.begin() + rowOffsets.back(), rows.end(), [&](const PassRow &first, const PassRow &second) {
            return fileRows[first.file][first.row].loadFirstKey < fileRows[second.file][second.row].loadFirstKey;
        });
        rowOffsets.push_back(rows.size());
    }
}

// the pair file of this node and its size
struct AssignedPairFile {
    std::string name;
    FILE *file;
    BatchWriter *batchWriter;
    size_t bytes;
    // the rows of every flush are written after its records as a block: where the rows of every send range start in it, then its end, block after block
    std::vector<size_t> blockRowStarts;
    // the code bytes of every partition, send range after send range
    std::vector<size_t> partitionBytes;
};

// the Rice parameter of the key gaps of a row: the keys of its partition per record, rounded down to a power of two
static unsigned int getRepKeyDiffLowBits(size_t partitionKeys, size_t recordCount) {
    return getBitWidth(std::max<size_t>(1, partitionKeys / std::max<size_t>(1, recordCount))) - 1;
}

// the targets of the records of a rep that a thread merges: up to repTargetCapacity in repTargets, more from the records by targetQueue
struct RepTargetMerge {
    DBKeyType *repTargets;
    size_t repTargetCapacity;
    std::vector<size_t> recordTargets;
    TargetIdQueue targetQueue;
};

// the targets of the records [repStart, repEnd) of a rep that fit repTargets go back to the slots of the records sorted without repeats, so they follow each other in order
static void sortRepTargets(AssignedPairRecord *records, size_t repStart, size_t repEnd, DBKeyType *targets, RepTargetMerge &merge) {
    if (repEnd - repStart == 1) {
        return;
    }
    DBKeyType *repTargetsEnd = merge.repTargets;
    for (size_t record = repStart; record < repEnd; record++) {
        if (static_cast<size_t>(repTargetsEnd - merge.repTargets) + records[record].targetCount > merge.repTargetCapacity) {
            return;
        }
        repTargetsEnd = std::copy(targets + records[record].targetOffset, targets + records[record].targetOffset + records[record].targetCount, repTargetsEnd);
    }
    SORT_SERIAL(merge.repTargets, repTargetsEnd);
    repTargetsEnd = std::unique(merge.repTargets, repTargetsEnd);
    const DBKeyType *repTarget = merge.repTargets;
    for (size_t record = repStart; record < repEnd; record++) {
        records[record].targetCount = std::min<size_t>(records[record].targetCount, repTargetsEnd - repTarget);
        std::copy(repTarget, repTarget + records[record].targetCount, targets + records[record].targetOffset);
        repTarget += records[record].targetCount;
    }
}

// gives the targets of the records [repStart, repEnd) of a rep to addTarget in id order without repeats: one after the other if they fit repTargets, as a record and
// sortRepTargets keep them so, otherwise merged from the records
template <typename AddTarget>
static void mergeRepTargets(const AssignedPairRecord *records, size_t repStart, size_t repEnd, const DBKeyType *targets, RepTargetMerge &merge, AddTarget addTarget) {
    size_t repTargetCount = 0;
    for (size_t record = repStart; record < repEnd; record++) {
        repTargetCount += records[record].targetCount;
    }
    if (repEnd - repStart == 1 || repTargetCount <= merge.repTargetCapacity) {
        for (size_t record = repStart; record < repEnd; record++) {
            for (size_t target = 0; target < records[record].targetCount; target++) {
                addTarget(targets[records[record].targetOffset + target]);
            }
        }
        return;
    }
    std::vector<size_t> &recordTargets = merge.recordTargets;
    TargetIdQueue &targetQueue = merge.targetQueue;
    recordTargets.assign(repEnd - repStart, 0);
    for (size_t record = repStart; record < repEnd; record++) {
        targetQueue.push(FileTargetId(targets[records[record].targetOffset], record - repStart));
    }
    DBKeyType prevTarget = DB_KEY_INVALID;
    while (targetQueue.empty() == false) {
        const FileTargetId target = targetQueue.top();
        targetQueue.pop();
        if (target.targetId != prevTarget) {
            addTarget(target.targetId);
            prevTarget = target.targetId;
        }
        const AssignedPairRecord &record = records[repStart + target.record];
        recordTargets[target.record]++;
        if (recordTargets[target.record] < record.targetCount) {
            targetQueue.push(FileTargetId(targets[record.targetOffset + recordTargets[target.record]], target.record));
        }
    }
}

// the targets of a rep without repeats, and the bits of every gap between them, the width of its largest gap less one
static unsigned int getRepTargetGapBits(const AssignedPairRecord *records, size_t repStart, size_t repEnd, const DBKeyType *targets, RepTargetMerge &merge, size_t &repTargetCount) {
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

// writes the records, sorted by rep, as a row per partition at reserved bytes of the pair file, then the rows as a block;
// a thread writes through the first half of its write buffer and sorts the targets of a rep in the second half
static void writeAssignedPairs(const KmerExchange &exchange, AssignedPairFile &pairFile, AssignedPairRecord *records, size_t recordCount, DBKeyType *targets,
                               char *writeBuffers, int threads) {
    if (recordCount == 0) {
        return;
    }
    // the partitions are key ranges, so the records of a row follow each other
    const std::vector<size_t> &sendRangeKeyOffsets = exchange.sendRangeKeyOffsets;
    std::vector<size_t> rowRecordOffsets(1, 0);
    std::vector<size_t> rowFirstKeys;
    std::vector<size_t> rowKeys;
    std::vector<AssignedPairPartition> rows;
    // the first row of every send range, then the row count
    std::vector<size_t> sendRangeRows;
    while (rowRecordOffsets.back() < recordCount) {
        const size_t rep = records[rowRecordOffsets.back()].rep;
        const size_t sendRange = std::upper_bound(sendRangeKeyOffsets.begin(), sendRangeKeyOffsets.end(), rep) - sendRangeKeyOffsets.begin() - 1;
        const size_t partition = (rep - sendRangeKeyOffsets[sendRange]) * exchange.partitionsPerRange / (sendRangeKeyOffsets[sendRange + 1] - sendRangeKeyOffsets[sendRange]);
        rowFirstKeys.push_back(getPartitionFirstKey(exchange, sendRange, partition));
        rowKeys.push_back(getPartitionFirstKey(exchange, sendRange, partition + 1) - rowFirstKeys.back());
        rowRecordOffsets.push_back(std::partition_point(records + rowRecordOffsets.back(), records + recordCount,
                                                        [&](const AssignedPairRecord &record) { return record.rep < rowFirstKeys.back() + rowKeys.back(); }) - records);
        while (sendRangeRows.size() <= sendRange) {
            sendRangeRows.push_back(rows.size());
        }
        const AssignedPairPartition row = {{0, 0, 0}, partition, 0, 0};
        rows.push_back(row);
    }
    sendRangeRows.resize(exchange.sendRangeCount + 1, rows.size());
    const unsigned int targetGapBitsWidth = getBitWidth(exchange.globalIdBits);
    // the bytes of every code of a row; the sorted targets go back to the slots of the records, so the write takes them in order
#pragma omp parallel num_threads(threads)
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        RepTargetMerge merge;
        merge.repTargets = reinterpret_cast<DBKeyType *>(writeBuffers + thread_idx * KMER_FILE_BUFFER_SIZE + KMER_FILE_BUFFER_SIZE / 2);
        merge.repTargetCapacity = KMER_FILE_BUFFER_SIZE / 2 / sizeof(DBKeyType);
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
                sortRepTargets(records, repStart, repEnd, targets, merge);
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
    for (size_t sendRange = 0; sendRange < exchange.sendRangeCount; sendRange++) {
        for (size_t row = sendRangeRows[sendRange]; row < sendRangeRows[sendRange + 1]; row++) {
            rows[row].offsets[0] = pairFile.bytes;
            for (size_t code = 0; code < PAIR_CODE_COUNT; code++) {
                rows[row].offsets[code + 1] += rows[row].offsets[code];
            }
            pairFile.bytes = rows[row].offsets[PAIR_CODE_COUNT];
            pairFile.partitionBytes[sendRange * exchange.partitionsPerRange + rows[row].partition] += pairFile.bytes - rows[row].offsets[0];
        }
    }
    // a record: the gap to the key after the record before, the target count, the first target and the bits of the gaps to the other targets
#pragma omp parallel num_threads(threads)
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        char *writeBuffer = writeBuffers + thread_idx * KMER_FILE_BUFFER_SIZE;
        RepTargetMerge merge;
        merge.repTargets = reinterpret_cast<DBKeyType *>(writeBuffer + KMER_FILE_BUFFER_SIZE / 2);
        merge.repTargetCapacity = KMER_FILE_BUFFER_SIZE / 2 / sizeof(DBKeyType);
#pragma omp for schedule(dynamic, 1)
        for (size_t row = 0; row < rows.size(); row++) {
            KmerCodeWriter indexWrite = {fileno(pairFile.file), pairFile.name.c_str(), writeBuffer, KMER_FILE_BUFFER_SIZE / 4, 0, rows[row].offsets[PAIR_INDEX_CODES], 0, 0, pairFile.batchWriter};
            KmerCodeWriter targetWrite = {fileno(pairFile.file), pairFile.name.c_str(), writeBuffer + KMER_FILE_BUFFER_SIZE / 4, KMER_FILE_BUFFER_SIZE / 4, 0, rows[row].offsets[PAIR_TARGET_CODES],
                                          0, 0, pairFile.batchWriter};
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
            writeKmerCodeBuffer(indexWrite);
            writeKmerCodeBuffer(targetWrite);
        }
    }
    pairFile.batchWriter->write(fileno(pairFile.file), pairFile.name.c_str(), reinterpret_cast<const char *>(rows.data()), rows.size() * sizeof(AssignedPairPartition), pairFile.bytes);
    for (size_t sendRange = 0; sendRange <= exchange.sendRangeCount; sendRange++) {
        pairFile.blockRowStarts.push_back(pairFile.bytes + sendRangeRows[sendRange] * sizeof(AssignedPairPartition));
    }
    pairFile.bytes += rows.size() * sizeof(AssignedPairPartition);
}

// the output of a receiving node, kept over its passes and hash ranges and written once it is full: the threads take blocks of records from its start and of targets
// from its end, and the record blocks that they filled are moved together before the records are sorted
struct PairOutput {
    AssignedPairRecord *records;
    DBKeyType *targets;
    size_t bytes;
    size_t blockBytes;
    size_t recordsTaken;
    size_t targetsTaken;
    std::vector<std::pair<size_t, size_t> > recordBlocks;
    size_t flushes;
};

// the record block and the target block that a thread fills
struct PairOutputBlock {
    size_t recordStart;
    size_t recordPos;
    size_t recordEnd;
    size_t targetPos;
    size_t targetEnd;
};

// a thread keeps its blocks while they hold recordCount records and targetCount targets more, otherwise it takes new ones of blockBytes, or of the size it needs if
// these do not fit; false if the output is full
static bool takePairOutput(PairOutput &output, PairOutputBlock &block, size_t recordCount, size_t targetCount) {
    const bool hasRecords = (block.recordEnd - block.recordPos >= recordCount);
    const bool hasTargets = (block.targetEnd - block.targetPos >= targetCount);
    if (hasRecords && hasTargets) {
        return true;
    }
    bool isTaken = false;
#pragma omp critical(takePairOutput)
    {
        size_t newRecords = hasRecords ? 0 : std::max(recordCount, output.blockBytes / sizeof(AssignedPairRecord));
        size_t newTargets = hasTargets ? 0 : std::max(targetCount, output.blockBytes / sizeof(DBKeyType));
        if ((output.recordsTaken + newRecords) * sizeof(AssignedPairRecord) + (output.targetsTaken + newTargets) * sizeof(DBKeyType) > output.bytes) {
            newRecords = hasRecords ? 0 : recordCount;
            newTargets = hasTargets ? 0 : targetCount;
        }
        if ((output.recordsTaken + newRecords) * sizeof(AssignedPairRecord) + (output.targetsTaken + newTargets) * sizeof(DBKeyType) <= output.bytes) {
            if (newRecords > 0) {
                if (block.recordPos > block.recordStart) {
                    output.recordBlocks.push_back(std::make_pair(block.recordStart, block.recordPos));
                }
                block.recordStart = output.recordsTaken;
                block.recordPos = output.recordsTaken;
                output.recordsTaken += newRecords;
                block.recordEnd = output.recordsTaken;
            }
            if (newTargets > 0) {
                output.targetsTaken += newTargets;
                block.targetPos = output.bytes / sizeof(DBKeyType) - output.targetsTaken;
                block.targetEnd = block.targetPos + newTargets;
            }
            isTaken = true;
        }
    }
    return isTaken;
}

// writes the records of the output sorted by rep and empties the output and the blocks of every thread
static void flushPairOutput(const KmerExchange &exchange, AssignedPairFile &pairFile, PairOutput &output, std::vector<PairOutputBlock> &blocks, char *writeBuffers, int threads) {
    for (size_t thread = 0; thread < blocks.size(); thread++) {
        if (blocks[thread].recordPos > blocks[thread].recordStart) {
            output.recordBlocks.push_back(std::make_pair(blocks[thread].recordStart, blocks[thread].recordPos));
        }
        const PairOutputBlock emptyBlock = {0, 0, 0, 0, 0};
        blocks[thread] = emptyBlock;
    }
    SORT_SERIAL(output.recordBlocks.begin(), output.recordBlocks.end());
    size_t recordCount = 0;
    for (size_t i = 0; i < output.recordBlocks.size(); i++) {
        memmove(output.records + recordCount, output.records + output.recordBlocks[i].first, (output.recordBlocks[i].second - output.recordBlocks[i].first) * sizeof(AssignedPairRecord));
        recordCount += output.recordBlocks[i].second - output.recordBlocks[i].first;
    }
    if (threads > 1) {
        SORT_PARALLEL(output.records, output.records + recordCount, AssignedPairRecord::compareByRep);
    } else {
        SORT_SERIAL(output.records, output.records + recordCount, AssignedPairRecord::compareByRep);
    }
    writeAssignedPairs(exchange, pairFile, output.records, recordCount, output.targets, writeBuffers, threads);
    output.flushes += (recordCount > 0);
    output.recordsTaken = 0;
    output.targetsTaken = 0;
    output.recordBlocks.clear();
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

// writes the entry of a key, the key and then its targets up to keyTargetsEnd or DB_KEY_INVALID, and its index cell
static void writePrefEntry(size_t key, const DBKeyType *keyTargets, const DBKeyType *keyTargetsEnd, KmerCodeWriter &dataWrite, KmerCodeWriter &indexWrite) {
    const uint64_t indexCell[2] = {static_cast<uint64_t>(key), static_cast<uint64_t>(dataWrite.writePos + dataWrite.bufferPos)};
    putBytes(indexWrite, reinterpret_cast<const char *>(indexCell), sizeof(indexCell));
    putPrefKey(dataWrite, key);
    for (DBKeyType prevKey = 0; keyTargets < keyTargetsEnd && *keyTargets != DB_KEY_INVALID; prevKey = *keyTargets, keyTargets++) {
        putPrefKey(dataWrite, *keyTargets - prevKey);
    }
    putBytes(dataWrite, "", 1);
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

static inline void readNextTarget(PairRowRead &read) {
    read.target += getBits(read.codeReads[PAIR_TARGET_CODES], read.targetGapBits) + 1;
    read.targetsLeft--;
}

// gives the targets of key in the rows whose record has it to addTarget, merged by id without repeats, then queues the next records of these rows
template <typename AddTarget>
static void mergeKeyTargets(std::vector<PairRowRead> &reads, RepKeyQueue &repKeyQueue, TargetIdQueue &targetQueue, std::vector<unsigned int> &keyRows, size_t key,
                            unsigned int globalIdBits, AddTarget addTarget) {
    keyRows.clear();
    while (repKeyQueue.empty() == false && repKeyQueue.top().repKey == key) {
        keyRows.push_back(repKeyQueue.top().row);
        repKeyQueue.pop();
    }
    for (size_t row = 0; row < keyRows.size(); row++) {
        targetQueue.push(FileTargetId(reads[keyRows[row]].target, keyRows[row]));
    }
    DBKeyType prevTarget = DB_KEY_INVALID;
    while (targetQueue.empty() == false) {
        const FileTargetId target = targetQueue.top();
        targetQueue.pop();
        if (target.targetId != prevTarget) {
            addTarget(target.targetId);
            prevTarget = target.targetId;
        }
        PairRowRead &read = reads[target.record];
        if (read.targetsLeft > 0) {
            readNextTarget(read);
            targetQueue.push(FileTargetId(read.target, target.record));
        }
    }
    for (size_t row = 0; row < keyRows.size(); row++) {
        PairRowRead &read = reads[keyRows[row]];
        if (read.recordsLeft > 0) {
            readPairRecord(read, globalIdBits);
            repKeyQueue.push(FileRepKey(read.repKey, keyRows[row]));
        }
    }
}

// the readers of the rows of a partition at their start, with their codes in codeBuffer or, with windowBytes, in windows from codeBuffer on
static void getPartitionRowReads(const std::vector<ReceivedFile> &receivedPairFiles, const std::vector<std::vector<AssignedPairPartition> > &pairFileRows, const PassRow *partitionPassRows,
                                 size_t partitionRowCount, size_t partitionKeys, char *codeBuffer, size_t windowBytes, std::vector<PairRowRead> &rowReads) {
    rowReads.assign(partitionRowCount, PairRowRead());
    for (size_t passRow = 0; passRow < partitionRowCount; passRow++) {
        const PassRow &partitionPassRow = partitionPassRows[passRow];
        const AssignedPairPartition &pairRow = pairFileRows[partitionPassRow.file][partitionPassRow.row];
        PairRowRead &read = rowReads[passRow];
        for (size_t code = 0; code < PAIR_CODE_COUNT; code++) {
            read.codeReads[code] = getRowCodeReader(receivedPairFiles[partitionPassRow.file], pairRow, partitionPassRow.bufferOffset, code, codeBuffer,
                                                    codeBuffer + (passRow * PAIR_CODE_COUNT + code) * windowBytes, windowBytes);
        }
        read.repKeyDiffLowBits = getRepKeyDiffLowBits(partitionKeys, pairRow.recordCount);
        read.recordsLeft = pairRow.recordCount;
    }
}

// with threads: the row readers of a partition, the target count of every key, key chunks of equal work and where every row starts them
static void countPartitionTargets(const KmerExchange &exchange, const std::vector<ReceivedFile> &receivedPairFiles, const std::vector<std::vector<AssignedPairPartition> > &pairFileRows,
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
    // chunks of keys with about the same keys and targets: a chunk starts at the first key whose keys and targets before it reach a multiple of keyChunkWeight
    const size_t keyChunkWeight = std::max<size_t>(1, (partitionTargets + partitionKeys + CHUNKS_PER_THREAD * threads - 1) / (CHUNKS_PER_THREAD * threads));
    merge.keyChunkOffsets.assign(1, 0);
    for (size_t chunk = 1; merge.keyChunkOffsets.back() < partitionKeys; chunk++) {
        size_t low = merge.keyChunkOffsets.back();
        size_t high = partitionKeys;
        while (low < high) {
            const size_t mid = low + (high - low) / 2;
            if (keyOffsets[mid] + mid < chunk * keyChunkWeight) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        if (low > merge.keyChunkOffsets.back()) {
            merge.keyChunkOffsets.push_back(low);
        }
    }
    // every row starts a key chunk at its first record with a key from the chunk on
    const size_t keyChunks = merge.keyChunkOffsets.size() - 1;
    merge.chunkStarts.resize(keyChunks * rowReads.size());
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (size_t row = 0; row < rowReads.size(); row++) {
        PairRowRead read = rowReads[row];
        PairChunkStart start = {read.recordsLeft, 0, {0, 0}};
        merge.chunkStarts[row] = start;
        for (size_t chunk = 1; chunk < keyChunks;) {
            start.recordsLeft = read.recordsLeft;
            start.nextRepKey = read.nextRepKey;
            start.bitPos[PAIR_INDEX_CODES] = getKmerCodeBitPos(read.codeReads[PAIR_INDEX_CODES]);
            if (read.recordsLeft > 0) {
                readPairRecord(read, exchange.globalIdBits);
            }
            for (; chunk < keyChunks && (start.recordsLeft == 0 || read.repKey >= merge.keyChunkOffsets[chunk]); chunk++) {
                merge.chunkStarts[chunk * rowReads.size() + row] = start;
            }
            start.bitPos[PAIR_TARGET_CODES] += read.targetsLeft * read.targetGapBits;
        }
    }
}

// with threads: merges the targets of every key from the rows by id without repeats, a key chunk at a time, and finds where the entries and index cells of every key chunk start
static void mergePartitionTargets(const KmerExchange &exchange, DBReader<DBKeyType> &seqDbr, size_t partitionFirstKey, PartitionMerge &merge, DBKeyType *targets, int threads) {
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    const std::vector<size_t> &keyOffsets = merge.keyOffsets;
    merge.targets = targets;
    const size_t keyChunks = merge.keyChunkOffsets.size() - 1;
    const size_t rowCount = merge.rowReads.size();
    merge.keyChunkDataOffsets.assign(keyChunks + 1, 0);
    merge.keyChunkEntryOffsets.assign(keyChunks + 1, 0);
    merge.keyChunkCostOffsets.assign(keyChunks + 1, 0);
    merge.keyChunkCuts.assign(keyChunks, ChunkCuts());
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (size_t keyChunk = 0; keyChunk < keyChunks; keyChunk++) {
        std::vector<PairRowRead> reads(merge.rowReads);
        RepKeyQueue repKeyQueue;
        TargetIdQueue targetQueue;
        std::vector<unsigned int> keyRows;
        for (size_t row = 0; row < rowCount; row++) {
            const PairChunkStart &start = merge.chunkStarts[keyChunk * rowCount + row];
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
                repKeyQueue.push(FileRepKey(read.repKey, row));
            }
        }
        for (size_t key = merge.keyChunkOffsets[keyChunk]; key < merge.keyChunkOffsets[keyChunk + 1]; key++) {
            DBKeyType *keyTargets = targets + keyOffsets[key];
            size_t keyTargetCount = 0;
            mergeKeyTargets(reads, repKeyQueue, targetQueue, keyRows, key, exchange.globalIdBits, [&](DBKeyType target) { keyTargets[keyTargetCount++] = target; });
            if (keyOffsets[key] + keyTargetCount < keyOffsets[key + 1]) {
                keyTargets[keyTargetCount] = DB_KEY_INVALID;
            }
            if (isRepresentative(partitionFirstKey + key) == false) {
                continue;
            }
            size_t entryBytes = getPrefKeyBytes(partitionFirstKey + key) + 1;
            size_t targetRange = SIZE_MAX;
            size_t targetResidues = 0;
            for (size_t target = 0; target < keyTargetCount; target++) {
                entryBytes += getPrefKeyBytes(keyTargets[target] - ((target > 0) ? keyTargets[target - 1] : 0));
                targetResidues += seqDbr.getSeqLenFromRange(targetRange, keyTargets[target]);
            }
            merge.keyChunkDataOffsets[keyChunk + 1] += entryBytes;
            merge.keyChunkEntryOffsets[keyChunk + 1]++;
            if (keyTargetCount > 0) {
                merge.keyChunkCostOffsets[keyChunk + 1] += (index[seqDbr.findIdRange(partitionFirstKey + key)].length - 2) * targetResidues;
                const PartitionCut cut = { key + 1, merge.keyChunkDataOffsets[keyChunk + 1], merge.keyChunkCostOffsets[keyChunk + 1] };
                merge.keyChunkCuts[keyChunk].add(cut);
            }
        }
    }
    std::vector<PairRowRead>().swap(merge.rowReads);
    std::vector<PairChunkStart>().swap(merge.chunkStarts);
    for (size_t keyChunk = 0; keyChunk < keyChunks; keyChunk++) {
        merge.keyChunkDataOffsets[keyChunk + 1] += merge.keyChunkDataOffsets[keyChunk];
        merge.keyChunkEntryOffsets[keyChunk + 1] += merge.keyChunkEntryOffsets[keyChunk];
        merge.keyChunkCostOffsets[keyChunk + 1] += merge.keyChunkCostOffsets[keyChunk];
    }
    merge.cost = merge.keyChunkCostOffsets[keyChunks];
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

// with threads: writes the entries of a merged partition at dataOffset and indexOffset of the pref files, a key chunk at a time
static void writePartition(const PrefFiles &prefFiles, size_t partitionFirstKey, const PartitionMerge &merge, size_t dataOffset, size_t indexOffset, size_t firstWriteBuffer, int threads) {
    const size_t keyChunks = merge.keyChunkOffsets.size() - 1;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (size_t keyChunk = 0; keyChunk < keyChunks; keyChunk++) {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        char *writeBuffer = prefFiles.writeBuffers + 2 * (firstWriteBuffer + thread_idx) * KMER_FILE_BUFFER_SIZE;
        KmerCodeWriter dataWrite = {fileno(prefFiles.dataFile), prefFiles.dataName.c_str(), writeBuffer, KMER_FILE_BUFFER_SIZE, 0, dataOffset + merge.keyChunkDataOffsets[keyChunk], 0, 0, prefFiles.batchWriter};
        KmerCodeWriter indexWrite = {fileno(prefFiles.indexFile), prefFiles.indexName.c_str(), writeBuffer + KMER_FILE_BUFFER_SIZE, KMER_FILE_BUFFER_SIZE, 0,
                                     indexOffset + merge.keyChunkEntryOffsets[keyChunk] * 2 * sizeof(uint64_t), 0, 0, prefFiles.batchWriter};
        for (size_t key = merge.keyChunkOffsets[keyChunk]; key < merge.keyChunkOffsets[keyChunk + 1]; key++) {
            if (isRepresentative(partitionFirstKey + key)) {
                writePrefEntry(partitionFirstKey + key, merge.targets + merge.keyOffsets[key], merge.targets + merge.keyOffsets[key + 1], dataWrite, indexWrite);
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
    std::vector<std::vector<KmerHashSubRange> > hashRangeRows;
    std::vector<size_t> hashRanges;
};

// --createdb-mode 3 sending: the nodes take the send ranges one at a time, the k-mers of a send range in loads, written to <pref>.kmers.<node>
template <typename T, bool includeAdjacency, typename L>
static size_t sendKmers(Parameters &par, DBReader<DBKeyType> &seqDbr, BaseMatrix *subMat, const KmerExchange &exchange, MMseqsNode::WorkQueue &sendQueue,
                        size_t representativeBitsMemory, std::vector<size_t> &sendRanges, std::string &sendValues, OwnKmers &ownKmers) {
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    const float kmersPerSequenceScale = par.kmersPerSequenceScale.values.aminoacid();
    const size_t readChunkBytes = std::min(MAX_INFLIGHT_READ_SIZE, exchange.loadMemoryLimit / 4);
    // a load spans so few keys that their ids fit next to the whole sequence bit, the hash and the adjKey
    const size_t loadKeyLimit = (1ULL << (128 - 1 - 64 - exchange.adjKeyBits)) - 1;
    const std::string kmerFile = par.db2 + ".kmers." + SSTR(MMseqsNode::rank);
    FILE *kmerFileHandle = FileUtil::openAndDelete(kmerFile.c_str(), "w");
    size_t kmerFileBytes = 0;
    // the rows of every load are written after its codes as a block: where the rows of every hash range start in it, then its end, block after block
    std::vector<size_t> blockRowStarts;
    // the rows of a load, hash range after hash range
    std::vector<KmerHashSubRange> loadRows(exchange.hashRangeCount * exchange.rowsPerHashRange);
    std::vector<size_t> hashSubRangeOffsets(exchange.hashSubRangeCount + 1);
    std::vector<size_t> wholeSequenceOffsets(exchange.hashSubRangeCount + 1);
    char *kmerWriteBuffer = new(std::nothrow) char[par.threads * KMER_FILE_BUFFER_SIZE];
    Util::checkAllocation(kmerWriteBuffer, "Can not allocate kmerWriteBuffer memory in sendKmers");
    BatchWriter batchWriter(exchange.writeBehindMemory, par.threads, fileno(kmerFileHandle));
    // the k-mers of the largest send range so far, kept for the next ones so their pages are not mapped again, while they fit with the group buffers
    KmerPositionBit *hashSeqPair = NULL;
    size_t hashSeqPairCapacity = 0;
    std::vector<char> isOwnLoadHashRange(exchange.hashRangeCount);
    for (size_t sendRange = sendQueue.claimNext(); sendRange < exchange.sendRangeCount; sendRange = sendQueue.claimNext()) {
        sendRanges.push_back(sendRange);
        setSendRange(par, exchange, sendRange);
        const size_t dbFrom = exchange.sendRangeKeyOffsets[sendRange];
        const size_t dbSize = exchange.sendRangeKeyOffsets[sendRange + 1] - dbFrom;
        const size_t firstRange = (dbSize > 0) ? seqDbr.findIdRange(dbFrom) : seqDbr.getIndexSize();
        const auto countRepresentatives = [&](size_t from, size_t to) {
            return (representativeBits == NULL) ? to - from : Util::countSetBits(representativeBits, from - dbFrom, to - dbFrom);
        };
        // the k-mers of a load and a group buffer per thread fit the memory beside the own rows, with fewer k-mers per load if needed
        const size_t loadMemoryLimit = exchange.loadMemoryLimit - ownKmers.bytes;
        size_t kmerCapacity = loadMemoryLimit / sizeof(KmerPositionBit);
        std::vector<size_t> loadOffsets;
        size_t maxLoadKmers = 0;
        size_t maxLoadKeys = 0;
        size_t groupBufferWords = 0;
        size_t fixedBytes = 0;
        while (true) {
            loadOffsets.assign(1, dbFrom);
            size_t loadKmers = 0;
            maxLoadKmers = 0;
            for (size_t range = firstRange; range < seqDbr.getIndexSize() && index[range].id < dbFrom + dbSize; range++) {
                const size_t rangeFrom = std::max(static_cast<size_t>(index[range].id), dbFrom);
                const size_t rangeTo = std::min((range + 1 < seqDbr.getIndexSize()) ? static_cast<size_t>(index[range + 1].id) : seqDbr.getSize(), dbFrom + dbSize);
                const size_t seqKmers = computeSeqKmerCount(static_cast<int>(index[range].length - 2), par.kmerSize, par.kmersPerSequence, kmersPerSequenceScale);
                if (seqKmers > kmerCapacity) {
                    Debug(Debug::ERROR) << "Not enough memory to extract the k-mers of a sequence of length " << index[range].length - 2 << ". Raise --split-memory-limit.\n";
                    EXIT(EXIT_FAILURE);
                }
                size_t rangeRepresentatives = countRepresentatives(rangeFrom, rangeTo);
                // the next load starts at the first representative that does not fit, or where its keys would need more id bits
                size_t cutKey = rangeFrom;
                while (loadKmers + rangeRepresentatives * seqKmers > kmerCapacity || rangeTo - loadOffsets.back() > loadKeyLimit) {
                    size_t fitCount = std::min(rangeRepresentatives, (kmerCapacity - loadKmers) / seqKmers);
                    size_t nextLoadKey = rangeTo;
                    if (fitCount < rangeRepresentatives && representativeBits == NULL) {
                        nextLoadKey = cutKey + fitCount;
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
                        nextLoadKey = dbFrom + bit / 64 * 64 + __builtin_ctzll(word);
                    }
                    if (nextLoadKey - loadOffsets.back() > loadKeyLimit) {
                        nextLoadKey = loadOffsets.back() + loadKeyLimit;
                        fitCount = countRepresentatives(cutKey, nextLoadKey);
                    }
                    maxLoadKmers = std::max(maxLoadKmers, loadKmers + fitCount * seqKmers);
                    rangeRepresentatives -= fitCount;
                    cutKey = nextLoadKey;
                    loadOffsets.push_back(cutKey);
                    loadKmers = 0;
                }
                loadKmers += rangeRepresentatives * seqKmers;
                maxLoadKmers = std::max(maxLoadKmers, loadKmers);
            }
            loadOffsets.push_back(dbFrom + dbSize);
            maxLoadKeys = 0;
            for (size_t load = 0; load + 1 < loadOffsets.size(); load++) {
                maxLoadKeys = std::max(maxLoadKeys, loadOffsets[load + 1] - loadOffsets[load]);
            }
            // a group buffer holds the ids of a group, or for a bigger group a bit per key of the load and the count of set bits before every 512 of them
            groupBufferWords = (maxLoadKeys + 63) / 64 + (maxLoadKeys + 511) / 512;
            fixedBytes = par.threads * groupBufferWords * sizeof(uint64_t);
            if (fixedBytes + maxLoadKmers * sizeof(KmerPositionBit) <= loadMemoryLimit) {
                break;
            }
            if (maxLoadKmers == 0) {
                Debug(Debug::ERROR) << "Not enough memory to run the kmermatcher. Memory limit: " << exchange.memoryLimit
                                    << " bytes, the group buffers need: " << representativeBitsMemory + fixedBytes << " bytes.\n"
                                    << "Raise --split-memory-limit.\n";
                EXIT(EXIT_FAILURE);
            }
            // fewer k-mers per load take fewer keys per group buffer
            kmerCapacity = std::min(maxLoadKmers - 1, (fixedBytes < loadMemoryLimit) ? (loadMemoryLimit - fixedBytes) / sizeof(KmerPositionBit) : maxLoadKmers / 2);
        }
        const size_t loadCount = loadOffsets.size() - 1;

        const unsigned int idBits = getBitWidth(maxLoadKeys);
        KmerPositionBit::idShift = 0;
        KmerPositionBit::seqLenShift = idBits;
        KmerPositionBit::adjKeyShift = idBits;
        KmerPositionBit::kmerShift = idBits + exchange.adjKeyBits;
        KmerPositionBit::maxSeqLen = 0;

        if (maxLoadKmers > hashSeqPairCapacity || fixedBytes + hashSeqPairCapacity * sizeof(KmerPositionBit) > loadMemoryLimit) {
            Util::releasePages(hashSeqPair, hashSeqPairCapacity * sizeof(KmerPositionBit));
            delete[] hashSeqPair;
            hashSeqPairCapacity = maxLoadKmers;
            hashSeqPair = new(std::nothrow) KmerPositionBit[hashSeqPairCapacity];
            Util::checkAllocation(hashSeqPair, "Can not allocate hashSeqPair memory in sendKmers");
        }
        uint64_t *groupBuffers = new(std::nothrow) uint64_t[par.threads * groupBufferWords];
        Util::checkAllocation(groupBuffers, "Can not allocate groupBuffers memory in sendKmers");
        for (size_t load = 0; load < loadCount; load++) {
            std::vector<size_t> readChunkOffsets(1, loadOffsets[load]);
            size_t readChunkUsedBytes = 0;
            const size_t loadFirstRange = (loadOffsets[load] < loadOffsets[load + 1]) ? seqDbr.findIdRange(loadOffsets[load]) : seqDbr.getIndexSize();
            for (size_t range = loadFirstRange; range < seqDbr.getIndexSize() && index[range].id < loadOffsets[load + 1]; range++) {
                const size_t rangeFrom = std::max(static_cast<size_t>(index[range].id), loadOffsets[load]);
                const size_t rangeTo = std::min((range + 1 < seqDbr.getIndexSize()) ? static_cast<size_t>(index[range + 1].id) : seqDbr.getSize(), loadOffsets[load + 1]);
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
            readChunkOffsets.push_back(loadOffsets[load + 1]);
            const size_t kmerCount = fillKmerPositionArray<Parameters::DBTYPE_AMINO_ACIDS, T, includeAdjacency, L>(
                hashSeqPair, maxLoadKmers, seqDbr, par, subMat, true, 0, SIZE_T_MAX, NULL, readChunkOffsets).first;

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
            const size_t loadWords = (loadOffsets[load + 1] - loadOffsets[load] + 63) / 64;
            // the code bytes of every row, as they are written below
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
            for (size_t row = 0; row < loadRows.size(); row++) {
                unsigned int thread_idx = 0;
#ifdef OPENMP
                thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
                uint64_t *groupBuffer = groupBuffers + thread_idx * groupBufferWords;
                uint64_t *setBitCounts = groupBuffer + (maxLoadKeys + 63) / 64;
                const bool isWholeSequence = (row % exchange.rowsPerHashRange >= exchange.hashSubRangesPerRange);
                size_t hashSubRange = 0;
                size_t hashSubRangeEnd = 0;
                getRowHashSubRanges(exchange, row / exchange.rowsPerHashRange, row % exchange.rowsPerHashRange, hashSubRange, hashSubRangeEnd);
                const size_t from = (isWholeSequence ? wholeSequenceOffsets : hashSubRangeOffsets)[hashSubRange];
                const size_t to = (isWholeSequence ? wholeSequenceOffsets : hashSubRangeOffsets)[hashSubRangeEnd];
                const unsigned int hashShift = isWholeSequence ? 0 : exchange.kmerHashShift;
                KmerHashSubRange &hashSubRangeRow = loadRows[row];
                const KmerHashSubRange emptyRow = {static_cast<DBKeyType>(loadOffsets[load]), idBits, {0, 0, 0, 0}, 0};
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
                    const size_t memberCount = getGroupIds(hashSeqPair, groupStart, groupEnd, groupBuffer, groupBufferWords, setBitCounts, loadWords);
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
            // the own hash ranges keep their rows of the load in memory up to the first that does not fit beside the load
            const size_t ownLoadLimit = std::min(exchange.loadMemoryLimit - ownKmers.bytes - fixedBytes - hashSeqPairCapacity * sizeof(KmerPositionBit),
                                                 ownKmers.limit - std::min(ownKmers.limit, ownKmers.bytes));
            size_t ownLoadBytes = 0;
            isOwnLoadHashRange.assign(exchange.hashRangeCount, false);
            for (size_t i = 0; i < ownKmers.hashRanges.size(); i++) {
                const size_t hashRange = ownKmers.hashRanges[i];
                size_t hashRangeBytes = 0;
                for (size_t row = hashRange * exchange.rowsPerHashRange; row < (hashRange + 1) * exchange.rowsPerHashRange; row++) {
                    hashRangeBytes += loadRows[row].offsets[KMER_CODE_COUNT];
                }
                if (ownLoadBytes + hashRangeBytes > ownLoadLimit) {
                    break;
                }
                isOwnLoadHashRange[hashRange] = true;
                ownLoadBytes += hashRangeBytes;
            }
            // the rows of the hash ranges follow each other in the k-mer file or in memory
            for (size_t row = 0; row < loadRows.size(); row++) {
                size_t &bytes = isOwnLoadHashRange[row / exchange.rowsPerHashRange] ? ownKmers.bytes : kmerFileBytes;
                for (size_t code = 0; code <= KMER_CODE_COUNT; code++) {
                    loadRows[row].offsets[code] += bytes;
                }
                bytes = loadRows[row].offsets[KMER_CODE_COUNT];
            }
            if (ownLoadBytes > 0) {
                ownKmers.data = static_cast<char *>(realloc(ownKmers.data, ownKmers.bytes));
                Util::checkAllocation(ownKmers.data, "Can not allocate ownKmers memory in sendKmers");
            }
            // a row writes its codes in one pass over its groups: the ids of the members, the rep candidates and the index, each at its offset
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
            for (size_t row = 0; row < loadRows.size(); row++) {
                unsigned int thread_idx = 0;
#ifdef OPENMP
                thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
                uint64_t *groupBuffer = groupBuffers + thread_idx * groupBufferWords;
                uint64_t *setBitCounts = groupBuffer + (maxLoadKeys + 63) / 64;
                const bool isWholeSequence = (row % exchange.rowsPerHashRange >= exchange.hashSubRangesPerRange);
                size_t hashSubRange = 0;
                size_t hashSubRangeEnd = 0;
                getRowHashSubRanges(exchange, row / exchange.rowsPerHashRange, row % exchange.rowsPerHashRange, hashSubRange, hashSubRangeEnd);
                const size_t from = (isWholeSequence ? wholeSequenceOffsets : hashSubRangeOffsets)[hashSubRange];
                const size_t to = (isWholeSequence ? wholeSequenceOffsets : hashSubRangeOffsets)[hashSubRangeEnd];
                const unsigned int hashShift = isWholeSequence ? 0 : exchange.kmerHashShift;
                const KmerHashSubRange &hashSubRangeRow = loadRows[row];
                const size_t codeBufferSize = KMER_FILE_BUFFER_SIZE / KMER_CODE_COUNT;
                KmerCodeWriter codeWrites[KMER_CODE_COUNT];
                for (size_t code = 0; code < KMER_CODE_COUNT; code++) {
                    const size_t writePos = hashSubRangeRow.offsets[code];
                    codeWrites[code] = isOwnLoadHashRange[row / exchange.rowsPerHashRange]
                        ? KmerCodeWriter{-1, NULL, ownKmers.data + writePos, ownKmers.bytes - writePos, 0, writePos, 0, 0, NULL}
                        : KmerCodeWriter{fileno(kmerFileHandle), kmerFile.c_str(), kmerWriteBuffer + thread_idx * KMER_FILE_BUFFER_SIZE + code * codeBufferSize, codeBufferSize, 0, writePos, 0, 0, &batchWriter};
                }
                const unsigned int hashDiffLowBits = getHashDiffLowBits(hashSubRange, hashSubRangeEnd, exchange.hashSubRangeCount, hashShift, hashSubRangeRow.groupCount);
                uint64_t prevHash = static_cast<uint64_t>(getHashSubRangeStart(hashSubRange, exchange.hashSubRangeCount, hashShift));
                for (size_t groupStart = from, groupEnd = from; groupStart < to; groupStart = groupEnd) {
                    const uint64_t hash = static_cast<uint64_t>(hashSeqPair[groupStart].getKmer()) >> hashShift;
                    size_t repCandidateCount = 1;
                    for (groupEnd = groupStart + 1; groupEnd < to && hashSeqPair[groupEnd].getKmer() == hashSeqPair[groupStart].getKmer(); groupEnd++) {
                        repCandidateCount += (hashSeqPair[groupEnd].getAdjKey() != hashSeqPair[groupEnd - 1].getAdjKey());
                    }
                    const size_t memberCount = getGroupIds(hashSeqPair, groupStart, groupEnd, groupBuffer, groupBufferWords, setBitCounts, loadWords);
                    if (groupEnd - groupStart <= groupBufferWords) {
                        for (size_t member = 0; member < memberCount; member++) {
                            putBits(codeWrites[KMER_MEMBER_CODES], groupBuffer[member], idBits);
                        }
                    } else {
                        for (size_t word = 0; word < loadWords; word++) {
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
                        putBits(codeWrites[KMER_REP_CANDIDATE_CODES], packAdjKey(exchange, hashSeqPair[pos].getAdjKey()), exchange.adjKeyCodeBits);
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
                blockRowStarts.push_back(kmerFileBytes);
                const KmerHashSubRange *hashRangeRows = loadRows.data() + hashRange * exchange.rowsPerHashRange;
                if (isOwnLoadHashRange[hashRange]) {
                    ownKmers.hashRangeRows[hashRange].insert(ownKmers.hashRangeRows[hashRange].end(), hashRangeRows, hashRangeRows + exchange.rowsPerHashRange);
                    continue;
                }
                batchWriter.write(fileno(kmerFileHandle), kmerFile.c_str(), reinterpret_cast<const char *>(hashRangeRows), exchange.rowsPerHashRange * sizeof(KmerHashSubRange), kmerFileBytes);
                kmerFileBytes += exchange.rowsPerHashRange * sizeof(KmerHashSubRange);
            }
            blockRowStarts.push_back(kmerFileBytes);
        }
        delete[] groupBuffers;
    }
    seqDbr.releaseDataPages();
    batchWriter.waitWrites();
    MMseqsNode::addPhaseSeconds(sendValues, "waiting to write k-mers", batchWriter.getWaitSeconds() / par.threads);
    delete[] kmerWriteBuffer;
    Util::releasePages(hashSeqPair, hashSeqPairCapacity * sizeof(KmerPositionBit));
    delete[] hashSeqPair;
    writeRowTable(kmerFileHandle, kmerFile, exchange.hashRangeCount, blockRowStarts, std::vector<size_t>(), kmerFileBytes);
    return kmerFileBytes;
}

// a hash range that a node receives: its rows of every k-mer file, the reads and bytes of every hash sub range, its passes with the bytes they read and receive,
// and their buffers
struct ReceivedHashRange {
    size_t hashRange;
    std::vector<std::vector<KmerHashSubRange> > fileRows;
    std::vector<size_t> hashSubRangeReads;
    std::vector<size_t> hashSubRangeBytes;
    std::vector<size_t> passHashSubRangeOffsets;
    std::vector<size_t> passBytes;
    std::vector<size_t> passReceivedBytes;
    size_t maxChunkStarts;
    size_t codeBufferBytes;
};

// a chunk of a receiving pass is done, or stops at a group whose records do not fit the output that is full, or that do not fit the empty output either
enum { CHUNK_DONE, CHUNK_OUTPUT_FULL, CHUNK_GROUP_TOO_BIG };

// --createdb-mode 3 receiving: the nodes take the hash ranges one at a time, in passes, and write the pairs of the chosen reps to <pref>.pairs.<node>
static size_t receiveKmers(Parameters &par, DBReader<DBKeyType> &seqDbr, const KmerExchange &exchange, MMseqsNode::WorkQueue &hashQueue, std::vector<size_t> &hashRanges,
                           std::string &receiveValues, size_t &pairBytes, const OwnKmers &ownKmers) {
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();

    AssignedPairFile pairFile;
    pairFile.name = par.db2 + ".pairs." + SSTR(MMseqsNode::rank);
    pairFile.file = FileUtil::openAndDelete(pairFile.name.c_str(), "w+");
    BatchWriter batchWriter(exchange.writeBehindMemory, par.threads, fileno(pairFile.file));
    pairFile.batchWriter = &batchWriter;
    pairFile.bytes = 0;
    pairFile.partitionBytes.assign(exchange.sendRangeCount * exchange.partitionsPerRange, 0);
    char *pairWriteBuffer = new(std::nothrow) char[par.threads * KMER_FILE_BUFFER_SIZE];
    Util::checkAllocation(pairWriteBuffer, "Can not allocate pairWriteBuffer memory in receiveKmers");
    // the k-mer file of every sender, open for the whole phase, then the own rows in memory
    std::vector<ReceivedFile> receivedFiles(MMseqsNode::count + 1);
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
    for (size_t sender = 0; sender < MMseqsNode::count; sender++) {
        receivedFiles[sender].fileName = par.db2 + ".kmers." + SSTR(sender);
        openReceivedFile(receivedFiles[sender], exchange.hashRangeCount, 0);
    }
    receivedFiles[MMseqsNode::count].data = ownKmers.data;
    size_t receivedBlocks = 0;
    for (size_t sender = 0; sender < MMseqsNode::count; sender++) {
        receivedBlocks += receivedFiles[sender].blockCount;
    }

    std::vector<KmerChunkStart> chunkStarts;
    // a pass and the prefetched next one fit their limit beside the rows of a flush, the own rows and the rows of every load of a hash range and the next one;
    // a bigger hash sub range is a pass that reads its codes in windows
    const size_t receivedRowsMemory = 2 * receivedBlocks * exchange.rowsPerHashRange * sizeof(KmerHashSubRange);
    const size_t passLimit = getPassLimit(exchange.loadMemoryLimit - std::min(exchange.loadMemoryLimit, exchange.pairRowsMemory + ownKmers.bytes + receivedRowsMemory));
    // the records and targets of the passes until the output is full, written then and at the end
    const size_t outputBytes = passLimit;
    // a thread that is alone with a group too big for the output takes half of it for records and half for targets
    const size_t halfRecords = outputBytes / 2 / sizeof(AssignedPairRecord);
    const size_t halfTargets = (outputBytes - halfRecords * sizeof(AssignedPairRecord)) / sizeof(DBKeyType);
    if (halfRecords == 0 || halfTargets == 0) {
        Debug(Debug::ERROR) << "Not enough memory to receive the k-mers. Raise --split-memory-limit.\n";
        EXIT(EXIT_FAILURE);
    }
    PairOutput output;
    output.bytes = outputBytes;
    output.blockBytes = std::min(KMER_FILE_BUFFER_SIZE, outputBytes / (OUTPUT_BLOCKS_PER_THREAD * par.threads));
    output.recordsTaken = 0;
    output.targetsTaken = 0;
    output.flushes = 0;
    char *outputBuffer = new(std::nothrow) char[outputBytes];
    Util::checkAllocation(outputBuffer, "Can not allocate outputBuffer memory in receiveKmers");
    output.records = reinterpret_cast<AssignedPairRecord *>(outputBuffer);
    output.targets = reinterpret_cast<DBKeyType *>(outputBuffer);
    const PairOutputBlock emptyBlock = {0, 0, 0, 0, 0};
    std::vector<PairOutputBlock> outputBlocks(par.threads, emptyBlock);
    // the groups and their k-mer bytes, and those of the groups with one member that give no pairs
    size_t receivedGroups = 0;
    size_t oneMemberGroups = 0;
    size_t oneMemberGroupBits = 0;
    size_t repCandidateBytes = 0;
    size_t windowedPasses = 0;
    size_t aloneChunks = 0;
    size_t punchedRanges = 0;
    // the buffer of the largest hash range so far, kept for the next ones so their pages are not mapped again
    char *passBuffer = NULL;
    size_t codeBufferBytes = 0;
    BatchReader *batchReader = NULL;
    // the rows of a pass with groups, hash sub range after hash sub range in id order, and where the rows of every hash sub range start, per code buffer
    std::vector<PassRow> passRows[2];
    std::vector<size_t> passRowOffsets[2];

    // a pass is cut into about CHUNKS_PER_THREAD chunks per thread of about equal bytes, of equal hash width per hash sub range; a windowed pass is one chunk
    const auto getChunkCount = [&](const ReceivedHashRange &received, size_t rangeHashSubRange, size_t pass) {
        const size_t passReceivedBytes = std::max<size_t>(1, received.passReceivedBytes[pass]);
        return (received.passBytes[pass] > passLimit) ? 1 : std::max<size_t>(1, (CHUNKS_PER_THREAD * par.threads * received.hashSubRangeBytes[rangeHashSubRange] + passReceivedBytes - 1) / passReceivedBytes);
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
            const std::vector<KmerHashSubRange> &rows = received.fileRows[sender];
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
        const bool hasWindowedPass = (*std::max_element(received.passBytes.begin(), received.passBytes.end()) > passLimit);
        received.codeBufferBytes = hasWindowedPass ? passLimit : maxPassBytes;
        received.passReceivedBytes.assign(received.passBytes.size(), 0);
        // a pass holds a start per row and chunk
        received.maxChunkStarts = 0;
        for (size_t pass = 0; pass < received.passBytes.size(); pass++) {
            for (size_t rangeHashSubRange = received.passHashSubRangeOffsets[pass]; rangeHashSubRange < received.passHashSubRangeOffsets[pass + 1]; rangeHashSubRange++) {
                received.passReceivedBytes[pass] += received.hashSubRangeBytes[rangeHashSubRange];
                receivedBytes += received.hashSubRangeBytes[rangeHashSubRange];
            }
            size_t passChunkStarts = 0;
            for (size_t rangeHashSubRange = received.passHashSubRangeOffsets[pass]; rangeHashSubRange < received.passHashSubRangeOffsets[pass + 1]; rangeHashSubRange++) {
                passChunkStarts += (getChunkCount(received, rangeHashSubRange, pass) + 1) * received.hashSubRangeReads[rangeHashSubRange];
            }
            received.maxChunkStarts = std::max(received.maxChunkStarts, passChunkStarts);
        }
    };
    const auto readPass = [&](const ReceivedHashRange &received, size_t pass, size_t codeBuffer, bool isAsync) {
        getPassRows(exchange, received.fileRows, received.passHashSubRangeOffsets[pass], received.passHashSubRangeOffsets[pass + 1], passRows[codeBuffer], passRowOffsets[codeBuffer]);
        readPassRows(receivedFiles, received.fileRows, passRows[codeBuffer], passBuffer + codeBuffer * codeBufferBytes, isAsync, batchReader, par.threads);
    };

    // the hash ranges of the own rows first, which this node took while sending
    size_t ownHashRanges = 0;
    const auto takeHashRange = [&]() {
        return (ownHashRanges < ownKmers.hashRanges.size()) ? ownKmers.hashRanges[ownHashRanges++] : hashQueue.claimNext();
    };
    ReceivedHashRange receivedHashRanges[2];
    size_t hashRange = takeHashRange();
    if (hashRange < exchange.hashRangeCount) {
        planHashRange(receivedHashRanges[0], hashRange);
    }
    // the code buffer of the next pass that is not windowed, and whether it is read already
    size_t passCodeBuffer = 0;
    bool isPrefetched = false;
    for (size_t taken = 0; hashRange < exchange.hashRangeCount; taken++) {
        hashRanges.push_back(hashRange);
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
        chunkStarts.reserve(received.maxChunkStarts);
        size_t nextHashRange = exchange.hashRangeCount;
        for (size_t pass = 0; pass < passCount; pass++) {
            const size_t passFirstHashSubRange = received.passHashSubRangeOffsets[pass];
            const size_t passHashSubRanges = received.passHashSubRangeOffsets[pass + 1] - passFirstHashSubRange;
            // a windowed pass is a single hash sub range that reads its codes in windows of windowBytes
            const bool isWindowedPass = (received.passBytes[pass] > passLimit);
            char *codeBuffer = passBuffer + passCodeBuffer * codeBufferBytes;
            size_t windowBytes = 0;
            if (isWindowedPass) {
                windowedPasses++;
                codeBuffer = passBuffer;
                windowBytes = 2 * passLimit / (received.hashSubRangeReads[passFirstHashSubRange] * KMER_CODE_COUNT);
                if (windowBytes == 0) {
                    Debug(Debug::ERROR) << "Not enough memory to receive the k-mers. Raise --split-memory-limit.\n";
                    EXIT(EXIT_FAILURE);
                }
                getPassRows(exchange, received.fileRows, passFirstHashSubRange, received.passHashSubRangeOffsets[pass + 1], passRows[passCodeBuffer], passRowOffsets[passCodeBuffer]);
            } else if (isPrefetched == false) {
                readPass(received, pass, passCodeBuffer, false);
            } else {
                batchReader->waitAsyncRead();
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
            isPrefetched = isWindowedPass == false && prefetched != NULL && prefetched->codeBufferBytes <= codeBufferBytes
                           && prefetched->passBytes[prefetchedPass] <= passLimit && prefetched->passBytes[prefetchedPass] > 0;
            if (isPrefetched) {
                readPass(*prefetched, prefetchedPass, 1 - passCodeBuffer, true);
            }
            const std::vector<PassRow> &hashSubRangeRows = passRows[passCodeBuffer];
            const std::vector<size_t> &hashSubRangeRowOffsets = passRowOffsets[passCodeBuffer];
            if (isWindowedPass == false) {
                passCodeBuffer = 1 - passCodeBuffer;
            }
            const auto isWholeSequenceHashSubRange = [&](size_t passHashSubRange) {
                return passFirstHashSubRange + passHashSubRange >= exchange.hashSubRangesPerRange;
            };
            std::vector<size_t> hashSubRangeChunkOffsets(passHashSubRanges + 1, 0);
            // the chunk starts of a hash sub range: a start per row for every chunk and then the end of every row
            std::vector<size_t> chunkStartOffsets(passHashSubRanges + 1, 0);
            std::vector<size_t> chunkHashSubRanges;
            for (size_t passHashSubRange = 0; passHashSubRange < passHashSubRanges; passHashSubRange++) {
                const size_t chunkCount = getChunkCount(received, passFirstHashSubRange + passHashSubRange, pass);
                hashSubRangeChunkOffsets[passHashSubRange + 1] = hashSubRangeChunkOffsets[passHashSubRange] + chunkCount;
                chunkStartOffsets[passHashSubRange + 1] = chunkStartOffsets[passHashSubRange]
                                                          + (chunkCount + 1) * (hashSubRangeRowOffsets[passHashSubRange + 1] - hashSubRangeRowOffsets[passHashSubRange]);
                chunkHashSubRanges.insert(chunkHashSubRanges.end(), chunkCount, passHashSubRange);
            }
            const size_t passChunks = chunkHashSubRanges.size();
            chunkStarts.resize(chunkStartOffsets[passHashSubRanges]);
            // every row decodes its index codes once to find its first group in every chunk; member and rep candidate codes have a fixed width per group
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
            for (size_t passRow = 0; passRow < hashSubRangeRows.size(); passRow++) {
                const size_t passHashSubRange = std::upper_bound(hashSubRangeRowOffsets.begin(), hashSubRangeRowOffsets.end(), passRow) - hashSubRangeRowOffsets.begin() - 1;
                const size_t chunkCount = hashSubRangeChunkOffsets[passHashSubRange + 1] - hashSubRangeChunkOffsets[passHashSubRange];
                const size_t rowCount = hashSubRangeRowOffsets[passHashSubRange + 1] - hashSubRangeRowOffsets[passHashSubRange];
                KmerChunkStart *rowStarts = chunkStarts.data() + chunkStartOffsets[passHashSubRange] + passRow - hashSubRangeRowOffsets[passHashSubRange];
                const ReceivedFile &receivedFile = receivedFiles[hashSubRangeRows[passRow].file];
                const KmerHashSubRange &hashSubRangeRow = received.fileRows[hashSubRangeRows[passRow].file][hashSubRangeRows[passRow].row];
                size_t hashSubRange = 0;
                size_t hashSubRangeEnd = 0;
                getRowHashSubRanges(exchange, received.hashRange, passFirstHashSubRange + passHashSubRange, hashSubRange, hashSubRangeEnd);
                const bool isWholeSequence = isWholeSequenceHashSubRange(passHashSubRange);
                const unsigned int hashShift = isWholeSequence ? 0 : exchange.kmerHashShift;
                const KmerPositionBit::uint128 hashStart = getHashSubRangeStart(hashSubRange, exchange.hashSubRangeCount, hashShift);
                const KmerPositionBit::uint128 hashWidth = getHashSubRangeStart(hashSubRangeEnd, exchange.hashSubRangeCount, hashShift) - hashStart;
                KmerChunkStart start = {0, static_cast<uint64_t>(hashStart), {0, 0, 0}};
                rowStarts[0] = start;
                if (chunkCount > 1) {
                    KmerCodeReader indexRead = getRowCodeReader(receivedFile, hashSubRangeRow, hashSubRangeRows[passRow].bufferOffset, KMER_INDEX_CODES, codeBuffer, NULL, 0);
                    const unsigned int hashDiffLowBits = getHashDiffLowBits(hashSubRange, hashSubRangeEnd, exchange.hashSubRangeCount, hashShift, hashSubRangeRow.groupCount);
                    size_t chunk = 1;
                    for (; start.group < hashSubRangeRow.groupCount; start.group++) {
                        start.bitPos[KMER_INDEX_CODES] = getKmerCodeBitPos(indexRead);
                        const uint64_t hash = start.hash + getRice(indexRead, hashDiffLowBits) + ((start.group > 0) ? 1 : 0);
                        for (; chunk < chunkCount && hash >= hashStart + hashWidth * chunk / chunkCount; chunk++) {
                            rowStarts[chunk * rowCount] = start;
                        }
                        const size_t memberCount = getEliasGamma(indexRead);
                        const size_t repCandidateCount = isWholeSequence ? 0 : getEliasGamma(indexRead);
                        start.bitPos[KMER_MEMBER_CODES] += memberCount * hashSubRangeRow.idBits;
                        start.bitPos[KMER_REP_CANDIDATE_CODES] += repCandidateCount * (exchange.adjKeyCodeBits + getBitWidth(memberCount - 1));
                        start.hash = hash;
                    }
                    start.bitPos[KMER_INDEX_CODES] = getKmerCodeBitPos(indexRead);
                    for (; chunk < chunkCount; chunk++) {
                        rowStarts[chunk * rowCount] = start;
                    }
                } else {
                    for (size_t code = 0; code < KMER_CODE_COUNT; code++) {
                        start.bitPos[code] = (hashSubRangeRow.offsets[code + 1] - hashSubRangeRow.offsets[code]) * CHAR_BIT;
                    }
                }
                start.group = hashSubRangeRow.groupCount;
                rowStarts[chunkCount * rowCount] = start;
            }
            // the code bits of every chunk, so the threads take the largest first
            std::vector<size_t> chunkBits(passChunks, 0);
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
            for (size_t chunk = 0; chunk < passChunks; chunk++) {
                const size_t passHashSubRange = chunkHashSubRanges[chunk];
                const size_t rowCount = hashSubRangeRowOffsets[passHashSubRange + 1] - hashSubRangeRowOffsets[passHashSubRange];
                const KmerChunkStart *rowStarts = chunkStarts.data() + chunkStartOffsets[passHashSubRange] + (chunk - hashSubRangeChunkOffsets[passHashSubRange]) * rowCount;
                for (size_t row = 0; row < rowCount; row++) {
                    for (size_t code = 0; code < KMER_CODE_COUNT; code++) {
                        chunkBits[chunk] += rowStarts[rowCount + row].bitPos[code] - rowStarts[row].bitPos[code];
                    }
                }
            }
            std::vector<size_t> roundChunks(passChunks);
            for (size_t chunk = 0; chunk < passChunks; chunk++) {
                roundChunks[chunk] = chunk;
            }
            SORT_SERIAL(roundChunks.begin(), roundChunks.end(), [&](size_t first, size_t second) {
                return chunkBits[first] > chunkBits[second] || (chunkBits[first] == chunkBits[second] && first < second);
            });
            // the threads fill the output until it is full, then it is written and the chunks that stopped go on where they stopped;
            // a chunk with a group that does not fit the empty output goes on alone after the others and writes the output whenever it is full
            std::vector<std::vector<KmerChunkStart> > chunkResumes(passChunks);
            std::vector<char> chunkStops(passChunks, CHUNK_DONE);
            std::vector<size_t> tooBigChunks;
            for (bool isAlone = false; roundChunks.empty() == false;) {
#pragma omp parallel num_threads(isAlone ? 1 : par.threads)
                {
                    unsigned int thread_idx = 0;
#ifdef OPENMP
                    thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
                    PairOutputBlock &block = outputBlocks[thread_idx];
                    size_t threadGroups = 0;
                    size_t threadOneMemberGroups = 0;
                    size_t threadOneMemberGroupBits = 0;
                    // only a thread that is alone writes the output, when a group outgrows it
                    const auto flushAloneOutput = [&]() {
                        flushPairOutput(exchange, pairFile, output, outputBlocks, pairWriteBuffer, 1);
                        takePairOutput(output, block, halfRecords, halfTargets);
                    };
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
                    repCandidates.reserve(KMER_FILE_BUFFER_SIZE / sizeof(GroupRepCandidate));
                    std::vector<DBKeyType> repSeqIds;
                    KmerAdjKeyQueue adjKeyQueue;
                    // the readers of a code of the group rows go back to the group start
                    const auto rewindGroupReads = [&](size_t code, const std::vector<KmerCodeReader> &groupStarts) {
                        for (size_t row = 0; row < groupRows.size(); row++) {
                            KmerCodeReader &codeRead = reads[groupRows[row]].codeReads[code];
                            if (codeRead.fileOffset != groupStarts[row].fileOffset) {
                                readKmerCodeWindow(codeRead, groupStarts[row].fileOffset);
                            }
                            codeRead = groupStarts[row];
                        }
                    };
#pragma omp for schedule(dynamic, 1)
                    for (size_t i = 0; i < roundChunks.size(); i++) {
                        const size_t chunk = roundChunks[i];
                        chunkStops[chunk] = CHUNK_DONE;
                        const size_t passHashSubRange = chunkHashSubRanges[chunk];
                        size_t hashSubRange = 0;
                        size_t hashSubRangeEnd = 0;
                        getRowHashSubRanges(exchange, received.hashRange, passFirstHashSubRange + passHashSubRange, hashSubRange, hashSubRangeEnd);
                        const bool isWholeSequence = isWholeSequenceHashSubRange(passHashSubRange);
                        const unsigned int hashShift = isWholeSequence ? 0 : exchange.kmerHashShift;
                        const size_t groupMaxRepSeqs = isWholeSequence ? 1 : exchange.maxRepSeqs;
                        const size_t rowCount = hashSubRangeRowOffsets[passHashSubRange + 1] - hashSubRangeRowOffsets[passHashSubRange];
                        const KmerChunkStart *rowStarts = chunkStarts.data() + chunkStartOffsets[passHashSubRange] + (chunk - hashSubRangeChunkOffsets[passHashSubRange]) * rowCount;
                        const KmerChunkStart *rowResumes = chunkResumes[chunk].empty() ? rowStarts : chunkResumes[chunk].data();
                        reads.clear();
                        for (size_t row = 0; row < rowCount; row++) {
                            const KmerChunkStart &start = rowResumes[row];
                            const KmerChunkStart &end = rowStarts[rowCount + row];
                            if (start.group == end.group) {
                                continue;
                            }
                            const PassRow &passRow = hashSubRangeRows[hashSubRangeRowOffsets[passHashSubRange] + row];
                            const KmerHashSubRange &hashSubRangeRow = received.fileRows[passRow.file][passRow.row];
                            KmerHashSubRangeRead read;
                            for (size_t code = 0; code < KMER_CODE_COUNT; code++) {
                                read.codeReads[code] = getRowCodeReader(receivedFiles[passRow.file], hashSubRangeRow, passRow.bufferOffset, code, codeBuffer,
                                                                        codeBuffer + (reads.size() * KMER_CODE_COUNT + code) * windowBytes, windowBytes);
                                read.codeStartBitPos[code] = getKmerCodeBitPos(read.codeReads[code]);
                                if (start.bitPos[code] > 0) {
                                    seekKmerCodeBits(read.codeReads[code], read.codeStartBitPos[code] + start.bitPos[code]);
                                }
                            }
                            read.hashSubRangeRow = &hashSubRangeRow;
                            read.chunkRow = row;
                            read.isWholeSequence = isWholeSequence;
                            read.hashDiffLowBits = getHashDiffLowBits(hashSubRange, hashSubRangeEnd, exchange.hashSubRangeCount, hashShift, hashSubRangeRow.groupCount);
                            read.group = start.group;
                            read.groupEnd = end.group;
                            read.hash = start.hash;
                            reads.push_back(read);
                        }

                        // merges the groups of the rows by hash
                        KmerHashQueue kmerHashQueue;
                        for (size_t row = 0; row < reads.size(); row++) {
                            queueNextKmerGroup(kmerHashQueue, row, reads[row]);
                        }
                        while (kmerHashQueue.empty() == false) {
                            const uint64_t hash = kmerHashQueue.top().hash;
                            groupRows.clear();
                            size_t memberCount = 0;
                            while (kmerHashQueue.empty() == false && kmerHashQueue.top().hash == hash) {
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
                                // the records and targets of the group at most: a record per representative with every member as its target
                                const size_t groupRecords = groupMaxRepSeqs;
                                const size_t groupTargets = groupMaxRepSeqs * memberCount;
                                if (isAlone == false && takePairOutput(output, block, groupRecords, groupTargets) == false) {
                                    chunkStops[chunk] = (groupRecords * sizeof(AssignedPairRecord) + groupTargets * sizeof(DBKeyType) > output.bytes) ? CHUNK_GROUP_TOO_BIG : CHUNK_OUTPUT_FULL;
                                    break;
                                }
                                if (isAlone && takePairOutput(output, block, groupRecords, groupTargets) == false) {
                                    flushPairOutput(exchange, pairFile, output, outputBlocks, pairWriteBuffer, 1);
                                    takePairOutput(output, block, std::min(groupRecords, halfRecords), std::min(groupTargets, halfTargets));
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
                                    if (groupRepCandidateCount <= KMER_FILE_BUFFER_SIZE / sizeof(GroupRepCandidate)) {
                                        repCandidates.clear();
                                        rewindGroupReads(KMER_REP_CANDIDATE_CODES, groupRepCandidateStarts);
                                        for (size_t row = 0; row < groupRows.size(); row++) {
                                            KmerHashSubRangeRead &read = reads[groupRows[row]];
                                            const unsigned int memberIndexBits = getBitWidth(read.groupMemberCount - 1);
                                            for (size_t repCandidate = 0; repCandidate < read.groupRepCandidateCount; repCandidate++) {
                                                const unsigned int adjKey = unpackAdjKey(exchange, getBits(read.codeReads[KMER_REP_CANDIDATE_CODES], exchange.adjKeyCodeBits));
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
                                                    const unsigned int adjKey = unpackAdjKey(exchange, getBits(read.codeReads[KMER_REP_CANDIDATE_CODES], exchange.adjKeyCodeBits));
                                                    visit(adjKey, std::pair<size_t, size_t>(0, getBits(read.codeReads[KMER_REP_CANDIDATE_CODES], memberIndexBits)));
                                                }
                                            } else {
                                                // the rows merged by adjKey, at the same adjKey in row order
                                                repCandidatesLeft.resize(groupRows.size());
                                                for (size_t row = 0; row < groupRows.size(); row++) {
                                                    KmerHashSubRangeRead &read = reads[groupRows[row]];
                                                    repCandidatesLeft[row] = read.groupRepCandidateCount - 1;
                                                    adjKeyQueue.push(FileKmerAdjKey(unpackAdjKey(exchange, getBits(read.codeReads[KMER_REP_CANDIDATE_CODES], exchange.adjKeyCodeBits)), row));
                                                }
                                                while (adjKeyQueue.empty() == false) {
                                                    const FileKmerAdjKey repCandidate = adjKeyQueue.top();
                                                    adjKeyQueue.pop();
                                                    KmerHashSubRangeRead &read = reads[groupRows[repCandidate.row]];
                                                    const size_t memberIndex = getBits(read.codeReads[KMER_REP_CANDIDATE_CODES], getBitWidth(read.groupMemberCount - 1));
                                                    visit(repCandidate.adjKey, std::pair<size_t, size_t>(repCandidate.row, memberIndex));
                                                    if (repCandidatesLeft[repCandidate.row] > 0) {
                                                        repCandidatesLeft[repCandidate.row]--;
                                                        adjKeyQueue.push(FileKmerAdjKey(unpackAdjKey(exchange, getBits(read.codeReads[KMER_REP_CANDIDATE_CODES], exchange.adjKeyCodeBits)), repCandidate.row));
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
                                    repSeqIds.push_back(getBits(memberRead, read.hashSubRangeRow->idBits) + read.hashSubRangeRow->loadFirstKey);
                                }
                                // a pair record per representative: the members with a length it can cover, without itself, in id order without repeats as the rows of a group are loads in key order with distinct ids in order
                                for (size_t rep = 0; rep < repSeqIds.size(); rep++) {
                                    const DBKeyType repId = repSeqIds[rep];
                                    const size_t repRange = seqDbr.findIdRange(repId);
                                    const float repLength = static_cast<float>(index[repRange].length - 2);
                                    const auto isCovered = [&](const DBReader<DBKeyType>::Index &range) {
                                        return Util::canBeCovered(par.covThr, par.covMode, repLength, static_cast<float>(range.length - 2));
                                    };
                                    // ranges are in length descending order and every --cov-mode covers an interval of lengths that holds the representative length
                                    const DBReader<DBKeyType>::Index *coveredFirstRange = std::partition_point(index, index + repRange,
                                                                                                               [&](const DBReader<DBKeyType>::Index &range) { return isCovered(range) == false; });
                                    const DBReader<DBKeyType>::Index *coveredEndRange = std::partition_point(index + repRange, index + seqDbr.getIndexSize(), isCovered);
                                    const size_t coveredFrom = coveredFirstRange->id;
                                    const size_t coveredTo = (coveredEndRange < index + seqDbr.getIndexSize()) ? static_cast<size_t>(coveredEndRange->id) : seqDbr.getSize();
                                    AssignedPairRecord record = {repId, 0, 0};
                                    const auto startRecord = [&]() {
                                        if (block.recordPos == block.recordEnd) {
                                            flushAloneOutput();
                                        }
                                        record.targetOffset = block.targetPos;
                                        record.targetCount = 0;
                                    };
                                    const auto endRecord = [&]() {
                                        if (record.targetCount > 0) {
                                            output.records[block.recordPos++] = record;
                                        }
                                    };
                                    rewindGroupReads(KMER_MEMBER_CODES, groupMemberStarts);
                                    startRecord();
                                    for (size_t row = 0; row < groupRows.size(); row++) {
                                        KmerHashSubRangeRead &read = reads[groupRows[row]];
                                        for (size_t member = 0; member < read.groupMemberCount; member++) {
                                            const size_t id = getBits(read.codeReads[KMER_MEMBER_CODES], read.hashSubRangeRow->idBits) + read.hashSubRangeRow->loadFirstKey;
                                            if (id == repId || id < coveredFrom || id >= coveredTo) {
                                                continue;
                                            }
                                            // a record that outgrows the output of a thread that is alone continues in the next flush
                                            if (block.targetPos == block.targetEnd) {
                                                endRecord();
                                                flushAloneOutput();
                                                startRecord();
                                            }
                                            output.targets[block.targetPos++] = id;
                                            record.targetCount++;
                                        }
                                    }
                                    endRecord();
                                }
                            }
                            for (size_t row = 0; row < groupRows.size(); row++) {
                                queueNextKmerGroup(kmerHashQueue, groupRows[row], reads[groupRows[row]]);
                            }
                        }
                        // a stopped chunk goes on at the group that it stopped at, where its rows have read only the index codes of their next group
                        if (chunkStops[chunk] != CHUNK_DONE) {
                            if (chunkResumes[chunk].empty()) {
                                chunkResumes[chunk].assign(rowStarts, rowStarts + rowCount);
                            }
                            for (size_t row = 0; row < reads.size(); row++) {
                                const KmerHashSubRangeRead &read = reads[row];
                                KmerChunkStart &resume = chunkResumes[chunk][read.chunkRow];
                                resume.group = read.pendingGroup;
                                resume.hash = read.pendingHash;
                                resume.bitPos[KMER_INDEX_CODES] = read.pendingIndexBitPos - read.codeStartBitPos[KMER_INDEX_CODES];
                                resume.bitPos[KMER_MEMBER_CODES] = getKmerCodeBitPos(read.codeReads[KMER_MEMBER_CODES]) - read.codeStartBitPos[KMER_MEMBER_CODES];
                                resume.bitPos[KMER_REP_CANDIDATE_CODES] = getKmerCodeBitPos(read.codeReads[KMER_REP_CANDIDATE_CODES]) - read.codeStartBitPos[KMER_REP_CANDIDATE_CODES];
                            }
                        }
                    }
#pragma omp atomic
                    receivedGroups += threadGroups;
#pragma omp atomic
                    oneMemberGroups += threadOneMemberGroups;
#pragma omp atomic
                    oneMemberGroupBits += threadOneMemberGroupBits;
                }
                std::vector<size_t> fullChunks;
                for (size_t i = 0; i < roundChunks.size(); i++) {
                    if (chunkStops[roundChunks[i]] == CHUNK_OUTPUT_FULL) {
                        fullChunks.push_back(roundChunks[i]);
                    } else if (chunkStops[roundChunks[i]] == CHUNK_GROUP_TOO_BIG) {
                        tooBigChunks.push_back(roundChunks[i]);
                    }
                }
                // the output is full, or the chunks that go on alone start with an empty output
                isAlone = fullChunks.empty();
                if (isAlone == false || tooBigChunks.empty() == false) {
                    flushPairOutput(exchange, pairFile, output, outputBlocks, pairWriteBuffer, par.threads);
                }
                aloneChunks += isAlone ? tooBigChunks.size() : 0;
                roundChunks.swap(isAlone ? tooBigChunks : fullChunks);
                if (isAlone) {
                    tooBigChunks.clear();
                }
            }
        }
        // the receiving node of the hash range is the only reader of its codes
        std::vector<PassRow> hashRangeRows;
        std::vector<size_t> hashRangeRowOffsets;
        getPassRows(exchange, received.fileRows, 0, exchange.rowsPerHashRange, hashRangeRows, hashRangeRowOffsets);
        punchedRanges += punchPassRows(receivedFiles, received.fileRows, hashRangeRows);
        hashRange = nextHashRange;
    }
    flushPairOutput(exchange, pairFile, output, outputBlocks, pairWriteBuffer, par.threads);
    delete batchReader;
    closeReceivedFiles(receivedFiles);
    batchWriter.waitWrites();
    MMseqsNode::addPhaseSeconds(receiveValues, "waiting to write pairs", batchWriter.getWaitSeconds() / par.threads);
    MMseqsNode::addPhaseValue(receiveValues, "Pair flushes", SSTR(output.flushes));
    MMseqsNode::addPhaseValue(receiveValues, "Passes read in windows", SSTR(windowedPasses));
    MMseqsNode::addPhaseValue(receiveValues, "Chunks with a group too big for the output", SSTR(aloneChunks));
    MMseqsNode::addPhaseValue(receiveValues, "K-mer file ranges freed", SSTR(punchedRanges));
    MMseqsNode::addPhaseValue(receiveValues, "K-mer groups", SSTR(receivedGroups));
    MMseqsNode::addPhaseValue(receiveValues, "K-mer groups with one member", SSTR(oneMemberGroups));
    MMseqsNode::addPhaseValue(receiveValues, "K-mer bytes of groups with one member", SSTR((oneMemberGroupBits + CHAR_BIT - 1) / CHAR_BIT));
    MMseqsNode::addPhaseValue(receiveValues, "K-mer rep candidate bytes", SSTR(repCandidateBytes));
    Util::releasePages(outputBuffer, outputBytes);
    delete[] outputBuffer;
    Util::releasePages(passBuffer, 2 * codeBufferBytes);
    delete[] passBuffer;
    delete[] pairWriteBuffer;
    writeRowTable(pairFile.file, pairFile.name, exchange.sendRangeCount, pairFile.blockRowStarts, pairFile.partitionBytes, pairFile.bytes);
    pairBytes = pairFile.bytes;
    return receivedBytes;
}

// --createdb-mode 3 rep node: the nodes take the merge splits one at a time and merge their partitions from the pair file of every node into <pref>.<node>; returns its partition rows
static std::string mergeAssignedPairs(Parameters &par, DBReader<DBKeyType> &seqDbr, const KmerExchange &exchange, MMseqsNode::WorkQueue &mergeQueue, std::vector<size_t> &mergeSplits,
                                      std::string &mergeValues) {
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    PrefFiles prefFiles;
    prefFiles.dataName = par.db2 + "." + SSTR(MMseqsNode::rank);
    prefFiles.indexName = par.db2 + ".index." + SSTR(MMseqsNode::rank);
    prefFiles.dataFile = FileUtil::openAndDelete(prefFiles.dataName.c_str(), "w");
    prefFiles.indexFile = FileUtil::openAndDelete(prefFiles.indexName.c_str(), "w");
    BatchWriter batchWriter(exchange.writeBehindMemory, par.threads, fileno(prefFiles.dataFile));
    prefFiles.batchWriter = &batchWriter;
    size_t prefDataBytes = 0;
    size_t prefIndexBytes = 0;
    prefFiles.writeBuffers = new(std::nothrow) char[par.threads * 2 * KMER_FILE_BUFFER_SIZE];
    Util::checkAllocation(prefFiles.writeBuffers, "Can not allocate prefWriteBuffer memory in mergeAssignedPairs");
    // a partition is its start, the starts and cuts of its key chunks and its end, with the pref bytes before them in the pref file and the alignment cells before them in the partition
    std::vector<PartitionCut> partitionCuts;
    std::vector<size_t> partitionCutOffsets(1, 0);
    KmerPositionBit::uint128 mergedCost = 0;
    const auto addKeyChunkCuts = [&](size_t partitionFirstKey, size_t chunkFirstKey, size_t chunkPrefOffset, size_t chunkCost, const ChunkCuts &chunkCuts) {
        const PartitionCut chunkStart = { partitionFirstKey + chunkFirstKey, chunkPrefOffset, chunkCost };
        partitionCuts.push_back(chunkStart);
        for (size_t i = 0; i < chunkCuts.cuts.size(); i++) {
            const PartitionCut cut = { partitionFirstKey + chunkCuts.cuts[i].key, chunkPrefOffset + chunkCuts.cuts[i].prefBytes, chunkCost + chunkCuts.cuts[i].cost };
            partitionCuts.push_back(cut);
        }
    };
    const auto addPartitionEnd = [&](size_t prefEnd, size_t cost) {
        const PartitionCut end = { 0, prefEnd, cost };
        partitionCuts.push_back(end);
        partitionCutOffsets.push_back(partitionCuts.size());
        mergedCost += cost;
    };
    // the buffer of the largest send range so far, kept for the next ones so their pages are not mapped again
    char *pairPassBuffer = NULL;
    size_t pairPassBufferBytes = 0;
    // the targets of the partitions of a pass, one after the other, kept like pairPassBuffer
    DBKeyType *pairPassTargets = NULL;
    size_t pairPassTargetsSize = 0;
    // the pair file of every receiving node, open for the whole phase, and its rows of a send range
    std::vector<ReceivedFile> receivedPairFiles(MMseqsNode::count);
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
    for (size_t receiver = 0; receiver < receivedPairFiles.size(); receiver++) {
        receivedPairFiles[receiver].fileName = par.db2 + ".pairs." + SSTR(receiver);
        openReceivedFile(receivedPairFiles[receiver], exchange.sendRangeCount, exchange.sendRangeCount * exchange.partitionsPerRange);
    }
    std::vector<std::vector<AssignedPairPartition> > pairFileRows(MMseqsNode::count);
    // every node cuts the partitions of all send ranges into as many merge splits as send ranges, key ranges of whole partitions and about equal cost, like decomposeDomainByAminoAcid;
    // a partition costs its pair bytes and an entry per key
    const size_t partitionCount = exchange.sendRangeCount * exchange.partitionsPerRange;
    std::vector<size_t> partitionPairBytes(partitionCount, 0);
    std::vector<size_t> partitionCosts(partitionCount, 0);
    size_t globalCost = 0;
    for (size_t partition = 0; partition < partitionCount; partition++) {
        for (size_t receiver = 0; receiver < receivedPairFiles.size(); receiver++) {
            partitionPairBytes[partition] += receivedPairFiles[receiver].itemBytes[partition];
        }
        const size_t sendRange = partition / exchange.partitionsPerRange;
        const size_t rangePartition = partition % exchange.partitionsPerRange;
        partitionCosts[partition] = partitionPairBytes[partition] + getPartitionFirstKey(exchange, sendRange, rangePartition + 1) - getPartitionFirstKey(exchange, sendRange, rangePartition);
        globalCost += partitionCosts[partition];
    }
    for (size_t receiver = 0; receiver < receivedPairFiles.size(); receiver++) {
        std::vector<size_t>().swap(receivedPairFiles[receiver].itemBytes);
    }
    const size_t mergeSplitCount = exchange.sendRangeCount;
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
    std::vector<size_t> splitPairBytes(mergeSplitCount, 0);
    std::vector<size_t> splitCosts(mergeSplitCount, 0);
    std::vector<size_t> mergeOrder(mergeSplitCount);
    for (size_t split = 0; split < mergeSplitCount; split++) {
        for (size_t partition = splitPartitionOffsets[split]; partition < splitPartitionOffsets[split + 1]; partition++) {
            splitPairBytes[split] += partitionPairBytes[partition];
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
    for (size_t item = mergeQueue.claimNext(); item < mergeSplitCount; item = mergeQueue.claimNext()) {
        const size_t mergeSplit = mergeOrder[item];
        mergedPairBytes += splitPairBytes[mergeSplit];
        mergeSplits.push_back(mergeSplit);
        const size_t splitBegin = splitPartitionOffsets[mergeSplit];
        const size_t splitEnd = splitPartitionOffsets[mergeSplit + 1];
        for (size_t sendRange = splitBegin / exchange.partitionsPerRange; splitBegin < splitEnd && sendRange * exchange.partitionsPerRange < splitEnd; sendRange++) {
            // the partitions [firstPartition, endPartition) of the send range that the split holds
            const size_t firstPartition = std::max(splitBegin, sendRange * exchange.partitionsPerRange) - sendRange * exchange.partitionsPerRange;
            const size_t endPartition = std::min(splitEnd, (sendRange + 1) * exchange.partitionsPerRange) - sendRange * exchange.partitionsPerRange;
            setSendRange(par, exchange, sendRange);
            // the rows of the send range lie in every flush block of a pair file, so all of them are read in one batch
            std::vector<std::vector<BatchReader::ReadRequest> > rowReadRequests(receivedPairFiles.size());
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
            for (size_t receiver = 0; receiver < receivedPairFiles.size(); receiver++) {
                readItemRows(receivedPairFiles[receiver], sendRange, pairFileRows[receiver], &rowReadRequests[receiver]);
            }
            for (size_t receiver = 1; receiver < rowReadRequests.size(); receiver++) {
                rowReadRequests[0].insert(rowReadRequests[0].end(), rowReadRequests[receiver].begin(), rowReadRequests[receiver].end());
            }
            BatchReader *batchReader = new BatchReader(par.threads, fileno(receivedPairFiles[0].file));
            batchReader->readSync(rowReadRequests[0]);
            std::vector<size_t> partitionCodeBytes(exchange.partitionsPerRange, 0);
            std::vector<size_t> partitionTargets(exchange.partitionsPerRange, 0);
            // the received file and row of every row of a partition
            std::vector<std::vector<std::pair<size_t, size_t> > > partitionRows(exchange.partitionsPerRange);
            std::vector<size_t> partitionRowCounts(exchange.partitionsPerRange, 0);
            for (size_t receiver = 0; receiver < receivedPairFiles.size(); receiver++) {
                const std::vector<AssignedPairPartition> &rows = pairFileRows[receiver];
                for (size_t row = 0; row < rows.size(); row++) {
                    partitionRowCounts[rows[row].partition] += (rows[row].partition >= firstPartition && rows[row].partition < endPartition);
                }
            }
            for (size_t partition = 0; partition < exchange.partitionsPerRange; partition++) {
                partitionRows[partition].reserve(partitionRowCounts[partition]);
            }
            for (size_t receiver = 0; receiver < receivedPairFiles.size(); receiver++) {
                const std::vector<AssignedPairPartition> &rows = pairFileRows[receiver];
                for (size_t row = 0; row < rows.size(); row++) {
                    const size_t partition = rows[row].partition;
                    if (partition < firstPartition || partition >= endPartition) {
                        continue;
                    }
                    partitionCodeBytes[partition] += getRowEnd(rows[row]) - rows[row].offsets[0];
                    partitionTargets[partition] += rows[row].targetCount;
                    partitionRows[partition].push_back(std::make_pair(receiver, row));
                }
            }
            const size_t pairPassLimit = getPassLimit(exchange.loadMemoryLimit);
            // a partition takes its codes, its target ids, where the targets of every key start and its merge with a thread
            std::vector<size_t> partitionBytes(endPartition - firstPartition);
            for (size_t partition = firstPartition; partition < endPartition; partition++) {
                const size_t partitionKeys = getPartitionFirstKey(exchange, sendRange, partition + 1) - getPartitionFirstKey(exchange, sendRange, partition);
                partitionBytes[partition - firstPartition] = partitionCodeBytes[partition] + partitionTargets[partition] * sizeof(DBKeyType) + (partitionKeys + 1) * sizeof(size_t)
                                                             + getPartitionMergeBytes(partitionRowCounts[partition], 1);
            }
            // a pass and the prefetched codes of the next one fit their limit; a bigger partition is a pass that reads its codes in windows
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
            bool hasWindowedPairPass = false;
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
                    hasWindowedPairPass = true;
                }
            }

            // two code buffers for a pass and the prefetched next one, or the windows of a windowed pass
            if (std::max(2 * maxPairPassCodeBytes, hasWindowedPairPass ? 2 * pairPassLimit : 0) > pairPassBufferBytes) {
                Util::releasePages(pairPassBuffer, pairPassBufferBytes);
                delete[] pairPassBuffer;
                pairPassBufferBytes = std::max(2 * maxPairPassCodeBytes, hasWindowedPairPass ? 2 * pairPassLimit : 0);
                pairPassBuffer = new(std::nothrow) char[pairPassBufferBytes];
                Util::checkAllocation(pairPassBuffer, "Can not allocate pairPassBuffer memory in mergeAssignedPairs");
            }
            // the threads that place the targets touch their pages first
            if (maxPairPassTargets > pairPassTargetsSize) {
                Util::releasePages(pairPassTargets, pairPassTargetsSize * sizeof(DBKeyType));
                delete[] pairPassTargets;
                pairPassTargetsSize = maxPairPassTargets;
                pairPassTargets = new(std::nothrow) DBKeyType[pairPassTargetsSize];
                Util::checkAllocation(pairPassTargets, "Can not allocate pairPassTargets memory in mergeAssignedPairs");
            }
            // the rows of a pass, partition after partition, and where the rows of every partition start
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
                // a windowed pass is a single partition that reads its codes in windows and merges its rows by key straight to the pref files
                const bool isWindowedPass = (pairPassBytes[pass] > pairPassLimit);
                char *codeBuffer = pairPassBuffer + (pass % 2) * maxPairPassCodeBytes;
                if (isWindowedPass) {
                    codeBuffer = pairPassBuffer;
                    getPairPassRows(partitionRows, passFirstPartition, passPartitionOffsets[pass + 1], pairPassRows[pass % 2], pairPassRowOffsets[pass % 2]);
                } else {
                    if (isPrefetched == false) {
                        readPairPass(pass, false);
                    } else {
                        batchReader->waitAsyncRead();
                    }
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
                        const size_t partitionFirstKey = getPartitionFirstKey(exchange, sendRange, partition);
                        countPartitionTargets(exchange, receivedPairFiles, pairFileRows, partitionPassRows.data() + partitionPassRowOffsets[passPartition],
                                              partitionPassRowOffsets[passPartition + 1] - partitionPassRowOffsets[passPartition], codeBuffer,
                                              getPartitionFirstKey(exchange, sendRange, partition + 1) - partitionFirstKey, partitionTargets[partition], partitionMerges[passPartition], 1);
                        mergePartitionTargets(exchange, seqDbr, partitionFirstKey, partitionMerges[passPartition], pairPassTargets + partitionTargetOffsets[passPartition], 1);
                    }
                    std::vector<size_t> partitionDataOffsets(passPartitions + 1, prefDataBytes);
                    std::vector<size_t> partitionIndexOffsets(passPartitions + 1, prefIndexBytes);
                    for (size_t passPartition = 0; passPartition < passPartitions; passPartition++) {
                        const PartitionMerge &merge = partitionMerges[passPartition];
                        partitionDataOffsets[passPartition + 1] = partitionDataOffsets[passPartition] + merge.keyChunkDataOffsets.back();
                        partitionIndexOffsets[passPartition + 1] = partitionIndexOffsets[passPartition] + merge.keyChunkEntryOffsets.back() * 2 * sizeof(uint64_t);
                        for (size_t keyChunk = 0; keyChunk + 1 < merge.keyChunkOffsets.size(); keyChunk++) {
                            addKeyChunkCuts(getPartitionFirstKey(exchange, sendRange, passFirstPartition + passPartition), merge.keyChunkOffsets[keyChunk],
                                            partitionDataOffsets[passPartition] + merge.keyChunkDataOffsets[keyChunk], merge.keyChunkCostOffsets[keyChunk], merge.keyChunkCuts[keyChunk]);
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
                        writePartition(prefFiles, getPartitionFirstKey(exchange, sendRange, passFirstPartition + passPartition), partitionMerges[passPartition],
                                       partitionDataOffsets[passPartition], partitionIndexOffsets[passPartition], thread_idx, 1);
                        partitionMerges[passPartition] = PartitionMerge();
                    }
                    prefDataBytes = partitionDataOffsets[passPartitions];
                    prefIndexBytes = partitionIndexOffsets[passPartitions];
                } else {
                    // a pass of few partitions: the partitions one after the other in one PartitionMerge that keeps its buffers, all threads merge a partition
                    for (size_t passPartition = 0; passPartition < passPartitions; passPartition++) {
                        const size_t partition = passFirstPartition + passPartition;
                        const size_t partitionFirstKey = getPartitionFirstKey(exchange, sendRange, partition);
                        const size_t partitionKeys = getPartitionFirstKey(exchange, sendRange, partition + 1) - partitionFirstKey;
                        const size_t partitionRowCount = partitionPassRowOffsets[passPartition + 1] - partitionPassRowOffsets[passPartition];
                        const size_t partitionDataOffset = prefDataBytes;
                        size_t partitionEntryCount = 0;
                        size_t partitionCost = 0;
                        if (isWindowedPass == false) {
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
                            for (size_t keyChunk = 0; keyChunk + 1 < merge.keyChunkOffsets.size(); keyChunk++) {
                                addKeyChunkCuts(partitionFirstKey, merge.keyChunkOffsets[keyChunk], prefDataBytes + merge.keyChunkDataOffsets[keyChunk], merge.keyChunkCostOffsets[keyChunk],
                                                merge.keyChunkCuts[keyChunk]);
                            }
                            partitionEntryCount = merge.keyChunkEntryOffsets.back();
                            partitionCost = merge.cost;
                            prefDataBytes += merge.keyChunkDataOffsets.back();
                        } else {
                            const size_t windowBytes = 2 * pairPassLimit / (std::max<size_t>(1, partitionRowCount) * PAIR_CODE_COUNT);
                            if (windowBytes == 0) {
                                Debug(Debug::ERROR) << "Not enough memory to merge the pairs. Raise --split-memory-limit.\n";
                                EXIT(EXIT_FAILURE);
                            }
                            std::vector<PairRowRead> reads;
                            getPartitionRowReads(receivedPairFiles, pairFileRows, partitionPassRows.data() + partitionPassRowOffsets[passPartition], partitionRowCount, partitionKeys, codeBuffer,
                                                 windowBytes, reads);
                            RepKeyQueue repKeyQueue;
                            TargetIdQueue targetQueue;
                            std::vector<unsigned int> keyRows;
                            for (size_t row = 0; row < reads.size(); row++) {
                                readPairRecord(reads[row], exchange.globalIdBits);
                                repKeyQueue.push(FileRepKey(reads[row].repKey, row));
                            }
                            // key chunks like those of a merge with all threads, so the partition gets as many places to cut
                            const size_t windowedChunkKeys = std::max<size_t>(1, partitionKeys / (CHUNKS_PER_THREAD * par.threads));
                            ChunkCuts windowedCuts;
                            size_t chunkFirstKey = 0;
                            size_t chunkPrefOffset = partitionDataOffset;
                            size_t chunkCost = 0;
                            KmerCodeWriter windowedDataWrite = {fileno(prefFiles.dataFile), prefFiles.dataName.c_str(), prefFiles.writeBuffers, KMER_FILE_BUFFER_SIZE, 0, prefDataBytes, 0, 0, &batchWriter};
                            KmerCodeWriter windowedIndexWrite = {fileno(prefFiles.indexFile), prefFiles.indexName.c_str(), prefFiles.writeBuffers + KMER_FILE_BUFFER_SIZE, KMER_FILE_BUFFER_SIZE, 0, prefIndexBytes, 0, 0, &batchWriter};
                            for (size_t key = 0; key < partitionKeys; key++) {
                                if (key == chunkFirstKey + windowedChunkKeys) {
                                    addKeyChunkCuts(partitionFirstKey, chunkFirstKey, chunkPrefOffset, chunkCost, windowedCuts);
                                    windowedCuts = ChunkCuts();
                                    chunkFirstKey = key;
                                    chunkPrefOffset = windowedDataWrite.writePos + windowedDataWrite.bufferPos;
                                    chunkCost = partitionCost;
                                }
                                const bool isRep = isRepresentative(partitionFirstKey + key);
                                if (isRep) {
                                    const uint64_t indexCell[2] = {static_cast<uint64_t>(partitionFirstKey + key), static_cast<uint64_t>(windowedDataWrite.writePos + windowedDataWrite.bufferPos)};
                                    putBytes(windowedIndexWrite, reinterpret_cast<const char *>(indexCell), sizeof(indexCell));
                                    putPrefKey(windowedDataWrite, partitionFirstKey + key);
                                }
                                size_t keyTargetCount = 0;
                                DBKeyType prevTarget = 0;
                                size_t targetRange = SIZE_MAX;
                                size_t targetResidues = 0;
                                mergeKeyTargets(reads, repKeyQueue, targetQueue, keyRows, key, exchange.globalIdBits, [&](DBKeyType target) {
                                    if (isRep) {
                                        putPrefKey(windowedDataWrite, target - prevTarget);
                                        prevTarget = target;
                                        targetResidues += seqDbr.getSeqLenFromRange(targetRange, target);
                                        keyTargetCount++;
                                    }
                                });
                                if (isRep) {
                                    putBytes(windowedDataWrite, "", 1);
                                    partitionEntryCount++;
                                    if (keyTargetCount > 0) {
                                        partitionCost += (index[seqDbr.findIdRange(partitionFirstKey + key)].length - 2) * targetResidues;
                                        const PartitionCut cut = { key + 1, windowedDataWrite.writePos + windowedDataWrite.bufferPos - chunkPrefOffset, partitionCost - chunkCost };
                                        windowedCuts.add(cut);
                                    }
                                }
                            }
                            writeKmerCodeBuffer(windowedDataWrite);
                            writeKmerCodeBuffer(windowedIndexWrite);
                            prefDataBytes = windowedDataWrite.writePos;
                            addKeyChunkCuts(partitionFirstKey, chunkFirstKey, chunkPrefOffset, chunkCost, windowedCuts);
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
            punchedRanges += punchPassRows(receivedPairFiles, pairFileRows, splitRows);
        }
    }
    closeReceivedFiles(receivedPairFiles);
    batchWriter.waitWrites();
    MMseqsNode::addPhaseSeconds(mergeValues, "waiting to write the prefilter", batchWriter.getWaitSeconds() / par.threads);
    MMseqsNode::addPhaseValue(mergeValues, "Pair bytes merged", SSTR(mergedPairBytes));
    MMseqsNode::addPhaseValue(mergeValues, "Pair file ranges freed", SSTR(punchedRanges));
    Util::releasePages(pairPassBuffer, pairPassBufferBytes);
    delete[] pairPassBuffer;
    Util::releasePages(pairPassTargets, pairPassTargetsSize * sizeof(DBKeyType));
    delete[] pairPassTargets;
    delete[] prefFiles.writeBuffers;
    MMseqsNode::closeClusterFiles(prefFiles.dataFile, prefFiles.dataName, prefFiles.indexFile, prefFiles.indexName, prefDataBytes, prefIndexBytes);
    // a row per part of a partition with entries: firstKey, prefOffset, prefBytes, alignment cells (query times target lengths) and the node of its pref file;
    // a part ends at the first cut where its cells reach 1 / (nodes * PARTITIONS_PER_NODE) of what this node merged, which is at most that share of what all nodes merged
    const size_t rowCost = std::max<size_t>(1, static_cast<size_t>(mergedCost / (MMseqsNode::count * PARTITIONS_PER_NODE)));
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
    // every node cuts the same send ranges, hash ranges and partitions, sized by the k-mers of all representatives of the database
    size_t globalKmers = 0;
    size_t globalBytes = 0;
    for (size_t range = 0; range < seqDbr.getIndexSize(); range++) {
        const size_t rangeKeys = ((range + 1 < seqDbr.getIndexSize()) ? static_cast<size_t>(index[range + 1].id) : seqDbr.getSize()) - index[range].id;
        globalKmers += seqDbr.getRangeRepresentativeCount(range) * computeSeqKmerCount(static_cast<int>(index[range].length - 2), par.kmerSize, par.kmersPerSequence, kmersPerSequenceScale);
        globalBytes += rangeKeys * index[range].length;
    }
    // RANGES_PER_NODE send ranges a node, one without other nodes to balance, cut like decomposeDomainByAminoAcid on sequence bytes plus SEND_KMER_BYTES a k-mer
    exchange.sendRangeCount = (MMseqsNode::count > 1) ? RANGES_PER_NODE * MMseqsNode::count : 1;
    std::vector<size_t> &sendRangeKeyOffsets = exchange.sendRangeKeyOffsets;
    sendRangeKeyOffsets.assign(exchange.sendRangeCount + 1, seqDbr.getSize());
    sendRangeKeyOffsets[0] = 0;
    const size_t sendRangeCost = std::max<size_t>(1, (globalKmers * SEND_KMER_BYTES + globalBytes + exchange.sendRangeCount - 1) / exchange.sendRangeCount);
    size_t costBefore = 0;
    for (size_t range = 0, sendRange = 1; range < seqDbr.getIndexSize(); range++) {
        const size_t rangeKeys = ((range + 1 < seqDbr.getIndexSize()) ? static_cast<size_t>(index[range + 1].id) : seqDbr.getSize()) - index[range].id;
        const size_t rangeKmers = seqDbr.getRangeRepresentativeCount(range) * computeSeqKmerCount(static_cast<int>(index[range].length - 2), par.kmerSize, par.kmersPerSequence, kmersPerSequenceScale);
        const size_t rangeCost = rangeKmers * SEND_KMER_BYTES + rangeKeys * index[range].length;
        for (; sendRange < exchange.sendRangeCount && costBefore + rangeCost >= sendRange * sendRangeCost; sendRange++) {
            const size_t cutCost = sendRange * sendRangeCost - costBefore;
            sendRangeKeyOffsets[sendRange] = index[range].id + static_cast<size_t>((static_cast<KmerPositionBit::uint128>(cutCost) * rangeKeys + rangeCost - 1) / rangeCost);
        }
        costBefore += rangeCost;
    }
    size_t maxSendRangeKeys = 0;
    for (size_t sendRange = 0; sendRange < exchange.sendRangeCount; sendRange++) {
        maxSendRangeKeys = std::max(maxSendRangeKeys, sendRangeKeyOffsets[sendRange + 1] - sendRangeKeyOffsets[sendRange]);
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

    // hash ranges are equal parts of the 64 bit hash, RANGES_PER_NODE a node and enough for the threads that write the k-mer file of a load
    exchange.hashRangeCount = std::max(RANGES_PER_NODE * MMseqsNode::count, CHUNKS_PER_THREAD * par.threads);
    exchange.globalIdBits = getBitWidth(seqDbr.getSize() - 1);
    exchange.maxRepSeqs = 1 + (includeAdjacency ? par.adjIteration : 0);
    // a receiving node reads about a member id, a rep candidate and its index codes per record
    const size_t receiverKmerBytes = (2 * exchange.globalIdBits + exchange.adjKeyCodeBits + CHAR_BIT - 1) / CHAR_BIT;
    // a representative node reads a target id and sorts it, a record can be a target of every representative of its group
    const size_t pairTargetBytes = (exchange.globalIdBits + CHAR_BIT - 1) / CHAR_BIT + sizeof(DBKeyType);
    // the nodes take the items of these queues; every node removes the claims of its own items that a killed run left before the barrier of the memory limits
    MMseqsNode::WorkQueue sendQueue(par.db2 + ".send_ranges", exchange.sendRangeCount);
    MMseqsNode::WorkQueue hashQueue(par.db2 + ".hash_ranges", exchange.hashRangeCount);
    MMseqsNode::WorkQueue mergeQueue(par.db2 + ".merge_splits", exchange.sendRangeCount);
    sendQueue.removeClaims();
    hashQueue.removeClaims();
    mergeQueue.removeClaims();
    const size_t minPassLimit = std::max<size_t>(1, getPassLimit(MMseqsNode::getSmallestMemoryLimit(par.db2 + ".memory", memoryLimit, par.threads)));
    const size_t passes = std::max<size_t>(1, (globalKmers * receiverKmerBytes + minPassLimit - 1) / minPassLimit);
    // a pass holds about CHUNKS_PER_THREAD hash sub ranges per thread, so the threads take whole hash sub ranges without decoding a row to cut it
    exchange.hashSubRangesPerRange = (std::max(passes, exchange.hashRangeCount) * CHUNKS_PER_THREAD * par.threads + exchange.hashRangeCount - 1) / exchange.hashRangeCount;
    exchange.hashSubRangeCount = exchange.hashRangeCount * exchange.hashSubRangesPerRange;
    exchange.rowsPerHashRange = exchange.hashSubRangesPerRange + 1;
    const size_t representativeBitsMemory = exchange.hasRepresentativeBits ? ((maxSendRangeKeys + 63) / 64 + 1) * sizeof(uint64_t) : 0;
    exchange.writeBehindMemory = memoryLimit / WRITE_BEHIND_MEMORY_DIVISOR;
    // every thread writes through two buffers at most, a load holds its rows until it writes them after its codes
    const size_t fixedMemory = representativeBitsMemory + 2 * par.threads * KMER_FILE_BUFFER_SIZE + exchange.writeBehindMemory
                               + exchange.hashRangeCount * exchange.rowsPerHashRange * sizeof(KmerHashSubRange);
    exchange.loadMemoryLimit = (fixedMemory < memoryLimit) ? static_cast<size_t>(static_cast<double>(memoryLimit - fixedMemory) * 0.95) : 0;
    const size_t rangePairPasses = std::max<size_t>(1, (globalKmers / exchange.sendRangeCount * exchange.maxRepSeqs * pairTargetBytes + minPassLimit - 1) / minPassLimit);
    // all threads merge a partition, which is about half a pass or smaller, so only a few are too big for a pass
    exchange.partitionsPerRange = std::max(PARTITIONS_PER_THREAD * par.threads, 2 * rangePairPasses);
    exchange.pairRowsMemory = exchange.sendRangeCount * exchange.partitionsPerRange * sizeof(AssignedPairPartition);

    Timer phaseTimer;
    std::string sendValues;
    std::vector<size_t> sendRanges;
    // the own rows of the first own hash ranges that fit and that a node half as fast as the others takes itself
    OwnKmers ownKmers = {NULL, 0, getPassLimit(exchange.loadMemoryLimit - std::min(exchange.loadMemoryLimit, exchange.pairRowsMemory)),
                         std::vector<std::vector<KmerHashSubRange> >(exchange.hashRangeCount), std::vector<size_t>()};
    const size_t sendRangeKmerBytes = globalKmers / exchange.sendRangeCount * sizeof(KmerPositionBit);
    const size_t ownMemory = std::min(ownKmers.limit, exchange.loadMemoryLimit - std::min(exchange.loadMemoryLimit, sendRangeKmerBytes));
    const size_t ownHashRangeBytes = std::max<size_t>(1, globalKmers * receiverKmerBytes / (MMseqsNode::count * exchange.hashRangeCount));
    const size_t ownItems = (exchange.hashRangeCount - MMseqsNode::rank + MMseqsNode::count - 1) / MMseqsNode::count;
    const size_t ownHashRanges = std::min(ownItems * MMseqsNode::count / (2 * MMseqsNode::count - 1), ownMemory / ownHashRangeBytes);
    for (size_t i = 0; i < ownHashRanges; i++) {
        ownKmers.hashRanges.push_back(MMseqsNode::rank + i * MMseqsNode::count);
    }
    if (MMseqsNode::isMaster()) {
        Debug(Debug::INFO) << "Send k-mers in " << exchange.sendRangeCount << " ranges\n";
        // a sent k-mer takes a member id, a rep candidate and about a byte of index codes
        const size_t nodeKmerFileBytes = globalKmers * ((exchange.globalIdBits + exchange.adjKeyCodeBits) / CHAR_BIT + 1) / MMseqsNode::count;
        if (nodeKmerFileBytes > LARGE_FILE_BYTES) {
            Debug(Debug::WARNING) << "The k-mer file of a node can reach " << nodeKmerFileBytes << " bytes, more than the " << LARGE_FILE_BYTES << " bytes of a file on ext4 or of a Lustre object.\n"
                                  << "Run on more nodes, or on Lustre stripe the tmp directory over all OSTs with lfs setstripe -c -1 <tmpDir> before the run.\n";
        }
    }
    const size_t sentBytes = sendKmers<T, includeAdjacency, L>(par, seqDbr, subMat, exchange, sendQueue, representativeBitsMemory, sendRanges, sendValues, ownKmers);
    // no node takes hash ranges before the barrier, so claimNext takes the first own hash ranges in order
    for (size_t i = 0; i < ownKmers.hashRanges.size(); i++) {
        hashQueue.claimNext();
    }
    MMseqsNode::addPhaseTime(sendValues, "sending k-mers", phaseTimer);
    MMseqsNode::addPhaseValue(sendValues, "K-mer bytes sent", SSTR(sentBytes + ownKmers.bytes));
    MMseqsNode::addPhaseValue(sendValues, "K-mer bytes kept in memory", SSTR(ownKmers.bytes));
    MMseqsNode::addPhaseValue(sendValues, "Send ranges", SSTR(sendRanges.size()));
    MMseqsNode::printPhaseValues(sendValues);
    MMseqsNode::barrier(par.db2 + ".kmers.done");
    const double sendWaitSeconds = phaseTimer.getTimediff();
    // every node has passed the send ranges
    sendQueue.removeClaims();
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
    const size_t receivedBytes = receiveKmers(par, seqDbr, exchange, hashQueue, hashRanges, receiveValues, pairBytes, ownKmers);
    Util::releasePages(ownKmers.data, ownKmers.bytes);
    free(ownKmers.data);
    MMseqsNode::addPhaseTime(receiveValues, "receiving k-mers", phaseTimer);
    MMseqsNode::addPhaseValue(receiveValues, "K-mer bytes received", SSTR(receivedBytes));
    MMseqsNode::addPhaseValue(receiveValues, "Pair bytes written", SSTR(pairBytes));
    MMseqsNode::addPhaseValue(receiveValues, "Hash ranges", SSTR(hashRanges.size()));
    MMseqsNode::printPhaseValues(receiveValues);
    MMseqsNode::barrier(par.db2 + ".pairs.done");
    std::string mergeValues;
    MMseqsNode::addPhaseTime(mergeValues, "waiting for the other nodes to receive k-mers", phaseTimer);
    // every node has passed the hash ranges and read the k-mer file of this node
    hashQueue.removeClaims();
    // freeing the blocks and cached pages of a large k-mer file takes long, so it is removed while the pairs are merged
    const std::string kmerFile = par.db2 + ".kmers." + SSTR(MMseqsNode::rank);
    std::thread kmerFileRemoval([&kmerFile]() { FileUtil::remove(kmerFile.c_str()); });
    if (MMseqsNode::isMaster()) {
        FileUtil::remove((par.db2 + ".kmers.done").c_str());
    }

    std::vector<size_t> mergeSplits;
    if (MMseqsNode::isMaster()) {
        Debug(Debug::INFO) << "Merge pairs in " << exchange.sendRangeCount << " splits\n";
    }
    const std::string partitionsData = mergeAssignedPairs(par, seqDbr, exchange, mergeQueue, mergeSplits, mergeValues);
    kmerFileRemoval.join();
    MMseqsNode::addPhaseTime(mergeValues, "merging pairs", phaseTimer);
    MMseqsNode::addPhaseValue(mergeValues, "Merge splits", SSTR(mergeSplits.size()));
    MMseqsNode::printPhaseValues(mergeValues);
    MMseqsNode::publishRunFile(par.db2 + ".partitions." + SSTR(MMseqsNode::rank), partitionsData);
    // freeing a large file takes long, so every node removes its pair file while the master writes the partitions
    const std::string pairFile = par.db2 + ".pairs." + SSTR(MMseqsNode::rank);
    if (MMseqsNode::isMaster()) {
        // every node has merged its send ranges and read the pair files
        const std::string partitions = MMseqsNode::gatherRunFiles(par.db2 + ".partitions.");
        std::string mergeWaitValues;
        MMseqsNode::addPhaseTime(mergeWaitValues, "waiting for the other nodes to merge pairs", phaseTimer);
        MMseqsNode::publishRunFile(par.db2 + ".merged", "");
        std::thread pairFileRemoval([&pairFile]() { FileUtil::remove(pairFile.c_str()); });
        MMseqsNode::printPhaseValues(mergeWaitValues);
        mergeQueue.removeClaims();
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
        mergeQueue.removeClaims();
        FileUtil::remove(pairFile.c_str());
    }
}

template void kmermatcherLinfinity<short, true, unsigned short>(Parameters &, DBReader<DBKeyType> &, BaseMatrix *, size_t);
template void kmermatcherLinfinity<short, false, unsigned short>(Parameters &, DBReader<DBKeyType> &, BaseMatrix *, size_t);
template void kmermatcherLinfinity<int, true, unsigned short>(Parameters &, DBReader<DBKeyType> &, BaseMatrix *, size_t);
template void kmermatcherLinfinity<int, false, unsigned short>(Parameters &, DBReader<DBKeyType> &, BaseMatrix *, size_t);
template void kmermatcherLinfinity<int, true, int>(Parameters &, DBReader<DBKeyType> &, BaseMatrix *, size_t);
template void kmermatcherLinfinity<int, false, int>(Parameters &, DBReader<DBKeyType> &, BaseMatrix *, size_t);

#undef SIZE_T_MAX
