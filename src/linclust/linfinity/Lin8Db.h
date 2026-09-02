#ifndef MMSEQS_CREATELINDB_H
#define MMSEQS_CREATELINDB_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <cstring>
#include <vector>
#include <algorithm>
#include "Debug.h"
#include "FileUtil.h"
#include "Util.h"
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <sys/resource.h>
#ifdef OPENMP
#include <omp.h>
#endif

class SequenceLocator {
public:
    static const unsigned int RANK_BITS = 44;
    static const uint64_t MAX_RANK = (1ull << RANK_BITS) - 1;
    static const uint64_t MAX_BYTE = (1ull << 48) - 1;
    static const uint32_t MAX_FILE = (1u << 16) - 1;
    static const uint32_t MAX_SEQ_LEN = 32764;
    static const uint32_t MAX_ENTRY_LEN = 65535;

    struct LengthRun {
        uint64_t rankAndLen;
        uint64_t byteAndFile;
        uint64_t hdrByte;

        uint64_t rankBase() const { return rankAndLen & MAX_RANK; }
        uint32_t seqLen() const { return static_cast<uint32_t>((rankAndLen >> RANK_BITS) & 0xFFFFu); }
        uint64_t byteBase() const { return byteAndFile & MAX_BYTE; }
        uint32_t fileIdx() const { return static_cast<uint32_t>(byteAndFile >> 48); }
        uint64_t hdrBase() const { return hdrByte; }
    };

    SequenceLocator();

    void reserve(size_t runs);
    void append(uint64_t rankBase, uint32_t seqLen, uint64_t byteBase, uint32_t fileIdx,
                uint64_t hdrBase);

    size_t size() const { return runs.size(); }
    uint64_t entryCount() const { return entries; }
    const LengthRun *data() const { return runs.data(); }
    const LengthRun &operator[](size_t at) const { return runs[at]; }

    size_t runOf(uint64_t rank) const;
    size_t runOfFrom(uint64_t rank, size_t cursor) const;

    uint32_t seqLen(uint64_t rank) const { return runs[runOf(rank)].seqLen(); }
    uint32_t maxSeqLen() const { return runs.empty() ? 0 : runs[0].seqLen(); }
    uint32_t fileIdx(uint64_t rank) const { return runs[runOf(rank)].fileIdx(); }
    uint64_t fileOffset(uint64_t rank) const { return offsetIn(runOf(rank), rank); }
    uint64_t offsetIn(size_t segment, uint64_t rank) const;
    uint64_t rankEnd(size_t segment) const {
        return (segment + 1 < runs.size()) ? runs[segment + 1].rankBase() : entries;
    }

    uint64_t rankAtByte(uint64_t globalByte) const;
    uint64_t byteAtRank(uint64_t rank) const;
    uint64_t totalBytes() const { return bytes; }

    void write(const std::string &path) const;
    void read(const std::string &path);
    unsigned int nodeCount() const { return nodes; }
    unsigned int filesPerNode() const { return perNodeFiles; }
    unsigned int fileCount() const { return nodes * perNodeFiles; }
    void setLayout(unsigned int nodeCount, unsigned int filesPerNode) {
        nodes = nodeCount;
        perNodeFiles = filesPerNode;
    }
    void finish(uint64_t totalEntries);
    void checkLengthsDescend() const;

private:
    std::vector<LengthRun> runs;
    std::vector<uint64_t> byteStarts;
    uint64_t entries;
    uint64_t bytes;
    unsigned int nodes;
    unsigned int perNodeFiles;

    void rebuildByteStarts();
};

struct __attribute__((packed)) KmerRecord {
    uint64_t low;
    uint64_t high;

    static const unsigned int BUCKET_BITS = 13;
    static const unsigned int KEY_BITS = 51 - BUCKET_BITS;
    static const unsigned int RANK_BITS = SequenceLocator::RANK_BITS;
    static const unsigned int POS_BITS = 15;

    static const unsigned int SUB_BUCKET_BITS = 8;
    static const size_t SUB_BUCKET_COUNT = size_t(1) << SUB_BUCKET_BITS;

    static const unsigned int ADJACENT_COUNT = 6;
    static const unsigned int ADJACENT_BITS = 5;

    static const uint64_t BUCKET_COUNT = uint64_t(1) << BUCKET_BITS;
    static const uint64_t KEY_MAX = (uint64_t(1) << KEY_BITS) - 1;
    static const uint64_t RANK_MAX = (uint64_t(1) << RANK_BITS) - 1;
    static const uint64_t POS_MAX = (uint64_t(1) << POS_BITS) - 1;
    static const uint64_t ADJACENT_MAX = (uint64_t(1) << ADJACENT_BITS) - 1;

    static const unsigned int RANK_HIGH_BITS = 64 - KEY_BITS;
    static const unsigned int RANK_LOW_BITS = RANK_BITS - RANK_HIGH_BITS;
    static const unsigned int POS_SHIFT = RANK_LOW_BITS + POS_BITS;
    static const unsigned int ADJACENT_SHIFT =
        64 - RANK_LOW_BITS - POS_BITS - ADJACENT_COUNT * ADJACENT_BITS;
    static const uint64_t ADJACENT_ALL = (uint64_t(1) << (ADJACENT_COUNT * ADJACENT_BITS)) - 1;

    static const size_t DISK_BYTES = 16;
    void pack(unsigned char *to) const { memcpy(to, this, DISK_BYTES); }
    void unpack(const unsigned char *from) { memcpy(this, from, DISK_BYTES); }
    static_assert(RANK_LOW_BITS + POS_BITS + ADJACENT_COUNT * ADJACENT_BITS <= 64,
                  "the k-mer record's second word is over budget");

    void set(uint64_t key, uint64_t rank, uint64_t pos, uint64_t adjacent) {
        low = (key << RANK_HIGH_BITS) | (rank >> RANK_LOW_BITS);
        high = ((rank & ((uint64_t(1) << RANK_LOW_BITS) - 1)) << (64 - RANK_LOW_BITS))
               | (pos << (64 - POS_SHIFT)) | (adjacent << ADJACENT_SHIFT);
    }

    uint64_t key() const { return low >> RANK_HIGH_BITS; }
    uint64_t rank() const {
        return ((low & ((uint64_t(1) << RANK_HIGH_BITS) - 1)) << RANK_LOW_BITS)
               | (high >> (64 - RANK_LOW_BITS));
    }
    uint64_t pos() const { return (high >> (64 - POS_SHIFT)) & POS_MAX; }
    uint64_t adjacent() const { return (high >> ADJACENT_SHIFT) & ADJACENT_ALL; }
    unsigned int adjacentAt(unsigned int slot) const {
        return (high >> (ADJACENT_SHIFT + slot * ADJACENT_BITS)) & ADJACENT_MAX;
    }

    unsigned int subBucket() const {
        return static_cast<unsigned int>(key() >> (KEY_BITS - SUB_BUCKET_BITS));
    }

    static bool byKeyAndRank(const KmerRecord &first, const KmerRecord &second) {
        if (first.low != second.low) {
            return first.low < second.low;
        }
        return first.high < second.high;
    }
};

struct __attribute__((packed)) PairRecord {
    uint64_t low;
    uint64_t high;

    static const unsigned int RANK_BITS = KmerRecord::RANK_BITS;
    static const unsigned int DIAGONAL_BITS = 16;
    static const int DIAGONAL_BIAS = 1 << (DIAGONAL_BITS - 1);

    static const size_t DEFAULT_REP_RANK_BLOCKS = 512;
    static const size_t MAX_REP_RANK_BLOCKS = 4096;
    static_assert(MAX_REP_RANK_BLOCKS == (size_t(1) << 12),
                  "the fine repRankBlock packing below reserves 12 bits for the repRankBlock count");

    static size_t repRankBlockOf(uint64_t rep, uint64_t ranks, size_t repRankBlocks) {
        return ranks == 0 ? 0 : std::min(rep * repRankBlocks / ranks, repRankBlocks - 1);
    }

    static uint64_t firstRankOf(size_t repRankBlock, uint64_t ranks, size_t repRankBlocks) {
        return (repRankBlock * ranks + repRankBlocks - 1) / repRankBlocks;
    }

    static const unsigned int REP_RANK_SUB_BLOCK_BITS = 8;
    static_assert(SequenceLocator::RANK_BITS + 12 + REP_RANK_SUB_BLOCK_BITS <= 64,
                  "a rank times the repRankBlock count times the sub repRankBlock count is over 64 bits");
    static const size_t REP_RANK_SUB_BLOCKS = size_t(1) << REP_RANK_SUB_BLOCK_BITS;
    static size_t fineOf(uint64_t rep, uint64_t ranks, size_t repRankBlocks) {
        return ranks == 0 ? 0 : std::min(rep * repRankBlocks * REP_RANK_SUB_BLOCKS / ranks,
                                         repRankBlocks * REP_RANK_SUB_BLOCKS - 1);
    }
    static size_t repRankSubBlockOf(uint64_t rep, uint64_t ranks, size_t repRankBlocks) {
        return fineOf(rep, ranks, repRankBlocks) % REP_RANK_SUB_BLOCKS;
    }

    static const unsigned int MEMBER_HIGH_BITS = 64 - RANK_BITS;
    static const unsigned int MEMBER_LOW_BITS = RANK_BITS - MEMBER_HIGH_BITS;
    static const unsigned int DIAGONAL_SHIFT = 64 - MEMBER_LOW_BITS - DIAGONAL_BITS;

    void set(uint64_t rep, uint64_t member, int diagonal) {
        low = (rep << MEMBER_HIGH_BITS) | (member >> MEMBER_LOW_BITS);
        high = ((member & ((uint64_t(1) << MEMBER_LOW_BITS) - 1)) << (64 - MEMBER_LOW_BITS))
               | (uint64_t(uint16_t(diagonal + DIAGONAL_BIAS)) << DIAGONAL_SHIFT);
    }

    uint64_t rep() const { return low >> MEMBER_HIGH_BITS; }
    uint64_t member() const {
        return ((low & ((uint64_t(1) << MEMBER_HIGH_BITS) - 1)) << MEMBER_LOW_BITS)
               | (high >> (64 - MEMBER_LOW_BITS));
    }
    int diagonal() const {
        return int((high >> DIAGONAL_SHIFT) & 0xFFFF) - DIAGONAL_BIAS;
    }

    static bool sameRepAndMember(const PairRecord &first, const PairRecord &second) {
        return first.rep() == second.rep() && first.member() == second.member();
    }

    static bool byRepAndMember(const PairRecord &first, const PairRecord &second) {
        if (first.low != second.low) {
            return first.low < second.low;
        }
        return first.high < second.high;
    }

    static const size_t DISK_BYTES = 13;
    void pack(unsigned char *to) const {
        const uint64_t top = high >> SPARE_BITS;
        memcpy(to, &low, sizeof(uint64_t));
        memcpy(to + sizeof(uint64_t), &top, DISK_BYTES - sizeof(uint64_t));
    }
    void unpack(const unsigned char *from) {
        uint64_t top = 0;
        memcpy(&low, from, sizeof(uint64_t));
        memcpy(&top, from + sizeof(uint64_t), DISK_BYTES - sizeof(uint64_t));
        high = top << SPARE_BITS;
    }

private:
    static const unsigned int SPARE_BITS = 8 * (16 - DISK_BYTES);
    static_assert(2 * RANK_BITS + DIAGONAL_BITS <= 8 * DISK_BYTES,
                  "a pair record's fields are wider than the bytes that reach the disk");
};

class KSeqWrapper;

struct InputSplit {
    size_t file;
    uint64_t from;
    uint64_t until;
    bool compressed;
    uint64_t bytes() const { return until - from; }
};

std::vector<InputSplit> planInputSplits(const std::vector<std::string> &filenames, size_t want);

class InputSplitReader {
public:
    static const size_t PACKED_READ_BYTES = 4u << 20;

    InputSplitReader(const std::string &filename, const InputSplit &chunk);
    ~InputSplitReader();

    bool next(const char *&header, size_t &headerLength, const char *&sequence, size_t &length);

private:
    InputSplitReader(const InputSplitReader &);
    InputSplitReader &operator=(const InputSplitReader &);

    bool fill();

    int fd;
    InputSplit chunk;
    std::string name;
    std::vector<char> buffer;
    size_t at;
    size_t filled;
    uint64_t readTo;
    bool started;
    std::string header;
    std::string sequence;
    KSeqWrapper *whole;
    void *stream;
    std::vector<char> packed;
    size_t packedAt;
    size_t packedFilled;
    uint64_t packedTo;
    uint64_t fileSize;
    uint64_t endsAt;
};

template <typename Record>
class BucketWriter {
public:
    BucketWriter(const std::string &prefix, size_t buckets, unsigned int threads, size_t budget)
        : prefix(prefix), buckets(buckets), written(buckets, 0),
          offsets(buckets, 0), staged(threads, std::vector<std::vector<Record> >(buckets)) {
        const size_t share = budget / STAGING_SHARE;
        depth = std::max<size_t>(256, (share / 4) / (threads * buckets * sizeof(Record)));
        depth = std::min<size_t>(depth, FLUSH_BYTES / sizeof(Record));
        poolDepth = std::max<size_t>(depth, (share - share / 4) / (buckets * sizeof(Record)));
        poolDepth = std::min<size_t>(poolDepth, POOL_BYTES / sizeof(Record));
        pooled.resize(buckets);
        leaving.resize(threads);
        gate.assign(buckets, 0);
        held = (threads * depth + poolDepth + depth) * buckets * sizeof(Record);
    }

    size_t bytesHeld() const { return held; }

    ~BucketWriter() { close(); }

    int openBucket(size_t bucket) const {
        const std::string path = name(bucket);
        const int fd = open(path.c_str(), O_WRONLY | O_CREAT, 0666);
        if (fd < 0) {
            Debug(Debug::ERROR) << "Cannot open " << path << " for writing, error " << errno << "\n";
            EXIT(EXIT_FAILURE);
        }
        return fd;
    }

    void closeBucket(int fd, size_t bucket) const {
        if (::close(fd) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << name(bucket) << "\n";
            EXIT(EXIT_FAILURE);
        }
    }

    static const size_t DESCRIPTOR_SLACK = 512;

    size_t lendDescriptors() const {
        struct rlimit limit;
        if (getrlimit(RLIMIT_NOFILE, &limit) != 0) {
            return 0;
        }
        const rlim_t want = (rlim_t) buckets + DESCRIPTOR_SLACK;
        if (limit.rlim_cur < want) {
            limit.rlim_cur = std::min(want, limit.rlim_max);
            if (setrlimit(RLIMIT_NOFILE, &limit) != 0 || getrlimit(RLIMIT_NOFILE, &limit) != 0) {
                return 0;
            }
        }
        const size_t lent = limit.rlim_cur > DESCRIPTOR_SLACK
                                ? std::min<size_t>(buckets, (size_t) limit.rlim_cur - DESCRIPTOR_SLACK)
                                : 0;
        Debug(Debug::INFO) << "Keeping " << lent << " of " << buckets << " buckets open"
                           << (lent < buckets ? "; the rest are opened for the write" : "") << "\n";
        return lent;
    }

    int fdOf(size_t bucket) const { return kept[bucket] >= 0 ? kept[bucket] : openBucket(bucket); }

    void release(int fd, size_t bucket) const {
        if (kept[bucket] < 0) {
            closeBucket(fd, bucket);
        }
    }

    void openAt(const std::vector<uint64_t> &keep) {
        const size_t lent = lendDescriptors();
        kept.assign(buckets, -1);
        for (size_t i = 0; i < buckets; i++) {
            const int fd = openBucket(i);
            offsets[i] = keep[i] * Record::DISK_BYTES;
            if (ftruncate(fd, static_cast<off_t>(offsets[i])) != 0) {
                Debug(Debug::ERROR) << "Cannot cut " << name(i) << " to " << offsets[i] << " byte\n";
                EXIT(EXIT_FAILURE);
            }
            if (i < lent) {
                kept[i] = fd;
            } else {
                closeBucket(fd, i);
            }
            written[i] = 0;
        }
    }

    void add(unsigned int thread, const Record &record, size_t bucket) {
        std::vector<Record> &buffer = staged[thread][bucket];
        if (buffer.capacity() < depth) {
            buffer.reserve(depth);
        }
        buffer.push_back(record);
        if (buffer.size() >= depth) {
            drain(bucket, buffer);
        }
    }

    void flushAll(unsigned int threads) {
#pragma omp parallel for schedule(dynamic, 16) num_threads(threads)
        for (size_t bucket = 0; bucket < buckets; bucket++) {
            for (size_t thread = 0; thread < staged.size(); thread++) {
                drain(bucket, staged[thread][bucket]);
            }
            flush(bucket, pooled[bucket]);
        }
    }

    void endChunk(unsigned int threads) {
#pragma omp parallel for schedule(dynamic, 16) num_threads(threads)
        for (size_t bucket = 0; bucket < buckets; bucket++) {
            for (size_t thread = 0; thread < staged.size(); thread++) {
                drain(bucket, staged[thread][bucket]);
            }
            flush(bucket, pooled[bucket]);
            if (written[bucket] > 0) {
                const int fd = fdOf(bucket);
                sync_file_range(fd, 0, 0, SYNC_FILE_RANGE_WRITE);
                release(fd, bucket);
            }
        }
#pragma omp parallel for schedule(dynamic, 16) num_threads(threads)
        for (size_t bucket = 0; bucket < buckets; bucket++) {
            if (written[bucket] == 0) {
                continue;
            }
            const int fd = fdOf(bucket);
            if (fdatasync(fd) != 0) {
                Debug(Debug::ERROR) << "Cannot flush " << name(bucket) << " to storage\n";
                EXIT(EXIT_FAILURE);
            }
            release(fd, bucket);
        }
    }

    void close() {
        for (size_t i = 0; i < kept.size(); i++) {
            if (kept[i] >= 0) {
                closeBucket(kept[i], i);
                kept[i] = -1;
            }
        }
    }

    const std::vector<uint64_t> &chunkCounts() const { return written; }
    void resetCounts() { std::fill(written.begin(), written.end(), 0); }

private:
    std::string name(size_t bucket) const { return prefix + "." + SSTR(bucket); }

    void drain(size_t bucket, std::vector<Record> &buffer) {
        if (buffer.empty()) {
            return;
        }
        while (__sync_lock_test_and_set(&gate[bucket], 1) != 0) {
            while (__atomic_load_n(&gate[bucket], __ATOMIC_RELAXED) != 0) {
            }
        }
        std::vector<Record> &pool = pooled[bucket];
        if (pool.capacity() < poolDepth + depth) {
            pool.reserve(poolDepth + depth);
        }
        pool.insert(pool.end(), buffer.begin(), buffer.end());
        size_t here = 0;
#ifdef OPENMP
        here = (size_t) omp_get_thread_num();
#endif
        if (here >= leaving.size()) {
            if (pool.size() >= poolDepth) {
                flush(bucket, pool);
            }
            __sync_lock_release(&gate[bucket]);
            buffer.clear();
            return;
        }
        std::vector<Record> &outbound = leaving[here];
        if (pool.size() >= poolDepth) {
            outbound.swap(pool);
        }
        __sync_lock_release(&gate[bucket]);
        buffer.clear();
        flush(bucket, outbound);
    }

    void flush(size_t bucket, std::vector<Record> &buffer) {
        if (buffer.empty()) {
            return;
        }
        const size_t bytes = buffer.size() * Record::DISK_BYTES;
        std::vector<unsigned char> packed;
        const void *bytesOut = buffer.data();
        if (Record::DISK_BYTES != sizeof(Record)) {
            packed.resize(bytes);
            for (size_t i = 0; i < buffer.size(); i++) {
                buffer[i].pack(&packed[i * Record::DISK_BYTES]);
            }
            bytesOut = packed.data();
        }
        const uint64_t at = __sync_fetch_and_add(&offsets[bucket], bytes);
        __sync_fetch_and_add(&written[bucket], buffer.size());
        const int fd = fdOf(bucket);
        const ssize_t wrote = pwrite(fd, bytesOut, bytes, static_cast<off_t>(at));
        if (wrote < 0 || static_cast<size_t>(wrote) != bytes) {
            Debug(Debug::ERROR) << "Cannot write " << bytes << " byte to " << name(bucket) << "\n";
            EXIT(EXIT_FAILURE);
        }
        release(fd, bucket);
        buffer.clear();
    }

    static const size_t STAGING_SHARE = 8;
    static const size_t FLUSH_BYTES = 4 * 1024 * 1024;
    static const size_t POOL_BYTES = 16 * 1024 * 1024;
    std::string prefix;
    size_t buckets;
    size_t depth;
    size_t poolDepth;
    std::vector<std::vector<Record> > pooled;
    std::vector<std::vector<Record> > leaving;
    std::vector<int> gate;
    mutable std::vector<int> kept;
    size_t held;
    std::vector<uint64_t> written;
    std::vector<uint64_t> offsets;
    std::vector<std::vector<std::vector<Record> > > staged;
};

template <class Record>
size_t writeRecords(const Record *from, size_t count, FILE *out) {
    if (Record::DISK_BYTES == sizeof(Record)) {
        return fwrite(from, sizeof(Record), count, out);
    }
    thread_local std::vector<unsigned char> packed;
    packed.resize(count * Record::DISK_BYTES);
    for (size_t i = 0; i < count; i++) {
        from[i].pack(&packed[i * Record::DISK_BYTES]);
    }
    return fwrite(packed.data(), Record::DISK_BYTES, count, out);
}

template <class Record>
size_t readRecords(Record *into, size_t count, FILE *in) {
    if (Record::DISK_BYTES == sizeof(Record)) {
        return fread(into, sizeof(Record), count, in);
    }
    thread_local std::vector<unsigned char> packed;
    packed.resize(count * Record::DISK_BYTES);
    const size_t read = fread(packed.data(), Record::DISK_BYTES, count, in);
    for (size_t i = 0; i < read; i++) {
        into[i].unpack(&packed[i * Record::DISK_BYTES]);
    }
    return read;
}

void publishAllAtomically(std::vector<std::pair<std::string, std::string> > &pending,
                          unsigned int threads);

void requireArena(const std::string &what, size_t bytes, size_t budget,
                  const std::string &narrower);
static const uint64_t PUBLISH_BATCH_BYTES = 4ull * 1024 * 1024 * 1024;
static const size_t PUBLISH_BATCH_FILES = 64;

void writeBucketManifest(const std::string &path, const std::vector<uint64_t> &counts,
                         const std::string &spanKey, uint64_t spanBegin, uint64_t spanEnd);

inline std::string uniqueTmpSuffix() {
    char host[HOST_NAME_MAX + 1];
    memset(host, 0, sizeof(host));
    gethostname(host, HOST_NAME_MAX);
    return std::string(host) + "." + SSTR(getpid());
}

inline std::string nodeDonePath(const std::string &path, unsigned int node) {
    return path + "." + SSTR(node) + ".done";
}

// the alignments beside a pair file, one text line per pair, accepted ones under a "#\trep" line
inline std::string alnTextPath(const std::string &prefix, unsigned int node, size_t block) {
    return prefix + "_text." + SSTR(node) + "." + SSTR(block);
}
static const char ALN_TEXT_REP_MARK = '#';

class AlnTextReader {
public:
    AlnTextReader(const std::string &prefix, unsigned int node, size_t block, bool required)
        : path(alnTextPath(prefix, node, block)), line(NULL), cap(0) {
        file = fopen(path.c_str(), "r");
        if (file == NULL && required) {
            Debug(Debug::ERROR) << "Cannot open " << path
                                << ", which the aligning pass should have written\n";
            EXIT(EXIT_FAILURE);
        }
        if (file != NULL) {
            setvbuf(file, NULL, _IOFBF, 1u << 20);
        }
    }

    ~AlnTextReader() {
        if (file != NULL) {
            fclose(file);
        }
        free(line);
    }

    bool next(char *&begin, size_t &length) {
        if (file == NULL) {
            return false;
        }
        const ssize_t got = getline(&line, &cap, file);
        if (got > 0) {
            begin = line;
            length = (size_t) got;
            return true;
        }
        if (ferror(file) != 0) {
            Debug(Debug::ERROR) << "Cannot read " << path << "\n";
            EXIT(EXIT_FAILURE);
        }
        return false;
    }

    const std::string &name() const { return path; }

private:
    AlnTextReader(const AlnTextReader &);
    AlnTextReader &operator=(const AlnTextReader &);

    std::string path;
    FILE *file;
    char *line;
    size_t cap;
};
void markNodeDone(const std::string &path, unsigned int node);

// a counter rewritten in place, because a new name on NFS is a negative dentry another node caches
void publishProgress(const std::string &path, uint64_t value);

void dropConsumed(const std::string &prefix, unsigned int nodes, size_t from, size_t until,
                  size_t step);
void requireEveryNodeDone(const std::string &path, unsigned int nodes);
void waitNodeDone(const std::string &path, unsigned int node, unsigned int limitSeconds);
void waitEveryNodeDone(const std::string &path, unsigned int nodes, unsigned int limitSeconds);

void writeSubBucketCounts(const std::string &path, const std::vector<uint64_t> &base,
                          const std::vector<std::vector<uint64_t> > &perThread, size_t chunks);
std::vector<uint64_t> readSubBucketCounts(const std::string &path, size_t entries, size_t &chunks);

class BucketCounts {
public:
    BucketCounts(const std::string &prefix, unsigned int nodes, size_t subBuckets,
                 size_t buckets);
    ~BucketCounts();
    std::vector<uint64_t> of(size_t bucket) const;
    std::vector<uint64_t> of(size_t bucket, size_t node) const;

private:
    BucketCounts(const BucketCounts &);
    BucketCounts &operator=(const BucketCounts &);
    std::vector<FILE *> files;
    std::vector<std::string> paths;
    size_t subBuckets;
};

size_t readBucketManifests(const std::string &prefix, size_t chunks, std::vector<uint64_t> &into,
                           uint64_t *resumeAt = NULL);

template <typename T>
class RawArray {
public:
    RawArray() : items(NULL), count(0) {}
    ~RawArray() { delete[] items; }

    void resize(size_t n) {
        delete[] items;
        items = new T[n];
        count = n;
    }

    T *begin() const { return items; }
    T &operator[](size_t i) const { return items[i]; }
    size_t size() const { return count; }

private:
    RawArray(const RawArray &);
    RawArray &operator=(const RawArray &);
    T *items;
    size_t count;
};

#endif
