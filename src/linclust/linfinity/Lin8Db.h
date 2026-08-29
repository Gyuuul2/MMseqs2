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
#include <fcntl.h>
#include <unistd.h>

class RunTable {
public:
    // one place, so the table cannot hold a rank the k-mer record cannot name. 2.2e12.
    static const unsigned int RANK_BITS = 44;
    static const uint64_t MAX_RANK = (1ull << RANK_BITS) - 1;
    static const uint64_t MAX_BYTE = (1ull << 48) - 1;
    static const uint32_t MAX_FILE = (1u << 16) - 1;
    // a sequence between the two is kept and clustered alone rather than dropped
    static const uint32_t MAX_SEQ_LEN = 32764;
    static const uint32_t MAX_ENTRY_LEN = 65535;

    struct Segment {
        uint64_t rankAndLen;
        uint64_t byteAndFile;
        uint64_t hdrByte;

        uint64_t rankBase() const { return rankAndLen & MAX_RANK; }
        uint32_t seqLen() const { return static_cast<uint32_t>((rankAndLen >> RANK_BITS) & 0xFFFFu); }
        uint64_t byteBase() const { return byteAndFile & MAX_BYTE; }
        uint32_t fileIdx() const { return static_cast<uint32_t>(byteAndFile >> 48); }
        uint64_t hdrBase() const { return hdrByte; }
    };


    RunTable();

    void reserve(size_t segments);
    void append(uint64_t rankBase, uint32_t seqLen, uint64_t byteBase, uint32_t fileIdx,
                uint64_t hdrBase);

    size_t size() const { return segments.size(); }
    uint64_t entryCount() const { return entries; }
    const Segment *data() const { return segments.data(); }
    const Segment &operator[](size_t at) const { return segments[at]; }

    // the caller keeps the returned index as a cursor, so a forward scan never binary searches again
    size_t segmentOf(uint64_t rank) const;
    size_t segmentOfFrom(uint64_t rank, size_t cursor) const;

    uint32_t seqLen(uint64_t rank) const { return segments[segmentOf(rank)].seqLen(); }
    uint32_t maxSeqLen() const { return segments.empty() ? 0 : segments[0].seqLen(); }
    uint32_t fileIdx(uint64_t rank) const { return segments[segmentOf(rank)].fileIdx(); }
    uint64_t fileOffset(uint64_t rank) const { return offsetIn(segmentOf(rank), rank); }
    uint64_t offsetIn(size_t segment, uint64_t rank) const;
    uint64_t rankEnd(size_t segment) const {
        return (segment + 1 < segments.size()) ? segments[segment + 1].rankBase() : entries;
    }

    uint64_t rankAtByte(uint64_t globalByte) const;
    uint64_t byteAtRank(uint64_t rank) const;
    uint64_t totalBytes() const { return bytes; }

    void write(const std::string &path) const;
    void read(const std::string &path);
    // the reader stops at the first gap in db.0, db.1, ..., so it needs the declared count
    unsigned int nodeCount() const { return nodes; }
    unsigned int blocksPerNode() const { return blocks; }
    unsigned int fileCount() const { return nodes * blocks; }
    void setLayout(unsigned int nodeCount, unsigned int blockCount) {
        nodes = nodeCount;
        blocks = blockCount;
    }
    void finish(uint64_t totalEntries);
    void checkLengthsDescend() const;

private:
    std::vector<Segment> segments;
    std::vector<uint64_t> byteStarts;
    uint64_t entries;
    uint64_t bytes;
    unsigned int nodes;
    unsigned int blocks;

    void rebuildByteStarts();
};

struct __attribute__((packed)) KmerRecord {
    uint64_t low;
    uint64_t high;

    static const unsigned int BUCKET_BITS = 13;
    static const unsigned int KEY_BITS = 51 - BUCKET_BITS;
    static const unsigned int RANK_BITS = RunTable::RANK_BITS;
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

    // the rank does not fit beside the key in one word, so it is split across the two
    static const unsigned int RANK_HIGH_BITS = 64 - KEY_BITS;
    static const unsigned int RANK_LOW_BITS = RANK_BITS - RANK_HIGH_BITS;
    static const unsigned int POS_SHIFT = RANK_LOW_BITS + POS_BITS;
    // derived, so a wider key cannot silently drop the adjacency on top of the flag
    static const unsigned int ADJACENT_SHIFT =
        64 - RANK_LOW_BITS - POS_BITS - 1 - ADJACENT_COUNT * ADJACENT_BITS;
    static const uint64_t ADJACENT_ALL = (uint64_t(1) << (ADJACENT_COUNT * ADJACENT_BITS)) - 1;

    static const size_t DISK_BYTES = 16;
    void pack(unsigned char *to) const { memcpy(to, this, DISK_BYTES); }
    void unpack(const unsigned char *from) { memcpy(this, from, DISK_BYTES); }
    // ADJACENT_SHIFT is a subtraction, so an over budget word wraps to a huge shift and still compiles
    static_assert(RANK_LOW_BITS + POS_BITS + 1 + ADJACENT_COUNT * ADJACENT_BITS <= 64,
                  "the k-mer record's second word is over budget");

    void set(uint64_t key, uint64_t rank, uint64_t pos, bool isIdentity, uint64_t adjacent) {
        low = (key << RANK_HIGH_BITS) | (rank >> RANK_LOW_BITS);
        high = ((rank & ((uint64_t(1) << RANK_LOW_BITS) - 1)) << (64 - RANK_LOW_BITS))
               | (pos << (64 - POS_SHIFT)) | (uint64_t(isIdentity ? 1 : 0) << (63 - POS_SHIFT))
               | (adjacent << ADJACENT_SHIFT);
    }

    uint64_t key() const { return low >> RANK_HIGH_BITS; }
    uint64_t rank() const {
        return ((low & ((uint64_t(1) << RANK_HIGH_BITS) - 1)) << RANK_LOW_BITS)
               | (high >> (64 - RANK_LOW_BITS));
    }
    uint64_t pos() const { return (high >> (64 - POS_SHIFT)) & POS_MAX; }
    bool isIdentity() const { return ((high >> (63 - POS_SHIFT)) & 1) != 0; }
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
    // shifted up by half its repRankBlock, so comparing the field unsigned compares the diagonals signed
    static const int DIAGONAL_BIAS = 1 << (DIAGONAL_BITS - 1);

    static const size_t DEFAULT_REP_RANK_BLOCKS = 1024;
    static const size_t MAX_REP_RANK_BLOCKS = 4096;

    static size_t repRankBlockOf(uint64_t rep, uint64_t ranks, size_t repRankBlocks) {
        return ranks == 0 ? 0 : std::min(rep * repRankBlocks / ranks, repRankBlocks - 1);
    }

    static uint64_t firstRankOf(size_t repRankBlock, uint64_t ranks, size_t repRankBlocks) {
        return (repRankBlock * ranks + repRankBlocks - 1) / repRankBlocks;
    }

    static const unsigned int REP_RANK_SUB_BLOCK_BITS = 8;
    // repRankSubBlockOf multiplies a rank by both counts before dividing, unsigned
    static_assert(RunTable::RANK_BITS + 12 + REP_RANK_SUB_BLOCK_BITS <= 64,
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
    // pack drops the spare bits, so a field that grows into them is silently truncated on the way out
    static_assert(2 * RANK_BITS + DIAGONAL_BITS <= 8 * DISK_BYTES,
                  "a pair record's fields are wider than the bytes that reach the disk");
};

class KSeqWrapper;

struct InputSplit {
    size_t file;
    uint64_t from;   // where to start looking for the first record
    uint64_t until;  // where to stop starting new records; the last one is finished past it
    bool compressed;  // whether from and until are compressed offsets at frame boundaries
    uint64_t bytes() const { return until - from; }
};

std::vector<InputSplit> planInputSplits(const std::vector<std::string> &filenames, size_t want);

class InputSplitReader {
public:
    // one read a thread at a time, so it has to be large enough to be worth a shared filesystem
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
    size_t at;        // where in the buffer the next unread byte is
    size_t filled;    // how much of the buffer holds file bytes
    uint64_t readTo;  // how far into the file the buffer reaches
    bool started;
    std::string header;
    std::string sequence;
    KSeqWrapper *whole;
    void *stream;       // a ZSTD_DStream when the input is compressed
    std::vector<char> packed;
    size_t packedAt;
    size_t packedFilled;
    uint64_t packedTo;  // how far into the compressed file the staging buffer reaches
    uint64_t fileSize;  // asked once, not once a buffer: over shared storage a stat is a round trip
    uint64_t endsAt;
};


template <typename Record>
class BucketWriter {
public:
    BucketWriter(const std::string &prefix, size_t buckets, unsigned int threads, size_t budget)
        : prefix(prefix), buckets(buckets), files(buckets, -1), written(buckets, 0),
          offsets(buckets, 0), staged(threads, std::vector<std::vector<Record> >(buckets)) {
        // a thread's buffer is small and lock free, the bucket's is shared and is what reaches the disk
        const size_t share = budget / STAGING_SHARE;
        depth = std::max<size_t>(256, (share / 4) / (threads * buckets * sizeof(Record)));
        depth = std::min<size_t>(depth, FLUSH_BYTES / sizeof(Record));
        poolDepth = std::max<size_t>(depth, (share - share / 4) / (buckets * sizeof(Record)));
        poolDepth = std::min<size_t>(poolDepth, POOL_BYTES / sizeof(Record));
        pooled.resize(buckets);
        gate.assign(buckets, 0);
        held = (threads * depth + poolDepth + depth) * buckets * sizeof(Record);
    }

    size_t bytesHeld() const { return held; }

    ~BucketWriter() { close(); }

    void openAt(const std::vector<uint64_t> &keep) {
        for (size_t i = 0; i < buckets; i++) {
            const std::string path = name(i);
            files[i] = open(path.c_str(), O_WRONLY | O_CREAT, 0666);
            if (files[i] < 0) {
                Debug(Debug::ERROR) << "Cannot open " << path << " for writing\n";
                if (errno == EMFILE || errno == ENFILE) {
                    Debug(Debug::ERROR) << buckets << " buckets need that many open files at once."
                                        << " Raise the limit with ulimit -n\n";
                }
                EXIT(EXIT_FAILURE);
            }
            offsets[i] = keep[i] * Record::DISK_BYTES;
            if (ftruncate(files[i], static_cast<off_t>(offsets[i])) != 0) {
                Debug(Debug::ERROR) << "Cannot cut " << path << " to " << offsets[i] << " byte\n";
                EXIT(EXIT_FAILURE);
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
                sync_file_range(files[bucket], 0, 0, SYNC_FILE_RANGE_WRITE);
            }
        }
#pragma omp parallel for schedule(dynamic, 16) num_threads(threads)
        for (size_t bucket = 0; bucket < buckets; bucket++) {
            if (written[bucket] > 0 && fdatasync(files[bucket]) != 0) {
                Debug(Debug::ERROR) << "Cannot flush " << name(bucket) << " to storage\n";
                EXIT(EXIT_FAILURE);
            }
        }
    }

    void close() {
        for (size_t i = 0; i < buckets; i++) {
            if (files[i] >= 0 && ::close(files[i]) != 0) {
                Debug(Debug::ERROR) << "Cannot close " << name(i) << "\n";
                EXIT(EXIT_FAILURE);
            }
            files[i] = -1;
        }
    }

    const std::vector<uint64_t> &chunkCounts() const { return written; }
    void resetCounts() { std::fill(written.begin(), written.end(), 0); }

private:
    std::string name(size_t bucket) const { return prefix + "." + SSTR(bucket); }

    // holds the bucket across the write, so one write a bucket is in flight at a time
    void drain(size_t bucket, std::vector<Record> &buffer) {
        if (buffer.empty()) {
            return;
        }
        while (__sync_lock_test_and_set(&gate[bucket], 1) != 0) {
            // atomic, or the compiler hoists the load out and spins on a value from before
            while (__atomic_load_n(&gate[bucket], __ATOMIC_RELAXED) != 0) {
            }
        }
        std::vector<Record> &pool = pooled[bucket];
        if (pool.capacity() < poolDepth + depth) {
            pool.reserve(poolDepth + depth);
        }
        pool.insert(pool.end(), buffer.begin(), buffer.end());
        if (pool.size() >= poolDepth) {
            flush(bucket, pool);
        }
        __sync_lock_release(&gate[bucket]);
        buffer.clear();
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
        const ssize_t wrote = pwrite(files[bucket], bytesOut, bytes, static_cast<off_t>(at));
        if (wrote < 0 || static_cast<size_t>(wrote) != bytes) {
            Debug(Debug::ERROR) << "Cannot write " << bytes << " byte to " << name(bucket) << "\n";
            EXIT(EXIT_FAILURE);
        }
        buffer.clear();
    }

    std::string prefix;
    size_t buckets;
    static const size_t STAGING_SHARE = 8;
    static const size_t FLUSH_BYTES = 4 * 1024 * 1024;
    static const size_t POOL_BYTES = 16 * 1024 * 1024;
    size_t depth;
    size_t poolDepth;
    std::vector<std::vector<Record> > pooled;
    std::vector<int> gate;
    size_t held;
    std::vector<int> files;
    std::vector<uint64_t> written;
    std::vector<uint64_t> offsets;
    std::vector<std::vector<std::vector<Record> > > staged;
};

template <class Record>
size_t writeRecords(const Record *from, size_t count, FILE *out) {
    if (Record::DISK_BYTES == sizeof(Record)) {
        return fwrite(from, sizeof(Record), count, out);
    }
    std::vector<unsigned char> packed(count * Record::DISK_BYTES);
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
    std::vector<unsigned char> packed(count * Record::DISK_BYTES);
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

// a pid alone repeats across machines, so a temporary on shared storage needs the host as well
inline std::string uniqueTmpSuffix() {
    char host[HOST_NAME_MAX + 1];
    memset(host, 0, sizeof(host));
    gethostname(host, HOST_NAME_MAX);
    return std::string(host) + "." + SSTR(getpid());
}

inline std::string nodeDonePath(const std::string &path, unsigned int node) {
    return path + "." + SSTR(node) + ".done";
}
void markNodeDone(const std::string &path, unsigned int node);

void dropConsumed(const std::string &prefix, unsigned int nodes, size_t from, size_t until,
                  size_t step);
void requireEveryNodeDone(const std::string &path, unsigned int nodes);

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
