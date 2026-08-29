#ifndef MMSEQS_READLINDB_H
#define MMSEQS_READLINDB_H

#include "Lin8Db.h"
#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include "Debug.h"
#include "FileUtil.h"
#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

struct IoRing {
    struct Read {
        void *into;
        int fd;
        uint64_t offset;
        size_t length;
        size_t required;
    };

    IoRing();
    ~IoRing();

    bool open(unsigned depth);
    bool isOpen() const { return ready; }

    void submit(const std::vector<Read> &reads, const char *what);

private:
    IoRing(const IoRing &);
    IoRing &operator=(const IoRing &);
    void preadAll(const std::vector<Read> &reads, const char *what);

    bool ready;
    unsigned depth;
    void *state;
};

class RunDbReader {
public:
    static const uint64_t VALID_MAGIC;

    // holds the last run so a scan in rank order never repeats the search
    class Cursor {
    public:
        Cursor() : at(0) {}
        size_t at;
    };

    RunDbReader(const std::string &db, const std::string &validPath = "",
                bool withHeaders = false);
    ~RunDbReader();

    void open();
    void close();

    uint64_t getSize() const { return runs.entryCount(); }
    uint64_t getTotalBytes() const { return runs.totalBytes(); }
    const RunTable &getRunTable() const { return runs; }

    uint32_t getSeqLen(uint64_t rank) const { return runs.seqLen(rank); }
    const char *getData(uint64_t rank) const;

    uint32_t getSeqLen(uint64_t rank, Cursor &cursor) const;
    const char *getData(uint64_t rank, Cursor &cursor) const;

    void openBatch(unsigned int threads, size_t arenaBytes);
    size_t loadBatch(const uint64_t *ranks, size_t n, unsigned int thread) const;
    const char *batchAt(unsigned int thread, size_t k) const;

    bool isValid(uint64_t rank) const;
    uint64_t countValid() const;
    bool hasValid() const { return validLoaded; }
    const uint64_t *validWords() const { return valid; }
    size_t validWordCount() const { return validCount; }

    void releaseLengthBlock(size_t suffix);

    uint64_t rankAtByte(uint64_t globalByte) const { return runs.rankAtByte(globalByte); }

    class HeaderStream {
    public:
        HeaderStream(const RunDbReader &owner);
        bool next(const char *&begin, size_t &length);
    private:
        const RunDbReader &owner;
        size_t segment;
        uint64_t left;
        size_t at;
    };

private:
    struct BatchSlot {
        size_t arenaOffset;
        size_t length;
    };
    struct BatchRead {
        size_t arenaOffset;
        uint32_t file;
        uint64_t offset;
        size_t length;
        size_t required;
    };
    struct BatchWorker {
        std::vector<char> arena;
        char *aligned;
        std::vector<BatchSlot> slots;
        std::vector<BatchRead> reads;
        IoRing ring;
        BatchWorker() : aligned(NULL) {}
    };
    // pointers, because a worker owns a ring and a ring is not copyable
    mutable std::vector<BatchWorker *> batch;

    int directOf(uint32_t file) const;
    const char *fileData(uint32_t file, uint64_t offset) const;
    void mapFile(uint32_t file) const;
    void mapHeader(uint32_t file) const;

    std::string db;
    std::string validPath;
    bool withHeaders;
    RunTable runs;
    mutable std::vector<char *> data;
    mutable std::vector<int> dataFd;
    // a second descriptor a file, opened direct, so a batch read takes only its own blocks
    mutable std::vector<int> directFd;
    mutable std::vector<int> headerFd;
    std::vector<size_t> dataSize;
    mutable std::vector<char *> headers;
    std::vector<size_t> headerSize;
    const uint64_t *valid;
    void *validMap;
    size_t validSize;
    size_t validCount;
    bool validLoaded;
};

class RankBitmap {
public:
    RankBitmap() : words(NULL), ranks(0), wordCount(0), appliedRanges(0) {}

    ~RankBitmap() {
        if (words != NULL) {
            munmap(words, bytes());
        }
    }

    // an empty path means nothing is carried between ranges, which is what a single range wants
    void open(const std::string &path, uint64_t ranks) {
        this->ranks = ranks;
        this->cache = path;
        wordCount = (ranks + 63) / 64;
        void *at = mmap(NULL, bytes(), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (at == MAP_FAILED) {
            Debug(Debug::ERROR) << "Cannot reserve " << bytes() << " byte for the taken bitmap\n";
            EXIT(EXIT_FAILURE);
        }
        words = static_cast<uint64_t *>(at);
        if (path.empty() || FileUtil::fileExists(path.c_str()) == false) {
            return;
        }
        FILE *in = FileUtil::openFileOrDie(path.c_str(), "r", true);
        uint64_t header[HEADER_WORDS] = {0, 0, 0};
        if (fread(header, sizeof(uint64_t), HEADER_WORDS, in) != HEADER_WORDS) {
            Debug(Debug::ERROR) << "Cannot read the header of " << path << "\n";
            EXIT(EXIT_FAILURE);
        }
        // a bitmap of another database would otherwise be read as if it described this one
        if (header[0] != MAGIC || header[1] != ranks) {
            Debug(Debug::ERROR) << path << " covers " << header[1] << " sequences and this database "
                                << "holds " << ranks << "\n";
            EXIT(EXIT_FAILURE);
        }
        if (wordCount > 0 && fread(words, sizeof(uint64_t), wordCount, in) != wordCount) {
            Debug(Debug::ERROR) << "Cannot read " << path << "\n";
            EXIT(EXIT_FAILURE);
        }
        if (fclose(in) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << path << "\n";
            EXIT(EXIT_FAILURE);
        }
        appliedRanges = header[2];
    }

    bool taken(uint64_t rank) const { return (words[rank >> 6] >> (rank & 63) & 1) != 0; }
    void take(uint64_t rank) { words[rank >> 6] |= uint64_t(1) << (rank & 63); }
    uint64_t size() const { return ranks; }
    size_t applied() const { return appliedRanges; }

    void catchUpTo(const std::string &decided, size_t range) {
        std::vector<PairRecord> buffer(1u << 16);
        for (size_t at = applied(); at < range; at++) {
            const std::string path = decided + ".0." + SSTR(at);
            FILE *in = fopen(path.c_str(), "r");
            if (in == NULL) {
                Debug(Debug::ERROR) << "Cannot open " << path
                                    << ", which range " << at << " should have decided\n";
                EXIT(EXIT_FAILURE);
            }
            size_t read = 0;
            while ((read = readRecords(buffer.data(), buffer.size(), in)) > 0) {
                for (size_t k = 0; k < read; k++) {
                    take(buffer[k].rep());
                    take(buffer[k].member());
                }
            }
            if (ferror(in) != 0) {
                Debug(Debug::ERROR) << "Cannot read " << path << "\n";
                EXIT(EXIT_FAILURE);
            }
            fclose(in);
        }
        if (range > appliedRanges) {
            appliedRanges = range;
        }
    }

    void save(size_t range) {
        if (cache.empty()) {
            return;
        }
        if (range > appliedRanges) {
            appliedRanges = range;
        }
        const std::string tmp = cache + ".tmp";
        FILE *out = FileUtil::openAndDelete(tmp.c_str(), "w");
        const uint64_t header[HEADER_WORDS] = {MAGIC, ranks, appliedRanges};
        if (fwrite(header, sizeof(uint64_t), HEADER_WORDS, out) != HEADER_WORDS
            || (wordCount > 0 && fwrite(words, sizeof(uint64_t), wordCount, out) != wordCount)) {
            Debug(Debug::ERROR) << "Cannot write " << tmp << "\n";
            EXIT(EXIT_FAILURE);
        }
        if (fclose(out) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << tmp << "\n";
            EXIT(EXIT_FAILURE);
        }
        FileUtil::publishAtomically(tmp, cache);
    }

private:
    RankBitmap(const RankBitmap &);
    RankBitmap &operator=(const RankBitmap &);

    size_t bytes() const { return wordCount * sizeof(uint64_t); }

    static const size_t HEADER_WORDS = 3;
    static const uint64_t MAGIC = 0x4C494E4354414B4Eull;
    uint64_t *words;
    uint64_t ranks;
    size_t wordCount;
    size_t appliedRanges;
    std::string cache;
};

inline bool takeCluster(uint64_t rep, const uint64_t *members, size_t count, RankBitmap &taken,
                        std::vector<PairRecord> &out, uint64_t &assigned) {
    if (taken.taken(rep)) {
        return false;
    }
    taken.take(rep);
    PairRecord line;
    line.set(rep, rep, 0);
    out.push_back(line);
    for (size_t i = 0; i < count; i++) {
        const uint64_t member = members[i];
        if (member == rep || taken.taken(member)) {
            continue;
        }
        taken.take(member);
        line.set(rep, member, 0);
        out.push_back(line);
        assigned++;
    }
    return true;
}

#endif
