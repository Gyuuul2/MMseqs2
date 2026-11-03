#ifndef MMSEQS_NODE_H
#define MMSEQS_NODE_H

#include "IndexTypes.h"

#include <cstdio>
#include <ctime>
#include <string>
#include <vector>

class Timer;
template <typename T> class DBReader;

// a member of a cluster with its representative, the record of most exchanges between nodes
struct ClusterPair {
    DBKeyType repKey;
    DBKeyType memberKey;

    static bool compareByMember(const ClusterPair &first, const ClusterPair &second) {
        return first.memberKey < second.memberKey;
    }

    // the size of the pair at record, 0 if bytes do not hold all of it
    static size_t recordSize(const char *, size_t bytes) {
        return (bytes >= sizeof(ClusterPair)) ? sizeof(ClusterPair) : 0;
    }

    // the representative itself first, so it is the first line of its entry
    static bool compareByRep(const ClusterPair &first, const ClusterPair &second) {
        if (first.repKey != second.repKey) {
            return first.repKey < second.repKey;
        }
        if ((first.memberKey == first.repKey) != (second.memberKey == second.repKey)) {
            return first.memberKey == first.repKey;
        }
        return first.memberKey < second.memberKey;
    }
};

class MMseqsNode {
public:
    static const unsigned int MASTER = 0;

    static unsigned int rank;
    static unsigned int count;

    static void init();
    static inline bool isMaster() {
        return rank == MASTER;
    }

    // sleeps before poll number polls of a file that another node publishes: ten milliseconds, twice as long each poll, up to 10 << maxShift milliseconds
    static void sleepBeforePoll(unsigned int polls, unsigned int maxShift = 4);
    // every node tells of a wait for what that takes seconds, then twice as long each time, as Debug::ERROR so that every node prints it
    class WaitWarning {
    public:
        explicit WaitWarning(size_t seconds = 600);
        void check(const std::string &what);
        void reset();

    private:
        time_t start;
        const size_t firstSeconds;
        size_t nextSeconds;
    };
    static void publishFile(const std::string &file, const std::string &content);
    // reads a file that an earlier step wrote
    static std::string readFile(const std::string &file);
    // publishFile and waiting for a file of this run, so a file that a killed run left is not read
    static void publishRunFile(const std::string &file, const std::string &content);
    static std::string waitForRunFile(const std::string &file);
    // waits for prefix + node of every node and returns their content in node order; the files are removed
    static std::string gatherRunFiles(const std::string &prefix);
    // a log that one node appends to while another reads it: its synced byte count and a key are published as <log>.progress, read from a new open
    static void publishLog(FILE *file, const std::string &log, size_t key, size_t bytes);
    static bool readLogProgress(const std::string &log, size_t &key, size_t &bytes);
    static void waitForLog(const std::string &log, size_t &key, size_t &bytes);
    static void readLog(const std::string &log, size_t from, size_t to, char *data);
    // closes file, a flush of bytes that only this node writes, then appends line with its last word set to this run's magic to indexFile
    static void publishExchangeFlush(FILE *file, const std::string &fileName, size_t bytes, const std::string &indexFile, const std::vector<uint64_t> &line);
    // reads line number flush, of line.size() words, of indexFile from a new open; false until this run's line is there
    static bool readExchangeFlush(const std::string &indexFile, size_t flush, std::vector<uint64_t> &line);
    static const size_t EXCHANGE_SLICE_HEADER_BYTES = 2 * sizeof(uint64_t);
    static void writeExchangeSliceHeader(FILE *file, const std::string &fileName, uint64_t generation);
    static int tryOpenExchangeSlice(const std::string &file, size_t expectedBytes, uint64_t expectedGeneration, time_t &lastDirectoryRefresh,
                                    size_t &observedBytes, uint64_t &observedGeneration, int &lastError);
    static bool tryReadExchangeSlice(int fd, char *data, size_t bytes, size_t offset, int &readError);
    static void truncateExchangeSlice(const std::string &file);
    // every node publishes content as file + "." + node; returns all contents in node order, gathered by the master into file
    static std::string barrier(const std::string &file, const std::string &content = "");
    // this node has read file; the master removes it once every node has, the others do not wait
    static void removeOnceRead(const std::string &file);
    // phase values that a node appends to the content of a barrier as lines name<TAB>node<TAB>value, for the master to print
    static void addPhaseValue(std::string &content, const std::string &name, const std::string &value);
    // the seconds since timer started, which then starts again
    static void addPhaseTime(std::string &content, const std::string &phase, Timer &timer);
    static void addPhaseSeconds(std::string &content, const std::string &phase, double seconds);
    // the master prints every name of the gathered phase values with its largest value and node, and its smallest value
    static void printPhaseValues(const std::string &values);
    // the items of a work queue that this node takes: its own items rank, rank + count, ... from the front, then those of the others from the back, round-robin from rank + 1
    class WorkQueue {
    public:
        WorkQueue(const std::string &name, size_t itemCount);
        // the next item that this node has taken by creating <name>.<item> exclusively, itemCount once none is left
        size_t claimNext();
        // removes <name>.<item> of the own items of this node, while no node takes items of the queue
        void removeClaims();

    private:
        bool claim(size_t item);

        const std::string name;
        const size_t itemCount;
        size_t nextItem;
        // the items of every other node not tried yet from the back, and the node to try next
        std::vector<size_t> itemsLeft;
        size_t nextNode;
    };
    // a barrier that returns the smallest memoryLimit of all nodes, so they all cut the same splits; with threads all nodes must run with as many
    static size_t getSmallestMemoryLimit(const std::string &file, size_t memoryLimit, int threads = 0);

    // an exchange file <prefix>.<node> holds flushes sorted by receiving node, then P + 1 columns of one offset per flush: column c is where the part of node c starts, the last column where the flush ends; a block is a part
    struct ExchangeBlock {
        FILE *file;
        size_t pos;
        size_t end;
        std::vector<char> buffer;
        size_t bufferPos;
        size_t bufferEnd;
        // end of the records that the buffer holds in full and start of the last of them
        size_t recordEnd;
        size_t lastRecord;
        // the records that this node sent to itself, read from memory instead of file and freed once read
        char *ownRecords;
    };
    // appends the offsets of a flush to indexData; nodeOffsets: where the bytes of every node start in the flush, then the flush size
    static void addExchangeFlush(std::string &indexData, size_t &writtenBytes, const std::vector<size_t> &nodeOffsets);
    // openExchange reads records, from malloc, as the records that this node sends to itself through exchange in its last flush
    static void keepOwnExchangeRecords(const std::string &exchange, char *records, size_t bytes);
    // appends the offsets of indexData to the file of this node, closes it and returns a line of every node, its record bytes and flush count, once every node has closed its file
    static std::string closeExchange(FILE *file, const std::string &fileName, const std::string &indexFile, const std::string &indexData);
    static void openExchange(const std::string &index, const std::string &prefix, std::vector<ExchangeBlock> &blocks, std::vector<FILE *> &files);
    // the smallest buffer of a block in readExchangeRange
    static const size_t EXCHANGE_BUFFER_SIZE = 1024 * 1024;
    // moves the buffered records up to last of every block to records, recordOffsets their starts then the end; returns their count, 0 once none is left
    static size_t readExchangeRange(std::vector<ExchangeBlock> &blocks, size_t bufferMemory, size_t (*recordSize)(const char *, size_t),
                                    bool (*compare)(const ClusterPair &, const ClusterPair &), const ClusterPair &last,
                                    std::vector<char> &records, std::vector<size_t> *recordOffsets, const std::string &prefix);
    // copies each segment { fileType, offset, size } then bytes of fileType that starts before windowEnd to window + offset - windowBegin and adds { offset, size } to segments; returns the offset of the next one, SIZE_MAX if none is left
    static size_t readExchangeSegments(std::vector<ExchangeBlock> &blocks, size_t bufferMemory, size_t maxSegmentSize, uint64_t fileType, size_t windowBegin, size_t windowEnd,
                                       char *window, std::vector<std::pair<size_t, size_t> > &segments, const std::string &prefix);
    static void closeExchangeFiles(std::vector<FILE *> &files, const std::string &prefix);
    class ClusterMemberLookup {
    public:
        struct Cursor {
            size_t block;
            size_t offset;
            DBKeyType key;
            size_t bufferBegin;
            size_t bufferEnd;
            size_t nextReadBytes;
            std::vector<ClusterPair> buffer;
        };

        ClusterMemberLookup(const std::string &index, const std::string &prefix, size_t maxBytes);
        ~ClusterMemberLookup();
        void find(DBKeyType key, Cursor &cursor) const;
        bool next(Cursor &cursor, DBKeyType &memberKey) const;
        size_t memoryBytes() const;

    private:
        struct Range {
            DBKeyType firstKey;
            DBKeyType lastKey;
            size_t firstAnchor;
            size_t lastAnchor;
        };
        struct Anchor {
            DBKeyType key;
            size_t offset;
        };
        ClusterPair readPair(size_t block, size_t offset) const;
        ClusterMemberLookup(const ClusterMemberLookup &);
        ClusterMemberLookup &operator=(const ClusterMemberLookup &);

        const std::string prefix;
        std::vector<ExchangeBlock> blocks;
        std::vector<FILE *> files;
        std::vector<Range> ranges;
        std::vector<Anchor> anchors;
    };

    // the pairs that this node sends through the exchange file <exchange>.<node>, sorted and written once pairCapacity are buffered; nodeKeyOffsets: keys of every node
    struct ClusterPairWriter {
        ClusterPairWriter(const std::string &exchange, size_t pairCapacity, bool (*compare)(const ClusterPair &, const ClusterPair &), const std::vector<size_t> &nodeKeyOffsets);
        ~ClusterPairWriter();
        void add(DBKeyType repKey, DBKeyType memberKey) {
            if (pairCount == pairCapacity) {
                flush();
            }
            pairs[pairCount].repKey = repKey;
            pairs[pairCount].memberKey = memberKey;
            pairCount++;
        }
        // makes room for count more pairs, at most pairCapacity, and returns where they go
        ClusterPair *reservePairs(size_t count) {
            if (pairCount + count > pairCapacity) {
                flush();
            }
            ClusterPair *reserved = pairs + pairCount;
            pairCount += count;
            return reserved;
        }
        void flush();
        // writes the buffered pairs but keeps those for this node in memory, and returns the line of every node from closeExchange
        std::string close();

        const std::string exchange;
        const std::string fileName;
        FILE *file;
        ClusterPair *pairs;
        const size_t pairCapacity;
        size_t pairCount;
        bool (*compare)(const ClusterPair &, const ClusterPair &);
        const std::vector<size_t> &nodeKeyOffsets;
        std::string indexData;
        size_t writtenBytes;
    };
    // the representative of entry p becomes the one received for p in the exchange receivedIndex, receivedPrefix if given
    // representativeBits: the keys [firstKey, firstKey + keyCount) of clu, which holds no entries for clusters of one key: a received member without an entry
    // goes on to its representative, and the bits left, those of the keys that stay alone or represent others, are published as <singletons>.bits.<node>
    static std::string sendClusterPairs(const std::string &clu, bool withRepresentative, bool (*compare)(const ClusterPair &, const ClusterPair &), const std::vector<size_t> &nodeKeyOffsets,
                                        const std::string &exchange, const std::string &receivedIndex = "", const std::string &receivedPrefix = "",
                                        uint64_t *representativeBits = NULL, size_t firstKey = 0, size_t keyCount = 0, const std::string &singletons = "");
    // --filter-cludb-file: a request { key, requesting node } of the exchange requestsIndex, requests gets the other members of key in <clu>.<node> as { key, member } through <members>.<node>
    static std::string sendClusterMembers(const std::string &clu, const std::string &requestsIndex, const std::string &requests, const std::string &members);
    // once every node has read lastExchange, a node removes its file of it; the master removes the indexes of exchanges and prints phaseValues
    static void removeExchanges(const std::string &prefix, const std::string &lastExchange, const std::vector<std::string> &exchanges, std::string phaseValues = "");
    // ORs the bits of keys [from, to) of a file, bit k at fileOffset = key fileFirstKey + k, into bit key - firstKey of bits, a word longer than its keys
    static void readKeyBits(int fd, const std::string &fileName, size_t fileOffset, size_t fileFirstKey, size_t from, size_t to, size_t firstKey, uint64_t *bits);
    // the representative bits of keys [firstKey, firstKey + keyCount) from the files <db>.bits.<node>, each {firstKey, keyCount} then its bits
    static uint64_t *readRepresentativeBits(const std::string &db, const std::vector<size_t> &fileKeyOffsets, size_t firstKey, size_t keyCount);
    // writes the representative bits of this node as <db>.bits.<node> and its ranges with their representative counts as <db>.index.<node>, gathered by the master
    static void writeRepresentativeBits(const std::string &db, DBReader<DBKeyType> &seqDbr, const uint64_t *bits, size_t firstKey, size_t keyCount);
    // a barrier that returns where the keys of every <clu>.<node> start, then keyCount, from their first entries, for parts in key order; only keys that are entries of the clustering are routed
    static std::vector<size_t> getClusterKeyOffsets(const std::string &file, const std::string &clu, size_t keyCount, const std::string &firstKeyFile = "");
    // the entry keys in [keyBegin, keyEnd) of the parts <clu>.<node>, whose keys start at cluKeyOffsets, in key order
    static void readClusterKeys(const std::string &clu, const std::vector<size_t> &cluKeyOffsets, size_t keyBegin, size_t keyEnd, std::vector<DBKeyType> &keys);
    // --createdb-mode 3 cluster databases: <clu>.index.<node> holds uint64 (key, offset) per entry in key order, then (DB_KEY_INVALID, data size)
    // appends the data and (key, offset in the data of the thread) entries of every thread to dataFile and indexFile, which hold dataBytes and indexBytes, on all threads
    static void writeClusterThreadParts(FILE *dataFile, const std::string &dataFileName, FILE *indexFile, const std::string &indexFileName,
                                        std::vector<std::string> &threadData, std::vector<std::vector<uint64_t> > &threadEntries, size_t &dataBytes, size_t &indexBytes);
    // writes the end entry (DB_KEY_INVALID, dataBytes) at indexBytes and closes dataFile and indexFile
    static void closeClusterFiles(FILE *dataFile, const std::string &dataFileName, FILE *indexFile, const std::string &indexFileName, size_t dataBytes, size_t indexBytes);
    // writes the pairs of an exchange in rep order as the cluster database <clu>.<node>; singletonBits: bit k makes key firstKey + k a singleton
    static void writeClusterPairs(const std::string &index, const std::string &prefix, const std::string &clu, const uint64_t *singletonBits, size_t firstKey, size_t keyCount);
};

#endif
