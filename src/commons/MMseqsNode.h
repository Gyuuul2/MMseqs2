#ifndef MMSEQS_NODE_H
#define MMSEQS_NODE_H

#include "IndexTypes.h"

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <string>
#include <vector>

class Timer;
class Parameters;
template <typename T> class DBReader;

struct ClusterPair {
    DBKeyType repKey;
    DBKeyType memberKey;

    static bool compareByMember(const ClusterPair &first, const ClusterPair &second) {
        return first.memberKey < second.memberKey;
    }

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

    static void sleepBeforePoll(unsigned int polls, unsigned int maxShift = 4);

    class WaitWarning {
    public:
        WaitWarning();
        void check(const std::string &what);
        void reset();

    private:
        time_t start;
        size_t nextSeconds;
    };

    static void publishFile(const std::string &file, const std::string &content);
    static std::string readFile(const std::string &file);
    static void publishRunFile(const std::string &file, const std::string &content);
    static std::string waitForRunFile(const std::string &file);
    static void waitForRunFilesChange(const std::string &prefix, std::string &content);
    static std::string gatherRunFiles(const std::string &prefix);

    static void publishLog(FILE *file, const std::string &log, size_t key, size_t size, size_t takenItemCount = 0);
    static bool readLogProgress(const std::string &log, size_t &key, size_t &size, size_t *takenItemCount = NULL);
    static void waitForLog(const std::string &log, size_t &key, size_t &size);
    static bool readLog(const std::string &log, size_t from, size_t to, char *data, bool wait = true);
    static void removeLog(const std::string &log);

    static void publishExchangeFile(FILE *file, const std::string &fileName, size_t size, const std::string &indexFile, const std::vector<uint64_t> &line);
    static size_t readExchangeIndexLines(const std::string &indexFile, size_t firstLine, size_t lineWords, std::vector<uint64_t> &lines);
    static size_t waitForExchangeIndexLines(const std::string &indexFile, size_t &nextLine, size_t lineWords, std::vector<uint64_t> &lines, bool wait);
    static const size_t EXCHANGE_FILE_HEADER_BYTES = 2 * sizeof(uint64_t);
    static FILE *createExchangeFile(const std::string &fileName, uint64_t exchangeFileNumber, const char *mode = "w");
    static void writeExchangeFile(const std::string &fileName, uint64_t exchangeFileNumber, const char *data, size_t size, const std::string &indexFile,
                                  const std::vector<uint64_t> &line, int threads);
    static int tryOpenExchangeFile(const std::string &file, size_t expectedSize, uint64_t expectedNumber,
                                   size_t &observedSize, uint64_t &observedNumber, int &lastError, int openFlags = 0);
    static int waitForExchangeFile(const std::string &file, size_t size, uint64_t exchangeFileNumber);
    static bool tryPreadAll(int fd, char *data, size_t size, size_t offset, int &readError);

    static std::string barrier(const std::string &file, const std::string &content = "");
    static void removeOnceRead(const std::string &file);

    static void addPhaseValue(std::string &content, const std::string &name, const std::string &value);
    static void addPhaseTime(std::string &content, const std::string &phase, Timer &timer);
    static void addPhaseSeconds(std::string &content, const std::string &phase, double seconds);
    static void printPhaseValues(const std::string &values);

    class WorkQueue {
    public:
        WorkQueue(const std::string &name, size_t itemCount);
        size_t takeNextItem();
        size_t takeNextItemInOrder();
        size_t getTakenItemCount() const;
        unsigned int getNode(size_t item) const;
        void removeTakenItemFiles();

    private:
        bool takeItem(size_t item);

        const std::string name;
        const size_t itemCount;
        size_t nextItem;
        size_t nextOrderedItem;
        std::vector<size_t> itemsLeft;
        size_t nextNode;
    };

    static size_t getSmallestMemoryLimit(const std::string &file, size_t memoryLimit, int threads = 0, const std::string &localDirectory = "", size_t *localNodes = NULL);
    // Reserve tmpfs sequence copies only in linfinity paths; localCopyBytes is for a module-owned copy.
    static size_t getMemoryLimit(const Parameters &par, size_t localCopyBytes = 0);

    struct ExchangeReader {
        FILE *file;
        size_t pos;
        size_t end;
        std::vector<char> buffer;
        size_t bufferPos;
        size_t bufferEnd;
        size_t recordEnd;
        size_t lastRecord;
        char *ownRecords;
        size_t punchEnd;
    };

    static size_t writeExchangeTable(int fd, const std::string &fileName, const std::vector<size_t> &flushOffsets, size_t columns, const std::vector<size_t> &tail, size_t tableOffset);
    static void readExchangeTableColumns(int fd, const std::string &fileName, size_t tableOffset, size_t flushCount, size_t column, std::vector<size_t> &offsets);
    static size_t getExchangeFileOffset(size_t offset, size_t ownBegin, size_t ownEnd) {
        return offset - (std::min(offset, ownEnd) - std::min(offset, ownBegin));
    }
    static void addExchangeFlush(std::vector<size_t> &indexData, size_t &writtenSize, const std::vector<size_t> &nodeOffsets, size_t ownBegin, size_t ownEnd, size_t lineBegin, size_t lineEnd);
    static void keepOwnExchangeRecords(const std::string &exchange, char *records, size_t size);
    static void addSentExchangeBytes(size_t size, size_t otherNodeSize);
    static std::string closeExchange(FILE *file, const std::string &fileName, const std::string &indexFile, const std::vector<size_t> &indexData);
    static void openExchange(const std::string &index, const std::string &prefix, std::vector<ExchangeReader> &readers, std::vector<FILE *> &files);
    static size_t readExchangeRecordsUpTo(std::vector<ExchangeReader> &readers, size_t bufferMemory, size_t (*recordSize)(const char *, size_t),
                                          bool (*compare)(const ClusterPair &, const ClusterPair &), const ClusterPair &last,
                                          std::vector<char> &records, std::vector<size_t> *recordOffsets, const std::string &prefix);
    static size_t readExchangeCompressionUnits(std::vector<ExchangeReader> &readers, const std::vector<std::vector<std::pair<size_t, size_t> > > &readerLengthOffsets, size_t bufferMemory, size_t maxCompressionUnitSize,
                                               size_t bufferBegin, size_t bufferEnd, char *buffer, std::vector<std::pair<size_t, size_t> > *compressionUnits, const std::string &prefix);
    static void closeExchangeFiles(std::vector<FILE *> &files, const std::string &prefix);

    struct ClusterPairWriter {
        ClusterPairWriter(const std::string &exchange, size_t pairCapacity, bool (*compare)(const ClusterPair &, const ClusterPair &), const std::vector<size_t> &nodeKeyOffsets,
                          std::vector<size_t> *balancedKeyOffsets = NULL);
        ~ClusterPairWriter();
        void add(DBKeyType repKey, DBKeyType memberKey) {
            if (pairCount == pairCapacity) {
                flush();
            }
            pairs[pairCount].repKey = repKey;
            pairs[pairCount].memberKey = memberKey;
            pairCount++;
        }

        ClusterPair *reservePairs(size_t count) {
            if (pairCount + count > pairCapacity) {
                flush();
            }
            ClusterPair *reserved = pairs + pairCount;
            pairCount += count;
            return reserved;
        }

        void flush();
        std::string close(const uint64_t *singletonBits = NULL, size_t firstKey = 0, size_t keyCount = 0);

        const std::string exchange;
        const std::string fileName;
        FILE *file;
        ClusterPair *pairs;
        const size_t pairCapacity;
        size_t pairCount;
        bool (*compare)(const ClusterPair &, const ClusterPair &);
        const std::vector<size_t> &nodeKeyOffsets;
        std::vector<size_t> *balancedKeyOffsets;
        std::vector<size_t> binOffsets;
        std::vector<size_t> indexData;
        size_t writtenSize;
        size_t keptPairCount;
    };

    static std::string sendClusterPairs(const std::string &clu, bool withRepresentative, bool (*compare)(const ClusterPair &, const ClusterPair &), const std::vector<size_t> &nodeKeyOffsets,
                                        const std::string &exchange, const std::string &receivedIndex = "", const std::string &receivedPrefix = "",
                                        uint64_t *representativeBits = NULL, const std::vector<size_t> &cluKeyOffsets = std::vector<size_t>(), size_t firstKey = 0, size_t keyCount = 0,
                                        std::vector<size_t> *balancedKeyOffsets = NULL);
    static void removeExchanges(const std::string &prefix, const std::string &lastExchange, const std::vector<std::string> &exchanges, std::string phaseValues = "");

    static void readKeyBits(int fd, const std::string &fileName, size_t fileOffset, size_t fileFirstKey, size_t from, size_t to, size_t firstKey, uint64_t *bits);
    static uint64_t *readRepresentativeBits(const std::string &db, const std::vector<size_t> &fileKeyOffsets, size_t firstKey, size_t keyCount);
    static void writeRepresentativeBits(const std::string &db, DBReader<DBKeyType> &seqDbr, const uint64_t *bits, size_t firstKey, size_t keyCount);

    static std::vector<size_t> getClusterKeyOffsets(const std::string &barrierFile, const std::string &clu, size_t keyCount, const std::string &firstKeySource = "");
    static std::vector<size_t> getBalancedClusterKeyOffsets(const std::string &file, const std::string &clu, size_t keyCount);
    static void readClusterKeys(const std::string &clu, const std::vector<size_t> &cluKeyOffsets, size_t keyBegin, size_t keyEnd, std::vector<DBKeyType> &keys);
    static void writeClusterThreadEntries(FILE *dataFile, const std::string &dataFileName, FILE *indexFile, const std::string &indexFileName,
                                          std::vector<std::string> &threadData, std::vector<std::vector<uint64_t> > &threadEntries, size_t &dataSize, size_t &indexSize);
    static void closeClusterFiles(FILE *dataFile, const std::string &dataFileName, FILE *indexFile, const std::string &indexFileName, size_t dataSize, size_t indexSize);
    static void writeClusterPairs(const std::string &index, const std::string &prefix, const std::string &clu, const uint64_t *singletonBits, size_t firstKey, size_t keyCount);
};

#endif
