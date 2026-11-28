#include "kmermatcher.h"
#include "Debug.h"
#include "FileUtil.h"
#include "FastSort.h"
#include "MMseqsNode.h"
#include "BatchReader.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <deque>
#include <functional>
#include <algorithm>
#include <map>

#ifdef OPENMP
#include <omp.h>
#endif

static const size_t KMER_FILE_BUFFER_SIZE = 16 * 1024 * 1024;
static const size_t TASKS_PER_THREAD = 2;
static const size_t RANGES_PER_NODE = 8;
// passes overlap reading, answering and writing
static const size_t MIN_ANSWER_PASSES = 4;
static const size_t MAX_TARGET_SPLITS_PER_NODE = 8;
static const size_t SEND_KMER_SIZE = 12;

enum { FINGERPRINT_FILES, MARK_FILES, CANDIDATE_FILES, REP_FILES, PIECE_COUNT_FILES, PREF_FILE_KINDS };

// hash sub range start in units of 2^shift, rounded up
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

// best effort, a background removal must not fail the run
static void removeExchangeFile(const std::string &file) {
    if (unlink(file.c_str()) != 0 && errno != ENOENT) {
        Debug(Debug::WARNING) << "Cannot delete " << file << ": " << strerror(errno) << "\n";
    }
}

static void addFinishedItems(const Parameters &par, std::string &progress, size_t kind, size_t group, size_t items) {
    progress += SSTR(kind) + "\t" + SSTR(group) + "\t" + SSTR(items) + "\n";
    MMseqsNode::publishRunFile(par.db2 + ".progress." + SSTR(MMseqsNode::rank), progress);
}

// kind k: groupItems[k][g] items finish group g of prefixes[k]
static void releaseFileGroups(const Parameters &par, std::vector<std::string> prefixes, std::vector<std::vector<size_t> > groupItems) {
    std::vector<std::vector<char> > isReleased(prefixes.size());
    size_t groupsLeft = 0;
    for (size_t kind = 0; kind < prefixes.size(); kind++) {
        isReleased[kind].assign(groupItems[kind].size(), false);
        groupsLeft += groupItems[kind].size();
    }
    std::string progress;
    while (true) {
        std::vector<std::vector<size_t> > finishedItems(prefixes.size());
        for (size_t kind = 0; kind < prefixes.size(); kind++) {
            finishedItems[kind].assign(groupItems[kind].size(), 0);
        }
        for (char *line = (char *) progress.c_str(); *line != '\0'; line = Util::skipLine(line)) {
            const char *columns[3];
            Util::getWordsOfLine(line, columns, 3);
            finishedItems[Util::fast_atoi<size_t>(columns[0])][Util::fast_atoi<size_t>(columns[1])] += Util::fast_atoi<size_t>(columns[2]);
        }
        for (size_t kind = 0; kind < prefixes.size(); kind++) {
            for (size_t i = 0; i < groupItems[kind].size(); i++) {
                if (isReleased[kind][i] == false && finishedItems[kind][i] == groupItems[kind][i]) {
                    MMseqsNode::publishRunFile(prefixes[kind] + ".released." + SSTR(i), "");
                    isReleased[kind][i] = true;
                    groupsLeft--;
                }
            }
        }
        if (groupsLeft == 0) {
            break;
        }
        MMseqsNode::waitForRunFilesChange(par.db2 + ".progress.", progress);
    }
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

// flushes: sequences in rep order, (rep, offset) pairs, end pair
struct SeedMemberFile {
    int fd;
    std::string name;
    size_t tableOffset;
    size_t flushCount;
};

// seed members of a send split in one flush
struct SeedMemberFlush {
    size_t file;
    std::vector<DBKeyType> repKeys;
    std::vector<size_t> offsets;
    // k-mer array slots before each member (k-mers + sequence)
    std::vector<size_t> slotOffsets;
};

static std::vector<SeedMemberFile> seedMemberFiles;
static std::vector<SeedMemberFlush> seedMemberFlushes;
static size_t seedMemberSendSplit = SIZE_MAX;

struct KmerExchange {
    std::vector<size_t> sendSplitKeyOffsets;
    size_t sendSplitCount;
    std::vector<size_t> fileKeyOffsets;
    bool hasRepresentativeBits;
    bool hasSeedMembers;
    size_t writeBehindSize;
    size_t keySplitMemoryLimit;
    size_t readChunkLimit;
    size_t readBufferSize;
    unsigned int adjKeyBits;
    unsigned int adjKeyBase;
    unsigned int adjKeyCodeBits;
    unsigned int kmerHashShift;
    unsigned int globalIdBits;
    size_t maxRepSeqCount;
    size_t partitionsPerRange;
};

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

static void readSeedMemberFlushes(const Parameters &par, size_t sendSplit) {
    if (seedMemberFiles.empty() || sendSplit == seedMemberSendSplit) {
        return;
    }
    std::vector<std::vector<size_t> > columns(seedMemberFiles.size());
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
    for (size_t file = 0; file < seedMemberFiles.size(); file++) {
        const SeedMemberFile &memberFile = seedMemberFiles[file];
        MMseqsNode::readExchangeTableColumns(memberFile.fd, memberFile.name, memberFile.tableOffset, memberFile.flushCount, sendSplit, columns[file]);
    }
    seedMemberFlushes.clear();
    std::vector<size_t> pairOffsets;
    std::vector<size_t> pairEnds;
    for (size_t file = 0; file < seedMemberFiles.size(); file++) {
        size_t flushCount = seedMemberFiles[file].flushCount;
        for (size_t flush = 0; flush < flushCount; flush++) {
            if (columns[file][flushCount + flush] > columns[file][flush]) {
                seedMemberFlushes.push_back(SeedMemberFlush());
                seedMemberFlushes.back().file = file;
                pairOffsets.push_back(columns[file][flush]);
                pairEnds.push_back(columns[file][flushCount + flush]);
            }
        }
    }
    float kmersPerSequenceScale = par.kmersPerSequenceScale.values.aminoacid();
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
    for (size_t flush = 0; flush < seedMemberFlushes.size(); flush++) {
        SeedMemberFlush &memberFlush = seedMemberFlushes[flush];
        const SeedMemberFile &memberFile = seedMemberFiles[memberFlush.file];
        size_t memberCount = (pairEnds[flush] - pairOffsets[flush]) / (2 * sizeof(uint64_t));
        std::vector<uint64_t> pairs(2 * (memberCount + 1));
        FileUtil::preadAll(memberFile.fd, memberFile.name.c_str(), (char *) pairs.data(), pairs.size() * sizeof(uint64_t), pairOffsets[flush]);
        memberFlush.repKeys.resize(memberCount);
        memberFlush.offsets.resize(memberCount + 1);
        memberFlush.slotOffsets.assign(memberCount + 1, 0);
        memberFlush.offsets[memberCount] = pairs[2 * memberCount + 1];
        for (size_t i = 0; i < memberCount; i++) {
            size_t length = pairs[2 * i + 3] - pairs[2 * i + 1];
            memberFlush.repKeys[i] = pairs[2 * i];
            memberFlush.offsets[i] = pairs[2 * i + 1];
            memberFlush.slotOffsets[i + 1] = memberFlush.slotOffsets[i] + computeSeqKmerCount(static_cast<int>(length), par.kmerSize, par.kmersPerSequence, kmersPerSequenceScale)
                                             + (length + sizeof(PackedKmerPosition) - 1) / sizeof(PackedKmerPosition);
        }
    }
    seedMemberSendSplit = sendSplit;
}

static void setSendSplit(const Parameters &par, const KmerExchange &exchange, size_t sendSplit) {
    sendSplitFirstKey = exchange.sendSplitKeyOffsets[sendSplit];
    delete[] representativeBits;
    representativeBits = NULL;
    if (exchange.hasRepresentativeBits) {
        representativeBits = MMseqsNode::readRepresentativeBits(par.db1, exchange.fileKeyOffsets, sendSplitFirstKey, exchange.sendSplitKeyOffsets[sendSplit + 1] - sendSplitFirstKey);
    }
}

static size_t countRepresentatives(size_t dbFrom, size_t from, size_t to) {
    return (representativeBits == NULL) ? to - from : Util::countSetBits(representativeBits, from - dbFrom, to - dbFrom);
}

static size_t getSeedMemberSlots(size_t from, size_t to) {
    size_t slots = 0;
    for (size_t flush = 0; flush < seedMemberFlushes.size(); flush++) {
        const std::vector<DBKeyType> &repKeys = seedMemberFlushes[flush].repKeys;
        size_t first = std::lower_bound(repKeys.begin(), repKeys.end(), static_cast<DBKeyType>(from)) - repKeys.begin();
        size_t last = std::lower_bound(repKeys.begin(), repKeys.end(), static_cast<DBKeyType>(to)) - repKeys.begin();
        slots += seedMemberFlushes[flush].slotOffsets[last] - seedMemberFlushes[flush].slotOffsets[first];
    }
    return slots;
}

static bool isKmerRecord(const PackedKmerPosition &kmer) {
    return kmer.isWholeSequence() == false;
}

static bool compareKmerHash(const PackedKmerPosition &kmer, uint64_t hash) {
    return static_cast<uint64_t>(kmer.getKmer()) < hash;
}

static size_t addKeySplits(const Parameters &par, DBReader<DBKeyType> &seqDbr, const KmerExchange &exchange, size_t dbFrom, size_t from, size_t to, size_t kmerCapacity,
                           std::vector<size_t> &keySplitOffsets) {
    // ids must fit next to the hash and the adjKey
    size_t keyLimit = (1ULL << (128 - 1 - 64 - exchange.adjKeyBits)) - 1;
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    float kmersPerSequenceScale = par.kmersPerSequenceScale.values.aminoacid();
    size_t firstRange = (to > from) ? seqDbr.findIdRange(from) : seqDbr.getIndexSize();
    size_t keySplitKmers = 0;
    size_t maxKmers = 0;
    for (size_t range = firstRange; range < seqDbr.getIndexSize() && index[range].id < to; range++) {
        size_t rangeFrom = std::max(static_cast<size_t>(index[range].id), from);
        size_t rangeTo = std::min(seqDbr.getRangeEndId(range), to);
        size_t seqKmers = computeSeqKmerCount(static_cast<int>(index[range].length - 2), par.kmerSize, par.kmersPerSequence, kmersPerSequenceScale);
        size_t rangeReps = countRepresentatives(dbFrom, rangeFrom, rangeTo);
        size_t cutKey = rangeFrom;
        while (keySplitKmers + rangeReps * seqKmers + getSeedMemberSlots(cutKey, rangeTo) > kmerCapacity || rangeTo - keySplitOffsets.back() > keyLimit) {
            size_t fitCount = std::min(rangeReps, (kmerCapacity - keySplitKmers) / seqKmers);
            size_t nextKey = rangeTo;
            if (fitCount < rangeReps && representativeBits == NULL) {
                nextKey = cutKey + fitCount;
            } else if (fitCount < rangeReps) {
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
                nextKey = dbFrom + bit / 64 * 64 + __builtin_ctzll(word);
            }
            if (nextKey - keySplitOffsets.back() > keyLimit) {
                nextKey = keySplitOffsets.back() + keyLimit;
            }
            // largest end whose reps and seed members fit
            size_t fitKey = cutKey;
            while (fitKey < nextKey) {
                size_t midKey = nextKey - (nextKey - fitKey) / 2;
                if (countRepresentatives(dbFrom, cutKey, midKey) * seqKmers + getSeedMemberSlots(cutKey, midKey) <= kmerCapacity - keySplitKmers) {
                    fitKey = midKey;
                } else {
                    nextKey = midKey - 1;
                }
            }
            if (nextKey == keySplitOffsets.back()) {
                Debug(Debug::ERROR) << "Not enough memory to extract the k-mers of a sequence of length " << index[range].length - 2
                                    << ". Raise --split-memory-limit.\n";
                EXIT(EXIT_FAILURE);
            }
            fitCount = countRepresentatives(dbFrom, cutKey, nextKey);
            maxKmers = std::max(maxKmers, keySplitKmers + fitCount * seqKmers + getSeedMemberSlots(cutKey, nextKey));
            rangeReps -= fitCount;
            cutKey = nextKey;
            keySplitOffsets.push_back(cutKey);
            keySplitKmers = 0;
        }
        keySplitKmers += rangeReps * seqKmers + getSeedMemberSlots(cutKey, rangeTo);
        maxKmers = std::max(maxKmers, keySplitKmers);
    }
    keySplitOffsets.push_back(to);
    return maxKmers;
}

enum { TASK_DONE, TASK_OUTPUT_FULL, TASK_GROUP_TOO_BIG };

struct CoverCheck {
    float covThr;
    int covMode;
    float repLength;
    bool isCovered;

    bool operator()(const DBReader<DBKeyType>::Index &range) const {
        return Util::canBeCovered(covThr, covMode, repLength, static_cast<float>(range.length - 2)) == isCovered;
    }
};

// ranges are in length descending order
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

// whole blocks
static std::vector<size_t> getTargetSplitBlockOffsets(DBReader<DBKeyType> &seqDbr) {
    size_t alignNodes = MMseqsNode::count - MMseqsNode::getFirstAlignNode();
    size_t blockCount = seqDbr.getBlockKeyOffsets().size() - 1;
    size_t targetSplitCount = std::min(blockCount, MAX_TARGET_SPLITS_PER_NODE * alignNodes);
    std::vector<size_t> offsets(targetSplitCount + 1);
    for (size_t split = 0; split <= targetSplitCount; split++) {
        offsets[split] = split * blockCount / targetSplitCount;
    }
    return offsets;
}

// every key split writes its own prefilter
enum { KMER_GROUPS, WHOLE_SEQUENCE_GROUPS, GROUP_KINDS };

static const size_t RECENT_PAIR_SLOTS = 1 << 16;
static const size_t PAIR_BLOCK_SIZE = 4096;

struct KeySplit {
    size_t firstKey;
    size_t endKey;
    unsigned int idBits;
};

// as published at the barrier, part p starts at partStarts[p]
struct PrefExchangeFile {
    size_t size;
    size_t tableOffset;
    size_t flushCount;
    std::vector<size_t> partStarts;
};

// below the 16 TiB file limit of ext4 and Lustre ldiskfs
static const size_t EXCHANGE_PART_SIZE = 1ULL << 40;

// own rows of an exchange file in --local-tmp, read by this process only
struct LocalExchangeFile {
    std::vector<int> fds;
    std::vector<std::string> names;
    // file offsets of kept flushes, in write order
    std::vector<std::pair<size_t, size_t> > ranges;

    bool holds(size_t offset) const {
        std::vector<std::pair<size_t, size_t> >::const_iterator range = std::upper_bound(ranges.begin(), ranges.end(), std::make_pair(offset, SIZE_MAX));
        return range != ranges.begin() && offset < (range - 1)->second;
    }

    // unlinked when opened, closing frees them
    void close() const {
        for (size_t part = 0; part < fds.size(); part++) {
            if (fds[part] >= 0 && ::close(fds[part]) != 0) {
                Debug(Debug::WARNING) << "Cannot close " << names[part] << ": " << strerror(errno) << "\n";
            }
        }
    }
};

// parts <name>.<part>, a flush stays in one part
struct PrefExchangeWriter {
    PrefExchangeWriter(const std::string &name) : name(name), file(NULL), size(MMseqsNode::EXCHANGE_FILE_HEADER_BYTES) {
        startFlush(0, NULL);
    }

    size_t getPartOffset(size_t offset) const {
        return offset - partStarts.back() + MMseqsNode::EXCHANGE_FILE_HEADER_BYTES;
    }

    // start the next part once the writes are done
    void startFlush(size_t flushSize, BatchWriter *batchWriter) {
        if (file != NULL && (size == partStarts.back() || getPartOffset(size) + flushSize <= EXCHANGE_PART_SIZE)) {
            return;
        }
        if (file != NULL) {
            if (batchWriter != NULL) {
                batchWriter->waitWrites();
            }
            // set the full size, kept own rows may leave a hole
            if (ftruncate(fileno(file), getPartOffset(size)) != 0) {
                Debug(Debug::ERROR) << "Cannot truncate " << partName << ": " << strerror(errno) << "\n";
                EXIT(EXIT_FAILURE);
            }
            MMseqsNode::publishExchangeFile(file, partName, getPartOffset(size), "", std::vector<uint64_t>(1, 0));
        }
        partStarts.push_back(size);
        partName = name + "." + SSTR(partStarts.size() - 1);
        file = MMseqsNode::createExchangeFile(partName, partStarts.size() - 1);
    }

    // flush table ends the last part
    std::string publish(size_t columns) {
        size_t flushCount = flushOffsets.size() / columns;
        startFlush(flushOffsets.size() * sizeof(size_t), NULL);
        size_t tableOffset = size;
        size_t partSize = MMseqsNode::writeExchangeTable(fileno(file), partName, flushOffsets, columns, std::vector<size_t>(), getPartOffset(tableOffset));
        size = tableOffset + flushOffsets.size() * sizeof(size_t);
        MMseqsNode::publishExchangeFile(file, partName, partSize, "", std::vector<uint64_t>(1, 0));
        file = NULL;
        std::string line = SSTR(size) + "\t" + SSTR(tableOffset) + "\t" + SSTR(flushCount) + "\t" + SSTR(partStarts.size());
        for (size_t part = 0; part < partStarts.size(); part++) {
            line += "\t" + SSTR(partStarts[part]);
        }
        return line + "\n";
    }

    std::string name;
    std::string partName;
    FILE *file;
    size_t size;
    std::vector<size_t> partStarts;
    std::vector<size_t> flushOffsets;
};

struct PrefExchangeParts {
    PrefExchangeParts() : local(NULL) {}

    std::vector<std::vector<int> > fds;
    std::vector<std::vector<std::string> > names;
    // own rows in --local-tmp, fds stay on the shared storage
    const LocalExchangeFile *local;

    // a read lies in one flush: kept rows from --local-tmp, tables from the part
    BatchReader::ReadRequest getRead(const PrefExchangeFile &file, size_t node, char *destination, size_t offset, size_t size) const {
        size_t part = std::upper_bound(file.partStarts.begin(), file.partStarts.end(), offset) - file.partStarts.begin() - 1;
        BatchReader::ReadRequest read = {fds[node][part], names[node][part].c_str(), destination, offset - file.partStarts[part] + MMseqsNode::EXCHANGE_FILE_HEADER_BYTES, size, size};
        if (node == MMseqsNode::rank && local != NULL && offset < file.tableOffset && local->holds(offset)) {
            read.fileToRead = local->fds[part];
            read.fileName = local->names[part].c_str();
        }
        return read;
    }
};

// marks or reps of the own key splits, opened and finished per group
struct AnswerGroupParts {
    AnswerGroupParts(const std::string &prefix, size_t kind, const std::vector<std::vector<PrefExchangeFile> > &groups)
        : prefix(prefix), kind(kind), groups(groups), parts(groups.size()), keySplitsLeft(groups.size(), 0) {}

    std::string prefix;
    size_t kind;
    const std::vector<std::vector<PrefExchangeFile> > &groups;
    std::vector<PrefExchangeParts> parts;
    std::vector<size_t> keySplitsLeft;
};

// a queued punch keeps its fds open, the running one counts too
static const size_t MAX_QUEUED_REMOVALS = 8;

// run removals in order on one thread, punched fds stay open until wait()
struct BackgroundRemoval {
    BackgroundRemoval() : isClosing(false) {}

    ~BackgroundRemoval() {
        if (worker.joinable()) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                isClosing = true;
            }
            condition.notify_all();
            worker.join();
        }
    }

    template <typename Removal>
    void run(Removal removal) {
        std::unique_lock<std::mutex> lock(mutex);
        while (removals.size() >= MAX_QUEUED_REMOVALS) {
            condition.wait(lock);
        }
        removals.emplace_back(removal);
        if (worker.joinable() == false) {
            worker = std::thread(&BackgroundRemoval::runRemovals, this);
        }
        condition.notify_all();
    }

    void removeFile(const std::string &file) {
        run([file]() { removeExchangeFile(file); });
    }

    void punch(const std::vector<BatchReader::ReadRequest> &readRequests) {
        if (readRequests.empty()) {
            return;
        }
        run([readRequests]() {
            for (size_t i = 0; i < readRequests.size(); i++) {
                FileUtil::punchWholeBlocks(readRequests[i].fileToRead, readRequests[i].fileOffset, readRequests[i].bytesToRead);
            }
        });
    }

    bool isStopping() {
        std::lock_guard<std::mutex> lock(mutex);
        return isClosing;
    }

    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        while (removals.empty() == false) {
            condition.wait(lock);
        }
    }

    void wait(const std::string &what) {
        MMseqsNode::WaitWarning warning;
        std::unique_lock<std::mutex> lock(mutex);
        while (removals.empty() == false) {
            condition.wait_for(lock, std::chrono::seconds(10));
            warning.check(what);
        }
    }

    // a removal leaves the queue once done
    void runRemovals() {
        std::unique_lock<std::mutex> lock(mutex);
        while (true) {
            while (removals.empty() && isClosing == false) {
                condition.wait(lock);
            }
            if (removals.empty()) {
                return;
            }
            std::function<void()> removal = removals.front();
            lock.unlock();
            removal();
            lock.lock();
            removals.pop_front();
            condition.notify_all();
        }
    }

    std::vector<std::pair<size_t, size_t> > takeFinished() {
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<std::pair<size_t, size_t> > taken;
        std::swap(taken, finished);
        return taken;
    }

    std::deque<std::function<void()> > removals;
    // (kind, group) closed by the worker, published by the main thread
    std::vector<std::pair<size_t, size_t> > finished;
    bool isClosing;
    std::mutex mutex;
    std::condition_variable condition;
    std::thread worker;
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

struct KeySplitKmers {
    KeySplitKmers() : kmers(NULL) {}
    PackedKmerPosition *kmers;
    std::vector<size_t> rangeOffsets[GROUP_KINDS];
};

enum { SENT_KMER_GROUPS, SENT_WHOLE_SEQUENCE_GROUPS, LOCAL_KMER_GROUPS, LOCAL_WHOLE_SEQUENCE_GROUPS, GROUP_SEGMENTS };

// needed groups in --local-tmp, nodes: ranks on its device (0 if unusable)
struct NeededSpill {
    std::string name;
    std::string directory;
    size_t nodes;
    FILE *file;
    size_t size;
};

// own fingerprint and rep rows in --local-tmp, nodes: ranks on its device (0 if unusable)
struct LocalExchange {
    std::string directory;
    size_t nodes;
    std::map<std::string, LocalExchangeFile> files;
    size_t keptSize;
    // kept in files not closed yet
    size_t heldSize;
};

enum { NOT_COPIED_YET, COPIED, NOT_COPIED };

// sequences to extract again, copied to --local-tmp at data file offsets
struct SequenceCopy {
    // false once a copy cannot be made
    bool isCopying;
    std::vector<std::string> names;
    // -1 until a key split is copied into the file
    std::vector<int> fds;
    // per own key split
    std::vector<unsigned char> keySplitStates;
    std::vector<std::vector<BatchReader::ReadRequest> > keySplitCopyReads;
    size_t keySplits;
    size_t size;
    size_t maxSize;
};

struct NeededGroups {
    NeededGroups() : data(NULL), dataSize(0), spillOffset(SIZE_MAX) {}
    char *data;
    size_t dataSize;
    size_t spillOffset;
    std::vector<size_t> rowOffsets;
};

// marked fingerprints by own hash range, then key split
struct MarkedFingerprints {
    // own hash ranges [0, keptHashRanges) are kept
    size_t keptHashRanges;
    bool isKeeping;
    std::vector<size_t> rowOffsets;
    // in memory: a malloc block per sub pass, data per own hash range
    std::vector<char *> blocks;
    std::vector<size_t> blockOffsets;
    std::vector<char *> hashRangeData;
    size_t memorySize;
    // in --local-tmp, unlinked
    int fd;
    std::string name;

    void freeBlocks() {
        for (size_t block = 0; block < blocks.size(); block++) {
            free(blocks[block]);
        }
        blocks.clear();
        blockOffsets.clear();
        hashRangeData.clear();
        memorySize = 0;
    }

    // read and punched only
    void close() {
        freeBlocks();
        if (fd >= 0 && ::close(fd) != 0) {
            Debug(Debug::WARNING) << "Cannot close " << name << ": " << strerror(errno) << "\n";
        }
        fd = -1;
        std::vector<size_t>().swap(rowOffsets);
    }
};

struct KeySplitPref {
    size_t hashRangeCount;
    std::vector<size_t> hashRangeNodeOffsets;
    unsigned int codeShifts[GROUP_KINDS];
    unsigned int fingerprintShifts[GROUP_KINDS];
    std::vector<size_t> targetSplitBlockOffsets;
    std::vector<size_t> targetSplitKeyOffsets;
    std::vector<size_t> partitionFirstKeys;
    std::vector<size_t> partitionNodeOffsets;
    std::vector<size_t> sendSplits;
    std::vector<size_t> sendSplitKeySplitOffsets;
    std::vector<KeySplit> keySplits;
    std::vector<size_t> nodeKeySplitOffsets;
    // numbered in key order
    std::vector<size_t> keySplitFiles;
    // own key split, hash range, kind
    std::vector<size_t> fingerprintCounts;
    // per own key split
    std::vector<size_t> maxKmerCounts;
    std::vector<std::pair<DBKeyType, DBKeyType> > rangeCoveredKeys;
    // reps that can cover the key split
    std::vector<std::pair<DBKeyType, DBKeyType> > keySplitRepKeys;
    size_t keySplitKmerCapacity;
    std::vector<KeySplitKmers> keptKeySplits;
    size_t keptSize;
    bool isKeeping;
    KeySplitKmers extractedKeySplit;
    std::vector<NeededGroups> neededGroups;
    size_t neededSize;
    size_t neededLimit;
    // keySplitMemoryLimit without the read buffers
    size_t keySplitMemory;
    NeededSpill spill;
    SequenceCopy sequenceCopy;
    LocalExchange localExchange;
    size_t passLimit;
    char *writeBuffers;
    // hash range groups, fixed before writing
    size_t hashRangeGroups;
    std::string progress;
    MarkedFingerprints marked;
    // per node: first hash ranges whose codes are sent below the marked fingerprint
    std::vector<size_t> keptMarkedHashRanges;
};

static size_t getKeySplitIndex(const KeySplitPref &pref, size_t keySplit) {
    return pref.nodeKeySplitOffsets[MMseqsNode::rank] + keySplit;
}

static size_t getHeldMemory(const KeySplitPref &pref) {
    return pref.keptSize * sizeof(PackedKmerPosition) + pref.neededSize + pref.fingerprintCounts.size() * sizeof(size_t)
           + ((pref.extractedKeySplit.kmers != NULL) ? pref.keySplitKmerCapacity * sizeof(PackedKmerPosition) : 0)
           + pref.marked.memorySize + pref.marked.rowOffsets.size() * sizeof(size_t);
}

// copy is not read again
static std::vector<BatchReader::ReadRequest> takeSequenceCopyReads(KeySplitPref &pref, size_t ownKeySplit) {
    std::vector<BatchReader::ReadRequest> copyReads;
    if (ownKeySplit < pref.sequenceCopy.keySplitCopyReads.size()) {
        copyReads.swap(pref.sequenceCopy.keySplitCopyReads[ownKeySplit]);
    }
    return copyReads;
}

// key splits not copied yet are read from the shared folder
static void stopSequenceCopy(const Parameters &par, SequenceCopy &copy, const std::string &problem) {
    MMseqsNode::reportUnusableLocalTmp("Cannot keep the sequences in --local-tmp " + par.localTmp + ": " + problem);
    copy.isCopying = false;
}

// copy files open now or needed by key splits not copied yet
static size_t countSequenceCopyFiles(const KeySplitPref &pref, DBReader<DBKeyType> &seqDbr) {
    const SequenceCopy &copy = pref.sequenceCopy;
    std::vector<bool> isCounted(copy.fds.size(), false);
    for (size_t file = 0; file < copy.fds.size(); file++) {
        isCounted[file] = copy.fds[file] >= 0;
    }
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    for (size_t ownKeySplit = 0; copy.isCopying && ownKeySplit < copy.keySplitStates.size(); ownKeySplit++) {
        if (copy.keySplitStates[ownKeySplit] != NOT_COPIED_YET) {
            continue;
        }
        const KeySplit &keySplit = pref.keySplits[getKeySplitIndex(pref, ownKeySplit)];
        for (size_t range = seqDbr.findIdRange(keySplit.firstKey); range < seqDbr.getIndexSize() && index[range].id < keySplit.endKey; range++) {
            isCounted[seqDbr.getRangeDataFile(range)] = true;
        }
    }
    return std::count(isCounted.begin(), isCounted.end(), true);
}

static void freeKeptKeySplits(KeySplitPref &pref) {
    for (size_t keySplit = 0; keySplit < pref.keptKeySplits.size(); keySplit++) {
        delete[] pref.keptKeySplits[keySplit].kmers;
        pref.keptKeySplits[keySplit].kmers = NULL;
    }
    pref.keptSize = 0;
}

static size_t getRiceSize(uint64_t value, unsigned int bits) {
    return (value >> bits) + 1 + bits;
}

static size_t getEliasGammaSize(size_t value) {
    return 2 * getBitWidth(value) - 1;
}

static unsigned int getRowLowBits(const KeySplitPref &pref, size_t hashRange, unsigned int shift, size_t groupCount) {
    return getHashDiffLowBits(hashRange, hashRange + 1, pref.hashRangeCount, shift, groupCount);
}

// rounded down for the fingerprint shift
static uint64_t getRowFirstValue(const KeySplitPref &pref, size_t hashRange, unsigned int shift) {
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

static PrefExchangeFile readPrefExchangeLine(const char *line) {
    const char *columns[4];
    Util::getWordsOfLine(line, columns, 4);
    PrefExchangeFile file = { Util::fast_atoi<size_t>(columns[0]), Util::fast_atoi<size_t>(columns[1]), Util::fast_atoi<size_t>(columns[2]), std::vector<size_t>() };
    const char *word = columns[3];
    for (size_t part = Util::fast_atoi<size_t>(columns[3]); part > 0; part--) {
        word += Util::skipNoneWhitespace(word);
        word += Util::skipWhitespace(word);
        file.partStarts.push_back(Util::fast_atoi<size_t>(word));
    }
    return file;
}

static std::vector<PrefExchangeFile> readPrefExchangeFiles(const std::string &barrierContent) {
    std::vector<PrefExchangeFile> files;
    for (char *line = (char *) barrierContent.c_str(); *line != '\0'; line = Util::skipLine(line)) {
        files.push_back(readPrefExchangeLine(line));
    }
    return files;
}

static std::string getPrefExchangeName(const std::string &prefix, size_t node, size_t group) {
    return prefix + "." + SSTR(node) + "." + SSTR(group);
}

// barrier lines come per node, then per group
static std::vector<std::vector<PrefExchangeFile> > getPrefExchangeGroups(const std::vector<PrefExchangeFile> &files, size_t groupCount) {
    std::vector<std::vector<PrefExchangeFile> > groups(groupCount);
    for (size_t i = 0; i < files.size(); i++) {
        groups[i % groupCount].push_back(files[i]);
    }
    return groups;
}

// every node on the --local-tmp device can reserve size next to reservedSize and the copy share
static bool hasLocalRoom(const KeySplitPref &pref, size_t size, size_t reservedSize) {
    const LocalExchange &local = pref.localExchange;
    const SequenceCopy &copy = pref.sequenceCopy;
    size_t freeSize = FileUtil::getFreeSpace(local.directory.c_str());
    // copy sizes its share as if no rows were kept
    size_t copyShare = 0;
    if (copy.isCopying) {
        copyShare = copy.fds.empty() ? freeSize / (2 * local.nodes) + local.heldSize / 2 : copy.maxSize - std::min(copy.maxSize, copy.size);
    }
    return freeSize != SIZE_MAX && (2 * size + reservedSize + copyShare) * local.nodes <= freeSize;
}

// own rows [offset, offset + size) go to --local-tmp if hasLocalRoom, else to the part
static int getOwnRowFd(KeySplitPref &pref, const PrefExchangeWriter &writer, size_t offset, size_t size, size_t reservedSize, const char *&fileName) {
    LocalExchange &local = pref.localExchange;
    fileName = writer.partName.c_str();
    if (local.nodes == 0 || size == 0 || hasLocalRoom(pref, size, reservedSize) == false) {
        return fileno(writer.file);
    }
    LocalExchangeFile &kept = local.files[writer.name];
    size_t part = writer.partStarts.size() - 1;
    if (kept.fds.size() <= part) {
        kept.fds.resize(part + 1, -1);
        kept.names.resize(part + 1);
    }
    if (kept.fds[part] < 0) {
        kept.names[part] = local.directory + "/" + FileUtil::baseName(writer.partName);
        kept.fds[part] = open(kept.names[part].c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if (kept.fds[part] < 0 || unlink(kept.names[part].c_str()) != 0) {
            int error = errno;
            if (kept.fds[part] >= 0) {
                close(kept.fds[part]);
                kept.fds[part] = -1;
                removeExchangeFile(kept.names[part]);
            }
            // the needed groups and the sequence copy may still use it
            Debug(Debug::WARNING) << "Cannot keep exchange rows in --local-tmp " << local.directory << ": cannot create " << kept.names[part] << ": " << strerror(error)
                                  << "; the rows a node sends to itself go to the shared folder\n";
            local.nodes = 0;
            return fileno(writer.file);
        }
    }
    // reserved first, a full device must not stop the writes
    int failed = -1;
#ifdef __linux__
    do {
        failed = fallocate(kept.fds[part], 0, writer.getPartOffset(offset), size);
    } while (failed != 0 && errno == EINTR);
#endif
    if (failed != 0) {
        return fileno(writer.file);
    }
    if (kept.ranges.empty() == false && kept.ranges.back().second == offset) {
        kept.ranges.back().second += size;
    } else {
        kept.ranges.emplace_back(offset, offset + size);
    }
    local.keptSize += size;
    local.heldSize += size;
    fileName = kept.names[part].c_str();
    return kept.fds[part];
}

// kept files, marked fingerprints and the next writer file
static size_t countLocalExchangeFiles(const KeySplitPref &pref) {
    size_t files = ((pref.localExchange.nodes > 0) ? 1 : 0) + ((pref.marked.fd >= 0) ? 1 : 0);
    for (std::map<std::string, LocalExchangeFile>::const_iterator kept = pref.localExchange.files.begin(); kept != pref.localExchange.files.end(); ++kept) {
        files += kept->second.fds.size();
    }
    return files;
}

// closed by the caller once its reads are punched
static LocalExchangeFile takeLocalExchangeFile(KeySplitPref &pref, const std::string &name) {
    LocalExchangeFile file;
    std::map<std::string, LocalExchangeFile>::iterator kept = pref.localExchange.files.find(name);
    if (kept != pref.localExchange.files.end()) {
        std::swap(file, kept->second);
        pref.localExchange.files.erase(kept);
    }
    for (size_t i = 0; i < file.ranges.size(); i++) {
        pref.localExchange.heldSize -= file.ranges[i].second - file.ranges[i].first;
    }
    return file;
}

static PrefExchangeParts openPrefExchangeFiles(const KeySplitPref &pref, const std::string &prefix, size_t group, const std::vector<PrefExchangeFile> &files) {
    PrefExchangeParts parts;
    std::map<std::string, LocalExchangeFile>::const_iterator kept = pref.localExchange.files.find(getPrefExchangeName(prefix, MMseqsNode::rank, group));
    if (kept != pref.localExchange.files.end()) {
        parts.local = &kept->second;
    }
    parts.fds.resize(files.size());
    parts.names.resize(files.size());
    for (size_t node = 0; node < files.size(); node++) {
        const std::vector<size_t> &partStarts = files[node].partStarts;
        for (size_t part = 0; part < partStarts.size(); part++) {
            size_t partEnd = (part + 1 < partStarts.size()) ? partStarts[part + 1] : files[node].size;
            parts.names[node].push_back(getPrefExchangeName(prefix, node, group) + "." + SSTR(part));
            parts.fds[node].push_back(MMseqsNode::waitForExchangeFile(parts.names[node].back(), partEnd - partStarts[part] + MMseqsNode::EXCHANGE_FILE_HEADER_BYTES, part));
        }
    }
    return parts;
}

// largest group, a reader holds one open (sendCandidates two at a group change)
static size_t getPartCount(const std::vector<std::vector<PrefExchangeFile> > &groups) {
    size_t maxParts = 0;
    for (size_t group = 0; group < groups.size(); group++) {
        size_t parts = 0;
        for (size_t node = 0; node < groups[group].size(); node++) {
            parts += groups[group][node].partStarts.size();
        }
        maxParts = std::max(maxParts, parts);
    }
    return maxParts;
}

// read and punched only, a failed close loses no data
static void closePrefExchangeFiles(PrefExchangeParts &parts) {
    for (size_t node = 0; node < parts.fds.size(); node++) {
        for (size_t part = 0; part < parts.fds[node].size(); part++) {
            if (close(parts.fds[node][part]) != 0) {
                Debug(Debug::WARNING) << "Cannot close " << parts.names[node][part] << ": " << strerror(errno) << "\n";
            }
        }
    }
}

// remove own parts once every reader closed them
// one removal, the queue bound must not hold back the releasing reads
static void removeReleasedPrefExchangeFiles(BackgroundRemoval &removal, const std::string &prefix, const std::vector<std::vector<PrefExchangeFile> > &groups) {
    std::vector<size_t> groupParts;
    for (size_t group = 0; group < groups.size(); group++) {
        groupParts.push_back(groups[group][MMseqsNode::rank].partStarts.size());
    }
    removal.run([&removal, prefix, groupParts]() {
        for (size_t group = 0; group < groupParts.size(); group++) {
            // open first, readRunFile must not exit the worker
            std::string released = prefix + ".released." + SSTR(group);
            std::string content;
            for (unsigned int polls = 0;; polls++) {
                int fd = FileUtil::openListedFile(released);
                if (fd >= 0) {
                    close(fd);
                    if (MMseqsNode::readRunFile(released, content)) {
                        break;
                    }
                }
                if (removal.isStopping()) {
                    return;
                }
                FileUtil::sleepBeforePoll(polls, 7);
            }
            std::string name = getPrefExchangeName(prefix, MMseqsNode::rank, group);
            for (size_t part = 0; part < groupParts[group]; part++) {
                removeExchangeFile(name + "." + SSTR(part));
            }
        }
    });
}

// close after its punches, publishFinishedGroups reports it
static void finishPrefExchangeGroup(BackgroundRemoval &punches, KeySplitPref &pref, const std::string &prefix, PrefExchangeParts &parts, size_t kind, size_t group) {
    PrefExchangeParts closed;
    std::swap(closed, parts);
    LocalExchangeFile local = takeLocalExchangeFile(pref, getPrefExchangeName(prefix, MMseqsNode::rank, group));
    punches.run([&punches, closed, local, kind, group]() mutable {
        closePrefExchangeFiles(closed);
        local.close();
        std::lock_guard<std::mutex> lock(punches.mutex);
        punches.finished.emplace_back(kind, group);
    });
}

static void publishFinishedGroups(const Parameters &par, BackgroundRemoval &punches, std::string &progress) {
    std::vector<std::pair<size_t, size_t> > finished = punches.takeFinished();
    for (size_t i = 0; i < finished.size(); i++) {
        addFinishedItems(par, progress, finished[i].first, finished[i].second, 1);
    }
}

static std::vector<PrefExchangeWriter> createPrefExchangeWriters(const std::string &prefix, size_t groupCount) {
    std::vector<PrefExchangeWriter> writers;
    writers.reserve(groupCount);
    for (size_t group = 0; group < groupCount; group++) {
        writers.emplace_back(getPrefExchangeName(prefix, MMseqsNode::rank, group));
    }
    return writers;
}

// every node splits its own hash ranges into the same groups
static size_t getGroupStart(size_t group, size_t groupCount, size_t ownHashRanges) {
    return (group * ownHashRanges + groupCount - 1) / groupCount;
}

// group file table: columns of its group only, node after node
static std::string publishPrefExchangeWriters(const KeySplitPref &pref, std::vector<PrefExchangeWriter> &writers, size_t &sentSize) {
    std::string lines;
    sentSize = 0;
    size_t ownHashRanges = pref.hashRangeCount / MMseqsNode::count;
    for (size_t group = 0; group < writers.size(); group++) {
        sentSize += writers[group].size;
        size_t groupHashRanges = getGroupStart(group + 1, writers.size(), ownHashRanges) - getGroupStart(group, writers.size(), ownHashRanges);
        lines += writers[group].publish(MMseqsNode::count * (groupHashRanges + 1));
    }
    return lines;
}

static size_t getHashRangeGroup(const KeySplitPref &pref, size_t hashRange) {
    size_t node = std::upper_bound(pref.hashRangeNodeOffsets.begin(), pref.hashRangeNodeOffsets.end(), hashRange) - pref.hashRangeNodeOffsets.begin() - 1;
    size_t first = pref.hashRangeNodeOffsets[node];
    return (hashRange - first) * pref.hashRangeGroups / (pref.hashRangeNodeOffsets[node + 1] - first);
}

// every node splits its own key splits into the groups of its hash ranges
static size_t getKeySplitGroup(const KeySplitPref &pref, size_t keySplit) {
    size_t node = std::upper_bound(pref.nodeKeySplitOffsets.begin(), pref.nodeKeySplitOffsets.end(), keySplit) - pref.nodeKeySplitOffsets.begin() - 1;
    size_t first = pref.nodeKeySplitOffsets[node];
    return (keySplit - first) * pref.hashRangeGroups / (pref.nodeKeySplitOffsets[node + 1] - first);
}

// key splits of the group before keySplit, a column each in answer tables
static size_t getKeySplitColumn(const KeySplitPref &pref, size_t group, size_t keySplit) {
    size_t column = 0;
    for (size_t node = 0; node + 1 < pref.nodeKeySplitOffsets.size(); node++) {
        size_t first = pref.nodeKeySplitOffsets[node];
        size_t groupFirst = first + getGroupStart(group, pref.hashRangeGroups, pref.nodeKeySplitOffsets[node + 1] - first);
        size_t groupEnd = first + getGroupStart(group + 1, pref.hashRangeGroups, pref.nodeKeySplitOffsets[node + 1] - first);
        column += std::min(groupEnd, std::max(groupFirst, keySplit)) - groupFirst;
    }
    return column;
}

static void readTables(const PrefExchangeParts &parts, const std::vector<PrefExchangeFile> &files, size_t firstColumn, size_t columnCount,
                       std::vector<std::vector<size_t> > &tables) {
    for (size_t node = 0; node < files.size(); node++) {
        tables[node].resize(columnCount * files[node].flushCount);
        BatchReader::ReadRequest read = parts.getRead(files[node], node, (char *) tables[node].data(), files[node].tableOffset + firstColumn * files[node].flushCount * sizeof(size_t),
                                                      tables[node].size() * sizeof(size_t));
        FileUtil::preadAll(read.fileToRead, read.fileName, read.destination, read.bytesToRead, read.fileOffset);
    }
}

// read columns [first, end) of every node, return the reads for punching
static std::vector<BatchReader::ReadRequest> readTableColumns(const PrefExchangeParts &parts, const std::vector<PrefExchangeFile> &files,
                                                              const std::vector<std::vector<size_t> > &tables, size_t first, size_t end, char *data, bool isAsync, BatchReader &batchReader) {
    std::vector<BatchReader::ReadRequest> readRequests;
    for (size_t node = 0; node < files.size(); node++) {
        size_t flushCount = files[node].flushCount;
        for (size_t flush = 0; flush < flushCount; flush++) {
            size_t start = tables[node][first * flushCount + flush];
            size_t size = tables[node][end * flushCount + flush] - start;
            if (size > 0) {
                readRequests.push_back(parts.getRead(files[node], node, data, start, size));
            }
            data += size;
        }
    }
    if (readRequests.empty()) {
        return readRequests;
    }
    if (isAsync) {
        batchReader.readAsync(readRequests);
    } else {
        batchReader.readSync(readRequests);
    }
    return readRequests;
}

// open a group after the last one closed, return the answer buffer size
static size_t readKeySplitAnswerOffsets(const KeySplitPref &pref, AnswerGroupParts &groupParts, BackgroundRemoval &punches, size_t ownKeySplit,
                                        std::vector<std::vector<size_t> > &offsets) {
    size_t keySplit = getKeySplitIndex(pref, ownKeySplit);
    size_t group = getKeySplitGroup(pref, keySplit);
    const std::vector<PrefExchangeFile> &files = groupParts.groups[group];
    PrefExchangeParts &parts = groupParts.parts[group];
    if (parts.fds.empty()) {
        punches.wait();
        parts = openPrefExchangeFiles(pref, groupParts.prefix, group, files);
    }
    offsets.resize(files.size());
    readTables(parts, files, getKeySplitColumn(pref, group, keySplit), 2, offsets);
    size_t size = 0;
    for (size_t node = 0; node < files.size(); node++) {
        for (size_t flush = 0; flush < files[node].flushCount; flush++) {
            size += offsets[node][files[node].flushCount + flush] - offsets[node][flush];
        }
    }
    return size + sizeof(uint64_t);
}

// return the reads for punching
static std::vector<BatchReader::ReadRequest> readKeySplitAnswers(const KeySplitPref &pref, AnswerGroupParts &groupParts, BackgroundRemoval &punches, size_t ownKeySplit,
                                                             char *&answers, size_t &answerSize, bool isAsync, BatchReader &batchReader) {
    std::vector<std::vector<size_t> > offsets;
    size_t size = readKeySplitAnswerOffsets(pref, groupParts, punches, ownKeySplit, offsets);
    size_t group = getKeySplitGroup(pref, getKeySplitIndex(pref, ownKeySplit));
    delete[] answers;
    answerSize = size;
    answers = new(std::nothrow) char[answerSize];
    Util::checkAllocation(answers, "Can not allocate answers memory in readKeySplitAnswers");
    return readTableColumns(groupParts.parts[group], groupParts.groups[group], offsets, 0, 1, answers, isAsync, batchReader);
}

// a group without own key splits is finished at once
static void countKeySplitAnswers(const Parameters &par, KeySplitPref &pref, AnswerGroupParts &groupParts) {
    size_t ownKeySplits = pref.sendSplitKeySplitOffsets.back();
    for (size_t group = 0; group < groupParts.groups.size(); group++) {
        groupParts.keySplitsLeft[group] = getGroupStart(group + 1, groupParts.groups.size(), ownKeySplits) - getGroupStart(group, groupParts.groups.size(), ownKeySplits);
        if (groupParts.keySplitsLeft[group] == 0) {
            addFinishedItems(par, pref.progress, groupParts.kind, group, 1);
        }
    }
}

// after its reads are queued for punching, nextOwnKeySplit SIZE_MAX if none
static void closeKeySplitAnswers(const Parameters &par, KeySplitPref &pref, AnswerGroupParts &groupParts, BackgroundRemoval &punches, size_t ownKeySplit, size_t nextOwnKeySplit,
                                 bool isReadAgain) {
    size_t group = getKeySplitGroup(pref, getKeySplitIndex(pref, ownKeySplit));
    if (isReadAgain == false && --groupParts.keySplitsLeft[group] == 0) {
        finishPrefExchangeGroup(punches, pref, groupParts.prefix, groupParts.parts[group], groupParts.kind, group);
    } else if (nextOwnKeySplit == SIZE_MAX || getKeySplitGroup(pref, getKeySplitIndex(pref, nextOwnKeySplit)) != group) {
        PrefExchangeParts closed;
        std::swap(closed, groupParts.parts[group]);
        punches.run([closed]() mutable { closePrefExchangeFiles(closed); });
    }
    publishFinishedGroups(par, punches, pref.progress);
}

static void setKeySplitShifts(const KmerExchange &exchange, const KeySplit &keySplit) {
    PackedKmerPosition::idShift = 0;
    PackedKmerPosition::seqLenShift = keySplit.idBits;
    PackedKmerPosition::adjKeyShift = keySplit.idBits;
    PackedKmerPosition::kmerShift = keySplit.idBits + exchange.adjKeyBits;
    PackedKmerPosition::maxSeqLen = 0;
}

template <typename T, bool includeAdjacency, typename L>
static void extractKeySplitKmers(Parameters &par, DBReader<DBKeyType> &seqDbr, BaseMatrix *subMat, const KmerExchange &exchange, const KeySplitPref &pref,
                             const KeySplit &keySplit, KeySplitKmers &keySplitKmers, size_t capacity) {
    setKeySplitShifts(exchange, keySplit);
    PackedKmerPosition *kmers = keySplitKmers.kmers;
    // seed member sequences fill the end of the k-mer array
    readSeedMemberFlushes(par, std::upper_bound(exchange.sendSplitKeyOffsets.begin(), exchange.sendSplitKeyOffsets.end(), keySplit.firstKey) - exchange.sendSplitKeyOffsets.begin() - 1);
    std::vector<size_t> flushFirst(seedMemberFlushes.size());
    std::vector<size_t> flushEnd(seedMemberFlushes.size());
    std::vector<size_t> dataOffsets(seedMemberFlushes.size() + 1, 0);
    std::vector<size_t> memberOffsets(seedMemberFlushes.size() + 1, 0);
    for (size_t flush = 0; flush < seedMemberFlushes.size(); flush++) {
        const SeedMemberFlush &memberFlush = seedMemberFlushes[flush];
        flushFirst[flush] = std::lower_bound(memberFlush.repKeys.begin(), memberFlush.repKeys.end(), static_cast<DBKeyType>(keySplit.firstKey)) - memberFlush.repKeys.begin();
        flushEnd[flush] = std::lower_bound(memberFlush.repKeys.begin(), memberFlush.repKeys.end(), static_cast<DBKeyType>(keySplit.endKey)) - memberFlush.repKeys.begin();
        dataOffsets[flush + 1] = dataOffsets[flush] + memberFlush.offsets[flushEnd[flush]] - memberFlush.offsets[flushFirst[flush]];
        memberOffsets[flush + 1] = memberOffsets[flush] + flushEnd[flush] - flushFirst[flush];
    }
    size_t memberSlots = (dataOffsets.back() + sizeof(PackedKmerPosition) - 1) / sizeof(PackedKmerPosition);
    char *memberData = (char *) (kmers + capacity - memberSlots);
    seedMembers.resize(memberOffsets.back());
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
    for (size_t flush = 0; flush < seedMemberFlushes.size(); flush++) {
        const SeedMemberFlush &memberFlush = seedMemberFlushes[flush];
        const SeedMemberFile &memberFile = seedMemberFiles[memberFlush.file];
        FileUtil::preadAll(memberFile.fd, memberFile.name.c_str(), memberData + dataOffsets[flush], dataOffsets[flush + 1] - dataOffsets[flush], memberFlush.offsets[flushFirst[flush]]);
        for (size_t i = flushFirst[flush]; i < flushEnd[flush]; i++) {
            SeedMember seedMember = { 0, memberFlush.repKeys[i], 0, memberData + dataOffsets[flush] + (memberFlush.offsets[i] - memberFlush.offsets[flushFirst[flush]]),
                                          memberFlush.offsets[i + 1] - memberFlush.offsets[i] };
            seedMembers[memberOffsets[flush] + i - flushFirst[flush]] = seedMember;
        }
    }
    std::vector<size_t> readChunkOffsets = seqDbr.getSplitKeyOffsets(keySplit.firstKey, keySplit.endKey, exchange.readChunkLimit, 0);
    size_t kmerCount = fillKmerPositionArray<Parameters::DBTYPE_AMINO_ACIDS, T, includeAdjacency, L>(
        kmers, capacity - memberSlots, seqDbr, par, subMat, true, 0, SIZE_MAX, NULL, readChunkOffsets).first;
    std::vector<SeedMember>().swap(seedMembers);

    SORT_PARALLEL(kmers, kmers + kmerCount);
    size_t wholeSequenceStart = std::partition_point(kmers, kmers + kmerCount, isKmerRecord) - kmers;
    std::vector<size_t> *rangeOffsets = keySplitKmers.rangeOffsets;
    for (size_t kind = 0; kind < GROUP_KINDS; kind++) {
        rangeOffsets[kind].resize(pref.hashRangeCount + 1);
    }
#pragma omp parallel for schedule(static) num_threads(par.threads)
    for (size_t hashRange = 0; hashRange < pref.hashRangeCount; hashRange++) {
        uint64_t rangeStart = getRowFirstValue(pref, hashRange, 0);
        rangeOffsets[KMER_GROUPS][hashRange] = std::lower_bound(kmers, kmers + wholeSequenceStart, rangeStart, compareKmerHash) - kmers;
        rangeOffsets[WHOLE_SEQUENCE_GROUPS][hashRange] =
            std::lower_bound(kmers + wholeSequenceStart, kmers + kmerCount, rangeStart, compareKmerHash) - kmers;
    }
    rangeOffsets[KMER_GROUPS][pref.hashRangeCount] = wholeSequenceStart;
    rangeOffsets[WHOLE_SEQUENCE_GROUPS][pref.hashRangeCount] = kmerCount;
}

// keptCapacity > 0: extract a new key split into its kept k-mers
template <typename T, bool includeAdjacency, typename L>
static const KeySplitKmers &getKeySplitKmers(Parameters &par, DBReader<DBKeyType> &seqDbr, BaseMatrix *subMat, const KmerExchange &exchange, KeySplitPref &pref,
                                     const KeySplit &keySplit, size_t ownKeySplit, size_t keptCapacity = 0) {
    if (pref.keptKeySplits[ownKeySplit].kmers != NULL && keptCapacity == 0) {
        setKeySplitShifts(exchange, keySplit);
        return pref.keptKeySplits[ownKeySplit];
    }
    KeySplitKmers &keySplitKmers = (keptCapacity > 0) ? pref.keptKeySplits[ownKeySplit] : pref.extractedKeySplit;
    if (keySplitKmers.kmers == NULL) {
        pref.extractedKeySplit.kmers = new(std::nothrow) PackedKmerPosition[pref.keySplitKmerCapacity];
        Util::checkAllocation(pref.extractedKeySplit.kmers, "Can not allocate kmers memory in getKeySplitKmers");
    }
    SequenceCopy &copy = pref.sequenceCopy;
    if (ownKeySplit >= copy.keySplitStates.size()) {
        copy.keySplitStates.resize(ownKeySplit + 1, NOT_COPIED_YET);
        copy.keySplitCopyReads.resize(ownKeySplit + 1);
    }
    unsigned char &copyState = copy.keySplitStates[ownKeySplit];
    std::vector<BatchReader::ReadRequest> &copyReads = copy.keySplitCopyReads[ownKeySplit];
    // kept k-mers are freed if a later send split does not fit
    bool isWritingCopy = copyState == NOT_COPIED_YET && (keptCapacity > 0 || pref.isKeeping == false) && copy.isCopying;
    if (isWritingCopy && copy.fds.empty()) {
        std::string problem = MMseqsNode::getLocalTmpProblem(par);
        size_t freeSize = FileUtil::getFreeSpace(par.localTmp.c_str());
        if (problem.empty() && freeSize == SIZE_MAX) {
            problem = "its free space is unknown";
        }
        if (problem.empty() == false) {
            stopSequenceCopy(par, copy, problem);
        }
        // copies share half of the free space, the needed groups the other half
        copy.maxSize = freeSize / (2 * pref.spill.nodes) + pref.localExchange.heldSize / 2;
        copy.fds.assign(seqDbr.getDataFileCnt(), -1);
        for (size_t file = 0; file < seqDbr.getDataFileCnt(); file++) {
            copy.names.push_back(par.localTmp + "/" + FileUtil::baseName(par.db2) + ".sequences." + SSTR(MMseqsNode::rank) + "." + SSTR(file));
        }
    }
    size_t copySize = 0;
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    for (size_t range = seqDbr.findIdRange(keySplit.firstKey); copy.isCopying && isWritingCopy && range < seqDbr.getIndexSize() && index[range].id < keySplit.endKey; range++) {
        size_t rangeEnd = (range + 1 < seqDbr.getIndexSize()) ? static_cast<size_t>(index[range + 1].id) : seqDbr.getSize();
        size_t begin = index[range].offset + (std::max(static_cast<size_t>(index[range].id), keySplit.firstKey) - index[range].id) * index[range].length;
        size_t end = index[range].offset + (std::min(rangeEnd, keySplit.endKey) - index[range].id) * index[range].length;
        size_t file = seqDbr.getRangeDataFile(range);
        if (copy.fds[file] < 0) {
            // unlinked, read through the fd only
            copy.fds[file] = open(copy.names[file].c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
            if (copy.fds[file] < 0 || unlink(copy.names[file].c_str()) != 0) {
                stopSequenceCopy(par, copy, "cannot create " + copy.names[file] + ": " + strerror(errno));
                break;
            }
        }
        if (copyReads.empty() == false && copyReads.back().fileToRead == copy.fds[file] && copyReads.back().fileOffset + copyReads.back().bytesToRead == begin) {
            copyReads.back().bytesToRead += end - begin;
            copyReads.back().minBytesToRead += end - begin;
        } else {
            BatchReader::ReadRequest copyRead = { copy.fds[file], copy.names[file].c_str(), NULL, begin, end - begin, end - begin };
            copyReads.push_back(copyRead);
        }
        copySize += end - begin;
    }
    if (isWritingCopy && copy.isCopying && copy.size + copySize > copy.maxSize) {
        stopSequenceCopy(par, copy, "the sequences need more than " + MMseqsNode::formatBytes(copy.maxSize) + ", the share of this node in half of its free space");
    }
    if (isWritingCopy && copy.isCopying == false) {
        isWritingCopy = false;
        copyState = NOT_COPIED;
        copyReads.clear();
    }
    if (isWritingCopy) {
        sequenceCopyFds = copy.fds;
    } else if (copyState == COPIED) {
        seqDbr.setLocalCopyFds(copy.fds);
    }
    extractKeySplitKmers<T, includeAdjacency, L>(par, seqDbr, subMat, exchange, pref, keySplit, keySplitKmers, (keptCapacity > 0) ? keptCapacity : pref.keySplitKmerCapacity);
    seqDbr.setLocalCopyFds(std::vector<int>());
    sequenceCopyFds.clear();
    if (isWritingCopy && sequenceCopyError != 0) {
        stopSequenceCopy(par, copy, std::string("cannot write to it: ") + strerror(sequenceCopyError));
        copyState = NOT_COPIED;
        for (size_t i = 0; i < copyReads.size(); i++) {
            FileUtil::punchWholeBlocks(copyReads[i].fileToRead, copyReads[i].fileOffset, copyReads[i].bytesToRead);
        }
        copyReads.clear();
    } else if (isWritingCopy) {
        copyState = COPIED;
        copy.keySplits++;
        copy.size += copySize;
    }
    return keySplitKmers;
}

// k-mer array slots of keys [from, to) in the send split
static size_t getMaxKmerCount(const Parameters &par, DBReader<DBKeyType> &seqDbr, size_t from, size_t to) {
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    float kmersPerSequenceScale = par.kmersPerSequenceScale.values.aminoacid();
    size_t maxKmerCount = 0;
    for (size_t range = seqDbr.findIdRange(from); range < seqDbr.getIndexSize() && index[range].id < to; range++) {
        size_t rangeFrom = std::max(static_cast<size_t>(index[range].id), from);
        size_t rangeTo = std::min(seqDbr.getRangeEndId(range), to);
        size_t seqKmers = computeSeqKmerCount(static_cast<int>(index[range].length - 2), par.kmerSize, par.kmersPerSequence, kmersPerSequenceScale);
        maxKmerCount += countRepresentatives(sendSplitFirstKey, rangeFrom, rangeTo) * seqKmers;
    }
    return maxKmerCount + getSeedMemberSlots(from, to);
}

// keep the k-mers while all key splits fit
template <typename T, bool includeAdjacency, typename L>
static const KeySplitKmers &extractNewKeySplitKmers(Parameters &par, DBReader<DBKeyType> &seqDbr, BaseMatrix *subMat, const KmerExchange &exchange,
                                            KeySplitPref &pref, const KeySplit &keySplit) {
    size_t ownKeySplit = pref.keptKeySplits.size();
    size_t maxKmerCount = getMaxKmerCount(par, seqDbr, keySplit.firstKey, keySplit.endKey);
    pref.maxKmerCounts.push_back(maxKmerCount);

    pref.keptKeySplits.push_back(KeySplitKmers());
    if (pref.isKeeping) {
        KeySplitKmers &kept = pref.keptKeySplits.back();
        kept.kmers = new(std::nothrow) PackedKmerPosition[maxKmerCount];
        Util::checkAllocation(kept.kmers, "Can not allocate kmers memory in extractNewKeySplitKmers");
        pref.keptSize += maxKmerCount;
        return getKeySplitKmers<T, includeAdjacency, L>(par, seqDbr, subMat, exchange, pref, keySplit, ownKeySplit, maxKmerCount);
    }
    return getKeySplitKmers<T, false, L>(par, seqDbr, subMat, exchange, pref, keySplit, ownKeySplit);
}

// fileStart: offset of the first byte of file, own hash range rows go to ownRowFd
template <typename WriteRow>
static void writeKeySplitRows(const Parameters &par, const KeySplitPref &pref, FILE *file, const std::string &fileName, size_t &fileSize, std::vector<size_t> &flushOffsets,
                          size_t fileStart, const std::vector<size_t> &rowBits, int ownRowFd, const char *ownRowFileName, BatchWriter &batchWriter, WriteRow writeRow) {
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
        bool isOwn = hashRange >= pref.hashRangeNodeOffsets[MMseqsNode::rank] && hashRange < pref.hashRangeNodeOffsets[MMseqsNode::rank + 1];
        KmerCodeWriter writer(isOwn ? ownRowFd : fileno(file), isOwn ? ownRowFileName : fileName.c_str(), pref.writeBuffers + thread_idx * KMER_FILE_BUFFER_SIZE, KMER_FILE_BUFFER_SIZE,
                              rowOffsets[hashRange] - fileStart, &batchWriter);
        writeRow(hashRange, writer);
        endCodeBits(writer);
        writeKmerCodeBuffer(writer);
    }
}

// a flush in every group file, rows of other groups empty
template <typename WriteRow>
static void writeKeySplitFlush(const Parameters &par, KeySplitPref &pref, std::vector<PrefExchangeWriter> &writers, const std::vector<size_t> &rowBits, bool mayKeepOwnRows, size_t reservedSize,
                               BatchWriter &batchWriter, WriteRow writeRow) {
    std::vector<size_t> groupRowBits(pref.hashRangeCount);
    std::vector<size_t> flushOffsets;
    size_t ownHashRanges = pref.hashRangeCount / MMseqsNode::count;
    for (size_t group = 0; group < writers.size(); group++) {
        PrefExchangeWriter &writer = writers[group];
        size_t flushSize = 0;
        size_t ownStart = 0;
        size_t ownSize = 0;
        for (size_t hashRange = 0; hashRange < pref.hashRangeCount; hashRange++) {
            groupRowBits[hashRange] = (getHashRangeGroup(pref, hashRange) == group) ? rowBits[hashRange] : 0;
            size_t rowSize = (groupRowBits[hashRange] + CHAR_BIT - 1) / CHAR_BIT;
            if (hashRange < pref.hashRangeNodeOffsets[MMseqsNode::rank]) {
                ownStart += rowSize;
            } else if (hashRange < pref.hashRangeNodeOffsets[MMseqsNode::rank + 1]) {
                ownSize += rowSize;
            }
            flushSize += rowSize;
        }
        writer.startFlush(flushSize, &batchWriter);
        const char *ownRowFileName = writer.partName.c_str();
        int ownRowFd = mayKeepOwnRows ? getOwnRowFd(pref, writer, writer.size + ownStart, ownSize, reservedSize, ownRowFileName) : fileno(writer.file);
        flushOffsets.clear();
        writeKeySplitRows(par, pref, writer.file, writer.partName, writer.size, flushOffsets, writer.partStarts.back() - MMseqsNode::EXCHANGE_FILE_HEADER_BYTES, groupRowBits, ownRowFd, ownRowFileName,
                          batchWriter, writeRow);
        // keep the group columns only
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            std::vector<size_t>::const_iterator first = flushOffsets.begin() + node * ownHashRanges;
            writer.flushOffsets.insert(writer.flushOffsets.end(), first + getGroupStart(group, writers.size(), ownHashRanges), first + getGroupStart(group + 1, writers.size(), ownHashRanges) + 1);
        }
    }
}

// writer NULL: count the bits only
static size_t putFingerprintRow(const KeySplitPref &pref, const PackedKmerPosition *kmers, const std::vector<size_t> *rangeOffsets, const size_t *counts,
                                size_t hashRange, KmerCodeWriter *writer) {
    if (counts[KMER_GROUPS] + counts[WHOLE_SEQUENCE_GROUPS] == 0) {
        return 0;
    }
    size_t bits = getEliasGammaSize(counts[KMER_GROUPS] + 1) + getEliasGammaSize(counts[WHOLE_SEQUENCE_GROUPS] + 1);
    if (writer != NULL) {
        putEliasGamma(*writer, counts[KMER_GROUPS] + 1);
        putEliasGamma(*writer, counts[WHOLE_SEQUENCE_GROUPS] + 1);
    }
    for (size_t kind = 0; kind < GROUP_KINDS; kind++) {
        unsigned int shift = pref.fingerprintShifts[kind];
        unsigned int lowBits = getRowLowBits(pref, hashRange, shift, counts[kind]);
        uint64_t prev = getRowFirstValue(pref, hashRange, shift);
        size_t from = rangeOffsets[kind][hashRange];
        for (size_t pos = from; pos < rangeOffsets[kind][hashRange + 1]; pos++) {
            uint64_t value = getGroupValue(kmers[pos], shift);
            if (pos == from || value != getGroupValue(kmers[pos - 1], shift)) {
                uint64_t gap = (pos == from) ? value - prev : value - prev - 1;
                bits += getRiceSize(gap, lowBits);
                if (writer != NULL) {
                    putRice(*writer, gap, lowBits);
                }
                prev = value;
            }
        }
    }
    return bits;
}

template <typename T, bool includeAdjacency, typename L>
static std::string sendFingerprints(Parameters &par, DBReader<DBKeyType> &seqDbr, BaseMatrix *subMat, const KmerExchange &exchange, KeySplitPref &pref, MMseqsNode::WorkQueue &sendQueue,
                                    size_t &sentSize) {
    std::vector<PrefExchangeWriter> writers = createPrefExchangeWriters(par.db2 + ".fingerprints", pref.hashRangeGroups);
    BatchWriter batchWriter(exchange.writeBehindSize, par.threads, fileno(writers[0].file));
    std::string keySplitLines;
    std::vector<size_t> rowBits(pref.hashRangeCount);
    pref.sendSplitKeySplitOffsets.assign(1, 0);
    std::vector<KeySplit> ownKeySplits;
    Debug::Progress progress(100);
    size_t ownSendSplits = sendQueue.getItemCount(MMseqsNode::rank);
    for (size_t sendSplit = sendQueue.takeNextItem(); sendSplit < exchange.sendSplitCount; sendSplit = sendQueue.takeNextItem()) {
        setSendSplit(par, exchange, sendSplit);
        readSeedMemberFlushes(par, sendSplit);
        size_t dbFrom = exchange.sendSplitKeyOffsets[sendSplit];
        size_t dbTo = exchange.sendSplitKeyOffsets[sendSplit + 1];
        // a key split stays in one target split
        std::vector<size_t> keySplitOffsets(1, dbFrom);
        size_t split = std::upper_bound(pref.targetSplitKeyOffsets.begin(), pref.targetSplitKeyOffsets.end(), dbFrom) - pref.targetSplitKeyOffsets.begin() - 1;
        for (size_t from = dbFrom; from < dbTo; split++) {
            size_t to = std::min(dbTo, pref.targetSplitKeyOffsets[split + 1]);
            addKeySplits(par, seqDbr, exchange, dbFrom, from, to, pref.keySplitKmerCapacity, keySplitOffsets);
            from = to;
        }

        pref.sendSplits.push_back(sendSplit);
        size_t sendSplitKmerCount = 0;
        for (size_t i = 0; i + 1 < keySplitOffsets.size(); i++) {
            if (keySplitOffsets[i + 1] == keySplitOffsets[i] || countRepresentatives(dbFrom, keySplitOffsets[i], keySplitOffsets[i + 1]) == 0) {
                continue;
            }
            KeySplit keySplit = { keySplitOffsets[i], keySplitOffsets[i + 1], getBitWidth(keySplitOffsets[i + 1] - keySplitOffsets[i]) };
            ownKeySplits.push_back(keySplit);
            keySplitLines += SSTR(keySplit.firstKey) + "\t" + SSTR(keySplit.endKey) + "\n";
            sendSplitKmerCount += getMaxKmerCount(par, seqDbr, keySplit.firstKey, keySplit.endKey);
        }
        // keep none of a send split that does not fit, freed k-mers are read again
        if (pref.isKeeping && pref.keptSize + sendSplitKmerCount > pref.keySplitKmerCapacity) {
            freeKeptKeySplits(pref);
            pref.isKeeping = false;
        }

        for (size_t ownKeySplit = pref.sendSplitKeySplitOffsets.back(); ownKeySplit < ownKeySplits.size(); ownKeySplit++) {
            const KeySplit &keySplit = ownKeySplits[ownKeySplit];
            const KeySplitKmers &keySplitKmers = extractNewKeySplitKmers<T, includeAdjacency, L>(par, seqDbr, subMat, exchange, pref, keySplit);
            const PackedKmerPosition *kmers = keySplitKmers.kmers;
            const std::vector<size_t> *rangeOffsets = keySplitKmers.rangeOffsets;
            size_t firstCount = pref.fingerprintCounts.size();
            pref.fingerprintCounts.resize(firstCount + pref.hashRangeCount * GROUP_KINDS);
            size_t *counts = pref.fingerprintCounts.data() + firstCount;
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
            for (size_t hashRange = 0; hashRange < pref.hashRangeCount; hashRange++) {
                for (size_t kind = 0; kind < GROUP_KINDS; kind++) {
                    unsigned int shift = pref.fingerprintShifts[kind];
                    size_t from = rangeOffsets[kind][hashRange];
                    size_t count = 0;
                    for (size_t pos = from; pos < rangeOffsets[kind][hashRange + 1]; pos++) {
                        count += (pos == from || getGroupValue(kmers[pos], shift) != getGroupValue(kmers[pos - 1], shift));
                    }
                    counts[hashRange * GROUP_KINDS + kind] = count;
                }
                rowBits[hashRange] = putFingerprintRow(pref, kmers, rangeOffsets, counts + hashRange * GROUP_KINDS, hashRange, NULL);
            }
            writeKeySplitFlush(par, pref, writers, rowBits, true, 0, batchWriter, [&](size_t hashRange, KmerCodeWriter &writer) {
                putFingerprintRow(pref, kmers, rangeOffsets, counts + hashRange * GROUP_KINDS, hashRange, &writer);
            });
        }
        pref.sendSplitKeySplitOffsets.push_back(ownKeySplits.size());
        MMseqsNode::printProgress(progress, pref.sendSplits.size() - 1, pref.sendSplits.size(), ownSendSplits);
    }
    // other nodes may have taken some of the own send splits
    MMseqsNode::printProgress(progress, pref.sendSplits.size(), ownSendSplits, ownSendSplits);
    seqDbr.releaseDataPages();
    batchWriter.waitWrites();
    return publishPrefExchangeWriters(pref, writers, sentSize) + keySplitLines;
}

struct PrefRowRead {
    KmerCodeReader reader;
    size_t keySplit;
    size_t groupCounts[GROUP_KINDS];
    size_t groupsLeft;
    uint64_t value;
    unsigned int lowBits;
    KmerCodeWriter answer;
    // fingerprint row if codes come below it, NULL if sent whole
    PrefRowRead *fingerprints;
    bool isSameFingerprint;
};

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

static bool startRowValue(const KeySplitPref &pref, size_t hashRange, unsigned int shift, size_t kind, PrefRowRead &read) {
    read.groupsLeft = read.groupCounts[kind];
    read.lowBits = getRowLowBits(pref, hashRange, shift, read.groupsLeft);
    read.value = getRowFirstValue(pref, hashRange, shift);
    if (read.groupsLeft == 0) {
        return false;
    }
    read.value += getRice(read.reader, read.lowBits);
    return true;
}

static bool nextRowValue(PrefRowRead &read) {
    if (--read.groupsLeft == 0) {
        return false;
    }
    read.value += getRice(read.reader, read.lowBits) + 1;
    return true;
}

// codes that share a fingerprint differ in the bits below it
static void readCodeBelowFingerprint(const KeySplitPref &pref, size_t kind, PrefRowRead &read) {
    unsigned int lowBits = pref.fingerprintShifts[kind] - pref.codeShifts[kind];
    read.value = (read.fingerprints->value << lowBits) | getBits(read.reader, lowBits);
    read.isSameFingerprint = getBits(read.reader, lowBits > 0) != 0;
}

static bool startCodeValue(const KeySplitPref &pref, size_t hashRange, size_t kind, PrefRowRead &read) {
    if (read.fingerprints == NULL) {
        return startRowValue(pref, hashRange, pref.codeShifts[kind], kind, read);
    }
    read.groupsLeft = read.groupCounts[kind];
    if (read.groupsLeft == 0) {
        return false;
    }
    startRowValue(pref, hashRange, pref.fingerprintShifts[kind], kind, *read.fingerprints);
    readCodeBelowFingerprint(pref, kind, read);
    return true;
}

static bool nextCodeValue(const KeySplitPref &pref, size_t kind, PrefRowRead &read) {
    if (read.fingerprints == NULL) {
        return nextRowValue(read);
    }
    if (--read.groupsLeft == 0) {
        return false;
    }
    if (read.isSameFingerprint == false) {
        nextRowValue(*read.fingerprints);
    }
    readCodeBelowFingerprint(pref, kind, read);
    return true;
}

static void markFingerprints(const KeySplitPref &pref, size_t hashRange, std::vector<PrefRowRead *> &rows) {
    std::vector<RowValue> heap;
    std::vector<unsigned int> valueRows;
    for (size_t kind = 0; kind < GROUP_KINDS; kind++) {
        unsigned int shift = pref.fingerprintShifts[kind];
        heap.clear();
        for (size_t row = 0; row < rows.size(); row++) {
            if (startRowValue(pref, hashRange, shift, kind, *rows[row])) {
                heap.push_back(RowValue(rows[row]->value, row));
            }
        }
        // RowValue::operator< makes this a min heap
        std::make_heap(heap.begin(), heap.end());
        while (heap.empty() == false) {
            uint64_t value = heap[0].value;
            valueRows.clear();
            while (heap.empty() == false && heap[0].value == value) {
                PrefRowRead &read = *rows[heap[0].row];
                valueRows.push_back(heap[0].row);
                if (nextRowValue(read)) {
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

// write marked fingerprints as a fingerprint row, read starts at its first value
static void putMarkedFingerprintRow(const KeySplitPref &pref, size_t hashRange, PrefRowRead read, KmerCodeReader marks, KmerCodeWriter &writer) {
    size_t markedCounts[GROUP_KINDS] = {0, 0};
    KmerCodeReader countRead = marks;
    for (size_t kind = 0; kind < GROUP_KINDS; kind++) {
        for (size_t left = read.groupCounts[kind]; left > 0;) {
            unsigned int bits = std::min(left, (size_t) KMER_FILE_STEP_BITS);
            markedCounts[kind] += __builtin_popcountll(getBits(countRead, bits));
            left -= bits;
        }
    }
    if (markedCounts[KMER_GROUPS] + markedCounts[WHOLE_SEQUENCE_GROUPS] == 0) {
        return;
    }
    putEliasGamma(writer, markedCounts[KMER_GROUPS] + 1);
    putEliasGamma(writer, markedCounts[WHOLE_SEQUENCE_GROUPS] + 1);
    for (size_t kind = 0; kind < GROUP_KINDS; kind++) {
        unsigned int shift = pref.fingerprintShifts[kind];
        unsigned int lowBits = getRowLowBits(pref, hashRange, shift, markedCounts[kind]);
        uint64_t prev = getRowFirstValue(pref, hashRange, shift);
        bool isFirst = true;
        for (bool hasValue = startRowValue(pref, hashRange, shift, kind, read); hasValue; hasValue = nextRowValue(read)) {
            if (getBits(marks, 1) == 0) {
                continue;
            }
            putRice(writer, isFirst ? read.value - prev : read.value - prev - 1, lowBits);
            prev = read.value;
            isFirst = false;
        }
    }
}

struct FingerprintMarks {
    FingerprintMarks() : data(NULL), dataSize(0) {}
    ~FingerprintMarks() {
        delete[] data;
    }

    char *data;
    size_t dataSize;
    std::vector<size_t> rowOffsets;

    void init(const KeySplitPref &pref, size_t ownKeySplit) {
        rowOffsets.assign(pref.hashRangeCount + 1, 0);
        const size_t *counts = pref.fingerprintCounts.data() + ownKeySplit * pref.hashRangeCount * GROUP_KINDS;
        for (size_t hashRange = 0; hashRange < pref.hashRangeCount; hashRange++) {
            size_t fingerprintCount = counts[hashRange * GROUP_KINDS] + counts[hashRange * GROUP_KINDS + 1];
            rowOffsets[hashRange + 1] = rowOffsets[hashRange] + (fingerprintCount + CHAR_BIT - 1) / CHAR_BIT;
        }
    }
};

struct MarkedGroups {
    MarkedGroups(const PackedKmerPosition *kmers, size_t from, size_t to, unsigned int fingerprintShift, const KmerCodeReader &marks)
        : kmers(kmers), to(to), fingerprintShift(fingerprintShift), marks(marks), groupStart(from), groupEnd(from), isMarked(false), isFirst(true),
          fingerprint(0) {}

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

static void collectNeededGroups(const KeySplitPref &pref, const PackedKmerPosition *kmers, const std::vector<size_t> *rangeOffsets, const FingerprintMarks &marks,
                                size_t hashRange, std::vector<std::pair<size_t, size_t> > *neededGroups) {
    size_t rowOffset = marks.rowOffsets[hashRange];
    KmerCodeReader marksRead = getMemoryCodeReader(marks.data + rowOffset, marks.rowOffsets[hashRange + 1] - rowOffset);
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

// held if they fit, else collected again in every walk
struct NeededGroupLists {
    NeededGroupLists(const Parameters &par, const KeySplitPref &pref, const KeySplitKmers &keySplitKmers, const FingerprintMarks &marks) : pref(pref), keySplitKmers(keySplitKmers), marks(marks) {
        size_t kmerCount = keySplitKmers.rangeOffsets[WHOLE_SEQUENCE_GROUPS][pref.hashRangeCount] - keySplitKmers.rangeOffsets[KMER_GROUPS][0];
        isHeld = 2 * kmerCount * sizeof(std::pair<size_t, size_t>) + getHeldMemory(pref) + (pref.neededLimit - std::min(pref.neededLimit, pref.neededSize)) + marks.dataSize
                 <= pref.keySplitMemory;
        lists.resize((isHeld ? pref.hashRangeCount : par.threads) * GROUP_SEGMENTS);
        isCollected.assign(isHeld ? pref.hashRangeCount : 0, false);
        groupCounts.resize(isHeld ? 0 : pref.hashRangeCount * GROUP_SEGMENTS);
    }

    const std::vector<std::pair<size_t, size_t> > *get(size_t hashRange) {
        if (isHeld) {
            std::vector<std::pair<size_t, size_t> > *groups = lists.data() + hashRange * GROUP_SEGMENTS;
            if (isCollected[hashRange] == false) {
                for (size_t segment = 0; segment < GROUP_SEGMENTS && groupCounts.empty() == false; segment++) {
                    groups[segment].reserve(groupCounts[hashRange * GROUP_SEGMENTS + segment]);
                }
                collectNeededGroups(pref, keySplitKmers.kmers, keySplitKmers.rangeOffsets, marks, hashRange, groups);
                isCollected[hashRange] = true;
            }
            return groups;
        }
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
        std::vector<std::pair<size_t, size_t> > *groups = lists.data() + thread_idx * GROUP_SEGMENTS;
        collectNeededGroups(pref, keySplitKmers.kmers, keySplitKmers.rangeOffsets, marks, hashRange, groups);
        for (size_t segment = 0; segment < GROUP_SEGMENTS; segment++) {
            groupCounts[hashRange * GROUP_SEGMENTS + segment] = groups[segment].size();
        }
        return groups;
    }

    void holdIfFits(size_t rowSize) {
        size_t listSize = 0;
        for (size_t i = 0; i < groupCounts.size(); i++) {
            listSize += groupCounts[i] * sizeof(std::pair<size_t, size_t>);
        }
        if (isHeld || listSize + getHeldMemory(pref) + rowSize + marks.dataSize > pref.keySplitMemory) {
            return;
        }
        std::vector<std::vector<std::pair<size_t, size_t> > >().swap(lists);
        lists.resize(pref.hashRangeCount * GROUP_SEGMENTS);
        isCollected.assign(pref.hashRangeCount, false);
        isHeld = true;
    }

    const KeySplitPref &pref;
    const KeySplitKmers &keySplitKmers;
    const FingerprintMarks &marks;
    bool isHeld;
    std::vector<std::vector<std::pair<size_t, size_t> > > lists;
    std::vector<char> isCollected;
    std::vector<size_t> groupCounts;
};

static std::vector<BatchReader::ReadRequest> readFingerprintMarks(const KeySplitPref &pref, AnswerGroupParts &markParts, BackgroundRemoval &punches, size_t ownKeySplit,
                                                           FingerprintMarks &marks, bool isAsync, BatchReader &batchReader) {
    marks.init(pref, ownKeySplit);
    return readKeySplitAnswers(pref, markParts, punches, ownKeySplit, marks.data, marks.dataSize, isAsync, batchReader);
}

// writer NULL: count the bits only
static size_t putCandidateRow(const KmerExchange &exchange, const KeySplitPref &pref, const KeySplit &keySplit, const PackedKmerPosition *kmers,
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
    // owner kept the marked fingerprints, sent groups follow them in order
    size_t owner = std::upper_bound(pref.hashRangeNodeOffsets.begin(), pref.hashRangeNodeOffsets.end(), hashRange) - pref.hashRangeNodeOffsets.begin() - 1;
    bool isBelowFingerprint = hashRange - pref.hashRangeNodeOffsets[owner] < pref.keptMarkedHashRanges[owner];
    for (size_t kind = 0; kind < GROUP_KINDS; kind++) {
        unsigned int shift = pref.codeShifts[kind];
        unsigned int lowBits = getRowLowBits(pref, hashRange, shift, sentCounts[kind]);
        uint64_t prev = getRowFirstValue(pref, hashRange, shift);
        unsigned int fingerprintShift = pref.fingerprintShifts[kind];
        unsigned int fingerprintLowBits = fingerprintShift - shift;
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
            bits += (isBelowFingerprint ? fingerprintLowBits + (fingerprintLowBits > 0) : getRiceSize(gap, lowBits)) + 1 + candidateCount * keySplit.idBits;
            if (kind == KMER_GROUPS) {
                bits += getEliasGammaSize(candidateCount) + candidateCount * exchange.adjKeyCodeBits;
            }
            if (writer == NULL) {
                continue;
            }
            if (isBelowFingerprint) {
                bool isSameFingerprint = group + 1 < sentCounts[kind]
                                         && getGroupValue(kmers[sentGroups[kind][group + 1].first], fingerprintShift) == getGroupValue(kmers[groupStart], fingerprintShift);
                putBits(*writer, value, fingerprintLowBits);
                putBits(*writer, isSameFingerprint, fingerprintLowBits > 0);
            } else {
                putRice(*writer, gap, lowBits);
            }
            putBits(*writer, hasOtherMember(kmers, groupStart, groupEnd), 1);
            if (kind == KMER_GROUPS) {
                putEliasGamma(*writer, candidateCount);
            }
            for (size_t pos = groupStart; pos < groupEnd; pos++) {
                if (pos == groupStart || (kind == KMER_GROUPS && kmers[pos].getAdjKey() != kmers[pos - 1].getAdjKey())) {
                    if (kind == KMER_GROUPS) {
                        putAdjKeyCode(*writer, exchange, kmers[pos].getAdjKey());
                    }
                    putBits(*writer, kmers[pos].getId(), keySplit.idBits);
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

// writer NULL: count the bits only
static size_t putNeededRow(const KmerExchange &exchange, const KeySplit &keySplit, const PackedKmerPosition *kmers,
                           const std::vector<std::pair<size_t, size_t> > *groups, KmerCodeWriter *writer) {
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
            bits += getEliasGammaSize(members.size()) + members.size() * keySplit.idBits;
            if (writer != NULL) {
                putEliasGamma(*writer, members.size());
                for (size_t member = 0; member < members.size(); member++) {
                    putBits(*writer, members[member], keySplit.idBits);
                }
            }
            if (segment != SENT_KMER_GROUPS && segment != LOCAL_KMER_GROUPS) {
                continue;
            }
            reps.clear();
            for (size_t candidate = 0; candidate < candidates.size(); candidate++) {
                reps.push_back(candidates[candidate].id);
            }
            if (segment == LOCAL_KMER_GROUPS) {
                SORT_SERIAL(reps.begin(), reps.end());
                reps.erase(std::unique(reps.begin(), reps.end()), reps.end());
                // size depends on the rep count only
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

// room for queued spill writes, neededRowSize and needed groups from firstKeySplit on
// bound 2 idBits + 6 bits per k-mer + 11 per row, twice the largest for storeNeededGroups
static size_t getSpillReserve(const KmerExchange &exchange, const KeySplitPref &pref, size_t firstKeySplit, size_t neededRowSize) {
    size_t spillLeft = neededRowSize;
    size_t largestSpill = neededRowSize;
    for (size_t next = firstKeySplit; next < pref.maxKmerCounts.size(); next++) {
        size_t kmerCount = pref.maxKmerCounts[next];
        size_t bound = (kmerCount * (2 * pref.keySplits[getKeySplitIndex(pref, next)].idBits + 6) + 11 * std::min(kmerCount, pref.hashRangeCount) + CHAR_BIT - 1) / CHAR_BIT;
        spillLeft += bound;
        largestSpill = std::max(largestSpill, bound);
    }
    bool maySpill = pref.spill.nodes > 0 && pref.neededSize + spillLeft > pref.neededLimit;
    return std::min(pref.spill.size, exchange.writeBehindSize) + (maySpill ? spillLeft + largestSpill : 0);
}

static void storeNeededGroups(const Parameters &par, const KmerExchange &exchange, KeySplitPref &pref, NeededGroups &needed, const KeySplit &keySplit,
                              const PackedKmerPosition *kmers, NeededGroupLists &groups, const std::vector<size_t> &neededBits, bool isPass3, BatchWriter &batchWriter) {
    std::vector<size_t> rowOffsets(pref.hashRangeCount + 1, 0);
    for (size_t hashRange = 0; hashRange < pref.hashRangeCount; hashRange++) {
        rowOffsets[hashRange + 1] = rowOffsets[hashRange] + (neededBits[hashRange] + CHAR_BIT - 1) / CHAR_BIT;
    }
    size_t size = rowOffsets.back();
    NeededSpill &neededSpill = pref.spill;
    if (isPass3 == false && pref.neededSize + size > pref.neededLimit && neededSpill.nodes > 0) {
        size_t freeSize = FileUtil::getFreeSpace(neededSpill.directory.c_str());
        if (freeSize == SIZE_MAX || 2 * size * neededSpill.nodes > freeSize) {
            MMseqsNode::reportUnusableLocalTmp("Cannot keep k-mer groups in --local-tmp " + neededSpill.directory + ": "
                                               + ((freeSize == SIZE_MAX) ? std::string("its free space is unknown") : "it has " + MMseqsNode::formatBytes(freeSize) + " free"));
            neededSpill.nodes = 0;
        }
    }
    if (isPass3 == false && pref.neededSize + size > pref.neededLimit && neededSpill.nodes > 0) {
        if (neededSpill.file == NULL) {
            neededSpill.file = FileUtil::openAndDelete(neededSpill.name.c_str(), "w+");
        }
        needed.spillOffset = neededSpill.size;
        writeKeySplitRows(par, pref, neededSpill.file, neededSpill.name, neededSpill.size, needed.rowOffsets, 0, neededBits, fileno(neededSpill.file), neededSpill.name.c_str(), batchWriter,
                          [&](size_t hashRange, KmerCodeWriter &writer) {
            putNeededRow(exchange, keySplit, kmers, groups.get(hashRange), &writer);
        });
        return;
    }
    if (isPass3 == false && pref.neededSize + size > pref.neededLimit) {
        return;
    }
    pref.neededSize += isPass3 ? 0 : size;
    needed.rowOffsets.swap(rowOffsets);
    needed.dataSize = size + sizeof(uint64_t);
    needed.data = new(std::nothrow) char[needed.dataSize];
    Util::checkAllocation(needed.data, "Can not allocate needed memory in storeNeededGroups");
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
    for (size_t hashRange = 0; hashRange < pref.hashRangeCount; hashRange++) {
        if (neededBits[hashRange] == 0) {
            continue;
        }
        KmerCodeWriter writer(needed.data, needed.rowOffsets[hashRange + 1], needed.rowOffsets[hashRange]);
        putNeededRow(exchange, keySplit, kmers, groups.get(hashRange), &writer);
        endCodeBits(writer);
    }
}

template <typename T, bool includeAdjacency, typename L>
static std::string sendCandidates(Parameters &par, DBReader<DBKeyType> &seqDbr, BaseMatrix *subMat, const KmerExchange &exchange, KeySplitPref &pref,
                                  const std::vector<std::vector<PrefExchangeFile> > &markGroups, size_t &sentSize) {
    Debug::Progress progress(100);
    std::vector<PrefExchangeWriter> writers = createPrefExchangeWriters(par.db2 + ".candidates", pref.hashRangeGroups);
    BatchWriter batchWriter(exchange.writeBehindSize, par.threads, fileno(writers[0].file));
    AnswerGroupParts markParts(par.db2 + ".marks", MARK_FILES, markGroups);
    countKeySplitAnswers(par, pref, markParts);
    BatchReader batchReader(par.threads, fileno(writers[0].file));
    BackgroundRemoval punches;
    std::vector<size_t> rowBits(pref.hashRangeCount);
    std::vector<size_t> neededBits(pref.hashRangeCount);
    FingerprintMarks marks[2];
    pref.neededGroups.resize(pref.sendSplitKeySplitOffsets.back());
    std::vector<BatchReader::ReadRequest> nextMarkReads;
    if (pref.neededGroups.empty() == false) {
        nextMarkReads = readFingerprintMarks(pref, markParts, punches, 0, marks[0], true, batchReader);
    }
    for (size_t i = 0; i < pref.sendSplits.size(); i++) {
        setSendSplit(par, exchange, pref.sendSplits[i]);
        for (size_t ownKeySplit = pref.sendSplitKeySplitOffsets[i]; ownKeySplit < pref.sendSplitKeySplitOffsets[i + 1]; ownKeySplit++) {
            const KeySplit &keySplit = pref.keySplits[getKeySplitIndex(pref, ownKeySplit)];
            batchReader.waitAsyncRead();
            std::vector<BatchReader::ReadRequest> markReads;
            markReads.swap(nextMarkReads);
            // read the next marks while sending this one
            if (ownKeySplit + 1 < pref.neededGroups.size()) {
                nextMarkReads = readFingerprintMarks(pref, markParts, punches, ownKeySplit + 1, marks[(ownKeySplit + 1) % 2], true, batchReader);
            }
            const KeySplitKmers &keySplitKmers = getKeySplitKmers<T, includeAdjacency, L>(par, seqDbr, subMat, exchange, pref, keySplit, ownKeySplit);
            const PackedKmerPosition *kmers = keySplitKmers.kmers;
            NeededGroupLists groups(par, pref, keySplitKmers, marks[ownKeySplit % 2]);
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
            for (size_t hashRange = 0; hashRange < pref.hashRangeCount; hashRange++) {
                const std::vector<std::pair<size_t, size_t> > *neededGroups = groups.get(hashRange);
                rowBits[hashRange] = putCandidateRow(exchange, pref, keySplit, kmers, neededGroups, hashRange, NULL);
                neededBits[hashRange] = putNeededRow(exchange, keySplit, kmers, neededGroups, NULL);
            }
            size_t neededRowSize = 0;
            for (size_t hashRange = 0; hashRange < pref.hashRangeCount; hashRange++) {
                neededRowSize += (neededBits[hashRange] + CHAR_BIT - 1) / CHAR_BIT;
            }
            groups.holdIfFits((pref.neededSize + neededRowSize <= pref.neededLimit) ? neededRowSize : 0);
            size_t reservedSize = getSpillReserve(exchange, pref, ownKeySplit + 1, neededRowSize);
            // only a rank alone on its device, other spills are not reserved
            writeKeySplitFlush(par, pref, writers, rowBits, pref.localExchange.nodes == 1, reservedSize, batchWriter, [&](size_t hashRange, KmerCodeWriter &writer) {
                putCandidateRow(exchange, pref, keySplit, kmers, groups.get(hashRange), hashRange, &writer);
            });
            storeNeededGroups(par, exchange, pref, pref.neededGroups[ownKeySplit], keySplit, kmers, groups, neededBits, false, batchWriter);
            // writeKeySplitPrefs reads the marks again if extracting again
            bool isExtractedAgain = pref.neededGroups[ownKeySplit].data == NULL && pref.neededGroups[ownKeySplit].spillOffset == SIZE_MAX;
            if (isExtractedAgain == false) {
                punches.punch(markReads);
                punches.punch(takeSequenceCopyReads(pref, ownKeySplit));
            }
            closeKeySplitAnswers(par, pref, markParts, punches, ownKeySplit, (ownKeySplit + 1 < pref.neededGroups.size()) ? ownKeySplit + 1 : SIZE_MAX, isExtractedAgain);
            if (pref.keptKeySplits[ownKeySplit].kmers != NULL) {
                delete[] pref.keptKeySplits[ownKeySplit].kmers;
                pref.keptKeySplits[ownKeySplit].kmers = NULL;
            }
            MMseqsNode::printProgress(progress, ownKeySplit, ownKeySplit + 1, pref.neededGroups.size());
        }
    }
    pref.keptSize = 0;
    seqDbr.releaseDataPages();
    batchWriter.waitWrites();
    punches.wait();
    publishFinishedGroups(par, punches, pref.progress);
    return publishPrefExchangeWriters(pref, writers, sentSize);
}

static size_t getRepAnswerBits(const KmerExchange &exchange, const size_t *groupCounts) {
    size_t repBits = 1 + std::max(exchange.globalIdBits, exchange.adjKeyCodeBits);
    return groupCounts[KMER_GROUPS] * (getEliasGammaSize(exchange.maxRepSeqCount + 1) + exchange.maxRepSeqCount * repBits)
           + groupCounts[WHOLE_SEQUENCE_GROUPS] * (getEliasGammaSize(2) + 1 + exchange.globalIdBits);
}

static void putRepAnswer(KmerCodeWriter &answer, const std::vector<DBKeyType> &reps, const std::vector<size_t> &repIndexes, size_t candidateCount,
                         const std::pair<DBKeyType, DBKeyType> &repKeys) {
    size_t repCount = 0;
    for (size_t rep = 0; rep < reps.size(); rep++) {
        repCount += reps[rep] >= repKeys.first && reps[rep] < repKeys.second;
    }
    putEliasGamma(answer, repCount + 1);
    for (size_t rep = 0; rep < reps.size(); rep++) {
        if (reps[rep] < repKeys.first || reps[rep] >= repKeys.second) {
            continue;
        }
        putBits(answer, repIndexes[rep] != SIZE_MAX, 1);
        if (repIndexes[rep] != SIZE_MAX) {
            putBits(answer, repIndexes[rep], getBitWidth(candidateCount - 1));
        } else {
            putBits(answer, reps[rep] - repKeys.first, getBitWidth(repKeys.second - repKeys.first - 1));
        }
    }
}

static void pickReps(const KmerExchange &exchange, const KeySplitPref &pref, size_t hashRange, std::vector<PrefRowRead *> &rows) {
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
        RowValueQueue queue;
        for (size_t row = 0; row < rows.size(); row++) {
            if (startCodeValue(pref, hashRange, kind, *rows[row])) {
                queue.push(RowValue(rows[row]->value, row));
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
                    const KeySplit &keySplit = pref.keySplits[read.keySplit];
                    for (size_t j = rowCandidateOffsets[i]; j < rowCandidateOffsets[i + 1]; j++) {
                        candidates[j].adjKey = (kind == WHOLE_SEQUENCE_GROUPS) ? 0 : getAdjKeyCode(read.reader, exchange);
                        candidates[j].id = keySplit.firstKey + getBits(read.reader, keySplit.idBits);
                    }
                }
                size_t candidateCount = rowCandidateOffsets.back();
                if (hasTwoMembers && kind == WHOLE_SEQUENCE_GROUPS) {
                    reps.push_back(std::min_element(candidates, candidates + candidateCount)->id);
                } else if (hasTwoMembers) {
                    // first candidate of an adjKey has the smallest id
                    std::copy(candidates, candidates + candidateCount, sortedCandidates);
                    SORT_SERIAL(sortedCandidates, sortedCandidates + candidateCount);
                    selectGroupRepSeqs(reps, exchange.maxRepSeqCount, [&](const auto &visit) {
                        for (size_t j = 0; j < candidateCount; j++) {
                            visit(sortedCandidates[j].adjKey, sortedCandidates[j].id);
                        }
                    }, std::less<DBKeyType>());
                }
                for (size_t i = 0; i < valueRows.size(); i++) {
                    size_t first = rowCandidateOffsets[i];
                    size_t rowCandidateCount = rowCandidateOffsets[i + 1] - first;
                    repIndexes.assign(reps.size(), SIZE_MAX);
                    for (size_t rep = 0; rep < reps.size(); rep++) {
                        for (size_t j = first; j < first + rowCandidateCount; j++) {
                            if (candidates[j].id == reps[rep]) {
                                repIndexes[rep] = j - first;
                                break;
                            }
                        }
                    }
                    putRepAnswer(rows[valueRows[i]]->answer, reps, repIndexes, rowCandidateCount, pref.keySplitRepKeys[rows[valueRows[i]]->keySplit]);
                }
            } else {
                if (hasTwoMembers && kind == WHOLE_SEQUENCE_GROUPS) {
                    DBKeyType rep = 0;
                    for (size_t i = 0; i < valueRows.size(); i++) {
                        const KeySplit &keySplit = pref.keySplits[rows[valueRows[i]]->keySplit];
                        KmerCodeReader candidateRead = candidateStarts[i];
                        DBKeyType id = keySplit.firstKey + getBits(candidateRead, keySplit.idBits);
                        rep = (i == 0) ? id : std::min(rep, id);
                    }
                    reps.push_back(rep);
                } else if (hasTwoMembers) {
                    // in key split order candidates come by (adjKey, id)
                    rowOrder.clear();
                    for (size_t i = 0; i < valueRows.size(); i++) {
                        rowOrder.push_back(std::make_pair(pref.keySplits[rows[valueRows[i]]->keySplit].firstKey, i));
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
                            const KeySplit &keySplit = pref.keySplits[rows[valueRows[rowOrder[candidate.row].second]]->keySplit];
                            DBKeyType id = keySplit.firstKey + getBits(candidateReads[candidate.row], keySplit.idBits);
                            visit(static_cast<unsigned int>(candidate.value), id);
                            if (--candidatesLeft[candidate.row] > 0) {
                                adjKeyQueue.push(RowValue(getAdjKeyCode(candidateReads[candidate.row], exchange), candidate.row));
                            }
                        }
                    }, std::less<DBKeyType>());
                }
                for (size_t i = 0; i < valueRows.size(); i++) {
                    PrefRowRead &read = *rows[valueRows[i]];
                    const KeySplit &keySplit = pref.keySplits[read.keySplit];
                    size_t rowCandidateCount = rowCandidateOffsets[i + 1] - rowCandidateOffsets[i];
                    repIndexes.assign(reps.size(), SIZE_MAX);
                    for (size_t j = 0; j < rowCandidateCount; j++) {
                        if (kind == KMER_GROUPS) {
                            getAdjKeyCode(read.reader, exchange);
                        }
                        DBKeyType id = keySplit.firstKey + getBits(read.reader, keySplit.idBits);
                        for (size_t rep = 0; rep < reps.size(); rep++) {
                            if (repIndexes[rep] == SIZE_MAX && id == reps[rep]) {
                                repIndexes[rep] = j;
                            }
                        }
                    }
                    putRepAnswer(read.answer, reps, repIndexes, rowCandidateCount, pref.keySplitRepKeys[read.keySplit]);
                }
            }
            for (size_t i = 0; i < valueRows.size(); i++) {
                PrefRowRead &read = *rows[valueRows[i]];
                if (nextCodeValue(pref, kind, read)) {
                    queue.push(RowValue(read.value, valueRows[i]));
                }
            }
        }
    }
}

// a pass stays in one group: one read per key split, then local marked fingerprints
static std::vector<BatchReader::ReadRequest> readSentRows(const KeySplitPref &pref, const PrefExchangeParts &parts, const std::vector<PrefExchangeFile> &sentFiles, const std::vector<size_t> &rowOffsets,
                                                          const size_t *groupShifts, size_t groupCount, size_t ownHashRanges, size_t passFirst, size_t passEnd, bool isPickingReps, char *rowData, bool isAsync,
                                                          BatchReader &batchReader) {
    std::vector<BatchReader::ReadRequest> readRequests;
    for (size_t node = 0; node < sentFiles.size(); node++) {
        for (size_t keySplit = pref.nodeKeySplitOffsets[node]; keySplit < pref.nodeKeySplitOffsets[node + 1]; keySplit++) {
            const size_t *offsets = rowOffsets.data() + keySplit * (ownHashRanges + 1);
            size_t readSize = offsets[passEnd] - offsets[passFirst];
            if (readSize > 0) {
                readRequests.push_back(parts.getRead(sentFiles[node], node, rowData, offsets[passFirst] + groupShifts[keySplit * groupCount], readSize));
            }
            rowData += readSize;
        }
    }
    const MarkedFingerprints &marked = pref.marked;
    size_t keySplitCount = pref.keySplits.size();
    if (isPickingReps && marked.fd >= 0 && passFirst < marked.keptHashRanges) {
        size_t markedStart = marked.rowOffsets[passFirst * keySplitCount];
        size_t markedSize = marked.rowOffsets[std::min(passEnd, marked.keptHashRanges) * keySplitCount] - markedStart;
        BatchReader::ReadRequest markedRead = { marked.fd, marked.name.c_str(), rowData, markedStart, markedSize, markedSize };
        if (markedSize > 0) {
            readRequests.push_back(markedRead);
        }
    }
    if (readRequests.empty()) {
        return readRequests;
    }
    if (isAsync) {
        batchReader.readAsync(readRequests);
    } else {
        batchReader.readSync(readRequests);
    }
    return readRequests;
}

// keep sub pass marks (at most bound) in memory, else in --local-tmp; false: stop keeping
static bool reserveMarkedFingerprints(KeySplitPref &pref, size_t bound, size_t memoryLimit, size_t reservedSize, BatchWriter &batchWriter) {
    MarkedFingerprints &marked = pref.marked;
    if (marked.fd < 0 && marked.memorySize + bound <= memoryLimit) {
        return true;
    }
    // only a rank alone on its device, other spills are not reserved
    if (marked.fd < 0 && pref.localExchange.nodes == 1) {
        marked.fd = open(marked.name.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if (marked.fd >= 0 && unlink(marked.name.c_str()) != 0) {
            ::close(marked.fd);
            marked.fd = -1;
            removeExchangeFile(marked.name);
        }
    }
    // in-memory blocks move along
    size_t reserveStart = marked.blocks.empty() ? marked.rowOffsets.back() : 0;
    size_t reserveSize = marked.rowOffsets.back() + bound - reserveStart;
    // reserved first, a full device must not stop the writes
    int failed = -1;
#ifdef __linux__
    if (marked.fd >= 0 && hasLocalRoom(pref, reserveSize, reservedSize)) {
        do {
            failed = (reserveSize > 0) ? fallocate(marked.fd, 0, reserveStart, reserveSize) : 0;
        } while (failed != 0 && errno == EINTR);
    }
#endif
    if (failed != 0) {
        return false;
    }
    for (size_t i = 0; i < marked.blocks.size(); i++) {
        size_t end = (i + 1 < marked.blocks.size()) ? marked.blockOffsets[i + 1] : marked.memorySize;
        batchWriter.write(marked.fd, marked.name.c_str(), marked.blocks[i], end - marked.blockOffsets[i], marked.blockOffsets[i]);
    }
    marked.freeBlocks();
    return true;
}

static std::string answerHashRanges(const Parameters &par, const KmerExchange &exchange, KeySplitPref &pref, size_t sentKind, const std::string &sentPrefix,
                                    const std::vector<std::vector<PrefExchangeFile> > &sentGroups, const std::string &answerPrefix, bool isPickingReps, size_t &receivedSize,
                                    size_t &passCount) {
    Debug::Progress progress(100);
    size_t memoryLimit = exchange.keySplitMemoryLimit - std::min(exchange.keySplitMemoryLimit, getHeldMemory(pref));
    size_t firstHashRange = pref.hashRangeNodeOffsets[MMseqsNode::rank];
    size_t ownHashRanges = pref.hashRangeNodeOffsets[MMseqsNode::rank + 1] - firstHashRange;
    size_t keySplitCount = pref.keySplits.size();
    size_t groupCount = sentGroups.size();
    std::vector<size_t> groupStarts(groupCount + 1);
    for (size_t group = 0; group <= groupCount; group++) {
        groupStarts[group] = getGroupStart(group, groupCount, ownHashRanges);
    }
    // read own table columns, rowOffsets run across groups, groupShifts map back to group files
    std::vector<size_t> rowOffsets(keySplitCount * (ownHashRanges + 1), 0);
    std::vector<size_t> groupShifts(keySplitCount * groupCount, 0);
    for (size_t group = 0; group < groupCount; group++) {
        size_t groupFirst = groupStarts[group];
        size_t groupEnd = groupStarts[group + 1];
        if (groupEnd == groupFirst) {
            continue;
        }
        const std::vector<PrefExchangeFile> &sentFiles = sentGroups[group];
        PrefExchangeParts parts = openPrefExchangeFiles(pref, sentPrefix, group, sentFiles);
        for (size_t node = 0; node < sentFiles.size(); node++) {
            size_t flushCount = sentFiles[node].flushCount;
            std::vector<size_t> table((groupEnd - groupFirst + 1) * flushCount);
            BatchReader::ReadRequest tableRead = parts.getRead(sentFiles[node], node, (char *) table.data(), sentFiles[node].tableOffset + MMseqsNode::rank * (groupEnd - groupFirst + 1) * flushCount * sizeof(size_t),
                                                               table.size() * sizeof(size_t));
            FileUtil::preadAll(tableRead.fileToRead, tableRead.fileName, tableRead.destination, tableRead.bytesToRead, tableRead.fileOffset);
            for (size_t flush = 0; flush < flushCount; flush++) {
                size_t keySplit = pref.nodeKeySplitOffsets[node] + flush;
                size_t *offsets = rowOffsets.data() + keySplit * (ownHashRanges + 1);
                if (groupFirst == 0) {
                    offsets[0] = table[flush];
                }
                size_t shift = table[flush] - offsets[groupFirst];
                groupShifts[keySplit * groupCount + group] = shift;
                for (size_t i = 0; i <= groupEnd - groupFirst; i++) {
                    offsets[groupFirst + i] = table[i * flushCount + flush] - shift;
                }
            }
        }
        closePrefExchangeFiles(parts);
    }
    MarkedFingerprints &marked = pref.marked;
    // picking: own hash ranges [0, markedHashRanges) read their marked fingerprints
    size_t markedHashRanges = isPickingReps ? marked.keptHashRanges : 0;
    bool isLocalMarked = markedHashRanges > 0 && marked.fd >= 0;
    // per row a PrefRowRead and three sizes, one more for marked fingerprints
    std::vector<size_t> hashRangeSizes(ownHashRanges, keySplitCount * (sizeof(PrefRowRead) + 3 * sizeof(size_t)));
    for (size_t keySplit = 0; keySplit < keySplitCount; keySplit++) {
        for (size_t i = 0; i < ownHashRanges; i++) {
            hashRangeSizes[i] += rowOffsets[keySplit * (ownHashRanges + 1) + i + 1] - rowOffsets[keySplit * (ownHashRanges + 1) + i];
        }
    }
    for (size_t i = 0; i < markedHashRanges; i++) {
        hashRangeSizes[i] += keySplitCount * sizeof(PrefRowRead) + (isLocalMarked ? marked.rowOffsets[(i + 1) * keySplitCount] - marked.rowOffsets[i * keySplitCount] : 0);
    }
    size_t ownSize = 0;
    for (size_t i = 0; i < ownHashRanges; i++) {
        ownSize += hashRangeSizes[i];
    }
    // two PrefRowRead per key split and thread, marked fingerprint offsets
    size_t fixedMemory = rowOffsets.size() * sizeof(size_t) + 2 * par.threads * keySplitCount * sizeof(PrefRowRead)
                         + ((isPickingReps == false && marked.isKeeping) ? ownHashRanges * keySplitCount * sizeof(size_t) : 0);
    size_t answerMemory = memoryLimit - std::min(memoryLimit, fixedMemory);
    std::vector<size_t> passOffsets(1, 0);
    std::vector<size_t> passSizes;
    // thirds: rows, next rows, answers
    size_t passLimit = std::min(answerMemory / 3, (ownSize + MIN_ANSWER_PASSES - 1) / MIN_ANSWER_PASSES);
    // marking: room left next to the passes and the peak of sendCandidates
    size_t markedMemoryLimit = 0;
    if (isPickingReps == false && pref.isKeeping) {
        size_t maxKeySplitKmers = 0;
        size_t maxMarksSize = 0;
        for (size_t keySplit = 0; keySplit < pref.keptKeySplits.size(); keySplit++) {
            const std::vector<size_t> *kmerOffsets = pref.keptKeySplits[keySplit].rangeOffsets;
            maxKeySplitKmers = std::max(maxKeySplitKmers, kmerOffsets[WHOLE_SEQUENCE_GROUPS][pref.hashRangeCount] - kmerOffsets[KMER_GROUPS][0]);
            FingerprintMarks marks;
            marks.init(pref, keySplit);
            maxMarksSize = std::max(maxMarksSize, marks.rowOffsets.back() + sizeof(uint64_t));
        }
        size_t sendMemory = getHeldMemory(pref) + pref.neededLimit + 2 * maxKeySplitKmers * sizeof(std::pair<size_t, size_t>) + 2 * maxMarksSize
                            + ownHashRanges * keySplitCount * sizeof(size_t);
        markedMemoryLimit = std::min(answerMemory - std::min(answerMemory, 3 * passLimit), pref.keySplitMemory - std::min(pref.keySplitMemory, sendMemory));
    }
    // reserve the largest needed group spill
    size_t markedReservedSize = isPickingReps ? 0 : getSpillReserve(exchange, pref, 0, 0);
    for (size_t group = 0; group < groupCount; group++) {
        if (groupStarts[group + 1] == groupStarts[group]) {
            continue;
        }
        std::vector<size_t> groupSizes(hashRangeSizes.begin() + groupStarts[group], hashRangeSizes.begin() + groupStarts[group + 1]);
        std::vector<size_t> groupPassOffsets;
        std::vector<size_t> groupPassSizes;
        planPasses(groupSizes, passLimit, groupPassOffsets, groupPassSizes);
        for (size_t pass = 1; pass < groupPassOffsets.size(); pass++) {
            passOffsets.push_back(groupStarts[group] + groupPassOffsets[pass]);
        }
        passSizes.insert(passSizes.end(), groupPassSizes.begin(), groupPassSizes.end());
    }
    size_t rowBufferSize = 0;
    for (size_t pass = 0; pass + 1 < passOffsets.size(); pass++) {
        size_t passRowSize = sizeof(uint64_t);
        for (size_t keySplit = 0; keySplit < keySplitCount; keySplit++) {
            passRowSize += rowOffsets[keySplit * (ownHashRanges + 1) + passOffsets[pass + 1]] - rowOffsets[keySplit * (ownHashRanges + 1) + passOffsets[pass]];
        }
        if (isLocalMarked && passOffsets[pass] < markedHashRanges) {
            passRowSize += marked.rowOffsets[std::min(passOffsets[pass + 1], markedHashRanges) * keySplitCount] - marked.rowOffsets[passOffsets[pass] * keySplitCount];
        }
        rowBufferSize = std::max(rowBufferSize, passRowSize);
    }
    char *rowBuffers = new(std::nothrow) char[2 * rowBufferSize];
    Util::checkAllocation(rowBuffers, "Can not allocate rowBuffers memory in answerHashRanges");
    std::vector<BatchReader::ReadRequest> readRequests;
    std::vector<PrefExchangeWriter> writers = createPrefExchangeWriters(answerPrefix, pref.hashRangeGroups);
    BatchReader *batchReader = NULL;
    std::vector<PrefExchangeParts> groupParts(groupCount);
    size_t nextGroup = 0;
    BackgroundRemoval punches;
    // answer a sub pass while the last one is written
    BatchWriter batchWriter(exchange.writeBehindSize, par.threads, fileno(writers[0].file));
    // marking: keep them only on shared storage
    StorageKind sharedStorage = getStorageKind(fileno(writers[0].file));
    if (isPickingReps == false && (sharedStorage == STORAGE_LOCAL_FLASH || sharedStorage == STORAGE_LOCAL_DISK || sharedStorage == STORAGE_MEMORY)) {
        marked.isKeeping = false;
    }
    if (isPickingReps == false && markedMemoryLimit == 0 && pref.localExchange.nodes != 1) {
        marked.isKeeping = false;
    }
    passCount = passOffsets.size() - 1;
    size_t answeredHashRanges = 0;
    for (size_t pass = 0; pass + 1 < passOffsets.size(); pass++) {
        size_t passFirst = passOffsets[pass];
        size_t passEnd = passOffsets[pass + 1];
        size_t passHashRanges = passEnd - passFirst;

        std::vector<size_t> keySplitDataOffsets(keySplitCount + 1, 0);
        for (size_t keySplit = 0; keySplit < keySplitCount; keySplit++) {
            const size_t *offsets = rowOffsets.data() + keySplit * (ownHashRanges + 1);
            keySplitDataOffsets[keySplit + 1] = keySplitDataOffsets[keySplit] + offsets[passEnd] - offsets[passFirst];
        }
        char *rowData = rowBuffers + (pass % 2) * rowBufferSize;
        if (pass == 0) {
            size_t group = std::upper_bound(groupStarts.begin(), groupStarts.end(), passFirst) - groupStarts.begin() - 1;
            groupParts[group] = openPrefExchangeFiles(pref, sentPrefix, group, sentGroups[group]);
            batchReader = new BatchReader(par.threads, groupParts[group].fds[0][0]);
            readRequests = readSentRows(pref, groupParts[group], sentGroups[group], rowOffsets, groupShifts.data() + group, groupCount, ownHashRanges, passFirst, passEnd, isPickingReps, rowData, false,
                                        *batchReader);
        } else {
            batchReader->waitAsyncRead();
        }
        // every row is read once
        punches.punch(readRequests);
        for (; nextGroup < groupCount && groupStarts[nextGroup + 1] <= passEnd; nextGroup++) {
            finishPrefExchangeGroup(punches, pref, sentPrefix, groupParts[nextGroup], sentKind, nextGroup);
        }
        receivedSize += keySplitDataOffsets[keySplitCount];
        // read the next pass while answering this one
        if (pass + 2 < passOffsets.size()) {
            size_t group = std::upper_bound(groupStarts.begin(), groupStarts.end(), passEnd) - groupStarts.begin() - 1;
            if (groupParts[group].fds.empty()) {
                punches.wait();
                publishFinishedGroups(par, punches, pref.progress);
                groupParts[group] = openPrefExchangeFiles(pref, sentPrefix, group, sentGroups[group]);
            }
            readRequests = readSentRows(pref, groupParts[group], sentGroups[group], rowOffsets, groupShifts.data() + group, groupCount, ownHashRanges, passEnd, passOffsets[pass + 2],
                                        isPickingReps, rowBuffers + ((pass + 1) % 2) * rowBufferSize, true, *batchReader);
        }
        // row of a key split at its first value
        auto readRow = [&](size_t keySplit, size_t i) {
            const size_t *offsets = rowOffsets.data() + keySplit * (ownHashRanges + 1);
            PrefRowRead row = PrefRowRead();
            row.keySplit = keySplit;
            if (offsets[passFirst + i + 1] > offsets[passFirst + i]) {
                row.reader = getMemoryCodeReader(rowData + keySplitDataOffsets[keySplit] + offsets[passFirst + i] - offsets[passFirst], offsets[passFirst + i + 1] - offsets[passFirst + i]);
                row.groupCounts[KMER_GROUPS] = getEliasGamma(row.reader) - 1;
                row.groupCounts[WHOLE_SEQUENCE_GROUPS] = getEliasGamma(row.reader) - 1;
            }
            return row;
        };
        std::vector<PrefRowRead> rows(keySplitCount * passHashRanges);
        std::vector<PrefRowRead> markedRows((passFirst < markedHashRanges) ? keySplitCount * passHashRanges : 0);
        // marking: marked fingerprints fit in the row + 7 bits each
        std::vector<size_t> markedBounds(isPickingReps ? 0 : keySplitCount * passHashRanges);
        std::vector<size_t> answerSizes(keySplitCount * passHashRanges);
        std::vector<size_t> sizeFieldSizes(keySplitCount * passHashRanges);
        std::vector<size_t> rangeAnswerSizes(passHashRanges, 0);
        for (size_t keySplit = 0; keySplit < keySplitCount; keySplit++) {
            const size_t *offsets = rowOffsets.data() + keySplit * (ownHashRanges + 1);
            for (size_t i = 0; i < passHashRanges; i++) {
                PrefRowRead &row = rows[keySplit * passHashRanges + i];
                row = readRow(keySplit, i);
                size_t bits = isPickingReps ? getRepAnswerBits(exchange, row.groupCounts) : row.groupCounts[KMER_GROUPS] + row.groupCounts[WHOLE_SEQUENCE_GROUPS];
                sizeFieldSizes[keySplit * passHashRanges + i] = (isPickingReps && row.groupCounts[KMER_GROUPS] + row.groupCounts[WHOLE_SEQUENCE_GROUPS] > 0) ? sizeof(uint64_t) : 0;
                answerSizes[keySplit * passHashRanges + i] = sizeFieldSizes[keySplit * passHashRanges + i] + (bits + CHAR_BIT - 1) / CHAR_BIT;
                rangeAnswerSizes[i] += answerSizes[keySplit * passHashRanges + i];
                if (isPickingReps == false && marked.isKeeping) {
                    markedBounds[keySplit * passHashRanges + i] = offsets[passFirst + i + 1] - offsets[passFirst + i] + (7 * bits + CHAR_BIT - 1) / CHAR_BIT;
                    rangeAnswerSizes[i] += markedBounds[keySplit * passHashRanges + i];
                }
                if (passFirst + i >= markedHashRanges) {
                    continue;
                }
                size_t markedRow = (passFirst + i) * keySplitCount + keySplit;
                size_t markedStart = marked.rowOffsets[markedRow];
                PrefRowRead &fingerprints = markedRows[keySplit * passHashRanges + i];
                fingerprints = PrefRowRead();
                if (marked.rowOffsets[markedRow + 1] > markedStart) {
                    char *markedData = isLocalMarked ? rowData + keySplitDataOffsets[keySplitCount] + (markedStart - marked.rowOffsets[passFirst * keySplitCount])
                                                     : marked.hashRangeData[passFirst + i] + (markedStart - marked.rowOffsets[(passFirst + i) * keySplitCount]);
                    fingerprints.reader = getMemoryCodeReader(markedData, marked.rowOffsets[markedRow + 1] - markedStart);
                    fingerprints.groupCounts[KMER_GROUPS] = getEliasGamma(fingerprints.reader) - 1;
                    fingerprints.groupCounts[WHOLE_SEQUENCE_GROUPS] = getEliasGamma(fingerprints.reader) - 1;
                }
                row.fingerprints = &fingerprints;
            }
        }

        // answer in sub passes, a flush each
        std::vector<size_t> subPassOffsets;
        std::vector<size_t> subPassSizes;
        planPasses(rangeAnswerSizes, answerMemory - std::min(answerMemory, passSizes[pass] - keySplitDataOffsets[keySplitCount] + 2 * rowBufferSize + (isPickingReps ? 0 : marked.memorySize)),
                   subPassOffsets, subPassSizes);
        for (size_t subPass = 0; subPass + 1 < subPassOffsets.size(); subPass++) {
            size_t subPassFirst = subPassOffsets[subPass];
            size_t subPassHashRanges = subPassOffsets[subPass + 1] - subPassFirst;
            // marking: slots of the bound, fixed before writing
            std::vector<size_t> markedOffsets(isPickingReps ? 0 : subPassHashRanges * keySplitCount + 1, 0);
            for (size_t i = 0; isPickingReps == false && i < subPassHashRanges; i++) {
                for (size_t keySplit = 0; keySplit < keySplitCount; keySplit++) {
                    size_t slot = i * keySplitCount + keySplit;
                    markedOffsets[slot + 1] = markedOffsets[slot] + markedBounds[keySplit * passHashRanges + subPassFirst + i];
                }
            }
            if (isPickingReps == false && marked.isKeeping) {
                marked.isKeeping = reserveMarkedFingerprints(pref, markedOffsets.back(), markedMemoryLimit, markedReservedSize, batchWriter);
            }
            bool isKeepingMarked = isPickingReps == false && marked.isKeeping;
            char *markedData = NULL;
            if (isKeepingMarked) {
                markedData = (char *) malloc(std::max(markedOffsets.back(), (size_t) 1));
                Util::checkAllocation(markedData, "Can not allocate markedData memory in answerHashRanges");
            }
            std::vector<size_t> markedSizes(isKeepingMarked ? subPassHashRanges * keySplitCount : 0);
            std::vector<size_t> answerOffsets(keySplitCount * subPassHashRanges + 1, 0);
            for (size_t keySplit = 0; keySplit < keySplitCount; keySplit++) {
                for (size_t i = 0; i < subPassHashRanges; i++) {
                    answerOffsets[keySplit * subPassHashRanges + i + 1] = answerOffsets[keySplit * subPassHashRanges + i] + answerSizes[keySplit * passHashRanges + subPassFirst + i];
                }
            }
            char *answers = new(std::nothrow) char[answerOffsets.back() + sizeof(uint64_t)];
            Util::checkAllocation(answers, "Can not allocate answers memory in answerHashRanges");
            for (size_t keySplit = 0; keySplit < keySplitCount; keySplit++) {
                for (size_t i = 0; i < subPassHashRanges; i++) {
                    size_t row = keySplit * passHashRanges + subPassFirst + i;
                    size_t answer = keySplit * subPassHashRanges + i;
                    rows[row].answer = KmerCodeWriter(answers, answerOffsets[answer + 1], answerOffsets[answer] + sizeFieldSizes[row]);
                }
            }
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
            for (size_t i = subPassFirst; i < subPassFirst + subPassHashRanges; i++) {
                // copy, neighbor rows share cache lines with other threads
                std::vector<PrefRowRead> threadRows;
                std::vector<PrefRowRead *> hashRangeRows;
                threadRows.reserve(keySplitCount);
                hashRangeRows.reserve(keySplitCount);
                for (size_t keySplit = 0; keySplit < keySplitCount; keySplit++) {
                    threadRows.push_back(rows[keySplit * passHashRanges + i]);
                }
                for (size_t keySplit = 0; keySplit < keySplitCount; keySplit++) {
                    hashRangeRows.push_back(&threadRows[keySplit]);
                }
                if (isPickingReps) {
                    pickReps(exchange, pref, firstHashRange + passFirst + i, hashRangeRows);
                } else {
                    markFingerprints(pref, firstHashRange + passFirst + i, hashRangeRows);
                }
                for (size_t keySplit = 0; keySplit < keySplitCount; keySplit++) {
                    KmerCodeWriter &answer = hashRangeRows[keySplit]->answer;
                    endCodeBits(answer);
                    if (sizeFieldSizes[keySplit * passHashRanges + i] > 0) {
                        uint64_t size = answer.bufferPos;
                        memcpy(answer.buffer - sizeof(uint64_t), &size, sizeof(uint64_t));
                    }
                    rows[keySplit * passHashRanges + i].answer = answer;
                }
                for (size_t keySplit = 0; isKeepingMarked && keySplit < keySplitCount; keySplit++) {
                    size_t slot = (i - subPassFirst) * keySplitCount + keySplit;
                    const KmerCodeWriter &marks = hashRangeRows[keySplit]->answer;
                    KmerCodeWriter markedWriter(markedData, markedOffsets[slot + 1], markedOffsets[slot]);
                    putMarkedFingerprintRow(pref, firstHashRange + passFirst + i, readRow(keySplit, i), getMemoryCodeReader(marks.buffer, marks.bufferPos), markedWriter);
                    endCodeBits(markedWriter);
                    markedSizes[slot] = markedWriter.bufferPos;
                }
                size_t answered = __sync_add_and_fetch(&answeredHashRanges, 1);
                MMseqsNode::printProgress(progress, answered - 1, answered, ownHashRanges);
            }
            if (isKeepingMarked) {
                size_t blockStart = marked.rowOffsets.back();
                size_t blockSize = 0;
                for (size_t slot = 0; slot < markedSizes.size(); slot++) {
                    memmove(markedData + blockSize, markedData + markedOffsets[slot], markedSizes[slot]);
                    blockSize += markedSizes[slot];
                    marked.rowOffsets.push_back(blockStart + blockSize);
                }
                if (marked.fd >= 0) {
                    batchWriter.write(marked.fd, marked.name.c_str(), markedData, blockSize, blockStart);
                    free(markedData);
                } else {
                    markedData = (char *) realloc(markedData, std::max(blockSize, (size_t) 1));
                    Util::checkAllocation(markedData, "Can not allocate markedData memory in answerHashRanges");
                    for (size_t i = 0; i < subPassHashRanges; i++) {
                        marked.hashRangeData.push_back(markedData + (marked.rowOffsets[(passFirst + subPassFirst + i) * keySplitCount] - blockStart));
                    }
                    marked.blocks.push_back(markedData);
                    marked.blockOffsets.push_back(blockStart);
                    marked.memorySize += blockSize;
                }
                marked.keptHashRanges += subPassHashRanges;
            }
            // pack the answers per key split and write behind
            std::vector<size_t> flushStarts(writers.size());
            std::vector<size_t> flushEnds(writers.size());
            for (size_t group = 0; group < writers.size(); group++) {
                flushStarts[group] = writers[group].flushOffsets.size();
                flushEnds[group] = writers[group].size;
            }
            std::vector<size_t> answerStarts(keySplitCount);
            for (size_t keySplit = 0; keySplit < keySplitCount; keySplit++) {
                size_t group = getKeySplitGroup(pref, keySplit);
                answerStarts[keySplit] = flushEnds[group];
                writers[group].flushOffsets.push_back(flushEnds[group]);
                for (size_t i = 0; i < subPassHashRanges; i++) {
                    size_t row = keySplit * passHashRanges + subPassFirst + i;
                    flushEnds[group] += sizeFieldSizes[row] + rows[row].answer.bufferPos;
                }
            }
            // marks stay shared, the needed groups spill to --local-tmp
            size_t ownFirst = pref.nodeKeySplitOffsets[MMseqsNode::rank];
            size_t ownEnd = pref.nodeKeySplitOffsets[MMseqsNode::rank + 1];
            std::vector<const char *> ownNames(writers.size());
            std::vector<int> ownFds(writers.size());
            for (size_t group = 0; group < writers.size(); group++) {
                PrefExchangeWriter &writer = writers[group];
                writer.flushOffsets.push_back(flushEnds[group]);
                writer.startFlush(flushEnds[group] - writer.size, &batchWriter);
                writer.size = flushEnds[group];
                size_t ownStart = writer.flushOffsets[flushStarts[group] + getKeySplitColumn(pref, group, ownFirst)];
                size_t ownSize = writer.flushOffsets[flushStarts[group] + getKeySplitColumn(pref, group, ownEnd)] - ownStart;
                ownNames[group] = writer.partName.c_str();
                ownFds[group] = isPickingReps ? getOwnRowFd(pref, writer, ownStart, ownSize, 0, ownNames[group]) : fileno(writer.file);
            }
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
            for (size_t keySplit = 0; keySplit < keySplitCount; keySplit++) {
                char *keySplitAnswers = answers + answerOffsets[keySplit * subPassHashRanges];
                size_t answerSize = 0;
                for (size_t i = 0; i < subPassHashRanges; i++) {
                    size_t row = keySplit * passHashRanges + subPassFirst + i;
                    size_t size = sizeFieldSizes[row] + rows[row].answer.bufferPos;
                    memmove(keySplitAnswers + answerSize, answers + answerOffsets[keySplit * subPassHashRanges + i], size);
                    answerSize += size;
                }
                size_t group = getKeySplitGroup(pref, keySplit);
                const PrefExchangeWriter &writer = writers[group];
                bool isOwn = keySplit >= ownFirst && keySplit < ownEnd;
                batchWriter.write(isOwn ? ownFds[group] : fileno(writer.file), isOwn ? ownNames[group] : writer.partName.c_str(), keySplitAnswers, answerSize,
                                  writer.getPartOffset(answerStarts[keySplit]));
            }
            delete[] answers;
        }
    }
    delete[] rowBuffers;
    delete batchReader;
    batchWriter.waitWrites();
    for (; nextGroup < groupCount; nextGroup++) {
        finishPrefExchangeGroup(punches, pref, sentPrefix, groupParts[nextGroup], sentKind, nextGroup);
    }
    punches.wait();
    publishFinishedGroups(par, punches, pref.progress);
    if (isPickingReps) {
        marked.close();
    }
    std::string lines;
    for (size_t group = 0; group < writers.size(); group++) {
        lines += writers[group].publish(getKeySplitColumn(pref, group, keySplitCount) + 1);
    }
    return lines;
}

// pieces: piece counts, piecePartitions their partition and start
struct PartitionWriter {
    std::vector<char> data;
    std::vector<uint64_t> indexData;
    size_t nextPartition;
    bool isOpen;
    DBKeyType prevRep;
    std::vector<char> pieces;
    std::vector<std::pair<size_t, size_t> > piecePartitions;
};

static void putPrefKey(std::vector<char> &data, uint64_t key) {
    char buffer[10];
    data.insert(data.end(), buffer, buffer + Util::putPrefKey(buffer, key));
}

static void startPartition(const KeySplitPref &pref, PartitionWriter &writer, size_t partition) {
    for (; writer.nextPartition <= partition; writer.nextPartition++) {
        if (writer.isOpen) {
            writer.data.push_back('\0');
        }
        writer.indexData.push_back(pref.partitionFirstKeys[writer.nextPartition]);
        writer.indexData.push_back(writer.data.size());
        writer.prevRep = pref.partitionFirstKeys[writer.nextPartition];
        writer.isOpen = true;
    }
}

static void closePartitions(const KeySplitPref &pref, PartitionWriter &writer, size_t partitionEnd) {
    if (writer.nextPartition < partitionEnd) {
        startPartition(pref, writer, partitionEnd - 1);
    }
    if (writer.isOpen) {
        writer.data.push_back('\0');
        writer.isOpen = false;
    }
}

static size_t getRepPartition(const KeySplitPref &pref, DBKeyType key) {
    return std::upper_bound(pref.partitionFirstKeys.begin(), pref.partitionFirstKeys.end() - 1, key) - pref.partitionFirstKeys.begin() - 1;
}

struct compareClusterPairs {
    bool operator()(const ClusterPair &first, const ClusterPair &second) const {
        return first.repKey < second.repKey || (first.repKey == second.repKey && first.memberKey < second.memberKey);
    }
};

struct isSameClusterPair {
    bool operator()(const ClusterPair &first, const ClusterPair &second) const {
        return first.repKey == second.repKey && first.memberKey == second.memberKey;
    }
};

// piece bytes, cost, target split and rep count, then per rep its gap and target count
static void putPieceCounts(PartitionWriter &writer, size_t partition, size_t targetSplit, size_t pieceSize, size_t cost, size_t repCount, const std::vector<char> &repCounts) {
    writer.piecePartitions.push_back(std::make_pair(partition, writer.pieces.size()));
    putPrefKey(writer.pieces, pieceSize);
    putPrefKey(writer.pieces, cost);
    putPrefKey(writer.pieces, targetSplit);
    putPrefKey(writer.pieces, repCount);
    writer.pieces.insert(writer.pieces.end(), repCounts.begin(), repCounts.end());
}

static void writeTaskPartitions(const KeySplitPref &pref, DBReader<DBKeyType> &seqDbr, const KeySplit &keySplit, const ClusterPair *pairs, size_t pairCount, PartitionWriter &writer) {
    writer.data.clear();
    writer.indexData.clear();
    writer.pieces.clear();
    writer.piecePartitions.clear();
    size_t targetSplit = std::upper_bound(pref.targetSplitKeyOffsets.begin(), pref.targetSplitKeyOffsets.end(), keySplit.firstKey) - pref.targetSplitKeyOffsets.begin() - 1;
    std::vector<char> repCounts;
    size_t countPartition = SIZE_MAX;
    size_t pieceSize = 0;
    size_t cost = 0;
    size_t repCount = 0;
    DBKeyType prevCountRep = 0;
    for (size_t repStart = 0, repEnd = 0; repStart < pairCount; repStart = repEnd) {
        DBKeyType rep = pairs[repStart].repKey;
        repEnd = repStart + 1;
        while (repEnd < pairCount && pairs[repEnd].repKey == rep) {
            repEnd++;
        }
        size_t partition = getRepPartition(pref, rep);
        if (partition != countPartition) {
            if (repCount > 0) {
                putPieceCounts(writer, countPartition, targetSplit, pieceSize, cost, repCount, repCounts);
            }
            countPartition = partition;
            pieceSize = 0;
            cost = 0;
            repCount = 0;
            repCounts.clear();
            prevCountRep = pref.partitionFirstKeys[partition];
        }
        startPartition(pref, writer, partition);
        size_t pieceStart = writer.data.size();
        putPrefKey(writer.data, rep - writer.prevRep);
        putPrefKey(writer.data, 0);
        DBKeyType prevTarget = keySplit.firstKey - 1;
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
        putPieceCounts(writer, countPartition, targetSplit, pieceSize, cost, repCount, repCounts);
    }
}

static size_t closePrefFile(const std::string &indexName, FILE *dataFile, const std::string &dataName, size_t dataSize, const std::vector<uint64_t> &indexData) {
    FILE *indexFile = FileUtil::openAndDelete(indexName.c_str(), "w");
    size_t indexSize = indexData.size() * sizeof(uint64_t);
    FileUtil::pwriteAll(fileno(indexFile), indexName.c_str(), (const char *) indexData.data(), indexSize, 0);
    MMseqsNode::closeClusterFiles(dataFile, dataName, indexFile, indexName, dataSize, indexSize);
    return dataSize + indexSize + 2 * sizeof(uint64_t);
}

static void appendPartitions(BatchWriter &batchWriter, FILE *dataFile, const std::string &dataName, const PartitionWriter &writer, std::vector<uint64_t> &indexData, size_t &dataSize) {
    for (size_t i = 0; i < writer.indexData.size(); i += 2) {
        indexData.push_back(writer.indexData[i]);
        indexData.push_back(dataSize + writer.indexData[i + 1]);
    }
    batchWriter.write(fileno(dataFile), dataName.c_str(), writer.data.data(), writer.data.size(), dataSize);
    dataSize += writer.data.size();
}

// every member is a target of its reps
template <typename T, bool includeAdjacency, typename L>
static std::string writeKeySplitPrefs(Parameters &par, DBReader<DBKeyType> &seqDbr, BaseMatrix *subMat, const KmerExchange &exchange, KeySplitPref &pref,
                                  const std::vector<std::vector<PrefExchangeFile> > &markGroups, const std::vector<std::vector<PrefExchangeFile> > &repGroups,
                                  size_t &prefSize, size_t &pairCount, size_t &sortedPairCount, size_t &repSplitCount) {
    Debug::Progress progress(100);
    PrefExchangeWriter countWriter(getPrefExchangeName(par.db2 + ".piece_counts", MMseqsNode::rank, 0));
    AnswerGroupParts repParts(par.db2 + ".reps", REP_FILES, repGroups);
    countKeySplitAnswers(par, pref, repParts);
    BatchReader batchReader(par.threads, fileno(countWriter.file));
    BatchReader spillReader(par.threads, fileno(pref.spill.file != NULL ? pref.spill.file : countWriter.file));
    BackgroundRemoval punches;
    BackgroundRemoval spillPunches;
    FingerprintMarks marks;
    char *repAnswers = NULL;
    size_t repAnswerSize = 0;
    char *nextRepAnswers = NULL;
    size_t nextRepAnswerSize = 0;
    std::vector<size_t> repRowOffsets(pref.hashRangeCount);
    size_t partitionCount = pref.partitionFirstKeys.size() - 1;
    std::vector<std::vector<ClusterPair> > threadRecentPairs(par.threads, std::vector<ClusterPair>(RECENT_PAIR_SLOTS));
    std::vector<PartitionWriter> taskWriters;
    BatchWriter batchWriter(exchange.writeBehindSize, par.threads, fileno(countWriter.file));
    // in memory first, then spilled, then extracted again
    std::vector<std::pair<size_t, size_t> > keySplitOrder;
    std::vector<size_t> keySplitSendSplits(pref.sendSplitKeySplitOffsets.back());
    for (size_t i = 0; i < pref.sendSplits.size(); i++) {
        for (size_t ownKeySplit = pref.sendSplitKeySplitOffsets[i]; ownKeySplit < pref.sendSplitKeySplitOffsets[i + 1]; ownKeySplit++) {
            const NeededGroups &needed = pref.neededGroups[ownKeySplit];
            keySplitSendSplits[ownKeySplit] = i;
            size_t source = 2;
            if (needed.data != NULL) {
                source = 0;
            } else if (needed.spillOffset != SIZE_MAX) {
                source = 1;
            }
            keySplitOrder.push_back(std::make_pair(source, ownKeySplit));
        }
    }
    SORT_SERIAL(keySplitOrder.begin(), keySplitOrder.end());
    // remove a spill file after its last key split
    size_t spillsLeft = 0;
    for (size_t keySplit = 0; keySplit < pref.neededGroups.size(); keySplit++) {
        spillsLeft += pref.neededGroups[keySplit].spillOffset != SIZE_MAX;
    }
    // only for extracting again, the other marks may be gone
    AnswerGroupParts markParts(par.db2 + ".marks", MARK_FILES, markGroups);
    for (size_t order = 0; order < keySplitOrder.size(); order++) {
        markParts.keySplitsLeft[getKeySplitGroup(pref, getKeySplitIndex(pref, keySplitOrder[order].second))] += keySplitOrder[order].first == 2;
    }
    std::vector<size_t> neededBits(pref.hashRangeCount);
    std::vector<BatchReader::ReadRequest> spillReads;
    std::vector<BatchReader::ReadRequest> markReads;
    std::vector<BatchReader::ReadRequest> repReads;
    std::vector<BatchReader::ReadRequest> nextRepReads;
    // needed groups and reps, only marks if extracting again
    auto startKeySplitReads = [&](size_t order, bool isRepRead) {
        NeededGroups &needed = pref.neededGroups[keySplitOrder[order].second];
        if (keySplitOrder[order].first == 2) {
            markReads = readFingerprintMarks(pref, markParts, punches, keySplitOrder[order].second, marks, true, batchReader);
            return;
        }
        if (keySplitOrder[order].first == 1) {
            size_t first = needed.rowOffsets[0];
            needed.dataSize = needed.rowOffsets.back() - first + sizeof(uint64_t);
            needed.data = new(std::nothrow) char[needed.dataSize];
            Util::checkAllocation(needed.data, "Can not allocate needed memory in writeKeySplitPrefs");
            NeededSpill &neededSpill = pref.spill;
            spillReads.clear();
            BatchReader::ReadRequest readRequest = {fileno(neededSpill.file), neededSpill.name.c_str(), needed.data, first, needed.rowOffsets.back() - first, needed.rowOffsets.back() - first};
            if (readRequest.bytesToRead > 0) {
                spillReads.push_back(readRequest);
                spillReader.readAsync(spillReads);
            }
        }
        if (isRepRead) {
            repReads = readKeySplitAnswers(pref, repParts, punches, keySplitOrder[order].second, repAnswers, repAnswerSize, true, batchReader);
        }
    };
    if (keySplitOrder.empty() == false) {
        startKeySplitReads(0, true);
    }
    // write a key split while the next collects pairs, close before it writes
    std::string dataNames[2];
    FILE *pendingDataFile = NULL;
    const std::string *pendingDataName = NULL;
    std::string pendingIndexName;
    size_t pendingDataSize = 0;
    std::vector<uint64_t> pendingIndexData;
    auto closePendingPrefFile = [&]() {
        if (pendingDataFile == NULL) {
            return;
        }
        batchWriter.waitWrites();
        prefSize += closePrefFile(pendingIndexName, pendingDataFile, *pendingDataName, pendingDataSize, pendingIndexData);
        pendingDataFile = NULL;
        std::vector<uint64_t>().swap(pendingIndexData);
    };
    for (size_t order = 0; order < keySplitOrder.size(); order++) {
        size_t ownKeySplit = keySplitOrder[order].second;
        size_t nextOwnKeySplit = (order + 1 < keySplitOrder.size()) ? keySplitOrder[order + 1].second : SIZE_MAX;
        setSendSplit(par, exchange, pref.sendSplits[keySplitSendSplits[ownKeySplit]]);
        size_t keySplitIndex = getKeySplitIndex(pref, ownKeySplit);
        const KeySplit &keySplit = pref.keySplits[keySplitIndex];
        NeededGroups &needed = pref.neededGroups[ownKeySplit];

        // read the needed groups
        if (keySplitOrder[order].first == 1) {
            spillReader.waitAsyncRead();
            NeededSpill &neededSpill = pref.spill;
            size_t first = needed.rowOffsets[0];
            for (size_t hashRange = 0; hashRange <= pref.hashRangeCount; hashRange++) {
                needed.rowOffsets[hashRange] -= first;
            }
            if (--spillsLeft > 0) {
                spillPunches.punch(spillReads);
            } else {
                // the other punches use the file
                spillPunches.wait();
                FILE *spillFile = neededSpill.file;
                std::string spillName = neededSpill.name;
                neededSpill.file = NULL;
                spillPunches.run([spillFile, spillName]() {
                    if (fclose(spillFile) != 0) {
                        Debug(Debug::ERROR) << "Cannot close " << spillName << "\n";
                        EXIT(EXIT_FAILURE);
                    }
                    removeExchangeFile(spillName);
                });
            }
        } else if (keySplitOrder[order].first == 2) {
            batchReader.waitAsyncRead();
            punches.punch(markReads);
            closeKeySplitAnswers(par, pref, markParts, punches, ownKeySplit, nextOwnKeySplit, false);
            // the reps are read while the k-mers are extracted
            repReads = readKeySplitAnswers(pref, repParts, punches, ownKeySplit, repAnswers, repAnswerSize, true, batchReader);
            const KeySplitKmers &keySplitKmers = getKeySplitKmers<T, includeAdjacency, L>(par, seqDbr, subMat, exchange, pref, keySplit, ownKeySplit);
            punches.punch(takeSequenceCopyReads(pref, ownKeySplit));
            NeededGroupLists groups(par, pref, keySplitKmers, marks);
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
            for (size_t hashRange = 0; hashRange < pref.hashRangeCount; hashRange++) {
                neededBits[hashRange] = putNeededRow(exchange, keySplit, keySplitKmers.kmers, groups.get(hashRange), NULL);
            }
            storeNeededGroups(par, exchange, pref, needed, keySplit, keySplitKmers.kmers, groups, neededBits, true, batchWriter);
        }

        // read the reps
        batchReader.waitAsyncRead();
        punches.punch(repReads);
        closeKeySplitAnswers(par, pref, repParts, punches, ownKeySplit, nextOwnKeySplit, false);
        // rows of sent groups start with their size
        for (size_t hashRange = 0, pos = 0; hashRange < pref.hashRangeCount; hashRange++) {
            bool hasSent = false;
            if (needed.rowOffsets[hashRange + 1] > needed.rowOffsets[hashRange]) {
                KmerCodeReader neededRead = getMemoryCodeReader(needed.data + needed.rowOffsets[hashRange], needed.rowOffsets[hashRange + 1] - needed.rowOffsets[hashRange]);
                hasSent = getEliasGamma(neededRead) > 1;
                hasSent |= getEliasGamma(neededRead) > 1;
            }
            uint64_t rowSize = 0;
            if (hasSent) {
                memcpy(&rowSize, repAnswers + pos, sizeof(uint64_t));
                pos += sizeof(uint64_t);
            }
            repRowOffsets[hashRange] = pos;
            pos += rowSize;
        }
        if (keySplitOrder[order].first == 2) {
            delete[] pref.extractedKeySplit.kmers;
            pref.extractedKeySplit.kmers = NULL;
            seqDbr.releaseDataPages();
        }
        DBKeyType keySplitRepBegin = pref.keySplitRepKeys[keySplitIndex].first;
        DBKeyType keySplitRepEnd = pref.keySplitRepKeys[keySplitIndex].second;
        // pending writes of the last key split use the other name
        std::string &dataName = dataNames[order % 2];
        dataName = par.db2 + "." + SSTR(pref.keySplitFiles[keySplitIndex]);
        FILE *dataFile = FileUtil::openAndDelete(dataName.c_str(), "w");
        size_t dataSize = 0;
        std::vector<uint64_t> indexData;
        size_t usedMemory = getHeldMemory(pref) + ((keySplitOrder[order].first == 0) ? 0 : needed.dataSize) + repAnswerSize + partitionCount * (3 * sizeof(uint64_t) + 1 + 4 * getPrefKeySize(SIZE_MAX));
        size_t pairBytes = sizeof(ClusterPair) + 4 * getPrefKeySize(seqDbr.getSize()) + 2;
        // a rep keeps its pairs in one rep split
        auto getSplitPairLimit = [&](size_t memory) {
            return std::max((exchange.keySplitMemoryLimit - std::min(exchange.keySplitMemoryLimit, memory)) / pairBytes, 2 * (keySplit.endKey - keySplit.firstKey)) / 2;
        };
        std::vector<size_t> hashRangePairBounds(pref.hashRangeCount);
        size_t pairBound = 0;
        for (size_t hashRange = 0; hashRange < pref.hashRangeCount; hashRange++) {
            // a member takes at least idBits of a row
            hashRangePairBounds[hashRange] = (needed.rowOffsets[hashRange + 1] - needed.rowOffsets[hashRange]) * CHAR_BIT / std::max(1u, keySplit.idBits) * exchange.maxRepSeqCount;
            pairBound += hashRangePairBounds[hashRange];
        }
        // read the next reps while collecting pairs if their buffer fits next to the pairs and adds no rep split, the marks of extracting again come first
        bool isNextRepRead = order + 1 < keySplitOrder.size() && keySplitOrder[order + 1].first != 2;
        if (isNextRepRead) {
            std::vector<std::vector<size_t> > nextOffsets;
            size_t nextSize = readKeySplitAnswerOffsets(pref, repParts, punches, nextOwnKeySplit, nextOffsets);
            size_t repSplitLimit = getSplitPairLimit(usedMemory);
            size_t readAheadLimit = getSplitPairLimit(usedMemory + nextSize);
            isNextRepRead = usedMemory + nextSize + 2 * readAheadLimit * pairBytes <= exchange.keySplitMemoryLimit
                            && (pairBound + readAheadLimit - 1) / readAheadLimit == (pairBound + repSplitLimit - 1) / repSplitLimit;
        }
        if (isNextRepRead) {
            nextRepReads = readKeySplitAnswers(pref, repParts, punches, nextOwnKeySplit, nextRepAnswers, nextRepAnswerSize, true, batchReader);
            usedMemory += nextRepAnswerSize;
        }
        size_t splitPairLimit = getSplitPairLimit(usedMemory);
        // a chunk adds at most half of the capacity
        std::vector<size_t> chunkHashRanges(1, 0);
        std::vector<size_t> chunkPairBounds(1, 0);
        for (size_t hashRange = 0; hashRange < pref.hashRangeCount; hashRange++) {
            if (hashRange > chunkHashRanges.back() && chunkPairBounds.back() + hashRangePairBounds[hashRange] > splitPairLimit) {
                chunkHashRanges.push_back(hashRange);
                chunkPairBounds.push_back(0);
            }
            chunkPairBounds.back() += hashRangePairBounds[hashRange];
        }
        chunkHashRanges.push_back(pref.hashRangeCount);
        size_t arraySize = splitPairLimit + *std::max_element(chunkPairBounds.begin(), chunkPairBounds.end()) + par.threads * PAIR_BLOCK_SIZE;
        ClusterPair *pairs = new(std::nothrow) ClusterPair[arraySize];
        Util::checkAllocation(pairs, "Can not allocate pairs memory in writeKeySplitPrefs");
        const ClusterPair sentinel = { DB_KEY_INVALID, DB_KEY_INVALID };
        // a rep split may end inside a partition
        PartitionWriter keySplitWriter;
        keySplitWriter.nextPartition = 0;
        keySplitWriter.isOpen = false;
        keySplitWriter.prevRep = 0;
        // one piece counts flush per key split
        size_t countColumns = 0;
        for (DBKeyType repBegin = keySplitRepBegin; repBegin < keySplitRepEnd; repSplitCount++) {
            // collect the pairs of a rep split
            // a dropped pair may come again in the next rep split
#pragma omp parallel for schedule(static) num_threads(par.threads)
            for (size_t thread = 0; thread < threadRecentPairs.size(); thread++) {
                std::fill(threadRecentPairs[thread].begin(), threadRecentPairs[thread].end(), sentinel);
            }
            DBKeyType repEnd = keySplitRepEnd;
            size_t pairEnd = 0;
            for (size_t chunk = 0; chunk + 1 < chunkHashRanges.size(); chunk++) {
                if (pairEnd + chunkPairBounds[chunk] + par.threads * PAIR_BLOCK_SIZE > arraySize) {
                    SORT_PARALLEL(pairs, pairs + pairEnd, compareClusterPairs());
                    pairEnd = std::lower_bound(pairs, pairs + pairEnd, sentinel, compareClusterPairs()) - pairs;
                    // unique each thread range, a range drops the pairs equal to the last input pair before it
                    std::vector<ClusterPair> rangeLastPairs(par.threads, sentinel);
                    for (size_t range = 1; range < (size_t) par.threads; range++) {
                        size_t rangeBegin = pairEnd * range / par.threads;
                        if (rangeBegin > 0) {
                            rangeLastPairs[range] = pairs[rangeBegin - 1];
                        }
                    }
                    std::vector<size_t> uniqueStarts(par.threads);
                    std::vector<size_t> uniqueOffsets(par.threads + 1, 0);
#pragma omp parallel for schedule(static, 1) num_threads(par.threads)
                    for (size_t range = 0; range < (size_t) par.threads; range++) {
                        size_t rangeBegin = pairEnd * range / par.threads;
                        size_t rangeEnd = pairEnd * (range + 1) / par.threads;
                        while (rangeBegin < rangeEnd && isSameClusterPair()(pairs[rangeBegin], rangeLastPairs[range])) {
                            rangeBegin++;
                        }
                        uniqueStarts[range] = rangeBegin;
                        uniqueOffsets[range + 1] = std::unique(pairs + rangeBegin, pairs + rangeEnd, isSameClusterPair()) - (pairs + rangeBegin);
                    }
                    size_t moveBegin = pairEnd;
                    for (size_t range = 0; range < (size_t) par.threads; range++) {
                        if (moveBegin == pairEnd && uniqueStarts[range] != uniqueOffsets[range]) {
                            moveBegin = uniqueOffsets[range];
                        }
                        uniqueOffsets[range + 1] += uniqueOffsets[range];
                    }
                    pairEnd = uniqueOffsets[par.threads];
                    // pairs only move left, so a round reads all its blocks before it writes them
#pragma omp parallel num_threads(par.threads)
                    {
                        unsigned int thread_idx = 0;
                        size_t teamSize = 1;
#ifdef OPENMP
                        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
                        teamSize = static_cast<size_t>(omp_get_num_threads());
#endif
                        std::vector<ClusterPair> moveBuffer(PAIR_BLOCK_SIZE);
                        for (size_t round = moveBegin; round < pairEnd; round += teamSize * PAIR_BLOCK_SIZE) {
                            size_t blockBegin = std::min(pairEnd, round + thread_idx * PAIR_BLOCK_SIZE);
                            size_t blockEnd = std::min(pairEnd, blockBegin + PAIR_BLOCK_SIZE);
                            for (size_t pos = blockBegin; pos < blockEnd;) {
                                size_t range = std::upper_bound(uniqueOffsets.begin(), uniqueOffsets.end(), pos) - uniqueOffsets.begin() - 1;
                                size_t count = std::min(blockEnd, uniqueOffsets[range + 1]) - pos;
                                memcpy(moveBuffer.data() + (pos - blockBegin), pairs + uniqueStarts[range] + (pos - uniqueOffsets[range]), count * sizeof(ClusterPair));
                                pos += count;
                            }
#pragma omp barrier
                            memcpy(pairs + blockBegin, moveBuffer.data(), (blockEnd - blockBegin) * sizeof(ClusterPair));
#pragma omp barrier
                        }
                    }
                    // end at half of the capacity, keep one rep
                    ClusterPair firstRepEnd = { pairs[0].repKey, DB_KEY_INVALID };
                    const ClusterPair *cut = pairs + pairEnd;
                    if (pairEnd > splitPairLimit) {
                        cut = std::max(pairs + splitPairLimit, std::upper_bound(pairs, pairs + pairEnd, firstRepEnd, compareClusterPairs()));
                    }
                    if (cut < pairs + pairEnd) {
                        repEnd = cut->repKey;
                        ClusterPair end = { repEnd, 0 };
                        pairEnd = std::lower_bound(pairs, pairs + pairEnd, end, compareClusterPairs()) - pairs;
                    }
                }
                size_t chunkStart = chunkHashRanges[chunk];
                size_t chunkEnd = chunkHashRanges[chunk + 1];
#pragma omp parallel num_threads(par.threads)
                {
                    unsigned int thread_idx = 0;
#ifdef OPENMP
                    thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
                    // claim blocks, pad the last with the sentinel
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
                        KmerCodeReader neededRead = getMemoryCodeReader(needed.data + needed.rowOffsets[hashRange], needed.rowOffsets[hashRange + 1] - needed.rowOffsets[hashRange]);
                        KmerCodeReader repRead = getMemoryCodeReader(repAnswers + repRowOffsets[hashRange], repAnswerSize - repRowOffsets[hashRange]);
                        size_t segmentGroups[GROUP_SEGMENTS];
                        for (size_t segment = 0; segment < GROUP_SEGMENTS; segment++) {
                            segmentGroups[segment] = getEliasGamma(neededRead) - 1;
                        }
                        for (size_t segment = 0; segment < GROUP_SEGMENTS; segment++) {
                            for (size_t group = 0; group < segmentGroups[segment]; group++) {
                                members.resize(getEliasGamma(neededRead));
                                for (size_t member = 0; member < members.size(); member++) {
                                    members[member] = keySplit.firstKey + getBits(neededRead, keySplit.idBits);
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
                                    size_t repCount = getEliasGamma(repRead) - 1;
                                    for (size_t rep = 0; rep < repCount; rep++) {
                                        bool isOwn = getBits(repRead, 1) != 0;
                                        reps.push_back(isOwn ? members[candidateMembers[getBits(repRead, getBitWidth(candidateMembers.size() - 1))]]
                                                            : keySplitRepBegin + getBits(repRead, getBitWidth(keySplitRepEnd - keySplitRepBegin - 1)));
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
                                        if (isSameClusterPair()(recent, pair)) {
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
            }
            SORT_PARALLEL(pairs, pairs + pairEnd, compareClusterPairs());
            pairEnd = std::lower_bound(pairs, pairs + pairEnd, sentinel, compareClusterPairs()) - pairs;
            sortedPairCount += pairEnd;

            // write the partitions in tasks of about equal pairs
            size_t firstInnerPartition = getRepPartition(pref, repBegin) + 1;
            size_t endPartition = std::lower_bound(pref.partitionFirstKeys.begin(), pref.partitionFirstKeys.end(), static_cast<size_t>(repEnd)) - pref.partitionFirstKeys.begin();
            size_t taskCount = std::max((size_t) 1, std::min(endPartition - std::min(endPartition, firstInnerPartition) + 1, TASKS_PER_THREAD * par.threads));
            std::vector<DBKeyType> taskReps(1, repBegin);
            for (size_t task = 1; task < taskCount; task++) {
                size_t low = std::max(firstInnerPartition, getRepPartition(pref, taskReps.back()) + 1);
                size_t high = endPartition;
                while (low < high) {
                    size_t partition = low + (high - low) / 2;
                    ClusterPair first = { static_cast<DBKeyType>(pref.partitionFirstKeys[partition]), 0 };
                    if (static_cast<size_t>(std::lower_bound(pairs, pairs + pairEnd, first, compareClusterPairs()) - pairs) < task * pairEnd / taskCount) {
                        low = partition + 1;
                    } else {
                        high = partition;
                    }
                }
                if (low < endPartition && pref.partitionFirstKeys[low] < repEnd) {
                    taskReps.push_back(pref.partitionFirstKeys[low]);
                }
            }
            taskReps.push_back(repEnd);
            taskCount = taskReps.size() - 1;
            taskWriters.resize(taskCount);
            size_t repSplitPairCount = 0;
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads) reduction(+:repSplitPairCount)
            for (size_t task = 0; task < taskCount; task++) {
                ClusterPair first = { taskReps[task], 0 };
                ClusterPair end = { taskReps[task + 1], 0 };
                ClusterPair *taskPairs = std::lower_bound(pairs, pairs + pairEnd, first, compareClusterPairs());
                size_t taskPairCount = std::unique(taskPairs, std::lower_bound(taskPairs, pairs + pairEnd, end, compareClusterPairs()), isSameClusterPair()) - taskPairs;
                PartitionWriter &writer = taskWriters[task];
                if (task == 0) {
                    writer.nextPartition = keySplitWriter.nextPartition;
                    writer.isOpen = keySplitWriter.isOpen;
                    writer.prevRep = keySplitWriter.prevRep;
                } else {
                    // the task before ends at the first partition that starts at this rep
                    writer.nextPartition = std::lower_bound(pref.partitionFirstKeys.begin(), pref.partitionFirstKeys.end(), static_cast<size_t>(taskReps[task])) - pref.partitionFirstKeys.begin();
                    writer.isOpen = false;
                    writer.prevRep = taskReps[task];
                }
                writeTaskPartitions(pref, seqDbr, keySplit, taskPairs, taskPairCount, writer);
                repSplitPairCount += taskPairCount;
                size_t taskEndPartition = std::lower_bound(pref.partitionFirstKeys.begin(), pref.partitionFirstKeys.end(), static_cast<size_t>(taskReps[task + 1])) - pref.partitionFirstKeys.begin();
                if (pref.partitionFirstKeys[taskEndPartition] == taskReps[task + 1]) {
                    closePartitions(pref, writer, taskEndPartition);
                } else {
                    startPartition(pref, writer, taskEndPartition - 1);
                }
            }
            pairCount += repSplitPairCount;
            // tasks follow each other in both files
            std::vector<size_t> taskDataOffsets(taskCount);
            std::vector<size_t> taskPieceOffsets(taskCount);
            size_t repSplitPieceSize = 0;
            for (size_t task = 0; task < taskCount; task++) {
                repSplitPieceSize += taskWriters[task].pieces.size();
            }
            closePendingPrefFile();
            // end the flush early if the rep split would pass the part
            if (countColumns > 0 && countWriter.getPartOffset(countWriter.size) + repSplitPieceSize > EXCHANGE_PART_SIZE) {
                for (; countColumns <= partitionCount; countColumns++) {
                    countWriter.flushOffsets.push_back(countWriter.size);
                }
                countColumns = 0;
            }
            if (countColumns == 0) {
                countWriter.startFlush(repSplitPieceSize, &batchWriter);
            }
            for (size_t task = 0; task < taskCount; task++) {
                const PartitionWriter &writer = taskWriters[task];
                for (size_t i = 0; i < writer.indexData.size(); i += 2) {
                    indexData.push_back(writer.indexData[i]);
                    indexData.push_back(dataSize + writer.indexData[i + 1]);
                }
                for (size_t i = 0; i < writer.piecePartitions.size(); i++) {
                    for (; countColumns <= writer.piecePartitions[i].first; countColumns++) {
                        countWriter.flushOffsets.push_back(countWriter.size + writer.piecePartitions[i].second);
                    }
                }
                taskDataOffsets[task] = dataSize;
                taskPieceOffsets[task] = countWriter.getPartOffset(countWriter.size);
                dataSize += writer.data.size();
                countWriter.size += writer.pieces.size();
            }
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
            for (size_t task = 0; task < taskCount; task++) {
                PartitionWriter &writer = taskWriters[task];
                batchWriter.write(fileno(dataFile), dataName.c_str(), writer.data.data(), writer.data.size(), taskDataOffsets[task]);
                batchWriter.write(fileno(countWriter.file), countWriter.partName.c_str(), writer.pieces.data(), writer.pieces.size(), taskPieceOffsets[task]);
                std::vector<char>().swap(writer.data);
                std::vector<char>().swap(writer.pieces);
            }
            keySplitWriter.nextPartition = taskWriters[taskCount - 1].nextPartition;
            keySplitWriter.isOpen = taskWriters[taskCount - 1].isOpen;
            keySplitWriter.prevRep = taskWriters[taskCount - 1].prevRep;
            repBegin = repEnd;
        }
        delete[] pairs;
        for (; countColumns <= partitionCount; countColumns++) {
            countWriter.flushOffsets.push_back(countWriter.size);
        }
        keySplitWriter.data.clear();
        keySplitWriter.indexData.clear();
        closePendingPrefFile();
        closePartitions(pref, keySplitWriter, partitionCount);
        appendPartitions(batchWriter, dataFile, dataName, keySplitWriter, indexData, dataSize);
        pref.neededSize -= (keySplitOrder[order].first == 0) ? needed.rowOffsets.back() : 0;
        delete[] needed.data;
        needed.data = NULL;
        if (isNextRepRead) {
            std::swap(repAnswers, nextRepAnswers);
            std::swap(repAnswerSize, nextRepAnswerSize);
            repReads.swap(nextRepReads);
            delete[] nextRepAnswers;
            nextRepAnswers = NULL;
            nextRepAnswerSize = 0;
        }
        // read the next key split while writing this one
        if (order + 1 < keySplitOrder.size()) {
            startKeySplitReads(order + 1, isNextRepRead == false);
        }
        pendingDataFile = dataFile;
        pendingDataName = &dataName;
        pendingIndexName = par.db2 + ".index." + SSTR(pref.keySplitFiles[keySplitIndex]);
        pendingDataSize = dataSize;
        pendingIndexData.swap(indexData);
        MMseqsNode::printProgress(progress, order, order + 1, keySplitOrder.size());
    }
    closePendingPrefFile();
    delete[] repAnswers;
    if (countWriter.flushOffsets.empty()) {
        countWriter.flushOffsets.assign(partitionCount + 1, countWriter.size);
    }
    punches.wait();
    publishFinishedGroups(par, punches, pref.progress);
    return countWriter.publish(pref.partitionFirstKeys.size());
}

static std::string writeSizeClasses(const Parameters &par, const KmerExchange &exchange, KeySplitPref &pref, const std::vector<PrefExchangeFile> &countFiles, size_t &classSize) {
    Debug::Progress progress(100);
    std::string prefix = par.db2 + ".piece_counts";
    size_t firstPartition = pref.partitionNodeOffsets[MMseqsNode::rank];
    size_t ownPartitions = pref.partitionNodeOffsets[MMseqsNode::rank + 1] - firstPartition;
    size_t targetSplitCount = pref.targetSplitKeyOffsets.size() - 1;
    PrefExchangeParts parts = openPrefExchangeFiles(pref, prefix, 0, countFiles);
    size_t nodeCount = countFiles.size();
    std::vector<std::vector<size_t> > tables(nodeCount);
    readTables(parts, countFiles, firstPartition, ownPartitions + 1, tables);
    std::vector<size_t> partitionSizes(ownPartitions, 0);
    for (size_t node = 0; node < nodeCount; node++) {
        size_t flushCount = countFiles[node].flushCount;
        for (size_t flush = 0; flush < flushCount; flush++) {
            for (size_t partition = 0; partition < ownPartitions; partition++) {
                partitionSizes[partition] += tables[node][(partition + 1) * flushCount + flush] - tables[node][partition * flushCount + flush];
            }
        }
    }
    // per rep count (>= 2 bytes) a PieceCount and <= 12 class bytes, per target split 1 byte and 3 words
    for (size_t partition = 0; partition < ownPartitions; partition++) {
        partitionSizes[partition] += partitionSizes[partition] / 2 * (sizeof(PieceCount) + 12) + targetSplitCount * (1 + 3 * sizeof(size_t));
    }
    std::vector<size_t> passOffsets;
    std::vector<size_t> passSizes;
    planPasses(partitionSizes, pref.passLimit, passOffsets, passSizes);
    std::vector<std::string> dataNames(targetSplitCount);
    std::vector<FILE *> dataFiles(targetSplitCount);
    std::vector<size_t> dataSizes(targetSplitCount, 0);
    std::vector<std::vector<uint64_t> > indexData(targetSplitCount);
    for (size_t split = 0; split < targetSplitCount; split++) {
        dataNames[split] = par.db2 + "." + SSTR(pref.keySplits.size() + MMseqsNode::rank * targetSplitCount + split);
        dataFiles[split] = FileUtil::openAndDelete(dataNames[split].c_str(), "w");
    }
    BatchWriter batchWriter(exchange.writeBehindSize, par.threads, fileno(dataFiles[0]));
    BatchReader batchReader(par.threads, parts.fds[0][0]);
    BackgroundRemoval punches;
    std::vector<char> counts;
    std::string rows;
    for (size_t pass = 0; pass + 1 < passOffsets.size(); pass++) {
        size_t passFirst = passOffsets[pass];
        size_t passEnd = passOffsets[pass + 1];
        size_t passPartitions = passEnd - passFirst;

        // read the piece counts of a pass
        // start in counts per node and flush
        std::vector<size_t> flushStarts;
        size_t readSize = 0;
        for (size_t node = 0; node < nodeCount; node++) {
            size_t flushCount = countFiles[node].flushCount;
            for (size_t flush = 0; flush < flushCount; flush++) {
                flushStarts.push_back(readSize);
                readSize += tables[node][passEnd * flushCount + flush] - tables[node][passFirst * flushCount + flush];
            }
        }
        counts.resize(readSize);
        punches.punch(readTableColumns(parts, countFiles, tables, passFirst, passEnd, counts.data(), false, batchReader));

        // decode, sort and encode the partitions
        std::vector<size_t> partitionPrefSizes(passPartitions, 0);
        std::vector<size_t> partitionCosts(passPartitions, 0);
        // class pieces in split order
        std::vector<std::vector<char> > partitionClasses(passPartitions);
        std::vector<size_t> splitOffsets(passPartitions * (targetSplitCount + 1));
#pragma omp parallel num_threads(par.threads)
        {
            std::vector<PieceCount> pieceCounts;
#pragma omp for schedule(dynamic, 1)
            for (size_t partition = 0; partition < passPartitions; partition++) {
                pieceCounts.clear();
                for (size_t node = 0, k = 0; node < nodeCount; node++) {
                    size_t flushCount = countFiles[node].flushCount;
                    const size_t *table = tables[node].data();
                    for (size_t flush = 0; flush < flushCount; flush++, k++) {
                        const char *data = counts.data() + flushStarts[k] + table[(passFirst + partition) * flushCount + flush] - table[passFirst * flushCount + flush];
                        const char *columnEnd = data + table[(passFirst + partition + 1) * flushCount + flush] - table[(passFirst + partition) * flushCount + flush];
                        while (data < columnEnd) {
                            partitionPrefSizes[partition] += Util::getPrefKey(data);
                            partitionCosts[partition] += Util::getPrefKey(data);
                            size_t targetSplit = Util::getPrefKey(data);
                            size_t repCount = Util::getPrefKey(data);
                            PieceCount pieceCount = { static_cast<DBKeyType>(pref.partitionFirstKeys[firstPartition + passFirst + partition]), targetSplit, 0 };
                            for (size_t i = 0; i < repCount; i++) {
                                pieceCount.rep += Util::getPrefKey(data);
                                pieceCount.targetCount = Util::getPrefKey(data);
                                pieceCounts.push_back(pieceCount);
                            }
                        }
                    }
                }
                SORT_SERIAL(pieceCounts.begin(), pieceCounts.end(), PieceCount::compareByRep);
                // size class from all splits, once per split with targets
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
                std::vector<char> &data = partitionClasses[partition];
                size_t *offsets = splitOffsets.data() + partition * (targetSplitCount + 1);
                for (size_t split = 0, pos = 0; split < targetSplitCount; split++) {
                    offsets[split] = data.size();
                    DBKeyType prevRep = pref.partitionFirstKeys[firstPartition + passFirst + partition];
                    for (; pos < pieceCounts.size() && pieceCounts[pos].targetSplit == split; pos++) {
                        putPrefKey(data, pieceCounts[pos].rep - prevRep);
                        putPrefKey(data, getBitWidth(pieceCounts[pos].targetCount) - 1);
                        data.push_back('\0');
                        prevRep = pieceCounts[pos].rep;
                    }
                    data.push_back('\0');
                }
                offsets[targetSplitCount] = data.size();
                partitionPrefSizes[partition] += data.size();
            }
        }

        // write the class files and rows
        for (size_t split = 0; split < targetSplitCount; split++) {
            for (size_t partition = 0; partition < passPartitions; partition++) {
                const size_t *offsets = splitOffsets.data() + partition * (targetSplitCount + 1);
                indexData[split].push_back(pref.partitionFirstKeys[firstPartition + passFirst + partition]);
                indexData[split].push_back(dataSizes[split]);
                dataSizes[split] += offsets[split + 1] - offsets[split];
            }
        }
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
        for (size_t partition = 0; partition < passPartitions; partition++) {
            const size_t *offsets = splitOffsets.data() + partition * (targetSplitCount + 1);
            for (size_t split = 0; split < targetSplitCount; split++) {
                batchWriter.write(fileno(dataFiles[split]), dataNames[split].c_str(), partitionClasses[partition].data() + offsets[split], offsets[split + 1] - offsets[split],
                                  indexData[split][2 * (passFirst + partition) + 1]);
            }
        }
        for (size_t partition = 0; partition < passPartitions; partition++) {
            rows += SSTR(pref.partitionFirstKeys[firstPartition + passFirst + partition]) + "\t" + SSTR(firstPartition + passFirst + partition) + "\t" + SSTR(partitionPrefSizes[partition]) + "\t" + SSTR(partitionCosts[partition])
                    + "\t" + SSTR(MMseqsNode::rank) + "\n";
        }
        MMseqsNode::printProgress(progress, passFirst, passEnd, ownPartitions);
    }
    batchWriter.waitWrites();
    classSize = 0;
    for (size_t split = 0; split < targetSplitCount; split++) {
        classSize += closePrefFile(par.db2 + ".index." + SSTR(pref.keySplits.size() + MMseqsNode::rank * targetSplitCount + split), dataFiles[split], dataNames[split], dataSizes[split],
                                   indexData[split]);
    }
    punches.wait();
    closePrefExchangeFiles(parts);
    addFinishedItems(par, pref.progress, PIECE_COUNT_FILES, 0, 1);
    return rows;
}

// step 0 serves the seed members and only runs if there are some
static void printStep(const KmerExchange &exchange, size_t step, const std::string &description) {
    if (MMseqsNode::isMaster()) {
        Debug(Debug::INFO) << "kmermatcher step " << step + exchange.hasSeedMembers << " of " << 6 + exchange.hasSeedMembers << ": " << description << "\n";
    }
}

template <typename T, bool includeAdjacency, typename L>
static void kmermatcherKeySplitPref(Parameters &par, DBReader<DBKeyType> &seqDbr, BaseMatrix *subMat, const KmerExchange &exchange, MMseqsNode::WorkQueue &sendQueue,
                                  size_t totalKmerCount, size_t minPassLimit, size_t receiverKmerSize, size_t localNodes, std::string values) {
    KeySplitPref pref;
    size_t nodeCount = MMseqsNode::count;
    size_t hashRangePasses = std::max((size_t) 1, (totalKmerCount * receiverKmerSize / nodeCount + minPassLimit - 1) / minPassLimit);
    pref.hashRangeCount = nodeCount * std::max(RANGES_PER_NODE, std::max(hashRangePasses, MIN_ANSWER_PASSES) * TASKS_PER_THREAD * par.threads);
    pref.hashRangeNodeOffsets.resize(nodeCount + 1);
    for (size_t node = 0; node <= nodeCount; node++) {
        pref.hashRangeNodeOffsets[node] = node * pref.hashRangeCount / nodeCount;
    }
    pref.codeShifts[KMER_GROUPS] = exchange.kmerHashShift;
    pref.codeShifts[WHOLE_SEQUENCE_GROUPS] = 0;
    // one in 2^8 fingerprints collides
    unsigned int fingerprintBits = getBitWidth(totalKmerCount) + 8;
    for (size_t kind = 0; kind < GROUP_KINDS; kind++) {
        unsigned int codeBits = 64 - pref.codeShifts[kind];
        pref.fingerprintShifts[kind] = pref.codeShifts[kind] + ((codeBits > fingerprintBits) ? codeBits - fingerprintBits : 0);
    }

    pref.targetSplitBlockOffsets = getTargetSplitBlockOffsets(seqDbr);
    for (size_t split = 0; split < pref.targetSplitBlockOffsets.size(); split++) {
        pref.targetSplitKeyOffsets.push_back(seqDbr.getBlockKeyOffsets()[pref.targetSplitBlockOffsets[split]]);
    }
    // partitions of the pair merge
    size_t partitionCount = exchange.sendSplitCount * exchange.partitionsPerRange;
    pref.partitionFirstKeys.resize(partitionCount + 1);
    for (size_t partition = 0; partition < partitionCount; partition++) {
        pref.partitionFirstKeys[partition] = getPartitionFirstKey(exchange, partition / exchange.partitionsPerRange, partition % exchange.partitionsPerRange);
    }
    pref.partitionFirstKeys[partitionCount] = seqDbr.getSize();
    pref.partitionNodeOffsets.resize(nodeCount + 1);
    for (size_t node = 0; node <= nodeCount; node++) {
        pref.partitionNodeOffsets[node] = node * partitionCount / nodeCount;
    }
    pref.rangeCoveredKeys.resize(seqDbr.getIndexSize());
#pragma omp parallel for schedule(static) num_threads(par.threads)
    for (size_t range = 0; range < seqDbr.getIndexSize(); range++) {
        pref.rangeCoveredKeys[range] = getCoveredKeys(par, seqDbr, seqDbr.getIndex()[range].id);
    }

    size_t keySplitMemory = exchange.keySplitMemoryLimit - std::min(exchange.keySplitMemoryLimit, exchange.readBufferSize);
    // 2/3 for the k-mers, 1/3 for the needed groups
    pref.keySplitKmerCapacity = keySplitMemory / 3 * 2 / sizeof(PackedKmerPosition);
    pref.passLimit = std::max((size_t) 1, getPassLimit(exchange.keySplitMemoryLimit));
    pref.writeBuffers = new(std::nothrow) char[par.threads * KMER_FILE_BUFFER_SIZE];
    Util::checkAllocation(pref.writeBuffers, "Can not allocate writeBuffers memory in kmermatcherKeySplitPref");
    pref.keptSize = 0;
    pref.isKeeping = true;
    pref.neededSize = 0;
    pref.neededLimit = keySplitMemory / 3;
    // same on every node, totalKmerCount and minPassLimit are global
    pref.hashRangeGroups = hashRangePasses;
    pref.keySplitMemory = keySplitMemory;
    NeededSpill spill = { par.localTmp + "/" + FileUtil::baseName(par.db2) + ".needed." + SSTR(MMseqsNode::rank), par.localTmp, par.localTmp.empty() ? 0 : localNodes, NULL, 0 };
    pref.spill = spill;
    SequenceCopy &copy = pref.sequenceCopy;
    copy.isCopying = spill.nodes > 0;
    copy.keySplits = 0;
    copy.size = 0;
    sequenceCopyError = 0;
    // keep own rows only if --local-tmp is another device
    LocalExchange &localExchange = pref.localExchange;
    localExchange.directory = par.localTmp;
    localExchange.nodes = 0;
    localExchange.keptSize = 0;
    localExchange.heldSize = 0;
    std::string localProblem = "it is not local flash";
    if (spill.nodes > 0) {
        localProblem = MMseqsNode::getLocalTmpProblem(par);
        struct stat localTmpStat, sharedStat;
        if (localProblem.empty() && stat(par.localTmp.c_str(), &localTmpStat) == 0 && stat(FileUtil::dirName(par.db2).c_str(), &sharedStat) == 0
            && localTmpStat.st_dev == sharedStat.st_dev) {
            localProblem = "it is on the device of the shared temporary folder";
        }
        localExchange.nodes = localProblem.empty() ? spill.nodes : 0;
    }
    MarkedFingerprints &marked = pref.marked;
    marked.keptHashRanges = 0;
    marked.isKeeping = true;
    marked.rowOffsets.assign(1, 0);
    marked.memorySize = 0;
    marked.fd = -1;
    marked.name = par.localTmp + "/" + FileUtil::baseName(par.db2) + ".marked_fingerprints." + SSTR(MMseqsNode::rank);
    printStep(exchange, 1, "send the k-mer hashes of each sequence batch to the nodes that own their hash range");
    if (MMseqsNode::isMaster()) {
        std::string layout;
        MMseqsNode::addPhaseValue(layout, "Layout", "about " + SSTR(totalKmerCount) + " k-mers, " + SSTR(exchange.sendSplitCount) + " parts of the database, " + SSTR(pref.hashRangeCount) + " hash ranges");
        MMseqsNode::printPhaseValues(layout);
    }

    // remove a file after the barrier past its last read
    BackgroundRemoval removal;
    // an exchange file group goes once all readers have closed it
    BackgroundRemoval releasedRemovals[PREF_FILE_KINDS];
    std::string prefPrefixes[PREF_FILE_KINDS] = { par.db2 + ".fingerprints", par.db2 + ".marks", par.db2 + ".candidates", par.db2 + ".reps", par.db2 + ".piece_counts" };
    size_t prefGroupCounts[PREF_FILE_KINDS] = { pref.hashRangeGroups, pref.hashRangeGroups, pref.hashRangeGroups, pref.hashRangeGroups, 1 };
    std::thread fileGroupRelease;
    if (MMseqsNode::isMaster()) {
        std::vector<std::string> prefixes(prefPrefixes, prefPrefixes + PREF_FILE_KINDS);
        std::vector<std::vector<size_t> > groupItems(PREF_FILE_KINDS);
        for (size_t kind = 0; kind < PREF_FILE_KINDS; kind++) {
            groupItems[kind].assign(prefGroupCounts[kind], nodeCount);
        }
        fileGroupRelease = std::thread(releaseFileGroups, std::cref(par), prefixes, groupItems);
    }
    if (exchange.hasSeedMembers) {
        seedMemberFiles.resize(nodeCount);
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
        for (size_t node = 0; node < nodeCount; node++) {
            SeedMemberFile &memberFile = seedMemberFiles[node];
            memberFile.name = par.db2 + ".served_seed_members." + SSTR(node);
            memberFile.fd = MMseqsNode::openExchangeTableFile(memberFile.name, memberFile.tableOffset, memberFile.flushCount);
        }
    }
    Timer phaseTimer;
    size_t sentSize = 0;
    // the data files and their copies, the group files and their kept rows
    FileUtil::checkRlimitNoFile(3 * seqDbr.getDataFileCnt() + seedMemberFiles.size() + FileUtil::UNCOUNTED_OPEN_FILES + pref.hashRangeGroups * ((localExchange.nodes > 0) ? 2 : 1));
    std::string keySplitLines = sendFingerprints<T, includeAdjacency, L>(par, seqDbr, subMat, exchange, pref, sendQueue, sentSize);
    delete[] pref.extractedKeySplit.kmers;
    pref.extractedKeySplit.kmers = NULL;
    MMseqsNode::addPhaseTime(values, "sending k-mer hashes", phaseTimer);
    MMseqsNode::addPhaseValue(values, "K-mer hash data sent", MMseqsNode::formatBytes(sentSize));
    MMseqsNode::addPhaseValue(values, "Sequence batches", SSTR(pref.sendSplitKeySplitOffsets.back()) + (pref.isKeeping ? " (k-mers kept in memory)" : " (k-mers computed again later)"));
    keySplitLines = MMseqsNode::barrier(par.db2 + ".fingerprints.done", keySplitLines);
    removal.run([&sendQueue]() { sendQueue.removeTakenItemFiles(); });
    if (MMseqsNode::isMaster()) {
        removal.removeFile(par.db2 + ".memory");
    }
    // per node: its group files, then a line per key split
    std::vector<PrefExchangeFile> fingerprintFiles;
    pref.nodeKeySplitOffsets.assign(1, 0);
    for (char *line = (char *) keySplitLines.c_str(); *line != '\0';) {
        PrefExchangeFile file = readPrefExchangeLine(line);
        for (size_t group = 0; group < pref.hashRangeGroups; group++, line = Util::skipLine(line)) {
            fingerprintFiles.push_back(readPrefExchangeLine(line));
        }
        for (size_t i = 0; i < file.flushCount; i++, line = Util::skipLine(line)) {
            const char *columns[2];
            Util::getWordsOfLine(line, columns, 2);
            KeySplit keySplit = { Util::fast_atoi<size_t>(columns[0]), Util::fast_atoi<size_t>(columns[1]), 0 };
            keySplit.idBits = getBitWidth(keySplit.endKey - keySplit.firstKey);
            pref.keySplits.push_back(keySplit);
        }
        pref.nodeKeySplitOffsets.push_back(pref.keySplits.size());
    }
    pref.keySplitRepKeys.assign(pref.keySplits.size(), std::make_pair(0, 0));
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
    for (size_t keySplit = 0; keySplit < pref.keySplits.size(); keySplit++) {
        size_t firstRepRange = SIZE_MAX;
        size_t endRepRange = 0;
        for (size_t range = 0; range < seqDbr.getIndexSize(); range++) {
            if (pref.rangeCoveredKeys[range].first < pref.keySplits[keySplit].endKey && pref.rangeCoveredKeys[range].second > pref.keySplits[keySplit].firstKey) {
                firstRepRange = std::min(firstRepRange, range);
                endRepRange = range + 1;
            }
        }
        if (firstRepRange < endRepRange) {
            pref.keySplitRepKeys[keySplit].first = seqDbr.getIndex()[firstRepRange].id;
            pref.keySplitRepKeys[keySplit].second = (endRepRange < seqDbr.getIndexSize()) ? seqDbr.getIndex()[endRepRange].id : seqDbr.getSize();
        }
    }
    // fixed: data files, copies, seed member files; then marks and reps or class files
    const size_t fixedOpenFiles = 2 * seqDbr.getDataFileCnt() + countSequenceCopyFiles(pref, seqDbr) + seedMemberFiles.size() + FileUtil::UNCOUNTED_OPEN_FILES;
    FileUtil::checkRlimitNoFile(fixedOpenFiles + std::max(2 * nodeCount + pref.sendSplitKeySplitOffsets.back() + 2, nodeCount + pref.targetSplitKeyOffsets.size()));
    std::vector<std::vector<PrefExchangeFile> > fingerprintGroups = getPrefExchangeGroups(fingerprintFiles, pref.hashRangeGroups);
    removeReleasedPrefExchangeFiles(releasedRemovals[FINGERPRINT_FILES], prefPrefixes[FINGERPRINT_FILES], fingerprintGroups);
    std::vector<std::pair<size_t, size_t> > keySplitOrder;
    for (size_t keySplit = 0; keySplit < pref.keySplits.size(); keySplit++) {
        keySplitOrder.push_back(std::make_pair(pref.keySplits[keySplit].firstKey, keySplit));
    }
    SORT_SERIAL(keySplitOrder.begin(), keySplitOrder.end());
    pref.keySplitFiles.resize(pref.keySplits.size());
    for (size_t i = 0; i < keySplitOrder.size(); i++) {
        pref.keySplitFiles[keySplitOrder[i].second] = i;
    }
    MMseqsNode::addPhaseTime(values, "waiting for the other nodes to send k-mer hashes", phaseTimer);
    MMseqsNode::printPhaseValues(values);
    values.clear();

    printStep(exchange, 2, "find the k-mer hashes that several sequence batches share");
    size_t receivedSize = 0;
    size_t passCount = 0;
    FileUtil::checkRlimitNoFile(fixedOpenFiles + getPartCount(fingerprintGroups) + countLocalExchangeFiles(pref) + pref.hashRangeGroups);
    std::string answer = answerHashRanges(par, exchange, pref, FINGERPRINT_FILES, prefPrefixes[FINGERPRINT_FILES], fingerprintGroups, prefPrefixes[MARK_FILES], false, receivedSize,
                                          passCount);
    MMseqsNode::addPhaseTime(values, "marking shared k-mer hashes", phaseTimer);
    MMseqsNode::addPhaseValue(values, "K-mer hash data received", MMseqsNode::formatBytes(receivedSize) + " in " + SSTR(passCount) + " passes");
    size_t ownHashRanges = pref.hashRangeNodeOffsets[MMseqsNode::rank + 1] - pref.hashRangeNodeOffsets[MMseqsNode::rank];
    MMseqsNode::addPhaseValue(values, "Marked k-mer hashes kept", (marked.keptHashRanges == 0) ? std::string("none, the shared k-mers come with their whole hash")
                                  : MMseqsNode::formatBytes(marked.rowOffsets.back()) + ((marked.fd >= 0) ? " in --local-tmp" : " in memory") + " for " + SSTR(marked.keptHashRanges) + " of "
                                    + SSTR(ownHashRanges) + " hash ranges");
    // append the kept marked hash range count to each line
    std::string markLines;
    for (char *line = (char *) answer.c_str(); *line != '\0'; line = Util::skipLine(line)) {
        markLines += std::string(line, strchr(line, '\n')) + "\t" + SSTR(marked.keptHashRanges) + "\n";
    }
    markLines = MMseqsNode::barrier(par.db2 + ".marks.done", markLines);
    size_t markLine = 0;
    for (char *line = (char *) markLines.c_str(); *line != '\0'; line = Util::skipLine(line), markLine++) {
        const char *word = strchr(line, '\n');
        while (word[-1] != '\t') {
            word--;
        }
        if (markLine % pref.hashRangeGroups == 0) {
            pref.keptMarkedHashRanges.push_back(Util::fast_atoi<size_t>(word));
        }
    }
    std::vector<std::vector<PrefExchangeFile> > markGroups = getPrefExchangeGroups(readPrefExchangeFiles(markLines), pref.hashRangeGroups);
    removeReleasedPrefExchangeFiles(releasedRemovals[MARK_FILES], prefPrefixes[MARK_FILES], markGroups);
    if (MMseqsNode::isMaster()) {
        removal.removeFile(par.db2 + ".fingerprints.done");
    }
    MMseqsNode::addPhaseTime(values, "waiting for the other nodes to mark shared k-mer hashes", phaseTimer);
    MMseqsNode::printPhaseValues(values);
    values.clear();

    printStep(exchange, 3, "send the shared k-mers to the nodes that own their hash range");
    FileUtil::checkRlimitNoFile(fixedOpenFiles + std::min((size_t) 2, pref.hashRangeGroups) * getPartCount(markGroups) + ((localExchange.nodes == 1) ? pref.hashRangeGroups : 0)
                                + ((marked.fd >= 0) ? 1 : 0));
    size_t keptFingerprintSize = localExchange.keptSize;
    std::string sent = sendCandidates<T, includeAdjacency, L>(par, seqDbr, subMat, exchange, pref, markGroups, sentSize);
    delete[] pref.extractedKeySplit.kmers;
    pref.extractedKeySplit.kmers = NULL;
    size_t spilledKeySplits = 0;
    size_t extractedKeySplits = 0;
    for (size_t keySplit = 0; keySplit < pref.neededGroups.size(); keySplit++) {
        spilledKeySplits += pref.neededGroups[keySplit].spillOffset != SIZE_MAX;
        extractedKeySplits += pref.neededGroups[keySplit].data == NULL && pref.neededGroups[keySplit].spillOffset == SIZE_MAX;
    }
    MMseqsNode::addPhaseTime(values, "sending shared k-mers", phaseTimer);
    MMseqsNode::addPhaseValue(values, "K-mer groups kept for the pairs", MMseqsNode::formatBytes(pref.neededSize) + " in memory, " + MMseqsNode::formatBytes(pref.spill.size) + " in --local-tmp ("
                                                                         + SSTR(spilledKeySplits) + " sequence batches), k-mers of " + SSTR(extractedKeySplits) + " sequence batches computed again later");
    MMseqsNode::addPhaseValue(values, "Shared k-mer data sent", MMseqsNode::formatBytes(sentSize)
                                      + (localProblem.empty() ? ", " + MMseqsNode::formatBytes(localExchange.keptSize - keptFingerprintSize) + " of it kept in --local-tmp" : ""));
    if (copy.keySplits > 0) {
        MMseqsNode::addPhaseValue(values, "Local tmp", "holds the sequences of " + SSTR(copy.keySplits) + " sequence batches (" + MMseqsNode::formatBytes(copy.size) + ")");
    }
    std::vector<std::vector<PrefExchangeFile> > candidateGroups = getPrefExchangeGroups(readPrefExchangeFiles(MMseqsNode::barrier(par.db2 + ".candidates.done", sent)), pref.hashRangeGroups);
    removeReleasedPrefExchangeFiles(releasedRemovals[CANDIDATE_FILES], prefPrefixes[CANDIDATE_FILES], candidateGroups);
    if (MMseqsNode::isMaster()) {
        removal.removeFile(par.db2 + ".marks.done");
    }
    MMseqsNode::addPhaseTime(values, "waiting for the other nodes to send shared k-mers", phaseTimer);
    MMseqsNode::printPhaseValues(values);
    values.clear();

    printStep(exchange, 4, "pick representatives for the shared k-mers");
    receivedSize = 0;
    FileUtil::checkRlimitNoFile(fixedOpenFiles + getPartCount(candidateGroups) + countLocalExchangeFiles(pref) + pref.hashRangeGroups * ((localExchange.nodes > 0) ? 2 : 1));
    answer = answerHashRanges(par, exchange, pref, CANDIDATE_FILES, prefPrefixes[CANDIDATE_FILES], candidateGroups, prefPrefixes[REP_FILES], true, receivedSize, passCount);
    MMseqsNode::addPhaseTime(values, "picking representatives", phaseTimer);
    MMseqsNode::addPhaseValue(values, "Shared k-mer data received", MMseqsNode::formatBytes(receivedSize) + " in " + SSTR(passCount) + " passes");
    std::vector<std::vector<PrefExchangeFile> > repGroups = getPrefExchangeGroups(readPrefExchangeFiles(MMseqsNode::barrier(par.db2 + ".reps.done", answer)), pref.hashRangeGroups);
    removeReleasedPrefExchangeFiles(releasedRemovals[REP_FILES], prefPrefixes[REP_FILES], repGroups);
    if (MMseqsNode::isMaster()) {
        removal.removeFile(par.db2 + ".candidates.done");
    }
    MMseqsNode::addPhaseTime(values, "waiting for the other nodes to pick representatives", phaseTimer);
    MMseqsNode::printPhaseValues(values);
    values.clear();

    printStep(exchange, 5, "write the candidate pairs (representative, member)");
    size_t prefSize = 0;
    size_t pairCount = 0;
    size_t sortedPairCount = 0;
    size_t repSplitCount = 0;
    FileUtil::checkRlimitNoFile(fixedOpenFiles + getPartCount(markGroups) + getPartCount(repGroups) + countLocalExchangeFiles(pref) + pref.sendSplitKeySplitOffsets.back() + 2);
    sent = writeKeySplitPrefs<T, includeAdjacency, L>(par, seqDbr, subMat, exchange, pref, markGroups, repGroups, prefSize, pairCount, sortedPairCount, repSplitCount);
    std::vector<int> copyFds;
    copyFds.swap(copy.fds);
    removal.run([copyFds]() {
        for (size_t file = 0; file < copyFds.size(); file++) {
            if (copyFds[file] >= 0) {
                close(copyFds[file]);
            }
        }
    });
    for (size_t file = 0; file < seedMemberFiles.size(); file++) {
        if (close(seedMemberFiles[file].fd) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << seedMemberFiles[file].name << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
    std::vector<SeedMemberFile>().swap(seedMemberFiles);
    std::vector<SeedMemberFlush>().swap(seedMemberFlushes);
    seedMemberSendSplit = SIZE_MAX;
    freeKeptKeySplits(pref);
    delete[] pref.extractedKeySplit.kmers;
    delete[] pref.writeBuffers;
    MMseqsNode::addPhaseTime(values, "writing candidate pairs", phaseTimer);
    MMseqsNode::addPhaseValue(values, "Candidate pairs", SSTR(pairCount) + " written in " + SSTR(repSplitCount) + " write batches, " + SSTR(sortedPairCount) + " sorted");
    std::vector<PrefExchangeFile> countFiles = readPrefExchangeFiles(MMseqsNode::barrier(par.db2 + ".piece_counts.done", sent));
    std::vector<std::vector<PrefExchangeFile> > countGroups(1, countFiles);
    removeReleasedPrefExchangeFiles(releasedRemovals[PIECE_COUNT_FILES], prefPrefixes[PIECE_COUNT_FILES], countGroups);
    if (MMseqsNode::isMaster()) {
        removal.removeFile(par.db2 + ".reps.done");
    }
    if (exchange.hasSeedMembers) {
        removal.removeFile(par.db2 + ".served_seed_members." + SSTR(MMseqsNode::rank));
    }
    MMseqsNode::addPhaseTime(values, "waiting for the other nodes to write candidate pairs", phaseTimer);
    MMseqsNode::printPhaseValues(values);
    values.clear();

    printStep(exchange, 6, "count the candidate pairs of each representative");
    size_t classSize = 0;
    FileUtil::checkRlimitNoFile(fixedOpenFiles + getPartCount(countGroups) + pref.targetSplitKeyOffsets.size());
    std::string rows = writeSizeClasses(par, exchange, pref, countFiles, classSize);
    MMseqsNode::addPhaseTime(values, "counting the candidate pairs of each representative", phaseTimer);
    MMseqsNode::addPhaseValue(values, "Candidate pair data written", MMseqsNode::formatBytes(prefSize + classSize));
    if (par.localTmp.empty() == false) {
        size_t exchangeSize = countFiles[MMseqsNode::rank].size;
        for (size_t group = 0; group < pref.hashRangeGroups; group++) {
            exchangeSize += fingerprintGroups[group][MMseqsNode::rank].size + markGroups[group][MMseqsNode::rank].size + candidateGroups[group][MMseqsNode::rank].size
                            + repGroups[group][MMseqsNode::rank].size;
        }
        MMseqsNode::addPhaseValue(values, "Exchange data kept in --local-tmp", localProblem.empty() ? MMseqsNode::formatBytes(localExchange.keptSize) + ", "
                                  + MMseqsNode::formatBytes(exchangeSize - localExchange.keptSize) + " in the shared folder" : "none, " + localProblem);
    }
    MMseqsNode::printPhaseValues(values);
    for (size_t kind = 0; kind < PREF_FILE_KINDS; kind++) {
        releasedRemovals[kind].wait(prefPrefixes[kind] + ".released.<group>");
    }
    MMseqsNode::publishRunFile(par.db2 + ".partitions." + SSTR(MMseqsNode::rank), rows);
    if (MMseqsNode::isMaster()) {
        rows = MMseqsNode::gatherRunFiles(par.db2 + ".partitions.");
        // every node has removed its released files
        fileGroupRelease.join();
        for (size_t kind = 0; kind < PREF_FILE_KINDS; kind++) {
            for (size_t group = 0; group < prefGroupCounts[kind]; group++) {
                FileUtil::removeSharedFile(prefPrefixes[kind] + ".released." + SSTR(group));
            }
        }
        MMseqsNode::publishRunFile(par.db2 + ".merged", "");
        removal.removeFile(par.db2 + ".piece_counts.done");
        // head: key splits, target split blocks, key split first keys, partition node offsets
        std::string partitions = "key_splits\t" + SSTR(pref.keySplits.size()) + "\n";
        for (size_t split = 0; split < pref.targetSplitBlockOffsets.size(); split++) {
            partitions += SSTR(pref.targetSplitBlockOffsets[split]) + ((split + 1 < pref.targetSplitBlockOffsets.size()) ? "\t" : "\n");
        }
        for (size_t i = 0; i < keySplitOrder.size(); i++) {
            partitions += SSTR(keySplitOrder[i].first) + "\t";
        }
        partitions += SSTR(seqDbr.getSize()) + "\n";
        for (size_t node = 0; node <= nodeCount; node++) {
            partitions += SSTR(pref.partitionNodeOffsets[node]) + ((node < nodeCount) ? "\t" : "\n");
        }
        FileUtil::publishFile(par.db2 + ".partitions", partitions + rows);
        MMseqsNode::removeOnceRead(par.db2 + ".merged");
        DBWriter::writeDbtypeFile(par.db2.c_str(), DBReader<DBKeyType>::setExtendedDbtype(Parameters::DBTYPE_PREFILTER_RES, Parameters::DBTYPE_EXTENDED_LINFINITY), false);
    } else {
        MMseqsNode::waitForRunFile(par.db2 + ".merged");
        MMseqsNode::removeOnceRead(par.db2 + ".merged");
    }
    FileUtil::removeSharedFile(par.db2 + ".progress." + SSTR(MMseqsNode::rank));
}

// read the requests for the members in the own data file, write their sequences in rep order
static void serveSeedMembers(const Parameters &par, DBReader<DBKeyType> &seqDbr, const KmerExchange &exchange, std::string &values) {
    Timer timer;
    Debug::Progress progress(100);
    std::vector<int> requestFds(MMseqsNode::count);
    std::vector<size_t> flushCounts(MMseqsNode::count);
    std::vector<std::vector<size_t> > columns(MMseqsNode::count);
    std::vector<size_t> requestOffsets(MMseqsNode::count + 1, 0);
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
    for (size_t node = 0; node < MMseqsNode::count; node++) {
        std::string requestFile = MMseqsNode::getSeedMemberRequestsFile(par.db1, node);
        size_t tableOffset;
        requestFds[node] = MMseqsNode::openExchangeTableFile(requestFile, tableOffset, flushCounts[node]);
        MMseqsNode::readExchangeTableColumns(requestFds[node], requestFile, tableOffset, flushCounts[node], MMseqsNode::rank, columns[node]);
        for (size_t flush = 0; flush < flushCounts[node]; flush++) {
            requestOffsets[node + 1] += columns[node][flushCounts[node] + flush] - columns[node][flush];
        }
    }
    for (size_t node = 0; node < MMseqsNode::count; node++) {
        requestOffsets[node + 1] += requestOffsets[node];
    }
    // the reads below fill all of it
    size_t requestCount = requestOffsets.back() / sizeof(ClusterPair);
    ClusterPair *requests = new(std::nothrow) ClusterPair[requestCount];
    Util::checkAllocation(requests, "Can not allocate requests memory in serveSeedMembers");
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
    for (size_t node = 0; node < MMseqsNode::count; node++) {
        std::string requestFile = MMseqsNode::getSeedMemberRequestsFile(par.db1, node);
        size_t requestOffset = requestOffsets[node];
        for (size_t flush = 0; flush < flushCounts[node]; flush++) {
            size_t size = columns[node][flushCounts[node] + flush] - columns[node][flush];
            FileUtil::preadAll(requestFds[node], requestFile.c_str(), (char *) requests + requestOffset, size, columns[node][flush]);
            requestOffset += size;
        }
        if (close(requestFds[node]) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << requestFile << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
    SORT_PARALLEL(requests, requests + requestCount, ClusterPair::compareByMember);

    // a flush takes the memory left next to the requests, at least a read batch
    size_t keySplitMemory = exchange.keySplitMemoryLimit - std::min(exchange.keySplitMemoryLimit, exchange.readBufferSize);
    size_t flushLimit = std::max(exchange.readChunkLimit, keySplitMemory - std::min(keySplitMemory, requestCount * sizeof(ClusterPair)));
    std::string servedFile = par.db2 + ".served_seed_members." + SSTR(MMseqsNode::rank);
    FILE *served = FileUtil::openAndDelete(servedFile.c_str(), "w");
    size_t servedSize = 0;
    std::vector<size_t> flushOffsets;
    std::vector<uint64_t> pairs;
    char *data = NULL;
    std::vector<DBKeyType> keys;
    size_t pageSize = Util::getPageSize();
    for (size_t flushBegin = 0, flushEnd; flushBegin < requestCount; flushBegin = flushEnd) {
        size_t dataSize = 0;
        size_t range = SIZE_MAX;
        for (flushEnd = flushBegin; flushEnd < requestCount; flushEnd++) {
            size_t seqLen = seqDbr.getSeqLenFromRange(range, requests[flushEnd].memberKey);
            if (flushEnd > flushBegin && dataSize + seqLen + (flushEnd - flushBegin + 2) * 2 * sizeof(uint64_t) > flushLimit) {
                break;
            }
            dataSize += seqLen;
        }
        size_t memberCount = flushEnd - flushBegin;
        size_t pairsOffset = servedSize + dataSize;
        SORT_PARALLEL(requests + flushBegin, requests + flushEnd, ClusterPair::compareByRep);
        for (size_t sendSplit = 0; sendSplit <= exchange.sendSplitCount; sendSplit++) {
            ClusterPair firstPair = { static_cast<DBKeyType>(exchange.sendSplitKeyOffsets[sendSplit]), static_cast<DBKeyType>(exchange.sendSplitKeyOffsets[sendSplit]) };
            size_t first = std::lower_bound(requests + flushBegin, requests + flushEnd, firstPair, ClusterPair::compareByRep) - requests - flushBegin;
            flushOffsets.push_back(pairsOffset + first * 2 * sizeof(uint64_t));
        }
        // the pairs keep the reps, repKey keeps the data offset of the member
        pairs.resize(2 * (memberCount + 1));
        size_t memberOffset = 0;
        for (size_t i = 0; i < memberCount; i++) {
            ClusterPair &request = requests[flushBegin + i];
            pairs[2 * i] = request.repKey;
            pairs[2 * i + 1] = servedSize + memberOffset;
            request.repKey = memberOffset;
            range = SIZE_MAX;
            memberOffset += seqDbr.getSeqLenFromRange(range, request.memberKey);
        }
        pairs[2 * memberCount] = DB_KEY_INVALID;
        pairs[2 * memberCount + 1] = pairsOffset;
        SORT_PARALLEL(requests + flushBegin, requests + flushEnd, ClusterPair::compareByMember);
        // the members fill all of it
        delete[] data;
        data = new(std::nothrow) char[dataSize];
        Util::checkAllocation(data, "Can not allocate data memory in serveSeedMembers");
        for (size_t batchBegin = flushBegin, batchEnd; batchBegin < flushEnd; batchBegin = batchEnd) {
            keys.clear();
            size_t batchSize = 0;
            range = SIZE_MAX;
            for (batchEnd = batchBegin; batchEnd < flushEnd; batchEnd++) {
                size_t readSize = seqDbr.getSeqLenFromRange(range, requests[batchEnd].memberKey) + 2 + 2 * pageSize;
                if (batchEnd > batchBegin && batchSize + readSize > exchange.readChunkLimit) {
                    break;
                }
                batchSize += readSize;
                keys.push_back(requests[batchEnd].memberKey);
            }
            seqDbr.loadData(keys);
#pragma omp parallel num_threads(par.threads)
            {
                DBReader<DBKeyType>::LoadedBufferCursor cursor;
#pragma omp for schedule(static)
                for (size_t i = batchBegin; i < batchEnd; i++) {
                    const char *seqData = seqDbr.getDataFromLoadedBuffer(requests[i].memberKey, cursor);
                    memcpy(data + requests[i].repKey, seqData, seqDbr.getSeqLenFromRange(cursor.range, requests[i].memberKey));
                }
            }
            MMseqsNode::printProgress(progress, batchBegin, batchEnd, requestCount);
        }
        if (fwrite(data, 1, dataSize, served) != dataSize || fwrite(pairs.data(), sizeof(uint64_t), pairs.size(), served) != pairs.size()) {
            Debug(Debug::ERROR) << "Cannot write " << servedFile << "\n";
            EXIT(EXIT_FAILURE);
        }
        servedSize = pairsOffset + pairs.size() * sizeof(uint64_t);
    }
    delete[] data;
    delete[] requests;
    size_t flushCount = flushOffsets.size() / (exchange.sendSplitCount + 1);
    size_t fileSize = MMseqsNode::closeExchangeTableFile(served, servedFile, flushOffsets, exchange.sendSplitCount + 1, servedSize);
    MMseqsNode::addPhaseTime(values, "sending cluster members", timer);
    MMseqsNode::addPhaseValue(values, "Cluster member data sent", MMseqsNode::formatBytes(fileSize) + " in " + SSTR(flushCount) + " writes");
}

template <typename T, bool includeAdjacency, typename L>
void kmermatcherLinfinity(Parameters &par, DBReader<DBKeyType> &seqDbr, BaseMatrix *subMat, size_t memoryLimit) {
    const DBReader<DBKeyType>::Index *index = seqDbr.getIndex();
    float kmersPerSequenceScale = par.kmersPerSequenceScale.values.aminoacid();
    KmerExchange exchange;
    size_t totalKmerCount = 0;
    size_t totalCost = 0;
    size_t seedMemberCount = 0;
    std::vector<size_t> rangeCosts(seqDbr.getIndexSize());
    // an estimate for the plan: a seed member counts with the length of its rep
    for (size_t range = 0; range < seqDbr.getIndexSize(); range++) {
        size_t rangeKeys = seqDbr.getRangeEndId(range) - index[range].id;
        size_t rangeSeedMembers = seqDbr.getRangeSeedMemberCount(range);
        size_t rangeKmers = (seqDbr.getRangeRepresentativeCount(range) + rangeSeedMembers) * computeSeqKmerCount(static_cast<int>(index[range].length - 2), par.kmerSize, par.kmersPerSequence, kmersPerSequenceScale);
        rangeCosts[range] = rangeKmers * SEND_KMER_SIZE + (rangeKeys + rangeSeedMembers) * index[range].length;
        seedMemberCount += rangeSeedMembers;
        totalKmerCount += rangeKmers;
        totalCost += rangeCosts[range];
    }

    exchange.sendSplitCount = (MMseqsNode::count > 1) ? RANGES_PER_NODE * MMseqsNode::count : 1;
    std::vector<size_t> &sendSplitKeyOffsets = exchange.sendSplitKeyOffsets;
    sendSplitKeyOffsets.assign(exchange.sendSplitCount + 1, seqDbr.getSize());
    sendSplitKeyOffsets[0] = 0;
    size_t sendSplitCost = std::max((size_t) 1, (totalCost + exchange.sendSplitCount - 1) / exchange.sendSplitCount);
    size_t costBefore = 0;
    for (size_t range = 0, sendSplit = 1; range < seqDbr.getIndexSize(); range++) {
        size_t rangeKeys = seqDbr.getRangeEndId(range) - index[range].id;
        size_t rangeCost = rangeCosts[range];
        for (; sendSplit < exchange.sendSplitCount && costBefore + rangeCost >= sendSplit * sendSplitCost; sendSplit++) {
            size_t cutCost = sendSplit * sendSplitCost - costBefore;
            sendSplitKeyOffsets[sendSplit] = index[range].id + static_cast<size_t>((static_cast<PackedKmerPosition::uint128>(cutCost) * rangeKeys + rangeCost - 1) / rangeCost);
        }
        costBefore += rangeCost;
    }
    size_t maxSplitKeys = 0;
    size_t maxSeedMembers = 0;
    for (size_t sendSplit = 0; sendSplit < exchange.sendSplitCount; sendSplit++) {
        maxSplitKeys = std::max(maxSplitKeys, sendSplitKeyOffsets[sendSplit + 1] - sendSplitKeyOffsets[sendSplit]);
        size_t splitSeedMembers = 0;
        for (size_t range = (sendSplitKeyOffsets[sendSplit] < seqDbr.getSize()) ? seqDbr.findIdRange(sendSplitKeyOffsets[sendSplit]) : seqDbr.getIndexSize();
             range < seqDbr.getIndexSize() && index[range].id < sendSplitKeyOffsets[sendSplit + 1]; range++) {
            splitSeedMembers += seqDbr.getRangeSeedMemberCount(range);
        }
        maxSeedMembers = std::max(maxSeedMembers, splitSeedMembers);
    }

    exchange.fileKeyOffsets = seqDbr.getFileKeyOffsets(MMseqsNode::count);
    exchange.hasRepresentativeBits = FileUtil::statSharedFile(par.db1 + ".bits." + SSTR(MMseqsNode::rank));
    exchange.hasSeedMembers = seedMemberCount > 0;
    seqDbr.setBlockCacheSize(0);
    // the master writes the dbtype last
    if (MMseqsNode::isMaster()) {
        FileUtil::removeSharedFile(par.db2 + ".dbtype");
    }

    exchange.adjKeyBits = (includeAdjacency ? 6 : 0) * PackedKmerPosition::RESIDUE_BITS;
    exchange.adjKeyBase = includeAdjacency ? adjSubMat->alphabetSize : 1;
    exchange.adjKeyCodeBits = includeAdjacency ? getBitWidth(MathUtil::ipow<size_t>(exchange.adjKeyBase, 6) - 1) : 0;
    exchange.kmerHashShift = 64 - getKmerBits(subMat, par.kmerSize);

    exchange.globalIdBits = getBitWidth(seqDbr.getSize() - 1);
    exchange.maxRepSeqCount = 1 + (includeAdjacency ? par.adjIteration : 0);
    size_t receiverKmerSize = (2 * exchange.globalIdBits + exchange.adjKeyCodeBits + CHAR_BIT - 1) / CHAR_BIT;
    size_t pairTargetSize = (exchange.globalIdBits + CHAR_BIT - 1) / CHAR_BIT + sizeof(DBKeyType);

    MMseqsNode::WorkQueue sendQueue(par.db2 + ".send_splits", exchange.sendSplitCount);
    sendQueue.removeTakenItemFiles();
    FileUtil::removeSharedFile(par.db2 + ".progress." + SSTR(MMseqsNode::rank));
    size_t repBitsMemory = exchange.hasRepresentativeBits ? ((maxSplitKeys + 63) / 64 + 1) * sizeof(uint64_t) : 0;
    // the flushes of a send split, then the members of a key split
    size_t seedMemberMemory = maxSeedMembers * (sizeof(DBKeyType) + 2 * sizeof(size_t) + sizeof(SeedMember));
    exchange.writeBehindSize = memoryLimit / 8;
    // plus the block table and a refilled pass
    size_t fixedMemory = repBitsMemory + seedMemberMemory + 2 * par.threads * KMER_FILE_BUFFER_SIZE + exchange.writeBehindSize
                         + seqDbr.getBlockKeyOffsets().size() * sizeof(size_t) + 2 * KMER_FILE_BUFFER_SIZE;
    exchange.keySplitMemoryLimit = (fixedMemory < memoryLimit) ? static_cast<size_t>(static_cast<double>(memoryLimit - fixedMemory) * 0.95) : 0;
    exchange.readChunkLimit = std::min(BatchReader::MAX_INFLIGHT_READ_SIZE, exchange.keySplitMemoryLimit / 6);
    // the current and the prefetched input buffer, page aligned in every data file
    exchange.readBufferSize = 2 * (std::max(exchange.readChunkLimit, seqDbr.getMaxSeqLen()) + 2 * Util::getPageSize() * seqDbr.getDataFileCnt());
    // the barrier below orders the served files before every read
    std::string values;
    if (exchange.hasSeedMembers) {
        printStep(exchange, 0, "send " + SSTR(seedMemberCount) + " cluster members that add k-mers to their representative");
        serveSeedMembers(par, seqDbr, exchange, values);
    }
    size_t localNodes = 0;
    Timer timer;
    size_t minMemoryLimit = MMseqsNode::getSmallestMemoryLimit(par.db2 + ".memory", memoryLimit, par.threads, par.localTmp, &localNodes);
    if (exchange.hasSeedMembers) {
        MMseqsNode::addPhaseTime(values, "waiting for the other nodes to send cluster members", timer);
    }
    size_t minPassLimit = std::max((size_t) 1, getPassLimit(minMemoryLimit));
    // localNodes counts this node too when its --local-tmp is local flash
    if (par.localTmp.empty() == false && localNodes == 0) {
        MMseqsNode::reportUnusableLocalTmp("Cannot keep k-mer groups in --local-tmp " + par.localTmp + ": it is not local flash");
    }
    size_t rangePairPasses = std::max((size_t) 1, (totalKmerCount / exchange.sendSplitCount * exchange.maxRepSeqCount * pairTargetSize + minPassLimit - 1) / minPassLimit);
    exchange.partitionsPerRange = std::max((size_t) 4 * par.threads, 2 * rangePairPasses);
    kmermatcherKeySplitPref<T, includeAdjacency, L>(par, seqDbr, subMat, exchange, sendQueue, totalKmerCount, minPassLimit, receiverKmerSize, localNodes, values);
}

template void kmermatcherLinfinity<short, true, unsigned short>(Parameters &, DBReader<DBKeyType> &, BaseMatrix *, size_t);
template void kmermatcherLinfinity<short, false, unsigned short>(Parameters &, DBReader<DBKeyType> &, BaseMatrix *, size_t);
template void kmermatcherLinfinity<int, true, unsigned short>(Parameters &, DBReader<DBKeyType> &, BaseMatrix *, size_t);
template void kmermatcherLinfinity<int, false, unsigned short>(Parameters &, DBReader<DBKeyType> &, BaseMatrix *, size_t);
template void kmermatcherLinfinity<int, true, int>(Parameters &, DBReader<DBKeyType> &, BaseMatrix *, size_t);
template void kmermatcherLinfinity<int, false, int>(Parameters &, DBReader<DBKeyType> &, BaseMatrix *, size_t);
