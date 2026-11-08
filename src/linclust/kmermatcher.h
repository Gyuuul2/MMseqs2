#ifndef MMSEQS_KMERMATCHER_H
#define MMSEQS_KMERMATCHER_H
#include "DBReader.h"
#include "DBWriter.h"
#include "Parameters.h"
#include "BaseMatrix.h"
#include "MathUtil.h"

#include <climits>
#include <queue>
#include <type_traits>
#ifndef SIZE_T_MAX
#define SIZE_T_MAX ((size_t) -1)
#endif

using AssignGroupMask = uint32_t;

struct AssignGroupFeature {
    static constexpr AssignGroupMask Default        = 0;
    static constexpr AssignGroupMask AdjacentSeq = 1u << 0;
    static constexpr AssignGroupMask CountTable  = 1u << 1;
};

enum class ComputationPhase {
    Main,
    SetupCountTable
};

inline bool hasFeature(AssignGroupMask mask, AssignGroupMask feature) {
    return (mask & feature) != 0;
}

struct SequencePosition{
    size_t kmer;
    unsigned int pos;
    unsigned short score;
    static bool compareByScore(const SequencePosition &first, const SequencePosition &second){
        if(first.score < second.score)
            return true;
        if(second.score < first.score)
            return false;
        if(first.kmer < second.kmer)
            return true;
        if(second.kmer < first.kmer)
            return false;
        if(first.pos < second.pos)
            return true;
        if(second.pos < first.pos)
            return false;
        return false;
    }
    static bool compareByScoreReverse(const SequencePosition &first, const SequencePosition &second){
        if(first.score < second.score)
            return true;
        if(second.score < first.score)
            return false;

        size_t firstKmer  = BIT_SET(first.kmer, 63);
        size_t secondKmer = BIT_SET(second.kmer, 63);
        if(firstKmer < secondKmer)
            return true;
        if(secondKmer < firstKmer)
            return false;
        if(first.pos < second.pos)
            return true;
        if(second.pos < first.pos)
            return false;
        return false;
    }
};

template <bool Include>
struct AdjacentData {
    unsigned char data[6];
    void set(int i, unsigned char v) { data[i] = v; }
    unsigned char get(int i) const { return data[i]; }
};

template <>
struct __attribute__((__packed__)) AdjacentData<false> {
    void set(int, unsigned char) { }
    unsigned char get(int) const { return '\0'; }
};

// [ isWholeSequence | k-mer | adjKey | maxSeqLen - seqLen | id | pos ] in integer order; a whole sequence holds its hash from adjKeyShift up
struct PackedKmerPosition {
    static const unsigned int WHOLE_SEQUENCE_SHIFT = 127;
    static const unsigned int RESIDUE_BITS = 5;
    static unsigned int kmerShift;
    static unsigned int adjKeyShift;
    static unsigned int seqLenShift;
    static unsigned int idShift;
    static size_t maxSeqLen;
    __extension__ typedef unsigned __int128 uint128;
    uint128 value;

    void set(size_t kmer, unsigned int adjKey, size_t seqLen, DBKeyType id, size_t pos, bool isWholeSequence) {
        value = (static_cast<uint128>(isWholeSequence) << WHOLE_SEQUENCE_SHIFT)
                | (static_cast<uint128>(id) << idShift);
        // linfinity records have no length and no position field
        if (adjKeyShift > seqLenShift) {
            value |= static_cast<uint128>(maxSeqLen - seqLen) << seqLenShift;
        }
        if (isWholeSequence) {
            value |= static_cast<uint128>(kmer) << adjKeyShift;
        } else {
            value |= (static_cast<uint128>(kmer) << kmerShift) | (static_cast<uint128>(adjKey) << adjKeyShift);
            if (idShift > 0) {
                value |= pos;
            }
        }
    }

    bool isWholeSequence() const { return (value >> WHOLE_SEQUENCE_SHIFT) != 0; }
    uint128 getKmer() const { return value >> (isWholeSequence() ? adjKeyShift : kmerShift); }
    unsigned int getAdjKey() const { return getBits(adjKeyShift, kmerShift); }
    static unsigned char getAdjacentSeq(unsigned int adjKey, int index) { return (adjKey >> (RESIDUE_BITS * index)) & ((1u << RESIDUE_BITS) - 1); }
    unsigned char getAdjacentSeq(int index) const { return getAdjacentSeq(getAdjKey(), index); }
    size_t getSeqLen() const { return maxSeqLen - getBits(seqLenShift, adjKeyShift); }
    DBKeyType getId() const { return getBits(idShift, seqLenShift); }
    size_t getPos() const { return getBits(0, idShift); }

    bool operator<(const PackedKmerPosition &other) const {
        const uint64_t upper = static_cast<uint64_t>(value >> 64);
        const uint64_t otherUpper = static_cast<uint64_t>(other.value >> 64);
        return (upper != otherUpper) ? upper < otherUpper : static_cast<uint64_t>(value) < static_cast<uint64_t>(other.value);
    }

private:
    size_t getBits(unsigned int shift, unsigned int nextShift) const {
        return (value >> shift) & ((static_cast<uint128>(1) << (nextShift - shift)) - 1);
    }
};

// [ 0 | k-mer | maxSeqLen - seqLen | id | pos | adjKey | used as rep ] in integer order, the order of KmerPosition::compareRepSequenceAndIdAndPos;
// a key above the k-mer field, a whole sequence hash, is [ 1 | key | maxSeqLen - seqLen | id | used as rep ] with pos 0 and all adjacent residues X
struct PackedDiagonalKmerPosition {
    static const unsigned int WHOLE_SEQUENCE_SHIFT = 127;
    static const unsigned int ADJ_KEY_SHIFT = 1;
    static unsigned int posShift;
    static unsigned int idShift;
    static unsigned int seqLenShift;
    static unsigned int kmerShift;
    static size_t maxSeqLen;
    static unsigned char xIndex;
    __extension__ typedef unsigned __int128 uint128;
    uint128 value;

    void set(size_t kmer, unsigned int adjKey, size_t seqLen, DBKeyType id, size_t pos, bool) {
        const bool isWholeSequence = (static_cast<uint128>(kmer) >> (WHOLE_SEQUENCE_SHIFT - kmerShift)) != 0;
        const unsigned int shift = isWholeSequence ? idShift - 1 : 0;
        value = (static_cast<uint128>(isWholeSequence) << WHOLE_SEQUENCE_SHIFT)
                | (static_cast<uint128>(kmer) << (kmerShift - shift))
                | (static_cast<uint128>(maxSeqLen - seqLen) << (seqLenShift - shift))
                | (static_cast<uint128>(id) << (idShift - shift));
        if (isWholeSequence == false) {
            value |= (static_cast<uint128>(pos) << posShift) | (static_cast<uint128>(adjKey) << ADJ_KEY_SHIFT);
        }
    }

    bool isWholeSequence() const { return (value >> WHOLE_SEQUENCE_SHIFT) != 0; }
    size_t getKmer() const { return getBits(kmerShift - getWholeSequenceShift(), WHOLE_SEQUENCE_SHIFT); }
    size_t getSeqLen() const { return maxSeqLen - getBits(seqLenShift - getWholeSequenceShift(), kmerShift - getWholeSequenceShift()); }
    DBKeyType getId() const { return getBits(idShift - getWholeSequenceShift(), seqLenShift - getWholeSequenceShift()); }
    size_t getPos() const { return isWholeSequence() ? 0 : getBits(posShift, idShift); }
    unsigned char getAdjacentSeq(int index) const {
        if (index == 0 && (value & 1)) {
            return UCHAR_MAX;
        }
        return isWholeSequence() ? xIndex : getBits(ADJ_KEY_SHIFT + PackedKmerPosition::RESIDUE_BITS * index, ADJ_KEY_SHIFT + PackedKmerPosition::RESIDUE_BITS * (index + 1));
    }
    // assignGroup only marks a used rep, setAdjacentSeq(0, UCHAR_MAX)
    void setAdjacentSeq(int, unsigned char) { value |= 1; }

    bool operator<(const PackedDiagonalKmerPosition &other) const {
        const uint64_t upper = static_cast<uint64_t>(value >> 64);
        const uint64_t otherUpper = static_cast<uint64_t>(other.value >> 64);
        return (upper != otherUpper) ? upper < otherUpper : static_cast<uint64_t>(value) < static_cast<uint64_t>(other.value);
    }

private:
    unsigned int getWholeSequenceShift() const { return isWholeSequence() ? idShift - 1 : 0; }
    size_t getBits(unsigned int shift, unsigned int nextShift) const {
        return (value >> shift) & ((static_cast<uint128>(1) << (nextShift - shift)) - 1);
    }
};

// [ rep id | id | diagonal + 2^(bits of T - 1) ] in integer order, the order of KmerPosition::compareRepSequenceAndIdAndDiag;
// the rep id field is wider than the id and read sign-extended, so the 0xFF end marker that assignGroup writes after the last pair reads as SIZE_T_MAX
template <typename T, typename Upper>
struct __attribute__((__packed__)) PackedSeqPair {
    typedef typename std::make_unsigned<T>::type UnsignedT;
    static const unsigned int ID_SHIFT = 8 * sizeof(T);
    static const unsigned int BITS = 64 + 8 * sizeof(Upper);
    static const UnsignedT DIAGONAL_OFFSET = static_cast<UnsignedT>(1) << (ID_SHIFT - 1);
    static unsigned int repShift;
    __extension__ typedef unsigned __int128 uint128;
    __extension__ typedef __int128 int128;
    uint64_t lower;
    Upper upper;

    void setRepSequenceAndIdAndDiag(size_t repSeqId, DBKeyType id, T diagonal) {
        const uint128 value = (static_cast<uint128>(repSeqId) << repShift) | (static_cast<uint128>(id) << ID_SHIFT)
                              | static_cast<UnsignedT>(static_cast<UnsignedT>(diagonal) ^ DIAGONAL_OFFSET);
        lower = static_cast<uint64_t>(value);
        upper = static_cast<Upper>(value >> 64);
    }

    size_t getKmer() const { return static_cast<int128>(getValue() << (128 - BITS)) >> (128 - BITS + repShift); }
    DBKeyType getId() const { return (getValue() >> ID_SHIFT) & ((static_cast<uint128>(1) << (repShift - ID_SHIFT)) - 1); }
    size_t getPos() const { return static_cast<T>(static_cast<UnsignedT>(lower) ^ DIAGONAL_OFFSET); }

    bool operator<(const PackedSeqPair &other) const {
        return (upper != other.upper) ? upper < other.upper : lower < other.lower;
    }

private:
    uint128 getValue() const { return (static_cast<uint128>(upper) << 64) | lower; }
};

template <typename T, typename Upper>
unsigned int PackedSeqPair<T, Upper>::repShift = 0;

template <typename T, bool includeAdjacency = false, typename L = T>
struct __attribute__((__packed__)) KmerPosition {
    size_t kmer;
    DBKeyType id;
    T pos;
    L seqLen;
    AdjacentData<includeAdjacency> adj;

    void setAdjacentSeq(int index, unsigned char val) {
        adj.set(index, val);
    }
    unsigned char getAdjacentSeq(int index) {
        return adj.get(index);
    }

    bool isWholeSequence() const { return false; }
    size_t getKmer() const { return kmer; }
    size_t getAdjKey() const {
        size_t adjKey = 0;
        for (int i = 5; i >= 0; i--) {
            adjKey = (adjKey << PackedKmerPosition::RESIDUE_BITS) | adj.get(i);
        }
        return adjKey;
    }
    size_t getSeqLen() const { return seqLen; }
    DBKeyType getId() const { return id; }
    size_t getPos() const { return pos; }
    void setRepSequenceAndIdAndDiag(size_t repSeqId, DBKeyType id, T diagonal) {
        kmer = repSeqId;
        this->id = id;
        pos = diagonal;
    }

    static bool compareKmer(const KmerPosition &first, const KmerPosition &second) {
        return first.kmer < second.kmer;
    }

    static bool compareKmerAndAdjKeyAndSeqLenAndIdAndPos(const KmerPosition &first, const KmerPosition &second) {
        if(first.kmer < second.kmer) return true;
        if(second.kmer < first.kmer) return false;
        if(first.getAdjKey() < second.getAdjKey()) return true;
        if(second.getAdjKey() < first.getAdjKey()) return false;
        if(first.seqLen > second.seqLen) return true;
        if(second.seqLen > first.seqLen) return false;
        if(first.id < second.id) return true;
        if(second.id < first.id) return false;
        if(first.pos < second.pos) return true;
        if(second.pos < first.pos) return false;
        return false;
    }

    static bool compareRepSequenceAndIdAndPos(
            const KmerPosition &first, const KmerPosition &second) {
        if(first.kmer < second.kmer) return true;
        if(second.kmer < first.kmer) return false;
        if(first.id != DB_KEY_INVALID && second.id != DB_KEY_INVALID) {
            L len1 = first.seqLen;
            L len2 = second.seqLen;
            if(len1 > len2) return true;
            if(len2 > len1) return false;
        }
        if(first.id < second.id) return true;
        if(second.id < first.id) return false;
        if(first.pos < second.pos) return true;
        if(second.pos < first.pos) return false;
        return false;
    }

    static bool compareRepSequenceAndIdAndPosReverse(
            const KmerPosition &first, const KmerPosition &second) {
        size_t firstKmer  = BIT_SET(first.kmer, 63);
        size_t secondKmer = BIT_SET(second.kmer, 63);
        if(firstKmer < secondKmer) return true;
        if(secondKmer < firstKmer) return false;
        if(first.id != DB_KEY_INVALID && second.id != DB_KEY_INVALID) {
            L len1 = first.seqLen;
            L len2 = second.seqLen;
            if(len1 > len2) return true;
            if(len2 > len1) return false;
        }
        if(first.id < second.id) return true;
        if(second.id < first.id) return false;
        if(first.pos < second.pos) return true;
        if(second.pos < first.pos) return false;
        return false;
    }

    static bool compareRepSequenceAndIdAndDiagReverse(
            const KmerPosition &first, const KmerPosition &second) {
        size_t firstKmer  = BIT_SET(first.kmer, 63);
        size_t secondKmer = BIT_SET(second.kmer, 63);
        if(firstKmer < secondKmer) return true;
        if(secondKmer < firstKmer) return false;
        if(first.id < second.id) return true;
        if(second.id < first.id) return false;
        if(first.pos < second.pos) return true;
        if(second.pos < first.pos) return false;
        return false;
    }

    static bool compareRepSequenceAndIdAndDiag(
            const KmerPosition &first, const KmerPosition &second) {
        if(first.kmer < second.kmer) return true;
        if(second.kmer < first.kmer) return false;
        if(first.id < second.id) return true;
        if(second.id < first.id) return false;
        if(first.pos < second.pos) return true;
        if(second.pos < first.pos) return false;
        return false;
    }
};

struct __attribute__((__packed__)) KmerEntry {
    DBKeyType seqId;
    short diagonal;
    unsigned char score;
    void setReverse(bool ){
        ;
    }
    unsigned char getRev(){
        return 0;
    }
};

struct __attribute__((__packed__)) KmerEntryRev {
    DBKeyType seqId;
    short diagonal;
    unsigned char score;
    unsigned char rev;
    void setReverse(bool rev){
        this->rev = rev;
    }
    unsigned char getRev(){
        return this->rev;
    }
};

struct FileKmerPosition {
    DBKeyType repSeq;
    DBKeyType id;
    short pos;
    unsigned char score;
    unsigned int file;
    char reverse;
    FileKmerPosition(){}
    FileKmerPosition(DBKeyType repSeq, DBKeyType id,short pos, unsigned char score, unsigned int file):
            repSeq(repSeq), id(id), pos(pos), score(score), file(file), reverse(0) {}
    FileKmerPosition(DBKeyType repSeq, DBKeyType id,short pos, unsigned char score, char reverse, unsigned int file):
            repSeq(repSeq), id(id), pos(pos), score(score), file(file), reverse(reverse) {}
};

class CompareResultBySeqId {
public:
    bool operator() (FileKmerPosition & first, FileKmerPosition & second) const {
        //return (first.eval < second.eval);
        if(first.repSeq > second.repSeq )
            return true;
        if(second.repSeq > first.repSeq )
            return false;
        if(first.id > second.id )
            return true;
        if(second.id > first.id )
            return false;
        if(first.pos > second.pos )
            return true;
        if(second.pos > first.pos )
            return false;
        return false;
    }
};

template <int TYPE, typename T, bool includeAdjacency = false>
void mergeKmerFilesAndOutput(DBWriter & dbw, std::vector<std::string> tmpFiles, std::vector<char> &repSequence, int numThreads = 1, int maxIter = 1);

typedef std::priority_queue<FileKmerPosition, std::vector<FileKmerPosition>, CompareResultBySeqId> KmerPositionQueue;

template <int TYPE, typename T>
size_t queueNextEntry(KmerPositionQueue &queue, int file, size_t offsetPos, T *entries, size_t entrySize);

void setKmerLengthAndAlphabet(Parameters &parameters, size_t aaDbSize, int seqType);

template <int TYPE, typename T, typename posType, typename KmerPositionType>
void writeKmersToDisk(std::string tmpFile, KmerPositionType *kmers, size_t totalKmers, int numThreads = 1, std::vector<size_t> *threadQueryOffsets = NULL, int iteration = 0);

template <int TYPE, typename T, typename KmerPositionType>
void writeKmerMatcherResult(DBWriter & dbw, KmerPositionType *hashSeqPair, size_t totalKmers,
                            std::vector<char> &repSequence, size_t threads);


template <typename T, bool includeAdjacency = false, typename L = T>
KmerPosition<T, includeAdjacency, L> * doComputation(size_t totalKmers, size_t split, size_t splits, std::string splitFile,
                                DBReader<DBKeyType> & seqDbr, Parameters & par, BaseMatrix  * subMat,
                                size_t KMER_SIZE, size_t chooseTopKmer, float chooseTopKmerScale = 0.0);

template <typename T, bool includeAdjacency = false, typename L = T>
KmerPosition<T, includeAdjacency, L> *initKmerPositionMemory(size_t size);

template <int TYPE, typename T, bool includeAdjacency = false, typename L = T, typename KmerPositionType>
std::pair<size_t, size_t> fillKmerPositionArray(KmerPositionType * kmerArray, size_t kmerArraySize, DBReader<DBKeyType> &seqDbr,
                                                 Parameters & par, BaseMatrix * subMat, bool hashWholeSequence,
                                                 size_t hashStartRange, size_t hashEndRange, size_t * hashDistribution,
                                                 KmerPositionType *countTableKmerArray = NULL, size_t countTableKmers = 0, size_t countTableEndRange = 0);


void maskSequence(int maskMode, int maskLowerCase,
                  Sequence &seq, int maskLetter, ProbabilityMatrix * probMatrix);

template <typename T, bool includeAdjacency = false, typename L = T>
size_t computeMemoryNeededLinearfilter(size_t totalKmer);

template <typename T, bool includeAdjacency = false, typename L = T>
std::vector<std::pair<size_t, size_t>> setupKmerSplits(Parameters &par, BaseMatrix * subMat, DBReader<DBKeyType> &seqDbr, size_t totalKmers, size_t splits,
                                                       size_t kmerMemory = sizeof(KmerPosition<T, includeAdjacency, L>));
size_t computeKmerCount(DBReader<DBKeyType> &reader, size_t KMER_SIZE, size_t chooseTopKmer,
                        float chooseTopKmerScale = 0.0);

void setLinearFilterDefault(Parameters *p);

size_t computeMemoryNeededLinearfilter(size_t totalKmer);

extern BaseMatrix *adjSubMat;
// linfinity: the first key of the send split that this node sends or merges, only the keys set in representativeBits unless NULL
extern size_t sendSplitFirstKey;
extern uint64_t *representativeBits;

static inline bool isRepresentative(size_t id) {
    return representativeBits == NULL || ((representativeBits[(id - sendSplitFirstKey) / 64] >> ((id - sendSplitFirstKey) % 64)) & 1);
}

static inline unsigned int getBitWidth(uint64_t value) {
    return (value == 0) ? 0 : 64 - __builtin_clzll(value);
}

static inline unsigned int getKmerBits(const BaseMatrix *subMat, int kmerSize) {
    return (kmerSize * MathUtil::log2(subMat->alphabetSize - 1) >= 64) ? 64 : getBitWidth(MathUtil::ipow<size_t>(subMat->alphabetSize - 1, kmerSize) - 1);
}

static inline size_t computeSeqKmerCount(int seqLen, size_t kmerSize, size_t chooseTopKmer, float chooseTopKmerScale) {
    int kmerAdjustedSeqLen = std::max(1, seqLen - static_cast<int>(kmerSize) + 2);
    return std::min(kmerAdjustedSeqLen, static_cast<int>(chooseTopKmer + (chooseTopKmerScale * seqLen)));
}

// the next rep of a k-mer group: the adjKey candidate scoring lowest against the previous one, at a tie the first by isBefore
template <typename Candidate, typename ForEachRepCandidate, typename IsBefore>
static inline void selectGroupRepSeqs(std::vector<Candidate> &reps, size_t maxRepSeqCount, ForEachRepCandidate forEachRepCandidate, IsBefore isBefore) {
    short *subMatPos[6] = {NULL, NULL, NULL, NULL, NULL, NULL};
    unsigned int repAdjKey = 0;
    while (reps.size() < maxRepSeqCount) {
        if (reps.empty() == false) {
            for (int j = 0; j < 6; j++) {
                subMatPos[j] = adjSubMat->subMatrix[PackedKmerPosition::getAdjacentSeq(repAdjKey, j)];
            }
        }
        bool hasBest = false;
        Candidate bestCandidate = Candidate();
        unsigned int bestAdjKey = 0;
        int bestScore = INT_MAX;
        bool isFirst = true;
        unsigned int prevAdjKey = 0;
        forEachRepCandidate([&](unsigned int adjKey, const Candidate &candidate) {
            const bool isOtherAdjKey = isFirst || adjKey != prevAdjKey;
            isFirst = false;
            prevAdjKey = adjKey;
            if (isOtherAdjKey == false) {
                return;
            }
            for (size_t rep = 0; rep < reps.size(); rep++) {
                if (isBefore(candidate, reps[rep]) == false && isBefore(reps[rep], candidate) == false) {
                    return;
                }
            }
            int score = 0;
            for (int j = 0; reps.empty() == false && j < 6; j++) {
                score += subMatPos[j][PackedKmerPosition::getAdjacentSeq(adjKey, j)];
            }
            if (hasBest == false || score < bestScore || (score == bestScore && isBefore(candidate, bestCandidate))) {
                hasBest = true;
                bestScore = score;
                bestCandidate = candidate;
                bestAdjKey = adjKey;
            }
        });
        if (hasBest == false) {
            break;
        }
        reps.push_back(bestCandidate);
        repAdjKey = bestAdjKey;
    }
}

template <int TYPE, typename T, bool includeAdjacency, typename L, typename KmerPositionType>
std::pair<size_t, size_t> fillKmerPositionArray(KmerPositionType * kmerArray, size_t kmerArraySize, DBReader<DBKeyType> &seqDbr,
                                                Parameters & par, BaseMatrix * subMat, bool hashWholeSequence,
                                                size_t hashStartRange, size_t hashEndRange, size_t * hashDistribution,
                                                const std::vector<size_t> &readChunkOffsets,
                                                KmerPositionType *countTableKmerArray = NULL, size_t countTableKmers = 0, size_t countTableEndRange = 0);

template <typename T, bool includeAdjacency, typename L>
void kmermatcherLinfinity(Parameters &par, DBReader<DBKeyType> &seqDbr, BaseMatrix *subMat, size_t memoryLimit);

#undef SIZE_T_MAX


#endif
