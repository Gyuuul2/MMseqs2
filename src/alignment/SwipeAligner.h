#ifndef SWIPE_ALIGNER_H
#define SWIPE_ALIGNER_H

#include "StripedSmithWaterman.h"
#include "simd.h"

#include <string>
#include <vector>

class BaseMatrix;
class Sequence;

// Smith-Waterman of one query against several targets at once, one target in every SIMD lane (SWIPE)
class SwipeAligner {
public:
    SwipeAligner(BaseMatrix *subMat, size_t maxSeqLen, bool aaBiasCorrection, float aaBiasCorrectionScale, int gapOpen, int gapExtend);
    ~SwipeAligner();

    void initQuery(const Sequence *query);

    // score1, qEndPos1, dbEndPos1 and word of every target, starting with 8 bit lanes if word is 0 and 16 bit if 1; word 2 if 16 bit saturates
    void alignForward(size_t count, const unsigned char *const *targets, const int *targetLengths, s_align *aligns);
    // qStartPos1 and dbStartPos1 from the forward results of alignForward
    void alignBackward(size_t count, const unsigned char *const *targets, s_align *aligns);

    // banded around diagonals (target - query position): score1 and the end, starting with the lanes of word as alignForward,
    // or with backtraces the whole alignment; bandWidths are grown while the best cell is at the band edge or below minScores
    void alignBanded(size_t count, const unsigned char *const *targets, const int *targetLengths, const int *diagonals, int *bandWidths,
                     s_align *aligns, std::string *backtraces, const int *columnEnds, const int *minScores);

    static const int MIN_BAND_WIDTH = 16;

private:
    struct BandJob {
        size_t id;
        int offset;
        int width;
        int rowBegin;
        int rowEnd;
        int columnBegin;
        int columnEnd;
    };

    template <typename T, bool Reverse>
    void alignFull(const std::vector<size_t> &jobs, const unsigned char *const *targets, const int *targetLengths, s_align *aligns);

    template <typename T, bool Trace>
    void alignBand(const BandJob *jobs, size_t jobCount, const unsigned char *const *targets, const int *targetLengths,
                   int *bestScores, int *bestRows, int *bestColumns, int *startRows, int *startColumns, std::string *backtraces, uint32_t *identities);

    void runBanded(std::vector<BandJob> &jobs, bool trace, const unsigned char *const *targets, const int *targetLengths,
                   int *bandWidths, s_align *aligns, std::string *backtraces, const int *minScores);

    BaseMatrix *subMat;
    int alphabetSize;
    bool aaBiasCorrection;
    float aaBiasCorrectionScale;
    int gapOpen;
    int gapExtend;

    int queryLength;
    std::vector<unsigned char> query;
    std::vector<unsigned char> queryRev;
    std::vector<int8_t> bias;
    std::vector<int8_t> biasRev;
    float *tmpBias;

    // per query letter, the scores of the target letters 0-15 and 16-31 for 8 bit lanes and biased by 64 for 16 bit lanes
    simd_int *scoreLo8;
    simd_int *scoreHi8;
    simd_int *scoreLo16;
    simd_int *scoreHi16;

    size_t capacity;
    simd_int *rowBias8;
    simd_int *rowBias16;
    simd_int *rowBiasRev8;
    simd_int *rowBiasRev16;
    // 64 - composition bias of the query rows, subtracted with saturation from the 16 bit scores biased by 64
    simd_int *rowSub16;
    simd_int *rowSubRev16;
    simd_int *rowIndex;
    simd_int *hColumn;
    simd_int *eColumn;
    simd_int *rowMask;
    // the lane scores of each query letter for the columns of a pass
    simd_int *columnScores;
    simd_int *ring;
    size_t ringCapacity;
    // the H and F rows of the band
    simd_int *band;
    size_t bandCapacity;
    simd_int *trace;
    size_t traceCapacity;

    std::vector<size_t> jobBuffer;
    std::vector<BandJob> bandJobs;
    std::vector<int> bestScores;
    std::vector<int> bestRows;
    std::vector<int> bestColumns;
    std::vector<int> startRows;
    std::vector<int> startColumns;
    std::vector<uint32_t> identities;

    void resize(size_t length);
};

// banded X-drop extensions from seeds with backtrace in 16 bit SWIPE lanes, as BlockAligner::bandedalign; every lane takes the next
// extension of any query once it is free
class SwipeExtender {
public:
    SwipeExtender(BaseMatrix *subMat, size_t maxSeqLen, bool aaBiasCorrection, float aaBiasCorrectionScale, int gapOpen, int gapExtend);
    ~SwipeExtender();

    // a query of the next extensions, its index among the queries since clearQueries
    size_t addQuery(const Sequence *query);
    void clearQueries();

    // the extensions from the seeds against their queries to the right, or to the left if backward; word 2 for the ones that do not fit the lanes
    void extendBanded(size_t count, const size_t *queryIds, const unsigned char *const *targets, const int *targetLengths, const int *seedQueries,
                      const int *seedTargets, bool backward, int xDrop, int bandSide, s_align *aligns, std::string *backtraces);

    int cellScore(size_t queryId, int queryPos, unsigned char targetLetter) const {
        const size_t position = queryOffsets[queryId] + queryPos;
        return subMat->subMatrix[queries[position]][targetLetter] + biases[position];
    }

    // the lane rows that extended a target and all lane rows
    size_t extendLaneRows;
    size_t extendSlotRows;

private:
    struct ExtendJob {
        size_t id;
        size_t query;
        int seedRow;
        int seedColumn;
        int left;
        int right;
    };

    void queue(std::vector<ExtendJob> &jobs, const ExtendJob &job, s_align *aligns) const;

    template <bool Solo>
    void extendBand(const ExtendJob *jobs, size_t jobCount, bool backward, int xDrop, const unsigned char *const *targets, const int *targetLengths,
                    s_align *aligns, std::string *backtraces);

    BaseMatrix *subMat;
    bool aaBiasCorrection;
    float aaBiasCorrectionScale;
    int gapOpen;
    int gapExtend;

    // the queries, each forward then reversed, and their composition bias at the same positions
    std::vector<unsigned char> queries;
    std::vector<int8_t> biases;
    std::vector<size_t> queryOffsets;
    std::vector<int> queryLengths;
    std::vector<int8_t> bias;
    float *tmpBias;

    // per query letter, the scores of the target letters 0-31
    int8_t *scoreRows;

    // the H and F rows of the band
    simd_int *band;
    size_t bandCapacity;
    // the trace codes of the last rows of the lanes
    simd_int *trace;
    size_t traceCapacity;
    // the target of every lane forward or reversed, after as many zeros as the left side of its band and before a band width of zeros
    std::vector<std::vector<unsigned char>> laneTargets;
    std::vector<unsigned char> zeroLetters;
    // the trace of the lane of a job that goes on alone
    std::vector<uint16_t> soloTrace;

    std::vector<ExtendJob> extendJobs;
    std::vector<ExtendJob> extendPending;
    std::vector<ExtendJob> soloJobs;
};

#endif
