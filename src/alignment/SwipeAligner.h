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
    simd_int *rowIndex;
    simd_int *hColumn;
    simd_int *eColumn;
    simd_int *rowMask;
    simd_int *columnScores;
    simd_int *ring;
    size_t ringCapacity;
    simd_int *bandH;
    simd_int *bandF;
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

#endif
