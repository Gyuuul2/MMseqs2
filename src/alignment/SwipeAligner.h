#ifndef SWIPE_ALIGNER_H
#define SWIPE_ALIGNER_H

#include "StripedSmithWaterman.h"
#include "simd.h"

#include <string>
#include <vector>

class BaseMatrix;
class Sequence;

// SWIPE: one query against one target per SIMD lane
class SwipeAligner {
public:
    SwipeAligner(BaseMatrix *subMat, size_t maxSeqLen, bool aaBiasCorrection, float aaBiasCorrectionScale, int gapOpen, int gapExtend);
    ~SwipeAligner();

    void initQuery(const Sequence *query);

    // word 0 starts in 8 bit lanes, 1 in 16 bit, 2 if 16 bit saturates
    void alignForward(size_t count, const unsigned char *const *targets, const int *targetLengths, s_align *aligns);

    void alignBanded(size_t count, const unsigned char *const *targets, const int *targetLengths, const int *diagonals,
                     s_align *aligns, std::string *backtraces, const int *columnEnds, const int *minScores);

    // frees the buffers that grow with the batches; the next batch allocates them again
    void release();

private:
    static const int MIN_BAND_WIDTH = 16;

    struct BandJob {
        size_t id;
        int offset;
        int width;
        int rowBegin;
        int rowEnd;
        int columnBegin;
        int columnEnd;
    };

    template <typename T>
    void alignFull(const std::vector<size_t> &jobs, const unsigned char *const *targets, const int *targetLengths, s_align *aligns);

    template <typename T>
    void fillRing(const BandJob *jobs, size_t jobCount, const unsigned char *const *targets, int ringSize, int m);

    template <typename T>
    void alignBand(const BandJob *jobs, size_t jobCount, const unsigned char *const *targets, const int *targetLengths,
                   s_align *aligns, std::string *backtraces);

    BaseMatrix *subMat;
    int alphabetSize;
    bool aaBiasCorrection;
    float aaBiasCorrectionScale;
    int gapOpen;
    int gapExtend;

    int queryLength;
    std::vector<unsigned char> query;
    std::vector<int8_t> bias;
    float *queryBiasBuffer;

    int matrixBias;
    int byteBias;

    simd_int *scoreLo8;
    simd_int *scoreHi8;
    simd_int *scoreLo16;
    simd_int *scoreHi16;

    size_t capacity;
    simd_int *rowBias8;
    simd_int *rowBias16;
    simd_int *rowSub16;
    simd_int *rowIndex;
    simd_int *hColumn;
    simd_int *eColumn;
    simd_int *columnScores;
    simd_int *ring;
    size_t ringCapacity;
    simd_int *band;
    size_t bandCapacity;
    simd_int *trace;
    size_t traceCapacity;

    std::vector<size_t> jobBuffer;
    std::vector<BandJob> bandJobs;

    void resize(size_t length);
};

// banded X-drop extensions from seeds in SWIPE lanes, as BlockAligner::bandedalign
class SwipeExtender {
public:
    static const int LANES = VECSIZE_INT * 4;

    SwipeExtender(BaseMatrix *subMat, size_t maxSeqLen, bool aaBiasCorrection, float aaBiasCorrectionScale, int gapOpen, int gapExtend);
    ~SwipeExtender();

    size_t addQuery(const Sequence *query);
    void clearQueries();
    // frees the buffers that grow with the batches; the next batch allocates them again
    void release();

    void extendBanded(size_t count, const size_t *queryIds, const unsigned char *const *targets, const int *targetLengths, const int *seedQueries,
                      const int *seedTargets, bool backward, int xDrop, int bandSide, s_align *aligns, std::string *backtraces);

    int getCellScore(size_t queryId, int queryPos, unsigned char targetLetter) const {
        size_t pos = queryOffsets[queryId] + queryPos;
        return subMat->subMatrix[queries[pos]][targetLetter] + biases[pos];
    }

    size_t extensionActiveLaneRowCount;
    size_t extensionTotalLaneRowCount;
    size_t byteExtensionActiveLaneRowCount;
    size_t byteExtensionTotalLaneRowCount;
    size_t longExtensionCount;
    size_t rerunExtensionCount;

private:
    // a dead cell is this far below the best score of its lane
    static const int FLOOR = 222;

    struct ExtendJob {
        size_t id;
        size_t query;
        int seedRow;
        int seedColumn;
        int left;
        int right;
        bool isWordAlignment;
    };

    template <typename T>
    void extendBand(const ExtendJob *jobs, size_t jobCount, bool backward, int xDrop, const unsigned char *const *targets, const int *targetLengths,
                    s_align *aligns, std::string *backtraces);

    BaseMatrix *subMat;
    bool aaBiasCorrection;
    float aaBiasCorrectionScale;
    int gapOpen;
    int gapExtend;
    int matrixBias;

    // forward then reversed
    std::vector<unsigned char> queries;
    std::vector<int8_t> biases;
    std::vector<size_t> queryOffsets;
    std::vector<int> queryLengths;
    std::vector<int8_t> bias;
    float *queryBiasBuffer;

    int8_t *scoreRows;

    simd_int *band;
    size_t bandCapacity;
    simd_int *trace;
    size_t traceCapacity;
    std::vector<std::vector<unsigned char>> laneTargets;
    std::vector<unsigned char> zeroLetters;
    std::vector<std::vector<uint16_t>> laneTraces;

    std::vector<ExtendJob> extendJobs;
    std::vector<ExtendJob> extendPending;
};

#endif
