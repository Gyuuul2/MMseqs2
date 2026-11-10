#include "SwipeAligner.h"
#include "SubstitutionMatrix.h"
#include "Sequence.h"
#include "Debug.h"

#include <algorithm>

template <typename T>
struct SwipeLanes;

// unsigned and biased by byteBias, as SSW sw_sse2_byte
template <>
struct SwipeLanes<uint8_t> {
    static const int COUNT = VECSIZE_INT * 4;
    static const int MAX = UINT8_MAX;
    static inline simd_int set(int x) { return simdi8_set(static_cast<char>(x)); }
    static inline simd_int addScore(simd_int a, simd_int b, simd_int bias) { return simdui8_subs(simdui8_adds(a, b), bias); }
    static inline simd_int subs(simd_int a, simd_int b) { return simdui8_subs(a, b); }
    static inline simd_int max(simd_int a, simd_int b) { return simdui8_max(a, b); }
    static inline simd_int gt(simd_int a, simd_int b) { return simdi8_gt(simdi_xor(a, simdi8_set(-128)), simdi_xor(b, simdi8_set(-128))); }
    static inline simd_int eq(simd_int a, simd_int b) { return simdi8_eq(a, b); }
    static inline simd_int addBias(simd_int a, simd_int b) { return simdui8_adds(a, b); }
};

template <>
struct SwipeLanes<int16_t> {
    static const int COUNT = VECSIZE_INT * 2;
    static const int MAX = INT16_MAX;
    static inline simd_int set(int x) { return simdi16_set(static_cast<short>(x)); }
    static inline simd_int addScore(simd_int a, simd_int b, simd_int) { return simdi16_max(simdi16_adds(a, b), simdi_setzero()); }
    static inline simd_int subs(simd_int a, simd_int b) { return simdi16_subs(a, b); }
    static inline simd_int max(simd_int a, simd_int b) { return simdi16_max(a, b); }
    static inline simd_int gt(simd_int a, simd_int b) { return simdi16_gt(a, b); }
    static inline simd_int eq(simd_int a, simd_int b) { return simdi16_eq(a, b); }
    static inline simd_int addBias(simd_int a, simd_int b) { return simdi16_add(a, b); }
};

static const int SWIPE_MAX_LANES = VECSIZE_INT * 4;

// shuffle indices into the 0-15 and 16-31 score tables
template <typename T>
static inline void getLaneIndices(const unsigned char *letters, simd_int &indexLo, simd_int &indexHi) {
    unsigned char lo[SWIPE_MAX_LANES];
    unsigned char hi[SWIPE_MAX_LANES];
    for (int l = 0; l < SwipeLanes<T>::COUNT; l++) {
        unsigned char c = letters[l];
        lo[l * sizeof(T)] = (c < 16) ? c : 0x80;
        hi[l * sizeof(T)] = (c >= 16) ? static_cast<unsigned char>(c - 16) : 0x80;
        if (sizeof(T) == 2) {
            lo[l * 2 + 1] = 0x80;
            hi[l * 2 + 1] = 0x80;
        }
    }
    indexLo = simdi_loadu((const simd_int *) lo);
    indexHi = simdi_loadu((const simd_int *) hi);
}

static inline simd_int computeLaneScores(simd_int scoreLo, simd_int scoreHi, simd_int indexLo, simd_int indexHi) {
    return simdi_or(simdi8_shuffle(scoreLo, indexLo), simdi8_shuffle(scoreHi, indexHi));
}

template <typename T>
static inline T getLaneValue(const simd_int *v, int lane) {
    return ((const T *) v)[lane];
}

template <typename T>
static inline void setLaneValue(simd_int *v, int lane, T value) {
    ((T *) v)[lane] = value;
}

static void computeQueryBias(BaseMatrix *subMat, const Sequence *q, bool aaBiasCorrection, float aaBiasCorrectionScale, float *queryBiasBuffer, std::vector<int8_t> &bias) {
    bias.assign(q->L, 0);
    if (aaBiasCorrection) {
        SubstitutionMatrix::calcLocalAaBiasCorrection(subMat, q->numSequence, q->L, queryBiasBuffer, aaBiasCorrectionScale);
        for (int i = 0; i < q->L; i++) {
            bias[i] = static_cast<int8_t>((queryBiasBuffer[i] < 0.0) ? queryBiasBuffer[i] - 0.5 : queryBiasBuffer[i] + 0.5);
        }
    }
}

static void reserveVectors(simd_int *&buffer, size_t &capacity, size_t size) {
    if (size > capacity) {
        free(buffer);
        capacity = size;
        buffer = malloc_simd_int(capacity * sizeof(simd_int));
    }
}

SwipeAligner::SwipeAligner(BaseMatrix *subMat, size_t maxSeqLen, bool aaBiasCorrection, float aaBiasCorrectionScale, int gapOpen, int gapExtend)
    : subMat(subMat), alphabetSize(subMat->alphabetSize), aaBiasCorrection(aaBiasCorrection), aaBiasCorrectionScale(aaBiasCorrectionScale),
      gapOpen(gapOpen), gapExtend(gapExtend), queryLength(0), matrixBias(0), byteBias(0), capacity(0), rowBias8(NULL), rowBias16(NULL), rowSub16(NULL), rowIndex(NULL),
      hColumn(NULL), eColumn(NULL), ring(NULL), ringCapacity(0), band(NULL), bandCapacity(0), trace(NULL), traceCapacity(0) {
    if (alphabetSize > 32) {
        Debug(Debug::ERROR) << "SWIPE supports alphabets of at most 32 letters\n";
        EXIT(EXIT_FAILURE);
    }
    queryBiasBuffer = new float[maxSeqLen + 1];
    columnScores = malloc_simd_int(2 * alphabetSize * sizeof(simd_int));
    scoreLo8 = malloc_simd_int(alphabetSize * sizeof(simd_int));
    scoreHi8 = malloc_simd_int(alphabetSize * sizeof(simd_int));
    scoreLo16 = malloc_simd_int(alphabetSize * sizeof(simd_int));
    scoreHi16 = malloc_simd_int(alphabetSize * sizeof(simd_int));
    for (int a = 0; a < alphabetSize; a++) {
        for (int b = 0; b < alphabetSize; b++) {
            matrixBias = std::max(matrixBias, -static_cast<int>(subMat->subMatrix[a][b]));
        }
    }
    for (int a = 0; a < alphabetSize; a++) {
        unsigned char lo8[sizeof(simd_int)], hi8[sizeof(simd_int)], lo16[sizeof(simd_int)], hi16[sizeof(simd_int)];
        for (size_t p = 0; p < sizeof(simd_int); p++) {
            int bLo = p % 16;
            int bHi = 16 + p % 16;
            int sLo = (bLo < alphabetSize) ? subMat->subMatrix[bLo][a] : 0;
            int sHi = (bHi < alphabetSize) ? subMat->subMatrix[bHi][a] : 0;
            lo8[p] = static_cast<unsigned char>(sLo + matrixBias);
            hi8[p] = static_cast<unsigned char>(sHi + matrixBias);
            lo16[p] = static_cast<unsigned char>(sLo + 64);
            hi16[p] = static_cast<unsigned char>(sHi + 64);
        }
        scoreLo8[a] = simdi_loadu((const simd_int *) lo8);
        scoreHi8[a] = simdi_loadu((const simd_int *) hi8);
        scoreLo16[a] = simdi_loadu((const simd_int *) lo16);
        scoreHi16[a] = simdi_loadu((const simd_int *) hi16);
    }
}

SwipeAligner::~SwipeAligner() {
    delete[] queryBiasBuffer;
    free(scoreLo8);
    free(scoreHi8);
    free(scoreLo16);
    free(scoreHi16);
    free(rowBias8);
    free(rowBias16);
    free(rowSub16);
    free(rowIndex);
    free(hColumn);
    free(eColumn);
    free(columnScores);
    free(ring);
    free(band);
    free(trace);
}

void SwipeAligner::release() {
    free(ring);
    free(band);
    free(trace);
    ring = NULL;
    band = NULL;
    trace = NULL;
    ringCapacity = 0;
    bandCapacity = 0;
    traceCapacity = 0;
    std::vector<size_t>().swap(jobBuffer);
    std::vector<BandJob>().swap(bandJobs);
}

void SwipeAligner::resize(size_t length) {
    if (length <= capacity) {
        return;
    }
    free(rowBias8);
    free(rowBias16);
    free(rowSub16);
    free(rowIndex);
    free(hColumn);
    free(eColumn);
    capacity = length;
    rowBias8 = malloc_simd_int(capacity * sizeof(simd_int));
    rowBias16 = malloc_simd_int(capacity * sizeof(simd_int));
    rowSub16 = malloc_simd_int(capacity * sizeof(simd_int));
    rowIndex = malloc_simd_int(capacity * sizeof(simd_int));
    hColumn = malloc_simd_int((capacity + 1) * sizeof(simd_int));
    eColumn = malloc_simd_int((capacity + 1) * sizeof(simd_int));
}

void SwipeAligner::initQuery(const Sequence *q) {
    queryLength = q->L;
    resize(static_cast<size_t>(queryLength) + 1);
    query.assign(q->numSequence, q->numSequence + queryLength);
    computeQueryBias(subMat, q, aaBiasCorrection, aaBiasCorrectionScale, queryBiasBuffer, bias);
    int minBias = 0;
    for (int i = 0; i < queryLength; i++) {
        minBias = std::min(minBias, static_cast<int>(bias[i]));
    }
    byteBias = matrixBias - minBias;
    for (int i = 0; i < queryLength; i++) {
        rowBias8[i] = simdi8_set(static_cast<char>(bias[i] - minBias));
        rowBias16[i] = simdi16_set(static_cast<short>(bias[i] - 64));
        rowSub16[i] = simdi16_set(static_cast<short>(64 - bias[i]));
    }
}

template <typename T>
static inline void setLaneResult(s_align &a, int best, int column, int row, int saturated) {
    a.score1 = best;
    a.dbEndPos1 = (best == 0) ? -1 : column;
    a.qEndPos1 = (best == 0) ? -1 : row;
    a.word = (best >= saturated) ? ((sizeof(T) == 1) ? 1 : 2) : ((sizeof(T) == 1) ? 0 : 1);
}

template <typename T>
void SwipeAligner::alignFull(const std::vector<size_t> &jobs, const unsigned char *const *targets, const int *targetLengths, s_align *aligns) {
    typedef SwipeLanes<T> V;
    int lanes = V::COUNT;
    int Lq = queryLength;
    const unsigned char *rowLetters = query.data();
    const simd_int *rowBias = (sizeof(T) == 1) ? rowBias8 : rowBias16;
    const simd_int *scoreLo = (sizeof(T) == 1) ? scoreLo8 : scoreLo16;
    const simd_int *scoreHi = (sizeof(T) == 1) ? scoreHi8 : scoreHi16;
    simd_int vZero = simdi_setzero();
    // 16 bit lanes subtract up to 192 from a saturated score
    int saturated = (sizeof(T) == 2) ? V::MAX - 255 : V::MAX - byteBias;
    for (int i = 0; i < Lq; i++) {
        hColumn[i] = vZero;
        eColumn[i] = vZero;
        rowIndex[i] = V::set(i);
    }

    long laneJob[SWIPE_MAX_LANES];
    int laneColumn[SWIPE_MAX_LANES];
    int laneColumns[SWIPE_MAX_LANES];
    int laneBest[SWIPE_MAX_LANES];
    int laneBestColumn[SWIPE_MAX_LANES];
    int laneBestRow[SWIPE_MAX_LANES];
    for (int l = 0; l < lanes; l++) {
        laneJob[l] = -1;
    }
    size_t next = 0;
    int active = 0;

    // 16 bit lanes take two columns per pass
    int columnCount = (sizeof(T) == 2) ? 2 : 1;
    const simd_int *rowSub = rowSub16;
    simd_int vGapOpen = V::set(gapOpen);
    simd_int vGapExtend = V::set(gapExtend);
    unsigned char letters[SWIPE_MAX_LANES];
    T colMax[2][SWIPE_MAX_LANES];
    T colPos[2][SWIPE_MAX_LANES];
    // row bits above 16 of long queries
    T colPosHigh[2][SWIPE_MAX_LANES];
    bool finished[SWIPE_MAX_LANES];
    while (true) {
        for (int l = 0; l < lanes; l++) {
            while (laneJob[l] < 0 && next < jobs.size()) {
                size_t id = jobs[next++];
                if (targetLengths[id] <= 0) {
                    setLaneResult<T>(aligns[id], 0, 0, Lq - 1, saturated);
                    continue;
                }
                laneJob[l] = static_cast<long>(id);
                laneColumn[l] = 0;
                laneColumns[l] = targetLengths[id];
                laneBest[l] = 0;
                laneBestColumn[l] = 0;
                laneBestRow[l] = Lq - 1;
                active++;
                for (int i = 0; i < Lq; i++) {
                    setLaneValue<T>(hColumn + i, l, 0);
                    setLaneValue<T>(eColumn + i, l, 0);
                }
            }
        }
        if (active == 0) {
            break;
        }

        for (int c = 0; c < columnCount; c++) {
            for (int l = 0; l < lanes; l++) {
                long id = laneJob[l];
                int column = laneColumn[l] + c;
                if (id < 0 || column >= laneColumns[l]) {
                    letters[l] = 0;
                } else {
                    letters[l] = targets[id][column];
                }
            }
            simd_int indexLo, indexHi;
            getLaneIndices<T>(letters, indexLo, indexHi);
            for (int a = 0; a < alphabetSize; a++) {
                columnScores[c * alphabetSize + a] = computeLaneScores(scoreLo[a], scoreHi[a], indexLo, indexHi);
            }
        }
        simd_int vColMax[2] = {vZero, vZero};
        simd_int vColPos[2] = {vZero, vZero};
        simd_int vColPosHigh[2] = {vZero, vZero};
        if (sizeof(T) == 2) {
            const simd_int *W0 = columnScores;
            const simd_int *W1 = columnScores + alphabetSize;
            simd_int vHd0 = vZero;
            simd_int vHd1 = vZero;
            simd_int vF0 = vZero;
            simd_int vF1 = vZero;
            for (int i = 0; i < Lq; i++) {
                simd_int rs = rowSub[i];
                simd_int e = eColumn[i];
                simd_int hl = hColumn[i];
                // next F from H before F
                simd_int hp0 = simdi16_max(simdui16_subs(simdi16_adds(vHd0, W0[rowLetters[i]]), rs), e);
                simd_int h0 = simdi16_max(hp0, vF0);
                simd_int e1 = simdi16_max(simdui16_subs(e, vGapExtend), simdui16_subs(h0, vGapOpen));
                vF0 = simdi16_max(simdui16_subs(vF0, vGapExtend), simdui16_subs(hp0, vGapOpen));
                simd_int hp1 = simdi16_max(simdui16_subs(simdi16_adds(vHd1, W1[rowLetters[i]]), rs), e1);
                simd_int h1 = simdi16_max(hp1, vF1);
                eColumn[i] = simdi16_max(simdui16_subs(e1, vGapExtend), simdui16_subs(h1, vGapOpen));
                vF1 = simdi16_max(simdui16_subs(vF1, vGapExtend), simdui16_subs(hp1, vGapOpen));
                vHd0 = hl;
                vHd1 = h0;
                hColumn[i] = h1;
                simd_int gt0 = V::gt(h0, vColMax[0]);
                simd_int gt1 = V::gt(h1, vColMax[1]);
                vColPos[0] = simdi8_blend(vColPos[0], rowIndex[i], gt0);
                vColMax[0] = V::max(vColMax[0], h0);
                vColPos[1] = simdi8_blend(vColPos[1], rowIndex[i], gt1);
                vColMax[1] = V::max(vColMax[1], h1);
                if (Lq > 65536) {
                    simd_int high = V::set(i >> 16);
                    vColPosHigh[0] = simdi8_blend(vColPosHigh[0], high, gt0);
                    vColPosHigh[1] = simdi8_blend(vColPosHigh[1], high, gt1);
                }
            }
        } else {
            const simd_int *W = columnScores;
            simd_int vBias = V::set(byteBias);
            simd_int vHd = vZero;
            simd_int vF = vZero;
            for (int i = 0; i < Lq; i++) {
                simd_int s = V::addBias(W[rowLetters[i]], rowBias[i]);
                simd_int e = eColumn[i];
                // next F from H before F
                simd_int hp = V::max(V::addScore(vHd, s, vBias), e);
                simd_int h = V::max(hp, vF);
                vHd = hColumn[i];
                hColumn[i] = h;
                simd_int colMax = V::max(vColMax[0], h);
                vColPos[0] = simdi8_blend(rowIndex[i], vColPos[0], V::eq(colMax, vColMax[0]));
                vColMax[0] = colMax;
                eColumn[i] = V::max(V::subs(e, vGapExtend), V::subs(h, vGapOpen));
                vF = V::max(V::subs(vF, vGapExtend), V::subs(hp, vGapOpen));
            }
        }
        for (int c = 0; c < columnCount; c++) {
            simdi_storeu((simd_int *) colMax[c], vColMax[c]);
            simdi_storeu((simd_int *) colPos[c], vColPos[c]);
            if (sizeof(T) == 2 && Lq > 65536) {
                simdi_storeu((simd_int *) colPosHigh[c], vColPosHigh[c]);
            }
        }
        // skip the second column after a target ends
        for (int l = 0; l < lanes; l++) {
            finished[l] = (laneJob[l] < 0);
        }
        for (int c = 0; c < columnCount; c++) {
            for (int l = 0; l < lanes; l++) {
                if (finished[l]) {
                    continue;
                }
                if (colMax[c][l] > laneBest[l]) {
                    laneBest[l] = colMax[c][l];
                    laneBestColumn[l] = laneColumn[l];
                    if (sizeof(T) == 2) {
                        laneBestRow[l] = static_cast<int>(static_cast<uint16_t>(colPos[c][l])
                            | (Lq > 65536 ? (static_cast<uint32_t>(static_cast<uint16_t>(colPosHigh[c][l])) << 16) : 0));
                    } else {
                        int row = static_cast<uint8_t>(colPos[c][l]);
                        while (row < Lq && getLaneValue<T>(hColumn + row, l) != colMax[c][l]) {
                            row += 256;
                        }
                        laneBestRow[l] = row;
                    }
                }
                laneColumn[l]++;
                if (laneColumn[l] == laneColumns[l] || laneBest[l] >= saturated) {
                    setLaneResult<T>(aligns[laneJob[l]], laneBest[l], laneBestColumn[l], laneBestRow[l], saturated);
                    laneJob[l] = -1;
                    active--;
                    finished[l] = true;
                }
            }
        }
    }
}

void SwipeAligner::alignForward(size_t count, const unsigned char *const *targets, const int *targetLengths, s_align *aligns) {
    // longest targets first
    auto longestFirst = [targetLengths](size_t a, size_t b) {
        if (targetLengths[a] != targetLengths[b]) {
            return targetLengths[a] > targetLengths[b];
        }
        return a < b;
    };
    jobBuffer.clear();
    for (size_t i = 0; i < count; i++) {
        aligns[i].score1 = 0;
        aligns[i].score2 = 0;
        aligns[i].dbStartPos1 = -1;
        aligns[i].qStartPos1 = -1;
        aligns[i].ref_end2 = -1;
        aligns[i].cigar = NULL;
        aligns[i].cigarLen = 0;
        aligns[i].identicalAACnt = 0;
        if (aligns[i].word == 0) {
            jobBuffer.push_back(i);
        }
    }
    if (jobBuffer.empty() == false) {
        std::sort(jobBuffer.begin(), jobBuffer.end(), longestFirst);
        alignFull<uint8_t>(jobBuffer, targets, targetLengths, aligns);
    }
    jobBuffer.clear();
    for (size_t i = 0; i < count; i++) {
        if (aligns[i].word == 1) {
            jobBuffer.push_back(i);
        }
    }
    if (jobBuffer.empty() == false) {
        std::sort(jobBuffer.begin(), jobBuffer.end(), longestFirst);
        alignFull<int16_t>(jobBuffer, targets, targetLengths, aligns);
    }
}

template <typename T>
void SwipeAligner::fillRing(const BandJob *jobs, size_t jobCount, const unsigned char *const *targets, int ringSize, int m) {
    unsigned char letters[SWIPE_MAX_LANES];
    T valid[SWIPE_MAX_LANES];
    for (int l = 0; l < SwipeLanes<T>::COUNT; l++) {
        letters[l] = 0;
        valid[l] = 0;
        if (static_cast<size_t>(l) >= jobCount) {
            continue;
        }
        const BandJob &job = jobs[l];
        int j = m + job.offset;
        if (j >= job.columnBegin && j < job.columnEnd) {
            letters[l] = targets[job.id][j];
            valid[l] = static_cast<T>(-1);
        }
    }
    int idx = m & (ringSize - 1);
    getLaneIndices<T>(letters, ring[idx], ring[ringSize + idx]);
    ring[2 * ringSize + idx] = simdi_loadu((const simd_int *) valid);
}

template <typename T>
static inline int getTraceCode(const simd_int *trace, int traceWidth, int lane, int row, int column) {
    int codesPerLane = 2 * sizeof(T);
    int codes = getLaneValue<T>(trace + static_cast<size_t>(row) * traceWidth + column / codesPerLane, lane);
    return (codes >> (4 * (codesPerLane - 1 - column % codesPerLane))) & 15;
}

template <typename T>
void SwipeAligner::alignBand(const BandJob *jobs, size_t jobCount, const unsigned char *const *targets, const int *targetLengths,
                             s_align *aligns, std::string *backtraces) {
    typedef SwipeLanes<T> V;
    int width = jobs[0].width;
    int ringSize = 1;
    while (ringSize < width) {
        ringSize *= 2;
    }
    int ringMask = ringSize - 1;
    reserveVectors(ring, ringCapacity, 3 * ringSize);
    simd_int *ringLo = ring;
    simd_int *ringHi = ring + ringSize;
    simd_int *ringValid = ring + 2 * ringSize;
    reserveVectors(band, bandCapacity, 2 * (width + 1));
    const unsigned char *rowLetters = query.data();
    const simd_int *rowBias = (sizeof(T) == 1) ? rowBias8 : rowBias16;
    const simd_int *scoreLo = (sizeof(T) == 1) ? scoreLo8 : scoreLo16;
    const simd_int *scoreHi = (sizeof(T) == 1) ? scoreHi8 : scoreHi16;
    simd_int vZero = simdi_setzero();
    simd_int vGapOpen = V::set(gapOpen);
    simd_int vGapExtend = V::set(gapExtend);
    simd_int vBias = V::set(byteBias);

    int bests[SWIPE_MAX_LANES];
    int rows[SWIPE_MAX_LANES];
    int columns[SWIPE_MAX_LANES];
    int rowLo = INT_MAX;
    int rowHi = 0;
    for (size_t l = 0; l < jobCount; l++) {
        rowLo = std::min(rowLo, jobs[l].rowBegin);
        rowHi = std::max(rowHi, jobs[l].rowEnd);
        bests[l] = 0;
        aligns[jobs[l].id].score1 = 0;
    }
    if (rowLo >= rowHi) {
        return;
    }
    // 4 bit trace codes, 2 or 4 cells per lane
    int codesPerLane = 2 * sizeof(T);
    int traceWidth = (width + codesPerLane - 1) / codesPerLane;
    reserveVectors(trace, traceCapacity, static_cast<size_t>(rowHi - rowLo) * traceWidth);
    simd_int *hRow = band;
    simd_int *fRow = band + width + 1;
    for (int k = 0; k <= width; k++) {
        hRow[k] = vZero;
        fRow[k] = vZero;
    }

    for (int m = rowLo; m < rowLo + width - 1; m++) {
        fillRing<T>(jobs, jobCount, targets, ringSize, m);
    }

    T rowMax[SWIPE_MAX_LANES];
    simd_int vBest = vZero;
    simd_int vOne = V::set(1);
    simd_int vTwo = V::set(2);
    simd_int vThree = V::set(3);
    simd_int vFour = V::set(4);
    simd_int vEight = V::set(8);
    for (int r = rowLo; r < rowHi; r++) {
        fillRing<T>(jobs, jobCount, targets, ringSize, r + width - 1);
        simd_int tLo = scoreLo[rowLetters[r]];
        simd_int tHi = scoreHi[rowLetters[r]];
        simd_int rb = rowBias[r];
        simd_int vHl = vZero;
        simd_int vE = vZero;
        simd_int vRowMax = vZero;
        simd_int *traceRow = trace + static_cast<size_t>(r - rowLo) * traceWidth;
        simd_int codes = vZero;
        for (int k = 0; k < width; k++) {
            int ringIndex = (r + k) & ringMask;
            simd_int s = V::addBias(computeLaneScores(tLo, tHi, ringLo[ringIndex], ringHi[ringIndex]), rb);
            simd_int diag = V::addScore(hRow[k], s, vBias);
            // E from H before E of the left cell
            simd_int eo = V::subs(vHl, vGapOpen);
            simd_int ee = V::subs(vE, vGapExtend);
            vE = V::max(eo, ee);
            simd_int fo = V::subs(hRow[k + 1], vGapOpen);
            simd_int fe = V::subs(fRow[k + 1], vGapExtend);
            simd_int f = V::max(fo, fe);
            simd_int hp = simdi_and(V::max(diag, f), ringValid[ringIndex]);
            simd_int h = simdi_and(V::max(hp, vE), ringValid[ringIndex]);
            // H from 0, 1 diagonal, 2 E, 3 F; +4 E extends, +8 F extends
            simd_int code = simdi8_blend(vThree, vTwo, V::eq(h, vE));
            code = simdi8_blend(code, vOne, V::eq(h, diag));
            code = simdi_andnot(V::eq(h, vZero), code);
            code = simdi_or(code, simdi_and(V::eq(vE, ee), vFour));
            code = simdi_or(code, simdi_and(V::eq(f, fe), vEight));
            codes = (k % codesPerLane == 0) ? code : simdi_or(simdi16_slli(codes, 4), code);
            if (k % codesPerLane == codesPerLane - 1) {
                traceRow[k / codesPerLane] = codes;
            }
            hRow[k] = h;
            fRow[k] = f;
            vHl = hp;
            vRowMax = V::max(vRowMax, h);
        }
        if (width % codesPerLane != 0) {
            for (int k = width; k % codesPerLane != 0; k++) {
                codes = simdi16_slli(codes, 4);
            }
            traceRow[width / codesPerLane] = codes;
        }
        // ties as Farrar
        if (simd_any(simdi_andnot(V::gt(vBest, vRowMax), V::gt(vRowMax, vZero)))) {
            simdi_storeu((simd_int *) rowMax, vRowMax);
            for (size_t l = 0; l < jobCount; l++) {
                if (rowMax[l] < bests[l] || rowMax[l] == 0) {
                    continue;
                }
                int k = 0;
                while (getLaneValue<T>(hRow + k, l) != rowMax[l]) {
                    k++;
                }
                if (rowMax[l] > bests[l] || r + k < rows[l] + columns[l]) {
                    bests[l] = rowMax[l];
                    rows[l] = r;
                    columns[l] = k;
                }
            }
            vBest = V::max(vBest, vRowMax);
        }
    }

    for (size_t l = 0; l < jobCount; l++) {
        s_align &a = aligns[jobs[l].id];
        std::string &backtrace = backtraces[jobs[l].id];
        backtrace.clear();
        a.score1 = bests[l];
        if (bests[l] <= 0) {
            continue;
        }
        const unsigned char *t = targets[jobs[l].id];
        int offset = jobs[l].offset;
        int r = rows[l];
        int k = columns[l];
        int state = 0;
        uint32_t identities = 0;
        size_t maxSteps = static_cast<size_t>(rowHi - rowLo) + width + targetLengths[jobs[l].id] + 1;
        while (backtrace.size() <= maxSteps) {
            int code = getTraceCode<T>(trace, traceWidth, l, r - rowLo, k);
            if (state == 0) {
                if ((code & 3) == 1) {
                    backtrace.push_back('M');
                    identities += (rowLetters[r] == t[r + offset + k]);
                    if (r == rowLo || (getTraceCode<T>(trace, traceWidth, l, r - 1 - rowLo, k) & 3) == 0) {
                        break;
                    }
                    r--;
                    continue;
                }
                state = ((code & 3) == 2) ? 1 : 2;
            }
            if (state == 1) {
                backtrace.push_back('D');
                state = (code & 4) ? 1 : 0;
                k--;
            } else {
                backtrace.push_back('I');
                state = (code & 8) ? 2 : 0;
                r--;
                k++;
            }
            if (r < rowLo || k < 0 || k >= width) {
                break;
            }
        }
        std::reverse(backtrace.begin(), backtrace.end());
        a.qEndPos1 = rows[l];
        a.dbEndPos1 = rows[l] + offset + columns[l];
        a.qStartPos1 = r;
        a.dbStartPos1 = r + offset + k;
        a.identicalAACnt = identities;
    }
}

void SwipeAligner::alignBanded(size_t count, const unsigned char *const *targets, const int *targetLengths, const int *diagonals,
                               s_align *aligns, std::string *backtraces, const int *columnEnds, const int *minScores) {
    int Lq = queryLength;
    bandJobs.clear();
    for (size_t i = 0; i < count; i++) {
        s_align &a = aligns[i];
        if (a.word >= 2) {
            continue;
        }
        a.word = (minScores[i] >= SwipeLanes<uint8_t>::MAX - byteBias) ? 1 : 0;
        a.qStartPos1 = -1;
        a.dbStartPos1 = -1;
        a.score2 = 0;
        a.ref_end2 = -1;
        a.cigar = NULL;
        a.cigarLen = 0;
        a.identicalAACnt = 0;
        BandJob job;
        job.id = i;
        job.width = 2 * MIN_BAND_WIDTH + 1;
        job.offset = diagonals[i] - MIN_BAND_WIDTH;
        job.columnBegin = 0;
        job.columnEnd = columnEnds[i];
        job.rowBegin = std::max(0, job.columnBegin - job.offset - (job.width - 1));
        job.rowEnd = std::min(Lq, job.columnEnd - job.offset);
        bandJobs.push_back(job);
    }
    std::vector<BandJob> pending;
    while (bandJobs.empty() == false) {
        std::sort(bandJobs.begin(), bandJobs.end(), [aligns](const BandJob &a, const BandJob &b) {
            int wordA = aligns[a.id].word;
            int wordB = aligns[b.id].word;
            if (wordA != wordB) {
                return wordA < wordB;
            }
            if (a.width != b.width) {
                return a.width < b.width;
            }
            if (a.rowBegin != b.rowBegin) {
                return a.rowBegin < b.rowBegin;
            }
            return a.id < b.id;
        });
        pending.clear();
        size_t start = 0;
        while (start < bandJobs.size()) {
            int word = aligns[bandJobs[start].id].word;
            int lanes = (word == 0) ? SwipeLanes<uint8_t>::COUNT : SwipeLanes<int16_t>::COUNT;
            size_t end = start + 1;
            while (end < bandJobs.size() && end - start < static_cast<size_t>(lanes) && bandJobs[end].width == bandJobs[start].width
                   && aligns[bandJobs[end].id].word == word) {
                end++;
            }
            // at most 1M trace cells
            size_t traceCells = static_cast<size_t>(Lq + 1) * bandJobs[start].width;
            if (traceCells > 1024 * 1024) {
                for (size_t l = start; l < end; l++) {
                    aligns[bandJobs[l].id].word = 2;
                }
                start = end;
                continue;
            }
            if (word == 0) {
                alignBand<uint8_t>(&bandJobs[start], end - start, targets, targetLengths, aligns, backtraces);
            } else {
                alignBand<int16_t>(&bandJobs[start], end - start, targets, targetLengths, aligns, backtraces);
            }
            for (size_t l = start; l < end; l++) {
                const BandJob &job = bandJobs[l];
                s_align &a = aligns[job.id];
                int best = a.score1;
                int maxScore = (word == 0) ? SwipeLanes<uint8_t>::MAX - byteBias : SwipeLanes<int16_t>::MAX;
                if (best >= maxScore) {
                    a.word = word + 1;
                    if (a.word == 1) {
                        pending.push_back(job);
                    }
                    continue;
                }
                int w = (job.width - 1) / 2;
                int center = job.offset + w;
                bool fullBand = (center - w <= job.columnBegin - (Lq - 1)) && (center + w >= job.columnEnd - 1);
                bool grow = best < minScores[job.id];
                if (grow && fullBand == false) {
                    BandJob wider = job;
                    int w2 = 2 * w;
                    wider.offset = center - w2;
                    wider.width = 2 * w2 + 1;
                    wider.rowBegin = std::max(0, job.columnBegin - wider.offset - (wider.width - 1));
                    wider.rowEnd = std::min(Lq, job.columnEnd - wider.offset);
                    pending.push_back(wider);
                }
            }
            start = end;
        }
        bandJobs.swap(pending);
    }
}

// a where mask is set, else b
static inline simd_int laneSelect(simd_int mask, simd_int a, simd_int b) {
#if defined(SIMDE_ARM_NEON_A64V8_NATIVE) && !defined(AVX2)
    return vreinterpretq_s64_u8(vbslq_u8(vreinterpretq_u8_s64(mask), vreinterpretq_u8_s64(a), vreinterpretq_u8_s64(b)));
#else
    return simdi8_blend(b, a, mask);
#endif
}

// band columns compare signed after adding COLUMN_BIAS
template <typename T>
struct ExtendLanes;

template <>
struct ExtendLanes<uint8_t> {
    static const int COUNT = VECSIZE_INT * 4;
    static const int MAX = UINT8_MAX;
    static const int COLUMN_BIAS = 128;
    static inline simd_int set(int x) { return simdi8_set(static_cast<char>(x)); }
#if defined(AVX2)
    static inline simd_int add(simd_int a, simd_int b) { return _mm256_add_epi8(a, b); }
    static inline simd_int sub(simd_int a, simd_int b) { return _mm256_sub_epi8(a, b); }
#else
    static inline simd_int add(simd_int a, simd_int b) { return _mm_add_epi8(a, b); }
    static inline simd_int sub(simd_int a, simd_int b) { return _mm_sub_epi8(a, b); }
#endif
    static inline simd_int adds(simd_int a, simd_int b) { return simdui8_adds(a, b); }
    static inline simd_int subs(simd_int a, simd_int b) { return simdui8_subs(a, b); }
    static inline simd_int max(simd_int a, simd_int b) { return simdui8_max(a, b); }
    static inline simd_int eq(simd_int a, simd_int b) { return simdi8_eq(a, b); }
    static inline simd_int gt(simd_int a, simd_int b) { return SwipeLanes<uint8_t>::gt(a, b); }
    static inline simd_int columnGt(simd_int a, simd_int b) { return simdi8_gt(a, b); }
    // x where h is above d, 0 elsewhere
    static inline simd_int keepAbove(simd_int x, simd_int h, simd_int d) {
#if defined(SIMDE_ARM_NEON_A64V8_NATIVE) && !defined(AVX2)
        return vreinterpretq_s64_u8(vandq_u8(vcgtq_u8(vreinterpretq_u8_s64(h), vreinterpretq_u8_s64(d)), vreinterpretq_u8_s64(x)));
#else
        return simdi_andnot(simdi8_eq(simdui8_max(h, d), d), x);
#endif
    }
#if defined(SIMDE_ARM_NEON_A64V8_NATIVE) && !defined(AVX2)
    // codes << 4 with the low 4 bits of code
    static inline simd_int insertCode(simd_int codes, simd_int code) {
        return vreinterpretq_s64_u8(vsliq_n_u8(vreinterpretq_u8_s64(code), vreinterpretq_u8_s64(codes), 4));
    }
    static inline simd_int shiftCodes(simd_int codes) { return vreinterpretq_s64_u8(vshlq_n_u8(vreinterpretq_u8_s64(codes), 4)); }
#else
    static inline simd_int shiftCodes(simd_int codes) { return simdi16_slli(codes, 4); }
#endif
};

template <>
struct ExtendLanes<uint16_t> {
    static const int COUNT = VECSIZE_INT * 2;
    static const int MAX = UINT16_MAX;
    static const int COLUMN_BIAS = 0;
    static inline simd_int set(int x) { return simdi16_set(static_cast<short>(x)); }
    static inline simd_int add(simd_int a, simd_int b) { return simdi16_add(a, b); }
    static inline simd_int sub(simd_int a, simd_int b) { return simdi16_sub(a, b); }
    // scores stay far below INT16_MAX
    static inline simd_int adds(simd_int a, simd_int b) { return simdi16_add(a, b); }
    static inline simd_int subs(simd_int a, simd_int b) { return simdui16_subs(a, b); }
    static inline simd_int max(simd_int a, simd_int b) { return simdi16_max(a, b); }
    static inline simd_int eq(simd_int a, simd_int b) { return simdi16_eq(a, b); }
    static inline simd_int gt(simd_int a, simd_int b) { return simdi16_gt(a, b); }
    static inline simd_int columnGt(simd_int a, simd_int b) { return simdi16_gt(a, b); }
    static inline simd_int keepAbove(simd_int x, simd_int h, simd_int d) { return simdi_and(simdi16_gt(h, d), x); }
#if defined(SIMDE_ARM_NEON_A64V8_NATIVE) && !defined(AVX2)
    static inline simd_int insertCode(simd_int codes, simd_int code) {
        return vreinterpretq_s64_u16(vsliq_n_u16(vreinterpretq_u16_s64(code), vreinterpretq_u16_s64(codes), 4));
    }
    static inline simd_int shiftCodes(simd_int codes) { return vreinterpretq_s64_u16(vshlq_n_u16(vreinterpretq_u16_s64(codes), 4)); }
#else
    static inline simd_int shiftCodes(simd_int codes) { return simdi16_slli(codes, 4); }
#endif
};

// codes << 4 | code: H from E 1, from F 2, E opens 4, F opens 8
template <typename T>
static inline simd_int addTraceCode(simd_int codes, simd_int fromE, simd_int fromF, simd_int openE, simd_int openF, bool first) {
    typedef ExtendLanes<T> V;
#if defined(SIMDE_ARM_NEON_A64V8_NATIVE) && !defined(AVX2)
    simd_int code = laneSelect(V::set(1), fromE, fromF);
    code = laneSelect(V::set(3), code, openE);
    code = laneSelect(V::set(7), code, openF);
    return first ? code : V::insertCode(codes, code);
#else
    // the masks are -1, the sum is minus the code
    simd_int code = V::add(V::add(openF, openF), openE);
    code = V::add(V::add(code, code), fromF);
    code = V::add(V::add(code, code), fromE);
    return V::sub(first ? simdi_setzero() : simdi16_slli(codes, 4), code);
#endif
}

#if defined(SIMDE_ARM_NEON_A64V8_NATIVE) && !defined(AVX2)
#define SWIPE_ZIP(n, bits, a, b) vreinterpretq_s64_u##bits(vzip##n##q_u##bits(vreinterpretq_u##bits##_s64(a), vreinterpretq_u##bits##_s64(b)))
#define SWIPE_LO(bits, a, b) SWIPE_ZIP(1, bits, a, b)
#define SWIPE_HI(bits, a, b) SWIPE_ZIP(2, bits, a, b)
#elif defined(AVX2)
#define SWIPE_LO(bits, a, b) _mm256_unpacklo_epi##bits(a, b)
#define SWIPE_HI(bits, a, b) _mm256_unpackhi_epi##bits(a, b)
#else
#define SWIPE_LO(bits, a, b) _mm_unpacklo_epi##bits(a, b)
#define SWIPE_HI(bits, a, b) _mm_unpackhi_epi##bits(a, b)
#endif
#if defined(AVX2)
#define SWIPE_LOAD_LETTERS(p) _mm_loadu_si128((const __m128i *) (p))
#define SWIPE_WIDEN(x) _mm256_cvtepi8_epi16(x)
#define SWIPE_LOAD_LANES(a, b) _mm256_inserti128_si256(_mm256_castsi128_si256(SWIPE_LOAD_LETTERS(a)), SWIPE_LOAD_LETTERS(b), 1)
#else
#define SWIPE_LOAD_LANES(a, b) _mm_loadu_si128((const __m128i *) (a))
#define SWIPE_LOAD_LETTERS(p) _mm_loadl_epi64((const __m128i *) (p))
#define SWIPE_WIDEN(x) _mm_cvtepi8_epi16(x)
#endif

// transpose the 16 bit values of v[0-7] in each 128 bit lane
static inline void transposeLanes8(simd_int *v) {
    simd_int a[8];
    for (int p = 0; p < 4; p++) {
        a[2 * p] = SWIPE_LO(16, v[2 * p], v[2 * p + 1]);
        a[2 * p + 1] = SWIPE_HI(16, v[2 * p], v[2 * p + 1]);
    }
    simd_int b[8];
    for (int p = 0; p < 2; p++) {
        b[4 * p] = SWIPE_LO(32, a[4 * p], a[4 * p + 2]);
        b[4 * p + 1] = SWIPE_HI(32, a[4 * p], a[4 * p + 2]);
        b[4 * p + 2] = SWIPE_LO(32, a[4 * p + 1], a[4 * p + 3]);
        b[4 * p + 3] = SWIPE_HI(32, a[4 * p + 1], a[4 * p + 3]);
    }
    for (int p = 0; p < 4; p++) {
        v[2 * p] = SWIPE_LO(64, b[p], b[p + 4]);
        v[2 * p + 1] = SWIPE_HI(64, b[p], b[p + 4]);
    }
}

// transpose the 8 bit values of v[0-15] in each 128 bit lane
static inline void transposeLanes16(simd_int *v) {
    simd_int a[16];
    for (int p = 0; p < 8; p++) {
        a[2 * p] = SWIPE_LO(8, v[2 * p], v[2 * p + 1]);
        a[2 * p + 1] = SWIPE_HI(8, v[2 * p], v[2 * p + 1]);
    }
    simd_int b[16];
    for (int q = 0; q < 4; q++) {
        b[4 * q] = SWIPE_LO(16, a[4 * q], a[4 * q + 2]);
        b[4 * q + 1] = SWIPE_HI(16, a[4 * q], a[4 * q + 2]);
        b[4 * q + 2] = SWIPE_LO(16, a[4 * q + 1], a[4 * q + 3]);
        b[4 * q + 3] = SWIPE_HI(16, a[4 * q + 1], a[4 * q + 3]);
    }
    for (int r = 0; r < 2; r++) {
        for (int j = 0; j < 4; j++) {
            a[8 * r + 2 * j] = SWIPE_LO(32, b[8 * r + j], b[8 * r + 4 + j]);
            a[8 * r + 2 * j + 1] = SWIPE_HI(32, b[8 * r + j], b[8 * r + 4 + j]);
        }
    }
    for (int m = 0; m < 8; m++) {
        v[2 * m] = SWIPE_LO(64, a[m], a[8 + m]);
        v[2 * m + 1] = SWIPE_HI(64, a[m], a[8 + m]);
    }
}

// scores[c] = rows[l][letters[l][kb + c]] + adds[l] in lane l
template <typename T>
static inline void laneCellScores(const unsigned char *const *letters, const int8_t *const *rows, const T *adds, int kb, simd_int *scores);

template <>
inline void laneCellScores<uint16_t>(const unsigned char *const *letters, const int8_t *const *rows, const uint16_t *adds, int kb, simd_int *scores) {
    int lanes = ExtendLanes<uint16_t>::COUNT;
    for (int l = 0; l < lanes; l++) {
#if defined(SIMDE_ARM_NEON_A64V8_NATIVE) && !defined(AVX2)
        int8x16x2_t row = {{vld1q_s8(rows[l]), vld1q_s8(rows[l] + 16)}};
        scores[l] = vreinterpretq_s64_s16(vaddq_s16(vmovl_s8(vqtbl2_s8(row, vld1_u8(letters[l] + kb))), vdupq_n_s16(adds[l])));
#else
        __m128i index = SWIPE_LOAD_LETTERS(letters[l] + kb);
        __m128i lo = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *) rows[l]), index);
        __m128i hi = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *) (rows[l] + 16)), index);
        // bit 4 of the letter to the blend bit 7
        scores[l] = SWIPE_WIDEN(_mm_blendv_epi8(lo, hi, _mm_slli_epi16(index, 3)));
#endif
    }
    transposeLanes8(scores);
#if defined(AVX2)
    transposeLanes8(scores + 8);
    for (int c = 0; c < 8; c++) {
        simd_int low = scores[c];
        scores[c] = _mm256_permute2x128_si256(low, scores[c + 8], 0x20);
        scores[c + 8] = _mm256_permute2x128_si256(low, scores[c + 8], 0x31);
    }
#endif
#if !defined(SIMDE_ARM_NEON_A64V8_NATIVE) || defined(AVX2)
    // add the bias after the transpose on x86
    simd_int vAdds = simdi_loadu((const simd_int *) adds);
    for (int c = 0; c < lanes; c++) {
        scores[c] = simdi16_add(scores[c], vAdds);
    }
#endif
}

// AVX2 holds lanes l and l + 16 in scores[l]
template <>
inline void laneCellScores<uint8_t>(const unsigned char *const *letters, const int8_t *const *rows, const uint8_t *adds, int kb, simd_int *scores) {
#if defined(SIMDE_ARM_NEON_A64V8_NATIVE) && !defined(AVX2)
    for (int l = 0; l < 16; l++) {
        int8x16x2_t row = {{vld1q_s8(rows[l]), vld1q_s8(rows[l] + 16)}};
        scores[l] = vreinterpretq_s64_u8(vaddq_u8(vreinterpretq_u8_s8(vqtbl2q_s8(row, vld1q_u8(letters[l] + kb))), vdupq_n_u8(adds[l])));
    }
    transposeLanes16(scores);
#else
    for (int l = 0; l < 16; l++) {
        simd_int index = SWIPE_LOAD_LANES(letters[l] + kb, letters[l + 16] + kb);
        simd_int lo = simdi8_shuffle(SWIPE_LOAD_LANES(rows[l], rows[l + 16]), index);
        simd_int hi = simdi8_shuffle(SWIPE_LOAD_LANES(rows[l] + 16, rows[l + 16] + 16), index);
        scores[l] = simdi8_blend(lo, hi, simdi16_slli(index, 3));
    }
    transposeLanes16(scores);
    simd_int vAdds = simdi_loadu((const simd_int *) adds);
    for (int c = 0; c < 16; c++) {
        scores[c] = ExtendLanes<uint8_t>::add(scores[c], vAdds);
    }
#endif
}
#undef SWIPE_ZIP
#undef SWIPE_LO
#undef SWIPE_HI
#undef SWIPE_LOAD_LETTERS
#undef SWIPE_LOAD_LANES
#undef SWIPE_WIDEN

SwipeExtender::SwipeExtender(BaseMatrix *subMat, size_t maxSeqLen, bool aaBiasCorrection, float aaBiasCorrectionScale, int gapOpen, int gapExtend)
    : extensionActiveLaneRowCount(0), extensionTotalLaneRowCount(0), byteExtensionActiveLaneRowCount(0), byteExtensionTotalLaneRowCount(0), longExtensionCount(0), rerunExtensionCount(0),
      subMat(subMat), aaBiasCorrection(aaBiasCorrection), aaBiasCorrectionScale(aaBiasCorrectionScale), gapOpen(gapOpen), gapExtend(gapExtend), matrixBias(0),
      band(NULL), bandCapacity(0), trace(NULL), traceCapacity(0), laneTargets(ExtendLanes<uint8_t>::COUNT), laneTraces(ExtendLanes<uint8_t>::COUNT) {
    int alphabetSize = subMat->alphabetSize;
    if (alphabetSize > 32) {
        Debug(Debug::ERROR) << "SWIPE supports alphabets of at most 32 letters\n";
        EXIT(EXIT_FAILURE);
    }
    queryBiasBuffer = new float[maxSeqLen + 1];
    scoreRows = new int8_t[32 * 32];
    for (int a = 0; a < 32; a++) {
        for (int t = 0; t < 32; t++) {
            scoreRows[a * 32 + t] = (a < alphabetSize && t < alphabetSize) ? static_cast<int8_t>(subMat->subMatrix[t][a]) : 0;
            matrixBias = std::max(matrixBias, -static_cast<int>(scoreRows[a * 32 + t]));
        }
    }
}

SwipeExtender::~SwipeExtender() {
    delete[] queryBiasBuffer;
    delete[] scoreRows;
    free(band);
    free(trace);
}

size_t SwipeExtender::addQuery(const Sequence *q) {
    queryOffsets.push_back(queries.size());
    queryLengths.push_back(q->L);
    queries.insert(queries.end(), q->numSequence, q->numSequence + q->L);
    queries.insert(queries.end(), std::reverse_iterator<const unsigned char *>(q->numSequence + q->L), std::reverse_iterator<const unsigned char *>(q->numSequence));
    computeQueryBias(subMat, q, aaBiasCorrection, aaBiasCorrectionScale, queryBiasBuffer, bias);
    biases.insert(biases.end(), bias.begin(), bias.end());
    biases.insert(biases.end(), bias.rbegin(), bias.rend());
    return queryLengths.size() - 1;
}

void SwipeExtender::clearQueries() {
    queries.clear();
    biases.clear();
    queryOffsets.clear();
    queryLengths.clear();
}

void SwipeExtender::release() {
    std::vector<unsigned char>().swap(queries);
    std::vector<int8_t>().swap(biases);
    std::vector<size_t>().swap(queryOffsets);
    std::vector<int>().swap(queryLengths);
    free(band);
    free(trace);
    band = NULL;
    trace = NULL;
    bandCapacity = 0;
    traceCapacity = 0;
    for (size_t l = 0; l < laneTargets.size(); l++) {
        std::vector<unsigned char>().swap(laneTargets[l]);
        std::vector<uint16_t>().swap(laneTraces[l]);
    }
    std::vector<unsigned char>().swap(zeroLetters);
    std::vector<ExtendJob>().swap(extendJobs);
    std::vector<ExtendJob>().swap(extendPending);
}

// a lane holds its scores less its floor, FLOOR below its best score
template <typename T>
void SwipeExtender::extendBand(const ExtendJob *jobs, size_t jobCount, bool backward, int xDrop, const unsigned char *const *targets,
                               const int *targetLengths, s_align *aligns, std::string *backtraces) {
    typedef ExtendLanes<T> V;
    int lanes = V::COUNT;
    int block = (sizeof(T) == 1) ? 16 : lanes;
    int width = jobs[0].left + jobs[0].right + 1;
    int paddedWidth = (width + block - 1) / block * block;
    // 4 bit trace codes, 2 or 4 cells per lane
    int codesPerLane = 2 * sizeof(T);
    int traceWidth = (width + codesPerLane - 1) / codesPerLane;
    // older rows go to laneTraces
    size_t traceSteps = (size_t) 4 * 1024 * 1024 / (codesPerLane * traceWidth);
    reserveVectors(trace, traceCapacity, traceSteps * traceWidth);
    reserveVectors(band, bandCapacity, 2 * (width + 1));
    simd_int *hRow = band;
    simd_int *fRow = band + width + 1;
    simd_int vZero = simdi_setzero();
    // 8 bit scores stay below V::MAX
    simd_int vGapO = V::set(std::min(gapOpen, static_cast<int>(V::MAX)));
    simd_int vGapE = V::set(std::min(gapExtend, static_cast<int>(V::MAX)));
    simd_int vXDrop = V::set(std::min(xDrop, static_cast<int>(V::MAX)));
    simd_int vOne = V::set(1);
    simd_int vTop = V::set(V::MAX);
    simd_int vWidth = V::set(width + V::COLUMN_BIAS);
    simd_int vFloor = V::set(FLOOR);
    // stop after 16 rows below the X-drop, as BlockAligner
    simd_int vDropRows = V::set(15);
    for (int k = 0; k <= width; k++) {
        hRow[k] = vZero;
        fRow[k] = vZero;
    }
    zeroLetters.assign(paddedWidth, 0);

    const ExtendJob *laneJobs[SWIPE_MAX_LANES];
    int laneOffsets[SWIPE_MAX_LANES];
    int laneRows[SWIPE_MAX_LANES];
    int laneEnds[SWIPE_MAX_LANES];
    int columnBeginRows[SWIPE_MAX_LANES];
    int columnEndRows[SWIPE_MAX_LANES];
    const unsigned char *laneQueries[SWIPE_MAX_LANES];
    const int8_t *laneBiases[SWIPE_MAX_LANES];
    const unsigned char *laneLetters[SWIPE_MAX_LANES];
    size_t laneStarts[SWIPE_MAX_LANES];
    int bestRows[SWIPE_MAX_LANES];
    int bestColumns[SWIPE_MAX_LANES];
    int floors[SWIPE_MAX_LANES];
    T laneBests[SWIPE_MAX_LANES] = {};
    T rises[SWIPE_MAX_LANES] = {};
    T laneDropped[SWIPE_MAX_LANES] = {};
    T edgeLefts[SWIPE_MAX_LANES] = {};
    T edgeRights[SWIPE_MAX_LANES] = {};
    T active[SWIPE_MAX_LANES];
    T maskBegins[SWIPE_MAX_LANES];
    T columnEnds[SWIPE_MAX_LANES];
    T scoreAdds[SWIPE_MAX_LANES];
    T scoreSubs[SWIPE_MAX_LANES];
    const int8_t *rows[SWIPE_MAX_LANES];
    const unsigned char *letters[SWIPE_MAX_LANES];
    for (int l = 0; l < lanes; l++) {
        laneJobs[l] = NULL;
    }
    size_t next = 0;
    size_t step = 0;

    auto finish = [&](int l) {
        const ExtendJob &job = *laneJobs[l];
        laneJobs[l] = NULL;
        if (edgeLefts[l] != 0 || edgeRights[l] != 0) {
            ExtendJob wider = job;
            wider.left *= (edgeLefts[l] != 0) ? 2 : 1;
            wider.right *= (edgeRights[l] != 0) ? 2 : 1;
            wider.isWordAlignment = job.isWordAlignment || wider.left + wider.right + 1 > UINT8_MAX;
            // bands grow up to 4096 columns
            if (wider.left + wider.right + 1 <= 4096) {
                extendPending.push_back(wider);
                return;
            }
        }
        s_align &a = aligns[job.id];
        int row = bestRows[l];
        int column = row + laneOffsets[l] + bestColumns[l];
        a.score1 = std::max(0, laneBests[l] + floors[l]);
        if (backward) {
            a.qStartPos1 = queryLengths[job.query] - 1 - row;
            a.dbStartPos1 = targetLengths[job.id] - 1 - column;
        } else {
            a.qEndPos1 = row;
            a.dbEndPos1 = column;
        }
        if (a.score1 == 0) {
            return;
        }
        std::string &backtrace = backtraces[job.id];
        backtrace.clear();
        int i = row;
        int k = bestColumns[l];
        int state = 0;
        while (true) {
            size_t rowStep = static_cast<size_t>(i - job.seedRow);
            int codes;
            if (rowStep * traceWidth < laneTraces[l].size()) {
                codes = laneTraces[l][rowStep * traceWidth + k / codesPerLane];
            } else {
                codes = getLaneValue<T>(trace + (laneStarts[l] + rowStep) % traceSteps * traceWidth + k / codesPerLane, l);
            }
            int code = (codes >> (4 * (codesPerLane - 1 - k % codesPerLane))) & 15;
            if (state == 0) {
                if ((code & 3) == 0) {
                    backtrace.push_back('M');
                    if (i == job.seedRow && k == job.left) {
                        break;
                    }
                    i--;
                    continue;
                }
                state = (code & 1) ? 1 : 2;
            }
            if (state == 1) {
                backtrace.push_back('D');
                state = (code & 4) ? 0 : 1;
                k--;
            } else {
                backtrace.push_back('I');
                state = (code & 8) ? 0 : 2;
                i--;
                k++;
            }
        }
        // a backward trace is already in alignment order
        if (backward == false) {
            std::reverse(backtrace.begin(), backtrace.end());
        }
    };
    T rowMax[SWIPE_MAX_LANES];
    T rowPos[SWIPE_MAX_LANES];
    T laneFlags[SWIPE_MAX_LANES];
    T overflows[SWIPE_MAX_LANES];
    simd_int scores[SWIPE_MAX_LANES];
    // before fullStep the lanes only step on
    size_t fullStep = 0;
    size_t traceStep = SIZE_MAX;
    T columnEndSteps[SWIPE_MAX_LANES];
    int activeCount = 0;
    int endingCount = 0;
    int columnBegin = width;
    int columnEnd = 0;
    int validBegin = 0;
    int validEnd = width;
    while (true) {
        if (step >= fullStep) {
            // free lanes take the next job
            fullStep = SIZE_MAX;
            traceStep = SIZE_MAX;
            activeCount = 0;
            endingCount = 0;
            columnBegin = width;
            columnEnd = 0;
            validBegin = 0;
            validEnd = width;
            for (int l = 0; l < lanes; l++) {
                while (true) {
                    if (laneJobs[l] == NULL) {
                        if (next == jobCount) {
                            break;
                        }
                        const ExtendJob &job = jobs[next++];
                        int targetLength = targetLengths[job.id];
                        const unsigned char *target = targets[job.id];
                        std::vector<unsigned char> &laneTarget = laneTargets[l];
                        laneTarget.assign(job.left + targetLength + paddedWidth, 0);
                        if (backward) {
                            std::reverse_copy(target, target + targetLength, laneTarget.begin() + job.left);
                        } else {
                            std::copy(target, target + targetLength, laneTarget.begin() + job.left);
                        }
                        size_t pos = queryOffsets[job.query] + (backward ? queryLengths[job.query] : 0);
                        laneJobs[l] = &job;
                        laneOffsets[l] = job.seedColumn - job.seedRow - job.left;
                        laneRows[l] = job.seedRow;
                        columnBeginRows[l] = job.seedRow + job.left;
                        columnEndRows[l] = targetLength - laneOffsets[l];
                        laneEnds[l] = std::min(queryLengths[job.query], columnEndRows[l]);
                        laneQueries[l] = queries.data() + pos;
                        laneBiases[l] = biases.data() + pos;
                        laneLetters[l] = laneTarget.data() + job.seedColumn;
                        laneStarts[l] = step;
                        bestRows[l] = job.seedRow;
                        bestColumns[l] = job.left;
                        floors[l] = -FLOOR;
                        rises[l] = 0;
                        laneBests[l] = 0;
                        laneDropped[l] = 0;
                        edgeLefts[l] = 0;
                        edgeRights[l] = 0;
                        for (int k = 0; k <= width; k++) {
                            setLaneValue<T>(hRow + k, l, 0);
                            setLaneValue<T>(fRow + k, l, 0);
                        }
                        setLaneValue<T>(hRow + job.left, l, FLOOR);
                        laneTraces[l].clear();
                    }
                    int i = laneRows[l];
                    if (i >= laneEnds[l]) {
                        finish(l);
                        continue;
                    }
                    int lo = std::max(0, columnBeginRows[l] - i);
                    int hi = std::min(width, columnEndRows[l] - i);
                    bool first = (laneStarts[l] == step);
                    int maskBegin = first ? lo : 0;
                    int bias = laneBiases[l][i];
                    activeCount++;
                    endingCount += (hi < width);
                    columnBegin = std::min(columnBegin, lo);
                    columnEnd = std::max(columnEnd, hi);
                    validBegin = std::max(validBegin, maskBegin);
                    validEnd = std::min(validEnd, hi);
                    fullStep = std::min(fullStep, step + (first ? 1 : laneEnds[l] - i));
                    if (hi == width) {
                        fullStep = std::min(fullStep, step + (columnEndRows[l] - width + 1 - i));
                    }
                    traceStep = std::min(traceStep, laneStarts[l] + traceSteps);
                    active[l] = static_cast<T>(-1);
                    maskBegins[l] = static_cast<T>(maskBegin + V::COLUMN_BIAS);
                    columnEnds[l] = static_cast<T>(hi + V::COLUMN_BIAS);
                    columnEndSteps[l] = (hi < width);
                    scoreAdds[l] = matrixBias + std::max(0, bias);
                    scoreSubs[l] = matrixBias + std::max(0, -bias);
                    rows[l] = scoreRows + 32 * laneQueries[l][i];
                    letters[l] = laneLetters[l] + (step - laneStarts[l]);
                    break;
                }
                // free lanes run on garbage
                if (laneJobs[l] == NULL) {
                    active[l] = 0;
                    maskBegins[l] = static_cast<T>(V::COLUMN_BIAS);
                    columnEnds[l] = static_cast<T>(width + V::COLUMN_BIAS);
                    columnEndSteps[l] = 0;
                    scoreAdds[l] = 0;
                    scoreSubs[l] = 0;
                    rows[l] = scoreRows;
                    letters[l] = zeroLetters.data();
                }
            }
            if (activeCount == 0) {
                break;
            }
        } else {
            columnBegin = std::max(0, columnBegin - 1);
            columnEnd -= (endingCount == activeCount);
            validEnd -= (endingCount > 0);
            simdi_storeu((simd_int *) columnEnds, V::sub(simdi_loadu((const simd_int *) columnEnds),
                                                                        simdi_loadu((const simd_int *) columnEndSteps)));
            for (int l = 0; l < lanes; l++) {
                if (active[l]) {
                    int i = laneRows[l];
                    int bias = laneBiases[l][i];
                    scoreAdds[l] = matrixBias + std::max(0, bias);
                    scoreSubs[l] = matrixBias + std::max(0, -bias);
                    rows[l] = scoreRows + 32 * laneQueries[l][i];
                    letters[l]++;
                }
            }
        }
        extensionActiveLaneRowCount += activeCount;
        extensionTotalLaneRowCount += lanes;
        if (sizeof(T) == 1) {
            byteExtensionActiveLaneRowCount += activeCount;
            byteExtensionTotalLaneRowCount += lanes;
        }
        simd_int vActive = simdi_loadu((const simd_int *) active);
        simd_int vMaskBegin = simdi_loadu((const simd_int *) maskBegins);
        simd_int vColumnEnd = simdi_loadu((const simd_int *) columnEnds);
        simd_int vRise = simdi_loadu((const simd_int *) rises);
        simd_int vScoreSub = V::add(simdi_loadu((const simd_int *) scoreSubs), vRise);
        simd_int vGapOpenRise = V::adds(vGapO, vRise);
        simd_int vGapExtendRise = V::adds(vGapE, vRise);
        simd_int vBest = simdi_loadu((const simd_int *) laneBests);
        simd_int vDropped = simdi_loadu((const simd_int *) laneDropped);
        simd_int vEdgeLeft = simdi_loadu((const simd_int *) edgeLefts);
        simd_int vEdgeRight = simdi_loadu((const simd_int *) edgeRights);
        simd_int *traceRow = trace + step % traceSteps * traceWidth;
        // keep the trace row that this row overwrites
        for (int l = 0; step >= traceStep && l < lanes; l++) {
            if (laneJobs[l] != NULL && step - laneStarts[l] >= traceSteps) {
                longExtensionCount += (step - laneStarts[l] == traceSteps);
                std::vector<uint16_t> &laneTrace = laneTraces[l];
                for (int w = 0; w < traceWidth; w++) {
                    laneTrace.push_back(getLaneValue<T>(traceRow + w, l));
                }
            }
        }
        simd_int codes = vZero;
        simd_int vHl = vZero;
        simd_int vE = vZero;
        simd_int vRowMax = vZero;
        int kb = columnBegin / block * block;
        simd_int vK = V::set(kb + V::COLUMN_BIAS);
        simd_int vPos = vK;
        simd_int hDiag = hRow[kb];
        auto cell = [&](int k, simd_int score, bool masked, bool first) {
            simd_int hAbove = hRow[k + 1];
            // dead cells start no diagonal
            simd_int diag = V::keepAbove(V::subs(V::adds(hDiag, score), vScoreSub), hDiag, vRise);
            // E from H before E of the left cell
            simd_int eo = V::subs(vHl, vGapO);
            simd_int ee = V::subs(vE, vGapE);
            vE = V::max(eo, ee);
            simd_int fo = V::subs(hAbove, vGapOpenRise);
            simd_int fe = V::subs(fRow[k + 1], vGapExtendRise);
            simd_int f = V::max(fo, fe);
            simd_int hp = V::max(diag, f);
            simd_int h = V::max(hp, vE);
            if (masked) {
                simd_int valid = simdi_andnot(V::columnGt(vMaskBegin, vK), V::columnGt(vColumnEnd, vK));
                hp = simdi_and(hp, valid);
                h = simdi_and(h, valid);
                fRow[k] = simdi_and(f, valid);
            } else {
                fRow[k] = f;
            }
            // ties go to the gap and its opening, as BlockAligner
            codes = addTraceCode<T>(codes, V::eq(h, vE), V::eq(h, f), V::eq(vE, eo), V::eq(f, fo), first);
            hRow[k] = h;
            hDiag = hAbove;
            vHl = hp;
            simd_int cellMax = V::max(vRowMax, h);
            vPos = laneSelect(V::eq(cellMax, vRowMax), vPos, vK);
            vRowMax = cellMax;
            vK = V::add(vK, vOne);
        };
        for (; kb < columnEnd && kb + block <= width; kb += block) {
            laneCellScores<T>(letters, rows, scoreAdds, kb, scores);
            bool masked = (kb < validBegin || kb + block > validEnd);
            for (int c = 0; c < block; c += codesPerLane) {
                for (int d = 0; d < codesPerLane; d++) {
                    cell(kb + c + d, scores[c + d], masked, d == 0);
                }
                traceRow[(kb + c) / codesPerLane] = codes;
            }
        }
        if (kb < columnEnd) {
            laneCellScores<T>(letters, rows, scoreAdds, kb, scores);
            for (int k = kb; k < width; k++) {
                cell(k, scores[k - kb], true, k % codesPerLane == 0);
                if (k % codesPerLane == codesPerLane - 1) {
                    traceRow[k / codesPerLane] = codes;
                }
            }
            for (int k = width; k % codesPerLane != 0; k++) {
                codes = V::shiftCodes(codes);
            }
            if (width % codesPerLane != 0) {
                traceRow[width / codesPerLane] = codes;
            }
        }
        // ties end nearest the seed, as Farrar
        if (simd_any(simdi_andnot(V::gt(vBest, vRowMax), vActive))) {
            simdi_storeu((simd_int *) rowMax, vRowMax);
            simdi_storeu((simd_int *) rowPos, vPos);
            for (int l = 0; l < lanes; l++) {
                int pos = static_cast<T>(rowPos[l] - V::COLUMN_BIAS);
                bool better = active[l] && (rowMax[l] > laneBests[l] || (rowMax[l] == laneBests[l] && laneRows[l] + pos < bestRows[l] + bestColumns[l]));
                bestRows[l] = better ? laneRows[l] : bestRows[l];
                bestColumns[l] = better ? pos : bestColumns[l];
            }
            vBest = V::max(vBest, vRowMax);
        }
        // grow a band side whose edge is within the X-drop
        simd_int vAlive = V::max(V::subs(vBest, vXDrop), vOne);
        vEdgeLeft = simdi_or(vEdgeLeft, simdi_andnot(V::gt(vAlive, hRow[0]), vActive));
        vEdgeRight = simdi_or(vEdgeRight, simdi_andnot(V::gt(vAlive, hRow[width - 1]), simdi_and(V::eq(vColumnEnd, vWidth), vActive)));
        vDropped = simdi_and(V::add(vDropped, vOne), simdi_and(V::gt(vAlive, vRowMax), vActive));
        simd_int stop = simdi_or(V::columnGt(vDropped, vDropRows), simdi_and(vEdgeLeft, vEdgeRight));
        // 8 bit lanes that may have saturated
        simd_int overflow = (sizeof(T) == 1) ? simdi_and(V::eq(V::adds(vRowMax, vScoreSub), vTop), vActive) : vZero;
        // the floor follows the best score
        simd_int vNextRise = V::subs(vBest, vFloor);
        simdi_storeu((simd_int *) laneBests, V::subs(vBest, vNextRise));
        simdi_storeu((simd_int *) rises, vNextRise);
        simdi_storeu((simd_int *) laneDropped, vDropped);
        simdi_storeu((simd_int *) edgeLefts, vEdgeLeft);
        simdi_storeu((simd_int *) edgeRights, vEdgeRight);
        for (int l = 0; l < lanes; l++) {
            floors[l] += rises[l];
            laneRows[l]++;
        }
        if (simd_any(simdi_and(simdi_or(stop, overflow), vActive))) {
            simdi_storeu((simd_int *) laneFlags, stop);
            simdi_storeu((simd_int *) overflows, overflow);
            for (int l = 0; l < lanes; l++) {
                if (active[l] == 0) {
                    continue;
                }
                if (overflows[l] != 0) {
                    ExtendJob rerun = *laneJobs[l];
                    rerun.isWordAlignment = true;
                    extendPending.push_back(rerun);
                    laneJobs[l] = NULL;
                    rerunExtensionCount++;
                    fullStep = step + 1;
                } else if (laneFlags[l] != 0) {
                    finish(l);
                    fullStep = step + 1;
                }
            }
        }
        step++;
    }
}

void SwipeExtender::extendBanded(size_t count, const size_t *queryIds, const unsigned char *const *targets, const int *targetLengths, const int *seedQueries,
                                 const int *seedTargets, bool backward, int xDrop, int bandSide, s_align *aligns, std::string *backtraces) {
    extendJobs.clear();
    for (size_t i = 0; i < count; i++) {
        int queryLength = queryLengths[queryIds[i]];
        ExtendJob job;
        job.id = i;
        job.query = queryIds[i];
        job.seedRow = backward ? queryLength - 1 - seedQueries[i] : seedQueries[i];
        job.seedColumn = backward ? targetLengths[i] - 1 - seedTargets[i] : seedTargets[i];
        job.left = bandSide;
        job.right = bandSide;
        job.isWordAlignment = 2 * bandSide + 1 > UINT8_MAX;
        extendJobs.push_back(job);
    }
    while (extendJobs.empty() == false) {
        // most query rows first, 8 bit before 16 bit
        std::sort(extendJobs.begin(), extendJobs.end(), [this](const ExtendJob &a, const ExtendJob &b) {
            if (a.left + a.right != b.left + b.right) {
                return a.left + a.right < b.left + b.right;
            }
            if (a.isWordAlignment != b.isWordAlignment) {
                return b.isWordAlignment;
            }
            int rowsA = queryLengths[a.query] - a.seedRow;
            int rowsB = queryLengths[b.query] - b.seedRow;
            if (rowsA != rowsB) {
                return rowsA > rowsB;
            }
            return a.id < b.id;
        });
        extendPending.clear();
        for (size_t start = 0; start < extendJobs.size(); ) {
            int width = extendJobs[start].left + extendJobs[start].right;
            bool isWordAlignment = extendJobs[start].isWordAlignment;
            size_t end = start + 1;
            while (end < extendJobs.size() && extendJobs[end].left + extendJobs[end].right == width && extendJobs[end].isWordAlignment == isWordAlignment) {
                end++;
            }
            if (isWordAlignment) {
                extendBand<uint16_t>(&extendJobs[start], end - start, backward, xDrop, targets, targetLengths, aligns, backtraces);
            } else {
                extendBand<uint8_t>(&extendJobs[start], end - start, backward, xDrop, targets, targetLengths, aligns, backtraces);
            }
            start = end;
        }
        extendJobs.swap(extendPending);
    }
}
