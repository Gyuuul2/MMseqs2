#include "SwipeAligner.h"
#include "SubstitutionMatrix.h"
#include "Sequence.h"
#include "Util.h"
#include "Debug.h"

#include <algorithm>
#include <climits>
#include <cstring>

template <typename T>
struct SwipeLanes;

template <>
struct SwipeLanes<int8_t> {
    static const int COUNT = VECSIZE_INT * 4;
    static const int MAX = INT8_MAX;
    static inline simd_int set(int x) { return simdi8_set(static_cast<char>(x)); }
    static inline simd_int adds(simd_int a, simd_int b) { return simdi8_adds(a, b); }
    static inline simd_int max(simd_int a, simd_int b) { return simdi8_max(a, b); }
    static inline simd_int gt(simd_int a, simd_int b) { return simdi8_gt(a, b); }
    static inline simd_int eq(simd_int a, simd_int b) { return simdi8_eq(a, b); }
    static inline simd_int addBias(simd_int a, simd_int b) { return simdi8_adds(a, b); }
};

template <>
struct SwipeLanes<int16_t> {
    static const int COUNT = VECSIZE_INT * 2;
    static const int MAX = INT16_MAX;
    static inline simd_int set(int x) { return simdi16_set(static_cast<short>(x)); }
    static inline simd_int adds(simd_int a, simd_int b) { return simdi16_adds(a, b); }
    static inline simd_int max(simd_int a, simd_int b) { return simdi16_max(a, b); }
    static inline simd_int gt(simd_int a, simd_int b) { return simdi16_gt(a, b); }
    static inline simd_int eq(simd_int a, simd_int b) { return simdi16_eq(a, b); }
    static inline simd_int addBias(simd_int a, simd_int b) { return simdi16_add(a, b); }
};

static const int SWIPE_MAX_LANES = VECSIZE_INT * 4;
static const unsigned char SWIPE_ZERO_INDEX = 0x80;
static const size_t SWIPE_MAX_TRACE_CELLS = 16 * 1024 * 1024 / sizeof(simd_int);

// shuffle indices of the lane letters into the 0-15 and 16-31 score tables, the high byte of 16 bit lanes reads 0
template <typename T>
static inline void laneIndices(const unsigned char *letters, simd_int &indexLo, simd_int &indexHi) {
    unsigned char lo[SWIPE_MAX_LANES];
    unsigned char hi[SWIPE_MAX_LANES];
    for (int l = 0; l < SwipeLanes<T>::COUNT; l++) {
        const unsigned char c = letters[l];
        lo[l * sizeof(T)] = (c < 16) ? c : SWIPE_ZERO_INDEX;
        hi[l * sizeof(T)] = (c >= 16) ? static_cast<unsigned char>(c - 16) : SWIPE_ZERO_INDEX;
        if (sizeof(T) == 2) {
            lo[l * 2 + 1] = SWIPE_ZERO_INDEX;
            hi[l * 2 + 1] = SWIPE_ZERO_INDEX;
        }
    }
    indexLo = simdi_loadu(reinterpret_cast<const simd_int *>(lo));
    indexHi = simdi_loadu(reinterpret_cast<const simd_int *>(hi));
}

static inline simd_int laneScores(simd_int scoreLo, simd_int scoreHi, simd_int indexLo, simd_int indexHi) {
    return simdi_or(simdi8_shuffle(scoreLo, indexLo), simdi8_shuffle(scoreHi, indexHi));
}

template <typename T>
static inline T laneValue(const simd_int *v, int lane) {
    return reinterpret_cast<const T *>(v)[lane];
}

template <typename T>
static inline void setLaneValue(simd_int *v, int lane, T value) {
    reinterpret_cast<T *>(v)[lane] = value;
}

SwipeAligner::SwipeAligner(BaseMatrix *subMat, size_t maxSeqLen, bool aaBiasCorrection, float aaBiasCorrectionScale, int gapOpen, int gapExtend)
    : extendLaneRows(0), extendSlotRows(0), subMat(subMat), alphabetSize(subMat->alphabetSize), aaBiasCorrection(aaBiasCorrection), aaBiasCorrectionScale(aaBiasCorrectionScale),
      gapOpen(gapOpen), gapExtend(gapExtend), queryLength(0), capacity(0), rowBias8(NULL), rowBias16(NULL), rowBiasRev8(NULL),
      rowBiasRev16(NULL), rowSub16(NULL), rowSubRev16(NULL), rowIndex(NULL), hColumn(NULL), eColumn(NULL), rowMask(NULL), ring(NULL), ringCapacity(0), bandH(NULL), bandF(NULL), bandCapacity(0),
      trace(NULL), traceCapacity(0) {
    if (alphabetSize > 32) {
        Debug(Debug::ERROR) << "SWIPE supports alphabets of at most 32 letters\n";
        EXIT(EXIT_FAILURE);
    }
    tmpBias = new float[maxSeqLen + 1];
    columnScores = malloc_simd_int(2 * alphabetSize * sizeof(simd_int));
    scoreLo8 = malloc_simd_int(alphabetSize * sizeof(simd_int));
    scoreHi8 = malloc_simd_int(alphabetSize * sizeof(simd_int));
    scoreLo16 = malloc_simd_int(alphabetSize * sizeof(simd_int));
    scoreHi16 = malloc_simd_int(alphabetSize * sizeof(simd_int));
    const int bytes = sizeof(simd_int);
    for (int a = 0; a < alphabetSize; a++) {
        unsigned char lo8[sizeof(simd_int)], hi8[sizeof(simd_int)], lo16[sizeof(simd_int)], hi16[sizeof(simd_int)];
        for (int p = 0; p < bytes; p++) {
            const int bLo = p % 16;
            const int bHi = 16 + p % 16;
            const int sLo = (bLo < alphabetSize) ? subMat->subMatrix[bLo][a] : 0;
            const int sHi = (bHi < alphabetSize) ? subMat->subMatrix[bHi][a] : 0;
            lo8[p] = static_cast<unsigned char>(static_cast<int8_t>(sLo));
            hi8[p] = static_cast<unsigned char>(static_cast<int8_t>(sHi));
            lo16[p] = static_cast<unsigned char>(sLo + 64);
            hi16[p] = static_cast<unsigned char>(sHi + 64);
        }
        scoreLo8[a] = simdi_loadu(reinterpret_cast<const simd_int *>(lo8));
        scoreHi8[a] = simdi_loadu(reinterpret_cast<const simd_int *>(hi8));
        scoreLo16[a] = simdi_loadu(reinterpret_cast<const simd_int *>(lo16));
        scoreHi16[a] = simdi_loadu(reinterpret_cast<const simd_int *>(hi16));
    }
}

SwipeAligner::~SwipeAligner() {
    delete[] tmpBias;
    free(scoreLo8);
    free(scoreHi8);
    free(scoreLo16);
    free(scoreHi16);
    free(rowBias8);
    free(rowBias16);
    free(rowBiasRev8);
    free(rowBiasRev16);
    free(rowSub16);
    free(rowSubRev16);
    free(rowIndex);
    free(hColumn);
    free(eColumn);
    free(rowMask);
    free(columnScores);
    free(ring);
    free(bandH);
    free(bandF);
    free(trace);
}

void SwipeAligner::resize(size_t length) {
    if (length <= capacity) {
        return;
    }
    free(rowBias8);
    free(rowBias16);
    free(rowBiasRev8);
    free(rowBiasRev16);
    free(rowSub16);
    free(rowSubRev16);
    free(rowIndex);
    free(hColumn);
    free(eColumn);
    free(rowMask);
    capacity = length;
    rowBias8 = malloc_simd_int(capacity * sizeof(simd_int));
    rowBias16 = malloc_simd_int(capacity * sizeof(simd_int));
    rowBiasRev8 = malloc_simd_int(capacity * sizeof(simd_int));
    rowBiasRev16 = malloc_simd_int(capacity * sizeof(simd_int));
    rowSub16 = malloc_simd_int(capacity * sizeof(simd_int));
    rowSubRev16 = malloc_simd_int(capacity * sizeof(simd_int));
    rowIndex = malloc_simd_int(capacity * sizeof(simd_int));
    hColumn = malloc_simd_int((capacity + 1) * sizeof(simd_int));
    eColumn = malloc_simd_int((capacity + 1) * sizeof(simd_int));
    rowMask = malloc_simd_int(capacity * sizeof(simd_int));
}

void SwipeAligner::initQuery(const Sequence *q) {
    queryLength = q->L;
    resize(static_cast<size_t>(queryLength) + 1);
    query.assign(q->numSequence, q->numSequence + queryLength);
    queryRev.assign(query.rbegin(), query.rend());
    bias.assign(queryLength, 0);
    if (aaBiasCorrection) {
        SubstitutionMatrix::calcLocalAaBiasCorrection(subMat, q->numSequence, queryLength, tmpBias, aaBiasCorrectionScale);
        for (int i = 0; i < queryLength; i++) {
            bias[i] = static_cast<int8_t>((tmpBias[i] < 0.0) ? tmpBias[i] - 0.5 : tmpBias[i] + 0.5);
        }
    }
    biasRev.assign(bias.rbegin(), bias.rend());
    for (int i = 0; i < queryLength; i++) {
        rowBias8[i] = simdi8_set(static_cast<char>(bias[i]));
        rowBias16[i] = simdi16_set(static_cast<short>(bias[i] - 64));
        rowBiasRev8[i] = simdi8_set(static_cast<char>(biasRev[i]));
        rowBiasRev16[i] = simdi16_set(static_cast<short>(biasRev[i] - 64));
        rowSub16[i] = simdi16_set(static_cast<short>(64 - bias[i]));
        rowSubRev16[i] = simdi16_set(static_cast<short>(64 - biasRev[i]));
    }
}

template <typename T, bool Reverse>
void SwipeAligner::alignFull(const std::vector<size_t> &jobs, const unsigned char *const *targets, const int *targetLengths, s_align *aligns) {
    typedef SwipeLanes<T> V;
    const int lanes = V::COUNT;
    const int Lq = queryLength;
    const unsigned char *rowLetters = Reverse ? queryRev.data() : query.data();
    const simd_int *rowBias = Reverse ? ((sizeof(T) == 1) ? rowBiasRev8 : rowBiasRev16) : ((sizeof(T) == 1) ? rowBias8 : rowBias16);
    const simd_int *scoreLo = (sizeof(T) == 1) ? scoreLo8 : scoreLo16;
    const simd_int *scoreHi = (sizeof(T) == 1) ? scoreHi8 : scoreHi16;
    const simd_int vZero = simdi_setzero();
    const simd_int vGapO = V::set(-gapOpen);
    const simd_int vGapE = V::set(-gapExtend);
    // the 16 bit lanes subtract up to 192 from a score saturated at INT16_MAX, so they count as saturated 255 below it
    const int saturated = (sizeof(T) == 2) ? V::MAX - 255 : V::MAX;
    for (int i = 0; i < Lq; i++) {
        hColumn[i] = vZero;
        eColumn[i] = vZero;
        rowIndex[i] = V::set(i);
        if (Reverse) {
            rowMask[i] = vZero;
        }
    }

    long laneJob[SWIPE_MAX_LANES];
    int laneColumn[SWIPE_MAX_LANES];
    int laneColumns[SWIPE_MAX_LANES];
    int laneBest[SWIPE_MAX_LANES];
    int laneBestColumn[SWIPE_MAX_LANES];
    int laneBestRow[SWIPE_MAX_LANES];
    size_t next = 0;
    int active = 0;

    const auto finish = [&](int l) {
        const size_t id = laneJob[l];
        s_align &a = aligns[id];
        if (Reverse) {
            if (laneBest[l] >= static_cast<int>(a.score1)) {
                a.dbStartPos1 = a.dbEndPos1 - laneBestColumn[l];
                a.qStartPos1 = Lq - 1 - laneBestRow[l];
            }
        } else {
            a.score1 = laneBest[l];
            a.dbEndPos1 = (laneBest[l] == 0) ? -1 : laneBestColumn[l];
            a.qEndPos1 = (laneBest[l] == 0) ? -1 : laneBestRow[l];
            a.word = (laneBest[l] >= saturated) ? ((sizeof(T) == 1) ? 1 : 2) : ((sizeof(T) == 1) ? 0 : 1);
        }
        laneJob[l] = -1;
        active--;
    };
    const auto assign = [&](int l) {
        while (next < jobs.size()) {
            const size_t id = jobs[next++];
            const int columns = Reverse ? aligns[id].dbEndPos1 + 1 : targetLengths[id];
            laneJob[l] = static_cast<long>(id);
            laneColumn[l] = 0;
            laneColumns[l] = columns;
            laneBest[l] = 0;
            laneBestColumn[l] = 0;
            laneBestRow[l] = Reverse ? (Lq - 1 - aligns[id].qEndPos1) : (Lq - 1);
            active++;
            if (columns <= 0) {
                finish(l);
                continue;
            }
            for (int i = 0; i < Lq; i++) {
                setLaneValue<T>(hColumn + i, l, 0);
                setLaneValue<T>(eColumn + i, l, 0);
            }
            if (Reverse) {
                const int rowStart = Lq - 1 - aligns[id].qEndPos1;
                for (int i = 0; i < Lq; i++) {
                    setLaneValue<T>(rowMask + i, l, (i >= rowStart) ? static_cast<T>(-1) : 0);
                }
            }
            return;
        }
        laneJob[l] = -1;
    };
    for (int l = 0; l < lanes; l++) {
        laneJob[l] = -1;
        assign(l);
    }

    // 16 bit lanes take two columns per pass over the query, with H, E and F kept unsigned (saturating at 0 instead of max with 0)
    const int columnCount = (sizeof(T) == 2) ? 2 : 1;
    const simd_int *rowSub = Reverse ? rowSubRev16 : rowSub16;
    const simd_int vGapOpen = V::set(gapOpen);
    const simd_int vGapExtend = V::set(gapExtend);
    unsigned char letters[SWIPE_MAX_LANES];
    T colMax[2][SWIPE_MAX_LANES];
    T colPos[2][SWIPE_MAX_LANES];
    bool finished[SWIPE_MAX_LANES];
    while (active > 0) {
        for (int c = 0; c < columnCount; c++) {
            for (int l = 0; l < lanes; l++) {
                const long id = laneJob[l];
                const int column = laneColumn[l] + c;
                if (id < 0 || column >= laneColumns[l]) {
                    letters[l] = 0;
                } else {
                    letters[l] = Reverse ? targets[id][aligns[id].dbEndPos1 - column] : targets[id][column];
                }
            }
            simd_int indexLo, indexHi;
            laneIndices<T>(letters, indexLo, indexHi);
            for (int a = 0; a < alphabetSize; a++) {
                columnScores[c * alphabetSize + a] = laneScores(scoreLo[a], scoreHi[a], indexLo, indexHi);
            }
        }
        simd_int vColMax[2] = {vZero, vZero};
        simd_int vColPos[2] = {vZero, vZero};
        if (sizeof(T) == 2) {
            const simd_int *W0 = columnScores;
            const simd_int *W1 = columnScores + alphabetSize;
            simd_int vHd0 = vZero;
            simd_int vHd1 = vZero;
            simd_int vF0 = vZero;
            simd_int vF1 = vZero;
            for (int i = 0; i < Lq; i++) {
                const simd_int rs = rowSub[i];
                const simd_int e = eColumn[i];
                const simd_int hl = hColumn[i];
                // H before F gives the next F (F - gap open is below F - gap extend), so F stays off the H chain
                simd_int hp0 = simdi16_max(simdui16_subs(simdi16_adds(vHd0, W0[rowLetters[i]]), rs), e);
                if (Reverse) {
                    hp0 = simdi_and(hp0, rowMask[i]);
                }
                const simd_int h0 = simdi16_max(hp0, vF0);
                const simd_int e1 = simdi16_max(simdui16_subs(e, vGapExtend), simdui16_subs(h0, vGapOpen));
                vF0 = simdi16_max(simdui16_subs(vF0, vGapExtend), simdui16_subs(hp0, vGapOpen));
                simd_int hp1 = simdi16_max(simdui16_subs(simdi16_adds(vHd1, W1[rowLetters[i]]), rs), e1);
                if (Reverse) {
                    hp1 = simdi_and(hp1, rowMask[i]);
                }
                const simd_int h1 = simdi16_max(hp1, vF1);
                eColumn[i] = simdi16_max(simdui16_subs(e1, vGapExtend), simdui16_subs(h1, vGapOpen));
                vF1 = simdi16_max(simdui16_subs(vF1, vGapExtend), simdui16_subs(hp1, vGapOpen));
                vHd0 = hl;
                vHd1 = h0;
                hColumn[i] = h1;
                vColPos[0] = simdi8_blend(vColPos[0], rowIndex[i], V::gt(h0, vColMax[0]));
                vColMax[0] = V::max(vColMax[0], h0);
                vColPos[1] = simdi8_blend(vColPos[1], rowIndex[i], V::gt(h1, vColMax[1]));
                vColMax[1] = V::max(vColMax[1], h1);
            }
        } else {
            const simd_int *W = columnScores;
            simd_int vHd = vZero;
            simd_int vF = vZero;
            for (int i = 0; i < Lq; i++) {
                const simd_int s = V::addBias(W[rowLetters[i]], rowBias[i]);
                const simd_int e = eColumn[i];
                // H before F gives the next F (F - gap open is below F - gap extend), so F stays off the H chain
                simd_int hp = V::max(V::max(V::adds(vHd, s), e), vZero);
                if (Reverse) {
                    hp = simdi_and(hp, rowMask[i]);
                }
                const simd_int h = V::max(hp, vF);
                vHd = hColumn[i];
                hColumn[i] = h;
                const simd_int gt = V::gt(h, vColMax[0]);
                vColMax[0] = V::max(vColMax[0], h);
                vColPos[0] = simdi8_blend(vColPos[0], rowIndex[i], gt);
                eColumn[i] = V::max(V::adds(e, vGapE), V::adds(h, vGapO));
                vF = V::max(V::adds(vF, vGapE), V::adds(hp, vGapO));
            }
        }
        for (int c = 0; c < columnCount; c++) {
            simdi_storeu(reinterpret_cast<simd_int *>(colMax[c]), vColMax[c]);
            simdi_storeu(reinterpret_cast<simd_int *>(colPos[c]), vColPos[c]);
        }
        // the lanes whose target ends with the first column of a pair ignore the second
        for (int l = 0; l < lanes; l++) {
            finished[l] = (laneJob[l] < 0);
        }
        for (int c = 0; c < columnCount; c++) {
            for (int l = 0; l < lanes; l++) {
                if (finished[l]) {
                    continue;
                }
                const long id = laneJob[l];
                if (colMax[c][l] > laneBest[l]) {
                    laneBest[l] = colMax[c][l];
                    laneBestColumn[l] = laneColumn[l];
                    if (sizeof(T) == 2) {
                        laneBestRow[l] = static_cast<uint16_t>(colPos[c][l]);
                    } else {
                        int row = static_cast<uint8_t>(colPos[c][l]);
                        while (row < Lq && laneValue<T>(hColumn + row, l) != colMax[c][l]) {
                            row += 256;
                        }
                        laneBestRow[l] = row;
                    }
                }
                laneColumn[l]++;
                const bool reached = Reverse && laneBest[l] >= static_cast<int>(aligns[id].score1);
                if (laneColumn[l] == laneColumns[l] || laneBest[l] >= saturated || reached) {
                    finish(l);
                    finished[l] = true;
                }
            }
        }
        for (int l = 0; l < lanes; l++) {
            if (laneJob[l] < 0) {
                assign(l);
            }
        }
    }
}

void SwipeAligner::alignForward(size_t count, const unsigned char *const *targets, const int *targetLengths, s_align *aligns) {
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
        alignFull<int8_t, false>(jobBuffer, targets, targetLengths, aligns);
    }
    jobBuffer.clear();
    for (size_t i = 0; i < count; i++) {
        if (aligns[i].word == 1) {
            jobBuffer.push_back(i);
        }
    }
    if (jobBuffer.empty() == false) {
        alignFull<int16_t, false>(jobBuffer, targets, targetLengths, aligns);
    }
}

void SwipeAligner::alignBackward(size_t count, const unsigned char *const *targets, s_align *aligns) {
    for (int word = 0; word < 2; word++) {
        jobBuffer.clear();
        for (size_t i = 0; i < count; i++) {
            if (aligns[i].word == word && aligns[i].dbStartPos1 == -1) {
                jobBuffer.push_back(i);
            }
        }
        if (jobBuffer.empty()) {
            continue;
        }
        if (word == 0) {
            alignFull<int8_t, true>(jobBuffer, targets, NULL, aligns);
        } else {
            alignFull<int16_t, true>(jobBuffer, targets, NULL, aligns);
        }
    }
}

template <typename T, bool Trace>
void SwipeAligner::alignBand(const BandJob *jobs, size_t jobCount, const unsigned char *const *targets, const int *targetLengths,
                             int *bests, int *rows, int *columns, int *firstRows, int *firstColumns, std::string *backtraces, uint32_t *ids) {
    typedef SwipeLanes<T> V;
    const int lanes = V::COUNT;
    const int width = jobs[0].width;
    int ringSize = 1;
    while (ringSize < width) {
        ringSize *= 2;
    }
    const int ringMask = ringSize - 1;
    if (static_cast<size_t>(3 * ringSize) > ringCapacity) {
        free(ring);
        ringCapacity = 3 * ringSize;
        ring = malloc_simd_int(ringCapacity * sizeof(simd_int));
    }
    simd_int *ringLo = ring;
    simd_int *ringHi = ring + ringSize;
    simd_int *ringValid = ring + 2 * ringSize;
    if (static_cast<size_t>(width) + 1 > bandCapacity) {
        free(bandH);
        free(bandF);
        bandCapacity = width + 1;
        bandH = malloc_simd_int(bandCapacity * sizeof(simd_int));
        bandF = malloc_simd_int(bandCapacity * sizeof(simd_int));
    }
    const unsigned char *rowLetters = query.data();
    const simd_int *rowBias = (sizeof(T) == 1) ? rowBias8 : rowBias16;
    const simd_int *scoreLo = (sizeof(T) == 1) ? scoreLo8 : scoreLo16;
    const simd_int *scoreHi = (sizeof(T) == 1) ? scoreHi8 : scoreHi16;
    const simd_int vZero = simdi_setzero();
    const simd_int vGapO = V::set(-gapOpen);
    const simd_int vGapE = V::set(-gapExtend);

    int rowLo = INT_MAX;
    int rowHi = 0;
    for (size_t l = 0; l < jobCount; l++) {
        rowLo = std::min(rowLo, jobs[l].rowBegin);
        rowHi = std::max(rowHi, jobs[l].rowEnd);
        bests[l] = 0;
        rows[l] = -1;
        columns[l] = -1;
    }
    if (rowLo >= rowHi) {
        return;
    }
    // the 4 bit trace codes of 2 (8 bit) or 4 (16 bit) neighbouring cells share a lane
    const int codesPerLane = 2 * sizeof(T);
    const int traceWidth = (width + codesPerLane - 1) / codesPerLane;
    if (Trace) {
        const size_t cells = static_cast<size_t>(rowHi - rowLo) * traceWidth;
        if (cells > traceCapacity) {
            free(trace);
            traceCapacity = cells;
            trace = malloc_simd_int(traceCapacity * sizeof(simd_int));
        }
    }
    simd_int *hRow = bandH;
    simd_int *fRow = bandF;
    for (int k = 0; k <= width; k++) {
        hRow[k] = vZero;
        fRow[k] = vZero;
    }

    unsigned char letters[SWIPE_MAX_LANES];
    T valid[SWIPE_MAX_LANES];
    const auto fillSlot = [&](int m) {
        for (int l = 0; l < lanes; l++) {
            letters[l] = 0;
            valid[l] = 0;
            if (static_cast<size_t>(l) >= jobCount) {
                continue;
            }
            const BandJob &job = jobs[l];
            const int j = m + job.offset;
            if (j >= job.columnBegin && j < job.columnEnd) {
                const unsigned char *t = targets[job.id];
                letters[l] = t[j];
                valid[l] = static_cast<T>(-1);
            }
        }
        const int slot = m & ringMask;
        laneIndices<T>(letters, ringLo[slot], ringHi[slot]);
        ringValid[slot] = simdi_loadu(reinterpret_cast<const simd_int *>(valid));
    };
    for (int m = rowLo; m < rowLo + width - 1; m++) {
        fillSlot(m);
    }

    T rowMax[SWIPE_MAX_LANES];
    simd_int vBest = vZero;
    const simd_int vOne = V::set(1);
    const simd_int vTwo = V::set(2);
    const simd_int vThree = V::set(3);
    const simd_int vFour = V::set(4);
    const simd_int vEight = V::set(8);
    for (int r = rowLo; r < rowHi; r++) {
        fillSlot(r + width - 1);
        const simd_int tLo = scoreLo[rowLetters[r]];
        const simd_int tHi = scoreHi[rowLetters[r]];
        const simd_int rb = rowBias[r];
        simd_int vHl = vZero;
        simd_int vE = vZero;
        simd_int vRowMax = vZero;
        simd_int *traceRow = Trace ? trace + static_cast<size_t>(r - rowLo) * traceWidth : NULL;
        simd_int codes = vZero;
        for (int k = 0; k < width; k++) {
            const int slot = (r + k) & ringMask;
            const simd_int s = V::addBias(laneScores(tLo, tHi, ringLo[slot], ringHi[slot]), rb);
            const simd_int diag = V::adds(hRow[k], s);
            // H before E of the left cell gives E (E - gap open is below E - gap extend), so E stays off the H chain
            const simd_int eo = V::adds(vHl, vGapO);
            const simd_int ee = V::adds(vE, vGapE);
            vE = V::max(eo, ee);
            const simd_int fo = V::adds(hRow[k + 1], vGapO);
            const simd_int fe = V::adds(fRow[k + 1], vGapE);
            const simd_int f = V::max(fo, fe);
            const simd_int hp = simdi_and(V::max(V::max(diag, f), vZero), ringValid[slot]);
            const simd_int h = simdi_and(V::max(hp, vE), ringValid[slot]);
            if (Trace) {
                // H from 0 (H is 0), 1 (diagonal), 2 (E) or 3 (F), 4 if E extends and 8 if F extends
                simd_int code = simdi8_blend(vThree, vTwo, V::eq(h, vE));
                code = simdi8_blend(code, vOne, V::eq(h, diag));
                code = simdi_andnot(V::eq(h, vZero), code);
                code = simdi_or(code, simdi_andnot(V::gt(eo, ee), vFour));
                code = simdi_or(code, simdi_andnot(V::gt(fo, fe), vEight));
                codes = (k % codesPerLane == 0) ? code : simdi_or(simdi16_slli(codes, 4), code);
                if (k % codesPerLane == codesPerLane - 1) {
                    traceRow[k / codesPerLane] = codes;
                }
            }
            hRow[k] = h;
            fRow[k] = f;
            vHl = hp;
            vRowMax = V::max(vRowMax, h);
        }
        if (Trace && width % codesPerLane != 0) {
            for (int k = width; k % codesPerLane != 0; k++) {
                codes = simdi16_slli(codes, 4);
            }
            traceRow[width / codesPerLane] = codes;
        }
        if (simdi8_movemask(V::gt(vRowMax, vBest)) != 0) {
            simdi_storeu(reinterpret_cast<simd_int *>(rowMax), vRowMax);
            for (size_t l = 0; l < jobCount; l++) {
                if (rowMax[l] > bests[l]) {
                    bests[l] = rowMax[l];
                    rows[l] = r;
                    int k = 0;
                    while (laneValue<T>(hRow + k, l) != rowMax[l]) {
                        k++;
                    }
                    columns[l] = k;
                }
            }
            vBest = V::max(vBest, vRowMax);
        }
    }

    if (Trace) {
        for (size_t l = 0; l < jobCount; l++) {
            std::string &backtrace = backtraces[jobs[l].id];
            backtrace.clear();
            ids[l] = 0;
            if (bests[l] <= 0) {
                continue;
            }
            const unsigned char *t = targets[jobs[l].id];
            const int offset = jobs[l].offset;
            int r = rows[l];
            int k = columns[l];
            int state = 0;
            uint32_t identities = 0;
            const auto traceCode = [&](int row, int column) {
                const int codes = laneValue<T>(trace + static_cast<size_t>(row - rowLo) * traceWidth + column / codesPerLane, l);
                return (codes >> (4 * (codesPerLane - 1 - column % codesPerLane))) & 15;
            };
            const size_t maxSteps = static_cast<size_t>(rowHi - rowLo) + width + targetLengths[jobs[l].id] + 1;
            while (backtrace.size() <= maxSteps) {
                const int code = traceCode(r, k);
                if (state == 0) {
                    if ((code & 3) == 1) {
                        backtrace.push_back('M');
                        identities += (rowLetters[r] == t[r + offset + k]);
                        if (r == rowLo || (traceCode(r - 1, k) & 3) == 0) {
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
            ids[l] = identities;
            firstRows[l] = r;
            firstColumns[l] = k;
        }
    }
}

void SwipeAligner::runBanded(std::vector<BandJob> &jobs, bool doTrace, const unsigned char *const *targets, const int *targetLengths,
                             int *bandWidths, s_align *aligns, std::string *backtraces, const int *minScores) {
    const int Lq = queryLength;
    std::vector<BandJob> pending;
    while (jobs.empty() == false) {
        std::sort(jobs.begin(), jobs.end(), [aligns](const BandJob &a, const BandJob &b) {
            const int wordA = aligns[a.id].word;
            const int wordB = aligns[b.id].word;
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
        while (start < jobs.size()) {
            const int word = aligns[jobs[start].id].word;
            const int lanes = (word == 0) ? SwipeLanes<int8_t>::COUNT : SwipeLanes<int16_t>::COUNT;
            size_t end = start + 1;
            while (end < jobs.size() && end - start < static_cast<size_t>(lanes) && jobs[end].width == jobs[start].width
                   && aligns[jobs[end].id].word == word) {
                end++;
            }
            const size_t n = end - start;
            bestScores.resize(n);
            bestRows.resize(n);
            bestColumns.resize(n);
            identities.resize(n);
            startRows.resize(n);
            startColumns.resize(n);
            const size_t traceCells = static_cast<size_t>(Lq + 1) * jobs[start].width;
            const bool canTrace = traceCells <= SWIPE_MAX_TRACE_CELLS;
            if (doTrace && canTrace == false) {
                for (size_t l = start; l < end; l++) {
                    aligns[jobs[l].id].word = 2;
                }
                start = end;
                continue;
            }
#define SWIPE_BAND(T, TR) alignBand<T, TR>(&jobs[start], n, targets, targetLengths, bestScores.data(), bestRows.data(), bestColumns.data(), startRows.data(), startColumns.data(), backtraces, identities.data())
            if (word == 0) {
                if (doTrace) { SWIPE_BAND(int8_t, true); } else { SWIPE_BAND(int8_t, false); }
            } else {
                if (doTrace) { SWIPE_BAND(int16_t, true); } else { SWIPE_BAND(int16_t, false); }
            }
#undef SWIPE_BAND
            for (size_t l = 0; l < n; l++) {
                const BandJob &job = jobs[start + l];
                s_align &a = aligns[job.id];
                const int best = bestScores[l];
                const int maxScore = (word == 0) ? SwipeLanes<int8_t>::MAX : SwipeLanes<int16_t>::MAX;
                if (best >= maxScore) {
                    a.word = word + 1;
                    if (a.word == 1) {
                        pending.push_back(job);
                    }
                    continue;
                }
                const int w = (job.width - 1) / 2;
                const int center = job.offset + w;
                const bool fullBand = (center - w <= job.columnBegin - (Lq - 1)) && (center + w >= job.columnEnd - 1);
                const int k = bestColumns[l];
                const bool grow = (best > 0 && doTrace == false && (k <= 1 || k >= job.width - 2)) || (minScores != NULL && best < minScores[job.id]);
                if (grow && fullBand == false) {
                    BandJob wider = job;
                    const int w2 = 2 * w;
                    wider.offset = center - w2;
                    wider.width = 2 * w2 + 1;
                    wider.rowBegin = std::max(0, job.columnBegin - wider.offset - (wider.width - 1));
                    wider.rowEnd = std::min(Lq, job.columnEnd - wider.offset);
                    bandWidths[job.id] = w2;
                    pending.push_back(wider);
                    continue;
                }
                if (bestRows[l] < 0) {
                    a.score1 = 0;
                    continue;
                }
                const int row = bestRows[l];
                const int column = row + job.offset + bestColumns[l];
                if (doTrace) {
                    a.score1 = best;
                    a.qEndPos1 = row;
                    a.dbEndPos1 = column;
                    a.qStartPos1 = startRows[l];
                    a.dbStartPos1 = startRows[l] + job.offset + startColumns[l];
                    a.identicalAACnt = identities[l];
                } else {
                    a.score1 = best;
                    a.qEndPos1 = row;
                    a.dbEndPos1 = column;
                }
            }
            start = end;
        }
        jobs.swap(pending);
    }
}

void SwipeAligner::alignBanded(size_t count, const unsigned char *const *targets, const int *targetLengths, const int *diagonals, int *bandWidths,
                               s_align *aligns, std::string *backtraces, const int *columnEnds, const int *minScores) {
    const int Lq = queryLength;
    const bool doTrace = (backtraces != NULL);
    bandJobs.clear();
    for (size_t i = 0; i < count; i++) {
        s_align &a = aligns[i];
        if (doTrace == false) {
            a.score1 = 0;
            a.qEndPos1 = -1;
            a.dbEndPos1 = -1;
            a.word = (a.word == 1) ? 1 : 0;
        } else {
            a.word = (minScores != NULL && minScores[i] >= SwipeLanes<int8_t>::MAX) ? 1 : ((a.score1 >= static_cast<uint32_t>(SwipeLanes<int8_t>::MAX)) ? 1 : 0);
        }
        a.qStartPos1 = -1;
        a.dbStartPos1 = -1;
        a.score2 = 0;
        a.ref_end2 = -1;
        a.cigar = NULL;
        a.cigarLen = 0;
        a.identicalAACnt = 0;
        if (bandWidths[i] <= 0) {
            bandWidths[i] = MIN_BAND_WIDTH;
        }
        BandJob job;
        job.id = i;
        job.width = 2 * bandWidths[i] + 1;
        job.offset = diagonals[i] - bandWidths[i];
        job.columnBegin = 0;
        job.columnEnd = (columnEnds != NULL) ? columnEnds[i] : targetLengths[i];
        job.rowBegin = std::max(0, job.columnBegin - job.offset - (job.width - 1));
        job.rowEnd = std::min(Lq, job.columnEnd - job.offset);
        bandJobs.push_back(job);
    }
    runBanded(bandJobs, doTrace, targets, targetLengths, bandWidths, aligns, backtraces, minScores);
}


// a lane stops after as many rows in a row below the X-drop as BlockAligner takes (two steps of eight)
static const int SWIPE_XDROP_ROWS = 16;
// the scores of a lane whose best score passes this go down by it, so that 16 bit lanes extend any length
static const int SWIPE_REBASE_SCORE = 16384;

void SwipeAligner::extendBand(const ExtendJob *jobs, size_t jobCount, bool backward, int xDrop, const unsigned char *const *targets,
                              const int *targetLengths, s_align *aligns, std::string *backtraces, bool *growLefts, bool *growRights) {
    typedef SwipeLanes<int16_t> V;
    const int lanes = V::COUNT;
    const int Lq = queryLength;
    const int width = jobs[0].left + jobs[0].right + 1;
    int ringSize = 1;
    while (ringSize < width + 1) {
        ringSize *= 2;
    }
    const int ringMask = ringSize - 1;
    if (static_cast<size_t>(3 * ringSize) > ringCapacity) {
        free(ring);
        ringCapacity = 3 * ringSize;
        ring = malloc_simd_int(ringCapacity * sizeof(simd_int));
    }
    simd_int *ringLo = ring;
    simd_int *ringHi = ring + ringSize;
    simd_int *ringValid = ring + 2 * ringSize;
    if (static_cast<size_t>(width) + 1 > bandCapacity) {
        free(bandH);
        free(bandF);
        bandCapacity = width + 1;
        bandH = malloc_simd_int(bandCapacity * sizeof(simd_int));
        bandF = malloc_simd_int(bandCapacity * sizeof(simd_int));
    }
    simd_int *hRow = bandH;
    simd_int *fRow = bandF;
    const unsigned char *rowLetters = backward ? queryRev.data() : query.data();
    const simd_int *rowBias = backward ? rowBiasRev16 : rowBias16;
    const simd_int vNeg = V::set(SHRT_MIN);
    const simd_int vZero = simdi_setzero();
    const simd_int vGapO = V::set(-gapOpen);
    const simd_int vGapE = V::set(-gapExtend);
    const simd_int vXDrop = V::set(-xDrop);
    const simd_int vOne = V::set(1);
    const simd_int vTwo = V::set(2);
    const simd_int vThree = V::set(3);
    const simd_int vFour = V::set(4);
    const simd_int vEight = V::set(8);
    const simd_int vDropRows = V::set(SWIPE_XDROP_ROWS - 1);
    const simd_int vRebaseMin = V::set(SWIPE_REBASE_SCORE - 1);
    const simd_int vRebase = V::set(-SWIPE_REBASE_SCORE);
    for (int k = 0; k <= width; k++) {
        hRow[k] = vNeg;
        fRow[k] = vNeg;
    }

    int offsets[SWIPE_MAX_LANES];
    int lengths[SWIPE_MAX_LANES];
    const unsigned char *letters[SWIPE_MAX_LANES];
    bool done[SWIPE_MAX_LANES];
    int bests[SWIPE_MAX_LANES];
    int bestRows[SWIPE_MAX_LANES];
    int bestColumns[SWIPE_MAX_LANES];
    int shifts[SWIPE_MAX_LANES];
    int rowLo = INT_MAX;
    for (int l = 0; l < lanes; l++) {
        done[l] = (static_cast<size_t>(l) >= jobCount);
        offsets[l] = 0;
        lengths[l] = 0;
        if (done[l]) {
            continue;
        }
        const ExtendJob &job = jobs[l];
        offsets[l] = job.seedColumn - job.seedRow - job.left;
        lengths[l] = targetLengths[job.id];
        letters[l] = targets[job.id];
        bests[l] = SHRT_MIN;
        bestRows[l] = job.seedRow;
        bestColumns[l] = job.left;
        shifts[l] = 0;
        rowLo = std::min(rowLo, job.seedRow);
    }

    unsigned char slotLetters[SWIPE_MAX_LANES];
    int16_t slotValid[SWIPE_MAX_LANES];
    const auto fillSlot = [&](int m) {
        for (int l = 0; l < lanes; l++) {
            slotLetters[l] = 0;
            slotValid[l] = 0;
            if (static_cast<size_t>(l) >= jobCount) {
                continue;
            }
            const int j = m + offsets[l];
            if (j >= jobs[l].seedColumn && j < lengths[l]) {
                slotLetters[l] = backward ? letters[l][lengths[l] - 1 - j] : letters[l][j];
                slotValid[l] = -1;
            }
        }
        const int slot = m & ringMask;
        laneIndices<int16_t>(slotLetters, ringLo[slot], ringHi[slot]);
        ringValid[slot] = simdi_loadu(reinterpret_cast<const simd_int *>(slotValid));
    };

    // the 4 bit trace codes of 4 neighbouring cells share a lane
    const int traceWidth = (width + 3) / 4;
    int16_t active[SWIPE_MAX_LANES];
    int16_t rowMax[SWIPE_MAX_LANES];
    int16_t rowPos[SWIPE_MAX_LANES];
    int16_t laneFlags[SWIPE_MAX_LANES];
    simd_int vBest = vNeg;
    simd_int vDropped = vZero;
    simd_int vEdgeLeft = vZero;
    simd_int vEdgeRight = vZero;
    int fillEnd = rowLo;
    for (int r = rowLo; r < Lq; r++) {
        int kBegin = INT_MAX;
        int kEnd = 0;
        int activeCount = 0;
        bool running = false;
        for (int l = 0; l < lanes; l++) {
            active[l] = 0;
            if (done[l]) {
                continue;
            }
            running = true;
            if (r < jobs[l].seedRow) {
                continue;
            }
            const int lo = std::max(0, jobs[l].seedColumn - r - offsets[l]);
            const int hi = std::min(width, lengths[l] - r - offsets[l]);
            if (lo >= hi) {
                done[l] = true;
                continue;
            }
            active[l] = -1;
            activeCount++;
            kBegin = std::min(kBegin, lo);
            kEnd = std::max(kEnd, hi);
            if (r == jobs[l].seedRow) {
                setLaneValue<int16_t>(hRow + jobs[l].left, l, 0);
            }
        }
        if (running == false) {
            break;
        }
        if (activeCount == 0) {
            continue;
        }
        extendLaneRows += activeCount;
        extendSlotRows += lanes;
        const size_t cells = static_cast<size_t>(r - rowLo + 1) * traceWidth;
        if (cells > traceCapacity) {
            const size_t grownCapacity = std::max(cells, 2 * traceCapacity);
            simd_int *grown = malloc_simd_int(grownCapacity * sizeof(simd_int));
            if (trace != NULL) {
                memcpy(grown, trace, traceCapacity * sizeof(simd_int));
                free(trace);
            }
            trace = grown;
            traceCapacity = grownCapacity;
        }
        // a lane starting later can start left of the others in the band, only the slots left of the row are never read again
        fillEnd = std::max(fillEnd, r);
        while (fillEnd < r + kEnd) {
            fillSlot(fillEnd++);
        }
        const simd_int vActive = simdi_loadu(reinterpret_cast<const simd_int *>(active));
        const simd_int tLo = scoreLo16[rowLetters[r]];
        const simd_int tHi = scoreHi16[rowLetters[r]];
        const simd_int rb = rowBias[r];
        simd_int *traceRow = trace + static_cast<size_t>(r - rowLo) * traceWidth;
        simd_int codes = vZero;
        simd_int vHl = vNeg;
        simd_int vE = vNeg;
        simd_int vRowMax = vNeg;
        simd_int vPos = vZero;
        simd_int vK = V::set(kBegin);
        for (int k = kBegin; k < kEnd; k++) {
            const int slot = (r + k) & ringMask;
            const simd_int s = simdi16_add(laneScores(tLo, tHi, ringLo[slot], ringHi[slot]), rb);
            const simd_int diag = simdi16_adds(hRow[k], s);
            // H before E of the left cell gives E (E - gap open is below E - gap extend), so E stays off the H chain
            const simd_int eo = simdi16_adds(vHl, vGapO);
            const simd_int ee = simdi16_adds(vE, vGapE);
            vE = simdi16_max(eo, ee);
            const simd_int fo = simdi16_adds(hRow[k + 1], vGapO);
            const simd_int fe = simdi16_adds(fRow[k + 1], vGapE);
            const simd_int f = simdi16_max(fo, fe);
            const simd_int valid = simdi_and(ringValid[slot], vActive);
            const simd_int hp = simdi8_blend(vNeg, simdi16_max(diag, f), valid);
            const simd_int h = simdi8_blend(vNeg, simdi16_max(hp, vE), valid);
            // H from 1 (diagonal), 2 (E) or 3 (F), 4 if E extends and 8 if F extends; ties go to the gap and to its opening, as in BlockAligner
            simd_int code = simdi8_blend(vOne, vThree, simdi16_eq(h, f));
            code = simdi8_blend(code, vTwo, simdi16_eq(h, vE));
            code = simdi_or(code, simdi_and(simdi16_gt(ee, eo), vFour));
            code = simdi_or(code, simdi_and(simdi16_gt(fe, fo), vEight));
            codes = simdi_or(simdi16_slli(codes, 4), code);
            if (k % 4 == 3) {
                traceRow[k / 4] = codes;
            }
            hRow[k] = h;
            fRow[k] = simdi8_blend(vNeg, f, valid);
            vHl = hp;
            vPos = simdi8_blend(vK, vPos, simdi16_gt(vRowMax, h));
            vRowMax = simdi16_max(vRowMax, h);
            vK = simdi16_add(vK, vOne);
        }
        if (kEnd % 4 != 0) {
            for (int k = kEnd; k % 4 != 0; k++) {
                codes = simdi16_slli(codes, 4);
            }
            traceRow[kEnd / 4] = codes;
        }
        // ties go to the larger target position, then the later row, as in BlockAligner
        if (simdi8_movemask(simdi_andnot(simdi16_gt(vBest, vRowMax), vActive)) != 0) {
            simdi_storeu(reinterpret_cast<simd_int *>(rowMax), vRowMax);
            simdi_storeu(reinterpret_cast<simd_int *>(rowPos), vPos);
            for (int l = 0; l < lanes; l++) {
                if (active[l] && (rowMax[l] > bests[l] || (rowMax[l] == bests[l] && r + rowPos[l] >= bestRows[l] + bestColumns[l]))) {
                    bests[l] = rowMax[l];
                    bestRows[l] = r;
                    bestColumns[l] = rowPos[l];
                }
            }
            vBest = simdi16_max(vBest, vRowMax);
        }
        // a side of the band grows if its edge is within the X-drop of the best score
        const simd_int vAlive = simdi16_adds(vBest, vXDrop);
        vEdgeLeft = simdi_or(vEdgeLeft, simdi_andnot(simdi16_gt(vAlive, hRow[0]), vActive));
        vEdgeRight = simdi_or(vEdgeRight, simdi_andnot(simdi16_gt(vAlive, hRow[width - 1]), vActive));
        vDropped = simdi_and(simdi16_add(vDropped, vOne), simdi_and(simdi16_gt(vAlive, vRowMax), vActive));
        const simd_int stop = simdi16_gt(vDropped, vDropRows);
        if (simdi8_movemask(stop) != 0) {
            simdi_storeu(reinterpret_cast<simd_int *>(laneFlags), stop);
            for (int l = 0; l < lanes; l++) {
                done[l] = done[l] || laneFlags[l] != 0;
            }
        }
        const simd_int rebase = simdi16_gt(vBest, vRebaseMin);
        if (simdi8_movemask(rebase) != 0) {
            const simd_int vShift = simdi_and(rebase, vRebase);
            for (int k = 0; k <= width; k++) {
                hRow[k] = simdi16_adds(hRow[k], vShift);
                fRow[k] = simdi16_adds(fRow[k], vShift);
            }
            vBest = simdi16_adds(vBest, vShift);
            simdi_storeu(reinterpret_cast<simd_int *>(laneFlags), rebase);
            for (int l = 0; l < lanes; l++) {
                if (laneFlags[l] != 0) {
                    bests[l] -= SWIPE_REBASE_SCORE;
                    shifts[l] += SWIPE_REBASE_SCORE;
                }
            }
        }
    }

    int16_t edgeLefts[SWIPE_MAX_LANES];
    int16_t edgeRights[SWIPE_MAX_LANES];
    simdi_storeu(reinterpret_cast<simd_int *>(edgeLefts), vEdgeLeft);
    simdi_storeu(reinterpret_cast<simd_int *>(edgeRights), vEdgeRight);
    for (size_t l = 0; l < jobCount; l++) {
        const ExtendJob &job = jobs[l];
        s_align &a = aligns[job.id];
        growLefts[l] = (edgeLefts[l] != 0);
        growRights[l] = (edgeRights[l] != 0);
        const int row = bestRows[l];
        const int column = row + offsets[l] + bestColumns[l];
        a.score1 = std::max(0, bests[l] + shifts[l]);
        if (backward) {
            a.qStartPos1 = Lq - 1 - row;
            a.dbStartPos1 = lengths[l] - 1 - column;
        } else {
            a.qEndPos1 = row;
            a.dbEndPos1 = column;
        }
        if (growLefts[l] || growRights[l] || a.score1 == 0) {
            continue;
        }
        std::string &backtrace = backtraces[job.id];
        backtrace.clear();
        int i = row;
        int k = bestColumns[l];
        int state = 0;
        while (true) {
            const int laneCodes = static_cast<uint16_t>(laneValue<int16_t>(trace + static_cast<size_t>(i - rowLo) * traceWidth + k / 4, l));
            const int code = (laneCodes >> (4 * (3 - k % 4))) & 15;
            if (state == 0) {
                if ((code & 3) == 1) {
                    backtrace.push_back('M');
                    if (i == job.seedRow && k == job.left) {
                        break;
                    }
                    i--;
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
                i--;
                k++;
            }
        }
        // a backward extension walks from its start to the seed, which is already the alignment order
        if (backward == false) {
            std::reverse(backtrace.begin(), backtrace.end());
        }
    }
}

void SwipeAligner::extendBanded(size_t count, const unsigned char *const *targets, const int *targetLengths, const int *seedQueries, const int *seedTargets,
                                bool backward, int xDrop, int bandSide, s_align *aligns, std::string *backtraces) {
    const int Lq = queryLength;
    // a job whose trace may not fit or whose band column passes 16 bit goes back with word 2, as in alignBand
    const auto queue = [&](std::vector<ExtendJob> &jobs, const ExtendJob &job) {
        const int width = job.left + job.right + 1;
        if (width > SHRT_MAX || static_cast<size_t>(Lq - job.seedRow) * ((width + 3) / 4) > SWIPE_MAX_TRACE_CELLS) {
            aligns[job.id].word = 2;
            return;
        }
        jobs.push_back(job);
    };
    extendJobs.clear();
    for (size_t i = 0; i < count; i++) {
        aligns[i].word = 0;
        ExtendJob job;
        job.id = i;
        job.seedRow = backward ? Lq - 1 - seedQueries[i] : seedQueries[i];
        job.seedColumn = backward ? targetLengths[i] - 1 - seedTargets[i] : seedTargets[i];
        job.left = bandSide;
        job.right = bandSide;
        queue(extendJobs, job);
    }
    std::vector<ExtendJob> &pending = extendPending;
    bool growLefts[SWIPE_MAX_LANES];
    bool growRights[SWIPE_MAX_LANES];
    const size_t lanes = SwipeLanes<int16_t>::COUNT;
    while (extendJobs.empty() == false) {
        // the lanes of a batch share the band width and start on close rows
        std::sort(extendJobs.begin(), extendJobs.end(), [](const ExtendJob &a, const ExtendJob &b) {
            if (a.left + a.right != b.left + b.right) {
                return a.left + a.right < b.left + b.right;
            }
            if (a.seedRow != b.seedRow) {
                return a.seedRow < b.seedRow;
            }
            return a.id < b.id;
        });
        pending.clear();
        for (size_t start = 0; start < extendJobs.size(); ) {
            const int width = extendJobs[start].left + extendJobs[start].right;
            size_t end = start + 1;
            while (end < extendJobs.size() && end - start < lanes && extendJobs[end].left + extendJobs[end].right == width) {
                end++;
            }
            extendBand(&extendJobs[start], end - start, backward, xDrop, targets, targetLengths, aligns, backtraces, growLefts, growRights);
            for (size_t l = 0; l < end - start; l++) {
                if (growLefts[l] || growRights[l]) {
                    ExtendJob job = extendJobs[start + l];
                    job.left *= (growLefts[l] ? 2 : 1);
                    job.right *= (growRights[l] ? 2 : 1);
                    queue(pending, job);
                }
            }
            start = end;
        }
        extendJobs.swap(pending);
    }
}
