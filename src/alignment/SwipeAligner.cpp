#include "SwipeAligner.h"
#include "SubstitutionMatrix.h"
#include "Sequence.h"
#include "Debug.h"

#include <algorithm>

template <typename T>
struct SwipeLanes;

// 8 bit lanes are unsigned with scores biased by byteBias like SSW sw_sse2_byte, so they reach UINT8_MAX - byteBias
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
static const unsigned char SWIPE_ZERO_INDEX = 0x80;
// the band rows times width of the lanes of alignBanded with backtrace, the same on every ISA
static const size_t SWIPE_MAX_BAND_TRACE_CELLS = 1 << 20;
static const int SWIPE_MIN_BAND_WIDTH = 16;

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

// the composition bias of the query positions, rounded
static void queryBias(BaseMatrix *subMat, const Sequence *q, bool aaBiasCorrection, float aaBiasCorrectionScale, float *tmpBias, std::vector<int8_t> &bias) {
    bias.assign(q->L, 0);
    if (aaBiasCorrection) {
        SubstitutionMatrix::calcLocalAaBiasCorrection(subMat, q->numSequence, q->L, tmpBias, aaBiasCorrectionScale);
        for (int i = 0; i < q->L; i++) {
            bias[i] = static_cast<int8_t>((tmpBias[i] < 0.0) ? tmpBias[i] - 0.5 : tmpBias[i] + 0.5);
        }
    }
}

// a buffer of at least size vectors, without its old contents
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
    tmpBias = new float[maxSeqLen + 1];
    columnScores = malloc_simd_int(2 * alphabetSize * sizeof(simd_int));
    scoreLo8 = malloc_simd_int(alphabetSize * sizeof(simd_int));
    scoreHi8 = malloc_simd_int(alphabetSize * sizeof(simd_int));
    scoreLo16 = malloc_simd_int(alphabetSize * sizeof(simd_int));
    scoreHi16 = malloc_simd_int(alphabetSize * sizeof(simd_int));
    const int bytes = sizeof(simd_int);
    for (int a = 0; a < alphabetSize; a++) {
        for (int b = 0; b < alphabetSize; b++) {
            matrixBias = std::max(matrixBias, -static_cast<int>(subMat->subMatrix[a][b]));
        }
    }
    for (int a = 0; a < alphabetSize; a++) {
        unsigned char lo8[sizeof(simd_int)], hi8[sizeof(simd_int)], lo16[sizeof(simd_int)], hi16[sizeof(simd_int)];
        for (int p = 0; p < bytes; p++) {
            const int bLo = p % 16;
            const int bHi = 16 + p % 16;
            const int sLo = (bLo < alphabetSize) ? subMat->subMatrix[bLo][a] : 0;
            const int sHi = (bHi < alphabetSize) ? subMat->subMatrix[bHi][a] : 0;
            lo8[p] = static_cast<unsigned char>(sLo + matrixBias);
            hi8[p] = static_cast<unsigned char>(sHi + matrixBias);
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
    free(rowSub16);
    free(rowIndex);
    free(hColumn);
    free(eColumn);
    free(columnScores);
    free(ring);
    free(band);
    free(trace);
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
    queryBias(subMat, q, aaBiasCorrection, aaBiasCorrectionScale, tmpBias, bias);
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
void SwipeAligner::alignFull(const std::vector<size_t> &jobs, const unsigned char *const *targets, const int *targetLengths, s_align *aligns) {
    typedef SwipeLanes<T> V;
    const int lanes = V::COUNT;
    const int Lq = queryLength;
    const unsigned char *rowLetters = query.data();
    const simd_int *rowBias = (sizeof(T) == 1) ? rowBias8 : rowBias16;
    const simd_int *scoreLo = (sizeof(T) == 1) ? scoreLo8 : scoreLo16;
    const simd_int *scoreHi = (sizeof(T) == 1) ? scoreHi8 : scoreHi16;
    const simd_int vZero = simdi_setzero();
    // the 16 bit lanes subtract up to 192 from a score saturated at INT16_MAX, so they count as saturated 255 below it
    const int saturated = (sizeof(T) == 2) ? V::MAX - 255 : V::MAX - byteBias;
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
    size_t next = 0;
    int active = 0;

    const auto finish = [&](int l) {
        const size_t id = laneJob[l];
        s_align &a = aligns[id];
        a.score1 = laneBest[l];
        a.dbEndPos1 = (laneBest[l] == 0) ? -1 : laneBestColumn[l];
        a.qEndPos1 = (laneBest[l] == 0) ? -1 : laneBestRow[l];
        a.word = (laneBest[l] >= saturated) ? ((sizeof(T) == 1) ? 1 : 2) : ((sizeof(T) == 1) ? 0 : 1);
        laneJob[l] = -1;
        active--;
    };
    const auto assign = [&](int l) {
        while (next < jobs.size()) {
            const size_t id = jobs[next++];
            const int columns = targetLengths[id];
            laneJob[l] = static_cast<long>(id);
            laneColumn[l] = 0;
            laneColumns[l] = columns;
            laneBest[l] = 0;
            laneBestColumn[l] = 0;
            laneBestRow[l] = Lq - 1;
            active++;
            if (columns <= 0) {
                finish(l);
                continue;
            }
            for (int i = 0; i < Lq; i++) {
                setLaneValue<T>(hColumn + i, l, 0);
                setLaneValue<T>(eColumn + i, l, 0);
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
    const simd_int *rowSub = rowSub16;
    const simd_int vGapOpen = V::set(gapOpen);
    const simd_int vGapExtend = V::set(gapExtend);
    unsigned char letters[SWIPE_MAX_LANES];
    T colMax[2][SWIPE_MAX_LANES];
    T colPos[2][SWIPE_MAX_LANES];
    // the high row bits of queries longer than 65536 residues
    T colPosHigh[2][SWIPE_MAX_LANES];
    bool finished[SWIPE_MAX_LANES];
    while (active > 0) {
        for (int c = 0; c < columnCount; c++) {
            for (int l = 0; l < lanes; l++) {
                const long id = laneJob[l];
                const int column = laneColumn[l] + c;
                if (id < 0 || column >= laneColumns[l]) {
                    letters[l] = 0;
                } else {
                    letters[l] = targets[id][column];
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
        simd_int vColPosHigh[2] = {vZero, vZero};
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
                const simd_int hp0 = simdi16_max(simdui16_subs(simdi16_adds(vHd0, W0[rowLetters[i]]), rs), e);
                const simd_int h0 = simdi16_max(hp0, vF0);
                const simd_int e1 = simdi16_max(simdui16_subs(e, vGapExtend), simdui16_subs(h0, vGapOpen));
                vF0 = simdi16_max(simdui16_subs(vF0, vGapExtend), simdui16_subs(hp0, vGapOpen));
                const simd_int hp1 = simdi16_max(simdui16_subs(simdi16_adds(vHd1, W1[rowLetters[i]]), rs), e1);
                const simd_int h1 = simdi16_max(hp1, vF1);
                eColumn[i] = simdi16_max(simdui16_subs(e1, vGapExtend), simdui16_subs(h1, vGapOpen));
                vF1 = simdi16_max(simdui16_subs(vF1, vGapExtend), simdui16_subs(hp1, vGapOpen));
                vHd0 = hl;
                vHd1 = h0;
                hColumn[i] = h1;
                const simd_int gt0 = V::gt(h0, vColMax[0]);
                const simd_int gt1 = V::gt(h1, vColMax[1]);
                vColPos[0] = simdi8_blend(vColPos[0], rowIndex[i], gt0);
                vColMax[0] = V::max(vColMax[0], h0);
                vColPos[1] = simdi8_blend(vColPos[1], rowIndex[i], gt1);
                vColMax[1] = V::max(vColMax[1], h1);
                if (Lq > 65536) {
                    const simd_int high = V::set(i >> 16);
                    vColPosHigh[0] = simdi8_blend(vColPosHigh[0], high, gt0);
                    vColPosHigh[1] = simdi8_blend(vColPosHigh[1], high, gt1);
                }
            }
        } else {
            const simd_int *W = columnScores;
            const simd_int vBias = V::set(byteBias);
            simd_int vHd = vZero;
            simd_int vF = vZero;
            for (int i = 0; i < Lq; i++) {
                const simd_int s = V::addBias(W[rowLetters[i]], rowBias[i]);
                const simd_int e = eColumn[i];
                // H before F gives the next F (F - gap open is below F - gap extend), so F stays off the H chain
                const simd_int hp = V::max(V::addScore(vHd, s, vBias), e);
                const simd_int h = V::max(hp, vF);
                vHd = hColumn[i];
                hColumn[i] = h;
                const simd_int colMax = V::max(vColMax[0], h);
                vColPos[0] = simdi8_blend(rowIndex[i], vColPos[0], V::eq(colMax, vColMax[0]));
                vColMax[0] = colMax;
                eColumn[i] = V::max(V::subs(e, vGapExtend), V::subs(h, vGapOpen));
                vF = V::max(V::subs(vF, vGapExtend), V::subs(hp, vGapOpen));
            }
        }
        for (int c = 0; c < columnCount; c++) {
            simdi_storeu(reinterpret_cast<simd_int *>(colMax[c]), vColMax[c]);
            simdi_storeu(reinterpret_cast<simd_int *>(colPos[c]), vColPos[c]);
            if (sizeof(T) == 2 && Lq > 65536) {
                simdi_storeu(reinterpret_cast<simd_int *>(colPosHigh[c]), vColPosHigh[c]);
            }
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
                if (colMax[c][l] > laneBest[l]) {
                    laneBest[l] = colMax[c][l];
                    laneBestColumn[l] = laneColumn[l];
                    if (sizeof(T) == 2) {
                        laneBestRow[l] = static_cast<int>(static_cast<uint16_t>(colPos[c][l])
                            | (Lq > 65536 ? (static_cast<uint32_t>(static_cast<uint16_t>(colPosHigh[c][l])) << 16) : 0));
                    } else {
                        int row = static_cast<uint8_t>(colPos[c][l]);
                        while (row < Lq && laneValue<T>(hColumn + row, l) != colMax[c][l]) {
                            row += 256;
                        }
                        laneBestRow[l] = row;
                    }
                }
                laneColumn[l]++;
                if (laneColumn[l] == laneColumns[l] || laneBest[l] >= saturated) {
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
    // the longest targets go first so the short ones fill the lanes that become free
    const auto longestFirst = [targetLengths](size_t a, size_t b) {
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
void SwipeAligner::alignBand(const BandJob *jobs, size_t jobCount, const unsigned char *const *targets, const int *targetLengths,
                             s_align *aligns, std::string *backtraces) {
    typedef SwipeLanes<T> V;
    const int lanes = V::COUNT;
    const int width = jobs[0].width;
    int ringSize = 1;
    while (ringSize < width) {
        ringSize *= 2;
    }
    const int ringMask = ringSize - 1;
    reserveVectors(ring, ringCapacity, 3 * ringSize);
    simd_int *ringLo = ring;
    simd_int *ringHi = ring + ringSize;
    simd_int *ringValid = ring + 2 * ringSize;
    reserveVectors(band, bandCapacity, 2 * (width + 1));
    const unsigned char *rowLetters = query.data();
    const simd_int *rowBias = (sizeof(T) == 1) ? rowBias8 : rowBias16;
    const simd_int *scoreLo = (sizeof(T) == 1) ? scoreLo8 : scoreLo16;
    const simd_int *scoreHi = (sizeof(T) == 1) ? scoreHi8 : scoreHi16;
    const simd_int vZero = simdi_setzero();
    const simd_int vGapOpen = V::set(gapOpen);
    const simd_int vGapExtend = V::set(gapExtend);
    const simd_int vBias = V::set(byteBias);

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
    // the 4 bit trace codes of 2 (8 bit) or 4 (16 bit) neighbouring cells share a lane
    const int codesPerLane = 2 * sizeof(T);
    const int traceWidth = (width + codesPerLane - 1) / codesPerLane;
    reserveVectors(trace, traceCapacity, static_cast<size_t>(rowHi - rowLo) * traceWidth);
    simd_int *hRow = band;
    simd_int *fRow = band + width + 1;
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
        simd_int *traceRow = trace + static_cast<size_t>(r - rowLo) * traceWidth;
        simd_int codes = vZero;
        for (int k = 0; k < width; k++) {
            const int slot = (r + k) & ringMask;
            const simd_int s = V::addBias(laneScores(tLo, tHi, ringLo[slot], ringHi[slot]), rb);
            const simd_int diag = V::addScore(hRow[k], s, vBias);
            // H before E of the left cell gives E (E - gap open is below E - gap extend), so E stays off the H chain
            const simd_int eo = V::subs(vHl, vGapOpen);
            const simd_int ee = V::subs(vE, vGapExtend);
            vE = V::max(eo, ee);
            const simd_int fo = V::subs(hRow[k + 1], vGapOpen);
            const simd_int fe = V::subs(fRow[k + 1], vGapExtend);
            const simd_int f = V::max(fo, fe);
            const simd_int hp = simdi_and(V::max(diag, f), ringValid[slot]);
            const simd_int h = simdi_and(V::max(hp, vE), ringValid[slot]);
            // H from 0 (H is 0), 1 (diagonal), 2 (E) or 3 (F), 4 if E extends and 8 if F extends
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
        if (simd_any(V::gt(vRowMax, vBest))) {
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

    for (size_t l = 0; l < jobCount; l++) {
        s_align &a = aligns[jobs[l].id];
        std::string &backtrace = backtraces[jobs[l].id];
        backtrace.clear();
        a.score1 = bests[l];
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
        a.qEndPos1 = rows[l];
        a.dbEndPos1 = rows[l] + offset + columns[l];
        a.qStartPos1 = r;
        a.dbStartPos1 = r + offset + k;
        a.identicalAACnt = identities;
    }
}

void SwipeAligner::alignBanded(size_t count, const unsigned char *const *targets, const int *targetLengths, const int *diagonals,
                               s_align *aligns, std::string *backtraces, const int *columnEnds, const int *minScores) {
    const int Lq = queryLength;
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
        job.width = 2 * SWIPE_MIN_BAND_WIDTH + 1;
        job.offset = diagonals[i] - SWIPE_MIN_BAND_WIDTH;
        job.columnBegin = 0;
        job.columnEnd = columnEnds[i];
        job.rowBegin = std::max(0, job.columnBegin - job.offset - (job.width - 1));
        job.rowEnd = std::min(Lq, job.columnEnd - job.offset);
        bandJobs.push_back(job);
    }
    std::vector<BandJob> pending;
    while (bandJobs.empty() == false) {
        std::sort(bandJobs.begin(), bandJobs.end(), [aligns](const BandJob &a, const BandJob &b) {
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
        while (start < bandJobs.size()) {
            const int word = aligns[bandJobs[start].id].word;
            const int lanes = (word == 0) ? SwipeLanes<uint8_t>::COUNT : SwipeLanes<int16_t>::COUNT;
            size_t end = start + 1;
            while (end < bandJobs.size() && end - start < static_cast<size_t>(lanes) && bandJobs[end].width == bandJobs[start].width
                   && aligns[bandJobs[end].id].word == word) {
                end++;
            }
            const size_t traceCells = static_cast<size_t>(Lq + 1) * bandJobs[start].width;
            const bool canTrace = traceCells <= SWIPE_MAX_BAND_TRACE_CELLS;
            if (canTrace == false) {
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
                const int best = a.score1;
                const int maxScore = (word == 0) ? SwipeLanes<uint8_t>::MAX - byteBias : SwipeLanes<int16_t>::MAX;
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
                const bool grow = best < minScores[job.id];
                if (grow && fullBand == false) {
                    BandJob wider = job;
                    const int w2 = 2 * w;
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

// a lane stops after as many rows in a row below the X-drop as BlockAligner takes (two steps of eight)
static const int SWIPE_XDROP_ROWS = 16;
// an extension cell this far below the best score of its lane so far, or below 0 before it, is dead: a lane holds the scores above that
static const int SWIPE_FLOOR = 222;
// the target letters of a score row of SwipeExtender
static const int SWIPE_SCORE_ROW = 32;
// the widest band of an extension, its band columns are 16 bit lane values, or 8 bit up to UINT8_MAX columns
static const int SWIPE_MAX_WIDTH = 4096;
// the band cells of a lane in the trace of the last rows of SwipeExtender, the same on every ISA
static const size_t SWIPE_MAX_TRACE_CELLS = 4 * 1024 * 1024;

// a where the bits of mask are set and b elsewhere, whole lanes on x86; NEON selects without the sign spread of every byte of simdi8_blend
static inline simd_int laneSelect(simd_int mask, simd_int a, simd_int b) {
#if defined(SIMDE_ARM_NEON_A64V8_NATIVE) && !defined(AVX2)
    return vreinterpretq_s64_u8(vbslq_u8(vreinterpretq_u8_s64(mask), vreinterpretq_u8_s64(a), vreinterpretq_u8_s64(b)));
#else
    return simdi8_blend(b, a, mask);
#endif
}

// the lanes of SwipeExtender: unsigned scores above the dead level 0, and band columns that order with signed compares after adding COLUMN_BIAS
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
    // the scores stay far below INT16_MAX, so they add without saturation and order with signed compares
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

// codes << 4 plus the trace code of a cell: H from E 1, else from F 2, E opens 4, F opens 8; on NEON the first code has more bits that later shifts drop
template <typename T>
static inline simd_int addTraceCode(simd_int codes, simd_int fromE, simd_int fromF, simd_int openE, simd_int openF, bool first) {
    typedef ExtendLanes<T> V;
#if defined(SIMDE_ARM_NEON_A64V8_NATIVE) && !defined(AVX2)
    simd_int code = laneSelect(V::set(1), fromE, fromF);
    code = laneSelect(V::set(3), code, openE);
    code = laneSelect(V::set(7), code, openF);
    return first ? code : V::insertCode(codes, code);
#else
    // the masks are -1, so their sum bit by bit is minus the code
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
#define SWIPE_LOAD_LETTERS(p) _mm_loadu_si128(reinterpret_cast<const __m128i *>(p))
#define SWIPE_WIDEN(x) _mm256_cvtepi8_epi16(x)
#define SWIPE_LOAD_LANES(a, b) _mm256_inserti128_si256(_mm256_castsi128_si256(SWIPE_LOAD_LETTERS(a)), SWIPE_LOAD_LETTERS(b), 1)
#else
#define SWIPE_LOAD_LANES(a, b) _mm_loadu_si128(reinterpret_cast<const __m128i *>(a))
#define SWIPE_LOAD_LETTERS(p) _mm_loadl_epi64(reinterpret_cast<const __m128i *>(p))
#define SWIPE_WIDEN(x) _mm_cvtepi8_epi16(x)
#endif

// the 16 bit values of v[0-7] transposed within each 128 bit lane
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

// the 8 bit values of v[0-15] transposed within each 128 bit lane
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

// scores[c] holds the scores of band cell kb + c of the lanes: lane l scores letters[l][kb + c] with its score row rows[l] plus adds[l]
template <typename T>
static inline void laneCellScores(const unsigned char *const *letters, const int8_t *const *rows, const T *adds, int kb, simd_int *scores);

template <>
inline void laneCellScores<uint16_t>(const unsigned char *const *letters, const int8_t *const *rows, const uint16_t *adds, int kb, simd_int *scores) {
    const int lanes = ExtendLanes<uint16_t>::COUNT;
    for (int l = 0; l < lanes; l++) {
#if defined(SIMDE_ARM_NEON_A64V8_NATIVE) && !defined(AVX2)
        const int8x16x2_t row = {{vld1q_s8(rows[l]), vld1q_s8(rows[l] + 16)}};
        scores[l] = vreinterpretq_s64_s16(vaddq_s16(vmovl_s8(vqtbl2_s8(row, vld1_u8(letters[l] + kb))), vdupq_n_s16(adds[l])));
#else
        const __m128i index = SWIPE_LOAD_LETTERS(letters[l] + kb);
        const __m128i lo = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(rows[l])), index);
        const __m128i hi = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(rows[l] + 16)), index);
        // the shift moves bit 4 of the letters 16-31 to the bit 7 that the blend reads
        scores[l] = SWIPE_WIDEN(_mm_blendv_epi8(lo, hi, _mm_slli_epi16(index, 3)));
#endif
    }
    transposeLanes8(scores);
#if defined(AVX2)
    transposeLanes8(scores + 8);
    for (int c = 0; c < 8; c++) {
        const simd_int low = scores[c];
        scores[c] = _mm256_permute2x128_si256(low, scores[c + 8], 0x20);
        scores[c + 8] = _mm256_permute2x128_si256(low, scores[c + 8], 0x31);
    }
#endif
#if !defined(SIMDE_ARM_NEON_A64V8_NATIVE) || defined(AVX2)
    // x86 broadcasts on the shuffle port, so the lanes add their bias after the transpose
    const simd_int vAdds = simdi_loadu(reinterpret_cast<const simd_int *>(adds));
    for (int c = 0; c < lanes; c++) {
        scores[c] = simdi16_add(scores[c], vAdds);
    }
#endif
}

// 8 bit lanes take 16 band columns at once, with AVX2 lane l in the low and lane l + 16 in the high 128 bits of scores[l] before the transpose
template <>
inline void laneCellScores<uint8_t>(const unsigned char *const *letters, const int8_t *const *rows, const uint8_t *adds, int kb, simd_int *scores) {
#if defined(SIMDE_ARM_NEON_A64V8_NATIVE) && !defined(AVX2)
    for (int l = 0; l < 16; l++) {
        const int8x16x2_t row = {{vld1q_s8(rows[l]), vld1q_s8(rows[l] + 16)}};
        scores[l] = vreinterpretq_s64_u8(vaddq_u8(vreinterpretq_u8_s8(vqtbl2q_s8(row, vld1q_u8(letters[l] + kb))), vdupq_n_u8(adds[l])));
    }
    transposeLanes16(scores);
#else
    // AVX2 looks up lanes l and l + 16 in one shuffle
    for (int l = 0; l < 16; l++) {
        const simd_int index = SWIPE_LOAD_LANES(letters[l] + kb, letters[l + 16] + kb);
        const simd_int lo = simdi8_shuffle(SWIPE_LOAD_LANES(rows[l], rows[l + 16]), index);
        const simd_int hi = simdi8_shuffle(SWIPE_LOAD_LANES(rows[l] + 16, rows[l + 16] + 16), index);
        scores[l] = simdi8_blend(lo, hi, simdi16_slli(index, 3));
    }
    transposeLanes16(scores);
    const simd_int vAdds = simdi_loadu(reinterpret_cast<const simd_int *>(adds));
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
    : extendLaneRows(0), extendSlotRows(0), extendByteLaneRows(0), extendByteSlotRows(0), extendLongJobs(0), extendRerunJobs(0),
      subMat(subMat), aaBiasCorrection(aaBiasCorrection), aaBiasCorrectionScale(aaBiasCorrectionScale), gapOpen(gapOpen), gapExtend(gapExtend), matrixBias(0),
      band(NULL), bandCapacity(0), trace(NULL), traceCapacity(0), laneTargets(ExtendLanes<uint8_t>::COUNT), laneTraces(ExtendLanes<uint8_t>::COUNT) {
    const int alphabetSize = subMat->alphabetSize;
    if (alphabetSize > SWIPE_SCORE_ROW) {
        Debug(Debug::ERROR) << "SWIPE supports alphabets of at most 32 letters\n";
        EXIT(EXIT_FAILURE);
    }
    tmpBias = new float[maxSeqLen + 1];
    scoreRows = new int8_t[SWIPE_SCORE_ROW * SWIPE_SCORE_ROW];
    for (int a = 0; a < SWIPE_SCORE_ROW; a++) {
        for (int t = 0; t < SWIPE_SCORE_ROW; t++) {
            scoreRows[a * SWIPE_SCORE_ROW + t] = (a < alphabetSize && t < alphabetSize) ? static_cast<int8_t>(subMat->subMatrix[t][a]) : 0;
            matrixBias = std::max(matrixBias, -static_cast<int>(scoreRows[a * SWIPE_SCORE_ROW + t]));
        }
    }
}

SwipeExtender::~SwipeExtender() {
    delete[] tmpBias;
    delete[] scoreRows;
    free(band);
    free(trace);
}

size_t SwipeExtender::addQuery(const Sequence *q) {
    queryOffsets.push_back(queries.size());
    queryLengths.push_back(q->L);
    queries.insert(queries.end(), q->numSequence, q->numSequence + q->L);
    queries.insert(queries.end(), std::reverse_iterator<const unsigned char *>(q->numSequence + q->L), std::reverse_iterator<const unsigned char *>(q->numSequence));
    queryBias(subMat, q, aaBiasCorrection, aaBiasCorrectionScale, tmpBias, bias);
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

// a lane holds its scores less its floor, SWIPE_FLOOR below its best score, the same in 8 and 16 bit lanes; an 8 bit lane that saturates hands its job on
template <typename T>
void SwipeExtender::extendBand(const ExtendJob *jobs, size_t jobCount, bool backward, int xDrop, const unsigned char *const *targets,
                               const int *targetLengths, s_align *aligns, std::string *backtraces) {
    typedef ExtendLanes<T> V;
    const int lanes = V::COUNT;
    // the band columns of the cell scores taken at once
    const int block = (sizeof(T) == 1) ? 16 : lanes;
    const int width = jobs[0].left + jobs[0].right + 1;
    const int paddedWidth = (width + block - 1) / block * block;
    // the 4 bit trace codes of 2 or 4 neighbouring cells share a lane
    const int codesPerLane = 2 * sizeof(T);
    const int traceWidth = (width + codesPerLane - 1) / codesPerLane;
    // the trace of the last traceSteps rows of all lanes, a lane that needs more keeps its older rows in laneTraces
    const size_t traceSteps = SWIPE_MAX_TRACE_CELLS / (codesPerLane * traceWidth);
    reserveVectors(trace, traceCapacity, traceSteps * traceWidth);
    reserveVectors(band, bandCapacity, 2 * (width + 1));
    simd_int *hRow = band;
    simd_int *fRow = band + width + 1;
    const simd_int vZero = simdi_setzero();
    // the scores of 8 bit lanes stay below V::MAX, so larger penalties act as V::MAX
    const simd_int vGapO = V::set(std::min(gapOpen, static_cast<int>(V::MAX)));
    const simd_int vGapE = V::set(std::min(gapExtend, static_cast<int>(V::MAX)));
    const simd_int vXDrop = V::set(std::min(xDrop, static_cast<int>(V::MAX)));
    const simd_int vOne = V::set(1);
    const simd_int vTop = V::set(V::MAX);
    const simd_int vWidth = V::set(width + V::COLUMN_BIAS);
    const simd_int vFloor = V::set(SWIPE_FLOOR);
    const simd_int vDropRows = V::set(SWIPE_XDROP_ROWS - 1);
    for (int k = 0; k <= width; k++) {
        hRow[k] = vZero;
        fRow[k] = vZero;
    }
    zeroLetters.assign(paddedWidth, 0);

    // the lane state: job, target column of band column 0 minus the row, row, end row, band begin and end plus the row, query, bias and target rows, first step, best cell, floor and rise
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
    // the lanes of a row: active, mask begin in the first row and band end (plus COLUMN_BIAS), score bias above and below the score row, score row and target letters
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

    const auto load = [&](int l, const ExtendJob &job) {
        const int targetLength = targetLengths[job.id];
        const unsigned char *target = targets[job.id];
        std::vector<unsigned char> &laneTarget = laneTargets[l];
        laneTarget.assign(job.left + targetLength + paddedWidth, 0);
        if (backward) {
            std::reverse_copy(target, target + targetLength, laneTarget.begin() + job.left);
        } else {
            std::copy(target, target + targetLength, laneTarget.begin() + job.left);
        }
        const size_t position = queryOffsets[job.query] + (backward ? queryLengths[job.query] : 0);
        laneJobs[l] = &job;
        laneOffsets[l] = job.seedColumn - job.seedRow - job.left;
        laneRows[l] = job.seedRow;
        // the band leaves the target at the row columnEndRows, as the seed is in the target
        columnBeginRows[l] = job.seedRow + job.left;
        columnEndRows[l] = targetLength - laneOffsets[l];
        laneEnds[l] = std::min(queryLengths[job.query], columnEndRows[l]);
        laneQueries[l] = queries.data() + position;
        laneBiases[l] = biases.data() + position;
        laneLetters[l] = laneTarget.data() + job.seedColumn;
        laneStarts[l] = step;
        bestRows[l] = job.seedRow;
        bestColumns[l] = job.left;
        floors[l] = -SWIPE_FLOOR;
        rises[l] = 0;
        laneBests[l] = 0;
        laneDropped[l] = 0;
        edgeLefts[l] = 0;
        edgeRights[l] = 0;
        for (int k = 0; k <= width; k++) {
            setLaneValue<T>(hRow + k, l, 0);
            setLaneValue<T>(fRow + k, l, 0);
        }
        setLaneValue<T>(hRow + job.left, l, SWIPE_FLOOR);
        laneTraces[l].clear();
    };
    const auto finish = [&](int l) {
        const ExtendJob &job = *laneJobs[l];
        laneJobs[l] = NULL;
        if (edgeLefts[l] != 0 || edgeRights[l] != 0) {
            ExtendJob wider = job;
            wider.left *= (edgeLefts[l] != 0) ? 2 : 1;
            wider.right *= (edgeRights[l] != 0) ? 2 : 1;
            wider.wide = job.wide || wider.left + wider.right + 1 > UINT8_MAX;
            // a band grows up to SWIPE_MAX_WIDTH columns, the extension of a band that would pass it keeps its result
            if (wider.left + wider.right + 1 <= SWIPE_MAX_WIDTH) {
                extendPending.push_back(wider);
                return;
            }
        }
        s_align &a = aligns[job.id];
        const int row = bestRows[l];
        const int column = row + laneOffsets[l] + bestColumns[l];
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
        const auto traceCode = [&](int i, int k) {
            const size_t rowStep = static_cast<size_t>(i - job.seedRow);
            const int laneCodes = (rowStep * traceWidth < laneTraces[l].size()) ? laneTraces[l][rowStep * traceWidth + k / codesPerLane]
                                  : laneValue<T>(trace + (laneStarts[l] + rowStep) % traceSteps * traceWidth + k / codesPerLane, l);
            return (laneCodes >> (4 * (codesPerLane - 1 - k % codesPerLane))) & 15;
        };
        std::string &backtrace = backtraces[job.id];
        backtrace.clear();
        int i = row;
        int k = bestColumns[l];
        int state = 0;
        while (true) {
            const int code = traceCode(i, k);
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
        // a backward extension walks from its start to the seed, which is already the alignment order
        if (backward == false) {
            std::reverse(backtrace.begin(), backtrace.end());
        }
    };
    T rowMax[SWIPE_MAX_LANES];
    T rowPos[SWIPE_MAX_LANES];
    T laneFlags[SWIPE_MAX_LANES];
    T overflows[SWIPE_MAX_LANES];
    simd_int scores[SWIPE_MAX_LANES];
    while (true) {
        // a free lane takes the next job; the band columns of any active lane, the others are not read, and the ones of all lanes, which skip the mask
        int activeCount = 0;
        int columnBegin = width;
        int columnEnd = 0;
        int validBegin = 0;
        int validEnd = width;
        for (int l = 0; l < lanes; l++) {
            while (true) {
                if (laneJobs[l] == NULL) {
                    if (next == jobCount) {
                        break;
                    }
                    load(l, jobs[next++]);
                }
                const int i = laneRows[l];
                if (i >= laneEnds[l]) {
                    finish(l);
                    continue;
                }
                const int lo = std::max(0, columnBeginRows[l] - i);
                const int hi = std::min(width, columnEndRows[l] - i);
                const int maskBegin = (laneStarts[l] == step) ? lo : 0;
                const int bias = laneBiases[l][i];
                activeCount++;
                columnBegin = std::min(columnBegin, lo);
                columnEnd = std::max(columnEnd, hi);
                validBegin = std::max(validBegin, maskBegin);
                validEnd = std::min(validEnd, hi);
                active[l] = static_cast<T>(-1);
                maskBegins[l] = static_cast<T>(maskBegin + V::COLUMN_BIAS);
                columnEnds[l] = static_cast<T>(hi + V::COLUMN_BIAS);
                scoreAdds[l] = matrixBias + std::max(0, bias);
                scoreSubs[l] = matrixBias + std::max(0, -bias);
                rows[l] = scoreRows + SWIPE_SCORE_ROW * laneQueries[l][i];
                letters[l] = laneLetters[l] + (step - laneStarts[l]);
                break;
            }
            // a free lane runs on garbage that only it reads and the next job resets
            if (laneJobs[l] == NULL) {
                active[l] = 0;
                maskBegins[l] = static_cast<T>(V::COLUMN_BIAS);
                columnEnds[l] = static_cast<T>(width + V::COLUMN_BIAS);
                scoreAdds[l] = 0;
                scoreSubs[l] = 0;
                rows[l] = scoreRows;
                letters[l] = zeroLetters.data();
            }
        }
        if (activeCount == 0) {
            break;
        }
        extendLaneRows += activeCount;
        extendSlotRows += lanes;
        if (sizeof(T) == 1) {
            extendByteLaneRows += activeCount;
            extendByteSlotRows += lanes;
        }
        const simd_int vActive = simdi_loadu(reinterpret_cast<const simd_int *>(active));
        const simd_int vMaskBegin = simdi_loadu(reinterpret_cast<const simd_int *>(maskBegins));
        const simd_int vColumnEnd = simdi_loadu(reinterpret_cast<const simd_int *>(columnEnds));
        // the rows before are above the floor of this row by the rise
        const simd_int vRise = simdi_loadu(reinterpret_cast<const simd_int *>(rises));
        const simd_int vScoreSub = V::add(simdi_loadu(reinterpret_cast<const simd_int *>(scoreSubs)), vRise);
        const simd_int vGapOpenRise = V::adds(vGapO, vRise);
        const simd_int vGapExtendRise = V::adds(vGapE, vRise);
        simd_int vBest = simdi_loadu(reinterpret_cast<const simd_int *>(laneBests));
        simd_int vDropped = simdi_loadu(reinterpret_cast<const simd_int *>(laneDropped));
        simd_int vEdgeLeft = simdi_loadu(reinterpret_cast<const simd_int *>(edgeLefts));
        simd_int vEdgeRight = simdi_loadu(reinterpret_cast<const simd_int *>(edgeRights));
        simd_int *traceRow = trace + step % traceSteps * traceWidth;
        // a lane keeps its row of the trace that this row overwrites
        for (int l = 0; l < lanes; l++) {
            if (laneJobs[l] != NULL && step - laneStarts[l] >= traceSteps) {
                extendLongJobs += (step - laneStarts[l] == traceSteps);
                std::vector<uint16_t> &laneTrace = laneTraces[l];
                for (int w = 0; w < traceWidth; w++) {
                    laneTrace.push_back(laneValue<T>(traceRow + w, l));
                }
            }
        }
        simd_int codes = vZero;
        simd_int vHl = vZero;
        simd_int vE = vZero;
        simd_int vRowMax = vZero;
        int kb = columnBegin / block * block;
        simd_int vPos = vZero;
        simd_int vK = V::set(kb + V::COLUMN_BIAS);
        simd_int hDiag = hRow[kb];
        const auto cell = [&](int k, simd_int score, bool masked, bool first) {
            const simd_int hAbove = hRow[k + 1];
            // a dead cell of the row before starts no diagonal
            const simd_int diag = V::keepAbove(V::subs(V::adds(hDiag, score), vScoreSub), hDiag, vRise);
            // H before E of the left cell gives E (E - gap open is below E - gap extend), so E stays off the H chain
            const simd_int eo = V::subs(vHl, vGapO);
            const simd_int ee = V::subs(vE, vGapE);
            vE = V::max(eo, ee);
            const simd_int fo = V::subs(hAbove, vGapOpenRise);
            const simd_int fe = V::subs(fRow[k + 1], vGapExtendRise);
            const simd_int f = V::max(fo, fe);
            simd_int hp = V::max(diag, f);
            simd_int h = V::max(hp, vE);
            if (masked) {
                const simd_int valid = simdi_andnot(V::columnGt(vMaskBegin, vK), V::columnGt(vColumnEnd, vK));
                hp = simdi_and(hp, valid);
                h = simdi_and(h, valid);
                fRow[k] = simdi_and(f, valid);
            } else {
                fRow[k] = f;
            }
            // ties go to the gap and to its opening, as in BlockAligner
            codes = addTraceCode<T>(codes, V::eq(h, vE), V::eq(h, f), V::eq(vE, eo), V::eq(f, fo), first);
            hRow[k] = h;
            hDiag = hAbove;
            vHl = hp;
            const simd_int cellMax = V::max(vRowMax, h);
            vPos = laneSelect(V::eq(cellMax, h), vK, vPos);
            vRowMax = cellMax;
            vK = V::add(vK, vOne);
        };
        // the trace codes of 2 or 4 cells go out together
        for (; kb < columnEnd && kb + block <= width; kb += block) {
            laneCellScores<T>(letters, rows, scoreAdds, kb, scores);
            const bool masked = (kb < validBegin || kb + block > validEnd);
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
        // ties go to the larger target position, then the later row, so an extension ends at the farthest cell of its best score; BlockAligner does so only within the step that raised it
        if (simd_any(simdi_andnot(V::gt(vBest, vRowMax), vActive))) {
            simdi_storeu(reinterpret_cast<simd_int *>(rowMax), vRowMax);
            simdi_storeu(reinterpret_cast<simd_int *>(rowPos), vPos);
            for (int l = 0; l < lanes; l++) {
                if (active[l] == 0 || rowMax[l] < laneBests[l]) {
                    continue;
                }
                const int pos = static_cast<T>(rowPos[l] - V::COLUMN_BIAS);
                if (rowMax[l] > laneBests[l] || laneRows[l] + pos >= bestRows[l] + bestColumns[l]) {
                    bestRows[l] = laneRows[l];
                    bestColumns[l] = pos;
                }
            }
            vBest = V::max(vBest, vRowMax);
        }
        // a band side grows if its edge is within the X-drop of the best score; the last column counts only where the band ends, the others keep an earlier row
        const simd_int vAlive = V::max(V::subs(vBest, vXDrop), vOne);
        vEdgeLeft = simdi_or(vEdgeLeft, simdi_andnot(V::gt(vAlive, hRow[0]), vActive));
        vEdgeRight = simdi_or(vEdgeRight, simdi_andnot(V::gt(vAlive, hRow[width - 1]), simdi_and(V::eq(vColumnEnd, vWidth), vActive)));
        vDropped = simdi_and(V::add(vDropped, vOne), simdi_and(V::gt(vAlive, vRowMax), vActive));
        // a lane also stops once both sides of its band grow
        const simd_int stop = simdi_or(V::columnGt(vDropped, vDropRows), simdi_and(vEdgeLeft, vEdgeRight));
        // an 8 bit lane whose best cell of the row reached the top in the sum of the cell scores may have saturated
        const simd_int overflow = (sizeof(T) == 1) ? simdi_and(V::eq(V::adds(vRowMax, vScoreSub), vTop), vActive) : vZero;
        // the floor follows the best score, the next row reads this row less the rise
        const simd_int vNextRise = V::subs(vBest, vFloor);
        simdi_storeu(reinterpret_cast<simd_int *>(laneBests), V::subs(vBest, vNextRise));
        simdi_storeu(reinterpret_cast<simd_int *>(rises), vNextRise);
        simdi_storeu(reinterpret_cast<simd_int *>(laneDropped), vDropped);
        simdi_storeu(reinterpret_cast<simd_int *>(edgeLefts), vEdgeLeft);
        simdi_storeu(reinterpret_cast<simd_int *>(edgeRights), vEdgeRight);
        simdi_storeu(reinterpret_cast<simd_int *>(laneFlags), stop);
        simdi_storeu(reinterpret_cast<simd_int *>(overflows), overflow);
        for (int l = 0; l < lanes; l++) {
            if (active[l] == 0) {
                continue;
            }
            floors[l] += rises[l];
            if (overflows[l] != 0) {
                ExtendJob rerun = *laneJobs[l];
                rerun.wide = true;
                extendPending.push_back(rerun);
                laneJobs[l] = NULL;
                extendRerunJobs++;
            } else if (laneFlags[l] != 0) {
                finish(l);
            } else {
                laneRows[l]++;
            }
        }
        step++;
    }
}

void SwipeExtender::extendBanded(size_t count, const size_t *queryIds, const unsigned char *const *targets, const int *targetLengths, const int *seedQueries,
                                 const int *seedTargets, bool backward, int xDrop, int bandSide, s_align *aligns, std::string *backtraces) {
    extendJobs.clear();
    for (size_t i = 0; i < count; i++) {
        const int queryLength = queryLengths[queryIds[i]];
        ExtendJob job;
        job.id = i;
        job.query = queryIds[i];
        job.seedRow = backward ? queryLength - 1 - seedQueries[i] : seedQueries[i];
        job.seedColumn = backward ? targetLengths[i] - 1 - seedTargets[i] : seedTargets[i];
        job.left = bandSide;
        job.right = bandSide;
        job.wide = 2 * bandSide + 1 > UINT8_MAX;
        extendJobs.push_back(job);
    }
    while (extendJobs.empty() == false) {
        // the lanes of a band width take the jobs with the most query rows first, 8 bit lanes before 16 bit
        std::sort(extendJobs.begin(), extendJobs.end(), [this](const ExtendJob &a, const ExtendJob &b) {
            if (a.left + a.right != b.left + b.right) {
                return a.left + a.right < b.left + b.right;
            }
            if (a.wide != b.wide) {
                return b.wide;
            }
            const int rowsA = queryLengths[a.query] - a.seedRow;
            const int rowsB = queryLengths[b.query] - b.seedRow;
            if (rowsA != rowsB) {
                return rowsA > rowsB;
            }
            return a.id < b.id;
        });
        extendPending.clear();
        for (size_t start = 0; start < extendJobs.size(); ) {
            const int width = extendJobs[start].left + extendJobs[start].right;
            const bool wide = extendJobs[start].wide;
            size_t end = start + 1;
            while (end < extendJobs.size() && extendJobs[end].left + extendJobs[end].right == width && extendJobs[end].wide == wide) {
                end++;
            }
            if (wide) {
                extendBand<uint16_t>(&extendJobs[start], end - start, backward, xDrop, targets, targetLengths, aligns, backtraces);
            } else {
                extendBand<uint8_t>(&extendJobs[start], end - start, backward, xDrop, targets, targetLengths, aligns, backtraces);
            }
            start = end;
        }
        extendJobs.swap(extendPending);
    }
}
