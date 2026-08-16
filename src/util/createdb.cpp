/*
 * createdb
 * written by Martin Steinegger <martin.steinegger@snu.ac.kr>.
 * modified by Maria Hauser <mhauser@genzentrum.lmu.de> (splitting into sequences/headers databases)
 * modified by Milot Mirdita <milot@mirdita.de>
 */
#include "FileUtil.h"
#include "DBWriter.h"
#include "Debug.h"
#include "Util.h"
#include "KSeqWrapper.h"
#include "itoa.h"
#include "FastSort.h"
#include "Masker.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstring>
#include <fcntl.h>
#include <sys/resource.h>
#include <unistd.h>

#ifdef OPENMP
#include <omp.h>
#endif

// both get the per split length sort and the joint merge; only mode 2 has the GPU layout
static bool isSortedCreatedbMode(int mode) {
    return mode == Parameters::SEQUENCE_SPLIT_MODE_GPU
        || mode == Parameters::SEQUENCE_SPLIT_MODE_LENGTH_DESC;
}

// (file, rank) sorts like the sequential id would, so parallel parsers need no global entry count
static const size_t CREATEDB_MIN_SPLIT_BYTES = 1ull * 1024 * 1024 * 1024;

static const int CREATEDB_KEY_FILE_SHIFT = 40;

static DBKeyType packInputKey(size_t fileIdx, size_t rankInFile) {
    return static_cast<DBKeyType>((static_cast<uint64_t>(fileIdx) << CREATEDB_KEY_FILE_SHIFT) | rankInFile);
}

static unsigned int unpackInputFile(DBKeyType key) {
    return static_cast<unsigned int>(static_cast<uint64_t>(key) >> CREATEDB_KEY_FILE_SHIFT);
}

struct IndexOffset {
    bool operator()(const DBReader<DBKeyType>::Index &first, const DBReader<DBKeyType>::Index &second) const {
        return DBReader<DBKeyType>::Index::compareByOffset(first, second);
    }
};

// .id holds the entry position while sorting, so the key comes from the still position ordered header index
struct IndexLengthByKey {
    const DBReader<DBKeyType>::Index *keySource;
    bool descending;
    IndexLengthByKey(const DBReader<DBKeyType>::Index *keySource, bool descending)
        : keySource(keySource), descending(descending) {}
    bool operator()(const DBReader<DBKeyType>::Index &first, const DBReader<DBKeyType>::Index &second) const {
        if (first.length != second.length) {
            return descending ? first.length > second.length : first.length < second.length;
        }
        return keySource[first.id].id < keySource[second.id].id;
    }
};

static const size_t SORT_IO_RANGE_BYTES = 8 * 1024 * 1024;
static const size_t SORT_STAGE_BYTES = 32 * 1024 * 1024;

static void preadFully(int fd, char *dst, size_t bytes, size_t offset, const char *what) {
    size_t done = 0;
    while (done < bytes) {
        const ssize_t got = pread(fd, dst + done, bytes - done, static_cast<off_t>(offset + done));
        if (got < 0 && errno == EINTR) {
            continue;
        }
        if (got <= 0) {
            Debug(Debug::ERROR) << "Short read of " << bytes << " bytes at " << offset << " from " << what << "\n";
            EXIT(EXIT_FAILURE);
        }
        done += static_cast<size_t>(got);
    }
}

static void writeFully(FILE *file, const char *src, size_t bytes, const char *what) {
    if (bytes != 0 && fwrite(src, 1, bytes, file) != bytes) {
        Debug(Debug::ERROR) << "Can not write to " << what << "\n";
        EXIT(EXIT_FAILURE);
    }
}

static void pwriteFully(int fd, const char *src, size_t bytes, size_t offset, const char *what) {
    size_t done = 0;
    while (done < bytes) {
        const ssize_t wrote = pwrite(fd, src + done, bytes - done, static_cast<off_t>(offset + done));
        if (wrote < 0 && errno == EINTR) {
            continue;
        }
        if (wrote <= 0) {
            Debug(Debug::ERROR) << "Can not write to " << what << "\n";
            EXIT(EXIT_FAILURE);
        }
        done += static_cast<size_t>(wrote);
    }
}

// one split is read whole into one buffer, so a lone fread stream leaves the device at queue depth 1
static void readWholeFileParallel(const char *path, char *dst, size_t bytes, unsigned int threads) {
    if (bytes == 0) {
        return;
    }
    const int fd = open(path, O_RDONLY);
    if (fd < 0) {
        Debug(Debug::ERROR) << "Cannot open " << path << " for reading\n";
        EXIT(EXIT_FAILURE);
    }
    const size_t ranges = std::max<size_t>(1, std::min<size_t>(std::max<unsigned int>(1, threads),
                                                               (bytes + SORT_IO_RANGE_BYTES - 1) / SORT_IO_RANGE_BYTES));
#pragma omp parallel for schedule(static) num_threads((int) ranges)
    for (size_t r = 0; r < ranges; r++) {
        const size_t begin = bytes / ranges * r + std::min<size_t>(bytes % ranges, r);
        const size_t end = bytes / ranges * (r + 1) + std::min<size_t>(bytes % ranges, r + 1);
        preadFully(fd, dst + begin, end - begin, begin, path);
    }
    if (close(fd) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << path << "\n";
        EXIT(EXIT_FAILURE);
    }
}

// Emit entries in sorted order without the serial fwrite: cut the entries into contiguous byte
// bounded slices, prefix sum the slice sizes, then let every worker gather its own slice and pwrite
// it where the prefix sum put it. Same bytes as the serial loop, only the staging is bounded.
template <typename Entry>
static void writeSortedParallel(const char *path, size_t entries, unsigned int threads, Entry entry) {
    const int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (fd < 0) {
        Debug(Debug::ERROR) << "Cannot open " << path << " for writing\n";
        EXIT(EXIT_FAILURE);
    }
    const size_t workers = std::max<unsigned int>(1, threads);
    const size_t stageBytes = std::max<size_t>(1024 * 1024,
                                               std::min<size_t>(SORT_STAGE_BYTES,
                                                                Util::computeMemory(0) / 64 / workers));
    std::vector<std::vector<char> > stage(workers);
    std::vector<size_t> begin(workers + 1, 0), base(workers, 0), sliceBytes(workers, 0);
    size_t at = 0, offset = 0;
    while (at < entries) {
        size_t active = 0, cursor = at;
        begin[0] = at;
        while (active < workers && cursor < entries) {
            size_t bytes = 0;
            // the first entry always fits, so one longer than the cap still makes progress
            while (cursor < entries && (bytes == 0 || bytes + entry.length(cursor) <= stageBytes)) {
                bytes += entry.length(cursor);
                cursor++;
            }
            base[active] = offset;
            sliceBytes[active] = bytes;
            offset += bytes;
            begin[++active] = cursor;
        }
#pragma omp parallel for schedule(static) num_threads((int) active)
        for (size_t w = 0; w < active; w++) {
            if (stage[w].size() < sliceBytes[w]) {
                stage[w].resize(sliceBytes[w]);
            }
            size_t pos = 0, out = base[w];
            for (size_t i = begin[w]; i < begin[w + 1]; i++) {
                const size_t len = entry.length(i);
                memcpy(stage[w].data() + pos, entry.data(i), len);
                // after the copy: place() may overwrite the very fields length()/data() read
                entry.place(i, out);
                pos += len;
                out += len;
            }
            pwriteFully(fd, stage[w].data(), pos, base[w], path);
        }
        at = begin[active];
    }
    if (close(fd) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << path << "\n";
        EXIT(EXIT_FAILURE);
    }
}

// the merged output would otherwise stay dirty in page cache until the process exits, which at this
// scale competes with the index for memory. Start writeback on the round just written and release
// the one before it, so the cache footprint stays flat whatever the db size.
static void retireRound(int fd, size_t base, size_t bytes, size_t prevBase, size_t prevBytes) {
#if defined(__linux__)
    if (bytes != 0) {
        sync_file_range(fd, static_cast<off_t>(base), static_cast<off_t>(bytes), SYNC_FILE_RANGE_WRITE);
    }
    if (prevBytes != 0) {
        sync_file_range(fd, static_cast<off_t>(prevBase), static_cast<off_t>(prevBytes),
                        SYNC_FILE_RANGE_WAIT_BEFORE | SYNC_FILE_RANGE_WRITE | SYNC_FILE_RANGE_WAIT_AFTER);
        posix_fadvise(fd, static_cast<off_t>(prevBase), static_cast<off_t>(prevBytes), POSIX_FADV_DONTNEED);
    }
#else
    (void) fd; (void) base; (void) bytes; (void) prevBase; (void) prevBytes;
#endif
}

// ips4o inlines a functor but not a function pointer, so the direction is a type, not an argument
template <typename Comp>
static void sortIndexRange(DBReader<DBKeyType>::Index *index, size_t size, Comp comp) {
    SORT_PARALLEL(index, index + size, comp);
}

// Sort the data file in-place using your index array
int sortWithIndex(const char *dataFileSeq,
                  const char *indexFileSeq,
                  const char *dataFileHeader,
                  const char *indexFileHeader,
                  unsigned int threads,
                  bool descending)
{
    DBReader<DBKeyType> reader(dataFileSeq, indexFileSeq, 1, DBReader<DBKeyType>::USE_INDEX);
    reader.open(DBReader<DBKeyType>::HARDNOSORT);
    DBReader<DBKeyType>::Index *index = reader.getIndex();
    struct stat st;
    if (stat(dataFileSeq, &st) != 0) {
        Debug(Debug::ERROR) << "stat failed: " << dataFileSeq << "\n";
        EXIT(EXIT_FAILURE);
    }
    const size_t seqfileSize = st.st_size;

    char *buf = new(std::nothrow) char[seqfileSize];
    Util::checkAllocation(buf, "Can not allocate sort buffer in createdb; raise --shuffle-splits");
    readWholeFileParallel(dataFileSeq, buf, seqfileSize, threads);
    // the writer emitted both files entry by entry, so position i is the same entry in either index
    DBReader<DBKeyType> header(dataFileHeader, indexFileHeader, 1, DBReader<DBKeyType>::USE_INDEX);
    header.open(DBReader<DBKeyType>::HARDNOSORT);
    DBReader<DBKeyType>::Index *headerIndex = header.getIndex();
    if (header.getSize() != reader.getSize()) {
        Debug(Debug::ERROR) << "Split has " << reader.getSize() << " sequences but "
                            << header.getSize() << " headers\n";
        EXIT(EXIT_FAILURE);
    }
    // needed to keep the information in what line the id was originally
    for (size_t i = 0; i < reader.getSize(); i++) {
        index[i].id = i;
    }

    sortIndexRange(index, reader.getSize(), IndexLengthByKey(headerIndex, descending));

    struct SeqEntry {
        DBReader<DBKeyType>::Index *index;
        const char *buf;
        size_t length(size_t i) const { return index[i].length; }
        const char *data(size_t i) const { return buf + index[i].offset; }
        void place(size_t i, size_t offset) { index[i].offset = offset; }
    };
    SeqEntry seqEntry = { index, buf };
    writeSortedParallel(dataFileSeq, reader.getSize(), threads, seqEntry);

    if (stat(dataFileHeader, &st) != 0) {
        Debug(Debug::ERROR) << "stat failed: " << dataFileHeader << "\n";
        EXIT(EXIT_FAILURE);
    }
    const size_t headFileSize = st.st_size;
    if(headFileSize > seqfileSize){
        delete [] buf;
        buf = new(std::nothrow) char[headFileSize];
        Util::checkAllocation(buf, "Can not allocate sort buffer in createdb; raise --shuffle-splits");
    }
    readWholeFileParallel(dataFileHeader, buf, headFileSize, threads);

    // index[i].id is a permutation of the original positions, so every sortedId is touched once
    struct HdrEntry {
        DBReader<DBKeyType>::Index *index;
        DBReader<DBKeyType>::Index *headerIndex;
        const char *buf;
        size_t length(size_t i) const { return headerIndex[index[i].id].length; }
        const char *data(size_t i) const { return buf + headerIndex[index[i].id].offset; }
        void place(size_t i, size_t offset) {
            const size_t sortedId = static_cast<size_t>(index[i].id);
            headerIndex[sortedId].offset = offset;
            // reconstruct old id
            index[i].id = headerIndex[sortedId].id;
        }
    };
    HdrEntry hdrEntry = { index, headerIndex, buf };
    writeSortedParallel(dataFileHeader, header.getSize(), threads, hdrEntry);
    delete [] buf;

    sortIndexRange(index, reader.getSize(), IndexOffset());
    // the merge pairs the two indexes by position, so both are published in physical order
    sortIndexRange(headerIndex, header.getSize(), IndexOffset());
    {
        std::string tmpIndex = std::string(indexFileSeq) + ".tmp";
        FILE *indexout = FileUtil::openFileOrDie(tmpIndex.c_str(), "wb", false);
        DBWriter::writeIndex(indexout, reader.getSize(), index, threads);
        fclose(indexout);
        FileUtil::move(tmpIndex.c_str(), indexFileSeq);

        std::string tmpHeaderIndex = std::string(indexFileHeader) + ".tmp";
        FILE *headerIndexOut = FileUtil::openFileOrDie(tmpHeaderIndex.c_str(), "wb", false);
        DBWriter::writeIndex(headerIndexOut, header.getSize(), headerIndex, threads);
        fclose(headerIndexOut);
        FileUtil::move(tmpHeaderIndex.c_str(), indexFileHeader);
    }

    reader.close();
    header.close();
    return 0;
}

int mergeSequentialByJointIndex(
        char ** dataFiles,
        char ** indexFiles,
        char ** dataFilesHeader,
        char ** indexFilesHeader,
        char * outDataFile,
        char * outIndexFile,
        char * outHeaderDataFile,
        char * outHeaderIndexFile,
        const char * outLookupFile,
        std::vector<unsigned int>* sourceLookup,
        const std::vector<DBKeyType>* fileKeyStarts,
        size_t totalEntries,
        size_t shuffleSplits,
        bool descending,
        bool gpuLayout
) {
    struct JointEntry {
        unsigned int fileIdx;
        DBKeyType id;
        unsigned int length;
        unsigned int hdrLength;
        JointEntry() = default;
        JointEntry(unsigned int fileIdx, DBKeyType id, unsigned int length, unsigned int hdrLength)
            : fileIdx(fileIdx), id(id), length(length), hdrLength(hdrLength) {}

        struct Asc {
            bool operator()(const JointEntry &first, const JointEntry &second) const {
                if (first.length != second.length) {
                    return first.length < second.length;
                }
                return first.id < second.id;
            }
        };
        struct Desc {
            bool operator()(const JointEntry &first, const JointEntry &second) const {
                if (first.length != second.length) {
                    return first.length > second.length;
                }
                return first.id < second.id;
            }
        };
    };

    std::vector<JointEntry> joint(totalEntries);
    size_t maxLen = 0;
    size_t maxHdrLen = 0;
    // the sort below is a total order on (length, id), so the fill order does not reach the output
    std::atomic<size_t> jointCursor(0);
    // both index files are opened per split now, so the per split cost is twice one index
    const size_t indexBytesPerSplit = totalEntries / std::max<size_t>(shuffleSplits, 1)
                                      * sizeof(DBReader<DBKeyType>::Index) * 2;
    const size_t fillThreads = std::max<size_t>(1, std::min<size_t>(
        std::min<size_t>(shuffleSplits, omp_get_max_threads()),
        indexBytesPerSplit ? Util::computeMemory(0) / 8 / indexBytesPerSplit : shuffleSplits));
#pragma omp parallel for schedule(dynamic, 1) num_threads(fillThreads) reduction(max: maxLen) reduction(max: maxHdrLen)
    for (size_t i = 0; i < shuffleSplits; i++) {
        DBReader<DBKeyType> reader(
                dataFiles[i],
                indexFiles[i],
                1,
                DBReader<DBKeyType>::USE_INDEX
        );
        reader.open(DBReader<DBKeyType>::HARDNOSORT);
        DBReader<DBKeyType>::Index* index = reader.getIndex();
        // sortWithIndex published both index files in physical order, so position j is the same entry
        DBReader<DBKeyType> headerReader(
                dataFilesHeader[i],
                indexFilesHeader[i],
                1,
                DBReader<DBKeyType>::USE_INDEX
        );
        headerReader.open(DBReader<DBKeyType>::HARDNOSORT);
        DBReader<DBKeyType>::Index* headerIndex = headerReader.getIndex();
        if (headerReader.getSize() != reader.getSize()) {
            Debug(Debug::ERROR) << "Split " << i << " has " << reader.getSize() << " sequences but "
                                << headerReader.getSize() << " headers\n";
            EXIT(EXIT_FAILURE);
        }
        const size_t at = jointCursor.fetch_add(reader.getSize(), std::memory_order_relaxed);
        for(size_t j = 0; j < reader.getSize(); j++){
            joint[at + j] = JointEntry((unsigned int)i, index[j].id, index[j].length,
                                       headerIndex[j].length);
            maxLen = std::max(maxLen, static_cast<size_t>(index[j].length));
            maxHdrLen = std::max(maxHdrLen, static_cast<size_t>(headerIndex[j].length));
        }
        headerReader.close();
        reader.close();
    }
    if (jointCursor.load() != totalEntries) {
        Debug(Debug::ERROR) << "The splits hold " << jointCursor.load() << " entries, expected "
                            << totalEntries << "\n";
        EXIT(EXIT_FAILURE);
    }

    if (descending) {
        SORT_PARALLEL(joint.begin(), joint.end(), JointEntry::Desc());
    } else {
        SORT_PARALLEL(joint.begin(), joint.end(), JointEntry::Asc());
    }

    // one descriptor per split, read with pread so the workers can share it without a seek race
    std::vector<int> inFdSeq(shuffleSplits, -1);
    std::vector<int> inFdHeader(shuffleSplits, -1);
    for (size_t i = 0; i < shuffleSplits; i++) {
        inFdSeq[i] = open(dataFiles[i], O_RDONLY);
        inFdHeader[i] = open(dataFilesHeader[i], O_RDONLY);
        if (inFdSeq[i] < 0 || inFdHeader[i] < 0) {
            Debug(Debug::ERROR) << "Cannot open split " << i << " for reading\n";
            EXIT(EXIT_FAILURE);
        }
    }

    const char *outNames[5] = {outDataFile, outHeaderDataFile, outLookupFile, outIndexFile, outHeaderIndexFile};
    int outFds[5];
    for (int i = 0; i < 5; i++) {
        // O_TRUNC alone follows a symlink and overwrites its target, and mode 1 leaves the db as one
        unlink(outNames[i]);
        outFds[i] = open(outNames[i], O_WRONLY | O_CREAT | O_TRUNC, 0666);
        if (outFds[i] < 0) {
            Debug(Debug::ERROR) << "Cannot open " << outNames[i] << " for writing\n";
            EXIT(EXIT_FAILURE);
        }
    }

    // every offset is a prefix sum over the round and the streams flush in worker order, so the output is serial identical
    const int ALIGN = 4;
    const char pad_buffer[4] = {20, 20, 20, 20};
    const size_t workers = std::max<size_t>(1, std::min<size_t>(omp_get_max_threads(), joint.size()));
    // per worker slice target; the buffers are sized from what the scan actually measured
    const size_t sliceBudget = std::max<size_t>(4 * 1024 * 1024,
                                               std::min<size_t>(Util::computeMemory(0) / 32 / workers,
                                                                (size_t) 64 * 1024 * 1024));

    struct WorkerScratch {
        std::vector<char> seqStage, hdrStage;
        std::vector<char> seqOut, hdrOut, seqIdxOut, hdrIdxOut;
        std::string lookupOut;
        std::vector<size_t> splitSeqBytes, splitHdrBytes, splitSeqBase, splitHdrBase;
        std::vector<size_t> splitSeqReadAt, splitHdrReadAt;
        size_t seqOutBytes, hdrOutBytes, seqIdxBytes, hdrIdxBytes;
    };
    std::vector<WorkerScratch> scratchPerWorker(workers);
    for (size_t w = 0; w < workers; w++) {
        scratchPerWorker[w].splitSeqBytes.assign(shuffleSplits, 0);
        scratchPerWorker[w].splitHdrBytes.assign(shuffleSplits, 0);
        scratchPerWorker[w].splitSeqBase.assign(shuffleSplits, 0);
        scratchPerWorker[w].splitHdrBase.assign(shuffleSplits, 0);
        scratchPerWorker[w].splitSeqReadAt.assign(shuffleSplits, 0);
        scratchPerWorker[w].splitHdrReadAt.assign(shuffleSplits, 0);
    }
    // read cursors carried across rounds; joint order restricted to a split is that split's file order
    std::vector<size_t> splitSeqCursor(shuffleSplits, 0);
    std::vector<size_t> splitHdrCursor(shuffleSplits, 0);
    std::vector<size_t> chunkBegin(workers + 1, 0);

    size_t mergedOffset = 0;
    size_t mergedOffsetHeader = 0;
    // the three text streams have variable width rows, so their per worker offsets only exist once
    // the round is formatted; the two data streams already got theirs from the round prefix sum
    size_t mergedLookupOffset = 0, mergedIdxOffset = 0, mergedHdrIdxOffset = 0;
    size_t prevBase[5] = {0, 0, 0, 0, 0};
    size_t prevBytes[5] = {0, 0, 0, 0, 0};
    // mean bytes per entry of the previous round; joint is length ordered, so it drifts slowly
    size_t meanEntryBytes = std::max<size_t>(1, maxLen + maxHdrLen);
    size_t roundStart = 0;
    while (roundStart < joint.size()) {
        const size_t perWorker = std::max<size_t>(1, sliceBudget / meanEntryBytes);
        const size_t roundEnd = std::min(joint.size(), roundStart + perWorker * workers);
        const size_t roundEntries = roundEnd - roundStart;
        const size_t activeWorkers = std::min(workers, (roundEntries + perWorker - 1) / perWorker);
        for (size_t w = 0; w <= activeWorkers; w++) {
            chunkBegin[w] = roundStart + std::min(roundEntries, w * perWorker);
        }

        // 1) measure: per worker per split byte counts, and the worker's own output volume
#pragma omp parallel for schedule(static) num_threads(activeWorkers)
        for (size_t w = 0; w < activeWorkers; w++) {
            WorkerScratch &s = scratchPerWorker[w];
            std::fill(s.splitSeqBytes.begin(), s.splitSeqBytes.end(), 0);
            std::fill(s.splitHdrBytes.begin(), s.splitHdrBytes.end(), 0);
            s.seqOutBytes = 0;
            s.hdrOutBytes = 0;
            for (size_t i = chunkBegin[w]; i < chunkBegin[w + 1]; i++) {
                const JointEntry &qe = joint[i];
                const size_t padding = (gpuLayout == false || qe.length % ALIGN == 0)
                                       ? 0 : ALIGN - qe.length % ALIGN;
                s.splitSeqBytes[qe.fileIdx] += qe.length;
                s.splitHdrBytes[qe.fileIdx] += qe.hdrLength;
                s.seqOutBytes += qe.length + padding;
                s.hdrOutBytes += qe.hdrLength;
            }
        }

        // 2) prefix sums: where each worker reads from, and what output offset its first entry lands on
        std::vector<size_t> seqOutBase(activeWorkers, 0), hdrOutBase(activeWorkers, 0);
        {
            std::vector<size_t> seqAt(splitSeqCursor), hdrAt(splitHdrCursor);
            size_t seqOutRunning = mergedOffset, hdrOutRunning = mergedOffsetHeader;
            for (size_t w = 0; w < activeWorkers; w++) {
                WorkerScratch &s = scratchPerWorker[w];
                size_t stageSeq = 0, stageHdr = 0;
                for (size_t f = 0; f < shuffleSplits; f++) {
                    s.splitSeqBase[f] = stageSeq;
                    s.splitHdrBase[f] = stageHdr;
                    stageSeq += s.splitSeqBytes[f];
                    stageHdr += s.splitHdrBytes[f];
                    s.splitSeqReadAt[f] = seqAt[f];
                    s.splitHdrReadAt[f] = hdrAt[f];
                    seqAt[f] += s.splitSeqBytes[f];
                    hdrAt[f] += s.splitHdrBytes[f];
                }
                if (s.seqStage.size() < stageSeq) { s.seqStage.resize(stageSeq); }
                if (s.hdrStage.size() < stageHdr) { s.hdrStage.resize(stageHdr); }
                seqOutBase[w] = seqOutRunning;
                hdrOutBase[w] = hdrOutRunning;
                seqOutRunning += s.seqOutBytes;
                hdrOutRunning += s.hdrOutBytes;
            }
        }

        // 3) read, format
#pragma omp parallel for schedule(static) num_threads(activeWorkers)
        for (size_t w = 0; w < activeWorkers; w++) {
            WorkerScratch &s = scratchPerWorker[w];
            for (size_t f = 0; f < shuffleSplits; f++) {
                if (s.splitSeqBytes[f] != 0) {
                    preadFully(inFdSeq[f], s.seqStage.data() + s.splitSeqBase[f], s.splitSeqBytes[f],
                               s.splitSeqReadAt[f], dataFiles[f]);
                }
                if (s.splitHdrBytes[f] != 0) {
                    preadFully(inFdHeader[f], s.hdrStage.data() + s.splitHdrBase[f], s.splitHdrBytes[f],
                               s.splitHdrReadAt[f], dataFilesHeader[f]);
                }
            }

            if (s.seqOut.size() < s.seqOutBytes) { s.seqOut.resize(s.seqOutBytes); }
            if (s.hdrOut.size() < s.hdrOutBytes) { s.hdrOut.resize(s.hdrOutBytes); }
            const size_t entries = chunkBegin[w + 1] - chunkBegin[w];
            const size_t idxRoom = entries * (3 * 20 + 4);
            if (s.seqIdxOut.size() < idxRoom) { s.seqIdxOut.resize(idxRoom); }
            if (s.hdrIdxOut.size() < idxRoom) { s.hdrIdxOut.resize(idxRoom); }
            s.lookupOut.clear();

            std::vector<size_t> seqTake(s.splitSeqBase), hdrTake(s.splitHdrBase);
            size_t seqPos = 0, hdrPos = 0, seqIdxPos = 0, hdrIdxPos = 0;
            size_t seqOff = seqOutBase[w], hdrOff = hdrOutBase[w];
            DBReader<DBKeyType>::LookupEntry lookupEntry;
            for (size_t i = chunkBegin[w]; i < chunkBegin[w + 1]; i++) {
                const JointEntry &qe = joint[i];
                memcpy(s.seqOut.data() + seqPos, s.seqStage.data() + seqTake[qe.fileIdx], qe.length);
                seqTake[qe.fileIdx] += qe.length;
                seqPos += qe.length;
                const size_t padding = (gpuLayout == false || qe.length % ALIGN == 0)
                                       ? 0 : ALIGN - qe.length % ALIGN;
                memcpy(s.seqOut.data() + seqPos, pad_buffer, padding);
                seqPos += padding;

                const char *headerAt = s.hdrStage.data() + hdrTake[qe.fileIdx];
                memcpy(s.hdrOut.data() + hdrPos, headerAt, qe.hdrLength);
                hdrTake[qe.fileIdx] += qe.hdrLength;
                hdrPos += qe.hdrLength;

                lookupEntry.id = i;
                // the writer terminated every header, so the index length covers the terminator
                lookupEntry.entryName = Util::parseFastaHeader(headerAt);
                if (lookupEntry.entryName.empty()) {
                    Debug(Debug::WARNING) << "Cannot extract identifier from entry " << i << "\n";
                }
                // no table: the key encodes the input file, or the counted key ranges locate it
                lookupEntry.fileNumber = (sourceLookup != NULL)
                    ? sourceLookup[qe.fileIdx][(qe.id - qe.fileIdx) / shuffleSplits]
                    : (fileKeyStarts != NULL
                       ? static_cast<unsigned int>(std::upper_bound(fileKeyStarts->begin(), fileKeyStarts->end(), qe.id) - fileKeyStarts->begin() - 1)
                       : unpackInputFile(qe.id));
                DBReader<DBKeyType>::lookupEntryToBuffer(s.lookupOut, lookupEntry);

                // the GPU layout stores neither, so its index still has to account for newline and terminator
                seqIdxPos += DBWriter::indexToBuffer(s.seqIdxOut.data() + seqIdxPos, i, seqOff,
                                                     gpuLayout ? qe.length + 2 : qe.length);
                hdrIdxPos += DBWriter::indexToBuffer(s.hdrIdxOut.data() + hdrIdxPos, i, hdrOff, qe.hdrLength);
                seqOff += qe.length + padding;
                hdrOff += qe.hdrLength;
            }
            s.seqIdxBytes = seqIdxPos;
            s.hdrIdxBytes = hdrIdxPos;
        }

        // 4) every worker lands at an offset the prefix sum already fixed, so the five files stay
        // byte identical to a serial pass while the writes themselves run in parallel
        const size_t roundBytesBefore = mergedOffset + mergedOffsetHeader;
        const size_t roundBase[5] = {mergedOffset, mergedOffsetHeader, mergedLookupOffset,
                                     mergedIdxOffset, mergedHdrIdxOffset};
        std::vector<size_t> lookupBase(activeWorkers), idxBase(activeWorkers), hdrIdxBase(activeWorkers);
        for (size_t w = 0; w < activeWorkers; w++) {
            WorkerScratch &s = scratchPerWorker[w];
            lookupBase[w] = mergedLookupOffset;
            idxBase[w] = mergedIdxOffset;
            hdrIdxBase[w] = mergedHdrIdxOffset;
            mergedLookupOffset += s.lookupOut.size();
            mergedIdxOffset += s.seqIdxBytes;
            mergedHdrIdxOffset += s.hdrIdxBytes;
            mergedOffset += s.seqOutBytes;
            mergedOffsetHeader += s.hdrOutBytes;
            for (size_t f = 0; f < shuffleSplits; f++) {
                splitSeqCursor[f] += s.splitSeqBytes[f];
                splitHdrCursor[f] += s.splitHdrBytes[f];
            }
        }

#pragma omp parallel for schedule(static) num_threads(activeWorkers)
        for (size_t w = 0; w < activeWorkers; w++) {
            WorkerScratch &s = scratchPerWorker[w];
            pwriteFully(outFds[0], s.seqOut.data(), s.seqOutBytes, seqOutBase[w], outDataFile);
            pwriteFully(outFds[1], s.hdrOut.data(), s.hdrOutBytes, hdrOutBase[w], outHeaderDataFile);
            pwriteFully(outFds[2], s.lookupOut.data(), s.lookupOut.size(), lookupBase[w], outLookupFile);
            pwriteFully(outFds[3], s.seqIdxOut.data(), s.seqIdxBytes, idxBase[w], outIndexFile);
            pwriteFully(outFds[4], s.hdrIdxOut.data(), s.hdrIdxBytes, hdrIdxBase[w], outHeaderIndexFile);
        }

        const size_t roundEnd5[5] = {mergedOffset, mergedOffsetHeader, mergedLookupOffset,
                                     mergedIdxOffset, mergedHdrIdxOffset};
        for (int st = 0; st < 5; st++) {
            retireRound(outFds[st], roundBase[st], roundEnd5[st] - roundBase[st], prevBase[st], prevBytes[st]);
            prevBase[st] = roundBase[st];
            prevBytes[st] = roundEnd5[st] - roundBase[st];
        }

        meanEntryBytes = std::max<size_t>(1, (mergedOffset + mergedOffsetHeader - roundBytesBefore)
                                             / std::max<size_t>(1, roundEntries));
        roundStart = roundEnd;
    }

    for (int i = 0; i < 5; i++) {
        if (close(outFds[i]) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << outNames[i] << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
    for (size_t i = 0; i < shuffleSplits; i++) {
        close(inFdSeq[i]);
        close(inFdHeader[i]);
        FileUtil::remove(dataFiles[i]);
        FileUtil::remove(indexFiles[i]);
        FileUtil::remove(dataFilesHeader[i]);
        FileUtil::remove(indexFilesHeader[i]);
    }
    return 0;
}

void processSeqBatch(Parameters & par, DBWriter &seqWriter, DBWriter &hdrWriter, BaseMatrix *subMat, int querySeqType,
                     Masker ** masker, Sequence ** seqs, size_t currId,
                     std::vector<std::pair<std::vector<char>, std::string>> &entries, const size_t entriesSize,
                     unsigned int shuffleSplits, bool gpuLayout){
    if (gpuLayout == false) {
        const char newline = '\n';
        for (size_t i = 0; i < entriesSize; i++) {
            const size_t id = currId + i;
            const size_t splitIdx = id % shuffleSplits;
            seqWriter.writeStart(splitIdx);
            seqWriter.writeAdd(entries[i].first.data(), entries[i].first.size(), splitIdx);
            seqWriter.writeAdd(&newline, 1, splitIdx);
            seqWriter.writeEnd(id, splitIdx, true);
            hdrWriter.writeData(entries[i].second.c_str(), entries[i].second.length(), id, splitIdx);
        }
        return;
    }
    if(masker[0] == NULL){
        for(int i = 0; i < par.threads; i++){
            masker[i] = new Masker(*subMat);
            seqs[i] = new Sequence(par.maxSeqLen, querySeqType, subMat, 0, false, false);
        }
    }

#pragma omp parallel num_threads(par.threads)
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
#pragma omp for schedule(dynamic, 10)
        for (size_t i = 0; i < entriesSize; i++) {
            seqs[thread_idx]->mapSequence(currId + i, currId + i, entries[i].first.data(), entries[i].first.size());
            const unsigned char *numSequence = seqs[thread_idx]->numSequence;
            std::copy_n(numSequence, seqs[thread_idx]->L, entries[i].first.begin());
            masker[thread_idx]->maskSequence(*seqs[thread_idx], par.maskMode, par.maskProb, par.maskLowerCaseMode,
                                             par.maskNrepeats);

            for (int j = 0; j < seqs[thread_idx]->L; j++) {
                entries[i].first[j] = (numSequence[j] == masker[thread_idx]->maskLetterNum) ? entries[i].first[j] + 32
                                                                                            : entries[i].first[j];
            }
        }
    }

    for (size_t i = 0; i < entriesSize; i++) {
        size_t id = currId + i;
        size_t splitIdx = id % shuffleSplits;
        seqWriter.writeData(entries[i].first.data(), entries[i].first.size(), currId + i, splitIdx, false);
        hdrWriter.writeData(entries[i].second.c_str(), entries[i].second.length(), currId + i, splitIdx);
    }
}

static bool sequenceLooksNucleotide(const char *sequence, size_t length);

// One worker per input file, each owning a disjoint range of the splits, so the writers never race.
static DBKeyType parseInputFilesParallel(const std::vector<std::string> &filenames,
                                         const std::vector<size_t> &declaredCounts,
                                         std::vector<DBKeyType> &fileKeyStarts,
                                         DBWriter &seqWriter, DBWriter &hdrWriter, FILE *source,
                                         unsigned int shuffleSplits, unsigned int workers,
                                         size_t testForNucSequence, size_t &isNuclCnt, size_t &sampleCount) {
    const size_t fileCount = filenames.size();
    const bool counted = declaredCounts.empty() == false;
    for (size_t fileIdx = 0; fileIdx < fileCount; fileIdx++) {
        const std::string sourceName = FileUtil::baseName(filenames[fileIdx]);
        char buffer[4096];
        const size_t len = snprintf(buffer, sizeof(buffer), "%zu\t%s\n", fileIdx, sourceName.c_str());
        if (fwrite(buffer, sizeof(char), len, source) != len) {
            Debug(Debug::ERROR) << "Cannot write to source file\n";
            EXIT(EXIT_FAILURE);
        }
    }

    if (counted) {
        // exclusive prefix sum: with final keys assigned up front, the declared total must fit the key type
        const uint64_t maxKey = static_cast<uint64_t>(std::numeric_limits<DBKeyType>::max());
        uint64_t running = 0;
        fileKeyStarts.resize(fileCount);
        for (size_t fileIdx = 0; fileIdx < fileCount; fileIdx++) {
            fileKeyStarts[fileIdx] = static_cast<DBKeyType>(running);
            if (static_cast<uint64_t>(declaredCounts[fileIdx]) > maxKey - running) {
                Debug(Debug::ERROR) << "Declared sequence counts exceed the " << sizeof(DBKeyType) * 8
                                    << "-bit key space of " << maxKey << " at " << filenames[fileIdx]
                                    << " (" << running << " sequences declared before it)\n";
                EXIT(EXIT_FAILURE);
            }
            running += declaredCounts[fileIdx];
        }
    } else if (fileCount >= (static_cast<size_t>(1) << (64 - CREATEDB_KEY_FILE_SHIFT))) {
        Debug(Debug::ERROR) << "More input files than the key layout allows\n";
        EXIT(EXIT_FAILURE);
    }
    Debug(Debug::INFO) << "Parallel createdb reads " << fileCount << " input files with " << workers
                       << " workers over " << shuffleSplits << " splits\n";
    std::vector<size_t> entriesPerFile(fileCount, 0);
    size_t nuclHits = 0, nuclSamples = 0;
    Debug::Progress progress;
#pragma omp parallel num_threads(workers) reduction(+: nuclHits) reduction(+: nuclSamples)
    {
        unsigned int worker = 0;
#ifdef OPENMP
        worker = static_cast<unsigned int>(omp_get_thread_num());
#endif
        // contiguous blocks, so every split has exactly one writer
        const unsigned int splitBegin = static_cast<size_t>(worker) * shuffleSplits / workers;
        const unsigned int splitEnd = static_cast<size_t>(worker + 1) * shuffleSplits / workers;
        const unsigned int splitCount = splitEnd - splitBegin;
        const char newline = '\n';
        std::string header;
        header.reserve(1024);

#pragma omp for schedule(dynamic, 1)
        for (size_t fileIdx = 0; fileIdx < fileCount; fileIdx++) {
            KSeqWrapper *kseq = KSeqFactory(filenames[fileIdx].c_str());
            size_t rank = 0;
            while (kseq->ReadEntry()) {
                const KSeqWrapper::KSeqEntry &e = kseq->entry;
                if (e.name.l == 0) {
                    Debug(Debug::ERROR) << "Fasta entry " << rank << " of " << filenames[fileIdx] << " is invalid\n";
                    EXIT(EXIT_FAILURE);
                }
                // an undeclared entry would take the next file's first key, so stop before writing it
                if (counted && rank >= declaredCounts[fileIdx]) {
                    Debug(Debug::ERROR) << "File " << filenames[fileIdx] << " declares "
                                        << declaredCounts[fileIdx] << " sequences but holds more\n";
                    EXIT(EXIT_FAILURE);
                }
                // per file, so the guess does not depend on which worker took which file
                if (rank < testForNucSequence) {
                    nuclSamples += 1;
                    nuclHits += sequenceLooksNucleotide(e.sequence.s, e.sequence.l) ? 1 : 0;
                }
                header.clear();
                header.append(e.name.s, e.name.l);
                if (e.comment.l > 0) {
                    header.append(" ", 1);
                    header.append(e.comment.s, e.comment.l);
                }
                if (Util::parseFastaHeader(header.c_str()).empty()) {
#pragma omp critical(createdb_parallel_warning)
                    Debug(Debug::WARNING) << "Cannot extract identifier from entry " << rank << "\n";
                }
                header.push_back('\n');

                const DBKeyType key = counted ? fileKeyStarts[fileIdx] + static_cast<DBKeyType>(rank)
                                              : packInputKey(fileIdx, rank);
                const unsigned int splitIdx = splitBegin + static_cast<unsigned int>(rank % splitCount);
                hdrWriter.writeData(header.c_str(), header.length(), key, splitIdx);
                seqWriter.writeStart(splitIdx);
                seqWriter.writeAdd(e.sequence.s, e.sequence.l, splitIdx);
                seqWriter.writeAdd(&newline, 1, splitIdx);
                seqWriter.writeEnd(key, splitIdx, true);
                rank++;
                if ((rank & 0xFFFF) == 0) {
                    progress.updateProgress(rank);
                }
            }
            if (counted && rank != declaredCounts[fileIdx]) {
                Debug(Debug::ERROR) << "File " << filenames[fileIdx] << " declares "
                                    << declaredCounts[fileIdx] << " sequences but holds " << rank << "\n";
                EXIT(EXIT_FAILURE);
            }
            if (rank == 0) {
#pragma omp critical(createdb_parallel_warning)
                Debug(Debug::WARNING) << "File " << filenames[fileIdx] << " is empty or invalid and was ignored\n";
            }
            entriesPerFile[fileIdx] = rank;
            delete kseq;
        }
    }
    Debug(Debug::INFO) << "\n";

    isNuclCnt = nuclHits;
    sampleCount = nuclSamples;
    DBKeyType total = 0;
    for (size_t fileIdx = 0; fileIdx < fileCount; fileIdx++) {
        if (counted == false && entriesPerFile[fileIdx] >= (static_cast<size_t>(1) << CREATEDB_KEY_FILE_SHIFT)) {
            Debug(Debug::ERROR) << "File " << filenames[fileIdx] << " has more entries than the key layout allows\n";
            EXIT(EXIT_FAILURE);
        }
        total += entriesPerFile[fileIdx];
    }
    return total;
}

struct CreatedbHardModePart {
    std::string seqData;
    std::string seqIndex;
    std::string hdrData;
    std::string hdrIndex;
    std::string sourceName;
    std::vector<char> sampleIsNucleotide;
    DBKeyType entries;
    bool empty;

    CreatedbHardModePart() : entries(0), empty(false) {}
};

static bool sequenceLooksNucleotide(const char *sequence, size_t length) {
    if (length == 0) {
        return false;
    }
    size_t cnt = 0;
    for (size_t i = 0; i < length; i++) {
        switch (toupper(static_cast<unsigned char>(sequence[i]))) {
            case 'T':
            case 'A':
            case 'G':
            case 'C':
            case 'U':
            case 'N':
                cnt++;
                break;
        }
    }
    const float nuclDNAFraction = static_cast<float>(cnt) / static_cast<float>(length);
    return nuclDNAFraction > 0.9;
}

static void removeIfExists(const std::string &path) {
    if (FileUtil::fileExists(path.c_str()) == true || FileUtil::symlinkExists(path) == true) {
        FileUtil::remove(path.c_str());
    }
}

// this concurrency is a storage property, not a cpu count, so it does not scale with --threads
static const size_t CREATEDB_DEFAULT_FILE_WORKERS = 8;

static unsigned int getCreatedbThreads(unsigned int requestedThreads, int requestedFileThreads,
                                       size_t fileCount) {
#ifndef OPENMP
    (void) requestedThreads;
    (void) requestedFileThreads;
    (void) fileCount;
    return 1;
#else
    const size_t threadBudget = std::max<size_t>(1, requestedThreads);
    size_t threads = (requestedFileThreads > 0)
        ? static_cast<size_t>(requestedFileThreads)
        : CREATEDB_DEFAULT_FILE_WORKERS;
    if (requestedFileThreads > 0 && static_cast<size_t>(requestedFileThreads) > threadBudget) {
        Debug(Debug::WARNING) << "--createdb-threads " << requestedFileThreads << " exceeds --threads "
                              << threadBudget << ", using " << threadBudget << "\n";
    }
    // a worker is an OpenMP thread, so --threads stays the budget a cgroup or scheduler set
    threads = std::min(threads, threadBudget);
    threads = std::min(threads, std::max<size_t>(1, fileCount));

    struct rlimit limit;
    if (getrlimit(RLIMIT_NOFILE, &limit) == 0 && limit.rlim_cur != RLIM_INFINITY) {
        const size_t fdSlack = 32;
        const size_t fdPerWorker = 6;
        size_t fdLimit = static_cast<size_t>(limit.rlim_cur);
        size_t fdThreads = (fdLimit > fdSlack) ? ((fdLimit - fdSlack) / fdPerWorker) : 1;
        threads = std::min(threads, std::max<size_t>(1, fdThreads));
    }

    return static_cast<unsigned int>(std::max<size_t>(1, threads));
#endif
}

static size_t appendFileToOpenFile(const std::string &input, FILE *output, std::vector<char> &buffer) {
    FILE *in = FileUtil::openFileOrDie(input.c_str(), "rb", true);
    size_t total = 0;
    while (true) {
        size_t read = fread(buffer.data(), 1, buffer.size(), in);
        if (read > 0) {
            size_t written = fwrite(buffer.data(), 1, read, output);
            if (written != read) {
                Debug(Debug::ERROR) << "Can not write merged createdb data\n";
                EXIT(EXIT_FAILURE);
            }
            total += read;
        }
        if (read < buffer.size()) {
            if (ferror(in)) {
                Debug(Debug::ERROR) << "Can not read partial createdb data " << input << "\n";
                EXIT(EXIT_FAILURE);
            }
            break;
        }
    }
    if (fclose(in) != 0) {
        Debug(Debug::ERROR) << "Cannot close partial createdb data " << input << "\n";
        EXIT(EXIT_FAILURE);
    }
    return total;
}

static void rewritePartialIndex(const std::string &dataFile, const std::string &indexFile,
                                FILE *outIndex, size_t mergedDataOffset, DBKeyType keyOffset) {
    DBReader<DBKeyType> reader(dataFile.c_str(), indexFile.c_str(), 1, DBReader<DBKeyType>::USE_INDEX);
    reader.open(DBReader<DBKeyType>::HARDNOSORT);
    char indexBuffer[1024];
    for (size_t i = 0; i < reader.getSize(); i++) {
        DBReader<DBKeyType>::Index entry = *reader.getIndex(i);
        entry.id = keyOffset + static_cast<DBKeyType>(i);
        entry.offset += mergedDataOffset;
        DBWriter::writeIndexEntryToFile(outIndex, indexBuffer, entry);
    }
    reader.close();
}

static void mergeCreatedbHardModeParts(const std::vector<CreatedbHardModePart> &parts,
                                       const std::string &dataFile, const std::string &indexFile,
                                       const std::string &hdrDataFile, const std::string &hdrIndexFile,
                                       DBKeyType identifierOffset) {
    FILE *seqOut = FileUtil::openAndDelete(dataFile.c_str(), "wb");
    FILE *seqIndexOut = FileUtil::openAndDelete(indexFile.c_str(), "wb");
    FILE *hdrOut = FileUtil::openAndDelete(hdrDataFile.c_str(), "wb");
    FILE *hdrIndexOut = FileUtil::openAndDelete(hdrIndexFile.c_str(), "wb");
    setvbuf(seqOut, NULL, _IOFBF, 1024 * 1024 * 50);
    setvbuf(seqIndexOut, NULL, _IOFBF, 1024 * 1024 * 50);
    setvbuf(hdrOut, NULL, _IOFBF, 1024 * 1024 * 50);
    setvbuf(hdrIndexOut, NULL, _IOFBF, 1024 * 1024 * 50);

    std::vector<char> copyBuffer(8 * 1024 * 1024);
    size_t seqOffset = 0;
    size_t hdrOffset = 0;
    DBKeyType keyOffset = identifierOffset;
    for (size_t fileIdx = 0; fileIdx < parts.size(); fileIdx++) {
        const CreatedbHardModePart &part = parts[fileIdx];
        rewritePartialIndex(part.seqData, part.seqIndex, seqIndexOut, seqOffset, keyOffset);
        rewritePartialIndex(part.hdrData, part.hdrIndex, hdrIndexOut, hdrOffset, keyOffset);
        seqOffset += appendFileToOpenFile(part.seqData, seqOut, copyBuffer);
        hdrOffset += appendFileToOpenFile(part.hdrData, hdrOut, copyBuffer);
        keyOffset += part.entries;
    }

    if (fclose(seqOut) != 0) {
        Debug(Debug::ERROR) << "Cannot close data file " << dataFile << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (fclose(seqIndexOut) != 0) {
        Debug(Debug::ERROR) << "Cannot close index file " << indexFile << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (fclose(hdrOut) != 0) {
        Debug(Debug::ERROR) << "Cannot close header file " << hdrDataFile << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (fclose(hdrIndexOut) != 0) {
        Debug(Debug::ERROR) << "Cannot close header index file " << hdrIndexFile << "\n";
        EXIT(EXIT_FAILURE);
    }
}

static void writeLookupForCreatedbHardModeParts(const std::vector<CreatedbHardModePart> &parts,
                                                const std::string &dataFile,
                                                const std::string &hdrDataFile,
                                                const std::string &hdrIndexFile) {
    DBReader<DBKeyType> readerHeader(hdrDataFile.c_str(), hdrIndexFile.c_str(), 1,
                                     DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_INDEX);
    readerHeader.open(DBReader<DBKeyType>::NOSORT);

    std::string lookupFile = dataFile + ".lookup";
    FILE *file = FileUtil::openAndDelete(lookupFile.c_str(), "w");
    std::string buffer;
    buffer.reserve(2048);
    DBReader<DBKeyType>::LookupEntry entry;
    size_t id = 0;
    for (size_t fileIdx = 0; fileIdx < parts.size(); fileIdx++) {
        for (DBKeyType localId = 0; localId < parts[fileIdx].entries; localId++, id++) {
            char *header = readerHeader.getData(id, 0);
            entry.id = static_cast<DBKeyType>(id);
            entry.entryName = Util::parseFastaHeader(header);
            if (entry.entryName.empty()) {
                Debug(Debug::WARNING) << "Cannot extract identifier from entry " << id << "\n";
            }
            entry.fileNumber = static_cast<DBKeyType>(fileIdx);
            DBReader<DBKeyType>::lookupEntryToBuffer(buffer, entry);
            size_t written = fwrite(buffer.c_str(), sizeof(char), buffer.size(), file);
            if (written != buffer.size()) {
                Debug(Debug::ERROR) << "Cannot write to lookup file " << lookupFile << "\n";
                EXIT(EXIT_FAILURE);
            }
            buffer.clear();
        }
    }
    if (id != readerHeader.getSize()) {
        Debug(Debug::ERROR) << "Parallel createdb lookup size mismatch: wrote " << id
                            << " entries but header DB contains " << readerHeader.getSize() << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (fclose(file) != 0) {
        Debug(Debug::ERROR) << "Cannot close file " << lookupFile << "\n";
        EXIT(EXIT_FAILURE);
    }
    readerHeader.close();
}

static void cleanupCreatedbHardModeParts(const std::vector<CreatedbHardModePart> &parts) {
    for (size_t i = 0; i < parts.size(); i++) {
        removeIfExists(parts[i].seqData);
        removeIfExists(parts[i].seqIndex);
        removeIfExists(parts[i].hdrData);
        removeIfExists(parts[i].hdrIndex);
        removeIfExists(parts[i].seqData + ".dbtype");
        removeIfExists(parts[i].hdrData + ".dbtype");
    }
}

static CreatedbHardModePart writeCreatedbHardModePart(const Parameters &par, const std::vector<std::string> &filenames,
                                                      const std::string &dataFile, const std::string &hdrDataFile,
                                                      size_t fileIdx, int dbType,
                                                      Debug::Progress &progress, size_t &progressCounter) {
    CreatedbHardModePart part;
    part.sourceName = FileUtil::baseName(filenames[fileIdx]);
    part.seqData = dataFile + ".parallel." + SSTR(fileIdx);
    part.seqIndex = part.seqData + ".index";
    part.hdrData = hdrDataFile + ".parallel." + SSTR(fileIdx);
    part.hdrIndex = part.hdrData + ".index";
    part.sampleIsNucleotide.reserve(10);

    removeIfExists(part.seqData);
    removeIfExists(part.seqIndex);
    removeIfExists(part.hdrData);
    removeIfExists(part.hdrIndex);
    removeIfExists(part.seqData + ".dbtype");
    removeIfExists(part.hdrData + ".dbtype");

    const size_t writerBufferSize = 8 * 1024 * 1024;
    DBWriter hdrWriter(part.hdrData.c_str(), part.hdrIndex.c_str(), 1, par.compressed, Parameters::DBTYPE_GENERIC_DB);
    DBWriter seqWriter(part.seqData.c_str(), part.seqIndex.c_str(), 1, par.compressed,
                       (dbType == -1) ? Parameters::DBTYPE_OMIT_FILE : dbType);
#pragma omp critical(createdb_parallel_writer_open)
    {
        hdrWriter.open(writerBufferSize);
        seqWriter.open(writerBufferSize);
    }

    KSeqWrapper *kseq = KSeqFactory(filenames[fileIdx].c_str());
    std::string header;
    header.reserve(1024);
    const char newline = '\n';
    size_t numEntriesInCurrFile = 0;
    while (kseq->ReadEntry()) {
        size_t done = __sync_add_and_fetch(&progressCounter, 1);
        if ((done % 10000) == 0) {
#pragma omp critical(createdb_parallel_progress)
            {
                progress.updateProgress(done);
            }
        }

        const KSeqWrapper::KSeqEntry &e = kseq->entry;
        if (e.name.l == 0) {
            Debug(Debug::ERROR) << "Fasta entry " << numEntriesInCurrFile << " is invalid\n";
            EXIT(EXIT_FAILURE);
        }
        if (part.sampleIsNucleotide.size() < 10) {
            part.sampleIsNucleotide.push_back(sequenceLooksNucleotide(e.sequence.s, e.sequence.l) ? 1 : 0);
        }

        header.append(e.name.s, e.name.l);
        if (e.comment.l > 0) {
            header.append(" ", 1);
            header.append(e.comment.s, e.comment.l);
        }
        std::string headerId = Util::parseFastaHeader(header.c_str());
        if (headerId.empty()) {
#pragma omp critical(createdb_parallel_warning)
            {
                Debug(Debug::WARNING) << "Cannot extract identifier from entry " << numEntriesInCurrFile
                                      << " in " << part.sourceName << "\n";
            }
        }
        header.push_back('\n');

        DBKeyType localId = static_cast<DBKeyType>(part.entries);
        hdrWriter.writeData(header.c_str(), header.length(), localId, 0);
        seqWriter.writeStart(0);
        seqWriter.writeAdd(e.sequence.s, e.sequence.l, 0);
        seqWriter.writeAdd(&newline, 1, 0);
        seqWriter.writeEnd(localId, 0, true);

        part.entries++;
        numEntriesInCurrFile++;
        header.clear();
    }

    part.empty = (numEntriesInCurrFile == 0);
    delete kseq;
#pragma omp critical(createdb_parallel_writer_close)
    {
        hdrWriter.close(true, false);
        seqWriter.close(true, false);
    }
    return part;
}

static int createdbHardModeParallelInputFiles(Parameters &par, const std::vector<std::string> &filenames,
                                              const std::string &dataFile, const std::string &indexFile,
                                              const std::string &hdrDataFile, const std::string &hdrIndexFile,
                                              const std::string &sourceFile, int dbType) {
    const size_t fileCount = filenames.size();
    unsigned int parallelFiles = getCreatedbThreads(static_cast<unsigned int>(par.threads),
                                                    par.createdbThreads, fileCount);
    Debug(Debug::INFO) << "Parallel hard-mode createdb uses " << parallelFiles << " file workers for "
                       << fileCount << " input files\n";

    FILE *source = fopen(sourceFile.c_str(), "w");
    if (source == NULL) {
        Debug(Debug::ERROR) << "Cannot open " << sourceFile << " for writing\n";
        EXIT(EXIT_FAILURE);
    }
    for (size_t fileIdx = 0; fileIdx < fileCount; fileIdx++) {
        std::string sourceName = FileUtil::baseName(filenames[fileIdx]);
        char buffer[4096];
        size_t len = snprintf(buffer, sizeof(buffer), "%zu\t%s\n", fileIdx, sourceName.c_str());
        size_t written = fwrite(buffer, sizeof(char), len, source);
        if (written != len) {
            Debug(Debug::ERROR) << "Cannot write to source file " << sourceFile << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
    if (fclose(source) != 0) {
        Debug(Debug::ERROR) << "Cannot close file " << sourceFile << "\n";
        EXIT(EXIT_FAILURE);
    }

    Debug::Progress progress;
    progress.updateProgress(0);
    size_t progressCounter = 0;
    std::vector<CreatedbHardModePart> parts(fileCount);
#pragma omp parallel for schedule(dynamic, 1) num_threads(parallelFiles)
    for (size_t fileIdx = 0; fileIdx < fileCount; fileIdx++) {
        parts[fileIdx] = writeCreatedbHardModePart(par, filenames, dataFile, hdrDataFile,
                                                   fileIdx, dbType, progress, progressCounter);
    }
    Debug(Debug::INFO) << "\n";

    DBKeyType entriesNum = 0;
    size_t sampleCount = 0;
    size_t isNuclCnt = 0;
    for (size_t fileIdx = 0; fileIdx < fileCount; fileIdx++) {
        if (parts[fileIdx].empty) {
            Debug(Debug::WARNING) << "File " << parts[fileIdx].sourceName << " is empty or invalid and was ignored\n";
        }
        entriesNum += parts[fileIdx].entries;
        for (size_t i = 0; i < parts[fileIdx].sampleIsNucleotide.size() && sampleCount < 10; i++) {
            isNuclCnt += (parts[fileIdx].sampleIsNucleotide[i] != 0);
            sampleCount++;
        }
    }

    if (entriesNum == 0) {
        Debug(Debug::ERROR) << "The input files have no entry: ";
        for (size_t fileIdx = 0; fileIdx < filenames.size(); fileIdx++) {
            Debug(Debug::ERROR) << " - " << filenames[fileIdx] << "\n";
        }
        Debug(Debug::ERROR) << "Please check your input files. Only files in fasta/fastq[.gz|.bz2|.zst] are supported\n";
        cleanupCreatedbHardModeParts(parts);
        EXIT(EXIT_FAILURE);
    }

    Timer timer;
    mergeCreatedbHardModeParts(parts, dataFile, indexFile, hdrDataFile, hdrIndexFile,
                               static_cast<DBKeyType>(par.identifierOffset));
    Debug(Debug::INFO) << "Merge parallel createdb files " << timer.lap() << "\n";

    if (dbType == -1) {
        dbType = (isNuclCnt == sampleCount) ? Parameters::DBTYPE_NUCLEOTIDES : Parameters::DBTYPE_AMINO_ACIDS;
    }
    DBWriter::writeDbtypeFile(dataFile.c_str(), dbType, par.compressed);
    DBWriter::writeDbtypeFile(hdrDataFile.c_str(), Parameters::DBTYPE_GENERIC_DB, par.compressed);
    Debug(Debug::INFO) << "Database type: " << Parameters::getDbTypeName(dbType) << "\n";

    if (par.writeLookup == true) {
        timer.reset();
        writeLookupForCreatedbHardModeParts(parts, dataFile, hdrDataFile, hdrIndexFile);
        Debug(Debug::INFO) << "Write parallel createdb lookup " << timer.lap() << "\n";
    }

    cleanupCreatedbHardModeParts(parts);
    return EXIT_SUCCESS;
}


int createdb(int argc, const char **argv, const Command& command) {
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, false, Parameters::PARSE_VARIADIC, 0);
    if(par.gpu){
        par.createdbMode = Parameters::SEQUENCE_SPLIT_MODE_GPU;
        par.maskMode = 1;
    }else{
        par.maskMode = 0;
    }
    par.printParameters(command.cmd, argc, argv, *command.params);

    std::vector<std::string> filenames(par.filenames);
    std::string dataFile = filenames.back();
    filenames.pop_back();

    std::vector<size_t> declaredCounts;
    if (Util::endsWith(".tsv", filenames[0])) {
	    if (filenames.size() > 1) {
		    Debug(Debug::ERROR) << "Only one tsv file can be given\n";
		    EXIT(EXIT_FAILURE);
	    }
	    std::string tsv = filenames.back();
	    filenames.pop_back();

	    FILE* file = FileUtil::openFileOrDie(tsv.c_str(), "r", true);
	    char* line = NULL;
	    size_t len = 0;
	    ssize_t read;
	    while ((read = getline(&line, &len, file)) != -1) {
		    if (line[read - 1] == '\n') {
			    line[read - 1] = '\0';
			    read--;
		    }
		    // optional second column: the file's sequence count, so parallel parsers know their key ranges
		    char* tab = strchr(line, '\t');
		    if (tab != NULL) {
			    *tab = '\0';
			    char* end = NULL;
			    errno = 0;
			    const unsigned long long count = strtoull(tab + 1, &end, 10);
			    if (isdigit(static_cast<unsigned char>(tab[1])) == 0 || errno != 0 || *end != '\0') {
				    Debug(Debug::ERROR) << "Invalid sequence count \"" << (tab + 1) << "\" for " << line << " in " << tsv << "\n";
				    EXIT(EXIT_FAILURE);
			    }
			    declaredCounts.push_back(count);
		    }
		    filenames.push_back(line);
	    }
	    free(line);
	    fclose(file);
	    if (declaredCounts.empty() == false && declaredCounts.size() != filenames.size()) {
		    Debug(Debug::ERROR) << "The tsv " << tsv << " mixes rows with and without a sequence count: all rows need one, or none\n";
		    EXIT(EXIT_FAILURE);
	    }
    }

    // consistent order
    if (declaredCounts.empty()) {
        SORT_SERIAL(filenames.begin(), filenames.end(), [](const std::string &a, const std::string &b) {
            return FileUtil::baseName(a) < FileUtil::baseName(b);
        });
    } else {
        // counts ride the same basename order, so row i keeps describing filenames[i]
        std::vector<size_t> order(filenames.size());
        for (size_t i = 0; i < order.size(); i++) {
            order[i] = i;
        }
        SORT_SERIAL(order.begin(), order.end(), [&filenames](size_t a, size_t b) {
            return FileUtil::baseName(filenames[a]) < FileUtil::baseName(filenames[b]);
        });
        std::vector<std::string> sortedNames(filenames.size());
        std::vector<size_t> sortedCounts(filenames.size());
        for (size_t i = 0; i < order.size(); i++) {
            sortedNames[i] = filenames[order[i]];
            sortedCounts[i] = declaredCounts[order[i]];
        }
        filenames.swap(sortedNames);
        declaredCounts.swap(sortedCounts);
    }

    for (size_t i = 0; i < filenames.size(); i++) {
        if (FileUtil::directoryExists(filenames[i].c_str()) == true) {
            Debug(Debug::ERROR) << "File " << filenames[i] << " is a directory\n";
            EXIT(EXIT_FAILURE);
        }
    }

    bool dbInput = false;
    if (FileUtil::fileExists(par.db1dbtype.c_str()) == true) {
        if (filenames.size() > 1) {
            Debug(Debug::ERROR) << "Only one database can be used with database input\n";
            EXIT(EXIT_FAILURE);
        }
        dbInput = true;
        par.createdbMode = Parameters::SEQUENCE_SPLIT_MODE_HARD;
    }

    int dbType = -1;
    BaseMatrix * subMat = NULL;
    if (par.dbType == 2) {
        dbType = Parameters::DBTYPE_NUCLEOTIDES;
        subMat = new NucleotideMatrix(par.scoringMatrixFile.values.nucleotide().c_str(), 2.0, -0.0f);
    } else if(par.dbType == 1) {
        dbType = Parameters::DBTYPE_AMINO_ACIDS;
        subMat = new SubstitutionMatrix(par.scoringMatrixFile.values.aminoacid().c_str(), 2.0, -0.0f);
    }

    std::string indexFile = dataFile + ".index";
    if (par.createdbMode == Parameters::SEQUENCE_SPLIT_MODE_SOFT && par.shuffleDatabase) {
        Debug(Debug::WARNING) << "Shuffle database cannot be combined with --createdb-mode " << Parameters::SEQUENCE_SPLIT_MODE_SOFT << "\n";
        Debug(Debug::WARNING) << "We recompute with --shuffle 0\n";
        par.shuffleDatabase = false;
    }

    if (isSortedCreatedbMode(par.createdbMode) && par.shuffleDatabase == false) {
        Debug(Debug::WARNING) << "Shuffle database cannot be turned off for --createdb-mode " << par.createdbMode << "\n";
        Debug(Debug::WARNING) << "We recompute with --shuffle 1\n";
        par.shuffleDatabase = true;
    }

    if (par.createdbMode == Parameters::SEQUENCE_SPLIT_MODE_SOFT && par.filenames[0] == "stdin") {
        Debug(Debug::WARNING) << "Stdin input cannot be combined with --createdb-mode " << Parameters::SEQUENCE_SPLIT_MODE_SOFT << "\n";
        Debug(Debug::WARNING) << "We recompute with --createdb-mode " << Parameters::SEQUENCE_SPLIT_MODE_HARD << "\n";
        par.createdbMode = Parameters::SEQUENCE_SPLIT_MODE_HARD;
    }

    if ((par.maskMode == 1 || par.maskNrepeats > 0) && par.createdbMode != Parameters::SEQUENCE_SPLIT_MODE_GPU) {
        Debug(Debug::WARNING) << "Masking can only be used with --createdb-mode " << Parameters::SEQUENCE_SPLIT_MODE_GPU << "\n";
        Debug(Debug::WARNING) << "We recompute with --mask 0 or --mask-n-repeat 0\n";
        par.maskMode = false;
        par.maskNrepeats = 0;
    }



    unsigned int shuffleSplits = par.shuffleDatabase ? (unsigned int)par.shuffleSplits : 1;
    // a split carries four writer buffers; packed input would understate its size, so skip those
    if (par.createdbMode == Parameters::SEQUENCE_SPLIT_MODE_LENGTH_DESC
        && par.PARAM_SHUFFLE_SPLITS.wasSet == false && shuffleSplits > 1) {
        size_t inputBytes = 0;
        bool packedInput = false;
        for (size_t fileIdx = 0; fileIdx < filenames.size(); fileIdx++) {
            const std::string &name = filenames[fileIdx];
            packedInput = packedInput || Util::endsWith(".gz", name) || Util::endsWith(".bz2", name)
                          || Util::endsWith(".zst", name) || name == "stdin";
            inputBytes += FileUtil::getFileSize(name);
        }
        if (packedInput == false) {
            const unsigned int fits = (unsigned int) (inputBytes / CREATEDB_MIN_SPLIT_BYTES);
            // a parse worker owns at least one split, so never drop below what the parse can use
            const unsigned int parseFloor = (unsigned int) std::min<size_t>(filenames.size(), par.threads);
            const unsigned int reduced = std::min(shuffleSplits, std::max<unsigned int>(1,
                                                  std::max(fits, parseFloor)));
            if (reduced < shuffleSplits) {
                Debug(Debug::INFO) << "Input is " << inputBytes << " byte, using " << reduced
                                   << " of " << shuffleSplits << " splits\n";
                shuffleSplits = reduced;
            }
        }
    }
    if (par.createdbMode == Parameters::SEQUENCE_SPLIT_MODE_SOFT && par.compressed) {
        Debug(Debug::WARNING) << "Compressed database cannot be combined with --createdb-mode " << Parameters::SEQUENCE_SPLIT_MODE_SOFT << "\n";
        Debug(Debug::WARNING) << "We recompute with --compressed 0\n";
        par.compressed = 0;
    }

    std::string hdrDataFile = dataFile + "_h";
    std::string hdrIndexFile = dataFile + "_h.index";

    DBKeyType entries_num = 0;
    const char newline = '\n';

    size_t sampleCount = 0;
    const size_t testForNucSequence = 100;
    size_t isNuclCnt = 0;
    Debug::Progress progress;
    Debug(Debug::INFO) << "Converting sequences\n";

    std::string sourceFile = dataFile + ".source";
    bool allInputsAreFiles = true;
    for (size_t fileIdx = 0; fileIdx < filenames.size(); fileIdx++) {
        allInputsAreFiles = allInputsAreFiles && (filenames[fileIdx] != "stdin");
    }
    const bool canUseParallelInputFiles =
        dbInput == false &&
        par.createdbMode == Parameters::SEQUENCE_SPLIT_MODE_HARD &&
        filenames.size() > 1 &&
        shuffleSplits == 1 &&
        allInputsAreFiles == true;
    const bool requestedParallelFiles = par.createdbThreads > 1;
    if (requestedParallelFiles && canUseParallelInputFiles == false) {
        Debug(Debug::ERROR) << "--createdb-threads > 1 requires --createdb-mode "
                            << Parameters::SEQUENCE_SPLIT_MODE_HARD
                            << ", multiple file inputs, --shuffle 0, and no database/stdin input\n";
        EXIT(EXIT_FAILURE);
    }
#ifndef OPENMP
    if (requestedParallelFiles) {
        Debug(Debug::ERROR) << "--createdb-threads > 1 requires an OpenMP build\n";
        EXIT(EXIT_FAILURE);
    }
#endif
    if (canUseParallelInputFiles) {
        int ret = createdbHardModeParallelInputFiles(par, filenames, dataFile, indexFile,
                                                     hdrDataFile, hdrIndexFile, sourceFile, dbType);
        if (subMat != NULL) {
            delete subMat;
        }
        return ret;
    }

    const bool needSourceLookup = par.writeLookup || isSortedCreatedbMode(par.createdbMode);
    std::vector<unsigned int>* sourceLookup = needSourceLookup ? new std::vector<unsigned int>[shuffleSplits]() : NULL;
    if (sourceLookup != NULL) {
        for (size_t i = 0; i < shuffleSplits; ++i) {
            sourceLookup[i].reserve(16384);
        }
    }
    std::vector<DBKeyType> fileKeyStarts;

    redoComputation:
    FILE *source = fopen(sourceFile.c_str(), "w");
    if (source == NULL) {
        Debug(Debug::ERROR) << "Cannot open " << sourceFile << " for writing\n";
        EXIT(EXIT_FAILURE);
    }
    DBWriter hdrWriter(hdrDataFile.c_str(), hdrIndexFile.c_str(), shuffleSplits, par.compressed, Parameters::DBTYPE_GENERIC_DB);
    hdrWriter.open();
    DBWriter seqWriter(dataFile.c_str(), indexFile.c_str(), shuffleSplits, par.compressed, (dbType == -1) ? Parameters::DBTYPE_OMIT_FILE : dbType );
    seqWriter.open();
    size_t headerFileOffset = 0;
    size_t seqFileOffset = 0;

    size_t fileCount = filenames.size();
    DBReader<DBKeyType>* reader = NULL;
    if (dbInput == true) {
        reader = new DBReader<DBKeyType>(par.db1.c_str(), par.db1Index.c_str(), 1, DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_LOOKUP);
        reader->open(DBReader<DBKeyType>::LINEAR_ACCCESS);
        fileCount = reader->getSize();
    }

    // setup Sequence object pointer
    Sequence ** seqs = new Sequence * [par.threads];
    Masker ** masker = new Masker * [par.threads];
    masker[0] = NULL;
    const size_t BATCH_SIZE = par.threads * 10000;
    std::vector<std::pair<std::vector<char>, std::string>> batchEntries(BATCH_SIZE);
    size_t batchPos = 0;
    // packed (file, rank) keys need 64 bit; declared counts give final key ranges at any key width
    const bool parseFilesInParallel =
        dbInput == false
        && par.createdbMode == Parameters::SEQUENCE_SPLIT_MODE_LENGTH_DESC
        && filenames.size() > 1
        && allInputsAreFiles == true
        && (declaredCounts.empty() == false || sizeof(DBKeyType) == sizeof(uint64_t));
    if (parseFilesInParallel) {
        // a worker owns at least one split, so more workers than splits would leave some with none
        const size_t wantWorkers = std::min<size_t>(filenames.size(),
            (par.createdbThreads > 1) ? (size_t) par.createdbThreads : (size_t) par.threads);
        const unsigned int workers = std::max<size_t>(1, std::min<size_t>(shuffleSplits, wantWorkers));
        if (workers < wantWorkers) {
            Debug(Debug::WARNING) << "Only " << workers << " of " << wantWorkers
                                  << " parse workers can run: raise --shuffle-splits above "
                                  << shuffleSplits << "\n";
        }
        entries_num = parseInputFilesParallel(filenames, declaredCounts, fileKeyStarts,
                                              seqWriter, hdrWriter, source,
                                              shuffleSplits, workers, testForNucSequence,
                                              isNuclCnt, sampleCount);
        fileCount = 0;
    }
    for (size_t fileIdx = 0; fileIdx < fileCount; fileIdx++) {
        size_t numEntriesInCurrFile = 0;
        std::string header;
        header.reserve(1024);

        std::string sourceName;
        if (dbInput == true) {
            DBKeyType dbKey = reader->getDbKey(fileIdx);
            size_t lookupId = reader->getLookupIdByKey(dbKey);
            sourceName = reader->getLookupEntryName(lookupId);
        } else {
            sourceName = FileUtil::baseName(filenames[fileIdx]);
        }
        char buffer[4096];
        size_t len = snprintf(buffer, sizeof(buffer), "%zu\t%s\n", fileIdx, sourceName.c_str());
        size_t written = fwrite(buffer, sizeof(char), len, source);
        if (written != len) {
            Debug(Debug::ERROR) << "Cannot write to source file " << sourceFile << "\n";
            EXIT(EXIT_FAILURE);
        }

        KSeqWrapper* kseq = NULL;
        if (dbInput == true) {
            kseq = new KSeqBuffer(reader->getData(fileIdx, 0), reader->getEntryLen(fileIdx) - 1);
        } else {
            kseq = KSeqFactory(filenames[fileIdx].c_str());
        }

        bool resetNotFile = par.createdbMode == Parameters::SEQUENCE_SPLIT_MODE_SOFT && kseq->type != KSeqWrapper::KSEQ_FILE;
        if (resetNotFile) {
            Debug(Debug::WARNING) << "Only uncompressed fasta files can be used with --createdb-mode " << Parameters::SEQUENCE_SPLIT_MODE_SOFT << "\n";
            Debug(Debug::WARNING) << "We recompute with --createdb-mode " << Parameters::SEQUENCE_SPLIT_MODE_HARD << "\n";
        }

        bool resetIncorrectNewline = false;
        if (par.createdbMode == Parameters::SEQUENCE_SPLIT_MODE_SOFT && kseq->type == KSeqWrapper::KSEQ_FILE) {
            // get last byte from filenames[fileIdx].c_str()
            FILE* fp = fopen(filenames[fileIdx].c_str(), "rb");
            if (fp == NULL) {
                Debug(Debug::ERROR) << "Cannot open file " << filenames[fileIdx] << "\n";
                EXIT(EXIT_FAILURE);
            }
            int res = fseek(fp, -1, SEEK_END);
            if (res != 0) {
                Debug(Debug::ERROR) << "Cannot seek at the end of file " << filenames[fileIdx] << "\n";
                EXIT(EXIT_FAILURE);
            }
            int lastChar = fgetc(fp);
            if (lastChar == EOF) {
                Debug(Debug::ERROR) << "Error reading from " << filenames[fileIdx] << "\n";
                EXIT(EXIT_FAILURE);
            }
            if (fclose(fp) != 0) {
                Debug(Debug::ERROR) << "Error closing " << filenames[fileIdx] << "\n";
                EXIT(EXIT_FAILURE);
            }
            if (lastChar != '\n') {
                Debug(Debug::WARNING) << "Last byte is not a newline. We recompute with --createdb-mode " << Parameters::SEQUENCE_SPLIT_MODE_HARD << "\n";
                resetIncorrectNewline = true;
            }
        }
        if (resetNotFile || resetIncorrectNewline) {
            par.createdbMode = Parameters::SEQUENCE_SPLIT_MODE_HARD;
            progress.reset(SIZE_MAX);
            hdrWriter.close();
            seqWriter.close();
            delete kseq;
            if (fclose(source) != 0) {
                Debug(Debug::ERROR) << "Cannot close file " << sourceFile << "\n";
                EXIT(EXIT_FAILURE);
            }
            if (sourceLookup != NULL) {
                for (size_t i = 0; i < shuffleSplits; ++i) {
                    sourceLookup[i].clear();
                }
            }
            goto redoComputation;
        }
        while (kseq->ReadEntry()) {
            progress.updateProgress();
            const KSeqWrapper::KSeqEntry &e = kseq->entry;
            if (e.name.l == 0) {
                Debug(Debug::ERROR) << "Fasta entry " << numEntriesInCurrFile << " is invalid\n";
                EXIT(EXIT_FAILURE);
            }

            DBKeyType id = static_cast<DBKeyType>(par.identifierOffset) + entries_num;
            if (dbType == -1) {
                // check for the first 10 sequences if they are nucleotide sequences
                if (sampleCount < 10 || (sampleCount % 100) == 0) {
                    if (sampleCount < testForNucSequence) {
                        if (sequenceLooksNucleotide(e.sequence.s, e.sequence.l)) {
                            isNuclCnt += true;
                        }
                    }
                    sampleCount++;
                }
            }

            if (par.createdbMode == Parameters::SEQUENCE_SPLIT_MODE_SOFT) {
                if (e.newlineCount != 1) {
                    if (e.newlineCount == 0) {
                        Debug(Debug::WARNING) << "Fasta entry " << numEntriesInCurrFile << " has no newline character\n";
                    } else if (e.newlineCount > 1) {
                        Debug(Debug::WARNING) << "Multiline fasta can not be combined with --createdb-mode " << Parameters::SEQUENCE_SPLIT_MODE_SOFT << "\n";
                    }
                    Debug(Debug::WARNING) << "We recompute with --createdb-mode " << Parameters::SEQUENCE_SPLIT_MODE_HARD << "\n";
                    par.createdbMode = Parameters::SEQUENCE_SPLIT_MODE_HARD;
                    progress.reset(SIZE_MAX);
                    hdrWriter.close();
                    seqWriter.close();
                    delete kseq;
                    if (fclose(source) != 0) {
                        Debug(Debug::ERROR) << "Cannot close file " << sourceFile << "\n";
                        EXIT(EXIT_FAILURE);
                    }
                    if (sourceLookup != NULL) {
                        for (size_t i = 0; i < shuffleSplits; ++i) {
                            sourceLookup[i].clear();
                        }
                    }
                    goto redoComputation;
                }
            }

            header.append(e.name.s, e.name.l);
            if (e.comment.l > 0) {
                header.append(" ", 1);
                header.append(e.comment.s, e.comment.l);
            }

            std::string headerId = Util::parseFastaHeader(header.c_str());
            if (headerId.empty()) {
                // An identifier is necessary for these two cases, so we should just give up
                Debug(Debug::WARNING) << "Cannot extract identifier from entry " << numEntriesInCurrFile << "\n";
            }
            header.push_back('\n');

            // Finally write down the entry
            unsigned int splitIdx = id % shuffleSplits;
            if (sourceLookup != NULL) {
                sourceLookup[splitIdx].emplace_back(fileIdx);
            }
            if (par.createdbMode == Parameters::SEQUENCE_SPLIT_MODE_SOFT) {
                // +2 to emulate the \n\0
                hdrWriter.writeIndexEntry(id, headerFileOffset + e.headerOffset, (e.sequenceOffset-e.headerOffset)+1, 0);
                seqWriter.writeIndexEntry(id, seqFileOffset + e.sequenceOffset, e.sequence.l+2, 0);
            } else if (par.createdbMode == Parameters::SEQUENCE_SPLIT_MODE_HARD
                       || par.createdbMode == Parameters::SEQUENCE_SPLIT_MODE_LENGTH_DESC) {
                hdrWriter.writeData(header.c_str(), header.length(), id, splitIdx);
                seqWriter.writeStart(splitIdx);
                seqWriter.writeAdd(e.sequence.s, e.sequence.l, splitIdx);
                seqWriter.writeAdd(&newline, 1, splitIdx);
                seqWriter.writeEnd(id, splitIdx, true);
            } else {
                // Add to batch, allocated sequence and masker and genrate submat on the fly
                batchEntries[batchPos].first.clear();
                //batchEntries[batchPos].first.reserve(e.sequence.l);
                for(size_t j = 0; j < e.sequence.l; j++){
                    batchEntries[batchPos].first.push_back(e.sequence.s[j]);
                }
//                batchEntries[batchPos].first.push_back('\n');
                //std::copy_n(e.sequence.s, e.sequence.l, batchEntries[batchPos].first.begin());
                batchEntries[batchPos].second.clear();
                batchEntries[batchPos].second.append(header.begin(), header.end());
                batchPos++;
                if(batchPos == BATCH_SIZE){
                    if(subMat == NULL){
                        if (isNuclCnt == sampleCount) {
                            subMat = new NucleotideMatrix(par.scoringMatrixFile.values.nucleotide().c_str(), 2.0, -0.0f);
                            dbType = Parameters::DBTYPE_NUCLEOTIDES;
                        } else{
                            subMat = new SubstitutionMatrix(par.scoringMatrixFile.values.aminoacid().c_str(), 2.0, -0.0f);
                            dbType = Parameters::DBTYPE_AMINO_ACIDS;
                        }
                    }
                    processSeqBatch(par, seqWriter, hdrWriter, subMat, dbType, masker, seqs,
                                    id - (batchPos - 1), batchEntries, batchPos, shuffleSplits,
                                    true);
                    batchPos = 0;
                }
            }

            entries_num++;
            numEntriesInCurrFile++;
            header.clear();
        }

        if(par.createdbMode == Parameters::SEQUENCE_SPLIT_MODE_GPU && batchPos > 0){
            if(subMat == NULL){
                if (isNuclCnt == sampleCount) {
                    subMat = new NucleotideMatrix(par.scoringMatrixFile.values.nucleotide().c_str(), 2.0, -0.0f);
                    dbType = Parameters::DBTYPE_NUCLEOTIDES;
                } else {
                    subMat = new SubstitutionMatrix(par.scoringMatrixFile.values.aminoacid().c_str(), 2.0, -0.0f);
                    dbType = Parameters::DBTYPE_AMINO_ACIDS;
                }
            }
            processSeqBatch(par, seqWriter, hdrWriter, subMat, dbType, masker, seqs,
                            (par.identifierOffset + entries_num) - (batchPos - 1), batchEntries, batchPos, shuffleSplits,
                            true);
            batchPos = 0;
        }

        if (numEntriesInCurrFile == 0) {
            Debug(Debug::WARNING) << "File " << sourceName << " is empty or invalid and was ignored\n";
        }

        delete kseq;
        if (filenames.size() > 1 && par.createdbMode == Parameters::SEQUENCE_SPLIT_MODE_SOFT) {
            size_t fileSize = FileUtil::getFileSize(filenames[fileIdx].c_str());
            headerFileOffset += fileSize;
            seqFileOffset += fileSize;
        }
    }
    Debug(Debug::INFO) << "\n";
    if (fclose(source) != 0) {
        Debug(Debug::ERROR) << "Cannot close file " << sourceFile << "\n";
        EXIT(EXIT_FAILURE);
    }

    // sort
    bool gpuCompatibleDB = false;
    Timer timer;
    const bool descendingLength = (par.createdbMode == Parameters::SEQUENCE_SPLIT_MODE_LENGTH_DESC);
    if(isSortedCreatedbMode(par.createdbMode)){
        gpuCompatibleDB = (par.createdbMode == Parameters::SEQUENCE_SPLIT_MODE_GPU);
        hdrWriter.closeFiles();
        seqWriter.closeFiles();
        size_t maxSplitBytes = 0;
        for (unsigned int i = 0; i < shuffleSplits; i++) {
            maxSplitBytes = std::max(maxSplitBytes,
                                     std::max(FileUtil::getFileSize(seqWriter.getDataFileNames()[i]),
                                              FileUtil::getFileSize(hdrWriter.getDataFileNames()[i])));
        }
        // one split is read whole into one buffer, so its size is the sort's resident footprint
        if (maxSplitBytes > Util::computeMemory(0) / 4) {
            Debug(Debug::WARNING) << "Largest split is " << maxSplitBytes
                                  << " byte and the sort loads one whole: raise --shuffle-splits above "
                                  << shuffleSplits << "\n";
        }
        for(unsigned int i = 0; i < shuffleSplits; i++){
            sortWithIndex(seqWriter.getDataFileNames()[i], seqWriter.getIndexFileNames()[i],
                          hdrWriter.getDataFileNames()[i], hdrWriter.getIndexFileNames()[i],
                          par.threads, descendingLength);
        }
        Debug(Debug::INFO) << "Sort single files in " << timer.lap() << "\n";
        std::string lookupFile = dataFile + ".lookup";

        timer.reset();
        mergeSequentialByJointIndex(seqWriter.getDataFileNames(), seqWriter.getIndexFileNames(),
                                    hdrWriter.getDataFileNames(), hdrWriter.getIndexFileNames(),
                                    seqWriter.getDataFileName(), seqWriter.getIndexFileName(),
                                    hdrWriter.getDataFileName(), hdrWriter.getIndexFileName(),
                                    lookupFile.c_str(), parseFilesInParallel ? NULL : sourceLookup,
                                    fileKeyStarts.empty() ? NULL : &fileKeyStarts,
                                    entries_num, shuffleSplits, descendingLength, gpuCompatibleDB);
        Debug(Debug::INFO) << "Merge all files " << timer.lap() << "\n";
        hdrWriter.clearMemory();
        seqWriter.clearMemory();
        //TODO we cannot have compressed seq. dbs anymore if wanted ot have GPU compatible dbs
    } else {
        hdrWriter.close(true, false);
        seqWriter.close(true, false);
        if (par.shuffleDatabase == true) {
            DBWriter::createRenumberedDB(dataFile, indexFile, "", "", DBReader<DBKeyType>::LINEAR_ACCCESS);
            DBWriter::createRenumberedDB(hdrDataFile, hdrIndexFile, "", "", DBReader<DBKeyType>::LINEAR_ACCCESS);
        }
        if (par.createdbMode == Parameters::SEQUENCE_SPLIT_MODE_SOFT) {
            if (filenames.size() == 1) {
                FileUtil::symlinkAbs(filenames[0], dataFile);
                FileUtil::symlinkAbs(filenames[0], hdrDataFile);
            } else {
                for (size_t fileIdx = 0; fileIdx < filenames.size(); fileIdx++) {
                    FileUtil::symlinkAbs(filenames[fileIdx], dataFile + "." + SSTR(fileIdx));
                    FileUtil::symlinkAbs(filenames[fileIdx], hdrDataFile + "." + SSTR(fileIdx));
                }
            }
        }
        if (par.writeLookup == true) {
            DBReader<DBKeyType> readerHeader(hdrDataFile.c_str(), hdrIndexFile.c_str(), 1, DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_INDEX);
            readerHeader.open(DBReader<DBKeyType>::NOSORT);
            // create lookup file
            std::string lookupFile = dataFile + ".lookup";
            FILE* file = FileUtil::openAndDelete(lookupFile.c_str(), "w");
            std::string buffer;
            buffer.reserve(2048);
            size_t splitIdx = 0;
            size_t splitCounter = 0;
            DBReader<DBKeyType>::LookupEntry entry;
            for (size_t id = 0; id < readerHeader.getSize(); id++) {
                size_t splitSize = sourceLookup[splitIdx].size();
                if (splitSize == 0 || splitCounter > sourceLookup[splitIdx].size() - 1) {
                    splitIdx++;
                    splitCounter = 0;
                }
                char *header = readerHeader.getData(id, 0);
                entry.id = id;
                entry.entryName = Util::parseFastaHeader(header);
                if (entry.entryName.empty()) {
                    Debug(Debug::WARNING) << "Cannot extract identifier from entry " << entries_num << "\n";
                }
                entry.fileNumber = sourceLookup[splitIdx][splitCounter];
                readerHeader.lookupEntryToBuffer(buffer, entry);
                int written = fwrite(buffer.c_str(), sizeof(char), buffer.size(), file);
                if (written != (int)buffer.size()) {
                    Debug(Debug::ERROR) << "Cannot write to lookup file " << lookupFile << "\n";
                    EXIT(EXIT_FAILURE);
                }
                buffer.clear();
                splitCounter++;
            }
            if (fclose(file) != 0) {
                Debug(Debug::ERROR) << "Cannot close file " << lookupFile << "\n";
                EXIT(EXIT_FAILURE);
            }
            readerHeader.close();
        }
    }

    if (dbType == -1) {
        if (isNuclCnt == sampleCount) {
            dbType = Parameters::DBTYPE_NUCLEOTIDES;
        } else {
            dbType = Parameters::DBTYPE_AMINO_ACIDS;
        }
    }
    if(gpuCompatibleDB){
        dbType = DBReader<DBKeyType>::setExtendedDbtype(dbType, Parameters::DBTYPE_EXTENDED_GPU);
    }
    DBWriter::writeDbtypeFile(seqWriter.getDataFileName(), dbType ,par.compressed);
    DBWriter::writeDbtypeFile(hdrWriter.getDataFileName(), Parameters::DBTYPE_GENERIC_DB, par.compressed);

    Debug(Debug::INFO) << "Database type: " << Parameters::getDbTypeName(dbType) << "\n";
    if (dbInput == true) {
        reader->close();
        delete reader;
    }

    if (entries_num == 0) {
        Debug(Debug::ERROR) << "The input files have no entry: ";
        for (size_t fileIdx = 0; fileIdx < filenames.size(); fileIdx++) {
            Debug(Debug::ERROR) << " - " << filenames[fileIdx] << "\n";
        }
        Debug(Debug::ERROR) << "Please check your input files. Only files in fasta/fastq[.gz|.bz2|.zst] are supported\n";
        EXIT(EXIT_FAILURE);
    }

    if(masker[0] != NULL){
        for(int i = 0; i < par.threads; i++){
            delete masker[i];
            delete seqs[i];
        }
    }
    delete [] masker;
    delete [] seqs;

    if (subMat != NULL) {
        delete subMat;
    }

    delete[] sourceLookup;

    return EXIT_SUCCESS;
}
