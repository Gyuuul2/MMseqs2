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
#include "MMseqsNode.h"
#include "BatchReader.h"

#include "zstd/contrib/seekable_format/zstd_seekable.h"
#include <sys/stat.h>
#include <algorithm>

#ifdef OPENMP
#include <omp.h>
#endif

// Sort the data file in-place using your index array
int sortWithIndex(const char *dataFileSeq,
                  const char *indexFileSeq,
                  const char *dataFileHeader,
                  const char *indexFileHeader)
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

    FILE *fin = FileUtil::openFileOrDie(dataFileSeq, "rb", true);
    char * buf = new char [seqfileSize];
    size_t got = fread(buf, 1, seqfileSize, fin);
    if (got != seqfileSize) {
        Debug(Debug::ERROR) << "short read: " << dataFileSeq << "\n";
        EXIT(EXIT_FAILURE);
    }
    fclose(fin);
    // needed to keep the information in what line the id was originally
    for (size_t i = 0; i < reader.getSize(); i++) {
        index[i].id = i;
    }

    SORT_PARALLEL(index, index + reader.getSize(), DBReader<DBKeyType>::Index::compareByLength);

    FILE *seqOut = FileUtil::openFileOrDie(dataFileSeq, "wb", true);
    setvbuf(seqOut, NULL, _IOFBF, 1024*1024*50);
    size_t offset = 0;
    for (size_t i = 0; i < reader.getSize(); i++) {
        size_t written = fwrite(buf + index[i].offset, 1, index[i].length, seqOut);
        if (written != static_cast<size_t>(index[i].length)) {
            Debug(Debug::ERROR) << "Can not write to data file " << dataFileSeq << "\n";
            EXIT(EXIT_FAILURE);
        }
        index[i].offset = offset;
        offset += written;
    }
    fclose(seqOut);

    if (stat(dataFileHeader, &st) != 0) {
        Debug(Debug::ERROR) << "stat failed: " << dataFileHeader << "\n";
        EXIT(EXIT_FAILURE);
    }
    const size_t headFileSize = st.st_size;
    if(headFileSize > seqfileSize){
        delete [] buf;
        buf = new char [headFileSize];
    }
    fin = FileUtil::openFileOrDie(dataFileHeader, "rb", true);
    got = fread(buf, 1, headFileSize, fin);
    if (got != headFileSize) {
        Debug(Debug::ERROR) << "short read: " << dataFileHeader << "\n";
        EXIT(EXIT_FAILURE);
    }
    fclose(fin);

    DBReader<DBKeyType> header(dataFileHeader, indexFileHeader, 1, DBReader<DBKeyType>::USE_INDEX);
    header.open(DBReader<DBKeyType>::HARDNOSORT);
    DBReader<DBKeyType>::Index *headerIndex = header.getIndex();
    FILE *headerout = FileUtil::openFileOrDie(dataFileHeader, "wb", true);
    setvbuf(headerout, NULL, _IOFBF, 1024*1024*50);
    offset = 0;
    for (size_t i = 0; i < header.getSize(); i++) {
        size_t sortedId = static_cast<size_t>(index[i].id);
        size_t written = fwrite(buf + headerIndex[sortedId].offset, 1, headerIndex[sortedId].length, headerout);
        // reconstruct old id
        index[i].id = headerIndex[sortedId].id;
        if (written != static_cast<size_t>(headerIndex[sortedId].length)) {
            Debug(Debug::ERROR) << "Can not write to data file " << dataFileHeader << "\n";
            EXIT(EXIT_FAILURE);
        }
        headerIndex[sortedId].offset = offset;
        offset += written;
    }
    fclose(headerout);
    delete [] buf;

    SORT_PARALLEL(index, index + reader.getSize(), DBReader<DBKeyType>::Index::compareByOffset);
    {
        std::string tmpIndex = std::string(indexFileSeq) + ".tmp";
        FILE *indexout = FileUtil::openFileOrDie(tmpIndex.c_str(), "wb", false);
        DBWriter::writeIndex(indexout, reader.getSize(), index);
        fclose(indexout);
        FileUtil::move(tmpIndex.c_str(), indexFileSeq);

        std::string tmpHeaderIndex = std::string(indexFileHeader) + ".tmp";
        FILE *headerIndexOut = FileUtil::openFileOrDie(tmpHeaderIndex.c_str(), "wb", false);
        DBWriter::writeIndex(headerIndexOut, header.getSize(), headerIndex);
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
        size_t totalEntries,
        size_t shuffleSplits
) {
    struct JointEntry {
        unsigned int fileIdx;
        DBKeyType id;
        unsigned int length;
        JointEntry(unsigned int fileIdx, DBKeyType id, unsigned int length)
            : fileIdx(fileIdx), id(id), length(length) {}

        bool operator<(JointEntry const &o) const {
            if (length != o.length){
                return length < o.length;
            }
            return id < o.id;
        }
    };

    std::vector<JointEntry> joint;
    joint.reserve(totalEntries);
    size_t maxLen = 0;
    for (size_t i = 0; i < shuffleSplits; i++) {
        DBReader<DBKeyType> reader(
                dataFiles[i],
                indexFiles[i],
                1,
                DBReader<DBKeyType>::USE_INDEX
        );
        reader.open(DBReader<DBKeyType>::HARDNOSORT);
        DBReader<DBKeyType>::Index* index = reader.getIndex();
        for(size_t j = 0; j < reader.getSize(); j++){
            joint.emplace_back((unsigned int)i, index[j].id, index[j].length);
            maxLen = std::max(maxLen, static_cast<size_t>(index[j].length));
        }
        reader.close();
    }

    SORT_PARALLEL(joint.begin(), joint.end());

    // 4) Open each data file once (no fseek later)
    std::vector<FILE*> inFileSeq(shuffleSplits);
    std::vector<FILE*> inFileHeader(shuffleSplits);

    for (size_t i = 0; i < shuffleSplits; i++) {
        inFileSeq[i] = FileUtil::openFileOrDie(
            dataFiles[i], "rb", true
        );
        inFileHeader[i] = FileUtil::openFileOrDie(
            dataFilesHeader[i], "rb", true
        );
    }

    FILE *fout = FileUtil::openAndDelete(outDataFile, "wb");
    setvbuf(fout, NULL, _IOFBF, 1024*1024*50);
    FILE *idxOut = FileUtil::openAndDelete(outIndexFile, "wb");
    setvbuf(idxOut, NULL, _IOFBF, 1024*1024*50);
    FILE *foutHeader  = FileUtil::openAndDelete(outHeaderDataFile, "wb");
    setvbuf(foutHeader, NULL, _IOFBF, 1024*1024*50);
    FILE *foutLookup  = FileUtil::openAndDelete(outLookupFile, "wb");
    setvbuf(foutLookup, NULL, _IOFBF, 1024*1024*50);
    FILE *idxOutHeader = FileUtil::openAndDelete(outHeaderIndexFile, "wb");
    setvbuf(idxOutHeader, NULL, _IOFBF, 1024*1024*50);

    struct FileBuffer {
        std::vector<char> buffer;
        size_t pos, end;
        FILE *file;
        FileBuffer(FILE* f) : buffer(1024 * 1024), pos(0), end(0), file(f) {}

        bool refill() {
            pos = 0;
            // read 1 MB
            end = fread(buffer.data(), 1, 1024 * 1024, file);
            return end != 0;
        }

        bool getChar(char &ch) {
            if (pos >= end && !refill()) { return false; }
            ch = buffer[pos++];
            return true;
        }
    };

    // Create buffers for each header file
    std::vector<FileBuffer> headerBuffers;
    std::vector<char> writeHeaderBuf;
    headerBuffers.reserve(shuffleSplits);
    for (size_t i = 0; i < shuffleSplits; i++) {
        headerBuffers.emplace_back(inFileHeader[i]);
    }
    size_t mergedOffset = 0;
    size_t mergedOffsetHeader = 0;
    std::vector<char> scratch(maxLen);
    DBReader<DBKeyType>::Index entry;
    DBReader<DBKeyType>::LookupEntry lookupEntry;

    char indexBuffer[1024];
    std::string lookupBuffer;
    const int ALIGN = 4;
    const char pad_buffer[4] = {20, 20, 20, 20}; // pre-filled buffer
    for(size_t i = 0; i < joint.size(); i++){
        JointEntry qe = joint[i];
        size_t read = fread(scratch.data(), 1, qe.length, inFileSeq[qe.fileIdx]);
        if (UNLIKELY(read != static_cast<size_t>(qe.length))) {
            Debug(Debug::ERROR) << "Can not read to data file " << dataFiles[qe.fileIdx] << "\n";
            EXIT(EXIT_FAILURE);
        }
        size_t written = fwrite(scratch.data(), 1, qe.length, fout);
        const size_t sequencepadding = (qe.length % ALIGN == 0) ? 0 : ALIGN - qe.length % ALIGN;
        written +=  fwrite(pad_buffer, 1, sequencepadding, fout);
        if (UNLIKELY(written != qe.length + sequencepadding)) {
            Debug(Debug::ERROR) << "Can not write to data file " << outDataFile << "\n";
            EXIT(EXIT_FAILURE);
        }

        writeHeaderBuf.clear();
        char ch;
        do {
            if (!headerBuffers[qe.fileIdx].getChar(ch)) {
                Debug(Debug::ERROR) << "Unexpected EOF in header file " <<  qe.fileIdx << "\n";
                EXIT(EXIT_FAILURE);
            }
            writeHeaderBuf.push_back(ch);
        } while (ch != '\0');

        lookupEntry.id = i;
        lookupEntry.entryName = Util::parseFastaHeader(writeHeaderBuf.data());
        if (lookupEntry.entryName.empty()) {
            Debug(Debug::WARNING) << "Cannot extract identifier from entry " << lookupEntry.id  << "\n";
        }
        lookupEntry.fileNumber = sourceLookup[qe.fileIdx][(qe.id - qe.fileIdx) / 32];
        lookupBuffer.clear();
        DBReader<DBKeyType>::lookupEntryToBuffer(lookupBuffer, lookupEntry);
        written = fwrite(lookupBuffer.data(), 1, lookupBuffer.size(), foutLookup);
        if (UNLIKELY(written != lookupBuffer.size())) {
            Debug(Debug::ERROR) << "Can not write to lookup file " << outLookupFile << "\n";
            EXIT(EXIT_FAILURE);
        }
        written = fwrite(writeHeaderBuf.data(), 1, writeHeaderBuf.size(), foutHeader);
        if (UNLIKELY(written != writeHeaderBuf.size())) {
            Debug(Debug::ERROR) << "Can not write to header file " << outHeaderDataFile << "\n";
            EXIT(EXIT_FAILURE);
        }

        entry.offset = mergedOffset;
        // + 2 is needed for newline and null character
        entry.length = qe.length + 2;
        entry.id = i;
        DBWriter::writeIndexEntryToFile(idxOut, indexBuffer, entry);
        entry.length = writeHeaderBuf.size();
        entry.offset = mergedOffsetHeader;
        DBWriter::writeIndexEntryToFile(idxOutHeader, indexBuffer, entry);
        mergedOffset += qe.length + sequencepadding;
        mergedOffsetHeader += writeHeaderBuf.size();
    }

    fclose(fout);
    fclose(idxOut);
    fclose(foutHeader);
    fclose(foutLookup);
    fclose(idxOutHeader);
    for (size_t i = 0; i < inFileSeq.size(); i++) {
        fclose(inFileSeq[i]);
        fclose(inFileHeader[i]);
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
                     unsigned int shuffleSplits){
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

enum DataFileType { SEQUENCE_DATA, HEADER_DATA, HEADER_OFFSETS, DATA_FILE_TYPES };
static const char *const dataFileSuffixes[DATA_FILE_TYPES] = {".", "_h.", "_h.index."};

struct LengthBin {
    size_t length;
    size_t sequenceCount;
    size_t headerBytes;
    size_t startKey;
    size_t offsets[DATA_FILE_TYPES];
    size_t file;
};

struct InputPiece {
    size_t file;
    size_t start;
    size_t end;
};

// a segment of the bin of a length goes to the node of the length as { length, offset in the bin, size, stored size } then its bytes, compressed with zstd if that is smaller
static const size_t SEGMENT_HEADER_SIZE = 4 * sizeof(uint64_t);
// the most bytes of a segment, so the buffers of the exchange blocks stay small
static const size_t MAX_SEGMENT_SIZE = 64 * 1024;

// the most bytes of the segment headers of a flush of entryCount entries with dataSize bytes and lengths below lengthCount
static size_t getSegmentHeadersSize(size_t entryCount, size_t lengthCount, size_t dataSize) {
    return (DATA_FILE_TYPES * (std::min(entryCount, lengthCount) + 1) + dataSize / MAX_SEGMENT_SIZE) * SEGMENT_HEADER_SIZE;
}

static std::vector<LengthBin> makeLengthBins(const std::vector<size_t> &sequenceCount, const std::vector<size_t> &headerBytes) {
    std::vector<LengthBin> histogram;
    for (size_t length = sequenceCount.size(); length-- > 0; ) {
        if (sequenceCount[length] > 0) {
            LengthBin bin = {length, sequenceCount[length], headerBytes[length], 0, {}, 0};
            histogram.push_back(bin);
        }
    }
    return histogram;
}

static std::vector<LengthBin> mergeLengthHistograms(const std::vector<std::vector<LengthBin>> &histograms) {
    size_t maxLength = 0;
    for (size_t i = 0; i < histograms.size(); i++) {
        if (histograms[i].empty() == false) {
            maxLength = std::max(maxLength, histograms[i][0].length);
        }
    }
    std::vector<size_t> sequenceCount(maxLength + 1, 0);
    std::vector<size_t> headerBytes(maxLength + 1, 0);
    for (size_t i = 0; i < histograms.size(); i++) {
        for (size_t j = 0; j < histograms[i].size(); j++) {
            sequenceCount[histograms[i][j].length] += histograms[i][j].sequenceCount;
            headerBytes[histograms[i][j].length] += histograms[i][j].headerBytes;
        }
    }
    return makeLengthBins(sequenceCount, headerBytes);
}

static std::vector<LengthBin> getLengthToBin(const std::vector<LengthBin> &histogram) {
    std::vector<LengthBin> lengthToBin(histogram.empty() ? 0 : histogram[0].length + 1);
    for (size_t i = 0; i < histogram.size(); i++) {
        lengthToBin[histogram[i].length] = histogram[i];
    }
    return lengthToBin;
}

// copies data behind the written bytes of a bin, leaving room for the headers of its segments
static void copyToBin(char *buffer, size_t &written, const char *data, size_t size) {
    while (size > 0) {
        const size_t bytes = std::min(size, MAX_SEGMENT_SIZE - written % MAX_SEGMENT_SIZE);
        memcpy(buffer + (written / MAX_SEGMENT_SIZE + 1) * SEGMENT_HEADER_SIZE + written, data, bytes);
        written += bytes;
        data += bytes;
        size -= bytes;
    }
}

// a length bin of a flush in the exchange file, the flushes of a node in piece order: where its bytes of every data file type start and end
struct SentBin {
    uint64_t flush;
    uint64_t length;
    uint64_t sequenceCount;
    uint64_t headerBytes;
    uint64_t starts[DATA_FILE_TYPES];
    uint64_t ends[DATA_FILE_TYPES];

    // the order of the table, so that the bins of a range of lengths follow each other
    static bool compareByLength(const SentBin &first, const SentBin &second) {
        if (first.length != second.length) {
            return first.length > second.length;
        }
        return first.flush < second.flush;
    }

    static bool compareByFlush(const SentBin &first, const SentBin &second) {
        if (first.flush != second.flush) {
            return first.flush < second.flush;
        }
        return first.length > second.length;
    }
};

// a flush of a piece at offset of the exchange file, its bins longer first with offsets from there
struct SentFlush {
    size_t piece;
    size_t pieceFlush;
    size_t offset;
    std::vector<SentBin> bins;

    static bool compareByPiece(const SentFlush &first, const SentFlush &second) {
        if (first.piece != second.piece) {
            return first.piece < second.piece;
        }
        return first.pieceFlush < second.pieceFlush;
    }
};

static bool isLongerBin(const LengthBin &bin, size_t length) {
    return bin.length > length;
}

// sorts the entries by length into segments of sortedBuffer, compresses them into buffer and appends them to the exchange file; a header offset counts from the headers of its length in the flush
static void flushBufferToExchange(char *buffer, size_t bufferPos, char *sortedBuffer, ZSTD_CCtx *cctx, size_t lengthCount, size_t piece, size_t pieceFlush,
                                  FILE *exchangeFile, const std::string &exchangeFileName, std::vector<SentFlush> &sentFlushes, size_t &exchangeBytes) {
    std::vector<size_t> binSizes(DATA_FILE_TYPES * lengthCount, 0);
    for (size_t pos = 0; pos < bufferPos; ) {
        const size_t seqSize = strlen(buffer + pos) + 1;
        const size_t hdrSize = strlen(buffer + pos + seqSize) + 1;
        binSizes[SEQUENCE_DATA * lengthCount + seqSize - 2] += seqSize;
        binSizes[HEADER_DATA * lengthCount + seqSize - 2] += hdrSize;
        binSizes[HEADER_OFFSETS * lengthCount + seqSize - 2] += sizeof(uint64_t);
        pos += seqSize + hdrSize;
    }
    SentFlush flush;
    flush.piece = piece;
    flush.pieceFlush = pieceFlush;
    for (size_t length = lengthCount; length-- > 0; ) {
        if (binSizes[SEQUENCE_DATA * lengthCount + length] > 0) {
            SentBin bin = {0, length, binSizes[SEQUENCE_DATA * lengthCount + length] / (length + 2), binSizes[HEADER_DATA * lengthCount + length], {}, {}};
            flush.bins.push_back(bin);
        }
    }
    std::vector<size_t> binStarts(DATA_FILE_TYPES * lengthCount, 0);
    size_t sortedPos = 0;
    for (size_t fileType = 0; fileType < DATA_FILE_TYPES; fileType++) {
        for (size_t i = 0; i < flush.bins.size(); i++) {
            const size_t bin = fileType * lengthCount + flush.bins[i].length;
            binStarts[bin] = sortedPos;
            for (size_t segment = 0; segment * MAX_SEGMENT_SIZE < binSizes[bin]; segment++) {
                const uint64_t segmentHeader[4] = { flush.bins[i].length, segment * MAX_SEGMENT_SIZE, std::min(MAX_SEGMENT_SIZE, binSizes[bin] - segment * MAX_SEGMENT_SIZE) };
                memcpy(sortedBuffer + sortedPos, segmentHeader, SEGMENT_HEADER_SIZE);
                sortedPos += SEGMENT_HEADER_SIZE + segmentHeader[2];
            }
        }
    }
    std::vector<size_t> binWritten(DATA_FILE_TYPES * lengthCount, 0);
    for (size_t pos = 0; pos < bufferPos; ) {
        const size_t seqSize = strlen(buffer + pos) + 1;
        const size_t hdrSize = strlen(buffer + pos + seqSize) + 1;
        const size_t seqBin = SEQUENCE_DATA * lengthCount + seqSize - 2;
        const size_t hdrBin = HEADER_DATA * lengthCount + seqSize - 2;
        const size_t offsetBin = HEADER_OFFSETS * lengthCount + seqSize - 2;
        const uint64_t headerOffset = binWritten[hdrBin];
        copyToBin(sortedBuffer + binStarts[seqBin], binWritten[seqBin], buffer + pos, seqSize);
        copyToBin(sortedBuffer + binStarts[hdrBin], binWritten[hdrBin], buffer + pos + seqSize, hdrSize);
        copyToBin(sortedBuffer + binStarts[offsetBin], binWritten[offsetBin], reinterpret_cast<const char *>(&headerOffset), sizeof(uint64_t));
        pos += seqSize + hdrSize;
    }
    // buffer is free once the entries are sorted; the segments are compressed into it
    size_t compressedPos = 0;
    size_t pos = 0;
    for (size_t fileType = 0; fileType < DATA_FILE_TYPES; fileType++) {
        for (size_t i = 0; i < flush.bins.size(); i++) {
            flush.bins[i].starts[fileType] = compressedPos;
            const size_t bin = fileType * lengthCount + flush.bins[i].length;
            for (size_t segment = 0; segment * MAX_SEGMENT_SIZE < binSizes[bin]; segment++) {
                uint64_t segmentHeader[4];
                memcpy(segmentHeader, sortedBuffer + pos, SEGMENT_HEADER_SIZE);
                const char *data = sortedBuffer + pos + SEGMENT_HEADER_SIZE;
                char *compressed = buffer + compressedPos + SEGMENT_HEADER_SIZE;
                size_t storedSize = ZSTD_compressCCtx(cctx, compressed, segmentHeader[2] - 1, data, segmentHeader[2], 1);
                if (ZSTD_isError(storedSize)) {
                    memcpy(compressed, data, segmentHeader[2]);
                    storedSize = segmentHeader[2];
                }
                segmentHeader[3] = storedSize;
                memcpy(buffer + compressedPos, segmentHeader, SEGMENT_HEADER_SIZE);
                compressedPos += SEGMENT_HEADER_SIZE + storedSize;
                pos += SEGMENT_HEADER_SIZE + segmentHeader[2];
            }
            flush.bins[i].ends[fileType] = compressedPos;
        }
    }
#pragma omp critical
    {
        flush.offset = exchangeBytes;
        exchangeBytes += compressedPos;
        sentFlushes.push_back(flush);
    }
    FileUtil::pwriteAll(fileno(exchangeFile), exchangeFileName.c_str(), buffer, compressedPos, flush.offset);
}

static int createdbLinfinity(Parameters &par, const std::vector<std::string> &filenames, const std::string &dataFile) {
    const std::string planPrefix = dataFile + ".splitplan.";
    if (MMseqsNode::isMaster()) {
        std::vector<size_t> fileSizes(filenames.size());
        size_t totalSize = 0;
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads) reduction(+:totalSize)
        for (size_t i = 0; i < filenames.size(); i++) {
            fileSizes[i] = FileUtil::getFileSize(filenames[i]);
            if (fileSizes[i] == SIZE_MAX) {
                Debug(Debug::ERROR) << "File " << filenames[i] << " does not exist\n";
                EXIT(EXIT_FAILURE);
            }
            totalSize += fileSizes[i];
        }
        const size_t pieceSize = std::max<size_t>(1, totalSize / (MMseqsNode::count * par.threads));
        std::vector<std::vector<size_t>> fileCuts(filenames.size(), std::vector<size_t>(1, 0));
        size_t unsplitFiles = 0;
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads) reduction(+:unsplitFiles)
        for (size_t i = 0; i < filenames.size(); i++) {
            std::vector<size_t> &cuts = fileCuts[i];
            if (fileSizes[i] > pieceSize) {
                bool isSplittable = Util::endsWith(".gz", filenames[i]) == false && Util::endsWith(".bz2", filenames[i]) == false;
                if (isSplittable) {
                    KSeqWrapper *kseq = KSeqFactory(filenames[i].c_str());
                    isSplittable = kseq->ReadEntry() && kseq->entry.qual.l == 0;
                    delete kseq;
                }
                // a pzstd file starts with a skippable frame of the size of the next frame, the reader of a piece finds the frames from its cuts
                if (isSplittable && Util::endsWith(".zst", filenames[i]) == true) {
                    FILE *handle = FileUtil::openFileOrDie(filenames[i].c_str(), "r", true);
                    uint32_t header[3];
                    isSplittable = fread(header, sizeof(uint32_t), 3, handle) == 3 && header[0] == ZSTD_MAGIC_SKIPPABLE_START && header[1] == 4;
                    if (fclose(handle) != 0) {
                        Debug(Debug::ERROR) << "Cannot close file " << filenames[i] << "\n";
                        EXIT(EXIT_FAILURE);
                    }
                }
                for (size_t cut = pieceSize; isSplittable && cut < fileSizes[i]; cut += pieceSize) {
                    cuts.push_back(cut);
                }
                unsplitFiles += isSplittable ? 0 : 1;
            }
            cuts.push_back(fileSizes[i]);
        }
        if (unsplitFiles > 0) {
            Debug(Debug::WARNING) << unsplitFiles << " input files are larger than " << pieceSize << " bytes but cannot be split, so fewer threads and one node read them. "
                                  << "Give FASTA uncompressed or compressed with pzstd\n";
        }
        std::vector<InputPiece> pieces;
        std::vector<size_t> pieceOffsets(1, 0);
        for (size_t i = 0; i < filenames.size(); i++) {
            for (size_t j = 0; j + 1 < fileCuts[i].size(); j++) {
                InputPiece piece = {i, fileCuts[i][j], fileCuts[i][j + 1]};
                pieces.push_back(piece);
                pieceOffsets.push_back(pieceOffsets.back() + (piece.end - piece.start));
            }
        }
        const size_t pieceCount = pieces.size();
        std::vector<size_t> nodePieceOffsets(MMseqsNode::count + 1, pieceCount);
        nodePieceOffsets[0] = 0;
        for (unsigned int node = 0; node + 1 < MMseqsNode::count; node++) {
            const size_t targetSize = totalSize / MMseqsNode::count * (node + 1);
            const size_t maxPieceEnd = (pieceCount >= MMseqsNode::count) ? pieceCount - (MMseqsNode::count - node - 1) : pieceCount;
            size_t nearestEnd = std::lower_bound(pieceOffsets.begin(), pieceOffsets.end(), targetSize) - pieceOffsets.begin();
            if (nearestEnd > 0 && targetSize - pieceOffsets[nearestEnd - 1] <= pieceOffsets[nearestEnd] - targetSize) {
                nearestEnd--;
            }
            nodePieceOffsets[node + 1] = std::min(std::max(nearestEnd, nodePieceOffsets[node] + 1), maxPieceEnd);
        }
        for (unsigned int node = 0; node < MMseqsNode::count; node++) {
            std::string planData;
            for (size_t i = nodePieceOffsets[node]; i < nodePieceOffsets[node + 1]; i++) {
                planData += SSTR(pieces[i].file) + "\t" + SSTR(pieces[i].start) + "\t" + SSTR(pieces[i].end) + "\n";
            }
            MMseqsNode::publishRunFile(planPrefix + SSTR(node), planData);
        }
    }
    const std::string planFile = planPrefix + SSTR(MMseqsNode::rank);
    const std::string nodePlanData = MMseqsNode::waitForRunFile(planFile);
    char *nodePlanDataChar = (char *) nodePlanData.c_str();
    std::vector<InputPiece> nodePieces;
    size_t currPos = 0;
    const char *cols[3];
    while (currPos < nodePlanData.size()) {
        Util::getWordsOfLine(nodePlanDataChar, cols, 3);
        InputPiece piece = {Util::fast_atoi<size_t>(cols[0]), Util::fast_atoi<size_t>(cols[1]), Util::fast_atoi<size_t>(cols[2])};
        nodePieces.push_back(piece);
        nodePlanDataChar = Util::skipLine(nodePlanDataChar);
        currPos = nodePlanDataChar - nodePlanData.c_str();
    }
    FileUtil::remove(planFile.c_str());
    if (MMseqsNode::isMaster()) {
        Debug(Debug::INFO) << "Converting sequences from " << filenames.size() << " files on " << MMseqsNode::count << " nodes\n";
    }
    std::string phaseValues;
    Timer timer;
    const std::string exchange = dataFile + "_sent_entries";
    const std::string exchangeFile = exchange + "." + SSTR(MMseqsNode::rank);
    FILE *exchangeHandle = FileUtil::openAndDelete(exchangeFile.c_str(), "w");
    std::vector<SentFlush> sentFlushes;
    size_t exchangeBytes = 0;
    size_t emptySequences = 0;
    const size_t activeThreads = std::max<size_t>(1, std::min<size_t>(par.threads, nodePieces.size()));
    const size_t memoryLimit = Util::computeMemory(0);
    const size_t bufferSize = static_cast<size_t>(memoryLimit * 0.95) / (2 * activeThreads);
#pragma omp parallel num_threads(activeThreads) reduction(+:emptySequences)
    {
        char *buffer = new(std::nothrow) char[bufferSize];
        Util::checkAllocation(buffer, "Can not allocate buffer memory in createdb");
        char *sortedBuffer = new(std::nothrow) char[bufferSize];
        Util::checkAllocation(sortedBuffer, "Can not allocate sortedBuffer memory in createdb");
        ZSTD_CCtx *cctx = ZSTD_createCCtx();
        if (cctx == NULL) {
            Debug(Debug::ERROR) << "ZSTD_createCCtx() error\n";
            EXIT(EXIT_FAILURE);
        }
#pragma omp for schedule(dynamic, 1)
        for (size_t i = 0; i < nodePieces.size(); i++) {
            size_t bufferPos = 0;
            size_t entryCount = 0;
            size_t lengthCount = 0;
            size_t pieceFlush = 0;
            KSeqWrapper *kseq = KSeqFactory(filenames[nodePieces[i].file].c_str(), nodePieces[i].start, nodePieces[i].end);
            while (kseq->ReadEntry()) {
                const KSeqWrapper::KSeqEntry &e = kseq->entry;
                const size_t nameLength = strlen(e.name.s);
                if (nameLength == 0) {
                    Debug(Debug::ERROR) << "Fasta entry in " << filenames[nodePieces[i].file] << " is invalid\n";
                    EXIT(EXIT_FAILURE);
                }
                if (e.sequence.l == 0) {
                    emptySequences++;
                    continue;
                }
                const size_t entrySize = e.sequence.l + nameLength + 4;
                const size_t dataSize = bufferPos + entrySize + (entryCount + 1) * sizeof(uint64_t);
                if (dataSize + getSegmentHeadersSize(entryCount + 1, std::max(lengthCount, e.sequence.l + 1), dataSize) > bufferSize) {
                    if (entrySize + sizeof(uint64_t) + getSegmentHeadersSize(1, e.sequence.l + 1, entrySize + sizeof(uint64_t)) > bufferSize) {
                        Debug(Debug::ERROR) << "Not enough memory for entry " << e.name.s << "\n";
                        EXIT(EXIT_FAILURE);
                    }
                    flushBufferToExchange(buffer, bufferPos, sortedBuffer, cctx, lengthCount, i, pieceFlush++, exchangeHandle, exchangeFile, sentFlushes, exchangeBytes);
                    bufferPos = 0;
                    entryCount = 0;
                    lengthCount = 0;
                }
                memcpy(buffer + bufferPos, e.sequence.s, e.sequence.l);
                std::replace(buffer + bufferPos, buffer + bufferPos + e.sequence.l, '\0', 'X');
                bufferPos += e.sequence.l;
                buffer[bufferPos++] = '\n';
                buffer[bufferPos++] = '\0';
                memcpy(buffer + bufferPos, e.name.s, nameLength);
                bufferPos += nameLength;
                buffer[bufferPos++] = '\n';
                buffer[bufferPos++] = '\0';
                lengthCount = std::max(lengthCount, e.sequence.l + 1);
                entryCount++;
            }
            if (bufferPos > 0) {
                flushBufferToExchange(buffer, bufferPos, sortedBuffer, cctx, lengthCount, i, pieceFlush, exchangeHandle, exchangeFile, sentFlushes, exchangeBytes);
            }
            delete kseq;
        }
        Util::releasePages(buffer, bufferSize);
        Util::releasePages(sortedBuffer, bufferSize);
        delete[] buffer;
        delete[] sortedBuffer;
        ZSTD_freeCCtx(cctx);
    }
    // the table of the bins of all flushes, numbered in piece order, follows the flushes in the exchange file
    SORT_SERIAL(sentFlushes.begin(), sentFlushes.end(), SentFlush::compareByPiece);
    std::vector<SentBin> sentBins;
    for (size_t i = 0; i < sentFlushes.size(); i++) {
        for (size_t j = 0; j < sentFlushes[i].bins.size(); j++) {
            SentBin bin = sentFlushes[i].bins[j];
            bin.flush = i;
            for (size_t fileType = 0; fileType < DATA_FILE_TYPES; fileType++) {
                bin.starts[fileType] += sentFlushes[i].offset;
                bin.ends[fileType] += sentFlushes[i].offset;
            }
            sentBins.push_back(bin);
        }
    }
    std::vector<SentFlush>().swap(sentFlushes);
    SORT_SERIAL(sentBins.begin(), sentBins.end(), SentBin::compareByLength);
    // the line of a node: where its table starts, its sequences without residues, then the sequence count, header bytes and table bins of each of its lengths
    std::string nodeLine = SSTR(exchangeBytes) + "\t" + SSTR(emptySequences);
    for (size_t i = 0, binEnd; i < sentBins.size(); i = binEnd) {
        size_t sequenceCount = 0;
        size_t headerBytes = 0;
        for (binEnd = i; binEnd < sentBins.size() && sentBins[binEnd].length == sentBins[i].length; binEnd++) {
            sequenceCount += sentBins[binEnd].sequenceCount;
            headerBytes += sentBins[binEnd].headerBytes;
        }
        nodeLine += "\t" + SSTR(sentBins[i].length) + "\t" + SSTR(sequenceCount) + "\t" + SSTR(headerBytes) + "\t" + SSTR(binEnd - i);
    }
    if (fseeko(exchangeHandle, exchangeBytes, SEEK_SET) != 0 || fwrite(sentBins.data(), sizeof(SentBin), sentBins.size(), exchangeHandle) != sentBins.size()) {
        Debug(Debug::ERROR) << "Cannot write " << exchangeFile << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (fclose(exchangeHandle) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << exchangeFile << "\n";
        EXIT(EXIT_FAILURE);
    }
    std::vector<SentBin>().swap(sentBins);
    const std::string nodeLines = MMseqsNode::barrier(exchange + ".index", nodeLine + "\n");
    MMseqsNode::addPhaseTime(phaseValues, "sending the sequences", timer);

    // every node has the length histograms of all nodes and makes the same layout
    std::vector<size_t> tableOffsets(MMseqsNode::count);
    std::vector<std::vector<LengthBin>> nodeHistograms(MMseqsNode::count);
    // where the table bins of each length of a node start, then their count
    std::vector<std::vector<size_t>> nodeTableOffsets(MMseqsNode::count, std::vector<size_t>(1, 0));
    size_t skippedSequences = 0;
    char *nodeLinesChar = (char *) nodeLines.c_str();
    for (unsigned int node = 0; node < MMseqsNode::count; node++) {
        tableOffsets[node] = strtoull(nodeLinesChar, &nodeLinesChar, 10);
        skippedSequences += strtoull(nodeLinesChar, &nodeLinesChar, 10);
        while (*nodeLinesChar != '\n') {
            LengthBin bin = {0, 0, 0, 0, {}, 0};
            bin.length = strtoull(nodeLinesChar, &nodeLinesChar, 10);
            bin.sequenceCount = strtoull(nodeLinesChar, &nodeLinesChar, 10);
            bin.headerBytes = strtoull(nodeLinesChar, &nodeLinesChar, 10);
            nodeHistograms[node].push_back(bin);
            nodeTableOffsets[node].push_back(nodeTableOffsets[node].back() + strtoull(nodeLinesChar, &nodeLinesChar, 10));
        }
        nodeLinesChar++;
    }
    if (skippedSequences > 0 && MMseqsNode::isMaster()) {
        Debug(Debug::WARNING) << skippedSequences << " sequences without residues are skipped\n";
    }
    std::vector<LengthBin> globalHistogram = mergeLengthHistograms(nodeHistograms);
    if (globalHistogram.empty()) {
        if (MMseqsNode::isMaster()) {
            Debug(Debug::ERROR) << "The input files have no entry: ";
            for (size_t fileIdx = 0; fileIdx < filenames.size(); fileIdx++) {
                Debug(Debug::ERROR) << " - " << filenames[fileIdx] << "\n";
            }
            Debug(Debug::ERROR) << "Please check your input files. Only files in fasta/fastq[.gz|bz2|zst] are supported\n";
        }
        EXIT(EXIT_FAILURE);
    }
    size_t key = 0;
    size_t dataSize = 0;
    std::vector<size_t> binBytes(globalHistogram.size());
    size_t maxFileBytes = 0;
    for (size_t i = 0; i < globalHistogram.size(); i++) {
        globalHistogram[i].startKey = key;
        key += globalHistogram[i].sequenceCount;
        binBytes[i] = globalHistogram[i].sequenceCount * (globalHistogram[i].length + 2 + sizeof(uint64_t)) + globalHistogram[i].headerBytes;
        dataSize += binBytes[i];
        maxFileBytes = std::max(maxFileBytes, binBytes[i]);
    }
    // the smallest largest file for which the lengths, filled into the files in order, fit; a length is never split
    maxFileBytes = std::max(maxFileBytes, (dataSize + MMseqsNode::count - 1) / MMseqsNode::count);
    for (size_t fitBytes = dataSize; maxFileBytes < fitBytes; ) {
        const size_t fileBytes = maxFileBytes + (fitBytes - maxFileBytes) / 2;
        size_t files = 1;
        size_t fill = 0;
        for (size_t i = 0; i < binBytes.size(); i++) {
            if (fill + binBytes[i] > fileBytes) {
                files++;
                fill = 0;
            }
            fill += binBytes[i];
        }
        if (files <= MMseqsNode::count) {
            fitBytes = fileBytes;
        } else {
            maxFileBytes = fileBytes + 1;
        }
    }
    std::vector<size_t> dataFileSizes(DATA_FILE_TYPES * MMseqsNode::count, 0);
    std::vector<LengthBin> localNodeStarts;
    size_t file = 0;
    size_t fill = 0;
    std::string indexData;
    for (size_t i = 0; i < globalHistogram.size(); i++) {
        LengthBin &bin = globalHistogram[i];
        if (fill + binBytes[i] > maxFileBytes) {
            file++;
            fill = 0;
        }
        fill += binBytes[i];
        bin.file = file;
        for (size_t fileType = 0; fileType < DATA_FILE_TYPES; fileType++) {
            bin.offsets[fileType] = dataFileSizes[fileType * MMseqsNode::count + file];
        }
        dataFileSizes[SEQUENCE_DATA * MMseqsNode::count + file] += bin.sequenceCount * (bin.length + 2);
        dataFileSizes[HEADER_DATA * MMseqsNode::count + file] += bin.headerBytes;
        dataFileSizes[HEADER_OFFSETS * MMseqsNode::count + file] += bin.sequenceCount * sizeof(uint64_t);
        indexData += SSTR(bin.startKey) + "\t" + SSTR(bin.offsets[SEQUENCE_DATA]) + "\t" + SSTR(bin.length + 2) + "\t" + SSTR(bin.sequenceCount) + "\t" + SSTR(bin.file) + "\n";
        if (bin.file == MMseqsNode::rank) {
            localNodeStarts.push_back(bin);
        }
    }
    if (MMseqsNode::isMaster()) {
        std::string nodeInfoData;
        size_t fileFirstKey = 0;
        for (size_t i = HEADER_OFFSETS * MMseqsNode::count; i < dataFileSizes.size(); i++) {
            const size_t fileKeyCount = dataFileSizes[i] / sizeof(uint64_t);
            nodeInfoData += SSTR(fileFirstKey) + "\t" + SSTR(fileKeyCount) + "\n";
            fileFirstKey += fileKeyCount;
        }
        // the data files take dataSize bytes besides the exchange files, which are mostly freed only once each data file type is written
        const size_t freeSpace = FileUtil::getFreeSpace(FileUtil::dirName(dataFile).c_str());
        if (freeSpace < dataSize) {
            Debug(Debug::ERROR) << FileUtil::dirName(dataFile) << " needs at least " << dataSize << " free bytes for the sequences of " << dataFile << ", but has " << freeSpace << "\n";
            EXIT(EXIT_FAILURE);
        }
        MMseqsNode::publishFile(dataFile + ".index", indexData);
        MMseqsNode::publishFile(dataFile + ".nodeinfo", nodeInfoData);
    }

    // the files of this node get their final sizes, so a file size limit fails before they are written
    std::string dataFileNames[DATA_FILE_TYPES];
    FILE *dataFiles[DATA_FILE_TYPES];
    const size_t headerDataSize = dataFileSizes[HEADER_DATA * MMseqsNode::count + MMseqsNode::rank];
    for (size_t fileType = 0; fileType < DATA_FILE_TYPES; fileType++) {
        dataFileNames[fileType] = dataFile + dataFileSuffixes[fileType] + SSTR(MMseqsNode::rank);
        const size_t dataFileSize = dataFileSizes[fileType * MMseqsNode::count + MMseqsNode::rank];
        const size_t fileSize = dataFileSize + ((fileType == HEADER_OFFSETS) ? sizeof(uint64_t) : 0);
        dataFiles[fileType] = FileUtil::openAndDelete(dataFileNames[fileType].c_str(), "w");
        if (ftruncate(fileno(dataFiles[fileType]), fileSize) != 0) {
            Debug(Debug::ERROR) << "Cannot resize " << dataFileNames[fileType] << " to " << fileSize << " bytes: " << strerror(errno) << ". Run on more nodes if the file system limits the file size\n";
            EXIT(EXIT_FAILURE);
        }
        if (fileType == HEADER_OFFSETS) {
            FileUtil::pwriteAll(fileno(dataFiles[fileType]), dataFileNames[fileType].c_str(), reinterpret_cast<const char *>(&headerDataSize), sizeof(uint64_t), dataFileSize);
        }
    }

    // the segments of the data files of this node from all nodes, written window by window
    double compressSeconds = 0;
    {
        // the table bins of the lengths of this node from the table of every node, which follow each other there, in flush order
        std::vector<FILE *> exchangeFiles(MMseqsNode::count, NULL);
        std::vector<std::vector<SentBin> > nodeBins(MMseqsNode::count);
        if (localNodeStarts.empty() == false) {
#pragma omp parallel for schedule(dynamic, 1)
            for (size_t node = 0; node < MMseqsNode::count; node++) {
                const std::vector<LengthBin> &histogram = nodeHistograms[node];
                const size_t firstBin = nodeTableOffsets[node][std::lower_bound(histogram.begin(), histogram.end(), localNodeStarts.front().length, isLongerBin) - histogram.begin()];
                const size_t lastBin = nodeTableOffsets[node][std::lower_bound(histogram.begin(), histogram.end(), localNodeStarts.back().length - 1, isLongerBin) - histogram.begin()];
                if (firstBin == lastBin) {
                    continue;
                }
                const std::string file = exchange + "." + SSTR(node);
                // read and write, so the bytes of a block are freed once read
                exchangeFiles[node] = FileUtil::openFileOrDie(file.c_str(), "r+", true);
                nodeBins[node].resize(lastBin - firstBin);
                FileUtil::preadAll(fileno(exchangeFiles[node]), file.c_str(), reinterpret_cast<char *>(nodeBins[node].data()), nodeBins[node].size() * sizeof(SentBin), tableOffsets[node] + firstBin * sizeof(SentBin));
                SORT_SERIAL(nodeBins[node].begin(), nodeBins[node].end(), SentBin::compareByFlush);
            }
        }
        // a block per flush and data file type with the offsets of its bins in the data files, given in key order: by node, flush, then longer first
        std::vector<LengthBin> lengthToStart = getLengthToBin(localNodeStarts);
        std::vector<std::vector<MMseqsNode::ExchangeBlock> > blocks(DATA_FILE_TYPES);
        std::vector<std::vector<std::vector<std::pair<size_t, size_t> > > > blockBins(DATA_FILE_TYPES);
        // where the header offsets of a bin start in their data file, with where its headers start
        std::vector<std::pair<size_t, size_t> > headerOffsetBins;
        size_t receivedTableBins = 0;
        size_t ownBytes = 0;
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            const std::vector<SentBin> &bins = nodeBins[node];
            for (size_t i = 0; i < bins.size(); i++) {
                LengthBin &start = lengthToStart[bins[i].length];
                const bool isFlushStart = i == 0 || bins[i - 1].flush != bins[i].flush;
                for (size_t fileType = 0; fileType < DATA_FILE_TYPES; fileType++) {
                    if (isFlushStart) {
                        const MMseqsNode::ExchangeBlock block = { exchangeFiles[node], bins[i].starts[fileType], bins[i].ends[fileType], std::vector<char>(), 0, 0, 0, 0, NULL, bins[i].starts[fileType] };
                        blocks[fileType].push_back(block);
                        blockBins[fileType].push_back(std::vector<std::pair<size_t, size_t> >());
                    }
                    blocks[fileType].back().end = bins[i].ends[fileType];
                    blockBins[fileType].back().push_back(std::make_pair(bins[i].length, start.offsets[fileType]));
                    ownBytes += (node == MMseqsNode::rank) ? bins[i].ends[fileType] - bins[i].starts[fileType] : 0;
                }
                headerOffsetBins.push_back(std::make_pair(start.offsets[HEADER_OFFSETS], start.offsets[HEADER_DATA]));
                start.offsets[SEQUENCE_DATA] += bins[i].sequenceCount * (bins[i].length + 2);
                start.offsets[HEADER_DATA] += bins[i].headerBytes;
                start.offsets[HEADER_OFFSETS] += bins[i].sequenceCount * sizeof(uint64_t);
            }
            receivedTableBins += bins.size();
            std::vector<SentBin>().swap(nodeBins[node]);
        }
        MMseqsNode::addSentExchangeBytes(exchangeBytes, exchangeBytes - ownBytes);
        SORT_SERIAL(headerOffsetBins.begin(), headerOffsetBins.end());
        // the send buffers are freed, so the receive takes the same budget but the bins: half of it buffers the blocks, so each read of a block is large, and the writes of a window go on behind the reads of the next one
        const size_t receiveMemory = static_cast<size_t>(memoryLimit * 0.95) - receivedTableBins * (sizeof(SentBin) + (DATA_FILE_TYPES + 1) * sizeof(std::pair<size_t, size_t>));
        const size_t blockMemory = receiveMemory / 2;
        const size_t windowSize = receiveMemory / 8;
        const size_t writeSize = std::max<size_t>(1, windowSize / par.threads);
        BatchWriter batchWriter(windowSize, par.threads, fileno(dataFiles[SEQUENCE_DATA]));
        // a segment that starts in the window ends at most MAX_SEGMENT_SIZE behind it
        char *window = NULL;
        if (blocks[SEQUENCE_DATA].empty() == false || par.compressHeaders) {
            window = new(std::nothrow) char[windowSize + MAX_SEGMENT_SIZE];
            Util::checkAllocation(window, "Can not allocate window memory in createdb");
        }
        // --compress-headers: the headers are compressed in frames of frameSize as they are received, the frames of a window by parts of frames on all threads
        const size_t frameSize = 1024 * 1024;
        const size_t frameBound = ZSTD_compressBound(frameSize);
        const std::string zstdFileName = dataFileNames[HEADER_DATA] + ".tmp";
        FILE *zstdFile = NULL;
        char *frames = NULL;
        std::vector<uint32_t> seekTable;
        size_t compressedBytes = 0;
        if (par.compressHeaders) {
            zstdFile = FileUtil::openAndDelete(zstdFileName.c_str(), "w");
            frames = new(std::nothrow) char[((windowSize + MAX_SEGMENT_SIZE) / frameSize + 1) * frameBound];
            Util::checkAllocation(frames, "Can not allocate frame memory in createdb");
        }
        std::vector<std::pair<size_t, size_t> > segments;
        std::vector<std::pair<size_t, size_t> > writes;
        for (size_t fileType = 0; fileType < DATA_FILE_TYPES; fileType++) {
            for (size_t windowBegin = 0; windowBegin != SIZE_MAX; ) {
                segments.clear();
                const size_t nextOffset = MMseqsNode::readExchangeSegments(blocks[fileType], blockBins[fileType], blockMemory, MAX_SEGMENT_SIZE, windowBegin, windowBegin + windowSize, window, segments, exchange);
                SORT_SERIAL(segments.begin(), segments.end());
                if (fileType == HEADER_OFFSETS) {
#pragma omp parallel for schedule(dynamic, 1)
                    for (size_t i = 0; i < segments.size(); i++) {
                        const uint64_t headerStart = (std::upper_bound(headerOffsetBins.begin(), headerOffsetBins.end(), std::make_pair(segments[i].first, SIZE_MAX)) - 1)->second;
                        uint64_t *headerOffsets = reinterpret_cast<uint64_t *>(window + (segments[i].first - windowBegin));
                        for (size_t j = 0; j < segments[i].second / sizeof(uint64_t); j++) {
                            headerOffsets[j] += headerStart;
                        }
                    }
                }
                writes.clear();
                if (par.compressHeaders && fileType == HEADER_DATA) {
                    Timer compressTimer;
                    // the segments fill the window up to the next one
                    const size_t windowEnd = std::min(nextOffset, headerDataSize);
                    // the frames up to the last whole one are compressed, the rest is carried to the next window
                    const size_t frameEnd = (windowEnd == headerDataSize) ? windowEnd : windowEnd - windowEnd % frameSize;
                    const size_t frameCount = (frameEnd - windowBegin + frameSize - 1) / frameSize;
                    const size_t partFrames = std::max<size_t>(1, (frameCount + par.threads - 1) / par.threads);
                    const size_t partCount = (frameCount + partFrames - 1) / partFrames;
                    const size_t firstFrame = seekTable.size() / 2;
                    seekTable.resize(seekTable.size() + 2 * frameCount);
                    std::vector<size_t> partOffsets(partCount + 1, 0);
#pragma omp parallel for schedule(dynamic, 1)
                    for (size_t part = 0; part < partCount; part++) {
                        for (size_t frame = part * partFrames; frame < std::min(frameCount, (part + 1) * partFrames); frame++) {
                            const size_t size = std::min(frameSize, frameEnd - windowBegin - frame * frameSize);
                            const size_t compressedSize = ZSTD_compress(frames + part * partFrames * frameBound + partOffsets[part + 1], frameBound, window + frame * frameSize, size, 3);
                            if (ZSTD_isError(compressedSize)) {
                                Debug(Debug::ERROR) << "ZSTD_compress " << ZSTD_getErrorName(compressedSize) << "\n";
                                EXIT(EXIT_FAILURE);
                            }
                            seekTable[2 * (firstFrame + frame)] = compressedSize;
                            seekTable[2 * (firstFrame + frame) + 1] = size;
                            partOffsets[part + 1] += compressedSize;
                        }
                    }
                    for (size_t part = 0; part < partCount; part++) {
                        partOffsets[part + 1] += partOffsets[part];
                    }
#pragma omp parallel for schedule(dynamic, 1)
                    for (size_t part = 0; part < partCount; part++) {
                        batchWriter.write(fileno(zstdFile), zstdFileName.c_str(), frames + part * partFrames * frameBound, partOffsets[part + 1] - partOffsets[part], compressedBytes + partOffsets[part]);
                    }
                    compressedBytes += partOffsets.back();
                    memmove(window, window + (frameEnd - windowBegin), windowEnd - frameEnd);
                    windowBegin = (frameEnd == headerDataSize) ? SIZE_MAX : frameEnd;
                    compressSeconds += compressTimer.getTimediff();
                } else {
                    // segments that follow each other are written at once
                    for (size_t i = 0; i < segments.size(); i++) {
                        if (writes.empty() == false && writes.back().first + writes.back().second == segments[i].first && writes.back().second + segments[i].second <= writeSize) {
                            writes.back().second += segments[i].second;
                        } else {
                            writes.push_back(segments[i]);
                        }
                    }
#pragma omp parallel for schedule(dynamic, 1)
                    for (size_t i = 0; i < writes.size(); i++) {
                        batchWriter.write(fileno(dataFiles[fileType]), dataFileNames[fileType].c_str(), window + (writes[i].first - windowBegin), writes[i].second, writes[i].first);
                    }
                    windowBegin = nextOffset;
                }
            }
        }
        batchWriter.waitWrites();
        Util::releasePages(window, windowSize + MAX_SEGMENT_SIZE);
        delete[] window;
        MMseqsNode::closeExchangeFiles(exchangeFiles, exchange);
        if (par.compressHeaders) {
            delete[] frames;
            const uint32_t skippableHeader[2] = {ZSTD_MAGIC_SKIPPABLE_START | 0xE, static_cast<uint32_t>(seekTable.size() * sizeof(uint32_t) + ZSTD_seekTableFooterSize)};
            const uint32_t footerFrames = seekTable.size() / 2;
            const char footerDescriptor = 0;
            const uint32_t footerMagic = ZSTD_SEEKABLE_MAGICNUMBER;
            if (fseeko(zstdFile, compressedBytes, SEEK_SET) != 0
                || fwrite(skippableHeader, sizeof(uint32_t), 2, zstdFile) != 2
                || fwrite(seekTable.data(), sizeof(uint32_t), seekTable.size(), zstdFile) != seekTable.size()
                || fwrite(&footerFrames, sizeof(uint32_t), 1, zstdFile) != 1
                || fwrite(&footerDescriptor, sizeof(char), 1, zstdFile) != 1
                || fwrite(&footerMagic, sizeof(uint32_t), 1, zstdFile) != 1) {
                Debug(Debug::ERROR) << "Can not write to data file " << zstdFileName << "\n";
                EXIT(EXIT_FAILURE);
            }
            if (fclose(zstdFile) != 0) {
                Debug(Debug::ERROR) << "Cannot close data file " << zstdFileName << "\n";
                EXIT(EXIT_FAILURE);
            }
        }
    }
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < DATA_FILE_TYPES; i++) {
        if (fclose(dataFiles[i]) != 0) {
            Debug(Debug::ERROR) << "Cannot close data file " << dataFileNames[i] << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
    if (par.compressHeaders) {
        FileUtil::move((dataFileNames[HEADER_DATA] + ".tmp").c_str(), dataFileNames[HEADER_DATA].c_str());
        MMseqsNode::addPhaseSeconds(phaseValues, "compressing the headers", compressSeconds);
    }
    MMseqsNode::addPhaseSeconds(phaseValues, "writing the sequences", timer.getTimediff() - compressSeconds);
    MMseqsNode::addPhaseValue(phaseValues, "Bytes written", SSTR(dataFileSizes[SEQUENCE_DATA * MMseqsNode::count + MMseqsNode::rank] + headerDataSize + dataFileSizes[HEADER_OFFSETS * MMseqsNode::count + MMseqsNode::rank] + sizeof(uint64_t)));
    MMseqsNode::removeExchanges(dataFile, exchange, std::vector<std::string>(1, exchange), phaseValues);
    if (MMseqsNode::isMaster()) {
        std::string sourceData;
        for (size_t i = 0; i < filenames.size(); i++) {
            sourceData += SSTR(i) + "\t" + FileUtil::baseName(filenames[i]) + "\n";
        }
        MMseqsNode::publishFile(dataFile + ".source", sourceData);
        DBWriter::writeDbtypeFile((dataFile + "_h").c_str(), DBReader<DBKeyType>::setExtendedDbtype(Parameters::DBTYPE_GENERIC_DB, Parameters::DBTYPE_EXTENDED_LINFINITY), par.compressHeaders);
        const int dbType = DBReader<DBKeyType>::setExtendedDbtype(Parameters::DBTYPE_AMINO_ACIDS, Parameters::DBTYPE_EXTENDED_LINFINITY);
        DBWriter::writeDbtypeFile(dataFile.c_str(), dbType, false);
        Debug(Debug::INFO) << "Database type: " << Parameters::getDbTypeName(dbType) << "\n";
    }

    return EXIT_SUCCESS;
}

int createdb(int argc, const char **argv, const Command& command) {
    MMseqsNode::init();
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
		    filenames.push_back(line);
	    }
	    free(line);
	    fclose(file);
    }

    // consistent order
    SORT_SERIAL(filenames.begin(), filenames.end(), [](const std::string &a, const std::string &b) {
        return FileUtil::baseName(a) < FileUtil::baseName(b);
    });
    
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
        if (par.createdbMode == Parameters::SEQUENCE_SPLIT_MODE_LINFINITY_LENGTH) {
            Debug(Debug::ERROR) << "Database input cannot be combined with --createdb-mode " << Parameters::SEQUENCE_SPLIT_MODE_LINFINITY_LENGTH << "\n";
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

    if (par.createdbMode == Parameters::SEQUENCE_SPLIT_MODE_GPU && par.shuffleDatabase == false) {
        Debug(Debug::WARNING) << "Shuffle database cannot be turned off for --createdb-mode " << Parameters::SEQUENCE_SPLIT_MODE_GPU << "\n";
        Debug(Debug::WARNING) << "We recompute with --shuffle 1\n";
        par.shuffleDatabase = true;
    }

    if ((par.createdbMode == Parameters::SEQUENCE_SPLIT_MODE_SOFT || par.createdbMode == Parameters::SEQUENCE_SPLIT_MODE_LINFINITY_LENGTH) && par.filenames[0] == "stdin") {
        if (MMseqsNode::count > 1) {
            Debug(Debug::ERROR) << "Stdin input cannot be combined with --createdb-mode " << par.createdbMode << " on multiple nodes\n";
            EXIT(EXIT_FAILURE);
        }
        Debug(Debug::WARNING) << "Stdin input cannot be combined with --createdb-mode " << par.createdbMode << "\n";
        Debug(Debug::WARNING) << "We recompute with --createdb-mode 1\n";
        par.createdbMode = Parameters::SEQUENCE_SPLIT_MODE_HARD;
    }

    if ((par.maskMode == 1 || par.maskNrepeats > 0) && par.createdbMode != Parameters::SEQUENCE_SPLIT_MODE_GPU) {
        Debug(Debug::WARNING) << "Masking can only be used with --createdb-mode " << Parameters::SEQUENCE_SPLIT_MODE_GPU << "\n";
        Debug(Debug::WARNING) << "We recompute with --mask 0 or --mask-n-repeat 0\n";
        par.maskMode = false;
        par.maskNrepeats = 0;
    }

    const unsigned int shuffleSplits = par.shuffleDatabase ? 32 : 1;
    if ((par.createdbMode == Parameters::SEQUENCE_SPLIT_MODE_SOFT || par.createdbMode == Parameters::SEQUENCE_SPLIT_MODE_LINFINITY_LENGTH) && par.compressed) {
        Debug(Debug::WARNING) << "Compressed database cannot be combined with --createdb-mode " << par.createdbMode << "\n";
        Debug(Debug::WARNING) << "We recompute with --compressed 0\n";
        par.compressed = 0;
    }

    if (par.createdbMode != Parameters::SEQUENCE_SPLIT_MODE_LINFINITY_LENGTH && par.compressHeaders) {
        Debug(Debug::WARNING) << "Header compression can only be used with --createdb-mode " << Parameters::SEQUENCE_SPLIT_MODE_LINFINITY_LENGTH << "\n";
        Debug(Debug::WARNING) << "We recompute with --compress-headers 0\n";
        par.compressHeaders = 0;
    }

    if (par.createdbMode == Parameters::SEQUENCE_SPLIT_MODE_LINFINITY_LENGTH && par.identifierOffset != 0) {
        Debug(Debug::WARNING) << "Identifier offset cannot be combined with --createdb-mode " << par.createdbMode << "\n";
        Debug(Debug::WARNING) << "We recompute with --id-offset 0\n";
        par.identifierOffset = 0;
    }

    if (par.createdbMode == Parameters::SEQUENCE_SPLIT_MODE_LINFINITY_LENGTH && par.dbType == 2) {
        Debug(Debug::WARNING) << "Nucleotide database cannot be combined with --createdb-mode " << par.createdbMode << "\n";
        Debug(Debug::WARNING) << "We recompute with --dbtype 1\n";
        par.dbType = 1;
    }

    if(par.createdbMode == Parameters::SEQUENCE_SPLIT_MODE_LINFINITY_LENGTH) {
        delete subMat;
        return createdbLinfinity(par, filenames, dataFile);
    }

    std::string hdrDataFile = dataFile + "_h";
    std::string hdrIndexFile = dataFile + "_h.index";

    DBKeyType entries_num = 0;
    const char newline = '\n';

    size_t sampleCount = 0;
    const size_t testForNucSequence = 100;
    size_t isNuclCnt = 0;
    Debug::Progress progress;
    std::vector<unsigned int>* sourceLookup = new std::vector<unsigned int>[shuffleSplits]();
    for (size_t i = 0; i < shuffleSplits; ++i) {
        sourceLookup[i].reserve(16384);
    }
    Debug(Debug::INFO) << "Converting sequences\n";

    std::string sourceFile = dataFile + ".source";

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
            Debug(Debug::WARNING) << "Only uncompressed fasta files can be used with --createdb-mode 0\n";
            Debug(Debug::WARNING) << "We recompute with --createdb-mode 1\n";
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
                Debug(Debug::WARNING) << "Last byte is not a newline. We recompute with --createdb-mode 1\n";
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
            for (size_t i = 0; i < shuffleSplits; ++i) {
                sourceLookup[i].clear();
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
                        size_t cnt = 0;
                        for (size_t i = 0; i < e.sequence.l; i++) {
                            switch (toupper(e.sequence.s[i])) {
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
                        const float nuclDNAFraction = static_cast<float>(cnt) / static_cast<float>(e.sequence.l);
                        if (nuclDNAFraction > 0.9) {
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
                        Debug(Debug::WARNING) << "Multiline fasta can not be combined with --createdb-mode 1\n";
                    }
                    Debug(Debug::WARNING) << "We recompute with --createdb-mode 0\n";
                    par.createdbMode = Parameters::SEQUENCE_SPLIT_MODE_HARD;
                    progress.reset(SIZE_MAX);
                    hdrWriter.close();
                    seqWriter.close();
                    delete kseq;
                    if (fclose(source) != 0) {
                        Debug(Debug::ERROR) << "Cannot close file " << sourceFile << "\n";
                        EXIT(EXIT_FAILURE);
                    }
                    for (size_t i = 0; i < shuffleSplits; ++i) {
                        sourceLookup[i].clear();
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
            sourceLookup[splitIdx].emplace_back(fileIdx);
            if (par.createdbMode == Parameters::SEQUENCE_SPLIT_MODE_SOFT) {
                // +2 to emulate the \n\0
                hdrWriter.writeIndexEntry(id, headerFileOffset + e.headerOffset, (e.sequenceOffset-e.headerOffset)+1, 0);
                seqWriter.writeIndexEntry(id, seqFileOffset + e.sequenceOffset, e.sequence.l+2, 0);
            } else if (par.createdbMode == Parameters::SEQUENCE_SPLIT_MODE_HARD) {
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
                                    id - (batchPos - 1), batchEntries, batchPos, shuffleSplits);
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
                            (par.identifierOffset + entries_num) - (batchPos - 1), batchEntries, batchPos, shuffleSplits);
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
    if(par.createdbMode == Parameters::SEQUENCE_SPLIT_MODE_GPU){
        gpuCompatibleDB = true;
        hdrWriter.closeFiles();
        seqWriter.closeFiles();
        for(unsigned int i = 0; i < shuffleSplits; i++){
            sortWithIndex(seqWriter.getDataFileNames()[i], seqWriter.getIndexFileNames()[i],
                          hdrWriter.getDataFileNames()[i], hdrWriter.getIndexFileNames()[i]);
        }
        Debug(Debug::INFO) << "Sort single files in " << timer.lap() << "\n";
        std::string lookupFile = dataFile + ".lookup";

        timer.reset();
        mergeSequentialByJointIndex(seqWriter.getDataFileNames(), seqWriter.getIndexFileNames(),
                                    hdrWriter.getDataFileNames(), hdrWriter.getIndexFileNames(),
                                    seqWriter.getDataFileName(), seqWriter.getIndexFileName(),
                                    hdrWriter.getDataFileName(), hdrWriter.getIndexFileName(),
                                    lookupFile.c_str(), sourceLookup, entries_num, shuffleSplits);
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
        Debug(Debug::ERROR) << "Please check your input files. Only files in fasta/fastq[.gz|bz2|zst] are supported\n";
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
