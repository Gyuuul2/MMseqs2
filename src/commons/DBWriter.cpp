#include "DBWriter.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include "FastSort.h"
#include "DBReader.h"
#include "Debug.h"
#include "Util.h"
#include "FileUtil.h"
#include "Concat.h"
#include "itoa.h"
#include "Timer.h"
#include "Parameters.h"
#include "MemoryMapped.h"

#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/simde-common.h>

#include <cstdlib>
#include <cstdio>
#include <sstream>
#include <unistd.h>

#ifdef OPENMP
#include <omp.h>
#endif

DBWriter::DBWriter(const char *dataFileName_, const char *indexFileName_, unsigned int threads, size_t mode, int dbtype)
        : threads(threads), mode(mode), dbtype(dbtype) {
    dataFileName = strdup(dataFileName_);
    indexFileName = strdup(indexFileName_);

    dataFiles = new FILE *[threads];
    dataFilesBuffer = new char *[threads];

    dataFileNames = new char *[threads];

    indexFiles = new FILE *[threads];
    indexFileNames = new char *[threads];
    compressedBuffers=NULL;
    compressedBufferSizes=NULL;
    if((mode & Parameters::WRITER_COMPRESSED_MODE) != 0){
        compressedBuffers = new char*[threads];
        compressedBufferSizes = new size_t[threads];
        cstream = new ZSTD_CStream*[threads];
        state = new int[threads];
        threadBuffer = new char*[threads];
        threadBufferSize = new size_t[threads];
        threadBufferOffset = new size_t[threads];
    }

    starts = new size_t[threads];
    std::fill(starts, starts + threads, 0);
    offsets = new size_t[threads];
    std::fill(offsets, offsets + threads, 0);
    if((mode & Parameters::WRITER_COMPRESSED_MODE) != 0 ){
        datafileMode = "wb+";
    } else {
        datafileMode = "wb";
    }

    closed = true;
}

size_t DBWriter::addToThreadBuffer(const void *data, size_t itmesize, size_t nitems, int threadIdx) {
    size_t bytesToWrite = (itmesize*nitems);
    size_t bytesLeftInBuffer = threadBufferSize[threadIdx] - threadBufferOffset[threadIdx];
    if( (itmesize*nitems)  >= bytesLeftInBuffer ){
        size_t newBufferSize = std::max(threadBufferSize[threadIdx] + bytesToWrite, threadBufferSize[threadIdx] * 2 );
        threadBufferSize[threadIdx] = newBufferSize;
        threadBuffer[threadIdx] = (char*) realloc(threadBuffer[threadIdx], newBufferSize);
        if(compressedBuffers[threadIdx] == NULL){
            Debug(Debug::ERROR) << "Realloc of buffer for " << threadIdx << " failed. Buffer size = " << threadBufferSize[threadIdx] << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
    memcpy(threadBuffer[threadIdx] + threadBufferOffset[threadIdx], data, bytesToWrite);
    threadBufferOffset[threadIdx] += bytesToWrite;
    return bytesToWrite;
}

DBWriter::~DBWriter() {
    delete[] offsets;
    delete[] starts;
    delete[] indexFileNames;
    delete[] indexFiles;
    delete[] dataFileNames;
    delete[] dataFilesBuffer;
    delete[] dataFiles;
    free(indexFileName);
    free(dataFileName);
    if(compressedBuffers){
        delete [] threadBuffer;
        delete [] threadBufferSize;
        delete [] threadBufferOffset;
        delete [] compressedBuffers;
        delete [] compressedBufferSizes;
        delete [] cstream;
        delete [] state;
    }
}

void DBWriter::sortDatafileByIdOrder(DBReader<DBKeyType> &dbr) {
#pragma omp parallel
    {
        int thread_idx = 0;
#ifdef OPENMP
        thread_idx = omp_get_thread_num();
#endif

#pragma omp for schedule(static)
        for (size_t id = 0; id < dbr.getSize(); id++) {
            char *data = dbr.getData(id, thread_idx);
            size_t length = dbr.getEntryLen(id);
            writeData(data, (length == 0 ? 0 : length - 1), dbr.getDbKey(id), thread_idx);
        }
    };

}

// allocates heap memory, careful
char* makeResultFilename(const char* name, size_t split) {
    std::ostringstream ss;
    ss << name << "." << split;
    std::string s = ss.str();
    return strdup(s.c_str());
}

void DBWriter::open(size_t bufferSize) {
    if (bufferSize == SIZE_MAX) {
        if (Util::getTotalSystemMemory() < (8ull * 1024 * 1024 * 1024)) {
            // reduce this buffer if our system does not have much memory
            // createdb runs into trouble since it creates 2x32 splits with 64MB each (=4GB)
            // 8MB should be enough
            bufferSize = 8ull * 1024 * 1024;
        } else {
            bufferSize = 32ull * 1024 * 1024;
        }
    }
    for (unsigned int i = 0; i < threads; i++) {
        dataFileNames[i] = makeResultFilename(dataFileName, i);
        indexFileNames[i] = makeResultFilename(indexFileName, i);

        dataFiles[i] = FileUtil::openAndDelete(dataFileNames[i], datafileMode.c_str());
        int fd = fileno(dataFiles[i]);
        int flags;
        if ((flags = fcntl(fd, F_GETFL, 0)) < 0 || fcntl(fd, F_SETFD, flags | FD_CLOEXEC) == -1) {
            Debug(Debug::ERROR) << "Can not set mode for " << dataFileNames[i] << "!\n";
            EXIT(EXIT_FAILURE);
        }

        dataFilesBuffer[i] = new(std::nothrow) char[bufferSize];
        Util::checkAllocation(dataFilesBuffer[i], "Cannot allocate buffer for DBWriter");
        incrementMemory(bufferSize);
        this->bufferSize = bufferSize;

        // set buffer to 64
        if (setvbuf(dataFiles[i], dataFilesBuffer[i], _IOFBF, bufferSize) != 0) {
            Debug(Debug::WARNING) << "Write buffer could not be allocated (bufferSize=" << bufferSize << ")\n";
        }

        indexFiles[i] = FileUtil::openAndDelete(indexFileNames[i], "w");
        fd = fileno(indexFiles[i]);
        if ((flags = fcntl(fd, F_GETFL, 0)) < 0 || fcntl(fd, F_SETFD, flags | FD_CLOEXEC) == -1) {
            Debug(Debug::ERROR) << "Can not set mode for " << indexFileNames[i] << "!\n";
            EXIT(EXIT_FAILURE);
        }

        if (setvbuf(indexFiles[i], NULL, _IOFBF, bufferSize) != 0) {
            Debug(Debug::WARNING) << "Write buffer could not be allocated (bufferSize=" << bufferSize << ")\n";
        }

        if (dataFiles[i] == NULL) {
            perror(dataFileNames[i]);
            EXIT(EXIT_FAILURE);
        }

        if (indexFiles[i] == NULL) {
            perror(indexFileNames[i]);
            EXIT(EXIT_FAILURE);
        }

        if((mode & Parameters::WRITER_COMPRESSED_MODE) != 0){
            compressedBufferSizes[i] = 2097152;
            threadBufferSize[i] = 2097152;
            state[i] = false;
            compressedBuffers[i] = (char*) malloc(compressedBufferSizes[i]);
            incrementMemory(compressedBufferSizes[i]);
            threadBuffer[i] = (char*) malloc(threadBufferSize[i]);
            incrementMemory(threadBufferSize[i]);
            cstream[i] = ZSTD_createCStream();
        }
    }

    closed = false;
}

void DBWriter::writeDbtypeFile(const char* path, int dbtype, bool isCompressed) {
    if (dbtype == Parameters::DBTYPE_OMIT_FILE) {
        return;
    }

    std::string name = std::string(path) + ".dbtype";
    FILE* file = FileUtil::openAndDelete(name.c_str(), "wb");
    dbtype = isCompressed ? dbtype | (1 << 31) : dbtype & ~(1 << 31);
#if SIMDE_ENDIAN_ORDER == SIMDE_ENDIAN_BIG
    dbtype = __builtin_bswap32(dbtype);
#endif
    size_t written = fwrite(&dbtype, sizeof(int), 1, file);
    if (written != 1) {
        Debug(Debug::ERROR) << "Can not write to data file " << name << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (fclose(file) != 0) {
        Debug(Debug::ERROR) << "Cannot close file " << name << "\n";
        EXIT(EXIT_FAILURE);
    }
}

void DBWriter::closeFiles(){
    if(closed == false){
        // close all datafiles
        for (unsigned int i = 0; i < threads; i++) {
            if (fclose(dataFiles[i]) != 0) {
                Debug(Debug::ERROR) << "Cannot close data file " << dataFileNames[i] << "\n";
                EXIT(EXIT_FAILURE);
            }
            if (fclose(indexFiles[i]) != 0) {
                Debug(Debug::ERROR) << "Cannot close index file " << indexFileNames[i] << "\n";
                EXIT(EXIT_FAILURE);
            }
        }
        closed = true;
    }
}

// DONTNEED skips dirty pages, so writeback has to run first; only worth that barrier on a large file
static const size_t DBWRITER_CACHE_DROP_MIN_BYTES = 1ull * 1024 * 1024 * 1024;

static void releaseWrittenCache(const char *fileName) {
#if defined(HAVE_POSIX_FADVISE)
    const int fd = ::open(fileName, O_RDONLY);
    if (fd < 0) {
        return;
    }
    struct stat sb;
    if (fstat(fd, &sb) == 0 && static_cast<size_t>(sb.st_size) >= DBWRITER_CACHE_DROP_MIN_BYTES) {
        // a fresh descriptor cannot fdatasync another one's writes, so sync through the same inode
        if (fdatasync(fd) == 0) {
            posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        }
    }
    ::close(fd);
#else
    (void) fileName;
#endif
}

void DBWriter::close(bool merge, bool needsSort) {
    closeFiles();

    merge = getenv("MMSEQS_FORCE_MERGE") != NULL ? true : merge;
    mergeResults(dataFileName, indexFileName, (const char **) dataFileNames, (const char **) indexFileNames,
                 threads, merge, ((mode & Parameters::WRITER_LEXICOGRAPHIC_MODE) != 0), needsSort);

    writeDbtypeFile(dataFileName, dbtype, (mode & Parameters::WRITER_COMPRESSED_MODE) != 0);
    // after mergeResults, so the merge still reads its inputs from cache
    releaseWrittenCache(dataFileName);
    releaseWrittenCache(indexFileName);
    clearMemory();
    closed = true;
}

void DBWriter::clearMemory(){
    if(compressedBuffers){
        for (unsigned int i = 0; i < threads; i++) {
            free(compressedBuffers[i]);
            decrementMemory(compressedBufferSizes[i]);
            free(threadBuffer[i]);
            decrementMemory(threadBufferSize[i]);
            ZSTD_freeCStream(cstream[i]);
        }
    }

    for (unsigned int i = 0; i < threads; i++) {
        delete [] dataFilesBuffer[i];
        decrementMemory(bufferSize);
        free(dataFileNames[i]);
        free(indexFileNames[i]);
    }
}

void DBWriter::writeStart(unsigned int thrIdx) {
    checkClosed();
    if (thrIdx >= threads) {
        Debug(Debug::ERROR) << "Thread index " << thrIdx << " > maximum thread number " << threads << "\n";
        EXIT(EXIT_FAILURE);
    }
    starts[thrIdx] = offsets[thrIdx];
    if((mode & Parameters::WRITER_COMPRESSED_MODE) != 0){
        state[thrIdx] = INIT_STATE;
        threadBufferOffset[thrIdx]=0;
        int cLevel = 3;
        size_t const initResult = ZSTD_initCStream(cstream[thrIdx], cLevel);
        if (ZSTD_isError(initResult)) {
            Debug(Debug::ERROR) << "ZSTD_initCStream() error in thread " << thrIdx << ". Error "
                                << ZSTD_getErrorName(initResult) << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
}

size_t DBWriter::writeAdd(const char* data, size_t dataSize, unsigned int thrIdx) {
    checkClosed();
    if (thrIdx >= threads) {
        Debug(Debug::ERROR) << "Thread index " << thrIdx << " > maximum thread number " << threads << "\n";
        EXIT(EXIT_FAILURE);
    }
    bool isCompressedDB = (mode & Parameters::WRITER_COMPRESSED_MODE) != 0;
    if(isCompressedDB && state[thrIdx] == INIT_STATE && dataSize < 60){
        state[thrIdx] = NOTCOMPRESSED;
    }
    size_t totalWriten = 0;
    if(isCompressedDB && (state[thrIdx] == INIT_STATE || state[thrIdx] == COMPRESSED) ) {
        state[thrIdx] = COMPRESSED;
        // zstd seems to have a hard time with elements < 60
        ZSTD_inBuffer input = { data, dataSize, 0 };
        while (input.pos < input.size) {
            ZSTD_outBuffer output = {compressedBuffers[thrIdx], compressedBufferSizes[thrIdx], 0};
            size_t toRead = ZSTD_compressStream( cstream[thrIdx], &output, &input);   /* toRead is guaranteed to be <= ZSTD_CStreamInSize() */
            if (ZSTD_isError(toRead)) {
                Debug(Debug::ERROR) << "ZSTD_compressStream() error in thread " << thrIdx << ". Error "
                                    << ZSTD_getErrorName(toRead) << "\n";
                EXIT(EXIT_FAILURE);
            }
            size_t written = addToThreadBuffer(compressedBuffers[thrIdx], sizeof(char), output.pos, thrIdx);
            if (written != output.pos) {
                Debug(Debug::ERROR) << "Can not write to data file " << dataFileNames[thrIdx] << "\n";
                EXIT(EXIT_FAILURE);
            }
            offsets[thrIdx] += written;
            totalWriten += written;
        }
    }else{
        size_t written;
        if(isCompressedDB){
            written = addToThreadBuffer(data, sizeof(char), dataSize,  thrIdx);
        }else{
            written = fwrite(data, sizeof(char), dataSize, dataFiles[thrIdx]);
        }
        if (written != dataSize) {
            Debug(Debug::ERROR) << "Can not write to data file " << dataFileNames[thrIdx] << "\n";
            EXIT(EXIT_FAILURE);
        }
        offsets[thrIdx] += written;
    }

    return totalWriten;
}

void DBWriter::writeEnd(DBKeyType key, unsigned int thrIdx, bool addNullByte, bool addIndexEntry) {
    // close stream
    bool isCompressedDB = (mode & Parameters::WRITER_COMPRESSED_MODE) != 0;
    if(isCompressedDB) {
        size_t compressedLength = 0;
        if(state[thrIdx] == COMPRESSED) {
            ZSTD_outBuffer output = {compressedBuffers[thrIdx], compressedBufferSizes[thrIdx], 0};
            size_t remainingToFlush = ZSTD_endStream(cstream[thrIdx], &output); /* close frame */

            //        std::cout << compressedLength << std::endl;
            if (ZSTD_isError(remainingToFlush)) {
                Debug(Debug::ERROR) << "ZSTD_endStream() error in thread " << thrIdx << ". Error "
                                    << ZSTD_getErrorName(remainingToFlush) << "\n";
                EXIT(EXIT_FAILURE);
            }
            if (remainingToFlush) {
                Debug(Debug::ERROR) << "Stream not flushed\n";
                EXIT(EXIT_FAILURE);
            }
            size_t written = addToThreadBuffer(compressedBuffers[thrIdx], sizeof(char), output.pos, thrIdx);
            compressedLength = threadBufferOffset[thrIdx];
            offsets[thrIdx] += written;
            if (written != output.pos) {
                Debug(Debug::ERROR) << "Can not write to data file " << dataFileNames[thrIdx] << "\n";
                EXIT(EXIT_FAILURE);
            }
        }else {
            compressedLength = offsets[thrIdx] - starts[thrIdx];
        }
        unsigned int compressedLengthInt = static_cast<unsigned int>(compressedLength);
        size_t written2 = fwrite(&compressedLengthInt, sizeof(unsigned int), 1, dataFiles[thrIdx]);
        if (written2 != 1) {
            Debug(Debug::ERROR) << "Can not write entry length to data file " << dataFileNames[thrIdx] << "\n";
            EXIT(EXIT_FAILURE);
        }
        offsets[thrIdx] +=  sizeof(unsigned int);
        writeThreadBuffer(thrIdx, compressedLength);
    }


    size_t totalWritten = 0;
// entries are always separated by a null byte
    if (addNullByte == true) {
        char nullByte = '\0';
        if(isCompressedDB && state[thrIdx]==NOTCOMPRESSED){
            nullByte = static_cast<char>(0xFF);
        }
        const size_t written = fwrite(&nullByte, sizeof(char), 1, dataFiles[thrIdx]);
        if (written != 1) {
            Debug(Debug::ERROR) << "Can not write to data file " << dataFileNames[thrIdx] << "\n";
            EXIT(EXIT_FAILURE);
        }
        totalWritten += written;
        offsets[thrIdx] += 1;
    }

    if (addIndexEntry == true) {
        size_t length = offsets[thrIdx] - starts[thrIdx];
// keep original size in index
        if (isCompressedDB && state[thrIdx]==COMPRESSED) {
            ZSTD_frameProgression progression = ZSTD_getFrameProgression(cstream[thrIdx]);
            length = progression.consumed + totalWritten;
        }
        if (isCompressedDB && state[thrIdx]==NOTCOMPRESSED) {
            length -= sizeof(unsigned int);
        }
        writeIndexEntry(key, starts[thrIdx], length, thrIdx);
    }
}

void DBWriter::writeIndexEntry(DBKeyType key, size_t offset, size_t length, unsigned int thrIdx){
    char buffer[1024];
    size_t len = indexToBuffer(buffer, key, offset, length );
    size_t written = fwrite(buffer, sizeof(char), len, indexFiles[thrIdx]);
    if (written != len) {
        Debug(Debug::ERROR) << "Can not write to data file " << dataFileName[thrIdx] << "\n";
        EXIT(EXIT_FAILURE);
    }
}


void DBWriter::writeData(const char *data, size_t dataSize, DBKeyType key, unsigned int thrIdx, bool addNullByte, bool addIndexEntry) {
    writeStart(thrIdx);
    writeAdd(data, dataSize, thrIdx);
    writeEnd(key, thrIdx, addNullByte, addIndexEntry);
}

size_t DBWriter::indexToBuffer(char *buff1, DBKeyType key, size_t offsetStart, size_t len){
    char * basePos = buff1;
    char * tmpBuff = Itoa::u64toa_sse2(static_cast<uint64_t>(key), buff1);
    *(tmpBuff-1) = '\t';
    tmpBuff = Itoa::u64toa_sse2(static_cast<uint64_t>(offsetStart), tmpBuff);
    *(tmpBuff-1) = '\t';
    tmpBuff = Itoa::u64toa_sse2(static_cast<uint64_t>(len), tmpBuff);
    *(tmpBuff-1) = '\n';
    *(tmpBuff) = '\0';
    return tmpBuff - basePos;
}

void DBWriter::alignToPageSize(int thrIdx) {
    size_t currentOffset = offsets[thrIdx];
    size_t pageSize = Util::getPageSize();
    size_t newOffset = ((pageSize - 1) & currentOffset) ? ((currentOffset + pageSize) & ~(pageSize - 1)) : currentOffset;
    char nullByte = '\0';
    for (size_t i = currentOffset; i < newOffset; ++i) {
        size_t written = fwrite(&nullByte, sizeof(char), 1, dataFiles[thrIdx]);
        if (written != 1) {
            Debug(Debug::ERROR) << "Can not write to data file " << dataFileNames[thrIdx] << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
    offsets[thrIdx] = newOffset;
}


void DBWriter::checkClosed() {
    if (closed == true) {
        Debug(Debug::ERROR) << "Trying to read a closed database. Datafile=" << dataFileName  << "\n";
        EXIT(EXIT_FAILURE);
    }
}

void DBWriter::mergeResults(const std::string &outFileName, const std::string &outFileNameIndex,
                            const std::vector<std::pair<std::string, std::string >> &files,
                            const bool lexicographicOrder) {
    const char **datafilesNames = new const char *[files.size()];
    const char **indexFilesNames = new const char *[files.size()];
    for (size_t i = 0; i < files.size(); i++) {
        datafilesNames[i] = files[i].first.c_str();
        indexFilesNames[i] = files[i].second.c_str();
    }
    mergeResults(outFileName.c_str(), outFileNameIndex.c_str(), datafilesNames, indexFilesNames, files.size(), true, lexicographicOrder);
    delete[] datafilesNames;
    delete[] indexFilesNames;

    // leave only one dbtype file behind
    if (files.size() > 0) {
        std::string typeSrc = files[0].first + ".dbtype";
        std::string typeDest = outFileName + ".dbtype";
        if (FileUtil::fileExists(typeSrc.c_str())) {
            std::rename(typeSrc.c_str(), typeDest.c_str());
        }
        for (size_t i = 1; i < files.size(); i++) {
            std::string typeFile = files[i].first + ".dbtype";
            if (FileUtil::fileExists(typeFile.c_str())) {
                FileUtil::remove(typeFile.c_str());
            }
        }
    }
}

template <>
void DBWriter::writeIndexEntryToFile(FILE *outFile, char *buff1, DBReader<DBKeyType>::Index &index){
    char * tmpBuff = Itoa::u64toa_sse2(static_cast<uint64_t>(index.id), buff1);
    *(tmpBuff-1) = '\t';
    size_t currOffset = index.offset;
    tmpBuff = Itoa::u64toa_sse2(currOffset, tmpBuff);
    *(tmpBuff-1) = '\t';
    uint32_t sLen = index.length;
    tmpBuff = Itoa::u32toa_sse2(sLen,tmpBuff);
    *(tmpBuff-1) = '\n';
    *(tmpBuff) = '\0';
    fwrite(buff1, sizeof(char), (tmpBuff - buff1), outFile);
}

template <>
void DBWriter::writeIndexEntryToFile(FILE *outFile, char *buff1, DBReader<std::string>::Index &index)
{
    size_t keyLen = index.id.length();
    char * tmpBuff = (char*)memcpy((void*)buff1, (void*)index.id.c_str(), keyLen);
    tmpBuff+=keyLen;
    *(tmpBuff) = '\t';
    tmpBuff++;
    size_t currOffset = index.offset;
    tmpBuff = Itoa::u64toa_sse2(currOffset, tmpBuff);
    *(tmpBuff-1) = '\t';
    uint32_t sLen = index.length;
    tmpBuff = Itoa::u32toa_sse2(sLen,tmpBuff);
    *(tmpBuff-1) = '\n';
    *(tmpBuff) = '\0';
    fwrite(buff1, sizeof(char), (tmpBuff - buff1), outFile);
}

// one fwrite per entry pays a stdio lock per entry, which is what dominates a billion-entry index
static const size_t INDEX_WRITE_BUFFER_SIZE = 8 * 1024 * 1024;

static void flushIndexBuffer(FILE *outFile, const char *buffer, size_t used) {
    if (used == 0) {
        return;
    }
    if (fwrite(buffer, sizeof(char), used, outFile) != used) {
        Debug(Debug::ERROR) << "Can not write index buffer\n";
        EXIT(EXIT_FAILURE);
    }
}

// three itoa per record is 210 s per pass at six billion entries, and the flush order keeps it serial identical
template <>
void DBWriter::writeIndex(FILE *outFile, size_t indexSize, DBReader<DBKeyType>::Index *index, unsigned int threads) {
    // three u64 as decimal plus two tabs, a newline and the terminator indexToBuffer appends
    const size_t maxEntry = 3 * 20 + 4;
    const size_t stride = INDEX_WRITE_BUFFER_SIZE + maxEntry;
    const size_t perWorker = INDEX_WRITE_BUFFER_SIZE / maxEntry;
    const size_t workers = std::max<size_t>(1, std::min<size_t>(threads, (indexSize / perWorker) + 1));
    std::vector<char> buffers(workers * stride);
    std::vector<size_t> used(workers, 0);
    for (size_t start = 0; start < indexSize; start += workers * perWorker) {
#pragma omp parallel for schedule(static) num_threads(workers)
        for (size_t worker = 0; worker < workers; worker++) {
            const size_t from = std::min(indexSize, start + worker * perWorker);
            const size_t to = std::min(indexSize, from + perWorker);
            char *out = &buffers[worker * stride];
            size_t pos = 0;
            for (size_t id = from; id < to; id++) {
                pos += indexToBuffer(out + pos, index[id].id, index[id].offset, index[id].length);
            }
            used[worker] = pos;
        }
        for (size_t worker = 0; worker < workers; worker++) {
            flushIndexBuffer(outFile, &buffers[worker * stride], used[worker]);
        }
    }
}

template <>
void DBWriter::writeIndex(FILE *outFile, size_t indexSize, DBReader<std::string>::Index *index, unsigned int){
    // the key is a string here, so the flush point has to account for its length
    const size_t maxTail = 20 + 10 + 4;
    char *buffer = (char *) malloc(INDEX_WRITE_BUFFER_SIZE + maxTail);
    Util::checkAllocation(buffer, "Can not allocate index write buffer");
    size_t capacity = INDEX_WRITE_BUFFER_SIZE + maxTail;
    size_t used = 0;
    for (size_t id = 0; id < indexSize; id++) {
        const size_t keyLen = index[id].id.length();
        if (used + keyLen + maxTail > capacity) {
            flushIndexBuffer(outFile, buffer, used);
            used = 0;
            if (keyLen + maxTail > capacity) {
                capacity = keyLen + maxTail;
                free(buffer);
                buffer = (char *) malloc(capacity);
                Util::checkAllocation(buffer, "Can not allocate index write buffer");
            }
        }
        char *tmpBuff = buffer + used;
        memcpy(tmpBuff, index[id].id.c_str(), keyLen);
        tmpBuff += keyLen;
        *(tmpBuff) = '\t';
        tmpBuff++;
        tmpBuff = Itoa::u64toa_sse2(index[id].offset, tmpBuff);
        *(tmpBuff-1) = '\t';
        tmpBuff = Itoa::u32toa_sse2(static_cast<uint32_t>(index[id].length), tmpBuff);
        *(tmpBuff-1) = '\n';
        used = tmpBuff - buffer;
    }
    flushIndexBuffer(outFile, buffer, used);
    free(buffer);
}


// batching the rows costs one fwrite instead of one per row, and lets writeIndex format in parallel
void DBWriter::writeIndexEntries(DBReader<DBKeyType>::Index *index, size_t indexSize, unsigned int thrIdx,
                                 unsigned int threads) {
    writeIndex(indexFiles[thrIdx], indexSize, index, threads);
}

void DBWriter::mergeResults(const char *outFileName, const char *outFileNameIndex,
                            const char **dataFileNames, const char **indexFileNames,
                            unsigned long fileCount, bool mergeDatafiles,
                            bool lexicographicOrder, bool indexNeedsToBeSorted) {
    Timer timer;
    std::vector<std::vector<std::string>> dataFilenames;
    for (unsigned int i = 0; i < fileCount; ++i) {
        dataFilenames.emplace_back(FileUtil::findDatafiles(dataFileNames[i]));
    }

    // merge results into one result file
    DBReader<DBKeyType>::Index *mergedIndex = NULL;
    size_t mergedIndexSize = 0;
    if (dataFilenames.size() > 1) {
        std::vector<FILE*> datafiles;
        std::vector<size_t> mergedSizes;
        for (unsigned int i = 0; i < dataFilenames.size(); i++) {
            std::vector<std::string>& filenames = dataFilenames[i];
            size_t cumulativeSize = 0;
            for (size_t j = 0; j < filenames.size(); ++j) {
                FILE* fh = fopen(filenames[j].c_str(), "r");
                if (fh == NULL) {
                    Debug(Debug::ERROR) << "Can not open result file " << filenames[j] << "!\n";
                    EXIT(EXIT_FAILURE);
                }
                struct stat sb;
                if (fstat(fileno(fh), &sb) < 0) {
                    int errsv = errno;
                    Debug(Debug::ERROR) << "Failed to fstat file " << filenames[j] << ". Error " << errsv << ".\n";
                    EXIT(EXIT_FAILURE);
                }
                datafiles.emplace_back(fh);
                cumulativeSize += sb.st_size;
            }
            mergedSizes.push_back(cumulativeSize);
        }

        if (mergeDatafiles) {
            FILE *outFh = FileUtil::openAndDelete(outFileName, "w");
            Concat::concatFiles(datafiles, outFh);
            if (fclose(outFh) != 0) {
                Debug(Debug::ERROR) << "Cannot close data file " << outFileName << "\n";
                EXIT(EXIT_FAILURE);
            }
        }

        for (unsigned int i = 0; i < datafiles.size(); ++i) {
            if (fclose(datafiles[i]) != 0) {
                Debug(Debug::ERROR) << "Cannot close data file in merge\n";
                EXIT(EXIT_FAILURE);
            }
        }

        if (mergeDatafiles) {
            for (unsigned int i = 0; i < dataFilenames.size(); i++) {
                std::vector<std::string>& filenames = dataFilenames[i];
                for (size_t j = 0; j < filenames.size(); ++j) {
                    FileUtil::remove(filenames[j].c_str());
                }
            }
        }

        // merge index
        if (indexNeedsToBeSorted && lexicographicOrder == false) {
            // the sort needs every entry resident anyway, so skip formatting rows just to parse them back
            mergedIndex = mergeIndexInMemory(indexFileNames, dataFilenames.size(), mergedSizes, fileCount,
                                             mergedIndexSize);
        } else {
            mergeIndex(indexFileNames, dataFilenames.size(), mergedSizes, fileCount);
        }
    } else if (dataFilenames.size() == 1) {
        std::vector<std::string>& filenames = dataFilenames[0];
        if (filenames.size() == 1) {
            // In single thread dbreader mode it will create a .0
            // that should be moved to the final destination dest instead of dest.0
            FileUtil::move(filenames[0].c_str(), outFileName);
        } else {
            DBReader<DBKeyType>::moveDatafiles(filenames, outFileName);
        }
    } else {
        FILE *outFh = FileUtil::openAndDelete(outFileName, "w");
        if (fclose(outFh) != 0) {
            Debug(Debug::ERROR) << "Cannot close data file " << outFileName << "\n";
            EXIT(EXIT_FAILURE);
        }
        outFh = FileUtil::openAndDelete(outFileNameIndex, "w");
        if (fclose(outFh) != 0) {
            Debug(Debug::ERROR) << "Cannot close index file " << outFileNameIndex << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
    if (dataFilenames.size() > 0) {
        if (mergedIndex != NULL) {
            // mergeIndexInMemory already removed every per thread index file
            sortIndex(mergedIndex, mergedIndexSize, outFileNameIndex, fileCount);
            delete[] mergedIndex;
        } else if (indexNeedsToBeSorted) {
            DBWriter::sortIndex(indexFileNames[0], outFileNameIndex, lexicographicOrder, fileCount);
            FileUtil::remove(indexFileNames[0]);
        } else {
            FileUtil::move(indexFileNames[0], outFileNameIndex);
        }
    }
    Debug(Debug::INFO) << "Time for merging to " << FileUtil::baseName(outFileName) << ": " << timer.lap() << "\n";
}

void DBWriter::mergeIndex(const char** indexFilenames, unsigned int fileCount, const std::vector<size_t> &dataSizes,
                          unsigned int threads) {
    FILE *index_file = fopen(indexFilenames[0], "a");
    if (index_file == NULL) {
        perror(indexFilenames[0]);
        EXIT(EXIT_FAILURE);
    }
    size_t globalOffset = dataSizes[0];
    for (unsigned int fileIdx = 1; fileIdx < fileCount; fileIdx++) {
        DBReader<DBKeyType> reader(indexFilenames[fileIdx], indexFilenames[fileIdx], threads, DBReader<DBKeyType>::USE_INDEX);
        reader.open(DBReader<DBKeyType>::HARDNOSORT);
        if (reader.getSize() > 0) {
            DBReader<DBKeyType>::Index * index = reader.getIndex();
            for (size_t i = 0; i < reader.getSize(); i++) {
                size_t currOffset = index[i].offset;
                index[i].offset = globalOffset + currOffset;
            }
            writeIndex(index_file, reader.getSize(), index, threads);
        }
        reader.close();
        FileUtil::remove(indexFilenames[fileIdx]);

        globalOffset += dataSizes[fileIdx];
    }
    if (fclose(index_file) != 0) {
        Debug(Debug::ERROR) << "Cannot close index file " << indexFilenames[0] << "\n";
        EXIT(EXIT_FAILURE);
    }
}

DBReader<DBKeyType>::Index *DBWriter::mergeIndexInMemory(const char **indexFilenames, unsigned int fileCount,
                                                         const std::vector<size_t> &dataSizes, unsigned int threads,
                                                         size_t &mergedSize) {
    // one sequential scan per file to size the array, the files themselves running in parallel
    std::vector<size_t> sizes(fileCount, 0);
    const int countThreads = std::max(1u, std::min(threads, fileCount));
#pragma omp parallel for schedule(dynamic, 1) num_threads(countThreads)
    for (size_t fileIdx = 0; fileIdx < fileCount; fileIdx++) {
        MemoryMapped indexData(indexFilenames[fileIdx], MemoryMapped::WholeFile, MemoryMapped::SequentialScan);
        if (indexData.isValid() == false) {
            Debug(Debug::ERROR) << "Cannot open index file " << indexFilenames[fileIdx] << "\n";
            EXIT(EXIT_FAILURE);
        }
        sizes[fileIdx] = Util::countLines((char *) indexData.getData(), indexData.size());
        indexData.close();
    }

    std::vector<size_t> prefix(fileCount + 1, 0);
    for (unsigned int fileIdx = 0; fileIdx < fileCount; fileIdx++) {
        prefix[fileIdx + 1] = prefix[fileIdx] + sizes[fileIdx];
    }
    mergedSize = prefix[fileCount];

    // an empty db still has to return a pointer, so the caller can tell this route was taken
    DBReader<DBKeyType>::Index *index = new(std::nothrow) DBReader<DBKeyType>::Index[std::max<size_t>(1, mergedSize)];
    Util::checkAllocation(index, "Cannot allocate merged index memory in DBWriter");

    size_t globalOffset = 0;
    for (unsigned int fileIdx = 0; fileIdx < fileCount; fileIdx++) {
        if (sizes[fileIdx] > 0) {
            DBReader<DBKeyType> reader(indexFilenames[fileIdx], indexFilenames[fileIdx], threads,
                                       DBReader<DBKeyType>::USE_INDEX);
            reader.open(DBReader<DBKeyType>::HARDNOSORT);
            if (reader.getSize() != sizes[fileIdx]) {
                Debug(Debug::ERROR) << "Index file " << indexFilenames[fileIdx] << " changed while merging\n";
                EXIT(EXIT_FAILURE);
            }
            // file fileIdx lands at prefix[fileIdx], so the sort sees the old concatenation order
            const DBReader<DBKeyType>::Index *src = reader.getIndex();
            DBReader<DBKeyType>::Index *dst = index + prefix[fileIdx];
#pragma omp parallel for schedule(static) num_threads(threads)
            for (size_t i = 0; i < sizes[fileIdx]; i++) {
                dst[i] = src[i];
                dst[i].offset += globalOffset;
            }
            reader.close();
        }
        FileUtil::remove(indexFilenames[fileIdx]);
        globalOffset += dataSizes[fileIdx];
    }
    return index;
}

void DBWriter::sortIndex(DBReader<DBKeyType>::Index *index, size_t indexSize, const char *outFileNameIndex,
                         unsigned int threads) {
    SORT_PARALLEL(index, index + indexSize, DBReader<DBKeyType>::Index::compareById);
    FILE *index_file = FileUtil::openAndDelete(outFileNameIndex, "w");
    writeIndex(index_file, indexSize, index, threads);
    if (fclose(index_file) != 0) {
        Debug(Debug::ERROR) << "Cannot close index file " << outFileNameIndex << "\n";
        EXIT(EXIT_FAILURE);
    }
}

void DBWriter::sortIndex(const char *inFileNameIndex, const char *outFileNameIndex, const bool lexicographicOrder,
                         unsigned int threads){
    if (lexicographicOrder == false) {
        // sort the index
        // only the written order matters here, so sort the array and skip HARDNOSORT's serial permutation
        DBReader<DBKeyType> indexReader(inFileNameIndex, inFileNameIndex, threads, DBReader<DBKeyType>::USE_INDEX);
        indexReader.open(DBReader<DBKeyType>::HARDNOSORT);
        DBReader<DBKeyType>::Index *index = indexReader.getIndex();
        SORT_PARALLEL(index, index + indexReader.getSize(), DBReader<DBKeyType>::Index::compareById);
        FILE *index_file  = FileUtil::openAndDelete(outFileNameIndex, "w");
        writeIndex(index_file, indexReader.getSize(), index, threads);
        if (fclose(index_file) != 0) {
            Debug(Debug::ERROR) << "Cannot close index file " << outFileNameIndex << "\n";
            EXIT(EXIT_FAILURE);
        }
        indexReader.close();

    } else {
        DBReader<std::string> indexReader(inFileNameIndex, inFileNameIndex, 1, DBReader<std::string>::USE_INDEX);
        indexReader.open(DBReader<std::string>::SORT_BY_ID);
        DBReader<std::string>::Index *index = indexReader.getIndex();
        FILE *index_file  = FileUtil::openAndDelete(outFileNameIndex, "w");
        writeIndex(index_file, indexReader.getSize(), index);
        if (fclose(index_file) != 0) {
            Debug(Debug::ERROR) << "Cannot close index file " << outFileNameIndex << "\n";
            EXIT(EXIT_FAILURE);
        }
        indexReader.close();
    }
}

void DBWriter::writeThreadBuffer(unsigned int idx, size_t dataSize) {
    size_t written = fwrite(threadBuffer[idx], 1, dataSize, dataFiles[idx]);
    if (written != dataSize) {
        Debug(Debug::ERROR) << "writeThreadBuffer: Could not write to data file " << dataFileNames[idx] << "\n";
        EXIT(EXIT_FAILURE);
    }
}

void DBWriter::createRenumberedDB(const std::string& dataFile, const std::string& indexFile, const std::string& origData, const std::string& origIndex, int sortMode) {
    DBReader<DBKeyType>* lookupReader = NULL;
    FILE *sLookup = NULL;
    if (origData.empty() == false && origIndex.empty() == false) {
        lookupReader = new DBReader<DBKeyType>(origData.c_str(), origIndex.c_str(), 1, DBReader<DBKeyType>::USE_LOOKUP);
        lookupReader->open(DBReader<DBKeyType>::NOSORT);
        sLookup = FileUtil::openAndDelete((dataFile + ".lookup").c_str(), "w");
    }

    DBReader<DBKeyType> reader(dataFile.c_str(), indexFile.c_str(), 1, DBReader<DBKeyType>::USE_INDEX);
    reader.open(sortMode);
    std::string indexTmp = indexFile + "_tmp";
    FILE *sIndex = FileUtil::openAndDelete(indexTmp.c_str(), "w");

    char buffer[1024];
    std::string strBuffer;
    strBuffer.reserve(1024);
    DBReader<DBKeyType>::LookupEntry* lookup = NULL;
    if (lookupReader != NULL) {
        lookup = lookupReader->getLookup();
    }
    for (size_t i = 0; i < reader.getSize(); i++) {
        DBReader<DBKeyType>::Index *idx = (reader.getIndex(i));
        size_t len = DBWriter::indexToBuffer(buffer, static_cast<DBKeyType>(i), idx->offset, idx->length);
        int written = fwrite(buffer, sizeof(char), len, sIndex);
        if (written != (int) len) {
            Debug(Debug::ERROR) << "Can not write to data file " << indexFile << "_tmp\n";
            EXIT(EXIT_FAILURE);
        }
        if (lookupReader != NULL) {
            size_t lookupId = lookupReader->getLookupIdByKey(idx->id);
            DBReader<DBKeyType>::LookupEntry copy = lookup[lookupId];
            copy.id = static_cast<DBKeyType>(i);
            copy.entryName = SSTR(idx->id);
            lookupReader->lookupEntryToBuffer(strBuffer, copy);
            written = fwrite(strBuffer.c_str(), sizeof(char), strBuffer.size(), sLookup);
            if (written != (int) strBuffer.size()) {
                Debug(Debug::ERROR) << "Could not write to lookup file " << indexFile << "_tmp\n";
                EXIT(EXIT_FAILURE);
            }
            strBuffer.clear();
        }
    }
    if (fclose(sIndex) != 0) {
        Debug(Debug::ERROR) << "Cannot close index file " << indexTmp << "\n";
        EXIT(EXIT_FAILURE);
    }
    reader.close();
    std::rename(indexTmp.c_str(), indexFile.c_str());

    if (lookupReader != NULL) {
        if (fclose(sLookup) != 0) {
            Debug(Debug::ERROR) << "Cannot close file " << dataFile << ".lookup\n";
            EXIT(EXIT_FAILURE);
        }
        lookupReader->close();
        delete lookupReader;
    }
}
