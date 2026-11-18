#include "KSeqWrapper.h"
#include "kseq.h"
#include "FileUtil.h"
#include "MMseqsNode.h"
#include "Util.h"
#include "Debug.h"
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

// a pipe has no size
static int openInput(const char *fileName, size_t &size) {
    int fd = MMseqsNode::openSharedFd(fileName, O_RDONLY);
    struct stat sb;
    if (fstat(fd, &sb) != 0) {
        Debug(Debug::ERROR) << "Cannot read " << fileName << ": " << strerror(errno) << "\n";
        EXIT(EXIT_FAILURE);
    }
    size = S_ISREG(sb.st_mode) ? sb.st_size : SIZE_MAX;
    return fd;
}

namespace KSEQFILE {
    // a file is read at its offset with the retries of preadAll, a pipe as it comes
    static ssize_t readInput(KSeqFile::InputFile *file, char *buffer, int length) {
        if (file->size == SIZE_MAX) {
            return read(file->fd, buffer, length);
        }
        size_t size = std::min((size_t) length, file->size - file->offset);
        FileUtil::preadAll(file->fd, file->name.c_str(), buffer, size, file->offset);
        file->offset += size;
        return size;
    }
    KSEQ_INIT(KSeqFile::InputFile*, readInput)
}

KSeqFile::KSeqFile(const char* fileName, size_t start, size_t end) : end(end) {
    file.fd = openInput(fileName, file.size);
    file.name = fileName;
    file.offset = start;
    seq = (void*) KSEQFILE::kseq_init(&file);
    type = KSEQ_FILE;
    if (start > 0) {
        KSEQFILE::kseq_t* s = (KSEQFILE::kseq_t*) seq;
        s->f->cur_buf_pos = start;
        int c;
        while ((c = KSEQFILE::ks_getc(s->f)) >= 0 && c != '\n');
    }
}

bool KSeqFile::ReadEntry() {
    KSEQFILE::kseq_t* s = (KSEQFILE::kseq_t*) seq;
    int64_t result = KSEQFILE::kseq_read(s);
    // kseq ends at a read error as at the end of the input
    if (ks_err(s->f)) {
        Debug(Debug::ERROR) << "Cannot read " << file.name << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (result < 0 || s->headerOffset - 1 > end)
        return false;
    entry.headerOffset = s->headerOffset;
    entry.sequenceOffset = s->sequenceOffset;
    entry.newlineCount = s->newlineCount;
    entry.name = s->name;
    entry.comment = s->comment;
    entry.sequence = s->seq;
    entry.qual = s->qual;

    return true;
}

KSeqFile::~KSeqFile() {
    kseq_destroy((KSEQFILE::kseq_t*)seq);
    if (close(file.fd) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << file.name << "\n";
        EXIT(EXIT_FAILURE);
    }
}


namespace KSEQSTREAM {
    KSEQ_INIT(int, read)
}

KSeqStream::KSeqStream() {
    seq = (void*) KSEQSTREAM::kseq_init(STDIN_FILENO);
    type = KSEQ_STREAM;
}

bool KSeqStream::ReadEntry() {
    KSEQSTREAM::kseq_t* s = (KSEQSTREAM::kseq_t*) seq;
    if (KSEQSTREAM::kseq_read(s) < 0)
        return false;

    entry.name = s->name;
    entry.comment = s->comment;
    entry.sequence = s->seq;
    entry.qual = s->qual;

    return true;
}

KSeqStream::~KSeqStream() {
    kseq_destroy((KSEQSTREAM::kseq_t*)seq);
}

#ifdef HAVE_ZLIB
namespace KSEQGZIP {
    KSEQ_INIT(gzFile, gzread)
}

KSeqGzip::KSeqGzip(const char* fileName) : fileName(fileName) {
    file = gzdopen(MMseqsNode::openSharedFd(fileName, O_RDONLY), "r");
    if(file == NULL) {
        perror(fileName); EXIT(EXIT_FAILURE);
    }

    seq = (void*) KSEQGZIP::kseq_init(file);
    type = KSEQ_GZIP;
}

bool KSeqGzip::ReadEntry() {
    KSEQGZIP::kseq_t* s = (KSEQGZIP::kseq_t*) seq;
    int64_t result = KSEQGZIP::kseq_read(s);
    // gzread ends a truncated file as the end of the input
    int error = Z_OK;
    const char *message = gzerror(file, &error);
    if (ks_err(s->f) || error != Z_OK) {
        Debug(Debug::ERROR) << "Cannot read " << fileName << ": " << message << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (result < 0)
        return false;

    entry.name = s->name;
    entry.comment = s->comment;
    entry.sequence = s->seq;
    entry.qual = s->qual;
    entry.headerOffset = 0;
    entry.sequenceOffset = 0;
    entry.newlineCount = s->newlineCount;

    return true;
}

KSeqGzip::~KSeqGzip() {
    kseq_destroy((KSEQGZIP::kseq_t*)seq);
    gzclose(file);
}
#endif


#ifdef HAVE_BZLIB
namespace KSEQBZIP {
    KSEQ_INIT(BZFILE *, BZ2_bzread)
}

KSeqBzip::KSeqBzip(const char* fileName) : fileName(fileName) {
    if(FileUtil::fileExists(fileName) == false) {
        errno = ENOENT;
        perror(fileName);
        EXIT(EXIT_FAILURE);
    }
    FILE *fp = FileUtil::openFileOrDie(fileName, "r+b", true);
    int bzError;
    file = BZ2_bzReadOpen(&bzError, fp, 0, 0, NULL, 0);
    if(bzError != 0){
        perror(fileName); EXIT(EXIT_FAILURE);
    }
    seq = (void*) KSEQBZIP::kseq_init(file);
    type = KSEQ_BZIP;
}

bool KSeqBzip::ReadEntry() {
    KSEQBZIP::kseq_t* s = (KSEQBZIP::kseq_t*) seq;
    int64_t result = KSEQBZIP::kseq_read(s);
    if (ks_err(s->f)) {
        Debug(Debug::ERROR) << "Cannot read " << fileName << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (result < 0)
        return false;

    entry.name = s->name;
    entry.comment = s->comment;
    entry.sequence = s->seq;
    entry.qual = s->qual;
    entry.headerOffset = 0;
    entry.sequenceOffset = 0;
    entry.newlineCount = s->newlineCount;

    return true;
}

KSeqBzip::~KSeqBzip() {
    kseq_destroy((KSEQBZIP::kseq_t*)seq);
    int bzError;
    BZ2_bzReadClose(&bzError, file);
}
#endif

namespace KSEQZSTD {
    static int zstdread(KSeqZstd::ZstdFile *file, char *buffer, int length) {
        ZSTD_outBuffer output = {buffer, (size_t) length, 0};
        while (output.pos == 0) {
            if (file->input.pos == file->input.size) {
                size_t readSize = ZSTD_DStreamInSize();
                if (file->compressedOffset < file->compressedEndOffset) {
                    readSize = std::min(readSize, file->compressedEndOffset - file->compressedOffset);
                } else if (file->decompressedEndOffset == SIZE_MAX) {
                    readSize = 0;
                }
                // preadAll retries a shared file system that fails for a while
                readSize = std::min(readSize, file->fileSize - file->compressedOffset);
                FileUtil::preadAll(file->fd, file->fileName.c_str(), (char *) file->input.src, readSize, file->compressedOffset);
                file->input.size = readSize;
                file->input.pos = 0;
                file->compressedOffset += readSize;
                if (readSize == 0 && file->readHintBytes == 0) {
                    break;
                }
            }
            file->readHintBytes = ZSTD_decompressStream(file->dstream, &output, &file->input);
            if (ZSTD_isError(file->readHintBytes)) {
                Debug(Debug::ERROR) << "Cannot read " << file->fileName << ": " << ZSTD_getErrorName(file->readHintBytes) << "\n";
                EXIT(EXIT_FAILURE);
            }
            if (file->input.size == 0 && output.pos == 0) {
                Debug(Debug::ERROR) << "Cannot read " << file->fileName << ": it ends inside a zstd frame\n";
                EXIT(EXIT_FAILURE);
            }
            if (file->decompressedEndOffset == SIZE_MAX && file->compressedOffset >= file->compressedEndOffset
                && file->input.pos == file->input.size && file->readHintBytes == 0) {
                file->decompressedEndOffset = file->decompressedOffset + output.pos;
            }
        }
        file->decompressedOffset += output.pos;
        return output.pos;
    }

    KSEQ_INIT(KSeqZstd::ZstdFile*, zstdread)

    // find the first pzstd frame in [pos, limit)
    static size_t findFrame(int fd, const char *fileName, size_t fileBytes, size_t pos, size_t limit) {
        limit = std::min(limit, fileBytes);
        size_t scanBytes = 1024 * 1024;
        std::vector<char> buffer(scanBytes + 4 * sizeof(uint32_t));
        for (; pos < limit; pos += scanBytes) {
            size_t bytes = std::min(buffer.size(), fileBytes - pos);
            FileUtil::preadAll(fd, fileName, buffer.data(), bytes, pos);
            for (size_t i = 0; i < scanBytes && pos + i < limit && i + 4 * sizeof(uint32_t) <= bytes; i++) {
                uint32_t header[4];
                memcpy(header, buffer.data() + i, sizeof(header));
                if (header[0] != ZSTD_MAGIC_SKIPPABLE_START || header[1] != 4 || header[3] != ZSTD_MAGICNUMBER) {
                    continue;
                }
                size_t next = pos + i + 3 * sizeof(uint32_t) + header[2];
                uint32_t nextHeader[2] = {0, 0};
                if (next + sizeof(nextHeader) <= fileBytes) {
                    FileUtil::preadAll(fd, fileName, (char *) nextHeader, sizeof(nextHeader), next);
                }
                if (next == fileBytes || (nextHeader[0] == ZSTD_MAGIC_SKIPPABLE_START && nextHeader[1] == 4)) {
                    return pos + i;
                }
            }
        }
        return limit;
    }
}

KSeqZstd::KSeqZstd(const char* fileName, size_t start, size_t end) {
    file.fd = openInput(fileName, file.fileSize);
    file.fileName = fileName;
    // a split holds the pzstd frames that start in it
    if (start > 0) {
        start = KSEQZSTD::findFrame(file.fd, fileName, file.fileSize, start, end);
    }
    if (end != SIZE_MAX && start < end) {
        end = KSEQZSTD::findFrame(file.fd, fileName, file.fileSize, end, SIZE_MAX);
    }
    file.dstream = ZSTD_createDStream();
    if (file.dstream == NULL) {
        Debug(Debug::ERROR) << "ZSTD_createDStream() error\n";
        EXIT(EXIT_FAILURE);
    }
    void *input = malloc(ZSTD_DStreamInSize());
    Util::checkAllocation(input, "Can not allocate input memory in KSeqZstd");
    file.input.src = input;
    file.input.size = 0;
    file.input.pos = 0;
    file.readHintBytes = 0;
    file.compressedOffset = start;
    file.compressedEndOffset = end;
    file.decompressedOffset = 0;
    file.decompressedEndOffset = SIZE_MAX;
    seq = (void*) KSEQZSTD::kseq_init(&file);
    type = KSEQ_ZSTD;
    if (start > 0) {
        KSEQZSTD::kseq_t* s = (KSEQZSTD::kseq_t*) seq;
        int c;
        while ((c = KSEQZSTD::ks_getc(s->f)) >= 0 && c != '\n');
    }
}

bool KSeqZstd::ReadEntry() {
    KSEQZSTD::kseq_t* s = (KSEQZSTD::kseq_t*) seq;
    if (KSEQZSTD::kseq_read(s) < 0 || s->headerOffset - 1 > file.decompressedEndOffset)
        return false;

    entry.name = s->name;
    entry.comment = s->comment;
    entry.sequence = s->seq;
    entry.qual = s->qual;
    entry.headerOffset = 0;
    entry.sequenceOffset = 0;
    entry.newlineCount = s->newlineCount;

    return true;
}

KSeqZstd::~KSeqZstd() {
    kseq_destroy((KSEQZSTD::kseq_t*)seq);
    ZSTD_freeDStream(file.dstream);
    free((void*) file.input.src);
    if (close(file.fd) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << file.fileName << "\n";
        EXIT(EXIT_FAILURE);
    }
}

KSeqWrapper* KSeqFactory(const char* file, size_t start, size_t end) {
    KSeqWrapper* kseq = NULL;
    if( strcmp(file, "stdin") == 0 ){
        kseq = new KSeqStream();
        return kseq;
    }

    if(Util::endsWith(".gz", file) == false && Util::endsWith(".bz2", file) == false
       && Util::endsWith(".zst", file) == false && Util::endsWith(".zstd", file) == false) {
        kseq = new KSeqFile(file, start, end);
        return kseq;
    }
#ifdef HAVE_ZLIB
    else if(Util::endsWith(".gz", file) == true) {
        kseq = new KSeqGzip(file);
        return kseq;
    }
#else
    else if(Util::endsWith(".gz", file) == true) {
        Debug(Debug::ERROR) << "MMseqs was not compiled with zlib support. Can not read compressed input!\n";
        EXIT(EXIT_FAILURE);
    }
#endif

#ifdef HAVE_BZLIB
    else if(Util::endsWith(".bz2", file) == true) {
        kseq = new KSeqBzip(file);
        return kseq;
    }
#else
    else if(Util::endsWith(".bz2", file) == true) {
        Debug(Debug::ERROR) << "MMseqs was not compiled with bz2lib support. Can not read compressed input!\n";
        EXIT(EXIT_FAILURE);
    }
#endif
    else if(Util::endsWith(".zst", file) == true || Util::endsWith(".zstd", file) == true) {
        kseq = new KSeqZstd(file, start, end);
        return kseq;
    }

    return kseq;
}

namespace KSEQBUFFER {
    KSEQ_INIT(kseq_buffer_t*, kseq_buffer_reader)
}

KSeqBuffer::KSeqBuffer(const char* buffer, size_t length) {
    d.buffer = (char*)buffer;
    d.length = length;
    d.position = 0;
    seq = (void*) KSEQBUFFER::kseq_init(&d);
    type = KSEQ_BUFFER;
}

bool KSeqBuffer::ReadEntry() {
    KSEQBUFFER::kseq_t* s = (KSEQBUFFER::kseq_t*) seq;
    if (KSEQBUFFER::kseq_read(s) < 0)
        return false;
    entry.headerOffset = s->headerOffset;
    entry.sequenceOffset = s->sequenceOffset;
    entry.newlineCount = s->newlineCount;
    entry.name = s->name;
    entry.comment = s->comment;
    entry.sequence = s->seq;
    entry.qual = s->qual;

    return true;
}

KSeqBuffer::~KSeqBuffer() {
    kseq_destroy((KSEQBUFFER::kseq_t*)seq);
}
