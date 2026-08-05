#include "KSeqWrapper.h"
#include "kseq.h"
#include "FileUtil.h"
#include "Util.h"
#include "Debug.h"
#include <unistd.h>
#include <cerrno>
#include <cstdlib>
#include <zstd.h>

namespace KSEQFILE {
    KSEQ_INIT(int, read)
}

KSeqFile::KSeqFile(const char* fileName) {
    file = FileUtil::openFileOrDie(fileName, "r", true);
    seq = (void*) KSEQFILE::kseq_init(fileno(file));
    type = KSEQ_FILE;
}

bool KSeqFile::ReadEntry() {
    KSEQFILE::kseq_t* s = (KSEQFILE::kseq_t*) seq;
    if (KSEQFILE::kseq_read(s) < 0)
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
    if (fclose(file) != 0) {
        Debug(Debug::ERROR) << "Cannot close KSeq input file\n";
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

KSeqGzip::KSeqGzip(const char* fileName) {
    if(FileUtil::fileExists(fileName) == false) {
        errno = ENOENT;
        perror(fileName);
        EXIT(EXIT_FAILURE);
    }

    file = gzopen(fileName, "r");
    if(file == NULL) {
        perror(fileName); EXIT(EXIT_FAILURE);
    }

    seq = (void*) KSEQGZIP::kseq_init(file);
    type = KSEQ_GZIP;
}

bool KSeqGzip::ReadEntry() {
    KSEQGZIP::kseq_t* s = (KSEQGZIP::kseq_t*) seq;
    if (KSEQGZIP::kseq_read(s) < 0)
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

KSeqBzip::KSeqBzip(const char* fileName) {
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
    if (KSEQBZIP::kseq_read(s) < 0)
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


// Wrap zstd streaming decompression into the (handle, buf, len) reader kseq expects.
namespace KSEQZSTD {
    struct ZstdReader {
        FILE* fp;
        ZSTD_DStream* dstream;
        unsigned char* inBuf;   // compressed input staging buffer
        size_t inBufCap;
        ZSTD_inBuffer input;    // { src, size, pos } over inBuf
        int eof;                // 1 once the compressed file is fully read
        int frameComplete;      // set after ZSTD_decompressStream returns 0 for a complete frame
        std::string fileName;
    };

    // Returns decompressed bytes (>0), 0 at EOF, or -1 on error, like read()/gzread().
    static int zstdReaderRead(ZstdReader* r, char* buf, int len) {
        if (len <= 0) {
            return 0;
        }
        ZSTD_outBuffer output = { buf, (size_t) len, 0 };
        // Loop until we produce >=1 byte or hit real EOF (returning 0 early means EOF to kseq).
        while (output.pos == 0) {
            if (r->input.pos == r->input.size && r->eof == 0) {
                size_t n = fread(r->inBuf, 1, r->inBufCap, r->fp);
                if (n == 0 && ferror(r->fp)) {
                    Debug(Debug::ERROR) << "Cannot read zstd file " << r->fileName << "\n";
                    return -1;
                }
                r->input.src = r->inBuf;
                r->input.size = n;
                r->input.pos = 0;
                if (n == 0) {
                    r->eof = 1;
                }
            }
            if (r->input.pos == r->input.size && r->eof) {
                if (r->frameComplete == 0) {
                    Debug(Debug::ERROR) << "Truncated zstd frame in " << r->fileName << "\n";
                    return -1;
                }
                break;  // no more compressed input after a completed frame -> genuine EOF
            }
            size_t code = ZSTD_decompressStream(r->dstream, &output, &r->input);
            if (ZSTD_isError(code)) {
                Debug(Debug::ERROR) << "ZSTD_decompressStream failed for " << r->fileName << ": "
                                    << ZSTD_getErrorName(code) << "\n";
                return -1;
            }
            r->frameComplete = (code == 0);
        }
        return (int) output.pos;
    }

    KSEQ_INIT(ZstdReader*, zstdReaderRead)
}

KSeqZstd::KSeqZstd(const char* fileName) {
    if (FileUtil::fileExists(fileName) == false) {
        errno = ENOENT;
        perror(fileName);
        EXIT(EXIT_FAILURE);
    }
    KSEQZSTD::ZstdReader* r = new KSEQZSTD::ZstdReader();
    r->fp = FileUtil::openFileOrDie(fileName, "rb", true);
    r->dstream = ZSTD_createDStream();
    if (r->dstream == NULL) {
        Debug(Debug::ERROR) << "ZSTD_createDStream() failed for " << fileName << "\n";
        EXIT(EXIT_FAILURE);
    }
    size_t initResult = ZSTD_initDStream(r->dstream);
    if (ZSTD_isError(initResult)) {
        Debug(Debug::ERROR) << "ZSTD_initDStream() failed for " << fileName << ": "
                            << ZSTD_getErrorName(initResult) << "\n";
        EXIT(EXIT_FAILURE);
    }
    r->inBufCap = ZSTD_DStreamInSize();
    r->inBuf = (unsigned char*) malloc(r->inBufCap);
    if (r->inBuf == NULL) {
        Debug(Debug::ERROR) << "Cannot allocate zstd input buffer\n";
        EXIT(EXIT_FAILURE);
    }
    r->input.src = r->inBuf;
    r->input.size = 0;
    r->input.pos = 0;
    r->eof = 0;
    r->frameComplete = 0;
    r->fileName = fileName;
    reader = (void*) r;
    seq = (void*) KSEQZSTD::kseq_init(r);
    type = KSEQ_ZSTD;
}

bool KSeqZstd::ReadEntry() {
    KSEQZSTD::kseq_t* s = (KSEQZSTD::kseq_t*) seq;
    int64_t ret = KSEQZSTD::kseq_read(s);
    if (ret == -1) {
        return false;
    }
    if (ret < 0) {
        KSEQZSTD::ZstdReader* r = (KSEQZSTD::ZstdReader*) reader;
        const std::string name = (r != NULL) ? r->fileName : "<unknown>";
        Debug(Debug::ERROR) << "Error reading zstd FASTA " << name
                            << " (kseq rc=" << ret << ")\n";
        EXIT(EXIT_FAILURE);
    }

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
    KSEQZSTD::ZstdReader* r = (KSEQZSTD::ZstdReader*) reader;
    if (r != NULL) {
        if (r->dstream != NULL) {
            ZSTD_freeDStream(r->dstream);
        }
        if (r->inBuf != NULL) {
            free(r->inBuf);
        }
        if (r->fp != NULL) {
            fclose(r->fp);
        }
        delete r;
    }
}

KSeqWrapper* KSeqFactory(const char* file) {
    KSeqWrapper* kseq = NULL;
    if( strcmp(file, "stdin") == 0 ){
        kseq = new KSeqStream();
        return kseq;
    }

    if(Util::endsWith(".gz", file) == false && Util::endsWith(".bz2", file) == false && Util::endsWith(".zst", file) == false ) {
        kseq = new KSeqFile(file);
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
    else if(Util::endsWith(".zst", file) == true) {
        kseq = new KSeqZstd(file);
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
