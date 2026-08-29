#include "Lin8DbReader.h"

#include "Debug.h"
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include "FileUtil.h"
#include "Util.h"
#include <fcntl.h>
#include <sys/stat.h>

// ---- IoRing ----
#if defined(__linux__) && defined(HAVE_LINUX_IO_URING)

namespace {
struct Ring {
    int fd;
    unsigned *sqTail, *sqMask, *sqArray;
    unsigned *cqHead, *cqTail, *cqMask;
    struct io_uring_sqe *sqes;
    struct io_uring_cqe *cqes;
    void *sqPtr; size_t sqSize;
    void *cqPtr; size_t cqSize;
    void *sqePtr; size_t sqeSize;
    unsigned entries;
};

bool ringOpen(Ring &r, unsigned entries) {
    struct io_uring_params p;
    memset(&p, 0, sizeof(p));
    const int fd = syscall(__NR_io_uring_setup, entries, &p);
    if (fd < 0) {
        return false;
    }
    // IORING_OP_READ is 5.6 and later, so a 5.1 ring that accepts setup still has to use pread
    const size_t probeSize = sizeof(struct io_uring_probe)
                             + (IORING_OP_READ + 1) * sizeof(struct io_uring_probe_op);
    struct io_uring_probe *probe = (struct io_uring_probe *) calloc(1, probeSize);
    if (probe == NULL) {
        ::close(fd);
        return false;
    }
    const bool readSupported =
        syscall(__NR_io_uring_register, fd, IORING_REGISTER_PROBE, probe, IORING_OP_READ + 1) >= 0
        && probe->ops_len > IORING_OP_READ
        && (probe->ops[IORING_OP_READ].flags & IO_URING_OP_SUPPORTED) != 0;
    free(probe);
    if (readSupported == false) {
        ::close(fd);
        return false;
    }
    size_t sqSize = p.sq_off.array + p.sq_entries * sizeof(unsigned);
    size_t cqSize = p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe);
    if (p.features & IORING_FEAT_SINGLE_MMAP) {
        sqSize = (cqSize > sqSize) ? cqSize : sqSize;
        cqSize = sqSize;
    }
    void *sq = mmap(NULL, sqSize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd,
                    IORING_OFF_SQ_RING);
    if (sq == MAP_FAILED) {
        ::close(fd);
        return false;
    }
    void *cq = sq;
    if ((p.features & IORING_FEAT_SINGLE_MMAP) == 0) {
        cq = mmap(NULL, cqSize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd,
                  IORING_OFF_CQ_RING);
        if (cq == MAP_FAILED) {
            munmap(sq, sqSize);
            ::close(fd);
            return false;
        }
    }
    const size_t sqeSize = p.sq_entries * sizeof(struct io_uring_sqe);
    void *sqes = mmap(NULL, sqeSize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd,
                      IORING_OFF_SQES);
    if (sqes == MAP_FAILED) {
        if (cq != sq) {
            munmap(cq, cqSize);
        }
        munmap(sq, sqSize);
        ::close(fd);
        return false;
    }
    r.fd = fd;
    r.sqPtr = sq;    r.sqSize = sqSize;
    r.cqPtr = cq;    r.cqSize = cqSize;
    r.sqePtr = sqes; r.sqeSize = sqeSize;
    r.entries = p.sq_entries;
    r.sqTail  = (unsigned *) ((char *) sq + p.sq_off.tail);
    r.sqMask  = (unsigned *) ((char *) sq + p.sq_off.ring_mask);
    r.sqArray = (unsigned *) ((char *) sq + p.sq_off.array);
    r.cqHead  = (unsigned *) ((char *) cq + p.cq_off.head);
    r.cqTail  = (unsigned *) ((char *) cq + p.cq_off.tail);
    r.cqMask  = (unsigned *) ((char *) cq + p.cq_off.ring_mask);
    r.cqes    = (struct io_uring_cqe *) ((char *) cq + p.cq_off.cqes);
    r.sqes    = (struct io_uring_sqe *) sqes;
    return true;
}

void ringClose(Ring &r) {
    munmap(r.sqePtr, r.sqeSize);
    if (r.cqPtr != r.sqPtr) {
        munmap(r.cqPtr, r.cqSize);
    }
    munmap(r.sqPtr, r.sqSize);
    ::close(r.fd);
}
}  // namespace
#endif

IoRing::IoRing() : ready(false), depth(0), state(NULL) {}

IoRing::~IoRing() {
#if defined(__linux__) && defined(HAVE_LINUX_IO_URING)
    if (state != NULL) {
        Ring *r = static_cast<Ring *>(state);
        if (ready) {
            ringClose(*r);
        }
        delete r;
    }
#endif
}

bool IoRing::open(unsigned depth) {
    this->depth = depth;
#if defined(__linux__) && defined(HAVE_LINUX_IO_URING)
    Ring *r = new Ring();
    ready = ringOpen(*r, depth);
    if (ready == false) {
        delete r;
        r = NULL;
    }
    state = r;
#else
    (void) depth;
#endif
    return ready;
}

void IoRing::preadAll(const std::vector<Read> &reads, const char *what) {
    for (size_t i = 0; i < reads.size(); i++) {
        size_t got = 0;
        while (got < reads[i].length) {
            const ssize_t n = pread(reads[i].fd, static_cast<char *>(reads[i].into) + got,
                                    reads[i].length - got, reads[i].offset + got);
            if (n == 0) {
                break;
            }
            if (n < 0) {
                Debug(Debug::ERROR) << "Cannot read " << reads[i].length << " byte from " << what
                                    << " at " << reads[i].offset << "\n";
                EXIT(EXIT_FAILURE);
            }
            got += static_cast<size_t>(n);
        }
        if (got < reads[i].required) {
            Debug(Debug::ERROR) << "Short read of " << got << " byte from " << what << " at "
                                << reads[i].offset << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
}

void IoRing::submit(const std::vector<Read> &reads, const char *what) {
    if (reads.empty()) {
        return;
    }
#if defined(__linux__) && defined(HAVE_LINUX_IO_URING)
    if (ready == false) {
        preadAll(reads, what);
        return;
    }
    Ring &r = *static_cast<Ring *>(state);
    const unsigned sqMask = *r.sqMask;
    const unsigned cqMask = *r.cqMask;
    size_t next = 0;
    size_t done = 0;
    unsigned inflight = 0;
    while (done < reads.size()) {
        unsigned queued = 0;
        while (next < reads.size() && inflight + queued < r.entries) {
            const unsigned sqeIdx = (unsigned) ((inflight + queued) % r.entries);
            struct io_uring_sqe *sqe = &r.sqes[sqeIdx];
            memset(sqe, 0, sizeof(*sqe));
            sqe->opcode = IORING_OP_READ;
            sqe->fd = reads[next].fd;
            sqe->off = reads[next].offset;
            sqe->addr = (uint64_t) (uintptr_t) reads[next].into;
            sqe->len = static_cast<unsigned>(reads[next].length);
            sqe->user_data = next;
            const unsigned tail = *r.sqTail;
            r.sqArray[tail & sqMask] = sqeIdx;
            __atomic_store_n(r.sqTail, tail + 1, __ATOMIC_RELEASE);
            queued++;
            next++;
        }
        const unsigned waitFor = (next >= reads.size()) ? (inflight + queued) : 1u;
        long ret;
        do {
            ret = syscall(__NR_io_uring_enter, r.fd, queued, waitFor, IORING_ENTER_GETEVENTS, NULL, 0);
        } while (ret < 0 && errno == EINTR);
        if (ret < 0) {
            Debug(Debug::ERROR) << "Cannot submit reads for " << what << ". Error " << errno << "\n";
            EXIT(EXIT_FAILURE);
        }
        inflight += queued;
        unsigned head = *r.cqHead;
        const unsigned cqTail = __atomic_load_n(r.cqTail, __ATOMIC_ACQUIRE);
        while (head != cqTail) {
            struct io_uring_cqe *cqe = &r.cqes[head & cqMask];
            if (cqe->res < 0) {
                Debug(Debug::ERROR) << "Cannot read " << what << ". Error " << -cqe->res << "\n";
                EXIT(EXIT_FAILURE);
            }
            const size_t which = static_cast<size_t>(cqe->user_data);
            if (static_cast<size_t>(cqe->res) < reads[which].required) {
                Debug(Debug::ERROR) << "Short read of " << cqe->res << " byte from " << what
                                    << " at " << reads[which].offset << "\n";
                EXIT(EXIT_FAILURE);
            }
            head++;
            inflight--;
            done++;
        }
        __atomic_store_n(r.cqHead, head, __ATOMIC_RELEASE);
    }
#else
    preadAll(reads, what);
#endif
}


const uint64_t RunDbReader::VALID_MAGIC = 0x4C494E4356414C44ull;

namespace {
size_t sizeOf(const std::string &path, bool &exists) {
    struct stat sb;
    exists = (stat(path.c_str(), &sb) == 0);
    return exists ? static_cast<size_t>(sb.st_size) : 0;
}

void unmapAndDrop(char *&at, int &fd, size_t length) {
    if (at != NULL) {
        munmap(at, length);
        at = NULL;
    }
    if (fd >= 0) {
#if defined(__linux__)
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
        close(fd);
        fd = -1;
    }
}

char *mapWhole(const std::string &path, size_t length, int &keptFd) {
    keptFd = -1;
    if (length == 0) {
        return NULL;
    }
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        Debug(Debug::ERROR) << "Cannot open " << path << "\n";
        EXIT(EXIT_FAILURE);
    }
    char *at = static_cast<char *>(mmap(NULL, length, PROT_READ, MAP_PRIVATE, fd, 0));
    keptFd = fd;
    if (at == MAP_FAILED) {
        close(fd);
        Debug(Debug::ERROR) << "Cannot map " << path << "\n";
        EXIT(EXIT_FAILURE);
    }
    Util::madviseLogged(at, length, POSIX_MADV_SEQUENTIAL, path.c_str());
    return at;
}
}

RunDbReader::RunDbReader(const std::string &db, const std::string &validPath, bool withHeaders)
    : db(db), validPath(validPath), withHeaders(withHeaders), valid(NULL),
      validMap(NULL), validSize(0), validCount(0), validLoaded(false) {}

RunDbReader::~RunDbReader() {
    close();
}

void RunDbReader::open() {
    runs.read(db + ".runs");
    runs.checkLengthsDescend();
    // the layout states how many files there are, so a missing one is an error and not a shorter db
    for (unsigned int i = 0; i < runs.fileCount(); i++) {
        bool exists = false;
        const size_t length = sizeOf(db + "." + SSTR(i), exists);
        if (exists == false) {
            Debug(Debug::ERROR) << "Data file " << (db + "." + SSTR(i)) << " is missing, the run "
                                << "table declares " << runs.fileCount() << " of them\n";
            EXIT(EXIT_FAILURE);
        }
        data.push_back(NULL);
        dataFd.push_back(-1);
        dataSize.push_back(length);
    }
    if (withHeaders) {
        for (unsigned int i = 0; i < data.size(); i++) {
            bool exists = false;
            headerSize.push_back(sizeOf(db + "_h." + SSTR(i), exists));
            headers.push_back(NULL);
            headerFd.push_back(-1);
            if (exists == false) {
                Debug(Debug::ERROR) << "Header file " << (db + "_h." + SSTR(i)) << " is missing\n";
                EXIT(EXIT_FAILURE);
            }
        }
    }
    // an absent bitmap means every entry is valid, which is the state a fresh database is in
    if (validPath.empty() == false) {
        const int fd = ::open(validPath.c_str(), O_RDONLY);
        if (fd < 0) {
            Debug(Debug::ERROR) << "Cannot open the valid bitmap " << validPath << "\n";
            EXIT(EXIT_FAILURE);
        }
        // the bitmap has to say which database it belongs to, or a stale one is read past its end
        uint64_t header[3];
        const uint64_t words = runs.entryCount() / 64 + (runs.entryCount() % 64 != 0);
        struct stat sb;
        if (fstat(fd, &sb) != 0
            || (size_t) sb.st_size < sizeof(header) + words * sizeof(uint64_t)) {
            Debug(Debug::ERROR) << "Bitmap " << validPath << " is truncated\n";
            EXIT(EXIT_FAILURE);
        }
        if (pread(fd, header, sizeof(header), 0) != (ssize_t) sizeof(header)
            || header[0] != VALID_MAGIC) {
            Debug(Debug::ERROR) << "File " << validPath << " is not a valid bitmap\n";
            EXIT(EXIT_FAILURE);
        }
        if (header[1] != runs.entryCount()) {
            Debug(Debug::ERROR) << "Bitmap " << validPath << " covers " << header[1]
                                << " sequences, " << db << " holds " << runs.entryCount() << "\n";
            EXIT(EXIT_FAILURE);
        }
        // One bit a sequence is a hundred and twenty five gigabytes at a trillion, and a pass reads
        // the bits of the ranks it owns. Mapped, it costs the pages it touches; read, it costs all
        // of them on every node and against no budget.
        validSize = sizeof(header) + words * sizeof(uint64_t);
        void *at = mmap(NULL, validSize, PROT_READ, MAP_PRIVATE, fd, 0);
        if (at == MAP_FAILED) {
            Debug(Debug::ERROR) << "Cannot map " << validPath << ", " << validSize << " byte\n";
            EXIT(EXIT_FAILURE);
        }
        ::close(fd);
        validMap = at;
        valid = reinterpret_cast<const uint64_t *>(static_cast<const char *>(at) + sizeof(header));
        validCount = words;
        validLoaded = true;
    }
}

void RunDbReader::close() {
    if (validMap != NULL) {
        munmap(validMap, validSize);
        validMap = NULL;
        valid = NULL;
        validLoaded = false;
    }
    for (size_t i = 0; i < batch.size(); i++) {
        delete batch[i];
    }
    batch.clear();
    for (size_t i = 0; i < directFd.size(); i++) {
        if (directFd[i] >= 0) {
            ::close(directFd[i]);
            directFd[i] = -1;
        }
    }
    for (size_t i = 0; i < data.size(); i++) {
        unmapAndDrop(data[i], dataFd[i], dataSize[i]);
    }
    for (size_t i = 0; i < headers.size(); i++) {
        unmapAndDrop(headers[i], headerFd[i], headerSize[i]);
    }
    dataFd.clear();
    headerFd.clear();
    data.clear();
    dataSize.clear();
    headers.clear();
    headerSize.clear();
}

// the fast path reads a pointer another thread may be publishing, so that read is atomic too
void RunDbReader::mapFile(uint32_t file) const {
    char *at = NULL;
#pragma omp atomic read
    at = data[file];
    if (at != NULL || dataSize[file] == 0) {
        return;
    }
#pragma omp critical(rundb_map)
    {
        if (data[file] == NULL) {
            int fd = -1;
            char *mapped = mapWhole(db + "." + SSTR(file), dataSize[file], fd);
            dataFd[file] = fd;
#pragma omp atomic write
            data[file] = mapped;
        }
    }
}

void RunDbReader::mapHeader(uint32_t file) const {
    char *at = NULL;
#pragma omp atomic read
    at = headers[file];
    if (at != NULL || headerSize[file] == 0) {
        return;
    }
#pragma omp critical(rundb_map)
    {
        if (headers[file] == NULL) {
            int fd = -1;
            char *mapped = mapWhole(db + "_h." + SSTR(file), headerSize[file], fd);
            headerFd[file] = fd;
#pragma omp atomic write
            headers[file] = mapped;
        }
    }
}

const char *RunDbReader::fileData(uint32_t file, uint64_t offset) const {
    if (file >= data.size() || offset > dataSize[file]) {
        Debug(Debug::ERROR) << "Run table points at file " << file << " offset " << offset
                            << ", outside the " << data.size() << " data files of " << db << "\n";
        EXIT(EXIT_FAILURE);
    }
    mapFile(file);
    return data[file] + offset;
}

const char *RunDbReader::getData(uint64_t rank) const {
    const size_t segment = runs.segmentOf(rank);
    return fileData(runs[segment].fileIdx(), runs.offsetIn(segment, rank));
}

uint32_t RunDbReader::getSeqLen(uint64_t rank, Cursor &cursor) const {
    cursor.at = runs.segmentOfFrom(rank, cursor.at);
    return runs[cursor.at].seqLen();
}

const char *RunDbReader::getData(uint64_t rank, Cursor &cursor) const {
    cursor.at = runs.segmentOfFrom(rank, cursor.at);
    return fileData(runs[cursor.at].fileIdx(), runs.offsetIn(cursor.at, rank));
}

bool RunDbReader::isValid(uint64_t rank) const {
    if (validLoaded == false) {
        return true;
    }
    return (valid[rank >> 6] & (uint64_t(1) << (rank & 63))) != 0;
}


void RunDbReader::releaseLengthBlock(size_t lengthBlock) {
    for (size_t file = lengthBlock; file < data.size(); file += runs.blocksPerNode()) {
        unmapAndDrop(data[file], dataFd[file], dataSize[file]);
        if (file < headers.size()) {
            unmapAndDrop(headers[file], headerFd[file], headerSize[file]);
        }
    }
}

uint64_t RunDbReader::countValid() const {
    if (validLoaded == false) {
        return runs.entryCount();
    }
    uint64_t count = 0;
    for (size_t i = 0; i < validCount; i++) {
        count += static_cast<uint64_t>(__builtin_popcountll(valid[i]));
    }
    return count;
}

RunDbReader::HeaderStream::HeaderStream(const RunDbReader &owner)
    : owner(owner), segment(0), left(0), at(0) {
    if (owner.headers.empty()) {
        Debug(Debug::ERROR) << "Headers of " << owner.db << " were not opened\n";
        EXIT(EXIT_FAILURE);
    }
}

bool RunDbReader::HeaderStream::next(const char *&begin, size_t &length) {
    while (left == 0) {
        if (segment >= owner.runs.size()) {
            return false;
        }
        left = owner.runs.rankEnd(segment) - owner.runs[segment].rankBase();
        at = owner.runs[segment].hdrBase();
        segment++;
    }
    const uint32_t file = owner.runs[segment - 1].fileIdx();
    if (file >= owner.headers.size() || at >= owner.headerSize[file]) {
        Debug(Debug::ERROR) << "Segment " << (segment - 1) << " of " << owner.db
                            << " points past header file " << file << "\n";
        EXIT(EXIT_FAILURE);
    }
    owner.mapHeader(file);
    const char *from = owner.headers[file] + at;
    const char *end = static_cast<const char *>(memchr(from, '\n', owner.headerSize[file] - at));
    if (end == NULL) {
        Debug(Debug::ERROR) << "Header file " << file << " of " << owner.db
                            << " does not end with a newline\n";
        EXIT(EXIT_FAILURE);
    }
    begin = from;
    length = static_cast<size_t>(end - from);
    at += length + 1;
    left--;
    return true;
}

static const size_t DIRECT_BLOCK = 512;
static const unsigned RING_DEPTH = 256;

void RunDbReader::openBatch(unsigned int threads, size_t arenaBytes) {
    directFd.assign(data.size(), -1);
    for (unsigned int i = 0; i < threads; i++) {
        BatchWorker *worker = new BatchWorker();
        // room to grow every span outward to whole blocks
        worker->arena.resize(arenaBytes + DIRECT_BLOCK);
        char *at = worker->arena.data();
        const size_t off = reinterpret_cast<uintptr_t>(at) % DIRECT_BLOCK;
        worker->aligned = at + (off == 0 ? 0 : DIRECT_BLOCK - off);
        worker->ring.open(RING_DEPTH);
        batch.push_back(worker);
    }
}

// opened on demand, so a pass that never batches never opens them
int RunDbReader::directOf(uint32_t file) const {
    if (directFd[file] >= 0) {
        return directFd[file];
    }
    const std::string path = db + "." + SSTR(file);
    int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) {
        // a filesystem that refuses direct reads still has to work, just with the page cache
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            Debug(Debug::ERROR) << "Cannot open " << path << " for reading\n";
            EXIT(EXIT_FAILURE);
        }
    }
    directFd[file] = fd;
    return fd;
}

size_t RunDbReader::loadBatch(const uint64_t *ranks, size_t n, unsigned int thread) const {
    BatchWorker &worker = *batch[thread];
    worker.slots.clear();
    worker.reads.clear();
    const size_t room = worker.arena.size() - DIRECT_BLOCK;
    size_t used = 0;
    Cursor cursor;
    size_t taken = 0;
    for (; taken < n; taken++) {
        cursor.at = runs.segmentOfFrom(ranks[taken], cursor.at);
        const uint32_t file = runs[cursor.at].fileIdx();
        const uint64_t offset = runs.offsetIn(cursor.at, ranks[taken]);
        const size_t length = runs[cursor.at].seqLen();
        const uint64_t blockFrom = offset - offset % DIRECT_BLOCK;
        const uint64_t blockUntil =
            ((offset + length + DIRECT_BLOCK - 1) / DIRECT_BLOCK) * DIRECT_BLOCK;

        BatchSlot slot;
        if (worker.reads.empty() == false) {
            BatchRead &last = worker.reads.back();
            const uint64_t end = last.offset + last.length;
            if (last.file == file && blockFrom <= end) {
                const size_t extra = (blockUntil > end) ? (size_t) (blockUntil - end) : 0;
                if (used + extra > room) {
                    break;
                }
                last.length += extra;
                last.required = (size_t) (offset + length - last.offset);
                used += extra;
                slot.arenaOffset = last.arenaOffset + (size_t) (offset - last.offset);
                slot.length = length;
                worker.slots.push_back(slot);
                continue;
            }
        }
        const size_t span = (size_t) (blockUntil - blockFrom);
        if (used + span > room) {
            break;
        }
        BatchRead read;
        read.arenaOffset = used;
        read.file = file;
        read.offset = blockFrom;
        read.length = span;
        read.required = (size_t) (offset + length - blockFrom);
        worker.reads.push_back(read);
        slot.arenaOffset = used + (size_t) (offset - blockFrom);
        slot.length = length;
        worker.slots.push_back(slot);
        used += span;
    }

    std::vector<IoRing::Read> submit(worker.reads.size());
    for (size_t i = 0; i < worker.reads.size(); i++) {
        submit[i].into = worker.aligned + worker.reads[i].arenaOffset;
        submit[i].fd = directOf(worker.reads[i].file);
        submit[i].offset = worker.reads[i].offset;
        submit[i].length = worker.reads[i].length;
        submit[i].required = worker.reads[i].required;
    }
    worker.ring.submit(submit, db.c_str());
    return taken;
}

const char *RunDbReader::batchAt(unsigned int thread, size_t k) const {
    return batch[thread]->aligned + batch[thread]->slots[k].arenaOffset;
}
