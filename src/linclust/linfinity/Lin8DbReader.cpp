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

IoRing::IoRing() : ready(false), depth(0), state(NULL), queued(0), done(0), inflight(0) {}

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

void IoRing::preadAll(const char *what) {
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

// Hands the queued reads to the kernel. With untilDone it stays until every one has come back,
// otherwise it returns as soon as the kernel has taken what fits, which is what leaves the caller
// free to compute while the disk works.
void IoRing::pump(const char *what, bool untilDone) {
#if defined(__linux__) && defined(HAVE_LINUX_IO_URING)
    Ring &r = *static_cast<Ring *>(state);
    const unsigned sqMask = *r.sqMask;
    const unsigned cqMask = *r.cqMask;
    do {
        unsigned toQueue = 0;
        while (queued < reads.size() && inflight + toQueue < r.entries) {
            const unsigned sqeIdx = (unsigned) ((inflight + toQueue) % r.entries);
            struct io_uring_sqe *sqe = &r.sqes[sqeIdx];
            memset(sqe, 0, sizeof(*sqe));
            sqe->opcode = IORING_OP_READ;
            sqe->fd = reads[queued].fd;
            sqe->off = reads[queued].offset;
            sqe->addr = (uint64_t) (uintptr_t) reads[queued].into;
            sqe->len = static_cast<unsigned>(reads[queued].length);
            sqe->user_data = queued;
            const unsigned tail = *r.sqTail;
            r.sqArray[tail & sqMask] = sqeIdx;
            __atomic_store_n(r.sqTail, tail + 1, __ATOMIC_RELEASE);
            toQueue++;
            queued++;
        }
        unsigned waitFor = 0;
        if (untilDone) {
            waitFor = (queued >= reads.size()) ? (inflight + toQueue) : 1u;
        }
        long ret;
        do {
            ret = syscall(__NR_io_uring_enter, r.fd, toQueue, waitFor,
                          waitFor > 0 ? IORING_ENTER_GETEVENTS : 0u, NULL, 0);
        } while (ret < 0 && errno == EINTR);
        if (ret < 0) {
            Debug(Debug::ERROR) << "Cannot submit reads for " << what << ". Error " << errno << "\n";
            EXIT(EXIT_FAILURE);
        }
        inflight += toQueue;
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
    } while (untilDone && done < reads.size());
#else
    (void) what;
    (void) untilDone;
#endif
}

void IoRing::submit(const char *what) {
    queued = 0;
    done = 0;
    inflight = 0;
    if (reads.empty()) {
        return;
    }
#if defined(__linux__) && defined(HAVE_LINUX_IO_URING)
    if (ready == false) {
        preadAll(what);
        done = reads.size();
        return;
    }
    pump(what, false);
#else
    preadAll(what);
    done = reads.size();
#endif
}

void IoRing::await(const char *what) {
    if (done >= reads.size()) {
        return;
    }
#if defined(__linux__) && defined(HAVE_LINUX_IO_URING)
    pump(what, true);
#else
    (void) what;
#endif
}


const uint64_t RunDbReader::VALID_MAGIC = 0x4C494E4356414C44ull;

namespace {
size_t fileSizeIfExists(const std::string &path, bool &exists) {
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

char *mapFileReadOnly(const std::string &path, size_t length, int &keptFd) {
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

const char *RunDbReader::KEPT_BITMAP_SUFFIX = ".clusthash_kept";

RunDbReader::RunDbReader(const std::string &db, bool withHeaders)
    : db(db), withHeaders(withHeaders), valid(NULL),
      validMap(NULL), validSize(0), validCount(0), validLoaded(false),
      wantDirect(true) {}

RunDbReader::~RunDbReader() {
    close();
}

void RunDbReader::open() {
    runs.read(db + ".runs");
    runs.checkLengthsDescend();
    // the layout states how many files there are, so a missing one is an error and not a shorter db
    for (unsigned int i = 0; i < runs.fileCount(); i++) {
        bool exists = false;
        const size_t length = fileSizeIfExists(db + "." + SSTR(i), exists);
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
            headerSize.push_back(fileSizeIfExists(db + "_h." + SSTR(i), exists));
            headers.push_back(NULL);
            headerFd.push_back(-1);
            if (exists == false) {
                Debug(Debug::ERROR) << "Header file " << (db + "_h." + SSTR(i)) << " is missing\n";
                EXIT(EXIT_FAILURE);
            }
        }
    }
    // an absent bitmap means every entry is kept, which is the state a database is in until the
    // redundancy pass has run over it
    const std::string validPath = db + KEPT_BITMAP_SUFFIX;
    {
        const int fd = ::open(validPath.c_str(), O_RDONLY);
        if (fd < 0) {
            return;
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
            char *mapped = mapFileReadOnly(db + "." + SSTR(file), dataSize[file], fd);
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
            char *mapped = mapFileReadOnly(db + "_h." + SSTR(file), headerSize[file], fd);
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
// deep enough for a whole batch in one submission: 64 byte a submission entry makes even a
// hundred threads a few megabyte, and a shallower ring only drip-feeds the same reads
static const unsigned RING_DEPTH = 1024;

void RunDbReader::openBatch(unsigned int threads, size_t arenaBytes,
                            size_t memoryBudget) {
    directFd.assign(data.size(), -1);
    // Direct reads exist so that a database far larger than memory does not fill the page cache with
    // sequences it will not see again and push the read arenas out. A database that fits has the
    // opposite problem: a sequence is a member of several representatives, and every one of them
    // then fetches it from the disk again. Measured on a 22 GB database in 121 GB of memory, direct
    // reads moved 107 GB to read it 4.9 times and the pass took 492s; through the cache it read
    // 47 GB and took 217s, with sixteen of twenty threads running rather than ten.
    //
    // The rule is the size of the sequences against the memory the run was given, so a machine that
    // can hold its database uses the cache and one that cannot does not.
    const size_t arenaTotal = (size_t) threads * (arenaBytes + LANES * DIRECT_BLOCK);
    if (arenaTotal >= memoryBudget) {
        Debug(Debug::ERROR) << "Read arenas for " << threads << " thread need "
                            << (arenaTotal >> 20) << " MB, which is the whole "
                            << (memoryBudget >> 20) << " MB limit\n";
        EXIT(EXIT_FAILURE);
    }
    // the arenas are this pass's own memory, so what is left is what the page cache could hold
    const size_t budget = memoryBudget - arenaTotal;
    const uint64_t sequenceBytes = runs.totalBytes();
    wantDirect = sequenceBytes > budget / 2;
    Debug(Debug::INFO) << "Sequences are " << (sequenceBytes >> 30) << " GB against a "
                       << (budget >> 30) << " GB limit past " << (arenaTotal >> 20)
                       << " MB of read arena, reading "
                       << (wantDirect ? "past the page cache" : "through the page cache") << "\n";
    // The memory a thread gets is split between its lanes, so prefetching costs no more of it than
    // waiting did. A lane holds the query and at least one member, each grown outward to whole
    // blocks, or it could not make progress, and demanding that here is what lets startBatch have
    // no failure case.
    const size_t laneBytes = arenaBytes / LANES;
    const size_t longest = 2 * ((size_t) runs.maxSeqLen() + DIRECT_BLOCK);
    if (laneBytes < longest) {
        Debug(Debug::ERROR) << "A read lane of " << laneBytes << " byte cannot hold two sequences of "
                            << runs.maxSeqLen() << " byte, which needs " << longest << "\n";
        EXIT(EXIT_FAILURE);
    }
    for (unsigned int i = 0; i < threads; i++) {
        BatchWorker *worker = new BatchWorker();
        for (unsigned int l = 0; l < LANES; l++) {
            BatchLane &lane = worker->lane[l];
            // room to grow every span outward to whole blocks
            lane.arena.resize(laneBytes + DIRECT_BLOCK);
            char *at = lane.arena.data();
            const size_t off = reinterpret_cast<uintptr_t>(at) % DIRECT_BLOCK;
            lane.aligned = at + (off == 0 ? 0 : DIRECT_BLOCK - off);
            lane.ring.open(RING_DEPTH);
        }
        batch.push_back(worker);
    }
}

// opened on demand, so a pass that never batches never opens them
// the fast path reads a descriptor another thread may be publishing, so that read is atomic too
int RunDbReader::directOf(uint32_t file) const {
    int fd = -1;
#pragma omp atomic read
    fd = directFd[file];
    if (fd >= 0) {
        return fd;
    }
#pragma omp critical(rundb_direct)
    {
        if (directFd[file] < 0) {
            const std::string path = db + "." + SSTR(file);
            int opened = ::open(path.c_str(), wantDirect ? (O_RDONLY | O_DIRECT) : O_RDONLY);
            if (opened < 0) {
                // a filesystem that refuses direct reads still has to work, just with the page cache
                opened = ::open(path.c_str(), O_RDONLY);
                if (opened < 0) {
                    Debug(Debug::ERROR) << "Cannot open " << path << " for reading\n";
                    EXIT(EXIT_FAILURE);
                }
            }
#pragma omp atomic write
            directFd[file] = opened;
        }
    }
#pragma omp atomic read
    fd = directFd[file];
    return fd;
}

const char *RunDbReader::batchQueryAt(unsigned int thread, unsigned int lane) const {
    return batch[thread]->lane[lane].queryAt;
}

const char *RunDbReader::batchAt(unsigned int thread, unsigned int lane, size_t member) const {
    return batch[thread]->lane[lane].memberAt[member];
}

void RunDbReader::awaitBatch(unsigned int thread, unsigned int lane) const {
    batch[thread]->lane[lane].ring.await(db.c_str());
}

// Grows the last read to cover this rank when the two touch on the disk, and starts a new one when
// they do not. Answers false once the arena is full, which openBatch guarantees cannot happen before
// a query and one member are in.
bool RunDbReader::appendBatchRead(BatchLane &lane, uint64_t rank, Cursor &cursor,
                                  const char *&at) const {
    cursor.at = runs.segmentOfFrom(rank, cursor.at);
    const uint64_t offset = runs.offsetIn(cursor.at, rank);
    const size_t length = runs[cursor.at].seqLen();
    const int fd = directOf(runs[cursor.at].fileIdx());
    const uint64_t blockFrom = offset - offset % DIRECT_BLOCK;
    const uint64_t blockUntil = ((offset + length + DIRECT_BLOCK - 1) / DIRECT_BLOCK) * DIRECT_BLOCK;

    const size_t room = lane.arena.size() - DIRECT_BLOCK;
    size_t used = 0;
    std::vector<IoRing::Read> &reads = lane.ring.list();
    if (reads.empty() == false) {
        IoRing::Read &last = reads.back();
        char *into = static_cast<char *>(last.into);
        used = (size_t) (into - lane.aligned) + last.length;
        const uint64_t end = last.offset + last.length;
        if (last.fd == fd && last.offset <= blockFrom && blockFrom <= end) {
            const size_t extra = (blockUntil > end) ? (size_t) (blockUntil - end) : 0;
            if (used + extra > room) {
                return false;
            }
            last.length += extra;
            last.required = (size_t) (offset + length - last.offset);
            at = into + (size_t) (offset - last.offset);
            return true;
        }
    }
    const size_t span = (size_t) (blockUntil - blockFrom);
    if (used + span > room) {
        return false;
    }
    IoRing::Read read;
    read.into = lane.aligned + used;
    read.fd = fd;
    read.offset = blockFrom;
    read.length = span;
    read.required = (size_t) (offset + length - blockFrom);
    reads.push_back(read);
    at = lane.aligned + used + (size_t) (offset - blockFrom);
    return true;
}

size_t RunDbReader::startBatch(uint64_t queryRank, const uint64_t *members, size_t n,
                              unsigned int thread, unsigned int lane) const {
    BatchLane &at = batch[thread]->lane[lane];
    at.memberAt.clear();
    at.ring.list().clear();
    Cursor cursor;
    // the query goes in first because it is the lowest rank of the batch, which keeps the run cursor
    // and the disk offsets moving forward and lets a member that lands next to it share its read
    appendBatchRead(at, queryRank, cursor, at.queryAt);
    size_t taken = 0;
    for (; taken < n; taken++) {
        const char *where = NULL;
        if (appendBatchRead(at, members[taken], cursor, where) == false) {
            break;
        }
        at.memberAt.push_back(where);
    }
    if (taken == 0 && n > 0) {
        // openBatch sized the lanes so this cannot happen; a hung node costs far more to find than
        // a stopped one, so it is worth the one comparison a batch to say so out loud
        Debug(Debug::ERROR) << "A read lane of " << at.arena.size() << " byte took none of "
                            << n << " member, so the pass would not move\n";
        EXIT(EXIT_FAILURE);
    }
    at.ring.submit(db.c_str());
    return taken;
}

