#include "BatchEntryFetcher.h"
#include "Debug.h"
#include "Parameters.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>

// O_DIRECT needs offset, length and buffer address block aligned, 4096 covers 512e and 4Kn devices
static const size_t BATCH_DIRECT_ALIGN = 4096;

// a handler without SA_RESTART turns a signal into EINTR, and the read itself is idempotent
static ssize_t preadRetry(int fd, void *buf, size_t len, size_t off) {
    ssize_t read;
    do {
        read = pread(fd, buf, len, static_cast<off_t>(off));
    } while (read < 0 && errno == EINTR);
    return read;
}

BatchEntryFetcher::BatchEntryFetcher()
    : reader(NULL), align(1), arenaCap(0), enabled(false), stopping(false) {}

BatchEntryFetcher::~BatchEntryFetcher() {
    close();
}

unsigned int BatchEntryFetcher::defaultIoThreads(unsigned int workerThreads) {
    // io threads sit blocked in pread, so they are sized for device depth rather than for cores
    return std::min(128u, std::max(8u, workerThreads * 4));
}

void BatchEntryFetcher::open(DBReader<DBKeyType> *reader, unsigned int workerThreads,
                             unsigned int ioThreads, size_t arenaCapPerWorker, bool directIo,
                             bool batched) {
    close();
    if (reader == NULL || workerThreads == 0) {
        return;
    }
    this->reader = reader;
    workers.resize(workerThreads);
    // compressed and padded both hand back a reader-owned buffer rather than the stored bytes
    if (batched == false || reader->getDataFileCnt() == 0 || reader->isCompressed() != 0
        || (DBReader<DBKeyType>::getExtendedDbtype(reader->getDbtype()) & Parameters::DBTYPE_EXTENDED_GPU)) {
        return;
    }
    this->align = directIo ? BATCH_DIRECT_ALIGN : 1;

    std::vector<std::string> names = reader->getDataFileNames();
    size_t base = 0;
    for (size_t i = 0; i < names.size(); i++) {
#if defined(O_DIRECT)
        int fd = ::open(names[i].c_str(), directIo ? (O_RDONLY | O_DIRECT) : O_RDONLY);
        if (fd < 0 && directIo) {
            // tmpfs and some network filesystems reject O_DIRECT, batching still pays without it
            fd = ::open(names[i].c_str(), O_RDONLY);
        }
#else
        const int fd = ::open(names[i].c_str(), O_RDONLY);
#endif
        if (fd < 0) {
            DBReader<DBKeyType> *keep = reader;
            close();
            this->reader = keep;
            workers.resize(workerThreads);
            return;
        }
#if !defined(O_DIRECT) && defined(F_NOCACHE)
        if (directIo) {
            fcntl(fd, F_NOCACHE, 1);
        }
#endif
        fds.push_back(fd);
        fileBase.push_back(base);
        base += reader->getDataSizeForFile(i);
    }

    // an entry larger than this grows only the arena of the worker that asks for it
    this->arenaCap = std::max(arenaCapPerWorker, align);
    for (unsigned int i = 0; i < workerThreads; i++) {
        reserveArena(workers[i], this->arenaCap);
        workers[i].batch = new Batch();
    }

    stopping = false;
    for (unsigned int i = 0; i < ioThreads; i++) {
        pool.push_back(std::thread(&BatchEntryFetcher::runJobs, this));
    }
    enabled = true;
}

void BatchEntryFetcher::close() {
    if (pool.empty() == false) {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stopping = true;
        }
        queueCv.notify_all();
        for (size_t i = 0; i < pool.size(); i++) {
            pool[i].join();
        }
        pool.clear();
    }
    for (size_t i = 0; i < workers.size(); i++) {
        free(workers[i].arena);
        delete workers[i].batch;
    }
    workers.clear();
    for (size_t i = 0; i < fds.size(); i++) {
        ::close(fds[i]);
    }
    fds.clear();
    fileBase.clear();
    queue.clear();
    stopping = false;
    enabled = false;
    reader = NULL;
}

void BatchEntryFetcher::reserveArena(Worker &worker, size_t bytes) {
    if (bytes <= worker.capacity) {
        return;
    }
    // doubling bounds the number of allocations, posix_memalign is slow under contention
    size_t capacity = std::max(bytes, worker.capacity * 2);
    capacity = (capacity + align - 1) / align * align;
    void *arena = NULL;
    if (posix_memalign(&arena, std::max(align, sizeof(void *)), capacity) != 0) {
        Debug(Debug::ERROR) << "Cannot allocate " << capacity << " byte fetch arena\n";
        EXIT(EXIT_FAILURE);
    }
    free(worker.arena);
    worker.arena = static_cast<char *>(arena);
    worker.capacity = capacity;
}

void BatchEntryFetcher::runJobs() {
    while (true) {
        Job *job = NULL;
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            queueCv.wait(lock, [this] { return stopping || queue.empty() == false; });
            if (queue.empty()) {
                return;
            }
            job = queue.front();
            queue.pop_front();
        }
        const ssize_t read = preadRetry(job->fd, job->buf, job->len, job->off);
        job->avail = (read > 0) ? static_cast<size_t>(read) : 0;
        // the counter has to drop under the batch mutex, so the waiter cannot miss the notify
        Batch *batch = job->batch;
        std::unique_lock<std::mutex> lock(batch->mutex);
        if (read < 0) {
            batch->failed = true;
        }
        if (--batch->remaining == 0) {
            batch->cv.notify_one();
        }
    }
}

size_t BatchEntryFetcher::load(const size_t *ids, size_t n, unsigned int workerIdx) {
    Worker &worker = workers[workerIdx];
    worker.slots.clear();
    worker.jobs.clear();
    if (n == 0) {
        return 0;
    }
    if (enabled == false) {
        Slot slot;
        slot.fallbackId = ids[0];
        worker.slots.push_back(slot);
        return 1;
    }

    const size_t dbSize = reader->getSize();
    size_t used = 0;
    size_t taken = 0;
    for (; taken < n; taken++) {
        if (ids[taken] >= dbSize) {
            Debug(Debug::ERROR) << "Invalid database read for id=" << ids[taken] << ", db size " << dbSize << "\n";
            EXIT(EXIT_FAILURE);
        }
        // one lookup instead of getOffset plus getEntryLen, both of which redo the local id mapping
        const DBReader<DBKeyType>::Index *entry = reader->getIndex(ids[taken]);
        const size_t offset = entry->offset;
        const size_t length = entry->length;
        size_t file = fileBase.size() - 1;
        while (file > 0 && fileBase[file] > offset) {
            file--;
        }
        const size_t fileOffset = offset - fileBase[file];
        const size_t alignedOffset = fileOffset & ~(align - 1);
        const size_t delta = fileOffset - alignedOffset;
        // the extra byte closes off a database whose index is one short of the entry terminator
        const size_t slotLen = (delta + length + 1 + align - 1) & ~(align - 1);
        if (used + slotLen > worker.capacity) {
            if (taken > 0) {
                break;
            }
            // a single entry larger than the arena has to be read anyway, so make room for it
            reserveArena(worker, slotLen);
        }

        Slot slot;
        slot.arenaOffset = used;
        slot.delta = delta;
        slot.avail = length;
        worker.slots.push_back(slot);

        Job job;
        job.fd = fds[file];
        job.buf = worker.arena + used;
        job.len = slotLen;
        job.off = alignedOffset;
        job.avail = 0;
        worker.jobs.push_back(job);

        used += slotLen;
    }

    bool failed = false;
    if (pool.empty() || taken == 1) {
        for (size_t i = 0; i < taken; i++) {
            Job &job = worker.jobs[i];
            const ssize_t read = preadRetry(job.fd, job.buf, job.len, job.off);
            job.avail = (read > 0) ? static_cast<size_t>(read) : 0;
            failed = failed || (read < 0);
        }
    } else {
        Batch *batch = worker.batch;
        {
            // the previous load's io threads may still be releasing this mutex, so take it first
            std::unique_lock<std::mutex> lock(batch->mutex);
            batch->remaining = taken;
            batch->failed = false;
        }
        for (size_t i = 0; i < taken; i++) {
            worker.jobs[i].batch = batch;
        }
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            for (size_t i = 0; i < taken; i++) {
                queue.push_back(&worker.jobs[i]);
            }
        }
        queueCv.notify_all();
        std::unique_lock<std::mutex> lock(batch->mutex);
        batch->cv.wait(lock, [batch] { return batch->remaining == 0; });
        failed = batch->failed;
    }
    if (failed) {
        Debug(Debug::ERROR) << "Failed to read entries from " << reader->getDataFileName()
                            << ". Error " << errno << ".\n";
        EXIT(EXIT_FAILURE);
    }

    for (size_t i = 0; i < taken; i++) {
        Slot &slot = worker.slots[i];
        const size_t got = worker.jobs[i].avail;
        // a short read only happens past the end of the file, so it bounds the entry rather than failing it
        slot.avail = (got > slot.delta) ? std::min(slot.avail, got - slot.delta) : 0;
        worker.arena[slot.arenaOffset + slot.delta + slot.avail] = '\0';
    }
    return taken;
}

BatchEntryFetcher::Cursor::Cursor(BatchEntryFetcher &fetcher, const size_t *ids, size_t count,
                                  unsigned int workerIdx)
    : fetcher(fetcher), ids(ids), count(count), workerIdx(workerIdx), pos(0), chunkStart(0),
      chunkEnd(0), started(false) {}

bool BatchEntryFetcher::Cursor::next() {
    if (started) {
        pos++;
    }
    started = true;
    if (pos >= count) {
        return false;
    }
    if (pos >= chunkEnd) {
        chunkStart = pos;
        chunkEnd = pos + fetcher.load(ids + pos, count - pos, workerIdx);
    }
    return true;
}

size_t BatchEntryFetcher::Cursor::length() const {
    return fetcher.lengthAt(workerIdx, pos - chunkStart);
}

const char *BatchEntryFetcher::Cursor::data() const {
    return fetcher.at(workerIdx, pos - chunkStart);
}

const char *BatchEntryFetcher::at(unsigned int workerIdx, size_t k) const {
    const Slot &slot = workers[workerIdx].slots[k];
    if (enabled == false) {
        return reader->getData(slot.fallbackId, workerIdx);
    }
    return workers[workerIdx].arena + slot.arenaOffset + slot.delta;
}

size_t BatchEntryFetcher::lengthAt(unsigned int workerIdx, size_t k) const {
    const Slot &slot = workers[workerIdx].slots[k];
    return (enabled == false) ? reader->getEntryLen(slot.fallbackId) : slot.avail;
}
