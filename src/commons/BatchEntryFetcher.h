#ifndef BATCHENTRYFETCHER_H
#define BATCHENTRYFETCHER_H

// One blocking pread per worker caps the device at a queue depth of one per thread. Tools that know
// a whole list of entry ids before they need any of the bodies hand the list over here instead, so
// many reads are in flight at once. The depth comes from cheap io threads that hold no aligner
// state, which is what raising --threads could not do.
#include "DBReader.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

class BatchEntryFetcher {
public:
    BatchEntryFetcher();
    ~BatchEntryFetcher();

    // directIo bypasses the page cache, for databases far larger than RAM. batched == false, a
    // compressed or padded database, and any file that cannot be opened all fall back to plain
    // DBReader::getData, which stays the better choice while the database still fits in memory.
    // Always call this, even to take the fallback: load() and at() need the per-worker state.
    void open(DBReader<DBKeyType> *reader, unsigned int workerThreads, unsigned int ioThreads,
              size_t arenaCapPerWorker, bool directIo, bool batched);

    void close();

    bool isEnabled() const { return enabled; }

    static unsigned int defaultIoThreads(unsigned int workerThreads);

    static const size_t DEFAULT_ARENA_CAP = 8 * 1024 * 1024;

    // Loads as many of ids[0..n) as the arena holds and returns that count, always at least one.
    // Callers that must keep their own iteration order use this directly and advance a window;
    // Cursor is the convenience wrapper for callers that just walk the list.
    size_t load(const size_t *ids, size_t n, unsigned int workerIdx);

    // k indexes into the prefix loaded by the last load() on this worker
    const char *at(unsigned int workerIdx, size_t k) const;

    size_t lengthAt(unsigned int workerIdx, size_t k) const;

    // Walks ids[0..count) in order, refilling the worker arena whenever it runs out. data() stays
    // valid until the next next() on the same cursor, which is all a per-entry loop body needs.
    class Cursor {
    public:
        Cursor(BatchEntryFetcher &fetcher, const size_t *ids, size_t count, unsigned int workerIdx);

        bool next();

        size_t id() const { return ids[pos]; }

        size_t length() const;

        const char *data() const;

    private:
        BatchEntryFetcher &fetcher;
        const size_t *ids;
        size_t count;
        unsigned int workerIdx;
        size_t pos;
        size_t chunkStart;
        size_t chunkEnd;
        bool started;
    };

private:
    // remaining and failed are guarded by mutex, not atomic, so the notify cannot outlive the waiter
    struct Batch {
        size_t remaining;
        bool failed;
        std::mutex mutex;
        std::condition_variable cv;
    };

    struct Job {
        int fd;
        char *buf;
        size_t len;    // aligned read length
        size_t off;    // aligned file offset
        size_t avail;  // bytes actually returned, written by the io thread
        Batch *batch;
    };

    struct Slot {
        size_t fallbackId;  // only used when the fetcher is disabled and reads go through getData
        size_t arenaOffset;
        size_t delta;  // where the entry starts inside the slot
        size_t avail;  // usable entry bytes, terminator excluded
    };

    struct Worker {
        char *arena;
        size_t capacity;  // grown on demand, so one huge entry does not size every worker arena
        std::vector<Slot> slots;
        std::vector<Job> jobs;
        // outlives every job it is handed to, unlike a batch on the load() stack
        Batch *batch;
        Worker() : arena(NULL), capacity(0), batch(NULL) {}
    };

    void runJobs();

    // grows this worker's arena to hold bytes, the old contents are not needed across a load
    void reserveArena(Worker &worker, size_t bytes);

    DBReader<DBKeyType> *reader;
    std::vector<int> fds;
    std::vector<size_t> fileBase;
    std::vector<Worker> workers;
    size_t align;
    size_t arenaCap;
    bool enabled;

    std::vector<std::thread> pool;
    std::deque<Job *> queue;
    std::mutex queueMutex;
    std::condition_variable queueCv;
    bool stopping;
};

#endif
