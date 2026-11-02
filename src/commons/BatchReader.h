#ifndef MMSEQS_BATCHREADER_H
#define MMSEQS_BATCHREADER_H

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

// a read is split into requests of at most MAX_READ_REQUEST_SIZE at its multiples in the file
static const size_t MAX_READ_REQUEST_SIZE = 16 * 1024 * 1024;
// the largest read that is in flight at once: starting unaligned, it splits into 64 requests, the smallest io_uring queue (MIN_RING_DEPTH)
static const size_t MAX_INFLIGHT_READ_SIZE = 63 * MAX_READ_REQUEST_SIZE;

#ifdef HAVE_IO_URING
struct io_uring_sqe;
struct io_uring_cqe;
#endif

// the storage of a file decides how many reads are in flight and if the range loads of a large file bypass the page cache
enum StorageKind { STORAGE_LOCAL_FLASH, STORAGE_LOCAL_DISK, STORAGE_MEMORY, STORAGE_PARALLEL_FS, STORAGE_NFS, STORAGE_UNKNOWN };
StorageKind getStorageKind(int fd);
const char *getStorageName(StorageKind storage);
// MMSEQS_IO_MODE=buffered|direct instead of the choice of O_DIRECT by storage and size; every node must see the same
enum IoMode { IO_MODE_AUTO, IO_MODE_BUFFERED, IO_MODE_DIRECT };
IoMode getIoMode();

class BatchReader {
public:
    struct ReadRequest {
        int fileToRead;
        const char *fileName;
        char *destination;
        size_t fileOffset;
        size_t bytesToRead;
        size_t minBytesToRead;
    };

    BatchReader(int maxIoThreads, int fdToDetectStorage);
    ~BatchReader();

    // with io_uring, the thread calling readAsync must call waitAsyncRead
    void readAsync(const std::vector<ReadRequest> &requests);
    void waitAsyncRead();
    void readSync(const std::vector<ReadRequest> &requests);

    StorageKind getStorage() const {
        return storage;
    }
    int getConcurrency() const {
        return concurrencyRequests;
    }

private:
    void startRead(const std::vector<ReadRequest> &requests, bool callingThreadReads);
    void readWithPread();

    StorageKind storage;
    int concurrencyRequests;
    std::vector<ReadRequest> splitReadRequests;
    size_t nextSplitReadRequest;
    // errno of the first failed read, -1 for an unexpected end of file
    int readError;
    const char *readErrorFileName;
    std::vector<std::thread> preadThreads;
#ifdef HAVE_IO_URING
    void readWithRing(bool waitForAll);

    std::thread ringThread;
    int ringFd;
    bool ringWorkersLimited;
    unsigned int inflightReadRequests;
    unsigned int inflightReadLimit;
    char *ring;
    size_t ringSize;
    io_uring_sqe *sqes;
    size_t sqesSize;
    unsigned int *sqHead;
    unsigned int *sqTail;
    unsigned int sqMask;
    unsigned int *cqHead;
    unsigned int *cqTail;
    unsigned int cqMask;
    io_uring_cqe *cqes;
#endif
};

// write copies its bytes to a buffer and threads write the full buffers, sorted by file position, so the callers go on computing until all buffers wait for the storage
class BatchWriter {
public:
    // memoryBytes: the buffers of BUFFER_SIZE, at least two, and one per write thread
    BatchWriter(size_t memoryBytes, int maxIoThreads, int fdToDetectStorage);
    ~BatchWriter();

    void write(int fd, const char *fileName, const char *data, size_t bytes, size_t fileOffset);
    // no caller is in write; afterwards all bytes are written
    void waitWrites();
    // the seconds that the callers of write waited for a buffer, summed over them
    double getWaitSeconds() const {
        return waitSeconds;
    }

private:
    static const size_t BUFFER_SIZE = 64 * 1024 * 1024;
    struct Piece {
        int fd;
        const char *fileName;
        size_t bufferOffset;
        size_t bytes;
        size_t fileOffset;

        static bool compareByFile(const Piece &first, const Piece &second) {
            return first.fd < second.fd || (first.fd == second.fd && first.fileOffset < second.fileOffset);
        }
    };
    struct Buffer {
        char *data;
        size_t bytes;
        // the callers that still copy to it
        size_t copies;
        std::vector<Piece> pieces;
    };
    void queueCopiedBuffer(Buffer *buffer);
    void writeBuffers(char *joinedPieces);

    char *memory;
    std::vector<Buffer> buffers;
    std::vector<Buffer *> freeBuffers;
    std::deque<Buffer *> fullBuffers;
    Buffer *currentBuffer;
    // the buffers that are not free
    size_t busyBuffers;
    bool isClosing;
    double waitSeconds;
    std::mutex bufferMutex;
    std::condition_variable freeCondition;
    std::condition_variable fullCondition;
    std::vector<std::thread> writeThreads;
};

#endif
