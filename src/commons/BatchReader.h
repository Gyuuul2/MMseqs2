#ifndef MMSEQS_BATCHREADER_H
#define MMSEQS_BATCHREADER_H

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#ifdef HAVE_IO_URING
struct io_uring_sqe;
struct io_uring_cqe;
#endif

enum StorageKind { STORAGE_LOCAL_FLASH, STORAGE_LOCAL_DISK, STORAGE_MEMORY, STORAGE_PARALLEL_FS, STORAGE_NFS, STORAGE_UNKNOWN };
StorageKind getStorageKind(int fd);
const char *getStorageName(StorageKind storage);
// MMSEQS_IO_MODE
enum IoMode { IO_MODE_AUTO, IO_MODE_BUFFERED, IO_MODE_DIRECT };
IoMode getIoMode();

class BatchReader {
public:
    // reads are split at multiples of MAX_READ_REQUEST_SIZE
    static const size_t MAX_READ_REQUEST_SIZE = 16 * 1024 * 1024;
    // 64 requests if unaligned, the smallest ring
    static const size_t MAX_INFLIGHT_READ_SIZE = 63 * MAX_READ_REQUEST_SIZE;

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
    int getMaxInflightRequests() const {
        return maxInflightRequests;
    }

private:
    void startRead(const std::vector<ReadRequest> &requests, bool callingThreadReads);
    void readWithPread();

    StorageKind storage;
    int maxInflightRequests;
    std::vector<ReadRequest> splitReadRequests;
    size_t nextSplitReadRequest;
    // aws backend: reads of another node's file go from S3 into the buffers, in the pread threads
    bool hasRemoteReads;
    // errno, -1 for an unexpected end of file
    int readError;
    const char *readErrorFileName;
    std::vector<std::thread> preadThreads;
#ifdef HAVE_IO_URING
    void readWithRing(bool waitForAll);

    std::thread ringThread;
    int ringFd;
    bool isRingWorkerCountLimited;
    unsigned int inflightReadRequestCount;
    unsigned int inflightReadLimit;
    char *ring;
    size_t ringSize;
    io_uring_sqe *sqes;
    size_t sqeSize;
    unsigned int *sqHead;
    unsigned int *sqTail;
    unsigned int sqMask;
    unsigned int *cqHead;
    unsigned int *cqTail;
    unsigned int cqMask;
    io_uring_cqe *cqes;
#endif
};

class BatchWriter {
public:
    BatchWriter(size_t memorySize, int maxIoThreads, int fdToDetectStorage);
    ~BatchWriter();

    void write(int fd, const char *fileName, const char *data, size_t size, size_t fileOffset);
    // no caller may be in write
    void waitWrites();

private:
    static const size_t BUFFER_SIZE = 64 * 1024 * 1024;
    struct WriteRequest {
        int fd;
        const char *fileName;
        size_t bufferOffset;
        size_t size;
        size_t fileOffset;

        static bool compareByFile(const WriteRequest &first, const WriteRequest &second) {
            return first.fd < second.fd || (first.fd == second.fd && first.fileOffset < second.fileOffset);
        }
    };
    struct Buffer {
        char *data;
        size_t size;
        size_t activeCopyCount;
        std::vector<WriteRequest> writeRequests;
    };
    void queueCopiedBuffer(Buffer *buffer);
    void writeBuffers(char *joinedWrites);

    const size_t bufferSize;
    char *memory;
    std::vector<Buffer> buffers;
    std::vector<Buffer *> freeBuffers;
    std::deque<Buffer *> fullBuffers;
    Buffer *currentBuffer;
    size_t busyBufferCount;
    bool isClosing;
    std::mutex bufferMutex;
    std::condition_variable freeCondition;
    std::condition_variable fullCondition;
    std::vector<std::thread> writeThreads;
};

#endif
