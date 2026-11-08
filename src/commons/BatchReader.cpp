#include "BatchReader.h"
#include "Debug.h"
#include "FastSort.h"
#include "FileUtil.h"
#include "Util.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>

#ifdef __linux__
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/vfs.h>
#endif

#ifdef HAVE_IO_URING
#include <fcntl.h>
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>

static const unsigned int MAX_RING_DEPTH = 4096;
static const unsigned int MIN_RING_DEPTH = 64;
#endif

StorageKind getStorageKind(int fd) {
#ifdef __linux__
    struct statfs fs;
    if (fstatfs(fd, &fs) != 0) {
        return STORAGE_UNKNOWN;
    }
    // GPFS, Lustre | NFS | tmpfs | ext2/3/4, xfs, btrfs, f2fs, zfs
    switch (static_cast<unsigned int>(fs.f_type)) {
        case 0x47504653: case 0x0BD00BD0:
            return STORAGE_PARALLEL_FS;
        case 0x6969:
            return STORAGE_NFS;
        case 0x01021994:
            return STORAGE_MEMORY;
        case 0xEF53: case 0x58465342: case 0x9123683E: case 0xF2F52010: case 0x2FC12FC1:
            break;
        default:
            return STORAGE_UNKNOWN;
    }
    struct stat st;
    if (fstat(fd, &st) == 0) {
        const char *queues[] = { "queue", "../queue" };
        for (size_t i = 0; i < 2; i++) {
            char path[64];
            snprintf(path, sizeof(path), "/sys/dev/block/%u:%u/%s/rotational", major(st.st_dev), minor(st.st_dev), queues[i]);
            FILE *file = fopen(path, "r");
            if (file != NULL) {
                int rotational = fgetc(file);
                fclose(file);
                return (rotational == '1') ? STORAGE_LOCAL_DISK : STORAGE_LOCAL_FLASH;
            }
        }
    }
#endif
    return STORAGE_LOCAL_FLASH;
}

const char *getStorageName(StorageKind storage) {
    switch (storage) {
        case STORAGE_LOCAL_FLASH: return "local flash";
        case STORAGE_LOCAL_DISK: return "local disk";
        case STORAGE_MEMORY: return "memory";
        case STORAGE_PARALLEL_FS: return "parallel file system";
        case STORAGE_NFS: return "NFS";
        default: return "unknown storage";
    }
}

IoMode getIoMode() {
    const char *mode = getenv("MMSEQS_IO_MODE");
    if (mode == NULL || strcmp(mode, "auto") == 0) {
        return IO_MODE_AUTO;
    }
    if (strcmp(mode, "buffered") == 0) {
        return IO_MODE_BUFFERED;
    }
    if (strcmp(mode, "direct") == 0) {
        return IO_MODE_DIRECT;
    }
    Debug(Debug::ERROR) << "MMSEQS_IO_MODE must be auto, buffered or direct, not " << mode << "\n";
    EXIT(EXIT_FAILURE);
}

static int getStorageMaxInflightRequests(StorageKind storage, int maxIoThreads) {
    const char *concurrency = getenv("MMSEQS_IO_CONCURRENCY");
    if (concurrency != NULL) {
        int value = atoi(concurrency);
        if (value < 1) {
            Debug(Debug::ERROR) << "MMSEQS_IO_CONCURRENCY must be a positive number, not " << concurrency << "\n";
            EXIT(EXIT_FAILURE);
        }
        return value;
    }
    switch (storage) {
        case STORAGE_PARALLEL_FS: return 128;
        case STORAGE_NFS: return 16;
        case STORAGE_LOCAL_DISK: return 1;
        case STORAGE_LOCAL_FLASH: case STORAGE_MEMORY: return maxIoThreads;
        default: return 16;
    }
}

BatchReader::BatchReader(int maxIoThreads, int fdToDetectStorage) : nextSplitReadRequest(0), readError(0), readErrorFileName(NULL) {
    storage = getStorageKind(fdToDetectStorage);
    maxInflightRequests = std::max(1, std::min(getStorageMaxInflightRequests(storage, maxIoThreads), maxIoThreads));
#ifdef HAVE_IO_URING
    ringFd = -1;
    isRingWorkerCountLimited = false;
    inflightReadRequestCount = 0;
    inflightReadLimit = MIN_RING_DEPTH;
    io_uring_params params;
    memset(&params, 0, sizeof(params));
    int fd = syscall(__NR_io_uring_setup, MAX_RING_DEPTH, &params);
    // RLIMIT_MEMLOCK before Linux 5.12
    if (fd < 0) {
        memset(&params, 0, sizeof(params));
        fd = syscall(__NR_io_uring_setup, MIN_RING_DEPTH, &params);
    }
    if (fd < 0) {
        return;
    }
    unsigned int probeOps = IORING_OP_READ + 1;
    io_uring_probe *probe = (io_uring_probe *) calloc(1, sizeof(io_uring_probe) + probeOps * sizeof(io_uring_probe_op));
    bool canRead = probe != NULL && syscall(__NR_io_uring_register, fd, IORING_REGISTER_PROBE, probe, probeOps) >= 0
                         && probe->ops_len > IORING_OP_READ && (probe->ops[IORING_OP_READ].flags & IO_URING_OP_SUPPORTED);
    free(probe);
    if (canRead == false || (params.features & IORING_FEAT_SINGLE_MMAP) == 0) {
        close(fd);
        return;
    }
    ringSize = std::max(params.sq_off.array + params.sq_entries * sizeof(unsigned int), params.cq_off.cqes + params.cq_entries * sizeof(io_uring_cqe));
    ring = (char *) mmap(NULL, ringSize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQ_RING);
    if (ring == MAP_FAILED) {
        close(fd);
        return;
    }
    sqeSize = params.sq_entries * sizeof(io_uring_sqe);
    sqes = (io_uring_sqe *) mmap(NULL, sqeSize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQES);
    if (sqes == MAP_FAILED) {
        munmap(ring, ringSize);
        close(fd);
        return;
    }
    sqHead = (unsigned int *) (ring + params.sq_off.head);
    sqTail = (unsigned int *) (ring + params.sq_off.tail);
    sqMask = *(unsigned int *) (ring + params.sq_off.ring_mask);
    cqHead = (unsigned int *) (ring + params.cq_off.head);
    cqTail = (unsigned int *) (ring + params.cq_off.tail);
    cqMask = *(unsigned int *) (ring + params.cq_off.ring_mask);
    cqes = (io_uring_cqe *) (ring + params.cq_off.cqes);
    unsigned int *sqArray = (unsigned int *) (ring + params.sq_off.array);
    for (unsigned int i = 0; i < params.sq_entries; i++) {
        sqArray[i] = i;
    }
    ringFd = fd;
#endif
}

BatchReader::~BatchReader() {
    waitAsyncRead();
#ifdef HAVE_IO_URING
    if (ringFd >= 0) {
        munmap(sqes, sqeSize);
        munmap(ring, ringSize);
        close(ringFd);
    }
#endif
}

void BatchReader::readAsync(const std::vector<ReadRequest> &requests) {
    startRead(requests, false);
}

void BatchReader::readSync(const std::vector<ReadRequest> &requests) {
    startRead(requests, true);
    waitAsyncRead();
}

void BatchReader::waitAsyncRead() {
#ifdef HAVE_IO_URING
    if (ringThread.joinable()) {
        ringThread.join();
    }
    if (ringFd >= 0) {
        readWithRing(true);
    }
#endif
    readWithPread();
    for (size_t i = 0; i < preadThreads.size(); i++) {
        preadThreads[i].join();
    }
    preadThreads.clear();
    if (readError != 0) {
        Debug(Debug::ERROR) << "Cannot read " << readErrorFileName << ": " << (readError > 0 ? strerror(readError) : "unexpected end of file") << "\n";
        EXIT(EXIT_FAILURE);
    }
}

void BatchReader::startRead(const std::vector<ReadRequest> &requests, bool callingThreadReads) {
    waitAsyncRead();
    splitReadRequests.clear();
    nextSplitReadRequest = 0;
    for (size_t i = 0; i < requests.size(); i++) {
        const ReadRequest &request = requests[i];
        size_t end = request.fileOffset + request.bytesToRead;
        size_t minEnd = request.fileOffset + request.minBytesToRead;
        for (size_t at = request.fileOffset; at < minEnd;) {
            size_t next = std::min((at / MAX_READ_REQUEST_SIZE + 1) * MAX_READ_REQUEST_SIZE, end);
            ReadRequest split = request;
            split.destination += at - request.fileOffset;
            split.fileOffset = at;
            split.bytesToRead = next - at;
            split.minBytesToRead = std::min(next, minEnd) - at;
            splitReadRequests.push_back(split);
            at = next;
        }
    }
#ifdef HAVE_IO_URING
    if (ringFd >= 0) {
        if (isRingWorkerCountLimited == false) {
            unsigned int maxWorkers[2] = { static_cast<unsigned int>(maxInflightRequests), static_cast<unsigned int>(maxInflightRequests) };
            // no cap before Linux 5.15
            isRingWorkerCountLimited = syscall(__NR_io_uring_register, ringFd, IORING_REGISTER_IOWQ_MAX_WORKERS, maxWorkers, 2) >= 0;
        }
        inflightReadLimit = isRingWorkerCountLimited ? sqMask + 1 : MIN_RING_DEPTH;
        // refill a full ring from a thread
        if (callingThreadReads == false && splitReadRequests.size() > inflightReadLimit) {
            ringThread = std::thread(&BatchReader::readWithRing, this, true);
        } else {
            readWithRing(false);
        }
        return;
    }
#endif
    size_t threads = std::min(static_cast<size_t>(maxInflightRequests), splitReadRequests.size());
    for (size_t i = callingThreadReads ? 1 : 0; i < threads; i++) {
        preadThreads.emplace_back(&BatchReader::readWithPread, this);
    }
}

void BatchReader::readWithPread() {
    size_t pageSize = Util::getPageSize();
    while (__atomic_load_n(&readError, __ATOMIC_RELAXED) == 0) {
        size_t i = __sync_fetch_and_add(&nextSplitReadRequest, 1);
        if (i >= splitReadRequests.size()) {
            return;
        }
        const ReadRequest &request = splitReadRequests[i];
        for (size_t done = 0; done < request.minBytesToRead;) {
            ssize_t count = pread(request.fileToRead, request.destination + done, request.bytesToRead - done, request.fileOffset + done);
            if (count < 0 && errno == EINTR) {
                continue;
            }
            if (count < static_cast<ssize_t>(std::min(pageSize, request.minBytesToRead - done))) {
                if (__sync_bool_compare_and_swap(&readError, 0, count < 0 ? errno : -1)) {
                    readErrorFileName = request.fileName;
                }
                return;
            }
            done += count;
            if (done < request.minBytesToRead) {
                done -= done % pageSize;
            }
        }
    }
}

#ifdef HAVE_IO_URING
void BatchReader::readWithRing(bool waitForAll) {
    size_t pageSize = Util::getPageSize();
    do {
        unsigned int tail = *sqTail;
        for (; readError == 0 && nextSplitReadRequest < splitReadRequests.size() && inflightReadRequestCount < inflightReadLimit; nextSplitReadRequest++, inflightReadRequestCount++, tail++) {
            io_uring_sqe &sqe = sqes[tail & sqMask];
            memset(&sqe, 0, sizeof(sqe));
            sqe.opcode = IORING_OP_READ;
            sqe.fd = splitReadRequests[nextSplitReadRequest].fileToRead;
            sqe.off = splitReadRequests[nextSplitReadRequest].fileOffset;
            sqe.addr = reinterpret_cast<unsigned long>(splitReadRequests[nextSplitReadRequest].destination);
            sqe.len = splitReadRequests[nextSplitReadRequest].bytesToRead;
            sqe.flags = (splitReadRequests[nextSplitReadRequest].bytesToRead == MAX_READ_REQUEST_SIZE) ? IOSQE_ASYNC : 0;
            sqe.user_data = nextSplitReadRequest;
        }
        __atomic_store_n(sqTail, tail, __ATOMIC_RELEASE);
        unsigned int toSubmit = tail - __atomic_load_n(sqHead, __ATOMIC_RELAXED);
        unsigned int toWait = (waitForAll && inflightReadRequestCount > 0) ? 1 : 0;
        if (syscall(__NR_io_uring_enter, ringFd, toSubmit, toWait, IORING_ENTER_GETEVENTS, NULL, 0) < 0 && errno != EAGAIN && errno != EINTR) {
            Debug(Debug::ERROR) << "Cannot read with io_uring: " << strerror(errno) << "\n";
            EXIT(EXIT_FAILURE);
        }
        unsigned int head = *cqHead;
        for (const unsigned int end = __atomic_load_n(cqTail, __ATOMIC_ACQUIRE); head != end; head++, inflightReadRequestCount--) {
            const io_uring_cqe &cqe = cqes[head & cqMask];
            ReadRequest rest = splitReadRequests[cqe.user_data];
            if (cqe.res == -EAGAIN || cqe.res == -EINTR) {
                splitReadRequests.push_back(rest);
                continue;
            }
            if (cqe.res < static_cast<int>(std::min(pageSize, rest.minBytesToRead))) {
                if (readError == 0) {
                    readError = (cqe.res < 0) ? -cqe.res : -1;
                    readErrorFileName = rest.fileName;
                }
                continue;
            }
            if (static_cast<size_t>(cqe.res) < rest.minBytesToRead) {
                size_t done = cqe.res - cqe.res % pageSize;
                rest.destination += done;
                rest.fileOffset += done;
                rest.bytesToRead -= done;
                rest.minBytesToRead -= done;
                splitReadRequests.push_back(rest);
            }
        }
        __atomic_store_n(cqHead, head, __ATOMIC_RELEASE);
    } while (waitForAll && (inflightReadRequestCount > 0 || (readError == 0 && nextSplitReadRequest < splitReadRequests.size())));
}
#endif

const size_t BatchReader::MAX_READ_REQUEST_SIZE;
const size_t BatchReader::MAX_INFLIGHT_READ_SIZE;
const size_t BatchWriter::BUFFER_SIZE;

BatchWriter::BatchWriter(size_t memorySize, int maxIoThreads, int fdToDetectStorage)
    : bufferSize(std::min(BUFFER_SIZE, std::max((size_t) 1, memorySize / 3))), currentBuffer(NULL), busyBufferCount(0), isClosing(false) {
    // each thread has a buffer to join writes, so threads take at most half
    size_t threads = std::max((size_t) 1, std::min((size_t) std::min(getStorageMaxInflightRequests(getStorageKind(fdToDetectStorage), maxIoThreads), maxIoThreads), memorySize / bufferSize / 2));
    size_t bufferCount = std::max((size_t) 2, memorySize / bufferSize - std::min(memorySize / bufferSize, threads));
    memory = new(std::nothrow) char[(bufferCount + threads) * bufferSize];
    Util::checkAllocation(memory, "Can not allocate memory in BatchWriter");
    buffers.resize(bufferCount);
    for (size_t i = 0; i < bufferCount; i++) {
        buffers[i].data = memory + i * bufferSize;
        buffers[i].size = 0;
        buffers[i].activeCopyCount = 0;
        freeBuffers.push_back(&buffers[i]);
    }
    for (size_t i = 0; i < threads; i++) {
        writeThreads.emplace_back(&BatchWriter::writeBuffers, this, memory + (bufferCount + i) * bufferSize);
    }
}

BatchWriter::~BatchWriter() {
    waitWrites();
    {
        std::lock_guard<std::mutex> lock(bufferMutex);
        isClosing = true;
    }
    fullCondition.notify_all();
    for (size_t i = 0; i < writeThreads.size(); i++) {
        writeThreads[i].join();
    }
    Util::releasePages(memory, (buffers.size() + writeThreads.size()) * bufferSize);
    delete[] memory;
}

void BatchWriter::queueCopiedBuffer(Buffer *buffer) {
    if (buffer != currentBuffer && buffer->activeCopyCount == 0) {
        fullBuffers.push_back(buffer);
        fullCondition.notify_one();
    }
}

void BatchWriter::write(int fd, const char *fileName, const char *data, size_t size, size_t fileOffset) {
    for (size_t done = 0; done < size;) {
        std::unique_lock<std::mutex> lock(bufferMutex);
        while (currentBuffer == NULL && freeBuffers.empty()) {
            freeCondition.wait(lock);
        }
        if (currentBuffer == NULL) {
            currentBuffer = freeBuffers.back();
            freeBuffers.pop_back();
            busyBufferCount++;
        }
        Buffer *buffer = currentBuffer;
        size_t bufferOffset = buffer->size;
        size_t copySize = std::min(size - done, bufferSize - bufferOffset);
        WriteRequest request = { fd, fileName, bufferOffset, copySize, fileOffset + done };
        buffer->writeRequests.push_back(request);
        buffer->size += copySize;
        buffer->activeCopyCount++;
        if (buffer->size == bufferSize) {
            currentBuffer = NULL;
        }
        lock.unlock();
        memcpy(buffer->data + bufferOffset, data + done, copySize);
        lock.lock();
        buffer->activeCopyCount--;
        queueCopiedBuffer(buffer);
        done += copySize;
    }
}

void BatchWriter::waitWrites() {
    std::unique_lock<std::mutex> lock(bufferMutex);
    if (currentBuffer != NULL) {
        Buffer *buffer = currentBuffer;
        currentBuffer = NULL;
        queueCopiedBuffer(buffer);
    }
    while (busyBufferCount != 0) {
        freeCondition.wait(lock);
    }
}

void BatchWriter::writeBuffers(char *joinedWrites) {
    std::unique_lock<std::mutex> lock(bufferMutex);
    while (true) {
        while (fullBuffers.empty() && isClosing == false) {
            fullCondition.wait(lock);
        }
        if (fullBuffers.empty()) {
            return;
        }
        Buffer *buffer = fullBuffers.front();
        fullBuffers.pop_front();
        lock.unlock();
        std::vector<WriteRequest> &writeRequests = buffer->writeRequests;
        SORT_SERIAL(writeRequests.begin(), writeRequests.end(), WriteRequest::compareByFile);
        // join consecutive writes
        for (size_t first = 0, last = 0; first < writeRequests.size(); first = last) {
            size_t joinedSize = 0;
            for (last = first; last < writeRequests.size() && writeRequests[last].fd == writeRequests[first].fd && writeRequests[last].fileOffset == writeRequests[first].fileOffset + joinedSize; last++) {
                joinedSize += writeRequests[last].size;
            }
            const char *writeData = buffer->data + writeRequests[first].bufferOffset;
            if (last > first + 1) {
                writeData = joinedWrites;
                for (size_t request = first, joinedPos = 0; request < last; joinedPos += writeRequests[request].size, request++) {
                    memcpy(joinedWrites + joinedPos, buffer->data + writeRequests[request].bufferOffset, writeRequests[request].size);
                }
            }
            FileUtil::pwriteAll(writeRequests[first].fd, writeRequests[first].fileName, writeData, joinedSize, writeRequests[first].fileOffset);
        }
        writeRequests.clear();
        buffer->size = 0;
        lock.lock();
        freeBuffers.push_back(buffer);
        busyBufferCount--;
        freeCondition.notify_all();
    }
}
