/*
 * convert2fasta
 * written by Milot Mirdita <milot@mirdita.de>
 */

#include <cstring>
#include <cstdio>
#include <cerrno>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "Parameters.h"
#include "DBReader.h"
#include "Debug.h"
#include "Util.h"
#include "FileUtil.h"

#ifdef OPENMP
#include <omp.h>
#endif

// a worker holds one slice of the output before it lands, so this bounds threads * this much memory
static const size_t CONVERT2FASTA_SLICE_BYTES = 8 * 1024 * 1024;
// MADV_DONTNEED shoots down the TLB on every thread, so reclaim in large steps
static const size_t CONVERT2FASTA_DROP_STRIDE = 256 * 1024 * 1024;

// the whole output would otherwise sit in page cache as dirty until the process exits, which on a
// multi terabyte fasta competes with the index for memory. Start writeback on what was just written
// and release the round before it, so the cache footprint stays flat.
static void retireWritten(int fd, size_t base, size_t bytes, size_t prevBase, size_t prevBytes) {
#if defined(__linux__)
    if (bytes != 0) {
        sync_file_range(fd, static_cast<off_t>(base), static_cast<off_t>(bytes),
                        SYNC_FILE_RANGE_WRITE);
    }
    if (prevBytes != 0) {
        sync_file_range(fd, static_cast<off_t>(prevBase), static_cast<off_t>(prevBytes),
                        SYNC_FILE_RANGE_WAIT_BEFORE | SYNC_FILE_RANGE_WRITE | SYNC_FILE_RANGE_WAIT_AFTER);
        posix_fadvise(fd, static_cast<off_t>(prevBase), static_cast<off_t>(prevBytes),
                      POSIX_FADV_DONTNEED);
    }
#else
    (void) fd; (void) base; (void) bytes; (void) prevBase; (void) prevBytes;
#endif
}

// shard k of "dir/rep.fa" is "dir/rep.split<k>.fa", so every shard keeps a real FASTA extension
static std::string splitFastaName(const std::string &out, int split) {
    const size_t slash = out.find_last_of('/');
    size_t dot = out.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
        dot = out.size();
    }
    char suffix[16];
    snprintf(suffix, sizeof(suffix), ".split%05d", split);
    return out.substr(0, dot) + suffix + out.substr(dot);
}

static void pwriteFully(int fd, const char *src, size_t bytes, size_t offset, const char *name) {
    size_t done = 0;
    while (done < bytes) {
        const ssize_t wrote = pwrite(fd, src + done, bytes - done, static_cast<off_t>(offset + done));
        if (wrote < 0 && errno == EINTR) {
            continue;
        }
        if (wrote <= 0) {
            Debug(Debug::ERROR) << "Cannot write to " << name << "\n";
            EXIT(EXIT_FAILURE);
        }
        done += static_cast<size_t>(wrote);
    }
}

int convert2fasta(int argc, const char **argv, const Command& command) {
    Parameters& par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);

    DBReader<DBKeyType> db(par.db1.c_str(), par.db1Index.c_str(), par.threads, DBReader<DBKeyType>::USE_DATA|DBReader<DBKeyType>::USE_INDEX);
    db.setIoCacheAdvice(true);
    db.open(DBReader<DBKeyType>::NOSORT);
    // a length ordered subdb walks its parent's data front to back, so the readahead is worth having
    // and the pages behind the sweep are never touched again
    db.setSequentialAdvice();

    DBReader<DBKeyType> db_header(par.hdr1.c_str(), par.hdr1Index.c_str(), par.threads, DBReader<DBKeyType>::USE_DATA|DBReader<DBKeyType>::USE_INDEX);
    db_header.setIoCacheAdvice(true);
    db_header.open(DBReader<DBKeyType>::NOSORT);
    db_header.setSequentialAdvice();

    DBReader<DBKeyType>* from = &db;
    if(par.useHeaderFile) {
        from = &db_header;
    }
    // NOSORT leaves the index key ordered and builds no id map, so looking the key back up is the identity
    const bool bodyFollowsFrom = (from == &db);
    const size_t entries = from->getSize();
    const size_t workers = std::max<size_t>(1, static_cast<size_t>(par.threads));

    // entry i goes to shard i % splits, so a worker owns whole shards and no offset is shared
    if (par.fastaSplits > 0) {
        const int splits = par.fastaSplits;
        Debug(Debug::INFO) << "Start writing " << splits << " split files for " << par.db2 << "\n";
        const int splitWorkers = static_cast<int>(std::min<size_t>(workers, static_cast<size_t>(splits)));
#pragma omp parallel for schedule(dynamic, 1) num_threads(splitWorkers)
        for (int split = 0; split < splits; split++) {
            unsigned int thread_idx = 0;
#ifdef OPENMP
            thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
            const std::string name = splitFastaName(par.db2, split);
            const int fd = open(name.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
            if (fd < 0) {
                perror(name.c_str());
                EXIT(EXIT_FAILURE);
            }
            std::string out;
            size_t fileOffset = 0, prevBase = 0, prevBytes = 0;
            for (size_t i = static_cast<size_t>(split); i < entries; i += static_cast<size_t>(splits)) {
                DBKeyType key = from->getDbKey(i);
                size_t headerKey = db_header.getId(key);
                const char* headerData = db_header.getData(headerKey, thread_idx);
                const size_t headerLen = db_header.getEntryLen(headerKey);
                out.append(">", 1);
                out.append(headerData, headerLen - 2);
                out.append("\n", 1);
                size_t bodyKey = bodyFollowsFrom ? i : db.getId(key);
                const char* bodyData = db.getData(bodyKey, thread_idx);
                const size_t bodyLen = db.getEntryLen(bodyKey);
                out.append(bodyData, bodyLen - 2);
                out.append("\n", 1);
                if (out.size() >= CONVERT2FASTA_SLICE_BYTES) {
                    pwriteFully(fd, out.data(), out.size(), fileOffset, name.c_str());
                    retireWritten(fd, fileOffset, out.size(), prevBase, prevBytes);
                    prevBase = fileOffset;
                    prevBytes = out.size();
                    fileOffset += out.size();
                    out.clear();
                }
            }
            if (out.empty() == false) {
                pwriteFully(fd, out.data(), out.size(), fileOffset, name.c_str());
                retireWritten(fd, fileOffset, out.size(), prevBase, prevBytes);
            }
            if (close(fd) != 0) {
                Debug(Debug::ERROR) << "Cannot close file " << name << "\n";
                EXIT(EXIT_FAILURE);
            }
        }
        db_header.close();
        db.close();
        return EXIT_SUCCESS;
    }

    int fastaFd = open(par.db2.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (fastaFd < 0) {
        perror(par.db2.c_str());
        EXIT(EXIT_FAILURE);
    }

    Debug(Debug::INFO) << "Start writing file to " << par.db2 << "\n";
    const DBReader<DBKeyType>::Index *fromIndex = from->getIndex();
    std::vector<std::string> slices(workers);
    std::vector<size_t> bounds(workers + 1, 0);
    std::vector<size_t> sliceOffset(workers, 0);

    size_t at = 0;
    size_t fileOffset = 0;
    size_t droppedBody = 0, droppedHeader = 0;
    size_t roundBase = 0;
    size_t prevBase = 0;
    size_t prevBytes = 0;
    while (at < entries) {
        roundBase = fileOffset;
        // length ordered inputs put the longest entries first, so slice by bytes and not by count
        bounds[0] = at;
        size_t active = 0;
        for (size_t worker = 0; worker < workers && at < entries; worker++) {
            size_t bytes = 0;
            while (at < entries && bytes < CONVERT2FASTA_SLICE_BYTES) {
                bytes += fromIndex[at].length;
                at++;
            }
            bounds[worker + 1] = at;
            active++;
        }

#pragma omp parallel for schedule(static) num_threads(active)
        for (size_t worker = 0; worker < active; worker++) {
            unsigned int thread_idx = 0;
#ifdef OPENMP
            thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
            std::string &out = slices[worker];
            out.clear();
            for (size_t i = bounds[worker]; i < bounds[worker + 1]; i++) {
                DBKeyType key = from->getDbKey(i);
                size_t headerKey = db_header.getId(key);
                const char* headerData = db_header.getData(headerKey, thread_idx);
                const size_t headerLen = db_header.getEntryLen(headerKey);

                out.append(">", 1);
                out.append(headerData, headerLen - 2);
                out.append("\n", 1);

                size_t bodyKey = bodyFollowsFrom ? i : db.getId(key);
                const char* bodyData = db.getData(bodyKey, thread_idx);
                const size_t bodyLen = db.getEntryLen(bodyKey);
                out.append(bodyData, bodyLen - 2);
                out.append("\n", 1);
            }
        }

        for (size_t worker = 0; worker < active; worker++) {
            sliceOffset[worker] = fileOffset;
            fileOffset += slices[worker].size();
        }

#pragma omp parallel for schedule(static) num_threads(active)
        for (size_t worker = 0; worker < active; worker++) {
            pwriteFully(fastaFd, slices[worker].data(), slices[worker].size(),
                        sliceOffset[worker], par.db2.c_str());
        }

        retireWritten(fastaFd, roundBase, fileOffset - roundBase, prevBase, prevBytes);
        prevBase = roundBase;
        prevBytes = fileOffset - roundBase;
        // the sweep is monotone in the parent's offsets, so everything below the round it just left
        // is dead; hand those pages back instead of letting them push the index into swap
        if (at > 0) {
            const size_t bodyUpTo = db.getIndex()[db.getId(from->getDbKey(at - 1))].offset;
            const size_t headerUpTo = db_header.getIndex()[db_header.getId(from->getDbKey(at - 1))].offset;
            if (bodyUpTo > droppedBody + CONVERT2FASTA_DROP_STRIDE) {
                db.dropCacheRange(droppedBody, bodyUpTo);
                droppedBody = bodyUpTo;
            }
            if (headerUpTo > droppedHeader + CONVERT2FASTA_DROP_STRIDE) {
                db_header.dropCacheRange(droppedHeader, headerUpTo);
                droppedHeader = headerUpTo;
            }
        }
    }

    if (close(fastaFd) != 0) {
        Debug(Debug::ERROR) << "Cannot close file " << par.db2 << "\n";
        EXIT(EXIT_FAILURE);
    }
    db_header.close();
    db.close();

    return EXIT_SUCCESS;
}
