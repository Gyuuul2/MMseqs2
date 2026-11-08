#include "Parameters.h"
#include "DBReader.h"
#include "DBWriter.h"
#include "Debug.h"
#include "Util.h"
#include "IndexReader.h"
#include "FileUtil.h"
#include "MMseqsNode.h"
#include "FastSort.h"

#ifdef OPENMP
#include <omp.h>

#endif

#ifndef SIZE_T_MAX
#define SIZE_T_MAX ((size_t) -1)
#endif

struct MemberAccession {
    DBKeyType repKey;
    DBKeyType memberKey;
    size_t accessionOffset;
    unsigned int accessionLength;

    static bool compareByRep(const MemberAccession &first, const MemberAccession &second) {
        ClusterPair firstPair = { first.repKey, first.memberKey };
        ClusterPair secondPair = { second.repKey, second.memberKey };
        return ClusterPair::compareByRep(firstPair, secondPair);
    }
};

static const size_t MEMBER_ACCESSION_HEADER_BYTES = 2 * sizeof(DBKeyType) + sizeof(unsigned int);

// 0 if the record is incomplete
static size_t getMemberAccessionBytes(const char *record, size_t bytes) {
    if (bytes < MEMBER_ACCESSION_HEADER_BYTES) {
        return 0;
    }
    unsigned int accessionLength;
    memcpy(&accessionLength, record + 2 * sizeof(DBKeyType), sizeof(unsigned int));
    return (bytes >= MEMBER_ACCESSION_HEADER_BYTES + accessionLength) ? MEMBER_ACCESSION_HEADER_BYTES + accessionLength : 0;
}

// no zero-fill, the threads touch new elements first
template <typename T>
static void growBuffer(T *&buffer, size_t &capacity, size_t size) {
    if (size > capacity) {
        capacity = std::max(size, 2 * capacity);
        buffer = static_cast<T *>(realloc(buffer, capacity * sizeof(T)));
        Util::checkAllocation(buffer, "Can not allocate buffer memory in createtsv");
    }
}

// file offset, own records are not written
static size_t getFileOffset(size_t offset, size_t ownBegin, size_t ownEnd) {
    return offset - (std::min(offset, ownEnd) - std::min(offset, ownBegin));
}

static size_t writeMemberAccessions(MemberAccession *memberAccessions, size_t accessionCount, const char *accessions, const std::vector<size_t> &repKeyOffsets,
                                    FILE *file, const std::string &fileName, std::string &indexData, size_t &writtenBytes, const std::string &ownExchange, size_t ownBytesLimit) {
    SORT_PARALLEL(memberAccessions, memberAccessions + accessionCount, MemberAccession::compareByRep);
    std::vector<size_t> nodeAccessionOffsets(MMseqsNode::count + 1, accessionCount);
    for (size_t node = 0; node < MMseqsNode::count; node++) {
        MemberAccession nodeFirst = { static_cast<DBKeyType>(repKeyOffsets[node]), static_cast<DBKeyType>(repKeyOffsets[node]), 0, 0 };
        nodeAccessionOffsets[node] = std::lower_bound(memberAccessions, memberAccessions + accessionCount, nodeFirst, MemberAccession::compareByRep) - memberAccessions;
    }

    int threads = Parameters::getInstance().threads;
    std::vector<size_t> threadOffsets(threads + 1, 0);
#pragma omp parallel for schedule(static)
    for (int thread = 0; thread < threads; thread++) {
        for (size_t i = accessionCount * thread / threads; i < accessionCount * (thread + 1) / threads; i++) {
            threadOffsets[thread + 1] += MEMBER_ACCESSION_HEADER_BYTES + memberAccessions[i].accessionLength;
        }
    }
    for (int thread = 0; thread < threads; thread++) {
        threadOffsets[thread + 1] += threadOffsets[thread];
    }

    size_t writeBufferBytes = threadOffsets[threads];
    char *writeBuffer = static_cast<char *>(malloc(writeBufferBytes));
    Util::checkAllocation(writeBuffer, "Can not allocate writeBuffer memory in createtsv");
    std::vector<size_t> nodeOffsets(MMseqsNode::count + 1, writeBufferBytes);
#pragma omp parallel for schedule(static)
    for (int thread = 0; thread < threads; thread++) {
        size_t begin = accessionCount * thread / threads;
        size_t node = std::lower_bound(nodeAccessionOffsets.begin(), nodeAccessionOffsets.end(), begin) - nodeAccessionOffsets.begin();
        size_t bufferPos = threadOffsets[thread];
        for (size_t i = begin; i < accessionCount * (thread + 1) / threads; i++) {
            for (; node <= MMseqsNode::count && nodeAccessionOffsets[node] == i; node++) {
                nodeOffsets[node] = bufferPos;
            }
            const MemberAccession &memberAccession = memberAccessions[i];
            memcpy(writeBuffer + bufferPos, &memberAccession.repKey, sizeof(DBKeyType));
            memcpy(writeBuffer + bufferPos + sizeof(DBKeyType), &memberAccession.memberKey, sizeof(DBKeyType));
            memcpy(writeBuffer + bufferPos + 2 * sizeof(DBKeyType), &memberAccession.accessionLength, sizeof(unsigned int));
            memcpy(writeBuffer + bufferPos + MEMBER_ACCESSION_HEADER_BYTES, accessions + memberAccession.accessionOffset, memberAccession.accessionLength);
            bufferPos += MEMBER_ACCESSION_HEADER_BYTES + memberAccession.accessionLength;
        }
    }

    size_t ownBegin = nodeOffsets[MMseqsNode::rank];
    size_t ownEnd = (nodeOffsets[MMseqsNode::rank + 1] - ownBegin <= ownBytesLimit) ? nodeOffsets[MMseqsNode::rank + 1] : ownBegin;
#pragma omp parallel for schedule(static)
    for (int thread = 0; thread < threads; thread++) {
        size_t begin = threadOffsets[thread];
        size_t end = threadOffsets[thread + 1];
        size_t beforeOwn = std::max(begin, std::min(end, ownBegin));
        size_t afterOwn = std::min(end, std::max(begin, ownEnd));
        FileUtil::pwriteAll(fileno(file), fileName.c_str(), writeBuffer + begin, beforeOwn - begin, writtenBytes + begin);
        FileUtil::pwriteAll(fileno(file), fileName.c_str(), writeBuffer + afterOwn, end - afterOwn, writtenBytes + getFileOffset(afterOwn, ownBegin, ownEnd));
    }

    // index lines per thread group, so receivers scan on all threads
    int indexLines = std::max(1, threads / static_cast<int>(MMseqsNode::count));
    std::vector<char *> ownRecords(indexLines, NULL);
    std::vector<size_t> ownRecordBytes(indexLines, 0);
#pragma omp parallel for schedule(dynamic, 1)
    for (int indexLine = 0; indexLine < indexLines; indexLine++) {
        size_t indexLineBegin = std::max(ownBegin, threadOffsets[threads * indexLine / indexLines]);
        size_t indexLineEnd = std::min(ownEnd, threadOffsets[threads * (indexLine + 1) / indexLines]);
        if (indexLineBegin < indexLineEnd) {
            ownRecordBytes[indexLine] = indexLineEnd - indexLineBegin;
            ownRecords[indexLine] = static_cast<char *>(malloc(ownRecordBytes[indexLine]));
            Util::checkAllocation(ownRecords[indexLine], "Can not allocate own record memory in createtsv");
            memcpy(ownRecords[indexLine], writeBuffer + indexLineBegin, ownRecordBytes[indexLine]);
        }
    }
    for (int indexLine = 0; indexLine < indexLines; indexLine++) {
        if (ownRecordBytes[indexLine] > 0) {
            MMseqsNode::keepOwnExchangeRecords(ownExchange, ownRecords[indexLine], ownRecordBytes[indexLine]);
        }
    }

    Util::releasePages(writeBuffer, writeBufferBytes);
    free(writeBuffer);
    std::vector<size_t> threadNodeOffsets(MMseqsNode::count + 1);
    for (int indexLine = 0; indexLine < indexLines; indexLine++) {
        size_t indexLineBegin = getFileOffset(threadOffsets[threads * indexLine / indexLines], ownBegin, ownEnd);
        size_t indexLineEnd = getFileOffset(threadOffsets[threads * (indexLine + 1) / indexLines], ownBegin, ownEnd);
        for (size_t node = 0; node <= MMseqsNode::count; node++) {
            threadNodeOffsets[node] = std::min(std::max(getFileOffset(nodeOffsets[node], ownBegin, ownEnd), indexLineBegin), indexLineEnd) - indexLineBegin;
        }
        MMseqsNode::addExchangeFlush(indexData, writtenBytes, threadNodeOffsets);
    }
    return ownEnd - ownBegin;
}

static std::string readMemberAccessions(const std::string &sortedByMemberIndex, const std::string &sortedByMember, DBReader<DBKeyType> &headers, const std::vector<size_t> &nodeKeyOffsets,
                                        const std::vector<size_t> &tsvKeyOffsets, size_t memoryLimit, const std::string &sortedByRep) {
    std::vector<MMseqsNode::ExchangeReader> memberReaders;
    std::vector<FILE *> memberFiles;
    MMseqsNode::openExchange(sortedByMemberIndex, sortedByMember, memberReaders, memberFiles);

    size_t localFirstKey = nodeKeyOffsets[MMseqsNode::rank];
    size_t localEndKey = nodeKeyOffsets[MMseqsNode::rank + 1];
    // name, buffer offset and split offset
    size_t keyBytes = headers.getNamesBytes(localFirstKey, localEndKey) / std::max((size_t) 1, localEndKey - localFirstKey) + sizeof(uint64_t) + sizeof(size_t);
    size_t splitKeys = std::max((size_t) 1, memoryLimit / 8 / keyBytes);
    // halve splits whose names do not fit
    std::vector<size_t> splitKeyOffsets(1, localFirstKey);
    while (splitKeyOffsets.back() < localEndKey) {
        size_t keyBegin = splitKeyOffsets.back();
        size_t keyEnd = keyBegin + std::min(splitKeys, localEndKey - keyBegin);
        while (keyEnd - keyBegin > 1 && headers.getNamesBytes(keyBegin, keyEnd) + (keyEnd - keyBegin) * (sizeof(uint64_t) + sizeof(size_t)) > memoryLimit / 8) {
            keyEnd = keyBegin + (keyEnd - keyBegin) / 2;
        }
        splitKeyOffsets.push_back(keyEnd);
    }

    std::string repFile = sortedByRep + "." + SSTR(MMseqsNode::rank);
    FILE *repHandle = FileUtil::openAndDelete(repFile.c_str(), "w");
    std::string repIndexData;
    size_t writtenBytes = 0;
    MemberAccession *memberAccessions = NULL;
    size_t accessionCount = 0;
    size_t memberAccessionCapacity = 0;
    char *accessions = NULL;
    size_t accessionsBytes = 0;
    size_t accessionsCapacity = 0;
    int threads = Parameters::getInstance().threads;
    std::vector<std::string> threadAccessions(threads);
    std::vector<size_t> threadAccessionOffsets(threads + 1);
    std::vector<char> receivedPairs;
    // keep own records in memory, up to a quarter of it
    size_t keptOwnBytes = 0;
    for (size_t split = 0; split + 1 < splitKeyOffsets.size(); split++) {
        size_t keyBegin = splitKeyOffsets[split];
        size_t keyEnd = splitKeyOffsets[split + 1];
        headers.loadData(keyBegin, keyEnd);
        if (split + 2 < splitKeyOffsets.size()) {
            headers.prefetchData(keyEnd, splitKeyOffsets[split + 2]);
        }
        ClusterPair splitLastPair = { static_cast<DBKeyType>(keyEnd - 1), static_cast<DBKeyType>(keyEnd - 1) };
        size_t pairCount;
        while ((pairCount = MMseqsNode::readExchangeRecordsUpTo(memberReaders, memoryLimit / 64, NULL, ClusterPair::compareByMember, splitLastPair,
                                                                receivedPairs, NULL, sortedByMember)) > 0) {
            ClusterPair *pairs = reinterpret_cast<ClusterPair *>(receivedPairs.data());
            SORT_PARALLEL(pairs, pairs + pairCount, ClusterPair::compareByMember);
            size_t firstAccession = accessionCount;
            accessionCount += pairCount;
            growBuffer(memberAccessions, memberAccessionCapacity, accessionCount);
#pragma omp parallel
            {
                int thread_idx = 0;
#ifdef OPENMP
                thread_idx = omp_get_thread_num();
#endif
#pragma omp for schedule(dynamic, 1)
                for (int thread = 0; thread < threads; thread++) {
                    threadAccessions[thread].clear();
                    for (size_t i = pairCount * thread / threads; i < pairCount * (thread + 1) / threads; i++) {
                        DBKeyType memberKey = pairs[i].memberKey;
                        std::string accession = Util::parseFastaHeader(headers.getData(memberKey, thread_idx));
                        MemberAccession memberAccession = { pairs[i].repKey, memberKey, threadAccessions[thread].size(), static_cast<unsigned int>(accession.size()) };
                        memberAccessions[firstAccession + i] = memberAccession;
                        threadAccessions[thread].append(accession);
                    }
                }
            }
            threadAccessionOffsets[0] = accessionsBytes;
            for (int thread = 0; thread < threads; thread++) {
                threadAccessionOffsets[thread + 1] = threadAccessionOffsets[thread] + threadAccessions[thread].size();
            }
            accessionsBytes = threadAccessionOffsets[threads];
            growBuffer(accessions, accessionsCapacity, accessionsBytes);
#pragma omp parallel for schedule(dynamic, 1)
            for (int thread = 0; thread < threads; thread++) {
                memcpy(accessions + threadAccessionOffsets[thread], threadAccessions[thread].data(), threadAccessions[thread].size());
                for (size_t i = pairCount * thread / threads; i < pairCount * (thread + 1) / threads; i++) {
                    memberAccessions[firstAccession + i].accessionOffset += threadAccessionOffsets[thread];
                }
            }
            if (accessionsBytes + accessionCount * sizeof(MemberAccession) >= memoryLimit / 8) {
                keptOwnBytes += writeMemberAccessions(memberAccessions, accessionCount, accessions, tsvKeyOffsets, repHandle, repFile, repIndexData, writtenBytes,
                                                      sortedByRep, memoryLimit / 4 - keptOwnBytes);
                accessionCount = 0;
                accessionsBytes = 0;
            }
        }
    }

    ClusterPair lastPair = { DB_KEY_INVALID, DB_KEY_INVALID };
    if (MMseqsNode::readExchangeRecordsUpTo(memberReaders, memoryLimit / 64, NULL, ClusterPair::compareByMember, lastPair, receivedPairs, NULL, sortedByMember) > 0) {
        Debug(Debug::ERROR) << "Key " << reinterpret_cast<ClusterPair *>(receivedPairs.data())->memberKey << " of the clustering is not in the sequence database\n";
        EXIT(EXIT_FAILURE);
    }
    writeMemberAccessions(memberAccessions, accessionCount, accessions, tsvKeyOffsets, repHandle, repFile, repIndexData, writtenBytes,
                          sortedByRep, SIZE_MAX);
    Util::releasePages(memberAccessions, memberAccessionCapacity * sizeof(MemberAccession));
    free(memberAccessions);
    Util::releasePages(accessions, accessionsCapacity);
    free(accessions);
    MMseqsNode::closeExchangeFiles(memberFiles, sortedByMember);
    return MMseqsNode::closeExchange(repHandle, repFile, sortedByRep + ".index", repIndexData);
}

static void writeClusterTsvLines(const std::string &sortedByRepIndex, const std::string &sortedByRep, const std::string &tsv, const Parameters &par) {
    std::vector<MMseqsNode::ExchangeReader> repReaders;
    std::vector<FILE *> repFiles;
    MMseqsNode::openExchange(sortedByRepIndex, sortedByRep, repReaders, repFiles);

    size_t memoryLimit = Util::computeMemory(par.splitMemoryLimit);
    std::string tsvFile = tsv + "." + SSTR(MMseqsNode::rank);
    FILE *tsvHandle = FileUtil::openAndDelete(tsvFile.c_str(), "w");
    size_t tsvBytes = 0;
    ClusterPair lastPair = { DB_KEY_INVALID, DB_KEY_INVALID };
    std::vector<char> records;
    std::vector<size_t> recordOffsets;
    MemberAccession *memberAccessions = NULL;
    size_t memberAccessionCapacity = 0;
    std::vector<std::string> threadLines(par.threads);
    std::vector<size_t> threadLineOffsets(par.threads + 1);
    size_t writeBufferBytes = std::min(MAX_READ_REQUEST_BYTES, std::max((size_t) 1, memoryLimit / 8 / par.threads));

    // representative of the cluster the previous range ended in
    std::string repAccession;
    DBKeyType previousRepKey = DB_KEY_INVALID;
    size_t recordCount;
    while ((recordCount = MMseqsNode::readExchangeRecordsUpTo(repReaders, memoryLimit / 8, getMemberAccessionBytes, ClusterPair::compareByRep, lastPair,
                                                              records, &recordOffsets, sortedByRep)) > 0) {
        growBuffer(memberAccessions, memberAccessionCapacity, recordCount);
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < recordCount; i++) {
            const char *record = records.data() + recordOffsets[i];
            MemberAccession &memberAccession = memberAccessions[i];
            memcpy(&memberAccession.repKey, record, sizeof(DBKeyType));
            memcpy(&memberAccession.memberKey, record + sizeof(DBKeyType), sizeof(DBKeyType));
            memcpy(&memberAccession.accessionLength, record + 2 * sizeof(DBKeyType), sizeof(unsigned int));
            memberAccession.accessionOffset = recordOffsets[i] + MEMBER_ACCESSION_HEADER_BYTES;
        }
        SORT_PARALLEL(memberAccessions, memberAccessions + recordCount, MemberAccession::compareByRep);
        // count bytes first, long representative names grow the TSV
        for (int pass = 0; pass < 2; pass++) {
#pragma omp parallel for schedule(dynamic, 1)
            for (int thread = 0; thread < par.threads; thread++) {
                std::string &lines = threadLines[thread];
                lines.clear();
                size_t begin = recordCount * thread / par.threads;
                size_t end = recordCount * (thread + 1) / par.threads;
                DBKeyType repKey = (begin > 0) ? memberAccessions[begin - 1].repKey : previousRepKey;
                const char *repName = repAccession.data();
                size_t repLength = repAccession.size();
                // find the representative line of a continued cluster
                MemberAccession repLine = { repKey, repKey, 0, 0 };
                size_t repLinePos = std::lower_bound(memberAccessions, memberAccessions + begin, repLine, MemberAccession::compareByRep) - memberAccessions;
                if (repLinePos < begin && memberAccessions[repLinePos].memberKey == repKey) {
                    repName = records.data() + memberAccessions[repLinePos].accessionOffset;
                    repLength = memberAccessions[repLinePos].accessionLength;
                }
                size_t writeOffset = (pass == 0) ? 0 : threadLineOffsets[thread];
                for (size_t i = begin; i < end; i++) {
                    const MemberAccession &memberAccession = memberAccessions[i];
                    if (memberAccession.repKey != repKey) {
                        repKey = memberAccession.repKey;
                        repName = records.data() + memberAccession.accessionOffset;
                        repLength = memberAccession.accessionLength;
                    }
                    if (pass == 0) {
                        writeOffset += repLength + memberAccession.accessionLength + 2;
                        continue;
                    }
                    lines.append(repName, repLength);
                    lines.push_back('\t');
                    lines.append(records.data() + memberAccession.accessionOffset, memberAccession.accessionLength);
                    lines.push_back('\n');
                    if (lines.size() >= writeBufferBytes) {
                        FileUtil::pwriteAll(fileno(tsvHandle), tsvFile.c_str(), lines.data(), lines.size(), writeOffset);
                        writeOffset += lines.size();
                        lines.clear();
                    }
                }
                if (pass == 0) {
                    threadLineOffsets[thread + 1] = writeOffset;
                } else {
                    FileUtil::pwriteAll(fileno(tsvHandle), tsvFile.c_str(), lines.data(), lines.size(), writeOffset);
                }
            }
            if (pass == 0) {
                threadLineOffsets[0] = tsvBytes;
                for (int thread = 0; thread < par.threads; thread++) {
                    threadLineOffsets[thread + 1] += threadLineOffsets[thread];
                }
            }
        }
        tsvBytes = threadLineOffsets[par.threads];
        previousRepKey = memberAccessions[recordCount - 1].repKey;
        MemberAccession repLine = { previousRepKey, previousRepKey, 0, 0 };
        size_t repLinePos = std::lower_bound(memberAccessions, memberAccessions + recordCount, repLine, MemberAccession::compareByRep) - memberAccessions;
        if (memberAccessions[repLinePos].memberKey == previousRepKey) {
            repAccession.assign(records.data() + memberAccessions[repLinePos].accessionOffset, memberAccessions[repLinePos].accessionLength);
        }
    }

    Util::releasePages(memberAccessions, memberAccessionCapacity * sizeof(MemberAccession));
    free(memberAccessions);
    if (fclose(tsvHandle) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << tsvFile << "\n";
        EXIT(EXIT_FAILURE);
    }
    MMseqsNode::closeExchangeFiles(repFiles, sortedByRep);
}

int createtsv(int argc, const char **argv, const Command &command) {
    MMseqsNode::init();
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, Parameters::PARSE_VARIADIC, 0);
    const bool isLinfinity = DBReader<DBKeyType>::getExtendedDbtype(FileUtil::parseDbType(par.db1.c_str())) & Parameters::DBTYPE_EXTENDED_LINFINITY;
    if (isLinfinity) {
        int clusterDbtype = FileUtil::parseDbType(par.db3.c_str());
        if (Parameters::isEqualDbtype(clusterDbtype, Parameters::DBTYPE_CLUSTER_RES) == false
            || (DBReader<DBKeyType>::getExtendedDbtype(clusterDbtype) & Parameters::DBTYPE_EXTENDED_LINFINITY) == 0) {
            Debug(Debug::ERROR) << par.db3 << " is not a linfinity clustering (--createdb-mode " << Parameters::SEQUENCE_SPLIT_MODE_LINFINITY_LENGTH << ")\n";
            EXIT(EXIT_FAILURE);
        }

        DBReader<DBKeyType> headers(par.hdr1.c_str(), (par.db1 + ".file_keys").c_str(), par.threads, DBReader<DBKeyType>::USE_INDEX|DBReader<DBKeyType>::USE_DATA|DBReader<DBKeyType>::USE_LINFINITY);
        headers.open(DBReader<DBKeyType>::NOSORT);
        std::vector<size_t> nodeKeyOffsets = headers.getEvenNodeKeyOffsets(MMseqsNode::count);
        std::string phaseValues;
        Timer timer;

        std::string sortedByMember = par.db4 + "_by_member";
        std::string sortedByMemberIndex = MMseqsNode::sendClusterPairs(par.db3, true, ClusterPair::compareByMember, nodeKeyOffsets, sortedByMember);
        MMseqsNode::addPhaseTime(phaseValues, "sending cluster pairs", timer);

        std::string sortedByRep = par.db4 + "_by_rep";
        std::vector<size_t> tsvKeyOffsets = MMseqsNode::getBalancedClusterKeyOffsets(par.db4 + ".cluster_key_offsets", par.db3, headers.getSize());
        std::string sortedByRepIndex = readMemberAccessions(sortedByMemberIndex, sortedByMember, headers, nodeKeyOffsets, tsvKeyOffsets,
                                                            Util::computeMemory(par.splitMemoryLimit), sortedByRep);
        headers.close();
        MMseqsNode::addPhaseTime(phaseValues, "sending member accessions", timer);
        FileUtil::remove((sortedByMember + "." + SSTR(MMseqsNode::rank)).c_str());

        writeClusterTsvLines(sortedByRepIndex, sortedByRep, par.db4, par);
        MMseqsNode::addPhaseTime(phaseValues, "writing TSV", timer);

        std::vector<std::string> exchanges;
        exchanges.push_back(sortedByMember);
        exchanges.push_back(sortedByRep);
        MMseqsNode::removeExchanges(sortedByRep, sortedByRep, exchanges, phaseValues);
        return EXIT_SUCCESS;
    }

    bool queryNucs = Parameters::isEqualDbtype(FileUtil::parseDbType(par.db1.c_str()), Parameters::DBTYPE_NUCLEOTIDES);
    bool targetNucs = Parameters::isEqualDbtype(FileUtil::parseDbType(par.db2.c_str()), Parameters::DBTYPE_NUCLEOTIDES);
    const bool touch = (par.preloadMode != Parameters::PRELOAD_MODE_MMAP);
    int queryHeaderType = (queryNucs) ? IndexReader::SRC_HEADERS : IndexReader::HEADERS;
    queryHeaderType = (par.idxSeqSrc == 0) ? queryHeaderType :  (par.idxSeqSrc == 1) ?  IndexReader::HEADERS : IndexReader::SRC_HEADERS;
    IndexReader qDbrHeader(par.db1, par.threads, queryHeaderType, (touch) ? (IndexReader::PRELOAD_INDEX | IndexReader::PRELOAD_DATA) : 0);
    IndexReader * tDbrHeader=NULL;
    DBReader<DBKeyType> * queryDB = qDbrHeader.sequenceReader;
    DBReader<DBKeyType> * targetDB = NULL;
    bool sameDB = (par.db2.compare(par.db1) == 0);
    const bool hasTargetDB = par.filenames.size() > 3;
    DBReader<DBKeyType>::Index * qHeaderIndex = qDbrHeader.sequenceReader->getIndex();
    DBReader<DBKeyType>::Index * tHeaderIndex = NULL;

    if (hasTargetDB) {
        if (sameDB) {
            tDbrHeader = &qDbrHeader;
            tHeaderIndex = qHeaderIndex;
            targetDB = queryDB;
        } else {

            int targetHeaderType = (targetNucs) ? IndexReader::SRC_HEADERS : IndexReader::HEADERS;
            targetHeaderType = (par.idxSeqSrc == 0) ? targetHeaderType :  (par.idxSeqSrc == 1) ?  IndexReader::HEADERS : IndexReader::SRC_HEADERS;

            tDbrHeader = new IndexReader(par.db2, par.threads, targetHeaderType, touch);
            tHeaderIndex = tDbrHeader->sequenceReader->getIndex();
            targetDB = tDbrHeader->sequenceReader;
        }
    }

    DBReader<DBKeyType> *reader;
    if (hasTargetDB) {

        reader = new DBReader<DBKeyType>(par.db3.c_str(), par.db3Index.c_str(), par.threads, DBReader<DBKeyType>::USE_INDEX|DBReader<DBKeyType>::USE_DATA);
    } else {

        reader = new DBReader<DBKeyType>(par.db2.c_str(), par.db2Index.c_str(), par.threads, DBReader<DBKeyType>::USE_INDEX|DBReader<DBKeyType>::USE_DATA);
    }
    reader->open(DBReader<DBKeyType>::LINEAR_ACCCESS);

    uint16_t extended = DBReader<DBKeyType>::getExtendedDbtype(reader->getDbtype());
    bool needSET = false;
    std::map<DBKeyType, std::string> qSetToSource, tSetToSource;
    if (extended & Parameters::DBTYPE_EXTENDED_SET) {
        needSET = true;
        if (hasTargetDB) {
            qSetToSource = Util::readLookup((par.db1 + ".source"), 2);
            tSetToSource = Util::readLookup((par.db2 + ".source"), 2);
        } else {
            qSetToSource = Util::readLookup((par.db1 + ".source"), 2);
        }
    }

    const std::string& dataFile = hasTargetDB ? par.db4 : par.db3;
    const std::string& indexFile = hasTargetDB ? par.db4Index : par.db3Index;
    const bool shouldCompress = par.dbOut == true && par.compressed == true;
    const int dbType = par.dbOut == true ? Parameters::DBTYPE_GENERIC_DB : Parameters::DBTYPE_OMIT_FILE;
    DBWriter writer(dataFile.c_str(), indexFile.c_str(), par.threads, shouldCompress, dbType);
    writer.open();

    const size_t targetColumn = (par.targetTsvColumn == 0) ? SIZE_T_MAX :  par.targetTsvColumn - 1;
#pragma omp parallel
    {
        unsigned int thread_idx = 0;
#ifdef OPENMP
        thread_idx = (unsigned int) omp_get_thread_num();
#endif

        const char *columnPointer[255];
        char *dbKey = new char[par.maxSeqLen + 1];

        std::string outputBuffer;
        outputBuffer.reserve(10 * 1024);

#pragma omp for schedule(dynamic, 1000)
        for (size_t i = 0; i < reader->getSize(); ++i) {
            DBKeyType queryKey = reader->getDbKey(i);
            size_t queryIndex;
            char *headerData;
            if(needSET == false) {
                queryIndex = queryDB->getId(queryKey);
                headerData = queryDB->getData(queryIndex, thread_idx);
                if (headerData == NULL) {
                    Debug(Debug::WARNING) << "Invalid header entry in query " << queryKey << "!\n";
                    continue;
                }
            }

            std::string queryHeader;
            if (needSET == true) {
                queryHeader = qSetToSource[queryKey];
            } else if (par.fullHeader) {
                queryHeader = "\"";
                queryHeader.append(headerData, qHeaderIndex[queryIndex].length - 2);
                queryHeader.append("\"");
            } else {
                queryHeader = Util::parseFastaHeader(headerData);
            }

            size_t entryIndex = 0;

            char *data = reader->getData(i, thread_idx);
            while (*data != '\0') {
                if(targetColumn != SIZE_T_MAX){
                    size_t foundElements = Util::getWordsOfLine(data, columnPointer, 255);
                    if (foundElements < targetColumn) {
                        Debug(Debug::WARNING) << "Not enough columns!" << "\n";
                        continue;
                    }
                    Util::parseKey(columnPointer[targetColumn], dbKey);
                }
                std::string targetAccession;
                if(targetColumn == SIZE_T_MAX){
                    targetAccession = "";
                } else if (hasTargetDB) {
                    DBKeyType targetKey = Util::fast_atoi<DBKeyType>(dbKey);
                    size_t targetIndex = targetDB->getId(targetKey);
                    char *targetData;
                    if(needSET == false) {
                        targetData = targetDB->getData(targetIndex, thread_idx);
                        if (targetData == NULL) {
                            Debug(Debug::WARNING) << "Invalid header entry in query " << queryKey << " and target " << targetKey << "!\n";
                            continue;
                        }
                        if (par.fullHeader) {
                            targetAccession = "\"";
                            targetAccession.append(targetData, tHeaderIndex[targetIndex].length - 2);
                            targetAccession.append("\"");
                        } else {
                            targetAccession = Util::parseFastaHeader(targetData);
                        }
                    } else {
                        targetAccession = tSetToSource[targetKey];
                    }
                } else {
                    targetAccession = dbKey;
                }

                if (par.firstSeqRepr && !entryIndex) {
                    queryHeader = targetAccession;
                }

                outputBuffer.append(queryHeader);
                outputBuffer.append("\t");
                outputBuffer.append(targetAccession);

                size_t offset = 0;
                if (targetColumn != 0) {
                    outputBuffer.append("\t");
                    offset = 0;
                } else {
                    offset = strlen(dbKey);
                }

                char *nextLine = Util::skipLine(data);
                outputBuffer.append(data + offset, (nextLine - (data + offset)) - 1);
                outputBuffer.append("\n");
                data = nextLine;
                entryIndex++;
            }
            writer.writeData(outputBuffer.c_str(), outputBuffer.length(), queryKey, thread_idx, par.dbOut);
            outputBuffer.clear();
        }
        delete[] dbKey;
    }
    writer.close(par.dbOut == false);

    if (par.dbOut == false) {
        if (hasTargetDB) {
            FileUtil::remove(par.db4Index.c_str());
        } else {
            FileUtil::remove(par.db3Index.c_str());
        }
    }

    reader->close();
    delete reader;
    if (hasTargetDB) {
        if (sameDB == false) {
            delete tDbrHeader;
        }
    }
    qSetToSource.clear();
    tSetToSource.clear();

    return EXIT_SUCCESS;
}
#undef SIZE_T_MAX
