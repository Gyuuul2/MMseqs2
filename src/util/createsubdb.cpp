#include "Parameters.h"
#include "FileUtil.h"
#include "DBReader.h"
#include "DBWriter.h"
#include "Debug.h"
#include "Util.h"
#include "MMseqsNode.h"
#include "FastSort.h"

#include <climits>

#ifdef OPENMP
#include <omp.h>
#endif

int createsubdb(int argc, const char **argv, const Command& command) {
    MMseqsNode::init();
    Parameters& par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);

    const bool isLinfinity = DBReader<DBKeyType>::getExtendedDbtype(FileUtil::parseDbType(par.db2.c_str())) & Parameters::DBTYPE_EXTENDED_LINFINITY;
    if (isLinfinity) {
        // entry keys become representative bits and seed member requests
        DBReader<DBKeyType> reader(par.db2.c_str(), par.db2Index.c_str(), 1, DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_LINFINITY);
        reader.open(DBReader<DBKeyType>::NOSORT);
        std::vector<size_t> fileKeyOffsets = reader.getFileKeyOffsets(MMseqsNode::count);
        std::vector<size_t> cluKeyOffsets = MMseqsNode::getClusterKeyOffsets(par.db3 + ".cluster_key_offsets", par.db1, reader.getSize());
        size_t firstKey = fileKeyOffsets[MMseqsNode::rank];
        size_t keyCount = fileKeyOffsets[MMseqsNode::rank + 1] - firstKey;
        std::vector<uint64_t> bits((keyCount + 63) / 64, 0);
        // (rep, seed member) pairs by member, a column per data file
        std::string requestFile = MMseqsNode::getSeedMemberRequestsFile(par.db3, MMseqsNode::rank);
        FILE *requestHandle = FileUtil::openAndDelete(requestFile.c_str(), "w");
        std::vector<size_t> requestOffsets;
        size_t requestSize = 0;
        std::vector<size_t> rangeSeedCounts(reader.getIndexSize(), 0);
        std::vector<ClusterPair> seedMembers;
        const size_t membersPerCluster = static_cast<size_t>(par.seedMemberPerRep);
        size_t splitSize = Util::computeMemory(par.splitMemoryLimit) / 16;
        size_t splitEntries = std::max((size_t) 1, splitSize / (2 * membersPerCluster * sizeof(ClusterPair)));

        for (size_t node = 0; node < MMseqsNode::count; node++) {
            size_t from = std::max(firstKey, cluKeyOffsets[node]);
            size_t to = std::min(firstKey + keyCount, cluKeyOffsets[node + 1]);
            if (from >= to) {
                continue;
            }
            DBReader<DBKeyType> cluDb((par.db1 + "." + SSTR(node)).c_str(), (par.db1 + ".index." + SSTR(node)).c_str(), par.threads,
                                      DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
            cluDb.open(DBReader<DBKeyType>::NOSORT);
            size_t entryEnd = cluDb.findEntryByKey(to);
            size_t splitBegin = cluDb.findEntryByKey(from);
            // a larger entry is read alone
            size_t splitEnd = (splitBegin < entryEnd) ? std::max(splitBegin + 1, MMseqsNode::findSplitEnd(cluDb, splitBegin, cluDb.getIndexEntry(splitBegin).offset, splitSize, splitEntries, entryEnd)) : entryEnd;
            for (size_t nextSplitEnd = splitEnd; splitBegin < entryEnd; splitBegin = splitEnd, splitEnd = nextSplitEnd) {
                cluDb.loadData(splitBegin, splitEnd);
                if (splitEnd < entryEnd) {
                    nextSplitEnd = std::max(splitEnd + 1, MMseqsNode::findSplitEnd(cluDb, splitEnd, cluDb.getIndexEntry(splitEnd).offset, splitSize, splitEntries, entryEnd));
                    cluDb.prefetchData(splitEnd, nextSplitEnd);
                }
                seedMembers.resize((splitEnd - splitBegin) * membersPerCluster);
#pragma omp parallel num_threads(par.threads)
                {
                    unsigned int thread_idx = 0;
#ifdef OPENMP
                    thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
                    char keyBuffer[255];
                    std::vector<DBKeyType> members;
#pragma omp for schedule(dynamic, 100)
                    for (size_t i = splitBegin; i < splitEnd; i++) {
                        DBKeyType repKey = cluDb.getDbKey(i);
                        members.clear();
                        for (char *data = cluDb.getData(i, thread_idx); *data != '\0'; data = Util::skipLine(data)) {
                            Util::parseKey(data, keyBuffer);
                            DBKeyType key = Util::fast_atoi<DBKeyType>(keyBuffer);
                            if (key != repKey) {
                                members.push_back(key);
                            }
                        }
                        // key order is length order
                        size_t memberCount = std::min(members.size(), membersPerCluster);
                        std::vector<DBKeyType>::iterator unranked = members.begin();
                        for (size_t m = 0; m < membersPerCluster; m++) {
                            ClusterPair &seedMember = seedMembers[(i - splitBegin) * membersPerCluster + m];
                            seedMember.repKey = repKey;
                            seedMember.memberKey = DB_KEY_INVALID;
                            if (m < memberCount) {
                                std::vector<DBKeyType>::iterator rank = members.begin() + (m + 1) * members.size() / (memberCount + 1);
                                std::nth_element(unranked, rank, members.end());
                                seedMember.memberKey = *rank;
                                unranked = rank + 1;
                            }
                        }
                    }
                }

                size_t seedMemberCount = 0;
                for (size_t i = 0; i < seedMembers.size(); i++) {
                    DBKeyType repKey = seedMembers[i].repKey;
                    bits[(repKey - firstKey) / 64] |= 1ULL << ((repKey - firstKey) % 64);
                    if (seedMembers[i].memberKey != DB_KEY_INVALID) {
                        rangeSeedCounts[reader.findIdRange(repKey)]++;
                        seedMembers[seedMemberCount++] = seedMembers[i];
                    }
                }
                seedMembers.resize(seedMemberCount);
                SORT_PARALLEL(seedMembers.begin(), seedMembers.end(), ClusterPair::compareByMember);
                for (size_t file = 0; file <= MMseqsNode::count; file++) {
                    ClusterPair firstPair = { static_cast<DBKeyType>(fileKeyOffsets[file]), static_cast<DBKeyType>(fileKeyOffsets[file]) };
                    requestOffsets.push_back(requestSize + (std::lower_bound(seedMembers.begin(), seedMembers.end(), firstPair, ClusterPair::compareByMember) - seedMembers.begin()) * sizeof(ClusterPair));
                }
                if (fwrite(seedMembers.data(), sizeof(ClusterPair), seedMemberCount, requestHandle) != seedMemberCount) {
                    Debug(Debug::ERROR) << "Cannot write " << requestFile << "\n";
                    EXIT(EXIT_FAILURE);
                }
                requestSize += seedMemberCount * sizeof(ClusterPair);
            }
            cluDb.close();
        }

        MMseqsNode::closeExchangeTableFile(requestHandle, requestFile, requestOffsets, MMseqsNode::count + 1, requestSize);
        MMseqsNode::writeRepresentativeBits(par.db3, reader, bits.data(), firstKey, keyCount, rangeSeedCounts);
        MMseqsNode::gatherRepresentativeIndex(par.db3);

        if (MMseqsNode::isMaster()) {
            // the next steps read no headers, lookup or source
            DBReader<DBKeyType>::softlinkDb(par.db2, par.db3, DBFiles::DATA);
            DBWriter::writeDbtypeFile(par.db3.c_str(), reader.getDbtype(), reader.isCompressed());
        }
        reader.close();
        return EXIT_SUCCESS;
    }

    FILE *orderFile = NULL;
    if (FileUtil::fileExists(par.db1Index.c_str())) {
        orderFile = fopen(par.db1Index.c_str(), "r");
    } else {
        if(FileUtil::fileExists(par.db1.c_str())){
            orderFile = fopen(par.db1.c_str(), "r");
        }else{
            Debug(Debug::ERROR) << "File " << par.db1 << " does not exist.\n";
            EXIT(EXIT_FAILURE);
        }
    }

    const bool lookupMode = par.dbIdMode == Parameters::ID_MODE_LOOKUP;
    int dbMode = DBReader<DBKeyType>::USE_INDEX|DBReader<DBKeyType>::USE_DATA;
    if (lookupMode) {
        dbMode |= DBReader<DBKeyType>::USE_LOOKUP_REV;
    }
    DBReader<DBKeyType> reader(par.db2.c_str(), par.db2Index.c_str(), 1, dbMode);
    reader.open(DBReader<DBKeyType>::NOSORT);
    const bool isCompressed = reader.isCompressed();

    DBWriter writer(par.db3.c_str(), par.db3Index.c_str(), 1, 0, Parameters::DBTYPE_OMIT_FILE);
    writer.open();
    // getline reallocs automatic
    char *line = NULL;
    size_t len = 0;
    char dbKey[256];
    DBKeyType prevKey = 0;
    bool isOrdered = true;
    while (getline(&line, &len, orderFile) != -1) {
        Util::parseKey(line, dbKey);
        DBKeyType key;
        if (lookupMode) {
            size_t lookupId = reader.getLookupIdByAccession(dbKey);
            if (lookupId == SIZE_MAX) {
                Debug(Debug::WARNING) << "Could not find name " << dbKey << " in lookup\n";
                continue;
            }
            key = reader.getLookupKey(lookupId);
        } else {
            key = Util::fast_atoi<DBKeyType>(dbKey);
        }

        isOrdered &= (prevKey <= key);
        prevKey = key;
        const size_t id = reader.getId(key);
        if (id == SIZE_MAX) {
            Debug(Debug::WARNING) << "Key " << dbKey << " not found in database\n";
            continue;
        }
        if (par.subDbMode == Parameters::SUBDB_MODE_SOFT) {
            writer.writeIndexEntry(key, reader.getOffset(id), reader.getEntryLen(id), 0);
        } else {
            char* data = reader.getDataUncompressed(id);
            size_t originalLength = reader.getEntryLen(id);
            size_t entryLength = std::max(originalLength, static_cast<size_t>(1)) - 1;

            if (isCompressed) {
                // copy also the null byte since it contains the information if compressed or not
                entryLength = *(reinterpret_cast<unsigned int *>(data)) + sizeof(unsigned int) + 1;
                writer.writeData(data, entryLength, key, 0, false, false);
            } else {
                writer.writeData(data, entryLength, key, 0, true, false);
            }
            // do not write null byte since
            writer.writeIndexEntry(key, writer.getStart(0), originalLength, 0);
        }
    }
    // merge any kind of sequence database
    const bool shouldMerge = Parameters::isEqualDbtype(reader.getDbtype(), Parameters::DBTYPE_HMM_PROFILE)
                             || Parameters::isEqualDbtype(reader.getDbtype(), Parameters::DBTYPE_AMINO_ACIDS)
                             || Parameters::isEqualDbtype(reader.getDbtype(), Parameters::DBTYPE_NUCLEOTIDES);
    writer.close(shouldMerge, !isOrdered);
    if (par.subDbMode == Parameters::SUBDB_MODE_SOFT) {
        DBReader<DBKeyType>::softlinkDb(par.db2, par.db3, DBFiles::DATA);
    }
    DBWriter::writeDbtypeFile(par.db3.c_str(), reader.getDbtype(), isCompressed);
    DBReader<DBKeyType>::softlinkDb(par.db2, par.db3, DBFiles::SEQUENCE_ANCILLARY);

    free(line);
    reader.close();
    if (fclose(orderFile) != 0) {
        Debug(Debug::ERROR) << "Cannot close file " << par.db1 << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
