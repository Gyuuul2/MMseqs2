#include "Parameters.h"
#include "FileUtil.h"
#include "DBReader.h"
#include "DBWriter.h"
#include "Debug.h"
#include "Util.h"
#include "MMseqsNode.h"

#include <climits>

int createsubdb(int argc, const char **argv, const Command& command) {
    MMseqsNode::init();
    Parameters& par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);

    // linfinity: entry keys become representative bits
    const bool isLinfinity = DBReader<DBKeyType>::getExtendedDbtype(FileUtil::parseDbType(par.db2.c_str())) & Parameters::DBTYPE_EXTENDED_LINFINITY;
    if (isLinfinity) {
        DBReader<DBKeyType> reader(par.db2.c_str(), par.db2Index.c_str(), 1, DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_LINFINITY);
        reader.open(DBReader<DBKeyType>::NOSORT);
        std::vector<size_t> fileKeyOffsets = reader.getFileKeyOffsets(MMseqsNode::count);
        std::vector<size_t> cluKeyOffsets = MMseqsNode::getClusterKeyOffsets(par.db3 + ".cluster_key_offsets", par.db1, reader.getSize());
        size_t firstKey = fileKeyOffsets[MMseqsNode::rank];
        size_t keyCount = fileKeyOffsets[MMseqsNode::rank + 1] - firstKey;
        std::vector<uint64_t> bits((keyCount + 63) / 64, 0);
        std::vector<DBKeyType> repKeys;
        size_t splitKeys = std::max((size_t) 1, Util::computeMemory(par.splitMemoryLimit) / 16 / sizeof(DBKeyType));
        for (size_t splitBegin = firstKey; splitBegin < firstKey + keyCount; splitBegin += splitKeys) {
            MMseqsNode::readClusterKeys(par.db1, cluKeyOffsets, splitBegin, std::min(firstKey + keyCount, splitBegin + splitKeys), repKeys);
            for (size_t i = 0; i < repKeys.size(); i++) {
                bits[(repKeys[i] - firstKey) / 64] |= 1ULL << ((repKeys[i] - firstKey) % 64);
            }
        }
        MMseqsNode::writeRepresentativeBits(par.db3, reader, bits.data(), firstKey, keyCount);

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
