#include "Parameters.h"
#include "FileUtil.h"
#include "DBReader.h"
#include "DBWriter.h"
#include "Debug.h"
#include "Util.h"
#include "MMseqsNode.h"

#include <climits>

int createsubdb(int argc, const char **argv, const Command& command) {
    Parameters& par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);

    // --createdb-mode 3: the entry keys of the cluster database <subset>.<node> become the representative bits of the data file of a node and the sixth index column
    const bool isLinfinity = DBReader<DBKeyType>::getExtendedDbtype(FileUtil::parseDbType(par.db2.c_str())) & Parameters::DBTYPE_EXTENDED_LINFINITY;
    if (isLinfinity) {
        MMseqsNode::init();
        DBReader<DBKeyType> reader(par.db2.c_str(), par.db2Index.c_str(), 1, DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_LINFINITY);
        reader.open(DBReader<DBKeyType>::NOSORT);
        const std::vector<size_t> fileKeyOffsets = reader.getFileKeyOffsets(MMseqsNode::count);
        const std::vector<size_t> nodeKeyOffsets = MMseqsNode::getClusterKeyOffsets(par.db3 + ".keys", par.db1, reader.getSize());
        const size_t firstKey = fileKeyOffsets[MMseqsNode::rank];
        const size_t keyCount = fileKeyOffsets[MMseqsNode::rank + 1] - firstKey;
        std::vector<uint64_t> bits((keyCount + 63) / 64, 0);
        // (key, offset) cells in key order
        std::vector<uint64_t> cells(1024 * 1024 / sizeof(uint64_t));
        for (size_t node = 0; node < MMseqsNode::count; node++) {
            const size_t from = std::max(firstKey, nodeKeyOffsets[node]);
            const size_t to = std::min(firstKey + keyCount, nodeKeyOffsets[node + 1]);
            if (from >= to) {
                continue;
            }
            const std::string clu = par.db1 + "." + SSTR(node);
            const std::string cluIndex = par.db1 + ".index." + SSTR(node);
            DBReader<DBKeyType> cluDb(clu.c_str(), cluIndex.c_str(), 1, DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA | DBReader<DBKeyType>::USE_LINFINITY);
            cluDb.open(DBReader<DBKeyType>::NOSORT);
            const size_t entryBegin = cluDb.findEntryByKey(from);
            const size_t entryEnd = cluDb.findEntryByKey(to);
            cluDb.close();
            FILE *cluIndexFile = FileUtil::openFileOrDie(cluIndex.c_str(), "r", true);
            if (fseek(cluIndexFile, entryBegin * 2 * sizeof(uint64_t), SEEK_SET) != 0) {
                Debug(Debug::ERROR) << "Cannot read " << cluIndex << "\n";
                EXIT(EXIT_FAILURE);
            }
            for (size_t entry = entryBegin; entry < entryEnd;) {
                const size_t cellCount = fread(cells.data(), 2 * sizeof(uint64_t), std::min(cells.size() / 2, entryEnd - entry), cluIndexFile);
                if (cellCount == 0) {
                    Debug(Debug::ERROR) << "Cannot read " << cluIndex << "\n";
                    EXIT(EXIT_FAILURE);
                }
                for (size_t i = 0; i < cellCount; i++) {
                    const size_t key = cells[2 * i];
                    bits[(key - firstKey) / 64] |= 1ULL << ((key - firstKey) % 64);
                }
                entry += cellCount;
            }
            if (fclose(cluIndexFile) != 0) {
                Debug(Debug::ERROR) << "Cannot close " << cluIndex << "\n";
                EXIT(EXIT_FAILURE);
            }
        }
        MMseqsNode::writeRepresentativeBits(par.db3, reader, bits.data(), firstKey, keyCount);
        if (MMseqsNode::isMaster()) {
            // a mode 3 subset only feeds the next pipeline steps, which read no headers, lookup or source
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
