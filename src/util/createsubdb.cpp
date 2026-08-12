#include "Parameters.h"
#include "FileUtil.h"
#include "DBReader.h"
#include "DBWriter.h"
#include "Debug.h"
#include "Util.h"

#include <climits>
#include <vector>

int createsubdb(int argc, const char **argv, const Command& command) {
    Parameters& par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);

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
    DBReader<DBKeyType> reader(par.db2.c_str(), par.db2Index.c_str(), par.threads, dbMode);
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
    // NOSORT keeps the index key sorted, so an ascending order file needs only a forward cursor, not getId
    const DBReader<DBKeyType>::Index *index = reader.getIndex();
    const size_t readerSize = reader.getSize();
    size_t cursor = 0;
    // one fwrite per row is one stdio lock per row, so batch them and let writeIndex format in parallel
    const size_t indexBatch = 1024 * 1024;
    std::vector<DBReader<DBKeyType>::Index> pending;
    if (par.subDbMode == Parameters::SUBDB_MODE_SOFT) {
        pending.reserve(indexBatch);
    }
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

        const bool ascending = isOrdered && (prevKey <= key);
        isOrdered &= (prevKey <= key);
        prevKey = key;
        size_t id;
        if (ascending) {
            while (cursor < readerSize && index[cursor].id < key) {
                cursor++;
            }
            id = (cursor < readerSize && index[cursor].id == key) ? cursor : SIZE_MAX;
        } else {
            id = reader.getId(key);
        }
        if (id == SIZE_MAX) {
            Debug(Debug::WARNING) << "Key " << dbKey << " not found in database\n";
            continue;
        }
        if (par.subDbMode == Parameters::SUBDB_MODE_SOFT) {
            // getEntryLen returns the index's own unsigned int length, so this stores it unchanged
            DBReader<DBKeyType>::Index entry;
            entry.id = key;
            entry.offset = reader.getOffset(id);
            entry.length = static_cast<unsigned int>(reader.getEntryLen(id));
            pending.push_back(entry);
            if (pending.size() == indexBatch) {
                writer.writeIndexEntries(pending.data(), pending.size(), 0, par.threads);
                pending.clear();
            }
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
    if (pending.empty() == false) {
        writer.writeIndexEntries(pending.data(), pending.size(), 0, par.threads);
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
