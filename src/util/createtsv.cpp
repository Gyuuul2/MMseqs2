#include "Parameters.h"
#include "DBReader.h"
#include "DBWriter.h"
#include "Debug.h"
#include "Util.h"
#include "IndexReader.h"
#include "FileUtil.h"

#include <mutex>
#include <vector>

#ifdef OPENMP
#include <omp.h>

#endif

#ifndef SIZE_T_MAX
#define SIZE_T_MAX ((size_t) -1)
#endif

// byte-identical port of batch_clustering.sh SPLIT_HASH_AWK: h = (h*131 + byte) % B over the column's bytes
static unsigned int tsvSplitOfColumn(const std::string &line, int column, unsigned int splits) {
    size_t start = 0;
    for (int c = 1; c < column; ++c) {
        size_t tab = line.find('\t', start);
        start = (tab == std::string::npos) ? line.size() : tab + 1;
    }
    uint64_t h = 0;
    for (size_t i = start; i < line.size(); ++i) {
        const unsigned char byte = line[i];
        if (byte == '\t' || byte == '\n') {
            break;
        }
        h = (h * 131 + byte) % splits;
    }
    return (unsigned int) h;
}

int createtsv(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, Parameters::PARSE_VARIADIC, 0);

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
    // opt-in split mode (batch clustering): dataFile becomes a prefix for <prefix>.split%05d.tsv
    const unsigned int splits = par.tsvSplits > 0 ? (unsigned int) par.tsvSplits : 0;
    const bool splitMode = splits > 0;
    if (splitMode && par.dbOut) {
        Debug(Debug::ERROR) << "--tsv-splits cannot be combined with --db-output\n";
        return EXIT_FAILURE;
    }
    DBWriter writer(dataFile.c_str(), indexFile.c_str(), par.threads, shouldCompress, dbType);
    std::vector<FILE*> splitFiles(splitMode ? splits : 0);
    std::vector<std::mutex> splitLocks(splitMode ? splits : 0);
    if (splitMode) {
        char splitName[FILENAME_MAX];
        for (unsigned int b = 0; b < splits; ++b) {
            snprintf(splitName, sizeof(splitName), "%s.split%05u.tsv", dataFile.c_str(), b);
            splitFiles[b] = FileUtil::openAndDelete(splitName, "w");
        }
    } else {
        writer.open();
    }

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
        std::string lineBuffer;
        lineBuffer.reserve(1024);
        const size_t splitFlushSize = 64 * 1024;
        std::vector<std::string> splitBuffers(splitMode ? splits : 0);

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

                lineBuffer.clear();
                lineBuffer.append(queryHeader);
                lineBuffer.append("\t");
                lineBuffer.append(targetAccession);

                size_t offset = 0;
                if (targetColumn != 0) {
                    lineBuffer.append("\t");
                    offset = 0;
                } else {
                    offset = strlen(dbKey);
                }

                char *nextLine = Util::skipLine(data);
                lineBuffer.append(data + offset, (nextLine - (data + offset)) - 1);
                lineBuffer.append("\n");
                if (splitMode) {
                    const unsigned int b = tsvSplitOfColumn(lineBuffer, par.tsvSplitColumn, splits);
                    std::string &buf = splitBuffers[b];
                    buf.append(lineBuffer);
                    if (buf.size() >= splitFlushSize) {
                        std::lock_guard<std::mutex> lock(splitLocks[b]);
                        if (fwrite(buf.c_str(), sizeof(char), buf.size(), splitFiles[b]) != buf.size()) {
                            Debug(Debug::ERROR) << "Cannot write to split file " << b << "\n";
                            EXIT(EXIT_FAILURE);
                        }
                        buf.clear();
                    }
                } else {
                    outputBuffer.append(lineBuffer);
                }
                data = nextLine;
                entryIndex++;
            }
            if (splitMode == false) {
                writer.writeData(outputBuffer.c_str(), outputBuffer.length(), queryKey, thread_idx, par.dbOut);
                outputBuffer.clear();
            }
        }
        if (splitMode) {
            for (unsigned int b = 0; b < splits; ++b) {
                if (splitBuffers[b].empty()) {
                    continue;
                }
                std::lock_guard<std::mutex> lock(splitLocks[b]);
                if (fwrite(splitBuffers[b].c_str(), sizeof(char), splitBuffers[b].size(), splitFiles[b]) != splitBuffers[b].size()) {
                    Debug(Debug::ERROR) << "Cannot write to split file " << b << "\n";
                    EXIT(EXIT_FAILURE);
                }
                splitBuffers[b].clear();
            }
        }
        delete[] dbKey;
    }
    if (splitMode) {
        for (unsigned int b = 0; b < splits; ++b) {
            if (fclose(splitFiles[b]) != 0) {
                Debug(Debug::ERROR) << "Cannot close split file " << b << "\n";
                return EXIT_FAILURE;
            }
        }
    } else {
        writer.close(par.dbOut == false);
        if (par.dbOut == false) {
            if (hasTargetDB) {
                FileUtil::remove(par.db4Index.c_str());
            } else {
                FileUtil::remove(par.db3Index.c_str());
            }
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
