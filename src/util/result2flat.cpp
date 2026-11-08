#include <cstdio>
#include <Parameters.h>

#include "DBReader.h"
#include "Debug.h"
#include "Util.h"
#include "FileUtil.h"
#include "MMseqsNode.h"
#include "Timer.h"

#ifdef OPENMP
#include <omp.h>
#endif


int result2flat(int argc, const char **argv, const Command &command) {
    MMseqsNode::init();
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);
    const bool isLinfinity = DBReader<DBKeyType>::getExtendedDbtype(FileUtil::parseDbType(par.db1.c_str())) & Parameters::DBTYPE_EXTENDED_LINFINITY;
    // linfinity: every node writes <out>.<node>
    std::string nodeSuffix = isLinfinity ? "." + SSTR(MMseqsNode::rank) : "";

    DBReader<DBKeyType> querydb_header(par.hdr1.c_str(), isLinfinity ? (par.db1 + ".file_keys").c_str() : par.hdr1Index.c_str(), par.threads, DBReader<DBKeyType>::USE_INDEX|DBReader<DBKeyType>::USE_DATA|DBReader<DBKeyType>::USE_LINFINITY);
    querydb_header.open(DBReader<DBKeyType>::NOSORT);
    querydb_header.readMmapedDataInMemory();

    DBReader<DBKeyType> targetdb_header(par.hdr2.c_str(), par.hdr2Index.c_str(), par.threads, DBReader<DBKeyType>::USE_INDEX|DBReader<DBKeyType>::USE_DATA);
    DBReader<DBKeyType> seqDbr(par.db1.c_str(), par.db1Index.c_str(), par.threads, DBReader<DBKeyType>::USE_INDEX|DBReader<DBKeyType>::USE_DATA|DBReader<DBKeyType>::USE_LINFINITY);
    if (isLinfinity) {
        seqDbr.open(DBReader<DBKeyType>::NOSORT);
    } else {
        targetdb_header.open(DBReader<DBKeyType>::NOSORT);
        targetdb_header.readMmapedDataInMemory();
    }

    DBReader<DBKeyType> dbr_data((par.db3 + nodeSuffix).c_str(), (par.db3Index + nodeSuffix).c_str(), par.threads, DBReader<DBKeyType>::USE_INDEX|DBReader<DBKeyType>::USE_DATA|DBReader<DBKeyType>::USE_LINFINITY);
    if (isLinfinity == false) {
        dbr_data.open(DBReader<DBKeyType>::LINEAR_ACCCESS);
    }

    std::string fastaFile = par.db4 + nodeSuffix;
    FILE *fastaFP = FileUtil::openAndDelete(fastaFile.c_str(), "w");

    // linfinity: splits of the node file keys whose sequences and names fit
    std::vector<size_t> splitKeyOffsets;
    std::vector<size_t> cluKeyOffsets;
    if (isLinfinity) {
        cluKeyOffsets = MMseqsNode::getClusterKeyOffsets(par.db4 + ".cluster_key_offsets", par.db3, seqDbr.getSize());
        std::vector<size_t> fileKeyOffsets = seqDbr.getFileKeyOffsets(MMseqsNode::count);
        size_t fileFirstKey = fileKeyOffsets[MMseqsNode::rank];
        size_t fileEndKey = fileKeyOffsets[MMseqsNode::rank + 1];
        // name, offset and representative key
        size_t keyBytes = querydb_header.getNamesBytes(fileFirstKey, fileEndKey) / std::max((size_t) 1, fileEndKey - fileFirstKey) + 2 * sizeof(uint64_t);
        size_t splitBytes = std::max((size_t) 1, Util::computeMemory(par.splitMemoryLimit) / 8);
        std::vector<size_t> sequenceKeyOffsets = seqDbr.getSplitKeyOffsets(fileFirstKey, fileEndKey, splitBytes, keyBytes);
        // halve splits whose names do not fit
        splitKeyOffsets.push_back(fileFirstKey);
        for (size_t split = 1; split < sequenceKeyOffsets.size(); split++) {
            while (splitKeyOffsets.back() < sequenceKeyOffsets[split]) {
                size_t keyBegin = splitKeyOffsets.back();
                size_t keyEnd = sequenceKeyOffsets[split];
                while (keyEnd - keyBegin > 1 && querydb_header.getNamesBytes(keyBegin, keyEnd) + (keyEnd - keyBegin) * 2 * sizeof(uint64_t) > splitBytes) {
                    keyEnd = keyBegin + (keyEnd - keyBegin) / 2;
                }
                splitKeyOffsets.push_back(keyEnd);
            }
        }
    }
    size_t splitCount = isLinfinity ? splitKeyOffsets.size() - 1 : 1;

    size_t entriesPerTask = isLinfinity ? 4096 : 1024;
    size_t tasksPerBatch = (isLinfinity ? 4 : 16) * static_cast<size_t>(par.threads);
    std::vector<std::string> taskFastaText(tasksPerBatch);
    std::vector<size_t> taskOffsets(tasksPerBatch + 1);
    size_t fastaBytes = 0;
    std::vector<DBKeyType> repKeys;
    Timer timer;
    double loadSeconds = 0;
    double formatSeconds = 0;
    double writeSeconds = 0;
    for (size_t split = 0; split < splitCount; split++) {
        size_t splitEnd = isLinfinity ? 0 : dbr_data.getSize();
        if (isLinfinity) {
            querydb_header.loadData(splitKeyOffsets[split], splitKeyOffsets[split + 1]);
            seqDbr.loadData(splitKeyOffsets[split], splitKeyOffsets[split + 1]);
            if (split + 1 < splitCount) {
                querydb_header.prefetchData(splitKeyOffsets[split + 1], splitKeyOffsets[split + 2]);
                seqDbr.prefetchData(splitKeyOffsets[split + 1], splitKeyOffsets[split + 2]);
            }
            MMseqsNode::readClusterKeys(par.db3, cluKeyOffsets, splitKeyOffsets[split], splitKeyOffsets[split + 1], repKeys);
            splitEnd = repKeys.size();
        }
        loadSeconds += timer.getTimediff();
        timer.reset();
        size_t taskCount = (splitEnd + entriesPerTask - 1) / entriesPerTask;
        for (size_t batchBegin = 0; batchBegin < taskCount; batchBegin += tasksPerBatch) {
            size_t batchEnd = std::min(taskCount, batchBegin + tasksPerBatch);
#pragma omp parallel
            {
                int thread_idx = 0;
#ifdef OPENMP
                thread_idx = omp_get_thread_num();
#endif
#pragma omp for schedule(dynamic, 1)
                for (size_t task = batchBegin; task < batchEnd; task++) {
                    std::string &buffer = taskFastaText[task - batchBegin];
                    buffer.clear();
                    for (size_t i = task * entriesPerTask; i < std::min(splitEnd, (task + 1) * entriesPerTask); i++) {

                        // Write the header, taken from the original queryDB
                        buffer.push_back('>');
                        DBKeyType key = isLinfinity ? repKeys[i] : dbr_data.getDbKey(i);
                        char *header_data = isLinfinity ? querydb_header.getData(key, thread_idx) : querydb_header.getDataByDBKey(key, thread_idx);

                        std::string headerStr;
                        if (par.useHeader == true){
                            size_t lineLen = Util::skipLine(header_data) - header_data;
                            headerStr.assign(header_data, lineLen);

                            if (headerStr.length() > 0) {
                                if (headerStr[headerStr.length() - 1] == '\n') {
                                    headerStr[headerStr.length() - 1] = ' ';
                                }
                            }
                        }else{
                            headerStr=Util::parseFastaHeader(header_data);
                        }
                        buffer.append(headerStr);
                        buffer.push_back('\n');

                        // linfinity: sequence with newline, without null byte
                        if (isLinfinity) {
                            buffer.append(seqDbr.getData(key, thread_idx), seqDbr.getSeqLen(key) + 1);
                            continue;
                        }

                        // write data
                        char *data = dbr_data.getData(i, thread_idx);
                        std::string dbKeyBuffer;
                        const char * words[2];
                        while (*data != '\0') {
                            // dbKeyBuffer can contain sequence
                            Util::getWordsOfLine(data, words, 2);
                            //Util::parseKey(data, dbKeyBuffer);
                            char *target_header_data = NULL;
                            size_t keyLen = 0;
                            for(size_t  i = 0; i < DbValidator::resultDb.size(); i++){
                                if(Parameters::isEqualDbtype(dbr_data.getDbtype(), DbValidator::resultDb[i])  ) {
                                    keyLen = (words[1] - words[0]);
                                    dbKeyBuffer.append(words[0], keyLen);
                                    const DBKeyType dbKey = Util::fast_atoi<DBKeyType>(dbKeyBuffer.c_str());
                                    target_header_data = targetdb_header.getDataByDBKey(dbKey, thread_idx);
                                }
                            }
                            std::string dataStr;
                            if (par.useHeader == true && target_header_data != NULL)
                            {
                                dataStr = Util::parseFastaHeader(target_header_data);
                                char *endLenData = Util::skipLine(data);
                                char *dataWithoutKey = data + keyLen;
                                size_t dataToCopySize = endLenData - dataWithoutKey;
                                dataStr.append(dataWithoutKey, dataToCopySize);
                            } else {
                                char *startLine = data;
                                char *endLine = Util::skipLine(data);
                                size_t n = endLine - startLine;
                                dataStr = std::string(startLine, n);
                            }

                            // newline at the end
                            if (dataStr.length() > 0) {
                                if (dataStr[dataStr.length() - 1] != '\n') {
                                    dataStr.push_back('\n');
                                }
                            }

//                        std::cout << dataStr << std::endl;
                            buffer.append(dataStr);
                            dataStr.clear();
                            data = Util::skipLine(data);
                        }
                    }
                }
            }
            formatSeconds += timer.getTimediff();
            timer.reset();
            taskOffsets[0] = fastaBytes;
            for (size_t task = batchBegin; task < batchEnd; task++) {
                taskOffsets[task - batchBegin + 1] = taskOffsets[task - batchBegin] + taskFastaText[task - batchBegin].size();
            }
#pragma omp parallel for schedule(dynamic, 1)
            for (size_t task = batchBegin; task < batchEnd; task++) {
                const std::string &buffer = taskFastaText[task - batchBegin];
                FileUtil::pwriteAll(fileno(fastaFP), fastaFile.c_str(), buffer.data(), buffer.size(), taskOffsets[task - batchBegin]);
            }
            fastaBytes = taskOffsets[batchEnd - batchBegin];
            writeSeconds += timer.getTimediff();
            timer.reset();
        }
    }

    if (fclose(fastaFP) != 0) {
        Debug(Debug::ERROR) << "Cannot close file " << fastaFile << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (isLinfinity) {
        writeSeconds += timer.getTimediff();
        std::string phaseValues;
        MMseqsNode::addPhaseSeconds(phaseValues, "loading sequences", loadSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "formatting sequences", formatSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "writing sequences", writeSeconds);
        MMseqsNode::printPhaseValues(phaseValues);
        seqDbr.close();
    } else {
        targetdb_header.close();
        dbr_data.close();
    }
    querydb_header.close();

    return EXIT_SUCCESS;
}
