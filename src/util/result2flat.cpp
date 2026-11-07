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
    // linfinity: every node writes the representatives of its cluster database <clu>.<node> with their sequences to <out>.<node>
    const std::string nodeSuffix = isLinfinity ? "." + SSTR(MMseqsNode::rank) : "";

    DBReader<DBKeyType> querydb_header(par.hdr1.c_str(), isLinfinity ? (par.db1 + ".nodeinfo").c_str() : par.hdr1Index.c_str(), par.threads, DBReader<DBKeyType>::USE_INDEX|DBReader<DBKeyType>::USE_DATA|DBReader<DBKeyType>::USE_LINFINITY);
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

    const std::string fastaFile = par.db4 + nodeSuffix;
    FILE *fastaFP = FileUtil::openAndDelete(fastaFile.c_str(), "w");

    // one split of all entries; linfinity: splits of the keys of the data file of this node, cut by sequence bytes as the written sequences are, whose sequences and names fit the memory, loaded and prefetched
    std::vector<size_t> splitKeyOffsets;
    std::vector<size_t> cluKeyOffsets;
    if (isLinfinity) {
        cluKeyOffsets = MMseqsNode::getClusterKeyOffsets(par.db4 + ".keys", par.db3, seqDbr.getSize());
        const std::vector<size_t> fileKeyOffsets = seqDbr.getFileKeyOffsets(MMseqsNode::count);
        const size_t fileFirstKey = fileKeyOffsets[MMseqsNode::rank];
        const size_t fileEndKey = fileKeyOffsets[MMseqsNode::rank + 1];
        // a key also holds its name with its offset and its representative key
        const size_t keyBytes = querydb_header.getNamesSize(fileFirstKey, fileEndKey) / std::max<size_t>(1, fileEndKey - fileFirstKey) + 2 * sizeof(uint64_t);
        const size_t splitBytes = std::max<size_t>(1, Util::computeMemory(par.splitMemoryLimit) / 8);
        // the average sizes the sequence splits; uneven names must still fit the split
        const std::vector<size_t> sequenceKeyOffsets = seqDbr.getSplitKeyOffsets(fileFirstKey, fileEndKey, splitBytes, keyBytes);
        splitKeyOffsets.push_back(fileFirstKey);
        for (size_t split = 1; split < sequenceKeyOffsets.size(); split++) {
            while (splitKeyOffsets.back() < sequenceKeyOffsets[split]) {
                const size_t keyBegin = splitKeyOffsets.back();
                size_t keyEnd = sequenceKeyOffsets[split];
                while (keyEnd - keyBegin > 1 && querydb_header.getNamesSize(keyBegin, keyEnd) + (keyEnd - keyBegin) * 2 * sizeof(uint64_t) > splitBytes) {
                    keyEnd = keyBegin + (keyEnd - keyBegin) / 2;
                }
                splitKeyOffsets.push_back(keyEnd);
            }
        }
    }
    const size_t splitCount = isLinfinity ? splitKeyOffsets.size() - 1 : 1;

    const size_t chunkEntries = isLinfinity ? 4096 : 1024;
    // the chunks that the threads format at once before they are written at their prefix-summed offsets
    const size_t batchChunks = (isLinfinity ? 4 : 16) * static_cast<size_t>(par.threads);
    std::vector<std::string> chunkData(batchChunks);
    std::vector<size_t> chunkOffsets(batchChunks + 1);
    size_t fastaSize = 0;
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
        const size_t chunkCount = (splitEnd + chunkEntries - 1) / chunkEntries;
        for (size_t batchBegin = 0; batchBegin < chunkCount; batchBegin += batchChunks) {
            const size_t batchEnd = std::min(chunkCount, batchBegin + batchChunks);
#pragma omp parallel
            {
                int thread_idx = 0;
#ifdef OPENMP
                thread_idx = omp_get_thread_num();
#endif
#pragma omp for schedule(dynamic, 1)
                for (size_t chunk = batchBegin; chunk < batchEnd; chunk++) {
                    std::string &buffer = chunkData[chunk - batchBegin];
                    buffer.clear();
                    for (size_t i = chunk * chunkEntries; i < std::min(splitEnd, (chunk + 1) * chunkEntries); i++) {

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

                        // linfinity: the sequence of the representative with its newline, without its null byte
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
            chunkOffsets[0] = fastaSize;
            for (size_t chunk = batchBegin; chunk < batchEnd; chunk++) {
                chunkOffsets[chunk - batchBegin + 1] = chunkOffsets[chunk - batchBegin] + chunkData[chunk - batchBegin].size();
            }
#pragma omp parallel for schedule(dynamic, 1)
            for (size_t chunk = batchBegin; chunk < batchEnd; chunk++) {
                const std::string &buffer = chunkData[chunk - batchBegin];
                FileUtil::pwriteAll(fileno(fastaFP), fastaFile.c_str(), buffer.data(), buffer.size(), chunkOffsets[chunk - batchBegin]);
            }
            fastaSize = chunkOffsets[batchEnd - batchBegin];
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
        MMseqsNode::addPhaseSeconds(phaseValues, "loading the sequences", loadSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "formatting the sequences", formatSeconds);
        MMseqsNode::addPhaseSeconds(phaseValues, "writing the sequences", writeSeconds);
        MMseqsNode::printPhaseValues(phaseValues);
        seqDbr.close();
    } else {
        targetdb_header.close();
        dbr_data.close();
    }
    querydb_header.close();

    return EXIT_SUCCESS;
}
