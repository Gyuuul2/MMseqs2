/*
 * convert2fasta
 * written by Milot Mirdita <milot@mirdita.de>
 */

#include <cstring>
#include <cstdio>
#include <string>
#include <vector>

#include "Parameters.h"
#include "DBReader.h"
#include "Debug.h"
#include "Util.h"
#include "FileUtil.h"
#include "Sequence.h"
#include "SubstitutionMatrix.h"

#ifdef OPENMP
#include <omp.h>
#endif

const char headerStart[] = {'>'};
const char newline[] = {'\n'};

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

static void writeFastaEntry(FILE *out, const char *headerData, size_t headerLen,
                            const char *bodyData, size_t seqLen) {
    fwrite(headerStart, sizeof(char), 1, out);
    fwrite(headerData, sizeof(char), headerLen - 2, out);
    fwrite(newline, sizeof(char), 1, out);
    fwrite(bodyData, sizeof(char), seqLen, out);
    fwrite(newline, sizeof(char), 1, out);
}

int convert2fasta(int argc, const char **argv, const Command& command) {
    Parameters& par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);

    DBReader<DBKeyType> db(par.db1.c_str(), par.db1Index.c_str(), par.threads, DBReader<DBKeyType>::USE_DATA|DBReader<DBKeyType>::USE_INDEX);
    db.open(DBReader<DBKeyType>::NOSORT);

    DBReader<DBKeyType> db_header(par.hdr1.c_str(), par.hdr1Index.c_str(), par.threads, DBReader<DBKeyType>::USE_DATA|DBReader<DBKeyType>::USE_INDEX);
    db_header.open(DBReader<DBKeyType>::NOSORT);

    DBReader<DBKeyType>* order = par.useHeaderFile ? &db_header : &db;
    // entry i goes to shard i % splits, so a worker owns whole shards and no offset is shared
    if (par.fastaSplits > 0) {
        if (Sequence::getAuxInfo(db.getDbtype()) != NULL) {
            Debug(Debug::ERROR) << "--fasta-splits cannot be combined with an auxiliary sequence database\n";
            EXIT(EXIT_FAILURE);
        }
        const int splits = par.fastaSplits;
        Debug(Debug::INFO) << "Start writing " << splits << " split files for " << par.db2 << "\n";
        const int splitThreads = std::min(par.threads, splits);
#pragma omp parallel for schedule(dynamic, 1) num_threads(splitThreads)
        for (int split = 0; split < splits; split++) {
            unsigned int thread_idx = 0;
#ifdef OPENMP
            thread_idx = static_cast<unsigned int>(omp_get_thread_num());
#endif
            const std::string name = splitFastaName(par.db2, split);
            FILE *splitFP = fopen(name.c_str(), "w");
            if (splitFP == NULL) {
                perror(name.c_str());
                EXIT(EXIT_FAILURE);
            }
            for (size_t i = split; i < order->getSize(); i += splits) {
                const DBKeyType key = order->getDbKey(i);
                const size_t headerKey = db_header.getId(key);
                const size_t bodyKey = db.getId(key);
                writeFastaEntry(splitFP, db_header.getData(headerKey, thread_idx),
                                db_header.getEntryLen(headerKey),
                                db.getData(bodyKey, thread_idx), db.getEntryLen(bodyKey) - 2);
            }
            if (fclose(splitFP) != 0) {
                Debug(Debug::ERROR) << "Cannot close file " << name << "\n";
                EXIT(EXIT_FAILURE);
            }
        }
        db_header.close();
        db.close();
        return EXIT_SUCCESS;
    }

    FILE* fastaFP = fopen(par.db2.c_str(), "w");
    if(fastaFP == NULL) {
        perror(par.db2.c_str());
        EXIT(EXIT_FAILURE);
    }

    // Check for combined primary+aux format
    const Sequence::SeqAuxInfo* auxInfo = Sequence::getAuxInfo(db.getDbtype());
    FILE* auxFastaFP = NULL;
    std::string auxPath;
    SubstitutionMatrix* subMat = NULL;
    if (auxInfo != NULL) {
        const std::string ext = ".fasta";
        if (par.db2.length() >= ext.length() &&
            par.db2.substr(par.db2.length() - ext.length()) == ext) {
            auxPath = par.db2.substr(0, par.db2.length() - ext.length()) + ".aux.fasta";
        } else {
            auxPath = par.db2 + ".aux";
        }
        auxFastaFP = fopen(auxPath.c_str(), "w");
        if (auxFastaFP == NULL) {
            perror(auxPath.c_str());
            fclose(fastaFP);
            EXIT(EXIT_FAILURE);
        }
        subMat = new SubstitutionMatrix(par.scoringMatrixFile.values.aminoacid().c_str(), 2.1, 0.0);
    }

    DBReader<DBKeyType>* from = &db;
    if(par.useHeaderFile) {
        from = &db_header;
    }

    Debug(Debug::INFO) << "Start writing file to " << par.db2 << "\n";
    std::vector<char> decodedBuf;
    for(size_t i = 0; i < from->getSize(); i++){
        DBKeyType key = from->getDbKey(i);
        size_t headerKey = db_header.getId(key);
        const char* headerData = db_header.getData(headerKey, 0);
        const size_t headerLen = db_header.getEntryLen(headerKey);

        fwrite(headerStart, sizeof(char), 1, fastaFP);
        fwrite(headerData, sizeof(char), headerLen - 2, fastaFP);
        fwrite(newline, sizeof(char), 1, fastaFP);

        size_t bodyKey = db.getId(key);
        const char* bodyData = db.getData(bodyKey, 0);
        const size_t bodyLen = db.getEntryLen(bodyKey);
        size_t seqLen = bodyLen - 2;

        if (auxInfo != NULL) {
            if (decodedBuf.size() < seqLen) {
                decodedBuf.resize(seqLen);
            }
            for (size_t j = 0; j < seqLen; j++) {
                decodedBuf[j] = subMat->num2aa[auxInfo->primaryRemap[(unsigned char)bodyData[j]]];
            }
            fwrite(decodedBuf.data(), sizeof(char), seqLen, fastaFP);
            fwrite(newline, sizeof(char), 1, fastaFP);

            fwrite(headerStart, sizeof(char), 1, auxFastaFP);
            fwrite(headerData, sizeof(char), headerLen - 2, auxFastaFP);
            fwrite(newline, sizeof(char), 1, auxFastaFP);
            for (size_t j = 0; j < seqLen; j++) {
                decodedBuf[j] = subMat->num2aa[auxInfo->auxRemap[(unsigned char)bodyData[j]]];
            }
            fwrite(decodedBuf.data(), sizeof(char), seqLen, auxFastaFP);
            fwrite(newline, sizeof(char), 1, auxFastaFP);
        } else {
            fwrite(bodyData, sizeof(char), seqLen, fastaFP);
            fwrite(newline, sizeof(char), 1, fastaFP);
        }
    }
    if (fclose(fastaFP) != 0) {
        Debug(Debug::ERROR) << "Cannot close file " << par.db2 << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (auxFastaFP != NULL && fclose(auxFastaFP) != 0) {
        Debug(Debug::ERROR) << "Cannot close file " << auxPath << "\n";
        EXIT(EXIT_FAILURE);
    }
    delete subMat;
    db_header.close();
    db.close();

    return EXIT_SUCCESS;
}
