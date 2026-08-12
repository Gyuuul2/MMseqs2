#include "Debug.h"
#include "DBReader.h"
#include "DBWriter.h"
#include "Util.h"
#include "itoa.h"

#include <list>

#ifdef OPENMP
#include <omp.h>
#endif

// every key of every input clustering has to be a sequence of db1, or the id indexes the flat arrays out of bounds
static size_t requireId(DBReader<DBKeyType> &dbr, DBKeyType key) {
    const size_t id = dbr.getId(key);
    if (id == DB_ENTRY_NOT_FOUND) {
        Debug(Debug::ERROR) << "Key " << key << " of a cluster db is not in the sequence db\n";
        EXIT(EXIT_FAILURE);
    }
    return id;
}

int mergeclusters(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, true, 0);

    std::list<std::string> clusterings;
    for (size_t i = 2; i < par.filenames.size(); i++) {
        clusterings.push_back(par.filenames[i]);
    }

    // the sequence database will serve as the reference for sequence indexes
    DBReader<DBKeyType> dbr(par.db1.c_str(), par.db1Index.c_str(), par.threads, DBReader<DBKeyType>::USE_INDEX);
    dbr.open(DBReader<DBKeyType>::NOSORT);

    // init the structure for cluster merging
    // it has the size of all possible cluster (sequence amount)
    // circular singly linked lists: the head is next[tail], so holding the tail drops one array per sequence
    const size_t NO_MEMBER = SIZE_MAX;
    size_t *clusterTail = new(std::nothrow) size_t[dbr.getSize()];
    size_t *nextMember = new(std::nothrow) size_t[dbr.getSize()];
    Util::checkAllocation(clusterTail, "Cannot allocate clusterTail memory in mergeclusters");
    Util::checkAllocation(nextMember, "Cannot allocate nextMember memory in mergeclusters");
    // also the parallel first touch, so the pages are not all faulted in by one thread
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < dbr.getSize(); i++) {
        clusterTail[i] = NO_MEMBER;
        nextMember[i] = NO_MEMBER;
    }

    // read the clustering from the first clustering step
    std::string firstClu = clusterings.front();
    std::string firstCluStepIndex = firstClu + ".index";
    clusterings.pop_front();

    Debug(Debug::INFO) << "Clustering step 1\n";
    DBReader<DBKeyType> cluDb(firstClu.c_str(), firstCluStepIndex.c_str(), par.threads, DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA);
    // cluDb is read by local id only, so the maps LINEAR_ACCCESS builds are dead weight here
    cluDb.open(DBReader<DBKeyType>::NOSORT);

    Debug::Progress progress(cluDb.getSize());
#pragma omp parallel
    {
        int thread_idx = 0;
#ifdef OPENMP
        thread_idx = omp_get_thread_num();
#endif

        char keyBuffer[255];
#pragma omp for schedule(dynamic, 100)
        for (size_t i = 0; i < cluDb.getSize(); i++) {
            progress.updateProgress();
            DBKeyType clusterId = cluDb.getDbKey(i);
            size_t cluId = requireId(dbr, clusterId);
            char *data = cluDb.getData(i, thread_idx);
            // go through the sequences in the cluster and add them to the initial clustering
            while (*data != '\0') {
                Util::parseKey(data, keyBuffer);
                DBKeyType key = Util::fast_atoi<DBKeyType>(keyBuffer);
                size_t seqId = requireId(dbr, key);
                if (clusterTail[cluId] == NO_MEMBER) {
                    nextMember[seqId] = seqId;
                } else {
                    nextMember[seqId] = nextMember[clusterTail[cluId]];
                    nextMember[clusterTail[cluId]] = seqId;
                }
                clusterTail[cluId] = seqId;
                data = Util::skipLine(data);
            }
        }
    }
    cluDb.close();

    // merge later clustering steps into the initial clustering step
    int cnt = 2;
    while (!clusterings.empty()) {
        Debug(Debug::INFO) << "Clustering step " << cnt << "\n";

        std::string cluStep = clusterings.front();
        std::string cluStepIndex = cluStep + ".index";
        clusterings.pop_front();

        DBReader<DBKeyType> cluDb(cluStep.c_str(), cluStepIndex.c_str(), par.threads, DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA);
        cluDb.open(DBReader<DBKeyType>::NOSORT);

        progress.reset(cluDb.getSize());
        // go through the clusters and merge them into the clusters from the previous clustering step
#pragma omp parallel
        {
            int thread_idx = 0;
#ifdef OPENMP
            thread_idx = omp_get_thread_num();
#endif
            char keyBuffer[255];
#pragma omp for schedule(dynamic, 100)
            for (size_t i = 0; i < cluDb.getSize(); i++) {
                progress.updateProgress();
                // go through the sequences in the cluster and add them and their clusters to the cluster of cluId
                // afterwards, delete the added cluster from the clustering
                size_t cluId = requireId(dbr, cluDb.getDbKey(i));
                char *data = cluDb.getData(i, thread_idx);
                while (*data != '\0') {
                    Util::parseKey(data, keyBuffer);
                    DBKeyType key = Util::fast_atoi<DBKeyType>(keyBuffer);
                    size_t seqId = requireId(dbr, key);
                    // to avoid copies of the same cluster list
                    if (seqId != cluId && clusterTail[seqId] != NO_MEMBER) {
                        if (clusterTail[cluId] != NO_MEMBER) {
                            // swap the two heads so this list runs first and the other one follows
                            const size_t ownHead = nextMember[clusterTail[cluId]];
                            nextMember[clusterTail[cluId]] = nextMember[clusterTail[seqId]];
                            nextMember[clusterTail[seqId]] = ownHead;
                        }
                        clusterTail[cluId] = clusterTail[seqId];
                        // splice leaves the source empty
                        clusterTail[seqId] = NO_MEMBER;
                    }
                    data = Util::skipLine(data);
                }
            }
        }
        cluDb.close();
        cnt++;
    }

    Debug(Debug::INFO) << "Write merged clustering\n";
    DBWriter dbw(par.db2.c_str(), par.db2Index.c_str(), par.threads, par.compressed, Parameters::DBTYPE_CLUSTER_RES);
    dbw.open();
    progress.reset(dbr.getSize());
#pragma omp parallel
    {
        int thread_idx = 0;
#ifdef OPENMP
        thread_idx = omp_get_thread_num();
#endif

        std::string res;
        res.reserve(1024 * 1024);

        char buffer[32];

        // go through all sequences in the database
#pragma omp for schedule(dynamic, 100)
        for (size_t i = 0; i < dbr.getSize(); i++) {
            progress.updateProgress();

            // no cluster for this representative
            if (clusterTail[i] == NO_MEMBER)
                continue;

            // representative
            DBKeyType dbKey = dbr.getDbKey(i);
            const size_t tail = clusterTail[i];
            size_t member = nextMember[tail];
            while (true) {
                char *tmpBuff = Itoa::u64toa_sse2(static_cast<uint64_t>(dbr.getDbKey(member)), buffer);
                size_t length = tmpBuff - buffer - 1;
                res.append(buffer, length);
                res.push_back('\n');
                if (member == tail) {
                    break;
                }
                member = nextMember[member];
            }

            dbw.writeData(res.c_str(), res.length(), dbKey, thread_idx);
            res.clear();
        }
    }
    dbw.close();
    dbr.close();

    delete[] nextMember;
    delete[] clusterTail;

    return EXIT_SUCCESS;
}
