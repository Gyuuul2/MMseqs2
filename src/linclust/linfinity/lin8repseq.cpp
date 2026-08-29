#include "Lin8DbReader.h"
#include "Parameters.h"
#include "Debug.h"
#include "DBReader.h"
#include "FileUtil.h"
#include "Util.h"
#include "Timer.h"

#include <algorithm>

#ifdef OPENMP
#include <omp.h>
#endif
#include <cstdio>
#include <string>
#include <vector>

int lin8repseq(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);

    RunDbReader reader(par.db1, "", true);
    reader.open();

    Timer timer;
    DBReader<DBKeyType> clusters(par.db2.c_str(), par.db2Index.c_str(), par.threads,
                                 DBReader<DBKeyType>::USE_INDEX);
    clusters.open(DBReader<DBKeyType>::LINEAR_ACCCESS);
    std::vector<uint64_t> reps(clusters.getSize());
    for (size_t i = 0; i < clusters.getSize(); i++) {
        reps[i] = clusters.getDbKey(i);
    }
    std::sort(reps.begin(), reps.end());

    // Only the names of the ranks that head a cluster. Keeping one for every sequence is thirty two
    // byte a rank before a single character of it, which at a trillion is more than the machine has.
    std::vector<std::string> nameOfRep(reps.size());
    RunDbReader::HeaderStream headers(reader);
    const char *begin = NULL;
    size_t length = 0;
    uint64_t rank = 0;
    size_t want = 0;
    while (headers.next(begin, length)) {
        if (rank >= reader.getSize()) {
            Debug(Debug::ERROR) << "The headers hold more entries than the " << reader.getSize()
                                << " the run table names\n";
            EXIT(EXIT_FAILURE);
        }
        while (want < reps.size() && reps[want] < rank) {
            want++;
        }
        if (want < reps.size() && reps[want] == rank) {
            nameOfRep[want] = Util::parseFastaHeader(begin);
        }
        rank++;
    }
    if (rank != reader.getSize()) {
        Debug(Debug::ERROR) << "The headers hold " << rank << " entries and the run table names "
                            << reader.getSize() << "\n";
        EXIT(EXIT_FAILURE);
    }
    Debug(Debug::INFO) << "Read " << reps.size() << " representative names in "
                       << timer.lap() << "\n";


    const size_t splits = par.fastaSplits > 1 ? (size_t) par.fastaSplits : 1;
    const unsigned int threads = (unsigned int) std::min<size_t>(splits, par.threads);

    std::vector<size_t> edge(splits + 1, reps.size());
    {
        RunDbReader::Cursor at;
        std::vector<uint64_t> upto(reps.size() + 1, 0);
        for (size_t i = 0; i < reps.size(); i++) {
            upto[i + 1] = upto[i] + reader.getSeqLen(reps[i], at) + nameOfRep[i].size() + 2;
        }
        edge[0] = 0;
        for (size_t split = 1; split < splits; split++) {
            const uint64_t want = upto.back() * split / splits;
            edge[split] = std::lower_bound(upto.begin(), upto.end(), want) - upto.begin();
            if (edge[split] > reps.size()) {
                edge[split] = reps.size();
            }
        }
    }
    std::vector<uint64_t> counted(splits, 0);
    Debug::Progress progress(reps.size());
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (size_t split = 0; split < splits; split++) {
        const size_t from = edge[split];
        const size_t until = edge[split + 1];
        // zero padded, so listing the shards in name order lists them in rank order
        char suffix[16];
        snprintf(suffix, sizeof(suffix), ".%05zu", split);
        const std::string path = splits == 1 ? par.db3 : par.db3 + suffix;
        const std::string tmp = path + ".tmp";
        FILE *out = FileUtil::openAndDelete(tmp.c_str(), "w");
        RunDbReader::Cursor cursor;
        std::string record;
        record.reserve(1u << 20);
        for (size_t i = from; i < until; i++) {
            const uint64_t rep = reps[i];
            if (rep >= reader.getSize()) {
                Debug(Debug::ERROR) << "The clustering names rank " << rep << ", past the database\n";
                EXIT(EXIT_FAILURE);
            }
            const uint32_t len = reader.getSeqLen(rep, cursor);
            const char *residues = reader.getData(rep, cursor);
            record.append(1, '>');
            record.append(nameOfRep[i]);
            record.append(1, '\n');
            record.append(residues, len);
            record.append(1, '\n');
            if (record.size() >= (1u << 20)) {
                if (fwrite(record.c_str(), 1, record.size(), out) != record.size()) {
                    Debug(Debug::ERROR) << "Cannot write to " << tmp << "\n";
                    EXIT(EXIT_FAILURE);
                }
                record.clear();
            }
            counted[split]++;
            progress.updateProgress();
        }
        if (record.empty() == false
            && fwrite(record.c_str(), 1, record.size(), out) != record.size()) {
            Debug(Debug::ERROR) << "Cannot write to " << tmp << "\n";
            EXIT(EXIT_FAILURE);
        }
        if (fclose(out) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << tmp << "\n";
            EXIT(EXIT_FAILURE);
        }
        FileUtil::publishAtomically(tmp, path);
    }
    uint64_t written = 0;
    for (size_t i = 0; i < splits; i++) {
        written += counted[i];
    }

    clusters.close();
    reader.close();
    Debug(Debug::INFO) << "Wrote " << written << " representatives in " << timer.lap() << "\n";
    return EXIT_SUCCESS;
}
