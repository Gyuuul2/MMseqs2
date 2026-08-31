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
#include <fcntl.h>
#include <unistd.h>

static void writeAt(int fd, const std::string &path, std::string &record, uint64_t &at) {
    if (record.empty()) {
        return;
    }
    const ssize_t wrote = pwrite(fd, record.c_str(), record.size(), (off_t) at);
    if (wrote < 0 || (size_t) wrote != record.size()) {
        Debug(Debug::ERROR) << "Cannot write " << record.size() << " byte to " << path << "\n";
        EXIT(EXIT_FAILURE);
    }
    at += record.size();
    record.clear();
}

int lin8createrepseqfasta(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);

    RunDbReader reader(par.db1, true);
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

    // only the ranks that head a cluster: one name a sequence is more than the machine has
    const size_t budget = Util::computeMemory(par.splitMemoryLimit);
    const size_t need = reps.size() * (sizeof(std::string) + 24);
    if (need > budget) {
        Debug(Debug::ERROR) << "Naming " << reps.size() << " representatives needs about "
                            << (need >> 30) << " GB of names and the limit is " << (budget >> 30)
                            << " GB. The cluster database holds the same answer without them\n";
        EXIT(EXIT_FAILURE);
    }
    std::vector<std::string> nameOfRep(reps.size());
    RunDbReader::HeaderStream headers(reader);
    const char *begin = NULL;
    size_t length = 0;
    uint64_t rank = 0;
    size_t want = 0;
    while (headers.next(begin, length)) {
        if (rank >= reader.getSize()) {
            Debug(Debug::ERROR) << "The headers hold more entries than the " << reader.getSize()
                                << " the sequence locator names\n";
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
        Debug(Debug::ERROR) << "The headers hold " << rank << " entries and the sequence locator names "
                            << reader.getSize() << "\n";
        EXIT(EXIT_FAILURE);
    }
    Debug(Debug::INFO) << "Read " << reps.size() << " representative names in "
                       << timer.lap() << "\n";


    // pieces are a question about threads, files are a question about --fasta-splits
    const bool sharded = par.fastaSplits > 1;
    const size_t splits = sharded ? (size_t) par.fastaSplits
                                  : std::max<size_t>(1, (size_t) par.threads);
    const unsigned int threads = (unsigned int) std::min<size_t>(splits, par.threads);

    // two walks rather than a running total a representative, the byte being what it pwrites at
    std::vector<size_t> edge(splits + 1, reps.size());
    std::vector<uint64_t> edgeByte(splits + 1, 0);
    {
        edge[0] = 0;
        uint64_t total = 0;
        RunDbReader::Cursor at;
        for (size_t i = 0; i < reps.size(); i++) {
            total += reader.getSeqLen(reps[i], at) + nameOfRep[i].size() + 3;
        }
        RunDbReader::Cursor again;
        uint64_t sum = 0;
        size_t next = 1;
        for (size_t i = 0; i < reps.size() && next < splits; i++) {
            sum += reader.getSeqLen(reps[i], again) + nameOfRep[i].size() + 3;
            while (next < splits && sum >= total * next / splits) {
                edge[next] = i + 1;
                edgeByte[next] = sum;
                next++;
            }
        }
        edgeByte[splits] = total;
    }
    std::vector<uint64_t> counted(splits, 0);
    Debug::Progress progress(reps.size());

    // one file written by every thread at its own byte; shards keep a file each
    const std::string wholeTmp = par.db3 + ".tmp";
    int whole = -1;
    if (sharded == false) {
        whole = open(wholeTmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
        if (whole < 0 || ftruncate(whole, (off_t) edgeByte[splits]) != 0) {
            Debug(Debug::ERROR) << "Cannot make " << wholeTmp << " of " << edgeByte[splits]
                                << " byte\n";
            EXIT(EXIT_FAILURE);
        }
    }

#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (size_t split = 0; split < splits; split++) {
        const size_t from = edge[split];
        const size_t until = edge[split + 1];
        // zero padded, so listing the shards in name order lists them in rank order
        char suffix[16];
        snprintf(suffix, sizeof(suffix), ".%05zu", split);
        const std::string path = sharded ? par.db3 + suffix : par.db3;
        const std::string tmp = path + ".tmp";
        int out = whole;
        if (sharded) {
            out = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
            if (out < 0) {
                Debug(Debug::ERROR) << "Cannot open " << tmp << " for writing\n";
                EXIT(EXIT_FAILURE);
            }
        }
        uint64_t at = sharded ? 0 : edgeByte[split];
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
                writeAt(out, tmp, record, at);
            }
            counted[split]++;
            progress.updateProgress();
        }
        writeAt(out, tmp, record, at);
        if (sharded) {
            if (::close(out) != 0) {
                Debug(Debug::ERROR) << "Cannot close " << tmp << "\n";
                EXIT(EXIT_FAILURE);
            }
            FileUtil::publishAtomically(tmp, path);
        }
    }
    if (sharded == false) {
        if (::close(whole) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << wholeTmp << "\n";
            EXIT(EXIT_FAILURE);
        }
        FileUtil::publishAtomically(wholeTmp, par.db3);
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
