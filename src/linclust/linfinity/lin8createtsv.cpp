#include "Lin8DbReader.h"
#include "Parameters.h"
#include "Debug.h"
#include "DBReader.h"
#include "FileUtil.h"
#include "Util.h"
#include "Timer.h"

#include <cstdio>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

static void writeAt(int fd, const std::string &path, std::string &line, uint64_t &at) {
    if (line.empty()) {
        return;
    }
    const ssize_t wrote = pwrite(fd, line.c_str(), line.size(), (off_t) at);
    if (wrote < 0 || (size_t) wrote != line.size()) {
        Debug(Debug::ERROR) << "Cannot write " << line.size() << " byte to " << path << "\n";
        EXIT(EXIT_FAILURE);
    }
    at += line.size();
    line.clear();
}

int lin8createtsv(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);

    RunDbReader reader(par.db1, true);
    reader.open();

    // every row names both sides, so there is no subset to hold and no answer at a trillion
    const size_t budget = Util::computeMemory(par.splitMemoryLimit);
    const size_t need = reader.getSize() * (sizeof(std::string) + 24);
    if (need > budget) {
        Debug(Debug::ERROR) << "Naming " << reader.getSize() << " sequences needs about "
                            << (need >> 30) << " GB of names and the limit is " << (budget >> 30)
                            << " GB. The cluster database and lin8-createrepseqfasta hold the same answer "
                            << "without a name for every sequence\n";
        EXIT(EXIT_FAILURE);
    }

    Timer timer;
    const unsigned int threads = std::max<unsigned int>(1, par.threads);
    std::vector<std::string> nameOfRank(reader.getSize());
    // finding a header is a memchr and stays in order; making a name out of it is what is spread
    const size_t NAME_BATCH = 1u << 16;
    std::vector<const char *> beginOf(NAME_BATCH, NULL);
    RunDbReader::HeaderStream headers(reader);
    Debug(Debug::INFO) << "Naming " << reader.getSize() << " sequences, one step a "
                       << NAME_BATCH << "\n";
    Debug::Progress nameProgress(reader.getSize() / NAME_BATCH + 1);
    const char *begin = NULL;
    size_t length = 0;
    uint64_t rank = 0;
    while (true) {
        size_t got = 0;
        while (got < NAME_BATCH && headers.next(begin, length)) {
            if (rank + got >= nameOfRank.size()) {
                Debug(Debug::ERROR) << "The headers hold more entries than the " << reader.getSize()
                                    << " the sequence locator names\n";
                EXIT(EXIT_FAILURE);
            }
            beginOf[got] = begin;
            got++;
        }
        if (got == 0) {
            break;
        }
        nameProgress.updateProgress();
#pragma omp parallel for schedule(static) num_threads(threads)
        for (size_t i = 0; i < got; i++) {
            // the same name createtsv would give, so the two can be compared
            nameOfRank[rank + i] = Util::parseFastaHeader(beginOf[i]);
        }
        rank += got;
    }
    if (rank != reader.getSize()) {
        Debug(Debug::ERROR) << "The headers hold " << rank << " entries and the sequence locator names "
                            << reader.getSize() << "\n";
        EXIT(EXIT_FAILURE);
    }
    Debug(Debug::INFO) << "Read " << rank << " names in " << timer.lap() << "\n";

    DBReader<DBKeyType> clusters(par.db2.c_str(), par.db2Index.c_str(), par.threads,
                                 DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA);
    clusters.open(DBReader<DBKeyType>::LINEAR_ACCCESS);

    // sized first so every thread writes its own stretch and the rows keep the clustering's order
    std::vector<size_t> edge(threads + 1, 0);
    for (unsigned int t = 0; t <= threads; t++) {
        edge[t] = clusters.getSize() * t / threads;
    }
    std::vector<uint64_t> bytesOf(threads, 0);
    std::vector<uint64_t> rowsOf(threads, 0);
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (unsigned int t = 0; t < threads; t++) {
        uint64_t sum = 0;
        uint64_t mine = 0;
        for (size_t i = edge[t]; i < edge[t + 1]; i++) {
            const uint64_t rep = clusters.getDbKey(i);
            if (rep >= nameOfRank.size()) {
                Debug(Debug::ERROR) << "The clustering names rank " << rep << ", past the database\n";
                EXIT(EXIT_FAILURE);
            }
            char *data = clusters.getData(i, t);
            while (data != NULL && *data != '\0') {
                const uint64_t member = strtoull(data, NULL, 10);
                if (member >= nameOfRank.size()) {
                    Debug(Debug::ERROR) << "The clustering names rank " << member
                                        << ", past the database\n";
                    EXIT(EXIT_FAILURE);
                }
                sum += nameOfRank[rep].size() + nameOfRank[member].size() + 2;
                mine++;
                data = Util::skipLine(data);
            }
        }
        bytesOf[t] = sum;
        rowsOf[t] = mine;
    }
    std::vector<uint64_t> startOf(threads + 1, 0);
    for (unsigned int t = 0; t < threads; t++) {
        startOf[t + 1] = startOf[t] + bytesOf[t];
    }

    const std::string tmp = par.db3 + ".tmp";
    const int out = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
    Debug(Debug::INFO) << "Writing " << (startOf[threads] >> 20) << " MB of rows in "
                       << threads << " parts\n";
    Debug::Progress writeProgress(threads);
    if (out < 0 || ftruncate(out, (off_t) startOf[threads]) != 0) {
        Debug(Debug::ERROR) << "Cannot make " << tmp << " of " << startOf[threads] << " byte\n";
        EXIT(EXIT_FAILURE);
    }
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (unsigned int t = 0; t < threads; t++) {
        uint64_t at = startOf[t];
        std::string line;
        line.reserve(1u << 20);
        for (size_t i = edge[t]; i < edge[t + 1]; i++) {
            const uint64_t rep = clusters.getDbKey(i);
            char *data = clusters.getData(i, t);
            while (data != NULL && *data != '\0') {
                const uint64_t member = strtoull(data, NULL, 10);
                line.append(nameOfRank[rep]);
                line.push_back('\t');
                line.append(nameOfRank[member]);
                line.push_back('\n');
                data = Util::skipLine(data);
                // on rows, because a cluster can be larger than the machine
                if (line.size() >= (1u << 20)) {
                    writeAt(out, tmp, line, at);
                }
            }
        }
        writeAt(out, tmp, line, at);
        if (at != startOf[t + 1]) {
            Debug(Debug::ERROR) << "Thread " << t << " wrote to " << at << " and was sized to "
                                << startOf[t + 1] << "\n";
            EXIT(EXIT_FAILURE);
        }
        writeProgress.updateProgress();
    }
    if (::close(out) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << tmp << "\n";
        EXIT(EXIT_FAILURE);
    }
    FileUtil::publishAtomically(tmp, par.db3);
    uint64_t rows = 0;
    for (unsigned int t = 0; t < threads; t++) {
        rows += rowsOf[t];
    }

    clusters.close();
    reader.close();
    Debug(Debug::INFO) << "Wrote " << rows << " rows in " << timer.lap() << "\n";
    return EXIT_SUCCESS;
}
