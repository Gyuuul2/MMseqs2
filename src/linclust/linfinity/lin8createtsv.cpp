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

int lin8createtsv(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);

    RunDbReader reader(par.db1, true);
    reader.open();

    // Every row names both sides, so every rank's name is needed and there is no subset to hold.
    // An empty string is thirty two byte before a character of it: at a trillion sequences that is
    // thirty two terabyte, and the tsv it would write is forty. Say so rather than dying in the
    // allocator, and name the two commands that do work at that size.
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
    std::vector<std::string> nameOfRank(reader.getSize());
    RunDbReader::HeaderStream headers(reader);
    const char *begin = NULL;
    size_t length = 0;
    uint64_t rank = 0;
    while (headers.next(begin, length)) {
        if (rank >= nameOfRank.size()) {
            Debug(Debug::ERROR) << "The headers hold more entries than the " << reader.getSize()
                                << " the run table names\n";
            EXIT(EXIT_FAILURE);
        }
        // the same name createtsv would give, so the two can be compared
        nameOfRank[rank] = Util::parseFastaHeader(begin);
        rank++;
    }
    if (rank != reader.getSize()) {
        Debug(Debug::ERROR) << "The headers hold " << rank << " entries and the run table names "
                            << reader.getSize() << "\n";
        EXIT(EXIT_FAILURE);
    }
    Debug(Debug::INFO) << "Read " << rank << " names in " << timer.lap() << "\n";

    DBReader<DBKeyType> clusters(par.db2.c_str(), par.db2Index.c_str(), par.threads,
                                 DBReader<DBKeyType>::USE_INDEX | DBReader<DBKeyType>::USE_DATA);
    clusters.open(DBReader<DBKeyType>::LINEAR_ACCCESS);

    const std::string tmp = par.db3 + ".tmp";
    FILE *out = FileUtil::openAndDelete(tmp.c_str(), "w");
    std::string line;
    line.reserve(1u << 20);
    uint64_t rows = 0;
    for (size_t i = 0; i < clusters.getSize(); i++) {
        const uint64_t rep = clusters.getDbKey(i);
        if (rep >= nameOfRank.size()) {
            Debug(Debug::ERROR) << "The clustering names rank " << rep << ", past the database\n";
            EXIT(EXIT_FAILURE);
        }
        char *data = clusters.getData(i, 0);
        while (data != NULL && *data != '\0') {
            const uint64_t member = strtoull(data, NULL, 10);
            if (member >= nameOfRank.size()) {
                Debug(Debug::ERROR) << "The clustering names rank " << member
                                    << ", past the database\n";
                EXIT(EXIT_FAILURE);
            }
            line.append(nameOfRank[rep]);
            line.push_back('\t');
            line.append(nameOfRank[member]);
            line.push_back('\n');
            rows++;
            data = Util::skipLine(data);
            // a cluster can be larger than the machine, so the buffer empties on rows and not on
            // clusters, which is where it used to
            if (line.size() >= (1u << 20)) {
                if (fwrite(line.c_str(), 1, line.size(), out) != line.size()) {
                    Debug(Debug::ERROR) << "Cannot write " << tmp << "\n";
                    EXIT(EXIT_FAILURE);
                }
                line.clear();
            }
        }
    }
    if (line.empty() == false && fwrite(line.c_str(), 1, line.size(), out) != line.size()) {
        Debug(Debug::ERROR) << "Cannot write " << tmp << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (fclose(out) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << tmp << "\n";
        EXIT(EXIT_FAILURE);
    }
    FileUtil::publishAtomically(tmp, par.db3);

    clusters.close();
    reader.close();
    Debug(Debug::INFO) << "Wrote " << rows << " rows in " << timer.lap() << "\n";
    return EXIT_SUCCESS;
}
