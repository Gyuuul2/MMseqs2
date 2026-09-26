#include "Debug.h"
#include "FileUtil.h"
#include "Parameters.h"
#include "ZstdReader.h"
#include "ZstdWriter.h"
#include "Util.h"

#include <cstdio>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#ifdef OPENMP
#include <omp.h>
#endif

static std::vector<std::string> readManifest(const std::string &manifest) {
    std::ifstream file(manifest.c_str());
    if (file.fail()) {
        Debug(Debug::ERROR) << "Cannot open manifest " << manifest << "\n";
        EXIT(EXIT_FAILURE);
    }
    std::vector<std::string> shards;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const size_t tab = line.find('\t');
        shards.push_back((tab == std::string::npos) ? line : line.substr(0, tab));
    }
    if (shards.empty()) {
        Debug(Debug::ERROR) << "Manifest " << manifest << " lists no shard\n";
        EXIT(EXIT_FAILURE);
    }
    return shards;
}

int splittsv(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);

    const unsigned int splits = (unsigned int) par.tsvSplits;
    if (splits < 1 || splits > Util::TSV_SPLIT_MAX) {
        Debug(Debug::ERROR) << "--tsv-splits must be between 1 and " << Util::TSV_SPLIT_MAX << "\n";
        EXIT(EXIT_FAILURE);
    }
    const std::string suffix = par.compressed ? ".zst" : "";
    std::vector<std::string> shards = readManifest(par.db1);

    std::vector<ZstdWriter *> writers(splits);
    std::vector<std::mutex> locks(splits);
    std::string manifest;
    for (unsigned int k = 0; k < splits; k++) {
        const std::string name = Util::tsvSplitName(par.db2, k) + suffix;
        writers[k] = new ZstdWriter(name);
        manifest.append(name);
        manifest.append(1, '\n');
    }

    const size_t flushSize = std::max(Util::TSV_SPLIT_BUFFER_MIN,
                                      Util::TSV_SPLIT_BUFFER_TOTAL / ((size_t) par.threads * splits));
    Debug::Progress progress(shards.size());
    size_t routedTotal = 0;
#pragma omp parallel reduction(+:routedTotal)
    {
        std::vector<std::string> buffers(splits);
        std::string line;

#pragma omp for schedule(dynamic, 1)
        for (size_t f = 0; f < shards.size(); f++) {
            progress.updateProgress();
            ZstdReader reader(shards[f]);
            while (reader.getLine(line)) {
                const unsigned int k = Util::tsvSplitOfColumn(line, par.tsvSplitColumn, splits);
                buffers[k].append(line);
                buffers[k].append(1, '\n');
                routedTotal++;
                if (buffers[k].size() >= flushSize) {
                    std::lock_guard<std::mutex> guard(locks[k]);
                    writers[k]->write(buffers[k]);
                    buffers[k].clear();
                }
            }
            for (unsigned int k = 0; k < splits; k++) {
                if (buffers[k].empty() == false) {
                    std::lock_guard<std::mutex> guard(locks[k]);
                    writers[k]->write(buffers[k]);
                    buffers[k].clear();
                }
            }
        }
    }

    for (unsigned int k = 0; k < splits; k++) {
        writers[k]->close();
        delete writers[k];
    }

    FILE *manifestFile = FileUtil::openAndDelete(par.db2.c_str(), "w");
    if (fwrite(manifest.c_str(), sizeof(char), manifest.size(), manifestFile) != manifest.size()
        || fclose(manifestFile) != 0) {
        Debug(Debug::ERROR) << "Cannot write " << par.db2 << "\n";
        EXIT(EXIT_FAILURE);
    }

    Debug(Debug::INFO) << "Routed " << routedTotal << " row(s) into " << splits << " split(s)\n";
    return EXIT_SUCCESS;
}
