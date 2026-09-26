#include "Debug.h"
#include "FileUtil.h"
#include "Parameters.h"
#include "ZstdReader.h"
#include "ZstdWriter.h"
#include "Util.h"

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <mutex>
#include <utility>
#include <vector>

#ifdef OPENMP
#include <omp.h>
#endif

// a split is served by every manifest file whose name ends in .split<k>.tsv, with as many chunks
// contributing to it as the round that wrote them had
static std::vector<std::vector<std::string> > readManifest(const std::string &manifest, unsigned int splits) {
    std::ifstream file(manifest.c_str());
    if (file.fail()) {
        Debug(Debug::ERROR) << "Cannot open manifest " << manifest << "\n";
        EXIT(EXIT_FAILURE);
    }
    std::vector<std::vector<std::string> > shards(splits);
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const size_t tab = line.find('\t');
        const std::string path = (tab == std::string::npos) ? line : line.substr(0, tab);
        const size_t marker = path.rfind(".split");
        const size_t start = (marker == std::string::npos) ? std::string::npos : marker + 6;
        const size_t dot = (start == std::string::npos) ? std::string::npos : path.find('.', start);
        if (dot == std::string::npos || Util::isNumber(path.substr(start, dot - start)) == false) {
            Debug(Debug::ERROR) << "Cannot read a split number out of " << path << "\n";
            EXIT(EXIT_FAILURE);
        }
        const unsigned int split = (unsigned int) strtoul(path.substr(start, dot - start).c_str(), NULL, 10);
        if (split >= splits) {
            Debug(Debug::ERROR) << "Manifest " << manifest << " lists split " << split
                                << " but --tsv-splits is " << splits << "\n";
            EXIT(EXIT_FAILURE);
        }
        shards[split].push_back(path);
    }
    return shards;
}

static void splitLine(const std::string &line, std::string &first, std::string &second) {
    const size_t tab = line.find('\t');
    if (tab == std::string::npos) {
        Debug(Debug::ERROR) << "Expected two tab separated columns in \"" << line << "\"\n";
        EXIT(EXIT_FAILURE);
    }
    first.assign(line, 0, tab);
    second.assign(line, tab + 1, line.size() - tab - 1);
}

int propagateclusters(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);

    const unsigned int splits = (unsigned int) par.tsvSplits;
    if (splits < 1 || splits > Util::TSV_SPLIT_MAX) {
        Debug(Debug::ERROR) << "--tsv-splits must be between 1 and " << Util::TSV_SPLIT_MAX << "\n";
        EXIT(EXIT_FAILURE);
    }
    const std::string suffix = par.compressed ? ".zst" : "";

    std::vector<std::vector<std::string> > childShards = readManifest(par.db1, splits);
    std::vector<std::vector<std::string> > parentShards = readManifest(par.db2, splits);

    std::vector<ZstdWriter *> writers(splits);
    std::vector<std::mutex> locks(splits);
    std::string manifest;
    for (unsigned int k = 0; k < splits; k++) {
        const std::string name = Util::tsvSplitName(par.db3, k) + suffix;
        writers[k] = new ZstdWriter(name);
        manifest.append(name);
        manifest.append(1, '\n');
    }

    const size_t flushSize = std::max(Util::TSV_SPLIT_BUFFER_MIN,
                                      Util::TSV_SPLIT_BUFFER_TOTAL / ((size_t) par.threads * splits));
    Debug::Progress progress(splits);
    size_t joinedTotal = 0;
#pragma omp parallel reduction(+:joinedTotal)
    {
        std::vector<std::string> buffers(splits);
        // the parent names each representative once, so it is the smaller side and the one to hold;
        // the child carries a row per sequence and only ever streams past
        std::vector<std::pair<std::string, std::string> > parentByMember;
        std::string line, first, second;

#pragma omp for schedule(dynamic, 1)
        for (unsigned int b = 0; b < splits; b++) {
            progress.updateProgress();
            parentByMember.clear();
            for (size_t f = 0; f < parentShards[b].size(); f++) {
                ZstdReader parent(parentShards[b][f]);
                while (parent.getLine(line)) {
                    splitLine(line, first, second);
                    parentByMember.push_back(std::make_pair(second, first));
                }
            }
            std::sort(parentByMember.begin(), parentByMember.end());
            for (size_t i = 1; i < parentByMember.size(); i++) {
                if (parentByMember[i].first == parentByMember[i - 1].first) {
                    Debug(Debug::ERROR) << "The parent names " << parentByMember[i].first
                                        << " in more than one cluster\n";
                    EXIT(EXIT_FAILURE);
                }
            }

            size_t childRows = 0;
            size_t joinedRows = 0;
            for (size_t f = 0; f < childShards[b].size(); f++) {
                ZstdReader child(childShards[b][f]);
                while (child.getLine(line)) {
                    splitLine(line, first, second);
                    childRows++;
                    std::vector<std::pair<std::string, std::string> >::const_iterator it =
                        std::lower_bound(parentByMember.begin(), parentByMember.end(),
                                         std::make_pair(first, std::string()));
                    if (it == parentByMember.end() || it->first != first) {
                        continue;
                    }
                    const unsigned int k = Util::tsvSplitOfKey(it->second, splits);
                    buffers[k].append(it->second);
                    buffers[k].append(1, '\t');
                    buffers[k].append(second);
                    buffers[k].append(1, '\n');
                    joinedRows++;
                    if (buffers[k].size() >= flushSize) {
                        std::lock_guard<std::mutex> guard(locks[k]);
                        writers[k]->write(buffers[k]);
                        buffers[k].clear();
                    }
                }
            }

            // every child representative is named by exactly one parent row
            if (joinedRows != childRows) {
                Debug(Debug::ERROR) << "Split " << b << " lost cluster members: the child has "
                                    << childRows << " row(s), the join produced " << joinedRows << "\n";
                EXIT(EXIT_FAILURE);
            }
            joinedTotal += joinedRows;

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

    FILE *manifestFile = FileUtil::openAndDelete(par.db3.c_str(), "w");
    if (fwrite(manifest.c_str(), sizeof(char), manifest.size(), manifestFile) != manifest.size()
        || fclose(manifestFile) != 0) {
        Debug(Debug::ERROR) << "Cannot write " << par.db3 << "\n";
        EXIT(EXIT_FAILURE);
    }

    Debug(Debug::INFO) << "Propagated " << joinedTotal << " cluster member(s) into " << splits << " split(s)\n";
    return EXIT_SUCCESS;
}
