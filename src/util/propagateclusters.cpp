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

    // The parent of one split is held in memory. Measure each split's parent first, then run only
    // as many splits at once as the budget allows: re-reading the child in extra passes costs far
    // more than giving up parallelism, because the child is the larger side. Passes are the last
    // resort, for a single split that does not fit on its own.
    const size_t budget = Util::computeMemory(par.splitMemoryLimit);
    // A plain shard holds at least one byte per row, so its parent costs at most the file size plus
    // one pair per byte. When even that bound fits, the measuring pass is skipped: it reads the
    // parent a second time, which is not worth paying on a machine with room to spare.
    bool mustMeasure = false;
    size_t widestBound = 0;
    for (unsigned int b = 0; b < splits; b++) {
        size_t bound = 0;
        for (size_t f = 0; f < parentShards[b].size(); f++) {
            if (Util::endsWith(".zst", parentShards[b][f])) {
                mustMeasure = true;
                break;
            }
            bound += FileUtil::getFileSize(parentShards[b][f]);
        }
        widestBound = std::max(widestBound, bound * (1 + sizeof(std::pair<std::string, std::string>)));
    }

    // A live split also holds one output buffer per split, because any row can route to any of
    // them, so the buffers are part of what has to fit rather than a separate allowance.
    const size_t buffersPerSplit = (size_t) splits * Util::TSV_SPLIT_BUFFER_MIN;

    unsigned int concurrency = (unsigned int) par.threads;
    size_t widestParent = 0;
    if (mustMeasure || widestBound == 0 || budget / (widestBound + buffersPerSplit) < (size_t) par.threads) {
        std::vector<size_t> parentResident(splits, 0);
#pragma omp parallel for schedule(dynamic, 1)
        for (unsigned int b = 0; b < splits; b++) {
            size_t rows = 0;
            size_t bytes = 0;
            std::string probe;
            for (size_t f = 0; f < parentShards[b].size(); f++) {
                ZstdReader parent(parentShards[b][f]);
                while (parent.getLine(probe)) {
                    rows++;
                    bytes += probe.size();
                }
            }
            parentResident[b] = bytes + rows * sizeof(std::pair<std::string, std::string>);
        }
        for (unsigned int b = 0; b < splits; b++) {
            widestParent = std::max(widestParent, parentResident[b]);
        }
        const size_t perSplit = widestParent + buffersPerSplit;
        if (perSplit > 0 && budget / perSplit < concurrency) {
            concurrency = (unsigned int) std::max((size_t) 1, budget / perSplit);
        }
        Debug(Debug::INFO) << "Holding the parent of " << concurrency << " split(s) at once, "
                           << (widestParent / (1024 * 1024)) << " MB for the widest\n";
    }

    const size_t splitBudget = budget / concurrency;
    const size_t forBuffers = (splitBudget > widestParent) ? splitBudget - widestParent : 0;
    const size_t flushSize = std::max(Util::TSV_SPLIT_BUFFER_MIN,
                                      std::min(Util::TSV_SPLIT_BUFFER_TOTAL / ((size_t) concurrency * splits),
                                               forBuffers / splits));
    Debug::Progress progress(splits);
    size_t joinedTotal = 0;
#pragma omp parallel num_threads(concurrency) reduction(+:joinedTotal)
    {
        std::vector<std::string> buffers(splits);
        // the parent names each representative once, so it is the smaller side and the one to hold;
        // the child carries a row per sequence and only ever streams past
        std::vector<std::pair<std::string, std::string> > parentByMember;
        std::string line, first, second;

#pragma omp for schedule(dynamic, 1)
        for (unsigned int b = 0; b < splits; b++) {
            progress.updateProgress();
            size_t childRows = 0;
            size_t joinedRows = 0;

            // Util::hash is a different function from the one that assigned the split, so a pass
            // holds its share of the parent rather than all or none of it
            unsigned int passes = 1;
            for (unsigned int pass = 0; pass < passes; pass++) {
                parentByMember.clear();
                size_t resident = 0;
                bool overBudget = false;
                for (size_t f = 0; f < parentShards[b].size() && overBudget == false; f++) {
                    ZstdReader parent(parentShards[b][f]);
                    while (parent.getLine(line)) {
                        splitLine(line, first, second);
                        if (passes > 1 && Util::hash(second.c_str(), second.size()) % passes != pass) {
                            continue;
                        }
                        parentByMember.push_back(std::make_pair(second, first));
                        resident += first.size() + second.size() + sizeof(std::pair<std::string, std::string>);
                        if (resident > splitBudget) {
                            overBudget = true;
                            break;
                        }
                    }
                }
                if (overBudget) {
                    if (passes > splits * 1024) {
                        Debug(Debug::ERROR) << "Split " << b << " does not fit --split-memory-limit "
                                            << "even taken in " << passes << " passes\n";
                        EXIT(EXIT_FAILURE);
                    }
                    passes *= 2;
                    pass = (unsigned int) -1;
                    childRows = 0;
                    joinedRows = 0;
                    continue;
                }

                std::sort(parentByMember.begin(), parentByMember.end());
                for (size_t i = 1; i < parentByMember.size(); i++) {
                    if (parentByMember[i].first == parentByMember[i - 1].first) {
                        Debug(Debug::ERROR) << "The parent names " << parentByMember[i].first
                                            << " in more than one cluster\n";
                        EXIT(EXIT_FAILURE);
                    }
                }

                for (size_t f = 0; f < childShards[b].size(); f++) {
                    ZstdReader child(childShards[b][f]);
                    while (child.getLine(line)) {
                        splitLine(line, first, second);
                        if (pass == 0) {
                            childRows++;
                        }
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
