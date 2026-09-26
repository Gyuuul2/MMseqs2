#include "Debug.h"
#include "FileUtil.h"
#include "Parameters.h"
#include "Util.h"

#include <zstd.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifdef OPENMP
#include <omp.h>
#endif

// The shards the batch workflow writes are optionally zstd compressed, so read and write both here
// rather than making the workflow materialise a plain copy of every shard first.
namespace {

class ShardReader {
public:
    ShardReader(const std::string &fileName) : name(fileName), dstream(NULL), pos(0), eof(false) {
        file = FileUtil::openFileOrDie(fileName.c_str(), "rb", true);
        if (Util::endsWith(".zst", fileName)) {
            dstream = ZSTD_createDStream();
            if (dstream == NULL || ZSTD_isError(ZSTD_initDStream(dstream))) {
                Debug(Debug::ERROR) << "Cannot start zstd decompression of " << fileName << "\n";
                EXIT(EXIT_FAILURE);
            }
            in.resize(ZSTD_DStreamInSize());
            input.src = in.data();
            input.size = 0;
            input.pos = 0;
        }
        out.resize(ZSTD_DStreamOutSize());
    }

    ~ShardReader() {
        if (dstream != NULL) {
            ZSTD_freeDStream(dstream);
        }
        fclose(file);
    }

    bool getLine(std::string &line) {
        line.clear();
        while (true) {
            while (pos < buffer.size()) {
                const char c = buffer[pos++];
                if (c == '\n') {
                    if (line.empty() == false && line[line.size() - 1] == '\r') {
                        line.erase(line.size() - 1);
                    }
                    return true;
                }
                line.append(1, c);
            }
            buffer.clear();
            pos = 0;
            if (fill() == false) {
                return line.empty() == false;
            }
        }
    }

private:
    bool fill() {
        if (dstream == NULL) {
            const size_t n = fread(&out[0], 1, out.size(), file);
            if (n == 0) {
                return false;
            }
            buffer.assign(&out[0], n);
            return true;
        }
        while (true) {
            if (input.pos == input.size) {
                const size_t n = fread(&in[0], 1, in.size(), file);
                if (n == 0) {
                    return false;
                }
                input.src = in.data();
                input.size = n;
                input.pos = 0;
            }
            ZSTD_outBuffer output = { &out[0], out.size(), 0 };
            const size_t code = ZSTD_decompressStream(dstream, &output, &input);
            if (ZSTD_isError(code)) {
                Debug(Debug::ERROR) << "Cannot decompress " << name << ": " << ZSTD_getErrorName(code) << "\n";
                EXIT(EXIT_FAILURE);
            }
            if (output.pos > 0) {
                buffer.assign(&out[0], output.pos);
                return true;
            }
        }
    }

    std::string name;
    FILE *file;
    ZSTD_DStream *dstream;
    ZSTD_inBuffer input;
    std::vector<char> in;
    std::vector<char> out;
    std::string buffer;
    size_t pos;
    bool eof;
};

class ShardWriter {
public:
    ShardWriter(const std::string &fileName) : name(fileName), cstream(NULL) {
        file = FileUtil::openAndDelete(fileName.c_str(), "wb");
        if (Util::endsWith(".zst", fileName)) {
            cstream = ZSTD_createCStream();
            if (cstream == NULL || ZSTD_isError(ZSTD_initCStream(cstream, 3))) {
                Debug(Debug::ERROR) << "Cannot start zstd compression of " << fileName << "\n";
                EXIT(EXIT_FAILURE);
            }
            out.resize(ZSTD_CStreamOutSize());
        }
    }

    void write(const std::string &data) {
        if (cstream == NULL) {
            if (fwrite(data.c_str(), sizeof(char), data.size(), file) != data.size()) {
                Debug(Debug::ERROR) << "Cannot write " << name << "\n";
                EXIT(EXIT_FAILURE);
            }
            return;
        }
        ZSTD_inBuffer input = { data.c_str(), data.size(), 0 };
        while (input.pos < input.size) {
            ZSTD_outBuffer output = { &out[0], out.size(), 0 };
            const size_t code = ZSTD_compressStream(cstream, &output, &input);
            if (ZSTD_isError(code)) {
                Debug(Debug::ERROR) << "Cannot compress " << name << ": " << ZSTD_getErrorName(code) << "\n";
                EXIT(EXIT_FAILURE);
            }
            flush(output.pos);
        }
    }

    void close() {
        if (cstream != NULL) {
            ZSTD_outBuffer output = { &out[0], out.size(), 0 };
            const size_t code = ZSTD_endStream(cstream, &output);
            if (ZSTD_isError(code) || code != 0) {
                Debug(Debug::ERROR) << "Cannot finish " << name << "\n";
                EXIT(EXIT_FAILURE);
            }
            flush(output.pos);
            ZSTD_freeCStream(cstream);
            cstream = NULL;
        }
        if (fclose(file) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << name << "\n";
            EXIT(EXIT_FAILURE);
        }
    }

private:
    void flush(size_t n) {
        if (n > 0 && fwrite(&out[0], sizeof(char), n, file) != n) {
            Debug(Debug::ERROR) << "Cannot write " << name << "\n";
            EXIT(EXIT_FAILURE);
        }
    }

    std::string name;
    FILE *file;
    ZSTD_CStream *cstream;
    std::vector<char> out;
};

}

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

    std::vector<ShardWriter *> writers(splits);
    std::vector<std::mutex> locks(splits);
    std::string manifest;
    for (unsigned int k = 0; k < splits; k++) {
        const std::string name = Util::tsvSplitName(par.db3, k) + suffix;
        writers[k] = new ShardWriter(name);
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
        std::unordered_map<std::string, std::vector<std::string> > childByRep;
        std::string line, first, second;

#pragma omp for schedule(dynamic, 1)
        for (unsigned int b = 0; b < splits; b++) {
            progress.updateProgress();
            childByRep.clear();
            size_t childRows = 0;
            for (size_t f = 0; f < childShards[b].size(); f++) {
                ShardReader child(childShards[b][f]);
                while (child.getLine(line)) {
                    splitLine(line, first, second);
                    childByRep[first].push_back(second);
                    childRows++;
                }
            }

            size_t joinedRows = 0;
            for (size_t f = 0; f < parentShards[b].size(); f++) {
                ShardReader parent(parentShards[b][f]);
                while (parent.getLine(line)) {
                    splitLine(line, first, second);
                    std::unordered_map<std::string, std::vector<std::string> >::const_iterator it = childByRep.find(second);
                    if (it == childByRep.end()) {
                        continue;
                    }
                    const unsigned int k = Util::tsvSplitOfKey(first, splits);
                    for (size_t i = 0; i < it->second.size(); i++) {
                        buffers[k].append(first);
                        buffers[k].append(1, '\t');
                        buffers[k].append(it->second[i]);
                        buffers[k].append(1, '\n');
                    }
                    joinedRows += it->second.size();
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
