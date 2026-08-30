#include "Lin8Db.h"
#include "Parameters.h"
#include "DBReader.h"
#include "DBWriter.h"
#include "Debug.h"
#include "FileUtil.h"
#include "Util.h"
#include "KSeqWrapper.h"
#include "NodePlacement.h"
#include "Timer.h"
#include "FastSort.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

#ifdef OPENMP
#include <omp.h>
#endif

static const unsigned int LINCLUSTERDB_MAX_SEQ_LEN = RunTable::MAX_ENTRY_LEN;
static const size_t LINCLUSTERDB_HISTOGRAM_SIZE = LINCLUSTERDB_MAX_SEQ_LEN + 1;
static const uint64_t LINCLUSTERDB_HISTOGRAM_MAGIC = 0x4C494E4348495354ull;


// a step of the placing bar, so it moves on a minute rather than on a whole input file
static const uint64_t PROGRESS_STEP = 1000000;

static std::vector<InputSplit> inputSplitsForNode(const std::vector<std::string> &filenames,
                                             const NodePlacement &placement, unsigned int threads) {
    const size_t want = std::max<size_t>(1, (size_t) placement.count * std::max(threads, 1u) * 4);
    const std::vector<InputSplit> all = planInputSplits(filenames, want);
    std::vector<uint64_t> upTo(all.size() + 1, 0);
    for (size_t i = 0; i < all.size(); i++) {
        upTo[i + 1] = upTo[i] + std::max<uint64_t>(all[i].bytes(), 1);
    }
    const uint64_t total = upTo.back();
    std::vector<InputSplit> nodeInputSplits;
    for (size_t i = 0; i < all.size(); i++) {
        const uint64_t middle = (upTo[i] + upTo[i + 1]) / 2;
        const unsigned int owner = static_cast<unsigned int>(std::min<uint64_t>(
            total == 0 ? 0 : middle * placement.count / total, placement.count - 1));
        if (owner == placement.index) {
            nodeInputSplits.push_back(all[i]);
        }
    }
    return nodeInputSplits;
}


static std::string nodePartName(const std::string &db, const char *suffix, unsigned int node) {
    return db + "." + suffix + "." + SSTR(node);
}

struct LocalCounts {
    std::vector<uint64_t> histogram;
    std::vector<uint64_t> perInputSplitCount;
    std::vector<uint64_t> perInputSplitHeaderBytes;
    unsigned int inputSplitCount;
    uint64_t sequences;
    uint64_t residues;
    uint64_t headerBytes;
    uint64_t rejected;

    LocalCounts() : inputSplitCount(0), sequences(0), residues(0), headerBytes(0), rejected(0) {}

    uint64_t &at(std::vector<uint64_t> &of, size_t length, unsigned int inputSplit) {
        return of[length * inputSplitCount + inputSplit];
    }
    uint64_t get(const std::vector<uint64_t> &of, size_t length, unsigned int inputSplit) const {
        return of[length * inputSplitCount + inputSplit];
    }
};

static LocalCounts countSequenceLengths(const std::vector<std::string> &filenames,
                                   const std::vector<InputSplit> &nodeInputSplits, unsigned int threads,
                                   uint32_t maxSeqLen) {
    LocalCounts nodeCounts;
    nodeCounts.inputSplitCount = static_cast<unsigned int>(std::max<size_t>(nodeInputSplits.size(), 1));
    nodeCounts.histogram.assign(LINCLUSTERDB_HISTOGRAM_SIZE, 0);
    nodeCounts.perInputSplitCount.assign(LINCLUSTERDB_HISTOGRAM_SIZE * nodeCounts.inputSplitCount, 0);
    nodeCounts.perInputSplitHeaderBytes.assign(LINCLUSTERDB_HISTOGRAM_SIZE * nodeCounts.inputSplitCount, 0);
    std::vector<std::pair<std::string, size_t> > skippedAll;
    std::vector<std::pair<std::string, size_t> > aloneAll;
    Debug::Progress progress(nodeInputSplits.size());
#pragma omp parallel num_threads(threads)
    {
        uint64_t sequences = 0;
        uint64_t residues = 0;
        uint64_t headerBytes = 0;
        uint64_t rejected = 0;
        std::vector<std::pair<std::string, size_t> > skipped;
        std::vector<std::pair<std::string, size_t> > alone;
        // one file is one counting group, so dynamic scheduling cannot change any count
#pragma omp for schedule(dynamic, 1) nowait
        for (size_t at = 0; at < nodeInputSplits.size(); at++) {
            const unsigned int inputSplit = static_cast<unsigned int>(at);
            InputSplitReader reader(filenames[nodeInputSplits[at].file], nodeInputSplits[at]);
            const char *name = NULL;
            const char *seq = NULL;
            size_t nameLength = 0;
            size_t length = 0;
            while (reader.next(name, nameLength, seq, length)) {
                if (length == 0 || length > maxSeqLen) {
                    rejected++;
                    skipped.push_back(std::make_pair(std::string(name, nameLength), length));
                    continue;
                }
                if (length > RunTable::MAX_SEQ_LEN) {
                    alone.push_back(std::make_pair(std::string(name, nameLength), length));
                }
                const size_t header = nameLength + 1;
                nodeCounts.at(nodeCounts.perInputSplitCount, length, inputSplit)++;
                nodeCounts.at(nodeCounts.perInputSplitHeaderBytes, length, inputSplit) += header;
                sequences++;
                residues += length;
                headerBytes += header;
            }
            progress.updateProgress();
        }
#pragma omp critical
        {
            nodeCounts.sequences += sequences;
            nodeCounts.residues += residues;
            nodeCounts.headerBytes += headerBytes;
            nodeCounts.rejected += rejected;
            skippedAll.insert(skippedAll.end(), skipped.begin(), skipped.end());
            aloneAll.insert(aloneAll.end(), alone.begin(), alone.end());
        }
    }
    std::sort(aloneAll.begin(), aloneAll.end());
    std::sort(skippedAll.begin(), skippedAll.end());
    for (size_t i = 0; i < aloneAll.size(); i++) {
        Debug(Debug::WARNING) << "Too long to take k-mers from, clustered alone: "
                              << aloneAll[i].first << " (" << aloneAll[i].second << " residues)\n";
    }
    for (size_t i = 0; i < skippedAll.size(); i++) {
        Debug(Debug::ERROR) << "Outside 1 to " << maxSeqLen << " residues, left out of the database: "
                            << skippedAll[i].first << " (" << skippedAll[i].second << " residues)\n";
    }
    for (size_t length = 0; length < LINCLUSTERDB_HISTOGRAM_SIZE; length++) {
        for (unsigned int inputSplit = 0; inputSplit < nodeCounts.inputSplitCount; inputSplit++) {
            nodeCounts.histogram[length] += nodeCounts.get(nodeCounts.perInputSplitCount, length, inputSplit);
        }
    }
    return nodeCounts;
}

static void writeHistogram(const std::string &path, const LocalCounts &nodeCounts, unsigned int threads) {
    const std::string tmp = path + ".tmp" + uniqueTmpSuffix();
    FILE *out = FileUtil::openAndDelete(tmp.c_str(), "wb");
    const uint64_t fields[6] = {LINCLUSTERDB_HISTOGRAM_MAGIC, threads, nodeCounts.sequences,
                                nodeCounts.residues, nodeCounts.headerBytes, nodeCounts.rejected};
    if (fwrite(fields, sizeof(uint64_t), 6, out) != 6
        || fwrite(nodeCounts.histogram.data(), sizeof(uint64_t), LINCLUSTERDB_HISTOGRAM_SIZE, out)
               != LINCLUSTERDB_HISTOGRAM_SIZE) {
        Debug(Debug::ERROR) << "Cannot write the length histogram to " << path << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (fclose(out) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << tmp << "\n";
        EXIT(EXIT_FAILURE);
    }
    FileUtil::publishAtomically(tmp, path);
}

static LocalCounts readHistogram(const std::string &path, unsigned int threads) {
    FILE *in = fopen(path.c_str(), "rb");
    if (in == NULL) {
        Debug(Debug::ERROR) << "Cannot open the length histogram " << path << "\n";
        EXIT(EXIT_FAILURE);
    }
    LocalCounts nodeCounts;
    nodeCounts.inputSplitCount = 0;
    uint64_t fields[6];
    nodeCounts.histogram.assign(LINCLUSTERDB_HISTOGRAM_SIZE, 0);
    if (fread(fields, sizeof(uint64_t), 6, in) != 6
        || fread(nodeCounts.histogram.data(), sizeof(uint64_t), LINCLUSTERDB_HISTOGRAM_SIZE, in)
               != LINCLUSTERDB_HISTOGRAM_SIZE) {
        Debug(Debug::ERROR) << "Length histogram " << path << " is truncated\n";
        EXIT(EXIT_FAILURE);
    }
    fclose(in);
    if (fields[0] != LINCLUSTERDB_HISTOGRAM_MAGIC) {
        Debug(Debug::ERROR) << "File " << path << " is not a length histogram\n";
        EXIT(EXIT_FAILURE);
    }
    if (fields[1] != threads) {
        Debug(Debug::ERROR) << "Histogram " << path << " was written with " << fields[1]
                            << " threads, this node uses " << threads
                            << ", every node must use the same value\n";
        EXIT(EXIT_FAILURE);
    }
    nodeCounts.sequences = fields[2];
    nodeCounts.residues = fields[3];
    nodeCounts.headerBytes = fields[4];
    nodeCounts.rejected = fields[5];
    return nodeCounts;
}

// the ranks every node agrees on: rank ordered by length descending, then node, then slot
struct RankLayout {
    std::vector<uint64_t> rankBase;
    uint64_t sequences;
    uint64_t bytes;
    uint64_t headerBytes;
};

static RankLayout computeGlobalRankLayout(const std::vector<LocalCounts> &everyNodeCounts, unsigned int self) {
    RankLayout ranks;
    ranks.rankBase.assign(LINCLUSTERDB_HISTOGRAM_SIZE, 0);
    ranks.sequences = 0;
    ranks.bytes = 0;
    ranks.headerBytes = 0;
    uint64_t rank = 0;
    for (size_t length = LINCLUSTERDB_MAX_SEQ_LEN; length >= 1; length--) {
        for (size_t node = 0; node < everyNodeCounts.size(); node++) {
            if (node == self) {
                ranks.rankBase[length] = rank;
            }
            rank += everyNodeCounts[node].histogram[length];
        }
    }
    ranks.sequences = rank;
    for (size_t node = 0; node < everyNodeCounts.size(); node++) {
        ranks.headerBytes += everyNodeCounts[node].headerBytes;
    }
    for (size_t length = 1; length < LINCLUSTERDB_HISTOGRAM_SIZE; length++) {
        uint64_t total = 0;
        for (size_t node = 0; node < everyNodeCounts.size(); node++) {
            total += everyNodeCounts[node].histogram[length];
        }
        ranks.bytes += total * length;
    }
    return ranks;
}

struct DbLayout {
    std::vector<uint64_t> seqAt;
    std::vector<uint64_t> hdrAt;
    std::vector<uint64_t> seqFileStart;
    std::vector<uint64_t> hdrFileStart;
    unsigned int inputSplitCount;
    uint64_t seqBytes;
    uint64_t hdrBytes;
};

static void planDatabaseLayout(const std::vector<LocalCounts> &everyNodeCounts, const LocalCounts &nodeCounts,
                       const RankLayout &ranks, unsigned int files, unsigned int fileBase,
                       DbLayout &layout, RunTable &partial) {
    std::vector<uint64_t> bytesAbove(LINCLUSTERDB_HISTOGRAM_SIZE, 0);
    uint64_t total = 0;
    for (size_t length = LINCLUSTERDB_MAX_SEQ_LEN; length >= 1; length--) {
        bytesAbove[length] = total;
        uint64_t count = 0;
        for (size_t node = 0; node < everyNodeCounts.size(); node++) {
            count += everyNodeCounts[node].histogram[length];
        }
        total += count * length;
    }
    const uint64_t perFile = std::max<uint64_t>(1, (total + files - 1) / files);

    const unsigned int inputSplitCount = nodeCounts.inputSplitCount;
    layout.inputSplitCount = inputSplitCount;
    layout.seqAt.assign(LINCLUSTERDB_HISTOGRAM_SIZE * inputSplitCount, 0);
    layout.hdrAt.assign(LINCLUSTERDB_HISTOGRAM_SIZE * inputSplitCount, 0);
    layout.seqFileStart.assign(files + 1, 0);
    layout.hdrFileStart.assign(files + 1, 0);

    uint64_t seqAt = 0;
    uint64_t hdrAt = 0;
    unsigned int open = 0;
    for (size_t length = LINCLUSTERDB_MAX_SEQ_LEN; length >= 1; length--) {
        const unsigned int file = static_cast<unsigned int>(
            std::min<uint64_t>(bytesAbove[length] / perFile, files - 1));
        while (open < file) {
            open++;
            layout.seqFileStart[open] = seqAt;
            layout.hdrFileStart[open] = hdrAt;
        }
        const uint64_t seqFirst = seqAt;
        const uint64_t hdrFirst = hdrAt;
        for (unsigned int inputSplit = 0; inputSplit < inputSplitCount; inputSplit++) {
            layout.seqAt[length * inputSplitCount + inputSplit] = seqAt;
            layout.hdrAt[length * inputSplitCount + inputSplit] = hdrAt;
            seqAt += nodeCounts.get(nodeCounts.perInputSplitCount, length, inputSplit)
                     * length;
            hdrAt += nodeCounts.get(nodeCounts.perInputSplitHeaderBytes, length, inputSplit);
        }
        if (nodeCounts.histogram[length] > 0) {
            partial.append(ranks.rankBase[length], static_cast<uint32_t>(length),
                           seqFirst - layout.seqFileStart[file], fileBase + file,
                           hdrFirst - layout.hdrFileStart[file]);
        }
    }
    while (open < files) {
        open++;
        layout.seqFileStart[open] = seqAt;
        layout.hdrFileStart[open] = hdrAt;
    }
    layout.seqBytes = seqAt;
    layout.hdrBytes = hdrAt;
}

class SequenceWriter {
public:
    static const size_t WRITE_CHUNK = 1ull << 30;

    SequenceWriter(const std::vector<int> &seqFd, const std::vector<int> &hdrFd,
               const LocalCounts &nodeCounts, const DbLayout &layout, unsigned int file, size_t budget)
        : seqFd(seqFd), hdrFd(hdrFd), layout(layout), file(file) {
        seqBuf.resize(LINCLUSTERDB_HISTOGRAM_SIZE);
        hdrBuf.resize(LINCLUSTERDB_HISTOGRAM_SIZE);
        seqRoom.assign(LINCLUSTERDB_HISTOGRAM_SIZE, 0);
        hdrRoom.assign(LINCLUSTERDB_HISTOGRAM_SIZE, 0);
        seqAt.assign(LINCLUSTERDB_HISTOGRAM_SIZE, 0);
        hdrAt.assign(LINCLUSTERDB_HISTOGRAM_SIZE, 0);
        uint64_t want = 0;
        for (size_t length = 1; length < LINCLUSTERDB_HISTOGRAM_SIZE; length++) {
            seqAt[length] = layout.seqAt[length * layout.inputSplitCount + file];
            hdrAt[length] = layout.hdrAt[length * layout.inputSplitCount + file];
            seqRoom[length] = nodeCounts.get(nodeCounts.perInputSplitCount, length, file)
                              * length;
            hdrRoom[length] = nodeCounts.get(nodeCounts.perInputSplitHeaderBytes, length, file);
            want += seqRoom[length] + hdrRoom[length];
        }
        if (want > budget && want > 0) {
            // a floor a length, or a rare one rounds to nothing and writes a sequence at a time
            const uint64_t floorRoom = (budget / 2) / LINCLUSTERDB_HISTOGRAM_SIZE;
            for (size_t length = 1; length < LINCLUSTERDB_HISTOGRAM_SIZE; length++) {
                const uint64_t share = budget / 2;
                seqRoom[length] = std::min<uint64_t>(
                    seqRoom[length], std::max<uint64_t>(seqRoom[length] * share / want, floorRoom));
                hdrRoom[length] = std::min<uint64_t>(
                    hdrRoom[length], std::max<uint64_t>(hdrRoom[length] * share / want, floorRoom));
            }
        }
    }

    void add(size_t length, const unsigned char *sequence, const char *header, size_t headerLen) {
        seqBuf[length].insert(seqBuf[length].end(), sequence, sequence + length);
        hdrBuf[length].insert(hdrBuf[length].end(), header, header + headerLen);
        if (seqBuf[length].size() >= seqRoom[length] || hdrBuf[length].size() >= hdrRoom[length]) {
            flushLength(length);
        }
    }

    void flush() {
        for (size_t length = LINCLUSTERDB_MAX_SEQ_LEN; length >= 1; length--) {
            flushLength(length);
        }
    }

    uint64_t bytesWritten() const { return written; }

private:
    void flushLength(size_t length) {
        if (seqBuf[length].empty() == false) {
            writeSpanning(seqFd, layout.seqFileStart, seqAt[length], seqBuf[length]);
            seqAt[length] += seqBuf[length].size();
            seqBuf[length].clear();
        }
        if (hdrBuf[length].empty() == false) {
            writeSpanning(hdrFd, layout.hdrFileStart, hdrAt[length], hdrBuf[length]);
            hdrAt[length] += hdrBuf[length].size();
            hdrBuf[length].clear();
        }
    }

    // the outputs are sized up front, so an offset mistake would leave zeros that read as valid
    void writeSpanning(const std::vector<int> &fd, const std::vector<uint64_t> &fileStart,
                       uint64_t at, const std::vector<unsigned char> &data) {
        size_t done = 0;
        while (done < data.size()) {
            const uint64_t now = at + done;
            size_t which = 0;
            while (which + 1 < fd.size() && fileStart[which + 1] <= now) {
                which++;
            }
            // linux truncates a single pwrite past 2 GiB, and nfs may shorten one at any size
            const size_t room = std::min<size_t>(data.size() - done,
                                                 static_cast<size_t>(fileStart[which + 1] - now));
            const size_t chunk = std::min<size_t>(room, WRITE_CHUNK);
            if (chunk == 0) {
                Debug(Debug::ERROR) << "Write of " << data.size() << " byte at " << at
                                    << " ran past the last data file\n";
                EXIT(EXIT_FAILURE);
            }
            const ssize_t wrote = pwrite(fd[which], data.data() + done, chunk,
                                         static_cast<off_t>(now - fileStart[which]));
            if (wrote <= 0) {
                Debug(Debug::ERROR) << "Cannot write " << chunk << " byte to data file " << which
                                    << "\n";
                EXIT(EXIT_FAILURE);
            }
            done += static_cast<size_t>(wrote);
        }
        written += data.size();
    }

    const std::vector<int> &seqFd;
    const std::vector<int> &hdrFd;
    const DbLayout &layout;
    unsigned int file;
    uint64_t written = 0;
    std::vector<std::vector<unsigned char> > seqBuf;
    std::vector<std::vector<unsigned char> > hdrBuf;
    std::vector<uint64_t> seqRoom;
    std::vector<uint64_t> hdrRoom;
    std::vector<uint64_t> seqAt;
    std::vector<uint64_t> hdrAt;
};


static void writeSequencesAndHeaders(const std::vector<std::string> &filenames, const std::vector<InputSplit> &nodeInputSplits,
                      const LocalCounts &nodeCounts, const DbLayout &layout,
                      const std::vector<int> &seqFd, const std::vector<int> &hdrFd,
                      unsigned int threads, size_t budget, uint32_t maxSeqLen) {
    uint64_t written = 0;
    // one step a million sequences, because there are far fewer splits than there are minutes
    Debug::Progress progress(nodeCounts.sequences / PROGRESS_STEP + 1);
#pragma omp parallel num_threads(threads)
    {
        std::string header;
#pragma omp for schedule(dynamic, 1)
        for (size_t at = 0; at < nodeInputSplits.size(); at++) {
            SequenceWriter writer(seqFd, hdrFd, nodeCounts, layout, static_cast<unsigned int>(at),
                              std::max<size_t>(budget / threads, 1u << 20));
            InputSplitReader reader(filenames[nodeInputSplits[at].file], nodeInputSplits[at]);
            const char *name = NULL;
            const char *seq = NULL;
            size_t nameLength = 0;
            size_t length = 0;
            uint64_t placed = 0;
            while (reader.next(name, nameLength, seq, length)) {
                if (length == 0 || length > maxSeqLen) {
                    continue;
                }
                if (++placed % PROGRESS_STEP == 0) {
                    progress.updateProgress();
                }
                header.assign(name, nameLength);
                header.append(1, '\n');
                writer.add(length, reinterpret_cast<const unsigned char *>(seq), header.c_str(),
                           header.size());
            }
            // the bar closes on an exact count, so a split hands back what its last step did not fill
            if (placed % PROGRESS_STEP != 0) {
                progress.updateProgress();
            }
            writer.flush();
#pragma omp atomic
            written += writer.bytesWritten();
        }
    }
    const uint64_t expect = layout.seqBytes + layout.hdrBytes;
    if (written != expect) {
        Debug(Debug::ERROR) << "Placed " << written << " byte but the ranks reserved " << expect
                            << ", the output would hold unwritten gaps\n";
        EXIT(EXIT_FAILURE);
    }
}

static std::vector<int> openDataFiles(const std::string &prefix, unsigned int fileBase,
                                      unsigned int files, uint64_t bytes,
                                      const std::vector<uint64_t> &fileStart) {
    std::vector<int> fd(files, -1);
    for (unsigned int i = 0; i < files; i++) {
        const std::string path = prefix + "." + SSTR(fileBase + i);
        fd[i] = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
        if (fd[i] < 0) {
            Debug(Debug::ERROR) << "Cannot open " << path << " for writing\n";
            EXIT(EXIT_FAILURE);
        }
        if (ftruncate(fd[i], static_cast<off_t>(fileStart[i + 1] - fileStart[i])) != 0) {
            Debug(Debug::ERROR) << "Cannot size " << path << "\n";
            EXIT(EXIT_FAILURE);
        }
    }
    (void) bytes;
    return fd;
}

// the output is never read back here, so it must not be left to evict what the next pass needs
static void dropCacheAndCloseFiles(std::vector<int> &fd) {
    for (size_t i = 0; i < fd.size(); i++) {
        if (fd[i] < 0) {
            continue;
        }
#if defined(__linux__)
        sync_file_range(fd[i], 0, 0, SYNC_FILE_RANGE_WAIT_BEFORE | SYNC_FILE_RANGE_WRITE
                                         | SYNC_FILE_RANGE_WAIT_AFTER);
        posix_fadvise(fd[i], 0, 0, POSIX_FADV_DONTNEED);
#endif
        if (close(fd[i]) != 0) {
            Debug(Debug::ERROR) << "Cannot close data file " << i << "\n";
            EXIT(EXIT_FAILURE);
        }
        fd[i] = -1;
    }
}
static void writeFileManifest(const std::string &db, const RunTable &runs, unsigned int filesPerNode) {
    const size_t files = runs.fileCount();
    std::vector<uint32_t> maxLen(files, 0);
    std::vector<uint32_t> minLen(files, 0);
    std::vector<uint64_t> entries(files, 0);
    for (size_t i = 0; i < runs.size(); i++) {
        const uint32_t file = runs[i].fileIdx();
        if (entries[file] == 0) {
            maxLen[file] = runs[i].seqLen();
        }
        minLen[file] = runs[i].seqLen();
        entries[file] += runs.rankEnd(i) - runs[i].rankBase();
    }
    const std::string tmp = db + ".files.tmp" + uniqueTmpSuffix();
    FILE *out = FileUtil::openAndDelete(tmp.c_str(), "w");
    fprintf(out, "#file\tnode\tsuffix\tmaxLen\tminLen\tentries\n");
    for (size_t file = 0; file < files; file++) {
        fprintf(out, "%zu\t%zu\t%zu\t%u\t%u\t%zu\n", file, file / filesPerNode, file % filesPerNode,
                maxLen[file], minLen[file], static_cast<size_t>(entries[file]));
    }
    if (fclose(out) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << tmp << "\n";
        EXIT(EXIT_FAILURE);
    }
    FileUtil::publishAtomically(tmp, db + ".files");
}

// every node states its segments with the global rank it computed, so the merge is a sort by rank
static RunTable mergeRunTables(const std::string &db, unsigned int nodeCount,
                               unsigned int filesPerNode, uint64_t sequences) {
    std::vector<RunTable::Segment> all;
    for (unsigned int node = 0; node < nodeCount; node++) {
        RunTable part;
        part.read(nodePartName(db, "runs", node));
        all.insert(all.end(), part.data(), part.data() + part.size());
    }
    SORT_SERIAL(all.begin(), all.end(),
                  [](const RunTable::Segment &first, const RunTable::Segment &second) {
                      return first.rankBase() < second.rankBase();
                  });
    RunTable merged;
    merged.setLayout(nodeCount, filesPerNode);
    merged.reserve(all.size());
    for (size_t i = 0; i < all.size(); i++) {
        merged.append(all[i].rankBase(), all[i].seqLen(), all[i].byteBase(), all[i].fileIdx(),
                      all[i].hdrBase());
    }
    merged.finish(sequences);
    merged.write(db + ".runs");
    return merged;
}

int lin8createdb(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, Parameters::PARSE_VARIADIC, 0);

    const size_t inputCount = par.filenames.size() - 1;
    const std::string db = par.filenames.back();
    const std::string headerDb = db + "_h";
    std::vector<std::string> filenames(par.filenames.begin(), par.filenames.begin() + inputCount);
    // the rank definition needs one fixed input order, and the full path is what stays unique
    SORT_SERIAL(filenames.begin(), filenames.end());

    const NodePlacement node = NodePlacement::resolve(par);
    // the record format caps the length, so a larger --max-seq-len can only be honoured up to it
    const uint32_t maxSeqLen =
        static_cast<uint32_t>(std::min<size_t>(par.maxSeqLen, LINCLUSTERDB_MAX_SEQ_LEN));
    const bool placed = FileUtil::fileExists(nodeDonePath(db, node.index).c_str());
    std::vector<InputSplit> nodeInputSplits;
    size_t budget = 0;
    Timer timer;
    LocalCounts nodeCounts;
    if (placed) {
        nodeCounts = readHistogram(nodePartName(db, "hist", node.index), par.threads);
        Debug(Debug::INFO) << "Node " << node.index << " of " << node.count << " already placed "
                           << nodeCounts.sequences << " sequences, rerunning to merge\n";
    } else {
        nodeInputSplits = inputSplitsForNode(filenames, node, par.threads);
        // no margin: this staging buffer is the only large allocation the pass makes
        budget = Util::computeMemory(par.splitMemoryLimit);
        Debug(Debug::INFO) << "Node " << node.index << " of " << node.count << " takes " << nodeInputSplits.size() << " inputSplit" << (nodeInputSplits.size() == 1 ? "" : "s")
                           << " of " << filenames.size() << " input file"
                           << (filenames.size() == 1 ? "" : "s") << ", buffer budget "
                           << (budget / (1024 * 1024)) << " MB\n";
        Debug(Debug::INFO) << "Counting the sequences and their lengths\n";
        nodeCounts = countSequenceLengths(filenames, nodeInputSplits, par.threads, maxSeqLen);
        Debug(Debug::INFO) << "Counted " << nodeCounts.sequences << " sequences, " << nodeCounts.residues
                           << " residues in " << timer.lap() << "\n";
        if (nodeCounts.rejected > 0) {
            Debug(Debug::WARNING) << "Skipped " << nodeCounts.rejected << " sequences outside 1 to "
                                  << maxSeqLen << " residues\n";
        }
        writeHistogram(nodePartName(db, "hist", node.index), nodeCounts, par.threads);
    }

    std::vector<LocalCounts> everyNodeCounts;
    if (node.count == 1) {
        everyNodeCounts.push_back(nodeCounts);
    } else {
        for (unsigned int other = 0; other < node.count; other++) {
            const std::string path = nodePartName(db, "hist", other);
            if (FileUtil::fileExists(path.c_str()) == false) {
                Debug(Debug::INFO) << "Wrote " << nodePartName(db, "hist", node.index)
                                   << ", rerun once every node has written its own\n";
                return EXIT_SUCCESS;
            }
            everyNodeCounts.push_back(readHistogram(path, par.threads));
        }
    }
    const RankLayout ranks = computeGlobalRankLayout(everyNodeCounts, node.index);
    if (ranks.sequences == 0) {
        Debug(Debug::ERROR) << "The input files have no usable entry\n";
        EXIT(EXIT_FAILURE);
    }

    if (placed == false) {
        for (unsigned int slot = 0; slot < par.threads; slot++) {
            const std::string data = db + "." + SSTR(node.index * par.threads + slot);
            if (FileUtil::fileExists(data.c_str()) && FileUtil::getFileSize(data) > 0) {
                Debug(Debug::ERROR) << data << " holds data but node " << node.index
                                    << " left no marker. Another run wrote it and did not finish;"
                                    << " remove this database and build it again\n";
                EXIT(EXIT_FAILURE);
            }
        }
        DbLayout layout;
        RunTable partial;
        planDatabaseLayout(everyNodeCounts, nodeCounts, ranks, par.threads, node.index * par.threads, layout, partial);
        partial.setLayout(node.count, par.threads);
        partial.finish(ranks.sequences);
        Debug(Debug::INFO) << "Planned " << partial.size() << " run segments over " << par.threads
                           << " data files, " << layout.seqBytes << " byte\n";

        timer.reset();
        std::vector<int> seqFd = openDataFiles(db, node.index * par.threads, par.threads,
                                               layout.seqBytes, layout.seqFileStart);
        std::vector<int> hdrFd = openDataFiles(headerDb, node.index * par.threads, par.threads,
                                               layout.hdrBytes, layout.hdrFileStart);
        Debug(Debug::INFO) << "Placing " << nodeCounts.sequences << " sequences by length, "
                           << PROGRESS_STEP / 1000000 << "M a step\n";
        writeSequencesAndHeaders(filenames, nodeInputSplits, nodeCounts, layout, seqFd, hdrFd, par.threads, budget,
                                 maxSeqLen);
        dropCacheAndCloseFiles(seqFd);
        dropCacheAndCloseFiles(hdrFd);
        // the run part is what tells the other nodes this one is done, so it must follow the data
        partial.write(nodePartName(db, "runs", node.index));
        Debug(Debug::INFO) << "Placed " << nodeCounts.sequences << " sequences in " << timer.lap() << "\n";

        markNodeDone(db, node.index);
    }

    bool complete = true;
    for (unsigned int other = 0; other < node.count; other++) {
        complete = complete && FileUtil::fileExists(nodePartName(db, "runs", other).c_str());
    }
    if (complete) {
        const RunTable merged = mergeRunTables(db, node.count, par.threads, ranks.sequences);
        writeFileManifest(db, merged, par.threads);
        const int dbtype = DBReader<DBKeyType>::setExtendedDbtype(Parameters::DBTYPE_AMINO_ACIDS,
                                                                  Parameters::DBTYPE_EXTENDED_RUNS);
        const std::string dbtypeBase = db + ".new" + uniqueTmpSuffix();
        DBWriter::writeDbtypeFile(dbtypeBase.c_str(), dbtype, false);
        FileUtil::publishAtomically(dbtypeBase + ".dbtype", db + ".dbtype");
        Debug(Debug::INFO) << "Database holds " << ranks.sequences << " sequences in "
                           << ranks.bytes << " byte\n";
    } else {
        Debug(Debug::INFO) << "Wrote the parts of node " << node.index
                           << ", rerun on this node once every node has written its own\n";
    }
    return EXIT_SUCCESS;
}
