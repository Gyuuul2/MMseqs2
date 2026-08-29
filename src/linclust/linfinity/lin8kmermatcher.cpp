
#include "Lin8Db.h"
#include "Lin8DbReader.h"
#include "Parameters.h"
#include "Debug.h"
#include "FileUtil.h"
#include "Util.h"
#include "Timer.h"
#include "NodePlacement.h"
#include "SubstitutionMatrix.h"
#include "ReducedMatrix.h"
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstring>
#include <cmath>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <omp.h>
#include "FastSort.h"
#include <cstdio>

class KmerExtractor {
public:
    KmerExtractor(unsigned int kmerSize, unsigned int alphabetSize, unsigned int mostPerSequence);

    size_t extract(const char *letters, size_t length, const unsigned char *letterToCode,
                   unsigned int keepPerSequence);

    static std::vector<unsigned char> buildLetterTable(const unsigned char *aa2num,
                                                       unsigned int reducedSize,
                                                       bool maskLowerCase);
    unsigned char unknownClass() const { return static_cast<unsigned char>(alphabetSize); }

    const uint64_t *keys() const { return keptKey.data(); }
    const uint16_t *positions() const { return keptPos.data(); }

    // the score is this many low bits; the rest names the bucket and must never be compared
    static const unsigned int SCORE_BITS = 48;
    static const uint64_t SCORE_MASK = (uint64_t(1) << SCORE_BITS) - 1;

    static uint64_t scoreOf(uint64_t mixed) { return mixed & SCORE_MASK; }
    static uint64_t bucketOf(uint64_t mixed, size_t buckets) {
        return (mixed >> SCORE_BITS) % buckets;
    }

    static const unsigned int CODE_BITS = 51;
    static const uint64_t CODE_MASK = (uint64_t(1) << CODE_BITS) - 1;

    static uint64_t spread(uint64_t code) {
        uint64_t v = code & CODE_MASK;
        v = (v * 0x9e3779b97f4a7c15ull) & CODE_MASK;
        v ^= v >> 26;
        v = (v * 0xbf58476d1ce4e5b9ull) & CODE_MASK;
        v ^= v >> 31;
        return v & CODE_MASK;
    }

    static uint64_t bucketOfKmer(uint64_t spreadCode, uint64_t buckets) {
        return (spreadCode >> (CODE_BITS - KmerRecord::BUCKET_BITS)) % buckets;
    }
    static uint64_t storedKey(uint64_t spreadCode) { return spreadCode & KmerRecord::KEY_MAX; }

    static uint64_t mix(uint64_t value) {
        // splitmix64 finaliser: two multiplies and three shifts, enough to make the halves independent
        value ^= value >> 30;
        value *= 0xbf58476d1ce4e5b9ull;
        value ^= value >> 27;
        value *= 0x94d049bb133111ebull;
        value ^= value >> 31;
        return value;
    }

    static uint64_t identityKey(const char *letters, size_t length,
                                const unsigned char *letterToCode);

    uint64_t adjacentAt(const char *letters, size_t length, const unsigned char *letterToCode,
                        size_t pos) const;
    uint64_t adjacentUnknown() const;

private:
    void offer(uint64_t score, uint64_t key, uint16_t pos);

    unsigned int kmerSize;
    unsigned int alphabetSize;
    unsigned int most;
    unsigned int keep;
    uint64_t dropHighDigit;
    std::vector<uint64_t> keptScore;
    std::vector<uint64_t> keptKey;
    std::vector<uint16_t> keptPos;
};

KmerExtractor::KmerExtractor(unsigned int kmerSize, unsigned int alphabetSize,
                             unsigned int mostPerSequence)
    : kmerSize(kmerSize), alphabetSize(alphabetSize), most(mostPerSequence), keep(mostPerSequence) {
    uint64_t span = 1;
    for (unsigned int i = 0; i < kmerSize; i++) {
        if (span > CODE_MASK / alphabetSize) {
            Debug(Debug::ERROR) << "A k-mer of " << kmerSize << " over an alphabet of "
                                << alphabetSize << " needs more than " << CODE_BITS
                                << " bit. Lower -k to " << i << " or fewer\n";
            EXIT(EXIT_FAILURE);
        }
        span *= alphabetSize;
    }
    dropHighDigit = span / alphabetSize;
    keptScore.reserve(most);
    keptKey.reserve(most);
    keptPos.reserve(most);

}

uint64_t KmerExtractor::identityKey(const char *letters, size_t length,
                                    const unsigned char *letterToCode) {
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < length; i++) {
        hash = (hash ^ letterToCode[static_cast<unsigned char>(letters[i])]) * 1099511628211ull;
    }
    return mix(hash) & CODE_MASK;
}

uint64_t KmerExtractor::adjacentAt(const char *letters, size_t length,
                                   const unsigned char *letterToCode, size_t pos) const {
    const unsigned int half = KmerRecord::ADJACENT_COUNT / 2;
    uint64_t packed = 0;
    for (unsigned int slot = 0; slot < KmerRecord::ADJACENT_COUNT; slot++) {
        uint64_t code = unknownClass();
        if (slot < half) {
            const size_t before = half - slot;
            if (pos >= before) {
                code = letterToCode[static_cast<unsigned char>(letters[pos - before])];
            }
        } else {
            const size_t after = pos + kmerSize + (slot - half);
            if (after < length) {
                code = letterToCode[static_cast<unsigned char>(letters[after])];
            }
        }
        packed |= (code & KmerRecord::ADJACENT_MAX) << (slot * KmerRecord::ADJACENT_BITS);
    }
    return packed;
}

uint64_t KmerExtractor::adjacentUnknown() const {
    uint64_t packed = 0;
    for (unsigned int slot = 0; slot < KmerRecord::ADJACENT_COUNT; slot++) {
        packed |= uint64_t(unknownClass()) << (slot * KmerRecord::ADJACENT_BITS);
    }
    return packed;
}

std::vector<unsigned char> KmerExtractor::buildLetterTable(const unsigned char *aa2num,
                                                           unsigned int reducedSize,
                                                           bool maskLowerCase) {
    std::vector<unsigned char> table(256, static_cast<unsigned char>(reducedSize));
    for (int letter = 'A'; letter <= 'Z'; letter++) {
        const unsigned char code = aa2num[letter];
        if (code < reducedSize) {
            table[letter] = code;
            // reading a masked database without honouring the mask is what the flag turns off
            table[letter - 'A' + 'a'] = maskLowerCase ? static_cast<unsigned char>(reducedSize) : code;
        }
    }
    return table;
}

void KmerExtractor::offer(uint64_t score, uint64_t key, uint16_t pos) {
    if (keptScore.size() < keep) {
        keptScore.push_back(score);
        keptKey.push_back(key);
        keptPos.push_back(pos);
        return;
    }
    size_t worst = 0;
    for (size_t i = 1; i < keptScore.size(); i++) {
        if (keptScore[i] > keptScore[worst]
            || (keptScore[i] == keptScore[worst] && keptPos[i] > keptPos[worst])) {
            worst = i;
        }
    }
    // ties break on position so the choice does not depend on the order they were offered in
    if (score > keptScore[worst] || (score == keptScore[worst] && pos >= keptPos[worst])) {
        return;
    }
    keptScore[worst] = score;
    keptKey[worst] = key;
    keptPos[worst] = pos;
}

size_t KmerExtractor::extract(const char *letters, size_t length,
                              const unsigned char *letterToCode,
                              unsigned int keepPerSequence) {
    keep = keepPerSequence < most ? keepPerSequence : most;
    const unsigned char unknownCode = unknownClass();
    keptScore.clear();
    keptKey.clear();
    keptPos.clear();
    if (length < kmerSize) {
        return 0;
    }
    uint64_t code = 0;
    unsigned int filled = 0;
    for (size_t at = 0; at < length; at++) {
        const unsigned char in = letterToCode[static_cast<unsigned char>(letters[at])];
        if (in >= unknownCode) {
            // the window cannot span an unknown residue, so it starts again after one
            code = 0;
            filled = 0;
            continue;
        }
        if (filled == kmerSize) {
            code -= static_cast<uint64_t>(
                        letterToCode[static_cast<unsigned char>(letters[at - kmerSize])])
                    * dropHighDigit;
            filled--;
        }
        code = code * alphabetSize + in;
        filled++;
        if (filled == kmerSize) {
            offer(scoreOf(mix(code)), code, static_cast<uint16_t>(at + 1 - kmerSize));
        }
    }
    return keptScore.size();
}

// ---- lin8kmers ----
#ifdef OPENMP
#endif

static const uint64_t CHUNK_BYTES = 64ull * 1024 * 1024 * 1024;

struct ChunkPlan {
    uint64_t rankBegin;
    uint64_t rankEnd;
    size_t lengthBlock;
};

static std::string chunkName(const std::string &out, unsigned int node, size_t chunk) {
    return out + "." + SSTR(node) + "." + SSTR(chunk);
}

static std::vector<ChunkPlan> planChunks(const RunDbReader &reader, const std::vector<size_t> &blocks,
                                         uint64_t capBytes) {
    std::vector<ChunkPlan> chunks;
    const RunTable &runs = reader.getRunTable();
    for (size_t at = 0; at < blocks.size(); at++) {
        uint64_t begin = 0;
        uint64_t end = 0;
        for (size_t i = 0; i < runs.size(); i++) {
            if (runs[i].fileIdx() % runs.blocksPerNode() != blocks[at]) {
                continue;
            }
            begin = (end == 0) ? runs[i].rankBase() : std::min(begin, runs[i].rankBase());
            end = std::max(end, runs.rankEnd(i));
        }
        if (end <= begin) {
            continue;
        }
        const uint64_t fromByte = runs.byteAtRank(begin);
        const uint64_t toByte = runs.byteAtRank(end);
        const uint64_t span = toByte - fromByte;
        const uint64_t parts = std::max<uint64_t>(1, (span + capBytes - 1) / capBytes);
        const uint64_t step = std::max<uint64_t>(1, (span + parts - 1) / parts);
        for (uint64_t byteAt = fromByte; byteAt < toByte; byteAt += step) {
            ChunkPlan chunk;
            chunk.lengthBlock = blocks[at];
            chunk.rankBegin = runs.rankAtByte(byteAt);
            chunk.rankEnd = std::min<uint64_t>(runs.rankAtByte(std::min(byteAt + step, toByte)), end);
            if (chunk.rankEnd > chunk.rankBegin) {
                chunks.push_back(chunk);
            }
        }
    }
    return chunks;
}

static unsigned int keepFor(unsigned int keepPerSequence, float keepScale, uint32_t length) {
    const double asked = (double) keepPerSequence + (double) keepScale * (double) length;
    return (unsigned int) (asked < (double) length ? asked : (double) length);
}

static unsigned int mostPerSequence(unsigned int keepPerSequence, float keepScale) {
    return keepFor(keepPerSequence, keepScale, RunTable::MAX_SEQ_LEN);
}

static uint64_t extractChunk(const RunDbReader &reader, const ChunkPlan &chunk, BucketWriter<KmerRecord> &writer,
                         const unsigned char *letterToCode, unsigned int kmerSize,
                         unsigned int alphabetSize, unsigned int keepPerSequence, float keepScale,
                         unsigned int threads, std::vector<std::vector<uint64_t> > &subBucketCounts) {
    const size_t buckets = KmerRecord::BUCKET_COUNT;
    uint64_t emitted = 0;
#pragma omp parallel num_threads(threads) reduction(+ : emitted)
    {
        unsigned int thread = 0;
#ifdef OPENMP
        thread = static_cast<unsigned int>(omp_get_thread_num());
#endif
        KmerExtractor extractor(kmerSize, alphabetSize, mostPerSequence(keepPerSequence, keepScale));
        RunDbReader::Cursor cursor;
        KmerRecord record;
        std::vector<uint64_t> &counts = subBucketCounts[thread];
#pragma omp for schedule(dynamic, 1024)
        for (uint64_t rank = chunk.rankBegin; rank < chunk.rankEnd; rank++) {
            if (reader.isValid(rank) == false) {
                continue;
            }
            const uint32_t length = reader.getSeqLen(rank, cursor);
            if (length > RunTable::MAX_SEQ_LEN) {
                continue;
            }
            const char *letters = reader.getData(rank, cursor);
            const size_t kept = extractor.extract(letters, length, letterToCode,
                                                  keepFor(keepPerSequence, keepScale, length));
            for (size_t i = 0; i < kept; i++) {
                const uint64_t spread = KmerExtractor::spread(extractor.keys()[i]);
                const uint64_t pos = extractor.positions()[i];
                record.set(KmerExtractor::storedKey(spread), rank, pos, false,
                           extractor.adjacentAt(letters, length, letterToCode, pos));
                const size_t bucket = KmerExtractor::bucketOfKmer(spread, buckets);
                writer.add(thread, record, bucket);
                counts[bucket * KmerRecord::SUB_BUCKET_COUNT + record.subBucket()]++;
                emitted++;
            }
            // a sequence with no k-mer of its own would otherwise never be seen again
            if (kept == 0) {
                const uint64_t spread =
                    KmerExtractor::spread(KmerExtractor::identityKey(letters, length, letterToCode));
                record.set(KmerExtractor::storedKey(spread), rank, 0, true,
                           extractor.adjacentUnknown());
                const size_t bucket = KmerExtractor::bucketOfKmer(spread, buckets);
                writer.add(thread, record, bucket);
                counts[bucket * KmerRecord::SUB_BUCKET_COUNT + record.subBucket()]++;
                emitted++;
            }
        }
    }
    return emitted;
}

static unsigned int longestKmer(unsigned int classes) {
    uint64_t span = 1;
    unsigned int k = 0;
    while (span <= KmerExtractor::CODE_MASK / classes) {
        span *= classes;
        k++;
    }
    return k;
}

static void setKmerLengthAndAlphabet(Parameters &par, uint64_t residues) {
    const bool nearIdentical = par.seqIdThr + 0.001 >= 0.99;
    if (par.alphabetSize.values.aminoacid() == 0) {
        const int reduced = (par.kmerSize == 0 && nearIdentical)
                                ? 21
                                : Parameters::CLUST_LINEAR_DEFAULT_ALPH_SIZE;
        par.alphabetSize = MultiParam<NuclAA<int> >(NuclAA<int>(reduced, 5));
    }
    // one class short of the alphabet: the unknown class ends a window rather than joining it
    const unsigned int classes = (unsigned int) par.alphabetSize.values.aminoacid() - 1;
    const unsigned int longest = longestKmer(classes);
    if (par.kmerSize == 0) {
        par.kmerSize = par.seqIdThr + 0.001 >= 0.9
                           ? 14
                           : std::max(10, (int)(log((double)residues) / log(8.7)));
        if ((unsigned int) par.kmerSize > longest) {
            Debug(Debug::WARNING) << "A database this size asks for a k-mer of " << par.kmerSize
                                  << ", which does not fit " << KmerExtractor::CODE_BITS
                                  << " bit over " << classes << " classes. Using " << longest << "\n";
            par.kmerSize = (int) longest;
        }
    }
}

int lin8kmers(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.kmerSize = 0;
    par.alphabetSize = MultiParam<NuclAA<int> >(NuclAA<int>(0, 5));
    par.parseParameters(argc, argv, command, true, 0, 0);

    // a bucket file stays open for the whole run, so the default limit of a thousand is not enough
    FileUtil::fixRlimitNoFile();

    const NodePlacement node = NodePlacement::resolve(par);
    RunDbReader reader(par.db1, par.linclusthashValid);
    reader.open();
    setKmerLengthAndAlphabet(par, reader.getTotalBytes());

    SubstitutionMatrix full(par.scoringMatrixFile.values.aminoacid().c_str(), 2.0, -0.2);
    const int reduced = par.alphabetSize.values.aminoacid();
    ReducedMatrix subMat(full.probMatrix, full.subMatrixPseudoCounts, full.aa2num, full.num2aa,
                         full.alphabetSize, reduced, 2.0);
    const unsigned int classes = static_cast<unsigned int>(reduced) - 1;
    const std::vector<unsigned char> letterToCode =
        KmerExtractor::buildLetterTable(subMat.aa2num, classes, par.maskLowerCaseMode == 1);

    const unsigned int keepPerSequence =
        par.kmersPerSequence > 1 ? par.kmersPerSequence - 1 : 1;
    const size_t budget = static_cast<size_t>(Util::computeMemory(par.splitMemoryLimit) * 0.95);
    const std::vector<size_t> blocks = lengthBlocksForNode(reader.getRunTable(), node);
    const std::vector<ChunkPlan> chunks = planChunks(reader, blocks, CHUNK_BYTES);
    Debug(Debug::INFO) << "Node " << node.index << " of " << node.count << " takes " << blocks.size()
                       << " length blocks in " << chunks.size() << " chunks, k " << par.kmerSize
                       << ", alphabet " << reduced << ", keeping " << keepPerSequence << "\n";

    const std::string prefix = par.db2 + "." + SSTR(node.index);
    std::vector<uint64_t> keep(KmerRecord::BUCKET_COUNT, 0);
    const size_t countEntries = KmerRecord::BUCKET_COUNT * KmerRecord::SUB_BUCKET_COUNT;
    const std::string countsPath = par.db2 + "." + SSTR(node.index) + ".counts";
    size_t tableChunks = 0;
    std::vector<uint64_t> subBucketBase =
        readSubBucketCounts(countsPath, countEntries, tableChunks);
    // a manifest covers a run of planned chunks and names in its span how many it reaches
    uint64_t coveredPlanned = 0;
    const size_t doneManifests =
        readBucketManifests(par.db2 + "." + SSTR(node.index), chunks.size(), keep, &coveredPlanned);
    size_t manifest = std::min(doneManifests, tableChunks);
    if (manifest < doneManifests) {
        keep.assign(keep.size(), 0);
        coveredPlanned = 0;
        readBucketManifests(par.db2 + "." + SSTR(node.index), manifest, keep, &coveredPlanned);
    }
    const size_t resume = coveredPlanned;
    if (resume > 0) {
        Debug(Debug::INFO) << "Resuming after " << resume << " chunks a previous run finished\n";
    }

    std::vector<std::vector<uint64_t> > subBucketCounts(par.threads,
                                                        std::vector<uint64_t>(countEntries, 0));

    Timer timer;
    BucketWriter<KmerRecord> writer(prefix, KmerRecord::BUCKET_COUNT, par.threads, budget);
    writer.openAt(keep);
    uint64_t written = 0;
    Debug::Progress progress(chunks.size() - resume);
    size_t chunkFirst = resume;
    uint64_t pendingRecords = 0;
    writer.resetCounts();
    for (size_t at = resume; at < chunks.size(); at++) {
        pendingRecords += extractChunk(reader, chunks[at], writer, letterToCode.data(), par.kmerSize,
                                       classes, keepPerSequence,
                                       par.kmersPerSequenceScale.values.aminoacid(), par.threads,
                                       subBucketCounts);
        // a durable chunk is a run of planned chunks ended by bytes, so a small database is one flush
        if (pendingRecords * KmerRecord::DISK_BYTES >= CHUNK_BYTES || at + 1 == chunks.size()) {
            writer.endChunk(par.threads);
            // the manifest goes down after the data it describes, so its presence means it is whole
            writeBucketManifest(chunkName(par.db2, node.index, manifest) + ".manifest",
                                writer.chunkCounts(), "chunk", chunkFirst, at + 1);
            writeSubBucketCounts(countsPath, subBucketBase, subBucketCounts, manifest + 1);
            written += pendingRecords;
            writer.resetCounts();
            chunkFirst = at + 1;
            pendingRecords = 0;
            manifest++;
        }
        if (at + 1 == chunks.size() || chunks[at + 1].lengthBlock != chunks[at].lengthBlock) {
            reader.releaseLengthBlock(chunks[at].lengthBlock);
        }
        progress.updateProgress();
    }
    writer.close();
    writeSubBucketCounts(countsPath, subBucketBase, subBucketCounts, manifest);

    const std::string nodesTmp = par.db2 + "." + SSTR(node.index) + ".tmp";
    FILE *nodes = FileUtil::openAndDelete(nodesTmp.c_str(), "w");
    fprintf(nodes, "nodes\t%u\nbuckets\t%zu\nalphabet\t%d\nkmer\t%d\nranks\t%zu\n",
            node.count, (size_t) KmerRecord::BUCKET_COUNT, reduced, par.kmerSize,
            (size_t) reader.getSize());
    if (fclose(nodes) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << nodesTmp << "\n";
        EXIT(EXIT_FAILURE);
    }
    FileUtil::publishAtomically(nodesTmp, par.db2);
    markNodeDone(par.db2, node.index);

    Debug(Debug::INFO) << "Wrote " << written << " k-mer records in " << timer.lap() << "\n";
    reader.close();
    return EXIT_SUCCESS;
}

// ---- lin8pairs ----
#ifdef OPENMP
#endif

#ifdef OPENMP
#endif

static void readBucketShape(const std::string &path, unsigned int &nodes, size_t &buckets, int &alphabet,
                      uint64_t &ranks) {
    FILE *in = fopen(path.c_str(), "r");
    if (in == NULL) {
        Debug(Debug::ERROR) << "Cannot open " << path << ". Run lin8kmers first\n";
        EXIT(EXIT_FAILURE);
    }
    nodes = 0;
    buckets = 0;
    int kmerSize = 0;
    size_t seen = 0;
    const bool read = fscanf(in, "nodes\t%u\nbuckets\t%zu\nalphabet\t%d\nkmer\t%d\nranks\t%zu",
                             &nodes, &buckets, &alphabet, &kmerSize, &seen) == 5;
    fclose(in);
    ranks = seen;
    if (read == false || nodes == 0 || alphabet < 2 || buckets != KmerRecord::BUCKET_COUNT) {
        Debug(Debug::ERROR) << path << " names " << nodes << " nodes and " << buckets
                            << " buckets, and this build makes " << KmerRecord::BUCKET_COUNT << "\n";
        EXIT(EXIT_FAILURE);
    }
}

struct FileSlice {
    unsigned int node;
    size_t first;
    size_t count;
};

struct BySubBucket {
    size_t operator()(const KmerRecord &record) const { return record.subBucket(); }
};

struct BySubRange {
    uint64_t ranks;
    size_t ranges;
    BySubRange(uint64_t ranks, size_t ranges) : ranks(ranks), ranges(ranges) {}
    size_t operator()(const PairRecord &record) const {
        return PairRecord::subRangeOf(record.rep(), ranks, ranges);
    }
};

template <typename Record, typename SubOf>
static void scanSlice(const std::string &path, const FileSlice &slice, const SubOf &subOf,
                      std::vector<uint64_t> &perPrefix, std::vector<size_t> &place,
                      RawArray<Record> &into) {
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        Debug(Debug::ERROR) << "Cannot open " << path << ", which node " << slice.node
                            << " should have written\n";
        EXIT(EXIT_FAILURE);
    }
    const off_t at = static_cast<off_t>(slice.first * Record::DISK_BYTES);
    const off_t span = static_cast<off_t>(slice.count * Record::DISK_BYTES);
    posix_fadvise(fd, at, span, POSIX_FADV_SEQUENTIAL);
    const size_t batch = std::min<size_t>(slice.count, 1u << 16);
    std::vector<unsigned char> raw(batch * Record::DISK_BYTES);
    for (size_t done = 0; done < slice.count; ) {
        const size_t want = std::min<size_t>(slice.count - done, batch);
        size_t got = 0;
        while (got < want * Record::DISK_BYTES) {
            const off_t at = static_cast<off_t>((slice.first + done) * Record::DISK_BYTES + got);
            const ssize_t read = pread(fd, raw.data() + got, want * Record::DISK_BYTES - got, at);
            if (read <= 0) {
                Debug(Debug::ERROR) << "Cannot read " << path << "\n";
                EXIT(EXIT_FAILURE);
            }
            got += static_cast<size_t>(read);
        }
        for (size_t i = 0; i < want; i++) {
            Record record;
            record.unpack(&raw[i * Record::DISK_BYTES]);
            const size_t prefix = subOf(record);
            if (place.empty()) {
                perPrefix[prefix]++;
            } else {
                into[place[prefix]++] = record;
            }
        }
        done += want;
    }
    if (place.empty() == false) {
        posix_fadvise(fd, at, span, POSIX_FADV_DONTNEED);
    }
    close(fd);
}

template <typename Record, typename SubOf>
static std::vector<size_t> loadBucket(const std::string &prefix, const BucketCounts &counts,
                                      unsigned int nodes, size_t bucket, size_t prefixes,
                                      const SubOf &subOf, size_t budget, unsigned int threads,
                                      const char *what, const char *narrower, const char *producer,
                                      RawArray<Record> &into) {
    const std::vector<uint64_t> subCounts = counts.of(bucket);
    std::vector<size_t> starts(prefixes + 1, 0);
    for (size_t i = 0; i < prefixes; i++) {
        starts[i + 1] = starts[i] + subCounts[i];
    }
    requireArena(std::string(what) + " " + SSTR(bucket), starts.back() * sizeof(Record), budget,
                 narrower);
    into.resize(starts.back());

    std::vector<FileSlice> slices;
    std::vector<std::string> path(nodes);
    for (unsigned int node = 0; node < nodes; node++) {
        path[node] = prefix + "." + SSTR(node) + "." + SSTR(bucket);
        const std::vector<uint64_t> mine = counts.of(bucket, node);
        size_t records = 0;
        for (size_t i = 0; i < prefixes; i++) {
            records += mine[i];
        }
        const size_t cut = std::min<size_t>(std::max<size_t>(threads / nodes, 1), records);
        for (size_t i = 0; i < cut; i++) {
            FileSlice slice;
            slice.node = node;
            slice.first = records * i / cut;
            slice.count = records * (i + 1) / cut - slice.first;
            slices.push_back(slice);
        }
    }

    std::vector<std::vector<uint64_t> > sliceCounts(slices.size(),
                                                    std::vector<uint64_t>(prefixes, 0));
    std::vector<size_t> counting;
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (size_t i = 0; i < slices.size(); i++) {
        scanSlice(path[slices[i].node], slices[i], subOf, sliceCounts[i], counting, into);
    }

    std::vector<std::vector<size_t> > at(slices.size());
    {
        std::vector<size_t> running(starts.begin(), starts.end() - 1);
        for (size_t i = 0; i < slices.size(); i++) {
            at[i] = running;
            for (size_t j = 0; j < prefixes; j++) {
                running[j] += sliceCounts[i][j];
            }
        }
        for (size_t j = 0; j < prefixes; j++) {
            if (running[j] != starts[j + 1]) {
                Debug(Debug::ERROR) << what << " " << bucket << " prefix " << j << " holds "
                                    << (running[j] - starts[j]) << " records and the counts say "
                                    << subCounts[j] << ". Was " << producer << " still running?\n";
                EXIT(EXIT_FAILURE);
            }
        }
    }

#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (size_t i = 0; i < slices.size(); i++) {
        scanSlice(path[slices[i].node], slices[i], subOf, sliceCounts[i], at[i], into);
    }
    return starts;
}


static int adjacencyScore(const KmerRecord &member, const short **centerRow) {
    int score = 0;
    for (unsigned int slot = 0; slot < KmerRecord::ADJACENT_COUNT; slot++) {
        score += centerRow[slot][member.adjacentAt(slot)];
    }
    return score;
}

static void swapCenterSequence(KmerRecord *group, size_t size, size_t round, BaseMatrix *subMat) {
    const short *centerRow[KmerRecord::ADJACENT_COUNT];
    for (unsigned int slot = 0; slot < KmerRecord::ADJACENT_COUNT; slot++) {
        centerRow[slot] = subMat->subMatrix[group[round - 1].adjacentAt(slot)];
    }
    size_t best = round;
    int lowest = INT_MAX;
    for (size_t i = round; i < size; i++) {
        bool spent = false;
        for (size_t used = 0; used < round && spent == false; used++) {
            spent = group[used].rank() == group[i].rank();
        }
        if (spent) {
            continue;
        }
        const int score = adjacencyScore(group[i], centerRow);
        if (score <= lowest) {
            lowest = score;
            best = i;
        }
    }
    std::swap(group[round], group[best]);
}

static void assignGroup(KmerRecord *group, size_t size, const RunDbReader &reader,
                         float covThr, int covMode, bool onlyExtendable, BaseMatrix *subMat,
                         int adjacentRounds, std::vector<PairRecord> &out) {
    if (size < 2) {
        return;
    }
    const size_t before = out.size();
    for (size_t round = 0; round <= (size_t) adjacentRounds && round < size; round++) {
        if (round > 0) {
            swapCenterSequence(group, size, round, subMat);
        }
        const uint64_t rep = group[round].rank();
        const uint64_t repPos = group[round].pos();
        const uint32_t queryLen = reader.getSeqLen(rep);
        RunDbReader::Cursor at;
        for (size_t i = 0; i < size; i++) {
            const uint64_t member = group[i].rank();
            const uint32_t targetLen = reader.getSeqLen(member, at);
            const int diagonal = static_cast<int>(repPos) - static_cast<int>(group[i].pos());
            const bool extendable =
                diagonal < 0 || diagonal > static_cast<int>(queryLen) - static_cast<int>(targetLen);
            const bool covered = Util::canBeCovered(covThr, covMode, queryLen, targetLen);
            if (onlyExtendable ? extendable : covered) {
                PairRecord pair;
                pair.set(rep, member, diagonal);
                out.push_back(pair);
            }
        }
    }
    if (adjacentRounds > 0) {
        std::sort(out.begin() + before, out.end(), PairRecord::byRepAndMember);
        out.erase(std::unique(out.begin() + before, out.end(), PairRecord::sameRepAndMember),
                  out.end());
    }
}

static std::vector<size_t> setupThreadOffsets(const RawArray<KmerRecord> &records, size_t pieces) {
    std::vector<size_t> threadOffsets(1, 0);
    const size_t even = records.size() / std::max<size_t>(1, pieces);
    for (size_t thread = 1; thread < pieces; thread++) {
        size_t at = std::max<size_t>(std::max(threadOffsets.back(), thread * even), 1);
        while (at < records.size() && records[at].key() == records[at - 1].key()) {
            at++;
        }
        threadOffsets.push_back(at);
    }
    threadOffsets.push_back(records.size());
    return threadOffsets;
}

int lin8pairs(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);

    FileUtil::fixRlimitNoFile();
    const NodePlacement node = NodePlacement::resolve(par);
    RunDbReader reader(par.db1, par.linclusthashValid);
    reader.open();
    unsigned int writerNodes = 0;
    size_t buckets = 0;
    int alphabet = 0;
    uint64_t ranks = 0;
    readBucketShape(par.db2, writerNodes, buckets, alphabet, ranks);
    requireEveryNodeDone(par.db2, writerNodes);
    const BucketCounts bucketCounts(par.db2, writerNodes, KmerRecord::SUB_BUCKET_COUNT,
                                    KmerRecord::BUCKET_COUNT);

    // the same reduced alphabet the extraction pass stored the residues in
    SubstitutionMatrix full(par.scoringMatrixFile.values.aminoacid().c_str(), 2.0, -0.2);
    ReducedMatrix subMat(full.probMatrix, full.subMatrixPseudoCounts, full.aa2num, full.num2aa,
                         full.alphabetSize, alphabet, 2.0);
    const int adjacentRounds = par.includeAdjacency ? par.adjIteration : 0;

    Timer timer;
    uint64_t pairs = 0;
    uint64_t groups = 0;
    size_t myBuckets = 0;
    for (size_t bucket = node.index; bucket < buckets; bucket += node.count) {
        myBuckets++;
    }
    Debug(Debug::INFO) << "Node " << node.index << " of " << node.count << " takes " << myBuckets
                       << " of " << buckets << " buckets\n";

    if (par.linclustRanges < 1 || (size_t) par.linclustRanges > PairRecord::MAX_RANGE_COUNT) {
        Debug(Debug::ERROR) << "--ranges must be between 1 and " << PairRecord::MAX_RANGE_COUNT
                            << ", past which the sub range arithmetic leaves sixty four bits\n";
        EXIT(EXIT_FAILURE);
    }
    const size_t rangeCount = (size_t) par.linclustRanges;

    const uint64_t CHUNK_BYTES = 4ull * 1024 * 1024 * 1024;
    const std::string prefix = par.db3 + "." + SSTR(node.index);
    const size_t budget = static_cast<size_t>(Util::computeMemory(par.splitMemoryLimit) * 0.95);
    std::vector<uint64_t> keep(rangeCount, 0);
    // as in the extraction pass: the lower of what the manifests claim and what the table covers
    uint64_t resumeAt = 0;
    const size_t countEntries = rangeCount * PairRecord::SUB_RANGE_COUNT;
    const std::string countsPath = par.db3 + "." + SSTR(node.index) + ".counts";
    size_t tableChunks = 0;
    std::vector<uint64_t> subRangeBase =
        readSubBucketCounts(countsPath, countEntries, tableChunks);
    const size_t doneChunks = readBucketManifests(prefix, myBuckets, keep, &resumeAt);
    const size_t done = std::min(doneChunks, tableChunks);
    if (done < doneChunks) {
        keep.assign(keep.size(), 0);
        resumeAt = 0;
        readBucketManifests(prefix, done, keep, &resumeAt);
    }
    if (done > 0) {
        Debug(Debug::INFO) << "Resuming after " << done << " chunks a previous run finished\n";
    }
    std::vector<std::vector<uint64_t> > subRangeCounts(par.threads,
                                                       std::vector<uint64_t>(countEntries, 0));
    BucketWriter<PairRecord> writer(prefix, rangeCount, par.threads, budget);
    writer.openAt(keep);

    Debug::Progress progress(myBuckets);
    uint64_t pending = 0;
    size_t chunk = done;
    size_t first = node.index;
    while (first < resumeAt) {
        first += node.count;
    }
    size_t chunkFirst = first;
    writer.resetCounts();
    for (size_t bucket = first; bucket < buckets; bucket += node.count) {
        RawArray<KmerRecord> records;
        const std::vector<size_t> starts =
            loadBucket(par.db2, bucketCounts, writerNodes, bucket, KmerRecord::SUB_BUCKET_COUNT,
                       BySubBucket(), budget - writer.bytesHeld(), par.threads, "Bucket",
                       "build with more buckets", "lin8-kmers", records);
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
        for (size_t i = 0; i < KmerRecord::SUB_BUCKET_COUNT; i++) {
            SORT_SERIAL(records.begin() + starts[i], records.begin() + starts[i + 1],
                        KmerRecord::byKeyAndRank);
        }
        // a thread owns a range of the sorted bucket, so moving a center only touches its own
        const size_t pieces = par.threads * 8;
        const std::vector<size_t> threadOffsets = setupThreadOffsets(records, pieces);
        std::vector<uint64_t> made(par.threads, 0);
        std::vector<uint64_t> seen(par.threads, 0);
#pragma omp parallel num_threads(par.threads)
        {
            unsigned int thread = 0;
#ifdef OPENMP
            thread = static_cast<unsigned int>(omp_get_thread_num());
#endif
            std::vector<PairRecord> out;
            std::vector<uint64_t> &counts = subRangeCounts[thread];
#pragma omp for schedule(dynamic, 1)
            for (int part = 0; part < (int) pieces; part++) {
                size_t begin = threadOffsets[part];
                while (begin < threadOffsets[part + 1]) {
                    size_t end = begin + 1;
                    while (end < threadOffsets[part + 1] && records[end].key() == records[begin].key()) {
                        end++;
                    }
                    out.clear();
                    assignGroup(&records[begin], end - begin, reader, par.covThr, par.covMode,
                                 par.includeOnlyExtendable, &subMat, adjacentRounds, out);
                    for (size_t i = 0; i < out.size(); i++) {
                        // one division for the file and the place inside it, not one each
                        const size_t fine = PairRecord::fineOf(out[i].rep(), ranks, rangeCount);
                        writer.add(thread, out[i], fine / PairRecord::SUB_RANGE_COUNT);
                        counts[fine]++;
                    }
                    made[thread] += out.size();
                    seen[thread] += (end - begin > 1);
                    begin = end;
                }
            }
        }
        uint64_t inBucket = 0;
        for (unsigned int thread = 0; thread < par.threads; thread++) {
            inBucket += made[thread];
            groups += seen[thread];
        }
        pairs += inBucket;
        pending += inBucket * PairRecord::DISK_BYTES;
        const bool last = bucket + node.count >= buckets;
        if (pending >= CHUNK_BYTES || last) {
            writer.endChunk(par.threads);
            writeBucketManifest(prefix + "." + SSTR(chunk) + ".manifest", writer.chunkCounts(),
                                "bucket", chunkFirst, bucket + 1);
            writeSubBucketCounts(countsPath, subRangeBase, subRangeCounts, chunk + 1);
            if (par.removeTmpFiles) {
                dropConsumed(par.db2, writerNodes, chunkFirst, bucket + 1, node.count);
            }
            writer.resetCounts();
            chunkFirst = bucket + node.count;
            pending = 0;
            chunk++;
        }
        progress.updateProgress();
    }
    writer.close();

    const std::string shapeTmp = par.db3 + "." + SSTR(node.index) + ".tmp";
    FILE *shape = FileUtil::openAndDelete(shapeTmp.c_str(), "w");
    fprintf(shape, "nodes\t%u\nranges\t%zu\nranks\t%zu\n", node.count, rangeCount,
            (size_t) ranks);
    if (fclose(shape) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << shapeTmp << "\n";
        EXIT(EXIT_FAILURE);
    }
    FileUtil::publishAtomically(shapeTmp, par.db3);
    markNodeDone(par.db3, node.index);
    Debug(Debug::INFO) << "Made " << pairs << " pairs from " << groups << " groups in " << timer.lap() << "\n";
    reader.close();
    return EXIT_SUCCESS;
}

// ---- lin8pref ----
#ifdef OPENMP
#endif

static void readRangeShape(const std::string &path, unsigned int &nodes, size_t &ranges,
                      uint64_t &ranks) {
    FILE *in = fopen(path.c_str(), "r");
    if (in == NULL) {
        Debug(Debug::ERROR) << "Cannot open " << path << ". Run lin8pairs first\n";
        EXIT(EXIT_FAILURE);
    }
    nodes = 0;
    ranges = 0;
    size_t seen = 0;
    const bool read = fscanf(in, "nodes\t%u\nranges\t%zu\nranks\t%zu", &nodes, &ranges, &seen) == 3;
    fclose(in);
    ranks = seen;
    if (read == false || nodes == 0 || ranges == 0 || ranges > PairRecord::MAX_RANGE_COUNT) {
        Debug(Debug::ERROR) << path << " names " << nodes << " nodes and " << ranges
                            << " ranges, which is not a set of candidate pairs\n";
        EXIT(EXIT_FAILURE);
    }
}

static void pickBestDiagonal(const PairRecord *run, size_t size, PairRecord &out) {
    size_t best = 0;
    size_t bestCount = 0;
    size_t at = 0;
    while (at < size) {
        size_t end = at + 1;
        while (end < size && run[end].diagonal() == run[at].diagonal()) {
            end++;
        }
        if (end - at >= bestCount) {
            bestCount = end - at;
            best = at;
        }
        at = end;
    }
    out = run[best];
}

static void addRanksWithNoRows(uint64_t from, uint64_t until,
                               const std::vector<std::vector<PairRecord> > &rows,
                               const RunDbReader *live, std::vector<PairRecord> &out) {
    size_t piece = 0;
    size_t at = 0;
    for (uint64_t rank = from; rank < until; rank++) {
        while (piece < rows.size()) {
            if (at >= rows[piece].size()) {
                piece++;
                at = 0;
            } else if (rows[piece][at].rep() < rank) {
                at++;
            } else {
                break;
            }
        }
        if (piece < rows.size() && rows[piece][at].rep() == rank) {
            continue;
        }
        if (live != NULL && live->isValid(rank) == false) {
            continue;
        }
        PairRecord alone;
        alone.set(rank, rank, 0);
        out.push_back(alone);
    }
}

static void foldSubRange(const PairRecord *pairs, size_t size, std::vector<PairRecord> &out) {
    size_t at = 0;
    while (at < size) {
        size_t end = at + 1;
        while (end < size && PairRecord::sameRepAndMember(pairs[at], pairs[end])) {
            end++;
        }
        PairRecord folded;
        pickBestDiagonal(pairs + at, end - at, folded);
        out.push_back(folded);
        at = end;
    }
}

int lin8pref(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);

    FileUtil::fixRlimitNoFile();
    const NodePlacement node = NodePlacement::resolve(par);
    unsigned int writerNodes = 0;
    size_t ranges = 0;
    uint64_t ranks = 0;
    readRangeShape(par.db1, writerNodes, ranges, ranks);
    const size_t budget = static_cast<size_t>(Util::computeMemory(par.splitMemoryLimit) * 0.95);
    requireEveryNodeDone(par.db1, writerNodes);
    const BucketCounts rangeCounts(par.db1, writerNodes, PairRecord::SUB_RANGE_COUNT, ranges);

    RunDbReader *live = NULL;
    if (par.linclusthashValid.empty() == false) {
        live = new RunDbReader(par.db2, par.linclusthashValid);
        live->open();
    }

    const size_t outEntries = ranges * PairRecord::SUB_RANGE_COUNT;
    const std::string outCounts = par.db3 + "." + SSTR(node.index) + ".counts";
    std::vector<std::vector<uint64_t> > outSubRange(1, std::vector<uint64_t>(outEntries, 0));
    std::vector<uint64_t> outBase(outEntries, 0);

    Timer timer;
    uint64_t read = 0;
    uint64_t kept = 0;
    size_t myRanges = 0;
    for (size_t range = node.index; range < ranges; range += node.count) {
        myRanges++;
    }
    Debug(Debug::INFO) << "Node " << node.index << " of " << node.count << " takes " << myRanges
                       << " of " << ranges << " ranges\n";

    Debug::Progress progress(myRanges);
    std::vector<std::pair<std::string, std::string> > pending;
    size_t pendingFirst = node.index;
    uint64_t pendingBytes = 0;
    std::vector<std::vector<PairRecord> > folded(PairRecord::SUB_RANGE_COUNT);
    std::vector<PairRecord> alone;
    std::vector<size_t> aloneAt(PairRecord::SUB_RANGE_COUNT + 1, 0);
    std::vector<size_t> outAt(PairRecord::SUB_RANGE_COUNT + 1, 0);
    std::vector<std::vector<unsigned char> > packed(par.threads);
    RawArray<PairRecord> pairs;
    for (size_t range = node.index; range < ranges; range += node.count) {
        const std::vector<size_t> starts =
            loadBucket(par.db1, rangeCounts, writerNodes, range, PairRecord::SUB_RANGE_COUNT,
                       BySubRange(ranks, ranges), budget, par.threads, "Range", "raise --ranges",
                       "lin8-pairs", pairs);
        read += pairs.size();
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
        for (size_t i = 0; i < PairRecord::SUB_RANGE_COUNT; i++) {
            SORT_SERIAL(pairs.begin() + starts[i], pairs.begin() + starts[i + 1],
                        PairRecord::byRepAndMember);
        }

#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
        for (size_t i = 0; i < PairRecord::SUB_RANGE_COUNT; i++) {
            folded[i].clear();
            foldSubRange(pairs.begin() + starts[i], starts[i + 1] - starts[i], folded[i]);
        }

        alone.clear();
        addRanksWithNoRows(PairRecord::firstRankOf(range, ranks, ranges),
                           PairRecord::firstRankOf(range + 1, ranks, ranges), folded, live, alone);

        {
            size_t at = 0;
            for (size_t i = 0; i < PairRecord::SUB_RANGE_COUNT; i++) {
                aloneAt[i] = at;
                while (at < alone.size()
                       && PairRecord::subRangeOf(alone[at].rep(), ranks, ranges) == i) {
                    at++;
                }
            }
            aloneAt[PairRecord::SUB_RANGE_COUNT] = alone.size();
        }

        std::vector<uint64_t> &outCount = outSubRange[0];
        outAt[0] = 0;
        for (size_t i = 0; i < PairRecord::SUB_RANGE_COUNT; i++) {
            const size_t mine = folded[i].size() + (aloneAt[i + 1] - aloneAt[i]);
            outAt[i + 1] = outAt[i] + mine;
            outCount[range * PairRecord::SUB_RANGE_COUNT + i] = mine;
        }

        const std::string path = par.db3 + "." + SSTR(node.index) + "." + SSTR(range);
        const std::string tmp = path + ".tmp";
        const int out = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
        if (out < 0 || ftruncate(out, static_cast<off_t>(outAt.back() * PairRecord::DISK_BYTES)) != 0) {
            Debug(Debug::ERROR) << "Cannot write " << tmp << "\n";
            EXIT(EXIT_FAILURE);
        }
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
        for (size_t i = 0; i < PairRecord::SUB_RANGE_COUNT; i++) {
            const size_t mine = outAt[i + 1] - outAt[i];
            if (mine == 0) {
                continue;
            }
            unsigned int thread = 0;
#ifdef OPENMP
            thread = static_cast<unsigned int>(omp_get_thread_num());
#endif
            std::vector<unsigned char> &into = packed[thread];
            into.resize(mine * PairRecord::DISK_BYTES);
            size_t wrote = 0;
            size_t row = 0;
            size_t next = aloneAt[i];
            while (row < folded[i].size()) {
                while (next < aloneAt[i + 1] && alone[next].rep() < folded[i][row].rep()) {
                    alone[next++].pack(&into[wrote++ * PairRecord::DISK_BYTES]);
                }
                folded[i][row++].pack(&into[wrote++ * PairRecord::DISK_BYTES]);
            }
            while (next < aloneAt[i + 1]) {
                alone[next++].pack(&into[wrote++ * PairRecord::DISK_BYTES]);
            }
            size_t got = 0;
            const size_t bytes = mine * PairRecord::DISK_BYTES;
            while (got < bytes) {
                const off_t at = static_cast<off_t>(outAt[i] * PairRecord::DISK_BYTES + got);
                const ssize_t put = pwrite(out, into.data() + got, bytes - got, at);
                if (put <= 0) {
                    Debug(Debug::ERROR) << "Cannot write " << tmp << "\n";
                    EXIT(EXIT_FAILURE);
                }
                got += static_cast<size_t>(put);
            }
        }
        kept += outAt.back();
        sync_file_range(out, 0, 0, SYNC_FILE_RANGE_WRITE);
        if (close(out) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << tmp << "\n";
            EXIT(EXIT_FAILURE);
        }
        pending.push_back(std::make_pair(tmp, path));
        pendingBytes += outAt.back() * PairRecord::DISK_BYTES;
        // batched like a chunk: a restart redoes the whole pass, so a range needs no flush of its own
        if (pendingBytes >= PUBLISH_BATCH_BYTES || pending.size() >= PUBLISH_BATCH_FILES
            || range + node.count >= ranges) {
            publishAllAtomically(pending, par.threads);
            if (par.removeTmpFiles) {
                dropConsumed(par.db1, writerNodes, pendingFirst, range + 1, node.count);
            }
            pendingFirst = range + node.count;
            pendingBytes = 0;
        }
        progress.updateProgress();
    }
    writeSubBucketCounts(outCounts, outBase, outSubRange, ranges);

    const std::string shapeTmp = par.db3 + "." + SSTR(node.index) + ".tmp";
    FILE *shape = FileUtil::openAndDelete(shapeTmp.c_str(), "w");
    fprintf(shape, "nodes\t%u\nranges\t%zu\nranks\t%zu\n", node.count, ranges, (size_t) ranks);
    if (fclose(shape) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << shapeTmp << "\n";
        EXIT(EXIT_FAILURE);
    }
    FileUtil::publishAtomically(shapeTmp, par.db3);
    markNodeDone(par.db3, node.index);
    if (live != NULL) {
        live->close();
        delete live;
    }
    Debug(Debug::INFO) << "Read " << read << " pairs, kept " << kept << " in " << timer.lap() << "\n";
    return EXIT_SUCCESS;
}
