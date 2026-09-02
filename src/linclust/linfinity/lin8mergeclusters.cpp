
#include "Lin8Db.h"

#include <cstring>
#include "Parameters.h"
#include "Debug.h"
#include "FileUtil.h"
#include "Util.h"
#include "Timer.h"
#include "FastSort.h"
#include <algorithm>
#include <cstdio>
#include <vector>
#include "DBWriter.h"
#include "itoa.h"

static const uint64_t LINCLUSTHASH_MAGIC = 0x4C494E4348504153ull;
struct PairFileHeader {
    uint64_t magic;
    uint64_t version;
    uint64_t keyWidth;
    uint64_t pairs;
};

struct JoinRow {
    uint64_t on;
    uint64_t carried;

    static const size_t DISK_BYTES = 16;
    void pack(unsigned char *to) const { memcpy(to, this, DISK_BYTES); }
    void unpack(const unsigned char *from) { memcpy(this, from, DISK_BYTES); }

    static bool byJoinKey(const JoinRow &first, const JoinRow &second) {
        if (first.on != second.on) {
            return first.on < second.on;
        }
        return first.carried < second.carried;
    }
};

static void readJoinRowsFromFile(const std::string &path, std::vector<JoinRow> &into) {
    into.clear();
    FILE *in = fopen(path.c_str(), "r");
    if (in == NULL) {
        return;
    }
    std::vector<JoinRow> buffer(1u << 16);
    size_t read = 0;
    while ((read = fread(buffer.data(), sizeof(JoinRow), buffer.size(), in)) > 0) {
        into.insert(into.end(), buffer.begin(), buffer.begin() + read);
    }
    if (ferror(in) != 0) {
        Debug(Debug::ERROR) << "Cannot read " << path << "\n";
        EXIT(EXIT_FAILURE);
    }
    fclose(in);
}

int lin8mergehashredundancy(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);

    size_t repRankBlocks = 0;
    size_t ranks = 0;
    FILE *shape = fopen(par.db1.c_str(), "r");
    if (shape == NULL || fscanf(shape, "repRankBlocks\t%zu\nranks\t%zu", &repRankBlocks, &ranks) != 2
        || repRankBlocks == 0) {
        Debug(Debug::ERROR) << "Cannot read " << par.db1 << ". Run lin8align2clustmulti first\n";
        EXIT(EXIT_FAILURE);
    }
    fclose(shape);

    const size_t budget = static_cast<size_t>(Util::computeMemory(par.splitMemoryLimit) * 0.95);
    const std::string byMember = par.db3 + ".member";
    const std::string byKept = par.db3 + ".kept";
    const std::string extra = par.db3 + ".extra";
    Timer timer;

    {
        BucketWriter<JoinRow> members(byMember, repRankBlocks, par.threads, budget);
        BucketWriter<JoinRow> kept(byKept, repRankBlocks, par.threads, budget);
        std::vector<uint64_t> nothing(repRankBlocks, 0);
        members.openAt(nothing);
        kept.openAt(nothing);

        Debug(Debug::INFO) << "Routing " << repRankBlocks << " blocks by rank\n";
        Debug::Progress routeProgress(repRankBlocks);
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads)
        for (size_t repRankBlock = 0; repRankBlock < repRankBlocks; repRankBlock++) {
            unsigned int thread = 0;
#ifdef OPENMP
            thread = static_cast<unsigned int>(omp_get_thread_num());
#endif
            std::vector<PairRecord> pairs(1u << 16);
            const std::string path = par.db1 + ".0." + SSTR(repRankBlock);
            FILE *in = fopen(path.c_str(), "r");
            if (in == NULL) {
                Debug(Debug::ERROR) << "Cannot open " << path << "\n";
                EXIT(EXIT_FAILURE);
            }
            size_t read = 0;
            while ((read = readRecords(pairs.data(), pairs.size(), in)) > 0) {
                for (size_t k = 0; k < read; k++) {
                    JoinRow row;
                    row.on = pairs[k].member();
                    row.carried = pairs[k].rep();
                    members.add(thread, row, PairRecord::repRankBlockOf(row.on, ranks, repRankBlocks));
                }
            }
            if (ferror(in) != 0) {
                Debug(Debug::ERROR) << "Cannot read " << path << "\n";
                EXIT(EXIT_FAILURE);
            }
            fclose(in);
            routeProgress.updateProgress();
        }

        FILE *in = fopen(par.db2.c_str(), "r");
        if (in == NULL) {
            Debug(Debug::ERROR) << "Cannot open " << par.db2 << ". Run lin8clusthash first\n";
            EXIT(EXIT_FAILURE);
        }
        PairFileHeader header;
        if (fread(&header, sizeof(PairFileHeader), 1, in) != 1
            || header.magic != LINCLUSTHASH_MAGIC) {
            Debug(Debug::ERROR) << par.db2 << " is not a set of redundancy pairs\n";
            EXIT(EXIT_FAILURE);
        }
        std::vector<JoinRow> redundancy(1u << 16);
        size_t read = 0;
        while ((read = fread(redundancy.data(), sizeof(JoinRow), redundancy.size(), in)) > 0) {
            for (size_t k = 0; k < read; k++) {
                JoinRow row;
                row.on = redundancy[k].carried;
                row.carried = redundancy[k].on;
                kept.add(0, row, PairRecord::repRankBlockOf(row.on, ranks, repRankBlocks));
            }
        }
        if (ferror(in) != 0) {
            Debug(Debug::ERROR) << "Cannot read " << par.db2 << "\n";
            EXIT(EXIT_FAILURE);
        }
        fclose(in);
        members.flushAll(par.threads);
        kept.flushAll(par.threads);
        members.close();
        kept.close();
    }

    uint64_t added = 0;
    {
        BucketWriter<PairRecord> writer(extra, repRankBlocks, par.threads, budget);
        std::vector<uint64_t> nothing(repRankBlocks, 0);
        writer.openAt(nothing);
        Debug(Debug::INFO) << "Putting redundant sequences back into " << repRankBlocks << " blocks\n";
        Debug::Progress backProgress(repRankBlocks);
#pragma omp parallel for schedule(dynamic, 1) num_threads(par.threads) reduction(+ : added)
        for (size_t repRankBlock = 0; repRankBlock < repRankBlocks; repRankBlock++) {
            unsigned int thread = 0;
#ifdef OPENMP
            thread = static_cast<unsigned int>(omp_get_thread_num());
#endif
            std::vector<JoinRow> left;
            std::vector<JoinRow> right;
            readJoinRowsFromFile(byKept + "." + SSTR(repRankBlock), right);
            if (right.empty()) {
                backProgress.updateProgress();
                continue;
            }
            readJoinRowsFromFile(byMember + "." + SSTR(repRankBlock), left);
            SORT_SERIAL(left.begin(), left.end(), JoinRow::byJoinKey);
            SORT_SERIAL(right.begin(), right.end(), JoinRow::byJoinKey);
            size_t at = 0;
            for (size_t i = 0; i < right.size(); i++) {
                while (at < left.size() && left[at].on < right[i].on) {
                    at++;
                }
                for (size_t k = at; k < left.size() && left[k].on == right[i].on; k++) {
                    PairRecord row;
                    row.set(left[k].carried, right[i].carried, 0);
                    writer.add(thread, row, PairRecord::repRankBlockOf(left[k].carried, ranks, repRankBlocks));
                    added++;
                }
            }
            backProgress.updateProgress();
        }
        writer.flushAll(par.threads);
        writer.close();
    }

    uint64_t rows = 0;
    std::vector<std::pair<std::string, std::string> > pending;
    uint64_t pendingBytes = 0;
    for (size_t repRankBlock = 0; repRankBlock < repRankBlocks; repRankBlock++) {
        std::vector<PairRecord> all;
        std::vector<PairRecord> buffer(1u << 16);
        const std::string a = par.db1 + ".0." + SSTR(repRankBlock);
        const std::string b = extra + "." + SSTR(repRankBlock);
        for (int which = 0; which < 2; which++) {
            const std::string path = (which == 0) ? a : b;
            FILE *in = fopen(path.c_str(), "r");
            if (in == NULL) {
                continue;
            }
            size_t read = 0;
            while ((read = readRecords(buffer.data(), buffer.size(), in)) > 0) {
                all.insert(all.end(), buffer.begin(), buffer.begin() + read);
            }
            fclose(in);
        }
        SORT_PARALLEL(all.begin(), all.end(), PairRecord::byRepAndMember);
        const std::string outPath = par.db3 + ".0." + SSTR(repRankBlock);
        const std::string outTmp = outPath + ".tmp";
        FILE *out = FileUtil::openAndDelete(outTmp.c_str(), "w");
        if (all.empty() == false
            && writeRecords(all.data(), all.size(), out) != all.size()) {
            Debug(Debug::ERROR) << "Cannot write " << outTmp << "\n";
            EXIT(EXIT_FAILURE);
        }
        if (fclose(out) != 0) {
            Debug(Debug::ERROR) << "Cannot close " << outTmp << "\n";
            EXIT(EXIT_FAILURE);
        }
        pending.push_back(std::make_pair(outTmp, outPath));
        pendingBytes += all.size() * PairRecord::DISK_BYTES;
        if (pendingBytes >= PUBLISH_BATCH_BYTES || pending.size() >= PUBLISH_BATCH_FILES
            || repRankBlock + 1 == repRankBlocks) {
            publishAllAtomically(pending, par.threads);
            pendingBytes = 0;
        }
        rows += all.size();
    }

    const std::string shapeTmp = par.db3 + ".shape.tmp";
    FILE *shapeOut = FileUtil::openAndDelete(shapeTmp.c_str(), "w");
    fprintf(shapeOut, "repRankBlocks\t%zu\nranks\t%zu\n", repRankBlocks, ranks);
    if (fclose(shapeOut) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << shapeTmp << "\n";
        EXIT(EXIT_FAILURE);
    }
    FileUtil::publishAtomically(shapeTmp, par.db3);

    Debug(Debug::INFO) << "Put back " << added << " redundant sequences, " << rows
                       << " rows in all, in " << timer.lap() << "\n";
    return EXIT_SUCCESS;
}

static void appendDecimalKey(std::string &into, uint64_t key) {
    char buffer[32];
    char *end = Itoa::u64toa_sse2(key, buffer);
    into.append(buffer, end - buffer - 1);
    into.push_back('\n');
}

int lin8createclusterdb(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);

    size_t repRankBlocks = 0;
    size_t ranks = 0;
    FILE *shape = fopen(par.db1.c_str(), "r");
    if (shape == NULL || fscanf(shape, "repRankBlocks\t%zu\nranks\t%zu", &repRankBlocks, &ranks) != 2
        || repRankBlocks == 0) {
        Debug(Debug::ERROR) << "Cannot read " << par.db1 << ". Run lin8align2clustmulti first\n";
        EXIT(EXIT_FAILURE);
    }
    fclose(shape);
    if (ranks > (size_t) DB_KEY_INVALID) {
        Debug(Debug::ERROR) << ranks << " sequences do not fit a cluster key. "
                            << "Rebuild with -DMMSEQS_INT64_IDS=1\n";
        EXIT(EXIT_FAILURE);
    }

    const unsigned int threads = std::max<unsigned int>(1, par.threads);
    DBWriter writer(par.db2.c_str(), par.db2Index.c_str(), threads, par.compressed,
                    Parameters::DBTYPE_CLUSTER_RES);
    writer.open();

    Timer timer;
    uint64_t clusters = 0;
    uint64_t members = 0;
    Debug::Progress progress(repRankBlocks);
#pragma omp parallel for schedule(static, 1) num_threads(threads) reduction(+ : clusters, members)
    for (unsigned int part = 0; part < threads; part++) {
        const size_t from = repRankBlocks * part / threads;
        const size_t until = repRankBlocks * (part + 1) / threads;
        uint64_t lastRep = 0;
        bool haveRep = false;
        std::string entry;
        std::vector<PairRecord> buffer(1u << 16);
        for (size_t repRankBlock = from; repRankBlock < until; repRankBlock++) {
            const std::string path = par.db1 + ".0." + SSTR(repRankBlock);
            FILE *in = fopen(path.c_str(), "r");
            if (in == NULL) {
                Debug(Debug::ERROR) << "Cannot open " << path << ", which repRankBlock " << repRankBlock
                                    << " should have decided\n";
                EXIT(EXIT_FAILURE);
            }
            size_t read = 0;
            while ((read = readRecords(buffer.data(), buffer.size(), in)) > 0) {
                for (size_t k = 0; k < read; k++) {
                    const uint64_t rep = buffer[k].rep();
                    const uint64_t member = buffer[k].member();
                    if (haveRep == false || rep != lastRep) {
                        if (haveRep) {
                            writer.writeData(entry.c_str(), entry.length(), lastRep, part);
                        }
                        if (haveRep && rep < lastRep) {
                            Debug(Debug::ERROR) << "The pairs go back from representative " << lastRep
                                                << " to " << rep << ", so they are not in order\n";
                            EXIT(EXIT_FAILURE);
                        }
                        entry.clear();
                        appendDecimalKey(entry, rep);
                        lastRep = rep;
                        haveRep = true;
                        clusters++;
                    }
                    if (member != rep) {
                        appendDecimalKey(entry, member);
                        members++;
                    }
                }
            }
            if (ferror(in) != 0) {
                Debug(Debug::ERROR) << "Cannot read " << path << "\n";
                EXIT(EXIT_FAILURE);
            }
            fclose(in);
            progress.updateProgress();
        }
        if (haveRep) {
            writer.writeData(entry.c_str(), entry.length(), lastRep, part);
        }
    }
    writer.close(false, false);

    Debug(Debug::INFO) << "Wrote " << clusters << " clusters holding " << (clusters + members)
                       << " sequences in " << timer.lap() << "\n";
    return EXIT_SUCCESS;
}
