#ifndef MMSEQS_ZSTDWRITER_H
#define MMSEQS_ZSTDWRITER_H

#include <cstdio>
#include <string>
#include <vector>

#include <zstd.h>

// Writes a text file, compressing it when the name ends in .zst.
class ZstdWriter {
public:
    ZstdWriter(const std::string &fileName);
    void write(const std::string &data);
    void close();

private:
    void flush(size_t n);

    std::string name;
    FILE *file;
    ZSTD_CStream *cstream;
    std::vector<char> out;
};

#endif
