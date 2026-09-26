#ifndef MMSEQS_ZSTDREADER_H
#define MMSEQS_ZSTDREADER_H

#include <cstdio>
#include <string>
#include <vector>

#include <zstd.h>

// Reads a text file line by line, transparently decompressing it when the name ends in .zst.
// GzReader does the same for .gz.
class ZstdReader {
public:
    ZstdReader(const std::string &fileName);
    ~ZstdReader();
    bool getLine(std::string &line);

private:
    bool fill();

    std::string name;
    FILE *file;
    ZSTD_DStream *dstream;
    ZSTD_inBuffer input;
    std::vector<char> in;
    std::vector<char> out;
    std::string buffer;
    size_t pos;
};

#endif
