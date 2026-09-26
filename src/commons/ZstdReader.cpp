#include "ZstdReader.h"

#include "Debug.h"
#include "FileUtil.h"
#include "Util.h"

ZstdReader::ZstdReader(const std::string &fileName)
        : name(fileName), dstream(NULL), pos(0) {
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

ZstdReader::~ZstdReader() {
    if (dstream != NULL) {
        ZSTD_freeDStream(dstream);
    }
    fclose(file);
}

bool ZstdReader::getLine(std::string &line) {
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

bool ZstdReader::fill() {
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
