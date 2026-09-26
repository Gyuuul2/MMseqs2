#include "ZstdWriter.h"

#include "Debug.h"
#include "FileUtil.h"
#include "Util.h"

ZstdWriter::ZstdWriter(const std::string &fileName) : name(fileName), cstream(NULL) {
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

void ZstdWriter::write(const std::string &data) {
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

void ZstdWriter::close() {
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

void ZstdWriter::flush(size_t n) {
    if (n > 0 && fwrite(&out[0], sizeof(char), n, file) != n) {
        Debug(Debug::ERROR) << "Cannot write " << name << "\n";
        EXIT(EXIT_FAILURE);
    }
}
