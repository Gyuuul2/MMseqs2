#ifndef MMSEQS_S3CLIENT_H
#define MMSEQS_S3CLIENT_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

class S3Client {
public:
    static bool isEnabled();
    static void init(const std::string &bucketPrefix, const std::string &runId);
    // below the working directory the relative path, so ranks in different directories share objects
    static std::string getObjectName(const std::string &file);
    // whole objects; path is relative to the run root
    static void put(const std::string &path, const char *data, size_t size);
    // data must be unique to the caller, an ambiguous outcome is resolved by comparing it
    static bool putIfAbsent(const std::string &path, const char *data, size_t size);
    static bool get(const std::string &path, std::string &data);
    static bool head(const std::string &path, size_t *size);
    static void remove(const std::string &path);
    // 256 MiB segment objects of a local file, manifest written last
    static void putFile(const std::string &path, const std::string &localFile, size_t offset = 0, size_t size = SIZE_MAX);
    // segmented or plain objects; any error including 404 is fatal
    static void getRange(const std::string &path, size_t offset, size_t size, char *destination);
    static void getRanges(const std::string &path, const std::vector<std::pair<size_t, size_t> > &ranges, const std::string &localFile);
    static size_t getSize(const std::string &path);

    // SigV4 Authorization of a request without query; headers hold host, x-amz-date and x-amz-content-sha256
    static std::string getAuthorization(const std::string &method, const std::string &path, std::vector<std::pair<std::string, std::string> > headers,
                                        const std::string &region, const std::string &accessKey, const std::string &secretKey);
};

#endif
