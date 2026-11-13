#include "S3Client.h"
#include "Debug.h"
#include "FileUtil.h"
#include "Util.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

const char *binary_name = "test_s3client";

static void check(bool condition, const std::string &what) {
    if (condition == false) {
        printf("FAIL %s\n", what.c_str());
        exit(EXIT_FAILURE);
    }
}

// GET Object example of the AWS documentation "Signature Calculations for the Authorization Header"
static void testSignature() {
    std::vector<std::pair<std::string, std::string> > headers;
    headers.push_back(std::make_pair("host", "examplebucket.s3.amazonaws.com"));
    headers.push_back(std::make_pair("range", "bytes=0-9"));
    headers.push_back(std::make_pair("x-amz-content-sha256", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
    headers.push_back(std::make_pair("x-amz-date", "20130524T000000Z"));
    std::string authorization = S3Client::getAuthorization("GET", "/test.txt", headers, "us-east-1", "AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY");
    printf("%s\n", authorization.c_str());
    check(authorization == "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request,"
                           "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date,"
                           "Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41", "GET Object signature");
    printf("OK sigv4\n");
}

static std::string readLocal(int fd, size_t offset, size_t size) {
    std::string data(size, '\0');
    check(pread(fd, &data[0], size, offset) == static_cast<ssize_t>(size), "local read");
    return data;
}

// put, get, head, ranges, segments, claims and removal below <prefix>/<run id>/s3test
static void testRoundtrip(const std::string &localFile) {
    const std::string base = "s3test/" + SSTR(getpid());
    S3Client::put(base + "/small", "hello", 5);
    std::string data;
    size_t size = 0;
    check(S3Client::get(base + "/small", data) && data == "hello", "get");
    check(S3Client::head(base + "/small", &size) && size == 5, "head");
    char range[3];
    S3Client::getRange(base + "/small", 1, 3, range);
    check(memcmp(range, "ell", 3) == 0, "getRange");
    check(S3Client::putIfAbsent(base + "/small", "x", 1) == false, "putIfAbsent on an existing object");
    check(S3Client::putIfAbsent(base + "/claim", "a", 1) && S3Client::putIfAbsent(base + "/claim", "b", 1) == false, "putIfAbsent");
    check(S3Client::get(base + "/missing", data) == false && S3Client::head(base + "/missing", NULL) == false, "404");

    std::string emptyFile = localFile + ".empty";
    FileUtil::writeFile(emptyFile, (const unsigned char *) "", 0);
    S3Client::putFile(base + "/empty", emptyFile);
    check(S3Client::getSize(base + "/empty") == 0 && S3Client::get(base + "/empty", data) && data.empty(), "empty file");
    FileUtil::remove(emptyFile.c_str());

    int fd = open(localFile.c_str(), O_RDONLY);
    check(fd >= 0, "open " + localFile);
    size_t fileSize = FileUtil::getFileSize(localFile);
    S3Client::putFile(base + "/file", localFile);
    check(S3Client::getSize(base + "/file") == fileSize, "getSize");
    size_t segment = 256 * 1024 * 1024;
    std::vector<std::pair<size_t, size_t> > ranges;
    ranges.push_back(std::make_pair(0, std::min<size_t>(fileSize, 1000)));
    ranges.push_back(std::make_pair(fileSize - std::min<size_t>(fileSize, 777), std::min<size_t>(fileSize, 777)));
    if (fileSize > segment + 1000) {
        ranges.push_back(std::make_pair(segment - 1000, 2000));
    }
    for (size_t i = 0; i < ranges.size(); i++) {
        std::string remote(ranges[i].second, '\0');
        S3Client::getRange(base + "/file", ranges[i].first, ranges[i].second, &remote[0]);
        check(remote == readLocal(fd, ranges[i].first, ranges[i].second), "getRange " + SSTR(ranges[i].first));
    }
    std::string copy = localFile + ".copy";
    FILE *copyHandle = FileUtil::openAndDelete(copy.c_str(), "w");
    check(ftruncate(fileno(copyHandle), fileSize) == 0 && fclose(copyHandle) == 0, "sparse copy");
    S3Client::getRanges(base + "/file", ranges, copy);
    int copyFd = open(copy.c_str(), O_RDONLY);
    for (size_t i = 0; i < ranges.size(); i++) {
        check(readLocal(copyFd, ranges[i].first, ranges[i].second) == readLocal(fd, ranges[i].first, ranges[i].second), "getRanges " + SSTR(ranges[i].first));
    }
    close(copyFd);
    FileUtil::remove(copy.c_str());
    check(S3Client::get(base + "/file", data) && data == readLocal(fd, 0, fileSize), "get of a segmented file");
    close(fd);

    S3Client::remove(base + "/small");
    S3Client::remove(base + "/claim");
    S3Client::remove(base + "/empty");
    S3Client::remove(base + "/file");
    check(S3Client::head(base + "/small", NULL) == false && S3Client::head(base + "/file", NULL) == false
          && (fileSize <= segment || S3Client::head(base + "/file/0", NULL) == false), "remove");
    printf("OK roundtrip\n");
}

int main(int argc, const char **argv) {
    if (argc < 2 || strcmp(argv[1], "sigv4") == 0) {
        testSignature();
        return EXIT_SUCCESS;
    }
    const char *bucket = getenv("S3_TEST_BUCKET");
    const char *runId = getenv("MMSEQS_RUN_ID");
    check(bucket != NULL && runId != NULL, "S3_TEST_BUCKET and MMSEQS_RUN_ID are needed");
    S3Client::init(bucket, runId);
    if (strcmp(argv[1], "putifabsent") == 0 && argc == 3) {
        char host[256] = "";
        gethostname(host, sizeof(host) - 1);
        std::string body = std::string(host) + "\t" + SSTR(getpid()) + "\t" + SSTR(time(NULL));
        printf("%s\n", S3Client::putIfAbsent(argv[2], body.data(), body.size()) ? "won" : "lost");
    } else if (strcmp(argv[1], "roundtrip") == 0 && argc == 3) {
        testRoundtrip(argv[2]);
    } else {
        printf("Usage: test_s3client [sigv4 | putifabsent <object> | roundtrip <local file>]\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
