#include "S3Client.h"
#include "Debug.h"
#include "FileUtil.h"
#include "Util.h"
#include "MMseqsNode.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <mutex>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#ifdef OPENMP
#include <omp.h>
#endif

#ifdef HAVE_S3
#include <curl/curl.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#endif

static bool enabled = false;

bool S3Client::isEnabled() {
    return enabled;
}

// inside the working directory the relative path, else the absolute path without the leading /
std::string S3Client::getObjectName(const std::string &file) {
    static const std::string workingDirectory = FileUtil::getCurrentWorkingDirectory() + "/";
    std::vector<std::string> parts;
    for (size_t begin = 0; begin <= file.size();) {
        size_t end = std::min(file.find('/', begin), file.size());
        std::string part = file.substr(begin, end - begin);
        if (part == ".." && parts.empty() == false && parts.back() != "..") {
            parts.pop_back();
        } else if (part.empty() == false && part != ".") {
            parts.push_back(part);
        }
        begin = end + 1;
    }
    std::string name;
    for (size_t i = 0; i < parts.size(); i++) {
        name += (i > 0 ? "/" : "") + parts[i];
    }
    if (file.empty() == false && file[0] == '/' && ("/" + name).compare(0, workingDirectory.size(), workingDirectory) == 0) {
        name.erase(0, workingDirectory.size() - 1);
    }
    return name;
}

#ifdef HAVE_S3
static const size_t SEGMENT_SIZE = 256 * 1024 * 1024;
static const unsigned int MAX_ATTEMPTS = 10;
// a prefix that S3 scales answers 503 SlowDown for minutes
static const time_t MAX_SLOWDOWN_SECONDS = 300;

static std::string bucket;
static std::string keyPrefix;
static std::string region;
static std::string endpoint;
static std::string host;
static bool isPathStyle = false;
static double faultRate = 0.0;
static unsigned int latencyMilliseconds = 0;

// counted per attempt, printed at exit
static const char *verbNames[] = { "GET", "HEAD", "PUT", "conditional PUT", "DELETE", "LIST" };
static const char *outcomeNames[] = { "2xx", "404", "412", "other 3xx/4xx", "5xx", "transport errors" };
static size_t verbCounts[6];
static size_t outcomeCounts[6];
static size_t sentBytes = 0;
static size_t receivedBytes = 0;
static size_t requestMicroseconds = 0;

struct Credentials {
    std::string accessKey;
    std::string secretKey;
    std::string sessionToken;

    bool operator==(const Credentials &other) const {
        return accessKey == other.accessKey && secretKey == other.secretKey && sessionToken == other.sessionToken;
    }
};
static Credentials credentials;
static std::mutex credentialsMutex;

// size and segment size of write-once objects, segment size 0 for a single object
static std::map<std::string, std::pair<size_t, size_t> > layouts;
static std::mutex layoutMutex;

static std::string trim(const std::string &value) {
    size_t begin = value.find_first_not_of(" \t\r\n");
    size_t end = value.find_last_not_of(" \t\r\n");
    return (begin == std::string::npos) ? "" : value.substr(begin, end - begin + 1);
}

static bool readCredentialsFile(Credentials &result) {
    const char *file = getenv("AWS_SHARED_CREDENTIALS_FILE");
    const char *home = getenv("HOME");
    std::string path = (file != NULL) ? file : ((home != NULL) ? std::string(home) + "/.aws/credentials" : "");
    if (path.empty() || FileUtil::fileExists(path.c_str()) == false) {
        return false;
    }
    const char *profile = getenv("AWS_PROFILE");
    std::string section = "[" + std::string((profile != NULL) ? profile : "default") + "]";
    std::string content = FileUtil::readFile(path);
    Credentials read;
    bool isSection = false;
    for (size_t begin = 0; begin < content.size();) {
        size_t end = std::min(content.find('\n', begin), content.size());
        std::string line = trim(content.substr(begin, end - begin));
        begin = end + 1;
        if (line.empty() == false && line[0] == '[') {
            isSection = line == section;
        }
        size_t equals = line.find('=');
        if (isSection == false || equals == std::string::npos) {
            continue;
        }
        std::string name = trim(line.substr(0, equals));
        std::string value = trim(line.substr(equals + 1));
        if (name == "aws_access_key_id") {
            read.accessKey = value;
        } else if (name == "aws_secret_access_key") {
            read.secretKey = value;
        } else if (name == "aws_session_token") {
            read.sessionToken = value;
        }
    }
    if (read.accessKey.empty() || read.secretKey.empty()) {
        return false;
    }
    result = read;
    return true;
}

// true if newer credentials than the used ones are there
static bool reloadCredentials(const Credentials &used) {
    std::lock_guard<std::mutex> lock(credentialsMutex);
    Credentials read;
    if ((credentials == used) && readCredentialsFile(read)) {
        credentials = read;
    }
    return (credentials == used) == false;
}

static std::string toHex(const std::string &data) {
    static const char digits[] = "0123456789abcdef";
    std::string hex(2 * data.size(), '0');
    for (size_t i = 0; i < data.size(); i++) {
        hex[2 * i] = digits[static_cast<unsigned char>(data[i]) >> 4];
        hex[2 * i + 1] = digits[static_cast<unsigned char>(data[i]) & 15];
    }
    return hex;
}

static std::string sha256(const std::string &data) {
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int size = 0;
    EVP_Digest(data.data(), data.size(), digest, &size, EVP_sha256(), NULL);
    return std::string((const char *) digest, size);
}

static std::string hmac(const std::string &key, const std::string &data) {
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int size = 0;
    HMAC(EVP_sha256(), key.data(), static_cast<int>(key.size()), (const unsigned char *) data.data(), data.size(), digest, &size);
    return std::string((const char *) digest, size);
}

std::string S3Client::getAuthorization(const std::string &method, const std::string &path, std::vector<std::pair<std::string, std::string> > headers,
                                       const std::string &region, const std::string &accessKey, const std::string &secretKey, const std::string &query) {
    std::sort(headers.begin(), headers.end());
    std::string canonicalHeaders;
    std::string signedHeaders;
    std::string amzDate;
    std::string payloadHash;
    for (size_t i = 0; i < headers.size(); i++) {
        canonicalHeaders += headers[i].first + ":" + headers[i].second + "\n";
        signedHeaders += (i > 0 ? ";" : "") + headers[i].first;
        if (headers[i].first == "x-amz-date") {
            amzDate = headers[i].second;
        } else if (headers[i].first == "x-amz-content-sha256") {
            payloadHash = headers[i].second;
        }
    }
    std::string scope = amzDate.substr(0, 8) + "/" + region + "/s3/aws4_request";
    std::string request = method + "\n" + path + "\n" + query + "\n" + canonicalHeaders + "\n" + signedHeaders + "\n" + payloadHash;
    std::string stringToSign = "AWS4-HMAC-SHA256\n" + amzDate + "\n" + scope + "\n" + toHex(sha256(request));
    std::string key = hmac(hmac(hmac(hmac("AWS4" + secretKey, amzDate.substr(0, 8)), region), "s3"), "aws4_request");
    return "AWS4-HMAC-SHA256 Credential=" + accessKey + "/" + scope + ",SignedHeaders=" + signedHeaders + ",Signature=" + toHex(hmac(key, stringToSign));
}

static std::string encodePath(const std::string &path, bool isQuery = false) {
    std::string encoded;
    for (size_t i = 0; i < path.size(); i++) {
        unsigned char c = path[i];
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '~' || (c == '/' && isQuery == false)) {
            encoded += c;
        } else {
            char hex[4];
            snprintf(hex, sizeof(hex), "%%%02X", c);
            encoded += hex;
        }
    }
    return encoded;
}

struct CurlHandle {
    CURL *curl;
    CurlHandle() : curl(curl_easy_init()) {}
    ~CurlHandle() {
        curl_easy_cleanup(curl);
    }
};

struct Request {
    Request(const char *method, const std::string &path) : method(method), path(path), upload(NULL), uploadFd(-1), uploadOffset(0), uploadSize(0),
                                                           download(NULL), downloadFd(-1), downloadOffset(0), downloadSize(0) {}
    std::string method;
    std::string path;
    std::vector<std::pair<std::string, std::string> > headers;
    // a request with a query goes to the bucket
    std::vector<std::pair<std::string, std::string> > query;
    // a PUT body from memory or a file
    const char *upload;
    int uploadFd;
    size_t uploadOffset;
    size_t uploadSize;
    // a GET body into memory or a file, else into the response
    char *download;
    int downloadFd;
    size_t downloadOffset;
    size_t downloadSize;
};

struct Response {
    long status;
    std::string body;
    std::map<std::string, std::string> headers;
    size_t received;
};

struct Transfer {
    CURL *curl;
    const Request *request;
    Response *response;
    size_t sent;
    bool isTooLong;
};

static size_t onBody(char *data, size_t size, size_t count, void *pointer) {
    Transfer &transfer = *(Transfer *) pointer;
    const Request &request = *transfer.request;
    Response &response = *transfer.response;
    size_t bytes = size * count;
    long status = 0;
    curl_easy_getinfo(transfer.curl, CURLINFO_RESPONSE_CODE, &status);
    if (status < 200 || status >= 300 || (request.download == NULL && request.downloadFd < 0)) {
        response.body.append(data, bytes);
        return bytes;
    }
    if (response.received + bytes > request.downloadSize) {
        transfer.isTooLong = true;
        return 0;
    }
    if (request.download != NULL) {
        memcpy(request.download + response.received, data, bytes);
    } else {
        FileUtil::pwriteAll(request.downloadFd, request.path.c_str(), data, bytes, request.downloadOffset + response.received);
    }
    response.received += bytes;
    return bytes;
}

static size_t onHeader(char *data, size_t size, size_t count, void *pointer) {
    Response &response = *((Transfer *) pointer)->response;
    std::string line(data, size * count);
    size_t colon = line.find(':');
    if (colon != std::string::npos) {
        std::string name = line.substr(0, colon);
        std::transform(name.begin(), name.end(), name.begin(), ::tolower);
        response.headers[name] = trim(line.substr(colon + 1));
    }
    return size * count;
}

static size_t onUpload(char *buffer, size_t size, size_t count, void *pointer) {
    Transfer &transfer = *(Transfer *) pointer;
    const Request &request = *transfer.request;
    size_t bytes = std::min(size * count, request.uploadSize - transfer.sent);
    if (request.upload != NULL) {
        memcpy(buffer, request.upload + transfer.sent, bytes);
    }
    for (size_t done = 0; request.upload == NULL && done < bytes;) {
        ssize_t readSize = pread(request.uploadFd, buffer + done, bytes - done, request.uploadOffset + transfer.sent + done);
        if (readSize < 0 && errno == EINTR) {
            continue;
        }
        if (readSize <= 0) {
            return CURL_READFUNC_ABORT;
        }
        done += readSize;
    }
    transfer.sent += bytes;
    return bytes;
}

static int onSeek(void *pointer, curl_off_t offset, int origin) {
    if (origin != SEEK_SET) {
        return CURL_SEEKFUNC_CANTSEEK;
    }
    ((Transfer *) pointer)->sent = offset;
    return CURL_SEEKFUNC_OK;
}

static std::string getXmlValue(const std::string &xml, const std::string &tag) {
    size_t begin = xml.find("<" + tag + ">");
    size_t end = xml.find("</" + tag + ">");
    if (begin == std::string::npos || end == std::string::npos || end < begin) {
        return "";
    }
    begin += tag.size() + 2;
    return xml.substr(begin, end - begin);
}

static void fail(const Request &request, const Response &response) {
    std::string code = getXmlValue(response.body, "Code");
    std::string message = getXmlValue(response.body, "Message");
    Debug(Debug::ERROR) << "S3 " << request.method << " s3://" << bucket << "/" << keyPrefix << "/" << request.path << " failed with HTTP " << response.status
                        << (code.empty() ? "" : " " + code) << (message.empty() ? "" : ": " + message) << "\n";
    EXIT(EXIT_FAILURE);
}

static unsigned int getRandom() {
    static thread_local unsigned int seed = static_cast<unsigned int>(time(NULL)) ^ static_cast<unsigned int>(getpid()) ^ static_cast<unsigned int>((size_t) &seed);
    return rand_r(&seed);
}

static void printStatistics() {
    std::string requests;
    std::string outcomes;
    for (size_t i = 0; i < 6; i++) {
        requests += (i > 0 ? ", " : "") + std::string(verbNames[i]) + " " + SSTR(verbCounts[i]);
        outcomes += (i > 0 ? ", " : "") + std::string(outcomeNames[i]) + " " + SSTR(outcomeCounts[i]);
    }
    char seconds[32];
    snprintf(seconds, sizeof(seconds), "%.3f s", requestMicroseconds / 1e6);
    std::string values;
    MMseqsNode::addPhaseValue(values, "S3 requests", requests);
    MMseqsNode::addPhaseValue(values, "S3 responses", outcomes);
    MMseqsNode::addPhaseValue(values, "S3 bytes", SSTR(sentBytes) + " up, " + SSTR(receivedBytes) + " down");
    MMseqsNode::addPhaseValue(values, "S3 request time summed over threads", seconds);
    MMseqsNode::printPhaseValues(values);
}

// full jitter
static void sleepBeforeRetry(unsigned int attempt) {
    unsigned int maxMilliseconds = std::min(20000u, 100u << std::min(attempt, 8u));
    usleep((getRandom() % (maxMilliseconds + 1)) * 1000u);
}

// retries transport errors, 5xx, 429 and timeouts; an ambiguous PUT may have been applied
static long perform(const Request &request, Response &response, bool *isAmbiguous = NULL) {
    static thread_local CurlHandle handle;
    CURL *curl = handle.curl;
    const bool isPut = request.method == "PUT";
    bool isConditional = false;
    for (size_t i = 0; i < request.headers.size(); i++) {
        isConditional |= request.headers[i].first == "if-none-match";
    }
    const size_t verb = (request.query.empty() == false) ? 5 : (request.method == "GET") ? 0 : (request.method == "HEAD") ? 1 : (request.method == "DELETE") ? 4 : (isConditional ? 3 : 2);
    const time_t firstAttempt = time(NULL);
    unsigned int failures = 0;
    const bool isFixedDownload = request.download != NULL || request.downloadFd >= 0;
    const std::string uriPath = encodePath(request.query.empty() ? (isPathStyle ? "/" + bucket : "") + "/" + keyPrefix + "/" + request.path : (isPathStyle ? "/" + bucket : "/"));
    std::vector<std::pair<std::string, std::string> > sortedQuery = request.query;
    std::sort(sortedQuery.begin(), sortedQuery.end());
    std::string query;
    for (size_t i = 0; i < sortedQuery.size(); i++) {
        query += (i > 0 ? "&" : "") + encodePath(sortedQuery[i].first, true) + "=" + encodePath(sortedQuery[i].second, true);
    }
    for (unsigned int attempt = 1;; attempt++) {
        Credentials used;
        {
            std::lock_guard<std::mutex> lock(credentialsMutex);
            used = credentials;
        }
        char amzDate[17];
        time_t now = time(NULL);
        struct tm utc;
        gmtime_r(&now, &utc);
        strftime(amzDate, sizeof(amzDate), "%Y%m%dT%H%M%SZ", &utc);
        std::vector<std::pair<std::string, std::string> > headers = request.headers;
        headers.push_back(std::make_pair("host", host));
        headers.push_back(std::make_pair("x-amz-content-sha256", "UNSIGNED-PAYLOAD"));
        headers.push_back(std::make_pair("x-amz-date", std::string(amzDate)));
        if (used.sessionToken.empty() == false) {
            headers.push_back(std::make_pair("x-amz-security-token", used.sessionToken));
        }
        struct curl_slist *list = NULL;
        for (size_t i = 0; i < headers.size(); i++) {
            list = curl_slist_append(list, (headers[i].first + ": " + headers[i].second).c_str());
        }
        list = curl_slist_append(list, ("Authorization: " + S3Client::getAuthorization(request.method, uriPath, headers, region, used.accessKey, used.secretKey, query)).c_str());
        list = curl_slist_append(list, "Expect:");

        response.status = 0;
        response.body.clear();
        response.headers.clear();
        response.received = 0;
        Transfer transfer = { curl, &request, &response, 0, false };
        curl_easy_reset(curl);
        curl_easy_setopt(curl, CURLOPT_URL, (endpoint + uriPath + (query.empty() ? "" : "?" + query)).c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, list);
        curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(curl, CURLOPT_PATH_AS_IS, 1L);
        curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
        curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 256L * 1024);
        curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 30L);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, onBody);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &transfer);
        curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, onHeader);
        curl_easy_setopt(curl, CURLOPT_HEADERDATA, &transfer);
        if (request.method == "HEAD") {
            curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
        } else if (isPut) {
            curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);
            curl_easy_setopt(curl, CURLOPT_READFUNCTION, onUpload);
            curl_easy_setopt(curl, CURLOPT_READDATA, &transfer);
            curl_easy_setopt(curl, CURLOPT_SEEKFUNCTION, onSeek);
            curl_easy_setopt(curl, CURLOPT_SEEKDATA, &transfer);
            curl_easy_setopt(curl, CURLOPT_INFILESIZE_LARGE, (curl_off_t) request.uploadSize);
        } else if (request.method != "GET") {
            curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, request.method.c_str());
        }
        // MMSEQS_S3_LATENCY_MS: a delay before each request; MMSEQS_S3_FAULT_RATE: a 503 or a timeout instead of the request, or a reset after it
        if (latencyMilliseconds > 0) {
            usleep(latencyMilliseconds * 1000u);
        }
        unsigned int fault = (faultRate > 0.0 && getRandom() < faultRate * RAND_MAX) ? 1 + getRandom() % 3 : 0;
        CURLcode code = (fault == 1 || fault == 2) ? CURLE_OK : curl_easy_perform(curl);
        curl_slist_free_all(list);
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response.status);
        if (fault == 1) {
            response.status = 503;
        } else if (fault > 1) {
            code = (fault == 2) ? CURLE_OPERATION_TIMEDOUT : CURLE_RECV_ERROR;
        }
        if (fault == 0 || fault == 3) {
            curl_off_t up = 0;
            curl_off_t down = 0;
            double seconds = 0;
            curl_easy_getinfo(curl, CURLINFO_SIZE_UPLOAD_T, &up);
            curl_easy_getinfo(curl, CURLINFO_SIZE_DOWNLOAD_T, &down);
            curl_easy_getinfo(curl, CURLINFO_TOTAL_TIME, &seconds);
            size_t outcome = (code != CURLE_OK) ? 5 : (response.status >= 500) ? 4 : (response.status / 100 == 2) ? 0 : (response.status == 404) ? 1 : (response.status == 412) ? 2 : 3;
            __sync_fetch_and_add(&verbCounts[verb], 1);
            __sync_fetch_and_add(&outcomeCounts[outcome], 1);
            __sync_fetch_and_add(&sentBytes, (size_t) up);
            __sync_fetch_and_add(&receivedBytes, (size_t) down);
            __sync_fetch_and_add(&requestMicroseconds, (size_t) (seconds * 1e6));
        }
        if (transfer.isTooLong) {
            Debug(Debug::ERROR) << "S3 GET s3://" << bucket << "/" << keyPrefix << "/" << request.path << " returned more than the " << request.downloadSize << " requested bytes\n";
            EXIT(EXIT_FAILURE);
        }
        std::string error;
        std::string errorCode = getXmlValue(response.body, "Code");
        if (code != CURLE_OK) {
            error = curl_easy_strerror(code);
        } else if (response.status >= 500 || response.status == 429) {
            error = "HTTP " + SSTR(response.status) + (errorCode.empty() ? "" : " " + errorCode);
        } else if (response.status / 100 == 2 && isFixedDownload && response.received != request.downloadSize) {
            error = "short read of " + SSTR(response.received) + " bytes";
        } else if (errorCode == "RequestTimeout" || errorCode == "ConditionalRequestConflict" || errorCode == "OperationAborted") {
            error = errorCode;
        } else if ((response.status == 400 || response.status == 403)
                   && (reloadCredentials(used) || errorCode == "ExpiredToken" || errorCode == "TokenRefreshRequired" || request.method == "HEAD")) {
            // a HEAD answer has no body that names the error
            error = "expired credentials";
        }
        if (error.empty()) {
            return response.status;
        }
        if (isAmbiguous != NULL && isPut && (code != CURLE_OK || response.status >= 500)) {
            *isAmbiguous = true;
        }
        const bool isSlowDown = code == CURLE_OK && (response.status >= 500 || response.status == 429);
        failures += isSlowDown ? 0 : 1;
        if (isSlowDown ? time(NULL) - firstAttempt >= MAX_SLOWDOWN_SECONDS : failures == MAX_ATTEMPTS) {
            Debug(Debug::ERROR) << "S3 " << request.method << " s3://" << bucket << "/" << keyPrefix << "/" << request.path << " failed " << attempt << " times in "
                                << time(NULL) - firstAttempt << " s, last: " << error << "\n";
            EXIT(EXIT_FAILURE);
        }
        Debug(Debug::WARNING) << "S3 " << request.method << " s3://" << bucket << "/" << keyPrefix << "/" << request.path << ": " << error << ", retrying\n";
        sleepBeforeRetry(attempt);
    }
}

static std::pair<size_t, size_t> parseLayout(const Response &response) {
    std::map<std::string, std::string>::const_iterator segmentSize = response.headers.find("x-amz-meta-segment-size");
    std::map<std::string, std::string>::const_iterator fileSize = response.headers.find("x-amz-meta-file-size");
    if (segmentSize != response.headers.end() && fileSize != response.headers.end()) {
        return std::make_pair(strtoull(fileSize->second.c_str(), NULL, 10), strtoull(segmentSize->second.c_str(), NULL, 10));
    }
    std::map<std::string, std::string>::const_iterator length = response.headers.find("content-length");
    return std::make_pair((length != response.headers.end()) ? strtoull(length->second.c_str(), NULL, 10) : response.body.size(), (size_t) 0);
}

static bool findLayout(const std::string &path, std::pair<size_t, size_t> &layout) {
    {
        std::lock_guard<std::mutex> lock(layoutMutex);
        std::map<std::string, std::pair<size_t, size_t> >::const_iterator it = layouts.find(path);
        if (it != layouts.end()) {
            layout = it->second;
            return true;
        }
    }
    Request request("HEAD", path);
    Response response;
    long status = perform(request, response);
    if (status == 404) {
        return false;
    }
    if (status != 200) {
        fail(request, response);
    }
    layout = parseLayout(response);
    std::lock_guard<std::mutex> lock(layoutMutex);
    layouts[path] = layout;
    return true;
}

static std::pair<size_t, size_t> getLayout(const std::string &path) {
    std::pair<size_t, size_t> layout;
    if (findLayout(path, layout) == false) {
        Debug(Debug::ERROR) << "Cannot find s3://" << bucket << "/" << keyPrefix << "/" << path << "\n";
        EXIT(EXIT_FAILURE);
    }
    return layout;
}

static void putObject(const Request &request) {
    Response response;
    if (perform(request, response) != 200) {
        fail(request, response);
    }
}

static void deleteObject(const std::string &path) {
    Request request("DELETE", path);
    Response response;
    long status = perform(request, response);
    if (status != 200 && status != 204 && status != 404) {
        fail(request, response);
    }
}

// pieces of at most one segment, in parallel, into memory or a file at the same offsets
static void readRange(const std::string &path, const std::pair<size_t, size_t> &layout, size_t offset, size_t size, char *destination, int fd) {
    if (offset > layout.first || size > layout.first - offset) {
        Debug(Debug::ERROR) << "Cannot read bytes " << offset << " to " << offset + size << " of s3://" << bucket << "/" << keyPrefix << "/" << path << " with " << layout.first << " bytes\n";
        EXIT(EXIT_FAILURE);
    }
    std::vector<size_t> pieceOffsets(1, offset);
    while (pieceOffsets.back() < offset + size) {
        size_t pos = pieceOffsets.back();
        size_t segmentEnd = (layout.second > 0) ? std::min(pos / layout.second * layout.second + layout.second, layout.first) : layout.first;
        pieceOffsets.push_back(std::min(offset + size, segmentEnd));
    }
    const size_t pieces = pieceOffsets.size() - 1;
#pragma omp parallel for schedule(dynamic, 1) if (omp_in_parallel() == 0 && pieces > 1)
    for (size_t i = 0; i < pieces; i++) {
        size_t pos = pieceOffsets[i];
        size_t done = pos - offset;
        size_t segmentBegin = (layout.second > 0) ? pos / layout.second * layout.second : 0;
        size_t bytes = pieceOffsets[i + 1] - pos;
        Request request("GET", (layout.second > 0) ? path + "/" + SSTR(pos / layout.second) : path);
        request.headers.push_back(std::make_pair("range", "bytes=" + SSTR(pos - segmentBegin) + "-" + SSTR(pos - segmentBegin + bytes - 1)));
        request.download = (destination != NULL) ? destination + done : NULL;
        request.downloadFd = fd;
        request.downloadOffset = pos;
        request.downloadSize = bytes;
        Response response;
        long status = perform(request, response);
        if (status != 206 && status != 200) {
            fail(request, response);
        }
    }
}

void S3Client::init(const std::string &bucketPrefix, const std::string &runId) {
    size_t slash = bucketPrefix.find('/');
    bucket = bucketPrefix.substr(0, slash);
    std::string prefix = (slash == std::string::npos) ? "" : bucketPrefix.substr(slash + 1);
    while (prefix.empty() == false && prefix[prefix.size() - 1] == '/') {
        prefix.erase(prefix.size() - 1);
    }
    if (bucket.empty() || runId.empty() || runId.find('/') != std::string::npos) {
        Debug(Debug::ERROR) << "--backend aws needs --aws-bucket and a run id without / that all steps of the run share in MMSEQS_RUN_ID or SLURM_JOB_ID\n";
        EXIT(EXIT_FAILURE);
    }
    keyPrefix = prefix.empty() ? runId : prefix + "/" + runId;
    const char *regionName = getenv("AWS_REGION");
    region = (regionName != NULL && *regionName != '\0') ? regionName : "us-east-1";
    const char *rate = getenv("MMSEQS_S3_FAULT_RATE");
    faultRate = (rate != NULL) ? strtod(rate, NULL) : 0.0;
    const char *latency = getenv("MMSEQS_S3_LATENCY_MS");
    latencyMilliseconds = (latency != NULL) ? strtoul(latency, NULL, 10) : 0;
    const char *endpointUrl = getenv("AWS_ENDPOINT_URL");
    if (endpointUrl != NULL && *endpointUrl != '\0') {
        endpoint = endpointUrl;
        while (endpoint.empty() == false && endpoint[endpoint.size() - 1] == '/') {
            endpoint.erase(endpoint.size() - 1);
        }
        size_t hostBegin = endpoint.find("://");
        if ((endpoint.compare(0, 7, "http://") != 0 && endpoint.compare(0, 8, "https://") != 0) || endpoint.find('/', hostBegin + 3) != std::string::npos) {
            Debug(Debug::ERROR) << "AWS_ENDPOINT_URL " << endpointUrl << " must be http(s)://host[:port]\n";
            EXIT(EXIT_FAILURE);
        }
        host = endpoint.substr(hostBegin + 3);
        isPathStyle = true;
    } else {
        host = bucket + ".s3." + region + ".amazonaws.com";
        endpoint = "https://" + host;
        isPathStyle = false;
    }
    const char *accessKey = getenv("AWS_ACCESS_KEY_ID");
    const char *secretKey = getenv("AWS_SECRET_ACCESS_KEY");
    const char *sessionToken = getenv("AWS_SESSION_TOKEN");
    if (accessKey != NULL && *accessKey != '\0' && secretKey != NULL && *secretKey != '\0') {
        credentials.accessKey = accessKey;
        credentials.secretKey = secretKey;
        credentials.sessionToken = (sessionToken != NULL) ? sessionToken : "";
    } else if (readCredentialsFile(credentials) == false) {
        Debug(Debug::ERROR) << "--backend aws needs credentials in AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY or in AWS_SHARED_CREDENTIALS_FILE\n";
        EXIT(EXIT_FAILURE);
    }
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
        Debug(Debug::ERROR) << "Cannot initialize libcurl\n";
        EXIT(EXIT_FAILURE);
    }
    // the working directory of the run, before any chdir
    getObjectName("");
    atexit(printStatistics);
    // a server that ignores If-None-Match gives a claim two winners
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    char hostName[256] = "";
    gethostname(hostName, sizeof(hostName) - 1);
    const std::string probe = "if-none-match." + std::string(hostName) + "." + SSTR(getpid()) + "." + SSTR(now.tv_sec) + "." + SSTR(now.tv_nsec);
    // putIfAbsent resolves a retried PUT by its data, so the two PUTs differ
    if (putIfAbsent(probe, "1", 1) == false || putIfAbsent(probe, "2", 1)) {
        Debug(Debug::ERROR) << "Cannot use s3://" << bucket << "/" << keyPrefix << ": the S3 server ignores If-None-Match\n";
        EXIT(EXIT_FAILURE);
    }
    deleteObject(probe);
    enabled = true;
}

void S3Client::put(const std::string &path, const char *data, size_t size) {
    Request request("PUT", path);
    request.upload = data;
    request.uploadSize = size;
    putObject(request);
    std::lock_guard<std::mutex> lock(layoutMutex);
    layouts.erase(path);
}

bool S3Client::putIfAbsent(const std::string &path, const char *data, size_t size) {
    Request request("PUT", path);
    request.headers.push_back(std::make_pair("if-none-match", "*"));
    request.upload = data;
    request.uploadSize = size;
    Response response;
    bool isAmbiguous = false;
    long status = perform(request, response, &isAmbiguous);
    if (status == 412 && isAmbiguous) {
        std::string stored;
        return get(path, stored) && stored.size() == size && memcmp(stored.data(), data, size) == 0;
    }
    if (status != 200 && status != 412) {
        fail(request, response);
    }
    return status == 200;
}

bool S3Client::get(const std::string &path, std::string &data) {
    Request request("GET", path);
    Response response;
    long status = perform(request, response);
    if (status == 404) {
        return false;
    }
    if (status != 200) {
        fail(request, response);
    }
    std::pair<size_t, size_t> layout = parseLayout(response);
    if (layout.second == 0) {
        data.swap(response.body);
        return true;
    }
    data.assign(layout.first, '\0');
    readRange(path, layout, 0, layout.first, &data[0], -1);
    return true;
}

bool S3Client::head(const std::string &path, size_t *size) {
    {
        std::lock_guard<std::mutex> lock(layoutMutex);
        layouts.erase(path);
    }
    std::pair<size_t, size_t> layout;
    if (findLayout(path, layout) == false) {
        return false;
    }
    if (size != NULL) {
        *size = layout.first;
    }
    return true;
}

// the manifest first, so no reader finds it without its segments
void S3Client::remove(const std::string &path) {
    std::pair<size_t, size_t> layout;
    if (findLayout(path, layout) == false) {
        return;
    }
    deleteObject(path);
    size_t segments = (layout.second > 0) ? (layout.first + layout.second - 1) / layout.second : 0;
#pragma omp parallel for schedule(dynamic, 1) if (omp_in_parallel() == 0)
    for (size_t i = 0; i < segments; i++) {
        deleteObject(path + "/" + SSTR(i));
    }
    std::lock_guard<std::mutex> lock(layoutMutex);
    layouts.erase(path);
}

// ListObjectsV2, a request per 1000 names
void S3Client::list(const std::string &path, std::vector<std::string> &names) {
    names.clear();
    const std::string prefix = keyPrefix + "/" + path + "/";
    std::string token;
    do {
        Request request("GET", path + "/");
        request.query.push_back(std::make_pair("list-type", "2"));
        request.query.push_back(std::make_pair("prefix", prefix));
        if (token.empty() == false) {
            request.query.push_back(std::make_pair("continuation-token", token));
        }
        Response response;
        if (perform(request, response) != 200) {
            fail(request, response);
        }
        for (size_t begin = response.body.find("<Key>"); begin != std::string::npos; begin = response.body.find("<Key>", begin)) {
            begin += 5;
            size_t end = response.body.find("</Key>", begin);
            if (end == std::string::npos || response.body.compare(begin, prefix.size(), prefix) != 0) {
                fail(request, response);
            }
            names.push_back(response.body.substr(begin + prefix.size(), end - begin - prefix.size()));
            begin = end;
        }
        token = (getXmlValue(response.body, "IsTruncated") == "true") ? getXmlValue(response.body, "NextContinuationToken") : "";
    } while (token.empty() == false);
}

void S3Client::putFile(const std::string &path, const std::string &localFile, size_t offset, size_t size) {
    int fd = open(localFile.c_str(), O_RDONLY | O_CLOEXEC);
    struct stat sb;
    if (fd < 0 || fstat(fd, &sb) != 0 || offset > static_cast<size_t>(sb.st_size) || (size != SIZE_MAX && size > sb.st_size - offset)) {
        Debug(Debug::ERROR) << "Cannot read " << localFile << " to upload it to s3://" << bucket << "/" << keyPrefix << "/" << path << "\n";
        EXIT(EXIT_FAILURE);
    }
    if (size == SIZE_MAX) {
        size = sb.st_size - offset;
    }
    size_t segments = (size > SEGMENT_SIZE) ? (size + SEGMENT_SIZE - 1) / SEGMENT_SIZE : 0;
#pragma omp parallel for schedule(dynamic, 1) if (omp_in_parallel() == 0)
    for (size_t i = 0; i < std::max<size_t>(segments, 1); i++) {
        Request request("PUT", (segments > 0) ? path + "/" + SSTR(i) : path);
        request.uploadFd = fd;
        request.uploadOffset = offset + i * SEGMENT_SIZE;
        request.uploadSize = std::min(SEGMENT_SIZE, size - i * SEGMENT_SIZE);
        putObject(request);
    }
    if (segments > 0) {
        Request manifest("PUT", path);
        manifest.headers.push_back(std::make_pair("x-amz-meta-file-size", SSTR(size)));
        manifest.headers.push_back(std::make_pair("x-amz-meta-segment-size", SSTR(SEGMENT_SIZE)));
        putObject(manifest);
    }
    close(fd);
    std::lock_guard<std::mutex> lock(layoutMutex);
    layouts[path] = std::make_pair(size, (segments > 0) ? SEGMENT_SIZE : 0);
}

void S3Client::getRange(const std::string &path, size_t offset, size_t size, char *destination) {
    if (size > 0) {
        readRange(path, getLayout(path), offset, size, destination, -1);
    }
}

void S3Client::getRanges(const std::string &path, const std::vector<std::pair<size_t, size_t> > &ranges, const std::string &localFile) {
    if (ranges.empty()) {
        return;
    }
    std::pair<size_t, size_t> layout = getLayout(path);
    int fd = open(localFile.c_str(), O_WRONLY | O_CLOEXEC);
    if (fd < 0) {
        Debug(Debug::ERROR) << "Cannot open " << localFile << ": " << strerror(errno) << "\n";
        EXIT(EXIT_FAILURE);
    }
#pragma omp parallel for schedule(dynamic, 1) if (omp_in_parallel() == 0)
    for (size_t i = 0; i < ranges.size(); i++) {
        readRange(path, layout, ranges[i].first, ranges[i].second, NULL, fd);
    }
    if (close(fd) != 0) {
        Debug(Debug::ERROR) << "Cannot close " << localFile << "\n";
        EXIT(EXIT_FAILURE);
    }
}

size_t S3Client::getSize(const std::string &path) {
    return getLayout(path).first;
}

#else
static void notBuilt() {
    Debug(Debug::ERROR) << "MMseqs2 was built without S3 support: --backend aws needs libcurl and the OpenSSL crypto library at build time\n";
    EXIT(EXIT_FAILURE);
}

void S3Client::init(const std::string &, const std::string &) {
    notBuilt();
}

void S3Client::put(const std::string &, const char *, size_t) {
    notBuilt();
}

bool S3Client::putIfAbsent(const std::string &, const char *, size_t) {
    notBuilt();
    return false;
}

bool S3Client::get(const std::string &, std::string &) {
    notBuilt();
    return false;
}

bool S3Client::head(const std::string &, size_t *) {
    notBuilt();
    return false;
}

void S3Client::remove(const std::string &) {
    notBuilt();
}

void S3Client::list(const std::string &, std::vector<std::string> &) {
    notBuilt();
}

void S3Client::putFile(const std::string &, const std::string &, size_t, size_t) {
    notBuilt();
}

void S3Client::getRange(const std::string &, size_t, size_t, char *) {
    notBuilt();
}

void S3Client::getRanges(const std::string &, const std::vector<std::pair<size_t, size_t> > &, const std::string &) {
    notBuilt();
}

size_t S3Client::getSize(const std::string &) {
    notBuilt();
    return 0;
}

std::string S3Client::getAuthorization(const std::string &, const std::string &, std::vector<std::pair<std::string, std::string> >,
                                       const std::string &, const std::string &, const std::string &, const std::string &) {
    notBuilt();
    return "";
}
#endif
