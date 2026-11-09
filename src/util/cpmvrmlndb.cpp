#include "FileUtil.h"
#include "Parameters.h"
#include "DBReader.h"
#include "MMseqsNode.h"
#include "Debug.h"

#include <unistd.h>

int rmdb(int argc, const char **argv, const Command &command) {
    MMseqsNode::init();
    Parameters &par = Parameters::getInstance();
    par.overrideParameterDescription(par.PARAM_LOCAL_DB_DIR, "Remove <dir>/<DB>.full.<rank>/<DB>; DB must be a name without a directory");
    par.parseParameters(argc, argv, command, true, 0, 0);
    if (par.localDbDir.empty() == false) {
        if (FileUtil::isDbName(par.db1) == false) {
            Debug(Debug::ERROR) << "With --local-db-dir, rmdb takes a database name without a directory\n";
            EXIT(EXIT_FAILURE);
        }
        const std::string localDb = FileUtil::getLocalDbPath(par.localDbDir + "/" + par.db1, MMseqsNode::rank);
        // A failed copy can leave a temporary completion file.
        const std::string dbtypeTmp = localDb + ".dbtype.tmp";
        if (FileUtil::fileExists(dbtypeTmp.c_str())) {
            FileUtil::remove(dbtypeTmp.c_str());
        }
        DBReader<DBKeyType>::removeDb(localDb);
        if (FileUtil::directoryExists(FileUtil::dirName(localDb).c_str()) && rmdir(FileUtil::dirName(localDb).c_str()) != 0) {
            Debug(Debug::ERROR) << "Cannot remove local database dir " << FileUtil::dirName(localDb) << "\n";
            EXIT(EXIT_FAILURE);
        }
        return EXIT_SUCCESS;
    }
    DBReader<DBKeyType>::removeDb(par.db1);
    return EXIT_SUCCESS;
}

int mvdb(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);
    DBReader<DBKeyType>::moveDb(par.db1.c_str(), par.db2.c_str());
    return EXIT_SUCCESS;
}

int cpdb(int argc, const char **argv, const Command &command) {
    MMseqsNode::init();
    Parameters &par = Parameters::getInstance();
    par.overrideParameterDescription(par.PARAM_LOCAL_DB_DIR, "Copy linfinity sequences to <dir>/<dstDB>.full.<rank>/<dstDB>; dstDB must be a name without a directory");
    par.parseParameters(argc, argv, command, true, 0, 0);
    if (par.localDbDir.empty() == false) {
        if (FileUtil::isDbName(par.db2) == false) {
            Debug(Debug::ERROR) << "With --local-db-dir, cpdb takes a destination database name without a directory\n";
            EXIT(EXIT_FAILURE);
        }
        FileUtil::refreshDirectory(par.db1);
        const int dbtype = FileUtil::parseDbType(par.db1.c_str());
        if ((DBReader<DBKeyType>::getExtendedDbtype(dbtype) & Parameters::DBTYPE_EXTENDED_LINFINITY) == 0
            || Parameters::isEqualDbtype(dbtype, Parameters::DBTYPE_AMINO_ACIDS) == false) {
            Debug(Debug::ERROR) << "--local-db-dir copies only linfinity sequence databases (--createdb-mode 3)\n";
            EXIT(EXIT_FAILURE);
        }
        if (FileUtil::directoryExists(par.localDbDir.c_str()) == false && FileUtil::makeDir(par.localDbDir.c_str()) == false
            && FileUtil::directoryExists(par.localDbDir.c_str()) == false) {
            Debug(Debug::ERROR) << "Cannot create local database dir " << par.localDbDir << "\n";
            EXIT(EXIT_FAILURE);
        }
        const std::string localDb = FileUtil::getLocalDbPath(par.localDbDir + "/" + par.db2, MMseqsNode::rank);
        std::vector<std::string> copies;
        size_t memoryBytes = 0;
        FileUtil::copyToLocalDir(par.db1, FileUtil::dirName(localDb), par.threads, copies, memoryBytes, NULL, par.db2);
        return EXIT_SUCCESS;
    }
    DBReader<DBKeyType>::copyDb(par.db1.c_str(), par.db2.c_str());
    return EXIT_SUCCESS;
}

int lndb(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);
    DBReader<DBKeyType>::softlinkDb(par.db1.c_str(), par.db2.c_str());
    return EXIT_SUCCESS;
}

int aliasdb(int argc, const char **argv, const Command &command) {
    Parameters &par = Parameters::getInstance();
    par.parseParameters(argc, argv, command, true, 0, 0);
    std::string alias = FileUtil::baseName(par.db2.c_str());
    DBReader<DBKeyType>::aliasDb(par.db1.c_str(), alias);
    return EXIT_SUCCESS;
}
