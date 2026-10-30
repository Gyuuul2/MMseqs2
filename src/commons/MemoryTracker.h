//
// Created by Martin Steinegger on 6/21/20.
//

#ifndef MMSEQS_MEMORYTRACKER_H
#define MMSEQS_MEMORYTRACKER_H
#include "stddef.h"
class MemoryTracker{
public:
    static size_t getSize() { return totalMemorySizeInst;};
protected:
    static size_t totalMemorySizeInst;
    static void incrementMemory(size_t memorySize) { __sync_fetch_and_add(&totalMemorySizeInst, memorySize); }
    static void decrementMemory(size_t memorySize) { __sync_fetch_and_sub(&totalMemorySizeInst, memorySize); }
};
#endif //MMSEQS_MEMORYTRACKER_H
