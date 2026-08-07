# lz4 vendored for the k-mer bucket temp files, frame API only
cmake_minimum_required(VERSION 3.0)
add_library(liblz4_static STATIC
    lib/lz4.c
    lib/lz4hc.c
    lib/lz4frame.c
    lib/xxhash.c
)
target_include_directories(liblz4_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lib)
# the repo vendors its own xxhash, so lz4's copy has to keep its symbols to itself
target_compile_definitions(liblz4_static PRIVATE XXH_NAMESPACE=LZ4_)
set_target_properties(liblz4_static PROPERTIES POSITION_INDEPENDENT_CODE ON)
