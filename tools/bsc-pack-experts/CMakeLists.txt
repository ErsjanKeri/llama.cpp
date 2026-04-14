set(TARGET llama-bsc-pack-experts)
add_executable(${TARGET} bsc-pack-experts.cpp)
# Standalone tool: only uses POSIX I/O + the shared format header in src/
target_include_directories(${TARGET} PRIVATE ../../src)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
