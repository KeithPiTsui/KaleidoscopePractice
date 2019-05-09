cmake_minimum_required (VERSION 2.6)
project (Toy)

set (LLVM_CONFIG "/data/llvm_master/llvm-project/install_release/bin/llvm-config")

execute_process (COMMAND ${LLVM_CONFIG} --ldflags OUTPUT_VARIABLE LDFLAGS)

set (COMMON_FLAGS -Wall -Wextra -fno-rtti -fno-exception)
execute_process (COMMAND ${LLVM_CONFIG} --cxxflags OUTPUT_VARIABLE CXXFLAGS)

set (CXXFLAGS ${COMMON_FLAGS} ${CXXFLAGS})

message (STATUS "LLVM_CONFIG: ${LLVM_CONFIG}")
message (STATUS "LDFLAGS: ${LDFLAGS}")
message (STATUS "COMMON_FLAGS: ${COMMON_FLAGS}")
message (STATUS "CXXFLAGS: ${CXXFLAGS}")


# add_executable (toy toy.cpp)
# target_link_libraries(toy )
