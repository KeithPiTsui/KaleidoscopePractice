cmake_minimum_required (VERSION 3.10)
project (Toy)


# set llvm-config command to get command arguments for compiling with llvm
set (LLVM_CONFIG "/data/llvm_master/llvm-project/install_release/bin/llvm-config")

# set c++ compiler
set (CMAKE_CXX_COMPILER /data/llvm_master/llvm-project/install_release/bin/clang++)



# set compiler flags for compiler toy.cpp to toy.o
# 1. set common flags
set (COMMON_FLAGS -Wall -Wextra -fno-rtti -fno-exception)
message ("COMMON_FLAGS: ${COMMON_FLAGS}")
add_definitions (${COMMON_FLAGS})

# 2. get llvm-config cxxflags
execute_process (COMMAND ${LLVM_CONFIG} --cxxflags OUTPUT_VARIABLE CXXFLAGS_STRING)
message ("CXXFLAGS_STRING: ${CXXFLAGS_STRING}")

# 3. seperate cxxflags string into list of flag string
string (REPLACE " " ";" CXXFLAGS_LIST ${CXXFLAGS_STRING})
message ("CXXFLAGS_LIST: ${CXXFLAGS_LIST}")

# 3.1 combine COMMON_FLAGS into CXXFLAGS
set (CXXFLAGS_LIST ${COMMON_FLAGS} ${CXXFLAGS_LIST})
message ("CXXFLAGS_LIST combined: ${CXXFLAGS_LIST}")

string (REPLACE ";" " " CXXFLAGS ${CXXFLAGS_LIST})

# 4. get llvm-config CPPLAGS
execute_process (COMMAND ${LLVM_CONFIG} --cppflags OUTPUT_VARIABLE CPPFLAGS_STRING)
message ("CPPFLAGS_STRING: ${CPPFLAGS_STRING}")

# 5. seperate cppflags string into list of flag string
string (REPLACE " " ";" CPPFLAGS_LIST ${CPPFLAGS_STRING})
message ("CPPFLAGS_LIST: ${CPPFLAGS_LIST}")

string (REPLACE ";" " " CPPFLAGS ${CPPFLAGS_LIST})

add_definitions(${CXXFLAGS} ${CPPFLAGS})



# set compiler and linker flags
# 1. get llvm-config ldflags
execute_process (COMMAND ${LLVM_CONFIG} --ldflags OUTPUT_VARIABLE LDFLAGS_STRING)
message ("LDFLAGS_STRING: ${LDFLAGS_STRING}")

# 2. seperate ldflags string into list of flag string
string (REPLACE " " ";" LDFLAGS_LIST ${LDFLAGS_STRING})
message ("LDFLAGS_LIST: ${LDFLAGS_LIST}")

string (REPLACE ";" " " LDFLAGS ${LDFLAGS_LIST})

# 3. get llvm-config system-libs
execute_process (COMMAND ${LLVM_CONFIG} --system-libs OUTPUT_VARIABLE LLVMSYSLIBS_STRING)
message ("LLVMSYSLIBS_STRING: ${LLVMSYSLIBS_STRING}")

# 4. seperate llvmsyslibs string into list of flag string
string (REPLACE " " ";" LLVMSYSLIBS_LIST ${LLVMSYSLIBS_STRING})
message ("LLVMSYSLIBS_LIST: ${LLVMSYSLIBS_LIST}")

string (REPLACE ";" " " LLVMSYSLIBS ${LLVMSYSLIBS_LIST})

# 5. get llvm-config libs
execute_process (COMMAND ${LLVM_CONFIG} --libs OUTPUT_VARIABLE LLVMLIBS_STRING)
message ("LLVMLIBS_STRING: ${LLVMLIBS_STRING}")

# 6. seperate llvmsyslibs string into list of flag string
string (REPLACE " " ";" LLVMLIBS_LIST ${LLVMLIBS_STRING})
message ("LLVMLIBS_LIST: ${LLVMLIBS_LIST}")

string (REPLACE ";" " " LLVMLIBS ${LLVMLIBS_LIST})

set (CMAKE_CXX_FLAGS "${LLVMLIBS} ${CXXFLAGS} ${LDFLAGS} ${LLVMSYSLIBS}")
message ("CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
# include_directories("${PROJECT_SOURCE_DIR}/include")
# include_directories("/data/llvm_master/llvm-project/install_release/include")

add_executable (${PROJECT_NAME} toy.cpp)
