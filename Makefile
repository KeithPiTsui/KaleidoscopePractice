LLVM_CONFIG?=/data/llvm_master/llvm-project/install_release/bin/llvm-config
CLANG?=/data/llvm_master/llvm-project/install_release/bin/clang++

SRC_DIR?=$(PWD)
LDFLAGS+=$(shell $(LLVM_CONFIG) --ldflags)
COMMON_FLAGS=-Wall -Wextra -fno-rtti -fno-exception 
CXXFLAGS+=$(COMMOM_FLAGS) $(shell $(LLVM_CONFIG) --cxxflags)
CPPFLAGS+=$(shell $(LLVM_CONFIG) --cppflags) -I$(SRC_DIR)

LLVMSYSLIBS=$(shell $(LLVM_CONFIG) --system-libs)

LLVMLIBS=$(shell $(LLVM_CONFIG) --libs)

PROJECT=toy
PROJECT_OBJECTS=toy.o

default: $(PROJECT)

%.o: $(SRC_DIR)/%.cpp
	@echo Compiling $*.cpp
	$(CLANG) -c $(CPPFLAGS) $(CXXFLAGS) $<

$(PROJECT): $(PROJECT_OBJECTS)
	@echo Linking $@
	$(CLANG) -o $@  $^ $(LLVMLIBS) $(CXXFLAGS) $(LDFLAGS) $(LLVMSYSLIBS)

clean::
	rm -f $(PROJECT) $(PROJECT_OBJECTS)
