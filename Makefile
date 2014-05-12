LOCAL_LIBS=/usr/local
EXTRA_LIBS=/data/taipan/karher/extra_lib
override CXXFLAGS += -g -std=c++0x -fopenmp -Ofast -Wall -c -MMD -isystem${LOCAL_LIBS}/include -m64 -isystem/usr/include/eigen3 -isystem${EXTRA_LIBS}/include -Isrc
override LDFLAGS += -g -std=c++0x -fopenmp -Ofast -L${LOCAL_LIBS}/lib64 -L${LOCAL_LIBS}/lib -L${EXTRA_LIBS}/lib -lboost_program_options -lboost_serialization -llbfgs -lboost_filesystem -lboost_system -lboost_iostreams -lboost_regex

DIRS    := src/common src/pugi src/models/additive src/models/flattree
SOURCES := $(foreach dir, $(DIRS), $(wildcard $(dir)/*.cc) $(wildcard $(dir)/*.cpp))
OBJS    := $(patsubst %.cc, %.o, $(SOURCES))
OBJS    := $(patsubst %.cpp, %.o, $(OBJS))
OBJS    := $(foreach o,$(OBJS),build/$(o))

DEPFILES:= $(patsubst %.o, %.P, $(OBJS))

CXX = g++ -DLBFGS_FLOAT=32
# CXX = /usr/local/gcc-4.8.2/bin/g++

#link the executable
all: dbltrain doctrain extract-vectors

dbltrain: $(OBJS) build/src/dbltrain.o
	$(CXX) -o dbltrain $^ $(LDFLAGS)

doctrain: $(OBJS) build/src/doctrain.o
	$(CXX) -o doctrain $^ $(LDFLAGS)

extract-vectors: $(OBJS) build/src/extract-vectors.o
	$(CXX) -o extract-vectors $^ $(LDFLAGS)

ted-to-libsvm: $(OBJS) build/src/ted-to-libsvm.o
	$(CXX) -o ted-to-libsvm $^ $(LDFLAGS)


#generate dependency information and compile
build/%.o : %.cc
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $@ -MF build/$*.P $<
	@sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < build/$*.P >> build/$*.P;

build/%.o : %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $@ -MF build/$*.P $<
	@sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < build/$*.P >> build/$*.P;

#remove all generated files
clean:
	rm -f main
	rm -rf build

#include the dependency information
-include $(DEPFILES)
