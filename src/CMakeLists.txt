# CMake Configuration File for OxLM Project.
cmake_minimum_required(VERSION 2.6)
project(OpenQA)

#############################################
# Directories
#############################################

# Store all binaries in bin/ directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../lib)

# Load additional CMake modules from cmake/ directory
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake)

# Include project path (src/)
include_directories(${PROJECT_SOURCE_DIR})

#############################################
# Libraries
#############################################

# Boost
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.45.0 COMPONENTS serialization program_options filesystem
  system date_time regex REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
# link_directories(${Boost_LIBRARY_DIRS})
# set(link_libraries(train_cnlm ${Boost_LIBRARIES} cnlm)

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# OpenMP
find_package(OpenMP QUIET)

#############################################
# Hackery courtesy of Karl Moritz
#############################################

# Overrides standard add_executable command and by default links all executables
# to boost. This will need changing only should we end up including code not
# reliant on boost. As if that was ever going to happen ...
macro (add_executable _name)
  _add_executable(${ARGV})
  if (TARGET ${_name})
    target_link_libraries(${_name} ${Boost_LIBRARIES})
  endif()
endmacro()

#############################################
# Compiler Settings
#############################################

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

set(LBFGS_FLOAT "32")

# SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -pg")
# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -pg")
# SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g -pg")
# SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g -pg")

#############################################
# Compiler Settings
#############################################

add_subdirectory(common)
add_subdirectory(models/additive)
add_subdirectory(models/flattree)
add_subdirectory(models/additive_avg)

add_library(pugi SHARED pugi/pugixml.cpp)

#############################################
# Executables
#############################################

set(EXECUTABLES dbltrain doctrain extract-vectors extract-comp-vectors
  ted-to-libsvm qqtrain qq-extract-relent qq-extract-text qq-rerank-eval
  bordestrain fastbtrain)
foreach(f ${EXECUTABLES})
  add_executable(${f} ${f}.cc)
  target_link_libraries(${f} basics loaders trainers basemod dualtrain pugi)
  target_link_libraries(${f} modadditive modflattree modadditive_avg)
endforeach(f)
