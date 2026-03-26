# VolEsti (volume computation and sampling library)
# counting linear extensions via hmc on order polytopes
# Agnik

project( VolEsti )

CMAKE_MINIMUM_REQUIRED(VERSION 3.11)

set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

set(CMAKE_EXPORT_COMPILE_COMMANDS "ON")

option(DISABLE_NLP_ORACLES "Disable non-linear oracles (used in collocation)" ON)
if(DISABLE_NLP_ORACLES)
  add_definitions(-DDISABLE_NLP_ORACLES)
endif(DISABLE_NLP_ORACLES)

# eigen
option(BUILTIN_EIGEN "Use eigen from ../../external" OFF)
include("../../external/cmake-files/Eigen.cmake")
GetEigen()

if (${CMAKE_VERSION} VERSION_LESS "3.12.0")
  add_compile_options(-D "EIGEN_NO_DEBUG")
else ()
  add_compile_definitions("EIGEN_NO_DEBUG")
endif ()

# boost
include("../../external/cmake-files/Boost.cmake")
GetBoost()

# lp_solve
include("../../external/cmake-files/LPSolve.cmake")
GetLPSolve()

include_directories (BEFORE ../../external)
include_directories (BEFORE ../../external/minimum_ellipsoid)
include_directories (BEFORE ../../include/generators)
include_directories (BEFORE ../../include/volume)
include_directories (BEFORE ../../include)
include_directories (BEFORE ../../include/convex_bodies)
include_directories (BEFORE ../../include/random_walks)
include_directories (BEFORE ../../include/annealing)
include_directories (BEFORE ../../include/ode_solvers)
include_directories (BEFORE ../../include/root_finders)
include_directories (BEFORE ../../include/samplers)
include_directories (BEFORE ../../include/lp_oracles)
include_directories (BEFORE ../../include/nlp_oracles)
include_directories (BEFORE ../../include/misc)
include_directories (BEFORE ../../include/optimization)
include_directories (BEFORE ../../include/preprocess)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(${CMAKE_CXX_FLAGS} "-O3")
add_definitions(${CXX_COVERAGE_COMPILE_FLAGS} "-lm")
add_definitions(${CXX_COVERAGE_COMPILE_FLAGS} "-ldl")
add_definitions(${CXX_COVERAGE_COMPILE_FLAGS} "-DBOOST_NO_AUTO_PTR")

add_executable(count_le count_linear_extensions.cpp)
target_link_libraries(count_le PUBLIC lp_solve)

add_executable(test_trig_oracle test_trigonometric_oracle.cpp)
target_link_libraries(test_trig_oracle PUBLIC lp_solve)

add_executable(benchmark_oracle benchmark_oracle.cpp)
target_link_libraries(benchmark_oracle PUBLIC lp_solve)
