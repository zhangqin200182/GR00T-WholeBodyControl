cmake_minimum_required(VERSION 3.18)
project(physx_core)

set(PhysX_DIR /tmp/physx_build/physx CACHE PATH "PhysX SDK root")
set(PhysX_LIB_DIR ${PhysX_DIR}/bin/linux.aarch64/release)

# PhysX 5 SDK — all required libraries (PhysXTask is inside PhysXExtensions)
find_library(PX_FOUNDATION NAMES PhysXFoundation_static_64 PhysXFoundation_64
             PATHS ${PhysX_LIB_DIR} REQUIRED)
find_library(PX_COMMON NAMES PhysXCommon_static_64 PhysXCommon_64
             PATHS ${PhysX_LIB_DIR} REQUIRED)
find_library(PX_EXTENSIONS NAMES PhysXExtensions_static_64 PhysXExtensions_64
             PATHS ${PhysX_LIB_DIR} REQUIRED)
find_library(PX_PHYSICS NAMES PhysX_static_64 PhysX_64
             PATHS ${PhysX_LIB_DIR} REQUIRED)
find_library(PX_COOKING NAMES PhysXCooking_static_64 PhysXCooking_64
             PATHS ${PhysX_LIB_DIR} REQUIRED)

# pybind11: try pip-installed location first (inside container)
set(pybind11_DIR /usr/local/python3.11.15/lib/python3.11/site-packages/pybind11/share/cmake/pybind11
    CACHE PATH "pybind11 cmake config directory")
find_package(pybind11 REQUIRED)

pybind11_add_module(physx_core physx_bindings.cpp)
target_link_libraries(physx_core PRIVATE
    -Wl,--whole-archive ${PX_FOUNDATION} ${PX_COMMON} ${PX_EXTENSIONS} ${PX_PHYSICS} ${PX_COOKING} -Wl,--no-whole-archive
    pthread dl)
target_include_directories(physx_core PRIVATE
    ${PhysX_DIR}/include
    ${PhysX_DIR}/include/physx
    ${PhysX_DIR}/include/physx/extensions)

# C++17 required by PhysX 5
target_compile_features(physx_core PRIVATE cxx_std_17)
