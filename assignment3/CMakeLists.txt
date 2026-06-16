cmake_minimum_required(VERSION 3.28)
project(cuda-webcam-filter LANGUAGES CXX CUDA)

# ---------------------------------------------------------------
# C++ / CUDA standard
# ---------------------------------------------------------------
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)

# ---------------------------------------------------------------
# CUDA architecture — adjust to your GPU
# RTX 20xx → 75 | RTX 30xx → 86 | RTX 40xx → 89 | A100 → 80
# ---------------------------------------------------------------
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures")

# ---------------------------------------------------------------
# Find packages
# ---------------------------------------------------------------
find_package(OpenCV REQUIRED)

# ---------------------------------------------------------------
# External / bundled headers
# ---------------------------------------------------------------
set(EXT_DIR ${CMAKE_SOURCE_DIR}/external)

# ---------------------------------------------------------------
# Source files
# ---------------------------------------------------------------
set(SOURCES
    src/main.cpp
    src/kernels/convolution_kernels.cu   # contains all CUDA kernels
    src/utils/filter_utils.cpp
    src/utils/input_handler.cpp
    src/input_args_parser/input_args_parser.cpp
)

# ---------------------------------------------------------------
# Executable
# ---------------------------------------------------------------
add_executable(cuda-webcam-filter ${SOURCES})

target_include_directories(cuda-webcam-filter PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${EXT_DIR}/cxxopts/include
    ${EXT_DIR}/plog/include
    src
)

target_link_libraries(cuda-webcam-filter PRIVATE
    ${OpenCV_LIBS}
    CUDA::cudart
)

# ---------------------------------------------------------------
# CUDA compile flags
#   --use_fast_math  : faster trig/exp for the HDR kernels
#   --ptxas-options  : report shared-memory usage per kernel
# ---------------------------------------------------------------
target_compile_options(cuda-webcam-filter PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --ptxas-options=-v
    >
)

# ---------------------------------------------------------------
# Unit tests (optional)
# ---------------------------------------------------------------
option(RUN_UNIT_TESTS "Build unit tests" OFF)
if (RUN_UNIT_TESTS)
    enable_testing()
    add_subdirectory(tests/unit_tests)
endif()
