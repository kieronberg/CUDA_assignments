cmake_minimum_required(VERSION 3.28)

find_package(OpenCV REQUIRED)

# Google Test (bundled)
set(GTEST_DIR ${CMAKE_SOURCE_DIR}/../../external/gtest)
add_subdirectory(${GTEST_DIR} ${CMAKE_BINARY_DIR}/gtest)

add_executable(unit_tests
    test_convolution.cpp
    test_utils.cpp
    # CPU-only sources (no CUDA needed for unit tests)
    ../../src/utils/filter_utils.cpp
    ../../src/utils/input_handler.cpp
)

target_include_directories(unit_tests PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ../../src
    ${CMAKE_SOURCE_DIR}/../../external/cxxopts/include
    ${CMAKE_SOURCE_DIR}/../../external/plog/include
)

target_link_libraries(unit_tests PRIVATE
    gtest
    gtest_main
    ${OpenCV_LIBS}
)

# Unit tests don't need CUDA (no kernel calls)
target_compile_definitions(unit_tests PRIVATE UNIT_TEST_NO_CUDA)

include(CTest)
add_test(NAME unit_tests COMMAND unit_tests)
