cmake_minimum_required(VERSION 3.5)

project(algorithm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV:
set(OpenCV_DIR /usr/local/opencv2/lib/cmake/opencv4)
FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIR})

add_executable(algorithm main.cpp anomaly_detector.cpp anomaly_detector.h)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
