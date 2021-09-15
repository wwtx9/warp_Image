cmake_minimum_required(VERSION 3.17)
project(Homography)

set(CMAKE_CXX_STANDARD 11)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
find_package(OpenCV 3.0 QUIET)

find_package(Eigen3 3.1.0 REQUIRED)

include_directories(
        ${EIGEN3_INCLUDE_DIR}
)


add_executable(Homography main.cpp )
target_link_libraries(Homography ${OpenCV_LIBS} ${EIGEN3_LIBS})
