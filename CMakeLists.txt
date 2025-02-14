cmake_minimum_required(VERSION 3.26)
project(BVA_PR2)

#include_directories("/opt/depthai/depthai-c/include/ /opt/depthai/depthai-core/examples/utility /opt/depthai/depthai-c/lib/cmake/depthai/dependencies/include/ /opt/depthai/depthai-c/include/depthai-shared/3rdparty/")
include_directories("/opt/depthai/depthai-c/include/")
include_directories("/opt/depthai/depthai-core/examples/utility")
include_directories("/opt/depthai/depthai-c/lib/cmake/depthai/dependencies/include/")
include_directories("/opt/depthai/depthai-c/include/depthai-shared/3rdparty/")

link_directories("/opt/depthai/depthai-c/lib/")
link_directories("/usr/lib/x86_64-linux-gnu/")

set(CMAKE_CXX_STANDARD 17)

add_executable(BVA_PR2 main.cpp)
target_link_libraries(BVA_PR2 libopencv_core.so libopencv_imgproc.so libopencv_highgui.so libopencv_videoio.so libdepthai-core.so libdepthai-opencv.so)