cmake_minimum_required(VERSION 2.8)
project(Hands_Detection)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCES 
    HandDetection/src/detection.cpp
    HandDetection/src/utils.cpp
    HandDetection/src/segmentation.cpp
    HandDetection/test/test.cpp)

add_executable(main ${SOURCES})

target_link_libraries(main ${OpenCV_LIBS})
set(DATASET_PATH ${PROJECT_SOURCE_DIR}/Dataset/rgb/)
set(BBOX_PATH ${PROJECT_SOURCE_DIR}/Dataset/det)
set(MODEL_PATH ${PROJECT_SOURCE_DIR}/Model/best.onnx)

add_definitions(-DDATASET_PATH=\"${DATASET_PATH}\")
add_definitions(-DMODEL_PATH=\"${MODEL_PATH}\")
add_definitions(-DBBOX_PATH=\"${BBOX_PATH}\")
