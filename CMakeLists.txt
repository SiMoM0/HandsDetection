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
set(DATASET_PATH_CM ${PROJECT_SOURCE_DIR}/Dataset/rgb/)
set(BBOX_PATH_CM ${PROJECT_SOURCE_DIR}/Dataset/det)
set(MASK_PATH_CM ${PROJECT_SOURCE_DIR}/Dataset/mask)
set(MODEL_PATH_CM ${PROJECT_SOURCE_DIR}/Model/best.onnx)

add_definitions(-DDATASET_PATH_CM=\"${DATASET_PATH_CM}\")
add_definitions(-DBBOX_PATH_CM=\"${BBOX_PATH_CM}\")
add_definitions(-DMASK_PATH_CM=\"${MASK_PATH_CM}\")
add_definitions(-DMODEL_PATH_CM=\"${MODEL_PATH_CM}\")
