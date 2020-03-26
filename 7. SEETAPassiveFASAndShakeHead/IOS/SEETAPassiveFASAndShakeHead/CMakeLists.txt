cmake_minimum_required(VERSION 3.0)
project(SeetaPassiveFaceAntiSpoofing100)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

# library in subdir
SET(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/SeetaPassiveFaceAntiSpoofing)
include_directories(${SRC_DIR})
include_directories(${HOLIDAY_INCLUDE_DIRS})
include_directories(${PE_INCLUDE_DIRS})
include_directories(${EBD_INCLUDE_DIRS})
include_directories(${FR_INCLUDE_DIRS})

file(GLOB ROOT_SRC_FILES
        ${SRC_DIR}/*.c
        ${SRC_DIR}/*.cc
        ${SRC_DIR}/*.cpp
        )

file(GLOB_RECURSE SEETA_SRC_FILES
        ${SRC_DIR}/seeta/*.c
        ${SRC_DIR}/seeta/*.cc
        ${SRC_DIR}/seeta/*.cpp
        )

file(GLOB INCLUDE_FILES
        ${SRC_DIR}/VIPL*.h
        ${SRC_DIR}/VIPL*.hh
        ${SRC_DIR}/VIPL*.hpp
        )

file(GLOB_RECURSE SEETA_QA_FILES
    ${SRC_DIR}/quality/*.c
    ${SRC_DIR}/quality/*.cc
    ${SRC_DIR}/quality/*.cpp
    )

list(APPEND SRC_FILES ${SEETA_SRC_FILES})
list(APPEND SRC_FILES ${ROOT_SRC_FILES})
list(APPEND SRC_FILES ${SEETA_QA_FILES})

# add library
add_library(${PROJECT_NAME} STATIC ${SRC_FILES})

# build dir
set(OUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/release)

set(LIBRARY_OUTPUT_PATH ${OUT_DIR})

# add command to copy header files
add_custom_command(TARGET ${PROJECT_NAME}
        PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT_DIR}/include
        )

# copy every header files
foreach(header ${INCLUDE_FILES})
    add_custom_command(TARGET ${PROJECT_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header} ${OUT_DIR}/include
            )
endforeach()

