# test

SET(TEST_PROJECT_NAME testit)

file(GLOB_RECURSE TEST_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    )

add_executable(${TEST_PROJECT_NAME} ${TEST_SRC_FILES})

set(OUT_DIR ${SOLUTION_DIR}/build)

include_directories(${OUT_DIR}/include)

if (${CONFIGURATION} STREQUAL "Debug")
    set_target_properties(${TEST_PROJECT_NAME} PROPERTIES OUTPUT_NAME ${TEST_PROJECT_NAME}d)
else()
    set_target_properties(${TEST_PORJECT_NAME} PROPERTIES OUTPUT_NAME ${TEST_PROJECT_NAME})
endif()

#set(EXECUTABLE_OUTPUT_PATH ${OUT_DIR}/bin/${PLATFORM})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin/${PLATFORM})

target_link_libraries(${TEST_PROJECT_NAME} ${PROJECT_NAME}_SHARED)

target_link_libraries(${TEST_PROJECT_NAME} opencv_highgui opencv_imgproc opencv_core)



