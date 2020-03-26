cmake_minimum_required(VERSION 2.8)
project(holiday)

# gether moduls
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# option for platform
set(PLATFORM "x64" CACHE STRING "x86 or x64")
set(CONFIGURATION "Release" CACHE STRING "Debug or Release")

# option for extern
option(WITH_CUDA  		    "Set to ON use CUDA"  OFF)
option(WITH_CUDNN  		    "Set to ON use CUDNN"  OFF)

# option for install
set(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/build CACHE STRING "set install prefix" FORCE)


# check options
if (${PLATFORM} STREQUAL "x86")
    message(STATUS "Platform: " ${PLATFORM})
else()
    message(STATUS "Platform: " ${PLATFORM})
    add_definitions(-D_X64_)
    set(X64_ARCH ON)
endif()

if (${CONFIGURATION} STREQUAL "Debug")
    message(STATUS "Configuration: " ${CONFIGURATION})
    set(CMAKE_BUILD_TYPE "Debug")
else()
    message(STATUS "Configuration: " ${CONFIGURATION})
    set(CMAKE_BUILD_TYPE "Release")
endif()

if (WITH_CUDA) 
	message(STATUS "Use CUDA [Optional]")
	ADD_DEFINITIONS(-DHOLIDAY_GPU)
endif()

if (WITH_CUDNN)
	message(STATUS "Use CUDNN [Optional]")
	ADD_DEFINITIONS(-DGPU_CUDNN)
endif()

# add cuda
if (WITH_CUDA) 
	FIND_PACKAGE(CUDA REQUIRED)
endif()

# set CUDA compiling architecture
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_ARCH  	-gencode arch=compute_30,code=sm_30
                -gencode arch=compute_35,code=sm_35
	            -gencode arch=compute_50,code=sm_50
                -gencode arch=compute_60,code=sm_60)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_ARCH} -std=c++11")

if (${CONFIGURATION} STREQUAL "Debug")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O0 -g ")
endif()


# set flags
if(UNIX)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -fPIC")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -fpermissive -std=c++11")
    if (${CONFIGURATION} STREQUAL "Debug")
        set(CMAKE_C_FLAGS "-O0 -g -ggdb ${CMAKE_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "-O0 -g -ggdb ${CMAKE_CXX_FLAGS}")
    else()
        set(CMAKE_C_FLAGS "-O2 ${CMAKE_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "-O2 ${CMAKE_CXX_FLAGS}")
    endif()
    if (${PLATFORM} STREQUAL "x86")
        set(CMAKE_C_FLAGS "-m32 -msse ${CMAKE_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "-m32 -msse ${CMAKE_CXX_FLAGS}")
    else()
        set(CMAKE_C_FLAGS "-m64 -msse ${CMAKE_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "-m64 -msse ${CMAKE_CXX_FLAGS}")
    endif()
elseif(WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

# add ref library
#if (${PLATFORM} STREQUAL "x86")
#    set(OPENBLAS_LIBRARIES /opt/OpenBLAS32/lib/libopenblas.so.0)
#    set(OPENBLAS_INCLUDE_DIRS /opt/OpenBLAS32/include)
#    set(PROTOBUF_LIBRARIES /opt/protobuf32/lib/libprotobuf.so.9)
#    set(PROTOBUF_INCLUDE_DIRS /opt/protobuf32/include)
#else()
#    set(OPENBLAS_LIBRARIES /opt/OpenBLAS/lib/libopenblas.so.0)
#    set(OPENBLAS_INCLUDE_DIRS /opt/OpenBLAS/include)
#    set(PROTOBUF_LIBRARIES /opt/protobuf/lib/libprotobuf.so.9)
#    set(PROTOBUF_INCLUDE_DIRS /opt/protobuf/include)
#endif()

if (OPENBLAS_INCLUDE_DIRS)
    include_directories(${OPENBLAS_INCLUDE_DIRS})
endif ()
if (PROTOBUF_INCLUDE_DIRS)
    include_directories(${PROTOBUF_INCLUDE_DIRS})
endif ()

# add headers
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src/include_inner)
include_directories(${PROJECT_SOURCE_DIR}/src/proto)
include_directories(${PROJECT_SOURCE_DIR}/src)

# search files
FILE(GLOB_RECURSE INCLUDE_FILES
        ${PROJECT_SOURCE_DIR}/include/*.h
        ${PROJECT_SOURCE_DIR}/include/*.hpp
        )
FILE(GLOB_RECURSE SRC_FILES
        ${PROJECT_SOURCE_DIR}/src/*.c
        ${PROJECT_SOURCE_DIR}/src/*.cpp
        ${PROJECT_SOURCE_DIR}/src/*.cc
        )
FILE(GLOB_RECURSE SRC_INCLUDE_FILES
        ${PROJECT_SOURCE_DIR}/src/*.h
        ${PROJECT_SOURCE_DIR}/src/*.hpp
        )
FILE(GLOB_RECURSE SRC_CUDA_FILES
        ${PROJECT_SOURCE_DIR}/src/*.cu
        )

# debug log  
# message("INCLUDE_FILES: " ${INCLUDE_FILES})
# message("SRC_FILES: " ${SRC_FILES})
# message("SRC_CUDA_FILES: " ${SRC_CUDA_FILES})

# for compile
if (WITH_CUDA) 
	CUDA_ADD_LIBRARY(${PROJECT_NAME}_SHARED SHARED ${INCLUDE_FILES} ${SRC_INCLUDE_FILES} ${SRC_FILES} ${SRC_CUDA_FILES})
	if (X64_ARCH)
		TARGET_LINK_LIBRARIES(${PROJECT_NAME}_SHARED  ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_cublas_LIBRARY} ${CUDA_curand_LIBRARY})
	else(X64_ARCH)
		TARGET_LINK_LIBRARIES(${PROJECT_NAME}_SHARED  ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
	endif()
else ()
	ADD_LIBRARY(${PROJECT_NAME}_SHARED SHARED ${INCLUDE_FILES} ${SRC_INCLUDE_FILES} ${SRC_FILES})
endif()

# set library properties
if (${CONFIGURATION} STREQUAL "Debug")
    set_target_properties(${PROJECT_NAME}_SHARED PROPERTIES OUTPUT_NAME ${PROJECT_NAME}d)
else()
    set_target_properties(${PROJECT_NAME}_SHARED PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
endif()

set(OUT_DIR ${PROJECT_SOURCE_DIR}/build)
set(LIBRARY_OUTPUT_PATH ${OUT_DIR}/lib/${PLATFORM})

# add command to copy header files
add_custom_command(TARGET ${PROJECT_NAME}_SHARED
        PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT_DIR}/include
        )

# copy every header files
foreach(header ${INCLUDE_FILES})
    add_custom_command(TARGET ${PROJECT_NAME}_SHARED
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header} ${OUT_DIR}/include
            )
endforeach()


# link library
if (UNIX)
    if (OPENBLAS_LIBRARIES)
        message(STATUS "Link local openblas: " ${OPENBLAS_LIBRARIES})
        target_link_libraries(${PROJECT_NAME}_SHARED ${OPENBLAS_LIBRARIES})
    else ()
        target_link_libraries(${PROJECT_NAME}_SHARED -lopenblas)
    endif ()
    if (PROTOBUF_LIBRARIES)
        message(STATUS "Link local protobuf: " ${PROTOBUF_LIBRARIES})
        target_link_libraries(${PROJECT_NAME}_SHARED ${PROTOBUF_LIBRARIES})
    else ()
        target_link_libraries(${PROJECT_NAME}_SHARED -lprotobuf)
    endif ()
    target_link_libraries(${PROJECT_NAME}_SHARED -lpthread)
elseif (WIN32)
    if (X64_ARCH)
        # target_link_libraries(${PROJECT_NAME}_SHARED /LIBPATH:"${PROJECT_SOURCE_DIR}/thirdParty/winlib/x64")
        set(WINLIB_PATH ${PROJECT_SOURCE_DIR}/thirdParty/winlib/x64)
    else(X64_ARCH)
        # target_link_libraries(${PROJECT_NAME}_SHARED /LIBPATH:"${PROJECT_SOURCE_DIR}/thirdParty/winlib/x86")
        set(WINLIB_PATH ${PROJECT_SOURCE_DIR}/thirdParty/winlib/x86)
    endif()
    target_link_libraries(${PROJECT_NAME}_SHARED
        ${WINLIB_PATH}/libprotobuf.lib
        ${WINLIB_PATH}/libopenblas.lib
        )
endif()

# install
install(TARGETS ${PROJECT_NAME}_SHARED DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/${PLATFORM})
install(FILES ${INCLUDE_FILES} DESTINATION ${CMAKE_INSTALL_PREFIX}/include)

if (OPENBLAS_LIBRARIES)
    install(FILES ${OPENBLAS_LIBRARIES} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/${PLATFORM})
endif ()
if (PROTOBUF_LIBRARIES)
    install(FILES ${PROTOBUF_LIBRARIES} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/${PLATFORM})
endif ()

