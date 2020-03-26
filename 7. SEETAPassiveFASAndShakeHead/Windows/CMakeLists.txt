cmake_minimum_required(VERSION 2.8)
project(SeetaPassiveFaceAntiSpoofing100)

set(SOLUTION_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# gether moduls
list(APPEND CMAKE_MODULE_PATH ${SOLUTION_DIR}/cmake)

# install dir
set(CMAKE_INSTALL_PREFIX ${SOLUTION_DIR}/build CACHE STRING "set install prefix" FORCE)

# set base flags
set(CMAKE_CFLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -Wall -std=c++11")

# option for platform
set(PLATFORM "x64" CACHE STRING "x86 or x64")
set(CONFIGURATION "Release" CACHE STRING "Debug or Release")

# option for lock
option(RELEASE_DOG "Using Dog Lock" OFF)
option(TIME_LOCK "Using Time Lock"  OFF)
option(SIMPLE_LOCK "Using Simple Lock" OFF)
set(TIME_LOCK_YEAR "2017" CACHE STRING "Set limit year")
set(TIME_LOCK_MONTH "9" CACHE STRING "Set limit mouth")
set(TIME_LOCK_DAY "30" CACHE STRING "Set limit day")
option(POST_DOG "Changing the buding dir" OFF)

#set(POST_DOG ON)
#set(RELEASE_DOG ON)
#set(TIME_LOCK ON)
#set(SIMPLE_LOCK ON)

add_definitions(-DTIME_LOCK_YEAR=${TIME_LOCK_YEAR})
add_definitions(-DTIME_LOCK_MONTH=${TIME_LOCK_MONTH})
add_definitions(-DTIME_LOCK_DAY=${TIME_LOCK_DAY})

if (RELEASE_DOG)
    set(NEED_CHECK ON)
    add_definitions(-DRELEASE_DOG)
    add_definitions(-DNEED_CHECK)
    message(STATUS "Dog lock: ON")
else()
    message(STATUS "Dog lock: OFF")
endif()

if (SIMPLE_LOCK)
    set(NEED_CHECK ON)
    add_definitions(-DSIMPLE_LOCK)
    add_definitions(-DNEED_CHECK)
    message(STATUS "Simple lock: ON")
else()
    message(STATUS "Simple lock: OFF")
endif()

if (TIME_LOCK)
    set(NEED_CHECK ON)
    add_definitions(-DTIME_LOCK)
    add_definitions(-DNEED_CHECK)
    message(STATUS "Time lock: ON: " ${TIME_LOCK_YEAR} "-" ${TIME_LOCK_MONTH} "-" ${TIME_LOCK_DAY})
else()
    message(STATUS "Time lock: OFF")
endif()

if (${PLATFORM} STREQUAL "x86")
    message(STATUS "Platform: " ${PLATFORM})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m32")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
else()
    message(STATUS "Platform: " ${PLATFORM})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
endif()

if (${CONFIGURATION} STREQUAL "Debug")
    message(STATUS "Configuration: " ${CONFIGURATION})
    set(CMAKE_CX_FLAGS "${CMAKE_CX_FLAGS} -O0 -g -ggdb")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -ggdb")
else()
    message(STATUS "Configuration: " ${CONFIGURATION})
    set(CMAKE_CX_FLAGS "${CMAKE_CX_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

add_subdirectory(${SOLUTION_DIR}/SeetaPassiveFaceAntiSpoofing)
add_subdirectory(${SOLUTION_DIR}/testit)
