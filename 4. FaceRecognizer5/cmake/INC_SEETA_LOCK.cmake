# include seeta_lock

# option 
if (MSVC)
set(SEETA_LOCK_ROOT_DIR "D:/3rd/local" CACHE STRING "The ORZ library")
else()
set(SEETA_LOCK_ROOT_DIR "/usr/local" CACHE STRING "The ORZ library")
endif()

message(STATUS "Found SeetaLock in ${SEETA_LOCK_ROOT_DIR}")

message(STATUS "Build with SeetaLock")
include_directories(${SEETA_LOCK_ROOT_DIR}/${ENV_HEADER_DIR})
link_directories(${SEETA_LOCK_ROOT_DIR}/${ENV_ARCHIVE_DIR})
link_directories(${SEETA_LOCK_ROOT_DIR}/lib)

#option
option(SEETA_LOCK_SDK  "Using SeetaLock" OFF)
option(SEETA_LOCK_SDK_ONLINE "Using SeetaLock online"  OFF)
option(SEETA_LOCK_SDK_LOCAL "Using SeetaLock local" OFF)
option(SEETA_LOCK_MODEL "Using SeetaLock lock model" OFF)
#option(TIME_LOCK "Using Time Lock"  OFF)
#set(TIME_LOCK_YEAR "2017" CACHE STRING "Set limit year")
#set(TIME_LOCK_MONTH "9" CACHE STRING "Set limit mouth")
#set(TIME_LOCK_DAY "30" CACHE STRING "Set limit day")
set(SEETA_LOCK_ABILITY "" CACHE STRING "local check ability, like 1002 1003")
set(SEETA_LOCK_FINCID "" CACHE STRING "auto check func_id, like 1002")
set(SEETA_LOCK_KEY "" CACHE STRING "lock key, like XJ37DUTJ")
set(SEETA_LOCK_MODEL_KEY "" CACHE STRING "lock model key, like XJ37DUTJ")


if (SEETA_LOCK_SDK_ONLINE)
    set(SEETA_LOCK_SDK ON)
    add_definitions(-DSEETA_LOCK_SDK_ONLINE)
    message(STATUS "SeetaLock online: ON")
else()
    message(STATUS "SeetaLock online: OFF")
endif()

if (SEETA_LOCK_SDK_LOCAL)
    set(SEETA_LOCK_SDK ON)
    add_definitions(-DSEETA_LOCK_SDK_LOCAL)
    message(STATUS "SeetaLock local: ON")
else()
    message(STATUS "SeetaLock local: OFF")
endif()

if (SEETA_LOCK_MODEL)
    set(SEETA_LOCK_SDK ON)
    add_definitions(-DSEETA_LOCK_MODEL)
    message(STATUS "SeetaLock model: ON")
else()
    message(STATUS "SeetaLock model: OFF")
endif()

if (SEETA_LOCK_ABILITY STREQUAL "")
else()
	add_definitions(-DSEETA_LOCK_ABILITY=${SEETA_LOCK_ABILITY})
    message(STATUS "SeetaLock ability: ${SEETA_LOCK_ABILITY}")
endif()

if (SEETA_LOCK_FINCID STREQUAL "")
else()
	add_definitions(-DSEETA_LOCK_FINCID=${SEETA_LOCK_FINCID})
    message(STATUS "SeetaLock funcid: ${SEETA_LOCK_FINCID}")
endif()

if (SEETA_LOCK_KEY STREQUAL "")
else()
	add_definitions(-DSEETA_LOCK_KEY=${SEETA_LOCK_KEY})
    message(STATUS "SeetaLock key: ${SEETA_LOCK_KEY}")
endif()

if (SEETA_LOCK_MODEL_KEY STREQUAL "")
    set(SEETA_LOCK_MODEL_KEY ${SEETA_LOCK_KEY})
endif()

if (SEETA_LOCK_MODEL_KEY STREQUAL "")
else()
	add_definitions(-DSEETA_LOCK_MODEL_KEY=${SEETA_LOCK_MODEL_KEY})
    message(STATUS "SeetaLock model key: ${SEETA_LOCK_MODEL_KEY}")
endif()

if (SEETA_LOCK_SDK)
    add_definitions(-DSEETA_LOCK_SDK)
    message(STATUS "SeetaLock: ON")
else()
    message(STATUS "SeetaLock: OFF")
endif()
