# find PEv100
#<NAME>_FOUND
#<NAME>_INCLUDE_DIRS or <NAME>_INCLUDES
#<NAME>_LIBRARIES or <NAME>_LIBRARIES or <NAME>_LIBS
#<NAME>_DEFINITIONS

set(PE_HOME ${SOLUTION_DIR}/PEv110)
set(PE_NAME VIPLPoseEstimation)
set(PE_VERSION 110)

set(PE_INCLUDE_DIRS ${PE_HOME}/include)
set(PE_INCLUDES ${PE_INCLUDE_DIRS})

if (${PLATFORM} STREQUAL "x86")
    set(PE_LIBRARY_HOME ${PE_HOME}/lib/x86)
else ()
    set(PE_LIBRARY_HOME ${PE_HOME}/lib/x64)
endif ()

if (${CONFIGURATION} STREQUAL "Debug")
    set(PE_LIBRARIES
        ${PE_LIBRARY_HOME}/lib${PE_NAME}${PE_VERSION}.so
    )
else ()
    set(PE_LIBRARIES
        ${PE_LIBRARY_HOME}/lib${PE_NAME}${PE_VERSION}.so
    )
endif ()

set(PE_LIBS ${PE_LIBRAIES})

foreach (inc ${PE_INCLUDE_DIRS})
    message(STATUS "PoseEstimation include: " ${inc})
endforeach ()
foreach (lib ${PE_LIBRARIES})
    message(STATUS "PoseEstimation library: " ${lib})
endforeach ()

