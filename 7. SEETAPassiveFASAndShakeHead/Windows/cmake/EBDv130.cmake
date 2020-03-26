# find EBDv130
#<NAME>_FOUND
#<NAME>_INCLUDE_DIRS or <NAME>_INCLUDES
#<NAME>_LIBRARIES or <NAME>_LIBRARIES or <NAME>_LIBS
#<NAME>_DEFINITIONS

set(EBD_VERSION 130)
set(EBD_HOME ${SOLUTION_DIR}/EBDv${EBD_VERSION})
set(EBD_NAME VIPLEyeBlinkDetector)

set(EBD_INCLUDE_DIRS ${EBD_HOME}/include)
set(EBD_INCLUDES ${EBD_INCLUDE_DIRS})

if (${PLATFORM} STREQUAL "x86")
    set(EBD_LIBRARY_HOME ${EBD_HOME}/lib/x86)
else ()
    set(EBD_LIBRARY_HOME ${EBD_HOME}/lib/x64)
endif ()

if (${CONFIGURATION} STREQUAL "Debug")
    set(EBD_LIBRARIES
        ${EBD_LIBRARY_HOME}/lib${EBD_NAME}${EBD_VERSION}.so
    )
else ()
    set(EBD_LIBRARIES
        ${EBD_LIBRARY_HOME}/lib${EBD_NAME}${EBD_VERSION}.so
    )
endif ()

set(EBD_LIBS ${EBD_LIBRAIES})

foreach (inc ${EBD_INCLUDE_DIRS})
    message(STATUS "PoseEstimation include: " ${inc})
endforeach ()
foreach (lib ${EBD_LIBRARIES})
    message(STATUS "PoseEstimation library: " ${lib})
endforeach ()

