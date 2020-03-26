
# find holiday
#<NAME>_FOUND
#<NAME>_INCLUDE_DIRS or <NAME>_INCLUDES
#<NAME>_LIBRARIES or <NAME>_LIBRARIES or <NAME>_LIBS
#<NAME>_DEFINITIONS

set(HOLIDAY_HOME "${SOLUTION_DIR}/holiday" CACHE STRING "The holiday library root")
set(HOLIDAY_NAME "holiday" CACHE STRING "The holiday library name")
set(HOLIDAY_VERSION "" CACHE STRING "The holiday library version")

set(HOLIDAY_INCLUDE_DIRS ${HOLIDAY_HOME}/include)
set(HOLIDAY_INCLUDES ${HOLIDAY_INCLUDE_DIRS})

if ("${PLATFORM}" STREQUAL "x86")
    set(HOLIDAY_LIBRARY_HOME ${HOLIDAY_HOME}/lib/x86)
elseif ("${PLATFORM}" STREQUAL "x64")
    set(HOLIDAY_LIBRARY_HOME ${HOLIDAY_HOME}/lib/x64)
else ()
    set(HOLIDAY_LIBRARY_HOME ${HOLIDAY_HOME}/lib)
endif ()

if ("${CONFIGURATION}" STREQUAL "Debug")
	file(GLOB_RECURSE HOLIDAY_LIBRARIES
        ${HOLIDAY_LIBRARY_HOME}/lib${HOLIDAY_NAME}${HOLIDAY_VERSION}d.a
        ${HOLIDAY_LIBRARY_HOME}/lib${HOLIDAY_NAME}${HOLIDAY_VERSION}d.so
        # ${HOLIDAY_LIBRARY_HOME}/libopenblas.so.0
        # ${HOLIDAY_LIBRARY_HOME}/libprotobuf.so.9
    )
else ()
	file(GLOB_RECURSE HOLIDAY_LIBRARIES
        ${HOLIDAY_LIBRARY_HOME}/lib${HOLIDAY_NAME}${HOLIDAY_VERSION}.a
        ${HOLIDAY_LIBRARY_HOME}/lib${HOLIDAY_NAME}${HOLIDAY_VERSION}.so
        # ${HOLIDAY_LIBRARY_HOME}/libopenblas.so.0
        # ${HOLIDAY_LIBRARY_HOME}/libprotobuf.so.9
    )
endif ()

set(HOLIDAY_LIBS ${HOLIDAY_LIBRAIES})

foreach (inc ${HOLIDAY_INCLUDE_DIRS})
    message(STATUS "Holiday include: " ${inc})
endforeach ()
foreach (lib ${HOLIDAY_LIBRARIES})
    message(STATUS "Holiday library: " ${lib})
endforeach ()

