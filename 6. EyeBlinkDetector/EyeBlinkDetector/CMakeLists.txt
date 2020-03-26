# library in subdir

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

add_definitions(-DVIPL_EXPORTS)

file(GLOB ROOT_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    )

file(GLOB_RECURSE SEETA_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/seeta/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/seeta/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/seeta/*.cpp
    )

file(GLOB INCLUDE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/VIPL*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/VIPL*.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/VIPL*.hpp
    )

list(APPEND SRC_FILES ${SEETA_SRC_FILES})
list(APPEND SRC_FILES ${ROOT_SRC_FILES})

if (NEED_CHECK)
    file(GLOB_RECURSE ENCRYPTION_SRC_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/encryption/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/encryption/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/encryption/*.cpp
        )
    list(APPEND SRC_FILES ${ENCRYPTION_SRC_FILES})
endif ()

# add library
add_library(${PROJECT_NAME}_SHARED SHARED ${SRC_FILES})

# simple lock
if (SIMPLE_LOCK)
    target_link_libraries(${PROJECT_NAME}_SHARED -lsimple_crypto -lcrypto)
endif()

# link SeetaLock
if (SEETA_LOCK_SDK)
    target_link_libraries(${PROJECT_NAME}_SHARED SeetaLock_static${ENV_SUFFIX})
endif()
# link ORZ
target_link_libraries(${PROJECT_NAME}_SHARED ORZ_static${ENV_SUFFIX})
# link OpenSSL
if (SEETA_LOCK_SDK)
    find_package(OpenSSL REQUIRED)
    include_directories(${OPENSSL_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME}_SHARED ${OPENSSL_CRYPTO_LIBRARY})
    target_link_libraries(${PROJECT_NAME}_SHARED ${OPENSSL_SSL_LIBRARY})
endif()


# more link library
include(holiday)
include_directories(${HOLIDAY_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}_SHARED ${HOLIDAY_LIBRARIES})

# set library properties
if (${CONFIGURATION} STREQUAL "Debug")
    set_target_properties(${PROJECT_NAME}_SHARED PROPERTIES OUTPUT_NAME ${PROJECT_NAME}d)
else()
    set_target_properties(${PROJECT_NAME}_SHARED PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
endif()

# build dir
set(OUT_DIR ${SOLUTION_DIR}/build)
if (POST_DOG AND RELEASE_DOG)
    set(OUT_DIR ${OUT_DIR}_dog)
endif()
if (POST_DOG AND SIMPLE_LOCK)
    set(OUT_DIR ${OUT_DIR}_sim)
endif()
if (POST_DOG AND TIME_LOCK)
    set(OUT_DIR ${OUT_DIR}_${TIME_LOCK_YEAR}_${TIME_LOCK_MONTH}_${TIME_LOCK_DAY})
endif()

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

install(TARGETS ${PROJECT_NAME}_SHARED DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(FILES ${INCLUDE_FILES} DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
