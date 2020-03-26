# library in subdir

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

add_definitions(-DVIPL_EXPORTS)
add_definitions(-DSEETA_EXPORTS)

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
	
file(GLOB_RECURSE SEETA_V6_INCLUDE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp
	)

file(GLOB_RECURSE SEETA_V6_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
	)

list(APPEND SRC_FILES ${SEETA_SRC_FILES})
list(APPEND SRC_FILES ${ROOT_SRC_FILES})
list(APPEND SRC_FILES ${SEETA_V6_SRC_FILES})

include(ORZResources)
add_orz_resources(${CMAKE_CURRENT_SOURCE_DIR} resources_headers resources_sources)
list(APPEND SRC_FILES ${resources_sources})

if (NEED_CHECK)
    file(GLOB_RECURSE ENCRYPTION_SRC_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/encryption/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/encryption/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/encryption/*.cpp
        )
    list(APPEND SRC_FILES ${ENCRYPTION_SRC_FILES})
endif ()

# add library
add_library(${PROJECT_NAME}_SHARED
	SHARED
	${INCLUDE_FILES}
	${SEETA_V6_INCLUDE_FILES}
	${SRC_FILES}
	)

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
set_target_properties(${PROJECT_NAME}_SHARED PROPERTIES OUTPUT_NAME ${PROJECT_NAME}${ENV_SUFFIX})

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

set(LIBRARY_OUTPUT_PATH ${OUT_DIR}/${ENV_LIBRARY_DIR})

# add command to copy header files
add_custom_command(TARGET ${PROJECT_NAME}_SHARED
        PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT_DIR}/${ENV_INCLUDE_DIR}
        )
add_custom_command(TARGET ${PROJECT_NAME}_SHARED
        PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT_DIR}/${ENV_INCLUDE_DIR}/seeta
        )

# copy every header files
foreach(header ${INCLUDE_FILES})
    add_custom_command(TARGET ${PROJECT_NAME}_SHARED
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header} ${OUT_DIR}/${ENV_INCLUDE_DIR}
            )
endforeach()
foreach(header ${SEETA_V6_INCLUDE_FILES})
    add_custom_command(TARGET ${PROJECT_NAME}_SHARED
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header} ${OUT_DIR}/${ENV_INCLUDE_DIR}/seeta
            )
endforeach()

install(TARGETS ${PROJECT_NAME}_SHARED
	RUNTIME DESTINATION ${ENV_RUNTIME_DIR}
	LIBRARY DESTINATION ${ENV_LIBRARY_DIR}
	ARCHIVE DESTINATION ${ENV_ARCHIVE_DIR}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
	DESTINATION ${ENV_HEADER_DIR}
)

install(FILES ${INCLUDE_FILES}
	DESTINATION ${ENV_HEADER_DIR}
)
