# for supporting ORZ resources

option(USE_ORZ_RESOURCES  "Using ORZ resources" ON)

if (USE_ORZ_RESOURCES)
    add_definitions(-DUSE_ORZ_RESOURCES)
    message(STATUS "Using ORZ resources: ON")
else()
    message(STATUS "Using ORZ resources: OFF")
endif()

if ("${ORZ_RESOURCES_EXE}" STREQUAL "" OR
    "${ORZ_RESOURCES_EXE}" STREQUAL "ORZ_RESOURCES_EXE-NOTFOUND")
    message(STATUS "Detecting orc file compiler")
    find_program(ORZ_RESOURCES_EXE orz_resources)
    if ("${ORZ_RESOURCES_EXE}" STREQUAL "ORZ_RESOURCES_EXE-NOTFOUND")
        message(STATUS "Detecting orc file compiler - not found")
    else()
        message(STATUS "Detecting orc file compiler - found ${ORZ_RESOURCES_EXE}")
    endif()
endif()

function(add_orz_resources var_output_dir var_headers var_sources)
    # find compiler
    if ("${ORZ_RESOURCES_EXE}" STREQUAL "ORZ_RESOURCES_EXE-NOTFOUND")
        message(FATAL_ERROR "
Can not find orc compiler: orz_resources
Please try install ORZ: OpenRoleZoo to fix this problem
")
    endif()

    # set names
    set(OUTPUT_SUB_DIR .resources)
    set(OUTPUT_FILENAME orz_resources)
    # checking orc file
    file(GLOB FOUND_ORZ_RESOURCES_ORC
            "${var_output_dir}/${OUTPUT_FILENAME}.orc")

    list(LENGTH FOUND_ORZ_RESOURCES_ORC FOUND_ORZ_RESOURCES_ORC_SIZE)
    # outputing orc file
    if (${FOUND_ORZ_RESOURCES_ORC_SIZE} LESS 1)
        message(STATUS "Generating empty orc file: ${var_output_dir}/${OUTPUT_FILENAME}.orc")
        file(WRITE "${var_output_dir}/${OUTPUT_FILENAME}.orc" "# Add every line to the format /url:path.\n")
    endif()

    set(${var_headers})
    set(${var_sources})

    list(APPEND ${var_headers} "${var_output_dir}/${OUTPUT_SUB_DIR}/${OUTPUT_FILENAME}.h")
    list(APPEND ${var_sources} "${var_output_dir}/${OUTPUT_SUB_DIR}/${OUTPUT_FILENAME}.c")

    include_directories("${var_output_dir}/${OUTPUT_SUB_DIR}")

    # compiling orc file
    add_custom_command(
            OUTPUT "${var_output_dir}/${OUTPUT_SUB_DIR}/${OUTPUT_FILENAME}.c"
            COMMAND ${ORZ_RESOURCES_EXE} "${var_output_dir}/${OUTPUT_FILENAME}.orc"
                                  "--out_dir=${var_output_dir}/${OUTPUT_SUB_DIR}"
                                  "--in_dir=${var_output_dir}"
            # "--filename=${OUTPUT_FILENAME}"
            DEPENDS "${var_output_dir}/${OUTPUT_FILENAME}.orc"
            COMMENT "Compiling resources file ${var_output_dir}/${OUTPUT_FILENAME}.orc"
            VERBATIM )

    set_source_files_properties(${${var_headers}} ${${var_sources}} PROPERTIES GENERATED TRUE)
    set(${var_headers} ${${var_headers}} PARENT_SCOPE)
    set(${var_sources} ${${var_sources}} PARENT_SCOPE)
endfunction()