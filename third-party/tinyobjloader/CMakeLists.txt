cmake_minimum_required(VERSION 3.15)

project(third-party)

include(ExternalProject)

get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if (NOT isMultiConfig)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
    endif()
    set(build_type_dir ${CMAKE_BUILD_TYPE})
    set(build_type_arg -DCMAKE_BUILD_TYPE=$<CONFIG>)
endif()

ExternalProject_Add(
    tinyobjloader  
    GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
    GIT_TAG        v2.0.0rc8
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/tinyobjloader/build/${build_type_dir}
    INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}
    CMAKE_ARGS ${build_type_arg}
               -D CMAKE_INSTALL_PREFIX=<INSTALL_DIR>)
