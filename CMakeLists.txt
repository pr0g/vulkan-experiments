cmake_minimum_required(VERSION 3.15)

project(vulkan-fruit LANGUAGES CXX)

find_package(Vulkan REQUIRED)
find_package(SDL2 REQUIRED CONFIG)
find_package(tinyobjloader REQUIRED CONFIG)

include(FetchContent)

FetchContent_Declare(
  as
  GIT_REPOSITORY https://github.com/pr0g/as.git
  GIT_TAG 4bd7768323edfc7d819874439d40ce21a3ea6388)
FetchContent_Declare(
  as-camera-input
  GIT_REPOSITORY https://github.com/pr0g/as-camera-input.git
  GIT_TAG        a779bd361b2351940bf07e70d130530dfe293d27)
FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/pr0g/stb.git
  GIT_TAG        4230eecc3c7416841e89cf1bec1df9abb814e89b)
FetchContent_Declare(
  thh_handles
  GIT_REPOSITORY https://github.com/pr0g/cpp-handle-container.git
  GIT_TAG        7f6c9a3d8c074bed6db9d37cc7a03bb03bed589b)
FetchContent_MakeAvailable(as-camera-input stb thh_handles)

add_executable(${PROJECT_NAME} main.cpp as-vulkan.cpp)
target_link_libraries(
  ${PROJECT_NAME} PRIVATE
  Vulkan::Vulkan as-camera-input stb SDL2::SDL2 SDL2::SDL2main
  tinyobjloader::tinyobjloader thh_handles)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_compile_definitions(${PROJECT_NAME}
  PRIVATE $<$<BOOL:${AS_PRECISION_FLOAT}>:AS_PRECISION_FLOAT>
          $<$<BOOL:${AS_PRECISION_DOUBLE}>:AS_PRECISION_DOUBLE>
          $<$<BOOL:${AS_COL_MAJOR}>:AS_COL_MAJOR>
          $<$<BOOL:${AS_ROW_MAJOR}>:AS_ROW_MAJOR>
          -DAS_VULKAN_SDL)
