cmake_minimum_required(VERSION 3.15)

project(vulkan-fruit LANGUAGES CXX)

find_package(Vulkan REQUIRED)
find_package(SDL2 REQUIRED CONFIG)
find_package(tinyobjloader REQUIRED CONFIG)

include(FetchContent)

FetchContent_Declare(
  as-camera-input
  GIT_REPOSITORY https://github.com/pr0g/as-camera-input.git
  GIT_TAG        777d7895874115bb81933f8e117414bf8efc9d29)
FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/pr0g/stb.git
  GIT_TAG        4230eecc3c7416841e89cf1bec1df9abb814e89b)
FetchContent_Declare(
  thh_handles
  GIT_REPOSITORY https://github.com/pr0g/cpp-handle-container.git
  GIT_TAG        8c30500bd53b75cc88e4a5e71083a506c45b63c8)
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
