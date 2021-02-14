cmake_minimum_required(VERSION 3.15)

project(vulkan-fruit LANGUAGES CXX)

find_package(Vulkan REQUIRED)
find_package(SDL2 REQUIRED CONFIG)
find_package(tinyobjloader REQUIRED CONFIG)

include(FetchContent)

FetchContent_Declare(
    as-camera-input
    GIT_REPOSITORY https://github.com/pr0g/as-camera-input.git
    GIT_TAG        77e638bbac2d2806ca0020ee0151c832c7f7d5f2)

FetchContent_MakeAvailable(as-camera-input)

add_executable(${PROJECT_NAME} main.cpp as-vulkan.cpp)
target_link_libraries(
  ${PROJECT_NAME} PRIVATE
  Vulkan::Vulkan as-camera-input SDL2::SDL2 SDL2::SDL2main
  tinyobjloader::tinyobjloader)
target_include_directories(
  ${PROJECT_NAME} PRIVATE
  third-party/stb)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_compile_definitions(${PROJECT_NAME}
    PRIVATE $<$<BOOL:${AS_PRECISION_FLOAT}>:AS_PRECISION_FLOAT>
            $<$<BOOL:${AS_PRECISION_DOUBLE}>:AS_PRECISION_DOUBLE>
            $<$<BOOL:${AS_COL_MAJOR}>:AS_COL_MAJOR>
            $<$<BOOL:${AS_ROW_MAJOR}>:AS_ROW_MAJOR>
            -DAS_VULKAN_SDL)
