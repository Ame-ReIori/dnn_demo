cmake_minimum_required(VERSION 2.8)

project(dnn_demo)

aux_source_directory(. DIR_SRCS)

add_subdirectory(src)

add_executable(demo ${DIR_SRCS})

target_link_libraries(demo DNNLib)