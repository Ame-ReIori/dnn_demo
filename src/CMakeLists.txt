cmake_minimum_required(VERSION 2.8)

aux_source_directory(. DIR_LIB_SRCS)
aux_source_directory(./optimizer DIR_LIB_SRCS)
aux_source_directory(./loss DIR_LIB_SRCS)
aux_source_directory(./layer DIR_LIB_SRCS)

add_library(DNNLib ${DIR_LIB_SRCS})