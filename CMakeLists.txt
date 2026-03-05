cmake_minimum_required(VERSION 3.16)
project(vivid_wavetable)

add_library(wavetable_synth MODULE src/wavetable_synth.cpp)
target_include_directories(wavetable_synth PRIVATE ${VIVID_SRC_DIR}/src)
target_compile_features(wavetable_synth PRIVATE cxx_std_17)
set_target_properties(wavetable_synth PROPERTIES
    PREFIX ""
    SUFFIX "${VIVID_PLUGIN_SUFFIX}"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
