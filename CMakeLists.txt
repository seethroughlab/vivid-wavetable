cmake_minimum_required(VERSION 3.16)
project(vivid_wavetable)

if(NOT DEFINED VIVID_SRC_DIR)
  message(FATAL_ERROR "VIVID_SRC_DIR must be set to the vivid-core source directory")
endif()

if(NOT DEFINED VIVID_PLUGIN_SUFFIX)
  set(VIVID_PLUGIN_SUFFIX .dylib)
endif()

option(VIVID_PACKAGE_BUILD_TESTS "Build vivid-wavetable package tests" OFF)

set(VIVID_WAVETABLE_OPS
  wavetable_synth
  poly_voice_allocator
  wavetable_osc
  voice_mixer
  sub_osc
  analog_osc
)

# Sources that need wavetable bank/DSP shared libs
set(VIVID_WAVETABLE_SYNTH_SOURCES
  src/wavetable_synth.cpp
  src/wavetable_bank.cpp
  src/wavetable_dsp.cpp
  ${VIVID_SRC_DIR}/operators/control/envelope/envelope_composable.cpp
  ${VIVID_SRC_DIR}/src/operator_api/filter_dsp.cpp
)
set(VIVID_WAVETABLE_OSC_SOURCES
  src/wavetable_osc.cpp
  src/wavetable_bank.cpp
  src/wavetable_dsp.cpp
)

# miniaudio: decoder only (no device I/O, no threading, no encoding)
add_library(wavetable_miniaudio STATIC src/miniaudio_impl.c)
target_include_directories(wavetable_miniaudio PRIVATE ${VIVID_SRC_DIR}/deps/miniaudio)

foreach(name IN LISTS VIVID_WAVETABLE_OPS)
  if(name STREQUAL wavetable_synth)
    add_library(${name} MODULE ${VIVID_WAVETABLE_SYNTH_SOURCES})
  elseif(name STREQUAL wavetable_osc)
    add_library(${name} MODULE ${VIVID_WAVETABLE_OSC_SOURCES})
  else()
    add_library(${name} MODULE src/${name}.cpp)
  endif()
  target_include_directories(${name} PRIVATE ${VIVID_SRC_DIR}/src ${VIVID_SRC_DIR}/deps/miniaudio ${VIVID_SRC_DIR}/operators/control)
  target_compile_features(${name} PRIVATE cxx_std_17)
  target_link_libraries(${name} PRIVATE wavetable_miniaudio)
  set_target_properties(${name} PROPERTIES
    PREFIX ""
    SUFFIX "${VIVID_PLUGIN_SUFFIX}"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  )
endforeach()

add_custom_target(operators DEPENDS ${VIVID_WAVETABLE_OPS})

add_executable(test_package_manifest tests/test_package_manifest.cpp)
target_compile_features(test_package_manifest PRIVATE cxx_std_17)

if(VIVID_PACKAGE_BUILD_TESTS)
  enable_testing()
  add_executable(test_wavetable_interp tests/cpp/test_wavetable_interp.cpp)
  target_compile_features(test_wavetable_interp PRIVATE cxx_std_17)
  add_test(NAME test_wavetable_interp COMMAND test_wavetable_interp)

  # Audio output correctness tests (spectral/amplitude property assertions)
  # Uses a minimal inline dlopen loader to avoid operator_loader.cpp's heavy
  # WebGPU transitive dependencies.
  add_executable(test_audio_correctness
      tests/cpp/test_audio_correctness.cpp
      ${VIVID_SRC_DIR}/src/runtime/output_analyzer.cpp
  )
  target_include_directories(test_audio_correctness PRIVATE ${VIVID_SRC_DIR}/src)
  target_compile_features(test_audio_correctness PRIVATE cxx_std_17)
  target_link_libraries(test_audio_correctness PRIVATE ${CMAKE_DL_LIBS})
  add_dependencies(test_audio_correctness analog_osc sub_osc wavetable_osc)
  add_test(NAME test_audio_correctness COMMAND test_audio_correctness
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()
