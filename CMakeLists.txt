cmake_minimum_required(VERSION 3.16)
project(vivid_wavetable)

if(NOT DEFINED VIVID_SRC_DIR)
  message(FATAL_ERROR "VIVID_SRC_DIR must be set to the vivid-core source directory")
endif()

if(NOT DEFINED VIVID_PLUGIN_SUFFIX)
  set(VIVID_PLUGIN_SUFFIX .dylib)
endif()

option(VIVID_PACKAGE_BUILD_TESTS "Build vivid-wavetable package tests" OFF)

# WebGPU header — needed for operator_api/thumbnail.h
if(NOT DEFINED VIVID_WEBGPU_INCLUDE_DIR)
  # Auto-detect from vivid build tree (macOS/aarch64 default)
  file(GLOB _wgpu_candidates "${VIVID_SRC_DIR}/build/_deps/wgpu-*/include")
  if(_wgpu_candidates)
    list(GET _wgpu_candidates 0 VIVID_WEBGPU_INCLUDE_DIR)
  endif()
endif()

if(NOT DEFINED VIVID_NLOHMANN_INCLUDE_DIR)
  find_path(VIVID_NLOHMANN_INCLUDE_DIR
    NAMES nlohmann/json.hpp
    PATHS
      "${VIVID_SRC_DIR}/build/_deps/nlohmann_json-src/include"
      "${VIVID_SRC_DIR}/build/_deps/nlohmann_json-src/single_include"
      "${VIVID_SRC_DIR}/build-asan/_deps/nlohmann_json-src/include"
      "${VIVID_SRC_DIR}/build-asan/_deps/nlohmann_json-src/single_include")
endif()

set(VIVID_WAVETABLE_OPS
  poly_voice_allocator
  wavetable_osc
  voice_mixer
  voice_drive
  sub_osc
  analog_osc
  noise_layer
)

set(VIVID_WAVETABLE_OSC_SOURCES
  src/wavetable_osc.cpp
  src/wavetable_osc_process.cpp
  src/wavetable_osc_thumbnail.cpp
  src/wavetable_bank.cpp
  src/wavetable_dsp.cpp
)

# miniaudio: decoder only (no device I/O, no threading, no encoding)
add_library(wavetable_miniaudio STATIC src/miniaudio_impl.c)
target_include_directories(wavetable_miniaudio PRIVATE ${VIVID_SRC_DIR}/deps/miniaudio)

foreach(name IN LISTS VIVID_WAVETABLE_OPS)
  if(name STREQUAL wavetable_osc)
    add_library(${name} MODULE ${VIVID_WAVETABLE_OSC_SOURCES})
  elseif(name STREQUAL analog_osc)
    add_library(${name} MODULE src/${name}.cpp src/wavetable_dsp.cpp)
  else()
    add_library(${name} MODULE src/${name}.cpp)
  endif()
  target_include_directories(${name} PRIVATE ${VIVID_SRC_DIR}/src ${VIVID_SRC_DIR}/deps/miniaudio ${VIVID_SRC_DIR}/operators/control)
  if(VIVID_WEBGPU_INCLUDE_DIR)
    target_include_directories(${name} PRIVATE ${VIVID_WEBGPU_INCLUDE_DIR})
  endif()
  target_compile_features(${name} PRIVATE cxx_std_17)
  target_link_libraries(${name} PRIVATE wavetable_miniaudio)
  set_target_properties(${name} PROPERTIES
    PREFIX ""
    SUFFIX "${VIVID_PLUGIN_SUFFIX}"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  if(APPLE)
    target_link_options(${name} PRIVATE -undefined dynamic_lookup)
  endif()
endforeach()

add_custom_target(operators DEPENDS ${VIVID_WAVETABLE_OPS})

add_executable(test_package_manifest tests/test_package_manifest.cpp)
target_compile_features(test_package_manifest PRIVATE cxx_std_17)

if(VIVID_PACKAGE_BUILD_TESTS)
  enable_testing()
  add_test(NAME test_package_manifest COMMAND test_package_manifest
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
  add_executable(test_wavetable_interp tests/cpp/test_wavetable_interp.cpp)
  target_compile_features(test_wavetable_interp PRIVATE cxx_std_17)
  add_test(NAME test_wavetable_interp COMMAND test_wavetable_interp)

  add_executable(test_wavetable_dsp
      tests/cpp/test_wavetable_dsp.cpp
      src/wavetable_dsp.cpp
  )
  target_include_directories(test_wavetable_dsp PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_compile_features(test_wavetable_dsp PRIVATE cxx_std_17)
  add_test(NAME test_wavetable_dsp COMMAND test_wavetable_dsp)

  add_executable(test_wavetable_bank
      tests/cpp/test_wavetable_bank.cpp
      src/wavetable_bank.cpp
  )
  target_include_directories(test_wavetable_bank PRIVATE ${CMAKE_SOURCE_DIR}/src ${VIVID_SRC_DIR}/deps/miniaudio)
  target_compile_features(test_wavetable_bank PRIVATE cxx_std_17)
  target_link_libraries(test_wavetable_bank PRIVATE wavetable_miniaudio)
  add_test(NAME test_wavetable_bank COMMAND test_wavetable_bank)

  # Audio output correctness tests (spectral/amplitude property assertions)
  # Uses a minimal inline dlopen loader to avoid operator_loader.cpp's heavy
  # WebGPU transitive dependencies.
  add_executable(test_audio_correctness
      tests/cpp/test_audio_correctness.cpp
      ${VIVID_SRC_DIR}/src/runtime/debug/output_analyzer.cpp
  )
  target_include_directories(test_audio_correctness PRIVATE
      ${VIVID_SRC_DIR}/src
      ${VIVID_SRC_DIR}/operators/control/envelope
  )
  target_compile_features(test_audio_correctness PRIVATE cxx_std_17)
  target_link_libraries(test_audio_correctness PRIVATE ${CMAKE_DL_LIBS})
  add_dependencies(test_audio_correctness analog_osc sub_osc wavetable_osc voice_mixer voice_drive noise_layer)
  add_test(NAME test_audio_correctness COMMAND test_audio_correctness
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  add_executable(test_graph_envelope_wiring tests/cpp/test_graph_envelope_wiring.cpp)
  target_compile_features(test_graph_envelope_wiring PRIVATE cxx_std_17)
  add_test(NAME test_graph_envelope_wiring COMMAND test_graph_envelope_wiring
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

  if(NOT VIVID_NLOHMANN_INCLUDE_DIR)
    message(FATAL_ERROR "VIVID_NLOHMANN_INCLUDE_DIR must point to nlohmann/json.hpp when VIVID_PACKAGE_BUILD_TESTS=ON")
  endif()

  add_executable(test_module_surface_contract
      tests/cpp/test_module_surface_contract.cpp
      ${VIVID_SRC_DIR}/src/runtime/graph/graph.cpp
      ${VIVID_SRC_DIR}/src/runtime/graph/subgraph_module.cpp
  )
  target_include_directories(test_module_surface_contract PRIVATE
      ${VIVID_SRC_DIR}/src
      ${VIVID_NLOHMANN_INCLUDE_DIR}
  )
  target_compile_features(test_module_surface_contract PRIVATE cxx_std_17)
  add_test(NAME test_module_surface_contract COMMAND test_module_surface_contract
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  add_executable(test_poly_voice_allocator tests/cpp/test_poly_voice_allocator.cpp)
  target_include_directories(test_poly_voice_allocator PRIVATE ${VIVID_SRC_DIR}/src)
  target_compile_features(test_poly_voice_allocator PRIVATE cxx_std_17)
  target_link_libraries(test_poly_voice_allocator PRIVATE ${CMAKE_DL_LIBS})
  add_dependencies(test_poly_voice_allocator poly_voice_allocator)
  add_test(NAME test_poly_voice_allocator COMMAND test_poly_voice_allocator
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()
