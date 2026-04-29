cmake_minimum_required(VERSION 3.16)
project(vivid_wavetable)

if(NOT DEFINED VIVID_SRC_DIR)
  message(FATAL_ERROR "VIVID_SRC_DIR must be set to the vivid-core source directory")
endif()

if(NOT DEFINED VIVID_PLUGIN_SUFFIX)
  if(APPLE)
    set(VIVID_PLUGIN_SUFFIX .dylib)
  elseif(WIN32)
    set(VIVID_PLUGIN_SUFFIX .dll)
  else()
    set(VIVID_PLUGIN_SUFFIX .so)
  endif()
endif()

option(VIVID_PACKAGE_BUILD_TESTS "Build vivid-wavetable package tests" OFF)
if(APPLE)
  set(_VIVID_WAVETABLE_ACCELERATE_DEFAULT ON)
else()
  set(_VIVID_WAVETABLE_ACCELERATE_DEFAULT OFF)
endif()
option(VIVID_WAVETABLE_ENABLE_ACCELERATE
  "Build the optional macOS Accelerate backend for WavetableLayer"
  ${_VIVID_WAVETABLE_ACCELERATE_DEFAULT})
option(VIVID_WAVETABLE_PREFER_ACCELERATE
  "Prefer the experimental Accelerate WavetableLayer backend over Highway"
  ${_VIVID_WAVETABLE_ACCELERATE_DEFAULT})
# When ON, WavetableLayer always dispatches to the scalar reference renderer
# regardless of which optimized backends were compiled in. The optimized
# sources still build (so test_wavetable_layer_equivalence remains valid),
# but production audio runs through the canonical algorithm. Use this when
# diagnosing audio quality issues to bound "is this a backend bug" vs
# "is the algorithm itself wrong" — the scalar renderer is the ground truth.
option(VIVID_WAVETABLE_REFERENCE_ONLY
  "Force WavetableLayer to use only the scalar reference renderer at runtime"
  OFF)

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

if(NOT DEFINED VIVID_DRAGONBOX_INCLUDE_DIR)
  find_path(VIVID_DRAGONBOX_INCLUDE_DIR
    NAMES dragonbox/dragonbox_to_chars.h
    PATHS
      "${VIVID_SRC_DIR}/build/_deps/dragonbox-src/include"
      "${VIVID_SRC_DIR}/build-local/_deps/dragonbox-src/include"
      "${VIVID_SRC_DIR}/build-asan/_deps/dragonbox-src/include")
endif()

if(NOT DEFINED VIVID_DRAGONBOX_LIBRARY)
  find_library(VIVID_DRAGONBOX_LIBRARY
    NAMES dragonbox_to_chars libdragonbox_to_chars
    PATHS
      "${VIVID_SRC_DIR}/build/_deps/dragonbox-build"
      "${VIVID_SRC_DIR}/build-local/_deps/dragonbox-build"
      "${VIVID_SRC_DIR}/build-asan/_deps/dragonbox-build"
    NO_DEFAULT_PATH)
endif()

set(VIVID_WAVETABLE_OPS
  wavetable_osc
  wavetable_layer
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
  src/wavetable_osc_editor.cpp
  src/wavetable_osc_editor_shared.cpp
  src/wavetable_bank.cpp
  src/wavetable_dsp.cpp
)

set(VIVID_WAVETABLE_LAYER_SOURCES
  src/wavetable_layer.cpp
  src/wavetable_layer_renderer.cpp
  src/wavetable_bank.cpp
  src/wavetable_dsp.cpp
)

# miniaudio: decoder only (no device I/O, no threading, no encoding)
add_library(wavetable_miniaudio STATIC src/miniaudio_impl.c)
target_include_directories(wavetable_miniaudio PRIVATE ${VIVID_SRC_DIR}/deps/miniaudio)

foreach(name IN LISTS VIVID_WAVETABLE_OPS)
  if(name STREQUAL wavetable_osc)
    add_library(${name} MODULE ${VIVID_WAVETABLE_OSC_SOURCES})
  elseif(name STREQUAL wavetable_layer)
    add_library(${name} MODULE ${VIVID_WAVETABLE_LAYER_SOURCES})
  elseif(name STREQUAL analog_osc)
    add_library(${name} MODULE src/${name}.cpp src/wavetable_dsp.cpp)
  else()
    add_library(${name} MODULE src/${name}.cpp)
  endif()
  target_include_directories(${name} PRIVATE ${VIVID_SRC_DIR}/src ${VIVID_SRC_DIR}/deps/miniaudio ${VIVID_SRC_DIR}/operators/control ${VIVID_SRC_DIR}/operators/shared/sequencer)
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

# Highway SIMD backend for WavetableLayer (when available from Vivid core)
if(DEFINED VIVID_HIGHWAY_INCLUDE_DIR)
  target_sources(wavetable_layer PRIVATE src/wavetable_layer_renderer_simd.cpp)
  target_include_directories(wavetable_layer PRIVATE
      ${VIVID_HIGHWAY_INCLUDE_DIR}
      ${CMAKE_SOURCE_DIR}/src)
  target_compile_definitions(wavetable_layer PRIVATE VIVID_HAS_HIGHWAY=1)
  if(DEFINED VIVID_HIGHWAY_LIBRARY)
    target_link_libraries(wavetable_layer PRIVATE ${VIVID_HIGHWAY_LIBRARY})
  endif()
endif()

# macOS Accelerate backend for WavetableLayer hot paths. This is an internal
# platform backend; Highway remains the portable SIMD baseline for Windows.
if(APPLE AND VIVID_WAVETABLE_ENABLE_ACCELERATE)
  find_library(VIVID_ACCELERATE_FRAMEWORK Accelerate REQUIRED)
  set(VIVID_WAVETABLE_HAS_ACCELERATE ON)
  target_sources(wavetable_layer PRIVATE src/wavetable_layer_renderer_accelerate.cpp)
  target_compile_definitions(wavetable_layer PRIVATE VIVID_HAS_ACCELERATE=1)
  if(VIVID_WAVETABLE_PREFER_ACCELERATE)
    target_compile_definitions(wavetable_layer PRIVATE VIVID_WAVETABLE_PREFER_ACCELERATE=1)
  endif()
  target_link_libraries(wavetable_layer PRIVATE ${VIVID_ACCELERATE_FRAMEWORK})
endif()

if(VIVID_WAVETABLE_REFERENCE_ONLY)
  target_compile_definitions(wavetable_layer PRIVATE VIVID_WAVETABLE_REFERENCE_ONLY=1)
endif()

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

  # AnalogOsc note-stream smoke: notes_in produces stereo audio, ADSR release
  # decay, polyphonic chord summing, and silence at rest.
  add_executable(test_analog_osc_midi
      tests/cpp/test_analog_osc_midi.cpp
      ${VIVID_SRC_DIR}/src/runtime/debug/output_analyzer.cpp
  )
  target_include_directories(test_analog_osc_midi PRIVATE ${VIVID_SRC_DIR}/src)
  target_compile_features(test_analog_osc_midi PRIVATE cxx_std_17)
  target_link_libraries(test_analog_osc_midi PRIVATE ${CMAKE_DL_LIBS})
  add_dependencies(test_analog_osc_midi analog_osc)
  add_test(NAME test_analog_osc_midi COMMAND test_analog_osc_midi ${CMAKE_BINARY_DIR}
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  # WavetableOsc MIDI-input smoke: same shape as test_analog_osc_midi but for
  # the wavetable oscillator (which dispatches MIDI input through synthetic
  # lanes into the existing lane-driven render). Links wgpu_stubs.c because
  # WavetableOsc imports WGPU symbols for its thumbnail renderer; the audio
  # render path doesn't call them, so stubs are sufficient for the test.
  add_executable(test_wavetable_osc_midi
      tests/cpp/test_wavetable_osc_midi.cpp
      tests/cpp/wgpu_stubs.c
      ${VIVID_SRC_DIR}/src/runtime/debug/output_analyzer.cpp
  )
  target_include_directories(test_wavetable_osc_midi PRIVATE ${VIVID_SRC_DIR}/src)
  target_compile_features(test_wavetable_osc_midi PRIVATE cxx_std_17)
  target_link_libraries(test_wavetable_osc_midi PRIVATE ${CMAKE_DL_LIBS})
  set_target_properties(test_wavetable_osc_midi PROPERTIES
      ENABLE_EXPORTS ON)  # export symbols so dlopen'd dylib can find wgpu stubs
  add_dependencies(test_wavetable_osc_midi wavetable_osc)
  add_test(NAME test_wavetable_osc_midi COMMAND test_wavetable_osc_midi ${CMAKE_BINARY_DIR}
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  # Wavetable synth voice-breakouts: AnalogOsc and WavetableOsc surface
  # check (voices_out + voice_*/control lanes ADVANCED-tagged) plus
  # AnalogOsc 3-note chord behavioral check on the new stereo `output`.
  add_executable(test_wavetable_synth_breakouts
      tests/cpp/test_wavetable_synth_breakouts.cpp
      ${VIVID_SRC_DIR}/src/runtime/debug/output_analyzer.cpp
  )
  target_include_directories(test_wavetable_synth_breakouts PRIVATE ${VIVID_SRC_DIR}/src)
  target_compile_features(test_wavetable_synth_breakouts PRIVATE cxx_std_17)
  target_link_libraries(test_wavetable_synth_breakouts PRIVATE ${CMAKE_DL_LIBS})
  add_dependencies(test_wavetable_synth_breakouts analog_osc)
  add_test(NAME test_wavetable_synth_breakouts
           COMMAND test_wavetable_synth_breakouts ${CMAKE_BINARY_DIR}
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  # SubOsc note-stream smoke: notes_in produces stereo + voice_* breakouts;
  # 3-note chord aligns voice_ids ascending; voices_out per-voice channels;
  # ADSR release decay.
  add_executable(test_sub_osc_midi
      tests/cpp/test_sub_osc_midi.cpp
      ${VIVID_SRC_DIR}/src/runtime/debug/output_analyzer.cpp
  )
  target_include_directories(test_sub_osc_midi PRIVATE ${VIVID_SRC_DIR}/src)
  target_compile_features(test_sub_osc_midi PRIVATE cxx_std_17)
  target_link_libraries(test_sub_osc_midi PRIVATE ${CMAKE_DL_LIBS})
  add_dependencies(test_sub_osc_midi sub_osc)
  add_test(NAME test_sub_osc_midi COMMAND test_sub_osc_midi ${CMAKE_BINARY_DIR}
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  # NoiseLayer MIDI-input smoke: same shape as test_sub_osc_midi.
  add_executable(test_noise_layer_midi
      tests/cpp/test_noise_layer_midi.cpp
      ${VIVID_SRC_DIR}/src/runtime/debug/output_analyzer.cpp
  )
  target_include_directories(test_noise_layer_midi PRIVATE ${VIVID_SRC_DIR}/src)
  target_compile_features(test_noise_layer_midi PRIVATE cxx_std_17)
  target_link_libraries(test_noise_layer_midi PRIVATE ${CMAKE_DL_LIBS})
  add_dependencies(test_noise_layer_midi noise_layer)
  add_test(NAME test_noise_layer_midi COMMAND test_noise_layer_midi ${CMAKE_BINARY_DIR}
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  # WavetableLayer note-stream smoke: notes_in produces stereo and ADSR release
  # decays.
  add_executable(test_wavetable_layer_midi
      tests/cpp/test_wavetable_layer_midi.cpp
      ${VIVID_SRC_DIR}/src/runtime/debug/output_analyzer.cpp
  )
  target_include_directories(test_wavetable_layer_midi PRIVATE ${VIVID_SRC_DIR}/src)
  target_compile_features(test_wavetable_layer_midi PRIVATE cxx_std_17)
  target_link_libraries(test_wavetable_layer_midi PRIVATE ${CMAKE_DL_LIBS})
  add_dependencies(test_wavetable_layer_midi wavetable_layer)
  add_test(NAME test_wavetable_layer_midi COMMAND test_wavetable_layer_midi ${CMAKE_BINARY_DIR}
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  if(NOT VIVID_NLOHMANN_INCLUDE_DIR)
    message(FATAL_ERROR "VIVID_NLOHMANN_INCLUDE_DIR must point to nlohmann/json.hpp when VIVID_PACKAGE_BUILD_TESTS=ON")
  endif()

  # Phase 3 PR2 guard: no checked-in graph references VoiceAllocator or
  # the synth lane-input ports being removed in PR3. CI fence so future
  # graph edits can't silently re-introduce the legacy pattern.
  add_executable(test_no_voice_allocator_in_graphs
      tests/cpp/test_no_voice_allocator_in_graphs.cpp
  )
  target_include_directories(test_no_voice_allocator_in_graphs PRIVATE
      ${VIVID_NLOHMANN_INCLUDE_DIR}
  )
  target_compile_features(test_no_voice_allocator_in_graphs PRIVATE cxx_std_17)
  add_test(NAME test_no_voice_allocator_in_graphs
           COMMAND test_no_voice_allocator_in_graphs
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # Positive CI fence for the maintained native-note package model. Checks
  # representative graphs/modules that teach the current `notes_in` +
  # `NoteBreakout` contract and keeps metadata validation on maintained
  # checked-in graphs.
  add_executable(test_native_note_graph_contract
      tests/cpp/test_native_note_graph_contract.cpp
  )
  target_include_directories(test_native_note_graph_contract PRIVATE
      ${VIVID_NLOHMANN_INCLUDE_DIR}
  )
  target_compile_features(test_native_note_graph_contract PRIVATE cxx_std_17)
  add_test(NAME test_native_note_graph_contract
           COMMAND test_native_note_graph_contract
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  add_executable(test_module_surface_contract
      tests/cpp/test_module_surface_contract.cpp
      ${VIVID_SRC_DIR}/src/runtime/graph/graph.cpp
      ${VIVID_SRC_DIR}/src/runtime/graph/subgraph_module.cpp
      ${VIVID_SRC_DIR}/src/runtime/graph/operator_aliases.cpp
  )
  target_include_directories(test_module_surface_contract PRIVATE
      ${VIVID_SRC_DIR}/src
      ${VIVID_NLOHMANN_INCLUDE_DIR}
      ${VIVID_DRAGONBOX_INCLUDE_DIR}
  )
  target_compile_features(test_module_surface_contract PRIVATE cxx_std_17)
  if(VIVID_DRAGONBOX_LIBRARY)
    target_link_libraries(test_module_surface_contract PRIVATE ${VIVID_DRAGONBOX_LIBRARY})
  endif()
  add_test(NAME test_module_surface_contract COMMAND test_module_surface_contract
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  add_executable(test_layerpad_reference_graphs
      tests/cpp/test_layerpad_reference_graphs.cpp
      ${VIVID_SRC_DIR}/src/runtime/debug/output_analyzer.cpp
      ${VIVID_SRC_DIR}/src/runtime/graph/graph.cpp
      ${VIVID_SRC_DIR}/src/runtime/graph/subgraph_module.cpp
      ${VIVID_SRC_DIR}/src/runtime/graph/operator_aliases.cpp
  )
  target_include_directories(test_layerpad_reference_graphs PRIVATE
      ${VIVID_SRC_DIR}/src
      ${VIVID_NLOHMANN_INCLUDE_DIR}
      ${VIVID_DRAGONBOX_INCLUDE_DIR}
  )
  target_compile_features(test_layerpad_reference_graphs PRIVATE cxx_std_17)
  target_compile_definitions(test_layerpad_reference_graphs PRIVATE
      "VIVID_CORE_BUILD_DIR_STR=\"${VIVID_BUILD_DIR}\""
      "VIVID_PACKAGE_BUILD_DIR_STR=\"${CMAKE_BINARY_DIR}\""
      "VIVID_PLUGIN_SUFFIX_STR=\"${VIVID_PLUGIN_SUFFIX}\"")
  target_link_libraries(test_layerpad_reference_graphs PRIVATE ${CMAKE_DL_LIBS})
  if(VIVID_DRAGONBOX_LIBRARY)
    target_link_libraries(test_layerpad_reference_graphs PRIVATE ${VIVID_DRAGONBOX_LIBRARY})
  endif()
  add_test(NAME test_layerpad_reference_graphs COMMAND test_layerpad_reference_graphs
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  add_executable(test_factory_wavetable_assets
      tests/cpp/test_factory_wavetable_assets.cpp
      src/wavetable_bank.cpp
  )
  target_include_directories(test_factory_wavetable_assets PRIVATE
      ${CMAKE_SOURCE_DIR}/src
      ${VIVID_SRC_DIR}/deps/miniaudio
  )
  target_compile_features(test_factory_wavetable_assets PRIVATE cxx_std_17)
  target_link_libraries(test_factory_wavetable_assets PRIVATE wavetable_miniaudio)
  add_test(NAME test_factory_wavetable_assets COMMAND test_factory_wavetable_assets
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

  add_executable(test_instrument_metadata tests/cpp/test_instrument_metadata.cpp)
  target_include_directories(test_instrument_metadata PRIVATE
      ${VIVID_NLOHMANN_INCLUDE_DIR}
  )
  target_compile_features(test_instrument_metadata PRIVATE cxx_std_17)
  add_test(NAME test_instrument_metadata COMMAND test_instrument_metadata
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

  add_executable(test_wavetable_layer_surface tests/cpp/test_wavetable_layer_surface.cpp)
  target_include_directories(test_wavetable_layer_surface PRIVATE ${VIVID_SRC_DIR}/src)
  target_compile_features(test_wavetable_layer_surface PRIVATE cxx_std_17)
  target_compile_definitions(test_wavetable_layer_surface PRIVATE
      "VIVID_PLUGIN_SUFFIX_STR=\"${VIVID_PLUGIN_SUFFIX}\"")
  target_link_libraries(test_wavetable_layer_surface PRIVATE ${CMAKE_DL_LIBS})
  add_dependencies(test_wavetable_layer_surface wavetable_layer)
  add_test(NAME test_wavetable_layer_surface COMMAND test_wavetable_layer_surface
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  add_executable(test_wavetable_layer_renderer
      tests/cpp/test_wavetable_layer_renderer.cpp
      ${VIVID_SRC_DIR}/src/runtime/debug/output_analyzer.cpp
  )
  target_include_directories(test_wavetable_layer_renderer PRIVATE ${VIVID_SRC_DIR}/src)
  target_compile_features(test_wavetable_layer_renderer PRIVATE cxx_std_17)
  target_compile_definitions(test_wavetable_layer_renderer PRIVATE
      "VIVID_PLUGIN_SUFFIX_STR=\"${VIVID_PLUGIN_SUFFIX}\"")
  target_link_libraries(test_wavetable_layer_renderer PRIVATE ${CMAKE_DL_LIBS})
  add_dependencies(test_wavetable_layer_renderer wavetable_layer)
  add_test(NAME test_wavetable_layer_renderer COMMAND test_wavetable_layer_renderer
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  # --- Lessons-as-tests: regressions for fixes shipped in commit a0955b1 ----
  # Each of these tests pins down a fix that was made in source today, so a
  # future refactor cannot quietly invert it.
  # The opt-in tests are currently expected to fail / skip:
  #   - test_position_zero_renders_clean: enable with -DVIVID_TEST_POSITION_ZERO=1
  #     once the bank/renderer bug at family=0/member=0/position=0 is fixed
  #   - test_filter_lane_input_routing: a placeholder; the real fix and its
  #     regression test belong in the vivid-core repo (Filter operator)
  foreach(_lesson_test
      test_dc_blocked_output
      test_expression_defaults_inert
      test_voice_breakout_lane_routing
      test_retrigger_phase_continuity
      test_position_zero_renders_clean
      test_filter_lane_input_routing)
    add_executable(${_lesson_test} tests/cpp/${_lesson_test}.cpp)
    target_include_directories(${_lesson_test} PRIVATE ${VIVID_SRC_DIR}/src)
    target_compile_features(${_lesson_test} PRIVATE cxx_std_17)
    target_compile_definitions(${_lesson_test} PRIVATE
        "VIVID_PLUGIN_SUFFIX_STR=\"${VIVID_PLUGIN_SUFFIX}\"")
    target_link_libraries(${_lesson_test} PRIVATE ${CMAKE_DL_LIBS})
    add_dependencies(${_lesson_test} wavetable_layer)
    add_test(NAME ${_lesson_test} COMMAND ${_lesson_test}
             WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  endforeach()

  # --- WavetableOsc editor tests ---------------------------------------------
  # Helpers: pure-logic (family/member naming, waveform sampling math,
  # unison layout, cell hit-test). Links a small set of sources.
  add_executable(test_wavetable_osc_editor_helpers
      tests/cpp/test_wavetable_osc_editor_helpers.cpp
      src/wavetable_osc_editor_shared.cpp
      src/wavetable_bank.cpp
      src/wavetable_dsp.cpp
  )
  target_include_directories(test_wavetable_osc_editor_helpers PRIVATE
      ${CMAKE_SOURCE_DIR}/src
      ${VIVID_SRC_DIR}/src
      ${VIVID_SRC_DIR}/deps/miniaudio)
  if(VIVID_WEBGPU_INCLUDE_DIR)
    target_include_directories(test_wavetable_osc_editor_helpers PRIVATE
        ${VIVID_WEBGPU_INCLUDE_DIR})
  endif()
  target_compile_features(test_wavetable_osc_editor_helpers PRIVATE cxx_std_17)
  target_link_libraries(test_wavetable_osc_editor_helpers PRIVATE wavetable_miniaudio)
  add_test(NAME test_wavetable_osc_editor_helpers
           COMMAND test_wavetable_osc_editor_helpers)

  # End-to-end editor test — constructs a WavetableOsc and drives its
  # draw_editor with a synthesised VividEditorContext. Needs the full
  # operator compile unit (including thumbnail/process to satisfy the
  # vtable) but no WebGPU device or audio device.
  add_executable(test_wavetable_osc_editor
      tests/cpp/test_wavetable_osc_editor.cpp
      src/wavetable_osc.cpp
      src/wavetable_osc_editor.cpp
      src/wavetable_osc_editor_shared.cpp
      src/wavetable_osc_thumbnail.cpp
      src/wavetable_osc_process.cpp
      src/wavetable_bank.cpp
      src/wavetable_dsp.cpp
  )
  target_include_directories(test_wavetable_osc_editor PRIVATE
      ${CMAKE_SOURCE_DIR}/src
      ${VIVID_SRC_DIR}/src
      ${VIVID_SRC_DIR}/deps/miniaudio
      ${VIVID_SRC_DIR}/operators/shared/sequencer)
  if(VIVID_WEBGPU_INCLUDE_DIR)
    target_include_directories(test_wavetable_osc_editor PRIVATE
        ${VIVID_WEBGPU_INCLUDE_DIR})
  endif()
  target_compile_features(test_wavetable_osc_editor PRIVATE cxx_std_17)
  target_link_libraries(test_wavetable_osc_editor PRIVATE wavetable_miniaudio)
  # The thumbnail TU references WGPU symbols (pulled in by the destructor
  # path). draw_editor never reaches them, so allow dynamic-lookup at
  # link time so the test binary can link without a WGPU runtime.
  if(APPLE)
    target_link_options(test_wavetable_osc_editor PRIVATE -undefined dynamic_lookup)
  endif()
  add_test(NAME test_wavetable_osc_editor COMMAND test_wavetable_osc_editor)

  # Scalar/SIMD equivalence test — links renderer sources directly to call both backends
  set(EQUIVALENCE_SOURCES
      tests/cpp/test_wavetable_layer_equivalence.cpp
      src/wavetable_layer_renderer.cpp
      src/wavetable_bank.cpp
      src/wavetable_dsp.cpp
  )
  if(DEFINED VIVID_HIGHWAY_INCLUDE_DIR)
    list(APPEND EQUIVALENCE_SOURCES src/wavetable_layer_renderer_simd.cpp)
  endif()
  if(VIVID_WAVETABLE_HAS_ACCELERATE)
    list(APPEND EQUIVALENCE_SOURCES src/wavetable_layer_renderer_accelerate.cpp)
  endif()
  add_executable(test_wavetable_layer_equivalence ${EQUIVALENCE_SOURCES})
  target_include_directories(test_wavetable_layer_equivalence PRIVATE
      ${VIVID_SRC_DIR}/src
      ${VIVID_SRC_DIR}/deps/miniaudio
      ${CMAKE_SOURCE_DIR}/src)
  target_compile_features(test_wavetable_layer_equivalence PRIVATE cxx_std_17)
  target_link_libraries(test_wavetable_layer_equivalence PRIVATE wavetable_miniaudio)
  if(DEFINED VIVID_HIGHWAY_INCLUDE_DIR)
    target_include_directories(test_wavetable_layer_equivalence PRIVATE ${VIVID_HIGHWAY_INCLUDE_DIR})
    target_compile_definitions(test_wavetable_layer_equivalence PRIVATE VIVID_HAS_HIGHWAY=1)
    if(DEFINED VIVID_HIGHWAY_LIBRARY)
      target_link_libraries(test_wavetable_layer_equivalence PRIVATE ${VIVID_HIGHWAY_LIBRARY})
    endif()
  endif()
  if(VIVID_WAVETABLE_HAS_ACCELERATE)
    target_compile_definitions(test_wavetable_layer_equivalence PRIVATE VIVID_HAS_ACCELERATE=1)
    target_link_libraries(test_wavetable_layer_equivalence PRIVATE ${VIVID_ACCELERATE_FRAMEWORK})
  endif()
  add_test(NAME test_wavetable_layer_equivalence COMMAND test_wavetable_layer_equivalence)
endif()
