cmake_minimum_required(VERSION 3.20)

project(totton_audio_de_mirroring_dsp LANGUAGES CXX)

add_library(tadm_dsp
    src/multistage_upsampler.cpp
    src/fir_design.cpp
    src/fir_config.cpp
    src/stage2_quality_benchmark.cpp
)

target_include_directories(tadm_dsp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(tadm_dsp PUBLIC cxx_std_17)

target_compile_options(tadm_dsp PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(tadm_dsp PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(tadm_dsp_capi SHARED
    src/multistage_upsampler_c_api.cpp
)

target_link_libraries(tadm_dsp_capi PRIVATE tadm_dsp)
target_include_directories(tadm_dsp_capi PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(tadm_dsp_capi PUBLIC cxx_std_17)
target_compile_options(tadm_dsp_capi PRIVATE -Wall -Wextra -Wpedantic)

add_executable(tadm_dsp_tests
    tests/test_multistage_upsampler.cpp
)

target_link_libraries(tadm_dsp_tests PRIVATE tadm_dsp)

add_executable(tadm_fir_tests
    tests/test_min_phase_fir.cpp
)

target_link_libraries(tadm_fir_tests PRIVATE tadm_dsp)

add_executable(tadm_stage2_quality_tests
    tests/test_stage2_quality_benchmark.cpp
)

target_link_libraries(tadm_stage2_quality_tests PRIVATE tadm_dsp)

add_executable(tadm_fir_design
    src/fir_design_cli.cpp
)

target_link_libraries(tadm_fir_design PRIVATE tadm_dsp)

add_executable(tadm_stage2_quality_bench
    src/stage2_quality_bench_cli.cpp
)

target_link_libraries(tadm_stage2_quality_bench PRIVATE tadm_dsp)

enable_testing()
add_test(NAME multistage_upsampler COMMAND tadm_dsp_tests)
add_test(NAME fir_design COMMAND tadm_fir_tests)
add_test(NAME stage2_quality_benchmark COMMAND tadm_stage2_quality_tests)
