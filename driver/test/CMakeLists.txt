add_executable(
  test_input_parser
  test_input_parser.cpp
  ../input_parser.cpp
)
add_test(NAME test_input_parser COMMAND $<TARGET_FILE:test_input_parser>)

add_executable(
  test_reader_sternheimer_qpoints
  test_reader_sternheimer_qpoints.cpp
  ../reader_sternheimer.cpp
  ../reader_sternheimer_qpoints.cpp
  ../rpa_qsum.cpp
)
target_link_libraries(test_reader_sternheimer_qpoints PRIVATE rpa_lib)
target_compile_options(test_reader_sternheimer_qpoints PRIVATE -UNDEBUG)
if(BUILD_SHARED_LIBS)
  target_link_libraries(test_reader_sternheimer_qpoints PRIVATE ${math_libs} ${parallel_libs})
endif()
add_test(NAME test_reader_sternheimer_qpoints
         COMMAND $<TARGET_FILE:test_reader_sternheimer_qpoints>)

add_executable(
  test_reader_sternheimer_partial
  test_reader_sternheimer_partial.cpp
  ../reader_sternheimer.cpp
  ../reader_sternheimer_partial.cpp
)
target_link_libraries(test_reader_sternheimer_partial PRIVATE rpa_lib)
target_compile_options(test_reader_sternheimer_partial PRIVATE -UNDEBUG)
if(BUILD_SHARED_LIBS)
  target_link_libraries(test_reader_sternheimer_partial PRIVATE ${math_libs} ${parallel_libs})
endif()
add_test(NAME test_reader_sternheimer_partial
         COMMAND $<TARGET_FILE:test_reader_sternheimer_partial>)

add_executable(
  test_sternheimer_partial_reconstruction
  test_sternheimer_partial_reconstruction.cpp
  ../sternheimer_partial_reconstruction.cpp
  ../rpa_qsum.cpp
)
target_link_libraries(test_sternheimer_partial_reconstruction PRIVATE rpa_lib)
target_compile_options(test_sternheimer_partial_reconstruction PRIVATE -UNDEBUG)
if(BUILD_SHARED_LIBS)
  target_link_libraries(test_sternheimer_partial_reconstruction PRIVATE ${math_libs} ${parallel_libs})
endif()
add_test(NAME test_sternheimer_partial_reconstruction
         COMMAND $<TARGET_FILE:test_sternheimer_partial_reconstruction>)

add_executable(
  test_rpa_qsum
  test_rpa_qsum.cpp
  ../rpa_qsum.cpp
)
add_test(NAME test_rpa_qsum COMMAND $<TARGET_FILE:test_rpa_qsum>)

add_executable(
  test_bz_sampling_utils
  test_bz_sampling_utils.cpp
  ../bz_sampling_utils.cpp
)
add_test(NAME test_bz_sampling_utils COMMAND $<TARGET_FILE:test_bz_sampling_utils>)

add_executable(
  test_reader_fermi_surface_payload
  test_reader_fermi_surface_payload.cpp
  ../reader_fermi_surface_payload.cpp
)
target_link_libraries(test_reader_fermi_surface_payload PRIVATE rpa_lib)
target_compile_options(test_reader_fermi_surface_payload PRIVATE -UNDEBUG)
if(BUILD_SHARED_LIBS)
  target_link_libraries(test_reader_fermi_surface_payload PRIVATE ${math_libs} ${parallel_libs})
endif()
add_test(NAME test_reader_fermi_surface_payload
         COMMAND $<TARGET_FILE:test_reader_fermi_surface_payload>)
