add_executable(
  test_input_parser
  test_input_parser.cpp
  ../input_parser.cpp
)
add_test(NAME test_input_parser COMMAND $<TARGET_FILE:test_input_parser>)

add_executable(test_driver_params test_driver_params.cpp ../driver.cpp)
target_link_libraries(test_driver_params PRIVATE librpa_file_reader)
add_test(NAME test_driver_params COMMAND $<TARGET_FILE:test_driver_params>)

add_executable(test_velocity_reader test_velocity_reader.cpp ../driver.cpp ../read_data.cpp)
target_link_libraries(test_velocity_reader PRIVATE librpa_file_reader)
add_test(NAME test_velocity_reader COMMAND $<TARGET_FILE:test_velocity_reader>)

add_executable(
  test_reader_sternheimer_qpoints
  test_reader_sternheimer_qpoints.cpp
  ../reader_sternheimer.cpp
  ../reader_sternheimer_qpoints.cpp
  ../rpa_qsum.cpp
)
target_link_libraries(test_reader_sternheimer_qpoints PRIVATE rpa_lib)
if(BUILD_SHARED_LIBS)
  target_link_libraries(test_reader_sternheimer_qpoints PRIVATE ${math_libs} ${parallel_libs})
endif()
add_test(NAME test_reader_sternheimer_qpoints
         COMMAND $<TARGET_FILE:test_reader_sternheimer_qpoints>)

add_executable(
  test_rpa_qsum
  test_rpa_qsum.cpp
  ../rpa_qsum.cpp
)
add_test(NAME test_rpa_qsum COMMAND $<TARGET_FILE:test_rpa_qsum>)
