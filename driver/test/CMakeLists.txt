add_executable(
  test_input_parser
  test_input_parser.cpp
  ../input_parser.cpp
)
add_test(NAME test_input_parser COMMAND $<TARGET_FILE:test_input_parser>)

add_executable(
  test_bz_sampling_utils
  test_bz_sampling_utils.cpp
  ../bz_sampling_utils.cpp
)
add_test(NAME test_bz_sampling_utils COMMAND $<TARGET_FILE:test_bz_sampling_utils>)
