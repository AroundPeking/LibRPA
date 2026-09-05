macro(config_test_f03 TARGET NUMPROC)
  add_executable(${TARGET} ${TARGET}.f90)
  add_dependencies(${TARGET} rpa_f03_lib rpa_lib)
  target_link_libraries(${TARGET} PRIVATE rpa_f03_lib rpa_lib)
  if(BUILD_SHARED_LIBS)
    target_link_libraries(${TARGET} PRIVATE ${math_libs} ${parallel_libs})
  endif()
  set_target_properties(${TARGET} PROPERTIES LINKER_LANGUAGE Fortran)
  target_include_directories(${TARGET} PRIVATE
    $<TARGET_PROPERTY:rpa_f03_lib,Fortran_MODULE_DIRECTORY>
  )
  if(${NUMPROC} GREATER 1)
    # message("parallel test")
    add_test(
      NAME ${TARGET}
      COMMAND mpirun -np ${NUMPROC} $<TARGET_FILE:${TARGET}>
    )
  else()
    # message("serial test")
    add_test(
      NAME ${TARGET}
      COMMAND $<TARGET_FILE:${TARGET}>
    )
  endif()
endmacro()

config_test_f03(test_f03_thermal_grid 1)
config_test_f03(test_f03_thermal_gw_grid 1)
target_sources(test_f03_thermal_gw_grid PRIVATE inspect_thermal_gw_grid.cpp)
if(LIBRPA_USE_LIBRI)
  target_include_directories(test_f03_thermal_gw_grid PRIVATE
    ${Cereal_INCLUDE_DIR} ${LibComm_INCLUDE_DIR} ${LibRI_INCLUDE_DIR})
endif()
add_test(NAME test_f03_thermal_gw_grid_mpi2
         COMMAND mpirun -np 2 $<TARGET_FILE:test_f03_thermal_gw_grid>)
foreach(shape b_shape f_shape)
  add_test(NAME test_f03_thermal_gw_grid_${shape}
           COMMAND $<TARGET_FILE:test_f03_thermal_gw_grid> ${shape})
  set_tests_properties(test_f03_thermal_gw_grid_${shape} PROPERTIES WILL_FAIL TRUE)
endforeach()
add_test(NAME test_f03_thermal_grid_bad_shape
         COMMAND $<TARGET_FILE:test_f03_thermal_grid> shape)
set_tests_properties(test_f03_thermal_grid_bad_shape PROPERTIES WILL_FAIL TRUE)


if(LIBRPA_USE_LIBRI)
  config_test_f03(test_f03_binding_libri 1)
  config_test_f03(test_f03_binding_call_internal_test 2)
  config_test_f03(test_f03_binding_exx_bccHe 4)
endif()
