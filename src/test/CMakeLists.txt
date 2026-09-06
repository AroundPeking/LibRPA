# a simple macro to add source, include, link main library and run test
# see also
# - https://stackoverflow.com/questions/53868015/avoid-repetitive-cmake-code-with-multiple-targets
# - https://stackoverflow.com/questions/19278510/cmake-ctest-add-test-command-appends-extra-quotes-to-my-command-parameters
macro(config_test TARGET NUMPROC)
  add_executable(${TARGET} ${TARGET}.cpp)
  target_link_libraries(${TARGET} PRIVATE rpa_lib)
  # Math and MPI symbols will have been included in rpa_lib for static library case.
  # Only link them when shared library is built.
  if(BUILD_SHARED_LIBS)
    target_link_libraries(${TARGET} PRIVATE ${math_libs} ${parallel_libs})
  endif()
  # TODO: separate libri test targets
  if(LIBRPA_USE_LIBRI)
    target_include_directories(${TARGET}
      PRIVATE
        ${Cereal_INCLUDE_DIR}
        ${LibComm_INCLUDE_DIR}
        ${LibRI_INCLUDE_DIR}
      )
  endif()
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

# add serial tests
foreach(target
    test_analycont
    test_atom
    test_atomic_basis
    test_base_utility
    test_blas_lapack
    test_complexmatrix
    test_fitting
    test_io
    test_symmetry_context
    test_si_density_symmetry
    test_task_helper
    test_arbitrary_float
    test_geometry
    test_lebedev_laikov
    test_libri_utils
    # test_librpa
    test_matrix
    test_matrix_m
    test_meanfield
    test_pbc
    test_profiler
    test_qpe_solver
    test_rsh
    test_rpa_headwing
    test_rpa_logdet
    test_strict_2d_coulomb_head_metadata
    test_sternheimer_rpa
    test_sternheimer_symmetry
    test_sternheimer_qsum
    test_symmetry
    test_timefreq
    test_vector3
    test_wigner_rotation
    )
  config_test(${target} 1)
endforeach()

target_sources(test_task_helper PRIVATE ../../driver/tasks/task_helper.cpp)
target_sources(test_strict_2d_coulomb_head_metadata
  PRIVATE ../../driver/reader_2d_coulomb_head.cpp)
# These suites use standard assert() and must remain active in Release builds.
target_compile_options(test_sternheimer_rpa PRIVATE -UNDEBUG)
target_compile_options(test_sternheimer_symmetry PRIVATE -UNDEBUG)
target_compile_options(test_symmetry_context PRIVATE -UNDEBUG)
target_compile_options(test_meanfield PRIVATE -UNDEBUG)

# add 4-process tests
foreach(target
  test_atomic_basis_blacs
  test_base_mpi
  test_blacs
  test_dataset
  test_matrix_m_mpi
  test_meanfield_mpi
  test_kpoint_blacs_parallel_context
  test_shared_memory_mpi
  )
  config_test(${target} 4)
endforeach()

target_compile_options(test_meanfield_mpi PRIVATE -UNDEBUG)
set_tests_properties(test_symmetry_context test_si_density_symmetry
  test_meanfield test_meanfield_mpi PROPERTIES LABELS "symmetry")

if(LIBRPA_USE_LIBRI AND _LIBRPA_USE_BUNDLED_LIBCOMM)
  config_test(test_libcomm_bounded_exchange 4)
endif()

foreach(target test_rpa_headwing test_symmetry)
  target_compile_options(${target} PRIVATE -UNDEBUG)
endforeach()

add_test(NAME test_rpa_logdet_mpi
  COMMAND mpirun -np 4 $<TARGET_FILE:test_rpa_logdet>)
add_test(NAME test_rpa_headwing_mpi
  COMMAND mpirun -np 4 $<TARGET_FILE:test_rpa_headwing>)
set_tests_properties(test_rpa_headwing_mpi PROPERTIES TIMEOUT 120)
config_test(test_headwing_descriptor 2)
set_tests_properties(test_headwing_descriptor PROPERTIES TIMEOUT 30)

# TODO(minye): make use of GTest

target_compile_definitions(test_si_density_symmetry PRIVATE
  SI_DENSITY_REFERENCE="${CMAKE_CURRENT_SOURCE_DIR}/data/si_k444_density_reference.txt")
