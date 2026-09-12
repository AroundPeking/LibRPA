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
    test_thermal_api
    test_thermal_chi0
    test_external_thermal_grid
    test_external_thermal_api
    test_external_thermal_rpa_grid
    test_external_thermal_rpa_api
    test_external_thermal_gw_api
    test_thermal_gw_transform
    test_thermal_gw_screening
    test_thermal_occupation
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
target_compile_definitions(test_external_thermal_grid PRIVATE
  LIBRPA_SPARSE_TAU_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/data/sparse_ir_tau_beta8_wmax2.dat")
config_test(test_external_thermal_gw 1)
target_compile_definitions(test_external_thermal_gw PRIVATE
  LIBRPA_SPARSE_GW_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/data/sparse_ir_gw_beta8_g1_w2.dat")
target_compile_options(test_external_thermal_gw PRIVATE -UNDEBUG)
set_tests_properties(test_external_thermal_gw PROPERTIES
  ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
add_test(NAME test_rpa_logdet_mpi2 COMMAND mpirun -np 2 $<TARGET_FILE:test_rpa_logdet>)
add_test(NAME test_external_thermal_gw_api_mpi2 COMMAND mpirun -np 2 $<TARGET_FILE:test_external_thermal_gw_api>)
add_test(NAME test_rpa_logdet_mpi4 COMMAND mpirun -np 4 $<TARGET_FILE:test_rpa_logdet>)
add_test(NAME test_rpa_headwing_mpi4 COMMAND mpirun -np 4 $<TARGET_FILE:test_rpa_headwing>)
set_tests_properties(test_rpa_headwing test_rpa_headwing_mpi4 PROPERTIES
  ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
target_sources(test_strict_2d_coulomb_head_metadata
  PRIVATE ../../driver/reader_2d_coulomb_head.cpp)
# These suites use standard assert() and must remain active in Release builds.
target_compile_options(test_sternheimer_rpa PRIVATE -UNDEBUG)
target_compile_options(test_sternheimer_symmetry PRIVATE -UNDEBUG)
target_compile_options(test_symmetry_context PRIVATE -UNDEBUG)
target_compile_options(test_rpa_headwing PRIVATE -UNDEBUG)
target_compile_options(test_si_density_symmetry PRIVATE -UNDEBUG)
target_compile_options(test_thermal_gw_transform PRIVATE -UNDEBUG)
target_compile_definitions(test_si_density_symmetry PRIVATE
  SI_DENSITY_REFERENCE="${CMAKE_CURRENT_SOURCE_DIR}/data/si_k444_density_reference.txt")

# Internal thermal full-q Wc spacetime transform.
config_test(test_thermal_wc_spacetime 1)
target_compile_options(test_thermal_wc_spacetime PRIVATE -UNDEBUG)
add_test(NAME test_thermal_wc_spacetime_mpi2
  COMMAND mpirun -np 2 $<TARGET_FILE:test_thermal_wc_spacetime>)
set_tests_properties(test_thermal_wc_spacetime test_thermal_wc_spacetime_mpi2 PROPERTIES
  TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
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

if(LIBRPA_USE_LIBRI AND _LIBRPA_USE_BUNDLED_LIBCOMM)
  config_test(test_libcomm_bounded_exchange 4)
endif()

# TODO(minye): make use of GTest

if(LIBRPA_USE_LIBRI)
  config_test(test_thermal_gw_spacetime 1)
  config_test(test_thermal_sigc_symmetry 1)
  config_test(test_thermal_sigc_symmetry_api 1)
  add_test(NAME test_thermal_sigc_symmetry_api_mpi4
    COMMAND mpirun -np 4 $<TARGET_FILE:test_thermal_sigc_symmetry_api>)
  add_test(NAME test_thermal_sigc_symmetry_mpi4
    COMMAND mpirun -np 4 $<TARGET_FILE:test_thermal_sigc_symmetry>)
  set_tests_properties(test_thermal_sigc_symmetry test_thermal_sigc_symmetry_mpi4
    test_thermal_sigc_symmetry_api test_thermal_sigc_symmetry_api_mpi4 PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
  config_test(test_generic_thermal_gw_api 1)
  config_test(test_thermal_gw_prepare 1)
  config_test(test_thermal_headwing_cache 1)
  add_test(NAME test_thermal_headwing_cache_mpi4
    COMMAND mpirun -np 4 $<TARGET_FILE:test_thermal_headwing_cache>)
  add_test(NAME test_thermal_gw_prepare_mpi4
    COMMAND mpirun -np 4 $<TARGET_FILE:test_thermal_gw_prepare>)
  config_test(test_thermal_sigc_projection 1)
  add_test(NAME test_generic_thermal_gw_api_mpi4
    COMMAND mpirun -np 4 $<TARGET_FILE:test_generic_thermal_gw_api>)
  add_test(NAME test_thermal_sigc_projection_mpi4
    COMMAND mpirun -np 4 $<TARGET_FILE:test_thermal_sigc_projection>)
  add_test(NAME test_thermal_gw_spacetime_mpi2
    COMMAND mpirun -np 2 $<TARGET_FILE:test_thermal_gw_spacetime>)
  set_tests_properties(test_thermal_gw_spacetime test_thermal_gw_spacetime_mpi2 PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
  foreach(target test_thermal_gw_spacetime test_thermal_sigc_symmetry
      test_thermal_sigc_symmetry_api test_generic_thermal_gw_api
      test_thermal_gw_prepare test_thermal_headwing_cache test_thermal_sigc_projection)
    target_compile_options(${target} PRIVATE -UNDEBUG)
  endforeach()
endif()
