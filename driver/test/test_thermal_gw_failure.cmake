if(NOT CASE STREQUAL "missing" AND NOT CASE STREQUAL "malformed")
  message(FATAL_ERROR "CASE must be missing or malformed")
endif()
if(NOT MPIEXEC)
  # Legacy cmake.inc configurations do not necessarily call find_package(MPI).
  unset(MPIEXEC)
  unset(MPIEXEC CACHE)
  find_program(MPIEXEC NAMES mpiexec mpirun)
  if(NOT MPIEXEC)
    message(FATAL_ERROR "A working mpiexec or mpirun is required for this test")
  endif()
endif()
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef suffix)
set(work "${WORK_ROOT}/${CASE}_${suffix}")
file(MAKE_DIRECTORY "${work}")
file(WRITE "${work}/librpa.in"
  "task = rpa\ntfgrids_type = fd_matsubara\nnfreq = 2\nntau = 3\n"
  "fn_thermal_gw_grid = thermal_gw.dat\n")
# Two half-filled states at mu=0 give exact FD occupations at any positive kBT.
file(WRITE "${work}/band_out" "1\n1\n2\n2\n0\n1 1\n1 1 0 0\n2 1 0 0\n")
file(WRITE "${work}/stru_out"
  "1 0 0\n0 1 0\n0 0 1\n"
  "6.283185307179586 0 0\n0 6.283185307179586 0\n0 0 6.283185307179586\n"
  "1\n0 0 0 1\n")
file(WRITE "${work}/bz_sampling_out" "1 1 1\n1 1\n1 1 0 0 0 0 0 0 1 1\n")
file(WRITE "${work}/thermal_occupation_v1.dat"
  "format thermal_occupation_v1\noccupation_model fermi_dirac\n"
  "chemical_potential_ha 0\nkbt_ha 0.125\nsmearing_sigma_ry 0.25\n"
  "occupation_storage band_out_times_nk\nmax_occupation_per_band 2\n"
  "spin_channels 1\nkpoints_per_spin 1\nbands 2\n")
if(CASE STREQUAL "malformed")
  file(WRITE "${work}/thermal_gw.dat" "LIBRPA_THERMAL_GW_V2\n")
  set(expected "external thermal GW grid ./thermal_gw.dat: unsupported version")
else()
  set(expected "cannot open external thermal GW grid: ./thermal_gw.dat")
endif()

execute_process(
  COMMAND "${MPIEXEC}" -np 2 "${CMAKE_COMMAND}" -E env
          OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "${RPA_EXE}"
  WORKING_DIRECTORY "${work}"
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 30)
set(log "case=${CASE}\nwork=${work}\nexit=${result}\n${output}\n${error}")
if(NOT LOG_FILE)
  set(LOG_FILE "${work}/launcher.log")
endif()
get_filename_component(log_directory "${LOG_FILE}" DIRECTORY)
file(MAKE_DIRECTORY "${log_directory}")
file(WRITE "${LOG_FILE}" "${log}")
if(NOT result MATCHES "^[0-9]+$" OR result STREQUAL "0")
  message(FATAL_ERROR "Expected a nonzero normal exit, got '${result}'; see ${LOG_FILE}")
endif()
foreach(required "Fermi-Dirac occupation reference enabled:" "Error: libRPA failed"
    "Thermal GW metadata on MPI rank 0: ${expected}"
    "Thermal GW metadata on MPI rank 1: ${expected}")
  string(FIND "${log}" "${required}" found)
  if(found EQUAL -1)
    message(FATAL_ERROR "Missing '${required}'; see ${LOG_FILE}")
  endif()
endforeach()
# Supplemental only: Open MPI 5 can omit missing-finalize diagnostics for nonzero exits.
# This integration test verifies the input error, not that MPI_Finalize was called.
string(TOLOWER "${log}" lower_log)
if(lower_log MATCHES "exit(ed|ing) improperly|without calling.*finalize|mpi_abort|segmentation fault|exited on signal")
  message(FATAL_ERROR "MPI did not exit cleanly; see ${LOG_FILE}")
endif()
message(STATUS "Expected ${CASE} GW input failure on both MPI ranks; log: ${LOG_FILE}")
