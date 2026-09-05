program test_f03_thermal_gw_grid
   use iso_c_binding, only: c_int
   use mpi
   use librpa_f03
   implicit none
   interface
      function inspect_gw_grid(expected) bind(c, name="inspect_f03_thermal_gw_grid") result(ok)
         import c_int
         integer(c_int), value :: expected
         integer(c_int) :: ok
      end function
   end interface
   type(LibrpaHandler) :: h
   type(LibrpaOptions) :: opts
   integer :: ierr, provided, j
   integer :: bosons(3), fermions(4)
   real(dp) :: energies(2,1,1), occupations(2,1,1), times(2)
   complex(dp) :: b(3,2), f(2,4)
   real(dp), allocatable :: frequencies(:), weights(:)
   character(len=16) :: mode

   call MPI_Init_thread(MPI_THREAD_MULTIPLE, provided, ierr)
   call h%init(MPI_COMM_WORLD)
   call h%set_scf_dimension(1, 1, 2, 2)
   energies(:,1,1) = [-0.25_dp, 0.75_dp]
   occupations = 1.0_dp
   call h%set_wg_ekb_efermi(1, 1, 2, occupations, energies, 0.5_dp)
   call h%set_fermi_dirac_reference(0.125_dp, 0.5_dp, 2.0_dp, 1e-10_dp)
   times = [0.5_dp, 7.0_dp]
   bosons = [2, 0, -2]
   fermions = [3, -1, 0, -4]
   b(:,1) = cmplx([1.0_dp, 0.125_dp, 2.0_dp], [2.0_dp, 0.0_dp, 3.0_dp], kind=dp)
   b(:,2) = cmplx([3.0_dp, 0.125_dp, 4.0_dp], [4.0_dp, 0.0_dp, 5.0_dp], kind=dp)
   do j = 1, 4
      f(:,j) = cmplx([real(2*j-1,dp), real(2*j,dp)], [real(10-2*j,dp), real(9-2*j,dp)], kind=dp)
   end do
   call get_command_argument(1, mode)
   if (trim(mode) == "b_shape") then
      call h%set_external_thermal_gw_grid(8.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, 1e-10_dp, &
                                        times, bosons, fermions, transpose(b), f)
      stop 0
   else if (trim(mode) == "f_shape") then
      call h%set_external_thermal_gw_grid(8.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, 1e-10_dp, &
                                        times, bosons, fermions, b, transpose(f))
      stop 0
   end if
   call h%set_external_thermal_gw_grid(8.0_dp, 1.0_dp, 2.0_dp, 3.0_dp, 1e-10_dp, &
                                     times, bosons, fermions, b, f)
   times = 0
   bosons = 0
   fermions = 0
   b = (0.0_dp, 0.0_dp)
   f = (0.0_dp, 0.0_dp)
   if (inspect_gw_grid(1_c_int) /= 1) error stop "Fortran GW operator data changed"
   call opts%init()
   opts%tfgrids_type = LIBRPA_TFGRID_FD_MATSUBARA
   opts%nfreq = 5
   opts%ntau = 8
   call h%get_imaginary_frequency_grids(opts, frequencies, weights)
   if (size(frequencies) /= 5) error stop "GW dimensions replaced response grid"
   call h%clear_external_thermal_time_grid()
   if (inspect_gw_grid(1_c_int) /= 1) error stop "response clear removed GW operators"
   call h%clear_external_thermal_gw_grid()
   if (inspect_gw_grid(0_c_int) /= 1) error stop "GW clear changed FD reference"
   call h%free()
   call MPI_Finalize(ierr)
end program
