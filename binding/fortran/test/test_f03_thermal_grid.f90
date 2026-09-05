program test_f03_thermal_grid
   use mpi
   use librpa_f03
   implicit none
   type(LibrpaHandler) :: h
   type(LibrpaOptions) :: opts
   integer :: ierr, provided
   real(dp) :: energies(2,1,1), occupations(2,1,1), times(3)
   complex(dp) :: transform(3,2)
   real(dp), allocatable :: frequencies(:), weights(:)
   character(len=16) :: mode

   call MPI_Init_thread(MPI_THREAD_MULTIPLE, provided, ierr)
   call h%init(MPI_COMM_WORLD)
   call h%set_scf_dimension(1, 1, 2, 2)
   energies(:,1,1) = [-0.25_dp, 0.75_dp]
   occupations = 1.0_dp
   call h%set_wg_ekb_efermi(1, 1, 2, occupations, energies, 0.0_dp)
   call h%set_fermi_dirac_reference(0.125_dp, 0.0_dp, 2.0_dp, 1e-10_dp)
   times = [0.5_dp, 3.0_dp, 7.0_dp]
   transform(:,1) = cmplx([1.0_dp, 2.0_dp, 5.0_dp], 0.0_dp, kind=dp)
   transform(:,2) = cmplx([1.0_dp, -3.0_dp, 2.0_dp], [2.0_dp, -1.0_dp, -1.0_dp], kind=dp)
   call get_command_argument(1, mode)
   if (trim(mode) == "shape") then
      call h%set_external_thermal_time_grid(8.0_dp, 2.0_dp, 1e-10_dp, times, transpose(transform))
      stop 0
   end if
   call h%set_external_thermal_time_grid(8.0_dp, 2.0_dp, 1e-10_dp, times, transform)
   ! Ensure the wrapper and C API own their copy after returning.
   times = 0.0_dp
   transform = (0.0_dp, 0.0_dp)
   call opts%init()
   opts%tfgrids_type = LIBRPA_TFGRID_FD_MATSUBARA
   opts%nfreq = 2
   opts%ntau = 3
   call h%get_imaginary_frequency_grids(opts, frequencies, weights)
   if (abs(frequencies(2) - acos(-1.0_dp)/4.0_dp) > 1e-13_dp) error stop "wrong frequency"
   ! This API returns quadrature weights; the correlation energy divides by 2*pi.
   if (abs(weights(1) - acos(-1.0_dp)/8.0_dp) > 1e-13_dp) error stop "wrong DC weight"
   if (abs(weights(2) - acos(-1.0_dp)/4.0_dp) > 1e-13_dp) error stop "wrong nonzero weight"
   call h%clear_external_thermal_time_grid()
   opts%ntau = 8
   call h%get_imaginary_frequency_grids(opts, frequencies, weights)
   if (abs(weights(1) - acos(-1.0_dp)/8.0_dp) > 1e-13_dp) error stop "clear removed FD reference"
   call h%free()
   call MPI_Finalize(ierr)
end program test_f03_thermal_grid
