// Public API headers
#include "librpa_enums.h"

// Internal headers
#include "../utils/error.h"
#include "../utils/profiler.h"
#include "../io/global_io.h"
#include "../math/rsh.h"
#include "../math/symmetry.h"
#include "dataset_helper.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace librpa_int
{

namespace
{

int max_basis_l(const std::vector<const AtomicBasis*>& bases)
{
    int lmax = -1;
    for (const auto* basis : bases)
    {
        if (basis != nullptr && basis->has_l_shells())
        {
            lmax = std::max(lmax, basis->get_max_l());
        }
    }
    return lmax;
}

} // namespace

void initialize_symmetry_context(Dataset &ds, const bool build_shell_rotations)
{
    auto &spg_symops = ds.spg_symops;

    if (spg_symops.empty())
    {
        spg_symops.push_back(SpaceGroupSymOp::IDENTITY);
    }

    auto &ctx = ds.symmetry_context;
    ctx.clear();
    ctx.set_crystal_structure(ds.pbc.latvec, ds.pbc.G, ds.atoms.types, ds.atoms.coords_frac);
    ctx.set_rspace_operations(spg_symops);
    ctx.build_periodic_mappings(ds.pbc, ds.pbc.Rlist);

    auto mark_available = [&ctx]() {
        ctx.set_available();
        ctx.print_summary(global::ofs_myid);
    };

    if (!build_shell_rotations)
    {
        mark_available();
        return;
    }

    const int lmax = max_basis_l({&ds.basis_wfc, &ds.basis_aux, &ds.basis_aux_shrink});
    if (lmax < 0)
    {
        mark_available();
        return;
    }

    auto basis_convention =
        is_basis_convention_set(ctx.basis_convention) ? ctx.basis_convention : ds.basis_convention;
    if (!is_basis_convention_set(basis_convention))
    {
        throw LIBRPA_RUNTIME_ERROR("Cannot initialize symmetry context without basis convention");
    }
    ds.basis_convention = basis_convention;
    ctx.build_rsh_rotations(basis_convention, lmax);
    ctx.build_kstar_member_rotations(lmax);
    mark_available();
}

void require_symmetry_shell_layouts(const Dataset &ds, const char *calculation)
{
    const auto &ctx = ds.symmetry_context;
    if (!ctx.available || ctx.rspace_operations.empty())
    {
        return;
    }

    if (!ds.basis_wfc.has_l_shells() || !ds.basis_aux.has_l_shells())
    {
        throw LIBRPA_RUNTIME_ERROR(
            std::string("Cannot use ") + calculation
            + " symmetry without l-shell basis layouts for AO and ABF species");
    }
}

void reject_spinor_symmetry_speedup(const Dataset &ds, const char *calculation)
{
    if (ds.mf.get_n_spinor() <= 1)
    {
        return;
    }
    throw LIBRPA_RUNTIME_ERROR(
        std::string("Cannot use ") + calculation
        + " symmetry speed-up with spinor wave functions; disable symmetry for spinor runs");
}

void validate_external_thermal_time_grid_reference(
    const Dataset &ds, const ExternalThermalTimeGrid &grid)
{
    const auto &reference = ds.mf.get_fermi_dirac_reference();
    if (!ds.is_scf_eigocc_set || !reference.enabled)
        throw LIBRPA_RUNTIME_ERROR(
            "external thermal grid requires SCF eigenvalues and an FD reference");
    const double beta_kbt = grid.beta_ha_inv * reference.kbt_ha;
    if (!std::isfinite(grid.beta_ha_inv) || grid.beta_ha_inv <= 0.0 || !std::isfinite(beta_kbt) ||
        std::abs(beta_kbt - 1.0) > 1e-10)
        throw LIBRPA_RUNTIME_ERROR("external thermal grid beta differs from the FD reference");
    if (!std::isfinite(grid.wmax_ha) || grid.wmax_ha <= 0.0 || !std::isfinite(grid.tolerance) ||
        grid.tolerance <= 0.0 || grid.tolerance >= 1.0)
        throw LIBRPA_RUNTIME_ERROR(
            "external thermal grid requires positive wmax and tolerance in (0,1)");
    if (!grid.frequency_indices.empty() &&
        (!std::isfinite(grid.rpa_wmax_ha) || grid.rpa_wmax_ha < grid.wmax_ha))
        throw LIBRPA_RUNTIME_ERROR("external thermal RPA grid requires finite rpa_wmax >= wmax");

    double largest_transition = 0.0;
    for (const auto &energies : ds.mf.get_eigenvals())
    {
        if (energies.size <= 0)
            throw LIBRPA_RUNTIME_ERROR("external thermal grid requires nonempty SCF energies");
        double lowest = std::numeric_limits<double>::infinity();
        double highest = -lowest;
        for (std::size_t i = 0; i < energies.size; ++i)
        {
            if (!std::isfinite(energies.c[i]))
                throw LIBRPA_RUNTIME_ERROR("external thermal grid found nonfinite SCF energies");
            lowest = std::min(lowest, energies.c[i]);
            highest = std::max(highest, energies.c[i]);
        }
        largest_transition = std::max(largest_transition, highest - lowest);
    }
    if (!std::isfinite(largest_transition) ||
        largest_transition > grid.wmax_ha + 1e-10 * std::max(1.0, grid.wmax_ha))
        throw LIBRPA_RUNTIME_ERROR(
            "external thermal grid wmax does not cover the SCF transition range");
}

void validate_external_thermal_gw_grid_reference(const Dataset &ds,
                                                 const ExternalThermalGWGrid &grid)
{
    const auto &reference = ds.mf.get_fermi_dirac_reference();
    if (!ds.is_scf_eigocc_set || !reference.enabled)
        throw LIBRPA_RUNTIME_ERROR(
            "external thermal GW grid requires SCF eigenvalues and an FD reference");
    const double beta_kbt = grid.transform.get_beta_ha_inv() * reference.kbt_ha;
    if (!std::isfinite(beta_kbt) || std::abs(beta_kbt - 1.0) > 1e-10)
        throw LIBRPA_RUNTIME_ERROR("external thermal GW grid beta differs from the FD reference");
    validate_fermi_dirac_chemical_potential(reference, ds.mf.get_efermi());
    for (const auto &energies : ds.mf.get_eigenvals())
    {
        if (energies.size <= 0)
            throw LIBRPA_RUNTIME_ERROR("external thermal GW grid requires nonempty SCF energies");
        for (std::size_t i = 0; i < energies.size; ++i)
        {
            const double xi = energies.c[i] - reference.chemical_potential_ha;
            if (!std::isfinite(energies.c[i]) || !std::isfinite(xi))
                throw LIBRPA_RUNTIME_ERROR("external thermal GW grid found nonfinite SCF energies");
            if (std::abs(xi) > grid.g_wmax_ha * (1.0 + 64 * std::numeric_limits<double>::epsilon()))
                throw LIBRPA_RUNTIME_ERROR(
                    "external thermal GW grid g_wmax does not cover abs(epsilon-mu)");
        }
    }
}

void initialize_ds_tfgrids(Dataset &ds, const LibrpaOptions &opts)
{
    global::profiler.start("initialize_ds_tfgrids");
    if (opts.nfreq <= 0)
        throw LIBRPA_RUNTIME_ERROR("number of frequency points must be positive");
    if (ds.external_thermal_gw_grid)
    {
        if (opts.tfgrids_type != LIBRPA_TFGRID_FD_MATSUBARA)
            throw LIBRPA_RUNTIME_ERROR("external thermal GW grid requires tfgrids_type = fd_matsubara");
        validate_external_thermal_gw_grid_reference(ds, *ds.external_thermal_gw_grid);
    }
    if (ds.external_thermal_time_grid)
    {
        const auto &grid = *ds.external_thermal_time_grid;
        if (opts.tfgrids_type != LIBRPA_TFGRID_FD_MATSUBARA)
            throw LIBRPA_RUNTIME_ERROR("external thermal grid requires tfgrids_type = fd_matsubara");
        if (opts.nfreq != grid.transform.nr || opts.ntau != grid.transform.nc)
            throw LIBRPA_RUNTIME_ERROR("external thermal grid dimensions differ from nfreq/ntau");
        validate_external_thermal_time_grid_reference(ds, grid);
        ds.tfg.reset(opts.nfreq);
        if (grid.frequency_indices.empty())
            ds.tfg.set_finite_beta_time_grid(grid.times, grid.beta_ha_inv, grid.transform);
        else
            ds.tfg.set_finite_beta_rpa_grid(grid.times, grid.beta_ha_inv, grid.frequency_indices,
                                          grid.correlation_weights, grid.transform);
        global::profiler.stop("initialize_ds_tfgrids");
        return;
    }
    ds.tfg.reset(opts.nfreq);
    if (opts.tfgrids_type == LIBRPA_TFGRID_FD_MATSUBARA)
    {
        if (!ds.mf.get_fermi_dirac_reference().enabled)
            throw LIBRPA_RUNTIME_ERROR(
                "fd_matsubara requires an explicit Fermi-Dirac occupation reference");
        if (opts.ntau <= 0)
            throw LIBRPA_RUNTIME_ERROR("fd_matsubara requires positive ntau");
        ds.tfg.generate_finite_beta_matsubara(
            static_cast<std::size_t>(opts.ntau),
            1.0 / ds.mf.get_fermi_dirac_reference().kbt_ha);
        global::profiler.stop("initialize_ds_tfgrids");
        return;
    }
    if (ds.mf.get_fermi_dirac_reference().enabled)
        throw LIBRPA_RUNTIME_ERROR(
            "Fermi-Dirac finite-temperature mode requires tfgrids_type = fd_matsubara");
    double emin = opts.tfgrids_freq_min;
    double eintv = opts.tfgrids_freq_interval;
    double emax = opts.tfgrids_freq_max;
    double tmin = opts.tfgrids_time_min;
    double tintv = opts.tfgrids_time_interval;
    double regulation = opts.minimax_regulation;
    if (opts.tfgrids_type == LIBRPA_TFGRID_MINIMAX)
    {
        double emin_mf, emax_mf;
        ds.mf.get_E_min_max(emin_mf, emax_mf);
        emax = opts.minimax_emax > 0 ? opts.minimax_emax : emax_mf;
        emin = opts.minimax_emin > 0 ? opts.minimax_emin : emin_mf;
    }
    ds.tfg.generate(opts.tfgrids_type, emin, eintv, emax, tmin, tintv, regulation);
    global::profiler.stop("initialize_ds_tfgrids");
}

void initialize_ds_global_ddla(Dataset &ds, const LibrpaOptions &opts)
{
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    if (opts.use_gpu_replace_scalapack == LIBRPA_SWITCH_ON && ds.blacs_h.ddla_handle == nullptr)
    {
        global::profiler.start("initialize_ds_global_ddla");
        ds.blacs_h.init_ddla_handle();
        global::profiler.stop("initialize_ds_global_ddla");
    }
#else
    (void)ds;
    (void)opts;
#endif
}

static void collect_atpairs_all(Dataset &ds)
{
    const auto &comm_h = ds.comm_h;
    const auto &atpairs_local = ds.atpairs_local;
    const int np_this = atpairs_local.size();
    std::vector<int> np_all(comm_h.nprocs, 0);
    np_all[comm_h.myid] = np_this;
    int np_max;
    comm_h.allreduce(&np_this, &np_max, 1, MPI_MAX);
    comm_h.allreduce(MPI_IN_PLACE, np_all.data(), comm_h.nprocs, MPI_SUM);
    if (np_max == 0) return;
    std::vector<size_t> pairs_all(comm_h.nprocs * np_max * 2, 0);
    const int st = comm_h.myid * np_max * 2;
    for (int ip = 0; ip < np_this; ip++)
    {
        pairs_all[st + ip * 2] = atpairs_local[ip].first;
        pairs_all[st + ip * 2 + 1] = atpairs_local[ip].second;
    }
    comm_h.allreduce(MPI_IN_PLACE, pairs_all.data(), pairs_all.size(), MPI_SUM);
    for (int pid = 0; pid < comm_h.nprocs; pid++)
    {
        if (pid == comm_h.myid)
        {
            std::set<atpair_t> atpairs_this(atpairs_local.cbegin(), atpairs_local.cend());
            ds.atpairs_unique_all.emplace(pid, atpairs_this);
        }
        else
        {
            const int np_this = np_all[pid];
            std::set<atpair_t> atpairs_this;
            const int st = pid * np_max * 2;
            for (int ip = 0; ip < np_this; ip++)
            {
                atpairs_this.insert({pairs_all[st + ip * 2], pairs_all[st + ip * 2 + 1]});
            }
            ds.atpairs_unique_all.emplace(pid, atpairs_this);
        }
    }
}

void initialize_ds_atpairs_local(Dataset &ds, LibrpaParallelRouting routing)
{
    global::profiler.start(__FUNCTION__);

    ds.atpairs_local.clear();
    const int n_atoms_basis_wfc = ds.basis_wfc.n_atoms;
    const int n_atoms_basis_aux = ds.basis_aux.n_atoms;
    const int n_atoms_struc = ds.atoms.size();
    const int n_atoms =
        n_atoms_struc > 0 ? n_atoms_struc : std::max(n_atoms_basis_aux, n_atoms_basis_wfc);
    if (n_atoms == 0)
        throw LIBRPA_RUNTIME_ERROR(
            "Number of atoms can not be extracted, please set structure or basis first");

    if (routing == LIBRPA_ROUTING_AUTO)
    {
        throw LIBRPA_RUNTIME_ERROR(
            "internal error: routing should be decided before initialize_ds_atpairs_local, not "
            "AUTO");
    }
    else if (routing == LIBRPA_ROUTING_ATOMPAIR || routing == LIBRPA_ROUTING_LIBRI)
    {
        auto tri_local_atpair = librpa_int::dispatch_upper_triangular_tasks(
            n_atoms, ds.blacs_h.myid, ds.blacs_h.nprows, ds.blacs_h.npcols, ds.blacs_h.myprow,
            ds.blacs_h.mypcol);
        for (const auto &p : tri_local_atpair) ds.atpairs_local.emplace_back(p);
    }
    else
    {
        ds.atpairs_local = generate_atom_pair_from_nat(n_atoms, false);
    }
    collect_atpairs_all(ds);
    global::profiler.stop(__FUNCTION__);
}

void initialize_ds_exx(Dataset &ds, const LibrpaOptions &opts)
{
    global::profiler.start("initialize_ds_exx");
    const bool use_symmetry = opts.use_symmetry_exx == LIBRPA_SWITCH_ON;
    if (use_symmetry)
    {
        reject_spinor_symmetry_speedup(ds, "EXX");
    }
    initialize_symmetry_context(ds, use_symmetry);
    if (use_symmetry)
    {
        require_symmetry_shell_layouts(ds, "EXX");
    }
    const bool is_eigvec_k_distributed = opts.use_kpara_scf_eigvec == LIBRPA_SWITCH_ON;
    ds.p_exx = std::make_unique<librpa_int::Exx>(ds.mf, ds.basis_wfc, ds.pbc, ds.symmetry_context,
                                                 ds.scfk_blacs_ctxt, ds.bandk_blacs_ctxt,
                                                 ds.desc_wfc_kb_full,
                                                 is_eigvec_k_distributed,
                                                 use_symmetry);
    ds.p_exx->libri_threshold_C = opts.libri_exx_threshold_C;
    ds.p_exx->libri_threshold_D = opts.libri_exx_threshold_D;
    ds.p_exx->libri_threshold_V = opts.libri_exx_threshold_V;
    global::profiler.stop("initialize_ds_exx");
}

void initialize_ds_chi0(Dataset &ds, const LibrpaOptions &opts)
{
    global::profiler.start("initialize_ds_chi0");
    const bool use_symmetry = opts.use_symmetry_rpa == LIBRPA_SWITCH_ON;
    if (use_symmetry)
    {
        reject_spinor_symmetry_speedup(ds, "RPA/chi0");
    }
    initialize_symmetry_context(ds, use_symmetry);
    if (use_symmetry)
    {
        require_symmetry_shell_layouts(ds, "RPA/chi0");
    }
    const bool is_eigvec_k_distributed = opts.use_kpara_scf_eigvec == LIBRPA_SWITCH_ON;
    if (opts.use_shrink_abfs == LIBRPA_SWITCH_ON)
        ds.p_chi0 = std::make_unique<librpa_int::Chi0>(ds.mf, ds.basis_wfc, ds.basis_aux_shrink, ds.pbc,
                                                       ds.symmetry_context,
                                                       ds.tfg, ds.scfk_blacs_ctxt, ds.desc_wfc_kb_full,
                                                       is_eigvec_k_distributed,
                                                       use_symmetry);
    else
        ds.p_chi0 = std::make_unique<librpa_int::Chi0>(ds.mf, ds.basis_wfc, ds.basis_aux, ds.pbc,
                                                       ds.symmetry_context,
                                                       ds.tfg, ds.scfk_blacs_ctxt, ds.desc_wfc_kb_full,
                                                       is_eigvec_k_distributed,
                                                       use_symmetry);
    ds.p_chi0->gf_threshold = opts.gf_threshold;
    ds.p_chi0->libri_collect_s0_chunk = opts.libri_chi0_collect_s0_chunk;
    ds.p_chi0->libri_collect_max_bytes = opts.libri_chi0_collect_max_bytes;
    ds.p_chi0->nbands_G = opts.n_bands_chi0;
    ds.p_chi0->libri_threshold_C = opts.libri_chi0_threshold_C;
    ds.p_chi0->libri_threshold_G = opts.libri_chi0_threshold_G;
    global::profiler.stop("initialize_ds_chi0");
}

void initialize_ds_g0w0(Dataset &ds, const LibrpaOptions &opts)
{
    global::profiler.start("initialize_ds_g0w0");
    const bool use_symmetry = opts.use_symmetry_gw == LIBRPA_SWITCH_ON;
    if (use_symmetry)
    {
        reject_spinor_symmetry_speedup(ds, "GW");
    }
    initialize_symmetry_context(ds, use_symmetry);
    if (use_symmetry)
    {
        require_symmetry_shell_layouts(ds, "GW");
    }
    const bool is_eigvec_k_distributed = opts.use_kpara_scf_eigvec == LIBRPA_SWITCH_ON;
    // global::ofs_myid << "is_eigvec_k_distributed " << is_eigvec_k_distributed << std::endl;
    ds.p_g0w0 = std::make_unique<librpa_int::G0W0>(ds.mf, ds.basis_wfc, ds.pbc,
                                                   ds.symmetry_context, ds.tfg,
                                                   ds.scfk_blacs_ctxt, ds.bandk_blacs_ctxt,
                                                   ds.desc_wfc_kb_full,
                                                   is_eigvec_k_distributed,
                                                   use_symmetry);
    ds.p_g0w0->libri_threshold_C = opts.libri_g0w0_threshold_C;
    ds.p_g0w0->libri_threshold_G = opts.libri_g0w0_threshold_G;
    ds.p_g0w0->libri_threshold_Wc = opts.libri_g0w0_threshold_Wc;
    ds.p_g0w0->output_dir = opts.output_dir;
    ds.p_g0w0->output_sigc_ks_kf = opts.output_gw_sigc_ks_kf == LIBRPA_SWITCH_ON;
    ds.p_g0w0->output_sigc_ks_mat_kf = opts.output_gw_sigc_ks_mat_kf == LIBRPA_SWITCH_ON;
    ds.p_g0w0->output_sigc_mat_kf = opts.output_gw_sigc_mat_kf == LIBRPA_SWITCH_ON;
    ds.p_g0w0->output_sigc_mat_rt = opts.output_gw_sigc_mat_rt == LIBRPA_SWITCH_ON;
    ds.p_g0w0->output_sigc_mat_rf = opts.output_gw_sigc_mat_rf == LIBRPA_SWITCH_ON;
    ds.p_g0w0->output_wc_rf = opts.output_wc_rf == LIBRPA_SWITCH_ON;
    ds.p_g0w0->output_wc_rf_atom_pair = opts.output_wc_rf_atom_pair == LIBRPA_SWITCH_ON;
    ds.p_g0w0->ifreq_output_wc_start = opts.ifreq_output_wc_start;
    ds.p_g0w0->ifreq_output_wc_end = opts.ifreq_output_wc_end;
    global::profiler.stop("initialize_ds_g0w0");
}

void initialize_ds_headwing(Dataset &ds, const LibrpaOptions &opts, const bool need_wing)
{
    if (opts.replace_w_head != LIBRPA_SWITCH_ON ||
        (opts.option_dielect_func != 3 && opts.option_dielect_func != 4))
    {
        return;
    }

    global::profiler.start("initialize_ds_headwing");

    if (ds.p_headwing && (!need_wing || ds.p_headwing->has_wing()))
    {
        global::profiler.stop("initialize_ds_headwing");
        return;
    }

    if (ds.p_headwing && need_wing)
    {
        const auto &headwing_cs =
            opts.use_shrink_abfs == LIBRPA_SWITCH_ON ? ds.cs_data_shrink : ds.cs_data;
        ds.p_headwing->cal_wing(headwing_cs, opts.sqrt_coulomb_threshold, ds.vq);
        ds.p_headwing->test_wing();
        global::profiler.stop("initialize_ds_headwing");
        return;
    }

    if (ds.velocity_matrix.empty())
    {
        throw LIBRPA_RUNTIME_ERROR(
            "analytic head/wing requested but velocity matrix is not set");
    }

    MeanField &mf = ds.mf;
    if (!mf.initialized())
        throw LIBRPA_RUNTIME_ERROR("analytic head/wing meanfield is not initialized");

    if (static_cast<int>(ds.pbc.kfrac_list.size()) != mf.get_n_kpoints())
        throw LIBRPA_RUNTIME_ERROR(
            "analytic head/wing k-point list is inconsistent with meanfield");

    const auto &headwing_basis_aux =
        opts.use_shrink_abfs == LIBRPA_SWITCH_ON ? ds.basis_aux_shrink : ds.basis_aux;
    if (!headwing_basis_aux.initialized())
        throw LIBRPA_RUNTIME_ERROR("analytic head/wing auxiliary basis is not initialized");

    const auto &freqs = ds.tfg.get_freq_nodes();
    ds.p_headwing = std::make_unique<diele_func>(
        mf, ds.velocity_matrix, ds.pbc.kfrac_list, ds.basis_wfc,
        headwing_basis_aux, freqs, mf.get_n_aos(), mf.get_n_states(),
        mf.get_n_spins(), headwing_basis_aux.nb_total, ds.pbc, ds.comm_h, ds.blacs_h,
        &ds.scfk_blacs_ctxt);
    ds.p_headwing->configure_strict_2d_coulomb_head(
        opts.use_2d_dielectric == LIBRPA_SWITCH_ON);
    ds.p_headwing->use_soc = mf.get_n_spinor() > 1;
    ds.p_headwing->debug = global::should_output(LIBRPA_VERBOSE_DEBUG);
    if (ds.symmetry_context.available && !ds.symmetry_context.kstars.empty())
    {
        ds.p_headwing->set_symmetry_context(ds.symmetry_context);
        ds.p_headwing->use_symmetry = true;
        ds.p_headwing->atom_nw = ds.basis_wfc.get_atom_nb_map();
        ds.p_headwing->coord_frac = ds.symmetry_context.input_coord_frac;
    }
    ds.p_headwing->init(opts.sqrt_coulomb_threshold, ds.vq);
    ds.p_headwing->cal_head();
    ds.epsmacs_imagfreq = ds.p_headwing->get_head_vec();
    ds.omegas_imagfreq = freqs;
    ds.p_headwing->test_head();

    if (need_wing)
    {
        const auto &headwing_cs =
            opts.use_shrink_abfs == LIBRPA_SWITCH_ON ? ds.cs_data_shrink : ds.cs_data;
        ds.p_headwing->cal_wing(headwing_cs, opts.sqrt_coulomb_threshold, ds.vq);
        ds.p_headwing->test_wing();
    }

    global::profiler.stop("initialize_ds_headwing");
}

}  // namespace librpa_int
