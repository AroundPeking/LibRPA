/*!
 @file meanfield.h
 @brief Utilities to handle the mean-field starting point for many-body calculation
 */
#ifndef MEANFIELD_H
#define MEANFIELD_H

#include <array>
#include <complex>
#include <vector>
#include <map>
#include <utility>
#include "atom.h"
#include "pbc.h"
#include "symmetry_context.h"
#include "symmetry_spin_kernel.h"
#include "thermal_occupation.h"
#include "../math/matrix.h"
#include "../math/complexmatrix.h"
#include "../math/vector3_order.h"

namespace librpa_int {

//! Opt-in real-space tetrahedron transform for the finite-temperature metal path.
bool tetrahedron_full_gw_requested();

//! Opt-in source-G tetrahedron transform used by the isolated consistency control.
bool tetrahedron_source_g_requested();

//! Image multiplicity for the opt-in unfolded real-space tetrahedron path.
Vector3_Order<int> tetrahedron_unfolded_image_factors();

//! Return normalized Fourier weights for a piecewise-linear tetrahedron transform.
//!
//! The entries are ordered like kfrac_list and already include the BZ integration
//! weight and exp(-i 2 pi k.R).  At R=(0,0,0) they sum to one.
std::vector<std::complex<double>> build_tetrahedron_fourier_weights(
    const std::vector<Vector3_Order<double>>& kfrac_list, const Vector3_Order<int>& R);

//! Map the BvK real-space coefficients back to a continuous interpolant coefficient.
//! The returned entries correspond to bvk_R and retain the full mesh twist.
std::vector<std::complex<double>> build_tetrahedron_unfolded_coefficients(
    const std::vector<Vector3_Order<double>>& kfrac_list,
    const std::vector<Vector3_Order<int>>& bvk_R, const Vector3_Order<int>& X);

//! Expanded real-space tile used before the four LRI contraction sectors are folded.
struct TetrahedronUnfoldedGrid
{
    Vector3_Order<int> period;
    std::vector<Vector3_Order<int>> Rlist;
};

//! Construct an odd-image real-space tile around the original centered BvK cell.
TetrahedronUnfoldedGrid build_tetrahedron_unfolded_grid(
    const Vector3_Order<int>& bvk_period, const Vector3_Order<int>& image_factors);

//! Fold an unfolded lattice translation into the centered original BvK representative.
Vector3_Order<int> fold_tetrahedron_unfolded_translation(
    const Vector3_Order<int>& translation, const Vector3_Order<int>& bvk_period);

//! Object of the meanfield input
/*!
  @note Energies are saved in Hartree unit.
 */
class MeanField
{
private:
    //! number of spin channels
    int n_spins;
    //! number of atomic orbitals
    int n_aos;
    //! number of bands
    int n_states;
    //! number of kpoints
    int n_kpoints;
    //! number of spinor components per wavefunction, e.g. 1 or 2
    int n_spinor;

    // Local dimensions for parallel distribution of eigenvectors
    int n_aos_local;
    int i_ao_start;
    int n_states_local;
    int i_state_start;

    //! eigenvalues, (n_spins, n_kpoints, n_states)
    std::vector<matrix> eskb;
    //! occupation weight, scaled by n_kpoints. (n_spins, n_kpoints, n_states)
    std::vector<matrix> wg;
    //! eigenvector, (n_spins, n_spinor, n_kpoints_local, n_states_local, n_aos_local)
    // TODO: parallelize
    std::map<int, std::map<int, std::map<int, ComplexMatrix>>> wfc;
    //! Fermi energy
    double efermi;
    //! Explicit finite-temperature occupation model. Disabled by default.
    FermiDiracReference fermi_dirac_reference;
    void resize(int ns, int nk, int nb, int nao, int n_spinor, int st_ib, int nb_local, int st_iao, int nao_local);

public:
    MeanField()
        : n_spins(0),
          n_aos(0),
          n_states(0),
          n_kpoints(0),
          n_spinor(0),
          n_aos_local(0),
          i_ao_start(-1),
          n_states_local(0),
          i_state_start(-1),
          eskb(),
          wg(),
          wfc(),
          efermi(0),
          fermi_dirac_reference() {};
    MeanField(int ns, int nk, int nb, int nao, int n_spinor = 1);
    MeanField(int ns, int nk, int nb, int nao, int n_spinor, int st_ib, int nb_local, int st_iao, int nao_local);
    MeanField(int ns, int nk, int nb, int nao, int st_ib, int nb_local, int st_iao, int nao_local)  // backward compability
        : MeanField(ns, nk, nb, nao, 1, st_ib, nb_local, st_iao, nao_local)
    {}
    ~MeanField() {};
    void set(int ns, int nk, int nb, int nao, int n_spinor = 1);
    void set(int ns, int nk, int nb, int nao, int n_spinor, int st_ib, int nb_local, int st_iao, int nao_local);
    void set(int ns, int nk, int nb, int nao, int st_ib, int nb_local, int st_iao, int nao_local)  // backward compability
    {
        set(ns, nk, nb, nao, 1, st_ib, nb_local, st_iao, nao_local);
    }
    MeanField(const MeanField&) = default;
    bool initialized() const { return n_spins > 0 && n_kpoints > 0 && n_states > 0 && n_aos > 0; }
    inline int get_n_bands() const { return n_states; }
    inline int get_n_states() const { return n_states; } // alias
    inline int get_n_spins() const { return n_spins; }
    inline int get_n_kpoints() const { return n_kpoints; }
    inline int get_n_aos() const { return n_aos; }
    inline int get_n_spinor() const { return n_spinor; }
    inline double& get_efermi() { return efermi; }
    inline const double& get_efermi() const { return efermi; }
    inline const FermiDiracReference& get_fermi_dirac_reference() const
    {
        return fermi_dirac_reference;
    }
    void set_fermi_dirac_reference(const FermiDiracReference& reference);
    void clear_fermi_dirac_reference();
    std::vector<matrix>& get_eigenvals() { return eskb; }
    const std::vector<matrix>& get_eigenvals() const { return eskb; }
    std::vector<matrix>& get_weight() { return wg; }
    const std::vector<matrix>& get_weight() const { return wg; }
    std::map<int, std::map<int, std::map<int, ComplexMatrix>>>& get_eigenvectors() { return wfc; }
    const std::map<int, std::map<int, std::map<int, ComplexMatrix>>>& get_eigenvectors() const { return wfc; }
    ComplexMatrix* find_wfc(int ispin, int ispinor, int ikpt) noexcept;
    const ComplexMatrix* find_wfc(int ispin, int ispinor, int ikpt) const noexcept;
    double get_E_min_max(double& emin, double& emax) const;
    double get_band_gap() const;
    //! Highest-energy occupied state for a spin. Use ikpt < 0 to search all k-points.
    //! occupation_tol is in unscaled occupation units; stored k-point weights are
    //! compared against occupation_tol / n_kpoints.
    //! Returns {-1, -1} when no occupied state is found.
    std::pair<int, int> find_highest_occupied_state(int ispin, int ikpt = -1,
                                                    double occupation_tol = 1.0e-8) const;
    //! Largest state index whose state and all lower-indexed states are below energy.
    //! Returns -1 if no state satisfies the condition.
    int get_max_state_below_energy(double energy) const;
    //! Smallest state index whose state and all higher-indexed states are above energy.
    //! Returns n_states if no state satisfies the condition.
    int get_min_state_above_energy(double energy) const;

    // Extract local k-point indices, used for MPI parallelization
    std::vector<int> get_iks_local() const;

    //! Get the density matrix of a particular spin and kpoint
    ComplexMatrix get_dmat_cplx(int ispin, int ispinor_bra, int ispinor_ket, int ikpt) const;

    // Density matrix and green's function calculation, serial version
    //! Positive spectral amplitude including the k-point weight but not the
    //! negative-time Green-function branch sign.
    double green_spectral_amplitude(int ispin, int ikpt, int iband,
                                    double tau, double kpoint_weight) const;
    ComplexMatrix get_dmat_cplx_R(int ispin, int ispinor_bra, int ispinor_ket,
                                  const std::vector<Vector3_Order<double>>& kfrac_list,
                                  const Vector3_Order<int>& R) const;
    std::map<Vector3_Order<int>, ComplexMatrix> get_dmat_cplx_Rs(
        int ispin, int ispinor_bra, int ispinor_ket,
        const std::vector<Vector3_Order<double>>& kfrac_list,
        const std::vector<Vector3_Order<int>>& Rs) const;

    ComplexMatrix get_gf_cplx_imagtime(int ispin, int ispinor_bra, int ispinor_ket, int ikpt,
                                       double tau) const;
    std::map<double, std::map<Vector3_Order<int>, ComplexMatrix>> get_gf_cplx_imagtimes_Rs(
        int ispin, int ispinor_bra, int ispinor_ket,
        const std::vector<Vector3_Order<double>>& kfrac_list, std::vector<double> imagtimes,
        const std::vector<Vector3_Order<int>>& Rs) const;
    std::map<double, std::map<Vector3_Order<int>, matrix>> get_gf_real_imagtimes_Rs(
        int ispin, int ispinor_bra, int ispinor_ket,
        const std::vector<Vector3_Order<double>>& kfrac_list, std::vector<double> imagtimes,
        const std::vector<Vector3_Order<int>>& Rs) const;

    // void allredue_wfc_isk();
};

bool can_restore_symmetry_kstar_meanfield(
    const SymmetryContext& ctx,
    const std::vector<SpeciesBasisLayout>& wfc_layouts,
    const MeanField& mf,
    const std::vector<Vector3_Order<double>>& kfrac_list,
    const std::map<atom_t, size_t>& atom_nw);

symmetry_kstar_member_kfrac_targets_t build_symmetry_kstar_member_kfrac_targets(
    const SymmetryContext& ctx,
    const PeriodicBoundaryData& pbc);

ComplexMatrix get_symmetry_restored_dmat_cplx_R(
    const SymmetryContext& ctx,
    const std::vector<SpeciesBasisLayout>& wfc_layouts,
    const MeanField& mf,
    int ispin, int ispinor_bra, int ispinor_ket,
    const std::vector<Vector3_Order<double>>& kfrac_list,
    const Vector3_Order<int>& R,
    const std::map<atom_t, size_t>& atom_nw,
    const symmetry_kstar_member_kfrac_targets_t* member_kfrac_targets = nullptr,
    const symmetry_kstar_representative_indices_t* representative_k_indices = nullptr);

std::map<double, std::map<Vector3_Order<int>, ComplexMatrix>>
get_symmetry_restored_gf_cplx_imagtimes_Rs(
    const SymmetryContext& ctx,
    const std::vector<SpeciesBasisLayout>& wfc_layouts,
    const MeanField& mf,
    int ispin, int ispinor_bra, int ispinor_ket,
    const std::vector<Vector3_Order<double>>& kfrac_list,
    const std::vector<double>& imagtimes,
    const std::vector<Vector3_Order<int>>& Rs,
    const std::map<atom_t, size_t>& atom_nw,
    int nbands_G = -1,
    const symmetry_kstar_member_kfrac_targets_t* member_kfrac_targets = nullptr,
    const symmetry_kstar_representative_indices_t* representative_k_indices = nullptr);

/*!
 * @brief Validate that a Green's-function band cutoff does not slice through a
 * degenerate band multiplet on the k-point set used as symmetry-restore source.
 *
 * A symmetry restore mixes all bands below the cutoff across each k-star; when
 * the cutoff index separates two (nearly) degenerate eigenvalues the truncated
 * band space is not closed under the star operations and the restored GF would
 * depend on the arbitrary gauge inside the degenerate subspace. Throws
 * LIBRPA_RUNTIME_ERROR naming the k-point, the band indices and the gap when
 * |E[nbands_G] - E[nbands_G - 1]| < degen_tol at any spin/k-point.
 *
 * No-op when nbands_G < 0 (no truncation), nbands_G == 0, or
 * nbands_G >= n_bands (truncation outside the band window).
 */
void validate_kstar_band_cutoff_closure(
    const SymmetryContext& ctx,
    const MeanField& mf,
    int nbands_G,
    double degen_tol = 1e-8);

/*!
 * @brief Four-channel spinor variant of get_symmetry_restored_gf_cplx_imagtimes_Rs.
 *
 * Returns, for each tau and R, the four spin blocks G^{ab} (a = bra, b = ket,
 * channel-outermost convention C8) restored from the IBZ representatives.
 * Each star member is resolved through
 * `member.action_id -> ctx.kspace_actions -> ctx.spin_operations[canonical]`
 * into (spatial_id, U_s, eta). The orbital transform matrix A is built once
 * per member (gauge phases excluded) and reused for all four source blocks;
 * the SU(2) mixing and, for antiunitary members, the Theta remap
 * {conj(Y11), -conj(Y10), -conj(Y01), conj(Y00)} are applied by
 * transform_spinor_bilinear. The target-kpoint gauge phases are plain unitary
 * re-gauging factors applied after the kernel, so antiunitary members match
 * the scalar restore convention. Transforms act in the tau domain point by
 * point (tau is real, z* = tau; report section 6.8).
 *
 * Requires mf.get_n_spinor() == 2. Missing (bra, ket) source blocks are
 * zero-filled. nbands_G >= 0 is guarded by validate_kstar_band_cutoff_closure.
 */
std::map<double, std::map<Vector3_Order<int>, SpinorBlocks4<ComplexMatrix>>>
get_symmetry_restored_gf_cplx_imagtimes_Rs_spinor(
    const SymmetryContext& ctx,
    const std::vector<SpeciesBasisLayout>& wfc_layouts,
    const MeanField& mf,
    int ispin,
    const std::vector<Vector3_Order<double>>& kfrac_list,
    const std::vector<double>& imagtimes,
    const std::vector<Vector3_Order<int>>& Rs,
    const std::map<atom_t, size_t>& atom_nw,
    int nbands_G = -1,
    const symmetry_kstar_member_kfrac_targets_t* member_kfrac_targets = nullptr,
    const symmetry_kstar_representative_indices_t* representative_k_indices = nullptr);

/*!
 * @brief Extract one (bra, ket) channel out of a four-channel spinor GF map,
 * moving the blocks out of the input.
 */
std::map<double, std::map<Vector3_Order<int>, ComplexMatrix>>
extract_spinor_gf_block(
    std::map<double, std::map<Vector3_Order<int>, SpinorBlocks4<ComplexMatrix>>>&& gf_spinor,
    int ispinor_bra, int ispinor_ket);

}

#endif // ! MEANFIELD_H
