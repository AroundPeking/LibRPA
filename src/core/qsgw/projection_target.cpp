#include "projection_target.h"

#include <cmath>
#include <stdexcept>

#include "../meanfield_mpi.h"

namespace librpa_int
{
namespace qsgw
{

void validate_projection_target(const MeanField& reference,
                                const std::vector<Vector3_Order<double>>& kpoints,
                                const int expected_n_spins, const int expected_n_spinors,
                                const int expected_n_aos, const std::string& label)
{
    if (!reference.initialized())
    {
        throw std::invalid_argument(
            "QSGW " + label + " projection mean field is not initialized");
    }
    if (reference.get_n_spins() != expected_n_spins ||
        reference.get_n_spinor() != expected_n_spinors ||
        reference.get_n_aos() != expected_n_aos)
    {
        throw std::invalid_argument(
            "QSGW " + label +
            " projection target is incompatible with the source basis");
    }
    if (reference.get_n_kpoints() != static_cast<int>(kpoints.size()))
    {
        throw std::invalid_argument(
            "QSGW " + label +
            " projection k-point coordinates do not match the mean field");
    }

    for (const auto& kpoint : kpoints)
    {
        if (!std::isfinite(kpoint.x) || !std::isfinite(kpoint.y) ||
            !std::isfinite(kpoint.z))
        {
            throw std::invalid_argument(
                "QSGW " + label +
                " projection contains a non-finite k-point");
        }
    }

    for (int spin = 0; spin < reference.get_n_spins(); ++spin)
    {
        for (int spinor = 0; spinor < reference.get_n_spinor(); ++spinor)
        {
            for (int kpoint = 0; kpoint < reference.get_n_kpoints(); ++kpoint)
            {
                const ComplexMatrix* wavefunction =
                    reference.find_wfc(spin, spinor, kpoint);
                if (wavefunction == nullptr ||
                    wavefunction->nr != reference.get_n_bands() ||
                    wavefunction->nc != reference.get_n_aos())
                {
                    throw std::invalid_argument(
                        "QSGW " + label +
                        " projection wavefunction map is incomplete or malformed");
                }
                for (int index = 0; index < wavefunction->size; ++index)
                {
                    const cplxdb value = wavefunction->c[index];
                    if (!std::isfinite(value.real()) ||
                        !std::isfinite(value.imag()))
                    {
                        throw std::invalid_argument(
                            "QSGW " + label +
                            " projection wavefunction contains non-finite data");
                    }
                }
            }
        }
    }
}

void validate_live_symmetry_state(const MeanField& meanfield, const AtomicBasis& basis,
                                  const PeriodicBoundaryData& pbc,
                                  const SymmetryContext& symmetry, const bool expanded_from_ibz)
{
    validate_projection_target(meanfield, pbc.kfrac_list, meanfield.get_n_spins(),
                               meanfield.get_n_spinor(), meanfield.get_n_aos(), "live symmetry");

    const auto& energies = meanfield.get_eigenvals();
    const auto& occupations = meanfield.get_weight();
    const int spins = meanfield.get_n_spins();
    const int kpoints = meanfield.get_n_kpoints();
    const int bands = meanfield.get_n_bands();
    if (energies.size() != static_cast<std::size_t>(spins) ||
        occupations.size() != static_cast<std::size_t>(spins))
        throw std::invalid_argument("QSGW live symmetry E/f storage is incomplete");
    for (int spin = 0; spin < spins; ++spin)
    {
        if (energies[spin].nr != kpoints || energies[spin].nc != bands ||
            occupations[spin].nr != kpoints || occupations[spin].nc != bands)
            throw std::invalid_argument("QSGW live symmetry E/f storage is malformed");
        for (int kpoint = 0; kpoint < kpoints; ++kpoint)
            for (int band = 0; band < bands; ++band)
                if (!std::isfinite(energies[spin](kpoint, band)) ||
                    !std::isfinite(occupations[spin](kpoint, band)))
                    throw std::invalid_argument("QSGW live symmetry contains non-finite E/f");
    }
    validate_spin_operations_for_storage(symmetry, spins, meanfield.get_n_spinor());

    // Expansion replaces the loaded mesh by the full one, so its source
    // provenance must be retained independently of the current grid sizes.
    const bool legacy_scalar_full_grid =
        spins == 1 && meanfield.get_n_spinor() == 1 && bands == meanfield.get_n_aos() &&
        !symmetry.has_explicit_spin_operations && !expanded_from_ibz &&
        symmetry.available && basis.has_l_shells() && kpoints == pbc.get_n_cells_bvk() &&
        pbc.kfrac_list_full.size() == static_cast<std::size_t>(kpoints);
    if (!legacy_scalar_full_grid)
        validate_symmetry_band_weights(meanfield, basis, pbc, symmetry);
}

} // namespace qsgw
} // namespace librpa_int
