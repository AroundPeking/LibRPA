#ifndef LIBRPA_THERMAL_GW_TRANSFORM_H
#define LIBRPA_THERMAL_GW_TRANSFORM_H

#include <vector>

#include "../math/complexmatrix.h"

namespace librpa_int
{
/** Generate a validated Matsubara grid once; copies retain the stored values. */
class ThermalFrequencyGrid
{
public:
    ThermalFrequencyGrid(double beta_ha_inv, const std::vector<int> &indices, bool fermionic);
    double get_beta_ha_inv() const { return beta_ha_inv_; }
    bool is_fermionic() const { return fermionic_; }
    const std::vector<int> &get_indices() const { return indices_; }
    const std::vector<double> &get_frequencies_ha() const { return frequencies_ha_; }

private:
    double beta_ha_inv_;
    bool fermionic_;
    std::vector<int> indices_;
    std::vector<double> frequencies_ha_;
};

/** Owned, integral-normalized finite-temperature GW transforms (internal only).
 *
 * B(j,m) maps Wc(i*nu_m) to Wc(tau_j), with nu_m = 2*pi*m/beta.
 * F(n,j) maps Sigma(tau_j) to Sigma(i*omega_n), omega_n = (2*n+1)*pi/beta.
 * Supplied coefficients include ALL integration/normalization factors. The
 * reference builder uses B = exp(-i*nu*tau)/beta and F = w_j*exp(+i*omega*tau).
 * It does not invert a sparse sampling matrix or estimate omitted-mode tails.
 *
 * Times increase strictly inside (0,beta); signed, unique integer mode labels
 * retain caller order. No missing negative modes, conjugation, real projection,
 * half weights at m=0, or evenness assumptions are introduced. Input samples
 * have source-grid rows and independent (e.g. flattened AO/ABF) columns.
 * ComplexMatrix storage is row-major. Empty sample batches are rejected.
 *
 * The supplied matrices' accuracy and normalization are a caller contract,
 * not certified by shape/finite checks. Transform Sigma = G_lib*Wc directly:
 * current LibRPA G_lib = -G_standard already supplies the GW minus sign.
 * Use correlation Wc, not the instantaneous bare interaction.
 */
class ThermalGWTransform
{
public:
    ThermalGWTransform(double beta_ha_inv, const std::vector<double> &times,
                       const std::vector<int> &bosonic_indices,
                       const std::vector<int> &fermionic_indices,
                       const ComplexMatrix &bosonic_frequency_to_time,
                       const ComplexMatrix &fermionic_time_to_frequency);

    static ThermalGWTransform from_quadrature(double beta_ha_inv, const std::vector<double> &times,
                                              const std::vector<double> &time_weights,
                                              const std::vector<int> &bosonic_indices,
                                              const std::vector<int> &fermionic_indices);

    double get_beta_ha_inv() const { return fermionic_grid_.get_beta_ha_inv(); }
    const std::vector<double> &get_times() const { return times_; }
    const ThermalFrequencyGrid &get_bosonic_grid() const { return bosonic_grid_; }
    const ThermalFrequencyGrid &get_fermionic_grid() const { return fermionic_grid_; }
    const std::vector<int> &get_bosonic_indices() const { return bosonic_grid_.get_indices(); }
    const std::vector<int> &get_fermionic_indices() const { return fermionic_grid_.get_indices(); }
    const std::vector<double> &get_bosonic_frequencies_ha() const
    {
        return bosonic_grid_.get_frequencies_ha();
    }
    const std::vector<double> &get_fermionic_frequencies_ha() const
    {
        return fermionic_grid_.get_frequencies_ha();
    }

    // Return deep copies: even a const ComplexMatrix exposes a writable c pointer.
    ComplexMatrix copy_bosonic_frequency_to_time() const;
    ComplexMatrix copy_fermionic_time_to_frequency() const;
    ComplexMatrix apply_bosonic_frequency_to_time(const ComplexMatrix &samples) const;
    ComplexMatrix apply_fermionic_time_to_frequency(const ComplexMatrix &samples) const;

private:
    std::vector<double> times_;
    ThermalFrequencyGrid bosonic_grid_;
    ThermalFrequencyGrid fermionic_grid_;
    ComplexMatrix bosonic_frequency_to_time_;
    ComplexMatrix fermionic_time_to_frequency_;
};
}  // namespace librpa_int

#endif
