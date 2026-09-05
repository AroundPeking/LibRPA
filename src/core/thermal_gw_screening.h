#ifndef LIBRPA_THERMAL_GW_SCREENING_H
#define LIBRPA_THERMAL_GW_SCREENING_H

#include "thermal_gw_transform.h"

namespace librpa_int
{
/** Internal full-matrix reference result; no production thermal GW route.
 * Rows retain the explicitly supplied signed bosonic label order; column
 * r*N+c is the ordered matrix element Wc(r,c), not a Hermitian triangle.
 * Owns its data. The adapter revalidates this mutable value type before use.
 */
struct ThermalGWScreening
{
    double beta_ha_inv;
    std::vector<int> bosonic_indices;
    ComplexMatrix wc_frequency;
};

/** Compute epsilon = I-S*chi*S and Wc = T*(epsilon^-1-I)*T by general LU.
 * Solve epsilon*Delta = (S*chi*S)*T, then Wc=T*Delta. Since epsilon*T =
 * T-(S*chi*S)*T, Delta is exactly X-T for epsilon*X=T, without subtracting
 * nearly equal matrices. This preserves sub-epsilon weak-screening tails.
 * All samples are complete square complex matrices of the same order as S,T.
 * S,T must be finite Hermitian roots; their origin/positivity is the caller's
 * contract. No root inversion, regularization, symmetrization, missing-mode
 * completion, tail correction, spatial weight, or beta factor is applied.
 * Labels are unique signed integers; completeness is relative to the supplied
 * list, not a claim of complete Matsubara coverage or of physical validity.
 *
 * Uses general pivoted LU with explicit column-major RAII copies. Extra identity
 * RHS columns estimate rcond_inf = 1/(||epsilon||_inf*||epsilon^-1||_inf), at the
 * cost of N additional RHS solves, for this bounded dense reference only.
 * Rejects rcond_inf <= 64*N*machine_epsilon and scale-aware normwise backward
 * residuals > 64*N*machine_epsilon for both physical and identity RHS blocks.
 * A passing backward-error/condition check does not certify forward accuracy
 * of Wc or remove uncertainty in the supplied chi. Nonfinite products/outputs
 * and unrepresentable norms fail closed; there is no equilibration or refinement.
 *
 * Each of the four matrix products conservatively rejects subnormal component
 * multiplication risk before GEMM: at each inner index, take the smallest
 * nonzero absolute real/imaginary component a in the left column and b in the
 * right row, and reject a < min_normal_double/b. Zero components are skipped.
 * This O(N^2) restriction can reject products whose terms would later cancel;
 * it does not promise to detect all underflow, including summation cancellation
 * below the normal threshold or underflow within LU/transform operations.
 * Extreme-range support is not provided by these checks.
 *
 * Invalid metadata/storage/root inputs throw invalid_argument; nonfinite
 * arithmetic throws overflow_error; component product risk throws underflow_error;
 * LU INFO or numerical checks throw
 * runtime_error with the offending mode. Inputs are never modified.
 */
ThermalGWScreening screen_thermal_gw(double beta_ha_inv, const std::vector<int>& bosonic_indices,
                                     const std::vector<ComplexMatrix>& chi, const ComplexMatrix& s,
                                     const ComplexMatrix& t);

/** Apply the existing integral-normalized B verbatim to the packed samples.
 * Requires exact beta and ordered integer-label equality, including signs.
 * Output is N_tau by N*N; never reorder, conjugate, project to real or add
 * weights. A quadrature B remains a finite reference sum, not a sparse fit.
 */
ComplexMatrix thermal_gw_screening_to_time(const ThermalGWScreening& screening,
                                           const ThermalGWTransform& transform);
}  // namespace librpa_int

#endif
