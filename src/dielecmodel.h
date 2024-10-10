#pragma once
#include <functional>
#include <vector>

#include "atomic_basis.h"
#include "complexmatrix.h"
#include "constants.h"
#include "envs_io.h"
#include "envs_mpi.h"
#include "libri_stub.h"
#include "matrix_m_parallel_utils.h"
#include "meanfield.h"
#include "parallel_mpi.h"
#include "params.h"
#include "pbc.h"
#include "ri.h"
#include "vec.h"

//! double-dispersion Havriliak-Negami model
struct DoubleHavriliakNegami
{
    static const int d_npar;
    static const std::function<double(double, const std::vector<double> &)> func_imfreq;
    static const std::function<void(std::vector<double> &, double, const std::vector<double> &)>
        grad_imfreq;
};

class diele_func
{
   private:
    // ( alpha, beta, omega )
    std::vector<std::vector<std::vector<std::complex<double>>>> head;
    // ( alpha, lambda:n_abfs-n_singular-1, omega )
    std::vector<std::vector<std::vector<std::complex<double>>>> wing;
    // ( lambda: n_abfs-n_singular-1, mu: n_abfs)
    std::vector<std::vector<std::complex<double>>> Coul_vector;
    // ( lambda: n_abfs-n_singular-1 )
    std::vector<std::complex<double>> Coul_value;
    // ( mu: n_abfs, m: n_bands, n: n_bands, k )
    std::vector<std::vector<std::vector<std::map<Vector3_Order<double>, std::complex<double>>>>>
        Ctri_mn;
    // ( mu: n_abfs, i: i atom basis, j: j atom basis, k, I atom, J atom, R cell  )
    // Ctri_ij.data_libri[I][{J, k_array}](mu, i, j)
    Cs_LRI_clx Ctri_ij;

    const MeanField &meanfield_df;
    const std::vector<double> &omega;
    const std::vector<Vector3_Order<double>> &kfrac_band;
    const int n_basis, n_states, n_spin;
    size_t n_nonsingular;

   public:
    diele_func(const MeanField &mf, const std::vector<Vector3_Order<double>> &kfrac,
               const std::vector<double> &frequencies_target, const int nbasis, const int nstates,
               const int nspin)
        : meanfield_df(mf),
          kfrac_band(kfrac),
          omega(frequencies_target),
          n_basis(nbasis),
          n_states(nstates),
          n_spin(nspin)
    {
        init_headwing();
    };
    ~diele_func() {};
    void init_headwing();
    void init_Cs();
    // All calculation in unit: Ang and eV.
    void cal_head();
    double cal_factor(string name);
    void test_head();

    void cal_wing();  // atpair_k_cplx_mat_t &Vq_cut, Cs_LRI &Cs_data
    // tranform Cs_ij(R) to Cs_ij(k)
    void FT_R2k();
    void Cs_ij2mn();
    void get_Xv();  // diagonalize Vq_cut(q=0)
    void test_wing();
};
