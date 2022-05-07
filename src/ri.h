/*!
 @file ri.h
 @brief Utilies related to resolution of identity
 */
#ifndef RI_H
#define RI_H

#include <map>
#include <memory>
#include <utility>
#include "vector3_order.h"
#include "atoms.h"
#include "complexmatrix.h"
using std::map;

extern int n_irk_points;
extern int natom;
extern int ncell;
extern map<Vector3_Order<double>, double> irk_weight;
extern map<atom_t, size_t> atom_nw;
extern map<atom_t, size_t> atom_mu;

//! type alias of atom-pair mapping to real matrix indexed by unit-cell vector
typedef atom_mapping< map<Vector3_Order<int>, std::shared_ptr<matrix>> >::pair_t_old atpair_R_mat_t;
//! type alias of atom-pair mapping to complex matrix indexed by reciprocal vector
typedef atom_mapping< map<Vector3_Order<double>, std::shared_ptr<ComplexMatrix>> >::pair_t_old atpair_k_cplx_mat_t;

//! Tri-coefficient of localized RI (LRI) in real space. ABF on the first atom of the atom pair
extern atpair_R_mat_t Cs;
//! Coulomb matrix in ABF, represented in the reciprocal space.
extern atpair_k_cplx_mat_t Vq;

int atom_iw_loc2glo(const int &atom_index, const int &iw_lcoal);
int atom_mu_loc2glo(const int &atom_index, const int &mu_lcoal);

#endif // !RI_H

