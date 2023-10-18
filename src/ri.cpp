#include "ri.h"
#include <memory.h>
#include "mpi.h"
#include "parallel_mpi.h"
int n_irk_points;
int natom;
int ncell;
map<Vector3_Order<double>, double> irk_weight;
map<atom_t, size_t> atom_nw;
map<atom_t, size_t> atom_mu;
map<atom_t, size_t> atom_nw_loc;
map<atom_t, size_t> atom_mu_loc;
vector<size_t> atom_mu_part_range;
int N_all_mu;

vector<atpair_t> tot_atpair;
vector<atpair_t> local_atpair;
vector<atpair_t> tot_atpair_ordered;

atpair_R_mat_t Cs;
atpair_k_cplx_mat_t Vq;
atpair_k_cplx_mat_t Vq_cut;
map<Vector3_Order<double>, ComplexMatrix> Vq_block_loc;
using LIBRPA::mpi_comm_world_h;
int atom_iw_loc2glo(const int &atom_index, const int &iw_lcoal)
{
    int nb = 0;
    for (int ia = 0; ia != atom_index; ia++)
        nb += atom_nw[ia];
    return iw_lcoal + nb;
}

int atom_mu_loc2glo(const int &atom_index, const int &mu_lcoal)
{
    // int nb = 0;
    // for (int ia = 0; ia != atom_index; ia++)
    //     nb += atom_mu[ia];
    // return mu_lcoal + nb;
    return atom_mu_part_range[atom_index]+mu_lcoal;
}

int atom_mu_glo2loc(const int &glo_index, int &mu_index)
{
    for(int I=0;I!=atom_mu.size();I++)
    {
        if((mu_index=glo_index-atom_mu_part_range[I])<atom_mu[I])
            return I;
    }
    throw invalid_argument("invalid glo_index");
}

vector<int> get_part_range()
{
    vector<int> part_range;
    part_range.resize(atom_mu.size());
    part_range[0] = 0;

    int count_range = 0;
    for (int Mu = 0; Mu != atom_mu.size() - 1; Mu++)
    {
        count_range += atom_mu[Mu];
        part_range[Mu + 1] = count_range;
    }
    return part_range;
}

matrix reshape_Cs(size_t n1, size_t n2, size_t n3, const shared_ptr<matrix> &Csmat) //(n1*n2,n3) -> (n2,n1*n3)
{
    const auto length = sizeof(double) * n3;
    const auto n13 = n1 * n3;
    const double *m_ptr = (*Csmat).c;
    matrix m_new(n2, n1 * n3, false);
    for (size_t i1 = 0; i1 != n1; ++i1)
    {
        double *m_new_ptr = m_new.c + i1 * n3;
        for (size_t i2 = 0; i2 != n2; ++i2, m_ptr += n3, m_new_ptr += n13)
            memcpy(m_new_ptr, m_ptr, length);
    }
    return m_new;
}

matrix reshape_mat(const size_t n1, const size_t n2, const size_t n3, const matrix &mat) //(n1,n2*n3) -> (n2,n1*n3)
{
    const auto length = sizeof(double) * n3;
    const auto n13 = n1 * n3;
    const double *m_ptr = mat.c;
    matrix m_new(n2, n1 * n3, false);
    for (size_t i1 = 0; i1 != n1; ++i1)
    {
        double *m_new_ptr = m_new.c + i1 * n3;
        for (size_t i2 = 0; i2 != n2; ++i2, m_ptr += n3, m_new_ptr += n13)
            memcpy(m_new_ptr, m_ptr, length);
    }
    return m_new;
}

matrix reshape_mat_21(const size_t n1, const size_t n2, const size_t n3, const matrix &mat) //(n1,n2*n3) -> (n1*n2,n3)
{
    const auto length = sizeof(double) * n1 * n2 * n3;
    const auto n13 = n1 * n2 * n3;
    const double *m_ptr = mat.c;
    matrix m_new(n1 * n2, n3, false);
    double *m_new_ptr = m_new.c;
    memcpy(m_new_ptr, m_ptr, length);
    return m_new;
}

void init_N_all_mu()
{
    // printf("begin init_N_all_mu   atom_mu.size: %d  myid: %d\n",atom_mu.size(),mpi_comm_world_h.myid);
    // mpi_comm_world_h.barrier();
    // vector<size_t> loc_mu(natom);
    // vector<size_t> loc_nw(natom);
    // for(const auto &nw_p:atom_mu_loc)
    // {
    //     loc_nw[nw_p.first]=nw_p.second;
    // }
    // for(const auto &mu_p:atom_mu_loc)
    // {
    //     loc_mu[mu_p.first]=mu_p.second;
    // }
    // printf(" mid init_N_all_mu\n");
    // vector<size_t> glo_nw(natom);
    // vector<size_t> glo_mu(natom);
    // printf(" mid init_N_all_mu myid: %d\n",mpi_comm_world_h.myid);
    // MPI_Allreduce(loc_nw.data(),glo_nw.data(),natom,MPI_UNSIGNED_LONG_LONG,MPI_MAX,MPI_COMM_WORLD);
    // MPI_Allreduce(loc_mu.data(),glo_mu.data(),natom,MPI_UNSIGNED_LONG_LONG,MPI_MAX,MPI_COMM_WORLD);

    // atom_nw.clear();
    // atom_mu.clear();
    // for(int i =0; i!=natom;i++ )
    // {
    //     atom_nw.insert(pair<atom_t,size_t>(i,glo_nw[i]));
    //     atom_mu.insert(pair<atom_t,size_t>(i,glo_mu[i]));
    // }
    printf("Begin init_N_all_mu, atom_mu.size: %d\n",atom_mu.size());
    atom_mu_part_range.resize(atom_mu.size());
    atom_mu_part_range[0]=0;
    for(int I=1;I!=atom_mu.size();I++)
        atom_mu_part_range[I]=atom_mu.at(I-1)+atom_mu_part_range[I-1];
    
    N_all_mu=atom_mu_part_range[natom-1]+atom_mu[natom-1];
    printf("end init_N_all_mu, atom_mu.size: %d\n",atom_mu.size());
  //  MPI_Barrier(MPI_COMM_WORLD);
}

void allreduce_2D_coulomb_to_atompair(map<Vector3_Order<double>, ComplexMatrix> &Vq_loc, atpair_k_cplx_mat_t &coulomb_mat,double threshold )
{
    printf("Begin allreduce_2D_coulomb_to_atompair!\n");
    size_t vq_save = 0;
    size_t vq_discard = 0;
    for (auto &vf_p : Vq_loc)
    {
        auto qvec = vf_p.first;
        // cout << "Qvec:" << qvec << endl;
        
        for (int I = 0; I != atom_mu.size(); I++)
            for (int J = 0; J != atom_mu.size(); J++)
            {
                if (I > J)
                    continue;
                shared_ptr<ComplexMatrix> vq_ptr = make_shared<ComplexMatrix>();
                vq_ptr->create(atom_mu[I], atom_mu[J]);
                // vq_ptr_tran->create(atom_mu[J],atom_mu[I]);
                // cout << "I J: " << I << "  " << J << "   mu,nu: " << atom_mu[I] << "  " << atom_mu[J] << endl;
                for (int i_mu = 0; i_mu != atom_mu[I]; i_mu++)
                {

                    for (int i_nu = 0; i_nu != atom_mu[J]; i_nu++)
                    {
                        //(*vq_ptr)(i_mu, i_nu) = vf_p.second(atom_mu_loc2glo(J, i_nu), atom_mu_loc2glo(I, i_mu)); ////for aims
                        (*vq_ptr)(i_mu, i_nu) = vf_p.second(atom_mu_loc2glo(I, i_mu), atom_mu_loc2glo(J, i_nu)); // for abacus
                    }
                }

                // if (I == J)
                // {
                //     (*vq_ptr).set_as_identity_matrix();
                // }

                if ((*vq_ptr).real().absmax() >= threshold)
                {
                    coulomb_mat[I][J][qvec] = vq_ptr;
                    vq_save++;
                }
                else
                {
                    vq_discard++;
                }
            }
    }
    printf("End allreduce_2D_coulomb_to_atompair!\n");
}