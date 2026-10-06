#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "../../src/api/instance_manager.h"
#include "../../src/io/global_io.h"
#include "../../src/mpi/global_mpi.h"
#include "../../src/utils/constants.h"
#include "../reader/read_data.h"
#include "../reader/reader_context.h"

namespace
{
void write_sampling(const std::filesystem::path &path, const std::string &tail)
{
    std::ofstream out(path);
    out << std::setprecision(17);
    out << "1 1 3\n2 2\n";
    out << "1 " << 1.0 / 3 << " 0 0 0 0 0 0 1 1\n";
    out << "2 " << 2.0 / 3 << " 0 0 " << 1.0 / 3 << " 0 0 " << librpa_int::TWO_PI / 3 << " 2 2\n";
    out << tail;
}

std::string valid_tail()
{
    std::ostringstream out;
    out << std::setprecision(17);
    out << "full_kmap 3\n1 0 0 0 1\n2 0 0 " << librpa_int::TWO_PI / 3 << " 2\n3 0 0 "
        << 2 * librpa_int::TWO_PI / 3 << " 2\n";
    return out.str();
}

void check_mapping(const std::filesystem::path &file, const std::vector<int> &sizes)
{
    librpa::Handler h(MPI_COMM_SELF);
    const double lattice[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const double reciprocal[9] = {librpa_int::TWO_PI, 0, 0, 0, librpa_int::TWO_PI, 0, 0, 0,
                                  librpa_int::TWO_PI};
    h.set_latvec_and_G(lattice, reciprocal);
    const int n_scf = static_cast<int>(sizes.size());
    h.set_scf_dimension(1, n_scf, 1, 1, 1);
    librpa::reader::ReaderState state;
    state.n_kpoints = n_scf;
    librpa::reader::ReaderContext ctx{
        h, state, {}, {}, librpa_int::MpiCommHandler(MPI_COMM_SELF, true)};
    librpa::reader::read_bz_sampling(ctx, file.string());
    const auto ds = librpa_int::api::get_dataset_instance(h.get_c_handler());
    const auto &pbc = ds->pbc;
    assert(pbc.klist.size() == sizes.size());
    assert(pbc.klist_full.size() == 3);
    for (int ik = 0; ik < n_scf; ++ik)
    {
        assert(pbc.map_irk_ks.at(pbc.klist_coul[ik]).size() == static_cast<std::size_t>(sizes[ik]));
        assert(std::abs(pbc.weight_k[ik] - sizes[ik] / 3.0) < 1e-12);
        assert(std::abs(pbc.weight_q[ik] - sizes[ik] / 3.0) < 1e-12);
    }
}
}  // namespace

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    librpa_int::global::init_global_mpi(MPI_COMM_WORLD);
    librpa_int::global::init_global_io();
    const auto file = std::filesystem::temp_directory_path() / "librpa_bz_mapping.txt";
    write_sampling(file, valid_tail());
    check_mapping(file, {1, 2});
    for (const std::string tail :
         std::vector<std::string>{"full_kmap 2\n", "full_kmap 3\n1 0 0 0 3\n",
                                  "full_kmap 3\n1 0 0 0 1\n2 0 0 0 2\n3 0 0 0 2\n",
                                  "unexpected 3\n", valid_tail() + "extra\n"})
    {
        write_sampling(file, tail);
        bool rejected = false;
        try
        {
            check_mapping(file, {1, 2});
        }
        catch (const std::exception &)
        {
            rejected = true;
        }
        assert(rejected);
    }
    // Reject a complete, nonduplicated mapping that assigns the wrong star.
    std::string wrong_star = valid_tail();
    wrong_star.replace(wrong_star.find("1 0 0 0 1"), 9, "1 0 0 0 2");
    write_sampling(file, wrong_star);
    bool rejected = false;
    try
    {
        check_mapping(file, {1, 2});
    }
    catch (const std::exception &)
    {
        rejected = true;
    }
    assert(rejected);
    // Existing full-grid files require no extension and retain unit stars.
    {
        std::ofstream out(file);
        out << std::setprecision(17) << "1 1 3\n3 3\n";
        for (int ik = 0; ik < 3; ++ik)
            out << ik + 1 << ' ' << 1.0 / 3 << " 0 0 " << ik / 3.0 << " 0 0 "
                << ik * librpa_int::TWO_PI / 3 << ' ' << ik + 1 << ' ' << ik + 1 << '\n';
    }
    check_mapping(file, {1, 1, 1});
    {
        std::ofstream out(file, std::ios::app);
        out << "3\n1 0.3333333333333333\n2 0.3333333333333333\n3 0.3333333333333333\n";
    }
    check_mapping(file, {1, 1, 1});
    std::filesystem::remove(file);
    librpa_int::global::finalize_global_io();
    librpa_int::global::finalize_global_mpi();
    MPI_Finalize();
}
